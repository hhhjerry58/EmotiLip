"""
EmotiLip: Full model that ties all modules together.

Pipeline:
  Video (lip crop) → Visual Encoder → content features
  Video (face crop) → Emotion Encoder → emotion embedding + V-A
  [content + emotion + speaker] → Prosody Predictor → pitch + energy
  [noisy mel, timestep, content, emotion, prosody] → MelGen Denoiser → predicted noise
  Generated mel → HiFi-GAN Vocoder → waveform
"""

import torch
import torch.nn as nn
from dataclasses import dataclass

from losses import EmotionConsistencyLoss

from .visual_encoder import VisualEncoder
from .emotion_encoder import EmotionEncoder, EmotionOutput
from .prosody_predictor import ProsodyPredictor
from .melgen import MelGen
from .vocoder import HiFiGANVocoder


@dataclass
class EmotiLipOutput:
    """Container for all model outputs during training."""
    diffusion_loss: torch.Tensor
    prosody_loss: torch.Tensor | None
    emotion_consistency_loss: torch.Tensor | None
    emotion_classifier_loss: torch.Tensor | None
    pitch_pred: torch.Tensor | None
    energy_pred: torch.Tensor | None
    emotion_logits: torch.Tensor | None
    emotion_output: EmotionOutput | None
    total_loss: torch.Tensor


class MelEmotionClassifier(nn.Module):
    """Small mel-level emotion classifier used as a train-time proxy critic."""

    def __init__(self, mel_dim: int, hidden_dim: int = 128, num_classes: int = 8, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(mel_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, mel: torch.Tensor, mel_mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self.net(mel)
        if mel_mask is None:
            pooled = features.mean(dim=-1)
        else:
            mask = mel_mask.to(device=features.device, dtype=features.dtype).unsqueeze(1)
            pooled = (features * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
        return self.head(pooled)


class EmotiLip(nn.Module):
    """
    Full Emotion-Aware Lip-to-Speech model.

    Args:
        visual_cfg: Config dict for VisualEncoder.
        emotion_cfg: Config dict for EmotionEncoder.
        prosody_cfg: Config dict for ProsodyPredictor.
        melgen_cfg: Config dict for MelGen.
        vocoder_cfg: Config dict for HiFiGANVocoder.
        speaker_dim: Speaker embedding dimension (0 to disable speaker conditioning).
        num_speakers: Number of speakers for speaker embedding table.
        lambda_prosody: Weight for prosody prediction loss.
        load_vocoder: Whether to instantiate the vocoder immediately.
        use_emotion: Whether to condition MelGen/prosody on face emotion.
        use_prosody: Whether to use the prosody predictor module.
    """

    def __init__(
        self,
        visual_cfg: dict | None = None,
        emotion_cfg: dict | None = None,
        prosody_cfg: dict | None = None,
        melgen_cfg: dict | None = None,
        vocoder_cfg: dict | None = None,
        speaker_dim: int = 256,
        num_speakers: int = 60,
        lambda_prosody: float = 1.0,
        lambda_emotion_consistency: float = 0.0,
        emotion_consistency_cfg: dict | None = None,
        load_vocoder: bool = False,
        use_emotion: bool = True,
        use_prosody: bool = True,
        use_emotion_consistency: bool = False,
    ):
        super().__init__()
        visual_cfg = visual_cfg or {}
        emotion_cfg = emotion_cfg or {}
        prosody_cfg = prosody_cfg or {}
        melgen_cfg = melgen_cfg or {}
        vocoder_cfg = vocoder_cfg or {}

        self.use_emotion = use_emotion
        self.use_prosody = use_prosody
        self.use_emotion_consistency = use_emotion_consistency
        self.lambda_prosody = lambda_prosody
        self.lambda_emotion_consistency = lambda_emotion_consistency
        self.speaker_dim = speaker_dim
        self.emotion_dim = melgen_cfg.get("emotion_dim", emotion_cfg.get("embed_dim", 256))
        self.emotion_cfg = emotion_cfg
        self.vocoder_cfg = vocoder_cfg
        emotion_consistency_cfg = emotion_consistency_cfg or {}

        # --- Sub-modules ---
        self.visual_encoder = VisualEncoder(**visual_cfg)
        self.emotion_encoder = EmotionEncoder(**emotion_cfg) if use_emotion else None
        self.melgen = MelGen(**melgen_cfg)
        self.emotion_consistency_loss = None
        self.mel_emotion_classifier = None
        if use_emotion_consistency and lambda_emotion_consistency > 0:
            num_classes = emotion_consistency_cfg.get("num_emotion_classes", emotion_cfg.get("n_expression", 8))
            self.emotion_consistency_loss = EmotionConsistencyLoss(
                lambda_cls=1.0,
                lambda_va=0.0,
                lambda_emb=0.0,
                num_emotion_classes=num_classes,
            )
            self.mel_emotion_classifier = MelEmotionClassifier(
                mel_dim=melgen_cfg.get("mel_dim", 80),
                hidden_dim=emotion_consistency_cfg.get("hidden_dim", 128),
                num_classes=num_classes,
                dropout=emotion_consistency_cfg.get("dropout", 0.1),
            )

        # Vocoder is frozen and only needed for inference/evaluation. Keep it
        # lazy by default so training does not waste GPU memory or checkpoint space.
        self.vocoder = HiFiGANVocoder(**vocoder_cfg) if load_vocoder else None

        # Speaker embedding
        if speaker_dim > 0 and num_speakers > 0:
            self.speaker_embedding = nn.Embedding(num_speakers, speaker_dim)
        else:
            self.speaker_embedding = None

        # Prosody predictor
        if use_prosody:
            prosody_defaults = {
                "content_dim": visual_cfg.get("output_dim", 512),
                "emotion_dim": self.emotion_dim,
                "speaker_dim": speaker_dim,
            }
            prosody_defaults.update(prosody_cfg)
            self.prosody_predictor = ProsodyPredictor(**prosody_defaults)
        else:
            self.prosody_predictor = None

    def forward(
        self,
        lip_video: torch.Tensor,
        face_crop: torch.Tensor,
        mel_target: torch.Tensor,
        speaker_id: torch.Tensor | None = None,
        pitch_target: torch.Tensor | None = None,
        energy_target: torch.Tensor | None = None,
        mel_mask: torch.Tensor | None = None,
        emotion_label: torch.Tensor | None = None,
    ) -> EmotiLipOutput:
        """
        Training forward pass.

        Args:
            lip_video: (B, 1, T_vid, H, W) — grayscale lip-cropped video.
            face_crop: (B, 3, H_face, W_face) — RGB face crop for emotion.
            mel_target: (B, mel_dim, T_mel) — ground truth mel spectrogram.
            speaker_id: (B,) — speaker index (optional).
            pitch_target: (B, T_mel, 1) — ground truth log-F0 (optional).
            energy_target: (B, T_mel, 1) — ground truth log-energy (optional).
            mel_mask: (B, T_mel) — valid-frame mask for padded batches.

        Returns:
            EmotiLipOutput with losses and predictions.
        """
        # 1. Extract visual features
        visual_feat = self.visual_encoder(lip_video)  # (B, T_vid, D)

        # 2. Extract emotion embedding (frozen), or use a zero vector for baseline.
        emotion_out = None
        if self.use_emotion:
            if self.emotion_encoder is None:
                self.emotion_encoder = EmotionEncoder(**self.emotion_cfg).to(lip_video.device)
            emotion_out = self.emotion_encoder(face_crop)  # EmotionOutput
            emotion_emb = emotion_out.embedding            # (B, emotion_dim)
        else:
            emotion_emb = lip_video.new_zeros(lip_video.shape[0], self.emotion_dim)

        # 3. Speaker embedding
        speaker_emb = None
        if self.speaker_embedding is not None and speaker_id is not None:
            speaker_emb = self.speaker_embedding(speaker_id)  # (B, speaker_dim)

        # 4. Align temporal dimensions (visual T_vid → mel T_mel)
        T_mel = mel_target.shape[-1]
        visual_feat = self._align_temporal(visual_feat, T_mel)  # (B, T_mel, D)

        # 5. Prosody prediction
        prosody = None
        prosody_loss = None
        pitch_pred = None
        energy_pred = None
        emotion_consistency_loss = None
        emotion_classifier_loss = None
        emotion_logits = None

        if self.use_prosody and self.prosody_predictor is not None:
            pitch_pred, energy_pred = self.prosody_predictor(
                visual_feat, emotion_emb, speaker_emb
            )
            prosody = torch.cat([pitch_pred, energy_pred], dim=-1)  # (B, T, 2)

            if pitch_target is not None and energy_target is not None:
                prosody_loss = self._masked_mse(pitch_pred, pitch_target, mel_mask)
                prosody_loss = prosody_loss + self._masked_mse(energy_pred, energy_target, mel_mask)

        # 6. Diffusion loss
        needs_clean_mel = self.mel_emotion_classifier is not None and emotion_label is not None
        diffusion_result = self.melgen.compute_loss(
            mel_target, visual_feat, emotion_emb, prosody, mel_mask=mel_mask, return_pred=needs_clean_mel
        )
        if needs_clean_mel:
            diffusion_loss, predicted_clean_mel = diffusion_result
        else:
            diffusion_loss = diffusion_result
            predicted_clean_mel = None

        # 7. Total loss
        total_loss = diffusion_loss
        if prosody_loss is not None:
            total_loss = total_loss + self.lambda_prosody * prosody_loss

        if needs_clean_mel and self.emotion_consistency_loss is not None:
            target_logits = self.mel_emotion_classifier(mel_target.detach(), mel_mask)
            emotion_classifier_loss = nn.functional.cross_entropy(target_logits, emotion_label)
            emotion_logits = self.mel_emotion_classifier(predicted_clean_mel, mel_mask)
            losses = self.emotion_consistency_loss(
                audio_emotion_logits=emotion_logits,
                target_emotion_label=emotion_label,
            )
            emotion_consistency_loss = losses["total"]
            total_loss = total_loss + self.lambda_emotion_consistency * (
                emotion_classifier_loss + emotion_consistency_loss
            )

        return EmotiLipOutput(
            diffusion_loss=diffusion_loss,
            prosody_loss=prosody_loss,
            emotion_consistency_loss=emotion_consistency_loss,
            emotion_classifier_loss=emotion_classifier_loss,
            pitch_pred=pitch_pred,
            energy_pred=energy_pred,
            emotion_logits=emotion_logits,
            emotion_output=emotion_out,
            total_loss=total_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        lip_video: torch.Tensor,
        face_crop: torch.Tensor,
        speaker_id: torch.Tensor | None = None,
        mel_length: int | None = None,
        guidance_scale: float = 2.0,
        emotion_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inference: generate waveform from video.

        Args:
            lip_video: (B, 1, T_vid, H, W)
            face_crop: (B, 3, H, W)
            speaker_id: (B,) optional
            mel_length: Override mel length.
            guidance_scale: Classifier-free guidance scale.
            emotion_override: (B, emotion_dim) — override emotion embedding
                              (for emotion interpolation demos).

        Returns:
            audio: (B, T_audio) — generated waveform.
            mel: (B, mel_dim, T_mel) — generated mel spectrogram.
        """
        visual_feat = self.visual_encoder(lip_video)

        if emotion_override is not None:
            emotion_emb = emotion_override
        elif self.use_emotion:
            if self.emotion_encoder is None:
                self.emotion_encoder = EmotionEncoder(**self.emotion_cfg).to(lip_video.device)
            emotion_out = self.emotion_encoder(face_crop)
            emotion_emb = emotion_out.embedding
        else:
            emotion_emb = lip_video.new_zeros(lip_video.shape[0], self.emotion_dim)

        speaker_emb = None
        if self.speaker_embedding is not None and speaker_id is not None:
            speaker_emb = self.speaker_embedding(speaker_id)

        T = mel_length or visual_feat.shape[1]
        visual_feat = self._align_temporal(visual_feat, T)

        # Prosody prediction
        prosody = None
        if self.use_prosody and self.prosody_predictor is not None:
            pitch, energy = self.prosody_predictor(visual_feat, emotion_emb, speaker_emb)
            prosody = torch.cat([pitch, energy], dim=-1)

        # Diffusion sampling
        mel = self.melgen.sample(
            visual_feat, emotion_emb, prosody,
            mel_length=T, guidance_scale=guidance_scale,
        )

        # Vocoder
        audio = self._get_vocoder(mel.device)(mel)

        return audio, mel

    def _get_vocoder(self, device: torch.device) -> HiFiGANVocoder:
        """Instantiate the frozen vocoder on first use."""
        if self.vocoder is None:
            self.vocoder = HiFiGANVocoder(**self.vocoder_cfg)
        self.vocoder.to(device)
        self.vocoder.eval()
        return self.vocoder

    @staticmethod
    def _align_temporal(feat: torch.Tensor, target_len: int) -> torch.Tensor:
        """Interpolate temporal dimension to match target length."""
        if feat.shape[1] == target_len:
            return feat
        # (B, T, D) -> (B, D, T) -> interpolate -> (B, D, T') -> (B, T', D)
        feat = feat.transpose(1, 2)
        feat = nn.functional.interpolate(feat, size=target_len, mode="linear", align_corners=False)
        return feat.transpose(1, 2)

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """MSE over valid temporal frames only."""
        loss = nn.functional.mse_loss(pred, target, reduction="none")
        if mask is None:
            return loss.mean()

        mask = mask.to(device=loss.device, dtype=loss.dtype)
        while mask.dim() < loss.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(loss)
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)
