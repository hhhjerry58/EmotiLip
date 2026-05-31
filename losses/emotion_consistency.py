"""
Emotion Consistency Loss.

Ensures the emotion in the generated speech matches the emotion in the input face.
Uses frozen pre-trained Speech Emotion Recognition (SER) models as critics.

Two complementary paths:
  1. Classification loss: SER classifies generated audio → cross-entropy with face emotion label.
  2. V-A regression loss: SER predicts V-A from audio → MSE with face V-A from EmoNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionConsistencyLoss(nn.Module):
    """
    Computes emotion consistency between generated speech and input face expression.

    Uses frozen pre-trained SER models. The models are loaded externally
    and passed in, or can be loaded lazily via load_ser_model().

    Args:
        lambda_cls: Weight for classification loss.
        lambda_va: Weight for V-A regression loss.
        lambda_emb: Weight for embedding cosine similarity loss.
        num_emotion_classes: Number of discrete emotion classes.
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_va: float = 1.0,
        lambda_emb: float = 0.5,
        num_emotion_classes: int = 8,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_va = lambda_va
        self.lambda_emb = lambda_emb
        self.num_emotion_classes = num_emotion_classes

        self.cls_loss = nn.CrossEntropyLoss()
        self.va_loss = nn.MSELoss()

        # Placeholders for external SER models (set via load methods)
        self._ser_classifier = None
        self._ser_va_regressor = None

    def load_emotion2vec(self, model_name: str = "iic/emotion2vec_plus_base"):
        """
        Load emotion2vec+ as the SER classifier.
        Requires: pip install funasr
        """
        try:
            from funasr import AutoModel
            self._ser_classifier = AutoModel(model=model_name)
            print(f"[EmotionConsistencyLoss] Loaded emotion2vec: {model_name}")
        except ImportError:
            print("[EmotionConsistencyLoss] funasr not installed. SER classifier unavailable.")

    def load_wav2vec2_emotion(
        self,
        model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    ):
        """
        Load wav2vec2-emotion as the V-A regressor.
        Requires: pip install transformers
        """
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            self._ser_va_regressor = {
                "model_name": model_name,
                "loaded": False,
            }
            print(f"[EmotionConsistencyLoss] wav2vec2-emotion registered: {model_name}")
        except ImportError:
            print("[EmotionConsistencyLoss] transformers not installed. V-A regressor unavailable.")

    def forward(
        self,
        generated_audio: torch.Tensor | None = None,
        target_emotion_label: torch.Tensor | None = None,
        target_va: torch.Tensor | None = None,
        audio_emotion_logits: torch.Tensor | None = None,
        audio_va: torch.Tensor | None = None,
        audio_emb: torch.Tensor | None = None,
        visual_emb: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute emotion consistency loss.

        Can accept either raw audio (for end-to-end) or pre-computed
        emotion predictions (for modular training).

        Args:
            generated_audio: (B, T_audio) raw waveform (optional).
            target_emotion_label: (B,) discrete emotion labels from FER.
            target_va: (B, 2) valence-arousal from FER.
            audio_emotion_logits: (B, C) pre-computed emotion logits (optional).
            audio_va: (B, 2) pre-computed V-A from audio (optional).
            audio_emb: (B, D) pre-computed audio emotion embedding (optional).
            visual_emb: (B, D) visual emotion embedding for cosine sim (optional).

        Returns:
            Dict with 'cls_loss', 'va_loss', 'emb_loss', 'total'.
        """
        losses = {}
        total = torch.tensor(0.0, device=self._get_device(
            target_emotion_label, target_va, audio_emotion_logits
        ))

        # Classification loss
        if audio_emotion_logits is not None and target_emotion_label is not None:
            cls = self.cls_loss(audio_emotion_logits, target_emotion_label)
            losses["cls_loss"] = cls
            total = total + self.lambda_cls * cls

        # V-A regression loss
        if audio_va is not None and target_va is not None:
            va = self.va_loss(audio_va, target_va)
            losses["va_loss"] = va
            total = total + self.lambda_va * va

        # Embedding cosine similarity loss
        if audio_emb is not None and visual_emb is not None:
            cos_sim = F.cosine_similarity(audio_emb, visual_emb, dim=-1)
            emb_loss = (1 - cos_sim).mean()
            losses["emb_loss"] = emb_loss
            total = total + self.lambda_emb * emb_loss

        losses["total"] = total
        return losses

    @staticmethod
    def _get_device(*tensors) -> torch.device:
        for t in tensors:
            if t is not None:
                return t.device
        return torch.device("cpu")
