"""
Prosody Predictor: predicts frame-level pitch (F0) and energy from
visual content features, emotion embeddings, and speaker embeddings.

Inspired by LipSody (ICASSP 2026) and FastSpeech2's variance adaptor.
Uses self-attention to capture temporal dependencies in prosody.
"""

import torch
import torch.nn as nn
import math


class ProsodyPredictor(nn.Module):
    """
    Predicts frame-level pitch (F0) and energy contours conditioned on
    content, emotion, and speaker information.

    Args:
        content_dim: Dimension of visual content features.
        emotion_dim: Dimension of emotion embedding.
        speaker_dim: Dimension of speaker embedding (0 to disable).
        hidden_dim: Internal hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        content_dim: int = 512,
        emotion_dim: int = 256,
        speaker_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.speaker_dim = max(0, speaker_dim)
        input_dim = content_dim + emotion_dim + self.speaker_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Separate heads for pitch and energy
        self.pitch_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        content_feat: torch.Tensor,
        emotion_emb: torch.Tensor,
        speaker_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            content_feat: (B, T, content_dim) — visual encoder output.
            emotion_emb: (B, emotion_dim) — emotion embedding from FER.
            speaker_emb: (B, speaker_dim) — optional speaker embedding.

        Returns:
            pitch: (B, T, 1) — predicted log-F0 (speaker-normalized).
            energy: (B, T, 1) — predicted log-energy (speaker-normalized).
        """
        B, T, _ = content_feat.shape

        # Expand global embeddings to temporal dimension
        emo_exp = emotion_emb.unsqueeze(1).expand(B, T, -1)  # (B, T, emotion_dim)

        parts = [content_feat, emo_exp]
        if self.speaker_dim > 0:
            if speaker_emb is None:
                speaker_emb = content_feat.new_zeros(B, self.speaker_dim)
            spk_exp = speaker_emb.unsqueeze(1).expand(B, T, -1)
            parts.append(spk_exp)

        x = torch.cat(parts, dim=-1)   # (B, T, input_dim)
        x = self.input_proj(x)         # (B, T, hidden_dim)
        x = self.transformer(x)        # (B, T, hidden_dim)

        pitch = self.pitch_head(x)     # (B, T, 1)
        energy = self.energy_head(x)   # (B, T, 1)

        return pitch, energy
