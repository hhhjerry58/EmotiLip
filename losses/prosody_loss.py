"""
Prosody prediction loss.

Computes MSE between predicted and ground-truth pitch (log-F0) and energy.
Supports speaker-wise normalization of pitch and energy targets.
"""

import torch
import torch.nn as nn


class ProsodyLoss(nn.Module):
    """
    MSE loss on predicted pitch and energy contours.

    Args:
        lambda_pitch: Weight for pitch loss.
        lambda_energy: Weight for energy loss.
    """

    def __init__(self, lambda_pitch: float = 1.0, lambda_energy: float = 1.0):
        super().__init__()
        self.lambda_pitch = lambda_pitch
        self.lambda_energy = lambda_energy
        self.mse = nn.MSELoss()

    def forward(
        self,
        pitch_pred: torch.Tensor,
        pitch_target: torch.Tensor,
        energy_pred: torch.Tensor,
        energy_target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pitch_pred: (B, T, 1) predicted log-F0.
            pitch_target: (B, T, 1) ground truth log-F0.
            energy_pred: (B, T, 1) predicted log-energy.
            energy_target: (B, T, 1) ground truth log-energy.
            mask: (B, T) optional mask for valid frames (1=valid, 0=padding).

        Returns:
            Dict with 'pitch_loss', 'energy_loss', 'total'.
        """
        if mask is None:
            pitch_loss = self.mse(pitch_pred, pitch_target)
            energy_loss = self.mse(energy_pred, energy_target)
        else:
            mask = mask.to(device=pitch_pred.device, dtype=pitch_pred.dtype).unsqueeze(-1)
            pitch_sq = (pitch_pred - pitch_target).pow(2) * mask
            energy_sq = (energy_pred - energy_target).pow(2) * mask
            denom = mask.sum().clamp_min(1.0)
            pitch_loss = pitch_sq.sum() / denom
            energy_loss = energy_sq.sum() / denom

        total = self.lambda_pitch * pitch_loss + self.lambda_energy * energy_loss

        return {
            "pitch_loss": pitch_loss,
            "energy_loss": energy_loss,
            "total": total,
        }
