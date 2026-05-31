"""
AdaIN (Adaptive Instance Normalization) conditioning.

Reference: Huang & Belongie, "Arbitrary Style Transfer in Real-time with AdaIN", ICCV 2017.

AdaIN(x, c) = sigma(c) * ((x - mu(x)) / sigma(x)) + mu(c)

First normalizes features via instance norm, then applies learned affine transform
from the conditioning signal. Effective for global style/emotion conditioning.
Used in StyleGAN, StyleTTS, and various style-transfer speech systems.
"""

import torch
import torch.nn as nn


class AdaINConditioning(nn.Module):
    """
    Adaptive Instance Normalization conditioned on an emotion embedding.

    Args:
        condition_dim: Dimension of the conditioning embedding.
        feature_dim: Channel dimension of the features.
        hidden_dim: Hidden dimension of the conditioning MLP.
    """

    def __init__(self, condition_dim: int, feature_dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.norm = nn.InstanceNorm1d(feature_dim, affine=False)

        if hidden_dim is not None:
            self.affine = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, feature_dim * 2),
            )
        else:
            self.affine = nn.Linear(condition_dim, feature_dim * 2)

        self._init_weights()

    def _init_weights(self):
        last = self.affine[-1] if isinstance(self.affine, nn.Sequential) else self.affine
        nn.init.zeros_(last.weight)
        # gamma=1, beta=0 at init
        last.bias.data[:last.bias.shape[0] // 2].fill_(1.0)
        last.bias.data[last.bias.shape[0] // 2:].fill_(0.0)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D, T) — channel-first format for InstanceNorm1d.
            condition: (B, condition_dim)

        Returns:
            Modulated features (B, D, T).
        """
        # If features come in (B, T, D), transpose for InstanceNorm1d
        transposed = False
        if features.shape[-1] != features.shape[1] and features.dim() == 3:
            # Heuristic: if last dim matches condition projection output, it's (B, T, D)
            if features.shape[-1] == self.norm.num_features:
                features = features.transpose(1, 2)  # -> (B, D, T)
                transposed = True

        h = self.affine(condition)  # (B, 2*D)
        gamma, beta = h.chunk(2, dim=-1)  # each (B, D)
        gamma = gamma.unsqueeze(-1)  # (B, D, 1)
        beta = beta.unsqueeze(-1)

        out = gamma * self.norm(features) + beta

        if transposed:
            out = out.transpose(1, 2)  # back to (B, T, D)

        return out
