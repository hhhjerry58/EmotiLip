"""
FiLM (Feature-wise Linear Modulation) conditioning.

Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018.

FiLM(x | c) = gamma(c) * x + beta(c)

Simplest and most general conditioning method. Does not normalize x first
(unlike AdaIN). Can be inserted at any layer of the main network.
"""

import torch
import torch.nn as nn


class FiLMConditioning(nn.Module):
    """
    Apply Feature-wise Linear Modulation conditioned on an emotion embedding.

    Args:
        condition_dim: Dimension of the conditioning embedding (e.g., 256 from EmoNet).
        feature_dim: Dimension of the features to modulate (channel dim).
        hidden_dim: Hidden dimension of the conditioning MLP. If None, uses a single linear layer.
    """

    def __init__(self, condition_dim: int, feature_dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is not None:
            self.gamma_net = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, feature_dim),
            )
            self.beta_net = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, feature_dim),
            )
        else:
            self.gamma_net = nn.Linear(condition_dim, feature_dim)
            self.beta_net = nn.Linear(condition_dim, feature_dim)

        # Initialize gamma to 1 and beta to 0 (identity transform at init)
        self._init_weights()

    def _init_weights(self):
        if isinstance(self.gamma_net, nn.Sequential):
            nn.init.zeros_(self.gamma_net[-1].weight)
            nn.init.ones_(self.gamma_net[-1].bias)
            nn.init.zeros_(self.beta_net[-1].weight)
            nn.init.zeros_(self.beta_net[-1].bias)
        else:
            nn.init.zeros_(self.gamma_net.weight)
            nn.init.ones_(self.gamma_net.bias)
            nn.init.zeros_(self.beta_net.weight)
            nn.init.zeros_(self.beta_net.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, D) or (B, D, T) — the feature tensor to modulate.
            condition: (B, condition_dim) — the conditioning embedding (e.g., emotion).

        Returns:
            Modulated features with the same shape as input.
        """
        gamma = self.gamma_net(condition)  # (B, D)
        beta = self.beta_net(condition)    # (B, D)

        if features.dim() == 3:
            if features.shape[-1] == gamma.shape[-1]:
                # (B, T, D) format
                gamma = gamma.unsqueeze(1)  # (B, 1, D)
                beta = beta.unsqueeze(1)
            else:
                # (B, D, T) format
                gamma = gamma.unsqueeze(-1)  # (B, D, 1)
                beta = beta.unsqueeze(-1)

        return gamma * features + beta
