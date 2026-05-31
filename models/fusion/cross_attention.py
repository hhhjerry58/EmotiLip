"""
Cross-Attention conditioning.

Reference: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022.

The conditioning embedding is projected to keys (K) and values (V),
while the main features serve as queries (Q). Standard multi-head attention
allows the model to learn WHERE and HOW MUCH to attend to the conditioning signal.

Most expressive fusion method, but higher computational cost.
"""

import torch
import torch.nn as nn


class CrossAttentionConditioning(nn.Module):
    """
    Cross-attention fusion: features attend to conditioning signal.

    Supports both single-vector conditioning (B, D_cond) and
    sequence conditioning (B, S, D_cond).

    Args:
        condition_dim: Dimension of the conditioning embedding.
        feature_dim: Dimension of the main features (query dim).
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.to_q = nn.Linear(feature_dim, feature_dim)
        self.to_k = nn.Linear(condition_dim, feature_dim)
        self.to_v = nn.Linear(condition_dim, feature_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, D) — sequence of feature vectors.
            condition: (B, D_cond) or (B, S, D_cond) — conditioning signal.

        Returns:
            Residual-connected output (B, T, D).
        """
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)  # (B, 1, D_cond)

        q = self.to_q(features)   # (B, T, D)
        k = self.to_k(condition)  # (B, S, D)
        v = self.to_v(condition)  # (B, S, D)

        attn_out, _ = self.mha(q, k, v)  # (B, T, D)
        attn_out = self.out_proj(attn_out)

        return self.norm(features + attn_out)
