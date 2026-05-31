"""
Emotion Encoder based on EmoNet.

Reference: Toisoul et al., "Estimation of continuous valence and arousal levels from faces
           in naturalistic conditions", Nature Machine Intelligence 2021.
GitHub: https://github.com/face-analysis/emonet

Outputs:
  - emotion_emb: 256-dim embedding (from pooled features before FC)
  - valence: continuous scalar in [-1, 1]
  - arousal: continuous scalar in [-1, 1]
  - expression: 8-class logits (neutral, happy, sad, surprise, fear, disgust, anger, contempt)

The model is always FROZEN — used only as a feature extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class EmotionOutput:
    """Container for emotion encoder outputs."""
    embedding: torch.Tensor     # (B, embed_dim) — main emotion embedding
    valence: torch.Tensor       # (B,) — continuous valence
    arousal: torch.Tensor       # (B,) — continuous arousal
    expression: torch.Tensor    # (B, n_expression) — class logits


# --------------------------------------------------------------------------- #
#  Lightweight stand-in architecture (EmoNet-like)
#  When the real emonet package is not available, this provides a compatible
#  interface for development and testing.
# --------------------------------------------------------------------------- #

class ConvBlock(nn.Module):
    """Residual convolution block used in EmoNet's HourGlass."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = out_ch // 2
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, mid, 1)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_ch, 1)

        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        return out + residual


class EmoNetStandin(nn.Module):
    """
    Simplified stand-in for EmoNet, matching its interface.

    Architecture: ResNet-18 backbone → global pool → FC heads for expression + V-A.
    This is used for development when the real emonet repo is not cloned.
    Replace with the real EmoNet for production training.
    """

    def __init__(self, n_expression: int = 8, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

        # Simplified backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
            nn.MaxPool2d(2),
            ConvBlock(256, 256),
            nn.AdaptiveAvgPool2d(4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed_fc = nn.Linear(256, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, n_expression + 2),  # expression logits + valence + arousal
        )

    def forward(self, x: torch.Tensor) -> dict:
        feat = self.features(x)                        # (B, 256, 4, 4)
        pooled = self.pool(feat).flatten(1)            # (B, 256)
        embedding = self.embed_fc(pooled)              # (B, embed_dim)
        out = self.head(embedding)                     # (B, n_expression + 2)
        expression = out[:, :-2]
        valence = out[:, -2]
        arousal = out[:, -1]
        return {
            "embedding": embedding,
            "expression": expression,
            "valence": valence,
            "arousal": arousal,
        }


class EmotionEncoder(nn.Module):
    """
    Frozen emotion encoder that extracts emotion embeddings from face crops.

    Supports two backends:
      1. "emonet" — loads the real EmoNet model (requires the emonet package).
      2. "standin" — uses a lightweight stand-in for development/testing.

    Args:
        backend: "emonet" or "standin".
        n_expression: Number of expression classes (default: 8).
        embed_dim: Dimension of the output emotion embedding (default: 256).
        checkpoint_path: Path to EmoNet checkpoint (for emonet backend).
    """

    def __init__(
        self,
        backend: str = "standin",
        n_expression: int = 8,
        embed_dim: int = 256,
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        self.backend_name = backend
        self.embed_dim = embed_dim
        self.n_expression = n_expression

        if backend == "emonet":
            self.model = self._load_emonet(checkpoint_path, n_expression)
            # EmoNet outputs 68-dim heatmap (pooled); project to embed_dim
            self.emo_proj = nn.Linear(68, embed_dim)
            nn.init.xavier_uniform_(self.emo_proj.weight)
            nn.init.zeros_(self.emo_proj.bias)
        elif backend == "standin":
            self.model = EmoNetStandin(n_expression=n_expression, embed_dim=embed_dim)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'emonet' or 'standin'.")

        # Freeze all parameters
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @staticmethod
    def _load_emonet(checkpoint_path: str | None, n_expression: int) -> nn.Module:
        try:
            from emonet.models import EmoNet as RealEmoNet
            model = RealEmoNet(n_expression=n_expression)
            if checkpoint_path is not None:
                state = EmotionEncoder._load_checkpoint_state(checkpoint_path)
                EmotionEncoder._load_state_or_fail(model, state, checkpoint_path)
                print(f"[EmotionEncoder] Loaded EmoNet from {checkpoint_path}")
            return model
        except ImportError as exc:
            raise RuntimeError(
                "emotion_encoder.backend='emonet' requires the emonet package. "
                "Use backend='standin' for development runs, or install EmoNet "
                "before running reportable emotion-conditioned experiments."
            ) from exc

    @staticmethod
    def _load_checkpoint_state(checkpoint_path: str) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model", "net"):
                nested = checkpoint.get(key)
                if isinstance(nested, dict):
                    checkpoint = nested
                    break
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"EmoNet checkpoint does not contain a state dict: {checkpoint_path}")
        return checkpoint

    @staticmethod
    def _select_best_state_dict(model: nn.Module, state_dict: dict) -> dict:
        expected = set(model.state_dict().keys())
        prefixes = ("module.", "model.", "net.", "emonet.")

        def strip_prefix_once(key: str, prefix: str) -> str:
            return key[len(prefix):] if key.startswith(prefix) else key

        def strip_all_known_prefixes(key: str) -> str:
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                        changed = True
            return key

        candidates = [state_dict]
        for prefix in prefixes:
            candidates.append({strip_prefix_once(str(key), prefix): value for key, value in state_dict.items()})
        candidates.append({strip_all_known_prefixes(str(key)): value for key, value in state_dict.items()})

        def score(candidate: dict) -> tuple[int, int]:
            keys = set(candidate.keys())
            return len(keys & expected), -len(keys - expected)

        return max(candidates, key=score)

    @staticmethod
    def _load_state_or_fail(model: nn.Module, state_dict: dict, checkpoint_path: str) -> None:
        state_dict = EmotionEncoder._select_best_state_dict(model, state_dict)
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except Exception as exc:
            raise RuntimeError(f"Could not load EmoNet checkpoint from {checkpoint_path}: {exc}") from exc

        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing or unexpected:
            details = []
            if missing:
                shown = ", ".join(missing[:5])
                extra = f"; {len(missing) - 5} more" if len(missing) > 5 else ""
                details.append(f"Missing keys: {shown}{extra}")
            if unexpected:
                shown = ", ".join(unexpected[:5])
                extra = f"; {len(unexpected) - 5} more" if len(unexpected) > 5 else ""
                details.append(f"Unexpected keys: {shown}{extra}")
            raise RuntimeError(
                f"EmoNet checkpoint/model mismatch for {checkpoint_path}: "
                + "; ".join(details)
            )

    @torch.no_grad()
    def forward(self, face: torch.Tensor) -> EmotionOutput:
        """
        Args:
            face: (B, 3, H, W) — RGB face crop, recommended 256x256.

        Returns:
            EmotionOutput with embedding, valence, arousal, expression.
        """
        out = self.model(face)
        if "embedding" in out:
            embedding = out["embedding"]
        elif "heatmap" in out:
            # Real EmoNet outputs heatmap (B, 68, 64, 64) — pool to embedding
            hm = out["heatmap"]
            embedding = hm.mean(dim=(2, 3))  # (B, 68)
            embedding = self.emo_proj(embedding)
        else:
            raise KeyError(f"EmoNet output missing 'embedding' or 'heatmap': {list(out.keys())}")
        return EmotionOutput(
            embedding=embedding,
            valence=out["valence"],
            arousal=out["arousal"],
            expression=out["expression"],
        )

    def get_va(self, face: torch.Tensor) -> torch.Tensor:
        """Convenience: return just (valence, arousal) as (B, 2)."""
        out = self.forward(face)
        return torch.stack([out.valence, out.arousal], dim=-1)

    def train(self, mode: bool = True):
        """Override: always stay in eval mode."""
        return super().train(False)
