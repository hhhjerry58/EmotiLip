"""
Visual Encoder for lip-to-speech synthesis.

Two backends:
1. AV-HuBERT (preferred): Pre-trained self-supervised audio-visual model from Meta.
   Extracts rich visual representations from lip video. Frozen during training.
2. ResNet3D: Lightweight 3D-CNN for when AV-HuBERT is unavailable or compute is
   limited. Select this explicitly instead of relying on silent fallback.

Both produce frame-level features: (B, T, D).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet3DBlock(nn.Module):
    """Basic 3D residual block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=(1, stride, stride), padding=1
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=1, stride=(1, stride, stride),
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class ResNet3DBackbone(nn.Module):
    """
    Lightweight 3D ResNet-18 style backbone for lip video encoding.

    Input: (B, 1, T, H, W) — grayscale lip-cropped video.
    Output: (B, T, output_dim)
    """

    def __init__(self, output_dim: int = 512, base_channels: int = 64):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        self.frontend = nn.Sequential(
            nn.Conv3d(1, c1, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(c1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        self.layer1 = self._make_layer(c1, c1, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(c1, c2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(c2, c3, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(c3, c4, num_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(c4, output_dim) if output_dim != c4 else nn.Identity()

    @staticmethod
    def _make_layer(
        in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [ResNet3DBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNet3DBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T, H, W) — grayscale lip video, H=W=96 recommended.

        Returns:
            (B, T, output_dim)
        """
        B, C, T, H, W = x.shape
        out = self.frontend(x)           # (B, 64, T, H/4, W/4)
        out = self.layer1(out)           # (B, 64, T, H/4, W/4)
        out = self.layer2(out)           # (B, 128, T, H/8, W/8)
        out = self.layer3(out)           # (B, 256, T, H/16, W/16)
        out = self.layer4(out)           # (B, 512, T, H/32, W/32)

        # Pool spatial dims, keep temporal
        B2, C2, T2, H2, W2 = out.shape
        out = out.permute(0, 2, 1, 3, 4).reshape(B2 * T2, C2, H2, W2)
        out = self.pool(out).squeeze(-1).squeeze(-1)  # (B*T, 512)
        out = out.reshape(B2, T2, -1)                 # (B, T, 512)
        out = self.proj(out)                           # (B, T, output_dim)
        return out


class AVHuBERTWrapper(nn.Module):
    """
    Wrapper around a pre-trained AV-HuBERT model used as a frozen feature extractor.

    Expects fairseq/AV-HuBERT support to be installed and a checkpoint path
    provided. Loading fails eagerly so final runs cannot silently use a
    different visual encoder than the config declares.

    Output: (B, T, output_dim) where output_dim is projected from AV-HuBERT's hidden size.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        output_dim: int = 512,
        layer: int = -1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.layer = layer
        self.encoder = None
        self.proj = None

        if checkpoint_path is None:
            raise ValueError("visual_encoder.backend='avhubert' requires visual_encoder.checkpoint_path.")
        self._load_avhubert(checkpoint_path)

    def _load_avhubert(self, checkpoint_path: str):
        try:
            from fairseq import checkpoint_utils
        except ImportError as exc:
            raise RuntimeError(
                "visual_encoder.backend='avhubert' requires fairseq and the "
                "AV-HuBERT code path. Use backend='resnet3d' for development "
                "runs, or install the required package before reportable "
                "AV-HuBERT experiments."
            ) from exc

        try:
            models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
        except Exception as exc:
            raise RuntimeError(f"Could not load AV-HuBERT checkpoint from {checkpoint_path}: {exc}") from exc

        if not models:
            raise RuntimeError(f"AV-HuBERT checkpoint loaded no models: {checkpoint_path}")

        self.encoder = models[0]
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # AV-HuBERT hidden size: Base=768, Large=1024
        hidden_size = self.encoder.cfg.encoder_embed_dim
        self.proj = nn.Linear(hidden_size, self.output_dim)
        print(f"[AVHuBERTWrapper] Loaded AV-HuBERT (hidden={hidden_size}) from {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        self._load_avhubert(checkpoint_path)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, 1, T, H, W) grayscale lip video.

        Returns:
            (B, T', output_dim) — T' may differ from T due to subsampling.
        """
        if self.encoder is None:
            raise RuntimeError(
                "AV-HuBERT encoder not loaded. Call load_checkpoint() or use ResNet3D."
            )

        with torch.no_grad():
            # AV-HuBERT expects {'video': (B, 1, T, H, W), 'audio': None}
            features = self.encoder.extract_features(
                source={"video": video, "audio": None},
                padding_mask=None,
                output_layer=self.layer,
            )[0]  # (B, T', hidden_size)

        return self.proj(features)


class VisualEncoder(nn.Module):
    """
    Unified visual encoder interface.

    Args:
        backend: "avhubert" or "resnet3d".
        output_dim: Output feature dimension.
        checkpoint_path: Path to AV-HuBERT checkpoint (only for avhubert backend).
        freeze: Whether to freeze the backbone (recommended for avhubert).
    """

    def __init__(
        self,
        backend: str = "resnet3d",
        output_dim: int = 512,
        checkpoint_path: str | None = None,
        freeze: bool = True,
        base_channels: int = 64,
    ):
        super().__init__()
        self.backend_name = backend
        self.output_dim = output_dim

        if backend == "avhubert":
            self.backbone = AVHuBERTWrapper(
                checkpoint_path=checkpoint_path,
                output_dim=output_dim,
            )
            if freeze:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                # Keep projection trainable
                if self.backbone.proj is not None:
                    for p in self.backbone.proj.parameters():
                        p.requires_grad = True

        elif backend == "resnet3d":
            self.backbone = ResNet3DBackbone(output_dim=output_dim, base_channels=base_channels)
            if freeze:
                for p in self.backbone.parameters():
                    p.requires_grad = False

        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'avhubert' or 'resnet3d'.")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, 1, T, H, W) grayscale lip video.

        Returns:
            (B, T, output_dim)
        """
        return self.backbone(video)
