"""
HiFi-GAN Vocoder wrapper for mel spectrogram to waveform conversion.

Reference: Kong et al., "HiFi-GAN: Generative Adversarial Networks for
           Efficient and High Fidelity Speech Synthesis", NeurIPS 2020.
GitHub: https://github.com/jik876/hifi-gan

This wrapper supports:
  1. Loading a pre-trained HiFi-GAN generator from a checkpoint.
  2. A lightweight placeholder generator for development.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math


class ResBlock(nn.Module):
    """Multi-receptive field residual block for HiFi-GAN."""

    def __init__(self, channels: int, kernel_size: int = 3, dilations: tuple = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size,
                          dilation=d, padding=(kernel_size * d - d) // 2)
            )
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
            )
            for _ in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class HiFiGANGenerator(nn.Module):
    """
    Simplified HiFi-GAN generator (V1 config).

    Converts mel spectrograms to audio waveforms via transposed convolution
    upsampling with multi-receptive field residual blocks.
    """

    def __init__(
        self,
        mel_dim: int = 80,
        upsample_rates: tuple = (8, 8, 2, 2),
        upsample_kernel_sizes: tuple = (16, 16, 4, 4),
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: tuple = (3, 7, 11),
        resblock_dilation_sizes: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(mel_dim, upsample_initial_channel, 7, padding=3)
        )

        ch = upsample_initial_channel
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.utils.parametrizations.weight_norm(
                    nn.ConvTranspose1d(ch, ch // 2, k, stride=u, padding=(k - u) // 2)
                )
            )
            ch_new = ch // 2
            for j, (rk, rd) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch_new, rk, rd))
            ch = ch_new

        self.conv_post = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(ch, 1, 7, padding=3)
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, mel_dim, T_mel)

        Returns:
            audio: (B, 1, T_audio) where T_audio = T_mel * prod(upsample_rates)
        """
        x = self.conv_pre(mel)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)

            # Apply all resblocks for this upsampling level
            n_resblocks = len(self.resblocks) // self.num_upsamples
            xs = None
            for j in range(n_resblocks):
                idx = i * n_resblocks + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs += self.resblocks[idx](x)
            x = xs / n_resblocks

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x


class HiFiGANVocoder(nn.Module):
    """
    Wrapper that loads a pre-trained HiFi-GAN and provides a simple interface.
    Always frozen — no gradient computation.

    Args:
        checkpoint_path: Path to HiFi-GAN generator checkpoint (.pth).
        config_path: Path to HiFi-GAN config.json (optional, for custom configs).
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        config_path: str | None = None,
        mel_dim: int = 80,
        mel_scale: str = "log",
    ):
        super().__init__()
        self.generator = self._build_generator(config_path, mel_dim=mel_dim)
        self.mel_scale = mel_scale  # "log" (HiFi-GAN native) or "db" (dB-scaled input)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, config_path)
        else:
            print("[HiFiGANVocoder] No checkpoint provided; using an untrained development vocoder.")

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @staticmethod
    def _as_tuple(value, default):
        if value is None:
            return default
        if isinstance(value, list):
            return tuple(HiFiGANVocoder._as_tuple(v, v) for v in value)
        return value

    def _build_generator(self, config_path: str | None = None, mel_dim: int = 80) -> HiFiGANGenerator:
        """Build a HiFi-GAN generator from a config.json when provided."""
        if config_path is None:
            return HiFiGANGenerator(mel_dim=mel_dim)

        with open(config_path) as f:
            cfg = json.load(f)

        mel_dim = cfg.get("num_mels", cfg.get("n_mels", cfg.get("mel_dim", mel_dim)))
        return HiFiGANGenerator(
            mel_dim=mel_dim,
            upsample_rates=self._as_tuple(cfg.get("upsample_rates"), (8, 8, 2, 2)),
            upsample_kernel_sizes=self._as_tuple(
                cfg.get("upsample_kernel_sizes"), (16, 16, 4, 4)
            ),
            upsample_initial_channel=cfg.get("upsample_initial_channel", 512),
            resblock_kernel_sizes=self._as_tuple(
                cfg.get("resblock_kernel_sizes"), (3, 7, 11)
            ),
            resblock_dilation_sizes=self._as_tuple(
                cfg.get("resblock_dilation_sizes"), ((1, 3, 5), (1, 3, 5), (1, 3, 5))
            ),
        )

    @staticmethod
    def _extract_generator_state_dict(ckpt):
        if not isinstance(ckpt, dict):
            return ckpt
        for key in ("generator", "model_g", "state_dict", "model"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                ckpt = value
                break
        if not isinstance(ckpt, dict):
            return ckpt

        normalized = {}
        for key, value in ckpt.items():
            name = str(key)
            for prefix in ("module.", "generator.", "model_g.", "model."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            normalized[name] = value
        return normalized

    @staticmethod
    def _convert_speechbrain_keys(state_dict: dict) -> dict:
        """Convert SpeechBrain HiFi-GAN keys to standard format.

        SpeechBrain wraps convolutions as Conv1d.conv, so keys have an extra
        `.conv.` segment and use old-style weight_norm (weight_g, weight_v).
        This reconstructs plain weight = weight_v * (weight_g / ||weight_v||).
        """
        # Strip `.conv.` from SpeechBrain wrapper
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace(".conv.", ".")] = v

        # Fuse weight_g + weight_v into plain weight
        fused = {}
        handled = set()
        for k, v in cleaned.items():
            if k.endswith(".weight_g"):
                base = k[:-len(".weight_g")]
                wv_key = base + ".weight_v"
                if wv_key in cleaned:
                    wg = cleaned[k]
                    wv = cleaned[wv_key]
                    # weight = weight_v * (weight_g / ||weight_v||)
                    norm = wv.norm(dim=tuple(range(1, wv.dim())), keepdim=True)
                    fused[base + ".weight"] = wv * (wg / (norm + 1e-8))
                    handled.add(k)
                    handled.add(wv_key)
            elif k.endswith(".weight_v"):
                pass  # handled above
        for k, v in cleaned.items():
            if k not in handled:
                fused[k] = v
        return fused

    def _load_checkpoint(self, checkpoint_path: str, config_path: str | None = None):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = self._extract_generator_state_dict(ckpt)
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"HiFi-GAN checkpoint does not contain a state dict: {checkpoint_path}")

        # Detect SpeechBrain format (has .conv.weight_g keys)
        if any(".conv.weight_g" in k or ".conv.weight_v" in k for k in state_dict):
            state_dict = self._convert_speechbrain_keys(state_dict)

        # Remove weight_norm from our generator so we can load plain weights
        for module in self.generator.modules():
            if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
                nn.utils.parametrize.remove_parametrizations(module, "weight")

        try:
            incompatible = self.generator.load_state_dict(state_dict, strict=False)
        except Exception as exc:
            raise RuntimeError(f"Could not load HiFi-GAN checkpoint {checkpoint_path}: {exc}") from exc

        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing:
            shown = ", ".join(missing[:5])
            extra = f"; {len(missing) - 5} more" if len(missing) > 5 else ""
            raise RuntimeError(
                "HiFi-GAN checkpoint did not load all generator parameters. "
                f"Missing keys: {shown}{extra}. "
                "Check that vocoder.config_path matches the checkpoint architecture."
            )

        detail = f"; ignored unexpected keys={len(unexpected)}" if unexpected else ""
        print(f"[HiFiGANVocoder] Loaded checkpoint from {checkpoint_path}{detail}")

    @torch.no_grad()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, mel_dim, T) mel spectrogram. Scale governed by self.mel_scale.

        Returns:
            audio: (B, T_audio) waveform.
        """
        if self.mel_scale == "db":
            # Convert dB-scaled mel (e.g. ref=np.max in [-80, 0]) to natural log.
            # ln(x) = log10(x) * ln(10), and dB = 20 * log10(amp).
            mel = mel * (math.log(10.0) / 20.0)
        audio = self.generator(mel)  # (B, 1, T_audio)
        return audio.squeeze(1)

    def train(self, mode: bool = True):
        return super().train(False)
