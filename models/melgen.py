"""
MelGen: Conditional Diffusion-based Mel Spectrogram Generator.

Implements a DDPM (Denoising Diffusion Probabilistic Model) that generates
80-bin mel spectrograms conditioned on:
  - Visual features (from lip video encoder)
  - Emotion embeddings (via FiLM / AdaIN / Cross-Attention)
  - Prosody features (predicted pitch + energy)

Architecture follows the DiffWave / LipVoicer paradigm:
  - 1D dilated residual blocks with gated activations
  - Diffusion timestep embedding via sinusoidal + MLP
  - Conditioning injected at each residual block

References:
  - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
  - Kong et al., "DiffWave: A Versatile Diffusion Model for Audio Synthesis", ICLR 2021
  - Yocha et al., "LipVoicer", ICLR 2024
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import build_fusion


# --------------------------------------------------------------------------- #
#  Diffusion schedule utilities
# --------------------------------------------------------------------------- #

def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)


class DiffusionSchedule:
    """Precomputes and stores all diffusion schedule tensors."""

    def __init__(self, timesteps: int = 1000, schedule: str = "linear"):
        self.timesteps = timesteps

        if schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule '{schedule}'")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def to(self, device: torch.device) -> "DiffusionSchedule":
        for attr in [
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas", "posterior_variance",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Gather values from `a` at indices `t` and reshape for broadcasting."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# --------------------------------------------------------------------------- #
#  Timestep embedding
# --------------------------------------------------------------------------- #

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# --------------------------------------------------------------------------- #
#  Denoiser: dilated residual blocks with conditioning
# --------------------------------------------------------------------------- #

class ResidualBlock(nn.Module):
    """
    Gated residual block with dilated convolution.
    Conditioned on diffusion timestep + visual features + emotion.
    """

    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        cond_dim: int,
        dilation: int = 1,
        kernel_size: int = 3,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.dilated_conv = nn.Conv1d(
            channels, 2 * channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.time_proj = nn.Linear(time_emb_dim, channels)
        self.cond_proj = nn.Conv1d(cond_dim, 2 * channels, 1)
        self.res_conv = nn.Conv1d(channels, channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) noisy mel features
            time_emb: (B, time_emb_dim)
            cond: (B, cond_dim, T) conditioning features

        Returns:
            (B, C, T) residual output
        """
        h = x + self.time_proj(time_emb).unsqueeze(-1)
        h = self.dilated_conv(h) + self.cond_proj(cond)

        # Gated activation
        h_a, h_b = h.chunk(2, dim=1)
        h = torch.tanh(h_a) * torch.sigmoid(h_b)

        h = self.res_conv(h)
        return (h + x) / math.sqrt(2.0)


class Denoiser(nn.Module):
    """
    U-Net-like denoiser for mel spectrogram generation.

    Uses stacked dilated residual blocks with gated activations,
    conditioned on timestep, visual features, and emotion embedding.
    """

    def __init__(
        self,
        mel_dim: int = 80,
        channels: int = 256,
        cond_dim: int = 512,
        emotion_dim: int = 256,
        time_emb_dim: int = 256,
        num_layers: int = 20,
        dilation_cycle: int = 10,
        fusion_type: str = "film",
    ):
        super().__init__()

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.GELU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Input projection: noisy mel → channels
        self.input_proj = nn.Conv1d(mel_dim, channels, 1)

        # Conditioning projection: visual features → cond_dim
        self.cond_proj = nn.Conv1d(cond_dim, cond_dim, 1)

        # Emotion conditioning via fusion module (applied to cond)
        self.emotion_fusion = build_fusion(
            fusion_type,
            condition_dim=emotion_dim,
            feature_dim=cond_dim,
        )

        # Residual blocks with cycling dilation
        self.blocks = nn.ModuleList([
            ResidualBlock(
                channels=channels,
                time_emb_dim=time_emb_dim,
                cond_dim=cond_dim,
                dilation=2 ** (i % dilation_cycle),
            )
            for i in range(num_layers)
        ])

        # Output projection: channels → mel_dim
        self.output_proj = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.Conv1d(channels, mel_dim, 1),
        )
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        noisy_mel: torch.Tensor,
        t: torch.Tensor,
        visual_cond: torch.Tensor,
        emotion_emb: torch.Tensor,
        prosody: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict noise epsilon given noisy mel and conditions.

        Args:
            noisy_mel: (B, mel_dim, T) — noisy mel spectrogram at timestep t.
            t: (B,) — integer diffusion timesteps.
            visual_cond: (B, T, cond_dim) — visual features from encoder.
            emotion_emb: (B, emotion_dim) — emotion embedding.
            prosody: (B, T, 2) — optional predicted [pitch, energy].

        Returns:
            (B, mel_dim, T) — predicted noise.
        """
        # Timestep embedding
        time_emb = self.time_mlp(t)  # (B, time_emb_dim)

        # Build conditioning: visual features + optional prosody
        cond = visual_cond  # (B, T, cond_dim)
        if prosody is not None:
            # Project prosody to cond_dim and add
            # Simple addition — prosody is low-dim and carries supplementary info
            B, T, _ = cond.shape
            prosody_padded = F.pad(prosody, (0, cond.shape[-1] - prosody.shape[-1]))
            cond = cond + prosody_padded

        # Apply emotion fusion to conditioning
        cond = self.emotion_fusion(cond, emotion_emb)  # (B, T, cond_dim)
        cond = cond.transpose(1, 2)  # (B, cond_dim, T) for Conv1d
        cond = self.cond_proj(cond)

        # Input projection
        x = self.input_proj(noisy_mel)  # (B, channels, T)

        # Residual blocks
        for block in self.blocks:
            x = block(x, time_emb, cond)

        return self.output_proj(x)  # (B, mel_dim, T)


# --------------------------------------------------------------------------- #
#  MelGen: full diffusion pipeline
# --------------------------------------------------------------------------- #

class MelGen(nn.Module):
    """
    Complete diffusion-based mel spectrogram generator.

    Wraps the denoiser with the DDPM forward/reverse process.

    Args:
        mel_dim: Number of mel bins.
        channels: Denoiser internal channel width.
        cond_dim: Visual conditioning dimension.
        emotion_dim: Emotion embedding dimension.
        num_layers: Number of residual blocks in the denoiser.
        timesteps: Number of diffusion timesteps.
        schedule: Beta schedule type ("linear" or "cosine").
        fusion_type: Emotion fusion method ("film", "adain", "cross_attention").
    """

    def __init__(
        self,
        mel_dim: int = 80,
        channels: int = 256,
        cond_dim: int = 512,
        emotion_dim: int = 256,
        num_layers: int = 20,
        timesteps: int = 1000,
        schedule: str = "linear",
        fusion_type: str = "film",
    ):
        super().__init__()
        self.mel_dim = mel_dim
        self.timesteps = timesteps

        self.schedule = DiffusionSchedule(timesteps, schedule)
        self.denoiser = Denoiser(
            mel_dim=mel_dim,
            channels=channels,
            cond_dim=cond_dim,
            emotion_dim=emotion_dim,
            num_layers=num_layers,
            fusion_type=fusion_type,
        )

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward diffusion: add noise to x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = _extract(self.schedule.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha = _extract(
            self.schedule.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def compute_loss(
        self,
        mel: torch.Tensor,
        visual_cond: torch.Tensor,
        emotion_emb: torch.Tensor,
        prosody: torch.Tensor | None = None,
        mel_mask: torch.Tensor | None = None,
        return_pred: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Compute training loss (simple MSE on predicted noise).

        Args:
            mel: (B, mel_dim, T) — ground truth mel spectrogram.
            visual_cond: (B, T, cond_dim) — visual features.
            emotion_emb: (B, emotion_dim) — emotion embedding.
            prosody: (B, T, 2) — optional prosody features.
            mel_mask: (B, T) — 1 for valid frames, 0 for padded frames.

            return_pred: Return the one-step predicted clean mel in addition to
                the diffusion loss. This is useful for auxiliary mel-level
                losses without running full reverse diffusion during training.

        Returns:
            Scalar loss, or (loss, predicted_clean_mel) when return_pred=True.
        """
        self.schedule.to(mel.device)

        B = mel.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=mel.device)
        noise = torch.randn_like(mel)

        noisy_mel = self.q_sample(mel, t, noise)

        predicted_noise = self.denoiser(noisy_mel, t, visual_cond, emotion_emb, prosody)

        loss = F.mse_loss(predicted_noise, noise, reduction="none")
        predicted_clean = None
        if return_pred:
            sqrt_alpha = _extract(self.schedule.sqrt_alphas_cumprod, t, mel.shape)
            sqrt_one_minus_alpha = _extract(self.schedule.sqrt_one_minus_alphas_cumprod, t, mel.shape)
            predicted_clean = (noisy_mel - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha.clamp_min(1e-8)

        if mel_mask is None:
            diffusion_loss = loss.mean()
            return (diffusion_loss, predicted_clean) if return_pred else diffusion_loss

        mask = mel_mask.to(device=loss.device, dtype=loss.dtype)
        while mask.dim() < loss.dim():
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(loss)
        diffusion_loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        return (diffusion_loss, predicted_clean) if return_pred else diffusion_loss

    @torch.no_grad()
    def sample(
        self,
        visual_cond: torch.Tensor,
        emotion_emb: torch.Tensor,
        prosody: torch.Tensor | None = None,
        mel_length: int | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Reverse diffusion sampling (DDPM).

        Args:
            visual_cond: (B, T, cond_dim)
            emotion_emb: (B, emotion_dim)
            prosody: (B, T, 2) optional
            mel_length: Override temporal length. If None, inferred from visual_cond.
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance).

        Returns:
            (B, mel_dim, T) — generated mel spectrogram.
        """
        self.schedule.to(visual_cond.device)
        device = visual_cond.device
        B = visual_cond.shape[0]
        T = mel_length or visual_cond.shape[1]

        # Start from pure noise
        x = torch.randn(B, self.mel_dim, T, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            predicted_noise = self.denoiser(x, t, visual_cond, emotion_emb, prosody)

            # Classifier-free guidance
            if guidance_scale != 1.0:
                # Unconditional prediction (zero emotion embedding)
                uncond_noise = self.denoiser(
                    x, t, visual_cond,
                    torch.zeros_like(emotion_emb),
                    prosody,
                )
                predicted_noise = (
                    uncond_noise + guidance_scale * (predicted_noise - uncond_noise)
                )

            # DDPM reverse step
            alpha = _extract(self.schedule.alphas, t, x.shape)
            alpha_cumprod = _extract(self.schedule.alphas_cumprod, t, x.shape)
            beta = _extract(self.schedule.betas, t, x.shape)

            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            )

            if i > 0:
                noise = torch.randn_like(x)
                posterior_var = _extract(self.schedule.posterior_variance, t, x.shape)
                x = x + torch.sqrt(posterior_var) * noise

        return x
