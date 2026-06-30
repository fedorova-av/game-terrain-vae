"""Conditional convolutional VAE for 1-channel DEM heightmap tiles.

`ConvCVAE` extends `ConvBetaVAE` with terrain-class and elevation-scale
conditioning while keeping the same convolutional backbone, Gaussian bottleneck,
and final `Tanh` output. This makes beta-VAE and CVAE comparisons mostly about
the effect of conditioning rather than a larger model capacity.

Data contract:
    input/output shape is `[B, 1, 256, 256]`, with values in `[-1, 1]`.

Condition vector:
    cond = [class_embed(terrain_type) || scale_proj(elevation_range_norm)]

The terrain type is categorical (`flat`, `hilly`, `mountain`), while
`elevation_range_norm` preserves absolute scale information that per-patch
min-max normalization removes from the image tensor.

The encoder receives condition channels broadcast over H x W, and the decoder
receives the condition concatenated to latent vector `z`. The forward signature
matches the beta-VAE interface: `forward(x, label, scale) -> (recon, mu, logvar)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


def _group_count(channels: int) -> int:
    return 8 if channels % 8 == 0 else 1


class ConvBlock(nn.Module):
    """Downsampling блок: Conv stride2 + GroupNorm + SiLU (как в beta_vae.py)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """Upsampling блок: ConvTranspose stride2 + GroupNorm + SiLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@dataclass
class CVAEConfig:
    in_channels: int = 1
    image_size: int = 256
    latent_dim: int = 128
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 8, 16)
    num_classes: int = 3
    class_embed_dim: int = 16
    scale_embed_dim: int = 8


class ConvCVAE(nn.Module):
    """Conditional convolutional VAE.

    Параметры повторяют `ConvBetaVAE` плюс три про условие:
      num_classes      — число классов рельефа (3: flat/hilly/mountain);
      class_embed_dim  — размер embedding класса;
      scale_embed_dim  — размер проекции непрерывного масштаба.
    """

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 256,
        latent_dim: int = 128,
        base_channels: int = 32,
        channel_multipliers: tuple[int, ...] | list[int] = (1, 2, 4, 8, 16),
        num_classes: int = 3,
        class_embed_dim: int = 16,
        scale_embed_dim: int = 8,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.class_embed_dim = int(class_embed_dim)
        self.scale_embed_dim = int(scale_embed_dim)
        self.cond_dim = self.class_embed_dim + self.scale_embed_dim

        # Условие: класс (embedding) + непрерывный масштаб (linear проекция скаляра).
        self.class_embed = nn.Embedding(self.num_classes, self.class_embed_dim)
        self.scale_proj = nn.Linear(1, self.scale_embed_dim)

        channels = [base_channels * int(m) for m in channel_multipliers]

        # Энкодер видит высоту + cond_dim доп. каналов условия.
        encoder_layers: list[nn.Module] = []
        current = in_channels + self.cond_dim
        for out_channels in channels:
            encoder_layers.append(ConvBlock(current, out_channels))
            current = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        # Выводим форму бутылочного горлышка прогоном нулей (как в beta_vae.py).
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels + self.cond_dim, image_size, image_size)
            encoded = self.encoder(dummy)
        self.feature_shape = tuple(encoded.shape[1:])
        self.flatten_dim = int(np.prod(self.feature_shape))
        self.latent_dim = int(latent_dim)
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)

        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.latent_dim)
        # Декодер видит z + условие — это главный путь управляемой генерации.
        self.fc_decode = nn.Linear(self.latent_dim + self.cond_dim, self.flatten_dim)

        decoder_layers: list[nn.Module] = []
        decoder_channels = list(reversed(channels))
        for in_ch, out_ch in zip(decoder_channels[:-1], decoder_channels[1:]):
            decoder_layers.append(DeconvBlock(in_ch, out_ch))
        decoder_layers.append(
            nn.ConvTranspose2d(decoder_channels[-1], in_channels, kernel_size=4, stride=2, padding=1)
        )
        decoder_layers.append(nn.Tanh())  # выход в [-1, 1] по контракту данных
        self.decoder = nn.Sequential(*decoder_layers)

    # --- условие --------------------------------------------------------------
    def cond_vector(self, label: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """label: [B] long; scale: [B] float (нормированный) -> [B, cond_dim]."""
        emb = self.class_embed(label)
        sc = self.scale_proj(scale.float().view(-1, 1))
        return torch.cat([emb, sc], dim=1)

    # --- энкодер / латент / декодер -------------------------------------------
    def encode(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cond_map = cond[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
        h = self.encoder(torch.cat([x, cond_map], dim=1)).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(torch.cat([z, cond], dim=1))
        h = h.view(z.size(0), *self.feature_shape)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor, label: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = self.cond_vector(label, scale)
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar

    # --- генерация ------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        n: int,
        label: int | torch.Tensor,
        scale: float | torch.Tensor,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Сгенерировать n плиток для заданного класса и масштаба.

        label — int или [n] long; scale — float или [n] float (нормированный
        elevation_range, та же нормировка, что при обучении).
        """
        if isinstance(label, int):
            label = torch.full((n,), label, dtype=torch.long, device=device)
        if isinstance(scale, (int, float)):
            scale = torch.full((n,), float(scale), device=device)
        cond = self.cond_vector(label.to(device), scale.to(device))
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z, cond).clamp(-1, 1)
