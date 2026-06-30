"""Convolutional beta-VAE for 1-channel DEM heightmap tiles

The model matches the project data contract:
input/output tensor shape is [B, 1, 256, 256] and values are in [-1, 1]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def _group_count(channels: int) -> int:
    return 8 if channels % 8 == 0 else 1


class ConvBlock(nn.Module):
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
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@dataclass
class BetaVAEConfig:
    in_channels: int = 1
    image_size: int = 256
    latent_dim: int = 128
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 8, 16)


class ConvBetaVAE(nn.Module):
    """Compact convolutional beta-VAE

    The architecture is intentionally simple and reproducible: strided conv
    encoder, Gaussian latent bottleneck, transposed-conv decoder with tanh
    output. It is a good research baseline for beta/KL and terrain-metric
    ablations without hiding results behind a huge model
    """

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 256,
        latent_dim: int = 128,
        base_channels: int = 32,
        channel_multipliers: tuple[int, ...] | list[int] = (1, 2, 4, 8, 16),
    ):
        super().__init__()
        channels = [base_channels * int(m) for m in channel_multipliers]

        encoder_layers: list[nn.Module] = []
        current = in_channels
        for out_channels in channels:
            encoder_layers.append(ConvBlock(current, out_channels))
            current = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            encoded = self.encoder(dummy)

        self.feature_shape = tuple(encoded.shape[1:])
        self.flatten_dim = int(np.prod(self.feature_shape))
        self.latent_dim = int(latent_dim)
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)

        self.fc_mu = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, self.latent_dim)
        self.fc_decode = nn.Linear(self.latent_dim, self.flatten_dim)

        decoder_layers: list[nn.Module] = []
        decoder_channels = list(reversed(channels))
        for in_ch, out_ch in zip(decoder_channels[:-1], decoder_channels[1:]):
            decoder_layers.append(DeconvBlock(in_ch, out_ch))
        decoder_layers.append(
            nn.ConvTranspose2d(
                decoder_channels[-1],
                in_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(z.size(0), *self.feature_shape)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

