"""Loss functions for beta-VAE terrain experiments"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    """Mean KL(q(z|x) || N(0, I)) over batch

    If free_bits > 0, each latent dimension gets a small KL allowance. This is
    useful when KL annealing alone is not enough to avoid posterior collapse
    """

    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.sum(dim=1).mean()


def gradient_components(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


def gradient_l1_loss(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """L1 error between heightmap gradients

    This loss directly targets the observed VAE failure mode: over-smoothed
    ridges and micro-relief
    """

    dx_t, dy_t = gradient_components(target)
    dx_p, dy_p = gradient_components(pred)
    return 0.5 * (F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t))


def kl_anneal_factor(epoch: int, anneal_epochs: int) -> float:
    if anneal_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, epoch / anneal_epochs))


def beta_vae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    epoch: int = 1,
    kl_anneal_epochs: int = 0,
    free_bits: float = 0.0,
    grad_loss_weight: float = 0.0,
    recon_type: str = "mse",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute total beta-VAE loss and scalar logging parts"""

    if recon_type == "mse":
        recon_loss = F.mse_loss(recon, x, reduction="mean")
    elif recon_type in {"l1", "mae"}:
        recon_loss = F.l1_loss(recon, x, reduction="mean")
    else:
        raise ValueError(f"Unknown recon_type={recon_type!r}")

    kl_raw = kl_divergence(mu, logvar, free_bits=free_bits)
    num_pixels = x.shape[1] * x.shape[2] * x.shape[3]
    kl_scaled = kl_raw / num_pixels
    kl_weight = float(beta) * kl_anneal_factor(epoch, kl_anneal_epochs)

    grad_loss = (
        gradient_l1_loss(x, recon)
        if grad_loss_weight and grad_loss_weight > 0
        else recon_loss.new_tensor(0.0)
    )
    total = recon_loss + kl_weight * kl_scaled + float(grad_loss_weight) * grad_loss

    parts = {
        "loss": float(total.detach().cpu()),
        "recon_loss": float(recon_loss.detach().cpu()),
        "kl_raw": float(kl_raw.detach().cpu()),
        "kl_scaled": float(kl_scaled.detach().cpu()),
        "kl_weight": kl_weight,
        "grad_loss": float(grad_loss.detach().cpu()),
    }
    return total, parts

