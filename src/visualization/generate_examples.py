"""Generate final heightmap examples from a trained beta-VAE checkpoint"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LightSource

from src.models import ConvBetaVAE


def to_heightmap(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.float32)


def hillshade(heightmap: np.ndarray) -> np.ndarray:
    return LightSource(azdeg=315, altdeg=45).hillshade(heightmap, vert_exag=1.5)


def load_model(checkpoint_path: str | Path, device: torch.device) -> ConvBetaVAE:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", {})
    model = ConvBetaVAE(
        in_channels=int(model_cfg.get("in_channels", 1)),
        image_size=int(model_cfg.get("image_size", checkpoint.get("input_size", 256))),
        latent_dim=int(model_cfg.get("latent_dim", checkpoint.get("latent_dim", 128))),
        base_channels=int(model_cfg.get("base_channels", 32)),
        channel_multipliers=tuple(model_cfg.get("channel_multipliers", [1, 2, 4, 8, 16])),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def save_16bit_png(heightmap: np.ndarray, path: Path) -> None:
    from PIL import Image

    arr = np.clip((heightmap + 1.0) / 2.0, 0.0, 1.0)
    arr16 = (arr * 65535).round().astype(np.uint16)
    Image.fromarray(arr16, mode="I;16").save(path)


def plot_showcase(samples: torch.Tensor, out_path: Path, title: str) -> None:
    n = len(samples)
    fig = plt.figure(figsize=(12, 3.8 * n))
    for i in range(n):
        h = to_heightmap(samples[i])
        ax1 = fig.add_subplot(n, 3, 3 * i + 1)
        ax1.imshow(h, cmap="terrain", vmin=-1, vmax=1)
        ax1.set_title(f"generated #{i + 1}: heightmap")
        ax1.axis("off")

        ax2 = fig.add_subplot(n, 3, 3 * i + 2)
        ax2.imshow(hillshade(h), cmap="gray")
        ax2.set_title("hillshade")
        ax2.axis("off")

        ax3 = fig.add_subplot(n, 3, 3 * i + 3, projection="3d")
        stride = max(1, h.shape[0] // 64)
        hs = h[::stride, ::stride]
        yy, xx = np.mgrid[0 : hs.shape[0], 0 : hs.shape[1]]
        ax3.plot_surface(xx, yy, hs, cmap="terrain", linewidth=0, antialiased=True)
        ax3.set_title("3D surface")
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_zticks([])
        ax3.view_init(elev=45, azim=-60)

    fig.suptitle(title, y=1.005, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="figures/final_generation_showcase")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    z = args.temperature * torch.randn(args.n, model.latent_dim, generator=generator, device=device)
    samples = model.decode(z).clamp(-1, 1).detach().cpu()

    plot_showcase(
        samples,
        out_dir / "final_generated_examples_heightmap_hillshade_3d.png",
        f"Final generated terrain examples, tau={args.temperature:.2f}",
    )
    for i, sample in enumerate(samples):
        h = to_heightmap(sample)
        np.save(out_dir / f"final_generated_{i:02d}.npy", h)
        save_16bit_png(h, out_dir / f"final_generated_{i:02d}_16bit.png")
    print(f"saved generated examples to {out_dir}")


if __name__ == "__main__":
    main()

