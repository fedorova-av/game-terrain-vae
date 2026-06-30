"""
eval_terrain_fusion.py — оценка качества TerraFusion.

Считает метрики на сгенерированных плитках и сохраняет metrics.json
в формате, совместимом с compare_experiments.py.

Стратегия оценки:
  - Генерируем N плиток через DDIM
  - Сравниваем их статистики с реальными плитками val-split
  - Метрики: gradient_mae, roughness, slope_diff — считаем как
    разницу между распределениями (mean реальных vs mean сгенерированных)
  - rmse считаем через ближайшего соседа в латентном пространстве
    (каждому сгенерированному z находим ближайший реальный, меряем
    расстояние в пикселях)

Запуск:
    python -m src.evaluation.eval_terrain_fusion ^
        --vae-ckpt  checkpoints/vae_latent256/vae_e1_best.pt ^
        --diff-ckpt checkpoints/terrain_fusion/diff_best.pt ^
        --out       results/terrain_fusion
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from src.data.dataset import HeightmapDataset
    from src.models.vae import VAE
    from src.models.terrain_fusion import (
        LatentDenoiser, make_noise_schedule, ddim_sample
    )
    from src.evaluation.metrics import (
        batch_metrics, aggregate_sample_metrics
    )
except Exception as e:
    print(f"Import warning: {e}")
    HeightmapDataset = VAE = LatentDenoiser = None


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

@torch.no_grad()
def load_vae(vae_ckpt_path: str, device: str):
    ckpt   = torch.load(vae_ckpt_path, map_location=device, weights_only=False)
    cargs  = ckpt.get("args", {})
    latent_dim = cargs.get("latent_dim", 256)
    base       = cargs.get("base", 32)
    vae = VAE(latent_dim, base).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae, latent_dim


@torch.no_grad()
def load_denoiser(diff_ckpt_path: str, latent_dim: int, device: str):
    ckpt  = torch.load(diff_ckpt_path, map_location=device, weights_only=False)
    dargs = ckpt.get("args", {})
    denoiser = LatentDenoiser(
        latent_dim = latent_dim,
        hidden_dim = dargs.get("hidden_dim", 512),
        n_blocks   = dargs.get("n_blocks",   4),
        time_dim   = dargs.get("time_dim",   256),
    ).to(device)
    denoiser.load_state_dict(ckpt["model"])
    denoiser.eval()
    return denoiser


@torch.no_grad()
def get_real_tiles(processed_dir: str, metadata_csv: str,
                   split: str, device: str, bs: int = 64):
    """Загружаем реальные плитки val/test split."""
    ds = HeightmapDataset(
        processed_dir=processed_dir,
        metadata_csv=metadata_csv,
        split=split,
        return_label=False,
    )
    loader = DataLoader(ds, batch_size=bs, shuffle=False,
                        num_workers=0, pin_memory=(device == "cuda"))
    tiles = []
    for x in loader:
        tiles.append(x.to(device))
    return torch.cat(tiles, dim=0)   # [N, 1, 256, 256]


@torch.no_grad()
def generate_tiles(vae, denoiser, schedule, n: int,
                   latent_dim: int, device: str,
                   ddim_steps: int = 50, eta: float = 0.0,
                   bs: int = 64) -> torch.Tensor:
    """Генерируем n плиток батчами."""
    tiles = []
    generated = 0
    while generated < n:
        curr = min(bs, n - generated)
        z0 = ddim_sample(denoiser, schedule, curr, latent_dim,
                         device, ddim_steps=ddim_steps, eta=eta)
        x  = vae.decoder(z0)   # [curr, 1, 256, 256]
        tiles.append(x.cpu())
        generated += curr
    return torch.cat(tiles, dim=0)   # [n, 1, 256, 256]


# ---------------------------------------------------------------------------
# Метрики распределения
# ---------------------------------------------------------------------------

def distribution_metrics(real: torch.Tensor,
                          fake: torch.Tensor,
                          bs: int = 64) -> dict:
    """
    Считаем метрики качества рельефа на сгенерированных плитках.

    Подход: вычисляем gradient_mae, slope_diff, roughness для каждой
    плитки отдельно (mean по плиткам), затем сравниваем распределения
    real vs fake.

    rmse — среднеквадратичная ошибка между сгенерированной плиткой
    и её ближайшим соседом среди реальных (по L2 в пространстве пикселей,
    усреднённая по батчу из 512 плиток для скорости).
    """
    from src.evaluation.metrics import (
        gradient_components, slope_magnitude, roughness_value
    )

    def per_tile_stats(tiles: torch.Tensor):
        gx, gy = gradient_components(tiles)
        gmae = 0.5 * (gx.abs().mean(dim=(1,2,3)) + gy.abs().mean(dim=(1,2,3)))
        slope = slope_magnitude(tiles).mean(dim=(1,2,3))
        rough = roughness_value(tiles)
        return {
            "gradient_mae": gmae,
            "slope":        slope,
            "roughness":    rough,
        }

    # Статистики по реальным и сгенерированным
    real_stats = per_tile_stats(real.cpu())
    fake_stats = per_tile_stats(fake.cpu())

    # gradient_mae: разница средних (как slope_diff в reconstruction eval)
    gradient_mae   = float((fake_stats["gradient_mae"].mean()
                            - real_stats["gradient_mae"].mean()).abs())
    slope_diff     = float((fake_stats["slope"].mean()
                            - real_stats["slope"].mean()).abs())
    roughness_real  = float(real_stats["roughness"].mean())
    roughness_recon = float(fake_stats["roughness"].mean())
    roughness_diff  = abs(roughness_recon - roughness_real)

    # rmse: ближайший сосед на подвыборке
    n_nn = min(512, len(real), len(fake))
    real_sub = real[:n_nn].view(n_nn, -1).float()   # [n_nn, 256*256]
    fake_sub = fake[:n_nn].view(n_nn, -1).float()

    # Считаем попарные расстояния батчами (экономим память)
    nn_rmse = []
    sub_bs = 64
    for i in range(0, n_nn, sub_bs):
        fb = fake_sub[i:i+sub_bs].unsqueeze(1)          # [sub_bs, 1, D]
        rb = real_sub.unsqueeze(0)                       # [1, n_nn, D]
        dists = ((fb - rb) ** 2).mean(dim=2).sqrt()     # [sub_bs, n_nn]
        nn_rmse.append(dists.min(dim=1).values)
    rmse = float(torch.cat(nn_rmse).mean())

    # mae: аналогично через L1
    nn_mae = []
    for i in range(0, n_nn, sub_bs):
        fb = fake_sub[i:i+sub_bs].unsqueeze(1)
        rb = real_sub.unsqueeze(0)
        dists = (fb - rb).abs().mean(dim=2)
        nn_mae.append(dists.min(dim=1).values)
    mae = float(torch.cat(nn_mae).mean())

    return {
        "mae":              round(mae,            5),
        "rmse":             round(rmse,           5),
        "gradient_mae":     round(gradient_mae,   5),
        "slope_diff":       round(slope_diff,     5),
        "roughness_real":   round(roughness_real,  5),
        "roughness_recon":  round(roughness_recon, 5),
        "roughness_diff":   round(roughness_diff,  5),
        # Для compare_experiments.py нужны эти поля:
        "eval_type":        "generative",
        "n_generated":      len(fake),
        "n_real":           len(real),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    # Загружаем модели
    vae, latent_dim = load_vae(args.vae_ckpt, device)
    denoiser        = load_denoiser(args.diff_ckpt, latent_dim, device)
    schedule        = make_noise_schedule(args.T, device=device)

    print(f"VAE latent_dim={latent_dim} | DDIM steps={args.ddim_steps}")

    # Реальные плитки
    print(f"Загружаем реальные плитки ({args.split}) ...")
    real = get_real_tiles(args.processed_dir, args.metadata_csv,
                          args.split, device="cpu")
    print(f"  реальных: {len(real)}")

    # Генерируем
    print(f"Генерируем {args.n_samples} плиток ...")
    fake = generate_tiles(vae, denoiser, schedule,
                          n=args.n_samples,
                          latent_dim=latent_dim,
                          device=device,
                          ddim_steps=args.ddim_steps,
                          eta=args.eta)
    print(f"  сгенерировано: {fake.shape}")

    # Считаем метрики
    print("Считаем метрики ...")
    metrics = distribution_metrics(real, fake)
    metrics["name"] = args.name

    # Выводим
    print(f"\n{'='*50}")
    print(f"TerraFusion ({args.name})")
    print(f"  mae:           {metrics['mae']:.5f}")
    print(f"  rmse:          {metrics['rmse']:.5f}")
    print(f"  gradient_mae:  {metrics['gradient_mae']:.5f}")
    print(f"  slope_diff:    {metrics['slope_diff']:.5f}")
    print(f"  roughness_diff:{metrics['roughness_diff']:.5f}")
    print(f"{'='*50}")

    # Сохраняем в формате compare_experiments.py:
    # outputs/terrain_fusion/metrics.json
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Метрики: {out_dir / 'metrics.json'}")

    # Сохраняем фигуру сгенерированных плиток
    import matplotlib.pyplot as plt
    cols = min(args.n_samples, 4)
    rows = (args.n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.8*cols, 2.8*rows))
    axes = np.array(axes).reshape(rows, cols)
    for k in range(rows * cols):
        ax = axes[k // cols, k % cols]
        ax.set_xticks([]); ax.set_yticks([])
        if k < args.n_samples:
            ax.imshow(fake[k, 0].numpy(), cmap="terrain")
        else:
            ax.axis("off")
    fig.suptitle(f"TerraFusion: {args.ddim_steps} DDIM шагов | val_loss={args.diff_val_loss:.4f}" 
                 if args.diff_val_loss else f"TerraFusion: {args.ddim_steps} DDIM шагов")
    fig.tight_layout()
    fig_path = out_dir / "samples.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Фигура:  {fig_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Eval TerraFusion")
    p.add_argument("--vae-ckpt",      default="checkpoints/vae_latent256/vae_e1_best.pt")
    p.add_argument("--diff-ckpt",     default="checkpoints/terrain_fusion/diff_best.pt")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--metadata-csv",  default="data/metadata.csv")
    p.add_argument("--split",         default="val")
    p.add_argument("--out",           default="results/terrain_fusion")
    p.add_argument("--name",          default="terrain_fusion")
    p.add_argument("--T",             type=int,   default=1000)
    p.add_argument("--ddim-steps",    type=int,   default=50)
    p.add_argument("--eta",           type=float, default=0.0)
    p.add_argument("--n-samples",     type=int,   default=1000,
                   help="сколько плиток сгенерировать для оценки")
    p.add_argument("--diff-val-loss", type=float, default=None,
                   help="val loss денойзера (для подписи на фигуре)")
    return p.parse_args()


if __name__ == "__main__":
    import numpy as np
    main(parse_args())
