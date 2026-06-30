"""
eval_finetuned_decoder.py — оценка дообученного декодера VAE.

Загружает VAE энкодер из оригинального чекпойнта + дообученный декодер,
считает полный набор метрик совместимых с compare_experiments.py.

Запуск:
    python -m src.evaluation.eval_finetuned_decoder --mode recon
    python -m src.evaluation.eval_finetuned_decoder --mode diff
    python -m src.evaluation.eval_finetuned_decoder --mode both
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
    from src.evaluation.metrics import batch_metrics, aggregate_sample_metrics
except Exception as e:
    print(f"Import warning: {e}")


LABEL_TO_IDX = {"flat": 0, "hilly": 1, "mountain": 2}
IDX_TO_LABEL = {0: "flat", 1: "hilly", 2: "mountain"}


# ---------------------------------------------------------------------------
# Загрузка модели
# ---------------------------------------------------------------------------

def load_model(vae_ckpt: str, decoder_ckpt: str, device: str):
    """
    Загружает VAE с оригинальным энкодером и дообученным декодером.
    """
    # Оригинальный VAE (энкодер + декодер как основа)
    ckpt = torch.load(vae_ckpt, map_location=device, weights_only=False)
    cargs = ckpt.get("args", {})
    latent_dim = cargs.get("latent_dim", 256)
    base = cargs.get("base", 32)
    vae = VAE(latent_dim, base).to(device)
    vae.load_state_dict(ckpt["model"])

    # Заменяем декодер дообученным
    dec_ckpt = torch.load(decoder_ckpt, map_location=device, weights_only=False)
    vae.decoder.load_state_dict(dec_ckpt["decoder"])
    vae.eval()

    print(f"VAE загружен | latent_dim={latent_dim}")
    print(f"Декодер: {decoder_ckpt} "
          f"(epoch={dec_ckpt.get('epoch','?')}, "
          f"roughness={dec_ckpt.get('roughness', '?'):.5f}, "
          f"val_l1={dec_ckpt.get('val_l1', '?'):.4f})")
    return vae, latent_dim


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(vae, latent_dim, args, device):
    import pandas as pd

    # Загружаем split
    ds = HeightmapDataset(
        processed_dir=args.processed_dir,
        metadata_csv=args.metadata_csv,
        split=args.split,
        return_label=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    import pandas as pd_
    meta = pd_.read_csv(args.metadata_csv)
    meta = meta[meta["split"] == args.split].reset_index(drop=True)
    terrain = meta["terrain_type"].to_numpy()
    scale = ((meta["orig_max"] - meta["orig_min"]) / 2.0).to_numpy()
    scale_t = torch.tensor(scale, dtype=torch.float32)

    labels_all = torch.tensor([LABEL_TO_IDX[t] for t in terrain])

    metric_batches, labels_batches = [], []
    l1_norm_all, l1_m_all, kl_all = [], [], []

    offset = 0
    for x in loader:
        x = x.to(device)
        bs = x.size(0)

        # Реконструкция через энкодер → дообученный декодер
        mu, logvar = vae.encoder(x)
        recon = vae.decoder(mu)

        # KL
        logvar_c = logvar.clamp(-10, 10)
        kl = -0.5 * torch.sum(
            1 + logvar_c - mu.pow(2) - logvar_c.exp(), dim=1
        )
        kl_all.append(kl.cpu())

        # L1 нормализованный
        l1_norm = (recon - x).abs().mean(dim=(1, 2, 3)).cpu()
        l1_norm_all.append(l1_norm)

        # L1 в метрах
        sc = scale_t[offset:offset + bs]
        l1_m_all.append(l1_norm * sc)

        # Terrain метрики
        er = 2.0 * sc.to(device)
        mb = batch_metrics(x, recon, elevation_range=er)
        metric_batches.append({k: v.detach().cpu() for k, v in mb.items()})
        labels_batches.append(labels_all[offset:offset + bs])
        offset += bs

    l1_norm_t = torch.cat(l1_norm_all)
    l1_m_t = torch.cat(l1_m_all)
    kl_t = torch.cat(kl_all)

    # Агрегируем terrain метрики
    terr = aggregate_sample_metrics(metric_batches, labels_batches, IDX_TO_LABEL)
    terr = {k: round(v, 5) if isinstance(v, float) else v for k, v in terr.items()}

    # Добавляем L1 / KL поверх terrain метрик
    report = dict(terr)
    report["L1_norm_overall"] = round(float(l1_norm_t.mean()), 5)
    report["L1_m_overall"] = round(float(l1_m_t.mean()), 3)
    report["KL_overall"] = round(float(kl_t.mean()), 3)
    report["n"] = len(l1_norm_t)
    report["mode"] = args.mode
    report["split"] = args.split

    # Per-class L1 / KL
    for cls in ("flat", "hilly", "mountain"):
        mask = torch.from_numpy(terrain == cls)
        if mask.sum() == 0:
            continue
        report[f"{cls}_L1_norm"] = round(float(l1_norm_t[mask].mean()), 5)
        report[f"{cls}_L1_m"] = round(float(l1_m_t[mask].mean()), 3)
        report[f"{cls}_KL"] = round(float(kl_t[mask].mean()), 3)

    return report


# ---------------------------------------------------------------------------
# Фигура реконструкций
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_recon_figure(vae, processed_dir, metadata_csv, split,
                      device, per_class, out_path, title):
    import matplotlib.pyplot as plt

    ds = HeightmapDataset(
        processed_dir=processed_dir,
        metadata_csv=metadata_csv,
        split=split, return_label=False,
    )
    import pandas as pd_
    meta = pd_.read_csv(metadata_csv)
    meta = meta[meta["split"] == split].reset_index(drop=True)
    terrain = meta["terrain_type"].to_numpy()

    idx = []
    for cls in ("flat", "hilly", "mountain"):
        found = np.where(terrain == cls)[0][:per_class]
        idx.extend(found.tolist())

    xs = torch.stack([ds[i] for i in idx]).to(device)
    mu, _ = vae.encoder(xs)
    recons = vae.decoder(mu).cpu()
    xs = xs.cpu()

    rows = len(idx)
    fig, axes = plt.subplots(rows, 3, figsize=(7.5, 2.5 * rows))
    if rows == 1:
        axes = axes[None, :]
    for r, j in enumerate(idx):
        inp = xs[r, 0].numpy()
        rec = recons[r, 0].numpy()
        err = np.abs(inp - rec)
        axes[r, 0].imshow(inp, cmap="terrain")
        axes[r, 0].set_ylabel(terrain[j], fontsize=10)
        axes[r, 1].imshow(rec, cmap="terrain")
        axes[r, 2].imshow(err, cmap="magma", vmin=0, vmax=0.5)
        if r == 0:
            axes[r, 0].set_title("вход")
            axes[r, 1].set_title("реконструкция")
            axes[r, 2].set_title("|ошибка|")
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    decoder_ckpt = (args.decoder_ckpt or
                    f"checkpoints/decoder_{args.mode}/decoder_{args.mode}_best.pt")
    tag = args.tag or f"decoder_{args.mode}"

    vae, latent_dim = load_model(args.vae_ckpt, decoder_ckpt, device)

    print(f"\nОцениваем {tag} на split={args.split} ...")
    report = run_eval(vae, latent_dim, args, device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем terrain_metrics.json (полный)
    with open(out_dir / f"{tag}_terrain_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Сохраняем metrics.json (для compare_experiments.py)
    compare_metrics = {
        "name":          tag,
        "mae":           report.get("mae"),
        "rmse":          report.get("rmse"),
        "gradient_mae":  report.get("gradient_mae"),
        "slope_diff":    report.get("slope_diff"),
        "roughness_real":  report.get("roughness_real"),
        "roughness_recon": report.get("roughness_recon"),
        "roughness_diff":  report.get("roughness_diff"),
        "L1_norm":       report.get("L1_norm_overall"),
        "L1_m":          report.get("L1_m_overall"),
        "KL":            report.get("KL_overall"),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(compare_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # Вывод в консоль
    print(f"\n{'='*55}")
    print(f"{tag.upper()} | split={args.split} | n={report['n']}")
    print(f"  mae:            {report.get('mae', '?'):.5f}")
    print(f"  rmse:           {report.get('rmse', '?'):.5f}")
    print(f"  gradient_mae:   {report.get('gradient_mae', '?'):.5f}")
    print(f"  roughness_real: {report.get('roughness_real', '?'):.5f}")
    print(f"  roughness_recon:{report.get('roughness_recon', '?'):.5f}")
    print(f"  roughness_diff: {report.get('roughness_diff', '?'):.5f}")
    print(f"  L1_norm:        {report.get('L1_norm_overall', '?'):.5f}")
    print(f"  KL:             {report.get('KL_overall', '?'):.3f}")
    print(f"{'='*55}")

    # Фигура реконструкций
    title = f"{tag}: реконструкции по классам"
    make_recon_figure(
        vae, args.processed_dir, args.metadata_csv,
        args.split, device, args.per_class,
        out_dir / f"{tag}_recon.png", title
    )
    print(f"Метрики: {out_dir / f'{tag}_terrain_metrics.json'}")
    print(f"Фигура:  {out_dir / f'{tag}_recon.png'}")


def parse_args():
    p = argparse.ArgumentParser(description="Eval дообученного декодера VAE")
    p.add_argument("--mode",          default="recon",
                   choices=["recon", "diff", "both"])
    p.add_argument("--vae-ckpt",      default="checkpoints/vae_latent256/vae_e1_best.pt")
    p.add_argument("--decoder-ckpt",  default=None,
                   help="путь к чекпойнту декодера (по умолчанию: "
                        "checkpoints/decoder_{mode}/decoder_{mode}_best.pt)")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--metadata-csv",  default="data/metadata.csv")
    p.add_argument("--split",         default="val")
    p.add_argument("--out",           default="results")
    p.add_argument("--tag",           default=None)
    p.add_argument("--batch-size",    type=int, default=32)
    p.add_argument("--per-class",     type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
