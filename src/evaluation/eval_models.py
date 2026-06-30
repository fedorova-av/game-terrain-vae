"""
eval_models.py - оценка AE (E0) и Vanilla VAE (E1)

Запуск:
    python -m src.models.eval_models --model ae                       # E0
    python -m src.models.eval_models --model vae                      # E1
    python -m src.models.eval_models --model vae --split test
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import batch_metrics, aggregate_sample_metrics

def load_split(processed_dir, metadata_csv, split):
    x = torch.load(Path(processed_dir) / f"{split}.pt", map_location="cpu").float()
    meta = pd.read_csv(metadata_csv)
    meta = meta[meta["split"] == split].reset_index(drop=True)
    assert len(meta) == len(x), (
        f"Рассинхрон: {len(x)} плиток в {split}.pt, но {len(meta)} строк в metadata "
        f"для split={split}. Порядок должен совпадать (контракт данных)."
    )
    terrain = meta["terrain_type"].to_numpy()
    scale = ((meta["orig_max"] - meta["orig_min"]) / 2.0).to_numpy()  # метров на 1 норм-единицу
    return x, terrain, torch.tensor(scale, dtype=torch.float32)


@torch.no_grad()
def reconstruct(model, xb, model_type):
    """Единый интерфейс реконструкции."""
    if model_type in ("vae", "vae_unet"):
        mu, _ = model.encoder(xb)
        return model.decoder(mu)
    rb, _ = model(xb)
    return rb


@torch.no_grad()
def per_tile_l1(model, x, device, model_type, bs=32):
    errs = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(device)
        rb = reconstruct(model, xb, model_type)
        errs.append((rb - xb).abs().mean(dim=(1, 2, 3)).cpu())
    return torch.cat(errs)  # [N], L1 в норм-единицах по каждой плитке

LABEL_TO_IDX = {"flat": 0, "hilly": 1, "mountain": 2}
IDX_TO_LABEL = {0: "flat", 1: "hilly", 2: "mountain"}


@torch.no_grad()
def terrain_metrics(model, x, scale, terrain, device, model_type, bs=32):
    labels_all = torch.tensor([LABEL_TO_IDX[t] for t in terrain])
    metric_batches, labels_batches = [], []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(device)
        rb = reconstruct(model, xb, model_type)
        er = 2.0 * scale[i:i + bs].to(device)
        mb = batch_metrics(xb, rb, elevation_range=er)
        metric_batches.append({k: v.detach().cpu() for k, v in mb.items()})
        labels_batches.append(labels_all[i:i + bs])
    return aggregate_sample_metrics(metric_batches, labels_batches, IDX_TO_LABEL)



@torch.no_grad()
def per_tile_kl(model, x, device, bs=32):
    kls = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(device)
        mu, logvar = model.encoder(xb)
        logvar = logvar.clamp(-10, 10)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kls.append(kl.cpu())
    return torch.cat(kls)


def make_recon_figure(model, x, terrain, device, model_type, out_path, per_class, title):
    import matplotlib.pyplot as plt

    idx = []
    for cls in ("flat", "hilly", "mountain"):
        found = np.where(terrain == cls)[0][:per_class]
        idx.extend(found.tolist())
    if not idx:
        return None

    with torch.no_grad():
        xb = x[idx].to(device)
        rb = reconstruct(model, xb, model_type)
    xb, rb = xb.cpu(), rb.cpu()

    rows = len(idx)
    fig, axes = plt.subplots(rows, 3, figsize=(7.5, 2.5 * rows))
    if rows == 1:
        axes = axes[None, :]
    for r, j in enumerate(idx):
        inp = xb[r, 0].numpy()
        rec = rb[r, 0].numpy()
        err = np.abs(inp - rec)
        axes[r, 0].imshow(inp, cmap="terrain"); axes[r, 0].set_ylabel(terrain[j], fontsize=10)
        axes[r, 1].imshow(rec, cmap="terrain")
        axes[r, 2].imshow(err, cmap="magma", vmin=0, vmax=0.5)
        if r == 0:
            axes[r, 0].set_title("вход"); axes[r, 1].set_title("реконструкция"); axes[r, 2].set_title("|ошибка|")
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


@torch.no_grad()
def make_prior_figure(model, n, latent_dim, device, out_path):
    """Генерация из приора N(0,I)"""
    import matplotlib.pyplot as plt

    z = torch.randn(n, latent_dim, device=device)
    samples = model.decoder(z).cpu()
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for k in range(rows * cols):
        ax = axes[k // cols, k % cols]
        ax.set_xticks([]); ax.set_yticks([])
        if k < n:
            ax.imshow(samples[k, 0].numpy(), cmap="terrain")
        else:
            ax.axis("off")
    fig.suptitle("E1 (VAE): генерация из приора N(0,I)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_model(model_type, cargs, device):
    latent_dim = cargs.get("latent_dim", 128)
    base = cargs.get("base", 32)
    if model_type == "vae":
        from src.models.vae import VAE
        model = VAE(latent_dim, base)
    elif model_type == "vae_unet":
        from src.models.vae_unet import UNetVAE
        model = UNetVAE(latent_dim, base)
    else:
        from src.models.ae import Autoencoder
        model = Autoencoder(latent_dim, base)
    return model.to(device), latent_dim


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = args.ckpt or ("checkpoints/vae_unet/vae_unet_best.pt" if args.model == "vae_unet" else
                              "checkpoints/vae_e1_best.pt" if args.model == "vae" else
                              "checkpoints/ae_e0_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cargs = ckpt.get("args", {})
    model, latent_dim = build_model(args.model, cargs, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tag = args.tag or {"vae":"vae_e1", "vae_unet": "vae_unet", "ae":"ae_e0",}[args.model]
    x, terrain, scale = load_split(args.processed_dir, args.metadata_csv, args.split)
    l1_norm = per_tile_l1(model, x, device, args.model, args.batch_size)  # [N]
    l1_m = l1_norm * scale

    report = {
        "model": args.model,
        "split": args.split,
        "n": int(len(x)),
        "ckpt": str(ckpt_path),
        "L1_norm_overall": round(float(l1_norm.mean()), 5),
        "L1_m_overall": round(float(l1_m.mean()), 3),
        "per_class": {},
    }

    kl = per_tile_kl(model, x, device, args.batch_size) if args.model == "vae" else None
    if kl is not None:
        report["KL_overall"] = round(float(kl.mean()), 4)

    for cls in ("flat", "hilly", "mountain"):
        mask = torch.from_numpy(terrain == cls)
        if mask.sum() == 0:
            continue
        entry = {
            "n": int(mask.sum()),
            "L1_norm": round(float(l1_norm[mask].mean()), 5),
            "L1_m": round(float(l1_m[mask].mean()), 3),
        }
        if kl is not None:
            entry["KL"] = round(float(kl[mask].mean()), 4)
        report["per_class"][cls] = entry

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{tag}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        
    terr = terrain_metrics(model, x, scale, terrain, device, args.model, args.batch_size)
    terr = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in terr.items()}
    with open(out_dir / f"{tag}_terrain_metrics.json", "w", encoding="utf-8") as f:
        json.dump(terr, f, ensure_ascii=False, indent=2)
    print(f"Общие метрики: {out_dir / f'{tag}_terrain_metrics.json'}")

    title = f"{tag.upper()}: реконструкции по классам"
    fig_path = make_recon_figure(model, x, terrain, device, args.model,
                                 out_dir / f"{tag}_recon.png", args.per_class, title)
    prior_path = None
    if args.model in ("vae", "vae_unet"):
        prior_path = make_prior_figure(model, args.n_samples, latent_dim, device,
                                       out_dir / f"{tag}_prior.png")

    print(f"\n{tag.upper()} — split={args.split}, n={report['n']}")
    line = f"  overall: L1_norm {report['L1_norm_overall']:.4f} | L1_m {report['L1_m_overall']:.2f} м"
    if kl is not None:
        line += f" | KL {report['KL_overall']:.3f}"
    print(line)
    for cls, d in report["per_class"].items():
        line = f"  {cls:9s} (n={d['n']:4d}): L1_norm {d['L1_norm']:.4f} | L1_m {d['L1_m']:.2f} м"
        if "KL" in d:
            line += f" | KL {d['KL']:.3f}"
        print(line)
    print(f"\nМетрики: {out_dir / f'{tag}_metrics.json'}")
    if fig_path:
        print(f"Фигура:  {fig_path}")
    if prior_path:
        print(f"Приор:   {prior_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Оценка AE (E0) и Vanilla VAE (E1)")
    p.add_argument("--model", default="ae", choices=["ae", "vae", "vae_unet"])
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--metadata-csv", default="data/metadata.csv")
    p.add_argument("--ckpt", default=None, help="по умолчанию checkpoints/ae_e0_best.pt или vae_e1_best.pt")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default=None, help="префикс выходных файлов (по умолчанию ae_e0 / vae_e1)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--per-class", type=int, default=2, help="сколько примеров класса в фигуре реконструкции")
    p.add_argument("--n-samples", type=int, default=8, help="сколько плиток генерировать из приора (VAE)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())