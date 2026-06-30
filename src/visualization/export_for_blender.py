"""Мост CVAE -> Blender.

Blender не знает про torch, поэтому heightmap'ы сохраняются как .npy
(float32, [H, W], значения в [-1, 1]) + JSON-манифест с масштабом высоты
по классам. Blender-скрипт читает их и строит меши.

Запуск (в окружении с torch и обученным CVAE):
    python -m src.visualization.export_for_blender \
        --ckpt outputs/cvae/best.pt --config configs/cvae.yaml \
        --out blender/heightmaps --per-class 2 --upsample 512
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TERRAIN = ["flat", "hilly", "mountain"]


def _load_cvae(ckpt: str, config: str):
    import torch
    import yaml
    from src.models.cvae import ConvCVAE

    cfg = yaml.safe_load(open(config, encoding="utf-8"))
    m = cfg.get("model", {})
    model = ConvCVAE(
        in_channels=int(m.get("in_channels", 1)),
        image_size=int(m.get("image_size", 256)),
        latent_dim=int(m.get("latent_dim", 128)),
        base_channels=int(m.get("base_channels", 32)),
        channel_multipliers=tuple(m.get("channel_multipliers", [1, 2, 4, 8, 16])),
        num_classes=int(m.get("num_classes", 3)),
        class_embed_dim=int(m.get("class_embed_dim", 16)),
        scale_embed_dim=int(m.get("scale_embed_dim", 8)),
    )
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    # наш train_cvae хранит веса под "model_state"; поддержим и "model"/голый state_dict
    if isinstance(ck, dict):
        state_dict = ck.get("model_state", ck.get("model", ck))
    else:
        state_dict = ck
    model.load_state_dict(state_dict)
    model.eval()
    # статистика нормировки elevation_range из чекпойнта
    elev = ck.get("scale_stats", {}) or ck.get("elev_norm", {}) if isinstance(ck, dict) else {}
    return model, elev


def _upsample(h: np.ndarray, size: int) -> np.ndarray:
    """Бикубический апсемплинг heightmap для гладкого меша (чистит зубчатость)."""
    if size <= h.shape[0]:
        return h
    try:
        from scipy.ndimage import zoom
        return zoom(h, size / h.shape[0], order=3)  # cubic
    except Exception:
        # запасной билинейный, если нет scipy
        from numpy import linspace, interp
        ys = linspace(0, h.shape[0] - 1, size)
        xs = linspace(0, h.shape[1] - 1, size)
        tmp = np.stack([interp(xs, np.arange(h.shape[1]), row) for row in h])
        return np.stack([interp(ys, np.arange(h.shape[0]), col) for col in tmp.T]).T


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="blender/heightmaps")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--upsample", type=int, default=512, help="0 = без апсемплинга")
    ap.add_argument("--seed", type=int, default=42)
    # масштаб высоты (метры) на класс — для корректных пропорций в Blender
    ap.add_argument("--scale-flat", type=float, default=60.0)
    ap.add_argument("--scale-hilly", type=float, default=300.0)
    ap.add_argument("--scale-mountain", type=float, default=1200.0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, elev = _load_cvae(args.ckpt, args.config)
    scales = {"flat": args.scale_flat, "hilly": args.scale_hilly, "mountain": args.scale_mountain}

    # нормированное условие масштаба: середина диапазона класса (если есть статистика)
    def scale_cond(cls_idx):
        mu = elev.get("log1p_mean", elev.get("mean", 0.0))
        sd = elev.get("log1p_std", elev.get("std", 1.0))
        med = {"flat": 50.0, "hilly": 250.0, "mountain": 900.0}[TERRAIN[cls_idx]]
        return (float(np.log1p(med)) - mu) / (sd + 1e-8)

    manifest = []
    with torch.no_grad():
        for cls_idx, cls in enumerate(TERRAIN):
            for k in range(args.per_class):
                sc = scale_cond(cls_idx)
                tile = model.sample(1, cls_idx, sc).cpu().numpy()[0, 0]  # [H,W] в [-1,1]
                if args.upsample:
                    tile = _upsample(tile, args.upsample)
                    tile = np.clip(tile, -1.0, 1.0)
                fname = f"{cls}_{k:02d}.npy"
                np.save(out / fname, tile.astype(np.float32))
                manifest.append({
                    "file": fname,
                    "terrain_type": cls,
                    "height_scale_m": scales[cls],
                    "resolution": int(tile.shape[0]),
                })
                print(f"saved {fname}  shape={tile.shape}  range=[{tile.min():.2f},{tile.max():.2f}]")

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nmanifest: {out/'manifest.json'}  ({len(manifest)} tiles)")


if __name__ == "__main__":
    main()
