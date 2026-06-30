"""Generate synthetic data that follows the project data contract.

The dummy dataset is useful for smoke tests before the full Major TOM terrain
dataset is available locally. Tensor shapes, metadata columns, and value ranges
match the real preprocessing pipeline.

These are not final training data: generated heightmaps are smoothed random
fields, not real terrain. The goal is interface validation rather than realism.

Example:
    python -m src.data.make_dummy_data --n 300 --out data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

TERRAIN_TYPES = ["flat", "hilly", "mountain"]
# Synthetic terrain amplitudes in meters for three distinguishable modes.
TERRAIN_AMPLITUDE_M = {"flat": 30.0, "hilly": 300.0, "mountain": 1500.0}


def _smooth_field(size: int, scale: int, rng: np.random.Generator) -> np.ndarray:
    """Сглаженное случайное поле: маленький шум, растянутый до size×size."""
    low = rng.standard_normal((scale, scale)).astype(np.float32)
    t = torch.from_numpy(low)[None, None]
    up = F.interpolate(t, size=(size, size), mode="bicubic", align_corners=False)
    field = up[0, 0].numpy()
    # нормируем поле в [0, 1] по форме
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return field


def _slope_descriptors(height_m: np.ndarray, pixel_size_m: float = 30.0):
    """Грубые дескрипторы наклона по градиенту высот (как в реальном пайплайне).

    Возвращает (slope_mean, slope_std) в градусах.
    """
    gy, gx = np.gradient(height_m, pixel_size_m)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = np.degrees(slope_rad)
    return float(slope_deg.mean()), float(slope_deg.std())


def make_dummy(n_total: int, size: int, seed: int, out_dir: str):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    processed = out_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    # Сбалансированные классы (важно для обучения CVAE).
    labels = np.array([TERRAIN_TYPES[i % 3] for i in range(n_total)])
    rng.shuffle(labels)

    patches_norm = np.zeros((n_total, 1, size, size), dtype=np.float32)
    rows = []

    for i in range(n_total):
        terrain = labels[i]
        amp = TERRAIN_AMPLITUDE_M[terrain]
        # «частота» рельефа: горы более изрезанные, равнины — гладкие
        scale = {"flat": 4, "hilly": 8, "mountain": 16}[terrain]

        field = _smooth_field(size, scale, rng)          # [0, 1]
        height_m = field * amp + rng.uniform(-50, 4000)  # сдвиг абс. высоты

        # дескрипторы считаем на абсолютных высотах в метрах
        elev_range = float(height_m.max() - height_m.min())
        slope_mean, slope_std = _slope_descriptors(height_m)

        # per-patch min-max нормализация в [-1, 1] (как в контракте)
        lo, hi = float(height_m.min()), float(height_m.max())
        norm = (height_m - lo) / (hi - lo + 1e-8) * 2.0 - 1.0
        patches_norm[i, 0] = norm.astype(np.float32)

        rows.append(
            {
                "id": f"dummy_{i:06d}",
                "terrain_type": terrain,
                "elevation_range": round(elev_range, 2),
                "slope_mean": round(slope_mean, 4),
                "slope_std": round(slope_std, 4),
                "orig_min": round(lo, 2),
                "orig_max": round(hi, 2),
            }
        )

    # split 70/15/15 с фиксированным seed (как в контракте)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    split_of = {}
    for j, k in enumerate(idx):
        if j < n_train:
            split_of[k] = "train"
        elif j < n_train + n_val:
            split_of[k] = "val"
        else:
            split_of[k] = "test"

    meta = pd.DataFrame(rows)
    meta["split"] = [split_of[k] for k in range(n_total)]
    # порядок колонок по контракту
    meta = meta[
        ["id", "split", "terrain_type", "elevation_range",
         "slope_mean", "slope_std", "orig_min", "orig_max"]
    ]

    # сохраняем тензоры по split В ТОМ ЖЕ ПОРЯДКЕ, что строки metadata
    tensors = torch.from_numpy(np.ascontiguousarray(patches_norm))
    for split in ("train", "val", "test"):
        mask = (meta["split"] == split).to_numpy().copy()
        split_tensor = tensors[mask].contiguous()
        torch.save(split_tensor, processed / f"{split}.pt")
        # metadata для split уже в правильном порядке, т.к. mask сохраняет порядок
    meta.to_csv(out_dir / "metadata.csv", index=False)

    print("Dummy-данные созданы:")
    for split in ("train", "val", "test"):
        n = int((meta["split"] == split).sum())
        print(f"  {processed / (split + '.pt')}  ->  N={n}")
    print(f"  {out_dir / 'metadata.csv'}  ->  {len(meta)} строк")
    print(f"  баланс классов: {meta['terrain_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Генератор dummy-данных по контракту проекта")
    p.add_argument("--n", type=int, default=300, help="всего патчей (delится на 3 класса)")
    p.add_argument("--size", type=int, default=256, help="размер heightmap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data", help="каталог вывода")
    args = p.parse_args()
    make_dummy(args.n, args.size, args.seed, args.out)
