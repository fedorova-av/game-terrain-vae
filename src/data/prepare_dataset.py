"""
Запуск:
  python -m src.data.prepare_dataset --num-parquets 8 --subset 3000

Только обработка уже скачанного:
  python -m src.data.prepare_dataset --skip-download --subset 3000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ID = "Major-TOM/Core-DEM"
NODATA = -32767                # метка пропуска в Major TOM Core-DEM
PHYS_MIN_M = -450.0            # ниже самых низких впадин суши (артефакт)
PHYS_MAX_M = 9000.0            # выше Эвереста (8849 м) (артефакт)
MIN_RELIEF_M = 1.0            # перепад меньше (вода/константа)
NATIVE_SIZE = 356               # размер патча в датасете
NATIVE_PIXEL_M = 30.0           # разрешение, метры/пиксель

TERRAIN_THRESHOLDS = {"flat": 100.0, "mountain": 500.0}

# Поиск и скачивание parquet-файлов
def list_parquet_files() -> list:
    """Список всех parquet-файлов датасета"""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    parquets = sorted(f for f in files if f.startswith("images/") and f.endswith(".parquet"))
    if not parquets:
        raise RuntimeError("Не найдено parquet-файлов в датасете — проверь доступ к HF.")
    return parquets


def download_parquets(num_parquets: int, raw_dir: str, seed: int) -> list:
    """Скачивает num_parquets файлов
    """
    from huggingface_hub import hf_hub_download

    all_parquets = list_parquet_files()
    n = min(num_parquets, len(all_parquets))
    idx = np.unique(np.linspace(0, len(all_parquets) - 1, n).astype(int))
    chosen = [all_parquets[i] for i in idx]

    print(f"Скачиваю {len(chosen)} parquet-файлов из {len(all_parquets)} доступных...")
    local_paths = []
    for fn in chosen:
        p = hf_hub_download(repo_id=REPO_ID, filename=fn, repo_type="dataset", local_dir=raw_dir)
        local_paths.append(p)
        print(f"  ok: {fn}")
    return local_paths


def find_local_parquets(raw_dir: str) -> list:
    """Находит уже скачанные parquet-файлы"""
    paths = sorted(Path(raw_dir).rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"В {raw_dir} нет parquet-файлов. Сначала запусти без --skip-download."
        )
    return [str(p) for p in paths]

# Чтение и обработка одной плитки
def read_dem_tile(dem_bytes: bytes) -> np.ndarray:
    """Распаковывает GeoTIFF из байтов колонки DEM"""
    from rasterio.io import MemoryFile

    with MemoryFile(dem_bytes) as mem:
        with mem.open() as src:
            arr = src.read(1).astype(np.float32)
    return arr


def clean_and_resize(dem: np.ndarray, size: int, max_missing: float):
    """Чистит пропуски и приводит к size x size.
    """
    missing = (dem <= -32000) | (dem < PHYS_MIN_M) | (dem > PHYS_MAX_M)
    frac = float(missing.mean())
    if frac > max_missing:
        return None, False

    if missing.any():
        valid_median = float(np.median(dem[~missing]))
        dem = dem.copy()
        dem[missing] = valid_median
    t = torch.from_numpy(dem)[None, None]
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy(), True


def terrain_descriptors(height_m: np.ndarray, pixel_size_m: float):
    """Дескрипторы рельефа на абсолютных высотах в метрах.
    """
    elev_range = float(height_m.max() - height_m.min())
    gy, gx = np.gradient(height_m, pixel_size_m)
    slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    return elev_range, float(slope_deg.mean()), float(slope_deg.std())


def classify_terrain(elev_range: float) -> str:
    """Метка рельефа по фиксированным порогам перепада высот."""
    if elev_range < TERRAIN_THRESHOLDS["flat"]:
        return "flat"
    if elev_range > TERRAIN_THRESHOLDS["mountain"]:
        return "mountain"
    return "hilly"


def normalize_patch(height_m: np.ndarray):
    """Per-patch min-max нормализация в [-1, 1]."""
    lo, hi = float(height_m.min()), float(height_m.max())
    if hi - lo < 1e-6:           # вырожденная плоская плитка
        norm = np.zeros_like(height_m, dtype=np.float32)
    else:
        norm = ((height_m - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)
    return norm, lo, hi


# Сборка пула плиток из всех parquet
def build_pool(parquet_paths: list, size: int, max_missing: float):
    """Читает все parquet, обрабатывает плитки, возвращает пул записей.
    """
    import pyarrow.parquet as pq

    eff_pixel = NATIVE_PIXEL_M * NATIVE_SIZE / size  # размер пикселя после resize
    pool = []
    skipped = 0

    for path in parquet_paths:
        table = pq.read_table(path, columns=["DEM", "grid_cell"])
        dem_col = table.column("DEM")
        cell_col = table.column("grid_cell")
        print(f"  обрабатываю {Path(path).name}: {len(dem_col)} плиток")

        for i in range(len(dem_col)):
            dem = read_dem_tile(dem_col[i].as_py())
            height_m, ok = clean_and_resize(dem, size, max_missing)
            if not ok:
                skipped += 1
                continue

            elev_range, slope_mean, slope_std = terrain_descriptors(height_m, eff_pixel)
            if elev_range < MIN_RELIEF_M:
                skipped += 1
                continue
            terrain = classify_terrain(elev_range)
            norm, lo, hi = normalize_patch(height_m)

            pool.append(
                {
                    "id": str(cell_col[i].as_py()),
                    "terrain_type": terrain,
                    "elevation_range": round(elev_range, 2),
                    "slope_mean": round(slope_mean, 4),
                    "slope_std": round(slope_std, 4),
                    "orig_min": round(lo, 2),
                    "orig_max": round(hi, 2),
                    "patch": norm[None],  # [1, size, size]
                }
            )

    print(f"Пул собран: {len(pool)} плиток (отброшено по пропускам: {skipped})")
    dist = pd.Series([r["terrain_type"] for r in pool]).value_counts().to_dict()
    print(f"  распределение классов в пуле: {dist}")
    return pool

# Баланс классов и стратифицированный split
def balance_pool(pool: list, subset: int, seed: int):
    """Берёт примерно поровну плиток каждого класса."""
    if not subset or subset <= 0:
        from collections import Counter
        dist = dict(Counter(r["terrain_type"] for r in pool))
        print(f"  баланс ОТКЛЮЧЁН — берём всё: {dist}, итого {len(pool)}")
        return list(pool)
    rng = np.random.default_rng(seed)
    by_class = {}
    for r in pool:
        by_class.setdefault(r["terrain_type"], []).append(r)

    per_class = subset // 3
    balanced = []
    for cls in ("flat", "hilly", "mountain"):
        items = by_class.get(cls, [])
        take = min(per_class, len(items))
        if take < per_class:
            print(
                f"  ВНИМАНИЕ: класса '{cls}' всего {len(items)} (< {per_class}). "
                f"Возьму {take}. Чтобы добрать — увеличь --num-parquets."
            )
        idx = rng.choice(len(items), size=take, replace=False) if items else []
        balanced.extend(items[j] for j in idx)

    rng.shuffle(balanced)
    print(f"После баланса: {len(balanced)} плиток")
    return balanced


def split_stratified(records: list, seed: int):
    """Стратифицированный 70/15/15 по terrain_type."""
    rng = np.random.default_rng(seed)
    by_class = {}
    for r in records:
        by_class.setdefault(r["terrain_type"], []).append(r)

    for cls, items in by_class.items():
        idx = rng.permutation(len(items))
        n_train = int(0.70 * len(items))
        n_val = int(0.15 * len(items))
        for j, k in enumerate(idx):
            if j < n_train:
                items[k]["split"] = "train"
            elif j < n_train + n_val:
                items[k]["split"] = "val"
            else:
                items[k]["split"] = "test"
    return records

# Сохранение по контракту
def save_contract(records: list, out_dir: str):
    """Сохраняет .pt по split и metadata.csv в порядке, согласованном с тензорами."""
    out_dir = Path(out_dir)
    processed = out_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    for split in ("train", "val", "test"):
        split_recs = [r for r in records if r["split"] == split]
        patches = np.stack([r["patch"] for r in split_recs]).astype(np.float32)
        torch.save(torch.from_numpy(patches), processed / f"{split}.pt")
        for r in split_recs:
            meta_rows.append(
                {k: r[k] for k in
                 ["id", "split", "terrain_type", "elevation_range",
                  "slope_mean", "slope_std", "orig_min", "orig_max"]}
            )

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(out_dir / "metadata.csv", index=False)

    print("\nГотово. Контракт сохранён:")
    for split in ("train", "val", "test"):
        n = int((meta["split"] == split).sum())
        print(f"  {processed / (split + '.pt')}  ->  N={n}")
    print(f"  {out_dir / 'metadata.csv'}  ->  {len(meta)} строк")
    print(f"  классы итог: {meta['terrain_type'].value_counts().to_dict()}")


# main
def main():
    p = argparse.ArgumentParser(description="Подготовка Major TOM Core-DEM -> контракт проекта")
    p.add_argument("--num-parquets", type=int, default=8,
                   help="сколько parquet-файлов скачать")
    p.add_argument("--subset", type=int, default=3000, help="целевой размер выборки после баланса")
    p.add_argument("--size", type=int, default=256, help="размер плитки на выходе")
    p.add_argument("--max-missing", type=float, default=0.05,
                   help="макс. доля пропусков в плитке, иначе отбросить")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--raw-dir", type=str, default="data/raw", help="куда качать parquet")
    p.add_argument("--out", type=str, default="data", help="каталог вывода контракта")
    p.add_argument("--skip-download", action="store_true",
                   help="не качать, использовать уже скачанные parquet из --raw-dir")
    args = p.parse_args()

    if args.skip_download:
        parquet_paths = find_local_parquets(args.raw_dir)
        print(f"Использую {len(parquet_paths)} уже скачанных parquet-файлов.")
    else:
        parquet_paths = download_parquets(args.num_parquets, args.raw_dir, args.seed)

    pool = build_pool(parquet_paths, args.size, args.max_missing)
    pool_cols = ["id", "terrain_type", "elevation_range", "slope_mean", "slope_std"]
    pd.DataFrame([{k: r[k] for k in pool_cols} for r in pool]).to_csv(
        Path(args.out) / "pool_stats.csv", index=False
    )
    print(f"  статистика пула: {Path(args.out) / 'pool_stats.csv'}")
    balanced = balance_pool(pool, args.subset, args.seed)
    records = split_stratified(balanced, args.seed)
    save_contract(records, args.out)


if __name__ == "__main__":
    main()
