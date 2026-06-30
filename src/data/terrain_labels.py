"""
terrain_labels.py — условные метки рельефа для Conditional VAE.

Зачем нужен этот модуль
-----------------------
CVAE управляет генерацией по типу рельефа `flat / hilly / mountain`. Метки
уже лежат в `metadata.csv` (колонка `terrain_type`), но важно
уметь:
  1. объяснить, ОТКУДА берётся метка (по каким порогам), и воспроизвести её,
  2. восстановить метку, если её вдруг нет (fallback по `elevation_range`),
  3. посчитать баланс классов и масштабы — это вход для conditioning и для
     condition-consistency проверки.

Пороги — те же физические значения, что зафиксированы в
`report/data_plan.md` (раздел 6). Они абсолютные и не зависят от subset,
поэтому метка одного и того же патча не «плавает» между запусками.

    | тип       | условие на elevation_range (метры) |
    |-----------|------------------------------------|
    | flat      | er < 100                           |
    | hilly     | 100 <= er <= 500                   |
    | mountain  | er > 500                           |

Соответствие индексов (`flat=0, hilly=1, mountain=2`) совпадает с
`src/data/dataset.py` и `src/training/train.py`, чтобы embedding класса в
CVAE и стратификация метрик (metrics.py) указывали на один и тот же класс.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- Канонический порядок классов (единый для всего проекта) ----------------
TERRAIN_TYPES: list[str] = ["flat", "hilly", "mountain"]
TERRAIN_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(TERRAIN_TYPES)}
IDX_TO_TERRAIN: dict[int, str] = {i: t for t, i in TERRAIN_TO_IDX.items()}
NUM_TERRAIN_CLASSES: int = len(TERRAIN_TYPES)

# --- Физические пороги по перепаду высот, метры (см. data_plan.md, п.6) ------
FLAT_MAX_M: float = 100.0     # er < 100        -> flat
HILLY_MAX_M: float = 500.0    # 100 <= er <= 500 -> hilly, иначе mountain


def label_from_elevation_range(elevation_range: float) -> str:
    """Перепад высот (метры) -> строковая метка рельефа."""
    er = float(elevation_range)
    if er < FLAT_MAX_M:
        return "flat"
    if er <= HILLY_MAX_M:
        return "hilly"
    return "mountain"


def label_id_from_elevation_range(elevation_range: float) -> int:
    """Перепад высот (метры) -> индекс класса (0/1/2)."""
    return TERRAIN_TO_IDX[label_from_elevation_range(elevation_range)]


def labels_from_metadata(meta: pd.DataFrame, derive_if_missing: bool = True) -> np.ndarray:
    """Вернуть массив индексов класса [N] для строк metadata.

    По умолчанию использует готовую колонку `terrain_type`. Если её нет (или
    `derive_if_missing=True` и встретилось неизвестное значение) — выводит
    метку из `elevation_range` по физическим порогам.
    """
    if "terrain_type" in meta.columns:
        ids = []
        for i, value in enumerate(meta["terrain_type"].astype(str).str.lower()):
            if value in TERRAIN_TO_IDX:
                ids.append(TERRAIN_TO_IDX[value])
            elif derive_if_missing and "elevation_range" in meta.columns:
                ids.append(label_id_from_elevation_range(meta["elevation_range"].iloc[i]))
            else:
                raise ValueError(f"Неизвестный terrain_type={value!r} в строке {i}")
        return np.asarray(ids, dtype=np.int64)

    if "elevation_range" not in meta.columns:
        raise ValueError("В metadata нет ни terrain_type, ни elevation_range — метку не вывести")
    return meta["elevation_range"].map(label_id_from_elevation_range).to_numpy(dtype=np.int64)


def add_terrain_labels(meta: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    """Добавить (или пересчитать) колонку `terrain_type` из `elevation_range`.

    Полезно как fallback, если metadata пришла без меток. При
    `overwrite=False` существующая колонка не трогается.
    """
    out = meta.copy()
    if "terrain_type" in out.columns and not overwrite:
        return out
    if "elevation_range" not in out.columns:
        raise ValueError("Нужна колонка elevation_range для разметки")
    out["terrain_type"] = out["elevation_range"].map(label_from_elevation_range)
    return out


def class_balance(meta: pd.DataFrame) -> dict[str, int]:
    """Сколько патчей каждого класса (в каноническом порядке)."""
    ids = labels_from_metadata(meta)
    counts = np.bincount(ids, minlength=NUM_TERRAIN_CLASSES)
    return {IDX_TO_TERRAIN[i]: int(counts[i]) for i in range(NUM_TERRAIN_CLASSES)}


def class_scale_meters(meta: pd.DataFrame) -> dict[str, float]:
    """Средний размах высот (orig_max - orig_min), метры, по каждому классу.

    Нужен визуализации, чтобы перевести нормированный выход модели [-1, 1]
    обратно в метры для отображения (см. src/visualization/cvae_viz.py).
    """
    if not {"orig_min", "orig_max"}.issubset(meta.columns):
        # fallback: если нет orig_min/max, берём средний elevation_range
        return class_median_elevation_range(meta)
    span = (meta["orig_max"].astype(float) - meta["orig_min"].astype(float))
    out = {}
    for t in TERRAIN_TYPES:
        mask = labels_from_metadata(meta) == TERRAIN_TO_IDX[t]
        out[t] = float(span[mask].mean()) if mask.any() else 0.0
    return out


def class_median_elevation_range(meta: pd.DataFrame) -> dict[str, float]:
    """Медианный elevation_range класса, метры — типичное условие при генерации."""
    out = {}
    ids = labels_from_metadata(meta)
    er = meta["elevation_range"].astype(float).to_numpy()
    for t in TERRAIN_TYPES:
        mask = ids == TERRAIN_TO_IDX[t]
        out[t] = float(np.median(er[mask])) if mask.any() else 0.0
    return out


def _self_check(metadata_csv: str) -> None:
    """CLI-проверка: баланс классов и согласованность меток с порогами."""
    meta = pd.read_csv(metadata_csv)
    print(f"metadata: {metadata_csv}  | строк: {len(meta)}")
    print("баланс классов:", class_balance(meta))
    if {"terrain_type", "elevation_range"}.issubset(meta.columns):
        derived = meta["elevation_range"].map(label_from_elevation_range)
        disagree = int((derived.str.lower() != meta["terrain_type"].astype(str).str.lower()).sum())
        print(f"меток не совпадает с порогами (<100/100-500/>500): {disagree} / {len(meta)}")
    print("масштаб класса (средний размах, м):",
          {k: round(v, 1) for k, v in class_scale_meters(meta).items()})
    print("медианный elevation_range класса (м):",
          {k: round(v, 1) for k, v in class_median_elevation_range(meta).items()})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Проверка условных меток рельефа")
    parser.add_argument("--metadata", default="data/metadata.csv", help="путь к metadata.csv")
    args = parser.parse_args()
    path = Path(args.metadata)
    if not path.exists():
        raise SystemExit(f"Не найден {path}. Укажи --metadata путь до metadata.csv")
    _self_check(str(path))
