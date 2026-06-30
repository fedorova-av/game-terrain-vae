"""terrain_metrics.py — геоморфологические метрики реализма рельефа.

Оценивают, насколько обработанный heightmap приблизился к РЕАЛЬНЫМ плиткам по
структуре (а не только по амплитуде). Все метрики считаются на нормированной в
[0,1] высоте, чтобы сравнение раздельно от масштаба.

  roughness   — средняя |градиента| (общая изрезанность);
  slope_mean / slope_p90 — среднее и хвост уклонов (крутизна склонов/стенок русел);
  tri         — Terrain Ruggedness Index (Riley): средний перепад к 8 соседям;
  hf_energy   — доля спектральной энергии в средне-ВЫСОКОЙ полосе Фурье
                (структура мельче макроформы — то, чего не хватает гладкому VAE);
  rel_std     — std высокочастотной компоненты (детали после вычитания крупной формы).

class_separability — насколько распределения метрики разнесены между flat/hilly/
mountain (главный критерий пункта 2: ПОСЛЕ обработки типы должны различаться сильнее).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _n01(h):
    h = h.astype(np.float64)
    return (h - h.min()) / (h.max() - h.min() + 1e-12)


def roughness(h):
    gy, gx = np.gradient(_n01(h))
    return float(np.hypot(gx, gy).mean())


def slope_stats(h):
    gy, gx = np.gradient(_n01(h))
    s = np.hypot(gx, gy)
    return float(s.mean()), float(np.quantile(s, 0.90))


def tri(h):
    """Terrain Ruggedness Index (Riley 1999): средний по полю sqrt(sum (h-h_nb)^2)
    по 8 соседям. Растёт с изрезанностью."""
    hn = _n01(h)
    acc = np.zeros_like(hn)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nb = np.roll(np.roll(hn, dy, 0), dx, 1)
            acc += (hn - nb) ** 2
    return float(np.sqrt(acc)[1:-1, 1:-1].mean())


def hf_energy(h, band=(0.08, 0.5)):
    """Доля спектральной энергии в кольце нормированных частот [lo,hi] (1=Найквист).
    Средне-высокая полоса = структура мельче макроформы (русла/гребни/фактура)."""
    hn = _n01(h)
    F = np.abs(np.fft.fftshift(np.fft.fft2(hn))) ** 2
    H, W = hn.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - H // 2, xx - W // 2) / (min(H, W) / 2)
    ring = (r >= band[0]) & (r <= band[1])
    total = F.sum() - F[H // 2, W // 2]  # без DC
    return float(F[ring].sum() / (total + 1e-12))


def rel_std(h, sigma=6.0):
    """std высокочастотной части (после вычитания крупной формы) — «сколько детали»."""
    hn = _n01(h)
    return float((hn - gaussian_filter(hn, sigma)).std())


def tile_metrics(h) -> dict:
    sm, s90 = slope_stats(h)
    return {
        "roughness": roughness(h),
        "slope_mean": sm,
        "slope_p90": s90,
        "tri": tri(h),
        "hf_energy": hf_energy(h),
        "rel_std": rel_std(h),
    }


def class_separability(values_by_class: dict) -> float:
    """Разнесённость распределений метрики между классами: (разброс средних классов)
    / (средний внутриклассовый разброс). Больше = типы различаются сильнее."""
    means = np.array([np.mean(v) for v in values_by_class.values()])
    within = np.mean([np.std(v) for v in values_by_class.values()]) + 1e-12
    between = float(np.std(means))
    return between / within
