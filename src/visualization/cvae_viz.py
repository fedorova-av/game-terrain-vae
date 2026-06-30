"""Визуализация Conditional VAE и главная проверка управляемости.

Все функции принимают обученную `ConvCVAE` и работают в нормированном
пространстве [-1, 1]; для отображения высот в метрах используется средний
размах класса (из metadata). Нормировка масштаба-условия совпадает с обучением:
`scale_norm = (log1p(elevation_range_m) - log1p_mean) / log1p_std`, где
`log1p_mean/std` берутся из чекпойнта `best.pt` (ключ `scale_stats`).

Содержимое (покрывает требования части 3 из ТЗ):
  build_scale_dicts      — масштабы классов и медианные elevation_range из metadata
  condition_consistency  — ГЛАВНОЕ доказательство, что условие влияет (boxplots)
  sample_grid            — сетка сгенерированных плиток по классам
  reconstruction_grid    — реальные vs реконструкции по классам
  surface_3d             — 3D-поверхности рельефа
  latent_interpolation   — интерполяция z при фиксированном условии
  scale_sweep            — фикс. класс и z, меняем elevation_range (непрерывный контроль)
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

TERRAIN = ["flat", "hilly", "mountain"]

# Землистая палитра для heightmap в метрах (низины -> вершины).
MOUNTAIN_CMAP = LinearSegmentedColormap.from_list("mountain", [
    (0.00, "#2d4a2b"), (0.30, "#5a6b3c"), (0.50, "#8a7a52"),
    (0.70, "#9c8569"), (0.88, "#b5a99a"), (1.00, "#ffffff"),
])


# --------------------------------------------------------------------------
def build_scale_dicts(metadata, split: str = "train"):
    """Из metadata вернуть (scale_meters, elev_med_meters) по классам.

    scale_meters[c] — средний размах высот класса (orig_max-orig_min), м;
                      переводит выход модели [-1,1] в метры для отображения.
    elev_med[c]     — медианный elevation_range класса, м; подаётся как условие
                      при генерации «типичной» плитки этого класса.
    """
    import pandas as pd  # локальный импорт, чтобы модуль грузился и без pandas

    df = metadata if isinstance(metadata, pd.DataFrame) else pd.read_csv(metadata)
    if "split" in df.columns and split:
        sub = df[df["split"].astype(str).str.lower().eq(split)]
        if len(sub):
            df = sub
    scale_meters, elev_med = {}, {}
    for c in TERRAIN:
        sel = df[df["terrain_type"].astype(str).str.lower().eq(c)]
        if len(sel) and {"orig_min", "orig_max"}.issubset(df.columns):
            scale_meters[c] = float((sel["orig_max"] - sel["orig_min"]).mean())
        else:
            scale_meters[c] = float(sel["elevation_range"].mean()) if len(sel) else 1.0
        elev_med[c] = float(sel["elevation_range"].median()) if len(sel) else 0.0
    return scale_meters, elev_med


def norm_scale_value(er_raw: float, scale_stats: dict) -> float:
    """метры elevation_range -> нормированное условие (как при обучении)."""
    mean = float(scale_stats["log1p_mean"])
    std = float(scale_stats["log1p_std"])
    return (float(np.log1p(max(er_raw, 0.0))) - mean) / (std + 1e-12)


@torch.no_grad()
def _gen(model, cls_idx, n, elev_med, scale_stats, device):
    sc = norm_scale_value(elev_med[TERRAIN[cls_idx]], scale_stats)
    return model.sample(n, cls_idx, sc, device=device).cpu()


# --------------------------------------------------------------------------
@torch.no_grad()
def condition_consistency(model, scale_meters, elev_med, scale_stats,
                          device="cpu", n=200, save=None):
    """Генерим n плиток на класс, считаем elev_range и slope, рисуем boxplots.

    Доказательство управляемости: распределения должны расти flat→hilly→mountain.
    Возвращает dict {класс: (elevation_range[], slope[])} для отчёта.
    """
    def descriptors(batch, cls_name):
        h = (batch.squeeze(1) + 1) / 2 * scale_meters[cls_name]  # денормализация в метры
        er = (h.amax((1, 2)) - h.amin((1, 2))).numpy()
        gy, gx = torch.gradient(h, dim=(1, 2))
        sl = torch.sqrt(gy ** 2 + gx ** 2).mean((1, 2)).numpy()
        return er, sl

    results = {}
    for i, c in enumerate(TERRAIN):
        batch = _gen(model, i, n, elev_med, scale_stats, device)
        results[c] = descriptors(batch, c)

    print("Сгенерированные (медиана по классу):")
    for c in TERRAIN:
        er, sl = results[c]
        print(f"  {c:9s} elev_range={np.median(er):8.1f} м   slope≈{np.median(sl):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#4C9F70", "#E0A458", "#A4453A"]
    for ax, key, title, unit in [(axes[0], 0, "Сген. elevation range", "м"),
                                  (axes[1], 1, "Сген. slope (прокси)", "")]:
        bp = ax.boxplot([results[c][key] for c in TERRAIN],
                        patch_artist=True, showfliers=False)
        ax.set_xticks(range(1, len(TERRAIN) + 1)); ax.set_xticklabels(TERRAIN)  # совместимо со старым matplotlib
        for p, col in zip(bp["boxes"], colors):
            p.set_facecolor(col); p.set_alpha(.7)
        ax.set_title(title); ax.set_ylabel(unit); ax.grid(axis="y", alpha=.3)
    fig.suptitle("Condition-consistency: рельеф растёт flat → hilly → mountain")
    fig.tight_layout()
    if save:
        _ensure_parent(save); fig.savefig(save, dpi=130, bbox_inches="tight")
    plt.show()
    return results


@torch.no_grad()
def sample_grid(model, scale_meters, elev_med, scale_stats, device="cpu", n=4, save=None):
    vmax = max(scale_meters.values())
    fig, axes = plt.subplots(3, n, figsize=(2.4 * n, 7)); ims = None
    for r, c in enumerate(TERRAIN):
        s = _gen(model, r, n, elev_med, scale_stats, device).squeeze(1).numpy()
        for j in range(n):
            ims = axes[r, j].imshow((s[j] + 1) / 2 * scale_meters[c], cmap=MOUNTAIN_CMAP, vmin=0, vmax=vmax)
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            if j == 0:
                axes[r, j].set_ylabel(c, fontsize=13)
    fig.suptitle("Сгенерированные heightmap по условию", fontsize=13)
    fig.colorbar(ims, ax=axes, fraction=0.025, label="высота, м")
    if save:
        _ensure_parent(save); fig.savefig(save, dpi=130, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def reconstruction_grid(model, x, label, scale, scale_meters, device="cpu", save=None):
    """Реальные плитки vs реконструкции (по строке на пример).

    x: [B,1,H,W], label: [B] long, scale: [B] float (нормированный) — обычно
    один батч из val/test. Сверху real, снизу recon.
    """
    x = x.to(device); label = label.to(device); scale = scale.to(device)
    recon, _, _ = model(x, label, scale)
    x = x.cpu().squeeze(1).numpy(); recon = recon.clamp(-1, 1).cpu().squeeze(1).numpy()
    lab = label.cpu().numpy()
    b = min(len(x), 6)
    fig, axes = plt.subplots(2, b, figsize=(2.2 * b, 4.6))
    for j in range(b):
        sm = scale_meters[TERRAIN[int(lab[j])]]
        axes[0, j].imshow((x[j] + 1) / 2 * sm, cmap=MOUNTAIN_CMAP); axes[0, j].set_title(TERRAIN[int(lab[j])], fontsize=10)
        axes[1, j].imshow((recon[j] + 1) / 2 * sm, cmap=MOUNTAIN_CMAP)
        for r in (0, 1):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("real", fontsize=12); axes[1, 0].set_ylabel("recon", fontsize=12)
    fig.suptitle("Реконструкция CVAE (real vs recon)", fontsize=13)
    fig.tight_layout()
    if save:
        _ensure_parent(save); fig.savefig(save, dpi=130, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def surface_3d(model, scale_meters, elev_med, scale_stats, device="cpu", seed=1, save=None):
    vmax = max(scale_meters.values()); step = 4
    gg = np.arange(0, 256, step); X, Y = np.meshgrid(gg, gg)
    fig = plt.figure(figsize=(13, 4.5))
    for r, c in enumerate(TERRAIN):
        torch.manual_seed(seed)
        s = _gen(model, r, 1, elev_med, scale_stats, device).squeeze().numpy()
        Z = ((s + 1) / 2 * scale_meters[c])[::step, ::step]
        ax = fig.add_subplot(1, 3, r + 1, projection="3d")
        ax.plot_surface(X, Y, Z, cmap=MOUNTAIN_CMAP, vmin=0, vmax=vmax, linewidth=0, antialiased=True)
        ax.set_zlim(0, vmax); ax.set_title(c, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zlabel("м", fontsize=9)
    fig.suptitle("3D-поверхность сгенерированного рельефа", fontsize=13)
    fig.tight_layout()
    if save:
        _ensure_parent(save); fig.savefig(save, dpi=130, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def latent_interpolation(model, scale_meters, elev_med, scale_stats,
                         device="cpu", steps=8, seed=0, save=None):
    vmax = max(scale_meters.values())
    fig, axes = plt.subplots(3, steps, figsize=(1.6 * steps, 6))
    for r, c in enumerate(TERRAIN):
        torch.manual_seed(seed)
        z0 = torch.randn(1, model.latent_dim, device=device)
        z1 = torch.randn(1, model.latent_dim, device=device)
        sc = norm_scale_value(elev_med[c], scale_stats)
        label = torch.full((1,), r, dtype=torch.long, device=device)
        scale_t = torch.full((1,), sc, device=device)
        cond = model.cond_vector(label, scale_t)
        for j, a in enumerate(np.linspace(0, 1, steps)):
            z = (1 - a) * z0 + a * z1
            img = model.decode(z, cond).clamp(-1, 1).cpu().squeeze().numpy()
            axes[r, j].imshow((img + 1) / 2 * scale_meters[c], cmap=MOUNTAIN_CMAP, vmin=0, vmax=vmax)
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            if j == 0:
                axes[r, j].set_ylabel(c, fontsize=12)
    fig.suptitle("Latent interpolation (z0 → z1) при фиксированном условии", fontsize=13)
    fig.tight_layout()
    if save:
        _ensure_parent(save); fig.savefig(save, dpi=130, bbox_inches="tight")
    plt.show()


@torch.no_grad()
def scale_sweep(model, scale_meters, scale_stats, cls_idx=2,
                er_values=(200, 600, 1200, 2000), device="cpu", seed=0, save=None):
    """Фикс. класс и z, меняем elevation_range — непрерывный контроль масштаба.

    Сильный пункт демо: условие не только 3 категории, но и непрерывная величина.
    """
    c = TERRAIN[cls_idx]; vmax = max(scale_meters.values())
    torch.manual_seed(seed)
    z = torch.randn(1, model.latent_dim, device=device)
    label = torch.full((1,), cls_idx, dtype=torch.long, device=device)
    fig, axes = plt.subplots(1, len(er_values), figsize=(2.6 * len(er_values), 3))
    for j, er in enumerate(er_values):
        sc = torch.full((1,), norm_scale_value(er, scale_stats), device=device)
        cond = model.cond_vector(label, sc)
        img = model.decode(z, cond).clamp(-1, 1).cpu().squeeze().numpy()
        axes[j].imshow((img + 1) / 2 * scale_meters[c], cmap=MOUNTAIN_CMAP, vmin=0, vmax=vmax)
        axes[j].set_title(f"{er} м"); axes[j].set_xticks([]); axes[j].set_yticks([])
    fig.suptitle(f"Управляемость масштабом: класс={c}, фикс. z, разный elevation_range", fontsize=12)
    fig.tight_layout()
    if save:
        _ensure_parent(save); fig.savefig(save, dpi=130, bbox_inches="tight")
    plt.show()


def _ensure_parent(path) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
