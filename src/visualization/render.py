"""Рендер heightmap в финальную картинку «для игр».

Большая часть визуальной красоты — здесь, а не в модели: hillshade (отмывка
рельефа направленным светом) превращает плоскую карту высот в объёмный рельеф,
а высотная палитра (зелёный лес -> камень -> снег) добавляет реалистичный цвет.

Функции:
  - hillshade:        теневая отмывка по азимуту/высоте солнца
  - colorize:         высотная палитра + смешивание с отмывкой
  - render_tile:      одна плитка -> RGB-изображение для игры
  - batch_render:     оптовый рендер сетки плиток в один PNG
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LightSource

# Высотная палитра: низины — лес, середина — камень/трава, верх — снег.
TERRAIN_PALETTE = LinearSegmentedColormap.from_list("game_terrain", [
    (0.00, "#2f5d3a"),  # тёмный лес
    (0.25, "#5a7d44"),  # трава
    (0.45, "#8a8159"),  # предгорья
    (0.65, "#9c8569"),  # камень
    (0.82, "#b9b0a4"),  # голый склон
    (0.92, "#e4e0db"),  # снеговая линия
    (1.00, "#ffffff"),  # снег
])


def _to_meters(tile_norm: np.ndarray, height_scale_m: float) -> np.ndarray:
    """[-1,1] -> метры по масштабу класса (для корректной отмывки)."""
    return (tile_norm + 1.0) / 2.0 * height_scale_m


def hillshade(height_m: np.ndarray, azimuth=315.0, altitude=45.0,
              cell_size=30.0, z_factor=1.0) -> np.ndarray:
    """Классическая отмывка рельефа (как в ГИС). Возвращает [0,1] яркость.

    azimuth  — азимут света (315° = свет с северо-запада, стандарт для карт)
    altitude — высота солнца над горизонтом, градусы
    cell_size— размер пикселя в метрах (Core-DEM ~30 м)
    z_factor — вертикальное преувеличение (>1 усиливает рельеф)
    """
    az = np.deg2rad(360.0 - azimuth + 90.0)
    alt = np.deg2rad(altitude)
    dy, dx = np.gradient(height_m * z_factor, cell_size)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)
    shaded = (np.sin(alt) * np.sin(slope)
              + np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return np.clip((shaded + 1.0) / 2.0, 0.0, 1.0)


def colorize(tile_norm: np.ndarray, height_scale_m: float,
             shade_strength=0.55, **hs_kwargs) -> np.ndarray:
    """heightmap [-1,1] -> RGB [H,W,3], палитра по высоте × отмывка."""
    height_m = _to_meters(tile_norm, height_scale_m)
    # цвет по нормированной высоте
    norm = (tile_norm + 1.0) / 2.0
    rgb = TERRAIN_PALETTE(np.clip(norm, 0, 1))[..., :3]
    # отмывка
    shade = hillshade(height_m, **hs_kwargs)[..., None]
    # смешиваем: цвет затемняется тенью, светлые склоны подсвечиваются
    shaded = rgb * (1.0 - shade_strength + shade_strength * 2.0 * shade)
    return np.clip(shaded, 0.0, 1.0)


def render_tile(tile_norm, height_scale_m, save=None, title=None,
                figsize=(5, 5), **kwargs):
    """Одна плитка -> сохранённая картинка для игры."""
    if hasattr(tile_norm, "detach"):
        tile_norm = tile_norm.detach().cpu().numpy()
    tile_norm = np.squeeze(tile_norm)
    rgb = colorize(tile_norm, height_scale_m, **kwargs)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight", pad_inches=0)
    return fig


def batch_render(tiles_norm, height_scales, labels=None, ncols=4, save=None,
                 class_names=("flat", "hilly", "mountain"), **kwargs):
    """Оптовый рендер: список плиток -> сетка RGB в одном PNG.

    tiles_norm:    [N,1,H,W] или [N,H,W] (tensor/np), значения [-1,1]
    height_scales: float (один на всех) или список длины N (масштаб класса, м)
    labels:        опционально [N] для подписей классов
    """
    if hasattr(tiles_norm, "detach"):
        tiles_norm = tiles_norm.detach().cpu().numpy()
    tiles_norm = np.asarray(tiles_norm)
    if tiles_norm.ndim == 4:
        tiles_norm = tiles_norm[:, 0]
    n = tiles_norm.shape[0]
    if np.isscalar(height_scales):
        height_scales = [float(height_scales)] * n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(axes)):
        axes[i].axis("off")
        if i >= n:
            continue
        rgb = colorize(tiles_norm[i], height_scales[i], **kwargs)
        axes[i].imshow(rgb)
        if labels is not None:
            lab = int(labels[i]) if not hasattr(labels[i], "item") else labels[i].item()
            axes[i].set_title(class_names[lab], fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def sharpen_heightmap(h: np.ndarray, amount: float = 0.8, sigma: float = 2.0) -> np.ndarray:
    """Unsharp mask: усиливает локальный контраст гребней.

    Presentation-приём для 3D — делает пики и долины резче. Модель не трогает
    (CVAE по-прежнему сглажен, это только визуальное усиление при рендере).
    """
    from scipy.ndimage import gaussian_filter

    blur = gaussian_filter(h, sigma=sigma)
    return h + amount * (h - blur)


def render_surface_3d(tile_norm, ax=None, cmap=TERRAIN_PALETTE, sharpen: float = 0.9,
                      gamma: float = 1.6, exag: float = 95.0, step: int = 2,
                      elev: float = 24.0, azim: float = -60.0, sun_alt: float = 28.0,
                      save=None, figsize=(10, 7)):
    """3D-перспектива поверхности рельефа (а не вид сверху).

    sharpen — сила unsharp для резкости пиков; gamma — акцент высоты (долины ниже,
    пики выше); exag — вертикальное преувеличение; step — прореживание сетки.
    Затенение через LightSource. Возвращает ax.
    """
    if hasattr(tile_norm, "detach"):
        tile_norm = tile_norm.detach().cpu().numpy()
    h = np.squeeze(tile_norm)[::step, ::step]
    H = np.clip((h + 1.0) / 2.0, 0, 1)
    if sharpen:
        H = np.clip(sharpen_heightmap(H, amount=sharpen), 0, 1)
    Hp = H ** gamma
    ny, nx = Hp.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    Z = Hp * exag
    rgb = cmap(Hp)[..., :3]
    shaded = LightSource(azdeg=315, altdeg=sun_alt).shade_rgb(rgb, Z, vert_exag=2.5, blend_mode="soft")
    own = ax is None
    if own:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=shaded, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)
    ax.set_box_aspect((nx, ny, exag * 1.1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_zlim(0, exag * 1.5)
    if own and save:
        ax.figure.savefig(save, dpi=145, bbox_inches="tight")
    return ax
