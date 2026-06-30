"""terrain_ops.py — слой ПОСТОБРАБОТКИ heightmap между CVAE и рендером.

Зачем: выход CVAE гладкий/низкочастотный (пиксельный лосс + усреднение латента →
размытость — известное свойство VAE). Эти операции добавляют геоморфологическую
структуру (долины, гребни, осыпи, изломанные хребты) ПОВЕРХ макроформы CVAE.

Концептуальная рамка (честно для защиты): CVAE задаёт управляемую МАКРОСТРУКТУРУ
(где горы, какой тип, какой масштаб — это её вклад и метрики). Постобработка
УСИЛИВАЕТ эту форму, а не создаёт свою (так делают StyleDEM и pro terrain-tools).
Эрозия держится умеренной; проверяется, что низкочастотная форма осталась от модели
(см. macroform_correlation).

Всё — чистый numpy/scipy, без Blender. Каждая операция принимает heightmap [H,W]
float и возвращает обработанный heightmap того же размера.

Алгоритм гидравлической эрозии адаптирован из общеизвестных реализаций
(Sebastian Lague / dandrino terrain-erosion-3-ways / Job Talle): droplet-based —
капля катится по градиенту, набирает/осаждает осадок, прорезает русла и гребни.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


# --------------------------------------------------------------------------
def normalize01(h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float64)
    lo, hi = float(h.min()), float(h.max())
    return (h - lo) / (hi - lo + 1e-12)


def _make_brush(radius: int, shape):
    """Веса кисти эрозии (распределяем эрозию по окрестности — без пик-артефактов).

    Возвращает (offsets [K,2] int, weights [K]) — смещения вокруг ячейки и веса
    (1 - dist/radius), нормированные к сумме 1.
    """
    H, W = shape
    offs, wts = [], []
    r = int(radius)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d = np.hypot(dx, dy)
            if d <= radius:
                offs.append((dy, dx))
                wts.append(1.0 - d / radius)
    offs = np.array(offs, np.int64)
    wts = np.array(wts, np.float64)
    wts /= wts.sum()
    return offs, wts


def _height_and_grad(h, x, y):
    """Билинейная высота + градиент в точке (x, y) непрерывных координат."""
    H, W = h.shape
    x0 = int(x); y0 = int(y)
    fx = x - x0; fy = y - y0
    x1 = min(x0 + 1, W - 1); y1 = min(y0 + 1, H - 1)
    h00 = h[y0, x0]; h10 = h[y0, x1]; h01 = h[y1, x0]; h11 = h[y1, x1]
    # градиент (по Лагью): интерполяция разностей по краям ячейки
    gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
    gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
    height = (h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy)
              + h01 * (1 - fx) * fy + h11 * fx * fy)
    return height, gx, gy


# --------------------------------------------------------------------------
def hydraulic_erosion(
    h: np.ndarray,
    num_droplets: int = 60000,
    *,
    seed: int = 0,
    lifetime: int = 30,
    inertia: float = 0.05,
    sediment_capacity: float = 4.0,
    min_slope: float = 0.01,
    erode_speed: float = 0.3,
    deposit_speed: float = 0.3,
    evaporation: float = 0.01,
    gravity: float = 4.0,
    erosion_radius: int = 3,
    init_speed: float = 1.0,
    init_water: float = 1.0,
) -> np.ndarray:
    """Droplet-based гидравлическая эрозия. Вход/выход — heightmap [H,W].

    Нормируется в [0,1] внутри (параметры калиброваны под этот диапазон), на выходе
    масштабируется обратно в исходный диапазон значений входа.

    num_droplets ~ 30-60k на 256x256. Больше капель -> выраженнее русла/гребни.
    """
    lo, hi = float(h.min()), float(h.max())
    H, W = h.shape
    hm = ((h.astype(np.float64) - lo) / (hi - lo + 1e-12)).copy()  # [0,1]
    rng = np.random.default_rng(seed)
    brush_off, brush_w = _make_brush(erosion_radius, (H, W))

    for _ in range(num_droplets):
        x = rng.uniform(1, W - 2)
        y = rng.uniform(1, H - 2)
        dx = dy = 0.0
        speed = init_speed
        water = init_water
        sediment = 0.0

        for _step in range(lifetime):
            x0 = int(x); y0 = int(y)
            fx = x - x0; fy = y - y0
            height, gx, gy = _height_and_grad(hm, x, y)

            # направление с инерцией
            dx = dx * inertia - gx * (1 - inertia)
            dy = dy * inertia - gy * (1 - inertia)
            mag = np.hypot(dx, dy)
            if mag < 1e-8:
                break
            dx /= mag; dy /= mag
            nx = x + dx; ny = y + dy
            if nx < 1 or nx >= W - 1 or ny < 1 or ny >= H - 1:
                break

            new_height, _, _ = _height_and_grad(hm, nx, ny)
            dh = new_height - height  # <0 = вниз

            capacity = max(-dh, min_slope) * speed * water * sediment_capacity

            if sediment > capacity or dh > 0:
                # осаждение (вверх — сбрасываем; избыток осадка — осаждаем)
                if dh > 0:
                    amt = min(dh, sediment)
                else:
                    amt = (sediment - capacity) * deposit_speed
                sediment -= amt
                # билинейно в 4 ячейки текущей позиции
                hm[y0, x0] += amt * (1 - fx) * (1 - fy)
                hm[y0, min(x0 + 1, W - 1)] += amt * fx * (1 - fy)
                hm[min(y0 + 1, H - 1), x0] += amt * (1 - fx) * fy
                hm[min(y0 + 1, H - 1), min(x0 + 1, W - 1)] += amt * fx * fy
            else:
                # эрозия (не больше, чем перепад вниз; распределяем кистью)
                amt = min((capacity - sediment) * erode_speed, -dh)
                ys = y0 + brush_off[:, 0]; xs = x0 + brush_off[:, 1]
                ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
                take = amt * brush_w[ok]
                # не уносим ниже нуля
                take = np.minimum(take, hm[ys[ok], xs[ok]])
                hm[ys[ok], xs[ok]] -= take
                sediment += take.sum()

            speed = np.sqrt(max(0.0, speed * speed + dh * gravity))
            water *= (1 - evaporation)
            x, y = nx, ny

    hm = np.clip(hm, 0.0, None)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-12)  # ренормируем [0,1]
    return (hm * (hi - lo) + lo).astype(np.float32)        # обратно в диапазон входа


# --- 3. термальная эрозия (осыпи на крутых склонах) -----------------------
def thermal_erosion(h, iterations=60, talus=0.004, factor=0.5):
    """Где локальный перепад к соседу > угла естественного откоса (talus) — часть
    материала осыпается вниз. Несколько итераций -> прямые осыпные (scree) склоны.
    Векторизовано (8 соседей через np.roll), материал сохраняется."""
    lo, hi = float(h.min()), float(h.max())
    hm = ((h.astype(np.float64) - lo) / (hi - lo + 1e-12)).copy()
    for _ in range(iterations):
        out = np.zeros_like(hm)
        inflow = np.zeros_like(hm)
        for axis, sh in [(1, 1), (1, -1), (0, 1), (0, -1)]:
            nb = np.roll(hm, -sh, axis=axis)            # сосед в направлении +sh
            d = hm - nb
            move = np.where(d > talus, (d - talus) * factor * 0.25, 0.0)
            out += move
            inflow += np.roll(move, sh, axis=axis)       # материал приходит к соседу
        hm = hm - out + inflow
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-12)
    return (hm * (hi - lo) + lo).astype(np.float32)


# --- 4. нелинейное усиление пиков (power transform) -----------------------
def power_transform(h, gamma=0.7):
    """h_norm ** gamma. gamma<1 поднимает вершины (острые пики), gamma>1 прижимает.
    Долины (near 0) почти не меняются, вершины резко выше."""
    lo, hi = float(h.min()), float(h.max())
    hn = np.clip((h.astype(np.float64) - lo) / (hi - lo + 1e-12), 0, 1)
    return (hn ** float(gamma) * (hi - lo) + lo).astype(np.float32)


# --- 5. domain warping (изломанные гребни вместо плавных волн) -------------
def domain_warp(h, strength=7.0, noise_scale=16.0, seed=0):
    """Сэмплируем h по ИСКАЖЁННЫМ координатам: warped(x,y)=h(x+a*n1, y+a*n2).
    Гладкие волны -> изломанные гребни. strength в пикселях, noise_scale=sigma шума."""
    from scipy.ndimage import map_coordinates
    H, W = h.shape
    r = np.random.default_rng(seed)
    ox = gaussian_filter(r.standard_normal((H, W)), noise_scale)
    oy = gaussian_filter(r.standard_normal((H, W)), noise_scale)
    ox /= (np.abs(ox).max() + 1e-9); oy /= (np.abs(oy).max() + 1e-9)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    warped = map_coordinates(h.astype(np.float64),
                             [yy + strength * oy, xx + strength * ox],
                             order=1, mode="reflect")
    return warped.astype(np.float32)


# --- 6. маска деталей по уклону/кривизне (детали в логичных местах) --------
def detail_by_curvature(h, amp=0.06, detail_sigma=1.1, slope_gain=2.2, seed=0):
    """Высокочастотная деталь ТОЛЬКО на склонах/гребнях (маска по уклону), равнины и
    долины остаются гладкими -> бьёт в «шум везде как пластилин»."""
    lo, hi = float(h.min()), float(h.max())
    hn = (h.astype(np.float64) - lo) / (hi - lo + 1e-12)
    gy, gx = np.gradient(hn)
    slope = np.hypot(gx, gy)
    mask = np.clip(slope / (slope.max() + 1e-9) * slope_gain, 0, 1)
    r = np.random.default_rng(seed)
    noise = gaussian_filter(r.standard_normal(h.shape), detail_sigma)
    noise -= noise.mean(); noise /= (np.abs(noise).max() + 1e-9)
    out = hn + amp * mask * noise
    out = (out - out.min()) / (out.max() - out.min() + 1e-12)
    return (out * (hi - lo) + lo).astype(np.float32)


# --- комбинированный конвейер ---------------------------------------------
def full_pipeline(h, *, gamma=0.8, warp_strength=6.0, droplets=70000,
                  thermal_iters=50, detail_amp=0.05, seed=0):
    """Лучшая комбинация (порядок важен): warp -> power -> hydraulic -> thermal -> detail.
    Сначала ломаем гребни и поднимаем пики, потом режем русла, осыпаем склоны и
    добавляем мелкую фактуру только на склонах."""
    x = domain_warp(h, strength=warp_strength, seed=seed)
    x = power_transform(x, gamma=gamma)
    x = hydraulic_erosion(x, num_droplets=droplets, seed=seed, lifetime=40,
                          erode_speed=0.4, sediment_capacity=6.0, min_slope=0.001,
                          erosion_radius=2, deposit_speed=0.25)
    x = thermal_erosion(x, iterations=thermal_iters, talus=0.005, factor=0.5)
    x = detail_by_curvature(x, amp=detail_amp, seed=seed)
    return x


# --- 2. сила обработки ПО КЛАССУ (решает «типы не различаются») ------------
def process_by_class(h, terrain_type, seed=0):
    """Разная постобработка по типу рельефа -> типы становятся СТРУКТУРНО разными,
    не только по амплитуде (по мотивам Minecraft worldgen: эрозия решает плоское
    vs гористое). Термальная держится ЛЁГКОЙ, чтобы не съесть русла эрозии.
      flat     — почти не трогаем (gamma прижимает в равнину, чуть эрозии);
      hilly    — слабая эрозия + мягкая gamma -> пологие складки;
      mountain — warp + острые пики + сильная эрозия -> выраженные гребни/долины.
    """
    t = terrain_type
    if t == "flat":
        x = power_transform(h, gamma=2.2)
        x = hydraulic_erosion(x, num_droplets=8000, seed=seed, lifetime=28,
                              erode_speed=0.3, sediment_capacity=4.0, min_slope=0.002,
                              erosion_radius=2)
        return detail_by_curvature(x, amp=0.02, seed=seed)
    if t == "hilly":
        x = domain_warp(h, strength=5.0, noise_scale=16.0, seed=seed)
        x = power_transform(x, gamma=1.4)
        x = hydraulic_erosion(x, num_droplets=35000, seed=seed, lifetime=35,
                              erode_speed=0.35, sediment_capacity=5.0, min_slope=0.0015,
                              erosion_radius=2)
        x = thermal_erosion(x, iterations=6, talus=0.006, factor=0.5)
        return detail_by_curvature(x, amp=0.035, seed=seed)
    # mountain
    x = domain_warp(h, strength=7.0, noise_scale=13.0, seed=seed)
    x = power_transform(x, gamma=0.72)
    x = hydraulic_erosion(x, num_droplets=85000, seed=seed, lifetime=42,
                          erode_speed=0.45, sediment_capacity=7.0, min_slope=0.001,
                          erosion_radius=2, deposit_speed=0.25)
    x = thermal_erosion(x, iterations=6, talus=0.007, factor=0.5)  # очень лёгкая: не съесть русла
    return detail_by_curvature(x, amp=0.05, seed=seed)


# --------------------------------------------------------------------------
def macroform_correlation(h_raw: np.ndarray, h_proc: np.ndarray, sigma: float = 8.0) -> float:
    """Корреляция НИЗКОЧАСТОТНЫХ форм raw и обработанного (контроль: эрозия усилила,
    а не подменила макроформу CVAE). Близко к 1 = крупная форма сохранена."""
    a = gaussian_filter(normalize01(h_raw), sigma)
    b = gaussian_filter(normalize01(h_proc), sigma)
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
    return float((a * b).sum() / denom)


# --------------------------------------------------------------------------
def quick_stats(h: np.ndarray) -> dict:
    """Грубые метрики для быстрой проверки (полный набор — в terrain_metrics.py)."""
    hn = normalize01(h)
    gy, gx = np.gradient(hn)
    slope = np.hypot(gx, gy)
    return {
        "roughness_grad": float(slope.mean()),
        "slope_p90": float(np.quantile(slope, 0.90)),
        "std_local": float(hn.std()),
    }
