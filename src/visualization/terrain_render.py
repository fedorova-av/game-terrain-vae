"""
terrain_render.py — рендер heightmap в насыщенную 3D-картинку.
Принципы (против "блёкло/гладко" и против "катышков"):
  - детализация в НОРМАЛЯХ (bump), не в геометрии -> текстура без крошки силуэта
  - 3 источника света: тёплый key (солнце) + холодный небесный ambient + слабый fill
  - AO (низины темнее) -> глубина
  - альбедо по ВЫСОТЕ и КРУТИЗНЕ: трава(низ/пологое) -> камень(круто/средне) -> снег(пики/пологое)
  - вариация альбедо шумом -> не плоско; лёгкая атмосфера + тон-маппинг
"""
import numpy as np
from scipy.ndimage import gaussian_filter, zoom as ndzoom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

rng = np.random.default_rng(7)

# ---------- noise ----------
def value_noise(shape, scale, seed):
    """гладкий многооктавный value-noise в [0,1]."""
    r = np.random.default_rng(seed)
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp, tot = 1.0, 0.0
    f = scale
    for _ in range(5):
        gh, gw = max(2, int(h / f)), max(2, int(w / f))
        g = r.random((gh, gw)).astype(np.float32)
        g = ndzoom(g, (h / gh, w / gw), order=1)[:h, :w]
        g = gaussian_filter(g, sigma=0.8)
        out += amp * g
        tot += amp
        amp *= 0.5
        f *= 0.5
    out /= tot
    out -= out.min(); out /= (out.max() + 1e-9)
    return out

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)

def normalize01(a):
    a = a - a.min(); return a / (a.max() + 1e-9)

# ---------- shading pipeline ----------
def render(height_norm, height_scale_m, *,
           N=220, vert_exag=1.0,
           sun_az=315.0, sun_alt=34.0,
           bump_amp=0.6, detail_scale=14.0,
           snow_level=0.72, grass_level=0.34, rock_slope=0.62,
           ao_strength=0.55, atmo=0.18,
           key_col=(1.0, 0.93, 0.78), sky_col=(0.55, 0.66, 0.85),
           key_int=0.95, sky_int=0.38, sat=1.25, gamma=1.0,
           seed=7):
    # resize -> mesh res
    H = ndzoom(height_norm.astype(np.float32), (N / height_norm.shape[0], N / height_norm.shape[1]), order=3)
    H = H[:N, :N]
    # профиль формы по типу: gamma>1 прижимает низы -> равнина; gamma<1 поднимает
    # верхи -> острые пики. Так flat/hilly/mountain различаются ФОРМОЙ, не только цветом.
    if gamma != 1.0:
        h01 = np.clip((H + 1.0) / 2.0, 0, 1) ** gamma
        H = h01 * 2.0 - 1.0
    t = normalize01(H)                       # 0..1 высотный индекс
    # метрическая высота для геометрии
    cell = 30.0 * (height_norm.shape[0] / N)
    z = (H + 1) / 2 * height_scale_m * vert_exag
    # кромочный скирт: края плавно опускаем к базе, чтобы не было "стенок" плиты
    ii = np.arange(N)
    de = np.minimum.outer(np.minimum(ii, N - 1 - ii), np.ones(N))  # placeholder
    dx_e = np.minimum(ii, N - 1 - ii)
    edge = np.minimum.outer(dx_e, dx_e)
    edge = smoothstep(0, 12, edge)
    z = z * edge + z.min() * (1 - edge)

    # ---- bump-детали ТОЛЬКО для нормалей (не для геометрии) ----
    fine = value_noise((N, N), detail_scale, seed)         # мелкая скальная текстура
    medium = value_noise((N, N), detail_scale * 3, seed + 1)
    bump = (fine - 0.5) * 1.0 + (medium - 0.5) * 0.6
    # больше bump на крутом/высоком (скала шершавая), меньше на траве
    slope_pre = np.hypot(*np.gradient(z, cell))
    slope_pre = normalize01(np.arctan(slope_pre))
    bump_mask = 0.35 + 0.65 * np.clip(slope_pre * 1.3, 0, 1) * (0.5 + 0.5 * t)
    z_bump = z + bump * bump_amp * (height_scale_m / 60.0) * bump_mask * 12.0

    # ---- нормали из (геометрия + bump) ----
    gy, gx = np.gradient(z_bump, cell)
    nz = np.ones_like(gx)
    nlen = np.sqrt(gx * gx + gy * gy + 1) + 1e-9
    nx, ny, nz = -gx / nlen, -gy / nlen, nz / nlen

    # истинная крутизна (по гладкой геометрии) для палитры
    gy2, gx2 = np.gradient(z, cell)
    slope = np.arctan(np.hypot(gx2, gy2))
    slope01 = normalize01(slope)

    # ---- свет ----
    az = np.deg2rad(sun_az); al = np.deg2rad(sun_alt)
    Lx, Ly, Lz = np.cos(al) * np.cos(az), np.cos(al) * np.sin(az), np.sin(al)
    ndotl = np.clip(nx * Lx + ny * Ly + nz * Lz, 0, 1)
    ndotl = ndotl ** 0.85                                  # мягкий терминатор
    sky = np.clip(0.5 + 0.5 * nz, 0, 1)                    # небесный ambient (сверху ярче)
    fill = np.clip(nx * (-Lx) + ny * (-Ly), 0, 1) * 0.25   # слабая подсветка с тени

    key_col  = np.asarray(key_col, np.float32)             # солнце (тёплое/закатное)
    sky_col  = np.asarray(sky_col, np.float32)             # небо (холодное -> синие тени)
    fill_col = np.array([0.7, 0.72, 0.78])
    light = (key_col * (key_int * ndotl)[..., None]
             + sky_col * (sky_int * sky)[..., None]
             + fill_col * fill[..., None])

    # ---- AO: низины/вогнутости темнее ----
    big = gaussian_filter(z, sigma=N * 0.045)
    openness = normalize01(z - big)                        # >0.5 хребты, <0.5 ложбины
    ao = 1.0 - ao_strength * (1 - smoothstep(0.15, 0.75, openness))

    # ---- альбедо: трава -> камень -> снег ----
    grass = np.array([0.30, 0.42, 0.20])
    grass2 = np.array([0.42, 0.50, 0.26])
    rock = np.array([0.46, 0.41, 0.35])
    rock2 = np.array([0.36, 0.32, 0.29])
    snow = np.array([0.92, 0.94, 0.99])
    soil = np.array([0.40, 0.34, 0.24])

    var = value_noise((N, N), 9.0, seed + 5)[..., None]
    # база: трава с вариацией (низ), к середине переход в почву/камень
    alb = grass * (0.7 + 0.6 * var) + (grass2 - grass) * var
    alb = alb * (1 - smoothstep(grass_level, grass_level + 0.18, t)[..., None]) \
        + soil * smoothstep(grass_level, grass_level + 0.18, t)[..., None]
    # камень по высоте
    rock_mix = rock * (0.7 + 0.5 * var) + (rock2 - rock) * (1 - var)
    m_rock = smoothstep(grass_level + 0.1, snow_level, t)[..., None]
    alb = alb * (1 - m_rock) + rock_mix * m_rock
    # снег только на пологом и высоком; крутое остаётся камнем
    snow_mask = smoothstep(snow_level, snow_level + 0.12, t) * (1 - smoothstep(rock_slope, rock_slope + 0.15, slope01))
    snow_mask = snow_mask[..., None]
    alb = alb * (1 - snow_mask) + snow * (0.9 + 0.1 * var) * snow_mask
    # крутые склоны -> камень в любой зоне (трава/снег не держится)
    steep = smoothstep(rock_slope, rock_slope + 0.12, slope01)[..., None]
    alb = alb * (1 - 0.8 * steep) + rock_mix * (0.8 * steep)

    # ---- композит ----
    col = alb * light * ao[..., None]
    # лёгкий блеск на снегу
    spec = (np.clip(nx * Lx + ny * Ly + nz * Lz, 0, 1) ** 40)[..., None] * snow_mask * 0.28
    col = col + spec
    # атмосфера: дальние/высокие чуть уходят в небо
    haze = (atmo * normalize01(z))[..., None]
    col = col * (1 - haze) + sky_col * haze
    # тон-маппинг (борьба с выбеливанием) + лёгкая сатурация
    col = col / (1 + col * 0.25)
    g = col.mean(-1, keepdims=True)
    col = g + (col - g) * sat
    col = np.clip(col, 0, 1) ** (1 / 1.05)
    return np.dstack([col, np.ones((N, N, 1))]).astype(np.float32), z, N


def show3d(height_norm, hscale, path, *, title=None, elev=20, azim=-60, drama=0.52,
           dpi=125, bg="#9fb2cb", **kw):
    rgba, z, N = render(height_norm, hscale, **kw)
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    fig = plt.figure(figsize=(11, 7.2), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.plot_surface(X, Y, z, rcount=N, ccount=N, facecolors=rgba,
                    shade=False, antialiased=False, linewidth=0)
    zr = z.max() - z.min()
    ax.set_box_aspect((1, 1, drama))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_zlim(z.min(), z.min() + max(zr, 1) * 1.02)
    try:
        ax.set_proj_type("persp", focal_length=0.9)
    except Exception:
        pass
    if title:
        ax.set_title(title, color="#22303f", fontsize=13, y=0.96)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("saved", path)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="3D matplotlib-рендер heightmap по классам (grass/rock/snow)")
    ap.add_argument("--heightmaps", default="blender/heightmaps",
                    help="папка с {flat,hilly,mountain}_00.npy")
    ap.add_argument("--out", default="figures/final/terrain3d", help="папка для рендеров")
    ap.add_argument("--hero", action="store_true",
                    help="набор драматичных hero-кадров горы (золотой час), а не by-class")
    args = ap.parse_args()

    base = Path(args.heightmaps)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mtn = np.load(base / "mountain_00.npy")

    if args.hero:
        # Золотой час: низкое тёплое солнце (длинные тени лепят форму), холодное
        # синее небо -> тени синеватые (тёплый свет / холодная тень = «дорогой» кадр).
        golden = dict(sun_alt=20, key_col=(1.0, 0.74, 0.45), sky_col=(0.46, 0.60, 0.95),
                      key_int=1.22, sky_int=0.34, atmo=0.20, bump_amp=0.85, sat=1.32,
                      snow_level=0.70, grass_level=0.32, rock_slope=0.56,
                      vert_exag=2.0, drama=0.6, dpi=165, bg="#c89a72")
        # солнце спереди-сбоку от камеры -> видимые грани освещены (не контровой силуэт)
        shots = [
            dict(name="hero_golden_1", sun_az=305, azim=-62, elev=17),                 # тёплый бок, длинные тени
            dict(name="hero_golden_2", sun_az=255, azim=-104, elev=18),                # другой ракурс, тоже лицом к свету
            dict(name="hero_golden_3", sun_az=340, azim=-30, elev=15, vert_exag=2.25, bg="#b98363"),  # макс. драма
        ]
        for s in shots:
            p = {**golden, **s}; name = p.pop("name")
            show3d(mtn, 1200.0, str(out / f"{name}.png"), **p)
        print(f"done (hero) -> {out}")
    else:
        hil = np.load(base / "hilly_00.npy")
        flt = np.load(base / "flat_00.npy")
        # by-class: трава у подножия, камень на склонах, снег на пиках
        show3d(mtn, 1200.0, str(out / "mountain.png"),
               gamma=0.8, vert_exag=1.9, sun_alt=32, bump_amp=0.75, drama=0.58, elev=22, azim=-62,
               snow_level=0.70, grass_level=0.32, rock_slope=0.58)
        # hilly -> пологие округлые ЗЕЛЁНЫЕ холмы (тёплый мягкий свет, сочная трава)
        show3d(hil, 300.0, str(out / "hilly.png"),
               gamma=1.9, vert_exag=1.7, sun_alt=30, bump_amp=0.45, drama=0.42, elev=24, azim=-62,
               snow_level=0.99, grass_level=0.7, rock_slope=0.82,
               key_col=(1.0, 0.92, 0.74), sky_col=(0.55, 0.68, 0.9), sat=1.45)
        # flat -> сочная травяная равнина с мягкими волнами
        show3d(flt, 60.0, str(out / "flat.png"),
               gamma=2.8, vert_exag=2.3, sun_alt=33, bump_amp=0.3, drama=0.24, elev=20, azim=-62,
               snow_level=1.6, grass_level=0.9, rock_slope=0.97,   # snow_level>1 -> ни скалы, ни снега: чистая трава
               key_col=(1.0, 0.92, 0.74), sky_col=(0.55, 0.68, 0.9), sat=1.45)
        print(f"done -> {out}")
