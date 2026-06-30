"""
terrain_fusion.py — TerraFusion: DDPM в латентном пространстве VAE (эксперимент E1-C).

Схема:
  Обучение : реальная плитка → VAE encoder (заморожен) → z[256]
             → зашумить до z_t → денойзер предсказывает шум → MSE loss
  Генерация: z_T ~ N(0,I) → денойзер (1000 шагов DDIM) → z_0
             → VAE decoder (заморожен) → heightmap 256×256

Запуск:
    # smoke-тест (без данных и чекпойнта)
    python -m src.models.terrain_fusion --smoke

    # обучение
    python -m src.models.terrain_fusion --vae-ckpt checkpoints/vae_latent256/vae_e1_best.pt

    # генерация (после обучения)
    python -m src.models.terrain_fusion --generate --n-samples 8 ^
        --vae-ckpt checkpoints/vae_latent256/vae_e1_best.pt ^
        --diff-ckpt checkpoints/terrain_fusion/diff_best.pt ^
        --out-fig results/terrain_fusion_samples.png
"""

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.data.dataset import HeightmapDataset
    from src.models.vae import VAE
except Exception:
    HeightmapDataset = None
    VAE = None


# ---------------------------------------------------------------------------
# 1. DDPM: расписание шумов
# ---------------------------------------------------------------------------

def make_noise_schedule(T: int = 1000, beta_start: float = 1e-4,
                         beta_end: float = 0.02, device="cpu"):
    """
    Линейное расписание β_1..β_T.
    Возвращает словарь тензоров, нужных для forward/reverse процесса.
    """
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)          # ᾱ_t = ∏ α_i
    alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

    return {
        "betas":          betas,
        "alphas":         alphas,
        "alpha_bar":      alpha_bar,
        "alpha_bar_prev": alpha_bar_prev,
        "sqrt_alpha_bar":       alpha_bar.sqrt(),
        "sqrt_one_minus_ab":    (1.0 - alpha_bar).sqrt(),
        "posterior_var":  betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar),
    }


# ---------------------------------------------------------------------------
# 2. Денойзер: простой MLP для 256-мерного латента
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Синусоидальное позиционное кодирование шага t → вектор dim."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] int или float
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        return torch.cat([args.sin(), args.cos()], dim=1)    # [B, dim]


class LatentDenoiser(nn.Module):
    """
    MLP-денойзер: принимает (z_t, t) → предсказывает шум ε.

    Архитектура: 4 residual блока, каждый — Linear → SiLU → Linear,
    с добавлением временного эмбеддинга на входе каждого блока.
    Латентное пространство 256D — небольшое, MLP достаточно.
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512,
                 n_blocks: int = 4, time_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_emb = SinusoidalEmbedding(time_dim)

        # Проекция входа: z_t + time_emb → hidden
        self.input_proj = nn.Linear(latent_dim + time_dim, hidden_dim)

        # Residual блоки
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_blocks)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_blocks)])

        # Выход → предсказание шума той же размерности что z
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z_t: [B, latent_dim]  — зашумлённый латент
        t:   [B]              — шаг диффузии (int, 0..T-1)
        → [B, latent_dim]     — предсказанный шум ε
        """
        te = self.time_emb(t)                        # [B, time_dim]
        h = self.input_proj(torch.cat([z_t, te], dim=1))  # [B, hidden]
        for block, norm in zip(self.blocks, self.norms):
            h = h + block(norm(h))                   # residual
        return self.out(h)


# ---------------------------------------------------------------------------
# 3. Forward process: зашумление q(z_t | z_0)
# ---------------------------------------------------------------------------

def q_sample(z0: torch.Tensor, t: torch.Tensor,
             schedule: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Зашумляем z0 до уровня t за один шаг (closed-form).
    z_t = √ᾱ_t · z0 + √(1-ᾱ_t) · ε,  ε ~ N(0,I)
    Возвращает (z_t, ε).
    """
    eps = torch.randn_like(z0)
    sqrt_ab  = schedule["sqrt_alpha_bar"][t].view(-1, 1)
    sqrt_omab = schedule["sqrt_one_minus_ab"][t].view(-1, 1)
    z_t = sqrt_ab * z0 + sqrt_omab * eps
    return z_t, eps


# ---------------------------------------------------------------------------
# 4. Reverse process: DDIM сэмплинг (быстрее DDPM, детерминированный)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(denoiser: LatentDenoiser, schedule: dict,
                n: int, latent_dim: int, device: str,
                ddim_steps: int = 50, eta: float = 0.0) -> torch.Tensor:
    """
    DDIM сэмплинг за ddim_steps шагов (по умолчанию 50, вместо 1000).
    eta=0 → детерминированный; eta=1 → стохастический (≈ DDPM).
    Возвращает z_0: [n, latent_dim].
    """
    T = len(schedule["betas"])
    # Равномерно выбираем ddim_steps временных шагов
    step_seq = torch.linspace(T - 1, 0, ddim_steps, dtype=torch.long, device=device)

    z = torch.randn(n, latent_dim, device=device)   # z_T ~ N(0,I)
    denoiser.eval()

    for i, t_val in enumerate(step_seq):
        t_batch = t_val.expand(n)
        eps_pred = denoiser(z, t_batch)

        ab_t   = schedule["alpha_bar"][t_val]
        ab_prev = schedule["alpha_bar"][step_seq[i + 1]] if i + 1 < ddim_steps \
                  else torch.tensor(1.0, device=device)

        # DDIM update
        z0_pred = (z - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
        z0_pred = z0_pred.clamp(-3, 3)   # мягкий клип для стабильности
        dir_xt  = (1 - ab_prev - eta ** 2 * (1 - ab_t) / (1 - ab_prev + 1e-8)).clamp(0).sqrt() * eps_pred
        noise   = eta * (1 - ab_prev).sqrt() * torch.randn_like(z) if eta > 0 else 0
        z = ab_prev.sqrt() * z0_pred + dir_xt + noise

    return z


# ---------------------------------------------------------------------------
# 5. Кэширование латентов (чтобы не гнать VAE на каждой эпохе)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_dataset(vae: VAE, loader: DataLoader,
                   device: str) -> torch.Tensor:
    """Прогоняем весь split через замороженный VAE encoder, возвращаем μ."""
    vae.eval()
    mus = []
    for batch in loader:
        x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        mu, _ = vae.encoder(x)
        mus.append(mu.cpu())
    return torch.cat(mus, dim=0)   # [N, latent_dim]


# ---------------------------------------------------------------------------
# 6. Обучение денойзера
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"Устройство: {device} | T={args.T} | latent_dim={args.latent_dim}")

    # --- Загружаем замороженный VAE ---
    print(f"Загружаем VAE из {args.vae_ckpt} ...")
    ckpt = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    cargs = ckpt.get("args", {})
    latent_dim = cargs.get("latent_dim", args.latent_dim)
    vae = VAE(latent_dim, cargs.get("base", 32)).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"VAE загружен (latent_dim={latent_dim}). Параметры заморожены.")

    # --- Кэшируем латенты (один раз) ---
    def raw_loader(split, shuffle):
        ds = HeightmapDataset(
            processed_dir=args.processed_dir,
            metadata_csv=args.metadata_csv,
            split=split, return_label=False,
        )
        return DataLoader(ds, batch_size=64, shuffle=shuffle,
                          num_workers=args.num_workers,
                          pin_memory=(device == "cuda"))

    print("Кэшируем латенты train/val ...")
    z_train = encode_dataset(vae, raw_loader("train", False), device)
    z_val   = encode_dataset(vae, raw_loader("val",   False), device)
    print(f"  train: {z_train.shape} | val: {z_val.shape}")

    train_loader = DataLoader(TensorDataset(z_train), batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(TensorDataset(z_val),   batch_size=args.batch_size,
                              shuffle=False)

    # --- Расписание шумов ---
    schedule = make_noise_schedule(args.T, device=device)

    # --- Денойзер ---
    denoiser = LatentDenoiser(
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        n_blocks=args.n_blocks,
        time_dim=args.time_dim,
    ).to(device)
    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Денойзер: {n_params / 1e6:.2f}M параметров")

    # СТАЛО
    opt = torch.optim.AdamW(denoiser.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Resume из существующего чекпойнта
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        res = torch.load(args.resume, map_location=device, weights_only=False)
        denoiser.load_state_dict(res["model"])
        if "opt" in res:
            opt.load_state_dict(res["opt"])
        if "scheduler" in res:
            scheduler.load_state_dict(res["scheduler"])
        start_epoch = res.get("epoch", 0) + 1
        best_val = res.get("val_loss", float("inf"))
        print(f"Resume с эпохи {start_epoch}, best_val={best_val:.5f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        # --- Train ---
        denoiser.train()
        tr_loss, tr_n = 0.0, 0
        for (z0,) in train_loader:
            z0 = z0.to(device)
            t = torch.randint(0, args.T, (z0.size(0),), device=device)
            z_t, eps = q_sample(z0, t, schedule)
            eps_pred = denoiser(z_t, t)
            loss = F.mse_loss(eps_pred, eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * z0.size(0)
            tr_n    += z0.size(0)
        scheduler.step()

        # --- Val ---
        denoiser.eval()
        va_loss, va_n = 0.0, 0
        with torch.no_grad():
            for (z0,) in val_loader:
                z0 = z0.to(device)
                t = torch.randint(0, args.T, (z0.size(0),), device=device)
                z_t, eps = q_sample(z0, t, schedule)
                eps_pred = denoiser(z_t, t)
                loss = F.mse_loss(eps_pred, eps)
                va_loss += loss.item() * z0.size(0)
                va_n    += z0.size(0)

        tr_mean = tr_loss / max(tr_n, 1)
        va_mean = va_loss / max(va_n, 1)
        flag = ""
        if va_mean < best_val:
            best_val = va_mean
            torch.save({"model": denoiser.state_dict(), "args": vars(args),
                        "epoch": epoch, "val_loss": va_mean,
                        "latent_dim": latent_dim,
                        "opt": opt.state_dict(),
                        "scheduler": scheduler.state_dict()},
                out_dir / "diff_best.pt",)
            flag = "  <- best"
        print(f"epoch {epoch:3d}/{args.epochs} | "
              f"train {tr_mean:.5f} | val {va_mean:.5f}{flag}")

    print(f"Готово. Лучший val loss: {best_val:.5f}")


# ---------------------------------------------------------------------------
# 7. Генерация: DDIM → VAE decoder → фигура
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(args):
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Загружаем VAE
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    cargs    = vae_ckpt.get("args", {})
    latent_dim = cargs.get("latent_dim", args.latent_dim)
    vae = VAE(latent_dim, cargs.get("base", 32)).to(device)
    vae.load_state_dict(vae_ckpt["model"])
    vae.eval()

    # Загружаем денойзер
    diff_ckpt  = torch.load(args.diff_ckpt, map_location=device, weights_only=False)
    dargs      = diff_ckpt.get("args", {})
    denoiser = LatentDenoiser(
        latent_dim=latent_dim,
        hidden_dim=dargs.get("hidden_dim", 512),
        n_blocks=dargs.get("n_blocks",   4),
        time_dim=dargs.get("time_dim",   256),
    ).to(device)
    denoiser.load_state_dict(diff_ckpt["model"])
    denoiser.eval()

    schedule = make_noise_schedule(args.T, device=device)

    print(f"Генерируем {args.n_samples} сэмплов ({args.ddim_steps} DDIM шагов) ...")
    z0 = ddim_sample(denoiser, schedule, args.n_samples, latent_dim,
                     device, ddim_steps=args.ddim_steps, eta=args.eta)
    samples = vae.decoder(z0).cpu()   # [N, 1, 256, 256]

    cols = min(args.n_samples, 4)
    rows = (args.n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.8 * cols, 2.8 * rows))
    import numpy as np
    axes = np.array(axes).reshape(rows, cols)
    for k in range(rows * cols):
        ax = axes[k // cols, k % cols]
        ax.set_xticks([]); ax.set_yticks([])
        if k < args.n_samples:
            ax.imshow(samples[k, 0].numpy(), cmap="terrain")
        else:
            ax.axis("off")
    fig.suptitle(f"TerraFusion: генерация (DDIM {args.ddim_steps} шагов)")
    fig.tight_layout()
    out_path = Path(args.out_fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {out_path}")


# ---------------------------------------------------------------------------
# 8. Smoke test
# ---------------------------------------------------------------------------

def smoke():
    print("=== Smoke test ===")
    device = "cpu"
    T, latent_dim, B = 1000, 256, 4

    schedule = make_noise_schedule(T, device=device)
    print(f"Расписание: betas {tuple(schedule['betas'].shape)}, "
          f"alpha_bar [{schedule['alpha_bar'][0]:.4f} .. {schedule['alpha_bar'][-1]:.6f}]")

    denoiser = LatentDenoiser(latent_dim=latent_dim)
    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Денойзер: {n_params/1e6:.2f}M параметров")

    z0  = torch.randn(B, latent_dim)
    t   = torch.randint(0, T, (B,))
    z_t, eps = q_sample(z0, t, schedule)
    eps_pred = denoiser(z_t, t)
    loss = F.mse_loss(eps_pred, eps)

    print(f"z0:       {tuple(z0.shape)}")
    print(f"z_t:      {tuple(z_t.shape)}")
    print(f"eps_pred: {tuple(eps_pred.shape)}")
    print(f"loss:     {loss.item():.4f}")

    z_gen = ddim_sample(denoiser, schedule, n=2, latent_dim=latent_dim,
                        device=device, ddim_steps=5)
    print(f"DDIM (5 шагов): {tuple(z_gen.shape)}")
    assert z_gen.shape == (2, latent_dim)
    print("OK: все формы сходятся")


# ---------------------------------------------------------------------------
# 9. CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TerraFusion: DDPM в латентном пространстве VAE")
    p.add_argument("--smoke",    action="store_true", help="быстрый тест без данных")
    p.add_argument("--generate", action="store_true", help="генерация из обученного чекпойнта")

    # Пути
    p.add_argument("--vae-ckpt",      default="checkpoints/vae_latent256/vae_e1_best.pt")
    p.add_argument("--diff-ckpt",     default="checkpoints/terrain_fusion/diff_best.pt")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--metadata-csv",  default="data/metadata.csv")
    p.add_argument("--out",           default="checkpoints/terrain_fusion")
    p.add_argument("--out-fig",       default="results/terrain_fusion_samples.png")

    # Диффузия
    p.add_argument("--T",           type=int,   default=1000)
    p.add_argument("--latent-dim",  type=int,   default=256)
    p.add_argument("--hidden-dim",  type=int,   default=512)
    p.add_argument("--n-blocks",    type=int,   default=4)
    p.add_argument("--time-dim",    type=int,   default=256)
    p.add_argument("--ddim-steps",  type=int,   default=50,
                   help="шагов DDIM при генерации (меньше = быстрее)")
    p.add_argument("--eta",         type=float, default=0.0,
                   help="0=детерминированный DDIM, 1=стохастический")

    # Обучение
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch-size",  type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num-workers", type=int,   default=0)
    p.add_argument("--resume", default=None,
               help="путь к чекпойнту для продолжения обучения")

    # Генерация
    p.add_argument("--n-samples",   type=int,   default=8)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        smoke()
    elif args.generate:
        generate(args)
    else:
        train(args)
