"""
finetune_decoder.py — дообучение декодера VAE с gradient loss.

Три режима:
  --mode recon   : пары (encoder(x), x) из train split
  --mode diff    : пары (ddim_sample(), nearest_real) из латентного пространства
  --mode both    : чередование батчей recon и diff

Loss = L1(recon, real) + lambda_grad * gradient_loss(recon, real)

Запуск:
    python -m src.models.finetune_decoder --mode recon ^
        --vae-ckpt checkpoints/vae_latent256/vae_e1_best.pt ^
        --diff-ckpt checkpoints/terrain_fusion/diff_best.pt ^
        --out checkpoints/decoder_recon

    python -m src.models.finetune_decoder --mode diff ^
        --vae-ckpt checkpoints/vae_latent256/vae_e1_best.pt ^
        --diff-ckpt checkpoints/terrain_fusion/diff_best.pt ^
        --out checkpoints/decoder_diff

    python -m src.models.finetune_decoder --mode both ^
        --vae-ckpt checkpoints/vae_latent256/vae_e1_best.pt ^
        --diff-ckpt checkpoints/terrain_fusion/diff_best.pt ^
        --out checkpoints/decoder_both
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.data.dataset import HeightmapDataset
    from src.models.vae import VAE
    from src.models.terrain_fusion import (
        LatentDenoiser, make_noise_schedule, ddim_sample
    )
except Exception as e:
    print(f"Import warning: {e}")
    HeightmapDataset = VAE = LatentDenoiser = None


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def gradient_loss(recon: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """
    Штраф за сглаживание градиентов высот.
    Считаем разницу горизонтальных и вертикальных градиентов.
    """
    dx_r = recon[..., :, 1:] - recon[..., :, :-1]
    dy_r = recon[..., 1:, :] - recon[..., :-1, :]
    dx_x = real[...,  :, 1:] - real[...,  :, :-1]
    dy_x = real[...,  1:, :] - real[...,  :-1, :]
    return 0.5 * ((dx_r - dx_x).abs().mean() + (dy_r - dy_x).abs().mean())

def fft_loss(recon: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """
    Штраф в частотном домене — нормализованный по средней амплитуде.
    """
    fft_r = torch.fft.rfft2(recon.float())
    fft_x = torch.fft.rfft2(real.float())
    amp_r = fft_r.abs()
    amp_x = fft_x.abs()
    # Нормализуем на среднюю амплитуду реального спектра
    norm = amp_x.mean().detach() + 1e-8
    return F.l1_loss(amp_r / norm, amp_x / norm)


def combined_loss(recon, real, lambda_grad, lambda_fft=0.0):
    l1   = F.l1_loss(recon, real)
    grad = gradient_loss(recon, real)
    fft  = fft_loss(recon, real) if lambda_fft > 0 else torch.tensor(0.0)
    # L1 только для логирования, не для оптимизации
    loss = lambda_grad * grad + lambda_fft * fft
    return loss, l1.item(), grad.item(), fft.item()


# ---------------------------------------------------------------------------
# Загрузка моделей
# ---------------------------------------------------------------------------

def load_vae(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cargs = ckpt.get("args", {})
    latent_dim = cargs.get("latent_dim", 256)
    base = cargs.get("base", 32)
    vae = VAE(latent_dim, base).to(device)
    vae.load_state_dict(ckpt["model"])
    return vae, latent_dim, base


def load_denoiser(path: str, latent_dim: int, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    dargs = ckpt.get("args", {})
    denoiser = LatentDenoiser(
        latent_dim=latent_dim,
        hidden_dim=dargs.get("hidden_dim", 512),
        n_blocks=dargs.get("n_blocks", 4),
        time_dim=dargs.get("time_dim", 256),
    ).to(device)
    denoiser.load_state_dict(ckpt["model"])
    denoiser.eval()
    for p in denoiser.parameters():
        p.requires_grad_(False)
    return denoiser


# ---------------------------------------------------------------------------
# Подготовка данных
# ---------------------------------------------------------------------------

@torch.no_grad()
def cache_recon_pairs(vae: VAE, processed_dir: str, metadata_csv: str,
                      split: str, device: str,
                      bs: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Прогоняем split через замороженный энкодер.
    Возвращает (z_all [N, latent_dim], x_all [N, 1, 256, 256]).
    """
    ds = HeightmapDataset(
        processed_dir=processed_dir,
        metadata_csv=metadata_csv,
        split=split, return_label=False,
    )
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0,
                        pin_memory=(device == "cuda"))
    zs, xs = [], []
    vae.eval()
    for x in loader:
        x = x.to(device)
        mu, _ = vae.encoder(x)
        zs.append(mu.cpu())
        xs.append(x.cpu())
    return torch.cat(zs), torch.cat(xs)


@torch.no_grad()
def cache_diff_pairs(denoiser: LatentDenoiser, schedule: dict,
                     z_real: torch.Tensor, x_real: torch.Tensor,
                     n: int, latent_dim: int, device: str,
                     ddim_steps: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Генерируем n латентов через DDIM, находим ближайший реальный heightmap
    по L2 в латентном пространстве.
    Возвращает (z_fake [n, latent_dim], x_nearest [n, 1, 256, 256]).
    """
    print(f"  Генерируем {n} латентов через DDIM ({ddim_steps} шагов)...")
    z_fake = ddim_sample(denoiser, schedule, n, latent_dim,
                         device, ddim_steps=ddim_steps).cpu()

    print("  Ищем ближайших соседей в латентном пространстве...")
    # z_real: [N_real, latent_dim], z_fake: [n, latent_dim]
    # Считаем попарные расстояния батчами
    z_real_d = z_real.to(device)
    nearest_x = []
    sub_bs = 128
    for i in range(0, len(z_fake), sub_bs):
        zf = z_fake[i:i + sub_bs].to(device)           # [sub, D]
        dists = torch.cdist(zf, z_real_d)               # [sub, N_real]
        idx = dists.argmin(dim=1).cpu()                 # [sub]
        nearest_x.append(x_real[idx])
    return z_fake, torch.cat(nearest_x)


# ---------------------------------------------------------------------------
# Один шаг обучения
# ---------------------------------------------------------------------------

def train_step(decoder: nn.Module, z: torch.Tensor, x_real: torch.Tensor,
               opt: torch.optim.Optimizer, device: str,
               lambda_grad: float, lambda_fft: float, use_amp: bool):
    z = z.to(device)
    x_real = x_real.to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16,
                        enabled=use_amp):
        recon = decoder(z)
    loss, l1_val, grad_val, fft_val = combined_loss(
        recon.float(), x_real, lambda_grad, lambda_fft
    )
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
    opt.step()
    return loss.item(), l1_val, grad_val, fft_val


# ---------------------------------------------------------------------------
# Оценка roughness (главная метрика чёткости)
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_roughness(decoder: nn.Module, z_val: torch.Tensor,
                   device: str, bs: int = 64) -> float:
    """Средний roughness (Лапласиан) сгенерированных плиток."""
    decoder.eval()
    roughs = []
    for i in range(0, len(z_val), bs):
        zb = z_val[i:i + bs].to(device)
        recon = decoder(zb).float()
        lap = (
            -4.0 * recon[..., 1:-1, 1:-1]
            + recon[..., :-2, 1:-1] + recon[..., 2:,  1:-1]
            + recon[..., 1:-1, :-2] + recon[..., 1:-1, 2:]
        )
        roughs.append(lap.abs().mean(dim=(1, 2, 3)).cpu())
    decoder.train()
    return float(torch.cat(roughs).mean())


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and args.amp
    torch.manual_seed(args.seed)
    print(f"Режим: {args.mode} | λ_grad={args.lambda_grad} | λ_fft={args.lambda_fft} | epochs={args.epochs}")

    # --- Загружаем VAE ---
    print(f"Загружаем VAE из {args.vae_ckpt} ...")
    vae, latent_dim, base = load_vae(args.vae_ckpt, device)

    # Замораживаем энкодер, декодер будем дообучать
    vae.eval()
    for p in vae.encoder.parameters():
        p.requires_grad_(False)
    for p in vae.decoder.parameters():
        p.requires_grad_(True)
    decoder = vae.decoder
    print(f"  latent_dim={latent_dim} | "
          f"декодер разморожен ({sum(p.numel() for p in decoder.parameters())/1e6:.2f}M)")

    # --- Загружаем денойзер (нужен для режимов diff и both) ---
    denoiser, schedule = None, None
    if args.mode in ("diff", "both"):
        print(f"Загружаем денойзер из {args.diff_ckpt} ...")
        denoiser = load_denoiser(args.diff_ckpt, latent_dim, device)
        schedule = make_noise_schedule(args.T, device=device)

    # --- Кэшируем пары (один раз) ---
    print("Кэшируем реконструкционные пары (train) ...")
    z_recon, x_recon = cache_recon_pairs(
        vae, args.processed_dir, args.metadata_csv, "train", device
    )
    print(f"  train: z={z_recon.shape}, x={x_recon.shape}")

    print("Кэшируем реконструкционные пары (val) ...")
    z_val, x_val = cache_recon_pairs(
        vae, args.processed_dir, args.metadata_csv, "val", device
    )
    print(f"  val:   z={z_val.shape}, x={x_val.shape}")

    z_diff, x_diff = None, None
    if args.mode in ("diff", "both"):
        print("Кэшируем диффузионные пары ...")
        z_diff, x_diff = cache_diff_pairs(
            denoiser, schedule, z_recon, x_recon,
            n=args.n_diff, latent_dim=latent_dim,
            device=device, ddim_steps=args.ddim_steps,
        )
        print(f"  diff:  z={z_diff.shape}, x={x_diff.shape}")

    # --- DataLoader'ы ---
    recon_loader = DataLoader(
        TensorDataset(z_recon, x_recon),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    diff_loader = None
    if z_diff is not None:
        diff_loader = DataLoader(
            TensorDataset(z_diff, x_diff),
            batch_size=args.batch_size, shuffle=True, drop_last=True,
        )
    val_loader = DataLoader(
        TensorDataset(z_val, x_val),
        batch_size=args.batch_size, shuffle=False,
    )

    # --- Оптимизатор ---
    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Базовый roughness до дообучения
    base_roughness = eval_roughness(decoder, z_val, device)
    print(f"\nBefore fine-tuning | roughness={base_roughness:.5f} "
          f"(real=0.03572, цель: ↑)")

    best_roughness = base_roughness
    best_val_l1 = float("inf")

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        tr_loss, tr_l1, tr_grad, tr_fft, tr_n = 0.0, 0.0, 0.0, 0.0, 0

        if args.mode == "recon":
            for z_b, x_b in recon_loader:
                loss, l1_v, grad_v, fft_v = train_step(
                    decoder, z_b, x_b, opt, device,
                    args.lambda_grad, args.lambda_fft, use_amp
                )
                bs = z_b.size(0)
                tr_loss += loss * bs; tr_l1  += l1_v   * bs
                tr_grad += grad_v * bs; tr_fft += fft_v * bs
                tr_n += bs

        elif args.mode == "diff":
            for z_b, x_b in diff_loader:
                loss, l1_v, grad_v, fft_v = train_step(
                    decoder, z_b, x_b, opt, device,
                    args.lambda_grad, args.lambda_fft, use_amp
                )
                bs = z_b.size(0)
                tr_loss += loss * bs; tr_l1  += l1_v   * bs
                tr_grad += grad_v * bs; tr_fft += fft_v * bs
                tr_n += bs

        else:
            # both: чередуем батчи recon и diff
            diff_iter = iter(diff_loader)
            for z_b, x_b in recon_loader:
                # recon батч
                loss, l1_v, grad_v, fft_v = train_step(
                    decoder, z_b, x_b, opt, device,
                    args.lambda_grad, args.lambda_fft, use_amp
                )
                bs = z_b.size(0)
                tr_loss += loss * bs; tr_l1  += l1_v   * bs
                tr_grad += grad_v * bs; tr_fft += fft_v * bs
                tr_n += bs
                # diff батч
                try:
                    z_d, x_d = next(diff_iter)
                except StopIteration:
                    diff_iter = iter(diff_loader)
                    z_d, x_d = next(diff_iter)
                loss, l1_v, grad_v, fft_v = train_step(
                    decoder, z_d, x_d, opt, device,
                    args.lambda_grad, args.lambda_fft, use_amp
                )
                bs = z_d.size(0)
                tr_loss += loss * bs; tr_l1  += l1_v   * bs
                tr_grad += grad_v * bs; tr_fft += fft_v * bs
                tr_n += bs

        scheduler.step()

        # --- Val L1 ---
        decoder.eval()
        va_l1, va_n = 0.0, 0
        with torch.no_grad():
            for z_b, x_b in val_loader:
                z_b, x_b = z_b.to(device), x_b.to(device)
                recon = decoder(z_b)
                va_l1 += F.l1_loss(recon.float(), x_b).item() * z_b.size(0)
                va_n  += z_b.size(0)

        va_l1_mean = va_l1 / max(va_n, 1)
        roughness  = eval_roughness(decoder, z_val, device)

        # Сохраняем по roughness + не ухудшаем L1 > 20%
        l1_threshold = best_val_l1 * 1.20 if best_val_l1 < float("inf") else float("inf")
        flag = ""
        if roughness > best_roughness and va_l1_mean < l1_threshold:
            best_roughness = roughness
            best_val_l1    = va_l1_mean
            torch.save(
                {
                    "decoder":   decoder.state_dict(),
                    "args":      vars(args),
                    "epoch":     epoch,
                    "roughness": roughness,
                    "val_l1":    va_l1_mean,
                },
                out_dir / f"decoder_{args.mode}_best.pt",
            )
            flag = "  <- best"

        print(
            f"epoch {epoch:3d}/{args.epochs} | "
            f"loss {tr_loss/max(tr_n,1):.4f} "
            f"(L1 {tr_l1/max(tr_n,1):.4f}, "
            f"grad {tr_grad/max(tr_n,1):.4f}, "
            f"fft {tr_fft/max(tr_n,1):.4f}) | "
            f"val L1 {va_l1_mean:.4f} | roughness {roughness:.5f}{flag}"
        )

    print(f"\nГотово [{args.mode}] | "
          f"best roughness: {best_roughness:.5f} | "
          f"best val L1: {best_val_l1:.4f} | "
          f"чекпойнт: {out_dir / f'decoder_{args.mode}_best.pt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tuning декодера VAE")
    p.add_argument("--mode",          default="recon",
                   choices=["recon", "diff", "both"],
                   help="recon=реконструкции, diff=диффузия, both=оба")
    p.add_argument("--vae-ckpt",      default="checkpoints/vae_latent256/vae_e1_best.pt")
    p.add_argument("--diff-ckpt",     default="checkpoints/terrain_fusion/diff_best.pt")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--metadata-csv",  default="data/metadata.csv")
    p.add_argument("--out",           default="checkpoints/decoder_finetune")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-4,
                   help="маленький lr — декодер уже предобучен")
    p.add_argument("--lambda-grad",   type=float, default=0.5,
                   help="вес gradient loss")
    p.add_argument("--T",             type=int,   default=1000)
    p.add_argument("--ddim-steps",    type=int,   default=50)
    p.add_argument("--n-diff",        type=int,   default=5000,
                   help="сколько диффузионных пар кэшировать")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--amp",           action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--num-workers",   type=int,   default=0)
    p.add_argument("--lambda-fft", type=float, default=0.1,
               help="вес frequency loss (0 = отключён)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
