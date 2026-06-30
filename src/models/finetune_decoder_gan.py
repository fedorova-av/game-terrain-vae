"""
finetune_decoder_gan.py — дообучение декодера VAE с PatchGAN дискриминатором.

Loss декодера = L1(recon, real) + lambda_adv * adv_loss + lambda_grad * grad_loss
Loss дискриминатора = BCE(D(real)=1, D(recon)=0)

PatchGAN штрафует за неправдоподобную текстуру на уровне патчей 70×70,
а не за попиксельную ошибку — именно это убирает размытость.

Запуск:
    python -m src.models.finetune_decoder_gan ^
        --vae-ckpt checkpoints/vae_latent256/vae_e1_best.pt ^
        --out checkpoints/decoder_gan ^
        --epochs 100
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
    from src.models.finetune_decoder import (
        cache_recon_pairs, load_vae, eval_roughness, gradient_loss
    )
except Exception as e:
    print(f"Import warning: {e}")
    HeightmapDataset = VAE = None


# ---------------------------------------------------------------------------
# PatchGAN дискриминатор
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN (из pix2pix).
    Принимает heightmap [B, 1, 256, 256], выдаёт карту оценок [B, 1, H', W'].
    Каждый пиксель выхода = оценка правдоподобия патча 70×70 входа.
    BCE по всем пикселям → дискриминатор учится различать реальные/сгенерированные патчи.
    """

    def __init__(self, base: int = 64):
        super().__init__()
        # Архитектура: 5 блоков Conv stride=2, без BatchNorm на первом слое
        def block(in_ch, out_ch, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(1,        base,     norm=False),   # 256 → 128
            *block(base,     base * 2),               # 128 → 64
            *block(base * 2, base * 4),               # 64  → 32
            *block(base * 4, base * 8),               # 32  → 16
            nn.Conv2d(base * 8, 1, 4, stride=1, padding=1),  # 16 → 15
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Loss функции
# ---------------------------------------------------------------------------

def adversarial_loss_g(d_fake: torch.Tensor) -> torch.Tensor:
    """Генератор хочет чтобы D(fake) → 1."""
    return F.binary_cross_entropy_with_logits(
        d_fake, torch.ones_like(d_fake)
    )


def adversarial_loss_d(d_real: torch.Tensor,
                       d_fake: torch.Tensor) -> torch.Tensor:
    """Дискриминатор: D(real)→1, D(fake)→0."""
    loss_real = F.binary_cross_entropy_with_logits(
        d_real, torch.ones_like(d_real)
    )
    loss_fake = F.binary_cross_entropy_with_logits(
        d_fake, torch.zeros_like(d_fake)
    )
    return 0.5 * (loss_real + loss_fake)


def generator_loss(recon: torch.Tensor, real: torch.Tensor,
                   d_fake: torch.Tensor,
                   lambda_l1: float,
                   lambda_adv: float,
                   lambda_grad: float) -> tuple[torch.Tensor, dict]:
    l1   = F.l1_loss(recon, real)
    adv  = adversarial_loss_g(d_fake)
    grad = gradient_loss(recon, real)
    loss = lambda_l1 * l1 + lambda_adv * adv + lambda_grad * grad
    return loss, {
        "l1":   l1.item(),
        "adv":  adv.item(),
        "grad": grad.item(),
    }


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and args.amp
    torch.manual_seed(args.seed)
    print(f"PatchGAN | λ_l1={args.lambda_l1} | λ_adv={args.lambda_adv} "
          f"| λ_grad={args.lambda_grad} | epochs={args.epochs}")

    # --- VAE ---
    print(f"Загружаем VAE из {args.vae_ckpt} ...")
    vae, latent_dim, base = load_vae(args.vae_ckpt, device)
    vae.eval()
    for p in vae.encoder.parameters():
        p.requires_grad_(False)
    for p in vae.decoder.parameters():
        p.requires_grad_(True)
    decoder = vae.decoder
    print(f"  latent_dim={latent_dim} | "
          f"декодер ({sum(p.numel() for p in decoder.parameters())/1e6:.2f}M)")

    # --- Дискриминатор ---
    disc = PatchDiscriminator(base=args.disc_base).to(device)
    print(f"  дискриминатор ({sum(p.numel() for p in disc.parameters())/1e6:.2f}M)")

    # --- Данные ---
    print("Кэшируем пары (train) ...")
    z_train, x_train = cache_recon_pairs(
        vae, args.processed_dir, args.metadata_csv, "train", device
    )
    print("Кэшируем пары (val) ...")
    z_val, x_val = cache_recon_pairs(
        vae, args.processed_dir, args.metadata_csv, "val", device
    )
    print(f"  train={z_train.shape[0]} | val={z_val.shape[0]}")

    train_loader = DataLoader(
        TensorDataset(z_train, x_train),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(z_val, x_val),
        batch_size=args.batch_size, shuffle=False,
    )

    # --- Оптимизаторы ---
    # Декодер и дискриминатор обучаются раздельно
    opt_g = torch.optim.Adam(decoder.parameters(),
                              lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(),
                              lr=args.lr_d, betas=(0.5, 0.999))

    # lr decay: линейное уменьшение во второй половине обучения
    def lr_lambda(epoch):
        decay_start = args.epochs // 2
        if epoch < decay_start:
            return 1.0
        return 1.0 - (epoch - decay_start) / (args.epochs - decay_start + 1e-8)

    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_roughness = eval_roughness(decoder, z_val, device)
    print(f"\nBefore GAN | roughness={base_roughness:.5f} (real=0.03572, цель: ↑)\n")

    best_roughness = base_roughness
    best_val_l1    = float("inf")

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        disc.train()

        agg = {"g": 0.0, "d": 0.0, "l1": 0.0, "adv": 0.0, "grad": 0.0}
        n = 0

        for z_b, x_b in train_loader:
            z_b   = z_b.to(device)
            x_b   = x_b.to(device)
            bs    = z_b.size(0)

            # ── Шаг дискриминатора ──────────────────────────────────────
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                 enabled=use_amp):
                recon = decoder(z_b).detach()   # граф декодера не нужен
                d_real = disc(x_b)
                d_fake = disc(recon)

            loss_d = adversarial_loss_d(d_real.float(), d_fake.float())
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_d.step()

            # ── Шаг генератора (декодера) ────────────────────────────────
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                 enabled=use_amp):
                recon = decoder(z_b)            # свежий forward с графом
                d_fake = disc(recon)

            loss_g, metrics = generator_loss(
                recon.float(), x_b,
                d_fake.float(),
                args.lambda_l1,
                args.lambda_adv,
                args.lambda_grad,
            )
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt_g.step()

            agg["g"]    += loss_g.item() * bs
            agg["d"]    += loss_d.item() * bs
            agg["l1"]   += metrics["l1"]  * bs
            agg["adv"]  += metrics["adv"] * bs
            agg["grad"] += metrics["grad"] * bs
            n += bs

        sched_g.step()
        sched_d.step()

        # --- Val ---
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

        # Сохраняем по roughness, не ухудшая L1 > 30%
        # (GAN допустимо немного ухудшает L1 — поэтому порог шире)
        l1_threshold = best_val_l1 * 1.30 if best_val_l1 < float("inf") else float("inf")
        flag = ""
        if roughness > best_roughness and va_l1_mean < l1_threshold:
            best_roughness = roughness
            best_val_l1    = va_l1_mean
            torch.save(
                {
                    "decoder":    decoder.state_dict(),
                    "disc":       disc.state_dict(),
                    "args":       vars(args),
                    "epoch":      epoch,
                    "roughness":  roughness,
                    "val_l1":     va_l1_mean,
                },
                out_dir / "decoder_gan_best.pt",
            )
            flag = "  <- best"

        print(
            f"epoch {epoch:3d}/{args.epochs} | "
            f"G {agg['g']/n:.4f} "
            f"(L1 {agg['l1']/n:.4f}, adv {agg['adv']/n:.4f}, grad {agg['grad']/n:.4f}) | "
            f"D {agg['d']/n:.4f} | "
            f"val L1 {va_l1_mean:.4f} | roughness {roughness:.5f}{flag}"
        )

    print(f"\nГотово | best roughness: {best_roughness:.5f} | "
          f"best val L1: {best_val_l1:.4f} | "
          f"чекпойнт: {out_dir / 'decoder_gan_best.pt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PatchGAN fine-tuning декодера VAE")
    p.add_argument("--vae-ckpt",      default="checkpoints/vae_latent256/vae_e1_best.pt")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--metadata-csv",  default="data/metadata.csv")
    p.add_argument("--out",           default="checkpoints/decoder_gan")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=16,
                   help="GAN чувствителен к размеру батча — 16 стабильнее чем 32")
    p.add_argument("--lr-g",          type=float, default=1e-4,
                   help="lr генератора (декодера)")
    p.add_argument("--lr-d",          type=float, default=4e-4,
                   help="lr дискриминатора (обычно выше чем у генератора)")
    p.add_argument("--lambda-l1",     type=float, default=100.0,
                   help="вес L1 loss (высокий — сохраняет структуру)")
    p.add_argument("--lambda-adv",    type=float, default=1.0,
                   help="вес adversarial loss")
    p.add_argument("--lambda-grad",   type=float, default=10.0,
                   help="вес gradient loss")
    p.add_argument("--disc-base",     type=int,   default=64,
                   help="базовое число каналов дискриминатора")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--amp",           action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--num-workers",   type=int,   default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
