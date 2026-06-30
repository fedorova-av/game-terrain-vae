"""
vae_unet.py - VAE с U-Net декодером (эксперимент E1-B).

Запуск:
    python -m src.models.vae_unet --smoke
    python -m src.models.vae_unet --epochs 80 --out checkpoints/vae_unet
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from src.data.dataset import HeightmapDataset
except Exception:
    HeightmapDataset = None

# Энкодер: сохраняет skip-карты с каждого уровня

class UNetEncoder(nn.Module):
    """
    6 уровней downsampling (stride=2), выход каждого уровня сохраняется
    как skip connection для декодера.

    Каналы: 1 → 32 → 64 → 128 → 256 → 256 → 512
    Пространство: 256 → 128 → 64 → 32 → 16 → 8 → 4
    """

    def __init__(self, latent_dim: int = 128, base: int = 32):
        super().__init__()
        chs = [1, base, base * 2, base * 4, base * 8, base * 8, base * 16]
        self.levels = nn.ModuleList()
        for i in range(6):
            block = nn.Sequential(
                nn.Conv2d(chs[i], chs[i + 1], 4, stride=2, padding=1),
                nn.BatchNorm2d(chs[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.levels.append(block)

        flat_dim = chs[6] * 4 * 4  # 512 * 4 * 4 = 8192
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.out_chs = chs[1:]  # [32, 64, 128, 256, 256, 512]
        self.skip_drop = nn.Dropout2d(p=0.5)

    def forward(self, x):
        skips = []
        h = x
        for level in self.levels:
            h = level(h)
            skips.append(h)
        flat = skips[-1].flatten(1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        dropped = [self.skip_drop(s) if self.training else s for s in skips[:-1]]
        return mu, logvar, dropped


# ---------------------------------------------------------------------------
# Декодер: принимает skip-карты и конкатенирует их на каждом уровне
# ---------------------------------------------------------------------------

class UNetDecoder(nn.Module):
    """
    Зеркален энкодеру. На каждом уровне апсемплинга результат ConvTranspose
    конкатенируется со skip-картой энкодера того же разрешения, затем
    проходит через 1×1 Conv для уменьшения числа каналов.

    Уровни (снизу вверх):
      4×4×512  → fc  →
      4→8:   ConvTranspose → 256 + skip(256) → Conv1x1 → 256
      8→16:  ConvTranspose → 256 + skip(256) → Conv1x1 → 256
      16→32: ConvTranspose → 128 + skip(128) → Conv1x1 → 128
      32→64: ConvTranspose → 64  + skip(64)  → Conv1x1 → 64
      64→128:ConvTranspose → 32  + skip(32)  → Conv1x1 → 32
      128→256:ConvTranspose → 32 (последний уровень, skip нет)
      → out Conv 3×3 → Tanh
    """

    def __init__(self, latent_dim: int = 128, base: int = 32):
        super().__init__()
        self.start_ch = base * 16   # 512
        self.fc = nn.Linear(latent_dim, self.start_ch * 4 * 4)

        # Каналы декодера (вход ConvTranspose на каждом уровне)
        up_in  = [base * 16, base * 8, base * 8, base * 4, base * 2, base]
        up_out = [base * 8,  base * 8, base * 4, base * 2, base,     base]
        # skip-каналы (от энкодера, снизу вверх): уровни 4,3,2,1,0
        # skip_chs[i] соответствует up_in[i+1]
        skip_chs = [base * 8, base * 8, base * 4, base * 2, base]  # 5 skip'ов

        self.up_blocks = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()


        for i in range(6):
            up = nn.Sequential(
                nn.ConvTranspose2d(up_in[i], up_out[i], 4, stride=2, padding=1),
                nn.BatchNorm2d(up_out[i]),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.up_blocks.append(up)

            if i < 5:
                # После конкатенации каналов: up_out[i] + skip_chs[i]
                fuse_in = up_out[i] + skip_chs[i]
                fuse = nn.Sequential(
                    nn.Conv2d(fuse_in, up_out[i], 1),   # 1×1 conv для слияния
                    nn.BatchNorm2d(up_out[i]),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                self.fuse_convs.append(fuse)
            else:
                self.fuse_convs.append(None)   # последний уровень без skip

        self.out = nn.Conv2d(base, 1, 3, padding=1)
        self.act = nn.Tanh()

    def forward(self, z, skips=None):
        """
        z:     [B, latent_dim]
        skips: список из 5 тензоров [уровень 0..4 энкодера], порядок
               от мелкого к крупному разрешению (128, 64, 32, 16, 8 px).
               Для генерации из приора (skips=None) работает как обычный декодер.
        """
        h = self.fc(z).view(-1, self.start_ch, 4, 4)

        for i, (up, fuse) in enumerate(zip(self.up_blocks, self.fuse_convs)):
            h = up(h)
            if skips is not None and fuse is not None:
                # skip'ы в обратном порядке: уровень 4 → 3 → 2 → 1 → 0
                skip = skips[-(i + 1)]
                h = torch.cat([h, skip], dim=1)
                h = fuse(h)

        return self.act(self.out(h))


# ---------------------------------------------------------------------------
# Модель целиком
# ---------------------------------------------------------------------------

class UNetVAE(nn.Module):
    """
    VAE с U-Net декодером.

    Интерфейс намеренно совместим с VAE из vae.py:
      - forward(x)          → (recon, mu, logvar)
      - encoder(x)          → (mu, logvar)     ← нужен eval_models.py
      - decoder(z)          → recon            ← генерация из приора
      - sample(n, device)   → сэмплы из N(0,I)
    """

    def __init__(self, latent_dim: int = 128, base: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self._encoder = UNetEncoder(latent_dim, base)
        self._decoder = UNetDecoder(latent_dim, base)

    # --- публичный интерфейс (совместим с eval_models.py) ---

    def encoder(self, x):
        """Возвращает (mu, logvar) — без skip'ов, для eval."""
        mu, logvar, _ = self._encoder(x)
        return mu, logvar

    def decoder(self, z):
        """Декодирует z без skip'ов (генерация из приора)."""
        return self._decoder(z, skips=None)

    # --- внутренний forward с skip'ами ---

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar, skips = self._encoder(x)
        logvar = logvar.clamp(-10, 10)
        z = self.reparameterize(mu, logvar)
        recon = self._decoder(z, skips=skips)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Loss (идентична vae.py)
# ---------------------------------------------------------------------------

def vae_loss(recon, x, mu, logvar, beta):
    recon_term = F.l1_loss(recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    loss = recon_term + beta * kl
    l1_pixel = F.l1_loss(recon, x, reduction="mean")
    return loss, recon_term, kl, l1_pixel


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

def make_loader(split, args, shuffle):
    ds = HeightmapDataset(
        processed_dir=args.processed_dir,
        metadata_csv=args.metadata_csv,
        split=split,
        return_label=False,
    )
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, use_amp, beta, opt=None, clip=None):
    train = opt is not None
    model.train(train)
    agg = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "l1": 0.0}
    n = 0
    max_gnorm = 0.0
    scaler = torch.amp.GradScaler(device, enabled=False)
    torch.set_grad_enabled(train)
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            recon, mu, logvar = model(x)
        loss, recon_t, kl, l1 = vae_loss(recon.float(), x, mu.float(), logvar.float(), beta)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip is not None:
                scaler.unscale_(opt)
                g = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                max_gnorm = max(max_gnorm, float(g))
            scaler.step(opt)
            scaler.update()
        bs = x.size(0)
        agg["loss"] += loss.item() * bs
        agg["recon"] += recon_t.item() * bs
        agg["kl"]    += kl.item() * bs
        agg["l1"]    += l1.item() * bs
        n += bs
    torch.set_grad_enabled(True)
    out = {k: v / max(n, 1) for k, v in agg.items()}
    out["gnorm"] = max_gnorm
    return out


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and args.amp
    torch.manual_seed(args.seed)
    print(f"Устройство: {device} | AMP: {use_amp} | beta={args.beta} | latent_dim={args.latent_dim}")

    model = UNetVAE(args.latent_dim, args.base).to(device)
    print(f"Параметров: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = make_loader("train", args, True)
    val_loader   = make_loader("val",   args, False)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, use_amp, args.beta, opt, clip=args.clip)
        va = run_epoch(model, val_loader,   device, use_amp, args.beta)
        flag = ""
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                 "val_loss": va["loss"], "val_l1_pixel": va["l1"]},
                out_dir / "vae_unet_best.pt",
            )
            flag = "  <- best"
        print(
            f"epoch {epoch:3d}/{args.epochs} | "
            f"train loss {tr['loss']:.1f} (recon {tr['recon']:.1f}, KL {tr['kl']:.2f}) | "
            f"val L1px {va['l1']:.4f} | val KL {va['kl']:.2f} | "
            f"gnorm {tr['gnorm']:.1f}{flag}"
        )

    print(f"Готово. Лучший val loss: {best_val:.1f} | чекпойнт: {out_dir / 'vae_unet_best.pt'}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke():
    model = UNetVAE()
    x = torch.randn(2, 1, 256, 256)
    recon, mu, logvar = model(x)
    s = model.sample(2, "cpu")
    mu2, logvar2 = model.encoder(x)   # совместимость с eval_models
    dec = model.decoder(mu2)           # совместимость с eval_models

    print(f"вход:          {tuple(x.shape)}")
    print(f"mu / logvar:   {tuple(mu.shape)} / {tuple(logvar.shape)}")
    print(f"recon:         {tuple(recon.shape)}")
    print(f"sample (приор):{tuple(s.shape)}")
    print(f"encoder():     {tuple(mu2.shape)}  ← совместим с eval_models.py")
    print(f"decoder(mu):   {tuple(dec.shape)}  ← совместим с eval_models.py")
    print(f"параметров:    {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    assert recon.shape == x.shape
    assert s.shape == x.shape
    assert mu2.shape == (2, 128)
    assert dec.shape == x.shape
    print("OK: все формы сходятся, интерфейс совместим с eval_models.py")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="VAE с U-Net декодером (E1-B)")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--metadata-csv",  default="data/metadata.csv")
    p.add_argument("--out",           default="checkpoints/vae_unet")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--latent-dim",    type=int,   default=128)
    p.add_argument("--base",          type=int,   default=32)
    p.add_argument("--beta",          type=float, default=1.0)
    p.add_argument("--num-workers",   type=int,   default=0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--amp",           action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clip",          type=float, default=200000.0)
    p.add_argument("--smoke",         action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        smoke()
    else:
        train(args)
