"""
vae.py - Vanilla VAE (эксперимент E1).

Запуск:
    python -m src.models.vae --epochs 80
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.ae import Decoder

try:
    from src.data.dataset import HeightmapDataset
except Exception:
    HeightmapDataset = None


class VAEEncoder(nn.Module):

    def __init__(self, latent_dim: int = 128, base: int = 32):
        super().__init__()
        chs = [1, base, base * 2, base * 4, base * 8, base * 8, base * 16]
        blocks = []
        for i in range(6):
            blocks += [
                nn.Conv2d(chs[i], chs[i + 1], 4, stride=2, padding=1),
                nn.BatchNorm2d(chs[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.conv = nn.Sequential(*blocks)
        flat_dim = chs[6] * 4 * 4  # 8192
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 128, base: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(latent_dim, base)
        self.decoder = Decoder(latent_dim, base)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        logvar = logvar.clamp(-10, 10)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decoder(z)


def vae_loss(recon, x, mu, logvar, beta):
    recon_term = F.l1_loss(recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    loss = recon_term + beta * kl
    l1_pixel = F.l1_loss(recon, x, reduction="mean")
    return loss, recon_term, kl, l1_pixel


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


def run_epoch(model, loader, device, use_amp, beta, opt=None, scaler=None, clip=None):
    train = opt is not None
    model.train(train)
    agg = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "l1": 0.0}
    n = 0
    max_gnorm = 0.0
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
        agg["kl"] += kl.item() * bs
        agg["l1"] += l1.item() * bs
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

    model = VAE(args.latent_dim, args.base).to(device)
    print(f"Параметров: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device, enabled=False)
    train_loader = make_loader("train", args, True)
    val_loader = make_loader("val", args, False)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    tag = "vae" if args.beta == 1.0 else f"bvae_b{args.beta:g}"

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, use_amp, args.beta, opt, scaler, clip=args.clip)
        va = run_epoch(model, val_loader, device, use_amp, args.beta)
        flag = ""
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                 "val_loss": va["loss"], "val_l1_pixel": va["l1"]},
                out_dir / f"{tag}_e1_best.pt",
            )
            flag = "  <- best"
        print(f"epoch {epoch:3d}/{args.epochs} | "
              f"train loss {tr['loss']:.1f} (recon {tr['recon']:.1f}, KL {tr['kl']:.2f}) | "
              f"val L1px {va['l1']:.4f} | val KL {va['kl']:.2f} | gnorm {tr['gnorm']:.1f}{flag}")
        
    print(f"Готово. Лучший val loss: {best_val:.1f} | чекпойнт: {out_dir / f'{tag}_e1_best.pt'}")


def smoke():
    m = VAE()
    x = torch.randn(2, 1, 256, 256)
    recon, mu, logvar = m(x)
    s = m.sample(2, "cpu")
    print(f"вход:   {tuple(x.shape)}")
    print(f"mu/logvar: {tuple(mu.shape)} / {tuple(logvar.shape)}")
    print(f"recon:  {tuple(recon.shape)} | sample из приора: {tuple(s.shape)}")
    print(f"параметров: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")
    assert recon.shape == x.shape and mu.shape == (2, 128) and s.shape == x.shape
    print("OK: формы сходятся, sample из N(0,I) проходит через декодер")


def parse_args():
    p = argparse.ArgumentParser(description="Vanilla VAE (E1)")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--metadata-csv", default="data/metadata.csv")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--beta", type=float, default=1.0, help="1.0 = Vanilla VAE; >1 = beta-VAE")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--clip", type=float, default=200000.0, help="")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        smoke()
    else:
        train(args)
