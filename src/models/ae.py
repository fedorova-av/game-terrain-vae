"""
ae.py — детерминированный автоэнкодер (baseline E0).

Запуск:
    python -m src.models.ae --smoke                 # проверка форм без данных
    python -m src.models.ae --epochs 80             # полный прогон
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from src.data.dataset import HeightmapDataset
except Exception:
    HeightmapDataset = None

class Encoder(nn.Module):
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
        self.flat_dim = chs[6] * 4 * 4  # 512 * 4 * 4 = 8192
        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 128, base: int = 32):
        super().__init__()
        self.start_ch = base * 16  # 512
        self.fc = nn.Linear(latent_dim, self.start_ch * 4 * 4)
        chs = [base * 16, base * 8, base * 8, base * 4, base * 2, base, base]
        blocks = []
        for i in range(6):
            blocks += [
                nn.ConvTranspose2d(chs[i], chs[i + 1], 4, stride=2, padding=1),
                nn.BatchNorm2d(chs[i + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.deconv = nn.Sequential(*blocks)
        self.out = nn.Conv2d(base, 1, 3, padding=1)
        self.act = nn.Tanh()

    def forward(self, z):
        h = self.fc(z).view(-1, self.start_ch, 4, 4)
        h = self.deconv(h)
        return self.act(self.out(h))


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128, base: int = 32):
        super().__init__()
        self.encoder = Encoder(latent_dim, base)
        self.decoder = Decoder(latent_dim, base)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# Данные / обучение
def make_loader(split: str, args, shuffle: bool) -> DataLoader:
    ds = HeightmapDataset(
        processed_dir=args.processed_dir,
        metadata_csv=args.metadata_csv,
        split=split,
        return_label=False,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def run_epoch(model, loader, loss_fn, device, use_amp, opt=None, scaler=None, clip=None):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    torch.set_grad_enabled(train)
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            recon, _ = model(x)
            loss = loss_fn(recon, x)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
        total += loss.item() * x.size(0)
        n += x.size(0)
    torch.set_grad_enabled(True)
    return total / max(n, 1)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and args.amp
    torch.manual_seed(args.seed)
    print(f"Устройство: {device} | AMP: {use_amp} | latent_dim={args.latent_dim} | base={args.base}")

    model = Autoencoder(args.latent_dim, args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {n_params / 1e6:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss()
    scaler = torch.amp.GradScaler(device, enabled=False)

    train_loader = make_loader("train", args, shuffle=True)
    val_loader = make_loader("val", args, shuffle=False)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, loss_fn, device, use_amp, opt, scaler, clip=args.clip)
        va = run_epoch(model, val_loader, loss_fn, device, use_amp)
        flag = ""
        if va < best_val:
            best_val = va
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_l1": va},
                out_dir / "ae_e0_best.pt",
            )
            flag = "  <- best"
        print(f"epoch {epoch:3d}/{args.epochs} | train L1 {tr:.4f} | val L1 {va:.4f}{flag}")

    print(f"Готово. Лучший val L1: {best_val:.4f} | чекпойнт: {out_dir / 'ae_e0_best.pt'}")


def smoke():
    model = Autoencoder()
    x = torch.randn(2, 1, 256, 256)
    recon, z = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"вход:   {tuple(x.shape)}")
    print(f"латент: {tuple(z.shape)}")
    print(f"выход:  {tuple(recon.shape)} | диапазон [{recon.min():.2f}, {recon.max():.2f}]")
    print(f"параметров: {n_params / 1e6:.2f}M")
    assert recon.shape == x.shape, "форма реконструкции не совпала со входом"
    print("OK: формы сходятся, выход через tanh в [-1, 1]")


def parse_args():
    p = argparse.ArgumentParser(description="Autoencoder baseline (E0)")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--metadata-csv", default="data/metadata.csv")
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--clip", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        smoke()
    else:
        train(args)
