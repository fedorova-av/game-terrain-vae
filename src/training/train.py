"""Standalone beta-VAE training entrypoint.

This script expects prepared `train.pt`, `val.pt`, `test.pt` and `metadata.csv` files

Example:
    python -m src.training.train --config configs/beta_vae_beta_0_5.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import yaml

from src.evaluation.metrics import aggregate_sample_metrics, batch_metrics
from src.models import ConvBetaVAE
from src.training.losses import beta_vae_loss

TERRAIN_TO_ID = {"flat": 0, "hilly": 1, "mountain": 2}
ID_TO_TERRAIN = {v: k for k, v in TERRAIN_TO_ID.items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def torch_load_safely(path: Path):
    attempts = [
        {"map_location": "cpu", "mmap": True, "weights_only": True},
        {"map_location": "cpu", "mmap": True},
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu"},
    ]
    last_exc = None
    for kwargs in attempts:
        try:
            return torch.load(path, **kwargs)
        except Exception as exc:
            last_exc = exc
    raise last_exc


def extract_tensor(obj) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj.float()
    if isinstance(obj, dict):
        for key in ["data", "x", "heightmaps", "tensor", "tensors"]:
            if key in obj and torch.is_tensor(obj[key]):
                return obj[key].float()
        for value in obj.values():
            if torch.is_tensor(value):
                return value.float()
    raise TypeError(f"Unsupported .pt object type: {type(obj)}")


class TerrainTensorDataset(Dataset):
    def __init__(self, tensor: torch.Tensor, metadata: pd.DataFrame):
        self.tensor = tensor.float()
        self.metadata = metadata.reset_index(drop=True)
        labels = [TERRAIN_TO_ID.get(str(x).lower(), 0) for x in self.metadata.get("terrain_type", pd.Series(["flat"] * len(self.metadata)))]
        self.labels = torch.tensor(labels, dtype=torch.long)
        if "elevation_range" in self.metadata.columns:
            er = self.metadata["elevation_range"].values
        else:
            er = np.full(len(self.metadata), np.nan, dtype=np.float32)
        self.elevation_range = torch.tensor(er, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.tensor)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.tensor[idx]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return {
            "x": x.clamp(-1, 1),
            "label": self.labels[idx],
            "elevation_range": self.elevation_range[idx],
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def split_meta(metadata: pd.DataFrame, split: str) -> pd.DataFrame:
    out = metadata[metadata["split"].astype(str).str.lower().eq(split)].reset_index(drop=True)
    if len(out) == 0:
        raise ValueError(f"No metadata rows for split={split}")
    return out


def build_datasets(config: dict[str, Any]):
    data_cfg = config.get("data", {})
    processed_dir = Path(data_cfg.get("processed_dir", data_cfg.get("data_dir", "/kaggle/input")))
    metadata_path = Path(data_cfg.get("metadata_csv", processed_dir / "metadata.csv"))
    if not metadata_path.exists():
        matches = sorted(processed_dir.rglob("metadata.csv"))
        if not matches:
            raise FileNotFoundError("metadata.csv not found")
        metadata_path = matches[0]
    metadata = pd.read_csv(metadata_path)

    datasets = {}
    for split in ["train", "val", "test"]:
        path = Path(data_cfg.get(f"{split}_pt", processed_dir / f"{split}.pt"))
        if not path.exists():
            matches = sorted(processed_dir.rglob(f"{split}.pt"))
            if not matches:
                raise FileNotFoundError(f"{split}.pt not found")
            path = matches[0]
        tensor = extract_tensor(torch_load_safely(path))
        meta = split_meta(metadata, split)
        if len(tensor) != len(meta):
            raise ValueError(f"Tensor/meta mismatch for {split}: {len(tensor)} vs {len(meta)}")
        datasets[split] = TerrainTensorDataset(tensor, meta)
    return datasets["train"], datasets["val"], datasets["test"]


def maybe_subset(ds: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(np.arange(len(ds)), size=max_samples, replace=False))
    return Subset(ds, indices.tolist())


def make_loader(ds: Dataset, batch_size: int, num_workers: int, device: torch.device, shuffle: bool) -> DataLoader:
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, num_workers=num_workers, pin_memory=device.type == "cuda")
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs)


def build_model(config: dict[str, Any]) -> ConvBetaVAE:
    model_cfg = config.get("model", {})
    return ConvBetaVAE(
        in_channels=int(model_cfg.get("in_channels", 1)),
        image_size=int(model_cfg.get("image_size", 256)),
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        channel_multipliers=tuple(model_cfg.get("channel_multipliers", [1, 2, 4, 8, 16])),
    )


def aggregate_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    out = defaultdict(float)
    for row in rows:
        for key, value in row.items():
            out[key] += float(value)
    return {key: value / max(1, len(rows)) for key, value in out.items()}


def autocast_context(device: torch.device, use_amp: bool):
    return torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda"))


def run_loss_epoch(model, loader, config, device, use_amp, optimizer=None, scaler=None, epoch=1, grad_clip=1.0):
    is_train = optimizer is not None
    model.train(is_train)
    rows = []
    loss_cfg = config.get("loss", {})
    for batch in tqdm(loader, leave=False, desc="train" if is_train else "val"):
        x = batch["x"].to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            recon, mu, logvar = model(x)
            loss, parts = beta_vae_loss(
                x, recon, mu, logvar,
                beta=float(loss_cfg.get("beta", 1.0)),
                epoch=epoch,
                kl_anneal_epochs=int(loss_cfg.get("kl_anneal_epochs", 0)),
                free_bits=float(loss_cfg.get("free_bits", 0.0)),
                grad_loss_weight=float(loss_cfg.get("grad_loss_weight", 0.0)),
                recon_type=str(loss_cfg.get("recon_type", "mse")),
            )
        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        rows.append(parts)
    return aggregate_rows(rows)


@torch.no_grad()
def evaluate_model(model, loader, device, use_amp):
    model.eval()
    metric_batches, label_batches = [], []
    for batch in tqdm(loader, leave=False, desc="eval"):
        x = batch["x"].to(device, non_blocking=True)
        er = batch["elevation_range"].to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            recon, _, _ = model(x)
        metric_batches.append(batch_metrics(x, recon, er))
        label_batches.append(batch["label"])
    return aggregate_sample_metrics(metric_batches, label_batches, ID_TO_TERRAIN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = read_yaml(args.config)
    train_ds, val_ds, test_ds = build_datasets(config)
    data_cfg = config.get("data", {})
    train_ds = maybe_subset(train_ds, data_cfg.get("max_train_samples"), 42)
    val_ds = maybe_subset(val_ds, data_cfg.get("max_val_samples"), 43)
    test_ds = maybe_subset(test_ds, data_cfg.get("max_test_samples"), 44)

    training = config.get("training", {})
    seed = int(training.get("seed", 42))
    seed_everything(seed)
    device_name = args.device or training.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    use_amp = bool(training.get("use_amp", True))
    batch_size = int(training.get("batch_size", 32))
    num_workers = int(training.get("num_workers", 2))

    train_loader = make_loader(train_ds, batch_size, num_workers, device, True)
    val_loader = make_loader(val_ds, batch_size, num_workers, device, False)
    test_loader = make_loader(test_ds, batch_size, num_workers, device, False)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("lr", 2e-4)), weight_decay=float(training.get("weight_decay", 1e-4)))
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    out_dir = Path(config.get("output", {}).get("out_dir", f"outputs/{config['experiment_name']}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(args.epochs or training.get("epochs", 20))
    patience = int(training.get("early_stopping_patience", 5))
    best_val = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, epochs + 1):
        train_stats = run_loss_epoch(model, train_loader, config, device, use_amp, optimizer, scaler, epoch, float(training.get("grad_clip", 1.0)))
        val_stats = run_loss_epoch(model, val_loader, config, device, use_amp, None, None, epoch)
        val_metrics = evaluate_model(model, val_loader, device, use_amp)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}, **{f"val_metric_{k}": v for k, v in val_metrics.items()}})
        print(f"epoch {epoch:03d}: val_loss={val_stats['loss']:.4f}, val_rmse={val_metrics['rmse']:.4f}")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch}, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    metrics = evaluate_model(model, test_loader, device, use_amp)
    save_json({"name": config["experiment_name"], "best_val_loss": best_val, **metrics}, out_dir / "metrics.json")


if __name__ == "__main__":
    main()
