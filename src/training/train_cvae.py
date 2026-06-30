"""Conditional VAE training entrypoint.

The loop mirrors `src/training/train.py`: config parsing, self-contained dataset,
AdamW + AMP, gradient clipping, early stopping, and artifact writing all follow
the same format. Differences are limited to conditional generation:

  * model: `ConvCVAE`, `forward(x, label, scale) -> (recon, mu, logvar)`;
  * batch contains `label`, normalized `scale`, and raw `elevation_range`;
  * scale normalization statistics are saved in the checkpoint for reproducible
    generation.

Loss and metrics reuse shared project modules:
`src.training.losses.beta_vae_loss` and `src.evaluation.metrics`.

Example:
    python -m src.training.train_cvae --config configs/cvae.yaml
    python -m src.training.train_cvae --config configs/cvae_smoke.yaml   # быстрый CPU smoke
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from contextlib import nullcontext
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
from src.models import ConvCVAE
from src.training.losses import beta_vae_loss

TERRAIN_TO_ID = {"flat": 0, "hilly": 1, "mountain": 2}
ID_TO_TERRAIN = {v: k for k, v in TERRAIN_TO_ID.items()}


# ----------------------------- утилиты -------------------------------------
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
    """Грузим .pt максимально безопасно (как train.py): mmap + weights_only сперва."""
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
        except Exception as exc:  # noqa: BLE001
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


# ----------------------------- данные --------------------------------------
class TerrainCondDataset(Dataset):
    """Тот же контракт, что в train.py, плюс условие для CVAE.

    Каждый элемент:
        x                — [1, 256, 256], clamp([-1, 1])
        label            — long, индекс класса рельефа
        elevation_range  — float, СЫРОЙ перепад высот (метры) — для метрик в метрах
        scale            — float, НОРМИРОВАННЫЙ масштаб (вход модели)
        idx              — long, индекс строки

    Нормировка масштаба: log1p(elevation_range), затем стандартизация по
    статистике train (задаётся через set_scale_stats после построения сплитов).
    """

    def __init__(self, tensor: torch.Tensor, metadata: pd.DataFrame):
        self.tensor = tensor.float()
        self.metadata = metadata.reset_index(drop=True)
        labels = [
            TERRAIN_TO_ID.get(str(x).lower(), 0)
            for x in self.metadata.get("terrain_type", pd.Series(["flat"] * len(self.metadata)))
        ]
        self.labels = torch.tensor(labels, dtype=torch.long)
        if "elevation_range" in self.metadata.columns:
            er = self.metadata["elevation_range"].to_numpy(dtype=np.float32)
        else:
            er = np.zeros(len(self.metadata), dtype=np.float32)
        self.elevation_range = torch.tensor(er, dtype=torch.float32)
        # по умолчанию нормировка-заглушка; реальную статистику ставит set_scale_stats
        self.scale_mean = 0.0
        self.scale_std = 1.0

    def set_scale_stats(self, mean: float, std: float) -> None:
        self.scale_mean = float(mean)
        self.scale_std = float(std) + 1e-8

    def _norm_scale(self, er_raw: torch.Tensor) -> torch.Tensor:
        return (torch.log1p(er_raw.clamp(min=0.0)) - self.scale_mean) / self.scale_std

    def __len__(self) -> int:
        return len(self.tensor)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.tensor[idx]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        er = self.elevation_range[idx]
        return {
            "x": x.clamp(-1, 1),
            "label": self.labels[idx],
            "elevation_range": er,
            "scale": self._norm_scale(er),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def split_meta(metadata: pd.DataFrame, split: str) -> pd.DataFrame:
    out = metadata[metadata["split"].astype(str).str.lower().eq(split)].reset_index(drop=True)
    if len(out) == 0:
        raise ValueError(f"No metadata rows for split={split}")
    return out


def build_datasets(config: dict[str, Any]) -> tuple[Dataset, Dataset, Dataset, dict[str, float]]:
    data_cfg = config.get("data", {})
    processed_dir = Path(data_cfg.get("processed_dir", data_cfg.get("data_dir", "/kaggle/input")))
    metadata_path = Path(data_cfg.get("metadata_csv", processed_dir / "metadata.csv"))
    if not metadata_path.exists():
        matches = sorted(processed_dir.rglob("metadata.csv"))
        if not matches:
            raise FileNotFoundError("metadata.csv not found")
        metadata_path = matches[0]
    metadata = pd.read_csv(metadata_path)

    datasets: dict[str, TerrainCondDataset] = {}
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
        datasets[split] = TerrainCondDataset(tensor, meta)

    # Статистику нормировки масштаба считаем ТОЛЬКО на train и переносим на val/test
    # (никакой утечки из val/test в условие).
    train_log = torch.log1p(datasets["train"].elevation_range.clamp(min=0.0))
    scale_mean = float(train_log.mean())
    scale_std = float(train_log.std())
    for ds in datasets.values():
        ds.set_scale_stats(scale_mean, scale_std)

    scale_stats = {"log1p_mean": scale_mean, "log1p_std": scale_std + 1e-8}
    return datasets["train"], datasets["val"], datasets["test"], scale_stats


def maybe_subset(ds: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(np.arange(len(ds)), size=max_samples, replace=False))
    return Subset(ds, indices.tolist())


def make_loader(ds: Dataset, batch_size: int, num_workers: int, device: torch.device, shuffle: bool) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs)


# ----------------------------- модель --------------------------------------
def build_model(config: dict[str, Any]) -> ConvCVAE:
    model_cfg = config.get("model", {})
    return ConvCVAE(
        in_channels=int(model_cfg.get("in_channels", 1)),
        image_size=int(model_cfg.get("image_size", 256)),
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        channel_multipliers=tuple(model_cfg.get("channel_multipliers", [1, 2, 4, 8, 16])),
        num_classes=int(model_cfg.get("num_classes", 3)),
        class_embed_dim=int(model_cfg.get("class_embed_dim", 16)),
        scale_embed_dim=int(model_cfg.get("scale_embed_dim", 8)),
    )


def aggregate_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for row in rows:
        for key, value in row.items():
            out[key] += float(value)
    return {key: value / max(1, len(rows)) for key, value in out.items()}


def resolve_device(name: str | None) -> torch.device:
    """cuda -> mps (Apple GPU) -> cpu. 'cuda' в конфиге локально авто-падает на mps."""
    name = (name or "auto").lower()
    if name in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "mps":
        ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        return torch.device("mps" if ok else "cpu")
    return torch.device(name)


def autocast_context(device: torch.device, use_amp: bool):
    # AMP только на CUDA; на MPS/CPU считаем в fp32 (стабильнее, без GradScaler-возни).
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def run_loss_epoch(model, loader, config, device, use_amp, optimizer=None, scaler=None, epoch=1, grad_clip=1.0):
    is_train = optimizer is not None
    model.train(is_train)
    rows = []
    loss_cfg = config.get("loss", {})
    for batch in tqdm(loader, leave=False, desc="train" if is_train else "val"):
        x = batch["x"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        scale = batch["scale"].to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train), autocast_context(device, use_amp):
            recon, mu, logvar = model(x, label, scale)
            # Условие влияет только через сеть — лосс тот же, что у β-VAE.
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
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        rows.append(parts)
    return aggregate_rows(rows)


@torch.no_grad()
def evaluate_model(model, loader, device, use_amp):
    """Compute stratified reconstruction metrics with the shared metrics module."""
    model.eval()
    metric_batches, label_batches = [], []
    for batch in tqdm(loader, leave=False, desc="eval"):
        x = batch["x"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        scale = batch["scale"].to(device, non_blocking=True)
        er = batch["elevation_range"].to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            recon, _, _ = model(x, label, scale)
        metric_batches.append(batch_metrics(x, recon, er))
        label_batches.append(batch["label"])
    return aggregate_sample_metrics(metric_batches, label_batches, ID_TO_TERRAIN)


# ----------------------------- артефакты эксперимента ----------------------
def class_median_scale(train_ds_meta: pd.DataFrame) -> dict[int, float]:
    """Медианный elevation_range (метры) по каждому классу — типичное условие генерации."""
    out: dict[int, float] = {}
    for name, cid in TERRAIN_TO_ID.items():
        sel = train_ds_meta[train_ds_meta["terrain_type"].astype(str).str.lower().eq(name)]
        out[cid] = float(sel["elevation_range"].median()) if len(sel) else 0.0
    return out


@torch.no_grad()
def save_condition_grid(model, scale_stats, class_med_er, out_path: Path, device, n: int = 6) -> None:
    """Сетка сгенерированных плиток: строка на класс. Удовлетворяет контракту 'минимум 1 PNG'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    mean, std = scale_stats["log1p_mean"], scale_stats["log1p_std"]
    fig, axes = plt.subplots(len(TERRAIN_TO_ID), n, figsize=(2.0 * n, 2.0 * len(TERRAIN_TO_ID)))
    if len(TERRAIN_TO_ID) == 1:
        axes = axes[None, :]
    for cid, name in ID_TO_TERRAIN.items():
        er = max(class_med_er.get(cid, 0.0), 0.0)
        scale_norm = (float(np.log1p(er)) - mean) / std
        samples = model.sample(n, int(cid), scale_norm, device=device).cpu().squeeze(1).numpy()
        for j in range(n):
            ax = axes[cid, j]
            ax.imshow(samples[j], cmap="terrain", vmin=-1, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, fontsize=12)
    fig.suptitle("CVAE: сгенерированные heightmap по условию (строка = класс)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_model_card(path: Path, config: dict, metrics: dict, n_params: float, scale_stats: dict, epochs_run: int) -> None:
    exp = config.get("experiment_name", "cvae")
    loss_cfg = config.get("loss", {})
    model_cfg = config.get("model", {})

    def g(key: str) -> str:
        v = metrics.get(key)
        return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"

    card = f"""# Model card: {exp} (Conditional VAE)

Conditional beta-VAE for controllable heightmap generation by terrain type
(`flat`, `hilly`, `mountain`) and continuous elevation range.

## Architecture
- `ConvCVAE` extends `ConvBetaVAE` with class and scale conditioning.
- Backbone: Conv + GroupNorm + SiLU, multipliers {model_cfg.get('channel_multipliers', [1,2,4,8,16])}.
- Latent dimension: `latent_dim={model_cfg.get('latent_dim', 128)}`.
- Condition: `[class_embed(terrain_type) || scale_proj(elevation_range_norm)]`.
- Parameters: {n_params:.2f}M.

## Objective
- `src.training.losses.beta_vae_loss`: recon ({loss_cfg.get('recon_type', 'mse')}) +
  beta * (KL / num_pixels) + grad_loss.
- beta={loss_cfg.get('beta')}, kl_anneal_epochs={loss_cfg.get('kl_anneal_epochs')},
  free_bits={loss_cfg.get('free_bits')}, grad_loss_weight={loss_cfg.get('grad_loss_weight')}.

## Data and scale normalization
- Tensor format: `[N,1,256,256]`, per-patch min-max to [-1, 1].
- `elevation_range` uses log1p + train-set standardization:
  mean={scale_stats['log1p_mean']:.4f}, std={scale_stats['log1p_std']:.4f}
  and is saved in `best.pt` for reproducible generation.

## Test metrics
- RMSE={g('rmse')}, MAE={g('mae')}, gradient_MAE={g('gradient_mae')},
  slope_diff={g('slope_diff')}, roughness_diff={g('roughness_diff')}.
- Trained epochs: {epochs_run}. Full curves are in `history.csv`; class-stratified
  metrics are in `metrics.json`.

## Controllability
- The key validation is condition consistency: generated elevation range and
  slope increase from flat to hilly to mountain.

## Limitations
- Concatenation-based conditioning could be improved with FiLM.
- VAE models smooth microrelief; gradient loss partially compensates for it.
- Meter-scale metrics are approximate because they rely on per-patch range.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(card, encoding="utf-8")


# ----------------------------- главный цикл --------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = read_yaml(args.config)
    train_ds, val_ds, test_ds, scale_stats = build_datasets(config)

    # медианный масштаб класса считаем по полному train ДО subset (для grid-генерации)
    train_meta = train_ds.metadata.copy()
    class_med_er = class_median_scale(train_meta)

    data_cfg = config.get("data", {})
    train_ds = maybe_subset(train_ds, data_cfg.get("max_train_samples"), 42)
    val_ds = maybe_subset(val_ds, data_cfg.get("max_val_samples"), 43)
    test_ds = maybe_subset(test_ds, data_cfg.get("max_test_samples"), 44)

    training = config.get("training", {})
    seed = int(training.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(args.device or training.get("device", "cuda"))
    use_amp = bool(training.get("use_amp", True))
    batch_size = int(training.get("batch_size", 32))
    num_workers = int(training.get("num_workers", 2))

    train_loader = make_loader(train_ds, batch_size, num_workers, device, True)
    val_loader = make_loader(val_ds, batch_size, num_workers, device, False)
    test_loader = make_loader(test_ds, batch_size, num_workers, device, False)

    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: ConvCVAE, params={n_params:.2f}M, device={device}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    out_dir = Path(config.get("output", {}).get("out_dir", f"outputs/{config['experiment_name']}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(args.epochs or training.get("epochs", 20))
    patience = int(training.get("early_stopping_patience", 5))
    grad_clip = float(training.get("grad_clip", 1.0))
    best_val = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, epochs + 1):
        train_stats = run_loss_epoch(model, train_loader, config, device, use_amp, optimizer, scaler, epoch, grad_clip)
        val_stats = run_loss_epoch(model, val_loader, config, device, use_amp, None, None, epoch)
        val_metrics = evaluate_model(model, val_loader, device, use_amp)
        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            **{f"val_metric_{k}": v for k, v in val_metrics.items()},
        })
        print(f"epoch {epoch:03d}: val_loss={val_stats['loss']:.4f}, "
              f"val_rmse={val_metrics['rmse']:.4f}, train_kl_raw={train_stats['kl_raw']:.3f}")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            bad_epochs = 0
            torch.save(
                {"model_state": model.state_dict(), "config": config, "epoch": epoch,
                 "scale_stats": scale_stats, "class_median_elevation_range": class_med_er},
                out_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    epochs_run = len(history)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # финальная оценка на TEST лучшей моделью (как в train.py)
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    metrics = evaluate_model(model, test_loader, device, use_amp)

    # metrics.json — ПЛОСКИЙ формат как у β-VAE -> попадает в compare_experiments
    save_json({"name": config["experiment_name"], "best_val_loss": best_val, **metrics},
              out_dir / "metrics.json")

    # артефакты эксперимента по контракту README: samples/*.png + model_card.md
    try:
        save_condition_grid(model, ckpt["scale_stats"], ckpt["class_median_elevation_range"],
                            out_dir / "samples" / "conditional_grid.png", device,
                            n=int(config.get("output", {}).get("save_n_samples", 6)) // 2 or 6)
    except Exception as exc:  # noqa: BLE001
        print(f"warn: не удалось сохранить sample grid: {exc}")
    write_model_card(out_dir / "model_card.md", config, metrics, n_params, scale_stats, epochs_run)

    print(f"done. test_rmse={metrics.get('rmse'):.4f}  ->  {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
