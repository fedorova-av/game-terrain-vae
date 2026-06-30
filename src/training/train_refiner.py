"""DetailRefiner training entrypoint on top of CVAE.

The loop follows `src/training/train_cvae.py`: config parsing, self-contained
dataset, AMP, gradient clipping, early stopping, and artifact writing. The
training task is different:

  1. take a real tile x;
  2. x_coarse = coarsen(x, kernel, sigma), approximating VAE output;
  3. x_refined = refiner(x_coarse, label);
  4. loss = refiner_loss(x_refined, x), emphasizing gradients and roughness.

The training pair `coarsen(real) -> real` is independent of CVAE quality. At
inference time, the refiner is applied to smooth CVAE output. The coarsening
sigma is calibrated in configs/detail_refiner.yaml.

Metrics use the shared project evaluation module, so `metrics.json` keeps the
same flat format as the other experiments.

Example:
    python -m src.training.train_refiner --config configs/detail_refiner.yaml
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
from src.models.detail_refiner import DetailRefiner, refiner_loss, coarsen

TERRAIN_TO_ID = {"flat": 0, "hilly": 1, "mountain": 2}
ID_TO_TERRAIN = {v: k for k, v in TERRAIN_TO_ID.items()}


def load_cvae(checkpoint: str, device: torch.device):
    """Загрузить обученную CVAE (frozen) + статистику нормировки масштаба."""
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    mc = ck["config"]["model"]
    cvae = ConvCVAE(
        in_channels=1, image_size=int(mc.get("image_size", 256)), latent_dim=mc["latent_dim"],
        base_channels=mc["base_channels"], channel_multipliers=tuple(mc["channel_multipliers"]),
        num_classes=mc["num_classes"], class_embed_dim=mc["class_embed_dim"],
        scale_embed_dim=mc["scale_embed_dim"],
    ).to(device)
    cvae.load_state_dict(ck["model_state"])
    cvae.eval()
    for p in cvae.parameters():
        p.requires_grad_(False)
    return cvae, ck["scale_stats"]


def build_make_input(config, device):
    """Функция, превращающая реальную плитку x во вход refiner.

    mode='cvae_recon' (правильный путь): x_input = детерминированная реконструкция
        CVAE (encode->mu->decode) — refiner видит НАСТОЯЩИЕ артефакты CVAE и учится
        дорисовывать рельеф под них (а не усиливать шум, как при coarsen).
    mode='coarsen': гауссово размытие (домен не совпадает с выходом CVAE — хуже).
    """
    mode = config.get("refine_input", {}).get("mode", "coarsen")
    if mode == "cvae_recon":
        ckpt = config.get("stage1_cvae", {}).get("checkpoint", "outputs/cvae/best.pt")
        cvae, stats = load_cvae(ckpt, device)
        mean, std = float(stats["log1p_mean"]), float(stats["log1p_std"])

        @torch.no_grad()
        def make_input(x, label, er):
            scale = (torch.log1p(er.clamp(min=0).to(device)) - mean) / std
            cond = cvae.cond_vector(label.to(device), scale)
            mu, _ = cvae.encode(x, cond)
            return cvae.decode(mu, cond).clamp(-1, 1)

        return make_input, mode

    co = config.get("coarsen", {})
    kernel, sigma = int(co.get("kernel", 9)), float(co.get("sigma", 1.0))

    def make_input(x, label, er):
        return coarsen(x, kernel=kernel, sigma=sigma)

    return make_input, mode


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
    for kwargs in (
        {"map_location": "cpu", "mmap": True, "weights_only": True},
        {"map_location": "cpu", "mmap": True},
        {"map_location": "cpu", "weights_only": True},
        {"map_location": "cpu"},
    ):
        try:
            return torch.load(path, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise last


def extract_tensor(obj) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj.float()
    if isinstance(obj, dict):
        for key in ("data", "x", "heightmaps", "tensor", "tensors"):
            if key in obj and torch.is_tensor(obj[key]):
                return obj[key].float()
        for v in obj.values():
            if torch.is_tensor(v):
                return v.float()
    raise TypeError(f"Unsupported .pt object: {type(obj)}")


def resolve_device(name: str | None) -> torch.device:
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
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=True)
    return nullcontext()


# ----------------------------- данные --------------------------------------
class TerrainTileDataset(Dataset):
    """Реальные плитки + класс рельефа (для условного refiner) + elevation_range (метрики)."""

    def __init__(self, tensor: torch.Tensor, metadata: pd.DataFrame):
        self.tensor = tensor.float()
        self.metadata = metadata.reset_index(drop=True)
        labels = [TERRAIN_TO_ID.get(str(x).lower(), 0)
                  for x in self.metadata.get("terrain_type", pd.Series(["flat"] * len(self.metadata)))]
        self.labels = torch.tensor(labels, dtype=torch.long)
        er = (self.metadata["elevation_range"].to_numpy(dtype=np.float32)
              if "elevation_range" in self.metadata.columns
              else np.zeros(len(self.metadata), dtype=np.float32))
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
    processed_dir = Path(data_cfg.get("processed_dir", "/kaggle/input"))
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
        datasets[split] = TerrainTileDataset(tensor, meta)
    return datasets["train"], datasets["val"], datasets["test"]


def maybe_subset(ds: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(ds):
        return ds
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(len(ds)), size=max_samples, replace=False))
    return Subset(ds, idx.tolist())


def make_loader(ds: Dataset, batch_size: int, num_workers: int, device: torch.device, shuffle: bool) -> DataLoader:
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, drop_last=shuffle,
                  num_workers=num_workers, pin_memory=device.type == "cuda")
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs)


# ----------------------------- модель --------------------------------------
def build_model(config: dict[str, Any]) -> DetailRefiner:
    m = config.get("model", {})
    return DetailRefiner(
        in_channels=int(m.get("in_channels", 1)),
        base_channels=int(m.get("base_channels", 48)),
        num_blocks=int(m.get("num_blocks", 6)),
        num_classes=int(m.get("num_classes", 3)),
        use_class_cond=bool(m.get("use_class_cond", True)),
        residual_scale=float(m.get("residual_scale", 1.0)),
    )


def aggregate_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for row in rows:
        for k, v in row.items():
            out[k] += float(v)
    return {k: v / max(1, len(rows)) for k, v in out.items()}


def run_epoch(model, loader, make_input, config, device, use_amp, optimizer=None, scaler=None, grad_clip=1.0):
    is_train = optimizer is not None
    model.train(is_train)
    rows = []
    loss_cfg = config.get("loss", {})
    for batch in tqdm(loader, leave=False, desc="train" if is_train else "val"):
        x = batch["x"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        er = batch["elevation_range"].to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train), autocast_context(device, use_amp):
            x_input = make_input(x, label, er)
            refined = model(x_input, label)
            loss, parts = refiner_loss(
                refined, x,
                w_pixel=float(loss_cfg.get("w_pixel", 1.0)),
                w_grad=float(loss_cfg.get("w_grad", 1.0)),
                w_rough=float(loss_cfg.get("w_rough", 0.5)),
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
        rows.append({"loss": parts["total"], "pixel": parts["pixel"],
                     "grad": parts["grad"], "rough": parts["rough"]})
    return aggregate_rows(rows)


@torch.no_grad()
def evaluate_model(model, loader, make_input, device, use_amp):
    """Метрики восстановления: refined(вход) vs real, через общий metrics.py."""
    model.eval()
    metric_batches, label_batches = [], []
    for batch in tqdm(loader, leave=False, desc="eval"):
        x = batch["x"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        er = batch["elevation_range"].to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            refined = model(make_input(x, label, er), label)
        # x = real (target), recon = refined -> roughness_diff = |rough(refined)-rough(real)|
        metric_batches.append(batch_metrics(x, refined, er))
        label_batches.append(batch["label"])
    return aggregate_sample_metrics(metric_batches, label_batches, ID_TO_TERRAIN)


@torch.no_grad()
def save_before_after(model, loader, make_input, device, out_path: Path, n: int = 6):
    """Грид: real | вход (CVAE-recon/coarse) | refined — по строке на пример."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    batch = next(iter(loader))
    x = batch["x"][:n].to(device); label = batch["label"][:n].to(device)
    er = batch["elevation_range"][:n].to(device)
    xc = make_input(x, label, er)
    xr = model(xc, label)
    x, xc, xr = x.cpu().squeeze(1).numpy(), xc.cpu().squeeze(1).numpy(), xr.cpu().squeeze(1).numpy()
    rows = min(n, len(x))
    fig, axes = plt.subplots(rows, 3, figsize=(6, 2 * rows))
    if rows == 1:
        axes = axes[None, :]
    for i in range(rows):
        for j, (img, t) in enumerate([(x[i], "real"), (xc[i], "coarse"), (xr[i], "refined")]):
            axes[i, j].imshow(img, cmap="terrain", vmin=-1, vmax=1)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(t)
    fig.suptitle("DetailRefiner: real | coarse | refined")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------- главный цикл --------------------------------
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
    seed_everything(int(training.get("seed", 42)))
    device = resolve_device(args.device or training.get("device", "cuda"))
    use_amp = bool(training.get("use_amp", True))
    batch_size = int(training.get("batch_size", 16))
    num_workers = int(training.get("num_workers", 2))

    train_loader = make_loader(train_ds, batch_size, num_workers, device, True)
    val_loader = make_loader(val_ds, batch_size, num_workers, device, False)
    test_loader = make_loader(test_ds, batch_size, num_workers, device, False)

    model = build_model(config).to(device)
    make_input, mode = build_make_input(config, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: DetailRefiner, params={n_params:.3f}M, device={device}, refine_input_mode={mode}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("lr", 2e-4)),
                                  weight_decay=float(training.get("weight_decay", 1e-4)))
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    out_dir = Path(config.get("output", {}).get("out_dir", f"outputs/{config['experiment_name']}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(args.epochs or training.get("epochs", 15))
    patience = int(training.get("early_stopping_patience", 5))
    grad_clip = float(training.get("grad_clip", 1.0))
    best_val = float("inf")
    bad = 0
    history = []
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, make_input, config, device, use_amp, optimizer, scaler, grad_clip)
        va = run_epoch(model, val_loader, make_input, config, device, use_amp, None, None)
        vm = evaluate_model(model, val_loader, make_input, device, use_amp)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()},
                        **{f"val_metric_{k}": v for k, v in vm.items()}})
        print(f"epoch {epoch:03d}: val_loss={va['loss']:.4f} "
              f"val_rough_diff={vm['roughness_diff']:.4f} val_grad_mae={vm['gradient_mae']:.4f}")
        if va["loss"] < best_val:
            best_val = va["loss"]; bad = 0
            torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch},
                       out_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    metrics = evaluate_model(model, test_loader, make_input, device, use_amp)
    save_json({"name": config["experiment_name"], "best_val_loss": best_val, **metrics},
              out_dir / "metrics.json")
    try:
        save_before_after(model, test_loader, make_input, device, out_dir / "samples" / "real_input_refined.png")
    except Exception as exc:  # noqa: BLE001
        print(f"warn: sample grid не сохранён: {exc}")
    print(f"done. test roughness_diff={metrics.get('roughness_diff'):.4f}, "
          f"gradient_mae={metrics.get('gradient_mae'):.4f} -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
