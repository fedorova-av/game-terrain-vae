"""Terrain-specific metrics for normalized DEM heightmaps"""

from __future__ import annotations

from collections import defaultdict

import torch


def gradient_components(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


def slope_magnitude(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    gx = x[..., :-1, 1:] - x[..., :-1, :-1]
    gy = x[..., 1:, :-1] - x[..., :-1, :-1]
    return torch.sqrt(gx.pow(2) + gy.pow(2) + eps)


def roughness_value(x: torch.Tensor) -> torch.Tensor:
    """Mean absolute Laplacian per sample"""

    if x.shape[-1] < 3 or x.shape[-2] < 3:
        return torch.zeros(x.size(0), device=x.device)
    lap = (
        -4.0 * x[..., 1:-1, 1:-1]
        + x[..., :-2, 1:-1]
        + x[..., 2:, 1:-1]
        + x[..., 1:-1, :-2]
        + x[..., 1:-1, 2:]
    )
    return lap.abs().mean(dim=(1, 2, 3))


def batch_metrics(
    x: torch.Tensor,
    recon: torch.Tensor,
    elevation_range: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    diff = recon - x
    mae = diff.abs().mean(dim=(1, 2, 3))
    rmse = torch.sqrt(diff.pow(2).mean(dim=(1, 2, 3)) + 1e-8)

    dx_x, dy_x = gradient_components(x)
    dx_r, dy_r = gradient_components(recon)
    gradient_mae = 0.5 * (
        (dx_r - dx_x).abs().mean(dim=(1, 2, 3))
        + (dy_r - dy_x).abs().mean(dim=(1, 2, 3))
    )

    slope_x = slope_magnitude(x)
    slope_r = slope_magnitude(recon)
    slope_diff = (slope_r - slope_x).abs().mean(dim=(1, 2, 3))

    rough_real = roughness_value(x)
    rough_recon = roughness_value(recon)
    roughness_diff = (rough_recon - rough_real).abs()

    out = {
        "mae": mae,
        "rmse": rmse,
        "gradient_mae": gradient_mae,
        "slope_diff": slope_diff,
        "roughness_real": rough_real,
        "roughness_recon": rough_recon,
        "roughness_diff": roughness_diff,
    }

    if elevation_range is not None:
        scale = elevation_range.to(x.device).float().view(-1) / 2.0
        valid = torch.isfinite(scale) & (scale > 0)
        if valid.any():
            nan_values = torch.full_like(mae, float("nan"))
            out["mae_m_approx"] = torch.where(valid, mae * scale, nan_values)
            out["rmse_m_approx"] = torch.where(valid, rmse * scale, nan_values)
    return out


def aggregate_sample_metrics(
    metric_batches: list[dict[str, torch.Tensor]],
    labels_batches: list[torch.Tensor] | None = None,
    idx_to_label: dict[int, str] | None = None,
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    strat_totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    strat_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    strat_n: dict[int, int] = defaultdict(int)

    for batch_i, metrics in enumerate(metric_batches):
        labels = None if labels_batches is None else labels_batches[batch_i].cpu()
        for key, values in metrics.items():
            values_cpu = values.detach().cpu()
            finite = torch.isfinite(values_cpu)
            if finite.any():
                totals[key] += float(values_cpu[finite].sum())
                counts[key] += int(finite.sum())
            if labels is not None:
                for sample_i, label in enumerate(labels.tolist()):
                    value = float(values_cpu[sample_i])
                    if torch.isfinite(torch.tensor(value)):
                        strat_totals[int(label)][key] += value
                        strat_counts[int(label)][key] += 1
                for label in labels.tolist():
                    strat_n[int(label)] += 1

    result = {key: value / max(1, counts[key]) for key, value in totals.items()}
    if idx_to_label:
        for label_id, n in strat_n.items():
            label = idx_to_label.get(label_id, f"class_{label_id}")
            result[f"{label}_n"] = n
            for key, value in strat_totals[label_id].items():
                result[f"{label}_{key}"] = value / max(1, strat_counts[label_id][key])
    return result

