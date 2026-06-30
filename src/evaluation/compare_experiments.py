"""Collect metrics.json files into CSV/Markdown comparison tables"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten_metrics(metrics: dict) -> dict:
    return dict(metrics)


def collect_metrics(outputs_dir: str | Path) -> pd.DataFrame:
    outputs_dir = Path(outputs_dir)
    rows = []
    for metrics_path in sorted(outputs_dir.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            row = flatten_metrics(json.load(f))
        row.setdefault("name", metrics_path.parent.name)
        rows.append(row)
    return pd.DataFrame(rows)


def add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    weights = {
        "rmse": 0.35,
        "gradient_mae": 0.25,
        "roughness_diff": 0.25,
        "slope_diff": 0.15,
    }
    scored = df.copy()
    score = pd.Series(0.0, index=scored.index)
    used = []
    for metric, weight in weights.items():
        if metric not in scored.columns:
            continue
        values = pd.to_numeric(scored[metric], errors="coerce")
        if values.notna().sum() < 2:
            continue
        normalized = (values - values.min()) / (values.max() - values.min() + 1e-12)
        score = score + weight * normalized.fillna(1.0)
        used.append(metric)
    scored["composite_score"] = score
    scored["composite_metrics"] = ", ".join(used)
    return scored.sort_values("composite_score").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--out-csv", default="results/experiment_summary.csv")
    parser.add_argument("--out-md", default="results/experiment_summary.md")
    args = parser.parse_args()

    df = collect_metrics(args.outputs)
    if len(df) == 0:
        raise SystemExit(f"No metrics.json files found under {args.outputs}")
    df = add_composite_score(df)

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    try:
        out_md.write_text(df.to_markdown(index=False), encoding="utf-8")
    except Exception:
        out_md.write_text(df.to_csv(index=False), encoding="utf-8")
    print(df.head(20))
    print(f"saved {out_csv}")
    print(f"saved {out_md}")


if __name__ == "__main__":
    main()

