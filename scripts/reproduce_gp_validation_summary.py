#!/usr/bin/env python3
"""Reproduce the published GP validation summaries from point-level outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parents[1] / "derived_data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=Path, default=data_dir / "gp_production_cv_points.csv")
    parser.add_argument("--occupancy", type=Path, default=data_dir / "gp_occupancy.csv")
    parser.add_argument("--output-dir", type=Path, default=data_dir)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser.parse_args()


def summarize_points(points: pd.DataFrame, output_dir: Path, seed: int, n_boot: int) -> None:
    per_source = []
    for (source_id, variant), group in points.groupby(["source_id", "variant"]):
        row = {
            "source_id": source_id,
            "variant": variant,
            "median_abs_residual": float(group.residual.abs().median()),
        }
        if variant != "guide_only":
            row.update(
                mean_logp=float(group.logp.mean()),
                coverage1=float((group.z.abs() <= 1).mean()),
                coverage2=float((group.z.abs() <= 2).mean()),
                coverage3=float((group.z.abs() <= 3).mean()),
            )
        per_source.append(row)
    per_source = pd.DataFrame(per_source)
    per_source.to_csv(output_dir / "gp_metrics_by_source.csv", index=False)

    table_rows = []
    for variant, group in points.groupby("variant"):
        row = {
            "variant": variant,
            "n_sources": int(group.source_id.nunique()),
            "n_points": int(len(group)),
            "median_abs_residual": float(group.residual.abs().median()),
        }
        if variant != "guide_only":
            row.update(
                mean_logp=float(group.logp.mean()),
                coverage1=float((group.z.abs() <= 1).mean()),
                coverage2=float((group.z.abs() <= 2).mean()),
                coverage3=float((group.z.abs() <= 3).mean()),
            )
        table_rows.append(row)
    pd.DataFrame(table_rows).to_csv(output_dir / "gp_table_summary.csv", index=False)

    class_rows = []
    adopted = points[points.variant == "binned_complete"]
    for label, group in adopted.groupby("label"):
        class_rows.append(
            {
                "label": label,
                "n_sources": int(group.source_id.nunique()),
                "n_points": int(len(group)),
                "mean_logp": float(group.logp.mean()),
                "coverage1": float((group.z.abs() <= 1).mean()),
                "coverage2": float((group.z.abs() <= 2).mean()),
                "coverage3": float((group.z.abs() <= 3).mean()),
            }
        )
    pd.DataFrame(class_rows).to_csv(output_dir / "gp_binned_metrics_by_class.csv", index=False)

    rng = np.random.default_rng(seed)
    comparisons = [
        ("binned_complete", "raw"),
        ("binned_complete", "floor"),
        ("binned_complete", "inflation"),
    ]
    metrics = ["mean_logp", "median_abs_residual", "coverage3"]
    wide = {
        metric: per_source.pivot(index="source_id", columns="variant", values=metric)
        for metric in metrics
    }
    bootstrap_rows = []
    for variant_a, variant_b in comparisons:
        for metric, table in wide.items():
            paired = (table[variant_a] - table[variant_b]).dropna().to_numpy()
            samples = np.array(
                [np.mean(rng.choice(paired, len(paired), replace=True)) for _ in range(n_boot)]
            )
            bootstrap_rows.append(
                {
                    "variant_a": variant_a,
                    "variant_b": variant_b,
                    "metric": metric,
                    "mean_paired_difference": float(paired.mean()),
                    "ci_lo": float(np.quantile(samples, 0.025)),
                    "ci_hi": float(np.quantile(samples, 0.975)),
                    "n_sources": int(len(paired)),
                    "bootstrap_replicates": n_boot,
                    "seed": seed,
                }
            )
    pd.DataFrame(bootstrap_rows).to_csv(
        output_dir / "gp_source_bootstrap_comparisons.csv", index=False
    )

    source_scores = adopted.assign(abs_z=adopted.z.abs()).groupby(
        ["source_id", "label"], as_index=False
    ).abs_z.median()
    failure = source_scores.sort_values("abs_z", ascending=False).iloc[0]
    failure_points = adopted[adopted.source_id == failure.source_id].sort_values("phase")
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.scatter(failure_points.phase, failure_points.z, s=28, color="#b34a36", edgecolor="white", linewidth=0.4)
    ax.axhline(0, color="0.65", lw=0.8)
    for level in (1, 2, 3):
        ax.axhline(level, color="0.7", ls="--", lw=0.8)
        ax.axhline(-level, color="0.7", ls="--", lw=0.8)
    ax.set(
        xlabel="Held-out phase",
        ylabel="Standardized predictive residual",
        title=f"Source {failure.source_id}   {failure.label}",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "gp_heldout_failure_reproduced.png", dpi=180)
    plt.close(fig)
    (output_dir / "gp_failure_selection.json").write_text(
        json.dumps(
            {
                "selection_rule": "largest source-level median absolute standardized residual",
                "source_id": str(failure.source_id),
                "label": str(failure.label),
                "median_abs_z": float(failure.abs_z),
            },
            indent=2,
        )
        + "\n"
    )


def summarize_occupancy(occupancy: pd.DataFrame, output_dir: Path) -> None:
    occupied = occupancy.occupied_bins
    summary = {
        "n_sources": int(occupancy.source_id.nunique()),
        "n_phase_bins": 120,
        "median_n_obs": float(occupancy.n.median()),
        "median_occupied_bins": float(occupied.median()),
        "occupied_bins_iqr": [float(occupied.quantile(0.25)), float(occupied.quantile(0.75))],
        "occupied_bins_range": [int(occupied.min()), int(occupied.max())],
        "median_occupied_fraction": float((occupied / 120).median()),
    }
    (output_dir / "gp_occupancy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(args.points, dtype={"source_id": str})
    occupancy = pd.read_csv(args.occupancy, dtype={"source_id": str})
    summarize_points(points, args.output_dir, args.seed, args.bootstrap_replicates)
    summarize_occupancy(occupancy, args.output_dir)


if __name__ == "__main__":
    main()
