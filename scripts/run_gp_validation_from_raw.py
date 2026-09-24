#!/usr/bin/env python3
"""Re-run the five-fold GP validation when restricted raw inputs are available."""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from gp_model_impl import build_template_weighted, fit_george_template, sigma_clipping


VARIANTS = [
    "raw",
    "floor",
    "clipping",
    "inflation",
    "raw_plus_guide",
    "guide_only",
    "binned_complete",
]


def parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).resolve().parents[1] / "derived_data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--light-curves", type=Path, required=True)
    parser.add_argument("--match-table", type=Path, required=True)
    parser.add_argument("--selected-sources", type=Path, default=data_dir / "gp_selected_sources.csv")
    parser.add_argument("--output", type=Path, default=data_dir / "gp_production_cv_points_rerun.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = pd.read_csv(args.selected_sources, dtype={"source_id": str})
    match = pd.read_csv(args.match_table, dtype={"FileNumber": str}).rename(
        columns={"FileNumber": "source_id"}
    )
    selected = selected.merge(match[["source_id", "T0", "Per"]], on="source_id", validate="one_to_one")
    rows = []
    occupancy = []

    for source in selected.itertuples(index=False):
        data = pd.read_csv(args.light_curves / f"{source.source_id}.csv")
        phase = ((data.time.to_numpy() - float(source.T0)) / float(source.Per)) % 1
        mag = data.mag.to_numpy()
        err = data.mag_err.to_numpy()
        valid = (
            np.isfinite(phase)
            & np.isfinite(mag)
            & np.isfinite(err)
            & (err > 0)
            & (mag != 99)
            & (err != 99)
        )
        phase, mag, err = phase[valid], mag[valid], err[valid]
        occupied = len(np.unique(np.minimum((phase * 120).astype(int), 119)))
        occupancy.append(
            {"source_id": source.source_id, "label": source.label, "n": len(phase), "occupied_bins": occupied}
        )

        for fold in range(5):
            test = (phase >= fold / 5) & (phase < (fold + 1) / 5)
            train = ~test
            if not test.any() or train.sum() < 8:
                continue
            clipped_phase, clipped_mag, clipped_err = sigma_clipping(
                phase[train], mag[train], err[train], threshold=3, iteration=2
            )
            if len(clipped_phase) < 5:
                continue
            centers, guide, guide_err = build_template_weighted(
                clipped_phase, clipped_mag, None, nbins=120, smooth_window=6
            )

            for variant in VARIANTS:
                if variant == "guide_only":
                    mean = np.interp(phase[test], centers, guide)
                    variance = np.full(test.sum(), np.nan)
                else:
                    x, y, yerr = phase[train], mag[train], err[train]
                    if variant == "clipping":
                        x, y, yerr = clipped_phase, clipped_mag, clipped_err
                    elif variant == "raw_plus_guide":
                        x = np.r_[clipped_phase, centers]
                        y = np.r_[clipped_mag, guide]
                        yerr = np.r_[clipped_err, np.maximum(guide_err, 0.08)]
                    elif variant == "binned_complete":
                        x, y, yerr = centers, guide, guide_err
                    floor = 1e-6 if variant == "raw" else 0.05
                    inflation_passes = 3 if variant in {"inflation", "binned_complete"} else 0
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        fit = fit_george_template(
                            x,
                            y,
                            mag_errs=yerr,
                            nb_repeat=2,
                            optimize_hyper=True,
                            n_samples=0,
                            x_pred=np.array([0.0, 1.0]),
                            random_state=1,
                            prefer_binned=variant == "binned_complete",
                            sigma_clip=False,
                            min_noise=floor,
                            smooth=False,
                            reweight_iter=inflation_passes,
                        )
                        mean, variance = fit["gp"].predict(fit["y_train"], phase[test], return_var=True)
                    variance = np.maximum(variance, floor**2)

                residual = mag[test] - mean
                predictive_sd = np.sqrt(variance + np.maximum(err[test], 0.05) ** 2)
                z = residual / predictive_sd
                logp = -0.5 * (z**2 + np.log(2 * np.pi * predictive_sd**2))
                for ph, res, zi, lp, sd in zip(phase[test], residual, z, logp, predictive_sd):
                    rows.append(
                        {
                            "source_id": source.source_id,
                            "label": source.label,
                            "variant": variant,
                            "fold": fold,
                            "phase": float(ph),
                            "residual": float(res),
                            "z": float(zi),
                            "logp": float(lp),
                            "sd": float(sd),
                        }
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    pd.DataFrame(occupancy).to_csv(args.output.with_name("gp_occupancy_rerun.csv"), index=False)


if __name__ == "__main__":
    main()
