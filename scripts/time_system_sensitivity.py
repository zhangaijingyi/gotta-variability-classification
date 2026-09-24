"""Recompute phase features and cadence matching on a common HJD(UTC) midpoint basis."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE = ROOT / "derived_data"
GOTTA_DIR = Path(os.environ.get("GOTTA_LIGHT_CURVES", ROOT / "restricted_data/gotta_light_curves"))
ZTF_DIR = Path(os.environ.get("ZTF_LIGHT_CURVES", ROOT / "restricted_data/ztf_light_curves"))
MATCH_CSV = Path(os.environ.get("GOTTA_MATCH_TABLE", ROOT / "restricted_data/match.csv"))
OUT = PACKAGE / "time_system_rerun"
OUT.mkdir(parents=True, exist_ok=True)
CONTROL = PACKAGE / "time_system_control"
CONTROL.mkdir(parents=True, exist_ok=True)

GOTTA_EXPTIME_S = 60.0
XINGLONG = EarthLocation.from_geodetic(
    lon=(117 + 34 / 60 + 38 / 3600) * u.deg,
    lat=(40 + 23 / 60 + 36 / 3600) * u.deg,
    height=900 * u.m,
)

iers.conf.auto_download = False
sys.path.insert(0, str(ROOT))


def load_revision_module():
    path = SCRIPT_DIR / "revision_r3_analysis.py"
    spec = importlib.util.spec_from_file_location("revision_r3_analysis", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.OUT = OUT
    return module


def gotta_hjd_utc(times_mjd, ra_deg, dec_deg):
    midpoint = np.asarray(times_mjd, float) + 0.5 * GOTTA_EXPTIME_S / 86400.0
    times = Time(midpoint, format="mjd", scale="utc", location=XINGLONG)
    target = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg)
    correction = times.light_travel_time(target, kind="heliocentric")
    return times.utc.mjd + correction.to_value(u.day)


def rebuild_features(revision):
    original = pd.read_csv(PACKAGE / "corrected_features_with_split.csv", dtype={"source_id": str})
    match = pd.read_csv(MATCH_CSV, dtype={"FileNumber": str}).set_index("FileNumber")
    rows = []
    correction_rows = []
    for i, row in original.iterrows():
        sid = row.source_id
        lc = pd.read_csv(GOTTA_DIR / f"{sid}.csv")
        valid = np.isfinite(lc.time) & np.isfinite(lc.mag) & (lc.mag != 99)
        lc = lc.loc[valid].copy()
        ra, dec = float(lc.ra.median()), float(lc.dec.median())
        hjd = gotta_hjd_utc(lc.time.to_numpy(float), ra, dec)
        correction_s = (hjd - lc.time.to_numpy(float)) * 86400.0
        period = float(match.loc[sid, "Per"])
        t0 = float(match.loc[sid, "T0"])
        vals, _, default, occupied = revision.source_phase_features(hjd, lc.mag, period, t0)
        rec = row.to_dict()
        rec.update(vals)
        rows.append(rec)
        correction_rows.append({
            "source_id": sid,
            "period_d": period,
            "n": len(lc),
            "correction_median_s": float(np.median(correction_s)),
            "correction_min_s": float(np.min(correction_s)),
            "correction_max_s": float(np.max(correction_s)),
            "correction_span_s": float(np.ptp(correction_s)),
            "phase_span": float(np.ptp(correction_s) / 86400.0 / period),
            "default_phase": bool(default),
            "occupied_bins": occupied,
        })
        if (i + 1) % 500 == 0:
            print("HJD features", i + 1, flush=True)
    data = pd.DataFrame(rows)
    corrections = pd.DataFrame(correction_rows)
    data.to_csv(OUT / "features_hjd_utc_midpoint.csv", index=False)
    corrections.to_csv(OUT / "gotta_time_corrections.csv", index=False)
    return data, corrections


def rerun_classification(revision, data):
    revision.run_classification(data)
    old = pd.read_csv(CONTROL / "corrected_ablation_seed1.csv")
    new = pd.read_csv(OUT / "corrected_ablation_seed1.csv")
    merged = old.merge(
        new,
        on=["seed", "feature_set", "balance", "split"],
        suffixes=("_mjd_start", "_hjd_mid"),
        validate="one_to_one",
    )
    merged.to_csv(OUT / "classification_metric_comparison.csv", index=False)

    old_pred = pd.read_csv(CONTROL / "corrected_predictions_seed1.csv", dtype={"source_id": str})
    new_pred = pd.read_csv(OUT / "corrected_predictions_seed1.csv", dtype={"source_id": str})
    old_pred = old_pred[
        (old_pred.feature_set == "corrected full")
        & (old_pred.balance == "natural")
        & (old_pred.split == "test")
    ]
    new_pred = new_pred[
        (new_pred.feature_set == "corrected full")
        & (new_pred.balance == "natural")
        & (new_pred.split == "test")
    ]
    pred = old_pred[["source_id", "y_true", "y_pred"]].merge(
        new_pred[["source_id", "y_true", "y_pred"]], on=["source_id", "y_true"],
        suffixes=("_mjd_start", "_hjd_mid"), validate="one_to_one"
    )
    pred["changed"] = pred.y_pred_mjd_start != pred.y_pred_hjd_mid
    pred.to_csv(OUT / "classification_prediction_comparison.csv", index=False)
    return merged, pred


def phase_origin_check(revision, data):
    full = revision.BASIC + revision.PERIOD_WINDOW + revision.SHAPE + revision.LOCAL
    train = data[data.split.eq("train")]
    test = data[data.split.eq("test")]
    fitted = revision.model().fit(
        train[full].replace([np.inf, -np.inf], np.nan), train.label
    )
    match = pd.read_csv(MATCH_CSV, dtype={"FileNumber": str}).set_index("FileNumber")
    rows = []
    for _, row in test.iterrows():
        lc = pd.read_csv(GOTTA_DIR / f"{row.source_id}.csv")
        lc = lc[np.isfinite(lc.time) & np.isfinite(lc.mag) & (lc.mag != 99)]
        hjd = gotta_hjd_utc(
            lc.time.to_numpy(float), float(lc.ra.median()), float(lc.dec.median())
        )
        period = float(match.loc[row.source_id, "Per"])
        t0 = float(match.loc[row.source_id, "T0"])
        for offset in [0.0, 1 / 64, 1 / 32, 3 / 64, 1 / 16]:
            values, _, _, _ = revision.source_phase_features(hjd, lc.mag, period, t0, offset)
            record = row.copy()
            for key, value in values.items():
                record[key] = value
            prediction = fitted.predict(
                pd.DataFrame([record])[full].replace([np.inf, -np.inf], np.nan)
            )[0]
            rows.append({"source_id": row.source_id, "offset": offset, "prediction": prediction})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "phase_origin_shift_predictions.csv", index=False)
    return float(result.groupby("source_id").prediction.nunique().eq(1).mean())


def rerun_cadence():
    from bath_gp_improve import build_template_weighted, fit_george_template

    selected = pd.read_csv(PACKAGE / "cadence_subset_36_composition.csv", dtype={"source_id": str})
    feature_table = pd.read_csv(
        ROOT / "augmentation_balance_natural_test/natural_qc_features_with_split.csv",
        dtype={"source_id": str},
    )
    match_table = pd.read_csv(MATCH_CSV, dtype={"FileNumber": str}).rename(
        columns={"FileNumber": "source_id"}
    )
    sample_pool = feature_table[["source_id", "label"]].merge(
        match_table, on="source_id", validate="one_to_one"
    )
    original_sample = pd.concat(
        [group.sample(n=20, random_state=1) for _, group in sample_pool.groupby("label")]
    )
    original_order = {
        str(row.source_id): i for i, (_, row) in enumerate(original_sample.iterrows())
    }
    match = pd.read_csv(MATCH_CSV, dtype={"FileNumber": str}).set_index("FileNumber")
    rows = []
    ztf_corrections = []
    for i, source in selected.iterrows():
        sid = source.source_id
        original_i = original_order[sid]
        meta = match.loc[sid]
        period, t0 = float(meta.Per), float(meta.T0)
        z = pd.read_csv(ZTF_DIR / f"{sid}.csv")
        z = z[z.filtercode.eq("zg")].copy()
        z = z[np.isfinite(z.mjd) & np.isfinite(z.hjd) & np.isfinite(z.mag) & np.isfinite(z.magerr) & (z.magerr > 0)]
        g = pd.read_csv(GOTTA_DIR / f"{sid}.csv")
        g = g[np.isfinite(g.time) & np.isfinite(g.mag) & (g.mag != 99)].copy()
        ra, dec = float(g.ra.median()), float(g.dec.median())
        g_hjd = gotta_hjd_utc(g.time.to_numpy(float), ra, dec)
        z_hjd = z.hjd.to_numpy(float) - 2400000.5
        zcorr = (z_hjd - z.mjd.to_numpy(float)) * 86400.0
        ztf_corrections.append({
            "source_id": sid,
            "period_d": period,
            "n_ztf_g": len(z),
            "correction_median_s": float(np.median(zcorr)),
            "correction_min_s": float(np.min(zcorr)),
            "correction_max_s": float(np.max(zcorr)),
            "correction_span_s": float(np.ptp(zcorr)),
            "phase_span": float(np.ptp(zcorr) / 86400.0 / period),
        })
        phase_z = ((z_hjd - t0) / period) % 1.0
        phase_g = ((g_hjd - t0) / period) % 1.0
        y, e = z.mag.to_numpy(float), z.magerr.to_numpy(float)
        # Preserve the seed assigned before sources with insufficient ZTF data were skipped.
        rng = np.random.default_rng(1 + original_i)
        for draw in range(21):
            if draw == 0:
                idx = np.arange(len(z))
                mismatch = np.nan
            else:
                used, chosen, distances = set(), [], []
                for target in phase_g:
                    delta = np.abs(phase_z - target)
                    delta = np.minimum(delta, 1.0 - delta)
                    order = np.argsort(delta)
                    candidates = [int(j) for j in order if int(j) not in used][:5]
                    pick = int(rng.choice(candidates))
                    used.add(pick)
                    chosen.append(pick)
                    distances.append(delta[pick])
                idx = np.asarray(chosen)
                mismatch = float(np.median(distances))
            centers, template, errors = build_template_weighted(
                phase_z[idx], y[idx], None, nbins=120, smooth_window=6
            )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                fit = fit_george_template(
                    centers, template, mag_errs=errors, nb_repeat=2,
                    optimize_hyper=True, n_samples=0, x_pred=np.array([0.0, 1.0]),
                    random_state=1, min_noise=0.05, reweight_iter=3,
                )
                mu = fit["gp"].predict(fit["y_train"], phase_z[idx], return_cov=False)
            rows.append({
                "source_id": sid,
                "label": source.label,
                "draw": draw,
                "n": len(idx),
                "n_gotta": len(g),
                "phase_offset": mismatch,
                "std_residual": float(np.std(y[idx] - mu)),
                "median_mag": float(np.median(y[idx])),
                "amplitude": float(np.ptp(y[idx])),
            })
        print("HJD cadence", i + 1, "of", len(selected), flush=True)
    cadence = pd.DataFrame(rows)
    ztf_corr = pd.DataFrame(ztf_corrections)
    cadence.to_csv(OUT / "repeated_cadence_hjd_utc_midpoint.csv", index=False)
    ztf_corr.to_csv(OUT / "ztf_time_corrections.csv", index=False)
    return cadence, ztf_corr


def summarize(corrections, metrics, predictions, cadence, ztf_corrections):
    shortest = corrections.sort_values("period_d").iloc[0]
    cadence_draws = cadence[cadence.draw > 0]
    full = cadence[cadence.draw == 0][["source_id", "std_residual"]].rename(columns={"std_residual": "full_std"})
    ratios = cadence_draws.merge(full, on="source_id", validate="many_to_one")
    ratios["ratio"] = ratios.std_residual / ratios.full_std
    natural = metrics[
        (metrics.feature_set == "corrected full")
        & (metrics.balance == "natural")
        & (metrics.split == "test")
    ].iloc[0]
    old_cadence = pd.read_csv(CONTROL / "repeated_cadence_mjd_start.csv", dtype={"source_id": str})
    old_draws = old_cadence[old_cadence.draw > 0]
    old_full = old_cadence[old_cadence.draw == 0][["source_id", "std_residual"]].rename(
        columns={"std_residual": "full_std"}
    )
    old_ratios = old_draws.merge(old_full, on="source_id", validate="many_to_one")
    old_ratios["ratio"] = old_ratios.std_residual / old_ratios.full_std
    summary = {
        "time_definition": {
            "gotta_input": "MJD(UTC), exposure start",
            "gotta_conversion": "add 30 s for a 60 s exposure, then add heliocentric light-travel time at Xinglong",
            "ztf_input_used_by_original_code": "IRSA light-curve mjd, exposure start",
            "ztf_corrected": "IRSA hjd - 2400000.5, exposure midpoint and HJD(UTC)",
        },
        "gotta_correction_seconds_all_sources": {
            "median_of_source_medians": float(corrections.correction_median_s.median()),
            "range": [float(corrections.correction_min_s.min()), float(corrections.correction_max_s.max())],
            "median_source_span": float(corrections.correction_span_s.median()),
            "maximum_source_span": float(corrections.correction_span_s.max()),
        },
        "shortest_period_source": {
            "source_id": str(shortest.source_id),
            "period_d": float(shortest.period_d),
            "correction_median_s": float(shortest.correction_median_s),
            "correction_span_s": float(shortest.correction_span_s),
            "phase_span": float(shortest.phase_span),
            "midpoint_30s_phase": float(30.0 / 86400.0 / shortest.period_d),
            "utc_to_tdb_69p184s_phase": float(69.184 / 86400.0 / shortest.period_d),
            "maximum_heliocentric_499s_phase": float(499.0 / 86400.0 / shortest.period_d),
        },
        "classification_fixed_natural_training": {
            "accuracy_mjd_start": float(natural.accuracy_mjd_start),
            "accuracy_hjd_mid": float(natural.accuracy_hjd_mid),
            "balanced_accuracy_mjd_start": float(natural.balanced_accuracy_mjd_start),
            "balanced_accuracy_hjd_mid": float(natural.balanced_accuracy_hjd_mid),
            "macro_f1_mjd_start": float(natural.macro_f1_mjd_start),
            "macro_f1_hjd_mid": float(natural.macro_f1_hjd_mid),
            "changed_test_predictions": int(predictions.changed.sum()),
            "n_test": int(len(predictions)),
        },
        "ztf_correction_seconds_cadence_subset": {
            "range": [float(ztf_corrections.correction_min_s.min()), float(ztf_corrections.correction_max_s.max())],
            "median_source_span": float(ztf_corrections.correction_span_s.median()),
            "maximum_source_span": float(ztf_corrections.correction_span_s.max()),
        },
        "cadence_hjd_midpoint": {
            "n_sources": int(cadence.source_id.nunique()),
            "n_draws": int(len(cadence_draws)),
            "median_phase_mismatch": float(cadence_draws.phase_offset.median()),
            "median_sourcewise_residual_ratio": float(ratios.groupby("source_id").ratio.median().median()),
        },
        "cadence_mjd_start_control": {
            "n_sources": int(old_cadence.source_id.nunique()),
            "n_draws": int(len(old_draws)),
            "median_phase_mismatch": float(old_draws.phase_offset.median()),
            "median_sourcewise_residual_ratio": float(old_ratios.groupby("source_id").ratio.median().median()),
        },
    }
    (OUT / "time_system_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    revision = load_revision_module()
    control_feature_path = CONTROL / "features_mjd_start.csv"
    if not (CONTROL / "corrected_ablation_seed1.csv").exists():
        control_data = pd.read_csv(control_feature_path, dtype={"source_id": str})
        revision.OUT = CONTROL
        revision.run_classification(control_data)
        revision.OUT = OUT
    feature_path = OUT / "features_hjd_utc_midpoint.csv"
    correction_path = OUT / "gotta_time_corrections.csv"
    if feature_path.exists() and correction_path.exists():
        data = pd.read_csv(feature_path, dtype={"source_id": str})
        corrections = pd.read_csv(correction_path, dtype={"source_id": str})
    else:
        data, corrections = rebuild_features(revision)

    metric_path = OUT / "classification_metric_comparison.csv"
    prediction_path = OUT / "classification_prediction_comparison.csv"
    if metric_path.exists() and prediction_path.exists():
        metrics = pd.read_csv(metric_path)
        predictions = pd.read_csv(prediction_path, dtype={"source_id": str})
    else:
        metrics, predictions = rerun_classification(revision, data)

    phase_origin_path = OUT / "phase_origin_shift_predictions.csv"
    if not phase_origin_path.exists():
        phase_origin_check(revision, data)

    cadence_path = OUT / "repeated_cadence_hjd_utc_midpoint.csv"
    ztf_correction_path = OUT / "ztf_time_corrections.csv"
    if cadence_path.exists() and ztf_correction_path.exists():
        cadence = pd.read_csv(cadence_path, dtype={"source_id": str})
        ztf_corrections = pd.read_csv(ztf_correction_path, dtype={"source_id": str})
    else:
        cadence, ztf_corrections = rerun_cadence()
    summarize(corrections, metrics, predictions, cadence, ztf_corrections)


if __name__ == "__main__":
    main()
