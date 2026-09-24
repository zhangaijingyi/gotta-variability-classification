"""Targeted checks for the second Reviewer 1 revision.

This script keeps the established source partitions, repairs circular feature
definitions, separates period/window descriptors from phase-shape descriptors,
and summarizes existing GP held-out predictions at the source level.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT
LC_DIR = Path(os.environ.get("GOTTA_LIGHT_CURVES", REPO_ROOT / "restricted_data/gotta_light_curves"))
MATCH_CSV = Path(os.environ.get("GOTTA_MATCH_TABLE", REPO_ROOT / "restricted_data/match.csv"))
OUT = Path(os.environ.get("GOTTA_OUTPUT_DIR", REPO_ROOT / "derived_data"))
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

SEED = 1
LABELS = ["Ecl", "Pulsating", "rotational star"]
DISPLAY = {"Ecl": "ECL", "Pulsating": "PULS", "rotational star": "ROT"}

BASIC = [
    "n_obs", "median_mag", "mean_mag", "std_mag", "amp", "mad",
    "p90_p10", "skew", "kurt", "slope", "lag1_autocorr",
]
PERIOD_WINDOW = ["period", "phase_cycles_covered", "phase_points_per_cycle"]
SHAPE = [
    "phase_amp", "phase_mad", "phase_iqr", "phase_skew", "phase_kurt",
    "phase_lag1_circular", "phase_smooth_resid_std",
    "phase_slope_mean_abs", "phase_slope_asym", "phase_peak_count",
    "phase_trough_count",
]
LOCAL = [
    "phase_peak_prominence_1", "phase_peak_prominence_2",
    "phase_peak_prom_ratio", "phase_peak_sep", "phase_width_half_amp",
    "phase_width_deep_amp", "phase_deep_run_frac_circular",
    "phase_deep_run_count_circular",
]


def circular_mean(values: np.ndarray, window: int = 5) -> np.ndarray:
    pad = window // 2
    ext = np.r_[values[-pad:], values, values[:pad]]
    return np.convolve(ext, np.ones(window) / window, mode="valid")[: len(values)]


def circular_runs(mask: np.ndarray) -> tuple[int, int]:
    mask = np.asarray(mask, bool)
    n = len(mask)
    if not mask.any():
        return 0, 0
    if mask.all():
        return 1, n
    starts = mask & ~np.roll(mask, 1)
    count = int(starts.sum())
    longest = 0
    for start in np.flatnonzero(starts):
        length = 0
        while length < n and mask[(start + length) % n]:
            length += 1
        longest = max(longest, length)
    return count, longest


def curve_features(curve: np.ndarray) -> dict[str, float]:
    c = np.asarray(curve, float)
    smooth = circular_mean(c, 5)
    grad = np.roll(c, -1) - c
    centered = c - c.mean()
    denom = np.sum(centered**2) + 1e-12
    ext = np.r_[c[-1], c, c[0]]
    peak = (ext[1:-1] > ext[:-2]) & (ext[1:-1] >= ext[2:])
    trough = (ext[1:-1] < ext[:-2]) & (ext[1:-1] <= ext[2:])
    out = {
        "phase_amp": float(np.ptp(c)),
        "phase_mad": float(np.median(np.abs(c - np.median(c)))),
        "phase_iqr": float(np.percentile(c, 75) - np.percentile(c, 25)),
        "phase_skew": float(stats.skew(c)),
        "phase_kurt": float(stats.kurtosis(c)),
        "phase_lag1_circular": float(np.sum(centered * np.roll(centered, -1)) / denom),
        "phase_smooth_resid_std": float(np.std(c - smooth)),
        "phase_slope_mean_abs": float(np.mean(np.abs(grad))),
        "phase_slope_asym": 0.0,
        "phase_peak_count": float(peak.sum()),
        "phase_trough_count": float(trough.sum()),
        "phase_peak_prominence_1": 0.0,
        "phase_peak_prominence_2": 0.0,
        "phase_peak_prom_ratio": 0.0,
        "phase_peak_sep": 0.0,
        "phase_width_half_amp": 0.0,
        "phase_width_deep_amp": 0.0,
        "phase_deep_run_frac_circular": 0.0,
        "phase_deep_run_count_circular": 0.0,
    }
    up, down = np.abs(grad[grad > 0]), np.abs(grad[grad < 0])
    if len(up) and len(down):
        out["phase_slope_asym"] = float(up.mean() / (down.mean() + 1e-12))
    idx = np.flatnonzero(peak)
    if len(idx):
        prom = np.maximum(c[idx] - np.median(c), 0.0)
        order = np.argsort(prom)[::-1]
        prom, idx = prom[order], idx[order]
        out["phase_peak_prominence_1"] = float(prom[0])
        if len(prom) > 1:
            out["phase_peak_prominence_2"] = float(prom[1])
            out["phase_peak_prom_ratio"] = float(prom[0] / (prom[1] + 1e-12))
            # When several peaks have tied prominence, choose the smallest
            # circular separation among all pairs at the top-two prominence
            # levels. This tie rule is invariant to a circular index shift.
            cutoff = prom[1]
            candidates = idx[prom >= cutoff - 1e-12]
            seps = []
            for a in range(len(candidates)):
                for b in range(a + 1, len(candidates)):
                    sep = abs(float(candidates[a] - candidates[b])) / len(c)
                    seps.append(min(sep, 1.0 - sep))
            out["phase_peak_sep"] = float(min(seps))
    bright, faint = float(c.min()), float(c.max())
    amp = faint - bright
    if amp > 0:
        half = c >= bright + 0.50 * amp
        deep = c >= bright + 0.80 * amp
        out["phase_width_half_amp"] = float(half.mean())
        out["phase_width_deep_amp"] = float(deep.mean())
        count, longest = circular_runs(deep)
        out["phase_deep_run_count_circular"] = float(count)
        out["phase_deep_run_frac_circular"] = float(longest / len(c))
    return out


def source_phase_features(times, mags, period, t0, phase_offset=0.0):
    values = {c: 0.0 for c in SHAPE + LOCAL + PERIOD_WINDOW[1:]}
    times, mags = np.asarray(times, float), np.asarray(mags, float)
    ok = np.isfinite(times) & np.isfinite(mags) & (mags != 99)
    times, mags = times[ok], mags[ok]
    if len(times) < 8 or not np.isfinite(period) or period <= 0:
        return values, None, True, 0
    phase = ((times - t0) / period + phase_offset) % 1.0
    idx = np.floor(phase * 32).astype(int)
    curve = np.full(32, np.nan)
    for j in range(32):
        if np.any(idx == j):
            curve[j] = np.median(mags[idx == j])
    occupied = int(np.isfinite(curve).sum())
    if occupied < 8:
        return values, curve, True, occupied
    centers = (np.arange(32) + 0.5) / 32
    good = np.isfinite(curve)
    curve[~good] = np.interp(centers[~good], centers[good], curve[good], period=1.0)
    values.update(curve_features(curve))
    cycles = float((times.max() - times.min()) / period)
    values["phase_cycles_covered"] = cycles
    values["phase_points_per_cycle"] = float(len(times) / max(cycles, 1.0))
    return values, curve, False, occupied


def metrics(y, p):
    return {
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
    }


def model(seed=SEED):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(random_state=seed)),
    ])


def resample_train(df, strategy, seed):
    if strategy == "natural":
        return df
    rng = np.random.default_rng(seed)
    groups = {k: g.index.to_numpy() for k, g in df.groupby("label")}
    if strategy == "undersampling":
        target = min(map(len, groups.values()))
        idx = np.concatenate([rng.choice(v, target, replace=False) for v in groups.values()])
    elif strategy == "oversampling":
        target = max(map(len, groups.values()))
        idx = np.concatenate([np.r_[v, rng.choice(v, target-len(v), replace=True)] for v in groups.values()])
    else:
        raise ValueError(strategy)
    return df.loc[idx]


def extract_features():
    old = pd.read_csv(ROOT / "augmentation_balance_natural_test/natural_qc_features_with_split.csv", dtype={"source_id": str})
    match = pd.read_csv(MATCH_CSV, dtype={"FileNumber": str}).rename(columns={"FileNumber": "source_id"})
    meta = match.set_index("source_id")
    rows, defaults, roll_checks = [], [], []
    for i, row in old.iterrows():
        sid = row.source_id
        lc = pd.read_csv(LC_DIR / f"{sid}.csv")
        per = float(meta.loc[sid, "Per"])
        t0 = float(meta.loc[sid, "T0"])
        vals, curve, default, occupied = source_phase_features(lc.time, lc.mag, per, t0)
        rec = row.to_dict()
        rec.update(vals)
        rec["period"] = per
        rec["subtype"] = meta.loc[sid, "Type"]
        rows.append(rec)
        defaults.append({"source_id": sid, "label": row.label, "default_phase": default, "occupied_bins": occupied})
        if curve is not None and np.isfinite(curve).all():
            base = curve_features(curve)
            maxdiff = 0.0
            for shift in range(1, 32):
                cur = curve_features(np.roll(curve, shift))
                maxdiff = max(maxdiff, max(abs(base[k] - cur[k]) for k in SHAPE + LOCAL))
            roll_checks.append({"source_id": sid, "max_abs_difference": maxdiff})
        if (i + 1) % 500 == 0:
            print("features", i + 1, flush=True)
    data = pd.DataFrame(rows)
    data.to_csv(OUT / "corrected_features_with_split.csv", index=False)
    pd.DataFrame(defaults).to_csv(OUT / "phase_default_and_occupancy.csv", index=False)
    pd.DataFrame(roll_checks).to_csv(OUT / "integer_roll_invariance.csv", index=False)
    variability = []
    for col in PERIOD_WINDOW + SHAPE + LOCAL:
        q = pd.to_numeric(data[col], errors="coerce")
        variability.append({"feature": col, "n_unique": int(q.nunique()), "min": q.min(), "max": q.max(), "std": q.std()})
    pd.DataFrame(variability).to_csv(OUT / "corrected_feature_variability.csv", index=False)
    return data


def run_classification(data):
    sets = {
        "basic": BASIC,
        "basic + period": BASIC + ["period"],
        "basic + period/window": BASIC + PERIOD_WINDOW,
        "period/window + shape": BASIC + PERIOD_WINDOW + SHAPE,
        "corrected full": BASIC + PERIOD_WINDOW + SHAPE + LOCAL,
    }
    train, val, test = (data[data.split.eq(x)].copy() for x in ["train", "validation", "test"])
    rows, preds = [], []
    for name, cols in sets.items():
        m = model().fit(train[cols].replace([np.inf, -np.inf], np.nan), train.label)
        for split, ev in [("validation", val), ("test", test)]:
            p = m.predict(ev[cols].replace([np.inf, -np.inf], np.nan))
            rows.append({"seed": 1, "feature_set": name, "balance": "natural", "split": split, **metrics(ev.label, p)})
            if split == "test":
                preds.extend({"seed": 1, "feature_set": name, "balance": "natural", "source_id": s, "y_true": y, "y_pred": z} for s, y, z in zip(ev.source_id, ev.label, p))
    full_cols = sets["corrected full"]
    for balance in ["natural", "undersampling", "oversampling"]:
        work = resample_train(train, balance, 1)
        m = model().fit(work[full_cols].replace([np.inf, -np.inf], np.nan), work.label)
        for split, ev in [("validation", val), ("test", test)]:
            p = m.predict(ev[full_cols].replace([np.inf, -np.inf], np.nan))
            rows.append({"seed": 1, "feature_set": "corrected full", "balance": balance, "split": split, **metrics(ev.label, p)})
            preds.extend({"seed": 1, "feature_set": "corrected full", "balance": balance, "split": split, "source_id": s, "y_true": y, "y_pred": z} for s, y, z in zip(ev.source_id, ev.label, p))
    results = pd.DataFrame(rows).drop_duplicates()
    predictions = pd.DataFrame(preds).drop_duplicates()
    results.to_csv(OUT / "corrected_ablation_seed1.csv", index=False)
    predictions.to_csv(OUT / "corrected_predictions_seed1.csv", index=False)

    repeated = []
    for seed in range(1, 11):
        tv, te = train_test_split(data.index, test_size=.2, stratify=data.label, random_state=seed)
        tr, va = train_test_split(tv, test_size=.2, stratify=data.loc[tv, "label"], random_state=seed)
        for name, cols in sets.items():
            m = model(seed).fit(data.loc[tr, cols].replace([np.inf, -np.inf], np.nan), data.loc[tr, "label"])
            p = m.predict(data.loc[te, cols].replace([np.inf, -np.inf], np.nan))
            repeated.append({"seed": seed, "feature_set": name, **metrics(data.loc[te, "label"], p)})
    repeated = pd.DataFrame(repeated)
    repeated.to_csv(OUT / "corrected_ablation_repeated.csv", index=False)
    summary = repeated.groupby("feature_set")[["accuracy", "balanced_accuracy", "macro_f1"]].agg(["mean", "std"])
    summary.to_csv(OUT / "corrected_ablation_repeated_summary.csv")

    val_full = results[(results.feature_set == "corrected full") & (results.split == "validation")]
    selected = val_full.sort_values(["balanced_accuracy", "macro_f1"], ascending=False).iloc[0]
    balance = selected.balance
    pred = predictions[(predictions.feature_set == "corrected full") & (predictions.balance == balance) & (predictions.split == "test")].copy()
    pred = pred.merge(data[["source_id", "split", "subtype", "median_mag_source"]], on="source_id", how="left", validate="one_to_one")
    pred.to_csv(OUT / "final_test_predictions_and_bins.csv", index=False)
    make_figures(pred)

    wide = repeated.pivot(index="seed", columns="feature_set", values="balanced_accuracy")
    paired = pd.DataFrame({
        "seed": wide.index,
        "period_to_shape": wide["period/window + shape"] - wide["basic + period/window"],
        "shape_to_local": wide["corrected full"] - wide["period/window + shape"],
    })
    paired.to_csv(OUT / "paired_feature_gains_repeated.csv", index=False)

    # Sensitivity of the trained selected model to arbitrary phase-origin shifts.
    work = resample_train(train, balance, 1)
    final_model = model().fit(work[full_cols].replace([np.inf, -np.inf], np.nan), work.label)
    match = pd.read_csv(MATCH_CSV, dtype={"FileNumber": str}).set_index("FileNumber")
    shift_rows = []
    offsets = [0.0, 1/64, 1/32, 3/64, 1/16]
    for _, row in test.iterrows():
        lc = pd.read_csv(LC_DIR / f"{row.source_id}.csv")
        per, t0 = float(match.loc[row.source_id, "Per"]), float(match.loc[row.source_id, "T0"])
        base = row.copy()
        for offset in offsets:
            vals, _, _, _ = source_phase_features(lc.time, lc.mag, per, t0, offset)
            rec = base.copy()
            for k, v in vals.items(): rec[k] = v
            p = final_model.predict(pd.DataFrame([rec])[full_cols].replace([np.inf, -np.inf], np.nan))[0]
            shift_rows.append({"source_id": row.source_id, "offset": offset, "prediction": p})
    shifts = pd.DataFrame(shift_rows)
    shifts.to_csv(OUT / "phase_origin_shift_predictions.csv", index=False)
    consistency = shifts.groupby("source_id").prediction.nunique().eq(1).mean()

    payload = {
        "selected_balance": balance,
        "validation_balanced_accuracy": float(selected.balanced_accuracy),
        "test": metrics(pred.y_true, pred.y_pred),
        "repeated_paired_gain_mean_sd": {
            c: {"mean": float(paired[c].mean()), "sd": float(paired[c].std())}
            for c in ["period_to_shape", "shape_to_local"]
        },
        "phase_origin_prediction_consistency": float(consistency),
    }
    (OUT / "classification_summary.json").write_text(json.dumps(payload, indent=2))


def make_figures(pred):
    y, p = pred.y_true, pred.y_pred
    cm = confusion_matrix(y, p, labels=LABELS)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, color="white" if cm[i, j] > cm.max()/2 else "#222")
    ax.set_xticks(range(3), [DISPLAY[x] for x in LABELS]); ax.set_yticks(range(3), [DISPLAY[x] for x in LABELS])
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Reference class"); ax.set_title("Test-set confusion matrix")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(OUT / "confusion_matrix_test.png", dpi=220); plt.close(fig)

    edges = np.linspace(pred.median_mag_source.min(), pred.median_mag_source.max(), 8)
    pred["mag_bin"] = pd.cut(pred.median_mag_source, edges, include_lowest=True)
    rows = []
    for b, g in pred.groupby("mag_bin", observed=False):
        n, k = len(g), int((g.y_true == g.y_pred).sum())
        ci = binomtest(k, n).proportion_ci(method="wilson") if n else None
        rows.append({"mag_bin": str(b), "median_mag": g.median_mag_source.median(), "n": n, "correct": k, "accuracy": k/n if n else np.nan, "lo": ci.low if ci else np.nan, "hi": ci.high if ci else np.nan, **{f"n_{DISPLAY[x]}": int((g.y_true == x).sum()) for x in LABELS}})
    bins = pd.DataFrame(rows); bins.to_csv(OUT / "magnitude_bin_results.csv", index=False)
    x = np.arange(len(bins)); fig, ax = plt.subplots(figsize=(8.5, 5.8)); ax2 = ax.twinx()
    bottom = np.zeros(len(bins)); colors = ["#c8dcea", "#f3dfb8", "#cfe2c5"]
    for label, color in zip(["ECL", "PULS", "ROT"], colors):
        vals = bins[f"n_{label}"] / bins.n
        ax2.bar(x, vals*100, bottom=bottom*100, width=.48, alpha=.45, color=color, label=label)
        bottom += vals
    yv = bins.accuracy.to_numpy()*100
    ax.errorbar(x, yv, yerr=np.vstack([(bins.accuracy-bins.lo)*100, (bins.hi-bins.accuracy)*100]), fmt="o-", color="#245d87", capsize=4, lw=2, label="Accuracy")
    for xi, yy, n in zip(x, yv, bins.n): ax.text(xi, min(99, yy+3), f"N={n}", ha="center", fontsize=9)
    ax.set_xticks(x, [f"{v:.2f}" for v in bins.median_mag]); ax.set_xlabel("Median magnitude"); ax.set_ylabel("Classification accuracy (%)"); ax.set_ylim(50,100)
    ax2.set_ylabel("Class composition (%)"); ax2.set_ylim(0,100); ax2.legend(loc="lower left"); fig.tight_layout(); fig.savefig(OUT / "mag_vs_accuracy_test.png", dpi=220); plt.close(fig)
    pred.to_csv(OUT / "final_test_predictions_and_bins.csv", index=False)


def gp_source_bootstrap():
    points = pd.read_csv(REPO_ROOT / "derived_data/gp_production_cv_points.csv", dtype={"source_id": str})
    per = []
    for (sid, variant), g in points.groupby(["source_id", "variant"]):
        row = {"source_id": sid, "variant": variant, "median_abs_residual": float(g.residual.abs().median())}
        if variant != "guide_only":
            row.update(mean_logp=float(g.logp.mean()), coverage1=float((g.z.abs() <= 1).mean()), coverage2=float((g.z.abs() <= 2).mean()), coverage3=float((g.z.abs() <= 3).mean()))
        per.append(row)
    per = pd.DataFrame(per); per.to_csv(OUT / "gp_metrics_by_source.csv", index=False)
    rng = np.random.default_rng(1); source_ids = sorted(per.source_id.unique()); rows=[]
    comparisons = [("binned_complete", "raw"), ("binned_complete", "floor"), ("binned_complete", "inflation")]
    wide = {metric: per.pivot(index="source_id", columns="variant", values=metric) for metric in ["mean_logp", "median_abs_residual", "coverage3"]}
    for a, b in comparisons:
        for metric, table in wide.items():
            d = (table[a] - table[b]).dropna().to_numpy()
            samples = np.array([np.mean(rng.choice(d, len(d), replace=True)) for _ in range(5000)])
            rows.append({"variant_a": a, "variant_b": b, "metric": metric, "mean_paired_difference": float(d.mean()), "ci_lo": float(np.quantile(samples,.025)), "ci_hi": float(np.quantile(samples,.975)), "n_sources": len(d)})
    pd.DataFrame(rows).to_csv(OUT / "gp_source_bootstrap_comparisons.csv", index=False)
    occ = pd.read_csv(REPO_ROOT / "derived_data/gp_occupancy.csv")
    payload = {"n_sources": int(occ.source_id.nunique()), "median_n_obs": float(occ.n.median()), "median_occupied_bins": float(occ.occupied_bins.median()), "median_occupied_fraction": float((occ.occupied_bins/120).median())}
    (OUT / "gp_occupancy_summary.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    data = extract_features()
    run_classification(data)
    gp_source_bootstrap()
    print((OUT / "classification_summary.json").read_text())
