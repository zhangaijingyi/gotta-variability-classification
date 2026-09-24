from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("GOTTA_OUTPUT_DIR", ROOT / "derived_data"))
FEATURES = OUT / "corrected_features_with_split.csv"
POINTS = ROOT / "derived_data/gp_production_cv_points.csv"
MATCH = Path(os.environ.get("GOTTA_MATCH_TABLE", ROOT / "restricted_data/match.csv"))
LC_DIR = Path(os.environ.get("GOTTA_LIGHT_CURVES", ROOT / "restricted_data/gotta_light_curves"))
SEED = 1

BASIC = ["n_obs", "median_mag", "mean_mag", "std_mag", "amp", "mad", "p90_p10", "skew", "kurt", "slope", "lag1_autocorr"]
PERIOD_WINDOW = ["period", "phase_cycles_covered", "phase_points_per_cycle"]
SHAPE = ["phase_amp", "phase_mad", "phase_iqr", "phase_skew", "phase_kurt", "phase_lag1_circular", "phase_smooth_resid_std", "phase_slope_mean_abs", "phase_slope_asym", "phase_peak_count", "phase_trough_count"]
LOCAL = ["phase_peak_prominence_1", "phase_peak_prominence_2", "phase_peak_prom_ratio", "phase_peak_sep", "phase_width_half_amp", "phase_width_deep_amp", "phase_deep_run_frac_circular", "phase_deep_run_count_circular"]
FULL = BASIC + PERIOD_WINDOW + SHAPE + LOCAL


def score(y, p):
    return {
        "accuracy": accuracy_score(y, p),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro"),
    }


def classifier_benchmark():
    df = pd.read_csv(FEATURES)
    train, val, test = (df[df.split == x] for x in ["train", "validation", "test"])
    models = {
        "nearest centroid": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", NearestCentroid())]),
        "logistic regression": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, random_state=SEED))]),
        "linear discriminant analysis": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", LinearDiscriminantAnalysis())]),
        "random forest": Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1))]),
        "gradient boosting": Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", GradientBoostingClassifier(random_state=SEED))]),
    }
    rows = []
    for name, model in models.items():
        model.fit(train[FULL], train.label)
        for split_name, part in [("validation", val), ("test", test)]:
            pred = model.predict(part[FULL])
            rows.append({"classifier": name, "split": split_name, **score(part.label, pred)})
    pd.DataFrame(rows).to_csv(OUT / "corrected_classifier_benchmark.csv", index=False)


def fit_duplicated(phases, mags, errs, origin):
    p = (np.asarray(phases, float) + origin) % 1.0
    y = np.asarray(mags, float)
    e = np.clip(np.asarray(errs, float), 0.05, None)
    x = np.concatenate([p, p + 1.0])
    yy, ee = np.tile(y, 2), np.tile(e, 2)
    var = max(float(np.var(yy)), 1e-4)
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(var, constant_value_bounds="fixed") * Matern(length_scale=.12, length_scale_bounds="fixed", nu=2.5),
        alpha=ee ** 2, optimizer=None, normalize_y=True, random_state=SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(x[:, None], yy)
    return gp


def binned_profile(phases, mags):
    p = np.asarray(phases, float) % 1.0
    y = np.asarray(mags, float)
    edges = np.linspace(0.0, 1.0, 121)
    centers = (edges[:-1] + edges[1:]) / 2
    idx = np.clip(np.digitize(p, edges) - 1, 0, 119)
    val = np.full(120, np.nan)
    err = np.full(120, np.nan)
    for j in range(120):
        z = y[idx == j]
        if len(z):
            val[j] = np.median(z)
            err[j] = np.std(z) if len(z) > 1 else 0.01
    good = np.isfinite(val)
    val[~good] = np.interp(centers[~good], centers[good], val[good], period=1.0)
    egood = np.isfinite(err)
    err[~egood] = np.interp(centers[~egood], centers[egood], err[egood], period=1.0)
    val = savgol_filter(val, 5, 2, mode="wrap")
    return centers, val, np.maximum(err, 0.05)


def gp_boundary_origin():
    d = pd.read_csv(POINTS)
    d = d[d.variant == "binned_complete"].copy()
    d["source_id"] = d.source_id.astype(str)
    chosen = []
    for cls, g in d.groupby("label"):
        ids = sorted(g.source_id.astype(str).unique())[:10]
        chosen.extend(ids)
    mt = pd.read_csv(MATCH, dtype={"FileNumber": str}).set_index("FileNumber")
    rows = []
    grid = np.linspace(0, 1, 129, endpoint=False)
    origins = [0.0, 0.125, 0.25, 0.5]
    for sid in chosen:
        raw = pd.read_csv(LC_DIR / f"{sid}.csv")
        raw = raw.apply(pd.to_numeric, errors="coerce").dropna(subset=["time", "mag"])
        period = float(mt.loc[sid, "Per"])
        t0 = float(mt.loc[sid, "T0"])
        phase = ((raw.time.to_numpy(float) - t0) / period) % 1.0
        pbin, mbin, ebin = binned_profile(phase, raw.mag.to_numpy(float))
        cls = d.loc[d.source_id == sid, "label"].iloc[0]
        base_mu = base_sd = None
        for origin in origins:
            gp = fit_duplicated(pbin, mbin, ebin, origin)
            x0 = (grid + origin) % 1.0
            # Predictions are taken from the first represented cycle.
            mu, sd = gp.predict(x0[:, None], return_std=True)
            mu1, sd1 = gp.predict((x0 + 1.0)[:, None], return_std=True)
            seam_mu = abs(float(gp.predict(np.array([[origin]]))[0] - gp.predict(np.array([[origin + 1.0]]))[0]))
            seam_sd = abs(float(gp.predict(np.array([[origin]]), return_std=True)[1][0] - gp.predict(np.array([[origin + 1.0]]), return_std=True)[1][0]))
            if origin == 0:
                base_mu, base_sd = mu, sd
            rows.append({
                "source_id": sid, "broad_class": cls, "origin_shift": origin,
                "max_cycle_mean_difference": float(np.max(np.abs(mu - mu1))),
                "max_cycle_sd_difference": float(np.max(np.abs(sd - sd1))),
                "seam_mean_difference": seam_mu, "seam_sd_difference": seam_sd,
                "median_origin_mean_difference": float(np.median(np.abs(mu - base_mu))),
                "median_origin_sd_difference": float(np.median(np.abs(sd - base_sd))),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "gp_boundary_origin_detail.csv", index=False)
    s = out.groupby("origin_shift").agg(
        n_sources=("source_id", "nunique"),
        median_cycle_mean_difference=("max_cycle_mean_difference", "median"),
        median_cycle_sd_difference=("max_cycle_sd_difference", "median"),
        median_seam_mean_difference=("seam_mean_difference", "median"),
        median_seam_sd_difference=("seam_sd_difference", "median"),
        median_origin_mean_difference=("median_origin_mean_difference", "median"),
        median_origin_sd_difference=("median_origin_sd_difference", "median"),
    ).reset_index()
    s.to_csv(OUT / "gp_boundary_origin_summary.csv", index=False)
    (OUT / "gp_boundary_origin_summary.json").write_text(json.dumps(s.to_dict("records"), indent=2))


if __name__ == "__main__":
    classifier_benchmark()
    gp_boundary_origin()
    print((OUT / "gp_boundary_origin_summary.json").read_text())
