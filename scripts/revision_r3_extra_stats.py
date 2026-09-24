from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline

OUT = Path(__file__).resolve().parents[1] / "derived_data"
BASIC = ["n_obs", "median_mag", "mean_mag", "std_mag", "amp", "mad", "p90_p10", "skew", "kurt", "slope", "lag1_autocorr"]
FULL = BASIC + ["period", "phase_cycles_covered", "phase_points_per_cycle", "phase_amp", "phase_mad", "phase_iqr", "phase_skew", "phase_kurt", "phase_lag1_circular", "phase_smooth_resid_std", "phase_slope_mean_abs", "phase_slope_asym", "phase_peak_count", "phase_trough_count", "phase_peak_prominence_1", "phase_peak_prominence_2", "phase_peak_prom_ratio", "phase_peak_sep", "phase_width_half_amp", "phase_width_deep_amp", "phase_deep_run_frac_circular", "phase_deep_run_count_circular"]


def metric(y, p):
    return np.array([accuracy_score(y, p), balanced_accuracy_score(y, p), f1_score(y, p, average="macro")])


pred = pd.read_csv(OUT / "corrected_predictions_seed1.csv", dtype={"source_id": str})
selected_balance = json.loads((OUT / "classification_summary.json").read_text())["selected_balance"]
sel = pred[(pred.feature_set == "corrected full") & (pred.balance == selected_balance) & (pred.split == "test")].copy()
rng = np.random.default_rng(1)
vals = []
for _ in range(5000):
    i = rng.integers(0, len(sel), len(sel))
    vals.append(metric(sel.y_true.iloc[i], sel.y_pred.iloc[i]))
vals = np.asarray(vals)
summary = {k: [float(x) for x in np.percentile(vals[:, j], [2.5, 97.5])] for j, k in enumerate(["accuracy", "balanced_accuracy", "macro_f1"])}
(OUT / "selected_classifier_bootstrap.json").write_text(json.dumps(summary, indent=2))

bench = pd.read_csv(OUT / "corrected_features_with_split.csv", dtype={"source_id": str})
keep = {"EA", "EW", "RR", "RRc", "BYDra"}
bench = bench[bench.subtype.isin(keep)].copy()
tr, va, te = (bench[bench.split == x] for x in ["train", "validation", "test"])
groups = {k: g.index.to_numpy() for k, g in tr.groupby("label")}
if selected_balance == "undersampling":
    target = min(map(len, groups.values()))
    idx = np.concatenate([rng.choice(v, target, replace=False) for v in groups.values()])
elif selected_balance == "oversampling":
    target = max(map(len, groups.values()))
    idx = np.concatenate([np.r_[v, rng.choice(v, target - len(v), replace=True)] for v in groups.values()])
else:
    idx = tr.index.to_numpy()
model = Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", GradientBoostingClassifier(random_state=1))])
model.fit(tr.loc[idx, FULL], tr.loc[idx, "label"])
rows = []
for name, d in [("validation", va), ("test", te)]:
    p = model.predict(d[FULL]); a, b, f = metric(d.label, p)
    rows.append({"split": name, "n": len(d), "accuracy": a, "balanced_accuracy": b, "macro_f1": f})
pd.DataFrame(rows).to_csv(OUT / "restricted_subtype_corrected.csv", index=False)

# Paired comparisons on the same test sources among balance treatments.
wide = pred[(pred.feature_set == "corrected full") & pred.split.eq("test")].pivot(index="source_id", columns="balance", values=["y_true", "y_pred"])
rows = []
for other in [x for x in ["natural", "undersampling", "oversampling"] if x != selected_balance]:
    y = wide[("y_true", selected_balance)]
    a = wide[("y_pred", selected_balance)].eq(y)
    b = wide[("y_pred", other)].eq(y)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(min(n10, n01), n10 + n01, .5).pvalue if n10 + n01 else 1.0
    rows.append({"selected": selected_balance, "other": other, "selected_only_correct": n10, "other_only_correct": n01, "mcnemar_exact_p": p})
pd.DataFrame(rows).to_csv(OUT / "balance_mcnemar_corrected.csv", index=False)
print(json.dumps(summary, indent=2))
