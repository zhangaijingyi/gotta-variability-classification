from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_r3_bench_boundary import binned_profile, fit_duplicated

SOURCES = [("12629618", "ECL (EA)"), ("16767076", "PULS (RR)"), ("24139579", "ROT (BYDra)")]

parser = argparse.ArgumentParser()
parser.add_argument("--light-curves", type=Path, required=True)
parser.add_argument("--match-table", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "derived_data/gp_reconstruction_examples.png")
args = parser.parse_args()
LC = args.light_curves
MATCH = pd.read_csv(args.match_table, dtype={"FileNumber": str}).set_index("FileNumber")
OUT = args.output
OUT.parent.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.2), sharex=True)
colors = ["#0072B2", "#D55E00", "#009E73"]
legend_handles = None
legend_labels = None
for ax, (sid, label), color in zip(axes, SOURCES, colors):
    d = pd.read_csv(LC / f"{sid}.csv")
    d = d.apply(pd.to_numeric, errors="coerce").dropna(subset=["time", "mag"])
    period, t0 = float(MATCH.loc[sid, "Per"]), float(MATCH.loc[sid, "T0"])
    phase = ((d.time.to_numpy(float) - t0) / period) % 1.0
    pbin, mbin, ebin = binned_profile(phase, d.mag.to_numpy(float))
    gp = fit_duplicated(pbin, mbin, ebin, 0.0)
    grid = np.linspace(0, 1, 400, endpoint=False)
    mu, sd = gp.predict(grid[:, None], return_std=True)
    x = np.r_[grid, grid + 1]
    mm, ss = np.tile(mu, 2), np.tile(sd, 2)
    ax.fill_between(x, mm - 3 * ss, mm + 3 * ss, color=color, alpha=.18, label=r"latent $\pm3\sigma$")
    ax.plot(x, mm, color="black", lw=1.7, label="posterior mean")
    ax.scatter(np.r_[phase, phase + 1], np.tile(d.mag.to_numpy(float), 2), s=18, facecolor="white", edgecolor=color, linewidth=.8, zorder=3, label="observations")
    ax.set_title(f"{label}; source {sid}", loc="left", fontsize=10, pad=5)
    if legend_handles is None:
        legend_handles, legend_labels = ax.get_legend_handles_labels()
    ax.invert_yaxis()
    ax.set_ylabel("Magnitude")
    ax.grid(alpha=.16, linestyle=":")
fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.992),
    frameon=False,
    ncol=3,
    fontsize=8,
)
axes[-1].set_xlabel("Phase")
axes[-1].set_xlim(0, 2)
fig.subplots_adjust(top=0.935, bottom=0.075, left=0.11, right=0.985, hspace=0.28)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
