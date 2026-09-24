#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bath_gp_improve.py

Batch simulate light curves from observed CSVs in a folder with robust GP template fitting
and additional annotation from a period / match table.

Improvements in this version (to address user's requests):
- Avoid letting a single observed point with (apparently) tiny reported error dominate the GP fit:
  * Floor all per-point errors to --gp-min-noise (prevents interpolation-like fits).
  * Iterative reweighting: after an initial GP fit, inflate the uncertainties of points with large
    standardized residuals so they cannot pull the GP excessively (robust, redescending behavior).
  * Cap the maximum inflation factor to avoid completely removing a point.
- Help fitting "large dips" in the light curve which might be smoothed away by:
  * When fitting the GP on raw observations, optionally add the binned phase-template as soft
    "pseudo-observations" (a prior) so the GP has guidance about phase-localized features.
    These pseudo-points have a configurable prior noise (--gp-template-prior-noise) so they
    gently bias the fit toward template features but do not completely override data.
- Added CLI flags for the above behaviors and some defaults chosen conservatively.

Usage example:
/usr/local/bin/python3.12 bath_gp_improve.py --input-dir /path/to/input --out-dir /path/to/out \
    --period-table /path/to/match.csv --use-george-template --gp-fit-on-binned --sigma-clip \
    --cadence-min 30 --seed 1 --gp-min-noise 0.05 --gp-reweight-iter 3 --gp-template-prior-noise 0.08

Note: george must be installed to use --use-george-template. If not available the script falls back
to using the binned template for simulations.
"""
from __future__ import annotations
import argparse
import hashlib
import unicodedata
import sys
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Optional, Tuple, Any
import scipy.optimize as op

# Optional backend imports
try:
    import george
    from george import kernels as gkernels
    GEORGE_AVAILABLE = True
except Exception:
    GEORGE_AVAILABLE = False

# ---------------- utilities ----------------
def clean_filename(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = ''.join(ch for ch in str(s) if unicodedata.category(ch)[0] != 'C')
    s2 = s2.replace('\xa0', ' ').strip()
    return s2 or None

def make_deterministic_rng(seed: Optional[int], key: str) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    digest = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
    key_int = int.from_bytes(digest, "little", signed=False)
    seed_seq = np.random.SeedSequence([int(seed), key_int & 0xFFFFFFFF, key_int >> 32])
    return np.random.default_rng(seed_seq)

def fold_phase(times: np.ndarray, period: float, t0: float) -> np.ndarray:
    # Respect t0: fold using (times - t0) so phase 0 aligns with earliest time
    return np.mod(times - t0, period) * 1.0 / period

def cyc_interp(phase_template: np.ndarray, mag_template: np.ndarray, phase_query: np.ndarray) -> np.ndarray:
    """
    Circular interpolation for periodic templates (phase in [0,1)).
    """
    pt = np.asarray(phase_template) % 1.0
    mt = np.asarray(mag_template)
    pq = (np.asarray(phase_query) % 1.0)
    pt_ext = np.concatenate([pt, pt + 1.0])
    mt_ext = np.concatenate([mt, mt])
    order = np.argsort(pt_ext)
    pt_ext = pt_ext[order]
    mt_ext = mt_ext[order]
    return np.interp(pq, pt_ext, mt_ext)

# ---------------- sigma clipping ----------------
def sigma_clipping(x: np.ndarray, y: np.ndarray, yerr: Optional[np.ndarray]=None,
                   threshold: float=3.0, iteration: int=2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y)
    if yerr is None:
        # robust estimate if no errors provided
        yerr = np.full_like(y, np.nanmedian(np.abs(y - np.nanmedian(y))) + 1e-6)
    yerr = np.asarray(yerr)
    mask = np.isfinite(y)
    for _ in range(max(0, int(iteration))):
        ym = y[mask]
        if ym.size == 0:
            break
        med = np.median(ym)
        std = np.nanstd(ym) if np.nanstd(ym) > 0 else np.nanmedian(yerr[mask])
        new_mask = np.abs(y - med) <= threshold * (std + 1e-8)
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask
    return x[mask], y[mask], yerr[mask]

# ---------------- binned template ----------------
def build_template_weighted(phases: np.ndarray, mags: np.ndarray, mag_errs: Optional[np.ndarray]=None,
                            nbins: int=200, smooth_window: Optional[int]=11, polyorder: int=2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phases = np.asarray(phases) % 1.0
    mags = np.asarray(mags)
    if mag_errs is not None:
        mag_errs = np.asarray(mag_errs)
    bins = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.digitize(phases, bins) - 1
    phase_centers = 0.5 * (bins[:-1] + bins[1:])
    mag_template = np.full(nbins, np.nan)
    mag_err_bin = np.full(nbins, np.nan)
    for i in range(nbins):
        sel = (idx == i)
        if sel.sum() == 0:
            continue
        arr = mags[sel]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        if mag_errs is None:
            mag_template[i] = np.median(arr)
            mag_err_bin[i] = np.std(arr) if arr.size > 1 else 0.01
        else:
            e = np.asarray(mag_errs)[sel]
            valid = np.isfinite(arr)
            arrv = arr[valid]
            ev = e[valid] if valid.sum() > 0 else None
            if ev is None or ev.size == 0:
                mag_template[i] = np.median(arrv)
                mag_err_bin[i] = np.std(arrv) if arrv.size > 1 else 0.01
            else:
                w = 1.0 / np.clip(ev**2, 1e-12, None)
                # avoid very large weight ratios that indicate some reported errors are unrealistically tiny
                if np.max(w)/np.median(w) > 1e3 and arrv.size > 3:
                    mag_template[i] = np.median(arrv)
                    mag_err_bin[i] = np.std(arrv) if arrv.size > 1 else 0.01
                else:
                    mag_template[i] = np.sum(w * arrv) / np.sum(w)
                    mag_err_bin[i] = np.sqrt(1.0 / np.sum(w)) if np.sum(w) > 0 else np.std(arrv)
    nan = np.isnan(mag_template)
    if nan.all():
        raise RuntimeError("No data to build template: all bins empty")
    if nan.any():
        good = ~nan
        mag_template[nan] = np.interp(phase_centers[nan], phase_centers[good], mag_template[good])
        mag_err_bin[nan] = np.interp(phase_centers[nan], phase_centers[good], mag_err_bin[good])
    if smooth_window is not None and smooth_window >= 5:
        w = min(smooth_window, len(mag_template) if len(mag_template) % 2 == 1 else len(mag_template) - 1)
        if w % 2 == 0:
            w -= 1
        try:
            mag_template = savgol_filter(mag_template, window_length=w, polyorder=polyorder, mode='wrap')
        except Exception:
            pass
    return phase_centers, mag_template, mag_err_bin

# ---------------- george GP template (robust) ----------------
def fit_george_template(phases: np.ndarray, mags: np.ndarray, mag_errs: Optional[np.ndarray]=None,
                        nb_repeat: int=2, optimize_hyper: bool=True, n_samples: int=3,
                        x_pred: Optional[np.ndarray]=None, random_state: Optional[int]=None,
                        prefer_binned: bool=False,
                        sigma_clip: bool=False, clip_threshold: float=3.0, clip_iter: int=2,
                        min_noise: float=0.02, smooth: bool=False, smooth_window: int=11,
                        reweight_iter: int=3, reweight_threshold: float=3.0,
                        max_reweight_factor: float=8.0, center_mags: bool=True) -> dict:
    """
    Robust GP fit using george with anti-overfit measures and iterative reweighting to reduce influence
    of apparent outliers.

    Key options:
    - min_noise: floors per-point errors to avoid singular matrices / interpolation-like fits
    - reweight_iter / reweight_threshold: iteratively inflate errors of points with large standardized residuals
      so they don't dominate the fit.
    - prefer_binned: informational flag (does not change algorithmic behavior here but kept for compatibility).
    - center_mags: use the source median as a fixed GP mean. This is
      equivalent to fitting median-subtracted magnitudes and adding the
      median back to predictions.

    Returns dict with keys:
      gp, x_pred, mu, var, samples, x_train, y_train, optimized
    """
    if not GEORGE_AVAILABLE:
        raise RuntimeError("george not installed")
    rng = np.random.default_rng(random_state)
    phases = np.asarray(phases) % 1.0
    mags = np.asarray(mags)
    if mag_errs is None:
        mag_errs = np.full_like(mags, 0.01)
    mag_errs = np.asarray(mag_errs)

    # optional sigma clipping applied to the raw data before replication
    if sigma_clip:
        try:
            ph_sc, mg_sc, me_sc = sigma_clipping(phases, mags, mag_errs, threshold=clip_threshold, iteration=clip_iter)
            if mg_sc.size >= 5:
                phases = ph_sc % 1.0
                mags = mg_sc
                mag_errs = me_sc
        except Exception:
            pass

    # replicate training set to capture periodicity
    x_train = phases.copy()
    y_train = mags.copy()
    e_train = mag_errs.copy()
    for k in range(1, max(1, int(nb_repeat))):
        x_train = np.concatenate([x_train, phases + k])
        y_train = np.concatenate([y_train, mags])
        e_train = np.concatenate([e_train, mag_errs])

    # prediction grid
    if x_pred is None:
        x_pred = np.linspace(0.0, 2.0, 1000)
    x_pred = np.asarray(x_pred, dtype=float)
    x_pred = np.sort(x_pred)

    # ensure arrays are float and sorted
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    e_train = np.asarray(e_train, dtype=float)

    # floor measurement errors to min_noise (anti-overfit)
    e_train = np.clip(e_train, float(min_noise), None)

    order = np.argsort(x_train)
    x_train = x_train[order]
    y_train = y_train[order]
    e_train = e_train[order]

    # choose kernel: prefer Matern52 for smoothness; scale by data variance
    var_y = np.var(mags) if mags.size > 1 else 1.0
    try:
        kernel = var_y * gkernels.Matern52Kernel(metric=0.1, ndim=1)
    except Exception:
        try:
            kernel = var_y * gkernels.Matern52Kernel(0.1)
        except Exception:
            # fallback small RBF-like kernel if Matern unavailable
            try:
                kernel = var_y * gkernels.ExpSquaredKernel(metric=0.1)
            except Exception:
                kernel = var_y * gkernels.ConstantKernel(0.1)

    mag_offset = float(np.nanmedian(mags)) if center_mags else 0.0
    if not np.isfinite(mag_offset):
        mag_offset = 0.0
    gp = george.GP(
        kernel,
        mean=mag_offset,
        fit_mean=False,
        fit_white_noise=True,
    )

    # robust compute with small jitter if necessary
    def safe_compute(gp_obj, x, errs):
        try:
            gp_obj.compute(x, errs)
            return True
        except Exception:
            try:
                gp_obj.compute(x, errs + float(min_noise))
                return True
            except Exception:
                try:
                    gp_obj.compute(x, errs + 1e-6)
                    return True
                except Exception:
                    return False

    # initial compute before optimization
    if not safe_compute(gp, x_train, e_train):
        raise RuntimeError("Failed to compute GP covariance matrix even with jitter")

    # negative log-likelihood and gradient helpers
    def nll(p):
        try:
            gp.set_parameter_vector(p)
            ll = gp.log_likelihood(y_train, quiet=True)
            return -ll if np.isfinite(ll) else 1e25
        except Exception:
            return 1e25

    def grad_nll(p):
        try:
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(y_train, quiet=True)
        except Exception:
            return np.zeros_like(p)

    # Optionally optimize hyperparameters initially
    if optimize_hyper:
        try:
            p0 = gp.get_parameter_vector()
            results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B", options={'maxiter': 200})
            if results.success:
                gp.set_parameter_vector(results.x)
            else:
                gp.set_parameter_vector(p0)
        except Exception:
            try:
                gp.set_parameter_vector(p0)
            except Exception:
                pass

    # Iterative reweighting loop: reduce influence of points with large standardized residuals
    for itr in range(max(0, int(reweight_iter))):
        # ensure GP is computed with current errors
        if not safe_compute(gp, x_train, e_train):
            # try small jitter and continue
            try:
                gp.compute(x_train, e_train + 1e-6)
            except Exception:
                break
        try:
            pred_train, pred_var_train = gp.predict(y_train, x_train, return_var=True)
            pred_var_train = np.asarray(pred_var_train, dtype=float)
        except Exception:
            try:
                pred_train = gp.predict(y_train, x_train)
                pred_var_train = np.zeros_like(pred_train)
            except Exception:
                break

        # combined sigma: predictive uncertainty plus measurement error
        combined_sigma = np.sqrt(np.clip(pred_var_train, 0.0, np.inf) + e_train**2)
        # standardized residuals
        std_res = np.abs(y_train - pred_train) / (combined_sigma + 1e-12)

        # identify outliers by threshold
        mask_out = std_res > float(reweight_threshold)
        if not np.any(mask_out):
            # nothing to reweight; break early
            break

        # compute inflation factors and apply (clip to max_reweight_factor)
        inflation = np.clip(std_res / float(reweight_threshold), 1.0, float(max_reweight_factor))
        # only multiply errors for points exceeding the threshold (leave others)
        new_e = e_train.copy()
        new_e[mask_out] = e_train[mask_out] * inflation[mask_out]

        # if no substantial change, break
        if np.allclose(new_e, e_train, rtol=1e-6, atol=1e-9):
            break

        e_train = new_e

        # after reweighting, recompute and optionally re-optimize hyperparams
        if not safe_compute(gp, x_train, e_train):
            # revert tiny jitter and continue
            try:
                gp.compute(x_train, e_train + 1e-6)
            except Exception:
                pass
        if optimize_hyper:
            try:
                p0 = gp.get_parameter_vector()
                results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B", options={'maxiter': 150})
                if results.success:
                    gp.set_parameter_vector(results.x)
                else:
                    gp.set_parameter_vector(p0)
            except Exception:
                pass

    # Prediction grid inside training range to avoid extrapolation artifacts
    xp_min = np.min(x_train); xp_max = np.max(x_train)
    x_pred_dense = np.linspace(xp_min, xp_max, max(200, min(2000, len(x_pred))))

    try:
        pred, pred_var = gp.predict(y_train, x_pred_dense, return_var=True)
    except Exception:
        pred = gp.predict(y_train, x_pred_dense)
        pred_var = np.full_like(pred, np.nan)

    # clip extremely small predictive variance
    try:
        pred_var = np.asarray(pred_var, dtype=float)
        pred_var = np.clip(pred_var, float(min_noise)**2, None)
    except Exception:
        pass

    # optional smoothing for presentation only
    if smooth and pred.size >= 5:
        w = min(smooth_window, len(pred) if len(pred) % 2 == 1 else len(pred) - 1)
        if w >= 5:
            try:
                pred = savgol_filter(pred, window_length=w, polyorder=2, mode='wrap')
            except Exception:
                pass

    # draw conditional samples robustly
    samples_lc = None
    try:
        samples_lc = gp.sample_conditional(y_train, x_pred_dense, n_samples)
    except Exception:
        try:
            samples_lc = gp.sample_conditional(y_train, x_pred_dense, max(1, min(3, n_samples)))
        except Exception:
            samples_lc = None

    return {"gp": gp, "x_pred": x_pred_dense, "mu": pred, "var": pred_var, "samples": samples_lc,
            "x_train": x_train, "y_train": y_train, "e_train": e_train, "optimized": optimize_hyper,
            "mag_offset": mag_offset, "center_mags": bool(center_mags)}

# ---------------- estimate intrinsic noise & simulate ----------------
def estimate_intrinsic_noise(residuals: np.ndarray, phases: Optional[np.ndarray]=None,
                             mag_errs: Optional[np.ndarray]=None, nbins: int=50, method: str='phase'):
    residuals = np.asarray(residuals)
    if method == 'global':
        var_res = np.nanvar(residuals)
        mean_err2 = 0.0
        if mag_errs is not None:
            mean_err2 = np.nanmean(np.asarray(mag_errs)**2)
        var_intr = max(0.0, var_res - mean_err2)
        return np.sqrt(var_intr)
    if method == 'phase':
        if phases is None:
            raise ValueError("phases required for phase method")
        phases = np.asarray(phases) % 1.0
        bins = np.linspace(0.0, 1.0, nbins + 1)
        idx = np.digitize(phases, bins) - 1
        centers = 0.5 * (bins[:-1] + bins[1:])
        sigma_intr = np.full(nbins, np.nan)
        for i in range(nbins):
            sel = (idx == i)
            if sel.sum() < 2:
                continue
            res_bin = residuals[sel]
            var_res = np.nanvar(res_bin)
            mean_err2 = 0.0
            if mag_errs is not None:
                me = np.asarray(mag_errs)[sel]
                if np.any(np.isfinite(me)):
                    mean_err2 = np.nanmean(me[np.isfinite(me)]**2)
            var_intr = var_res - mean_err2
            sigma_intr[i] = np.sqrt(var_intr) if var_intr > 0 else 0.0
        nan = np.isnan(sigma_intr)
        if nan.all():
            global_intr = np.sqrt(max(0.0, np.nanvar(residuals) - (np.nanmean(np.asarray(mag_errs)**2) if mag_errs is not None else 0.0)))
            sigma_intr[:] = global_intr
        elif nan.any():
            good = ~nan
            sigma_intr[nan] = np.interp(centers[nan], centers[good], sigma_intr[good])
        return centers, sigma_intr
    raise ValueError("unknown method")

def simulate_lightcurve_from_template_at_times(period: float, t0: float,
                                               phase_centers: np.ndarray, mag_template: np.ndarray,
                                               times: np.ndarray,
                                               noise_intrinsic_param: Any=None,
                                               phot_err_param: Optional[Tuple[np.ndarray, np.ndarray]]=None,
                                               rng: Optional[np.random.Generator]=None) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng()
    times = np.asarray(times)
    phases = fold_phase(times, period, t0)
    mag_model = cyc_interp(phase_centers, mag_template, phases)
    n = len(times)
    mag_err_sim = np.zeros(n)
    if phot_err_param is not None:
        centers_pe, perr = phot_err_param
        if np.any(~np.isfinite(perr)):
            perr = np.nan_to_num(perr, nan=np.nanmedian(perr[np.isfinite(perr)]) if np.any(np.isfinite(perr)) else 0.01)
        mag_err_sim = np.interp(phases, centers_pe, perr, period=1.0)
    if isinstance(noise_intrinsic_param, tuple) and len(noise_intrinsic_param) == 2:
        centers_i, sigs = noise_intrinsic_param
        sigma_intr = np.interp(phases, centers_i, sigs, period=1.0)
        intrinsic_noise = rng.normal(0.0, sigma_intr, size=n)
    elif isinstance(noise_intrinsic_param, (float, np.floating)):
        intrinsic_noise = rng.normal(0.0, float(noise_intrinsic_param), size=n)
    elif noise_intrinsic_param is None:
        intrinsic_noise = np.zeros(n)
    else:
        samples_arr = np.asarray(noise_intrinsic_param)
        idx = rng.integers(0, len(samples_arr), size=n)
        intrinsic_noise = samples_arr[idx]
    meas_noise = rng.normal(0.0, mag_err_sim, size=n)
    mag_sim = mag_model + intrinsic_noise + meas_noise
    return pd.DataFrame({'time': times, 'phase': phases, 'mag_model': mag_model, 'mag_sim': mag_sim, 'mag_err_sim': mag_err_sim, 'intrinsic_noise': intrinsic_noise})

# ---------------- plotting helpers ----------------
def plot_phase_gp(phases_dup: np.ndarray, mags_dup: np.ndarray, errs_dup: np.ndarray,
                  phase_centers: np.ndarray, mag_template: np.ndarray, gp_result: dict, outpath: Path,
                  source_type: Optional[str]=None, median_mag: Optional[float]=None, std_residual: Optional[float]=None) -> None:
    x_pred = gp_result['x_pred']
    mu = gp_result['mu']; var = gp_result['var']; samples = gp_result['samples']
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.errorbar(phases_dup, mags_dup, yerr=errs_dup, ls='none', ecolor='#D2A36C', fmt='o', ms=4, markeredgecolor='k', markerfacecolor='w', alpha=0.9)
    std = np.sqrt(np.clip(var, 0.0, np.inf))
    ax.fill_between(x_pred, mu-3*std, mu+3*std, color='#D2A36C', alpha=0.25, label='GP ±3σ')
    ax.plot(x_pred, mu, color='k', lw=1.8, label='GP mean')
    if samples is not None:
        try:
            nshow = min(samples.shape[0], 6)
            for i in range(nshow):
                ax.plot(x_pred, samples[i], color='0.15', lw=0.9, alpha=0.6)
        except Exception:
            pass
    # annotate with meta
    info_lines = []
    if source_type is not None:
        info_lines.append(f"Type: {source_type}")
    if median_mag is not None and np.isfinite(median_mag):
        info_lines.append(f"Med mag: {median_mag:.3f}")
    if std_residual is not None and np.isfinite(std_residual):
        info_lines.append(f"Std resid: {std_residual:.3f}")
    if len(info_lines) > 0:
        info_text = "\n".join(info_lines)
        bbox = dict(facecolor='white', alpha=0.85, edgecolor='k', linewidth=0.5)
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, va='top', ha='left', fontsize=12, bbox=bbox)

    ax.set_xlim(0.0, 2.0); ax.invert_yaxis()
    ax.set_xlabel("Phase", fontsize=16)
    ax.set_ylabel("Mag", fontsize=16)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=12)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_time_comparison(times_obs: np.ndarray, mags_obs: np.ndarray, errs_obs: np.ndarray, sim_df: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(11,4))
    ax.errorbar(times_obs, mags_obs, yerr=errs_obs, fmt='o', ms=4, color='k', ecolor='gray', alpha=0.8, label='observed')
    ax.scatter(sim_df['time'], sim_df['mag_sim'], lw=0.8, alpha=0.8, label='simulated (mag_sim)')
    ax.invert_yaxis()
    ax.set_xlabel("MJD"); ax.set_ylabel("Mag")
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def load_lightcurve_table(path_in: Path, time_col: str, mag_col: str, magerr_col: str, default_err: float) -> pd.DataFrame:
    """
    Load one source light curve from either CSV or a compact NPY array.

    Supported NPY format: shape (4, N) or (N, 4), interpreted as
    [ra, dec, time, mag]. Since these arrays do not carry photometric
    uncertainties, a conservative constant uncertainty is assigned.
    """
    suffix = path_in.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path_in)

    if suffix != ".npy":
        raise ValueError(f"Unsupported input format: {path_in.suffix}")

    arr = np.load(path_in, allow_pickle=True)
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {path_in.name}, got shape {arr.shape}")

    if arr.shape[0] == 4:
        ra, dec, time, mag = arr
    elif arr.shape[1] == 4:
        ra, dec, time, mag = arr.T
    else:
        raise ValueError(f"Expected array shape (4, N) or (N, 4), got {arr.shape}")

    n = len(time)
    err = np.full(n, float(default_err), dtype=float)
    return pd.DataFrame({
        "ra": np.asarray(ra, dtype=float),
        "dec": np.asarray(dec, dtype=float),
        time_col: np.asarray(time, dtype=float),
        mag_col: np.asarray(mag, dtype=float),
        magerr_col: err,
    })

# ---------------- batch processing ----------------
def process_file(path_csv: Path, out_dir: Path, args: argparse.Namespace) -> dict:
    basename = path_csv.stem
    summary = {'File_Name': basename, 'period_used': None, 't0_used': None, 'n_points': 0, 'status': 'error', 'period_source': None, 'median_mag': np.nan, 'std_residual': np.nan, 'Type': None}
    print(f"Processing {basename} ...")
    try:
        df = load_lightcurve_table(path_csv, args.time_col, args.mag_col, args.magerr_col, args.gp_min_noise)
    except Exception as e:
        print(f"Failed to read {path_csv}: {e}")
        summary['status'] = 'error'
        return summary

    # required columns
    time_col = args.time_col; mag_col = args.mag_col; magerr_col = args.magerr_col
    if time_col not in df.columns or mag_col not in df.columns:
        print(f"File {path_csv} missing required columns '{time_col}' and '{mag_col}'")
        summary['status'] = 'error'
        return summary

    # compute t0 as earliest numeric time
    try:
        times_all = pd.to_numeric(df[time_col], errors='coerce')
        if times_all.dropna().size == 0:
            print(f"  No valid numeric times in {basename}; skipping.")
            summary['status'] = 'skipped'
            return summary
        t0_file = float(np.nanmin(times_all.values))
    except Exception:
        print(f"  Could not determine t0 (min time) from file {basename}; skipping.")
        summary['status'] = 'skipped'
        return summary
    summary['t0_used'] = t0_file

    # sort chronologically for downstream processing
    try:
        df_sorted = df.copy()
        df_sorted[time_col] = pd.to_numeric(df_sorted[time_col], errors='coerce')
        df_sorted = df_sorted.sort_values(by=time_col).reset_index(drop=True)
    except Exception:
        df_sorted = df.copy()

    times = pd.to_numeric(df_sorted[time_col], errors='coerce').to_numpy()
    mags = pd.to_numeric(df_sorted[mag_col], errors='coerce').to_numpy()
    mag_errs = pd.to_numeric(df_sorted[magerr_col], errors='coerce').to_numpy() if magerr_col in df_sorted.columns else None
    valid = np.isfinite(times) & np.isfinite(mags)
    if mag_errs is not None:
        mag_errs = mag_errs[valid]
    times = times[valid]
    mags = mags[valid]
    summary['n_points'] = int(len(times))
    if len(times) < 5:
        print(f"  Too few valid points in {basename} after numeric cleaning; skipping.")
        summary['status'] = 'skipped'
        return summary

    # determine period
    period_for_file = None
    period_source = None
    try:
        if hasattr(args, "period_map") and args.period_map is not None:
            if basename in args.period_map:
                period_for_file = args.period_map[basename]
                period_source = 'table'
            else:
                key_alt = basename.strip()
                if key_alt in args.period_map:
                    period_for_file = args.period_map[key_alt]
                    period_source = 'table'
    except Exception:
        pass
    if period_for_file is None and args.period is not None:
        period_for_file = args.period
        period_source = 'fallback'
    if period_for_file is None:
        print(f"  No period found for {basename}; skipping (provide --period or --period-table containing this File_Name).")
        summary['status'] = 'skipped'
        summary['period_source'] = 'none'
        return summary
    summary['period_used'] = float(period_for_file)
    summary['period_source'] = period_source

    # fold phases using per-file period and the t0_file
    phases = fold_phase(times, period_for_file, t0_file)

    # write folded-phase output
    try:
        folded_df = pd.DataFrame({time_col: times, 'phase': phases, mag_col: mags})
        if mag_errs is not None:
            folded_df[magerr_col] = mag_errs
        out_folded_csv = out_dir / f"folded_{basename}.csv"
        folded_df.to_csv(out_folded_csv, index=False)
        print(f"  Wrote folded-phase CSV: {out_folded_csv}")
    except Exception as e:
        print(f"  Failed to write folded-phase CSV for {basename}: {e}")

    # optional sigma clipping before binning
    if args.sigma_clip:
        phases_cl, mags_cl, mag_errs_cl = sigma_clipping(phases, mags, mag_errs, threshold=3.0, iteration=2)
    else:
        phases_cl, mags_cl, mag_errs_cl = phases, mags, mag_errs

    # binned template
    try:
        phase_centers, mag_template, mag_err_bin = build_template_weighted(
            phases_cl, mags_cl, mag_errs_cl if args.template_weighted else None,
            nbins=args.nbins_template, smooth_window=max(5, args.nbins_template//20))
    except Exception as e:
        print(f"  Failed to build template for {basename}: {e}")
        summary['status'] = 'error'
        return summary

    order_pc = np.argsort(phase_centers)
    phase_centers = phase_centers[order_pc]
    mag_template = mag_template[order_pc]
    mag_err_bin = mag_err_bin[order_pc]

    gp_result = None
    if args.use_george_template:
        if not GEORGE_AVAILABLE:
            print("george not available; skipping GP template fit.")
        else:
            # choose training data for GP
            if args.gp_fit_on_binned:
                # Fit on binned template only (recommended for noisy data). Those bins already have safeguards.
                gp_ph = phase_centers
                gp_mag = mag_template
                gp_err = mag_err_bin
            else:
                # Fit on raw points, but augment with soft template pseudo-observations to help
                # preserve template features (e.g., dips) that otherwise might be smoothed away.
                gp_ph_raw = phases_cl
                gp_mag_raw = mags_cl
                gp_err_raw = mag_errs_cl if mag_errs_cl is not None else np.full_like(gp_mag_raw, 0.01)

                # Optionally apply GP-specific sigma clipping on raw points before fit
                if getattr(args, 'gp_sigma_clip', False):
                    try:
                        gp_ph_tmp, gp_mag_tmp, gp_err_tmp = sigma_clipping(gp_ph_raw, gp_mag_raw, gp_err_raw, threshold=args.gp_clip_threshold, iteration=args.gp_clip_iter)
                        if gp_mag_tmp.size >= 5:
                            gp_ph_raw, gp_mag_raw, gp_err_raw = gp_ph_tmp, gp_mag_tmp, gp_err_tmp
                        else:
                            print(f"  GP sigma-clip removed too many points ({gp_mag_tmp.size}); keeping original data for GP fit.")
                    except Exception as e:
                        print(f"  GP sigma-clip failed: {e}; continuing with original data.")

                # build pseudo-observations from the binned template (soft prior)
                # use a somewhat larger uncertainty so data can override; configurable via CLI flag
                prior_noise = float(args.gp_template_prior_noise)
                # ensure template uncertainties exist and are sensible
                t_err = np.asarray(mag_err_bin)
                t_err = np.where(np.isfinite(t_err) & (t_err > 0), t_err, prior_noise)
                t_err = np.maximum(t_err, prior_noise)

                # Concatenate raw and template pseudo points; template phases are in [0,1) so OK.
                gp_ph = np.concatenate([gp_ph_raw, phase_centers])
                gp_mag = np.concatenate([gp_mag_raw, mag_template])
                gp_err = np.concatenate([gp_err_raw, t_err])

                # As a safety, if any gp_err are zero / tiny, floor them (this will also be done later)
                gp_err = np.clip(gp_err, args.gp_min_noise, None)

            try:
                gp_result = fit_george_template(
                    gp_ph, gp_mag, mag_errs=gp_err, nb_repeat=2,
                    optimize_hyper=not args.no_gp_opt, n_samples=args.gp_samples,
                    x_pred=None, random_state=args.seed, prefer_binned=args.gp_fit_on_binned,
                    sigma_clip=False,
                    clip_threshold=getattr(args, 'gp_clip_threshold', 3.0),
                    clip_iter=getattr(args, 'gp_clip_iter', 2),
                    min_noise=args.gp_min_noise,
                    smooth=args.gp_smooth,
                    smooth_window=args.gp_smooth_window,
                    reweight_iter=args.gp_reweight_iter,
                    reweight_threshold=args.gp_reweight_threshold,
                    max_reweight_factor=args.gp_reweight_max_factor)
            except Exception as e:
                print(f"  GP fit failed for {basename}: {e}")
                gp_result = None

    # obtain model at observed phases robustly
    # --- Modified here to also obtain predictive variance at observed phases so we can
    #     exclude >3-sigma points when computing residual std (as requested).
    pred_var_obs = None
    if gp_result is not None:
        try:
            out = gp_result['gp'].predict(gp_result['y_train'], phases, return_var=True)
            if isinstance(out, (tuple, list)):
                mu_obs, pred_var_obs = out[0], out[1]
            else:
                mu_obs = out
                pred_var_obs = np.zeros_like(mu_obs)
        except Exception:
            try:
                out = gp_result['gp'].predict(gp_result['y_train'], phases)
                mu_obs = out[0] if isinstance(out, (tuple, list)) else out
                pred_var_obs = np.zeros_like(mu_obs)
            except Exception:
                mu_obs = cyc_interp(phase_centers, mag_template, phases)
                pred_var_obs = np.zeros_like(mu_obs)
    else:
        mu_obs = cyc_interp(phase_centers, mag_template, phases)
        pred_var_obs = np.zeros_like(mu_obs)

    mu_obs = np.asarray(mu_obs).ravel()
    pred_var_obs = np.asarray(pred_var_obs).ravel()
    if mu_obs.shape[0] != mags.shape[0]:
        if gp_result is not None and 'x_pred' in gp_result:
            mu_obs = cyc_interp(gp_result['x_pred'] % 1.0, gp_result['mu'], phases)
            pred_var_obs = np.interp(phases, gp_result['x_pred'] % 1.0, gp_result.get('var', np.zeros_like(gp_result.get('mu'))), period=1.0)
        else:
            mu_obs = cyc_interp(phase_centers, mag_template, phases)
            pred_var_obs = np.zeros_like(mu_obs)

    residuals = mags - mu_obs

    # compute median magnitude & residual standard deviation for annotation and summary
    try:
        med_mag = float(np.nanmedian(mags[np.isfinite(mags)]))
    except Exception:
        med_mag = float('nan')
    # --- START: robust residual std calculation excluding points with |residual| > 3 * combined_sigma
    try:
        # prepare measurement errors (use zeros if not provided)
        mag_errs_arr = np.asarray(mag_errs) if mag_errs is not None else np.zeros_like(mags)
        mag_errs_arr = np.where(np.isfinite(mag_errs_arr), mag_errs_arr, 0.0)

        # predictive variance from GP may represent variance; ensure non-negative
        pred_var_obs = np.where(np.isfinite(pred_var_obs), pred_var_obs, 0.0)
        pred_var_obs = np.clip(pred_var_obs, 0.0, np.inf)

        combined_sigma = np.sqrt(pred_var_obs + mag_errs_arr**2)

        # mask of "good" points: those within 3 sigma of model prediction
        # add tiny epsilon to avoid dividing by zero issues
        eps = 1e-12
        mask_good = np.isfinite(residuals) & (np.abs(residuals) <= 3.0 * (combined_sigma + eps))

        if mask_good.sum() >= 2:
            std_resid = float(np.nanstd(residuals[mask_good]))
        else:
            # fallback: if too few points remain, compute std on all finite residuals
            std_resid = float(np.nanstd(residuals[np.isfinite(residuals)])) if np.any(np.isfinite(residuals)) else float('nan')
    except Exception:
        try:
            std_resid = float(np.nanstd(residuals[np.isfinite(residuals)]))
        except Exception:
            std_resid = float('nan')
    # --- END: robust residual std calculation

    summary['median_mag'] = med_mag
    summary['std_residual'] = std_resid

    # try to extract Type (and other metadata) from match_map if available
    source_type = None
    if getattr(args, 'match_map', None) is not None:
        mm = args.match_map
        tried_keys = [basename, basename.strip()]
        # if basename is numeric, try integer form
        if basename.isdigit():
            tried_keys.append(str(int(basename)))
        # also try filename without leading zeros
        for k in tried_keys:
            if k and k in mm:
                rowdict = mm[k]
                # rowdict values may all be strings since we read dtype=str
                source_type = rowdict.get('Type') or rowdict.get('type') or rowdict.get('TYPE')
                # also try Per or T0 if needed later
                break
    summary['Type'] = source_type

    # intrinsic scatter per phase
    centers_intr, sigma_intr = estimate_intrinsic_noise(residuals, phases=phases, mag_errs=mag_errs, nbins=50, method='phase')

    # photometric error per phase
    centers_pe = np.linspace(0.0, 1.0, 50, endpoint=False) + 0.5/50
    perr = np.zeros_like(centers_pe)
    if mag_errs is not None:
        bins = np.linspace(0.0,1,51)
        idx = np.digitize(phases, bins) - 1
        for i_ in range(len(centers_pe)):
            sel = (idx == i_)
            if sel.sum() > 0:
                arr = mag_errs[sel]
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    perr[i_] = np.median(arr)
        nan = np.isnan(perr)
        if nan.any():
            good = ~nan
            if good.any():
                perr[nan] = np.interp(centers_pe[nan], centers_pe[good], perr[good])
            else:
                perr[:] = np.nanmedian(mag_errs[np.isfinite(mag_errs)]) if np.any(np.isfinite(mag_errs)) else 0.01
    phot_err_param = (centers_pe, perr)

    # simulate at observed times
    rng = make_deterministic_rng(args.seed, basename)
    sim_df = simulate_lightcurve_from_template_at_times(period_for_file, t0_file, phase_centers, mag_template, times, noise_intrinsic_param=(centers_intr, sigma_intr), phot_err_param=phot_err_param, rng=rng)

    # write simulated CSV
    try:
        out_sim_csv = out_dir / f"sim_{basename}.csv"
        sim_df.to_csv(out_sim_csv, index=False)
        print(f"  Wrote simulated CSV: {out_sim_csv}")
    except Exception as e:
        print(f"  Failed to write simulated CSV for {basename}: {e}")

    # phase plot if GP fitted
    if gp_result is not None:
        try:
            phases_dup = np.concatenate([phases, phases + 1.0])
            mags_dup = np.concatenate([mags, mags])
            errs_dup = np.concatenate([mag_errs if mag_errs is not None else np.zeros_like(mags), mag_errs if mag_errs is not None else np.zeros_like(mags)])
            out_phase_png = out_dir / f"{basename}_phase_gp.png"
            plot_phase_gp(phases_dup, mags_dup, errs_dup, phase_centers, mag_template, gp_result, out_phase_png, source_type=source_type, median_mag=med_mag, std_residual=std_resid)
            print(f"  Wrote phase GP plot: {out_phase_png}")
        except Exception as e:
            print(f"  Failed to write phase GP plot for {basename}: {e}")

    # time comparison
    try:
        out_time_png = out_dir / f"{basename}_time_compare.png"
        plot_time_comparison(times, mags, mag_errs if mag_errs is not None else np.zeros_like(mags), sim_df, out_time_png)
        print(f"  Wrote time comparison plot: {out_time_png}")
    except Exception as e:
        print(f"  Failed to write time comparison plot for {basename}: {e}")

    # cadence-sampled GP simulation (regular sampling)
    if args.cadence_min is not None and args.cadence_min > 0 and gp_result is not None:
        try:
            step_days = float(args.cadence_min) / 1440.0  # minutes -> days
            tmin = float(np.min(times))
            tmax = float(np.max(times))
            cadence_times = np.arange(tmin, tmax + 0.5 * step_days, step_days)
            xpred = np.asarray(gp_result.get('x_pred'))
            mu_pred = np.asarray(gp_result.get('mu'))
            if xpred is None or mu_pred is None or len(xpred) == 0 or len(mu_pred) == 0:
                gp_phase_centers, gp_mu = phase_centers, mag_template
            else:
                gp_phase_centers = xpred % 1.0
                order_gp = np.argsort(gp_phase_centers)
                gp_phase_centers = gp_phase_centers[order_gp]
                gp_mu = mu_pred[order_gp]
            sim_cad_df = simulate_lightcurve_from_template_at_times(period_for_file, t0_file, gp_phase_centers, gp_mu, cadence_times, noise_intrinsic_param=(centers_intr, sigma_intr), phot_err_param=phot_err_param, rng=rng)
            out_sim_cad_csv = out_dir / f"sim_{basename}_cadence.csv"
            sim_cad_df[['time', 'mag_sim', 'mag_model']].to_csv(out_sim_cad_csv, index=False)
            print(f"  Wrote cadence simulated CSV (GP-template): {out_sim_cad_csv}")
            out_time_cad_png = out_dir / f"{basename}_time_compare_cadence.png"
            plot_time_comparison(times, mags, mag_errs if mag_errs is not None else np.zeros_like(mags), sim_cad_df, out_time_cad_png)
            print(f"  Wrote time comparison (cadence) plot: {out_time_cad_png}")
        except Exception as e:
            print(f"  Failed to produce cadence-sampled GP simulation: {e}")

    summary['status'] = 'ok'
    return summary

# ---------------- main CLI processing ----------------
def main_cli(argv: Optional[list]=None) -> None:
    parser = argparse.ArgumentParser(description="Batch simulate light curves from observed CSVs in a folder.")
    parser.add_argument('--input-dir', required=True, help='Directory containing observed CSV files (one source per CSV)')
    parser.add_argument('--pattern', default='*.csv', help='Glob pattern to select files (default: *.csv; auto-falls back to *.npy when no CSV matches)')
    parser.add_argument('--out-dir', default='sim_outputs', help='Directory to write simulated CSVs and plots')
    parser.add_argument('--time-col', default='time', help='Column name for time (MJD)')
    parser.add_argument('--mag-col', default='mag', help='Column name for magnitude')
    parser.add_argument('--magerr-col', default='mag_err', help='Column name for magnitude uncertainty')
    parser.add_argument('--period', type=float, default=None, help='Fallback period (used if period not found in --period-table)')
    parser.add_argument('--period-table', default=None, help='CSV table mapping File_Name -> Period (e.g. SSS_Per_Tab.csv)')
    parser.add_argument('--cadence-min', type=float, default=30.0, help='Cadence in minutes for GP-template regular sampling (default 30)')
    parser.add_argument('--nbins-template', type=int, default=120, help='Number of bins for phase template')
    parser.add_argument('--template-weighted', action='store_true', help='Weight bin averages by reported mag_err when building template')
    parser.add_argument('--use-george-template', action='store_true', help='Fit GP template using george')
    parser.add_argument('--gp-fit-on-binned', dest='gp_fit_on_binned', action='store_true', help='Fit GP on binned template (recommended)')
    parser.add_argument('--sigma-clip', action='store_true', help='Sigma clip raw points before binning/GP (legacy behavior)')
    parser.add_argument('--gp-sigma-clip', action='store_true', help='Apply sigma-clipping to points BEFORE GP fit (only when fitting GP on raw points)')
    parser.add_argument('--gp-clip-threshold', type=float, default=3.0, help='Sigma threshold for GP clipping (default 3.0)')
    parser.add_argument('--gp-clip-iter', type=int, default=2, help='Iterations for GP sigma clipping (default 2)')
    parser.add_argument('--gp-samples', type=int, default=5, help='Number of GP posterior samples to draw for plotting')
    parser.add_argument('--no-gp-opt', action='store_true', help='Do not optimize GP hyperparameters')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for simulations')

    # GP robustness args
    parser.add_argument('--gp-min-noise', type=float, default=0.02, help='Minimum per-point noise/jitter to use for GP fitting (default 0.02). Increase to reduce overfitting spikes.')
    parser.add_argument('--gp-smooth', action='store_true', help='Apply a small Savitzky-Golay smoothing to the GP mean prediction (presentation-level).')
    parser.add_argument('--gp-smooth-window', type=int, default=11, help='Window length for GP smoothing if --gp-smooth enabled (odd integer).')

    # New robustness controls
    parser.add_argument('--gp-reweight-iter', type=int, default=3, help='Number of iterative reweighting iterations to down-weight outliers (default 3)')
    parser.add_argument('--gp-reweight-threshold', type=float, default=3.0, help='Standardized residual threshold for reweighting (default 3.0)')
    parser.add_argument('--gp-reweight-max-factor', type=float, default=8.0, help='Max factor to inflate per-point error during reweighting (default 8.0)')
    parser.add_argument('--gp-template-prior-noise', type=float, default=0.08, help='Uncertainty to use for template pseudo-points when fitting GP on raw data (soft prior)')

    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory {input_dir} not found")
        sys.exit(1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args.time_col = args.time_col
    args.mag_col = args.mag_col
    args.magerr_col = args.magerr_col
    args.period = args.period
    args.cadence_min = args.cadence_min
    args.nbins_template = args.nbins_template
    args.template_weighted = args.template_weighted
    args.use_george_template = args.use_george_template
    args.gp_fit_on_binned = args.gp_fit_on_binned
    args.sigma_clip = args.sigma_clip
    args.gp_sigma_clip = args.gp_sigma_clip
    args.gp_clip_threshold = args.gp_clip_threshold
    args.gp_clip_iter = args.gp_clip_iter
    args.gp_samples = args.gp_samples
    args.no_gp_opt = args.no_gp_opt
    args.seed = args.seed
    args.gp_min_noise = args.gp_min_noise
    args.gp_smooth = args.gp_smooth
    args.gp_smooth_window = args.gp_smooth_window
    args.gp_reweight_iter = args.gp_reweight_iter
    args.gp_reweight_threshold = args.gp_reweight_threshold
    args.gp_reweight_max_factor = args.gp_reweight_max_factor
    args.gp_template_prior_noise = args.gp_template_prior_noise

    # load period table and optional match table if provided
    if args.period_table is not None:
        ppath = Path(args.period_table)
        if not ppath.exists():
            print(f"Period table {ppath} not found; continuing without it.")
            args.period_map = None
            args.match_map = None
        else:
            try:
                # read everything as string to preserve keys reliably
                per_df = pd.read_csv(ppath, sep=None, engine='python', dtype=str)
                cols = per_df.columns.tolist()
                if 'FileNumber' in cols:
                    fname_col = 'FileNumber'
                else:
                    fname_col = cols[0]
                # find a Per-like column
                pcols = [c for c in cols if c.strip().lower() == 'per']
                period_col = pcols[0] if len(pcols) > 0 else None
                if period_col is None:
                    for c in cols:
                        if 'per' in c.strip().lower():
                            period_col = c
                            break
                period_map = {}
                match_map = {}
                # build maps: keep row as dict of strings
                for _, row in per_df.iterrows():
                    key = str(row.get(fname_col, '')).strip()
                    # store full row as dict (string values)
                    match_map[key] = {c: (row.get(c) if row.get(c) is not None else "") for c in cols}
                    if period_col is not None:
                        try:
                            val = float(row.get(period_col, np.nan))
                            if np.isfinite(val):
                                period_map[key] = val
                        except Exception:
                            # ignore parse failures
                            pass
                args.period_map = period_map if len(period_map) > 0 else None
                args.match_map = match_map if len(match_map) > 0 else None
                if args.period_map is not None:
                    print(f"Loaded period table with {len(args.period_map)} period entries from {ppath}")
                else:
                    print(f"Loaded match table with {len(args.match_map)} rows from {ppath} (no Period column parsed)")
            except Exception as e:
                print(f"Failed to read period table {ppath}: {e}")
                args.period_map = None
                args.match_map = None
    else:
        args.period_map = None
        args.match_map = None

    files = sorted(glob.glob(str(input_dir / args.pattern)))
    if len(files) == 0 and args.pattern == '*.csv':
        npy_files = sorted(glob.glob(str(input_dir / '*.npy')))
        if len(npy_files) > 0:
            print(f"No CSV files matched; falling back to {len(npy_files)} NPY files in {input_dir}")
            files = npy_files
    if len(files) == 0:
        print("No files matched pattern in input directory.")
        sys.exit(1)

    print(f"Found {len(files)} files. Processing...")

    summary_list = []
    for fpath in files:
        try:
            res = process_file(Path(fpath), out_dir, args)
            if res is not None:
                summary_list.append(res)
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            basename = Path(fpath).stem
            summary_list.append({'File_Name': basename, 'period_used': None, 't0_used': None, 'n_points': 0, 'status': 'error', 'period_source': None, 'median_mag': np.nan, 'std_residual': np.nan, 'Type': None})

    # write summary CSV (including median_mag, std_residual, Type)
    try:
        summary_df = pd.DataFrame(summary_list, columns=['File_Name', 'period_used', 't0_used', 'n_points', 'status', 'period_source', 'median_mag', 'std_residual', 'Type'])
        summary_csv = out_dir / "t0_period_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Wrote summary of t0 and period for all files: {summary_csv}")
    except Exception as e:
        print(f"Failed to write summary CSV: {e}")

    # produce aggregate scatter: median magnitude vs residual standard deviation
    try:
        df_ok = pd.DataFrame(summary_list)
        df_plot = df_ok[df_ok['status'] == 'ok'].copy()
        if len(df_plot) > 0 and 'median_mag' in df_plot.columns and 'std_residual' in df_plot.columns:
            valid = np.isfinite(df_plot['median_mag'].astype(float)) & np.isfinite(df_plot['std_residual'].astype(float))
            df_valid = df_plot[valid]
            if len(df_valid) > 0:
                fig, ax = plt.subplots(figsize=(6,5))
                ax.scatter(df_valid['median_mag'].astype(float), df_valid['std_residual'].astype(float), alpha=0.8)
                ax.set_xlabel("Median magnitude")
                ax.set_ylabel("Residual std (obs - model)")
                ax.set_title("Median mag vs residual std (per source)")
                ax.grid(alpha=0.3)
                out_scatter = out_dir / "median_mag_vs_residual_std.png"
                plt.tight_layout()
                fig.savefig(out_scatter, dpi=200, bbox_inches='tight')
                plt.close(fig)
                print(f"Wrote aggregate median-mag vs residual-std plot: {out_scatter}")
    except Exception as e:
        print(f"Failed to produce aggregate scatter plot: {e}")

if __name__ == "__main__":
    main_cli()
