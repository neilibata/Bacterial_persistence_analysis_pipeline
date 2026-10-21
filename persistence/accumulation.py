"""
SGB accumulation curves with Hill–Michaelis-Menten fitting.

Analyses performed per reference, per breadth threshold (1/10/30/50/90%):
  A. Pooled: Industrial vs Non-Industrial (all + Unhealthy)
     → accumulation curve plots + MM fits + parameter-vs-N_max sweep
  B. Continent: Industrial & Non-Industrial by continent
     → accumulation curve plots + parameter-vs-N_max sweep

Figures produced match the style of the attached reference images.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plot_utils import (
    apply_oi_style,
    page_figsize,
    square_figsize,
    panel_grid,
    savefig_dual,
    share_axis_labels,
)
apply_oi_style()
from scipy.optimize import curve_fit
from math import isfinite

from . import config
from .common import classify_group, classify_macro_group
from .detection import build_presence, flat_bp_specs

plt.rcParams.update({
    "figure.dpi": 140,
    "axes.grid": True,
    "font.size": 18,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
})


# ════════════════════════════════════════════════════════════════════
#  HILL–MM MODEL (free exponent c, logistic-constrained asymptote)
# ════════════════════════════════════════════════════════════════════

def _logistic(a_raw):
    """Logistic transform: maps (-∞, +∞) → (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(a_raw, -20, 20)))


def mm_model_free(log_x, a_raw, b, c):
    """
    Hill–MM in fractional space:
        a_frac = logistic(a_raw) ∈ (0,1)
        y_frac = a_frac * x^c / (b^c + x^c)
    """
    x = np.exp(log_x)
    x_c = np.power(x, c)
    b_c = np.power(b, c)
    a_frac = _logistic(a_raw)
    return (a_frac * x_c) / (b_c + x_c)


def _mm_init_bounds(x, y_frac):
    """Initial guess and bounds for (a_raw, b, c)."""
    x = np.asarray(x, float)
    y_frac = np.asarray(y_frac, float)
    y_max = float(np.nanmax(y_frac)) if np.isfinite(np.nanmax(y_frac)) else 0.5

    a0_frac = min(max(y_max, 0.3), 0.95)
    a0 = np.log(a0_frac / (1.0 - a0_frac))

    c0 = 0.7
    try:
        idx = int(np.nanargmin(np.abs(y_frac - 0.5 * y_max)))
        N_half_est = max(1.0, float(x[idx]))
    except Exception:
        N_half_est = float(np.nanmedian(x)) if np.isfinite(np.nanmedian(x)) else 5.0

    b0 = max(1.0, min(N_half_est, 1e5))
    p0 = [a0, b0, c0]
    bounds = ([-8.0, 1.0, 0.1], [8.0, 1e7, 1.5])
    return p0, bounds


def fit_mm_free(x, y, s, n_total, min_points=5, use_weights=False):
    """
    Fit y = (a_frac * x^c)/(b^c + x^c) in fractional space.

    Returns:
        params = (a_abs, b, c)    — a_abs in absolute richness
        errs   = (a_abs, se_a, c, se_c)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    s = np.asarray(s, float)

    m = np.isfinite(x) & np.isfinite(y) & (x >= 1)
    x, y, s = x[m], y[m], s[m]
    if x.size < min_points:
        return None, (np.nan, np.nan, np.nan, np.nan)

    n_total = float(n_total) if n_total > 0 else 1.0
    y_frac = y / n_total
    s_frac = s / n_total

    p0, bounds = _mm_init_bounds(x, y_frac)

    sigma, abs_sig = None, False
    if use_weights:
        good = np.isfinite(s_frac) & (s_frac > 0)
        if np.any(good):
            s_use = s_frac.copy()
            s_use[~good] = np.nanmin(s_use[good])
            sigma, abs_sig = s_use, True

    try:
        popt, pcov = curve_fit(mm_model_free, np.log(x), y_frac,
                               p0=p0, bounds=bounds,
                               sigma=sigma, absolute_sigma=abs_sig,
                               maxfev=30000)
    except Exception:
        try:
            popt, pcov = curve_fit(mm_model_free, np.log(x), y_frac,
                                   p0=p0, bounds=bounds, maxfev=30000)
        except Exception:
            return None, (np.nan, np.nan, np.nan, np.nan)

    a_raw, b, c = map(float, popt)
    a_frac = _logistic(a_raw)

    se_c = np.nan
    se_a_frac = np.nan
    if pcov is not None and pcov.shape == (3, 3):
        if pcov[2, 2] >= 0:
            se_c = np.sqrt(pcov[2, 2])
        if pcov[0, 0] >= 0:
            se_a_raw = np.sqrt(pcov[0, 0])
            dadr = a_frac * (1.0 - a_frac)
            se_a_frac = abs(dadr) * se_a_raw

    a_abs = a_frac * n_total
    se_a_abs = se_a_frac * n_total if np.isfinite(se_a_frac) else np.nan

    return (a_abs, b, c), (a_abs, se_a_abs, c, se_c)


def _late_n_sigma(x, s, strength=12.0, power=2.0):
    """Shrink sigma at large N so tail points carry more fitting weight."""
    x = np.asarray(x, float)
    s = np.asarray(s, float).copy()
    if x.size == 0:
        return s
    xmax = max(float(np.nanmax(x)), 1.0)
    w = 1.0 + strength * np.power(np.clip(x / xmax, 0.0, 1.0), power)
    return s / np.sqrt(w)


def fit_mm_free_late_weighted(x, y, s, n_total):
    return fit_mm_free(x, y, _late_n_sigma(x, s), n_total, use_weights=True)


def richards_logN_abs(log_x, lower, upper, k, x0, nu):
    """4-parameter Richards curve in log-N space."""
    z = np.clip(-k * (log_x - x0), -60.0, 60.0)
    return lower + (upper - lower) / np.power(1.0 + np.exp(z), 1.0 / nu)


def fit_richards_weighted(x, y, s, n_total, min_points=5,
                          late_weight=False):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    m = np.isfinite(x) & np.isfinite(y) & (x >= 1)
    x, y, s = x[m], y[m], s[m]
    if x.size < min_points:
        return None, np.nan

    logx = np.log(x)
    sigma = _late_n_sigma(x, s) if late_weight else s.copy()
    good = np.isfinite(sigma) & (sigma > 0)
    if np.any(good):
        sigma[~good] = np.nanmin(sigma[good])
    else:
        sigma = None

    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    n_total = float(max(n_total, y_max + 1.0))
    p0 = [
        max(0.0, y_min * 0.5),
        min(n_total, max(y_max * 1.05, y_max + 1.0)),
        1.0,
        float(np.nanmedian(logx)),
        1.0,
    ]
    bounds = (
        [0.0, y_max, 0.02, float(np.nanmin(logx)) - 5.0, 0.1],
        [y_max, n_total, 20.0, float(np.nanmax(logx)) + 5.0, 10.0],
    )
    try:
        popt, _ = curve_fit(
            richards_logN_abs, logx, y, p0=p0, bounds=bounds,
            sigma=sigma, absolute_sigma=sigma is not None, maxfev=50000)
    except Exception:
        try:
            popt, _ = curve_fit(
                richards_logN_abs, logx, y, p0=p0, bounds=bounds,
                maxfev=50000)
        except Exception:
            return None, np.nan

    yhat = richards_logN_abs(logx, *popt)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return tuple(map(float, popt)), float(r2)


def fit_richards_late_weighted(x, y, s, n_total, min_points=5):
    return fit_richards_weighted(
        x, y, s, n_total, min_points=min_points, late_weight=True)


def compute_r2(x, y, n_total, params):
    """R² for Hill-MM fit in absolute richness space."""
    if params is None:
        return np.nan
    a_abs, b, c = params
    if not all(np.isfinite(v) for v in [a_abs, b, c]):
        return np.nan

    x, y = np.asarray(x, float), np.asarray(y, float)
    if 0 < a_abs < n_total:
        a_raw = np.log((a_abs / n_total) / (1 - a_abs / n_total))
    else:
        a_raw = 0.0

    yhat = mm_model_free(np.log(x), a_raw, b, c) * n_total
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 2:
        return np.nan
    ss_res = np.sum((y[m] - yhat[m]) ** 2)
    ss_tot = np.sum((y[m] - np.mean(y[m])) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def mm_model_fixed_a(log_x, b, c, a_frac):
    """
    Hill-MM with a fixed (not fitted).  Only b and c are free parameters.
    a_frac = fixed_a / n_total is baked in at call time via a closure or
    partial; this signature is compatible with scipy curve_fit.
    """
    x   = np.exp(log_x)
    x_c = np.power(np.maximum(x, 1e-12), c)
    b_c = np.power(max(b, 1e-12), c)
    return (a_frac * x_c) / (b_c + x_c)


def fit_mm_fixed_a(x, y, s, n_total, fixed_a, min_points=5, use_weights=False):
    """
    Fit  S(N) = fixed_a · Nᶜ / (bᶜ + Nᶜ)  with a held constant.
    Only b and c are free; this is a 2-parameter fit vs the 3-parameter free fit.

    fixed_a may exceed n_total (e.g. full catalogue size 3974 > SGBs detected
    in a lifestyle subset) — the fractional model handles this gracefully because
    y_frac = y / n_total can exceed fixed_a / n_total only if the data exceed
    the fixed asymptote, which would indicate a mis-specified fixed_a.

    Returns:
        params = (fixed_a, b, c)
        errs   = (fixed_a, np.nan, c, se_c)
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    s = np.asarray(s, float)

    m = np.isfinite(x) & np.isfinite(y) & (x >= 1)
    x, y, s = x[m], y[m], s[m]
    if x.size < min_points:
        return None, (np.nan, np.nan, np.nan, np.nan)

    n_total   = float(n_total) if n_total > 0 else 1.0
    a_frac    = fixed_a / n_total   # may be > 1 if fixed_a > n_total (fine)
    y_frac    = y / n_total
    s_frac    = s / n_total

    # Initial guess: b from the half-saturation point at a_frac/2
    half_target = a_frac / 2.0
    idx = int(np.nanargmin(np.abs(y_frac - half_target))) if y_frac.size > 0 else 0
    b0  = max(1.0, min(float(x[idx]), 1e5))
    c0  = 0.6
    p0  = [b0, c0]
    bounds = ([1.0, 0.1], [1e7, 1.5])

    from functools import partial
    model = partial(mm_model_fixed_a, a_frac=a_frac)

    sigma, abs_sig = None, False
    if use_weights:
        good = np.isfinite(s_frac) & (s_frac > 0)
        if np.any(good):
            s_use = s_frac.copy()
            s_use[~good] = np.nanmin(s_use[good])
            sigma, abs_sig = s_use, True

    try:
        popt, pcov = curve_fit(model, np.log(x), y_frac,
                               p0=p0, bounds=bounds,
                               sigma=sigma, absolute_sigma=abs_sig,
                               maxfev=30000)
    except Exception:
        try:
            popt, pcov = curve_fit(model, np.log(x), y_frac,
                                   p0=p0, bounds=bounds, maxfev=30000)
        except Exception:
            return None, (np.nan, np.nan, np.nan, np.nan)

    b, c = float(popt[0]), float(popt[1])
    se_c = float(np.sqrt(pcov[1, 1])) if (pcov is not None and
                                           pcov.shape == (2, 2) and
                                           pcov[1, 1] >= 0) else np.nan

    return (fixed_a, b, c), (fixed_a, np.nan, c, se_c)


def compute_r2_fixed_a(x, y, n_total, fixed_a, b, c):
    """R² for a fixed-a Hill-MM fit."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    a_frac = fixed_a / n_total
    yhat   = mm_model_fixed_a(np.log(np.maximum(x, 1e-12)), b, c, a_frac) * n_total
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 2:
        return np.nan
    ss_res = np.sum((y[m] - yhat[m]) ** 2)
    ss_tot = np.sum((y[m] - np.mean(y[m])) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan



def bootstrap_curve(coverage_block, Ns, num_bootstrap, rng, threshold):
    """
    Accumulation curve by subsampling.

    For each N in Ns, sample N rows without replacement,
    declare a microbe "present" if coverage > threshold (%) in any sample.

    Returns:
        N_vec, mean_vec, std_vec, n_mic, med_vec, q16_vec, q84_vec, boot_matrix
    """
    present = (coverage_block.values > threshold)
    n_samples, n_mic = present.shape
    Ns = np.asarray(Ns, int)
    Ns = Ns[(Ns >= 1) & (Ns <= n_samples)]
    if Ns.size == 0:
        return (Ns.astype(float), np.array([]), np.array([]), n_mic,
                np.array([]), np.array([]), np.array([]),
                np.empty((0, 0), dtype=float))

    Ns_sorted = np.sort(np.unique(Ns))
    nN = Ns_sorted.size
    boot_matrix = np.empty((num_bootstrap, nN), dtype=float)

    for r in range(num_bootstrap):
        perm = rng.permutation(n_samples)
        present_any = np.zeros(n_mic, dtype=bool)
        next_pos = 0

        for i, N in enumerate(Ns_sorted):
            while next_pos < N:
                present_any |= present[perm[next_pos]]
                next_pos += 1
            boot_matrix[r, i] = present_any.sum()

    mean_vec = boot_matrix.mean(axis=0)
    std_vec  = boot_matrix.std(axis=0, ddof=1) if num_bootstrap > 1 else np.zeros(nN)
    med_vec  = np.median(boot_matrix, axis=0)
    q16_vec, q84_vec = np.percentile(boot_matrix, [2.5, 97.5], axis=0)

    return (Ns_sorted.astype(float), mean_vec, std_vec, n_mic,
            med_vec, q16_vec, q84_vec, boot_matrix)


def fit_central_curve(block, Ns, threshold, rng, fixed_a=None):
    """
    Run bootstrap, compute central curve, fit Hill-MM.

    If fixed_a is given (e.g. the full reference catalogue size), the asymptote
    is held constant and only b and c are optimised.  This removes one degree of
    freedom and prevents the inflection point from being pulled toward the
    centre of the observed data range, which causes the tail underfit seen when
    the current N is far from saturation.

    Returns everything needed for plotting and parameter sweeps.
    """
    num_boot = config.ACCUM_NUM_INNER_BOOT
    use_median = config.ACCUM_USE_MEDIAN
    use_weights = config.ACCUM_USE_WEIGHTS

    (N_vec, mean_vec, std_vec, n_mic,
     med_vec, q16_vec, q84_vec, boot_matrix) = bootstrap_curve(
         block, Ns, num_boot, rng, threshold)

    if N_vec.size == 0:
        return (N_vec, mean_vec, std_vec, med_vec, q16_vec, q84_vec,
                np.array([]), np.array([]), n_mic,
                None, (np.nan,)*4, np.nan, boot_matrix)

    y_center = med_vec if use_median else mean_vec
    span = q84_vec - q16_vec
    scale_vec = span / 2.0
    pos = np.isfinite(scale_vec) & (scale_vec > 0)
    min_pos = np.nanmin(scale_vec[pos]) if np.any(pos) else 1.0
    scale_vec[~np.isfinite(scale_vec) | (scale_vec <= 0)] = min_pos

    if fixed_a is not None:
        params, errs = fit_mm_fixed_a(N_vec, y_center, scale_vec, n_mic,
                                      fixed_a=fixed_a, use_weights=use_weights)
        r2 = compute_r2_fixed_a(N_vec, y_center, n_mic,
                                 *params[1:]) if params is not None else np.nan
    else:
        params, errs = fit_mm_free(N_vec, y_center, scale_vec, n_mic,
                                   use_weights=use_weights)
        r2 = compute_r2(N_vec, y_center, n_mic, params)

    return (N_vec, mean_vec, std_vec, med_vec, q16_vec, q84_vec,
            y_center, scale_vec, n_mic, params, errs, r2, boot_matrix)


# ════════════════════════════════════════════════════════════════════
#  BOOTSTRAP PARAMETER DISTRIBUTION
# ════════════════════════════════════════════════════════════════════

def bootstrap_param_distribution(N_vec, boot_matrix, scale_vec, n_mic, rng,
                                 n_boot=None, tol=None):
    """
    Fit MM model to individual bootstrap curves to get uncertainty in (a, b, c).
    """
    if n_boot is None:
        n_boot = config.ACCUM_NUM_PARAM_BOOT
    if tol is None:
        tol = config.ACCUM_TOL_MICROBES

    N_vec = np.asarray(N_vec, float)
    scale_vec = np.asarray(scale_vec, float)
    use_weights = config.ACCUM_USE_WEIGHTS

    empty = {
        "a_mean": np.nan, "a_sd": np.nan,
        "a_q16": np.nan, "a_q50": np.nan, "a_q84": np.nan,
        "b_mean": np.nan, "b_sd": np.nan,
        "b_q16": np.nan, "b_q50": np.nan, "b_q84": np.nan,
        "c_mean": np.nan, "c_sd": np.nan,
        "c_q16": np.nan, "c_q50": np.nan, "c_q84": np.nan,
        "p_a_below_nmic_minus_tol": np.nan,
        "tol_microbes": float(tol),
    }

    if boot_matrix.size == 0 or N_vec.size < 5:
        return empty

    R, nN = boot_matrix.shape
    if nN != N_vec.size:
        return empty

    curves = boot_matrix
    if R > n_boot:
        idx = rng.choice(np.arange(R), size=n_boot, replace=False)
        curves = boot_matrix[idx]

    a_list, b_list, c_list = [], [], []
    for y_curve in curves:
        params, _ = fit_mm_free(N_vec, y_curve, scale_vec, n_mic,
                                use_weights=use_weights)
        if params is None:
            continue
        a, b, c = params
        if all(np.isfinite(v) for v in [a, b, c]):
            a_list.append(a)
            b_list.append(b)
            c_list.append(c)

    if not a_list:
        return empty

    def _stats(arr):
        a = np.array(arr)
        q16, q50, q84 = np.percentile(a, [16, 50, 84])
        return {
            "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else np.nan,
            "q16": float(q16), "q50": float(q50), "q84": float(q84),
        }

    sa, sb, sc = _stats(a_list), _stats(b_list), _stats(c_list)
    cutoff = float(n_mic) - float(tol)
    p_below = np.mean(np.array(a_list) <= cutoff)

    return {
        "a_mean": sa["mean"], "a_sd": sa["sd"],
        "a_q16": sa["q16"], "a_q50": sa["q50"], "a_q84": sa["q84"],
        "b_mean": sb["mean"], "b_sd": sb["sd"],
        "b_q16": sb["q16"], "b_q50": sb["q50"], "b_q84": sb["q84"],
        "c_mean": sc["mean"], "c_sd": sc["sd"],
        "c_q16": sc["q16"], "c_q50": sc["q50"], "c_q84": sc["q84"],
        "p_a_below_nmic_minus_tol": float(p_below),
        "tol_microbes": float(tol),
    }


# ════════════════════════════════════════════════════════════════════
#  PLOTTING — ACCUMULATION CURVES
# ════════════════════════════════════════════════════════════════════

def _best_corner(ax):
    """
    Return 'upper left' or 'lower right', whichever has fewer data points.
    Examines all Line2D and collection artists on the axis.
    """
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    xm = (xl + xr) / 2
    ym = (yb + yt) / 2
    n_ul = 0
    n_lr = 0
    for line in ax.get_lines():
        xd = np.asarray(line.get_xdata(), dtype=float)
        yd = np.asarray(line.get_ydata(), dtype=float)
        if len(xd) == 0:
            continue
        n_ul += int(np.sum((xd < xm) & (yd > ym)))
        n_lr += int(np.sum((xd > xm) & (yd < ym)))
    for coll in ax.collections:
        try:
            offs = coll.get_offsets()
            if len(offs) == 0:
                continue
            xd, yd = np.asarray(offs)[:, 0], np.asarray(offs)[:, 1]
            n_ul += int(np.sum((xd < xm) & (yd > ym)))
            n_lr += int(np.sum((xd > xm) & (yd < ym)))
        except Exception:
            pass
    return "upper left" if n_ul <= n_lr else "lower right"


def plot_accumulation_curves(series_data, thr, ref_name, out_dir, suffix="",
                             ylabel=None, title=None, log_x=True):
    """
    Plot accumulation curves with fitted Hill-MM lines and error bars.

    series_data: list of dicts with keys:
        name, N_vec, y_center, q16_vec, q84_vec, n_mic, params, color, ls
    log_x: if True (default), use a log x-axis (main figures);
            if False, use a linear x-axis (supplement figures).
    """
    fig, ax = plt.subplots(figsize=page_figsize())

    for sd in series_data:
        N = sd["N_vec"]
        y = sd["y_center"]
        q16, q84 = sd["q16_vec"], sd["q84_vec"]
        color, ls = sd["color"], sd["ls"]
        name = sd["name"]
        params = sd["params"]

        # Error bars
        yerr_lo = y - q16
        yerr_hi = q84 - y
        ax.errorbar(N, y, yerr=[yerr_lo, yerr_hi],
                    fmt="o", color=color, markersize=3, capsize=2,
                    alpha=0.6, linewidth=0.8, zorder=2)

        # Fitted curve
        if params is not None:
            a_abs, b, c = params
            if all(np.isfinite(v) for v in [a_abs, b, c]) and sd["n_mic"] > 0:
                if log_x:
                    x_fit = np.logspace(np.log10(max(N.min(), 3)),
                                        np.log10(N.max()), 200)
                else:
                    x_fit = np.linspace(max(N.min(), 1), N.max(), 400)
                if 0 < a_abs < sd["n_mic"]:
                    a_raw = np.log((a_abs / sd["n_mic"]) /
                                   (1 - a_abs / sd["n_mic"]))
                else:
                    a_raw = 0.0
                y_fit = mm_model_free(np.log(x_fit), a_raw, b, c) * sd["n_mic"]

                label = (f"{name} fit (a={a_abs:.0f}, "
                         f"b={b:.1f}, c={c:.2f})")
                ax.plot(x_fit, y_fit, color=color, linestyle=ls,
                        linewidth=2, label=label, zorder=3)

    if log_x:
        ax.set_xscale("log")
        x_label = "Number of samples N (log scale)"
        grid_which = "both"
    else:
        x_label = "Number of samples N (linear scale)"
        grid_which = "major"

    ax.set_xlabel(x_label, fontsize=20)
    if ylabel is None:
        ylabel = f"SGBs detected (coverage > {thr:.1f}% breadth)"
    ax.set_ylabel(ylabel, fontsize=20)
    if title is None:
        scale_note = "log x-axis" if log_x else "linear x-axis"
        title = (f"Number of SGBs found as function of the number of samples N\n"
                 f"pooled per lifestyle ({ref_name} ref) — {scale_note}")
    ax.set_title(title, fontsize=21)
    ax.grid(True, which=grid_which, linestyle="--", linewidth=0.5, alpha=0.5)

    # Place legend in whichever corner (upper-left / lower-right) has more space
    ax.legend(fontsize=12, loc=_best_corner(ax))

    plt.tight_layout()
    scale_tag = "" if log_x else "_linear"
    fname = f"accumulation_{thr:.0f}pct_{ref_name}{suffix}{scale_tag}.png"
    fpath = os.path.join(out_dir, fname)
    savefig_dual(fig, fpath, dpi=300)
    plt.close(fig)
    print(f"    Saved: {fpath}")
    return fpath


def _figure3_N_points(n_max, n_num=None):
    if n_num is None:
        n_num = config.ACCUM_N_NUM
    n_low = min(int(n_max), 400)
    vals = {
        int(round(x)) for x in np.logspace(np.log10(1), np.log10(max(n_low, 1)),
                                           n_num)
    }
    if n_max > 400:
        vals.update(range(500, int(n_max) + 1, 100))
    vals.add(int(n_max))
    return sorted(n for n in vals if 1 <= n <= n_max)


def _figure3_boot_counts(Ns):
    return {int(n): config.ACCUM_NUM_INNER_BOOT for n in Ns}


def _study_col(meta):
    for col in ["_study", "Study", "StudyFolder",
                "study_name [MBIOTEMP_study_name]", "BioProject ID"]:
        if col in meta.columns:
            return col
    return None


def bootstrap_curve_by_study(coverage_block, study_ids, Ns, num_bootstrap,
                             rng, threshold):
    present = coverage_block.values > threshold
    studies = pd.Series(study_ids, index=coverage_block.index).astype(str)
    unique_studies = np.array(sorted(studies.dropna().unique()))
    n_studies = len(unique_studies)
    n_mic = present.shape[1]
    Ns = np.asarray(Ns, int)
    Ns = np.sort(np.unique(Ns[(Ns >= 1) & (Ns <= n_studies)]))
    if Ns.size == 0:
        return (Ns.astype(float), np.array([]), np.array([]), n_mic,
                np.array([]), np.array([]), np.array([]),
                np.empty((0, 0), dtype=float))

    study_to_rows = {
        st: np.where(studies.values == st)[0]
        for st in unique_studies
    }
    boot_matrix = np.empty((num_bootstrap, Ns.size), dtype=float)
    for r in range(num_bootstrap):
        perm = rng.permutation(unique_studies)
        present_any = np.zeros(n_mic, dtype=bool)
        next_pos = 0
        for i, N in enumerate(Ns):
            while next_pos < N:
                rows = study_to_rows[perm[next_pos]]
                present_any |= present[rows].any(axis=0)
                next_pos += 1
            boot_matrix[r, i] = present_any.sum()

    mean_vec = boot_matrix.mean(axis=0)
    std_vec = boot_matrix.std(axis=0, ddof=1) if num_bootstrap > 1 else np.zeros(Ns.size)
    med_vec = np.median(boot_matrix, axis=0)
    q16_vec, q84_vec = np.percentile(boot_matrix, [16.0, 84.0], axis=0)
    return (Ns.astype(float), mean_vec, std_vec, n_mic,
            med_vec, q16_vec, q84_vec, boot_matrix)


def bootstrap_curve_variable_reps(coverage_block, Ns, boot_counts, rng,
                                  threshold, return_boot_matrix=False):
    """Sample-count accumulation with a per-N bootstrap replicate count."""
    present = coverage_block.values > threshold
    n_samples, n_mic = present.shape
    Ns = np.asarray(Ns, int)
    Ns = np.sort(np.unique(Ns[(Ns >= 1) & (Ns <= n_samples)]))
    if Ns.size == 0:
        result = (Ns.astype(float), np.array([]), np.array([]), n_mic,
                  np.array([]), np.array([]), np.array([]))
        if return_boot_matrix:
            return result + (np.empty((0, 0), dtype=float),)
        return result
    n_boot_by_N = np.array([
        max(1, int(boot_counts.get(int(N), config.ACCUM_NUM_INNER_BOOT)))
        for N in Ns
    ], dtype=int)
    max_boot = int(n_boot_by_N.max())
    vals_by_N = [np.empty(n_boot, dtype=float) for n_boot in n_boot_by_N]
    boot_matrix = np.full((max_boot, Ns.size), np.nan, dtype=float)
    fill_by_N = np.zeros(Ns.size, dtype=int)

    for r in range(max_boot):
        active = np.flatnonzero(r < n_boot_by_N)
        if active.size == 0:
            continue
        perm = rng.permutation(n_samples)
        present_any = np.zeros(n_mic, dtype=bool)
        next_pos = 0
        for i in active:
            N = int(Ns[i])
            if N > next_pos:
                present_any |= present[perm[next_pos:N]].any(axis=0)
                next_pos = N
            value = present_any.sum()
            vals_by_N[i][fill_by_N[i]] = value
            boot_matrix[r, i] = value
            fill_by_N[i] += 1

    means, stds, meds, q16s, q84s = [], [], [], [], []
    for vals in vals_by_N:
        means.append(float(vals.mean()))
        stds.append(float(vals.std(ddof=1)) if vals.size > 1 else 0.0)
        meds.append(float(np.median(vals)))
        q16s.append(float(np.percentile(vals, 2.5)))
        q84s.append(float(np.percentile(vals, 97.5)))
    result = (Ns.astype(float), np.array(means), np.array(stds), n_mic,
              np.array(meds), np.array(q16s), np.array(q84s))
    if return_boot_matrix:
        return result + (boot_matrix,)
    return result


def bootstrap_curve_by_study_variable_reps(coverage_block, study_ids, Ns,
                                           boot_counts, rng, threshold,
                                           return_boot_matrix=False):
    """Study-count accumulation with a per-N bootstrap replicate count."""
    present = coverage_block.values > threshold
    studies = pd.Series(study_ids, index=coverage_block.index).astype(str)
    unique_studies = np.array(sorted(studies.dropna().unique()))
    n_studies = len(unique_studies)
    n_mic = present.shape[1]
    Ns = np.asarray(Ns, int)
    Ns = np.sort(np.unique(Ns[(Ns >= 1) & (Ns <= n_studies)]))
    if Ns.size == 0:
        result = (Ns.astype(float), np.array([]), np.array([]), n_mic,
                  np.array([]), np.array([]), np.array([]))
        if return_boot_matrix:
            return result + (np.empty((0, 0), dtype=float),)
        return result
    study_codes = pd.Categorical(studies, categories=unique_studies).codes
    study_present = np.zeros((n_studies, n_mic), dtype=bool)
    for i in range(n_studies):
        study_present[i] = present[study_codes == i].any(axis=0)

    n_boot_by_N = np.array([
        max(1, int(boot_counts.get(int(N), config.ACCUM_NUM_INNER_BOOT)))
        for N in Ns
    ], dtype=int)
    max_boot = int(n_boot_by_N.max())
    vals_by_N = [np.empty(n_boot, dtype=float) for n_boot in n_boot_by_N]
    boot_matrix = np.full((max_boot, Ns.size), np.nan, dtype=float)
    fill_by_N = np.zeros(Ns.size, dtype=int)

    for r in range(max_boot):
        active = np.flatnonzero(r < n_boot_by_N)
        if active.size == 0:
            continue
        perm = rng.permutation(n_studies)
        present_any = np.zeros(n_mic, dtype=bool)
        next_pos = 0
        for i in active:
            N = int(Ns[i])
            if N > next_pos:
                present_any |= study_present[perm[next_pos:N]].any(axis=0)
                next_pos = N
            value = present_any.sum()
            vals_by_N[i][fill_by_N[i]] = value
            boot_matrix[r, i] = value
            fill_by_N[i] += 1

    means, stds, meds, q16s, q84s = [], [], [], [], []
    for vals in vals_by_N:
        means.append(float(vals.mean()))
        stds.append(float(vals.std(ddof=1)) if vals.size > 1 else 0.0)
        meds.append(float(np.median(vals)))
        q16s.append(float(np.percentile(vals, 2.5)))
        q84s.append(float(np.percentile(vals, 97.5)))
    result = (Ns.astype(float), np.array(means), np.array(stds), n_mic,
              np.array(meds), np.array(q16s), np.array(q84s))
    if return_boot_matrix:
        return result + (boot_matrix,)
    return result


def _median_fit_scale(boot_matrix):
    """Approximate SE of the bootstrap median at each N for fitting weights."""
    bm = np.asarray(boot_matrix, float)
    if bm.ndim != 2 or bm.size == 0:
        return None
    n_eff = np.sum(np.isfinite(bm), axis=0).astype(float)
    sd = np.nanstd(bm, axis=0, ddof=1)
    scale = 1.253 * sd / np.sqrt(np.maximum(n_eff, 1.0))
    good = np.isfinite(scale) & (scale > 0)
    if np.any(good):
        floor = max(0.25, float(np.nanmin(scale[good])))
        scale[~good] = floor
        scale = np.maximum(scale, floor)
    else:
        scale = np.full(bm.shape[1], 1.0, dtype=float)
    return scale


def _fit_figure3_curve(N_vec, y_center, q16_vec, q84_vec, n_mic,
                       fit_scale=None):
    if fit_scale is None:
        scale_vec = (np.asarray(q84_vec, float) - np.asarray(q16_vec, float)) / 2.0
    else:
        scale_vec = np.asarray(fit_scale, float).copy()
    good = np.isfinite(scale_vec) & (scale_vec > 0)
    min_good = np.nanmin(scale_vec[good]) if np.any(good) else 1.0
    scale_vec[~good] = min_good
    hill_params, _ = fit_mm_free(
        N_vec, y_center, scale_vec, n_mic, use_weights=True)
    hill_r2 = compute_r2(N_vec, y_center, n_mic, hill_params)
    richards_params, richards_r2 = fit_richards_weighted(
        N_vec, y_center, scale_vec, n_mic)
    return scale_vec, hill_params, hill_r2, richards_params, richards_r2


def _fit_figure3_asymptote_bootstrap(N_vec, boot_matrix, scale_vec, n_mic,
                                     model="hill"):
    """Refit bootstrap curves with full tail coverage to estimate asymptote CI."""
    if boot_matrix is None:
        return (np.nan, np.nan, np.nan)
    N_vec = np.asarray(N_vec, float)
    scale_vec = np.asarray(scale_vec, float)
    asymptotes = []
    for row in np.asarray(boot_matrix, float):
        m = np.isfinite(row) & np.isfinite(N_vec)
        if m.sum() < 5 or not np.isfinite(row[-1]):
            continue
        if model == "hill":
            params, _ = fit_mm_free(
                N_vec[m], row[m], scale_vec[m], n_mic, use_weights=True)
            if params is not None:
                asymptotes.append(float(params[0]))
        else:
            params, _ = fit_richards_weighted(
                N_vec[m], row[m], scale_vec[m], n_mic)
            if params is not None:
                asymptotes.append(float(params[1]))
    asymptotes = np.asarray(asymptotes, dtype=float)
    asymptotes = asymptotes[np.isfinite(asymptotes)]
    if asymptotes.size < 3:
        return (np.nan, np.nan, np.nan)
    return tuple(np.percentile(asymptotes, [2.5, 50.0, 97.5]))


def _figure3_model_y(model, x_fit, sd):
    if model == "hill":
        params = sd.get("hill_params")
        if params is None:
            return None
        a_abs, b, c = params
        n_mic = sd["n_mic"]
        if 0 < a_abs < n_mic:
            a_raw = np.log((a_abs / n_mic) / (1 - a_abs / n_mic))
        else:
            a_raw = 0.0
        return mm_model_free(np.log(x_fit), a_raw, b, c) * n_mic
    params = sd.get("richards_params")
    if params is None:
        return None
    return richards_logN_abs(np.log(x_fit), *params)


def _figure3_fit_grid(N):
    N = np.asarray(N, float)
    xmin = max(1.0, float(np.nanmin(N)))
    xmax = max(xmin, float(np.nanmax(N)))
    log_grid = np.logspace(np.log10(xmin), np.log10(xmax), 900)
    lin_grid = np.linspace(xmin, xmax, 700)
    return np.unique(np.r_[log_grid, lin_grid])


def _figure3_asymptote(model, sd):
    if model == "hill":
        params = sd.get("hill_params")
        return np.nan if params is None else float(params[0])
    params = sd.get("richards_params")
    return np.nan if params is None else float(params[1])


def _fit_ratio_richards(N_vec, y_center, q16_vec, q84_vec):
    scale_vec = (np.asarray(q84_vec, float) - np.asarray(q16_vec, float)) / 2.0
    good = np.isfinite(scale_vec) & (scale_vec > 0)
    min_good = np.nanmin(scale_vec[good]) if np.any(good) else 0.02
    scale_vec[~good] = min_good
    scale_vec = np.maximum(scale_vec, max(min_good, 0.02))
    upper = max(float(np.nanmax(q84_vec)) * 1.25,
                float(np.nanmax(y_center)) + 0.25, 1.1)
    params, r2 = fit_richards_weighted(N_vec, y_center, scale_vec, upper)
    return scale_vec, params, r2


def _ratio_series(series_data, name, color):
    by_name = {sd["name"]: sd for sd in series_data}
    if "Industrial" not in by_name or "Non-Industrial" not in by_name:
        return []
    ind = by_name["Industrial"]
    ni = by_name["Non-Industrial"]
    common_N = np.intersect1d(ind["N_vec"].astype(float),
                              ni["N_vec"].astype(float))
    if common_N.size < 5:
        return []

    def vals(sd, key):
        s = pd.Series(np.asarray(sd[key], float),
                      index=np.asarray(sd["N_vec"], float))
        return s.loc[common_N].values.astype(float)

    ind_y = vals(ind, "y_center")
    ni_y = vals(ni, "y_center")
    ind_lo = vals(ind, "q16_vec")
    ind_hi = vals(ind, "q84_vec")
    ni_lo = vals(ni, "q16_vec")
    ni_hi = vals(ni, "q84_vec")
    ratio = ni_y / np.maximum(ind_y, 1e-9)
    ratio_lo = ni_lo / np.maximum(ind_hi, 1e-9)
    ratio_hi = ni_hi / np.maximum(ind_lo, 1e-9)
    _, params, r2 = _fit_ratio_richards(common_N, ratio, ratio_lo, ratio_hi)
    return [dict(
        name=name, N_vec=common_N, y_center=ratio,
        q16_vec=ratio_lo, q84_vec=ratio_hi,
        n_mic=max(1.1, float(np.nanmax(ratio_hi))),
        richards_params=params, richards_r2=r2,
        richards_asymptote_ci=(np.nan, np.nan, np.nan),
        color=color, ls="-")]


def _annotate_ratio_targets(ax, sd, targets=(1, 10, 100, 1000, 3000)):
    N = np.asarray(sd["N_vec"], float)
    y = np.maximum(np.asarray(sd["y_center"], float), 1e-6)
    if N.size == 0:
        return
    used = set()
    for target in targets:
        idx = int(np.nanargmin(np.abs(np.log10(np.maximum(N, 1.0)) -
                                      np.log10(float(target)))))
        if idx in used:
            continue
        used.add(idx)
        ax.annotate(f"{y[idx]:.2f}", (N[idx], y[idx]),
                    xytext=(4, 5), textcoords="offset points",
                    fontsize=9, color=sd["color"])


def _draw_ratio_panel(ax, ratio_data, x_label=None, title=None,
                      show_legend=False):
    for sd in ratio_data:
        N = np.asarray(sd["N_vec"], float)
        y = np.maximum(np.asarray(sd["y_center"], float), 1e-6)
        q16 = np.maximum(np.asarray(sd["q16_vec"], float), 1e-6)
        q84 = np.maximum(np.asarray(sd["q84_vec"], float), 1e-6)
        color = sd["color"]
        ax.fill_between(N, q16, q84, color=color, alpha=0.15, linewidth=0)
        ax.plot(N, y, "o", color=color, ms=3.5, alpha=0.85,
                label=sd["name"])
        x_fit = _figure3_fit_grid(N)
        y_fit = _figure3_model_y("richards", x_fit, sd)
        if y_fit is not None:
            ax.plot(x_fit, np.maximum(y_fit, 1e-6), color=color, lw=2.4)
        _annotate_ratio_targets(ax, sd)

    ax.axhline(1.0, color="black", lw=1.4, ls=":", alpha=0.85,
               label="equal ratio")
    ax.text(0.02, 1.02, "equal ratio", transform=ax.get_yaxis_transform(),
            fontsize=9, va="bottom", ha="left", color="black")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if x_label:
        ax.set_xlabel(f"{x_label} (log scale)")
    ax.set_ylabel("NI / Industrial SGB ratio")
    if title:
        ax.set_title(title, fontsize=13)
    ax.grid(False)
    if show_legend:
        ax.legend(fontsize=10, loc="best", frameon=True)


def plot_figure3_ratio(ratio_data, threshold_label, ref_name, out_dir,
                       x_label, file_prefix):
    if not ratio_data:
        return None
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=square_figsize(margins=dict(top=0.90)))
    fig._margin_overrides = dict(top=0.90)
    _draw_ratio_panel(
        ax, ratio_data, x_label=x_label,
        title=f"{ref_name}: Non-Industrial / Industrial ratio at {threshold_label}",
        show_legend=True)
    fig.tight_layout()
    fpath = os.path.join(out_dir, f"{file_prefix}_richards.png")
    savefig_dual(fig, fpath, dpi=300)
    plt.close(fig)
    print(f"    Saved: {fpath}")
    return fpath


def plot_figure3_ratio_grid(panels, ref_name, out_dir, x_label, file_prefix,
                            title):
    panels = [(label, data) for label, data in panels if data]
    if not panels:
        return None
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=page_figsize(n_cols=3, n_rows=2),
                             sharex=False, sharey=False)
    for ax, (label, data) in zip(axes.flat, panels):
        _draw_ratio_panel(ax, data, x_label=x_label, title=label,
                          show_legend=False)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True,
               fontsize=11)
    fig.suptitle(f"{ref_name}: {title}", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fpath = os.path.join(out_dir, f"{file_prefix}_richards.png")
    savefig_dual(fig, fpath, dpi=300)
    plt.close(fig)
    print(f"    Saved: {fpath}")
    return fpath


def plot_figure3_accumulation(series_data, threshold_label, ref_name, out_dir,
                              x_label, file_prefix, model="hill",
                              y_label="SGBs detected", title_prefix="accumulation",
                              asym_decimals=0):
    if not series_data:
        return None
    os.makedirs(out_dir, exist_ok=True)
    # Square. This panel carries an inset, a legend and per-series labels; at
    # 1.42 aspect they were all competing across a short wide box.
    _MG = dict(top=0.90)
    fig, ax = plt.subplots(figsize=square_figsize(margins=_MG))
    fig._margin_overrides = _MG
    inset = ax.inset_axes([0.40, 0.12, 0.57, 0.47])
    all_N = np.concatenate([np.asarray(sd["N_vec"], float) for sd in series_data])
    x_min = float(np.nanmin(all_N))
    x_max = float(np.nanmax(all_N))
    text_x = {
        "Industrial": x_min + 0.03 * (x_max - x_min),
        "Non-Industrial": x_min + 0.44 * (x_max - x_min),
    }

    for sd in series_data:
        N = sd["N_vec"]
        y = sd["y_center"]
        q16 = sd["q16_vec"]
        q84 = sd["q84_vec"]
        color = sd["color"]
        name = sd["name"]
        asym = _figure3_asymptote(model, sd)
        asym_q16, asym_q50, asym_q84 = sd.get(
            f"{model}_asymptote_ci", (np.nan, np.nan, np.nan))
        asym_plot = asym_q50 if np.isfinite(asym_q50) else asym
        ax.fill_between(N, q16, q84, color=color, alpha=0.15, linewidth=0)
        ax.plot(N, y, "o", color=color, ms=3.5, alpha=0.85)
        inset.fill_between(N, q16, q84, color=color, alpha=0.12, linewidth=0)
        inset.plot(N, y, "o", color=color, ms=3.5, alpha=0.85, label=name)
        if np.isfinite(asym_q16) and np.isfinite(asym_q84):
            ax.axhspan(asym_q16, asym_q84, color=color, alpha=0.08, zorder=0)
            inset.axhspan(asym_q16, asym_q84, color=color, alpha=0.08, zorder=0)
        if np.isfinite(asym_plot):
            asym_label = f"{name}: {asym_plot:.{asym_decimals}f}"
            if np.isfinite(asym_q16) and np.isfinite(asym_q84):
                asym_label = (
                    f"{name}: {asym_plot:.{asym_decimals}f} "
                    f"[{asym_q16:.{asym_decimals}f}-"
                    f"{asym_q84:.{asym_decimals}f}]"
                )
            ax.axhline(asym_plot, color=color, lw=1.4, ls=":", alpha=0.8)
            inset.axhline(asym_plot, color=color, lw=1.4, ls=":", alpha=0.8)
            ax.text(text_x.get(name, x_min + 0.15 * (x_max - x_min)),
                    asym_plot, asym_label, color=color, fontsize=11,
                    ha="left", va="bottom")

        x_fit = _figure3_fit_grid(N)
        y_fit = _figure3_model_y(model, x_fit, sd)
        if y_fit is not None:
            ax.plot(x_fit, y_fit, color=color, lw=2.4)
            inset.plot(x_fit, y_fit, color=color, lw=2.0)

    ax.set_xlabel(f"{x_label} (linear scale)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{ref_name}: {title_prefix} at {threshold_label}")
    ax.grid(False)
    inset.set_xscale("log")
    inset.set_xlabel(f"{x_label} (log scale)", fontsize=14, labelpad=2)
    inset.set_ylabel(y_label, fontsize=14, labelpad=2)
    inset.grid(False)
    inset.tick_params(labelsize=13)
    inset.legend(fontsize=14, loc="lower right", frameon=True)
    fig.tight_layout()
    fpath = os.path.join(out_dir, f"{file_prefix}_{model}.png")
    savefig_dual(fig, fpath, dpi=300)
    plt.close(fig)
    print(f"    Saved: {fpath}")
    return fpath


def _make_figure3_series(matrix, meta, masks, style, study_col, threshold, rng):
    rows = []
    sample_series = []
    study_series = []
    for name, mask in masks.items():
        ids = mask[mask].index.intersection(matrix.index)
        block = matrix.loc[ids]
        if block.shape[0] < config.ACCUM_MIN_SAMPLES:
            continue
        color, ls = style[name]
        Ns = _figure3_N_points(block.shape[0])
        boot_counts = _figure3_boot_counts(Ns)
        (N_vec, mean_vec, std_vec, n_mic, med_vec, q16_vec, q84_vec,
         boot_matrix) = bootstrap_curve_variable_reps(
            block, Ns, boot_counts, rng, threshold, return_boot_matrix=True)
        if N_vec.size >= 5:
            y_center = med_vec
            fit_scale = _median_fit_scale(boot_matrix)
            (scale_vec, hill_params, hill_r2,
             richards_params, richards_r2) = _fit_figure3_curve(
                N_vec, y_center, q16_vec, q84_vec, n_mic,
                fit_scale=fit_scale)
            hill_asymptote_ci = _fit_figure3_asymptote_bootstrap(
                N_vec, boot_matrix, scale_vec, n_mic, model="hill")
            richards_asymptote_ci = _fit_figure3_asymptote_bootstrap(
                N_vec, boot_matrix, scale_vec, n_mic, model="richards")
            sample_series.append(dict(
                name=name, N_vec=N_vec, y_center=y_center,
                q16_vec=q16_vec, q84_vec=q84_vec, n_mic=n_mic,
                hill_params=hill_params, hill_r2=hill_r2,
                richards_params=richards_params, richards_r2=richards_r2,
                hill_asymptote_ci=hill_asymptote_ci,
                richards_asymptote_ci=richards_asymptote_ci,
                color=color, ls=ls))
            for n, yv, lo, hi in zip(N_vec, y_center, q16_vec, q84_vec):
                rows.append(dict(unit="samples", series=name, N=n, sgb=yv,
                                 q16=lo, q84=hi))

        if study_col is not None:
            studies = meta.loc[ids, study_col]
            n_studies = studies.astype(str).nunique()
            if n_studies >= 5:
                Ns_st = _figure3_N_points(n_studies)
                boot_counts = _figure3_boot_counts(Ns_st)
                (N_vec, mean_vec, std_vec, n_mic, med_vec, q16_vec,
                 q84_vec, boot_matrix) = (
                    bootstrap_curve_by_study_variable_reps(
                        block, studies, Ns_st, boot_counts, rng, threshold,
                        return_boot_matrix=True))
                if N_vec.size >= 5:
                    y_center = med_vec
                    fit_scale = _median_fit_scale(boot_matrix)
                    (scale_vec, hill_params, hill_r2,
                     richards_params, richards_r2) = _fit_figure3_curve(
                        N_vec, y_center, q16_vec, q84_vec, n_mic,
                        fit_scale=fit_scale)
                    hill_asymptote_ci = _fit_figure3_asymptote_bootstrap(
                        N_vec, boot_matrix, scale_vec, n_mic, model="hill")
                    richards_asymptote_ci = _fit_figure3_asymptote_bootstrap(
                        N_vec, boot_matrix, scale_vec, n_mic,
                        model="richards")
                    study_series.append(dict(
                        name=name, N_vec=N_vec, y_center=y_center,
                        q16_vec=q16_vec, q84_vec=q84_vec, n_mic=n_mic,
                        hill_params=hill_params, hill_r2=hill_r2,
                        richards_params=richards_params,
                        richards_r2=richards_r2,
                        hill_asymptote_ci=hill_asymptote_ci,
                        richards_asymptote_ci=richards_asymptote_ci,
                        color=color, ls=ls))
                    for n, yv, lo, hi in zip(N_vec, y_center, q16_vec, q84_vec):
                        rows.append(dict(unit="studies", series=name, N=n,
                                         sgb=yv, q16=lo, q84=hi))
    return rows, sample_series, study_series


def run_figure3_accumulation(data):
    """Figure3 accumulation plots: Industrial(all) vs Non-Industrial(all)."""
    ref = data.ref_name
    out = os.path.join(config.outdir(ref), "Figure3")
    os.makedirs(out, exist_ok=True)
    meta = data.meta.copy()
    breadth = data.breadth
    rng = np.random.default_rng(123)
    style = {
        "Industrial": (config.OI["sky"], "-"),
        "Non-Industrial": (config.OI["orange"], "-"),
    }
    masks = {
        "Industrial": meta["Lifestyle"] == "Industrial",
        "Non-Industrial": meta["Lifestyle"].isin(["H-G", "Transitional"]),
    }
    study_col = _study_col(meta)
    pct_ratio_sample_panels = []
    pct_ratio_study_panels = []
    flat_ratio_sample_panels = []
    flat_ratio_study_panels = []

    for thr in config.ACCUM_BREADTH_THRESHOLDS:
        threshold_label = f">={thr:.0f}% breadth coverage"
        rows = []
        sample_series = []
        study_series = []
        for name, mask in masks.items():
            ids = mask[mask].index.intersection(breadth.index)
            block = breadth.loc[ids]
            if block.shape[0] < config.ACCUM_MIN_SAMPLES:
                continue
            color, ls = style[name]
            Ns = _figure3_N_points(block.shape[0])
            boot_counts = _figure3_boot_counts(Ns)
            (N_vec, mean_vec, std_vec, n_mic, med_vec, q16_vec, q84_vec,
             boot_matrix) = bootstrap_curve_variable_reps(
                block, Ns, boot_counts, rng, thr, return_boot_matrix=True)
            if N_vec.size >= 5:
                y_center = med_vec
                fit_scale = _median_fit_scale(boot_matrix)
                (scale_vec, hill_params, hill_r2,
                 richards_params, richards_r2) = _fit_figure3_curve(
                    N_vec, y_center, q16_vec, q84_vec, n_mic,
                    fit_scale=fit_scale)
                hill_asymptote_ci = _fit_figure3_asymptote_bootstrap(
                    N_vec, boot_matrix, scale_vec, n_mic, model="hill")
                richards_asymptote_ci = _fit_figure3_asymptote_bootstrap(
                    N_vec, boot_matrix, scale_vec, n_mic, model="richards")
                sample_series.append(dict(
                    name=name, N_vec=N_vec, y_center=y_center,
                    q16_vec=q16_vec, q84_vec=q84_vec, n_mic=n_mic,
                    hill_params=hill_params, hill_r2=hill_r2,
                    richards_params=richards_params, richards_r2=richards_r2,
                    hill_asymptote_ci=hill_asymptote_ci,
                    richards_asymptote_ci=richards_asymptote_ci,
                    color=color, ls=ls))
                for n, yv, lo, hi in zip(N_vec, y_center, q16_vec, q84_vec):
                    rows.append(dict(unit="samples", series=name,
                                     threshold=thr, N=n, sgb=yv,
                                     q16=lo, q84=hi))

            if study_col is not None:
                studies = meta.loc[ids, study_col]
                n_studies = studies.astype(str).nunique()
                if n_studies >= 5:
                    Ns_st = _figure3_N_points(n_studies)
                    boot_counts = _figure3_boot_counts(Ns_st)
                    (N_vec, mean_vec, std_vec, n_mic, med_vec, q16_vec,
                     q84_vec, boot_matrix) = (
                        bootstrap_curve_by_study_variable_reps(
                            block, studies, Ns_st, boot_counts, rng, thr,
                            return_boot_matrix=True))
                    if N_vec.size >= 5:
                        y_center = med_vec
                        fit_scale = _median_fit_scale(boot_matrix)
                        (scale_vec, hill_params, hill_r2,
                         richards_params, richards_r2) = _fit_figure3_curve(
                            N_vec, y_center, q16_vec, q84_vec, n_mic,
                            fit_scale=fit_scale)
                        hill_asymptote_ci = _fit_figure3_asymptote_bootstrap(
                            N_vec, boot_matrix, scale_vec, n_mic, model="hill")
                        richards_asymptote_ci = _fit_figure3_asymptote_bootstrap(
                            N_vec, boot_matrix, scale_vec, n_mic,
                            model="richards")
                        study_series.append(dict(
                            name=name, N_vec=N_vec, y_center=y_center,
                            q16_vec=q16_vec, q84_vec=q84_vec, n_mic=n_mic,
                            hill_params=hill_params, hill_r2=hill_r2,
                            richards_params=richards_params,
                            richards_r2=richards_r2,
                            hill_asymptote_ci=hill_asymptote_ci,
                            richards_asymptote_ci=richards_asymptote_ci,
                            color=color, ls=ls))
                        for n, yv, lo, hi in zip(N_vec, y_center, q16_vec, q84_vec):
                            rows.append(dict(unit="studies", series=name,
                                             threshold=thr, N=n, sgb=yv,
                                             q16=lo, q84=hi))

        tag = f"{thr:.0f}pct"
        for model in ("hill", "richards"):
            plot_figure3_accumulation(
                sample_series, threshold_label, ref, out,
                "Number of samples", f"figure3_samples_{tag}", model=model)
            plot_figure3_accumulation(
                study_series, threshold_label, ref, out,
                "Number of studies", f"figure3_studies_{tag}", model=model)
        ratio_color = config.OI.get("green", "#009E73")
        sample_ratio = _ratio_series(
            sample_series, "Non-Industrial / Industrial", ratio_color)
        study_ratio = _ratio_series(
            study_series, "Non-Industrial / Industrial", ratio_color)
        pct_ratio_sample_panels.append((threshold_label, sample_ratio))
        pct_ratio_study_panels.append((threshold_label, study_ratio))
        plot_figure3_ratio(
            sample_ratio, threshold_label, ref, out,
            "Number of samples", f"figure3_ratio_samples_{tag}")
        plot_figure3_ratio(
            study_ratio, threshold_label, ref, out,
            "Number of studies", f"figure3_ratio_studies_{tag}")
        for unit, ratio_data in (("samples_ratio", sample_ratio),
                                 ("studies_ratio", study_ratio)):
            for sd in ratio_data:
                for n, yv, lo, hi in zip(sd["N_vec"], sd["y_center"],
                                         sd["q16_vec"], sd["q84_vec"]):
                    rows.append(dict(unit=unit, series=sd["name"],
                                     threshold=thr, N=n, ratio=yv,
                                     q16=lo, q84=hi))
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(out, f"figure3_accumulation_{tag}.csv"),
                index=False)

    for spec in flat_bp_specs(config.ACCUM_FLAT_THRESHOLDS_BP):
        threshold_label = spec.description
        tag = f"flat_{spec.short_label}"
        presence = build_presence(
            data.breadth, data.depth, data.lengths, spec=spec)
        pseudo_breadth = presence.astype(float) * 100.0
        rows, sample_series, study_series = _make_figure3_series(
            pseudo_breadth, meta, masks, style, study_col, 50.0, rng)
        for row in rows:
            row["threshold"] = spec.short_label
            row["threshold_description"] = threshold_label

        for model in ("hill", "richards"):
            plot_figure3_accumulation(
                sample_series, threshold_label, ref, out,
                "Number of samples", f"figure3_samples_{tag}", model=model)
            plot_figure3_accumulation(
                study_series, threshold_label, ref, out,
                "Number of studies", f"figure3_studies_{tag}", model=model)
        ratio_color = config.OI.get("green", "#009E73")
        sample_ratio = _ratio_series(
            sample_series, "Non-Industrial / Industrial", ratio_color)
        study_ratio = _ratio_series(
            study_series, "Non-Industrial / Industrial", ratio_color)
        flat_ratio_sample_panels.append((threshold_label, sample_ratio))
        flat_ratio_study_panels.append((threshold_label, study_ratio))
        plot_figure3_ratio(
            sample_ratio, threshold_label, ref, out,
            "Number of samples", f"figure3_ratio_samples_{tag}")
        plot_figure3_ratio(
            study_ratio, threshold_label, ref, out,
            "Number of studies", f"figure3_ratio_studies_{tag}")
        for unit, ratio_data in (("samples_ratio", sample_ratio),
                                 ("studies_ratio", study_ratio)):
            for sd in ratio_data:
                for n, yv, lo, hi in zip(sd["N_vec"], sd["y_center"],
                                         sd["q16_vec"], sd["q84_vec"]):
                    rows.append(dict(unit=unit, series=sd["name"],
                                     threshold=spec.short_label,
                                     threshold_description=threshold_label,
                                     N=n, ratio=yv, q16=lo, q84=hi))
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(out, f"figure3_accumulation_{tag}.csv"),
                index=False)

    plot_figure3_ratio_grid(
        pct_ratio_sample_panels, ref, out, "Number of samples",
        "figure3_ratio_samples_pct_all",
        "Non-Industrial / Industrial ratio across breadth thresholds")
    plot_figure3_ratio_grid(
        pct_ratio_study_panels, ref, out, "Number of studies",
        "figure3_ratio_studies_pct_all",
        "Non-Industrial / Industrial ratio across breadth thresholds")
    plot_figure3_ratio_grid(
        flat_ratio_sample_panels, ref, out, "Number of samples",
        "figure3_ratio_samples_flat_all",
        "Non-Industrial / Industrial ratio across mapped-base thresholds")
    plot_figure3_ratio_grid(
        flat_ratio_study_panels, ref, out, "Number of studies",
        "figure3_ratio_studies_flat_all",
        "Non-Industrial / Industrial ratio across mapped-base thresholds")


# ════════════════════════════════════════════════════════════════════
#  PLOTTING — PARAMETER vs N_max
# ════════════════════════════════════════════════════════════════════

def plot_param_vs_N(df, thr, title_main, style_map, outfile_png):
    """
    Three-panel plot of (a, b, c) vs N_max with bootstrap CIs.
    Legend placed just above the topmost subplot, flush with its top edge,
    with no extra white space — works for both medium and large saves.
    """
    if df.empty:
        return

    # Do NOT use constrained_layout — it fights with manual legend placement.
    fig, axes = plt.subplots(3, 1, figsize=page_figsize(n_rows=3), sharex=True)

    param_specs = [
        ("a (asymptote, microbes)", "a_q16_boot", "a_q50_boot", "a_q84_boot"),
        ("b (N_half)", "b_q16_boot", "b_q50_boot", "b_q84_boot"),
        ("c (Hill exponent)", "c_q16_boot", "c_q50_boot", "c_q84_boot"),
    ]

    series_list = sorted(df["Series"].unique())

    for ax, (ylabel, q16c, q50c, q84c) in zip(axes, param_specs):
        for series in series_list:
            sub = df[df["Series"] == series].sort_values("N_max")
            if sub.empty:
                continue
            N_vals = sub["N_max"].values.astype(float)
            y_med  = sub[q50c].values.astype(float)
            y_lo   = sub[q16c].values.astype(float)
            y_hi   = sub[q84c].values.astype(float)

            color, ls = style_map.get(series, ("black", "-"))
            ax.plot(N_vals, y_med, label=series, color=color,
                    linestyle=ls, linewidth=2)
            ax.fill_between(N_vals, y_lo, y_hi, color=color, alpha=0.2)

        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    axes[-1].set_xscale("log")
    axes[-1].set_xlabel("Total number of samples included (N_total)")
    fig.suptitle(f"{title_main}\ncoverage > {thr}% breadth", fontsize=21)

    # Legend just above the top subplot — anchored to the top of ax[0]
    # in axes-fraction coordinates, centred horizontally.
    # We collect handles from all axes (all series drawn across all panels).
    all_handles, all_labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                all_handles.append(h)
                all_labels.append(l)
                seen.add(l)

    n_series = len(all_handles)
    if all_handles:
        # Place legend in axes[0] coordinate space, just above it.
        # loc="lower left" + bbox_to_anchor in axes coords means the legend's
        # bottom-left corner sits at (0, 1.01) relative to axes[0] — i.e.
        # just above the top edge with 1% padding.
        axes[0].legend(
            all_handles, all_labels,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),   # just above axes[0] top edge
            borderaxespad=0,
            ncol=max(1, min(n_series, 4)), # up to 4 per row
            fontsize=14,
            framealpha=0.9,
        )

    # Reserve headroom for the legend + suptitle.
    # hspace=0.04 keeps panels tight; top gives room for legend + suptitle.
    fig.subplots_adjust(hspace=0.04, top=0.78, bottom=0.08,
                        left=0.10, right=0.97)

    savefig_dual(fig, outfile_png, dpi=300)
    plt.close(fig)
    print(f"    Saved: {outfile_png}")


# ════════════════════════════════════════════════════════════════════
#  SWEEP ENGINE — Parameter-vs-N_max for a set of series
# ════════════════════════════════════════════════════════════════════

def _run_sweep(series_inputs, thr, rng, out_dir, ref_name, label_prefix):
    """
    For each series, run bootstrap curve + N_max sweep.

    Returns:
        curve_data: list of dicts for plotting accumulation curves
        sweep_df:   DataFrame of parameter-vs-N_max results
    """
    N_points = sorted({
        int(round(x)) for x in np.logspace(
            np.log10(config.ACCUM_N_MIN),
            np.log10(config.ACCUM_N_MAX),
            config.ACCUM_N_NUM)
    })
    N_sweep = sorted({
        int(round(x)) for x in np.logspace(
            np.log10(config.ACCUM_SWEEP_N_MIN),
            np.log10(config.ACCUM_SWEEP_N_MAX),
            config.ACCUM_SWEEP_N_NUM)
    })

    sweep_rows = []
    curve_data = []

    for name, block, color, ls in series_inputs:
        n_samples = block.shape[0]
        if n_samples < config.ACCUM_MIN_SAMPLES:
            print(f"    ⚠️  Skipping {name}: only {n_samples} samples "
                  f"(< {config.ACCUM_MIN_SAMPLES})")
            continue

        Ns_curve = sorted({n for n in N_points if 1 <= n <= n_samples})
        if len(Ns_curve) < 5:
            continue

        (N_vec, mean_vec, std_vec, med_vec, q16_vec, q84_vec,
         y_center, scale_vec, n_mic, params, errs, r2,
         boot_matrix) = fit_central_curve(block, Ns_curve, thr, rng)

        if N_vec.size < 5 or boot_matrix.size == 0:
            continue

        curve_data.append({
            "name": name, "N_vec": N_vec, "y_center": y_center,
            "q16_vec": q16_vec, "q84_vec": q84_vec,
            "n_mic": n_mic, "params": params,
            "color": color, "ls": ls,
        })

        # N_max sweep
        Ns_sw = [n for n in N_sweep
                 if config.ACCUM_SWEEP_N_MIN <= n <= n_samples]
        if not Ns_sw:
            continue

        for N_max in Ns_sw:
            mask = N_vec <= N_max
            if mask.sum() < 5:
                continue

            N_sub = N_vec[mask]
            y_sub = y_center[mask]
            s_sub = scale_vec[mask]
            b_sub = boot_matrix[:, mask]

            params_sub, errs_sub = fit_mm_free(
                N_sub, y_sub, s_sub, n_mic,
                use_weights=config.ACCUM_USE_WEIGHTS)

            if params_sub is None:
                sweep_rows.append(_empty_sweep_row(
                    name, thr, N_max, n_samples, n_mic))
                continue

            a_sub, b_sub_val, c_sub = params_sub
            r2_sub = compute_r2(N_sub, y_sub, n_mic, params_sub)

            qd = bootstrap_param_distribution(
                N_sub, b_sub, s_sub, n_mic, rng)

            sweep_rows.append([
                name, thr, N_max, n_samples, int(n_mic),
                a_sub, errs_sub[1] if isfinite(errs_sub[1]) else np.nan,
                b_sub_val, c_sub,
                errs_sub[3] if isfinite(errs_sub[3]) else np.nan,
                r2_sub,
                qd["a_q16"], qd["a_q50"], qd["a_q84"],
                qd["b_q16"], qd["b_q50"], qd["b_q84"],
                qd["c_q16"], qd["c_q50"], qd["c_q84"],
                qd["a_mean"], qd["a_sd"],
                qd["b_mean"], qd["b_sd"],
                qd["c_mean"], qd["c_sd"],
                qd["p_a_below_nmic_minus_tol"], qd["tol_microbes"],
            ])

    sweep_df = pd.DataFrame()
    if sweep_rows:
        sweep_df = pd.DataFrame(sweep_rows, columns=_sweep_columns())

    return curve_data, sweep_df


def _empty_sweep_row(name, thr, N_max, n_samples, n_mic):
    return [name, thr, N_max, n_samples, int(n_mic)] + [np.nan] * 23


def _sweep_columns():
    return [
        "Series", "threshold", "N_max", "n_samples_total", "n_microbes",
        "a_abs_central", "se_a_abs_local",
        "b_central", "c_central", "se_c_local",
        "R2_central",
        "a_q16_boot", "a_q50_boot", "a_q84_boot",
        "b_q16_boot", "b_q50_boot", "b_q84_boot",
        "c_q16_boot", "c_q50_boot", "c_q84_boot",
        "a_mean_boot", "a_sd_boot",
        "b_mean_boot", "b_sd_boot",
        "c_mean_boot", "c_sd_boot",
        "p_a_below_nmic_minus_tol", "tol_microbes",
    ]


# ════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def run_accumulation(data, matched_idx=None):
    """
    Run accumulation curve analysis for one reference database.

    Analyses per breadth threshold (1/10/30/50/90%):
      A. Pooled: Industrial vs Non-Industrial +/- Unhealthy
      B. Continental: Industrial & Non-Industrial x continent

    Parameters
    ----------
    data : ReferenceData
    matched_idx : list, optional
        Not used directly (accumulation uses full coverage matrix),
        but kept for API compatibility.
    """
    ref = data.ref_name
    out = config.outdir(ref)
    meta = data.meta
    breadth = data.breadth

    rng = np.random.default_rng(42)

    print(f"\n  [{ref}] Accumulation curves "
          f"(thresholds: {config.ACCUM_BREADTH_THRESHOLDS})")

    # Build group classifications
    meta = meta.copy()
    meta["Group"] = meta.apply(classify_group, axis=1)
    meta["MacroGroup"] = meta.apply(classify_macro_group, axis=1)

    group_map = meta["Group"]
    macro_map = meta["MacroGroup"]
    cont_map  = meta.get("Continent", pd.Series(None, index=meta.index))

    print(f"    Group sizes:")
    for g, n in group_map.value_counts(dropna=False).items():
        print(f"      {g}: {n:,}")

    # ── Loop over breadth thresholds ──
    for thr in config.ACCUM_BREADTH_THRESHOLDS:
        thr_label = str(thr).replace(".", "p")
        print(f"\n    ═══ Threshold {thr}% breadth ═══")

        # ════════════════════════════════
        # A. POOLED (Industrial vs Non-Industrial +/- Unhealthy)
        # ════════════════════════════════
        #
        # Build masks directly from Lifestyle + Unhealthy rather than from
        # group_map.isin().  loading.py casts meta["Unhealthy"] to str, turning
        # NaN → "nan".  classify_group() returns None for "nan" samples, so
        # they are silently excluded from every group including "Industrial (all)".
        # Using Lifestyle (never cast to str) avoids this drop.
        lifestyle_col  = meta["Lifestyle"]
        unhealthy_col  = meta["Unhealthy"].astype(str).str.strip()
        is_industrial  = lifestyle_col == "Industrial"
        is_ni          = lifestyle_col.isin(["H-G", "Transitional"])
        is_uh_str      = unhealthy_col == "Unhealthy"

        pooled_mask_map = {
            "Industrial (all)":         is_industrial,
            "Non-Industrial (all)":     is_ni,
            "Industrial Unhealthy":     is_industrial & is_uh_str,
            "Non-Industrial Unhealthy": is_ni         & is_uh_str,
        }

        series_inputs = []
        style_map = {}
        for name, groups, color, ls in config.ACCUM_POOLED_SERIES:
            mask  = pooled_mask_map[name]
            block = breadth.loc[mask[mask].index.intersection(breadth.index)]
            series_inputs.append((name, block, color, ls))
            style_map[name] = (color, ls)
            print(f"    [Pooled] {name}: {block.shape[0]:,} samples")

        curve_data, sweep_df = _run_sweep(
            series_inputs, thr, rng, out, ref, "pooled")

        # Accumulation curve plots (log x-axis: main; linear x-axis: supplement)
        if curve_data:
            plot_accumulation_curves(
                curve_data, thr, ref, out, suffix="_pooled", log_x=True)
            plot_accumulation_curves(
                curve_data, thr, ref, out, suffix="_pooled", log_x=False)

        # Parameter-vs-N plot
        if not sweep_df.empty:
            csv_path = os.path.join(
                out, f"accum_pooled_fits_vsN_cov{thr_label}.csv")
            sweep_df.to_csv(csv_path, index=False)

            plot_param_vs_N(
                sweep_df, thr,
                f"Pooled Industrial/Non-Industrial ({ref})",
                style_map,
                os.path.join(out,
                    f"accum_pooled_params_vsN_cov{thr_label}.png"))

        # ════════════════════════════════
        # B. CONTINENT BREAKDOWN
        # ════════════════════════════════
        series_inputs_cont = []
        style_map_cont = {}

        for macro in ["Industrial", "Non-Industrial"]:
            ids_macro = macro_map.index[macro_map == macro]
            sub_meta = meta.loc[ids_macro].copy()
            continents = sorted(
                sub_meta["Continent"].dropna().unique().tolist())

            if not continents:
                print(f"    [Continent] No continent info for {macro}")
                continue

            for cont in continents:
                ids_cont = sub_meta.index[sub_meta["Continent"] == cont]
                block = breadth.loc[breadth.index.intersection(ids_cont)]

                if block.shape[0] < config.ACCUM_MIN_SAMPLES:
                    print(f"    [Continent] Skip {macro} - {cont}: "
                          f"{block.shape[0]} samples")
                    continue

                label = f"{macro} - {cont}"
                color = config.CONTINENT_COLORS.get(cont, "gray")
                ls = "-" if macro == "Industrial" else "--"
                series_inputs_cont.append((label, block, color, ls))
                style_map_cont[label] = (color, ls)
                print(f"    [Continent] {label}: {block.shape[0]:,} samples")

        if series_inputs_cont:
            curve_data_cont, sweep_df_cont = _run_sweep(
                series_inputs_cont, thr, rng, out, ref, "continent")

            if curve_data_cont:
                plot_accumulation_curves(
                    curve_data_cont, thr, ref, out, suffix="_continent",
                    log_x=True)
                plot_accumulation_curves(
                    curve_data_cont, thr, ref, out, suffix="_continent",
                    log_x=False)

            if not sweep_df_cont.empty:
                csv_path = os.path.join(
                    out, f"accum_continent_fits_vsN_cov{thr_label}.csv")
                sweep_df_cont.to_csv(csv_path, index=False)

                plot_param_vs_N(
                    sweep_df_cont, thr,
                    f"By Continent ({ref}) — Industrial (solid) vs "
                    f"Non-Industrial (dashed)",
                    style_map_cont,
                    os.path.join(out,
                        f"accum_continent_params_vsN_cov{thr_label}.png"))

    # ── Figure3-ready two-series plots (Industrial all vs Non-Industrial all) ──
    run_figure3_accumulation(data)

    # ── Save combined fit statistics across all thresholds ──
    print(f"\n    [{ref}] Accumulation analysis complete.")


# ════════════════════════════════════════════════════════════════════
#  BACKWARD-COMPATIBLE ALIASES
# ════════════════════════════════════════════════════════════════════
# chimp.py (and potentially other modules) import the old name `fit_mm`.
fit_mm = fit_mm_free


def record_accumulation_stats(data, stats):
    """Register the accumulation fit quality and asymptotes as manuscript stats.

    The fits are already persisted to accum_pooled_fits_vsN_cov*.csv; this reads
    them back and puts the headline numbers into the stats registry so prose that
    quotes them (R2 ~0.998-1.0, asymptotes reaching the catalogue total) can be
    bound to a computation rather than transcribed. Reading the CSVs rather than
    threading a stats object through the fitting code keeps this additive.
    """
    import glob
    import os

    ref = data.ref_name
    out = config.outdir(ref)
    n_cat = int(len(data.lengths))
    r2_seen = []

    for path in sorted(glob.glob(os.path.join(out, "accum_pooled_fits_vsN_cov*.csv"))):
        tag_thr = os.path.basename(path).split("cov")[-1].replace(".csv", "")
        try:
            d = pd.read_csv(path)
        except Exception:
            continue
        if not {"Series", "N_max", "a_abs_central", "R2_central"} <= set(d.columns):
            continue
        for series, g in d.groupby("Series"):
            r = g.sort_values("N_max").iloc[-1]
            slug = str(series).replace(" ", "_").replace("(", "").replace(")", "")
            if pd.notna(r["R2_central"]):
                r2_seen.append(float(r["R2_central"]))
                stats.add(f"accum_R2_{slug}_{tag_thr}", "s1", "supplement",
                          f"Accumulation fit R2: {series} [cov {tag_thr}]",
                          f"{float(r['R2_central']):.4f}")
            if pd.notna(r["a_abs_central"]):
                a = float(r["a_abs_central"])
                stats.add(f"accum_asymptote_{slug}_{tag_thr}", "s1", "supplement",
                          f"Accumulation asymptote: {series} [cov {tag_thr}]",
                          f"{a:.1f} SGBs ({100*a/n_cat:.1f}% of the {n_cat:,}-SGB catalogue)")

    if r2_seen:
        stats.add("accum_R2_range", "s1", "main",
                  "Range of accumulation fit R2 across series and thresholds",
                  f"{min(r2_seen):.3f}-{max(r2_seen):.3f}")
