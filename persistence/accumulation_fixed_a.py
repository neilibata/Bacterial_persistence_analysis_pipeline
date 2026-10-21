"""
accumulation_fixed_a.py
=======================
Compare the free-a Hill-MM accumulation fit (3 parameters) against a
fixed-a fit (2 parameters: b and c only) for the pooled Industrial vs
Non-Industrial groups.

Motivation
----------
The free fit places the asymptote at wherever minimises the loss over
all N points.  Because the N-axis is log-spaced, low-N points dominate
and the asymptote is pulled below the true catalogue size, causing a
systematic upward bias in the tail residuals (the last data points sit
above the fitted curve).  Fixing a = N_catalogue removes this degree of
freedom and forces the inflection point to a position consistent with
the known upper bound.

Usage
-----
    python -m persistence.accumulation_fixed_a --ref UHGG
    python -m persistence.accumulation_fixed_a --ref UHGG --fixed-a 3974
    python -m persistence.accumulation_fixed_a --ref UHGG lev --threshold 10
"""

import argparse
import os
from math import isfinite

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from . import config
from .loading import load_reference
from .common import classify_group, classify_macro_group
from .accumulation import (
    bootstrap_curve,
    fit_mm_free,
    fit_mm_fixed_a,
    compute_r2,
    compute_r2_fixed_a,
    mm_model_fixed_a,
)
from .plot_utils import apply_display_text

# ── default fixed-a value ─────────────────────────────────────────────────────
DEFAULT_FIXED_A = 3974


def _hill(N, a, b, c):
    Nc = np.power(np.maximum(np.asarray(N, float), 1e-12), c)
    bc = np.power(max(float(b), 1e-12), c)
    return a * Nc / (bc + Nc)


def _hill_intercept(N, s0, b, c, a):
    """
    S(N) = s0 + (a - s0) * N^c / (b^c + N^c)

    s0  : expected richness at N=1 (free parameter, > 0)
    a   : asymptote (fixed externally, not fitted)
    b   : half-saturation N for the *incremental* component (a - s0)
    c   : Hill exponent

    At N=0 this → s0 (intercept); at N→∞ this → a.
    The incremental component (a-s0)*N^c/(b^c+N^c) is a standard Hill-MM
    on the *remaining* diversity above the core.

    This directly addresses the structural bias: the standard Hill-MM
    forces S(0)=0, but in practice a single random sample already detects
    a substantial fraction of the common taxa.  s0 absorbs that intercept,
    letting b and c describe the shape of the *additional* accumulation.
    """
    Nc  = np.power(np.maximum(np.asarray(N, float), 1e-12), c)
    bc  = np.power(max(float(b), 1e-12), c)
    return s0 + (a - s0) * Nc / (bc + Nc)


def _weibull(N, s0, b, c, a):
    """
    Weibull saturation with intercept — the correct model family for
    accumulation curves that are always concave on a log-N axis.

        S(N) = s0 + (a − s0) · (1 − exp(−(N/b)ᶜ))

    Parameters
    ----------
    s0  : richness at N→0 (core detectable in any single sample)
    b   : characteristic scale (N where 63% of the s0→a range is covered)
    c   : shape exponent.  c < 1 → sub-exponential, strictly concave on log-N
          (no inflection), c = 1 → exponential, c > 1 → slight S-curve
    a   : asymptote (fixed externally)

    Why this and not Hill-MM?
    -------------------------
    Hill-MM on log-N is a logistic sigmoid — it always has an inflection.
    The empirical inflection in these accumulation curves is at N ≈ 1% of
    N_max (N ≈ 4-115 for the three lifestyles), so the entire observable
    range (N > 100) is on the purely concave right side.  Fitting a sigmoid
    to purely concave data produces the S-shaped residuals seen in previous
    figures regardless of weighting.  The Weibull with c ≤ 1 has no
    inflection and matches the data geometry exactly.
    """
    N   = np.maximum(np.asarray(N, float), 1e-12)
    arg = np.power(N / max(float(b), 1e-12), float(c))
    return s0 + (a - s0) * (1.0 - np.exp(-arg))


def fit_weibull(x, y, s, fixed_a, min_points=5, use_weights=False):
    """
    Fit  S(N) = s0 + (fixed_a − s0) · (1 − exp(−(N/b)ᶜ))

    Free parameters: s0, b, c.
    c is bounded to (0, 1] to enforce strict concavity on log-N.

    Returns (s0, b, c) or None on failure.
    """
    from scipy.optimize import curve_fit
    from functools import partial

    x = np.asarray(x, float); y = np.asarray(y, float); s = np.asarray(s, float)
    m = np.isfinite(x) & np.isfinite(y) & (x >= 1)
    x, y, s = x[m], y[m], s[m]
    if x.size < min_points:
        return None

    # Initialise s0 from y at smallest N
    s0_init = np.clip(float(y[np.argmin(x)]), 1.0, fixed_a * 0.45)
    incr    = fixed_a - s0_init

    # b: N at which the incremental component reaches 63% (1-1/e) of its range
    target  = s0_init + incr * (1 - 1/np.e)
    idx_b   = int(np.argmin(np.abs(y - target)))
    b0      = max(1.0, min(float(x[idx_b]), 1e6))
    c0      = 0.5   # sub-exponential starting guess

    p0     = [s0_init, b0, c0]
    bounds = ([0.0,  1.0, 0.05],
              [fixed_a * 0.5, 1e8, 1.0])   # c ≤ 1 enforced

    model = partial(_weibull, a=fixed_a)

    sigma, abs_sig = None, False
    if use_weights:
        good = np.isfinite(s) & (s > 0)
        if good.any():
            su = s.copy(); su[~good] = np.nanmin(su[good])
            sigma, abs_sig = su, True

    try:
        popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds,
                            sigma=sigma, absolute_sigma=abs_sig,
                            maxfev=60000)
    except Exception:
        try:
            popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=60000)
        except Exception:
            return None

    return tuple(float(v) for v in popt)   # (s0, b, c) from Weibull


# ── Profile-likelihood fitters: global optimum via logit transform + b grid ──

def _best_c_for_b(b, N, S, w, fixed_a, s0=0.0, c_bounds=(0.10, 1.50)):
    """
    For fixed b (and fixed s0), find the exact optimal c in original weighted space
    via Brent's method.  This is a strict 1D minimisation of:
        Σ w_i · (S_i − [s0 + (a−s0)·N_i^c/(b^c+N_i^c)])²
    Returns (ssr, c_opt).
    """
    from scipy.optimize import minimize_scalar
    incr = fixed_a - s0
    def f(c):
        Nc = np.power(np.maximum(N, 1e-12), c)
        bc = np.power(max(b, 1e-12), c)
        yhat = s0 + incr * Nc / (bc + Nc)
        return float(np.sum(w * (S - yhat)**2))
    res = minimize_scalar(f, bounds=c_bounds, method='bounded',
                          options={"xatol": 1e-5})
    return res.fun, float(res.x)


def fit_hill_profile(x, y, sigma, fixed_a, n_grid=300, min_points=5,
                     use_weights=True):
    """
    Fit  S(N) = fixed_a · Nᶜ / (bᶜ + Nᶜ)  — true global optimum.

    Strategy: exhaustive grid over b (300 log-spaced points).
    At each b, find the exact optimal c via Brent 1D minimisation in
    ORIGINAL weighted least-squares space.  No logit transform — no
    objective mismatch.  Guaranteed global minimum given a fine enough b grid.

    Returns (fixed_a, b_opt, c_opt) or None.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    s = np.asarray(sigma, float)
    m = np.isfinite(x) & np.isfinite(y) & (x >= 1)
    x, y, s = x[m], y[m], s[m]
    if x.size < min_points:
        return None

    w = 1.0 / np.maximum(s, 1e-3)**2 if use_weights else np.ones(len(x))

    b_grid = np.logspace(np.log10(x.min()), np.log10(x.max() * 5), n_grid)
    best_ssr, best_b, best_c = np.inf, b_grid[0], 0.5
    for b in b_grid:
        ssr, c = _best_c_for_b(b, x, y, w, fixed_a, s0=0.0)
        if np.isfinite(ssr) and ssr < best_ssr:
            best_ssr, best_b, best_c = ssr, b, c

    return (fixed_a, float(best_b), float(best_c)) if np.isfinite(best_ssr) else None


def fit_hill_intercept_profile(x, y, sigma, fixed_a,
                               n_grid_b=200, n_grid_s0=40,
                               min_points=5, use_weights=True):
    """
    Fit  S(N) = s0 + (fixed_a − s0) · Nᶜ / (bᶜ + Nᶜ)  — true global optimum.

    2D grid over (s0, b); for each pair find exact c via Brent 1D in original
    weighted space.  s0 grid: 0 … min(y)*0.9;  b grid: x_min … x_max*5.

    Returns (s0_opt, b_opt, c_opt) or None.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    s = np.asarray(sigma, float)
    m = np.isfinite(x) & np.isfinite(y) & (x >= 1)
    x, y, s = x[m], y[m], s[m]
    if x.size < min_points:
        return None

    w = 1.0 / np.maximum(s, 1e-3)**2 if use_weights else np.ones(len(x))

    s0_grid = np.linspace(0.0, min(float(y.min()) * 0.90, fixed_a * 0.40), n_grid_s0)
    b_grid  = np.logspace(np.log10(x.min()), np.log10(x.max() * 5), n_grid_b)

    best_ssr, best_s0, best_b, best_c = np.inf, 0.0, float(x.min()), 0.5
    for s0 in s0_grid:
        if fixed_a - s0 <= 0:
            continue
        for b in b_grid:
            ssr, c = _best_c_for_b(b, x, y, w, fixed_a, s0=s0)
            if np.isfinite(ssr) and ssr < best_ssr:
                best_ssr, best_s0, best_b, best_c = ssr, s0, b, c

    return (float(best_s0), float(best_b), float(best_c)) if np.isfinite(best_ssr) else None


def _fit_both(N_vec, mean_vec, median_vec, scale_vec, n_mic, fixed_a, use_weights):
    """
    Fit three models to the mean accumulation curve:

    1. Free Hill-MM  (a, b, c free)
    2. Fixed-a Hill-MM  (a pinned to fixed_a; b, c free)
    3. Fixed-a Hill-MM + intercept  (a pinned; s0, b, c free)
       S(N) = s0 + (a - s0) * N^c / (b^c + N^c)

    The intercept model (3) addresses the structural bias of (1) and (2):
    the standard Hill-MM forces S(0)=0, but in practice even one sample
    already detects many taxa.  s0 absorbs this core richness, leaving b
    and c to describe the shape of the *additional* accumulation curve.

    All three are fit to the bootstrap MEAN (not median) since the Hill-MM
    is a model for E[S(N)].

    Models
    ------
    1. Free Hill-MM  — gradient descent (unchanged from pipeline)
    2. Fixed-a Hill-MM  — profile likelihood over b, no local minima
    3. Fixed-a Hill-MM + s0  — profile likelihood over (s0, b) grid
    """
    params_free, _ = fit_mm_free(
        N_vec, mean_vec, scale_vec, n_mic, use_weights=use_weights)
    r2_free = compute_r2(N_vec, mean_vec, n_mic, params_free)

    # Profile-likelihood fixed-a: guaranteed global optimum
    params_profile = fit_hill_profile(
        N_vec, mean_vec, scale_vec, fixed_a, use_weights=use_weights)
    r2_profile = np.nan
    if params_profile is not None:
        _, b_p, c_p = params_profile
        yhat_p = _hill(N_vec, fixed_a, b_p, c_p)
        m = np.isfinite(mean_vec) & np.isfinite(yhat_p)
        if m.sum() >= 2:
            ss_res = np.sum((mean_vec[m] - yhat_p[m])**2)
            ss_tot = np.sum((mean_vec[m] - mean_vec[m].mean())**2)
            r2_profile = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Profile-likelihood with intercept s0
    params_inter_profile = fit_hill_intercept_profile(
        N_vec, mean_vec, scale_vec, fixed_a, use_weights=use_weights)
    r2_inter_profile = np.nan
    if params_inter_profile is not None:
        s0_ip, b_ip, c_ip = params_inter_profile
        yhat_ip = _hill_intercept(N_vec, s0_ip, b_ip, c_ip, fixed_a)
        m = np.isfinite(mean_vec) & np.isfinite(yhat_ip)
        if m.sum() >= 2:
            ss_res = np.sum((mean_vec[m] - yhat_ip[m])**2)
            ss_tot = np.sum((mean_vec[m] - mean_vec[m].mean())**2)
            r2_inter_profile = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return (params_free, r2_free,
            params_profile, r2_profile,
            params_inter_profile, r2_inter_profile)


def _build_series_blocks(meta, breadth):
    """
    Build sample blocks for each pooled series directly from Lifestyle and the
    raw Unhealthy column, rather than from classify_group().

    Why not just use group_map.isin()?
    ------------------------------------
    loading.py casts meta["Unhealthy"] to str (line: meta["Unhealthy"] =
    meta["Unhealthy"].astype(str)), which converts NaN → the literal string
    "nan".  classify_group() only matches "Unhealthy" and "Healthy", so any
    sample whose Unhealthy value ended up as "nan" returns None and is silently
    dropped by isin().  This can exclude a meaningful number of Industrial
    samples from "Industrial (all)".

    The fix: build the "(all)" groups from Lifestyle (which is never cast to
    string in loading.py and is always one of "Industrial", "H-G",
    "Transitional").  Build the "Unhealthy" sub-groups from the intersection of
    Lifestyle AND Unhealthy == "Unhealthy".

    Returns a list of (name, block_df, color, ls) matching ACCUM_POOLED_SERIES order.
    """
    lifestyle = meta["Lifestyle"]
    # Unhealthy may be the string "Unhealthy", "Healthy", or "nan" (loaded as str)
    unhealthy_col = meta["Unhealthy"].astype(str).str.strip()

    is_industrial    = lifestyle == "Industrial"
    is_ni            = lifestyle.isin(["H-G", "Transitional"])
    is_unhealthy_str = unhealthy_col == "Unhealthy"

    mask_map = {
        "Industrial (all)":         is_industrial,
        "Non-Industrial (all)":     is_ni,
        "Industrial Unhealthy":     is_industrial & is_unhealthy_str,
        "Non-Industrial Unhealthy": is_ni         & is_unhealthy_str,
    }

    series = []
    for name, groups, color, ls in config.ACCUM_POOLED_SERIES:
        mask  = mask_map[name]
        block = breadth.loc[mask[mask].index.intersection(breadth.index)]
        series.append((name, block, color, ls))

    return series


def _build_N_points(n_samp,
                    n_log=50,
                    n_linear_tail=25,
                    tail_frac=0.35):
    """
    Build N evaluation points for one series.

    Strategy
    --------
    • n_log log-spaced points from N=2 up to n_samp  (preserves low-N coverage)
    • n_linear_tail linearly-spaced points in [n_samp*(1-tail_frac), n_samp]

    The linear tail points give the fit equal leverage over the high-N range
    on a per-unit-N basis, countering the natural sparsity of log spacing near
    N_max.  Without them, a single log-step at high N spans thousands of samples
    while a step at low N spans just a few — the optimizer sees equal numbers of
    points in each log-decade and anchors the inflection accordingly.
    """
    log_pts = np.unique(np.round(
        np.logspace(np.log10(2), np.log10(n_samp), n_log)).astype(int))

    tail_lo = max(2, int(n_samp * (1 - tail_frac)))
    lin_pts = np.unique(np.round(
        np.linspace(tail_lo, n_samp, n_linear_tail)).astype(int))

    all_pts = np.unique(np.concatenate([log_pts, lin_pts]))
    return sorted(int(n) for n in all_pts if 1 <= n <= n_samp)


def _weighted_sigma(N_vec, q16_vec, q84_vec,
                    n_weight_exp=0.5,
                    tail_frac=0.10,
                    tail_tighten=6.0,
                    min_weight_frac=0.25):
    """
    Per-point sigma for curve_fit.

    Three components:

    1. Base sigma = (q84 - q16) / 2  (bootstrap uncertainty)

    2. N-upweighting  sigma /= (N_i / N_ref)^n_weight_exp
       exp=0.5 (sqrt): gentle ramp — a point at N=10k with N_ref=500 gets
       sqrt(20) ≈ 4.5× more weight, not 20^1.5=89× as before.  Low-N points
       keep meaningful influence on b and c; the curve is pulled toward the
       data at both ends.

    3. Tail anchor: top tail_frac of N points get sigma /= tail_tighten.
       6× (down from 15×) still firmly pins the tail without making it the
       only constraint.

    4. Minimum weight floor: no point's effective weight falls below
       min_weight_frac × mean(1/sigma).  Prevents any region from being
       completely ignored — which is what caused the low-N miss.
    """
    N_vec = np.asarray(N_vec, float)
    base  = np.maximum((q84_vec - q16_vec) / 2.0, 1.0)

    N_ref    = np.median(N_vec)
    if N_ref <= 0:
        N_ref = 1.0
    n_weight = np.maximum((N_vec / N_ref) ** n_weight_exp, 1e-6)
    sigma    = base / n_weight

    # Tail anchor
    tail_cutoff   = N_vec.max() * (1.0 - tail_frac)
    sigma[N_vec >= tail_cutoff] /= tail_tighten

    sigma = np.maximum(sigma, 1e-3)

    # Weight floor: raise any point whose weight is below min_weight_frac
    # of the mean weight so the low-N region stays in the fit
    weights      = 1.0 / sigma
    w_floor      = min_weight_frac * weights.mean()
    sigma        = np.where(weights < w_floor, 1.0 / w_floor, sigma)

    return sigma


def run_fixed_a_comparison(ref_name: str,
                            threshold: float = 10.0,
                            fixed_a: int = DEFAULT_FIXED_A):
    """
    Load data, run both fits for the pooled series, produce comparison figure.
    """
    data = load_reference(ref_name)
    out  = config.outdir(ref_name)
    os.makedirs(out, exist_ok=True)

    rng  = np.random.default_rng(42)
    meta = data.meta.copy()

    # N_catalogue: how many SGBs are in the breadth matrix (the true upper bound)
    n_catalogue = data.breadth.shape[1]
    print(f"\n  Reference:         {ref_name}")
    print(f"  Breadth threshold: {threshold}%")
    print(f"  SGBs in matrix:    {n_catalogue:,}")
    print(f"  Fixed-a value:     {fixed_a}")
    if fixed_a > n_catalogue:
        print(f"  ⚠  fixed_a ({fixed_a}) > SGBs in matrix ({n_catalogue}) — "
              "this is intentional if you want to model the full catalogue")

    # ── Build series blocks from Lifestyle directly (avoids nan-string bug) ───
    series_list = _build_series_blocks(meta, data.breadth)

    print(f"\n  Sample counts per series:")
    for name, block, _, _ in series_list:
        print(f"    {name:<28} {block.shape[0]:>6,} samples")

    results = []
    for name, block, color, ls in series_list:
        n_samp = block.shape[0]

        if n_samp < config.ACCUM_MIN_SAMPLES:
            print(f"\n  Skipping {name}: {n_samp} samples < {config.ACCUM_MIN_SAMPLES}")
            continue

        Ns_curve = _build_N_points(n_samp)
        if len(Ns_curve) < 5:
            continue

        print(f"\n  Series: {name}  ({n_samp:,} samples)")

        (N_vec, mean_vec, std_vec, n_mic,
         med_vec, q16_vec, q84_vec, _) = bootstrap_curve(
             block, Ns_curve, config.ACCUM_NUM_INNER_BOOT, rng, threshold)

        if N_vec.size < 5:
            continue

        # Weighted sigma: upweight high-N, hard-anchor the tail
        scale_vec = _weighted_sigma(N_vec, q16_vec, q84_vec)

        # Fit to mean (E[S(N)] is what the Hill-MM models).
        # Also fit to median so the pipeline-default is printed for comparison.
        (params_free, r2_free,
         params_profile, r2_profile,
         params_inter_profile, r2_inter_profile) = _fit_both(
            N_vec, mean_vec, med_vec, scale_vec, n_mic, fixed_a,
            use_weights=True)

        if params_free is not None:
            a_f, b_f, c_f = params_free
            print(f"    Hill free (gradient)     a={a_f:.0f}  b={b_f:.1f}  c={c_f:.3f}  R²={r2_free:.4f}")
        if params_profile is not None:
            _, b_p, c_p = params_profile
            print(f"    Hill profile (fixed a)   a={fixed_a}  b={b_p:.1f}  c={c_p:.3f}  R²={r2_profile:.4f}")
        if params_inter_profile is not None:
            s0_ip, b_ip, c_ip = params_inter_profile
            print(f"    Hill+s0 profile          a={fixed_a}  s0={s0_ip:.0f}  b={b_ip:.1f}  c={c_ip:.3f}  R²={r2_inter_profile:.4f}")

        # Pipeline default (median fit) — print for reference only
        pf_med, _ = fit_mm_free(N_vec, med_vec, scale_vec, n_mic, use_weights=True)
        if pf_med is not None:
            a_m, b_m, c_m = pf_med
            print(f"    Hill free (median)       a={a_m:.0f}  b={b_m:.1f}  c={c_m:.3f}  ← pipeline default")

        w = 1.0 / scale_vec; w_rel = w / w.mean()
        tail_mask = N_vec >= N_vec.max() * 0.90
        print(f"    Weight: low-N={w_rel[:5].mean():.2f}x  tail={w_rel[tail_mask].mean():.1f}x")

        # Pipeline default (median fit) — print for reference only
        pf_med, _ = fit_mm_free(N_vec, med_vec, scale_vec, n_mic,
                                use_weights=True)
        if pf_med is not None:
            a_m, b_m, c_m = pf_med
            print(f"    Free fit (median)    a={a_m:.0f}  b={b_m:.1f}  c={c_m:.3f}  ← pipeline default")

        # Weight ratio diagnostic
        w = 1.0 / scale_vec; w_rel = w / w.mean()
        tail_mask = N_vec >= N_vec.max() * 0.90
        print(f"    Weight: low-N={w_rel[:5].mean():.2f}x  tail={w_rel[tail_mask].mean():.1f}x  "
              f"(n_tail={tail_mask.sum()})")

        results.append(dict(
            name=name, color=color, ls=ls,
            N_vec=N_vec, mean_vec=mean_vec, med_vec=med_vec,
            scale_vec=scale_vec, q16_vec=q16_vec, q84_vec=q84_vec,
            n_mic=n_mic, fixed_a=fixed_a,
            params_free=params_free,                 r2_free=r2_free,
            params_profile=params_profile,           r2_profile=r2_profile,
            params_inter_profile=params_inter_profile, r2_inter_profile=r2_inter_profile,
        ))

    if not results:
        print("  No series produced results — check group sizes / threshold.")
        return

    _plot_comparison(results, threshold, fixed_a, ref_name, out, n_catalogue)


# ── Figure ─────────────────────────────────────────────────────────────────────

def _plot_comparison(results, threshold, fixed_a, ref_name, out, n_catalogue):
    n_series = len(results)

    # 3 rows: accumulation curve | residuals | weight profile
    fig, axes = plt.subplots(
        3, n_series,
        figsize=(8 * n_series, 13),
        gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.08},
        squeeze=False,
    )

    for col, res in enumerate(results):
        ax_curve  = axes[0][col]
        ax_res    = axes[1][col]
        ax_weight = axes[2][col]

        N_vec    = res["N_vec"]
        y_mean   = res["mean_vec"]
        y_med    = res["med_vec"]
        q16, q84 = res["q16_vec"], res["q84_vec"]
        color    = res["color"]
        n_mic    = res["n_mic"]
        p_free   = res["params_free"]
        p_prof   = res["params_profile"]
        p_interp = res["params_inter_profile"]
        fa       = res["fixed_a"]

        # ── Primary: mean ± q16/q84 (q16/q84 are the right uncertainty bounds
        #    regardless of whether you display mean or median) ─────────────────
        ax_curve.errorbar(
            N_vec, y_mean,
            yerr=[y_mean - q16, q84 - y_mean],
            fmt="o", color=color, ms=4.5, capsize=2.5, lw=0.9,
            alpha=0.80, zorder=4, label="Bootstrap mean ± q16/q84")

        # ── Secondary: median — dimmed crosses for reference ─────────────────
        ax_curve.plot(N_vec, y_med, "+", color=color, ms=4, alpha=0.30,
                      zorder=2, label="Bootstrap median (pipeline default)")

        # Dense x-axis for smooth curves
        N_plot = np.logspace(np.log10(max(N_vec.min(), 2)),
                             np.log10(N_vec.max() * 2.5), 500)

        # Free Hill-MM (gradient descent — baseline)
        if p_free is not None:
            a_f, b_f, c_f = p_free
            y_free = _hill(N_plot, a_f, b_f, c_f)
            ax_curve.plot(N_plot, y_free, color=color, lw=1.6, ls="-", alpha=0.45,
                          zorder=4,
                          label=f"Hill free   a={a_f:.0f}  b={b_f:.0f}  c={c_f:.2f}"
                                f"  R²={res['r2_free']:.3f}")
        else:
            a_f = None

        # Profile-likelihood Hill, fixed a
        b_p = c_p = None
        if p_prof is not None:
            _, b_p, c_p = p_prof
            y_prof = _hill(N_plot, fa, b_p, c_p)
            ax_curve.plot(N_plot, y_prof, color=color, lw=2.2, ls="--", alpha=0.85,
                          zorder=5,
                          label=f"Hill profile  a={fa}  b={b_p:.0f}  c={c_p:.2f}"
                                f"  R²={res['r2_profile']:.3f}")

        # Profile-likelihood Hill + s0
        s0_ip = b_ip = c_ip = None
        if p_interp is not None:
            s0_ip, b_ip, c_ip = p_interp
            y_interp = _hill_intercept(N_plot, s0_ip, b_ip, c_ip, fa)
            ax_curve.plot(N_plot, y_interp, color="black", lw=2.8, ls="-", alpha=0.90,
                          zorder=6,
                          label=f"Hill+s₀ profile  a={fa}  s₀={s0_ip:.0f}  "
                                f"b={b_ip:.0f}  c={c_ip:.2f}"
                                f"  R²={res['r2_inter_profile']:.3f}")

        # Asymptote lines
        if a_f is not None:
            ax_curve.axhline(a_f,     color=color,  lw=0.7, ls=":",  alpha=0.40)
        ax_curve.axhline(fa,          color="grey",  lw=1.0, ls="--", alpha=0.40)
        ax_curve.axhline(n_catalogue, color="black", lw=0.5, ls=":",  alpha=0.20)

        x_ann = N_plot[-1] * 0.97
        if a_f is not None:
            ax_curve.text(x_ann, a_f + n_catalogue*0.005, f"free a={a_f:.0f}",
                          ha="right", va="bottom", fontsize=7, color=color, alpha=0.65)
        ax_curve.text(x_ann, fa + n_catalogue*0.005, f"fixed a={fa}",
                      ha="right", va="bottom", fontsize=7, color="dimgrey")
        ax_curve.text(x_ann, n_catalogue + n_catalogue*0.005, f"matrix={n_catalogue}",
                      ha="right", va="bottom", fontsize=7, color="grey", alpha=0.5)

        ax_curve.set_xscale("log")
        ax_curve.set_xlim(N_plot[0] * 0.8, N_plot[-1])
        ax_curve.set_ylim(0, max(fa, n_catalogue) * 1.08)
        ax_curve.set_title(res["name"], fontsize=13, fontweight="bold", color=color)
        ax_curve.set_ylabel(f"SGBs detected  (breadth ≥ {threshold:.0f}%)", fontsize=10)
        ax_curve.tick_params(labelbottom=False)
        ax_curve.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
        ax_curve.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax_curve.grid(True, which="both", ls="--", lw=0.35, alpha=0.45)

        # ── Residuals ─────────────────────────────────────────────────────────
        ax_res.axhline(0, color="black", lw=0.9)

        if p_free is not None:
            ax_res.plot(N_vec, y_mean - _hill(N_vec, *p_free),
                        "o-", color=color, ms=2.5, lw=0.9, alpha=0.45,
                        label="Hill free")
        if p_prof is not None:
            ax_res.plot(N_vec, y_mean - _hill(N_vec, fa, b_p, c_p),
                        "s--", color=color, ms=2.5, lw=1.2, alpha=0.75,
                        label=f"Hill profile (a={fa})")
        if p_interp is not None:
            ax_res.plot(N_vec, y_mean - _hill_intercept(N_vec, s0_ip, b_ip, c_ip, fa),
                        "D-", color="black", ms=3, lw=1.8, alpha=0.90,
                        label="Hill+s₀ profile")

        ax_res.axhspan(-20, 20, color="grey", alpha=0.07, zorder=0)

        ax_res.set_xscale("log")
        ax_res.set_xlim(N_plot[0] * 0.8, N_plot[-1])
        ax_res.tick_params(labelbottom=False)
        ax_res.set_ylabel("Residual\n(data − fit)", fontsize=8.5)
        ax_res.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
        ax_res.grid(True, which="major", ls="--", lw=0.35, alpha=0.4)
        ax_res.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):+}"))

        # ── Weight profile ─────────────────────────────────────────────────────
        sc = res["scale_vec"]
        w  = 1.0 / sc
        w_rel = w / w.mean()

        ax_weight.fill_between(N_vec, 0, w_rel, color=color, alpha=0.25)
        ax_weight.plot(N_vec, w_rel, color=color, lw=1.2)

        # Mark the tail-anchor region
        tail_cut = N_vec.max() * 0.90
        tail_m   = N_vec >= tail_cut
        if tail_m.any():
            ax_weight.axvspan(N_vec[tail_m].min(), N_vec.max(),
                              color=color, alpha=0.12,
                              label=f"Tail anchor (top 10%)")

        ax_weight.set_xscale("log")
        ax_weight.set_xlim(N_plot[0] * 0.8, N_plot[-1])
        ax_weight.set_xlabel("Number of samples N  (log scale)", fontsize=10)
        ax_weight.set_ylabel("Relative\nfit weight", fontsize=8.5)
        ax_weight.legend(fontsize=7, loc="upper left", framealpha=0.9)
        ax_weight.grid(True, which="major", ls="--", lw=0.35, alpha=0.4)
        ax_weight.set_ylim(bottom=0)

    fig.suptitle(
        f"Hill-MM accumulation fit — profile-likelihood vs gradient descent  |  "
        f"{ref_name}  breadth ≥ {threshold:.0f}%\n"
        f"Black = Hill+s₀ profile (global optimum, 3 params)  |  "
        f"dashed = Hill profile (a={fixed_a} fixed)  |  faded = Hill free (local-min prone)\n"
        "Mean ± q16/q84 (filled); median (faint ×)  |  Row 2: residuals  |  Row 3: fit weights",
        fontsize=10, fontweight="bold", y=1.01)

    fname = (f"accumulation_fixed_a_comparison_"
             f"{ref_name}_{threshold:.0f}pct.png")
    fpath = os.path.join(out, fname)
    apply_display_text(fig)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → {fpath}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare free-a vs fixed-a Hill-MM accumulation fits.")
    parser.add_argument("--ref", nargs="+", default=["UHGG"],
                        metavar="REF",
                        help="Reference DB(s) to analyse  (default: UHGG)")
    parser.add_argument("--threshold", type=float, default=10.0,
                        metavar="PCT",
                        help="Breadth threshold %%  (default: 10)")
    parser.add_argument("--fixed-a", type=int, default=DEFAULT_FIXED_A,
                        dest="fixed_a",
                        metavar="N",
                        help=f"Asymptote to fix  (default: {DEFAULT_FIXED_A})")
    args = parser.parse_args(argv)

    for ref in args.ref:
        run_fixed_a_comparison(ref,
                               threshold=args.threshold,
                               fixed_a=args.fixed_a)


if __name__ == "__main__":
    main()
