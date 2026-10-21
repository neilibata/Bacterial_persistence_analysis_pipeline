"""
symbolic_regression_stratified.py
===================================
Stratified accumulation analysis: splits each broad lifestyle group into
biologically/geographically meaningful sub-populations, then for each:

  1. Fits all candidate accumulation models (Richards, LogNorm CDF, Gompertz,
     Hill-MM, Weibull, Log-linear) to the exact E[S(N)] curve.
  2. Plots accumulation curves + model fits.
  3. Plots doubling-bin prevalence histogram (1, 2-3, 4-7, 8-15, … detections)
     on a log y-axis with k=0 bar excluded from fit.

Stratification
--------------
  H-G:          Hadza  |  H-G other
  Transitional: Honduras  |  Transitional other
  Industrial:   Europe  |  North America  |  Other Industrial

Hadza detection follows the pipeline's flexible text-search across Population,
Detailed location, Sample ID, StudyFolder, Author columns (case-insensitive),
with Tanzania H-G as a fallback — identical to section1_figure.py.

Honduras detection: Country_norm == "Honduras"  (aliases HON / HND → Honduras
via config.COUNTRY_ALIASES, already applied by loading.py).

Industrial continent split: uses meta["Continent"] computed by loading.py.

Usage
-----
    python -m persistence.symbolic_regression_stratified --ref UHGG
    python -m persistence.symbolic_regression_stratified --ref UHGG lev
    python -m persistence.symbolic_regression_stratified --ref UHGG --threshold 10
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit
from scipy.special import erf

warnings.filterwarnings("ignore")

try:
    from .plot_utils import apply_display_text
except ImportError:
    from matplotlib.text import Text

    def apply_display_text(fig):
        for artist in fig.findobj(match=lambda item: isinstance(item, Text)):
            text = artist.get_text()
            for old in ("Agro-Pastorialist", "Transitional"):
                text = str(text).replace(old, "Agro-Pastoralist")
            artist.set_text(text)


# ════════════════════════════════════════════════════════════════════════════
#  MODEL LIBRARY  (identical to symbolic_regression_accumulation.py)
# ════════════════════════════════════════════════════════════════════════════

def _safe_pow(base, exp):
    return np.power(np.maximum(np.asarray(base, float), 1e-12), float(exp))

def hill(N, a, b, c):
    return a * _safe_pow(N, c) / (_safe_pow(b, c) + _safe_pow(N, c))

def gompertz_logN(N, a, b, c):
    return a * np.exp(-b * np.exp(-c * np.log(np.maximum(N, 1e-12))))

def lognorm_cdf(N, a, mu, sigma):
    z = (np.log(np.maximum(N, 1e-12)) - mu) / max(float(sigma), 1e-9)
    return a * 0.5 * (1.0 + erf(z / np.sqrt(2)))

def richards_logN(N, a, b, c, nu):
    logN = np.log(np.maximum(N, 1e-12))
    val  = 1.0 + np.exp(-float(c) * (logN - np.log(max(float(b), 1e-9))))
    return a / np.power(np.maximum(val, 1e-9), 1.0 / max(float(nu), 1e-9))

def weibull(N, a, b, c):
    return a * (1.0 - np.exp(-_safe_pow(
        np.asarray(N, float) / max(float(b), 1e-9), c)))

def log_linear(N, a, b):
    return a + b * np.log10(np.maximum(N, 1e-12))


def _registry(S_max, N_med):
    return {
        "Hill-MM":        dict(func=hill,         pnames=["a","b","c"],
            p0=[S_max*1.1, N_med, 0.55],
            bounds=([10,1,0.05],[S_max*1.5,1e7,2.0])),
        "Gompertz(logN)": dict(func=gompertz_logN, pnames=["a","b","c"],
            p0=[S_max*1.05, 5.0, 0.8],
            bounds=([10,0.01,0.05],[S_max*1.5,200,5.0])),
        "LogNorm CDF":    dict(func=lognorm_cdf,   pnames=["a","μ","σ"],
            p0=[S_max*1.1, np.log(N_med), 2.0],
            bounds=([10,-5,0.1],[S_max*1.5,20,12])),
        "Richards(logN)": dict(func=richards_logN, pnames=["a","b","c","ν"],
            p0=[S_max*1.05, N_med, 1.5, 0.5],
            bounds=([10,1,0.1,0.05],[S_max*1.5,1e6,10,5.0])),
        "Weibull":        dict(func=weibull,        pnames=["a","b","c"],
            p0=[S_max*1.1, N_med, 0.5],
            bounds=([10,1,0.05],[S_max*1.5,1e7,2.0])),
        "Log-linear":     dict(func=log_linear,     pnames=["a","b"],
            p0=[S_max*0.1, S_max/4],
            bounds=([-500,10],[S_max,S_max])),
    }

MODEL_STYLES = {
    "Richards(logN)": dict(color="#2b2d42", lw=1.4, ls="-",  zorder=7, alpha=0.92),
    "LogNorm CDF":    dict(color="#8d99ae", lw=1.2, ls="--", zorder=6, alpha=0.88),
    "Gompertz(logN)": dict(color="#ef233c", lw=1.6, ls="--", zorder=5, alpha=0.72),
    "Hill-MM":        dict(color="#06d6a0", lw=1.2, ls=":",  zorder=4, alpha=0.72),
    "Weibull":        dict(color="#f4a261", lw=1.4, ls=":",  zorder=3, alpha=0.62),
    "Log-linear":     dict(color="#adb5bd", lw=1.2, ls=":",  zorder=2, alpha=0.52),
}


# ════════════════════════════════════════════════════════════════════════════
#  EXACT ACCUMULATION
# ════════════════════════════════════════════════════════════════════════════

def bootstrap_accumulation(breadth_block, threshold, n_points=60, n_boot=100, rng=None):
    """
    Depth-aware accumulation curve via random subsampling (without replacement).

    This is the same approach used by the main pipeline (accumulation.py:
    bootstrap_curve) — it respects per-sample depth heterogeneity because it
    randomly draws actual samples, not synthetic Poisson replicates.

    Returns (N_vec, mean_vec, q16_vec, q84_vec, n_mic).
    """
    from .accumulation import bootstrap_curve
    if rng is None:
        rng = np.random.default_rng(42)

    n_samp = breadth_block.shape[0]
    Ns = np.unique(np.concatenate([
        np.arange(1, min(25, n_samp) + 1),
        np.round(np.logspace(np.log10(5), np.log10(n_samp), n_points)).astype(int),
        np.round(np.linspace(n_samp * 0.60, n_samp, 20)).astype(int),
    ]))
    Ns = np.array(sorted({int(n) for n in Ns if 1 <= n <= n_samp}), dtype=int)

    (N_vec, mean_vec, std_vec, n_mic,
     med_vec, q16_vec, q84_vec, _) = bootstrap_curve(
        breadth_block, Ns, n_boot, rng, threshold)

    return N_vec, mean_vec, std_vec, q16_vec, q84_vec, n_mic, n_boot


# ════════════════════════════════════════════════════════════════════════════
#  FITTING
# ════════════════════════════════════════════════════════════════════════════

def fit_all(N_vec, S_vec, sem_vec=None):
    """
    Fit all candidate models, optionally weighted by 1/SEM^2.

    Weighted R^2 (wR^2):
        wR^2 = 1 - sum(w*(y-yhat)^2) / sum(w*(y-ymean_w)^2)
    where w = 1/SEM^2 and ymean_w is the weighted mean.
    Points with small SEM (well-constrained bootstrap mean, i.e. high N)
    are given proportionally more influence.

    frac_in_ci: fraction of N-points whose |residual| <= SEM*1.96.
    A perfect model in a noisy dataset should have frac_in_ci ~ 0.95.
    """
    reg = _registry(float(S_vec.max()), float(np.median(N_vec)))
    n   = len(N_vec)

    use_weights = (sem_vec is not None and np.isfinite(sem_vec).all())
    if use_weights:
        sem_vec = np.maximum(sem_vec, 0.1)  # floor: avoids zero at N=1
        w = 1.0 / (sem_vec ** 2)
        w = w / w.sum() * n          # normalise so sum(w) = n
    else:
        w = np.ones(n)
    w_mean_y = float(np.dot(w, S_vec) / w.sum())

    out = {}
    for mname, spec in reg.items():
        k = len(spec["pnames"])
        try:
            sigma_arg = sem_vec if use_weights else None
            popt, _ = curve_fit(
                spec["func"], N_vec, S_vec,
                p0=spec["p0"], bounds=spec["bounds"],
                sigma=sigma_arg, absolute_sigma=use_weights,
                maxfev=40000, ftol=1e-12, xtol=1e-12,
            )
            yhat  = spec["func"](N_vec, *popt)
            resid = S_vec - yhat

            # Unweighted
            ssr  = float(np.sum(resid**2))
            ss_t = float(np.sum((S_vec - S_vec.mean())**2))
            r2   = 1.0 - ssr / ss_t if ss_t > 0 else np.nan
            aic  = n * np.log(max(ssr / n, 1e-20)) + 2 * k
            rmse = np.sqrt(ssr / n)

            # SEM-weighted
            wssr  = float(np.dot(w, resid**2))
            wss_t = float(np.dot(w, (S_vec - w_mean_y)**2))
            wr2   = 1.0 - wssr / wss_t if wss_t > 0 else np.nan
            waic  = n * np.log(max(wssr / n, 1e-20)) + 2 * k

            # Fraction of residuals inside 95% SEM CI
            if use_weights:
                ci95_hw = sem_vec * 1.96
                frac_in_ci = float(np.mean(np.abs(resid) <= ci95_hw))
            else:
                frac_in_ci = np.nan

            out[mname] = dict(
                popt=popt, pnames=spec["pnames"], k=k,
                r2=r2, aic=aic, rmse=rmse,
                wr2=wr2, waic=waic, frac_in_ci=frac_in_ci,
                failed=False,
            )
        except Exception:
            out[mname] = dict(
                popt=None, pnames=spec["pnames"], k=k,
                r2=np.nan, aic=np.nan, rmse=np.nan,
                wr2=np.nan, waic=np.nan, frac_in_ci=np.nan,
                failed=True,
            )
    return out


def rank_models_globally(all_fits):
    """
    Aggregate SEM-weighted model metrics across all strata.
    Returns a DataFrame sorted by mean wR^2 (descending).
    """
    model_names  = list(next(iter(all_fits.values())).keys())
    strata_names = list(all_fits.keys())
    rows = []
    for m in model_names:
        wr2s, waics, fracs, n_ok = [], [], [], 0
        for s in strata_names:
            fd = all_fits[s][m]
            if not fd["failed"] and np.isfinite(fd["wr2"]):
                wr2s.append(fd["wr2"])
                waics.append(fd["waic"])
                fracs.append(fd["frac_in_ci"])
                n_ok += 1
        if not wr2s:
            continue
        rows.append(dict(
            model=m,
            mean_wr2=float(np.mean(wr2s)),
            mean_waic=float(np.mean(waics)),
            mean_frac_ci=float(np.nanmean(fracs)),
            n_strata=n_ok,
        ))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("mean_wr2", ascending=False).reset_index(drop=True)
    wins = {m: 0 for m in model_names}
    for s in strata_names:
        valid = [(m, all_fits[s][m]["waic"]) for m in model_names
                 if not all_fits[s][m]["failed"]
                 and np.isfinite(all_fits[s][m]["waic"])]
        if valid:
            winner = min(valid, key=lambda x: x[1])[0]
            wins[winner] = wins.get(winner, 0) + 1
    df["n_waic_wins"] = df["model"].map(wins).fillna(0).astype(int)
    return df


# ════════════════════════════════════════════════════════════════════════════
#  DOUBLING-BIN HISTOGRAM
# ════════════════════════════════════════════════════════════════════════════

def _doubling_bins(counts):
    nonzero = counts[counts > 0]
    if not len(nonzero):
        return pd.DataFrame(columns=["k","lo","hi","n_sgbs","label"])
    max_k = int(np.floor(np.log2(nonzero.max()))) + 1
    rows  = []
    for k in range(0, max_k + 1):
        lo, hi = int(2**k), int(2**(k+1))
        n  = int(((nonzero >= lo) & (nonzero < hi)).sum())
        lbl = str(lo) if k == 0 else f"{lo}–{hi-1}"
        rows.append(dict(k=k, lo=lo, hi=hi, n_sgbs=n, label=lbl))
    return pd.DataFrame(rows)


def _fit_slope(bin_df, fit_from_k=1, fit_max_k=4):
    """Linear fit log(n_sgbs) ~ k on k in [fit_from_k, fit_max_k]."""
    sub = bin_df[(bin_df["k"] >= fit_from_k) &
                 (bin_df["k"] <= fit_max_k) &
                 (bin_df["n_sgbs"] > 0)]
    if len(sub) < 2:
        return None
    x = sub["k"].values.astype(float)
    y = np.log(sub["n_sgbs"].values.astype(float))
    w = np.sqrt(sub["n_sgbs"].values.astype(float))
    coeffs = np.polyfit(x, y, 1, w=w)
    return np.poly1d(coeffs)


# ════════════════════════════════════════════════════════════════════════════
#  HADZA / HONDURAS DETECTION  (mirrors pipeline logic exactly)
# ════════════════════════════════════════════════════════════════════════════

def _hadza_mask(meta):
    """
    Flexible Hadza detection: case-insensitive search for 'hadza' across
    Population, Detailed location, Sample ID, StudyFolder, Author.
    Falls back to Tanzania + H-G.  Mirrors section1_figure.py.
    """
    mask = pd.Series(False, index=meta.index)
    for col in ["Population", "Detailed location", "Sample ID",
                "StudyFolder", "Author"]:
        if col in meta.columns:
            mask |= meta[col].astype(str).str.contains(
                "hadza", case=False, na=False)
    if not mask.any():
        # Tanzania fallback
        for cc in ["Country", "Country (full)", "Country_norm"]:
            if cc in meta.columns:
                tz = meta[cc].astype(str).str.contains(
                    "TZA|Tanzania", case=False, na=False)
                mask = tz & (meta["Lifestyle"] == "H-G")
                if mask.any():
                    break
    return mask


def _honduras_mask(meta):
    """Samples whose normalised country is Honduras."""
    for cc in ["Country_norm", "Country", "Country (full)"]:
        if cc in meta.columns:
            return meta[cc].astype(str).str.strip() == "Honduras"
    return pd.Series(False, index=meta.index)


# ════════════════════════════════════════════════════════════════════════════
#  STRATIFICATION
# ════════════════════════════════════════════════════════════════════════════

def build_strata(meta, breadth_df, threshold):
    """
    Returns a dict: stratum_name → (counts_array, n_samples, colour, breadth_block)
    counts_array  = per-SGB integer detection counts (breadth >= threshold)
    breadth_block = the raw breadth DataFrame for that stratum (for bootstrap)

    Order: pooled group first, then its sub-strata, so the figure reads
    left-to-right as: [All H-G | Hadza | H-G other] [All Trans | Honduras | Trans other]
                      [All Industrial | Ind–Europe | Ind–N.Am | Ind–Other]
    """
    presence = (breadth_df >= threshold).astype(np.uint8)

    strata = {}

    def _add(name, ids, col):
        ids = ids.intersection(breadth_df.index)
        if len(ids) < 10:
            return
        c     = presence.loc[ids].sum(axis=0).values
        block = breadth_df.loc[ids]
        strata[name] = (c, len(ids), col, block)
        print(f"  {name:<28} {len(ids):>6,} samples  "
              f"({(c>0).sum():,} SGBs detected)")

    # ── H-G ──────────────────────────────────────────────────────────────────
    hg_ids   = meta.index[meta["Lifestyle"] == "H-G"].intersection(breadth_df.index)
    had_mask = _hadza_mask(meta.loc[hg_ids])
    _add("All H-G",      hg_ids,            "#8B2252")   # dark plum — pooled
    _add("Hadza",        hg_ids[had_mask],  "#CC79A7")
    _add("H-G (other)",  hg_ids[~had_mask], "#D55E00")

    # ── Transitional ─────────────────────────────────────────────────────────
    tr_ids = meta.index[meta["Lifestyle"] == "Transitional"].intersection(breadth_df.index)
    hon_m  = _honduras_mask(meta.loc[tr_ids])
    _add("All Transitional",     tr_ids,          "#8B6914")   # dark gold — pooled
    _add("Honduras",             tr_ids[hon_m],   "#E69F00")
    _add("Transitional (other)", tr_ids[~hon_m],  "#F0E442")

    # ── Industrial ────────────────────────────────────────────────────────────
    ind_ids   = meta.index[meta["Lifestyle"] == "Industrial"].intersection(breadth_df.index)
    continent = meta.loc[ind_ids, "Continent"].fillna("Other")
    _add("All Industrial",   ind_ids,                                       "#1A5276")  # dark blue — pooled
    for cont, col in [("Europe", "#56B4E9"), ("North America", "#0072B2")]:
        _add(f"Industrial – {cont}", ind_ids[continent == cont], col)
    _add("Industrial – Other", ind_ids[~continent.isin(["Europe","North America"])], "#009E73")

    return strata


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE: one column per stratum, 3 rows
#    Row 1 — accumulation curves + model fits
#    Row 2 — model residuals
#    Row 3 — doubling-bin histogram with slope fit
# ════════════════════════════════════════════════════════════════════════════

def plot_all(strata, all_fits, accumulations, out_path, threshold,
             global_ranking=None):
    names = list(strata.keys())
    n_s   = len(names)
    if n_s == 0:
        print("  No strata to plot.")
        return

    # Identify which columns are pooled (bold title + separator line before sub-strata)
    POOLED = {"All H-G", "All Transitional", "All Industrial"}
    # Map each pooled name to its colour for the group divider
    GROUP_DIVIDERS = {}   # col_index → colour; drawn as left-edge highlight

    fig, axes = plt.subplots(
        4, n_s,
        figsize=(6.5 * n_s, 15.5),
        gridspec_kw={"height_ratios": [3, 1, 0.35, 1.6], "hspace": 0.18},
        squeeze=False,
    )

    for col, name in enumerate(names):
        counts, n_samp, colour, _block = strata[name]
        N_vec, mean_vec, std_vec, q16_vec, q84_vec, n_boot = accumulations[name]
        sem_vec = std_vec / np.sqrt(max(n_boot, 1))
        sem95   = sem_vec * 1.96   # 95% CI half-width (SEM-based)
        fits         = all_fits[name]
        ax_c, ax_r, ax_f, ax_d = axes[0][col], axes[1][col], axes[2][col], axes[3][col]

        # Draw a visible left spine for pooled columns so groups are clear
        is_pooled = name in POOLED
        for ax_ in (ax_c, ax_r, ax_f, ax_d):
            if is_pooled and col > 0:
                ax_.spines["left"].set_linewidth(2.5)
                ax_.spines["left"].set_color(colour)
            if is_pooled:
                ax_.patch.set_alpha(0.04)
                ax_.patch.set_facecolor(colour)

        # ── Row 1: linear-space main + log-space inset (bottom-right) ────────
        SHOW_MODELS = ("Richards(logN)", "LogNorm CDF", "Hill-MM")

        # -- Linear-space main axes --
        # Data: q16-q84 band + thinned mean dots with SEM error bars
        ax_c.fill_between(N_vec, q16_vec, q84_vec,
                          color=colour, alpha=0.22, zorder=2)
        step = max(1, len(N_vec) // 40)
        ax_c.errorbar(
            N_vec[::step], mean_vec[::step],
            yerr=sem95[::step],
            fmt="o", color=colour, ms=3.0, capsize=2, lw=0,
            elinewidth=0.9, alpha=0.75, zorder=10,
            label="Bootstrap mean \u00b1 95% SEM CI",
        )
        ax_c.plot(N_vec, mean_vec, color=colour, lw=1.0, alpha=0.40, zorder=1)

        # N_plot in LINEAR space for the main axes
        N_plot_lin = np.linspace(1, n_samp * 1.05, 600)
        # N_plot in LOG space for the inset
        N_plot_log = np.logspace(np.log10(1), np.log10(max(n_samp * 2.5, 1e5)), 700)

        ranked  = sorted(fits.items(),
                         key=lambda kv: kv[1]["waic"]
                         if np.isfinite(kv[1]["waic"]) else np.inf)
        best_waic = ranked[0][1]["waic"] if ranked and not ranked[0][1]["failed"] else np.nan
        reg = _registry(float(mean_vec.max()), float(np.median(N_vec)))

        residuals_by_model = {}
        legend_handles = []
        for mname, fd in ranked:
            if fd["failed"] or mname not in SHOW_MODELS:
                residuals_by_model[mname] = None  # still compute but don't show
                continue
            st    = MODEL_STYLES.get(mname, {})
            dwaic = fd["waic"] - best_waic
            ci_s  = f"  CI={fd['frac_in_ci']:.0%}" if np.isfinite(fd.get("frac_in_ci", float("nan"))) else ""
            pstr  = "  ".join(f"{pn}={v:.2g}" for pn, v in zip(fd["pnames"], fd["popt"]))
            lbl   = (f"{mname}  wR\u00b2={fd['wr2']:.4f}  \u0394wAIC={dwaic:+.0f}{ci_s}"
                     f"\n  {pstr}")

            # Plot on linear-space main axes (no legend here — goes in inset)
            yp_lin = reg[mname]["func"](N_plot_lin, *fd["popt"])
            line,  = ax_c.plot(N_plot_lin, yp_lin, label=lbl, **st)
            legend_handles.append(line)

            yhat = reg[mname]["func"](N_vec, *fd["popt"])
            resid = mean_vec - yhat
            residuals_by_model[mname] = (resid, st)
            ax_r.plot(N_vec, resid,
                      color=st.get("color", "grey"),
                      lw=st.get("lw", 1) * 0.8, ls=st.get("ls", "-"),
                      alpha=st.get("alpha", 0.6), ms=2)

        # Main axes: linear scale, no legend
        ax_c.set_xscale("linear")
        ax_c.set_xlim(0, n_samp * 1.05)
        # y_ceil is set dynamically from model asymptotes (see below)
        ax_c.set_title(name,
                       fontsize=12 if is_pooled else 10,
                       fontweight="bold",
                       color=colour,
                       pad=6 if is_pooled else 3)
        ax_c.set_ylabel(f"SGBs detected  (breadth \u2265 {threshold:.0f}%)", fontsize=9)
        ax_c.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax_c.grid(True, ls="--", lw=0.28, alpha=0.38)
        ax_c.set_xlabel(f"N samples  (n\u2009=\u2009{n_samp:,})",
                        fontsize=8, fontweight="bold", color=colour, labelpad=3)

        # ── Extract asymptotes from the three displayed models ──────────────
        # Each model has 'a' as its first free parameter (the saturation level).
        # Collect them to set a shared y-ceiling and draw reference lines.
        model_asymptotes = {}   # mname → asymptote value
        for mname_a, fd_a in ranked:
            if fd_a["failed"] or mname_a not in SHOW_MODELS:
                continue
            a_val = float(fd_a["popt"][0])  # always the first param for all 3 models
            model_asymptotes[mname_a] = a_val

        n_total_sgbs = int(mean_vec.size > 0 and counts.shape[0])  # total SGBs in matrix
        # Better: use n_sg from the counts array (set later in Row 3); define it here
        n_total_sgbs = int(counts.shape[0])

        # y ceiling: highest asymptote + 50 SGBs headroom
        y_ceil = max(list(model_asymptotes.values()) + [float(mean_vec.max()), float(n_total_sgbs)]) + 80
        y_ceil = float(np.ceil(y_ceil / 50) * 50)  # round to nearest 50

        def _draw_reference_lines(ax, x_max, fontsize=6.5, is_log=False):
            """Draw total-SGBs line and per-model asymptote lines on ax.
            Labels are staggered both horizontally AND vertically when
            values are close together to prevent overlap.
            """
            # Collect all (y_value, label, colour, linestyle) sorted by value
            all_entries = []
            for mname_l, a_v in model_asymptotes.items():
                st_l = MODEL_STYLES.get(mname_l, {})
                mc   = st_l.get("color", "grey")
                short = mname_l.split("(")[0].strip()
                all_entries.append((a_v, f"{short}={a_v:.0f}", mc, "--", 0.7, 0.50))
            all_entries.append((float(n_total_sgbs), f"all={n_total_sgbs:,}",
                                "dimgrey", ":", 0.8, 0.55))
            all_entries.sort(key=lambda t: t[0])

            # x positions: left-side of the plot (5-45%) where curves are low
            # and there is always clear whitespace. This keeps labels away from
            # the right edge and from the mid-range where curves are densest.
            n_l = len(all_entries)
            x_pos = np.linspace(0.05, 0.42, n_l) * x_max

            # Minimum vertical gap between labels (fraction of y_ceil)
            MIN_GAP = y_ceil * 0.028

            # Assign y-label positions: start at line + small offset,
            # then push up any label that overlaps the previous one
            y_label_pos = []
            for i, (a_v, lbl, mc, ls, lw, alpha) in enumerate(all_entries):
                y_lbl = a_v + y_ceil * 0.005
                if i > 0 and y_lbl < y_label_pos[-1] + MIN_GAP:
                    y_lbl = y_label_pos[-1] + MIN_GAP
                y_label_pos.append(y_lbl)

            for i, (a_v, lbl, mc, ls, lw, alpha) in enumerate(all_entries):
                ax.axhline(a_v, color=mc, lw=lw, ls=ls, alpha=alpha, zorder=2)
                ax.text(x_pos[i], y_label_pos[i], lbl,
                        ha="center", va="bottom",
                        fontsize=fontsize - 0.5, color=mc, alpha=0.85)

        # Linear axes: top = S(n_samp) from best model * 1.08
        # This matches the y-value at the right edge of the linear plot
        # to the same point on the log curves, so the two panels share
        # a consistent y-reference without forcing the full asymptote.
        best_fd_shown = next(
            (fd for mname, fd in ranked
             if not fd["failed"] and mname in SHOW_MODELS), None)
        if best_fd_shown is not None:
            s_at_nmax = float(reg[
                next(m for m, fd in ranked
                     if not fd["failed"] and m in SHOW_MODELS)
            ]["func"](n_samp, *best_fd_shown["popt"]))
        else:
            s_at_nmax = float(mean_vec.max())
        y_top_linear = max(s_at_nmax, float(mean_vec.max())) * 1.08
        ax_c.set_ylim(0, y_top_linear)

        # -- Log-space inset: bottom-right, sized to not overlap curves --
        # Curves are densest at low N (left side), so place inset on the right.
        # Use the upper 40% of N (where curves are separating / flattening)
        # to determine safe vertical position.
        ax_ins = ax_c.inset_axes([0.33, 0.05, 0.66, 0.65])  # bottom-right, minimal bottom gap
        ax_ins.fill_between(N_vec, q16_vec, q84_vec,
                            color=colour, alpha=0.20, zorder=1)
        ax_ins.plot(N_vec, mean_vec, color=colour, lw=0.9, alpha=0.40, zorder=1)
        ax_ins.errorbar(
            N_vec[::step], mean_vec[::step], yerr=sem95[::step],
            fmt="o", color=colour, ms=2.5, capsize=1.5, lw=0,
            elinewidth=0.7, alpha=0.70, zorder=10,
        )
        for mname, fd in ranked:
            if fd["failed"] or mname not in SHOW_MODELS:
                continue
            st   = MODEL_STYLES.get(mname, {})
            dwaic = fd["waic"] - best_waic
            ci_s  = f"  CI={fd['frac_in_ci']:.0%}" if np.isfinite(fd.get("frac_in_ci", float("nan"))) else ""
            pstr  = "  ".join(f"{pn}={v:.2g}" for pn, v in zip(fd["pnames"], fd["popt"]))
            lbl   = (f"{mname}  wR\u00b2={fd['wr2']:.4f}  \u0394wAIC={dwaic:+.0f}{ci_s}"
                     f"\n  {pstr}")
            yp_log = reg[mname]["func"](N_plot_log, *fd["popt"])
            ax_ins.plot(N_plot_log, yp_log, label=lbl, **{**st, "lw": st.get("lw",1)*0.85})
        ax_ins.set_xscale("log")
        # Shade the inter-model spread in the inset to highlight divergence
        shown_yhats = []
        for mname_sh, fd_sh in ranked:
            if fd_sh["failed"] or mname_sh not in SHOW_MODELS:
                continue
            shown_yhats.append(reg[mname_sh]["func"](N_plot_log, *fd_sh["popt"]))
        if len(shown_yhats) >= 2:
            y_lo_sh = np.min(shown_yhats, axis=0)
            y_hi_sh = np.max(shown_yhats, axis=0)
            ax_ins.fill_between(N_plot_log, y_lo_sh, y_hi_sh,
                                color="gold", alpha=0.28, zorder=0,
                                label="Model spread")
        ax_ins.set_xlim(0.8, N_plot_log[-1])
        ax_ins.set_ylim(0, y_ceil)
        _draw_reference_lines(ax_ins, N_plot_log[-1] * 0.92,
                              fontsize=5.5, is_log=True)
        # On the linear axes draw only the reference lines that fall within
        # its y-range (i.e. below y_top_linear) — avoids clutter from
        # asymptotes that are well above the visible data range.
        def _draw_reference_lines_clipped(ax, x_max, y_top, fontsize=6.5):
            x_ann = x_max * 0.97
            if n_total_sgbs <= y_top:
                ax.axhline(n_total_sgbs, color="dimgrey", lw=0.8, ls=":", alpha=0.55, zorder=2)
                ax.text(x_ann, n_total_sgbs + y_top * 0.008,
                        f"total SGBs = {n_total_sgbs:,}",
                        ha="right", va="bottom", fontsize=fontsize,
                        color="dimgrey", alpha=0.80)
            for mname_l, a_v in model_asymptotes.items():
                if a_v > y_top * 1.02:
                    continue  # asymptote above visible area — skip
                st_l = MODEL_STYLES.get(mname_l, {})
                mc   = st_l.get("color", "grey")
                ax.axhline(a_v, color=mc, lw=0.7, ls="--", alpha=0.50, zorder=2)
                ax.text(x_ann, a_v + y_top * 0.006,
                        f"{mname_l.split('(')[0].strip()} a={a_v:.0f}",
                        ha="right", va="bottom", fontsize=fontsize - 0.5,
                        color=mc, alpha=0.85)
        _draw_reference_lines_clipped(ax_c, n_samp * 1.04, y_top_linear,
                                      fontsize=6.5)
        ax_ins.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax_ins.tick_params(labelsize=5.5, pad=1)
        ax_ins.set_title("log N", fontsize=6, pad=2, color="dimgrey")
        ax_ins.grid(True, which="both", ls="--", lw=0.22, alpha=0.32)
        # Legend inside the inset
        ax_ins.legend(fontsize=5.0, loc="lower right", framealpha=0.92,
                      handlelength=1.5, borderpad=0.4, labelspacing=0.3)

        # Indicate the inset on the main axes
        # Manual connector lines: from right edge of linear main axes
        # (top = y_top_linear, bottom = 0) to the matching y-positions
        # on the RIGHT edge of the log inset.
        # We draw in axes-fraction coords using ConnectionPatch.
        from matplotlib.patches import ConnectionPatch
        for y_main, y_ins_frac in [
            (y_top_linear, y_top_linear / (y_ceil)),   # top connector
            (0.0,          0.0),                        # bottom connector
        ]:
            # Clip inset fraction to [0, 1]
            y_ins_frac = float(np.clip(y_ins_frac, 0.0, 1.0))
            con = ConnectionPatch(
                xyA=(1.0, y_ins_frac),   # right edge of inset, in axes coords
                xyB=(n_samp * 1.04, y_main),  # right of linear main, in data coords
                coordsA="axes fraction", coordsB="data",
                axesA=ax_ins, axesB=ax_c,
                color="dimgrey", lw=0.7, alpha=0.45, ls="-",
                arrowstyle="-",
            )
            ax_c.add_artist(con)
        # Draw a light border around the inset
        for spine in ax_ins.spines.values():
            spine.set_edgecolor("dimgrey"); spine.set_linewidth(0.6); spine.set_alpha(0.6)

        # ── Row 2: residuals, y-axis = ±2×(SEM×1.96) ────────────────────────
        # SEM = std across n_boot bootstrap replicates / sqrt(n_boot).
        # 95% CI half-width = SEM × 1.96.  y-limits = ±2× that max.
        y_range = float(np.nanmax(sem95)) * 2.0
        y_range = max(y_range, 5.0)

        ax_r.axhline(0, color="black", lw=0.9, zorder=4)
        ax_r.fill_between(N_vec, -sem95, sem95,
                          color=colour, alpha=0.25, zorder=1,
                          label="±95% CI (SEM×1.96)")

        ax_r.set_xscale("log"); ax_r.set_xlim(0.8, N_plot_log[-1])
        ax_r.set_ylim(-y_range, y_range)
        ax_r.set_ylabel("Residual\n(mean\u2009\u2212\u2009fit)", fontsize=8)
        ax_r.grid(True, which="major", ls="--", lw=0.28, alpha=0.38)
        ax_r.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:+.0f}"))
        ax_r.legend(fontsize=5.5, loc="upper left", framealpha=0.85)
        # n= label on x-axis of row 2
        ax_r.set_xlabel(f"N samples  (n\u2009=\u2009{n_samp:,})",
                        fontsize=7.5, fontweight="bold", color=colour, labelpad=3)

        # ── Row 2b: frac_in_ci strip (thin) ─────────────────────────────────
        # One horizontal dot per model showing the fraction of residuals
        # that fall inside the ±95% SEM CI.  0.95 = perfect; lower = bias.
        SHOW_MODELS_CI = ("Richards(logN)", "LogNorm CDF", "Hill-MM")
        ci_colours = {
            "Richards(logN)": MODEL_STYLES.get("Richards(logN)", {}).get("color", "#2b2d42"),
            "LogNorm CDF":    MODEL_STYLES.get("LogNorm CDF",    {}).get("color", "#8d99ae"),
            "Hill-MM":        MODEL_STYLES.get("Hill-MM",        {}).get("color", "#06d6a0"),
        }
        ax_f.axvline(0.95, color="black", lw=0.8, ls="--", alpha=0.5, zorder=3)
        ax_f.axvspan(0.90, 1.00, color="grey", alpha=0.07, zorder=1)
        for mi, mname in enumerate(SHOW_MODELS_CI):
            fd = fits.get(mname)
            if fd is None or fd["failed"]:
                continue
            fci = fd.get("frac_in_ci", np.nan)
            if not np.isfinite(fci):
                continue
            mc = ci_colours.get(mname, "grey")
            ax_f.scatter([fci], [mi], color=mc, s=55, zorder=5,
                         marker="D", edgecolors="white", lw=0.5)
            ax_f.plot([0, fci], [mi, mi], color=mc, lw=2.0, alpha=0.40, zorder=4)
            ax_f.text(fci + 0.005, mi, f"{fci:.0%}",
                      va="center", ha="left", fontsize=6.0, color=mc)
        ax_f.set_yticks(range(len(SHOW_MODELS_CI)))
        ax_f.set_yticklabels(
            [m.split("(")[0].strip() for m in SHOW_MODELS_CI],
            fontsize=6.0,
        )
        ax_f.set_xlim(0.0, 1.08)
        ax_f.set_ylim(-0.7, len(SHOW_MODELS_CI) - 0.3)
        ax_f.axvline(0.95, color="black", lw=0.8, ls="--", alpha=0.45)
        ax_f.set_xlabel("Frac. residuals\ninside 95% CI", fontsize=6.5, labelpad=2)
        ax_f.tick_params(axis="x", labelsize=6.0, pad=1)
        ax_f.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax_f.spines["top"].set_visible(False)
        ax_f.spines["right"].set_visible(False)
        ax_f.grid(True, axis="x", ls="--", lw=0.25, alpha=0.40)

        # ── Row 3: doubling-bin histogram + extrapolation ─────────────────────
        bdf  = _doubling_bins(counts)
        n_sg = len(counts)
        n_ab = int((counts == 0).sum())
        max_k = int(bdf["k"].max()) if not bdf.empty else 0
        # H-G (other) has a decline starting already at k=1-3 before the
        # plateau from the well-shared core.  Fit only k=1-3 so the
        # extrapolation reflects the rare-tail slope, not the plateau.
        # H-G (other): rare-tail decline is in k=1-3 only; k≥4 is plateau
        fit_from_k = 1
        fit_max_k  = 3 if name == "H-G (other)" else 4
        lin  = _fit_slope(bdf, fit_from_k=fit_from_k, fit_max_k=fit_max_k)

        # -- Observed bars (3 visual styles) --
        # For H-G (other), only show bins k=0..3 (fit is on k=1-3);
        # higher bins reflect the shared-core plateau, not the rare tail.
        display_max_k = 3 if name == "H-G (other)" else max_k
        for _, row in bdf.iterrows():
            k, nv = int(row["k"]), int(row["n_sgbs"])
            if nv == 0 or k > display_max_k:
                continue
            if k == 0:
                kw = dict(alpha=0.45, hatch="..", edgecolor=colour, lw=0.8)
            elif 1 <= k <= fit_max_k:
                kw = dict(alpha=0.90, hatch=None, edgecolor="white", lw=1.2)
            else:
                kw = dict(alpha=0.45, hatch=None, edgecolor="white", lw=0.5)
            ax_d.bar(k, nv, width=0.75, color=colour, zorder=3, **kw)

        # -- Extrapolation --
        MAX_DOUBLINGS = 10
        N_CATALOGUE   = n_sg          # total SGBs in reference
        extrap_rows   = []
        cumulative    = 0.0

        if lin is not None:
            slope = lin.coeffs[0]
            for neg_k in range(-1, -(MAX_DOUBLINGS + 1), -1):
                pred = np.exp(lin(neg_k))
                cumulative += pred
                extrap_rows.append(dict(
                    k=neg_k,
                    mult=int(2**(-neg_k)),
                    pred=pred,
                    cumul=cumulative,
                    cumul_frac=cumulative / n_ab if n_ab > 0 else np.nan,
                ))
                if pred < 0.5 or cumulative >= n_ab:
                    break
            extrap_df = pd.DataFrame(extrap_rows)

            # Draw extrapolated bars (hatched, faded)
            for _, er in extrap_df.iterrows():
                k_e, pred_e = int(er["k"]), er["pred"]
                if pred_e < 0.1:
                    continue
                ax_d.bar(k_e, pred_e, width=0.75, color=colour,
                         alpha=0.22, edgecolor=colour, lw=1.0,
                         hatch="///", zorder=2)

            # Separator
            ax_d.axvline(-0.5, color="grey", lw=1.0,
                         ls=(0, (5, 3)), alpha=0.65, zorder=5)
            trans = ax_d.get_xaxis_transform()
            ax_d.text(-0.55, 1.01, "← extrap.", ha="right", va="bottom",
                      fontsize=6.5, color="grey", style="italic",
                      transform=trans)
            ax_d.text(-0.45, 1.01, "observed →", ha="left", va="bottom",
                      fontsize=6.5, color="grey", style="italic",
                      transform=trans)

            # Fit line over observed + extrapolated range
            min_k_shown = int(extrap_df["k"].min()) if not extrap_df.empty else 0
            kk = np.linspace(min_k_shown - 0.4, max_k + 0.4, 400)
            ax_d.plot(kk, np.exp(lin(kk)), color="black", lw=1.8,
                      ls="--", alpha=0.80, zorder=4,
                      label="_nolegend_")
            ax_d.axvspan(0.625, 4.375, alpha=0.07, color=colour, zorder=1)

            # Annotation: doublings needed + recovery fraction
            n_extrap_doublings = len(extrap_df)
            total_pred = float(extrap_df["cumul"].iloc[-1]) if not extrap_df.empty else 0.0
            frac_recoverable  = total_pred / n_ab * 100 if n_ab > 0 else 0.0
            # Cap-limited message
            if total_pred >= n_ab:
                limit_msg = "→ absent pool exhausted"
            else:
                limit_msg = f"→ {total_pred:.0f} ({frac_recoverable:.0f}%) of absent"
            ann = (f"n = {n_samp:,} samples\n"
                   f"slope={slope:+.2f}/doubling (k=1–{fit_max_k})\n"
                   f"{n_extrap_doublings} doublings shown\n"
                   f"{limit_msg}\n"
                   f"absent: {n_ab:,} ({n_ab/n_sg*100:.1f}%)")
            ax_d.text(0.98, 0.98, ann,
                      transform=ax_d.transAxes, ha="right", va="top",
                      fontsize=6.8, family="monospace",
                      bbox=dict(boxstyle="round,pad=0.3",
                                fc="white", ec=colour, alpha=0.88))
        else:
            min_k_shown = 0
            ax_d.text(0.5, 0.5, "insufficient data for fit",
                      transform=ax_d.transAxes, ha="center", va="center",
                      fontsize=8, color="grey")

        ax_d.set_yscale("log")
        ax_d.set_ylim(10, 1000)
        ax_d.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax_d.yaxis.set_minor_formatter(mticker.NullFormatter())

        # X-ticks: negative = "×N more" | positive = bin lower bound
        tick_ks, tick_lbls = [], []
        all_ks = list(range(min_k_shown, max_k + 1))
        for k in all_ks:
            tick_ks.append(k)
            if k < 0:
                tick_lbls.append(f"×{2**(-k):,}")
            elif k == 0:
                tick_lbls.append("1")
            elif k <= 4 or k % 2 == 0:
                tick_lbls.append(f"{int(2**k):,}")
            else:
                tick_lbls.append("")
        ax_d.set_xticks(tick_ks)
        ax_d.set_xticklabels(tick_lbls, rotation=40, ha="right", fontsize=7)
        ax_d.set_xlim(min_k_shown - 0.7, max_k + 0.7)

        ax_d.set_xlabel("← more data needed  |  detection count  |  observed →",
                        fontsize=8)
        ax_d.set_ylabel("SGBs  (log scale)", fontsize=8.5)
        ax_d.legend(fontsize=7, loc="upper right", framealpha=0.9)
        ax_d.grid(True, which="major", ls="--", lw=0.28, alpha=0.38)

    # ── Global model ranking figure ─────────────────────────────────────────
    if global_ranking is not None and not global_ranking.empty:
        gr     = global_ranking.reset_index(drop=True)
        n_rows = len(gr)
        fig2, ax_gr = plt.subplots(figsize=(9, max(3.0, n_rows * 0.55 + 1.5)))
        y    = np.arange(n_rows)
        cols = [MODEL_STYLES.get(m, {}).get("color", "#aaa") for m in gr["model"]]
        ax_gr.barh(y, gr["mean_wr2"], color=cols, alpha=0.82, edgecolor="white")
        x_lo = max(0.0, float(gr["mean_wr2"].min()) - 0.002)
        ax_gr.set_xlim(x_lo, 1.0 + (1.0 - x_lo) * 0.30)
        for i, row_g in gr.iterrows():
            ax_gr.text(row_g["mean_wr2"] + (1.0 - x_lo) * 0.005, i,
                       f"  \u0394wAIC wins={int(row_g['n_waic_wins'])}  "
                       f"in-CI={row_g['mean_frac_ci']:.0%}  "
                       f"n={int(row_g['n_strata'])} strata",
                       va="center", fontsize=8.5, color="dimgrey")
        ax_gr.set_yticks(y)
        ax_gr.set_yticklabels(gr["model"], fontsize=10)
        ax_gr.invert_yaxis()
        ax_gr.set_xlabel("Mean SEM-weighted R\u00b2  (across all strata)", fontsize=10)
        ax_gr.set_title(
            "Global model ranking\n"
            "Primary: mean wR\u00b2  |  Secondary: \u0394wAIC wins per stratum  |  Tertiary: fraction of residuals inside 95% SEM CI",
            fontsize=9, fontweight="bold")
        ax_gr.grid(True, axis="x", ls="--", lw=0.4, alpha=0.5)
        gr_path = out_path.replace(".png", "_model_ranking.png")
        fig2.tight_layout()
        apply_display_text(fig2)
        fig2.savefig(gr_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved \u2192 {gr_path}")

    fig.suptitle(
        f"Stratified accumulation analysis  |  breadth ≥ {threshold:.0f}%\n"
        "Row 1: bootstrap mean ± 95% CI (SEM×1.96, dots); q16–84 band (depth-aware bootstrap) + model fits  |  "
        "Row 2: residuals vs mean, y-axis ≡ ±2×95% SEM-CI; ±95% CI band shown\n"
        "Row 3: doubling-bin histogram — bright=fitted (k=1–4), faded=excluded, "
        "hatched left=predicted with N× more data  (capped at 10 doublings or absent pool)",
        fontsize=10, fontweight="bold", y=1.01,
    )
    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  CSV OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def save_csv(strata, all_fits, out_path):
    rows = []
    for name, (counts, n_samp, _, _block) in strata.items():
        for mname, fd in all_fits[name].items():
            row = dict(stratum=name, n_samples=n_samp, model=mname,
                       k=fd["k"], r2=fd["r2"], aic=fd["aic"],
                       wr2=fd.get("wr2", np.nan), waic=fd.get("waic", np.nan),
                       frac_in_ci=fd.get("frac_in_ci", np.nan),
                       rmse=fd["rmse"], failed=fd["failed"])
            if fd["popt"] is not None:
                for pn, pv in zip(fd["pnames"], fd["popt"]):
                    row[f"param_{pn}"] = pv
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def run(ref_name, threshold=10.0):
    from . import config
    from .loading import load_reference

    print(f"\n{'='*65}")
    print(f"  Stratified accumulation — ref: {ref_name}  threshold: {threshold}%")
    print(f"{'='*65}")

    data = load_reference(ref_name)
    out  = config.outdir(ref_name)
    os.makedirs(out, exist_ok=True)

    print("\n  Building strata...")
    strata = build_strata(data.meta, data.breadth, threshold)

    if not strata:
        print("  No strata with ≥ 10 samples.  Check metadata.")
        return

    rng = np.random.default_rng(42)
    print("\n  Running bootstrap accumulation and fitting models...")
    accumulations = {}
    all_fits       = {}
    for name, (counts, n_samp, colour, block) in strata.items():
        print(f"  {name}...", end=" ", flush=True)
        N_vec, mean_vec, std_vec, q16_vec, q84_vec, n_mic, n_boot =             bootstrap_accumulation(block, threshold, rng=rng)
        accumulations[name] = (N_vec, mean_vec, std_vec, q16_vec, q84_vec, n_boot)
        sem_v = std_vec / np.sqrt(max(n_boot, 1))
        fits = fit_all(N_vec, mean_vec, sem_vec=sem_v)
        all_fits[name] = fits
        ranked = sorted([(m, fd) for m, fd in fits.items() if not fd["failed"]],
                        key=lambda kv: kv[1]["waic"])
        if ranked:
            best_m, best_fd = ranked[0]
            print(f"best: {best_m}  wR²={best_fd['wr2']:.5f}  "
                  f"CI={best_fd['frac_in_ci']:.1%}")
        else:
            print("all fits failed")

    fig_path = os.path.join(out,
        f"symbolic_regression_stratified_{threshold:.0f}pct.png")
    csv_path = os.path.join(out,
        f"symbolic_regression_stratified_{threshold:.0f}pct.csv")

    print("\n  Global model ranking (SEM-weighted):")
    ranking = rank_models_globally(all_fits)
    if not ranking.empty:
        print(ranking[["model","mean_wr2","n_waic_wins","mean_frac_ci","n_strata"]].to_string(index=False))

    print("\n  Generating figure...")
    plot_all(strata, all_fits, accumulations, fig_path, threshold,
             global_ranking=ranking)
    save_csv(strata, all_fits, csv_path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stratified symbolic regression on E[S(N)] accumulation.")
    parser.add_argument("--ref", nargs="+", default=["UHGG"], metavar="REF")
    parser.add_argument("--threshold", type=float, default=10.0, metavar="PCT")
    args = parser.parse_args(argv)
    for ref in args.ref:
        run(ref, threshold=args.threshold)


if __name__ == "__main__":
    main()
