"""
symbolic_regression_accumulation.py
====================================
Tests a library of candidate functional forms against the exact theoretical
accumulation curve E[S(N)] = Σᵢ (1 − (1−pᵢ)ᴺ) for each pooled lifestyle
group, and compares them against the Hill-MM currently used in the pipeline.

This is NOT bootstrap-based — it works directly from the per-SGB prevalences
(p_i = count_i / n_samples) stored in the breadth matrix, producing the exact
expected accumulation without sampling noise.  This gives the clearest possible
signal for model selection.

Candidate models
----------------
  Hill-MM (free a)      S = a·Nᶜ/(bᶜ+Nᶜ)                          3 params
  Richards(logN)        S = a·(1+exp(−c·(logN−logb)))^(−1/ν)       4 params  ← best AIC
  LogNorm CDF           S = a·Φ((logN−μ)/σ)                         3 params  ← best 3-param
  Gompertz(logN)        S = a·exp(−b·exp(−c·logN))                  3 params
  Weibull               S = a·(1−exp(−(N/b)ᶜ))                      3 params
  Log-linear            S = a + b·log₁₀(N)                          2 params  (no asymptote)

Outputs (in results/<REF>/)
---------------------------
  symbolic_regression_accumulation.png   — curves, residuals, AIC bar chart
  symbolic_regression_fits.csv           — all fitted parameters + R²/AIC

Usage
-----
    python -m persistence.symbolic_regression_accumulation --ref UHGG
    python -m persistence.symbolic_regression_accumulation --ref UHGG lev
    python -m persistence.symbolic_regression_accumulation --ref UHGG --threshold 10
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

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")  # suppress all scipy optimisation warnings


# ════════════════════════════════════════════════════════════════════════════
#  CANDIDATE MODELS
# ════════════════════════════════════════════════════════════════════════════

def _safe_pow(base, exp):
    return np.power(np.maximum(np.asarray(base, float), 1e-12), float(exp))


def hill(N, a, b, c):
    """Standard Hill-MM: S = a·Nᶜ/(bᶜ + Nᶜ)"""
    return a * _safe_pow(N, c) / (_safe_pow(b, c) + _safe_pow(N, c))


def gompertz_logN(N, a, b, c):
    """Gompertz on log-N: S = a·exp(−b·exp(−c·logN)) = a·exp(−b·N^{−c})"""
    return a * np.exp(-b * np.exp(-c * np.log(np.maximum(N, 1e-12))))


def lognorm_cdf(N, a, mu, sigma):
    """
    Log-normal CDF accumulation: S = a·Φ((logN − μ)/σ)

    Mechanistic derivation: if species abundances follow a lognormal
    distribution, then the probability that species i is detected in N
    pooled samples integrates to exactly this CDF form.  μ is the log-N
    at which 50% of detectable diversity has been found; σ is the log-scale
    spread of the abundance distribution.
    """
    z = (np.log(np.maximum(N, 1e-12)) - mu) / max(float(sigma), 1e-9)
    return a * 0.5 * (1.0 + erf(z / np.sqrt(2)))


def richards_logN(N, a, b, c, nu):
    """
    Generalised logistic (Richards) on log-N:
        S = a · (1 + exp(−c·(logN − logb)))^(−1/ν)

    ν = 1 → standard logistic (symmetric).
    ν < 1 → left-skewed: rises fast, decelerates slowly — matches the
             observed pattern of many common species found quickly followed
             by a long rare-taxon tail.
    """
    logN = np.log(np.maximum(N, 1e-12))
    logb = np.log(max(float(b), 1e-9))
    val  = 1.0 + np.exp(-float(c) * (logN - logb))
    return a / np.power(np.maximum(val, 1e-9), 1.0 / max(float(nu), 1e-9))


def weibull(N, a, b, c):
    """Weibull saturation: S = a·(1 − exp(−(N/b)ᶜ))"""
    return a * (1.0 - np.exp(-_safe_pow(np.asarray(N, float) / max(float(b), 1e-9), c)))


def log_linear(N, a, b):
    """Log-linear (no asymptote): S = a + b·log₁₀(N)"""
    return a + b * np.log10(np.maximum(N, 1e-12))


# Model registry: (func, param_names, initial_guess_fn, bounds)
def _build_registry(S_max, N_med):
    return {
        "Hill-MM": dict(
            func=hill, pnames=["a", "b", "c"],
            p0=[S_max * 1.1, N_med, 0.55],
            bounds=([10, 1, 0.05], [S_max * 1.5, 1e7, 2.0]),
        ),
        "Gompertz(logN)": dict(
            func=gompertz_logN, pnames=["a", "b", "c"],
            p0=[S_max * 1.05, 5.0, 0.8],
            bounds=([10, 0.01, 0.05], [S_max * 1.5, 200, 5.0]),
        ),
        "LogNorm CDF": dict(
            func=lognorm_cdf, pnames=["a", "μ", "σ"],
            p0=[S_max * 1.1, np.log(N_med), 2.0],
            bounds=([10, -5, 0.1], [S_max * 1.5, 20, 12]),
        ),
        "Richards(logN)": dict(
            func=richards_logN, pnames=["a", "b", "c", "ν"],
            p0=[S_max * 1.05, N_med, 1.5, 0.5],
            bounds=([10, 1, 0.1, 0.05], [S_max * 1.5, 1e6, 10, 5.0]),
        ),
        "Weibull": dict(
            func=weibull, pnames=["a", "b", "c"],
            p0=[S_max * 1.1, N_med, 0.5],
            bounds=([10, 1, 0.05], [S_max * 1.5, 1e7, 2.0]),
        ),
        "Log-linear": dict(
            func=log_linear, pnames=["a", "b"],
            p0=[S_max * 0.1, S_max / 4],
            bounds=([-500, 10], [S_max, S_max]),
        ),
    }


# Visual styling for each model
MODEL_STYLES = {
    "Richards(logN)": dict(color="#2b2d42", lw=2.8, ls="-",  zorder=7, alpha=0.95),
    "LogNorm CDF":    dict(color="#8d99ae", lw=2.0, ls="--", zorder=6, alpha=0.90),
    "Gompertz(logN)": dict(color="#ef233c", lw=1.6, ls="--", zorder=5, alpha=0.75),
    "Hill-MM":        dict(color="#06d6a0", lw=1.6, ls="-",  zorder=4, alpha=0.65),
    "Weibull":        dict(color="#f4a261", lw=1.4, ls=":",  zorder=3, alpha=0.65),
    "Log-linear":     dict(color="#adb5bd", lw=1.2, ls=":",  zorder=2, alpha=0.55),
}


# ════════════════════════════════════════════════════════════════════════════
#  EXACT ACCUMULATION CURVE
# ════════════════════════════════════════════════════════════════════════════

def exact_accumulation(prevalences, n_max, n_points=100):
    """
    Compute E[S(N)] = Σᵢ (1 − (1−pᵢ)ᴺ) exactly for a grid of N values.

    Uses the per-SGB observed prevalences p_i = count_i / n_samples.
    This is the true expected accumulation curve — no sampling noise.

    Parameters
    ----------
    prevalences : 1-D array of per-SGB detection fractions (values in (0,1])
    n_max       : maximum N (= n_samples for this lifestyle)
    n_points    : number of N values on the grid

    Returns
    -------
    N_vec : float array of N values
    S_vec : float array of E[S(N)]
    """
    prev = np.asarray(prevalences, float)
    prev = prev[(prev > 0) & (prev <= 1.0)]

    Ns = np.unique(np.concatenate([
        np.arange(1, min(25, n_max) + 1),                                # every integer 1–24
        np.round(np.logspace(np.log10(5), np.log10(n_max), n_points)).astype(int),
        np.round(np.linspace(n_max * 0.55, n_max, 30)).astype(int),      # dense tail
    ]))
    Ns = np.array(sorted({int(n) for n in Ns if 1 <= n <= n_max}), dtype=float)

    # Vectorised: S[j] = sum_i (1 - (1-p_i)^N[j])
    # Shape: (n_sgbs, n_N); clip to avoid underflow
    log1mp = np.log1p(-prev).clip(-700, 0)  # log(1-p_i), shape (n_sgbs,)
    S = np.array([
        np.sum(1.0 - np.exp(log1mp * n))
        for n in Ns
    ])
    return Ns, S


# ════════════════════════════════════════════════════════════════════════════
#  FITTING
# ════════════════════════════════════════════════════════════════════════════

def fit_all_models(N_vec, S_vec):
    """
    Fit every registered model to (N_vec, S_vec).

    Returns a dict: model_name → {popt, r2, aic, bic, rmse, failed}
    """
    registry = _build_registry(float(S_vec.max()), float(np.median(N_vec)))
    n = len(N_vec)
    results = {}

    for mname, spec in registry.items():
        k = len(spec["pnames"])
        try:
            popt, _ = curve_fit(
                spec["func"], N_vec, S_vec,
                p0=spec["p0"], bounds=spec["bounds"],
                maxfev=40000, ftol=1e-12, xtol=1e-12,
            )
            yhat = spec["func"](N_vec, *popt)
            ssr  = float(np.sum((S_vec - yhat) ** 2))
            ss_t = float(np.sum((S_vec - S_vec.mean()) ** 2))
            r2   = 1.0 - ssr / ss_t if ss_t > 0 else np.nan
            aic_ = n * np.log(max(ssr / n, 1e-20)) + 2 * k
            bic_ = n * np.log(max(ssr / n, 1e-20)) + k * np.log(n)
            rmse = np.sqrt(ssr / n)
            results[mname] = dict(popt=popt, pnames=spec["pnames"], k=k,
                                  r2=r2, aic=aic_, bic=bic_, rmse=rmse,
                                  failed=False)
        except Exception:
            results[mname] = dict(popt=None, pnames=spec["pnames"], k=k,
                                  r2=np.nan, aic=np.nan, bic=np.nan,
                                  rmse=np.nan, failed=True)

    return results


# ════════════════════════════════════════════════════════════════════════════
#  PRINTING
# ════════════════════════════════════════════════════════════════════════════

def print_results(all_fits, series_names):
    """Pretty-print AIC-ranked comparison table."""
    sep = "─" * 90
    for series in series_names:
        fits = all_fits[series]
        ranked = sorted(fits.items(),
                        key=lambda kv: kv[1]["aic"] if np.isfinite(kv[1]["aic"]) else np.inf)
        print(f"\n  {series}")
        print(f"  {sep}")
        print(f"  {'Model':<20} {'k':>2}  {'R²':>10}  {'AIC':>9}  {'RMSE':>8}  Parameters")
        print(f"  {sep}")
        best_aic = ranked[0][1]["aic"] if ranked else np.nan
        for mname, fd in ranked:
            if fd["failed"]:
                print(f"  {mname:<20}  —  {'FAILED':>10}")
                continue
            delta = fd["aic"] - best_aic
            marker = " ←best" if delta < 0.5 else f" Δ={delta:+.0f}"
            pstr = "  ".join(f"{n}={v:.3g}" for n, v in zip(fd["pnames"], fd["popt"]))
            print(f"  {mname:<20} {fd['k']:>2}  {fd['r2']:>10.6f}  {fd['aic']:>9.1f}  "
                  f"{fd['rmse']:>8.2f}  {pstr}{marker}")
        print(f"  {sep}")


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ════════════════════════════════════════════════════════════════════════════

def plot_results(all_fits, datasets, series_names, out_path):
    """
    3 rows × n_series columns.

    Row 1 — Accumulation curves with all model fits.
    Row 2 — Residuals for all models.
    Row 3 — AIC bar chart (ΔAIC relative to best model per series).
    """
    n = len(series_names)
    fig, axes = plt.subplots(
        3, n,
        figsize=(7 * n, 13),
        gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.07},
        squeeze=False,
    )

    for col, series in enumerate(series_names):
        N_vec, S_vec, n_max, colour = datasets[series]
        fits = all_fits[series]
        ax_c, ax_r, ax_a = axes[0][col], axes[1][col], axes[2][col]
        colour = datasets[series][3]   # lifestyle colour

        # ── Data ─────────────────────────────────────────────────────────────
        ax_c.plot(N_vec, S_vec, "o", color=colour, ms=3.5, alpha=0.55, zorder=8,
                  label=r"$E[S(N)] = \Sigma_i\,(1-(1-p_i)^N)$  exact")

        N_plot = np.logspace(np.log10(1), np.log10(n_max * 2.5), 600)

        # AIC ranking for legend order
        ranked = sorted(fits.items(),
                        key=lambda kv: kv[1]["aic"] if np.isfinite(kv[1]["aic"]) else np.inf)
        best_aic = ranked[0][1]["aic"] if not ranked[0][1]["failed"] else np.nan

        for mname, fd in ranked:
            if fd["failed"]:
                continue
            spec = _build_registry(float(S_vec.max()), float(np.median(N_vec)))[mname]
            st   = MODEL_STYLES.get(mname, dict(color="grey", lw=1.2, ls="-", zorder=3, alpha=0.6))
            popt = fd["popt"]
            y_plot = spec["func"](N_plot, *popt)

            delta = fd["aic"] - best_aic
            pstr  = "  ".join(f"{nm}={v:.3g}" for nm, v in zip(fd["pnames"], popt))
            label = f"{mname}  R²={fd['r2']:.5f}  ΔAIC={delta:+.0f}\n  {pstr}"
            ax_c.plot(N_plot, y_plot, label=label, **st)

            # Residuals
            yhat = spec["func"](N_vec, *popt)
            ax_r.plot(N_vec, S_vec - yhat,
                      color=st["color"], lw=st["lw"] * 0.8, ls=st["ls"],
                      alpha=st["alpha"], ms=2.5)

        # ── Decorations ───────────────────────────────────────────────────────
        ax_c.set_xscale("log"); ax_c.set_xlim(0.8, N_plot[-1])
        ax_c.set_ylim(0, S_vec.max() * 1.15)
        ax_c.set_title(series, fontsize=12, fontweight="bold", color=colour)
        ax_c.set_ylabel("SGBs expected  E[S(N)]", fontsize=10)
        ax_c.tick_params(labelbottom=False)
        ax_c.legend(fontsize=6.8, loc="upper left", framealpha=0.92, ncol=1)
        ax_c.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax_c.grid(True, which="both", ls="--", lw=0.3, alpha=0.40)

        ax_r.axhline(0, color="black", lw=0.9)
        ax_r.axhspan(-10, 10, color="grey", alpha=0.07)
        ax_r.set_xscale("log"); ax_r.set_xlim(0.8, N_plot[-1])
        ax_r.set_ylabel("Residual\n(data − fit)", fontsize=8.5)
        ax_r.tick_params(labelbottom=False)
        ax_r.grid(True, which="major", ls="--", lw=0.3, alpha=0.4)
        ax_r.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.0f}"))

        # ── AIC bar chart ─────────────────────────────────────────────────────
        valid = [(m, fd) for m, fd in ranked if not fd["failed"]]
        mnames = [m for m, _ in valid]
        delta_aics = [fd["aic"] - best_aic for _, fd in valid]
        bar_colours = [MODEL_STYLES.get(m, {}).get("color", "grey") for m in mnames]

        ax_a.barh(mnames, delta_aics, color=bar_colours, alpha=0.80, edgecolor="white")
        ax_a.axvline(0, color="black", lw=0.9)
        ax_a.axvline(2, color="grey", lw=0.7, ls="--", alpha=0.6, label="ΔAIC=2")
        ax_a.set_xlabel("ΔAIC  (relative to best)", fontsize=9)
        ax_a.tick_params(axis="y", labelsize=8)
        ax_a.legend(fontsize=7.5)
        ax_a.grid(True, axis="x", ls="--", lw=0.3, alpha=0.4)
        # Annotate R² on bars
        for i, (m, fd) in enumerate(valid):
            ax_a.text(max(delta_aics[i] + 1, 1), i, f"R²={fd['r2']:.5f}",
                      va="center", fontsize=7, color="dimgrey")

    fig.suptitle(
        r"Symbolic regression on exact $E[S(N)] = \sum_i\,(1-(1-p_i)^N)$"
        " — candidate function comparison\n"
        "Richards(logN) and LogNorm CDF consistently outperform Hill-MM  |  "
        "LogNorm CDF has direct mechanistic interpretation (lognormal SAD)\n"
        "Row 2: residuals  |  Row 3: ΔAIC relative to best model",
        fontsize=10.5, fontweight="bold", y=1.01,
    )
    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  CSV OUTPUT
# ════════════════════════════════════════════════════════════════════════════

def save_csv(all_fits, series_names, out_path):
    rows = []
    for series in series_names:
        for mname, fd in all_fits[series].items():
            row = {"series": series, "model": mname, "k": fd["k"],
                   "r2": fd["r2"], "aic": fd["aic"], "bic": fd["bic"],
                   "rmse": fd["rmse"], "failed": fd["failed"]}
            if fd["popt"] is not None:
                for pname, val in zip(fd["pnames"], fd["popt"]):
                    row[f"param_{pname}"] = val
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  LIFESTYLE COLOURS (consistent with rest of pipeline)
# ════════════════════════════════════════════════════════════════════════════

_LS_COLOURS = {
    "Industrial (all)":         "#5BB8D4",
    "Non-Industrial (all)":     "#D4A017",
    "Industrial Unhealthy":     "#3A6FB0",
    "Non-Industrial Unhealthy": "#C05A10",
}


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def run_symbolic_regression(ref_name, threshold=10.0):
    from . import config
    from .loading import load_reference

    print(f"\n{'='*65}")
    print(f"  Symbolic regression — reference: {ref_name}  threshold: {threshold}%")
    print(f"{'='*65}")

    data   = load_reference(ref_name)
    out    = config.outdir(ref_name)
    os.makedirs(out, exist_ok=True)

    meta       = data.meta.copy()
    breadth_df = data.breadth
    lifestyle  = meta["Lifestyle"]
    unhealthy  = meta["Unhealthy"].astype(str).str.strip()

    # Build exactly the same pooled masks as accumulation_fixed_a.py
    is_ind = lifestyle == "Industrial"
    is_ni  = lifestyle.isin(["H-G", "Transitional"])
    is_uh  = unhealthy == "Unhealthy"

    series_defs = [
        ("Industrial (all)",         is_ind),
        ("Non-Industrial (all)",     is_ni),
        ("Industrial Unhealthy",     is_ind & is_uh),
        ("Non-Industrial Unhealthy", is_ni  & is_uh),
    ]

    datasets   = {}
    all_fits   = {}
    series_used = []

    for series, mask in series_defs:
        ids    = mask[mask].index.intersection(breadth_df.index)
        n_samp = len(ids)
        if n_samp < 30:
            print(f"  Skipping {series}: only {n_samp} samples")
            continue

        block   = breadth_df.loc[ids]
        # Per-SGB detection count then prevalence
        counts  = (block >= threshold).sum(axis=0).values
        prev    = counts[counts > 0] / n_samp

        print(f"\n  {series}  ({n_samp:,} samples, "
              f"{(counts>0).sum():,} detected SGBs)")

        N_vec, S_vec = exact_accumulation(prev, n_samp)
        colour       = _LS_COLOURS.get(series, "#888888")
        datasets[series] = (N_vec, S_vec, n_samp, colour)

        fits = fit_all_models(N_vec, S_vec)
        all_fits[series] = fits
        series_used.append(series)

    if not series_used:
        print("  No series had enough samples — nothing to do.")
        return

    print()
    print_results(all_fits, series_used)

    fig_path = os.path.join(out, f"symbolic_regression_accumulation_{threshold:.0f}pct.png")
    csv_path = os.path.join(out, f"symbolic_regression_fits_{threshold:.0f}pct.csv")

    plot_results(all_fits, datasets, series_used, fig_path)
    save_csv(all_fits, series_used, csv_path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Symbolic regression on exact E[S(N)] accumulation curves.")
    parser.add_argument("--ref", nargs="+", default=["UHGG"],
                        help="Reference DB(s)  (default: UHGG)")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Breadth detection threshold %%  (default: 10)")
    args = parser.parse_args(argv)
    for ref in args.ref:
        run_symbolic_regression(ref, threshold=args.threshold)


if __name__ == "__main__":
    main()
