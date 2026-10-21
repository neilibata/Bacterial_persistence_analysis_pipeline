#!/usr/bin/env python3
"""
accumulation_summary_v2.py
--------------------------
2×2 summary figure: SGBs detected as a function of sampling effort.

Layout
------
Row 0: accumulation vs NUMBER OF SAMPLES (per-sample rarefaction)
Row 1: accumulation vs NUMBER OF STUDIES (per-study rarefaction)
Col 0: Carter reference (UHGG)
Col 1: Leviatan reference (lev)

Each panel
----------
- Outer axes: linear x-scale
  * Bootstrap mean ± 1 SEM error bars (one point per checkpoint)
  * Hill-type MM fit curve (solid)
  * Asymptote horizontal line + shaded ± CI band
  * R² of the Hill-MM fit annotated in the panel
  * y-axis ticks at 0, 1000, 2000, 3000, 4000 (integer labels)
  * No grid lines
  * x-axis: "Number of samples" or "Number of studies" as appropriate

- Inset (log-x scale, lower-right, 40% × 38% of axes):
  * Same bootstrap mean ± SEM and Hill-MM curve as outer (log-x)
  * Legend only: Industrial / Non-Industrial colour patches, no other text
  * Clean minimal frame, no ticks on top/right

Usage
-----
    python accumulation_summary_v2.py \\
        --ref-carter UHGG --ref-lev lev \\
        --out accumulation_summary_v2.png \\
        --module-path /path/to/rank_abundance_analysis.py \\
        --n-iter 200 --seed 0

NOTE: This figure shows posterior-weighted accumulation E[S(k)], not hard
detections under p_genuine_10pct >= 0.5.  S∞ is a liberal upper bound on
the number of SGBs with any above-prior evidence.
"""

import argparse, os, sys, importlib, importlib.util, warnings, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from scipy.stats import t as t_dist

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Typography ────────────────────────────────────────────────────────────────
_FS  = 11   # body: tick labels, annotations
_FSL = 12   # axis labels
_FST = 13   # panel title (column header)

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":         _FS,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,   # no grid lines anywhere
    "figure.dpi":        150,
})

# ── Colours (Wong 2011 palette, aligned to rank_abundance_analysis.py) ────────
PALETTE = {
    "sky":    "#56B4E9",   # Industrial
    "orange": "#E69F00",   # Non-Industrial
}
LS_COLORS  = {"Industrial": PALETTE["sky"], "Non-Industrial": PALETTE["orange"]}
LS_MARKERS = {"Industrial": "o",            "Non-Industrial": "s"}
LS_ORDER   = ["Industrial", "Non-Industrial"]

# ── Module import ─────────────────────────────────────────────────────────────
def _pre_parse_module_path():
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a in ("--module-path", "--module_path") and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(("--module-path=", "--module_path=")):
            return a.split("=", 1)[1]
    return None


def _import_ra(explicit_path=None):
    import re as _re
    sd = os.path.dirname(os.path.abspath(__file__)) if not explicit_path else None
    candidates = (
        [explicit_path] if explicit_path
        else [os.path.join(sd, "rank_abundance_analysis.py"),
              os.path.join(sd, "persistence", "rank_abundance_analysis.py"),
              os.path.join(os.path.dirname(sd), "rank_abundance_analysis.py"),
              os.path.join(os.path.dirname(sd), "persistence", "rank_abundance_analysis.py")]
    )
    for p in candidates:
        if p and os.path.isfile(p):
            spec   = importlib.util.spec_from_file_location("_ra", p)
            mod    = importlib.util.module_from_spec(spec)
            ra_src = open(p).read()
            ra_src = _re.sub(
                r'^[^\n]*=\s*_resolve_pipeline_imports\(\)[^\n]*$',
                'load_reference = None; _cfg = None  # stubbed',
                ra_src, flags=_re.MULTILINE,
            )
            exec(compile(ra_src, p, "exec"), mod.__dict__)
            print(f"  RA module: {os.path.abspath(p)}")
            return mod
    raise FileNotFoundError(
        "rank_abundance_analysis.py not found.\n"
        "Pass --module-path /path/to/rank_abundance_analysis.py"
    )


_mp = _pre_parse_module_path()
if _mp is None:
    warnings.warn(
        "\n  --module-path not supplied: auto-discovering rank_abundance_analysis.py.\n"
        "  If multiple copies exist this may silently load the wrong version.",
        stacklevel=1,
    )
print("  Importing helpers from rank_abundance_analysis.py ...")
_ra = _import_ra(explicit_path=_mp)

_estimate_sigma_b             = _ra._estimate_sigma_b
_estimate_coverage_efficiency = _ra._estimate_coverage_efficiency
_compute_per_sample_posterior = _ra._compute_per_sample_posterior
_collapse_to_study_posteriors = _ra._collapse_to_study_posteriors
_posterior_accumulation       = _ra._posterior_accumulation
_logspace_checkpoints         = _ra._logspace_checkpoints


# ── Pipeline bootstrap ────────────────────────────────────────────────────────
def _bootstrap_pkg(pkg_dir, alias="mbo"):
    spec = importlib.util.spec_from_file_location(
        alias, os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir])
    pkg  = importlib.util.module_from_spec(spec)
    pkg.__path__ = [pkg_dir]; pkg.__package__ = alias
    sys.modules[alias] = pkg; spec.loader.exec_module(pkg)
    return alias


def _resolve_pipeline():
    try:
        from .loading import load_reference as _lr
        from . import config as _cfg
        return _lr, _cfg
    except ImportError:
        pass
    for name in ("persistence", "microbiome_extinctions"):
        try:
            return (importlib.import_module(f"{name}.loading").load_reference,
                    importlib.import_module(f"{name}.config"))
        except ImportError:
            pass
    sd = os.path.dirname(os.path.abspath(__file__))
    for root in [sd, os.path.dirname(sd)]:
        for cand in ["persistence-patched", "persistence", "microbiome_extinctions"]:
            p = os.path.join(root, cand)
            if os.path.isdir(p) and os.path.isfile(os.path.join(p, "__init__.py")):
                al = _bootstrap_pkg(p)
                return (importlib.import_module(f"{al}.loading").load_reference,
                        importlib.import_module(f"{al}.config"))
    raise ImportError("Pipeline package not found.")


load_reference, _cfg = _resolve_pipeline()


# ── Hill-MM fitting ───────────────────────────────────────────────────────────
def _fit_hill(ns, Ss, S_ref, S_sem=None):
    """
    Fit S(n) = S_max * n^h / (K^h + n^h).
    Returns dict: S_max, K, h, S_pred, success, r2.
    """
    ns_f = np.asarray(ns, float); Ss_f = np.asarray(Ss, float)
    if S_sem is not None and len(S_sem) == len(Ss_f):
        w = 1.0 / np.maximum(S_sem, 0.5) ** 2
    else:
        w = np.ones(len(Ss_f))
    w = w / w.sum()

    def obj(p):
        Sm, K, h = p
        nh = ns_f ** h; Kh = K ** h
        return float(np.sum(w * (Sm * nh / (Kh + nh) - Ss_f) ** 2))

    bounds = [(1.0, float(S_ref)), (0.1, float(max(ns_f) * 10)), (0.1, 5.0)]
    best = None
    for K0 in [float(np.median(ns_f)), float(max(ns_f))]:
        for h0 in [0.5, 1.0, 2.0]:
            S0  = float(min(Ss_f[-1] * 2, S_ref))
            res = minimize(obj, x0=[S0, K0, h0], bounds=bounds, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res

    if best is None or (not best.success and best.fun > 1e8):
        return dict(S_max=np.nan, K=np.nan, h=np.nan,
                    S_pred=np.full_like(ns_f, np.nan), success=False, r2=np.nan)

    Sm, K, h = best.x
    nh = ns_f ** h; Kh = K ** h
    pred = Sm * nh / (Kh + nh)

    # Weighted R²
    ss_res = float(np.sum(w * (Ss_f - pred) ** 2))
    ss_tot = float(np.sum(w * (Ss_f - np.average(Ss_f, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return dict(S_max=float(Sm), K=float(K), h=float(h),
                S_pred=pred, success=True, r2=float(r2))


def _asymptote_ci(curves, ns, S_ref, ci=0.95):
    """
    Bootstrap CI on asymptote.  Fit Hill-MM to each bootstrap curve;
    return (S_max_mean_fit, lo, hi, mm_of_mean).
    """
    S_mean = curves.mean(axis=0)
    S_sem  = curves.std(axis=0) / max(np.sqrt(len(curves)), 1.0)
    mm     = _fit_hill(ns, S_mean, S_ref, S_sem=S_sem)

    boots = []
    for c in curves:
        if (c > 2).sum() < 10:
            continue
        r = _fit_hill(ns, c, S_ref)
        if r["success"] and np.isfinite(r["S_max"]):
            boots.append(r["S_max"])

    if len(boots) < 5:
        return mm["S_max"], np.nan, np.nan, mm

    tail = (1.0 - ci) / 2.0
    lo   = float(np.percentile(boots, 100 * tail))
    hi   = float(np.percentile(boots, 100 * (1.0 - tail)))
    return mm["S_max"], lo, hi, mm


# ── Data loading ──────────────────────────────────────────────────────────────
def load_and_prep(ref_key):
    print(f"    Loading '{ref_key}' ...")
    data   = load_reference(ref_key)
    meta   = data.meta.copy()
    breadth = data.breadth
    lengths = data.lengths

    if not (hasattr(lengths, "items") and len(lengths) > 0):
        raise ValueError(
            f"load_and_prep('{ref_key}'): data.lengths is empty or missing. "
            "Ensure all_SGB_lengths_UHGG.txt / all_SGB_lengths_lev.txt are "
            "available and that load_reference populates data.lengths. "
            "Do NOT fall back to a 3 Mb default — genome lengths vary "
            "substantially between SGBs and affect every null curve and prior."
        )
    gl = {str(k): int(v) for k, v in lengths.items()}

    raw_max = float(breadth.values[breadth.values > 0].max()) \
              if (breadth.values > 0).any() else 0.0
    bfrac   = (breadth / 100.0) if raw_max > 1.0 else breadth.copy()
    samps   = [s for s in meta.index if s in bfrac.index and s in data.depth.index]
    bfrac   = bfrac.loc[samps]
    depth   = data.depth.loc[samps]
    meta    = meta.loc[samps]

    c  = _estimate_coverage_efficiency(bfrac, depth)
    sb, se = _estimate_sigma_b(bfrac, depth, c=c)
    print(f"      c={c:.4f}  σ_b [{sb.min():.4f}, {sb.max():.4f}]")

    _SC = ("StudyID", "Study", "cohort", "_study")
    sc  = next((col for col in _SC if col in meta.columns), None)
    if sc:
        print(f"      study col='{sc}' ({meta[sc].nunique()} studies)")

    return dict(bfrac=bfrac, depth=depth, meta=meta, genome_lengths=gl,
                c=c, sigma_bins=sb, sigma_edges=se,
                S_ref=bfrac.shape[1], study_col=sc)


# ── Accumulation computation ──────────────────────────────────────────────────
def compute_accumulation(bfrac_s, depth_s, meta_s, genome_lengths,
                         level, n_iter, rng, study_col=None,
                         sigma_bins=None, sigma_edges=None, c=1.0, n_pts=80):
    _STUDY_COLS = ("StudyID", "Study", "cohort", "_study")
    if study_col is None:
        study_col = next((col for col in _STUDY_COLS if col in meta_s.columns), None)

    print(f"      Posteriors ({len(bfrac_s)} × {len(bfrac_s.columns)}) ...", flush=True)
    det  = _compute_per_sample_posterior(
        bfrac_s, depth_s, c=c,
        sigma_bins=sigma_bins, sigma_edges=sigma_edges,
        genome_lengths=genome_lengths)
    p_df = det["10pct"]

    if level == "study" and study_col is not None:
        n_u   = meta_s[study_col].nunique()
        p_acc = _collapse_to_study_posteriors(p_df, meta_s, study_col)
        N     = n_u
        xlabel = "Number of studies"
        print(f"      Study-level: {len(bfrac_s)} samples → {n_u} studies")
    else:
        p_acc  = p_df
        N      = len(bfrac_s)
        xlabel = "Number of samples"

    ns = _logspace_checkpoints(N, n_pts=n_pts)
    print(f"      Bootstrap ({n_iter} iter) ...", flush=True)
    curves, ns = _posterior_accumulation(p_acc, n_iter=n_iter, rng=rng, checkpoints=ns)

    return dict(
        ns=ns, curves=curves,
        S_mean=curves.mean(0), S_sem=curves.std(0) / max(np.sqrt(n_iter), 1.0),
        N_total=N, xlabel=xlabel,
    )


# ── Panel drawing ─────────────────────────────────────────────────────────────
def _yticks_integer(ax, y_max, step=1000):
    """Set y-ticks at multiples of `step` with plain integer labels."""
    ticks = list(range(0, int(y_max) + step, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:,}" for t in ticks], fontsize=_FS)


def _format_int(n):
    """Format a float as a plain integer string, never scientific notation."""
    return f"{int(round(n)):,}"


def draw_panel(ax, results_by_ls, S_ref, ref_label,
               show_title, show_xlabel, show_ylabel, show_row_label,
               row_label="", extrap_factor=2.0, ci_level=0.95,
               n_iter_asym=50, rng_asym=None):
    """
    Draw one accumulation panel with:
    - linear outer axes (bootstrap mean ± SEM, Hill-MM fit, asymptote ± CI band)
    - log-x inset in the lower-right corner (same data + legend)
    - y-axis ticks at integer multiples of 1000
    - no grid lines
    - R² annotated
    - all annotations checked against each other to avoid overlap
    """
    if rng_asym is None:
        rng_asym = np.random.default_rng(0)

    max_N  = max(r["N_total"] for r in results_by_ls.values())
    xlim   = max_N * extrap_factor
    y_lim  = S_ref * 1.12

    # ── Collect fits for both lifestyles ─────────────────────────────────────
    fits   = {}   # ls → (Sm, lo, hi, mm, ns, S_mean, S_sem)
    r2s    = {}
    for ls in LS_ORDER:
        if ls not in results_by_ls:
            continue
        r      = results_by_ls[ls]
        ns     = r["ns"];  Sm_arr = r["S_mean"];  Sem_arr = r["S_sem"]
        sub    = r["curves"][rng_asym.choice(len(r["curves"]),
                              min(n_iter_asym, len(r["curves"])), replace=False)]
        Sm_val, lo, hi, mm = _asymptote_ci(sub, ns, S_ref, ci=ci_level)
        fits[ls] = (Sm_val, lo, hi, mm, ns, Sm_arr, Sem_arr, r["N_total"])
        r2s[ls]  = mm.get("r2", np.nan)

    # ── Outer axes ────────────────────────────────────────────────────────────
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, y_lim)
    ax.tick_params(labelsize=_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _yticks_integer(ax, y_lim, step=1000)

    if show_xlabel:
        xlabel = next(iter(results_by_ls.values()))["xlabel"]
        ax.set_xlabel(xlabel, fontsize=_FSL, labelpad=4)
    if show_ylabel:
        ylabel = "Number of SGBs detected"
        if show_row_label and row_label:
            ylabel = f"{row_label}\n{ylabel}"
        ax.set_ylabel(ylabel, fontsize=_FSL, labelpad=4)
    if show_title:
        ax.set_title(ref_label, fontsize=_FST, fontweight="bold", pad=6)

    # Extrapolation shading (subtle)
    ax.axvspan(max_N, xlim, color="#f4f4f4", alpha=0.7, zorder=0)

    # Track asymptote annotation y-positions to avoid overlap
    asym_ypos = {}

    for ls in LS_ORDER:
        if ls not in fits:
            continue
        Sm_val, lo, hi, mm, ns, S_mean, S_sem, N_tot = fits[ls]
        col = LS_COLORS[ls]; mk = LS_MARKERS[ls]

        # --- Asymptote band + line (draw first, below data) ------------------
        if np.isfinite(Sm_val):
            if np.isfinite(lo) and np.isfinite(hi):
                ax.axhspan(lo, hi, color=col, alpha=0.12, zorder=1)
            ax.axhline(Sm_val, color=col, lw=1.0, ls=":", alpha=0.75, zorder=2)
            asym_ypos[ls] = Sm_val

        # --- Bootstrap mean ± SEM error bars ---------------------------------
        ax.errorbar(ns, S_mean, yerr=S_sem,
                    fmt="none", ecolor=col, elinewidth=0.6,
                    capsize=1.5, capthick=0.6, alpha=0.40, zorder=3)
        ax.scatter(ns, S_mean, color=col, s=4**2, marker=mk,
                   zorder=4, linewidths=0, label=ls)

        # --- Hill-MM fit curve -----------------------------------------------
        if mm["success"]:
            ns_ext = np.linspace(1, xlim, 600)
            nh = ns_ext ** mm["h"]; Kh = mm["K"] ** mm["h"]
            pred = mm["S_max"] * nh / (Kh + nh)
            ax.plot(ns_ext[ns_ext <= N_tot], pred[ns_ext <= N_tot],
                    color=col, lw=2.0, zorder=5)
            ax.plot(ns_ext[ns_ext > N_tot], pred[ns_ext > N_tot],
                    color=col, lw=1.5, ls="--", alpha=0.55, zorder=5)

    # --- Asymptote annotations (integer, no scientific notation) -------------
    # Nudge apart if too close (< 3% of S_ref)
    positions = {ls: v for ls, v in asym_ypos.items() if np.isfinite(v)}
    if len(positions) == 2:
        ls0, ls1 = LS_ORDER[0], LS_ORDER[1]
        if ls0 in positions and ls1 in positions:
            gap = abs(positions[ls0] - positions[ls1])
            if gap < S_ref * 0.04:
                mid = (positions[ls0] + positions[ls1]) / 2
                positions[ls0] = mid + S_ref * 0.022
                positions[ls1] = mid - S_ref * 0.022

    for ls in LS_ORDER:
        if ls not in fits or ls not in positions:
            continue
        Sm_val, lo, hi, mm, *_ = fits[ls]
        if not np.isfinite(Sm_val):
            continue
        col    = LS_COLORS[ls]
        y_ann  = positions[ls]
        # Annotation: plain integer, range in brackets
        if np.isfinite(lo) and np.isfinite(hi):
            ann = f"S∞ {_format_int(Sm_val)} [{_format_int(lo)}–{_format_int(hi)}]"
        else:
            ann = f"S∞ {_format_int(Sm_val)}"
        ax.text(xlim * 0.02, y_ann * 1.008, ann,
                color=col, fontsize=_FS - 1, va="bottom",
                fontweight="bold", zorder=10,
                clip_on=True)

    # --- R² annotation (upper-right, clear of asymptote labels) -------------
    r2_lines = []
    for ls in LS_ORDER:
        if ls in r2s and np.isfinite(r2s[ls]):
            r2_lines.append(f"{ls[:3]}: R²={r2s[ls]:.3f}")
    if r2_lines:
        ax.text(0.97, 0.97, "\n".join(r2_lines),
                transform=ax.transAxes,
                fontsize=_FS - 1, va="top", ha="right",
                color="#444444",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#cccccc", alpha=0.85),
                zorder=11)

    # ── Log-x inset (lower-right, 40% × 38%) ─────────────────────────────────
    # Position: [x0, y0, width, height] in axes-fraction coords.
    # Placed in lower-right to avoid the rising accumulation curve.
    ax_ins = ax.inset_axes([0.57, 0.03, 0.40, 0.38])
    ax_ins.set_xscale("log")
    ax_ins.set_xlim(1, xlim)
    ax_ins.set_ylim(0, y_lim)
    ax_ins.spines["top"].set_visible(False)
    ax_ins.spines["right"].set_visible(False)
    ax_ins.tick_params(labelsize=_FS - 2, which="both")
    ax_ins.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v):,}" if v >= 1 else ""))
    ax_ins.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, integer=True))
    ax_ins.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v):,}"))

    for ls in LS_ORDER:
        if ls not in fits:
            continue
        Sm_val, lo, hi, mm, ns, S_mean, S_sem, N_tot = fits[ls]
        col = LS_COLORS[ls]; mk = LS_MARKERS[ls]

        ax_ins.errorbar(ns, S_mean, yerr=S_sem,
                        fmt="none", ecolor=col, elinewidth=0.5,
                        capsize=1.0, capthick=0.5, alpha=0.35, zorder=3)
        ax_ins.scatter(ns, S_mean, color=col, s=3**2, marker=mk,
                       zorder=4, linewidths=0)

        if mm["success"]:
            ns_ext = np.logspace(0, np.log10(xlim), 400)
            nh = ns_ext ** mm["h"]; Kh = mm["K"] ** mm["h"]
            pred = mm["S_max"] * nh / (Kh + nh)
            ax_ins.plot(ns_ext[ns_ext <= N_tot], pred[ns_ext <= N_tot],
                        color=col, lw=1.5, zorder=5)
            ax_ins.plot(ns_ext[ns_ext > N_tot], pred[ns_ext > N_tot],
                        color=col, lw=1.2, ls="--", alpha=0.55, zorder=5)

    # --- Legend inside inset (colour patches only, no extra text) ------------
    legend_handles = [
        Patch(facecolor=LS_COLORS[ls], label=ls, alpha=0.85)
        for ls in LS_ORDER if ls in fits
    ]
    ax_ins.legend(handles=legend_handles,
                  loc="upper left", fontsize=_FS - 2,
                  framealpha=0.92, edgecolor="#cccccc",
                  handlelength=1.2, handleheight=0.8,
                  borderpad=0.4, labelspacing=0.3)

    # Inset axis labels — placed so they never overlap the outer axis ticks
    ax_ins.set_xlabel("log scale", fontsize=_FS - 2, labelpad=2, color="#555")
    ax_ins.set_ylabel("SGBs", fontsize=_FS - 2, labelpad=2, color="#555")
    return ax_ins  # caller can test inset via ax.child_axes[0]


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--module-path",  default=None,
                   help="Explicit path to rank_abundance_analysis.py (recommended).")
    p.add_argument("--ref-carter",   default="UHGG")
    p.add_argument("--ref-lev",      default="lev")
    p.add_argument("--out",          default="accumulation_summary_v2.png")
    p.add_argument("--n-iter",       type=int,   default=200)
    p.add_argument("--n-iter-asym",  type=int,   default=100)
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--extrap",       type=float, default=2.0)
    p.add_argument("--ci",           type=float, default=0.95)
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    rng  = np.random.default_rng(args.seed)

    refs = {
        "carter": (args.ref_carter, "Carter"),
        "lev":    (args.ref_lev,    "Leviatan"),
    }

    print("\n══ Loading data ══")
    ref_data = {}
    for rid, (key, label) in refs.items():
        print(f"\n  {label} ({key})")
        try:
            ref_data[rid] = load_and_prep(key)
        except Exception as e:
            print(f"  ERROR: {e}"); traceback.print_exc(); sys.exit(1)

    print("\n══ Computing accumulations ══")
    results = {rid: {"sample": {}, "study": {}} for rid in refs}
    for rid, (key, label) in refs.items():
        d = ref_data[rid]
        for level in ["sample", "study"]:
            print(f"\n  {label}  [{level}]")
            for ls in LS_ORDER:
                mask  = d["meta"]["MacroGroup"] == ls
                samps = d["meta"].index[mask]
                if len(samps) < 5:
                    print(f"    skip {ls}: only {len(samps)} samples"); continue
                n_units = samps.nunique() if level == "sample" else \
                          d["meta"].loc[mask, d["study_col"]].nunique() \
                          if d["study_col"] else len(samps)
                print(f"    {ls}  n_samples={len(samps)}  "
                      f"n_{level}s={n_units}")
                results[rid][level][ls] = compute_accumulation(
                    bfrac_s=d["bfrac"].loc[samps],
                    depth_s=d["depth"].loc[samps],
                    meta_s =d["meta"].loc[samps],
                    genome_lengths=d["genome_lengths"],
                    level=level, n_iter=args.n_iter, rng=rng,
                    study_col=d["study_col"],
                    sigma_bins=d["sigma_bins"],
                    sigma_edges=d["sigma_edges"],
                    c=d["c"],
                )

    print("\n══ Drawing figure ══")
    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 10),
        constrained_layout=True,
        squeeze=False,
    )

    row_labels = ["Per-sample rarefaction", "Per-study rarefaction"]

    for row_idx, level in enumerate(["sample", "study"]):
        for col_idx, (rid, (key, label)) in enumerate(refs.items()):
            ax      = axes[row_idx, col_idx]
            res_ls  = results[rid][level]
            S_ref   = ref_data[rid]["S_ref"]

            if not res_ls:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=_FSL)
                continue

            draw_panel(
                ax             = ax,
                results_by_ls  = res_ls,
                S_ref          = S_ref,
                ref_label      = label,
                show_title     = (row_idx == 0),
                show_xlabel    = True,              # always show x-label
                show_ylabel    = (col_idx == 0),
                show_row_label = (col_idx == 0),
                row_label      = row_labels[row_idx],
                extrap_factor  = args.extrap,
                ci_level       = args.ci,
                n_iter_asym    = args.n_iter_asym,
                rng_asym       = rng,
            )

    # Figure-level caption
    fig.text(
        0.5, -0.015,
        "y-axis: posterior-weighted accumulation E[S(k)] — softer than the "
        "primary p_genuine_10pct ≥ 0.5 detector.  "
        "S∞ is a liberal upper bound on SGBs with any above-prior evidence.  "
        "Shaded band: bootstrap CI on S∞.",
        ha="center", va="top", fontsize=_FS - 2, color="#555",
        style="italic",
    )

    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
