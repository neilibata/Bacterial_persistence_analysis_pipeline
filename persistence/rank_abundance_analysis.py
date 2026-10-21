#!/usr/bin/env python3
"""
rank_abundance_analysis.py
--------------------------
Standalone script: compare breadth-based vs depth-based rank-abundance curves
and mean-variance plots, stratified by sequencing-depth quartile and lifestyle.

Run from the same directory as the pipeline data files:
    python rank_abundance_analysis.py --ref UHGG [--out-dir figures_ra]

Physics of breadth → implied depth (Poisson coverage model):
    If a genome of length L is sequenced at mean depth λ (coverage),
    the expected fraction covered is:
        breadth_fraction = 1 - exp(-λ)
    Inverting:
        λ_implied = -log(1 - breadth_fraction)

    Breadth-based relative abundance per SGB:
        raw_i = λ_implied_i × genome_length_i
        rel_i = raw_i / Σ_j raw_j

Two analyses generated (each as depth-based and breadth-based variants):

  1. Rank-abundance curves  (1 × 4 quartile panels)
     - Each sample is one curve: SGBs sorted by abundance descending
     - Thin semi-transparent lines coloured by Lifestyle
     - Thick line = per-lifestyle group mean curve
     - Question: does curve shape (slope, evenness) vary with sequencing depth?

  2. Mean-variance plots  (1 × 4 quartile panels)
     - One point per SGB; mean and variance computed across samples
     - Axes log-log scale; Poisson reference line (var = mean) overlaid
     - VANISH / BloSSUM / Other colouring; top SGBs labelled by name
     - Taylor's Power Law slope also estimated and shown per panel
"""

import argparse
import sys
import os
import importlib.util
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)

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


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline import — three strategies in priority order
#
#  1. Relative import: script lives *inside* the pipeline package and is run
#     as  python -m <pkg>.rank_abundance_analysis  (most common case).
#
#  2. Sibling package on sys.path: script lives in the data root next to a
#     pipeline folder named  persistence  or  persistence-patched, and
#     is run as  python rank_abundance_analysis.py  from that root.
#
#  3. importlib bootstrap: handles the hyphenated  persistence-patched
#     directory when it cannot be imported directly.
# ══════════════════════════════════════════════════════════════════════════════

def _bootstrap_pkg(pkg_dir: str, alias: str = "mbo"):
    """Load a package whose directory name is not a valid Python identifier."""
    spec = importlib.util.spec_from_file_location(
        alias,
        os.path.join(pkg_dir, "__init__.py"),
        submodule_search_locations=[pkg_dir],
    )
    pkg = importlib.util.module_from_spec(spec)
    pkg.__path__ = [pkg_dir]
    pkg.__package__ = alias
    sys.modules[alias] = pkg
    spec.loader.exec_module(pkg)
    return alias


def _resolve_pipeline_imports():
    """
    Try each strategy in turn; return (load_reference, config_module).
    Raises ImportError if none succeed.
    """
    # ── Strategy 1: relative import (script is inside the package) ────────
    try:
        from .loading import load_reference as _lr
        from . import config as _cfg
        print("  [import] using relative imports (script is inside package)")
        return _lr, _cfg
    except ImportError:
        pass

    # ── Strategy 2: sibling package already importable ─────────────────────
    for pkg_name in ("persistence", "microbiome_extinctions"):
        try:
            _mod = importlib.import_module(pkg_name)
            _lr  = importlib.import_module(f"{pkg_name}.loading").load_reference
            _cfg = importlib.import_module(f"{pkg_name}.config")
            print(f"  [import] found package '{pkg_name}' on sys.path")
            return _lr, _cfg
        except ImportError:
            pass

    # ── Strategy 3: search for the directory and bootstrap ─────────────────
    # Search the script's own directory and its parent for known folder names.
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _candidates = [
        "persistence-patched",
        "persistence",
        "microbiome_extinctions",
    ]
    for search_root in [_script_dir, os.path.dirname(_script_dir)]:
        for name in _candidates:
            pkg_dir = os.path.join(search_root, name)
            if os.path.isdir(pkg_dir) and os.path.isfile(os.path.join(pkg_dir, "__init__.py")):
                alias = _bootstrap_pkg(pkg_dir, alias="mbo")
                _lr   = sys.modules[f"{alias}.loading"].load_reference if f"{alias}.loading" in sys.modules \
                        else importlib.import_module(f"{alias}.loading").load_reference
                _cfg  = importlib.import_module(f"{alias}.config")
                print(f"  [import] bootstrapped '{name}' from {search_root}")
                return _lr, _cfg

    raise ImportError(
        "Could not locate the microbiome pipeline package.\n"
        "Place rank_abundance_analysis.py either:\n"
        "  (a) inside your pipeline package directory (run with python -m <pkg>.rank_abundance_analysis), or\n"
        "  (b) in the parent directory that contains persistence/ or persistence-patched/\n"
        "      (run with python rank_abundance_analysis.py)"
    )


load_reference, _cfg = _resolve_pipeline_imports()

# ── Reinstatement guard ──────────────────────────────────────────────────────
# load_reference returns the PRE-reinstatement cohort. Every published number is
# computed after reinstate_distinct_secondaries promotes compositionally distinct
# secondary runs to independent samples, which changes N. Wrapping the resolved
# loader here keeps this script on the same cohort as the rest of the pipeline.
_lr_raw = load_reference


def load_reference(_ref, *a, **kw):            # noqa: F811 - deliberate shadow
    _d = _lr_raw(_ref, *a, **kw)
    try:
        import sys as _sys
        from pathlib import Path as _P
        _r = str(_P(__file__).resolve().parents[1])
        if _r not in _sys.path:
            _sys.path.insert(0, _r)
        from persistence.reinstate_secondary import reinstate_distinct_secondaries
        from persistence.stats_report import StatsReport
        return reinstate_distinct_secondaries(_d, StatsReport(str(_ref)))
    except Exception as _exc:
        print(f"    [WARN] reinstatement unavailable ({_exc}); this output will NOT "
              f"match the pipeline's cohort")
        return _d



# ══════════════════════════════════════════════════════════════════════════════
#  Visual style
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":        9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linewidth":   0.5,
    "figure.dpi":       150,
})

# ── Wong (2011) colour-blind-safe palette ─────────────────────────────────────
# One canonical reference for every visual category.  All plotting functions
# import colours from here so that a single change propagates everywhere.
PALETTE = {
    "vermillion": "#D55E00",   # VANISH, H-G
    "yellow":     "#F0E442",   # VANISH+, Transitional/Transitional
    "orange":     "#E69F00",   # VanishH+, Non-Industrial
    "blue":       "#0072B2",   # BloSSUM, IndU
    "green":      "#009E73",   # BloSSUM+, IndH, detected
    "sky":        "#56B4E9",   # BlossumH+, Industrial
    "pink":       "#CC79A7",   # Hadza/Carter/unlisted
}

LIFESTYLE_COLORS = {
    "Industrial":     PALETTE["sky"],       # #56B4E9
    "Non-Industrial": PALETTE["orange"],    # #E69F00
}
LIFESTYLE_ORDER = ["Industrial", "Non-Industrial"]

# VANISH/BloSSUM scatter style inside mean-variance plots
SGB_CAT_STYLE = {
    "VANISH":  dict(color=PALETTE["vermillion"], marker="o", zorder=5, s=28, alpha=0.85, linewidths=0.3, edgecolors="white"),
    "BloSSUM": dict(color=PALETTE["blue"],       marker="s", zorder=5, s=28, alpha=0.85, linewidths=0.3, edgecolors="white"),
    "Other":   dict(color="#aaaaaa",             marker=".", zorder=2, s=7,  alpha=0.25, linewidths=0),
}

QUARTILE_LABELS = [
    "Q1  (lowest depth)",
    "Q2",
    "Q3",
    "Q4  (highest depth)",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Breadth → implied depth (Poisson inversion)
# ══════════════════════════════════════════════════════════════════════════════

def breadth_to_implied_depth(breadth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Invert the Poisson coverage model to recover implied mean depth from breadth.

    Pipeline stores breadth as percent (0–100); we divide by 100 first.
    Edge cases:
        breadth = 0   →  implied_depth = 0            (undetected SGB)
        breadth = 100 →  implied_depth clipped to avoid log(0) singularity
    """
    # Detect scale: if any value > 1 the matrix is in percent
    sample_max = float(breadth_df.values[breadth_df.values > 0].max()) if (breadth_df.values > 0).any() else 0.0
    frac = breadth_df / 100.0 if sample_max > 1.0 else breadth_df.copy()

    # Clip away the singularity at b = 1 (but keep b = 0 → depth = 0)
    eps = 1e-6
    frac_clipped = frac.clip(lower=0.0, upper=1.0 - eps)

    implied = pd.DataFrame(
        np.where(frac_clipped > 0, -np.log1p(-frac_clipped), 0.0),
        index=frac.index,
        columns=frac.columns,
    )
    return implied


def compute_breadth_abundance(breadth_df: pd.DataFrame, lengths: pd.Series) -> pd.DataFrame:
    """
    Relative abundance from breadth via Poisson inversion.

    implied_depth_i  = -log(1 - breadth_fraction_i)
    raw_i            = implied_depth_i × genome_length_i
    rel_i            = raw_i / Σ raw_j
    """
    implied_depth = breadth_to_implied_depth(breadth_df)
    raw = implied_depth.mul(lengths, axis=1)
    total = raw.sum(axis=1).replace(0, np.nan)
    return raw.div(total, axis=0).fillna(0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: rank-abundance for one sample
# ══════════════════════════════════════════════════════════════════════════════

def _rank_curve(abundance_row: np.ndarray):
    """Return (ranks, values) for positive-abundance entries only, sorted desc."""
    pos = abundance_row[abundance_row > 0]
    if len(pos) == 0:
        return np.array([]), np.array([])
    sorted_vals = np.sort(pos)[::-1]
    ranks = np.arange(1, len(sorted_vals) + 1)
    return ranks, sorted_vals


def _group_mean_curve(abund_subset: pd.DataFrame):
    """
    Compute the mean rank-abundance curve for a group of samples.

    Each sample is ranked independently; we pad shorter curves with NaN
    and take nanmean so that the mean curve tracks the number of SGBs
    actually detected across samples rather than a fixed-rank grid.
    """
    curves = []
    for _, row in abund_subset.iterrows():
        _, vals = _rank_curve(row.values)
        curves.append(vals)

    if not curves:
        return np.array([]), np.array([])

    max_len = max(len(c) for c in curves)
    padded = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        padded[i, :len(c)] = c

    mean_vals = np.nanmean(padded, axis=0)
    valid = ~np.isnan(mean_vals)
    ranks = np.arange(1, max_len + 1)
    return ranks[valid], mean_vals[valid]


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1 — Rank-abundance curves
# ══════════════════════════════════════════════════════════════════════════════

def plot_rank_abundance(
    abundance:        pd.DataFrame,
    meta:             pd.DataFrame,
    depth_quartile:   pd.Series,
    title_prefix:     str,
    out_path:         str,
    n_sample_lines:   int = 200,
    rng:              np.random.Generator = None,
):
    """
    4-panel rank-abundance figure (one panel per sequencing-depth quartile).

    Each panel shows:
      - Thin semi-transparent lines: individual sample rank-abundance curves,
        coloured by Lifestyle.  Subsampled to ``n_sample_lines`` per lifestyle
        for readability.
      - Thick solid line: per-lifestyle group mean curve.

    Y axis (log scale):  relative abundance
    X axis (log scale):  rank
    """
    if rng is None:
        rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"{title_prefix}  —  Rank-Abundance Curves by Sequencing-Depth Quartile",
        fontsize=11, fontweight="bold", y=1.01,
    )

    for q_idx, ax in enumerate(axes):
        q_label = f"Q{q_idx + 1}"
        samples_q = meta.index[depth_quartile == q_label]
        abund_q   = abundance.loc[samples_q]

        legend_handles = []
        for lifestyle in LIFESTYLE_ORDER:
            ls_mask = meta.loc[samples_q, "MacroGroup"] == lifestyle
            sids    = samples_q[ls_mask]
            if len(sids) == 0:
                continue

            color = LIFESTYLE_COLORS[lifestyle]
            sub   = abund_q.loc[sids]

            # ── Individual sample lines (subsampled) ──────────────────────
            draw_n  = min(n_sample_lines, len(sids))
            draw_ids = rng.choice(sids, size=draw_n, replace=False)
            for sid in draw_ids:
                ranks, vals = _rank_curve(sub.loc[sid].values)
                if len(ranks) == 0:
                    continue
                ax.loglog(ranks, vals, color=color, alpha=0.035, linewidth=0.35)

            # ── Group mean curve ──────────────────────────────────────────
            mean_ranks, mean_vals = _group_mean_curve(sub)
            if len(mean_ranks) > 0:
                ax.loglog(
                    mean_ranks, mean_vals,
                    color=color, linewidth=2.0, alpha=0.92,
                    solid_capstyle="round",
                )

            legend_handles.append(
                Line2D([0], [0], color=color, linewidth=2.0,
                       label=f"{lifestyle}  (n={len(sids):,})")
            )

        # Depth range annotation
        q_depths = meta.loc[samples_q, "_depth"]
        lo_mb = q_depths.min() / 1e6
        hi_mb = q_depths.max() / 1e6
        med_mb = q_depths.median() / 1e6
        ax.set_title(
            f"{QUARTILE_LABELS[q_idx]}\n"
            f"depth [{lo_mb:.0f}–{hi_mb:.0f}] Mb  (med {med_mb:.0f})",
            fontsize=8.5,
        )
        ax.set_xlabel("Rank", fontsize=8)
        if q_idx == 0:
            ax.set_ylabel("Relative abundance", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=7, loc="upper right",
                      framealpha=0.85)

    plt.tight_layout()
    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Mean-variance plots
# ══════════════════════════════════════════════════════════════════════════════

def _taylor_slope(mean_vals: np.ndarray, var_vals: np.ndarray):
    """
    Estimate Taylor's Power Law exponent β where var ∝ mean^β.
    Uses OLS on log-log transformed values for detected (mean > 0) SGBs.
    Returns (slope, intercept, r²).
    """
    pos = (mean_vals > 0) & (var_vals > 0)
    if pos.sum() < 5:
        return np.nan, np.nan, np.nan
    lx = np.log10(mean_vals[pos])
    ly = np.log10(var_vals[pos])
    slope, intercept, r, _, _ = scipy_stats.linregress(lx, ly)
    return slope, intercept, r**2


def plot_mean_variance(
    abundance:      pd.DataFrame,
    meta:           pd.DataFrame,
    depth_quartile: pd.Series,
    vanish_set:     set,
    blossum_set:    set,
    title_prefix:   str,
    out_path:       str,
    n_label:        int = 12,
):
    """
    4-panel mean-variance plot (one panel per sequencing-depth quartile).

    Each panel shows one point per SGB:
      - x-axis: mean relative abundance across samples in that quartile
      - y-axis: variance of relative abundance
      - Colour:  VANISH (red) / BloSSUM (blue) / Other (grey)
      - Top ``n_label`` VANISH+BloSSUM SGBs labelled by name
      - Dashed diagonal: Poisson reference (var = mean)
      - Dotted line: Taylor's Power Law fit (slope β annotated)

    Both axes are log-scaled.
    """
    def _cat(sgb):
        if sgb in vanish_set:  return "VANISH"
        if sgb in blossum_set: return "BloSSUM"
        return "Other"

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    fig.suptitle(
        f"{title_prefix}  —  SGB Mean vs Variance by Sequencing-Depth Quartile",
        fontsize=11, fontweight="bold", y=1.01,
    )

    for q_idx, ax in enumerate(axes):
        q_label   = f"Q{q_idx + 1}"
        samples_q = meta.index[depth_quartile == q_label]
        abund_q   = abundance.loc[samples_q]

        sgb_mean = abund_q.mean(axis=0)
        sgb_var  = abund_q.var(axis=0, ddof=1)

        # Only keep SGBs detected in at least 1 sample
        detected  = sgb_mean > 0
        sgb_mean  = sgb_mean[detected]
        sgb_var   = sgb_var[detected]
        cat_series = pd.Series([_cat(s) for s in sgb_mean.index], index=sgb_mean.index)

        # ── Scatter per category ─────────────────────────────────────────
        for cat in ["Other", "VANISH", "BloSSUM"]:   # draw order: Other first
            mask = cat_series == cat
            if not mask.any():
                continue
            style = SGB_CAT_STYLE[cat].copy()
            ax.scatter(
                sgb_mean[mask], sgb_var[mask],
                label=f"{cat}  (n={mask.sum():,})",
                **style,
            )

        # ── Reference lines ──────────────────────────────────────────────
        xmin = float(sgb_mean.min()) if len(sgb_mean) else 1e-9
        xmax = float(sgb_mean.max()) if len(sgb_mean) else 1.0
        if xmin > 0 and xmax > xmin:
            xs = np.logspace(np.log10(xmin), np.log10(xmax), 200)

            # Poisson: var = mean  (for reference — real relative abundances
            # are always overdispersed relative to this)
            ax.plot(xs, xs, "k--", linewidth=0.8, alpha=0.45,
                    label="var = mean (Poisson)")

            # Taylor's Power Law fit
            slope, intercept, r2 = _taylor_slope(
                sgb_mean.values, sgb_var.values
            )
            if not np.isnan(slope):
                ys_tpl = 10**intercept * xs**slope
                ax.plot(xs, ys_tpl, color="#555555", linewidth=1.0,
                        linestyle=":", alpha=0.75,
                        label=f"Taylor β={slope:.2f}  (r²={r2:.2f})")

        # ── Label top VANISH+BloSSUM SGBs ────────────────────────────────
        highlight = cat_series.isin(["VANISH", "BloSSUM"])
        if highlight.any():
            top_sgbs = sgb_mean[highlight].nlargest(n_label).index
            for sgb in top_sgbs:
                ax.annotate(
                    sgb,
                    xy=(sgb_mean[sgb], sgb_var[sgb]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=4.2, alpha=0.80, color="#222222",
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa",
                                    lw=0.4, alpha=0.5),
                )

        # Depth range for panel subtitle
        q_depths = meta.loc[samples_q, "_depth"]
        lo_mb, hi_mb = q_depths.min() / 1e6, q_depths.max() / 1e6
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            f"{QUARTILE_LABELS[q_idx]}\n"
            f"depth [{lo_mb:.0f}–{hi_mb:.0f}] Mb  ·  {len(samples_q):,} samples",
            fontsize=8.5,
        )
        ax.set_xlabel("Mean relative abundance", fontsize=8)
        if q_idx == 0:
            ax.set_ylabel("Variance of relative abundance", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.85,
                  markerscale=1.4, handlelength=1.5)

    plt.tight_layout()
    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 3 — Side-by-side comparison (depth vs breadth) in a 2×4 grid
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison_grid(
    abund_depth:    pd.DataFrame,
    abund_breadth:  pd.DataFrame,
    meta:           pd.DataFrame,
    depth_quartile: pd.Series,
    plot_type:      str,        # "rank_abundance" or "mean_variance"
    vanish_set:     set,
    blossum_set:    set,
    title:          str,
    out_path:       str,
    n_sample_lines: int = 100,
    n_label:        int = 8,
    rng:            np.random.Generator = None,
):
    """
    2 rows × 4 cols figure for direct depth-vs-breadth comparison.
    Row 0 = depth-based; Row 1 = breadth-based.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    def _cat(sgb):
        if sgb in vanish_set:  return "VANISH"
        if sgb in blossum_set: return "BloSSUM"
        return "Other"

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    row_labels = ["Depth-based", "Breadth-based (Poisson inv.)"]

    for row, abund in enumerate([abund_depth, abund_breadth]):
        for q_idx in range(4):
            ax       = axes[row, q_idx]
            q_label  = f"Q{q_idx + 1}"
            samp_q   = meta.index[depth_quartile == q_label]
            abund_q  = abund.loc[samp_q]
            q_depths = meta.loc[samp_q, "_depth"]
            lo_mb, hi_mb = q_depths.min() / 1e6, q_depths.max() / 1e6

            if plot_type == "rank_abundance":
                legend_handles = []
                for lifestyle in LIFESTYLE_ORDER:
                    ls_mask = meta.loc[samp_q, "MacroGroup"] == lifestyle
                    sids    = samp_q[ls_mask]
                    if len(sids) == 0:
                        continue
                    color = LIFESTYLE_COLORS[lifestyle]
                    sub   = abund_q.loc[sids]

                    draw_ids = rng.choice(sids, size=min(n_sample_lines, len(sids)), replace=False)
                    for sid in draw_ids:
                        ranks, vals = _rank_curve(sub.loc[sid].values)
                        if len(ranks):
                            ax.loglog(ranks, vals, color=color, alpha=0.04, linewidth=0.35)

                    mr, mv = _group_mean_curve(sub)
                    if len(mr):
                        ax.loglog(mr, mv, color=color, linewidth=1.8, alpha=0.9)
                    legend_handles.append(
                        Line2D([0], [0], color=color, linewidth=1.8,
                               label=f"{lifestyle}  (n={len(sids):,})")
                    )
                ax.set_xlabel("Rank", fontsize=7.5)
                if q_idx == 0:
                    ax.set_ylabel("Relative abundance", fontsize=7.5)
                if legend_handles:
                    ax.legend(handles=legend_handles, fontsize=6.5, loc="upper right")

            else:  # mean_variance
                sgb_mean  = abund_q.mean(axis=0)
                sgb_var   = abund_q.var(axis=0, ddof=1)
                detected  = sgb_mean > 0
                sgb_mean  = sgb_mean[detected]
                sgb_var   = sgb_var[detected]
                cat_series = pd.Series([_cat(s) for s in sgb_mean.index],
                                        index=sgb_mean.index)

                for cat in ["Other", "VANISH", "BloSSUM"]:
                    mask = cat_series == cat
                    if not mask.any():
                        continue
                    style = SGB_CAT_STYLE[cat].copy()
                    ax.scatter(sgb_mean[mask], sgb_var[mask],
                               label=f"{cat} (n={mask.sum():,})", **style)

                xmin, xmax = float(sgb_mean.min()), float(sgb_mean.max())
                if xmin > 0 and xmax > xmin:
                    xs = np.logspace(np.log10(xmin), np.log10(xmax), 200)
                    ax.plot(xs, xs, "k--", lw=0.8, alpha=0.4, label="Poisson")
                    slope, intercept, r2 = _taylor_slope(sgb_mean.values, sgb_var.values)
                    if not np.isnan(slope):
                        ax.plot(xs, 10**intercept * xs**slope,
                                color="#555", lw=1.0, ls=":", alpha=0.75,
                                label=f"β={slope:.2f}")

                # Label top VANISH+BloSSUM
                hl = cat_series.isin(["VANISH", "BloSSUM"])
                for sgb in sgb_mean[hl].nlargest(n_label).index:
                    ax.annotate(sgb, xy=(sgb_mean[sgb], sgb_var[sgb]),
                                xytext=(3, 2), textcoords="offset points",
                                fontsize=4.0, alpha=0.75, color="#333333")

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Mean rel. abundance", fontsize=7.5)
                if q_idx == 0:
                    ax.set_ylabel("Variance", fontsize=7.5)
                ax.legend(fontsize=6, loc="upper left", markerscale=1.3)

            ax.tick_params(labelsize=6.5)
            col_title = f"{QUARTILE_LABELS[q_idx]}\n[{lo_mb:.0f}–{hi_mb:.0f} Mb]"
            if row == 0:
                ax.set_title(col_title, fontsize=8.5)
            if q_idx == 0:
                ax.annotate(
                    row_labels[row],
                    xy=(-0.22, 0.5), xycoords="axes fraction",
                    fontsize=9, fontweight="bold", rotation=90,
                    va="center", ha="center",
                )

    plt.tight_layout()
    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 4 (new) — Mean-variance stratified by lifestyle  (2 × 4 grid)
# ══════════════════════════════════════════════════════════════════════════════

def plot_mean_variance_by_lifestyle(
    abundance:      pd.DataFrame,
    meta:           pd.DataFrame,
    depth_quartile: pd.Series,
    vanish_set:     set,
    blossum_set:    set,
    title:          str,
    out_path:       str,
    n_label:        int = 10,
):
    """
    2 rows (Industrial / Non-Industrial) × 4 cols (depth quartiles).

    Mean and variance are computed using only the samples from that specific
    lifestyle × quartile cell.  Taylor's Power Law is fitted independently
    per cell so slopes can be compared across lifestyles and depth strata.
    """
    def _cat(sgb):
        if sgb in vanish_set:  return "VANISH"
        if sgb in blossum_set: return "BloSSUM"
        return "Other"

    lifestyles = ["Industrial", "Non-Industrial"]
    fig, axes = plt.subplots(2, 4, figsize=(26, 11), squeeze=False)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    for row, lifestyle in enumerate(lifestyles):
        ls_mask = meta["MacroGroup"] == lifestyle

        for q_idx in range(4):
            ax      = axes[row, q_idx]
            q_label = f"Q{q_idx + 1}"
            # Intersection: this lifestyle AND this depth quartile
            cell_mask = ls_mask & (depth_quartile == q_label)
            samp_cell = meta.index[cell_mask]
            abund_cell = abundance.loc[samp_cell]

            if len(samp_cell) < 5:
                ax.text(0.5, 0.5, f"n={len(samp_cell)}\n(too few)",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color="gray")
                ax.set_title(f"{QUARTILE_LABELS[q_idx]}" if row == 0 else "")
                continue

            sgb_mean  = abund_cell.mean(axis=0)
            sgb_var   = abund_cell.var(axis=0, ddof=1)
            detected  = sgb_mean > 0
            sgb_mean  = sgb_mean[detected]
            sgb_var   = sgb_var[detected]
            cat_series = pd.Series([_cat(s) for s in sgb_mean.index],
                                    index=sgb_mean.index)

            # ── Scatter ────────────────────────────────────────────────────
            for cat in ["Other", "VANISH", "BloSSUM"]:
                mask = cat_series == cat
                if not mask.any():
                    continue
                style = SGB_CAT_STYLE[cat].copy()
                ax.scatter(sgb_mean[mask], sgb_var[mask],
                           label=f"{cat} (n={mask.sum():,})", **style)

            # ── Reference lines ────────────────────────────────────────────
            xmin, xmax = float(sgb_mean.min()), float(sgb_mean.max())
            if xmin > 0 and xmax > xmin:
                xs = np.logspace(np.log10(xmin), np.log10(xmax), 200)
                ax.plot(xs, xs, "k--", lw=0.8, alpha=0.4, label="Poisson")
                slope, intercept, r2 = _taylor_slope(sgb_mean.values, sgb_var.values)
                if not np.isnan(slope):
                    ax.plot(xs, 10**intercept * xs**slope,
                            color="#555555", lw=1.1, ls=":", alpha=0.8,
                            label=f"Taylor β={slope:.2f}  r²={r2:.2f}")

            # ── Label top VANISH/BloSSUM ───────────────────────────────────
            hl = cat_series.isin(["VANISH", "BloSSUM"])
            for sgb in sgb_mean[hl].nlargest(n_label).index:
                ax.annotate(
                    sgb, xy=(sgb_mean[sgb], sgb_var[sgb]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=4.0, alpha=0.78, color="#111111",
                    arrowprops=dict(arrowstyle="-", color="#bbbbbb",
                                    lw=0.4, alpha=0.5),
                )

            # ── Axes ───────────────────────────────────────────────────────
            q_depths  = meta.loc[samp_cell, "_depth"]
            lo, hi    = q_depths.min() / 1e6, q_depths.max() / 1e6
            ax.set_xscale("log")
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(
                    f"{QUARTILE_LABELS[q_idx]}\n[{lo:.0f}–{hi:.0f} Mb]",
                    fontsize=8.5,
                )
            ax.set_xlabel("Mean rel. abundance", fontsize=8)
            if q_idx == 0:
                ax.set_ylabel("Variance", fontsize=8)
                # Row label on the left
                ax.annotate(
                    lifestyle,
                    xy=(-0.28, 0.5), xycoords="axes fraction",
                    fontsize=10, fontweight="bold", rotation=90,
                    va="center", ha="center",
                    color=LIFESTYLE_COLORS.get(lifestyle, "#333333"),
                )
            ax.tick_params(labelsize=7)
            ax.text(0.02, 0.02, f"n={len(samp_cell):,} samples",
                    transform=ax.transAxes, fontsize=6.5, color="#555555",
                    va="bottom")
            ax.legend(fontsize=6, loc="upper left", framealpha=0.85,
                      markerscale=1.3, handlelength=1.2)

    plt.tight_layout()
    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 5 (new) — Per-SGB maximum abundance, ranked, by lifestyle
# ══════════════════════════════════════════════════════════════════════════════

def plot_max_abundance(
    abund_depth:   pd.DataFrame,
    abund_breadth: pd.DataFrame,
    meta:          pd.DataFrame,
    vanish_set:    set,
    blossum_set:   set,
    title:         str,
    out_path:      str,
    n_label:       int = 25,
):
    """
    2 rows (depth / breadth-based) x 2 cols (Industrial / Non-Industrial).

    Main panel: log-log scatter of rank (starting at 1) vs max relative
    abundance, with a large linear-scale inset (2/3 the panel size) in the
    bottom-left that shows the full distribution on untransformed axes.
    Both views use the same VANISH/BloSSUM colouring.
    """
    def _cat(sgb):
        if sgb in vanish_set:  return "VANISH"
        if sgb in blossum_set: return "BloSSUM"
        return "Other"

    lifestyles = ["Industrial", "Non-Industrial"]
    methods    = [("Depth-based",             abund_depth),
                  ("Breadth-based (Poisson)", abund_breadth)]

    # Each main cell is drawn on its own Axes so we have full control over
    # the inset fraction.  We use gridspec for the 2x2 layout.
    fig = plt.figure(figsize=(22, 15))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    for row, (method_label, abund) in enumerate(methods):
        floor = 1e-4 if "Breadth" in method_label else 1e-5

        for col, lifestyle in enumerate(lifestyles):
            ax      = fig.add_subplot(gs[row, col])
            ls_mask = meta["MacroGroup"] == lifestyle
            samp_ls = meta.index[ls_mask]

            if len(samp_ls) == 0:
                ax.text(0.5, 0.5, "No samples", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            abund_ls = abund.loc[samp_ls]
            sgb_max  = abund_ls.max(axis=0)
            sgb_max  = sgb_max[sgb_max > 0].sort_values(ascending=False)
            ranks    = np.arange(1, len(sgb_max) + 1, dtype=float)
            cats     = pd.Series([_cat(s) for s in sgb_max.index],
                                  index=sgb_max.index)

            # ── Helper: draw scatter into any ax ──────────────────────────
            def _draw_scatter(target_ax, rs, vals, cs, s_scale=1.0,
                              log_x=True, log_y=True):
                for cat in ["Other", "VANISH", "BloSSUM"]:
                    mask = cs == cat
                    if not mask.any():
                        continue
                    style = SGB_CAT_STYLE[cat].copy()
                    style["s"] = style["s"] * s_scale
                    target_ax.scatter(
                        rs[mask.values], vals.values[mask.values],
                        label=f"{cat} (n={mask.sum():,})",
                        **style,
                    )
                target_ax.axhline(
                    floor, color="#e08000", lw=0.9, ls="--", alpha=0.7,
                    label=f"Floor ~{floor:.0e}",
                )
                if log_x:
                    target_ax.set_xscale("log")
                    target_ax.set_xlim(left=1)          # rank starts at 1
                if log_y:
                    target_ax.set_yscale("log")

            _draw_scatter(ax, ranks, sgb_max, cats, s_scale=1.0)

            # ── Annotate top VANISH+BloSSUM on main panel ─────────────────
            hl_mask = cats.isin(["VANISH", "BloSSUM"])
            for rp, (sgb, val) in zip(ranks[hl_mask.values],
                                       sgb_max[hl_mask].items()):
                if rp > n_label * 20:
                    break
                ax.annotate(
                    sgb, xy=(rp, val),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=4.2, alpha=0.80, color="#111111", va="center",
                    arrowprops=dict(arrowstyle="-", color="#cccccc",
                                    lw=0.4, alpha=0.5),
                )

            # ── Axes labels & decoration ───────────────────────────────────
            for spine in ax.spines.values():
                spine.set_edgecolor(LIFESTYLE_COLORS.get(lifestyle, "#333333"))
                spine.set_linewidth(1.6)

            if row == 0:
                ax.set_title(
                    f"{lifestyle}  (n={len(samp_ls):,} samples)",
                    fontsize=10, fontweight="bold",
                    color=LIFESTYLE_COLORS.get(lifestyle, "#333333"),
                )
            ax.set_xlabel("SGB rank (descending max abundance)", fontsize=8.5)
            if col == 0:
                ax.set_ylabel(
                    f"Max rel. abundance — {method_label}", fontsize=8.5)
                ax.annotate(
                    method_label,
                    xy=(-0.20, 0.5), xycoords="axes fraction",
                    fontsize=10, fontweight="bold", rotation=90,
                    va="center", ha="center", color="#333333",
                )
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=7, loc="upper right", framealpha=0.88,
                      markerscale=1.4, handlelength=1.3)

            # ── Large inset — bottom-left, 2/3 size ───────────────────────
            # Offset from corner so axis tick labels don't collide.
            # x-axis: linear rank (so the heavy tail is visible)
            # y-axis: log abundance (same scale as the main panel)
            axins = ax.inset_axes([0.11, 0.09, 0.60, 0.57])
            _draw_scatter(axins, ranks, sgb_max, cats,
                          s_scale=0.55, log_x=False, log_y=True)
            axins.set_xlim(left=1, right=len(sgb_max) + 1)
            axins.set_xlabel("Rank (linear)", fontsize=6, labelpad=2)
            axins.set_ylabel("Max abundance\n(log)", fontsize=6, labelpad=2)
            axins.tick_params(labelsize=5.5)
            if axins.get_legend():
                axins.get_legend().remove()
            # Subtle box border
            for spine in axins.spines.values():
                spine.set_linewidth(0.6)
                spine.set_edgecolor("#888888")
            axins.set_facecolor("#fafafa")

    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 6 (new) — Does abundance scale spuriously with sequencing depth?
# ══════════════════════════════════════════════════════════════════════════════

def _vectorized_spearman(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Spearman ρ between every column of X (n_samples × n_sgbs) and
    vector y (n_samples,).  Fully vectorised — ranks all columns at once.

    Returns shape (n_sgbs,).  Columns with zero variance get NaN.
    """
    from scipy.stats import rankdata
    n = X.shape[0]

    # Rank y once
    y_r = rankdata(y).astype(np.float64)
    y_r -= y_r.mean()
    y_ss = np.sqrt((y_r ** 2).sum())

    # Rank every column of X (apply_along_axis iterates columns)
    X_r = np.apply_along_axis(rankdata, 0, X).astype(np.float64)
    X_r -= X_r.mean(axis=0)
    X_ss = np.sqrt((X_r ** 2).sum(axis=0))

    num  = (y_r[:, None] * X_r).sum(axis=0)
    denom = y_ss * X_ss
    return np.where(denom > 0, num / denom, np.nan)


def _exclude_top_n_renorm(abund: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """
    Within each sample, exclude the top-n SGBs by *global* mean abundance
    (computed across all samples supplied), then renormalise the remainder to
    sum to 1.  Zeros remain zero.

    This strips the dominant taxa so that compositional dilution effects on
    the low-abundance tail can be assessed without being confounded by the
    relative sizes of the dominant species.
    """
    global_mean = abund.mean(axis=0)
    top_sgbs    = global_mean.nlargest(n).index
    abund_rest  = abund.drop(columns=top_sgbs, errors="ignore")
    row_sums    = abund_rest.sum(axis=1).replace(0, np.nan)
    return abund_rest.div(row_sums, axis=0).fillna(0.0)


def compute_depth_correlations(
    abund:          pd.DataFrame,
    log_depth:      np.ndarray,
    min_detect_frac: float = 0.05,
) -> pd.DataFrame:
    """
    For every SGB detected in at least ``min_detect_frac`` of samples,
    compute Spearman ρ between per-sample abundance and log10(total depth).

    Returns a DataFrame indexed by SGB with columns:
        mean_abund, detect_frac, spearman_rho
    """
    n = len(abund)
    detect_frac = (abund > 0).mean(axis=0)
    keep        = detect_frac >= min_detect_frac
    abund_keep  = abund.loc[:, keep]

    if abund_keep.shape[1] == 0:
        return pd.DataFrame(columns=["mean_abund", "detect_frac", "spearman_rho"])

    rho = _vectorized_spearman(abund_keep.values, log_depth)

    return pd.DataFrame({
        "mean_abund":   abund_keep.mean(axis=0).values,
        "detect_frac":  detect_frac[keep].values,
        "spearman_rho": rho,
    }, index=abund_keep.columns)


def _binned_trend(x: np.ndarray, y: np.ndarray,
                  n_bins: int = 20, pct: tuple = (25, 75)):
    """
    Log-space bins of x, return bin centres, medians, and percentile bands.
    Bins with fewer than 5 observations are dropped.
    """
    log_x   = np.log10(np.where(x > 0, x, np.nan))
    edges   = np.linspace(np.nanmin(log_x), np.nanmax(log_x), n_bins + 1)
    centres, medians, lo_pct, hi_pct = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (log_x >= lo) & (log_x < hi) & np.isfinite(y)
        if mask.sum() < 5:
            continue
        centres.append(10 ** ((lo + hi) / 2))
        medians.append(np.median(y[mask]))
        lo_pct.append(np.percentile(y[mask], pct[0]))
        hi_pct.append(np.percentile(y[mask], pct[1]))
    return (np.array(centres), np.array(medians),
            np.array(lo_pct), np.array(hi_pct))


def plot_depth_abundance_correlation(
    abund_depth:    pd.DataFrame,
    abund_breadth:  pd.DataFrame,
    meta:           pd.DataFrame,
    vanish_set:     set,
    blossum_set:    set,
    title:          str,
    out_path:       str,
    n_top_exclude:  int = 100,
    min_detect:     float = 0.05,
    n_label:        int = 10,
):
    """
    2 rows (Industrial / Non-Industrial) x 2 cols (depth-based / breadth-based).

    Each panel:
      x-axis  : mean relative abundance of the SGB (log scale)
      y-axis  : Spearman ρ between per-sample [SGB abundance] and log10(total depth)
      scatter : one point per SGB, coloured VANISH/BloSSUM/Other
      trend   : binned median rho (solid line) +/- 25–75th percentile band
      reference: dashed line at rho = 0

    Depth-based column  : all SGBs, standard relative abundance.
    Breadth-based column: top-100 SGBs excluded per lifestyle then renormalised,
                          which isolates the low-abundance tail from compositional
                          dilution by the dominant taxa.

    Interpretation:
      rho > 0  →  SGB appears more abundant in deeper samples; consistent with
                  spurious cross-mapping accumulating with sequencing effort.
      rho ~ 0  →  abundance estimate is depth-independent; likely genuine signal.
      rho < 0  →  SGB is relatively more abundant in shallower samples (dilution
                  or saturation effect).
    """
    def _cat(sgb):
        if sgb in vanish_set:  return "VANISH"
        if sgb in blossum_set: return "BloSSUM"
        return "Other"

    lifestyles   = ["Industrial", "Non-Industrial"]
    col_labels   = [
        "Depth-based\n(all SGBs)",
        f"Breadth-based (Poisson inv.)\n(top-{n_top_exclude} excluded, renorm.)",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 14), squeeze=False)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    for row, lifestyle in enumerate(lifestyles):
        ls_mask  = meta["MacroGroup"] == lifestyle
        samp_ls  = meta.index[ls_mask]
        log_dep  = np.log10(meta.loc[samp_ls, "_depth"].values)

        if len(samp_ls) < 20:
            for col in range(2):
                axes[row, col].text(0.5, 0.5, f"Too few samples (n={len(samp_ls)})",
                                    ha="center", va="center",
                                    transform=axes[row, col].transAxes)
            continue

        # Prepare the two abundance matrices for this lifestyle
        ab_depth   = abund_depth.loc[samp_ls]
        ab_breadth_full = abund_breadth.loc[samp_ls]
        ab_breadth_excl = _exclude_top_n_renorm(ab_breadth_full, n=n_top_exclude)

        col_data = [(ab_depth, "depth"), (ab_breadth_excl, "breadth")]

        for col, (ab, mode) in enumerate(col_data):
            ax = axes[row, col]

            print(f"    Computing Spearman rho: {lifestyle} / {mode} "
                  f"({ab.shape[1]} SGBs, {len(samp_ls)} samples) ...", flush=True)
            corr_df = compute_depth_correlations(
                ab, log_dep, min_detect_frac=min_detect)

            if len(corr_df) == 0:
                ax.text(0.5, 0.5, "No SGBs passed detection filter",
                        ha="center", va="center", transform=ax.transAxes)
                continue

            corr_df["cat"] = [_cat(s) for s in corr_df.index]

            # ── Scatter ────────────────────────────────────────────────────
            for cat in ["Other", "VANISH", "BloSSUM"]:
                sub = corr_df[corr_df["cat"] == cat]
                if sub.empty:
                    continue
                style = SGB_CAT_STYLE[cat].copy()
                ax.scatter(sub["mean_abund"], sub["spearman_rho"],
                           label=f"{cat} (n={len(sub):,})", **style)

            # ── Binned trend ───────────────────────────────────────────────
            valid = corr_df["spearman_rho"].notna()
            bx, bmed, blo, bhi = _binned_trend(
                corr_df.loc[valid, "mean_abund"].values,
                corr_df.loc[valid, "spearman_rho"].values,
            )
            if len(bx) > 1:
                ax.plot(bx, bmed, color="#222222", lw=2.0, zorder=6,
                        label="Binned median")
                ax.fill_between(bx, blo, bhi, color="#888888",
                                alpha=0.20, zorder=5, label="IQR band")

            # ── rho = 0 reference ─────────────────────────────────────────
            ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)

            # ── Label top VANISH/BloSSUM by rho value (most positive first) ─
            hl = corr_df[corr_df["cat"].isin(["VANISH", "BloSSUM"])]
            for sgb in hl["spearman_rho"].nlargest(n_label).index:
                row_s = corr_df.loc[sgb]
                ax.annotate(
                    sgb,
                    xy=(row_s["mean_abund"], row_s["spearman_rho"]),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=4.2, alpha=0.80, color="#111111",
                    arrowprops=dict(arrowstyle="-", color="#bbbbbb",
                                    lw=0.4, alpha=0.5),
                )

            # ── rho histogram inset (upper-right) ─────────────────────────
            axh = ax.inset_axes([0.68, 0.60, 0.30, 0.38])
            rho_vals = corr_df["spearman_rho"].dropna().values
            axh.hist(rho_vals, bins=40, color="#aaaaaa", edgecolor="none",
                     alpha=0.7, density=True)
            for cat, col_c in [("VANISH", PALETTE["vermillion"]), ("BloSSUM", PALETTE["blue"])]:
                sub_rho = corr_df.loc[corr_df["cat"] == cat, "spearman_rho"].dropna()
                if len(sub_rho):
                    axh.hist(sub_rho, bins=20, color=col_c,
                             alpha=0.6, density=True, edgecolor="none")
            axh.axvline(0, color="k", lw=0.7, ls="--")
            axh.axvline(np.nanmedian(rho_vals), color="#333333",
                        lw=1.0, ls=":", label=f"med={np.nanmedian(rho_vals):.2f}")
            axh.set_xlabel("ρ", fontsize=5.5)
            axh.set_title("ρ distribution", fontsize=5.5, pad=2)
            axh.tick_params(labelsize=5)
            for sp in axh.spines.values():
                sp.set_linewidth(0.5)

            # ── Axes ───────────────────────────────────────────────────────
            ax.set_xscale("log")
            ax.set_ylim(-1.05, 1.05)
            if row == 0:
                ax.set_title(col_labels[col], fontsize=9.5)
            ax.set_xlabel("Mean relative abundance (log scale)", fontsize=8.5)
            if col == 0:
                ax.set_ylabel(
                    f"Spearman ρ  [abundance vs log depth]\n{lifestyle}",
                    fontsize=8.5,
                    color=LIFESTYLE_COLORS.get(lifestyle, "#333333"),
                )
                ax.annotate(
                    lifestyle,
                    xy=(-0.20, 0.5), xycoords="axes fraction",
                    fontsize=10, fontweight="bold", rotation=90,
                    va="center", ha="center",
                    color=LIFESTYLE_COLORS.get(lifestyle, "#333333"),
                )
            ax.tick_params(labelsize=8)

            # Summary text
            n_pos_sig = (corr_df["spearman_rho"] > 0.2).sum()
            n_neg_sig = (corr_df["spearman_rho"] < -0.2).sum()
            ax.text(0.02, 0.97,
                    f"n={len(corr_df):,} SGBs\n"
                    f"rho>0.2: {n_pos_sig}  rho<-0.2: {n_neg_sig}\n"
                    f"med rho: {np.nanmedian(corr_df['spearman_rho'].values):.3f}",
                    transform=ax.transAxes, fontsize=6.5, va="top",
                    color="#333333",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#cccccc", alpha=0.85))
            ax.legend(fontsize=6.5, loc="lower right", framealpha=0.85,
                      markerscale=1.3, handlelength=1.2)

    plt.tight_layout()
    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")



# ══════════════════════════════════════════════════════════════════════════════
#  Figure 11 — Extended depth-correlation: pooled vs within-study
# ══════════════════════════════════════════════════════════════════════════════

def _within_study_correlations(abund, meta_ls, log_depth,
                                min_study_n=15, min_detect_frac=0.05):
    """
    Per-study Spearman rho(abundance, log_depth), then Fisher-z weighted mean
    across studies.  Removes study-level confounding.
    Returns DataFrame with columns: mean_abund, detect_frac, within_study_rho

    Study column resolved via the canonical _STUDY_COLS priority list so this
    function uses the same column as fit_sgb_asymptote and plot_sgb_accumulation.
    Within-study rho values are combined on a Fisher-z scale (not raw rho) to
    account for the bounded, skewed sampling distribution of Spearman rho.
    """
    _STUDY_COLS = ("StudyID", "Study", "cohort", "_study")
    _study_col_name = next((c for c in _STUDY_COLS if c in meta_ls.columns), None)
    if _study_col_name is None:
        # No study column at all — return empty
        return pd.DataFrame(columns=["mean_abund", "detect_frac", "within_study_rho"])
    study_col = meta_ls[_study_col_name].fillna("__unknown__")
    # Fisher-z accumulators: combine rho on the atanh scale, back-transform at end.
    # This handles the bounded, skewed sampling distribution of Spearman rho
    # near +/-1 correctly; combining raw rho with sqrt(n) weights is biased there.
    # Weight for each study = sqrt(n_study - 3), the approximate SE of atanh(rho).
    sgb_z_num, sgb_z_den = {}, {}

    for study in study_col.unique():
        s_mask  = (study_col == study).values
        n_s     = int(s_mask.sum())
        if n_s < min_study_n:
            continue
        s_abund   = abund.iloc[s_mask]
        s_logdep  = log_depth[s_mask]
        detect    = (s_abund > 0).mean(axis=0) >= min_detect_frac
        s_filt    = s_abund.loc[:, detect]
        if s_filt.shape[1] == 0:
            continue
        rho_arr  = _vectorized_spearman(s_filt.values, s_logdep)
        rho_clip = np.clip(rho_arr, -1 + 1e-7, 1 - 1e-7)
        z_arr    = np.arctanh(rho_clip)
        w = float(np.sqrt(max(n_s - 3, 1)))
        for sgb, z in zip(s_filt.columns, z_arr):
            if not np.isnan(z):
                sgb_z_num[sgb] = sgb_z_num.get(sgb, 0.0) + w * z
                sgb_z_den[sgb] = sgb_z_den.get(sgb, 0.0) + w

    gm = abund.mean(axis=0)
    gd = (abund > 0).mean(axis=0)
    records = [{"sgb": s, "mean_abund": float(gm.get(s, np.nan)),
                "detect_frac": float(gd.get(s, np.nan)),
                "within_study_rho": float(np.tanh(sgb_z_num[s] / sgb_z_den[s]))}
               for s in sgb_z_den if sgb_z_den[s] > 0]
    return (pd.DataFrame(records).set_index("sgb") if records
            else pd.DataFrame(columns=["mean_abund","detect_frac","within_study_rho"]))


def _draw_correlation_panel(ax, df, rho_col, cat_series, vanish_set, blossum_set,
                             show_scatter=True, n_bins=20):
    """Per-category binned median lines + IQR bands + optional scatter."""
    valid = df[rho_col].notna()
    x_all = df.loc[valid, "mean_abund"].values
    y_all = df.loc[valid, rho_col].values
    c_all = cat_series.loc[df.index[valid]].values

    if show_scatter:
        for cat_v, col_v, mrkr, zo, sz in [
            ("Other",   "#cccccc", ".",  1,  5),
            ("VANISH",  PALETTE["vermillion"], "o",  4, 18),
            ("BloSSUM", PALETTE["blue"], "s",  4, 18),
        ]:
            mk = c_all == cat_v
            if mk.any():
                ax.scatter(x_all[mk], y_all[mk], c=col_v, marker=mrkr,
                           s=sz, alpha=0.28, zorder=zo, linewidths=0)

    for cat_v, col_v, ls, lw in [
        ("Other",   "#888888", "-",  1.2),
        ("VANISH",  PALETTE["vermillion"], "-",  2.2),
        ("BloSSUM", PALETTE["blue"], "--", 2.2),
    ]:
        mk = c_all == cat_v
        if mk.sum() < 10:
            continue
        bx, bmed, blo, bhi = _binned_trend(x_all[mk], y_all[mk], n_bins=n_bins)
        if len(bx) < 2:
            continue
        ax.plot(bx, bmed, color=col_v, lw=lw, ls=ls, zorder=5,
                label=f"{cat_v} median")
        ax.fill_between(bx, blo, bhi, color=col_v, alpha=0.12, zorder=4)

    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.55)
    ax.set_xscale("log")


def plot_depth_correlation_extended(
    abund_depth, abund_breadth, meta, vanish_set, blossum_set,
    title, out_path, n_top_exclude=100, min_detect=0.05, min_study_n=15,
):
    """
    2 rows (lifestyle) x 4 cols figure.
    Cols: pooled-depth | within-study-depth | pooled-breadth | within-study-breadth

    Discriminates three hypotheses:
      H1 Spurious mapping  : rho persists within studies AND rises at low abundance
      H2 Study confounding : rho drops ~0 within studies
      H3 Compositional     : rho flat, equal across VANISH/BloSSUM/Other
    """
    from scipy.stats import mannwhitneyu

    def _cat(sgb):
        if sgb in vanish_set:  return "VANISH"
        if sgb in blossum_set: return "BloSSUM"
        return "Other"

    lifestyles = ["Industrial", "Non-Industrial"]
    col_labels = [
        "Pooled rho\ndepth-based (all SGBs)",
        "Within-study rho\ndepth-based",
        "Pooled rho\nbreadth (top-100 excl.)",
        "Within-study rho\nbreadth (top-100 excl.)",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(28, 12), squeeze=False)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    for row, lifestyle in enumerate(lifestyles):
        ls_mask = meta["MacroGroup"] == lifestyle
        samp_ls = meta.index[ls_mask]
        meta_ls = meta.loc[samp_ls]
        log_dep = np.log10(meta_ls["_depth"].values)

        if len(samp_ls) < 20:
            for c in range(4):
                axes[row, c].text(0.5, 0.5, f"n={len(samp_ls)}", ha="center",
                                  va="center", transform=axes[row, c].transAxes)
            continue

        ab_dep     = abund_depth.loc[samp_ls]
        ab_br      = _exclude_top_n_renorm(abund_breadth.loc[samp_ls], n=n_top_exclude)

        print(f"    {lifestyle}: pooled rho ...", flush=True)
        pool_dep = compute_depth_correlations(ab_dep, log_dep, min_detect)
        pool_br  = compute_depth_correlations(ab_br,  log_dep, min_detect)

        print(f"    {lifestyle}: within-study rho ...", flush=True)
        ws_dep  = _within_study_correlations(ab_dep, meta_ls, log_dep, min_study_n, min_detect)
        ws_br   = _within_study_correlations(ab_br,  meta_ls, log_dep, min_study_n, min_detect)

        datasets = [
            (pool_dep, "spearman_rho"),
            (ws_dep,   "within_study_rho"),
            (pool_br,  "spearman_rho"),
            (ws_br,    "within_study_rho"),
        ]

        for col, (df_raw, rho_col) in enumerate(datasets):
            ax  = axes[row, col]
            df  = df_raw.copy()
            df["cat"] = [_cat(s) for s in df.index]
            cat = df["cat"]

            _draw_correlation_panel(ax, df, rho_col, cat,
                                    vanish_set, blossum_set, show_scatter=True)

            # Summary + Mann-Whitney
            lines = []
            for cn in ["Other", "VANISH", "BloSSUM"]:
                v = df.loc[df["cat"] == cn, rho_col].dropna()
                if len(v):
                    lines.append(f"{cn}: med={np.median(v):.3f} n={len(v)}")
            v_rho = df.loc[df["cat"] == "VANISH",  rho_col].dropna()
            o_rho = df.loc[df["cat"] == "Other",   rho_col].dropna()
            if len(v_rho) >= 5 and len(o_rho) >= 5:
                _, pval = mannwhitneyu(v_rho, o_rho, alternative="two-sided")
                lines.append(f"MW VANISH vs Other p={pval:.2g}")
            ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes,
                    fontsize=6, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.88))

            ax.set_ylim(-0.65, 0.65)
            if row == 0:
                ax.set_title(col_labels[col].replace("\\n", "\n"), fontsize=9)
            ax.set_xlabel("Mean relative abundance", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"Spearman rho\n{lifestyle}", fontsize=8.5,
                              color=LIFESTYLE_COLORS.get(lifestyle, "#333"))
                ax.annotate(lifestyle, xy=(-0.18, 0.5), xycoords="axes fraction",
                            fontsize=10, fontweight="bold", rotation=90,
                            va="center", ha="center",
                            color=LIFESTYLE_COLORS.get(lifestyle, "#333"))
            else:
                ax.set_ylabel("")

            if row == 0 and col == 3:
                handles = [
                    Line2D([0],[0], color=PALETTE["vermillion"], lw=2,   label="VANISH median"),
                    Line2D([0],[0], color=PALETTE["blue"],       lw=2, ls="--", label="BloSSUM median"),
                    Line2D([0],[0], color="#888888", lw=1.2,  label="Other median"),
                    plt.Rectangle((0,0),1,1, fc=PALETTE["vermillion"], alpha=0.15, label="VANISH IQR"),
                    plt.Rectangle((0,0),1,1, fc=PALETTE["blue"], alpha=0.15, label="BloSSUM IQR"),
                ]
                ax.legend(handles=handles, fontsize=7.5, loc="upper right", framealpha=0.88)
            ax.tick_params(labelsize=7)

    # Interpretive strip between rows
    fig.text(0.5, 0.505,
             "H1 (spurious mapping): within-study rho > 0 AND rises at low abundance.  "
             "H2 (study confound): within-study rho drops to ~0.  "
             "H3 (compositional dilution): flat rho equal across categories.",
             ha="center", va="center", fontsize=8, color="#444",
             style="italic",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#ccc"))

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Diagnostics: depth-quartile summary table
# ══════════════════════════════════════════════════════════════════════════════

def print_quartile_summary(meta: pd.DataFrame, depth_quartile: pd.Series):
    print(f"\n  {'Quartile':<8} {'N':>6}  {'Depth min':>12}  "
          f"{'Depth med':>12}  {'Depth max':>12}  "
          f"{'Industrial':>12}  {'Non-Ind.':>10}")
    print("  " + "─" * 82)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sids  = meta.index[depth_quartile == q]
        dvals = meta.loc[sids, "_depth"]
        n_ind = (meta.loc[sids, "MacroGroup"] == "Industrial").sum()
        n_ni  = (meta.loc[sids, "MacroGroup"] == "Non-Industrial").sum()
        print(
            f"  {q:<8} {len(sids):>6}  "
            f"{dvals.min()/1e6:>10.1f}Mb  "
            f"{dvals.median()/1e6:>10.1f}Mb  "
            f"{dvals.max()/1e6:>10.1f}Mb  "
            f"{n_ind:>12,}  {n_ni:>10,}"
        )
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 12 — Low-MAPQ positive control for spurious-mapping rho test
# ══════════════════════════════════════════════════════════════════════════════

def load_lowmapq_abundance(filepath: str, meta: pd.DataFrame,
                            lengths: pd.Series) -> pd.DataFrame:
    """
    Load the low-MAPQ multi-mapper depth matrix and return relative abundance
    (depth × genome_length, normalised per sample) aligned to meta.index.

    Reads with MAPQ < 30 couldn't be uniquely placed; their apparent signal
    is definitionally cross-mapping contamination.  This matrix therefore
    serves as a positive control: if rho(low-MAPQ abundance, log_depth) >> rho
    from the high-MAPQ matrix, the test is sensitive to spurious mapping.
    """
    print(f"  Loading low-MAPQ matrix: {filepath}")
    df = pd.read_csv(filepath, index_col=0, low_memory=False)
    df.index = df.index.astype(str).str.strip()

    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    if mean_cols:
        # New-format matrices: columns are SGB_mean / SGB_std pairs
        df = df[mean_cols].copy()
        df.columns = [c.rsplit("_mean", 1)[0] for c in mean_cols]
        print(f"    Column format: *_mean suffix ({len(mean_cols)} cols)")
    else:
        # Old-format matrices: columns are bare SGB names, keep only numeric cols
        numeric_cols = [c for c in df.columns
                        if pd.api.types.is_numeric_dtype(df[c])]
        df = df[numeric_cols].copy()
        print(f"    Column format: bare SGB names, no suffix ({len(numeric_cols)} cols)")
    df = df.fillna(0.0)
    print(f"    Raw matrix: {df.shape[0]:,} samples x {df.shape[1]:,} SGBs")

    # Align samples to meta (some may be missing if the file is a subset)
    common_samp = [s for s in meta.index if s in df.index]
    n_miss = len(meta) - len(common_samp)
    if n_miss:
        print(f"    Note: {n_miss:,} meta samples absent from low-MAPQ matrix "
              f"(will be treated as all-zero)")
    df = df.reindex(meta.index, fill_value=0.0)

    # Align SGBs to lengths
    common_sgbs = [s for s in df.columns if s in lengths.index]
    df = df[common_sgbs]

    # Relative abundance: same depth × length normalisation as main pipeline
    raw   = df.mul(lengths[common_sgbs], axis=1)
    total = raw.sum(axis=1).replace(0, np.nan)
    abund = raw.div(total, axis=0).fillna(0.0)
    print(f"    Aligned abundance: {len(common_samp):,} samples x {len(common_sgbs):,} SGBs")
    return abund


def plot_lowmapq_positive_control(
    abund_high:    pd.DataFrame,   # high-MAPQ (regular) abundance
    abund_low:     pd.DataFrame,   # low-MAPQ abundance (positive control)
    meta:          pd.DataFrame,
    vanish_set:    set,
    blossum_set:   set,
    title:         str,
    out_path:      str,
    min_detect:    float = 0.05,
    min_study_n:   int   = 15,
):
    """
    2 rows (Industrial / Non-Industrial) x 3 cols figure.

    Col 1 — Pooled rho: high-MAPQ (solid) vs low-MAPQ (dashed) per category.
             If lines overlap, rho is not sensitive to multi-mapping.
             If low-MAPQ >> high-MAPQ, the test detects the spurious signal.

    Col 2 — Within-study rho: same overlay after study confound is removed.
             More discriminating: residual rho here is the cleanest estimate
             of depth-driven bias unexplained by between-study differences.

    Col 3 — rho distributions: box plots comparing high-MAPQ vs low-MAPQ
             for each category (VANISH / BloSSUM / Other), pooled and within-study.
             Lets you directly read off the magnitude of the test's sensitivity.
    """
    def _cat(sgb):
        if sgb in vanish_set:  return "VANISH"
        if sgb in blossum_set: return "BloSSUM"
        return "Other"

    CAT_COLORS = {"VANISH": PALETTE["vermillion"], "BloSSUM": PALETTE["blue"], "Other": "#888888"}
    lifestyles  = ["Industrial", "Non-Industrial"]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12), squeeze=False)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    col_titles = [
        "Pooled rho: high-MAPQ (solid) vs low-MAPQ (dashed)",
        "Within-study rho: high-MAPQ (solid) vs low-MAPQ (dashed)",
        "rho distributions by matrix and category",
    ]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=9.5)

    for row, lifestyle in enumerate(lifestyles):
        ls_mask  = meta["MacroGroup"] == lifestyle
        samp_ls  = meta.index[ls_mask]
        meta_ls  = meta.loc[samp_ls]
        log_dep  = np.log10(meta_ls["_depth"].values)

        if len(samp_ls) < 20:
            for c in range(3):
                axes[row, c].text(0.5, 0.5, f"n={len(samp_ls)}", ha="center",
                                  va="center", transform=axes[row, c].transAxes)
            continue

        ab_hi = abund_high.loc[samp_ls]
        ab_lo = abund_low.loc[samp_ls]

        print(f"    {lifestyle}: pooled rho (high + low MAPQ) ...", flush=True)
        pool_hi = compute_depth_correlations(ab_hi, log_dep, min_detect)
        pool_lo = compute_depth_correlations(ab_lo, log_dep, min_detect)

        print(f"    {lifestyle}: within-study rho (high + low MAPQ) ...", flush=True)
        ws_hi   = _within_study_correlations(ab_hi, meta_ls, log_dep, min_study_n, min_detect)
        ws_lo   = _within_study_correlations(ab_lo, meta_ls, log_dep, min_study_n, min_detect)

        def _add_cat(df):
            df = df.copy()
            df["cat"] = [_cat(s) for s in df.index]
            return df

        pool_hi = _add_cat(pool_hi);  pool_lo = _add_cat(pool_lo)
        ws_hi   = _add_cat(ws_hi);    ws_lo   = _add_cat(ws_lo)

        # ── Cols 0 & 1: overlay binned median lines ───────────────────────
        for col, (df_hi, df_lo, rho_col) in enumerate([
            (pool_hi, pool_lo, "spearman_rho"),
            (ws_hi,   ws_lo,   "within_study_rho"),
        ]):
            ax = axes[row, col]

            for label, df, ls in [("high-MAPQ", df_hi, "-"),
                                   ("low-MAPQ",  df_lo, "--")]:
                valid = df[rho_col].notna()
                xv = df.loc[valid, "mean_abund"].values
                yv = df.loc[valid, rho_col].values
                cv = df.loc[valid, "cat"].values

                for cat in ["Other", "VANISH", "BloSSUM"]:
                    mk = cv == cat
                    if mk.sum() < 10:
                        continue
                    bx, bmed, blo, bhi = _binned_trend(xv[mk], yv[mk])
                    if len(bx) < 2:
                        continue
                    color = CAT_COLORS[cat]
                    lw    = 1.5 if cat == "Other" else 2.5
                    alpha_band = 0.10 if ls == "-" else 0.06
                    ax.plot(bx, bmed, color=color, lw=lw, ls=ls, zorder=5,
                            label=f"{cat} {label}")
                    ax.fill_between(bx, blo, bhi, color=color,
                                    alpha=alpha_band, zorder=4)

            ax.axhline(0, color="k", lw=0.8, ls=":", alpha=0.55)
            ax.set_xscale("log")
            ax.set_ylim(-0.55, 0.55)
            ax.set_xlabel("Mean relative abundance", fontsize=8.5)
            if col == 0:
                ax.set_ylabel(
                    f"Spearman rho\n{lifestyle}", fontsize=8.5,
                    color=LIFESTYLE_COLORS.get(lifestyle, "#333"),
                )
                ax.annotate(lifestyle, xy=(-0.17, 0.5), xycoords="axes fraction",
                            fontsize=10, fontweight="bold", rotation=90,
                            va="center", ha="center",
                            color=LIFESTYLE_COLORS.get(lifestyle, "#333"))
            else:
                ax.set_ylabel("")
            ax.tick_params(labelsize=7.5)

            # Annotate overall medians
            lines_txt = []
            for lbl, df, rho_col_l in [("high", df_hi, rho_col),
                                        ("low",  df_lo, rho_col)]:
                med = np.nanmedian(df[rho_col_l].values)
                lines_txt.append(f"{lbl}-MAPQ med rho: {med:.3f}")
            ax.text(0.02, 0.98, "\n".join(lines_txt), transform=ax.transAxes,
                    fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="#ccc", alpha=0.88))

            if row == 0 and col == 1:
                # Legend: line styles
                handles = []
                for cat, col_c in CAT_COLORS.items():
                    handles += [
                        Line2D([0],[0], color=col_c, lw=2,   ls="-",  label=f"{cat} high-MAPQ"),
                        Line2D([0],[0], color=col_c, lw=2,   ls="--", label=f"{cat} low-MAPQ"),
                    ]
                ax.legend(handles=handles, fontsize=6.5, loc="lower right",
                          framealpha=0.88, ncol=2)

        # ── Col 2: box plots ──────────────────────────────────────────────
        ax = axes[row, 2]

        box_data, box_labels, box_colors, box_positions = [], [], [], []
        pos = 0
        tick_pos, tick_labels = [], []

        for cat in ["Other", "VANISH", "BloSSUM"]:
            color = CAT_COLORS[cat]
            group_centre = pos + 1.5
            tick_pos.append(group_centre)
            tick_labels.append(cat)

            for lbl, df_p, df_w in [
                ("pool_high", pool_hi, None),
                ("pool_low",  pool_lo, None),
                ("ws_high",   ws_hi,   None),
                ("ws_low",    ws_lo,   None),
            ]:
                # pick the right df and rho column
                if "pool" in lbl:
                    df_use = pool_hi if "high" in lbl else pool_lo
                    col_use = "spearman_rho"
                else:
                    df_use = ws_hi if "high" in lbl else ws_lo
                    col_use = "within_study_rho"

                vals = df_use.loc[df_use["cat"] == cat, col_use].dropna().values
                if len(vals) == 0:
                    vals = np.array([np.nan])
                box_data.append(vals)
                box_colors.append(color)
                box_positions.append(pos)
                box_labels.append(lbl)
                pos += 1
            pos += 1   # gap between categories

        bp = ax.boxplot(box_data, positions=box_positions, widths=0.7,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", lw=1.5))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        for whisker in bp["whiskers"]:
            whisker.set_linewidth(0.8)
        for cap in bp["caps"]:
            cap.set_linewidth(0.8)

        ax.axhline(0, color="k", lw=0.8, ls=":", alpha=0.55)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_ylabel("Spearman rho", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.set_ylim(-0.6, 0.6)

        # Small legend for the 4 sub-boxes within each category
        from matplotlib.patches import Patch
        sub_legend = [
            Patch(fc="#aaaaaa", alpha=0.7, label="pool high-MAPQ"),
            Patch(fc="#aaaaaa", alpha=0.35, ec="#888", label="pool low-MAPQ"),
            Patch(fc="#aaaaaa", alpha=0.7, hatch="///", label="ws high-MAPQ"),
            Patch(fc="#aaaaaa", alpha=0.35, hatch="///", label="ws low-MAPQ"),
        ]
        # Apply hatching to within-study boxes
        for i, patch in enumerate(bp["boxes"]):
            if i % 4 >= 2:   # positions 2,3 within each group = within-study
                patch.set_hatch("///")
        if row == 0:
            ax.legend(handles=sub_legend, fontsize=6.5, loc="upper right",
                      framealpha=0.85)

    plt.tight_layout()
    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")





def _poisson_breadth_L(L_arr, d, G, c):
    """
    Expected breadth fraction for represented length L (bp).

        mu(L; d, G, c) = (L/G) * (1 - exp(-c * d * G / L))

    This is the generalised Poisson coverage formula: L/G is the fraction of
    the genome represented; the exponent -c*d*G/L converts mean depth over the
    full genome into mean depth over the represented region.

    Boundary behaviour:
      L → G :  mu → 1 − exp(−c·d)     (reduces to the full-genome Poisson formula)
      L → 0 :  mu → 0                  (L/G → 0 dominates; no coverage possible)
      Initial slope w.r.t. d: dmu/dd|_{d=0} = c · L/G  (proportional to L)

    Parameters
    ----------
    L_arr : ndarray (n_quad,) — represented lengths in bp
    d     : float — observed mean depth over the full genome
    G     : float — genome length in bp
    c     : float — coverage efficiency constant

    Returns
    -------
    mu    : ndarray (n_quad,) in [0, 1]
    """
    L_arr = np.maximum(np.asarray(L_arr, dtype=float), 1.0)
    G     = max(float(G), 1.0)
    return np.clip((L_arr / G) * (1.0 - np.exp(-c * d * G / L_arr)), 0.0, 1.0)


def _latent_L_posterior_vec(b_arr, d_arr, G, c, sigma_arr, tau,
                             n_quad=60, L_min_bp=500, pi_present=0.5):
    """
    Vectorised P(L >= tau | b_i, d_i, G) for n_samples, one SGB.

    Likelihood: Beta on raw breadth fraction
    -----------------------------------------
    Replaces the old Gaussian approximation with a Beta distribution, which is
    naturally bounded to [0, 1] and heteroskedastic:

        b_i | L, d, G, c  ~  Beta(alpha, beta)
        alpha = mu(L; d, G, c) * phi_i
        beta  = (1 - mu(L; d, G, c)) * phi_i
        phi_i = 1/(4*sigma_i^2)   [principled heteroskedastic concentration]

    Var(Beta) = mu*(1-mu)/phi = 4*mu*(1-mu)*sigma^2: maximum variance sigma^2
    at mu=0.5, decreasing to 0 at mu=0 and mu=1.  This is physically correct
    (nearly-zero or nearly-full breadth is tightly constrained) and prevents
    small-L quadrature points from spuriously fitting large-b observations
    via a nearly-Uniform Beta when phi=mu*(1-mu)/sigma^2 is near 0.

    The concentration phi is derived from the per-sample depth-stratified sigma:
    at the most variable point (mu=0.5), Var(Beta) = 1/(4*phi) = sigma^2.
    For sigma=0.015 this gives phi≈1111, representing ~1000 effective genome
    windows — calibrated to the resolution of typical short-read coverage.

    Key improvements over the Gaussian:
      • Bounded: impossible values (b outside [0,1]) get zero probability.
      • Heteroskedastic: Var(b) = mu*(1-mu)/phi shrinks near 0 and 1, correctly
        making low-breadth distinctions sharper.
      • Low-breadth calibration: b=0.01 at a depth where mu(10%G)=0.10 gets a
        Beta penalty >> the Gaussian 6-sigma penalty, correctly flagging the
        sample as inconsistent with L >= tau.

    Mixture prior on L
    ------------------
    L follows a two-component mixture:

        (1 - pi_present) : "absent/fragment" component — L fixed at L_min.
        pi_present        : "present" component — L ~ log-Uniform([L_min, G]).

    The absent component is given extra log-weight at the first quadrature point:
        log_extra = log( (1-pi_present)/pi_present * n_quad )

    Effective prior mass above tau:
        alpha_tau_eff = pi_present * P_log-uniform(L >= tau)

    Parameters
    ----------
    b_arr      : (n_s,) observed breadth fractions  (must be > 0; use effective_min_b)
    d_arr      : (n_s,) observed mean depths
    G          : float — genome length in bp
    c          : float — coverage efficiency
    sigma_arr  : (n_s,) per-sample sigma from depth-stratified noise model
    tau        : float — threshold in bp
    n_quad     : int   — number of log-spaced quadrature points
    L_min_bp   : float — lower integration limit in bp (also effective window size)
    pi_present : float — prior probability of genuine presence; default 0.5

    Returns
    -------
    p_above : (n_s,) float32 — P(L >= tau | b_i, d_i, G) in [0, 1]
    """
    from scipy.special import betaln as _betaln

    G_f    = max(float(G), 1.0)
    L_min  = float(np.clip(L_min_bp, 1, G_f * 0.999))
    L_max  = G_f
    pi     = float(np.clip(pi_present, 1e-9, 1.0))

    if L_min >= L_max:
        L_min = max(1.0, L_max * 0.001)

    # Log-spaced quadrature points in L-space: (n_quad,)
    log_L  = np.linspace(np.log(L_min), np.log(L_max), n_quad)
    L_vals = np.exp(log_L)

    # mu[k, i] = expected breadth for L_k and sample i: (n_quad, n_s)
    d_bc   = np.asarray(d_arr, dtype=np.float64)[np.newaxis, :]      # (1, n_s)
    L_bc   = L_vals[:, np.newaxis]                                    # (n_quad, 1)
    mu     = np.clip((L_bc / G_f) * (1.0 - np.exp(-c * d_bc * G_f / L_bc)),
                     0.0, 1.0)                                        # (n_quad, n_s)

    # ── Beta log-likelihood ──────────────────────────────────────────────────
    # phi = 1/(4*sigma^2) — the principled heteroskedastic concentration.
    #
    # With this phi, Var(Beta) = mu*(1-mu)/phi = 4*mu*(1-mu)*sigma^2.
    # This is NOT a "mu=0.5 approximation" — it is the correct physical model:
    #   • Maximum variance sigma^2 at mu=0.5 (most uncertain midpoint).
    #   • Variance → 0 as mu→0 or mu→1 (nearly-full or zero breadth is
    #     highly constrained; there is little room for variability).
    #
    # The alternative phi=mu*(1-mu)/sigma^2 forces Var(b)=sigma^2 everywhere,
    # which is physically wrong: it makes small-L quadrature points (mu≈0)
    # nearly Uniform over [0,1], so they can spuriously "fit" large b values
    # and dominate the posterior via the mixture-prior extra weight.
    # This causes non-monotone posteriors (b=0.80 gives LOWER P(L>=tau)
    # than b=0.60 because L_min steals weight through a nearly-Uniform Beta).
    #
    # phi=1/(4*sigma^2) is the correct regularisation: it keeps small-L
    # distributions tightly concentrated near mu=0, ensuring that only the
    # L values whose mu matches b carry significant weight.
    s_bc   = np.maximum(np.asarray(sigma_arr, dtype=np.float64), 1e-9)[np.newaxis, :]
    phi    = 1.0 / (4.0 * s_bc ** 2)                                 # (1, n_s)
    b_clip = np.clip(np.asarray(b_arr, dtype=np.float64)[np.newaxis, :],
                     1e-9, 1.0 - 1e-9)                               # (1, n_s)
    alpha  = np.maximum(mu * phi, 1e-9)                               # (n_quad, n_s)
    beta_  = np.maximum((1.0 - mu) * phi, 1e-9)                      # (n_quad, n_s)
    log_w  = ((alpha - 1.0) * np.log(b_clip) +
              (beta_  - 1.0) * np.log(1.0 - b_clip) -
              _betaln(alpha, beta_))                                  # (n_quad, n_s)

    # ── Mixture prior: extra weight at L_min (absent/fragment component) ──
    if pi < 1.0:
        log_extra = np.log((1.0 - pi) / pi * float(n_quad))
        log_w[0, :] += log_extra

    # Per-sample softmax normalization (stable)
    log_w -= log_w.max(axis=0, keepdims=True)
    w      = np.exp(log_w)                                            # (n_quad, n_s)
    w_sum  = w.sum(axis=0)                                            # (n_s,)

    # Posterior probability mass above tau
    above  = L_vals >= tau                                            # (n_quad,)
    if not above.any():
        return np.zeros(len(b_arr), dtype=np.float32)
    w_above = w[above, :].sum(axis=0)                                 # (n_s,)

    p = (w_above / np.maximum(w_sum, 1e-300)).astype(np.float32)
    return np.clip(p, 0.0, 1.0)


def _aggregate_posterior_logodds(p_arr, prior=0.5, study_labels=None):
    """
    Combine per-sample posteriors P(L >= tau | b_i, d_i) via log-odds pooling.

    This properly handles negative evidence: a sample with p_i < 0.5
    contributes a negative log Bayes factor, actively reducing the aggregate.
    The old 1-prod(1-p) formula always increased with more samples; this one
    can decrease when data are null-consistent.

    Within-study aggregation: AVERAGE the per-sample log Bayes factors.
      Rationale: within a study, samples are correlated (same batch, library
      prep, etc.), so we should not sum all their evidence.  Averaging gives
      one study-level evidence estimate.

    Across-study aggregation: SUM the study-level log Bayes factors.
      Rationale: independent studies are truly independent evidence, and
      their evidence can legitimately compound.

    Parameters
    ----------
    p_arr       : (n,) per-sample posteriors in (0, 1)
    prior       : float — prior P(L >= tau).  0.5 = uninformative.
    study_labels: (n,) array of study IDs, or None (all pooled)

    Returns
    -------
    p_agg : float in [0, 1]
    """
    p_clip = np.clip(np.asarray(p_arr, dtype=float), 1e-9, 1 - 1e-9)
    lo_arr = np.log(p_clip / (1.0 - p_clip))          # log-odds, (n,)
    prior  = float(np.clip(prior, 1e-9, 1 - 1e-9))
    lo_prior = np.log(prior / (1.0 - prior))           # log prior odds
    # Log Bayes factor per sample = log-odds(posterior) - log-odds(prior)
    lbf_arr = lo_arr - lo_prior                        # (n,)

    if study_labels is not None and len(study_labels) == len(p_arr):
        study_lbf = []
        for s in np.unique(study_labels):
            mask = np.asarray(study_labels) == s
            # Average within study (conservative: correlated samples)
            study_lbf.append(float(lbf_arr[mask].mean()))
        total_lbf = float(np.sum(study_lbf))
    else:
        total_lbf = float(lbf_arr.sum())

    # Convert back: posterior odds = prior odds * exp(total log BF)
    lo_post = lo_prior + total_lbf
    # Use scipy expit for numerically stable sigmoid (avoids overflow in exp)
    from scipy.special import expit as _sp_expit
    return float(_sp_expit(lo_post))


def _latent_L_prior_mass(tau, G, L_min_bp=500, pi_present=0.5):
    """
    Effective prior probability P(L >= tau) under the mixture prior.

    Mixture prior:
        (1 - pi_present) : absent/fragment component at L_min (contributes 0 above tau)
        pi_present        : log-uniform on [L_min, G]

    P(L >= tau) = pi_present * (log(G) - log(tau)) / (log(G) - log(L_min))

    Used as the study-level BF baseline in _hierarchical_aggregate.
    Pass the same pi_present as used in _latent_L_posterior_vec.

    Parameters
    ----------
    tau        : float — threshold in bp
    G          : float — genome length in bp
    L_min_bp   : float — lower integration limit in bp
    pi_present : float — prior probability of genuine presence (same as in
                         _latent_L_posterior_vec, default 0.5)

    Returns
    -------
    alpha : float in [0, 1]
    """
    G_f   = max(float(G), 1.0)
    tau_f = float(np.clip(tau, 1.0, G_f))
    L_min = float(np.clip(L_min_bp, 1, G_f * 0.999))
    pi    = float(np.clip(pi_present, 1e-9, 1.0))
    denom = np.log(G_f) - np.log(L_min)
    if denom <= 0:
        return 0.0
    alpha_logunif = float(np.clip((np.log(G_f) - np.log(tau_f)) / denom, 0.0, 1.0))
    return float(np.clip(pi * alpha_logunif, 0.0, 1.0))


def _union_aggregate(p_arr, study_labels=None, alpha_tau=0.0):
    """
    Prior-corrected union-style study-aware aggregation of per-sample posteriors.

    Answers: "is there genuine evidence for this SGB anywhere in the lifestyle?"

    Prior-subtraction step (critical for latent-L posteriors)
    ----------------------------------------------------------
    Uninformative absent samples return P(L>=tau|data) ~ alpha_tau (the prior
    mass above tau under the log-uniform L prior), not zero.  Without correction
    many absent samples compound via 1-prod(1-p) into a spurious positive.
    We subtract alpha_tau and normalize before aggregating:

        p_excess_i = max(0, (p_i - alpha_tau) / (1 - alpha_tau))

    p_i = alpha_tau  -> p_excess = 0  (uninformative, no contribution)
    p_i = 1.0        -> p_excess = 1.0 (certain genuine)
    p_i < alpha_tau  -> 0 (clipped, not negative)

    Pass alpha_tau = _latent_L_prior_mass(tau, G) for correct calibration.
    Default alpha_tau=0.0 preserves the raw union formula for other uses.

    Study-aware structure
    ---------------------
    Within study  : p_study = 1 - prod_{i in study}(1 - p_excess_i)
    Across studies: p_agg   = 1 - prod_{studies}   (1 - p_study)

    Parameters
    ----------
    p_arr        : (n,) per-sample posteriors in [0, 1]
    study_labels : (n,) array of study IDs, or None (pooled as one group)
    alpha_tau    : float — prior P(L>=tau) from _latent_L_prior_mass(tau, G)

    Returns
    -------
    p_agg : float in [0, 1]
    """
    p_arr = np.clip(np.asarray(p_arr, dtype=np.float64), 0.0, 1.0)

    # Subtract and renormalize to excess-above-prior scale
    alpha = float(np.clip(alpha_tau, 0.0, 1.0 - 1e-9))
    if alpha > 0.0:
        p_arr = np.clip((p_arr - alpha) / (1.0 - alpha), 0.0, 1.0)

    if study_labels is not None and len(study_labels) == len(p_arr):
        p_studies = []
        for s in np.unique(study_labels):
            mask  = np.asarray(study_labels) == s
            lnd_s = np.sum(np.log1p(-np.clip(p_arr[mask], 0.0, 1 - 1e-9)))
            p_studies.append(float(np.clip(1.0 - np.exp(lnd_s), 0.0, 1.0)))
        lnd_cross = np.sum(np.log1p(-np.clip(p_studies, 0.0, 1 - 1e-9)))
        return float(np.clip(1.0 - np.exp(lnd_cross), 0.0, 1.0))
    else:
        lnd = np.sum(np.log1p(-np.clip(p_arr, 0.0, 1 - 1e-9)))
        return float(np.clip(1.0 - np.exp(lnd), 0.0, 1.0))

def _hierarchical_aggregate(p_arr, study_labels=None, alpha_tau=0.0, gamma=0.15,
                            return_study_details=False):
    """
    Two-level hierarchical aggregation with gamma-damped negative evidence.

    Level 1 — within study (union of raw posteriors):
        p_study_k = 1 - prod_{i in study k} (1 - p_i)
        Union lets one positive sample carry the study without absent samples
        drowning it.

    Level 2 — across studies (size-normalised log-odds BF, negatives damped):
        alpha_study_n = 1 - (1 - alpha_tau)^{n_k}
            Expected p_study for a null study of n_k samples all at the prior.
        BF_k = logit(p_study_k) - logit(alpha_study_n)
        BF_damped_k = BF_k          if BF_k >= 0   (positive: full weight)
                    = gamma * BF_k  if BF_k <  0   (negative: dampened)
        p_lifestyle = logistic( logit(alpha_tau) + sum_k BF_damped_k )

    gamma semantics
    ---------------
    gamma=0   : no negative cross-study evidence (prior-subtracted union)
    gamma=0.15: recommended — deeply absent studies mildly reduce p_lifestyle,
                but one strongly positive study still dominates (rare-but-real
                taxa with convincing evidence remain detectable)
    gamma=1   : symmetric — many absent studies can overwhelm one positive
                (too conservative for typical microbiome survey data)

    The mixture prior in _latent_L_posterior_vec reduces p_i to near zero for
    truly absent samples; gamma prevents this from translating to catastrophic
    negative BF that would mask rare-but-real taxa.

    Parameters
    ----------
    p_arr        : (n,) raw latent-L posteriors in [0,1]
    study_labels : (n,) study IDs, or None
    alpha_tau    : float — effective prior from _latent_L_prior_mass(tau, G, pi_present)
    gamma        : float in [0,1] — negative BF dampening factor
    return_study_details : bool
        If True, return (p_agg, study_details) where study_details is a list of
        dicts {study, n_samples, p_study, alpha_study_n, bf_raw, bf_damped}.
        Useful for visualizing which studies drove or suppressed detection.

    Returns
    -------
    p_agg : float in [0, 1]
        With return_study_details=True, (p_agg, study_details) instead.
    """
    from scipy.special import logit as _logit, expit as _expit

    p_arr = np.clip(np.asarray(p_arr, dtype=np.float64), 1e-9, 1 - 1e-9)
    alpha = float(np.clip(alpha_tau, 1e-9, 1 - 1e-9))
    gam   = float(np.clip(gamma, 0.0, 1.0))
    lo_prior = float(_logit(alpha))

    if study_labels is None or len(study_labels) != len(p_arr):
        bf_arr    = _logit(p_arr) - lo_prior
        bf_damped = np.where(bf_arr >= 0, bf_arr, gam * bf_arr)
        p_agg = float(_expit(lo_prior + float(np.sum(bf_damped))))
        if return_study_details:
            return p_agg, []
        return p_agg

    total_lbf    = 0.0
    study_details = []
    for s in np.unique(study_labels):
        mask  = np.asarray(study_labels) == s
        p_s   = p_arr[mask]
        n_s   = int(mask.sum())

        # Within-study union of raw posteriors
        lnd_s   = float(np.sum(np.log1p(-p_s)))
        p_study = float(np.clip(1.0 - np.exp(lnd_s), 1e-9, 1 - 1e-9))

        # Size-normalised null baseline
        alpha_n = float(np.clip(1.0 - (1.0 - alpha) ** n_s, 1e-9, 1 - 1e-9))

        # Log BF: dampen negatives
        bf_raw = float(_logit(p_study) - _logit(alpha_n))
        bf_k   = bf_raw * gam if bf_raw < 0 else bf_raw
        total_lbf += bf_k

        if return_study_details:
            study_details.append(dict(
                study=s, n_samples=n_s,
                p_study=p_study, alpha_study_n=alpha_n,
                bf_raw=bf_raw, bf_damped=bf_k,
            ))

    p_agg = float(_expit(lo_prior + total_lbf))
    if return_study_details:
        return p_agg, study_details
    return p_agg


def _union_anywhere(p_arr, study_labels=None, return_study_details=False):
    """
    Pure two-level union estimator: P(present in at least one sample anywhere).

    Level 1 — within study (same as _hierarchical_aggregate level 1):
        p_study_k = 1 - ∏_{i ∈ study k} (1 - p_i)

    Level 2 — across studies (pure union, no BF, no damping):
        p_anywhere = 1 - ∏_k (1 - p_study_k)

    Rationale
    ---------
    _hierarchical_aggregate converts within-study union into a log-odds BF
    and sums those BFs across studies, damping negative ones by gamma.  That
    is appropriate for the question "replicated across studies?" because it
    rewards consistent replication and mildly penalises widespread absence.

    For the extinction / persistence question — "present in at least one
    sample anywhere?" — that cross-study penalty is wrong.  One study with
    high p_study should be sufficient for p_anywhere to be high, regardless
    of how many other studies found nothing.  The pure union satisfies this
    exactly: a single study at p_study = 0.99 gives p_anywhere ≥ 0.99 no
    matter how many null studies exist, because ∏(1 - 0) for null studies
    contributes exactly 1 to the survival product and cancels cleanly.

    The within-study union (level 1) is identical to _hierarchical_aggregate
    level 1, so p_study values in return_study_details are directly comparable
    to those returned by _hierarchical_aggregate(..., return_study_details=True).

    Parameters
    ----------
    p_arr              : (n,) per-sample latent-L posteriors in [0, 1]
    study_labels       : (n,) study IDs, or None (all samples treated as one study)
    return_study_details : bool — if True return (p_anywhere, study_details)
                          where study_details is a list of dicts
                          {study, n_samples, p_study}.

    Returns
    -------
    p_anywhere : float in [0, 1]  (or (p_anywhere, study_details) if requested)
    """
    p_arr = np.clip(np.asarray(p_arr, dtype=np.float64), 1e-9, 1 - 1e-9)

    if study_labels is None or len(study_labels) != len(p_arr):
        # No study structure: single-level union over all samples
        p_out = float(np.clip(1.0 - np.exp(np.sum(np.log1p(-p_arr))), 0.0, 1.0))
        if return_study_details:
            return p_out, []
        return p_out

    log_surv      = 0.0   # accumulates log(1 - p_study) across studies
    study_details = []
    for s in np.unique(study_labels):
        mask    = np.asarray(study_labels) == s
        p_s     = p_arr[mask]
        n_s     = int(mask.sum())
        # Within-study union (identical to _hierarchical_aggregate level 1)
        p_study = float(np.clip(
            1.0 - np.exp(np.sum(np.log1p(-p_s))), 1e-9, 1 - 1e-9
        ))
        log_surv += np.log1p(-p_study)
        if return_study_details:
            study_details.append(dict(study=s, n_samples=n_s, p_study=p_study))

    p_anywhere = float(np.clip(1.0 - np.exp(log_surv), 0.0, 1.0))
    if return_study_details:
        return p_anywhere, study_details
    return p_anywhere


def _estimate_pi_present(bfrac_s, genome_lengths, min_map_bp=10_000):
    """
    Data-derived estimate of pi_present (prior probability of genuine presence).

    Estimates the fraction of (sample, SGB) pairs whose breadth exceeds the
    per-SGB numerical floor min_map_bp / G_j.  Each SGB has its own floor
    because genome lengths differ; a fixed 3 Mb default gives wrong floors
    for compact (e.g. 1 Mb) or large (e.g. 6 Mb) genomes.

    Clipped to [0.05, 0.95] to avoid degenerate priors.

    Parameters
    ----------
    bfrac_s       : (n_samples, n_sgbs) breadth DataFrame
    genome_lengths: dict {sgb_id: length_bp} — required, no default
    min_map_bp    : int, minimum mapped base pairs (numerical floor)

    Returns
    -------
    pi : float in [0.05, 0.95]
    """
    sgbs = list(bfrac_s.columns)
    # Per-SGB breadth floor: min_map_bp / G_j
    floors = np.array([
        max(1e-4, min_map_bp / max(
            int(genome_lengths.get(str(s)) or genome_lengths.get(s)), 1
        ))
        for s in sgbs
    ], dtype=np.float64)  # (n_sgbs,)
    # Count pairs where breadth exceeds that SGB's own floor
    frac = float((bfrac_s.values > floors[np.newaxis, :]).mean())
    return float(np.clip(frac, 0.05, 0.95))


def _estimate_gamma(study_labels, alpha_tau, rare_frac=0.10, target_bf_pos=5.0):
    """
    Data-derived estimate of the negative-BF dampening factor gamma.

    Sets gamma so that a "rare-but-real" SGB is still detectable: an SGB
    present in rare_frac of studies, where each positive study contributes
    a log Bayes factor of target_bf_pos, should give p_lifestyle > 0.5 even
    against the remaining fully-absent null studies.

    Derivation (at the detection boundary p_lifestyle = 0.5, logit(0.5) = 0):

        0 = logit(alpha_tau) + n_pos * target_bf_pos + n_neg * gamma * bf_null

        gamma = -(logit(alpha_tau) + n_pos * target_bf_pos) / (n_neg * bf_null)

    where bf_null is the log BF from a fully null study of median size (p_study ≈ 0).
    This replaces the hand-tuned default gamma=0.15.

    Parameters
    ----------
    study_labels  : (n,) array of study IDs, or None
    alpha_tau     : effective prior P(L >= tau) from _latent_L_prior_mass
    rare_frac     : fraction of studies expected to contain the SGB (default 0.10)
    target_bf_pos : target per-study log BF for a strongly positive study (default 5.0)

    Returns
    -------
    gamma : float in [0.01, 1.0]
    """
    from scipy.special import logit as _sp_logit

    if study_labels is None:
        return 0.15

    studies, counts = np.unique(study_labels, return_counts=True)
    K = len(studies)
    if K < 3:
        return 0.15

    n_pos = max(1, int(round(rare_frac * K)))
    n_neg = K - n_pos
    if n_neg < 1:
        return 0.15

    median_n = float(np.median(counts))
    # Expected BF for a fully null study (p_study ≈ 0, clipped to 1e-6)
    alpha_n  = float(np.clip(1.0 - (1.0 - alpha_tau) ** median_n, 1e-9, 1 - 1e-9))
    bf_null  = float(_sp_logit(1e-6) - _sp_logit(alpha_n))   # large negative

    alpha_lo = float(_sp_logit(np.clip(alpha_tau, 1e-9, 1 - 1e-9)))
    # Solve: 0 = alpha_lo + n_pos * target_bf_pos + n_neg * gamma * bf_null
    numerator   = -(alpha_lo + n_pos * target_bf_pos)
    denominator =  n_neg * bf_null   # bf_null < 0 → denominator < 0

    if abs(denominator) < 1e-9:
        return 0.15

    gamma = numerator / denominator
    return float(np.clip(gamma, 0.01, 1.0))

def _estimate_sigma_b(
    bfrac_s:      pd.DataFrame,
    depth_s:      pd.DataFrame,
    c:            float,
    f_null:       float = 0.10,
    n_bins:       int   = 12,
    min_pts_bin:  int   = 50,
    n_em_iter:    int   = 5,
    fallback:     float = None,
) -> tuple:
    """
    Depth-stratified breadth noise model estimated via EM.

    For each (sample, SGB) pair the residual is r = b - b_null_f(d).
    Within each log-depth bin a two-component mixture model is fitted:

        null component:    r ~ N(0, sigma^2)    [near-null breadth noise]
        genuine component: r ~ Uniform(r_min, r_max)

    The EM jointly estimates sigma (null spread) and pi (null fraction).
    This replaces the old hard |r/sigma| < 1.5 threshold, which required a
    pre-fixed sigma and created a circular dependency with the detection rule.

    If fallback is None it is estimated from the global MAD of all residuals,
    making the model fully self-contained with no hand-tuned constants.

    Returns
    -------
    sigma_by_bin : ndarray (n_bins,) — estimated sigma in each log-depth bin;
                   defaults to the data-derived fallback for sparse bins.
    bin_edges    : ndarray (n_bins+1,) — log10(depth) bin edges.
    """
    b_mat = bfrac_s.values.ravel()
    d_mat = depth_s[bfrac_s.columns].values.ravel()

    valid = (d_mat > 0) & np.isfinite(b_mat) & np.isfinite(d_mat)
    b_v   = b_mat[valid]
    d_v   = d_mat[valid]

    b_null = f_null * (1.0 - np.exp(-c * d_v / f_null))
    resid  = b_v - b_null

    # ── Fallback: global MAD — no hand-tuned constant ─────────────────────
    if fallback is None:
        global_mad = float(np.median(np.abs(resid - np.median(resid))))
        fallback   = max(global_mad / 0.6745, 1e-4)

    log_d = np.log10(np.maximum(d_v, 1e-9))
    edges = np.linspace(log_d.min(), log_d.max(), n_bins + 1)

    sigma_by_bin = np.full(n_bins, fallback, dtype=float)
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        in_bin = (log_d >= lo) & (log_d < hi)
        if in_bin.sum() < min_pts_bin:
            continue

        r_bin  = resid[in_bin]
        r_span = max(float(r_bin.max() - r_bin.min()), 1e-4)
        p_unif = 1.0 / r_span   # density of the non-null Uniform component

        # EM: jointly estimate sigma and null fraction pi
        sigma_k = fallback
        pi_k    = 0.80   # initial guess: 80% null-consistent

        for _ in range(n_em_iter):
            # E-step: soft null responsibility
            log_p_null = (- 0.5 * (r_bin / max(sigma_k, 1e-9)) ** 2
                          - np.log(max(sigma_k, 1e-9))
                          + np.log(max(pi_k, 1e-9)))
            log_p_gen  = np.log(p_unif) + np.log(max(1.0 - pi_k, 1e-9))
            # Numerically stable 2-component softmax
            log_max    = np.maximum(log_p_null, log_p_gen)
            pn_u  = np.exp(log_p_null - log_max)
            pg_u  = np.exp(log_p_gen  - log_max)
            w     = pn_u / np.maximum(pn_u + pg_u, 1e-300)

            # M-step: update sigma and pi
            w_sum   = max(float(w.sum()), 1e-9)
            sigma_k = float(np.sqrt(np.sum(w * r_bin ** 2) / w_sum))
            sigma_k = max(sigma_k, 1e-4)
            pi_k    = float(w.mean())

        sigma_by_bin[k] = sigma_k

    return sigma_by_bin, edges


def _lookup_sigma_b(d_val: float, sigma_by_bin: np.ndarray,
                    bin_edges: np.ndarray, fallback: float = 0.015) -> float:
    """Look up the learned sigma_b for a given depth value."""
    if sigma_by_bin is None or bin_edges is None:
        return fallback
    log_d = np.log10(max(d_val, 1e-9))
    k     = np.searchsorted(bin_edges[1:], log_d)
    k     = int(np.clip(k, 0, len(sigma_by_bin) - 1))
    return float(sigma_by_bin[k])

# =============================================================================
#  Figure 13 -- Per-SGB breadth-depth curves:  b = A * (1 - exp(-c * d))
#  Detection: p_genuine >= 0.5  (multi-sample soft posterior, z_detect=1.0)
#  A_fit is a characterisation score only (not a detection gate).
# =============================================================================

def _poisson_asymptote(d, A, c=1.0):
    """b = A * (1 - exp(-c * d)).  Asymptote = A, initial slope = A*c."""
    A = max(A, 1e-9)
    c = max(c, 1e-9)
    return A * (1.0 - np.exp(-c * d))



def _estimate_coverage_efficiency(bfrac_s, depth_s,
                                   max_b_frontier = 0.10,
                                   min_b          = 0.002,
                                   min_d          = 1e-6,
                                   n_bins         = 10,
                                   min_strip_pts  = 15):
    """
    Estimate c from the LEFT-MOST EDGE of the (depth, breadth) cloud.

    Physical constraint
    -------------------
    In the linear regime (b << 1): b ≈ c·d, so the slope b/d ≈ c.
    The maximum achievable breadth at depth d is b_max = c·d (Poisson model).
    No (d, b) pair can satisfy b > c·d — equivalently, no sample lies to the
    LEFT of the line b = c·d in (depth, breadth) space.  c is therefore the
    slope of the UPPER-LEFT FRONTIER of the cloud, not an interior percentile.

    Why the old 90th-percentile approach was wrong
    -----------------------------------------------
    Using the 90th percentile of b/d over b ∈ (0.002, 0.20):

      (a) Wrong range: points with b ∈ (0.10, 0.20) are no longer in the
          strictly linear regime — the Poisson curve is already bending, so
          b/d systematically UNDERESTIMATES c there.  Including them pulls
          the estimate down, producing an artificially low c.

      (b) Wrong percentile: the 90th percentile leaves 10% of samples with
          b/d > ĉ, i.e., to the LEFT of the fitted line.  The model then
          treats those samples as having excellent Poisson fit even at very
          low breadth/depth, causing overconfident posteriors where no genuine
          signal exists.

    Left-edge frontier estimation (new approach)
    --------------------------------------------
    1. Restrict to b ∈ (min_b, max_b_frontier=0.10).  Here b ≈ c·d holds to
       within ~5% (b=0.10 → 1-exp(-0.10) ≈ 0.095, error < 5%), so b/d is a
       clean, nearly unbiased estimator of c.

    2. Bin this region into n_bins breadth strips.  Within each strip take the
       99th percentile of b/d — the near-maximum (frontier) slope, excluding
       only the top ~1% of probable noise points per strip.

    3. c = max across strips.  Taking MAX (not median or mean) ensures c is at
       or above the frontier slope in EVERY breadth strip.  A sample is "to the
       left of the line" iff b/d > c; by taking the max of per-strip 99th
       percentiles, this can only happen for the top ~1% noise in each strip,
       which is the intended noise-rejection tolerance.

    The result: c is the tightest data-driven upper bound on b/d at low breadth,
    calibrated so that (nearly) no observed sample lies to the left of b = c·d.

    Parameters
    ----------
    max_b_frontier : float — upper breadth limit for frontier points (default 0.10).
                     Fallback to 0.20 if fewer than min_strip_pts * n_bins points
                     are available at this threshold.
    min_b          : float — lower breadth floor (numerical noise floor).
    min_d          : float — lower depth floor.
    n_bins         : int   — number of breadth strips for per-strip estimation.
    min_strip_pts  : int   — minimum points required to include a strip (strips
                     with fewer points are skipped to avoid noise-driven outliers).

    Returns
    -------
    c : float in [0.50, 1.00]
    """
    b_all = bfrac_s.values.ravel()
    d_all = depth_s.values.ravel()
    mask  = (b_all > min_b) & (b_all < max_b_frontier) & (d_all > min_d)

    # Fallback: if too few frontier points, expand to full linear regime
    if mask.sum() < min_strip_pts * n_bins:
        mask_fb = (b_all > min_b) & (b_all < 0.20) & (d_all > min_d)
        if mask_fb.sum() < min_strip_pts * n_bins:
            print("    WARNING: too few linear-regime points; defaulting c=1.0")
            return 1.0
        print(
            f"    NOTE: frontier expanded from max_b={max_b_frontier:.2f} to 0.20 "
            f"({mask_fb.sum():,} pts available, needed ≥{min_strip_pts * n_bins})"
        )
        mask = mask_fb

    b_m    = b_all[mask]
    d_m    = d_all[mask]
    slopes = b_m / d_m

    # Per-strip 99th-percentile of b/d → max across strips
    b_edges   = np.linspace(b_m.min(), b_m.max(), n_bins + 1)
    strip_p99 = []
    for i in range(n_bins):
        in_strip = (b_m >= b_edges[i]) & (b_m < b_edges[i + 1])
        if in_strip.sum() < min_strip_pts:
            continue   # too few points for a reliable 99th percentile
        strip_p99.append(float(np.percentile(slopes[in_strip], 99)))

    if len(strip_p99) < 2:
        # Degenerate: fall back to single global 99th percentile
        c_est = float(np.percentile(slopes, 99))
    else:
        c_est = float(np.max(strip_p99))

    c_est = float(np.clip(c_est, 0.50, 1.00))

    # Diagnostic: count points still to the left of the line (b/d > c)
    n_left = int((slopes > c_est).sum())
    print(
        f"    Coverage efficiency c (left-edge frontier, "
        f"{mask.sum():,} pts, {len(strip_p99)} strips, p99 per strip): "
        f"c = {c_est:.4f}  [{n_left} pts ({100*n_left/len(slopes):.2f}%) "
        f"above line — should be ≲1%]"
    )
    return c_est


def _discrimination_depth(A_fit, A_min=0.50, c=1.0, eps=0.02, d_max=10.0):
    """
    Return d* such that b(d*; A_fit, c) - b(d*; A_min, c) = eps.
    Returns np.inf if A_fit <= A_min or the gap never reaches eps within d_max.
    """
    from scipy.optimize import brentq
    if A_fit <= A_min + 1e-6:
        return np.inf
    def delta(d):
        return _poisson_asymptote(d, A_fit, c) - _poisson_asymptote(d, A_min, c) - eps
    if delta(d_max) < 0:
        return np.inf
    try:
        return float(brentq(delta, 1e-8, d_max, xtol=1e-6))
    except Exception:
        return np.inf


def print_back_of_envelope(A_min=0.50, c=1.0,
                            eps_values=(0.01, 0.02, 0.05),
                            A_fits=(0.55, 0.6, 0.7, 0.8, 0.9, 1.0)):
    """Print a table of discrimination depths for various A_fit / eps pairs."""
    print(f"\n  Back-of-envelope discrimination depths  (c={c:.3f})")
    print(f"  d* = min depth where |b(d;A_fit,c) - b(d;A_min={A_min},c)| > eps")
    header = f"  {'A_fit':>6}" + "".join(f"   eps={e:.2f}" for e in eps_values)
    sep    = "  " + "-" * (len(header) - 2)
    print(sep); print(header); print(sep)
    for A in A_fits:
        row = f"  {A:>6.2f}"
        for eps in eps_values:
            d = _discrimination_depth(A, A_min=A_min, c=c, eps=eps)
            row += f"   {d:>8.3f}" if np.isfinite(d) else f"   {'inf':>8}"
        print(row)
    print(sep)
    print("  Samples at d > d* that lie on the A_fit curve are genuine evidence")
    print("  against A_min, even at low breadth.\n")


def _fragment_null_breadth(d_arr, f_frag, c):
    """
    Poisson breadth curve for spurious reads mapping only to a genome fragment
    of fractional length f_frag (= fragment_bp / genome_bp).

        b_null(d; f) = f * (1 - exp(-c * d / f))

    At low d:  b_null ≈ c * d  (same initial slope as genuine mapping)
    At high d: b_null → f      (saturates at the fragment fraction)

    Points above this curve at a given d cannot be explained by spurious
    mapping to a fragment of that size — they are genuine whole-genome signals.
    """
    f = float(np.clip(f_frag, 1e-6, 1.0))
    return f * (1.0 - np.exp(-c * np.asarray(d_arr) / f))


NULL_MODELS = {           # name: (f_frag_or_None, kb_or_None)
    "1pct":   (0.01,  None),
    "3pct":   (0.03,  None),
    "10pct":  (0.10,  None),
    "30pct":  (0.30,  None),
    "50pct":  (0.50,  None),
    "90pct":  (0.90,  None),
    "10kb":   (None,  10),
    "30kb":   (None,  30),
    "100kb":  (None,  100),
    "300kb":  (None,  300),
    "500kb":  (None,  500),
    "1Mb":    (None,  1000),
}
NULL_STYLES = {           # name: (colour, linestyle, linewidth, label)
    "1pct":  ("#bdbdbd",            ":",  1.0, "null 1%"),
    "3pct":  ("#969696",            ":",  1.0, "null 3%"),
    "10pct": (PALETTE["sky"],       ":",  1.0, "null 10%"),
    "30pct": ("#888888",            "--", 1.0, "null 30%"),
    "50pct": ("#525252",            "--", 1.0, "null 50%"),
    "90pct": ("#252525",            "--", 1.0, "null 90%"),
    "10kb":  ("#f6c65b",            ":",  1.0, "null 10 kb"),
    "30kb":  ("#f0a330",            ":",  1.0, "null 30 kb"),
    "100kb": (PALETTE["orange"],    ":",  1.0, "null 100 kb"),
    "300kb": (PALETTE["pink"],      ":",  1.0, "null 300 kb"),
    "500kb": ("#8e3b78",            ":",  1.0, "null 500 kb"),
    "1Mb":   ("#5b2c6f",            ":",  1.0, "null 1 Mb"),
}


def fit_sgb_asymptote(d_arr, b_arr,
                      c            = 1.0,
                      A_min        = 0.30,
                      sigma_b_ref  = 0.015,
                      sigma_bins   = None,   # learned depth-stratified sigma array
                      sigma_edges  = None,   # log10(depth) bin edges for lookup
                      z_thresh     = 4.5,
                      z_detect     = 1.0,
                      eps_disc     = 0.02,
                      G_genome     = None,   # REQUIRED: SGB genome length in bp — no default, must be supplied from genome-length file
                      min_map_bp   = 10_000,
                      study_arr    = None,
                      pi_present   = 0.5,
                      gamma        = 0.15):   # negative BF dampening; set automatically by compute_asymptote_fits
    """
    Per-SGB threshold-specific detection using the latent represented-length model.

    Core model (per sample)
    -----------------------
    Latent represented length L ~ mixture(absent@L_min, log-Uniform([L_min, G])).
    Likelihood: b_i ~ Beta(mu * phi_i, (1-mu) * phi_i)
    where  mu(L; d, G, c) = (L/G) * (1 - exp(-c * d * G / L))
    and    phi_i = 1 / (4 * sigma_i^2)   (effective observation count from sigma model).

    The Beta likelihood replaces the Gaussian approximation:
      • Bounded to [0,1] — no probability mass outside the feasible range.
      • Heteroskedastic — Var(b) = mu*(1-mu)/phi, tighter near 0 and 1.
      • Correctly penalises low-breadth observations at high depth (e.g. b=0.01
        when mu(10%*G)=0.10 at d=1.0 is a ~10-sigma Beta event vs ~6-sigma Gaussian).

    pi_present controls the mixture prior:
      pi_present=0.5  (default) — equal prior on absent/present; most skeptical
      pi_present=0.9            — 90% prior on genuine presence; less shrinkage
    Sensitivity to pi_present can be tested via compute_asymptote_fits.

    Threshold map (in bp):
        10pct : tau = 0.10 * G   (genome-relative)
        30pct : tau = 0.30 * G
        100kb : tau = 100,000    (absolute, genome-size-independent)
        300kb : tau = 300,000

    Aggregation (biological estimand: "present anywhere?")
    -------------------------------------------------------
    Two-level hierarchical aggregation via _hierarchical_aggregate:

    Level 1 — within study (union of raw posteriors):
        p_study_k = 1 − prod_{i in study k} (1 − p_i)
        One genuinely positive sample carries the study even with many absent
        samples in the same study.

    Level 2 — across studies (size-normalised log-odds BF, negatives gamma-damped):
        alpha_study_n = 1 − (1 − alpha_tau)^{n_k}
            Expected p_study for a null study of n_k samples at the prior.
        BF_k          = logit(p_study_k) − logit(alpha_study_n)
        BF_damped_k   = BF_k  if BF_k >= 0  (positive: full weight)
                      = γ·BF_k if BF_k <  0  (negative: γ-damped, γ=0.15)
        p_lifestyle   = logistic(logit(alpha_tau) + Σ_k BF_damped_k)

    Deeply absent studies (p_i ≈ 0 from mixture prior, p_study << alpha_study_n)
    contribute mild negative BF, gently lowering p_lifestyle.  The γ=0.15
    dampening prevents many absent studies from overwhelming one strongly positive
    study (rare-but-real protection).  The mixture prior (pi_present=0.5) in
    _latent_L_posterior_vec reduces p_i to near 0 for truly absent samples,
    making the negative-evidence signal calibrated rather than arbitrary.

    Primary threshold and detection rule
    -------------------------------------
    p_genuine (= p_genuine_10pct) is the primary detection score.
    detectable = (p_genuine_10pct >= 0.5).

    p_genuine_* consistency-threshold are reported as support
    tiers for researchers to interpret but do NOT gate the detectable flag.

    Spread diagnostics (A_cv, gini_A)
    -----------------------------------
    Computed and stored in the CSV for downstream use, but NOT applied as a
    hard penalty to p_genuine.  They are cross-sample heterogeneity summaries
    and punish rare-but-real taxa with a few high-breadth observations in a
    way that within-sample localization features would not.

    Outputs
    -------
    p_genuine_{nm}  : union posterior P(present in at least one study), per threshold
    p_genuine       : p_genuine_10pct  [primary detection score]
    p_combined      : p_genuine_10pct  [kept for back-compat, same value]
    detectable      : p_genuine_10pct >= 0.5
    A_cv, gini_A    : spread diagnostics (CSV only, not used for detection)
    p_detect_{nm}   : legacy sigmoid score (diagnostic, not used for detection)
    n_above_{nm}    : geometric count above null curve (diagnostic)
    """

    if G_genome is None:
        raise ValueError(
            "fit_sgb_asymptote: G_genome is required. "
            "Load genome lengths from all_SGB_lengths_*.txt and pass "
            "genome_lengths[sgb_id] for each SGB."
        )
    # ── Numerical floor (not a biological gate) ───────────────────────────
    effective_min_b = max(1e-4, min_map_bp / max(G_genome, 1))

    # Samples that survive the numerical floor
    pos_all = ((d_arr > 0) & (b_arr >= effective_min_b)
               & np.isfinite(d_arr) & np.isfinite(b_arr))
    d_e = d_arr[pos_all]
    b_e = b_arr[pos_all]
    d_max = float(d_arr[(d_arr > 0) & np.isfinite(d_arr)].max()) \
            if ((d_arr > 0) & np.isfinite(d_arr)).any() else 0.0

    # Blank output
    out = dict(
        A_fit=np.nan, A_cv=np.nan, gini_A=np.nan, rmse_env=np.nan,
        n_points=int(pos_all.sum()), d_max=d_max,
        d_star=np.nan, p_detect=0.0, p_genuine=0.0, z_max=np.nan,
        effective_min_b=effective_min_b, G_genome=int(G_genome),
        has_data=False, fit_mode='no_data',
        detectable=False, fit_ok=False,
        env_d=None, env_b=None, env_p=None,
        study_contributions_10pct=[],
        # p_anywhere_* — primary detection score (pure two-level union).
        # Answers "present in at least one sample anywhere?".  One strong
        # study is sufficient; absent studies do not reduce the score.
        p_anywhere=0.0,
        # p_replication — secondary cross-study consistency score (BF-based).
        # Answers "replicated across studies?".  Kept as p_genuine for compat.
        p_replication=0.0,
    )
    _study_details_10pct = []  # initialised here; populated in threshold loop
    for nm in NULL_MODELS:
        out[f"n_above_{nm}"]    = 0
        out[f"n_bent_{nm}"]     = 0
        out[f"p_detect_{nm}"]   = 0.0
        out[f"p_genuine_{nm}"]  = 0.0   # BF replication score per threshold
        out[f"p_anywhere_{nm}"] = 0.0   # union detection score per threshold
    out["p_genuine_base"] = 0.0
    # p_genuine_30pct_support: 30%-threshold posterior tier (support only, not gate).
    # Previously a misleading name; now clearly a secondary tier, not a detection gate.
    out["p_genuine_30pct_support"] = 0.0
    out["p_combined"] = 0.0
    out["cv_factor"]      = 1.0

    if len(d_e) < 1:
        return out

    out["has_data"] = True
    out["fit_mode"] = "floor"

    # Denominator for Poisson inversion
    f_e = np.where(1.0 - np.exp(-c * d_e) > 1e-4,
                   1.0 - np.exp(-c * d_e), 1e-4)

    # Initialise A_cv here so it is always defined as a local variable
    # regardless of which branch is taken below.  Python marks any variable
    # assigned anywhere in a function body as local throughout — without
    # this line, reaching the A_cv computation block before the mid_mask
    # computation raises UnboundLocalError.
    A_cv = np.nan

    # ── Per-sample sigma_b from learned depth-stratified model ───────────
    # If sigma_bins/sigma_edges are provided (from compute_asymptote_fits),
    # each sample gets its own sigma looked up at its depth.  This replaces
    # the single scalar sigma_b_ref that was previously used for all samples.
    if sigma_bins is not None and sigma_edges is not None:
        sigma_e = np.array([
            _lookup_sigma_b(d, sigma_bins, sigma_edges, fallback=sigma_b_ref)
            for d in d_e
        ], dtype=float)
    else:
        sigma_e = np.full(len(d_e), sigma_b_ref, dtype=float)
    sigma_e = np.maximum(sigma_e, 1e-9)

    # ── Fragment null fractions (kept for geometric diagnostics + plots) ──
    f_fracs = {}
    for nm, (f_fixed, kb) in NULL_MODELS.items():
        f_fracs[nm] = f_fixed if f_fixed is not None \
                      else min(1.0, kb * 1000 / max(G_genome, 1))

    # ── Geometric above-null counts (diagnostic only) ─────────────────────
    for nm, f_frag in f_fracs.items():
        b_null_arr = _fragment_null_breadth(d_e, f_frag, c)
        above_eff  = ((b_e > b_null_arr) & (b_e > effective_min_b)).astype(int)
        z_k        = (b_e - b_null_arr) / sigma_e
        n_above    = int(above_eff.sum())
        n_bent     = int((z_k > z_thresh).sum())
        from scipy.special import expit as _expit
        p_each     = _expit((z_k - z_thresh) * 1.5)
        lnd        = np.sum(np.log1p(-np.clip(p_each, 0, 1 - 1e-6)))
        p_det      = float(np.clip(1.0 - np.exp(lnd), 0, 1))
        out[f"n_above_{nm}"]  = n_above
        out[f"n_bent_{nm}"]   = n_bent
        out[f"p_detect_{nm}"] = p_det   # legacy diagnostic, not used for detection

    # z_max and n_bent kept for A_fit envelope restriction (geometric anchor)
    b_null_10 = _fragment_null_breadth(d_e, f_fracs["10pct"], c)
    z_primary = (b_e - b_null_10) / sigma_e
    z_max     = float(z_primary.max()) if len(z_primary) > 0 else 0.0
    n_bent    = out["n_bent_10pct"]
    p_detect  = out["p_detect_10pct"]

    # ── Latent-L posteriors: one per null threshold ────────────────────────
    # For each threshold tau (bp), compute P(L >= tau | b_i, d_i, G) per sample
    # using _latent_L_posterior_vec, then aggregate with the union estimator
    # _hierarchical_aggregate (gamma-damped study-level log-odds BF).
    #
    # Primary threshold: 10pct drives detectable.
    # Support tiers: the configured consistency thresholds are additional outputs.

    NULL_TAU = {
        "10pct": 0.10 * G_genome,
        "30pct": 0.30 * G_genome,
        "10kb": 10_000.0,
        "30kb": 30_000.0,
        "100kb": 100_000.0,
        "300kb": 300_000.0,
        "500kb": 500_000.0,
        "1Mb": 1_000_000.0,
    }

    # study_e: study labels aligned to above-floor samples
    study_e = study_arr[pos_all] if (study_arr is not None
                                     and len(study_arr) == len(d_arr)) else None

    p_nm_aggregated = {}   # BF-based replication score per threshold
    p_nm_anywhere   = {}   # pure-union detection score per threshold
    for nm, tau in NULL_TAU.items():
        if tau >= G_genome:          # threshold exceeds genome (tiny contig)
            p_nm_aggregated[nm] = 0.0
            p_nm_anywhere[nm]   = 0.0
            out[f"p_genuine_{nm}"]  = 0.0
            out[f"p_anywhere_{nm}"] = 0.0
            continue

        # Effective prior mass under the mixture prior.
        # P(L >= tau) = pi_present * P_log-uniform(L >= tau)
        alpha_tau_nm = _latent_L_prior_mass(tau, G_genome, pi_present=pi_present)

        p_L_samples = _latent_L_posterior_vec(
            b_arr=b_e, d_arr=d_e, G=G_genome, c=c,
            sigma_arr=sigma_e, tau=tau, n_quad=60, pi_present=pi_present,
        ).astype(float)

        # ── BF replication score (secondary): rewards consistent across-study signal
        p_agg = _hierarchical_aggregate(p_L_samples, study_labels=study_e,
                                        alpha_tau=alpha_tau_nm, gamma=gamma)
        # For the primary threshold, capture per-study BF breakdown for plotting
        if nm == "10pct" and study_e is not None:
            _, _study_details_10pct = _hierarchical_aggregate(
                p_L_samples, study_labels=study_e,
                alpha_tau=alpha_tau_nm, gamma=gamma, return_study_details=True)
        elif nm == "10pct":
            _study_details_10pct = []
        p_nm_aggregated[nm]    = p_agg
        out[f"p_genuine_{nm}"] = p_agg   # BF replication score

        # ── Union detection score (primary): pure two-level union, no BF, no damping.
        # One study with high p_study is sufficient — absent studies contribute
        # exactly zero to the log-survival sum and do not reduce p_anywhere.
        p_union = _union_anywhere(p_L_samples, study_labels=study_e)
        p_nm_anywhere[nm]       = p_union
        out[f"p_anywhere_{nm}"] = p_union

    # ── Primary detection score: p_anywhere_10pct (pure union) ───────────────
    # Estimand: "present in at least one sample anywhere?"
    # One strong study is sufficient; absent studies are neutral.
    p_anywhere      = p_nm_anywhere.get("10pct", 0.0)

    # ── Secondary replication score: p_genuine_10pct (BF-based) ─────────────
    # Estimand: "replicated consistently across studies?"
    # Penalises localised or study-specific detections.  Useful for assessing
    # confidence but too conservative for the primary extinction question.
    p_genuine_base  = p_nm_aggregated.get("10pct", 0.0)
    p_genuine       = p_genuine_base
    cv_factor       = 1.0

    # p_replication is a clearer synonym for the BF score; p_genuine kept for compat.
    out["p_anywhere"]             = p_anywhere
    out["p_replication"]          = p_genuine_base
    out["p_genuine_base"]         = p_genuine_base
    out["p_genuine_30pct_support"] = p_nm_aggregated.get("30pct", 0.0)
    # p_combined now points to p_anywhere (the primary score) for back-compat.
    out["p_combined"]             = p_anywhere
    out["cv_factor"]              = cv_factor

    # ── Spread diagnostics: A_cv and gini_A (CSV output only) ────────────
    # Cross-sample heterogeneity of A_implied = b/(1-exp(-cd)).
    # Stored in the output CSV for downstream analysis.  NOT applied as a
    # penalty to p_genuine: rare-but-real taxa have high cross-sample
    # heterogeneity by nature (present in few samples at high breadth, absent
    # in most), and penalising that pattern suppresses exactly what we want to find.
    mid_mask = f_e > 0.10
    gini_A   = np.nan
    if mid_mask.sum() >= 3:
        A_imp = (b_e[mid_mask] / f_e[mid_mask])
        A_imp = A_imp[np.isfinite(A_imp) & (A_imp > 0)]
        if len(A_imp) >= 3:
            A_cv   = float(np.std(A_imp) / max(np.mean(A_imp), 1e-9))
            x      = np.sort(A_imp)
            n_g    = len(x)
            ranks  = np.arange(1, n_g + 1, dtype=float)
            gini_A = float((2.0*np.sum(ranks*x)/(n_g*np.sum(x))) - (n_g+1.0)/n_g)
        else:
            A_cv = np.nan
    else:
        A_cv = np.nan
    out["gini_A"] = gini_A

    # ── p_each_gen: per-sample 10pct posteriors for envelope-dot plotting ──
    # These are the individual sample contributions to p_genuine (the primary
    # 10pct union posterior).  Samples with p_each_gen > 0.5 are highlighted
    # in _draw_curve_panel as the ones driving detection.  Using the same
    # threshold as detectable (0.5) keeps plots and decision rule aligned.
    p_each_gen = _latent_L_posterior_vec(
        b_arr=b_e, d_arr=d_e, G=G_genome, c=c,
        sigma_arr=sigma_e, tau=NULL_TAU["10pct"], n_quad=60, pi_present=pi_present,
    ).astype(float)


    # ── A_fit: envelope RESTRICTED to above-null-10pct samples ──────────
    # Only points geometrically above the 10%-null anchor A_fit.
    # All-cloud binning let null-consistent points bias A downward.
    b_null_10   = _fragment_null_breadth(d_e, f_fracs["10pct"], c)
    above_10    = b_e > b_null_10
    fit_mode    = "floor"
    A_fit       = 0.005

    if above_10.any():
        d_ab     = d_e[above_10]
        b_ab     = b_e[above_10]
        log_d_ab = np.log10(np.maximum(d_ab, 1e-9))
        edges_ab = np.linspace(log_d_ab.min(), log_d_ab.max(), 21)
        env_A    = []
        for lo_b, hi_b in zip(edges_ab[:-1], edges_ab[1:]):
            mk = (log_d_ab >= lo_b) & (log_d_ab < hi_b)
            if mk.sum() < 1:
                continue
            b_top = float(np.percentile(b_ab[mk], 95))
            d_mid = 10 ** ((lo_b + hi_b) / 2.0)
            f_mid = max(1.0 - np.exp(-c * d_mid), 1e-6)
            A_bin = b_top / f_mid
            if np.isfinite(A_bin) and 0 < A_bin <= 1.05:
                env_A.append(A_bin)
        if env_A:
            A_fit    = float(np.mean(env_A))
            fit_mode = "envelope"
        else:
            # Single-sample fallback: highest-breadth above-null sample
            idx_best = int(np.argmax(b_ab))
            f_s      = max(1.0 - np.exp(-c * d_ab[idx_best]), 1e-6)
            A_fit    = float(b_ab[idx_best] / f_s)
            fit_mode = "single_sample"
    # else: fit_mode stays "floor", A_fit stays 0.005

    A_fit    = float(np.clip(A_fit, 0.005, 1.0))
    b_pred   = _poisson_asymptote(d_e, A_fit, c)
    rmse_env = float(np.sqrt(np.mean((b_e - b_pred) ** 2)))
    d_star   = _discrimination_depth(A_fit, A_min=A_min, c=c, eps=eps_disc)

    # ── Envelope dots: samples whose per-sample posterior exceeds 0.5 ────────
    # p_each_gen is the 10pct latent-L posterior — the same quantity that drives
    # p_genuine and detectable.  The 0.5 gate here matches the detection threshold
    # so the highlighted dots in the plot explain exactly the green/red decision.
    contrib_mask = p_each_gen > 0.5
    if contrib_mask.any():
        env_d = d_e[contrib_mask]
        env_b = b_e[contrib_mask]
        env_p = p_each_gen[contrib_mask]
    else:
        env_d = env_b = env_p = None

    out.update(
        A_fit=A_fit, A_cv=A_cv, gini_A=gini_A, rmse_env=rmse_env, d_star=d_star,
        z_max=z_max, p_detect=p_detect,
        p_genuine=p_genuine,           # BF replication score (secondary)
        p_anywhere=p_anywhere,         # pure-union detection score (primary)
        n_bent=n_bent,
        frac_bent=float(n_bent / max(len(z_primary), 1)),
        has_data=True, fit_mode=fit_mode,
        fit_ok=(fit_mode in ("envelope", "single_sample")),
        # PRIMARY detector: p_anywhere_10pct >= 0.5
        # "Present in at least one sample anywhere?" — pure two-level union.
        # One study with strong evidence is sufficient; absent studies are neutral.
        # SECONDARY: p_genuine_10pct >= 0.5 — BF replication score for consistency.
        detectable=(p_anywhere >= 0.5),
        env_d=env_d, env_b=env_b, env_p=env_p,
        # Per-study BF contributions for the primary 10% threshold.
        # List of dicts: {study, n_samples, p_study, alpha_study_n, bf_raw, bf_damped}.
        # p_study here is the within-study union (identical in both detectors).
        # Used by _draw_curve_panel to show which studies drove/suppressed detection.
        study_contributions_10pct=_study_details_10pct,
    )
    return out


def compute_asymptote_fits(
    breadth_df,
    depth_df,
    meta,
    lifestyle        = None,    # None=all, "Industrial", "Non-Industrial"
    min_breadth_pct  = 0.5,
    A_min            = 0.30,
    sigma_b_ref      = 0.015,
    z_thresh         = 4.5,
    eps_disc         = 0.02,
    c_global         = None,
    genome_lengths   = None,    # REQUIRED: dict {sgb_id: length_bp} from all_SGB_lengths_*.txt
    min_map_bp       = 10_000,
    pi_present       = None,    # None=estimate from data; or pass a float
    gamma            = None,    # None=estimate from study structure; or pass a float
):
    """Per-SGB latent-L posterior detection + asymptote characterisation for every SGB."""
    max_b   = float(breadth_df.values[breadth_df.values > 0].max())               if (breadth_df.values > 0).any() else 0.0
    bfrac   = (breadth_df / 100.0) if max_b > 1.0 else breadth_df.copy()

    # Optionally restrict to one lifestyle
    meta_use = meta
    if lifestyle is not None:
        meta_use = meta[meta["MacroGroup"] == lifestyle]
        print(f"    Lifestyle filter: {lifestyle}  ({len(meta_use):,} samples)")

    samp    = [s for s in meta_use.index
               if s in bfrac.index and s in depth_df.index]
    bfrac_s = bfrac.loc[samp]
    depth_s = depth_df.loc[samp]
    meta_s  = meta_use.loc[samp]

    raw     = depth_s.copy()
    row_sum = raw.sum(axis=1).replace(0, np.nan)
    abund   = raw.div(row_sum, axis=0).fillna(0.0)
    mean_abund  = abund.mean(axis=0)
    detect_frac = (bfrac_s >= min_breadth_pct / 100.0).mean(axis=0)

    sgbs = [s for s in breadth_df.columns
            if s in bfrac_s.columns and s in depth_s.columns]

    if c_global is None:
        c_global = _estimate_coverage_efficiency(bfrac_s, depth_s)

    # ── Learned sigma_b(depth) model ────────────────────────────────────
    # Estimated from null-consistent (b ≈ b_null) residuals in log-depth bins.
    # Each SGB uses sigma_b looked up at its median depth; this replaces the
    # fixed sigma_b_ref=0.015; the per-sample learned model replaces this.
    print("    Estimating depth-stratified sigma_b from null-consistent residuals ...")
    sigma_bins, sigma_edges = _estimate_sigma_b(
        bfrac_s, depth_s, c=c_global,
        f_null=0.10, n_bins=12,   # fallback estimated from data internally
    )
    print(f"    sigma_b range: [{sigma_bins.min():.4f}, {sigma_bins.max():.4f}] "
          f"(fallback auto-derived from data)")

    # ── Require genome_lengths before any per-SGB computation ────────────
    if genome_lengths is None:
        raise ValueError(
            "compute_asymptote_fits: genome_lengths is required. "
            "Load all_SGB_lengths_*.txt and pass as a dict {sgb_id: length_bp}."
        )

    # ── Auto-estimate pi_present and gamma from the data ─────────────────
    if pi_present is None:
        pi_present = _estimate_pi_present(bfrac_s, genome_lengths, min_map_bp=min_map_bp)
        print(f"    Estimated pi_present = {pi_present:.3f}  "
              f"(fraction of pairs with b > min_b)")

    # Study labels per sample — passed into fit_sgb_asymptote for
    # study-aware aggregation (prevents correlated same-study samples
    # from inflating p_genuine).
    # Study labels for correlated-sample correction.
    # MacroGroup is deliberately excluded: lifestyle is already used for
    # subsetting, so using it as a "study" would collapse all Industrial
    # samples into one group, defeating the independence assumption.
    # If no genuine study column exists, study-aware aggregation is
    # disabled entirely (no study correction applied).
    _STUDY_COLS = ("StudyID", "Study", "cohort", "_study")
    study_col_fit = next((c for c in _STUDY_COLS if c in meta_s.columns), None)
    if study_col_fit is None:
        print("    NOTE: no study column found in meta; study-aware "
              "aggregation disabled (MacroGroup intentionally excluded).")
    study_arr_sgb = meta_s[study_col_fit].values if study_col_fit else None

    # gamma: estimate after study structure is known
    if gamma is None:
        # Use median genome size and the primary (10%) threshold for calibration
        G_med = int(np.median([
            int(genome_lengths.get(str(s)) or genome_lengths.get(s))
            for s in sgbs
        ]))
        alpha_tau_primary = _latent_L_prior_mass(0.10 * G_med, G_med,
                                                  pi_present=pi_present)
        gamma = _estimate_gamma(study_arr_sgb, alpha_tau_primary)
        n_st  = len(np.unique(study_arr_sgb)) if study_arr_sgb is not None else 0
        print(f"    Estimated gamma = {gamma:.3f}  "
              f"(from {n_st} studies, alpha_tau={alpha_tau_primary:.4f})")

    records = []
    for k, sgb in enumerate(sgbs):
        if k % 500 == 0:
            print(f"      fitting SGB {k}/{len(sgbs)} ...", flush=True)
        if str(sgb) not in genome_lengths and sgb not in genome_lengths:
            raise KeyError(
                f"SGB '{sgb}' not found in genome_lengths. "
                "Ensure all_SGB_lengths_*.txt covers all SGBs in the breadth table."
            )
        G_bp = int(genome_lengths.get(str(sgb)) or genome_lengths.get(sgb))
        # Look up scalar sigma at median depth (kept for back-compat / CSV output).
        # The full sigma_bins / sigma_edges arrays are also passed so the fitter
        # can assign each sample its own per-depth sigma instead of one global value.
        d_vec_sgb = depth_s[sgb].values
        d_med_sgb = float(np.median(d_vec_sgb[d_vec_sgb > 0])) \
                    if (d_vec_sgb > 0).any() else 0.0
        sigma_sgb = _lookup_sigma_b(d_med_sgb, sigma_bins, sigma_edges,
                                    fallback=sigma_b_ref)
        r = fit_sgb_asymptote(
            depth_s[sgb].values, bfrac_s[sgb].values,
            c=c_global,
            A_min=A_min,
            sigma_b_ref=sigma_sgb,
            sigma_bins=sigma_bins,
            sigma_edges=sigma_edges,
            z_thresh=z_thresh,
            eps_disc=eps_disc,
            G_genome=G_bp, min_map_bp=min_map_bp,
            study_arr=study_arr_sgb,
            pi_present=pi_present,
            gamma=gamma,
        )
        r["sigma_b_used"]  = sigma_sgb
        # Store the global sigma model in each row so _draw_curve_panel can
        # reproduce the exact per-sample ring sizes without re-estimating.
        r["_sigma_bins"]   = sigma_bins
        r["_sigma_edges"]  = sigma_edges
        r["sgb"]           = sgb
        r["c_global"]      = c_global
        r["mean_abund"]    = float(mean_abund.get(sgb, np.nan))
        r["detect_frac"]   = float(detect_frac.get(sgb, 0.0))
        records.append(r)

    result   = pd.DataFrame(records).set_index("sgb")

    # ── Study-level independence: n_studies_above_10pct ───────────────────
    # Count distinct studies with >= 1 sample geometrically above 10%-null.
    study_col = next((c for c in ("StudyID", "Study", "cohort", "_study")
                      if c in meta_s.columns), None)
    if study_col is not None:
        n_studies_map = {}
        for sgb in sgbs:
            if sgb not in bfrac_s.columns:
                n_studies_map[sgb] = 0
                continue
            b_vec  = bfrac_s[sgb].values
            d_vec  = depth_s[sgb].values
            G_sgb  = int(genome_lengths.get(str(sgb)) or genome_lengths.get(sgb))
            floor  = max(1e-4, min_map_bp / max(G_sgb, 1))
            f10    = 0.10
            b_n10  = f10 * (1.0 - np.exp(-c_global * d_vec / f10))
            above  = (b_vec > b_n10) & (b_vec >= floor)
            n_studies_map[sgb] = int(
                meta_s.loc[meta_s.index[above], study_col].nunique()
                if above.any() else 0
            )
        result["n_studies_above_10pct"] = pd.Series(n_studies_map)
    else:
        result["n_studies_above_10pct"] = 0

    n_det    = int(result["detectable"].sum())
    n_valid  = int(result["fit_ok"].sum())
    n_total  = len(sgbs)
    n_no_dat = n_total - n_valid
    ls_tag   = f" [{lifestyle}]" if lifestyle else ""
    eff_b    = result["effective_min_b"].median() if "effective_min_b" in result.columns else 1e-4
    n_envelope = int((result["fit_mode"] == "envelope").sum())                if "fit_mode" in result.columns else n_valid
    n_single   = int((result["fit_mode"] == "single_sample").sum())                if "fit_mode" in result.columns else 0
    med_pg     = float(result["p_genuine"].median()) \
                 if "p_genuine" in result.columns else float("nan")
    print(f"    Detected{ls_tag}: {n_det:,} / {n_total:,} SGBs  "
          f"(fit_mode — envelope:{n_envelope:,} single:{n_single:,} "
          f"absent:{n_no_dat:,})  "
          f"[p_genuine_10pct>=0.5, med={med_pg:.3f}]")
    # Per-null detection counts
    # ------------------------------------------------------------------
    # PRIMARY DETECTOR (drives the 'detectable' flag — Figure 13):
    #   p_genuine_{nm} >= 0.5  — latent-L hierarchical aggregate posterior.
    #   Within-study union of raw posteriors, across-study gamma-damped
    #   log-odds BF with size-normalised baseline.  Mixture prior pi=0.5.
    #   p_genuine_10pct is the primary score; the configured consistency thresholds are tiers.
    #
    # GEOMETRIC DIAGNOSTIC (n_above_{nm}, not used for detection):
    #   n_above_{nm} >= 1 — any sample with b > b_null_k AND b > floor.
    #   Useful as a quick sanity check; not a gate on detection.
    #
    # LEGACY HEURISTIC (p_detect_{nm}, kept for sensitivity analysis):
    #   Sigmoid-weighted score from the old z-score path. Fixed sigma_b;
    #   treat as a rough cross-check only.
    # ------------------------------------------------------------------
    print(f"    Per-null detection counts{ls_tag}:")
    for nm in ["1pct", "3pct", "10pct", "30pct", "50pct", "90pct", "10kb", "30kb", "100kb", "300kb", "500kb", "1Mb"]:
        col_ab  = f"n_above_{nm}"
        col_pg  = f"p_genuine_{nm}"     # null-specific posterior (now per-null)
        col_pd  = f"p_detect_{nm}"
        n_bin   = int((result[col_ab] >= 1).sum())  if col_ab in result.columns else 0
        n_pgen  = int((result[col_pg] >= 0.5).sum()) if col_pg in result.columns else 0
        n_prob  = int((result[col_pd] >= 0.5).sum()) if col_pd in result.columns else 0
        f_frag  = {"1pct": 0.01, "3pct": 0.03, "10pct": 0.10, "30pct": 0.30,
                   "50pct": 0.50, "90pct": 0.90, "10kb": None, "30kb": None,
                   "100kb": None, "300kb": None, "500kb": None, "1Mb": None}[nm]
        sat_str = f"{f_frag*100:.0f}%" if f_frag else nm
        print(f"      {nm:>6} (sat {sat_str:>5}): "
              f"n_above>=1={n_bin:>5,}  "
              f"p_genuine_{nm}>=0.5={n_pgen:>5,}  "
              f"p_detect>=0.5={n_prob:>5,}  / {n_total:,}")

    return result, sigma_bins, sigma_edges


def _pick_examples(fit_df, n_per_bin=4):
    """
    Select representative SGBs for the example-curves figure.

    Six rows chosen to reveal the detector's strengths and failure modes,
    rather than binning by abundance (which obscures disagreements):

    Row 1  detected_strong       — highest p_genuine (most convincing detections)
    Row 2  detected_sparse       — detected with fewest above-floor samples
                                   (rare-but-real: latent-L earning its keep)
    Row 3  undetected_high_Afit  — NOT detected but highest A_fit
                                   (key failure mode: high envelope but posterior fails)
    Row 4  undetected_borderline — NOT detected, highest p_genuine (closest misses)
    Row 5  disagree_geo_wins     — many samples geometrically above null but low
                                   p_genuine (geometric says yes, posterior says no)
    Row 6  disagree_post_wins    — detected with few geometric above-null samples
                                   (posterior says yes despite sparse geometric support)

    Parameters
    ----------
    fit_df    : DataFrame produced by compute_asymptote_fits, already filtered
                to fit_ok == True rows (fit_mode in envelope/single_sample).
    n_per_bin : int — columns per row.

    Returns
    -------
    dict : row_key -> [sgb_id, ...]
    """
    det  = fit_df[fit_df["detectable"] == True]
    ndet = fit_df[fit_df["detectable"] == False]

    def _top(sub, by, ascending=False, n=None):
        n = n or n_per_bin
        if sub.empty or by not in sub.columns:
            return []
        return list(sub.sort_values(by, ascending=ascending).index[:n])

    # Disagreement scores (requires both columns present)
    has_geo = "n_above_10pct" in fit_df.columns
    has_pg  = "p_genuine"     in fit_df.columns

    # Row 5: geometrically above null but low posterior
    #   → high n_above_10pct relative to p_genuine
    if has_geo and has_pg:
        ndet_has_geo = ndet[ndet["n_above_10pct"] > 0].copy()
        ndet_has_geo["_disagree_score"] = (
            ndet_has_geo["n_above_10pct"].rank(pct=True)
            - ndet_has_geo["p_genuine"].rank(pct=True)
        )
        geo_wins = _top(ndet_has_geo, "_disagree_score", ascending=False)
    else:
        geo_wins = []

    # Row 6: detected but few geometric above-null samples
    if has_geo:
        det_few_geo = det[det["n_above_10pct"] > 0].copy()
        post_wins   = _top(det_few_geo, "n_above_10pct", ascending=True)
    else:
        post_wins = []

    return {
        "detected_strong":      _top(det,  "p_genuine",      ascending=False),
        "detected_sparse":      _top(det,  "n_points",       ascending=True),
        "undetected_high_Afit": _top(ndet, "A_fit",          ascending=False),
        "undetected_borderline":_top(ndet, "p_genuine",      ascending=False),
        "disagree_geo_wins":    geo_wins,
        "disagree_post_wins":   post_wins,
    }


def _draw_curve_panel(ax, sgb, bfrac_s, depth_s, meta_s, fit_row,
                      vanish_set, blossum_set):
    """
    Scatter + fitted curves for one SGB.

    Visual layers (bottom to top):
    1. All (depth, breadth) pairs for this SGB, coloured by log10(sample depth).
    2. Per-null hollow-ring overlays on samples geometrically above that null.
       Ring size is proportional to geometric excess above the null, so the
       reader can see both WHICH samples are above-null and HOW STRONGLY.
    3. A_fit characterisation curve (green=detected, red=not) + asymptote line.
    4. Envelope diamonds (the above-null-10pct points used to fit A_fit).
    5. Log-log inset (bottom-right) with same layers.

    Fixed-size null labels include the genome-specific fraction so curves
    at very different genome sizes are interpretable (e.g. "100kb (3.3%)").
    """
    d_all = depth_s[sgb].values
    b_all = bfrac_s[sgb].values
    pos   = (d_all > 0) & (b_all > 0)

    if pos.sum() == 0:
        ax.text(0.5, 0.5, "no detections", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return

    # ── Layer 1: base scatter coloured by total sample depth ─────────────
    log_dep_all = np.log10(meta_s["_depth"].values)
    ax.scatter(
        d_all[pos], b_all[pos],
        c=log_dep_all[pos], cmap="viridis",
        s=4, alpha=0.30, linewidths=0, zorder=2,
        vmin=log_dep_all.min(), vmax=log_dep_all.max(),
    )

    D_ZOOM  = 5.0
    c_val   = fit_row.get("c_global", 1.0) or 1.0
    G_bp = fit_row.get("G_genome")
    if not G_bp:
        raise ValueError(
            f"_draw_curve_panel: G_genome missing for SGB '{sgb}'. "
            "fit_sgb_asymptote must be called with a valid genome_lengths dict."
        )
    G_bp = float(G_bp)
    floor   = float(fit_row.get("effective_min_b", 1e-4))
    # Use the learned depth-stratified sigma model when available.
    # _sigma_bins / _sigma_edges are stored in the fit row by compute_asymptote_fits.
    _sb   = fit_row.get("_sigma_bins")
    _se   = fit_row.get("_sigma_edges")
    _sfb  = float(fit_row.get("sigma_b_used", 0.015))
    d_rng = np.linspace(0, D_ZOOM, 400)

    # Per-null ring colours (darker than the null curve colour)
    NULL_RING = {
        "10pct": "#004f82",   # dark sky-blue (palette["sky"] darkened)
        "30pct": "#444444",   # dark grey (neutral)
        "100kb": "#7a5000",   # dark orange (palette["orange"] darkened)
        "300kb": "#7a3b60",   # dark pink (palette["pink"] darkened)
    }

    # ── Layer 2: null curves + geometric above-null overlay ──────────────
    # For each null model, draw the null curve and overlay hollow rings on
    # samples geometrically above that null AND above the breadth floor.
    # Ring size scales with z-score (depth-normalised residual) as a
    # visual display heuristic — larger rings = further above null.
    # NOTE: these rings are NOT the detector.  Detection is driven by the
    # latent-L posterior via _hierarchical_aggregate (Layer 4 / orange diamonds).
    # The geometric rings help locate where breadth is elevated; they can
    # appear even when p_genuine_10pct is low (and vice versa).
    for nm, (col_n, ls_n, lw_n, _) in NULL_STYLES.items():
        f_fixed, kb = NULL_MODELS[nm]
        f_frag = f_fixed if f_fixed is not None \
                 else min(1.0, kb * 1000 / max(G_bp, 1))

        # Null curve label: fixed-size nulls include the genome-specific fraction
        if kb is not None:
            frac_pct = 100.0 * kb * 1000 / max(G_bp, 1)
            null_lbl = f"null {kb} kb ({frac_pct:.1f}%)"
        else:
            null_lbl = f"null {int(f_frag*100)}%"

        ax.plot(d_rng, _fragment_null_breadth(d_rng, f_frag, c_val),
                color=col_n, lw=lw_n, ls=ls_n, alpha=0.75,
                label=null_lbl, zorder=3)

        # Per-sample above-null test; ring size uses per-sample learned sigma
        b_null_s  = _fragment_null_breadth(d_all[pos], f_frag, c_val)
        above     = (b_all[pos] > b_null_s) & (b_all[pos] >= floor)
        if above.any():
            d_ab = d_all[pos][above]
            b_ab = b_all[pos][above]
            # Per-sample sigma: look up learned model when available
            if _sb is not None and _se is not None:
                sig_ab = np.array([_lookup_sigma_b(d, _sb, _se, _sfb) for d in d_ab])
            else:
                sig_ab = np.full(len(d_ab), _sfb)
            sig_ab = np.maximum(sig_ab, 1e-9)
            z_above = (b_ab - b_null_s[above]) / sig_ab
            # Ring size: floor at 10px (z≈0), grows with z, cap at 80px
            sizes = np.clip(10 + z_above * 7, 10, 80)
            ax.scatter(
                d_ab, b_ab,
                s=sizes, marker="o",
                facecolors="none", edgecolors=NULL_RING[nm],
                linewidths=0.8, alpha=0.85, zorder=5,
            )

    # ── Layer 3: A_fit characterisation curve ────────────────────────────
    col = PALETTE["green"] if fit_row.get("detectable") else PALETTE["vermillion"]
    if fit_row.get("has_data") and not np.isnan(fit_row.get("A_fit", np.nan)):
        A = fit_row["A_fit"]
        ax.plot(d_rng, _poisson_asymptote(d_rng, A, c_val),
                color=col, lw=1.8, zorder=6,
                label=f"A_fit={A:.2f} (char.)")
        ax.axhline(A, color=col, lw=0.7, ls=":", alpha=0.45, zorder=6)

        # Discrimination depth d*
        d_star = fit_row.get("d_star", np.nan)
        if np.isfinite(d_star) and 0 < d_star < D_ZOOM:
            ax.axvline(d_star, color="#9467bd", lw=0.9, ls=":",
                       alpha=0.80, label=f"d*={d_star:.2f}")

        # ── Layer 4: latent-L 10pct contributing samples ────────────────
        # Samples with per-sample 10pct posterior > 0.5 — the same gate as
        # the detectable threshold.  Ring size ∝ p_genuine_10pct per sample.
        env_d = fit_row.get("env_d")
        env_b = fit_row.get("env_b")
        env_p = fit_row.get("env_p")
        if env_d is not None and env_b is not None:
            mask = env_d <= D_ZOOM
            if mask.any():
                if env_p is not None and len(env_p) == len(env_d):
                    sizes_env = np.clip(8 + env_p[mask] * 40, 8, 60)
                else:
                    sizes_env = 18
                ax.scatter(env_d[mask], env_b[mask],
                           s=sizes_env, zorder=8,
                           marker="D", facecolors="none",
                           edgecolors="#ff7f0e", linewidths=0.7,
                           label="latent-L 10pct contributors (p>0.5)")

    ax.set_xlim(0, D_ZOOM)
    ax.set_ylim(-0.02, 1.05)

    # ── Layer 5: Study-evidence bar ───────────────────────────────────────
    # Show the BF contribution of each study that fed into the hierarchical
    # aggregate.  This directly explains why a panel is green or red, in a
    # way that per-sample diamonds cannot (a panel can be green with no
    # diamonds if several studies each contribute moderate positive BF).
    #
    # Each study gets a small rectangle in the figure border region:
    #   green  (BF_damped > 0) — study contributed positive evidence
    #   red    (BF_damped < 0) — study contributed negative evidence (damped)
    #   grey   (BF_damped ≈ 0) — study was approximately neutral
    #
    # Rectangles are stacked horizontally; width ∝ |BF_damped|, height fixed.
    study_contribs = fit_row.get("study_contributions_10pct", [])
    if study_contribs:
        ax_bar = ax.inset_axes([0.0, -0.13, 1.0, 0.09])   # below main axes
        ax_bar.set_xlim(0, 1); ax_bar.set_ylim(0, 1)
        ax_bar.set_axis_off()

        max_abs_bf = max((abs(s["bf_damped"]) for s in study_contribs), default=1e-9)
        max_abs_bf = max(max_abs_bf, 1e-9)

        n_studies = len(study_contribs)
        x_cursor  = 0.0
        x_total   = 1.0
        bar_h     = 0.7  # fraction of the inset height

        for sc in sorted(study_contribs, key=lambda s: -s["bf_damped"]):
            bf     = sc["bf_damped"]
            width  = max(abs(bf) / max_abs_bf * x_total / n_studies, 0.005)
            color  = (PALETTE["green"] if bf > 0.05
                      else PALETTE["vermillion"] if bf < -0.05
                      else "#bbbbbb")
            from matplotlib.patches import FancyBboxPatch
            rect = FancyBboxPatch((x_cursor, 0.15), width, bar_h,
                                  boxstyle="round,pad=0.005",
                                  facecolor=color, edgecolor="none", alpha=0.75)
            ax_bar.add_patch(rect)
            x_cursor += width + 0.002

        # Count summaries
        n_pos  = sum(1 for s in study_contribs if s["bf_damped"] > 0.05)
        n_neg  = sum(1 for s in study_contribs if s["bf_damped"] < -0.05)
        n_neu  = len(study_contribs) - n_pos - n_neg
        ax_bar.text(0.0, -0.2,
                    f"Studies: {n_pos}↑ {n_neg}↓ {n_neu}≈0  "
                    f"(BF: green=positive, red=negative, grey=neutral)",
                    fontsize=4.5, color="#444", va="top", ha="left",
                    transform=ax_bar.transAxes)

    # ── Panel title ───────────────────────────────────────────────────────
    cat    = ("VANISH"  if sgb in vanish_set  else
              "BloSSUM" if sgb in blossum_set else "")
    n_ab   = fit_row.get("n_above_10pct", fit_row.get("n_bent", 0))
    p_gen  = fit_row.get("p_genuine", fit_row.get("p_combined", 0.0))
    f_mode = fit_row.get("fit_mode", "?")
    A_val  = fit_row.get("A_fit", float("nan"))
    A_str  = f"A={A_val:.2f}" if not np.isnan(A_val) else "A=?"
    status = "DETECTED" if fit_row.get("detectable") else "not detected"
    ax.set_title(
        f"{sgb}{(' ['+cat+']') if cat else ''}\n"
        f"{A_str}  n={n_ab}  p10={p_gen:.2f}  [{f_mode}]  {status}  n={fit_row['n_points']}",
        fontsize=6.0, pad=2,
        color=col,
    )
    ax.set_xlabel("Depth (mean coverage)", fontsize=6)
    ax.set_ylabel("Breadth (fraction)", fontsize=6)
    ax.tick_params(labelsize=5.5)

    # ── Layer 5: Log-log inset (bottom-right) ────────────────────────────
    axz = ax.inset_axes([0.52, 0.02, 0.46, 0.44])
    det_ok = pos & (b_all >= 0.005) & (d_all >= 0.005)
    if det_ok.sum() > 0:
        axz.scatter(np.log10(d_all[det_ok]), np.log10(b_all[det_ok]),
                    c=log_dep_all[det_ok], cmap="viridis",
                    s=3, alpha=0.25, linewidths=0, zorder=2,
                    vmin=log_dep_all.min(), vmax=log_dep_all.max())

    if fit_row.get("has_data") and not np.isnan(fit_row.get("A_fit", np.nan)):
        A_z   = fit_row["A_fit"]
        col_z = "#2ca02c" if fit_row.get("detectable") else "#d62728"
        ld    = np.linspace(-2.5, 1.0, 300)
        d_z   = 10 ** ld

        # A_fit curve
        bz  = _poisson_asymptote(d_z, A_z, c_val)
        axz.plot(ld[bz > 0.001], np.log10(bz[bz > 0.001]),
                 color=col_z, lw=1.2, zorder=6)

        # A=0.30 reference (labelled as reference, not gate)
        bm  = _poisson_asymptote(d_z, 0.30, c_val)
        axz.plot(ld[bm > 0.001], np.log10(bm[bm > 0.001]),
                 color="#999999", lw=0.7, ls="--", alpha=0.6)

        # p_genuine contributors in log-log inset (open diamonds)
        env_d_z = fit_row.get("env_d"); env_b_z = fit_row.get("env_b")
        env_p_z = fit_row.get("env_p")
        if env_d_z is not None and env_b_z is not None:
            oe = (env_d_z > 0.005) & (env_b_z > 0.005)
            if oe.any():
                axz.scatter(np.log10(env_d_z[oe]), np.log10(env_b_z[oe]),
                            s=6, zorder=8, marker="D",
                            facecolors="none", edgecolors="#ff7f0e",
                            linewidths=0.4)

        # Null curves in log-log
        ld_g = np.linspace(-2.5, 1.0, 200); d_g = 10**ld_g
        for nm_g, (col_g, ls_g, lw_g, _) in NULL_STYLES.items():
            f_fixed_g, kb_g = NULL_MODELS[nm_g]
            f_g = f_fixed_g if f_fixed_g is not None \
                  else min(1.0, kb_g*1000/max(G_bp,1))
            b_g = _fragment_null_breadth(d_g, f_g, c_val)
            axz.plot(ld_g[b_g > 0.001], np.log10(b_g[b_g > 0.001]),
                     color=col_g, lw=lw_g, ls=ls_g, alpha=0.8, zorder=3)

        # Highlight above-10%-null samples in log-log inset
        f10 = 0.10
        if det_ok.sum() > 0:
            b_n10_z   = _fragment_null_breadth(d_all[det_ok], f10, c_val)
            above_10z = (b_all[det_ok] > b_n10_z) & (b_all[det_ok] >= floor)
            if above_10z.any():
                axz.scatter(
                    np.log10(d_all[det_ok][above_10z]),
                    np.log10(b_all[det_ok][above_10z]),
                    s=7, marker="o", facecolors="none",
                    edgecolors=NULL_RING["10pct"],
                    linewidths=0.5, alpha=0.85, zorder=6,
                )

    axz.set_xlim(-2.5, 1.0); axz.set_ylim(-2.5, 0.15)
    axz.set_xlabel("log10(d)", fontsize=4.5, labelpad=1)
    axz.set_ylabel("log10(b)", fontsize=4.5, labelpad=1)
    axz.tick_params(labelsize=4)
    for sp in axz.spines.values():
        sp.set_linewidth(0.5); sp.set_edgecolor("#666666")
    axz.set_facecolor("#f8f8f8")


def plot_detection_heatmap(
    fit_df:         pd.DataFrame,
    bfrac_s:        pd.DataFrame,
    depth_s:        pd.DataFrame,
    out_path:       str,
    n_depth_bins:   int   = 10,
    n_breadth_bins: int   = 12,
    lifestyle_label: str  = "",
):
    """
    2D diagnostic: detection rate and mean p_genuine as a heatmap over
    (mean depth, mean breadth) space.

    Motivation
    ----------
    The non-monotone detection pattern at 1–5% breadth is either:
      a) A projection artifact — the 2D detector is calibrated correctly
         in (depth, breadth) space but looks non-monotone in breadth alone
         because breadth at low depth is less informative than at high depth.
      b) A calibration error — the detector genuinely over-fires at 1% breadth
         across ALL depths.

    Interpretation
    --------------
    If the 1% shoulder disappears after conditioning on depth (i.e., within
    each depth stripe the detection rate is monotone in breadth), it is a
    projection artifact and the model is well-calibrated.  If it persists
    within depth bins, it is a real calibration problem.

    Parameters
    ----------
    fit_df   : per-SGB result DataFrame from compute_asymptote_fits.
               Must contain 'p_genuine' and 'detectable' columns.
    bfrac_s  : [samples × SGBs] breadth fraction DataFrame.
    depth_s  : [samples × SGBs] depth DataFrame.
    out_path : output figure path.
    """
    import warnings
    if "p_genuine" not in fit_df.columns:
        warnings.warn("plot_detection_heatmap: 'p_genuine' not in fit_df — skipping.")
        return

    sgbs    = [s for s in fit_df.index if s in bfrac_s.columns and s in depth_s.columns]
    if not sgbs:
        warnings.warn("plot_detection_heatmap: no SGBs overlap fit_df and bfrac_s — skipping.")
        return

    # Per-SGB summary statistics: mean depth and mean breadth (non-zero samples only)
    b_vals = bfrac_s[sgbs].values        # (n_samp, n_sgb)
    d_vals = depth_s[sgbs].values        # (n_samp, n_sgb)

    # Mask: only samples where depth > 0 contribute
    pos_mask = (d_vals > 0) & (b_vals >= 0)
    with np.errstate(invalid="ignore"):
        mean_b = np.where(pos_mask.any(axis=0),
                          np.nanmean(np.where(pos_mask, b_vals, np.nan), axis=0), np.nan)
        mean_d = np.where(pos_mask.any(axis=0),
                          np.nanmean(np.where(pos_mask, d_vals, np.nan), axis=0), np.nan)

    p_gen   = fit_df.loc[sgbs, "p_genuine"].values.astype(float)
    is_det  = fit_df.loc[sgbs, "detectable"].values.astype(bool)

    # Remove NaN
    valid = np.isfinite(mean_b) & np.isfinite(mean_d) & np.isfinite(p_gen)
    mean_b = mean_b[valid]; mean_d = mean_d[valid]
    p_gen  = p_gen[valid];  is_det = is_det[valid]

    if len(mean_b) < 5:
        warnings.warn("plot_detection_heatmap: fewer than 20 SGBs with valid data — skipping.")
        return

    # Log-spaced bin edges (depth and breadth are typically log-distributed)
    d_edges = np.percentile(mean_d, np.linspace(0, 100, n_depth_bins + 1))
    b_edges = np.percentile(mean_b, np.linspace(0, 100, n_breadth_bins + 1))
    d_edges = np.unique(d_edges); b_edges = np.unique(b_edges)

    # Assign each SGB to a bin
    d_idx = np.searchsorted(d_edges[1:], mean_d, side="right")
    b_idx = np.searchsorted(b_edges[1:], mean_b, side="right")
    d_idx = np.clip(d_idx, 0, len(d_edges) - 2)
    b_idx = np.clip(b_idx, 0, len(b_edges) - 2)

    nd = len(d_edges) - 1; nb = len(b_edges) - 1
    det_rate = np.full((nd, nb), np.nan)
    p_mean   = np.full((nd, nb), np.nan)
    n_count  = np.zeros((nd, nb), dtype=int)

    for di in range(nd):
        for bi in range(nb):
            mask = (d_idx == di) & (b_idx == bi)
            n = mask.sum()
            n_count[di, bi] = n
            if n >= 3:
                det_rate[di, bi] = is_det[mask].mean()
                p_mean[di, bi]   = p_gen[mask].mean()

    # ── Figure: two heatmaps side-by-side ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f"Detection rate in (depth, breadth) space  {lifestyle_label}\n"
        f"Monotone within depth stripes → projection artifact; "
        f"shoulder persists → calibration issue",
        fontsize=10, fontweight="bold",
    )

    def _label_ax(ax, data, title, cmap, vmin, vmax):
        im = ax.imshow(data, aspect="auto", origin="lower",
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # x-axis: breadth bins (columns)
        b_mids = 0.5 * (b_edges[:-1] + b_edges[1:])
        d_mids = 0.5 * (d_edges[:-1] + d_edges[1:])
        ax.set_xticks(range(nb))
        ax.set_xticklabels([f"{v:.3f}" for v in b_mids], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(nd))
        ax.set_yticklabels([f"{v:.2f}" for v in d_mids], fontsize=7)
        ax.set_xlabel("Mean breadth fraction (per SGB)", fontsize=9)
        ax.set_ylabel("Mean depth (per SGB)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        # Annotate cells with count
        for di in range(nd):
            for bi in range(nb):
                n = n_count[di, bi]
                if n >= 3:
                    ax.text(bi, di, str(n), ha="center", va="center",
                            fontsize=5.5, color="white",
                            fontweight="bold" if n >= 10 else "normal")

    _label_ax(axes[0], det_rate, "Fraction detected (p_genuine ≥ 0.5)",
              "RdYlGn", 0, 1)
    _label_ax(axes[1], p_mean, "Mean p_genuine",
              "viridis", 0, 1)

    plt.tight_layout()
    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path}")



def plot_null_stats(fit_df, out_path, lifestyle_label=''):
    """
    Per-null detection histogram figure.

    Left column  — PRIMARY SCORE: p_genuine_{nm}
        Latent-L union posterior P(L >= tau | data) aggregated via
        _hierarchical_aggregate (gamma-damped study-level log-odds BF, γ=0.15).
        detectable flag is set by p_genuine_10pct >= 0.5.
        The configured consistency-threshold columns are support tiers only.

    Right column — LEGACY SCORE: p_detect_{nm}  (diagnostic only)
        Older sigmoid-over-z_thresh heuristic; kept as a sensitivity
        check.  Do not use for primary detection decisions.
    """
    nrows, ncols = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 14))
    title_str = ('Per-null latent-L posteriors  ' + lifestyle_label + '\n'
                 'Left: p_genuine_{nm} (latent-L, gamma-damped study BF, detectable = 10pct >= 0.5)  '
                 'Right: p_detect_{nm} (legacy sigmoid, diagnostic only)\n'
                 'A_cv / gini_A: CSV diagnostics only — not applied to detection rule')
    fig.suptitle(title_str, fontsize=10, fontweight='bold')
    null_meta = [
        ('10pct',  '10% of genome',    PALETTE["sky"]),
        ('30pct',  '30% of genome',    '#888888'),
        ('100kb',  '100 kb fragment',  PALETTE["orange"]),
        ('300kb',  '300 kb fragment',  PALETTE["pink"]),
    ]
    for row, (nm, label, col) in enumerate(null_meta):
        col_pd = f'p_detect_{nm}'
        col_bn = f'n_bent_{nm}'  # kept for backwards-compat; not used as gate

        # ── Primary: p_genuine_{nm} — latent-L posterior, gamma-damped study BF
        col_pg_nm = f'p_genuine_{nm}'
        ax = axes[row, 0]
        if col_pg_nm in fit_df.columns:
            pg_vals = fit_df[col_pg_nm].fillna(0)
            n_det   = int((pg_vals >= 0.5).sum()); n_tot = len(fit_df)
            ax.hist(pg_vals, bins=40, color=col, alpha=0.75, edgecolor='white', lw=0.3)
            ax.axvline(0.5, color='k', lw=0.9, ls='--', alpha=0.7, label='threshold 0.5')
            ax.set_xlabel('p_genuine (latent-L hierarchical posterior, γ=0.15)', fontsize=8)
            ax.set_xlim(0, 1)
            pct = 100 * n_det / max(n_tot, 1)
            ax.set_title(f'{label} — p_genuine_{nm}\n'
                         f'>=0.5: {n_det:,}/{n_tot:,} ({pct:.1f}%)',
                         fontsize=8, color=col)
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, f'{col_pg_nm} not in fit_df', ha='center',
                    transform=ax.transAxes, fontsize=8)

        # ── Probabilistic (soft score, heuristic): p_detect >= 0.5 ───────
        ax = axes[row, 1]
        if col_pd in fit_df.columns:
            p_vals = fit_df[col_pd].fillna(0)
            n_prob = int((p_vals >= 0.5).sum())
            ax.hist(p_vals, bins=40, color=col, alpha=0.75, edgecolor='white', lw=0.3)
            ax.axvline(0.5, color='k', lw=0.8, ls='--', alpha=0.6, label='p=0.5')
            ax.set_xlabel('p_detect (heuristic, fixed σ_b)', fontsize=8)
            ax.set_xlim(0, 1)
            pct2 = 100 * n_prob / max(len(fit_df), 1)
            ax.set_title(f'{label} — p_detect (legacy heuristic, sigmoid)\n'
                         f'p≥0.5: {n_prob:,}/{len(fit_df):,} ({pct2:.1f}%)',
                         fontsize=8, color=col)
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, f'{col_pd} not in fit_df', ha='center',
                    transform=ax.transAxes, fontsize=8)

        for ax_l in axes[row]:
            ax_l.tick_params(labelsize=7)
            ax_l.set_ylabel('SGBs', fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved: {out_path}')

def plot_asymptote_consistency(
    fit_df,
    breadth_df,
    depth_df,
    meta,
    vanish_set,
    blossum_set,
    out_path_stats,
    out_path_examples,
    out_path_csv,
    n_label   = 20,
    n_per_bin = 4,
):
    """
    Figure A (stats): 1 row x 3 cols
      Col 0: mean_abund vs A_fit scatter
      Col 1: ranked A_fit
      Col 2: violin of A_fit by category (detected / not)

    Figure B (examples): 5 rows x n_per_bin cols
      Rows: detected-high / detected-med / detected-low /
            undetected-med / undetected-low
      Each panel: breadth vs depth scatter + fitted curve for one SGB.
      Green = detectable, Red = not detectable.
    """
    from matplotlib.patches import Patch

    def _cat(sgb):
        if sgb in vanish_set:  return "VANISH"
        if sgb in blossum_set: return "BloSSUM"
        return "Other"

    CAT_COLORS = {"VANISH": PALETTE["vermillion"], "BloSSUM": PALETTE["blue"], "Other": "#aaaaaa"}

    fit_df = fit_df.copy()
    fit_df["cat"] = [_cat(s) for s in fit_df.index]
    # valid_afit: SGBs with a reliable A_fit for scatter/rank plots
    # valid_det:  ALL SGBs with any data — the correct denominator for detection %
    valid      = fit_df["A_fit"].notna() & fit_df["fit_ok"]
    valid_det  = fit_df["has_data"].fillna(False)  # includes floor-fit rows

    max_b   = float(breadth_df.values[breadth_df.values > 0].max())               if (breadth_df.values > 0).any() else 0.0
    bfrac   = (breadth_df / 100.0) if max_b > 1.0 else breadth_df.copy()
    samp    = [s for s in meta.index
               if s in bfrac.index and s in depth_df.index]
    bfrac_s = bfrac.loc[samp]
    depth_s = depth_df.loc[samp]
    meta_s  = meta.loc[samp]

    # ================================================================
    # Figure A: statistics
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(
        "Per-SGB breadth-depth curves.  "
        "4 fragment-null models (1%, 3%, 10%, 30%, 50%, 90%, 10kb, 30kb, 100kb, 300kb, 500kb, 1Mb)\n"
        "Green = detected (p_genuine_10pct \u2265 0.5; latent-L hierarchical aggregate)  "
        "A_fit shown for characterisation only — not used as detection gate",
        fontsize=11, fontweight="bold",
    )

    # -- Col 0: abundance vs A_fit --
    ax = axes[0]
    for cat in ["Other", "VANISH", "BloSSUM"]:
        sub = fit_df[valid & (fit_df["cat"] == cat)]
        if sub.empty: continue
        style = SGB_CAT_STYLE[cat].copy()
        ax.scatter(sub["mean_abund"], sub["A_fit"],
                   label=f"{cat} (n={len(sub):,})", **style)

    ax.axhline(0.30, color="#e08000", lw=0.8, ls=":", alpha=0.5,
               label="A=0.30 reference (not a detection gate)")

    bx, bmed, blo, bhi = _binned_trend(
        fit_df.loc[valid, "mean_abund"].values,
        fit_df.loc[valid, "A_fit"].values, n_bins=20)
    if len(bx) > 1:
        ax.plot(bx, bmed, color="#333", lw=2.0, zorder=6, label="All-SGB median")
        ax.fill_between(bx, blo, bhi, color="#888", alpha=0.15)

    hl = fit_df[valid & fit_df["cat"].isin(["VANISH", "BloSSUM"])]
    for sgb in hl["A_fit"].nlargest(n_label).index:
        r = fit_df.loc[sgb]
        ax.annotate(sgb, xy=(r["mean_abund"], r["A_fit"]),
                    xytext=(4, 2), textcoords="offset points",
                    fontsize=3.8, alpha=0.8, color="#111",
                    arrowprops=dict(arrowstyle="-", color="#ccc", lw=0.4))

    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Mean relative abundance", fontsize=9)
    ax.set_ylabel("Fitted asymptote A  (fraction of genome)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.tick_params(labelsize=8)

    n_det  = int(fit_df["detectable"].sum())
    n_tot  = int(valid_det.sum())           # all SGBs with data, not just fit_ok
    n_afit = int(valid.sum())               # SGBs with reliable A_fit (shown above)
    med_p  = float(fit_df["p_genuine"].median()) if "p_genuine" in fit_df.columns else float("nan")
    ax.text(0.98, 0.02,
            f"Detected (p_genuine_10pct >= 0.5): {n_det:,} / {n_tot:,}  "
            f"({100*n_det/max(n_tot,1):.1f}%)\n"
            f"A_fit scatter: {n_afit:,} SGBs (fit_ok only)  |  med p_genuine={med_p:.3f}",
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))

    # -- Col 1: ranked A_fit --
    ax = axes[1]
    df_s = fit_df[valid].sort_values("A_fit", ascending=False).reset_index()
    ranks = np.arange(1, len(df_s) + 1)
    for cat in ["Other", "VANISH", "BloSSUM"]:
        mask = df_s["cat"] == cat
        if not mask.any(): continue
        style = SGB_CAT_STYLE[cat].copy()
        ax.scatter(ranks[mask.values], df_s.loc[mask, "A_fit"].values,
                   label=cat, **style)
    ax.axhline(0.30, color="#e08000", lw=1.2, ls="--", alpha=0.8, label="A=0.30")
    ax.set_xscale("log"); ax.set_xlim(left=1)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("SGB rank (by A_fit, descending)", fontsize=9)
    ax.set_ylabel("Fitted asymptote A", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.tick_params(labelsize=8)

    # -- Col 2: violin by category and detection status --
    ax = axes[2]
    offsets = {"detected": -0.18, "not_detected": 0.18}
    for xi, cat in enumerate(["Other", "VANISH", "BloSSUM"]):
        for det_val, offset, alpha in [(True, -0.18, 0.75), (False, 0.18, 0.40)]:
            vals = fit_df.loc[
                valid & (fit_df["cat"] == cat) & (fit_df["detectable"] == det_val),
                "A_fit"].dropna().values
            if len(vals) < 2: continue
            vp = ax.violinplot(vals, positions=[xi + offset], widths=0.28,
                               showmedians=True, showextrema=False)
            for pc in vp["bodies"]:
                pc.set_facecolor(CAT_COLORS[cat]); pc.set_alpha(alpha)
            vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(2)

    ax.axhline(0.30, color="#e08000", lw=1.2, ls="--", alpha=0.8)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Other", "VANISH", "BloSSUM"], fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Fitted asymptote A", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.axhline(0.30, color="#e08000", lw=1.2, ls="--", alpha=0.8)
    ax.legend(handles=[Patch(fc="#888", alpha=0.75, label="detectable"),
                        Patch(fc="#888", alpha=0.40, label="not detectable")],
              fontsize=7.5, loc="lower right")

    plt.tight_layout()
    apply_display_text(fig)
    fig.savefig(out_path_stats, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path_stats}")

    # ================================================================
    # Figure B: example curves
    # ================================================================
    row_labels = {
        "detected_strong":       "Detected — strongest posterior",
        "detected_sparse":       "Detected — fewest above-floor samples (rare-but-real)",
        "undetected_high_Afit":  "NOT detected — highest A_fit (high envelope, posterior fails)",
        "undetected_borderline": "NOT detected — closest misses (highest p_genuine < 0.5)",
        "disagree_geo_wins":     "Disagreement: many above-null geometrically, low posterior",
        "disagree_post_wins":    "Disagreement: detected despite sparse geometric support",
    }
    # Pass valid_det (has_data) not valid (fit_ok): floor-fit not-detected
    # rows are the most clearly absent SGBs and must appear in the
    # "undetected" example rows.  The A_fit scatter/rank/violin above
    # correctly continue to use `valid` (fit_ok only).
    examples = _pick_examples(fit_df[valid_det], n_per_bin=n_per_bin)
    n_rows   = len(examples)
    n_cols   = max((len(v) for v in examples.values()), default=1)

    fig2, axes2 = plt.subplots(n_rows, n_cols,
                                figsize=(4.5 * n_cols, 4.0 * n_rows),
                                squeeze=False)
    fig2.suptitle(
        "Breadth vs depth per SGB — 4 fragment-null curves: 1%, 3%, 10%, 30%, 50%, 90%, 10kb, 30kb, 100kb, 300kb, 500kb, 1Mb\n"
        "Green = detected (p_genuine_10pct \u2265 0.5; latent-L posterior, "
        "gamma-damped study-level BF)   Red = not detected\n"
        "Rows include floor-fit absent SGBs (no A_fit) — detector applies to all SGBs with data",
        fontsize=9, fontweight="bold", y=1.01,
    )

    for r_idx, (cat_key, sgb_list) in enumerate(examples.items()):
        row_label = row_labels.get(cat_key, cat_key)
        for c_idx in range(n_cols):
            ax = axes2[r_idx, c_idx]
            if c_idx >= len(sgb_list):
                ax.axis("off"); continue
            sgb = sgb_list[c_idx]
            if sgb not in fit_df.index or sgb not in bfrac_s.columns:
                ax.axis("off"); continue
            _draw_curve_panel(ax, sgb, bfrac_s, depth_s, meta_s,
                              fit_df.loc[sgb].to_dict(),
                              vanish_set, blossum_set)
            if c_idx == 0:
                ax.set_ylabel(f"{row_label}\\nBreadth (fraction)", fontsize=7)

    plt.tight_layout()
    fig2.savefig(out_path_examples, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"    Saved: {out_path_examples}")

    # CSV (optional — may already be saved as combined file)
    if out_path_csv:
        fit_df.reset_index().to_csv(out_path_csv, index=False)
        print(f"    Saved CSV: {out_path_csv}")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure 14 — SGB accumulation curves with bootstrap resampling
#
#  For each null model, estimate how many SGBs are "detected" as more samples
#  are added, using without-replacement (rarefaction) bootstrap to average over
#  orderings.  Fit Michaelis-Menten (Hill) and Richards accumulation models
#  with S_max constrained to the reference-catalogue size, and extrapolate.
#  A second figure shows how the estimated asymptote shifts as more samples are
#  included in the fit (diagnostic for convergence / saturation).
#
#  Per-sample accumulation criterion (Figure 14)
#  ───────────────────────────────────────────────
#  Uses prior-corrected excess posteriors from _compute_per_sample_posterior
#  (mixture-prior latent-L, prior-subtracted).  This is a softer quantity
#  than the gamma-damped study-level detector in fit_sgb_asymptote (Figure 13);
#  accumulation panels show E[S(k)] = Σ_j(1-prod(1-p_ij)) which is an
#  exploratory upper bound, not the same threshold as p_genuine_10pct >= 0.5.
#  p_genuine >= 0.5 (multi-sample soft posterior) drives the SGB-level call.
#
#  Resampling: WITHOUT replacement (posterior-weighted rarefaction)
#  ────────────────────────────────────────────────────────────────
#  Each iteration is a random permutation of the N sample indices.
#  The accumulation is now POSTERIOR-WEIGHTED:
#      E[S(k | perm)] = sum_j  1 - prod_{t=1}^{k} (1 - p_{perm[t], j})
#  rather than a boolean union of hard detections.  This uses all the
#  probabilistic evidence in p_ij instead of discarding it into 0/1.
#  Confidence intervals are empirical percentiles across orderings (ordering
#  uncertainty); the ribbon width reflects how much the curve depends on
#  which samples happen to appear first.
#
#  Accumulation models (S_max constrained to S_ref)
#  ─────────────────────────────────────────────────
#  1. Hill (generalised Michaelis-Menten):
#        S(n) = S_max · n^h / (K^h + n^h)
#     Parameters: S_max (asymptote), K (half-saturation), h (Hill exponent).
#     h=1 → classical MM.  h>1 → sigmoidal.  h<1 → sub-linear concave.
#
#  2. Richards (generalised Chapman-Richards / Weibull):
#        S(n) = S_max · (1 − exp(−b · n^δ))
#     Parameters: S_max, b (rate), δ (shape).
#     δ=1 → negative exponential (always concave).
#     δ>1 → sigmoidal (inflection point; fits datasets that start slowly).
#     δ<1 → faster initial rise.
#
#  Both use scipy.optimize.minimize with bounds:
#        0 < S_max ≤ S_ref,  K > 0,  b > 0,  0.1 ≤ δ ≤ 5
#
#  Asymptote-vs-n diagnostic
#  ──────────────────────────
#  At each prefix length k = k_min … N, both models are fitted to the mean
#  accumulation curve (x=1…k, y=S_mean(k)).  The resulting S_max(k) is plotted
#  against k to reveal whether the estimate has converged.
# ══════════════════════════════════════════════════════════════════════════════

def _compute_per_sample_posterior(
    bfrac_s:        pd.DataFrame,
    depth_s:        pd.DataFrame,
    c:              float,
    sigma_b_ref:    float = 0.015,
    sigma_bins:     np.ndarray = None,   # learned depth-stratified sigma
    sigma_edges:    np.ndarray = None,   # log10(depth) bin edges
    min_map_bp:     int   = 10_000,
    genome_lengths: dict  = None,
    pi_present:     float = 0.5,
    subtract_prior: bool  = True,        # if False, return raw posteriors (not excess)
) -> dict:
    """
    Per-sample latent-L posteriors P(L >= tau | b_ij, d_ij, G_j) for all
    (sample, SGB) pairs, using the mixture-prior Beta-likelihood model.

    Model (per sample i, SGB j, threshold k):
        b_ij | L, d ~ Beta(mu*phi, (1-mu)*phi)  [Beta likelihood]
        L ~ mixture(absent@L_min, log-Uniform([L_min, G]))  [mixture prior]
        pi_present controls absent/present weight (default 0.5)
        p_ij_k = P(L >= tau_k | b_ij, d_ij, G_j)   [_latent_L_posterior_vec]

    After computing raw posteriors, prior-subtraction is applied for use in
    accumulation curves:
        alpha_k = pi_present * P_log-uniform(L >= tau_k)
        p_excess_ij_k = max(0, (p_ij_k - alpha_k) / (1 - alpha_k))

    Returns a dict null_name -> DataFrame[samples x SGBs] of float32 posteriors.

    If subtract_prior=True (default): values are prior-subtracted excess scores
    p_excess = clip((p_raw - alpha) / (1 - alpha), 0, 1).  Values of 0 mean
    no above-prior evidence; values > 0 indicate positive excess.

    If subtract_prior=False: values are raw latent-L posteriors P(L >= tau),
    each in [alpha, 1].  These are the correct inputs for detector-consistent
    accumulation (counting detected SGBs under the union rule p_union >= 0.5)
    rather than accumulated above-prior evidence.

    subtract_prior=True is kept as the default for backward compatibility with
    _posterior_accumulation-based code.  New code should pass subtract_prior=False
    and use _hard_call_accumulation to obtain detector-consistent curves.
    """
    sgbs     = bfrac_s.columns.tolist()
    samps    = bfrac_s.index.tolist()
    n_s, n_g = len(samps), len(sgbs)

    b_mat = bfrac_s.values.astype(np.float32)           # (n_s, n_g)
    d_mat = depth_s[sgbs].values.astype(np.float32)

    if genome_lengths is None:
        raise ValueError(
            "_compute_per_sample_posterior: genome_lengths is required. "
            "Load all_SGB_lengths_*.txt and pass as a dict {sgb_id: length_bp}."
        )
    eff_min = np.array([
        max(1e-4, min_map_bp / max(
            int(genome_lengths.get(str(g)) or genome_lengths.get(g)), 1
        ))
        for g in sgbs
    ], dtype=np.float32)  # (n_g,)

    # Build per-sample sigma matrix (n_s, n_g) using depth-stratified model.
    # Each cell sigma_mat[i,j] = sigma at depth d_mat[i,j].
    if sigma_bins is not None and sigma_edges is not None:
        # Vectorized bin lookup over the full (n_s, n_g) depth matrix
        log_d_flat = np.log10(np.maximum(d_mat.ravel(), 1e-9))
        k_flat     = np.searchsorted(sigma_edges[1:], log_d_flat)
        k_flat     = np.clip(k_flat, 0, len(sigma_bins) - 1).astype(int)
        sigma_flat = sigma_bins[k_flat].astype(np.float32)
        sigma_mat  = sigma_flat.reshape(n_s, n_g)
    else:
        sigma_mat  = np.full((n_s, n_g), sigma_b_ref, dtype=np.float32)
    sigma_mat = np.maximum(sigma_mat, 1e-9)

    result = {}
    for nm, (f_fixed, kb) in NULL_MODELS.items():
        # Threshold tau in bp — same mapping as fit_sgb_asymptote
        # Each SGB uses its own G_g so fixed-kb thresholds are genome-independent
        tau_per_sgb = []
        for g_id in sgbs:
            G_g = int(genome_lengths.get(str(g_id)) or genome_lengths.get(g_id))
            tau_per_sgb.append(f_fixed * G_g if f_fixed is not None else float(kb) * 1000.0)

        p_mat = np.zeros((n_s, n_g), dtype=np.float32)
        for j, sgb in enumerate(sgbs):
            G_g = int(genome_lengths.get(str(sgb)) or genome_lengths.get(sgb))
            tau = tau_per_sgb[j]
            if tau >= G_g:
                continue   # threshold exceeds genome
            p_col = _latent_L_posterior_vec(
                b_arr     = b_mat[:, j],
                d_arr     = d_mat[:, j],
                G         = G_g,
                c         = c,
                sigma_arr = sigma_mat[:, j],
                tau       = tau,
                n_quad    = 40,
                pi_present = pi_present,
            ).astype(np.float64)

            # Prior subtraction (optional): converts raw posterior to excess score.
            # Only applied when subtract_prior=True.
            if subtract_prior:
                alpha = _latent_L_prior_mass(tau, G_g, pi_present=pi_present)
                if alpha > 0:
                    p_col = np.clip((p_col - alpha) / (1.0 - alpha), 0.0, 1.0)

            # Numerical floor: samples with too few mapped reads → 0
            # Applied regardless of subtract_prior (data quality, not prior).
            below_floor = b_mat[:, j] < eff_min[j]
            p_col[below_floor] = 0.0
            p_mat[:, j] = p_col.astype(np.float32)

        result[nm] = pd.DataFrame(p_mat, index=samps, columns=sgbs, dtype=np.float32)

    return result   # dict  nm -> DataFrame(float32)  [sample x sgb]


# Keep a thin compatibility alias so existing calls to _compute_per_sample_detection
# still work (e.g. from test scripts or notebooks).
def _compute_per_sample_detection(bfrac_s, depth_s, c,
                                   min_map_bp=10_000, genome_lengths=None):  # genome_lengths required
    """Thin alias → returns boolean DataFrames (p >= 0.5) from the posterior."""
    p_dict = _compute_per_sample_posterior(
        bfrac_s, depth_s, c,
        min_map_bp=min_map_bp, genome_lengths=genome_lengths)
    return {nm: (df >= 0.5) for nm, df in p_dict.items()}



def _collapse_to_study_posteriors(
    p_df:       pd.DataFrame,
    meta_s:     pd.DataFrame,
    study_col:  str,
) -> pd.DataFrame:
    """
    Collapse per-sample posteriors to per-study posteriors.

        p_study_j = 1 - prod_{i in study} (1 - p_{i,j})

    Returns a DataFrame indexed by study, with the same SGB columns as p_df.
    SGBs with no above-floor evidence in a study get p_study=0.
    """
    studies  = meta_s[study_col].unique()
    n_st     = len(studies)
    n_g      = len(p_df.columns)
    p_study  = np.zeros((n_st, n_g), dtype=np.float32)

    for k, study in enumerate(studies):
        samp_mask = meta_s[study_col] == study
        samp_ids  = meta_s.index[samp_mask]
        rows      = p_df.loc[p_df.index.intersection(samp_ids)].values
        if rows.shape[0] == 0:
            continue
        # 1 - prod(1-p)  =  1 - exp(sum log(1-p))
        # Clip to 1 - 1e-6 (not 1 - 1e-9): float32 has ~7 decimal digits of
        # precision, so 1 - 1e-9 rounds to exactly 1.0 and log1p(-1.0) = -inf.
        lnd = np.sum(np.log1p(-np.clip(rows.astype(np.float64), 0, 1 - 1e-6)), axis=0)
        p_study[k] = np.clip(1.0 - np.exp(lnd), 0, 1)

    return pd.DataFrame(p_study.astype(np.float32),
                        index=pd.Index(studies, name=study_col),
                        columns=p_df.columns)

def _logspace_checkpoints(N: int, n_pts: int = 80) -> np.ndarray:
    """
    Return an array of ~n_pts unique integers in [1, N], log-spaced.

    The sequence always opens with consecutive integers 1, 2, 3, ... for
    as long as raw log-spacing would produce duplicates after rounding —
    this is handled by detecting the geometric-step crossover analytically
    and anchoring those small values explicitly.  Above the crossover the
    remaining budget is filled with log-spaced integers.  This guarantees:
      - exactly n_pts points for N >= n_pts (up to ±2 from endpoint union)
      - the sequence always starts 1, 2, 3, ...
      - both 1 and N are always present as endpoints
    """
    if N <= n_pts:
        return np.arange(1, N + 1, dtype=int)

    # Crossover: smallest integer where consecutive log-spaced points
    # differ by >= 1 after rounding.  The geometric step ratio is
    # N^(1/(n_pts-1)); it exceeds 1 per unit when x > 1/(ratio-1).
    ratio     = N ** (1.0 / (n_pts - 1))
    crossover = max(2, int(np.ceil(1.0 / (ratio - 1))))

    # Head: all integers 1 .. crossover (consecutive, no gaps)
    head = np.arange(1, min(crossover, N) + 1, dtype=int)

    # Tail: remaining n_pts slots filled log-spaced from crossover+1 to N
    tail_budget = n_pts - len(head)
    if tail_budget > 0 and crossover < N:
        raw  = np.logspace(np.log10(crossover + 1), np.log10(N), tail_budget * 4)
        tail = np.unique(np.round(raw).astype(int))
        tail = np.clip(tail, crossover + 1, N)
        if len(tail) > tail_budget:
            idx  = np.round(np.linspace(0, len(tail) - 1, tail_budget)).astype(int)
            tail = tail[idx]
    else:
        tail = np.array([], dtype=int)

    pts = np.union1d(head, tail)
    pts = np.union1d(pts, [1, N])   # guarantee endpoints after subsampling
    return pts.astype(int)


def _posterior_accumulation(
    p_df:        pd.DataFrame,
    n_iter:      int,
    rng:         np.random.Generator,
    checkpoints: np.ndarray = None,
) -> tuple:
    """
    Posterior-weighted rarefaction.  Returns (curves, checkpoints) where
    curves[i, q] = E[S(checkpoints[q]) | ordering i] and E is taken over
    the joint distribution of SGBs being detected by at least one of the
    first k samples.

    Formula (exact, no Poisson approximation):
        E[S(k | perm)] = sum_j  1 - prod_{t=1}^{k} (1 - p_{perm[t], j})
                       = sum_j  1 - exp( sum_{t=1}^{k} log(1 - p_{perm[t], j}) )

    Efficient sparse implementation
    --------------------------------
    For each SGB j, only samples with p_ij > 0 contribute non-trivially to
    the log-product (log(1-0) = 0 contributes nothing).  We precompute a
    padded matrix of those non-zero (sample, log(1-p)) pairs, sort them by
    permuted rank for each iteration, then use cumulative sums + searchsorted
    to evaluate all checkpoints in one vectorised pass.

    Complexity per iteration:
        O(n_g * max_k * log(max_k))  argsort
        O(n_g * max_k * n_cp)        checkpoint evaluation
    where max_k = max non-zero entries per SGB.  For typical density ~5%,
    max_k ≈ 0.05 * n_s; with n_g=4000, n_s=14000: ~280 × log(700) ≈ 2500
    ops per SGB, × 4000 × n_cp=80 ≈ 800 M ops per iter vs. 56 B for dense.

    Parameters
    ----------
    p_df        : DataFrame[float32], shape (n_samples, n_sgbs), values in [0,1]
    n_iter      : number of random orderings
    rng         : numpy Generator
    checkpoints : 1-D int array (1-based sample counts); _logspace_checkpoints if None

    Returns
    -------
    curves      : float32 ndarray shape (n_iter, len(checkpoints))
    checkpoints : int ndarray
    """
    n_s, n_g = p_df.shape
    p_mat     = p_df.values.astype(np.float32)        # (n_s, n_g)

    if checkpoints is None:
        checkpoints = np.arange(1, n_s + 1, dtype=int)

    cp_idx = (checkpoints - 1).astype(np.int32)   # 0-based
    n_cp   = len(cp_idx)

    # ── Precompute sparse log(1-p) structure ─────────────────────────────
    # For each SGB j: store indices of samples with p_ij > EPS and their lp values.
    # Pad rows to max_k with sentinel index n_s (log(1-0)=0, no effect).
    EPS   = 1e-6
    dets  = []
    max_k = 1
    for j in range(n_g):
        idx_j = np.where(p_mat[:, j] > EPS)[0]
        dets.append(idx_j)
        if len(idx_j) > max_k:
            max_k = len(idx_j)

    det_idx = np.full((n_g, max_k), n_s, dtype=np.int32)    # sentinel = n_s
    det_lp  = np.zeros((n_g, max_k), dtype=np.float32)      # lp = 0 for sentinel

    for j, idx_j in enumerate(dets):
        k = len(idx_j)
        if k == 0:
            continue
        det_idx[j, :k] = idx_j
        det_lp[j,  :k] = np.log1p(-np.clip(p_mat[idx_j, j], 0, 1 - EPS))

    curves   = np.empty((n_iter, n_cp), dtype=np.float32)
    perm_inv = np.empty(n_s + 1, dtype=np.int32)
    perm_inv[n_s] = n_s      # sentinel maps to rank n_s (beyond all checkpoints)

    for it in range(n_iter):
        perm            = rng.permutation(n_s)
        perm_inv[perm]  = np.arange(n_s, dtype=np.int32)

        # Ranks of every detector in this permutation
        ranks = perm_inv[det_idx]                    # (n_g, max_k)

        # Sort each SGB's detectors by rank (ascending)
        sort_idx    = np.argsort(ranks, axis=1)
        s_ranks     = np.take_along_axis(ranks,  sort_idx, axis=1)   # (n_g, max_k)
        s_lp        = np.take_along_axis(det_lp, sort_idx, axis=1)   # (n_g, max_k)
        cum_lp      = np.cumsum(s_lp, axis=1)                         # (n_g, max_k)

        # For each checkpoint q: K_j(q) = # detectors with rank <= cp_idx[q].
        # Peak memory of the (n_g, max_k, n_cp) broadcast can be large, so
        # process checkpoints in chunks that stay under ~50 MB.
        CHUNK_MEM_B = 50 * 1024 * 1024   # 50 MB
        chunk_size  = max(1, CHUNK_MEM_B // max(n_g * max_k, 1))
        j_idx       = np.arange(n_g)[:, None]

        for q0 in range(0, n_cp, chunk_size):
            q1     = min(q0 + chunk_size, n_cp)
            cp_q   = cp_idx[q0:q1]                              # (q_chunk,)
            # K[j, q] = # detectors of SGB j with rank <= cp_q[q]
            K      = (s_ranks[:, :, None] <= cp_q[None, None, :]).sum(axis=1)  # (n_g, q_chunk)
            safe_K = np.maximum(K, 1) - 1
            sel_lp = np.where(K > 0, cum_lp[j_idx, safe_K], 0.0)               # (n_g, q_chunk)
            curves[it, q0:q1] = (1.0 - np.exp(sel_lp)).sum(axis=0)

    return curves, checkpoints


# Backward-compat alias for the boolean rarefaction path
def _bootstrap_accumulation(det_bool, n_iter, rng, checkpoints=None):
    """Boolean rarefaction (deprecated — use _posterior_accumulation)."""
    # Convert boolean to 0/1 float and delegate
    p_df = pd.DataFrame(
        det_bool.values.astype(np.float32),
        index=det_bool.index, columns=det_bool.columns,
    )
    return _posterior_accumulation(p_df, n_iter, rng, checkpoints)



def _fit_hill_mm(
    ns:    np.ndarray,
    Ss:    np.ndarray,
    S_ref: int,
    S_sem: np.ndarray = None,
) -> dict:
    """
    Fit the Hill (generalised Michaelis-Menten) accumulation model:

        S(n) = S_max * n^h / (K^h + n^h)

    Parameters
    ----------
    S_max : asymptote (constrained to (0, S_ref])
    K     : half-saturation sample count (K^h + n^h denominator)
    h     : Hill coefficient — shape parameter
              h < 1  sub-linear (faster-rising, no inflection)
              h = 1  classical Michaelis-Menten
              h > 1  sigmoidal (inflection at n = K)

    Fitting uses inverse-variance weighting: points with tighter bootstrap
    SEM contribute more to the objective.  If S_sem is None, equal weights
    are used (classical unweighted LSQ).

    Returns dict with keys: S_max, K, h, S_pred, success.
    """
    from scipy.optimize import minimize

    ns_f  = ns.astype(float)
    Ss_f  = Ss.astype(float)
    # Weights: 1/sigma^2, capped to avoid domination by a single near-zero SEM
    if S_sem is not None and len(S_sem) == len(Ss):
        w = 1.0 / np.maximum(S_sem, 0.5) ** 2
    else:
        w = np.ones(len(Ss_f))
    w = w / w.sum()

    def residuals(params):
        S_max, K, h = params
        n_h  = ns_f ** h
        K_h  = K    ** h
        pred = S_max * n_h / (K_h + n_h)
        return float(np.sum(w * (pred - Ss_f) ** 2))

    bounds = [(1.0, float(S_ref)),
              (0.1, float(max(ns_f) * 10)),
              (0.1, 5.0)]
    best = None
    for K0 in [float(np.median(ns_f)), float(max(ns_f))]:
        for h0 in [0.5, 1.0, 2.0]:
            S0  = float(min(Ss_f[-1] * 2, S_ref))
            res = minimize(residuals, x0=[S0, K0, h0],
                           bounds=bounds, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res

    if best is None or (not best.success and best.fun > 1e8):
        return dict(S_max=np.nan, K=np.nan, h=np.nan,
                    S_pred=np.full_like(ns_f, np.nan), success=False)

    S_max, K, h = best.x
    n_h = ns_f ** h; K_h = K ** h
    return dict(S_max=float(S_max), K=float(K), h=float(h),
                S_pred=S_max * n_h / (K_h + n_h), success=True)


def _fit_richards(
    ns:    np.ndarray,
    Ss:    np.ndarray,
    S_ref: int,
    S_sem: np.ndarray = None,
) -> dict:
    """
    Fit  S(n) = S_max · (1 − exp(−b · n^δ))  with
        0 < S_max ≤ S_ref,  b > 0,  0.1 ≤ δ ≤ 5.

    S_sem (optional): bootstrap SEM at each checkpoint.  Points with smaller
    SEM are weighted more heavily (inverse-variance weighting), exactly as in
    _fit_hill_mm.

    Multiple restarts over (b, δ) grid; returns best-residual solution.
    Returns dict with keys: S_max, b, delta, S_pred (on ns grid), success.
    """
    from scipy.optimize import minimize

    ns_f = ns.astype(float); Ss_f = Ss.astype(float)
    if S_sem is not None and len(S_sem) == len(Ss):
        w = 1.0 / np.maximum(S_sem, 0.5) ** 2
    else:
        w = np.ones(len(Ss_f))
    w = w / w.sum()

    def residuals(params):
        S_max, b, delta = params
        pred = S_max * (1.0 - np.exp(-b * ns_f ** delta))
        return float(np.sum(w * (pred - Ss_f) ** 2))

    bounds = [(1.0, float(S_ref)), (1e-6, 10.0), (0.1, 5.0)]

    best = None
    for b0 in [0.01, 0.1, 1.0]:
        for d0 in [0.5, 1.0, 2.0]:
            S0 = float(min(Ss_f[-1] * 1.5, S_ref))
            res = minimize(residuals, x0=[S0, b0, d0], bounds=bounds, method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res

    if best is None or not best.success and best.fun > 1e8:
        return dict(S_max=np.nan, b=np.nan, delta=np.nan,
                    S_pred=np.full_like(ns, np.nan, dtype=float), success=False)

    S_max, b, delta = best.x
    return dict(S_max=float(S_max), b=float(b), delta=float(delta),
                S_pred=S_max * (1.0 - np.exp(-b * ns ** delta)), success=True)


def _asymptote_vs_n(
    ns_obs:      np.ndarray,
    S_mean:      np.ndarray,
    S_ref:       int,
    S_sem:       np.ndarray = None,
    k_min:       int = 20,
) -> tuple:
    """
    For each prefix k = k_min … len(ns_obs), fit Hill MM and Richards to
    the mean accumulation curve up to k (with SEM weighting when available),
    and record S_max.  Returns (ks, mm_smax, ri_smax).
    """
    N = len(ns_obs)
    ks, mm_smax, ri_smax = [], [], []
    for k in range(k_min, N + 1):
        ns_k  = ns_obs[:k]
        Ss_k  = S_mean[:k]
        sem_k = S_sem[:k] if S_sem is not None else None
        if Ss_k[-1] < 2:
            continue
        mm = _fit_hill_mm(ns_k, Ss_k, S_ref, S_sem=sem_k)
        ri = _fit_richards(ns_k, Ss_k, S_ref, S_sem=sem_k)
        ks.append(ns_obs[k - 1])
        mm_smax.append(mm["S_max"] if mm["success"] else np.nan)
        ri_smax.append(ri["S_max"] if ri["success"] else np.nan)
    return np.array(ks), np.array(mm_smax), np.array(ri_smax)


def _run_one_accumulation_level(
    accum_dict:     dict,
    N_accum:        int,
    x_label:        str,
    accum_note:     str,
    S_ref:          int,
    n_iter:         int,
    ci_pct:         float,
    extrap_factor:  float,
    k_min_asym:     int,
    rng:            np.random.Generator,
    out_path_accum: str,
    out_path_asym:  str,
    lifestyle_label: str = "",
):
    """
    Run bootstrap rarefaction and produce the accumulation + asymptote figures
    for one rarefaction level (either study-level or sample-level).

    The x-axis is scaled correctly to N_accum (the number of units being
    rarefied) rather than the number of raw samples, so study-level figures
    use the study count on the x-axis and sample-level figures use the sample
    count.  The extrapolation region extends from N_accum to
    N_accum * extrap_factor, giving a meaningful look at projected saturation.
    """
    NULL_ORDER  = ["1pct", "3pct", "10pct", "30pct", "50pct", "90pct", "10kb", "30kb", "100kb", "300kb", "500kb", "1Mb"]
    NULL_LABELS = {
        "1pct": "1% of genome",
        "3pct": "3% of genome",
        "10pct": "10% of genome",
        "30pct": "30% of genome",
        "50pct": "50% of genome",
        "90pct": "90% of genome",
        "10kb": "10 kb fragment",
        "30kb": "30 kb fragment",
        "100kb": "100 kb fragment",
        "300kb": "300 kb fragment",
        "500kb": "500 kb fragment",
        "1Mb": "1 Mb fragment",
    }
    NULL_COLS = {
        "1pct": "#bdbdbd",
        "3pct": "#969696",
        "10pct": "#4393c3",
        "30pct": "#888888",
        "50pct": "#525252",
        "90pct": "#252525",
        "10kb": "#f6c65b",
        "30kb": "#f0a330",
        "100kb": "#e08000",
        "300kb": "#d62728",
        "500kb": "#8e3b78",
        "1Mb": "#5b2c6f",
    }

    # Log-spaced checkpoints over the actual rarefaction range
    ns_obs = _logspace_checkpoints(N_accum, n_pts=80)
    print(f"    [{accum_note}] checkpoints: {len(ns_obs)} pts "
          f"(min={ns_obs[0]}, max={ns_obs[-1]})")

    results = {}
    for nm in NULL_ORDER:
        print(f"      Bootstrap — null={nm} ({n_iter} iter) ...", flush=True)
        p_df           = accum_dict[nm]
        curves, ns_obs = _posterior_accumulation(
            p_df, n_iter=n_iter, rng=rng, checkpoints=ns_obs)

        S_mean  = curves.mean(axis=0)
        S_sem   = curves.std(axis=0) / max(np.sqrt(n_iter), 1.0)
        lo_pct  = (100 - ci_pct) / 2.0
        S_lo    = np.percentile(curves, lo_pct,          axis=0)
        S_hi    = np.percentile(curves, 100 - lo_pct,    axis=0)

        # n_det_obs: SGBs with any positive excess evidence in this null.
        # Prior-subtracted posteriors in p_df are already on the excess scale
        # (0 = uninformative), so > 0 matches the detection criterion.
        n_det_obs = int((p_df.values > 0.0).any(axis=0).sum())
        mm = _fit_hill_mm(ns_obs, S_mean, S_ref, S_sem=S_sem)
        ri = _fit_richards(ns_obs, S_mean, S_ref, S_sem=S_sem)

        # Extrapolation uses N_accum (not raw sample count) as the observed limit
        ns_ext = np.linspace(1, N_accum * extrap_factor, 400)
        if mm["success"]:
            n_h = ns_ext ** mm["h"]; K_h = mm["K"] ** mm["h"]
            mm_ext = mm["S_max"] * n_h / (K_h + n_h)
        else:
            mm_ext = None
        ri_ext = (ri["S_max"] * (1.0 - np.exp(-ri["b"] * ns_ext ** ri["delta"]))) \
                 if ri["success"] else None

        ks, mm_smax_k, ri_smax_k = _asymptote_vs_n(
            ns_obs, S_mean, S_ref, S_sem=S_sem, k_min=k_min_asym)

        results[nm] = dict(
            S_mean=S_mean, S_sem=S_sem, S_lo=S_lo, S_hi=S_hi,
            n_det_obs=n_det_obs, mm=mm, ri=ri,
            ns_ext=ns_ext, mm_ext=mm_ext, ri_ext=ri_ext,
            ks=ks, mm_smax_k=mm_smax_k, ri_smax_k=ri_smax_k,
        )

    # ── Figure A: accumulation curves ────────────────────────────────────
    fig_a, axes_a = plt.subplots(2, 2, figsize=(16, 12))
    axes_a = axes_a.ravel()
    fig_a.suptitle(
        f"SGB accumulation — posterior-weighted rarefaction  [{accum_note}]\n"
        f"E[S(k)] = \u03a3\u2c7c (1 \u2212 \u220f\u1d62 (1\u2212p\u1d62\u2c7c))  \u00b7  "
        f"{n_iter} orderings  \u00b7  {ci_pct:.0f}% CI ribbon  \u00b7  "
        f"S_ref={S_ref:,}  {lifestyle_label}",
        fontsize=11, fontweight="bold",
    )

    x_extrap_end = N_accum * extrap_factor

    for ax, nm in zip(axes_a, NULL_ORDER):
        r   = results[nm]
        col = NULL_COLS[nm]

        ax.fill_between(ns_obs, r["S_lo"], r["S_hi"],
                        color=col, alpha=0.15, label=f"{ci_pct:.0f}% CI (orderings)")
        ax.errorbar(ns_obs, r["S_mean"],
                    yerr=1.96 * r["S_sem"],
                    fmt="none", ecolor="k", elinewidth=0.5,
                    capsize=1.5, capthick=0.5, alpha=0.45,
                    label="Mean \u00b1 1.96 SEM")
        ax.plot(ns_obs, r["S_mean"], color="k", lw=1.6, label="Mean accumulation")

        if r["mm"]["success"]:
            ax.plot(ns_obs, r["mm"]["S_pred"], color="#2ca02c", lw=1.8, label=(
                f"Hill MM  S_max={r['mm']['S_max']:.0f}  "
                f"K={r['mm']['K']:.0f}  h={r['mm']['h']:.2f}"))
            # Use r["mm_ext"] — not an outer-scope variable — to avoid stale reference
            if r["mm_ext"] is not None:
                ext_mask = r["ns_ext"] > N_accum
                if ext_mask.any():
                    ax.plot(r["ns_ext"][ext_mask], r["mm_ext"][ext_mask],
                            color="#2ca02c", lw=1.4, ls="--", alpha=0.7)

        if r["ri"]["success"]:
            ax.plot(ns_obs, r["ri"]["S_pred"], color="#ff7f0e", lw=1.8, label=(
                f"Richards  S_max={r['ri']['S_max']:.0f}  "
                f"\u03b4={r['ri']['delta']:.2f}"))
            if r["ri_ext"] is not None:
                ext_mask = r["ns_ext"] > N_accum
                if ext_mask.any():
                    ax.plot(r["ns_ext"][ext_mask], r["ri_ext"][ext_mask],
                            color="#ff7f0e", lw=1.4, ls="--", alpha=0.7)

        ax.axhline(S_ref, color="#777", lw=0.9, ls=":",
                   label=f"Reference catalogue ({S_ref:,})")
        ax.axhline(r["n_det_obs"], color=col, lw=0.9, ls="--", alpha=0.6,
                   label=f"Observed ({r['n_det_obs']:,})")

        # x-axis spans from 0 to N_accum * extrap_factor — correctly scaled
        ax.set_xlim(0, x_extrap_end)
        ax.set_ylim(0, S_ref * 1.05)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel("Cumulative SGBs detected", fontsize=9)
        ax.set_title(f"Null: {NULL_LABELS[nm]}", fontsize=10, color=col, fontweight="bold")
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, lw=0.5)

        # Extrapolation shading — starts at N_accum, not N_obs
        ax.axvspan(N_accum, x_extrap_end, color="#eeeeee", alpha=0.5, zorder=0)
        ax.text(N_accum * 1.02, S_ref * 0.03, "extrapolation",
                fontsize=7, color="#888", va="bottom")

    plt.tight_layout()
    fig_a.savefig(out_path_accum, dpi=150, bbox_inches="tight")
    plt.close(fig_a)
    print(f"    Saved: {out_path_accum}")

    # ── Figure B: asymptote estimate vs units included ────────────────────
    fig_b, axes_b = plt.subplots(2, 2, figsize=(16, 11))
    axes_b = axes_b.ravel()
    fig_b.suptitle(
        f"Asymptote estimate S_max vs {x_label.lower()} [{accum_note}]\n"
        f"Convergence to a stable value \u2192 curve near saturation  "
        f"\u00b7  S_ref={S_ref:,}  {lifestyle_label}",
        fontsize=11, fontweight="bold",
    )

    for ax, nm in zip(axes_b, NULL_ORDER):
        r   = results[nm]
        col = NULL_COLS[nm]
        ks  = r["ks"]
        if len(ks) == 0:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        ax.plot(ks, r["mm_smax_k"],  color="#2ca02c", lw=1.6, label="MM S_max")
        ax.plot(ks, r["ri_smax_k"],  color="#ff7f0e", lw=1.6, label="Richards S_max")
        ax.axhline(S_ref, color="#777", lw=0.9, ls=":", label=f"S_ref={S_ref:,}")
        for est, c2 in [(r["mm"]["S_max"], "#2ca02c"), (r["ri"]["S_max"], "#ff7f0e")]:
            if np.isfinite(est):
                ax.axhline(est, color=c2, lw=0.7, ls="--", alpha=0.5)

        ax.set_xlim(ks[0], ks[-1])
        ax.set_ylim(0, S_ref * 1.1)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel("Estimated S_max", fontsize=9)
        ax.set_title(f"Null: {NULL_LABELS[nm]}", fontsize=10, color=col, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, lw=0.5)

    plt.tight_layout()
    fig_b.savefig(out_path_asym, dpi=150, bbox_inches="tight")
    plt.close(fig_b)
    print(f"    Saved: {out_path_asym}")


def plot_sgb_accumulation(
    bfrac_s:              pd.DataFrame,
    depth_s:              pd.DataFrame,
    meta_s:               pd.DataFrame,
    c:                    float,
    S_ref:                int,
    out_path_accum:       str,          # study-level accumulation figure
    out_path_asym:        str,          # study-level asymptote figure
    out_path_accum_samp:  str = None,   # sample-level accumulation figure (new)
    out_path_asym_samp:   str = None,   # sample-level asymptote figure (new)
    genome_lengths:       dict   = None,
    sigma_b_ref:          float  = 0.015,
    sigma_bins:           np.ndarray = None,
    sigma_edges:          np.ndarray = None,
    z_thresh:             float  = 4.5,
    min_map_bp:           int    = 10_000,
    n_iter:               int    = 200,
    ci_pct:               float  = 95.0,
    extrap_factor:        float  = 2.0,
    k_min_asym:           int    = 20,
    seed:                 int    = 0,
    lifestyle_label:      str    = "",
    study_col:            str    = None,
):
    """
    Accumulation analysis at both study level and sample level.

    Produces up to four figures:
      out_path_accum      — study-level accumulation curves (2×2, one per null)
      out_path_asym       — study-level asymptote-vs-n (2×2)
      out_path_accum_samp — sample-level accumulation curves
      out_path_asym_samp  — sample-level asymptote-vs-n

    When a study column is found the study-level figures show the curve with
    the study count on the x-axis, correctly scaled so the accumulation shape
    is visible.  The sample-level figures show the same posterior probabilities
    rarefied at the sample granularity, answering "how many SGBs detected as
    more individual samples are added?"  Both x-axes use N * extrap_factor as
    the right limit, so the extrapolation shading is always meaningful.

    If no study column is found, only the sample-level figures are produced
    and written to out_path_accum / out_path_asym (for back-compatibility).
    """
    rng = np.random.default_rng(seed)

    print(f"    Computing per-sample posteriors "
          f"({len(bfrac_s)} samples \u00d7 {len(bfrac_s.columns)} SGBs) ...")
    det_dict = _compute_per_sample_posterior(
        bfrac_s, depth_s, c=c,
        sigma_b_ref=sigma_b_ref,
        sigma_bins=sigma_bins,
        sigma_edges=sigma_edges,
        min_map_bp=min_map_bp,
        genome_lengths=genome_lengths,
    )

    _STUDY_COLS = ("StudyID", "Study", "cohort", "_study")
    if study_col is None:
        study_col = next((col for col in _STUDY_COLS if col in meta_s.columns), None)

    common_kw = dict(
        S_ref=S_ref, n_iter=n_iter, ci_pct=ci_pct,
        extrap_factor=extrap_factor, k_min_asym=k_min_asym,
        rng=rng, lifestyle_label=lifestyle_label,
    )

    # ── Study-level rarefaction ───────────────────────────────────────────
    if study_col is not None:
        n_studies = meta_s[study_col].nunique()
        print(f"    Study-level: collapsing {len(meta_s)} samples "
              f"\u2192 {n_studies} studies (col='{study_col}') ...")
        det_dict_study = {
            nm: _collapse_to_study_posteriors(p_df, meta_s, study_col)
            for nm, p_df in det_dict.items()
        }
        _run_one_accumulation_level(
            accum_dict   = det_dict_study,
            N_accum      = n_studies,
            x_label      = "Studies included (n)",
            accum_note   = f"study-level rarefaction ({n_studies} studies)",
            out_path_accum = out_path_accum,
            out_path_asym  = out_path_asym,
            **common_kw,
        )
    else:
        print("    No study column found — skipping study-level figure.")

    # ── Sample-level rarefaction ──────────────────────────────────────────
    # Always run.  Use the *_samp paths when provided, otherwise fall back to
    # the primary paths (so callers that only pass two paths still work).
    _samp_accum = out_path_accum_samp or (
        out_path_accum if study_col is None else
        out_path_accum.replace(".png", "_sample.png")
    )
    _samp_asym = out_path_asym_samp or (
        out_path_asym if study_col is None else
        out_path_asym.replace(".png", "_sample.png")
    )
    N_samp = len(bfrac_s)
    print(f"    Sample-level: {N_samp} samples ...")
    _run_one_accumulation_level(
        accum_dict   = det_dict,
        N_accum      = N_samp,
        x_label      = "Samples included (n)",
        accum_note   = f"sample-level rarefaction ({N_samp} samples)",
        out_path_accum = _samp_accum,
        out_path_asym  = _samp_asym,
        **common_kw,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ref", default="UHGG",
        help="Reference database key defined in config.py (default: UHGG)",
    )
    p.add_argument(
        "--out-dir", default="figures_rank_abundance",
        help="Directory to write output figures (default: figures_rank_abundance/)",
    )
    p.add_argument(
        "--n-lines", type=int, default=150,
        help="Max individual sample lines per lifestyle per panel (default: 150)",
    )
    p.add_argument(
        "--n-label", type=int, default=12,
        help="Max VANISH/BloSSUM SGBs labelled per mean-variance panel (default: 12)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for sample subsampling (default: 42)",
    )
    p.add_argument(
        "--accum-iter", type=int, default=200,
        help="Bootstrap iterations for SGB accumulation curves (default: 200)",
    )
    p.add_argument(
        "--lowmapq-file", default=None,
        help="Path to low-MAPQ depth matrix CSV (e.g. meandepth_and_std_matrix_low_UHGG.csv). "
             "If supplied, a positive-control figure is generated comparing "
             "high-MAPQ vs low-MAPQ Spearman rho.",
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'═'*65}")
    print(f"  Rank-Abundance & Mean-Variance Analysis  |  ref={args.ref}")
    print(f"{'═'*65}")

    # ── Load data via pipeline preprocessing engine ───────────────────────
    data = load_reference(args.ref)

    meta    = data.meta.copy()
    breadth = data.breadth
    lengths = data.lengths

    # Depth-based: already computed in loading.py (depth × length → normalised)
    abund_depth = data.abundance.loc[meta.index]

    # Breadth-based: Poisson inversion → implied depth → length-weight → normalise
    print("\n  Computing breadth-based abundance (Poisson inversion) ...")
    abund_breadth = compute_breadth_abundance(
        breadth_df=breadth.loc[meta.index],
        lengths=lengths,
    )

    # Quick sanity check: correlation of per-sample total implied depth vs actual.
    # Detect breadth scale (0-1 or 0-100) the same way breadth_to_implied_depth does.
    _b_sane = breadth.loc[meta.index]
    _b_max  = float(_b_sane.values[_b_sane.values > 0].max()) \
              if (_b_sane.values > 0).any() else 0.0
    _b_frac = (_b_sane / 100.0) if _b_max > 1.0 else _b_sane
    implied_total = (-np.log1p(-_b_frac.clip(upper=1 - 1e-6))
                     .mul(lengths, axis=1).sum(axis=1))
    actual_total  = data.total_bp.loc[meta.index]
    corr = np.corrcoef(np.log1p(implied_total), np.log1p(actual_total))[0, 1]
    print(f"    Sanity check — log(implied breadth total) vs log(depth total): "
          f"Pearson r={corr:.3f}")

    # ── Sequencing-depth quartiles ────────────────────────────────────────
    # Use meta["_depth"] = total mapped bp (sum of mean_depth × genome_length),
    # computed by loading.py from the depth matrix; guaranteed non-NaN.
    depth_vals = meta["_depth"]
    meta["_depth_quartile"] = pd.qcut(
        depth_vals, q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )
    depth_quartile = meta["_depth_quartile"]

    print_quartile_summary(meta, depth_quartile)

    prefix = args.ref

    # ── Figure 1: Rank-abundance (depth-based) ────────────────────────────
    print("  [1/6] Rank-abundance — depth-based ...")
    plot_rank_abundance(
        abundance=abund_depth,
        meta=meta,
        depth_quartile=depth_quartile,
        title_prefix=f"{prefix}  ·  Depth-based",
        out_path=os.path.join(args.out_dir, f"{prefix}_rank_abundance_depth.png"),
        n_sample_lines=args.n_lines,
        rng=np.random.default_rng(args.seed),
    )

    # ── Figure 2: Rank-abundance (breadth-based) ──────────────────────────
    print("  [2/6] Rank-abundance — breadth-based (Poisson inversion) ...")
    plot_rank_abundance(
        abundance=abund_breadth,
        meta=meta,
        depth_quartile=depth_quartile,
        title_prefix=f"{prefix}  ·  Breadth-based (Poisson inv.)",
        out_path=os.path.join(args.out_dir, f"{prefix}_rank_abundance_breadth.png"),
        n_sample_lines=args.n_lines,
        rng=np.random.default_rng(args.seed),
    )

    # ── Figure 3: Rank-abundance comparison grid (2×4) ────────────────────
    print("  [3/6] Rank-abundance comparison grid (2×4) ...")
    plot_comparison_grid(
        abund_depth=abund_depth,
        abund_breadth=abund_breadth,
        meta=meta,
        depth_quartile=depth_quartile,
        plot_type="rank_abundance",
        vanish_set=data.vanish_set,
        blossum_set=data.blossum_set,
        title=f"{prefix}  —  Rank-Abundance: Depth-based (top) vs Breadth-based (bottom)",
        out_path=os.path.join(args.out_dir, f"{prefix}_rank_abundance_comparison.png"),
        n_sample_lines=args.n_lines // 2,
        rng=np.random.default_rng(args.seed),
    )

    # ── Figure 4: Mean-variance (depth-based) ─────────────────────────────
    print("  [4/6] Mean-variance — depth-based ...")
    plot_mean_variance(
        abundance=abund_depth,
        meta=meta,
        depth_quartile=depth_quartile,
        vanish_set=data.vanish_set,
        blossum_set=data.blossum_set,
        title_prefix=f"{prefix}  ·  Depth-based",
        out_path=os.path.join(args.out_dir, f"{prefix}_mean_variance_depth.png"),
        n_label=args.n_label,
    )

    # ── Figure 5: Mean-variance (breadth-based) ───────────────────────────
    print("  [5/6] Mean-variance — breadth-based (Poisson inversion) ...")
    plot_mean_variance(
        abundance=abund_breadth,
        meta=meta,
        depth_quartile=depth_quartile,
        vanish_set=data.vanish_set,
        blossum_set=data.blossum_set,
        title_prefix=f"{prefix}  ·  Breadth-based (Poisson inv.)",
        out_path=os.path.join(args.out_dir, f"{prefix}_mean_variance_breadth.png"),
        n_label=args.n_label,
    )

    # ── Figure 6: Mean-variance comparison grid (2×4) ─────────────────────
    print("  [6/6] Mean-variance comparison grid (2×4) ...")
    plot_comparison_grid(
        abund_depth=abund_depth,
        abund_breadth=abund_breadth,
        meta=meta,
        depth_quartile=depth_quartile,
        plot_type="mean_variance",
        vanish_set=data.vanish_set,
        blossum_set=data.blossum_set,
        title=f"{prefix}  —  Mean-Variance: Depth-based (top) vs Breadth-based (bottom)",
        out_path=os.path.join(args.out_dir, f"{prefix}_mean_variance_comparison.png"),
        n_label=args.n_label // 2,
        rng=np.random.default_rng(args.seed),
    )

    # ── Figure 7: Mean-variance stratified by lifestyle (depth) ──────────
    print("  [7/9] Mean-variance by lifestyle — depth-based (2x4 grid) ...")
    plot_mean_variance_by_lifestyle(
        abundance=abund_depth,
        meta=meta,
        depth_quartile=depth_quartile,
        vanish_set=data.vanish_set,
        blossum_set=data.blossum_set,
        title=f"{prefix}  --  Mean-Variance by Lifestyle (Depth-based)",
        out_path=os.path.join(args.out_dir, f"{prefix}_mean_variance_lifestyle_depth.png"),
        n_label=args.n_label,
    )

    # -- Figure 8: Mean-variance stratified by lifestyle (breadth) --------
    print("  [8/9] Mean-variance by lifestyle -- breadth-based (2x4 grid) ...")
    plot_mean_variance_by_lifestyle(
        abundance=abund_breadth,
        meta=meta,
        depth_quartile=depth_quartile,
        vanish_set=data.vanish_set,
        blossum_set=data.blossum_set,
        title=f"{prefix}  --  Mean-Variance by Lifestyle (Breadth-based, Poisson inv.)",
        out_path=os.path.join(args.out_dir, f"{prefix}_mean_variance_lifestyle_breadth.png"),
        n_label=args.n_label,
    )

    # -- Figure 9: Per-SGB max abundance ranked by lifestyle --------------
    print("  [9/10] Max-abundance per SGB by lifestyle (2x2 grid) ...")
    plot_max_abundance(
        abund_depth=abund_depth,
        abund_breadth=abund_breadth,
        meta=meta,
        vanish_set=data.vanish_set,
        blossum_set=data.blossum_set,
        title=f"{prefix}  --  Per-SGB Maximum Abundance by Lifestyle",
        out_path=os.path.join(args.out_dir, f"{prefix}_max_abundance_by_lifestyle.png"),
        n_label=args.n_label * 2,
    )

    # -- Figure 10: Abundance vs sequencing depth correlation --------------
    print("  [10/10] Depth-abundance Spearman correlation (2x2 grid) ...")
    plot_depth_abundance_correlation(
        abund_depth=abund_depth,
        abund_breadth=abund_breadth,
        meta=meta,
        vanish_set=data.vanish_set,
        blossum_set=data.blossum_set,
        title=(f"{prefix}  --  Spearman rho: SGB abundance vs sequencing depth\n"
               "(rho > 0 = abundance inflates with depth; consistent with spurious mapping)"),
        out_path=os.path.join(args.out_dir, f"{prefix}_depth_abundance_correlation.png"),
        n_label=args.n_label,
    )

    # -- Figure 11: Extended depth-correlation: pooled vs within-study ----
    print("  [11/11] Extended depth-correlation analysis (2x4 grid) ...")
    plot_depth_correlation_extended(
        abund_depth=abund_depth,
        abund_breadth=abund_breadth,
        meta=meta,
        vanish_set=data.vanish_set,
        blossum_set=data.blossum_set,
        title=(f"{prefix}  --  Depth-correlation: pooled vs within-study\n"
               "(discriminates spurious mapping / study confounding / compositional dilution)"),
        out_path=os.path.join(args.out_dir, f"{prefix}_depth_correlation_extended.png"),
    )

    print(f"\n  All 11 figures written to:  {os.path.abspath(args.out_dir)}/\n")
    print("  Output summary:")
    print(f"    Rank-abundance  (depth)                     ->  {prefix}_rank_abundance_depth.png")
    print(f"    Rank-abundance  (breadth)                   ->  {prefix}_rank_abundance_breadth.png")
    print(f"    Rank-abundance  (2x4 grid)                  ->  {prefix}_rank_abundance_comparison.png")
    print(f"    Mean-variance   (depth)                     ->  {prefix}_mean_variance_depth.png")
    print(f"    Mean-variance   (breadth)                   ->  {prefix}_mean_variance_breadth.png")
    print(f"    Mean-variance   (2x4 grid)                  ->  {prefix}_mean_variance_comparison.png")
    print(f"    Mean-variance by lifestyle (depth)          ->  {prefix}_mean_variance_lifestyle_depth.png")
    print(f"    Mean-variance by lifestyle (breadth)        ->  {prefix}_mean_variance_lifestyle_breadth.png")
    print(f"    Max abundance by lifestyle  (2x2 grid)      ->  {prefix}_max_abundance_by_lifestyle.png")
    print(f"    Depth-abundance correlation (2x2 grid)      ->  {prefix}_depth_abundance_correlation.png")
    print(f"    Depth-correlation extended  (2x4 grid)      ->  {prefix}_depth_correlation_extended.png\n")

    # -- Figure 12 (optional): low-MAPQ positive control ------------------
    if args.lowmapq_file:
        lowmapq_path = args.lowmapq_file
        if not os.path.isabs(lowmapq_path):
            lowmapq_path = os.path.join(os.getcwd(), lowmapq_path)
        if not os.path.exists(lowmapq_path):
            print(f"  WARNING: --lowmapq-file not found: {lowmapq_path}  (skipping figure 12)")
        else:
            print("  [12] Low-MAPQ positive control (2x3 grid) ...")
            abund_low = load_lowmapq_abundance(lowmapq_path, meta, lengths)
            plot_lowmapq_positive_control(
                abund_high=abund_depth,
                abund_low=abund_low,
                meta=meta,
                vanish_set=data.vanish_set,
                blossum_set=data.blossum_set,
                title=(f"{prefix}  --  Low-MAPQ positive control for spurious-mapping rho\n"
                       "Solid = high-MAPQ (regular)  |  Dashed = low-MAPQ (multi-mappers)"),
                out_path=os.path.join(args.out_dir, f"{prefix}_lowmapq_positive_control.png"),
            )
            print(f"    ->  {prefix}_lowmapq_positive_control.png")
    else:
        print("  (skipping figure 12 — pass --lowmapq-file to run the positive control)")

    # -- Figure 13: Per-SGB asymptote fit, separated by lifestyle ------------
    print("  [13] Fitting b = A*(1-exp(-c*d)) per SGB, by lifestyle ...")
    breadth_all = breadth.loc[meta.index]
    depth_all   = data.depth.loc[meta.index]

    # Estimate c once from all samples (it is a mapping property, not biology)
    max_b_chk = float(breadth_all.values[breadth_all.values > 0].max())                 if (breadth_all.values > 0).any() else 0.0
    bfrac_all = (breadth_all / 100.0) if max_b_chk > 1.0 else breadth_all
    c_est = _estimate_coverage_efficiency(bfrac_all, depth_all)
    print(f"    Global coverage efficiency c = {c_est:.4f}")

    fit_dfs    = {}
    sigma_mdls = {}   # per-lifestyle: (sigma_bins, sigma_edges)
    for ls in ["Industrial", "Non-Industrial"]:
        print(f"  [13/{ls}] fitting ...")
        _gl = {str(k): int(v) for k, v in lengths.items()} \
              if hasattr(lengths, 'items') else None
        _fit_result, _sb, _se = compute_asymptote_fits(
            breadth_df=breadth_all,
            depth_df=depth_all,
            meta=meta,
            lifestyle=ls,
            c_global=c_est,
            genome_lengths=_gl,
        )
        _fit_result["lifestyle"] = ls
        fit_dfs[ls]    = _fit_result
        sigma_mdls[ls] = (_sb, _se)
        del _fit_result, _sb, _se

    # Save combined CSV
    combined_csv = os.path.join(args.out_dir, f"{prefix}_asymptote_fits.csv")
    pd.concat(list(fit_dfs.values())).reset_index().to_csv(combined_csv, index=False)
    print(f"    ->  {prefix}_asymptote_fits.csv")

    for ls, fit_df in fit_dfs.items():
        ls_tag = ls.replace("-", "").replace(" ", "_").lower()
        plot_asymptote_consistency(
            fit_df=fit_df,
            breadth_df=breadth_all,
            depth_df=depth_all,
            meta=meta[meta["MacroGroup"] == ls],
            vanish_set=data.vanish_set,
            blossum_set=data.blossum_set,
            out_path_stats=os.path.join(
                args.out_dir, f"{prefix}_asymptote_stats_{ls_tag}.png"),
            out_path_examples=os.path.join(
                args.out_dir, f"{prefix}_asymptote_examples_{ls_tag}.png"),
            out_path_csv=None,   # already saved combined above
        )
        print(f"    ->  {prefix}_asymptote_stats_{ls_tag}.png")
        print(f"    ->  {prefix}_asymptote_examples_{ls_tag}.png")
        null_stats_path = os.path.join(args.out_dir, f"{prefix}_null_stats_{ls_tag}.png")
        plot_null_stats(fit_df, null_stats_path, lifestyle_label=f"[{ls}]")
        print(f"    ->  {prefix}_null_stats_{ls_tag}.png")

        # ── Figure 14: SGB accumulation curves + asymptote-vs-n ──────────
        print(f"  [14/{ls}] SGB accumulation curves (posterior-weighted rarefaction) ...")
        S_ref_size = int(breadth_all.shape[1])   # total SGBs in reference catalogue

        ls_samps = [s for s in meta.index
                    if meta.loc[s, "MacroGroup"] == ls
                    and s in bfrac_all.index and s in depth_all.index]
        if len(ls_samps) < 10:
            print(f"    WARNING: only {len(ls_samps)} samples for [{ls}]; skipping accumulation.")
        else:
            # Determine study column (MacroGroup excluded by policy)
            _STUDY_COLS = ("StudyID", "Study", "cohort", "_study")
            study_col_accum = next((c for c in _STUDY_COLS
                                    if c in meta.columns), None)
            _sb_accum, _se_accum = sigma_mdls.get(ls, (None, None))
            plot_sgb_accumulation(
                bfrac_s             = bfrac_all.loc[ls_samps],
                depth_s             = depth_all.loc[ls_samps],
                meta_s              = meta.loc[ls_samps],
                c                   = c_est,
                S_ref               = S_ref_size,
                genome_lengths      = ({str(k): int(v) for k, v in lengths.items()}
                                       if hasattr(lengths, "items") else None),
                sigma_b_ref         = 0.015,
                sigma_bins          = _sb_accum,
                sigma_edges         = _se_accum,
                z_thresh            = 4.5,
                min_map_bp          = 10_000,
                n_iter              = args.accum_iter,
                seed                = args.seed,
                lifestyle_label     = f"[{ls}]",
                study_col           = study_col_accum,
                out_path_accum      = os.path.join(
                    args.out_dir, f"{prefix}_sgb_accumulation_study_{ls_tag}.png"),
                out_path_asym       = os.path.join(
                    args.out_dir, f"{prefix}_asymptote_vs_n_study_{ls_tag}.png"),
                out_path_accum_samp = os.path.join(
                    args.out_dir, f"{prefix}_sgb_accumulation_sample_{ls_tag}.png"),
                out_path_asym_samp  = os.path.join(
                    args.out_dir, f"{prefix}_asymptote_vs_n_sample_{ls_tag}.png"),
            )
            print(f"    ->  {prefix}_sgb_accumulation_study_{ls_tag}.png  (study-level)")
            print(f"    ->  {prefix}_asymptote_vs_n_study_{ls_tag}.png")
            print(f"    ->  {prefix}_sgb_accumulation_sample_{ls_tag}.png  (sample-level)")
            print(f"    ->  {prefix}_asymptote_vs_n_sample_{ls_tag}.png")


if __name__ == "__main__":
    main()
