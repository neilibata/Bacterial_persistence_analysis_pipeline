"""
presence_audit.py — Audit of per-lifestyle presence matrices.

Usage (from the project root, same directory as the data files):
    python -m persistence-patched.presence_audit --ref UHGG
    python -m persistence-patched.presence_audit --ref UHGG lev

Logic
-----
Presence is defined as breadth_coverage >= PRESENCE_THRESHOLD (default 10 %).
For each Lifestyle group (Industrial, H-G, Transitional) the script:
  1. Extracts the subset of the breadth matrix belonging to that lifestyle.
  2. Prints matrix dimensions (n_samples × n_SGBs).
  3. Computes, per SGB-column, how many samples have breadth >= threshold
     (i.e. the raw prevalence count that feeds into downstream prevalence
     computations).
  4. Plots a histogram of those per-SGB counts for every lifestyle on one
     shared figure, plus a separate figure with one panel per lifestyle for
     a closer look.

Output
------
  presence_audit_combined.png  — overlapping histograms (all lifestyles)
  presence_audit_panels.png    — one subplot per lifestyle
  presence_audit_summary.csv   — per-SGB counts for every lifestyle
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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

# ── colour palette (one colour per lifestyle) ────────────────────────────────
LIFESTYLE_COLOURS = {
    "Industrial":   "#E07B54",   # warm orange-red
    "H-G":          "#4A90A4",   # teal-blue
    "Transitional": "#7BBF72",   # muted green
}
LIFESTYLE_ORDER = ["Industrial", "H-G", "Transitional"]

PRESENCE_THRESHOLD = 10.0        # % breadth


# ════════════════════════════════════════════════════════════════════════════
#  CORE AUDIT
# ════════════════════════════════════════════════════════════════════════════

def build_presence_matrices(breadth_df: pd.DataFrame,
                             meta: pd.DataFrame,
                             threshold: float = PRESENCE_THRESHOLD
                             ) -> dict[str, pd.DataFrame]:
    """
    Slice breadth_df by Lifestyle and apply the presence threshold.

    Parameters
    ----------
    breadth_df : samples × SGBs (% breadth, 0–100)
    meta       : metadata DataFrame indexed by Chosen_match_ID
    threshold  : breadth % above which a taxon is counted as present

    Returns
    -------
    Dict mapping lifestyle label → boolean presence DataFrame (0/1)
    """
    lifestyles = [ls for ls in LIFESTYLE_ORDER
                  if ls in meta["Lifestyle"].values]

    presence_by_lifestyle = {}
    for lifestyle in lifestyles:
        sample_ids = meta.index[meta["Lifestyle"] == lifestyle]
        # Intersect with breadth index (safety check)
        sample_ids = sample_ids.intersection(breadth_df.index)
        sub = breadth_df.loc[sample_ids]
        presence = (sub >= threshold).astype(np.uint8)
        presence_by_lifestyle[lifestyle] = presence

    return presence_by_lifestyle


def print_dimensions(presence_by_lifestyle: dict[str, pd.DataFrame]) -> None:
    """Pretty-print matrix dimensions for each lifestyle."""
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  Presence matrices  (threshold ≥ {PRESENCE_THRESHOLD:.0f}% breadth)")
    print(sep)
    print(f"  {'Lifestyle':<18} {'Samples':>10}  {'SGBs':>10}")
    print(sep)
    for lifestyle in LIFESTYLE_ORDER:
        if lifestyle not in presence_by_lifestyle:
            continue
        df = presence_by_lifestyle[lifestyle]
        n_samples, n_sgbs = df.shape
        print(f"  {lifestyle:<18} {n_samples:>10,}  {n_sgbs:>10,}")
    print(sep)


def compute_nonzero_counts(
        presence_by_lifestyle: dict[str, pd.DataFrame]
) -> dict[str, pd.Series]:
    """
    For each lifestyle, count how many samples are present (breadth ≥ threshold)
    for each SGB column.  Returns a dict: lifestyle → pd.Series (index = SGB).
    """
    return {ls: pres.sum(axis=0)
            for ls, pres in presence_by_lifestyle.items()}


# ════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

from matplotlib.gridspec import GridSpecFromSubplotSpec

# Log-spaced bin edges for counts >= 1.  We use log10 so tick labels can be
# written as actual sample counts (1, 2, 5, 10, 50, …) rather than log values.
_LOG_TICKS   = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]
_LOG_TICK_LABELS = ["1", "2", "5", "10", "20", "50", "100",
                    "200", "500", "1k", "2k", "5k", "10k"]


def _log_bins(counts_by_lifestyle: dict[str, pd.Series],
              n_bins: int = 50) -> np.ndarray:
    """Log-spaced bin edges for the non-zero portion, shared across lifestyles."""
    all_nonzero = np.concatenate(
        [c.values[c.values > 0] for c in counts_by_lifestyle.values()]
    )
    if len(all_nonzero) == 0:
        return np.array([1, 2])
    lo = np.log10(max(all_nonzero.min(), 1))
    hi = np.log10(all_nonzero.max() + 1)
    return np.logspace(lo, hi, n_bins + 1)


def _draw_lifestyle_panel(ax_zero, ax_log,
                           counts: pd.Series,
                           n_samples: int,
                           colour: str,
                           lifestyle: str,
                           log_bins: np.ndarray,
                           shared_ylim=None) -> float:
    """
    Draw one lifestyle panel: a zero-bar on ax_zero and a log histogram on ax_log.

    Returns the y-value of the zero bar (so the caller can set a shared y limit).
    """
    n_sgbs  = len(counts)
    n_zero  = int((counts == 0).sum())
    nonzero = counts.values[counts.values > 0]

    # ── left panel: zero bar ────────────────────────────────────────────────
    ax_zero.bar(["absent\n(0)"], [n_zero],
                color=colour, alpha=0.75, width=0.55,
                edgecolor="white", linewidth=0.8,
                hatch="///", zorder=3)
    ax_zero.set_xlim(-0.6, 0.6)
    ax_zero.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_zero.set_ylabel("Number of SGBs", fontsize=9)
    ax_zero.tick_params(axis="x", labelsize=9)
    ax_zero.spines["right"].set_visible(False)
    # Diagonal break marks on the right spine to signal axis discontinuity
    for y_frac in (0.02, 0.05):
        ax_zero.annotate("", xy=(1.04, y_frac), xycoords="axes fraction",
                         xytext=(1.10, y_frac + 0.025),
                         arrowprops=dict(arrowstyle="-", color="grey", lw=1.2))

    # ── right panel: log histogram ──────────────────────────────────────────
    ax_log.hist(nonzero, bins=log_bins,
                color=colour, alpha=0.82, edgecolor="white", linewidth=0.25,
                zorder=3)
    ax_log.set_xscale("log")

    # Custom tick positions and labels (only show ticks that fall within range)
    visible_ticks = [t for t in _LOG_TICKS if log_bins[0] <= t <= log_bins[-1]]
    visible_labels = [_LOG_TICK_LABELS[_LOG_TICKS.index(t)] for t in visible_ticks]
    ax_log.set_xticks(visible_ticks)
    ax_log.set_xticklabels(visible_labels, fontsize=8)
    ax_log.xaxis.set_minor_locator(mticker.NullLocator())

    ax_log.set_xlabel("Samples with breadth ≥ 10%  (per SGB, log scale)", fontsize=9)
    ax_log.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_log.tick_params(axis="y", labelsize=8)
    ax_log.spines["left"].set_linestyle((0, (4, 2)))   # dashed left spine = break

    # Median line on log panel (non-zero counts only)
    if len(nonzero):
        med = float(np.median(nonzero))
        ax_log.axvline(med, color=colour, lw=1.6, ls="--", alpha=0.9,
                       label=f"median = {med:.0f}")
        ax_log.legend(fontsize=8, loc="upper right")

    # Annotation box
    frac_zero  = n_zero / n_sgbs * 100
    mean_nz    = float(nonzero.mean()) if len(nonzero) else 0.0
    info = (f"n samples  = {n_samples:,}\n"
            f"n SGBs     = {n_sgbs:,}\n"
            f"absent     = {n_zero:,}  ({frac_zero:.1f}%)\n"
            f"present    = {n_sgbs - n_zero:,}  ({100-frac_zero:.1f}%)\n"
            f"mean (present) = {mean_nz:.1f}")
    ax_log.text(0.985, 0.975, info,
                transform=ax_log.transAxes,
                ha="right", va="top", fontsize=7.8,
                bbox=dict(boxstyle="round,pad=0.35",
                          fc="white", ec=colour, alpha=0.88))

    return n_zero


def plot_panels(counts_by_lifestyle: dict[str, pd.Series],
                presence_by_lifestyle: dict[str, pd.DataFrame],
                out_path: str) -> None:
    """
    One dual-panel block per lifestyle:
      LEFT  — narrow bar showing how many SGBs have zero detections
      RIGHT — log-scale histogram of non-zero per-SGB detection counts

    The two sub-panels share a y-axis and are visually connected with break
    marks, making it unambiguous that the zero category is distinct from the
    continuous log distribution.
    """
    active   = [ls for ls in LIFESTYLE_ORDER if ls in counts_by_lifestyle]
    n_ls     = len(active)
    log_bins = _log_bins(counts_by_lifestyle)

    # Figure layout: n_ls columns, each column split into [1, 5] width ratio
    fig = plt.figure(figsize=(6.5 * n_ls, 5.2))
    outer_gs = fig.add_gridspec(1, n_ls, wspace=0.38)

    for col_idx, lifestyle in enumerate(active):
        counts    = counts_by_lifestyle[lifestyle]
        n_samples = presence_by_lifestyle[lifestyle].shape[0]
        colour    = LIFESTYLE_COLOURS[lifestyle]

        inner = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_gs[col_idx],
            width_ratios=[1, 5], wspace=0.0)

        ax_zero = fig.add_subplot(inner[0])
        ax_log  = fig.add_subplot(inner[1], sharey=ax_zero)

        _draw_lifestyle_panel(ax_zero, ax_log, counts, n_samples,
                              colour, lifestyle, log_bins)

        # Lifestyle title spans both sub-panels
        mid_x = (ax_zero.get_position().x0 + ax_log.get_position().x1) / 2
        fig.text(mid_x, 0.97, lifestyle,
                 ha="center", va="top", fontsize=13,
                 fontweight="bold", color=colour,
                 transform=fig.transFigure)

        # Hide y-axis tick labels on the log panel (shared with zero panel)
        plt.setp(ax_log.get_yticklabels(), visible=False)
        ax_log.set_ylabel("")

    fig.suptitle(
        "Per-SGB prevalence count distribution — per lifestyle\n"
        f"(presence threshold ≥ {PRESENCE_THRESHOLD:.0f}% breadth  |  "
        "left bar = absent in all samples of that lifestyle)",
        fontsize=11, fontweight="bold", y=1.04)

    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_combined(counts_by_lifestyle: dict[str, pd.Series],
                  presence_by_lifestyle: dict[str, pd.DataFrame],
                  out_path: str) -> None:
    """
    Combined view: LEFT = grouped zero bars, RIGHT = overlaid log histograms.
    Useful for quick cross-lifestyle comparison on a single figure.
    """
    active   = [ls for ls in LIFESTYLE_ORDER if ls in counts_by_lifestyle]
    log_bins = _log_bins(counts_by_lifestyle)

    fig = plt.figure(figsize=(12, 5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 5], wspace=0.05)
    ax_zero = fig.add_subplot(gs[0])
    ax_log  = fig.add_subplot(gs[1], sharey=ax_zero)

    bar_width = 0.22
    offsets   = np.linspace(-(len(active)-1)*bar_width/2,
                             (len(active)-1)*bar_width/2,
                             len(active))

    for offset, lifestyle in zip(offsets, active):
        counts    = counts_by_lifestyle[lifestyle]
        n_samples = presence_by_lifestyle[lifestyle].shape[0]
        colour    = LIFESTYLE_COLOURS[lifestyle]
        n_zero    = int((counts == 0).sum())
        nonzero   = counts.values[counts.values > 0]

        # Zero bar (grouped)
        ax_zero.bar([offset], [n_zero],
                    width=bar_width * 0.9,
                    color=colour, alpha=0.78,
                    edgecolor="white", linewidth=0.8,
                    hatch="///",
                    label=f"{lifestyle}  (n={n_samples:,})")

        # Log histogram overlay
        ax_log.hist(nonzero, bins=log_bins,
                    color=colour, alpha=0.50, edgecolor="none",
                    label=f"{lifestyle}  (n={n_samples:,})")

    # Zero panel cosmetics
    ax_zero.set_xlim(-0.55, 0.55)
    ax_zero.set_xticks([0])
    ax_zero.set_xticklabels(["absent\n(0)"], fontsize=10)
    ax_zero.set_ylabel("Number of SGBs", fontsize=11)
    ax_zero.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_zero.spines["right"].set_visible(False)
    for y_frac in (0.02, 0.05):
        ax_zero.annotate("", xy=(1.04, y_frac), xycoords="axes fraction",
                         xytext=(1.10, y_frac + 0.025),
                         arrowprops=dict(arrowstyle="-", color="grey", lw=1.2))
    ax_zero.legend(fontsize=8.5, loc="upper right")

    # Log panel cosmetics
    ax_log.set_xscale("log")
    visible_ticks  = [t for t in _LOG_TICKS if log_bins[0] <= t <= log_bins[-1]]
    visible_labels = [_LOG_TICK_LABELS[_LOG_TICKS.index(t)] for t in visible_ticks]
    ax_log.set_xticks(visible_ticks)
    ax_log.set_xticklabels(visible_labels, fontsize=9)
    ax_log.xaxis.set_minor_locator(mticker.NullLocator())
    ax_log.set_xlabel("Samples with breadth ≥ 10%  (per SGB, log scale)", fontsize=11)
    ax_log.spines["left"].set_linestyle((0, (4, 2)))
    plt.setp(ax_log.get_yticklabels(), visible=False)

    fig.suptitle(
        "Per-SGB prevalence count distribution — all lifestyles\n"
        f"(presence threshold ≥ {PRESENCE_THRESHOLD:.0f}% breadth  |  "
        "left = absent SGBs, right = log-scale count histogram)",
        fontsize=11, fontweight="bold")

    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  SUMMARY CSV
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
#  DOUBLING-BIN EXTRAPOLATION
# ════════════════════════════════════════════════════════════════════════════

def _doubling_bins(counts: pd.Series):
    """
    Aggregate per-SGB detection counts into doubling bins.

    Bin k spans [2^k, 2^(k+1)).  Returns a DataFrame with columns:
      k, lo, hi, n_sgbs, label
    Only bins with at least one SGB are returned (plus every bin from
    k=0 up to the max occupied bin, with zeros where empty).
    """
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return pd.DataFrame(columns=["k", "lo", "hi", "n_sgbs", "label"])

    max_k = int(np.floor(np.log2(nonzero.max()))) + 1
    rows  = []
    for k in range(0, max_k + 1):
        lo  = int(2 ** k)
        hi  = int(2 ** (k + 1))
        n   = int(((nonzero >= lo) & (nonzero < hi)).sum())
        lbl = str(lo) if k == 0 else f"{lo}–{hi - 1}"
        rows.append(dict(k=k, lo=lo, hi=hi, n_sgbs=n, label=lbl))
    return pd.DataFrame(rows)


def _fit_log_poly(bin_df: pd.DataFrame, degree: int = 1,
                  fit_from_k: int = 1, fit_max_k: int = 4):
    """
    Fit  log(n_sgbs) = poly(k, degree)  on bins with fit_from_k ≤ k ≤ fit_max_k.

    Using only the low-k (rare) tail bins avoids the high-k bins being
    anchored by the flat 'core microbiome' plateau, which follows a different
    biological mechanism.  fit_from_k=1 also excludes the k=0 (count=1) bin
    which is most truncated by finite sampling.

    Returns (coefficients, fit_fn) where fit_fn(k) → predicted log(n_sgbs).
    """
    sub = bin_df[(bin_df["k"] >= fit_from_k) &
                 (bin_df["k"] <= fit_max_k) &
                 (bin_df["n_sgbs"] > 0)].copy()
    if len(sub) < degree + 1:
        return None, None

    x = sub["k"].values.astype(float)
    y = np.log(sub["n_sgbs"].values.astype(float))
    # Weight by sqrt(n_sgbs) so dense, reliable bins anchor the fit more
    w = np.sqrt(sub["n_sgbs"].values.astype(float))

    coeffs = np.polyfit(x, y, deg=degree, w=w)
    fit_fn = np.poly1d(coeffs)
    return coeffs, fit_fn


def _extrapolate_missing(fit_fn, n_absent: int, max_doublings: int = 12):
    """
    Extrapolate to negative k values (k = -1, -2, ...) to predict how many
    SGBs would fall in each doubling bin if we had 2×, 4×, 8× … more data.

    Each step left (k → k-1) corresponds to needing 2× more samples, so
    the SGBs in bin k=-1 with current data would appear as the new k=0 bin
    (i.e., detected in exactly 1 sample) if we doubled our cohort.

    Stops when the predicted count drops below 0.5 or we reach max_doublings.

    Returns DataFrame with columns:
      k, predicted_n, cumulative_n, cumulative_frac_of_absent
      multiplier  (how many × current data)
    """
    rows = []
    cumulative = 0.0
    for k in range(-1, -(max_doublings + 1), -1):
        pred = np.exp(fit_fn(k))
        cumulative += pred
        rows.append(dict(
            k=k,
            multiplier=int(2 ** (-k)),
            predicted_n=pred,
            cumulative_n=cumulative,
            cumulative_frac_absent=(cumulative / n_absent) if n_absent > 0 else np.nan,
        ))
        if pred < 0.5:
            break
    return pd.DataFrame(rows)


def plot_doubling_extrapolation(
        counts_by_lifestyle: dict[str, pd.Series],
        presence_by_lifestyle: dict[str, pd.DataFrame],
        out_path: str,
        fit_from_k: int = 1,
        fit_max_k: int = 4,
        max_extrap_shown: int = 5) -> None:
    """
    For each lifestyle, plot the doubling-bin histogram on a log y-axis and
    overlay linear + quadratic extrapolations into negative-k (sub-threshold)
    bins.

    Layout (per panel)
    ------------------
    • Solid bars (k ≥ 1): observed doubling bins.
    • Stippled bar (k = 0, count = 1): detection-edge bin — shown separately
      because it is the most truncated by finite sampling.
    • Hatched bars (k < 0): predicted SGBs if we had 2×, 4×, 8× … more data.
      At most max_extrap_shown bins shown even if the fit predicts more.
    • Dashed line: linear fit on k ≥ fit_from_k.
    • Dotted line: quadratic fit.
    • Vertical separator at k = –0.5.
    • Annotation box: slope, absent count, total predicted recovery.

    Reading the figure
    ------------------
    Flat slope (≈ 0): same SGBs per doubling regardless of count → true power-law;
      absent taxa are mostly biologically absent; extrapolated bars are high but
      the cumulative predicted recovery is still a small fraction of n_absent
      because each extra doubling only adds one more bar-worth of taxa.
    Negative slope: SGBs per bin decrease toward low counts → absent pool has a
      modest detectable tail, recoverable with modest extra data.
    k=0 bar gap below fit: taxa currently detected in exactly 1 sample are
      under-represented relative to the k≥1 trend → sampling truncation.
    """
    active = [ls for ls in LIFESTYLE_ORDER if ls in counts_by_lifestyle]
    n      = len(active)

    # Tall figure — rotated x-tick labels + xlabel need generous bottom margin.
    # Industrial can have max_k ≈ 13-14 bins; height 9 + bottom=0.28 is enough.
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 9.0),
                             constrained_layout=False)
    if n == 1:
        axes = [axes]

    for ax, lifestyle in zip(axes, active):
        counts    = counts_by_lifestyle[lifestyle]
        n_samples = presence_by_lifestyle[lifestyle].shape[0]
        n_absent  = int((counts == 0).sum())
        n_sgbs    = len(counts)
        colour    = LIFESTYLE_COLOURS[lifestyle]

        bdf = _doubling_bins(counts)
        if bdf.empty:
            ax.set_title(lifestyle + "\n(no data)", fontsize=12)
            continue

        max_k    = int(bdf["k"].max())
        _, lin   = _fit_log_poly(bdf, degree=1, fit_from_k=fit_from_k, fit_max_k=fit_max_k)
        _, quad  = _fit_log_poly(bdf, degree=2, fit_from_k=fit_from_k, fit_max_k=fit_max_k)

        # Extrapolation — compute all bins but only display up to max_extrap_shown
        extrap_all = (_extrapolate_missing(lin, n_absent) if lin is not None
                      else pd.DataFrame())
        extrap_shown = extrap_all.head(max_extrap_shown) if not extrap_all.empty \
                       else pd.DataFrame()
        min_k_shown  = int(extrap_shown["k"].min()) if not extrap_shown.empty else 0

        x_lo = min_k_shown - 0.7
        x_hi = max_k + 0.7

        # ── Observed bars ─────────────────────────────────────────────────
        # k=0: stippled (detection-edge, excluded from fit)
        # k in [fit_from_k, fit_max_k]: solid + dark edge (anchored the fit)
        # k > fit_max_k: slightly faded (high-count plateau, not used in fit)
        for _, row in bdf.iterrows():
            k, n_val = int(row["k"]), int(row["n_sgbs"])
            if n_val == 0:
                continue
            if k == 0:
                alpha, hatch, ec, lw = 0.50, "..",  colour, 0.8
            elif fit_from_k <= k <= fit_max_k:
                alpha, hatch, ec, lw = 0.90, None,  "white", 1.4
            else:
                alpha, hatch, ec, lw = 0.50, None,  "white", 0.5
            ax.bar(k, n_val, width=0.75, color=colour, alpha=alpha,
                   edgecolor=ec, linewidth=lw, hatch=hatch, zorder=3)

        # ── Extrapolated bars ─────────────────────────────────────────────
        for _, row in extrap_shown.iterrows():
            k, pred = int(row["k"]), row["predicted_n"]
            if pred < 0.1:
                continue
            ax.bar(k, pred, width=0.75, color=colour, alpha=0.22,
                   edgecolor=colour, linewidth=1.0, hatch="///", zorder=2)

        # ── Fit lines ─────────────────────────────────────────────────────
        kk = np.linspace(min_k_shown - 0.4, max_k + 0.4, 400)
        if lin is not None:
            ax.plot(kk, np.exp(lin(kk)), color="black", lw=1.6, ls="--",
                    alpha=0.8, zorder=4,
                    label=f"linear fit  (b={lin.coeffs[0]:+.2f}/doubling)")
        if quad is not None:
            ax.plot(kk, np.exp(quad(kk)), color="dimgrey", lw=1.2, ls=":",
                    alpha=0.7, zorder=4, label="quadratic fit")

        # ── Separator ─────────────────────────────────────────────────────
        ax.axvline(-0.5, color="grey", lw=1.0, ls=(0, (5, 3)),
                   alpha=0.65, zorder=5)
        trans = ax.get_xaxis_transform()
        ax.text(-0.52, 1.01, "← extrapolated", ha="right", va="bottom",
                fontsize=7, color="grey", style="italic", transform=trans)
        ax.text(-0.48, 1.01, "observed →",     ha="left",  va="bottom",
                fontsize=7, color="grey", style="italic", transform=trans)

        # ── Log y-axis ────────────────────────────────────────────────────
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        # ── X-ticks: show every 2nd observed bin (lower bound), all extrap ─
        # Industrial can have max_k ≈ 13; only labelling even k prevents crowd.
        tick_ks, tick_lbls = [], []
        for k in range(min_k_shown, max_k + 1):
            tick_ks.append(k)
            if k < 0:
                tick_lbls.append(f"×{2**(-k):,}")       # ×2, ×4, ×8 …
            elif k == 0:
                tick_lbls.append("1")                    # always label k=0
            elif k % 2 == 0 or k <= 4:
                tick_lbls.append(f"{int(2**k):,}")       # 2,4,8,16,32 …
            else:
                tick_lbls.append("")                     # skip odd high-k

        ax.set_xticks(tick_ks)
        ax.set_xticklabels(tick_lbls, rotation=45, ha="right", fontsize=7.5)
        ax.set_xlim(x_lo, x_hi)

        ax.set_xlabel(
            "← need more data  |  lower bound of doubling bin  |  observed →",
            fontsize=9)
        ax.set_ylabel("Number of SGBs  (log scale)", fontsize=9)
        ax.set_title(lifestyle, fontsize=13, fontweight="bold", color=colour)
        ax.set_facecolor((*matplotlib.colors.to_rgb(colour), 0.03))

        # ── Annotation ───────────────────────────────────────────────────
        lin_slope   = lin.coeffs[0] if lin is not None else float("nan")
        # Total predicted taxa across ALL extrapolated bins (not just shown)
        pred_total  = float(extrap_all["cumulative_n"].iloc[-1]) \
                      if not extrap_all.empty else 0.0
        pred_shown  = float(extrap_shown["cumulative_n"].iloc[-1]) \
                      if not extrap_shown.empty else 0.0
        n_extrap_total = len(extrap_all)
        frac_all    = pred_total / n_absent * 100 if n_absent > 0 else 0.0
        frac_shown  = pred_shown / n_absent * 100 if n_absent > 0 else 0.0

        info_lines = [
            f"n samples  = {n_samples:,}",
            f"n SGBs     = {n_sgbs:,}",
            f"absent (0) = {n_absent:,}  ({n_absent/n_sgbs*100:.1f}%)",
            "",
            f"linear slope = {lin_slope:+.2f} per doubling",
            f"fit: k={fit_from_k}–{fit_max_k} (bright bars only)",
            "",
            f"Predicted recovery:",
            f"  {max_extrap_shown} doublings shown: {pred_shown:.0f}  ({frac_shown:.1f}%)",
            f"  all {n_extrap_total} doublings:     {pred_total:.0f}  ({frac_all:.1f}%)",
        ]
        ax.text(0.985, 0.975, "\n".join(info_lines),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, family="monospace",
                bbox=dict(boxstyle="round,pad=0.4",
                          fc="white", ec=colour, alpha=0.90))

        ax.legend(fontsize=7.5, loc="upper left",
                  framealpha=0.85, edgecolor=colour)

    fig.suptitle(
        "Doubling-bin prevalence histogram with depth-extrapolation\n"
        "Solid bars: SGBs per 2× bin of detection count  |  "
        "Hatched: predicted taxa detectable with N× more data\n"
        "Fit on k≥1; stippled k=0 bar: gap below fit = truncation at detection edge",
        fontsize=10.5, fontweight="bold", y=0.995)

    # bottom=0.28 gives enough room for 45°-rotated tick labels + x-label on all OSes.
    fig.subplots_adjust(top=0.92, bottom=0.28, left=0.06, right=0.97,
                        wspace=0.30)
    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def save_summary_csv(counts_by_lifestyle: dict[str, pd.Series],
                     out_path: str) -> None:
    """
    Save a CSV with one row per SGB and one column per lifestyle showing
    the raw per-SGB detection count.
    """
    df = pd.DataFrame(counts_by_lifestyle)
    df.index.name = "SGB"
    df.to_csv(out_path)
    print(f"  Saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def run_audit(ref_name: str) -> None:
    """Full audit pipeline for one reference database."""
    print(f"\n{'='*60}")
    print(f"  Presence audit — reference: {ref_name}")
    print(f"{'='*60}")

    # Lazy import to avoid circular imports at module level
    from .loading import load_reference
    from . import config

    rdata = load_reference(ref_name)

    presence_by_lifestyle = build_presence_matrices(
        rdata.breadth, rdata.meta, threshold=PRESENCE_THRESHOLD)

    print_dimensions(presence_by_lifestyle)

    counts_by_lifestyle = compute_nonzero_counts(presence_by_lifestyle)

    # Print a quick numeric summary
    print(f"\n  Per-SGB count summary  (samples with breadth ≥ {PRESENCE_THRESHOLD:.0f}%)")
    sep = "─" * 68
    print(f"  {sep}")
    header = f"  {'Lifestyle':<18} {'min':>6} {'p25':>8} {'median':>8} {'p75':>8} {'max':>6}"
    print(header)
    print(f"  {sep}")
    for ls in LIFESTYLE_ORDER:
        if ls not in counts_by_lifestyle:
            continue
        c = counts_by_lifestyle[ls]
        print(f"  {ls:<18} "
              f"{int(c.min()):>6,} "
              f"{int(c.quantile(0.25)):>8,} "
              f"{int(c.median()):>8,.0f} "
              f"{int(c.quantile(0.75)):>8,} "
              f"{int(c.max()):>6,}")
    print(f"  {sep}")

    # Output directory
    out_dir = config.outdir(ref_name)
    os.makedirs(out_dir, exist_ok=True)

    combined_path  = os.path.join(out_dir, "presence_audit_combined.png")
    panels_path    = os.path.join(out_dir, "presence_audit_panels.png")
    doubling_path  = os.path.join(out_dir, "presence_audit_doubling.png")
    csv_path       = os.path.join(out_dir, "presence_audit_summary.csv")

    print(f"\n  Writing outputs to: {out_dir}")
    plot_combined(counts_by_lifestyle, presence_by_lifestyle, combined_path)
    plot_panels(counts_by_lifestyle, presence_by_lifestyle, panels_path)
    plot_doubling_extrapolation(counts_by_lifestyle, presence_by_lifestyle, doubling_path)
    save_summary_csv(counts_by_lifestyle, csv_path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit per-lifestyle presence matrices (10% breadth threshold).")
    parser.add_argument("--ref", nargs="+", default=["UHGG"],
                        help="Reference DB(s) to audit (default: UHGG)")
    args = parser.parse_args(argv)

    for ref in args.ref:
        run_audit(ref)


if __name__ == "__main__":
    main()
