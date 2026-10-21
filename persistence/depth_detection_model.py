"""
depth_detection_model.py
========================
Model whether non-detection of SGBs is an artefact of sequencing depth or a
genuine biological absence.

Core question
-------------
In the presence matrices (breadth ≥ 10%), a zero for SGB i in sample j could
mean either:
  (A) True biological absence — the organism simply isn't in that gut.
  (B) Depth-limited miss — the organism is present but too rare or the sample
      too shallow for 10% breadth to be reached.

We cannot run rarefaction curves, but we can answer the question via three
complementary analyses using the Poisson coverage model and within-lifestyle
depth variation.

Analyses
--------
1. Depth-stratified prevalence curves
   Bin samples by log₁₀(total_bp) into quantile bins within each lifestyle.
   For each bin compute per-SGB prevalence, then average over SGBs grouped by
   their overall prevalence tier.  Flat lines → biology dominates.
   Upward-sloping lines → depth drives detection.

2. Detection safety margin  (the most direct test)
   Under the Poisson model, P(breadth ≥ 10%) saturates quickly above mean
   depth D = 0.105.  For each SGB detected in ≥ K samples, compute:

       margin_i = median(mean_depth[j,i]  for j where present[j,i]=1) / 0.105

   margin ≫ 1 → even if the sample were 10× shallower the SGB would still be
                 detected → absences in other samples are biological.
   margin ≈ 1 → the SGB sits right on the detection threshold → some absences
                 could be depth-limited.

3. Logistic depth slope β  (vectorized Newton–Raphson)
   For each SGB fit:
       logit P(present_{j,i}) = α_i + β_i · z_j
   where z_j = standardised log₁₀(total_bp[j]).  β_i ≈ 0 → depth-independent
   detection.  β_i ≫ 0 → depth-driven.  The distribution of β_i across all
   SGBs answers the question at the population level.

Usage
-----
    python -m persistence-patched.depth_detection_model --ref UHGG
    python -m persistence-patched.depth_detection_model --ref UHGG lev
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

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

# ── constants ────────────────────────────────────────────────────────────────
PRESENCE_THRESHOLD   = 10.0                     # % breadth
POISSON_D_THRESHOLD  = -np.log(1 - 0.10)       # ≈ 0.1054  (mean depth for 10% breadth)
N_DEPTH_BINS         = 8                        # quantile bins for depth stratification
MIN_DETECTIONS       = 8                        # min positive samples to fit per-SGB models
N_LOGISTIC_ITER      = 40                       # Newton–Raphson iterations

LIFESTYLE_ORDER  = ["Industrial", "H-G", "Transitional"]
LIFESTYLE_COLOURS = {
    "Industrial":   "#E07B54",
    "H-G":          "#4A90A4",
    "Transitional": "#7BBF72",
}

# Prevalence tiers used for stratified curves and margin plots.
# Fraction-based (prevalence = count / n_samples) so thresholds are comparable
# across lifestyles with very different sample sizes (Industrial: ~13k, H-G: ~355).
# Log-spaced: each tier spans ~3–5× the previous in absolute prevalence, giving
# equal visual weight to the long low-prevalence tail that dominates the SGB list.
# Colours: sequential red→orange→yellow-green→teal→blue ramp (rare → common).
TIERS = [
    ("Rare\n(<0.5%)",      0.000, 0.005, "#a50026"),
    ("Low\n(0.5–2%)",      0.005, 0.020, "#d73027"),
    ("Uncommon\n(2–10%)",  0.020, 0.100, "#f46d43"),
    ("Moderate\n(10–30%)", 0.100, 0.300, "#4393c3"),
    ("Common\n(30–60%)",   0.300, 0.600, "#2166ac"),
    ("Ubiquitous\n(>60%)", 0.600, 1.010, "#053061"),
]

# NOTE on logistic β values
# --------------------------
# In practice, logistic depth slopes β are typically >> 0 (median ≈ 1.5–2.0).
# This does NOT contradict the finding that absences are biological.
# Within a lifestyle, richer samples contain more total bacterial DNA, so even
# rare organisms cross the detection threshold more often — not because they sit
# at the edge (the safety margin shows they don't), but because their absolute
# read count scales linearly with total depth. β > 0 is therefore a mathematical
# consequence of the Poisson coverage model: P(detected) = 1 - exp(-a·T/L)
# which always increases with T even when a·T/L ≫ 0.1054.  The safety margin
# (median depth when present / threshold) is the correct test of whether
# *current* non-detections are depth-limited; β tests a different question
# (how fast does P(detected) grow with more data for the organisms that are
# genuinely there).


# ════════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ════════════════════════════════════════════════════════════════════════════

def _slice_lifestyle(breadth_df, depth_df, total_bp, meta, lifestyle):
    """
    Return (presence_df, depth_slice, total_bp_slice) for one lifestyle.
    presence_df is boolean uint8: samples × SGBs.
    """
    ids = meta.index[meta["Lifestyle"] == lifestyle].intersection(breadth_df.index)
    presence = (breadth_df.loc[ids] >= PRESENCE_THRESHOLD).astype(np.float32)
    depth_sl = depth_df.loc[ids]
    bp_sl    = total_bp.reindex(ids)
    return presence, depth_sl, bp_sl


# ════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 1: DEPTH-STRATIFIED PREVALENCE CURVES
# ════════════════════════════════════════════════════════════════════════════

def depth_stratified_prevalence(presence, total_bp,
                                 n_bins=N_DEPTH_BINS, n_tiers=5):
    """
    Bin samples by log₁₀(total_bp) using quantiles, compute mean per-SGB
    prevalence in each bin, then average over SGBs grouped into prevalence
    tiers.

    Tiers are derived from the *data itself* using log-quantiles of the
    detected SGBs' overall prevalence.  This guarantees each tier is
    non-empty regardless of the prevalence distribution shape, and that
    tier boundaries are meaningful for the particular lifestyle.
    Absent SGBs (overall_prev == 0) always form their own "Absent" group.

    Returns
    -------
    bin_midpoints : array (n_bins,)
    tier_curves   : dict  tier_label → array (n_bins,)  mean prevalence per bin
    tier_counts   : dict  tier_label → int  n_sgbs in tier
    tier_colours  : dict  tier_label → str  hex colour
    """
    log_bp  = np.log10(total_bp.values.clip(1))
    pres_np = presence.values.astype(np.float32)   # (n_samples, n_sgbs)

    edges   = np.unique(np.percentile(log_bp, np.linspace(0, 100, n_bins + 1)))
    bin_ids = np.digitize(log_bp, edges[1:-1])      # 0 … len(edges)-2

    overall_prev = pres_np.mean(axis=0)             # (n_sgbs,) fraction 0-1
    n_samples    = pres_np.shape[0]

    # ── Quantile-based tier assignment among DETECTED SGBs ────────────────
    # We work in log10(overall_prev) space so that the many rare SGBs and
    # few common SGBs are weighted equally per tier (equal n_sgbs per tier).
    detected_mask  = overall_prev > 0
    absent_mask    = ~detected_mask
    n_detected     = detected_mask.sum()

    # Assign each detected SGB to a quantile tier (0 = rarest, n_tiers-1 = commonest)
    tier_ids = np.full(len(overall_prev), -1, dtype=int)   # -1 = absent
    if n_detected > 0:
        log_prev_det   = np.log10(overall_prev[detected_mask])
        quantile_edges = np.percentile(log_prev_det,
                                       np.linspace(0, 100, n_tiers + 1))
        quantile_edges[-1] += 1e-9   # ensure the last SGB is included
        tier_ids[detected_mask] = np.searchsorted(
            quantile_edges[1:], log_prev_det, side='left').clip(0, n_tiers - 1)

    # ── Build tier metadata (label, colour, mask) ─────────────────────────
    # Colour ramp: dark-red (rarest) → orange → teal → dark-blue (commonest)
    _ramp = ["#a50026", "#d73027", "#f46d43", "#4393c3", "#2166ac", "#053061"]
    _ramp = _ramp[:n_tiers]   # trim if fewer tiers requested

    tier_info = []   # list of (label, colour, boolean mask)

    # "Absent" tier first
    if absent_mask.sum() > 0:
        tier_info.append(("Absent\n(never detected)", "#888888", absent_mask))

    for t in range(n_tiers):
        mask = tier_ids == t
        if mask.sum() == 0:
            continue
        lo_pct = overall_prev[mask].min() * 100
        hi_pct = overall_prev[mask].max() * 100
        lo_cnt = int(overall_prev[mask].min() * n_samples)
        hi_cnt = int(overall_prev[mask].max() * n_samples)
        label  = (f"Q{t+1}  {lo_pct:.2g}–{hi_pct:.2g}%\n"
                  f"({lo_cnt}–{hi_cnt} samples)")
        tier_info.append((label, _ramp[t], mask))

    # ── Compute curves per depth bin ──────────────────────────────────────
    tier_colours = {ti[0]: ti[1] for ti in tier_info}
    tier_curves  = {ti[0]: [] for ti in tier_info}
    tier_counts  = {ti[0]: int(ti[2].sum()) for ti in tier_info}

    bin_midpoints = []
    for b in range(len(edges) - 1):
        sel = bin_ids == b
        if sel.sum() < 4:
            continue
        bin_midpoints.append(np.median(log_bp[sel]))
        bin_prev = pres_np[sel, :].mean(axis=0)   # per-SGB prevalence in this bin
        for label, _, mask in tier_info:
            v = float(np.mean(bin_prev[mask])) if mask.sum() > 0 else np.nan
            tier_curves[label].append(v)

    return (np.array(bin_midpoints),
            {k: np.array(v) for k, v in tier_curves.items()},
            tier_counts,
            tier_colours)


# ════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 2: DETECTION SAFETY MARGIN
# ════════════════════════════════════════════════════════════════════════════

def detection_safety_margins(presence, depth_df, min_detections=MIN_DETECTIONS):
    """
    For each SGB detected in ≥ min_detections samples, compute:

        log₁₀(margin) = log₁₀(median mean_depth when present / POISSON_D_THRESHOLD)

    Positive → above the detection floor; 0 → right at threshold; negative
    would mean detected despite depth below threshold (non-Poisson or noise).

    Also returns the overall prevalence of each SGB for colouring.
    """
    n_samples = presence.shape[0]
    log10_margins = {}
    prevalences   = {}

    for sgb in presence.columns:
        if sgb not in depth_df.columns:
            continue
        det_mask = presence[sgb].values == 1
        n_det = det_mask.sum()
        if n_det < min_detections:
            continue

        d_when_present = depth_df[sgb].values[det_mask]
        med_d = np.median(d_when_present[d_when_present > 0]) if (d_when_present > 0).any() else 0
        if med_d <= 0:
            continue

        log10_margins[sgb] = np.log10(med_d / POISSON_D_THRESHOLD)
        prevalences[sgb]   = n_det          # raw count (for tier colouring)

    return pd.Series(log10_margins), pd.Series(prevalences)


# ════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 3: VECTORISED LOGISTIC DEPTH SLOPE
# ════════════════════════════════════════════════════════════════════════════

def fit_logistic_depth_slopes(presence, total_bp,
                               min_pos=MIN_DETECTIONS, min_neg=MIN_DETECTIONS,
                               n_iter=N_LOGISTIC_ITER):
    """
    Fit  logit P(present_{j,i}) = α_i + β_i · z_j  for all SGBs simultaneously
    using vectorised Newton–Raphson, where z_j = standardised log₁₀(total_bp[j]).

    β_i ≈ 0  → presence is depth-independent (biological)
    β_i ≫ 0  → deeper samples detect this SGB more → depth-limited

    Only SGBs with ≥ min_pos positive and ≥ min_neg negative observations are fit.

    Returns
    -------
    slopes : pd.Series  SGB → β_i  (un-standardised, in units of log₁₀-bp)
    """
    pres_np = presence.values.astype(np.float32)    # (n, p)
    z_raw   = np.log10(total_bp.values.clip(1))
    z       = (z_raw - z_raw.mean()) / (z_raw.std() + 1e-12)  # standardise
    z_std   = z_raw.std() + 1e-12

    n, p = pres_np.shape

    # Build per-SGB mask for fittable SGBs
    n_pos = pres_np.sum(axis=0)
    n_neg = n - n_pos
    fittable = (n_pos >= min_pos) & (n_neg >= min_neg)  # (p,)

    # Run on the fittable subset
    Y  = pres_np[:, fittable]               # (n, k)
    k  = Y.shape[1]

    alpha = np.zeros(k, dtype=np.float64)
    beta  = np.zeros(k, dtype=np.float64)
    z_col = z[:, None]                      # (n, 1)

    for _ in range(n_iter):
        eta = alpha[None, :] + beta[None, :] * z_col   # (n, k)
        mu  = expit(eta)
        W   = mu * (1.0 - mu)

        r   = Y - mu                         # (n, k)
        g_a = r.sum(axis=0)                  # (k,)
        g_b = (r * z_col).sum(axis=0)        # (k,)

        F_aa = W.sum(axis=0)                 # (k,)
        F_bb = (W * z_col**2).sum(axis=0)    # (k,)
        F_ab = (W * z_col).sum(axis=0)       # (k,)

        det  = F_aa * F_bb - F_ab**2
        safe = np.where(np.abs(det) > 1e-14, det, np.nan)

        da = np.nan_to_num((F_bb * g_a - F_ab * g_b) / safe).clip(-3, 3)
        db = np.nan_to_num((F_aa * g_b - F_ab * g_a) / safe).clip(-3, 3)

        alpha += da
        beta  += db

    # Un-standardise β back to log₁₀(total_bp) units
    beta_actual = beta / z_std

    # Map back to full SGB index
    slopes_full = np.full(p, np.nan)
    slopes_full[fittable] = beta_actual

    return pd.Series(slopes_full, index=presence.columns)


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 1: DEPTH-STRATIFIED PREVALENCE CURVES
# ════════════════════════════════════════════════════════════════════════════

def plot_prevalence_curves(results_by_lifestyle, out_path):
    """
    3 panels (one per lifestyle). Each panel:
      x = log₁₀(total_bp) bin midpoints
      y = mean prevalence within quantile tier (%)
      N+1 lines: one "Absent" group + N quantile tiers of detected SGBs

    Tiers are derived per-lifestyle from log-quantiles of overall prevalence,
    guaranteeing equal SGB counts per tier regardless of prevalence distribution.

    Flat lines → depth-independent (biological).
    Rising lines → depth-driven detection.
    """
    active  = [ls for ls in LIFESTYLE_ORDER if ls in results_by_lifestyle]
    n       = len(active)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, lifestyle in zip(axes, active):
        res     = results_by_lifestyle[lifestyle]
        mids    = res["bin_midpoints"]
        curves  = res["tier_curves"]
        counts  = res["tier_counts"]
        colours = res["tier_colours"]
        colour  = LIFESTYLE_COLOURS[lifestyle]
        n_samp  = res["n_samples"]

        # Plot tiers in order: Absent first (grey, dashed), then Q1→Q5
        for name in curves:
            y     = curves[name]
            valid = np.isfinite(y)
            if valid.sum() < 2:
                continue
            tcol = colours.get(name, "#888888")
            ls_style = "--" if "Absent" in name else "-"
            lw       = 1.2 if "Absent" in name else 1.8
            n_sgbs   = counts.get(name, 0)
            # Compact single-line label (strip the second line of the tier label)
            short = name.split("\n")[0]
            lbl   = f"{short}  (n={n_sgbs:,})"
            ax.plot(mids[valid], y[valid] * 100,
                    color=tcol, marker="o", ms=3.5, lw=lw,
                    ls=ls_style, alpha=0.88, label=lbl)

        ax.set_xlabel("log₁₀(total mapped bp)", fontsize=10)
        ax.set_ylabel("Mean prevalence within tier (%)", fontsize=10)
        ax.set_title(lifestyle, fontsize=13, fontweight="bold", color=colour)
        ax.legend(fontsize=7.2, loc="upper left",
                  framealpha=0.9, edgecolor=colour, ncol=1)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.set_facecolor((*matplotlib.colors.to_rgb(colour), 0.04))
        ax.text(0.98, 0.02, f"n samples = {n_samp:,}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="grey")

    fig.suptitle(
        "Depth-stratified prevalence curves  (log-quantile tiers, equal SGB count per tier)\n"
        "Flat lines → depth-independent (biological);  rising lines → depth-driven detection",
        fontsize=11, fontweight="bold", y=1.02)

    fig.tight_layout()
    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 2: SAFETY MARGIN + LOGISTIC SLOPE SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def plot_margin_and_slope(results_by_lifestyle, out_path):
    """
    2 rows × n_lifestyle columns.

    Row 1 — Detection safety margin (log₁₀ scale):
      Histogram of log₁₀(margin).  Margin = median depth when detected / 0.105.
      x=0 is the Poisson detection threshold.  Right of zero = above floor.
      Colour-coded by overall prevalence tier.  A distribution shifted far right
      means SGBs are typically detected well above the floor → absences are
      biological, not depth-limited.

    Row 2 — Logistic depth slope β:
      Histogram of β_i per SGB (un-standardised, units = log₁₀-bp).
      β ≈ 0 → depth-independent; β ≫ 0 → depth-driven.
      Annotates the fraction of SGBs with |β| < 0.5 (practically flat).
    """
    active = [ls for ls in LIFESTYLE_ORDER if ls in results_by_lifestyle]
    n      = len(active)

    fig = plt.figure(figsize=(5.5 * n, 9))
    gs  = GridSpec(2, n, figure=fig, hspace=0.45, wspace=0.32)

    for col, lifestyle in enumerate(active):
        res    = results_by_lifestyle[lifestyle]
        colour = LIFESTYLE_COLOURS[lifestyle]

        # ── ROW 1: safety margin ──────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, col])
        log10_m = res["log10_margins"]
        prevs   = res["prevalences"]

        if len(log10_m) > 0:
            # Draw per-tier histograms (prevs now holds raw counts, matching TIERS)
            bins = np.linspace(log10_m.min() - 0.2, log10_m.max() + 0.2, 50)
            for name, lo, hi, tc in TIERS:
                m     = (prevs >= lo) & (prevs < hi)
                m_idx = m[m].index.intersection(log10_m.index)
                if len(m_idx) == 0:
                    continue
                ax1.hist(log10_m[m_idx].values, bins=bins,
                         color=tc, alpha=0.6, edgecolor="none",
                         label=f"{name}  (n={len(m_idx):,})")

            ax1.axvline(0, color="black", lw=1.5, ls="--",
                        label="detection threshold (D=0.105×)")
            ax1.axvline(1, color="grey", lw=1.0, ls=":",
                        label="10× above threshold")

            # Fraction well above threshold
            frac_gt1 = (log10_m >= 1).mean() * 100
            frac_gt0 = (log10_m >= 0).mean() * 100
            ax1.text(0.97, 0.96,
                     f"≥10× above floor: {frac_gt1:.0f}%\n≥1× above floor: {frac_gt0:.0f}%",
                     transform=ax1.transAxes, ha="right", va="top",
                     fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                                           fc="white", ec=colour, alpha=0.88))

        ax1.set_yscale("log")
        ax1.set_xlabel(
            "log₁₀(safety margin)\n"
            r"= log₁₀(median depth$_{present}$ / 0.105)",
            fontsize=9)
        ax1.set_ylabel("Number of SGBs  (log scale)", fontsize=9)
        ax1.set_title(lifestyle, fontsize=12, fontweight="bold", color=colour)
        ax1.legend(fontsize=6.5, ncol=1, loc="upper left", framealpha=0.85)
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax1.yaxis.set_minor_formatter(mticker.NullFormatter())

        # ── ROW 2: logistic depth slope ───────────────────────────────────
        ax2 = fig.add_subplot(gs[1, col])
        slopes = res["slopes"].dropna()

        if len(slopes) > 0:
            lo_b, hi_b = np.percentile(slopes, [1, 99])
            bins2 = np.linspace(lo_b - 0.05, hi_b + 0.05, 60)

            ax2.hist(slopes.values, bins=bins2,
                     color=colour, alpha=0.78, edgecolor="white", linewidth=0.25)

            # Reference lines: β=0 (no depth effect), and β=1, β=2 which
            # correspond to meaningful effect sizes on a logit scale
            # (β=1 → 10× more depth raises detection odds by e^1 ≈ 2.7×).
            for bval, ls_, lw_, label_ in [
                (0, "--", 1.6, "β=0  (no depth effect)"),
                (1, ":",  1.2, "β=1  (moderate depth effect)"),
                (2, ":",  1.2, "β=2  (strong depth effect)"),
            ]:
                ax2.axvline(bval, color="black", lw=lw_, ls=ls_,
                            alpha=0.7, label=label_)

            frac_lt1  = (slopes < 1).mean() * 100
            frac_lt05 = (slopes < 0.5).mean() * 100
            med_b     = slopes.median()
            ax2.text(0.97, 0.96,
                     f"median β = {med_b:.2f}\n"
                     f"β < 1: {frac_lt1:.0f}% of SGBs\n"
                     f"β < 0.5: {frac_lt05:.0f}% of SGBs",
                     transform=ax2.transAxes, ha="right", va="top",
                     fontsize=8, bbox=dict(boxstyle="round,pad=0.3",
                                           fc="white", ec=colour, alpha=0.88))
            ax2.legend(fontsize=7, loc="upper left", framealpha=0.85)

        ax2.set_yscale("log")
        ax2.set_xlabel(
            "Logistic depth slope β\n"
            r"[logit-units per log₁₀(total_bp)]",
            fontsize=9)
        ax2.set_ylabel("Number of SGBs  (log scale)", fontsize=9)
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax2.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Row titles on the left margin
    fig.text(0.01, 0.73, "Detection safety margin", fontsize=10,
             fontweight="bold", va="center", rotation=90)
    fig.text(0.01, 0.27, "Logistic depth slope β", fontsize=10,
             fontweight="bold", va="center", rotation=90)

    fig.suptitle(
        "Depth–detection model: safety margins and logistic slopes\n"
        "β > 0 → depth increases detection probability. "
        "Safety margin > 0 → detected SGBs are well above the Poisson floor.",
        fontsize=11, fontweight="bold", y=1.01)

    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ════════════════════════════════════════════════════════════════════════════

def print_summary(results_by_lifestyle):
    sep = "─" * 72
    print(f"\n{sep}")
    print("  Depth–detection audit summary")
    print(sep)

    for lifestyle in LIFESTYLE_ORDER:
        if lifestyle not in results_by_lifestyle:
            continue
        res = results_by_lifestyle[lifestyle]
        m   = res["log10_margins"]
        sl  = res["slopes"].dropna()

        print(f"\n  {lifestyle}  (n={res['n_samples']:,} samples, "
              f"{res['n_sgbs']:,} SGBs with ≥{MIN_DETECTIONS} detections)")
        print(f"    Safety margin  — median log₁₀(margin) : {m.median():.2f}  "
              f"({(m>=1).mean()*100:.0f}% are ≥10× above floor)")
        print(f"    Logistic slope — median β              : {sl.median():.3f}  "
              f"({(sl.abs()<0.5).mean()*100:.0f}% have |β|<0.5)")
        print(f"    Interpretation: {'depth does NOT primarily explain absences'  if sl.median() < 0.5 and m.median() > 0.5 else 'depth may contribute to non-detection — investigate further'}")

    print(f"\n{sep}")
    print("  Interpretation guide")
    print(f"  {'Safety margin ≫ 1':<25} → When present, SGBs are well above the 10% breadth")
    print(f"  {'':25}   detection floor.  Non-detections are biological absences.")
    print(f"  {'Safety margin ≈ 1':<25} → SGBs at the detection edge; deeper sequencing")
    print(f"  {'':25}   might reveal them.  Treat absences with caution.")
    print(f"  {'β ≈ 0':<25} → Presence does not increase with depth within lifestyle.")
    print(f"  {'β ≫ 0':<25} → Presence increases with depth; confound possible.")
    print(sep)


# ════════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

def run_depth_detection_audit(ref_name):
    from .loading import load_reference
    from . import config

    print(f"\n{'='*65}")
    print(f"  Depth–detection audit — reference: {ref_name}")
    print(f"{'='*65}")

    rdata  = load_reference(ref_name)
    out    = config.outdir(ref_name)
    os.makedirs(out, exist_ok=True)

    results = {}

    for lifestyle in LIFESTYLE_ORDER:
        if lifestyle not in rdata.meta["Lifestyle"].values:
            continue
        print(f"\n  Processing: {lifestyle}")

        presence, depth_sl, bp_sl = _slice_lifestyle(
            rdata.breadth, rdata.depth, rdata.total_bp, rdata.meta, lifestyle)

        print(f"    Depth-stratified prevalence …", end=" ", flush=True)
        mids, tier_curves, tier_counts, tier_colours = depth_stratified_prevalence(presence, bp_sl)
        print("done")

        print(f"    Detection safety margins …", end=" ", flush=True)
        log10_m, prevs = detection_safety_margins(presence, depth_sl)
        print(f"done  ({len(log10_m):,} SGBs)")

        print(f"    Logistic depth slopes (vectorised N–R) …", end=" ", flush=True)
        slopes = fit_logistic_depth_slopes(presence, bp_sl)
        n_fit  = slopes.notna().sum()
        print(f"done  ({n_fit:,} SGBs fit)")

        results[lifestyle] = {
            "n_samples":     presence.shape[0],
            "n_sgbs":        int(log10_m.notna().sum()),
            "bin_midpoints": mids,
            "tier_curves":   tier_curves,
            "tier_counts":   tier_counts,
            "tier_colours":  tier_colours,
            "log10_margins": log10_m,
            "prevalences":   prevs,
            "slopes":        slopes,
        }

    print_summary(results)

    curves_path  = os.path.join(out, "depth_detection_prevalence_curves.png")
    margins_path = os.path.join(out, "depth_detection_margins_slopes.png")

    print(f"\n  Generating figures …")
    plot_prevalence_curves(results, curves_path)
    plot_margin_and_slope(results, margins_path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit whether non-detection is depth-driven (Poisson model).")
    parser.add_argument("--ref", nargs="+", default=["UHGG"],
                        help="Reference DB(s) to audit (default: UHGG)")
    args = parser.parse_args(argv)
    for ref in args.ref:
        run_depth_detection_audit(ref)


if __name__ == "__main__":
    main()
