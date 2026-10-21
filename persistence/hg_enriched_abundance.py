"""
hg_enriched_abundance.py
=========================
Identifies SGBs enriched in Hunter-Gatherers (H-G) relative to Industrial
populations, then characterises their prevalence and Poisson-estimated
relative abundance across four sub-groups:

    Hadza  |  Other H-G  |  Transitional  |  Industrial

Poisson abundance estimation
-----------------------------
Given observed breadth b ∈ [0, 1), the Poisson model for random read
placement gives mean depth d = −ln(1 − b).  From the depth and genome
length we get relative abundance:

    RA = d · L / T_bp

where L is genome length and T_bp is total mapped bases for that sample.
For undetected samples (b = 0), the SGB abundance is right-censored at the
detection floor  d_floor = −ln(1 − breadth_threshold) ≈ 0.105 (at 10%).

This uses only the breadth matrix (as requested) — the depth matrix is NOT
used.  The breadth-derived d is the Poisson MLE for mean depth.

Selection criterion
--------------------
HG prevalence > hg_prev_min  AND  Industrial prevalence < ind_prev_max
(defaults: 0.10 and 0.05, yielding ~1130 SGBs).

Outputs
-------
  hg_enriched_heatmap_<REF>.png    — ranked heatmap (all selected SGBs)
  hg_enriched_detail_<REF>.png     — abundance distributions (top 40 SGBs)
  hg_enriched_sgbs_<REF>.csv       — per-SGB statistics table

Usage
-----
    python -m persistence.hg_enriched_abundance --ref UHGG
    python -m persistence.hg_enriched_abundance --ref UHGG --hg-min 0.10 --ind-max 0.05
    python -m persistence.hg_enriched_abundance --ref UHGG --top 40
"""

import argparse
import os
import warnings

import zlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

warnings.filterwarnings("ignore")

try:
    from .plot_utils import apply_display_text
except ImportError:
    def apply_display_text(fig):
        for artist in fig.findobj(match=matplotlib.text.Text):
            text = artist.get_text()
            for old in ("Agro-Pastorialist", "Transitional"):
                text = str(text).replace(old, "Agro-Pastoralist")
            artist.set_text(text)
        return fig

BREADTH_THRESHOLD = 10.0        # % — detection threshold
D_FLOOR           = -np.log(1.0 - BREADTH_THRESHOLD / 100.0)  # ≈ 0.1054

GROUP_ORDER  = ["Hadza", "H-G other", "Transitional", "Industrial"]
GROUP_COLOURS = {
    "Hadza":        "#CC79A7",   # pink
    "H-G other":    "#D55E00",   # vermillion
    "Transitional": "#E69F00",   # orange
    "Industrial":   "#56B4E9",   # sky blue
}


# ════════════════════════════════════════════════════════════════════════════
#  GROUP MEMBERSHIP  (mirrors pipeline logic)
# ════════════════════════════════════════════════════════════════════════════

def _hadza_mask(meta):
    mask = pd.Series(False, index=meta.index)
    for col in ["Population", "Detailed location", "Sample ID",
                "StudyFolder", "Author"]:
        if col in meta.columns:
            mask |= meta[col].astype(str).str.contains(
                "hadza", case=False, na=False)
    if not mask.any():
        for cc in ["Country", "Country (full)", "Country_norm"]:
            if cc in meta.columns:
                tz   = meta[cc].astype(str).str.contains(
                    "TZA|Tanzania", case=False, na=False)
                mask = tz & (meta["Lifestyle"] == "H-G")
                if mask.any():
                    break
    return mask


def build_groups(meta, breadth_df):
    """
    Returns dict: group_name → sorted list of sample IDs (intersection
    with breadth_df.index already applied).
    """
    idx = breadth_df.index

    hg_ids   = meta.index[meta["Lifestyle"] == "H-G"].intersection(idx)
    hadza_m  = _hadza_mask(meta.loc[hg_ids])
    hadza_ids    = hg_ids[hadza_m]
    hg_other_ids = hg_ids[~hadza_m]

    tr_ids  = meta.index[meta["Lifestyle"] == "Transitional"].intersection(idx)
    ind_ids = meta.index[meta["Lifestyle"] == "Industrial"].intersection(idx)

    groups = {
        "Hadza":        hadza_ids,
        "H-G other":    hg_other_ids,
        "Transitional": tr_ids,
        "Industrial":   ind_ids,
    }
    for gname, ids in groups.items():
        print(f"  {gname:<16} {len(ids):>6,} samples")
    return groups


# ════════════════════════════════════════════════════════════════════════════
#  POISSON ABUNDANCE ESTIMATION
# ════════════════════════════════════════════════════════════════════════════

def poisson_depth_from_breadth(breadth_frac):
    """d_hat = −ln(1−b), the Poisson MLE for mean depth given breadth."""
    b = np.asarray(breadth_frac, float).clip(0, 0.9999)
    return -np.log1p(-b)


def compute_group_abundance(breadth_df, lengths, group_ids, sgbs):
    """
    Compute Poisson-estimated relative abundance from the breadth matrix only.

    For each (sample, SGB):
        d̂_SGB  = −ln(1 − b_SGB / 100)        Poisson MLE for mean depth
        RA_SGB  = d̂_SGB × L_SGB / T̂           relative abundance

    where T̂ = Σ_j d̂_j × L_j  is the breadth-derived estimate of total mapped
    bases (summed over ALL SGBs in the breadth matrix, not just the selected ones).

    Why breadth-only?
    -----------------
    Using data.abundance (depth-based) should be equivalent in principle but
    requires the sample index of the depth file to match the breadth file exactly.
    Any dtype mismatch in the pandas index silently produces NaN after reindex(),
    inflating RA to ~1.  Computing T̂ from the breadth matrix itself is fully
    self-consistent and requires no cross-file alignment.

    For undetected SGBs (b = 0) d̂ = 0 → RA = 0.  These are censored
    observations, not missing values.

    Parameters
    ----------
    breadth_df : full sample × SGB breadth DataFrame (% values 0-100)
    lengths    : SGB → genome length in bp
    group_ids  : sample IDs to include (intersected with breadth_df.index)
    sgbs       : list of SGB IDs to return columns for

    Returns
    -------
    DataFrame (n_samples × n_sgbs) of RA values; 0 = undetected.
    """
    ids = group_ids.intersection(breadth_df.index)
    if len(ids) == 0:
        return pd.DataFrame(index=pd.Index([]), columns=sgbs)

    # Subset to the group samples
    b_all = breadth_df.loc[ids].values.astype(float) / 100.0   # (n_samp, n_all_sgbs)
    d_all = -np.log1p(-b_all.clip(0, 0.9999))                  # Poisson d̂ per cell

    # Genome lengths aligned to ALL breadth columns
    L_all = lengths.reindex(breadth_df.columns).values.astype(float)
    L_all = np.where(np.isfinite(L_all), L_all, 0.0)

    # Per-sample total estimated mapped bases  T̂_i = Σ_j d̂_ij × L_j
    T_hat = d_all.dot(L_all)              # (n_samp,)
    T_hat = np.where(T_hat > 0, T_hat, np.nan)

    # Extract only the requested SGBs for the output
    valid  = [s for s in sgbs if s in breadth_df.columns]
    col_ix = [breadth_df.columns.get_loc(s) for s in valid]
    L_sel  = L_all[col_ix]               # (n_sel,)
    d_sel  = d_all[:, col_ix]            # (n_samp, n_sel)

    RA = d_sel * L_sel[None, :] / T_hat[:, None]   # (n_samp, n_sel)
    return pd.DataFrame(RA, index=ids, columns=valid)


# ════════════════════════════════════════════════════════════════════════════
#  HADZA-SPECIFICITY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def hadza_specificity(breadth_df, groups, summary,
                      hadza_min=0.10,
                      hg_other_max=0.01,
                      transitional_max=0.01,
                      industrial_max=0.01):
    """
    Classify H-G–enriched SGBs by how Hadza-specific they are.

    Three tiers:
        Tier 1 — Hadza-specific:
            Hadza ≥ hadza_min  AND  H-G other < hg_other_max
            AND  Transitional < transitional_max  AND  Industrial < industrial_max

        Tier 2 — H-G specific (shared across HG but not NI):
            Hadza ≥ hadza_min  AND  H-G other ≥ hg_other_max
            AND  Transitional < transitional_max  AND  Industrial < industrial_max

        Tier 3 — Present in NI (Transitional or Industrial detectable):
            All remaining selected SGBs.

    Returns an augmented copy of summary with a "tier" column, plus
    a dict of per-tier counts and threshold parameters used.
    """
    def _prev(gname):
        col = f"prev_{gname}"
        if col in summary.columns:
            return summary[col]
        # fallback: compute from breadth_df
        gi = groups.get(gname, pd.Index([])).intersection(breadth_df.index)
        if len(gi) == 0:
            return pd.Series(0.0, index=summary.index)
        return (breadth_df.loc[gi] >= BREADTH_THRESHOLD).mean(axis=0).reindex(summary.index).fillna(0)

    p_hadza = _prev("Hadza")
    p_hg_o  = _prev("H-G other")
    p_tr    = _prev("Transitional")
    p_ind   = _prev("Industrial")

    is_hadza_min = p_hadza >= hadza_min
    is_hgo_low   = p_hg_o  <  hg_other_max
    is_tr_low    = p_tr    <  transitional_max
    is_ind_low   = p_ind   <  industrial_max

    tier1 = is_hadza_min &  is_hgo_low  & is_tr_low & is_ind_low
    tier2 = is_hadza_min & ~is_hgo_low  & is_tr_low & is_ind_low
    tier3 = ~(tier1 | tier2)   # everything else in the selected set

    out = summary.copy()
    out["tier"] = "3_NI_detectable"
    out.loc[tier2, "tier"] = "2_HG_specific"
    out.loc[tier1, "tier"] = "1_Hadza_specific"

    counts = {
        "Tier 1 — Hadza-specific": int(tier1.sum()),
        "Tier 2 — H-G shared":     int(tier2.sum()),
        "Tier 3 — NI detectable":  int(tier3.sum()),
    }
    thresholds = dict(hadza_min=hadza_min, hg_other_max=hg_other_max,
                      transitional_max=transitional_max, industrial_max=industrial_max)
    return out, counts, thresholds


def plot_specificity(summary_tiered, out_path):
    """
    Two-panel figure quantifying Hadza specificity.

    Left  — Scatter: Hadza prev (x) vs H-G other prev (y), coloured by tier.
             Dashed lines mark the tier boundaries.
             Marginal histograms show count per tier.

    Right — Bar chart: how many SGBs in each tier; broken down by whether
             Transitional also detects them (shaded bar segment).
    """
    tier_colours = {
        "1_Hadza_specific": "#CC79A7",   # pink
        "2_HG_specific":    "#D55E00",   # vermillion
        "3_NI_detectable":  "#56B4E9",   # sky blue
    }
    tier_labels = {
        "1_Hadza_specific": "Tier 1: Hadza-specific\n(H-G other <1%, Trans <1%)",
        "2_HG_specific":    "Tier 2: H-G shared\n(Trans <1%)",
        "3_NI_detectable":  "Tier 3: NI detectable",
    }

    fig, (ax_s, ax_b) = plt.subplots(1, 2, figsize=(14, 7),
                                      gridspec_kw={"width_ratios": [3, 1.2],
                                                   "wspace": 0.25})

    for tier, grp in summary_tiered.groupby("tier"):
        col = tier_colours.get(tier, "grey")
        lbl = tier_labels.get(tier, tier)
        ax_s.scatter(grp["prev_Hadza"], grp["prev_H-G other"],
                     color=col, s=18, alpha=0.55, label=f"{lbl} (n={len(grp)})",
                     zorder=4, edgecolors="none")

    # Tier boundary lines
    ax_s.axhline(0.01, color="grey", lw=0.9, ls="--", alpha=0.55)
    ax_s.axvline(0.10, color="grey", lw=0.9, ls="--", alpha=0.55)
    ax_s.text(0.102, 0.98, "Hadza ≥ 10%", fontsize=7.5, color="grey",
              va="top", transform=ax_s.get_xaxis_transform())
    ax_s.text(0.98, 0.012, "H-G other = 1%", fontsize=7.5, color="grey",
              ha="right", transform=ax_s.get_yaxis_transform())

    ax_s.set_xlabel("Hadza prevalence", fontsize=11)
    ax_s.set_ylabel("H-G other prevalence", fontsize=11)
    ax_s.set_xlim(-0.01, 1.01); ax_s.set_ylim(-0.01, 1.01)
    ax_s.plot([0,1],[0,1], color="grey", lw=0.7, ls=":", alpha=0.4)
    ax_s.grid(True, ls="--", lw=0.3, alpha=0.35)
    ax_s.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax_s.set_title("Hadza-specificity tier classification", fontsize=11, fontweight="bold")

    # Bar chart: n per tier, with Transitional-detectable sub-bar
    tiers  = ["1_Hadza_specific", "2_HG_specific", "3_NI_detectable"]
    totals = [len(summary_tiered[summary_tiered["tier"] == t]) for t in tiers]
    tr_det = [int(((summary_tiered["tier"] == t) &
                   (summary_tiered["prev_Transitional"] >= 0.01)).sum())
              for t in tiers]
    xlbls  = [tier_labels[t].split("\n")[0] for t in tiers]
    cols   = [tier_colours[t] for t in tiers]
    xs     = np.arange(len(tiers))

    ax_b.bar(xs, totals, color=cols, alpha=0.75, edgecolor="white", lw=1.2,
             label="Total")
    ax_b.bar(xs, tr_det, color=cols, alpha=0.40, edgecolor=cols, lw=1.2,
             hatch="//", label="Transitional ≥ 1%")
    ax_b.set_xticks(xs); ax_b.set_xticklabels(xlbls, fontsize=9, rotation=15, ha="right")
    ax_b.set_ylabel("Number of SGBs", fontsize=10)
    ax_b.legend(fontsize=8.5, framealpha=0.9)
    ax_b.grid(True, axis="y", ls="--", lw=0.3, alpha=0.4)
    for xi, (tot, tr) in enumerate(zip(totals, tr_det)):
        ax_b.text(xi, tot + max(totals)*0.01, str(tot),
                  ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_b.set_title("SGBs per tier", fontsize=11, fontweight="bold")

    fig.suptitle(
        "Hadza-specificity of H-G–enriched SGBs\n"
        "Tier 1: enriched in Hadza but <1% in all other groups  |  "
        "Tier 2: shared across H-G but absent from NI  |  "
        "Tier 3: detectable in Transitional or Industrial",
        fontsize=10.5, fontweight="bold", y=1.01)

    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")



# ════════════════════════════════════════════════════════════════════════════
#  SGB SELECTION
# ════════════════════════════════════════════════════════════════════════════

def select_sgbs(breadth_df, groups, hg_prev_min=0.10, ind_prev_max=0.05):
    """
    Select SGBs with:
        HG prevalence  (Hadza + H-G other combined) >= hg_prev_min
        Industrial prevalence                        <= ind_prev_max

    Returns sorted list of SGB IDs and a summary DataFrame.
    """
    hg_ids  = groups["Hadza"].append(groups["H-G other"]) \
              if len(groups["H-G other"]) else groups["Hadza"]
    # pd.Index.append is deprecated in newer pandas; use union
    hg_ids  = groups["Hadza"].union(groups["H-G other"])
    ind_ids = groups["Industrial"]

    n_hg  = len(hg_ids.intersection(breadth_df.index))
    n_ind = len(ind_ids.intersection(breadth_df.index))

    if n_hg == 0 or n_ind == 0:
        return [], pd.DataFrame()

    hg_pres  = (breadth_df.loc[hg_ids.intersection(breadth_df.index)]
                >= BREADTH_THRESHOLD).mean(axis=0)
    ind_pres = (breadth_df.loc[ind_ids.intersection(breadth_df.index)]
                >= BREADTH_THRESHOLD).mean(axis=0)

    # Also compute per-subgroup
    rows = {}
    for gname, gids in groups.items():
        gi = gids.intersection(breadth_df.index)
        if len(gi) > 0:
            rows[gname] = (breadth_df.loc[gi] >= BREADTH_THRESHOLD).mean(axis=0)
        else:
            rows[gname] = pd.Series(0.0, index=breadth_df.columns)

    mask    = (hg_pres >= hg_prev_min) & (ind_pres <= ind_prev_max)
    sel     = hg_pres.index[mask].tolist()

    summary = pd.DataFrame({
        "hg_prevalence":  hg_pres[sel],
        "ind_prevalence": ind_pres[sel],
        "hg_minus_ind":   (hg_pres - ind_pres)[sel],
        **{f"prev_{g}": rows[g][sel] for g in GROUP_ORDER},
    }).sort_values("hg_minus_ind", ascending=False)

    print(f"\n  Selected {len(sel):,} SGBs  "
          f"(HG ≥ {hg_prev_min:.0%}, Industrial ≤ {ind_prev_max:.0%})")
    return summary.index.tolist(), summary


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — RANKED HEATMAP
# ════════════════════════════════════════════════════════════════════════════

def plot_heatmap(summary, n_top=None, out_path=None):
    """
    Left column: 4-group prevalence bars (stacked / grouped).
    Right four columns: Poisson-estimated log-mean RA heatmap, one col per group.

    Rows = SGBs sorted by HG − Industrial prevalence (descending).
    Only show n_top SGBs to keep the figure readable.
    """
    df = summary.copy()
    if n_top is not None:
        df = df.head(n_top)

    n_sgbs  = len(df)
    groups  = GROUP_ORDER

    # ── Heatmap data: mean log10(RA) when detected ──────────────────────────
    # We only have counts from the audit CSV at this point, not the full RA
    # matrices.  Use prevalence as a proxy for the heatmap until full RA data
    # is available from run_analysis().  The detail figure (Figure 2) shows
    # actual RA distributions.
    prev_mat = np.column_stack([df[f"prev_{g}"].values for g in groups])

    fig = plt.figure(figsize=(14, max(6, n_sgbs * 0.18 + 3)))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[2, 5], wspace=0.05)

    # ── Left: prevalence bars ────────────────────────────────────────────────
    ax_bar = fig.add_subplot(outer[0])
    y      = np.arange(n_sgbs)
    bar_h  = 0.78

    left = np.zeros(n_sgbs)
    for gname in groups:
        vals = df[f"prev_{gname}"].values * 100
        ax_bar.barh(y, vals, left=left, height=bar_h,
                    color=GROUP_COLOURS[gname], alpha=0.85,
                    label=gname)
        left += vals

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(df.index, fontsize=5.5 if n_sgbs > 60 else 7)
    ax_bar.set_xlabel("Cumulative prevalence (%)", fontsize=9)
    ax_bar.set_xlim(0, left.max() * 1.05)
    ax_bar.invert_yaxis()
    ax_bar.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    ax_bar.set_title("Prevalence per group", fontsize=9, fontweight="bold")

    # ── Right: heatmap ───────────────────────────────────────────────────────
    inner = GridSpecFromSubplotSpec(1, len(groups), subplot_spec=outer[1],
                                    wspace=0.02)
    cmap = plt.cm.YlOrRd

    for ci, gname in enumerate(groups):
        ax_h = fig.add_subplot(inner[ci])
        col  = prev_mat[:, ci][:, None]   # (n_sgbs, 1)
        im   = ax_h.imshow(col, aspect="auto", cmap=cmap,
                           vmin=0, vmax=1,
                           interpolation="nearest")
        ax_h.set_xticks([]); ax_h.set_yticks([])
        ax_h.set_title(gname, fontsize=8, fontweight="bold",
                       color=GROUP_COLOURS[gname], rotation=30, ha="left")
        if ci == len(groups) - 1:
            plt.colorbar(im, ax=ax_h, fraction=0.3, pad=0.04,
                         label="Prevalence")

    fig.suptitle(
        f"H-G–enriched SGBs: prevalence across sub-groups\n"
        f"({n_sgbs} SGBs, ranked by HG − Industrial prevalence)",
        fontsize=11, fontweight="bold", y=1.01)

    if out_path:
        apply_display_text(fig)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out_path}")
    return fig


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — ABUNDANCE DETAIL (top N SGBs)
# ════════════════════════════════════════════════════════════════════════════

def plot_abundance_detail(summary, ra_by_group, top_n, out_path):
    """
    For the top_n SGBs (by HG − Industrial prevalence), one row per SGB:

      LEFT  — Prevalence dot-and-line per group (x = fraction detected).
      RIGHT — log₁₀(RA) strip for detected samples, with IQR bar,
               median (◆) and mean (|) markers.

    The right panel x-axis is shared across all rows and formatted as
    plain powers of 10 (10⁻⁵, 10⁻⁴, …) so the tick labels are legible.
    """
    sgbs = summary.index[:top_n].tolist()
    n    = len(sgbs)

    fig_h = max(8, n * 0.52 + 2.5)
    fig, axes = plt.subplots(
        n, 2,
        figsize=(14, fig_h),
        gridspec_kw={"width_ratios": [1, 3], "wspace": 0.04, "hspace": 0.10},
        squeeze=False,
    )

    # Determine shared x-limits from all detected RA values
    all_log = []
    for gname in GROUP_ORDER:
        if gname not in ra_by_group:
            continue
        for sgb in sgbs:
            if sgb not in ra_by_group[gname].columns:
                continue
            det = ra_by_group[gname][sgb]
            det = det[det > 0]
            if len(det):
                all_log.extend(np.log10(det.values.clip(1e-12)))
    if all_log:
        x_lo = np.percentile(all_log,  1) - 0.3
        x_hi = np.percentile(all_log, 99) + 0.3
    else:
        x_lo, x_hi = -6, -1

    # Round to integer powers of 10 for clean tick labels
    x_lo = np.floor(x_lo)
    x_hi = np.ceil(x_hi)
    xticks = np.arange(x_lo, x_hi + 1, dtype=int)

    for row, sgb in enumerate(sgbs):
        ax_l = axes[row][0]
        ax_r = axes[row][1]

        ax_l.set_xlim(-0.02, 1.02)
        ax_r.set_xlim(x_lo - 0.1, x_hi + 0.1)
        ax_r.set_xticks(xticks)
        ax_r.set_xticklabels(
            [f"$10^{{{int(t)}}}$" for t in xticks],
            fontsize=7,
        )
        ax_r.tick_params(axis="x", which="both",
                         labelbottom=(row == n - 1))

        for gi, gname in enumerate(GROUP_ORDER):
            col   = GROUP_COLOURS[gname]
            pkey  = f"prev_{gname}"
            prev_val = float(summary.loc[sgb, pkey]) if pkey in summary.columns else 0.0

            # Prevalence panel
            ax_l.scatter([prev_val], [gi], color=col, s=55,
                         zorder=4, alpha=0.92, clip_on=False)
            ax_l.plot([0, prev_val], [gi, gi], color=col,
                      lw=1.8, alpha=0.45, zorder=3)

            # Abundance panel
            if gname not in ra_by_group:
                continue
            if sgb not in ra_by_group[gname].columns:
                continue
            ra_col   = ra_by_group[gname][sgb].dropna()
            detected = ra_col[ra_col > 0]
            if len(detected) == 0:
                continue

            log_ra = np.log10(detected.values.clip(1e-12))

            # Jittered strip
            # crc32, not hash(): CPython salts string hashing per process
            # unless PYTHONHASHSEED is set, and it is set nowhere here, so
            # this was the one seed in the codebase that changed on every
            # invocation -- the jitter in this strip was different in every
            # run of the same figure.
            rng_s = np.random.default_rng(
                zlib.crc32((gname + sgb).encode("utf-8")) & 0xFFFF)
            jit   = rng_s.uniform(-0.28, 0.28, len(log_ra))
            ax_r.scatter(log_ra, gi + jit, color=col, s=5,
                         alpha=0.30, zorder=3, lw=0)

            # IQR bar
            q25, q75 = np.percentile(log_ra, [25, 75])
            ax_r.plot([q25, q75], [gi, gi], color=col, lw=4.0,
                      alpha=0.40, zorder=4, solid_capstyle="round")

            # Median ◆
            med = float(np.median(log_ra))
            ax_r.scatter([med], [gi], color=col, s=60, zorder=6,
                         marker="D", edgecolors="white", lw=0.5)

            # Mean |
            mn = float(np.mean(log_ra))
            ax_r.plot([mn, mn], [gi - 0.30, gi + 0.30], color=col,
                      lw=2.0, alpha=0.85, zorder=5, solid_capstyle="round")

        # Y-axis labels on first row only
        ax_l.set_yticks(range(4))
        ax_r.set_yticks(range(4))
        ax_r.set_yticklabels([])
        if row == 0:
            ax_l.set_yticklabels(GROUP_ORDER, fontsize=7.5)
            ax_r.set_title(
                "log₁₀(RA from Poisson breadth)  "
                "[◆ = median  | = mean  ▬ = IQR  · = detected samples]",
                fontsize=8)
            ax_l.set_title("Prevalence", fontsize=8)
        else:
            ax_l.set_yticklabels([])

        ax_l.set_ylim(-0.7, 3.7)
        ax_r.set_ylim(-0.7, 3.7)

        # SGB label on right margin
        had_p = float(summary.loc[sgb, "prev_Hadza"]) * 100
        hgo_p = float(summary.loc[sgb, "prev_H-G other"]) * 100
        ind_p = float(summary.loc[sgb, "ind_prevalence"]) * 100
        tier  = summary.loc[sgb, "tier"] if "tier" in summary.columns else ""
        tier_short = {"1_Hadza_specific": "T1", "2_HG_specific": "T2",
                      "3_NI_detectable":  "T3"}.get(tier, "")
        ax_r.text(
            1.002, 0.5,
            f"{sgb}  {tier_short}\n"
            f"Hadza {had_p:.0f}% | H-G oth {hgo_p:.0f}% | Ind {ind_p:.1f}%",
            transform=ax_r.transAxes, va="center",
            fontsize=6.0, color="dimgrey",
        )

        for ax_ in [ax_l, ax_r]:
            ax_.spines["top"].set_visible(False)
            ax_.spines["right"].set_visible(False)
            ax_.grid(True, axis="x", ls="--", lw=0.3, alpha=0.35)

        if row == n - 1:
            ax_l.set_xlabel("Prevalence", fontsize=9)
            ax_r.set_xlabel("Relative abundance  (Poisson estimate from breadth)",
                            fontsize=9)

    fig.suptitle(
        f"Top {top_n} H-G–enriched SGBs: prevalence and Poisson-estimated abundance\n"
        "Left: fraction detected per group  |  "
        "Right: RA when detected (◆ median  | mean  ▬ IQR  · samples)  |  "
        "T1=Hadza-specific  T2=HG-shared  T3=NI-detectable",
        fontsize=10.5, fontweight="bold", y=1.005,
    )
    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — SUMMARY SCATTER + MARGINAL HISTOGRAMS
# ════════════════════════════════════════════════════════════════════════════

def plot_prevalence_scatter(summary, out_path):
    """
    Scatter: Hadza prevalence (x) vs H-G other prevalence (y).
    Points coloured by Industrial prevalence (cold=absent, warm=present).
    Size ∝ Transitional prevalence.
    Selected labelled SGBs annotated.
    Marginal histograms show prevalence distributions.
    """
    fig = plt.figure(figsize=(10, 9))
    gs  = GridSpec(2, 2, figure=fig, width_ratios=[4, 1],
                   height_ratios=[1, 4], hspace=0.04, wspace=0.04)
    ax_s  = fig.add_subplot(gs[1, 0])   # scatter
    ax_tx = fig.add_subplot(gs[0, 0], sharex=ax_s)  # top marginal
    ax_ry = fig.add_subplot(gs[1, 1], sharey=ax_s)  # right marginal

    x = summary["prev_Hadza"].values
    y = summary["prev_H-G other"].values
    c = summary["ind_prevalence"].values
    s = summary["prev_Transitional"].values * 250 + 10

    sc = ax_s.scatter(x, y, c=c, s=s, cmap="RdYlBu_r",
                      vmin=0, vmax=0.20,
                      alpha=0.65, edgecolors="none", zorder=3)
    plt.colorbar(sc, ax=ax_s, label="Industrial prevalence", fraction=0.035)

    ax_s.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--", alpha=0.5)
    ax_s.set_xlabel("Hadza prevalence", fontsize=11)
    ax_s.set_ylabel("H-G other prevalence", fontsize=11)
    ax_s.set_xlim(-0.01, 1.01); ax_s.set_ylim(-0.01, 1.01)
    ax_s.grid(True, ls="--", lw=0.3, alpha=0.4)

    # Marginals
    bins = np.linspace(0, 1, 40)
    ax_tx.hist(x, bins=bins, color=GROUP_COLOURS["Hadza"],    alpha=0.75)
    ax_ry.hist(y, bins=bins, color=GROUP_COLOURS["H-G other"],alpha=0.75,
               orientation="horizontal")
    ax_tx.set_ylabel("n SGBs", fontsize=8)
    ax_ry.set_xlabel("n SGBs", fontsize=8)
    plt.setp(ax_tx.get_xticklabels(), visible=False)
    plt.setp(ax_ry.get_yticklabels(), visible=False)

    # Legend for size
    for pct, lab in [(0.10, "10%"), (0.30, "30%"), (0.60, "60%")]:
        ax_s.scatter([], [], s=pct*250+10, c="grey", alpha=0.5,
                     label=f"Trans. {lab}")
    ax_s.legend(title="Transitional prev.", fontsize=8, title_fontsize=8,
                loc="upper left", framealpha=0.9)

    fig.suptitle(
        "H-G–enriched SGBs: Hadza vs Other H-G prevalence\n"
        "Colour = Industrial prevalence  |  Size = Transitional prevalence",
        fontsize=11, fontweight="bold")

    apply_display_text(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def run_analysis(ref_name, threshold=10.0, hg_prev_min=0.10,
                 ind_prev_max=0.05, top_n=40):
    from . import config
    from .loading import load_reference

    print(f"\n{'='*65}")
    print(f"  HG-enriched abundance — ref: {ref_name}  threshold: {threshold}%")
    print(f"  Selection: HG ≥ {hg_prev_min:.0%}  Industrial ≤ {ind_prev_max:.0%}")
    print(f"{'='*65}")

    data = load_reference(ref_name)
    out  = config.outdir(ref_name)
    os.makedirs(out, exist_ok=True)

    breadth_df = data.breadth   # samples × SGBs, values = % breadth
    lengths    = data.lengths   # SGB → genome length (bp)

    # ── Groups ───────────────────────────────────────────────────────────────
    print("\n  Sample counts per group:")
    groups = build_groups(data.meta, breadth_df)

    # ── Select SGBs ──────────────────────────────────────────────────────────
    sel_sgbs, summary = select_sgbs(breadth_df, groups,
                                    hg_prev_min=hg_prev_min,
                                    ind_prev_max=ind_prev_max)
    if not sel_sgbs:
        print("  No SGBs passed the selection filter.  "
              "Try relaxing --hg-min or --ind-max.")
        return

    # ── Poisson RA from breadth (self-contained, no depth-matrix dependency) ─
    print(f"\n  Computing Poisson RA for {len(sel_sgbs):,} SGBs × 4 groups...")
    ra_by_group = {}
    for gname, gids in groups.items():
        ra_by_group[gname] = compute_group_abundance(
            breadth_df, lengths, gids, sel_sgbs)
        n_samp = len(gids.intersection(breadth_df.index))
        n_det  = int((ra_by_group[gname] > 0).values.sum())
        print(f"    {gname:<16} {n_samp:>6,} samples  {n_det:,} SGB×sample detections")

    # Sanity check: median RA when detected should be ~1e-2 to 1e-4
    for gname in ["Hadza", "Industrial"]:
        ra = ra_by_group.get(gname)
        if ra is not None and not ra.empty:
            det_vals = ra.values[ra.values > 0]
            if len(det_vals):
                print(f"    {gname} median RA when detected: "
                      f"{np.median(det_vals):.2e}  "
                      f"(log10={np.log10(np.median(det_vals)):.1f})")

    # ── Per-SGB summary stats ────────────────────────────────────────────────
    for gname in GROUP_ORDER:
        ra = ra_by_group.get(gname)
        if ra is None or ra.empty:
            continue
        det  = ra > 0
        mean = ra.where(det).mean(axis=0)
        med  = ra.where(det).median(axis=0)
        summary[f"ra_mean_{gname}"]   = mean.reindex(summary.index)
        summary[f"ra_median_{gname}"] = med.reindex(summary.index)

    # ── Hadza-specificity classification ─────────────────────────────────────
    print("\n  Computing Hadza-specificity tiers...")
    summary_tiered, tier_counts, tier_thresh = hadza_specificity(
        breadth_df, groups, summary)
    for tier_name, n in tier_counts.items():
        print(f"    {tier_name}: {n:,} SGBs")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(out, f"hg_enriched_sgbs_{ref_name}.csv")
    summary_tiered.to_csv(csv_path)
    print(f"  Saved → {csv_path}")

    # ── Figure 1: Heatmap ────────────────────────────────────────────────────
    n_heatmap = min(len(summary), 120)
    hm_path = os.path.join(out, f"hg_enriched_heatmap_{ref_name}.png")
    print(f"\n  Plotting heatmap ({n_heatmap} SGBs)...")
    plot_heatmap(summary_tiered, n_top=n_heatmap, out_path=hm_path)

    # ── Figure 2: Abundance detail (top N) ───────────────────────────────────
    det_path = os.path.join(out, f"hg_enriched_detail_{ref_name}.png")
    print(f"  Plotting abundance detail (top {top_n} SGBs)...")
    plot_abundance_detail(summary_tiered, ra_by_group, top_n, det_path)

    # ── Figure 3: Prevalence scatter ─────────────────────────────────────────
    scat_path = os.path.join(out, f"hg_enriched_scatter_{ref_name}.png")
    print(f"  Plotting prevalence scatter...")
    plot_prevalence_scatter(summary_tiered, scat_path)

    # ── Figure 4: Hadza-specificity tier plot ────────────────────────────────
    spec_path = os.path.join(out, f"hg_enriched_specificity_{ref_name}.png")
    print(f"  Plotting Hadza-specificity tiers...")
    plot_specificity(summary_tiered, spec_path)

    print(f"\n  Done.  Outputs in {out}/")
    return summary_tiered


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="H-G–enriched SGB prevalence and abundance analysis.")
    parser.add_argument("--ref", nargs="+", default=["UHGG"], metavar="REF")
    parser.add_argument("--threshold", type=float, default=10.0,
                        metavar="PCT",
                        help="Breadth detection threshold %% (default 10)")
    parser.add_argument("--hg-min", type=float, default=0.10, dest="hg_min",
                        metavar="FRAC",
                        help="Min HG prevalence for selection (default 0.10)")
    parser.add_argument("--ind-max", type=float, default=0.05, dest="ind_max",
                        metavar="FRAC",
                        help="Max Industrial prevalence (default 0.05)")
    parser.add_argument("--top", type=int, default=40, metavar="N",
                        help="SGBs shown in detail plot (default 40)")
    args = parser.parse_args(argv)

    for ref in args.ref:
        run_analysis(ref,
                     threshold=args.threshold,
                     hg_prev_min=args.hg_min,
                     ind_prev_max=args.ind_max,
                     top_n=args.top)


if __name__ == "__main__":
    main()
