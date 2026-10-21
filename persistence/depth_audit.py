"""
Depth audit — the FIRST analysis run after data loading.

Checks for zero-depth samples (there should be none) and plots
depth distributions broken down by:
  1. Lifestyle (H-G, Transitional, Industrial)
  2. Health group (Non-Industrial, Industrial Healthy, Industrial Unhealthy)

Any zero-depth samples are flagged and removed from the data object
so that downstream analyses never see them.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plot_utils import (
    apply_oi_style,
    page_figsize,
    plain_axes,
    square_figsize,
    current_fs,
    panel_grid,
    savefig_dual,
    share_axis_labels,
)
apply_oi_style()

from . import config
from . import figure_source

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.dpi": 140,
    "axes.grid": True,
    "font.size": 12,
})


def run_depth_audit(data, stats, outdir_override=None, tag_prefix=""):
    """
    Run immediately after load_reference() — and again after reinstatement.

    Parameters
    ----------
    outdir_override : str | None
        If given, write outputs here instead of the default
        results/<ref>/depth_audit/.  Use to separate pre- vs
        post-reinstatement runs (e.g. "depth_audit_post_reinstatement").
    tag_prefix : str
        Prepended to every stats-report tag so post-reinstatement stats
        don't clobber pre-reinstatement ones.  Use e.g. "post_".

    1. Count zero-depth samples per lifestyle/health group.
    2. Plot depth distributions (before any matching).
    3. Return a cleaned ReferenceData with zero-depth samples removed,
       or the original if none found.
    """
    ref = data.ref_name
    meta = data.meta
    total_bp = data.total_bp

    if outdir_override:
        outdir = os.path.join("results", ref, outdir_override)
    else:
        outdir = os.path.join("results", ref, "depth_audit")
    os.makedirs(outdir, exist_ok=True)

    label = "Post-Reinstatement Depth Audit" if tag_prefix else "Depth Audit"
    print(f"\n{'='*60}")
    print(f"  {label} [{ref}]")
    print(f"{'='*60}")

    n_total = len(meta)

    # ── 1. Zero-depth check ──
    zero_mask = total_bp <= 0
    n_zero = int(zero_mask.sum())

    print(f"\n  Total samples: {n_total:,}")
    print(f"  Samples with total_bp <= 0: {n_zero}")

    if n_zero > 0:
        zero_ids = meta.index[zero_mask]
        # Breakdown by lifestyle
        zero_by_lf = meta.loc[zero_ids, "Lifestyle"].value_counts()
        print(f"\n  *** WARNING: {n_zero} zero-depth samples found! ***")
        print(f"  Breakdown by lifestyle:")
        for lf, cnt in zero_by_lf.items():
            print(f"    {lf}: {cnt}")

        # Breakdown by health
        zero_by_health = _health_group(meta.loc[zero_ids]).value_counts()
        print(f"  Breakdown by health group:")
        for hg, cnt in zero_by_health.items():
            print(f"    {hg}: {cnt}")

        # Save zero-depth sample list
        zero_df = pd.DataFrame({
            "sample_id": zero_ids,
            "Lifestyle": meta.loc[zero_ids, "Lifestyle"].values,
            "Health_group": _health_group(meta.loc[zero_ids]).values,
            "total_bp": total_bp.loc[zero_ids].values,
        })
        zero_df.to_csv(os.path.join(outdir, "zero_depth_samples.csv"),
                       index=False)
        print(f"  Saved: {os.path.join(outdir, 'zero_depth_samples.csv')}")

        stats.add(f"{tag_prefix}zero_depth_samples", "qc", "main",
                  "Samples with zero total mapped bp",
                  f"{n_zero} of {n_total} ({n_zero/n_total*100:.1f}%)")
    else:
        print(f"  OK: no zero-depth samples")
        stats.add(f"{tag_prefix}zero_depth_samples", "qc", "main",
                  "Samples with zero total mapped bp",
                  "0 (clean)")

    # ── 2. Depth distributions by lifestyle ──
    valid_mask = total_bp > 0
    valid_meta = meta[valid_mask]
    valid_bp = total_bp[valid_mask]
    log_bp = np.log10(valid_bp)

    print(f"\n  Valid samples (depth > 0): {len(valid_meta):,}")

    _plot_depth_by_lifestyle(valid_meta, log_bp, outdir, stats)

    # ── 3. Depth distributions by health group ──
    _plot_depth_by_health(valid_meta, log_bp, outdir, stats)

    # ── 4. Summary table ──
    _print_depth_summary(valid_meta, valid_bp, log_bp, stats, outdir, tag_prefix=tag_prefix)

    # ── 5. Remove zero-depth samples if any ──
    if n_zero > 0:
        print(f"\n  Removing {n_zero} zero-depth samples from data...")
        data = _remove_samples(data, zero_mask)
        print(f"  Data now has {len(data.meta):,} samples")

    print(f"\n  Depth audit complete [{ref}]")
    return data


def _health_group(meta_subset):
    """Classify samples into health groups for the audit."""
    groups = pd.Series("Unknown", index=meta_subset.index)

    is_ind = meta_subset["Lifestyle"] == "Industrial"
    is_ni = meta_subset["Lifestyle"].isin(["H-G", "Transitional"])

    groups[is_ni] = "Non-Industrial"

    if "Unhealthy" in meta_subset.columns:
        groups[is_ind & (meta_subset["Unhealthy"] == "Healthy")] = "Ind-Healthy"
        groups[is_ind & (meta_subset["Unhealthy"] == "Unhealthy")] = "Ind-Unhealthy"
        # Industrial with no health label
        groups[is_ind & ~meta_subset["Unhealthy"].isin(["Healthy", "Unhealthy"])] = "Ind-Unknown"
    else:
        groups[is_ind] = "Industrial"

    return groups


LIFESTYLE_PANELS = ["H-G", "Transitional", "Industrial"]


def _depth_by_lifestyle_table(meta, log_bp):
    """The source rows behind Figure A4: one per sample, per lifestyle panel.

    Split out from the drawing so the figure has a single input it can be
    rebuilt from. Everything the panel shows -- the histogram, the median rule,
    the sample count -- is derived from these columns and nothing else.
    """
    out = {}
    for lf in LIFESTYLE_PANELS:
        vals = log_bp[meta["Lifestyle"] == lf].dropna()
        out[config.display_text(lf)] = pd.DataFrame({
            "sample_id": vals.index.astype(str),
            "lifestyle": config.display_text(lf),
            "log10_total_mapped_bp": vals.values,
            "total_mapped_bp": 10.0 ** vals.values,
        })
    return out


def plot_depth_by_lifestyle(tables, outdir, stem="depth_by_lifestyle"):
    """Draw Figure A4 from its tables, and from nothing else.

    A pure function of `tables`: the pipeline hands it freshly computed ones,
    a replot hands it the same tables read back from disk, and the two are the
    same code path rather than two implementations that agree when tested.
    The annotations are computed here, from the same columns that are plotted,
    so there is no second copy of the arithmetic to drift.
    """
    names = list(tables)
    _MG = dict(top=0.82)
    fig, axes = plt.subplots(1, len(names), sharey=True,
                             figsize=square_figsize(n_cols=len(names),
                                                    margins=_MG))
    axes = np.atleast_1d(axes)

    allv = np.concatenate([t["log10_total_mapped_bp"].values
                           for t in tables.values() if len(t)] or [np.zeros(1)])
    lo, hi = np.quantile(allv, 0.005), np.quantile(allv, 0.995)
    bins = np.linspace(lo, hi, 50)

    for ax, name in zip(axes, names):
        df = tables[name]
        colour = config.LIFESTYLE_COLORS.get(
            config.canonical_lifestyle(name)
            if hasattr(config, "canonical_lifestyle") else name, "gray")
        if len(df):
            v = df["log10_total_mapped_bp"].values
            ax.hist(v, bins=bins, color=colour, alpha=0.7,
                    edgecolor="white", linewidth=0.3)
            med_log = float(np.median(v))
            ax.axvline(med_log, color="black", ls="--", lw=1.5)
            # Derived here from the same column that was just plotted.
            n = len(df)
            med_bp = float(np.median(df["total_mapped_bp"].values))
            ax.text(0.04, 0.95, f"n = {n:,}\nmedian = {med_bp:.1e} bp",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=current_fs() - 3)
        ax.set_title(name, fontweight="bold", color=colour)
        # Square panels whatever the margins end up as: the publication pass
        # drops the suptitle and hands its band back to the axes, which
        # stretched these taller than wide. The tight save crops the rest.
        ax.set_box_aspect(1)

    fig._margin_overrides = _MG
    fig._plain_axes_kw = dict(x_ticks=4, y_ticks=4)
    share_axis_labels(fig, "log$_{10}$(Total mapped bp)", "Count")
    fig.suptitle("Sequencing depth by lifestyle (before matching)",
                 fontsize=current_fs() + 1, fontweight="bold")
    fpath = os.path.join(outdir, f"{stem}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")
    return fpath


def _plot_depth_by_lifestyle(meta, log_bp, outdir, stats):
    """Compute the figure's data object, publish it, then draw from it."""
    tables = _depth_by_lifestyle_table(meta, log_bp)
    figure_source.emit(outdir, "S4", tables, stem="depth_by_lifestyle")
    return plot_depth_by_lifestyle(tables, outdir)


def _depth_by_health_table(meta, log_bp):
    """Source rows behind Figure S23: one per sample, per health-group panel."""
    health = _health_group(meta)
    order = ["Non-Industrial", "Ind-Healthy", "Ind-Unhealthy"]
    out = {}
    for grp in order:
        if not (health == grp).any():
            continue
        vals = log_bp[health == grp].dropna()
        out[grp] = pd.DataFrame({
            "sample_id": vals.index.astype(str),
            "health_group": grp,
            "log10_total_mapped_bp": vals.values,
            "total_mapped_bp": 10.0 ** vals.values,
        })
    return out


def plot_depth_by_health(tables, outdir, stem="depth_by_health_group"):
    """Draw Figure S23 from its tables and nothing else."""
    group_colors = {
        "Non-Industrial": config.OI["orange"],
        "Ind-Healthy":    config.OI["green"],
        "Ind-Unhealthy":  config.OI["blue"],
    }
    names = list(tables)
    if not names:
        return None
    _MG = dict(top=0.82)
    n_rows, n_cols = panel_grid(len(names))
    fig, axes = plt.subplots(n_rows, n_cols, sharey=True, squeeze=False,
                             figsize=square_figsize(n_cols=n_cols,
                                                    n_rows=n_rows,
                                                    margins=_MG))
    axes = axes.ravel()
    for ax in axes[len(names):]:
        ax.set_visible(False)

    allv = np.concatenate([t["log10_total_mapped_bp"].values
                           for t in tables.values() if len(t)] or [np.zeros(1)])
    bins = np.linspace(np.quantile(allv, 0.005), np.quantile(allv, 0.995), 50)

    for ax, name in zip(axes, names):
        df = tables[name]
        colour = group_colors.get(name, "gray")
        if len(df):
            v = df["log10_total_mapped_bp"].values
            ax.hist(v, bins=bins, color=colour, alpha=0.7,
                    edgecolor="white", linewidth=0.3)
            ax.axvline(float(np.median(v)), color="black", ls="--", lw=1.5)
            n = len(df)
            med_bp = float(np.median(df["total_mapped_bp"].values))
            ax.text(0.04, 0.95, f"n = {n:,}\nmedian = {med_bp:.1e} bp",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=current_fs() - 3)
        ax.set_title(name, fontweight="bold", color=colour)
        # Square panels, as in the lifestyle version (plot_depth_by_lifestyle).
        ax.set_box_aspect(1)

    fig._margin_overrides = _MG
    fig._plain_axes_kw = dict(x_ticks=4, y_ticks=4)
    share_axis_labels(fig, "log$_{10}$(Total mapped bp)", "Count")
    fig.suptitle("Sequencing depth by health group (before matching)",
                 fontsize=current_fs() + 1, fontweight="bold")
    fpath = os.path.join(outdir, f"{stem}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")
    return fpath


def _plot_depth_by_health(meta, log_bp, outdir, stats):
    """Compute the figure's data object, publish it, then draw from it."""
    tables = _depth_by_health_table(meta, log_bp)
    if not tables:
        return None
    figure_source.emit(outdir, "S23", tables, stem="depth_by_health_group")
    return plot_depth_by_health(tables, outdir)

def _print_depth_summary(meta, bp, log_bp, stats, outdir, tag_prefix=""):
    """Print and save per-group depth summary table."""
    health = _health_group(meta)

    rows = []
    print(f"\n  {'Group':<20s}  {'n':>7s}  {'median_bp':>12s}  "
          f"{'log10_med':>9s}  {'log10_IQR':>15s}")
    print(f"  {'-'*70}")

    # By lifestyle
    for lf in ["H-G", "Transitional", "Industrial"]:
        mask = meta["Lifestyle"] == lf
        _add_summary_row(rows, f"Lifestyle: {lf}", bp[mask], log_bp[mask],
                         stats, tag_prefix=tag_prefix)

    # By health group
    for grp in ["Non-Industrial", "Ind-Healthy", "Ind-Unhealthy", "Ind-Unknown"]:
        mask = health == grp
        if mask.any():
            _add_summary_row(rows, f"Health: {grp}", bp[mask], log_bp[mask],
                             stats, tag_prefix=tag_prefix)

    # Overall
    _add_summary_row(rows, "ALL", bp, log_bp, stats, tag_prefix=tag_prefix)

    pd.DataFrame(rows).to_csv(
        os.path.join(outdir, "depth_summary.csv"), index=False)
    print(f"\n  Saved: {os.path.join(outdir, 'depth_summary.csv')}")


def _add_summary_row(rows, label, bp_vals, log_vals, stats, tag_prefix=""):
    """Add one row to summary table and print it."""
    bp_vals = bp_vals.dropna()
    log_vals = log_vals.dropna()
    n = len(bp_vals)
    if n == 0:
        return

    med_bp = np.median(bp_vals)
    med_log = np.median(log_vals)
    q25, q75 = np.percentile(log_vals, [25, 75])

    print(f"  {label:<20s}  {n:>7,}  {med_bp:>12.1e}  "
          f"{med_log:>9.2f}  [{q25:.2f} - {q75:.2f}]")

    rows.append({
        "group": label,
        "n": n,
        "median_bp": med_bp,
        "mean_bp": np.mean(bp_vals),
        "log10_median": med_log,
        "log10_q25": q25,
        "log10_q75": q75,
        "log10_min": np.min(log_vals),
        "log10_max": np.max(log_vals),
    })

    tag = label.replace(" ", "_").replace(":", "")
    stats.add(f"depth_audit_{tag_prefix}{tag}", "qc", "main",
              f"Depth audit: {label}",
              f"n={n:,}, median={med_bp:.1e} bp, "
              f"log10 median={med_log:.2f}")


def _remove_samples(data, remove_mask):
    """
    Return a new ReferenceData with flagged samples removed.

    Works by reconstructing the namedtuple with filtered DataFrames.
    """
    keep_ids = data.meta.index[~remove_mask]

    from .loading import ReferenceData
    # _replace, not a field-by-field rebuild: the rebuild omitted
    # total_seq_bp, so this raised TypeError. It is per-sample and must be
    # filtered with the rest.
    return data._replace(
        breadth=data.breadth.loc[keep_ids],
        depth=data.depth.loc[keep_ids],
        abundance=data.abundance.loc[keep_ids],
        meta=data.meta.loc[keep_ids],
        total_bp=data.total_bp.loc[keep_ids],
        total_seq_bp=(data.total_seq_bp.loc[keep_ids]
                      if data.total_seq_bp is not None else None),
        cov_matrix=data.cov_matrix.loc[keep_ids],
        sec_map={k: v for k, v in (data.sec_map or {}).items()
                 if k in keep_ids},
    )


# ----------------------------------------------------------------
#  Post-matching depth audit
# ----------------------------------------------------------------

def audit_matched_depths(data, dm, stats):
    """
    Run immediately after depth matching.

    For every matched set, verify and plot:
      1. Overlaid depth distributions (matched group A vs group B)
      2. Per-pair fractional difference distribution
      3. Scatter bp_a vs bp_b
      4. Richness vs sequencing depth (coloured by group)
    """
    ref = data.ref_name
    total_bp = data.total_bp
    outdir = os.path.join("results", ref, "depth_audit")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n  {'='*56}")
    print(f"  Matched-set depth audit [{ref}]")
    print(f"  {'='*56}")

    # Build presence once for richness computation
    from .detection import DetectionSpec, build_presence
    spec = DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP)

    # ── Pairwise matches ──
    pairwise = [
        ("Ind_vs_Trans",
         dm.ind_vs_trans_ind, dm.ind_vs_trans_trans,
         "Industrial", "Transitional",
         config.LIFESTYLE_COLORS.get("Industrial", config.OI["vermillion"]),
         config.LIFESTYLE_COLORS.get("Transitional", config.OI["yellow"])),
        ("Ind_vs_NI",
         dm.ind_vs_ni_ind, dm.ind_vs_ni_ni,
         "Industrial", "Non-Industrial",
         config.LIFESTYLE_COLORS.get("Industrial", config.OI["vermillion"]),
         config.OI["orange"]),
    ]

    for tag, ids_a, ids_b, lab_a, lab_b, col_a, col_b in pairwise:
        if not ids_a or not ids_b:
            print(f"\n    {tag}: no matched pairs -- skipping")
            continue

        bp_a = total_bp.loc[ids_a].values.astype(float)
        bp_b = total_bp.loc[ids_b].values.astype(float)

        _audit_pair(bp_a, bp_b, lab_a, lab_b, col_a, col_b,
                    tag, outdir, stats)

        # Richness vs depth
        all_ids = list(ids_a) + list(ids_b)
        presence = build_presence(
            data.breadth.loc[all_ids], data.depth.loc[all_ids],
            data.lengths, spec=spec)
        richness = presence.sum(axis=1)
        _plot_richness_vs_depth(
            total_bp.loc[ids_a], total_bp.loc[ids_b],
            richness.loc[ids_a], richness.loc[ids_b],
            lab_a, lab_b, col_a, col_b,
            tag, spec, outdir)

    # ── Health triplets ──
    if dm.triplet_ni and dm.triplet_ind_h and dm.triplet_ind_u:
        bp_ni = total_bp.loc[dm.triplet_ni].values.astype(float)
        bp_h = total_bp.loc[dm.triplet_ind_h].values.astype(float)
        bp_u = total_bp.loc[dm.triplet_ind_u].values.astype(float)

        _audit_triplet(bp_ni, bp_h, bp_u, outdir, stats)

        # Richness vs depth for triplets
        all_trip = list(dm.triplet_ni) + list(dm.triplet_ind_h) + list(dm.triplet_ind_u)
        presence_t = build_presence(
            data.breadth.loc[all_trip], data.depth.loc[all_trip],
            data.lengths, spec=spec)
        richness_t = presence_t.sum(axis=1)
        _plot_richness_vs_depth_triplet(
            total_bp, richness_t,
            dm.triplet_ni, dm.triplet_ind_h, dm.triplet_ind_u,
            spec, outdir)
    else:
        print(f"\n    Health triplets: none -- skipping")

    print(f"\n  Matched-set depth audit complete [{ref}]")


def _audit_pair(bp_a, bp_b, lab_a, lab_b, col_a, col_b,
                tag, outdir, stats):
    """
    Audit one pairwise matched set.

    Three-panel figure:
      Left:   overlaid depth histograms
      Center: per-pair fractional difference histogram
      Right:  scatter of bp_a vs bp_b (should hug the diagonal)
    """
    n = len(bp_a)
    log_a = np.log10(bp_a)
    log_b = np.log10(bp_b)
    frac_diffs = np.abs(bp_a - bp_b) / np.maximum(bp_a, bp_b)

    # Stats
    med_frac = np.median(frac_diffs)
    max_frac = np.max(frac_diffs)
    q99_frac = np.percentile(frac_diffs, 99)
    pct_lt1 = np.mean(frac_diffs < 0.01) * 100
    pct_lt3 = np.mean(frac_diffs < 0.03) * 100

    print(f"\n    {tag}: {n:,} pairs")
    print(f"      {lab_a}: median={np.median(bp_a):.2e}, "
          f"log10 median={np.median(log_a):.2f}")
    print(f"      {lab_b}: median={np.median(bp_b):.2e}, "
          f"log10 median={np.median(log_b):.2f}")
    print(f"      Frac diff: median={med_frac:.5f}, "
          f"max={max_frac:.5f}, 99th={q99_frac:.5f}")
    print(f"      <1%: {pct_lt1:.1f}%,  <3%: {pct_lt3:.1f}%")

    stats.add(f"matched_depth_audit_{tag}", "qc", "main",
              f"Matched depth audit: {tag}",
              f"{n:,} pairs, median frac_diff={med_frac:.5f}, "
              f"max={max_frac:.5f}, <1%={pct_lt1:.1f}%, <3%={pct_lt3:.1f}%")

    # ── Figure ──
    fig, axes = plt.subplots(1, 3, figsize=page_figsize(n_cols=3))

    # Panel 1: overlaid histograms
    ax = axes[0]
    all_log = np.concatenate([log_a, log_b])
    bins = np.linspace(np.percentile(all_log, 0.5),
                       np.percentile(all_log, 99.5), 50)
    ax.hist(log_a, bins=bins, alpha=0.55, color=col_a, density=True,
            label=f"{lab_a} (n={n:,})")
    ax.hist(log_b, bins=bins, alpha=0.55, color=col_b, density=True,
            label=f"{lab_b} (n={n:,})")
    ax.set_xlabel("log$_{10}$(Total mapped bp)")
    ax.set_ylabel("Density")
    ax.set_title("Depth distributions (matched)")
    ax.legend(fontsize=9)

    # Panel 2: per-pair frac diff
    ax = axes[1]
    ax.hist(frac_diffs * 100, bins=50, color="#555555", alpha=0.7,
            edgecolor="white", linewidth=0.3)
    ax.axvline(3.0, color="red", ls="--", lw=1.5, label="3% tolerance")
    ax.set_xlabel("Per-pair fractional difference (%)")
    ax.set_ylabel("Count")
    ax.set_title("Matching quality")
    ax.legend(fontsize=9)
    ax.text(0.95, 0.92,
            f"median = {med_frac*100:.3f}%\n"
            f"max = {max_frac*100:.3f}%\n"
            f"<1%: {pct_lt1:.1f}%\n"
            f"<3%: {pct_lt3:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # Panel 3: scatter bp_a vs bp_b
    ax = axes[2]
    ax.scatter(log_a, log_b, s=2, alpha=0.15, color="#333333",
               edgecolors="none")
    lo = min(log_a.min(), log_b.min()) - 0.1
    hi = max(log_a.max(), log_b.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
    ax.set_xlabel(f"log$_{{10}}$(bp) {lab_a}")
    ax.set_ylabel(f"log$_{{10}}$(bp) {lab_b}")
    ax.set_title("Pair-wise depth comparison")
    ax.set_aspect("equal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=9)

    fig.suptitle(f"Matched-set depth audit: {lab_a} vs {lab_b}\n"
                 f"({n:,} pairs, frac_tol = 3%)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(outdir, f"matched_depth_audit_{tag}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {fpath}")


def _audit_triplet(bp_ni, bp_h, bp_u, outdir, stats):
    """
    Audit health triplets.

    Three-panel figure:
      Left:   overlaid depth histograms (3 groups)
      Center: per-triplet max fractional difference
      Right:  pairwise scatter (IndH vs IndU, NI vs IndH)
    """
    n = len(bp_ni)
    log_ni = np.log10(bp_ni)
    log_h = np.log10(bp_h)
    log_u = np.log10(bp_u)

    # Per-triplet max frac diff
    max_per_triplet = np.array([
        max(abs(a - b) / max(a, b),
            abs(a - c) / max(a, c),
            abs(b - c) / max(b, c))
        for a, b, c in zip(bp_ni, bp_h, bp_u)
    ])
    med_frac = np.median(max_per_triplet)
    max_frac = np.max(max_per_triplet)
    pct_lt1 = np.mean(max_per_triplet < 0.01) * 100

    print(f"\n    Health triplets: {n:,} triplets")
    print(f"      NI:          median={np.median(bp_ni):.2e}, "
          f"log10={np.median(log_ni):.2f}")
    print(f"      Ind-Healthy: median={np.median(bp_h):.2e}, "
          f"log10={np.median(log_h):.2f}")
    print(f"      Ind-Unhealthy: median={np.median(bp_u):.2e}, "
          f"log10={np.median(log_u):.2f}")
    print(f"      Max frac diff per triplet: "
          f"median={med_frac:.5f}, max={max_frac:.5f}, "
          f"<1%: {pct_lt1:.1f}%")

    stats.add("matched_depth_audit_triplets", "qc", "main",
              "Matched depth audit: Health triplets",
              f"{n:,} triplets, median max_frac={med_frac:.5f}, "
              f"max={max_frac:.5f}, <1%={pct_lt1:.1f}%")

    # ── Figure ──
    fig, axes = plt.subplots(1, 3, figsize=page_figsize(n_cols=3))

    # Panel 1: overlaid histograms
    ax = axes[0]
    all_log = np.concatenate([log_ni, log_h, log_u])
    bins = np.linspace(np.percentile(all_log, 0.5),
                       np.percentile(all_log, 99.5), 50)
    ax.hist(log_ni, bins=bins, alpha=0.45, color=config.OI["orange"], density=True,
            label=f"Non-Industrial (n={n:,})")
    ax.hist(log_h, bins=bins, alpha=0.45, color=config.OI["green"], density=True,
            label=f"Ind-Healthy (n={n:,})")
    ax.hist(log_u, bins=bins, alpha=0.45, color=config.OI["blue"], density=True,
            label=f"Ind-Unhealthy (n={n:,})")
    ax.set_xlabel("log$_{10}$(Total mapped bp)")
    ax.set_ylabel("Density")
    ax.set_title("Depth distributions (matched triplets)")
    ax.legend(fontsize=8)

    # Panel 2: per-triplet max frac diff
    ax = axes[1]
    ax.hist(max_per_triplet * 100, bins=50, color="#555555", alpha=0.7,
            edgecolor="white", linewidth=0.3)
    ax.axvline(1.0, color="red", ls="--", lw=1.5, label="1% tolerance")
    ax.set_xlabel("Max per-triplet fractional difference (%)")
    ax.set_ylabel("Count")
    ax.set_title("Triplet matching quality")
    ax.legend(fontsize=9)
    ax.text(0.95, 0.92,
            f"median = {med_frac*100:.4f}%\n"
            f"max = {max_frac*100:.4f}%\n"
            f"<1%: {pct_lt1:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9))

    # Panel 3: two scatter subplots
    ax = axes[2]
    lo = min(log_ni.min(), log_h.min(), log_u.min()) - 0.1
    hi = max(log_ni.max(), log_h.max(), log_u.max()) + 0.1
    ax.scatter(log_h, log_u, s=2, alpha=0.15, color="#9467bd",
               edgecolors="none", label="IndH vs IndU")
    ax.scatter(log_ni, log_h, s=2, alpha=0.15, color=config.OI["orange"],
               edgecolors="none", label="NI vs IndH")
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("log$_{10}$(bp) group 1")
    ax.set_ylabel("log$_{10}$(bp) group 2")
    ax.set_title("Pair-wise within triplets")
    ax.set_aspect("equal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=8, markerscale=5)

    fig.suptitle(f"Matched-set depth audit: Health triplets\n"
                 f"({n:,} triplets, frac_tol = 1%)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(outdir, "matched_depth_audit_triplets.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {fpath}")


# ----------------------------------------------------------------
#  Richness vs depth plots
# ----------------------------------------------------------------

def _plot_richness_vs_depth(bp_a, bp_b, rich_a, rich_b,
                            lab_a, lab_b, col_a, col_b,
                            tag, spec, outdir):
    """
    Two-panel figure for a pairwise match:
      Left:  richness vs log10(depth) scatter, both groups overlaid
      Right: LOESS/binned trend lines to see whether richness
             is truly flat across the matched depth range
    """
    log_a = np.log10(bp_a.values.astype(float))
    log_b = np.log10(bp_b.values.astype(float))
    r_a = rich_a.values.astype(float)
    r_b = rich_b.values.astype(float)
    n_a, n_b = len(r_a), len(r_b)

    fig, axes = plt.subplots(1, 2, figsize=page_figsize(n_cols=2))

    # ── Left: scatter ──
    ax = axes[0]
    ax.scatter(log_a, r_a, s=3, alpha=0.12, color=col_a,
               edgecolors="none", label=f"{lab_a} (n={n_a:,})")
    ax.scatter(log_b, r_b, s=3, alpha=0.12, color=col_b,
               edgecolors="none", label=f"{lab_b} (n={n_b:,})")
    ax.set_xlabel("log$_{10}$(Total mapped bp)")
    ax.set_ylabel(f"Richness (SGBs, {spec.short_label})")
    ax.set_title("Richness vs sequencing depth (matched samples)")
    ax.legend(fontsize=9, markerscale=5)
    ax.grid(alpha=0.3)

    # ── Right: binned medians with IQR ribbons ──
    ax = axes[1]
    all_log = np.concatenate([log_a, log_b])
    n_bins = 20
    bin_edges = np.linspace(np.percentile(all_log, 1),
                            np.percentile(all_log, 99), n_bins + 1)

    for log_vals, r_vals, label, color in [
        (log_a, r_a, lab_a, col_a),
        (log_b, r_b, lab_b, col_b),
    ]:
        bin_centers, medians, q25s, q75s = [], [], [], []
        for i in range(n_bins):
            mask = (log_vals >= bin_edges[i]) & (log_vals < bin_edges[i + 1])
            if mask.sum() < 5:
                continue
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            medians.append(np.median(r_vals[mask]))
            q25s.append(np.percentile(r_vals[mask], 25))
            q75s.append(np.percentile(r_vals[mask], 75))

        if bin_centers:
            bc = np.array(bin_centers)
            ax.plot(bc, medians, color=color, lw=2, label=label)
            ax.fill_between(bc, q25s, q75s, color=color, alpha=0.15)

    ax.set_xlabel("log$_{10}$(Total mapped bp)")
    ax.set_ylabel(f"Richness (SGBs, {spec.short_label})")
    ax.set_title("Binned median richness (IQR shaded)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Richness vs depth: {lab_a} vs {lab_b} (matched)\n"
                 f"{spec.description}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(outdir, f"richness_vs_depth_{tag}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {fpath}")


def _plot_richness_vs_depth_triplet(total_bp, richness,
                                     ids_ni, ids_h, ids_u,
                                     spec, outdir):
    """
    Richness vs depth for health triplet groups.
    Same layout as pairwise but with three groups.
    """
    groups = [
        (ids_ni, "Non-Industrial", config.OI["orange"]),
        (ids_h, "Ind-Healthy", config.OI["green"]),
        (ids_u, "Ind-Unhealthy", config.OI["blue"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=page_figsize(n_cols=2))

    # ── Left: scatter ──
    ax = axes[0]
    for ids, label, color in groups:
        log_d = np.log10(total_bp.loc[ids].values.astype(float))
        r = richness.loc[ids].values.astype(float)
        ax.scatter(log_d, r, s=3, alpha=0.12, color=color,
                   edgecolors="none", label=f"{label} (n={len(ids):,})")
    ax.set_xlabel("log$_{10}$(Total mapped bp)")
    ax.set_ylabel(f"Richness (SGBs, {spec.short_label})")
    ax.set_title("Richness vs sequencing depth (matched triplets)")
    ax.legend(fontsize=8, markerscale=5)
    ax.grid(alpha=0.3)

    # ── Right: binned medians ──
    ax = axes[1]
    all_log = np.log10(total_bp.loc[
        list(ids_ni) + list(ids_h) + list(ids_u)
    ].values.astype(float))
    n_bins = 20
    bin_edges = np.linspace(np.percentile(all_log, 1),
                            np.percentile(all_log, 99), n_bins + 1)

    for ids, label, color in groups:
        log_d = np.log10(total_bp.loc[ids].values.astype(float))
        r = richness.loc[ids].values.astype(float)

        bin_centers, medians, q25s, q75s = [], [], [], []
        for i in range(n_bins):
            mask = (log_d >= bin_edges[i]) & (log_d < bin_edges[i + 1])
            if mask.sum() < 5:
                continue
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            medians.append(np.median(r[mask]))
            q25s.append(np.percentile(r[mask], 25))
            q75s.append(np.percentile(r[mask], 75))

        if bin_centers:
            bc = np.array(bin_centers)
            ax.plot(bc, medians, color=color, lw=2, label=label)
            ax.fill_between(bc, q25s, q75s, color=color, alpha=0.15)

    ax.set_xlabel("log$_{10}$(Total mapped bp)")
    ax.set_ylabel(f"Richness (SGBs, {spec.short_label})")
    ax.set_title("Binned median richness (IQR shaded)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Richness vs depth: Health triplets (matched)\n"
                 f"{spec.description}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(outdir, "richness_vs_depth_triplets.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {fpath}")
