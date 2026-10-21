"""
Within-group variation & composition accumulation (Section 1D'-1F').

Supports the claim that individuals within each lifestyle group exhibit
substantial differences in both diversity and composition, such that
sampling multiple individuals is required to capture population diversity.

Analyses
--------
1D'. Within-group diversity variation
     - Richness violin/box by lifestyle (3-panel: H-G, Transitional, Industrial)
     - CV and IQR of richness within each group
     - All on depth-matched samples

1E'. Within-group compositional variation
     - Pairwise Jaccard (presence/absence) within each lifestyle
     - Pairwise Bray-Curtis (abundance) within each lifestyle
     - Overlaid density plots: within H-G, within Trans, within Ind
     - Optional between-group distances for context
     - Stats: median within-group distance, ANOSIM R

1F'. Composition accumulation
     - Union-SGB count vs number of samples pooled, by lifestyle
     - Bootstrap 95% CIs
     - Depth-matched samples

Integration
-----------
Called from run_section1() in section1_diversity.py, or standalone via
    python -m pipeline_v3 --only s1_within
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plot_utils import apply_oi_style, savefig_dual, current_fs
apply_oi_style()
FS = current_fs()   # base font size; changes per preset in savefig_dual
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu

from . import config
from .detection import DetectionSpec, build_presence

plt.rcParams.update({
    "figure.dpi": 140,
    "axes.grid": True,
    "font.size": 12,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
})

# Consistent lifestyle ordering & colors
LIFESTYLE_ORDER = ["H-G", "Transitional", "Industrial"]
NON_INDUSTRIAL_COLOR = config.OI["orange"]  # used when H-G and Trans are merged


# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════

def _lifestyle_split(data, ids):
    """Split sample IDs by Lifestyle, returning {lifestyle: [ids]}."""
    meta = data.meta
    groups = {}
    for lf in LIFESTYLE_ORDER:
        gids = [s for s in ids if s in meta.index and meta.loc[s, "Lifestyle"] == lf]
        if gids:
            groups[lf] = gids
    return groups


def _subsample_pairs(n_total, max_pairs=500_000, seed=42):
    """
    If n_total pairwise comparisons exceeds max_pairs, return a random
    subset of pair indices.  Otherwise return None (use all).
    """
    n_pairs = n_total * (n_total - 1) // 2
    if n_pairs <= max_pairs:
        return None
    rng = np.random.default_rng(seed)
    return rng.choice(n_pairs, size=max_pairs, replace=False)


# ════════════════════════════════════════════════════════════════════
#  1D'. WITHIN-GROUP DIVERSITY VARIATION
# ════════════════════════════════════════════════════════════════════

def compute_within_diversity(data, dm, stats, spec=None):
    """
    Violin/box plot of per-sample richness by lifestyle (depth-matched).
    Reports CV and IQR for each group.
    """
    ref = data.ref_name
    outdir = config.section_dir(ref, "s1", "main")

    if spec is None:
        spec = DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP)
    spec_tag = spec.short_label

    # Use depth-matched Ind vs NI
    ids_ind = dm.ind_vs_ni_ind
    ids_ni = dm.ind_vs_ni_ni
    if not ids_ind or not ids_ni:
        print("    [within-div] No depth-matched pairs - skipping")
        return

    all_ids = list(ids_ind) + list(ids_ni)

    # Build presence matrix
    presence = build_presence(
        data.breadth.loc[all_ids], data.depth.loc[all_ids],
        data.lengths, spec=spec)
    richness = presence.sum(axis=1)

    # Split NI by lifestyle
    groups = _lifestyle_split(data, ids_ni)
    groups["Industrial"] = ids_ind

    # Compute stats per lifestyle
    print(f"\n  1D'. Within-group diversity variation ({spec.description})")
    group_data = {}
    for lf in LIFESTYLE_ORDER:
        if lf not in groups:
            continue
        gids = groups[lf]
        vals = richness.reindex(gids).dropna().values
        if len(vals) < 2:
            continue
        med = np.median(vals)
        mean = np.mean(vals)
        sd = np.std(vals, ddof=1)
        cv = sd / mean * 100
        q25, q75 = np.percentile(vals, [25, 75])
        iqr = q75 - q25
        group_data[lf] = vals

        stats.add(
            f"within_richness_CV_{lf}_{spec_tag}", "s1", "main",
            f"Richness CV within {lf} (depth-matched, {spec.description})",
            f"n={len(vals)}, median={med:.0f}, CV={cv:.1f}%, IQR={iqr:.0f}",
            n1=len(vals))
        print(f"    {lf}: n={len(vals):,}, median={med:.0f}, "
              f"mean={mean:.0f}, CV={cv:.1f}%, IQR=[{q25:.0f}-{q75:.0f}]")

    if not group_data:
        return

    # ── Plot: violin + box ──
    _plot_richness_violins(group_data, spec, outdir, spec_tag)

    # ── Save CSV ──
    rows = []
    for lf, vals in group_data.items():
        for v in vals:
            rows.append({"Lifestyle": lf, "Richness": v})
    pd.DataFrame(rows).to_csv(
        os.path.join(outdir, f"within_group_richness_{spec_tag}.csv"),
        index=False)


def _plot_richness_violins(group_data, spec, outdir, spec_tag):
    """Violin + strip plot of richness by lifestyle."""
    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(42)

    labels = []
    positions = []
    for i, lf in enumerate(LIFESTYLE_ORDER):
        if lf not in group_data:
            continue
        labels.append(lf)
        positions.append(len(labels))

    data_list = [group_data[lf] for lf in LIFESTYLE_ORDER if lf in group_data]
    colors = [config.LIFESTYLE_COLORS.get(lf, "gray")
              for lf in LIFESTYLE_ORDER if lf in group_data]

    vparts = ax.violinplot(data_list, positions=range(1, len(data_list) + 1),
                           showmedians=False, showextrema=False)
    for body, c in zip(vparts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.35)

    bplot = ax.boxplot(data_list, positions=range(1, len(data_list) + 1),
                       patch_artist=True, widths=0.25, showfliers=False,
                       zorder=3)
    for patch, c in zip(bplot["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    for i, (vals, c) in enumerate(zip(data_list, colors), 1):
        n_show = min(len(vals), 2000)
        idx = rng.choice(len(vals), size=n_show, replace=False)
        jitter = rng.uniform(-0.1, 0.1, n_show)
        ax.scatter(np.full(n_show, i) + jitter, vals[idx],
                   c=c, alpha=0.06, s=3, edgecolors="none", zorder=2)

    tick_labels = [f"{lf}\n(n={len(group_data[lf]):,})"
                   for lf in LIFESTYLE_ORDER if lf in group_data]
    ax.set_xticks(range(1, len(data_list) + 1))
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Richness (SGBs detected)")
    ax.set_title(f"Within-group diversity variation\n"
                 f"(1:1 depth-matched, {spec.description})",
                 fontsize=FS+2, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fpath = os.path.join(outdir, f"within_group_richness_violin_{spec_tag}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")


# ════════════════════════════════════════════════════════════════════
#  1E'. WITHIN-GROUP COMPOSITIONAL VARIATION
# ════════════════════════════════════════════════════════════════════

def compute_within_beta(data, dm, stats, spec=None, max_per_group=1500):
    """
    Pairwise Jaccard & Bray-Curtis distances within each lifestyle.
    Produces overlaid density plots and reports median distances.

    Parameters
    ----------
    max_per_group : int
        Subsample to this many samples per group to keep distance
        computation tractable.  Set None for no limit.
    """
    ref = data.ref_name
    outdir = config.section_dir(ref, "s1", "main")

    if spec is None:
        spec = DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP)
    spec_tag = spec.short_label

    ids_ind = dm.ind_vs_ni_ind
    ids_ni = dm.ind_vs_ni_ni
    if not ids_ind or not ids_ni:
        print("    [within-beta] No depth-matched pairs - skipping")
        return

    all_ids = list(ids_ind) + list(ids_ni)

    # Presence matrix (for Jaccard)
    presence = build_presence(
        data.breadth.loc[all_ids], data.depth.loc[all_ids],
        data.lengths, spec=spec)

    # Abundance matrix (for Bray-Curtis)
    abund = data.abundance.loc[all_ids]

    # Split by lifestyle
    groups = _lifestyle_split(data, ids_ni)
    groups["Industrial"] = ids_ind

    rng = np.random.default_rng(42)

    print(f"\n  1E'. Within-group compositional variation ({spec.description})")

    results = {}  # {lifestyle: {"jaccard": array, "braycurtis": array}}

    for lf in LIFESTYLE_ORDER:
        if lf not in groups or len(groups[lf]) < 10:
            continue
        gids = groups[lf]

        # Subsample if needed
        if max_per_group and len(gids) > max_per_group:
            gids = list(rng.choice(gids, size=max_per_group, replace=False))
            print(f"    {lf}: subsampled to {max_per_group} "
                  f"(from {len(groups[lf]):,})")

        n = len(gids)
        n_pairs = n * (n - 1) // 2
        print(f"    {lf}: n={n:,}, {n_pairs:,} pairs")

        # Jaccard
        pres_g = presence.loc[gids].values.astype(float)
        jacc = pdist(pres_g, metric="jaccard")
        jacc = np.nan_to_num(jacc, nan=1.0)

        # Bray-Curtis
        abund_g = np.nan_to_num(abund.loc[gids].values, nan=0.0)
        bc = pdist(abund_g, metric="braycurtis")
        bc = np.nan_to_num(bc, nan=1.0)

        results[lf] = {"jaccard": jacc, "braycurtis": bc, "n": n}

        for metric, vals in [("Jaccard", jacc), ("Bray-Curtis", bc)]:
            med = np.median(vals)
            q25, q75 = np.percentile(vals, [25, 75])
            stats.add(
                f"within_beta_{metric}_{lf}_{spec_tag}", "s1", "main",
                f"Median within-group {metric} for {lf} "
                f"(depth-matched, {spec.description})",
                f"median={med:.3f}, IQR=[{q25:.3f}-{q75:.3f}]",
                n1=n, extra={"n_pairs": n_pairs})
            print(f"      {metric}: median={med:.3f}, "
                  f"IQR=[{q25:.3f}-{q75:.3f}]")

    if not results:
        return

    # ── Between-group distances (Industrial vs NI pooled) ──
    between = _compute_between_distances(
        presence, abund, ids_ind, ids_ni, max_per_group, rng)

    # ── Density plots ──
    _plot_within_beta_density(results, between, spec, outdir, spec_tag)

    # ── Boxplots with significance ──
    _plot_within_beta_boxplot(results, between, spec, outdir, spec_tag)

    # ── Save summary CSV ──
    rows = []
    raw_arrays = {}
    for lf, res in results.items():
        for metric in ["jaccard", "braycurtis"]:
            vals = res[metric]
            raw_arrays[f"{lf}_{metric}"] = vals.astype(np.float32, copy=False)
            rows.append({
                "Lifestyle": lf,
                "Metric": metric,
                "n_samples": res["n"],
                "n_pairs": len(vals),
                "median": np.median(vals),
                "mean": np.mean(vals),
                "q25": np.percentile(vals, 25),
                "q75": np.percentile(vals, 75),
            })
    if between:
        for metric in ["jaccard", "braycurtis"]:
            vals = between[metric]
            raw_arrays[f"Between (Ind vs NI)_{metric}"] = vals.astype(np.float32, copy=False)
            rows.append({
                "Lifestyle": "Between (Ind vs NI)",
                "Metric": metric,
                "n_samples": between["n"],
                "n_pairs": len(vals),
                "median": np.median(vals),
                "mean": np.mean(vals),
                "q25": np.percentile(vals, 25),
                "q75": np.percentile(vals, 75),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(outdir, f"within_group_beta_{spec_tag}.csv"),
        index=False)
    np.savez_compressed(
        os.path.join(outdir, f"within_group_beta_raw_{spec_tag}.npz"),
        **raw_arrays)


def _compute_between_distances(presence, abund, ids_ind, ids_ni,
                               max_per_group, rng):
    """Compute pairwise distances between Industrial and NI samples."""
    # Subsample both sides
    n_between = min(max_per_group or 1500, 1000)
    ind_sub = list(rng.choice(ids_ind,
                              size=min(n_between, len(ids_ind)),
                              replace=False))
    ni_sub = list(rng.choice(ids_ni,
                             size=min(n_between, len(ids_ni)),
                             replace=False))

    combined = ind_sub + ni_sub
    n_ind, n_ni = len(ind_sub), len(ni_sub)

    pres_c = presence.loc[combined].values.astype(float)
    abund_c = np.nan_to_num(abund.loc[combined].values, nan=0.0)

    jacc_sq = squareform(pdist(pres_c, metric="jaccard"))
    bc_sq = squareform(pdist(abund_c, metric="braycurtis"))

    # Extract between-group block
    jacc_between = jacc_sq[:n_ind, n_ind:].ravel()
    bc_between = bc_sq[:n_ind, n_ind:].ravel()

    return {
        "jaccard": np.nan_to_num(jacc_between, nan=1.0),
        "braycurtis": np.nan_to_num(bc_between, nan=1.0),
        "n": n_ind + n_ni,
    }


def _plot_within_beta_boxplot(results, between, spec, outdir, spec_tag):
    """
    Horizontal stacked boxplots of within-group and between-group pairwise
    distances, one panel per metric (Jaccard / Bray-Curtis).

    Layout
    ------
    Two side-by-side panels sharing the x-axis (distance 0–1).
    Each panel has one horizontal box per group, stacked vertically:
        H-G
        Transitional
        Industrial
        Between (Ind vs NI)   ← reference, shown in grey, dashed outline

    Statistics
    ----------
    For every pair of within-group distributions, and for each within-group
    vs between, a Mann-Whitney U test is run.  Results are shown as:
      * p < 0.05   ** p < 0.01   *** p < 0.001
    Brackets connect the two groups being compared, placed to the right of
    the boxes at increasing x offsets to avoid overlap.

    Saved: within_group_beta_boxplot_{spec_tag}.png
    """
    from scipy.stats import mannwhitneyu

    # All groups in display order (within groups first, then between)
    all_labels = [lf for lf in LIFESTYLE_ORDER if lf in results]
    if between:
        all_labels.append("Between (Ind vs NI)")
    n_groups = len(all_labels)
    if n_groups < 2:
        return

    def _star(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    def _get_vals(label, metric):
        if label == "Between (Ind vs NI)":
            return between[metric] if between else np.array([])
        return results[label][metric]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, max(4.5, n_groups * 0.85 + 2.5)),
        sharey=True)

    for ax, metric, title in zip(
        axes,
        ["jaccard", "braycurtis"],
        ["Jaccard (presence/absence)", "Bray-Curtis (abundance)"],
    ):
        # Draw boxes bottom-to-top (y=0 = bottom label)
        y_positions = np.arange(n_groups)
        box_data = []
        colors    = []
        for label in all_labels:
            vals = _get_vals(label, metric)
            box_data.append(vals[np.isfinite(vals)] if len(vals) else np.array([0.5]))
            if label == "Between (Ind vs NI)":
                colors.append("#888888")
            else:
                colors.append(config.LIFESTYLE_COLORS.get(label, "gray"))

        bp = ax.boxplot(
            box_data,
            positions=y_positions,
            vert=False,           # horizontal boxes
            widths=0.55,
            patch_artist=True,
            notch=False,
            showfliers=False,
            medianprops=dict(color="black", lw=2),
            whiskerprops=dict(lw=1.2),
            capprops=dict(lw=1.2),
            boxprops=dict(lw=1.2),
        )
        for patch, col, label in zip(bp["boxes"], colors, all_labels):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
            if label == "Between (Ind vs NI)":
                patch.set_linestyle("--")
                patch.set_linewidth(1.5)

        # Median markers
        for i, (vals, col) in enumerate(zip(box_data, colors)):
            if len(vals):
                ax.scatter(np.median(vals), i, color=col, s=40,
                           zorder=5, edgecolors="black", linewidths=0.6)

        # Y-axis labels with n
        y_labels = []
        for label in all_labels:
            vals = _get_vals(label, metric)
            n = results[label]["n"] if label in results else between.get("n", len(vals))
            y_labels.append(f"{config.LIFESTYLE_DISPLAY.get(label, label)}\n(n={n:,})")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=FS-1)

        ax.set_xlim(-0.02, 1.20)  # extra space right for significance brackets
        ax.set_xlabel("Pairwise distance", fontsize=FS)
        ax.set_title(title, fontsize=FS+1, fontweight="bold")
        ax.grid(axis="x", alpha=0.25)

        # ── Significance + effect size ────────────────────────────────────
        # Pairwise distances are NOT independent: n*(n-1)/2 distances share
        # samples, so effective df ≈ n_samples, not n_pairs.
        # Fix: subsample O(n_samples) distances for testing so the p-value
        # reflects practical rather than mathematical significance.
        # Also report rank-biserial r as effect size so readers can see
        # whether r=0.02*** means "real but tiny".
        #
        # n_samples per group ≈ sqrt(2 * n_pairs + 0.25) + 0.5
        def _n_samples_from_pairs(n_pairs):
            return int(round(0.5 + (0.25 + 2 * n_pairs) ** 0.5))

        def _rank_biserial(u, n1, n2):
            """Rank-biserial correlation: r = 1 - 2U/(n1*n2), range [-1, 1]."""
            return float(1.0 - 2.0 * u / max(n1 * n2, 1))

        sig_pairs = []
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                v1 = box_data[i]; v2 = box_data[j]
                if len(v1) < 5 or len(v2) < 5:
                    continue
                rng_mw = np.random.default_rng(42)
                # Subsample to ~n_samples distances per group (not n_pairs)
                n1_samp = max(20, min(_n_samples_from_pairs(len(v1)), 500))
                n2_samp = max(20, min(_n_samples_from_pairs(len(v2)), 500))
                s1 = v1[rng_mw.choice(len(v1), n1_samp, replace=False)]
                s2 = v2[rng_mw.choice(len(v2), n2_samp, replace=False)]
                u_stat, p = mannwhitneyu(s1, s2, alternative="two-sided")
                r = _rank_biserial(u_stat, n1_samp, n2_samp)
                sig_pairs.append((i, j, p, _star(p), r))

        # Draw significant pairs (p < 0.05), annotated with stars + |r|
        sig_pairs_draw = [(i, j, p, s, r)
                          for i, j, p, s, r in sig_pairs if s != "ns"]
        sig_pairs_draw.sort(key=lambda x: x[1] - x[0])

        x_bracket_start = 1.02
        bracket_step    = 0.07
        for bk, (i, j, p, star, r) in enumerate(sig_pairs_draw):
            x = x_bracket_start + bk * bracket_step
            y_lo, y_hi = y_positions[i], y_positions[j]
            ax.plot([x, x], [y_lo, y_hi],
                    color="black", lw=1.0, clip_on=False,
                    transform=ax.get_yaxis_transform())
            tick = 0.05
            for yy in (y_lo, y_hi):
                ax.plot([x - tick, x], [yy, yy], color="black", lw=1.0,
                        clip_on=False, transform=ax.get_yaxis_transform())
            label = f"{star}\nr={r:+.2f}"
            ax.text(x + 0.012, (y_lo + y_hi) / 2, label,
                    ha="left", va="center", fontsize=FS-2,
                    fontweight="bold", clip_on=False,
                    transform=ax.get_yaxis_transform())

    fig.suptitle(
        f"Within-group compositional variation — boxplots\n"
        f"(1:1 depth-matched, {spec.description}  |  "
        "whiskers = 5–95th pct  |  MWU on ~n_samples distances (not n_pairs)  |  "
        "r = rank-biserial effect size  |  * p<0.05  ** p<0.01  *** p<0.001)",
        fontsize=FS+2, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(outdir,
                         f"within_group_beta_boxplot_{spec_tag}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")


def _plot_within_beta_density(results, between, spec, outdir, spec_tag):
    """Overlaid density curves of within-group pairwise distances."""
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, metric, title in zip(
        axes,
        ["jaccard", "braycurtis"],
        ["Jaccard (presence/absence)", "Bray-Curtis (abundance)"],
    ):
        for lf in LIFESTYLE_ORDER:
            if lf not in results:
                continue
            vals = results[lf][metric]
            # Subsample for KDE if huge
            if len(vals) > 200_000:
                vals = np.random.default_rng(42).choice(
                    vals, size=200_000, replace=False)
            try:
                kde = gaussian_kde(vals, bw_method=0.04)
                x = np.linspace(0, 1, 500)
                color = config.LIFESTYLE_COLORS.get(lf, "gray")
                n = results[lf]["n"]
                ax.plot(x, kde(x), color=color, lw=2,
                        label=f"{config.LIFESTYLE_DISPLAY.get(lf, lf)} (n={n:,})")
                med = np.median(vals)
                ax.axvline(med, color=color, ls="--", lw=1, alpha=0.6)
            except Exception:
                pass

        # Between-group
        if between and metric in between:
            vals = between[metric]
            if len(vals) > 200_000:
                vals = np.random.default_rng(42).choice(
                    vals, size=200_000, replace=False)
            try:
                kde = gaussian_kde(vals, bw_method=0.04)
                x = np.linspace(0, 1, 500)
                ax.plot(x, kde(x), color="gray", lw=2, ls=":",
                        label="Between (Ind vs NI)")
                med = np.median(vals)
                ax.axvline(med, color="gray", ls=":", lw=1, alpha=0.6)
            except Exception:
                pass

        ax.set_xlabel("Pairwise distance")
        ax.set_ylabel("Density")
        ax.set_title(title, fontsize=FS+1)
        ax.legend(fontsize=FS-1)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Within-group compositional variation\n"
                 f"(1:1 depth-matched, {spec.description})",
                 fontsize=FS+2, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(outdir,
                         f"within_group_beta_density_{spec_tag}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")


# ════════════════════════════════════════════════════════════════════
#  1F'. COMPOSITION ACCUMULATION
# ════════════════════════════════════════════════════════════════════

def compute_composition_accumulation(data, dm, stats, spec=None,
                                     n_boot=100, seed=42):
    """
    Union-SGB accumulation curves by lifestyle.

    For increasing numbers of pooled depth-matched samples, count the
    total number of distinct SGBs detected in the union (absolute count).
    Bootstrap for 95% CIs.
    """
    ref = data.ref_name
    outdir = config.section_dir(ref, "s1", "main")

    if spec is None:
        spec = DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP)
    spec_tag = spec.short_label

    ids_ind = dm.ind_vs_ni_ind
    ids_ni = dm.ind_vs_ni_ni
    if not ids_ind or not ids_ni:
        print("    [comp-accum] No depth-matched pairs - skipping")
        return

    all_ids = list(ids_ind) + list(ids_ni)

    # Build presence matrix (boolean)
    presence = build_presence(
        data.breadth.loc[all_ids], data.depth.loc[all_ids],
        data.lengths, spec=spec)

    # Split by lifestyle
    groups = _lifestyle_split(data, ids_ni)
    groups["Industrial"] = ids_ind

    print(f"\n  1F'. Composition accumulation ({spec.description})")

    all_curves = {}

    for lf in LIFESTYLE_ORDER:
        if lf not in groups or len(groups[lf]) < 5:
            continue
        gids = groups[lf]
        pres_arr = presence.loc[gids].values  # bool (n_samples x n_sgbs)
        n_samples = len(gids)

        # Choose n-points: log-spaced from 1 to n_samples
        n_points = np.unique(
            np.geomspace(1, n_samples, num=60).astype(int))
        n_points = np.clip(n_points, 1, n_samples)
        n_points = np.unique(np.concatenate([[1], n_points, [n_samples]]))

        rng = np.random.default_rng(seed)
        curve_rows = []

        for n in n_points:
            union_counts = []
            for _ in range(n_boot):
                idx = rng.choice(n_samples, size=n, replace=False)
                # Union: any SGB detected in at least one of the n samples
                union_count = pres_arr[idx].any(axis=0).sum()
                union_counts.append(union_count)
            curve_rows.append({
                "n_samples": int(n),
                "mean": np.mean(union_counts),
                "median": np.median(union_counts),
                "lo": np.percentile(union_counts, 2.5),
                "hi": np.percentile(union_counts, 97.5),
            })

        curve_df = pd.DataFrame(curve_rows)
        all_curves[lf] = curve_df

        # Stat: total SGBs at max N
        total_sgbs = int(pres_arr.any(axis=0).sum())
        stats.add(
            f"comp_accum_total_{lf}_{spec_tag}", "s1", "main",
            f"Total SGBs detected across all {lf} samples "
            f"(depth-matched, {spec.description})",
            f"{total_sgbs:,} SGBs from {n_samples:,} samples",
            n1=n_samples)
        print(f"    {lf}: {total_sgbs:,} SGBs from {n_samples:,} samples")

        # Stat: SGBs at n=100 (a useful reference point)
        if n_samples >= 100:
            row_100 = curve_df.loc[(curve_df["n_samples"] - 100).abs().idxmin()]
            stats.add(
                f"comp_accum_n100_{lf}_{spec_tag}", "s1", "main",
                f"SGBs at n=100 for {lf} "
                f"(depth-matched, {spec.description})",
                f"{row_100['mean']:.0f} [{row_100['lo']:.0f}-"
                f"{row_100['hi']:.0f}] SGBs",
                n1=100)

    if not all_curves:
        return

    # ── Plot ──
    _plot_composition_accumulation(all_curves, spec, outdir, spec_tag)

    # ── Save CSV ──
    frames = []
    for lf, df in all_curves.items():
        df = df.copy()
        df.insert(0, "Lifestyle", lf)
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(
        os.path.join(outdir, f"composition_accumulation_{spec_tag}.csv"),
        index=False)


def _plot_composition_accumulation(all_curves, spec, outdir, spec_tag):
    """Overlaid composition accumulation curves by lifestyle."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for lf in LIFESTYLE_ORDER:
        if lf not in all_curves:
            continue
        df = all_curves[lf]
        color = config.LIFESTYLE_COLORS.get(lf, "gray")
        n_total = int(df["n_samples"].max())
        ax.plot(df["n_samples"], df["mean"], color=color, lw=2,
                label=f"{config.LIFESTYLE_DISPLAY.get(lf, lf)} (n<={n_total:,})")
        ax.fill_between(df["n_samples"], df["lo"], df["hi"],
                        color=color, alpha=0.15)

    ax.set_xlabel("Number of samples pooled")
    ax.set_ylabel("Total SGBs detected (union)")
    ax.set_xscale("log")
    ax.set_title(f"Composition accumulation by lifestyle\n"
                 f"(1:1 depth-matched, {spec.description})",
                 fontsize=FS+2, fontweight="bold")
    ax.legend(fontsize=FS)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    fpath = os.path.join(outdir,
                         f"composition_accumulation_{spec_tag}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")


# ════════════════════════════════════════════════════════════════════
#  Public entry point
# ════════════════════════════════════════════════════════════════════

def run_within_variation(data, dm, stats, specs=None):
    """Run all within-group variation analyses for each detection spec."""
    ref = data.ref_name
    print(f"\n{'='*60}")
    print(f"  Section 1 (within-group variation) [{ref}]")
    print(f"{'='*60}")

    if dm is None:
        print("  [WARN] No depth-matching - skipping within-group analyses")
        return

    if specs is None:
        specs = [
            DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP),
            DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH),
        ]
    elif not isinstance(specs, (list, tuple)):
        specs = [specs]

    for spec in specs:
        print(f"\n    [{spec.short_label}] detection: {spec.description}")
        compute_within_diversity(data, dm, stats, spec=spec)
        compute_within_beta(data, dm, stats, spec=spec)
        compute_composition_accumulation(data, dm, stats, spec=spec)

    print(f"\n  Within-group variation analyses complete [{ref}]")
