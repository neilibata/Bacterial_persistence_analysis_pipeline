from .. import config
#!/usr/bin/env python3
"""
Publication figures for Section 1 — reads exclusively from trace CSVs.

Usage
-----
    python -m persistence.figures --ref UHGG
    python -m persistence.figures --ref UHGG --traces-dir results/UHGG/.../traces

Produces:
    figures/section1/
        main_figure1.{pdf,png}         Main-text figure
        supp_health_triplets.{pdf,png}  Supplement: health analysis
        supp_300kb.{pdf,png}            Supplement: 300kb threshold
        supp_pairwise.{pdf,png}         Supplement: all pairwise comparisons
        section1_stats_summary.csv      Stats table for manuscript
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
from ..plot_utils import apply_oi_style, savefig_dual
apply_oi_style()
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu


# ═══════════════════════════════════════════════════════════════
#  STYLE
# ═══════════════════════════════════════════════════════════════

COLORS = {
    "H-G":          config.OI["vermillion"],
    "Transitional": config.OI["yellow"],
    "Industrial":   config.OI["sky"],
}
HEALTH_COLORS = {
    "Non-Industrial": config.OI["orange"],
    "Ind-Healthy":    config.OI["green"],
    "Ind-Unhealthy":  config.OI["blue"],
}
LIFESTYLE_ORDER = ["H-G", "Transitional", "Industrial"]
FS_TITLE = 10
FS_LABEL = 9
FS_TICK = 8
FS_ANNOT = 7.5


def _display_group(label):
    return config.LIFESTYLE_DISPLAY.get(label, label)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 9,
    "axes.linewidth": 0.6, "axes.grid": False,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
})


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def _fmt_p(p):
    if p < 1e-100:   return "p < 10⁻¹⁰⁰"
    elif p < 0.001:  return f"p = {p:.1e}"
    else:            return f"p = {p:.3f}"


def _stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def _sig_bracket(ax, x1, x2, y, p, fs=FS_ANNOT):
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y],
            lw=0.8, c="black", clip_on=False)
    ax.text((x1+x2)/2, y + h*1.2, f"{_fmt_p(p)} {_stars(p)}",
            ha="center", va="bottom", fontsize=fs, clip_on=False)


def _annotate_box_stats(ax, groups, positions, fs=FS_ANNOT):
    """
    Overlay white annotations *inside* each box:
      • median value   -> halfway between the median line and Q3
      • IQR string     -> halfway between Q1 and the median line

    This eliminates the need for a separate text-box legend.
    """
    for i, vals in enumerate(groups):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        x = positions[i]

        # Median label — upper half of box
        y_upper = (med + q3) / 2.0
        ax.text(x, y_upper, f"{med:.0f}",
                ha="center", va="center", fontsize=fs,
                color="white", fontweight="bold", zorder=5)

        # IQR label — lower half of box
        y_lower = (q1 + med) / 2.0
        ax.text(x, y_lower, f"[{q1:.0f}–{q3:.0f}]",
                ha="center", va="center", fontsize=fs - 0.5,
                color="white", zorder=5)


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════

class TraceData:
    """Lazy-load trace CSVs for one reference."""

    def __init__(self, traces_dir):
        self.dir = traces_dir
        self._cache = {}

    def _load(self, *parts):
        key = os.path.join(*parts)
        if key not in self._cache:
            path = os.path.join(self.dir, key)
            if os.path.exists(path):
                self._cache[key] = pd.read_csv(path)
            else:
                print(f"  [WARN] Missing: {path}")
                self._cache[key] = None
        return self._cache[key]

    def alpha(self, comparison, threshold):
        return self._load("alpha", f"{comparison}_{threshold}.csv")

    def pcoa(self, comparison, threshold):
        return self._load("beta", f"{comparison}_{threshold}_pcoa.csv")

    def permanova(self):
        return self._load("beta", "permanova_all.csv")

    def within_richness(self, threshold):
        return self._load("within", f"within_richness_{threshold}.csv")

    def within_beta(self, threshold):
        return self._load("within", f"within_beta_{threshold}.csv")


# ═══════════════════════════════════════════════════════════════
#  PANEL BUILDERS
# ═══════════════════════════════════════════════════════════════

def panel_richness_3group(ax, td, threshold="10pct"):
    """Richness boxplots for H-G / Transitional / Industrial."""
    df = td.within_richness(threshold)
    if df is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    rng = np.random.default_rng(42)
    groups, labels, colors, ns, medians = [], [], [], [], []

    for lf in LIFESTYLE_ORDER:
        vals = df.loc[df["Lifestyle"] == lf, "richness"].values
        if len(vals) == 0:
            continue
        groups.append(vals)
        labels.append(_display_group(lf))
        colors.append(COLORS[lf])
        ns.append(len(vals))
        medians.append(np.median(vals))

    positions = list(range(1, len(groups) + 1))
    bp = ax.boxplot(groups, positions=positions, patch_artist=True,
                    widths=0.55, showfliers=False,
                    medianprops=dict(color="white", lw=1.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
        patch.set_edgecolor("none")

    for i, (vals, c) in enumerate(zip(groups, colors)):
        n_show = min(len(vals), 1500)
        idx = rng.choice(len(vals), n_show, replace=False)
        jt = rng.uniform(-0.15, 0.15, n_show)
        ax.scatter(np.full(n_show, positions[i]) + jt, vals[idx],
                   c=c, alpha=0.06, s=3, edgecolors="none",
                   rasterized=True, zorder=1)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{l}\n(n={n:,})" for l, n in zip(labels, ns)],
                       fontsize=FS_TICK)
    ax.set_ylabel("Richness (SGBs detected)", fontsize=FS_LABEL)

    if len(groups) >= 3:
        _, p_it = mannwhitneyu(groups[2], groups[1], alternative="two-sided")
        _, p_ih = mannwhitneyu(groups[2], groups[0], alternative="two-sided")
        _, p_ht = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        ymax = max(np.percentile(g, 99) for g in groups)
        gap = ymax * 0.06
        _sig_bracket(ax, 2, 3, ymax + gap, p_it)
        _sig_bracket(ax, 1, 2, ymax + gap * 3.5, p_ht)
        _sig_bracket(ax, 1, 3, ymax + gap * 6, p_ih)
        ax.set_ylim(0, ymax + gap * 9)

    # White in-box annotations (median & IQR)
    _annotate_box_stats(ax, groups, positions)

    thr = ">=10% breadth" if "10" in threshold else ">=300 kb"
    ax.set_title(f"Richness by lifestyle\n(depth-matched, {thr})",
                 fontsize=FS_TITLE)


def panel_pcoa(ax, td, comparison="ind_vs_ni", threshold="10pct"):
    """PCoA scatter colored by lifestyle."""
    df = td.pcoa(comparison, threshold)
    if df is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    v1 = df["var_pct_PCo1"].iloc[0]
    v2 = df["var_pct_PCo2"].iloc[0]
    rng = np.random.default_rng(42)

    for lf in reversed(LIFESTYLE_ORDER):
        mask = df["Lifestyle"] == lf
        sub = df[mask]
        if len(sub) == 0:
            continue
        n_show = min(len(sub), 2000)
        idx = rng.choice(len(sub), n_show, replace=False)
        ax.scatter(sub.iloc[idx]["PCo1"], sub.iloc[idx]["PCo2"],
                   c=COLORS[lf], alpha=0.15, s=4, edgecolors="none",
                   label=f"{_display_group(lf)} (n={len(sub):,})",
                   rasterized=True)

    # Density contours
    for lf in LIFESTYLE_ORDER:
        sub = df[df["Lifestyle"] == lf]
        if len(sub) < 50:
            continue
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([sub["PCo1"].values, sub["PCo2"].values])
            kde = gaussian_kde(xy, bw_method=0.3)
            xg = np.linspace(df["PCo1"].min(), df["PCo1"].max(), 80)
            yg = np.linspace(df["PCo2"].min(), df["PCo2"].max(), 80)
            X, Y = np.meshgrid(xg, yg)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.contour(X, Y, Z, levels=3, colors=[COLORS[lf]],
                       linewidths=0.6, alpha=0.5)
        except Exception:
            pass

    ax.set_xlabel(f"PCo1 ({v1:.1f}%)", fontsize=FS_LABEL)
    ax.set_ylabel(f"PCo2 ({v2:.1f}%)", fontsize=FS_LABEL)

    perm = td.permanova()
    if perm is not None:
        row = perm[(perm["comparison"] == comparison) &
                   (perm["detection"].str.contains("10%"))].head(1)
        if len(row):
            ax.text(0.03, 0.03,
                    f"PERMANOVA\nF = {row.iloc[0]['F']:.1f}, "
                    f"{_fmt_p(row.iloc[0]['p'])}",
                    transform=ax.transAxes, fontsize=FS_ANNOT,
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="gray", alpha=0.8, lw=0.5))

    ax.legend(fontsize=FS_ANNOT, loc="upper right",
              markerscale=3, framealpha=0.8)
    ax.set_title("Compositional variation (PCoA)\n"
                 "(Bray-Curtis, depth-matched)", fontsize=FS_TITLE)


def panel_within_beta(ax, td, threshold="10pct"):
    """Within- and between-group Jaccard distance bars."""
    df = td.within_beta(threshold)
    if df is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    jdf = df[df["metric"] == "jaccard"]
    y_map = {"H-G": 3, "Transitional": 2, "Industrial": 1,
             "Between (Ind vs NI)": 0}

    for _, row in jdf.iterrows():
        lf = row["Lifestyle"]
        med, q25, q75 = row["median"], row["q25"], row["q75"]
        if lf == "Between (Ind vs NI)":
            c = "#555555"
        elif lf in COLORS:
            c = COLORS[lf]
        else:
            continue
        y_pos = y_map.get(lf, -1)
        if y_pos < 0:
            continue
        ax.barh(y_pos, q75 - q25, left=q25, height=0.5,
                color=c, alpha=0.4, edgecolor=c, lw=0.8)
        ax.plot(med, y_pos, "|", color=c, ms=15, mew=2)
        ax.text(q75 + 0.01, y_pos,
                f"{_display_group(lf)} (med={med:.2f})",
                va="center", fontsize=FS_ANNOT, color=c)

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["Between", "Industrial", "Transitional",
                        "Hunter-Gatherer"], fontsize=FS_TICK)
    ax.set_xlabel("Jaccard distance", fontsize=FS_LABEL)
    ax.set_title("Within- and between-group\ncompositional distances",
                 fontsize=FS_TITLE)
    ax.set_xlim(0, 1.05)


def panel_within_richness_violin(ax, td, threshold="10pct"):
    """Within-group richness violins showing spread."""
    df = td.within_richness(threshold)
    if df is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    rng = np.random.default_rng(42)
    groups, labels, colors = [], [], []

    for lf in LIFESTYLE_ORDER:
        vals = df.loc[df["Lifestyle"] == lf, "richness"].dropna().values
        if len(vals) < 10:
            continue
        groups.append(vals); labels.append(_display_group(lf)); colors.append(COLORS[lf])

    positions = list(range(1, len(groups) + 1))
    parts = ax.violinplot(groups, positions=positions,
                          showextrema=False, showmedians=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i]); pc.set_alpha(0.3)
        pc.set_edgecolor(colors[i])

    bp = ax.boxplot(groups, positions=positions, patch_artist=True,
                    widths=0.2, showfliers=False,
                    medianprops=dict(color="white", lw=1.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
        patch.set_edgecolor("none")

    # White in-box annotations (median & IQR)
    _annotate_box_stats(ax, groups, positions)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{l}\n(n={len(g):,})"
                        for l, g in zip(labels, groups)], fontsize=FS_TICK)
    ax.set_ylabel("Richness (SGBs)", fontsize=FS_LABEL)
    thr = ">=10% breadth" if "10" in threshold else ">=300 kb"
    ax.set_title(f"Within-group richness variation\n"
                 f"(depth-matched, {thr})", fontsize=FS_TITLE)
    ax.set_ylim(0, None)


def panel_triplet_health(ax, td, threshold="10pct"):
    """Triplet-matched NI / IndH / IndU boxplots."""
    from scipy.stats import wilcoxon

    df = td.alpha("triplet", threshold)
    if df is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    rng = np.random.default_rng(42)
    order = ["Non-Industrial", "Ind-Healthy", "Ind-Unhealthy"]
    groups, ns = [], []
    for g in order:
        vals = df.loc[df["group"] == g, "richness"].values
        groups.append(vals); ns.append(len(vals))

    color_list = [HEALTH_COLORS[g] for g in order]
    bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                    showfliers=False,
                    medianprops=dict(color="white", lw=1.5))
    for patch, c in zip(bp["boxes"], color_list):
        patch.set_facecolor(c); patch.set_alpha(0.6)
        patch.set_edgecolor("none")

    for i, (vals, c) in enumerate(zip(groups, color_list), 1):
        n_show = min(len(vals), 1500)
        idx = rng.choice(len(vals), n_show, replace=False)
        jt = rng.uniform(-0.15, 0.15, n_show)
        ax.scatter(np.full(n_show, i) + jt, vals[idx],
                   c=c, alpha=0.06, s=3, edgecolors="none",
                   rasterized=True, zorder=1)

    ax.set_xticklabels([f"{g}\n(n={n:,})" for g, n in zip(order, ns)],
                       fontsize=FS_TICK)
    ax.set_ylabel("Richness (SGBs)", fontsize=FS_LABEL)

    # White in-box annotations (median & IQR)
    _annotate_box_stats(ax, groups, list(range(1, len(groups) + 1)))

    if all(len(g) > 0 for g in groups):
        _, p_hu = wilcoxon(groups[1], groups[2], alternative="two-sided")
        _, p_nih = wilcoxon(groups[0], groups[1], alternative="two-sided")
        _, p_niu = wilcoxon(groups[0], groups[2], alternative="two-sided")
        ymax = max(np.percentile(g, 99) for g in groups)
        gap = ymax * 0.06
        _sig_bracket(ax, 2, 3, ymax + gap, p_hu)
        _sig_bracket(ax, 1, 2, ymax + gap * 3.5, p_nih)
        _sig_bracket(ax, 1, 3, ymax + gap * 6, p_niu)
        ax.set_ylim(0, ymax + gap * 9)

    thr = ">=10% breadth" if "10" in threshold else ">=300 kb"
    ax.set_title(f"Triplet-matched health ({thr})\n"
                 f"n={ns[0]:,} each", fontsize=FS_TITLE)


def panel_pcoa_triplet(ax, td, threshold="10pct"):
    """PCoA for triplet-matched health groups."""
    df = td.pcoa("triplet", threshold)
    if df is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    v1 = df["var_pct_PCo1"].iloc[0]
    v2 = df["var_pct_PCo2"].iloc[0]
    rng = np.random.default_rng(42)

    for g in reversed(["Non-Industrial", "Ind-Healthy", "Ind-Unhealthy"]):
        sub = df[df["group"] == g]
        if len(sub) == 0:
            continue
        n_show = min(len(sub), 2000)
        idx = rng.choice(len(sub), n_show, replace=False)
        ax.scatter(sub.iloc[idx]["PCo1"], sub.iloc[idx]["PCo2"],
                   c=HEALTH_COLORS[g], alpha=0.12, s=4,
                   edgecolors="none",
                   label=f"{g} (n={len(sub):,})", rasterized=True)

    ax.set_xlabel(f"PCo1 ({v1:.1f}%)", fontsize=FS_LABEL)
    ax.set_ylabel(f"PCo2 ({v2:.1f}%)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_ANNOT, loc="upper right",
              markerscale=3, framealpha=0.8)

    thr = ">=10% breadth" if "10" in threshold else ">=300 kb"
    ax.set_title(f"Health PCoA (Bray-Curtis, {thr})", fontsize=FS_TITLE)


# ═══════════════════════════════════════════════════════════════
#  FIGURE BUILDERS
# ═══════════════════════════════════════════════════════════════

def build_main_figure1(td, outdir):
    """Main-text Figure 1: between- and within-lifestyle diversity."""
    print("  Building main figure 1 …")
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    panel_richness_3group(ax, td, "10pct")
    ax.text(-0.10, 1.05, "A", transform=ax.transAxes,
            fontsize=16, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    panel_pcoa(ax, td, "ind_vs_ni", "10pct")
    ax.text(-0.10, 1.05, "B", transform=ax.transAxes,
            fontsize=16, fontweight="bold")

    ax = fig.add_subplot(gs[1, 0])
    panel_within_beta(ax, td, "10pct")
    ax.text(-0.10, 1.05, "C", transform=ax.transAxes,
            fontsize=16, fontweight="bold")

    ax = fig.add_subplot(gs[1, 1])
    panel_within_richness_violin(ax, td, "10pct")
    ax.text(-0.10, 1.05, "D", transform=ax.transAxes,
            fontsize=16, fontweight="bold")

    fig.suptitle("Gut microbial diversity across human lifestyles",
                 fontsize=13, fontweight="bold", y=0.98)

    for fmt in ["pdf", "png"]:
        savefig_dual(fig, os.path.join(outdir, f"main_figure1.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: main_figure1.pdf")


def build_supp_health(td, outdir):
    """Supplement: triplet health at both thresholds."""
    print("  Building supplement: health triplets …")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    panel_triplet_health(axes[0, 0], td, "10pct")
    axes[0, 0].text(-0.10, 1.05, "A", transform=axes[0, 0].transAxes,
                    fontsize=14, fontweight="bold")

    panel_pcoa_triplet(axes[0, 1], td, "10pct")
    axes[0, 1].text(-0.10, 1.05, "B", transform=axes[0, 1].transAxes,
                    fontsize=14, fontweight="bold")

    panel_triplet_health(axes[1, 0], td, "300kb")
    axes[1, 0].text(-0.10, 1.05, "C", transform=axes[1, 0].transAxes,
                    fontsize=14, fontweight="bold")

    panel_pcoa_triplet(axes[1, 1], td, "300kb")
    axes[1, 1].text(-0.10, 1.05, "D", transform=axes[1, 1].transAxes,
                    fontsize=14, fontweight="bold")

    fig.suptitle("Health status and gut microbial diversity\n"
                 "(triplet depth-matched)",
                 fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for fmt in ["pdf", "png"]:
        savefig_dual(fig, os.path.join(outdir, f"supp_health_triplets.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: supp_health_triplets.pdf")


def build_supp_300kb(td, outdir):
    """Supplement: 300kb threshold versions of main panels."""
    print("  Building supplement: 300kb threshold …")
    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    panel_richness_3group(ax, td, "300kb")
    ax.text(-0.10, 1.05, "A", transform=ax.transAxes,
            fontsize=14, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    panel_pcoa(ax, td, "ind_vs_ni", "300kb")
    ax.text(-0.10, 1.05, "B", transform=ax.transAxes,
            fontsize=14, fontweight="bold")

    ax = fig.add_subplot(gs[1, 0])
    panel_within_beta(ax, td, "300kb")
    ax.text(-0.10, 1.05, "C", transform=ax.transAxes,
            fontsize=14, fontweight="bold")

    ax = fig.add_subplot(gs[1, 1])
    panel_within_richness_violin(ax, td, "300kb")
    ax.text(-0.10, 1.05, "D", transform=ax.transAxes,
            fontsize=14, fontweight="bold")

    fig.suptitle("Supplementary: >=300 kb mapped bases threshold",
                 fontsize=12, fontweight="bold", y=0.98)

    for fmt in ["pdf", "png"]:
        savefig_dual(fig, os.path.join(outdir, f"supp_300kb.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: supp_300kb.pdf")


def _panel_pairwise_distances(ax, td, comp, title, metric, threshold="10pct"):
    """
    Bar-range plot of within-group distances for the two groups in a
    pairwise comparison, plus the between-group distance.

    Parameters
    ----------
    metric : str
        'jaccard' or 'braycurtis'
    """
    bdf = td.within_beta(threshold)
    if bdf is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    mdf = bdf[bdf["metric"] == metric]

    # Map comparison to relevant lifestyles
    comp_map = {
        "ind_vs_ni":    (["Industrial", "H-G", "Transitional",
                          "Between (Ind vs NI)"],),
        "ind_vs_trans": (["Industrial", "Transitional",
                          "Between (Ind vs NI)"],),
        "ind_vs_hg":    (["Industrial", "H-G",
                          "Between (Ind vs NI)"],),
    }
    want = comp_map.get(comp, (["Industrial", "H-G", "Transitional",
                                 "Between (Ind vs NI)"],))[0]

    y_pos = 0
    yticks, ylabels = [], []
    for lf in reversed(want):
        row = mdf[mdf["Lifestyle"] == lf]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        med, q25, q75 = row["median"], row["q25"], row["q75"]
        if lf == "Between (Ind vs NI)":
            c = "#555555"
        elif lf in COLORS:
            c = COLORS[lf]
        else:
            c = "#888888"
        ax.barh(y_pos, q75 - q25, left=q25, height=0.5,
                color=c, alpha=0.4, edgecolor=c, lw=0.8)
        ax.plot(med, y_pos, "|", color=c, ms=15, mew=2)
        ax.text(q75 + 0.01, y_pos, f"med={med:.2f}",
                va="center", fontsize=FS_ANNOT, color=c)
        yticks.append(y_pos)
        ylabels.append(_display_group(lf))
        y_pos += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=FS_TICK)
    metric_label = "Jaccard" if metric == "jaccard" else "Bray-Curtis"
    ax.set_xlabel(f"{metric_label} distance", fontsize=FS_LABEL)
    ax.set_title(f"{title}\n{metric_label} distances", fontsize=FS_TITLE)
    ax.set_xlim(0, 1.05)


def build_supp_pairwise(td, outdir):
    """
    Supplement: all pairwise comparisons.

    Layout per row: Richness | Jaccard | Bray-Curtis | PCoA
    """
    print("  Building supplement: pairwise comparisons …")

    comparisons = [
        ("ind_vs_ni", "Industrial vs Non-Industrial"),
        ("ind_vs_trans", "Industrial vs Transitional"),
        ("ind_vs_hg", "Industrial vs Hunter-Gatherer"),
    ]
    active = [(c, l) for c, l in comparisons
              if td.alpha(c, "10pct") is not None]
    if not active:
        print("    No pairwise data — skipping")
        return

    nrows = len(active)
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.0 * ncols, 4.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    rng = np.random.default_rng(42)
    panel_idx = 0

    for row, (comp, title) in enumerate(active):
        adf = td.alpha(comp, "10pct")
        group_names = adf["group"].unique()
        g1 = adf.loc[adf["group"] == group_names[0], "richness"].values
        g2 = adf.loc[adf["group"] == group_names[1], "richness"].values
        c1 = COLORS.get(group_names[0], config.OI["vermillion"])
        c2 = COLORS.get(group_names[1], config.OI["orange"])

        # ── Col 0: Richness boxplot ──
        ax = axes[row, 0]
        bp = ax.boxplot([g1, g2], patch_artist=True, widths=0.5,
                        showfliers=False,
                        medianprops=dict(color="white", lw=1.5))
        for patch, c in zip(bp["boxes"], [c1, c2]):
            patch.set_facecolor(c); patch.set_alpha(0.6)
            patch.set_edgecolor("none")
        ax.set_xticklabels([f"{_display_group(group_names[0])}\n(n={len(g1):,})",
                            f"{_display_group(group_names[1])}\n(n={len(g2):,})"],
                           fontsize=FS_TICK)
        ax.set_ylabel("Richness", fontsize=FS_LABEL)
        _annotate_box_stats(ax, [g1, g2], [1, 2])
        _, p = mannwhitneyu(g1, g2, alternative="two-sided")
        ax.set_title(f"{title}\nRichness {_fmt_p(p)} {_stars(p)}",
                     fontsize=FS_TITLE)
        ax.text(-0.12, 1.05, chr(65 + panel_idx),
                transform=ax.transAxes, fontsize=14, fontweight="bold")
        panel_idx += 1

        # ── Col 1: Jaccard distances ──
        ax = axes[row, 1]
        _panel_pairwise_distances(ax, td, comp, title, "jaccard")
        ax.text(-0.12, 1.05, chr(65 + panel_idx),
                transform=ax.transAxes, fontsize=14, fontweight="bold")
        panel_idx += 1

        # ── Col 2: Bray-Curtis distances ──
        ax = axes[row, 2]
        _panel_pairwise_distances(ax, td, comp, title, "braycurtis")
        ax.text(-0.12, 1.05, chr(65 + panel_idx),
                transform=ax.transAxes, fontsize=14, fontweight="bold")
        panel_idx += 1

        # ── Col 3: PCoA ──
        ax = axes[row, 3]
        pdf = td.pcoa(comp, "10pct")
        if pdf is not None:
            v1 = pdf["var_pct_PCo1"].iloc[0]
            v2 = pdf["var_pct_PCo2"].iloc[0]
            for g in group_names:
                sub = pdf[pdf["group"] == g]
                c = COLORS.get(g, "#888888")
                n_show = min(len(sub), 2000)
                idx = rng.choice(len(sub), n_show, replace=False)
                ax.scatter(sub.iloc[idx]["PCo1"], sub.iloc[idx]["PCo2"],
                           c=c, alpha=0.12, s=4, edgecolors="none",
                           label=_display_group(g), rasterized=True)
            ax.set_xlabel(f"PCo1 ({v1:.1f}%)", fontsize=FS_LABEL)
            ax.set_ylabel(f"PCo2 ({v2:.1f}%)", fontsize=FS_LABEL)
            ax.legend(fontsize=FS_ANNOT, markerscale=3)
            ax.set_title(f"{title}\nPCoA (Bray-Curtis)", fontsize=FS_TITLE)
        ax.text(-0.12, 1.05, chr(65 + panel_idx),
                transform=ax.transAxes, fontsize=14, fontweight="bold")
        panel_idx += 1

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        savefig_dual(fig, os.path.join(outdir, f"supp_pairwise.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: supp_pairwise.pdf")



def panel_richness_ni_split(ax, td, split_col, split_value, split_label,
                            other_label="Other NI", threshold="10pct"):
    """
    Robustness boxplot: Industrial vs *split_value* vs remaining NI.

    Parameters
    ----------
    split_col : str
        Column in the within-richness trace to split on
        ('Population' or 'Lifestyle').
    split_value : str
        The value that defines the focal NI subgroup (e.g. 'Hadza', 'H-G').
    split_label : str
        Display name for the focal group.
    other_label : str
        Display name for the remaining NI samples.
    """
    df = td.within_richness(threshold)
    if df is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    if split_col not in df.columns:
        ax.text(0.5, 0.5, f"No '{split_col}' column\nin traces \u2014 re-run traces",
                transform=ax.transAxes, ha="center", fontsize=FS_ANNOT)
        return

    # Build three groups
    ind_mask = df["Lifestyle"] == "Industrial"
    ni_mask = df["Lifestyle"].isin(["H-G", "Transitional"])
    focal_mask = ni_mask & (df[split_col] == split_value)
    other_mask = ni_mask & (df[split_col] != split_value)

    order = [
        (split_label, focal_mask, config.OI["orange"]),
        (other_label, other_mask, config.OI["yellow"]),
        ("Industrial", ind_mask, config.OI["sky"]),
    ]

    rng = np.random.default_rng(42)
    groups, labels, colors, ns = [], [], [], []
    for lbl, mask, c in order:
        vals = df.loc[mask, "richness"].values
        if len(vals) == 0:
            continue
        groups.append(vals)
        labels.append(lbl)
        colors.append(c)
        ns.append(len(vals))

    if len(groups) < 2:
        ax.text(0.5, 0.5, "Insufficient groups", transform=ax.transAxes,
                ha="center")
        return

    positions = list(range(1, len(groups) + 1))
    bp = ax.boxplot(groups, positions=positions, patch_artist=True,
                    widths=0.55, showfliers=False,
                    medianprops=dict(color="white", lw=1.5))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
        patch.set_edgecolor("none")

    for i, (vals, c) in enumerate(zip(groups, colors)):
        n_show = min(len(vals), 1500)
        idx = rng.choice(len(vals), n_show, replace=False)
        jt = rng.uniform(-0.15, 0.15, n_show)
        ax.scatter(np.full(n_show, positions[i]) + jt, vals[idx],
                   c=c, alpha=0.06, s=3, edgecolors="none",
                   rasterized=True, zorder=1)

    # In-box annotations
    _annotate_box_stats(ax, groups, positions)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{l}\n(n={n:,})" for l, n in zip(labels, ns)],
                       fontsize=FS_TICK)
    ax.set_ylabel("Richness (SGBs detected)", fontsize=FS_LABEL)

    # Significance brackets: pairwise Industrial vs each NI subgroup
    ind_idx = labels.index("Industrial") if "Industrial" in labels else -1
    if ind_idx >= 0 and len(groups) >= 2:
        ymax = max(np.percentile(g, 99) for g in groups)
        gap = ymax * 0.06
        bracket_y = ymax + gap
        for i in range(len(groups)):
            if i == ind_idx:
                continue
            _, p = mannwhitneyu(groups[i], groups[ind_idx],
                                alternative="two-sided")
            _sig_bracket(ax, positions[i], positions[ind_idx],
                         bracket_y, p)
            bracket_y += gap * 3.5
        # Bracket between the two NI sub-groups
        ni_idxs = [i for i in range(len(groups)) if i != ind_idx]
        if len(ni_idxs) == 2:
            _, p01 = mannwhitneyu(groups[ni_idxs[0]], groups[ni_idxs[1]],
                                   alternative="two-sided")
            _sig_bracket(ax, positions[ni_idxs[0]], positions[ni_idxs[1]],
                         bracket_y, p01)
            bracket_y += gap * 3.5
        ax.set_ylim(0, bracket_y + gap * 2)

    thr = "\u226510% breadth" if "10" in threshold else "\u2265300 kb"
    ax.set_title(f"Richness: {split_label} vs {other_label}\n"
                 f"vs Industrial (depth-matched, {thr})", fontsize=FS_TITLE)


def panel_pcoa_ni_split(ax, td, split_col, split_value, split_label,
                         other_label="Other NI",
                         comparison="ind_vs_ni", threshold="10pct"):
    """PCoA scatter with the same NI split colouring."""
    pdf = td.pcoa(comparison, threshold)
    if pdf is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return

    # We need Population in the PCoA trace too; if absent, try to join
    # from within_richness (which carries Population from the trace update)
    if split_col not in pdf.columns:
        wdf = td.within_richness(threshold)
        if wdf is not None and split_col in wdf.columns:
            pop_map = wdf.set_index("sample_id")[split_col]
            pdf = pdf.copy()
            pdf[split_col] = pdf["sample_id"].map(pop_map)
        else:
            ax.text(0.5, 0.5, f"No '{split_col}' in traces",
                    transform=ax.transAxes, ha="center", fontsize=FS_ANNOT)
            return

    v1 = pdf["var_pct_PCo1"].iloc[0]
    v2 = pdf["var_pct_PCo2"].iloc[0]
    rng = np.random.default_rng(42)

    ind_mask = pdf["Lifestyle"] == "Industrial"
    ni_mask = pdf["Lifestyle"].isin(["H-G", "Transitional"])
    focal_mask = ni_mask & (pdf[split_col] == split_value)
    other_mask = ni_mask & (pdf[split_col] != split_value)

    draw_order = [
        ("Industrial", ind_mask, config.OI["sky"]),
        (other_label, other_mask, config.OI["yellow"]),
        (split_label, focal_mask, config.OI["orange"]),
    ]

    for lbl, mask, c in draw_order:
        sub = pdf[mask]
        if len(sub) == 0:
            continue
        n_show = min(len(sub), 2000)
        idx = rng.choice(len(sub), n_show, replace=False)
        ax.scatter(sub.iloc[idx]["PCo1"], sub.iloc[idx]["PCo2"],
                   c=c, alpha=0.15, s=4, edgecolors="none",
                   label=f"{lbl} (n={len(sub):,})", rasterized=True)

    ax.set_xlabel(f"PCo1 ({v1:.1f}%)", fontsize=FS_LABEL)
    ax.set_ylabel(f"PCo2 ({v2:.1f}%)", fontsize=FS_LABEL)
    ax.legend(fontsize=FS_ANNOT, loc="upper right",
              markerscale=3, framealpha=0.8)
    thr = "\u226510% breadth" if "10" in threshold else "\u2265300 kb"
    ax.set_title(f"PCoA: {split_label} vs {other_label}\n"
                 f"vs Industrial ({thr})", fontsize=FS_TITLE)


def build_supp_robustness_splits(td, outdir):
    """
    Supplement: Robustness of Ind vs NI richness under NI sub-splits.

    Row 1: Hadza vs other NI vs Industrial  (split on Population)
    Row 2: H-G vs other NI vs Industrial    (split on Lifestyle)

    Columns: Richness boxplot | PCoA
    """
    print("  Building supplement: NI robustness splits \u2026")

    # Check that Population column is available
    wdf = td.within_richness("10pct")
    has_pop = wdf is not None and "Population" in wdf.columns

    splits = []
    if has_pop:
        splits.append(("Population", "Hadza", "Hadza", "Other NI"))
    else:
        print("    [WARN] No 'Population' column in traces \u2014 "
              "Hadza split skipped. Re-run traces to include it.")
    splits.append(("Lifestyle", "H-G", "Hunter-Gatherer", "Transitional"))

    nrows = len(splits)
    if nrows == 0:
        print("    No splits available \u2014 skipping")
        return

    fig, axes = plt.subplots(nrows, 2, figsize=(12, 5.5 * nrows),
                             squeeze=False)

    for row, (col, val, label, other) in enumerate(splits):
        panel_richness_ni_split(
            axes[row, 0], td,
            split_col=col, split_value=val,
            split_label=label, other_label=other)
        axes[row, 0].text(-0.10, 1.05, chr(65 + row * 2),
                          transform=axes[row, 0].transAxes,
                          fontsize=14, fontweight="bold")

        panel_pcoa_ni_split(
            axes[row, 1], td,
            split_col=col, split_value=val,
            split_label=label, other_label=other)
        axes[row, 1].text(-0.10, 1.05, chr(66 + row * 2),
                          transform=axes[row, 1].transAxes,
                          fontsize=14, fontweight="bold")

    fig.suptitle("Robustness: Industrial richness loss across\n"
                 "Non-Industrial sub-populations",
                 fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for fmt in ["pdf", "png"]:
        savefig_dual(fig, os.path.join(outdir, f"supp_robustness_ni_splits.{fmt}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: supp_robustness_ni_splits.pdf")


def build_stats_table(td, outdir):
    """Summary stats table from traces."""
    print("  Building stats summary …")
    rows = []
    perm = td.permanova()
    if perm is not None:
        for _, r in perm.iterrows():
            rows.append(dict(
                claim="composition", comparison=r["comparison"],
                detection=r["detection"], test="PERMANOVA",
                statistic=f"F = {r['F']:.1f}", p=r["p"],
                n=int(r.get("n_a", 0))))
            if np.isfinite(r.get("richness_ratio", np.nan)):
                rows.append(dict(
                    claim="richness", comparison=r["comparison"],
                    detection=r["detection"], test="Mann-Whitney U",
                    statistic=f"ratio = {r['richness_ratio']:.2f}x",
                    p=r.get("richness_p", np.nan),
                    n=int(r.get("n_a", 0))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "section1_stats_summary.csv"),
              index=False)
    print(f"    Saved: section1_stats_summary.csv")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate Section 1 publication figures from traces")
    parser.add_argument("--ref", default="UHGG")
    parser.add_argument("--traces-dir", default=None)
    parser.add_argument("--out", default="figures/section1")
    args = parser.parse_args()

    traces_dir = args.traces_dir or os.path.join(
        "results", args.ref, "section1_diversity", "traces")

    if not os.path.isdir(traces_dir):
        print(f"ERROR: Traces directory not found: {traces_dir}")
        print("Run  python -m persistence --only s1_traces  first.")
        sys.exit(1)

    outdir = _ensure(args.out)
    td = TraceData(traces_dir)

    print(f"\n  Traces: {traces_dir}")
    print(f"  Output: {outdir}")
    print(f"  {'='*50}\n")

    build_main_figure1(td, outdir)
    build_supp_health(td, outdir)
    build_supp_300kb(td, outdir)
    build_supp_pairwise(td, outdir)
    build_supp_robustness_splits(td, outdir)
    build_stats_table(td, outdir)

    print(f"\n  Done. Output in: {outdir}/")


if __name__ == "__main__":
    main()
