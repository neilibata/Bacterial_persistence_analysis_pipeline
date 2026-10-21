"""
supp_s5.py — Tables S4 / S5 and Figures S5A / S5B.

Generates supplementary outputs that document the VANISH / BloSSUM list
family: how the three contrasts are defined, the prevalence and abundance
signals for each, and the overlap between the original lists, the
GMW-derived VANISH+ / BloSSUM+, and the healthy-contrast VanishH+ /
BlossumH+ variants.

Entry point
-----------
    from .supp_s5 import run_supp_s5
    run_supp_s5(
        data,
        vanish_plus_ids, blossum_plus_ids,
        vanish_h_ids,    blossum_h_ids,
        abund_df=None,       # DataFrame from run_differential(); loaded
                             # from disk if not supplied
        taxonomy_df=None,    # optional GTDB taxonomy DataFrame
        out_dir=None,        # defaults to section_dir(ref, "s2", "supplement")
    )

Outputs
-------
    table_s4_list_membership.csv
    table_s5_overlap_summary.csv
    fig_s5a_list_derivation_{medium,large}{,_notitle}.png
    fig_s5b_overlap_analysis_{medium,large}{,_notitle}.png

Dependencies
------------
    numpy, pandas, scipy, matplotlib  (all already in the pipeline)
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from . import config
from .plot_utils import (apply_oi_style, savefig_dual, current_fs,
                         PAGE_WIDTH_IN, add_post_layout_hook)
from .common import assign_volcano_category_7
from .section2_pooling import _build_taxonomy_map

apply_oi_style()

# ══════════════════════════════════════════════════════════════════════
#  PALETTE  — aligned with config.LIST_COLORS and VOLCANO_PALETTE_7
# ══════════════════════════════════════════════════════════════════════

OI = config.OI

# VANISH-family colours (one per generation of list)
C_VANISH    = OI["vermillion"]   # #D55E00  original VANISH
C_VANISH_P  = OI["yellow"]       # #F0E442  VANISH+
C_VANISH_H  = OI["orange"]       # #E69F00  VanishH+
C_SHARED_V  = "#A0522D"          # brownish — original ∩ VANISH+ overlap

# BloSSUM-family colours
C_BLOSSUM   = OI["blue"]         # #0072B2  original BloSSUM
C_BLOSSUM_P = OI["green"]        # #009E73  BloSSUM+
C_BLOSSUM_H = OI["sky"]          # #56B4E9  BlossumH+
C_SHARED_B  = "#006400"          # dark green — original ∩ BloSSUM+ overlap

C_OTHER     = "#BBBBBB"          # background taxa


# ══════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════

def _load_gmw_csv(data, suffix: str) -> Optional[pd.DataFrame]:
    """Load a saved GMW prevalence CSV produced by run_differential_gmw."""
    path = os.path.join(config.outdir(data.ref_name),
                        f"prevalence_gmw_{suffix}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"    [supp_s5] WARNING: not found: {path}")
    return None


def _load_abund_csv(data) -> Optional[pd.DataFrame]:
    """Load the hard-matched abundance-differential CSV."""
    path = os.path.join(config.outdir(data.ref_name),
                        "abundance_ind_vs_nonind.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"    [supp_s5] WARNING: not found: {path}")
    return None


def _prev_lookups(df: Optional[pd.DataFrame]):
    """Return three dicts: {microbe: prev_g1}, {microbe: prev_g2}, {microbe: delta}."""
    if df is None or df.empty:
        return {}, {}, {}
    d1  = dict(zip(df["microbe"], df["prev_g1"]))
    d2  = dict(zip(df["microbe"], df["prev_g2"]))
    dd  = dict(zip(df["microbe"], df["prev_g1"] - df["prev_g2"]))
    return d1, d2, dd


def _jaccard(a, b) -> float:
    a, b = set(a), set(b)
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _overlap_row(comp_label, name_a, ids_a, name_b, ids_b, family) -> dict:
    """One comparison row for Table S5."""
    a, b = set(ids_a), set(ids_b)
    inter = a & b
    union = a | b
    return {
        "family":      family,
        "list_A":      name_a,
        "list_B":      name_b,
        "n_A":         len(a),
        "n_B":         len(b),
        "overlap":     len(inter),
        "A_only":      len(a - b),
        "B_only":      len(b - a),
        "Jaccard":     round(len(inter) / len(union), 4) if union else 0.0,
    }


def _spines_off(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)


def _median_nonan(values) -> float:
    vals = [v for v in values if not np.isnan(v)]
    return float(np.median(vals)) if vals else np.nan


# ══════════════════════════════════════════════════════════════════════
#  TABLE S4 — Full list-family membership
# ══════════════════════════════════════════════════════════════════════

def build_table_s4(data,
                   vanish_plus_ids, blossum_plus_ids,
                   vanish_h_ids,    blossum_h_ids,
                   taxonomy_df=None) -> pd.DataFrame:
    """
    One row per SGB present in at least one of the six lists.

    Columns
    -------
    sgb_id, taxonomy,
    original_VANISH, original_BloSSUM,
    VANISH_plus, BloSSUM_plus, VanishH_plus, BlossumH_plus,
    prev_I_gmw_all, prev_NI_gmw_all, prev_IndH_gmw_healthy,
    delta_p_all_vs_NI, delta_p_healthy_vs_NI,
    clr_diff_abundance
    """
    vanish  = data.vanish_set
    blossum = data.blossum_set
    vp_set  = set(vanish_plus_ids)
    bp_set  = set(blossum_plus_ids)
    vh_set  = set(vanish_h_ids)
    bh_set  = set(blossum_h_ids)

    all_sgbs = vanish | blossum | vp_set | bp_set | vh_set | bh_set
    if not all_sgbs:
        print("    [supp_s5] WARNING: no SGBs in any list — Table S4 empty")
        return pd.DataFrame()

    # ── Prevalence tables ──
    gmw_all     = _load_gmw_csv(data, "ind_all_vs_ni")
    gmw_healthy = _load_gmw_csv(data, "ind_healthy_vs_ni")
    abund       = _load_abund_csv(data)

    prev_I_all,  prev_NI_all,  delta_all  = _prev_lookups(gmw_all)
    prev_I_hlt,  _,            delta_hlt  = _prev_lookups(gmw_healthy)
    clr_d = (dict(zip(abund["microbe"], abund["clr_diff"]))
             if abund is not None and not abund.empty else {})

    # ── Taxonomy ──
    tax_map = _build_taxonomy_map(taxonomy_df)
    # Supplement from any available phylo annotation files
    for fpath in (getattr(config, "VANISH_PHYLO_FILE",  "Vanish_phylo.txt"),
                  getattr(config, "BLOSSUM_PHYLO_FILE", "Blossom_phylo.txt")):
        if not os.path.exists(fpath):
            continue
        try:
            tdf = pd.read_csv(fpath, sep=None, engine="python")
            tax_col = next(
                (c for c in tdf.columns
                 if any(k in c.lower() for k in ("gtdb", "tax", "lineage"))),
                None)
            id_col = tdf.columns[0]
            if tax_col:
                for _, row in tdf.iterrows():
                    sid = str(row[id_col]).strip()
                    if sid not in tax_map and pd.notna(row.get(tax_col)):
                        tax_map[sid] = str(row[tax_col]).strip()
        except Exception:
            pass

    # ── Build rows ──
    rows = []
    for sgb in sorted(all_sgbs, key=str):
        rows.append({
            "sgb_id":               sgb,
            "taxonomy":             tax_map.get(sgb, ""),
            # List membership flags (0/1)
            "original_VANISH":      int(sgb in vanish),
            "original_BloSSUM":     int(sgb in blossum),
            "VANISH_plus":          int(sgb in vp_set),
            "BloSSUM_plus":         int(sgb in bp_set),
            "VanishH_plus":         int(sgb in vh_set),
            "BlossumH_plus":        int(sgb in bh_set),
            # Weighted prevalences
            "prev_I_gmw_all":           _fmt(prev_I_all.get(sgb)),
            "prev_NI_gmw_all":          _fmt(prev_NI_all.get(sgb)),
            "prev_IndH_gmw_healthy":    _fmt(prev_I_hlt.get(sgb)),
            # Δp for both contrasts  (positive = enriched in Industrial)
            "delta_p_all_vs_NI":        _fmt(delta_all.get(sgb)),
            "delta_p_healthy_vs_NI":    _fmt(delta_hlt.get(sgb)),
            # CLR difference from depth-matched abundance test
            "clr_diff_abundance":       _fmt(clr_d.get(sgb)),
        })

    df = pd.DataFrame(rows)

    # Sort: VANISH-family first (by original, then plus, then H+), BloSSUM-family last
    df["_v_rank"] = (df["original_VANISH"] * 4
                     + df["VANISH_plus"]    * 2
                     + df["VanishH_plus"]   * 1)
    df["_b_rank"] = (df["original_BloSSUM"] * 4
                     + df["BloSSUM_plus"]    * 2
                     + df["BlossumH_plus"]   * 1)
    df["_family"] = np.where(df["_v_rank"] >= df["_b_rank"], 0, 1)

    df = (df.sort_values(["_family", "_v_rank", "_b_rank", "sgb_id"],
                         ascending=[True, False, False, True])
            .drop(columns=["_v_rank", "_b_rank", "_family"])
            .reset_index(drop=True))
    return df


def _fmt(val):
    """Round to 4 d.p., keeping NaN as NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    return round(float(val), 4)


# ══════════════════════════════════════════════════════════════════════
#  TABLE S5 — Pairwise overlap summary
# ══════════════════════════════════════════════════════════════════════

def build_table_s5(data,
                   vanish_plus_ids, blossum_plus_ids,
                   vanish_h_ids,    blossum_h_ids) -> pd.DataFrame:
    """
    Six rows: three pairwise comparisons for the VANISH family, three for BloSSUM.
    """
    orig_v = sorted(data.vanish_set)
    orig_b = sorted(data.blossum_set)

    rows = [
        # VANISH family
        _overlap_row("orig vs VANISH+",
                     "original VANISH",   orig_v,
                     "VANISH+",           vanish_plus_ids,  "VANISH"),
        _overlap_row("VANISH+ vs VanishH+",
                     "VANISH+",           vanish_plus_ids,
                     "VanishH+",          vanish_h_ids,     "VANISH"),
        _overlap_row("orig vs VanishH+",
                     "original VANISH",   orig_v,
                     "VanishH+",          vanish_h_ids,     "VANISH"),
        # BloSSUM family
        _overlap_row("orig vs BloSSUM+",
                     "original BloSSUM",  orig_b,
                     "BloSSUM+",          blossum_plus_ids, "BloSSUM"),
        _overlap_row("BloSSUM+ vs BlossumH+",
                     "BloSSUM+",          blossum_plus_ids,
                     "BlossumH+",         blossum_h_ids,    "BloSSUM"),
        _overlap_row("orig vs BlossumH+",
                     "original BloSSUM",  orig_b,
                     "BlossumH+",         blossum_h_ids,    "BloSSUM"),
    ]
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
#  FIGURE S5A — Contrast schematic + dual volcano
# ══════════════════════════════════════════════════════════════════════

def _draw_schematic(ax):
    """
    Panel A: three-contrast schematic with group boxes and labelled arrows.
    Drawn entirely with matplotlib patches so it exports cleanly as PDF.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ── helpers ──
    def _box(x, y, w, h, label, sub="", fc="white", ec="black", fs=8):
        patch = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08",
            facecolor=fc, edgecolor=ec, linewidth=1.1, zorder=3)
        ax.add_patch(patch)
        cy = y + h / 2
        if sub:
            ax.text(x + w / 2, cy + 0.18, label,
                    ha="center", va="center", fontsize=fs,
                    fontweight="bold", zorder=4)
            ax.text(x + w / 2, cy - 0.22, sub,
                    ha="center", va="center", fontsize=fs - 1,
                    color="#555555", style="italic", zorder=4)
        else:
            ax.text(x + w / 2, cy, label,
                    ha="center", va="center", fontsize=fs,
                    fontweight="bold", zorder=4)

    def _arrow(x1, y1, x2, y2, color="black", lw=1.1):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color,
                            lw=lw, mutation_scale=11),
            zorder=5)

    def _mid_label(x1, y1, x2, y2, txt, color="black", fs=7.5, offset=(0, 0.18)):
        mx = (x1 + x2) / 2 + offset[0]
        my = (y1 + y2) / 2 + offset[1]
        ax.text(mx, my, txt, ha="center", va="center",
                fontsize=fs, color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
                zorder=6)

    # ── Sample group boxes (left column) ──
    _box(0.25, 8.0,  3.2, 0.9, "Hadza",
         sub="(H-G, Tanzania)", fc="#FCF0F9", ec=OI["pink"])
    _box(0.25, 5.85, 3.2, 0.9, "Non-Industrial",
         sub="(H-G + Transitional)", fc="#FEF6E4", ec=OI["orange"])
    _box(0.25, 3.65, 3.2, 0.9, "Industrial  (all)",
         sub="all health statuses", fc="#EAF4FB", ec=OI["sky"])
    _box(0.25, 1.45, 3.2, 0.9, "Industrial  (healthy)",
         sub="Unhealthy == Healthy", fc="#E6F5F0", ec=OI["green"])

    # ── Result list boxes (right column) ──
    _box(6.55, 8.0,  3.2, 0.9, "Original VANISH",
         sub="& BloSSUM", fc="#F7F7F7", ec="#888888")
    _box(6.55, 5.6,  3.2, 0.9, "VANISH+",
         sub="& BloSSUM+", fc="#FEF6E4", ec=OI["yellow"])
    _box(6.55, 3.3,  3.2, 0.9, "VanishH+",
         sub="& BlossumH+", fc="#E6F5F0", ec=OI["green"])

    # ── Contrast 1: Hadza-centred (legacy) ──
    _arrow(3.45, 8.45, 6.55, 8.45, color="#888888")
    _arrow(3.45, 6.30, 6.55, 8.45, color="#888888")
    _mid_label(3.45, 8.45, 6.55, 8.45,
               "Contrast 1 (legacy)", color="#888888", fs=7.2, offset=(0, 0.28))

    # ── Contrast 2: Ind(all) vs NI → VANISH+/BloSSUM+ ──
    _arrow(3.45, 6.30, 6.55, 6.05, color=OI["yellow"], lw=1.3)
    _arrow(3.45, 4.10, 6.55, 6.05, color=OI["yellow"], lw=1.3)
    _mid_label(3.45, 6.30, 6.55, 6.05,
               "Contrast 2  →  VANISH+ / BloSSUM+",
               color=OI["yellow"], fs=7.2, offset=(0, 0.28))

    # ── Contrast 3: Ind(healthy) vs NI → VanishH+/BlossumH+ ──
    _arrow(3.45, 6.30, 6.55, 3.75, color=OI["green"], lw=1.3)
    _arrow(3.45, 1.90, 6.55, 3.75, color=OI["green"], lw=1.3)
    _mid_label(3.45, 1.90, 6.55, 3.75,
               "Contrast 3  →  VanishH+ / BlossumH+",
               color=OI["green"], fs=7.2, offset=(0.5, 0.28))

    ax.set_title("A   Three contrasts defining the list families",
                 fontsize=current_fs() - 1, fontweight="bold",
                 loc="left", pad=4)


def _volcano_ax(ax, df, x_col, color_fn, xlabel, title,
                sig_thresh=0.05, note=None):
    """
    Generic volcano scatter.  color_fn: microbe_id -> hex_colour_str.
    """
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#888888")
        ax.set_title(title, fontsize=current_fs() - 1, fontweight="bold",
                     loc="left", pad=4)
        return

    df = df.copy()
    df["_col"] = df["microbe"].map(color_fn)

    sig_line = -np.log10(sig_thresh)

    # Draw background taxa first (rasterised for file size)
    bg = df[df["_col"] == C_OTHER]
    ax.scatter(bg[x_col], bg["neg_log10_p_bonf"],
               c=C_OTHER, s=4, alpha=0.25, edgecolors="none",
               rasterized=True, zorder=1)

    # Foreground (list members) — in config draw order
    draw_order_colors = [
        C_VANISH, C_VANISH_P, C_VANISH_H,
        C_BLOSSUM, C_BLOSSUM_P, C_BLOSSUM_H,
        C_SHARED_V, C_SHARED_B,
    ]
    for col in draw_order_colors:
        sub = df[df["_col"] == col]
        if sub.empty:
            continue
        ax.scatter(sub[x_col], sub["neg_log10_p_bonf"],
                   c=col, s=16, alpha=0.80, edgecolors="none", zorder=3)

    ax.axhline(sig_line, ls="--", lw=0.7, color="#999999", alpha=0.7)
    ax.axvline(0, ls="-", lw=0.5, color="#999999", alpha=0.4)
    ax.set_xlabel(xlabel, fontsize=current_fs() - 2)
    ax.set_ylabel("−log₁₀(p)  [Bonferroni]", fontsize=current_fs() - 2)
    ax.set_title(title, fontsize=current_fs() - 1, fontweight="bold",
                 loc="left", pad=4)
    ax.tick_params(labelsize=current_fs() - 3)
    _spines_off(ax)

    if note:
        ax.text(0.97, 0.97, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=7, style="italic",
                color="#666666")


def figure_s5a(data,
               vanish_plus_ids, blossum_plus_ids,
               vanish_h_ids,    blossum_h_ids,
               abund_df=None,
               out_dir=None):
    """
    Figure S5A — three-panel figure.

    Panel A : schematic of the three contrasts
    Panel B : weighted-prevalence volcano  (Ind-all vs NI)
    Panel C : CLR-abundance volcano        (depth-matched Ind vs NI; descriptive)
    """
    ref = data.ref_name
    if out_dir is None:
        out_dir = config.section_dir(ref, "s2", "supplement")

    vp_set = set(vanish_plus_ids);  vh_set = set(vanish_h_ids)
    bp_set = set(blossum_plus_ids); bh_set = set(blossum_h_ids)
    orig_v = data.vanish_set;       orig_b = data.blossum_set

    # ── Colour function (7-category aligned with pipeline) ──
    def _color_7cat(sgb):
        cat = assign_volcano_category_7(sgb, orig_v, orig_b, vp_set, bp_set)
        # Map category strings -> palette colours
        mapping = {
            "VANISH only":        C_VANISH,
            "VANISH+ only":       C_VANISH_P,
            "VANISH ∩ VANISH+":   C_SHARED_V,
            "BloSSUM only":       C_BLOSSUM,
            "BloSSUM+ only":      C_BLOSSUM_P,
            "BloSSUM ∩ BloSSUM+": C_SHARED_B,
        }
        color = mapping.get(cat, C_OTHER)
        # VanishH+ / BlossumH+ override for taxa that also appear there
        if color == C_OTHER:
            if sgb in vh_set:
                return C_VANISH_H
            if sgb in bh_set:
                return C_BLOSSUM_H
        return color

    gmw_all = _load_gmw_csv(data, "ind_all_vs_ni")
    if abund_df is None:
        abund_df = _load_abund_csv(data)

    # Authored at the printed width. A figure drawn four times too wide is
    # reduced to fit the page, and the reduction takes every line and
    # symbol with it -- a 1 pt rule landed at 0.25 pt, under the 0.5 pt
    # floor, and the type had to be inflated to stay legible.
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, PAGE_WIDTH_IN * 5.2 / 15))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.36, left=0.05,
                            right=0.97, top=0.90, bottom=0.14)

    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # Panel A — schematic
    _draw_schematic(ax_a)

    # Panel B — weighted prevalence volcano
    _volcano_ax(
        ax_b, gmw_all, "prevalence_diff", _color_7cat,
        xlabel="Weighted Δp  (Industrial − Non-Industrial)",
        title="B   Prevalence volcano  (GMW)\nInd(all) vs Non-Industrial")

    # Panel C — CLR abundance volcano (descriptive label)
    _volcano_ax(
        ax_c, abund_df, "clr_diff", _color_7cat,
        xlabel="CLR difference  (Industrial − Non-Industrial)",
        title="C   CLR abundance (depth-matched)\n– descriptive; not used for list assignment –",
        note="abundance ≠ list-defining criterion")

    # ── Shared legend ──
    legend_items = [
        mpatches.Patch(fc=C_VANISH,    label="Original VANISH"),
        mpatches.Patch(fc=C_VANISH_P,  label="VANISH+ only"),
        mpatches.Patch(fc=C_SHARED_V,  label="Orig ∩ VANISH+"),
        mpatches.Patch(fc=C_VANISH_H,  label="VanishH+ only"),
        mpatches.Patch(fc=C_BLOSSUM,   label="Original BloSSUM"),
        mpatches.Patch(fc=C_BLOSSUM_P, label="BloSSUM+ only"),
        mpatches.Patch(fc=C_SHARED_B,  label="Orig ∩ BloSSUM+"),
        mpatches.Patch(fc=C_BLOSSUM_H, label="BlossumH+ only"),
        mpatches.Patch(fc=C_OTHER,     label="Other (not in any list)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=5,
               fontsize=7.5, framealpha=0.9, edgecolor="#CCCCCC",
               handlelength=1.2, handletextpad=0.5, columnspacing=1.0,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"Figure S5A — Ranking statistic derivation and list contrasts  [{ref}]",
        fontsize=current_fs(), fontweight="bold", y=0.98)

    stem = os.path.join(out_dir, "fig_s5a_list_derivation")
    savefig_dual(fig, stem, dpi=300)
    plt.close(fig)
    print(f"    [supp_s5] Fig S5A -> {stem}.png")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE S5B — Overlap bars / prevalence scatter / signal strength
# ══════════════════════════════════════════════════════════════════════

def _stacked_overlap_bar(ax, entries, title):
    """
    Horizontal stacked bars.

    entries : list of dicts with keys:
        label, n_A_only, n_shared, n_B_only, col_A, col_B
    """
    n = len(entries)
    ys = np.arange(n)[::-1]   # top-to-bottom reading order

    for y, e in zip(ys, entries):
        total = e["n_A_only"] + e["n_shared"] + e["n_B_only"]
        if total == 0:
            continue
        frac_a = e["n_A_only"] / total
        frac_s = e["n_shared"]  / total
        frac_b = e["n_B_only"]  / total

        ax.barh(y, frac_a, 0.55, color=e["col_A"],       edgecolor="white", lw=0.4)
        ax.barh(y, frac_s, 0.55, left=frac_a,
                color=e["col_shared"], edgecolor="white", lw=0.4)
        ax.barh(y, frac_b, 0.55, left=frac_a + frac_s,
                color=e["col_B"],       edgecolor="white", lw=0.4)

        # Inside the axes, in a strip reserved to the left of the bars. Drawn
        # outside the axes it sat on the tick label beside it; the bars start
        # at zero, so the room has to be made rather than found.
        ax.text(-0.015, y, f"n={total}", ha="right", va="center",
                fontsize=7, color="#444444")
        # fraction annotations inside bars (only if wide enough)
        for frac, left, col in [
            (frac_a, 0,              e["col_A"]),
            (frac_s, frac_a,         e["col_shared"]),
            (frac_b, frac_a+frac_s,  e["col_B"]),
        ]:
            if frac > 0.07:
                ax.text(left + frac / 2, y, f"{frac:.0%}",
                        ha="center", va="center", fontsize=6.5,
                        color="white", fontweight="bold")

    labels = [e["label"] for e in entries]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    # Wide enough for the longest count; the strip left of zero is
    # where the sample numbers live.
    ax.set_xlim(-0.30, 1.04)
    ax.set_xlabel("Fraction of union membership", fontsize=8)
    ax.set_title(title, fontsize=current_fs() - 1, fontweight="bold",
                 loc="left", pad=4)
    ax.tick_params(axis="x", labelsize=7)
    _spines_off(ax)


def figure_s5b(data,
               vanish_plus_ids, blossum_plus_ids,
               vanish_h_ids,    blossum_h_ids,
               out_dir=None):
    """
    Figure S5B — three-panel figure.

    Panel A : Stacked overlap bars (VANISH + BloSSUM families)
    Panel B : p_I vs p_NI scatter coloured by list-membership subgroup
    Panel C : Median weighted prevalence per list-membership subgroup
    """
    ref = data.ref_name
    if out_dir is None:
        out_dir = config.section_dir(ref, "s2", "supplement")

    orig_v  = data.vanish_set;   orig_b  = data.blossum_set
    vp_set  = set(vanish_plus_ids);  bp_set  = set(blossum_plus_ids)
    vh_set  = set(vanish_h_ids);     bh_set  = set(blossum_h_ids)

    gmw_all = _load_gmw_csv(data, "ind_all_vs_ni")

    from . import figure_source
    import pandas as _pd
    # Panel A's bars are overlaps against the original curated lists, so
    # those are columns too: without them the stacked bars cannot be
    # recounted from the table.
    _lists = {"VANISH": orig_v, "BloSSUM": orig_b,
              "VANISH+": vanish_plus_ids, "BloSSUM+": blossum_plus_ids,
              "VanishH+": vanish_h_ids, "BlossumH+": blossum_h_ids}
    _all = sorted({t for v in _lists.values() if v for t in v})
    if not _all:
        return None
    _mem = _pd.DataFrame({"taxon": _all})
    for _name, _ids_ in _lists.items():
        _mem[_name] = _mem["taxon"].isin(set(_ids_ or []))
    _tables = {"list_membership": _mem}
    # Panels B and C plot GMW-weighted prevalences, which membership alone
    # does not carry: B is p_Industrial against p_Non-Industrial per taxon,
    # and C is the median of the same p_NI within each list.
    if gmw_all is not None and not gmw_all.empty:
        _cols = [c for c in ("microbe", "prev_g1", "prev_g2")
                 if c in gmw_all.columns]
        if len(_cols) == 3:
            _tables["gmw_prevalence"] = gmw_all[_cols].copy()
    figure_source.emit_for_path(
        os.path.join(out_dir, "fig_s5b_overlap_analysis.png"), _tables,
        reference=ref)
    return plot_s5b(_tables, ref, out_dir)


def plot_s5b(tables, ref="combined", out_dir="."):
    """Draw Figure S5B from its table, and from nothing else.

    Panel A counts overlaps between the curated lists and the derived
    ones; panels B and C plot the GMW-weighted prevalences. Both come out
    of the two published tables, so the membership columns are read back
    into the sets the drawing code expects.
    """
    _mem = tables["list_membership"]

    def _col(name):
        return (set(_mem.loc[_mem[name].astype(bool), "taxon"])
                if name in _mem.columns else set())

    orig_v, orig_b = _col("VANISH"), _col("BloSSUM")
    vp_set, bp_set = _col("VANISH+"), _col("BloSSUM+")
    vh_set, bh_set = _col("VanishH+"), _col("BlossumH+")
    gmw_all = tables.get("gmw_prevalence")

    # Authored at the printed width. A figure drawn four times too wide is
    # reduced to fit the page, and the reduction takes every line and
    # symbol with it -- a 1 pt rule landed at 0.25 pt, under the 0.5 pt
    # floor, and the type had to be inflated to stay legible.
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, PAGE_WIDTH_IN * 5.5 / 15))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42, left=0.07,
                            right=0.97, top=0.88, bottom=0.13)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # ── Panel A: Overlap stacked bars ──
    entries = [
        # VANISH family
        dict(label="Orig VANISH\nvs VANISH+",
             n_A_only  = len(orig_v - vp_set),
             n_shared  = len(orig_v & vp_set),
             n_B_only  = len(vp_set - orig_v),
             col_A     = C_VANISH,
             col_shared= C_SHARED_V,
             col_B     = C_VANISH_P),
        dict(label="VANISH+\nvs VanishH+",
             n_A_only  = len(vp_set - vh_set),
             n_shared  = len(vp_set & vh_set),
             n_B_only  = len(vh_set - vp_set),
             col_A     = C_VANISH_P,
             col_shared= "#B87333",          # copper — shared between + and H+
             col_B     = C_VANISH_H),
        dict(label="Orig VANISH\nvs VanishH+",
             n_A_only  = len(orig_v - vh_set),
             n_shared  = len(orig_v & vh_set),
             n_B_only  = len(vh_set - orig_v),
             col_A     = C_VANISH,
             col_shared= C_SHARED_V,
             col_B     = C_VANISH_H),
        # BloSSUM family
        dict(label="Orig BloSSUM\nvs BloSSUM+",
             n_A_only  = len(orig_b - bp_set),
             n_shared  = len(orig_b & bp_set),
             n_B_only  = len(bp_set - orig_b),
             col_A     = C_BLOSSUM,
             col_shared= C_SHARED_B,
             col_B     = C_BLOSSUM_P),
        dict(label="BloSSUM+\nvs BlossumH+",
             n_A_only  = len(bp_set - bh_set),
             n_shared  = len(bp_set & bh_set),
             n_B_only  = len(bh_set - bp_set),
             col_A     = C_BLOSSUM_P,
             col_shared= "#006D5B",          # teal — shared between + and H+
             col_B     = C_BLOSSUM_H),
        dict(label="Orig BloSSUM\nvs BlossumH+",
             n_A_only  = len(orig_b - bh_set),
             n_shared  = len(orig_b & bh_set),
             n_B_only  = len(bh_set - orig_b),
             col_A     = C_BLOSSUM,
             col_shared= C_SHARED_B,
             col_B     = C_BLOSSUM_H),
    ]
    # Panel letters only; what each panel shows is in the caption.
    _stacked_overlap_bar(ax_a, entries, title="A")

    # Legend for Panel A
    leg_patches = [
        mpatches.Patch(fc=C_VANISH,    label="Original VANISH"),
        mpatches.Patch(fc=C_VANISH_P,  label="VANISH+"),
        mpatches.Patch(fc=C_VANISH_H,  label="VanishH+"),
        mpatches.Patch(fc=C_BLOSSUM,   label="Original BloSSUM"),
        mpatches.Patch(fc=C_BLOSSUM_P, label="BloSSUM+"),
        mpatches.Patch(fc=C_BLOSSUM_H, label="BlossumH+"),
    ]
    # Below its own panel, not inside it: the bars run the full width at
    # every row, so there is no clear corner. Two entries per line keeps the
    # key no wider than the panel it explains (three per line ran into panel
    # B's key); the VANISH family fills the left column, BloSSUM the right.
    # _place_keys centres each key under its panel.
    _leg_a = fig.legend(handles=leg_patches, loc="upper center", ncol=2,
                        frameon=False, handlelength=1.2, columnspacing=1.0,
                        handletextpad=0.4, labelspacing=0.3)
    _leg_b = None

    # ── Panel B: p_I vs p_NI scatter ──
    if gmw_all is not None and not gmw_all.empty:
        # Define six VANISH-side subgroups for scatter colouring
        def _scatter_group(sgb):
            in_v  = sgb in orig_v
            in_vp = sgb in vp_set
            in_vh = sgb in vh_set
            if in_v and in_vp:   return "Orig ∩ VANISH+", C_SHARED_V, 4
            if in_v:             return "Orig VANISH only", C_VANISH,  3
            if in_vp and in_vh:  return "VANISH+ ∩ H+",   C_VANISH_H, 3
            if in_vp:            return "VANISH+ only",    C_VANISH_P, 2
            if in_vh:            return "VanishH+ only",   C_VANISH_H, 2
            in_b  = sgb in orig_b
            in_bp = sgb in bp_set
            if in_b and in_bp:   return "Orig ∩ BloSSUM+", C_SHARED_B, 4
            if in_b:             return "Orig BloSSUM only", C_BLOSSUM, 3
            if in_bp:            return "BloSSUM+ only",    C_BLOSSUM_P,2
            in_bh = sgb in bh_set
            if in_bh:            return "BlossumH+ only",   C_BLOSSUM_H,2
            return "Other", C_OTHER, 0

        gmw_all = gmw_all.copy()
        gmw_all[["_grp","_col","_z"]] = pd.DataFrame(
            [_scatter_group(m) for m in gmw_all["microbe"]],
            columns=["_grp","_col","_z"])

        for z in sorted(gmw_all["_z"].unique()):
            sub = gmw_all[gmw_all["_z"] == z]
            for col in sub["_col"].unique():
                subsub = sub[sub["_col"] == col]
                grp_label = subsub["_grp"].iloc[0]
                ax_b.scatter(
                    subsub["prev_g1"], subsub["prev_g2"],
                    c=col,
                    s=5 if z == 0 else 18,
                    alpha=0.20 if z == 0 else 0.75,
                    edgecolors="none",
                    label=grp_label if z > 0 else "_nolegend_",
                    rasterized=(z == 0),
                    zorder=z + 1)

        # Diagonal reference line
        ax_b.plot([0, 1], [0, 1], ls="--", lw=0.7, color="#999999", alpha=0.6)

        # No quadrant labels: the lower-right one sat on the BloSSUM-side
        # points. Above the diagonal is NI-enriched (VANISH-like), below it
        # Industrial-enriched (BloSSUM-like); the caption says so.

        ax_b.set_xlabel("Prevalence in Industrial  (GMW-weighted)", fontsize=8)
        ax_b.set_ylabel("Prevalence in Non-Industrial  (GMW-weighted)", fontsize=8)
        ax_b.set_title("B", fontsize=current_fs() - 1, fontweight="bold",
                       loc="left", pad=4)
        ax_b.tick_params(labelsize=7)
        # Below its own panel too: this scatter fills its axes corner to
        # corner, so the key covers points wherever it is put inside. Two
        # entries per line, like panel A's, so it stays within the panel.
        _hb, _lb = ax_b.get_legend_handles_labels()
        # Same arrangement as panel A's key: VANISH family down the left
        # column, BloSSUM family down the right, each in panel C's order.
        _key_order = ["Orig VANISH only", "Orig ∩ VANISH+", "VANISH+ only",
                      "VANISH+ ∩ H+", "VanishH+ only",
                      "Orig BloSSUM only", "Orig ∩ BloSSUM+", "BloSSUM+ only",
                      "BlossumH+ only"]
        _rank = {lab: i for i, lab in enumerate(_key_order)}
        _pairs = sorted(zip(_hb, _lb), key=lambda hl: _rank.get(hl[1], len(_rank)))
        _hb, _lb = [h for h, _ in _pairs], [l for _, l in _pairs]
        # Legends fill column by column, so pad the shorter family with blank
        # entries to keep each family in its own column.
        _n_v = sum(1 for l in _lb if _rank.get(l, 99) < 5)
        _cols = [list(zip(_hb, _lb))[:_n_v], list(zip(_hb, _lb))[_n_v:]]
        _m = max(len(c) for c in _cols)
        _blank = (mpatches.Patch(alpha=0), "")
        _pairs = [hl for c in _cols for hl in c + [_blank] * (_m - len(c))]
        _hb, _lb = [h for h, _ in _pairs], [l for _, l in _pairs]
        _leg_b = fig.legend(_hb, _lb, loc="upper center", ncol=2,
                            frameon=False, markerscale=1.3,
                            handletextpad=0.35, columnspacing=0.9,
                            labelspacing=0.3)
        _spines_off(ax_b)
    else:
        ax_b.text(0.5, 0.5, "GMW prevalence data not available",
                  ha="center", va="center", transform=ax_b.transAxes,
                  fontsize=9, color="#888888")

    # ── Panel C: Median weighted NI prevalence per subgroup ──
    if gmw_all is not None and not gmw_all.empty:
        prev_ni = dict(zip(gmw_all["microbe"], gmw_all["prev_g2"]))

        def _med(ids):
            return _median_nonan([prev_ni.get(s, np.nan) for s in ids])

        # Six VANISH-side subgroups
        subgroups_v = [
            ("Orig VANISH only",   sorted(orig_v - vp_set),      C_VANISH),
            ("Orig ∩ VANISH+",     sorted(orig_v & vp_set),      C_SHARED_V),
            ("VANISH+ only",       sorted(vp_set - orig_v),      C_VANISH_P),
            ("VanishH+ only",      sorted(vh_set - vp_set - orig_v), C_VANISH_H),
        ]
        subgroups_b = [
            ("Orig BloSSUM only",  sorted(orig_b - bp_set),      C_BLOSSUM),
            ("Orig ∩ BloSSUM+",    sorted(orig_b & bp_set),      C_SHARED_B),
            ("BloSSUM+ only",      sorted(bp_set - orig_b),      C_BLOSSUM_P),
            ("BlossumH+ only",     sorted(bh_set - bp_set - orig_b), C_BLOSSUM_H),
        ]

        all_groups = subgroups_v + [("", [], "#FFFFFF")] + subgroups_b
        bar_labels, bar_meds, bar_cols, bar_ns = [], [], [], []
        for lbl, ids, col in all_groups:
            if not lbl:          # spacer
                bar_labels.append(""); bar_meds.append(0)
                bar_cols.append("#FFFFFF"); bar_ns.append(0)
                continue
            m = _med(ids)
            bar_labels.append(lbl)
            bar_meds.append(m if not np.isnan(m) else 0)
            bar_cols.append(col)
            bar_ns.append(len(ids))

        ys_c = np.arange(len(bar_labels))[::-1]
        bars_c = ax_c.barh(ys_c, bar_meds, 0.58,
                            color=bar_cols, edgecolor="white", lw=0.5)

        for y, m, n, lbl in zip(ys_c, bar_meds, bar_ns, bar_labels):
            if not lbl:
                continue
            if m > 0:
                ax_c.text(m + 0.005, y, f"{m:.2f}",
                          va="center", fontsize=7, color="#333333")

        ax_c.set_yticks(ys_c)
        ax_c.set_yticklabels(bar_labels, fontsize=7.5)
        # Same treatment as panel A: a strip inside the axes, left of the
        # bars, so the counts sit beside their rows without touching either
        # the tick labels or the bars.
        _xmax = max([m for m in bar_meds if np.isfinite(m)] or [1.0])
        ax_c.set_xlim(-0.22 * _xmax, _xmax * 1.16)
        for _y, _n, _lbl in zip(ys_c, bar_ns, bar_labels):
            if _lbl:
                ax_c.text(-0.02 * _xmax, _y, f"n={_n}", ha="right",
                          va="center", fontsize=6.8, color="#444444")
        ax_c.set_xlabel("Median GMW-weighted prevalence in NI", fontsize=8)
        ax_c.set_title("C", fontsize=current_fs() - 1, fontweight="bold",
                       loc="left", pad=4)
        ax_c.tick_params(labelsize=7)
        _spines_off(ax_c)

        # VANISH / BloSSUM section labels
        n_v = len(subgroups_v) + 1  # + spacer
        ax_c.axhline(len(ys_c) - len(subgroups_v) - 0.6,
                     ls=":", lw=0.7, color="#AAAAAA")
        ax_c.text(ax_c.get_xlim()[1] * 0.98,
                  ys_c[0] + 0.35, "VANISH family",
                  ha="right", va="bottom", fontsize=7,
                  color=C_VANISH, fontweight="bold")
        ax_c.text(ax_c.get_xlim()[1] * 0.98,
                  ys_c[len(subgroups_v) + 1] - 0.35, "BloSSUM family",
                  ha="right", va="top", fontsize=7,
                  color=C_BLOSSUM, fontweight="bold")
    else:
        ax_c.text(0.5, 0.5, "GMW prevalence data not available",
                  ha="center", va="center", transform=ax_c.transAxes,
                  fontsize=9, color="#888888")

    fig.suptitle(
        f"Figure S5B — Overlap, directional consistency, and signal strength  [{ref}]",
        fontsize=current_fs(), fontweight="bold", y=0.97)

    def _place_keys():
        """Put each key directly under its own panel, below its axis label.

        Measured from the drawn text, so it runs again once the fitting
        passes have resized that text; the tight save then extends the
        canvas down to take them. Both keys hang from the same line (the
        lowest axis label) so they read as one row.
        """
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        y = min(inv.transform((0, ax.get_tightbbox(r).y0))[1]
                for ax in (ax_a, ax_b, ax_c))
        for ax, leg in ((ax_a, _leg_a), (ax_b, _leg_b)):
            if leg is None:
                continue
            pos = ax.get_position()
            leg.set_bbox_to_anchor((pos.x0 + pos.width / 2, y - 0.02),
                                   transform=fig.transFigure)

    _place_keys()
    add_post_layout_hook(fig, _place_keys)

    stem = os.path.join(out_dir, "fig_s5b_overlap_analysis")
    savefig_dual(fig, stem, dpi=300)
    plt.close(fig)
    print(f"    [supp_s5] Fig S5A -> {stem}.png")
    return stem + ".png"


    # Authored at the printed width. A figure drawn four times too wide is
    # reduced to fit the page, and the reduction takes every line and
    # symbol with it -- a 1 pt rule landed at 0.25 pt, under the 0.5 pt
    # floor, and the type had to be inflated to stay legible.
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, PAGE_WIDTH_IN * 5.5 / 15))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42, left=0.07,
                            right=0.97, top=0.88, bottom=0.13)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    # ── Panel A: Overlap stacked bars ──
    entries = [
        # VANISH family
        dict(label="Orig VANISH\nvs VANISH+",
             n_A_only  = len(orig_v - vp_set),
             n_shared  = len(orig_v & vp_set),
             n_B_only  = len(vp_set - orig_v),
             col_A     = C_VANISH,
             col_shared= C_SHARED_V,
             col_B     = C_VANISH_P),
        dict(label="VANISH+\nvs VanishH+",
             n_A_only  = len(vp_set - vh_set),
             n_shared  = len(vp_set & vh_set),
             n_B_only  = len(vh_set - vp_set),
             col_A     = C_VANISH_P,
             col_shared= "#B87333",          # copper — shared between + and H+
             col_B     = C_VANISH_H),
        dict(label="Orig VANISH\nvs VanishH+",
             n_A_only  = len(orig_v - vh_set),
             n_shared  = len(orig_v & vh_set),
             n_B_only  = len(vh_set - orig_v),
             col_A     = C_VANISH,
             col_shared= C_SHARED_V,
             col_B     = C_VANISH_H),
        # BloSSUM family
        dict(label="Orig BloSSUM\nvs BloSSUM+",
             n_A_only  = len(orig_b - bp_set),
             n_shared  = len(orig_b & bp_set),
             n_B_only  = len(bp_set - orig_b),
             col_A     = C_BLOSSUM,
             col_shared= C_SHARED_B,
             col_B     = C_BLOSSUM_P),
        dict(label="BloSSUM+\nvs BlossumH+",
             n_A_only  = len(bp_set - bh_set),
             n_shared  = len(bp_set & bh_set),
             n_B_only  = len(bh_set - bp_set),
             col_A     = C_BLOSSUM_P,
             col_shared= "#006D5B",          # teal — shared between + and H+
             col_B     = C_BLOSSUM_H),
        dict(label="Orig BloSSUM\nvs BlossumH+",
             n_A_only  = len(orig_b - bh_set),
             n_shared  = len(orig_b & bh_set),
             n_B_only  = len(bh_set - orig_b),
             col_A     = C_BLOSSUM,
             col_shared= C_SHARED_B,
             col_B     = C_BLOSSUM_H),
    ]
    _stacked_overlap_bar(
        ax_a, entries,
        title="A   Overlap across list generations\n(VANISH family top, BloSSUM family bottom)")

    # Legend for Panel A
    leg_patches = [
        mpatches.Patch(fc=C_VANISH,    label="Original VANISH"),
        mpatches.Patch(fc=C_VANISH_P,  label="VANISH+"),
        mpatches.Patch(fc=C_VANISH_H,  label="VanishH+"),
        mpatches.Patch(fc=C_BLOSSUM,   label="Original BloSSUM"),
        mpatches.Patch(fc=C_BLOSSUM_P, label="BloSSUM+"),
        mpatches.Patch(fc=C_BLOSSUM_H, label="BlossumH+"),
    ]
    ax_a.legend(handles=leg_patches, fontsize=6.5, loc="lower right",
                framealpha=0.85, edgecolor="#CCCCCC", ncol=2)

    # ── Panel B: p_I vs p_NI scatter ──
    if gmw_all is not None and not gmw_all.empty:
        # Define six VANISH-side subgroups for scatter colouring
        def _scatter_group(sgb):
            in_v  = sgb in orig_v
            in_vp = sgb in vp_set
            in_vh = sgb in vh_set
            if in_v and in_vp:   return "Orig ∩ VANISH+", C_SHARED_V, 4
            if in_v:             return "Orig only",       C_VANISH,   3
            if in_vp and in_vh:  return "VANISH+ ∩ H+",   C_VANISH_H, 3
            if in_vp:            return "VANISH+ only",    C_VANISH_P, 2
            if in_vh:            return "VanishH+ only",   C_VANISH_H, 2
            in_b  = sgb in orig_b
            in_bp = sgb in bp_set
            if in_b and in_bp:   return "Orig ∩ BloSSUM+", C_SHARED_B, 4
            if in_b:             return "BloSSUM orig",     C_BLOSSUM,  3
            if in_bp:            return "BloSSUM+ only",    C_BLOSSUM_P,2
            in_bh = sgb in bh_set
            if in_bh:            return "BlossumH+ only",   C_BLOSSUM_H,2
            return "Other", C_OTHER, 0

        gmw_all = gmw_all.copy()
        gmw_all[["_grp","_col","_z"]] = pd.DataFrame(
            [_scatter_group(m) for m in gmw_all["microbe"]],
            columns=["_grp","_col","_z"])

        for z in sorted(gmw_all["_z"].unique()):
            sub = gmw_all[gmw_all["_z"] == z]
            for col in sub["_col"].unique():
                subsub = sub[sub["_col"] == col]
                grp_label = subsub["_grp"].iloc[0]
                ax_b.scatter(
                    subsub["prev_g1"], subsub["prev_g2"],
                    c=col,
                    s=5 if z == 0 else 18,
                    alpha=0.20 if z == 0 else 0.75,
                    edgecolors="none",
                    label=grp_label if z > 0 else "_nolegend_",
                    rasterized=(z == 0),
                    zorder=z + 1)

        # Diagonal reference line
        ax_b.plot([0, 1], [0, 1], ls="--", lw=0.7, color="#999999", alpha=0.6)

        # Annotate quadrants
        ax_b.text(0.02, 0.96, "Enriched in NI\n→ VANISH-like",
                  transform=ax_b.transAxes, fontsize=7, va="top",
                  color=C_VANISH, style="italic")
        ax_b.text(0.96, 0.06, "Enriched in Industrial\n→ BloSSUM-like",
                  transform=ax_b.transAxes, fontsize=7, ha="right",
                  color=C_BLOSSUM, style="italic")

        ax_b.set_xlabel("Prevalence in Industrial  (GMW-weighted)", fontsize=8)
        ax_b.set_ylabel("Prevalence in Non-Industrial  (GMW-weighted)", fontsize=8)
        ax_b.set_title("B   Prevalence scatter\np_I vs p_NI  (Ind-all contrast)",
                        fontsize=current_fs() - 1, fontweight="bold",
                        loc="left", pad=4)
        ax_b.tick_params(labelsize=7)
        ax_b.legend(fontsize=6.5, loc="upper right", framealpha=0.85,
                    edgecolor="#CCCCCC", markerscale=1.6,
                    handletextpad=0.4, labelspacing=0.3)
        _spines_off(ax_b)
    else:
        ax_b.text(0.5, 0.5, "GMW prevalence data not available",
                  ha="center", va="center", transform=ax_b.transAxes,
                  fontsize=9, color="#888888")

    # ── Panel C: Median weighted NI prevalence per subgroup ──
    if gmw_all is not None and not gmw_all.empty:
        prev_ni = dict(zip(gmw_all["microbe"], gmw_all["prev_g2"]))

        def _med(ids):
            return _median_nonan([prev_ni.get(s, np.nan) for s in ids])

        # Six VANISH-side subgroups
        subgroups_v = [
            ("Orig VANISH only",   sorted(orig_v - vp_set),      C_VANISH),
            ("Orig ∩ VANISH+",     sorted(orig_v & vp_set),      C_SHARED_V),
            ("VANISH+ only",       sorted(vp_set - orig_v),      C_VANISH_P),
            ("VanishH+ only",      sorted(vh_set - vp_set - orig_v), C_VANISH_H),
        ]
        subgroups_b = [
            ("Orig BloSSUM only",  sorted(orig_b - bp_set),      C_BLOSSUM),
            ("Orig ∩ BloSSUM+",    sorted(orig_b & bp_set),      C_SHARED_B),
            ("BloSSUM+ only",      sorted(bp_set - orig_b),      C_BLOSSUM_P),
            ("BlossumH+ only",     sorted(bh_set - bp_set - orig_b), C_BLOSSUM_H),
        ]

        all_groups = subgroups_v + [("", [], "#FFFFFF")] + subgroups_b
        bar_labels, bar_meds, bar_cols, bar_ns = [], [], [], []
        for lbl, ids, col in all_groups:
            if not lbl:          # spacer
                bar_labels.append(""); bar_meds.append(0)
                bar_cols.append("#FFFFFF"); bar_ns.append(0)
                continue
            m = _med(ids)
            bar_labels.append(lbl)
            bar_meds.append(m if not np.isnan(m) else 0)
            bar_cols.append(col)
            bar_ns.append(len(ids))

        ys_c = np.arange(len(bar_labels))[::-1]
        bars_c = ax_c.barh(ys_c, bar_meds, 0.58,
                            color=bar_cols, edgecolor="white", lw=0.5)

        for y, m, n, lbl in zip(ys_c, bar_meds, bar_ns, bar_labels):
            if not lbl:
                continue
            if m > 0:
                ax_c.text(m + 0.005, y, f"{m:.2f}",
                          va="center", fontsize=7, color="#333333")
            ax_c.text(-0.005, y, f"(n={n})",
                      ha="right", va="center", fontsize=6.5, color="#666666")

        ax_c.set_yticks(ys_c)
        ax_c.set_yticklabels(bar_labels, fontsize=7.5)
        ax_c.set_xlabel("Median GMW-weighted prevalence in NI", fontsize=8)
        ax_c.set_title(
            "C   Signal strength by subgroup\n"
            "Overlap taxa show the strongest shared signal",
            fontsize=current_fs() - 1, fontweight="bold",
            loc="left", pad=4)
        ax_c.tick_params(labelsize=7)
        _spines_off(ax_c)

        # VANISH / BloSSUM section labels
        n_v = len(subgroups_v) + 1  # + spacer
        ax_c.axhline(len(ys_c) - len(subgroups_v) - 0.6,
                     ls=":", lw=0.7, color="#AAAAAA")
        ax_c.text(ax_c.get_xlim()[1] * 0.98,
                  ys_c[0] + 0.35, "VANISH family",
                  ha="right", va="bottom", fontsize=7,
                  color=C_VANISH, fontweight="bold")
        ax_c.text(ax_c.get_xlim()[1] * 0.98,
                  ys_c[len(subgroups_v) + 1] - 0.35, "BloSSUM family",
                  ha="right", va="top", fontsize=7,
                  color=C_BLOSSUM, fontweight="bold")
    else:
        ax_c.text(0.5, 0.5, "GMW prevalence data not available",
                  ha="center", va="center", transform=ax_c.transAxes,
                  fontsize=9, color="#888888")

    fig.suptitle(
        f"Figure S5B — Overlap, directional consistency, and signal strength  [{ref}]",
        fontsize=current_fs(), fontweight="bold", y=0.97)

    stem = os.path.join(out_dir, "fig_s5b_overlap_analysis")
    # The figure's data object: list membership per taxon. Every panel is a
    # view of it -- the stacked overlap bars are counts of the membership
    # combinations, and the scatter and the median bars are prevalences of the
    # same taxa -- so one membership table reproduces all three.
    from . import figure_source
    import pandas as _pd
    # Panel A's bars are overlaps *against the original curated lists*, so
    # those have to be columns too: without them the stacked bars cannot be
    # recounted from the table.
    # The sets this function reconstructed from the table, not the `*_ids`
    # parameters of `figure_s5b` -- this block was copied from there, and the
    # four names do not exist here, so the redraw raised NameError at the emit.
    _lists = {"VANISH": orig_v, "BloSSUM": orig_b,
              "VANISH+": vp_set, "BloSSUM+": bp_set,
              "VanishH+": vh_set, "BlossumH+": bh_set}
    _all = sorted({t for v in _lists.values() if v for t in v})
    if _all:
        _mem = _pd.DataFrame({"taxon": _all})
        for _name, _ids in _lists.items():
            _mem[_name] = _mem["taxon"].isin(set(_ids or []))
        _tables = {"list_membership": _mem}
        # Panels B and C plot GMW-weighted prevalences, which membership alone
        # does not carry: B is p_Industrial against p_Non-Industrial per taxon,
        # and C is the median of the same p_NI within each list. Without these
        # columns the data object describes only panel A.
        if gmw_all is not None and not gmw_all.empty:
            _cols = [c for c in ("microbe", "prev_g1", "prev_g2")
                     if c in gmw_all.columns]
            if len(_cols) == 3:
                _tables["gmw_prevalence"] = gmw_all[_cols].copy()
        figure_source.emit_for_path(stem + ".png", _tables, reference=ref)
    savefig_dual(fig, stem, dpi=300)
    plt.close(fig)
    print(f"    [supp_s5] Fig S5B -> {stem}.png")


# ══════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_supp_s5(data,
                vanish_plus_ids, blossum_plus_ids,
                vanish_h_ids,    blossum_h_ids,
                abund_df=None,
                taxonomy_df=None,
                out_dir=None):
    """
    Generate Tables S4 / S5 and Figures S5A / S5B.

    Parameters
    ----------
    data               : ReferenceData
    vanish_plus_ids    : list[str]  — from run_differential_gmw pass 2 (NI-enriched)
    blossum_plus_ids   : list[str]  — from run_differential_gmw pass 2 (Ind-enriched)
    vanish_h_ids       : list[str]  — from run_differential_gmw pass 3 (NI-enriched)
    blossum_h_ids      : list[str]  — from run_differential_gmw pass 3 (IndH-enriched)
    abund_df           : DataFrame, optional
        CLR abundance result from run_differential().  Loaded from disk
        (abundance_ind_vs_nonind.csv) if not supplied.
    taxonomy_df        : DataFrame, optional
        GTDB taxonomy table (e.g. Hadza_UHGG_SGB_matching.csv).
    out_dir            : str, optional
        Output directory.  Defaults to
        config.section_dir(ref, "s2", "supplement").
    """
    ref = data.ref_name
    if out_dir is None:
        out_dir = config.section_dir(ref, "s2", "supplement")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  [supp_s5]  Supplementary tables & figures S4/S5  [{ref}]")
    print(f"             Output -> {out_dir}")

    # ── Table S4 ──
    print("\n    Building Table S4 (list membership)...")
    t4 = build_table_s4(data,
                        vanish_plus_ids, blossum_plus_ids,
                        vanish_h_ids,    blossum_h_ids,
                        taxonomy_df=taxonomy_df)
    if not t4.empty:
        p4 = os.path.join(out_dir, "table_s4_list_membership.csv")
        t4.to_csv(p4, index=False)
        print(f"    Table S4: {len(t4)} SGBs  ->  {p4}")
    else:
        print("    Table S4: empty — check that list IDs are populated")

    # ── Table S5 ──
    # Careful: this is the SUPPLEMENT'S TABLE S6. The internal names predate
    # the journal numbering and run one behind it from S5 onward. The
    # document's Table S6 was last transcribed before the lists were
    # re-derived, so on 2026-10-09 five of its six rows disagreed with this
    # output (34 vs 40, 36 vs 42, 22 vs 38, 83 vs 82, 22 vs 39) and with
    # Fig. S21, which is drawn from the same ID sets. Re-transcribe it from
    # the CSV below after any run that touches list membership.
    print("\n    Building Table S5 (= the supplement's Table S6, "
          "overlap summary)...")
    t5 = build_table_s5(data,
                        vanish_plus_ids, blossum_plus_ids,
                        vanish_h_ids,    blossum_h_ids)
    p5 = os.path.join(out_dir, "table_s5_overlap_summary.csv")
    t5.to_csv(p5, index=False)
    print(f"    Table S5: {len(t5)} rows  ->  {p5}")
    # Print to console for quick inspection
    print(t5.to_string(index=False))

    # ── Figure S5A ──
    print("\n    Generating Figure S5A...")
    figure_s5a(data,
               vanish_plus_ids, blossum_plus_ids,
               vanish_h_ids,    blossum_h_ids,
               abund_df=abund_df,
               out_dir=out_dir)

    # ── Figure S5B ──
    print("\n    Generating Figure S5B...")
    figure_s5b(data,
               vanish_plus_ids, blossum_plus_ids,
               vanish_h_ids,    blossum_h_ids,
               out_dir=out_dir)

    print(f"\n  [supp_s5]  Done  [{ref}]")
