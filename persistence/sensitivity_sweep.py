"""
Sensitivity sweep: re-run Sections 2 and 3 with alternative VANISH/BloSSUM
definitions derived from the data.

This module:
  1. Computes data-driven VANISH/BloSSUM sets for each VBDefinition
  2. Re-runs Section 2's differential analysis (volcanos)
  3. Re-runs Section 3's health analyses (triplet Fisher, per-condition CMH)
  4. Produces comparison summaries across all definitions

Usage (from __main__):
    python -m persistence --sensitivity
    python -m persistence --sensitivity --sensitivity-defs tight loose
"""

import os
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib
from .plot_utils import apply_oi_style, savefig_dual, current_fs
apply_oi_style()
FS = current_fs()   # base font size; changes per preset in savefig_dual
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import OrderedDict

from . import config
from .vanish_blossum_definitions import (
    VBDefinition, VBResult, define_vanish_blossum, sweep_definitions,
    compare_with_original, compute_prevalence, option_b_score,
    PRESET_DEFINITIONS, DEFAULT_DEFINITION, DEFAULT_STUDY_AWARE,
)
from .loading import ReferenceData
from .stats_report import StatsReport
from .detection import DetectionSpec


# ════════════════════════════════════════════════════════════════════
#  NAMED DEFINITION SETS
# ════════════════════════════════════════════════════════════════════

# Curated subsets for common use cases

TIGHT_DEFINITIONS = [
    VBDefinition(r=0.01, m=0.10, u=0.80, label="tight_pool",
                 study_aware=False),
    VBDefinition(r=0.01, m=0.10, u=0.80, label="tight_SA",
                 study_aware=True),
    VBDefinition(r=0.01, m=0.20, u=0.80, label="tight_strict_pool",
                 study_aware=False),
]

LOOSE_DEFINITIONS = [
    VBDefinition(r=0.05, m=0.05, u=0.80, label="loose_pool",
                 study_aware=False),
    VBDefinition(r=0.05, m=0.05, u=0.80, label="loose_SA",
                 study_aware=True),
    VBDefinition(r=0.05, m=0.10, u=0.90, label="loose_wide_pool",
                 study_aware=False),
]

CORE_DEFINITIONS = [
    DEFAULT_DEFINITION,
    DEFAULT_STUDY_AWARE,
    VBDefinition(r=0.01, m=0.10, u=0.80, label="strict",
                 study_aware=False),
    VBDefinition(r=0.05, m=0.10, u=0.80, label="lenient",
                 study_aware=False),
    VBDefinition(r=0.02, m=0.05, u=0.80, label="low_m",
                 study_aware=False),
    VBDefinition(r=0.02, m=0.20, u=0.80, label="high_m",
                 study_aware=False),
]

NAMED_SETS = {
    "core": CORE_DEFINITIONS,
    "tight": TIGHT_DEFINITIONS,
    "loose": LOOSE_DEFINITIONS,
    "full": PRESET_DEFINITIONS,
}


# ════════════════════════════════════════════════════════════════════
#  HELPER: CREATE A MODIFIED ReferenceData WITH NEW V/B SETS
# ════════════════════════════════════════════════════════════════════

def _replace_vb_sets(data, vb_result):
    """
    Return a new ReferenceData with vanish_set/blossum_set replaced
    by the data-driven sets from vb_result.
    """
    # _replace, not a field-by-field rebuild: the rebuild omitted
    # total_seq_bp, which has no default, so this raised TypeError and the
    # sweep aborted before producing any Section 2/3 result.
    return data._replace(vanish_set=vb_result.vanish_set,
                         blossum_set=vb_result.blossum_set)


# ════════════════════════════════════════════════════════════════════
#  SECTION 2 RE-RUN (per definition)
# ════════════════════════════════════════════════════════════════════

def _rerun_section2_for_definition(data_mod, dm, vb_result, stats, out_dir):
    """
    Run Section 2's differential analysis using the modified V/B sets.

    Returns (prev_df, abund_df, vanish_plus_ids, blossum_plus_ids).
    """
    from .differential import run_differential

    tag = vb_result.definition.tag

    # Use depth-matched samples if available
    matched_idx = dm.all_matched_ind_vs_ni if dm else list(data_mod.meta.index)

    prev_df, abund_df, vanish_plus_ids, blossum_plus_ids = \
        run_differential(data_mod, matched_idx)

    # Move outputs to tagged subdirectory
    ref_out = config.outdir(data_mod.ref_name)
    for fn in os.listdir(ref_out):
        src = os.path.join(ref_out, fn)
        if os.path.isfile(src) and any(fn.startswith(p) for p in [
                "volcano_", "prevalence_ind_", "abundance_ind_",
                "top100_"]):
            dst = os.path.join(out_dir, f"{tag}_{fn}")
            shutil.copy2(src, dst)

    return prev_df, abund_df, vanish_plus_ids, blossum_plus_ids


# ════════════════════════════════════════════════════════════════════
#  SECTION 3 RE-RUN (per definition)
# ════════════════════════════════════════════════════════════════════

def _rerun_section3_for_definition(data_mod, dm, vb_result,
                                    vanish_plus_ids, blossum_plus_ids,
                                    stats, out_dir):
    """
    Run Section 3's health analyses using the modified V/B sets.

    Returns a DataFrame of health results.
    """
    from .section3_health import run_triplet_health_analysis

    tag = vb_result.definition.tag

    if dm is None or not dm.triplet_ni:
        print(f"      [{tag}] No triplet samples — skipping Section 3")
        return pd.DataFrame()

    combined = run_triplet_health_analysis(
        data_mod, dm, vanish_plus_ids, blossum_plus_ids,
        stats)

    # Copy health outputs to tagged subdirectory
    s3_main = config.section_dir(data_mod.ref_name, "s3", "main")
    if os.path.isdir(s3_main):
        for fn in os.listdir(s3_main):
            src = os.path.join(s3_main, fn)
            if os.path.isfile(src):
                dst = os.path.join(out_dir, f"{tag}_{fn}")
                shutil.copy2(src, dst)

    return combined


# ════════════════════════════════════════════════════════════════════
#  SUMMARY: STABILITY ACROSS DEFINITIONS
# ════════════════════════════════════════════════════════════════════

def _build_overlap_heatmap(vb_results, data, out_dir):
    """
    Build and save a heatmap showing pairwise Jaccard overlap of
    VANISH sets (and separately BloSSUM sets) across definitions.
    """
    n = len(vb_results)
    tags = [vb.definition.tag for vb in vb_results]

    for list_name, attr in [("VANISH", "vanish_set"),
                             ("BloSSUM", "blossum_set")]:
        sets = [getattr(vb, attr) for vb in vb_results]
        jaccard = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                union = sets[i] | sets[j]
                jaccard[i, j] = (len(sets[i] & sets[j]) / len(union)
                                 if union else 1.0)

        fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.5)))
        im = ax.imshow(jaccard, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tags, rotation=90, fontsize=FS-3)
        ax.set_yticklabels(tags, fontsize=FS-3)
        ax.set_title(f"{list_name} Jaccard overlap across definitions",
                     fontsize=FS)
        fig.colorbar(im, ax=ax, shrink=0.8)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{jaccard[i,j]:.2f}",
                        ha="center", va="center", fontsize=FS-4,
                        color="white" if jaccard[i, j] > 0.5 else "black")

        plt.tight_layout()
        fp = os.path.join(out_dir, f"overlap_heatmap_{list_name}.png")
        savefig_dual(fig, fp, dpi=200)
        plt.close()
        print(f"    Saved: {fp}")


def _build_comparison_with_original(vb_results, data, out_dir):
    """
    Compare each data-driven definition with the original file-based
    VANISH/BloSSUM lists and save a summary CSV.
    """
    rows = []
    for vb in vb_results:
        comp = compare_with_original(
            vb, data.vanish_set, data.blossum_set)
        comp["description"] = vb.definition.description
        rows.append(comp)

    df = pd.DataFrame(rows)
    fp = os.path.join(out_dir, "definitions_vs_original.csv")
    df.to_csv(fp, index=False)
    print(f"    Saved: {fp}")
    return df


def _build_list_size_barplot(vb_results, data, out_dir):
    """Bar chart showing VANISH and BloSSUM counts per definition."""
    tags = [vb.definition.tag for vb in vb_results]
    n_v = [vb.n_vanish for vb in vb_results]
    n_b = [vb.n_blossum for vb in vb_results]

    x = np.arange(len(tags))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(tags) * 0.7), 5))
    ax.bar(x - w/2, n_v, w, label="VANISH", color=config.OI["vermillion"], alpha=0.8)
    ax.bar(x + w/2, n_b, w, label="BloSSUM", color=config.OI["blue"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=90, fontsize=FS-3)
    ax.set_ylabel("Number of taxa")
    ax.set_title("VANISH / BloSSUM list sizes across definitions")
    ax.legend()

    # Add original sizes as horizontal lines
    if data.vanish_set:
        ax.axhline(len(data.vanish_set), ls="--", color=config.OI["vermillion"],
                    alpha=0.5, lw=0.8, label=f"Original VANISH (n={len(data.vanish_set)})")
    if data.blossum_set:
        ax.axhline(len(data.blossum_set), ls="--", color=config.OI["blue"],
                    alpha=0.5, lw=0.8, label=f"Original BloSSUM (n={len(data.blossum_set)})")
    ax.legend(fontsize=FS-3, loc="upper right")

    plt.tight_layout()
    fp = os.path.join(out_dir, "list_sizes_barplot.png")
    savefig_dual(fig, fp, dpi=200)
    plt.close()
    print(f"    Saved: {fp}")


def _build_health_summary(all_health_results, out_dir):
    """
    Build a combined CSV and bar chart of health-association fractions
    across all definitions.
    """
    if not all_health_results:
        return

    rows = []
    for tag, health_df in all_health_results.items():
        if health_df is None or health_df.empty:
            continue
        health_df = health_df.copy()
        health_df["defn_tag"] = tag
        rows.append(health_df)

    if not rows:
        return

    combined = pd.concat(rows, ignore_index=True)
    fp = os.path.join(out_dir, "health_all_definitions_combined.csv")
    combined.to_csv(fp, index=False)
    print(f"    Saved: {fp}")

    # Summary plot: for each definition x taxon list, show
    # depleted fraction (Unhealthy condition, Depth-matched CMH)
    _plot_health_fraction_comparison(combined, out_dir)


def _plot_health_fraction_comparison(combined, out_dir):
    """
    For each definition, plot the fraction of depleted VANISH taxa
    in Unhealthy (depth-matched CMH) as a grouped bar chart.
    """
    # Filter to "Unhealthy" condition and "Depth-matched CMH" method
    mask = (
        (combined.get("Condition", pd.Series()) == "Unhealthy") &
        (combined.get("Method", pd.Series()) == "Depth-matched CMH")
    )
    sub = combined[mask].copy()
    if sub.empty:
        # Try "Within-study CMH" as fallback
        mask = (
            (combined.get("Condition", pd.Series()) == "Unhealthy") &
            (combined.get("Method", pd.Series()) == "Within-study CMH")
        )
        sub = combined[mask].copy()
    if sub.empty:
        return

    # Compute depleted fraction
    sub["depleted_frac"] = sub["depleted_count"] / sub["total_taxa"].clip(lower=1)

    tags = sub["defn_tag"].unique()
    lists_present = [l for l in ["VANISH", "BloSSUM", "VANISH+", "BloSSUM+"]
                     if l in sub["Taxon_list"].values]

    if not lists_present:
        return

    x = np.arange(len(tags))
    n_lists = len(lists_present)
    w = 0.8 / max(n_lists, 1)

    colors = {"VANISH": config.OI["vermillion"], "BloSSUM": config.OI["blue"],
              "VANISH+": config.OI["yellow"], "BloSSUM+": config.OI["green"]}

    fig, ax = plt.subplots(figsize=(max(10, len(tags) * 0.8), 5))

    for i, ln in enumerate(lists_present):
        vals = []
        for t in tags:
            row = sub[(sub["defn_tag"] == t) & (sub["Taxon_list"] == ln)]
            vals.append(row["depleted_frac"].values[0] if not row.empty else 0)
        ax.bar(x + i * w - (n_lists - 1) * w / 2, vals, w,
               label=ln, color=colors.get(ln, "#888"), alpha=0.8)

    ax.axhline(0.5, ls="--", color="grey", alpha=0.4, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=90, fontsize=FS-3)
    ax.set_ylabel("Fraction depleted in Unhealthy")
    ax.set_title("Health signal stability across VANISH/BloSSUM definitions")
    ax.legend(fontsize=FS-3, loc="upper right")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fp = os.path.join(out_dir, "health_fraction_across_definitions.png")
    savefig_dual(fig, fp, dpi=200)
    plt.close()
    print(f"    Saved: {fp}")


# ════════════════════════════════════════════════════════════════════
#  PREVALENCE LANDSCAPE PLOT
# ════════════════════════════════════════════════════════════════════

def _plot_prevalence_landscape(data, vb_results, out_dir):
    """
    Scatter plot of p_I vs p_N for all taxa, coloured by VANISH/BloSSUM
    membership across definitions. Shows the "band-pass" selection zones.
    """
    # Use the first definition's prevalence table as the base
    if not vb_results:
        return

    prev_df = vb_results[0].prevalence_df
    if prev_df is None or prev_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Background: all taxa
    ax.scatter(prev_df["prev_I"], prev_df["prev_N"],
               c="#DDDDDD", s=3, alpha=0.3, rasterized=True, zorder=1)

    # Highlight VANISH/BloSSUM for default definition
    for vb in vb_results[:3]:  # show first 3 definitions
        tag = vb.definition.tag
        v_df = prev_df[prev_df["taxon"].isin(vb.vanish_set)]
        b_df = prev_df[prev_df["taxon"].isin(vb.blossum_set)]

        if not v_df.empty:
            ax.scatter(v_df["prev_I"], v_df["prev_N"],
                       s=12, alpha=0.5, marker="o",
                       label=f"VANISH ({tag}, n={len(v_df)})", zorder=3)
        if not b_df.empty:
            ax.scatter(b_df["prev_I"], b_df["prev_N"],
                       s=12, alpha=0.5, marker="s",
                       label=f"BloSSUM ({tag}, n={len(b_df)})", zorder=3)

    # Draw band-pass zones for the default definition
    defn = vb_results[0].definition
    ax.axvline(defn.r, ls=":", color="red", alpha=0.4, lw=0.8)
    ax.axhline(defn.r, ls=":", color="blue", alpha=0.4, lw=0.8)
    ax.axhline(defn.m, ls="--", color="green", alpha=0.3, lw=0.8)
    ax.axhline(defn.u, ls="--", color="green", alpha=0.3, lw=0.8)
    ax.axvline(defn.m, ls="--", color="green", alpha=0.3, lw=0.8)
    ax.axvline(defn.u, ls="--", color="green", alpha=0.3, lw=0.8)

    ax.set_xlabel("Prevalence (Industrial)", fontsize=FS)
    ax.set_ylabel("Prevalence (Non-Industrial)", fontsize=FS)
    ax.set_title("Prevalence landscape: VANISH/BloSSUM selection zones",
                 fontsize=FS+1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.plot([0, 1], [0, 1], ls="-", color="grey", alpha=0.2, lw=0.5)
    ax.legend(fontsize=FS-4, loc="upper left", ncol=1)

    plt.tight_layout()
    fp = os.path.join(out_dir, "prevalence_landscape.png")
    savefig_dual(fig, fp, dpi=250)
    plt.close()
    print(f"    Saved: {fp}")


# ════════════════════════════════════════════════════════════════════
#  MAIN SWEEP ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def run_sensitivity_sweep(data, dm, stats,
                           definitions=None,
                           run_section2=True,
                           run_section3=True):
    """
    Full sensitivity sweep: derive V/B from data, re-run Sections 2 & 3.

    Parameters
    ----------
    data : ReferenceData
        Original data (with file-based vanish_set/blossum_set).
    dm : DepthMatchResult
        Depth-matched sample sets.
    stats : StatsReport
    definitions : list of VBDefinition, optional
        Defaults to CORE_DEFINITIONS.
    run_section2 : bool
    run_section3 : bool

    Returns
    -------
    dict: {defn_tag: {"vb": VBResult, "s2": ..., "s3": ...}}
    """
    if definitions is None:
        definitions = CORE_DEFINITIONS

    ref = data.ref_name
    sweep_dir = os.path.join("results", ref, "sensitivity_sweep")
    os.makedirs(sweep_dir, exist_ok=True)
    s2_dir = os.path.join(sweep_dir, "section2")
    s3_dir = os.path.join(sweep_dir, "section3")
    os.makedirs(s2_dir, exist_ok=True)
    os.makedirs(s3_dir, exist_ok=True)

    t0 = time.time()
    print(f"\n{'='*65}")
    print(f"  SENSITIVITY SWEEP: {len(definitions)} definitions [{ref}]")
    print(f"{'='*65}")

    # ── Step 1: Compute all VB definitions ──
    print(f"\n  Step 1: Computing VANISH/BloSSUM for each definition...")
    vb_results = sweep_definitions(data, definitions)

    # Save per-definition prevalence tables
    for vb in vb_results:
        if vb.prevalence_df is not None:
            fp = os.path.join(sweep_dir,
                              f"prevalence_{vb.definition.tag}.csv")
            vb.prevalence_df.to_csv(fp, index=False)

    # ── Step 2: Summary diagnostics ──
    print(f"\n  Step 2: Building summary diagnostics...")
    _build_comparison_with_original(vb_results, data, sweep_dir)
    _build_list_size_barplot(vb_results, data, sweep_dir)
    _build_overlap_heatmap(vb_results, data, sweep_dir)
    _plot_prevalence_landscape(data, vb_results, sweep_dir)

    # ── Step 3: Re-run Section 2 per definition ──
    all_results = OrderedDict()
    all_health_results = OrderedDict()

    for vb in vb_results:
        tag = vb.definition.tag
        print(f"\n  {'─'*55}")
        print(f"  Definition: {vb.definition.description}")
        print(f"    VANISH={vb.n_vanish}, BloSSUM={vb.n_blossum}")
        print(f"  {'─'*55}")

        data_mod = _replace_vb_sets(data, vb)
        result_entry = {"vb": vb}

        if run_section2:
            print(f"\n    Re-running Section 2...")
            try:
                prev_df, abund_df, vp_ids, bp_ids = \
                    _rerun_section2_for_definition(
                        data_mod, dm, vb, stats, s2_dir)
                result_entry["s2_prev"] = prev_df
                result_entry["s2_abund"] = abund_df
                result_entry["vanish_plus"] = vp_ids
                result_entry["blossum_plus"] = bp_ids
            except Exception as e:
                print(f"    [ERROR] Section 2 failed for {tag}: {e}")
                result_entry["s2_error"] = str(e)
                vp_ids, bp_ids = [], []
        else:
            vp_ids, bp_ids = [], []

        if run_section3:
            print(f"\n    Re-running Section 3...")
            try:
                health_df = _rerun_section3_for_definition(
                    data_mod, dm, vb,
                    vp_ids, bp_ids,
                    stats, s3_dir)
                result_entry["s3_health"] = health_df
                all_health_results[tag] = health_df
            except Exception as e:
                print(f"    [ERROR] Section 3 failed for {tag}: {e}")
                result_entry["s3_error"] = str(e)

        all_results[tag] = result_entry

    # ── Step 4: Cross-definition health summary ──
    print(f"\n  Step 4: Building cross-definition health summary...")
    _build_health_summary(all_health_results, sweep_dir)

    # ── Step 5: Save master summary ──
    summary_rows = []
    for tag, entry in all_results.items():
        vb = entry["vb"]
        row = {
            "definition": tag,
            "description": vb.definition.description,
            "n_vanish": vb.n_vanish,
            "n_blossum": vb.n_blossum,
            "r": vb.definition.r,
            "m": vb.definition.m,
            "u": vb.definition.u,
            "study_aware": vb.definition.study_aware,
        }
        # Add overlap with original
        comp = compare_with_original(vb, data.vanish_set, data.blossum_set)
        row.update(comp)

        # Add health signal if available
        health_df = entry.get("s3_health", pd.DataFrame())
        if health_df is not None and not health_df.empty:
            # Extract VANISH depleted fraction for "Unhealthy"
            for method in ["Depth-matched CMH", "Within-study CMH"]:
                uh_v = health_df[
                    (health_df.get("Condition", pd.Series()) == "Unhealthy") &
                    (health_df.get("Method", pd.Series()) == method) &
                    (health_df.get("Taxon_list", pd.Series()) == "VANISH")]
                if not uh_v.empty:
                    r = uh_v.iloc[0]
                    total = int(r.get("total_taxa", 0))
                    dep = int(r.get("depleted_count", 0))
                    row[f"vanish_depleted_frac_{method.replace(' ', '_')}"] = \
                        dep / total if total > 0 else np.nan
                    row[f"vanish_depleted_n_{method.replace(' ', '_')}"] = dep
                    row[f"vanish_total_{method.replace(' ', '_')}"] = total

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    fp = os.path.join(sweep_dir, "sensitivity_sweep_summary.csv")
    summary_df.to_csv(fp, index=False)
    print(f"    Saved: {fp}")

    elapsed = time.time() - t0
    print(f"\n  Sensitivity sweep complete [{ref}] in {elapsed/60:.1f} min")
    print(f"  Results in: {sweep_dir}/")

    return all_results
