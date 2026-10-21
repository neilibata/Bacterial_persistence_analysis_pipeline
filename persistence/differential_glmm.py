"""
Section 2 (alternative): Depth-corrected differential prevalence via
mixed-effects logistic regression — SPEED-OPTIMISED.

Pipeline:
  1. Per-taxon logistic regression with continent FEs + depth spline
  2. G-computation for depth-standardised prevalence differences
  3. (Optional) Within-study meta-analysis sensitivity
  4. (Optional) Threshold sensitivity
"""

import os
import numpy as np
import pandas as pd
import matplotlib
from .plot_utils import apply_oi_style, savefig_dual
apply_oi_style()
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config
from .common import assign_volcano_category_7
from .detection import DetectionSpec, build_presence
from .glmm_utils import run_taxon_glmm_batch, run_meta_sensitivity, bh_fdr


# ════════════════════════════════════════════════════════════════════
#  MAIN DIFFERENTIAL GLMM
# ════════════════════════════════════════════════════════════════════

def run_differential_glmm(data, stats, spec=None, depth_knots=3,
                          study_mode="continent", run_meta=False,
                          n_jobs=-1):
    """
    Per-taxon depth-corrected differential prevalence.

    Uses ALL samples. Depth is modelled as a spline, geography is
    controlled via continent fixed effects (fast) or study FEs (slow).

    Parameters
    ----------
    data : ReferenceData
    stats : StatsReport
    spec : DetectionSpec
    depth_knots : int (3 is fast, 4-5 is more flexible)
    study_mode : "continent" (fast) or "study_fe" (slow, thorough)
    run_meta : bool — include within-study meta-analysis
    n_jobs : int — parallel workers (-1 = all cores)

    Returns
    -------
    results_df, vanish_plus_ids, blossum_plus_ids
    """
    ref = data.ref_name
    s2_main = config.section_dir(ref, "s2", "main")

    if spec is None:
        spec = DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)
    spec_tag = spec.short_label

    meta = data.meta
    presence = build_presence(data.breadth, data.depth, data.lengths, spec=spec)

    # Group: 1 = Industrial, 0 = Non-Industrial
    valid = meta["Lifestyle"].isin(["Industrial", "H-G", "Transitional"])
    valid = valid.reindex(presence.index, fill_value=False)
    presence = presence.loc[valid]

    group = (meta.loc[presence.index, "Lifestyle"] == "Industrial").astype(int)
    log_depth = meta.loc[presence.index, "_log_depth"]
    studies = meta.loc[presence.index, "_study"]
    continents = meta.loc[presence.index, "Continent"]

    n_ind = int((group == 1).sum())
    n_ni = int((group == 0).sum())
    print(f"\n    GLMM differential: Ind={n_ind:,}, NI={n_ni:,}, "
          f"total={len(group):,}")
    print(f"    Detection: {spec.description}  |  mode={study_mode}  |  "
          f"knots={depth_knots}  |  meta={run_meta}")

    stats.add(f"glmm_diff_n_{spec_tag}", "s2", "main",
              f"GLMM samples ({spec.description})",
              f"Ind={n_ind}, NI={n_ni}", n1=n_ind, n2=n_ni)

    # ── Run batch ──
    results_df = run_taxon_glmm_batch(
        presence_df=presence,
        group_series=group,
        log_depth_series=log_depth,
        study_series=studies,
        continent_series=continents,
        study_mode=study_mode,
        depth_knots=depth_knots,
        run_meta=run_meta,
        min_prev=0.002,
        max_prev=0.998,
        n_jobs=n_jobs,
        verbose=True,
        label_g0="NI",
        label_g1="Ind",
    )

    if results_df.empty:
        print("    [WARN] No GLMM results")
        return pd.DataFrame(), [], []

    n_conv = results_df["converged"].sum()
    n_sig = (results_df["q_group"] < 0.05).sum() if "q_group" in results_df.columns else 0
    print(f"\n    Results: {len(results_df)} taxa, {n_conv} converged, "
          f"{n_sig} FDR<0.05")

    # ── VANISH+ / BloSSUM+ from ranking ──
    sorted_df = results_df.sort_values("prev_diff", ascending=True)
    n_list = min(100, len(sorted_df))
    vanish_plus_ids = sorted_df.head(n_list)["taxon"].tolist()
    blossum_plus_ids = sorted_df.tail(n_list)["taxon"].tolist()

    # ── Save ──
    csv_path = os.path.join(s2_main, f"differential_glmm_{spec_tag}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")

    _save_list(vanish_plus_ids,
               os.path.join(s2_main, f"glmm_top{n_list}_NI_enriched_{spec_tag}.txt"))
    _save_list(blossum_plus_ids,
               os.path.join(s2_main, f"glmm_top{n_list}_Ind_enriched_{spec_tag}.txt"))

    # ── Stats ──
    stats.add(f"glmm_n_tested_{spec_tag}", "s2", "main",
              f"Taxa tested by GLMM", len(results_df))
    stats.add(f"glmm_n_sig_{spec_tag}", "s2", "main",
              f"Taxa FDR<0.05", int(n_sig))

    # ── Figures ──
    _plot_volcano_glmm(results_df, data, vanish_plus_ids, blossum_plus_ids,
                       ref, s2_main, spec_tag, spec.description)
    _plot_depth_correction(results_df, ref, s2_main, spec_tag)

    return results_df, vanish_plus_ids, blossum_plus_ids


# ════════════════════════════════════════════════════════════════════
#  META-ANALYSIS SENSITIVITY (separate, optional)
# ════════════════════════════════════════════════════════════════════

def run_meta_analysis_sensitivity(data, stats, results_df=None,
                                   spec=None, n_jobs=-1):
    """
    Run within-study meta-analysis as a separate sensitivity analysis.
    Merges results into the main GLMM output.

    Call this AFTER run_differential_glmm to add meta columns.
    """
    ref = data.ref_name
    s2_main = config.section_dir(ref, "s2", "main")

    if spec is None:
        spec = DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)
    spec_tag = spec.short_label

    meta = data.meta
    presence = build_presence(data.breadth, data.depth, data.lengths, spec=spec)

    valid = meta["Lifestyle"].isin(["Industrial", "H-G", "Transitional"])
    valid = valid.reindex(presence.index, fill_value=False)
    presence = presence.loc[valid]

    group = (meta.loc[presence.index, "Lifestyle"] == "Industrial").astype(int)
    log_depth = meta.loc[presence.index, "_log_depth"]
    studies = meta.loc[presence.index, "_study"]

    taxon_ids = (list(results_df["taxon"]) if results_df is not None
                 else None)

    print(f"\n    Meta-analysis sensitivity ({spec.description})")
    meta_df = run_meta_sensitivity(
        presence, group, log_depth, studies,
        taxon_ids=taxon_ids,
        n_jobs=n_jobs, verbose=True)

    if meta_df.empty:
        print("    [WARN] No meta-analysis results")
        return results_df

    meta_path = os.path.join(s2_main, f"meta_sensitivity_{spec_tag}.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"    Saved: {meta_path}")

    n_meta_sig = (meta_df["meta_q"] < 0.05).sum()
    n_consistent = meta_df["meta_direction_consistent"].sum()
    print(f"    Meta: {len(meta_df)} taxa, {n_meta_sig} FDR<0.05, "
          f"{n_consistent} direction-consistent")

    stats.add(f"meta_n_sig_{spec_tag}", "s2", "main",
              f"Meta-analysis FDR<0.05", int(n_meta_sig))

    # Merge into main results
    if results_df is not None and not results_df.empty:
        # Drop any existing meta columns
        drop_cols = [c for c in results_df.columns if c.startswith("meta_")]
        main = results_df.drop(columns=drop_cols, errors="ignore")
        merged = main.merge(meta_df, on="taxon", how="left")

        # High-confidence taxa
        hc = ((merged["q_group"] < 0.05) &
              (merged["meta_q"] < 0.05) &
              merged["meta_direction_consistent"])
        print(f"    High-confidence (both FDR<0.05 + consistent): "
              f"{hc.sum()}")
        stats.add(f"glmm_high_confidence_{spec_tag}", "s2", "main",
                  f"High-confidence taxa", int(hc.sum()))

        # Consistency plot
        _plot_meta_consistency(merged, ref, s2_main, spec_tag)

        # Save merged
        merged.to_csv(
            os.path.join(s2_main, f"differential_glmm_with_meta_{spec_tag}.csv"),
            index=False)
        return merged

    return results_df


# ════════════════════════════════════════════════════════════════════
#  FIGURES
# ════════════════════════════════════════════════════════════════════

def _plot_volcano_glmm(results_df, data, vanish_plus_ids, blossum_plus_ids,
                       ref, outdir, spec_tag, spec_desc):
    if results_df.empty or "q_group" not in results_df.columns:
        return

    df = results_df.copy()
    df["neg_log10_q"] = -np.log10(df["q_group"].clip(lower=1e-300))

    vp_set = set(vanish_plus_ids)
    bp_set = set(blossum_plus_ids)
    df["category_7"] = df["taxon"].apply(
        lambda m: assign_volcano_category_7(
            m, data.vanish_set, data.blossum_set, vp_set, bp_set))

    fig, ax = plt.subplots(figsize=(11, 8))
    for cat in config.VOLCANO_DRAW_ORDER_7:
        sub = df[df["category_7"] == cat]
        if sub.empty:
            continue
        color = config.VOLCANO_PALETTE_7.get(cat, "#AAA")
        is_bg = (cat == "None")
        ax.scatter(sub["prev_diff"], sub["neg_log10_q"],
                   c=color, s=8 if is_bg else 18,
                   alpha=0.30 if is_bg else 0.75,
                   edgecolors="none",
                   zorder=2 if is_bg else 3,
                   rasterized=is_bg)

    ax.axhline(-np.log10(0.05), ls="--", color="grey", alpha=0.5, lw=0.6)
    ax.axvline(0, ls="-", color="grey", alpha=0.3, lw=0.5)
    ax.set_xlabel("Depth-standardised prevalence diff (Ind − NI)", fontsize=10)
    ax.set_ylabel("−log₁₀(q) [BH FDR]", fontsize=10)
    ax.set_title(f"{ref}: GLMM Prevalence — {spec_desc}",
                 fontsize=11, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=config.VOLCANO_PALETTE_7[cat],
              edgecolor="grey", linewidth=0.3,
              label=f"{cat} (n={len(df[df['category_7'] == cat])})")
        for cat in config.VOLCANO_DRAW_ORDER_7
        if not df[df["category_7"] == cat].empty
    ]
    ax.legend(handles=legend_elements, fontsize=7.5, loc="upper right",
              framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(outdir, f"volcano_glmm_{spec_tag}.png")
    savefig_dual(fig, path, dpi=300)
    plt.close()
    print(f"    Saved: {path}")


def _plot_meta_consistency(df, ref, outdir, spec_tag):
    sub = df.dropna(subset=["beta_group", "meta_effect"])
    if len(sub) < 20:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = np.where(sub["meta_direction_consistent"], "#2ca02c", "#d62728")
    ax.scatter(sub["beta_group"], sub["meta_effect"],
               c=colors, s=10, alpha=0.5, edgecolors="none")
    lims = [min(sub["beta_group"].min(), sub["meta_effect"].min()) * 1.1,
            max(sub["beta_group"].max(), sub["meta_effect"].max()) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.3, lw=0.8)
    ax.set_xlabel("Pooled GLMM β (log-odds)", fontsize=10)
    ax.set_ylabel("Within-study meta-analysis β", fontsize=10)
    ax.set_title(f"{ref}: Pooled vs Meta consistency", fontsize=11)

    from scipy.stats import spearmanr
    rho, _ = spearmanr(sub["beta_group"], sub["meta_effect"])
    ax.text(0.05, 0.95, f"ρ = {rho:.3f}, n = {len(sub)}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))
    plt.tight_layout()
    path = os.path.join(outdir, f"meta_vs_pooled_{spec_tag}.png")
    savefig_dual(fig, path, dpi=200)
    plt.close()
    print(f"    Saved: {path}")


def _plot_depth_correction(results_df, ref, outdir, spec_tag):
    df = results_df.copy()
    g0_col = [c for c in df.columns if c.startswith("raw_prev_") and "NI" in c]
    g1_col = [c for c in df.columns if c.startswith("raw_prev_") and "Ind" in c]
    if not g0_col or not g1_col:
        return

    df["raw_diff"] = df[g1_col[0]] - df[g0_col[0]]
    valid = np.isfinite(df["raw_diff"]) & np.isfinite(df["prev_diff"])
    if valid.sum() < 20:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df.loc[valid, "raw_diff"], df.loc[valid, "prev_diff"],
               s=8, alpha=0.4, c="#333", edgecolors="none")
    lims = [min(df.loc[valid, "raw_diff"].min(),
                df.loc[valid, "prev_diff"].min()) * 1.1,
            max(df.loc[valid, "raw_diff"].max(),
                df.loc[valid, "prev_diff"].max()) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.3, lw=0.8)
    ax.set_xlabel("Raw prevalence diff (Ind − NI)")
    ax.set_ylabel("Depth-standardised prevalence diff")
    ax.set_title(f"{ref}: Depth correction effect")

    from scipy.stats import spearmanr
    rho, _ = spearmanr(df.loc[valid, "raw_diff"],
                       df.loc[valid, "prev_diff"])
    ax.text(0.05, 0.95, f"ρ = {rho:.3f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))
    plt.tight_layout()
    path = os.path.join(outdir, f"depth_correction_{spec_tag}.png")
    savefig_dual(fig, path, dpi=200)
    plt.close()
    print(f"    Saved: {path}")


def _save_list(ids, path):
    with open(path, "w") as f:
        for i in ids:
            f.write(f"{i}\n")


# ════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def run_section2_glmm(data, stats, main_spec=None,
                      study_mode="continent", depth_knots=3,
                      run_meta=False, n_jobs=-1):
    """
    Full GLMM-based Section 2.

    Default settings optimised for speed:
      - continent FEs (~10 cols vs ~90)
      - delta-method CI (no bootstrap)
      - meta-analysis OFF (run separately via run_meta_analysis_sensitivity)
      - all CPU cores

    Typical runtime: 2-5 min for ~5,000 taxa on 18k samples.
    """
    ref = data.ref_name
    print(f"\n{'='*60}")
    print(f"  Section 2 (GLMM): Differential Prevalence [{ref}]")
    print(f"{'='*60}")

    if main_spec is None:
        main_spec = DetectionSpec("breadth_pct",
                                  config.ACCUM_MAIN_TEXT_BREADTH)

    # Primary analysis
    print(f"\n  2A-GLMM. Primary differential analysis")
    results_df, vanish_plus_ids, blossum_plus_ids = \
        run_differential_glmm(data, stats, spec=main_spec,
                              study_mode=study_mode,
                              depth_knots=depth_knots,
                              run_meta=run_meta,
                              n_jobs=n_jobs)

    # Meta-analysis sensitivity — skip by default, enable with run_meta=True
    if run_meta:
        print(f"\n  2B-GLMM. Within-study meta-analysis sensitivity")
        results_df = run_meta_analysis_sensitivity(
            data, stats, results_df=results_df,
            spec=main_spec, n_jobs=n_jobs)
    else:
        print(f"\n  [Skipping 2B meta-analysis — pass run_meta=True to enable]")

    print(f"\n  Section 2 GLMM complete [{ref}]")
    return results_df, vanish_plus_ids, blossum_plus_ids
