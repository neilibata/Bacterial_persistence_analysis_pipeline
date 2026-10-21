"""
Section 3 (GLMM sensitivity): Per-condition health associations via
logistic regression with continent FEs + depth splines.

For each condition x taxon:
    presence_i ~ condition_i + continent_FE + ns(log_depth, k)

This runs alongside the CMH analysis as a sensitivity check.
Results are saved to the same section3 output directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import OrderedDict

from . import config
from .detection import DetectionSpec, build_presence
from .health_cmh import build_group_masks, wilson_ci
from .glmm_utils import (
    build_design_matrix, bh_fdr, _fit_one_taxon, _ensure_statsmodels,
)


# ====================================================================
#  PER-CONDITION GLMM
# ====================================================================

def _run_condition_glmm(bdf, taxon_ids, meta, mask_g1, mask_g2, thresh,
                        list_name="", condition="",
                        depth_knots=3, n_jobs=-1):
    """
    Per-taxon logistic regression for one condition.

    g1 = control (e.g. non-diseased), g2 = case (e.g. diseased).
    Model: presence ~ is_case + continent_FE + ns(log_depth, k)

    Returns dict compatible with CMH result format + GLMM extras.
    """
    from joblib import Parallel, delayed

    # Subset to g1  |  g2
    idx = meta.index[mask_g1 | mask_g2]
    idx = [i for i in idx if i in bdf.index]
    if len(idx) < 20:
        return None

    group = mask_g2.loc[idx].astype(int)  # 0=control, 1=case
    log_depth = meta.loc[idx, "_log_depth"]
    continent = meta.loc[idx, "Continent"]

    # Drop samples with missing depth or continent
    valid = log_depth.notna() & continent.notna()
    idx = [i for i in idx if valid.get(i, False)]
    if len(idx) < 20:
        return None

    group = group.loc[idx]
    log_depth = log_depth.loc[idx]
    continent = continent.loc[idx]

    n_g1 = int((group == 0).sum())
    n_g2 = int((group == 1).sum())
    if n_g1 < 10 or n_g2 < 10:
        return None

    # Build design matrix
    covariates = build_design_matrix(
        log_depth, continent=continent,
        mode="continent", depth_knots=depth_knots)
    covariates = covariates.loc[idx]

    # Filter taxa
    mics = [m for m in taxon_ids if m in bdf.columns]
    if not mics:
        return None

    # Per-taxon regressions
    def _worker(mic):
        y = (bdf.loc[idx, mic] >= thresh).astype(float).values
        return _fit_one_taxon(
            mic, y, group.values, covariates.values,
            label_g0="ctrl", label_g1="case")

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_worker)(m) for m in mics)

    df = pd.DataFrame(results)
    if df.empty or "p_group" not in df.columns:
        return None

    df["q_group"] = bh_fdr(df["p_group"])

    # Count significant at FDR < 0.05
    sig = df[df["q_group"] < 0.05]
    dep = int((sig["beta_group"] < 0).sum())   # depleted in case
    enr = int((sig["beta_group"] > 0).sum())   # enriched in case

    total = len(mics)

    # Pack in CMH-compatible format
    def _ci(count, tot):
        f = count / tot if tot > 0 else 0.0
        if tot > 0:
            lo, hi = wilson_ci(count, tot)
        else:
            lo, hi = 0.0, 0.0
        return f, max(0, f - lo), max(0, hi - f)

    fd, ld, hd = _ci(dep, total)
    fe, le, he = _ci(enr, total)

    n_converged = int(df["converged"].sum())

    return {
        "n_group1": n_g1, "n_group2": n_g2, "total_taxa": total,
        "depleted_count": dep, "depleted_frac": fd,
        "depleted_ci_lo": ld, "depleted_ci_hi": hd,
        "enriched_count": enr, "enriched_frac": fe,
        "enriched_ci_lo": le, "enriched_ci_hi": he,
        "condition": condition, "method": "GLMM (continent FE)",
        "list_name": list_name,
        "n_converged": n_converged,
        "n_sig_fdr05": dep + enr,
    }


# ====================================================================
#  MAIN ENTRY POINT
# ====================================================================

LIST_ORDER = ["VANISH", "BloSSUM", "VANISH+", "BloSSUM+",
              "VanishH+", "BlossumH+"]

def run_health_glmm(data, vanish_plus_ids, blossum_plus_ids,
                     stats,
                     vanish_h_ids=None, blossum_h_ids=None,
                     depth_knots=3, n_jobs=-1):
    """
    GLMM sensitivity analysis for Section 3 health comparisons.

    Runs per-condition per-taxon-list logistic regressions with
    continent FEs + depth splines. Compares results with CMH.
    """
    ref = data.ref_name
    s3_main = config.section_dir(ref, "s3", "main")
    bdf = data.cov_matrix
    meta = data.meta

    print(f"\n  {'=' * 55}")
    print(f"  Section 3 GLMM sensitivity [{ref}]")
    print(f"  {'=' * 55}")

    # Build taxon lists
    vanish_ids = sorted(m for m in (data.vanish_set or set())
                        if m in bdf.columns)
    blossum_ids = sorted(m for m in (data.blossum_set or set())
                         if m in bdf.columns)
    vp  = [m for m in (vanish_plus_ids or []) if m in bdf.columns]
    bp  = [m for m in (blossum_plus_ids or []) if m in bdf.columns]
    vh  = [m for m in (vanish_h_ids   or []) if m in bdf.columns]
    bh  = [m for m in (blossum_h_ids  or []) if m in bdf.columns]

    taxon_lists = OrderedDict()
    if vanish_ids:  taxon_lists["VANISH"]    = vanish_ids
    if blossum_ids: taxon_lists["BloSSUM"]   = blossum_ids
    if vp:          taxon_lists["VANISH+"]   = vp
    if bp:          taxon_lists["BloSSUM+"]  = bp
    if vh:          taxon_lists["VanishH+"]  = vh
    if bh:          taxon_lists["BlossumH+"] = bh

    # Build comparisons (same as CMH)
    comparisons = OrderedDict([
        ("Unhealthy", ("unhealthy", None)),
        ("Obese",     ("obese",     None)),
    ])
    for cond_label, cond_spec in config.DISEASE_NCD_MAP.items():
        ctype = cond_spec["type"]
        dcols = cond_spec.get("cols", None)
        if ctype in ("obese", "overweight"):
            continue
        comparisons[cond_label] = (ctype, dcols)

    # Detection specs
    health_specs = [
        DetectionSpec("breadth_pct", config.BREADTH_THRESHOLD),
    ]

    all_results = []

    for spec in health_specs:
        spec_tag = spec.short_label
        spec_desc = spec.description

        if spec.mode == "breadth_pct":
            matrix, thresh = bdf, spec.threshold
        else:
            presence = build_presence(
                data.breadth, data.depth, data.lengths, spec=spec)
            matrix = presence.astype(float) * 100.0
            thresh = 50.0

        print(f"\n  GLMM health analysis: {spec_desc}")

        for cond_label, (comp_type, dcols) in comparisons.items():
            mask_g1, mask_g2 = build_group_masks(meta, comp_type, dcols)
            n_g1, n_g2 = int(mask_g1.sum()), int(mask_g2.sum())

            if n_g1 < 20 or n_g2 < 20:
                print(f"      {cond_label}: too few "
                      f"(g1={n_g1}, g2={n_g2}) — skipped")
                continue

            print(f"\n    --- {cond_label} ({comp_type}) "
                  f"g1={n_g1:,}, g2={n_g2:,} [{spec_desc}] ---")

            for list_name, mics in taxon_lists.items():
                res = _run_condition_glmm(
                    matrix, mics, meta, mask_g1, mask_g2, thresh,
                    list_name=list_name, condition=cond_label,
                    depth_knots=depth_knots, n_jobs=n_jobs)

                if res is None:
                    print(f"      {list_name:10s} GLMM: skipped")
                    continue

                all_results.append({
                    "Condition": cond_label,
                    "Method": "GLMM (continent FE)",
                    "Taxon_list": list_name,
                    "Detection": spec_tag,
                    **res,
                })

                print(f"      {list_name:10s} GLMM  "
                      f"dep={res['depleted_count']:3d}/"
                      f"{res['total_taxa']}  "
                      f"enr={res['enriched_count']:3d}/"
                      f"{res['total_taxa']}  "
                      f"({res['n_converged']} converged)")

        # Save per-spec results
        results_df = pd.DataFrame(all_results)
        if not results_df.empty:
            csv_path = os.path.join(
                s3_main,
                f"health_glmm_results_{spec_tag}.csv")
            results_df.to_csv(csv_path, index=False)
            print(f"\n    Saved: {csv_path}")

    # Combined results
    if not all_results:
        print("    [WARN] No GLMM health results")
        return pd.DataFrame()

    combined = pd.DataFrame(all_results)
    combined.to_csv(
        os.path.join(s3_main, "health_glmm_results_combined.csv"),
        index=False)

    # Print summary table
    _print_glmm_summary(combined)

    # Plot comparison with CMH if available
    cmh_path = os.path.join(s3_main,
                             "health_per_condition_cmh_combined.csv")
    if os.path.exists(cmh_path):
        cmh_df = pd.read_csv(cmh_path)
        _plot_cmh_vs_glmm(cmh_df, combined, ref, s3_main)

    # Log key stats
    for _, row in combined.iterrows():
        cond = row["Condition"]
        tl = row.get("Taxon_list", row.get("list_name", ""))
        dep = int(row["depleted_count"])
        enr = int(row["enriched_count"])
        total = int(row["total_taxa"])
        if total > 0:
            stats.add(
                f"glmm_{tl}_{cond}",
                "s3", "supplement",
                f"GLMM {tl} in {cond}: "
                f"{dep}/{total} depleted, {enr}/{total} enriched "
                f"(FDR<0.05)",
                f"dep={dep / total:.1%}, enr={enr / total:.1%}")

    print(f"\n  Section 3 GLMM complete [{ref}]")
    return combined


# ====================================================================
#  SUMMARY & COMPARISON PLOTS
# ====================================================================

def _print_glmm_summary(results_df):
    """Print compact GLMM summary table."""
    if results_df.empty:
        return

    cond_order = list(dict.fromkeys(results_df["Condition"]))
    lists_present = [l for l in LIST_ORDER
                     if l in results_df["Taxon_list"].values]

    print(f"\n{'=' * 70}")
    print(f"  HEALTH GLMM SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Condition':<22s} {'List':<9s} {'GLMM (FDR<0.05)':>20s}")
    print("-" * 70)

    for cond in cond_order:
        for ln in lists_present:
            m = results_df[(results_df["Condition"] == cond) &
                           (results_df["Taxon_list"] == ln)]
            if m.empty:
                continue
            r = m.iloc[0]
            d = int(r["depleted_count"])
            e = int(r["enriched_count"])
            t = int(r["total_taxa"])
            print(f"  {cond:<20s} {ln:<9s} "
                  f"{d:>3d}↓ {e:>3d}↑ /{t:<4d}")


LIST_COLORS = {
    "VANISH":    config.OI["vermillion"], "BloSSUM":  config.OI["blue"],
    "VANISH+":   config.OI["yellow"],    "BloSSUM+": config.OI["green"],
    "VanishH+":  config.OI["orange"],    "BlossumH+": config.OI["sky"],
}


def _plot_cmh_vs_glmm(cmh_df, glmm_df, ref, outdir):
    """
    Scatter plot: CMH depleted/enriched fractions vs GLMM.
    Shows whether the two methods agree.
    """
    # Use depth-matched CMH as the comparison
    cmh_sub = cmh_df[cmh_df["Method"] == "Depth-matched CMH"].copy()
    if cmh_sub.empty:
        cmh_sub = cmh_df.copy()

    # Merge on Condition x Taxon_list
    merged = cmh_sub.merge(
        glmm_df,
        on=["Condition", "Taxon_list"],
        suffixes=("_cmh", "_glmm"),
        how="inner")

    if len(merged) < 5:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, direction, key in zip(
            axes,
            ["Depleted", "Enriched"],
            ["depleted_frac", "enriched_frac"]):

        cmh_key = f"{key}_cmh"
        glmm_key = f"{key}_glmm"

        if cmh_key not in merged.columns or glmm_key not in merged.columns:
            continue

        for ln in LIST_ORDER:
            sub = merged[merged["Taxon_list"] == ln]
            if sub.empty:
                continue
            color = LIST_COLORS.get(ln, "#888")
            ax.scatter(sub[cmh_key], sub[glmm_key],
                       c=color, s=40, alpha=0.7,
                       edgecolors="white", lw=0.5,
                       label=ln, zorder=3)

        lim = max(merged[cmh_key].max(),
                  merged[glmm_key].max()) * 1.15
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, lw=0.8)
        ax.axhline(0.05, color='gray', ls=':', alpha=0.3, lw=0.5)
        ax.axvline(0.05, color='gray', ls=':', alpha=0.3, lw=0.5)
        ax.set_xlabel(f"Depth-matched CMH ({direction.lower()} frac.)")
        ax.set_ylabel(f"GLMM continent FE ({direction.lower()} frac.)")
        ax.set_title(direction, fontweight='bold')
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect('equal')
        ax.grid(alpha=0.15)

        if direction == "Depleted":
            ax.legend(fontsize=7, loc='upper left', framealpha=0.9)

        # Correlation
        from scipy.stats import spearmanr
        rho, p = spearmanr(merged[cmh_key], merged[glmm_key])
        ax.text(0.95, 0.05, f"rho={rho:.2f}",
                transform=ax.transAxes, fontsize=9,
                ha='right', va='bottom',
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))

    plt.suptitle(f"CMH vs GLMM Agreement ({ref})",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(outdir, f"cmh_vs_glmm_{ref}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {path}")
