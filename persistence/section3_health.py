"""
Section 3: Persistence of VANISH or VANISH+ taxa is associated with
           health in industrial contexts  (v4).

v4 changes vs v3
-----------------
 - Per-condition analyses now use within-study CMH (Tier 2) and
   depth-matched within-study CMH (Tier 3), not pooled Fisher.
 - Mundlak within–between GLMM (Tier 4) decomposes the group effect
   into within-study (beta_w, same estimand as CMH) and between-study
   (beta_b, potentially confounded) components.  Hausman diagnostic
   flags conditions where naive pooling would be biased.
 - Obesity uses a SYMMETRIC comparison: healthy non-obese vs healthy
   obese within Industrial, with its own depth matching.
 - Disease groupings updated: T2D+IGT as "Glucose Intolerance" (T1D excluded),
   IBD includes Crohn's+UC, CRC+adenoma together.
 - Summary figure: bidirectional bar chart of depth-matched CMH
   fractions across all conditions and taxon lists.

Analysis tiers (per condition x taxon list)
-------------------------------------------
  Tier 2 : Within-study CMH (study-stratified, unmatched)
  Tier 3 : Depth-matched within-study CMH (1:1 NN within each study)
  Tier 4 : Mundlak GLMM (within–between decomposition, FDR < 0.05
           on beta_w, cluster-robust SEs by study)

Section structure
-----------------
  3A. Triplet-based main comparisons (NI x IndH x IndU) — Fisher
  3B. Per-condition within-study CMH (Tier 2 + Tier 3 + Tier 4)
  3C. Binomial significance tests (fraction > 5% chance?)
  3D. Summary figure (bidirectional bar chart)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
from .plot_utils import (
    apply_oi_style,
    current_fs,
    page_figsize,
    pin_ylabel,
    square_panels,
    SUBMISSION_DPI,
    add_panel_letters,
    add_post_layout_hook,
    panel_grid,
    savefig_dual,
    share_axis_labels,
)
apply_oi_style()
FS = current_fs()   # base font size; changes per preset in savefig_dual

#: Vertical inches per taxon row in the driver heatmaps (S26-S29). Printed row
#: pitch is this x the 6.3/7.24 publication magnification: 0.30 in gave 18.8 pt
#: for a 4.4 pt label, over four times the type height, which is far looser than
#: a heatmap needs. Row count is now set by how many taxa clear p_thresh rather
#: than by a fixed cap, so the allowance has to stay tight: at 0.19 in the
#: 30-row ceiling is a 7.0 in figure, which still leaves a page for its caption.
DRIVER_ROW_IN = 0.19
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as _pe
from matplotlib.patches import Patch
from scipy.stats import binomtest, spearmanr, chi2_contingency, fisher_exact as scipy_fisher, wilcoxon
from collections import OrderedDict

from . import config
from . import figure_source
from . import health_background as hbg
from .plot_utils import (background_curve_legend,
                         draw_background_curve_grid,
                         background_curve_style,
                         background_curve_figsize,
                         display_order, group_dividers,
                         CELL_WSPACE, CELL_HSPACE, PAGE_WIDTH_IN)
from .detection import DetectionSpec, build_presence
from .health_cmh import (
    fisher_enrichment_test,
    cmh_enrichment_test,
    depth_matched_cmh_enrichment_test,
    geom_mean_weighted_cmh_enrichment_test,
    geom_mean_weighted_cmh_per_taxon,
    stratified_shifts_per_taxon,
    mundlak_enrichment_test,
    build_group_masks,
    wilson_ci,
)


# ====================================================================
#  WHICH SHIFT, WHICH WEIGHT  (one place, so every consumer agrees)
# ====================================================================
#
# Two quantities changed when the prevalence background was corrected on
# 2026-10-06, and about a dozen places read them. Those places now all call
# these accessors, so no figure can end up drawing the pooled shift against a
# matched background, or one list's weights against another's taxa.
#
#   the shift   `rd_mh`      study-stratified (Mantel-Haenszel) case-control
#                            risk difference -- the corrected quantity
#               `prev_diff`  the pooled difference the 29 Sep run published,
#                            and all its pickles still carry
#   the weight  `bg_weight`  flat-kernel prevalence-matched weight
#               `importance_weight`  the old rounded importance weight
#
# The fallback is what keeps `tools/replot_from_data.py` able to redraw the
# published package from its own pickles: an old data object has only the old
# columns, and must keep drawing the figure it was published as.

SHIFT_COLUMNS = ("rd_mh", "prev_diff")
BG_WEIGHT_COLUMNS = ("bg_weight", "importance_weight")


def _shift_column(df):
    """Name of the case-control shift column present in `df`, or None."""
    for c in SHIFT_COLUMNS:
        if c in getattr(df, "columns", ()):
            return c
    return None


def _shift_values(df, signed_pp=False):
    """The case-control shift.

    Default sign is the canonical one: control - case, so POSITIVE = depleted
    in disease. `signed_pp=True` gives case - control in percentage points,
    the form the figures plot (positive = enriched in disease).
    """
    col = _shift_column(df)
    if col is None:
        return np.array([])
    v = pd.to_numeric(df[col], errors="coerce").values.astype(float)
    return -v * 100.0 if signed_pp else v


def _bg_weight_values(df, normalise=False):
    """Background weights: the matched weights if present, else the old IW ones."""
    n = len(df)
    for c in BG_WEIGHT_COLUMNS:
        if c in getattr(df, "columns", ()):
            w = pd.to_numeric(df[c], errors="coerce").fillna(1.0).values.astype(float)
            w = np.clip(w, 0.0, None)
            if normalise and w.sum() > 0:
                w = w / w.sum()
            return w
    w = np.ones(n)
    return (w / n) if (normalise and n) else w


def _common_shift_frames(*frames):
    """Make a set of frames agree on which shift column to use.

    A list table and a background table are routinely concatenated and then
    filtered on the shift column. If one of them carries `rd_mh` and the other
    does not -- a table written before the stratified shift was merged on, or a
    published CSV read back -- the concatenation gives the missing side all-NaN
    and the filter silently drops every one of its rows. (That emptied Figure
    5E completely the first time it was drawn this way.) So: keep `rd_mh` only
    where every frame has it, and otherwise fall back to `prev_diff` for all of
    them.
    """
    frames = [f for f in frames if f is not None and len(f)]
    if len(frames) < 2:
        return
    have = [c for c in SHIFT_COLUMNS
            if any(c in f.columns for f in frames)]
    for c in have:
        if not all(c in f.columns for f in frames):
            print(f"  [WARN] {c} is missing from one of the tables being "
                  "compared, so the comparison falls back to the pooled "
                  "shift. Rerun section 3 to get the corrected one.")
            for f in frames:
                if c in f.columns:
                    f.drop(columns=[c], inplace=True)


def _dropna_shift(df, extra=()):
    """Drop rows with no case-control shift (and optionally no p-value)."""
    col = _shift_column(df)
    cols = ([col] if col else []) + [c for c in extra if c in getattr(df, "columns", ())]
    return df.dropna(subset=cols) if cols else df


def _bg_label(df=None, short=False):
    """What to call the background in a figure's own words.

    Follows the data rather than the config, so redrawing a published object
    still labels it as the background it actually holds.
    """
    cols = getattr(df, "columns", ()) if df is not None else ()
    iw = ("bg_weight" not in cols and "importance_weight" in cols) or \
         (df is None and _bg_method() == "iw")
    if iw:
        return "IW background" if short else "importance-weighted background"
    return ("matched background" if short
            else "prevalence-matched background")


def _bg_method():
    """"curve" (corrected) or "iw" (reproduce the published figures)."""
    return str(getattr(config, "HEALTH_BG_METHOD", "curve")).lower()


# ====================================================================
#  DETECTION-SPEC HELPERS
# ====================================================================

def _build_presence_matrix(data, spec):
    """
    Build a presence DataFrame for a given DetectionSpec.
    For breadth_pct: use raw breadth matrix with spec.threshold.
    For flat_bp: build boolean presence, scale to 0/100, use thresh=50.
    Returns (matrix_df, effective_threshold).
    """
    if spec.mode == "breadth_pct":
        return data.cov_matrix, spec.threshold
    else:
        presence = build_presence(
            data.breadth, data.depth, data.lengths, spec=spec)
        pseudo = presence.astype(float) * 100.0
        return pseudo, 50.0


# ====================================================================
#  3A. TRIPLET-BASED HEALTH COMPARISONS (per detection spec)
# ====================================================================

def _run_triplet_analysis(data, dm, taxon_lists, stats, spec, s3_main):
    """
    Run Fisher-based triplet comparisons for one detection spec.
    These are the NI vs IndH vs IndU comparisons on globally
    triplet-matched samples.
    """
    spec_tag = spec.short_label
    spec_desc = spec.description

    bdf, thresh = _build_presence_matrix(data, spec)

    ids_ni = dm.triplet_ni
    ids_h = dm.triplet_ind_h
    ids_u = dm.triplet_ind_u

    method_label = f"Triplet Fisher ({spec_desc})"
    rows = []

    # -- Main: Healthy vs Unhealthy (within Industrial) --
    print(f"\n    --- Ind_Healthy vs Ind_Unhealthy [{spec_desc}] ---")
    for list_name, mics in taxon_lists.items():
        result = fisher_enrichment_test(
            bdf, mics, ids_h, ids_u, thresh,
            list_name=list_name, condition="Unhealthy",
            method=method_label)
        if result:
            rows.append(result)
            print(f"      {list_name:10s}  dep={result['depleted_count']:3d}/"
                  f"{result['total_taxa']}  "
                  f"enr={result['enriched_count']:3d}/"
                  f"{result['total_taxa']}")

    # -- NI vs Industrial_Healthy --
    print(f"\n    --- NI vs Ind_Healthy [{spec_desc}] ---")
    for list_name, mics in taxon_lists.items():
        result = fisher_enrichment_test(
            bdf, mics, ids_ni, ids_h, thresh,
            list_name=list_name, condition="NI_vs_IndH",
            method=method_label)
        if result:
            rows.append(result)

    # -- NI vs Industrial_Unhealthy --
    print(f"\n    --- NI vs Ind_Unhealthy [{spec_desc}] ---")
    for list_name, mics in taxon_lists.items():
        result = fisher_enrichment_test(
            bdf, mics, ids_ni, ids_u, thresh,
            list_name=list_name, condition="NI_vs_IndU",
            method=method_label)
        if result:
            rows.append(result)

    triplet_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not triplet_df.empty:
        triplet_df.to_csv(
            os.path.join(s3_main,
                         f"health_triplet_results_{spec_tag}.csv"),
            index=False)
        _log_health_stats(triplet_df, stats, spec_tag)
        _plot_prevalence_gradient(data, dm, stats, s3_main,
                                  bdf, thresh, spec_tag, spec_desc)

    return triplet_df


# ====================================================================
#  3B. PER-CONDITION WITHIN-STUDY CMH (Tier 2 + Tier 3)
# ====================================================================

def _run_per_condition_cmh(data, taxon_lists, stats, spec, s3_main):
    """
    Run within-study CMH and depth-matched CMH for every condition
    in config.DISEASE_NCD_MAP, plus Unhealthy and Obese.

    Returns a DataFrame with all results.
    """
    spec_tag = spec.short_label
    spec_desc = spec.description
    bdf, thresh = _build_presence_matrix(data, spec)
    meta = data.meta

    # The comparison map, from the one function that defines it. This block
    # was a hand-kept copy; _health_comparisons' own docstring says the three
    # per-condition runners must iterate the identical set.
    comparisons = _health_comparisons()

    TIERS = OrderedDict([
        ("Within-study CMH",        cmh_enrichment_test),
        ("Depth-matched CMH",       depth_matched_cmh_enrichment_test),
        ("Geom-mean-weighted CMH",  geom_mean_weighted_cmh_enrichment_test),
        ("Mundlak GLMM",            mundlak_enrichment_test),
    ])

    all_results = []

    for cond_label, (comp_type, dcols) in comparisons.items():
        mask_g1, mask_g2 = build_group_masks(meta, comp_type, dcols)
        n_g1, n_g2 = int(mask_g1.sum()), int(mask_g2.sum())

        if n_g1 < 10 or n_g2 < 10:
            print(f"      {cond_label}: too few (g1={n_g1}, g2={n_g2}) — skipped")
            continue

        print(f"\n    --- {cond_label} ({comp_type}) "
              f"g1={n_g1:,}, g2={n_g2:,} [{spec_desc}] ---")

        for list_name, mics in taxon_lists.items():
            for tier_name, tier_fn in TIERS.items():
                res = tier_fn(
                    bdf, mics, meta, mask_g1, mask_g2, thresh,
                    list_name=list_name, condition=cond_label,
                    method=tier_name)

                if res is None:
                    continue

                all_results.append({
                    "Condition": cond_label,
                    "Method": tier_name,
                    "Taxon_list": list_name,
                    **res,
                })

                ns = res.get("n_studies", "")
                ns_str = f"  ({ns} studies)" if ns else ""
                dd = res.get("median_log_depth_diff", "")
                dd_str = ""
                if isinstance(dd, float) and not np.isnan(dd):
                    dd_str = f"  dlogD={dd:.3f}"
                # Mundlak-specific diagnostics
                conv = res.get("n_converged", "")
                conv_str = f"  conv={conv}" if conv != "" else ""
                haus = res.get("n_hausman_sig", "")
                haus_str = f"  Haus={haus}sig" if haus != "" else ""
                # Geom-mean-weighted CMH diagnostics
                ess_frac = res.get("mean_ess_frac", "")
                ess_str = ""
                if isinstance(ess_frac, float) and not np.isnan(ess_frac):
                    ess_str = f"  ESS={ess_frac:.2f}"
                print(f"      {list_name:10s} {tier_name:26s}  "
                      f"dep={res['depleted_count']:3d}/"
                      f"{res['total_taxa']}  "
                      f"enr={res['enriched_count']:3d}/"
                      f"{res['total_taxa']}  "
                      f"n1={res['n_group1']:,} n2={res['n_group2']:,}"
                      f"{ns_str}{dd_str}{conv_str}{haus_str}{ess_str}")

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        out_csv = os.path.join(
            s3_main,
            f"health_per_condition_cmh_{spec_tag}.csv")
        results_df.to_csv(out_csv, index=False)
        print(f"\n    Saved: {out_csv}")

        # Log key stats
        _log_cmh_stats(results_df, stats, spec_tag)

        # Print Mundlak Hausman diagnostic
        _print_hausman_diagnostic(results_df, spec_tag)

    return results_df


# ====================================================================
#  3C. BINOMIAL SIGNIFICANCE TESTS
# ====================================================================

def _run_binomial_tests(results_df, stats, spec_tag, s3_main):
    """
    For each result row, test whether the fraction of significant taxa
    exceeds the 5% false-positive rate expected under the null.
    """
    if results_df.empty:
        return pd.DataFrame()

    NULL_RATE = 0.05
    binom_rows = []

    for _, row in results_df.iterrows():
        total = int(row["total_taxa"])
        for direction, key in [("Depleted", "depleted_count"),
                               ("Enriched", "enriched_count")]:
            k = int(row[key])
            result = binomtest(k, total, NULL_RATE, alternative='greater')
            p = result.pvalue
            sig = ""
            if   p < 0.001: sig = "***"
            elif p < 0.01:  sig = "**"
            elif p < 0.05:  sig = "*"
            elif p < 0.1:   sig = "."

            binom_rows.append({
                "Condition": row["Condition"],
                "Method": row["Method"],
                "Taxon_list": row.get("Taxon_list", row.get("list_name", "")),
                "Direction": direction,
                "k": k, "n": total,
                "Fraction": round(k / total, 4) if total > 0 else 0,
                "Binom_p": round(p, 6),
                "Sig": sig,
            })

    binom_df = pd.DataFrame(binom_rows)
    if not binom_df.empty:
        binom_df.to_csv(
            os.path.join(s3_main,
                         f"health_binomial_tests_{spec_tag}.csv"),
            index=False)

        # Print significant results summary
        sig_mask = binom_df["Binom_p"] < 0.05
        n_sig = sig_mask.sum()
        print(f"\n    Binomial tests: {n_sig}/{len(binom_df)} "
              f"above 5% chance level")

        for _, r in binom_df[sig_mask].sort_values(
                ["Direction", "Condition", "Taxon_list"]).iterrows():
            print(f"      {r['Condition']:<22s} {r['Taxon_list']:<9s} "
                  f"{r['Direction']:<10s} "
                  f"{r['k']:>3d}/{r['n']} = {r['Fraction']:.1%}  "
                  f"p={r['Binom_p']:.2e} {r['Sig']}")

    return binom_df


# ====================================================================
#  3C-bis. VANISH-vs-BloSSUM CONTINGENCY TESTS
# ====================================================================

def _run_list_contrast_tests(results_df, stats, spec_tag, s3_main):
    """
    For each condition x method, test whether VANISH and BloSSUM taxa
    have significantly different proportions of depleted/enriched taxa.

    Uses a 2x2 contingency table:
                   Depleted   Not-depleted
        VANISH  [   a,           b        ]
        BloSSUM [   c,           d        ]

    And similarly for enriched.  Tests via Fisher's exact (small n)
    or chi-squared (large n).  A significant result means the two
    lists behave differently for that condition — e.g. VANISH taxa
    are preferentially depleted while BloSSUM taxa are not.

    Also tests VANISH+ vs BloSSUM+ when both are present.
    """
    if results_df.empty:
        return pd.DataFrame()

    # Pairs to compare
    LIST_PAIRS = [("VANISH", "BloSSUM"), ("VANISH+", "BloSSUM+")]
    methods = results_df["Method"].unique()
    conditions = results_df["Condition"].unique()

    rows = []
    for list_a, list_b in LIST_PAIRS:
        for method in methods:
            for cond in conditions:
                row_a = results_df[
                    (results_df["Taxon_list"] == list_a) &
                    (results_df["Method"] == method) &
                    (results_df["Condition"] == cond)]
                row_b = results_df[
                    (results_df["Taxon_list"] == list_b) &
                    (results_df["Method"] == method) &
                    (results_df["Condition"] == cond)]

                if row_a.empty or row_b.empty:
                    continue

                ra = row_a.iloc[0]
                rb = row_b.iloc[0]

                for direction in ["depleted", "enriched"]:
                    k_a = int(ra[f"{direction}_count"])
                    n_a = int(ra["total_taxa"])
                    k_b = int(rb[f"{direction}_count"])
                    n_b = int(rb["total_taxa"])

                    # 2x2 table: [k, n-k] for each list
                    table = np.array([[k_a, n_a - k_a],
                                      [k_b, n_b - k_b]])

                    # Skip if both zero
                    if k_a + k_b == 0:
                        continue

                    # Use Fisher's exact for small samples, chi2 otherwise
                    if min(n_a, n_b) < 30 or table.min() < 5:
                        _, p = scipy_fisher(table)
                        test_used = "Fisher"
                    else:
                        chi2, p, _, _ = chi2_contingency(table,
                                                          correction=True)
                        test_used = "Chi2"

                    frac_a = k_a / n_a if n_a > 0 else 0
                    frac_b = k_b / n_b if n_b > 0 else 0

                    sig = ""
                    if   p < 0.001: sig = "***"
                    elif p < 0.01:  sig = "**"
                    elif p < 0.05:  sig = "*"

                    rows.append({
                        "Condition": cond,
                        "Method": method,
                        "Direction": direction,
                        "List_A": list_a,
                        "List_B": list_b,
                        "k_A": k_a, "n_A": n_a, "frac_A": round(frac_a, 4),
                        "k_B": k_b, "n_B": n_b, "frac_B": round(frac_b, 4),
                        "frac_diff": round(frac_a - frac_b, 4),
                        "test": test_used,
                        "p_value": round(p, 6),
                        "sig": sig,
                    })

    contrast_df = pd.DataFrame(rows)
    if not contrast_df.empty:
        fp = os.path.join(s3_main,
                          f"health_list_contrast_{spec_tag}.csv")
        contrast_df.to_csv(fp, index=False)
        print(f"\n    Saved: {fp}")

        # Print significant contrasts
        sig_mask = contrast_df["p_value"] < 0.05
        n_sig = sig_mask.sum()
        print(f"\n    VANISH-vs-BloSSUM contrasts: {n_sig}/{len(contrast_df)} "
              f"significant (p<0.05)")
        if n_sig > 0:
            print(f"    {'Condition':<22s} {'Method':<22s} {'Dir':<10s} "
                  f"{'ListA':<9s} {'ListB':<9s} "
                  f"{'fracA':>6s} {'fracB':>6s} {'D':>7s} {'p':>10s}")
            for _, r in contrast_df[sig_mask].sort_values("p_value").iterrows():
                print(f"    {r['Condition']:<22s} {r['Method']:<22s} "
                      f"{r['Direction']:<10s} "
                      f"{r['List_A']:<9s} {r['List_B']:<9s} "
                      f"{r['frac_A']:>6.1%} {r['frac_B']:>6.1%} "
                      f"{r['frac_diff']:>+7.1%} "
                      f"{r['p_value']:>10.2e} {r['sig']}")

        # Log key contrasts to stats
        for _, r in contrast_df[sig_mask].iterrows():
            tag = (f"contrast_{r['List_A']}_vs_{r['List_B']}_"
                   f"{r['Condition']}_{r['Direction']}_{spec_tag}")
            stats.add(tag, "s3", "main",
                      f"{r['List_A']} vs {r['List_B']} {r['Direction']} "
                      f"in {r['Condition']} ({r['Method']}): "
                      f"{r['frac_A']:.1%} vs {r['frac_B']:.1%}",
                      f"D={r['frac_diff']:+.1%}, p={r['p_value']:.2e}",
                      test=r["test"], p_value=r["p_value"])

    return contrast_df


# ====================================================================
#  3E. FILTERED PCoA  (VANISH+/BloSSUM+ taxa only)
# ====================================================================

def _run_filtered_pcoa(data, dm, taxon_lists, stats, s3_main):
    """
    PCoA on triplet-matched NI x IndH x IndU using only VANISH+/BloSSUM+
    taxa (renormalised to relative abundance).

    Rationale: filtering to the taxa that differ between Industrial and
    Non-Industrial, then looking at healthy vs unhealthy, may better
    reveal the 'intermediate' state of healthy industrialised individuals
    compared to the full-community PCoA.
    """
    ref = data.ref_name

    ids_ni = [s for s in dm.triplet_ni if s in data.abundance.index]
    ids_h  = [s for s in dm.triplet_ind_h if s in data.abundance.index]
    ids_u  = [s for s in dm.triplet_ind_u if s in data.abundance.index]

    if not ids_ni or not ids_h or not ids_u:
        print("    [filtered PCoA] Missing triplet samples — skipping")
        return

    all_ids = list(ids_ni) + list(ids_h) + list(ids_u)

    # Build filter sets
    filter_sets = OrderedDict()

    # VANISH + BloSSUM combined
    vb_combined = set()
    if "VANISH" in taxon_lists:
        vb_combined |= set(taxon_lists["VANISH"])
    if "BloSSUM" in taxon_lists:
        vb_combined |= set(taxon_lists["BloSSUM"])
    if vb_combined:
        filter_sets["VANISH_BloSSUM"] = sorted(
            t for t in vb_combined if t in data.abundance.columns)

    # VANISH+ + BloSSUM+ combined
    vbp_combined = set()
    if "VANISH+" in taxon_lists:
        vbp_combined |= set(taxon_lists["VANISH+"])
    if "BloSSUM+" in taxon_lists:
        vbp_combined |= set(taxon_lists["BloSSUM+"])
    if vbp_combined:
        filter_sets["VANISHplus_BloSSUMplus"] = sorted(
            t for t in vbp_combined if t in data.abundance.columns)

    # VanishH+ + BlossumH+ combined
    vbh_combined = set()
    if "VanishH+" in taxon_lists:
        vbh_combined |= set(taxon_lists["VanishH+"])
    if "BlossumH+" in taxon_lists:
        vbh_combined |= set(taxon_lists["BlossumH+"])
    if vbh_combined:
        filter_sets["VanishH_BlossumH"] = sorted(
            t for t in vbh_combined if t in data.abundance.columns)

    # VANISH only
    if "VANISH" in taxon_lists:
        v_only = [t for t in taxon_lists["VANISH"]
                  if t in data.abundance.columns]
        if len(v_only) >= 10:
            filter_sets["VANISH_only"] = v_only

    from scipy.spatial.distance import pdist, squareform

    for label, taxa in filter_sets.items():
        if len(taxa) < 10:
            print(f"    [filtered PCoA] {label}: only {len(taxa)} taxa — "
                  f"skipping")
            continue

        print(f"    [filtered PCoA] {label}: {len(taxa)} taxa")

        # Subset and renormalise
        abund_sub = data.abundance.loc[all_ids, taxa].copy()
        row_sums = abund_sub.sum(axis=1)
        row_sums_safe = row_sums.replace(0, np.nan)
        abund_renorm = abund_sub.div(row_sums_safe, axis=0).fillna(0.0)

        vals = np.nan_to_num(abund_renorm.values, nan=0.0)

        # Bray-Curtis + PCoA
        D = pdist(vals, metric="braycurtis")
        D = np.nan_to_num(D, nan=0.0)
        D_sq = squareform(D)

        n = len(all_ids)
        D2 = D_sq ** 2
        row_means = D2.mean(axis=1)
        col_means = D2.mean(axis=0)
        grand_mean = D2.mean()
        B = -0.5 * (D2 - row_means[:, np.newaxis] -
                     col_means[np.newaxis, :] + grand_mean)

        n_axes = 3
        try:
            from scipy.linalg import eigh as scipy_eigh
            eigenvalues, eigenvectors = scipy_eigh(
                B, subset_by_index=[n - n_axes, n - 1])
            eigenvalues = eigenvalues[::-1]
            eigenvectors = eigenvectors[:, ::-1]
        except (ImportError, TypeError):
            eigenvalues, eigenvectors = np.linalg.eigh(B)
            idx_sort = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx_sort[:n_axes]]
            eigenvectors = eigenvectors[:, idx_sort[:n_axes]]

        total_var = max(np.trace(B), 1e-10)
        var_explained = np.array([
            max(ev, 0) / total_var * 100 for ev in eigenvalues])
        top_ev = np.maximum(eigenvalues, 0)
        coords = eigenvectors * np.sqrt(top_ev)

        print(f"      PCo1: {var_explained[0]:.1f}%, "
              f"PCo2: {var_explained[1]:.1f}%")

        pcoa_df = pd.DataFrame(
            coords, index=all_ids,
            columns=[f"PCo{i+1}" for i in range(n_axes)])

        n_ni, n_h, n_u = len(ids_ni), len(ids_h), len(ids_u)
        pcoa_df["Health_group"] = (
            ["Non-Industrial"] * n_ni +
            ["Ind-Healthy"] * n_h +
            ["Ind-Unhealthy"] * n_u)
        pcoa_df.to_csv(
            os.path.join(s3_main,
                         f"health_pcoa_filtered_{label}.csv"),
            index=False)

        # PERMANOVA
        from .section1_diversity import _permanova
        binary_labels = np.array(pcoa_df["Health_group"].tolist())
        try:
            F_obs, p_perm = _permanova(
                D_sq, binary_labels, n_perm=999)
            print(f"      PERMANOVA: F={F_obs:.1f}, p={p_perm:.3f}")
            stats.add(f"filtered_pcoa_permanova_{label}", "s3", "main",
                      f"Filtered PCoA PERMANOVA ({label}, {len(taxa)} taxa)",
                      f"F={F_obs:.1f}", test="PERMANOVA", p_value=p_perm,
                      n1=n_ni, n2=n_h)
        except Exception as e:
            F_obs, p_perm = np.nan, np.nan
            print(f"      PERMANOVA failed: {e}")

        # Pairwise distance quantification + intermediacy test
        dist_results = _quantify_group_distances(
            D_sq, ids_ni, ids_h, ids_u, all_ids,
            label, ref, stats, s3_main)

        # Plot
        _plot_filtered_pcoa(pcoa_df, var_explained, label,
                            len(taxa), F_obs, p_perm, ref, s3_main,
                            dist_results=dist_results)


def _quantify_group_distances(D_sq, ids_ni, ids_h, ids_u, all_ids,
                               label, ref, stats, s3_main):
    """
    Compute pairwise Bray-Curtis distances between the three health groups
    and test the intermediacy hypothesis: is Ind-Healthy closer to
    Non-Industrial than Ind-Unhealthy is?

    Returns a dict with results for annotation on the PCoA plot.
    """
    from .section1_diversity import _permanova

    idx_map = {sid: i for i, sid in enumerate(all_ids)}
    ni_idx = np.array([idx_map[s] for s in ids_ni])
    h_idx  = np.array([idx_map[s] for s in ids_h])
    u_idx  = np.array([idx_map[s] for s in ids_u])

    def _between_dists(idx_a, idx_b):
        """Extract all pairwise distances between two index sets."""
        return D_sq[np.ix_(idx_a, idx_b)].ravel()

    def _within_dists(idx_a):
        """Extract all pairwise distances within one index set (upper tri)."""
        sub = D_sq[np.ix_(idx_a, idx_a)]
        return sub[np.triu_indices(len(idx_a), k=1)]

    # ── Pairwise mean/median distances ──
    pairs = [
        ("NI_vs_IndH", ni_idx, h_idx),
        ("NI_vs_IndU", ni_idx, u_idx),
        ("IndH_vs_IndU", h_idx, u_idx),
    ]
    within_groups = [
        ("within_NI", ni_idx),
        ("within_IndH", h_idx),
        ("within_IndU", u_idx),
    ]

    dist_rows = []
    dist_summary = {}

    for pair_name, ia, ib in pairs:
        d = _between_dists(ia, ib)
        dist_summary[pair_name] = {
            "mean": np.mean(d), "median": np.median(d), "std": np.std(d)}
        dist_rows.append({
            "comparison": pair_name, "type": "between",
            "mean_BC": np.mean(d), "median_BC": np.median(d),
            "std_BC": np.std(d), "n_pairs": len(d)})

    for grp_name, ia in within_groups:
        d = _within_dists(ia)
        dist_rows.append({
            "comparison": grp_name, "type": "within",
            "mean_BC": np.mean(d), "median_BC": np.median(d),
            "std_BC": np.std(d), "n_pairs": len(d)})

    print(f"\n      Pairwise Bray-Curtis distances ({label}):")
    print(f"        {'Comparison':<18s} {'Mean':>7s} {'Median':>7s} {'SD':>7s}")
    for r in dist_rows:
        print(f"        {r['comparison']:<18s} "
              f"{r['mean_BC']:>7.4f} {r['median_BC']:>7.4f} "
              f"{r['std_BC']:>7.4f}")

    # ── Pairwise PERMANOVA (each pair) ──
    print(f"\n      Pairwise PERMANOVA ({label}):")
    pairwise_perm = {}
    for pair_name, ia, ib in pairs:
        sub_idx = np.concatenate([ia, ib])
        sub_D = D_sq[np.ix_(sub_idx, sub_idx)]
        sub_labels = np.array(
            ["A"] * len(ia) + ["B"] * len(ib))
        try:
            F_pw, p_pw = _permanova(sub_D, sub_labels, n_perm=999)
            pairwise_perm[pair_name] = {"F": F_pw, "p": p_pw}
            print(f"        {pair_name:<18s} F={F_pw:>7.1f}  p={p_pw:.3f}")
            stats.add(
                f"filtered_pairwise_{pair_name}_{label}", "s3", "main",
                f"Pairwise PERMANOVA {pair_name} ({label})",
                f"F={F_pw:.1f}", test="PERMANOVA", p_value=p_pw)
        except Exception as e:
            pairwise_perm[pair_name] = {"F": np.nan, "p": np.nan}
            print(f"        {pair_name:<18s} failed: {e}")

    # ── Intermediacy test ──
    # H0: d(IndH, NI) = d(IndU, NI)
    # H1: d(IndH, NI) < d(IndU, NI)  (healthy is closer to NI)
    #
    # Test statistic: D = mean d(IndU, NI) − mean d(IndH, NI)
    # Positive D means healthy is closer to NI (intermediate).
    # Permutation: shuffle H/U labels among industrial samples.

    d_h_ni = _between_dists(h_idx, ni_idx)
    d_u_ni = _between_dists(u_idx, ni_idx)
    delta_obs = np.mean(d_u_ni) - np.mean(d_h_ni)

    n_perm = 9999
    rng = np.random.default_rng(42)
    ind_all = np.concatenate([h_idx, u_idx])
    n_h = len(h_idx)
    count_ge = 1  # include observed

    for _ in range(n_perm):
        perm = rng.permutation(ind_all)
        perm_h = perm[:n_h]
        perm_u = perm[n_h:]
        d_ph = D_sq[np.ix_(perm_h, ni_idx)].mean()
        d_pu = D_sq[np.ix_(perm_u, ni_idx)].mean()
        if (d_pu - d_ph) >= delta_obs:
            count_ge += 1

    p_intermediacy = count_ge / (n_perm + 1)

    # Effect size: ratio d(IndH,NI) / d(IndU,NI)
    ratio = np.mean(d_h_ni) / np.mean(d_u_ni) if np.mean(d_u_ni) > 0 else 1.0

    print(f"\n      Intermediacy test ({label}):")
    print(f"        mean d(IndH, NI) = {np.mean(d_h_ni):.4f}")
    print(f"        mean d(IndU, NI) = {np.mean(d_u_ni):.4f}")
    print(f"        D = d(IndU,NI) − d(IndH,NI) = {delta_obs:.4f}")
    print(f"        ratio d(IndH,NI)/d(IndU,NI) = {ratio:.4f}")
    print(f"        p(H1: healthy closer to NI) = {p_intermediacy:.4f}  "
          f"({n_perm} permutations)")

    sig = ""
    if   p_intermediacy < 0.001: sig = "***"
    elif p_intermediacy < 0.01:  sig = "**"
    elif p_intermediacy < 0.05:  sig = "*"

    stats.add(
        f"intermediacy_{label}", "s3", "main",
        f"Intermediacy test ({label}): "
        f"d(IndH,NI)={np.mean(d_h_ni):.4f} vs "
        f"d(IndU,NI)={np.mean(d_u_ni):.4f}",
        f"D={delta_obs:.4f}, ratio={ratio:.4f}, "
        f"p={p_intermediacy:.4f}{sig}",
        test="Permutation (intermediacy)", p_value=p_intermediacy,
        effect_size=delta_obs)

    # Also test: d(IndH, IndU) — how different are the two industrial groups?
    d_h_u = _between_dists(h_idx, u_idx)
    print(f"        mean d(IndH, IndU) = {np.mean(d_h_u):.4f}")

    # ── Save CSV ──
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(
        os.path.join(s3_main,
                     f"health_pcoa_distances_{label}.csv"),
        index=False)

    # Intermediacy row
    interm_df = pd.DataFrame([{
        "label": label,
        "mean_d_IndH_NI": np.mean(d_h_ni),
        "mean_d_IndU_NI": np.mean(d_u_ni),
        "mean_d_IndH_IndU": np.mean(d_h_u),
        "delta": delta_obs,
        "ratio_H_over_U": ratio,
        "p_intermediacy": p_intermediacy,
        "n_perm": n_perm,
    }])
    interm_df.to_csv(
        os.path.join(s3_main,
                     f"health_pcoa_intermediacy_{label}.csv"),
        index=False)
    print(f"      Saved: health_pcoa_distances_{label}.csv, "
          f"health_pcoa_intermediacy_{label}.csv")

    return {
        "dist_summary": dist_summary,
        "pairwise_perm": pairwise_perm,
        "delta": delta_obs,
        "ratio": ratio,
        "p_intermediacy": p_intermediacy,
        "mean_d_h_ni": np.mean(d_h_ni),
        "mean_d_u_ni": np.mean(d_u_ni),
        "mean_d_h_u": np.mean(d_h_u),
    }


def _plot_filtered_pcoa(pcoa_df, var_explained, label, n_taxa,
                         F_obs, p_perm, ref, outdir,
                         dist_results=None):
    """
    Health PCoA restricted to VANISH/BloSSUM taxa.

    Groups: Non-Industrial (orange), Ind-Healthy (green), Ind-Unhealthy (blue).
    Includes 2D KDE contours for each group.
    """
    import matplotlib as _mpl
    _saved = {k: _mpl.rcParams.get(k) for k in [
        "font.size", "axes.linewidth", "pdf.fonttype", "ps.fonttype",
        "font.family", "font.sans-serif"]}
    _mpl.rcParams.update({
        "font.size": 9, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    })

    fig, ax = plt.subplots(figsize=page_figsize())

    group_style = OrderedDict([
        ("Ind-Unhealthy",  {"color": "#0072B280", "ec": "none",
                            "contour": config.OI["blue"], "label_c": config.OI["blue"]}),
        ("Ind-Healthy",    {"color": "#009E7380", "ec": "none",
                            "contour": config.OI["green"], "label_c": config.OI["green"]}),
        ("Non-Industrial", {"color": "#E69F0080", "ec": "none",
                            "contour": config.OI["orange"], "label_c": config.OI["orange"]}),
    ])

    for grp, style in group_style.items():
        sub = pcoa_df[pcoa_df["Health_group"] == grp]
        if sub.empty:
            continue
        ax.scatter(sub["PCo1"], sub["PCo2"],
                   c=style["color"], s=8, alpha=0.5,
                   edgecolors="none", zorder=2, rasterized=True)

        # KDE contours
        if len(sub) > 30:
            try:
                from scipy.stats import gaussian_kde
                xy = np.vstack([sub["PCo1"].values, sub["PCo2"].values])
                kde = gaussian_kde(xy, bw_method=0.3)
                xg = np.linspace(sub["PCo1"].min() - 0.05,
                                 sub["PCo1"].max() + 0.05, 80)
                yg = np.linspace(sub["PCo2"].min() - 0.05,
                                 sub["PCo2"].max() + 0.05, 80)
                X, Y = np.meshgrid(xg, yg)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                ax.contour(X, Y, Z, levels=3, colors=style["contour"],
                           linewidths=0.8, alpha=0.6)
            except Exception:
                pass

        ax.scatter([], [], c=style["label_c"], s=30,
                   label=f"{grp} (n={len(sub):,})")

    # PERMANOVA annotation
    if not np.isnan(F_obs):
        ax.text(0.02, 0.02,
                f"PERMANOVA\nF = {F_obs:.1f}, p = {p_perm:.3f}",
                transform=ax.transAxes, fontsize=FS-2, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="gray", alpha=0.9))

    # Distance & intermediacy annotation
    if dist_results is not None:
        d_h_ni = dist_results["mean_d_h_ni"]
        d_u_ni = dist_results["mean_d_u_ni"]
        d_h_u  = dist_results["mean_d_h_u"]
        delta  = dist_results["delta"]
        p_int  = dist_results["p_intermediacy"]
        sig = ""
        if   p_int < 0.001: sig = "***"
        elif p_int < 0.01:  sig = "**"
        elif p_int < 0.05:  sig = "*"

        dist_text = (
            f"Mean Bray-Curtis distances\n"
            f"  d(IndH, NI)   = {d_h_ni:.4f}\n"
            f"  d(IndU, NI)   = {d_u_ni:.4f}\n"
            f"  d(IndH, IndU) = {d_h_u:.4f}\n"
            f"D = {delta:+.4f},  p = {p_int:.4f}{sig}\n"
            f"(H₁: healthy closer to NI)"
        )
        ax.text(0.02, 0.98, dist_text,
                transform=ax.transAxes, fontsize=FS-2, va="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                          ec="gray", alpha=0.9))

    ax.set_xlabel(f"PCo1 ({var_explained[0]:.1f}%)", fontsize=FS)
    ax.set_ylabel(f"PCo2 ({var_explained[1]:.1f}%)", fontsize=FS)
    ax.set_title(f"Health PCoA — {label.replace('_', ' ')} taxa only\n"
                 f"({n_taxa} taxa, renormalised, Bray-Curtis)",
                 fontsize=FS+1, fontweight="bold")
    ax.legend(fontsize=FS-2, loc="upper right", framealpha=0.9,
              edgecolor="grey")
    ax.grid(alpha=0.15)

    plt.tight_layout()
    fp = os.path.join(outdir,
                      f"health_pcoa_filtered_{label}_{ref}.png")
    savefig_dual(fig, fp, dpi=300)
    plt.close(fig)
    print(f"    Saved: {fp}")

    _mpl.rcParams.update(_saved)


# ====================================================================
#  3D. SUMMARY FIGURE
# ====================================================================

LIST_ORDER  = ["VANISH", "BloSSUM", "VANISH+", "BloSSUM+",
               "VanishH+", "BlossumH+"]
LIST_COLORS = config.CMH_LIST_COLORS


def _plot_summary_figure(results_df, ref, s3_main, spec_tag, spec_desc):
    """
    Bidirectional bar chart using GMW-CMH as primary method.
    ↑ = enriched in condition (taxa blooming in disease)
    ↓ = depleted in condition (taxa lost in disease)
    """
    if results_df.empty:
        return

    best = "Geom-mean-weighted CMH"
    cmh_df = results_df[results_df["Method"] == best]
    if cmh_df.empty:
        best = "Within-study CMH"
        cmh_df = results_df[results_df["Method"] == best]
    if cmh_df.empty:
        best = "Depth-matched CMH"
        cmh_df = results_df[results_df["Method"] == best]
    if cmh_df.empty:
        return

    # Filter out conditions not suitable for plotting
    cond_order = [c for c in dict.fromkeys(results_df["Condition"])
                  if not _condition_dropped(c)]
    cond_order = _display_order(cond_order)
    available_lists = [l for l in LIST_ORDER
                       if l in cmh_df["Taxon_list"].values]
    n_lists = len(available_lists)
    n_cond = len(cond_order)

    if n_lists == 0 or n_cond == 0:
        return

    fig, ax = plt.subplots(figsize=page_figsize(height_in=3.6))
    bw = 0.17
    x_base = np.arange(n_cond)

    for i, ln in enumerate(available_lists):
        xp = x_base + i * bw - (n_lists * bw) / 2 + bw / 2
        color = LIST_COLORS.get(ln, "#888888")

        for cond_i, cond in enumerate(cond_order):
            row = cmh_df[(cmh_df["Taxon_list"] == ln) &
                         (cmh_df["Condition"] == cond)]
            if row.empty:
                continue
            r = row.iloc[0]

            # Enriched in condition -> UP (positive)
            if r["enriched_frac"] > 0:
                ax.bar(xp[cond_i], r["enriched_frac"], bw * 0.9,
                       color=color, alpha=0.85,
                       edgecolor='white', lw=0.5)
                ax.errorbar(xp[cond_i], r["enriched_frac"],
                            yerr=[[r["enriched_ci_lo"]],
                                  [r["enriched_ci_hi"]]],
                            fmt='none', color='black',
                            capsize=1.5, lw=0.5)

            # Depleted in condition -> DOWN (negative)
            if r["depleted_frac"] > 0:
                ax.bar(xp[cond_i], -r["depleted_frac"], bw * 0.9,
                       color=color, alpha=0.85,
                       edgecolor='white', lw=0.5)
                ax.errorbar(xp[cond_i], -r["depleted_frac"],
                            yerr=[[r["depleted_ci_hi"]],
                                  [r["depleted_ci_lo"]]],
                            fmt='none', color='black',
                            capsize=1.5, lw=0.5)

    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(0.05, color='gray', lw=0.5, ls=':', alpha=0.4)
    ax.axhline(-0.05, color='gray', lw=0.5, ls=':', alpha=0.4)

    ax.set_xticks(x_base)
    ax.set_xticklabels(cond_order, rotation=35, ha='right', fontsize=FS)
    # Kept: the direction cue, what is plotted, the threshold. Dropped: "in
    # condition" (the x-axis is conditions) and the method name (stated verbatim
    # in the title directly above). The old label ran longer than the panel is
    # tall and was clipped at both ends.
    ax.set_ylabel(
        r"$\leftarrow$ Depleted    Enriched $\rightarrow$"
        "\nfraction sig. (p < 0.05)", fontsize=FS-1)
    ax.set_title(
        f"Health Associations — {best}\n"
        f"Detection: {spec_desc}", fontsize=FS+2, fontweight='bold')

    # Symmetric y-limits
    ymax = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.15
    ax.set_ylim(-ymax, ymax)

    # Error-bar annotation (Wilson score 95% CI)
    ax.text(0.01, 0.01,
            "Error bars: Wilson score 95% CI on fraction",
            transform=ax.transAxes, fontsize=FS-2, color="gray",
            va="bottom", ha="left", style="italic")

    # Separator between main comparisons and diseases
    for _b in _group_dividers(cond_order):
        ax.axvline(_b, color='lightgray', ls='--', lw=1)

    leg = [Patch(fc=LIST_COLORS.get(n, "#888"), ec='white', label=n)
           for n in available_lists]
    ax.legend(handles=leg, loc='upper right', fontsize=FS-2, framealpha=.9)
    ax.grid(axis='y', alpha=.2)
    plt.tight_layout()

    out = os.path.join(s3_main,
                       f"health_bidirectional_{ref}_{spec_tag}.png")
    savefig_dual(fig, out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out}")


def _plot_tier_comparison(results_df, ref, s3_main, spec_tag, spec_desc,
                          binom_df=None):
    """
    Single bidirectional chart: Within-study CMH vs Depth-matched CMH
    vs Geom-mean-weighted CMH vs Mundlak GLMM.
    Enriched in condition -> UP (positive y)
    Depleted in condition -> DOWN (negative y)

    Bar width is calculated dynamically so bars never overlap across conditions.
    Error bars always shown regardless of bar height.
    """
    if results_df.empty:
        return

    tiers_present = results_df["Method"].unique().tolist()
    if len(tiers_present) < 2:
        return

    cond_order = [c for c in dict.fromkeys(results_df["Condition"])
                  if not _condition_dropped(c)]
    cond_order = _display_order(cond_order)
    available_lists = [l for l in LIST_ORDER
                       if l in results_df["Taxon_list"].values]
    n_lists = len(available_lists)
    n_tiers = len(tiers_present)
    n_cond  = len(cond_order)

    if n_lists == 0 or n_cond == 0:
        return

    # GMW-CMH primary (solid); WS-CMH reference (light); DM dotted; Mundlak diagonal
    tier_alpha = {"Within-study CMH": 0.35, "Depth-matched CMH": 0.45,
                  "Geom-mean-weighted CMH": 0.92, "Mundlak GLMM": 0.65}
    tier_hatch = {"Within-study CMH": "///",  "Depth-matched CMH": "...",
                  "Geom-mean-weighted CMH": "", "Mundlak GLMM": "xx"}
    tier_edge  = {"Within-study CMH": "#999",  "Depth-matched CMH": "#666",
                  "Geom-mean-weighted CMH": "black", "Mundlak GLMM": "#336"}

    total_bars = n_lists * n_tiers
    # Dynamic bar width: fit all bars within 0.82 of the 1-unit condition spacing
    bw = min(0.065, 0.82 / max(total_bars, 1))

    # ── Binomial lookup helper ─────────────────────────────────────────
    def _binom_sig(condition, list_name, method, direction):
        if binom_df is None or binom_df.empty:
            return ""
        mask = (
            (binom_df["Condition"]   == condition) &
            (binom_df["Taxon_list"]  == list_name) &
            (binom_df["Method"]      == method) &
            (binom_df["Direction"]   == direction)
        )
        rows = binom_df[mask]
        if rows.empty:
            return ""
        return rows.iloc[0].get("Sig", "")

    # Figure width: ensure enough pixels per bar
    fig, ax = plt.subplots(figsize=page_figsize(height_in=4.0))

    for ci, cond in enumerate(cond_order):
        csub = results_df[results_df["Condition"] == cond]
        bar_i = 0
        for li, ln in enumerate(available_lists):
            for ti, tn in enumerate(tiers_present):
                rmatch = csub[(csub["Taxon_list"] == ln) &
                              (csub["Method"] == tn)]
                x = ci + (bar_i - (total_bars - 1) / 2) * bw
                if rmatch.empty:
                    bar_i += 1
                    continue
                r = rmatch.iloc[0]
                color = LIST_COLORS.get(ln, "#888")
                alpha = tier_alpha.get(tn, 0.7)
                hatch = tier_hatch.get(tn, "")
                ec    = tier_edge.get(tn, "black")

                # ── Enriched -> UP ────────────────────────────────────
                enr_val   = float(r["enriched_frac"])
                enr_ci_lo = float(r["enriched_ci_lo"])
                enr_ci_hi = float(r["enriched_ci_hi"])
                enr_sig   = _binom_sig(cond, ln, tn, "Enriched")

                if enr_val > 0 or enr_sig:
                    ax.bar(x, max(enr_val, 0), bw * 0.88,
                           bottom=0, color=color, alpha=alpha,
                           edgecolor=ec, lw=0.5, hatch=hatch)
                # Always show error bar if there's any enriched signal
                if enr_val > 0:
                    ax.errorbar(x, enr_val,
                                yerr=[[enr_ci_lo], [enr_ci_hi]],
                                fmt='none', color='black',
                                capsize=1.2, lw=0.6)
                    count_y = enr_val + enr_ci_hi + 0.005
                    ax.text(x, count_y,
                            str(int(r["enriched_count"])),
                            ha='center', va='bottom',
                            fontsize=FS-2, color=color, rotation=90)
                    if enr_sig:
                        ax.text(x, count_y + 0.022, enr_sig,
                                ha='center', va='bottom',
                                fontsize=FS-2, color='black',
                                fontweight='bold', rotation=90)

                # ── Depleted -> DOWN ──────────────────────────────────
                dep_val   = float(r["depleted_frac"])
                dep_ci_lo = float(r["depleted_ci_lo"])
                dep_ci_hi = float(r["depleted_ci_hi"])
                dep_sig   = _binom_sig(cond, ln, tn, "Depleted")

                if dep_val > 0 or dep_sig:
                    ax.bar(x, -max(dep_val, 0), bw * 0.88,
                           bottom=0, color=color, alpha=alpha,
                           edgecolor=ec, lw=0.5, hatch=hatch)
                if dep_val > 0:
                    ax.errorbar(x, -dep_val,
                                yerr=[[dep_ci_hi], [dep_ci_lo]],
                                fmt='none', color='black',
                                capsize=1.2, lw=0.6)
                    count_y = -dep_val - dep_ci_hi - 0.005
                    ax.text(x, count_y,
                            str(int(r["depleted_count"])),
                            ha='center', va='top',
                            fontsize=FS-2, color=color, rotation=90)
                    if dep_sig:
                        ax.text(x, count_y - 0.022, dep_sig,
                                ha='center', va='top',
                                fontsize=FS-2, color='black',
                                fontweight='bold', rotation=90)

                bar_i += 1

    ax.axhline(0,     color='black', lw=0.8)
    ax.axhline( 0.05, color='gray',  lw=0.5, ls=':', alpha=0.5)
    ax.axhline(-0.05, color='gray',  lw=0.5, ls=':', alpha=0.5)

    for _b in _group_dividers(cond_order):
        ax.axvline(_b, color='lightgray', ls='--', lw=1, zorder=0)

    ax.set_xticks(range(n_cond))
    ax.set_xticklabels(cond_order, rotation=35, ha='right', fontsize=FS)

    ax.set_ylabel(
        "Depleted in condition (down) / Enriched in condition (up)\n"
        "(fraction of taxa sig. at p < 0.05; error bars always shown)",
        fontsize=FS-1)

    ymax = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.28
    ax.set_ylim(-ymax, ymax)
    ax.grid(axis='y', alpha=0.15)

    leg = []
    for n in available_lists:
        leg.append(Patch(fc=LIST_COLORS.get(n, "#888"), ec='white', label=n))
    for tn in tiers_present:
        leg.append(Patch(
            fc='gray',
            alpha=tier_alpha.get(tn, 0.7),
            ec=tier_edge.get(tn, "black"),
            hatch=tier_hatch.get(tn, ""),
            label=tn))
    if binom_df is not None and not binom_df.empty:
        from matplotlib.lines import Line2D
        leg.append(Line2D([], [], color='none', marker='$*$',
                          markerfacecolor='black', markersize=6,
                          label='Binom. sig. (* / ** / ***)'))
    ax.legend(handles=leg, loc='upper left', fontsize=FS-2,
              framealpha=0.95, ncol=2)

    ax.set_title(
        f"Health Associations: CMH vs Depth-matched CMH"
        f" vs Geom-mean-weighted CMH vs Mundlak GLMM\n"
        f"Detection: {spec_desc}"
        + (" — bars annotated with binomial significance"
           if binom_df is not None else ""),
        fontsize=FS+1, fontweight='bold')
    plt.tight_layout()

    out = os.path.join(s3_main,
                       f"health_tier_comparison_{ref}_{spec_tag}.png")
    savefig_dual(fig, out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out}")


# ====================================================================
#  PREVALENCE GRADIENT PLOT
# ====================================================================

def _plot_prevalence_gradient(data, dm, stats, outdir,
                               bdf, thresh, spec_tag, spec_desc):
    """NI prevalence vs health-association plot."""
    ref = data.ref_name
    ids_ni = [s for s in dm.triplet_ni if s in bdf.index]
    ids_h = [s for s in dm.triplet_ind_h if s in bdf.index]
    ids_u = [s for s in dm.triplet_ind_u if s in bdf.index]

    if not ids_ni or not ids_h or not ids_u:
        return

    prev_ni = (bdf.loc[ids_ni] >= thresh).mean(axis=0)
    prev_ih = (bdf.loc[ids_h] >= thresh).mean(axis=0)
    prev_iu = (bdf.loc[ids_u] >= thresh).mean(axis=0)

    gradient = pd.DataFrame({
        "prev_ni": prev_ni, "prev_ind_healthy": prev_ih,
        "prev_ind_unhealthy": prev_iu,
        "health_diff": prev_ih - prev_iu,
        "lifestyle_diff": prev_ni - prev_ih,
    }).dropna()

    if len(gradient) > 10:
        rho, p = spearmanr(gradient["lifestyle_diff"],
                            gradient["health_diff"])
        stats.add(f"gradient_spearman_{spec_tag}", "s3", "main",
                  f"Spearman: NI-enrichment vs health "
                  f"({spec_desc}, triplet)",
                  f"rho={rho:.3f}", test="Spearman", p_value=p,
                  effect_size=rho)

    gradient.to_csv(os.path.join(outdir,
                                  f"prevalence_gradient_{spec_tag}.csv"),
                    index=False)

    fig, ax = plt.subplots(figsize=page_figsize())
    ax.scatter(gradient["lifestyle_diff"], gradient["health_diff"],
               s=4, alpha=0.15, c="#888888", edgecolors="none")

    for tlist, color, label in [
        (data.vanish_set, config.OI["vermillion"], "VANISH"),
        (data.blossum_set, config.OI["blue"], "BloSSUM"),
    ]:
        mask = gradient.index.isin(tlist)
        sub = gradient[mask]
        if not sub.empty:
            ax.scatter(sub["lifestyle_diff"], sub["health_diff"],
                       s=20, alpha=0.7, c=color, label=label,
                       edgecolors="none")

    ax.axhline(0, color="grey", ls="-", alpha=0.3)
    ax.axvline(0, color="grey", ls="-", alpha=0.3)
    ax.set_xlabel("Non-Industrial enrichment\n"
                  "(prev_NI - prev_IndHealthy)")
    ax.set_ylabel("Health association\n"
                  "(prev_IndHealthy - prev_IndUnhealthy)")
    ax.set_title(f"Prevalence Gradient ({ref})\n"
                 f"Detection: {spec_desc}, triplet-matched")
    ax.legend(fontsize=FS-1, markerscale=2)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    savefig_dual(fig, os.path.join(outdir,
                              f"prevalence_gradient_{ref}_{spec_tag}.png"),
                dpi=300)
    plt.close(fig)
    print(f"    Saved: prevalence_gradient_{ref}_{spec_tag}.png")


# ====================================================================
#  STATS LOGGING
# ====================================================================

def _log_health_stats(results_df, stats, spec_tag):
    """Log key triplet health-association statistics."""
    for _, row in results_df.iterrows():
        cond = row.get("condition", row.get("Condition", ""))
        tlist = row.get("list_name", row.get("Taxon_list", ""))
        dep = row["depleted_count"]
        enr = row["enriched_count"]
        total = row["total_taxa"]
        if total == 0:
            continue

        # The tag keeps the internal key so the manuscript crosswalk still
        # matches on it; the description is what a reader sees, so it takes
        # the published name.
        tag = f"health_{tlist}_{cond}_{spec_tag}"
        stats.add(tag, "s3", "main",
                  f"{tlist} in {config.display_text(cond)} "
                  f"({row.get('method', row.get('Method', ''))}): "
                  f"{dep}/{total} depleted, {enr}/{total} enriched",
                  f"depleted={dep / total:.1%}, enriched={enr / total:.1%}",
                  n1=row.get("n_group1", None),
                  n2=row.get("n_group2", None))


def _log_cmh_stats(results_df, stats, spec_tag):
    """Log key per-condition CMH statistics."""
    # Focus on GMW-CMH as primary method for main-text stats
    best = "Geom-mean-weighted CMH"
    sub = results_df[results_df["Method"] == best]
    if sub.empty:
        best = "Depth-matched CMH"
        sub = results_df[results_df["Method"] == best]
    if sub.empty:
        sub = results_df

    for _, row in sub.iterrows():
        cond = row["Condition"]
        tlist = row["Taxon_list"]
        method = row["Method"]
        dep = int(row["depleted_count"])
        enr = int(row["enriched_count"])
        total = int(row["total_taxa"])
        if total == 0:
            continue

        tag = f"cmh_{tlist}_{cond}_{spec_tag}"
        ns = row.get("n_studies", "")
        ns_str = f", {ns} studies" if ns else ""

        stats.add(tag, "s3", "main",
                  f"{tlist} in {config.display_text(cond)} ({method}{ns_str}): "
                  f"{dep}/{total} depleted, {enr}/{total} enriched",
                  f"depleted={dep / total:.1%}, "
                  f"enriched={enr / total:.1%}",
                  n1=int(row.get("n_group1", 0)),
                  n2=int(row.get("n_group2", 0)))

    # ── Log Mundlak-specific stats (Hausman diagnostic) ──
    mundlak_sub = results_df[results_df["Method"] == "Mundlak GLMM"]
    if not mundlak_sub.empty:
        for _, row in mundlak_sub.iterrows():
            cond = row["Condition"]
            tlist = row["Taxon_list"]
            dep = int(row["depleted_count"])
            enr = int(row["enriched_count"])
            total = int(row["total_taxa"])
            n_haus = row.get("n_hausman_sig", 0)
            m_diff = row.get("median_hausman_diff", np.nan)

            if total == 0:
                continue

            tag = f"mundlak_{tlist}_{cond}_{spec_tag}"
            haus_str = ""
            if isinstance(n_haus, (int, float)) and not np.isnan(n_haus):
                haus_str = f", {int(n_haus)} taxa Hausman-sig"
            if isinstance(m_diff, float) and np.isfinite(m_diff):
                haus_str += f", median D(beta_w-beta_b)={m_diff:.3f}"

            stats.add(tag, "s3", "supplement",
                      f"Mundlak {tlist} in {config.display_text(cond)}: "
                      f"{dep}/{total} depleted, {enr}/{total} enriched "
                      f"(FDR<0.05 on beta_w{haus_str})",
                      f"depleted={dep / total:.1%}, "
                      f"enriched={enr / total:.1%}")


def _print_hausman_diagnostic(results_df, spec_tag):
    """
    Print Hausman diagnostic summary for Mundlak GLMM results.

    If many taxa show beta_w ≠ beta_b, this flags that naive pooling
    (or a vanilla GLMM) would be confounded by between-study
    differences.
    """
    mundlak_rows = results_df[results_df["Method"] == "Mundlak GLMM"]
    if mundlak_rows.empty:
        return

    print(f"\n    {'─' * 60}")
    print(f"    Mundlak beta_w vs beta_b diagnostic ({spec_tag})")
    print(f"    {'─' * 60}")
    print(f"    {'Condition':<22s} {'List':<9s} "
          f"{'Haus sig':>8s} {'med D':>8s} {'conv':>5s}  Flag")
    print(f"    {'-' * 58}")

    for _, row in mundlak_rows.iterrows():
        cond = row["Condition"]
        tlist = row["Taxon_list"]
        n_haus = row.get("n_hausman_sig", 0)
        m_diff = row.get("median_hausman_diff", np.nan)
        total = int(row["total_taxa"])
        conv = row.get("n_converged", 0)
        if not isinstance(conv, (int, float)):
            conv = 0
        conv = int(conv) if np.isfinite(float(conv)) else 0

        if not isinstance(n_haus, (int, float)):
            n_haus = 0
        n_haus = int(n_haus) if np.isfinite(n_haus) else 0

        diff_str = f"{m_diff:+.3f}" if isinstance(m_diff, float) and np.isfinite(m_diff) else "  n/a"

        # Flag conditions where beta_w ≠ beta_b for >20% of taxa
        flag = ""
        if total > 0 and n_haus / total > 0.20:
            flag = "⚠ BETWEEN-STUDY CONFOUNDING"
        elif total > 0 and n_haus / total > 0.10:
            flag = "~ moderate disagreement"

        print(f"    {cond:<22s} {tlist:<9s} "
              f"{n_haus:>5d}/{total:<3d} {diff_str:>8s} {conv:>5d}  {flag}")


# ====================================================================
#  SUMMARY TABLE (console)
# ====================================================================

def _print_summary_table(results_df, spec_tag):
    """Print a compact summary table to console."""
    if results_df.empty:
        return

    tiers = sorted(results_df["Method"].unique())
    cond_order = list(dict.fromkeys(results_df["Condition"]))
    lists_present = [l for l in LIST_ORDER
                     if l in results_df["Taxon_list"].values]

    print(f"\n{'=' * 90}")
    print(f"  HEALTH CMH SUMMARY ({spec_tag})")
    print(f"{'=' * 90}")
    print(f"{'Condition':<22s} {'List':<9s}", end="")
    for tn in tiers:
        short = tn.replace("Within-study ", "WS-").replace(
            "Depth-matched ", "DM-")
        print(f"  {short:>16s}", end="")
    print()
    print("-" * 90)

    for cond in cond_order:
        for ln in lists_present:
            print(f"  {cond:<20s} {ln:<9s}", end="")
            for tn in tiers:
                m = results_df[(results_df["Condition"] == cond) &
                               (results_df["Taxon_list"] == ln) &
                               (results_df["Method"] == tn)]
                if m.empty:
                    print(f"  {'—':>16s}", end="")
                else:
                    r = m.iloc[0]
                    d = int(r["depleted_count"])
                    e = int(r["enriched_count"])
                    t = int(r["total_taxa"])
                    print(f"  {d:>3d}↓ {e:>3d}↑ /{t:<4d}",
                          end="")
            print()


# ====================================================================
#  3B-bis. PER-TAXON GMW DIRECTIONAL ANALYSIS
# ====================================================================

# ====================================================================
#  DEMOGRAPHIC ASSOCIATIONS  (age / sex / BMI, as Figure 5D-5E)
# ====================================================================

def _band_role(field, band):
    """What a band does in the contrasts: reference, stratum, or excluded.

    Table S3 has to be readable without the code beside it, and "used_in_
    contrast: true" cannot distinguish the 18-29 band (which every age
    contrast is measured *against*) from the bands measured against it.
    """
    if field == "Age":
        return ("reference" if band == config.AGE_REFERENCE_BAND
                else "stratum")
    if field == "Gender":
        return "reference" if band == "Male" else "stratum"
    if band == "Underweight":
        return "excluded (distinct phenotype)"
    if band == "Normal":
        return "reference (both BMI contrasts)"
    return "stratum"


def _write_demographic_cohorts(data, s3_main, stats):
    """Cohort sizes and metadata coverage for the demographic strata.

    This is the Table S3 payload. Table S3 defines who is in each cohort, and
    a reader cannot check an age or BMI contrast without knowing how many
    samples carried the field at all -- Age, Gender and BMI are each missing
    for a large minority, and missingness is not random across studies.

    Two files: `demographic_cohorts.csv`, one row per contrast, and
    `demographic_field_coverage.csv`, one row per band including the bands
    excluded from every contrast (Underweight, and the 18-29 reference itself).
    """
    from .health_cmh import (age_band_series, bmi_band_series, sex_series,
                             build_group_masks)

    meta = data.meta
    ind = meta["Lifestyle"] == "Industrial"
    healthy = meta["Unhealthy"] == "Healthy"

    # ── per-contrast sizes ────────────────────────────────────────────
    rows = []
    for label, ctype in config.DEMOGRAPHIC_MAP.items():
        g1, g2 = build_group_masks(meta, ctype)
        field = ("Age" if ctype.startswith("age_band") else
                 "Gender" if ctype == "sex" else "BMI")
        rows.append(dict(
            Contrast=label,
            Description=config.DEMOGRAPHIC_CONTRAST_TEXT.get(label, ""),
            Metadata_field=field,
            Restricted_to_healthy=not ctype.startswith("bmi_"),
            n_reference=int(g1.sum()),
            n_stratum=int(g2.sum()),
            n_total=int(g1.sum() + g2.sum()),
            n_studies=int(meta.loc[g1 | g2, "Study"].nunique())
            if "Study" in meta.columns else -1,
        ))
        stats.add(f"demographic_cohort_{label}", "s3", "supp",
                  f"Demographic contrast cohort size, {label}",
                  f"{int(g1.sum()):,} vs {int(g2.sum()):,}")
    cohorts = pd.DataFrame(rows)
    out = os.path.join(s3_main, "demographic_cohorts.csv")
    cohorts.to_csv(out, index=False)
    print(f"    Saved: {os.path.basename(out)} ({len(cohorts)} contrasts)")

    # ── field coverage, including the bands no contrast uses ──────────
    cov = []
    series = {"Age": age_band_series(meta), "Gender": sex_series(meta),
              "BMI": bmi_band_series(meta)}
    bands = {"Age": [b for b, _lo, _hi in config.AGE_BANDS],
             "Gender": ["Female", "Male"],
             "BMI": [b for b, _lo, _hi in config.BMI_BANDS]}
    for field, ser in series.items():
        # BMI contrasts are not restricted to Healthy, so report the pool
        # each field's own contrasts actually draw from.
        pool = ind if field == "BMI" else ind & healthy
        for band in bands[field]:
            sel = ser.eq(band)
            cov.append(dict(
                Metadata_field=field, Band=band,
                n_all_samples=int(sel.sum()),
                n_industrial=int((ind & sel).sum()),
                n_in_contrast_pool=int((pool & sel).sum()),
                Role=_band_role(field, band),
            ))
        miss_pool = int((pool & ser.isna()).sum())
        cov.append(dict(Metadata_field=field, Band="(missing)",
                        n_all_samples=int(ser.isna().sum()),
                        n_industrial=int((ind & ser.isna()).sum()),
                        n_in_contrast_pool=miss_pool,
                        Role="excluded (field absent)"))
        stats.add(f"demographic_coverage_{field}", "s3", "supp",
                  f"Industrial samples carrying {field}",
                  f"{int((ind & ser.notna()).sum()):,} of {int(ind.sum()):,}")
    covdf = pd.DataFrame(cov)
    out2 = os.path.join(s3_main, "demographic_field_coverage.csv")
    covdf.to_csv(out2, index=False)
    print(f"    Saved: {os.path.basename(out2)} ({len(covdf)} rows)")
    return cohorts, covdf


def run_demographic_associations(data, taxon_lists, specs, s3_main, stats):
    """Age / sex / BMI taxon associations, computed exactly as Figure 5D-5E.

    Reuses the three per-condition runners rather than reimplementing them, so
    the demographic panels and the disease panels cannot drift apart in their
    statistics: the same GMW-CMH weighting, the same depth-corrected richness
    bootstrap, and the same prevalence-matched background. Only the group
    masks differ.

    Writes, per detection spec:
      demographic_richness_gmw_<spec>.csv           (as Figure 5D)
      demographic_per_taxon_gmw_<spec>.csv          (as Figure 5E, taxon lists)
      demographic_per_taxon_prevmatched_<spec>.csv  (as Figure 5E, background)
    """
    print("\n  3G. Demographic associations (age / sex / BMI)")
    _write_demographic_cohorts(data, s3_main, stats)

    out = {}
    for spec in specs:
        spec_tag = spec.short_label
        print(f"\n    -- detection spec {spec_tag} --")

        per_taxon = _run_per_taxon_gmw(data, taxon_lists, spec, s3_main,
                                      demographic=True)
        richness = _run_richness_comparison(data, spec, s3_main, stats,
                                           demographic=True)
        bg = None
        if per_taxon is not None and not per_taxon.empty:
            # The demographic strata get the corrected shift too. Without
            # this the background table carried rd_mh (the matched builder
            # adds it) while this table carried only prev_diff, and every
            # comparison between them silently fell back to the pooled shift.
            # No curve is fitted here: these contrasts are not a published
            # figure, and the bootstrap over fourteen strata would cost more
            # than the result is worth.
            _dstrat = {}
            if _bg_method() != "iw":
                _dstrat = _run_stratified_shifts(data, spec, s3_main,
                                                 demographic=True)
                per_taxon = _attach_stratified(per_taxon, _dstrat, s3_main,
                                               spec_tag, demographic=True)
            bg = _run_prevalence_matched_background(
                data, taxon_lists, per_taxon, spec, s3_main,
                demographic=True, strat=_dstrat)

            # The Table S10/S11 payload: the same energy-distance-first
            # directional summary the disease contrasts get, written under
            # its own prefix so neither run overwrites the other.
            #
            # energy_stats must be computed, not left empty: energy distance
            # and its permutation p are the PRIMARY columns of this table,
            # and passing {} silently produced a table whose leading two
            # columns were NaN for every row.
            from .table_s7 import write_table_s7
            _bg = bg if bg is not None else pd.DataFrame()
            try:
                _energy = _compute_energy_stats(per_taxon, _bg)
            except Exception as _e:
                print(f"    [WARN] demographic energy stats failed: {_e}")
                _energy = {}
            write_table_s7(per_taxon, _bg, None, _energy, s3_main, spec_tag,
                           stats, table_prefix="demographic_table_s10")

        out[spec_tag] = dict(per_taxon=per_taxon, richness=richness,
                             background=bg)
    return out


def _health_comparisons(demographic=False):
    """{label: (comparison_type, disease_cols)} for the per-condition runs.

    The three per-condition runners -- richness, per-taxon prevalence, and the
    prevalence-matched background -- must iterate the identical comparison set
    or panels D and E of the same figure disagree about who was compared. It
    lives here once so that adding a stratum adds it to all three.

    With demographic=True the same runners produce the age/sex/BMI strata of
    config.DEMOGRAPHIC_MAP instead, through exactly the same statistics.
    """
    if demographic:
        return OrderedDict((label, (ctype, None))
                           for label, ctype in config.DEMOGRAPHIC_MAP.items())

    comparisons = OrderedDict([
        ("Unhealthy",            ("unhealthy",          None)),
        ("Obese",                ("obese",              None)),
    ])
    for cond_label, cond_spec in config.DISEASE_NCD_MAP.items():
        ctype = cond_spec["type"]
        dcols = cond_spec.get("cols", None)
        if ctype in ("obese", "overweight"):
            continue
        comparisons[cond_label] = (ctype, dcols)
    return comparisons


#: Output filename stem, so the demographic run never overwrites the disease
#: run's CSVs.
def _out_prefix(demographic):
    return "demographic" if demographic else "health"


def _run_per_taxon_gmw(data, taxon_lists, spec, s3_main, demographic=False):
    """
    For every condition x taxon list, compute per-taxon weighted prevalence
    differences using GMW-CMH (no significance threshold).

    Returns a DataFrame with columns:
        Condition, Taxon_list, taxon, prev_g1, prev_g2, prev_diff, log_or,
        pval, direction, n_g1, n_g2
    """
    bdf, thresh = _build_presence_matrix(data, spec)
    meta = data.meta
    spec_tag = spec.short_label

    comparisons = _health_comparisons(demographic)

    all_rows = []
    for cond_label, (comp_type, dcols) in comparisons.items():
        mask_g1, mask_g2 = build_group_masks(meta, comp_type, dcols)
        n_g1, n_g2 = int(mask_g1.sum()), int(mask_g2.sum())
        if n_g1 < 10 or n_g2 < 10:
            continue

        for list_name, mics in taxon_lists.items():
            pt = geom_mean_weighted_cmh_per_taxon(
                bdf, mics, meta, mask_g1, mask_g2, thresh)
            if pt is None or pt.empty:
                continue
            pt["Condition"]  = cond_label
            pt["Taxon_list"] = list_name
            pt["n_g1"]       = n_g1
            pt["n_g2"]       = n_g2
            all_rows.append(pt)

    if not all_rows:
        return pd.DataFrame()

    out_df = pd.concat(all_rows, ignore_index=True)
    csv_path = os.path.join(
        s3_main, f"{_out_prefix(demographic)}_per_taxon_gmw_{spec_tag}.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"    Saved per-taxon GMW data: {csv_path}")
    return out_df


# ====================================================================
#  CORRECTED BACKGROUND: STRATIFIED SHIFTS, THE CURVE, THE CLADES
# ====================================================================

def _run_stratified_shifts(data, spec, s3_main, demographic=False):
    """Study-stratified case-control shift for EVERY catalogue taxon.

    One pass per condition over all ~5,000 taxa, not per list: the shift of a
    taxon does not depend on which list it sits on, and the background curve
    needs every non-list taxon anyway. The estimator is vectorised across taxa
    (one matrix product per study stratum), so this costs a fraction of the
    per-taxon CMH it sits beside.

    Returns {condition: (taxa_df, strata_df)} and writes both as CSVs.
    """
    bdf, thresh = _build_presence_matrix(data, spec)
    meta = data.meta
    spec_tag = spec.short_label
    comparisons = _health_comparisons(demographic)
    taxa = list(bdf.columns)

    out, taxa_rows, strata_rows = {}, [], []
    for cond_label, (comp_type, dcols) in comparisons.items():
        mask_g1, mask_g2 = build_group_masks(meta, comp_type, dcols)
        n_g1, n_g2 = int(mask_g1.sum()), int(mask_g2.sum())
        if n_g1 < 10 or n_g2 < 10:
            continue
        taxa_df, strata_df = stratified_shifts_per_taxon(
            bdf, taxa, meta, mask_g1, mask_g2, thresh)
        if taxa_df is None or taxa_df.empty:
            continue
        taxa_df["Condition"] = cond_label
        taxa_df["n_g1"] = n_g1
        taxa_df["n_g2"] = n_g2
        taxa_df["n_strata"] = len(strata_df)
        strata_df = strata_df.copy()
        strata_df["Condition"] = cond_label
        out[cond_label] = (taxa_df, strata_df)
        taxa_rows.append(taxa_df)
        strata_rows.append(strata_df)
        print(f"    [STRAT] {cond_label}: {len(strata_df)} strata, "
              f"{len(taxa_df)} taxa, median |shift| "
              f"{np.nanmedian(np.abs(taxa_df.rd_mh.values)) * 100:.2f} pp")

    prefix = _out_prefix(demographic)
    if taxa_rows:
        p = os.path.join(s3_main, f"{prefix}_per_taxon_stratified_{spec_tag}.csv")
        pd.concat(taxa_rows, ignore_index=True).to_csv(p, index=False)
        print(f"    Saved stratified per-taxon shifts: {p}")
    if strata_rows:
        p = os.path.join(s3_main, f"{prefix}_stratified_strata_{spec_tag}.csv")
        pd.concat(strata_rows, ignore_index=True).to_csv(p, index=False)
    return out


def _attach_stratified(per_taxon_df, strat, s3_main=None, spec_tag=None,
                       demographic=False):
    """Merge the stratified shift onto the per-taxon list table.

    After this, `_shift_values(per_taxon_df)` returns the corrected shift
    everywhere. `prev_diff` stays in the frame: it is what the published run
    used, several robustness panels still print it, and keeping both side by
    side is how a reader checks the correction.
    """
    if per_taxon_df is None or per_taxon_df.empty or not strat:
        return per_taxon_df
    cols = ["Condition", "taxon", "rd_mh", "rd_mh_se", "t_mh", "p_mh",
            "p_ctrl_mh", "p_case_mh"]
    add = pd.concat([t for t, _ in strat.values()], ignore_index=True)
    add = add[[c for c in cols if c in add.columns]]
    drop = [c for c in add.columns if c not in ("Condition", "taxon")
            and c in per_taxon_df.columns]
    base = per_taxon_df.drop(columns=drop) if drop else per_taxon_df
    merged = base.merge(add, on=["Condition", "taxon"], how="left")
    missing = int(merged["rd_mh"].isna().sum()) if "rd_mh" in merged else len(merged)
    if missing:
        print(f"    [STRAT] {missing}/{len(merged)} list rows have no stratified "
              f"shift (condition below the stratum floor)")
    # Rewrite the CSV, so the table on disk carries the same columns the
    # figures are drawn from. Figure 5E reads this file and the background
    # file side by side; if only one of them had the stratified shift, every
    # panel E cell came out empty.
    if s3_main and spec_tag:
        path = os.path.join(
            s3_main,
            f"{_out_prefix(demographic)}_per_taxon_gmw_{spec_tag}.csv")
        merged.to_csv(path, index=False)
        print(f"    Saved per-taxon GMW data with the stratified shift: {path}")
    return merged


def _cfg_bg(name, default):
    return getattr(config, name, default)


def _run_background_curve(strat, taxon_lists, s3_main, spec_tag,
                          demographic=False, lists=None):
    """Fit the background curve per condition x list and score every list taxon.

    This is the inferential core of the corrected method: see
    `persistence/health_background.py` for what the curve is and why it
    replaced the importance-weighted background.

    Returns a dict of three tidy frames plus the raw per-condition frames the
    figures draw from:
        taxa     one row per (condition, list, taxon): z, expected, supported
        summary  one row per (condition, list): mean z, bootstrap CI, p, MDE
        grid     the fitted curve on a 400-point grid, for drawing
        frames   {condition: shift_frame} so a figure can plot the background
                 scatter without refitting
    """
    if not strat:
        return {}
    lists = lists or ["VANISH", "BloSSUM", "VANISH+", "BloSSUM+",
                      "VanishH+", "BlossumH+"]
    # Every published list is kept out of every list's background, not just
    # the one under test. See health_background.background_z's `exclude`.
    _sibling_union = set()
    if bool(_cfg_bg("HEALTH_BG_EXCLUDE_SIBLING_LISTS", True)):
        for _l in lists:
            _sibling_union |= set(taxon_lists.get(_l, ()))
    taxa_rows, summary_rows, grid_rows, frames = [], [], [], {}
    for cond_label, (taxa_df, strata_df) in strat.items():
        frame = hbg.shift_frame(taxa_df)
        frames[cond_label] = frame
        for list_name in lists:
            if list_name not in taxon_lists:
                continue
            members = [m for m in taxon_lists[list_name] if m in frame.index]
            if len(members) < 5:
                continue
            per, summ, curve = hbg.background_z(frame, members,
                                                exclude=_sibling_union)
            if per is None:
                continue
            per = per.copy()
            per["Condition"] = cond_label
            per["Taxon_list"] = list_name
            taxa_rows.append(per)
            if curve is not None:
                curve = curve.copy()
                curve["Condition"] = cond_label
                curve["Taxon_list"] = list_name
                grid_rows.append(curve)
            if summ is not None:
                summary_rows.append(dict(Condition=cond_label,
                                         Taxon_list=list_name,
                                         n_strata=int(len(strata_df)), **summ))
                print(f"    [CURVE] {cond_label} / {list_name}: mean z "
                      f"{summ['mean_z']:+.2f} [{summ['ci_lo']:+.2f}, "
                      f"{summ['ci_hi']:+.2f}] p={summ['boot_p']:.3f} "
                      f"({summ['n_supported']}/{summ['n_list']} supported)")

    out = {
        "taxa": (pd.concat(taxa_rows, ignore_index=True)
                 if taxa_rows else pd.DataFrame()),
        "summary": (pd.DataFrame(summary_rows)
                    if summary_rows else pd.DataFrame()),
        "grid": (pd.concat(grid_rows, ignore_index=True)
                 if grid_rows else pd.DataFrame()),
        "frames": frames,
    }
    prefix = _out_prefix(demographic)
    for key in ("taxa", "summary", "grid"):
        if not out[key].empty:
            p = os.path.join(
                s3_main, f"{prefix}_background_curve_{key}_{spec_tag}.csv")
            out[key].to_csv(p, index=False)
    if not out["summary"].empty:
        print(f"    Saved background-curve tables: "
              f"{prefix}_background_curve_*_{spec_tag}.csv")
    return out


def _curve_stats(curve):
    """{(condition, list): summary row} from a `_run_background_curve` result."""
    if not curve or curve.get("summary") is None or curve["summary"].empty:
        return {}
    return {(r.Condition, r.Taxon_list): r._asdict()
            for r in curve["summary"].itertuples(index=False)}


# ====================================================================
#  NEW FIGURE: DIRECTIONAL PREVALENCE SHIFT (no significance cutoff)
# ====================================================================

def _plot_directional_prevalence(per_taxon_df, cmh_df, binom_df,
                                  ref, s3_main, spec_tag, spec_desc):
    """
    Single bidirectional bar chart showing the full directional shift of
    each taxon list under each condition — including taxa trending in a
    direction but not individually significant.

    Layout mirrors health_bidirectional but bars are stacked:
        dark shade  = fraction of taxa with p < 0.05 (significant)
        light shade = fraction trending the same way (p >= 0.05)
    Depleted in disease -> bar goes DOWN (negative y)
    Enriched in disease -> bar goes UP (positive y)

    Sign convention: prev_diff = prev_g1 − prev_g2 (g1=control, g2=disease).
    A positive shift (rd_mh, or prev_diff on an older table) ⟹ more prevalent
    in control ⟹ DEPLETED in disease ⟹ DOWN.

    Consistency guarantee: n_g2 labels come from cmh_df["n_group2"]
    (strata-eligible count), the same value used in all other plots.
    All fractions use per_taxon_df as single source of truth.

    Annotations:
      - Wilcoxon * / ** / *** below/above bar tip (net directional shift)
      - n=<n_g2> at the base of each condition group
    """
    if per_taxon_df is None or per_taxon_df.empty:
        return

    cond_order = [c for c in dict.fromkeys(per_taxon_df["Condition"])
                  if not _condition_dropped(c)]
    cond_order = _display_order(cond_order)
    list_order  = [l for l in LIST_ORDER
                   if l in per_taxon_df["Taxon_list"].values]

    if not cond_order or not list_order:
        return

    n_cond  = len(cond_order)
    n_lists = len(list_order)

    # ── figure geometry ────────────────────────────────────────────────
    bw     = 0.72 / max(n_lists, 1)
    x_base = np.arange(n_cond)
    fig, ax = plt.subplots(figsize=page_figsize(height_in=4.0))

    # ── n_g2 from cmh_df (strata-eligible count, consistent with all plots)
    n_g2_by_cond = {}
    if cmh_df is not None and not cmh_df.empty:
        gmw_sub = cmh_df[cmh_df["Method"] == "Geom-mean-weighted CMH"]
        for _, row in gmw_sub.iterrows():
            cond = row["Condition"]
            if cond not in n_g2_by_cond:
                n_g2_by_cond[cond] = int(row.get("n_group2", 0))
    # Fallback to per_taxon_df if cmh_df unavailable
    if not n_g2_by_cond:
        for cond in cond_order:
            sub = per_taxon_df[per_taxon_df["Condition"] == cond]
            if not sub.empty and "n_g2" in sub.columns:
                n_g2_by_cond[cond] = int(sub["n_g2"].iloc[0])

    for li, ln in enumerate(list_order):
        color  = LIST_COLORS.get(ln, "#888")
        offset = (li - (n_lists - 1) / 2) * bw

        for ci, cond in enumerate(cond_order):
            sub = per_taxon_df[
                (per_taxon_df["Condition"]  == cond) &
                (per_taxon_df["Taxon_list"] == ln)
            ].pipe(_dropna_shift)

            if sub.empty:
                continue

            # shift > 0 -> depleted in disease (g2) -> should go DOWN
            diffs = _shift_values(sub)
            pvals = sub["pval"].values
            n_total = len(diffs)
            if n_total == 0:
                continue

            sig_dep   = np.sum((diffs > 0) & (pvals < 0.05)) / n_total
            trend_dep = np.sum((diffs > 0) & (pvals >= 0.05)) / n_total
            sig_enr   = np.sum((diffs < 0) & (pvals < 0.05)) / n_total
            trend_enr = np.sum((diffs < 0) & (pvals >= 0.05)) / n_total

            x = x_base[ci] + offset

            # Depleted -> NEGATIVE (downward)
            ax.bar(x, -(sig_dep),
                   bw * 0.88, bottom=0,
                   color=color, alpha=0.92, edgecolor='white', lw=0.4)
            ax.bar(x, -(trend_dep),
                   bw * 0.88, bottom=-(sig_dep),
                   color=color, alpha=0.30, edgecolor='white', lw=0.4)

            # Enriched -> POSITIVE (upward)
            ax.bar(x, sig_enr,
                   bw * 0.88, bottom=0,
                   color=color, alpha=0.92, edgecolor='white', lw=0.4)
            ax.bar(x, trend_enr,
                   bw * 0.88, bottom=sig_enr,
                   color=color, alpha=0.30, edgecolor='white', lw=0.4)

            # Wilcoxon test on the full distribution of diffs
            wilcox_star = ""
            if len(diffs) >= 10 and not np.all(diffs == 0):
                try:
                    _, wp = wilcoxon(diffs, alternative="two-sided",
                                     zero_method="wilcox")
                    if   wp < 0.001: wilcox_star = "***"
                    elif wp < 0.01:  wilcox_star = "**"
                    elif wp < 0.05:  wilcox_star = "*"
                except Exception:
                    pass

            # Annotate Wilcoxon star beyond the longest bar
            if wilcox_star:
                net_dep = sig_dep + trend_dep
                net_enr = sig_enr + trend_enr
                if net_dep >= net_enr:
                    # net depleted — annotate below
                    tip = -(sig_dep + trend_dep)
                    ax.text(x, tip - 0.012, wilcox_star,
                            ha='center', va='top',
                            fontsize=FS-2, fontweight='bold', color=color)
                else:
                    tip = sig_enr + trend_enr
                    ax.text(x, tip + 0.012, wilcox_star,
                            ha='center', va='bottom',
                            fontsize=FS-2, fontweight='bold', color=color)

    # ── axes decoration ────────────────────────────────────────────────
    ax.axhline(0, color='black', lw=0.9)
    ax.axhline( 0.05, color='#999', lw=0.5, ls=':', alpha=0.6)
    ax.axhline(-0.05, color='#999', lw=0.5, ls=':', alpha=0.6)

    # Condition separator between general health and diseases
    for _b in _group_dividers(cond_order):
        ax.axvline(_b, color='lightgray', ls='--', lw=1, zorder=0)

    # x-axis: condition name + n_g2
    xtick_labels = [
        f"{c}\n(n={n_g2_by_cond[c]})" if c in n_g2_by_cond else c
        for c in cond_order
    ]
    ax.set_xticks(x_base)
    ax.set_xticklabels(xtick_labels, rotation=30, ha='right', fontsize=FS-1)

    # y-axis symmetric
    ymax = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.25
    ax.set_ylim(-ymax, ymax)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{abs(v):.0%}"))
    ax.set_ylabel(
        "Depleted in disease  (down)          (up)  Enriched in disease\n"
        "(fraction of taxa; dark=p<0.05, light=trending; "
        "annotation=Wilcoxon on all diffs)",
        fontsize=FS-2)

    ax.grid(axis='y', alpha=0.15)

    # ── legend ─────────────────────────────────────────────────────────
    legend_handles = []
    for ln in list_order:
        c = LIST_COLORS.get(ln, "#888")
        legend_handles.append(
            Patch(facecolor=c, alpha=0.92, edgecolor='none', label=ln))
    legend_handles += [
        Patch(facecolor='grey', alpha=0.92, edgecolor='none',
              label='Significant (p<0.05)'),
        Patch(facecolor='grey', alpha=0.30, edgecolor='none',
              label='Trending (not sig.)'),
    ]
    ax.legend(handles=legend_handles, loc='upper left',
              fontsize=FS-2, ncol=4, framealpha=0.95)

    ax.set_title(
        f"Directional Prevalence Shift — GMW-CMH [{ref}]\n"
        f"Detection: {spec_desc}  |  "
        f"dark=p<0.05, light=trending, annotation=Wilcoxon */**/***, "
        f"dashed=5% chance level",
        fontsize=FS, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(s3_main,
                       f"health_directional_{ref}_{spec_tag}.png")
    savefig_dual(fig, out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out}")


# ====================================================================
#  NEW FIGURE: CONDITION x LIST HEATMAP
# ====================================================================

def _plot_condition_heatmap(cmh_df, binom_df, ref, s3_main,
                             spec_tag, spec_desc, per_taxon_df=None):
    """
    Heatmap: rows = conditions, columns = taxon lists.

    Cell COLOUR = net directional fraction from per-taxon GMW data
    (including both significant AND trending taxa):
        (frac_trending_depleted - frac_trending_enriched)
        positive (red)  = more taxa trending depleted in disease
        negative (blue) = more taxa trending enriched in disease

    Using the full directional signal (not just p<0.05) makes the heatmap
    consistent with the directional prevalence plot and avoids VANISH
    appearing flat simply because it has few individually significant taxa.

    Cell TEXT = binomial significance asterisks (dominant direction),
    reflecting whether the *significant* fraction exceeds 5% chance.

    Falls back to (dep_sig - enr_sig)/total if per_taxon_df not available.
    """
    if cmh_df is None or cmh_df.empty:
        return

    gmw = cmh_df[cmh_df["Method"] == "Geom-mean-weighted CMH"]
    if gmw.empty:
        return

    cond_order = [c for c in dict.fromkeys(gmw["Condition"])
                  if not _condition_dropped(c)]
    cond_order = _display_order(cond_order)
    list_order = [l for l in LIST_ORDER if l in gmw["Taxon_list"].values]

    if not cond_order or not list_order:
        return

    mat   = np.full((len(cond_order), len(list_order)), np.nan)
    annot = np.full((len(cond_order), len(list_order)), "", dtype=object)
    n_g2_by_cond = {}

    # Binomial asterisk lookup (significant fraction > 5% chance level)
    sig_lookup = {}
    if binom_df is not None and not binom_df.empty:
        gmw_b = binom_df[binom_df["Method"] == "Geom-mean-weighted CMH"]
        for _, row in gmw_b.iterrows():
            sig_lookup[(row["Condition"], row["Taxon_list"],
                        row["Direction"])] = row.get("Sig", "")

    # Per-taxon directional lookup for heatmap colour
    pt_lookup = {}   # (cond, list) -> signed directional fraction
    if per_taxon_df is not None and not per_taxon_df.empty:
        for (cond, ln), grp in per_taxon_df.groupby(
                ["Condition", "Taxon_list"]):
            diffs = _shift_values(_dropna_shift(grp))
            n = len(diffs)
            if n == 0:
                continue
            # shift > 0 = depleted in disease, < 0 = enriched in disease
            n_dep = np.sum(diffs > 0)
            n_enr = np.sum(diffs < 0)
            pt_lookup[(cond, ln)] = (n_dep - n_enr) / n

    for ci, cond in enumerate(cond_order):
        for li, ln in enumerate(list_order):
            sub = gmw[(gmw["Condition"] == cond) & (gmw["Taxon_list"] == ln)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            n_g2_by_cond[cond] = int(r.get("n_group2", 0))

            # Colour: prefer full directional signal from per-taxon data
            if (cond, ln) in pt_lookup:
                mat[ci, li] = pt_lookup[(cond, ln)]
            else:
                # fallback: significant fraction only
                total = int(r["total_taxa"])
                if total > 0:
                    dep = int(r["depleted_count"])
                    enr = int(r["enriched_count"])
                    mat[ci, li] = (dep - enr) / total

            # Annotation: binomial sig on the dominant direction
            dep = int(r["depleted_count"])
            enr = int(r["enriched_count"])
            dom_dir = "Depleted" if dep >= enr else "Enriched"
            annot[ci, li] = sig_lookup.get((cond, ln, dom_dir), "")

    row_labels = [
        f"{c}  (n={n_g2_by_cond.get(c, '?')})" for c in cond_order
    ]

    fig, ax = plt.subplots(
        figsize=page_figsize(height_in=max(2.4, len(cond_order) * 0.38 + 1.0)))

    vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1.0
    vmax = max(vmax, 0.1)
    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.7,
                 label="Net directional fraction (trending dep - enr) / total")

    ax.set_xticks(range(len(list_order)))
    ax.set_xticklabels(list_order, rotation=35, ha='right', fontsize=FS-1)
    ax.set_yticks(range(len(cond_order)))
    ax.set_yticklabels(row_labels, fontsize=FS-2)

    for ci in range(len(cond_order)):
        for li in range(len(list_order)):
            txt = annot[ci, li]
            if txt:
                cell_val = mat[ci, li]
                txt_color = ('black' if np.isnan(cell_val) or abs(cell_val) < 0.35
                             else 'white')
                ax.text(li, ci, txt, ha='center', va='center',
                        fontsize=FS-1, fontweight='bold', color=txt_color)

    colour_note = ("colour=full directional trend (sig+trending), "
                   "text=binomial sig. of significant fraction")
    ax.set_title(
        f"Health Association Heatmap — GMW-CMH [{ref}]\n"
        f"{colour_note}\n"
        f"Detection: {spec_desc}",
        fontsize=FS, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(s3_main,
                       f"health_heatmap_{ref}_{spec_tag}.png")
    savefig_dual(fig, out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out}")


# ====================================================================
#  NEW FIGURE: H+ vs + COMPARISON
# ====================================================================

def _plot_hplus_comparison(cmh_df, binom_df, per_taxon_df,
                            ref, s3_main, spec_tag, spec_desc):
    """
    Single unified bidirectional bar chart comparing all 6 primary lists
    (VANISH, BloSSUM, VANISH+, BloSSUM+, VanishH+, BlossumH+) side by
    side for each condition, using GMW-CMH.

    Each bar is stacked: dark = significant (p<0.05), light = trending.
    Depleted in disease -> DOWN, Enriched -> UP.

    Shows simultaneously:
      1. The VANISH/BloSSUM anchor signal
      2. What the full differential (VANISH+/BloSSUM+) adds
      3. What restricting to healthy-industrial reference (H+ lists) changes

    Significance annotations:
      - Binomial * / ** / *** above/below bar tip (sig. fraction > 5% chance)
      - Wilcoxon annotation from per_taxon_df if available
    """
    if cmh_df is None or cmh_df.empty:
        return

    PLOT_LISTS = ["VANISH", "BloSSUM", "VANISH+", "BloSSUM+",
                  "VanishH+", "BlossumH+"]

    gmw = cmh_df[cmh_df["Method"] == "Geom-mean-weighted CMH"]
    available = set(gmw["Taxon_list"].unique())
    plot_lists = [l for l in PLOT_LISTS if l in available]
    if not plot_lists:
        return

    cond_order = [c for c in dict.fromkeys(gmw["Condition"])
                  if not _condition_dropped(c)]
    cond_order = _display_order(cond_order)
    if not cond_order:
        return

    n_lists = len(plot_lists)
    n_cond  = len(cond_order)
    bw      = 0.78 / max(n_lists, 1)
    x_base  = np.arange(n_cond)

    # ── binomial significance lookup ───────────────────────────────────
    binom_sig = {}
    if binom_df is not None and not binom_df.empty:
        gmw_b = binom_df[binom_df["Method"] == "Geom-mean-weighted CMH"]
        for _, row in gmw_b.iterrows():
            binom_sig[(row["Condition"], row["Taxon_list"],
                       row["Direction"])] = row.get("Sig", "")

    # ── Wilcoxon lookup from per-taxon data ───────────────────────────
    wilcox_lookup = {}
    if per_taxon_df is not None and not per_taxon_df.empty:
        for (cond, ln), grp in per_taxon_df.groupby(
                ["Condition", "Taxon_list"]):
            diffs = _shift_values(_dropna_shift(grp))
            if len(diffs) < 10 or np.all(diffs == 0):
                continue
            try:
                _, wp = wilcoxon(diffs, alternative="two-sided",
                                 zero_method="wilcox")
                star = ("***" if wp < 0.001 else
                        "**"  if wp < 0.01  else
                        "*"   if wp < 0.05  else "")
                if star:
                    wilcox_lookup[(cond, ln)] = star
            except Exception:
                pass

    # ── n_g2 per condition ─────────────────────────────────────────────
    n_g2_by_cond = {}
    for _, row in gmw.iterrows():
        cond = row["Condition"]
        if cond not in n_g2_by_cond:
            n_g2_by_cond[cond] = int(row.get("n_group2", 0))

    fig, ax = plt.subplots(figsize=page_figsize(height_in=4.2))

    for li, ln in enumerate(plot_lists):
        color  = LIST_COLORS.get(ln, "#888")
        offset = (li - (n_lists - 1) / 2) * bw

        for ci, cond in enumerate(cond_order):
            # ── All fractions from per_taxon_df (consistent denominator) ─
            dep_sig = enr_sig = dep_trend = enr_trend = 0.0
            dep_ci_lo = dep_ci_hi = enr_ci_lo = enr_ci_hi = 0.0

            if per_taxon_df is not None and not per_taxon_df.empty:
                pt = per_taxon_df[
                    (per_taxon_df["Condition"]  == cond) &
                    (per_taxon_df["Taxon_list"] == ln)
                ].pipe(_dropna_shift)
                if not pt.empty:
                    diffs  = _shift_values(pt)
                    pvals  = pt["pval"].values
                    n_total = len(diffs)
                    if n_total > 0:
                        n_dep_sig = int(np.sum((diffs > 0) & (pvals < 0.05)))
                        n_enr_sig = int(np.sum((diffs < 0) & (pvals < 0.05)))
                        dep_sig   = n_dep_sig / n_total
                        enr_sig   = n_enr_sig / n_total
                        dep_trend = np.sum(
                            (diffs > 0) & (pvals >= 0.05)) / n_total
                        enr_trend = np.sum(
                            (diffs < 0) & (pvals >= 0.05)) / n_total
                        # Wilson 95% CI on significant fractions
                        if n_dep_sig > 0:
                            lo, hi = wilson_ci(n_dep_sig, n_total)
                            dep_ci_lo = max(0.0, dep_sig - lo)
                            dep_ci_hi = max(0.0, hi - dep_sig)
                        if n_enr_sig > 0:
                            lo, hi = wilson_ci(n_enr_sig, n_total)
                            enr_ci_lo = max(0.0, enr_sig - lo)
                            enr_ci_hi = max(0.0, hi - enr_sig)
            else:
                # Fallback to cmh_df summary if no per_taxon data
                sub = gmw[(gmw["Condition"] == cond) &
                          (gmw["Taxon_list"] == ln)]
                if not sub.empty:
                    r = sub.iloc[0]
                    dep_sig   = float(r["depleted_frac"])
                    enr_sig   = float(r["enriched_frac"])
                    dep_ci_lo = float(r["depleted_ci_lo"])
                    dep_ci_hi = float(r["depleted_ci_hi"])
                    enr_ci_lo = float(r["enriched_ci_lo"])
                    enr_ci_hi = float(r["enriched_ci_hi"])

            x = x_base[ci] + offset

            # ── Depleted bars (downward / negative) ──────────────────
            ax.bar(x, -dep_sig,   bw * 0.88, bottom=0,
                   color=color, alpha=0.92, edgecolor='white', lw=0.4)
            ax.bar(x, -dep_trend, bw * 0.88, bottom=-dep_sig,
                   color=color, alpha=0.28, edgecolor='white', lw=0.4)
            if dep_sig > 0:
                ax.errorbar(x, -dep_sig,
                            yerr=[[dep_ci_hi], [dep_ci_lo]],
                            fmt='none', color='black', capsize=1.5, lw=0.7)

            # ── Enriched bars (upward / positive) ────────────────────
            ax.bar(x, enr_sig,   bw * 0.88, bottom=0,
                   color=color, alpha=0.92, edgecolor='white', lw=0.4)
            ax.bar(x, enr_trend, bw * 0.88, bottom=enr_sig,
                   color=color, alpha=0.28, edgecolor='white', lw=0.4)
            if enr_sig > 0:
                ax.errorbar(x, enr_sig,
                            yerr=[[enr_ci_lo], [enr_ci_hi]],
                            fmt='none', color='black', capsize=1.5, lw=0.7)

            # ── Binomial significance annotation (sig. fraction > 5%) ─
            dep_bstar = binom_sig.get((cond, ln, "Depleted"), "")
            enr_bstar = binom_sig.get((cond, ln, "Enriched"), "")
            tip_dep = -(dep_sig + dep_trend)
            tip_enr = enr_sig + enr_trend

            if dep_bstar:
                ax.text(x, tip_dep - 0.008, dep_bstar,
                        ha='center', va='top', fontsize=FS-2,
                        fontweight='bold', color=color)
            if enr_bstar:
                ax.text(x, tip_enr + 0.008, enr_bstar,
                        ha='center', va='bottom', fontsize=FS-2,
                        fontweight='bold', color=color)

            # ── Wilcoxon annotation (net directional shift) ───────────
            wstar = wilcox_lookup.get((cond, ln), "")
            if wstar and not dep_bstar and not enr_bstar:
                # Only show Wilcoxon if no binomial star already
                dom_tip = tip_dep if (dep_sig + dep_trend) >= \
                                     (enr_sig + enr_trend) else tip_enr
                va = 'top' if dom_tip < 0 else 'bottom'
                pad = -0.016 if dom_tip < 0 else 0.016
                ax.text(x, dom_tip + pad, f"({wstar})",
                        ha='center', va=va, fontsize=FS-2,
                        color=color, style='italic')

    # ── axes and labels ────────────────────────────────────────────────
    ax.axhline(0, color='black', lw=0.9)
    ax.axhline( 0.05, color='#999', lw=0.5, ls=':', alpha=0.6)
    ax.axhline(-0.05, color='#999', lw=0.5, ls=':', alpha=0.6)

    for _b in _group_dividers(cond_order):
        ax.axvline(_b, color='lightgray', ls='--', lw=1, zorder=0)

    xtick_labels = [
        f"{c}\n(n={n_g2_by_cond.get(c, '?')})" for c in cond_order
    ]
    ax.set_xticks(x_base)
    ax.set_xticklabels(xtick_labels, rotation=30, ha='right', fontsize=FS-1)

    ymax = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.30
    ax.set_ylim(-ymax, ymax)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{abs(v):.0%}"))
    ax.set_ylabel(
        "Depleted in disease  (down)  /  Enriched in disease  (up)\n"
        "GMW-CMH fraction  |  dark=sig. (p<0.05), light=trending  |  "
        "annotation: binomial binom.* ; Wilcoxon (*)  |  error bars: Wilson 95% CI",
        fontsize=FS-2)
    ax.grid(axis='y', alpha=0.15)

    # ── legend ─────────────────────────────────────────────────────────
    leg = []
    for ln in plot_lists:
        leg.append(Patch(facecolor=LIST_COLORS.get(ln, "#888"),
                         alpha=0.92, edgecolor='none', label=ln))
    leg += [
        Patch(facecolor='grey', alpha=0.92, edgecolor='none',
              label='Significant (p<0.05)'),
        Patch(facecolor='grey', alpha=0.28, edgecolor='none',
              label='Trending (not sig.)'),
    ]
    ax.legend(handles=leg, loc='upper left', fontsize=FS-2,
              ncol=4, framealpha=0.95)

    ax.set_title(
        f"Health Association Summary — GMW-CMH [{ref}]\n"
        f"Detection: {spec_desc}  |  "
        f"VANISH / BloSSUM / VANISH+ / BloSSUM+ / VanishH+ / BlossumH+",
        fontsize=FS, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(s3_main,
                       f"health_hplus_comparison_{ref}_{spec_tag}.png")
    savefig_dual(fig, out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out}")


# ====================================================================
#  NEW FIGURE: METHOD ROBUSTNESS FOREST PLOT
# ====================================================================

def _plot_method_forest(cmh_df, binom_df, ref, s3_main,
                         spec_tag, spec_desc):
    """
    Per-condition forest plot showing dep/enr fractions from all four
    methods (Within-study CMH, GMW-CMH, Depth-matched CMH, Mundlak GLMM)
    for the two primary taxon lists (VANISH and BloSSUM).

    Communicates that GMW-CMH results are supported by independent
    robustness checks.
    """
    if cmh_df is None or cmh_df.empty:
        return

    method_order = [
        "Within-study CMH",
        "Geom-mean-weighted CMH",
        "Depth-matched CMH",
        "Mundlak GLMM",
    ]
    method_colors = {
        "Within-study CMH":       "#999999",
        "Geom-mean-weighted CMH": "#1a6faf",
        "Depth-matched CMH":      config.OI["vermillion"],
        "Mundlak GLMM":           config.OI["orange"],
    }
    primary_lists = [l for l in ["VANISH", "BloSSUM"]
                     if l in cmh_df["Taxon_list"].values]
    if not primary_lists:
        primary_lists = [l for l in LIST_ORDER
                         if l in cmh_df["Taxon_list"].values][:2]

    cond_order = [c for c in dict.fromkeys(cmh_df["Condition"])
                  if not _condition_dropped(c)]
    cond_order = _display_order(cond_order)
    if not cond_order:
        return

    methods_present = [m for m in method_order
                       if m in cmh_df["Method"].values]
    n_methods = len(methods_present)
    n_lists   = len(primary_lists)
    n_cond    = len(cond_order)

    fig, axes = plt.subplots(
        n_lists, n_cond,
        figsize=page_figsize(n_cols=n_cond,
                             height_in=n_lists * n_methods * 0.42 + 0.9),
        squeeze=False)

    for li, ln in enumerate(primary_lists):
        for ci, cond in enumerate(cond_order):
            ax = axes[li][ci]
            y_pos = []
            for mi, method in enumerate(methods_present):
                sub = cmh_df[
                    (cmh_df["Condition"]  == cond) &
                    (cmh_df["Taxon_list"] == ln) &
                    (cmh_df["Method"]     == method)]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                dep  = float(r["depleted_frac"])
                enr  = float(r["enriched_frac"])
                dep_lo = float(r["depleted_ci_lo"])
                dep_hi = float(r["depleted_ci_hi"])
                enr_lo = float(r["enriched_ci_lo"])
                enr_hi = float(r["enriched_ci_hi"])
                col = method_colors.get(method, "#333")
                y   = -mi  # methods stacked vertically

                # Depleted (negative x)
                ax.errorbar(-dep, y, xerr=[[dep_lo], [dep_hi]],
                            fmt='o', color=col, markersize=4,
                            capsize=2, lw=1, elinewidth=0.8)
                # Enriched (positive x)
                ax.errorbar(enr, y, xerr=[[enr_lo], [enr_hi]],
                            fmt='s', color=col, markersize=4,
                            capsize=2, lw=1, elinewidth=0.8)
                y_pos.append(y)

            ax.axvline(0, color='black', lw=0.6)
            ax.axvline( 0.05, color='grey', lw=0.4, ls=':', alpha=0.5)
            ax.axvline(-0.05, color='grey', lw=0.4, ls=':', alpha=0.5)
            ax.set_yticks(list(range(0, -n_methods, -1)))
            ax.set_yticklabels(
                [m.replace(" CMH", "").replace("-", "\n")
                 for m in methods_present],
                fontsize=FS-2)
            ax.tick_params(axis='x', labelsize=6)

            if li == 0:
                ax.set_title(cond, fontsize=FS-2, fontweight='bold',
                             rotation=30, ha='left')
            if ci == 0:
                ax.set_ylabel(ln, fontsize=FS-1, fontweight='bold',
                              rotation=0, ha='right', labelpad=50)

    # Shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
               markersize=7, label=m)
        for m, c in method_colors.items() if m in methods_present
    ]
    handles.append(
        Line2D([0], [0], marker='o', color='grey', markersize=6,
               label='circle=depleted  square=enriched', linestyle='none'))
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=FS-2, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        f"Method Robustness — All Four Tiers [{ref}]\n"
        f"Detection: {spec_desc} | ○ depleted  □ enriched in disease | "
        f"error bars: Wilson 95% CI",
        fontsize=FS, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(s3_main,
                       f"health_method_forest_{ref}_{spec_tag}.png")
    savefig_dual(fig, out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out}")


# ====================================================================
#  NEW FIGURE: ESS DIAGNOSTIC
# ====================================================================

def _plot_ess_diagnostic(cmh_df, ref, s3_main, spec_tag, spec_desc):
    """
    Bar chart of GMW-CMH ESS fraction (mean_ess_frac) per condition x list.
    Shows what fraction of samples are effectively used after reweighting.
    ESS/n close to 1.0 means depth distributions are similar; lower values
    indicate more aggressive reweighting.
    """
    if cmh_df is None or cmh_df.empty:
        return

    gmw = cmh_df[
        (cmh_df["Method"] == "Geom-mean-weighted CMH") &
        cmh_df["mean_ess_frac"].notna()
    ] if "mean_ess_frac" in cmh_df.columns else pd.DataFrame()

    if gmw.empty:
        return

    cond_order = [c for c in dict.fromkeys(gmw["Condition"])
                  if not _condition_dropped(c)]
    cond_order = _display_order(cond_order)
    list_order = [l for l in LIST_ORDER if l in gmw["Taxon_list"].values]
    if not cond_order or not list_order:
        return

    x = np.arange(len(cond_order))
    bw = 0.8 / max(len(list_order), 1)
    fig, ax = plt.subplots(figsize=page_figsize(height_in=2.8))

    for li, ln in enumerate(list_order):
        color = LIST_COLORS.get(ln, "#888")
        ess_vals = []
        for cond in cond_order:
            sub = gmw[(gmw["Condition"] == cond) & (gmw["Taxon_list"] == ln)]
            ess_vals.append(float(sub["mean_ess_frac"].iloc[0])
                            if not sub.empty else np.nan)

        offsets = x + (li - len(list_order) / 2 + 0.5) * bw
        bars = ax.bar(offsets, ess_vals, bw * 0.9,
                      color=color, alpha=0.85, edgecolor='white', lw=0.5,
                      label=ln)

    ax.axhline(1.0, color='black', lw=0.8, ls='--', alpha=0.5,
               label='ESS/n = 1 (no reweighting needed)')
    ax.axhline(0.7, color='orange', lw=0.6, ls=':', alpha=0.6,
               label='ESS/n = 0.7 (moderate reweighting)')
    ax.set_xticks(x)
    ax.set_xticklabels(cond_order, rotation=35, ha='right', fontsize=FS-1)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean ESS / n", fontsize=FS-1)
    ax.set_title(
        f"GMW-CMH Effective Sample Size [{ref}]\n"
        f"Detection: {spec_desc}",
        fontsize=FS, fontweight='bold')
    ax.legend(fontsize=FS-2, ncol=4, loc='lower right')
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    out = os.path.join(s3_main,
                       f"health_ess_{ref}_{spec_tag}.png")
    savefig_dual(fig, out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out}")


# ====================================================================
#  IW-WEIGHTING HELPERS  (used by all prevalence-matched plots)
# ====================================================================

def _iw_weighted_kde(vals, weights, x_eval, bw_method='scott'):
    """
    Weighted KDE via scipy gaussian_kde(weights=...) so the background
    density is IW-reweighted rather than equal-treating replicated rows.
    """
    from scipy.stats import gaussian_kde as _gkde_iw
    vals    = np.asarray(vals,    dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0.0)
    mask    = np.isfinite(vals) & (weights > 0)
    vals, weights = vals[mask], weights[mask]
    if len(vals) < 2:
        return np.zeros_like(x_eval, dtype=float)
    weights = weights / weights.sum()
    try:
        return np.maximum(
            _gkde_iw(vals, weights=weights, bw_method=bw_method)(x_eval), 0.0)
    except Exception:
        try:
            return np.maximum(_gkde_iw(vals, bw_method=bw_method)(x_eval), 0.0)
        except Exception:
            return np.zeros_like(x_eval, dtype=float)


def _iw_fracs(sub_df):
    """
    Weighted depleted/enriched fractions for a background sub-DataFrame.
    Operates on unique taxa (grouped by taxon col if present), weighted by the
    prevalence-matched weight (`bg_weight`), or by the old `importance_weight`
    when reading a published data object. Returns the same dict as the original
    _fracs helper plus w (raw weights) and ess ((sum_w)^2 / sum_w^2).
    """
    shift_col = _shift_column(sub_df)
    if shift_col is None:
        sub = sub_df.iloc[0:0]
    else:
        sub = sub_df.dropna(subset=[shift_col])
    if sub.empty:
        return dict(dep_s=0., dep_t=0., enr_s=0., enr_t=0.,
                    d_lo=0., d_hi=0., e_lo=0., e_hi=0.,
                    diffs=np.array([]), w=np.array([]),
                    n=0, n_dep_sig=0, n_enr_sig=0, ess=1.)
    if "taxon" in sub.columns:
        uniq = sub.groupby("taxon", sort=False).first().reset_index()
    else:
        uniq = sub.copy()
    diffs = _shift_values(uniq)
    pvals = uniq["pval"].values
    w = _bg_weight_values(uniq)
    w_sum = max(w.sum(), 1e-12)
    w_n   = w / w_sum
    ess   = (w.sum() ** 2) / max((w ** 2).sum(), 1e-12)
    dep_s = float(w_n[(diffs > 0) & (pvals < 0.05)].sum())
    dep_t = float(w_n[(diffs > 0) & (pvals >= 0.05)].sum())
    enr_s = float(w_n[(diffs < 0) & (pvals < 0.05)].sum())
    enr_t = float(w_n[(diffs < 0) & (pvals >= 0.05)].sum())
    def _wci(frac, ess_n):
        if frac == 0: return 0., 0.
        k = max(int(round(frac * ess_n)), 1)
        n = max(int(round(ess_n)), 1)
        lo, hi = wilson_ci(k, n)
        return max(0., frac - lo), max(0., hi - frac)
    d_lo, d_hi = _wci(dep_s, ess)
    e_lo, e_hi = _wci(enr_s, ess)
    return dict(
        dep_s=dep_s, dep_t=dep_t, enr_s=enr_s, enr_t=enr_t,
        d_lo=d_lo,   d_hi=d_hi,   e_lo=e_lo,   e_hi=e_hi,
        diffs=diffs, w=w, n=len(diffs),
        n_dep_sig=int(((diffs > 0) & (pvals < 0.05)).sum()),
        n_enr_sig=int(((diffs < 0) & (pvals < 0.05)).sum()),
        ess=ess)


def _iw_energy_test(sl, sub_b_df, col=None, n_perm=5000, rng_seed=42):
    """
    One-sample weighted energy distance test.
    Observed: energy_distance(list_vals, large_weighted_resample_of_bg)
    Null:     energy_distance(n_list draws from the weighted bg, same reference)
    `col` defaults to whichever shift column the frame carries; pass prev_g1 or
    prev_g2 to test the prevalence balance instead.

    NOTE: this test treats the weighted background as known, so it ignores the
    background's own sampling error. Under the corrected method it is kept as a
    descriptive distributional statistic only -- the inference the manuscript
    quotes is the mean z and its curve-refitting bootstrap CI
    (`health_background.background_z`).
    Returns (E_obs, p_value, star_str, med_shift, mad_ratio).
    """
    from scipy.stats import energy_distance
    sl = np.asarray(sl, dtype=float); sl = sl[np.isfinite(sl)]
    if len(sl) < 3: return np.nan, np.nan, '', 0.0, 1.0
    col = col or _shift_column(sub_b_df)
    if col is None: return np.nan, np.nan, '', 0.0, 1.0
    sub_b = sub_b_df.dropna(subset=[col]).copy()
    bg_uniq = sub_b.groupby("taxon", sort=False).first().reset_index() if "taxon" in sub_b.columns else sub_b
    sb_vals = bg_uniq[col].values.astype(float)
    fin = np.isfinite(sb_vals); sb_vals = sb_vals[fin]
    if len(sb_vals) < 3: return np.nan, np.nan, '', 0.0, 1.0
    sb_w = _bg_weight_values(bg_uniq)[fin]
    w_sum = sb_w.sum()
    if w_sum < 1e-12: return np.nan, np.nan, '', 0.0, 1.0
    w_prob = sb_w / w_sum
    n_list = len(sl)
    rng    = np.random.default_rng(rng_seed)
    n_ref  = max(n_list * 20, 2000)
    sb_ref = sb_vals[rng.choice(len(sb_vals), size=n_ref, replace=True, p=w_prob)]
    obs_e  = energy_distance(sl, sb_ref)
    null   = np.array([energy_distance(
                 sb_vals[rng.choice(len(sb_vals), size=n_list, replace=True, p=w_prob)],
                 sb_ref) for _ in range(n_perm)])
    p = max(float(np.mean(null >= obs_e)), 1.0 / (n_perm + 1))
    sort_idx = np.argsort(sb_vals)
    cum_w    = np.cumsum(sb_w[sort_idx]) / w_sum
    sb_med   = float(sb_vals[sort_idx[np.searchsorted(cum_w, 0.5)]])
    mad_b    = float(np.abs(sb_vals[sort_idx] - sb_med)[np.searchsorted(cum_w, 0.5)])
    med_shift = float(np.median(sl)) - sb_med
    mad_l     = float(np.median(np.abs(sl - np.median(sl))))
    mad_ratio = mad_l / max(mad_b, 1e-12)
    def _s(pv):
        if not np.isfinite(pv): return ''
        return '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else 'ns'
    return obs_e, p, _s(p), med_shift, mad_ratio


def _iw_directional_test(fl, fb, dom, n_perm=2000, rng_seed=42):
    """
    IW-weighted directional permutation test replacing Mann-Whitney U.
    Returns (p_value, star_str, flag).  flag=+1 = list more extreme.
    """
    dl = fl['diffs']; db = fb['diffs']; db_w = fb.get('w', np.ones(len(db)))
    if len(dl) < 3 or len(db) < 3 or db_w.sum() < 1e-12: return np.nan, '', 0
    w_prob   = db_w / db_w.sum()
    rng      = np.random.default_rng(rng_seed)
    bg_wmean = float(np.average(db, weights=db_w))
    sign     = +1.0 if dom == 'dep' else -1.0
    obs_stat = sign * (float(np.mean(dl)) - bg_wmean)
    null = sign * np.array([
        np.mean(db[rng.choice(len(db), size=len(dl), replace=True, p=w_prob)]) - bg_wmean
        for _ in range(n_perm)])
    p = max(float(np.mean(null >= obs_stat)), 1.0 / (n_perm + 1))
    def _s(pv):
        if not np.isfinite(pv): return ''
        return '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else 'ns'
    return p, _s(p), 1 if obs_stat > 0 else -1


# ====================================================================
#  PREVALENCE-MATCHED BACKGROUND COMPARISON
# ====================================================================

def _stratum_prevalence_matrix(bdf, meta, mask_g1, mask_g2, thresh,
                                min_stratum=None):
    """
    Compute per-taxon control-group prevalence within qualifying Study strata.

    A stratum qualifies only when BOTH the control group (g1) AND the disease
    group (g2) have >= min_stratum samples within that study, mirroring
    _get_study_strata.  Only qualifying strata contribute to the distance
    matrix, so the matching only considers population contexts that actually
    appear in both arms of the comparison.

    Returns
    -------
    prev_mat : DataFrame, shape (n_taxa, n_strata)
        Each column is a study name; values are fraction of control
        samples in that study where the taxon is detected (>= thresh).
    weights  : Series indexed by study
        Proportional to stratum g1 size within qualifying strata; sums to 1.
    """
    if min_stratum is None:
        min_stratum = getattr(config, "CMH_MIN_STRATUM", 5)

    md = meta.loc[meta.index.isin(bdf.index)].copy()

    if "_study" not in md.columns or md["_study"].isna().all():
        return pd.DataFrame(), pd.Series(dtype=float)

    # Stratify by Study only — consistent with CMH stratification in the rest
    # of the pipeline.  StudyxCountry gave more strata in theory but reduced
    # power by splitting studies into small cells that failed the min_stratum
    # filter, particularly for rare-disease conditions.
    md["_stratum_key"] = md["_study"].astype(str).str.strip()

    rows  = {}
    sizes = {}
    for stratum_key, grp in md.groupby("_stratum_key", sort=False):
        if pd.isna(stratum_key) or stratum_key in ("nan", "None"):
            continue
        # Qualify: both g1 and g2 must meet min_stratum in this cell
        g1_idx = grp.index[mask_g1.reindex(grp.index, fill_value=False)]
        g2_idx = grp.index[mask_g2.reindex(grp.index, fill_value=False)]
        if len(g1_idx) < min_stratum or len(g2_idx) < min_stratum:
            continue
        prev                = (bdf.loc[g1_idx] >= thresh).mean(axis=0)
        rows[stratum_key]   = prev
        sizes[stratum_key]  = len(g1_idx)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float)

    prev_mat = pd.DataFrame(rows)    # taxa x strata
    total    = sum(sizes.values())
    weights  = pd.Series({s: n / total for s, n in sizes.items()})
    return prev_mat, weights


def _scalar_prevalence_distance(list_prev_mat, bg_prev_mat, weights):
    """
    1D distance on scalar weighted-mean control prevalence.

    For each taxon, collapse the stratum matrix to a single number:
        p = Σ_s  w_s * prev_s   (weighted-mean control prevalence)

    Then distance d(A, B) = |p_A - p_B|.

    This is the correct quantity to match on because:
      1. It is exactly what the QC plot shows on its x-axis, so matching
         on it guarantees the QC distributions will coincide.
      2. The downstream energy-distance test compares prev_diff
         distributions; the confound we are controlling is prevalence
         level, which is a scalar concept.
      3. Multi-dimensional stratum matching was over-constraining:
         finding a background taxon that matches in every stratum
         simultaneously is much harder and leads to poor scalar matches
         even when the vector distance looks acceptable.

    Returns
    -------
    dist      : ndarray  shape (n_list, n_bg) — absolute scalar differences
    p_list    : ndarray  shape (n_list,)       — list weighted-mean prevalences
    p_bg      : ndarray  shape (n_bg,)         — background weighted-mean prevalences
    """
    shared = list_prev_mat.columns.intersection(bg_prev_mat.columns)
    if shared.empty:
        n_l, n_b = len(list_prev_mat), len(bg_prev_mat)
        return (np.full((n_l, n_b), np.inf),
                np.zeros(n_l), np.zeros(n_b))

    w = weights.reindex(shared).fillna(0).values
    w /= max(w.sum(), 1e-12)

    L = list_prev_mat[shared].fillna(0).values   # (n_list, n_strata)
    B = bg_prev_mat[shared].fillna(0).values      # (n_bg,   n_strata)

    p_list = L @ w   # (n_list,)
    p_bg   = B @ w   # (n_bg,)

    dist = np.abs(p_list[:, np.newaxis] - p_bg[np.newaxis, :])  # (n_list, n_bg)
    return dist, p_list, p_bg


def _compute_iw_background(p_list, p_bg_all, bg_taxa_list,
                            n_target=None, weight_clip=10.0,
                            min_weight_frac=0.01):
    """
    Importance weights w_i = f_list(p_bg_i) / f_bg(p_bg_i) for the entire
    background pool.  Converts to integer counts via largest-remainder
    rounding.  Returns (match_counts, kde_list, kde_bg, w_cont).
    """
    from scipy.stats import gaussian_kde as _gkde_iw
    n_list = len(p_list); n_bg = len(p_bg_all)
    if n_target is None:
        n_target = max(10 * n_list, 2 * n_bg)
    kde_list = kde_bg = None
    try: kde_list = _gkde_iw(p_list,   bw_method="silverman")
    except Exception: pass
    try: kde_bg   = _gkde_iw(p_bg_all, bw_method="silverman")
    except Exception: pass
    if kde_list is not None and kde_bg is not None:
        f_list_vals = np.maximum(kde_list(p_bg_all), 0.0)
        f_bg_vals   = np.maximum(kde_bg(p_bg_all),   0.0)
        # Regularise: floor f_bg at its 5th percentile across all background
        # taxon positions.  When the bg pool has no taxa at certain prevalences
        # (e.g. no non-BloSSUM taxa at very high prevalences), f_bg → 0 there
        # causing w → ∞ before the clip is applied.  Flooring prevents this
        # while having negligible effect where f_bg is well-supported.
        f_bg_pos   = f_bg_vals[f_bg_vals > 0]
        f_bg_floor = float(np.percentile(f_bg_pos, 5)) if len(f_bg_pos) >= 10 else 1e-10
        f_bg_reg   = np.maximum(f_bg_vals, f_bg_floor)
        w_cont = f_list_vals / f_bg_reg
    else:
        w_cont = np.ones(n_bg, dtype=float)
    w_mean = w_cont.mean()
    w_cont[w_cont < min_weight_frac * w_mean] = 0.0
    if (w_cont > 0).sum() < 5:
        w_cont = np.ones(n_bg, dtype=float)
    clip_ceil = weight_clip * w_cont[w_cont > 0].mean()
    w_cont    = np.clip(w_cont, 0.0, clip_ceil)
    w_mean    = w_cont.mean()
    if w_mean > 1e-12: w_cont /= w_mean
    w_scaled  = w_cont * (n_target / max(w_cont.sum(), 1e-12))
    floors    = np.floor(w_scaled).astype(int)
    residuals = w_scaled - floors
    shortfall = int(n_target - floors.sum())
    if shortfall > 0:
        floors[np.argpartition(residuals, -shortfall)[-shortfall:]] += 1
    elif shortfall < 0:
        idx = np.argpartition(residuals, -shortfall)[:-shortfall]
        floors[idx] = np.maximum(floors[idx] - 1, 0)
    match_counts = {bg_taxa_list[i]: int(floors[i])
                    for i in range(n_bg) if floors[i] > 0}
    return match_counts, kde_list, kde_bg, w_cont


def _iw_data_from_table(df):
    """Rebuild the IW diagnostic inputs from the published table.

    The kernel densities are fitted here rather than stored: they are a
    deterministic function of the prevalences in the table, and a pickled
    scipy estimator would tie the data file to a scipy version.
    """
    from scipy.stats import gaussian_kde as _gkde_iw
    out = {}
    for (cond, ln), sub in df.groupby(["condition", "taxon_list"], sort=False):
        entry = {}
        for series, key in (("taxon list", "p_list"), ("background", "p_bg")):
            v = sub.loc[sub["series"] == series,
                        "control_prevalence"].to_numpy(dtype=float)
            if len(v):
                entry[key] = v
        if "p_list" not in entry or "p_bg" not in entry:
            continue
        w = sub.loc[sub["series"] == "background",
                    "importance_weight"].to_numpy(dtype=float)
        if len(w) == len(entry["p_bg"]) and np.isfinite(w).any():
            entry["w_cont"] = w
        if "n_strata" in sub.columns:
            entry["n_strata"] = float(sub["n_strata"].iloc[0])
        for key, src in (("kde_list", "p_list"), ("kde_bg", "p_bg")):
            try:
                entry[key] = _gkde_iw(entry[src], bw_method="silverman")
            except Exception:
                entry[key] = None
        out[(cond, ln)] = entry
    return out


def plot_iw_diagnostics(tables, spec_tag="10pct", s3_main="."):
    """Draw the IW diagnostics from the published table, and nothing else."""
    return _plot_iw_diagnostics(_iw_data_from_table(tables["iw"]),
                                spec_tag, s3_main, emit=False, atlases=False)


def _size_iw_text(fig, headers, notes, keys,
                  header_pt=8.5, note_pt=7.0, key_pt=6.0):
    """Set S25's headers, notes and keys in printed points, then fix them.

    The shared type pass sizes every annotation alike, which left the
    condition headers no larger than the tick labels; and a key it shrinks
    keeps the spacing of the size it was made at, which is the white space
    around these. Both are set here at their printed size and marked so the
    pass leaves them alone.
    """
    from .plot_utils import PAGE_WIDTH_IN, FONT_STACK
    try:
        fig.canvas.draw()
        w_in = float(fig.get_tightbbox(fig.canvas.get_renderer()).width)
    except Exception:
        w_in = float(fig.get_size_inches()[0])
    k = w_in / PAGE_WIDTH_IN          # figure points per printed point
    for t, pt in [(h, header_pt) for h in headers] + [(n, note_pt) for n in notes]:
        t.set_fontsize(pt * k)
        t.set_fontfamily(FONT_STACK)
        t._fixed_pt = True
    for ax, loc, anchor in keys:
        leg = ax.legend(loc=loc, bbox_to_anchor=anchor,
                        prop={"family": FONT_STACK, "size": key_pt * k},
                        frameon=True, framealpha=0.85, borderpad=0.3,
                        labelspacing=0.25, handlelength=1.3, handleheight=0.6,
                        handletextpad=0.4, borderaxespad=0.2)
        leg.get_frame().set_linewidth(0.4)
        leg._fixed_loc = True
        for t in leg.get_texts():
            t._fixed_pt = True


def _plot_iw_diagnostics(iw_data, spec_tag, s3_main, emit=True, prefix="health", atlases=True):
    """
    Three-row per-list diagnostic figure.
    Row A: f_list vs f_bg (raw mismatch).
    Row B: w(p) = f_list/f_bg (green=upweight, red=downweight, ESS%).
    Row C: f_bg*w normalised vs f_list (OVL>=95% green = correct IW).
    Saved: health_iw_diagnostics_{spec_tag}.png
    """
    if not iw_data: return
    all_conds_full = list(dict.fromkeys(c for c, _ in iw_data))
    # The conditions Figure 5D-E plots: those resting on at least
    # MIN_STRATA_TO_PLOT studies. This was a hardcoded list and drifted from
    # Figure 5 when the cohorts changed -- it kept MASLD (2 studies) and dropped
    # CRC / Adenoma (8). A table published before n_strata was recorded has no
    # counts, so it keeps every condition rather than guessing.
    _k = {}
    for (c, _), e in iw_data.items():
        if e.get("n_strata") is not None and np.isfinite(e["n_strata"]):
            _k[c] = int(e["n_strata"])
    if _k:
        all_conds = [c for c in all_conds_full
                     if _k.get(c, 0) >= config.MIN_STRATA_TO_PLOT]
        dropped = [f"{c} ({_k.get(c, 0)})" for c in all_conds_full
                   if c not in all_conds]
        if dropped:
            print(f"    [IW diagnostics] not shown, fewer than "
                  f"{config.MIN_STRATA_TO_PLOT} studies: {', '.join(dropped)}")
    else:
        all_conds = all_conds_full
    # Deliberately NOT passed through _display_order. This figure only runs
    # under HEALTH_BG_METHOD = "iw", whose entire purpose is reproducing the
    # 29 Sep figures exactly; reordering its columns would defeat that.
    all_lists_full = list(dict.fromkeys(l for _, l in iw_data))
    # The full six-list atlas is too dense for a Word-page additional figure.
    # Keep the canonical output focused on the health-restricted lists, and
    # save one-list atlases below for detailed inspection.
    focus_lists = [l for l in ("VanishH+", "BlossumH+") if l in all_lists_full]
    all_lists = focus_lists if focus_lists else all_lists_full
    n_cond, n_lists = len(all_conds), len(all_lists)
    if not n_cond or not n_lists: return
    N_ROWS  = 3
    FS_IW   = FS + 3
    fmt_pct = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    CLIP    = 10.0
    # Sized in printed points at the end, once the layout is fixed: the
    # condition headers and the notes stand above the tick labels, and the
    # keys are built at their final size so their spacing matches their text.
    _headers, _notes, _keys = [], [], []
    def _panel_header(ax, label):
        _headers.append(ax.text(0.5, 1.04, label, transform=ax.transAxes,
                                ha="center", va="bottom", fontsize=FS_IW,
                                fontweight="bold", clip_on=False))
    fig, axes = plt.subplots(n_lists * N_ROWS, n_cond,
                              figsize=page_figsize(
                                  n_cols=n_cond,
                                  height_in=n_lists * N_ROWS * 1.55),
                              squeeze=False)
    for li, list_name in enumerate(all_lists):
        col = LIST_COLORS.get(list_name, "#888888")
        r_d, r_w, r_c = li*N_ROWS, li*N_ROWS+1, li*N_ROWS+2
        for ci, cond in enumerate(all_conds):
            ax_d, ax_w, ax_c = axes[r_d][ci], axes[r_w][ci], axes[r_c][ci]
            entry = iw_data.get((cond, list_name))
            if entry is None or len(entry.get("p_list", [])) < 3:
                for ax in (ax_d, ax_w, ax_c):
                    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                            ha="center", va="center", fontsize=FS_IW-1, color="gray")
                    ax.set_xticks([]); ax.set_yticks([])
                if ci == 0:
                    ax_d.set_ylabel(f"{list_name}\nprevalence", fontsize=FS_IW-1, fontweight="bold", color=col)
                    ax_w.set_ylabel(f"{list_name}\nweight",     fontsize=FS_IW-1, fontweight="bold", color=col)
                    ax_c.set_ylabel(f"{list_name}\ncheck",      fontsize=FS_IW-1, fontweight="bold", color=col)
                if li == 0: _panel_header(ax_d, cond)
                continue
            p_list = np.asarray(entry["p_list"]); p_bg = np.asarray(entry["p_bg"])
            kde_list = entry.get("kde_list"); kde_bg = entry.get("kde_bg")
            w_cont   = np.asarray(entry.get("w_cont", np.ones(len(p_bg))))
            xlo, xhi = 0.0, max(float(p_list.max()), float(p_bg.max())) * 1.06 + 0.005
            xg = np.linspace(xlo, xhi, 600); dx = xg[1] - xg[0]
            def _safe(kde, x):
                if kde is None: return np.zeros_like(x)
                try: return np.maximum(kde(x), 0.0)
                except: return np.zeros_like(x)
            f_l = _safe(kde_list, xg); f_b = _safe(kde_bg, xg)
            # Row A
            ax_d.fill_between(xg, f_l, alpha=0.25, color=col,      zorder=2)
            ax_d.fill_between(xg, f_b, alpha=0.15, color="#555555", zorder=1)
            ax_d.plot(xg, f_l, color=col,       lw=2.1, label="target")
            ax_d.plot(xg, f_b, color="#555555", lw=1.7, ls="--", label="background")
            ax_d.axvline(float(np.median(p_list)), color=col,       lw=1.4, alpha=0.85, ls=":")
            ax_d.axvline(float(np.median(p_bg)),   color="#555555", lw=1.2, alpha=0.75, ls=":")
            ax_d.set_xlim(xlo, xhi); ax_d.xaxis.set_major_formatter(fmt_pct)
            ax_d.tick_params(labelsize=FS_IW-2); ax_d.grid(axis="x", alpha=0.10)
            # Top right: the densities rise from the left edge and run along
            # the bottom, so the bottom right sat on them.
            _notes.append(ax_d.text(0.97, 0.96, f"n={len(p_list)}\nbg={len(p_bg)}",
                                    transform=ax_d.transAxes, ha="right", va="top",
                                    fontsize=FS_IW-2, color="0.35"))
            if ci == 0:
                # Below the note, at mid-height: clear of the left-hand peak
                # in both lists.
                _keys.append((ax_d, "center right", (0.99, 0.55)))
                ax_d.set_ylabel(f"{list_name}\nprevalence", fontsize=FS_IW-1, fontweight="bold", color=col)
            if li == 0: _panel_header(ax_d, cond)
            # Row B
            f_bs   = np.where(f_b > 1e-10, f_b, np.nan)
            w_grid = np.where(np.isfinite(f_bs), f_l / f_bs, np.nan)
            wm     = float(np.nanmean(w_grid))
            w_grid = np.where(np.isfinite(w_grid), np.minimum(w_grid, CLIP * wm), np.nan)
            wp     = np.where(np.isfinite(w_grid), w_grid, 0.0)
            ax_w.plot(xg, np.where(np.isfinite(w_grid), w_grid, np.nan), color=col, lw=1.8, zorder=4)
            ax_w.fill_between(xg, wp, alpha=0.18, color=col, zorder=1)
            ax_w.fill_between(xg, np.where(w_grid>1, w_grid, 1.0), 1.0,
                              where=np.isfinite(w_grid)&(w_grid>1), alpha=0.28, color=config.OI["sky"], zorder=2, label="up")
            ax_w.fill_between(xg, np.where(w_grid<1, w_grid, 0.0), np.where(w_grid<1, 1.0, 0.0),
                              where=np.isfinite(w_grid)&(w_grid<1), alpha=0.28, color=config.OI["orange"], zorder=2, label="down")
            ax_w.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.55, label="w = 1")
            ax_w.set_xlim(xlo, xhi); ax_w.set_ylim(bottom=0)
            ax_w.xaxis.set_major_formatter(fmt_pct); ax_w.tick_params(labelsize=FS_IW-2); ax_w.grid(axis="x", alpha=0.10)
            if len(w_cont) > 0 and w_cont.sum() > 0:
                ess = (w_cont.sum()**2) / ((w_cont**2).sum() * len(w_cont))
                # Top middle: the weight curves peak at the left (VanishH+)
                # or at the far right (BlossumH+), and the upper left put the
                # note on the VanishH+ peak.
                _notes.append(ax_w.text(0.5, 0.96, f"ESS={ess:.0%}", transform=ax_w.transAxes,
                                        ha="center", va="top", fontsize=FS_IW-2, fontweight="bold",
                                        color=config.OI["sky"] if ess > 0.30 else config.OI["orange"]))
            if ci == 0:
                # Row 2's weights sit at the left, row 5's at the far right.
                _keys.append((ax_w, "upper right", (0.99, 0.84)) if li == 0
                             else (ax_w, "center left", (0.01, 0.5)))
                ax_w.set_ylabel(f"{list_name}\nweight", fontsize=FS_IW-1, fontweight="bold", color=col)
            # Row C
            rw  = np.where(np.isfinite(w_grid), f_b * w_grid, 0.0)
            rw /= max(float(np.trapz(rw, dx=dx)), 1e-12)
            fl_n = f_l / max(float(np.trapz(f_l, dx=dx)), 1e-12)
            ov = float(np.trapz(np.minimum(rw, fl_n), dx=dx))
            good = ov > 0.95
            ax_c.fill_between(xg, fl_n, alpha=0.30, color=col,      zorder=2, label="target")
            ax_c.fill_between(xg, rw,   alpha=0.22, color="#222222", zorder=1, label="weighted bg")
            ax_c.fill_between(xg, np.minimum(rw, fl_n), alpha=0.40, color="#bbbbbb", zorder=3, label="overlap")
            ax_c.plot(xg, fl_n, color=col,      lw=1.6, zorder=5)
            ax_c.plot(xg, rw,   color="#222222", lw=1.4, ls="--", zorder=4)
            ax_c.set_xlim(xlo, xhi); ax_c.xaxis.set_major_formatter(fmt_pct)
            ax_c.tick_params(labelsize=FS_IW-2); ax_c.grid(axis="x", alpha=0.10)
            _notes.append(ax_c.text(0.97, 0.96, f"OVL={ov:.0%}", transform=ax_c.transAxes,
                                    ha="right", va="top", fontsize=FS_IW-2, fontweight="bold",
                                    color=config.OI["sky"] if good else config.OI["orange"]))
            if ci == 0 and li == 0:
                # Under the OVL note, where row 3's overlap (at the left)
                # leaves room. Row 6 is a broad hump with no clear corner;
                # its key would repeat this one, directly above it.
                _keys.append((ax_c, "upper right", (0.99, 0.84)))
                ax_c.set_ylabel(f"{list_name}\ncheck", fontsize=FS_IW-1, fontweight="bold", color=col)
            # One axis label, on the bottom row: every panel shares the axis.
            if li == n_lists - 1:
                ax_c.set_xlabel("Control prevalence", fontsize=FS_IW-1)
    fig.suptitle(
        f"IW diagnostics  |  {spec_tag}",
        fontsize=FS_IW+1, fontweight="bold")
    plt.tight_layout()
    _size_iw_text(fig, _headers, _notes, _keys)
    # The figure's data object. One tidy table: the grid is condition x taxon
    # list, so those are columns rather than separate files. The KDE curves and
    # the ESS and overlap percentages are all derived from these prevalences
    # and weights, so storing them too would be a second copy of the same
    # numbers.
    if emit and spec_tag == "10pct":
        _rows = []
        for (_cond, _ln), _e in iw_data.items():
            for _series, _key in (("taxon list", "p_list"),
                                  ("background", "p_bg")):
                _v = np.asarray(_e.get(_key, []), dtype=float)
                if not len(_v):
                    continue
                _w = np.asarray(_e.get("w_cont", []), dtype=float)
                _rows.append(pd.DataFrame({
                    "condition": _cond,
                    "taxon_list": _ln,
                    "n_strata": _e.get("n_strata", np.nan),
                    "series": _series,
                    "control_prevalence": _v,
                    "importance_weight": (_w if _series == "background"
                                          and len(_w) == len(_v) else np.nan),
                }))
        # Not a numbered figure since the curve replaced it, so it publishes
        # under its own name. It claimed S25 until 2026-10-07, which would
        # now overwrite the prevalence-matched figure's data object on any
        # run with HEALTH_BG_METHOD="iw".
        if _rows and prefix == "health":
            figure_source.emit(
                s3_main, "IWdiag", {"iw": pd.concat(_rows, ignore_index=True)},
                stem=f"health_iw_diagnostics_{spec_tag}",
                notes="Grid is condition x taxon list; weights apply to the "
                      "background series.")

    # Five panels across the page share an axis range, so the last tick of one
    # panel lands against the first tick of the next. Drop it.
    fig._plain_axes_kw = dict(x_ticks=3, prune="upper")
    out = os.path.join(s3_main, f"{prefix}_iw_diagnostics_{spec_tag}.png")
    savefig_dual(fig, out, dpi=SUBMISSION_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved IW diagnostics: {out}")
    if atlases and len(all_lists_full) > len(all_lists):
        _plot_iw_diagnostics_by_list(iw_data, spec_tag, s3_main,
                                     all_lists_full, prefix=prefix)


def _plot_iw_diagnostics_by_list(iw_data, spec_tag, s3_main, all_lists,
                                 prefix="health"):
    """Emit one-list IW diagnostic atlases using the same enlarged layout."""
    if not iw_data:
        return
    for list_name in all_lists:
        sub_data = {
            (cond, ln): entry
            for (cond, ln), entry in iw_data.items()
            if ln == list_name
        }
        if not sub_data:
            continue
        safe = list_name.replace("+", "plus").replace(" ", "_")
        # Carry the prefix: without it the demographic pass writes its
        # per-list atlases over the disease run's.
        _plot_iw_diagnostics(sub_data, f"{safe}_{spec_tag}", s3_main,
                             prefix=prefix)


def _run_prevalence_matched_background(data, taxon_lists, per_taxon_df,
                                        spec, s3_main, demographic=False,
                                        strat=None, curve=None):
    """The background each list taxon is compared against.

    `config.HEALTH_BG_METHOD` picks the method:

      "curve"  (default, corrected 2026-10-06) every list taxon brings its own
               prevalence-matched set of background taxa, flat-kernel, nothing
               rounded and nothing dropped -- `_run_matched_background_curve`.
      "iw"     the published importance-weighted background, kept so the 29 Sep
               figures can be reproduced -- `_run_prevalence_matched_background_iw`.
    """
    if _bg_method() == "iw":
        return _run_prevalence_matched_background_iw(
            data, taxon_lists, per_taxon_df, spec, s3_main,
            demographic=demographic)
    return _run_matched_background_curve(
        data, taxon_lists, per_taxon_df, spec, s3_main,
        demographic=demographic, strat=strat, curve=curve)


def _run_matched_background_curve(data, taxon_lists, per_taxon_df, spec,
                                   s3_main, demographic=False, strat=None,
                                   curve=None):
    """Prevalence-matched background taxa, with no weights to get wrong.

    For each condition x list, `health_background.matched_background` gives
    every background taxon within one bandwidth of some list taxon's control
    prevalence, with the weight that makes a weighted statistic over the
    background equal the mean, over list taxa, of that statistic inside each
    taxon's own matched set. Against the method it replaces: no KDE ratio, no
    5th-percentile floor, no 10x clip, no rounding to whole copies, so no
    background taxon is silently dropped -- and the rows are unique, so
    nothing is double-counted either.

    The per-taxon CMH runs once per condition over the union of every list's
    matched taxa, rather than once per list: the p-value of a background taxon
    does not depend on which list matched it.

    Emits the same frame shape the IW path did (`Taxon_list` = "<list>_bg"),
    with `bg_weight` in place of `importance_weight` and the stratified shift
    columns alongside.
    """
    if per_taxon_df is None or per_taxon_df.empty:
        return pd.DataFrame()
    bdf, thresh = _build_presence_matrix(data, spec)
    meta        = data.meta
    spec_tag    = spec.short_label
    comparisons = _health_comparisons(demographic)
    PLOT_LISTS = ["VANISH", "BloSSUM", "VANISH+", "BloSSUM+", "VanishH+", "BlossumH+"]

    if strat is None:
        strat = _run_stratified_shifts(data, spec, s3_main,
                                       demographic=demographic)
    curve_taxa = (curve or {}).get("taxa")
    if curve_taxa is None or getattr(curve_taxa, "empty", True):
        curve_taxa = pd.DataFrame()

    all_rows = []
    for cond_label, (comp_type, dcols) in comparisons.items():
        if cond_label not in strat:
            continue
        taxa_df, strata_df = strat[cond_label]
        n_strata = int(len(strata_df))
        mask_g1, mask_g2 = build_group_masks(meta, comp_type, dcols)
        n_g1, n_g2 = int(mask_g1.sum()), int(mask_g2.sum())
        frame = hbg.shift_frame(taxa_df)

        matched = {}
        for list_name in PLOT_LISTS:
            if list_name not in taxon_lists:
                continue
            members = [m for m in taxon_lists[list_name] if m in frame.index]
            if len(members) < 5:
                continue
            mb = hbg.matched_background(frame, members)
            if len(mb) < 5:
                print(f"    [PREVMATCH] {cond_label}/{list_name}: "
                      f"only {len(mb)} matched bg taxa — skipping")
                continue
            matched[list_name] = (members, mb)
        if not matched:
            continue

        union = sorted(set().union(*(set(mb.index) for _, mb in matched.values())))
        cmh = geom_mean_weighted_cmh_per_taxon(
            bdf, union, meta, mask_g1, mask_g2, thresh)
        cmh = (cmh.set_index("taxon") if cmh is not None and not cmh.empty
               else pd.DataFrame())
        strat_idx = taxa_df.set_index("taxon")

        for list_name, (members, mb) in matched.items():
            rows = pd.DataFrame(index=mb.index)
            rows["taxon"] = mb.index
            for c in ("rd_mh", "rd_mh_se", "t_mh", "p_mh",
                      "p_ctrl_mh", "p_case_mh", "pooled_diff"):
                if c in strat_idx.columns:
                    rows[c] = strat_idx[c].reindex(mb.index).values
            for c in ("prev_g1", "prev_g2", "prev_diff", "log_or",
                      "pval", "direction"):
                rows[c] = (cmh[c].reindex(mb.index).values
                           if c in cmh.columns else np.nan)
            rows["bg_weight"]  = mb["match_weight"].values
            rows["match_count"] = mb["match_count"].values
            rows["ctrl_prev"]  = frame["p"].reindex(mb.index).values
            rows["Condition"]  = cond_label
            rows["Taxon_list"] = f"{list_name}_bg"
            rows["n_g1"], rows["n_g2"] = n_g1, n_g2
            rows["n_strata"] = n_strata
            all_rows.append(rows.reset_index(drop=True))

            ess = (mb["match_weight"].sum() ** 2
                   / (mb["match_weight"] ** 2).sum())
            print(f"    [PREVMATCH] {cond_label} / {list_name}: "
                  f"list n={len(members)}, matched bg={len(mb)}, ESS={ess:.0f}")

    if not all_rows:
        return pd.DataFrame()
    bg_df = pd.concat(all_rows, ignore_index=True)
    csv_path = os.path.join(
        s3_main,
        f"{_out_prefix(demographic)}_per_taxon_prevmatched_{spec_tag}.csv")
    bg_df.to_csv(csv_path, index=False)
    print(f"    Saved prevalence-matched background: {csv_path}")
    # The VanishH+/BlossumH+ background curve is main-text Figure 5E now, so
    # it is not drawn here; S26 is the same figure for the base VANISH and
    # BloSSUM lists, and it belongs in the supplement directory beside the
    # other base-list figures. The demographic pass draws the comparison for
    # its own strata but is not S26 and must not publish itself as S26's data
    # object.
    if curve and not demographic:
        _supp = os.path.join(os.path.dirname(s3_main), "supplement")
        os.makedirs(_supp, exist_ok=True)
        _ref = getattr(data, "ref_name", "combined")
        _plot_background_curve(curve, bg_df, spec_tag, _supp,
                               prefix=_out_prefix(demographic),
                               ref=_ref,
                               spec_desc=getattr(spec, "description", ""),
                               focus_lists=["VANISH", "BloSSUM"],
                               slot="S25",
                               stem=f"health_background_curve_base_"
                                    f"{_ref}_{spec_tag}")
    return bg_df


def _run_prevalence_matched_background_iw(data, taxon_lists, per_taxon_df,
                                          spec, s3_main, demographic=False):
    """
    Stratum-aware importance-weighted background for GMW-CMH directional
    shift comparisons. SUPERSEDED -- reachable only via
    `config.HEALTH_BG_METHOD = "iw"`, to reproduce the 29 Sep figures.

    Uses IW ratio w_i = f_list(p_bg_i) / f_bg(p_bg_i) on the ENTIRE
    background pool (no K-NN cap).  The continuous w_cont is stored in
    bg_df as 'importance_weight' for IW-aware tests and plots.
    Diagnostic: health_iw_diagnostics_{spec_tag}.png (replaces old QC fig).
    """
    K_NEIGHBOURS = getattr(config, "PREVMATCH_K_NEIGHBOURS", 10)
    if per_taxon_df is None or per_taxon_df.empty:
        return pd.DataFrame()
    bdf, thresh = _build_presence_matrix(data, spec)
    meta        = data.meta
    spec_tag    = spec.short_label
    comparisons = _health_comparisons(demographic)
    PLOT_LISTS = ["VANISH", "BloSSUM", "VANISH+", "BloSSUM+", "VanishH+", "BlossumH+"]
    all_rows = []
    iw_data  = {}
    for cond_label, (comp_type, dcols) in comparisons.items():
        mask_g1, mask_g2 = build_group_masks(meta, comp_type, dcols)
        n_g1, n_g2 = int(mask_g1.sum()), int(mask_g2.sum())
        if n_g1 < 10 or n_g2 < 10: continue
        prev_mat, weights = _stratum_prevalence_matrix(bdf, meta, mask_g1, mask_g2, thresh)
        n_strata = 0 if prev_mat.empty else len(prev_mat.columns)
        if prev_mat.empty:
            print(f"    [PREVMATCH] {cond_label}: 0 qualifying study strata — falling back to pooled")
            g1_idx    = meta.index[mask_g1 & meta.index.isin(bdf.index)]
            ctrl_prev = (bdf.loc[g1_idx] >= thresh).mean(axis=0)
            prev_mat  = pd.DataFrame({"_pooled": ctrl_prev})
            weights   = pd.Series({"_pooled": 1.0})
        else:
            print(f"    [PREVMATCH] {cond_label}: {len(prev_mat.columns)} qualifying study strata")
        for list_name in PLOT_LISTS:
            if list_name not in taxon_lists: continue
            mics = [m for m in taxon_lists[list_name] if m in bdf.columns]
            if not mics: continue
            list_excl  = set(mics)
            bg_pool    = [t for t in bdf.columns if t not in list_excl]
            if len(bg_pool) < 20:
                print(f"    [PREVMATCH] {cond_label}/{list_name}: too few bg taxa — skipping")
                continue
            bg_prev_mat   = prev_mat.loc[prev_mat.index.intersection(bg_pool)].copy()
            list_prev_mat = prev_mat.loc[prev_mat.index.intersection(mics)].copy()
            if list_prev_mat.empty or bg_prev_mat.empty: continue
            _, p_list, p_bg_all = _scalar_prevalence_distance(list_prev_mat, bg_prev_mat, weights)
            bg_taxa_list = list(bg_prev_mat.index)
            n_target     = K_NEIGHBOURS * len(p_list)
            match_counts, kde_list, kde_bg, w_cont = _compute_iw_background(
                p_list, p_bg_all, bg_taxa_list, n_target=n_target)
            unique_bg = sorted(match_counts.keys())
            n_bg_rows = sum(match_counts.values())
            max_reuse = max(match_counts.values()) if match_counts else 0
            if len(unique_bg) < 5:
                print(f"    [PREVMATCH] {cond_label}/{list_name}: too few surviving bg taxa — skipping")
                continue
            print(f"    [PREVMATCH] {cond_label} / {list_name}: "
                  f"list n={len(p_list)}, unique bg={len(unique_bg)}/{len(bg_pool)}, "
                  f"bg rows={n_bg_rows}, max_reuse={max_reuse}")
            iw_data[(cond_label, list_name)] = dict(
                p_list=p_list, p_bg=p_bg_all,
                kde_list=kde_list, kde_bg=kde_bg, w_cont=w_cont,
                n_strata=n_strata)
            pt_bg_unique = geom_mean_weighted_cmh_per_taxon(
                bdf, unique_bg, meta, mask_g1, mask_g2, thresh)
            if pt_bg_unique is None or pt_bg_unique.empty: continue
            bg_p_map = dict(zip(bg_taxa_list, p_bg_all))
            bg_w_map = dict(zip(bg_taxa_list, w_cont))
            pt_bg_unique = pt_bg_unique.set_index("taxon")
            replicated = []
            for bg_t, cnt in match_counts.items():
                if bg_t not in pt_bg_unique.index: continue
                row_data = pt_bg_unique.loc[[bg_t]].copy()
                row_data["importance_weight"] = float(bg_w_map.get(bg_t, 1.0))
                row_data["ctrl_prev"]         = float(bg_p_map.get(bg_t, np.nan))
                if cnt > 1:
                    row_data = pd.concat([row_data] * cnt, ignore_index=True)
                replicated.append(row_data)
            if not replicated: continue
            pt_bg = pd.concat(replicated, ignore_index=True)
            pt_bg["taxon"] = [t for t, cnt in match_counts.items()
                              if t in pt_bg_unique.index for _ in range(cnt)]
            pt_bg["Condition"]  = cond_label
            pt_bg["Taxon_list"] = f"{list_name}_bg"
            pt_bg["n_g1"] = n_g1; pt_bg["n_g2"] = n_g2
            # The study count the contrast rests on, so a figure drawn from
            # this frame alone can apply Figure 5's MIN_STRATA_TO_PLOT rule.
            pt_bg["n_strata"] = n_strata
            all_rows.append(pt_bg)
    if not all_rows:
        return pd.DataFrame()
    bg_df    = pd.concat(all_rows, ignore_index=True)
    csv_path = os.path.join(
        s3_main,
        f"{_out_prefix(demographic)}_per_taxon_prevmatched_{spec_tag}.csv")
    bg_df.to_csv(csv_path, index=False)
    print(f"    Saved IW-matched background: {csv_path}")
    if iw_data:
        _plot_iw_diagnostics(iw_data, spec_tag, s3_main,
                             prefix=_out_prefix(demographic))
    return bg_df

def rebuild_background_curve(panels, out_dir, spec_tag="10pct",
                             ref="combined", notes=""):
    """Redraw Figure S25 from its published data object (tools/replot_from_data).

    The object carries the fitted curve and every taxon's coordinates, so the
    redraw never refits anything -- which is the point: a redraw has to be the
    same figure, not a new fit of the same data.
    """
    taxa = panels.get("list_taxa")
    if taxa is None or taxa.empty:
        return None
    frames = {}
    all_taxa = panels.get("all_taxa")
    if all_taxa is not None and not all_taxa.empty:
        for cond, grp in all_taxa.groupby("Condition"):
            g = grp.set_index("taxon")
            frames[cond] = g[[c for c in ("p", "shift", "se", "x",
                                          "p_case", "x_case")
                              if c in g.columns]]
    curve = {"taxa": taxa, "grid": panels.get("curve", pd.DataFrame()),
             "summary": panels.get("summary", pd.DataFrame()),
             "frames": frames}
    _stem = f"health_background_curve_base_{ref}_{spec_tag}"
    _plot_background_curve(curve, None, spec_tag, out_dir, prefix="health",
                           ref=ref, spec_desc=notes,
                           focus_lists=["VANISH", "BloSSUM"],
                           slot="S25", stem=_stem)
    return os.path.join(out_dir, _stem + ".png")


def rebuild_within_clade_contrasts(panels, out_dir, spec_tag="10pct",
                                   ref="combined", notes=""):
    """Redraw Figure S30 from its published data object.

    `panels["panels"]` records which contrasts the published figure drew, so a
    redraw shows those and not whichever ones happen to be strongest now. The
    figure is written into `out_dir` itself: for a `--results-root` redraw that
    is the supplement directory it came from, and for a preview it is the
    preview directory, which must not be escaped.
    """
    within = panels.get("contrasts")
    points = panels.get("points")
    if within is None or within.empty or points is None or points.empty:
        return None
    wanted = None
    sel = panels.get("panels")
    if sel is not None and not sel.empty:
        wanted = [(r.Condition, r.rank, r.clade, r.list)
                  for r in sel.itertuples(index=False)]
    clade_data = {"within": within, "points": points,
                  "clades": panels.get("clades", pd.DataFrame())}
    # _plot_within_clade_contrasts writes into <parent>/supplement, so hand it
    # a main-like directory and let it put the figure where the manifest says.
    _plot_within_clade_contrasts(clade_data, out_dir, spec_tag, ref=ref,
                                 spec_desc=notes, panels=wanted,
                                 out_dir=out_dir)
    return os.path.join(out_dir,
                        f"health_within_clade_contrasts_{ref}_{spec_tag}.png")


def _run_clade_contrasts(curve, taxon_lists, data, s3_main, spec_tag,
                         ref="combined", spec_desc="", panels=None):
    """Figure S30: do list taxa differ from their own close relatives?

    The lists are not spread evenly over the tree -- VANISH is thick with
    *Prevotella*, BloSSUM with Lachnospiraceae -- so "list taxa behave
    differently in disease" could be nothing more than "those families behave
    differently in disease". This holds taxonomy fixed and asks the question
    inside each clade: within one genus or family, do the members on a list sit
    further from the background curve than the members that are not?

    Both halves of the comparison are computed with the clade itself excluded
    from the curve, so a large clade is not partly its own control, and taxa
    above config.HEALTH_BG_HIGHPREV_LIMIT are dropped because the catalogue has
    no comparable taxa for them.

    Writes the whole table (every genus and family with at least
    config.HEALTH_BG_CLADE_MIN_TAXA members) and draws the contrasts named in
    config.HEALTH_BG_CLADE_PANELS.

    `data` is unused: the clade labels come from the Section 3 taxonomy map, the
    same one S26-S29 label taxa with. It is in the signature so both callers
    pass the same arguments they pass every other step.
    """
    if spec_tag != "10pct":            # one pass, on the main detection rule
        return {}
    frames = (curve or {}).get("frames") or {}
    if not frames:
        return {}

    # The conditions the figure can show, i.e. the ones Figure 5 and S25 show:
    # a clade contrast on a two-study condition is not worth the minutes it
    # costs, and the panels this figure draws come from the primary five.
    summ = (curve or {}).get("summary")
    if summ is not None and not summ.empty and "n_strata" in summ.columns:
        k = summ.groupby("Condition")["n_strata"].max()
        keep = [c for c in frames
                if k.get(c, 0) >= config.MIN_STRATA_TO_PLOT]
        skipped = [c for c in frames if c not in keep]
        if skipped:
            print(f"    [CLADES] skipping {len(skipped)} condition(s) below "
                  f"{config.MIN_STRATA_TO_PLOT} studies: "
                  + ", ".join(sorted(skipped)))
        frames = {c: frames[c] for c in keep}
    frames = {c: f for c, f in frames.items() if not _condition_dropped(c)}
    if not frames:
        return {}

    tax_map = _load_taxonomy_map_s3()
    lists = [l for l in ["VANISH+", "VanishH+", "BloSSUM+", "BlossumH+",
                         "VANISH", "BloSSUM"] if l in taxon_lists]
    members = {l: set(taxon_lists[l]) for l in lists}
    if not members:
        return {}

    clade_rows, within_rows, point_rows = [], [], []
    for cond, frame in frames.items():
        if frame is None or frame.empty:
            continue
        lineage = hbg.lineage_frame(tax_map, list(frame.index))
        clades, within = hbg.clade_contrasts(frame, lineage, members)
        if not clades.empty:
            clades = clades.copy()
            clades["Condition"] = cond
            clade_rows.append(clades)
        if within.empty:
            continue
        within = within.copy()
        within["Condition"] = cond
        for r in within.itertuples(index=False):
            for t, z in zip(r.taxa_list, r.z_list):
                point_rows.append(dict(Condition=cond, rank=r.rank,
                                       clade=r.clade, list=r.list,
                                       taxon=t, z=float(z), on_list=True))
            for z in r.z_other:
                point_rows.append(dict(Condition=cond, rank=r.rank,
                                       clade=r.clade, list=r.list,
                                       taxon="", z=float(z), on_list=False))
        within_rows.append(within.drop(columns=["z_list", "z_other",
                                                "taxa_list"]))
        print(f"    [CLADES] {cond}: {len(clades)} clades, "
              f"{len(within)} within-clade contrasts, "
              f"{int((within.q < 0.05).sum())} at q<0.05")

    out = {
        "clades": (pd.concat(clade_rows, ignore_index=True)
                   if clade_rows else pd.DataFrame()),
        "within": (pd.concat(within_rows, ignore_index=True)
                   if within_rows else pd.DataFrame()),
        "points": pd.DataFrame(point_rows),
    }
    for key in ("clades", "within"):
        if not out[key].empty:
            p = os.path.join(s3_main, f"health_clade_{key}_{spec_tag}.csv")
            out[key].to_csv(p, index=False)
            print(f"    Saved: {p}")
    if not out["within"].empty:
        _plot_within_clade_contrasts(out, s3_main, spec_tag, ref=ref,
                                     spec_desc=spec_desc, panels=panels)
    return out


def _plot_within_clade_contrasts(clade_data, s3_main, spec_tag,
                                 ref="combined", spec_desc="", panels=None,
                                 out_dir=None):
    """Draw the within-clade contrasts named in config.HEALTH_BG_CLADE_PANELS.

    One panel per contrast: every member of the clade as a point, the list
    members beside the rest, each group's mean as a bar, and the three list
    members furthest from zero named. Saved into the supplement beside S28/S29
    and published as slot S30, or into `out_dir` when one is given -- a preview
    redraw must not write into the results tree.
    """
    panels = panels or getattr(config, "HEALTH_BG_CLADE_PANELS", [])
    within = clade_data.get("within")
    points = clade_data.get("points")
    if within is None or within.empty or points is None or points.empty:
        return
    tax_map = _load_taxonomy_map_s3()

    wanted = []
    for cond, rank, clade, list_name in panels:
        m = ((within.Condition == cond) & (within["rank"] == rank)
             & (within.clade == clade) & (within["list"] == list_name))
        if m.any():
            wanted.append((cond, rank, clade, list_name, within[m].iloc[0]))
    missing = [(c, r, cl, l) for c, r, cl, l in panels
               if not any(w[:4] == (c, r, cl, l) for w in wanted)]
    if missing:
        print("    [CLADES] config names contrasts this run does not have: "
              + "; ".join(f"{c}/{r} {cl} x {l}" for c, r, cl, l in missing))
    if not wanted:
        # Nothing the config names survived; fall back to the strongest
        # contrasts so the figure still describes this run.
        top = within.reindex(within["diff"].abs().sort_values(
            ascending=False).index).head(6)
        wanted = [(r.Condition, r["rank"], r.clade, r["list"], r)
                  for _, r in top.iterrows()]

    n = len(wanted)
    n_cols = 3 if n > 2 else n
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=page_figsize(aspect=0.95, n_cols=n_cols,
                                                  n_rows=n_rows))
    # A deeper gutter between rows than the shared default. Each panel carries
    # a three-line x-tick label -- "other / <clade> / (n = ...)" -- under it,
    # and at hspace 0.28 the "(n = ...)" line sat close enough to the next
    # row's bold title to read as belonging to the panel below it.
    if n_rows > 1:
        fig._margin_overrides = {"hspace": 0.52}
    rng = np.random.default_rng(9)

    def _short(t, cap=15):
        """A label that fits the panel: "B. fragilis", or genus + SGB number."""
        lab = _gtdb_short(tax_map.get(t, ""), n_ranks=1).replace("s__", "")
        if not lab:
            return str(t)[:cap]
        parts = lab.replace("_", " ").split()
        if len(parts) > 1 and not parts[1].startswith("sp"):
            lab = f"{parts[0][0]}. {' '.join(parts[1:])}"
        elif len(parts) > 1:
            # Unnamed species: the genus plus the last four digits of the SGB
            # id, which is what tells two of them apart on the same panel.
            lab = f"{parts[0]} sp{parts[1][-4:]}"
        return lab if len(lab) <= cap else lab[:cap - 1] + "\u2026"

    def _spread(vals, gap, lo=None, hi=None):
        """Label positions in the same order as `vals`, at least `gap` apart.

        Clamped into [lo, hi] afterwards, sliding the whole stack rather than
        squashing it, so a label never leaves the panel.
        """
        vals = np.asarray(vals, float)
        order = np.argsort(vals)
        pos = vals[order].copy()
        for k in range(1, len(pos)):
            pos[k] = max(pos[k], pos[k - 1] + gap)
        pos -= pos.mean() - vals[order].mean()
        if lo is not None and pos[0] < lo:
            pos += lo - pos[0]
        if hi is not None and pos[-1] > hi:
            pos -= pos[-1] - hi
        out = np.empty_like(pos)
        out[order] = pos
        return out

    lo, hi = -3.0, 6.6
    for k, (cond, rank, clade, list_name, row) in enumerate(wanted):
        ax = axes[k // n_cols][k % n_cols]
        colour = config.CMH_LIST_COLORS.get(list_name, "#333333")
        sel = points[(points.Condition == cond) & (points["rank"] == rank)
                     & (points.clade == clade) & (points["list"] == list_name)]
        zl = np.clip(sel[sel.on_list]["z"].values, lo, hi)
        zo = np.clip(sel[~sel.on_list]["z"].values, lo, hi)
        ids = sel[sel.on_list]["taxon"].values
        ax.scatter(rng.uniform(-0.22, 0.22, len(zo)), zo, s=9, color="0.62",
                   alpha=0.55, lw=0)
        ax.scatter(1.5 + rng.uniform(-0.16, 0.16, len(zl)), zl, s=26,
                   color=colour, edgecolor="k", lw=0.4, zorder=3)
        for xx, vals in ((0.0, zo), (1.5, zl)):
            if len(vals):
                ax.plot([xx - 0.3, xx + 0.3], [vals.mean()] * 2, color="k",
                        lw=2.2, zorder=4)
        ax.axhline(0, color="k", lw=0.7, ls=":")

        if len(zl):
            # Three, not four: at a third of page width a fourth label cannot
            # be placed without printing on its neighbour.
            top = np.argsort(-np.abs(zl))[:3]
            _gap = (hi - lo) * 0.24
            _margin = (hi - lo) * 0.04
            for idx, ly in zip(top, _spread(zl[top], _gap,
                                            lo + _margin, hi - _margin)):
                # Not a blanket italic: the placeholder epithet in
                # "Evtepia sp6345" is an identifier, not a Latin name, so
                # _italic_taxon_label leaves it upright.
                ax.annotate(_italic_taxon_label(_short(str(ids[idx]))),
                            xy=(1.68, zl[idx]),
                            xytext=(1.95, ly), fontsize=FS - 2,
                            va="center", ha="left",
                            annotation_clip=False,
                            arrowprops=dict(arrowstyle="-", lw=0.5,
                                            color="0.35"))
        p = float(row["p"])
        q = float(row["q"]) if np.isfinite(row.get("q", np.nan)) else np.nan
        ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        if np.isfinite(q):
            ptxt += f", q = {q:.3f}"
        ax.set_title(f"{config.display_text(cond)} · "
                     f"{_clade_label(clade, rank)}",
                     fontsize=FS, fontweight="bold", loc="left", pad=4)
        # Inside the panel: at a third of page width there is no room for a
        # line of statistics above it, and it ran into the next panel's title.
        ax.text(0.03, 0.97,
                f"{float(row['diff']):+.2f} SD\n{ptxt}",
                transform=ax.transAxes, fontsize=FS - 2, va="top",
                ha="left", color="0.25")
        ax.set_xlim(-0.85, 5.0)
        ax.set_ylim(lo, hi)
        ax.set_xticks([0, 1.5])
        # The clade is named in the title, so the tick carries "other" plus a
        # short form of the name: "rest of Lachnospiraceae" is wider than the
        # gap between the two groups and ran into the list's own label.
        _cl = clade if len(clade) <= 13 else clade[:12] + "\u2026"
        # The clade is a taxon name, italic at genus rank and upright at
        # family and above; the list name is never italic.
        ax.set_xticklabels([f"other\n{_clade_label(_cl, rank)}\n(n = {len(zo)})",
                            f"{config.display_text(list_name)}\n\n"
                            f"(n = {len(zl)})"], fontsize=FS - 2)
        ax.tick_params(axis="y", labelsize=FS - 2)
        if k % n_cols == 0:
            ax.set_ylabel("distance from background (SD)\n"
                          "higher = less depleted in cases", fontsize=FS - 2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for k in range(n, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")

    fig.suptitle("Within one genus or family, list members behave differently "
                 "from their relatives", fontsize=FS + 1, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.6, w_pad=1.2)

    supp = out_dir or os.path.join(os.path.dirname(s3_main), "supplement")
    os.makedirs(supp, exist_ok=True)
    stem = f"health_within_clade_contrasts_{ref}_{spec_tag}"
    emit_panels = {
        "points": points.reset_index(drop=True),
        "contrasts": within.reset_index(drop=True),
        "panels": pd.DataFrame(
            [dict(Condition=c, rank=r, clade=cl, list=l)
             for c, r, cl, l, _ in wanted]),
    }
    if clade_data.get("clades") is not None and not clade_data["clades"].empty:
        emit_panels["clades"] = clade_data["clades"].reset_index(drop=True)
    figure_source.emit(
        # S26 since the 2026-10-19 renumbering (was S30): this figure is
        # cited before the four taxon-driver panels, so it leads that run
        # of slots. Must match tools/additional_figures_manifest.txt.
        supp, "S26", emit_panels, stem=stem, reference=ref,
        # The conditions this figure actually drew, taken from `wanted`, which
        # is the panel list. It used to read `cond_order`, a name that does not
        # exist in this function -- the same copied idiom that was removed from
        # `_plot_prevalence_matched`, missed here. It raised NameError at the
        # emit, i.e. after the curve and clade tables were written but before
        # the section checkpointed, so a run died late and resumed from the
        # start of Section 3.
        extras={"drawn_conditions":
                list(dict.fromkeys(c for c, *_ in wanted))},
        notes=(spec_desc or "")
        + " | within-clade contrast: mean distance from the background curve, "
          "list members vs other members of the same clade, curve refitted "
          "with the clade excluded; p from permuting list labels inside the "
          "clade")
    out = os.path.join(supp, f"{stem}.png")
    savefig_dual(fig, out, dpi=SUBMISSION_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out}")


def _plot_background_curve(curve, bg_df, spec_tag, s3_main, prefix="health",
                           ref="combined", spec_desc="", focus_lists=None,
                           conditions=None, slot="S25", stem=None):
    """Figure S25: every list taxon against the prevalence background.

    Replaces the importance-weighting diagnostic, which diagnosed a weighting
    the pipeline no longer uses. The question is the same one S25 always asked
    -- is the background this list is being compared against really at the
    list's own prevalence? -- but it is now answerable by looking: each panel
    shows every background taxon, the fitted curve with its +/-1 SD band, and
    the list taxa on top of it.

    Rows: one per focus list. Columns: the conditions Figure 5E plots (at
    least config.MIN_STRATA_TO_PLOT studies).

    Called twice with different `focus_lists`: once by main-text Figure 5E for
    the H+ pair, and once here for the base VANISH/BloSSUM pair. The per-cell
    drawing is `plot_utils.draw_background_curve_cell` in both cases, so the
    two cannot drift apart.

    A third row counting background taxa per prevalence bin was carried for a
    while and dropped: the hollow markers already say which taxa have no
    comparable background, which is the only thing a reader needed the count
    for, and it cost a third of the figure's height to say it twice.
    """
    taxa_t = (curve or {}).get("taxa")
    grid_t = (curve or {}).get("grid")
    summ_t = (curve or {}).get("summary")
    frames = (curve or {}).get("frames") or {}
    if taxa_t is None or getattr(taxa_t, "empty", True):
        return
    focus_lists = focus_lists or ["VANISH", "BloSSUM"]
    focus_lists = [l for l in focus_lists
                   if l in set(taxa_t["Taxon_list"])] or \
                  sorted(set(taxa_t["Taxon_list"]))[:2]

    cond_order = [c for c in dict.fromkeys(taxa_t["Condition"]) if not _condition_dropped(c)]
    if conditions is not None:
        cond_order = [c for c in conditions if c in cond_order]
    elif summ_t is not None and not summ_t.empty and "n_strata" in summ_t.columns:
        k = summ_t.groupby("Condition")["n_strata"].max()
        cond_order = [c for c in cond_order
                      if k.get(c, 0) >= config.MIN_STRATA_TO_PLOT]
    cond_order = _display_order(cond_order)
    if not cond_order:
        return

    n_rows, n_cols = len(focus_lists), len(cond_order)
    # Height chosen so each gridspec cell is square at the shared spacing,
    # which is what set_box_aspect wants: otherwise the forced square box
    # shrinks inside a wider cell and the slack shows as horizontal
    # whitespace. page_figsize's fixed aspect gave exactly that.
    _LEG_IN = 0.46
    _fig_h, _ = background_curve_figsize(PAGE_WIDTH_IN, n_cols, n_rows,
                                         bottom_in=_LEG_IN)
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(PAGE_WIDTH_IN, _fig_h))
    floor = float(getattr(config, "HEALTH_BG_PREV_FLOOR", 1e-4))

    # Authored at the printed column width, so the magnification is 1 and the
    # printed sizes in plot_utils apply directly. Figure 5E passes its own.
    style = background_curve_style(1.0)
    summary = {}
    if summ_t is not None and not summ_t.empty:
        summary = {(str(r.Condition), str(r.Taxon_list)): r._asdict()
                   for r in summ_t.itertuples(index=False)}
    draw_background_curve_grid(
        axes, cond_order, focus_lists, frames, taxa_t,
        grid_t if grid_t is not None and not grid_t.empty else None,
        summary, config.CMH_LIST_COLORS,
        {l: config.display_text(l) for l in focus_lists},
        style=style, floor=floor, display=config.display_text,
        n_boot=int(getattr(config, "HEALTH_BG_N_BOOT", 2000)))

    handles = background_curve_legend(
        focus_lists, config.CMH_LIST_COLORS,
        {l: config.display_text(l) for l in focus_lists}, style=style)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=style["fs_legend"],
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Each list taxon against taxa of the same commonness "
                 "(study-stratified case $-$ control shift)", fontsize=FS)
    # Reserve the strip the tick labels, the axis label and the legend need,
    # in inches rather than as a fraction. savefig_dual's margin pass runs
    # after this function and resets whatever tight_layout chose, so the
    # reservation has to go through `_margin_overrides` to survive it -- and
    # it has to be absolute, because dropping the coverage row shortened the
    # canvas and the fraction that cleared three rows does not clear two.
    # No `top` here: the publication path strips the suptitle and raises the
    # top margin itself, and background_curve_figsize already sized the
    # canvas for that value through drawn_margins().
    fig._margin_overrides = {"left": 0.085, "right": 0.985,
                             "bottom": _LEG_IN / float(fig.get_figheight()),
                             "wspace": CELL_WSPACE, "hspace": CELL_HSPACE}
    # Opt out of the shared type scale. Every size in this figure is already
    # authored in printed points by plot_utils.background_curve_style, which
    # is what lets it match main-text Figure 5E exactly -- and Figure 5 goes
    # out through save_column_width, which does not run this pass. Letting it
    # run here is what made the two differ again: it rescaled the annotations
    # and dropped their bold.
    fig._skip_type_normalise = True
    fig.tight_layout()

    if spec_tag == "10pct" and prefix == "health":
        panels = {"list_taxa": taxa_t.reset_index(drop=True)}
        if grid_t is not None and not grid_t.empty:
            panels["curve"] = grid_t.reset_index(drop=True)
        if summ_t is not None and not summ_t.empty:
            panels["summary"] = summ_t.reset_index(drop=True)
        bg_rows = []
        for cond, frame in frames.items():
            if frame is None or frame.empty:
                continue
            f = frame.reset_index().rename(columns={"index": "taxon"})
            f["Condition"] = cond
            bg_rows.append(f)
        if bg_rows:
            panels["all_taxa"] = pd.concat(bg_rows, ignore_index=True)
        figure_source.emit(
            s3_main, slot, panels,
            stem=stem or f"{prefix}_background_curve_{spec_tag}",
            reference=ref,
            # What the figure drew, not what it was handed. The panels are a
            # superset -- they carry every condition the curve was fitted for
            # -- so without this a redraw cannot tell which contrasts the
            # published figure excluded.
            extras={"drawn_conditions": list(cond_order)},
            notes=(spec_desc or "")
            + " | background curve: fixed-bandwidth local-linear fit of the "
              "study-stratified case-control shift against log10 control "
              "prevalence; z floored at each taxon's own SE; bootstrap refits "
              "the curve")

    out = os.path.join(
        s3_main, (stem or f"{prefix}_background_curve_{spec_tag}") + ".png")
    savefig_dual(fig, out, dpi=SUBMISSION_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out}")


def _sig_pair_label(s_f, flag_f, s_u, flag_u):
    """One label for both tests: "Fisher/permutation".

    Two tests drawn as two bracketed rows need more vertical room than the
    panel has whenever both are significant, and no amount of nudging the
    labels fixes that -- a row is taller than the gap between rows. One
    bracket carrying both results removes the problem by construction. The
    caption explains the convention.

    Returns None when neither test is significant, so nothing is drawn.
    """
    def _one(sv, flag):
        if not sv or sv == "ns":
            return "\u2013"                     # en dash: tested, not significant
        return sv + ("" if flag > 0 else "~")

    if (not s_f or s_f == "ns") and (not s_u or s_u == "ns"):
        return None
    return f"{_one(s_f, flag_f)}/{_one(s_u, flag_u)}"


# One definition of each, in plot_utils, aliased here so the ~20 call sites
# below read unchanged. Two copies of the display rule is how they drift.
_display_order = display_order
_group_dividers = group_dividers


# ── Which contrasts a health figure may show ─────────────────────────────────
# One rule, read from the data, in place of the hand-written list this used to
# be. A study contributes a stratum only where BOTH arms carry at least
# config.CMH_MIN_STRATUM samples -- health_cmh enforces that when it builds
# the strata -- and a contrast is reported only where at least
# config.MIN_STRATA_TO_PLOT such strata exist. Hypertension (2 qualifying
# studies) and Metabolic Syndrome (1) fall out by that rule and are named
# nowhere. config.HEALTH_PLOT_SKIP is the explicit override alongside that
# rule, and is not empty: it holds the pooled "Unhealthy" contrast, which the
# count-of-studies rule cannot reject because it has plenty of studies. What
# disqualifies it is that its cases are the cases of the other four over
# again, so it is not an independent condition.
_ELIGIBLE_CONDITIONS = None   # (judged, admitted), set once the CMH table exists


def _note_eligible_conditions(cmh_df):
    """Record which contrasts clear the eligibility rule, for every figure."""
    global _ELIGIBLE_CONDITIONS
    admitted = _plotted_conditions(cmh_df)
    if admitted is None:
        return
    ccol = "Condition" if "Condition" in cmh_df.columns else "condition"
    judged = set(map(str, cmh_df[ccol]))
    _ELIGIBLE_CONDITIONS = (judged, set(admitted))
    dropped = sorted(judged - set(admitted))
    print(f"    [eligibility] {len(admitted)} contrast(s) clear "
          f">={config.CMH_MIN_STRATUM} samples per arm in "
          f">={config.MIN_STRATA_TO_PLOT} studies"
          + (f"; dropping {', '.join(dropped)}" if dropped else ""))


def _condition_dropped(cond):
    """True when `cond` must not be shown.

    Contrasts the rule never judged pass through untouched, which is what
    keeps this safe on the demographic path: the age, sex and BMI strata are
    absent from the CMH table and are governed by their own cohort sizes, not
    by a count of disease studies.
    """
    if cond in getattr(config, "HEALTH_PLOT_SKIP", set()):
        return True
    if _ELIGIBLE_CONDITIONS is None:
        return False
    judged, admitted = _ELIGIBLE_CONDITIONS
    return cond in judged and cond not in admitted


def _plotted_conditions(cmh_df):
    """The conditions the health figures show: Figure 5D-E's rule.

    Those resting on at least config.MIN_STRATA_TO_PLOT studies, counted from
    the CMH table's n_studies. HEALTH_PLOT_SKIP alone let MASLD (2 studies)
    and ASCVD (1) through to S25 and S26-S29 while Figure 5 dropped them.
    None when the table carries no counts, so callers keep what they had.
    """
    if cmh_df is None or getattr(cmh_df, "empty", True):
        return None
    if "n_studies" not in cmh_df.columns:
        return None
    ccol = "Condition" if "Condition" in cmh_df.columns else "condition"
    k = cmh_df.groupby(ccol)["n_studies"].max()
    return [c for c, v in k.items()
            if np.isfinite(v) and v >= config.MIN_STRATA_TO_PLOT]


def _plot_prevalence_matched(per_taxon_df, bg_df, cmh_df,
                              ref, s3_main, spec_tag, spec_desc):
    """
    Bidirectional bar chart: list vs its prevalence-matched background.
    Background fracs use _iw_fracs (unique taxa x continuous IW weights).
    Significance: F=IW-Fisher (ESS-adjusted), U=IW-directional permutation.
    """
    from scipy.stats import fisher_exact as _fisher
    import colorsys
    if per_taxon_df is None or per_taxon_df.empty: return
    if bg_df is None or bg_df.empty: return
    PAIRS = [("VANISH","VANISH_bg"),("BloSSUM","BloSSUM_bg"),
             ("VANISH+","VANISH+_bg"),("BloSSUM+","BloSSUM+_bg"),
             ("VanishH+","VanishH+_bg"),("BlossumH+","BlossumH+_bg")]
    _common_shift_frames(per_taxon_df, bg_df)
    _iw_rows: list = []
    combined = pd.concat([per_taxon_df, bg_df], ignore_index=True)
    cond_order = [c for c in dict.fromkeys(combined["Condition"]) if not _condition_dropped(c)]
    _shown = _plotted_conditions(cmh_df)
    if _shown is not None:
        cond_order = [c for c in cond_order if c in _shown]
    cond_order = _display_order(cond_order)
    avail = set(combined["Taxon_list"].unique())
    pairs = [(a,b) for a,b in PAIRS if a in avail and b in avail]
    if not pairs or not cond_order: return
    n_pairs, n_cond = len(pairs), len(cond_order)
    # The sizes passed below are starting points only: normalise_type_scale
    # resets every string by role at save time, so the knob that actually
    # governs this figure's type is `fig._type_boost`, set after the figure is
    # made.
    FS_PM = FS
    n_g2_by_cond = {}
    if cmh_df is not None and not cmh_df.empty:
        gmw_sub = cmh_df[cmh_df["Method"] == "Geom-mean-weighted CMH"]
        for _, row in gmw_sub.iterrows():
            cond = row["Condition"]
            if cond not in n_g2_by_cond:
                n_g2_by_cond[cond] = int(row.get("n_group2", 0))
    for cond in dict.fromkeys(combined["Condition"]):
        if cond not in n_g2_by_cond or n_g2_by_cond[cond] == 0:
            sub = combined[combined["Condition"] == cond]
            if "n_g2" in sub.columns and not sub["n_g2"].isna().all():
                n_g2_by_cond[cond] = int(sub["n_g2"].dropna().iloc[0])
    def _fracs_list(sub_df):
        sub = sub_df.pipe(_dropna_shift)
        if sub.empty:
            return dict(dep_s=0.,dep_t=0.,enr_s=0.,enr_t=0.,
                        d_lo=0.,d_hi=0.,e_lo=0.,e_hi=0.,
                        diffs=np.array([]),w=np.array([]),n=0,n_dep_sig=0,n_enr_sig=0,ess=0.)
        diffs=_shift_values(sub); pvals=sub["pval"].values; n=len(diffs)
        n_ds=int(np.sum((diffs>0)&(pvals<0.05))); n_es=int(np.sum((diffs<0)&(pvals<0.05)))
        dep_s=n_ds/n; enr_s=n_es/n
        dep_t=np.sum((diffs>0)&(pvals>=0.05))/n; enr_t=np.sum((diffs<0)&(pvals>=0.05))/n
        d_lo=d_hi=e_lo=e_hi=0.
        if n_ds>0:
            lo,hi=wilson_ci(n_ds,n); d_lo=max(0.,dep_s-lo); d_hi=max(0.,hi-dep_s)
        if n_es>0:
            lo,hi=wilson_ci(n_es,n); e_lo=max(0.,enr_s-lo); e_hi=max(0.,hi-enr_s)
        return dict(dep_s=dep_s,dep_t=dep_t,enr_s=enr_s,enr_t=enr_t,
                    d_lo=d_lo,d_hi=d_hi,e_lo=e_lo,e_hi=e_hi,
                    diffs=diffs,w=np.ones(n),n=n,n_dep_sig=n_ds,n_enr_sig=n_es,ess=float(n))
    def _star(p):
        if not np.isfinite(p): return ""
        return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    def _iw_fisher(fl, fb, dom):
        k_c = "n_dep_sig" if dom=="dep" else "n_enr_sig"
        k_f = "dep_s"     if dom=="dep" else "enr_s"
        a,na = fl[k_c], fl["n"]
        frac_b = fb[k_f]; ess_b = max(fb["ess"], 1.0)
        b = int(round(frac_b*ess_b)); nb = int(round(ess_b))
        if na==0 or nb==0: return np.nan,"",0
        _,p = _fisher([[a,max(na-a,0)],[b,max(nb-b,0)]])
        return p, _star(p), 1 if (a/na)>frac_b else -1
    # 2-column × 3-row grid (fills left-to-right, top-to-bottom)
    n_cols = 2
    n_rows = (n_pairs + n_cols - 1) // n_cols   # ceil division
    def _short_cond_label(cond):
        return {
            "Glucose Intolerance": "Glucose\nIntol.",
            "Fatty Liver": "MASLD",
            "CRC / Adenoma": "CRC /\nAdenoma",
        }.get(cond, cond)
    fig, axes = plt.subplots(n_rows, n_cols,
        # Squat, not square: these shifts cluster around zero and a square
        # panel spends most of its height on empty axis.
        figsize=page_figsize(n_cols=n_cols, n_rows=n_rows, aspect=1.9),
        squeeze=False)
    for pi, (list_name, bg_name) in enumerate(pairs):
        ri, ci_col = divmod(pi, n_cols)
        ax  = axes[ri][ci_col]
        col = LIST_COLORS.get(list_name, "#888")
        r_h=int(col.lstrip("#")[0:2],16)/255; r_s=int(col.lstrip("#")[2:4],16)/255; r_v=int(col.lstrip("#")[4:6],16)/255
        h,s,v = colorsys.rgb_to_hsv(r_h,r_s,r_v)
        bg_rgb = colorsys.hsv_to_rgb(h, s*0.35, min(v*1.25,1.0))
        bg_col = "#{:02x}{:02x}{:02x}".format(int(bg_rgb[0]*255),int(bg_rgb[1]*255),int(bg_rgb[2]*255))
        bw=0.36; x=np.arange(n_cond)
        for ci, cond in enumerate(cond_order):
            sub_l = combined[(combined["Condition"]==cond)&(combined["Taxon_list"]==list_name)]
            sub_b = combined[(combined["Condition"]==cond)&(combined["Taxon_list"]==bg_name)]
            if sub_l.empty or sub_b.empty: continue
            fl = _fracs_list(sub_l); fb = _iw_fracs(sub_b)
            dom = "dep" if (fl["dep_s"]+fl["dep_t"]) >= (fl["enr_s"]+fl["enr_t"]) else "enr"
            for offset, fn, color, hatch in [(-bw/2,fl,col,""),  (+bw/2,fb,bg_col,"///")]:
                xp = x[ci]+offset
                ax.bar(xp,-fn["dep_s"],bw*0.88,bottom=0,color=color,alpha=0.90,edgecolor="white",lw=0.4,hatch=hatch)
                ax.bar(xp,-fn["dep_t"],bw*0.88,bottom=-fn["dep_s"],color=color,alpha=0.28,edgecolor="white",lw=0.4,hatch=hatch)
                if fn["dep_s"]>0: ax.errorbar(xp,-fn["dep_s"],yerr=[[fn["d_hi"]],[fn["d_lo"]]],fmt="none",color="black",capsize=3.0,lw=1.0)
                ax.bar(xp,fn["enr_s"],bw*0.88,bottom=0,color=color,alpha=0.90,edgecolor="white",lw=0.4,hatch=hatch)
                ax.bar(xp,fn["enr_t"],bw*0.88,bottom=fn["enr_s"],color=color,alpha=0.28,edgecolor="white",lw=0.4,hatch=hatch)
                if fn["enr_s"]>0: ax.errorbar(xp,fn["enr_s"],yerr=[[fn["e_lo"]],[fn["e_hi"]]],fmt="none",color="black",capsize=3.0,lw=1.0)
            x_l,x_b,x_m = x[ci]-bw/2, x[ci]+bw/2, x[ci]
            p_f,s_f,flag_f = _iw_fisher(fl, fb, dom)
            p_u,s_u,flag_u = _iw_directional_test(fl, fb, dom)
            # Persisted, not only drawn. These two set the F and U stars on
            # this panel and appeared in no table, so the figure asserted a
            # significance a reader had no way to check.
            _iw_rows.append(dict(Condition=cond, Taxon_list=list_name,
                                 direction=dom, p_iw_fisher=float(p_f),
                                 p_iw_directional=float(p_u)))
            if dom=="dep":
                y_base = min(-(fl["dep_s"]+fl["dep_t"]),-(fb["dep_s"]+fb["dep_t"]))-0.022
                _lab = _sig_pair_label(s_f, flag_f, s_u, flag_u)
                if _lab:
                    y = y_base
                    ax.plot([x_l,x_l,x_b,x_b],[y-0.005,y,y,y-0.005],
                            color="black",lw=1.0)
                    ax.text(x_m, y+0.024, _lab, ha="center", va="bottom",
                            fontsize=FS_PM-3, fontweight="bold", color="black")
            else:
                y_base = max(fl["enr_s"]+fl["enr_t"],fb["enr_s"]+fb["enr_t"])+0.022
                _lab = _sig_pair_label(s_f, flag_f, s_u, flag_u)
                if _lab:
                    y = y_base
                    ax.plot([x_l,x_l,x_b,x_b],[y+0.005,y,y,y+0.005],
                            color="black",lw=1.0)
                    ax.text(x_m, y-0.024, _lab, ha="center", va="top",
                            fontsize=FS_PM-3, fontweight="bold", color="black")
        ax.axhline(0,color="black",lw=0.9); ax.axhline(0.05,color="#999",lw=0.5,ls=":",alpha=0.5); ax.axhline(-0.05,color="#999",lw=0.5,ls=":",alpha=0.5)
        for _b in _group_dividers(cond_order):
            ax.axvline(_b, color="lightgray", ls="--", lw=1, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{_short_cond_label(c)}\n(n={n_g2_by_cond.get(c,'?')})" for c in cond_order],rotation=0,ha="center",fontsize=FS_PM-2)
        if ri < n_rows - 1:
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_xticklabels([])
        # Symmetric about zero, since the axis reads as depleted below and
        # enriched above, but only far enough past the data to clear the
        # significance marks. At 1.75 the panels were three-quarters empty.
        ylo,yhi=ax.get_ylim(); yspan=max(abs(ylo),abs(yhi))
        ax.set_ylim(-(yspan*1.20), yspan*1.20)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{abs(v):.0%}"))
        if ci_col==0:
            # The dark/light reading is caption material, not axis label.
            ax.set_ylabel("Prevalence shift",fontsize=FS_PM-1)
        ax.grid(axis="y",alpha=0.15)
        # No per-panel key. Four of its six entries were the same on every
        # panel, and the fifth -- the taxon list -- is already the panel
        # title. Six keys in four different corners is what made this figure
        # read as cluttered; one key for the figure says the same thing once.
        ax.set_title(f"{list_name} vs {_bg_label(bg_df, short=True)}",
                     fontsize=FS_PM,fontweight="bold")
    # Hide unused axes if n_pairs is odd
    for _empty in range(n_pairs, n_rows * n_cols):
        _er, _ec = divmod(_empty, n_cols)
        axes[_er][_ec].set_visible(False)
    # Six panels of grouped bars at page width. At the default scale every
    # string hits the type ceiling (9.5 pt titles, 7.7 pt ticks) and the
    # labels crowd the bars they describe, which is the thing a reader is
    # meant to compare. 0.55 puts the titles at about 8 pt and the condition
    # labels at 6, which the panels have room for.
    fig._type_boost = 0.55
    fig.suptitle(f"Prevalence-matched shifts  |  {ref}  |  {spec_desc}",fontsize=FS_PM+1,fontweight="bold")

    # One key, below the grid, in the terms every panel shares. The taxon list
    # and its background take their colour from the panel they are in, so the
    # shared entries are drawn in neutral grey.
    from matplotlib.patches import Patch as _P
    from matplotlib.lines import Line2D as _L2
    _shared = [
        _P(facecolor="0.35", alpha=0.90, edgecolor="none",
           label="Taxon list (panel colour)"),
        _P(facecolor="0.35", alpha=0.90, hatch="///", edgecolor="none",
           label=_bg_label(bg_df, short=True)),
        _P(facecolor="grey", alpha=0.90, edgecolor="none", label="p<0.05"),
        _P(facecolor="grey", alpha=0.28, edgecolor="none", label="trend"),
        _L2([0], [0], color="black", lw=1.0, label="F: Fisher"),
        _L2([0], [0], color="black", lw=1.0, ls="--", label="U: permutation"),
    ]
    _figleg = fig.legend(handles=_shared, loc="lower center",
                         bbox_to_anchor=(0.5, -0.005), ncol=6, frameon=False,
                         handlelength=1.4, columnspacing=1.2,
                         handletextpad=0.4)
    _figleg._fixed_loc = True
    # Horizontal condition labels (five fit), and a band below them for the
    # key: slanted labels grew past a 3% band once the type pass sized them.
    plt.tight_layout(rect=(0, 0.045, 1, 1))
    # No figure_source.emit here any more. This figure held slot S25 until the
    # 2026-10-09 renumbering retired it: the naive directional comparison it
    # draws is superseded by the background curve, which is S25 now. Emitting a
    # data object under a slot the manifest no longer lists would publish a
    # figure that cannot be numbered, and emitting it under S25 would collide
    # with the curve. The PNG below is still written, as an internal diagnostic
    # and so the comparison can be shown if a referee asks for it.

    # The same two tests for the contrasts this figure does NOT draw.
    #
    # They were computed inside the panel loop above, so only the drawn
    # conditions ever reached health_iw_tests_*.csv and Table S12's two IW
    # columns came out blank for the rest -- 24 of its 54 cells. Those are the
    # cells a reader most wants to check the methods against, because they are
    # excluded for want of qualifying studies and not for want of signal, and
    # the table keeps them for exactly that reason (table_methods.py holds the
    # `eligible` column rather than filtering).
    #
    # Nothing here draws: no axis is touched and cond_order is not extended,
    # so the figure is byte-for-byte what it was. ~50 ms per cell.
    for cond in [c for c in dict.fromkeys(combined["Condition"])
                 if c not in cond_order]:
        for list_name, bg_name in pairs:
            sub_l = combined[(combined["Condition"] == cond) &
                             (combined["Taxon_list"] == list_name)]
            sub_b = combined[(combined["Condition"] == cond) &
                             (combined["Taxon_list"] == bg_name)]
            if sub_l.empty or sub_b.empty:
                continue
            fl, fb = _fracs_list(sub_l), _iw_fracs(sub_b)
            dom = ("dep" if (fl["dep_s"] + fl["dep_t"])
                   >= (fl["enr_s"] + fl["enr_t"]) else "enr")
            p_f, _, _ = _iw_fisher(fl, fb, dom)
            p_u, _, _ = _iw_directional_test(fl, fb, dom)
            _iw_rows.append(dict(Condition=cond, Taxon_list=list_name,
                                 direction=dom, p_iw_fisher=float(p_f),
                                 p_iw_directional=float(p_u)))

    if _iw_rows:
        _iwp = os.path.join(s3_main, f"health_iw_tests_{ref}_{spec_tag}.csv")
        pd.DataFrame(_iw_rows).to_csv(_iwp, index=False)
        _n_drawn = len(cond_order) * len(pairs)
        print(f"    Saved: {_iwp} ({len(_iw_rows)} cells: {_n_drawn} drawn, "
              f"{len(_iw_rows) - _n_drawn} table-only)")

    out = os.path.join(s3_main, f"health_prevmatched_{ref}_{spec_tag}.png")
    savefig_dual(fig, out, dpi=360, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out}")

def _plot_prevmatched_excess(per_taxon_df, bg_df, cmh_df,
                              ref, s3_main, spec_tag, spec_desc):
    """
    Excess directional shift of each list over its prevalence-matched background.
    Background fracs use _iw_fracs; significance uses _iw_directional_test.
    """
    if per_taxon_df is None or per_taxon_df.empty: return
    if bg_df is None or bg_df.empty: return
    PAIRS=[("VANISH","VANISH_bg"),("BloSSUM","BloSSUM_bg"),
           ("VANISH+","VANISH+_bg"),("BloSSUM+","BloSSUM+_bg"),
           ("VanishH+","VanishH+_bg"),("BlossumH+","BlossumH+_bg")]
    _common_shift_frames(per_taxon_df, bg_df)
    combined=pd.concat([per_taxon_df,bg_df],ignore_index=True)
    cond_order=[c for c in dict.fromkeys(combined["Condition"]) if not _condition_dropped(c)]
    cond_order = _display_order(cond_order)
    avail=set(combined["Taxon_list"].unique())
    pairs=[(a,b) for a,b in PAIRS if a in avail and b in avail]
    if not pairs or not cond_order: return
    n_lists,n_cond=len(pairs),len(cond_order)
    n_g2_by_cond={}
    if cmh_df is not None and not cmh_df.empty:
        gmw_sub=cmh_df[cmh_df["Method"]=="Geom-mean-weighted CMH"]
        for _,row in gmw_sub.iterrows():
            cond=row["Condition"]
            if cond not in n_g2_by_cond: n_g2_by_cond[cond]=int(row.get("n_group2",0))
    for cond in dict.fromkeys(combined["Condition"]):
        if cond not in n_g2_by_cond or n_g2_by_cond[cond]==0:
            sub=combined[combined["Condition"]==cond]
            if "n_g2" in sub.columns and not sub["n_g2"].isna().all():
                n_g2_by_cond[cond]=int(sub["n_g2"].dropna().iloc[0])
    def _fracs_list(sub_df):
        sub=sub_df.pipe(_dropna_shift)
        if sub.empty:
            return dict(dep_s=0.,dep_t=0.,enr_s=0.,enr_t=0.,
                        d_lo=0.,d_hi=0.,e_lo=0.,e_hi=0.,
                        diffs=np.array([]),w=np.array([]),n=0,n_dep_sig=0,n_enr_sig=0,ess=0.)
        diffs=_shift_values(sub); pvals=sub["pval"].values; n=len(diffs)
        n_ds=int(np.sum((diffs>0)&(pvals<0.05))); n_es=int(np.sum((diffs<0)&(pvals<0.05)))
        dep_s=n_ds/n; enr_s=n_es/n
        dep_t=np.sum((diffs>0)&(pvals>=0.05))/n; enr_t=np.sum((diffs<0)&(pvals>=0.05))/n
        d_lo=d_hi=e_lo=e_hi=0.
        if n_ds>0:
            lo,hi=wilson_ci(n_ds,n); d_lo=max(0.,dep_s-lo); d_hi=max(0.,hi-dep_s)
        if n_es>0:
            lo,hi=wilson_ci(n_es,n); e_lo=max(0.,enr_s-lo); e_hi=max(0.,hi-enr_s)
        return dict(dep_s=dep_s,dep_t=dep_t,enr_s=enr_s,enr_t=enr_t,
                    d_lo=d_lo,d_hi=d_hi,e_lo=e_lo,e_hi=e_hi,
                    diffs=diffs,w=np.ones(n),n=n,n_dep_sig=n_ds,n_enr_sig=n_es,ess=float(n))
    def _star(p):
        if not np.isfinite(p): return ""
        return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
    bw=0.78/max(n_lists,1); x_base=np.arange(n_cond)
    fig,ax=plt.subplots(figsize=page_figsize(height_in=4.6))
    for li,(list_name,bg_name) in enumerate(pairs):
        color=LIST_COLORS.get(list_name,"#888"); offset=(li-(n_lists-1)/2)*bw
        for ci,cond in enumerate(cond_order):
            sub_l=combined[(combined["Condition"]==cond)&(combined["Taxon_list"]==list_name)]
            sub_b=combined[(combined["Condition"]==cond)&(combined["Taxon_list"]==bg_name)]
            if sub_l.empty or sub_b.empty: continue
            fl=_fracs_list(sub_l); fb=_iw_fracs(sub_b)
            dom="dep" if (fl["dep_s"]+fl["dep_t"])>=(fl["enr_s"]+fl["enr_t"]) else "enr"
            exc_dep_s=fl["dep_s"]-fb["dep_s"]; exc_dep_t=fl["dep_t"]-fb["dep_t"]
            exc_enr_s=fl["enr_s"]-fb["enr_s"]; exc_enr_t=fl["enr_t"]-fb["enr_t"]
            x=x_base[ci]+offset
            ax.bar(x,-exc_dep_s,bw*0.88,bottom=0,color=color,alpha=0.92,edgecolor='white',lw=0.4)
            ax.bar(x,-exc_dep_t,bw*0.88,bottom=-exc_dep_s,color=color,alpha=0.28,edgecolor='white',lw=0.4)
            ax.bar(x,exc_enr_s,bw*0.88,bottom=0,color=color,alpha=0.92,edgecolor='white',lw=0.4)
            ax.bar(x,exc_enr_t,bw*0.88,bottom=exc_enr_s,color=color,alpha=0.28,edgecolor='white',lw=0.4)
            dep_ci=np.sqrt(fl["d_hi"]**2+fb["d_hi"]**2); enr_ci=np.sqrt(fl["e_hi"]**2+fb["e_hi"]**2)
            if abs(exc_dep_s)>0.002: ax.errorbar(x,-exc_dep_s,yerr=[[dep_ci],[0]],fmt='none',color='#444',capsize=2.0,lw=0.9)
            if abs(exc_enr_s)>0.002: ax.errorbar(x,exc_enr_s,yerr=[[0],[enr_ci]],fmt='none',color='#444',capsize=2.0,lw=0.9)
            _,star,_ = _iw_directional_test(fl, fb, dom)
            if star and star!="ns":
                if dom=="dep":
                    tip=-(exc_dep_s+max(exc_dep_t,0.))-dep_ci-0.006
                    ax.text(x,tip,star,ha='center',va='top',fontsize=FS-2,fontweight='bold',color=color)
                else:
                    tip=exc_enr_s+max(exc_enr_t,0.)+enr_ci+0.006
                    ax.text(x,tip,star,ha='center',va='bottom',fontsize=FS-2,fontweight='bold',color=color)
    ax.axhline(0,color='black',lw=1.0); ax.axhline(0.05,color='#ccc',lw=0.5,ls=':',alpha=0.8); ax.axhline(-0.05,color='#ccc',lw=0.5,ls=':',alpha=0.8)
    ax.text(n_cond-0.52,0.006,"background level",ha='right',va='bottom',fontsize=FS-2,color='#bbb',style='italic')
    for _b in _group_dividers(cond_order):
        ax.axvline(_b, color='lightgray', ls='--', lw=1, zorder=0)
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{c}\n(n={n_g2_by_cond.get(c,'?')})" for c in cond_order],rotation=30,ha='right',fontsize=FS-1)
    ymax=max(abs(ax.get_ylim()[0]),abs(ax.get_ylim()[1]))*1.40; ax.set_ylim(-ymax,ymax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{abs(v):.0%}"))
    ax.set_ylabel("Excess depletion (down) / Excess enrichment (up)\n"
                  "List fraction minus background fraction  |  dark=p<0.05, "
                  "light=trending\nerror bars=Wilson 95% CI  |  "
                  "*/  **/  ***=directional permutation",fontsize=FS-2)
    ax.grid(axis='y',alpha=0.12)
    leg=[Patch(facecolor=LIST_COLORS.get(ln,"#888"),alpha=0.92,edgecolor='none',label=ln) for ln,_ in pairs]
    leg+=[Patch(facecolor='grey',alpha=0.92,edgecolor='none',label='Sig. excess (p<0.05)'),
          Patch(facecolor='grey',alpha=0.28,edgecolor='none',label='Trending excess')]
    ax.legend(handles=leg,loc='upper left',fontsize=FS-2,ncol=4,framealpha=0.95)
    ax.set_title(f"Taxon-specific Health Association — Excess over the "
                 f"{_bg_label(bg_df)}\nGMW-CMH [{ref}]  |  Detection: "
                 f"{spec_desc}",fontsize=FS,fontweight='bold')
    plt.tight_layout()
    out=os.path.join(s3_main,f"health_prevmatched_excess_{ref}_{spec_tag}.png")
    savefig_dual(fig,out,dpi=360,bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out}")




# ====================================================================
#  NEW FIGURE: PREVALENCE-MATCHED DISTRIBUTION PLOTS (KDE)
# ====================================================================

def _compute_energy_stats(per_taxon_df, bg_df,
                          focus_lists=None, n_perm=10000, rng_seed=42):
    """
    Per-(condition, list) energy distance vs the weighted background.

    Bug fix: energy_distance(sl, sb) is size-dependent — comparing n_list ≈ 100
    against K×n_list ≈ 1000 replicated bg rows gives a systematically lower E than
    comparing 100 vs 100, so the null (which resamples n_list from bg) is always
    larger than obs_e, producing p=1 for every test.

    Fix: use a shared large IW-resampled reference (n_ref = 20×n_list) for both
    observed and null, exactly as in _iw_energy_test.  When no IW weights are
    available, fall back to a standard label-shuffle permutation.

    Returns dict {(cond, list_name): {E, p, n_list, n_bg}}.
    """
    from scipy.stats import energy_distance
    if per_taxon_df is None or per_taxon_df.empty: return {}
    if bg_df is None or bg_df.empty: return {}
    _common_shift_frames(per_taxon_df, bg_df)
    combined = pd.concat([per_taxon_df, bg_df], ignore_index=True)
    cond_order = [c for c in dict.fromkeys(combined["Condition"]) if not _condition_dropped(c)]
    cond_order = _display_order(cond_order)
    all_pairs  = [(ln, f"{ln}_bg") for ln in LIST_ORDER
                  if ln in combined["Taxon_list"].values
                  and f"{ln}_bg" in combined["Taxon_list"].values]
    if focus_lists is not None:
        all_pairs = [(a, b) for a, b in all_pairs if a in focus_lists]
    has_iw  = any(c in combined.columns for c in BG_WEIGHT_COLUMNS)
    results = {}
    rng     = np.random.default_rng(rng_seed)

    for list_name, bg_name in all_pairs:
        for cond in cond_order:
            sub_l = combined[(combined["Condition"]==cond) &
                             (combined["Taxon_list"]==list_name)].pipe(_dropna_shift)
            sub_b = combined[(combined["Condition"]==cond) &
                             (combined["Taxon_list"]==bg_name) ].pipe(_dropna_shift)
            if len(sub_l) < 3 or len(sub_b) < 3: continue

            sl     = _shift_values(sub_l)
            n_list = len(sl)

            if has_iw:
                # ── IW-aware one-sample test ──────────────────────────────
                # Use unique bg taxa + continuous IW weights, same as _iw_energy_test.
                if "taxon" in sub_b.columns:
                    bg_uniq = sub_b.groupby("taxon", sort=False).first().reset_index()
                else:
                    bg_uniq = sub_b.copy()

                sb_vals = _shift_values(bg_uniq)
                sb_w    = _bg_weight_values(bg_uniq)
                w_sum   = sb_w.sum()
                if w_sum < 1e-12 or len(sb_vals) < 3:
                    continue
                w_prob  = sb_w / w_sum

                # Large fixed IW-reference — same size for both obs and null
                n_ref   = max(n_list * 20, 2000)
                sb_ref  = sb_vals[rng.choice(len(sb_vals), size=n_ref,
                                             replace=True, p=w_prob)]
                obs_e   = energy_distance(sl, sb_ref)

                # Null: draw n_list from IW-bg, compare to same sb_ref
                null = np.array([
                    energy_distance(
                        sb_vals[rng.choice(len(sb_vals), size=n_list,
                                           replace=True, p=w_prob)],
                        sb_ref)
                    for _ in range(n_perm)])

                n_bg = len(sb_vals)

            else:
                # ── Standard label-shuffle permutation ────────────────────
                sb     = _shift_values(sub_b)
                n_bg   = len(sb)
                obs_e  = energy_distance(sl, sb)
                pooled = np.concatenate([sl, sb])
                null   = np.empty(n_perm)
                for i in range(n_perm):
                    rng.shuffle(pooled)
                    null[i] = energy_distance(pooled[:n_list], pooled[n_list:])

            p = max(float(np.mean(null >= obs_e)), 1.0 / (n_perm + 1))
            results[(cond, list_name)] = dict(E=obs_e, p=p,
                                              n_list=n_list, n_bg=n_bg)
    return results

def _hmp(p_values):
    """
    Harmonic Mean P-value (Wilson 2019).
    Valid under arbitrary dependence; conservative.
    p_hmp = k / sum(1/p_i)
    """
    ps = [p for p in p_values if np.isfinite(p) and p > 0]
    if not ps:
        return np.nan
    return len(ps) / sum(1.0 / p for p in ps)


def _combine_curve_specs(curve_summaries, s3_main, demographic=False):
    """Combine the primary mean-z statistic across detection specifications.

    The two detection rules (breadth percentage and flat base pairs) are
    applied to the same taxa in the same samples, so their bootstrap p-values
    are strongly positively dependent and Fisher's method would be
    anti-conservative. The harmonic mean p-value is valid under arbitrary
    dependence (Wilson 2019 PNAS 116:1195), which is why it is used here and
    for the energy-distance meta-statistic in `_plot_prevdist_meta`.

    Before this existed only the energy distance — now descriptive — was
    combined across specs, while the statistic the manuscript reports was
    left per spec with no stated rule for disagreement.

    Sign agreement is reported separately: the HMP says how strong the
    combined evidence is, not whether the two rules point the same way.

    Writes <prefix>_background_curve_meta.csv, one row per condition x list.
    """
    tags = sorted(curve_summaries)
    keep = ["Condition", "Taxon_list", "mean_z", "boot_p",
            "ci_lo", "ci_hi", "n_supported", "n_list"]
    frames = []
    for t in tags:
        f = curve_summaries[t]
        f = f[[c for c in keep if c in f.columns]].copy()
        f = f.rename(columns={c: f"{c}_{t}" for c in f.columns
                              if c not in ("Condition", "Taxon_list")})
        frames.append(f)
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["Condition", "Taxon_list"], how="outer")

    zc = [f"mean_z_{t}" for t in tags]
    pc = [f"boot_p_{t}" for t in tags]
    out["n_specs"] = out[pc].notna().sum(axis=1)
    out["mean_z_combined"] = out[zc].mean(axis=1, skipna=True)
    out["p_hmp"] = [_hmp(row) for row in out[pc].to_numpy()]
    out["n_sig_specs"] = (out[pc] < 0.05).sum(axis=1)
    _sgn = out[zc].apply(np.sign)
    out["sign_agree"] = (_sgn.nunique(axis=1, dropna=True) == 1)
    # The bootstrap floors each p at 1/n_boot, so the achievable star tiers
    # are bounded. Carried here so plot_utils.boot_star can drop the tiers
    # the resolution cannot support rather than printing an unreachable ***.
    out["n_boot"] = int(_cfg_bg("HEALTH_BG_N_BOOT", 2000))

    out = out.sort_values(["Taxon_list", "Condition"]).reset_index(drop=True)
    prefix = _out_prefix(demographic)
    path = os.path.join(s3_main, f"{prefix}_background_curve_meta.csv")
    out.to_csv(path, index=False)
    print(f"    Saved: {path}")
    print(f"      {len(out)} condition x list cells across specs "
          f"{', '.join(tags)}: {int((out['p_hmp'] < 0.05).sum())} with "
          f"HMP < 0.05, {int((~out['sign_agree']).sum())} disagreeing in sign")
    for r in out.itertuples(index=False):
        if r.Taxon_list in ("VanishH+", "BlossumH+"):
            print(f"      [META] {r.Condition} / {r.Taxon_list}: mean z "
                  f"{r.mean_z_combined:+.3f} HMP={r.p_hmp:.4f}"
                  f"{'' if r.sign_agree else '  (SIGN DISAGREEMENT)'}")
    return out


def _plot_prevdist_meta(prevdist_results, ref, s3_main,
                        focus_lists=None):
    """
    Meta-statistic bar chart combining energy-distance p-values across
    detection specs via the Harmonic Mean P-value (HMP).

    HMP = k / sum(1/p_i), valid under arbitrary dependence (Wilson 2019).
    Bar height = -log10(HMP).  Annotated with mean E and mean MAD ratio.
    Saved as: health_prevdist_meta_{ref}.png
    """
    if not prevdist_results:
        return

    spec_tags = sorted(prevdist_results.keys())
    all_keys  = set()
    for res in prevdist_results.values():
        all_keys.update(res.keys())

    if focus_lists is not None:
        all_keys = {(c, l) for c, l in all_keys if l in focus_lists}

    first_res  = prevdist_results[spec_tags[0]]
    cond_order = list(dict.fromkeys(c for c, _ in first_res.keys()
                                    if not _condition_dropped(c)))
    cond_order = _display_order(cond_order)
    list_names = [l for l in LIST_ORDER
                  if any(l == ln for _, ln in all_keys)]
    if focus_lists:
        list_names = [l for l in list_names if l in focus_lists]

    if not cond_order or not list_names:
        return

    n_cond  = len(cond_order)
    n_lists = len(list_names)

    def _star(p):
        if not np.isfinite(p): return ""
        return "***" if p < 0.001 else "**" if p < 0.01 \
               else "*" if p < 0.05 else "ns"

    fig, axes = plt.subplots(n_lists, 1,
                              figsize=page_figsize(
                                  height_in=min(8.6, n_lists * 1.9)),
                              squeeze=False)

    sig_thresh  = -np.log10(0.05)
    thresh_01   = -np.log10(0.01)
    thresh_001  = -np.log10(0.001)

    for li, list_name in enumerate(list_names):
        ax  = axes[li][0]
        col = LIST_COLORS.get(list_name, "#888")

        hmp_vals, e_means, mad_means = [], [], []
        for cond in cond_order:
            ps  = [prevdist_results[t].get((cond, list_name), {}).get("p",         np.nan) for t in spec_tags]
            es  = [prevdist_results[t].get((cond, list_name), {}).get("E",         np.nan) for t in spec_tags]
            mrs = [prevdist_results[t].get((cond, list_name), {}).get("mad_ratio", np.nan) for t in spec_tags]
            valid_ps = [p for p in ps if np.isfinite(p)]
            hmp_vals.append(_hmp(valid_ps) if valid_ps else np.nan)
            e_means.append(float(np.nanmean(es)))
            mad_means.append(float(np.nanmean(mrs)))

        x     = np.arange(n_cond)
        bar_h = [-np.log10(max(p, 1e-6)) if np.isfinite(p) else 0
                 for p in hmp_vals]

        ax.bar(x, bar_h, color=col, alpha=0.7, edgecolor="white", lw=0.5)
        ax.axhline(sig_thresh, color="black", lw=1.0, ls="--", alpha=0.5)
        ax.axhline(thresh_01,  color="gray",  lw=0.6, ls=":",  alpha=0.4)
        ax.axhline(thresh_001, color="gray",  lw=0.6, ls=":",  alpha=0.4)

        for xi, (p_hmp, E, mad_r) in enumerate(zip(hmp_vals, e_means, mad_means)):
            s = _star(p_hmp)
            if np.isfinite(E):
                label = "%s\nE=%.3f\nx%.1f" % (s, E, mad_r)
            else:
                label = s
            ax.text(xi, bar_h[xi] + 0.05, label,
                    ha="center", va="bottom", fontsize=FS-2,
                    fontweight="bold" if s not in ("", "ns") else "normal",
                    color="black" if s not in ("", "ns") else "gray")

        ylabel = "%s\n-log10(HMP)" % list_name
        ax.set_ylabel(ylabel, fontsize=FS, fontweight="bold", color=col)
        ax.set_xticks(x)
        if li == n_lists - 1:
            ax.set_xticklabels(cond_order, rotation=35, ha="right", fontsize=FS)
        else:
            ax.set_xticklabels([""] * n_cond)
        ax.set_yticks([0, sig_thresh, thresh_01, thresh_001])
        ax.set_yticklabels(["0", "p=0.05", "p=0.01", "p=0.001"], fontsize=FS-2)
        ax.grid(axis="y", alpha=0.15)
        ax.set_xlim(-0.5, n_cond - 0.5)

    fig.suptitle(
        "Meta-statistic: HMP across detection specs  -  %s\n"
        "Specs: %s  |  Bar = -log10(HMP p)  |  "
        "HMP valid under correlated tests  |  "
        "E = mean energy distance  |  MAD x = mean spread ratio"
        % (ref, ", ".join(spec_tags)),
        fontsize=FS, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(s3_main, "health_prevdist_meta_%s.png" % ref)
    savefig_dual(fig, out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("    Saved: %s" % out)


# _plot_prevmatched_distributions was removed on 2026-10-07. It drew the
# density form of panel E, which no longer exists: main-text Figure 5E and
# supplementary S26 are both the background curve, drawn by
# _plot_background_curve through plot_utils.draw_background_curve_cell.
# Keeping it would have left a third renderer for a panel that now has one.


def _plot_prevmatch_null_check(per_taxon_df, bg_df,
                                ref, s3_main, spec_tag, spec_desc,
                                focus_lists=None,
                                n_boot=2000, n_sample=100, rng_seed=42):
    """
    Null-distribution check: are the observed Vanish/BloSSUM list statistics
    above the noise floor expected from randomly choosing n_sample taxa from
    the IW-weighted background?

    For each (condition, list) pair:
      1. Draw n_boot IW-weighted random samples of n_sample background taxa.
      2. For each sample compute:
           a. Energy distance: E(sample_prev_diff, IW-bg_reference)
              where IW-bg_reference is a large (20 × n_sample) IW-resample
              of the full background — the same reference used in _iw_energy_test
              for the list, so the two statistics are directly comparable.
           b. Depletion fraction: fraction of sample taxa with
              shift > 0 AND pval < 0.05 (depleted in disease, p<0.05).
      3. Record where the actual list falls in each null distribution.
      4. Compute excess = list_stat - median(null) and empirical p-value.

    Layout: (n_pairs × 2) rows × n_cond columns
      Odd  rows: energy distance null (grey density) + list value (coloured line)
      Even rows: depletion/enrichment fraction null + list value

    Saved: health_prevmatch_null_{ref}_{spec_tag}.png
    """
    from scipy.stats import energy_distance, gaussian_kde as _gkde_nc

    if per_taxon_df is None or per_taxon_df.empty: return
    if bg_df is None or bg_df.empty: return

    import colorsys

    def _bg_color(hex_col, desat=0.40, bright=1.20):
        r, g, b = [int(hex_col[i:i+2], 16)/255 for i in (1, 3, 5)]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        nr, ng_, nb = colorsys.hsv_to_rgb(h, s*desat, min(v*bright, 1.0))
        return "#{:02x}{:02x}{:02x}".format(int(nr*255), int(ng_*255), int(nb*255))

    PAIRS = [("VANISH",    "VANISH_bg"),
             ("BloSSUM",   "BloSSUM_bg"),
             ("VANISH+",   "VANISH+_bg"),
             ("BloSSUM+",  "BloSSUM+_bg"),
             ("VanishH+",  "VanishH+_bg"),
             ("BlossumH+", "BlossumH+_bg")]

    _common_shift_frames(per_taxon_df, bg_df)
    combined   = pd.concat([per_taxon_df, bg_df], ignore_index=True)
    cond_order = [c for c in dict.fromkeys(combined["Condition"]) if not _condition_dropped(c)]
    cond_order = _display_order(cond_order)
    avail      = set(combined["Taxon_list"].unique())
    pairs      = [(a, b) for a, b in PAIRS if a in avail and b in avail]
    if focus_lists is not None:
        pairs = [(a, b) for a, b in pairs if a in focus_lists]
    if not pairs or not cond_order:
        return

    n_pairs  = len(pairs)
    n_cond   = len(cond_order)
    FS       = 10
    rng      = np.random.default_rng(rng_seed)

    # 2 metric-rows per list pair
    fig, axes = plt.subplots(
        n_pairs * 2, n_cond,
        figsize=page_figsize(n_cols=n_cond, n_rows=n_pairs * 2),
        squeeze=False,
    )

    for pi, (list_name, bg_name) in enumerate(pairs):
        col   = LIST_COLORS.get(list_name, "#888")
        bgcol = _bg_color(col)
        r_e   = pi * 2       # energy distance row
        r_f   = pi * 2 + 1   # fraction row

        for ci, cond in enumerate(cond_order):
            ax_e = axes[r_e][ci]
            ax_f = axes[r_f][ci]

            # ── gather data ───────────────────────────────────────────
            sub_l = combined[(combined["Condition"]  == cond) &
                              (combined["Taxon_list"] == list_name)
                              ].pipe(_dropna_shift)
            sub_b = combined[(combined["Condition"]  == cond) &
                              (combined["Taxon_list"] == bg_name)
                              ].pipe(_dropna_shift, extra=("pval",))

            if len(sub_l) < 3 or sub_b.empty:
                for ax in (ax_e, ax_f):
                    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                            ha="center", va="center", fontsize=FS, color="gray")
                    ax.set_xticks([]); ax.set_yticks([])
                if ci == 0:
                    ax_e.set_ylabel(f"{list_name}\nEnergy dist", fontsize=FS,
                                    fontweight="bold", color=col)
                    ax_f.set_ylabel(f"{list_name}\nDep frac", fontsize=FS,
                                    fontweight="bold", color=col)
                if pi == 0:
                    ax_e.set_title(cond, fontsize=FS-1, rotation=20, ha="left", pad=4)
                continue

            # Unique background taxa with IW weights
            if "taxon" in sub_b.columns:
                bg_uniq = sub_b.groupby("taxon", sort=False).first().reset_index()
            else:
                bg_uniq = sub_b.copy()

            sb_vals  = _shift_values(bg_uniq)
            sb_pvals = bg_uniq["pval"].values
            sb_w     = _bg_weight_values(bg_uniq)

            w_sum  = sb_w.sum()
            if w_sum < 1e-12:
                for ax in (ax_e, ax_f):
                    ax.text(0.5, 0.5, "w=0", transform=ax.transAxes,
                            ha="center", va="center", fontsize=FS, color="gray")
                continue
            w_prob = sb_w / w_sum

            # Effective sample size — determines whether n_sample makes sense
            ess = (w_sum ** 2) / (sb_w ** 2).sum()

            sl       = _shift_values(sub_l)
            sl_pvals = sub_l["pval"].values
            n_list   = len(sl)
            n_samp   = min(n_sample, n_list)   # match list size if it's < n_sample

            # ── IW-bg reference (stable large resample for E-distance) ──
            n_ref  = max(n_samp * 20, 2000)
            sb_ref = sb_vals[rng.choice(len(sb_vals), size=n_ref,
                                        replace=True, p=w_prob)]

            # ── Observed list stats ─────────────────────────────────────
            obs_E    = energy_distance(sl, sb_ref)
            obs_dep  = float(np.mean((sl > 0) & (sl_pvals < 0.05)))
            obs_enr  = float(np.mean((sl < 0) & (sl_pvals < 0.05)))
            # dominant direction
            dom      = "dep" if obs_dep >= obs_enr else "enr"
            obs_frac = obs_dep if dom == "dep" else obs_enr

            # ── Null distribution: n_boot IW-bg samples of n_samp taxa ─
            null_E    = np.empty(n_boot)
            null_frac = np.empty(n_boot)

            for bi in range(n_boot):
                idx   = rng.choice(len(sb_vals), size=n_samp,
                                   replace=True, p=w_prob)
                samp  = sb_vals[idx]
                sp    = sb_pvals[idx]
                null_E[bi]    = energy_distance(samp, sb_ref)
                null_frac[bi] = (float(np.mean((samp > 0) & (sp < 0.05)))
                                 if dom == "dep" else
                                 float(np.mean((samp < 0) & (sp < 0.05))))

            # ── Empirical p-values ──────────────────────────────────────
            p_E    = max(float(np.mean(null_E    >= obs_E)),    1.0 / (n_boot + 1))
            p_frac = max(float(np.mean(null_frac >= obs_frac)), 1.0 / (n_boot + 1))

            def _star(p):
                if not np.isfinite(p): return ""
                return "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"

            exc_E    = obs_E    - float(np.median(null_E))
            exc_frac = obs_frac - float(np.median(null_frac))

            # ── Panel: energy distance ──────────────────────────────────
            x_e  = np.linspace(max(0, null_E.min() * 0.8),
                               max(null_E.max(), obs_E) * 1.15, 200)
            try:
                kde_e = _gkde_nc(null_E)(x_e)
            except Exception:
                kde_e = np.zeros_like(x_e)

            ax_e.fill_between(x_e, kde_e, alpha=0.35, color="#999999")
            ax_e.plot(x_e, kde_e, color="#666666", lw=1.2)
            ax_e.axvline(obs_E, color=col, lw=2.0, zorder=5,
                         label=f"{list_name} obs")
            ax_e.axvline(np.median(null_E), color="#999999",
                         lw=1.2, ls="--", zorder=4, label="null median")
            # percentile rank of observed
            pct_rank = float(np.mean(null_E <= obs_E)) * 100
            sig_E    = _star(p_E)
            ax_e.text(0.97, 0.97,
                      f"{sig_E}  p={p_E:.3f}\nobs={obs_E:.3f}\npct={pct_rank:.0f}th\nexc={exc_E:+.3f}",
                      transform=ax_e.transAxes, ha="right", va="top",
                      fontsize=FS-2,
                      fontweight="bold" if sig_E not in ("", "ns") else "normal",
                      color=col if sig_E not in ("", "ns") else "#555555")
            ax_e.text(0.03, 0.97,
                      f"n_list={n_list}\nn_samp={n_samp}\nESS={ess:.0f}",
                      transform=ax_e.transAxes, ha="left", va="top",
                      fontsize=FS-2, color="#666666")
            ax_e.set_xlim(left=0)
            ax_e.tick_params(labelsize=FS-3)
            ax_e.grid(axis="x", alpha=0.12)
            if ci == 0:
                ax_e.set_ylabel(f"{list_name}\nEnergy dist vs IW-bg",
                                fontsize=FS, fontweight="bold", color=col)
            if pi == 0:
                ax_e.set_title(cond, fontsize=FS-1, rotation=20, ha="left", pad=4)
            if ci == 0 and pi == 0:
                from matplotlib.patches import Patch as _P
                from matplotlib.lines import Line2D as _L
                ax_e.legend(handles=[
                    _L([0],[0], color=col,      lw=2.0, label=f"{list_name}"),
                    _L([0],[0], color="#999999", lw=1.2, ls="--", label="Null median"),
                    _P(facecolor="#999999", alpha=0.35, edgecolor="none",
                       label=f"Null ({n_boot} IW-samples of {n_samp})"),
                ], fontsize=FS-2, loc="upper left", framealpha=0.85)

            # ── Panel: depletion/enrichment fraction ────────────────────
            x_f  = np.linspace(max(0, null_frac.min() - 0.02),
                               max(null_frac.max(), obs_frac) * 1.15 + 0.01, 200)
            try:
                kde_f = _gkde_nc(null_frac)(x_f) if null_frac.std() > 1e-6 else None
            except Exception:
                kde_f = None

            if kde_f is not None:
                ax_f.fill_between(x_f, kde_f, alpha=0.35, color="#999999")
                ax_f.plot(x_f, kde_f, color="#666666", lw=1.2)
            else:
                # All null fracs identical — show a spike
                ax_f.axvline(null_frac[0], color="#999999", lw=2, alpha=0.5)

            ax_f.axvline(obs_frac, color=col, lw=2.0, zorder=5)
            ax_f.axvline(np.median(null_frac), color="#999999",
                         lw=1.2, ls="--", zorder=4)

            pct_f  = float(np.mean(null_frac <= obs_frac)) * 100
            sig_f  = _star(p_frac)
            dir_lbl = "dep" if dom == "dep" else "enr"
            ax_f.text(0.97, 0.97,
                      f"{sig_f}  p={p_frac:.3f}\nobs={obs_frac:.2%}\npct={pct_f:.0f}th\nexc={exc_frac:+.2%}",
                      transform=ax_f.transAxes, ha="right", va="top",
                      fontsize=FS-2,
                      fontweight="bold" if sig_f not in ("", "ns") else "normal",
                      color=col if sig_f not in ("", "ns") else "#555555")
            ax_f.set_xlim(left=0)
            ax_f.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0%}"))
            ax_f.tick_params(labelsize=FS-3)
            ax_f.grid(axis="x", alpha=0.12)
            if ci == 0:
                ax_f.set_ylabel(f"{list_name}\n{dir_lbl} frac (p<0.05)",
                                fontsize=FS, fontweight="bold", color=col)
            if pi == n_pairs - 1:
                ax_e.set_xlabel("Energy distance", fontsize=FS-1)
                ax_f.set_xlabel(f"Fraction {dir_lbl} (p<0.05)", fontsize=FS-1)

    fig.suptitle(
        f"Null-distribution check: list statistics vs IW-weighted random subsets  —  "
        f"{ref}  |  {spec_desc}\n"
        f"Grey density = null distribution ({n_boot} IW-weighted random samples "
        f"of {n_sample} bg taxa)  |  "
        "Coloured line = observed list statistic\n"
        "Top rows: energy distance vs IW-bg reference  |  "
        "Bottom rows: fraction depleted/enriched (p<0.05)\n"
        "p = empirical fraction of null ≥ observed  |  "
        "pct = percentile rank of observed in null  |  "
        "exc = observed minus null median",
        fontsize=FS, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(s3_main, f"health_prevmatch_null_{ref}_{spec_tag}.png")
    savefig_dual(fig, out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out}")



# ====================================================================
#  TAXON DRIVER IDENTIFICATION  (Pass 1: IW-rank consistency; Pass 2: driver score)
# ====================================================================

def _load_taxonomy_map_s3():
    """
    Load SGB -> GTDB taxonomy string for section 3.

    The combined metadata workbook comes first: it is the catalogue-wide
    annotation and the only source that covers taxa outside the VANISH and
    BloSSUM lists, which is most of the driver table. The phylo files fill
    whatever it does not carry. This is the same precedence
    `section2_pooling._build_taxonomy_map` uses, so the two sections no longer
    disagree about what a genome is called.

    Returns dict {sgb_id: gtdb_string}.
    """
    tax = {}
    try:
        from .section2_pooling import _taxonomy_from_combined_metadata
        for gid, lineage in _taxonomy_from_combined_metadata().items():
            if lineage:
                tax.setdefault(gid, lineage)
    except Exception:                      # workbook absent or unreadable
        pass

    for fpath in [getattr(config, "VANISH_PHYLO_FILE", ""),
                  getattr(config, "BLOSSUM_PHYLO_FILE", "")]:
        if not fpath or not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath, sep=None, engine="python")
            tcols = [c for c in df.columns
                     if any(k in c.lower() for k in ("gtdb", "tax", "lineage"))]
            if df.shape[1] >= 2 and tcols:
                for _, row in df.iterrows():
                    gid = str(row.iloc[0]).strip()
                    tax.setdefault(gid, str(row[tcols[0]]).strip())
        except Exception:
            pass
    return tax


#: GTDB rank prefix -> the word a figure label uses for it.
_RANK_WORD = {"d": "domain", "p": "phylum", "c": "class", "o": "order",
              "f": "family", "g": "genus", "s": "species"}


#: Characters mathtext reads as markup rather than as themselves. GTDB names
#: carry underscores -- Phascolarctobacterium_A, Prevotella copri_M -- which
#: mathtext would otherwise take as subscripts.
_MATHTEXT_ESCAPE = {"\\": r"\backslash ", "_": r"\_", "^": r"\^",
                    "{": r"\{", "}": r"\}", "$": r"\$", "#": r"\#",
                    "%": r"\%", "&": r"\&", "~": r"\sim "}

#: Markers a figure label may end with that are not part of the name, and so
#: stay upright: the genus marker " sp.", a placeholder epithet such as
#: " sp934700615" or the shortened " sp6345", a parenthesised rank, the
#: accession `_labels` appends to break a tie, and the truncation ellipsis.
#:
#: A placeholder is an alphanumeric identifier standing in for an epithet that
#: GTDB has not assigned, not a Latin epithet, so it is set upright like the
#: "sp." it abbreviates. The digits are required: a real epithet beginning
#: "sp" -- sporogenes, sphaericus -- is left alone and stays italic.
_LABEL_TRAILER = re.compile(
    r"(\s*\[[^\]]*\]|\s*\([^)]*\)|\s+sp\.|\s+sp\d+|\.\.\.|…)$")


def _italic_taxon_label(label):
    """Italicise the taxon name in a figure label, leaving markers upright.

    Scientific names are set in italic, and a figure that prints them upright
    while its caption prints them italic is inconsistent on its own page.
    What is italicised is the name only: `Ventricola sp.` becomes an italic
    `Ventricola` followed by an upright `sp.`, because "sp." is an abbreviation
    standing in for an epithet rather than part of one. The same goes for a
    parenthesised rank, the disambiguating accession and the ellipsis left by
    truncation.

    A placeholder epithet -- `sp934700615` -- is upright too, and this
    paragraph used to say the opposite. `_LABEL_TRAILER` has matched
    `\\s+sp\\d+` for some time and the comment above it gives the reason: a
    placeholder is an alphanumeric catalogue identifier standing in for an
    epithet GTDB has not assigned, so italicising it would present a genome
    accession as a species name. Only this docstring still described the old
    behaviour.

    Returns a mathtext string. Everything mathtext would read as markup is
    escaped first, and a label that would come out empty is returned unchanged.
    """
    raw = str(label)
    body, trailer = raw, ""
    while True:
        m = _LABEL_TRAILER.search(body)
        if not m:
            break
        trailer = m.group(0) + trailer
        body = body[:m.start()]
    body = body.strip()
    if not body:
        return raw
    esc = "".join(_MATHTEXT_ESCAPE.get(ch, ch) for ch in body).replace(" ", r"\ ")
    return rf"$\mathit{{{esc}}}$" + trailer


def _clade_label(clade, rank):
    """A clade name, italicised only where its rank takes italics.

    Genus and species are set in italic; family and every rank above it are
    not. `_italic_taxon_label` italicises whatever it is handed, which is
    right for the taxon labels drawn inside a panel -- those are always a
    species or a genus -- but wrong for the clade a panel contrasts on, which
    `config.HEALTH_BG_CLADE_PANELS` names at either genus or family rank.
    Four of the six published panels contrast a family, so a blanket italic
    set `Oscillospiraceae` and `Lachnospiraceae` in italic in both the panel
    title and the x-tick label.
    """
    return _italic_taxon_label(clade) if rank == "genus" else str(clade)


def _gtdb_short(tax_str, n_ranks=2):
    """Return last n_ranks non-empty GTDB ranks as a compact label."""
    if not tax_str or str(tax_str) in ("", "nan"):
        return ""
    parts  = [p.strip() for p in str(tax_str).split(";")]
    filled = [p for p in parts if p and not p.endswith("__")]
    return "; ".join(filled[-n_ranks:]) if filled else ""


def _iw_rank(x, sb_vals, sb_w):
    """
    IW-weighted disease-enrichment rank of scalar x relative to background.

    Canonical sign: prev_diff = control - disease.
    Enriched in disease  <=>  prev_diff < 0  <=>  below background.

    r = P_IW(bg_prev_diff > x)   [fraction of bg MORE depleted than x]
      = 1 - P_IW(bg_prev_diff < x)

    dev = r - 0.5:
        dev > 0  =>  taxon MORE ENRICHED in disease than IW-bg median
        dev < 0  =>  taxon MORE DEPLETED in disease than IW-bg median

    Positive driver score = enriched in disease.  Matches the distributions
    plot convention where positive x-axis = enriched in disease.
    """
    w_sum = sb_w.sum()
    if w_sum < 1e-12:
        return 0.5
    # P_IW(bg > x): fraction of IW-bg that is MORE depleted than x
    return float(np.sum(sb_w[sb_vals > x]) / w_sum)


def _run_taxon_drivers(per_taxon_df, bg_df, focus_lists=None, conditions=None,
                       curve=None):
    """
    Identify which SGBs drive the divergence between a list and its background.

    The fundamental problem with LOO energy distance is that removing 1 taxon
    from n~100 changes E by O(1/n) — about 1% of the total effect, which is
    indistinguishable from noise.  The z-score approach is weak because it
    divides by the full background SD, which is dominated by the spread of
    thousands of background taxa rather than by how extreme the list taxon is.

    CORRECTED 2026-10-06. When a background curve is available (`curve`), the
    per-condition statistic is the taxon's standardised residual from it:

        z_ic = (shift_ic - mu_c(p_ic)) / sigma_c(p_ic)

    in SD units of the spread among taxa of the same commonness, with sigma
    floored at the taxon's own SE. Positive = less depleted / more enriched in
    cases than comparable taxa. Taxa with no comparable taxa in the catalogue
    (`supported` false, i.e. above ~85% prevalence) are not scored at all
    rather than scored against an extrapolation.

    Without a curve -- an old data object, or HEALTH_BG_METHOD = "iw" -- it
    falls back to the weighted percentile rank deviation it used to use:

        r_ic  =  P_IW(bg_shift_c  >  shift_ic),     d_ic = r_ic - 0.5

    which lies in [-0.5, +0.5]. Both metrics share the same sign convention
    (positive = more enriched in disease than the background expects), so the
    columns and the figures are unchanged; `driver_metric` records which one
    produced them.

    Pass 1 — Consistency across conditions
        For each taxon, compute d_ic for every condition where it appears.
        Test H0: mean(d_ic) = 0 via one-sample t-test (or sign test when n < 5).
        Effect size = mean_rank_deviation  (bounded ±0.5, null = 0).
        Also report: fraction of conditions with d_ic > 0 (consistency), and
        an FDR-adjusted q-value across all taxa.

    Pass 2 — Driver score
        driver_score_i  =  mean(d_ic)  across conditions
        Positive = consistently more depleted in disease than background expects.
        Negative = consistently more enriched in disease than background expects.
        Magnitude = how far the taxon sits from the background center on average.

        This is O(1) per taxon (no LOO recomputation), no dilution from large n_list,
        and scales linearly with how extreme the taxon is rather than with 1/n_list.

    Both passes use the same per-condition rank data; Pass 2 is just the
    aggregation, so they are computed together.

    Returns
    -------
    driver_df : DataFrame, one row per (Taxon_list, taxon), columns:
        Taxon_list, taxon, n_conditions,
        mean_rank_dev,      # mean(r_ic - 0.5), bounded ±0.5
        driver_score,       # alias for mean_rank_dev (clearer name for Pass 2)
        direction,          # "above" (depleted) or "below" (enriched)
        frac_above,         # fraction of conditions with d_ic > 0
        consistency,        # max(frac_above, 1-frac_above)
        ttest_p,            # one-sample t-test p (H0: mean_rank_dev=0)
        sign_p,             # sign test p (H0: P(d>0)=0.5)
        combined_p,         # min(ttest_p, sign_p) x 2 (Bonferroni over 2 tests)
        rank_devs,          # per-condition statistics, semicolon-joined
        conditions,         # condition labels, semicolon-joined
        driver_metric,      # "z" (curve) or "rank_dev" (weighted percentile)
    """
    from scipy.stats import ttest_1samp

    if per_taxon_df is None or per_taxon_df.empty:
        return pd.DataFrame()
    if bg_df is None or bg_df.empty:
        return pd.DataFrame()

    PAIRS = [("VANISH",    "VANISH_bg"),
             ("BloSSUM",   "BloSSUM_bg"),
             ("VANISH+",   "VANISH+_bg"),
             ("BloSSUM+",  "BloSSUM+_bg"),
             ("VanishH+",  "VanishH+_bg"),
             ("BlossumH+", "BlossumH+_bg")]

    _common_shift_frames(per_taxon_df, bg_df)
    combined   = pd.concat([per_taxon_df, bg_df], ignore_index=True)
    cond_order = [c for c in dict.fromkeys(combined["Condition"]) if not _condition_dropped(c)]
    # Scored over the conditions the figure shows, so each taxon's score,
    # consistency and q-value describe the columns beside them.
    if conditions is not None:
        cond_order = [c for c in cond_order if c in set(conditions)]
    elif "n_strata" in combined.columns:
        # No list from the caller: apply Figure 5's own rule rather than
        # showing every condition. MASLD (2 studies) and ASCVD (1) are not in
        # Figure 5, S25 or S27, and a driver column for them invites a
        # comparison the rest of the section does not support.
        _k = combined.groupby("Condition")["n_strata"].max()
        _dropped = [c for c in cond_order
                    if _k.get(c, 0) < config.MIN_STRATA_TO_PLOT]
        if _dropped:
            print(f"    [taxon_drivers] dropping condition(s) below "
                  f"{config.MIN_STRATA_TO_PLOT} studies: "
                  + ", ".join(f"{c} ({int(_k.get(c, 0))})" for c in _dropped))
        cond_order = [c for c in cond_order
                      if _k.get(c, 0) >= config.MIN_STRATA_TO_PLOT]
    cond_order = _display_order(cond_order)
    avail      = set(combined["Taxon_list"].unique())
    pairs      = [(a, b) for a, b in PAIRS if a in avail and b in avail]
    if focus_lists is not None:
        pairs = [(a, b) for a, b in pairs if a in focus_lists]
    if not pairs:
        return pd.DataFrame()

    # (Condition, Taxon_list, taxon) -> z, from the background curve
    curve_z = {}
    curve_taxa = (curve or {}).get("taxa")
    if curve_taxa is not None and not getattr(curve_taxa, "empty", True):
        _ct = curve_taxa
        if "supported" in _ct.columns:
            _ct = _ct[_ct["supported"].astype(bool)]
        curve_z = {(r.Condition, r.Taxon_list, r.taxon): float(r.z)
                   for r in _ct.itertuples(index=False)
                   if np.isfinite(getattr(r, "z", np.nan))}
    metric = "z" if curve_z else "rank_dev"

    all_rows = []

    for list_name, bg_name in pairs:
        # taxon -> {cond: per-condition statistic}
        taxon_rdev = {}

        for cond in cond_order:
            sub_l = combined[(combined["Condition"]  == cond) &
                              (combined["Taxon_list"] == list_name)
                              ].pipe(_dropna_shift)
            sub_b = combined[(combined["Condition"]  == cond) &
                              (combined["Taxon_list"] == bg_name)
                              ].pipe(_dropna_shift)

            if len(sub_l) < 3 or sub_b.empty:
                continue

            if "taxon" in sub_b.columns:
                bg_uniq = sub_b.groupby("taxon", sort=False).first().reset_index()
            else:
                bg_uniq = sub_b.copy()

            sb_vals = _shift_values(bg_uniq)
            sb_w    = (_bg_weight_values(bg_uniq))

            if sb_w.sum() < 1e-12:
                continue

            # Per-taxon statistic: the curve residual when we have one,
            # otherwise the weighted percentile rank deviation.
            for _, lrow in sub_l.iterrows():
                t    = str(lrow.get("taxon", ""))
                if not t:
                    continue
                # A taxon absent from BOTH arms says nothing about the
                # contrast: its shift is exactly zero by construction and its
                # z is whatever the curve reads at prevalence 0 -- the same
                # number for every such taxon. In the 9 Oct run nine
                # never-detected VANISH taxa shared z = -0.4752 to four
                # decimals in the unhealthy contrast, and being perfectly
                # consistent across all five conditions they came out as the
                # nine most significant "drivers" of the list. The support
                # rule does not catch them, because at prevalence 0 there are
                # plenty of equally absent taxa nearby to fit a curve on.
                #
                # This guard is the driver analysis's own and predates the
                # list-level one. Since 2026-10-09 `background_z` drops the
                # same taxa whenever config.HEALTH_BG_DROP_NEVER_DETECTED is
                # true, which is the default, so the two now agree and the
                # driver figures decompose the list mean they are drawn
                # beside. The guard stays because it also covers the replot
                # path, where the curve comes from a stored data object that
                # may predate the config flag.
                _p1, _p2 = lrow.get("prev_g1"), lrow.get("prev_g2")
                try:
                    if float(_p1) <= 0.0 and float(_p2) <= 0.0:
                        continue
                except (TypeError, ValueError):
                    pass
                if curve_z:
                    dev = curve_z.get((cond, list_name, t))
                    if dev is None:
                        continue
                else:
                    pd_v = float(lrow[_shift_column(sub_l) or "prev_diff"])
                    if not np.isfinite(pd_v):
                        continue
                    dev = _iw_rank(pd_v, sb_vals, sb_w) - 0.5
                if t not in taxon_rdev:
                    taxon_rdev[t] = {}
                taxon_rdev[t][cond] = dev

        # Aggregate across conditions per taxon
        for t, cond_dict in taxon_rdev.items():
            devs    = np.array([v for v in cond_dict.values() if np.isfinite(v)])
            n_conds = len(devs)
            if n_conds == 0:
                continue

            mean_dev   = float(np.mean(devs))
            frac_above = float(np.mean(devs > 0))
            consistency = max(frac_above, 1.0 - frac_above)
            direction  = "enriched" if mean_dev >= 0 else "depleted"

            # Significance: one-sample t-test of per-condition rank deviations
            # against zero.  This is directly reflected by whether the 95% CI
            # of the mean crosses zero — no sign test needed (it can flag taxa
            # whose CI clearly crosses zero if direction is consistent).
            if n_conds >= 3:
                _, tt_p = ttest_1samp(devs, popmean=0.0)
                combined_p = float(tt_p)
            else:
                combined_p = np.nan

            all_rows.append({
                "Taxon_list":    list_name,
                "taxon":         t,
                "n_conditions":  n_conds,
                "mean_rank_dev": mean_dev,
                "driver_score":  mean_dev,   # alias
                "direction":     direction,
                "frac_above":    frac_above,
                "consistency":   consistency,
                "ttest_p":       combined_p,
                "combined_p":    combined_p,
                "conditions":    ";".join(cond_dict.keys()),
                "rank_devs":     ";".join(f"{v:.4f}" for v in cond_dict.values()),
                "driver_metric": metric,
            })

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # FDR correction per list, through health_background._bh. The copy that
    # used to live here scattered the ranked values back to their original
    # positions BEFORE taking the cumulative minimum, so the monotonicity step
    # ran over input order instead of sorted-p order: the result was a running
    # minimum over an arbitrary sequence, not a BH-adjusted p-value. One
    # implementation now, and it is the one the background method already uses.
    for list_name in df["Taxon_list"].unique():
        mask = df["Taxon_list"] == list_name
        ps   = df.loc[mask, "combined_p"].fillna(1.0).values
        df.loc[mask, "q_value"] = hbg._bh(ps)

    return df


# Which taxon list is which Supplementary Figure. Four figures come out of one
# function, differing only in the list they draw, so one split covers them all.
#: Empty on purpose. S26-S29 are drawn by health_driver_family_panels, which
#: owns those slots and emits their data objects; the per-list t-test figures
#: below would otherwise overwrite them on the next full run with panels built
#: from a statistic these figures no longer use. The mapping is kept rather
#: than deleted so the old figures can be brought back by restoring it.
_TAXON_DRIVER_SLOTS = {}

#: What the mapping was, before S26-S29 moved to the family panels. These
#: are pre-2026-10-19 slot numbers and describe the past, not the manifest.
_TAXON_DRIVER_SLOTS_LEGACY = {
    "VanishH+":  "S26",
    "BlossumH+": "S27",
    "VANISH":    "S28",
    "BloSSUM":   "S29",
}


def _run_driver_family_panels(data, taxon_lists, curve, spec, s3_main,
                              tax_map, conditions=None, ref="combined",
                              spec_desc=""):
    """Compute the cluster-bootstrap driver statistic and draw S26-S29.

    Returns {slot: path}, or {} when the inputs for it are not present.
    """
    from . import health_driver_bootstrap as hdb
    from . import health_driver_family_panels as hdp

    list_taxa = curve.get("taxa")
    grid = curve.get("grid")
    if list_taxa is None or getattr(list_taxa, "empty", True):
        return {}
    if grid is None or getattr(grid, "empty", True):
        return {}

    spec_tag = spec.short_label
    strat_csv = os.path.join(s3_main,
                             f"health_per_taxon_stratified_{spec_tag}.csv")
    if not os.path.exists(strat_csv):
        print(f"    [taxon_drivers] no {os.path.basename(strat_csv)}; "
              "family panels need it for the per-taxon standard errors")
        return {}
    stratified = pd.read_csv(strat_csv)

    # The conditions the figures show: the same rule Figure 5D-E applies, so
    # the panels cannot drift from the columns beside them.
    if conditions is None:
        summ = curve.get("summary")
        if summ is None or getattr(summ, "empty", True):
            return {}
        k = summ.groupby("Condition")["n_strata"].max()
        conditions = [c for c in dict.fromkeys(summ["Condition"])
                      if not _condition_dropped(c)
                      and k.get(c, 0) >= config.MIN_STRATA_TO_PLOT]
    conditions = _display_order(list(conditions))
    if not conditions:
        return {}
    print(f"    [taxon_drivers] bootstrap over cohorts, conditions: "
          + ", ".join(conditions))

    # section_mod defaults to this module, imported at call time.
    ps = hdb.per_stratum_shifts(data, spec, conditions)
    if not [k for k in ps if k != "taxa"]:
        return {}
    lists = {k: list(v) for k, v in taxon_lists.items() if len(v)}
    df = hdb.driver_bootstrap(ps, grid, stratified, list_taxa, lists,
                              conditions)
    if df is None or df.empty:
        return {}

    out_csv = os.path.join(
        s3_main, f"health_taxon_drivers_bootstrap_{spec_tag}.csv")
    df.to_csv(out_csv, index=False)
    print(f"    Saved: {out_csv}")

    # emit() puts the object in <s3_main>/figure_data, as the other slots do.
    return hdp.emit_and_draw(df, tax_map, s3_main, conditions, ref=ref,
                             spec_tag=spec_tag, spec_desc=spec_desc,
                             figure_data_dir=s3_main)


def _driver_metric_style(driver_df):
    """How to label and scale the driver statistic.

    "z"        distance from the background curve, in SDs of the spread among
               taxa of the same commonness (the corrected method)
    "rank_dev" weighted percentile rank deviation, bounded +/-0.5 (the old one)
    """
    metric = "rank_dev"
    if driver_df is not None and "driver_metric" in getattr(driver_df, "columns", ()):
        vals = [v for v in driver_df["driver_metric"].dropna().unique()]
        if vals:
            metric = str(vals[0])
    if metric == "z":
        # `lim` is the heatmap colour scale and stays at +/-3: saturation there
        # is deliberate and the key says so. `pad_cap` is the lollipop x-limit,
        # and at 3.2 it truncated the panel silently. The star is drawn just
        # past the end of the interval, so any row whose interval reached the
        # cap lost its star altogether: 8 of 29 rows on S27 (BlossumH+ needs
        # 7.46), 2 of 16 on S26, 2 of 17 on S29, and Mediterraneibacter gnavus
        # was drawn outside the axes entirely. Those are the rows the captions
        # name as the strongest drivers. The width of those intervals is also
        # the point -- gnavus is +3.38 with a 95% CI of (+0.27, +6.49), which a
        # clipped panel renders as a tight, overwhelming effect.
        #
        # With the cap raised, the panel width is genuinely data-driven, so a
        # fixed tick ladder no longer fits every list. `score_ticks` stays as
        # the fallback; `_plot_taxon_drivers` derives the z panel's ticks from
        # the drawn range instead, which keeps the count between three and five
        # whether the panel runs to 1.5 SD or to 7.5.
        return dict(metric="z", lim=3.0, pad_cap=8.0, pad_floor=1.0,
                    ticks=[-3, -1.5, 0, 1.5, 3],
                    tick_labels=["\u22123", "\u22121.5", "0", "+1.5", "+3"],
                    score_ticks=([-2.0, -1.0, 0.0, 1.0, 2.0],
                                 ["\u22122", "\u22121", "0", "+1", "+2"]),
                    key_title="Distance from background (SD)",
                    axis_label="z vs background",
                    guides=(0.5, 2.0))
    return dict(metric="rank_dev", lim=0.5, pad_cap=0.52, pad_floor=0.20,
                ticks=[-0.5, -0.25, 0.0, 0.25, 0.5],
                tick_labels=["\u22120.5", "\u22120.25", "0", "+0.25", "+0.5"],
                score_ticks=([-0.4, 0.0, 0.4], ["-0.4", "0", "+0.4"]),
                key_title="Rank deviation",
                axis_label="Deviation",
                guides=(0.10, 0.25))


def _plot_taxon_drivers(driver_df, taxonomy_map,
                         ref, s3_main, spec_tag, spec_desc,
                         focus_lists=None, top_n=30,
                         min_conditions=3, min_consistency=0.70,
                         p_thresh=0.05):
    """
    Combined Pass-1 + Pass-2 figure on one statistic per taxon x condition:
    its distance from the background curve in SDs (corrected method), or the
    weighted percentile rank deviation when that is what the data carries.
    `_driver_metric_style` sets the scale and the labels from the data.

    Each list gets one figure with three panels:

    Left — Per-condition statistic heatmap (taxa x conditions)
        Cell colour = the taxon's z against the background curve, or the
        rank deviation in [-0.5, +0.5] when that is the metric.
        Blue  = ABOVE the background: less depleted in cases than taxa of
                the same baseline prevalence (dev > 0). This is COL_ENRICH
                and it is what the four captions describe.
        Orange/vermillion = BELOW the background: more depleted in cases
                than comparable taxa (dev < 0).
        White = indistinguishable from the background expectation.
        Grey  = condition not scored for this taxon.

        Rows filtered to n_conditions >= min_conditions AND
        consistency >= min_consistency, then restricted to the taxa
        individually significant at the uncorrected p < p_thresh, taken
        most-significant-first from each tail so neither direction can be
        hidden. Rows are drawn in driver_score order, enriched at the top.
        The count is whatever clears p_thresh, bounded below by a 10-row
        floor and above by top_n; both bounds are printed when they bite.

    Centre — Driver score lollipop  (=Pass 2)
        x-axis = mean rank deviation (bounded ±0.5).
        Error bar = 95% CI from per-condition rank deviations.
        Colour = orange (above bg, depleted) or blue (below bg, enriched).
        Star annotation from combined_p / q_value.
        Dashed grey line at 0 (null: indistinguishable from background).
        Reference lines at ±0.1 (>60% of bg), ±0.25 (>75%), ±0.4 (>90%).

    Right — Diagnostic scatter: driver_score vs consistency
        x = consistency (max(frac_above, 1-frac_above)), range [0.5, 1.0]
        y = driver_score (mean rank deviation), range [-0.5, +0.5]
        Colour intensity = -log10(q_value), so BH-corrected significance
        is visible even though selection is on the uncorrected p.
        Top-right quadrant: consistently enriched in disease (enriched drivers).
        Bottom-right quadrant: consistently depleted in disease (depleted drivers).

    Saved: health_taxon_drivers_{list}_{ref}_{spec_tag}.png
    """
    import matplotlib.colors as mcolors
    import matplotlib.cm as mcm

    if driver_df is None or driver_df.empty:
        return

    _mstyle = _driver_metric_style(driver_df)

    PAIRS = [("VANISH", ""), ("BloSSUM", ""), ("VANISH+", ""),
             ("BloSSUM+", ""), ("VanishH+", ""), ("BlossumH+", "")]
    avail = set(driver_df["Taxon_list"].unique())
    lists = [a for a, _ in PAIRS if a in avail]
    if focus_lists:
        lists = [l for l in lists if l in focus_lists]
    if not lists:
        return

    def _label(t):
        """Deepest named rank for a taxon, never a bare accession.

        GTDB leaves a rank blank rather than guessing, so the deepest filled
        rank is the most specific thing that can honestly be said: the species
        where there is one, otherwise the genus, otherwise the family. Ranks
        above species are marked, so `Ventricola sp.` is not mistaken for a
        species name nor `UBA932 (family)` for a genus.

        A placeholder epithet ("Cryptobacteroides sp934700615") is kept rather
        than collapsed to the genus: it is GTDB's own name for that genome and
        it is unique, whereas three Prevotella placeholders all shortened to
        "Prevotella sp." are three identical rows. `_labels` handles the
        collisions that remain.
        """
        tx     = str(taxonomy_map.get(t, "") or "")
        parts  = [p.strip() for p in tx.split(";")]
        filled = [p for p in parts if p and not p.endswith("__")]
        if not filled:
            return str(t)
        rank, _, name = filled[-1].partition("__")
        name = (name or filled[-1]).strip()
        if not name:
            return str(t)
        if rank == "g":
            name = name + " sp."
        elif rank and rank != "s":
            name = f"{name} ({_RANK_WORD.get(rank, rank)})"
        if len(name) > 30:
            name = name[:27] + "..."
        return name

    def _labels(taxa):
        """`_label` over a sequence, with ties broken so no two rows match.

        Two SGBs in the same genus with no species rank both read "Genus sp.",
        and after the 30-character truncation two long names can converge too.
        A reader cannot tell those rows apart, so the ones that collide get
        their accession back, abbreviated.
        """
        taxa = list(taxa)
        labs = [_label(t) for t in taxa]
        dupes = {l for l in labs if labs.count(l) > 1}
        if not dupes:
            return labs
        out = []
        for t, l in zip(taxa, labs):
            if l in dupes:
                tail = str(t).split("_")[-1].split(".")[0][-6:]
                l = (l[:23] + "..." if len(l) > 23 else l) + f" [{tail}]"
            out.append(l)
        return out

    def _short_cond_label(cond):
        return {
            "Glucose Intolerance": "Glucose\nIntol.",
            "Fatty Liver": "MASLD",
            "CRC / Adenoma": "CRC/\nAdenoma",
        }.get(cond, cond)

    def _star(p):
        if not np.isfinite(p): return ""
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    def _expand_rdev(sub):
        rows = {}
        for _, r in sub.iterrows():
            t     = r["taxon"]
            conds = r["conditions"].split(";") if isinstance(r.get("conditions"), str) else []
            devs  = ([float(x) for x in r["rank_devs"].split(";")]
                     if isinstance(r.get("rank_devs"), str) else [])
            rows[t] = dict(zip(conds, devs))
        return pd.DataFrame(rows).T if rows else pd.DataFrame()

    FS = 7
    # Colormap: orange/red for above bg, blue for below
    # OI colours: vermillion (#D55E00) for depleted, OI blue (#0072B2) for enriched
    cmap_heat = mcolors.LinearSegmentedColormap.from_list(
        "rdev",
        [config.OI["vermillion"], "#f7f7f7", config.OI["blue"]], N=256)
    # Orange for above, blue for below (lollipop / scatter)
    # OI palette: blue for enriched in disease, vermillion for depleted
    COL_ENRICH = config.OI["blue"]        # #0072B2 — enriched (dev > 0)
    COL_DEPLET = config.OI["vermillion"]  # #D55E00 — depleted (dev < 0)
    COL_ABOVE  = COL_ENRICH
    COL_BELOW  = COL_DEPLET

    for list_name in lists:
        sub = driver_df[driver_df["Taxon_list"] == list_name].copy()
        # Filter
        sub = sub[(sub["consistency"]  >= min_consistency) &
                  (sub["n_conditions"] >= min_conditions)].copy()
        if sub.empty:
            print(f"    [drivers] {list_name}: no taxa pass filters "
                  f"(consistency>={min_consistency:.0%}, "
                  f"n_cond>={min_conditions})")
            continue

        # Two-tailed, and the data decides how many rows there are.
        #
        # This used to be `sort_values("driver_score").head(top_n)` with
        # top_n=18: the top N by effect size, which is one-tailed by
        # construction. For every list in the 9 Oct run the top 18 were all
        # positive, and for the original VANISH list 60 of the 85 qualifying
        # taxa were negative and none was drawn -- the panel showed the
        # positive tail of a predominantly negative distribution while its
        # caption described the negative one.
        #
        # 18 was also arbitrary: nothing about the data stopped there, and it
        # showed under a quarter of the taxa the list qualifies. The rule now
        # is to plot every taxon that passes the per-taxon test, taking the
        # most significant first within each tail.
        #
        # The threshold is the UNCORRECTED p, which is what the centre panel
        # already stars and what the confidence interval already agrees with:
        # every drawn row therefore carries a star and an interval clear of
        # zero, and the selection states exactly what the figure shows. BH q
        # is still computed and still scales the right-hand panel, so a reader
        # can see which of the drawn taxa also survive correction -- but it is
        # not the gate. It cannot be: the per-taxon test has at most five
        # conditions, and they are not independent, since the pooled Unhealthy
        # contrast contains the cases of the other four. Correcting across ~70
        # taxa on that basis certifies almost nothing (3, 0, 1 and 0 taxa at
        # q < 0.05), which says more about the power of a five-point t-test
        # than about the taxa. These panels are interpretive follow-up to the
        # list-level bootstrap, not primary tests, and Supplementary Text 6F
        # says so.
        #
        # `top_n` survives only as a layout ceiling for the page, and `_FLOOR`
        # keeps a list with almost nothing significant from collapsing to a
        # two-row stub. Both are reported rather than applied silently, so a
        # caption can say what was left out.
        _FLOOR = 10
        _rank = next((c for c in ("combined_p", "ttest_p", "q_value")
                      if c in sub.columns), None)
        if _rank is None:                   # no per-taxon test (an old replot)
            sub_plot = sub.sort_values("driver_score", ascending=False).head(top_n)
            _n_sig, _capped = len(sub_plot), 0
        else:
            _sig    = sub[sub[_rank] < p_thresh]
            _n_sig  = len(_sig)
            _want   = min(max(_n_sig, min(_FLOOR, len(sub))), top_n, len(sub))
            _capped = max(0, _n_sig - _want)
            # Draw from the significant set, so every row is a taxon the
            # figure can make a claim about; only a list with fewer than
            # _FLOOR significant taxa reaches past it, and then by
            # significance order so the extra rows are the next best evidence.
            _pool = _sig if _n_sig >= _want else sub
            # Most significant on each side, half the rows each; a pool short
            # on one side lets the other fill the space rather than leaving it
            # blank, so no row is wasted.
            _half = _want // 2
            _pos  = _pool[_pool["driver_score"] > 0].nsmallest(_half, _rank)
            _neg  = _pool[_pool["driver_score"] < 0].nsmallest(_want - _half, _rank)
            _take = pd.concat([_pos, _neg])
            if len(_take) < _want:          # one side was short
                _fill = _pool.drop(index=_take.index).nsmallest(
                    _want - len(_take), _rank)
                _take = pd.concat([_take, _fill])
            sub_plot = _take
        # Enriched on top, depleted at the bottom, so the heatmap reads
        # continuously down the effect size whatever the selection was.
        sub_plot = sub_plot.sort_values("driver_score", ascending=False).copy()
        n_tax = len(sub_plot)
        _np   = int((sub_plot["driver_score"] > 0).sum())
        _msg  = (f"    [drivers] {list_name}: {len(sub)} pass filters "
                 f"({int((sub['driver_score'] > 0).sum())} above background, "
                 f"{int((sub['driver_score'] < 0).sum())} below); "
                 f"{_n_sig} significant at p<{p_thresh:g}; showing {n_tax} "
                 f"({_np} above, {n_tax - _np} below)")
        if _capped:
            _msg += f" -- {_capped} significant taxa omitted by the {top_n}-row cap"
        elif n_tax > _n_sig:
            _msg += f" -- {n_tax - _n_sig} added to reach the {_FLOOR}-row floor"
        print(_msg)

        # All conditions in the data (for heatmap columns)
        rdev_mat  = _expand_rdev(sub_plot)
        # The order the other health figures use (the order the conditions
        # were tested in), not alphabetical.
        _seen = list(dict.fromkeys(
            c for _s in sub_plot.get("conditions", pd.Series(dtype=str)).dropna()
            for c in str(_s).split(";") if c))
        all_conds = ([c for c in _seen if c in rdev_mat.columns]
                     + sorted(c for c in rdev_mat.columns if c not in _seen)
                     if not rdev_mat.empty else [])
        # The same display order the other health figures use: pooled
        # contrast last. This variable is `all_conds` rather than
        # `cond_order`, which is how it was missed when the rest were moved
        # -- the heatmap kept Unhealthy in column 1 while Figure 5, S25 and
        # S26 had it in column 5.
        all_conds = _display_order(all_conds)

        fig, axes = plt.subplots(
            1, 3,
            figsize=page_figsize(height_in=max(2.8, n_tax * DRIVER_ROW_IN + 1.3)),
            # Narrower cells. Each square carries one rank, while the row
            # label carries a species name and the two panels beside it carry
            # the score and its consistency -- all of which were being
            # squeezed so the squares could be wide.
            gridspec_kw={"width_ratios": [max(len(all_conds) * 0.34, 2.2), 1.20, 1.05]},
        )
        fig.subplots_adjust(left=0.40, right=0.99, top=0.92, bottom=0.16,
                            wspace=0.06)
        fig._skip_savefig_tight_layout = True
        fig._margin_overrides = dict(left=0.40, right=0.99, top=0.92,
                                     bottom=0.16, wspace=0.06)
        # The panels give up a little width so the species names, the axis
        # labels and the two summary panels can be read at a larger size.
        fig._type_boost = 1.38
        ax_heat, ax_loll, ax_sc = axes

        taxa_ordered = list(sub_plot["taxon"])

        # ── Heatmap ───────────────────────────────────────────────────────
        if all_conds:
            rdev_arr = np.array([[float(rdev_mat.loc[t, c])
                                  if (t in rdev_mat.index and
                                      c in rdev_mat.columns and
                                      not pd.isna(rdev_mat.loc[t, c]))
                                  else np.nan
                                  for c in all_conds]
                                 for t in taxa_ordered])

            norm = mcolors.TwoSlopeNorm(vmin=-_mstyle["lim"], vcenter=0.0,
                                        vmax=_mstyle["lim"])
            im   = ax_heat.imshow(rdev_arr, aspect="auto",
                                  cmap=cmap_heat, norm=norm,
                                  interpolation="nearest")
            ax_heat.set_xticks(range(len(all_conds)))
            ax_heat.set_xticklabels([_short_cond_label(c) for c in all_conds], rotation=35, ha="right",
                                    fontsize=FS-1)
            ax_heat.set_yticks(range(n_tax))
            ax_heat.set_yticklabels(
                [_italic_taxon_label(l) for l in _labels(taxa_ordered)],
                fontsize=FS-2)
            # A continuous key, carrying the heatmap's own cmap and norm.
            # Three swatches labelled "dep. / 0 / enr." implied three bins for
            # what is a continuous scale, and the abbreviations cost a reader
            # more than the two words they saved.
            key_ax = ax_heat.inset_axes([0.17, 1.045, 0.36, 0.026])
            key_ax.imshow(
                np.linspace(norm.vmin, norm.vmax, 256).reshape(1, -1),
                aspect="auto", cmap=cmap_heat, norm=norm,
                interpolation="bilinear")
            # The scale is the statistic itself: 0 is what taxa of the same
            # commonness do, and the ends are the display limit.
            _kt = _mstyle["ticks"]
            key_ax.set_xticks([np.interp(v, [norm.vmin, norm.vmax], [-0.5, 255.5])
                               for v in _kt])
            key_ax.set_xticklabels(_mstyle["tick_labels"], fontsize=FS-3)
            key_ax.tick_params(axis="x", length=2, width=0.4, pad=1)
            # The key's own title: the heatmap's sat above the tick labels.
            key_ax.set_title(_mstyle["key_title"], fontsize=FS, pad=3)
            key_ax.set_yticks([])
            for _sp in key_ax.spines.values():
                _sp.set_linewidth(0.4)
                _sp.set_color("#777777")
            for _x, _ha, _txt in ((-0.04, "right", "depleted"),
                                  (1.04, "left", "enriched")):
                key_ax.text(_x, 0.5, _txt, transform=key_ax.transAxes,
                            clip_on=False, va="center", ha=_ha,
                            fontsize=FS-2, color="#333333")

            for ri, t in enumerate(taxa_ordered):
                for ci, c in enumerate(all_conds):
                    val = rdev_mat.loc[t, c] if (t in rdev_mat.index and
                                                  c in rdev_mat.columns) else np.nan
                    if pd.isna(val):
                        ax_heat.add_patch(plt.Rectangle(
                            (ci-0.5, ri-0.5), 1, 1,
                            fill=True, color="#dddddd", zorder=0))
        else:
            ax_heat.text(0.5, 0.5, "No per-condition data",
                         transform=ax_heat.transAxes, ha="center", va="center")

        ax_heat.set_ylabel("Species", fontsize=FS)


        # ── Lollipop (driver score + CI) ──────────────────────────────────
        y_pos      = np.arange(n_tax)
        ds_arr     = sub_plot["driver_score"].values
        combined_p = sub_plot["combined_p"].fillna(1.0).values
        q_arr      = sub_plot.get("q_value", pd.Series(1.0, index=sub_plot.index)
                                 ).fillna(1.0).values

        # 95% CI using t_critical(df=n-1)×SE — matches t-test exactly.
        # Guarantees: CI crosses zero ⟺ t-test not significant (p > 0.05).
        from scipy.stats import t as _t_dist
        ci_arr = np.full(n_tax, np.nan)
        for yi, t in enumerate(taxa_ordered):
            devs_str = sub_plot[sub_plot["taxon"] == t]["rank_devs"].values
            if len(devs_str) and isinstance(devs_str[0], str):
                devs = np.array([float(x) for x in devs_str[0].split(";")])
                n_d  = len(devs)
                if n_d >= 2:
                    ci_arr[yi] = (_t_dist.ppf(0.975, df=n_d - 1)
                                  * np.std(devs, ddof=1) / np.sqrt(n_d))

        raw_p_arr = sub_plot["combined_p"].fillna(1.0).values

        for yi, (ds, qv, raw_p) in enumerate(zip(ds_arr, q_arr, raw_p_arr)):
            c_bar = COL_ENRICH if ds >= 0 else COL_DEPLET
            # Opacity/alpha driven by raw p so it tracks the CI visually
            sig   = np.isfinite(raw_p) and raw_p < 0.05
            alpha = 0.88 if sig else 0.35
            # Points only — no connector to 0-axis
            ax_loll.scatter(ds, yi, color=c_bar, s=45, zorder=4, alpha=alpha)
            if np.isfinite(ci_arr[yi]):
                ax_loll.errorbar(ds, yi, xerr=ci_arr[yi],
                                 fmt="none", color=c_bar, capsize=2.5,
                                 lw=0.9, alpha=alpha)
            # Stars from raw t-test p (guaranteed consistent with CI)
            star = _star(raw_p)
            if star:
                ci_end = ci_arr[yi] if np.isfinite(ci_arr[yi]) else 0.0
                x_star = (ds + ci_end + 0.012) if ds >= 0 else (ds - ci_end - 0.012)
                ax_loll.text(x_star, yi, star,
                             ha="left" if ds >= 0 else "right",
                             va="center", fontsize=FS,
                             fontweight="bold", color=c_bar)

        ax_loll.axvline(0,     color="black",  lw=0.9, ls="--", alpha=0.5)
        _g1, _g2 = _mstyle["guides"]
        ax_loll.axvline( _g1, color="#cccccc", lw=0.5, ls=":",  alpha=0.7)
        ax_loll.axvline(-_g1, color="#cccccc", lw=0.5, ls=":",  alpha=0.7)
        ax_loll.axvline( _g2, color="#aaaaaa", lw=0.5, ls=":",  alpha=0.7)
        ax_loll.axvline(-_g2, color="#aaaaaa", lw=0.5, ls=":",  alpha=0.7)
        ax_loll.set_yticks(y_pos); ax_loll.set_yticklabels([], fontsize=max(FS-8, 1))
        # xlim: data-driven with padding; min ±0.20 so reference lines are always visible
        _max_abs = max(float(np.nanmax(np.abs(ds_arr))), 0.20)
        _ci_max  = float(np.nanmax(ci_arr[np.isfinite(ci_arr)])) if np.any(np.isfinite(ci_arr)) else 0.0
        _pad = max(_max_abs + _ci_max, _mstyle["pad_floor"]) * 1.15
        _pad = min(_pad, _mstyle["pad_cap"])
        ax_loll.set_xlim(-_pad, _pad)
        # Ticks that belong to this statistic, and only the ones in range:
        # the +/-0.4 pair of the rank deviation collided in the middle of the
        # panel once the scale became SDs.
        #
        # The z panel's width is set by the data, and since the cap was raised
        # it ranges from about 1.5 SD (VANISH) to 7.5 (BlossumH+). A fixed
        # ladder is either too sparse at one end or unreadable at the other, so
        # the step is chosen from the drawn range and the count stays between
        # three and five whatever the list.
        if _mstyle["metric"] == "z":
            _step   = 1.0 if _pad <= 3.5 else (2.0 if _pad <= 7.0 else 3.0)
            _ticks  = [(v, "0" if abs(v) < 1e-9 else f"{v:+.0f}")
                       for v in np.arange(-_step * 6, _step * 6 + 1e-9, _step)
                       if abs(v) <= _pad]
        else:
            _ticks = [(t, l) for t, l in zip(*_mstyle["score_ticks"])
                      if abs(t) <= _pad]
        ax_loll.set_xticks([t for t, _ in _ticks])
        ax_loll.set_xticklabels([l for _, l in _ticks])
        # Say which p the stars are. They come from the raw per-taxon t-test
        # so that a star and a CI crossing zero can never disagree, while the
        # right-hand panel colours by the BH q. Unlabelled, a reader takes
        # them for corrected significance across ~100 taxa.
        ax_loll.set_xlabel(_mstyle["axis_label"] + "\n"
                           "stars: uncorrected p; BH q in the right panel",
                           fontsize=FS)
        ax_loll.tick_params(axis="x", labelsize=FS-1, pad=1)
        ax_loll.tick_params(axis="y", labelleft=False, length=0)
        ax_loll.set_title("Score", fontsize=FS+1)
        ax_loll.grid(axis="x", alpha=0.18)

        # ── Align heatmap and lollipop rows exactly ───────────────────
        # imshow places n_tax rows centred on integers 0..n_tax-1,
        # so its ylim is (-0.5, n_tax-0.5).  Force the same range on
        # ax_loll before inverting, so rows line up perfectly.
        ax_heat.set_ylim(n_tax - 0.5, -0.5)   # already inverted by imshow
        ax_loll.set_ylim(-0.5, n_tax - 0.5)   # set before invert
        ax_loll.invert_yaxis()                  # now matches heatmap

        # ── Diagnostic scatter: driver_score vs consistency ───────────────
        valid = sub.dropna(subset=["driver_score", "consistency"]).copy()
        q_sc  = valid.get("q_value", pd.Series(1.0, index=valid.index)
                          ).fillna(1.0).values
        logq  = -np.log10(np.maximum(q_sc, 1e-4))

        c_scatter = np.where(valid["driver_score"].values >= 0,
                             COL_ENRICH, COL_DEPLET)
        # One marker size, one opacity. Size and alpha both encoded the
        # q-value, which the significance stars in the panel beside this one
        # already carry; varying them only made the panel read as noisy, and
        # the smallest markers fell under the 6 pt symbol minimum.
        _SZ, _ALPHA = 26, 0.85

        for i, (_, row) in enumerate(valid.iterrows()):
            ax_sc.scatter(row["consistency"], row["driver_score"],
                          color=c_scatter[i], s=_SZ, alpha=_ALPHA,
                          edgecolors="white", linewidths=0.4, zorder=3)

        # No per-taxon labels — too cluttered to be meaningful

        ax_sc.axhline(0, color="black", lw=0.7, ls="--", alpha=0.4)
        ax_sc.axhline( _g1, color="#cccccc", lw=0.5, ls=":", alpha=0.7)
        ax_sc.axhline(-_g1, color="#cccccc", lw=0.5, ls=":", alpha=0.7)
        ax_sc.set_xlim(0.48, 1.02)
        ax_sc.set_ylim(-_mstyle["pad_cap"], _mstyle["pad_cap"])

        for xpos, ypos, txt in [
            (0.98, 0.94, "Enriched"),
            (0.98, 0.06, "Depleted"),
        ]:
            ax_sc.text(xpos, ypos, txt,
                       transform=ax_sc.transAxes,
                       ha="right",
                       va="top" if ypos > 0.5 else "bottom",
                       fontsize=FS-1, color="#777777", style="italic")

        ax_sc.set_xlabel("Consistency", fontsize=FS)
        ax_sc.set_ylabel("")
        ax_sc.tick_params(axis="x", labelsize=FS-1, pad=1)
        ax_sc.tick_params(axis="y", labelleft=False, length=0)
        ax_sc.set_title("Cons.", fontsize=FS+1)
        ax_sc.grid(alpha=0.15)

        fig.suptitle(
            f"{list_name} taxon drivers  |  {ref}  |  {spec_desc}",
            fontsize=FS+1, fontweight="bold")

        # Do NOT call tight_layout here — it overrides subplots_adjust
        safe = list_name.replace("+", "plus").replace(" ", "_")

        # The figure's data object. This function already takes driver_df as
        # its input, so the rows it plots are a filtered view of it; the only
        # thing added is the taxonomy label the axes show, which otherwise
        # lives in a lookup the table would not carry.
        _slot = _TAXON_DRIVER_SLOTS.get(list_name)
        if _slot and spec_tag == "10pct":
            _tbl = driver_df[driver_df["Taxon_list"] == list_name].copy()
            # The column is lowercase here; guarding on "Taxon" silently
            # skipped this and the axis fell back to raw SGB identifiers on
            # redraw. Carry the full taxonomy string, which is what builds
            # the label, plus the rendered label for a reader of the table.
            _tcol = "taxon" if "taxon" in _tbl.columns else "Taxon"
            if _tcol in _tbl.columns:
                # driver_df already carries "taxonomy"; only fill it in when
                # it is absent, and add the rendered axis label beside it.
                if "taxonomy" not in _tbl.columns:
                    _tbl.insert(1, "taxonomy",
                                [taxonomy_map.get(t, "") for t in _tbl[_tcol]])
                if "taxon_label" not in _tbl.columns:
                    _tbl.insert(2, "taxon_label",
                                _labels(_tbl[_tcol]))
            figure_source.emit(
                s3_main, _slot, {"drivers": _tbl.reset_index(drop=True)},
                stem=f"health_taxon_drivers_{safe}_{ref}_{spec_tag}",
                reference=ref,
                extras={"drawn_conditions": list(all_conds)},
                notes=f"Taxon list: {list_name}. {spec_desc}")

        out  = os.path.join(
            s3_main,
            f"health_taxon_drivers_{safe}_{ref}_{spec_tag}.png")
        savefig_dual(fig, out, dpi=SUBMISSION_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {out}")


def run_taxon_driver_analysis(data, taxon_lists, per_taxon_df, bg_df,
                               spec, s3_main, taxonomy_df=None,
                               focus_lists=None, conditions=None, curve=None):
    """
    Entry point: per-taxon driver identification (Figures S26-S29).

    Runs _run_taxon_drivers (per-condition statistic, driver score, FDR) and
    saves the combined CSV plus the three-panel figure per list. With `curve`
    the statistic is each taxon's distance from the background curve in SDs;
    without it, the old weighted percentile rank deviation.

    Parameters
    ----------
    data         : ReferenceData
    taxon_lists  : dict list_name -> [taxon_id, ...]  (kept for API parity)
    per_taxon_df : DataFrame from _run_per_taxon_gmw
    bg_df        : DataFrame from _run_prevalence_matched_background
    spec         : DetectionSpec
    s3_main      : str  output directory
    taxonomy_df  : optional DataFrame with GTDB taxonomy for labelling
    focus_lists  : list of list names to analyse (default: all available)
    curve        : dict from _run_background_curve, or None
    """
    ref       = data.ref_name
    spec_tag  = spec.short_label
    spec_desc = spec.description

    if per_taxon_df is None or per_taxon_df.empty:
        return
    if bg_df is None or bg_df.empty:
        return

    _metric = "curve residual (z)" if (curve or {}).get("taxa") is not None \
        else "weighted percentile rank"
    print(f"    [taxon_drivers] driver analysis by {_metric} ({spec_desc}) ...")

    if taxonomy_df is not None and not taxonomy_df.empty:
        try:
            from .section2_pooling import _build_taxonomy_map
            tax_map = _build_taxonomy_map(taxonomy_df)
        except ImportError:
            tax_map = _load_taxonomy_map_s3()
    else:
        tax_map = _load_taxonomy_map_s3()

    # ── S26-S29: the cluster-bootstrap statistic and the family panels ──────
    # Preferred path. The t-test below has df = 3 on the four focal contrasts
    # and returns zero significant taxa for BloSSUM+ and BlossumH+, so it
    # cannot carry these figures; health_driver_bootstrap resamples the study
    # cohorts instead. It needs the per-cohort shifts, which the Mantel-
    # Haenszel collapse discards, so they are rebuilt here from `data`.
    #
    # Everything after this point is the older per-condition t-test. It still
    # runs: it writes health_taxon_drivers_<spec>.csv, which Table S7 and the
    # supplementary text read. Only the FIGURES moved.
    _drew = {}
    if curve is not None and (curve or {}).get("taxa") is not None:
        try:
            _drew = _run_driver_family_panels(
                data, taxon_lists, curve, spec, s3_main, tax_map,
                conditions=conditions, ref=ref, spec_desc=spec_desc)
            if _drew:
                print(f"    [taxon_drivers] family panels: "
                      + ", ".join(f"{k} -> {os.path.basename(v)}"
                                  for k, v in sorted(_drew.items()) if v))
        except ImportError:
            # Deliberately NOT swallowed. health_driver_bootstrap and
            # health_driver_family_panels are what produce S26-S29; without
            # them the fallback below quietly draws the superseded per-list
            # t-test panels into the same four slots, which is a different
            # analysis wearing the right figure numbers. A clone that is
            # missing these modules must stop, not improvise.
            raise
        except Exception as exc:                       # a drawing failure
            import traceback
            _drew = {}
            print(f"    [taxon_drivers] family panels FAILED: {exc}")
            traceback.print_exc()

    driver_df = _run_taxon_drivers(per_taxon_df, bg_df, focus_lists=focus_lists,
                                   conditions=conditions, curve=curve)

    if driver_df is None or driver_df.empty:
        print(f"    [taxon_drivers] No driver data produced — skipping")
        return

    # Annotate with taxonomy
    driver_df["taxonomy"]       = driver_df["taxon"].map(lambda t: tax_map.get(t, ""))
    driver_df["taxonomy_short"] = driver_df["taxon"].map(
        lambda t: _gtdb_short(tax_map.get(t, ""), n_ranks=2))

    out_csv = os.path.join(s3_main, f"health_taxon_drivers_{spec_tag}.csv")
    driver_df.to_csv(out_csv, index=False)
    print(f"    Saved: {out_csv}")

    # Draw the old per-list panels only when the family panels did not run.
    # Both write into the same directory, and the per-list figures are the
    # same four lists scored by the superseded t-test: leaving both sets on
    # disk puts two different answers to one question side by side, and the
    # manifest resolves only the family stems. The CSV above is written
    # either way, so Table S7 and the supplementary text are unaffected.
    if _drew:
        print("    [taxon_drivers] per-list t-test panels skipped; "
              "S26-S29 are the family panels")
    else:
        _plot_taxon_drivers(
            driver_df, tax_map, ref, s3_main, spec_tag, spec_desc,
            focus_lists=focus_lists)



# ====================================================================
#  TRIPLET HEALTH RICHNESS  (NI x IndH x IndU, GMW depth-corrected)
# ====================================================================

# ── 3-group GMW helpers (defined here so they stay near usage) ──────────────

def _gmw3_weights(x, kde_own, kde_b, kde_c, weight_clip=10.0):
    """
    Weights for 3-group geometric-mean target: (f_a*f_b*f_c)^(1/3).
    w_x = target(x)/f_x(x) = (f_b(x)*f_c(x)/f_x(x)^2)^(1/3)
    Mean-normalised to 1, clipped at weight_clip.
    """
    f_x = np.maximum(kde_own(x), 1e-10)
    f_b = np.maximum(kde_b(x),   1e-10)
    f_c = np.maximum(kde_c(x),   1e-10)
    w = (f_b * f_c / f_x**2) ** (1.0/3.0)
    w = w / w.mean()
    return np.clip(w, 0.0, weight_clip)


def _gmw3_target_grid(depth_grid, kde_a, kde_b, kde_c):
    """(f_a(x)*f_b(x)*f_c(x))^(1/3) on a dense grid — for visualisation."""
    fa = np.maximum(kde_a(depth_grid), 0.0)
    fb = np.maximum(kde_b(depth_grid), 0.0)
    fc = np.maximum(kde_c(depth_grid), 0.0)
    return (fa * fb * fc) ** (1.0/3.0)


# ── Health-richness GMW figure: pieces hoisted out of _run_health_richness_gmw
#    so the figure can be rebuilt from its published table alone. ────────────

#: Which lifestyle/health groups each comparison set contrasts, and in which
#: order. The published table carries the set label only, so this is what
#: tells a redraw which pairs to rebuild.
HEALTH_PAIRWISE_SETS = [
    ("NI_health",    "NI vs IndH vs IndU",
     [("NI", "IndH"), ("NI", "IndU"), ("IndH", "IndU")]),
    ("lifestyle3",   "H-G vs Transitional vs IndH",
     [("H-G", "Transitional"), ("H-G", "IndH"), ("Transitional", "IndH")]),
    ("lifestyle_all","H-G vs Transitional vs Ind (all)",
     [("H-G", "Transitional"), ("H-G", "Ind (all)"), ("Transitional", "Ind (all)")]),
]


def health_group_colors():
    """Colour per comparison group, including the composite groups."""
    c = dict(config.LIFESTYLE_COLORS)
    c.update({
        "NI":        config.OI["orange"],
        "Ind (all)": config.OI["sky"],
        "IndH":      config.OI["green"],
        "IndU":      config.OI["blue"],
    })
    return c


def _health_pair_results_from_table(df, pairs, n_boot, rng, detection=None):
    """Re-derive the GMW comparisons for one set from the per-sample table.

    Mirrors the pipeline's own loop. Weights are recomputed rather than stored
    because a sample's weight depends on which pair it is compared within.
    """
    # These live in depth_bias_ipw and are imported where used, matching how
    # the pipeline function itself reaches for them.
    from .depth_bias_ipw import (
        _BinnedKDE, _geomean_weights, _ess, _gmw_bootstrap_2grp,
        _boot_pval, _raw_vs_corr_pval, _weighted_sem, _geomean_target_grid,
    )
    from scipy.stats import mannwhitneyu

    if detection is not None and "detection" in df.columns:
        df = df[df["detection"].astype(str) == str(detection)]

    rich = {g: sub["richness"].to_numpy(dtype=float)
            for g, sub in df.groupby("group", sort=False)}
    ld = {g: sub["log10_total_mapped_bp"].to_numpy(dtype=float)
          for g, sub in df.groupby("group", sort=False)}

    out = []
    for (ga, gb) in pairs:
        if ga not in rich or gb not in rich:
            continue
        ra, rb = rich[ga], rich[gb]
        ld_a, ld_b = ld[ga], ld[gb]
        kde_a, kde_b = _BinnedKDE(ld_a), _BinnedKDE(ld_b)
        w_a = _geomean_weights(ld_a, kde_a, kde_b)
        w_b = _geomean_weights(ld_b, kde_b, kde_a)
        mu_a = float(np.dot(w_a, ra) / max(w_a.sum(), 1e-300))
        mu_b = float(np.dot(w_b, rb) / max(w_b.sum(), 1e-300))
        boot = _gmw_bootstrap_2grp(ld_a, ld_b, ra, rb, w_a, w_b, n_boot, rng)
        valid = np.isfinite(boot[:, 2])
        pct = lambda c: (np.percentile(boot[valid, c], [2.5, 97.5])
                         if valid.any() else [np.nan, np.nan])
        _, p_raw = mannwhitneyu(ra, rb, alternative="two-sided")
        dg = np.linspace(min(ld_a.min(), ld_b.min()) - 0.1,
                         max(ld_a.max(), ld_b.max()) + 0.1, 300)
        out.append(dict(
            ga=ga, gb=gb, ra=ra, rb=rb, ld_a=ld_a, ld_b=ld_b,
            kde_a=kde_a, kde_b=kde_b, w_a=w_a, w_b=w_b,
            mu_a=mu_a, mu_b=mu_b, att=mu_a - mu_b,
            ci_a=pct(0), ci_b=pct(1), ci_att=pct(2),
            raw_a=float(ra.mean()), raw_b=float(rb.mean()),
            raw_sem_a=float(ra.std(ddof=1) / np.sqrt(len(ra))) if len(ra) > 1 else 0.,
            raw_sem_b=float(rb.std(ddof=1) / np.sqrt(len(rb))) if len(rb) > 1 else 0.,
            wsem_a=_weighted_sem(ra, w_a, mu_a),
            wsem_b=_weighted_sem(rb, w_b, mu_b),
            ess_a=_ess(w_a), ess_b=_ess(w_b),
            n_a=len(ra), n_b=len(rb),
            p_raw=p_raw,
            p_corr=_boot_pval(boot[valid, 2]) if valid.any() else np.nan,
            p_rvc_a=_raw_vs_corr_pval(ra, boot[valid, 0]) if valid.any() else np.nan,
            p_rvc_b=_raw_vs_corr_pval(rb, boot[valid, 1]) if valid.any() else np.nan,
            dg=dg, geom=_geomean_target_grid(dg, kde_a, kde_b),
        ))
    return out


#: Full names for the composite groups. config.LIFESTYLE_DISPLAY covers the
#: three lifestyles; these are the health-split and pooled groups this section
#: adds, and without them the internal short forms ("Ind (all)", "IndH") reach
#: the reader as-is.
GROUP_DISPLAY = {
    "Ind (all)": "Industrial (all)",
    "IndH": "Industrial, healthy",
    "IndU": "Industrial, unhealthy",
    "NI": "Non-Industrial",
}


def group_display(g):
    """The full published name of a comparison group."""
    return GROUP_DISPLAY.get(g, config.LIFESTYLE_DISPLAY.get(g, g))


#: Short group labels for places where the full name does not fit -- panel
#: headings, mainly. The caption defines them. Spelled out, a heading like
#: "Hunter-gatherer vs Agro-pastoralist" runs the width of the panel and
#: collides with the part label sitting in its top-left corner.
SHORT_GROUP = {
    "H-G": "H-G",
    "Transitional": "A-P",
    "Agro-pastoralist": "A-P",
    "Industrial": "I",
    "Ind (all)": "I (all)",
    "IndH": "I healthy",
    "IndU": "I unhealthy",
    "NI": "NI",
}


def short_group(g):
    """The short form of a group label, or the label itself."""
    return SHORT_GROUP.get(g, group_display(g))


def health_depth_ax(ax_d, ax_gm, groups_ld, geom_disp, dg, title,
                    FS=13, group_colors=None, show_ylabel=True):
    """Draw depth KDE panel: normalised KDE per group + GMW target on twinx.
    group_colors = health_group_colors() if group_colors is None else group_colors
    Guarantees at least 3 log-spaced x-axis ticks with Gb labels."""
    for grp, ld_g, ls in groups_ld:
        if len(ld_g) > 1:
            from scipy.stats import gaussian_kde as _gkde
            try:
                k  = _gkde(ld_g, bw_method="silverman")
                yv = k(dg); yv = yv / max(yv.max(), 1e-300)
                ax_d.plot(dg, yv, color=group_colors[grp],
                          linestyle=ls, linewidth=2.0,
                          label=group_display(grp))
                ax_d.fill_between(dg, yv, alpha=0.10, color=group_colors[grp])
            except Exception:
                pass
    gd_norm = geom_disp / max(geom_disp.max(), 1e-300)
    if ax_gm is None:
        # Both the group densities and the target are normalised to their own
        # maximum, so they already share a scale and the target needs no
        # second axis. Dropping the twin is what removes the right-hand label
        # that collided with the panel beside it.
        ax_d.plot(dg, gd_norm, color="black", lw=1.4, label="GMW target")
        ax_d.fill_between(dg, gd_norm, alpha=0.08, color="black")
    else:
        ax_gm.plot(dg, gd_norm, color="black", lw=2.5, label="GMW target")
        ax_gm.fill_between(dg, gd_norm, alpha=0.08, color="black")
        ax_gm.set_ylim(0, 1.3)
        ax_gm.set_yticks([0, 0.5, 1.0])
        ax_gm.set_yticklabels(["0", "0.5", "1.0"], fontsize=FS - 2)
        ax_gm.set_ylabel("GMW target (norm.)", fontsize=FS - 2)
        ax_gm.yaxis.set_label_coords(1.12, 0.5)
    lo_x = dg.min(); hi_x = dg.max()
    # dg is in log10(bases) — _ld() = log10(total_bp).
    # Candidate nice Gb values and their positions in log10(bases) space:
    #   0.1 Gb = 1e8 bases → log10 = 8.0
    #   1   Gb = 1e9 bases → log10 = 9.0
    #   10  Gb = 1e10 bases → log10 = 10.0  etc.
    NICE_GB  = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
    TICK_POS = [np.log10(v * 1e9) for v in NICE_GB]   # log10(bases)
    TICK_LBL = [str(v) for v in NICE_GB]

    ticks_in = [(pos, lbl)
                for pos, lbl in zip(TICK_POS, TICK_LBL)
                if lo_x <= pos <= hi_x]
    # Guarantee at least 3 ticks by expanding window ±0.6 log units
    if len(ticks_in) < 3:
        ticks_in = [(pos, lbl)
                    for pos, lbl in zip(TICK_POS, TICK_LBL)
                    if (lo_x - 0.6) <= pos <= (hi_x + 0.6)]
    # If still fewer than 3, generate evenly-spaced ticks from the data range
    if len(ticks_in) < 3:
        step = (hi_x - lo_x) / 4
        auto_pos = [lo_x + step * i for i in range(5)]
        ticks_in = [(p, f"{10**(p-9):.2g}") for p in auto_pos]

    t_pos = [p for p, _ in ticks_in]
    t_lbl = [l for _, l in ticks_in]
    ax_d.set_xticks(t_pos)
    ax_d.set_xticklabels(t_lbl, rotation=0, fontsize=FS - 1)
    ax_d.set_xlim(lo_x, hi_x)
    ax_d.set_xlabel("Sequencing depth (Gb)", fontsize=FS)
    # Science: a common axis label is not repeated across a row.
    ax_d.set_ylabel("Normalised density" if show_ylabel else "", fontsize=FS)
    ax_d.set_ylim(bottom=0)
    _t = ax_d.set_title(title, fontsize=FS + 1, fontweight="bold")
    # The heading is already in the short forms the caption defines; the
    # display pass would expand them back to the names that do not fit.
    _t._keep_text = True
    ax_d.tick_params(axis="both", labelsize=FS - 1)
    l1, lb1 = ax_d.get_legend_handles_labels()
    l2, lb2 = ([], []) if ax_gm is None else ax_gm.get_legend_handles_labels()
    ax_d.legend(l1 + l2, lb1 + lb2, fontsize=FS - 1, loc="upper left",
                framealpha=0.7)


DETECTION_TITLE = {"10pct": "\u226510% breadth", "300kb": "\u2265300 kb mapped"}


def _draw_health_richness_gmw(by_spec, out_png, set_title, ref,
                              spec_desc="", FS=13, group_colors=None,
                              rng=None):
    """Draw the health-richness GMW figure, laid out as S7 is.

    One comparison per row; the depth densities in column 1 and one column per
    detection threshold after that. Raw and depth-corrected are overlaid in
    each threshold panel rather than given a column each, which is what lets
    the thresholds be columns and removes two sets of panel titles: the column
    heading says which threshold, the row heading says which comparison, and
    the pale violin behind the solid one says which estimate.
    """
    from .depth_bias_ipw import _sig_bracket
    try:
        import seaborn as sns
    except ImportError:
        sns = None
    group_colors = health_group_colors() if group_colors is None else group_colors
    rng = np.random.default_rng(0) if rng is None else rng

    if not by_spec:
        return out_png
    n_pairs = len(by_spec[0][1])
    n_cols = 1 + len(by_spec)
    if not n_pairs:
        return out_png

    from matplotlib import gridspec as _gs
    fig = plt.figure(figsize=page_figsize(n_cols=n_cols, n_rows=n_pairs))
    gs = _gs.GridSpec(n_pairs, n_cols, figure=fig, hspace=0.38, wspace=0.22)

    axes = []
    for ri in range(n_pairs):
        pr0 = by_spec[0][1][ri]
        ga, gb = pr0["ga"], pr0["gb"]
        palette = {ga: group_colors[ga], gb: group_colors[gb]}
        order = [ga, gb]
        first_row = (ri == 0)

        # ── Column 0: the depth densities the weighting works from ───────
        ax_d = fig.add_subplot(gs[ri, 0])
        axes.append(ax_d)
        health_depth_ax(ax_d, None,
                        [(ga, pr0["ld_a"], "-"), (gb, pr0["ld_b"], "--")],
                        pr0["geom"], pr0["dg"],
                        "Sequencing depth" if first_row else "",
                        FS=FS, group_colors=group_colors, show_ylabel=True)
        ax_d.set_ylabel(f"{short_group(ga)} vs {short_group(gb)}\nNormalised density")
        ax_d.yaxis.label._keep_text = True

        # ── One column per detection threshold ───────────────────────────
        for ci, (det, prs) in enumerate(by_spec, start=1):
            pr = prs[ri]
            ax = fig.add_subplot(gs[ri, ci])
            axes.append(ax)

            # Raw behind, corrected in front, both on the same axis.
            raw_df = pd.concat(
                [pd.DataFrame({"Group": ga, "richness": pr["ra"]}),
                 pd.DataFrame({"Group": gb, "richness": pr["rb"]})],
                ignore_index=True)
            corr = []
            for grp, r_arr, w_g in ((ga, pr["ra"], pr["w_a"]),
                                    (gb, pr["rb"], pr["w_b"])):
                wn = w_g / w_g.sum()
                idx = rng.choice(len(r_arr), size=len(r_arr), replace=True, p=wn)
                corr.append(pd.DataFrame({"Group": grp, "richness": r_arr[idx]}))
            corr_df = pd.concat(corr, ignore_index=True)

            if sns is not None:
                sns.violinplot(data=raw_df, x="Group", y="richness", order=order,
                               hue="Group", hue_order=order, palette=palette,
                               legend=False, inner=None, cut=0, linewidth=0.0,
                               ax=ax, density_norm="width", alpha=0.30)
                sns.violinplot(data=corr_df, x="Group", y="richness", order=order,
                               hue="Group", hue_order=order, palette=palette,
                               legend=False, inner="box", cut=0, linewidth=0.7,
                               ax=ax, density_norm="width")

            for i, (raw_mu, raw_sem, mu, ci_, wsem) in enumerate([
                    (pr["raw_a"], pr["raw_sem_a"], pr["mu_a"], pr["ci_a"], pr["wsem_a"]),
                    (pr["raw_b"], pr["raw_sem_b"], pr["mu_b"], pr["ci_b"], pr["wsem_b"])]):
                # Raw in grey italic behind, corrected bold in front, as S7.
                # Placed as S7 places it: a hair off the violin centre, with
                # the gap made by trailing spaces rather than by a large
                # offset, which had pushed these out toward the axis.
                # A thin light halo: sitting this close to the centre the
                # label lies over the violin, and grey on a dark fill is what
                # moving it inward costs.
                _raw_t = ax.text(i - 0.04, raw_mu,
                                 f"{raw_mu:.0f}\u00b1{raw_sem:.0f}  ",
                                 va="center", ha="right", style="italic",
                                 color="0.30")
                _raw_t.set_path_effects([
                    _pe.withStroke(linewidth=1.6, foreground="white")])
                ax.errorbar(i, mu, yerr=[[abs(mu - ci_[0])], [abs(ci_[1] - mu)]],
                            fmt="D", color="white", ecolor="black",
                            elinewidth=1.1, capsize=3, markersize=4,
                            markeredgecolor="black", zorder=6)
                ax.text(i, mu, f"  {mu:.0f}\u00b1{wsem:.0f}",
                        va="center", ha="left", fontweight="bold", color="black")

            ax.set_xlim(-0.62, 1.62)
            ax.set_xlabel("")
            # Both threshold columns carry the unit. Aligned per column
            # below rather than pinned at one axes fraction: a fixed x set
            # for "1200" left the label of a "900" column adrift in the gap,
            # and the gap then had to be wide enough to hold it.
            ax.set_ylabel("SGBs per sample")
            ax.set_xticks([0, 1])
            _ticks = [f"{short_group(ga)}\nn={pr['n_a']:,}",
                      f"{short_group(gb)}\nn={pr['n_b']:,}"]
            for _lab in ax.set_xticklabels(_ticks):
                _lab._keep_text = True
            _sig_bracket(ax, 0, 1, 0.90, pr["p_corr"])
            if first_row:
                _t = ax.set_title(DETECTION_TITLE.get(str(det), str(det)))
                _t._keep_text = True

    # Each column's labels line up down the rows, at whatever its widest
    # tick label needs; resolved at draw time, so it follows the type pass.
    fig.align_ylabels(axes)
    add_panel_letters(fig, axes)
    plt.tight_layout()
    savefig_dual(fig, out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_png}")
    return out_png


def plot_health_richness_gmw(tables, out_png, set_label, set_title="",
                             ref="combined", spec_desc="", n_boot=None,
                             rng=None, FS=13):
    """Rebuild this figure from its published table, and from nothing else."""
    if rng is None:
        rng = np.random.default_rng(0)
    if n_boot is None:
        n_boot = config.OW_AIPWE_N_BOOT_BASE
    _sets = {lbl: (title, prs) for lbl, title, prs in HEALTH_PAIRWISE_SETS}
    _title, pairs = _sets[set_label]
    df = tables["samples"]

    # One column of the figure per detection threshold, in the order the
    # manuscript quotes them. A table written before both thresholds were
    # published carries only one, and the figure then has one column.
    if "detection" in df.columns:
        present = list(dict.fromkeys(df["detection"].astype(str)))
        order = [d for d in ("10pct", "300kb") if d in present]
        order += [d for d in present if d not in order]
    else:
        order = [None]

    by_spec = []
    for det in order:
        prs = _health_pair_results_from_table(df, pairs, n_boot, rng,
                                              detection=det)
        if prs:
            by_spec.append((det, prs))
    if not by_spec:
        return out_png
    return _draw_health_richness_gmw(by_spec, out_png,
                                     set_title or _title, ref, spec_desc, FS,
                                     rng=rng)


def _run_health_richness_gmw(data, dm, specs, s3_main, stats,
                              n_boot=500, rng_seed=42):
    """
    GMW depth-corrected richness comparisons.

    Produces two types of figures per spec:

    (A) Three 3x3 pairwise figures (n_pairs rows x 3 cols):
          NI_health   : NI vs IndH vs IndU
          lifestyle3  : H-G vs Transitional vs IndH
          lifestyle_all: H-G vs Transitional vs Ind (all)
        Col 0: depth KDEs + sqrt(f_a*f_b) pairwise target
        Col 1: raw richness violins, black mean+/-SEM diamond
        Col 2: GMW corrected violins, black mean+/-wSEM diamond + 95%CI

    (B) Two summary single-row figures (1 row x 3 cols):
          summary_NI_health   : NI, IndH, IndU
          summary_lifestyle   : H-G, Transitional, Ind (all)
        Uses 3-group GMW target (f_a*f_b*f_c)^(1/3) simultaneously.
        Col 0: depth KDEs for all 3 groups + cubic-root target
        Col 1: raw 3-group violins with mean+/-SEM
        Col 2: corrected 3-group violins with mean+/-wSEM + CI diamonds

    All groups use full cohort — no triplet subsetting.
    """
    from .depth_bias_ipw import (
        _BinnedKDE, _geomean_weights, _ess, _gmw_bootstrap_2grp,
        _boot_pval, _raw_vs_corr_pval, _weighted_sem,
        _geomean_target_grid, _sig_bracket,
        _TICK_LOG, _TICK_GB,
    )
    from .detection import build_presence
    from scipy.stats import mannwhitneyu
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    ref = data.ref_name
    rng = np.random.default_rng(rng_seed)
    meta = data.meta

    healthy_vals   = {"healthy", "no", "false", "0"}
    unhealthy_vals = {"unhealthy", "yes", "true", "1"}

    def _ids(lifestyle_list, unhealthy_filter=None):
        out = []
        for s in meta.index:
            if s not in data.breadth.index:
                continue
            if meta.loc[s, "Lifestyle"] not in lifestyle_list:
                continue
            if unhealthy_filter is not None:
                if str(meta.loc[s, "Unhealthy"]).strip().lower() \
                        not in unhealthy_filter:
                    continue
            out.append(s)
        return out

    GRP_IDS = {
        "H-G":          _ids(["H-G"]),
        "Transitional": _ids(["Transitional"]),
        "NI":           _ids(["H-G", "Transitional"]),
        "IndH":         _ids(["Industrial"], healthy_vals),
        "IndU":         _ids(["Industrial"], unhealthy_vals),
        "Ind (all)":    _ids(["Industrial"]),
    }
    GROUP_COLORS = dict(config.LIFESTYLE_COLORS)
    GROUP_COLORS.update({
        "NI":        config.OI["orange"],
        "Ind (all)": config.OI["sky"],
        "IndH":      config.OI["green"],
        "IndU":      config.OI["blue"],
    })
    # Any unlisted group (e.g. Hadza, Carter) falls back to pink
    _GROUP_COLORS_DEFAULT = config.OI["pink"]

    for g, ids in GRP_IDS.items():
        print(f"      {g}: n={len(ids):,}")

    # Which comparison set is which Supplementary Figure.
    # lifestyle3 and lifestyle_all moved down one on 2026-10-19 (were S13
    # and S12). NI_health is outside the rotated range.
    _GMW_RICHNESS_SLOTS = {
        "NI_health":     "S24",
        "lifestyle3":    "S12",
        "lifestyle_all": "S11",
    }
    PAIRWISE_SETS = HEALTH_PAIRWISE_SETS
    SUMMARY_SETS = [
        ("summary_NI_health",  "NI, IndH, IndU",
         ["NI", "IndH", "IndU"]),
        ("summary_lifestyle",  "H-G, Transitional, Ind (all)",
         ["H-G", "Transitional", "Ind (all)"]),
    ]

    def _ld(ids):
        return np.log10(np.maximum(
            data.total_bp.reindex(ids).fillna(1.0).values, 1.0))

    FS = 13   # base font size for summary richness figures
    def _depth_ax(ax_d, ax_gm, groups_ld, geom_disp, dg, title):
        return health_depth_ax(ax_d, ax_gm, groups_ld, geom_disp, dg,
                               title, FS=FS, group_colors=GROUP_COLORS)

    all_csv_rows = []
    # One figure per comparison set, not per detection threshold: the figure
    # shows both thresholds side by side, so the samples for every spec have
    # to be gathered before anything is drawn.
    _gmw_accum = {}

    for spec in specs:
        spec_tag  = spec.short_label
        spec_desc = spec.description
        print(f"\n    Spec: {spec_desc}")

        # Compute richness for every needed group
        all_needed = sorted({g for sets in [PAIRWISE_SETS, SUMMARY_SETS]
                             for _, _, grps in sets
                             for g in (grps if isinstance(grps[0], str)
                                       else [x for pair in grps for x in pair])})
        all_sample_ids = list(dict.fromkeys(
            s for g in all_needed for s in GRP_IDS[g]))
        presence = build_presence(
            data.breadth.reindex(all_sample_ids),
            data.depth.reindex(all_sample_ids),
            data.lengths, spec=spec)
        rich_all = presence.sum(axis=1)
        rich = {g: rich_all.reindex(GRP_IDS[g]).values.astype(float)
                for g in all_needed if GRP_IDS[g]}

        # ── (A) Pairwise 3x3 figures ──────────────────────────────────────
        for set_label, set_title, pairs in PAIRWISE_SETS:
            ok = all(len(GRP_IDS[g]) >= 10 for pair in pairs for g in pair)
            if not ok:
                continue
            n_pairs = len(pairs)

            pair_results = []
            for (ga, gb) in pairs:
                ra = rich[ga]; rb = rich[gb]
                ld_a = _ld(GRP_IDS[ga]); ld_b = _ld(GRP_IDS[gb])
                kde_a = _BinnedKDE(ld_a); kde_b = _BinnedKDE(ld_b)
                w_a = _geomean_weights(ld_a, kde_a, kde_b)
                w_b = _geomean_weights(ld_b, kde_b, kde_a)
                mu_a = float(np.dot(w_a, ra) / max(w_a.sum(), 1e-300))
                mu_b = float(np.dot(w_b, rb) / max(w_b.sum(), 1e-300))
                att  = mu_a - mu_b
                boot  = _gmw_bootstrap_2grp(ld_a, ld_b, ra, rb, w_a, w_b,
                                             n_boot, rng)
                valid = np.isfinite(boot[:,2])
                ci_a   = np.percentile(boot[valid,0],[2.5,97.5]) if valid.any() else [np.nan]*2
                ci_b   = np.percentile(boot[valid,1],[2.5,97.5]) if valid.any() else [np.nan]*2
                ci_att = np.percentile(boot[valid,2],[2.5,97.5]) if valid.any() else [np.nan]*2
                _, p_raw = mannwhitneyu(ra, rb, alternative="two-sided")
                p_corr   = _boot_pval(boot[valid,2]) if valid.any() else np.nan
                p_rvc_a  = _raw_vs_corr_pval(ra, boot[valid,0]) if valid.any() else np.nan
                p_rvc_b  = _raw_vs_corr_pval(rb, boot[valid,1]) if valid.any() else np.nan
                raw_sem_a = float(ra.std(ddof=1)/np.sqrt(len(ra))) if len(ra)>1 else 0.
                raw_sem_b = float(rb.std(ddof=1)/np.sqrt(len(rb))) if len(rb)>1 else 0.
                def _star(p):
                    return "***" if p<0.001 else "**" if p<0.01 \
                           else "*" if p<0.05 else "ns"
                dg = np.linspace(min(ld_a.min(),ld_b.min())-0.1,
                                  max(ld_a.max(),ld_b.max())+0.1, 300)
                print(f"      [{set_label}] {ga} vs {gb}: "
                      f"D={att:+.1f}  {_star(p_corr)}")
                stats.add(f"health_richness_{set_label}_{spec_tag}_{ga}_vs_{gb}",
                          "s3","main",
                          f"Richness GMW [{set_label}] {ga} vs {gb} [{spec_desc}]",
                          f"D={att:+.1f} [{ci_att[0]:+.1f},{ci_att[1]:+.1f}] {_star(p_corr)}",
                          n1=len(ra), n2=len(rb), p_value=p_corr)
                pair_results.append(dict(
                    ga=ga, gb=gb, ra=ra, rb=rb, ld_a=ld_a, ld_b=ld_b,
                    kde_a=kde_a, kde_b=kde_b,
                    w_a=w_a, w_b=w_b, mu_a=mu_a, mu_b=mu_b, att=att,
                    ci_a=ci_a, ci_b=ci_b, ci_att=ci_att,
                    raw_a=float(ra.mean()), raw_b=float(rb.mean()),
                    raw_sem_a=raw_sem_a, raw_sem_b=raw_sem_b,
                    wsem_a=_weighted_sem(ra, w_a, mu_a),
                    wsem_b=_weighted_sem(rb, w_b, mu_b),
                    ess_a=_ess(w_a), ess_b=_ess(w_b),
                    n_a=len(ra), n_b=len(rb),
                    p_raw=p_raw, p_corr=p_corr,
                    p_rvc_a=p_rvc_a, p_rvc_b=p_rvc_b, dg=dg,
                    geom=_geomean_target_grid(dg, kde_a, kde_b),
                ))
                all_csv_rows.append(dict(
                    spec=spec_desc, comparison=set_label,
                    group_a=ga, group_b=gb,
                    n_a=len(ra), n_b=len(rb),
                    ess_a=round(_ess(w_a),1), ess_b=round(_ess(w_b),1),
                    raw_a=round(float(ra.mean()),2), raw_b=round(float(rb.mean()),2),
                    corr_a=round(mu_a,2), corr_b=round(mu_b,2),
                    att=round(att,2), ci_lo=round(ci_att[0],2),
                    ci_hi=round(ci_att[1],2),
                    p_raw=round(p_raw,6), p_corr=round(p_corr,6),
                ))

            out_png = os.path.join(s3_main,
                f"health_richness_gmw_{ref}_{set_label}_{spec_tag}.png")

            # Three figures come out of this one loop, one per comparison set,
            # so the data has to be split by set_label rather than handed over
            # whole. Per sample: the identifier, its group, its richness and the
            # geometric-mean weight it was given. Everything the panels show --
            # the corrected means, the intervals, the effective sample sizes --
            # is computed from these four columns.
            _slot = _GMW_RICHNESS_SLOTS.get(set_label)
            if _slot:
                _seen, _blocks = set(), []
                for (_ga, _gb) in pairs:
                    for _g in (_ga, _gb):
                        if _g in _seen:
                            continue
                        _seen.add(_g)
                        _ids = list(GRP_IDS[_g])
                        _blocks.append(pd.DataFrame({
                            "sample_id": [str(x) for x in _ids],
                            # Which detection threshold this richness is at.
                            # Both are drawn, as two columns of the figure.
                            "detection": spec_tag,
                            "detection_desc": spec_desc,
                            "group": _g,
                            "richness": np.asarray(rich[_g], dtype=float),
                            "log10_total_mapped_bp": _ld(GRP_IDS[_g]),
                        }))
                if _blocks:
                    _acc = _gmw_accum.setdefault(
                        set_label, {"slot": _slot, "title": set_title,
                                    "blocks": []})
                    _acc["blocks"].extend(_blocks)

        # ── (B) Summary 1-row x 3-col figures (3-group GMW) ──────────────
        for set_label, set_title, grp_names in SUMMARY_SETS:
            if not all(len(GRP_IDS[g]) >= 10 for g in grp_names):
                continue
            ga, gb, gc = grp_names
            ra = rich[ga]; rb = rich[gb]; rc = rich[gc]
            ld_a = _ld(GRP_IDS[ga]); ld_b = _ld(GRP_IDS[gb]); ld_c = _ld(GRP_IDS[gc])
            kde_a = _BinnedKDE(ld_a); kde_b = _BinnedKDE(ld_b); kde_c = _BinnedKDE(ld_c)
            w_a = _gmw3_weights(ld_a, kde_a, kde_b, kde_c)
            w_b = _gmw3_weights(ld_b, kde_b, kde_a, kde_c)
            w_c = _gmw3_weights(ld_c, kde_c, kde_a, kde_b)
            def _wmean(r, w):
                return float(np.dot(w, r) / max(w.sum(), 1e-300))
            mu_a = _wmean(ra, w_a); mu_b = _wmean(rb, w_b); mu_c = _wmean(rc, w_c)
            def _bootstrap_wmean(r, w):
                vals = np.empty(n_boot)
                n = len(r)
                for b in range(n_boot):
                    idx = rng.choice(n, n, replace=True)
                    vals[b] = _wmean(r[idx], w[idx])
                return vals

            boot_a = _bootstrap_wmean(ra, w_a)
            boot_b = _bootstrap_wmean(rb, w_b)
            boot_c = _bootstrap_wmean(rc, w_c)
            def _ci(b): return [float(np.percentile(b, 2.5)),
                                  float(np.percentile(b, 97.5))]
            ci_a = _ci(boot_a); ci_b = _ci(boot_b); ci_c = _ci(boot_c)
            raw_sem  = {ga: float(ra.std(ddof=1)/np.sqrt(len(ra))),
                        gb: float(rb.std(ddof=1)/np.sqrt(len(rb))),
                        gc: float(rc.std(ddof=1)/np.sqrt(len(rc)))}
            raw_mu   = {ga: float(ra.mean()), gb: float(rb.mean()), gc: float(rc.mean())}
            corr_mu  = {ga: mu_a, gb: mu_b, gc: mu_c}
            corr_ci  = {ga: ci_a, gb: ci_b, gc: ci_c}
            corr_wsem = {ga: _weighted_sem(ra, w_a, mu_a),
                         gb: _weighted_sem(rb, w_b, mu_b),
                         gc: _weighted_sem(rc, w_c, mu_c)}
            ess_map  = {ga: _ess(w_a), gb: _ess(w_b), gc: _ess(w_c)}
            n_map    = {ga: len(ra), gb: len(rb), gc: len(rc)}
            w_map    = {ga: w_a, gb: w_b, gc: w_c}
            r_map    = {ga: ra,  gb: rb,  gc: rc}
            print(f"\n      [{set_label}] {ga}: corr={mu_a:.1f}  "
                  f"{gb}: corr={mu_b:.1f}  {gc}: corr={mu_c:.1f}")

            # ── Log 3-group GMW summary to stats report ─────────────
            for _g, _mu, _ci_, _wsem, _ess_val in [
                (ga, mu_a, ci_a, _weighted_sem(ra, w_a, mu_a), _ess(w_a)),
                (gb, mu_b, ci_b, _weighted_sem(rb, w_b, mu_b), _ess(w_b)),
                (gc, mu_c, ci_c, _weighted_sem(rc, w_c, mu_c), _ess(w_c)),
            ]:
                _tag = (f"gmw3_{set_label}_{_g}_{spec_tag}"
                        .replace(" ", "_").replace("(", "").replace(")", ""))
                stats.add(
                    _tag, "s1", "main",
	                    f"3-group GMW corrected richness: {_g} [{set_label}, {spec_desc}]",
	                    f"corr_mean={_mu:.1f} [95%CI:{_ci_[0]:.1f},{_ci_[1]:.1f}] "
	                    f"wsem={_wsem:.1f} ESS={_ess_val:.0f} n={len(GRP_IDS[_g]):,}",
	                    n1=len(GRP_IDS[_g]), effect_size=_ess_val,
	                )

            # ── Write 3-group summary CSV (read by within_group_stats) ──
            _summary3_rows = []
            for _g, _raw, _rsem, _mu, _ci_, _wsem, _ess_v, _w in [
                (ga, raw_mu[ga], raw_sem[ga], mu_a, ci_a,
                 _weighted_sem(ra, w_a, mu_a), _ess(w_a), w_a),
                (gb, raw_mu[gb], raw_sem[gb], mu_b, ci_b,
                 _weighted_sem(rb, w_b, mu_b), _ess(w_b), w_b),
                (gc, raw_mu[gc], raw_sem[gc], mu_c, ci_c,
                 _weighted_sem(rc, w_c, mu_c), _ess(w_c), w_c),
            ]:
                _summary3_rows.append(dict(
                    spec=spec_desc, set_label=set_label, group=_g,
                    n=len(GRP_IDS[_g]),
                    raw_mean=round(_raw, 2), raw_sem=round(_rsem, 3),
                    corr_mean=round(_mu, 2),
                    corr_ci_lo=round(_ci_[0], 2), corr_ci_hi=round(_ci_[1], 2),
                    corr_wsem=round(_wsem, 3), ess=round(_ess_v, 1),
                ))
            _s3_csv = os.path.join(
                s3_main,
                f"health_richness_gmw_{ref}_{set_label}_{spec_tag}.csv")
            pd.DataFrame(_summary3_rows).to_csv(_s3_csv, index=False)
            print(f"    Saved 3-group summary CSV: {_s3_csv}")

            _ps_rows = []
            for _g in grp_names:
                for _rich, _weight, _log_depth in zip(r_map[_g], w_map[_g], _ld(GRP_IDS[_g])):
                    _ps_rows.append(dict(
                        Group=_g,
                        richness=float(_rich),
                        weight=float(_weight),
                        log10_depth=float(_log_depth),
                    ))
            _ps_csv = os.path.join(
                s3_main,
                f"health_richness_gmw_{ref}_{set_label}_{spec_tag}_perSample.csv")
            pd.DataFrame(_ps_rows).to_csv(_ps_csv, index=False)
            print(f"    Saved 3-group per-sample CSV: {_ps_csv}")
            dg = np.linspace(
                min(ld_a.min(), ld_b.min(), ld_c.min()) - 0.1,
                max(ld_a.max(), ld_b.max(), ld_c.max()) + 0.1, 400)
            geom3  = _gmw3_target_grid(dg, kde_a, kde_b, kde_c)
            palette3 = {g: GROUP_COLORS[g] for g in grp_names}
            order3   = list(grp_names)
            fig, axes = plt.subplots(1, 3, figsize=page_figsize(n_cols=3))
            fig.suptitle(
                f"SGB richness: {set_title}  [{ref}]  |  {spec_desc}\n"
                f"3-group GMW target = (f_a * f_b * f_c)^(1/3)\n"
                "Col 0: depth KDEs + cubic-root target  |  "
                "Col 1: raw violins  |  Col 2: GMW corrected violins",
                fontsize=FS + 2, fontweight="bold")
            ax_d = axes[0]; ax_gm = ax_d.twinx()
            _depth_ax(ax_d, ax_gm,
                      [(ga, ld_a, "-"), (gb, ld_b, "--"), (gc, ld_c, ":")],
                      geom3, dg,
                      " / ".join(config.LIFESTYLE_DISPLAY.get(g, g)
                                 for g in (ga, gb, gc)))
            ax_r = axes[1]
            if sns is not None:
                raw_df = pd.DataFrame({
                    "Group":    [g for g in order3 for _ in range(n_map[g])],
                    "richness": np.concatenate([r_map[g] for g in order3]),
                })
                sns.violinplot(data=raw_df, x="Group", y="richness",
                               order=order3, hue="Group", hue_order=order3,
                               palette=palette3, legend=False,
                               inner="box", cut=0, linewidth=0.8,
                               ax=ax_r, density_norm="width")
            for i, g in enumerate(order3):
                mu = raw_mu[g]; sem = raw_sem[g]
                ax_r.errorbar(i, mu, yerr=[[sem],[sem]], fmt="D",
                              color="black", ecolor="black",
                              elinewidth=1.5, capsize=5, markersize=7, zorder=5)
                ax_r.text(i, mu, f"  {mu:.0f}\u00b1{sem:.0f}",
                          va="center", fontsize=FS, fontweight="bold", color="black")
            ax_r.set_title("Raw richness", fontsize=FS + 2, fontweight="bold")
            ax_r.set_ylabel("# SGBs per sample", fontsize=FS)
            ax_r.set_xlabel("")
            ax_r.set_xticks(range(len(order3)))
            ax_r.set_xticklabels(
                [config.LIFESTYLE_DISPLAY.get(g, g) + "\n" +
                 f"(n={n_map[g]:,})" for g in order3], fontsize=FS)
            ax_r.tick_params(axis="y", labelsize=FS - 1)
            ax_c = axes[2]
            if sns is not None:
                csubs = []
                for g in order3:
                    wn = w_map[g] / w_map[g].sum()
                    ir = rng.choice(len(r_map[g]), size=len(r_map[g]),
                                    replace=True, p=wn)
                    csubs.append(pd.DataFrame(
                        {"Group": g, "richness": r_map[g][ir]}))
                sns.violinplot(data=pd.concat(csubs, ignore_index=True),
                               x="Group", y="richness",
                               order=order3, hue="Group", hue_order=order3,
                               palette=palette3, legend=False,
                               inner="box", cut=0, linewidth=0.8,
                               ax=ax_c, density_norm="width")
            for i, g in enumerate(order3):
                mu = corr_mu[g]; ci_ = corr_ci[g]; wsem = corr_wsem[g]
                ax_c.errorbar(i, mu,
                              yerr=[[abs(mu - ci_[0])],[abs(ci_[1] - mu)]],
                              fmt="D", color="black", ecolor="black",
                              elinewidth=1.5, capsize=5, markersize=7, zorder=5)
                ax_c.text(i, mu, f"  {mu:.0f}\u00b1{wsem:.0f}",
                          va="center", fontsize=FS, fontweight="bold", color="black")
            ax_c.set_title("GMW corrected (3-group target)", fontsize=FS + 2,
                           fontweight="bold")
            ax_c.set_ylabel("# SGBs per sample", fontsize=FS)
            ax_c.set_xlabel("")
            ax_c.set_xticks(range(len(order3)))
            ax_c.set_xticklabels(
                [f"{g}\n(ESS={ess_map[g]:.0f})" for g in order3], fontsize=FS)
            ax_c.tick_params(axis="y", labelsize=FS - 1)
            plt.tight_layout()
            out_png = os.path.join(s3_main,
                f"health_richness_gmw_{ref}_{set_label}_{spec_tag}.png")
            savefig_dual(fig, out_png, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved: {out_png}")

    # Both thresholds are in hand now, so each set gets one data object and
    # one figure drawn from it.
    for _set_label, _acc in _gmw_accum.items():
        _t = {"samples": pd.concat(_acc["blocks"], ignore_index=True)}
        _sum = [r for r in all_csv_rows if r.get("comparison") == _set_label]
        if _sum:
            _t["comparisons"] = pd.DataFrame(_sum)
        _stem = f"health_richness_gmw_{ref}_{_set_label}_10pct"
        figure_source.emit(s3_main, _acc["slot"], _t, stem=_stem,
                           reference=ref, notes=_acc["title"])
        plot_health_richness_gmw(
            _t, os.path.join(s3_main, _stem + ".png"), _set_label,
            set_title=_acc["title"], ref=ref, rng=rng)

    if all_csv_rows:
        df = pd.DataFrame(all_csv_rows)
        df.to_csv(os.path.join(s3_main, f"health_richness_gmw_{ref}.csv"),
                  index=False)
        print(f"    Saved: health_richness_gmw_{ref}.csv")


# ====================================================================
#  RICHNESS DECOMPOSITION  (VANISH / BloSSUM / VanishH+ / BlossumH+)
# ====================================================================

def _run_richness_decomposition(data, spec, s3_main,
                                 vanish_h_ids=None, blossum_h_ids=None):
    """
    Decompose GMW-weighted richness differences into contributions from
    VANISH, BloSSUM, VanishH+, BlossumH+, and Other taxa, and report
    mean % microbial load carried by each set per group.

    Mirrors the standalone richness_decomposition.py script but runs
    inside the pipeline for both detection specs automatically.

    Saves: richness_decomposition_{spec_tag}.csv
    Also prints a formatted table to stdout.
    """
    from .depth_bias_ipw import _BinnedKDE, _geomean_weights

    bdf_raw, thresh = _build_presence_matrix(data, spec)
    meta    = data.meta
    adf     = data.abundance.reindex(meta.index).fillna(0.0)
    spec_tag = spec.short_label

    # Normalise breadth to 0-1 if stored as percent
    bdf = bdf_raw / 100.0 if bdf_raw.max().max() > 1.5 else bdf_raw

    vanish  = data.vanish_set  or set()
    blossum = data.blossum_set or set()
    vh_set  = set(vanish_h_ids  or [])
    bh_set  = set(blossum_h_ids or [])

    all_sgbs     = bdf.columns.tolist()
    vanish_cols  = [s for s in all_sgbs if s in vanish]
    blossum_cols = [s for s in all_sgbs if s in blossum]
    vh_cols      = [s for s in all_sgbs if s in vh_set]
    bh_cols      = [s for s in all_sgbs if s in bh_set]
    other_cols   = [s for s in all_sgbs
                    if s not in vanish and s not in blossum]

    def rvec(cols):
        if not cols:
            return pd.Series(0.0, index=meta.index)
        return (bdf[cols] >= thresh).sum(axis=1).reindex(meta.index).fillna(0)

    r_v  = rvec(vanish_cols)
    r_b  = rvec(blossum_cols)
    r_vh = rvec(vh_cols)
    r_bh = rvec(bh_cols)
    r_ot = rvec(other_cols)
    r_tot = r_v + r_b + r_ot   # total (VanishH+/BlossumH+ are subsets of other)

    # Load fractions
    def load_frac(cols):
        if not cols:
            return pd.Series(0.0, index=meta.index)
        return adf[cols].sum(axis=1)

    lf_v  = load_frac(vanish_cols)
    lf_b  = load_frac(blossum_cols)
    lf_vh = load_frac(vh_cols)
    lf_bh = load_frac(bh_cols)

    # GMW weighted mean helper
    def _gmw(idx, series):
        r   = series.reindex(idx).values.astype(float)
        ld  = meta["_log_depth"].reindex(idx)
        w   = np.ones(len(idx))
        valid = [i for i in idx if pd.notna(ld.loc[i])]
        if len(valid) >= 20:
            ldv = ld.loc[valid].values.astype(float)
            if len(np.unique(ldv)) >= 3:
                kde  = _BinnedKDE(ldv)
                wkde = _geomean_weights(ldv, kde, kde)
                pos  = {s: p for p, s in enumerate(valid)}
                for p, ix in enumerate(idx):
                    if ix in pos:
                        w[p] = wkde[pos[ix]]
        W = w.sum()
        return float(np.dot(w, r) / max(W, 1e-300))

    comparisons = _health_comparisons()

    rows = []
    W = 160
    print(f"\n{'─'*W}")
    hdr = (f"{'Condition':<28}  {'Dtot':>7}  {'DV':>7}  {'DB':>7}  "
           f"{'DVH':>7}  {'DBH':>7}  {'DOt':>7}  "
           f"{'V%':>6}  {'B%':>6}  {'VH%':>6}  {'BH%':>6}  "
           f"{'g1Vld':>7}  {'g2Vld':>7}  {'g1Bld':>7}  {'g2Bld':>7}")
    print(hdr)
    print(f"{'─'*W}")

    for cond_label, (comp_type, dcols) in comparisons.items():
        mask_g1, mask_g2 = build_group_masks(meta, comp_type, dcols)
        n1, n2 = int(mask_g1.sum()), int(mask_g2.sum())
        if n1 < 10 or n2 < 10:
            continue

        idx1 = meta.index[mask_g1 & meta.index.isin(bdf.index)]
        idx2 = meta.index[mask_g2 & meta.index.isin(bdf.index)]

        def diff(series):
            return _gmw(idx1, series) - _gmw(idx2, series)

        d_tot = diff(r_tot); d_v  = diff(r_v); d_b  = diff(r_b)
        d_vh  = diff(r_vh);  d_bh = diff(r_bh); d_ot = diff(r_ot)

        def pct(d):
            return 100 * d / d_tot if abs(d_tot) >= 0.01 else np.nan

        g1_v  = float(lf_v.reindex(idx1).mean()  * 100)
        g2_v  = float(lf_v.reindex(idx2).mean()  * 100)
        g1_b  = float(lf_b.reindex(idx1).mean()  * 100)
        g2_b  = float(lf_b.reindex(idx2).mean()  * 100)
        g1_vh = float(lf_vh.reindex(idx1).mean() * 100)
        g2_vh = float(lf_vh.reindex(idx2).mean() * 100)
        g1_bh = float(lf_bh.reindex(idx1).mean() * 100)
        g2_bh = float(lf_bh.reindex(idx2).mean() * 100)

        print(f"  {cond_label:<28}  {d_tot:+7.2f}  {d_v:+7.2f}  {d_b:+7.2f}  "
              f"{d_vh:+7.2f}  {d_bh:+7.2f}  {d_ot:+7.2f}  "
              f"{pct(d_v):5.1f}%  {pct(d_b):5.1f}%  "
              f"{pct(d_vh):5.1f}%  {pct(d_bh):5.1f}%  "
              f"{g1_v:6.2f}%  {g2_v:6.2f}%  {g1_b:6.2f}%  {g2_b:6.2f}%")

        rows.append(dict(
            condition=cond_label, threshold=spec.description,
            n_g1=n1, n_g2=n2,
            delta_total=round(d_tot, 3),
            delta_vanish=round(d_v, 3),   delta_blossum=round(d_b, 3),
            delta_vanishH=round(d_vh, 3), delta_blossumH=round(d_bh, 3),
            delta_other=round(d_ot, 3),
            pct_vanish=round(pct(d_v), 1)   if np.isfinite(pct(d_v))  else np.nan,
            pct_blossum=round(pct(d_b), 1)  if np.isfinite(pct(d_b))  else np.nan,
            pct_vanishH=round(pct(d_vh), 1) if np.isfinite(pct(d_vh)) else np.nan,
            pct_blossumH=round(pct(d_bh),1) if np.isfinite(pct(d_bh)) else np.nan,
            g1_vanish_load=round(g1_v, 3),  g2_vanish_load=round(g2_v, 3),
            g1_blossum_load=round(g1_b, 3), g2_blossum_load=round(g2_b, 3),
            g1_vanishH_load=round(g1_vh, 3),g2_vanishH_load=round(g2_vh, 3),
            g1_blossumH_load=round(g1_bh,3),g2_blossumH_load=round(g2_bh, 3),
        ))

    print(f"{'─'*W}")
    print("  D=GMW richness diff (g1-g2)  |  V%/B% = fraction of Dtotal  |"
          "  load = mean % mapped bases")

    df = pd.DataFrame(rows)
    if not df.empty:
        csv_path = os.path.join(s3_main, f"richness_decomposition_{spec_tag}.csv")
        df.to_csv(csv_path, index=False)
        print(f"    Saved: {csv_path}")
    return df

# ====================================================================
#  NEW FIGURE: GMW DEPTH-CORRECTED RICHNESS BY HEALTH GROUP
# ====================================================================

def _arm_weights(idx1, idx2, meta):
    """(w1, w2) depth weights for a pair of arms, ones where not estimable."""
    from .depth_bias_ipw import _BinnedKDE, _geomean_weights

    w1, w2 = np.ones(len(idx1)), np.ones(len(idx2))
    ld = meta["_log_depth"]
    g1 = [i for i in idx1 if pd.notna(ld.get(i))]
    g2 = [i for i in idx2 if pd.notna(ld.get(i))]
    if len(g1) >= 20 and len(g2) >= 20:
        l1 = ld.loc[g1].values.astype(float)
        l2 = ld.loc[g2].values.astype(float)
        if len(np.unique(l1)) >= 3 and len(np.unique(l2)) >= 3:
            k1, k2 = _BinnedKDE(l1), _BinnedKDE(l2)
            a = _geomean_weights(l1, k1, k2)
            b = _geomean_weights(l2, k2, k1)
            p1 = {i: p for p, i in enumerate(g1)}
            p2 = {i: p for p, i in enumerate(g2)}
            for p, i in enumerate(idx1):
                if i in p1:
                    w1[p] = a[p1[i]]
            for p, i in enumerate(idx2):
                if i in p2:
                    w2[p] = b[p2[i]]
    return w1, w2


def _wmean(vals, w):
    return float(np.dot(w, vals) / max(w.sum(), 1e-300))


#: A study needs this many samples and this much log10-depth spread before
#: its own slope is trusted over the cohort-wide one.
DEPTH_SLOPE_MIN_N = 30
DEPTH_SLOPE_MIN_SPREAD = 0.3


def _within_study_depth_slope(richness, meta, study_col="_study"):
    """Depth slope(s) of richness on log10 depth: per study, plus a fallback.

    Depth is a nuisance parameter, and this is the one place it is estimated.
    Two rejected alternatives, in the order they were tried:

    *Per stratum.* Refitting the slope inside each study stratum looks more
    flexible but is worse: five samples per arm barely identify a slope, and
    the adjustment applies `slope_error x depth_gap`, so an imprecise slope
    injects the noise it was meant to remove. In the smallest strata the two
    arms' depth ranges need not overlap at all, making it an extrapolation.

    *One cohort-wide slope.* Precise (+138.5 +/- 2.6 SGBs per log10 bp on the
    combined reference, t = 52) but misspecified: measured across the 71
    studies with enough samples and spread to support their own estimate, the
    slope has median 126 and an interquartile range of 76 to 170 -- a factor
    of 2.3 -- with an SD (125) nearly equal to the mean. Applying one slope
    to all of them would over-correct some studies and under-correct others.

    So: each study gets its own slope when it has DEPTH_SLOPE_MIN_N samples
    and DEPTH_SLOPE_MIN_SPREAD of log10 depth to fit it with, and otherwise
    inherits the cohort-wide within-study slope. A study slope is fitted on
    every sample in that study -- typically hundreds -- not on the handful
    inside one stratum, so it is both specific and stable.

    On functional form: the within-study relation is not exactly linear in
    log10 depth (binned residuals reach ~12 SGBs against a ~100 SGB range
    across depth deciles), but the adjustment only ever interpolates across a
    stratum's internal depth gap, which is on the order of 0.01-0.05 log10 bp.
    Over that span any smooth relation is locally linear, so the curvature
    over the full 3.5-log range does not enter.

    Returns (global_beta, global_se, {study: beta}, n_used, n_studies).
    """
    sc = study_col if study_col in meta.columns else (
        "Study" if "Study" in meta.columns else "StudyFolder")
    d = pd.DataFrame({"r": richness.reindex(meta.index),
                      "ld": meta["_log_depth"],
                      "s": meta[sc]}).dropna()
    if len(d) < 100 or d["s"].nunique() < 2:
        return np.nan, np.nan, {}, len(d), int(d["s"].nunique())

    g = d.groupby("s")
    r_c = d["r"] - g["r"].transform("mean")
    ld_c = d["ld"] - g["ld"].transform("mean")
    den = float((ld_c ** 2).sum())
    if den <= 0:
        return np.nan, np.nan, {}, len(d), int(d["s"].nunique())
    beta = float((r_c * ld_c).sum()) / den
    resid = r_c - beta * ld_c
    dof = max(len(d) - d["s"].nunique() - 1, 1)
    se = float(np.sqrt(float((resid ** 2).sum()) / dof / den))

    per_study = {}
    for study, dd in g:
        if len(dd) < DEPTH_SLOPE_MIN_N:
            continue
        if float(np.ptp(dd["ld"].values)) < DEPTH_SLOPE_MIN_SPREAD:
            continue
        x = dd["ld"].values - dd["ld"].values.mean()
        dxx = float((x ** 2).sum())
        if dxx <= 0:
            continue
        per_study[study] = float((x * (dd["r"].values
                                       - dd["r"].values.mean())).sum()) / dxx
    return beta, se, per_study, len(d), int(d["s"].nunique())


#: Depth bins for the small-stratum GMW. Few enough that three or four samples
#: per arm still populate them, many enough to remove a depth shift.
#:
#: Swept 2/3/4/6/8 on the combined reference: no plotted contrast moves by more
#: than 1.7 SGBs and most by under 0.3 (Unhealthy 0.09, Hypertension 0.10,
#: IBD 0.15, Obese 0.30, CRC 0.00), with no change to any p-value or sign.
#: The largest movement anywhere was 2.1 SGBs, on Metabolic Syndrome, which
#: has one stratum and is untested and unplotted. So this is a free parameter
#: rather than a tuned one -- worth knowing, since it was picked rather than
#: derived.
DEPTH_BINS_SMALL = 4


def _stratum_depth_adjusted(idx1, idx2, richness, meta, edges=None):
    """GMW-weighted arm means and difference within ONE study stratum.

    Both corrections apply in every stratum, and by the same estimator. Study
    stratification alone leaves the depth confound the GMW weighting exists to
    remove; GMW alone leaves the between-study confound. Doing either on its
    own was the defect in the two previous versions of this function.

    The only obstacle was resolution, not method: _BinnedKDE uses a Silverman
    bandwidth and needs a few dozen points per arm, while study strata are
    routinely 5-19 per arm, so GMW was silently skipped in 17-80% of strata
    depending on the contrast. But the weights only use the *ratio* of the two
    depth densities, and a ratio over four bins is estimable where a density
    is not. So both branches below are GMW -- w = sqrt(f_other/f_own),
    mean-normalised, clipped at WEIGHT_CLIP -- differing only in how the two
    densities are estimated:

      "gmw"        both arms >= 20 with depth: the published KDE version,
                   untouched, so large strata reproduce their previous values.
      "gmw_binned" otherwise: the same formula over DEPTH_BINS_SMALL
                   cohort-wide quantile bins, with a pseudo-count so a bin one
                   arm misses gives a large but finite weight.
      "none"       fewer than three depth-carrying samples in an arm. Counted.

    Two alternatives were tried and rejected. Refitting a depth *slope* per
    stratum is unstable at five per arm, and since an adjustment applies
    `slope_error x depth_gap` it injects the noise it should remove. One
    cohort-wide slope is precise (+138.5 +/- 2.6 SGBs per log10 bp, t = 52)
    but misspecified: across the 71 studies able to support their own
    estimate, the slope's interquartile range spans 76 to 170. Reweighting
    sidesteps the functional form altogether, which is why the original
    method used it.

    Returns (mu1, mu2, diff, method), diff = mu1 - mu2 as everywhere else.
    """
    from .depth_bias_ipw import binned_geomean_weights

    i1, i2 = list(idx1), list(idx2)
    r1 = richness.reindex(i1).values.astype(float)
    r2 = richness.reindex(i2).values.astype(float)
    ld = meta["_log_depth"]
    l1 = ld.reindex(i1).values.astype(float)
    l2 = ld.reindex(i2).values.astype(float)
    ok1, ok2 = np.isfinite(l1), np.isfinite(l2)

    if ok1.sum() >= 20 and ok2.sum() >= 20:
        w1, w2 = _arm_weights(i1, i2, meta)
        if not np.allclose(w1, 1.0) or not np.allclose(w2, 1.0):
            mu1, mu2 = _wmean(r1, w1), _wmean(r2, w2)
            return mu1, mu2, mu1 - mu2, "gmw"

    if edges is not None and len(edges) > 2 and ok1.sum() >= 3 and ok2.sum() >= 3:
        wa, wb = binned_geomean_weights(l1[ok1], l2[ok2], edges)
        # A sample with no depth cannot be reweighted; give it the arm's mean
        # weight so it neither drops out nor distorts the balance.
        w1 = np.full(len(r1), float(wa.mean()) if len(wa) else 1.0)
        w2 = np.full(len(r2), float(wb.mean()) if len(wb) else 1.0)
        w1[ok1] = wa
        w2[ok2] = wb
        mu1, mu2 = _wmean(r1, w1), _wmean(r2, w2)
        return mu1, mu2, mu1 - mu2, "gmw_binned"

    mu1 = float(np.mean(r1)) if len(r1) else np.nan
    mu2 = float(np.mean(r2)) if len(r2) else np.nan
    return mu1, mu2, mu1 - mu2, "none"


def _stratified_richness(richness, meta, bdf, mask_g1, mask_g2,
                         min_stratum=None, edges=None):
    """Study-stratified difference in depth-corrected richness.

    Why stratified. The pooled version of this took a depth-weighted mean of
    each arm across every study at once. Studies differ enormously in baseline
    richness -- measured on the combined reference, mean richness per study has
    SD 85 and spans 35 to 606 -- so whenever a condition is unevenly
    distributed across studies, the pooled difference carries that
    between-study spread rather than the within-study contrast. It is the same
    confound the per-taxon panel already controls for by running CMH over
    study strata, and the two halves of the figure disagreed about it.

    Estimator. Within each qualifying study the two arms are depth-weighted
    exactly as before, and the per-study differences are pooled with weights
    n1*n2/(n1+n2) -- the difference-of-means analogue of Mantel-Haenszel
    weighting, so panel D pools the way panel E does.

    Uncertainty. The bootstrap resamples *studies*, not samples. Samples
    within a study are not independent, and resampling them individually gave
    intervals far too narrow -- every contrast came back p < 0.005, including
    ones resting on a single study.

    Returns None when no study has `min_stratum` samples in both arms.
    """
    from .health_cmh import _get_study_strata

    if min_stratum is None:
        min_stratum = config.CMH_MIN_STRATUM
    strata = _get_study_strata(meta, mask_g1, mask_g2, bdf, min_stratum)
    if not strata:
        return None

    from .depth_bias_ipw import _ess as _ess_fn

    per = []
    for study, g1s, g2s in strata:
        mu1, mu2, d_s, method = _stratum_depth_adjusted(
            g1s, g2s, richness, meta, edges=edges)
        w1, w2 = _arm_weights(list(g1s), list(g2s), meta)
        n1, n2 = len(g1s), len(g2s)
        per.append(dict(study=study, mu1=mu1, mu2=mu2, d=d_s,
                        n1=n1, n2=n2, w=n1 * n2 / float(n1 + n2),
                        ess1=_ess_fn(w1), ess2=_ess_fn(w2),
                        method=method,
                        depth_weighted=(method != "none")))

    W = sum(x["w"] for x in per)
    diff = sum(x["w"] * x["d"] for x in per) / W
    mu_g1 = sum(x["w"] * x["mu1"] for x in per) / W
    mu_g2 = sum(x["w"] * x["mu2"] for x in per) / W

    # Uncertainty across studies, by robust variance estimation.
    #
    # Not a cluster bootstrap. Resampling studies looks right but fails
    # exactly where the strata are scarce: when every study agrees on the
    # sign, no replicate crosses zero and the p-value pins to its floor
    # (1/n_boot), so a contrast resting on one study reported p = 0.002 as
    # confidently as one resting on forty. Five of the nine disease contrasts
    # came back at precisely that floor.
    #
    # The estimator below is the standard sandwich SE for a weighted mean of
    # per-stratum effects, referred to a t distribution on k-1 degrees of
    # freedom. It costs power when k is small, which is the honest behaviour:
    # one study carries no information about between-study variability, so it
    # gets no interval and no p-value at all.
    k = len(per)
    d_arr = np.array([x["d"] for x in per], dtype=float)
    w_arr = np.array([x["w"] for x in per], dtype=float)
    if k >= 2:
        se_diff = float(np.sqrt(np.sum((w_arr * (d_arr - diff)) ** 2)) / W)
        if se_diff > 0:
            from scipy import stats as _st
            tstat = diff / se_diff
            pval = float(2 * _st.t.sf(abs(tstat), df=k - 1))
            crit = float(_st.t.ppf(0.975, df=k - 1))
            ci_lo, ci_hi = diff - crit * se_diff, diff + crit * se_diff
        else:
            ci_lo = ci_hi = pval = np.nan
    else:
        se_diff = np.nan
        ci_lo = ci_hi = pval = np.nan

    # Arm SEs by the same sandwich form as the difference, so all three
    # numbers in a row describe the same estimator. These are *between-study*
    # SEs of a stratum-weighted marginal mean, and they are an order of
    # magnitude larger than the within-arm SEMs the pooled version reported:
    # cohorts differ enormously in baseline richness (SD 85 across studies),
    # and that variance is present in each arm mean but cancels in the paired
    # within-study difference. That is the whole reason the difference can be
    # tight while the arm means are not, but it does mean the two panels of
    # the figure are answering different questions -- descriptive above,
    # inferential below -- and the caption has to say so.
    mu1_arr = np.array([x["mu1"] for x in per], dtype=float)
    mu2_arr = np.array([x["mu2"] for x in per], dtype=float)
    if k >= 2:
        se_mu1 = float(np.sqrt(np.sum((w_arr * (mu1_arr - mu_g1)) ** 2)) / W)
        se_mu2 = float(np.sqrt(np.sum((w_arr * (mu2_arr - mu_g2)) ** 2)) / W)
    else:
        se_mu1 = se_mu2 = np.nan

    # Effective sample size of the surviving strata, not of the whole arm:
    # samples in studies that failed the min-stratum test contribute nothing
    # to the reported estimate and must not be counted toward its ESS.
    ess1_s = float(sum(x["ess1"] for x in per))
    ess2_s = float(sum(x["ess2"] for x in per))

    # How many strata actually got depth-corrected. _arm_weights needs >= 20
    # depth-carrying samples in BOTH arms to fit the two KDEs and otherwise
    # passes weights of 1 through, so stratifying makes the depth correction
    # apply less often than it did pooled -- a small stratum is exactly where
    # it silently stops. Reported rather than hidden, because "GMW
    # depth-corrected" is only true of the strata this counts.
    frac_dw = float(np.mean([x["depth_weighted"] for x in per]))
    methods = [x["method"] for x in per]
    method_mix = "/".join(f"{m}:{methods.count(m)}"
                          for m in ("gmw", "gmw_binned", "none")
                          if methods.count(m))

    d = d_arr
    return dict(diff=diff, mu_g1=mu_g1, mu_g2=mu_g2, se_diff=se_diff,
                ci_lo=ci_lo, ci_hi=ci_hi, pval=pval,
                n_strata=k,
                n_g1_strat=int(sum(x["n1"] for x in per)),
                n_g2_strat=int(sum(x["n2"] for x in per)),
                frac_studies_same_sign=float((np.sign(d) == np.sign(diff)).mean()),
                se_g1=se_mu1, se_g2=se_mu2,
                ess_g1=ess1_s, ess_g2=ess2_s,
                frac_strata_depth_weighted=frac_dw,
                depth_method_mix=method_mix)


def _run_richness_comparison(data, spec, s3_main, stats, demographic=False):
    """
    Per-condition comparison of depth-corrected richness, Figure 5 panel D.

    Applies the same geom-mean-target IPW depth weighting as the GMW-CMH taxon
    tests to a *continuous* outcome (per-sample taxon richness), and -- since
    2026-09-25 -- over the same study strata. The previous version weighted on
    depth but pooled every study into one mean, so it answered a different
    question from panel E of the same figure. Between-study richness differs
    by far more than any contrast here (SD 85, range 35-606 across studies),
    which the pooled mean absorbed whenever a condition sat unevenly across
    studies; `Obese` came out enriched pooled and depleted stratified.

    Method
    ------
    1. Per-sample richness = number of taxa >= the detection threshold.
    2. Per condition, g1/g2 masks from build_group_masks().
    3. Pooled estimate, kept only as the `*_pooled` point-estimate columns so
       the size of the between-study component stays visible.
    4. Stratified estimate via _stratified_richness(): within each study with
       >= config.CMH_MIN_STRATUM samples per arm, depth-weight both arms and
       take the difference; pool those with weights n1*n2/(n1+n2), the
       difference-of-means analogue of Mantel-Haenszel.
    5. Uncertainty from the sandwich SE across strata on t(k-1) -- not a
       bootstrap. See _stratified_richness for why resampling failed here.

    A condition with fewer than two qualifying studies is reported with
    `sig="untested"` and no interval: one study carries no information about
    between-study variability. Figure 5 drops those columns
    (figure5.MIN_STRATA_TO_PLOT) rather than drawing a bar with no error bar.

    Returns
    -------
    DataFrame, one row per condition. Reported estimate: mu_g1, mu_g2, diff,
    se_g1, se_g2, ci_lo, ci_hi, pval, sig. Provenance: n_strata,
    n_g1_strat, n_g2_strat, frac_studies_same_sign,
    frac_strata_depth_weighted, diff_pooled, mu_g1_pooled, mu_g2_pooled.

    Note that se_g1/se_g2 are between-study SEs of a stratum-weighted
    marginal mean and are an order of magnitude larger than the within-arm
    SEMs the pooled version wrote into the same columns; the variance they
    carry cancels in the paired within-study difference, which is why the
    difference can be tight while the arm means are not.
    """
    from .depth_bias_ipw import _BinnedKDE, _geomean_weights, _ess

    bdf, thresh = _build_presence_matrix(data, spec)
    meta        = data.meta
    spec_tag    = spec.short_label

    # Per-sample richness vector (same index as meta)
    richness = (bdf >= thresh).sum(axis=1).reindex(meta.index).fillna(0)

    comparisons = _health_comparisons(demographic)

    # Cohort-wide depth bins for the small-stratum GMW, computed once so a
    # given depth lands in the same bin in every stratum.
    from .depth_bias_ipw import depth_bin_edges
    depth_edges = depth_bin_edges(meta["_log_depth"].values, DEPTH_BINS_SMALL)
    if len(depth_edges) > 2:
        print(f"    depth bins for small strata: {len(depth_edges)-1} "
              f"cohort quantiles of log10 depth "
              f"[{depth_edges[0]:.2f} .. {depth_edges[-1]:.2f}]")
    else:
        print("    [WARN] depth bins not estimable; small strata will be "
              "left unweighted and counted as method 'none'")

    rows = []
    for cond_label, (comp_type, dcols) in comparisons.items():
        mask_g1, mask_g2 = build_group_masks(meta, comp_type, dcols)
        n_g1, n_g2 = int(mask_g1.sum()), int(mask_g2.sum())
        if n_g1 < 10 or n_g2 < 10:
            continue

        idx1 = meta.index[mask_g1 & meta.index.isin(bdf.index)]
        idx2 = meta.index[mask_g2 & meta.index.isin(bdf.index)]

        r1 = richness.reindex(idx1).values.astype(float)
        r2 = richness.reindex(idx2).values.astype(float)

        ld1_series = meta.loc[idx1, "_log_depth"]
        ld2_series = meta.loc[idx2, "_log_depth"]

        # Build weights — uniform for depth-less samples
        w1 = np.ones(len(idx1))
        w2 = np.ones(len(idx2))

        g1_has = [i for i in idx1 if pd.notna(ld1_series.loc[i])]
        g2_has = [i for i in idx2 if pd.notna(ld2_series.loc[i])]

        if len(g1_has) >= 20 and len(g2_has) >= 20:
            ld1 = ld1_series.loc[g1_has].values.astype(float)
            ld2 = ld2_series.loc[g2_has].values.astype(float)
            if len(np.unique(ld1)) >= 3 and len(np.unique(ld2)) >= 3:
                kde1 = _BinnedKDE(ld1)
                kde2 = _BinnedKDE(ld2)
                w1_kde = _geomean_weights(ld1, kde1, kde2)
                w2_kde = _geomean_weights(ld2, kde2, kde1)
                g1_pos = {i: p for p, i in enumerate(g1_has)}
                g2_pos = {i: p for p, i in enumerate(g2_has)}
                for p, idx in enumerate(idx1):
                    if idx in g1_pos:
                        w1[p] = w1_kde[g1_pos[idx]]
                for p, idx in enumerate(idx2):
                    if idx in g2_pos:
                        w2[p] = w2_kde[g2_pos[idx]]

        # Weighted means
        W1, W2 = w1.sum(), w2.sum()
        mu1 = float(np.dot(w1, r1) / max(W1, 1e-300))
        mu2 = float(np.dot(w2, r2) / max(W2, 1e-300))
        diff = mu1 - mu2   # positive = g1 richer than g2

        # Weighted SEM
        def _wsem(r, w, mu):
            W = w.sum()
            return float(np.sqrt((w * (r - mu)**2).sum()) / max(W, 1e-300))

        se1 = _wsem(r1, w1, mu1)
        se2 = _wsem(r2, w2, mu2)

        # No interval or p-value is computed for the pooled estimate. The
        # sample-level bootstrap that used to live here treats samples within
        # a study as independent, which they are not, so its interval was too
        # narrow and its p-value pinned to 1/n_boot. Keeping it as a
        # "diagnostic" would only invite it to be quoted. The pooled *point*
        # estimate is kept below, because the gap between it and the
        # stratified one is exactly how much of the old number was
        # between-study.

        # ── Study stratification ──────────────────────────────────────
        # Everything above is the pooled estimate. It is kept, but only as a
        # diagnostic column: between-study differences in baseline richness
        # are several times larger than any contrast measured here, so the
        # pooled difference reports them whenever a condition sits unevenly
        # across studies. The reported estimate is the stratified one, which
        # is what the per-taxon panel of the same figure already does.
        pooled_diff, pooled_mu1, pooled_mu2 = diff, mu1, mu2

        # No rng: the estimator is closed-form, not resampled.
        strat = _stratified_richness(richness, meta, bdf, mask_g1, mask_g2,
                                     edges=depth_edges)
        if strat is None:
            # No study has both arms. A pooled number here would be a
            # between-study comparison wearing a within-study label, so the
            # row is dropped rather than reported.
            print(f"    {cond_label:<28s}  [SKIP] no study has "
                  f">={config.CMH_MIN_STRATUM} samples in both arms")
            continue

        mu1, mu2, diff = strat["mu_g1"], strat["mu_g2"], strat["diff"]
        ci_lo, ci_hi, pval = strat["ci_lo"], strat["ci_hi"], strat["pval"]
        se1, se2 = strat["se_g1"], strat["se_g2"]

        # "untested" is not "not significant": a single-study contrast has no
        # between-study variance to estimate, so it gets no p-value, and
        # showing "ns" there would claim a test that was never run.
        if not np.isfinite(pval):
            star = "untested"
        else:
            star = ("***" if pval < 0.001 else
                    "**"  if pval < 0.01  else
                    "*"   if pval < 0.05  else "ns")

        # From the surviving strata, matching the reported estimate. The
        # pooled _ess(w1)/_ess(w2) described every Industrial sample in the
        # arm, including those in studies that contribute nothing to it.
        ess1, ess2 = strat["ess_g1"], strat["ess_g2"]

        _p_txt = f"p={pval:.3e}" if np.isfinite(pval) else "p=n/a"
        print(f"    {cond_label:<28s}  mu_g1={mu1:6.1f}  mu_g2={mu2:6.1f}  "
              f"diff={diff:+6.1f}  {_p_txt} {star:<8s} "
              f"[{strat['n_strata']} studies, "
              f"{strat['frac_studies_same_sign']*100:.0f}% agree]"
              f"  pooled would be {pooled_diff:+.1f}")

        # Only the new run gets a prefix: renaming the disease tags would
        # silently change every manuscript number keyed on them.
        _tag_pfx = "demographic_" if demographic else ""
        stats.add(f"{_tag_pfx}richness_gmw_{cond_label}_{spec_tag}",
                  "s3", "main",
                  f"GMW richness {cond_label} [{spec.description}]: "
                  f"g1={mu1:.1f} vs g2={mu2:.1f}, diff={diff:+.1f}",
                  f"diff={diff:+.1f}, {_p_txt} {star}",
                  n1=n_g1, n2=n_g2, p_value=pval, effect_size=round(diff, 2))

        rows.append(dict(
            Condition=cond_label, comp_type=comp_type,
            n_g1=n_g1, n_g2=n_g2,
            ess_g1=round(ess1, 1), ess_g2=round(ess2, 1),
            mu_g1=round(mu1, 2), mu_g2=round(mu2, 2),
            diff=round(diff, 2), se_g1=round(se1, 2), se_g2=round(se2, 2),
            ci_lo=round(ci_lo, 2) if np.isfinite(ci_lo) else np.nan,
            ci_hi=round(ci_hi, 2) if np.isfinite(ci_hi) else np.nan,
            pval=round(pval, 6) if np.isfinite(pval) else np.nan,
            sig=star,
            # Stratification provenance, so a reader can see how much of the
            # pooled number was between-study and how consistent the studies
            # are, without re-running anything.
            n_strata=strat["n_strata"],
            n_g1_strat=strat["n_g1_strat"], n_g2_strat=strat["n_g2_strat"],
            frac_studies_same_sign=round(strat["frac_studies_same_sign"], 3),
            frac_strata_depth_weighted=round(
                strat["frac_strata_depth_weighted"], 3),
            depth_method_mix=strat["depth_method_mix"],

            se_diff=(round(strat["se_diff"], 3)
                     if np.isfinite(strat.get("se_diff", np.nan)) else np.nan),
            diff_pooled=round(pooled_diff, 2),
            mu_g1_pooled=round(pooled_mu1, 2),
            mu_g2_pooled=round(pooled_mu2, 2),
        ))

    df = pd.DataFrame(rows)
    if not df.empty:
        csv_path = os.path.join(
            s3_main,
            f"{_out_prefix(demographic)}_richness_gmw_{spec_tag}.csv")
        df.to_csv(csv_path, index=False)
        print(f"    Saved: {csv_path}")
    return df


def _plot_richness_comparison(richness_df, ref, s3_main,
                               spec_tag, spec_desc):
    """
    Two-panel grouped bar chart: GMW depth-corrected richness.

    Top panel   — paired bars (control light, disease dark) with SEM error bars.
                  No x-axis labels; no broken axis.
    Bottom panel — signed difference (control minus disease) with bootstrap
                  95% CI and permutation significance stars.
    """
    if richness_df is None or richness_df.empty:
        return

    df     = richness_df[~richness_df["Condition"].map(_condition_dropped)].copy()
    # Display order, pooled contrast last. The bars are drawn by iterating
    # df positionally, so the FRAME has to carry the order -- reordering only
    # the label list would put every bar under the wrong heading, which is
    # the same trap panel D had.
    conds  = _display_order(list(df["Condition"]))
    _rank  = {c: i for i, c in enumerate(conds)}
    df     = (df.assign(_o=df["Condition"].map(_rank))
                .sort_values("_o", kind="stable")
                .drop(columns="_o")
                .reset_index(drop=True))
    n_cond = len(conds)
    if n_cond == 0:
        return

    health_conds = {"Unhealthy", "Obese"}
    bar_color    = config.OI["sky"]   # uniform sky blue

    all_mu = list(df["mu_g1"]) + list(df["mu_g2"])
    mu_max = max(all_mu)

    # ── Layout: 2 rows, no broken axis ───────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=page_figsize(n_rows=2),
        gridspec_kw={"height_ratios": [1.0, 0.5]},
        sharex=True)

    x  = np.arange(n_cond)
    bw = 0.32

    # ── Top panel: paired bars, no x-axis labels ──────────────────────
    for ci, (_, row) in enumerate(df.iterrows()):
        # Control — light
        ax_top.bar(x[ci] - bw/2, row["mu_g1"], bw * 0.92,
                   color=bar_color, alpha=0.40,
                   edgecolor=bar_color, lw=1.0)
        ax_top.errorbar(x[ci] - bw/2, row["mu_g1"], yerr=row["se_g1"],
                        fmt="none", color=bar_color, capsize=3, lw=1.2)
        # Disease — dark
        ax_top.bar(x[ci] + bw/2, row["mu_g2"], bw * 0.92,
                   color=bar_color, alpha=0.88,
                   edgecolor="white", lw=0.5)
        ax_top.errorbar(x[ci] + bw/2, row["mu_g2"], yerr=row["se_g2"],
                        fmt="none", color=bar_color, capsize=3, lw=1.2)

    ax_top.set_ylim(0, mu_max * 1.15)
    ax_top.set_ylabel("GMW richness (weighted mean taxa)", fontsize=FS-1)
    ax_top.set_title(
        f"GMW Depth-corrected Richness [{ref}]  |  {spec_desc}\n"
        "Light = control (g1), dark = disease (g2)  |  "
        "error bars = weighted SEM",
        fontsize=FS, fontweight="bold")
    ax_top.grid(axis="y", alpha=0.2)
    # No x-axis labels or ticks on top panel (sharex with bottom panel)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    from matplotlib.patches import Patch as _P
    ax_top.legend(handles=[
        _P(facecolor="grey", alpha=0.40, edgecolor="grey",
           label="Control (g1, light)"),
        _P(facecolor="grey", alpha=0.88, edgecolor="white",
           label="Disease (g2, dark)"),
    ], fontsize=FS-2, loc="upper right")

    # Vertical separator between health conditions and NCDs
    for _b in _group_dividers(conds):
        for ax_ in (ax_top, ax_bot):
            ax_.axvline(_b, color="lightgray", ls="--", lw=1, zorder=0)

    # ── Bottom panel: signed difference ──────────────────────────────
    for ci, (_, row) in enumerate(df.iterrows()):
        diff  = row["diff"]
        pval  = row["pval"]
        star  = row["sig"]
        ci_lo = row.get("ci_lo", np.nan)
        ci_hi = row.get("ci_hi", np.nan)

        ax_bot.bar(x[ci], diff, bw * 1.6, color=bar_color,
                   alpha=0.75 if np.isfinite(pval) and pval < 0.05 else 0.35,
                   edgecolor=bar_color, lw=0.8)
        if np.isfinite(ci_lo) and np.isfinite(ci_hi):
            ax_bot.errorbar(x[ci], diff,
                            yerr=[[diff - ci_lo], [ci_hi - diff]],
                            fmt="none", color="black",
                            capsize=4, lw=1.2, capthick=1.2)
        # Same exclusion as figure5.draw_D: "untested" marks a contrast with
        # one qualifying study and no p-value, not a significance level.
        if star and star not in ("ns", "untested", "nan"):
            ylo_b, yhi_b = ax_bot.get_ylim()
            y_annot = (ci_hi if np.isfinite(ci_hi) else diff) +                       abs(yhi_b - ylo_b) * 0.04
            ax_bot.text(x[ci], y_annot, star,
                        ha="center", va="bottom",
                        fontsize=FS-1, fontweight="bold", color=bar_color)
        ess_str = f"ESS={row['ess_g1']:.0f}"
        ax_bot.text(x[ci], ax_bot.get_ylim()[0] * 0.92,
                    ess_str, ha="center", va="top",
                    fontsize=FS-2, color="#666")

    ax_bot.axhline(0, color="black", lw=0.9)
    ax_bot.set_ylabel("Richness diff.\n(control minus disease)", fontsize=FS-1)
    ax_bot.grid(axis="y", alpha=0.2)
    ax_bot.set_xticks(x)
    xtick_labels = [
        f"{row['Condition']}\n(n={row['n_g2']:,})"
        for _, row in df.iterrows()
    ]
    ax_bot.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=FS-1)

    ylo_b, yhi_b = ax_bot.get_ylim()
    ax_bot.set_ylim(ylo_b * 1.25, yhi_b * 1.35)
    ax_bot.text(0.99, 0.02,
                "Error bars: bootstrap 95% CI  |  stars: permutation p",
                transform=ax_bot.transAxes, fontsize=FS-2,
                ha="right", va="bottom", color="grey", style="italic")

    plt.tight_layout()
    out = os.path.join(s3_main, f"health_richness_{ref}_{spec_tag}.png")
    savefig_dual(fig, out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out}")

# ====================================================================
#  MAIN ENTRY POINT
# ====================================================================

def run_triplet_health_analysis(data, dm, vanish_plus_ids,
                                 blossum_plus_ids,
                                 stats,
                                 vanish_h_ids=None, blossum_h_ids=None,
                                 taxonomy_df=None):
    """
    Run health comparisons for BOTH detection thresholds:
      1. 10% breadth coverage
      2. >=300 kb mapped bases (genome-length-independent)
    """
    ref = data.ref_name
    s3_main = config.section_dir(ref, "s3", "main")
    bdf = data.cov_matrix

    ids_ni = dm.triplet_ni
    n_triplets = len(ids_ni)

    print(f"\n  Triplet-matched health analysis: {n_triplets} triplets")
    print(f"    NI={n_triplets}, Ind-Healthy={len(dm.triplet_ind_h)}, "
          f"Ind-Unhealthy={len(dm.triplet_ind_u)}")

    stats.add("health_n_triplets", "s3", "main",
              "Depth-matched triplets (NI x IndH x IndU)",
              f"n={n_triplets} per group", n1=n_triplets)

    if n_triplets < 50:
        print("    [WARN] Too few triplets for health analysis")
        return pd.DataFrame()

    # -- Build taxon lists --
    vanish_ids = sorted(m for m in (data.vanish_set or set())
                        if m in bdf.columns)
    blossum_ids = sorted(m for m in (data.blossum_set or set())
                         if m in bdf.columns)
    vanish_plus_valid = [m for m in (vanish_plus_ids or [])
                         if m in bdf.columns]
    blossum_plus_valid = [m for m in (blossum_plus_ids or [])
                          if m in bdf.columns]
    vanish_h_valid  = [m for m in (vanish_h_ids  or []) if m in bdf.columns]
    blossum_h_valid = [m for m in (blossum_h_ids or []) if m in bdf.columns]

    taxon_lists = OrderedDict()
    if vanish_ids:          taxon_lists["VANISH"]    = vanish_ids
    if blossum_ids:         taxon_lists["BloSSUM"]   = blossum_ids
    if vanish_plus_valid:   taxon_lists["VANISH+"]   = vanish_plus_valid
    if blossum_plus_valid:  taxon_lists["BloSSUM+"]  = blossum_plus_valid
    if vanish_h_valid:      taxon_lists["VanishH+"]  = vanish_h_valid
    if blossum_h_valid:     taxon_lists["BlossumH+"] = blossum_h_valid

    # -- Run for both detection specs --
    health_specs = [
        DetectionSpec("breadth_pct", config.BREADTH_THRESHOLD),
        DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP),
    ]

    all_triplet_results = []
    all_cmh_results = []
    _prevdist_results = {}   # spec_tag -> {(cond, list): (E, p)} for HMP
    _curve_summaries = {}    # spec_tag -> curve summary, for the mean-z HMP

    for spec in health_specs:
        spec_tag = spec.short_label
        spec_desc = spec.description

        print(f"\n  {'=' * 55}")
        print(f"  Health analysis: {spec_desc}")
        print(f"  {'=' * 55}")

        # 3A. Triplet-based (Fisher)
        print(f"\n  3A. Triplet-based Fisher ({spec_desc})")
        triplet_df = _run_triplet_analysis(
            data, dm, taxon_lists, stats, spec, s3_main)
        if triplet_df is not None and not triplet_df.empty:
            all_triplet_results.append(triplet_df)

        # 3B. Per-condition within-study CMH
        print(f"\n  3B. Per-condition CMH ({spec_desc})")
        cmh_df = _run_per_condition_cmh(
            data, taxon_lists, stats, spec, s3_main)
        if cmh_df is not None and not cmh_df.empty:
            all_cmh_results.append(cmh_df)
            # Which contrasts every figure below may show. From the data, by
            # the >=CMH_MIN_STRATUM / >=MIN_STRATA_TO_PLOT rule, not by name.
            _note_eligible_conditions(cmh_df)

            # 3B-bis. Per-taxon directional data (no significance threshold)
            print(f"\n  3B-bis. Per-taxon GMW directional analysis ({spec_desc})")
            per_taxon_df = _run_per_taxon_gmw(
                data, taxon_lists, spec, s3_main)

            # 3B-ter. The corrected shift and the prevalence background.
            # Under HEALTH_BG_METHOD = "iw" neither runs, and every figure
            # below falls back to the pooled prev_diff it was published with.
            strat_shifts, curve = {}, {}
            if _bg_method() != "iw":
                print(f"\n  3B-ter. Study-stratified shifts ({spec_desc})")
                strat_shifts = _run_stratified_shifts(data, spec, s3_main)
                per_taxon_df = _attach_stratified(
                    per_taxon_df, strat_shifts, s3_main, spec_tag)
                print(f"\n  3B-qua. Prevalence background curve ({spec_desc})")
                curve = _run_background_curve(
                    strat_shifts, taxon_lists, s3_main, spec_tag)
                # Kept so the primary statistic can be combined across
                # detection specs once every spec has run.
                if curve and curve.get("summary") is not None \
                        and not curve["summary"].empty:
                    _curve_summaries[spec_tag] = curve["summary"].copy()
                _run_clade_contrasts(curve, taxon_lists, data, s3_main,
                                     spec_tag, ref=ref, spec_desc=spec_desc)

            # 3C. Binomial tests
            print(f"\n  3C. Binomial significance tests ({spec_desc})")
            binom_df = _run_binomial_tests(cmh_df, stats, spec_tag, s3_main)

            # 3C-bis. VANISH-vs-BloSSUM list contrast tests
            print(f"\n  3C-bis. VANISH-vs-BloSSUM contingency tests ({spec_desc})")
            _run_list_contrast_tests(cmh_df, stats, spec_tag, s3_main)

            # 3D. Summary figures
            print(f"\n  3D. Summary figures ({spec_desc})")

            # Primary bidirectional bar chart (GMW-CMH, with binom stars)
            _plot_summary_figure(cmh_df, ref, s3_main,
                                 spec_tag, spec_desc)

            # Multi-method tier comparison
            _plot_tier_comparison(cmh_df, ref, s3_main,
                                  spec_tag, spec_desc,
                                  binom_df=binom_df)

            # Directional prevalence (no significance cutoff — handles small n)
            _plot_directional_prevalence(per_taxon_df, cmh_df, binom_df,
                                         ref, s3_main, spec_tag, spec_desc)

            # Condition x list heatmap
            _plot_condition_heatmap(cmh_df, binom_df, ref, s3_main,
                                    spec_tag, spec_desc,
                                    per_taxon_df=per_taxon_df)

            # H+ vs + comparison
            _plot_hplus_comparison(cmh_df, binom_df, per_taxon_df,
                                   ref, s3_main, spec_tag, spec_desc)

            # Method robustness forest (VANISH + BloSSUM across all 4 methods)
            _plot_method_forest(cmh_df, binom_df, ref, s3_main,
                                spec_tag, spec_desc)

            # ESS diagnostic
            _plot_ess_diagnostic(cmh_df, ref, s3_main, spec_tag, spec_desc)

            # GMW depth-corrected richness comparison
            print(f"\n  3D-rich. GMW richness comparison ({spec_desc})")
            richness_df = _run_richness_comparison(
                data, spec, s3_main, stats)
            _plot_richness_comparison(
                richness_df, ref, s3_main, spec_tag, spec_desc)

            # Richness decomposition by taxon set
            print(f"\n  3D-rich-decomp. Richness decomposition ({spec_desc})")
            _run_richness_decomposition(
                data, spec, s3_main,
                vanish_h_ids=vanish_h_valid,
                blossum_h_ids=blossum_h_valid)

            # Prevalence-matched background comparison (all plots via single wrapper)
            print(f"\n  3D-prevmatch. Prevalence-matched background ({spec_desc})")
            bg_df = _run_prevalence_matched_background(
                data, taxon_lists, per_taxon_df, spec, s3_main,
                strat=strat_shifts, curve=curve)
            run_prevmatch_plots(
                data, taxon_lists, per_taxon_df, bg_df, cmh_df,
                spec, stats, s3_main,
                taxonomy_df=taxonomy_df, curve=curve)
            _prevdist_results[spec_tag] = _compute_energy_stats(
                per_taxon_df, bg_df,
                focus_lists=["VanishH+", "BlossumH+"])

            # ── Plugin: Table S6A/S6B (energy-distance-first summary) ──
            # Rebuilds the health directional summary so energy distance is
            # the PRIMARY column (matching Note 6.4 and the prevmatch figure).
            # Writes table_s6a_unadjusted_<spec>.csv (transparency) and
            # table_s6b_energy_first_<spec>.csv (primary cited table).
            try:
                from .table_s7 import write_table_s7
                # Use full energy stats (all lists, not just VanishH+/BlossumH+)
                _energy_all = _compute_energy_stats(per_taxon_df, bg_df)
                write_table_s7(
                    per_taxon_df, bg_df, cmh_df,
                    _energy_all, s3_main, spec_tag, stats,
                    curve_stats=_curve_stats(curve))
            except Exception as _e:
                print(f"    [WARN] Table S7 writer failed: {_e}")

            # Summary table
            _print_summary_table(cmh_df, spec_tag)

    # Triplet health richness (NI x IndH x IndU, GMW depth-corrected).
    # Once, with every detection spec: the figure puts one threshold per
    # column, so a per-spec call would leave it with a single column and its
    # data object holding whichever spec ran last.
    print(f"\n  3D-trip-rich. Triplet richness ({len(health_specs)} detection specs)")
    _run_health_richness_gmw(data, dm, list(health_specs), s3_main, stats)

    # -- 3G. Demographic strata, through the same machinery as 5D/5E --
    if getattr(config, "RUN_DEMOGRAPHICS", True):
        run_demographic_associations(data, taxon_lists, list(health_specs),
                                     s3_main, stats)

    # -- 3D-curve-meta. The PRIMARY statistic combined across specs --
    if len(_curve_summaries) > 1:
        print(f"\n  3D-curve-meta. HMP over mean z across "
              f"{len(_curve_summaries)} detection specs")
        _combine_curve_specs(_curve_summaries, s3_main)
    elif _curve_summaries:
        print(f"\n  3D-curve-meta. Only one detection spec produced a "
              f"curve — no combination written")

    # -- 3D-methods. Table S12: every list-level p-value side by side --
    if len(_curve_summaries) > 1:
        try:
            from .table_methods import build_method_comparison
            print(f"\n  3D-methods. Table S12 method comparison")
            build_method_comparison(s3_main, ref=ref,
                                    specs=tuple(sorted(_curve_summaries)))
        except Exception as _e:
            print(f"    [WARN] Table S12 writer failed: {_e}")

    # -- 3D-prevmatch-meta. HMP over the energy distance (descriptive) --
    if len(_prevdist_results) > 1:
        print(f"\n  3D-prevmatch-meta. HMP meta-statistic across specs")
        _plot_prevdist_meta(
            _prevdist_results, ref, s3_main,
            focus_lists=["VanishH+", "BlossumH+"])

    # -- 3E. Filtered PCoA (VANISH/BloSSUM taxa only) --
    print(f"\n  3E. Filtered PCoA (VANISH/BloSSUM taxa only)")
    _run_filtered_pcoa(data, dm, taxon_lists, stats, s3_main)

    # -- Combined CSVs --
    combined_triplet = pd.DataFrame()
    if all_triplet_results:
        combined_triplet = pd.concat(all_triplet_results,
                                      ignore_index=True)
        combined_triplet.to_csv(
            os.path.join(s3_main,
                         "health_triplet_results_combined.csv"),
            index=False)

    combined_cmh = pd.DataFrame()
    if all_cmh_results:
        combined_cmh = pd.concat(all_cmh_results, ignore_index=True)
        combined_cmh.to_csv(
            os.path.join(s3_main,
                         "health_per_condition_cmh_combined.csv"),
            index=False)

    return combined_cmh if not combined_cmh.empty else combined_triplet


# ====================================================================
#  ENTRY POINT
# ====================================================================

def run_prevmatch_plots(data, taxon_lists, per_taxon_df, bg_df, cmh_df,
                        spec, stats, s3_main, taxonomy_df=None, curve=None):
    """
    Single entry point for all prevalence-matched background plots.

    Called both from run_triplet_health_analysis (inline) and from the
    --only prevmatch fast-path in the pipeline runner.  Keeps both paths
    in sync automatically.

    Parameters
    ----------
    data         : ReferenceData
    taxon_lists  : dict  list_name -> [taxon_id, ...]
    per_taxon_df : DataFrame from _run_per_taxon_gmw
    bg_df        : DataFrame from _run_prevalence_matched_background
                   (or None — will be recomputed if absent)
    cmh_df       : DataFrame from _run_per_condition_cmh  (for n_g2 labels)
    spec         : DetectionSpec
    stats        : Stats  (unused here but kept for API consistency)
    s3_main      : str  output directory
    taxonomy_df  : optional DataFrame with GTDB taxonomy for taxon labelling
    curve        : dict from _run_background_curve (taxa / summary / grid /
                   frames). Carries the corrected list-level statistic, so the
                   figures print the mean z and its bootstrap CI rather than an
                   energy-distance star. Recomputed here when absent, which is
                   what the --only prevmatch fast-path needs.
    """
    ref      = data.ref_name
    spec_tag = spec.short_label
    spec_desc = spec.description

    # Also reached by --only prevmatch, which runs no CMH step, so the
    # eligibility rule has to be read here too. Without it the fast path
    # would draw contrasts the full run excludes.
    if cmh_df is not None and not getattr(cmh_df, "empty", True):
        _note_eligible_conditions(cmh_df)

    if per_taxon_df is None or per_taxon_df.empty:
        print(f"    [run_prevmatch_plots] per_taxon_df empty — skipping")
        return pd.DataFrame()

    if not curve and _bg_method() != "iw":
        print(f"    [run_prevmatch_plots] curve absent — recomputing")
        _strat = _run_stratified_shifts(data, spec, s3_main)
        per_taxon_df = _attach_stratified(per_taxon_df, _strat, s3_main,
                                          spec_tag)
        curve = _run_background_curve(_strat, taxon_lists, s3_main, spec_tag)
        if bg_df is None or bg_df.empty:
            bg_df = _run_prevalence_matched_background(
                data, taxon_lists, per_taxon_df, spec, s3_main,
                strat=_strat, curve=curve)

    if bg_df is None or bg_df.empty:
        print(f"    [run_prevmatch_plots] bg_df absent — recomputing")
        bg_df = _run_prevalence_matched_background(
            data, taxon_lists, per_taxon_df, spec, s3_main, curve=curve)

    if bg_df is None or bg_df.empty:
        print(f"    [run_prevmatch_plots] bg_df still empty — skipping plots")
        return bg_df

    _plot_prevalence_matched(
        per_taxon_df, bg_df, cmh_df,
        ref, s3_main, spec_tag, spec_desc)

    _plot_prevmatched_excess(
        per_taxon_df, bg_df, cmh_df,
        ref, s3_main, spec_tag, spec_desc)

    precomputed = _compute_energy_stats(
        per_taxon_df, bg_df,
        focus_lists=["VanishH+", "BlossumH+"])

    # The density form of this panel is gone. Main-text Figure 5E and the
    # supplement's S25 both draw the background curve now, from
    # _plot_background_curve, so drawing it a third way here would be the
    # duplicate that let the two drift apart in the first place.

    _plot_prevmatch_null_check(
        per_taxon_df, bg_df,
        ref, s3_main, spec_tag, spec_desc,
        focus_lists=["VanishH+", "BlossumH+"])

    # Taxon driver identification (Pass 1 + Pass 2)
    run_taxon_driver_analysis(
        data, taxon_lists, per_taxon_df, bg_df,
        spec, s3_main,
        taxonomy_df=taxonomy_df,
        focus_lists=["VanishH+", "BlossumH+"],
        conditions=_plotted_conditions(cmh_df), curve=curve)

    # ── Supplement: repeat prevdist analysis for original VANISH / BloSSUM ──
    # Saves to results/<ref>/section3_health/supplement/ so the original-taxa
    # plots are available without overwriting the main VanishH+/BlossumH+ ones.
    _supp_dir = os.path.join(os.path.dirname(s3_main), "supplement")
    os.makedirs(_supp_dir, exist_ok=True)
    print(f"\n    [prevmatch] Supplement: VANISH/BloSSUM analysis -> {_supp_dir}")

    _orig_focus = ["VANISH", "BloSSUM"]

    _precomputed_orig = _compute_energy_stats(
        per_taxon_df, bg_df,
        focus_lists=_orig_focus)

    _plot_prevmatch_null_check(
        per_taxon_df, bg_df,
        ref, _supp_dir, spec_tag, spec_desc,
        focus_lists=_orig_focus)

    run_taxon_driver_analysis(
        data, taxon_lists, per_taxon_df, bg_df,
        spec, _supp_dir,
        taxonomy_df=taxonomy_df,
        focus_lists=_orig_focus,
        conditions=_plotted_conditions(cmh_df), curve=curve)

    return bg_df


def run_section3(data, dm, vanish_plus_ids, blossum_plus_ids,
                 stats,
                 vanish_h_ids=None, blossum_h_ids=None,
                 all_specs=None, main_specs=None,
                 taxonomy_df=None):
    """Run all Section 3 analyses."""
    ref = data.ref_name
    print(f"\n{'=' * 60}")
    print(f"  Section 3: Health Associations [{ref}]")
    print(f"{'=' * 60}")

    if dm is None or not dm.triplet_ni:
        print("  [WARN] No triplet-matched samples -- skipping Section 3")
        return pd.DataFrame()

    combined = run_triplet_health_analysis(
        data, dm, vanish_plus_ids, blossum_plus_ids,
        stats,
        vanish_h_ids=vanish_h_ids, blossum_h_ids=blossum_h_ids,
        taxonomy_df=taxonomy_df)

    print(f"\n  Section 3 complete [{ref}]")
    return combined
