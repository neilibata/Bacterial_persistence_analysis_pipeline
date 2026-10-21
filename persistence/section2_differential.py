"""
Section 2: Differing sets of low-prevalence microbes in industrial and
           hunter-gatherer populations.

KEY CHANGE (v2): Explicit 1:1 depth-matched differential analysis
-----------------------------------------------------------------
The differential prevalence (Fisher's exact) and abundance (Mann-Whitney
CLR) tests now run exclusively on the 1:1 depth-matched sample set
(Industrial ↔ Non-Industrial).  This ensures that the comparisons are
not biased by the large number of shallow Industrial samples that have
no non-Industrial counterpart.

Main-text outputs
-----------------
  - Fig: volcano_prevalence_7col        — 7-colour prevalence volcano
  - Fig: vanish_detection_with_pooling  — VANISH taxa detection
  - Stats: all inline statistics
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, fisher_exact

from . import config
from .common import classify_group, classify_macro_group
from .detection import DetectionSpec, build_presence


def _neg_log10_bonferroni(log10_pval, n_tests):
    """Compute -log10(min(1, p * n_tests)) from log10(p)."""
    return np.maximum(-log10_pval - np.log10(n_tests), 0.0)


# ════════════════════════════════════════════════════════════════════
#  2A. DIFFERENTIAL + VOLCANO PLOTS  (on depth-matched samples)
# ════════════════════════════════════════════════════════════════════

def run_differential_with_stats(data, dm, stats, main_specs=None):
    """
    Run differential analysis on depth-matched and GMW-weighted samples.

    Three passes:
      1. Hard depth-matched Ind vs NI  (volcano plots + original VANISH+/BloSSUM+)
      2. GMW-weighted full-cohort Ind (all) vs NI  -> VANISH+/BloSSUM+
      3. GMW-weighted full-cohort Ind (healthy) vs NI  -> VanishH+/BlossumH+

    Returns
    -------
    prev_df, abund_df, vanish_plus_ids, blossum_plus_ids,
    vanish_h_ids, blossum_h_ids
    """
    ref = data.ref_name
    s2_main = config.section_dir(ref, "s2", "main")
    s2_supp = config.section_dir(ref, "s2", "supplement")

    from .differential import run_differential, run_differential_gmw

    # -- Pass 1: hard depth-matched (volcano figures, unchanged) --
    matched_idx = dm.all_matched_ind_vs_ni if dm else list(data.meta.index)
    n_ind_matched = len(dm.ind_vs_ni_ind) if dm else 0
    n_ni_matched  = len(dm.ind_vs_ni_ni)  if dm else 0

    print(f"    [Pass 1] Hard depth-matched differential: "
          f"Industrial={n_ind_matched}, Non-Industrial={n_ni_matched}, "
          f"total={len(matched_idx)}")

    stats.add("differential_n_matched", "s2", "main",
              "Depth-matched samples used for differential analysis",
              f"Industrial={n_ind_matched}, NI={n_ni_matched}",
              n1=n_ind_matched, n2=n_ni_matched)

    prev_df, abund_df, vp_hard, bp_hard = run_differential(data, matched_idx)

    if prev_df is not None and not prev_df.empty:
        n_sig = (prev_df["neg_log10_p_bonf"] > -np.log10(0.05)).sum()
        n_ni_enriched  = ((prev_df["prevalence_diff"] < 0) &
                          (prev_df["neg_log10_p_bonf"] > -np.log10(0.05))).sum()
        n_ind_enriched = ((prev_df["prevalence_diff"] > 0) &
                          (prev_df["neg_log10_p_bonf"] > -np.log10(0.05))).sum()
        stats.add("n_sig_differential_prevalence", "s2", "main",
                  "SGBs with significant prevalence difference (Bonferroni p<0.05)",
                  n_sig)
        stats.add("n_ni_enriched_prevalence", "s2", "main",
                  "SGBs enriched in Non-Industrial (depth-matched)", n_ni_enriched)
        stats.add("n_ind_enriched_prevalence", "s2", "main",
                  "SGBs enriched in Industrial (depth-matched)", n_ind_enriched)

    # -- Pass 2: GMW full-cohort Ind (all) vs NI -> VANISH+/BloSSUM+ --
    print("\n    [Pass 2] GMW-weighted differential: Ind (all) vs NI")
    meta = data.meta
    ni_mask  = meta["Lifestyle"].isin(["H-G", "Transitional"])
    ind_mask = meta["Lifestyle"] == "Industrial"

    _, blossum_plus_ids, vanish_plus_ids = run_differential_gmw(
        data, ind_mask, ni_mask,
        g1_label="Industrial (all)", g2_label="Non-Industrial",
        out_suffix="ind_all_vs_ni")

    stats.add("n_vanish_plus_gmw", "s2", "main",
              "VANISH+ taxa (GMW Ind-all vs NI)", len(vanish_plus_ids))
    stats.add("n_blossum_plus_gmw", "s2", "main",
              "BloSSUM+ taxa (GMW Ind-all vs NI)", len(blossum_plus_ids))

    # -- Pass 3: GMW Ind (healthy) vs NI -> VanishH+/BlossumH+ --
    print("\n    [Pass 3] GMW-weighted differential: Ind (healthy) vs NI")
    ind_h_mask = ind_mask & (meta["Unhealthy"] == "Healthy")
    print(f"    Ind-Healthy n={int(ind_h_mask.sum())}, NI n={int(ni_mask.sum())}")

    _, blossum_h_ids, vanish_h_ids = run_differential_gmw(
        data, ind_h_mask, ni_mask,
        g1_label="Industrial (healthy)", g2_label="Non-Industrial",
        out_suffix="ind_healthy_vs_ni")

    stats.add("n_vanish_h_plus_gmw", "s2", "main",
              "VanishH+ taxa (GMW Ind-healthy vs NI)", len(vanish_h_ids))
    stats.add("n_blossum_h_plus_gmw", "s2", "main",
              "BlossumH+ taxa (GMW Ind-healthy vs NI)", len(blossum_h_ids))

    _log_list_overlap(data, vanish_plus_ids, blossum_plus_ids,
                      vanish_h_ids, blossum_h_ids, stats)

    # Save canonical list files so --only s3 can load them without s2 re-running
    _save_list(vanish_plus_ids,  os.path.join(s2_main, "vanish_plus_ids.txt"))
    _save_list(blossum_plus_ids, os.path.join(s2_main, "blossum_plus_ids.txt"))
    _save_list(vanish_h_ids,     os.path.join(s2_main, "vanishH_ids.txt"))
    _save_list(blossum_h_ids,    os.path.join(s2_main, "blossumH_ids.txt"))
    print(f"    Saved taxon lists: vanish_plus={len(vanish_plus_ids)}, "
          f"blossum_plus={len(blossum_plus_ids)}, "
          f"vanishH={len(vanish_h_ids)}, blossumH={len(blossum_h_ids)}")

    _rewrite_differential_membership_columns(
        ref, prev_df, abund_df,
        top_ni_ids=vp_hard,
        top_ind_ids=bp_hard,
        vanish_plus_ids=vanish_plus_ids,
        blossum_plus_ids=blossum_plus_ids,
        vanish_h_ids=vanish_h_ids,
        blossum_h_ids=blossum_h_ids)

    _copy_figures_to_sections(ref, s2_main, s2_supp)

    return (prev_df, abund_df,
            vanish_plus_ids, blossum_plus_ids,
            vanish_h_ids, blossum_h_ids)


def _rewrite_differential_membership_columns(
        ref, prev_df, abund_df, *,
        top_ni_ids, top_ind_ids,
        vanish_plus_ids, blossum_plus_ids,
        vanish_h_ids, blossum_h_ids):
    """Annotate differential CSVs with actual Section 2 list memberships.

    The volcano plot category is computed during the hard depth-matched pass,
    before the GMW-derived VANISH+/BloSSUM+/VanishH+/BlossumH+ lists exist.
    Preserve that value as ``volcano_category_7`` and make ``category_7`` the
    complete list-membership category used by downstream metadata exports.
    """
    out = config.outdir(ref)
    labels_and_ids = [
        ("Top100 NI-enriched", top_ni_ids),
        ("Top100 Industrial-enriched", top_ind_ids),
        ("VANISH+", vanish_plus_ids),
        ("BloSSUM+", blossum_plus_ids),
        ("VanishH+", vanish_h_ids),
        ("BlossumH+", blossum_h_ids),
    ]
    membership = {}
    for label, ids in labels_and_ids:
        for taxon in ids or []:
            membership.setdefault(taxon, []).append(label)

    def annotate(df):
        if df is None or df.empty or "microbe" not in df.columns:
            return df
        out_df = df.copy()
        if "category_7" in out_df.columns and "volcano_category_7" not in out_df.columns:
            out_df["volcano_category_7"] = out_df["category_7"]
        out_df["list_memberships"] = out_df["microbe"].map(
            lambda m: "; ".join(membership.get(m, [])))
        out_df["category_7"] = out_df["list_memberships"].where(
            out_df["list_memberships"].ne(""), "No list")
        return out_df

    prev_annotated = annotate(prev_df)
    if prev_annotated is not None and not prev_annotated.empty:
        prev_annotated.to_csv(os.path.join(out, "prevalence_ind_vs_nonind.csv"),
                              index=False)

    abund_annotated = annotate(abund_df)
    if abund_annotated is not None and not abund_annotated.empty:
        abund_annotated.to_csv(os.path.join(out, "abundance_ind_vs_nonind.csv"),
                               index=False)


def _log_list_overlap(data, vanish_plus_ids, blossum_plus_ids,
                       vanish_h_ids, blossum_h_ids, stats):
    """Log overlap between original VANISH/BloSSUM and GMW-derived lists."""
    vanish  = data.vanish_set
    blossum = data.blossum_set
    vp_set  = set(vanish_plus_ids)  if vanish_plus_ids  else set()
    bp_set  = set(blossum_plus_ids) if blossum_plus_ids else set()
    vh_set  = set(vanish_h_ids)     if vanish_h_ids     else set()
    bh_set  = set(blossum_h_ids)    if blossum_h_ids    else set()

    stats.add("vanish_overlap", "s2", "main",
              "VANISH  &  VANISH+", len(vanish & vp_set),
              extra={"VANISH_only":  len(vanish - vp_set),
                     "VANISH+_only": len(vp_set - vanish)})
    stats.add("blossum_overlap", "s2", "main",
              "BloSSUM  &  BloSSUM+", len(blossum & bp_set),
              extra={"BloSSUM_only":  len(blossum - bp_set),
                     "BloSSUM+_only": len(bp_set - blossum)})
    stats.add("vanish_h_overlap", "s2", "main",
              "VANISH  &  VanishH+", len(vanish & vh_set),
              extra={"VanishH+_only": len(vh_set - vanish),
                     "VANISH+_vs_VanishH+_overlap": len(vp_set & vh_set)})
    stats.add("blossum_h_overlap", "s2", "main",
              "BloSSUM  &  BlossumH+", len(blossum & bh_set),
              extra={"BlossumH+_only": len(bh_set - blossum),
                     "BloSSUM+_vs_BlossumH+_overlap": len(bp_set & bh_set)})


def _copy_figures_to_sections(ref, s2_main, s2_supp):
    """Copy volcano figures to section directories."""
    import shutil
    out = config.outdir(ref)
    for fn in ["volcano_prevalence_7col.png", "prevalence_ind_vs_nonind.csv",
               "top100_NI_enriched_prevalence.txt",
               "top100_Ind_enriched_prevalence.txt"]:
        src = os.path.join(out, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(s2_main, fn))
    for fn in ["volcano_abundance_7col.png", "volcano_prevalence.png",
               "volcano_abundance.png", "abundance_ind_vs_nonind.csv"]:
        src = os.path.join(out, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(s2_supp, fn))


# ════════════════════════════════════════════════════════════════════
#  2B. VANISH / BloSSUM DETECTION WITH POOLING  (unchanged logic)
# ════════════════════════════════════════════════════════════════════

def compute_detection_with_pooling(data, stats, spec=None):
    """Detection-with-pooling analysis (uses all samples, not matched)."""
    ref = data.ref_name
    s2_main = config.section_dir(ref, "s2", "main")

    if spec is None:
        spec = DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)
    spec_tag = spec.short_label

    meta = data.meta
    breadth = data.breadth
    vanish = data.vanish_set
    blossum = data.blossum_set

    presence = build_presence(breadth, data.depth, data.lengths, spec=spec)
    rng = np.random.default_rng(42)
    n_boot = 50

    # VANISH in Industrial
    ind_mask = meta["Lifestyle"] == "Industrial"
    ind_presence = presence.loc[ind_mask]
    vanish_in_ref = sorted(v for v in vanish if v in breadth.columns)

    if vanish_in_ref and ind_mask.sum() > 0:
        vanish_pooling = _pooling_curves_presence(
            ind_presence, vanish_in_ref, rng, n_boot,
            label="VANISH in Industrial")
        vanish_pooling.to_csv(
            os.path.join(s2_main,
                         f"vanish_detection_pooling_industrial_{spec_tag}.csv"),
            index=False)

        med_n_vanish = vanish_pooling.groupby("taxon")["n_needed"].median()
        if len(med_n_vanish) > 0:
            stats.add(f"vanish_median_n_needed_industrial_{spec_tag}",
                      "s2", "main",
                      f"Median # Industrial samples to detect VANISH taxon",
                      f"{med_n_vanish.median():.0f}")

    # BloSSUM in Non-Industrial
    ni_mask = meta["Lifestyle"].isin(["H-G", "Transitional"])
    ni_presence = presence.loc[ni_mask]
    blossum_in_ref = sorted(b for b in blossum if b in breadth.columns)

    if blossum_in_ref and ni_mask.sum() > 0:
        blossum_pooling = _pooling_curves_presence(
            ni_presence, blossum_in_ref, rng, n_boot,
            label="BloSSUM in Non-Industrial")
        blossum_pooling.to_csv(
            os.path.join(s2_main,
                         f"blossum_detection_pooling_ni_{spec_tag}.csv"),
            index=False)


def _pooling_curves_presence(presence_df, taxon_list, rng, n_boot, label=""):
    """Compute N-samples-to-detect for each taxon via bootstrap pooling."""
    n_total = presence_df.shape[0]
    rows = []
    pool_sizes = sorted(set(
        [int(round(x)) for x in np.logspace(
            0, np.log10(max(n_total, 2)), min(30, n_total))] + [n_total]))

    for taxon in taxon_list:
        if taxon not in presence_df.columns:
            continue
        prevalence = presence_df[taxon].mean()
        if prevalence == 0:
            rows.append({"taxon": taxon, "n_needed": n_total,
                         "prevalence": 0.0})
            continue

        # Find smallest pool size where detection probability > 50%
        n_needed = n_total
        for ps in pool_sizes:
            if ps > n_total:
                break
            # P(detect in pool of ps) = 1 - (1-prev)^ps
            p_detect = 1.0 - (1.0 - prevalence) ** ps
            if p_detect >= 0.5:
                n_needed = ps
                break
        rows.append({"taxon": taxon, "n_needed": n_needed,
                     "prevalence": prevalence})

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
#  2C. HADZA-SPECIFIC SGBs
# ════════════════════════════════════════════════════════════════════

def compute_hadza_specific(data, stats, taxonomy_df=None):
    """Identify SGBs significantly enriched in Hadza vs other Non-Industrial.

    Uses per-SGB Fisher's exact tests with Bonferroni correction to
    identify SGBs whose prevalence is significantly higher in Hadza
    samples than in all other Non-Industrial samples.  Results are
    annotated with GTDB taxonomy when *taxonomy_df* (e.g. from
    Hadza_UHGG_SGB_matching.csv) is provided.

    Parameters
    ----------
    data : ReferenceData
    stats : StatsReport
    taxonomy_df : pd.DataFrame, optional
        Genome ↔ GTDB taxonomy mapping (e.g. Hadza_UHGG_SGB_matching.csv).
    """
    ref = data.ref_name
    s2_main = config.section_dir(ref, "s2", "main")
    meta = data.meta
    breadth = data.breadth
    threshold = config.BREADTH_THRESHOLD

    hadza_mask = meta["Country_norm"].str.contains("Tanzania", na=False) & \
                 meta["Lifestyle"].isin(["H-G", "Transitional"])
    other_ni_mask = meta["Lifestyle"].isin(["H-G", "Transitional"]) & ~hadza_mask

    n_hadza = hadza_mask.sum()
    n_other = other_ni_mask.sum()
    if n_hadza < 10 or n_other < 10:
        print(f"    [hadza] Insufficient samples: Hadza={n_hadza}, other={n_other}")
        return

    print(f"    Hadza n={n_hadza:,}, other NI n={n_other:,}")

    # ── Build taxonomy lookup ──
    from .section2_pooling import _build_taxonomy_map
    tax_map = _build_taxonomy_map(taxonomy_df)

    # ── Per-SGB Fisher's exact test (Hadza vs other NI) ──
    hadza_idx = meta.index[hadza_mask]
    other_idx = meta.index[other_ni_mask]
    from .differential import _fisher_two_sided_log10_pvalue, _pvalue_from_log10

    results = []
    for mic in breadth.columns:
        d_hadza = int((breadth.loc[hadza_idx, mic] >= threshold).sum())
        d_other = int((breadth.loc[other_idx, mic] >= threshold).sum())
        if d_hadza == 0 and d_other == 0:
            continue
        odds, _ = fisher_exact(
            [[d_hadza, n_hadza - d_hadza],
             [d_other, n_other - d_other]])
        log10_pval = _fisher_two_sided_log10_pvalue(
            d_hadza, n_hadza, d_other, n_other)
        prev_h = d_hadza / n_hadza
        prev_o = d_other / n_other
        results.append({
            "sgb": mic,
            "hadza_prev": prev_h,
            "other_ni_prev": prev_o,
            "prevalence_diff": prev_h - prev_o,
            "odds_ratio": odds,
            "pval": _pvalue_from_log10(log10_pval),
            "log10_pval": log10_pval,
            "n_hadza_detected": d_hadza,
            "n_other_detected": d_other,
        })

    df = pd.DataFrame(results)
    if df.empty:
        print("    [hadza] No SGBs detected in either group")
        return

    # Bonferroni correction (log-space)
    n_tests = len(df)
    df["neg_log10_p_bonf"] = _neg_log10_bonferroni(df["log10_pval"], n_tests)
    df["p_bonferroni"] = (df["pval"] * n_tests).clip(upper=1.0)

    # Annotate with GTDB taxonomy
    df["gtdb_taxonomy"] = df["sgb"].map(tax_map).fillna("")
    df["lowest_rank"] = df["gtdb_taxonomy"].apply(_extract_lowest_rank)

    # Sort by prevalence difference descending (Hadza-enriched first)
    df = df.sort_values("prevalence_diff", ascending=False).reset_index(drop=True)

    # ── Identify significantly Hadza-enriched SGBs ──
    sig_mask = (df["p_bonferroni"] < 0.05) & (df["prevalence_diff"] > 0)
    n_sig = sig_mask.sum()
    sig_df = df.loc[sig_mask].copy()

    print(f"    {n_sig} SGBs significantly enriched in Hadza "
          f"(Fisher's exact, Bonferroni p < 0.05)")

    # Save full results
    df.to_csv(os.path.join(s2_main,
              f"hadza_vs_other_ni_fisher_{ref}.csv"), index=False)

    # Save significant Hadza-enriched subset
    if not sig_df.empty:
        sig_df.to_csv(os.path.join(s2_main,
                      f"hadza_enriched_significant_{ref}.csv"), index=False)

    # ── Stats ──
    stats.add("hadza_n_samples", "s2", "main",
              "Number of Hadza samples",
              n_hadza, extra={"other_ni": n_other})

    stats.add("hadza_n_sig_enriched", "s2", "main",
              "SGBs significantly enriched in Hadza "
              "(Fisher exact, Bonferroni p < 0.05)",
              n_sig)

    stats.add("hadza_n_tests", "s2", "main",
              "Number of SGBs tested (Hadza vs other NI)", n_tests)

    if not sig_df.empty:
        # Top enriched with taxonomy
        top_lines = []
        for _, row in sig_df.head(10).iterrows():
            tag = row["lowest_rank"] or row["sgb"]
            top_lines.append(
                f"{row['sgb']} ({tag}): "
                f"Hadza {row['hadza_prev']:.1%} vs other {row['other_ni_prev']:.1%}")
        stats.add("hadza_top_enriched", "s2", "main",
                  "Top Hadza-enriched SGBs (by prevalence difference)",
                  "; ".join(top_lines))

        # Taxonomy summary of significant set
        families = sig_df["gtdb_taxonomy"].apply(_extract_rank, rank="f__")
        family_counts = families.value_counts()
        if not family_counts.empty:
            fam_str = "; ".join(
                f"{fam} (n={cnt})" for fam, cnt in family_counts.head(10).items()
                if fam)
            stats.add("hadza_enriched_families", "s2", "main",
                      "Top families among Hadza-enriched SGBs", fam_str)


def _extract_lowest_rank(tax_str):
    """Extract the lowest non-empty GTDB rank from a taxonomy string."""
    if not tax_str or tax_str == "nan" or not isinstance(tax_str, str):
        return ""
    for part in reversed(tax_str.split(";")):
        part = part.strip()
        if part and not part.endswith("__"):
            return part
    return ""


def _extract_rank(tax_str, rank="f__"):
    """Extract a specific GTDB rank (e.g. 'f__Lachnospiraceae')."""
    if not tax_str or not isinstance(tax_str, str):
        return ""
    for part in tax_str.split(";"):
        part = part.strip()
        if part.startswith(rank) and len(part) > len(rank):
            return part
    return ""


def _save_list(ids, path):
    with open(path, "w") as f:
        for i in ids:
            f.write(f"{i}\n")


# ════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def run_section2(data, dm, stats,
                 all_specs=None, main_specs=None,
                 pooling_specs=None,
                 taxonomy_df=None):
    """
    Run all Section 2 analyses.

    Parameters
    ----------
    data : ReferenceData
    dm : DepthMatchResult
    stats : StatsReport
    all_specs, main_specs : list of DetectionSpec
    pooling_specs : list of DetectionSpec, optional
        Specs used for VANISH/BloSSUM pooling & extinction analyses (2B, 2E–2G).
        Defaults to [10%, 30%, 50% breadth; 300 kb, 500 kb flat] for
        robustness across stringent thresholds.
    taxonomy_df : optional DataFrame with Genome + GTDB Taxonomy columns
                  (e.g. from Hadza_UHGG_SGB_matching)

    Returns
    -------
    vanish_plus_ids, blossum_plus_ids, vanish_h_ids, blossum_h_ids
    """
    ref = data.ref_name
    print(f"\n{'='*60}")
    print(f"  Section 2: Differential Prevalence [{ref}]")
    print(f"{'='*60}")

    if main_specs is None:
        main_specs = [DetectionSpec("breadth_pct",
                                     config.ACCUM_MAIN_TEXT_BREADTH)]

    if pooling_specs is None:
        pooling_specs = [
            DetectionSpec("breadth_pct", 10.0),
            DetectionSpec("breadth_pct", 30.0),
            DetectionSpec("breadth_pct", 50.0),
            DetectionSpec("flat_bp", 300_000),
            DetectionSpec("flat_bp", 500_000),
        ]

    # 2A. Differential on depth-matched samples + GMW passes
    print("\n  2A. Differential analysis (depth-matched + GMW)")
    (prev_df, abund_df,
     vanish_plus_ids, blossum_plus_ids,
     vanish_h_ids, blossum_h_ids) = \
        run_differential_with_stats(data, dm, stats, main_specs=main_specs)

    # 2B. Detection with pooling — original VANISH/BloSSUM only
    #     (kept for backward compat; 2E below is comprehensive)
    print("\n  2B. VANISH/BloSSUM detection with pooling")
    for spec in pooling_specs:
        print(f"\n    --- {spec.description} ---")
        compute_detection_with_pooling(data, stats, spec=spec)

    # 2C. Hadza-specific SGBs
    print("\n  2C. Hadza-specific SGBs")
    compute_hadza_specific(data, stats, taxonomy_df=taxonomy_df)

    # 2D–2G. Extended pooling, continentxlifestyle, extinction candidates
    from .section2_pooling import run_section2_pooling
    run_section2_pooling(data, vanish_plus_ids, blossum_plus_ids,
                         stats, main_specs=pooling_specs,
                         taxonomy_df=taxonomy_df)

    print(f"\n  Section 2 complete [{ref}]")
    return vanish_plus_ids, blossum_plus_ids, vanish_h_ids, blossum_h_ids
