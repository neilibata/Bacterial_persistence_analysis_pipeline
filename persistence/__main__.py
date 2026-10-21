"""
Main manuscript analysis orchestrator.

Usage:
    python -m persistence                       # default UHGG analysis
    python -m persistence --ref UHGG lev        # selected references
    python -m persistence --only s1             # Section 1 only
    python -m persistence --frac-tol 0.03       # override depth matching
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")

from . import config
from .loading import load_reference
from .stats_report import StatsReport
from .detection import (DetectionSpec, breadth_pct_specs, flat_bp_specs)

VALID_SECTIONS = {"s1", "s1_alpha", "s1_traces", "s1_within", "s1_tech", "s1_depthbin", "s1_pcoa3", "s1_axis_angle", "s1_axis_stats", "s1_figure", "s2", "s3", "s3demo", "s3_axis", "table_s1a", "host_sens", "figures",
                  "visualization", "sensitivity", "depth_bias", "carter", "accum", "prevmatch", "health_richness",
                  "gmw_richness", "figure3", "graphical-abstract"}

MAIN_TEXT_FIGURE_TASKS = {"figure1", "figure2", "figure3", "figure4", "figure5"}


def _requested_figure_tasks(args):
    """Resolve figure task groups before deciding prerequisite sections."""
    from .figure_runner import resolve_figure_tasks
    return resolve_figure_tasks(
        getattr(args, "figure", None),
        quick=getattr(args, "figure_quick", False),
    )


def _main_text_prereq_sections(tasks):
    """Map main-text figure tasks to the analysis sections they require."""
    wanted = set(tasks) & MAIN_TEXT_FIGURE_TASKS
    prereqs = {}
    if {"figure2", "figure5"} & wanted:
        # Both default to --ref combined (figure2's --ref help says UHGG
        # "disagrees with the supplementary tables"), so naming UHGG here
        # built the prerequisites for a reference neither figure reads.
        prereqs.setdefault(config.MAIN_COHORT_REF,
                           set()).update({"s1", "s2", "s3"})
    if "figure4" in wanted:
        prereqs.setdefault("combined", set()).add("s2")
    return prereqs


def _build_detection_specs(mode):
    """Build list of DetectionSpecs based on the requested mode."""
    b_specs = breadth_pct_specs(config.ACCUM_BREADTH_THRESHOLDS) if mode in ("breadth", "both") else []
    f_specs = flat_bp_specs(config.ACCUM_FLAT_THRESHOLDS_BP) if mode in ("flat", "both") else []

    all_specs = b_specs + f_specs
    main_specs = []
    if b_specs:
        main_specs.append(DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH))
    if f_specs:
        main_specs.append(DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP))

    return all_specs, main_specs


def _stage_state_path(ref_name, stage):
    return os.path.join("results", ref_name, ".pipeline_state", f"{stage}.done")


def _reference_input_mtime(ref_name):
    ref_cfg = config.REFERENCES.get(ref_name, {})
    paths = [
        ref_cfg.get("breadth_file"),
        ref_cfg.get("depth_file"),
        ref_cfg.get("lengths_file"),
        ref_cfg.get("combined_sgb_metadata_file"),
        config.META_FILE,
        config.VANISH_FILE,
        config.BLOSSUM_FILE,
    ]
    mtimes = []
    for path in paths:
        if path and os.path.exists(path):
            mtimes.append(os.path.getmtime(path))
    return max(mtimes) if mtimes else 0.0


def _path_is_current(path, input_mtime):
    return os.path.exists(path) and os.path.getmtime(path) >= input_mtime


def _stage_artifacts_exist(ref_name, artifacts):
    if not artifacts:
        return False
    return all(os.path.exists(os.path.join("results", ref_name, relpath))
               for relpath in artifacts)


def _mark_stage_done(ref_name, stage):
    path = _stage_state_path(ref_name, stage)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"completed_at_unix={time.time():.0f}\n")


def _should_skip_stage(args, ref_name, stage, label, artifacts=()):
    if not getattr(args, "resume", False):
        return False

    input_mtime = _reference_input_mtime(ref_name)
    state_path = _stage_state_path(ref_name, stage)
    if os.path.exists(state_path):
        if not _path_is_current(state_path, input_mtime):
            print(f"\n  [resume] {label}: checkpoint predates inputs; running")
            return False
        if artifacts:
            artifact_paths = [os.path.join("results", ref_name, relpath)
                              for relpath in artifacts]
            missing = [path for path in artifact_paths if not os.path.exists(path)]
            if missing:
                print(f"\n  [resume] {label}: checkpoint present but expected outputs "
                      f"are missing; running")
                for path in missing[:5]:
                    print(f"    missing: {path}")
                if len(missing) > 5:
                    print(f"    ... and {len(missing) - 5} more")
                return False
            if not all(_path_is_current(path, input_mtime)
                       for path in artifact_paths):
                print(f"\n  [resume] {label}: checkpoint present but outputs "
                      f"predate inputs; running")
                return False
        print(f"\n  [resume] {label}: checkpoint present; skipping")
        return True

    if _stage_artifacts_exist(ref_name, artifacts):
        artifact_paths = [os.path.join("results", ref_name, relpath)
                          for relpath in artifacts]
        if not all(_path_is_current(path, input_mtime)
                   for path in artifact_paths):
            print(f"\n  [resume] {label}: outputs predate inputs; running")
            return False
        print(f"\n  [resume] {label}: expected outputs present; skipping")
        _mark_stage_done(ref_name, stage)
        return True

    print(f"\n  [resume] {label}: no complete checkpoint; running")
    return False


#: Taxonomy for annotating taxa with names. Optional: every stage that takes it
#: works without it, labelling by SGB identifier instead.
TAXONOMY_CANDIDATES = (
    "Hadza_UHGG_SGB_matching.csv",
    "Hadza_UHGG_SGB_matching.tsv",
    os.path.join("..", "analysis_code", "Hadza_UHGG_SGB_matching.csv"),
    "taxonomy.csv",
    "taxonomy.tsv",
)


def _load_taxonomy():
    """First taxonomy file that exists, or None.

    One function rather than a copy per caller: the copies had drifted, and a
    stage that loaded no taxonomy labelled its output by SGB identifier while
    the same stage in a full run labelled it by name.
    """
    import pandas as pd

    for cand in TAXONOMY_CANDIDATES:
        if not os.path.exists(cand):
            continue
        try:
            df = pd.read_csv(cand, sep=None, engine="python")
        except Exception:
            return None
        print(f"  Loaded taxonomy: {cand} ({len(df)} rows)")
        return df
    return None


def run_reference(ref_name, sections=None, detection_mode="both", cli_args=None,
                  run_figures=True):
    """Run all requested section analyses for a single reference."""
    t0 = time.time()
    do_all = sections is None
    # Allow getattr(args, ...) inside the function body
    args = cli_args  # may be None; guarded by getattr(..., default)

    print(f"\n{'#'*70}")
    print(f"#  Reference: {ref_name}   Detection mode: {detection_mode}")
    print(f"{'#'*70}")

    ref_cfg = config.REFERENCES.get(ref_name)
    if ref_cfg is None:
        print(f"  Unknown reference: {ref_name}")
        return
    for key in ("breadth_file", "depth_file", "lengths_file"):
        if not os.path.exists(ref_cfg[key]):
            print(f"  Missing: {ref_cfg[key]} — skipping {ref_name}")
            return

    stats = StatsReport(ref_name)
    all_specs, main_specs = _build_detection_specs(detection_mode)
    main_tags = [s.short_label for s in main_specs]
    print(f"  Detection specs: {len(all_specs)} total, "
          f"{len(main_specs)} main-text")
    for s in all_specs:
        tag = "★" if s in main_specs else " "
        print(f"    {tag} {s.description}")

    # ── Load data ──
    data = load_reference(ref_name)

    # Publish the cohort cascade so Table S2b can be generated from the registry
    # rather than transcribed. Each reference filters independently -- notably the
    # >=30 Mb mapped-bases cut is reference-dependent, since a sample maps more
    # bases against a larger catalogue -- so the analytic N differs per reference.
    from .loading import LAST_LOAD_COUNTS as _llc
    for _k, _v in (_llc.get(ref_name) or {}).items():
        stats.add(f"cohort_{_k}", "qc", "supplement",
                  f"Cohort construction ({ref_name}): {_k.replace('_', ' ')}", _v)

    # Sections that need the full depth audit + reinstatement QC printout.
    # Lightweight sections (s1_depthbin, s1_tech, visualization, sensitivity)
    # still need reinstatement for correct cohort Ns, but skip the audit output.
    _needs_audit = (do_all or
                    {"s1", "s1_alpha", "s1_traces", "s1_within", "s2", "s3",
                     "depth_bias", "gmw_richness", "health_richness", "accum",
                     "prevmatch"} & (sections or set()))

    # ── Depth audit (pre-reinstatement QC) ──
    if _needs_audit:
        from .depth_audit import run_depth_audit
        data = run_depth_audit(data, stats)

    # ── Secondary reinstatement (always — every section needs the full cohort) ──
    from .reinstate_secondary import reinstate_distinct_secondaries
    data = reinstate_distinct_secondaries(data, stats)

    # ── Publish the analytic cohort (always) ──
    # Cohort N is canonical from here on. Figure 1 and the metadata summaries
    # cannot re-derive it from Table M1 -- both exclusions are matrix-derived and
    # reference-dependent -- so write it out for them to intersect against.
    from .cohort_manifest import write_cohort_manifest
    write_cohort_manifest(data, stats)

    # ── Post-reinstatement depth audit (canonical cohort Ns for paper) ──
    if _needs_audit:
        from .depth_audit import run_depth_audit as _run_depth_audit_post
        _run_depth_audit_post(
            data, stats,
            outdir_override="depth_audit_post_reinstatement",
            tag_prefix="post_")

    # ── Depth-dependence audit (Ruhlemann-style) — post-reinstatement ──
    if do_all or {"depth_bias"} & (sections or set()):
        artifacts = [
            os.path.join("depth_audit",
                         f"depth_dependence_audit_{tag}.csv")
            for tag in main_tags
        ]
        if not _should_skip_stage(args, ref_name, "depth_dependence_audit",
                                  "Depth-dependence audit", artifacts):
            from .depth_dependence_audit import run_depth_dependence_audit
            run_depth_dependence_audit(data, stats)
            _mark_stage_done(ref_name, "depth_dependence_audit")


    # ── Depth matching (used by all sections) ──
    dm = None
    # s3demo is here because it too may have to derive Section 2's taxon lists,
    # and that derivation is depth-matched.
    if do_all or {"s1", "s1_alpha", "s1_traces", "s1_within", "s2", "s3",
                  "s3demo"} & (sections or set()):
        from .depth_matching import build_depth_matched_sets
        tol = config.DEPTH_MATCH_FRAC_TOL
        health_tol = config.HEALTH_TRIPLET_FRAC_TOL
        print(f"\n  Depth matching (1:1 NN, frac_tol = {tol:.0%}; "
              f"triplet frac_tol = {health_tol:.0%})")
        dm = build_depth_matched_sets(
            data, frac_tol=tol,
            health_frac_tol=health_tol)
        print(dm.summary())

        # Log matching diagnostics
        for comp_name, st in dm.stats.items():
            if "n_pairs" in st:
                stats.add(f"depth_match_{comp_name}", "s1", "main",
                          f"1:1 depth matching: {comp_name}",
                          f"{st['n_pairs']} pairs "
                          f"(median frac_diff = {st['median_frac_diff']:.4f}, "
                          f"max = {st['max_frac_diff']:.4f})")
            elif "n_triplets" in st:
                stats.add(f"depth_match_{comp_name}", "s3", "main",
                          f"Triplet matching: {comp_name}",
                          f"{st['n_triplets']} triplets "
                          f"(frac_tol = {st['frac_tol']})")

    matched_idx_all = list(data.meta.index)

    # ── Anchored depth matching (1-to-K, H-G and NI anchored) ──
    am = None
    if do_all or {"s1", "s1_alpha", "s1_traces"} & (sections or set()):
        from .depth_matching_anchored import build_anchored_matched_sets
        print(f"\n  Anchored depth matching (1-to-K)")
        am = build_anchored_matched_sets(
            data,
            hg_frac_tol=config.ANCHORED_HG_FRAC_TOL,
            hg_pct_keep=config.ANCHORED_HG_PCT_KEEP,
            ni_frac_tol=config.ANCHORED_NI_FRAC_TOL,
            ni_pct_keep=config.ANCHORED_NI_PCT_KEEP,
            k_max_try=config.ANCHORED_K_MAX_TRY,
        )
        print(am.summary())

    # ═══════════════════════════════════════════════════════
    #  SECTION 1: Diversity & Composition
    # ═══════════════════════════════════════════════════════
    if do_all or {"s1", "s1_alpha", "s1_traces"} & (sections or set()):
        from .section1_diversity import run_section1
        if sections and "s1_alpha" in sections and "s1" not in sections:
            stop = "alpha"
        elif sections and "s1_traces" in sections and "s1" not in sections:
            stop = "traces"
        else:
            stop = None
        stage = "section1"
        label = "Section 1: Diversity & Composition"
        artifacts = [
            os.path.join("section1_diversity", "main",
                         f"composition_accumulation_{tag}.png")
            for tag in main_tags
        ]
        artifacts.append(os.path.join("section1_diversity", "main",
                                      "gmw_richness_summary.csv"))
        if stop:
            stage = f"section1_{stop}"
            artifacts = []
        if not _should_skip_stage(args, ref_name, stage, label, artifacts):
            run_section1(data, dm, matched_idx_all, stats,
                         all_specs=all_specs, main_specs=main_specs,
                         stop_after=stop, am=am)
            _mark_stage_done(ref_name, stage)

    # ═══════════════════════════════════════════════════════
    #  SECTION 1 (within): Within-group variation
    # ═══════════════════════════════════════════════════════
    if do_all or {"s1", "s1_within"} & (sections or set()):
        artifacts = [
            os.path.join("section1_diversity", "main",
                         f"manuscript_within_stats_{tag}.txt")
            for tag in main_tags
        ]
        artifacts.extend(
            os.path.join("section1_diversity", "main",
                         f"within_group_beta_raw_{tag}.npz")
            for tag in main_tags
        )
        artifacts.extend(
            os.path.join("section1_diversity", "main",
                         f"within_group_beta_boxplot_{tag}_notitle.png")
            for tag in main_tags
        )
        if not _should_skip_stage(args, ref_name, "section1_within",
                                  "Section 1 within-group variation",
                                  artifacts):
            from .section1_within_variation import run_within_variation
            run_within_variation(data, dm, stats)

            # ── Plugin: within-group stats (Figure 2 [stat] placeholder filler) ──
            from .within_group_stats import run_within_group_stats
            run_within_group_stats(data, dm, stats)
            _mark_stage_done(ref_name, "section1_within")

    # ═══════════════════════════════════════════════════════
    #  SECTION 1 (depth-bin CV): residual richness variability
    # ═══════════════════════════════════════════════════════
    if do_all or {"s1", "s1_within", "s1_depthbin"} & (sections or set()):
        artifacts = [
            os.path.join("section1_diversity", "main",
                         f"depth_binned_richness_cv_{tag}.csv")
            for tag in main_tags
        ]
        if not _should_skip_stage(args, ref_name, "section1_depthbin",
                                  "Section 1 depth-bin CV", artifacts):
            from .within_group_stats import run_depth_binned_richness_cv
            run_depth_binned_richness_cv(data, stats)
            _mark_stage_done(ref_name, "section1_depthbin")

    # ═══════════════════════════════════════════════════════
    #  SECTION 1 (tech): Within-sample technical replicate robustness
    # ═══════════════════════════════════════════════════════
    if do_all or {"s1", "s1_tech"} & (sections or set()):
        if not _should_skip_stage(args, ref_name, "section1_tech",
                                  "Section 1 technical replicates"):
            from .section1_tech_replicates import run_tech_replicate_robustness
            run_tech_replicate_robustness(data, stats)
            _mark_stage_done(ref_name, "section1_tech")

    # ═══════════════════════════════════════════════════════
    #  DATA-DRIVEN VANISH/BloSSUM  (replaces file-based lists for S2+S3)
    # ═══════════════════════════════════════════════════════
    if getattr(config, '_DERIVE_VB', False):
        from .vanish_blossum_definitions import (
            VBDefinition, define_vanish_blossum,
        )
        from .loading import ReferenceData

        vb_defn = VBDefinition(
            r=getattr(config, '_VB_R', 0.02),
            m=getattr(config, '_VB_M', 0.10),
            u=getattr(config, '_VB_U', 0.80),
            study_aware=getattr(config, '_VB_STUDY_AWARE', False),
            label="derived",
        )
        print(f"\n  Deriving VANISH/BloSSUM from data: {vb_defn.description}")
        vb_result = define_vanish_blossum(data, defn=vb_defn)
        print(f"    → VANISH={vb_result.n_vanish}, "
              f"BloSSUM={vb_result.n_blossum}  "
              f"(original file-based: VANISH={len(data.vanish_set)}, "
              f"BloSSUM={len(data.blossum_set)})")

        # Save diagnostics
        out = config.outdir(ref_name)
        if vb_result.prevalence_df is not None:
            vb_result.prevalence_df.to_csv(
                os.path.join(out, "derived_vb_prevalence.csv"), index=False)

        # Replace data with new V/B sets
        # _replace, not a field-by-field rebuild: the rebuild omitted
        # total_seq_bp, so --derive-vb raised TypeError here.
        data = data._replace(vanish_set=vb_result.vanish_set,
                             blossum_set=vb_result.blossum_set)

    # ═══════════════════════════════════════════════════════
    #  SECTION 2: Differential Prevalence
    # ═══════════════════════════════════════════════════════
    vanish_plus_ids, blossum_plus_ids = [], []
    vanish_h_ids, blossum_h_ids = [], []
    taxonomy_df = None

    if do_all or "s2" in sections:
        from .section2_differential import run_section2

        # Optional, for extinction-candidate annotation.
        taxonomy_df = _load_taxonomy()

        artifacts = [
            os.path.join("section2_differential", "main", "vanish_plus_ids.txt"),
            os.path.join("section2_differential", "main", "blossum_plus_ids.txt"),
            os.path.join("section2_differential", "main", "vanishH_ids.txt"),
            os.path.join("section2_differential", "main", "blossumH_ids.txt"),
        ] + [
            os.path.join("section2_differential", "main",
                         f"s2d_list_overlap_{tag}.png")
            for tag in main_tags
        ]
        if not _should_skip_stage(args, ref_name, "section2",
                                  "Section 2: Differential Prevalence",
                                  artifacts):
            if getattr(args, "bayesian", False):
                from .section2_bayesian import run_section2_bayesian
                vanish_plus_ids, blossum_plus_ids, vanish_h_ids, blossum_h_ids = \
                    run_section2_bayesian(
                        data, dm, stats,
                        all_specs=all_specs, main_specs=main_specs,
                        taxonomy_df=taxonomy_df,
                        null_key=getattr(args, "bayes_null", "10pct"),
                        ra_module_path=getattr(args, "ra_module_path", None),
                    )
            else:
                vanish_plus_ids, blossum_plus_ids, vanish_h_ids, blossum_h_ids = \
                    run_section2(data, dm, stats,
                                 all_specs=all_specs, main_specs=main_specs,
                                 taxonomy_df=taxonomy_df)

            # Supplementary Tables S4/S5 and Figures S5A/S5B.
            #
            # This used to be reachable only through run_section2_bayesian,
            # which needs an opt-in --bayesian run, so a default --all
            # --figure full produced no fig_s5b_overlap_analysis at all --
            # a shipped supplementary figure with no producer in the normal
            # pipeline. It cannot be a figure_runner task either: those are
            # standalone figures/*.py subprocesses, and run_supp_s5 needs the
            # loaded ReferenceData and the taxon lists, which only exist
            # here. Failures are caught: this is supplementary output and
            # should not abort a reference that has otherwise succeeded.
            try:
                from .supp_s5 import run_supp_s5 as _run_supp_s5
                _run_supp_s5(data, vanish_plus_ids, blossum_plus_ids,
                             vanish_h_ids, blossum_h_ids,
                             taxonomy_df=taxonomy_df)
            except Exception as _exc:
                print(f"    [supp_s5] skipped: {_exc}")

            _mark_stage_done(ref_name, "section2")

    # ═══════════════════════════════════════════════════════
    #  HOST SENSITIVITY: repeat samples of one host (Supp. Text #5, #7C)
    #  Run: python -m persistence --ref combined --only host_sens
    # ═══════════════════════════════════════════════════════
    # After section 2, so it can confirm it rebuilds the lists section 2 wrote.
    # The chimpanzee reference is a negative control with no lifestyle
    # contrast to test.
    if (do_all or "host_sens" in (sections or set())) and ref_name != "chimp":
        if not _should_skip_stage(args, ref_name, "host_sensitivity",
                                  "Host sensitivity (richness ratio, list membership)",
                                  [os.path.join("host_sensitivity", f) for f in (
                                      "richness_host_sensitivity.csv",
                                      "list_host_sensitivity_summary.csv")]):
            from .host_sensitivity import run_host_sensitivity
            run_host_sensitivity(data, stats)
            _mark_stage_done(ref_name, "host_sensitivity")

    # ── Section 2's taxon lists are a prerequisite of Section 3 ───────────────
    # With --only s3, Section 2 is skipped and the ID lists stay empty. Section
    # 3 is defined on those lists, so they are an input it has to have: reuse
    # the ones a previous Section 2 wrote, and otherwise derive them here with
    # the same call --all makes, on the matrices loaded in this tree.
    #
    # Not from metadata/lists_as_analysed/: that snapshot is the membership the
    # manuscript reports, for comparison, and reading it here would report
    # Section 3 over a list that was never derived from these inputs.
    if ((do_all or (sections and {"s3", "s3demo"} & sections)) and
            not any([vanish_plus_ids, blossum_plus_ids,
                     vanish_h_ids, blossum_h_ids])):
        _s2_main = os.path.join("results", ref_name,
                                 "section2_differential", "main")
        _LIST_FILES = ("vanish_plus_ids.txt", "blossum_plus_ids.txt",
                       "vanishH_ids.txt", "blossumH_ids.txt")

        def _read_ids(fname):
            with open(os.path.join(_s2_main, fname)) as _f:
                ids = [l.strip() for l in _f if l.strip()]
            # Filter to taxa present in the loaded breadth matrix
            valid = set(data.breadth.columns)
            return [t for t in ids if t in valid]

        # Section 3 takes the taxonomy as well, and it was loaded only inside
        # the Section 2 branch -- so --only s3 labelled taxa by SGB identifier
        # where the same stage in a full run labelled them by name.
        if taxonomy_df is None:
            taxonomy_df = _load_taxonomy()

        if all(os.path.exists(os.path.join(_s2_main, f)) for f in _LIST_FILES):
            vanish_plus_ids  = _read_ids("vanish_plus_ids.txt")
            blossum_plus_ids = _read_ids("blossum_plus_ids.txt")
            vanish_h_ids     = _read_ids("vanishH_ids.txt")
            blossum_h_ids    = _read_ids("blossumH_ids.txt")
            print(f"  [s3 standalone] Loaded taxon lists from {_s2_main}:")
            # Present but older than the matrices: the H+ lists move whenever
            # the Healthy group moves, so Section 3 on stale lists is a
            # mismatch no output records. Warn rather than rebuild, because
            # that choice -- rerun Section 2, or report on the lists as they
            # stand -- is not one this should make silently.
            _in_mtime = _reference_input_mtime(ref_name)
            _stale = [f for f in _LIST_FILES
                      if not _path_is_current(os.path.join(_s2_main, f), _in_mtime)]
            if _stale:
                print(f"  [WARN] {len(_stale)} of these predate the matrices "
                      f"they were derived from ({', '.join(_stale)}) — rerun "
                      f"`--only s2` for {ref_name} if the inputs changed")
        else:
            print(f"\n  Section 2 has not run for {ref_name}, and Section 3 is "
                  f"defined on the taxon lists it derives — deriving them now.")
            from .section2_differential import run_section2
            vanish_plus_ids, blossum_plus_ids, vanish_h_ids, blossum_h_ids = \
                run_section2(data, dm, stats,
                             all_specs=all_specs, main_specs=main_specs,
                             taxonomy_df=taxonomy_df)
            # Deliberately not _mark_stage_done("section2"): this derived the
            # lists, not the rest of Section 2 (Tables S4/S5 and Figures
            # S5A/S5B come from run_supp_s5). Marking it done would make a
            # later --resume --all skip the stage and never produce those.
            print(f"  [s3 standalone] Derived taxon lists into {_s2_main}:")

        print(f"    VANISH+={len(vanish_plus_ids)}  BloSSUM+={len(blossum_plus_ids)}"
              f"  VanishH+={len(vanish_h_ids)}  BlossumH+={len(blossum_h_ids)}")
        if not any([vanish_plus_ids, blossum_plus_ids,
                    vanish_h_ids, blossum_h_ids]):
            print(f"  [WARN] Section 2 produced no taxon lists for {ref_name}; "
                  f"Section 3 has nothing to run on")

    # ═══════════════════════════════════════════════════════
    #  SECTION 3: Health Associations
    # ═══════════════════════════════════════════════════════
    # Section 3 is computed on config.SECTION3_REFS (combined) and skipped
    # elsewhere in the --all sweep: no published figure or table reads a
    # non-combined Section 3, and the background curve's bootstrap costs about
    # eight hours per reference. An explicit `--only s3` overrides, so a
    # single reference can still be computed on request.
    _s3_refs = getattr(config, "SECTION3_REFS", None)
    _s3_asked = bool(sections and "s3" in sections)
    if do_all and not _s3_asked and _s3_refs and ref_name not in _s3_refs:
        print(f"\n  Section 3: Health Associations — skipped for {ref_name}; "
              f"computed on {', '.join(_s3_refs)} "
              f"(config.SECTION3_REFS). Use --ref {ref_name} --only s3 "
              f"to compute it anyway.")
    elif do_all or "s3" in sections:
        artifacts = [
            os.path.join("section3_health", "main",
                         f"health_richness_gmw_{tag}.csv")
            for tag in main_tags
        ]
        artifacts.extend(
            os.path.join("section3_health", "main",
                         f"health_per_taxon_gmw_{tag}.csv")
            for tag in main_tags
        )
        artifacts.extend(
            os.path.join("section3_health", "main",
                         f"health_per_taxon_prevmatched_{tag}.csv")
            for tag in main_tags
        )
        if not _should_skip_stage(args, ref_name, "section3",
                                  "Section 3: Health Associations",
                                  artifacts):
            from .section3_health import run_section3
            run_section3(data, dm, vanish_plus_ids, blossum_plus_ids, stats,
                         vanish_h_ids=vanish_h_ids, blossum_h_ids=blossum_h_ids,
                         taxonomy_df=taxonomy_df)

            # GLMM sensitivity (requires statsmodels + joblib)
            if getattr(config, '_RUN_GLMM', False):
                try:
                    from .section3_health_glmm import run_health_glmm
                    run_health_glmm(data, vanish_plus_ids, blossum_plus_ids, stats,
                                    vanish_h_ids=vanish_h_ids,
                                    blossum_h_ids=blossum_h_ids,
                                    depth_knots=3, n_jobs=-1)
                except ImportError as e:
                    print(f"    [WARN] GLMM skipped: {e}")
                    print(f"    Install with: pip install statsmodels joblib patsy")
            _mark_stage_done(ref_name, "section3")

    # ═══════════════════════════════════════════════════════
    #  SECTION 3G: Demographic strata, on their own
    # ═══════════════════════════════════════════════════════
    # Section 3 already runs this, so the key exists only to produce or
    # refresh the demographic figure and its tables without repeating the
    # whole of Section 3, which is expensive and whose other outputs are
    # unaffected. (That figure held slot S32 until 2026-10-09 and is not a
    # numbered supplementary figure now, so this no longer names a slot.)
    if sections and "s3demo" in sections and "s3" not in sections:
        from collections import OrderedDict as _OD

        from .detection import DetectionSpec as _DS
        from .section3_health import run_demographic_associations

        _s3_main = os.path.join("results", ref_name, "section3_health", "main")
        os.makedirs(_s3_main, exist_ok=True)
        _bcols = set(data.breadth.columns)
        _tl = _OD()
        for _name, _ids in (
                ("VANISH", sorted(data.vanish_set or [])),
                ("BloSSUM", sorted(data.blossum_set or [])),
                ("VANISH+", vanish_plus_ids or []),
                ("BloSSUM+", blossum_plus_ids or []),
                ("VanishH+", vanish_h_ids or []),
                ("BlossumH+", blossum_h_ids or [])):
            _keep = [m for m in _ids if m in _bcols]
            if _keep:
                _tl[_name] = _keep
        _specs = [_DS("breadth_pct", config.BREADTH_THRESHOLD),
                  _DS("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP)]
        run_demographic_associations(data, _tl, _specs, _s3_main, stats)

    # ═══════════════════════════════════════════════════════
    #  SENSITIVITY SWEEP: Alternative VANISH/BloSSUM definitions
    # ═══════════════════════════════════════════════════════
    if do_all and getattr(config, '_RUN_SENSITIVITY', False) or \
       (sections and "sensitivity" in sections):
        from .sensitivity_sweep import (
            run_sensitivity_sweep, NAMED_SETS, CORE_DEFINITIONS,
        )

        # Determine which definition set to use
        defn_set_name = getattr(config, '_SENSITIVITY_DEFS', 'core')
        if isinstance(defn_set_name, str):
            definitions = NAMED_SETS.get(defn_set_name, CORE_DEFINITIONS)
        elif isinstance(defn_set_name, list):
            # Union of named sets
            definitions = []
            seen_tags = set()
            for name in defn_set_name:
                for d in NAMED_SETS.get(name, []):
                    if d.tag not in seen_tags:
                        definitions.append(d)
                        seen_tags.add(d.tag)
        else:
            definitions = CORE_DEFINITIONS

        if not _should_skip_stage(args, ref_name, "sensitivity",
                                  "Sensitivity sweep"):
            run_sensitivity_sweep(
                data, dm, stats,
                definitions=definitions,
                run_section2=True,
                run_section3=(dm is not None and dm.triplet_ni),
            )
            _mark_stage_done(ref_name, "sensitivity")

    # ═══════════════════════════════════════════════════════
    #  Visualization (optional)
    # ═══════════════════════════════════════════════════════
    if do_all or "visualization" in (sections or set()):
        if not _should_skip_stage(args, ref_name, "visualization",
                                  "Visualization"):
            from .visualization import run_visualization
            run_visualization(data)
            _mark_stage_done(ref_name, "visualization")

    # ═══════════════════════════════════════════════════════════
    #  DEPTH-BIAS IPW  (standalone invocation only)
    #  When S1 runs normally it already calls both functions
    #  internally (steps 1G + 1H).  This block only fires
    #  when the user requests --only depth_bias, allowing
    #  re-running IPW without re-running all of Section 1
    #  (e.g. to increase n_boot for final publication CIs).
    # ═══════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════
    #  Carter subset: Additional Figures A6-A9.
    #  Runs for combined / UHGG / lev; the module skips chimp,
    #  where the identifier join has no lifestyle to compare.
    # ═══════════════════════════════════════════════════════════
    if do_all or (sections and "carter" in sections):
        from .carter_analysis import run_carter_analysis
        _all_s, _main_s = _build_detection_specs(detection_mode)
        run_carter_analysis(data, stats, specs=_main_s)

    if sections and "depth_bias" in sections and \
            not ({"s1", "s1_alpha"} & sections):
        from .depth_bias_ipw import (run_gmw_richness,
                                     run_gmw_threshold_sweep,
                                     run_geomean_richness_pairwise)
        _all_s, _main_s = _build_detection_specs(detection_mode)
        print(f"\n  Standalone: GMW richness [{ref_name}]")
        run_gmw_richness(
            data, stats,
            specs=_main_s if _main_s else None,
            n_boot=getattr(config, "OW_AIPWE_N_BOOT_MAIN", 500))
        print(f"\n  Standalone: GMW threshold sweep [{ref_name}]")
        run_gmw_threshold_sweep(
            data, stats,
            all_specs=_all_s if _all_s else None,
            n_boot=getattr(config, "OW_AIPWE_N_BOOT_SWEEP", 200))
        print(f"\n  Standalone: Pairwise richness [{ref_name}]")
        run_geomean_richness_pairwise(
            data, stats,
            specs=_main_s if _main_s else None,
            n_boot=getattr(config, "OW_AIPWE_N_BOOT_MAIN", 500))

    # ═══════════════════════════════════════════════════════════════════
    #  ACCUMULATION ONLY  (--only accum)
    #  Runs breadth-threshold and flat-bp accumulation curves without
    #  depth-matching or any other section.
    # ═══════════════════════════════════════════════════════════════════
    if sections and "accum" in sections and \
            not ({"s1", "s1_alpha"} & sections):
        from .accumulation import record_accumulation_stats, run_accumulation
        from .section1_diversity import _run_flat_accumulation
        print(f"\n  Standalone: Accumulation curves (log + linear) [{ref_name}]")
        run_accumulation(data)
        record_accumulation_stats(data, stats)
        _run_flat_accumulation(
            data, list(data.meta.index), flat_bp_specs(config.ACCUM_FLAT_THRESHOLDS_BP))

    # ═══════════════════════════════════════════════════════════════════
    #  FIGURE3 ACCUMULATION ONLY  (--only figure3)
    #  Runs only the Industrial(all) vs Non-Industrial(all) Figure3 curves.
    # ═══════════════════════════════════════════════════════════════════
    if sections and "figure3" in sections and \
            not ({"s1", "s1_alpha", "accum"} & sections):
        from .accumulation import run_figure3_accumulation
        print(f"\n  Standalone: Figure3 accumulation curves [{ref_name}]")
        run_figure3_accumulation(data)

    if sections and "graphical-abstract" in sections and \
            not ({"s1", "s1_alpha", "accum", "figure3"} & sections):
        from .graphical_abstract import plot_graphical_abstract
        print(f"\n  Standalone: Graphical abstract outline [{ref_name}]")
        plot_graphical_abstract(data)


    # ═══════════════════════════════════════════════════════════════════
    #  PREVMATCH  (--only prevmatch)
    #  Re-runs stratum-aware prevalence-matched background without
    #  rerunning any other section.  Requires a completed s3 run so
    #  that health_per_taxon_gmw_<TAG>.csv exists.
    # ═══════════════════════════════════════════════════════════════════
    if sections and "prevmatch" in sections and \
            not ({"s1", "s2", "s3"} & sections):
        import glob as _glob
        import pandas as _pd
        from .section3_health import (
            _run_prevalence_matched_background,
            run_prevmatch_plots,
            _run_stratified_shifts,
            _attach_stratified,
            _run_background_curve,
            _run_clade_contrasts,
            _bg_method,
        )
        from collections import OrderedDict as _OD

        k = config.PREVMATCH_K_NEIGHBOURS
        print(f"\n  Standalone prevmatch (k={k}) [{ref_name}]")

        _ref_dir = config.outdir(ref_name)
        _s3_main = os.path.join("results", ref_name, "section3_health", "main")

        # For taxon driver labelling.
        _taxonomy_df = _load_taxonomy()

        def _load_txt_ids(path):
            if not os.path.exists(path):
                return []
            with open(path) as _f:
                return [_l.strip() for _l in _f if _l.strip()]

        def _find_gmw_txt(suffix, group):
            matches = _glob.glob(
                os.path.join(_ref_dir, f"top*_{group}_enriched_gmw_{suffix}.txt"))
            return sorted(matches)

        _vanish_ids  = sorted(m for m in (data.vanish_set  or set()) if m in data.breadth.columns)
        _blossum_ids = sorted(m for m in (data.blossum_set or set()) if m in data.breadth.columns)
        _vp  = _load_txt_ids(next(iter(_find_gmw_txt("ind_all_vs_ni",     "g2")), ""))
        _bp  = _load_txt_ids(next(iter(_find_gmw_txt("ind_all_vs_ni",     "g1")), ""))
        _vh  = _load_txt_ids(next(iter(_find_gmw_txt("ind_healthy_vs_ni", "g2")), ""))
        _bh  = _load_txt_ids(next(iter(_find_gmw_txt("ind_healthy_vs_ni", "g1")), ""))

        _taxon_lists = _OD()
        if _vanish_ids:  _taxon_lists["VANISH"]    = _vanish_ids
        if _blossum_ids: _taxon_lists["BloSSUM"]   = _blossum_ids
        if _vp:          _taxon_lists["VANISH+"]   = [t for t in _vp if t in data.breadth.columns]
        if _bp:          _taxon_lists["BloSSUM+"]  = [t for t in _bp if t in data.breadth.columns]
        if _vh:          _taxon_lists["VanishH+"]  = [t for t in _vh if t in data.breadth.columns]
        if _bh:          _taxon_lists["BlossumH+"] = [t for t in _bh if t in data.breadth.columns]

        _tag_suffix = "_no_obese" if "Obese" in config.HEALTH_PLOT_SKIP else ""
        for _spec in main_specs:
            _base_tag = _spec.short_label
            _tag      = _base_tag + _tag_suffix
            _gmw_csv  = os.path.join(_s3_main,
                f"health_per_taxon_gmw_{_base_tag}.csv")  # always read unmodified CSV
            if not os.path.exists(_gmw_csv):
                print(f"    [PREVMATCH] {_gmw_csv} not found — skipping {_tag}")
                continue

            print(f"\n  Running stratum-aware matching for {_spec.description} ...")
            _per_taxon_df = _pd.read_csv(_gmw_csv, low_memory=False)

            # The corrected background needs the study-stratified shift and the
            # curve fitted on it. Computed here and handed down, so the
            # fast-path does the same work once that a full run does.
            _strat, _curve = {}, {}
            if _bg_method() != "iw":
                _strat = _run_stratified_shifts(data, _spec, _s3_main)
                _per_taxon_df = _attach_stratified(_per_taxon_df, _strat,
                                                   _s3_main, _base_tag)
                _curve = _run_background_curve(
                    _strat, _taxon_lists, _s3_main, _base_tag)
                _run_clade_contrasts(_curve, _taxon_lists, data, _s3_main,
                                     _base_tag, ref=ref_name,
                                     spec_desc=_spec.description)

            _bg_df = _run_prevalence_matched_background(
                data, _taxon_lists, _per_taxon_df, _spec, _s3_main,
                strat=_strat, curve=_curve)

            if _bg_df is None or _bg_df.empty:
                print(f"    [PREVMATCH] No background produced for {_tag} — skipping")
                continue

            # The per-condition CMH table, which carries n_studies and so
            # decides which conditions S26 and S28-S31 show. This used to look
            # for health_cmh_results_<tag>.csv -- a name nothing in the
            # pipeline writes -- so the fast path always got None, lost the
            # MIN_STRATA_TO_PLOT filter, and drew MASLD and ASCVD columns that
            # a full run drops. Section 3 writes it as:
            _cmh_csv = os.path.join(
                _s3_main, f"health_per_condition_cmh_{_base_tag}.csv")
            _cmh_df = (_pd.read_csv(_cmh_csv, low_memory=False)
                       if os.path.exists(_cmh_csv) else None)
            if _cmh_df is None:
                print(f"    [PREVMATCH] {_cmh_csv} not found: S26 and the "
                      "driver figures will show every condition, not the "
                      "ones Figure 5 plots. Run --only s3 first.")

            # Build a spec with the (possibly suffixed) short_label for filenames.
            # DetectionSpec is a frozen dataclass; passing label= overrides short_label.
            if _tag_suffix:
                from .detection import DetectionSpec as _DS
                _plot_spec = _DS(_spec.mode, _spec.threshold, label=_tag)
            else:
                _plot_spec = _spec

            # Delegate all prevmatch plots (+ taxon driver analysis) to wrapper
            run_prevmatch_plots(
                data, _taxon_lists, _per_taxon_df, _bg_df, _cmh_df,
                _plot_spec, stats, _s3_main,
                taxonomy_df=_taxonomy_df, curve=_curve)

    # ═══════════════════════════════════════════════════════════════════
    #  GMW_RICHNESS  (--only gmw_richness)
    #  Runs Section-1 GMW depth-corrected richness (Ind vs NI) and the
    #  threshold robustness sweep without re-running any other section.
    # ═══════════════════════════════════════════════════════════════════
    if sections and "gmw_richness" in sections and \
            not ({"s1", "s2", "s3"} & sections):
        from .depth_bias_ipw import (run_gmw_richness as _run_gmw_rich,
                                     run_gmw_threshold_sweep as _run_gmw_sweep)
        _all_s, _main_s = _build_detection_specs(detection_mode)
        print(f"\n  Standalone: GMW richness [{ref_name}]")
        _run_gmw_rich(data, stats, specs=_main_s if _main_s else _all_s)
        print(f"\n  Standalone: GMW threshold sweep [{ref_name}]")
        _run_gmw_sweep(data, stats, all_specs=_all_s if _all_s else _main_s)

    # ═══════════════════════════════════════════════════════════════════
    #  HEALTH_RICHNESS  (--only health_richness)
    #  Runs triplet GMW richness without re-running any other section.
    #  Requires depth-matching to have run (dm must exist).
    # ═══════════════════════════════════════════════════════════════════
    if sections and "health_richness" in sections and \
            not ({"s1", "s2", "s3"} & sections):
        import pandas as _pd
        from .section3_health import (
            _run_health_richness_gmw,
            _run_richness_comparison,
            _plot_richness_comparison,
        )
        if dm is None:
            print("  [WARN] health_richness requires depth matching — "
                  "re-running depth match")
            from .depth_matching import build_depth_matched_sets
            dm = build_depth_matched_sets(
                data, frac_tol=config.DEPTH_MATCH_FRAC_TOL,
                health_frac_tol=config.HEALTH_TRIPLET_FRAC_TOL)
        _s3_main = os.path.join("results", ref_name,
                                 "section3_health", "main")
        os.makedirs(_s3_main, exist_ok=True)
        _all_s, _main_s = _build_detection_specs(detection_mode)
        _specs = _main_s if _main_s else _all_s

        print(f"\n  Standalone: Triplet health richness [{ref_name}]")

        for _spec in _specs:
            _tag  = _spec.short_label
            _desc = _spec.description

            # health_richness_{ref}_{tag}.png — two-panel bar chart
            print(f"    GMW richness comparison ({_desc})")
            _richness_df = _run_richness_comparison(data, _spec, _s3_main, stats)
            _plot_richness_comparison(_richness_df, ref_name, _s3_main, _tag, _desc)

        # health_richness_gmw_{ref}_*_{tag}.png — triplet violin summaries
        _run_health_richness_gmw(data, dm, _specs, _s3_main, stats)

    # ═══════════════════════════════════════════════════════
    #  SECTION 1 (pcoa3): 3-way GMW-weighted PCoA
    #  H-G / Transitional / Industrial with equal depth-target.
    #  Run: python -m persistence --ref UHGG --only s1_pcoa3
    # ═══════════════════════════════════════════════════════
    if do_all or {"s1", "s1_pcoa3"} & (sections or set()):
        if not _should_skip_stage(args, ref_name, "s1_pcoa3",
                                  "Section 1 PCoA3"):
            from .pcoa_gmw3 import run_pcoa_gmw3
            run_pcoa_gmw3(data, stats)
            _mark_stage_done(ref_name, "s1_pcoa3")

    # ═══════════════════════════════════════════════════════
    #  SECTION 1 (axis-angle): depth-binned extinction-axis
    #  projection and Trans/Ind angle analysis.
    #  Requires --only s1_pcoa3 to have run first.
    #  Run: python -m persistence --ref UHGG --only s1_axis_angle
    # ═══════════════════════════════════════════════════════
    if do_all or {"s1", "s1_axis_angle"} & (sections or set()):
        artifacts = [
            os.path.join("section1_diversity", "main",
                         f"axis_angle_v2_pointcloud_{tag}_notitle.png")
            for tag in main_tags
        ]
        if not _should_skip_stage(args, ref_name, "s1_axis_angle",
                                  "Section 1 axis-angle analysis",
                                  artifacts):
            from .axis_angle_analysis import run_axis_angle
            run_axis_angle(data, stats)
            _mark_stage_done(ref_name, "s1_axis_angle")

    # ═══════════════════════════════════════════════════════
    #  SECTION 1 (axis-stats): statistical tests on the
    #  extinction-axis projections (JT, Wilcoxon, PERMANOVA,
    #  t-test on theta).  Reads CSVs from s1_axis_angle.
    #  Run: python -m persistence --ref UHGG --only s1_axis_stats
    # ═══════════════════════════════════════════════════════
    if do_all or {"s1", "s1_axis_stats"} & (sections or set()):
        if not _should_skip_stage(args, ref_name, "s1_axis_stats",
                                  "Section 1 axis stats"):
            from .axis_angle_stats import run_axis_angle_stats
            run_axis_angle_stats(data, stats)
            _mark_stage_done(ref_name, "s1_axis_stats")

    # ═══════════════════════════════════════════════════════
    #  S3_AXIS: health-condition projection onto NI→Ind axis
    #  Run: python -m persistence --ref UHGG --only s3_axis
    # ═══════════════════════════════════════════════════════
    if do_all or {"s3_axis"} & (sections or set()):
        if not _should_skip_stage(args, ref_name, "s3_axis",
                                  "Section 3 health-axis projection"):
            from .health_axis_projection import run_health_axis_projection
            run_health_axis_projection(data, stats)
            _mark_stage_done(ref_name, "s3_axis")

    # ═══════════════════════════════════════════════════════
    #  TABLE S1A: per-study metadata summary (mean ± SD)
    #  Run: python -m persistence --ref UHGG --only table_s1a
    # ═══════════════════════════════════════════════════════
    if do_all or {"table_s1a"} & (sections or set()):
        if not _should_skip_stage(args, ref_name, "table_s1a",
                                  "Table S1A"):
            from .table_s1a import run_table_s1a
            run_table_s1a(data, stats)
            _mark_stage_done(ref_name, "table_s1a")


    # ═══════════════════════════════════════════════════════
    #  MANUSCRIPT / SUPPLEMENT FIGURE SCRIPTS
    # ═══════════════════════════════════════════════════════
    if run_figures and (do_all or "figures" in (sections or set())):
        if not _should_skip_stage(args, ref_name, "figures",
                                  "Manuscript/supplement figure scripts"):
            from .figure_runner import run_external_figures
            run_external_figures(
                [ref_name],
                selected=getattr(args, "figure", None),
                strict=getattr(args, "figure_strict", False),
                quick=getattr(args, "figure_quick", False),
                data_by_ref={ref_name: data},
            )
            _mark_stage_done(ref_name, "figures")

    # ── Save statistics report ──
    stats.save()

    elapsed = time.time() - t0
    print(f"\n  [{ref_name}] Done in {elapsed/60:.1f} min")


def _assert_repo_root():
    """Fail loudly if launched from the wrong directory.

    config.py names every input by bare filename and writes to results/, all
    relative to the working directory. Launched from elsewhere the pipeline finds
    no matrices, reports every reference as "inputs absent", and exits 0 having
    done nothing -- which looks like success. Checking costs nothing.
    """
    import os as _os
    from pathlib import Path as _P
    cwd = _P.cwd()
    marker = cwd / "persistence" / "config.py"
    if not marker.exists():
        expected = _P(__file__).resolve().parents[1]
        raise SystemExit(
            "ERROR: run the pipeline from the repository root.\n"
            f"  cwd is      : {cwd}\n"
            f"  expected    : {expected}\n"
            "  config.py resolves inputs relative to the working directory, so "
            "from here\n  every reference would look input-less and the run "
            "would do nothing.\n"
            f"  fix: cd {expected}")


def main():
    _assert_repo_root()
    parser = argparse.ArgumentParser(
        description="Microbiome extinctions manuscript analysis pipeline")
    parser.add_argument(
        "--ref", nargs="+", default=["UHGG"],
        help="Reference database(s): UHGG, lev, chimp, combined "
             "(default: UHGG)")
    parser.add_argument(
        "--only", nargs="+", default=None,
        choices=sorted(VALID_SECTIONS),
        help=("Run only specific sections. With '--only figures', missing "
              "main-text figure prerequisites are regenerated in resume mode."))
    parser.add_argument(
        "--detection-mode", dest="detection_mode",
        default="both", choices=["breadth", "flat", "both"],
        help="Detection mode (default: both)")
    parser.add_argument(
        "--no-resume", dest="no_resume", action="store_true",
        help="Redraw everything, ignoring checkpoints and existing outputs. "
             "Needed after a change to shared plotting code, where the figures "
             "that most need redrawing are exactly the ones resume skips.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip completed pipeline stages using checkpoints and conservative "
             "artifact checks, then continue from the first incomplete stage")
    parser.add_argument(
        "--frac-tol", dest="frac_tol",
        type=float, default=None,
        help="Override fractional tolerance for 1:1 depth matching (default: 0.03 = 3%%)")
    parser.add_argument(
        "--health-frac-tol", dest="health_frac_tol",
        type=float, default=None,
        help="Override fractional tolerance for health triplet matching")
    parser.add_argument(
        "--all", action="store_true",
        help="Run all references (UHGG + lev + combined + chimp)")
    parser.add_argument(
        "--glmm", action="store_true",
        help="Run GLMM sensitivity analyses alongside CMH (requires statsmodels)")
    parser.add_argument(
        "--anchored-hg-tol", dest="anchored_hg_tol",
        type=float, default=None,
        help="Override frac tolerance for H-G-anchored matching (default: 0.05 = 5%%)")
    parser.add_argument(
        "--anchored-ni-tol", dest="anchored_ni_tol",
        type=float, default=None,
        help="Override frac tolerance for NI-anchored matching (default: 0.05 = 5%%)")
    parser.add_argument(
        "--anchored-hg-keep", dest="anchored_hg_keep",
        type=float, default=None,
        help="Fraction of H-G anchors to keep (default: 0.90)")
    parser.add_argument(
        "--anchored-ni-keep", dest="anchored_ni_keep",
        type=float, default=None,
        help="Fraction of NI anchors to keep (default: 0.80)")
    parser.add_argument(
        "--sensitivity", action="store_true",
        help="Run VANISH/BloSSUM sensitivity sweep (re-derive taxa lists "
             "from data with multiple threshold definitions, re-run S2+S3)")
    parser.add_argument(
        "--sensitivity-defs", dest="sensitivity_defs",
        nargs="+", default=["core"],
        choices=["core", "tight", "loose", "full"],
        help="Definition sets for sensitivity sweep "
             "(default: core; options: core, tight, loose, full)")
    parser.add_argument(
        "--sensitivity-only-s2", dest="sensitivity_only_s2",
        action="store_true",
        help="Sensitivity sweep: only re-run Section 2 (skip Section 3)")
    parser.add_argument(
        "--derive-vb", dest="derive_vb", action="store_true",
        help="Replace file-based VANISH/BloSSUM with a single data-driven "
             "definition, then run the normal S2+S3 pipeline with those lists")
    parser.add_argument(
        "--vb-r", dest="vb_r", type=float, default=0.02,
        help="Rare cutoff for --derive-vb (default: 0.02 = 2%%)")
    parser.add_argument(
        "--vb-m", dest="vb_m", type=float, default=0.10,
        help="Min-present cutoff for --derive-vb (default: 0.10 = 10%%)")
    parser.add_argument(
        "--vb-u", dest="vb_u", type=float, default=0.80,
        help="Max-present cutoff for --derive-vb (default: 0.80 = 80%%)")
    parser.add_argument(
        "--vb-study-aware", dest="vb_study_aware", action="store_true",
        help="Use study-aware (median-of-studies) prevalence for --derive-vb")
    parser.add_argument(
        "--depth-bias-n-boot-main", dest="depth_bias_n_boot_main",
        type=int, default=None,
        help="GMW main figure bootstrap iterations "
             "(default: config.OW_AIPWE_N_BOOT_MAIN = 500)")
    parser.add_argument(
        "--depth-bias-n-boot-sweep", dest="depth_bias_n_boot_sweep",
        type=int, default=None,
        help="GMW threshold-sweep bootstrap iterations per threshold "
             "(default: config.OW_AIPWE_N_BOOT_SWEEP = 200)")
    parser.add_argument(
        "--prevmatch-k", dest="prevmatch_k",
        type=int, default=None,
        help="Neighbours per list taxon for stratum-aware prevalence matching "
             "(default: config.PREVMATCH_K_NEIGHBOURS = 10). "
             "Takes effect with --only prevmatch or the full pipeline.")
    parser.add_argument(
        "--skip-obese", dest="skip_obese", action="store_true",
        help="Exclude the 'Obese' condition from all plots (prevmatch, driver, "
             "distributions).  Sample counts in all other conditions are "
             "unaffected.  Produces separate output files so both runs can "
             "coexist in the same results directory.")
    parser.add_argument(
        "--bayesian", dest="bayesian", action="store_true", default=False,
        help="Use Bayesian latent-L presence detection for Section 2. "
             "P(L>=tau | b, d, G) >= 0.5 replaces the hard breadth threshold. "
             "Requires rank_abundance_analysis.py (auto-discovered from the "
             "package directory or via --ra-module-path / RA_MODULE_PATH).")
    parser.add_argument(
        "--bayes-null", dest="bayes_null", default="10pct",
        choices=["1pct", "3pct", "10pct", "30pct", "50pct", "90pct", "10kb", "30kb", "100kb", "300kb", "500kb", "1Mb"],
        help="Null threshold for Bayesian detection: '10pct' (10%% genome "
             "coverage, default — matches BREADTH_THRESHOLD=10) or "
             "the configured consistency thresholds.")
    parser.add_argument(
        "--ra-module-path", dest="ra_module_path", default=None,
        metavar="PATH",
        help="Explicit path to rank_abundance_analysis.py. "
             "Auto-discovered from the package directory if not supplied.")
    parser.add_argument(
        "--figure", dest="figure", nargs="+", default=None,
        help="Figure task(s) to run with --only figures. Use names such as "
             "main, full, figure1, figure2, figure3, figure4, figure5, roadmap, guide_fig5, tables_s4_s5, "
             "bayes_section2, rarefaction, accumulation_summary, "
             "metadata_summary, metadata_map. Default: figure1 through figure5")
    parser.add_argument(
        "--generate-all-additional-figures", dest="gen_additional",
        action="store_true",
        help="Run exactly the sections the Supplementary-Figure manifest needs, "
             "then check every slot was written. Equivalent to --only with the "
             "keys tools/figure_sets.py derives from the manifest, so the set "
             "follows the manifest rather than a list that can drift from it.")
    parser.add_argument(
        "--generate-main-figures", dest="gen_main", action="store_true",
        help="Run the main-text figures (figure1..figure5).")
    parser.add_argument(
        "--verify-figures", dest="verify_figures", action="store_true",
        help="Do not run anything; report which Supplementary-Figure slots are "
             "present and which are missing.")
    parser.add_argument(
        "--figure-strict", dest="figure_strict", action="store_true",
        help="Fail instead of skipping figure tasks with missing prerequisite outputs")
    parser.add_argument(
        "--figure-quick", dest="figure_quick", action="store_true",
        help="Run only quick figure tasks when --figure is not provided")

    args = parser.parse_args()

    # The figure-set flags are translations, not a second code path: they set
    # --only and --figure from the manifest and then the ordinary machinery
    # runs. That way the set cannot drift from the manifest, and there is no
    # duplicate list to forget to update.
    _fs = None
    if args.gen_additional or args.gen_main or args.verify_figures:
        import importlib.util as _ilu
        _p = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "tools", "figure_sets.py")
        _spec = _ilu.spec_from_file_location("figure_sets", _p)
        _fs = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_fs)

    if args.verify_figures:
        ok, _missing, _stale = _fs.verify()
        raise SystemExit(0 if ok else 1)

    if args.gen_additional:
        # Force the reference. --ref defaults to UHGG, and every Additional
        # Figure but the three Figure-3 derivatives is a combined-reference
        # output, so accepting the default here would run the whole set
        # against the wrong reference and write nothing the manifest points at.
        _refs = {r for _s, r, _st in _fs.manifest()} - {"Figure3_combined"}
        if _refs - {"combined"}:
            print(f"  [additional] manifest expects {sorted(_refs)}")
        if "--ref" in sys.argv:
            if args.ref != ["combined"]:
                print(f"  [additional] WARNING --ref {' '.join(args.ref)} was "
                      f"given, but the Supplementary Figures are combined-reference "
                      f"outputs. Running as asked; --verify-figures will fail.")
        else:
            args.ref = ["combined"]
            print("  [additional] --ref combined (the manifest's reference)")
        _keys, _unmapped = _fs.sections_for_manifest()
        args.only = sorted(set(args.only or []) | set(_keys))
        _tasks = _fs.figure_tasks_for_manifest()
        if _tasks:
            args.figure = sorted(set(args.figure or []) | set(_tasks))
        args.resume = not args.no_resume
        print(f"  [additional] manifest needs: --only {' '.join(args.only)}"
              + (f" --figure {' '.join(args.figure)}" if args.figure else ""))
        for _s, _r, _stem in _unmapped:
            print(f"  [additional] WARNING no section produces {_s}: {_r}/{_stem}")

    if args.no_resume:
        args.resume = False
        print("  [run] resume off: every figure will be redrawn")

    if args.gen_main:
        args.only = sorted(set(args.only or []) | {"figures"})
        args.figure = sorted(set(args.figure or []) | {"main"})
        print("  [main] --only figures --figure main")
    t_total = time.time()

    config.DETECTION_MODE = args.detection_mode
    config._RUN_GLMM = args.glmm
    if args.frac_tol is not None:
        config.DEPTH_MATCH_FRAC_TOL = args.frac_tol
    if args.health_frac_tol is not None:
        config.HEALTH_TRIPLET_FRAC_TOL = args.health_frac_tol
    if args.anchored_hg_tol is not None:
        config.ANCHORED_HG_FRAC_TOL = args.anchored_hg_tol
    if args.anchored_ni_tol is not None:
        config.ANCHORED_NI_FRAC_TOL = args.anchored_ni_tol
    if args.anchored_hg_keep is not None:
        config.ANCHORED_HG_PCT_KEEP = args.anchored_hg_keep
    if args.anchored_ni_keep is not None:
        config.ANCHORED_NI_PCT_KEEP = args.anchored_ni_keep

    # Sensitivity sweep config
    config._RUN_SENSITIVITY = args.sensitivity
    config._SENSITIVITY_DEFS = (args.sensitivity_defs
                                 if len(args.sensitivity_defs) > 1
                                 else args.sensitivity_defs[0])
    config._SENSITIVITY_ONLY_S2 = getattr(args, 'sensitivity_only_s2', False)

    # Data-driven V/B config
    config._DERIVE_VB = args.derive_vb
    config._VB_R = args.vb_r
    config._VB_M = args.vb_m
    config._VB_U = args.vb_u
    config._VB_STUDY_AWARE = args.vb_study_aware

    # GMW depth-corrected richness boot counts (override config defaults)
    if args.depth_bias_n_boot_main is not None:
        config.OW_AIPWE_N_BOOT_MAIN = args.depth_bias_n_boot_main
    if args.depth_bias_n_boot_sweep is not None:
        config.OW_AIPWE_N_BOOT_SWEEP = args.depth_bias_n_boot_sweep
    if args.prevmatch_k is not None:
        config.PREVMATCH_K_NEIGHBOURS = args.prevmatch_k
    if args.skip_obese:
        config.HEALTH_PLOT_SKIP = config.HEALTH_PLOT_SKIP | {"Obese"}
        print("  [--skip-obese] 'Obese' excluded from all plots for this run.")

    refs = args.ref
    if args.all:
        refs = ["UHGG", "lev", "combined", "chimp"]

    sections = set(args.only) if args.only else None
    run_figures_after_refs = sections is None or "figures" in (sections or set())
    figure_tasks = _requested_figure_tasks(args) if run_figures_after_refs else []
    prereq_by_ref = {}

    if sections == {"figures"}:
        prereq_by_ref = _main_text_prereq_sections(figure_tasks)
        if prereq_by_ref and not args.resume:
            args.resume = True
            print("  [figures] Enabling resume-style prerequisite checks for --only figures.")

    refs_to_run = refs
    if sections == {"figures"}:
        refs_to_run = list(dict.fromkeys(list(refs) + list(prereq_by_ref)))

    for ref_name in refs_to_run:
        ref_sections = sections
        if sections == {"figures"}:
            ref_sections = prereq_by_ref.get(ref_name, set())
            if not ref_sections:
                continue
        run_reference(
            ref_name,
            ref_sections,
            args.detection_mode,
            cli_args=args,
            run_figures=False,
        )

    # Cross-reference robustness forest plot (if multiple refs ran Section 1)
    if len(refs) > 1 and (sections is None or "s1" in (sections or set())):
        print("\n  Building cross-reference robustness forest plot...")
        from .section1_figure import build_cross_reference_forest
        build_cross_reference_forest(refs=refs)

    if run_figures_after_refs:
        print("\n  Building manuscript figure/table outputs...")
        from .figure_runner import run_external_figures
        run_external_figures(
            refs,
            selected=getattr(args, "figure", None),
            strict=getattr(args, "figure_strict", False),
            quick=getattr(args, "figure_quick", False),
        )

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  All done in {elapsed/60:.1f} min")
    print(f"  Results in: results/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
