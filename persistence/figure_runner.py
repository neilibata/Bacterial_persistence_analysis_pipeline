"""
Run the standalone/root-level manuscript figure scripts from the package CLI.

The historical pipeline kept several figure-building scripts beside the
``persistence`` package.  This module gives them a single package entrypoint
while preserving their standalone CLIs.
"""

from __future__ import annotations

import os
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class FigureTask:
    name: str
    script: str
    description: str
    args: Callable[[str, Path], list[str]]
    requires: Callable[[str, Path], list[Path]] = lambda ref, root: []
    ref_specific: bool = True
    quick_ok: bool = True


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rank_abundance_module(root: Path) -> Path:
    return root / "persistence" / "rank_abundance_analysis.py"


def _main_text_dir(root: Path) -> Path:
    return root / "main_text_figures"


def _result(ref: str, *parts: str) -> Path:
    return Path("results") / ref / Path(*parts)


def _existing_any(root: Path, patterns: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for pat in patterns:
        found.extend(root.glob(pat))
    return found


def _req_s1_main(ref: str, root: Path) -> list[Path]:
    return [root / _result(ref, "section1_diversity", "main")]


def _req_s3_main(ref: str, root: Path) -> list[Path]:
    return [root / _result(ref, "section3_health", "main")]


def _req_demographics(ref: str, root: Path) -> list[Path]:
    """The three CSVs run_demographic_associations writes.

    Named individually rather than as the directory: the figure is skipped
    with a clear reason when Section 3 ran before the demographic strata
    existed, instead of drawing an empty frame from a stale results tree.
    """
    s3 = root / _result(ref, "section3_health", "main")
    return [s3 / f"demographic_{n}_10pct.csv" for n in
            ("richness_gmw", "per_taxon_gmw", "per_taxon_prevmatched")]


def _req_read_accounting(ref: str, root: Path) -> list[Path]:
    """The three HUMAN catalogues' cohort manifests.

    Named individually rather than as the results directory: a run that built
    only the reporting reference would otherwise draw a one-catalogue version
    of a figure whose whole point is the comparison.

    The chimpanzee manifest is deliberately not required. That reference is
    optional by design -- its matrices are often absent and the pipeline
    treats that as non-fatal -- and a missing prerequisite skips a task
    silently, so requiring it would have dropped Figure S1 out of every run
    that did not build the control. The figure draws without the control line
    instead, and prints that it did.
    """
    return [root / _result(r, "cohort_samples.csv")
            for r in ("UHGG", "lev", "combined")]


def _req_accum_sinf(ref: str, root: Path) -> list[Path]:
    """The combined coverage matrix, which this reads directly.

    Slow (~12 min: it re-reads the 3 GB matrix for its bootstrap caches), but
    a 45-hour run absorbs that, and the alternative is a manual step that goes
    unrun -- S16/S17 were missing from a clean tree for exactly that reason.
    """
    from . import config
    ref_spec = config.REFERENCES.get("combined", {})
    f = ref_spec.get("breadth_file")
    return [root / f] if f else []


def _req_figure3_hill(ref: str, root: Path) -> list[Path]:
    """Figure 3's bootstrap cache, written by the `figure3` task.

    Declaring it means this task skips with a clear reason if figure3 has not
    run yet, rather than refitting the bootstrap from scratch.
    """
    return [root / "main_text_figures" / "_figure3_build"
            / "figure3_combined_regenerated_10pct_bootstrap_cache.pkl"]


def _req_s2_main(ref: str, root: Path) -> list[Path]:
    return [root / _result(ref, "section2_differential", "main")]


# The reference the main text reports. Figures 2 and 5 are not built per
# reference -- one version of each goes in the paper -- so they need this named
# once rather than spelled into each task. It was "UHGG" until 2026-09-11, which
# silently disagreed with the supplementary tables: Tables S5 and S9 are built
# from `combined`, and on BloSSUM-H+ the two references disagree about obesity
# (UHGG p=0.233 ns, combined p=0.029 significant). A hand-built combined Figure 5
# went into the manuscript and was then overwritten by a pipeline run that
# rebuilt it as UHGG. Change this, not the individual tasks.
MAIN_TEXT_REF = "combined"


def _req_s2_lists(ref: str, root: Path) -> list[Path]:
    s2 = root / _result(ref, "section2_differential", "main")
    return [
        s2 / "vanish_plus_ids.txt",
        s2 / "blossum_plus_ids.txt",
        s2 / "vanishH_ids.txt",
        s2 / "blossumH_ids.txt",
    ]


def _req_s3_health_for_figure5(ref: str, root: Path) -> list[Path]:
    s3 = root / _result(ref, "section3_health", "main")
    return [
        s3 / "health_richness_gmw_10pct.csv",
        # The reference, not a hardcoded "UHGG". The directory came from
        # `ref` while the filename did not, so for any reference other than
        # UHGG this named a path that cannot exist -- and because a missing
        # prerequisite skips rather than fails, the 44-hour run reported
        # zero errors while silently not building main-text Figure 5.
        s3 / f"health_richness_gmw_{ref}_summary_NI_health_10pct_perSample.csv",
        s3 / "health_per_taxon_gmw_10pct.csv",
        s3 / "health_per_taxon_prevmatched_10pct.csv",
        # Panels F-H, added to the figure on 2026-10-14. `load_list_comparison`
        # returns {} when this table is absent and the figure then draws A-E
        # and reports success -- which was the right behaviour while F-H were
        # a bonus block, and is the wrong one now that P73 cites "Fig. 5F-H"
        # and the caption describes panel H's counts. Requiring it turns a
        # silently short figure into a reported skip.
        #
        # section3_health writes it only when more than one detection spec
        # produced a curve, so a single-threshold run will skip Figure 5 here
        # rather than publish a five-panel version of an eight-panel figure.
        s3 / f"table_s12_method_comparison_{ref}.csv",
    ]


def _req_fig2(ref: str, root: Path) -> list[Path]:
    return _req_s1_main(ref, root) + _req_s3_main(ref, root)


def _meta_input(root: Path) -> Path:
    """Metadata file from config, so a repointed cohort is seen as stale.

    The March cohort filename was hard-coded here, so --resume compared these
    figures against a file the pipeline no longer reads.
    """
    from . import config
    return root / config.META_FILE


def _req_figure1(ref: str, root: Path) -> list[Path]:
    # Figure 1 reports the analytic cohort, which is matrix-derived and so not
    # reproducible from Table M1: it intersects against the cohort manifest.
    # Listing it here is what makes `--resume` rebuild the map when the cohort
    # changes -- without it a repointed matrix silently leaves the old Ns on the
    # figure. Pinned to config.MAIN_COHORT_REF (the cohort the manuscript
    # reports) regardless of `ref`, since this task is not ref-specific.
    from . import config
    from .cohort_manifest import manifest_path
    return [_meta_input(root), Path(manifest_path(config.MAIN_COHORT_REF, str(root)))]


def _ref_inputs(ref: str, root: Path) -> list[Path]:
    """Matrix and lengths files for a reference, read from config.

    These were hard-coded, which broke staleness detection the moment a
    reference was repointed: `--resume` compares a figure's mtime against its
    prerequisites, so a matrix that is not named here can be replaced without
    the figure ever being rebuilt. After the Leviatan backfill the listed files
    were the pre-backfill ones, and Figures 3 and 4 would have silently kept
    their old output.
    """
    from . import config
    spec = config.REFERENCES[ref]
    out = [root / spec["breadth_file"],
           root / spec["depth_file"],
           root / spec["lengths_file"]]
    extra = spec.get("combined_sgb_metadata_file")
    if extra:
        out.append(root / extra)
    return out


def _req_figure3(ref: str, root: Path) -> list[Path]:
    return ([_meta_input(root)]
            + _ref_inputs("UHGG", root)
            + _ref_inputs("lev", root)
            + _ref_inputs("combined", root))


def _req_figure4(ref: str, root: Path) -> list[Path]:
    return ([_meta_input(root)]
            + _ref_inputs("combined", root)
            + _req_s2_lists("combined", root))


def _req_bayes_fig(ref: str, root: Path) -> list[Path]:
    base = root / _result(ref)
    required = _req_s2_main(ref, root)
    if not _existing_any(base, ["bayes_soft_prev_industrial_*.csv", "bayes_plot_v*_*.npz"]):
        required.append(base / "bayes_soft_prev_industrial_10pct.csv")
    return required


def _req_supp(ref: str, root: Path) -> list[Path]:
    # The fixed-bases supplement is assembled from UHGG Section-1 flat-threshold
    # accumulation panels (the task is UHGG-specific regardless of `ref`). Require
    # a couple of representative input panels so the task runs only once Section 1
    # has produced them.
    base = root / "results" / "UHGG"
    return [
        base / "accumulation_50pct_UHGG_flat_300kb_pooled_notitle.png",
        base / "accum_flat_300kb_pooled_params_vsN_notitle.png",
    ]


def _req_occupancy(ref: str, root: Path) -> list[Path]:
    # Figure A17 (occupancy / detect-in->=k accumulation) uses the combined
    # matrices. Read via _ref_inputs so a repointed reference is seen as stale.
    return _ref_inputs("combined", root)


TASKS: dict[str, FigureTask] = {
    "figure1": FigureTask(
        "figure1", "figure1.py", "Main Figure 1 Equal Earth sample map",
        lambda ref, root: ["--out-dir", str(_main_text_dir(root))],
        requires=_req_figure1,
        ref_specific=False,
    ),
    "demographics": FigureTask(
        "demographics", "make_demographic_associations.py",
        # Not a numbered supplementary figure since the 2026-10-09
        # renumbering; it supports Supplementary Text 7.
        "demographic associations (age / sex / BMI)",
        lambda ref, root: [
            "--ref", ref,
            "--out", str(root / _result(
                ref, "section3_health", "supplement",
                f"demographic_associations_{ref}_10pct.png")),
        ],
        requires=_req_demographics,
    ),
    "accum_sinf": FigureTask(
        "accum_sinf", "tools/make_accumulation_sinf_figures.py",
        "Figures S16/S17 finite-catalogue S-infinity accumulation",
        lambda ref, root: [],
        requires=_req_accum_sinf,
        ref_specific=False,
    ),
    "figure3_hill": FigureTask(
        "figure3_hill", "make_combined_figure3.py",
        "Figure S15 accumulation composite, hill model",
        lambda ref, root: [
            "--models", "hill",
            "--out-dir", str(root / "results" / "Figure3_combined"),
            "--cache", str(root / "main_text_figures" / "_figure3_build"
                           / "figure3_combined_regenerated_10pct"
                             "_bootstrap_cache.pkl"),
        ],
        requires=_req_figure3_hill,
        ref_specific=False,
    ),
    "read_accounting": FigureTask(
        "read_accounting", "make_read_accounting.py",
        "Figure S1 read accounting and reference recovery",
        lambda ref, root: ["--ref", MAIN_TEXT_REF],
        requires=_req_read_accounting,
        ref_specific=False,
        quick_ok=False,
    ),
    "roadmap": FigureTask(
        "roadmap", "figure_s0_roadmap_v4.py", "Figure S31 analytic roadmap",
        lambda ref, root: ["--ref", ref, "--out", str(root / _result(ref))],
        requires=lambda ref, root: [],
    ),
    "figure2": FigureTask(
        "figure2", "figure2.py", "Main Figure 2 composite",
        lambda ref, root: [
            "--ref", MAIN_TEXT_REF,
            "--out", str(_main_text_dir(root) / "figure2.pdf"),
        ],
        requires=lambda ref, root: _req_fig2(MAIN_TEXT_REF, root),
        ref_specific=False,
    ),
    "figure3": FigureTask(
        "figure3", "figure3.py", "Main Figure 3 accumulation composite",
        lambda ref, root: ["--out-dir", str(_main_text_dir(root))],
        requires=_req_figure3,
        ref_specific=False,
        quick_ok=False,
    ),
    "figure4": FigureTask(
        "figure4", "figure4.py", "Main Figure 4 differential prevalence/abundance composite",
        lambda ref, root: ["--out-dir", str(_main_text_dir(root))],
        requires=_req_figure4,
        ref_specific=False,
        quick_ok=False,
    ),
    "figure5": FigureTask(
        "figure5", "figure5.py", "Main Figure 5 health composite",
        lambda ref, root: [
            "--ref", MAIN_TEXT_REF,
            "--out", str(_main_text_dir(root) / "figure5.pdf"),
        ],
        requires=lambda ref, root: (_req_s2_lists(MAIN_TEXT_REF, root)
                                    + _req_s3_health_for_figure5(MAIN_TEXT_REF, root)),
        ref_specific=False,
    ),
    "guide_fig5": FigureTask(
        "guide_fig5", "figure_s_guide_fig5.py", "Supplemental guide for Figure 5",
        lambda ref, root: ["--ref", ref, "--out", str(root / _result(ref))],
        requires=_req_s3_main,
    ),
    "tables_s4_s5": FigureTask(
        "tables_s4_s5", "make_tables_s4_s5.py", "Supplementary Tables S4/S5",
        lambda ref, root: ["--ref", ref, "--results-dir", str(root / "results"), "--data-dir", str(root)],
        requires=_req_s2_main,
    ),
    "bayes_section2": FigureTask(
        "bayes_section2", "section2_bayesian_figure.py", "Bayesian Section 2 figure",
        lambda ref, root: [
            "--ref", ref,
            "--out-dir", str(root / "results" / "section2_bayesian"),
            "--data-dir", str(root / _result(ref)),
            "--results-dir", str(root / "results"),
            "--ra-module-path", str(_rank_abundance_module(root)),
            "--use-pipeline-soft-prev",
        ],
        requires=_req_bayes_fig,
        quick_ok=False,
    ),
    "rarefaction": FigureTask(
        "rarefaction", "rarefaction_plugin.py", "Rarefaction/accumulation plugin figure",
        lambda ref, root: ["--ref", ref, "--out", str(root / f"rarefaction_plugin_{ref}_10p0pct.png")],
        requires=_ref_inputs,
        quick_ok=False,
    ),
    "accumulation_summary": FigureTask(
        "accumulation_summary", "summary_accumulation_figure.py", "Accumulation summary figure",
        lambda ref, root: ["--out", str(root / "accumulation_summary.png"), "--module-path", str(_rank_abundance_module(root))],
        requires=lambda ref, root: [],
        ref_specific=False,
        quick_ok=False,
    ),
    "accumulation_v3": FigureTask(
        "accumulation_v3", "accumulation_summary_v3.py", "Advanced accumulation summary v3",
        lambda ref, root: ["--out", str(root / "accumulation_summary_v3.png"), "--module-path", str(_rank_abundance_module(root))],
        requires=lambda ref, root: [],
        ref_specific=False,
        quick_ok=False,
    ),
    "metadata_summary": FigureTask(
        "metadata_summary", "make_metadata_summary.py", "Metadata summary figures and CSVs",
        lambda ref, root: ["--out-dir", str(root / "metadata_summary")],
        requires=lambda ref, root: [_meta_input(root)],
        ref_specific=False,
    ),
    "metadata_map": FigureTask(
        "metadata_map", "make_metadata_equal_earth.py", "Equal Earth metadata map",
        lambda ref, root: ["--out-dir", str(root / "metadata_summary")],
        requires=lambda ref, root: [_meta_input(root)],
        ref_specific=False,
    ),
    "supp_fixed_bases": FigureTask(
        "supp_fixed_bases", "make_supp_figure_fixed_bases.py", "Fixed-bases supplemental figure assembly",
        lambda ref, root: [
            "--ref", "UHGG",
            "--indir", str(root / "results" / "UHGG"),
            "--outdir", str(root / "main_text_figures"),
        ],
        requires=_req_supp,
        ref_specific=False,
    ),
    "occupancy_a17": FigureTask(
        "occupancy_a17", "figure_a17_occupancy.py",
        "Figure A17: occupancy / detect-in->=k accumulation (combined reference)",
        lambda ref, root: ["--refs", "combined",
                           "--out-dir", str(root / "results" / "occupancy_accumulation")],
        requires=_req_occupancy,
        ref_specific=False,
        quick_ok=False,
    ),
    # The two reader-facing views of the supplementary set. Both used to be
    # manual steps, and both went stale that way: staged_figures/ and
    # staged_figures_carter/ survived a metadata correction and a renumbering
    # (A-slots to S-slots) because nothing in a run rebuilt them. They run
    # last in FULL_TASKS, after every producer, and only need the manifest.
    "supp_link": FigureTask(
        "supp_link", "tools/link_supplementary_figures.py",
        "supplementary_figures/: S-numbered symlinks into results/",
        lambda ref, root: ["--out", str(root / "supplementary_figures")],
        requires=lambda ref, root: [root / "tools" / "additional_figures_manifest.txt"],
        ref_specific=False,
        quick_ok=False,
    ),
    # Table S2 is built from every reference's stats_report.csv, so it is a
    # final task too: it was run by hand and transcribed, and drifted.
    "table_s2b": FigureTask(
        "table_s2b", "tools/build_table_s2b_cohorts.py",
        "Table S2b: cohort cascade, samples and hosts, per reference",
        lambda ref, root: ["--results", str(root / "results")],
        requires=lambda ref, root: [root / "results" / "combined" / "stats_report.csv"],
        ref_specific=False,
        quick_ok=False,
    ),
    "supp_stage": FigureTask(
        "supp_stage", "tools/stage_figures_for_docx.py",
        "staged_figures/: the supplementary set resampled to 6.3 in / 220 dpi",
        lambda ref, root: ["--manifest", str(root / "tools" / "additional_figures_manifest.txt"),
                           "--out", str(root / "staged_figures"), "--clean", "--strict"],
        requires=lambda ref, root: [root / "tools" / "additional_figures_manifest.txt"],
        ref_specific=False,
        quick_ok=False,
    ),
}

DEFAULT_TASKS = ["figure1", "figure2", "figure3", "figure4", "figure5"]
FULL_TASKS = [
    "figure1",
    "figure2",
    "figure3",
    "figure4",
    "figure5",
    "roadmap",
    "guide_fig5",
    "tables_s4_s5",
    # bayes_section2 is deliberately not in the default full set. The Bayesian
    # Section 2 analysis was dropped from the 080826 manuscript -- none of the
    # four documents mentions a posterior or credible interval -- and it needs
    # an opt-in `--bayesian` run to produce its inputs, so leaving it here made
    # every full run report three spurious skips. Still registered, so
    # `--figure bayes_section2` works if it is reinstated.
    "metadata_summary",
    "metadata_map",
    "supp_fixed_bases",
    "rarefaction",
    "accumulation_summary",
    "accumulation_v3",
    "occupancy_a17",
    "demographics",
    "read_accounting",
    "figure3_hill",
    "accum_sinf",
    "supp_link",
    "supp_stage",
    "table_s2b",
]
QUICK_TASKS = ["figure1"]
FINAL_TASKS = {"read_accounting", "supp_link", "supp_stage", "table_s2b"}
TASK_GROUPS = {
    "main": DEFAULT_TASKS,
    "main_text": DEFAULT_TASKS,
    "full": FULL_TASKS,
    "all": FULL_TASKS,
}


def resolve_figure_tasks(selected: Iterable[str] | None = None, *, quick: bool = False) -> list[str]:
    names = list(selected or (QUICK_TASKS if quick else DEFAULT_TASKS))
    expanded: list[str] = []
    for name in names:
        expanded.extend(TASK_GROUPS.get(name, [name]))
    return list(dict.fromkeys(expanded))


def available_figures() -> list[str]:
    return sorted(set(TASKS) | set(TASK_GROUPS))


def run_external_figures(
    refs: Iterable[str],
    selected: Iterable[str] | None = None,
    *,
    strict: bool = False,
    quick: bool = False,
    continue_on_error: bool = True,
    data_by_ref: dict[str, object] | None = None,
) -> dict[str, int | None]:
    """Run selected root-level figure scripts and return task exit codes.

    Missing prerequisite files/directories cause a skip unless ``strict`` is set.
    """
    root = _project_root()
    refs = list(refs)
    names = resolve_figure_tasks(selected, quick=quick)
    unknown = [n for n in names if n not in TASKS]
    if unknown:
        raise ValueError(
            f"Unknown figure task(s): {unknown}. Available: {', '.join(available_figures())}"
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    (root / ".mplconfig").mkdir(exist_ok=True)

    # Views of finished output (FINAL_TASKS) run once, after every other task
    # for every reference -- a non-ref-specific task otherwise runs in the
    # first reference's pass, before later references have drawn their figures.
    body = [n for n in names if n not in FINAL_TASKS]
    tail = [n for n in names if n in FINAL_TASKS]
    schedule = [(ref, n) for ref in refs for n in body]
    schedule += [(refs[0], n) for n in tail] if refs else []

    results: dict[str, int | None] = {}
    ran_non_ref_specific: set[str] = set()
    for ref, name in schedule:
        task = TASKS[name]
        if quick and not task.quick_ok:
            print(f"  [figures] skip {name}: not marked quick-safe")
            results[f"{ref}:{name}"] = None
            continue
        if not task.ref_specific:
            if name in ran_non_ref_specific:
                continue
            ran_non_ref_specific.add(name)

        missing = [p for p in task.requires(ref, root) if not p.exists()]
        key = f"{ref}:{name}" if task.ref_specific else name
        if missing:
            msg = "; ".join(str(p) for p in missing)
            if strict:
                raise FileNotFoundError(f"Figure task {name} missing prerequisite(s): {msg}")
            print(f"  [figures] skip {name}: missing {msg}")
            results[key] = None
            continue

        # A task's script normally lives in figures/, but two shipped
        # supplementary figures are produced by tools/ scripts. Allow a
        # task to name a path containing "/" and resolve it from the repo
        # root, so those can be registered rather than left as manual
        # steps that get forgotten -- which is exactly what happened:
        # S15/S16/S17 were absent from a clean run because nothing in the
        # task list produced them.
        script = (root / task.script if "/" in task.script
                  else root / "figures" / task.script)
        print(f"\n  [figures] {name}: {task.description}")

        # Roadmap can use the ReferenceData already loaded by the pipeline,
        # avoiding a second full matrix read during --only figures smoke tests.
        if name == "roadmap" and data_by_ref and ref in data_by_ref:
            spec = importlib.util.spec_from_file_location("_figure_s0_roadmap", script)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load {script}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            outdir = root / _result(ref)
            outdir.mkdir(parents=True, exist_ok=True)
            mod.run(data_by_ref[ref], str(outdir))
            results[key] = 0
            continue

        cmd = [sys.executable, str(script)] + task.args(ref, root)
        print("    " + " ".join(cmd))
        proc = subprocess.run(cmd, cwd=root, env=env)
        results[key] = proc.returncode
        if proc.returncode and not continue_on_error:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    _report_task_status(results, root, quick=quick)
    return results


def _report_task_status(results: dict, root: Path, quick: bool = False) -> None:
    """Say, at the end of the log, which figure tasks did not produce output.

    Tasks run with continue_on_error so one broken figure cannot sink a 12-hour
    run -- but that also meant a failed task was one line in the middle of the
    log, and a run with no Figure 1 read as "completed with no errors". This
    prints every failed or skipped task in a block at the end, and writes
    results/figure_task_status.csv, which tools/audit_final_run.py checks.
    """
    rows = []
    for key, rc in results.items():
        status = "ok" if rc == 0 else ("skipped" if rc is None else f"failed (exit {rc})")
        rows.append({"task": key, "status": status})
    bad = [r for r in rows if r["status"] != "ok"]
    try:
        import csv
        out = root / "results" / ("figure_task_status_quick.csv" if quick
                                  else "figure_task_status.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["task", "status"])
            w.writeheader()
            w.writerows(rows)
    except OSError as exc:
        print(f"  [figures] could not write the task status file: {exc}")
    print("\n" + "=" * 70)
    if bad:
        print(f"  FIGURE TASKS THAT PRODUCED NO OUTPUT: {len(bad)} of {len(rows)}")
        for r in bad:
            print(f"    {r['task']:<28s} {r['status']}")
        print("  Fix and rerun just these with:  python -m persistence --only figures "
              "--figure " + " ".join(sorted({r['task'].split(':')[-1] for r in bad})))
    else:
        print(f"  All {len(rows)} figure tasks produced output.")
    print("=" * 70)
