#!/usr/bin/env python3
"""Produce every submission artifact that the run itself does not.

`python -m persistence --all --figure full` writes the results tree, the
figures and the figure data objects. It does **not** write the journal-facing
artifacts: the Data S1 workbook, the Word rendering of Table S12, the
slot-keyed inventories, or the reproducibility manifest. Each of those is a
separate tool, and a post-run step with nothing watching it is what gets
skipped between a run and a deposit.

So this is the one command that closes that gap, and the one place the list
lives:

    python tools/make_submission.py --check     # what exists, what is missing
    python tools/make_submission.py             # produce everything missing
    python tools/make_submission.py --force     # rebuild even what exists

`--check` exits non-zero when an artifact is missing, so it works as a gate
before packaging a deposit. Run it after the pipeline, from the finished tree.

What this does NOT do: run the pipeline. If `results/` is empty, every step
here fails, and it says so rather than writing empty artifacts.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: The condition-by-list comparison Table S12 is rendered from. Written by
#: Section 3 on the reported reference.
_S12_CSV = ("results/combined/section3_health/main/"
            "table_s12_method_comparison_combined.csv")

#: Evidence that the pipeline has actually run in this tree. `results/`
#: existing is not evidence -- it is created by anything that touches the
#: tree, including a tool that then finds nothing. The cohort manifest is
#: written once per reference, unconditionally, early in a real run.
_RUN_SENTINEL = "results/combined/cohort_samples.csv"


class Step:
    """One artifact, the command that writes it, and why it is wanted."""

    def __init__(self, name, artifact, argv, why, needs=()):
        self.name = name
        self.artifact = artifact          # path, relative to ROOT
        self.argv = argv
        self.why = why
        self.needs = needs                # inputs that must already exist

    def path(self) -> Path:
        return ROOT / self.artifact

    def missing_inputs(self) -> list[str]:
        return [n for n in self.needs if not (ROOT / n).exists()]

    def stale(self) -> bool:
        """Present, but older than the run it is supposed to describe.

        An artifact left over from a previous run is worse than a missing one:
        `--check` would call it present and a deposit would ship last month's
        numbers beside this month's figures.
        """
        sentinel = ROOT / _RUN_SENTINEL
        if not (self.path().exists() and sentinel.exists()):
            return False
        return self.path().stat().st_mtime < sentinel.stat().st_mtime


STEPS = [
    Step("data-s1", "Data_S1_figure_data.xlsx",
         ["tools/build_data_s1.py", "--out", "Data_S1_figure_data.xlsx"],
         "the tabulated data underlying every figure, which the journal "
         "requires as a multi-sheet workbook keyed to figure panels",
         needs=(_RUN_SENTINEL,)),
    Step("table-s12", "docs/Table_S12.docx",
         ["tools/build_table_s12_docx.py", _S12_CSV, "docs/Table_S12.docx"],
         "Table S12 as a Word table, ready to paste into the supplement",
         needs=(_RUN_SENTINEL, _S12_CSV)),
    Step("figure-text", "tools/figure_text_inventory.csv",
         ["tools/collect_figure_text.py", "--all"],
         "every string drawn on every figure, keyed by slot; goes stale on "
         "any renumbering, so it must be regenerated from the finished tree",
         needs=(_RUN_SENTINEL,)),
    Step("science-specs", "docs/science_compliance.md",
         ["tools/audit_science_specs.py", "--skip", "S31"],
         "the journal's figure specifications checked against what was drawn",
         needs=(_RUN_SENTINEL,)),
    # M3 is Data S4, and what it describes is the taxon lists the run derived.
    # It used to be built by hand from a committed snapshot of those lists,
    # the snapshot fell a run behind, and Data S4 ended up describing a
    # membership the paper never analysed: three of the five overlaps Table S6
    # reports disagreed with it. Building it here, from results/, is what
    # stops that happening again -- the lists cannot be a run behind when the
    # step that reads them writes the snapshot back.
    Step("metadata-tables", "metadata_M3.xlsx",
         ["tools/build_metadata_tables.py", "--lists-dir", "results"],
         "Metadata Tables M2 and M3 (Data S3 and Data S4), rebuilt from the "
         "run's own taxon lists so Data S4 describes the membership analysed",
         needs=(_RUN_SENTINEL,)),
    Step("code-map", "docs/code_map.md",
         ["tools/build_provenance_map.py"],
         "which code produces which published item; keyed by figure slot, so "
         "a renumbering invalidates it",
         needs=()),
    Step("naming-crosswalk", "docs/naming_crosswalk.md",
         ["tools/build_naming_crosswalk.py"],
         "what each figure, table and data file is called in the manuscript, "
         "here and in the deposit; keyed by slot for the same reason",
         needs=()),
    Step("repro-manifest", "verification_manifest.json",
         ["tools/verify_reproducibility.py", "--update"],
         "the hashes a reader checks a rebuild against; it can only be "
         "written from a complete tree",
         needs=(_RUN_SENTINEL,)),
]


def _run(step: Step, py: str) -> bool:
    print(f"\n── {step.name} ─────────────────────────────────────────")
    print(f"   {step.why}")
    print(f"   $ python {' '.join(step.argv)}")
    r = subprocess.run([py] + step.argv, cwd=ROOT)
    if r.returncode != 0:
        print(f"   FAILED (exit {r.returncode})")
        return False
    if not step.path().exists():
        print(f"   ran, but {step.artifact} was not written")
        return False
    size = step.path().stat().st_size
    print(f"   wrote {step.artifact}  ({size/1e6:.1f} MB)"
          if size > 1e6 else f"   wrote {step.artifact}  ({size:,} B)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report only; exit non-zero if anything is missing")
    ap.add_argument("--force", action="store_true",
                    help="rebuild artifacts that already exist")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help=f"a subset of: {', '.join(s.name for s in STEPS)}")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter to run each tool with")
    a = ap.parse_args()

    steps = STEPS
    if a.only:
        unknown = sorted(set(a.only) - {s.name for s in STEPS})
        if unknown:
            print(f"unknown step(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        steps = [s for s in STEPS if s.name in a.only]

    if a.check:
        print("Submission artifacts (produced after the run, not by it)\n")
        missing = stale = 0
        for s in steps:
            p = s.path()
            lack = s.missing_inputs()
            if p.exists() and s.stale():
                stale += 1
                print(f"  STALE    {s.artifact}")
                print(f"           older than {_RUN_SENTINEL}; rebuild with "
                      f"--force")
            elif p.exists():
                print(f"  present  {s.artifact}")
            elif lack:
                missing += 1
                print(f"  MISSING  {s.artifact}")
                print(f"           input not present yet: {', '.join(lack)}")
            else:
                missing += 1
                print(f"  MISSING  {s.artifact}"
                      f"   -> python {' '.join(s.argv)}")
        ok_n = len(steps) - missing - stale
        print(f"\n  {ok_n} of {len(steps)} current"
              + (f", {stale} stale" if stale else "")
              + (f", {missing} missing" if missing else ""))
        if missing:
            print("  run `python tools/make_submission.py` to produce them")
        if stale:
            print("  run `python tools/make_submission.py --force` to refresh")
        return 1 if (missing or stale) else 0

    produced = skipped = failed = 0
    for s in steps:
        lack = s.missing_inputs()
        if lack:
            print(f"\n── {s.name}: cannot run, missing input(s): "
                  f"{', '.join(lack)}")
            print("   the pipeline has not been run in this tree")
            failed += 1
            continue
        if s.path().exists() and not a.force and not s.stale():
            print(f"\n── {s.name}: {s.artifact} exists; --force to rebuild")
            skipped += 1
            continue
        if s.stale() and not a.force:
            print(f"\n── {s.name}: {s.artifact} is older than the run; "
                  f"rebuilding")
        if _run(s, a.python):
            produced += 1
        else:
            failed += 1

    print(f"\n{produced} produced, {skipped} already present, {failed} failed")
    if failed:
        print("A deposit is not complete until every one of these exists; "
              "`--check` is the gate.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
