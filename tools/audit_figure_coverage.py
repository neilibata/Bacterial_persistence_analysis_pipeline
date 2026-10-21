#!/usr/bin/env python3
"""Would a clean run produce every shipped figure, in the place it belongs?

`tools/figure_sets.py --verify` answers whether a slot's file is *there*.
This answers the prior question -- whether anything in the code still puts it
there -- because the two fail independently and only one of them is visible
after a run:

  PRODUCER   code exists that writes this stem (longest-prefix match, since
             these are f-strings with the reference and detection tag
             interpolated into the middle of the name)
  REACHABLE  that code is reachable from a run: a section the manifest maps to
             an --only key, or a FigureTask registered in a run set
  PRESENT    the results tree actually holds the file
  DATA       a Figure_S<n>_data.pkl sits beside it

PRESENT without PRODUCER is the dangerous combination: the file is there
because an older code path or a hand-run made it, and the next clean run will
not. That is how the supplement arrived at a state where three figures existed
only because someone had run a script by hand.

    python tools/audit_figure_coverage.py
    python tools/audit_figure_coverage.py --results-root <pkg>/repo/results

Exits non-zero if any slot fails PRODUCER, REACHABLE or PRESENT.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util as ilu
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence import figure_runner as FR                    # noqa: E402

#: Everything that could write a figure.
SEARCH_DIRS = ("persistence", "figures", "tools")


def _figure_sets():
    spec = ilu.spec_from_file_location("figure_sets", ROOT / "tools/figure_sets.py")
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _grep(needle: str) -> list[str]:
    """Repo-relative .py files containing this literal."""
    out = subprocess.run(
        ["grep", "-rl", "--include=*.py", "-F", needle]
        + [str(ROOT / d) for d in SEARCH_DIRS],
        capture_output=True, text=True).stdout
    return [os.path.relpath(p, ROOT) for p in out.split() if p]


def producer_for(stem: str) -> tuple[str, str]:
    """Longest prefix of the basename that some module mentions."""
    parts = stem.split("/")[-1].split("_")
    for n in range(len(parts), 1, -1):
        pref = "_".join(parts[:n])
        hits = [h for h in _grep(pref) if not h.startswith("tools/")] or _grep(pref)
        if hits:
            status = "ok" if n == len(parts) else f"ok(+{len(parts) - n})"
            return status, f"{pref!r} in " + ", ".join(sorted(hits)[:2])
    return "MISSING", "no module mentions any prefix of this stem"


def reachable(fs, slot: str, ref: str, stem: str) -> tuple[str, str]:
    if slot in fs.SLOT_TASKS:
        task = fs.SLOT_TASKS[slot]
        if task not in FR.TASKS:
            return "MISSING", f"task {task!r} is not registered"
        sets = [n for n, lst in (("FULL", FR.FULL_TASKS),
                                 ("DEFAULT", FR.DEFAULT_TASKS)) if task in lst]
        if not sets:
            return "MISSING", f"task {task!r} is in no run set"
        return "ok", f"task {task} ({'+'.join(sets)})"
    top = stem.split("/")[0] if "/" in stem else ref
    key = fs.DIR_SECTION.get(top) or fs.DIR_SECTION.get(ref)
    return ("ok", f"--only {key}") if key else ("MISSING",
                                                f"no --only key maps {top!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default=str(ROOT / "results"),
                    help="results/ tree to check for PRESENT (default: ./results)")
    a = ap.parse_args()
    res = Path(a.results_root)
    fs = _figure_sets()
    man = fs.manifest()

    print(f"\n  manifest: {len(man)} slots     results: {res}\n")
    print(f"  {'slot':5s} {'producer':11s} {'reachable':16s} {'present':8s} "
          f"{'data':5s}  stem")
    print("  " + "-" * 110)
    bad = []
    for slot, ref, stem in man:
        pstat, pev = producer_for(stem)
        rstat, rev = reachable(fs, slot, ref, stem)
        base = res / ref / stem
        present = "ok" if any(base.with_suffix(s).exists()
                              for s in (".png", ".pdf")) or \
            (base.parent / (base.name + "_notitle.png")).exists() else "MISSING"
        data = "ok" if (base.parent / "figure_data"
                        / f"Figure_{slot}_data.pkl").exists() else "none"
        print(f"  {slot:5s} {pstat:11s} {rstat:16s} {present:8s} {data:5s}  {stem}")
        if "MISSING" in (pstat, rstat, present):
            bad.append((slot, pstat, pev, rstat, rev, present))

    print()
    if bad:
        print(f"  {len(bad)} slot(s) need attention:")
        for slot, ps, pe, rs, rv, pr in bad:
            print(f"    {slot}: producer={ps} ({pe}); reachable={rs} ({rv}); "
                  f"present={pr}")
    else:
        print(f"  all {len(man)} slots: reachable producer, present, "
              f"data object beside it")

    print("\n  MAIN TEXT")
    for name in ("figure1", "figure2", "figure3", "figure4", "figure5"):
        task = FR.TASKS[name]
        need = task.requires(FR.MAIN_TEXT_REF, res.parent)
        missing = [p for p in need if not p.exists()]
        sets = [n for n, lst in (("DEFAULT", FR.DEFAULT_TASKS),
                                 ("FULL", FR.FULL_TASKS)) if name in lst]
        # The big breadth/depth matrices are git-ignored and symlinked in by
        # setup_symlinks.sh, so they are absent from a transfer package by
        # design. Absent on the machine that runs the pipeline is a real
        # problem; absent here usually is not.
        ext = [p for p in missing if "matrix" in p.name]
        note = ("ok" if not missing else
                f"{len(missing)}/{len(need)} prereq missing"
                + (f" ({len(ext)} external matrices)" if ext else ""))
        print(f"    {name:9s} {'+'.join(sets):14s} {note}")
        for p in missing[:3]:
            tag = "  [external, symlinked at run time]" if p in ext else ""
            print(f"        {p.name}{tag}")

    status = res / "figure_task_status.csv"
    if status.exists():
        rows = list(csv.DictReader(open(status)))
        notok = [r for r in rows if r["status"] != "ok"]
        print(f"\n  LAST RUN: {len(rows)} figure tasks, {len(notok)} not ok")
        for r in notok:
            print(f"    {r['task']:34s} {r['status']}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
