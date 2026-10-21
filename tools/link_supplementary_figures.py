#!/usr/bin/env python3
"""One directory holding every Supplementary Figure, as symlinks into results/.

The figures live where the section that drew them put them, which is the right
place for the pipeline and the wrong place for reading them in order. This lays
the manifest out flat:

    supplementary_figures/
        S01_figure_s0_roadmap.png -> ../results/combined/figure_s0_roadmap.png
        ...
        figure_data/
            S01_figure_s0_roadmap.pkl -> ../../results/combined/figure_data/Figure_S1_data.pkl

Symlinks, so nothing is copied and a figure rerun is picked up without doing
anything here. What does go stale is the mapping: edit
tools/additional_figures_manifest.txt and the old links still resolve, silently,
to the figures the manifest no longer names. So rebuild after a figure run or a
manifest change:

    python tools/link_supplementary_figures.py

Links are relative, so the directory survives moving the tree. Stale links --
ones this script made before and the manifest no longer asks for -- are removed;
anything that is not a symlink is left alone and reported.

A finished run's package ships this directory dereferenced (real files, see
README), so a redraw into a package refreshes it with --copy instead:

    python tools/link_supplementary_figures.py --results <pkg>/results \
        --out <pkg>/supplementary_figures --copy
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence import figure_source          # noqa: E402
from tools import figure_sets                     # noqa: E402


def slot_num(slot: str) -> int:
    return int(slot.lstrip("Ss"))


def planned(out: Path, results: Path = ROOT / "results"):
    """[(link, source, slot), ...] for the figures and their data objects."""
    plan = []
    for slot, ref, stem in sorted(figure_sets.manifest(), key=lambda r: slot_num(r[0])):
        name = f"S{slot_num(slot):02d}_{Path(stem).name}"
        fig = results / ref / f"{stem}.png"
        dat = (results / ref / stem).parent / figure_source.DATA_DIRNAME \
            / f"Figure_{slot}_data.pkl"
        plan.append((out / f"{name}.png", fig, slot))
        plan.append((out / figure_source.DATA_DIRNAME / f"{name}.pkl", dat, slot))
    return plan


def link(dest: Path, src: Path, dry_run: bool = False) -> str:
    """Point dest at src relatively. Returns what happened."""
    rel = os.path.relpath(src, dest.parent)
    if dest.is_symlink():
        if os.readlink(dest) == rel:
            return "kept"
        if not dry_run:
            dest.unlink()
    elif dest.exists():
        return "blocked"
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(rel)
    return "linked"


def copy(dest: Path, src: Path, dry_run: bool = False) -> str:
    """Write src's bytes to dest as a real file. Returns what happened."""
    if dest.is_file() and not dest.is_symlink() and \
            dest.read_bytes() == src.read_bytes():
        return "kept"
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink():
            dest.unlink()
        shutil.copy2(src, dest)
    return "linked"


#: What this script names a file: the slot, zero-padded, then the stem. A file
#: matching this in the view directory was put there by `planned()`, so it is
#: ours to drop once the manifest stops asking for it.
_OURS = re.compile(r"^S\d{2}_.+\.(png|pkl)$")


def prune(out: Path, keep: set, dry_run: bool = False,
          as_copies: bool = False):
    """Drop entries we made before that the manifest no longer asks for.

    A symlink view is easy: anything we linked and no longer want goes. A
    copied view -- what a package gets -- used to be skipped entirely, on the
    grounds that every file in it is real and so indistinguishable from one
    somebody else put there. That held until a renumbering. `S05_<stem>.png`
    copied under the old numbering is not overwritten when that stem moves to
    S14; both names then sit in the directory and the view carries two files
    for one figure, which is how the 2026-10-19 rotation first showed up.
    Recognise our own copies by name instead, and leave anything that does
    not match the pattern alone.
    """
    dropped, foreign = [], []
    for d in (out, out / figure_source.DATA_DIRNAME):
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_dir() and p.name == figure_source.DATA_DIRNAME:
                continue
            if p in keep:
                continue
            if p.is_symlink() or (as_copies and _OURS.match(p.name)):
                dropped.append(p)
                if not dry_run:
                    p.unlink()
            else:
                foreign.append(p)
    return dropped, foreign


def build(out: Path, dry_run: bool = False, quiet: bool = False,
          results: Path = ROOT / "results", as_copies: bool = False) -> int:
    plan = planned(out, results)
    counts = {"linked": 0, "kept": 0, "blocked": 0}
    missing, blocked = [], []

    for dest, src, slot in plan:
        if not src.exists():
            missing.append((slot, src))
            continue
        what = (copy if as_copies else link)(dest, src, dry_run)
        counts[what] += 1
        if what == "blocked":
            blocked.append(dest)

    # A copied view is all real files, so prune() identifies ours by name
    # rather than by being a symlink. Skipping the prune here was what let a
    # renumbering leave the old slot names sitting beside the new ones.
    dropped, foreign = prune(out, {d for d, _, _ in plan}, dry_run,
                             as_copies=as_copies)

    if not quiet:
        n = len(figure_sets.manifest())
        made = ("to copy" if dry_run else "copied") if as_copies else \
            ("to link" if dry_run else "linked")
        gone = "to drop" if dry_run else "dropped"
        try:
            where = out.relative_to(ROOT)
        except ValueError:
            where = out
        print(f"  {where}")
        print(f"  {n} slots, {2 * n} links: {counts['linked']} {made}, "
              f"{counts['kept']} unchanged")
        for slot, src in missing:
            print(f"    MISSING  {slot:<5} {src}")
        for p in blocked:
            print(f"    IN THE WAY  {p.name} exists and is not a symlink")
        for p in dropped:
            print(f"    {gone}  {p.name} (not in the manifest)")
        for p in foreign:
            print(f"    left alone  {p.name} (not a symlink)")

    return 0 if not (missing or blocked) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "supplementary_figures"),
                    help="directory to build (default: supplementary_figures/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without touching anything")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--results", default=str(ROOT / "results"),
                    help="results/ tree the figures live in (default: ./results)")
    ap.add_argument("--copy", action="store_true",
                    help="write real files, not symlinks (a package's view)")
    a = ap.parse_args()
    return build(Path(a.out).resolve(), a.dry_run, a.quiet,
                 Path(a.results).expanduser().resolve(), a.copy)


if __name__ == "__main__":
    raise SystemExit(main())
