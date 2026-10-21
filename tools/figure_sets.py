#!/usr/bin/env python3
"""Which sections produce the Supplementary Figures, and did they actually do it.

The manifest is the contract: 31 slots, each naming an output. This resolves
that contract two ways --

    sections_for_manifest()  -> the --only keys that must run to satisfy it
    verify(since=...)        -> which slots exist, and whether they are current

so that "generate all the Additional Figures" is a checkable claim rather than
an assumption that the right sections happened to be in the run.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Output directory -> the --only section key that writes into it. Derived from
# where each section puts its figures, not guessed from the figure's name.
DIR_SECTION = {
    "depth_audit":             "s1",
    "section1_diversity":      "s1",
    "carter_analysis":         "carter",
    "depth_bias_ipw":          "depth_bias",
    "section2_differential":   "s2",
    "section3_health":         "s3",
    "Figure3_combined":        "figure3",
}

# S1 is a standalone figure task rather than a section output.
SLOT_TASKS = {"S1": "read_accounting", "S31": "roadmap"}


def manifest(path: Path = None):
    """[(slot, reference, stem), ...] in document order."""
    p = path or ROOT / "tools/additional_figures_manifest.txt"
    out = []
    for line in p.read_text().splitlines():
        if line and not line.startswith("#"):
            slot, ref, stem = line.split("|")
            out.append((slot.strip(), ref.strip(), stem.strip()))
    return out


def resolve(ref: str, stem: str) -> Path:
    """Where a manifest entry's figure should be."""
    return ROOT / "results" / ref / f"{stem}.png"


def sections_for_manifest():
    """The --only keys needed to produce every slot, plus anything unmapped."""
    need, unmapped = set(), []
    for slot, ref, stem in manifest():
        top = stem.split("/")[0] if "/" in stem else ref
        if slot in SLOT_TASKS:
            need.add("figures")
            continue
        key = DIR_SECTION.get(top) or DIR_SECTION.get(ref)
        if key:
            need.add(key)
        else:
            unmapped.append((slot, ref, stem))
    return sorted(need), unmapped


def figure_tasks_for_manifest():
    """Figure-runner task names the manifest needs (as opposed to sections)."""
    slots = {s for s, _, _ in manifest()}
    return sorted({t for s, t in SLOT_TASKS.items() if s in slots})


def data_coverage(quiet: bool = False):
    """Which figures publish the data object they were drawn from.

    One `Figure_S<n>_data.pkl` per figure, beside it. A figure without one
    cannot be redrawn without re-running its section, and cannot be submitted
    as data S1 -- the two are the same gap.
    """
    have, missing = [], []
    for slot, ref, stem in manifest():
        d = (ROOT / "results" / ref / stem).parent / "figure_data"
        p = d / f"Figure_{slot}_data.pkl"
        (have if p.exists() else missing).append((slot, stem))
    if not quiet:
        print(f"  {len(have)}/{len(manifest())} figures publish their data object")
        for slot, stem in missing:
            print(f"    no data  {slot:<5} {stem}")
    return have, missing


def verify(since: float = None, quiet: bool = False):
    """Check the contract. Returns (ok, missing, stale)."""
    missing, stale, ok = [], [], []
    for slot, ref, stem in manifest():
        p = resolve(ref, stem)
        if not p.exists():
            missing.append((slot, str(p.relative_to(ROOT))))
        elif since is not None and p.stat().st_mtime < since:
            stale.append((slot, str(p.relative_to(ROOT)),
                          time.strftime("%Y-%m-%d %H:%M",
                                        time.localtime(p.stat().st_mtime))))
        else:
            ok.append(slot)
    if not quiet:
        print(f"  {len(ok)}/{len(manifest())} slots present"
              + (" and current" if since is not None else ""))
        for s, p in missing:
            print(f"    MISSING  {s:<5} {p}")
        for s, p, t in stale:
            print(f"    STALE    {s:<5} {p}  (written {t})")
    return (not missing and not stale), missing, stale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", action="store_true",
                    help="print the --only keys the manifest needs")
    ap.add_argument("--data", action="store_true",
                    help="report which figures publish a Figure_S<n>_data.pkl")
    ap.add_argument("--verify", action="store_true",
                    help="check every manifest slot exists")
    ap.add_argument("--since", default=None,
                    help="also require each file to postdate this "
                         "(epoch seconds, or a file to take the mtime of)")
    a = ap.parse_args()

    if a.sections:
        keys, unmapped = sections_for_manifest()
        print(f"  --only {' '.join(keys)}")
        tasks = figure_tasks_for_manifest()
        if tasks:
            print(f"  --figure {' '.join(tasks)}")
        for slot, ref, stem in unmapped:
            print(f"    (no section mapped for {slot}: {ref}/{stem})")
    if a.data:
        have, missing = data_coverage()
        return 0 if not missing else 1
    if a.verify:
        since = None
        if a.since:
            since = (os.path.getmtime(a.since) if os.path.exists(a.since)
                     else float(a.since))
        ok, _, _ = verify(since)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
