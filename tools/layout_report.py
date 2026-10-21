#!/usr/bin/env python3
"""Collect every figure's layout problems from a pipeline run log into one table.

`savefig_dual` reports overlapping text and occluded data for each figure as it
is written. This gathers those lines so the whole set can be judged at once,
rather than a figure at a time.

    python -m persistence --ref combined > run.log 2>&1
    python tools/layout_report.py run.log

Exit status is 1 when anything is still wrong, so it can gate a release.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEAD = re.compile(r"\[layout\]\s+(\S+):\s+(\d+) text overlaps, (\d+) data occlusions")
DETAIL = re.compile(r"(overlap|occlude)\s+(\d+)%\s+(.*)")


def parse(path: Path):
    figs = {}
    cur = None
    for line in path.read_text(errors="replace").splitlines():
        m = HEAD.search(line)
        if m:
            cur = m.group(1)
            figs[cur] = {"collisions": int(m.group(2)),
                         "occlusions": int(m.group(3)), "detail": []}
            continue
        if cur:
            d = DETAIL.search(line)
            if d:
                figs[cur]["detail"].append((d.group(1), int(d.group(2)), d.group(3).strip()))
            elif line.strip() and not line.startswith(" " * 14):
                cur = None
    return figs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--detail", action="store_true", help="list every offender")
    ap.add_argument("--quiet-ok", action="store_true",
                    help="omit figures with nothing wrong")
    a = ap.parse_args()

    figs = parse(a.log)
    if not figs:
        print("  no [layout] lines found. Was the run made with a current "
              "plot_utils, and did it write figures?")
        return 1

    bad = {k: v for k, v in figs.items() if v["collisions"] or v["occlusions"]}
    print(f"  {len(figs)} figures reported, {len(bad)} with problems\n")
    print(f"  {'figure':<52}{'overlaps':>9}{'occluded':>10}")
    order = sorted(figs.items(),
                   key=lambda kv: -(kv[1]["collisions"] + kv[1]["occlusions"]))
    for name, v in order:
        if a.quiet_ok and not (v["collisions"] or v["occlusions"]):
            continue
        flag = "" if not (v["collisions"] or v["occlusions"]) else "  <--"
        print(f"  {name[:50]:<52}{v['collisions']:>9}{v['occlusions']:>10}{flag}")
        if a.detail:
            for kind, pct, what in v["detail"]:
                print(f"      {kind:<8}{pct:>4}%  {what}")

    tot_c = sum(v["collisions"] for v in figs.values())
    tot_o = sum(v["occlusions"] for v in figs.values())
    print(f"\n  total: {tot_c} text overlaps, {tot_o} data occlusions "
          f"across {len(figs)} figures")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
