#!/usr/bin/env python3
"""Redraw a figure from the data the pipeline already wrote, without recomputing.

Every section persists what its figures plot as CSV beside the image. A change
to a layout does not change those numbers, so a formatting pass has no business
re-running an ordination or a rarefaction to see it. This reads the CSVs and
calls the drawing code directly.

    python tools/replot.py --list
    python tools/replot.py axis_angle
    python tools/replot.py all

Each entry names the CSVs it needs and the call that draws from them. A figure
whose drawing code is not separable from its computation cannot be registered
here, and `--list` says so rather than pretending otherwise.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import pandas as pd                                                # noqa: E402
from persistence import config                                  # noqa: E402
from persistence.plot_utils import apply_oi_style               # noqa: E402


def _s1(ref):
    return config.section_dir(ref, "s1", "main")


# ── Registry ───────────────────────────────────────────────────────────────
# name -> callable(ref) -> list of figure stems redrawn
REPLOT = {}


def register(name):
    def deco(fn):
        REPLOT[name] = fn
        return fn
    return deco


@register("axis_angle")
def _replot_axis_angle(ref: str):
    """A18, A19a, A19b -- the per-bin ordination is the expensive part and its
    output is already on disk as two CSVs per detection spec."""
    from persistence import axis_angle_analysis as m
    out = _s1(ref)
    drawn = []
    for tag in ("300kb", "10pct"):
        df_p = os.path.join(out, f"axis_angle_v2_{tag}.csv")
        sm_p = os.path.join(out, f"axis_angle_v2_samples_{tag}.csv")
        if not os.path.exists(df_p):
            print(f"    [skip] {tag}: no {os.path.basename(df_p)}")
            continue
        df = pd.read_csv(df_p)
        m._plot(df, tag, out, ref,
                m.N_BINS_DEFAULT, m.MIN_N_HG_DEFAULT, m.MIN_N_OTHER_DEFAULT)
        drawn.append(f"axis_angle_v2_{tag}")
        if os.path.exists(sm_p):
            smp = pd.read_csv(sm_p)
            if not smp.empty:
                m._plot_point_cloud(smp, df, tag, out, ref)
                drawn += [f"axis_angle_v2_pointcloud_{tag}",
                          f"axis_angle_v2_study_centroids_{tag}"]
    return drawn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all")
    ap.add_argument("--ref", default="combined")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        print(f"  registered: {', '.join(sorted(REPLOT))}")
        return 0

    names = sorted(REPLOT) if a.what == "all" else [a.what]
    apply_oi_style()
    total = []
    for n in names:
        if n not in REPLOT:
            print(f"  ?? no replot entry for {n}"); continue
        t0 = time.time()
        drawn = REPLOT[n](a.ref) or []
        total += drawn
        print(f"  {n}: {len(drawn)} figure(s) in {time.time()-t0:.1f}s")
    print(f"\n  redrew {len(total)} figure(s) without recomputing anything")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
