#!/usr/bin/env python3
"""Render one figure at several font sizes and measure each, so the choice is
made from evidence rather than taste.

The figures were built on a large canvas with small type ("medium", 12 pt on a
29 in canvas) which never collided because every label had four times the room
it needed, and printed at 2.8 pt. The "large" preset (42 pt) printed at 6.4 pt
and collided. This finds what sits between them, per figure, by building the
same figure repeatedly and counting overlaps.

Data is loaded once and reused across sizes, so the sweep costs one load plus
one render per size.

    python tools/calibrate_figure_fonts.py --ref combined \\
        --figure robustness --sizes 12 20 28 36 42
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

from persistence import config                                  # noqa: E402
from persistence.plot_utils import (                            # noqa: E402
    apply_oi_style, find_collisions, find_occlusions,
)
from persistence.stats_report import StatsReport                # noqa: E402


def _load(ref: str):
    """Reference data plus depth matching -- the spine, minus the sections."""
    from persistence.loading import load_reference
    from persistence.reinstate_secondary import reinstate_distinct_secondaries
    from persistence.depth_matching import build_depth_matched_sets

    stats = StatsReport(ref)
    print(f"  loading {ref} ...", flush=True)
    data = load_reference(ref)
    data = reinstate_distinct_secondaries(data, stats)
    print("  depth matching ...", flush=True)
    dm = build_depth_matched_sets(
        data, frac_tol=config.DEPTH_MATCH_FRAC_TOL,
        health_frac_tol=config.HEALTH_TRIPLET_FRAC_TOL)
    return data, dm, stats


FIGURES = {
    # name -> (module, builder). Each builder takes (data, dm, stats).
    "robustness": ("persistence.section1_figure", "build_robustness_figure"),
    "composite":  ("persistence.section1_figure", "build_composite_figure"),
    "readlength": ("persistence.section1_figure", "build_read_length_diagnostic"),
}


def _measure(fig) -> dict:
    """Collisions, occlusions and printed size for a figure as drawn."""
    import numpy as np
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    heights = []
    for ax in fig.axes:
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if t.get_visible() and t.get_text().strip():
                heights.append(t.get_fontsize())
    w_in = float(fig.get_size_inches()[0])
    try:
        bb = fig.get_tightbbox(r)
        saved_in = float(bb.width)
    except Exception:
        saved_in = w_in
    # Printed size once the saved canvas is placed at page width.
    scale = config.__dict__.get("PAGE_WIDTH_IN", 6.3) / max(saved_in, 1e-6)
    med = float(np.median(heights)) if heights else float("nan")
    return {
        "collisions": len(find_collisions(fig)),
        "occlusions": len(find_occlusions(fig)),
        "canvas_in": round(w_in, 2),
        "saved_in": round(saved_in, 2),
        "tick_pt": round(med, 1),
        "printed_tick_pt": round(med * scale, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="combined")
    ap.add_argument("--figure", default="robustness", choices=sorted(FIGURES))
    ap.add_argument("--sizes", type=float, nargs="+",
                    default=[12, 20, 28, 36, 42])
    ap.add_argument("--canvas-scales", type=float, nargs="+", default=[1.0],
                    help="multipliers on the figure's own canvas width. A "
                         "collision is a hard failure and small text is only a "
                         "cost, so a bigger canvas with relatively smaller text "
                         "is a legitimate answer -- this lets the sweep find it.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="also write each rendering here, for eyeballing")
    a = ap.parse_args()

    # A type-size sweep redraws each figure many times over, and the builders
    # emit their data object every time. This is a measuring tool; it must not
    # rewrite the run's submitted figure data.
    os.environ.setdefault("FIGURE_DATA_READONLY", "1")

    import importlib
    mod_name, fn_name = FIGURES[a.figure]
    builder = getattr(importlib.import_module(mod_name), fn_name)

    data, dm, stats = _load(a.ref)
    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True)

    # The builders save and close their own figures, so there is nothing left
    # to measure afterwards. They do print what savefig_dual measured, so
    # capture that instead -- it is the same numbers, taken at draw time.
    import contextlib, io
    LAY = re.compile(r"\[layout\]\s+(\S+):\s+(\d+) text overlaps, (\d+) data occlusions")
    OVF = re.compile(r"widens the save to ([\d.]+) in from ([\d.]+) in.*?(\d+) text artists")

    print(f"\n  {'canvas':>8}{'base pt':>9}{'saved':>8}{'printed':>9}"
          f"{'artists':>9}{'overlaps':>10}{'occluded':>10}")
    rows = []
    from persistence import plot_utils as _pu
    _page0 = _pu.PAGE_WIDTH_IN
    for scale in a.canvas_scales:
        _pu.PAGE_WIDTH_IN = _page0 * scale
        for fs in a.sizes:
            apply_oi_style(base=fs)
            plt.close("all")
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    builder(data, dm, stats)
            except Exception as exc:
                print(f"  {scale:>7.2f}x{fs:>9}  build failed: "
                      f"{type(exc).__name__}: {str(exc)[:40]}")
                continue
            text = buf.getvalue()
            lay = LAY.search(text)
            ovf = OVF.search(text)
            # No [overflow] line means the figure fits its own canvas -- which
            # is the good case, not a missing measurement. Fall back to the
            # figure's actual width, not the page width, or the printed size
            # comes out wildly wrong for large-canvas figures.
            if ovf:
                saved = float(ovf.group(1))
            else:
                figs = [plt.figure(n) for n in plt.get_fignums()]
                saved = (max(float(f.get_size_inches()[0]) for f in figs)
                         if figs else _page0 * scale)
            artists = int(ovf.group(3)) if ovf else 0
            over = int(lay.group(2)) if lay else -1
            occ = int(lay.group(3)) if lay else -1
            # tick labels sit at base-3 in the shared scale; printed size is that
            # shrunk by however much the saved canvas exceeds the page.
            printed = (fs - 3) * _page0 / max(saved, 1e-6)
            rows.append(dict(scale=scale, fs=fs, saved=saved, printed=printed,
                             artists=artists, over=over, occ=occ))
            print(f"  {scale:>7.2f}x{fs:>9}{saved:>7.1f}\"{printed:>9.1f}"
                  f"{artists:>9}{over:>10}{occ:>10}")
            plt.close("all")
    _pu.PAGE_WIDTH_IN = _page0

    print()
    clean = [r for r in rows if r["over"] == 0 and r["occ"] == 0]
    if clean:
        # Collisions are a hard failure and small text only a cost, so pick the
        # clean point with the largest printed text -- not the largest font.
        best = max(clean, key=lambda r: r["printed"])
        print(f"  clean at {len(clean)} of {len(rows)} settings. Best: "
              f"canvas {best['scale']:.2f}x, base {best['fs']:g} pt "
              f"-> prints at {best['printed']:.1f} pt")
    elif rows:
        best = min(rows, key=lambda r: r["over"] + r["occ"])
        print(f"  nothing clean in {len(rows)} settings. Fewest problems: "
              f"canvas {best['scale']:.2f}x, base {best['fs']:g} pt "
              f"({best['over']} overlaps, {best['occ']} occlusions).")
        print("  A figure that cannot be made clean at any size or canvas is "
              "carrying more than the layout holds -- split it or cut content.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
