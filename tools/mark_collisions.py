#!/usr/bin/env python3
"""Render one figure and draw a box round every pair of overlapping labels.

The layout report gives counts and strings; this makes them visible. Each
colliding pair gets a red rectangle over the intersection and a number, and the
same numbers are printed with the text involved, so a claim about collisions can
be checked by eye rather than taken on trust.

    python tools/mark_collisions.py --ref combined --figure robustness \\
        --canvas-scale 4.6 --base-pt 12 --out /tmp/marked.png
"""
from __future__ import annotations

import argparse, io, contextlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import Rectangle                           # noqa: E402

from persistence.plot_utils import (                            # noqa: E402
    apply_oi_style, find_collisions, _collect_text_artists,
)
import persistence.plot_utils as _pu                            # noqa: E402
from persistence.stats_report import StatsReport                # noqa: E402
from tools.calibrate_figure_fonts import _load, FIGURES            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="combined")
    ap.add_argument("--figure", default="robustness", choices=sorted(FIGURES))
    ap.add_argument("--canvas-scale", type=float, default=1.0)
    ap.add_argument("--base-pt", type=float, default=12.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import importlib
    mod, fn = FIGURES[a.figure]
    builder = getattr(importlib.import_module(mod), fn)

    data, dm, stats = _load(a.ref)

    page0 = _pu.PAGE_WIDTH_IN
    _pu.PAGE_WIDTH_IN = page0 * a.canvas_scale
    apply_oi_style(base=a.base_pt)

    # Keep the figure alive: the builders close theirs, so hold every figure
    # created during the build and pick the one with the panels.
    made = []
    real_close = plt.close
    plt.close = lambda *args, **kw: None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            builder(data, dm, stats)
        made = [plt.figure(n) for n in plt.get_fignums()]
    finally:
        plt.close = real_close
        _pu.PAGE_WIDTH_IN = page0

    fig = max(made, key=lambda f: len(f.axes)) if made else None
    if fig is None:
        print("  no figure produced"); return 1

    hits = find_collisions(fig)
    print(f"\n  {a.figure} at canvas {a.canvas_scale}x, base {a.base_pt:g} pt: "
          f"{len(hits)} overlapping pairs\n")

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = {id(t): t.get_window_extent(renderer=r)
             for t in _collect_text_artists(fig)
             if t.get_visible() and t.get_text().strip()}
    by_text = {}
    for t in _collect_text_artists(fig):
        if t.get_visible() and t.get_text().strip():
            by_text.setdefault(t.get_text(), []).append(boxes[id(t)])

    one = lambda s: " ".join(s.split())[:44]
    for i, (ta, tb, frac) in enumerate(hits, 1):
        print(f"   {i:>3}. {frac:>5.0%}  {one(ta)!r}")
        print(f"        x {one(tb)!r}")
        for bb_a in by_text.get(ta, []):
            for bb_b in by_text.get(tb, []):
                x0, y0 = max(bb_a.x0, bb_b.x0), max(bb_a.y0, bb_b.y0)
                x1, y1 = min(bb_a.x1, bb_b.x1), min(bb_a.y1, bb_b.y1)
                if x1 <= x0 or y1 <= y0:
                    continue
                fig.patches.append(Rectangle(
                    (x0 - 2, y0 - 2), (x1 - x0) + 4, (y1 - y0) + 4,
                    transform=None, facecolor="red", alpha=0.30,
                    edgecolor="red", linewidth=1.2, zorder=10_000))
                fig.text(x1 + 3, y1, str(i), transform=None, fontsize=9,
                         color="red", fontweight="bold", zorder=10_001)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"\n  marked -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
