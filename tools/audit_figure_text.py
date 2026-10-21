#!/usr/bin/env python3
"""Measure whether a figure's text will be legible at its placement size.

Works on rendered PNGs, so it tests exactly what goes into the document rather
than what the plotting code intended. For each image it reports:

  pt10 / pt50 / pt90  glyph height in points at the placement width. The 10th
                      percentile is the number that matters -- tick labels are
                      the smallest text and the first to become unreadable.
  span               pt90 / pt10, how far the type sizes spread inside one
                      figure. A figure whose annotations are 4x its ticks reads
                      as unbalanced however legible each part is.
  ink                fraction of the canvas carrying any mark. Very low values
                      mean the layout is mostly blank -- usually a symptom of
                      text forcing panels apart.

Run over a set of figures, the useful signal is the spread of pt50 *between*
figures: that is the consistency the eye picks up when flipping through a
supplement.

    python tools/audit_figure_text.py staged_figures/*.png
    python tools/audit_figure_text.py --width-in 6.3 --target 9 results/combined/**/*_fit_notitle.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image

GLYPH_INK_RATIO = 0.714   # measured; see tools/ calibration note in the docstring

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


def glyph_heights(path: Path, width_in: float) -> np.ndarray:
    """Heights, in points, of text-like connected components."""
    im = Image.open(path).convert("L")
    a = np.array(im)
    h, w = a.shape
    px_per_pt = (w / width_in) / 72.0          # rendered pixels per point
    dark = a < 110
    if ndimage is None:
        return np.array([])
    lab, n = ndimage.label(dark)
    if n == 0:
        return np.array([])
    objs = ndimage.find_objects(lab)
    out = []
    for sl in objs:
        hh = sl[0].stop - sl[0].start
        ww = sl[1].stop - sl[1].start
        if hh < 3 or ww < 2:
            continue                            # noise
        if hh > 0.10 * h:
            continue                            # axes, curves, big blocks
        if ww > 12 * hh:
            continue                            # rules, spines, long lines
        if ww / max(hh, 1) > 6 and hh < 6:
            continue                            # thin ticks
        out.append(hh / px_per_pt)
    g = np.array(out)
    if g.size == 0:
        return g
    # Punctuation and glyph fragments -- commas, the dot of an i, subscripts --
    # are real ink but not a legibility constraint, and they drag any low
    # percentile toward zero. Keep components at least 45% of the median height,
    # which leaves letters and digits at every size actually used.
    g = g[g >= 0.45 * np.median(g)]
    # Connected-component height is the glyph's ink, not its em box. Calibrated
    # against text rendered at known sizes, ink height = 0.714 x nominal point
    # size, so divide through and report numbers a typographer would recognise.
    return g / GLYPH_INK_RATIO


def ink_fraction(path: Path) -> float:
    a = np.array(Image.open(path).convert("L"))
    return float((a < 245).mean())


def audit(path: Path, width_in: float):
    g = glyph_heights(path, width_in)
    if g.size < 20:
        return None
    return dict(
        n=int(g.size),
        pt10=float(np.percentile(g, 10)),
        pt50=float(np.percentile(g, 50)),
        pt90=float(np.percentile(g, 90)),
        span=float(np.percentile(g, 90) / max(np.percentile(g, 10), 1e-6)),
        ink=ink_fraction(path),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--width-in", type=float, default=6.3)
    ap.add_argument("--target", type=float, default=9.0,
                    help="target median glyph size in points")
    ap.add_argument("--floor", type=float, default=6.0,
                    help="minimum acceptable 10th-percentile size")
    args = ap.parse_args()

    if ndimage is None:
        print("scipy is required (pip install scipy)", file=sys.stderr)
        return 2

    rows = []
    for f in args.files:
        r = audit(f, args.width_in)
        if r:
            r["name"] = f.name
            rows.append(r)
    if not rows:
        print("no measurable figures", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["pt50"])
    print(f"  {'figure':<46}{'pt10':>7}{'pt50':>7}{'pt90':>7}{'span':>7}{'ink':>7}  flags")
    print("  " + "-" * 94)
    for r in rows:
        flags = []
        if r["pt10"] < args.floor:            flags.append("TOO SMALL")
        if r["span"] > 3.0:                   flags.append("uneven")
        if r["ink"] < 0.045:                  flags.append("sparse")
        print(f"  {r['name'][:44]:<46}{r['pt10']:>7.1f}{r['pt50']:>7.1f}"
              f"{r['pt90']:>7.1f}{r['span']:>7.1f}{r['ink']:>7.3f}  {', '.join(flags)}")

    med = np.array([r["pt50"] for r in rows])
    print(f"\n  across {len(rows)} figures: median pt50 {np.median(med):.1f}, "
          f"range {med.min():.1f}-{med.max():.1f}, "
          f"spread (max/min) {med.max()/max(med.min(),1e-6):.1f}x")
    bad = [r for r in rows if r["pt10"] < args.floor]
    print(f"  below the {args.floor:.0f} pt floor: {len(bad)} / {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
