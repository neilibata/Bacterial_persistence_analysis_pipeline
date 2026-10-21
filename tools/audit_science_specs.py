#!/usr/bin/env python3
"""Check every shipped figure against the Science figure specification.

One rule per check, pass or fail per figure, so compliance is something the
repository can answer rather than something someone remembers looking at.
The rules are quoted from the guidelines in the docstrings below.

Text sizes come from tools/figure_text_inventory.csv, which
tools/collect_figure_text.py writes by redrawing each figure from its
published data object and reading the artists after the save. Everything else
is measured from the saved PNG or from the figure's own layout report.

    python tools/audit_science_specs.py            # all figures
    python tools/audit_science_specs.py --skip S1  # exclude one

Writes docs/science_compliance.md.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import figure_sets                      # noqa: E402

Image.MAX_IMAGE_PIXELS = None

INVENTORY = ROOT / "tools" / "figure_text_inventory.csv"
OUT_MD = ROOT / "docs" / "science_compliance.md"

# "The width of figures, when printed, will usually be 5.7 cm, 12.1 cm, or
# 18.4 cm." A figure authored far from the width it prints at has its rules
# and symbols reduced with it.
COLUMN_IN = (2.24, 4.76, 7.24)
PRINT_IN = 7.24

# "a reduced size of about 7 points high, and not smaller than 5 points"
TYPE_MIN, TYPE_NOMINAL, TYPE_MAX = 5.0, 7.0, 9.5
# "Line widths should be legible upon reduction (minimum of 0.5 point)"
LINE_MIN = 0.5
# "Size symbols so that they will be distinguishable when the figure is
# reduced (6 point minimum)"
SYMBOL_MIN = 6.0


def slot_num(slot):
    m = re.sub(r"\D", "", slot)
    return int(m) if m else 999


def load_type_sizes():
    """Printed point size per figure, from the final (post-save) harvest."""
    by = defaultdict(list)
    if not INVENTORY.exists():
        return by
    with INVENTORY.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("phase") != "final":
                continue
            try:
                by[r["slot"]].append((float(r["printed_pt"]), r["role"],
                                      r["text"]))
            except (ValueError, KeyError):
                continue
    return by


def thinnest_rule(png):
    """The thinnest rule the figure draws, in authored points.

    "Rules" are the lines a reader has to read as lines: the axis spines, the
    grid, and the plotted curves. Marker edge widths are deliberately left
    out -- they belong to a symbol, which the 6 pt symbol rule covers, and
    the shared style draws hollow markers at 0.15 pt on purpose
    (plot_utils.PRINTED_LW). A width of exactly 0 means no line is drawn at
    all, not a hairline, so it is skipped too.

    Measured from the figure object savefig_dual caches beside every image
    (plot_utils._cache_figure), which is the only per-figure record of what
    was drawn: a PNG does not carry line widths and the text inventory holds
    only type. Returns (pt, what) or (None, "NOT MEASURED - why") -- never a
    pass by default, because an absent measurement is not compliance.
    """
    cache = png.parent / ".figcache" / (png.stem + ".pkl")
    if not cache.exists():
        return None, "NOT MEASURED - no .figcache beside the image"
    try:
        import pickle

        import matplotlib
        matplotlib.use("Agg")
        with cache.open("rb") as fh:
            fig = pickle.load(fh)["figure"]
    except Exception as exc:                       # noqa: BLE001
        why = f"{type(exc).__name__}: {exc}"
        return None, f"NOT MEASURED - cache unreadable ({why[:90]})"

    widths = []
    for ax in fig.axes:
        for sp in ax.spines.values():
            if sp.get_visible():
                widths.append((float(sp.get_linewidth()), "spine"))
        for axis in (ax.xaxis, ax.yaxis):
            for gl in axis.get_gridlines():
                if gl.get_visible():
                    widths.append((float(gl.get_linewidth()), "grid line"))
        for ln in ax.get_lines():
            if (ln.get_visible()
                    and str(ln.get_linestyle()).lower() not in ("none", "")):
                widths.append((float(ln.get_linewidth()), "curve"))
    widths = [(w, k) for w, k in widths if w > 0.0]
    if not widths:
        return None, "NOT MEASURED - the figure draws no rules"
    return min(widths, key=lambda x: x[0])


def check(slot, ref, stem, sizes):
    """Every rule, for one figure. Returns {rule: (ok, detail)}."""
    out = {}
    png = ROOT / "results" / ref / (stem + ".png")
    if not png.exists():
        return {"figure on disk": (False, "not found")}

    # The dpi the figure was actually written at, not an assumed 300: the set
    # spans 180 to 360, and dividing every file by 300 mis-sized eight of them.
    with Image.open(png) as im:
        dpi = im.info.get("dpi")
        dpi = float(dpi[0]) if dpi else 300.0
        w_in = im.size[0] / dpi

    # "please save images at a resolution of 300 dots per inch (dpi) for
    # initial submission"
    out["at least 300 dpi"] = (dpi >= 299.0, f"{dpi:.0f} dpi")

    # Width: authored at a column width, so nothing is reduced on the page.
    near = min(COLUMN_IN, key=lambda c: abs(c - w_in))
    # Only exceeding the widest column matters: such a figure is reduced on
    # the page, and the reduction takes its rules and symbols with it. A
    # figure narrower than a column is placed as drawn, or enlarged.
    out["within the printed width"] = (
        w_in <= PRINT_IN + 0.10,
        f"{w_in:.2f} in (nearest column {near})")

    # Reduction: what the figure's thinnest rule becomes on the page.
    #
    # Two defects this is written against. `red = PRINT_IN / w_in` assumed
    # every figure prints at the full 7.24 in page width, so a figure authored
    # at 4.76 in for the 12.1 cm column was reported ENLARGED (x1.52) when it
    # is placed as drawn (x1.00), and one authored at 2.40 in came out x3.02 --
    # the sign of the error backwards, which hides exactly the thin rules this
    # rule exists to find. A figure is reduced to the column it is placed in,
    # which is the nearest one; tools/audit_science.py:38 models the placement
    # the same way.
    #
    # And asserting only `red >= 0.63` (a NOMINAL 0.8 pt spine staying above
    # 0.5 pt) is strictly implied by "within the printed width" above: it
    # could not fail unless that rule had already failed, so the column
    # co-authors read as an independent pass was a restatement of its
    # neighbour, and no figure's rules were ever actually measured. Measure
    # the thinnest rule the figure draws, and reduce THAT.
    red = near / w_in
    lw, kind = thinnest_rule(png)
    if lw is None:
        out["rules survive reduction"] = (None, kind)
    else:
        out["rules survive reduction"] = (
            lw * red >= LINE_MIN - 0.005,
            f"thinnest {kind} {lw:.2f} pt, reduced x{red:.2f}, prints at "
            f"{lw * red:.2f} pt (floor {LINE_MIN})")

    tp = sizes.get(slot, [])
    if tp:
        vals = [v for v, r, _t in tp if r != "marker"]
        txt = [x for x in tp if x[1] != "marker"]
        smallest = min(txt, key=lambda x: x[0]) if txt else (0, "", "")
        largest = max(txt, key=lambda x: x[0]) if txt else (0, "", "")
        out["type at or above 5 pt"] = (
            min(vals) >= TYPE_MIN - 0.05,
            f"smallest {smallest[0]:.1f} pt ({smallest[1]})")
        out["type not oversized"] = (
            max(vals) <= TYPE_MAX + 0.05,
            f"largest {largest[0]:.1f} pt ({largest[1]})")
        out["type varies little within figure"] = (
            (max(vals) - min(vals)) <= 4.0,
            f"spread {max(vals) - min(vals):.1f} pt")
        out["no title on the figure"] = (
            not any(r == "suptitle" for _v, r, _t in tp), "")

        # "Size symbols so that they will be distinguishable when the figure
        # is reduced (6 point minimum)."
        marks = [(v, t) for v, r, t in tp if r == "marker"]
        if marks:
            worst = min(marks, key=lambda x: x[0])
            out["symbols at or above 6 pt"] = (
                worst[0] >= SYMBOL_MIN - 0.05,
                f"smallest {worst[0]:.2f} pt ({worst[1]})")
        else:
            out["symbols at or above 6 pt"] = (None, "no symbols plotted")
    else:
        for k in ("type at or above 5 pt", "type not oversized",
                  "type varies little within figure", "no title on the figure",
                  "symbols at or above 6 pt"):
            out[k] = (None, "NOT MEASURED - figure absent from the harvest")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", action="append", default=[],
                    help="slot to exclude, e.g. S1")
    a = ap.parse_args()

    sizes = load_type_sizes()
    rows = []
    for slot, ref, stem in sorted(figure_sets.manifest(),
                                  key=lambda r: slot_num(r[0])):
        if slot in a.skip:
            continue
        rows.append((slot, Path(stem).name, check(slot, ref, stem, sizes)))

    rules = []
    for _s, _n, res in rows:
        for k in res:
            if k not in rules:
                rules.append(k)

    L = ["# Science specification compliance\n",
         f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by "
         "`tools/audit_science_specs.py`.\n"]
    if a.skip:
        L.append(f"Excluded: {', '.join(a.skip)}.\n")
    L.append("| Figure | " + " | ".join(rules) + " |")
    L.append("|" + "---|" * (len(rules) + 1))
    fails = defaultdict(list)
    for slot, name, res in rows:
        cells = []
        for r in rules:
            ok, _d = res.get(r, (None, ""))
            cells.append("pass" if ok else ("—" if ok is None else "**FAIL**"))
            if ok is False:
                fails[r].append(slot)
        L.append(f"| {slot} | " + " | ".join(cells) + " |")

    unknown = defaultdict(list)
    for slot, _n, res in rows:
        for r, (ok, _d) in res.items():
            if ok is None and "NOT MEASURED" in str(_d):
                unknown[r].append(slot)
    if unknown:
        L.append("\n> **Not measured.** A dash below is an absent measurement, "
                 "not a pass. " +
                 "; ".join(f"{r}: {' '.join(v)}" for r, v in unknown.items()) +
                 "\n")

    L.append("\n## What failed, and why\n")
    if not fails:
        L.append("Nothing. Every figure passes every rule checked here.\n")
    for rule, slots in fails.items():
        L.append(f"### {rule} — {len(slots)} figures\n")
        for slot, name, res in rows:
            if slot in slots:
                L.append(f"- **{slot}** (`{name}`): {res[rule][1]}")
        L.append("")
    OUT_MD.write_text("\n".join(L))

    print(f"  {OUT_MD}")
    n = len(rows)
    for rule in rules:
        bad = len(fails.get(rule, []))
        unk = sum(1 for _s, _nm, res in rows
                  if res.get(rule, (None, ""))[0] is None)
        line = f"  {rule:34} {n - bad - unk}/{n} pass"
        if bad:
            line += f"   fail: {' '.join(fails[rule])}"
        if unk:
            line += f"   ({unk} not measured)"
        print(line)


if __name__ == "__main__":
    main()
