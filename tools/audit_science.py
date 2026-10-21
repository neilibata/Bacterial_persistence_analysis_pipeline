#!/usr/bin/env python3
"""Check figures against the journal's figure guidelines.

Only the mechanically checkable rules. Reads the cached figures, so what is
measured is what was drawn.

    python tools/audit_science.py
"""
from __future__ import annotations

import pickle, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")

# 5.7 cm, 12.1 cm, 18.4 cm
COL_IN = {"1 col": 2.24, "2 col": 4.76, "3 col": 7.24}
TYPE_TARGET, TYPE_FLOOR = 7.0, 5.0
SYMBOL_FLOOR_PT = 6.0        # symbol height at final size
LINE_FLOOR_PT = 0.5
PANEL_LABEL_PT = 10.0


def nearest_column(w_in):
    name, col = min(COL_IN.items(), key=lambda kv: abs(kv[1] - w_in))
    return name, col


def check(pkl: Path):
    with pkl.open("rb") as f:
        blob = pickle.load(f)
    fig, spec = blob["figure"], blob["annotations"]
    w_in = float(fig.get_size_inches()[0])
    col_name, col_in = nearest_column(w_in)
    scale = col_in / w_in                      # what reduction to print at

    sizes, lws, marks = [], [], []
    minor = False
    for ax in fig.axes:
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if t.get_visible() and t.get_text().strip():
                sizes.append(t.get_fontsize())
        for ln in ax.get_lines():
            lws.append(ln.get_linewidth())
            if ln.get_marker() not in (None, "None", ""):
                marks.append(ln.get_markersize())
        for col in ax.collections:
            try:
                s = col.get_sizes()
                if len(s):
                    marks.append(float(max(s)) ** 0.5)   # area -> linear
            except Exception:
                pass
        for axis in (ax.xaxis, ax.yaxis):
            if any(t.get_visible() for t in axis.get_minorticklines()):
                minor = True
    for r in spec["axes"]:
        for k in ("title_fs", "label_fs"):
            if r.get(k):
                sizes.append(r[k])

    bad = []
    if abs(w_in - col_in) > 0.05:
        bad.append(f"width {w_in:.2f} in is not {col_name} ({col_in} in)")
    if sizes:
        lo, hi = min(sizes) * scale, max(sizes) * scale
        if lo < TYPE_FLOOR:
            bad.append(f"type {lo:.1f} pt at print (floor {TYPE_FLOOR})")
        if hi > 0 and lo > 0 and hi / lo > 2.0:
            bad.append(f"type range {lo:.1f}-{hi:.1f} pt (wide variation)")
    if marks and min(marks) * scale < SYMBOL_FLOOR_PT:
        bad.append(f"symbol {min(marks)*scale:.1f} pt (floor {SYMBOL_FLOOR_PT})")
    if lws and min(lws) * scale < LINE_FLOOR_PT:
        bad.append(f"line {min(lws)*scale:.2f} pt (floor {LINE_FLOOR_PT})")
    if minor:
        bad.append("minor tick marks")
    if spec.get("suptitle"):
        bad.append("title inside the figure")
    return w_in, col_name, scale, bad


def main() -> int:
    caches = sorted(ROOT.glob("results/**/.figcache/*.pkl"))
    if not caches:
        print("  no cached figures"); return 0
    print(f"  {'figure':<44}{'width':>7}{'->':>4}{'print':>7}  issues")
    n_bad = 0
    tally = {}
    for pkl in caches:
        try:
            w, col, scale, bad = check(pkl)
        except Exception as exc:
            print(f"  {pkl.stem[:42]:<44} unreadable ({exc})"); continue
        if bad:
            n_bad += 1
            for b in bad:
                tally[b.split()[0]] = tally.get(b.split()[0], 0) + 1
            print(f"  {pkl.stem[:42]:<44}{w:>6.2f}\"{'':>4}{col:>7}  {'; '.join(bad)}")
    print(f"\n  {n_bad} of {len(caches)} figures have at least one issue")
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"    {v:>3}  {k}")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
