#!/usr/bin/env python3
"""Harvest every piece of text in every shipped figure, with its printed size.

Redraws each figure from its published data object and walks the text artists
before the figure is saved. That gives two things the formatting pass needs:

  * the complete inventory of what a figure says, so nothing is deleted from a
    figure without a record of what the caption now has to carry;
  * the size each string will actually print at, so the Science floor (5 pt,
    7 pt nominal) can be checked rather than guessed.

    python tools/collect_figure_text.py --all
    python tools/collect_figure_text.py S12          # one figure

Writes tools/figure_text_inventory.csv.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Science prints multi-panel figures at three columns.
PRINT_WIDTH_IN = 7.24
OUT_CSV = ROOT / "tools" / "figure_text_inventory.csv"

FIELDS = ["slot", "phase", "role", "axes", "text", "pt", "printed_pt",
          "bold", "lines"]


def _replot():
    spec = importlib.util.spec_from_file_location(
        "replot", ROOT / "tools" / "replot_from_data.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def harvest(fig, slot, phase="authored", saved_w_in=None):
    """Every text artist on the figure, tagged by the role it plays."""
    from persistence import plot_utils

    r = None
    try:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
    except Exception:
        pass
    # The width of the file that was actually written, when we have it. The
    # tight bounding box recomputed after a save does not always match what
    # the save produced -- on a figure with artists outside the axes it came
    # back half as wide again, which scaled every printed size down with it
    # and reported type below the floor that was not.
    if saved_w_in:
        saved_w = float(saved_w_in)
    else:
        try:
            saved_w = float(fig.get_tightbbox(r).width)
        except Exception:
            saved_w = float(fig.get_size_inches()[0])
    scale = PRINT_WIDTH_IN / max(saved_w, 1e-6)

    rows = []

    def add(role, artist, ax_i=""):
        try:
            t = artist.get_text()
        except Exception:
            return
        if not t or not str(t).strip():
            return
        pt = float(artist.get_fontsize())
        try:
            bold = str(artist.get_fontweight()) in ("bold", "heavy", "600",
                                                    "700", "800", "900")
        except Exception:
            bold = False
        rows.append(dict(slot=slot, phase=phase, role=role, axes=ax_i,
                         text=str(t).replace("\n", " \\n "),
                         pt=round(pt, 2),
                         printed_pt=round(pt * scale, 2),
                         bold=int(bold),
                         lines=str(t).count("\n") + 1))

    if getattr(fig, "_suptitle", None) is not None:
        add("suptitle", fig._suptitle)
    for t in fig.texts:
        if t is not getattr(fig, "_suptitle", None):
            add("figure_text", t)

    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        add("axes_title", ax.title, i)
        add("xlabel", ax.xaxis.label, i)
        add("ylabel", ax.yaxis.label, i)
        for t in ax.texts:
            add("annotation", t, i)
        leg = ax.get_legend()
        if leg is not None:
            if leg.get_title() is not None:
                add("legend_title", leg.get_title(), i)
            for t in leg.get_texts():
                add("legend_entry", t, i)
        # Tick labels: one row per axis, since they share a size.
        for which, axis in (("xtick", ax.xaxis), ("ytick", ax.yaxis)):
            labs = [l for l in axis.get_ticklabels() if l.get_visible()
                    and str(l.get_text()).strip()]
            if labs:
                rows.append(dict(
                    slot=slot, phase=phase, role=which, axes=i,
                    text=" | ".join(str(l.get_text()) for l in labs[:8]),
                    pt=round(float(labs[0].get_fontsize()), 2),
                    printed_pt=round(float(labs[0].get_fontsize()) * scale, 2),
                    bold=0, lines=1))
    if fig.legends:
        for leg in fig.legends:
            if leg.get_title() is not None:
                add("legend_title", leg.get_title(), "fig")
            for t in leg.get_texts():
                add("legend_entry", t, "fig")
    # Plotted symbols, in the same printed points as the type. Science:
    # "Size symbols so that they will be distinguishable when the figure is
    # reduced (6 point minimum)." A marker is recorded as its diameter;
    # scatter takes an area in points squared, hence the square root.
    import numpy as _np
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        for ln in ax.lines:
            try:
                if ln.get_marker() in (None, "None", "", " ", "none"):
                    continue
                d = float(ln.get_markersize())
            except Exception:
                continue
            if d > 0:
                rows.append(dict(
                    slot=slot, phase=phase, role="marker", axes=i,
                    text=f"line marker {ln.get_marker()!r}",
                    pt=round(d, 2), printed_pt=round(d * scale, 2),
                    bold=0, lines=1))
        for col in ax.collections:
            try:
                sizes = _np.atleast_1d(col.get_sizes())
            except Exception:
                continue
            if not len(sizes):
                continue
            d = float(_np.sqrt(max(float(_np.min(sizes)), 0.0)))
            if d > 0:
                rows.append(dict(
                    slot=slot, phase=phase, role="marker", axes=i,
                    text=f"scatter ({len(sizes)} sizes, smallest)",
                    pt=round(d, 2), printed_pt=round(d * scale, 2),
                    bold=0, lines=1))
    return rows


def run_one(slot, out_json, results_root=None):
    """Redraw one figure and dump its text inventory.

    `results_root` is where the data objects live. It defaults to the
    repo's own results/, which on a working tree is usually a partial
    run; measuring a package means pointing this at that package, the
    same way tools/replot_from_data.py takes --results-root.
    """
    from persistence import plot_utils, figure_source
    from tools import figure_sets

    rp = _replot()
    if slot not in rp.DRAW:
        print(f"  {slot}: no replot wired", file=sys.stderr)
        return 1
    man = {s: (r, st) for s, r, st in figure_sets.manifest()}
    ref, stem = man[slot]
    root = Path(results_root) if results_root else ROOT / "results"
    outdir = str((root / ref / stem).parent)
    obj = figure_source.load(outdir, slot)

    collected = []
    real = plot_utils.savefig_dual

    def probe(fig, out_path, dpi=300, **kw):
        # Twice: once as the figure was authored, which is where the caption
        # ledger gets the titles and second lines from, and once after the
        # save, which is what actually prints -- the title is gone by then and
        # the type band has been enforced.
        try:
            collected.extend(harvest(fig, slot, "authored"))
        except Exception as exc:
            print(f"  {slot}: harvest failed {exc}", file=sys.stderr)
        res = real(fig, out_path, dpi=dpi, **kw)
        saved_w = None
        try:
            from PIL import Image as _I
            _p = str(out_path)
            if not _p.lower().endswith((".png", ".pdf")):
                _p += ".png"
            if _p.lower().endswith(".png") and os.path.exists(_p):
                with _I.open(_p) as _im:
                    saved_w = _im.size[0] / float(dpi)
        except Exception:
            saved_w = None
        try:
            collected.extend(harvest(fig, slot, "final", saved_w_in=saved_w))
        except Exception as exc:
            print(f"  {slot}: final harvest failed {exc}", file=sys.stderr)
        return res

    plot_utils.savefig_dual = probe
    for m in list(sys.modules.values()):
        if getattr(m, "__name__", "").startswith(("persistence", "figures",
                                                  "make_")):
            if getattr(m, "savefig_dual", None) is real:
                m.savefig_dual = probe

    # Four figures save with fig.savefig directly and so never reach the hook
    # above -- they were silently absent from the inventory, and the audit
    # read their missing rows as nothing to report. Catch them at the
    # Figure.savefig level, but only if the hook produced nothing, so figures
    # that do go through savefig_dual are not harvested twice.
    from matplotlib.figure import Figure as _Fig
    _real_savefig = _Fig.savefig

    def _savefig(self, fname, *args, **kw):
        res = _real_savefig(self, fname, *args, **kw)
        if not collected:
            try:
                _p = str(fname)
                _w = None
                if _p.lower().endswith(".png") and os.path.exists(_p):
                    from PIL import Image as _I
                    with _I.open(_p) as _im:
                        _d = _im.info.get("dpi")
                        _w = _im.size[0] / (float(_d[0]) if _d else
                                            float(kw.get("dpi", 300)))
                if _w:
                    collected.extend(harvest(self, slot, "final",
                                             saved_w_in=_w))
            except Exception as exc:
                print(f"  {slot}: fallback harvest failed {exc}",
                      file=sys.stderr)
        return res

    _Fig.savefig = _savefig
    try:
        with tempfile.TemporaryDirectory() as td:
            rp.DRAW[slot](obj, td)
    finally:
        _Fig.savefig = _real_savefig

    # A figure that saves more than once (title / no-title) repeats itself.
    seen, uniq = set(), []
    for row in collected:
        k = (row["phase"], row["role"], row["axes"], row["text"], row["pt"])
        if k not in seen:
            seen.add(k)
            uniq.append(row)
    Path(out_json).write_text(json.dumps(uniq))
    print(f"  {slot}: {len(uniq)} strings")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slot", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--json", help=argparse.SUPPRESS)
    ap.add_argument("--results-root", default=None,
                    help="measure the data objects under this results tree "
                         "instead of the repo's own results/")
    a = ap.parse_args()

    if a.json:
        return run_one(a.slot, a.json, a.results_root)

    rp = _replot()
    slots = (sorted(rp.DRAW, key=lambda s: int(re.sub(r"\D", "", s)))
             if a.all else [a.slot])

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for slot in slots:
            frag = Path(td) / f"{slot}.json"
            # One process each: style state carries between figures.
            cmd = [sys.executable, __file__, slot, "--json", str(frag)]
            if a.results_root:
                cmd += ["--results-root", a.results_root]
            r = subprocess.run(cmd, capture_output=True, text=True)
            sys.stdout.write(r.stdout)
            if frag.exists():
                rows.extend(json.loads(frag.read_text()))
            else:
                print(f"  {slot}: FAILED {r.stderr.strip().splitlines()[-1:]}")

    # Merge, do not replace. Harvesting one slot used to rewrite the whole
    # file with only that slot's rows, so `collect_figure_text.py S32` silently
    # deleted the other 31 figures' measurements -- and the compliance audit
    # then reported them as "not measured" rather than as missing, which reads
    # like a tooling gap instead of data loss.
    keep = []
    if OUT_CSV.exists() and not a.all:
        fresh = {r["slot"] for r in rows}
        with OUT_CSV.open() as fh:
            keep = [r for r in csv.DictReader(fh) if r["slot"] not in fresh]
        if keep:
            print(f"  keeping {len(keep)} rows for "
                  f"{len({r['slot'] for r in keep})} other figure(s)")
    with OUT_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(keep + rows)
    fin = [r for r in rows if r["phase"] == "final"]
    small = [r for r in fin if r["printed_pt"] < 5]
    big = [r for r in fin if r["printed_pt"] > 9.5]
    titles = [r for r in fin if r["role"] == "suptitle"]
    print(f"\n  {len(rows)} strings across {len(slots)} figures -> {OUT_CSV}")
    print(f"  as printed: {len(titles)} titles left on figures, "
          f"{len(small)} strings under 5 pt, {len(big)} over 9.5 pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
