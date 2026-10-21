#!/usr/bin/env python3
"""Redraw a main-text figure from its published data object.

The supplementary figures each rebuild from one pickle (tools/replot_from_data.py);
this is the same contract for the five main-text figures. A figure that cannot
be redrawn from its data object is not reproducible from the data the journal
publishes, so this doubles as the check that each object is complete: if a
panel comes back empty or a number is missing, the object was a summary rather
than a record.

    python tools/replot_main.py --list
    python tools/replot_main.py F5
    python tools/replot_main.py --all --out /tmp/check

Without --out the figure is rewritten in place, exactly as its generator would.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures"))

OUT_DIR = ROOT / "main_text_figures"


def _obj(slot):
    from persistence import figure_source
    return figure_source.load(str(OUT_DIR), slot)


def _tables(obj):
    t = dict(obj.get("panels", {}))
    t["_extras"] = obj.get("extras", {})
    return t


def draw_F1(out_dir):
    import make_metadata_equal_earth as m
    obj = _obj("F1")
    grp = obj["panels"]["countries"]
    m.plot_world_map_B_eqearth(None, os.path.join(out_dir, "figure1"), grp=grp,
                               show_title=False)


def draw_F2(out_dir):
    import figure2 as m
    obj = _obj("F2")
    m.build_figure2(s1_dir="", s3_dir="", ref=obj.get("reference", "UHGG"),
                    spec_tag="10pct",
                    out_path=os.path.join(out_dir, "figure2.png"),
                    tables=_tables(obj))


def draw_F3(out_dir):
    # Figure 3's generator already rebuilds from its own data object; it owns
    # the fitting, so it is driven through its documented entry point.
    import make_combined_figure3 as m
    obj = _obj("F3")
    built = m.rebuild_from_data_object(obj, Path(out_dir))

    # ...but it names its output figure3_<model>.png, while the shipped
    # main-text file is figure3.png -- figures/figure3.py copies the build
    # output into place as its last step. Without the same copy here,
    # `replot_main --all` refreshed four of the five main figures and left
    # figure3.png untouched, which is exactly the gap a "regenerated from the
    # pipeline" claim should not have. (The content matched, so nothing was
    # wrong with the figure; the redraw just wasn't reaching it.)
    import shutil
    built = Path(built)
    for src, dst in ((built, Path(out_dir) / "figure3.png"),
                     (built.with_suffix(".pdf"),
                      Path(out_dir) / "figure3.pdf")):
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
            print(f"  copied {src.name} -> {dst.name}")
    return built


def draw_F4(out_dir):
    import section2_bayesian_figure as m
    obj = _obj("F4")
    m.rebuild_from_data_object(obj, Path(out_dir))


def draw_F5(out_dir):
    import figure5 as m
    obj = _obj("F5")
    m.build_figure5(results_dir="", ref=obj.get("reference", "combined"),
                    spec_tag="10pct",
                    out_path=os.path.join(out_dir, "figure5.png"),
                    tables=_tables(obj))


DRAW = {"F1": draw_F1, "F2": draw_F2, "F3": draw_F3, "F4": draw_F4,
        "F5": draw_F5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slot", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--figures-dir", default=None,
                    help="main_text_figures/ to read the data objects from and, "
                         "without --out, redraw into (e.g. a finished run's package)")
    a = ap.parse_args()
    global OUT_DIR
    if a.figures_dir:
        OUT_DIR = Path(a.figures_dir).expanduser().resolve()

    if a.list:
        from persistence import figure_source
        for s in sorted(DRAW):
            p = Path(figure_source.path_for(str(OUT_DIR), s))
            print(f"  {s}: {'present' if p.exists() else 'NO DATA OBJECT'}  {p if p.exists() else ''}")
        return 0

    slots = sorted(DRAW) if a.all else ([a.slot] if a.slot else [])
    if not slots:
        ap.error("give a slot, or --all")
    out_dir = a.out or str(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    bad = 0
    for slot in slots:
        # One process each: matplotlib style state carries between figures,
        # which is how a redraw silently picks up the previous figure's rcParams.
        if a.all:
            r = subprocess.run([sys.executable, __file__, slot, "--out", out_dir,
                                "--figures-dir", str(OUT_DIR)], cwd=str(ROOT))
            bad += r.returncode != 0
            continue
        try:
            DRAW[slot](out_dir)
            print(f"  {slot}: redrawn -> {out_dir}")
        except Exception as exc:
            print(f"  {slot}: FAILED {type(exc).__name__}: {exc}")
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
