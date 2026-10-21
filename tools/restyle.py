#!/usr/bin/env python3
"""Re-draw cached figures with a new layout, without recomputing anything.

savefig_dual stores each figure it writes under `.figcache/` beside the image.
This loads those and puts them back through the layout passes, so a change to
type size, margins, tick counts, grids or legend placement costs seconds
instead of a section re-run.

    python tools/restyle.py --all                 # every cached figure
    python tools/restyle.py --manifest            # just the Additional Figures
    python tools/restyle.py --check               # can each one be reloaded?
"""
from __future__ import annotations

import argparse, importlib.util as ilu, pickle, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from persistence import plot_utils as pu                         # noqa: E402


def _fs():
    spec = ilu.spec_from_file_location("figure_sets", ROOT / "tools/figure_sets.py")
    m = ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def cached_path(png: Path) -> Path:
    return png.parent / pu.FIGCACHE_DIRNAME / (png.stem + ".pkl")


def restyle_one(pkl: Path, base: float = None, dpi: int = 300,
                write: bool = True):
    """Reload a figure, re-run the layout passes, write it back."""
    with pkl.open("rb") as f:
        blob = pickle.load(f)
    fig, spec = blob["figure"], blob["annotations"]
    was = blob.get("base_fs") or pu.BASE_FS
    if base is not None:
        pu.apply_oi_style(base=base)
        for ax in fig.axes:
            for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
                t.set_fontsize(max(pu.MIN_FS, base - 3))
    # The cached figure holds panels and data only. Everything written on it is
    # rebuilt here at the size the current style asks for, which is the point:
    # the annotation layer is the part that keeps needing to move, so it must
    # not be inherited.
    scale = (base / was) if (base and was) else 1.0
    pu.apply_annotations(fig, spec, scale=scale)
    out = pkl.parent.parent / (pkl.stem + ".png")
    if write:
        # Straight through the ordinary save, so the restyled figure gets the
        # same margin, fitting, legend and coherence passes as a fresh one.
        pu.savefig_dual(fig, str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--base", type=float, default=None)
    ap.add_argument("--ref", default="combined")
    a = ap.parse_args()

    if a.manifest or a.check:
        fs = _fs()
        targets = [(slot, fs.resolve(ref, stem))
                   for slot, ref, stem in fs.manifest()]
    else:
        targets = [(None, p) for p in (ROOT / "results").rglob("*.png")
                   if pu.FIGCACHE_DIRNAME not in str(p)]

    have = miss = fail = 0
    t0 = time.time()
    for slot, png in targets:
        pkl = cached_path(png)
        tag = f"{slot:<5}" if slot else ""
        if not pkl.exists():
            miss += 1
            if a.check:
                print(f"  {tag}{png.stem[:46]:<48} NO CACHE")
            continue
        if a.check:
            try:
                with pkl.open("rb") as f:
                    b = pickle.load(f)
                fg, sp = b["figure"], b["annotations"]
                n = len(fg.axes)
                n_ann = (sum(len(r["texts"]) + len(r["annotations"])
                             + (1 if r["legend"] else 0) for r in sp["axes"])
                         + len(sp["figtexts"]))
                plt.close(fg)
                print(f"  {tag}{png.stem[:46]:<48} ok ({n} axes, "
                      f"{n_ann} annotations, {pkl.stat().st_size/1e6:.1f} MB)")
                have += 1
            except Exception as exc:
                fail += 1
                print(f"  {tag}{png.stem[:46]:<48} UNREADABLE {exc}")
            continue
        try:
            restyle_one(pkl, base=a.base)
            have += 1
        except Exception as exc:
            fail += 1
            print(f"  {tag}{png.stem[:46]:<48} FAILED {exc}")
    print(f"\n  {have} usable, {miss} not cached, {fail} failed "
          f"({time.time()-t0:.1f}s)")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
