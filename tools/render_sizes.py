#!/usr/bin/env python3
"""Render one figure at several font sizes from a single data load, and save
each so they can be compared by eye.

    python tools/render_sizes.py --figure robustness --sizes 44 60 76 --out-dir /tmp/x
"""
from __future__ import annotations
import argparse, contextlib, io, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from persistence.plot_utils import apply_oi_style, find_collisions  # noqa: E402
from tools.calibrate_figure_fonts import _load, FIGURES            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="combined")
    ap.add_argument("--figure", default="robustness", choices=sorted(FIGURES))
    ap.add_argument("--sizes", type=float, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    import importlib
    mod, fn = FIGURES[a.figure]
    module = importlib.import_module(mod)
    builder = getattr(module, fn)
    # A builder that pins its own base size (so a pipeline run matches a sweep)
    # would otherwise ignore --sizes. Drive its constant instead.
    pinned = [k for k in vars(module)
              if k.endswith("_FS") and isinstance(vars(module)[k], (int, float))]
    data, dm, stats = _load(a.ref)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    real_close = plt.close
    print(f"\n  {'base pt':>8}{'canvas':>16}{'printed':>9}{'overlaps':>10}")
    for fs in a.sizes:
        apply_oi_style(base=fs)
        for k in pinned:
            setattr(module, k, float(fs))
        plt.close = lambda *x, **k: None
        try:
            _cap = io.StringIO()
            with contextlib.redirect_stdout(_cap):
                builder(data, dm, stats)
            for _ln in _cap.getvalue().splitlines():
                if "[fit]" in _ln or "[layout]" in _ln or "[forest]" in _ln:
                    print("   " + _ln.strip())
            figs = [plt.figure(n) for n in plt.get_fignums()]
        finally:
            plt.close = real_close
        if not figs:
            print(f"  {fs:>8}  no figure"); continue
        fig = max(figs, key=lambda f: len(f.axes))
        w, h = fig.get_size_inches()
        hits = find_collisions(fig)
        out = a.out_dir / f"{a.figure}_fs{fs:g}.png"
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"  {fs:>8}{f'{w:.1f} x {h:.1f} in':>16}"
              f"{(fs-3)*6.3/w:>9.1f}{len(hits):>10}")
        one = lambda t: " ".join(t.split())[:34]
        import collections
        _sz = collections.Counter()
        for _i, _ax in enumerate(fig.axes):
            for _t in list(_ax.get_xticklabels()) + list(_ax.get_yticklabels()):
                if _t.get_visible() and _t.get_text().strip():
                    _sz[round(_t.get_fontsize(), 1)] += 1
        print("           tick sizes in use: " +
              ", ".join(f"{k}pt x{v}" for k, v in sorted(_sz.items())))
        for ta, tb, frac in hits[:8]:
            print(f"           {frac:>5.0%}  {one(ta)!r} x {one(tb)!r}")
        for f in figs:
            real_close(f)
    print(f"\n  -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
