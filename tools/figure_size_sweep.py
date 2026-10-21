#!/usr/bin/env python3
"""Measure every figure a section produces, in one process, one data load.

The per-figure calibration harness needs a builder with a known signature, and
most of these figures are written from inside a section entry point rather than
by a builder anyone calls directly. This runs the entry points instead and
intercepts the save, so each figure is measured exactly as the pipeline draws
it -- canvas size, printed point size at page width, and text collisions.

    python tools/figure_size_sweep.py --ref combined --sections depth_audit ipw
    python tools/figure_size_sweep.py --ref combined --sections all --base 10
"""
from __future__ import annotations

import argparse, contextlib, io, os, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402

from persistence import config, plot_utils                      # noqa: E402
from persistence.plot_utils import (                            # noqa: E402
    apply_oi_style, find_collisions, PAGE_WIDTH_IN)
from persistence import plotdata                                # noqa: E402

MEASURED: list = []


def _install_probe():
    """Wrap savefig_dual so every figure it writes is measured first."""
    real = plot_utils.savefig_dual

    def probe(fig, out_path, dpi=300, **kw):
        res = real(fig, out_path, dpi=dpi, **kw)
        try:
            w_in = float(fig.get_size_inches()[0])
            try:
                saved_in = float(fig.get_tightbbox(
                    fig.canvas.get_renderer()).width)
            except Exception:
                saved_in = w_in
            base = plot_utils.current_fs()
            hits = find_collisions(fig)
            vis = [a for a in fig.axes if a.get_visible()]
            n_ax = len(vis)
            r = fig.canvas.get_renderer()

            # --- the standards, checked rather than assumed -----------------
            aspects, grids, ticks, outside, markers = [], 0, 0, 0, 0.0
            for ax in vis:
                bb = ax.get_window_extent(renderer=r)
                if bb.height > 0:
                    aspects.append(bb.width / bb.height)
                if any(gl.get_visible() for gl in
                       (ax.get_xgridlines() + ax.get_ygridlines())):
                    grids += 1
                for axis in (ax.xaxis, ax.yaxis):
                    # A categorical axis has one tick per category by
                    # definition; capping it would delete rows of the chart.
                    # Only auto-located axes are the tick-count question.
                    if isinstance(axis.get_major_locator(),
                                  matplotlib.ticker.FixedLocator):
                        continue
                    lo, hi = ((ax.get_xlim(), ax.get_ylim())
                              [axis is ax.yaxis])
                    lo, hi = min(lo, hi), max(lo, hi)
                    n = sum(1 for t in axis.get_majorticklocs()
                            if lo <= t <= hi)
                    ticks = max(ticks, n)
                for t in list(ax.texts) + ([ax.get_legend()]
                                           if ax.get_legend() else []):
                    # A label the figure deliberately hid is not on the page.
                    if not t.get_visible():
                        continue
                    try:
                        tb = t.get_window_extent(renderer=r)
                    except Exception:
                        continue
                    # Leaving the axes is not the fault; panel letters sit just
                    # outside the corner on purpose, and counting those flagged
                    # the one figure that had been hand-checked all afternoon.
                    # The fault is landing on a *different* panel, or leaving
                    # the figure. That is what "outside" should mean.
                    if (tb.x1 < 0 or tb.y1 < 0 or
                            tb.x0 > fig.bbox.width or tb.y0 > fig.bbox.height):
                        outside += 1
                        continue
                    for other in vis:
                        if other is ax:
                            continue
                        ob = other.get_window_extent(renderer=r)
                        # A twinx/twiny occupies its parent's rectangle, so
                        # every text in the parent would read as landing on
                        # another panel. Same rectangle means same panel.
                        if (abs(ob.x0 - bb.x0) < 2 and abs(ob.x1 - bb.x1) < 2
                                and abs(ob.y0 - bb.y0) < 2
                                and abs(ob.y1 - bb.y1) < 2):
                            continue
                        if not (tb.x1 <= ob.x0 or tb.x0 >= ob.x1 or
                                tb.y1 <= ob.y0 or tb.y0 >= ob.y1):
                            outside += 1
                            break
                for coll in ax.collections:
                    try:
                        sz = coll.get_sizes()
                        if len(sz):
                            markers = max(markers, float(max(sz)))
                    except Exception:
                        pass
            import statistics as _st
            aspect = round(_st.median(aspects), 2) if aspects else 0.0
            MEASURED.append(dict(
                name=os.path.basename(os.path.splitext(out_path)[0]),
                panels=n_ax, canvas_in=round(w_in, 1),
                saved_in=round(saved_in, 1), base_pt=round(base, 1),
                printed_pt=round((base - 3) * PAGE_WIDTH_IN / max(saved_in, 1e-6), 1),
                overlaps=len(hits),
                aspect=aspect, grids=grids, max_ticks=ticks,
                aspect_exempt=bool(getattr(fig, "_aspect_exempt", False)),
                outside=outside, max_marker=round(markers, 0),
                worst=[(f, " ".join(a.split())[:30], " ".join(b.split())[:30])
                       for a, b, f in hits[:3]],
            ))
        except Exception as exc:
            MEASURED.append(dict(name=os.path.basename(str(out_path)),
                                 panels=0, canvas_in=0, saved_in=0, base_pt=0,
                                 printed_pt=0, overlaps=-1,
                                 worst=[("probe failed", str(exc)[:60], "")]))
        return res

    # Every module imported savefig_dual by name, so patch each binding too.
    plot_utils.savefig_dual = probe
    import importlib
    for m in list(sys.modules.values()):
        if getattr(m, "__name__", "").startswith(("persistence", "figures")):
            if getattr(m, "savefig_dual", None) is real:
                m.savefig_dual = probe


def _taxon_lists(data):
    """VANISH+/BloSSUM+ as section 2 left them on disk.

    Sections 2E-2G and section 3 take these as arguments rather than deriving
    them, which is why their figures were out of reach of this harness. The
    core section 2 differential analysis writes them out, so read them back
    instead of recomputing an hour of statistics to redraw a figure.
    """
    from persistence import config as _cfg
    d = _cfg.section_dir(data.ref_name, "s2", "main")
    valid = set(data.breadth.columns)
    out = {}
    for key, fname in (("vanish_plus", "vanish_plus_ids.txt"),
                       ("blossum_plus", "blossum_plus_ids.txt"),
                       ("vanish_h", "vanishH_ids.txt"),
                       ("blossum_h", "blossumH_ids.txt")):
        path = os.path.join(d, fname)
        ids = []
        if os.path.exists(path):
            with open(path) as f:
                ids = [l.strip() for l in f if l.strip() and l.strip() in valid]
        out[key] = ids
    print(f"  taxon lists: VANISH+={len(out['vanish_plus'])} "
          f"BloSSUM+={len(out['blossum_plus'])} "
          f"VanishH={len(out['vanish_h'])} BlossumH={len(out['blossum_h'])}",
          flush=True)
    return out


def _run_s2(d, dm, s):
    from persistence.section2_pooling import run_section2_pooling
    t = _taxon_lists(d)
    if not t["vanish_plus"] and not t["blossum_plus"]:
        raise RuntimeError("no VANISH+/BloSSUM+ lists on disk; run --only s2 first")
    return run_section2_pooling(d, t["vanish_plus"], t["blossum_plus"], s)


def _run_s3(d, dm, s):
    from persistence.section3_health import run_section3
    t = _taxon_lists(d)
    if not t["vanish_plus"] and not t["blossum_plus"]:
        raise RuntimeError("no VANISH+/BloSSUM+ lists on disk; run --only s2 first")
    return run_section3(d, dm, t["vanish_plus"], t["blossum_plus"], s,
                        vanish_h_ids=t["vanish_h"] or None,
                        blossum_h_ids=t["blossum_h"] or None)


def _run_s5(d, dm, s):
    """A21: fig_s5b_overlap_analysis, the pairwise overlap among taxon lists."""
    from persistence.supp_s5 import run_supp_s5
    t = _taxon_lists(d)
    return run_supp_s5(d, t["vanish_plus"], t["blossum_plus"],
                       t["vanish_h"] or None, t["blossum_h"] or None)


SECTIONS = {
    "depth_audit": lambda d, dm, s: __import__(
        "persistence.depth_audit", fromlist=["x"]).run_depth_audit(d, s),
    "ipw_richness": lambda d, dm, s: __import__(
        "persistence.depth_bias_ipw", fromlist=["x"]).run_gmw_richness(d, s),
    "ipw_pairwise": lambda d, dm, s: __import__(
        "persistence.depth_bias_ipw", fromlist=["x"]
        ).run_geomean_richness_pairwise(d, s),
    "ipw_sweep": lambda d, dm, s: __import__(
        "persistence.depth_bias_ipw", fromlist=["x"]).run_gmw_threshold_sweep(d, s),
    "tech_reps": lambda d, dm, s: __import__(
        "persistence.section1_tech_replicates", fromlist=["x"]
        ).run_tech_replicate_robustness(d, s),
    "carter": lambda d, dm, s: __import__(
        "persistence.carter_analysis", fromlist=["x"]).run_carter_analysis(d, s),
    "axis_angle": lambda d, dm, s: __import__(
        "persistence.axis_angle_analysis", fromlist=["x"]).run_axis_angle(d, s),
    # The rest of the Additional Figures. Between them these close the manifest:
    # A15/A17a/A17b from the accumulation curves, A20 from the depth-binned
    # richness CV, A5/A13 from section 1, A1 from the roadmap script.
    "accumulation": lambda d, dm, s: __import__(
        "persistence.accumulation", fromlist=["x"]).run_figure3_accumulation(d),
    "richness_cv": lambda d, dm, s: __import__(
        "persistence.within_group_stats", fromlist=["x"]
        ).run_depth_binned_richness_cv(d, s),
    "supp_s5": _run_s5,
    "s2_pooling": _run_s2,
    "s3_health": _run_s3,
    "s1_figures": lambda d, dm, s: __import__(
        "persistence.section1_figure", fromlist=["x"]).run_section1_figure(d, dm, s),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="combined")
    ap.add_argument("--sections", nargs="+", default=["depth_audit"],
                    help="names from SECTIONS, or 'all'")
    ap.add_argument("--base", type=float, default=None,
                    help="base type size to force; default is each figure's own")
    ap.add_argument("--no-cache", action="store_true",
                    help="reload the reference instead of reusing the cache")
    a = ap.parse_args()

    names = list(SECTIONS) if a.sections == ["all"] else a.sections
    # SECTIONS calls the real section entry points, which emit each figure's
    # data object as a side effect. This is a measuring pass, and it runs on a
    # fresh, empty statistics registry from plotdata.reference(), so letting it
    # write would replace the run's submitted data with a thinner copy --
    # silently, because the calls below redirect stdout.
    os.environ.setdefault("FIGURE_DATA_READONLY", "1")
    # Reuse computed inputs as well as the load. This is a formatting tool;
    # the pipeline itself never turns this on.
    plotdata.enable(not a.no_cache)
    data, dm, stats = plotdata.reference(a.ref, use_cache=not a.no_cache)
    _install_probe()

    for n in names:
        if n not in SECTIONS:
            print(f"  ?? unknown section {n}"); continue
        if a.base:
            apply_oi_style(base=a.base)
        print(f"\n  === {n} ===", flush=True)
        before = len(MEASURED)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                SECTIONS[n](data, dm, stats)
        except Exception:
            print(f"  !! {n} raised:")
            print("     " + traceback.format_exc().strip().splitlines()[-1])
        print(f"  {len(MEASURED)-before} figure(s)")

    print(f"\n{'figure':<40}{'pan':>4}{'canvas':>8}{'saved':>7}"
          f"{'base':>6}{'printed':>8}{'overlaps':>9}{'aspect':>8}"
          f"{'grids':>7}{'ticks':>6}{'outside':>8}{'marker':>8}")
    for m in MEASURED:
        print(f"{m['name'][:39]:<40}{m['panels']:>4}{m['canvas_in']:>8}"
              f"{m['saved_in']:>7}{m['base_pt']:>6}{m['printed_pt']:>8}"
              f"{m['overlaps']:>9}{m.get('aspect',0):>8}{m.get('grids',0):>7}"
              f"{m.get('max_ticks',0):>6}{m.get('outside',0):>8}"
              f"{m.get('max_marker',0):>8}")

    # --- the standards, as pass/fail --------------------------------------
    def _bad(m):
        why = []
        if m["overlaps"] > 0:              why.append(f"{m['overlaps']} overlaps")
        if (not m.get("aspect_exempt") and
                not (0.8 <= m.get("aspect", 0) <= 1.25)):
            why.append(f"aspect {m.get('aspect')}")
        if m.get("grids", 0):              why.append(f"grids on {m['grids']} axes")
        if m.get("max_ticks", 0) > 6:      why.append(f"{m['max_ticks']} ticks")
        if m.get("outside", 0):            why.append(f"{m['outside']} outside panel")
        if m.get("max_marker", 0) > 250:   why.append(f"marker {m['max_marker']:.0f}")
        if m["printed_pt"] < 5:            why.append(f"{m['printed_pt']} pt")
        if m["saved_in"] > m["canvas_in"] * 1.05: why.append("canvas grew")
        return why

    fails = [(m, _bad(m)) for m in MEASURED]
    fails = [(m, w) for m, w in fails if w]
    print(f"\n  {len(MEASURED)-len(fails)}/{len(MEASURED)} meet every standard")
    for m, w in fails:
        print(f"    {m['name'][:44]:<46} {'; '.join(w)}")
    print()
    for m in MEASURED:
        if m["overlaps"] > 0:
            print(f"  {m['name']}")
            for f, x, y in m["worst"]:
                print(f"      {f:>5}  {x!r} x {y!r}" if isinstance(f, float)
                      else f"      {f}  {x} {y}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
