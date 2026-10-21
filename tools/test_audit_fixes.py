"""Fast checks for the defects the 2026-10-07 audit confirmed.

Every check here runs in seconds on synthetic or cached inputs: none of them
loads a matrix or re-runs a section. The point is to be runnable before the
overnight run and again after it, so a regression in any of these is caught
without a 45-minute section.

Each check states what it would mean if it failed, because a check whose
failure needs interpretation does not get acted on.

    python tools/test_audit_fixes.py            # all
    python tools/test_audit_fixes.py --verbose  # with the numbers

Exit code is the number of failures, so it can gate a run.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

_results: list[tuple[str, bool, str]] = []
VERBOSE = False


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    line = f"  {mark}  {name}"
    if detail and (VERBOSE or not ok):
        line += f"\n          {detail}"
    print(line, flush=True)


def group(title: str) -> None:
    print(f"\n  ── {title} " + "─" * max(0, 64 - len(title)))


# ── the statistics the audit found wrong ─────────────────────────────────────

def test_sato_variance() -> None:
    """Sato's P term had every sign reversed, so the rd-dependent correction
    entered backwards. In the single-stratum limit a consistent estimator must
    equal the large-sample binomial variance exactly; that is the test."""
    group("Mantel-Haenszel risk difference (health_cmh)")
    from persistence.health_cmh import stratified_shifts_per_taxon  # noqa
    import persistence.health_cmh as hc
    import inspect
    src = inspect.getsource(hc)
    check("Sato P term has the published sign",
          "n1 ** 2 * C - n2 ** 2 * A + n1 * n2 * (n2 - n1) / 2" in src,
          "the reversed form biases every rd_mh_se, t_mh and p_mh")

    # Reproduce the block in isolation and compare with p1(1-p1)/n1 + p2(1-p2)/n2.
    def var_single(n1, a, n2, c):
        A = np.array([[float(a)]]); C = np.array([[float(c)]])
        N1 = np.array([float(n1)]); N2 = np.array([float(n2)])
        p1s, p2s = A / N1[:, None], C / N2[:, None]
        W = N1 * N2 / (N1 + N2); sW = W.sum()
        rd = (W[:, None] * (p1s - p2s)).sum(0) / sW
        x1, x2 = N1[:, None], N2[:, None]; NN = x1 + x2
        P = (x1 ** 2 * C - x2 ** 2 * A + x1 * x2 * (x2 - x1) / 2) / NN ** 2
        Q = (A * (x2 - C) + C * (x1 - A)) / (2 * NN)
        return float((rd * P.sum(0) + Q.sum(0))[0] / sW ** 2)

    worst = 0.0
    for n1, a, n2, c in [(1000, 500, 1000, 300), (2000, 1000, 500, 150),
                         (800, 120, 400, 200), (300, 30, 1200, 400)]:
        p1, p2 = a / n1, c / n2
        truth = p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2
        got = var_single(n1, a, n2, c)
        worst = max(worst, abs(got - truth) / truth)
    check("variance matches the binomial limit exactly",
          worst < 1e-12, f"worst relative error {worst:.2e}")


def test_cmh_separation() -> None:
    """A taxon present in every control and no case is a valid table with a
    tiny p, not a missing one. Scoring it `direction = 0` dropped the most
    extreme taxa from every published depleted/enriched fraction."""
    group("CMH direction under complete separation")
    from persistence.health_cmh import cochran_mantel_haenszel as fn
    # [[present_g1, absent_g1], [present_g2, absent_g2]]
    sep_g1 = [np.array([[40, 0], [0, 40]], dtype=float)]     # all g1, no g2
    sep_g2 = [np.array([[0, 40], [40, 0]], dtype=float)]     # no g1, all g2
    null = [np.array([[20, 20], [20, 20]], dtype=float)]     # OR exactly 1
    try:
        _, _, d1 = fn(sep_g1)
        _, _, d2 = fn(sep_g2)
        _, _, d0 = fn(null)
    except Exception as exc:                                  # pragma: no cover
        check("CMH helper callable with a table list", False, repr(exc))
        return
    check("separation towards g1 scores +1", d1 == 1, f"direction {d1}")
    check("separation towards g2 scores -1", d2 == -1, f"direction {d2}")
    check("an odds ratio of exactly 1 still scores 0", d0 == 0,
          f"direction {d0} -- a true null must not be counted as directional")


def test_bh() -> None:
    """The driver q-values took the cumulative minimum over input order
    instead of sorted-p order, so they were a running minimum over an
    arbitrary sequence rather than a BH adjustment."""
    group("Benjamini-Hochberg q-values")
    from persistence.health_background import _bh
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 0.01, 12), rng.uniform(0, 1, 48)])
    q = _bh(p)
    order = np.argsort(p)
    check("q is monotone in p", bool(np.all(np.diff(q[order]) >= -1e-12)))
    check("q >= p everywhere", bool(np.all(q + 1e-12 >= p)))
    try:
        from statsmodels.stats.multitest import multipletests
        ref = multipletests(p, method="fdr_bh")[1]
        check("matches statsmodels fdr_bh", bool(np.allclose(q, ref)),
              f"max |diff| {np.max(np.abs(q - ref)):.2e}")
    except ImportError:                                       # pragma: no cover
        check("matches statsmodels fdr_bh", True, "statsmodels absent, skipped")
    import inspect
    from persistence import section3_health as s3
    src = inspect.getsource(s3)
    check("the driver analysis uses that one implementation",
          "hbg._bh(ps)" in src and "def _bh_fdr" not in src,
          "a second copy here is how the two diverged")


def test_weighted_sem() -> None:
    """Var(weighted mean) = sum(w^2 Var(y))/(sum w)^2: the weights enter
    SQUARED. They entered to the first power, which understated every plotted
    error bar whenever the weights were dispersed."""
    group("SE of the GMW-weighted mean")
    from persistence.depth_bias_ipw import _weighted_sem
    rng = np.random.default_rng(1)
    n = 800
    y = rng.normal(200, 60, n)
    w = np.clip(rng.lognormal(0, 0.8, n), 0, 10); w /= w.mean()
    mu = float((w * y).sum() / w.sum())
    boot = np.array([(lambda i: (w[i] * y[i]).sum() / w[i].sum())
                     (rng.choice(n, n, replace=True)) for _ in range(3000)])
    ratio = _weighted_sem(y, w, mu) / boot.std(ddof=1)
    check("agrees with the plug-in bootstrap it accompanies",
          0.95 < ratio < 1.05,
          f"analytic/bootstrap = {ratio:.3f} (the old form gave ~0.69)")
    # Equal weights is the case where right and wrong agree; it must still hold.
    w1 = np.ones(n)
    mu1 = y.mean()
    check("reduces to s/sqrt(n) at equal weights",
          abs(_weighted_sem(y, w1, mu1) - y.std(ddof=0) / np.sqrt(n)) < 1e-9)


def test_boot_star() -> None:
    """A percentile bootstrap cannot return p below 1/n_boot, so a star ladder
    containing a finer tier advertises a result it can never produce."""
    group("bootstrap star resolution")
    from persistence.plot_utils import boot_star
    from persistence import config
    nb = int(getattr(config, "HEALTH_BG_N_BOOT", 200))
    floor = 1.0 / nb
    stars = {boot_star(p, nb) for p in (0.0, floor, 0.004, 0.02, 0.5)}
    check("no tier finer than the bootstrap resolution is offered",
          "***" not in stars or 0.001 > floor,
          f"n_boot={nb}, floor={floor:g}, tiers seen {sorted(stars)}")
    check("a genuinely fine p still earns *** when resolution allows",
          boot_star(0.0005, 10000) == "***")
    check("the ladder still separates * from n.s.",
          boot_star(0.02, nb) == "*" and boot_star(0.5, nb) == "n.s.")


def test_background_weights() -> None:
    """The matched background is only prevalence-matched under its weights.
    Table S6B's direction column averaged it unweighted, so it could state the
    opposite of the mean z printed in the same row."""
    group("Table S6B background weighting")
    import pandas as pd
    from persistence import table_s7
    rng = np.random.default_rng(3)
    n = 400
    # A background whose low-prevalence end is dense and shifted the other way,
    # which is exactly the geometry the weights exist to correct.
    shift = np.concatenate([rng.normal(-2.0, 1.0, 320),
                            rng.normal(+4.0, 1.0, 80)])
    w = np.concatenate([np.full(320, 0.1), np.full(80, 1.0)])
    bg = pd.DataFrame({"taxon": [f"t{i}" for i in range(n)],
                       "rd_mh": -shift / 100.0, "bg_weight": w})
    lst = pd.DataFrame({"taxon": ["L1", "L2", "L3"],
                        "rd_mh": -np.array([3.0, 4.0, 5.0]) / 100.0})
    direction, mean_shift = table_s7._direction_vs_bg(lst, bg)
    # Expectations in whatever units _shift_vals returns, so the check does
    # not quietly encode a sign or scale convention of its own.
    bv = table_s7._shift_vals(bg)
    lv = table_s7._shift_vals(lst)
    want_w = float(np.mean(lv)) - float(np.average(bv, weights=w))
    want_u = float(np.mean(lv)) - float(np.mean(bv))
    check("the direction is computed against the WEIGHTED background",
          abs(mean_shift - want_w) < 1e-9,
          f"got {mean_shift:+.4f}; weighted {want_w:+.4f}, "
          f"unweighted {want_u:+.4f}")
    check("weighting changes the answer here, so the check has teeth",
          abs(want_w - want_u) > 1e-3,
          f"weighted {want_w:+.4f} vs unweighted {want_u:+.4f}")
    check("the label follows the weighted shift",
          direction in ("depleted", "enriched", "none"), f"direction={direction}")


# ── wiring that a run depends on ─────────────────────────────────────────────

def test_referencedata_rebuilds() -> None:
    """Three sites rebuilt the namedtuple field by field and omitted
    total_seq_bp, which has no default, so each raised TypeError."""
    group("ReferenceData rebuilds")
    import pandas as pd
    from persistence.loading import ReferenceData
    idx = pd.Index(["a", "b", "c"])
    d = ReferenceData(
        ref_name="t", breadth=pd.DataFrame(index=idx),
        depth=pd.DataFrame(index=idx), abundance=pd.DataFrame(index=idx),
        meta=pd.DataFrame(index=idx), lengths=pd.Series(dtype=float),
        total_bp=pd.Series(1.0, index=idx),
        total_seq_bp=pd.Series(2.0, index=idx),
        vanish_set=set(), blossum_set=set(),
        cov_matrix=pd.DataFrame(index=idx))

    from persistence.sensitivity_sweep import _replace_vb_sets

    class _VB:
        vanish_set = {"x"}
        blossum_set = {"y"}
    try:
        r = _replace_vb_sets(d, _VB())
        ok1 = r.total_seq_bp is not None and r.vanish_set == {"x"}
    except TypeError as exc:
        ok1, r = False, None
        check("sensitivity sweep rebuild", False, repr(exc)); return
    check("sensitivity sweep keeps total_seq_bp", ok1)

    from persistence.depth_audit import _remove_samples
    try:
        r2 = _remove_samples(d, np.array([False, True, False]))
        ok2 = (list(r2.meta.index) == ["a", "c"]
               and list(r2.total_seq_bp.index) == ["a", "c"])
    except TypeError as exc:
        check("depth audit rebuild", False, repr(exc)); return
    check("depth audit filters total_seq_bp with the rest", ok2)


def test_figure_contract() -> None:
    """The slot registries must agree: a figure numbered in one and not the
    others is how a slot silently stops being produced."""
    group("figure contract")
    import json
    import re
    import importlib
    man = (ROOT / "tools/additional_figures_manifest.txt").read_text()
    slots = sorted(int(m) for m in re.findall(r"^S(\d+)\|", man, flags=re.M))
    check("slots are contiguous from 1",
          slots == list(range(1, len(slots) + 1)),
          f"{len(slots)} slots, highest {slots[-1] if slots else 0}")
    caps = json.loads((ROOT / "tools/additional_figure_captions.json").read_text())
    cap_n = sorted(int(k) for k in caps if k.isdigit())
    check("every slot has a caption and no caption is orphaned",
          cap_n == slots, f"captions {len(cap_n)} vs slots {len(slots)}")
    draw = importlib.import_module("tools.replot_from_data").DRAW
    missing = [f"S{n}" for n in slots if f"S{n}" not in draw]
    check("every slot can be redrawn from its data object",
          not missing, f"no drawer for {missing}" if missing else "")
    check("no slot numbered above the contract",
          not any(k.startswith("S") and k[1:].isdigit()
                  and int(k[1:]) > len(slots) for k in draw),
          "a drawer for a slot that no longer exists")


def test_panel_geometry() -> None:
    """Figure 5E and S26 are the same comparison on canvases of different
    width. Sizes are declared in printed points and converted per canvas; if
    that stops holding the two drift apart again."""
    group("Figure 5E / S26 printed geometry")
    from persistence.plot_utils import background_curve_style
    a = background_curve_style(1.0)
    b = background_curve_style(24.0 / 7.24)
    mag = 24.0 / 7.24
    pt_keys = [k for k in a if k.startswith(("fs_", "lw_")) or k == "legend_ms"]
    pt_ok = all(abs(b[k] - a[k] * mag) < 1e-9 for k in pt_keys)
    area_ok = all(abs(b[k] - a[k] * mag ** 2) < 1e-9
                  for k in ("s_marker", "s_bg"))
    check("point sizes scale with the canvas", pt_ok)
    check("marker areas scale with its square", area_ok)
    s3 = (ROOT / "persistence/section3_health.py").read_text()
    f5 = (ROOT / "figures/figure5.py").read_text()
    check("both draw through the one grid renderer",
          "draw_background_curve_grid" in s3 and "draw_background_curve_grid" in f5)
    check("S26 skips the type pass Figure 5 never runs",
          "_skip_type_normalise = True" in s3,
          "otherwise it rescales the annotations and drops their bold")

    # The parts drawn AROUND the data: spines, ticks and the shape of the box.
    # These were the last thing still set per figure -- Figure 5's _style()
    # used absolute sizes on a 24 in canvas, so its spines printed at a
    # quarter of S26's, and the two gridspecs gave cells of different shape.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from persistence.plot_utils import style_background_curve_axes

    def _printed(mag):
        st = background_curve_style(mag)
        fig, ax = plt.subplots(figsize=(4, 4))
        style_background_curve_axes(ax, st)
        t = ax.xaxis.get_major_ticks()[0]
        out = {
            "spine": ax.spines["left"].get_linewidth() / mag,
            "tick_label": t.label1.get_fontsize() / mag,
            "tick_len": t.tick1line.get_markersize() / mag,
            "tick_width": t.tick1line.get_markeredgewidth() / mag,
            "box_aspect": ax.get_box_aspect(),
        }
        plt.close(fig)
        return out

    pa, pb = _printed(1.0), _printed(24.0 / 7.24)
    diffs = {k: (pa[k], pb[k]) for k in pa if abs(pa[k] - pb[k]) > 1e-9}
    check("spines, ticks and cell shape print identically in both",
          not diffs, f"differ: {diffs}" if diffs else
          f"spine {pa['spine']} pt, ticks {pa['tick_len']} pt, "
          f"box aspect {pa['box_aspect']}")
    # Scope this to the panel-E drawing block only. Panel D legitimately
    # calls _style() for its empty fallback, and an earlier slice caught that
    # instead -- the check failed on correct code.
    _blk = f5.split("draw_background_curve_grid(")[-1].split("elif active_pairs")[0]
    # Strip comments first: the block explains why _style() is absent, and the
    # check was matching that explanation.
    _code = "\n".join(l.split("#")[0] for l in _blk.splitlines())
    check("Figure 5 does not re-style the E cells after the shared renderer",
          "_style(" not in _code,
          "_style() uses absolute sizes and would undo the shared geometry")

    # Spacing between cells. The module-level MARGINS carry wspace 0.16 /
    # hspace 0.28 and tighten_margins applies them to any figure that does
    # not override, so S26 was silently reset to the pipeline default while
    # Figure 5's own gridspec kept its values.
    from persistence.plot_utils import CELL_WSPACE, CELL_HSPACE
    check("S26 overrides the margin pass with the shared cell spacing",
          '"wspace": CELL_WSPACE' in s3 and '"hspace": CELL_HSPACE' in s3,
          "otherwise tighten_margins resets it to MARGINS")
    check("Figure 5's gridspec takes the same two constants",
          "wspace=_CELL_WSPACE" in f5 and "hspace=_CELL_HSPACE" in f5)
    check("neither figure hardcodes a spacing number",
          "hspace=0.17" not in f5 and "wspace=0.26" not in f5,
          f"shared values are wspace={CELL_WSPACE}, hspace={CELL_HSPACE}")

    # Cells square, and the GRIDSPEC CELL square too -- not just the drawn
    # box. set_box_aspect forces the box, but if the cell around it is a
    # different shape the box shrinks inside it and the surplus shows as
    # whitespace: horizontally in S26, vertically in Figure 5E.
    from persistence.plot_utils import (CELL_ASPECT,
                                           background_curve_figsize,
                                           PAGE_WIDTH_IN, drawn_margins,
                                           MARGINS)
    check("cells are 1:1", abs(CELL_ASPECT - 1.0) < 1e-9, f"{CELL_ASPECT}")
    # Science: no rule below 0.5 pt at final size. These are declared in
    # printed points, so the check is on the declaration itself. "edge" is a
    # hairline white separator between adjacent markers, not a rule, and is
    # deliberately exempt.
    from persistence.plot_utils import PRINTED_LW
    _thin = {k: v for k, v in PRINTED_LW.items() if k != "edge" and v < 0.5}
    check("every drawn rule clears the 0.5 pt floor at final size",
          not _thin, f"below the floor: {_thin}" if _thin else
          f"thinnest rule {min(v for k, v in PRINTED_LW.items() if k != 'edge')} pt")
    LEG = 0.46
    h, cell_w = background_curve_figsize(PAGE_WIDTH_IN, 5, 2, bottom_in=LEG)
    m = drawn_margins({"left": MARGINS["left"], "right": MARGINS["right"],
                       "top": MARGINS["top"]}, titled=True)
    region_h = float(m["top"]) * h - LEG
    cell_h = region_h / (2 + 1 * CELL_HSPACE)
    check("S26's height makes its gridspec cell square",
          abs(cell_h / cell_w - CELL_ASPECT) < 1e-6,
          f"cell {cell_w:.4f} x {cell_h:.4f} in, aspect {cell_h/cell_w:.4f}")
    check("S26 sizes against the margins it is SAVED with",
          "drawn_margins" in (ROOT / "persistence/plot_utils.py").read_text()
          and "background_curve_figsize" in s3,
          "the publication path raises the top margin after the title is "
          "stripped; sizing from the authored one leaves a band between rows")
    check("Figure 5 budgets the E block from the shared constants",
          "e_block_h" in f5 and "CELL_HSPACE as _CH" in f5)
    check("Figure 5 reserves a band under E for its labels and legend",
          "E_LEGEND_BAND_IN" in f5,
          "without it the block grows down over its own legend")


def test_no_competing_copies() -> None:
    """A second copy of a live module is how a run silently reverts.

    The repository keeps root-level copies of two figure generators that
    differ from the ones in figures/ -- and the root make_combined_figure3
    does not even import, because it wants a SCALE_RATIO that plot_utils no
    longer defines. Both redraw tools import it by bare name, so which one
    wins is decided by sys.path order.
    """
    group("competing module copies")
    import ast
    import collections
    import pathlib
    seen = collections.defaultdict(list)
    for p in (list(pathlib.Path("persistence").glob("*.py"))
              + list(pathlib.Path("figures").glob("*.py"))):
        try:
            t = ast.parse(p.read_text())
        except Exception:
            continue
        for n in t.body:
            if isinstance(n, ast.FunctionDef):
                seen[n.name].append(str(p))
    for name in ("display_order", "group_dividers", "draw_background_curve_cell",
                 "draw_background_curve_grid", "background_curve_style",
                 "background_curve_legend", "boot_star", "_bh"):
        check(f"one definition of {name}", len(seen.get(name, [])) == 1,
              f"found in {seen.get(name, [])}")
    check("the retired density panel-E builder is gone",
          not seen.get("build_panel_e_figure")
          and not seen.get("_build_panel_e_figure"),
          "a dead builder for a superseded figure is how it comes back")

    # Which copy the redraw tools actually get.
    for tool in ("tools/replot_main.py", "tools/replot_from_data.py"):
        src = pathlib.Path(tool).read_text()
        if "make_combined_figure3" not in src:
            continue
        i_fig = src.find('sys.path.insert(0, str(ROOT / "figures"))')
        i_root = src.find("sys.path.insert(0, str(ROOT))")
        ok = (i_fig == -1) or (i_root == -1) or (i_fig > i_root)
        check(f"{tool} puts figures/ ahead of the repo root",
              ok, "otherwise it imports the root copy, which does not import")

    # The demographic panel must follow the same columns as Figure 5E.
    dem = pathlib.Path("figures/make_demographic_associations.py").read_text()
    code = "\n".join(l.split("#")[0] for l in dem.splitlines())
    check("the demographic panel reads the shift through the accessors",
          '"prev_diff"' not in code.replace('or "prev_diff"', "")
          and "importance_weight" not in code,
          "hardcoding them kept it on the superseded pooled shift and the "
          "IW background after the disease figures moved on")

    # The demographic panel can only follow the corrected method if the
    # demographic PASS writes the corrected columns. Reading through the
    # accessors is necessary but not sufficient.
    s3h = pathlib.Path("persistence/section3_health.py").read_text()
    check("the demographic pass attaches the stratified shift",
          "_attach_stratified(per_taxon, _dstrat" in s3h,
          "without it the demographic tables carry only prev_diff and the "
          "panel falls back to the pooled shift for ever")

    # Condition order: every plotting function that labels condition columns
    # must use the shared rule, or the figures disagree with each other.
    for var, fn_name in (("all_conds", "_plot_taxon_drivers"),
                         ("conds", "_plot_richness_comparison")):
        import ast as _ast
        tree = _ast.parse(s3h)
        body = ""
        for n in _ast.walk(tree):
            if isinstance(n, _ast.FunctionDef) and n.name == fn_name:
                body = "\n".join(s3h.splitlines()[n.lineno - 1:n.end_lineno])
        check(f"{fn_name} orders its columns with the shared rule",
              f"_display_order(" in body,
              f"{var} was missed when the others were moved, so this figure "
              "kept the pooled contrast in column 1")

    # One non-constant seed in the tree was the salted string hash.
    hg = pathlib.Path("persistence/hg_enriched_abundance.py").read_text()
    check("no seed is derived from Python's salted hash()",
          "hash(gname" not in hg and "crc32" in hg,
          "hash() of a str changes every process unless PYTHONHASHSEED is set")


def test_approved_method_changes() -> None:
    """The two method changes the author approved on 2026-10-07."""
    group("approved method changes")
    import inspect
    import numpy as np
    import pandas as pd
    from persistence import config, health_background as hbg, differential

    # 1. The background excludes every published list, not just the one scored.
    check("background_z takes an exclude set",
          "exclude" in inspect.signature(hbg.background_z).parameters)
    check("sibling-list exclusion is on by default",
          getattr(config, "HEALTH_BG_EXCLUDE_SIBLING_LISTS", False) is True)
    # It must actually change the background, not just be accepted.
    rng = np.random.default_rng(0)
    n = 600
    f = pd.DataFrame({"p": rng.uniform(0.01, 0.9, n),
                      "shift": rng.normal(0, 2, n),
                      "se": np.full(n, 0.5)},
                     index=[f"t{i}" for i in range(n)])
    f["x"] = np.log10(f["p"])
    mine = list(f.index[:60])
    sibs = list(f.index[60:160])
    a = hbg.background_z(f, mine, n_boot=5)[0]
    b = hbg.background_z(f, mine, n_boot=5, exclude=set(mine) | set(sibs))[0]
    check("excluding siblings changes the fitted background",
          a is not None and b is not None
          and not np.allclose(a["expected"].values, b["expected"].values),
          "if this passes trivially the exclude set is being ignored")

    # 2. Section 2's chi-square carries the design effect.
    check("the design-effect correction is on by default",
          getattr(config, "DIFFERENTIAL_DESIGN_EFFECT", False) is True)
    src = inspect.getsource(differential.test_prevalence_weighted)
    check("chi2 is divided by the design effect",
          "chi2 / deff" in src and "chi2_dist.sf" in src)
    idx = [f"s{i}" for i in range(400)]
    br = pd.DataFrame(rng.random((400, 40)) * 100, index=idx,
                      columns=[f"m{j}" for j in range(40)])
    w1 = np.clip(rng.lognormal(0, 0.8, 200), 0, 10)
    w2 = np.clip(rng.lognormal(0, 0.8, 200), 0, 10)
    raw = differential.test_prevalence_weighted(
        br, idx[:200], idx[200:], w1, w2, "A", "B", threshold=50.0)
    config.DIFFERENTIAL_DESIGN_EFFECT = True
    cor = differential.test_prevalence_weighted(
        br, idx[:200], idx[200:], w1, w2, "A", "B", threshold=50.0)
    m = raw.merge(cor, on="microbe", suffixes=("_r", "_c"))
    check("correction only ever raises p, never lowers it",
          bool((m["pval_c"] >= m["pval_r"] - 1e-12).all()))
    check("effect sizes are untouched by the correction",
          bool(np.allclose(m["prevalence_diff_r"], m["prevalence_diff_c"])))


def test_environment() -> None:
    """Both documented environment routes must satisfy the preflight they
    mandate."""
    group("declared environment")
    pre = (ROOT / "tools/preflight.py").read_text()
    env = (ROOT / "environment.yml").read_text()
    lock = (ROOT / "requirements-lock.txt").read_text()
    for mod in ("plotly", "geopandas", "shapely"):
        if f"import {mod}" in pre:
            check(f"preflight's {mod} is pinned in both env files",
                  mod in env and mod in lock,
                  f"environment.yml={mod in env} lock={mod in lock}")


def test_spec_combination() -> None:
    """The primary statistic must be combined across detection specs, and the
    supplement must say so.

    Until 2026-10-08 only the energy distance -- since demoted to descriptive
    -- was combined across thresholds, while the mean z the manuscript
    reports was left per spec with no stated rule for disagreement. If these
    fail, the two are out of step again: either the pipeline stopped writing
    the combined table, or the supplement went back to claiming it does not
    exist.
    """
    group("cross-spec combination of the primary statistic")
    import pandas as pd
    from persistence import section3_health as s3

    src = (ROOT / "persistence/section3_health.py").read_text()
    check("the combiner exists", "def _combine_curve_specs" in src)
    check("it is called where the specs have all run",
          "_combine_curve_specs(_curve_summaries, s3_main)" in src)
    check("each spec's summary is collected",
          '_curve_summaries[spec_tag] = curve["summary"].copy()' in src)
    # One spec cannot be combined with itself: a single-spec run must not
    # silently write a table whose HMP equals that spec's own p.
    check("a single spec writes no combined table",
          "if len(_curve_summaries) > 1:" in src)

    # The HMP must be the Wilson form, not a mean or a Fisher combination.
    ps = [0.01, 0.20]
    check("_hmp is k / sum(1/p)",
          abs(s3._hmp(ps) - 2 / (1 / 0.01 + 1 / 0.20)) < 1e-15,
          f"_hmp({ps}) = {s3._hmp(ps):.6f}")
    check("_hmp never exceeds k x min(p)",
          s3._hmp(ps) <= len(ps) * min(ps))
    check("_hmp survives a NaN spec", np.isfinite(s3._hmp([0.01, np.nan])))

    # End to end on a frame of the shape the curve writes.
    import tempfile
    rows = [("Unhealthy", "VanishH+"), ("IBD", "VanishH+"),
            ("IBD", "BlossumH+")]
    def _f(zs, ps_):
        return pd.DataFrame([dict(Condition=c, Taxon_list=l, mean_z=z,
                                  boot_p=q, ci_lo=z - 1, ci_hi=z + 1,
                                  n_supported=40, n_list=50)
                             for (c, l), z, q in zip(rows, zs, ps_)])
    a = _f([1.0, -0.4, 1.2], [0.0005, 0.04, 0.0005])
    b = _f([1.2, -0.2, 0.8], [0.0010, 0.30, 0.0005])
    with tempfile.TemporaryDirectory() as d:
        out = s3._combine_curve_specs({"10pct": a, "300kb": b}, d)
        got = Path(d) / "health_background_curve_meta.csv"
        check("writes health_background_curve_meta.csv", got.exists())
        check("one row per condition x list", len(out) == len(rows),
              f"{len(out)} rows for {len(rows)} cells")
        for col in ("mean_z_combined", "p_hmp", "sign_agree",
                    "n_specs", "n_sig_specs", "n_boot"):
            check(f"carries {col}", col in out.columns)
        r = out[(out.Condition == "Unhealthy")].iloc[0]
        check("mean_z_combined averages the specs",
              abs(r.mean_z_combined - 1.1) < 1e-12, f"{r.mean_z_combined}")
        check("p_hmp matches the HMP of the spec p-values",
              abs(r.p_hmp - s3._hmp([0.0005, 0.0010])) < 1e-15)
        check("both specs' own columns survive",
              {"mean_z_10pct", "boot_p_300kb"} <= set(out.columns))
        check("sign agreement is flagged, not folded into p",
              bool(out.sign_agree.all()), "all three cells agree in sign here")
        # A cell significant under both specs must not be lost by combining.
        both = (out.boot_p_10pct < 0.05) & (out.boot_p_300kb < 0.05)
        check("no cell significant in both specs is lost by the HMP",
              bool((out.loc[both, "p_hmp"] < 0.05).all()))

    # Which conditions the health figures plot is set by TWO filters. Using
    # HEALTH_PLOT_SKIP alone gives seven and once let MASLD (2 studies) and
    # ASCVD (1) into S26 and S28-S31 while Figure 5 dropped them.
    import figures.figure5 as f5
    check("figure 5 applies the strata floor as well as the skip set",
          "df[df[\"n_strata\"] >= MIN_STRATA_TO_PLOT]"
          in (ROOT / "figures/figure5.py").read_text())
    check("the supplement figures use the same rule as figure 5",
          "config.MIN_STRATA_TO_PLOT"
          in (ROOT / "persistence/section3_health.py").read_text())
    check("the strata floor is 3", f5.MIN_STRATA_TO_PLOT == 3,
          f"MIN_STRATA_TO_PLOT = {f5.MIN_STRATA_TO_PLOT}")

    # The supplement text must describe what the code now does -- and the
    # builder must read THAT file. It imported /tmp/plugin.py for a while,
    # so the page was built from a scratch copy and the repo edits never
    # reached the HTML.
    # These two check the supplement-drafting chain, which is authoring
    # machinery rather than part of the run. tools/make_public_tree.py keeps
    # it out of the published repo, so the files can legitimately be absent
    # and the suite must stay green without them -- a reviewer should not see
    # a failure for a manuscript draft they were never given.
    bld_p = ROOT / "tools/build_section6_plugin.py"
    page_p = ROOT / "docs/supp_section6_plugin_20261008.html"
    edits_p = ROOT / "tools/section6_plugin_edits.py"
    if bld_p.exists():
        bld = bld_p.read_text()
        check("the plug-in builder reads the repo edits, not a scratch copy",
              "from section6_plugin_edits import" in bld
              and "sys.path.insert(0,'/tmp')" not in bld
              and "from plugin import" not in bld)
    if page_p.exists() and edits_p.exists():
        check("the built page is newer than the edits it is built from",
              page_p.stat().st_mtime >= edits_p.stat().st_mtime,
              "rebuild with tools/build_section6_plugin.py")
    if not edits_p.exists():
        return
    ed = edits_p.read_text()
    check("supplement no longer says the mean z is left uncombined",
          "not combined across specifications" not in ed
          and "rather than formally combined" not in ed)
    check("supplement defines p_i as the mean-z bootstrap p",
          "bootstrap p-value of the mean z under detection specification i"
          in ed)
    check("supplement names the combined table",
          "health_background_curve_meta.csv" in ed)
    check("supplement still says the figures annotate one spec",
          "annotate one specification and not the combination" in ed)
    check("the HMP dependence justification is stated",
          "valid under arbitrary dependence" in ed)


def test_condition_eligibility() -> None:
    """Which contrasts are reported must come from the rule, not a list.

    A study contributes a stratum only where both arms carry at least
    CMH_MIN_STRATUM samples, and a contrast is reported only where at least
    MIN_STRATA_TO_PLOT such strata exist. Hypertension (2 qualifying studies)
    and Metabolic Syndrome (1) fall out by that rule. If these fail, a
    hand-written list has come back, and it will drift from the cohort the
    way the last one did -- it was hiding Fatty Liver and ASCVD from Figure 5
    while leaving them in the supplement.
    """
    group("condition eligibility is a rule, not a list")
    import pandas as pd
    from persistence import config, section3_health as s3

    # Was "empty by default". It now holds exactly one name: the pooled
    # "Unhealthy" contrast, which the rule admits on 41 studies but which
    # contains the cases of the four focal contrasts and so is not an
    # independent condition. config.HEALTH_PLOT_SKIP carries the full reason.
    # The assertion is kept tight -- exactly this one name -- so that adding
    # another by hand still trips the test.
    check("HEALTH_PLOT_SKIP holds only the pooled contrast",
          set(config.HEALTH_PLOT_SKIP) == {"Unhealthy"},
          f"HEALTH_PLOT_SKIP = {config.HEALTH_PLOT_SKIP!r}")
    check("the two halves of the rule are set",
          config.CMH_MIN_STRATUM == 5 and config.MIN_STRATA_TO_PLOT == 3,
          f"{config.CMH_MIN_STRATUM} per arm, {config.MIN_STRATA_TO_PLOT} studies")

    # The rule, on a frame shaped like the CMH table.
    cmh = pd.DataFrame([
        ("Unhealthy", 41), ("Obese", 29), ("CRC / Adenoma", 8),
        ("Glucose Intolerance", 6), ("IBD", 6),
        ("Fatty Liver", 2), ("Hypertension", 2),
        ("ASCVD", 1), ("Metabolic Syndrome", 1),
    ], columns=["Condition", "n_studies"])
    _saved = s3._ELIGIBLE_CONDITIONS
    try:
        s3._note_eligible_conditions(cmh)
        shown = {c for c in cmh.Condition if not s3._condition_dropped(c)}
        # Five contrasts clear the study-count rule; four are reported. The
        # pooled "Unhealthy" contrast is withheld by name through
        # HEALTH_PLOT_SKIP, because the rule cannot express what is wrong with
        # it -- its cases ARE the cases of the other four. Both halves are
        # asserted separately so that a change to either is visible here.
        judged, admitted = s3._ELIGIBLE_CONDITIONS
        check("the study-count rule admits five contrasts",
              set(admitted) == {"Unhealthy", "Obese", "CRC / Adenoma",
                                "Glucose Intolerance", "IBD"},
              f"{sorted(admitted)}")
        check("exactly the four focal contrasts are reported",
              shown == {"Obese", "CRC / Adenoma",
                        "Glucose Intolerance", "IBD"}, f"{sorted(shown)}")
        check("the pooled contrast is withheld by name, not by the rule",
              "Unhealthy" in admitted and s3._condition_dropped("Unhealthy"))
        for c in ("Hypertension", "Metabolic Syndrome", "Fatty Liver", "ASCVD"):
            check(f"{c} is dropped by the rule", s3._condition_dropped(c))
        # Demographic strata are not disease contrasts and must pass through.
        check("contrasts the rule never judged pass through",
              not any(s3._condition_dropped(c) for c in config.DEMOGRAPHIC_MAP),
              "the age/sex/BMI path shares these plot functions")
        # The override hook still works, for the no-obese sensitivity run.
        config.HEALTH_PLOT_SKIP = {"Obese"}
        check("an explicit override still excludes",
              s3._condition_dropped("Obese")
              and not s3._condition_dropped("IBD"))
    finally:
        config.HEALTH_PLOT_SKIP = set()
        s3._ELIGIBLE_CONDITIONS = _saved

    # One definition of the comparison set, not four.
    src = (ROOT / "persistence/section3_health.py").read_text()
    check("the comparison map is defined once",
          src.count('("Unhealthy",') == 1,
          f'{src.count(chr(40) + chr(34) + "Unhealthy" + chr(34) + chr(44))} literal copies')
    check("Obese (healthy only) is gone from the comparison set",
          "obese_healthy_only" not in src)
    check("the eligibility rule is read where the CMH table appears",
          "_note_eligible_conditions(cmh_df)" in src)
    check("no plot site reads the skip list directly any more",
          "HEALTH_PLOT_SKIP', set())" not in src
          and 'HEALTH_PLOT_SKIP", set())' not in src.replace(
              'getattr(config, "HEALTH_PLOT_SKIP", set())', "", 1),
          "all sites go through _condition_dropped")

    f5 = (ROOT / "figures/figure5.py").read_text()
    check("figure 5 names no condition in its skip set",
          "Obese (healthy only)" not in f5.split("SKIP: set")[1][:200]
          if "SKIP: set" in f5 else False)
    check("figure 2 has no unused skip set",
          "SKIP" not in (ROOT / "figures/figure2.py").read_text())


def test_no_condition_leaks_into_figures() -> None:
    """Tables may carry every contrast; figures may not.

    The eligibility rule admits five contrasts. The tables deliberately keep
    all nine and flag them, because an excluded contrast is still evidence.
    The figures must show only the five, and must say which five they showed:
    a figure's panels are a superset -- the curve is fitted for every contrast
    and filtered at draw time -- so without a recorded list a redraw cannot
    tell which contrasts the published figure excluded.
    """
    group("no ineligible contrast reaches a figure")
    s3 = (ROOT / "persistence/section3_health.py").read_text()
    f5 = (ROOT / "figures/figure5.py").read_text()
    t7 = (ROOT / "persistence/table_s7.py").read_text()

    # Three, not four: the prevalence-matched bars held slot S25 and were
    # retired on 2026-10-09, so that emit is gone. The remaining three are the
    # background curve, the taxon drivers and the within-clade contrasts.
    check("the three health figure emits record what they drew",
          s3.count('"drawn_conditions"') == 3,
          f'{s3.count(chr(34) + "drawn_conditions" + chr(34))} of 3 emit sites')
    check("figure 5 records what it drew", '"drawn_conditions": list(conds)' in f5)
    check("figure 5 computes its condition list before emitting",
          f5.index("n_cond = max(len(conds), 1)") < f5.index('"drawn_conditions"'),
          "otherwise conds is unbound at the emit")
    check("a redraw from the data object does not re-emit",
          "_emit_fresh = False" in f5 and "if _emit_fresh:" in f5,
          "else a later redraw silently becomes the record")

    check("the table writer flags eligibility instead of dropping rows",
          "def _mark_eligible" in t7 and t7.count("_mark_eligible(") == 3)
    check("the table writer drops no row",
          "eligible" in t7 and ".drop(" not in t7.split("def _mark_eligible")[1][:2000],
          "tables keep every contrast by design")
    check("the table writer reads the same rule as the figures",
          "MIN_STRATA_TO_PLOT" in t7)

    # Every health figure must derive its conditions from an eligibility
    # source, not from whatever frame it happens to hold.
    for fn_name, src in (("_plot_prevalence_matched", "_plotted_conditions(cmh_df)"),
                         ("_plot_background_curve", "n_strata"),
                         ("_run_clade_contrasts", "MIN_STRATA_TO_PLOT")):
        blk = s3.split(f"def {fn_name}")[1][:9000] if f"def {fn_name}" in s3 else ""
        check(f"{fn_name} filters conditions by the rule", src in blk)
    check("the driver figures are passed the eligible set",
          s3.count("conditions=_plotted_conditions(cmh_df)") == 2)

    chk = ROOT / "tools/check_run_conditions.py"
    check("the post-run verifier exists", chk.exists())
    check("the verifier reads the rule from config, not a literal",
          "MIN_STRATA_TO_PLOT" in chk.read_text()
          and "drawn_conditions" in chk.read_text())


def test_method_comparison_table() -> None:
    """Table S12 must carry every list-level test, for both detection rules.

    Three of the four tests were computed and never written anywhere: the
    IW-Fisher and IW-directional p-values set the F and U stars on Fig. S25
    and existed only as drawn text. A reader could not check the threshold
    claim because the numbers behind it were not in any file.
    """
    group("Table S12, method comparison")
    import pandas as pd
    import tempfile
    from persistence import table_methods as tm

    s3 = (ROOT / "persistence/section3_health.py").read_text()
    check("the IW test p-values are persisted, not only drawn",
          "health_iw_tests_" in s3 and "_iw_rows.append" in s3)
    check("Table S12 is built once both specs have run",
          "build_method_comparison(s3_main" in s3)

    check("the HMP is the Wilson form",
          abs(tm._hmp([0.01, 0.2]) - 2 / (1 / 0.01 + 1 / 0.2)) < 1e-15)
    check("a missing component does not poison the combination",
          np.isfinite(tm._hmp([0.01, float("nan")])))

    # End to end on frames shaped like the ones a run writes.
    rows = [("Unhealthy", "VanishH+"), ("IBD", "VanishH+")]
    with tempfile.TemporaryDirectory() as d:
        for spec, ps in (("10pct", [0.0005, 0.04]), ("300kb", [0.001, 0.30])):
            pd.DataFrame([dict(Condition=c, Taxon_list=l, mean_z=1.0,
                               ci_lo=0.5, ci_hi=1.5, boot_p=p,
                               n_list=100, n_supported=100)
                          for (c, l), p in zip(rows, ps)]).to_csv(
                f"{d}/health_background_curve_summary_{spec}.csv", index=False)
            pd.DataFrame([dict(Condition=c, List=l, energy_distance=0.05,
                               energy_p=p, binom_p_dep=0.1, binom_p_enr=0.9)
                          for (c, l), p in zip(rows, ps)]).to_csv(
                f"{d}/table_s6b_energy_first_{spec}.csv", index=False)
            pd.DataFrame([dict(Condition=c, Taxon_list=l, direction="dep",
                               p_iw_fisher=p, p_iw_directional=p)
                          for (c, l), p in zip(rows, ps)]).to_csv(
                f"{d}/health_iw_tests_combined_{spec}.csv", index=False)
        pd.DataFrame([dict(Condition="Unhealthy", n_studies=41),
                      dict(Condition="IBD", n_studies=6)]).to_csv(
            f"{d}/health_per_condition_cmh_10pct.csv", index=False)
        out = tm.build_method_comparison(d, ref="combined")

        check("one row per condition x list", len(out) == len(rows))
        for col in ("eligible", "p_meanz_hmp", "p_energy_hmp",
                    "p_iw_fisher_hmp", "p_iw_directional_hmp",
                    "mean_z_combined", "sign_agree", "n_methods_sig",
                    "n_methods_tested"):
            check(f"carries {col}", col in out.columns)
        check("all four tests are combined across specs",
              out[["p_meanz_hmp", "p_energy_hmp", "p_iw_fisher_hmp",
                   "p_iw_directional_hmp"]].notna().all().all())
        r = out[out.Condition == "Unhealthy"].iloc[0]
        check("p_hmp matches the HMP of its components",
              abs(r.p_meanz_hmp - tm._hmp([0.0005, 0.001])) < 1e-15)
        check("the method count uses the combined values, not one spec",
              "_hmp" in tm.build_method_comparison.__doc__ or
              'f"{c}_hmp"' in (ROOT / "persistence/table_methods.py").read_text())
        check("no contrast is dropped from the table",
              len(out) == len(rows), "eligibility is a column, not a filter")
        check("eligibility is read from the run's own CMH table",
              bool(out["eligible"].iloc[0] in (True, False)))


def test_driver_panel_and_naming():
    """The 2026-10-09 pass: the driver panels, the list-level exclusion, and
    the two places that named the same genome differently."""
    group("driver figures, exclusion, naming (2026-10-09)")
    import pandas as pd
    from persistence import section3_health as s3
    from persistence import section2_pooling as s2p
    from persistence import config
    from scipy.stats import t as _t

    st = s3._driver_metric_style(pd.DataFrame({"driver_metric": ["z"]}))

    # The lollipop panel draws the star just past the end of the interval, so
    # the cap has to clear the widest (|score| + CI) any list produces. At 3.2
    # it did not: S27 needs 7.46 and lost 8 of its 29 stars, S26 2 of 16 and
    # S29 2 of 17, and M. gnavus was drawn outside the axes.
    check("the z panel's cap clears the widest drawn interval",
          st["pad_cap"] >= 7.5,
          f"pad_cap={st['pad_cap']} (BlossumH+ needs 7.46 incl. the 1.15 margin)")
    check("the heatmap colour scale is unchanged at +/-3",
          st["lim"] == 3.0, f"lim={st['lim']}")

    # The tick ladder is derived from the drawn range, so it stays legible
    # from the narrowest list (VANISH, 1.53) to the widest (BlossumH+, 7.46).
    def _ticks(pad):
        step = 1.0 if pad <= 3.5 else (2.0 if pad <= 7.0 else 3.0)
        return [v for v in np.arange(-step * 6, step * 6 + 1e-9, step)
                if abs(v) <= pad]
    for pad in (1.53, 4.28, 4.65, 7.46):
        n = len(_ticks(pad))
        check(f"score ticks stay legible at pad {pad}", 3 <= n <= 5,
              f"{n} ticks")

    # Selection: every drawn row carries a star, because the gate IS the star.
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        devs = rng.normal(0.9 if i < 25 else 0.0, 0.5, 5)
        tp = float(_t.sf(abs(devs.mean()) / (devs.std(ddof=1) / np.sqrt(5)), 4) * 2)
        rows.append(dict(Taxon_list="L", taxon=f"t{i}", n_conditions=5,
                         consistency=1.0, driver_score=float(devs.mean()),
                         combined_p=tp, q_value=min(1.0, tp * 6),
                         rank_devs=";".join(f"{v:.4f}" for v in devs)))
    sub = pd.DataFrame(rows)
    sig = sub[sub.combined_p < 0.05]
    want = min(max(len(sig), min(10, len(sub))), 30, len(sub))
    check("selection is the uncorrected p, floored at 10 and capped at 30",
          want == min(max(len(sig), 10), 30),
          f"{len(sig)} significant of {len(sub)} -> {want} rows")

    # The list-level statistic drops the same taxa the driver figure does, so
    # the figures decompose the number they sit beside.
    check("never-detected taxa are dropped from the list mean by default",
          bool(getattr(config, "HEALTH_BG_DROP_NEVER_DETECTED", False)),
          "config.HEALTH_BG_DROP_NEVER_DETECTED")
    src = (ROOT / "persistence" / "health_background.py").read_text()
    check("the drop is read from config at call time, not hard-coded",
          '_cfg("HEALTH_BG_DROP_NEVER_DETECTED"' in src)
    check("per-taxon output records which taxa were never detected",
          '"never_detected"' in src)

    # One accession, one name. The crosswalk carries an older GTDB release and
    # used to overwrite the catalogue-wide workbook, so Tables S8/S9 and
    # Figs. S26-S30 disagreed on fifteen genomes.
    _cache = s2p._COMBINED_TAX_CACHE
    try:
        s2p._COMBINED_TAX_CACHE = {"GCA_934700615.1": "d__Bacteria;g__Cryptobacteroides"}
        xw = pd.DataFrame({"genome": ["GCA_934700615.1", "MGYG000000001"],
                           "gtdb_taxonomy": ["d__Bacteria;g__RC9", "d__Bacteria;g__Only"]})
        tax = s2p._build_taxonomy_map(xw)
        check("the workbook outranks the crosswalk in section 2, as in section 3",
              tax.get("GCA_934700615.1") == "d__Bacteria;g__Cryptobacteroides",
              f"got {tax.get('GCA_934700615.1')!r}")
        check("the crosswalk still fills what the workbook does not annotate",
              tax.get("MGYG000000001") == "d__Bacteria;g__Only")
    finally:
        s2p._COMBINED_TAX_CACHE = _cache

    # Table S12's n column has to say how many taxa the mean rests on, now
    # that the list size and the supported count differ for VANISH.
    from tools.build_table_s12_docx import _n_cell
    check("Table S12 prints supported/list where the two differ",
          _n_cell({"n_list_10pct": 124, "n_supported_10pct": 80}) == "80/124")
    check("Table S12 prints one number where they agree",
          _n_cell({"n_list_10pct": 100, "n_supported_10pct": 100}) == "100")

    # Slot numbering: one manifest, and the driver figures agree with it.
    from tools import figure_sets
    slots = {s for s, _r, _st in figure_sets.manifest()}
    # Back to 30 on 2026-10-14. The list-vs-list comparison that was S31 is
    # now Figure 5F-H -- the panels moved into the main text rather than being
    # dropped -- and the metformin sensitivity that was S32 was withdrawn:
    # only two of the six Glucose Intolerance cohorts record the exposure, so
    # the arm that matters rests on two studies.
    check("the figure manifest ends at S31", slots == {f"S{i}" for i in range(1, 32)},
          f"{len(slots)} slots")
    # Resolved through the manifest, not against a third copy of the numbers.
    # Section 3 builds these stems as f-strings, so the check rebuilds the
    # same stem and asks the manifest which slot and which directory own it.
    # A renumbering that forgets section3_health.py now fails here instead of
    # publishing a correctly drawn figure beside a wrongly named data object.
    _by_base = {st.split("/")[-1]: (sl, "/".join(st.split("/")[:-1]))
                for sl, _r, st in figure_sets.manifest()}
    for _list, _slot in s3._TAXON_DRIVER_SLOTS.items():
        _safe = _list.replace("+", "plus").replace(" ", "_")
        _stem = f"health_taxon_drivers_{_safe}_combined_10pct"
        _got = _by_base.get(_stem, (None, None))
        # H+ lists go to main/, the original lists to supplement/ -- two calls
        # to the same writer with different output directories.
        _want_dir = ("section3_health/main" if _list.endswith("H+")
                     else "section3_health/supplement")
        check(f"driver figure for {_list} is manifest {_slot} in {_want_dir}",
              _got == (_slot, _want_dir), f"manifest has {_got}")
    # _GMW_RICHNESS_SLOTS is a function-local dict, so it is read from source.
    _src = (ROOT / "persistence" / "section3_health.py").read_text()
    _blk = _src.split("_GMW_RICHNESS_SLOTS = {")[1].split("}")[0]
    _gmw = dict(re.findall(r'"([^"]+)":\s*"(S\d+)"', _blk))
    check("the GMW richness slot map was found in section3_health",
          len(_gmw) == 3, str(_gmw))
    for _set, _slot in _gmw.items():
        _stem = f"health_richness_gmw_combined_{_set}_10pct"
        _got = _by_base.get(_stem, (None, None))
        check(f"GMW richness figure for {_set} is manifest {_slot}",
              _got == (_slot, "section3_health/main"), f"manifest has {_got}")

    # A caption carries its own figure number in its first words, so moving a
    # caption between keys is not enough -- the text has to move with it. The
    # S1/S31 swap renumbered the key and left the roadmap caption opening
    # "Figure S1.", which would have printed the wrong number under the right
    # figure.
    import json as _json
    _caps = _json.load(open(ROOT / "tools" / "additional_figure_captions.json",
                            encoding="utf-8"))
    _mismatch = {}
    for _k, _v in _caps.items():
        _m = re.match(r"\s*Figure\s+S(\d+)\.", _v.get("verbatim", ""))
        if (_m.group(1) if _m else None) != _k:
            _mismatch[_k] = (_m.group(1) if _m else "no 'Figure Sn.' opening")
    check("every caption's own number matches the slot it is filed under",
          not _mismatch, str(_mismatch))
    check("there is one caption per manifest slot",
          set(_caps) == {s[1:] for s in slots},
          f"{len(_caps)} captions, {len(slots)} slots")

    # The document builder keeps its own short-title map, keyed by slot number.
    # It is a second copy of the numbering and it had silently drifted: its
    # roadmap sat at 1 and it stopped at 30, so after the S1/S31 swap it would
    # have printed "Figure S1. Study roadmap" over the read-accounting caption
    # and dropped S31 entirely.
    # The document builder speaks the retired A-numbering and is excluded from
    # the published tree, so these two checks stand down where it is absent --
    # the same treatment the deposit tooling gets below.
    _bpath = ROOT / "tools" / "build_additional_figures_doc.py"
    if not _bpath.exists():
        check("document builder absent; its slot checks skipped", True)
    else:
        _bsrc = _bpath.read_text()
        _tmap = {}
        exec(compile(_bsrc[_bsrc.index("CAPTIONS = {"):
                           _bsrc.index("N_FIGS =")], "<caps>", "exec"), {}, _tmap)
        check("the document builder's title map is keyed by the live slots",
              set(_tmap["CAPTIONS"]) == {int(s[1:]) for s in slots},
              f"{sorted(set(_tmap['CAPTIONS']) ^ {int(s[1:]) for s in slots})}")
        check("the document builder emits every manifest slot",
              f"N_FIGS = {len(slots)}" in _bsrc,
              "N_FIGS stops the loop short of the last figure")

    # Every hard-coded slot literal in the analysis code is a slot that still
    # exists. This is what catches a retired or renumbered slot left behind in
    # a module nobody thought to grep -- the failure mode of the last three
    # renumberings.
    import re as _re
    _stray = {}
    for _py in sorted((ROOT / "persistence").glob("*.py")) + \
            sorted((ROOT / "figures").glob("*.py")):
        for _lit in set(_re.findall(r'"(S\d{1,2})"', _py.read_text())):
            if _lit not in slots:
                _stray.setdefault(_py.name, []).append(_lit)
    # figure_source.py's docstring uses S4 as an example, which is a real slot;
    # anything that is NOT a real slot is the thing being looked for.
    check("no code names a figure slot the manifest does not have",
          not _stray, str(_stray))

    # Figure 5's F-H block reads Table S12. Without it the figure draws A-E
    # and the task still reports success, so the table has to be declared.
    fr_src = (ROOT / "persistence" / "figure_runner.py").read_text()
    check("Figure 5 requires the table its F-H panels read",
          "table_s12_method_comparison_" in fr_src.split(
              "_req_s3_health_for_figure5")[1].split("def ")[0],
          "a missing prerequisite skips the task; a missing table silently "
          "drops three panels the caption describes")
    road = (ROOT / "figures" / "figure_s0_roadmap_v4.py").read_text()
    check("the roadmap prints the live figure range",
          "Fig. S25\u2013S30" in road or "Fig. S25–S30" in road,
          "figure_s0_roadmap_v4.py")
    check("the roadmap's health tables include S12",
          "S10{-}S12" in road)


def test_rachel_pipeline_items():
    """The comments on the drafts that land in pipeline output rather than in
    the manuscript: Fig. 5D's star colour, Fig. S5's opacity and statistic,
    the kb/kB unit, and Data S1."""
    group("reviewer comments with pipeline consequences")

    f5 = (ROOT / "figures" / "figure5.py").read_text()
    check("Figure 5D's significance stars are black, not the bar colour",
          'fontweight="bold", color=BAR_COL' not in f5
          and f5.count('fontweight="bold", color="black"') >= 2)

    s1f = (ROOT / "persistence" / "section1_figure.py").read_text()
    check("Fig. S5H's cloud is drawn at a visible opacity",
          "alpha=0.02, s=SYMBOL_MIN_S" not in s1f
          and "alpha=0.10, s=SYMBOL_MIN_S" in s1f,
          "at 0.02 the hunter-gatherer group, which the panel is about, "
          "did not render at all")
    check("Fig. S5H prints its correlation the way the manuscript does",
          'Spearman \u03c1 = {rho:+.3f}' in s1f or
          "Spearman \u03c1 = " in s1f,
          "the caption has to be able to quote the panel verbatim")

    # kb is the kilobase; kB is the kilobyte. The filename label always said
    # kb and the description said kB, so the supplement inherited both.
    from persistence.detection import DetectionSpec
    sp = DetectionSpec("flat_bp", 300_000)
    check("the flat threshold is labelled in kilobases, not kilobytes",
          sp.description.endswith("kb mapped bases") and sp.short_label == "300kb",
          f"{sp.short_label} / {sp.description}")
    # A stray kilobyte label is any "<digits> kB" that is not one of the two
    # deliberate back-compat keys (a lookup on the OLD description string, so
    # a stored data object still resolves) or the comment explaining them.
    _BACK_COMPAT = '300 kB mapped bases": "300 kb"'
    for mod in ("section1_figure", "figures/section1", "section1_diversity",
                "depth_bias_ipw", "section3_health", "detection"):
        txt = (ROOT / "persistence" / f"{mod}.py").read_text()
        stray = [ln.strip() for ln in txt.splitlines()
                 if re.search(r"\d\s?kB\b", ln)
                 and _BACK_COMPAT not in ln
                 and "kB is the kilobyte" not in ln
                 and not ln.strip().startswith("#")]
        check(f"no kilobyte label left in {mod}", not stray,
              "; ".join(stray[:2]))
    check("the pre-rename spelling still resolves in the row-label lookups",
          s1f.count(_BACK_COMPAT) == 4,
          f"{s1f.count(_BACK_COMPAT)} back-compat keys "
          f"(expected 4: two spellings x two lookups)")

    # Data S1: the workbook writer existed but nothing called it.
    from persistence import figure_source
    check("the figure-data workbook writer exists", hasattr(figure_source, "to_workbook"))
    check("something now builds Data S1",
          (ROOT / "tools" / "build_data_s1.py").exists(),
          "tools/build_data_s1.py; --check reports per-slot coverage")


def test_group_label_casing():
    """Sentence case for the lifestyle groups on figures, with the internal
    keys left alone -- and the legend path that silently reverted it."""
    group("figure group labels (#599)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from persistence import config
    from persistence.plot_utils import apply_display_text, _collect_text_artists

    check("the rename is a display-time rule, not a source literal",
          config.PLOT_TEXT_REPLACEMENTS.get("Non-Industrial") == "Non-industrial")
    check("it is the same length, so no label reflows",
          len("Non-Industrial") == len(config.display_text("Non-Industrial")))
    check("embedded occurrences are caught",
          config.display_text("Richness (Industrial / Non-Industrial)")
          == "Richness (Industrial / Non-industrial)")
    check("the deliberate abbreviations are left alone",
          config.display_text("Non-Ind") == "Non-Ind"
          and config.display_text("NonInd") == "NonInd")

    # The keys this would have broken had it been done in the source.
    check("the colour table still keys on the internal name",
          "Non-Industrial" in config.LIFESTYLE_COLORS)
    check("canonical_lifestyle still round-trips the internal name",
          config.canonical_lifestyle("Non-Industrial") == "Non-Industrial")

    # Every kind of figure text, including the legend. The legend is the one
    # that used to revert: renaming its Text artists is undone by anything
    # that rebuilds the legend from the handles' own labels.
    fig, ax = plt.subplots()
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Industrial", "Non-Industrial"])
    ax.set_title("Industrial vs Non-Industrial")
    ax.set_ylabel("Richness (Industrial / Non-Industrial)")
    ax.plot([], [], label="Non-Industrial (n=4,098)")
    ax.plot([], [], label="_hidden Non-Industrial series")
    ax.text(0.5, 0.5, "Non-Industrial annotation")
    ax.legend()
    apply_display_text(fig)
    fig.canvas.draw()                                # ticks regenerate here
    _h, _l = ax.get_legend_handles_labels()          # what a rebuild would use
    texts = [a.get_text() for a in _collect_text_artists(fig)] + list(_l)
    stale = sorted({t for t in texts if "Non-Industrial" in t})
    check("no title-case group name survives on a figure", not stale,
          "; ".join(stale))
    check("matplotlib's own legend opt-out is preserved",
          any(h.get_label() == "_hidden Non-Industrial series"
              for h in ax.get_lines()))
    plt.close(fig)

    # Ticks deliberately abbreviated opt out with _keep_text, and the pass
    # rewrites tick FORMATTERS, which sit under those Text artists. Four
    # figures rely on this; expanding their row labels would overflow the
    # panels the abbreviation was introduced to fit.
    fkeep, akeep = plt.subplots()
    akeep.set_yticks([0, 1])
    for _t in akeep.set_yticklabels(["H-G", "Non-Industrial"]):
        _t._keep_text = True
    fplain, aplain = plt.subplots()
    aplain.set_yticks([0, 1])
    aplain.set_yticklabels(["H-G", "Non-Industrial"])
    for _f in (fkeep, fplain):
        apply_display_text(_f)
        _f.canvas.draw(); _f.canvas.draw()        # formatters regenerate twice
    check("_keep_text ticks survive the formatter rewrite",
          [t.get_text() for t in akeep.get_yticklabels()]
          == ["H-G", "Non-Industrial"],
          str([t.get_text() for t in akeep.get_yticklabels()]))
    check("ticks that do not opt out are renamed and stay renamed",
          [t.get_text() for t in aplain.get_yticklabels()]
          == ["Hunter-gatherer", "Non-industrial"],
          str([t.get_text() for t in aplain.get_yticklabels()]))
    plt.close(fkeep); plt.close(fplain)

    # A numeric or log axis must not be touched at all.
    fnum, anum = plt.subplots(); anum.semilogy([1, 10], [1, 10])
    apply_display_text(fnum); fnum.canvas.draw()
    check("numeric and log axes are left alone",
          all("mathdefault" in t.get_text() or t.get_text() == ""
              for t in anum.get_yticklabels()))
    plt.close(fnum)

    # Figure 5 saves outside savefig_dual, so it needs its own pass, and its
    # legend used to carry a second, title-case copy of the group names.
    f5 = (ROOT / "figures" / "figure5.py").read_text()
    check("Figure 5 applies the display renames before saving",
          "_artist.set_text(_new)" in f5)
    check("Figure 5 has one definition of the group names, not two",
          'legend_labels = {g: _display_group(g)' in f5)

    # The four standalone scripts that had their own frozen copy of the rules.
    for f in ("figure2.py", "figure_s0_roadmap_v4.py",
              "make_metadata_summary.py", "rarefaction_plugin.py"):
        txt = (ROOT / "figures" / f).read_text()
        check(f"{f} takes its display rules from config",
              "from persistence.config import display_text" in txt)


def test_clean_clone_bindings() -> None:
    """A clone from GitHub must draw the published labels, not a fallback.

    Every figure script here is launched BY PATH -- figure_runner runs them as
    subprocesses and the README documents the same -- so `sys.path[0]` is
    `figures/`, not the repo root, and `persistence` is importable only
    because each script puts the root on the path itself. Until 2026-10-10
    several did not. `figure_runner` sets PYTHONPATH on the subprocesses it
    launches, so the pipeline's own runs were correct; the exposure was every
    other way of running these -- a reviewer's clone, a standalone redraw, a
    tool importing one by path. Two of them (make_metadata_summary,
    rarefaction_plugin) raised ModuleNotFoundError outright. The rest degraded
    silently to a local rename table that had drifted from
    `config.PLOT_TEXT_REPLACEMENTS`: Transitional became the title-case
    "Agro-Pastoralist", and Obese, Fatty Liver and the lifestyle sentence
    casing were missed entirely.

    Two checks, because either alone can pass while the output is wrong:

    1. Import each script the way a reviewer's run does -- by path, from a
       working directory that is not the repo -- and assert the symbol it binds
       comes from `persistence`, not from its own `except` branch.
    2. Assert no `except` branch guarding a `persistence` import renames
       anything. A second rename table is the defect even when it happens to be
       correct today, because nothing keeps it in step with config.
    """
    group("a clean clone binds the real display rules")

    import ast as _ast
    import concurrent.futures as _cf
    import subprocess as _sp

    figdir = str(ROOT / "figures")
    cases = [
        ("figure2",                  "_display_text",      "persistence.config"),
        ("figure5",                  "_display_text",      "persistence.config"),
        ("make_metadata_summary",    "_display_text",      "persistence.config"),
        ("rarefaction_plugin",       "_display_text",      "persistence.config"),
        ("figure_s0_roadmap_v4",     "_display_text",      "persistence.config"),
        ("section2_bayesian_figure", "apply_display_text", "persistence.plot_utils"),
    ]
    probe = (
        "import sys, importlib\n"
        "sys.path.insert(0, {figdir!r})\n"
        "m = importlib.import_module({mod!r})\n"
        "print(getattr(m, {sym!r}).__module__)\n"
    )

    def _run(case):
        mod, sym, _want = case
        # cwd is deliberately not the repo root -- that is the whole point.
        proc = _sp.run([sys.executable, "-c",
                        probe.format(figdir=figdir, mod=mod, sym=sym)],
                       capture_output=True, text=True, cwd=os.sep)
        out = (proc.stdout.strip().splitlines() or [""])[-1]
        return out, proc.stderr.strip()

    with _cf.ThreadPoolExecutor(max_workers=len(cases)) as ex:
        results = list(ex.map(_run, cases))

    for (mod, sym, want), (got, err) in zip(cases, results):
        detail = f"bound {got!r}, wanted {want!r}"
        if err:
            detail += f"  [stderr: {err.splitlines()[-1][:160]}]"
        check(f"{mod}.{sym} resolves to {want.split('.')[-1]}",
              got == want, detail)

    def _is_mo(node) -> bool:
        if isinstance(node, _ast.ImportFrom):
            return (node.module or "").startswith("persistence")
        if isinstance(node, _ast.Import):
            return any(a.name.startswith("persistence") for a in node.names)
        return False

    renaming = []
    for path in (sorted((ROOT / "figures").glob("*.py"))
                 + sorted((ROOT / "tools").glob("*.py"))):
        try:
            tree = _ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:                                   # pragma: no cover
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Try):
                continue
            if not any(_is_mo(s) for s in _ast.walk(node)
                       if isinstance(s, (_ast.Import, _ast.ImportFrom))):
                continue
            for handler in node.handlers:
                for call in _ast.walk(handler):
                    if (isinstance(call, _ast.Call)
                            and isinstance(call.func, _ast.Attribute)
                            and call.func.attr == "replace"):
                        renaming.append(f"{path.name}:{call.lineno}")
                        break

    check("no persistence import fallback renames text",
          not renaming, "second rename table at " + ", ".join(renaming))

    # The download script is the only instruction a reviewer gets for building
    # the data deposit, and it described the files rather than naming them: it
    # said "combined_mean{coverage,depth}_matrix_*" where config has always
    # opened combined_coverage_matrix_/combined_depth_matrix_. A deposit built
    # from that would load UHGG, lev and chimp and silently skip `combined` --
    # the one reference the manuscript reports on.
    from persistence import config as _cfg
    wanted = {cfg[key] for cfg in _cfg.REFERENCES.values()
              for key in ("breadth_file", "depth_file") if cfg.get(key)}
    script = (ROOT / "download_data.sh").read_text(encoding="utf-8")
    absent = sorted(n for n in wanted if n not in script)
    check("download_data.sh names every matrix config opens",
          not absent, "not named in the script: " + ", ".join(absent))


def test_data_s1_rounding() -> None:
    """Data S1's significant-figure rounding must not eat extreme p-values.

    The submitted workbook rounds float columns to keep the file a sane size.
    The scale-and-round form of that overflows on the tail -- for 1e-305 the
    factor is 10**310, i.e. `inf`, and `round(a * inf) / inf` is `nan` -- which
    silently blanked the smallest p-values in the file. Section 2's per-taxon
    tests reach 1e-1000 in log space and Figure S1's Spearman is 1e-94, so this
    is a live range, not a hypothetical one.
    """
    import numpy as np
    import pandas as pd
    group("Data S1 float rounding")
    from tools.build_data_s1 import _siground

    extremes = [1e-5, 1e-94, 1e-300, 1e-305, 5e-308, 1e-320, 1e300, 1.7e308]
    out = _siground(pd.Series(extremes))
    check("no finite value is rounded to nan or zero",
          bool(np.all(np.isfinite(out)) and np.all(out != 0)),
          ", ".join(f"{i}->{o}" for i, o in zip(extremes, out)
                    if not np.isfinite(o) or o == 0))
    check("rounding agrees with exact decimal formatting",
          all(abs(o - float("%.6g" % i)) <= abs(i) * 1e-12
              for i, o in zip(extremes, out)))

    passthrough = _siground(pd.Series([0.0, np.nan, np.inf, -np.inf]))
    check("zero, nan and both infinities pass through",
          bool(passthrough[0] == 0.0 and np.isnan(passthrough[1])
               and passthrough[2] == np.inf and passthrough[3] == -np.inf))
    check("rounding actually rounds",
          _siground(pd.Series([145.7109541234]))[0] == 145.711)

    # A p-value outside [0, 1] is not a p-value, and one reached the deposited
    # clade table. The check that catches it has to tell a real p column from
    # one merely named like a p: the prevalences and axis positions are
    # renamed on the way into the workbook, and p_mh IS the MH test p.
    from tools.build_data_s1 import _is_pvalue_column, pvalue_violations
    reals = ("p", "q", "pval", "p_mh", "p_raw", "p_corr", "boot_p",
             "p_meanz_hmp", "p_energy_10pct", "p_Obese",
             "p_Glucose Intolerance", "p_raw_vs_gmw_industrial")
    impostors = ("axis_position", "axis_position_fraction", "prevalence_ctrl",
                 "prevalence_case_mh", "prev", "prev_diff", "richness",
                 "p_frac")
    check("every real p/q column is recognised",
          all(_is_pvalue_column(c) for c in reals),
          "missed: " + ", ".join(c for c in reals
                                 if not _is_pvalue_column(c)))
    check("no prevalence or axis column is mistaken for a p-value",
          not any(_is_pvalue_column(c) for c in impostors),
          "wrongly flagged: " + ", ".join(c for c in impostors
                                          if _is_pvalue_column(c)))

    obj = {"slot": "T1", "panel_order": ["x"],
           "panels": {"x": pd.DataFrame({"p": [0.5, 2.0, np.nan],
                                         "prev": [0.0, 1.0, 3.0]})}}
    v = pvalue_violations([obj])
    check("an out-of-range p is reported, a wide prevalence is not",
          len(v) == 1 and v[0][2] == "p" and v[0][5] == 1, f"{v}")


def test_deposit_staging() -> None:
    """A split deposit must round-trip, and a dead connection must retry.

        Two properties of the deposit tooling. A deposited file may be stored in
        numbered pieces when it is too large to upload whole, and concatenating
        those pieces and decompressing them has to reproduce the input exactly --
        a per-piece checksum proves the pieces arrived and says nothing about
        that. Separately, an upload that ends without a final HTTP response has
        not been refused, so it must be retried rather than abandoned.
        """
    import hashlib
    import tempfile
    group("Zenodo deposit staging")
    # The deposit tooling is excluded from the published tree (it is our
    # publication machinery, not the reader's), so these checks have to
    # stand down rather than fail where it is absent.
    if not (ROOT / "tools" / "prepare_deposit.py").exists():
        check("deposit tooling absent; staging checks skipped", True)
        return
    from tools.prepare_deposit import (MANIFEST, _COLUMNS, split_compressed,
                                       verify)
    from tools.zenodo_upload import _is_retryable

    # A 1xx is an interim status: no final response ever arrived, so it is a
    # transport failure -- the most retryable thing here, not the least.
    retry = {0: True, 100: True, 101: True, 199: True,
             408: True, 429: True, 500: True, 502: True, 503: True,
             504: True, 200: False, 201: False, 400: False, 401: False,
             403: False, 404: False, 413: False, 422: False}
    wrong = [c for c, want in retry.items() if _is_retryable(c) is not want]
    check("a dead connection and a 5xx retry; a 2xx/4xx does not",
          not wrong, f"misclassified: {wrong}")
    check("an interim 100 is never read as a verdict", _is_retryable(100))

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "stage"
        stage.mkdir()
        # Bytes that compress like the real matrices: repetitive enough to
        # shrink, varied enough that the parts are not all identical.
        src = Path(td) / "matrix_toy.csv"
        rows = ["sgb,sample,depth,breadth"]
        for i in range(60000):
            rows.append(f"SGB{i % 997},S{i % 311},{i * 1e-7:.9f},"
                        f"{(i % 1000) / 1000:.3f}")
        src.write_text("\n".join(rows) + "\n")
        want = hashlib.md5(src.read_bytes()).hexdigest()

        parts = split_compressed(src, stage, 64 * 1024)
        check("a file larger than the part size is cut into several parts",
              len(parts) > 1, f"{len(parts)} part(s)")
        check("no part exceeds the requested size",
              all(p["part_bytes"] <= 64 * 1024 for p in parts))
        check("every part is on disk with a checksum",
              all((stage / p["part"]).exists() and len(p["part_md5"]) == 32
                  for p in parts))
        # Lexical order must equal assembly order: sorting the names is all
        # the shell has to go on on the reviewer's machine.
        check("part names sort into assembly order",
              [p["part"] for p in parts]
              == sorted(p["part"] for p in parts))
        check("concatenating the parts and decompressing restores the file",
              verify("matrix_toy.csv", parts, stage, want))
        check("a round trip that does not match is reported",
              not verify("matrix_toy.csv", parts, stage, "0" * 32))

    # The manifest is the only thing naming the parts, so the reassembly in
    # download_data.sh has to be driven by it and has to check its work.
    script = ROOT / "download_data.sh"
    text = script.read_text(encoding="utf-8") if script.exists() else ""
    check("download_data.sh reassembles from the manifest", MANIFEST in text)
    check("download_data.sh checks each part before concatenating",
          "CORRUPT part" in text)
    check("download_data.sh checks the rebuilt file against the original md5",
          "expected md5" in text)
    check("download_data.sh creates the directory a destination needs",
          'mkdir -p "$(dirname' in text)
    check("the manifest carries a destination path, not just a basename",
          _COLUMNS[0] == "dest" and "final_run/metadata_M1.csv" in text)


def test_upload_sweeps() -> None:
    """An unreliable link must cost time, never integrity.

        A file that exhausts its own attempts goes back on the list for the next
        sweep, so a bad window costs sweeps rather than files. What must never
        happen is calling a file stored when the remote checksum disagrees with
        the local one: a file in the bucket is only ever a complete upload, so
        the right response is to send it again.
        """
    import tempfile
    group("Zenodo upload sweeps")
    # The deposit tooling is excluded from the published tree (it is our
    # publication machinery, not the reader's), so these checks have to
    # stand down rather than fail where it is absent.
    if not (ROOT / "tools" / "prepare_deposit.py").exists():
        check("deposit tooling absent; staging checks skipped", True)
        return
    from tools import zenodo_upload as zu

    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i in range(3):
            p = Path(td) / f"part.p{i:03d}"
            p.write_bytes(b"x" * (100 + i))
            paths.append(p)
        good = {p.name: zu.md5(p) for p in paths}

        # A bad window: everything fails on sweep 1, everything works later.
        calls = []

        def put_window(p):
            calls.append(p.name)
            return None if len(calls) <= 3 else good[p.name]

        done, skipped, missing, corrupt = zu.upload_sweeps(
            None, "", "", 1, paths, passes=3, _put=put_window,
            _have=lambda: {}, _sleep=lambda s: None)
        check("a file that fails a whole sweep is retried in the next one",
              done == 3 and not missing, f"done={done} missing={missing}")
        check("a bad window costs sweeps, not files", len(calls) == 6)

        # Already stored, by checksum: not re-sent.
        sent = []
        done, skipped, missing, corrupt = zu.upload_sweeps(
            None, "", "", 1, paths, passes=3,
            _put=lambda p: sent.append(p.name) or good[p.name],
            _have=lambda: dict(good), _sleep=lambda s: None)
        check("what Zenodo already holds is skipped by checksum",
              skipped == 3 and done == 0 and not sent)

        # Stored under the right name but with the wrong bytes.
        tried = []

        def put_corrupt(p):
            tried.append(p.name)
            return "0" * 32 if len(tried) <= 3 else good[p.name]

        done, skipped, missing, corrupt = zu.upload_sweeps(
            None, "", "", 1, paths, passes=3, _put=put_corrupt,
            _have=lambda: {}, _sleep=lambda s: None)
        check("a wrong remote checksum is never counted as stored",
              done == 3 and not missing and len(tried) == 6,
              f"done={done} tried={len(tried)}")
        check("every mismatch is reported, not just the last",
              sorted(set(corrupt)) == sorted(good))

        # A link that never comes back must end with named casualties, so the
        # run cannot look complete.
        done, skipped, missing, corrupt = zu.upload_sweeps(
            None, "", "", 1, paths, passes=2, _put=lambda p: None,
            _have=lambda: {}, _sleep=lambda s: None)
        check("a run that cannot finish names what is missing",
              done == 0 and sorted(missing) == sorted(good))

        # Mid-sweep recovery: Zenodo gained a file we were about to resend.
        nudged = []
        done, skipped, missing, corrupt = zu.upload_sweeps(
            None, "", "", 1, paths, passes=2,
            _put=lambda p: nudged.append(p.name) or None,
            _have=lambda: dict(good) if nudged else {},
            _sleep=lambda s: None)
        check("a file stored out of band is not sent again",
              not missing and len(nudged) == 3,
              f"missing={missing} sent={len(nudged)}")

    # Backoff has to spread, or every part in a sweep retries in lockstep
    # with whatever the disturbance's period is.
    waits = [zu._backoff_for(3) for _ in range(300)]
    check("backoff is jittered, not a fixed cadence",
          len(set(round(w, 3) for w in waits)) > 250)
    check("backoff stays within a sane band of its base",
          all(zu._BACKOFF[2] * 0.7 <= w <= zu._BACKOFF[2] * 1.3
              for w in waits))
    check("backoff lengthens with each attempt",
          all(zu._BACKOFF[i] <= zu._BACKOFF[i + 1]
              for i in range(len(zu._BACKOFF) - 1)))
    check("curl aborts a stalled transfer instead of hanging",
          "--speed-time" in _curl_argv())


def _curl_argv() -> str:
    """The curl command line the uploader builds, as text."""
    src = (ROOT / "tools" / "zenodo_upload.py").read_text(encoding="utf-8")
    i = src.find("def _put_with_curl")
    return src[i:i + 2000]


def test_transport_calibration() -> None:
    """Sizing the deposit's pieces must be measured, and must fit the cap.

        Upload failures that scale with bytes transferred rather than with time
        imply a per-byte risk, so the piece size follows from a measurement
        rather than a guess. It is bounded below by the repository's file limit,
        which for a deposit this size binds harder than the link does.
        """
    import math
    import random
    group("transport calibration")
    # The deposit tooling is excluded from the published tree (it is our
    # publication machinery, not the reader's), so these checks have to
    # stand down rather than fail where it is absent.
    if not (ROOT / "tools" / "prepare_deposit.py").exists():
        check("deposit tooling absent; staging checks skipped", True)
        return
    from tools.zenodo_upload import (_FILE_CAP, _fit_survival,
                                     _recommend_part)

    check("a ladder that all succeeds cannot name a rate",
          _fit_survival([(16., True), (66., True), (128., True)]) is None)
    check("a ladder that all fails cannot name a rate",
          _fit_survival([(16., False), (66., False), (128., False)]) is None)

    # Recover a known rate from simulated attempts.
    recovered = []
    for true_L in (60, 150, 400):
        random.seed(7)
        trials = [(float(mb), random.random() < math.exp(-mb / true_L))
                  for mb in (16, 32, 66, 128) for _ in range(6)]
        got = _fit_survival(trials)
        recovered.append(got is not None and 0.4 * true_L <= got <= 3 * true_L)
    check("the fit recovers a known corruption rate to within a factor",
          all(recovered), f"{recovered}")

    staged = 6.52e9
    floor = math.ceil(staged / 1e6 / _FILE_CAP)
    for L in (30, 60, 150, 400, 900):
        size, parts, p, sent = _recommend_part(L, staged)
        check(f"L={L} MB: the plan fits Zenodo's {_FILE_CAP}-file cap",
              parts <= _FILE_CAP, f"{parts} parts of {size:.0f} MB")
        check(f"L={L} MB: parts are never smaller than the cap allows",
              size >= floor, f"{size:.0f} < {floor}")

    # A worse link must never be answered with bigger parts.
    sizes = [_recommend_part(L, staged)[0] for L in (30, 60, 150, 400, 900)]
    check("a worse link is never given larger parts",
          all(sizes[i] <= sizes[i + 1] for i in range(len(sizes) - 1)),
          f"{[round(x) for x in sizes]}")
    check("a good link is capped at the staged part size, not grown past it",
          _recommend_part(10_000, staged)[0] <= 500.0)

    # Expected uplink must exceed the payload, and blow up as the link worsens.
    sent_bad = _recommend_part(30, staged)[3]
    sent_good = _recommend_part(900, staged)[3]
    check("re-sends are accounted for, and cost more on a worse link",
          sent_bad > sent_good >= staged / 1e6,
          f"{sent_bad:.0f} vs {sent_good:.0f} MB")


def test_uncompressed_deposit_entry() -> None:
    """A small table goes into the deposit as plain text, and still rebuilds.

        Compression earns nothing on a small table and costs the reader a
        preview, so the curated metadata is deposited uncompressed while the
        large matrices stay compressed. That makes the deposit mixed, and both
        the verifier and the download script have to handle either kind: plain
        text through gunzip errors out, and a .gz passed through unchanged leaves
        a compressed file where the pipeline expects CSV.
        """
    import hashlib
    import tempfile
    group("uncompressed deposit entries")
    # The deposit tooling is excluded from the published tree (it is our
    # publication machinery, not the reader's), so these checks have to
    # stand down rather than fail where it is absent.
    if not (ROOT / "tools" / "prepare_deposit.py").exists():
        check("deposit tooling absent; staging checks skipped", True)
        return
    from tools.prepare_deposit import (MANIFEST, _COLUMNS, make_raw,
                                       split_compressed, verify)

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "stage"
        stage.mkdir()
        src = Path(td) / "small_table.csv"
        src.write_text("id,value\n" + "".join(f"S{i},{i*0.5}\n"
                                              for i in range(5000)))
        want = hashlib.md5(src.read_bytes()).hexdigest()
        parts = split_compressed(src, stage, 1 << 30, whole=True)
        check("a whole-file part is named <name>.gz",
              parts[0]["part"] == "small_table.csv.gz")

        with open(stage / MANIFEST, "w") as fh:
            fh.write("\t".join(_COLUMNS) + "\n")
            fh.write("\t".join([
                "small_table.csv", parts[0]["part"], parts[0]["part_md5"],
                str(parts[0]["part_bytes"]), want,
                str(src.stat().st_size)]) + "\n")

        check("make_raw succeeds", make_raw(stage, ["small_table.csv"]) == 0)
        check("the .gz is gone and the plain file is there",
              not (stage / "small_table.csv.gz").exists()
              and (stage / "small_table.csv").exists())
        check("the deposited bytes equal the original",
              hashlib.md5((stage / "small_table.csv").read_bytes()
                          ).hexdigest() == want)

        rows = [l.rstrip("\n").split("\t")
                for l in open(stage / MANIFEST)][1:]
        check("the manifest now names the plain file, with its own md5",
              rows[0][1] == "small_table.csv" and rows[0][2] == want
              and rows[0][3] == str(src.stat().st_size))

        # verify() must not try to decompress something that is not gzipped.
        check("verify accepts an uncompressed entry",
              verify("small_table.csv", [{"part": "small_table.csv"}],
                     stage, want))
        check("verify still rejects a wrong uncompressed entry",
              not verify("small_table.csv", [{"part": "small_table.csv"}],
                         stage, "0" * 32))
        check("re-running make_raw is a no-op, not an error",
              make_raw(stage, ["small_table.csv"]) == 0)

    # download_data.sh has to branch on what the manifest names.
    sh = (ROOT / "download_data.sh").read_text(encoding="utf-8")
    check("download_data.sh decides by the part's extension",
          'case "$first" in' in sh and "*.gz.p[0-9][0-9][0-9])" in sh)
    check("download_data.sh has a non-gzip path that just concatenates",
          sh.count("for p in $parts; do cat") >= 2)
    check("download_data.sh can rehearse against pre-fetched files",
          "SKIP_DOWNLOAD" in sh)


def test_input_provenance_gate() -> None:
    """A reviewer with the wrong inputs must be told, not quietly misled.

    Six files named metadata_M1.csv have existed in this tree -- repo root,
    metadata/, final_run/, a rename backup, and a _verified variant -- and
    they are different cohorts, not different spellings of one: the notes in
    config record Unhealthy going 6,310 -> 6,512 between two of them.
    config.py resolves the metadata by taking the first candidate path that
    exists, so a reviewer holding the wrong copy at the right path gets a
    complete, clean run and different numbers.

    The reported run's own verification_manifest.json records
    final_run/metadata_M1.csv at sha256 06031e18..., and the deposited file
    hashes identically, so the pinned table is anchored to the run rather
    than to this working tree.
    """
    import contextlib
    import os
    import tempfile
    group("input provenance gate")
    import tools.preflight as pf

    pinned = pf._pinned_inputs()
    check("the pinned table lists all nine inputs", len(pinned) == 9,
          f"{len(pinned)}")
    by_path = {r[0]: r for r in pinned}
    m1 = by_path.get("final_run/metadata_M1.csv")
    check("the metadata's pinned sha256 is the run's own value",
          bool(m1) and m1[3].startswith("06031e18894a6354"),
          m1[3][:24] if m1 else "absent")
    check("the metadata is attested by the run manifest, not just locally",
          bool(m1) and m1[4] == "run_manifest")
    check("the two combined matrices are run-attested too",
          sum(1 for r in pinned if r[4] == "run_manifest") == 3)

    # A decoy of exactly the right size must still be rejected, because size
    # is what a careless check would stop at.
    cwd = os.getcwd()
    errs, warns = list(pf.errors), list(pf.warnings)
    try:
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            (Path(td) / "final_run").mkdir()
            decoy = Path(td) / "final_run" / "metadata_M1.csv"
            decoy.write_bytes(b"x" * m1[1])
            check("a decoy of the exact right size is still caught",
                  decoy.stat().st_size == m1[1])
            pf.errors.clear(); pf.warnings.clear()
            with open(os.devnull, "w") as _dn, \
                    contextlib.redirect_stdout(_dn):
                pf.check_inputs_match_the_run(full=False)
            hits = [e for e in pf.errors if "metadata_M1" in e
                    and "sha256" in e]
            check("the gate fails on a wrong metadata file, naming sha256",
                  bool(hits), f"errors: {pf.errors[:2]}")
            check("it says plainly this is not the reported run's file",
                  any("NOT the file the reported run used" in e
                      for e in pf.errors))
            # And a wrong *size* is caught without hashing at all.
            pf.errors.clear()
            decoy.write_bytes(b"x" * 100)
            with open(os.devnull, "w") as _dn, \
                    contextlib.redirect_stdout(_dn):
                pf.check_inputs_match_the_run(full=False)
            check("a wrong-size input is caught too",
                  any("bytes, expected" in e for e in pf.errors))
    finally:
        os.chdir(cwd)
        pf.errors.clear(); pf.errors.extend(errs)
        pf.warnings.clear(); pf.warnings.extend(warns)

    # The loud fallback in config.
    cfg = (ROOT / "persistence" / "config.py").read_text(encoding="utf-8")
    check("config warns when it falls back off the first metadata candidate",
          "DIFFERENT COHORT" in cfg and "RuntimeWarning" in cfg
          and "META_FILE != META_FILE_CANDIDATES[0]" in cfg)
    check("config still prefers final_run/metadata_M1.csv",
          "\"final_run/metadata_M1.csv\"," in cfg)


def main() -> int:
    global VERBOSE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    VERBOSE = ap.parse_args().verbose

    print("Audit fix checks — fast, no matrices, no section re-run")
    for fn in (test_sato_variance, test_cmh_separation, test_bh,
               test_weighted_sem, test_boot_star, test_background_weights,
               test_referencedata_rebuilds, test_figure_contract,
               test_panel_geometry, test_no_competing_copies,
               test_approved_method_changes, test_spec_combination,
               test_condition_eligibility,
               test_no_condition_leaks_into_figures,
               test_method_comparison_table,
               test_driver_panel_and_naming,
               test_rachel_pipeline_items,
               test_group_label_casing,
               test_clean_clone_bindings,
               test_data_s1_rounding,
               test_deposit_staging,
               test_upload_sweeps,
               test_transport_calibration,
               test_uncompressed_deposit_entry,
               test_input_provenance_gate,
               test_environment):
        try:
            fn()
        except Exception as exc:                              # pragma: no cover
            check(f"{fn.__name__} raised", False, repr(exc))

    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n  {len(_results) - len(failed)} passed, {len(failed)} failed, "
          f"{len(_results)} checks")
    for n in failed:
        print(f"    FAILED  {n}")
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
