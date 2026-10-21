#!/usr/bin/env python3
"""Redraw a Supplementary Figure from its data object alone.

Each figure publishes one `Figure_S<n>_data.pkl` holding every panel it plots.
This reads that and calls the drawing code, with nothing else in scope -- no
reference matrices, no section re-run. A figure that redraws this way is
reproducible from its published data, which is the same claim the journal's
data S1 requirement makes.

A slot counts as redrawn only when a file appears or changes on disk. Several
drawers return None by design, so a drawer that bailed out early -- a data
object missing a panel -- used to be reported as a success while whatever stale
figure was already there went unchanged.

    python tools/replot_from_data.py --list      # what can redraw today
    python tools/replot_from_data.py S4          # redraw one
    python tools/replot_from_data.py --all       # redraw everything wired

Run it on the pipeline's interpreter (plotting_env; see
tools/package_for_transfer.sh). It refuses to run without seaborn, which the
violin figures need to come out as the pipeline draws them.

A redraw writes in place, over the figure in results/, and then rebuilds the
two reader-facing views beside that tree -- supplementary_figures/ and
staged_figures/ -- exactly as a full run's last steps do. So a fixed figure
lands where a full run would put it, and nowhere else. To fix a finished run's
package rather than the local tree:

    python tools/replot_from_data.py S4 S23 --results-root <pkg>/results

(the package ships supplementary_figures/ as real files, so it is refreshed
by copying). --out writes previews to a scratch directory and leaves both
views alone; never copy those previews into a manuscript by hand.
"""
from __future__ import annotations

import argparse
import sys, importlib.util as ilu, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
from persistence import figure_source                            # noqa: E402
from persistence.plot_utils import apply_oi_style                # noqa: E402


def _fs():
    spec = ilu.spec_from_file_location("figure_sets", ROOT / "tools/figure_sets.py")
    m = ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


# slot -> how to draw it from its tables. Only figures whose drawing code takes
# the tables and nothing else can appear here; the rest still have their
# computation entangled with their drawing and need splitting first.
def _draw_S4(obj, outdir):
    from persistence.depth_audit import plot_depth_by_lifestyle
    return plot_depth_by_lifestyle(
        {k: obj["panels"][k] for k in obj["panel_order"]}, outdir)


def _draw_S23(obj, outdir):
    from persistence.depth_audit import plot_depth_by_health
    return plot_depth_by_health(
        {k: obj["panels"][k] for k in obj["panel_order"]}, outdir)


def _draw_S2(obj, outdir):
    from persistence.section1_tech_replicates import plot_alpha_scatter
    from persistence.detection import DetectionSpec
    from persistence import config
    spec = DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)
    return plot_alpha_scatter(
        {k: obj["panels"][k] for k in obj["panel_order"]}, spec, outdir)


def _draw_S3(obj, outdir):
    from persistence.section1_tech_replicates import plot_depth_vs_bc
    from persistence.detection import DetectionSpec
    from persistence import config
    spec = DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)
    return plot_depth_vs_bc(
        {k: obj["panels"][k] for k in obj["panel_order"]}, outdir, spec)


def _spec_10pct():
    from persistence.detection import DetectionSpec
    from persistence import config
    return DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)


def _draw_demographic(obj, outdir):
    """Demographic associations. Withdrawn from the numbered set on 2026-09-27,
    so it is not a slot any more; the script owns its own rebuild path and this
    stays reachable for it."""
    import os
    from figures.make_demographic_associations import rebuild_from_data_object
    stem = obj.get("stem") or "demographic_associations_combined_10pct"
    return rebuild_from_data_object(obj, os.path.join(outdir, stem + ".png"))


# The prevalence-matched panels drawer lived here. That figure was the old
# S25 -- the naive comparison -- and was retired on 2026-10-09 once Supplement
# Note 6 was cut back to the mean-z statistic; `_plot_prevalence_matched` no
# longer emits a slot. The S25 key below draws the background curve, which is
# `_draw_S26`'s figure under its pre-renumbering name.


def _family_driver_drawer(obj, outdir):
    """S26-S29: the taxon-driver family panels, from their data object.

    The object carries which family it is and which conditions it drew, so the
    redraw reproduces the published figure rather than re-deciding either. The
    taxonomy lookup is rebuilt from the table's own `taxonomy` column: the
    drawing builds its axis labels from the full GTDB string, and handing it
    the rendered short label instead makes it fall back to raw accessions.
    """
    from persistence import health_driver_family_panels as hdp
    df = obj["panels"]["drivers"]
    extras = obj.get("extras") or {}
    family = extras.get("family")
    if family is None:                      # pre-family object: nothing to do
        return None
    conds = list(extras.get("drawn_conditions") or [])
    tax = (dict(zip(df["taxon"], df["taxonomy"]))
           if {"taxon", "taxonomy"} <= set(df.columns) else {})
    return hdp.draw(df, tax, family, outdir, conds,
                    ref=obj.get("reference", "combined"), spec_tag="10pct",
                    spec_desc=_spec_10pct().description,
                    q_cut=float(extras.get("q_cut", hdp.Q_CUT)))


def _taxon_driver_drawer(list_name):
    def _draw(obj, outdir):
        from persistence.section3_health import _plot_taxon_drivers
        df = obj["panels"]["drivers"]
        # The drawing builds its axis labels from the full taxonomy string,
        # so hand back the same lookup it had rather than the rendered label:
        # passing the short label makes it fall back to raw SGB identifiers.
        tcol = "taxon" if "taxon" in df.columns else "Taxon"
        tax = (dict(zip(df[tcol], df["taxonomy"]))
               if {tcol, "taxonomy"} <= set(df.columns) else {})
        return _plot_taxon_drivers(
            df, tax, obj.get("reference", "combined"), outdir, "10pct",
            _spec_10pct().description, focus_lists=[list_name])
    return _draw


def _sweep_drawer(name):
    def _draw(obj, outdir):
        from persistence.depth_bias_ipw import _make_sweep_figure
        import os
        # S8 and S14 are distinct figures sharing this stem in two different
        # result directories. They only collide when both are written into one
        # --out directory, so the prefix is applied there and not in place.
        out = os.path.join(outdir, obj.get("_prefix", "") + f"{name}.png")
        _make_sweep_figure(obj["panels"]["sweep"], out)
        return out
    return _draw


def _draw_S18(obj, outdir):
    """Axis-angle fan and angle panels, from the per-bin frame."""
    from persistence import axis_angle_analysis as m
    m._plot(obj["panels"]["bins"], "10pct", outdir,
            obj.get("reference", "combined"),
            m.N_BINS_DEFAULT, m.MIN_N_HG_DEFAULT, m.MIN_N_OTHER_DEFAULT)
    import os
    return os.path.join(outdir, "axis_angle_v2_10pct.png")


def _gmw_main_drawer(stem):
    def _draw(obj, outdir):
        from persistence.depth_bias_ipw import plot_gmw_main
        import os
        out = os.path.join(outdir, obj.get("_prefix", "") + f"{stem}.png")
        plot_gmw_main(obj["panels"], out)
        return out
    return _draw


def _draw_S7(obj, outdir):
    from persistence.depth_bias_ipw import plot_gmw_pairwise
    import os
    out = os.path.join(outdir, obj.get("_prefix", "")
                       + "carter_richness_pairwise_corrected.png")
    plot_gmw_pairwise(obj["panels"], out)
    return out


def _draw_S9(obj, outdir):
    from persistence.carter_analysis import plot_weight_diagnostics
    import os
    out = os.path.join(outdir, obj.get("_prefix", "")
                       + "carter_gmw_weight_diagnostics_10pct.png")
    plot_weight_diagnostics(obj["panels"], out, obj.get("reference", "combined"),
                            obj.get("notes", ""))
    return out


def _health_gmw_drawer(set_label):
    def _draw(obj, outdir):
        from persistence.section3_health import plot_health_richness_gmw
        import os
        out = os.path.join(outdir,
                           obj.get("_prefix", "") + f"{obj.get('stem','health')}.png")
        # notes is "<set title>. <detection description>"; the drawing wants
        # only the second half, or the suptitle carries the set title twice
        # and the tight bounding box grows to fit it.
        notes = str(obj.get("notes", ""))
        title, _, desc = notes.partition(". ")
        plot_health_richness_gmw(
            obj["panels"], out, set_label, set_title=title,
            ref=obj.get("reference", "combined"),
            spec_desc=desc or notes)
        return out
    return _draw


def _draw_S26(obj, outdir):
    """The prevalence background: every list taxon against the fitted curve.

    S26 since 2026-10-07, when the VanishH+/BlossumH+ form of this figure
    became main-text Figure 5E and the supplement kept only the base VANISH
    and BloSSUM pair. An object published before 2026-10-06 holds the
    importance-weighting diagnostic the figure replaced (one `iw` table), so
    it is redrawn as the figure it was published as.
    """
    import os
    panels = obj["panels"]
    if "list_taxa" in panels:
        from persistence.section3_health import rebuild_background_curve
        return rebuild_background_curve(
            panels, outdir, ref=obj.get("reference", "combined"),
            notes=obj.get("notes", ""))
    from persistence.section3_health import plot_iw_diagnostics
    plot_iw_diagnostics(panels, "10pct", outdir)
    return os.path.join(outdir, "health_iw_diagnostics_10pct.png")


def _draw_within_clade(obj, outdir):
    """Figure S30: within-clade contrasts, list members vs their relatives."""
    from persistence.section3_health import rebuild_within_clade_contrasts
    return rebuild_within_clade_contrasts(
        obj["panels"], outdir, ref=obj.get("reference", "combined"),
        notes=obj.get("notes", ""))


def _draw_S19(obj, outdir):
    from persistence.axis_angle_analysis import plot_study_centroids
    return plot_study_centroids(obj["panels"], "10pct", outdir,
                                obj.get("reference", "combined"))


def _draw_S20(obj, outdir):
    """Depth-binned richness CV. The dashed grand-mean lines are the mean CV
    over qualifying bins, so they come back out of the same table."""
    from persistence.within_group_stats import _plot_depthbin
    from persistence.depth_bias_ipw import _SpecLabel
    from persistence import config
    import os
    df = obj["panels"]["bins"]
    min_n = 10
    grand_cv = {}
    for lf, g in df.groupby("Lifestyle", sort=False):
        vals = g.loc[g["n"] >= min_n, "cv_sd_median"].dropna()
        if len(vals):
            grand_cv[lf] = float(vals.mean())
    spec = _SpecLabel(str(obj.get("notes", "")) or
                      f"breadth >= {config.ACCUM_MAIN_TEXT_BREADTH}%")
    # Match the type scale the pipeline draws this figure at, or the saved
    # canvas comes out a different width and the printed point size shifts.
    from persistence.plot_utils import apply_oi_style, current_fs
    apply_oi_style()
    _plot_depthbin(df, grand_cv, spec, "10pct", outdir,
                   obj.get("reference", "combined"), min_n, current_fs())
    return os.path.join(outdir, "depth_binned_richness_cv_10pct.png")


def _draw_S22(obj, outdir):
    from persistence.section2_pooling import plot_continent_lifestyle_heatmap
    import os
    out = os.path.join(outdir, "s2c_continent_lifestyle_heatmap_10pct.png")
    return plot_continent_lifestyle_heatmap(obj["panels"], out,
                                            str(obj.get("notes", "")))


def _draw_S21(obj, outdir):
    from persistence.supp_s5 import plot_s5b
    return plot_s5b(obj["panels"], obj.get("reference", "combined"), outdir)


def _draw_S15(obj, outdir):
    """Figure 3: six accumulation panels, redrawn from the published curves."""
    import sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "figures"))
    from make_combined_figure3 import plot_figure3
    return str(plot_figure3(obj["panels"], outdir, model="hill",
                            threshold=10.0))


def _draw_read_accounting(obj, outdir):
    import sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "figures"))
    from make_read_accounting import draw
    return draw({k: obj["panels"][k] for k in obj["panel_order"]}, outdir,
                stem=obj.get("stem") or "read_accounting")


def _draw_S31(obj, outdir):
    """The roadmap is a schematic; its data is the cohort counts it prints."""
    import sys, os
    # The pipeline draws the roadmap as a standalone script, on matplotlib's
    # defaults; the house style main() applies first would change its font.
    matplotlib.rcdefaults()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "figures"))
    from figure_s0_roadmap_v4 import draw_roadmap
    df = obj["panels"]["cohort_counts"]
    counts = {str(r["label"]): r["value"] for _, r in df.iterrows()}
    out = os.path.join(outdir, "figure_s0_roadmap.pdf")
    draw_roadmap(counts, output_path=out)
    return out.replace(".pdf", ".png")


def _draw_S10(obj, outdir):
    from persistence.section1_figure import plot_robustness_alpha
    return plot_robustness_alpha(obj["panels"], outdir,
                                 obj.get("reference", "combined"))


def _sinf_drawer(stem, title):
    def _draw(obj, outdir):
        import sys, os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(root, "tools"))
        from make_accumulation_sinf_figures import plot_sinf_figure
        return plot_sinf_figure(obj["panels"], title,
                                os.path.join(outdir, stem + ".png"))
    return _draw


def _draw_S5(obj, outdir):
    from persistence.section1_figure import plot_read_length_diagnostic
    return plot_read_length_diagnostic(obj["panels"], outdir,
                                       obj.get("reference", "combined"))


# Slot -> drawer. The keys are slot numbers as the manuscript cites them, so
# they moved with the 2026-10-19 renumbering of
# tools/additional_figures_manifest.txt (S5 -> S14 with S6..S14 -> S5..S13,
# and S30 -> S26 with S26..S29 -> S27..S30). The `# was Sn` notes record where
# each drawer sat before that.
#
# Several drawer FUNCTION names still carry the slot they were written for --
# `_draw_S5` now serves S14, `_draw_S26` serves S25 -- which is the convention
# this file already followed across the 2026-10-07 renumbering: the name
# records where a figure came from, the key records where it is cited now.
# Sorted by slot, which the previous arrangement was not.
DRAW = {
    "S1": _draw_read_accounting,
    "S2": _draw_S2,
    "S3": _draw_S3,
    "S4": _draw_S4,
    "S5": _gmw_main_drawer("carter_gmw_richness_main"),    # was S6
    "S6": _draw_S7,                                        # was S7
    "S7": _sweep_drawer("gmw_threshold_sweep"),            # was S8
    "S8": _draw_S9,                                        # was S9
    "S9": _draw_S10,                                       # was S10
    "S10": _gmw_main_drawer("gmw_richness_main"),          # was S11
    "S11": _health_gmw_drawer("lifestyle_all"),            # was S12
    "S12": _health_gmw_drawer("lifestyle3"),               # was S13
    "S13": _sweep_drawer("gmw_threshold_sweep"),           # was S14
    "S14": _draw_S5,                                       # was S5
    "S15": _draw_S15,
    "S16": _sinf_drawer(
        "figure_a17_accumulation_sinf_mink_medium",
        "Figure A17: Combined catalogue accumulation across "
        "repeated-detection thresholds"),
    "S17": _sinf_drawer(
        "figure_a16_accumulation_sinf_thresholds_medium",
        "Figure A16: Combined catalogue accumulation across "
        "detection thresholds"),
    "S18": _draw_S18,
    "S19": _draw_S19,
    "S20": _draw_S20,
    "S21": _draw_S21,
    "S22": _draw_S22,
    "S23": _draw_S23,
    "S24": _health_gmw_drawer("NI_health"),
    "S25": _draw_S26,
    # The within-clade figure leads this run of slots since 2026-10-19,
    # because it is cited before the four taxon-driver family panels that
    # follow it. One drawer serves all four of those: each object names its
    # own family and conditions. `_taxon_driver_drawer` is kept above for an
    # object published before the move, which holds a single list and no
    # `family` extra.
    "S26": _draw_within_clade,                             # was S30
    "S27": _family_driver_drawer,                           # was S26
    "S28": _family_driver_drawer,                           # was S27
    "S29": _family_driver_drawer,                           # was S28
    "S30": _family_driver_drawer,                            # was S29
    "S31": _draw_S31,
}


#: What counts as "a figure was produced". A drawer that writes only its data
#: object or its log has not redrawn anything.
_IMAGE_SUFFIXES = (".png", ".pdf", ".svg", ".eps", ".tif", ".tiff")


def _sig(p: Path):
    """A fingerprint that changes when a file is rewritten, or None if absent.

    mtime alone is too coarse for a redraw that finishes inside the clock's
    granularity, so size goes in too.
    """
    try:
        st = p.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _image_sigs(d: Path) -> dict[Path, tuple]:
    try:
        return {p: _sig(p) for p in d.iterdir()
                if p.suffix.lower() in _IMAGE_SUFFIXES}
    except OSError:
        return {}


def _slot_targets(results_root: Path, ref: str, stem: str) -> list[Path]:
    """Where the slot's figure has to land for the views to pick it up.

    tools/additional_figures_manifest.txt resolves a slot as
    ``<stem>_notitle.png`` else ``<stem>.png``, so either one counts.
    """
    return [results_root / ref / (stem + "_notitle.png"),
            results_root / ref / (stem + ".png")]


def _written_since(targets: list[Path], before_t: dict[Path, tuple],
                   outdir: Path, before_d: dict[Path, tuple]) -> list[Path]:
    """Image files this redraw created or rewrote.

    The slot used to be counted as redrawn whenever the drawer returned without
    raising, and several drawers return None by design -- so the return value
    carried no signal at all. rebuild_within_clade_contrasts returns None when
    its `contrasts` or `points` panel is missing (section3_health.py:3668), and
    `--all` on a tree without that panel printed "S31: redrawn in 0.0s -> None"
    and "31 redrawn, 0 failed" with no PNG anywhere on disk. Ask the
    filesystem instead.

    Every wired drawer writes into the directory it is handed, so scanning that
    directory and the slot's two resolved names covers all of them. A future
    drawer that writes somewhere else would be reported as having written
    nothing -- a false alarm, but one whose message names the path it returned,
    which is the direction a verifier should err in.
    """
    fresh = {p for p in targets
             if _sig(p) is not None and _sig(p) != before_t.get(p)}
    for p, sg in _image_sigs(outdir).items():
        if sg is not None and sg != before_d.get(p):
            fresh.add(p)
    return sorted(fresh)


def _short(q: Path, base: Path) -> str:
    try:
        return str(q.relative_to(base))
    except ValueError:
        return str(q)


def _refresh_views(results_root: Path) -> int:
    """Rebuild supplementary_figures/ and staged_figures/ beside results_root.

    The same two steps a full run ends with (figure_runner's supp_link and
    supp_stage). The local tree's view is symlinks; a package's is real files.
    Returns the number of steps that failed.
    """
    import subprocess
    results_root = results_root.resolve()
    root = results_root.parent
    manifest = ROOT / "tools" / "additional_figures_manifest.txt"
    link = [sys.executable, str(ROOT / "tools" / "link_supplementary_figures.py"),
            "--results", str(results_root),
            "--out", str(root / "supplementary_figures"), "--quiet"]
    if root != ROOT.resolve():
        link.append("--copy")
    stage = [sys.executable, str(ROOT / "tools" / "stage_figures_for_docx.py"),
             "--manifest", str(manifest), "--results", str(results_root),
             "--out", str(root / "staged_figures"), "--clean", "--strict"]
    failed = 0
    for name, cmd in (("supplementary_figures/", link), ("staged_figures/", stage)):
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  {name:<23}{'rebuilt' if r.returncode == 0 else 'FAILED'}"
              f"  ({root / name})")
        if r.returncode:
            print((r.stdout + r.stderr).strip()[-2000:])
            failed += 1
    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slot", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=None, help="write here instead of in place")
    ap.add_argument("--no-views", action="store_true",
                    help="redraw only; do not rebuild supplementary_figures/ "
                         "and staged_figures/")
    ap.add_argument("--in-process", action="store_true",
                    help="draw in this process rather than one per figure")
    ap.add_argument("--results-root", default=None,
                    help="read the data objects from this results/ directory "
                         "(e.g. a finished run's package) instead of ./results")
    a = ap.parse_args()
    results_root = Path(a.results_root).expanduser() if a.results_root else ROOT / "results"
    fs = _fs()
    man = {s: (r, st) for s, r, st in fs.manifest()}

    if a.list:
        have, missing = fs.data_coverage(quiet=True)
        have = {s for s, _ in have}
        print(f"  {'slot':<6}{'data object':<14}{'redraws from it'}")
        for slot, _r, stem in fs.manifest():
            print(f"  {slot:<6}{'yes' if slot in have else 'no':<14}"
                  f"{'yes' if slot in DRAW else 'not yet -- needs split'}")
        print(f"\n  {len(have)}/{len(man)} publish data;  "
              f"{len(DRAW)}/{len(man)} redraw from it")
        return 0

    slots = sorted(DRAW) if a.all else [s if s.upper().startswith("S") else f"S{s}"
                                         for s in a.slot]
    if not slots:
        ap.error("name a slot (e.g. S4), or pass --all / --list")
    views = not (a.out or a.no_views or a.in_process)

    # The violin figures (S6, S7, S11-S13, S24) fall back to matplotlib
    # without seaborn -- different violins, and S12/S13/S24 lose theirs --
    # so an interpreter without it draws figures the pipeline never would.
    # Run on the pipeline's interpreter (tools/package_for_transfer.sh):
    #   ~/anaconda3/envs/plotting_env/bin/python tools/replot_from_data.py ...
    try:
        import seaborn  # noqa: F401
    except ImportError:
        print(f"  seaborn is not installed in {sys.executable}; redraws would "
              "not match the pipeline. Use the pipeline's interpreter "
              "(plotting_env).")
        return 2

    # Each figure gets its own process. Matplotlib carries style state between
    # figures drawn in one process, which shifts the tight bounding box by a
    # few pixels and so the saved canvas -- the same figure drawn alone comes
    # out identical to the pipeline's. --in-process skips the isolation.
    if len(slots) > 1 and not a.in_process:
        import subprocess
        ok = bad = 0
        for slot in slots:
            cmd = [sys.executable, __file__, slot, "--in-process", "--no-views"]
            if a.out:
                cmd += ["--out", a.out]
            if a.results_root:
                cmd += ["--results-root", a.results_root]
            r = subprocess.run(cmd, capture_output=True, text=True)
            tail = [ln for ln in r.stdout.splitlines()
                    if "redrawn in" in ln or "FAILED" in ln or "no data" in ln]
            print(tail[0] if tail else f"  {slot}: no output (exit {r.returncode})")
            if r.returncode == 0 and tail and "redrawn in" in tail[0]:
                ok += 1
            else:
                bad += 1
        print(f"\n  {ok} redrawn, {bad} failed")
        if views and ok:
            bad += _refresh_views(results_root)
        elif a.out and ok:
            print("  previews only: supplementary_figures/ and staged_figures/ "
                  "were not touched")
        return 1 if bad else 0

    ok = bad = 0
    for slot in slots:
        if slot not in DRAW:
            print(f"  {slot}: no replot wired yet"); bad += 1; continue
        ref, stem = man[slot]
        outdir = str((results_root / ref / stem).parent)
        try:
            obj = figure_source.load(outdir, slot)
            # Writing in place, each figure already has its own directory and
            # its canonical name; only a shared --out needs disambiguating.
            obj["_prefix"] = f"{slot}_" if a.out else ""
        except Exception as exc:
            print(f"  {slot}: no data object ({exc})"); bad += 1; continue
        apply_oi_style()
        # Fingerprint everything the drawer could write before it runs, so
        # "redrawn" means a file appeared or changed -- not merely that the
        # call returned.
        writedir = Path(a.out or outdir)
        targets = [] if a.out else _slot_targets(results_root, ref, stem)
        before_t = {p: _sig(p) for p in targets}
        before_d = _image_sigs(writedir)
        t0 = time.time()
        try:
            p = DRAW[slot](obj, a.out or outdir)
        except Exception as exc:
            print(f"  {slot}: FAILED {exc}"); bad += 1; continue
        wrote = _written_since(targets, before_t, writedir, before_d)
        if not wrote:
            want = (" / ".join(str(t.relative_to(results_root)) for t in targets)
                    if targets else f"any image in {writedir}")
            print(f"  {slot}: FAILED the drawer returned {p!r} but wrote no "
                  f"file ({want} unchanged) — its data object is probably "
                  "missing a panel")
            bad += 1
            continue
        shown = ", ".join(_short(q, writedir) for q in wrote[:3])
        note = ("" if p and Path(p) in wrote else
                f"  [returned {p!r}, which is not what it wrote]")
        print(f"  {slot}: redrawn in {time.time()-t0:.1f}s -> {shown}{note}")
        ok += 1
    print(f"\n  {ok} redrawn, {bad} failed")
    if views and ok:
        bad += _refresh_views(results_root)
    elif a.out and ok:
        print("  previews only: supplementary_figures/ and staged_figures/ "
              "were not touched")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
