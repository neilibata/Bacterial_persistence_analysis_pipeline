"""
figure5.py  —  Standalone Figure 5 generator  (v5)

Run pipeline prerequisite first:
    python -m persistence --ref UHGG --only health_richness

Then:
    python figure5.py
"""

from __future__ import annotations
import argparse, colorsys, glob, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, energy_distance

_trapz = getattr(np, "trapezoid", np.trapz)

# ── Palette ───────────────────────────────────────────────────────────────────
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermillion="#D55E00", pink="#CC79A7", black="#000000")

GROUP_COLORS = {"NI": OI["orange"], "IndH": OI["green"], "IndU": OI["blue"]}
LIST_COLORS  = {"VanishH+": OI["orange"], "BlossumH+": OI["sky"],
                "VANISH": OI["vermillion"], "VANISH+": OI["yellow"],
                "BloSSUM": OI["blue"],     "BloSSUM+": OI["green"]}

# Contrasts the panel never shows, for structural reasons rather than
# results: "Obese (healthy only)" is empty by construction now that obesity
# sets the Unhealthy label, and the other two are not in the comparison set.
#
# Hypertension used to be here and is not any more. Under the study-stratified
# estimator it rests on 3 studies that agree unanimously (p = 3e-04), while
# ASCVD and Metabolic Syndrome rest on one study each and cannot be tested at
# all -- so the hardcoded list was hiding the best-supported of the three and
# showing the two with no cross-study evidence. Which of those get dropped is
# now decided by MIN_STRATA_TO_PLOT below, from the data, so it stays correct
# if the cohort changes.
# Which contrasts appear is decided below by MIN_STRATA_TO_PLOT, from the
# data; this set is the explicit override for the few the rule cannot judge.
# It is no longer empty: it now mirrors config.HEALTH_PLOT_SKIP, which holds
# the pooled "Unhealthy" contrast. The rule admits that contrast on 41
# studies, but its cases ARE the cases of the four focal ones, so it is a
# replicate rather than a fifth condition -- see config for the full reason.
# Mirroring rather than redefining keeps panels D-H, S25 and S26-S29 showing
# the same columns; they disagreed once before, when the heatmap kept a column
# the other panels had dropped.
SKIP: set = set()      # replaced from config just below, when importable

#: A contrast needs at least this many qualifying studies to be plotted.
#:
#: One study gives no between-study variance at all, hence no interval and no
#: p-value, and a bar with neither reads as a null result rather than an
#: untested one. Two is barely better: the sandwich SE is then estimated from
#: a single degree of freedom, and it collapses toward zero when the two
#: strata happen to agree. Measured on this cohort, the ulcerative-colitis
#: flag gives p = 0.002 from two strata with 96% of the weight in one study,
#: and shifting the minor stratum by 2 SGBs moves that p sixfold. Three is
#: still thin (df = 2) but survives that perturbation: hypertension's
#: p = 3e-04 only reaches 3e-03 when its smallest study is moved by 4 SGBs.
#:
#: Nothing currently plotted has fewer than three strata, so this excludes no
#: present column -- it guards against a future cohort producing one.
# The repo root goes on the path BEFORE the first `persistence` import in
# this file: run by path, sys.path[0] is `figures/` and the package is not
# importable from it, so every except branch below ran instead, silently. See
# figure2.py for the full note.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from persistence.config import MIN_STRATA_TO_PLOT, HEALTH_PLOT_SKIP
    SKIP = set(HEALTH_PLOT_SKIP)
except Exception:                                  # standalone, package absent
    MIN_STRATA_TO_PLOT = 3
    SKIP = {"Unhealthy"}
# Type size, in points on this figure's own 24-inch canvas. The figure prints
# at one 3-column page width (7.24 in), a reduction of 3.3x, so what a reader
# sees is FS * 7.24 / 24. At FS = 17 that was 5.1 pt, and the annotations one
# point below it printed at 4.8 -- under Science's "not smaller than 5 points".
# FS = 23 puts the body at 6.9 pt and the smallest string at 6.4, which is the
# "about 7 points" the guide asks for, and matches what the supplementary
# figures get from persistence.plot_utils (5.2-9.5 pt at final size).
PAGE_W_IN = 7.24
FS      = 23
BAR_COL = OI["sky"]

# The type this canvas is authored at. Named, because the figure is also
# imported by the pipeline to draw Figure S27: applying it at import time only
# would set it for whichever call happened to trigger the import and leave
# every later call drawing at the importing module's scale.
RC = {
    "font.size":        FS,
    "axes.titlesize":   FS,
    "axes.labelsize":   FS,
    "xtick.labelsize":  FS,
    "ytick.labelsize":  FS,
    "legend.fontsize":  FS,
    "font.family":      "sans-serif",
}
matplotlib.rcParams.update(RC)


# build_panel_e_figure / _build_panel_e_figure were removed on 2026-10-07.
# They drew the density form of panel E for a second pair of lists, and
# nothing has called them since that form was retired; S26 is the background
# curve now. A dead builder for a superseded figure is how the superseded
# figure comes back.


# ══════════════════════════════════════════════════════════════════════════════
#  Micro-helpers
# ══════════════════════════════════════════════════════════════════════════════

def _star(p):
    if not np.isfinite(p): return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def _bgcol(h, desat=0.35, bright=1.20):
    r,g,b = [int(h[i:i+2],16)/255 for i in (1,3,5)]
    hh,s,v = colorsys.rgb_to_hsv(r,g,b)
    nr,ng,nb = colorsys.hsv_to_rgb(hh, s*desat, min(v*bright,1.0))
    return "#{:02x}{:02x}{:02x}".format(int(nr*255),int(ng*255),int(nb*255))

def _style(ax, box=True):
    """Uniform style. box=True → all 4 spines; box=False → only bottom+left."""
    ax.grid(False); ax.set_axisbelow(False)
    ax.tick_params(axis="both", labelsize=FS-1, length=3)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)
        sp.set_visible(True)
    if not box:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

def _iw_kde(vals, weights, x_eval):
    vals    = np.asarray(vals,    dtype=float)
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    mask    = np.isfinite(vals) & (weights > 0)
    vals, weights = vals[mask], weights[mask]
    if len(vals) < 2:
        return np.zeros_like(x_eval, dtype=float)
    try:
        return np.maximum(
            gaussian_kde(vals, weights=weights/weights.sum(),
                         bw_method="scott")(x_eval), 0.0)
    except Exception:
        return np.maximum(gaussian_kde(vals, bw_method="scott")(x_eval), 0.0)

def _display_group(g: str) -> str:
    return {"NI": "Non-industrial",
            "IndH": "Industrial\nhealthy",
            "IndU": "Industrial\nunhealthy"}.get(g, g)

# The published name for a condition. persistence.config owns the mapping
# (PLOT_TEXT_REPLACEMENTS); this script is also run standalone, so fall back to
# a local copy rather than failing to draw. Figure 5 needs its own call because
# it saves with plt.savefig and never reaches plot_utils.apply_display_text.
try:
    from persistence.config import display_text as _display_text
except Exception as _exc:                          # package genuinely absent
    # Not a second rename table. This copy knew only Fatty Liver -> MASLD, so
    # a fallback render kept "Obese" and the title-case lifestyle names that
    # config renames -- disagreeing with every other figure rather than
    # failing. Say so instead.
    print(f"  [warn] persistence.config unavailable ({_exc}); figure text "
          f"will carry internal keys, not published labels", file=sys.stderr)

    def _display_text(s):
        return str(s)


#: The smallest face on this figure. Authored at 24 in and printed at 7.24,
#: everything here is reduced to 31%, so 14 pt printed at 4.4 pt -- under the
#: 5 pt floor. 16 pt prints at 5.0 pt, which is what the abbreviations below
#: and the two-line legend entries make room for.
FS_SMALL = FS - 1

#: Abbreviations this figure still puts in front of a reader -- panel D names
#: each condition above its column. Every one has to be expanded in the
#: caption, so they are listed here rather than left for someone to notice.
CAPTION_ABBREVIATIONS = {
    "MASLD": "Metabolic Dysfunction-Associated Steatotic Liver Disease",
    "ASCVD": "atherosclerotic cardiovascular disease",
    "IBD": "inflammatory bowel disease",
    "CRC": "colorectal cancer",
}


def _cohort_note(row) -> str:
    """"(k studies, cases / controls)" for the analysed cohort of one contrast.

    The counts are the study-stratified ones (n_strata, n_g2_strat, n_g1_strat):
    one sample per host, studies with >=5 in each arm -- the cohort the bar was
    computed on and the one Tables S3 and S10 report. n_g1/n_g2 are every
    Industrial sample in each arm before study filtering and overstate it (4,517
    Unhealthy cases against 4,094 analysed). Falls back to n_g2 only for a
    results file that predates the stratified columns.
    """
    def _int(key):
        v = pd.to_numeric(row.get(key, np.nan), errors="coerce")
        return int(v) if np.isfinite(v) else None
    k, cases, ctrls = _int("n_strata"), _int("n_g2_strat"), _int("n_g1_strat")
    if None not in (k, cases, ctrls):
        return f"({k} stud{'y' if k == 1 else 'ies'}, {cases:,} / {ctrls:,})"
    n = _int("n_g2")
    return f"(n={n:,})" if n else ""


def _condition_label(cond: str, note: str = "") -> str:
    wrapped = {
        "Unhealthy": "Unhealthy",
        "Obese": "Obesity",
        "Glucose Intolerance": "Glucose Intolerance",
        "Fatty Liver": "MASLD",
        "ASCVD": "ASCVD",
        "IBD": "IBD",
        "CRC / Adenoma": "CRC / Adenoma",
        # Demographic strata (supplementary demographic figure), which reuse
        # draw_D and so come through this same label path.
        # One line each: _condition_label appends the cohort note and the result is
        # anchored va="bottom", so it grows upward -- a three-line label
        # climbs off the reserved band and into the bars.
        "Age <18": "Age <18",
        "Age 30-44": "Age 30-44",
        "Age 45-59": "Age 45-59",
        "Age 60+": "Age 60+",
        "Female": "Female",
        "Obese vs rest": "Obese vs rest",
        "Overweight or obese": "Overwt./obese",
    }.get(cond, _display_text(cond).replace(" / ", " /\n"))
    return f"{wrapped}\n{note}" if note else wrapped

def _native_violin(ax, vdf, order, palette):
    vals = [vdf.loc[vdf["Group"] == g, "richness"].dropna().values
            for g in order]
    vals = [v if len(v) else np.array([np.nan]) for v in vals]
    vp = ax.violinplot(vals, positions=np.arange(len(order)),
                       widths=0.78, showmeans=False, showmedians=False,
                       showextrema=False)
    for body, g in zip(vp["bodies"], order):
        body.set_facecolor(palette.get(g, OI["pink"]))
        body.set_edgecolor("#333333")
        body.set_alpha(0.55)
        body.set_linewidth(0.9)
    for i, arr in enumerate(vals):
        arr = arr[np.isfinite(arr)]
        if len(arr) < 2:
            continue
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        lo, hi = np.percentile(arr, [5, 95])
        ax.vlines(i, lo, hi, color="#333333", lw=1.1, zorder=4)
        ax.vlines(i, q1, q3, color="#333333", lw=4.0, zorder=5)
        ax.scatter([i], [med], s=20, color="white",
                   edgecolor="#333333", linewidth=0.7, zorder=6)


# ══════════════════════════════════════════════════════════════════════════════
#  File discovery
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_spec_tag(results_dir: str, spec_arg: str) -> str:
    defaults = {"breadth_pct": "10pct", "flat_bp": "300kb"}
    for tag in [spec_arg, defaults.get(spec_arg, spec_arg)]:
        if os.path.exists(os.path.join(results_dir,
                                       f"health_richness_gmw_{tag}.csv")):
            return tag
    found = [f for f in glob.glob(
        os.path.join(results_dir, "health_richness_gmw_*.csv"))
             if not os.path.basename(f)
             .replace("health_richness_gmw_", "").replace(".csv", "")
             .startswith(("UHGG","lev","chimp"))]
    if found:
        tag = (os.path.basename(found[0])
               .replace("health_richness_gmw_","").replace(".csv",""))
        print(f"  [INFO] Auto-detected spec_tag: {tag}")
        return tag
    return spec_arg


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_per_sample(results_dir: str, ref: str, spec_tag: str):
    path = os.path.join(results_dir,
        f"health_richness_gmw_{ref}_summary_NI_health_{spec_tag}_perSample.csv")
    if not os.path.exists(path):
        print(f"  [WARN] Per-sample CSV not found: {path}")
        print(f"         Run: python -m persistence --ref {ref} "
              "--only health_richness")
        return None
    return pd.read_csv(path)

def load_panel_d(results_dir: str, spec_tag: str):
    path = os.path.join(results_dir, f"health_richness_gmw_{spec_tag}.csv")
    if not os.path.exists(path):
        print(f"  [WARN] Panel D not found: {path}"); return None
    df = pd.read_csv(path)
    df = df[~df["Condition"].isin(SKIP)]
    if "n_strata" in df.columns:
        thin = df[df["n_strata"] < MIN_STRATA_TO_PLOT]
        if len(thin):
            print(f"  [panel D] dropping contrast(s) with fewer than "
                  f"{MIN_STRATA_TO_PLOT} studies: "
                  + ", ".join(f"{r.Condition} ({int(r.n_strata)})"
                              for r in thin.itertuples()))
        df = df[df["n_strata"] >= MIN_STRATA_TO_PLOT]
    return df.reset_index(drop=True)

# Which column holds the case-control shift, and which the background weight.
# `rd_mh` is the study-stratified shift the corrected method tests; `prev_diff`
# is the pooled one the 29 Sep run published, and is what its pickle carries.
# Mirrors section3_health._shift_values, so a redraw of an old data object keeps
# drawing the figure it was published as.
SHIFT_COLUMNS = ("rd_mh", "prev_diff")
BG_WEIGHT_COLUMNS = ("bg_weight", "importance_weight")


def _shift_col(df):
    for c in SHIFT_COLUMNS:
        if c in getattr(df, "columns", ()):
            return c
    return None


def _common_shift(*frames):
    """Drop `rd_mh` unless every frame has it, so a concat cannot blank a side.

    Panel E concatenates the list table and the background table and then
    filters on the shift column. A list table written before the stratified
    shift was merged on has only `prev_diff`; the concatenation then gives it
    an all-NaN `rd_mh` and the filter drops every list row, so every cell of
    panel E prints "n/a". Mirrors section3_health._common_shift_frames.
    """
    frames = [f for f in frames if f is not None and len(f)]
    if len(frames) < 2:
        return
    for c in SHIFT_COLUMNS:
        if any(c in f.columns for f in frames) and \
                not all(c in f.columns for f in frames):
            print(f"  [WARN] {c} is missing from one of the tables being "
                  "compared, so the comparison falls back to the pooled "
                  "shift. Rerun section 3 to get the corrected one.")
            for f in frames:
                if c in f.columns:
                    f.drop(columns=[c], inplace=True)


def _shift_pp(df):
    """Case - control shift in percentage points (positive = enriched in disease)."""
    c = _shift_col(df)
    if c is None:
        return np.array([])
    return -pd.to_numeric(df[c], errors="coerce").values.astype(float) * 100.0


def _bg_w(df):
    for c in BG_WEIGHT_COLUMNS:
        if c in getattr(df, "columns", ()):
            return np.maximum(
                pd.to_numeric(df[c], errors="coerce").fillna(1.0)
                .values.astype(float), 0.0)
    return np.ones(len(df))


def load_curve_summary(results_dir: str, spec_tag: str):
    """{(condition, list): row} from health_background_curve_summary_<spec>.csv.

    The list-level statistic panel E prints: the mean distance from the
    prevalence background in SDs, with a bootstrap CI that refits the curve.
    Returns {} when absent, and the panel falls back to the energy-distance
    star it used to print.
    """
    import os as _os
    path = _os.path.join(results_dir,
                         f"health_background_curve_summary_{spec_tag}.csv")
    if not _os.path.exists(path):
        print(f"  [panel E] no background-curve summary at {path}; "
              "falling back to the energy-distance star")
        return {}
    df = pd.read_csv(path)
    need = {"Condition", "Taxon_list", "mean_z", "ci_lo", "ci_hi", "boot_p"}
    if not need <= set(df.columns):
        return {}
    out = {(str(r.Condition), str(r.Taxon_list)): r._asdict()
           for r in df.itertuples(index=False)}
    print(f"  [panel E] {len(out)} mean-z statistics from "
          f"{_os.path.basename(path)}")
    return out


def load_curve_panels(results_dir: str, spec_tag: str):
    """The four frames behind panel E's background-curve cells.

    Panel E used to be a density contrast and is now the same background-curve
    cell the supplementary twin draws, so it needs what that figure needs: the
    fitted curve, each list taxon's position on it, the list-level statistic,
    and the full per-condition shift frame the background points come from.

    Returns {"taxa", "grid", "summary", "all_taxa"} or {} when the curve
    tables are absent, in which case the caller keeps the older density cells.
    """
    import os as _os
    paths = {
        "taxa":    f"health_background_curve_taxa_{spec_tag}.csv",
        "grid":    f"health_background_curve_grid_{spec_tag}.csv",
        "summary": f"health_background_curve_summary_{spec_tag}.csv",
        "strat":   f"health_per_taxon_stratified_{spec_tag}.csv",
    }
    full = {k: _os.path.join(results_dir, v) for k, v in paths.items()}
    missing = [v for k, v in paths.items() if not _os.path.exists(full[k])]
    if missing:
        print("\n  " + "!" * 72)
        print("  [panel E] WARNING: background-curve tables are missing, so "
              "panel E will be")
        print("            drawn in the SUPERSEDED density form with "
              "energy-distance stars.")
        print("            This is not the figure the manuscript describes. "
              "Missing:")
        for _m in missing:
            print(f"              {_m}")
        print("            Run section 3 first, or pass --results-root at a "
              "completed run.")
        print("  " + "!" * 72 + "\n")
        return {}
    out = {k: pd.read_csv(full[k], low_memory=False)
           for k in ("taxa", "grid", "summary")}
    strat = pd.read_csv(full["strat"], low_memory=False)
    # The background cloud, carried as one tidy frame rather than a dict of
    # frames so it travels in the data object like every other panel.
    from persistence import health_background as _hbg
    rows = []
    for cond, g in strat.groupby("Condition"):
        f = _hbg.shift_frame(g).reset_index()
        f = f.rename(columns={f.columns[0]: "taxon"})
        f["Condition"] = cond
        rows.append(f[["taxon", "Condition", "x", "shift"]])
    out["all_taxa"] = pd.concat(rows, ignore_index=True)
    print(f"  [panel E] background curve: {len(out['taxa']):,} list taxa, "
          f"{len(out['all_taxa']):,} background rows")
    return out


def load_list_comparison(results_dir: str, ref: str):
    """Panels F-H: the six lists head to head, from the Table S12 source.

    These three panels were Fig. S31 and are drawn by the same two functions
    that still draw it, so the supplementary figure and the main-text block
    cannot drift apart. `{}` when the table is absent, which drops the block
    and its height rather than failing the figure.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from list_comparison_panels import panels_from_csv
    except Exception as e:                              # pragma: no cover
        print(f"  [panel F-H] list_comparison_panels unavailable: {e}")
        return {}
    path = os.path.join(results_dir,
                        f"table_s12_method_comparison_{ref}.csv")
    panels = panels_from_csv(path)
    if not panels:
        print(f"  [panel F-H] skipped: no usable {os.path.basename(path)}")
    return panels


def load_energy_p(results_dir: str, spec_tag: str, prefix: str = "table_s6"):
    """{(condition, taxon_list): energy_p} from the Table S6B/S10B writer.

    The single source for panel E's stars. Returns {} when the table is
    absent, in which case draw_E_cell falls back to its own bootstrap and the
    figure still draws -- just with p-values that need not match a table that
    does not exist yet.
    """
    import os as _os
    path = _os.path.join(results_dir, f"{prefix}b_energy_first_{spec_tag}.csv")
    if not _os.path.exists(path):
        print(f"  [panel E] no energy table at {path}; "
              "stars will be recomputed at draw time")
        return {}
    df = pd.read_csv(path)
    if not {"Condition", "List", "energy_p"} <= set(df.columns):
        return {}
    out = {}
    for r in df.itertuples():
        if pd.notna(r.energy_p):
            out[(str(r.Condition), str(r.List))] = float(r.energy_p)
    print(f"  [panel E] {len(out)} energy p-values from "
          f"{_os.path.basename(path)}")
    return out


def load_panel_e(results_dir: str, spec_tag: str):
    pt_p = os.path.join(results_dir, f"health_per_taxon_gmw_{spec_tag}.csv")
    bg_p = os.path.join(results_dir,
                        f"health_per_taxon_prevmatched_{spec_tag}.csv")
    pt = pd.read_csv(pt_p, low_memory=False) if os.path.exists(pt_p) else None
    bg = pd.read_csv(bg_p, low_memory=False) if os.path.exists(bg_p) else None
    if pt is None: print(f"  [WARN] Panel E list CSV not found: {pt_p}")
    if bg is None: print(f"  [WARN] Panel E bg   CSV not found: {bg_p}")
    return pt, bg


# ══════════════════════════════════════════════════════════════════════════════
#  Panel A — depth KDE  (no twinx; GMW target plotted on same axis)
# ══════════════════════════════════════════════════════════════════════════════

def draw_A(ax, ps_df):
    _style(ax)
    ax.set_title("Sequencing depth distributions",
                 fontsize=FS+2, fontweight="bold", pad=5)

    if ps_df is None or "log10_depth" not in ps_df.columns:
        ax.text(0.5, 0.5,
                "Run pipeline with\n--only health_richness",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FS-2, color="#888", style="italic",
                multialignment="center")
        ax.set_ylabel("Normalised density", fontsize=FS)
        return

    NICE_GB  = [0.03, 0.1, 0.3, 1, 3, 10, 30]
    TICK_LOG = [np.log10(v*1e9) for v in NICE_GB]
    ls_map   = {"NI": "-", "IndH": "--", "IndU": ":"}
    # Two lines rather than one: the qualifier moves under "Industrial", which
    # halves the width of the widest entry and pays for a larger face. At the
    # printed column width the old single-line 14 pt set was 4.4 pt, under the
    # 5 pt floor.
    # One definition for the three group names, shared with panels C and D.
    # This was a second, title-case copy, so the same three groups were
    # labelled two ways on one figure: "Non-Industrial / Industrial Healthy"
    # in panel A's key and "Non-industrial / Industrial healthy" on the axes
    # below it.
    legend_labels = {g: _display_group(g) for g in ("NI", "IndH", "IndU")}
    order    = [g for g in ["NI","IndH","IndU"] if g in ps_df["Group"].values]

    all_ld   = ps_df["log10_depth"].dropna().values
    lo_x, hi_x = all_ld.min()-0.25, all_ld.max()+0.25
    dg = np.linspace(lo_x, hi_x, 400)

    # All KDEs normalised to max=1 and plotted on the same axis as the GMW target
    geom = np.ones(len(dg))
    for grp in order:
        ld  = ps_df.loc[ps_df["Group"]==grp, "log10_depth"].dropna().values
        col = GROUP_COLORS.get(grp, OI["pink"])
        if len(ld) < 5: continue
        k  = gaussian_kde(ld, bw_method="silverman")
        yv = k(dg); yv /= max(yv.max(), 1e-300)
        ax.plot(dg, yv, color=col, ls=ls_map.get(grp,"-"), lw=2.2,
                label=legend_labels.get(grp, grp))
        ax.fill_between(dg, yv, alpha=0.10, color=col)
        geom *= np.maximum(k(dg), 1e-300)

    geom = geom**(1/max(len(order),1))
    gn   = geom / max(geom.max(), 1e-300)
    ax.plot(dg, gn, color="black", lw=2.2, label="GMW target")
    ax.fill_between(dg, gn, alpha=0.08, color="black")

    ticks = [(p, str(v)) for p, v in zip(TICK_LOG, NICE_GB) if lo_x<=p<=hi_x]
    if len(ticks) < 3:
        ticks = list(zip(TICK_LOG, [str(v) for v in NICE_GB]))[:5]
    ax.set_xticks([p for p,_ in ticks])
    ax.set_xticklabels([l for _,l in ticks], fontsize=FS-1)
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1.0"], fontsize=FS-1)
    ax.set_xlabel("Sequencing depth (Gbases)", fontsize=FS)
    ax.set_ylabel("Normalised density",    fontsize=FS)
    ax.legend(fontsize=FS_SMALL, loc="upper left", framealpha=0.7,
              edgecolor="none", handlelength=2.0, labelspacing=0.35)


# ══════════════════════════════════════════════════════════════════════════════
#  Panels B + C — richness violins
# ══════════════════════════════════════════════════════════════════════════════

def draw_violin(ax, ps_df, corrected: bool, title: str, rng):
    _style(ax)
    ax.set_title(title, fontsize=FS+2, fontweight="bold", pad=5)
    order   = [g for g in ["NI","IndH","IndU"] if g in ps_df["Group"].values]
    palette = {g: GROUP_COLORS.get(g, OI["pink"]) for g in order}

    if corrected:
        rows = []
        for g in order:
            sub = ps_df[ps_df["Group"]==g]
            r   = sub["richness"].values.astype(float)
            w   = np.maximum(sub["weight"].values.astype(float), 0.0)
            w_s = w / max(w.sum(), 1e-300)
            idx = rng.choice(len(r), size=len(r), replace=True, p=w_s)
            for v in r[idx]: rows.append({"Group": g, "richness": v})
        vdf = pd.DataFrame(rows)
    else:
        vdf = ps_df[["Group","richness"]].copy()

    _native_violin(ax, vdf, order, palette)

    for i, g in enumerate(order):
        sub = ps_df[ps_df["Group"]==g]
        r   = sub["richness"].values.astype(float)
        w_raw = np.maximum(sub["weight"].values.astype(float), 0.0)
        w_sum = max(w_raw.sum(), 1e-300)
        w = w_raw / w_sum
        mu  = float(np.dot(w, r)) if corrected else float(r.mean())
        n   = len(r)

        if corrected:
            wsem = (float(np.sqrt((w_raw * (r - mu)**2).sum()) / w_sum)
                    if n > 1 else 0.0)
            ax.errorbar(i, mu, yerr=[[wsem],[wsem]],
                        fmt="D", color="white", ecolor="black",
                        elinewidth=2.0, capsize=5, markersize=8, zorder=6)
            ax.text(i+0.06, mu, f"{mu:.0f}±{wsem:.0f}",
                    va="center", fontsize=FS, fontweight="bold")
        else:
            sem = r.std(ddof=1)/np.sqrt(n) if n > 1 else 0.0
            ax.errorbar(i, mu, yerr=[[sem],[sem]],
                        fmt="D", color="white", ecolor="black",
                        elinewidth=2.0, capsize=5, markersize=8, zorder=6)
            ax.text(i+0.06, mu, f"{mu:.0f}±{sem:.0f}",
                    va="center", fontsize=FS, fontweight="bold")

    ax.set_xticks(range(len(order)))
    if corrected:
        xlabels = []
        for g in order:
            w_g = np.maximum(ps_df.loc[ps_df["Group"]==g,
                                       "weight"].values.astype(float), 0.0)
            ess = float(w_g.sum()**2 / max((w_g**2).sum(), 1e-300))
            xlabels.append(f"{_display_group(g)}\n(ESS={ess:.0f})")
        ax.set_xticklabels(xlabels, fontsize=FS)
    else:
        ax.set_xticklabels(
            [f"{_display_group(g)}\n(n={len(ps_df[ps_df['Group']==g]):,})"
             for g in order],
            fontsize=FS)
    ax.set_ylabel("# SGBs per sample", fontsize=FS)
    ax.set_xlabel("")
    ax.set_ylim(bottom=0)


# ══════════════════════════════════════════════════════════════════════════════
#  Panel D — health richness bar chart
# ══════════════════════════════════════════════════════════════════════════════

def draw_D(axes_top: list, axes_bot: list, df, health_bound):
    """Panel D. `health_bound` marks where a dashed divider is drawn.

    An int keeps the original meaning (one divider, after that many columns).
    An iterable of ints draws one after each, which the demographic figure
    needs to separate age from sex from BMI.

    `axes_bot=None` draws the single-row form: the case-minus-control row is
    dropped and its significance stars move onto the paired bars above, over a
    bracket that says which two bars the test compared. The row also inherits
    the band at the foot of the panel that carries each column's condition
    name, since that band belonged to the row being removed. The two-row form
    is unchanged and is what make_demographic_associations still draws.
    """
    single = axes_bot is None
    bounds = ({int(health_bound)} if isinstance(health_bound, (int, float))
              else {int(b) for b in health_bound})
    bw = 0.32

    # Pre-compute y extents across all conditions for shared axes
    all_top_hi, all_bot_lo, all_bot_hi = [], [], []
    for _, row in df.iterrows():
        all_top_hi.append(max(float(row["mu_g1"])+float(row["se_g1"]),
                              float(row["mu_g2"])+float(row["se_g2"])))
        diff  = -float(row["diff"])
        ci_lo = float(-row.get("ci_hi", row["diff"]))
        ci_hi = float(-row.get("ci_lo", row["diff"]))
        all_bot_lo.append(min(diff, ci_lo if np.isfinite(ci_lo) else diff))
        all_bot_hi.append(max(diff, ci_hi if np.isfinite(ci_hi) else diff))

    # Headroom above the bars. The single-row form needs less of it: the two
    # -row form left room for a star floating over a tall error bar, while
    # here the star sits on a bracket at a known height.
    mu_ymax   = max(all_top_hi) * (1.28 if axes_bot is None else 1.30)
    bot_span  = max(max(all_bot_hi) - min(all_bot_lo), 5.0)
    bot_ylo   = min(-150.0, min(all_bot_lo) - bot_span * 0.55)
    bot_yhi   = max(all_bot_hi) + bot_span * 0.60   # room for stars

    # In single-row form the condition labels have nowhere below to go, so the
    # top panel opens a band under zero for them -- the same proportion of the
    # panel the removed row gave them, so the bars keep their printed size.
    LABEL_BAND = 0.34
    top_ylo = -mu_ymax * LABEL_BAND if single else 0.0

    for ci, (_, row) in enumerate(df.iterrows()):
        at = axes_top[ci]; ab = None if single else axes_bot[ci]
        _style(at)
        if ab is not None:
            _style(ab)

        # Shared y range
        at.set_ylim(top_ylo, mu_ymax)
        if single:
            # Ticks for the data, not for the label band; zero is drawn as the
            # bars' own baseline.
            at.set_yticks([t for t in at.get_yticks() if 0 <= t <= mu_ymax])
            at.axhline(0, color="black", lw=0.9)
        else:
            ab.set_ylim(bot_ylo, bot_yhi)

        # The bottom of this panel is reserved for the two-line label naming
        # each column's condition, so ticks are chosen from the region above
        # it: otherwise the lowest one prints its label and tick mark behind
        # that text. The axis still reads from the ticks above, and zero is
        # drawn as its own line.
        # Keep the round values matplotlib chose for the whole range and drop
        # only those that fall in the band -- re-locating ticks over the
        # shortened range gives 40/-40/-80 where 50/0/-50 reads better.
        if not single:
            _label_band = bot_ylo + (bot_yhi - bot_ylo) * 0.30
            ab.set_yticks([t for t in ab.get_yticks()
                           if _label_band <= t <= bot_yhi])

        # ── Top: paired bars ─────────────────────────────────────────
        at.bar(-bw/2, row["mu_g1"], bw*0.88, color=BAR_COL, alpha=0.33,
               edgecolor=BAR_COL, lw=0.8, zorder=3)
        at.errorbar(-bw/2, row["mu_g1"], yerr=row["se_g1"],
                    fmt="none", color=BAR_COL, capsize=3, lw=1.1, zorder=4)
        at.bar(+bw/2, row["mu_g2"], bw*0.88, color=BAR_COL, alpha=0.92,
               edgecolor="white", lw=0.4, zorder=3)
        at.errorbar(+bw/2, row["mu_g2"], yerr=row["se_g2"],
                    fmt="none", color=BAR_COL, capsize=3, lw=1.1, zorder=4)

        at.set_xlim(-0.65, 0.65); at.set_xticks([])
        at.tick_params(bottom=False, labelbottom=False)
        if ci > 0:
            at.tick_params(labelleft=False)
            at.spines["left"].set_color("#dddddd")
        else:
            at.set_ylabel("GMW richness\n(weighted mean taxa)", fontsize=FS)

        # ── Bottom: case − control diff ──────────────────────────────
        diff  = -float(row["diff"])
        ci_lo = float(-row.get("ci_hi", row["diff"]))
        ci_hi = float(-row.get("ci_lo", row["diff"]))
        pval  = float(row["pval"])
        star  = str(row.get("sig","")).strip()

        drawn = str(star) if star not in ("", "ns", "untested", "nan") else ""

        if single:
            # The star belongs to the difference between the two bars, so it
            # is drawn over a bracket that spans them rather than floating
            # above one of them. Height is taken from the taller bar's error
            # bar, so a column with a long interval does not push its star
            # through the bracket.
            if drawn:
                _tip = max(float(row["mu_g1"]) + float(row["se_g1"]),
                           float(row["mu_g2"]) + float(row["se_g2"]))
                y_br = _tip + mu_ymax * 0.055
                _drop = mu_ymax * 0.022
                at.plot([-bw/2, -bw/2, bw/2, bw/2],
                        [y_br - _drop, y_br, y_br, y_br - _drop],
                        color="black", lw=0.9, solid_capstyle="butt",
                        clip_on=False, zorder=6)
                at.text(0, y_br + mu_ymax * 0.012, drawn, ha="center",
                        va="bottom", fontsize=FS+1, fontweight="bold",
                        color="black", zorder=7)
        else:
            sig_alpha = 0.85 if np.isfinite(pval) and pval < 0.05 else 0.28
            ab.bar(0, diff, 0.68, color=BAR_COL, alpha=sig_alpha,
                   edgecolor=BAR_COL, lw=0.8, zorder=3)

            if np.isfinite(ci_lo) and np.isfinite(ci_hi):
                ab.errorbar(0, diff,
                            yerr=[[diff-ci_lo],[ci_hi-diff]],
                            fmt="none", color="black",
                            capsize=4, lw=1.4, capthick=1.4, zorder=5)

            ab.axhline(0, color="black", lw=0.9)
            ab.set_xlim(-0.55, 0.55); ab.set_xticks([])

        # Significance star — above ci_hi or below ci_lo, not overlapping bars
        # "untested" is a real value of `sig` now: a contrast with a single
        # qualifying study has no between-study variance to test, so it gets
        # no p-value. Without it in this set the word itself was drawn on the
        # panel where a star would go.
        #
        # Black, not the bar colour: in OI sky blue on a white panel the stars
        # were hard to pick out at print size, and nothing else in the panel
        # is black except the axis, so they cannot be confused with the data.
        if not single and drawn:
            if diff >= 0:
                y_s = (ci_hi if np.isfinite(ci_hi) else diff) + bot_span*0.08
                ab.text(0, y_s, drawn, ha="center", va="bottom",
                        fontsize=FS+1, fontweight="bold", color="black", zorder=7)
            else:
                y_s = (ci_lo if np.isfinite(ci_lo) else diff) - bot_span*0.08
                ab.text(0, y_s, drawn, ha="center", va="top",
                        fontsize=FS+1, fontweight="bold", color="black", zorder=7)

        if not single:
            if ci > 0:
                ab.tick_params(labelleft=False)
                ab.spines["left"].set_color("#dddddd")
            else:
                ab.set_ylabel("Richness diff.\n(Case − control)", fontsize=FS)

        # The column's name sits in the band at the foot of whichever row is
        # the last one.
        _lab = at if single else ab
        _lo, _hi = (top_ylo, mu_ymax) if single else (bot_ylo, bot_yhi)
        label_y = _lo + (_hi - _lo) * 0.045
        # The condition name carries the column; the cohort note under it is
        # supporting detail and is set smaller, which is also what keeps
        # "(41 studies, 4,094 / 5,144)" inside a column this wide.
        _lab.text(0, label_y, _cohort_note(row),
                  ha="center", va="bottom", fontsize=FS-5,
                  linespacing=0.95, clip_on=True)
        _lab.annotate(_condition_label(row["Condition"]),
                      xy=(0, label_y), xycoords=_lab.transData,
                      xytext=(0, FS - 4), textcoords="offset points",
                      ha="center", va="bottom", fontsize=FS+1,
                      fontweight="bold", linespacing=0.90, clip_on=True)

        # Dashed right spine after health conditions
        if (ci + 1) in bounds:
            for ax_ in ((at,) if single else (at, ab)):
                ax_.spines["right"].set_linewidth(1.1)
                ax_.spines["right"].set_linestyle((0,(3,2)))
                ax_.spines["right"].set_color("#aaaaaa")


# ══════════════════════════════════════════════════════════════════════════════
#  Panel E — prevalence-matched distributions
# ══════════════════════════════════════════════════════════════════════════════

def draw_E_cell(ax, sl, sb_vals, sb_w, col, bgcol, n_boot, rng, n_sample=100,
                p_energy=None, curve_row=None):
    """One panel-E cell. The drawing lives in
    `persistence.plot_utils.draw_prevalence_cell`, which Figure S27 also
    uses, so the two figures cannot drift apart in how they show the same
    comparison.

    `curve_row`, when given, is this list's row of the background-curve summary
    and the panel prints its mean z and bootstrap CI. Otherwise the panel falls
    back to the energy-distance star, where `p_energy` is the p from the Table
    S6B/S10B writer.

    Recomputing that p here was the old default and it made the panel disagree
    with its own table: this ran 2,000 bootstrap replicates at draw time while
    _compute_energy_stats wrote the table at 10,000 permutations from a
    different random stream, so p-values differed by two- to three-fold near
    the star boundaries. Measured across the 28 demographic cells, 5 carried a
    different number of stars in the figure than in the table.

    The direction tag stays local, from the median shift, because the two
    median lines it describes are drawn right here; the table's
    direction_vs_bg is computed from a mean shift and could disagree with what
    the reader can see.
    """
    from persistence.plot_utils import draw_prevalence_cell, weighted_median

    fallback = None
    if curve_row is None or not np.isfinite(curve_row.get("mean_z", np.nan)):
        if p_energy is not None and np.isfinite(p_energy):
            p_e = float(p_energy)
        else:
            _wp = sb_w / max(sb_w.sum(), 1e-300)
            _ns = min(n_sample, len(sl))
            _n_ref = max(_ns * 20, 2000)
            _sb_ref = sb_vals[rng.choice(len(sb_vals), size=_n_ref,
                                         replace=True, p=_wp)]
            obs_e = energy_distance(sl, _sb_ref)
            null_d = np.array([
                energy_distance(sb_vals[rng.choice(len(sb_vals), size=_ns,
                                                   replace=True, p=_wp)],
                                _sb_ref)
                for _ in range(n_boot)])
            p_e = max(float(np.mean(null_d >= obs_e)), 1.0 / (n_boot + 1))
        s_e = _star(p_e)
        med_shift = float(np.median(sl)) - weighted_median(sb_vals, sb_w)
        direction = (" (enr)" if med_shift > 0.20
                     else " (dep)" if med_shift < -0.20 else "")
        fallback = (f"E: {s_e}{direction}", s_e not in ("", "ns"))

    draw_prevalence_cell(ax, sl, sb_vals, sb_w, col, bgcol, FS,
                         curve_row=curve_row, fallback_label=fallback)


def _place_legend(fig, ax_col1, handles, ylo_frac=0.15, yhi_frac=0.65,
                  fontsize=None, ncol=1, title=None, title_color=None):
    """
    Create a small floating axes centered on the right edge of ax_col1,
    spanning [ylo_frac, yhi_frac] of that axes' height in figure coordinates.
    This avoids the clipping issues of bbox_to_anchor on data axes.
    """
    if fontsize is None: fontsize = FS-2

    # Get axes position in figure fraction directly from the GridSpec position
    # (avoids needing a renderer draw call)
    pos = ax_col1.get_position()   # Bbox in figure fraction [0,1]
    x_center_fig = pos.x1          # right edge of col1

    y_lo_fig = pos.y0 + ylo_frac * pos.height
    y_hi_fig = pos.y0 + yhi_frac * pos.height
    h_fig    = y_hi_fig - y_lo_fig
    w_fig    = 0.065               # width of the floating legend axes

    # Create a tiny off-screen helper to measure the legend
    leg_ax = fig.add_axes(
        [x_center_fig - w_fig/2, y_lo_fig, w_fig, h_fig],
        zorder=10)
    leg_ax.axis("off")

    leg = leg_ax.legend(
        handles=handles,
        fontsize=fontsize,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        bbox_transform=leg_ax.transAxes,
        borderaxespad=0.4,
        framealpha=0.92,
        edgecolor="#cccccc",
        ncol=ncol,
        labelspacing=0.45,
        handlelength=1.5,
        title=title,
        title_fontsize=fontsize+1 if title else None,
    )
    if title and title_color and leg.get_title():
        leg.get_title().set_color(title_color)
        leg.get_title().set_fontweight("bold")
    return leg


# ══════════════════════════════════════════════════════════════════════════════
#  Main assembly
# ══════════════════════════════════════════════════════════════════════════════

def _display_order(conds):
    """The shared rule, from plot_utils -- see display_order there. A copy
    lived here and in section3_health, which is two implementations of one
    convention."""
    from persistence.plot_utils import display_order
    return display_order(conds)




def build_figure5(results_dir: str, ref: str, spec_tag: str,
                  out_path: str, n_boot_E: int = 2000, tables=None):

    rng = np.random.default_rng(42)

    # ── Load data ─────────────────────────────────────────────────────
    # Four frames determine everything this figure draws. Passing them in
    # rebuilds the figure from its published data object instead of from the
    # results tree, which is what makes the pickle a complete record rather
    # than a summary of one.
    # A redraw from the data object re-emits nothing: the object it was built
    # from is the record, and rewriting it on every redraw would let a later
    # redraw quietly become the record instead.
    _emit_fresh = False
    if tables is not None:
        ps_df = tables.get("per_sample")
        panel_d = tables.get("panel_d")
        pt_df = tables.get("panel_e_taxa")
        bg_df = tables.get("panel_e_background")
        # Panel E's p-values travel in the data object, not recomputed on
        # redraw: recomputing gave the pickle a different set of stars from
        # the figure it was published beside.
        # replot_main nests the data object's extras under "_extras"; a
        # caller passing tables directly may put them at the top level.
        _ex = tables.get("_extras") or {}
        _ep = tables.get("energy_p") or _ex.get("energy_p") or []
        if isinstance(_ep, dict):
            energy_p = {tuple(k): float(v) for k, v in _ep.items()}
        else:                       # list of [condition, taxon_list, p]
            energy_p = {(str(c), str(l)): float(v) for c, l, v in _ep}
        if energy_p:
            print(f"  [panel E] {len(energy_p)} energy p-values from the "
                  "data object")
        _cs = tables.get("panel_e_curve_summary")
        if _cs is None:
            _cs = _ex.get("curve_summary")
        curve_summary = {}
        if _cs is not None and len(_cs):
            _cs = pd.DataFrame(_cs)
            if {"Condition", "Taxon_list"} <= set(_cs.columns):
                curve_summary = {(str(r.Condition), str(r.Taxon_list)):
                                 r._asdict()
                                 for r in _cs.itertuples(index=False)}
                print(f"  [panel E] {len(curve_summary)} mean-z statistics "
                      "from the data object")
        curve_panels = {}
        for _k, _p in (("taxa", "panel_e_curve_taxa"),
                       ("grid", "panel_e_curve_grid"),
                       ("all_taxa", "panel_e_all_taxa")):
            _v = tables.get(_p)
            if _v is not None and len(_v):
                curve_panels[_k] = pd.DataFrame(_v)
        if _cs is not None and len(_cs):
            curve_panels["summary"] = pd.DataFrame(_cs)
        if len(curve_panels) == 4:
            print("  [panel E] background-curve panels from the data object")
        else:
            curve_panels = {}
        # F-H travel in the data object like everything else: replot_main
        # hands this function results_dir="" and there is nothing to read.
        _fgh_pc = tables.get("panel_fgh_per_cell")
        _fgh_pr = tables.get("panel_fgh_paired")
        fgh_panels = ({"per_cell": pd.DataFrame(_fgh_pc),
                       "paired":   pd.DataFrame(_fgh_pr)}
                      if _fgh_pc is not None and _fgh_pr is not None
                      and len(_fgh_pc) and len(_fgh_pr) else {})
        if fgh_panels:
            print("  [panel F-H] list comparison from the data object")
        for _n, _v in (("per_sample", ps_df), ("panel_d", panel_d)):
            if _v is not None and _v.empty:
                raise ValueError(f"data object has an empty {_n} table")
    else:
        ps_df   = load_per_sample(results_dir, ref, spec_tag)
        panel_d = load_panel_d(results_dir, spec_tag)
        pt_df, bg_df = load_panel_e(results_dir, spec_tag)
        energy_p = load_energy_p(results_dir, spec_tag)
        curve_summary = load_curve_summary(results_dir, spec_tag)
        _cs_df = (pd.DataFrame(list(curve_summary.values()))
                  if curve_summary else None)
        curve_panels = load_curve_panels(results_dir, spec_tag)
        fgh_panels = load_list_comparison(results_dir, ref)

        # Deferred to just below, once the condition list exists: the data
        # object records the conditions the figure actually drew, and the
        # panel_e frames are a superset carrying every contrast the curve was
        # fitted for.
        _emit_fresh = True

    # Condition list
    conds: list = []
    if panel_d is not None:
        conds = list(panel_d["Condition"])
    elif pt_df is not None and bg_df is not None:
        comb = pd.concat([pt_df, bg_df], ignore_index=True)
        conds = [c for c in dict.fromkeys(comb["Condition"]) if c not in SKIP]
    conds = _display_order(conds)
    n_cond = max(len(conds), 1)

    if _emit_fresh:
        from persistence import figure_source as _fsrc
        _fsrc.emit(
            os.path.dirname(out_path) or ".", "F5",
            {k: v for k, v in (("per_sample", ps_df),
                               ("panel_d", panel_d),
                               ("panel_e_taxa", pt_df),
                               ("panel_e_background", bg_df),
                               ("panel_e_curve_summary", _cs_df),
                               ("panel_e_curve_taxa",
                                curve_panels.get("taxa")),
                               ("panel_e_curve_grid",
                                curve_panels.get("grid")),
                               ("panel_e_all_taxa",
                                curve_panels.get("all_taxa")),
                               ("panel_fgh_per_cell",
                                fgh_panels.get("per_cell")),
                               ("panel_fgh_paired",
                                fgh_panels.get("paired")))
             if v is not None},
            stem=os.path.splitext(os.path.basename(out_path))[0],
            reference=ref,
            extras={"drawn_conditions": list(conds),
                    "energy_p": [[c, l, v] for (c, l), v in
                                 energy_p.items()]},
            notes=("Main Figure 5. per_sample drives A-C (depth and richness "
                   "violins), panel_d the per-condition richness bars, and the "
                   "panel_e pair the target-vs-background prevalence "
                   f"distributions. spec_tag={spec_tag}, n_boot_E={n_boot_E}, "
                   "rng seed 42."))

    # Active E list pairs
    focus_lists   = ["VanishH+", "BlossumH+"]
    active_pairs: list = []
    if pt_df is not None and bg_df is not None:
        avail = set(pd.concat([pt_df, bg_df],
                               ignore_index=True)["Taxon_list"].unique())
        for ln in focus_lists:
            if ln in avail and f"{ln}_bg" in avail:
                active_pairs.append((ln, f"{ln}_bg"))
    # In curve form panel E draws one row per focus list and needs no _bg
    # partner, so the row count follows the lists rather than the pairs.
    n_pairs = (len(focus_lists) if curve_panels
               else max(len(active_pairs), 1))

    health_conds = {"Unhealthy", "Obese"}
    # draw_D maps dataframe rows to axes positionally, so panel D's table has
    # to carry the display order too -- reordering `conds` alone would put
    # every bar under the wrong heading.
    if panel_d is not None and not panel_d.empty:
        _rank = {c: i for i, c in enumerate(conds)}
        panel_d = (panel_d.assign(_o=panel_d["Condition"].map(_rank))
                          .sort_values("_o", kind="stable")
                          .drop(columns="_o")
                          .reset_index(drop=True))
    # The dashed divider separates the pooled contrasts (Unhealthy, Obese)
    # from the specific diseases. They used to be the first two columns, so
    # one boundary sufficed; with Unhealthy moved to the end the groups are
    # no longer contiguous, and a boundary is drawn wherever the group
    # changes.
    _grp = [c in health_conds for c in conds]
    health_bound = {i + 1 for i in range(len(_grp) - 1)
                    if _grp[i] != _grp[i + 1]} or {0}

    n_g2_map: dict = {}
    ess_map:  dict = {}
    if panel_d is not None:
        for _, row in panel_d.iterrows():
            n_g2_map[row["Condition"]] = int(row.get("n_g2", 0))
            ess_map[row["Condition"]]  = float(row.get("ess_g1", 0))

    # ── Figure geometry ───────────────────────────────────────────────
    FIG_W  = 24.0
    lm, rm = 0.075, 0.975
    avail_w = (rm - lm) * FIG_W
    ecol_w  = avail_w / n_cond          # no label column — full width for data

    # Row one at 60% of the height it used to take. A-C set up the depth
    # correction and are read once; D and E carry the result and were being
    # given less of the page than the panels explaining the method.
    ABC_SQUISH = 0.60
    # Measured against a FIXED five-column width, not `ecol_w`. A-C do not have
    # columns, so their height has no business tracking how many conditions E
    # shows: when the pooled contrast was dropped, n_cond went 5 -> 4, ecol_w
    # grew by a quarter and row one silently grew with it, which is the taller
    # first line. This pins it to the height it had at five columns.
    ABC_REF_COLS = 5.0
    abc_h  = (avail_w / ABC_REF_COLS) * 1.75 * ABC_SQUISH
    # One D row, not two. It is taller than the old top row on its own
    # because it has taken over both the stars, above the bars, and the
    # condition labels, below them -- but shorter than the two rows together.
    dtop_h = ecol_w * 1.18
    dbot_h = 0.0
    erow_h = ecol_w * 1.02

    # ── F-H: the three list-comparison panels, formerly Fig. S31 ──────
    # Square, so the block's height is its cell width. Indented from the left
    # margin because panel F labels its rows with condition names, which do
    # not fit in the margin A-E need for a rotated axis label and some digits.
    # Two gaps, not one. G carries no y-axis label and only an unlabelled tick
    # strip, so it sits as close to F as their x-tick labels allow; H has a
    # label of its own and needs room for it. One uniform wspace has to be the
    # larger of the two, which left a hand's width of blank paper beside G.
    FGH_GAP_FG = 0.055
    FGH_GAP_GH = 0.34
    FGH_LEFT_PAD = 0.075
    # And held off the right edge: panel H's x-axis label is wider than the
    # square panel it is centred under, and at `rm` its last word ran off the
    # canvas.
    FGH_RIGHT_PAD = 0.022
    fgh_left   = lm + FGH_LEFT_PAD
    fgh_right  = rm - FGH_RIGHT_PAD
    fgh_cell_w = ((fgh_right - fgh_left) * FIG_W) / (3 + FGH_GAP_FG
                                                     + FGH_GAP_GH)
    FGH_LABEL_BAND_IN = 0.40 * (FIG_W / 7.24)   # tick labels + x-axis label
    FGH_GAP_IN        = 0.14 * (FIG_W / 7.24)   # clear of E's legend
    fgh_block_h = (fgh_cell_w + FGH_LABEL_BAND_IN + FGH_GAP_IN
                   if fgh_panels else 0.0)

    # In curve form panel E's cells are square at the shared spacing, so the
    # block needs exactly this much height. Budgeting for it here means the
    # gridspec span below never has to be clamped against the bottom margin
    # -- clamping shortened the span, and set_box_aspect then shrank the
    # square box in WIDTH, which showed up as a wider horizontal gap than
    # S26's.
    # Cell spacing comes from the shared constants, which S25 also takes, so
    # the two drawings of this grid cannot drift apart. They were tightened on
    # 2026-10-16; see plot_utils.CELL_WSPACE for why the horizontal gap cannot
    # go much lower. Used for BOTH the height budget and the GridSpec below,
    # which must agree or the cells stop being square.
    if curve_panels:
        from persistence.plot_utils import (CELL_WSPACE as _CW,
                                               CELL_HSPACE as _CH,
                                               CELL_ASPECT as _CA)
        _cell_w = avail_w / (n_cond + (n_cond - 1) * _CW)
        e_block_h = (n_pairs + (n_pairs - 1) * _CH) * _cell_w * _CA
    else:
        e_block_h = n_pairs * erow_h

    # Room under the E block for its x-axis labels and the figure legend.
    # Printed that band is about 0.46 in, the same reservation S26 makes;
    # this canvas is FIG_W / 7.24 times the column width.
    E_LEGEND_BAND_IN = 0.46 * (FIG_W / 7.24) if curve_panels else 1.0
    fig_h  = (abc_h + dtop_h + dbot_h + e_block_h + E_LEGEND_BAND_IN
              + fgh_block_h + 0.6)

    fig = plt.figure(figsize=(FIG_W, fig_h))

    # ── A / B / C (own GridSpec) ─────────────────────────────────────
    abc_top = 0.97
    abc_bot = abc_top - abc_h/fig_h
    abc_gs  = gridspec.GridSpec(
        1, 3, figure=fig,
        left=lm, right=rm, top=abc_top, bottom=abc_bot,
        wspace=0.22)
    ax_A = fig.add_subplot(abc_gs[0,0])
    ax_B = fig.add_subplot(abc_gs[0,1])
    ax_C = fig.add_subplot(abc_gs[0,2])

    draw_A(ax_A, ps_df)
    if ps_df is not None:
        draw_violin(ax_B, ps_df, corrected=False, title="Raw richness", rng=rng)
        draw_violin(ax_C, ps_df, corrected=True,  title="GMW corrected", rng=rng)
    else:
        for ax_, t in [(ax_B,"Raw richness"),(ax_C,"GMW corrected")]:
            _style(ax_)
            ax_.text(0.5,0.5,"Run pipeline with\n--only health_richness",
                     transform=ax_.transAxes, ha="center", va="center",
                     fontsize=FS-2, color="#888", style="italic",
                     multialignment="center")
            ax_.set_title(t, fontsize=FS+2, fontweight="bold", pad=5)

    for ax_, lbl in [(ax_A,"A"),(ax_B,"B"),(ax_C,"C")]:
        ax_.text(-0.10, 1.04, lbl, transform=ax_.transAxes,
                 fontsize=FS+5, fontweight="bold", va="bottom")

    # ── D+E: single GridSpec, no label column ────────────────────────
    gap_f  = 0.050
    de_top = abc_bot - gap_f
    de_bot = fgh_block_h / float(fig_h) + 0.04
    de_nr  = 1 + n_pairs

    # D's bars are butted deliberately. E's curve cells each carry their own
    # pp scale, so they need the gaps their tick labels and axis labels sit
    # in; butting them together is what made the earlier density form read as
    # one strip. One GridSpec cannot be both, so in curve form E gets its own.
    de_gs = gridspec.GridSpec(
        de_nr, n_cond,
        figure=fig,
        height_ratios=[dtop_h] + [erow_h]*n_pairs,
        width_ratios=[1.0]*n_cond,
        left=lm, right=rm,
        top=de_top, bottom=de_bot,
        hspace=0.0, wspace=0.0)
    if curve_panels:
        from persistence.plot_utils import (CELL_WSPACE as _CELL_WSPACE,
                                               CELL_HSPACE as _CELL_HSPACE,
                                               CELL_ASPECT)
        # Anchor the E block from the BOTTOM, above a reserved band for its
        # x-axis labels and the figure legend, and give it exactly the height
        # its square cells need. Hanging it from the top instead let it grow
        # down over the legend once the slack was removed. D then takes
        # whatever is left between the ABC block and the top of E.
        _e_bot = (fgh_block_h + E_LEGEND_BAND_IN) / float(fig_h)
        _e_top = _e_bot + e_block_h / float(fig_h)
        _split = _e_top + 0.008
        de_gs = gridspec.GridSpec(
            1, n_cond, figure=fig,
            height_ratios=[dtop_h], width_ratios=[1.0]*n_cond,
            left=lm, right=rm, top=de_top, bottom=_split,
            hspace=0.0, wspace=0.0)
        e_gs = gridspec.GridSpec(
            n_pairs, n_cond, figure=fig,
            height_ratios=[erow_h]*n_pairs, width_ratios=[1.0]*n_cond,
            left=lm, right=rm, top=_e_top, bottom=_e_bot,
            hspace=_CELL_HSPACE, wspace=_CELL_WSPACE)
    else:
        e_gs = None

    # ── D axes ───────────────────────────────────────────────────────
    D_top: list = []
    for ci in range(n_cond):
        sh_t = D_top[0] if D_top else None
        D_top.append(fig.add_subplot(de_gs[0, ci], sharey=sh_t))

    if panel_d is not None and not panel_d.empty:
        draw_D(D_top, None, panel_d, health_bound)
    else:
        for ax_ in D_top:
            _style(ax_); ax_.set_xticks([]); ax_.set_yticks([])

    # "D" panel label on top-left of first D_top
    D_top[0].text(-0.02, 1.02, "D", transform=D_top[0].transAxes,
                  fontsize=FS+5, fontweight="bold", va="bottom", ha="right")

    # D legend — between columns 1 and 2 (straddles boundary)
    if n_cond >= 2:
        d_leg_handles = [
            Patch(facecolor=BAR_COL, alpha=0.33, edgecolor=BAR_COL,
                  label="Control"),
            Patch(facecolor=BAR_COL, alpha=0.92, edgecolor="white",
                  label="Case"),
        ]
        _place_legend(fig, D_top[0], d_leg_handles,
                      ylo_frac=0.56, yhi_frac=0.96,
                      fontsize=FS_SMALL, title=None)

    # ── E axes and drawing ────────────────────────────────────────────
    E_rows: list = []   # E_rows[pi][ci] = ax
    for pi in range(n_pairs):
        row: list = []
        for ci in range(n_cond):
            sh_y = E_rows[pi][0] if pi < len(E_rows) and E_rows[pi] else None
            ax   = (fig.add_subplot(e_gs[pi, ci]) if curve_panels else
                    fig.add_subplot(de_gs[1+pi, ci],
                                    sharey=sh_y if ci > 0 else None))
            row.append(ax)
        E_rows.append(row)

    # "E" panel label
    if E_rows:
        E_rows[0][0].text(-0.02, 1.02, "E", transform=E_rows[0][0].transAxes,
                           fontsize=FS+5, fontweight="bold", va="bottom", ha="right")

    if curve_panels:
        # Panel E and supplementary S26 are the same figure for different
        # lists. Every size comes from plot_utils in printed points; this
        # canvas is FIG_W wide and is reduced to the column width on save, so
        # it passes that magnification and nothing here sets a size of its
        # own. That is what makes the two match.
        from persistence.plot_utils import (draw_background_curve_grid,
                                               background_curve_legend,
                                               background_curve_style)
        _st = background_curve_style(FIG_W / 7.24)
        try:
            from persistence import config as _cfg
            _N_BOOT_CURVE = int(getattr(_cfg, "HEALTH_BG_N_BOOT", 2000))
        except Exception:
            _N_BOOT_CURVE = 200
        _ct, _cg = curve_panels["taxa"], curve_panels["grid"]
        _call = curve_panels["all_taxa"]
        _frames = {c: g.set_index("taxon")
                   for c, g in _call.groupby("Condition")}
        draw_background_curve_grid(
            E_rows, conds, focus_lists, _frames, _ct, _cg,
            curve_summary, LIST_COLORS, {l: l for l in focus_lists},
            style=_st, floor=1e-4, titles=False,
            n_boot=_N_BOOT_CURVE, colour_row_labels=True)
        # No _style() here: it set an absolute 0.7 pt spine and 3 pt ticks,
        # which on this 24 in canvas print at a quarter of what S26 gets.
        # plot_utils.style_background_curve_axes does it for both now.
        _lg = background_curve_legend(
            focus_lists, LIST_COLORS, {l: l for l in focus_lists}, style=_st)
        fig.legend(handles=_lg, loc="lower center", ncol=len(_lg),
                   frameon=False, fontsize=_st["fs_legend"],
                   bbox_to_anchor=(0.5, 0.004 + fgh_block_h / float(fig_h)))

    elif active_pairs and pt_df is not None and bg_df is not None:
        _common_shift(pt_df, bg_df)
        combined_e = pd.concat([pt_df, bg_df], ignore_index=True)

        for pi, (list_name, bg_name) in enumerate(active_pairs):
            if pi >= len(E_rows): break
            col   = LIST_COLORS.get(list_name, "#888")
            bgcol = _bgcol(col)
            row   = E_rows[pi]

            for ci, cond in enumerate(conds):
                if ci >= len(row): break
                ax = row[ci]

                _sc = _shift_col(combined_e) or "prev_diff"
                sub_l = combined_e[
                    (combined_e["Condition"]  == cond) &
                    (combined_e["Taxon_list"] == list_name)
                ].dropna(subset=[_sc])
                sub_b = combined_e[
                    (combined_e["Condition"]  == cond) &
                    (combined_e["Taxon_list"] == bg_name)
                ].dropna(subset=[_sc])

                if len(sub_l) < 3 or sub_b.empty:
                    _style(ax)
                    ax.text(0.5,0.5,"n/a", transform=ax.transAxes,
                            ha="center",va="center",fontsize=FS-2,color="#aaa")
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    sl = _shift_pp(sub_l)
                    bg_uniq = (sub_b.groupby("taxon",sort=False).first().reset_index()
                               if "taxon" in sub_b.columns else sub_b.copy())
                    sb_vals = _shift_pp(bg_uniq)
                    sb_w    = _bg_w(bg_uniq)

                    draw_E_cell(ax, sl, sb_vals, sb_w,
                                col, bgcol, n_boot_E, rng,
                                p_energy=energy_p.get((cond, list_name)),
                                curve_row=curve_summary.get((cond, list_name)))

                ax.set_yticks([])
                ax.tick_params(labelleft=False)
                # Y-axis label on leftmost column only
                if ci == 0:
                    ax.set_ylabel("Density", fontsize=FS)
                    ax.tick_params(labelleft=False)  # no tick numbers, just label
                # Left spine: grey for non-leftmost (wspace=0, butted up)
                if ci > 0:
                    ax.spines["left"].set_color("#dddddd")

                # x-axis label: bottom E row, no rotation, no ESS
                if pi == n_pairs - 1:
                    ax.set_xlabel(
                        "Case − control\n"
                        "Prevalence diff. (%)",
                        fontsize=FS_SMALL, labelpad=3)
                    ax.tick_params(labelbottom=True, labelsize=FS_SMALL,
                                   rotation=0)
                else:
                    ax.set_xlabel("")
                    ax.tick_params(labelbottom=False)

            # E legend — between col 0 and col 1, spanning 15-65% height
            if len(row) >= 2:
                e_leg_handles = [
                    Patch(facecolor=col,       alpha=0.55, edgecolor="none",
                          label=list_name),
                    Patch(facecolor=bgcol,     alpha=0.40, edgecolor="none",
                          label="Background"),
                    Patch(facecolor="#bbbbbb",  alpha=0.55, edgecolor="none",
                          label="Overlap"),
                ]
                _place_legend(fig, row[0], e_leg_handles,
                              ylo_frac=0.15, yhi_frac=0.65,
                              fontsize=FS_SMALL)

    else:
        for pi, row in enumerate(E_rows):
            for ax_ in row:
                _style(ax_)
                ax_.text(0.5,0.5,"prevmatched CSVs not found",
                         transform=ax_.transAxes, ha="center",
                         fontsize=FS-2, color="#aaa", style="italic")
                ax_.set_xticks([]); ax_.set_yticks([])

    # ── F / G / H — the six lists head to head (was Fig. S31) ─────────
    if fgh_panels:
        from list_comparison_panels import (_family_panel, _paired_panel,
                                            _summary as _lc_summary)
        _fgh_sc = FIG_W / 7.24
        _fgh_bot = FGH_LABEL_BAND_IN / float(fig_h)
        # Five columns, three of them panels: a GridSpec's wspace is one
        # number for every gap, and these two gaps are not the same size.
        fgh_gs = gridspec.GridSpec(
            1, 5, figure=fig,
            width_ratios=[1.0, FGH_GAP_FG, 1.0, FGH_GAP_GH, 1.0],
            left=fgh_left, right=fgh_right,
            top=_fgh_bot + fgh_cell_w / float(fig_h), bottom=_fgh_bot,
            wspace=0.0)
        ax_F, ax_G, ax_H = (fig.add_subplot(fgh_gs[0, i]) for i in (0, 2, 4))
        _pc, _pr = fgh_panels["per_cell"], fgh_panels["paired"]
        _family_panel(ax_F, _pc, "VANISH",  show_ylabels=True,  scale=_fgh_sc)
        _family_panel(ax_G, _pc, "BloSSUM", show_ylabels=False, scale=_fgh_sc)
        _paired_panel(ax_H, _pr, _lc_summary(_pr, _pc), scale=_fgh_sc,
                      rich_labels=True)
        for ax_, lbl in ((ax_F, "F"), (ax_G, "G"), (ax_H, "H")):
            ax_.text(-0.02, 1.045, lbl, transform=ax_.transAxes,
                     fontsize=FS+5, fontweight="bold", va="bottom", ha="right")

    # The display-time renames, which this figure does not get for free: it
    # saves through save_column_width rather than plot_utils.savefig_dual, so
    # plot_utils.apply_display_text never runs on it. The comment beside
    # _display_text above said Figure 5 "needs its own call" and the call was
    # never added, which is why panel A's legend read "Non-Industrial" while
    # panels C and D read "Non-industrial" two inches away.
    for _artist in fig.findobj(match=matplotlib.text.Text):
        if getattr(_artist, "_keep_text", False):
            continue
        _old = _artist.get_text()
        _new = _display_text(_old)
        if _new != _old:
            _artist.set_text(_new)

    # Reduced to the printed column width in the file rather than on the
    # page; the drawing itself is untouched (see figure_rescale).
    from persistence.figure_rescale import save_column_width
    stem = os.path.splitext(out_path)[0]
    save_column_width(fig, stem)
    plt.close(fig)
    for _e in ("png", "pdf"):
        print(f"Saved: {stem}.{_e}")






# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Generate Figure 5.  Run pipeline --only health_richness first.")
    ap.add_argument("--ref",    default="combined",
                    help="reference to build from. The main text reports "
                         "combined; UHGG disagrees with the supplementary tables.")
    ap.add_argument("--spec",   default="breadth_pct")
    ap.add_argument("--results",default=None,
                    help="Section 3 output directory (default: results/<ref>/section3_health/main/)")
    ap.add_argument("--out",    default="main_text_figures/figure5.pdf")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    if args.results is None:
        args.results = os.path.join("results", args.ref, "section3_health", "main")

    if not os.path.isdir(args.results):
        print(f"Error: dir not found: {args.results}"); sys.exit(1)

    spec_tag = _resolve_spec_tag(args.results, args.spec)
    print(f"Using spec_tag: {spec_tag}")
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    build_figure5(results_dir=args.results, ref=args.ref,
                  spec_tag=spec_tag, out_path=args.out,
                  n_boot_E=args.n_boot)


if __name__ == "__main__":
    main()
