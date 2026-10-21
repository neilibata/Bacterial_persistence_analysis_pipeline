"""
figure_s_guide_fig5.py  —  Figure S6: How to read Figure 5 and Fig. S3
=======================================================================
A four-panel reader-orientation figure placed at the start of Note 6.

Layout
------
  A (top-left)     Cohort map — per-condition case/control counts, studies, ESS
  B (top-right)    Analysis workflow — left-to-right flow diagram
  C (bottom-left)  How to read Figure 5 — three annotated rows
  D (bottom-right) How to read Fig. S3 — Q&A table
  Footer           Sign conventions and notation

Panel A data source
-------------------
Loaded from the pipeline output file:
    {ref_outdir}/section3_health/health_per_condition_cmh_{spec_tag}.csv

Columns used: Condition, n_group1 (controls), n_group2 (cases),
              n_studies, mean_ess_frac, median_log_depth_diff
Method filter: "Geom-mean-weighted CMH" (one row per condition, any list).

The spec_tag is inferred automatically by scanning for available CSV files;
it falls back to "10pct" (the main-text breadth spec) if no other is found.

Colour scheme: Wong (2011) CB-safe palette, identical to figure_s0_roadmap.py.

Usage
-----
    # standalone
    python figure_s_guide_fig5.py --ref-outdir results/UHGG --out results/UHGG

    # pipeline module
    from persistence.figure_s_guide_fig5 import run
    run(ref_outdir="results/UHGG", outdir="results/UHGG")
"""

from __future__ import annotations

import os
import glob
import argparse
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter


# ══════════════════════════════════════════════════════════════════════════════
#  Palette  —  identical to figure_s0_roadmap.py
# ══════════════════════════════════════════════════════════════════════════════

_VER = "#D55E00"
_YEL = "#F0E442"
_ORA = "#E69F00"
_SKY = "#56B4E9"
_GRN = "#009E73"
_BLU = "#0072B2"
_PNK = "#CC79A7"


def _tint(hex6: str, alpha: float = 0.25) -> str:
    h = hex6.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    return (f"#{int(r*alpha + 255*(1-alpha)):02X}"
            f"{int(g*alpha + 255*(1-alpha)):02X}"
            f"{int(b*alpha + 255*(1-alpha)):02X}")


C: dict[str, str] = dict(
    ind_h=_GRN, ind_u=_BLU, nonind=_ORA, hg=_VER, trans=_YEL, ind_life=_SKY,
    screen    = "#F0F0F0",
    method    = _tint(_YEL, 0.28),
    figbox    = "#F5F5F5",
    preproc   = _tint(_GRN, 0.22),
    edge      = "#3A3A3A",
    arrow     = "#1A1A1A",
    dash      = "#909090",
)

_TEXT = "#111111"

FS: dict[str, float] = dict(
    fig_title   = 26.0,
    panel_label = 24.0,
    panel_title = 20.0,
    box_title   = 17.0,
    box_body    = 15.0,
    bar_tick    = 15.0,
    tbl_hdr     = 17.0,
    tbl_body    = 15.0,
    footnote    = 14.0,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Cohort data loading from pipeline output
# ══════════════════════════════════════════════════════════════════════════════

# Conditions shown in HEALTH_PLOT_SKIP in config.py are excluded from plots
# but still tested. We follow the same logic here: show all conditions that
# appear in the CSV, in the order they appear, without any catch-all grouping.
_PREFER_METHOD = "Geom-mean-weighted CMH"

# Conditions the pipeline runs but does not plot. Read from config rather
# than restated: this was a hand-kept copy that named a contrast which no
# longer exists, and the set in config is now empty by default because
# eligibility is decided by the >=CMH_MIN_STRATUM / >=MIN_STRATA_TO_PLOT rule.
# The repo root goes on the path BEFORE the import: run by path, sys.path[0] is
# `figures/` and `persistence` is not importable from it. The except branch
# below silently substitutes an empty skip set, which changes which panels are
# drawn. See figure2.py for the full note.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from persistence import config as _cfg
    _PLOT_SKIP = set(getattr(_cfg, "HEALTH_PLOT_SKIP", ()))
except Exception as _exc:
    print(f"  [warn] persistence.config unavailable ({_exc}); drawing every "
          f"condition, including any config.HEALTH_PLOT_SKIP names",
          file=_sys.stderr)
    _PLOT_SKIP = set()


def load_cohort_data(
    ref_outdir: str,
    spec_tag: Optional[str] = None,
) -> Optional[list[dict]]:
    """
    Load per-condition cohort statistics from the pipeline CMH output CSV.

    Searches (in order):
      1. {ref_outdir}/section3_health/health_per_condition_cmh_*.csv
      2. {ref_outdir}/health_per_condition_cmh_*.csv          (flat layout)
      3. {ref_outdir}/**/health_per_condition_cmh_*.csv       (recursive)
    """
    import pandas as pd

    search_dirs = [
        os.path.join(ref_outdir, "section3_health"),
        ref_outdir,
    ]

    print(f"  [figure_s_guide_fig5] Searching for health_per_condition_cmh CSV:")
    print(f"    ref_outdir = {os.path.abspath(ref_outdir)}")

    csv_path = None

    for search_dir in search_dirs:
        abs_dir = os.path.abspath(search_dir)
        if not os.path.isdir(search_dir):
            print(f"    {abs_dir}  -> directory not found, skipping")
            continue

        found = sorted(glob.glob(os.path.join(search_dir,
                                              "health_per_condition_cmh_*.csv")))
        print(f"    {abs_dir}  -> {[os.path.basename(f) for f in found] or 'no matches'}")

        if not found:
            continue

        if spec_tag is not None:
            target = os.path.join(search_dir,
                                  f"health_per_condition_cmh_{spec_tag}.csv")
            csv_path = target if os.path.exists(target) else found[0]
        else:
            def _priority(p: str) -> int:
                b = os.path.basename(p)
                if "10pct" in b:    return 0
                if "combined" in b: return 1
                return 2
            csv_path = sorted(found, key=_priority)[0]

        break

    # Recursive fallback
    if csv_path is None:
        pattern = os.path.join(ref_outdir, "**",
                               "health_per_condition_cmh_*.csv")
        found = sorted(glob.glob(pattern, recursive=True))
        print(f"    recursive search: "
              f"{[os.path.relpath(f, ref_outdir) for f in found] or 'nothing found'}")
        if found:
            csv_path = found[0]

    if csv_path is None:
        print(f"  [figure_s_guide_fig5] Could not locate file. "
              f"Panel A will show a placeholder.")
        return None

    print(f"  [figure_s_guide_fig5] Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    method_col = "Method"
    if method_col in df.columns:
        preferred = df[df[method_col] == _PREFER_METHOD]
        if preferred.empty:
            print(f"    Method '{_PREFER_METHOD}' not found; "
                  f"available: {df[method_col].unique().tolist()}")
            preferred = df
    else:
        preferred = df

    dedup = preferred.drop_duplicates(subset="Condition", keep="first")
    dedup = dedup[~dedup["Condition"].isin(_PLOT_SKIP)]

    records = []
    for _, row in dedup.iterrows():
        n_ctrl = int(row.get("n_group1", 0))
        n_case = int(row.get("n_group2", 0))
        n_stud = row.get("n_studies", None)
        ess    = row.get("mean_ess_frac", None)
        med_ld = row.get("median_log_depth_diff", None)

        n_stud = int(n_stud) if (n_stud is not None and not _isnan(n_stud)) else None
        ess    = float(ess)  if (ess    is not None and not _isnan(ess))    else None
        med_ld = float(med_ld) if (med_ld is not None and not _isnan(med_ld)) else None

        records.append(dict(
            condition   = str(row["Condition"]),
            n_controls  = n_ctrl,
            n_cases     = n_case,
            n_studies   = n_stud,
            ess_frac    = ess,
            med_ld_diff = med_ld,
        ))

    print(f"  [figure_s_guide_fig5] {len(records)} conditions: "
          f"{[r['condition'] for r in records]}")
    return records


def _isnan(v) -> bool:
    try:
        return float(v) != float(v)   # NaN != NaN
    except (TypeError, ValueError):
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Low-level drawing helpers (axes-fraction space)
# ══════════════════════════════════════════════════════════════════════════════

def _box(
    ax: plt.Axes,
    cx: float, cy: float, w: float, h: float,
    text: str, fc: str,
    ec: str | None = None,
    lw: float = 1.4,
    fs: float | None = None,
    bold: bool = True,
    ph: float = 10.0,
) -> None:
    ec      = ec or C["edge"]
    fs_body = fs if fs is not None else FS["box_body"]
    fs_ttl  = FS["box_title"] if bold else fs_body

    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.012",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        transform=ax.transAxes, clip_on=False, zorder=3,
    ))

    lines    = text.split("\n")
    n        = max(len(lines), 1)
    pt_in_ax = 1.0 / (72.0 * ph)
    spacing  = min(fs_body * pt_in_ax * 2.05, h * 0.80 / n)
    y_top    = cy + spacing * (n - 1) / 2

    for i, line in enumerate(lines):
        ax.text(
            cx, y_top - i * spacing, line,
            ha="center", va="center",
            fontsize=fs_ttl if (bold and i == 0) else fs_body,
            fontweight="bold" if (bold and i == 0) else "normal",
            color=_TEXT,
            transform=ax.transAxes, clip_on=False, zorder=4,
        )


def _arrow(
    ax: plt.Axes,
    x0: float, y0: float,
    x1: float, y1: float,
    dashed: bool = False,
    col: str | None = None,
    lw: float = 1.5,
) -> None:
    col = col or (C["dash"] if dashed else C["arrow"])
    ax.annotate(
        "",
        xy=(x1, y1), xytext=(x0, y0),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>", color=col, lw=lw,
            linestyle=(0, (5, 3)) if dashed else "solid",
            mutation_scale=12,
        ),
        zorder=5,
    )


def _seg(
    ax: plt.Axes,
    x0: float, y0: float,
    x1: float, y1: float,
    col: str | None = None, lw: float = 1.4, dashed: bool = False,
) -> None:
    ax.plot([x0, x1], [y0, y1],
            color=col or C["arrow"], lw=lw,
            ls=(0, (5, 3)) if dashed else "-",
            transform=ax.transAxes, clip_on=False, zorder=5)


def _panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.02, 1.04, letter, ha="left", va="bottom",
            fontsize=FS["panel_label"], fontweight="bold", color=_TEXT,
            transform=ax.transAxes, clip_on=False)


def _panel_title(ax: plt.Axes, text: str) -> None:
    ax.text(0.50, 1.01, text, ha="center", va="bottom",
            fontsize=FS["panel_title"], fontweight="bold", color="#333333",
            transform=ax.transAxes, clip_on=False)


# ══════════════════════════════════════════════════════════════════════════════
#  Panel A — Cohort map
# ══════════════════════════════════════════════════════════════════════════════

def _draw_A(ax: plt.Axes, cohort_data: Optional[list[dict]]) -> None:

    if not cohort_data:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(0.5, 0.5,
                "Cohort data not yet available.\n"
                "Run the Section 3 health analysis pipeline first,\n"
                "then re-run this figure script with --ref-outdir.",
                ha="center", va="center", fontsize=FS["box_body"] + 1,
                color="#888888", style="italic",
                transform=ax.transAxes)
        _panel_letter(ax, "A")
        _panel_title(ax, "Industrial health cohorts  (Table Sx)")
        return

    conds   = [d["condition"]  for d in cohort_data]
    cases   = np.array([d["n_cases"]    for d in cohort_data], float)
    ctrls   = np.array([d["n_controls"] for d in cohort_data], float)
    n_stud  = [d["n_studies"]  for d in cohort_data]
    ess_f   = [d["ess_frac"]   for d in cohort_data]
    med_ld  = [d["med_ld_diff"] for d in cohort_data]
    nc      = len(conds)
    yp      = np.arange(nc)

    case_fc = _tint(_BLU, 0.72)
    ctrl_fc = _tint(_GRN, 0.72)

    ax.barh(yp, -ctrls, height=0.60, color=ctrl_fc,
            edgecolor=C["edge"], linewidth=0.9,
            label="Controls  (healthy industrial)", zorder=2)
    ax.barh(yp,  cases,  height=0.60, color=case_fc,
            edgecolor=C["edge"], linewidth=0.9,
            label="Cases  (unhealthy industrial)", zorder=2)

    # Thicker border on well-powered conditions (≥4 studies)
    for i, ns in enumerate(n_stud):
        if ns is not None and ns >= 4:
            ax.barh(i, -ctrls[i], height=0.60, fill=False,
                    edgecolor="#1A5E20", linewidth=2.0, zorder=3)
            ax.barh(i,  cases[i],  height=0.60, fill=False,
                    edgecolor="#003070", linewidth=2.0, zorder=3)

    ax.axvline(0, color="#555555", lw=1.6, zorder=5)
    ax.invert_yaxis()

    ax.set_yticks(yp)
    ax.set_yticklabels(conds, fontsize=FS["bar_tick"], color=_TEXT)

    maxv = max(ctrls.max(), cases.max())
    ax.set_xlim(-maxv * 1.12, maxv * 1.82)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(abs(x)):,}"))
    ax.set_xlabel("Sample count", fontsize=FS["bar_tick"], color="#444444",
                  labelpad=6)

    # Column headers
    ax.text(-maxv * 0.54, -0.65, "Controls",
            ha="center", fontsize=FS["bar_tick"] + 0.5,
            fontweight="bold", color=_GRN)
    ax.text( maxv * 0.54, -0.65, "Cases",
            ha="center", fontsize=FS["bar_tick"] + 0.5,
            fontweight="bold", color=_BLU)
    ax.text( maxv * 1.38, -0.65, "studies  /  ESS_frac",
            ha="center", fontsize=FS["bar_tick"] - 0.5,
            fontweight="bold", color="#444444", clip_on=False)

    # Right-side annotations: n_studies and ESS_frac only
    # (med_LD_diff defined in footer; full values in Table S6)
    ann_x = maxv * 1.15
    for i, (ns, ef) in enumerate(zip(n_stud, ess_f)):
        ns_str = f"{ns}" if ns is not None else "?"
        ef_str = f"{ef:.2f}" if ef is not None else "?"
        ax.text(ann_x, i,
                f"{ns_str}  /  {ef_str}",
                ha="left", va="center",
                fontsize=FS["bar_tick"], color="#444444", clip_on=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=FS["bar_tick"])
    # Legend removed: bars already headed "Controls" / "Cases" at top

    _panel_letter(ax, "A")
    _panel_title(ax, "Industrial health cohorts  (Table S6)")
    # Notes below panel title
    ax.text(0.50, 1.005,
            "full depth-balance diagnostic (med_LD_diff) in Table S6",
            ha="center", va="top",
            fontsize=FS["bar_tick"] - 3.5, color="#888888", style="italic",
            transform=ax.transAxes, clip_on=False)


# ══════════════════════════════════════════════════════════════════════════════
#  Panel B — Analysis workflow
# ══════════════════════════════════════════════════════════════════════════════

def _draw_B(ax: plt.Axes, ph: float) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ── Cross-reference note (top-right) ──────────────────────────────────────
    ax.text(0.98, 0.98,
            "Health analysis branch of Fig. S0\n(see Fig. S0 for full study roadmap)",
            ha="right", va="top",
            fontsize=FS["box_body"] - 3.0, color="#AAAAAA", style="italic",
            transform=ax.transAxes, clip_on=False)

    # ── Shared input chain: three horizontal boxes (left side) ────────────────
    bw_in = 0.118; bh_in = 0.175; cy_in = 0.620
    xs_in = [0.068, 0.213, 0.358]

    in_specs = [
        ("Industrial\nhealth\ncohorts",           _tint(_SKY, 0.28)),
        ("Stratify\nby study",                    C["screen"]),
        ("Equalise depth\nwithin strata\n(GMW)",  C["preproc"]),
    ]
    for i, (txt, fc) in enumerate(in_specs):
        _box(ax, xs_in[i], cy_in, bw_in, bh_in, txt, fc, fs=FS["box_body"], ph=ph)
        if i < len(xs_in) - 1:
            _arrow(ax, xs_in[i] + bw_in / 2, cy_in,
                       xs_in[i + 1] - bw_in / 2, cy_in, lw=1.8)

    # Split point: right edge of equalize-depth box
    split_x = xs_in[2] + bw_in / 2   # ≈ 0.477

    # ── List families input box (Note 5) — feeds list-level branch ───────────
    lf_cx = 0.477; lf_cy = 0.260; lf_w = 0.170; lf_h = 0.155
    _box(ax, lf_cx, lf_cy, lf_w, lf_h,
         "List families\n(Note 5)\norig. VANISH/BloSSUM\nVANISH+/BloSSUM+\nH+ variants",
         _tint(_YEL, 0.28), fs=FS["box_body"] - 2.5, bold=True, ph=ph)

    # ── BRANCH 1 (upper): Richness contrasts → Fig. 5A–D ─────────────────────
    cy_rich = 0.855; cx_rich = 0.670; bw_rich = 0.155; bh_rich = 0.135

    _seg(ax, split_x, cy_in, split_x, cy_rich, lw=1.8)
    _arrow(ax, split_x, cy_rich, cx_rich - bw_rich / 2, cy_rich, lw=1.8)

    _box(ax, cx_rich, cy_rich, bw_rich, bh_rich,
         "Richness\ncontrasts", _tint(_GRN, 0.22), fs=FS["box_body"], ph=ph)

    ax.text(cx_rich, cy_rich - bh_rich / 2 - 0.025,
            "weighted richness",
            ha="center", va="top",
            fontsize=FS["box_body"] - 2.0, color="#444444", style="italic",
            transform=ax.transAxes, clip_on=False)

    ax.text(cx_rich + bw_rich / 2 + 0.015, cy_rich,
            "\u2192 Fig. 5A\u2013D",
            ha="left", va="center",
            fontsize=FS["box_body"], fontweight="bold", color="#222222",
            transform=ax.transAxes, clip_on=False)

    # ── BRANCH 2 (lower): Per-taxon GMW-CMH → list-level tests ───────────────
    cy_lst = 0.415; bw_lst = 0.115; bh_lst = 0.190
    cx_gmw = 0.640; cx_prv = 0.770; cx_lst = 0.895

    _seg(ax, split_x, cy_in, split_x, cy_lst, lw=1.8)
    _arrow(ax, split_x, cy_lst, cx_gmw - bw_lst / 2, cy_lst, lw=1.8)

    _box(ax, cx_gmw, cy_lst, bw_lst, bh_lst,
         "GMW-CMH\nper taxon", C["method"], fs=FS["box_body"], ph=ph)
    ax.text(cx_gmw + bw_lst / 2 + 0.008, cy_lst + 0.025,
            "log-OR,  prev_diff",
            ha="left", va="center",
            fontsize=FS["box_body"] - 2.0, color="#444444", style="italic",
            transform=ax.transAxes, clip_on=False)

    _arrow(ax, cx_gmw + bw_lst / 2, cy_lst, cx_prv - bw_lst / 2, cy_lst, lw=1.8)
    _box(ax, cx_prv, cy_lst, bw_lst, bh_lst,
         "Prevalence-\nmatched\nbackground", _tint(_ORA, 0.28),
         fs=FS["box_body"], ph=ph)

    _arrow(ax, cx_prv + bw_lst / 2, cy_lst, cx_lst - bw_lst / 2, cy_lst, lw=1.8)
    _box(ax, cx_lst, cy_lst, bw_lst, bh_lst,
         "List-level\ntests", _tint(_PNK, 0.25), fs=FS["box_body"], ph=ph)
    ax.text(cx_lst, cy_lst - bh_lst / 2 - 0.025,
            "energy distance  (primary)",
            ha="center", va="top",
            fontsize=FS["box_body"] - 2.0, color="#444444", style="italic",
            transform=ax.transAxes, clip_on=False)

    ax.text(cx_lst + bw_lst / 2 + 0.010, cy_lst,
            "\u2192 Fig. 5E\nS6D\u2013B,  S6E\u2013G",
            ha="left", va="center",
            fontsize=FS["box_body"] - 0.5, fontweight="bold", color="#222222",
            transform=ax.transAxes, clip_on=False)

    # List families → list-level branch (dashed input)
    _arrow(ax, lf_cx + lf_w / 2, lf_cy + lf_h / 2,
               cx_prv, cy_lst - bh_lst / 2,
           dashed=True, col=C["dash"], lw=1.4)

    # ── DRIVER-TAXA sub-branch (dashed, from GMW-CMH downward) ───────────────
    drv_cy = 0.145; drv_w = 0.160; drv_h = 0.125
    _arrow(ax, cx_gmw, cy_lst - bh_lst / 2,
               cx_gmw, drv_cy + drv_h / 2,
           dashed=True, col=C["dash"], lw=1.4)
    _box(ax, cx_gmw, drv_cy, drv_w, drv_h,
         "Driver taxa",
         _tint(_YEL, 0.20), fs=FS["box_body"] - 1.0, bold=True, ph=ph)
    ax.text(cx_gmw + drv_w / 2 + 0.012, drv_cy,
            "\u2192 S6F\u2013J",
            ha="left", va="center",
            fontsize=FS["box_body"] - 1.0, color="#444444",
            transform=ax.transAxes, clip_on=False)

    _panel_letter(ax, "B")
    _panel_title(ax, "Health analysis workflow  (Note 6, sections 6.2 \u2013 6.6)")


# ══════════════════════════════════════════════════════════════════════════════
#  Panel C — How to read Figure 5
# ══════════════════════════════════════════════════════════════════════════════

_FIG5_ROWS = [
    dict(
        label   = "Fig. 5A\u2013C",
        fc      = _tint(_GRN, 0.50),
        title   = "Depth-corrected richness",
        desc    = "GMW-corrected richness across non-industrial,\nhealthy industrial, and disease-associated groups.",
        yaxis   = "Statistic:   depth-corrected richness",
        lookfor = "Look for:   the overall richness gradient by lifestyle and health.",
    ),
    dict(
        label   = "Fig. 5D",
        fc      = _tint(_SKY, 0.45),
        title   = "Condition-specific richness differences",
        desc    = "Study-stratified richness differences within\nindustrial cohorts, shown per disease.",
        yaxis   = "Statistic:   corrected richness difference  (GMW-CMH)",
        lookfor = "Look for:   which disease categories show stronger depletion.",
    ),
    dict(
        label   = "Fig. 5E",
        fc      = _tint(_PNK, 0.45),
        title   = "List-level deviation from matched background",
        desc    = "VANISH+ / BloSSUM+ taxa compared to a\nprevalence-matched background.",
        yaxis   = "Statistic:   energy distance",
        lookfor = "Look for:   whether focal-list taxa deviate beyond equally prevalent background taxa.",
    ),
]


def _draw_C(ax: plt.Axes, ph: float) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    row_h  = 0.305
    top_y  = 0.955
    chip_x = 0.085; chip_w = 0.130; chip_h = 0.072
    text_x = 0.175

    for k, row in enumerate(_FIG5_ROWS):
        top = top_y - k * row_h
        mid = top - row_h / 2

        _box(ax, chip_x, mid + 0.055, chip_w, chip_h,
             row["label"], row["fc"],
             lw=1.2, fs=FS["box_body"] + 0.5, bold=False, ph=ph)

        ax.text(text_x, mid + 0.095, row["title"],
                ha="left", va="center",
                fontsize=FS["box_title"], fontweight="bold", color=_TEXT,
                transform=ax.transAxes, clip_on=False)

        ax.text(text_x, mid + 0.025, row["desc"],
                ha="left", va="top",
                fontsize=FS["box_body"], color=_TEXT,
                transform=ax.transAxes, clip_on=False, linespacing=1.45)

        ax.text(text_x, mid - 0.078, row["yaxis"],
                ha="left", va="top",
                fontsize=FS["box_body"] - 1.0, color="#555555", style="italic",
                transform=ax.transAxes, clip_on=False)

        ax.text(text_x, mid - 0.118, row["lookfor"],
                ha="left", va="top",
                fontsize=FS["box_body"] - 0.5, color="#222222",
                transform=ax.transAxes, clip_on=False)

        if k < len(_FIG5_ROWS) - 1:
            rule_y = top - row_h + 0.010
            ax.plot([0.02, 0.98], [rule_y, rule_y],
                    color="#DDDDDD", lw=0.8,
                    transform=ax.transAxes, clip_on=False, zorder=1)

    _panel_letter(ax, "C")
    _panel_title(ax, "How to read Figure 5")


# ══════════════════════════════════════════════════════════════════════════════
#  Panel D — How to read Fig. S3
# ══════════════════════════════════════════════════════════════════════════════

_S3_TABLE = [
    (
        "Did prevalence\nmatching work?",
        "S6D",
        "Check OVL and post-weighting overlap.\nHigh OVL = well-matched background.",
    ),
    (
        "Stable across\nlist families?",
        "S6E",
        "Check sign concordance across original, +, and H+ lists.\nVANISH+ / BloSSUM+ are primary.",
    ),
    (
        "Original lists\nconcur?",
        "S6E\u2013G",
        "Check whether original VANISH / BloSSUM show\nthe same broad directional pattern.",
    ),
    (
        "Which taxa drive\nthe signal?",
        "S6F\u2013J",
        "Check which taxa recur as directional drivers\nacross conditions.",
    ),
]


def _draw_D(ax: plt.Axes, ph: float) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    col_q = (0.01,  0.300)
    col_p = (0.31,  0.110)
    col_c = (0.42,  0.570)

    hdr_y = 0.935; hdr_h = 0.070

    for x0, w, label in [(*col_q, "Question"), (*col_p, "Panel"),
                          (*col_c, "What to check")]:
        _box(ax, x0 + w / 2, hdr_y, w - 0.005, hdr_h,
             label, C["screen"],
             lw=0.9, fs=FS["tbl_hdr"], bold=True, ph=ph)

    row_h  = 0.195
    alt_fc = ["#FAFAFA", "#EFEFEF"]

    for i, (q, panel, check) in enumerate(_S3_TABLE):
        ry_ctr = hdr_y - hdr_h / 2 - 0.010 - row_h * i - row_h / 2

        ax.add_patch(FancyBboxPatch(
            (0.01, ry_ctr - row_h / 2 + 0.006), 0.98, row_h - 0.012,
            boxstyle="round,pad=0.004",
            facecolor=alt_fc[i % 2], edgecolor="#DDDDDD", linewidth=0.7,
            transform=ax.transAxes, clip_on=False, zorder=2,
        ))

        ax.text(col_q[0] + 0.015, ry_ctr, q,
                ha="left", va="center",
                fontsize=FS["tbl_body"], color=_TEXT,
                transform=ax.transAxes, clip_on=False, zorder=4,
                linespacing=1.4)

        _box(ax, col_p[0] + col_p[1] / 2, ry_ctr,
             col_p[1] - 0.008, row_h * 0.65,
             panel, _tint(_PNK, 0.40),
             lw=0.8, fs=FS["tbl_body"], bold=False, ph=ph)

        ax.text(col_c[0] + 0.010, ry_ctr, check,
                ha="left", va="center",
                fontsize=FS["tbl_body"] - 0.5, color=_TEXT,
                transform=ax.transAxes, clip_on=False, zorder=4,
                linespacing=1.4)

    _panel_letter(ax, "D")
    _panel_title(ax, "How to read Fig. S6B\u2013J")


# ══════════════════════════════════════════════════════════════════════════════
#  Footer — sign conventions
# ══════════════════════════════════════════════════════════════════════════════

_SIGNS = [
    ("positive prev_diff",
     "higher prevalence in controls  \u2192  depleted in disease"),
    ("negative prev_diff",
     "higher prevalence in cases  \u2192  enriched in disease"),
    ("ESS_frac",
     "effective sample size fraction after GMW weighting  (1.0 = no weighting cost)"),
    ("OVL",
     "overlap between focal-list and matched-background prevalence distributions  "
     "(diagnostic for matching quality)"),
]


def _draw_footer(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(FancyBboxPatch(
        (0.005, 0.04), 0.990, 0.920,
        boxstyle="round,pad=0.010",
        facecolor="white", edgecolor="#CCCCCC", linewidth=0.9,
        transform=ax.transAxes, clip_on=False, zorder=2,
    ))

    ax.text(0.020, 0.900, "Sign conventions and notation:",
            ha="left", va="top",
            fontsize=FS["footnote"] + 1.0, fontweight="bold", color=_TEXT,
            transform=ax.transAxes, clip_on=False)

    y_start = 0.720; y_step = 0.165    # 4 items, generous spacing
    x_term, x_sep, x_def = 0.025, 0.245, 0.262

    for i, (term, defn) in enumerate(_SIGNS):
        y = y_start - i * y_step
        ax.text(x_term, y, term, ha="left", va="center",
                fontsize=FS["footnote"], fontweight="bold", color=_TEXT,
                transform=ax.transAxes, clip_on=False)
        ax.text(x_sep,  y, "=",  ha="center", va="center",
                fontsize=FS["footnote"], color="#555555",
                transform=ax.transAxes, clip_on=False)
        ax.text(x_def,  y, defn, ha="left", va="center",
                fontsize=FS["footnote"], color="#333333",
                transform=ax.transAxes, clip_on=False)

    # Note: med_LD_diff not shown in figure body — see Table S6 for full diagnostics
    ax.text(0.020, 0.070,
            "med_LD_diff and full per-condition diagnostics are reported in Table S6.",
            ha="left", va="bottom",
            fontsize=FS["footnote"] - 1.5, color="#888888", style="italic",
            transform=ax.transAxes, clip_on=False)


# ══════════════════════════════════════════════════════════════════════════════
#  Main draw function
# ══════════════════════════════════════════════════════════════════════════════

def draw_guide(
    cohort_data: Optional[list[dict]],
    output_path: str = "figure_s_guide_fig5.pdf",
    fig_w: float = 32.0,
    fig_h: float = 30.0,
) -> None:
    """Render and save the four-panel guide figure (PDF + PNG at 180 dpi)."""

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    lm, rm     = 0.025, 0.025
    tm, bm     = 0.015, 0.012
    gap_h      = 0.020
    gap_v      = 0.070   # large gap so Panel A x-axis never touches Panel C title
    title_h    = 0.035
    footer_h   = 0.075

    pw           = (1 - lm - rm - gap_h) / 2
    available_h  = 1 - tm - bm - footer_h - gap_v - title_h - gap_v
    ph_top_frac  = available_h * 0.52
    ph_bot_frac  = available_h * 0.48

    y_top_lo = bm + footer_h + gap_v + ph_bot_frac + gap_v
    y_bot_lo = bm + footer_h + gap_v

    ax_A = fig.add_axes([lm,          y_top_lo, pw, ph_top_frac])
    ax_B = fig.add_axes([lm + pw + gap_h, y_top_lo, pw, ph_top_frac])
    ax_C = fig.add_axes([lm,          y_bot_lo, pw, ph_bot_frac])
    ax_D = fig.add_axes([lm + pw + gap_h, y_bot_lo, pw, ph_bot_frac])
    ax_F = fig.add_axes([lm, bm, 1 - lm - rm, footer_h])

    fig.text(0.50, 1 - tm * 0.35,
             "Figure S6.  Health analysis branch: how to read Figure 5 and Fig. S6B\u2013J",
             ha="center", va="top",
             fontsize=FS["fig_title"], fontweight="bold", color=_TEXT)

    ph_top = ph_top_frac * fig_h
    ph_bot = ph_bot_frac * fig_h

    _draw_A(ax_A, cohort_data)
    _draw_B(ax_B, ph_top)
    _draw_C(ax_C, ph_bot)
    _draw_D(ax_D, ph_bot)
    _draw_footer(ax_F)

    pdf_path = str(output_path)
    if not pdf_path.endswith(".pdf"):
        pdf_path += ".pdf"
    png_path = pdf_path.replace(".pdf", ".png")

    out_dir = os.path.dirname(os.path.abspath(pdf_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for path in (pdf_path, png_path):
        fig.savefig(path, dpi=180, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline entry point
# ══════════════════════════════════════════════════════════════════════════════

def run(ref_outdir: str, outdir: str) -> None:
    """
    Called from the main pipeline after Section 3 health analysis.

    Parameters
    ----------
    ref_outdir : str  path to reference output, e.g. ``results/UHGG``
    outdir     : str  where to save the figure (usually same as ref_outdir)
    """
    print("\n" + "=" * 60)
    print("  Figure S6: How to read Figure 5 / Fig. S3")
    print("=" * 60)
    cohort_data = load_cohort_data(ref_outdir)
    draw_guide(
        cohort_data,
        output_path=os.path.join(outdir, "figure_s_guide_fig5.pdf"),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone entry
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw Figure S6: How to read Figure 5 and Fig. S3."
    )
    parser.add_argument(
        "--ref", default="UHGG",
        help="Reference name (config.REFERENCES key), e.g. UHGG.  "
             "The script looks in results/<ref>/section3_health/ for pipeline outputs."
    )
    parser.add_argument(
        "--spec-tag", default=None,
        help="Detection-spec tag override, e.g. '10pct'.  Auto-detected if omitted."
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory for the figure files.  Default: results/<ref>/."
    )
    args = parser.parse_args()

    ref_outdir = os.path.join("results", args.ref)
    outdir     = args.out or ref_outdir

    cohort_data = load_cohort_data(ref_outdir, spec_tag=args.spec_tag)

    os.makedirs(outdir, exist_ok=True)
    draw_guide(
        cohort_data,
        output_path=os.path.join(outdir, "figure_s_guide_fig5.pdf"),
    )
