"""
figure_s0_roadmap.py  —  Figure S0: Analytic roadmap
=====================================================
Derives all cohort counts directly from the UHGG master matrix
(ReferenceData.meta) and draws a
structured flowchart.

Layout (top to bottom):
  External curation   — studies screened → eligibility → study exclusions
  Pipeline derivation — run consolidation → sample QC → primary cohort
                        secondary run branch (technical replicates only)
  Final analytic cohort  (anchor; "FINAL ANALYTIC COHORT")
  Classification layer   — lifestyle-classified subset + unclear/mixed side box
                           HG | Transitional | Industrial lifestyle boxes
                           Non-industrial combined | health boxes from Industrial
                           Industrial case-control eligible subset
  Analysis pipeline      — preprocessing → Carter/Leviatan/combined refs → matrices
                           GMW | hard detection + empirical accumulation | taxon lists
  Figure-specific outputs

Arrow grammar (strictly enforced):
  ─▶  solid    data flows and analytic dependencies (all main pipeline arrows)
  ──▷ solid    removal / exclusion arrows (same solid, exclusion-box colour)
  ╌╌▶ dashed   conceptual / "used-by" / "data-source-for" cross-links only

Colour scheme: Wong (2011) colour-blind-safe palette.
  Vermillion #D55E00 → H-G
  Yellow     #F0E442 → Transitional
  Orange     #E69F00 → Non-Industrial
  Sky blue   #56B4E9 → Industrial (all)
  Green      #009E73 → Ind-Healthy
  Blue       #0072B2 → Ind-Unhealthy
  Pink       #CC79A7 → Carter

Standalone usage (from data directory):
    python figure_s0_roadmap.py [--ref UHGG] [--out results/UHGG/]

Pipeline module usage:
    from persistence.figure_s0_roadmap import run
    run(data, outdir="results/UHGG")
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ══════════════════════════════════════════════════════════════════════════════
#  Colour palette  —  Wong (2011) CB-safe
# ══════════════════════════════════════════════════════════════════════════════

_VER = "#D55E00"   # Vermillion — H-G
_YEL = "#F0E442"   # Yellow     — Transitional
_ORA = "#E69F00"   # Orange     — Non-Industrial
_SKY = "#56B4E9"   # Sky blue   — Industrial (all)
_GRN = "#009E73"   # Green      — Ind-Healthy
_BLU = "#0072B2"   # Blue       — Ind-Unhealthy
_PNK = "#CC79A7"   # Pink       — Carter / Hadza / unlisted


def _tint(hex6: str, alpha: float = 0.25) -> str:
    """Blend a hex colour with white (0 = white, 1 = full saturation)."""
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r2 = int(r * alpha + 255 * (1 - alpha))
    g2 = int(g * alpha + 255 * (1 - alpha))
    b2 = int(b * alpha + 255 * (1 - alpha))
    return f"#{r2:02X}{g2:02X}{b2:02X}"


# Delegates to config, which holds the single list of display-time renames
# (PLOT_TEXT_REPLACEMENTS). This was a hard-coded local copy that knew only
# about Agro-Pastorialist/Transitional, so it silently missed every rename
# added since: Fatty Liver to MASLD, Obese to Obesity, and the lifestyle
# sentence casing. The fallback is for running this script standalone, with
# the package absent.
# The package root goes on the path before the import, not in __main__ after
# it. When this file is run by path, sys.path[0] is `figures/`, not the repo
# root, so `persistence` was unimportable at module-import time and the
# fallback below ran instead -- silently, and with rules that did not match
# config's: the stub mapped Transitional to the title-case "Agro-Pastoralist"
# and left Hunter-Gatherer alone. figure_runner sets PYTHONPATH on the
# subprocesses it launches, so the pipeline's own runs bound config and drew
# the right labels; the exposure was running this script any other way, which
# is what a reviewer cloning the repo does.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from persistence.config import display_text as _display_text
except Exception as _exc:                          # package genuinely absent
    # Deliberately not a second rename table, even a correct one: a local copy
    # drifts from config.PLOT_TEXT_REPLACEMENTS and then disagrees with the
    # rest of the figure set, which is worse than an obviously raw key.
    print(f"  [warn] persistence.config unavailable ({_exc}); figure text "
          f"will carry internal keys, not published labels", file=sys.stderr)

    def _display_text(text: str) -> str:
        return str(text)


def _apply_display_text(fig):
    for artist in fig.findobj(match=matplotlib.text.Text):
        old = artist.get_text()
        new = _display_text(old)
        if new != old:
            artist.set_text(new)
    return fig


C: dict[str, str] = dict(
    # Lifestyle — full primary hues
    hg        = _VER,
    trans     = _YEL,
    ind_life  = _SKY,
    # Health — full primary hues
    nonind    = _ORA,
    ind_h     = _GRN,
    ind_u     = _BLU,
    # Consistent dark grey — industrial samples with no health annotation
    # (in the cohort, but not usable as a case or a control)
    ind_unk   = "#888888",
    # References
    carter    = _PNK,
    lev       = _tint(_SKY, 0.55),
    combined  = _tint(_ORA, 0.45),
    # Main-flow boxes — brown gradient (light at top, darkest at final cohort)
    brown_screen   = "#D6C0A2",            # lightest tan — studies screened
    brown_elig     = "#C8AC85",            # light tan — eligibility
    brown_runcons  = "#BC9971",            # medium tan — run consolidation
    brown_primary  = "#AB855B",            # bronze — primary cohort
    brown_final    = "#5C3A1E",            # dark saddle — FINAL cohort
    brown_lsclass  = "#AB855B",            # bronze — lifestyle-classified
    # Side-track boxes still use neutral / tinted backgrounds
    screen    = "#F0F0F0",                  # neutral — kept for sidecar boxes
    exclusion = _tint(_VER, 0.28),         # light vermillion — exclusion boxes
    pipeline  = _tint(_SKY, 0.14),         # (unused now) — kept for compat
    final_bg  = _tint(_SKY, 0.30),         # (unused now) — kept for compat
    preproc   = _tint(_GRN, 0.22),         # light green — preprocessing (unused)
    matrix_bg = _tint(_SKY, 0.18),         # very light sky — matrix box
    method    = "#E8E8E8",                  # NEW: light grey — method modules
    figbox    = "#F4DE7B",                  # NEW: warm gold — figure cohort row
    si_box    = "#C0C0C0",                  # NEW: medium grey — SI placeholder row
    secondary = _tint(_PNK, 0.20),         # light pink — secondary run handling
    cc_bg     = _tint(_GRN, 0.16),         # very light green — case-control eligible
    # Edges / arrows
    edge        = "#3A3A3A",
    edge_excl   = "#8B2500",               # dark vermillion — exclusion box border
    edge_master = "#3A1A05",               # dark brown — final cohort border
    edge_sec    = "#8A4060",               # dark pink — secondary run border
    arrow       = "#1A1A1A",               # solid data-flow arrow
    arrow_excl  = "#8B2500",               # removal/exclusion arrow
    dash        = "#909090",               # dashed cross-reference arrow
)

_TEXT = "#111111"


# ══════════════════════════════════════════════════════════════════════════════
#  Font sizes
# ══════════════════════════════════════════════════════════════════════════════

FS: dict[str, float] = dict(
    fig_title   = 44.0,
    box_title   = 36.0,   # bold first line
    box_body    = 30.0,   # body lines
    section_lbl = 36.0,   # left-margin section labels
    band_label  = 26.0,   # right-margin band labels
    footnote    = 15.0,   # unused
)


# ══════════════════════════════════════════════════════════════════════════════
#  Secondary-run stats  (tries secondary_bc_screening.csv; falls back to "?")
# ══════════════════════════════════════════════════════════════════════════════

def _secondary_stats(data: Any) -> dict:
    """Secondary-run counts, taken from sec_map on the ReferenceData itself.

    Previously read from secondary_bc_screening.csv, which the reinstatement
    step wrote. That step no longer runs (config.REINSTATE_SECONDARIES is
    False), so the file does not exist and every count rendered as "?".
    sec_map is present either way and is the same population, so the figure now
    derives the numbers directly and cannot silently fall back to placeholders.
    """
    sec_map = getattr(data, "sec_map", None) or {}
    return dict(n_sec_runs=sum(len(v) for v in sec_map.values()),
                n_sec_primaries=len(sec_map))


# ══════════════════════════════════════════════════════════════════════════════
#  Count derivation from ReferenceData
# ══════════════════════════════════════════════════════════════════════════════

def derive_counts(data: Any, ref_outdir: str | None = None) -> dict:
    """
    Extract all figure-relevant counts from a ReferenceData.

    Parameters
    ----------
    data       : ReferenceData namedtuple
    ref_outdir : path to the reference output directory (e.g. "results/UHGG");
                 used to locate secondary_bc_screening.csv.  Pass None to skip.

    Returns
    -------
    dict of int | str.  Unrecoverable pre-filter counts remain as "?".
    """
    meta   = data.meta
    counts: dict[str, Any] = {}

    # ── Final analytic cohort ─────────────────────────────────────────────────
    counts["n_total"]     = len(meta)
    counts["n_studies"]   = (int(meta["_study"].nunique())
                             if "_study" in meta.columns else "?")
    counts["n_countries"] = (int(meta["Country_norm"].nunique())
                             if "Country_norm" in meta.columns else "?")

    # ── Secondary runs (retained as technical replicates, never reinstated) ───
    counts.update(_secondary_stats(data))

    # ── Lifestyle split ────────────────────────────────────────────────────────
    life = meta.get("Lifestyle")
    if life is not None:
        counts["n_hg"]    = int((life == "H-G").sum())
        counts["n_trans"] = int((life == "Transitional").sum())
        counts["n_ind"]   = int((life == "Industrial").sum())
        n_classified      = counts["n_hg"] + counts["n_trans"] + counts["n_ind"]
        counts["n_lifestyle_classified"] = n_classified
        counts["n_lifestyle_unclear"]    = len(meta) - n_classified
        counts["n_nonind"] = counts["n_hg"] + counts["n_trans"]
    else:
        counts.update(n_hg="?", n_trans="?", n_ind="?",
                      n_lifestyle_classified="?", n_lifestyle_unclear="?",
                      n_nonind="?")

    # ── Health split — descends from Industrial only ───────────────────────────
    if life is not None and "Unhealthy" in meta.columns:
        uh     = meta["Unhealthy"].astype(str)
        is_ind = life == "Industrial"
        counts["n_ind_h"]   = int((is_ind & (uh == "Healthy")).sum())
        counts["n_ind_u"]   = int((is_ind & (uh == "Unhealthy")).sum())
        counts["n_ind_unk"] = int((is_ind & ~uh.isin(["Healthy", "Unhealthy"])).sum())
    else:
        counts.update(n_ind_h="?", n_ind_u="?", n_ind_unk="?")

    counts.update(_host_counts(meta))
    counts.update(_cascade_counts(getattr(data, "ref_name", None), ref_outdir))
    counts.update(_static_counts())
    return counts


def _static_counts() -> dict:
    return dict(n_sgbs_carter=3_971, n_sgbs_lev=3_594, n_sgbs_combined=5_081)


def _host_counts(meta) -> dict:
    """Hosts (distinct Subject_ID) beside every sample count on the figure.

    Counted exactly as cohort_manifest.write_cohort_manifest counts them for
    Table S2: a sample with no archived host identifier carries its own
    Subject_ID, so it is one host and n_h is an upper bound on people.
    """
    if "Subject_ID" not in meta.columns or "Lifestyle" not in meta.columns:
        return {}
    subj = meta["Subject_ID"].astype(str)
    life = meta["Lifestyle"].astype(str)
    out = dict(h_total=int(subj.nunique()),
               h_hg=int(subj[life == "H-G"].nunique()),
               h_trans=int(subj[life == "Transitional"].nunique()),
               h_ind=int(subj[life == "Industrial"].nunique()),
               h_nonind=int(subj[life != "Industrial"].nunique()))
    if "Unhealthy" in meta.columns:
        uh = meta["Unhealthy"].astype(str)
        is_ind = life == "Industrial"
        out["h_ind_h"] = int(subj[is_ind & (uh == "Healthy")].nunique())
        out["h_ind_u"] = int(subj[is_ind & (uh == "Unhealthy")].nunique())
        out["h_ind_unk"] = int(subj[is_ind & ~uh.isin(["Healthy", "Unhealthy"])].nunique())
    return out


# Cohort-cascade counts drawn in the exclusions box, keyed as the loader
# records them (loading.LAST_LOAD_COUNTS / stats tags "cohort_<key>").
_CASCADE_KEYS = ("matched_to_matrix", "dropped_unclassifiable_lifestyle",
                 "dropped_excluded_samples", "dropped_duplicate_biosample",
                 "dropped_below_min_bp", "dropped_below_min_sgbs")


def _cascade_counts(ref_name: str | None, ref_outdir: str | None) -> dict:
    """Exclusion counts, from the loader's in-process tally or else the stats
    registry the run wrote (results/<ref>/stats_report.csv)."""
    out: dict[str, Any] = {}
    try:
        from persistence.loading import LAST_LOAD_COUNTS
        out.update(LAST_LOAD_COUNTS.get(ref_name) or {})
    except Exception:
        pass
    reg = _read_registry(ref_outdir)
    for k in _CASCADE_KEYS:
        if k not in out and f"cohort_{k}" in reg:
            out[k] = reg[f"cohort_{k}"]
    return {k: _as_int(v) for k, v in out.items() if k in _CASCADE_KEYS}


def _as_int(v: Any) -> Any:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return "?"


def _read_registry(ref_outdir: str | None, column: str = "value") -> dict:
    if not ref_outdir:
        return {}
    path = os.path.join(ref_outdir, "stats_report.csv")
    if not os.path.exists(path):
        return {}
    import pandas as pd
    reg = pd.read_csv(path, usecols=["tag", column], low_memory=False)
    # A tag can repeat (a stage rerun appends); the last write is current.
    return reg.drop_duplicates("tag", keep="last").set_index("tag")[column].to_dict()


def counts_from_results(ref_outdir: str, m1_path: str) -> dict:
    """Every count on the roadmap, from a finished run's files alone.

    Uses results/<ref>/cohort_samples.csv (the analytic cohort), the stats
    registry and Metadata M1, so the figure can be redrawn without loading the
    alignment matrices. Secondary-run counts come from the registry; the
    number of samples they belong to is the technical-replicate ICC's
    n_primary (the same sec_map population).
    """
    import pandas as pd
    reg = _read_registry(ref_outdir)
    reg_n1 = _read_registry(ref_outdir, "n1")
    cohort = pd.read_csv(os.path.join(ref_outdir, "cohort_samples.csv"))
    m1 = pd.read_csv(m1_path, low_memory=False,
                     usecols=lambda c: c in ("Chosen_match_ID", "Subject_ID", "Unhealthy"))
    meta = cohort.merge(m1, on="Chosen_match_ID", how="left")
    life = meta["Lifestyle"]
    uh = meta["Unhealthy"].astype(str)
    is_ind = life == "Industrial"

    def _i(tag):
        return _as_int(reg.get(tag))

    counts: dict[str, Any] = dict(
        n_sec_primaries=_as_int(reg_n1.get("tech_rep_ICC_richness_all_10pct")),
        n_total=len(meta), n_studies=_i("n_studies"), n_countries=_i("n_countries"),
        n_hg=int((life == "H-G").sum()), n_trans=int((life == "Transitional").sum()),
        n_ind=int(is_ind.sum()),
        n_ind_h=int((is_ind & (uh == "Healthy")).sum()),
        n_ind_u=int((is_ind & (uh == "Unhealthy")).sum()),
        n_ind_unk=int((is_ind & ~uh.isin(["Healthy", "Unhealthy"])).sum()),
        n_sec_runs=_i("reinstate_disabled"),
    )
    counts["n_nonind"] = counts["n_hg"] + counts["n_trans"]
    counts["n_lifestyle_classified"] = counts["n_total"]
    counts["n_lifestyle_unclear"] = 0
    counts.update(_host_counts(meta))
    counts.update(_cascade_counts(None, ref_outdir))
    counts.update(_static_counts())
    return counts


# ══════════════════════════════════════════════════════════════════════════════
#  Drawing canvas
# ══════════════════════════════════════════════════════════════════════════════

class _Canvas:
    """Axes-fraction drawing helpers."""

    def __init__(self, ax: plt.Axes, fig_h_inches: float):
        self.ax    = ax
        self.fig_h = fig_h_inches

    # ── Box ───────────────────────────────────────────────────────────────────

    def box(
        self, name: str,
        cx: float, cy: float, w: float, h: float,
        text: str, fc: str,
        ec: str | None = None,
        lw: float = 1.4,
        fontsize: float | None = None,
        title_fontsize: float | None = None,
        bold_title: bool = True,
        text_color: str | None = None,
        text_dx: float = 0.0,
        text_dy: float = 0.0,
        box_pad: float = 0.003,
        bold_all: bool = False,
    ) -> None:
        ec      = ec or C["edge"]
        fs_body = fontsize if fontsize is not None else FS["box_body"]
        if title_fontsize is not None:
            fs_ttl = title_fontsize
        else:
            fs_ttl  = FS["box_title"] if bold_title else fs_body
        tc      = text_color if text_color is not None else _TEXT
        ax      = self.ax

        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle=f"round,pad={box_pad}",
            facecolor=fc, edgecolor=ec, linewidth=lw,
            transform=ax.transAxes, clip_on=False, zorder=3,
        ))

        lines = text.split("\n")
        n     = max(len(lines), 1)
        pt_in_ax = 1.0 / (72.0 * self.fig_h)
        # Spacing must accommodate whichever line is largest (usually the title)
        max_fs   = max(fs_ttl, fs_body)
        spacing  = min(max_fs * pt_in_ax * 1.45, h * 0.92 / n)
        x_text   = cx + text_dx
        y_top    = cy + text_dy + spacing * (n - 1) / 2

        for i, line in enumerate(lines):
            is_title = (i == 0)
            ax.text(
                x_text, y_top - i * spacing, line,
                ha="center", va="center",
                fontsize=fs_ttl if (bold_title and is_title) else fs_body,
                fontweight="bold" if (bold_all or (bold_title and is_title)) else "normal",
                color=tc,
                transform=ax.transAxes, clip_on=False, zorder=4,
            )

    # ── Arrows ────────────────────────────────────────────────────────────────

    def arrow(
        self,
        x0: float, y0: float,
        x1: float, y1: float,
        dashed: bool = False,
        color: str | None = None,
        lw: float = 2.5,
    ) -> None:
        color = color or (C["dash"] if dashed else C["arrow"])
        self.ax.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=lw,
                linestyle=(0, (5, 3)) if dashed else "solid",
                mutation_scale=22,
            ),
            zorder=5,
        )

    def line(
        self,
        x0: float, y0: float,
        x1: float, y1: float,
        dashed: bool = False,
        color: str | None = None,
        lw: float = 2.4,
    ) -> None:
        """Line segment with no arrowhead (for elbow first legs)."""
        color = color or (C["dash"] if dashed else C["arrow"])
        self.ax.plot(
            [x0, x1], [y0, y1],
            color=color, lw=lw,
            ls=(0, (5, 3)) if dashed else "-",
            transform=self.ax.transAxes, clip_on=False, zorder=5,
        )

    def elbow(
        self,
        x0: float, y_bend: float,
        x1: float, y1: float,
        dashed: bool = False,
        color: str | None = None,
        lw: float = 2.2,
    ) -> None:
        """Horizontal leg then vertical arrow (right-angle elbow)."""
        color = color or (C["dash"] if dashed else C["arrow"])
        self.line(x0, y_bend, x1, y_bend, dashed=dashed, color=color, lw=lw)
        self.arrow(x1, y_bend, x1, y1,   dashed=dashed, color=color, lw=lw)

    # ── Decorations ───────────────────────────────────────────────────────────

    def section_label(self, x: float, y: float, text: str) -> None:
        """Bold italic label for a section boundary (left margin)."""
        self.ax.text(
            x, y, text,
            ha="center", va="center",
            fontsize=FS["section_lbl"], color="#111111",
            style="normal", fontweight="bold",
            transform=self.ax.transAxes, clip_on=False, zorder=2,
        )

    def band_label(self, y: float, text: str) -> None:
        self.ax.text(
            0.970, y, text,
            ha="right", va="center",
            fontsize=FS["band_label"], color="#BBBBBB", style="italic",
            transform=self.ax.transAxes, clip_on=False,
        )

    def band_rule(self, y: float, dashed: bool = True) -> None:
        self.ax.plot(
            [0.01, 0.99], [y, y],
            color="#D0D0D0", lw=0.8, ls="--" if dashed else "-",
            transform=self.ax.transAxes, clip_on=False, zorder=1,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Main drawing function
# ══════════════════════════════════════════════════════════════════════════════

def draw_roadmap(
    counts: dict,
    output_path: str = "figure_s0_roadmap.pdf",
    fig_w: float = 46.0,
    fig_h: float = 33.0,
) -> None:
    """Render and save the analytic roadmap (PDF + PNG)."""

    # Unpack key counts
    n   = counts["n_total"]
    ns  = counts["n_studies"]
    nc  = counts["n_countries"]
    # With reinstatement off nothing is added after the primary cohort, so
    # the two are the same number by construction.
    npr = counts.get("n_primary", counts["n_total"])
    # Samples whose Lifestyle resolves to neither Industrial nor Non-Industrial
    # are now dropped at load (config.DROP_UNCLASSIFIED_LIFESTYLE), so the cohort
    # size and the lifestyle tallies agree by construction. This stays as a guard:
    # a non-zero count means the loader let something unclassifiable through.
    n_unclear = counts.get("n_lifestyle_unclear", 0)
    if isinstance(n_unclear, int) and n_unclear:
        print(f"  [WARN] {n_unclear:,} samples have an unclassifiable Lifestyle; "
              f"cohort size and lifestyle tallies will not agree.")
    n_tallied, npr_tallied = n, npr

    def _n(k: str) -> str:
        v = counts.get(k, "?")
        return f"{v:,}" if isinstance(v, int) else str(v)

    def _fmt(v: Any) -> str:
        return f"{v:,}" if isinstance(v, int) else str(v)

    # ── Figure & axes ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax  = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    D = _Canvas(ax, fig_h_inches=fig_h)

    # Left-margin section markers
    D.section_label(0.075, 0.962, "STUDY SELECTION\n(SEMI-AUTOMATED)")
    D.section_label(0.075, 0.905, "PREPROCESSING\nPIPELINE")
    D.section_label(0.075, 0.843, "AUTOMATED METADATA\nCURATION")
    D.section_label(0.075, 0.625, "CLASSIFICATION")
    D.section_label(0.075, 0.315, "ANALYSIS\nPIPELINE")

    _ANN_FS = FS["box_body"] - 1.0

    def _badge(cx: float, cy: float, text: str, w: float = 0.082,
               h: float = 0.020) -> None:
        """Grey pointer to the Supplement (Supp. Text / Table / Fig. S)."""
        D.box(f"nb_{cx:.3f}_{cy:.3f}", cx=cx, cy=cy, w=w, h=h, text=text,
              fc="#E8E8E8", ec="#888888", lw=0.7,
              fontsize=_ANN_FS, title_fontsize=_ANN_FS, bold_title=True,
              box_pad=0.001, bold_all=True)

    # ════════════════════════════════════════════════════════════════════════════
    #  STUDY SELECTION  →  PREPROCESSING  →  M1 CURATION
    # ════════════════════════════════════════════════════════════════════════════

    D.box("screen", cx=0.420, cy=0.962, w=0.490, h=0.036,
          text=("Human gut metagenome studies screened\n"
                "NCBI SRA + ENA  ·  multi-source seeding + manual search"),
          fc=C["brown_screen"])
    D.arrow(0.420, 0.944, 0.420, 0.930)

    D.box("elig", cx=0.420, cy=0.912, w=0.490, h=0.036,
          text=("Preprocessing pipeline and eligibility criteria\n"
                "Human gut · shotgun · paired-end · ≥100 Mb raw · metadata available"),
          fc=C["brown_elig"])

    D.box("study_excl", cx=0.865, cy=0.935, w=0.240, h=0.058,
          text=("Study-level exclusions\n"
                "non-gut or non-shotgun · infant cohorts\n"
                "antibiotic-treated · depth / batch outliers"),
          fc=C["exclusion"], ec=C["edge_excl"], lw=1.6,
          fontsize=FS["box_body"] - 4.0, title_fontsize=FS["box_title"] - 2.0)
    D.arrow(0.665, 0.916, 0.745, 0.926, color=C["arrow_excl"])
    _badge(0.705, 0.934, "Supp. Text 1B", w=0.105)
    _badge(0.705, 0.902, "Supp. Text 2", w=0.098)

    D.arrow(0.420, 0.894, 0.420, 0.882)

    D.box("m1", cx=0.420, cy=0.862, w=0.490, h=0.040,
          text=("Curated metadata  (Data S2)\n"
                f"{_n('matched_to_matrix')} samples matched to the alignment matrices"),
          fc=C["brown_runcons"])

    # Sample-level exclusions: the cohort cascade with its counts (Table S2).
    _SX0, _SX1 = 0.745, 0.985
    _SY_TOP, _SY_BOT = 0.892, 0.774
    D.box("sample_excl", cx=(_SX0 + _SX1) / 2, cy=(_SY_TOP + _SY_BOT) / 2,
          w=_SX1 - _SX0, h=_SY_TOP - _SY_BOT, text="",
          fc=C["exclusion"], ec=C["edge_excl"], lw=1.6)
    _drops = [
        ("lifestyle not classifiable",         "dropped_unclassifiable_lifestyle"),
        ("amplicon (16S) libraries",           "dropped_excluded_samples"),
        ("second library of a kept BioSample", "dropped_duplicate_biosample"),
        ("< 30 Mb mapped bases",               "dropped_below_min_bp"),
        ("< 5 SGBs at ≥10% breadth",           "dropped_below_min_sgbs"),
    ]
    _tot = [counts.get(k) for _, k in _drops]
    _tot = sum(_tot) if all(isinstance(v, int) for v in _tot) else None
    ax.text((_SX0 + _SX1) / 2, _SY_TOP - 0.013,
            "Sample-level exclusions" + (f"  (−{_tot:,})" if _tot else ""),
            ha="center", va="center", fontsize=FS["box_title"] - 2.0,
            fontweight="bold", color=_TEXT,
            transform=ax.transAxes, clip_on=False, zorder=4)
    _row_y = _SY_TOP - 0.033
    for label, key in _drops:
        ax.text(_SX0 + 0.010, _row_y, label, ha="left", va="center",
                fontsize=FS["box_body"] - 4.0, color=_TEXT,
                transform=ax.transAxes, clip_on=False, zorder=4)
        ax.text(_SX1 - 0.010, _row_y, _n(key), ha="right", va="center",
                fontsize=FS["box_body"] - 4.0, color=_TEXT, fontweight="bold",
                transform=ax.transAxes, clip_on=False, zorder=4)
        _row_y -= 0.0152
    ax.text((_SX0 + _SX1) / 2, _row_y - 0.001,
            "infants and non-gut samples were removed during metadata curation",
            ha="center", va="center", fontsize=FS["box_body"] - 8.0,
            color="#444444", style="italic",
            transform=ax.transAxes, clip_on=False, zorder=4)
    D.arrow(0.665, 0.862, 0.745, 0.862, color=C["arrow_excl"])
    _badge(0.705, 0.875, "Table S2", w=0.070)
    _badge(0.705, 0.849, "Fig. S4", w=0.060)

    # ════════════════════════════════════════════════════════════════════════════
    #  RUNS  →  SAMPLES (= BioSamples)  →  HOSTS
    # ════════════════════════════════════════════════════════════════════════════
    # The three units the cohort is counted in, and what the analysis does with
    # each: several runs per sample (one analysed, the rest technical
    # replicates); exactly one sample per BioSample; several samples per host
    # (all kept for lifestyle analyses, one per host for health analyses).

    _UY, _UH, _UW = 0.770, 0.090, 0.185
    _UX = [0.215, 0.420, 0.625]
    D.arrow(0.420, 0.842, 0.420, _UY + _UH / 2)
    _unit_fs = dict(fontsize=FS["box_body"] - 5.0, title_fontsize=FS["box_title"] - 6.0)
    D.box("u_runs", cx=_UX[0], cy=_UY, w=_UW, h=_UH,
          text=("Several runs per sample\n"
                "the primary run is analysed;\n"
                f"{_n('n_sec_runs')} further runs of {_n('n_sec_primaries')} samples\n"
                "are kept only as technical replicates\n"
                "(Figs. S2–S3; Supp. Text 1C)"),
          fc=C["secondary"], ec=C["edge_sec"], lw=1.5, **_unit_fs)
    D.box("u_biosample", cx=_UX[1], cy=_UY, w=_UW, h=_UH,
          text=("One sample per BioSample\n"
                f"{_n('dropped_duplicate_biosample')} further libraries of an\n"
                "already-kept BioSample become\n"
                "technical replicates, not samples\n"
                "(Supp. Text 1C; Table S2)"),
          fc=C["brown_runcons"], **_unit_fs)
    D.box("u_hosts", cx=_UX[2], cy=_UY, w=_UW, h=_UH,
          text=("Several samples per host\n"
                f"{_fmt(n)} samples from {_n('h_total')} hosts;\n"
                "repeat samples kept for lifestyle\n"
                "analyses, one sample per host for\n"
                "health analyses (Supp. Text 6B, 7C)"),
          fc=C["brown_primary"], **_unit_fs)
    for x0, x1 in ((_UX[0] + _UW / 2, _UX[1] - _UW / 2),
                   (_UX[1] + _UW / 2, _UX[2] - _UW / 2)):
        D.arrow(x0 + 0.003, _UY, x1 - 0.003, _UY, lw=2.0)

    # The BioSample and host rules both shape the cohort; the extra runs of the
    # left box do not enter it (technical replicates only), so no arrow there.
    D.arrow(_UX[1], _UY - _UH / 2, _UX[1], 0.704)
    D.arrow(_UX[2], _UY - _UH / 2, _UX[2], 0.704)

    # ── FINAL ANALYTIC COHORT ──────────────────────────────────────────────────
    D.box("final", cx=0.420, cy=0.672, w=0.620, h=0.060,
          text=(f"FINAL ANALYTIC COHORT\n"
                f"N = {_fmt(n_tallied)} samples  ·  {_n('h_total')} hosts  ·  "
                f"{ns} studies  ·  {nc} countries"),
          fc=C["brown_final"], ec=C["edge_master"], lw=2.4,
          fontsize=FS["box_body"] + 0.5,
          text_color="#FFFFFF")

    # ════════════════════════════════════════════════════════════════════════════
    #  CLASSIFICATION LAYER
    # ════════════════════════════════════════════════════════════════════════════

    # Final analytic cohort → 3-way fork to Hunter-Gatherer / Transitional / Industrial
    # Single descender from Final, then horizontal distributor, then 3 vertical arrows
    D.line( 0.420, 0.642, 0.420, 0.616, color=C["arrow"])    # descender from FINAL bot
    D.line( 0.112, 0.616, 0.540, 0.616, color=C["arrow"])    # horizontal distributor
    D.arrow(0.112, 0.616, 0.112, 0.606, color=C["arrow"])    # → HG
    D.arrow(0.262, 0.616, 0.262, 0.606, color=C["arrow"])    # → Trans
    D.arrow(0.540, 0.616, 0.540, 0.606, color=C["arrow"])    # → Industrial

    D.box("hg",    cx=0.112, cy=0.577, w=0.130, h=0.058,
          text=f"Hunter-Gatherer\n{_n('n_hg')} samples\n{_n('h_hg')} hosts", fc=C["hg"])

    D.box("trans", cx=0.262, cy=0.577, w=0.144, h=0.058,
          text=f"Transitional\n{_n('n_trans')} samples\n{_n('h_trans')} hosts", fc=C["trans"])

    D.box("ind",   cx=0.540, cy=0.577, w=0.305, h=0.058,
          text=f"Industrial\n{_n('n_ind')} samples  ·  {_n('h_ind')} hosts",   fc=C["ind_life"])

    # Non-industrial combined (derived from HG + Transitional, for comparisons)
    D.line( 0.112, 0.548, 0.112, 0.533, color=C["arrow"])
    D.line( 0.262, 0.548, 0.262, 0.533, color=C["arrow"])
    D.line( 0.112, 0.533, 0.262, 0.533, color=C["arrow"])
    D.arrow(0.187, 0.533, 0.187, 0.443, color=C["arrow"])

    D.box("nonind", cx=0.187, cy=0.417, w=0.200, h=0.052,
          text=f"Non-industrial  (combined)\n{_n('n_nonind')} samples  ·  {_n('h_nonind')} hosts",
          fc=C["nonind"])

    # Health split descends from Industrial only
    D.line(0.540, 0.548, 0.540, 0.522, color=C["arrow"])

    ax.text(0.556, 0.535, "health metadata",
            ha="left", va="center", fontsize=FS["box_body"],
            color="#555555", style="italic",
            transform=ax.transAxes, clip_on=False, zorder=4)

    # Equal-thirds subdivision of Industrial (spans 0.3875–0.6925, w=0.305)
    # Centers at 0.438, 0.540, 0.642; each width 0.100
    D.line(0.438, 0.522, 0.642, 0.522, color=C["arrow"])
    D.arrow(0.438, 0.522, 0.438, 0.513, color=C["arrow"])   # → Ind Healthy
    D.arrow(0.540, 0.522, 0.540, 0.513, color=C["arrow"])   # → Ind Unhealthy
    D.arrow(0.642, 0.522, 0.642, 0.513, color=C["arrow"])   # → Unknown

    D.box("ind_h",   cx=0.438, cy=0.486, w=0.100, h=0.052,
          text=f"Ind. healthy\n{_n('n_ind_h')} samples\n{_n('h_ind_h')} hosts", fc=C["ind_h"],
          fontsize=FS["box_body"] - 2.0)

    D.box("ind_u",   cx=0.540, cy=0.486, w=0.100, h=0.052,
          text=f"Ind. unhealthy\n{_n('n_ind_u')} samples\n{_n('h_ind_u')} hosts", fc=C["ind_u"],
          fontsize=FS["box_body"] - 2.0)

    D.box("ind_unk", cx=0.642, cy=0.486, w=0.100, h=0.052,
          text=f"Unknown\n{_n('n_ind_unk')} samples\n{_n('h_ind_unk')} hosts", fc=C["ind_unk"],
          fontsize=FS["box_body"] - 2.0)

    # Industrial case-control eligible subset
    # Merge only from Ind. healthy + Ind. unhealthy — Unknown is excluded
    D.line( 0.438, 0.460, 0.438, 0.452, color=C["arrow"])
    D.line( 0.540, 0.460, 0.540, 0.452, color=C["arrow"])
    D.line( 0.438, 0.452, 0.540, 0.452, color=C["arrow"])
    D.arrow(0.489, 0.452, 0.533, 0.443, color=C["arrow"])

    D.box("cc", cx=0.533, cy=0.417, w=0.290, h=0.050,
          text=("Industrial case-control eligible subset\n"
                "one sample per host  ·  ≥5 cases & ≥5 controls per study"),
          fc=C["cc_bg"], ec=C["edge"], lw=1.2,
          fontsize=FS["box_body"] - 2.0)

    # ════════════════════════════════════════════════════════════════════════════
    #  ANALYSIS PIPELINE
    # ════════════════════════════════════════════════════════════════════════════

    # Cohort/reference dependencies: both comparison cohorts are carried through
    # Carter/UHGG and Leviatan, then the two catalogues are merged as a third
    # combined reference before matrix construction.
    D.arrow(0.150, 0.391, 0.215, 0.373, color=C["arrow"], lw=1.7)
    D.arrow(0.224, 0.391, 0.600, 0.373, color=C["arrow"], lw=1.7)
    D.arrow(0.496, 0.392, 0.265, 0.373, color=C["arrow"], lw=1.7)
    D.arrow(0.570, 0.392, 0.705, 0.373, color=C["arrow"], lw=1.7)

    # Reference databases (preprocessing folded into descriptor)
    D.box("carter", cx=0.240, cy=0.349, w=0.215, h=0.048,
          text=(f"\nnon-industrial-biased  ·  {counts['n_sgbs_carter']:,} SGBs"),
          fc=C["carter"],
          bold_title=False)

    D.box("lev", cx=0.650, cy=0.349, w=0.215, h=0.048,
          text=(f"\nindustrial-biased  ·  {counts['n_sgbs_lev']:,} SGBs"),
          fc=C["lev"],
          bold_title=False)

    D.box("combined_ref", cx=0.445, cy=0.292, w=0.272, h=0.044,
          text=(f"Combined reference\nCarter + Leviatan  ·  {counts['n_sgbs_combined']:,} SGBs"),
          fc=C["combined"],
          fontsize=FS["box_body"] - 2.0,
          title_fontsize=FS["box_title"] - 1.0)

    _META_FS = _ANN_FS
    # Data S3 documents all three catalogues (Carter, Leviatan and their
    # merge), so it is badged once on the combined-reference box rather than
    # per catalogue. Mirrors the Table S4 badge on that box's left edge.
    D.box("nb_ref_meta", cx=0.590, cy=0.292, w=0.075, h=0.022,
          text="Data S3",
          fc="#FFFFFF", ec="#888888", lw=0.7,
          fontsize=_META_FS, title_fontsize=_META_FS, bold_title=True,
          box_pad=0.001)

    ax.text(0.200, 0.360, "Carter reference",
            ha="center", va="center",
            fontsize=FS["box_title"] - 2.0, fontweight="bold", color=_TEXT,
            transform=ax.transAxes, clip_on=False, zorder=5)
    ax.text(0.610, 0.360, "Leviatan reference",
            ha="center", va="center",
            fontsize=FS["box_title"] - 2.0, fontweight="bold", color=_TEXT,
            transform=ax.transAxes, clip_on=False, zorder=5)

    ax.text(0.445, 0.365,
            "uniform preprocessing\nhigh-MAPQ alignment",
            ha="center", va="center", fontsize=FS["box_body"] - 2.0,
            color="#555555", style="italic",
            transform=ax.transAxes, clip_on=False, zorder=4)

    # Light grey note badge near the reference-merge step.
    D.box("nb_preproc", cx=0.300, cy=0.292, w=0.065, h=0.028,
          text="Table S4\nSupp. Text 2D",
          fc="#E8E8E8", ec="#888888", lw=0.7,
          fontsize=_META_FS, title_fontsize=_META_FS,
          bold_title=True,
          box_pad=0.001,
          bold_all=True)

    # Merge arrows into the combined reference.
    D.arrow(0.330, 0.325, 0.390, 0.314)
    D.arrow(0.560, 0.325, 0.505, 0.314)

    # Direct source-reference arrows bypass the combined catalogue because
    # Carter/UHGG and Leviatan also feed downstream analyses independently.
    D.line(0.240, 0.325, 0.240, 0.268, color=C["arrow"], lw=1.7)
    D.arrow(0.240, 0.268, 0.325, 0.264, color=C["arrow"], lw=1.7)
    D.line(0.650, 0.325, 0.650, 0.268, color=C["arrow"], lw=1.7)
    D.arrow(0.650, 0.268, 0.565, 0.264, color=C["arrow"], lw=1.7)

    D.arrow(0.445, 0.270, 0.445, 0.260)

    D.box("matrices", cx=0.445, cy=0.242, w=0.390, h=0.036,
          text=("Per-sample breadth & depth matrices\n"
                "every retained sample × SGB  (three references)"),
          fc=C["matrix_bg"])

    # ── Four method modules ───────────────────────────────────────────────────
    # Spread evenly across the wider canvas; Health framework sits far right.
    # Shared column grids for the final three rows.  Each row uses the same
    # outer edges and the same inter-box gutter; widths are content-aware.
    _GRID_L = 0.0175
    _GRID_G = 0.0060
    _GRID_W = [0.2200, 0.1450, 0.2000, 0.1885, 0.1900]
    _GRID_X = []
    _x = _GRID_L
    for _w in _GRID_W:
        _GRID_X.append(_x + _w / 2)
        _x += _w + _GRID_G

    # Widths are content-aware, and the last two are set against their body
    # lines rather than split evenly. The health module's body line
    # ("Tables S10-S12 - within-study GMW-CMH + prevalence bg.") measures
    # 0.280 of the canvas; in a 0.246 box it hung 0.017 past each edge, so its
    # left end sat inside the taxon-list box and its right end ran off the axes
    # at 1.005. The taxon-list module's body is a 0.080 badge and a short bold
    # reference, 0.188 together, so it is the one with width to give. Moving
    # 0.049 from the third box to the fourth leaves both bodies clear by 0.012
    # and 0.007 and keeps the band's outer edges where they were.
    _METHOD_W = [0.2350, 0.2200, 0.2060, 0.2885]
    _METHOD_X = []
    _x = _GRID_L
    for _w in _METHOD_W:
        _METHOD_X.append(_x + _w / 2)
        _x += _w + _GRID_G

    _MX = _METHOD_X
    _MW = _METHOD_W
    _MY = 0.168    # method modules y-center
    _MH = 0.076
    _MTITLE_Y = _MY + 0.014
    _MBODY_Y  = _MY - 0.014

    # Arrows from matrices down to each module top.
    D.arrow(0.285, 0.224, _MX[0] + 0.012, _MY + _MH / 2)    # → GMW
    D.arrow(_MX[1], 0.224, _MX[1],         _MY + _MH / 2)   # → hard/empirical
    D.arrow(0.545, 0.224, _MX[2],          _MY + _MH / 2)   # → Taxon
    D.elbow(0.640, 0.242, _MX[3] - _MW[3]/2, _MY + _MH / 2)  # → Health fw.

    D.box("gmw", cx=_MX[0], cy=_MY, w=_MW[0], h=_MH,
          text="", fc=C["method"], fontsize=FS["box_body"] - 4.0)

    D.box("bayes", cx=_MX[1], cy=_MY, w=_MW[1], h=_MH,
          text="", fc=C["method"], fontsize=FS["box_body"] - 4.0)

    D.box("taxon", cx=_MX[2], cy=_MY, w=_MW[2], h=_MH,
          text="", fc=C["method"], fontsize=FS["box_body"] - 4.0)

    D.box("health_fw", cx=_MX[3], cy=_MY, w=_MW[3], h=_MH,
          text="", fc=C["method"], fontsize=FS["box_body"] - 4.0)

    # Two lines, name over reference. These used to read "(Note n)" on one
    # line; "(Supplementary Text n)" is six characters longer and the widest of
    # them -- "Hard detection + empirical fits" -- then measured 0.283 of the
    # canvas against a 0.220 column, so the four titles overran and collided
    # with each other (they are centred free text with clip_on off, so nothing
    # stopped them). Split, the widest is 0.189 and all four clear their
    # columns; shrinking the type alone did not, at 0.267.
    for x, title in [
        (_MX[0], "GMW depth correction\n(Supp. Text 3)"),
        (_MX[1], "Hard detection + empirical fits\n(Supp. Text 4)"),
        (_MX[2], "Ind/non-Ind taxon lists\n(Supp. Text 5)"),
        (_MX[3], "Health association\n(Supp. Text 6)"),
    ]:
        # Sized to the column rather than to the house title size. These four
        # read "(Supplementary Text n)" now rather than "(Note n)", six
        # characters longer, and at the full title size the widest of them
        # ("Hard detection + empirical fits") overran its column and collided
        # with its neighbours -- they are centred free text with clip_on off,
        # so nothing stopped them. Shrunk by the same ratio the longest string
        # grew; the band's body line is unchanged.
        ax.text(x, _MTITLE_Y, title,
                ha="center", va="center",
                fontsize=FS["box_title"] - 2.0, fontweight="bold", color=_TEXT,
                transform=ax.transAxes, clip_on=False, zorder=5)

    for x, body in [
        (_MX[0], "geometric-mean depth reweighting"),
        (_MX[1], "Hill-MM; Richards; beta-binomial mixture"),
    ]:
        ax.text(x, _MBODY_Y, body,
                ha="center", va="center",
                fontsize=FS["box_body"], color=_TEXT,
                transform=ax.transAxes, clip_on=False, zorder=5)

    # Offsets re-centred for the narrower box above: the badge and the bold
    # reference keep the 0.040 gap they had, but the pair now sits centred
    # rather than hard against the right edge.
    D.box("nb_taxon_meta", cx=_MX[2] - 0.0541, cy=_MBODY_Y,
          w=0.078, h=0.022,
          text="Data S4",
          fc="#FFFFFF", ec="#888888", lw=0.7,
          fontsize=_ANN_FS,
          title_fontsize=_ANN_FS,
          bold_title=True,
          box_pad=0.001)

    ax.text(_MX[2] + 0.0599, _MBODY_Y,
            r"$\bf{Tables\ S6{-}S9}$",
            ha="center", va="center",
            fontsize=_ANN_FS, color=_TEXT,
            transform=ax.transAxes, clip_on=False, zorder=5)

    ax.text(_MX[3], _MBODY_Y,
            # S12 carries the list-level mean-z test, which is the reported
            # one, so the health block spans S10-S12 rather than S10-S11.
            r"$\bf{Tables\ S10{-}S12}$  ·  within-study GMW-CMH + prevalence bg.",
            ha="center", va="center",
            fontsize=_ANN_FS, color=_TEXT,
            transform=ax.transAxes, clip_on=False, zorder=5)

    # Case-control eligible → Health framework (solid; right elbow via outer column)
    D.elbow(0.678, 0.417, _MX[3] - _MW[3]/2 + 0.045, _MY + _MH / 2)

    # ════════════════════════════════════════════════════════════════════════════
    #  FIGURE-SPECIFIC COHORTS
    # ════════════════════════════════════════════════════════════════════════════

    _FY  = 0.081
    _FH  = 0.070
    _FT  = _FY + _FH / 2

    _FIG_BOXES = [
        (_GRID_X[0], _GRID_W[0],
         "Sample richness/composition\n$\\bf{(Fig.\\ 2)}$\nGMW lifestyle cohort"),
        (_GRID_X[1], _GRID_W[1],
         "All-SGB accumulation\n$\\bf{(Fig.\\ 3)}$\n3 refs; hard + empirical fits"),
        (_GRID_X[2], _GRID_W[2],
         "Differentially prevalent taxa\n$\\bf{(Fig.\\ 4)}$\nVanish/BloSSUM (+/H+) taxa"),
        (_GRID_X[3], _GRID_W[3],
         "(Un)healthy vs Non-Ind richness\n$\\bf{(Fig.\\ 5A{-}C)}$\nNI+Ind-H+Ind-U, GMW"),
        (_GRID_X[4], _GRID_W[4],
         "Disease richness\n$\\bf{Ind/non{-}Ind\\ taxa\\ associations}$\n$\\bf{(Fig.\\ 5D{-}E)}$\nstudy-strat. GMW-CMH"),
    ]
    for xi, wi, lbl in _FIG_BOXES:
        D.box(f"fig_{xi:.3f}", cx=xi, cy=_FY, w=wi, h=_FH,
              text=lbl, fc=C["figbox"], ec="#909090", lw=0.9,
              fontsize=FS["box_body"] + 1.0,
              title_fontsize=FS["box_body"] + 2.0,
              bold_title=True)

    # ── SI placeholder row — medium grey, blank for now (SI fig numbering TBD) ──
    _SI_Y = 0.027
    _SI_H = 0.024
    for xi, wi, lbl in [
        (_GRID_X[0], _GRID_W[0], "Fig. S5–S14, Fig. S18–S20"),
        (_GRID_X[1], _GRID_W[1], "Fig. S15–S17"),
        (_GRID_X[2], _GRID_W[2], "Fig. S21–S22"),
        (_GRID_X[3], _GRID_W[3], "Fig. S23–S24"),
        # S31 and S32 were retired on 2026-10-09, when the naive-comparison
        # and IW-diagnostic figures came out and the health block was
        # renumbered. Two more were added and retired again: the list-vs-list
        # comparison (2026-10-13), whose panels are now Figure 5F-H, and the
        # metformin sensitivity (2026-10-14), withdrawn for want of power --
        # it rested on two cohorts. tools/figure_sets.py is the manifest and
        # it ends at S30.
        (_GRID_X[4], _GRID_W[4], "Fig. S25–S30"),
    ]:
        D.box(f"si_{xi:.3f}", cx=xi, cy=_SI_Y, w=wi, h=_SI_H,
              text=lbl, fc=C["si_box"], ec="#707070", lw=0.9,
              fontsize=_ANN_FS,
              title_fontsize=_ANN_FS,
              bold_title=True)

    # ── Module → figure arrows (solid = direct data/method flow) ──────────────
    _MY_BOT = _MY - _MH / 2
    D.arrow(_MX[0], _MY_BOT, _GRID_X[0], _FT)     # GMW → Fig 2
    D.arrow(_MX[0], _MY_BOT, _GRID_X[3], _FT)     # GMW → Fig 5A-C
    D.arrow(_MX[1], _MY_BOT, _GRID_X[1], _FT)     # hard/empirical → Fig 3
    D.arrow(_MX[1], _MY_BOT, _GRID_X[2], _FT)     # hard/empirical → Fig 4
    D.arrow(_MX[2], _MY_BOT, _GRID_X[2], _FT)     # Taxon → Fig 4
    D.arrow(_MX[3], _MY_BOT, _GRID_X[4], _FT)     # Health framework → Fig 5D-E

    # ════════════════════════════════════════════════════════════════════════════
    #  NOTE BADGES  (Supplement references next to each cohort step)
    # ════════════════════════════════════════════════════════════════════════════

    _NB_FC_LG = "#E8E8E8"                # light grey — Supplement reference
    _NB_EC    = "#888888"
    _NB_FS    = _ANN_FS                  # annotation text is standardized

    # Screen → Table S1, Note 1.A (inside the screen box, right side).
    # Eligibility, sample-exclusion and run/BioSample/host pointers are drawn
    # with their boxes above.
    D.box("nb_screen",  cx=0.628, cy=0.962, w=0.070, h=0.031,
          text="Table S1\nSupp. Text 1A",
          fc=_NB_FC_LG, ec=_NB_EC, lw=0.7,
          fontsize=_NB_FS, title_fontsize=_NB_FS, bold_title=True,
          box_pad=0.001, bold_all=True)

    # FINAL — TWO inset badges inside the FINAL ANALYTIC COHORT box, right side
    # White "Data S2" (the curated metadata table this paper produces) +
    # Grey  "Tables S2–S3 / Notes 1.D–E" (supplementary reference)
    D.box("nb_final_white", cx=0.6765, cy=0.688, w=0.095, h=0.023,
          text="Data S2",
          fc="#FFFFFF", ec=_NB_EC, lw=0.7,
          fontsize=_NB_FS, title_fontsize=_NB_FS, bold_title=True,
          box_pad=0.001, bold_all=True)

    D.box("nb_final_grey",  cx=0.6765, cy=0.661, w=0.095, h=0.034,
          text="Tables S2–S3\nSupp. Text 1D–E",
          fc=_NB_FC_LG, ec=_NB_EC, lw=0.7,
          fontsize=_NB_FS, title_fontsize=_NB_FS, bold_title=True,
          box_pad=0.001, bold_all=True)

    # ════════════════════════════════════════════════════════════════════════════
    #  LEGEND  (bottom-right blank space, where the unclear/mixed sidecar used to live)
    # ════════════════════════════════════════════════════════════════════════════

    _LG_X0, _LG_X1 = 0.810, 0.985
    _LG_Y_TOP      = 0.650
    _LG_Y_BOT      = 0.200
    _LG_CX         = (_LG_X0 + _LG_X1) / 2
    _LG_W          = _LG_X1 - _LG_X0
    _LG_H          = _LG_Y_TOP - _LG_Y_BOT

    # Outer rounded panel
    ax.add_patch(FancyBboxPatch(
        (_LG_X0, _LG_Y_BOT), _LG_W, _LG_H,
        boxstyle="round,pad=0.004",
        facecolor="#FAFAFA", edgecolor="#707070", linewidth=1.0,
        transform=ax.transAxes, clip_on=False, zorder=1,
    ))

    # Legend title
    ax.text(_LG_CX, _LG_Y_TOP - 0.014, "Colour key",
            ha="center", va="top",
            fontsize=FS["box_body"] + 1.0, fontweight="bold",
            color=_TEXT,
            transform=ax.transAxes, clip_on=False, zorder=2)

    # Entries: (color, label, is_subheader)
    _LG_ROWS = [
        ("SUB",  "Box styles"),
        (C["brown_primary"], "Cohort selection"),
        ("#FFFFFF",          "Data S1\u2013S4"),
        (C["method"],        "Supplementary Text"),
        (C["figbox"],        "Results figures"),
        (C["si_box"],        "Figures S1\u2013S31"),
        ("SUB",  "Lifestyle (non-industrial)"),
        (C["hg"],     "Hunter-Gatherer"),
        (C["trans"],  "Transitional"),
        (C["nonind"], "Non-industrial (HG + Trans)"),
        ("SUB",  "Lifestyle (industrial)"),
        (C["ind_life"], "Industrial (all)"),
        (C["ind_h"],    "Industrial healthy"),
        (C["ind_u"],    "Industrial unhealthy"),
        (C["ind_unk"],  "Industrial, health unknown"),
        ("SUB",  "References"),
        (C["carter"], "Carter (non-ind. biased)"),
        (C["lev"],    "Leviatan (industrial-biased)"),
        (C["combined"], "Combined (Carter + Leviatan)"),
    ]

    _SWATCH_W = 0.020
    _SWATCH_H = 0.014
    _SWATCH_X = _LG_X0 + 0.009
    _LABEL_X  = _SWATCH_X + _SWATCH_W + 0.007
    _ROW_H_ENTRY  = 0.0205
    _ROW_H_HEADER = 0.0220

    _yc = _LG_Y_TOP - 0.036   # start a bit below the title
    for color, label in _LG_ROWS:
        if color == "SUB":
            ax.text(_LG_X0 + 0.010, _yc, label,
                    ha="left", va="center",
                    fontsize=FS["box_body"] - 2.0,
                    fontweight="bold", style="italic", color="#444444",
                    transform=ax.transAxes, clip_on=False, zorder=2)
            _yc -= _ROW_H_HEADER
        else:
            ax.add_patch(FancyBboxPatch(
                (_SWATCH_X, _yc - _SWATCH_H / 2), _SWATCH_W, _SWATCH_H,
                boxstyle="round,pad=0.001",
                facecolor=color, edgecolor="#555555", linewidth=0.6,
                transform=ax.transAxes, clip_on=False, zorder=2,
            ))
            ax.text(_LABEL_X, _yc, label,
                    ha="left", va="center",
                    fontsize=FS["box_body"] - 2.0, color=_TEXT,
                    transform=ax.transAxes, clip_on=False, zorder=2)
            _yc -= _ROW_H_ENTRY

    # ── Save ──────────────────────────────────────────────────────────────────
    pdf_path = str(output_path)
    if not pdf_path.endswith(".pdf"):
        pdf_path += ".pdf"
    png_path = pdf_path.replace(".pdf", ".png")

    out_dir = os.path.dirname(os.path.abspath(pdf_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    _apply_display_text(fig)

    # The figure's data object. This one is a schematic rather than a plot, so
    # its data is the cohort cascade it prints: every count shown on it, as one
    # row per label. That is what a reader would want to check against.
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from persistence import figure_source as _fsrc
        import pandas as _pd
        _c = counts if isinstance(counts, dict) else {}
        if _c:
            _fsrc.emit_for_path(
                png_path,
                {"cohort_counts": _pd.DataFrame(
                    [{"label": k, "value": v} for k, v in _c.items()])},
                notes="Counts printed on the roadmap schematic.")
    except Exception as _exc:
        print(f"  [data] roadmap counts unavailable: {_exc}")

    try:
        from persistence.plot_utils import prepare_for_publication
        # A schematic with text sized to the boxes it sits in; the shared
        # type scale has nothing useful to say about it.
        fig._skip_type_normalise = True
        prepare_for_publication(fig, 'figure_s0_roadmap')
    except Exception as _exc:
        print(f'  [format] roadmap: {_exc}')
    for path in (pdf_path, png_path):
        # Resolution only; the layout is untouched.
        fig.savefig(path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  Console summary
# ══════════════════════════════════════════════════════════════════════════════

def _print_counts(counts: dict) -> None:
    def _n(k: str) -> str:
        v = counts.get(k, "?")
        return f"{v:,}" if isinstance(v, int) else str(v)

    def _fmt(v: Any) -> str:
        return f"{v:,}" if isinstance(v, int) else str(v)

    n_total_tallied = counts.get("n_total", "?")

    w = 50
    print(f"\n  {'─'*w}")
    print(f"  {'Figure S0 cohort accounting':^{w}}")
    print(f"  {'─'*w}")
    print(f"  M1 matched to matrices   : {_n('matched_to_matrix'):>10} samples")
    for k in _CASCADE_KEYS[1:]:
        print(f"    {k.replace('_', ' '):<34}: {_n(k):>6}")
    print(f"  Secondary runs           : {_n('n_sec_runs'):>10} runs from {_n('n_sec_primaries')} samples")
    print(f"    All retained as technical replicates (none reinstated)")
    print(f"  FINAL ANALYTIC COHORT    : {_fmt(n_total_tallied):>10} samples")
    print(f"                           : {_n('h_total'):>10} hosts")
    print(f"                           : {_n('n_studies'):>10} studies")
    print(f"                           : {_n('n_countries'):>10} countries")
    print(f"  ── Classification ──────────────────────────────────")
    print(f"  Lifestyle-classified     : {_n('n_lifestyle_classified'):>10}")
    print(f"    H-G                    : {_n('n_hg'):>10}")
    print(f"    Transitional           : {_n('n_trans'):>10}")
    print(f"    Industrial             : {_n('n_ind'):>10}")
    print(f"  Non-industrial combined  : {_n('n_nonind'):>10}")
    print(f"  ── Industrial health split ─────────────────────────")
    print(f"  Ind. healthy             : {_n('n_ind_h'):>10}")
    print(f"  Ind. unhealthy           : {_n('n_ind_u'):>10}")
    print(f"  Ind. unknown             : {_n('n_ind_unk'):>10}")
    print(f"  {'─'*w}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline entry point
# ══════════════════════════════════════════════════════════════════════════════

def run(data: Any, outdir: str) -> None:
    """Called from the main pipeline once the cohort is assembled."""
    print("\n" + "=" * 60)
    print("  Figure S0: Analytic roadmap")
    print("=" * 60)
    counts = derive_counts(data, ref_outdir=outdir)
    _print_counts(counts)
    draw_roadmap(counts, output_path=os.path.join(outdir, "figure_s0_roadmap.pdf"))


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone entry
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw Figure S0 analytic roadmap from the UHGG matrix."
    )
    parser.add_argument("--ref", default="UHGG")
    parser.add_argument("--out", default=None,
                        help="Output directory. Default: results/<ref>/")
    parser.add_argument("--from-results", default=None, metavar="REF_DIR",
                        help="Redraw from a finished run's results/<ref>/ "
                             "(cohort_samples.csv + stats_report.csv) and M1, "
                             "without loading the alignment matrices.")
    parser.add_argument("--m1", default="final_run/metadata_M1.csv",
                        help="Metadata M1 for --from-results (Subject_ID, Unhealthy).")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=N",
                        help="With --from-results: override one count, for a value "
                             "the run's registry predates (e.g. n_sec_runs=3352).")
    args = parser.parse_args()

    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(_here))

    if args.from_results:
        _counts = counts_from_results(args.from_results, args.m1)
        for _kv in args.set:
            _k, _v = _kv.split("=", 1)
            _counts[_k] = int(_v)
        _print_counts(_counts)
        _outdir = args.out or args.from_results
        os.makedirs(_outdir, exist_ok=True)
        draw_roadmap(_counts, output_path=os.path.join(_outdir, "figure_s0_roadmap.pdf"))
        sys.exit(0)

    try:
        from persistence.loading import load_reference
        from persistence.stats_report import StatsReport
        from persistence.reinstate_secondary import reinstate_distinct_secondaries
    except ImportError as exc:
        sys.exit(f"Cannot import pipeline modules: {exc}\n"
                 "Run from the directory containing the pipeline package, or "
                 "add the package root to PYTHONPATH.")

    print(f"\nLoading reference: {args.ref}")
    _data  = load_reference(args.ref)
    _stats = StatsReport(args.ref)
    # Kept so this script mirrors the pipeline exactly. With
    # config.REINSTATE_SECONDARIES False it is a no-op that reports what it
    # would have done; if the flag is ever turned back on for a sensitivity
    # run, the roadmap follows without a second edit.
    print("\nSecondary-run handling ...")
    _data  = reinstate_distinct_secondaries(_data, _stats)

    _outdir = args.out or os.path.join("results", args.ref)
    os.makedirs(_outdir, exist_ok=True)
    run(_data, _outdir)
