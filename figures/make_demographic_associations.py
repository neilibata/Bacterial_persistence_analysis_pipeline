#!/usr/bin/env python3
"""Supplementary demographic associations: age, sex and BMI, as Figure 5D/5E.

Same two stacked panels as the lower half of Figure 5, and deliberately the
same drawing code:

  A (as 5D)  GMW depth-corrected richness, stratum vs its reference, with the
             difference and its bootstrap CI underneath.
  B (as 5E)  Prevalence-difference distributions for each curated taxon family
             against its prevalence-matched background.

The seven columns are the contrasts of `config.DEMOGRAPHIC_MAP`, grouped by
metadata field with a dashed divider between groups: four age bands measured
against 18-29, then sex, then the two BMI contrasts. Cohort sizes for all of
them are in `demographic_cohorts.csv` (Table S3).

The age and sex contrasts are restricted to Industrial healthy samples so a
stratum difference is not just the disease burden that tracks age. The BMI
contrasts cannot be: obesity is what sets the Unhealthy label, so healthy and
obese is empty by construction. That asymmetry is recorded per contrast in
`demographic_cohorts.csv` and stated in the caption.

    python figures/make_demographic_associations.py                 # combined
    python figures/make_demographic_associations.py --ref UHGG
    python figures/make_demographic_associations.py --lists VANISH BloSSUM
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))     # figures/
ROOT = os.path.dirname(HERE)                          # repo root
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse Figure 5's panel drawing verbatim. The point of this figure is that a
# demographic contrast is computed and drawn exactly as a disease contrast is;
# a second implementation would eventually disagree with the first.
from figures.figure5 import (                          # noqa: E402
    FS, FS_SMALL, LIST_COLORS, _bgcol, _style, draw_D,
    load_energy_p,
    draw_E_cell,
    # The accessors, not the column names. This panel read `prev_diff` and
    # `importance_weight` directly, so it kept plotting the pooled shift
    # against the superseded importance-weighted background after the disease
    # figures moved to the stratified shift and the matched weights -- a
    # third copy of the panel-E data path, silently a version behind.
    _shift_col, _shift_pp, _bg_w, _common_shift,
)
from persistence import config, figure_source       # noqa: E402
from persistence.figure_rescale import (             # noqa: E402
    save_column_width,
)

#: Taxon families to give an E row, in order, when present.
DEFAULT_LISTS = ["VANISH", "BloSSUM", "VanishH+", "BlossumH+"]

FIG_W = 24.0


def _divider_columns(conds):
    """Column counts after which to draw a group divider.

    Grouped by which metadata field the contrast reads, so the reader sees
    age / sex / BMI as three blocks rather than seven unrelated columns.
    """
    field = []
    for c in conds:
        ctype = config.DEMOGRAPHIC_MAP.get(c, "")
        field.append("Age" if str(ctype).startswith("age_band")
                     else "Gender" if ctype == "sex" else "BMI")
    return {i + 1 for i in range(len(field) - 1) if field[i] != field[i + 1]}


def load_tables(results_dir, spec_tag):
    """The three frames this figure draws, or Nones where absent."""
    def _read(name):
        p = os.path.join(results_dir, f"demographic_{name}_{spec_tag}.csv")
        if not os.path.exists(p):
            print(f"  [WARN] not found: {p}")
            return None
        return pd.read_csv(p, low_memory=False)

    rich = _read("richness_gmw")
    if rich is not None and "n_strata" in rich.columns:
        # Same rule as Figure 5 panel D: a contrast resting on a single study
        # has no between-study variance, so no interval and no p-value, and a
        # bar with neither reads as a null result rather than an untested one.
        from figures.figure5 import MIN_STRATA_TO_PLOT
        thin = rich[rich["n_strata"] < MIN_STRATA_TO_PLOT]
        if len(thin):
            print("  [panel A] dropping single-study contrast(s): "
                  + ", ".join(str(c) for c in thin["Condition"]))
        rich = rich[rich["n_strata"] >= MIN_STRATA_TO_PLOT].reset_index(
            drop=True)
    return rich, _read("per_taxon_gmw"), _read("per_taxon_prevmatched")


#: The type environment figure5's drawing code is tuned for. Pinned rather
#: than inherited: figure5 sets font.size 17 at import and
#: persistence.section3_health sets it to 7 with an Arial stack, so the
#: same drawing code rendered at a different size depending on which module
#: the calling process had imported first. That changed every text extent,
#: and since save_column_width crops with bbox_inches="tight", it changed the
#: output canvas -- the figure built by this script and the one rebuilt from
#: its own data object came out 2174x1917 and 2167x1928.
_RC = {
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    # The face has to be pinned by name, not as "sans-serif":
    # persistence.section3_health rewrites font.sans-serif itself, so
    # asking for the generic family resolved to DejaVu Sans when only
    # figure5 was imported and to Arial once section3_health was, which is
    # what made the two render paths disagree. Arial is what the rest of the
    # supplement is drawn in, so this figure matches its neighbours.
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    # Geometry, not just type. section3_health changes 21 rcParams in all,
    # and these four move the tight bbox: pinning only the font sizes left a
    # 1 px / 4 px disagreement between the two render paths.
    "axes.labelpad": 4.0,
    "xtick.major.pad": 3.5,
    "ytick.major.pad": 3.5,
    "axes.grid": False,
}


def compose_figure(panel_a, pt_df, bg_df, lists, n_boot=2000, seed=42,
                   energy_p=None):
    """Draw the figure. Returns (fig, conds, active_pairs)."""
    with plt.rc_context(_RC):
        return _compose(panel_a, pt_df, bg_df, lists, n_boot, seed,
                        energy_p or {})


def _compose(panel_a, pt_df, bg_df, lists, n_boot, seed, energy_p):
    rng = np.random.default_rng(seed)

    # Column order follows config, not the CSV, so a reordered results file
    # cannot silently reorder the figure.
    order = list(config.DEMOGRAPHIC_MAP)
    conds = [c for c in order if c in set(panel_a["Condition"])]
    panel_a = (panel_a.set_index("Condition").loc[conds].reset_index())
    n_cond = len(conds)

    active_pairs = []
    if pt_df is not None and bg_df is not None:
        avail = set(pd.concat([pt_df, bg_df], ignore_index=True)["Taxon_list"])
        active_pairs = [(ln, f"{ln}_bg") for ln in lists
                        if ln in avail and f"{ln}_bg" in avail]
    n_pairs = max(len(active_pairs), 1)

    # Figure 5's geometry, not an approximation of it. The reused drawing code
    # is hand-tuned for a 3.09 in column at FS=17, which is what puts its type
    # exactly on the 5 pt floor once the page reduction is applied. Authoring
    # this figure at any other column width silently moves every string off
    # that floor.
    lm, rm = 0.075, 0.975
    ecol_w = (rm - lm) * FIG_W / n_cond
    dtop_h, dbot_h, erow_h = ecol_w * 0.88, ecol_w * 0.80, ecol_w * 1.02
    fig_h = dtop_h + dbot_h + n_pairs * erow_h + 2.1

    gap = 0.34                     # inches between A and B
    fig_h = fig_h + gap
    fig = plt.figure(figsize=(FIG_W, fig_h))

    # Two gridspecs, not one: within A and within B the axes are butted up
    # (hspace=0), but A's bottom sub-panel carries the column labels inside
    # it, and B's panel letter sits just above B's first row. Sharing one
    # gridspec puts both in the same band.
    a_top = 1.0 - 0.25 / fig_h
    a_bot = a_top - (dtop_h + dbot_h) / fig_h
    gs_a = gridspec.GridSpec(
        2, n_cond, figure=fig, height_ratios=[dtop_h, dbot_h],
        width_ratios=[1.0] * n_cond, left=lm, right=rm,
        top=a_top, bottom=a_bot, hspace=0.0, wspace=0.0)
    gs_b = gridspec.GridSpec(
        n_pairs, n_cond, figure=fig, width_ratios=[1.0] * n_cond,
        left=lm, right=rm, top=a_bot - gap / fig_h, bottom=1.85 / fig_h,
        hspace=0.0, wspace=0.0)

    # ── A: richness, as 5D ───────────────────────────────────────────
    A_top, A_bot = [], []
    for ci in range(n_cond):
        A_top.append(fig.add_subplot(gs_a[0, ci],
                                     sharey=A_top[0] if A_top else None))
        A_bot.append(fig.add_subplot(gs_a[1, ci],
                                     sharey=A_bot[0] if A_bot else None))
    draw_D(A_top, A_bot, panel_a, _divider_columns(conds))
    # draw_D labels the difference "Case - control", which is the disease
    # framing. Here g1 is the reference band, not a control group.
    A_bot[0].set_ylabel("Richness diff.\n(Stratum − reference)", fontsize=FS)
    A_top[0].text(-0.02, 1.02, "A", transform=A_top[0].transAxes,
                  fontsize=FS + 5, fontweight="bold", va="bottom", ha="right")

    # ── B: prevalence distributions, as 5E ───────────────────────────
    B_rows = []
    for pi in range(n_pairs):
        row = []
        for ci in range(n_cond):
            row.append(fig.add_subplot(
                gs_b[pi, ci], sharey=B_rows[pi][0] if ci > 0 and
                pi < len(B_rows) else None))
        B_rows.append(row)
    if B_rows:
        B_rows[0][0].text(-0.02, 1.02, "B", transform=B_rows[0][0].transAxes,
                          fontsize=FS + 5, fontweight="bold",
                          va="bottom", ha="right")

    if active_pairs and pt_df is not None and bg_df is not None:
        # Same guard Figure 5 uses: a table written before the stratified
        # shift was merged on has only `prev_diff`, and concatenating it with
        # one that has `rd_mh` blanks that side and prints "n/a" everywhere.
        _common_shift(pt_df, bg_df)
        combined = pd.concat([pt_df, bg_df], ignore_index=True)
        _SC = _shift_col(combined) or "prev_diff"
        print(f"  [panel B] shift column: {_SC}")
        for pi, (list_name, bg_name) in enumerate(active_pairs):
            col, bgcol = LIST_COLORS.get(list_name, "#888"), None
            bgcol = _bgcol(col)
            for ci, cond in enumerate(conds):
                ax = B_rows[pi][ci]
                sub_l = combined[(combined["Condition"] == cond) &
                                 (combined["Taxon_list"] == list_name)
                                 ].dropna(subset=[_SC])
                sub_b = combined[(combined["Condition"] == cond) &
                                 (combined["Taxon_list"] == bg_name)
                                 ].dropna(subset=[_SC])
                if len(sub_l) < 3 or sub_b.empty:
                    _style(ax)
                    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes,
                            ha="center", va="center", fontsize=FS - 2,
                            color="#aaa")
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    sl = _shift_pp(sub_l)
                    bg_u = (sub_b.groupby("taxon", sort=False).first()
                            .reset_index() if "taxon" in sub_b.columns
                            else sub_b.copy())
                    sb_vals = _shift_pp(bg_u)
                    sb_w = _bg_w(bg_u)
                    # Star from Table S10B, so the panel and the table
                    # cannot disagree. See draw_E_cell for why recomputing
                    # here was wrong.
                    draw_E_cell(ax, sl, sb_vals, sb_w, col, bgcol, n_boot,
                                rng,
                                p_energy=energy_p.get((cond, list_name)))

                ax.set_yticks([]); ax.tick_params(labelleft=False)
                if ci == 0:
                    ax.set_ylabel(f"{list_name}\ndensity", fontsize=FS_SMALL)
                else:
                    ax.spines["left"].set_color("#dddddd")
                if pi == n_pairs - 1:
                    # The label is a single figure-level line below, not an
                    # axis label: seven axis copies ran into each other, and
                    # the one surviving copy ran into the legend.
                    # FS_SMALL exactly: at this figure's x0.317 page
                    # reduction, 16 pt prints at 5.07 pt and 14 pt at 4.43,
                    # which is under the floor.
                    ax.tick_params(labelbottom=True, labelsize=FS_SMALL,
                                   rotation=0)
                    # draw_E_cell sets the x range from the data
                    # percentiles, so each column has its own scale. Two
                    # things then go wrong with the default ticks: a narrow
                    # stratum gets too many of them, and the outermost label
                    # of one column butts against the outermost label of the
                    # next, because wspace is 0. Asking for few ticks and
                    # pruning both ends fixes both -- the columns are butted
                    # deliberately, so the labels have to stay inboard.
                    ax.xaxis.set_major_locator(MaxNLocator(4, prune="both"))
                else:
                    ax.set_xlabel("")
                    ax.tick_params(labelbottom=False)


    if active_pairs:
        fig.text(0.5, 1.02 / fig_h,
                 "Stratum − reference     Prevalence difference (%)",
                 ha="center", va="bottom", fontsize=FS_SMALL)

    # A single legend for the whole of B. Each row's y-label already names and
    # colours its family, so the only thing left to explain is the style
    # convention; four per-row legends stacked on top of one another.
    if active_pairs:
        fig.legend(handles=[
            Patch(facecolor="#888888", alpha=0.55, edgecolor="none",
                  label="Taxon family (row label)"),
            Line2D([], [], color="#888888", lw=1.4, ls="--",
                   label="Prevalence-matched background"),
            Patch(facecolor="#bbbbbb", alpha=0.55, edgecolor="none",
                  label="Overlap"),
        ], loc="lower center", ncol=3, frameon=False, fontsize=FS_SMALL,
            bbox_to_anchor=(0.5, 0.18 / fig_h))

    return fig, conds, active_pairs


def rebuild_from_data_object(obj, out_path, n_boot=2000):
    """Redraw from a published data object, with no results tree present."""
    P = obj["panels"]
    ex = obj.get("extras", {}) or {}
    lists = ex.get("lists", DEFAULT_LISTS)
    ep = {(str(c), str(l)): float(v) for c, l, v in (ex.get("energy_p") or [])}
    if ep:
        print(f"  [panel B] {len(ep)} energy p-values from the data object")
    fig, _c, _p = compose_figure(P["richness"], P.get("per_taxon"),
                                 P.get("background"), lists, n_boot=n_boot,
                                 energy_p=ep)
    with plt.rc_context(_RC):
        save_column_width(fig, os.path.splitext(out_path)[0])
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="combined")
    ap.add_argument("--spec", default="10pct")
    ap.add_argument("--lists", nargs="*", default=DEFAULT_LISTS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    results_dir = os.path.join(ROOT, "results", a.ref, "section3_health",
                               "main")
    panel_a, pt_df, bg_df = load_tables(results_dir, a.spec)
    if panel_a is None or panel_a.empty:
        print("  [SKIP] no demographic richness table; run "
              "`python -m persistence --only s3` first")
        return 1

    out_dir = os.path.join(ROOT, "results", a.ref, "section3_health",
                           "supplement")
    os.makedirs(out_dir, exist_ok=True)
    out_path = a.out or os.path.join(
        out_dir, f"demographic_associations_{a.ref}_{a.spec}.png")

    energy_p = load_energy_p(results_dir, a.spec,
                             prefix="demographic_table_s10")
    fig, conds, pairs = compose_figure(panel_a, pt_df, bg_df, a.lists,
                                       n_boot=a.n_boot, energy_p=energy_p)

    panels = {"richness": panel_a}
    if pt_df is not None:
        panels["per_taxon"] = pt_df
    if bg_df is not None:
        panels["background"] = bg_df
    # emit(), not emit_for_path(): the latter looks the slot up in the figure
    # manifest and returns None when it is absent, so once S32 was withdrawn
    # from the publishable set the data object would have silently stopped
    # being written. The figure is still fully reproducible from its pickle;
    # it simply is not a numbered slot any more.
    figure_source.emit(
        os.path.dirname(out_path) or ".", "demographic", panels,
        stem=os.path.splitext(os.path.basename(out_path))[0],
        reference=a.ref,
        extras={"lists": list(a.lists), "contrasts": conds,
                "pairs": [p[0] for p in pairs], "seed": 42,
                "n_boot": a.n_boot,
                "energy_p": [[c, l, v] for (c, l), v in energy_p.items()]},
        notes="Demographic strata (age/sex/BMI) through the Figure 5D/5E "
              "machinery; cohort sizes in demographic_cohorts.csv.")

    # Not savefig_dual: its type-normalisation and auto-fit passes are tuned
    # for figures authored at their final size, and on this one they inflated
    # 188 strings to the 5 pt floor and reflowed the labels. save_column_width
    # touches no artist -- it only changes the size the file declares.
    with plt.rc_context(_RC):
        save_column_width(fig, os.path.splitext(out_path)[0])
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"  {len(conds)} contrasts x {len(pairs)} taxon families")
    return 0


if __name__ == "__main__":
    sys.exit(main())
