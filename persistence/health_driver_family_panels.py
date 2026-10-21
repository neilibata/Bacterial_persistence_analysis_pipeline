"""Figures S26-S29: which taxa carry each list's health association.

Four figures on the four focal contrasts (Obese, Glucose Intolerance, IBD,
CRC / Adenoma).  The pooled "Unhealthy" column is not among them: it contains
the cases of the other four, so a per-taxon test treating the columns as
independent observations was rewarding a replicate of itself.

  S26  VANISH family   -- VANISH+ and VanishH+ together
  S27  BloSSUM family  -- BloSSUM+ and BlossumH+ together
  S28  VANISH          -- the original list
  S29  BloSSUM         -- the original list

The derived pair of each family shares a figure because the two lists overlap
heavily -- 96 of 100 taxa for VANISH, 82 of 100 for BloSSUM -- so separate
panels would be near-duplicates.  A two-column membership strip left of the
heatmap says which list each taxon is on, and carries that list's own FDR
verdict, so the strip answers both "which taxa distinguish the lists" and
"does it matter".  It is where the BloSSUM+ advantage becomes visible: of the
18 taxa unique to BloSSUM+, 12 are FDR-significant; of the 18 unique to
BlossumH+, one is.

Each row shows:
  * the membership strip, one cell per list in the family;
  * the per-condition z, with a mark on every cell clearing FDR within that
    condition.  High-prevalence taxa reverse sign between conditions
    (Phocaeicola dorei is -1.8 in obesity and +7.3 in IBD), so a single
    collapsed score would hide what the cross-condition mean averages over;
  * the cross-condition mean z with its 95% bootstrap interval, and a mark
    when the taxon clears FDR across conditions;
  * directional consistency.

z is Figure 5E's own statistic and the FDR comes from the cluster bootstrap
over study cohorts in `health_driver_bootstrap`; see that module for why the
one-sample t-test it replaces could not be used here.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from . import config
from .plot_utils import (apply_oi_style, current_fs, page_figsize,
                         savefig_dual, SUBMISSION_DPI)

apply_oi_style()

#: Vertical inches per taxon row. Raised from 0.19 with the type: at ~7.5 pt a
#: label stands about 0.10 in, and 0.19 in of pitch left too little between
#: adjacent species names to read them as separate rows.
ROW_IN = 0.23
#: Passed to savefig_dual as `_type_boost`.
#:
#: savefig_dual sets every string's size itself, from
#: `clamp(median_panel_width_in * TYPE_BASE_PER_IN * boost, 5.2, 8.9)`, so the
#: fontsize arguments below only fix relative order -- the absolute size comes
#: from here. These panels are narrow (the median is about 0.83 in, and the
#: colour key counts as an axes, which drags that median down further), so at
#: the old 1.38 the computed base was 4.6 pt and every string was clamped up
#: to the 5.2 pt floor: the figure was being set at the smallest size the house
#: style permits. 2.25 puts the base near 7.5 pt, which is Science's body size,
#: and leaves the ratios in TYPE_RATIO to do the rest.
TYPE_BOOST = 2.25
#: Sizes pinned past the shared type pass, in printed points (the figure is
#: drawn at page width, so these are literal). The heading row and the key are
#: what a reader navigates the panel by, so they are set above body size and
#: in bold; the legend is reference matter and only needs to clear it.
#: Thinnest rule drawn anywhere on these figures, as DECLARED.
#:
#: Science allows no rule below 0.5 pt at final size, and declared is not
#: printed: audit_science_specs measures these figures reducing by x0.79 to
#: x0.82, because a tall panel is scaled down to fit the page. A declared 0.5
#: therefore prints at 0.41 and fails, which is what it did. 0.8 prints at
#: 0.63-0.66 across that range, clearing the floor with room for a taller
#: panel, and matches PRINTED_LW["spine"] = 0.6 as a printed target.
RULE_LW = 0.8
TITLE_PT = 9.5
KEY_PT = 8.0
LEGEND_PT = 7.6
#: Rows a figure will not drop below, so a list with almost nothing
#: significant does not collapse to a two-row stub.
ROW_FLOOR = 10
#: Layout ceiling for one page.
ROW_CAP = 56
#: FDR threshold for every mark on these figures.
Q_CUT = 0.20
#: Display limit for the heatmap colour scale, in z units.
Z_LIM = 3.0

#: These four moved up one on 2026-10-19, when the within-clade figure took
#: S26 to match its first citation; they were S26-S29. They must agree with
#: tools/additional_figures_manifest.txt, and tools/audit_final_run.py
#: cross-checks them against it.
FAMILIES = {
    "VANISH family":  ("S27", ["VANISH+", "VanishH+"]),
    "BloSSUM family": ("S28", ["BloSSUM+", "BlossumH+"]),
    "VANISH":         ("S29", ["VANISH"]),
    "BloSSUM":        ("S30", ["BloSSUM"]),
}
#: Short strip headers per list.
#: Which of the section's two directories each slot belongs in, matching
#: tools/additional_figures_manifest.txt.
SLOT_DIR = {"S27": "main", "S28": "main",
            "S29": "supplement", "S30": "supplement"}

STRIP_LABEL = {"VANISH+": "+", "VanishH+": "H+",
               "BloSSUM+": "+", "BlossumH+": "H+",
               "VANISH": "FDR", "BloSSUM": "FDR"}

COL_ENRICH = config.OI["blue"]
COL_DEPLET = config.OI["vermillion"]


def _short_cond(c):
    return {"Glucose Intolerance": "Glucose\nIntol.",
            "CRC / Adenoma": "CRC/\nAdenoma"}.get(c, c)


def _label_one(taxon, taxonomy_map):
    """Deepest named GTDB rank, never a bare accession."""
    tx = str(taxonomy_map.get(taxon, "") or "")
    filled = [p.strip() for p in tx.split(";")
              if p.strip() and not p.strip().endswith("__")]
    if not filled:
        return str(taxon)
    rank, _, name = filled[-1].partition("__")
    name = name.strip()
    if not name:
        return str(taxon)
    if rank == "s":
        return name
    suffix = {"g": " sp.", "f": " (family)", "o": " (order)",
              "c": " (class)", "p": " (phylum)"}.get(rank, "")
    return f"{name}{suffix}"


def _labels(taxa, taxonomy_map):
    """`_label_one` over a sequence, disambiguating collisions."""
    taxa = list(taxa)
    labs = [_label_one(t, taxonomy_map) for t in taxa]
    dupes = {l for l in labs if labs.count(l) > 1}
    if not dupes:
        return labs
    out = []
    for t, l in zip(taxa, labs):
        if l in dupes:
            tail = str(t).split("_")[-1].split(".")[0][-6:]
            l = (l[:23] + "..." if len(l) > 23 else l) + f" [{tail}]"
        out.append(l)
    return out


def _italic(label):
    """Italicise a taxon label, exactly as the rest of section 3 does.

    Delegated rather than reimplemented. The shared helper already leaves
    upright what is not part of the name -- "sp.", a placeholder epithet such
    as "sp000434975", a parenthesised rank, the accession appended to break a
    tie -- and, just as importantly, it escapes the underscore in GTDB's
    suffixed names. A local implementation rendered "Clostridium_A" with a
    subscript A, which is not how GTDB spells it and did not match S30 or the
    captions.
    """
    from .section3_health import _italic_taxon_label
    return _italic_taxon_label(label)


def _stars(p):
    """Conventional significance stars for an uncorrected within-condition p."""
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def select_rows(driver_df, lists, conditions):
    """The taxa a family figure draws, ordered by cross-condition mean z.

    A taxon is drawn when it clears FDR on ANY list of the family, so the rows
    are the union the figure is about. Only a family with fewer than ROW_FLOOR
    such taxa reaches past them, and then in significance order, so the extra
    rows are the next best evidence rather than an arbitrary top-N.
    """
    d = driver_df[driver_df["Taxon_list"].isin(lists) & driver_df["scored"]]
    if d.empty:
        return pd.DataFrame(), 0, 0
    sig_taxa = set(d.loc[d["q_value"] < Q_CUT, "taxon"])
    # One row per taxon: z, CI and consistency are list-independent (the
    # background curve excludes every sibling list, so both lists of a family
    # are scored against the same background), while q is per list and lives
    # in the strip.
    first = d.sort_values("q_value").drop_duplicates("taxon")
    n_sig = len(sig_taxa)
    want = min(max(n_sig, min(ROW_FLOOR, len(first))), ROW_CAP, len(first))
    capped = max(0, n_sig - want)
    pool = (first[first["taxon"].isin(sig_taxa)] if n_sig >= want else first)
    rows = pool.sort_values("q_value").head(want)
    return (rows.sort_values("mean_z", ascending=False).reset_index(drop=True),
            n_sig, capped)


def draw(driver_df, taxonomy_map, family, outdir, conditions,
         ref="combined", spec_tag="10pct", spec_desc="", q_cut=Q_CUT):
    """Draw one of S26-S29. Returns the PNG path, or None when there is nothing."""
    slot, lists = FAMILIES[family]
    lists = [l for l in lists if l in set(driver_df["Taxon_list"])]
    if not lists:
        print(f"    [drivers] {family}: no list present in the table")
        return None
    rows, n_sig, capped = select_rows(driver_df, lists, conditions)
    if rows.empty:
        print(f"    [drivers] {family}: nothing scored")
        return None
    conds = [c for c in conditions if f"z_{c}" in rows.columns]
    n_tax = len(rows)
    taxa = list(rows["taxon"])
    FS = current_fs()

    # per-list q for the strip
    qmap = {}
    for lst in lists:
        s = driver_df[driver_df["Taxon_list"] == lst].set_index("taxon")
        qmap[lst] = s["q_value"].reindex(taxa).values

    print(f"    [drivers] {family}: {n_sig} taxa at q<{q_cut:g}; showing "
          f"{n_tax}"
          + (f" -- {capped} omitted by the {ROW_CAP}-row cap" if capped else
             (f" -- {n_tax - n_sig} added to reach the {ROW_FLOOR}-row floor"
              if n_tax > n_sig else "")))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "z", [COL_DEPLET, "#f7f7f7", COL_ENRICH], N=256)
    norm = mcolors.TwoSlopeNorm(vmin=-Z_LIM, vcenter=0.0, vmax=Z_LIM)

    # Margins in INCHES, not figure fractions. These figures range from ten
    # rows to fifty, so a fixed fraction reserves a band that grows with the
    # figure: at bottom=0.16 the tall BloSSUM panel was leaving close to two
    # inches of blank paper under its axis labels, which is what pushed the
    # legend so far from the panels.
    TOP_IN = 0.40      # one title line, with the colour key on it
    # Rotated condition labels, the axis labels under them, then the legend.
    # Sized in inches so it does not grow with the figure.
    # Tight enough that the legend sits just under the axis labels rather
    # than adrift at the foot of the page: the rotated condition labels and
    # the two axis labels need about 0.55 in, the legend about 0.17 in.
    BOT_IN = 0.74
    LEGEND_IN = 0.02   # the legend's own baseline above the bottom edge
    fig_h = max(2.8, n_tax * ROW_IN + TOP_IN + BOT_IN)
    fig, axes = plt.subplots(
        1, 4, figsize=page_figsize(height_in=fig_h),
        gridspec_kw={"width_ratios": [0.20 * len(lists) + 0.10,
                                      max(len(conds) * 0.34, 2.0),
                                      1.20, 0.95]})
    top_f = 1.0 - TOP_IN / fig_h
    bot_f = BOT_IN / fig_h
    _m = dict(left=0.40, right=0.99, top=top_f, bottom=bot_f, wspace=0.14)
    fig.subplots_adjust(**_m)
    fig._skip_savefig_tight_layout = True
    fig._margin_overrides = dict(_m)
    fig._type_boost = TYPE_BOOST
    ax_strip, ax_heat, ax_loll, ax_sc = axes

    # ── membership strip ────────────────────────────────────────────────────
    # Four states per cell, which lets one strip carry membership AND that
    # list's cross-condition FDR verdict at both thresholds: absent, present,
    # present and q < 0.20 (dark), and q < 0.05 (dark plus a star). The star
    # is the finer threshold rather than a third fill because two dark greys
    # that close together are not separable at this size in print.
    C_ABSENT, C_MEMBER, C_SIG = "#ffffff", "#c9c9c9", "#3a3a3a"
    for j, lst in enumerate(lists):
        on = rows[f"on_{lst}"].values if f"on_{lst}" in rows else np.ones(n_tax, bool)
        q = qmap[lst]
        for i in range(n_tax):
            if not on[i]:
                fc = C_ABSENT
            elif np.isfinite(q[i]) and q[i] < q_cut:
                fc = C_SIG
            else:
                fc = C_MEMBER
            ax_strip.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                         facecolor=fc, edgecolor="#9a9a9a",
                                         linewidth=RULE_LW))
            if on[i] and np.isfinite(q[i]) and q[i] < 0.05:
                ax_strip.text(j, i, "*", ha="center", va="center",
                              fontsize=FS - 1, color="#ffffff",
                              fontweight="bold")
    ax_strip.set_xlim(-0.5, len(lists) - 0.5)
    ax_strip.set_ylim(n_tax - 0.5, -0.5)
    ax_strip.set_xticks(range(len(lists)))
    ax_strip.set_xticklabels([STRIP_LABEL.get(l, l) for l in lists],
                             fontsize=FS - 1)
    ax_strip.set_yticks(range(n_tax))
    ax_strip.set_yticklabels([_italic(l) for l in _labels(taxa, taxonomy_map)],
                             fontsize=FS - 2)
    ax_strip.tick_params(length=0, pad=1)
    for sp in ax_strip.spines.values():
        sp.set_visible(False)

    # ── per-condition heatmap, starred on the within-condition p ────────────
    Z = np.array([[rows.loc[i, f"z_{c}"] for c in conds] for i in range(n_tax)],
                 dtype=float)
    E = np.array([[bool(rows.loc[i, f"eligible_{c}"])
                   if f"eligible_{c}" in rows else True
                   for c in conds] for i in range(n_tax)])
    # The cell mark is the UNCORRECTED within-condition p, in conventional
    # stars. The question a cell answers is "does this taxon move in THIS
    # condition", which is its own test; the row's FDR verdict is the
    # cross-condition claim and lives in the strip. Reporting q here as well
    # would put two different multiplicity scales in one panel. The caption
    # has to say that with four conditions per row these marks are
    # descriptive -- a 51-row panel holds 204 cells, so about ten single
    # stars are expected by chance.
    P = np.array([[rows.loc[i, f"p_{c}"] if f"p_{c}" in rows else np.nan
                   for c in conds] for i in range(n_tax)], dtype=float)
    Zm = np.where(E, Z, np.nan)
    ax_heat.imshow(Zm, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")
    # A cell with no comparable-prevalence taxa is hatched, not left blank, so
    # "not measurable here" cannot be read as "no effect".
    for i in range(n_tax):
        for j in range(len(conds)):
            if not E[i, j]:
                ax_heat.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            facecolor="#ffffff",
                                            edgecolor="#bbbbbb",
                                            hatch="///", linewidth=RULE_LW))
                continue
            s = _stars(P[i, j])
            if s:
                # White halo so the stars read on both ends of the colour
                # scale without needing a per-cell luminance test.
                # va="center_baseline" keeps a row of asterisks optically
                # centred; plain "center" sits them high, because an asterisk
                # occupies only the top half of its em box.
                ax_heat.text(j, i, s, ha="center", va="center_baseline",
                             fontsize=FS - 1.5, color="#111111",
                             fontweight="bold",
                             path_effects=[pe.withStroke(linewidth=0.8,
                                                         foreground="white")])
    ax_heat.set_xticks(range(len(conds)))
    ax_heat.set_xticklabels([_short_cond(c) for c in conds], rotation=35,
                            ha="right", fontsize=FS - 1)
    ax_heat.set_yticks([])

    # ── cross-condition mean z with bootstrap interval ──────────────────────
    y = np.arange(n_tax)
    mz = rows["mean_z"].values
    lo, hi = rows["ci_lo"].values, rows["ci_hi"].values
    col = np.where(mz > 0, COL_ENRICH, COL_DEPLET)
    # The 95% bootstrap interval itself, lo to hi. It must NOT be stretched to
    # the zero line: drawn that way every bar touched zero, which is the
    # opposite of what these intervals say.
    ax_loll.hlines(y, lo, hi, color="#b0b0b0", linewidth=RULE_LW, zorder=1)
    ax_loll.scatter(mz, y, s=9, c=col, zorder=3, linewidths=0)
    ax_loll.axvline(0, color="#555555", linewidth=RULE_LW, zorder=2)
    # No overall-FDR mark here. Rows are selected BY clearing FDR, so a mark on
    # the score would sit on almost every row and read as a data column; the
    # strip already carries each list's own verdict, including which of the
    # rows added to reach the row floor did not clear it.
    ax_loll.set_ylim(n_tax - 0.5, -0.5)
    ax_loll.set_yticks([])
    ax_loll.set_xlabel("Mean z across conditions", fontsize=FS)
    ax_loll.tick_params(axis="x", labelsize=FS - 1, pad=1)
    ax_loll.grid(axis="x", alpha=0.15)

    # ── consistency ─────────────────────────────────────────────────────────
    ax_sc.scatter(rows["consistency"].values, y, s=9, c=col, zorder=3,
                  linewidths=0)
    ax_sc.set_xlim(0.45, 1.03)
    ax_sc.set_ylim(n_tax - 0.5, -0.5)
    ax_sc.set_yticks([])
    ax_sc.set_xlabel("Consistency", fontsize=FS)
    ax_sc.tick_params(axis="x", labelsize=FS - 1, pad=1)
    ax_sc.grid(alpha=0.15)

    # ── one title line, with the colour key on it ───────────────────────────
    # Every panel heading and the key share a single y, so the key no longer
    # sits on a line of its own above the headings. set_title would not do
    # this: matplotlib lifts a title clear of whatever is above its axes, so
    # the heatmap's heading rose above the inset key and the four headings
    # stopped being a row.
    title_y = top_f + 0.15 / fig_h

    def _pin(artist, pt, weight="normal"):
        """Set a size the shared type pass will not overwrite.

        savefig_dual reassigns every string by role -- a figure-level text
        counts as "label", at 1.00x base, and a legend entry at 0.86x -- so a
        fontsize given here is normally discarded. `_fixed_pt` is the opt-out
        the type pass honours, and it is what keeps the heading row and the
        key legible rather than letting them settle at body size. The figure
        is drawn at page width, so these are printed points.
        """
        artist.set_fontsize(pt)
        artist.set_fontweight(weight)
        artist._fixed_pt = True
        return artist

    for ax, lab in ((ax_strip, "List"), (ax_loll, "Score"),
                    (ax_sc, "Cons.")):
        p = ax.get_position()
        _pin(fig.text(0.5 * (p.x0 + p.x1), title_y, lab, ha="center",
                      va="center"), TITLE_PT, "bold")
    hp = ax_heat.get_position()
    bar_h = 0.055 / fig_h
    bar_w = 0.26 * (hp.x1 - hp.x0)
    bar_x1 = hp.x1 - 0.055 * (hp.x1 - hp.x0)
    cax = fig.add_axes([bar_x1 - bar_w, title_y - bar_h / 2, bar_w, bar_h])
    cax.imshow(np.linspace(norm.vmin, norm.vmax, 256).reshape(1, -1),
               aspect="auto", cmap=cmap, norm=norm, interpolation="bilinear")
    cax.set_xticks([])
    cax.set_yticks([])
    for sp in cax.spines.values():
        sp.set_linewidth(RULE_LW)
    # The scale's ends sit beside the bar rather than under it, which keeps
    # the whole key on the title line instead of claiming a second one.
    _pin(fig.text(bar_x1 - bar_w - 0.004, title_y, f"{-Z_LIM:g}", ha="right",
                  va="center"), KEY_PT, "bold")
    _pin(fig.text(bar_x1 + 0.004, title_y, f"+{Z_LIM:g}", ha="left",
                  va="center"), KEY_PT, "bold")
    _pin(fig.text(hp.x0, title_y, "Per-condition z", ha="left", va="center"),
         TITLE_PT, "bold")

    # ── legend, right-aligned under the last panel ──────────────────────────
    # Anchored to the figure's right margin so its right edge finishes with
    # the rightmost panel, and placed INSIDE the bottom band already reserved
    # for the axis labels, so it costs the figure no extra height.
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=C_SIG, edgecolor="#9a9a9a",
                  linewidth=RULE_LW, label=f"on list, q < {q_cut:g}"),
        Line2D([], [], marker=r"$\ast$", markersize=3.6, linestyle="none",
               color="#3a3a3a", label="q < 0.05"),
        Rectangle((0, 0), 1, 1, facecolor=C_MEMBER, edgecolor="#9a9a9a",
                  linewidth=RULE_LW, label="on list, n.s."),
        Rectangle((0, 0), 1, 1, facecolor=C_ABSENT, edgecolor="#9a9a9a",
                  linewidth=RULE_LW, label="not on list"),
        # Plain asterisks, not mathtext: "$***$" sets three maths operators
        # and renders them spaced out as "** *".
        Line2D([], [], marker=r"$\ast$", markersize=3.6, linestyle="none",
               color="#111111",
               label="within condition: * p<.05  ** p<.01  *** p<.001"),
        Rectangle((0, 0), 1, 1, facecolor="#ffffff", edgecolor="#bbbbbb",
                  hatch="///", linewidth=RULE_LW, label="no comparable taxa"),
    ]
    # One row, so the band under the axis labels holds it without growing.
    # bbox_transform is given explicitly: without it the anchor is read
    # against the last axes rather than the figure, which left the legend
    # short of the right-hand panel instead of finishing with it.
    leg = fig.legend(handles=handles, loc="lower right",
                     bbox_to_anchor=(_m["right"], LEGEND_IN / fig_h),
                     bbox_transform=fig.transFigure, ncol=len(handles),
                     frameon=False, handlelength=1.0, handleheight=0.9,
                     columnspacing=0.9, handletextpad=0.4,
                     borderpad=0.0, borderaxespad=0.0)
    for t in leg.get_texts():
        _pin(t, LEGEND_PT)

    safe = family.replace("+", "plus").replace(" ", "_")
    out = os.path.join(outdir,
                       f"health_taxon_drivers_{safe}_{ref}_{spec_tag}.png")
    savefig_dual(fig, out, dpi=SUBMISSION_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out}")
    return out


def emit_and_draw(driver_df, taxonomy_map, outdir, conditions, ref="combined",
                  spec_tag="10pct", spec_desc="", figure_data_dir=None):
    """Draw all four figures and write their data objects.

    `outdir` is the section's `main` directory. S28 and S29 are written to its
    `supplement` sibling instead, because that is where
    tools/additional_figures_manifest.txt resolves those two slots -- a figure
    written to the wrong one is simply not found, and the slot reports "no
    data object" on a redraw.
    """
    from . import figure_source
    supplement = os.path.join(os.path.dirname(os.path.normpath(outdir)),
                              "supplement")
    paths = {}
    for family, (slot, lists) in FAMILIES.items():
        lists_present = [l for l in lists if l in set(driver_df["Taxon_list"])]
        if not lists_present:
            continue
        where = outdir if SLOT_DIR.get(slot, "main") == "main" else supplement
        os.makedirs(where, exist_ok=True)
        tbl = driver_df[driver_df["Taxon_list"].isin(lists_present)].copy()
        tbl.insert(2, "taxonomy",
                   [taxonomy_map.get(t, "") for t in tbl["taxon"]])
        tbl.insert(3, "taxon_label", _labels(tbl["taxon"], taxonomy_map))
        if figure_data_dir is not None and spec_tag == "10pct":
            figure_source.emit(
                where, slot, {"drivers": tbl.reset_index(drop=True)},
                stem=("health_taxon_drivers_"
                      + family.replace("+", "plus").replace(" ", "_")
                      + f"_{ref}_{spec_tag}"),
                reference=ref,
                extras={"drawn_conditions": list(conditions),
                        "family": family, "lists": lists_present,
                        "q_cut": Q_CUT},
                notes=(f"{family}: {', '.join(lists_present)}. "
                       f"Cluster bootstrap over study cohorts on the four "
                       f"focal contrasts. {spec_desc}"))
        paths[slot] = draw(driver_df, taxonomy_map, family, where,
                           conditions, ref=ref, spec_tag=spec_tag,
                           spec_desc=spec_desc)
    return paths
