#!/usr/bin/env python
"""
list_comparison_panels.py
-------------------------
The three head-to-head panels of main-text Figure 5: do the derived lists
(VANISH+/BloSSUM+ and the health-contrast VANISHH+/BloSSUMH+) separate cases
from controls better than the original VANISH and BloSSUM lists?

  F  VANISH family   -- mean z per condition, VANISH vs VANISH+ vs VANISHH+
  G  BloSSUM family  -- the same for BloSSUM / BloSSUM+ / BloSSUMH+
  H  every original-list value against its counterparts, on equal axes

These were supplementary figure S31 until 2026-10-14, when the panels moved
into Figure 5 and the standalone figure was retired. What is left is a
library: `figures/figure5.py` is the only caller, and there is no CLI, no
figure of its own and no data object -- Figure 5's own object carries the two
frames. The retired script is in the session archive if it is ever wanted back.

Panels F and G print the primary 10% breadth rule, so the star beside a marker
belongs to the interval drawn with it (Supplementary Text 6G). Panel H is the
whole claim in one view: a point above the identity line is a condition where
the derived list sits further from its prevalence background than the original
list does.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The repo root before the first `persistence` import: run by path,
# sys.path[0] is `figures/`. See figure2.py for the full note.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from persistence import config

#: Condition order, matching Figure 5D-E and Fig. S25 so a reader moving between
#: them is not re-reading the axis.
#:
#: The pooled "Unhealthy" contrast used to sit last here and is now dropped via
#: config.HEALTH_PLOT_SKIP, because its cases ARE the cases of the other four:
#: the paired test below was counting a replicate of them as a fifth
#: independent pair. The n it runs on therefore falls from 10 pairs (5
#: conditions x 2 families) to 8, and both Wilcoxon p-values survive that --
#: VANISH+/BloSSUM+ 0.0020 -> 0.0078, VanishH+/BlossumH+ 0.0098 -> 0.0391.
#: Note 0.0078 is 2/2^8, the smallest value the test can return at n = 8, so
#: the + family saturates it: a smaller p is unavailable, not absent.
#:
#: `panels_from_csv` returns {} when a name here is missing from the data, so
#: this must list only contrasts the run actually produces.
_ALL_CONDITIONS = ["Obese", "Glucose Intolerance", "IBD", "CRC / Adenoma",
                   "Unhealthy"]
CONDITIONS = [c for c in _ALL_CONDITIONS
              if c not in set(getattr(config, "HEALTH_PLOT_SKIP", ()))]

#: (original, primary derived, health-contrast derived) per family.
FAMILIES = {
    "VANISH":  ("VANISH",  "VANISH+",  "VanishH+"),
    "BloSSUM": ("BloSSUM", "BloSSUM+", "BlossumH+"),
}

#: Row offsets inside one condition band, the original list at the top.
OFFSETS = (0.26, 0.0, -0.26)

FS = 8.0


def _display(s: str) -> str:
    return config.display_text(str(s))


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def panels_from_csv(path: str) -> dict:
    """The same panels from the Table S12 source, by path, or {} if it cannot.

    Figure 5 draws these as F-H and is handed the section3 `main` directory
    directly, so it cannot use `load_table`'s results-tree layout -- and a
    missing or short table there must skip the block rather than take the
    whole main-text figure down with a SystemExit.
    """
    if not os.path.exists(path):
        return {}
    try:
        d = pd.read_csv(path)
        if "eligible" in d.columns:
            d = d[d["eligible"]].copy()
        d = d[d["Condition"].isin(CONDITIONS)].copy()
        if [c for c in CONDITIONS if c not in set(d["Condition"])]:
            return {}
        return tidy_panels(d.set_index(["Condition", "Taxon_list"]))
    except Exception as e:                              # pragma: no cover
        print(f"  [F-H] list comparison unavailable: {e}")
        return {}


def tidy_panels(d: pd.DataFrame) -> dict:
    """The two tidy frames the figure draws, and that its data object ships."""
    rows = []
    for cond in CONDITIONS:
        for family, lists in FAMILIES.items():
            for role, name in zip(("original", "plus", "health"), lists):
                r = d.loc[(cond, name)]
                rows.append({
                    "Condition": cond,
                    "Family": family,
                    "Role": role,
                    "Taxon_list": name,
                    "mean_z": float(r["mean_z_10pct"]),
                    "ci_lo": float(r["z_lo_10pct"]),
                    "ci_hi": float(r["z_hi_10pct"]),
                    "p": float(r["p_meanz_10pct"]),
                    "n_supported": int(r["n_supported_10pct"]),
                    "n_list": int(r["n_list_10pct"]),
                })
    per_cell = pd.DataFrame(rows)

    # Panel C's pairing: each original list against its two derived partners.
    pairs = []
    for cond in CONDITIONS:
        for family, (orig, plus, hlt) in FAMILIES.items():
            base = d.loc[(cond, orig)]
            for role, name in (("plus", plus), ("health", hlt)):
                r = d.loc[(cond, name)]
                pairs.append({
                    "Condition": cond, "Family": family, "Role": role,
                    "original_list": orig,
                    "original_abs_z": abs(float(base["mean_z_10pct"])),
                    "original_p": float(base["p_meanz_10pct"]),
                    "derived_list": name,
                    "derived_abs_z": abs(float(r["mean_z_10pct"])),
                    "derived_p": float(r["p_meanz_10pct"]),
                })
    return {"per_cell": per_cell, "paired": pd.DataFrame(pairs)}


def _summary(paired: pd.DataFrame, per_cell: pd.DataFrame) -> dict:
    """The comparison's own statistics, including a test of it.

    The question this figure answers is whether the derived lists sit further
    from their prevalence background than the original lists do *systematically*,
    not whether any one condition does. So the test is a paired one across the
    ten condition-by-family comparisons, not ten separate tests: a per-pair
    interval would also have to assume the two estimates are independent, and
    they are not -- the lists overlap in membership and are scored against the
    same background, which correlates them positively and makes any such
    interval conservative by an unknown amount.
    """
    from scipy import stats

    emp = paired[paired["Role"] == "plus"]
    hlt = paired[paired["Role"] == "health"]
    sig = lambda df: int((df["p"] < 0.05).sum())

    delta = (emp["derived_abs_z"].to_numpy() - emp["original_abs_z"].to_numpy())
    delta_h = (hlt["derived_abs_z"].to_numpy() - hlt["original_abs_z"].to_numpy())
    out = {
        "n_pairs": len(emp),
        "plus_exceeds": int((delta > 0).sum()),
        "health_exceeds": int((hlt["derived_abs_z"].to_numpy()
                               > hlt["original_abs_z"].to_numpy()).sum()),
        "sig_original": sig(per_cell[per_cell["Role"] == "original"]),
        "sig_plus": sig(per_cell[per_cell["Role"] == "plus"]),
        "sig_health": sig(per_cell[per_cell["Role"] == "health"]),
        "rescued": int(((emp["original_p"] >= 0.05) & (emp["derived_p"] < 0.05)).sum()),
        "lost": int(((emp["original_p"] < 0.05) & (emp["derived_p"] >= 0.05)).sum()),
        "mean_delta": float(np.mean(delta)),
        "mean_delta_health": float(np.mean(delta_h)),
    }
    # Both families of derived list are tested, not just the primary one: the
    # panel draws + and H+ together, and a claim made for the pair has to be
    # supported for the pair. They do not give the same answer -- H+ ties the
    # original once, in IBD -- so one p-value standing for both would be an
    # overstatement.
    for key, d in (("", delta), ("_health", delta_h)):
        try:
            w = stats.wilcoxon(d, alternative="two-sided")
            out[f"wilcoxon_p{key}"] = float(w.pvalue)
            out[f"wilcoxon_stat{key}"] = float(w.statistic)
        except Exception:                               # degenerate input
            out[f"wilcoxon_p{key}"] = float("nan")
            out[f"wilcoxon_stat{key}"] = float("nan")
    out["sign_p"] = float(2 * 0.5 ** len(delta)) if out["plus_exceeds"] == len(delta) \
        else float(2 * stats.binom.sf(out["plus_exceeds"] - 1, len(delta), 0.5))
    return out




def _scale_axes(ax, scale):
    """Spines and tick marks in the same printed units as the text.

    Figure 5 draws these three panels as F-H on a 24-inch canvas that is
    reduced to the column width when it is saved, so every size has to carry
    that magnification or it prints at a third of what S31 gets. At scale 1.0
    this is a no-op, which is what keeps S31 itself unchanged.
    """
    if scale == 1.0:
        return
    for sp in ax.spines.values():
        sp.set_linewidth(0.8 * scale)
    ax.tick_params(width=0.8 * scale, length=3.5 * scale)


def _family_panel(ax, per_cell, family, show_ylabels, scale=1.0):
    """One family's three lists, per condition, on a square panel."""
    fs = FS * scale
    lists = FAMILIES[family]
    colours = config.CMH_LIST_COLORS
    sub = per_cell[per_cell["Family"] == family]

    for i, cond in enumerate(CONDITIONS):
        y0 = len(CONDITIONS) - 1 - i
        band = sub[sub["Condition"] == cond]
        zs = [float(band[band["Taxon_list"] == n]["mean_z"].iloc[0]) for n in lists]
        ax.plot(zs, [y0 + o for o in OFFSETS], color="#C6C6C6", lw=0.7 * scale,
                zorder=1, solid_capstyle="round")
        for name, off in zip(lists, OFFSETS):
            r = band[band["Taxon_list"] == name].iloc[0]
            y, col = y0 + off, colours.get(name, "#666666")
            ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], color=col, lw=1.4 * scale,
                    solid_capstyle="butt", zorder=2)
            ax.plot([r["mean_z"]], [y], marker="o", ms=4.2 * scale, color=col,
                    markeredgecolor="black", markeredgewidth=0.4 * scale,
                    zorder=3)
            s = _stars(r["p"])
            if s:
                # Beside the interval, not above it: above put the stars on the
                # neighbouring condition's interval in the BloSSUM panel, where
                # the bands sit close together.
                ax.text(r["ci_hi"], y, "  " + s, fontsize=fs - 1.6 * scale,
                        va="center", ha="left", color="black")
        # No separator rule between condition bands. The three-marker grouping
        # and the y-tick already read as a band, and a full-width rule put a
        # data artist under the legend in both family panels, which the layout
        # check then reported as a 100% occlusion.

    ax.axvline(0, color="black", lw=0.8 * scale, zorder=1)
    _scale_axes(ax, scale)
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_yticklabels([_display(c) for c in reversed(CONDITIONS)]
                       if show_ylabels else [""] * len(CONDITIONS),
                       fontsize=fs - 0.5 * scale)
    ax.set_ylim(-0.62, len(CONDITIONS) - 0.38)
    ax.set_xlabel("mean z", fontsize=fs - 0.3 * scale, labelpad=2 * scale)
    ax.tick_params(axis="x", labelsize=fs - 1.0 * scale, pad=1.5 * scale)
    ax.tick_params(axis="y", length=0, pad=2 * scale)
    ax.set_title(f"{_display(family)} family", fontsize=fs + 0.4 * scale,
                 fontweight="bold",
                 color=colours.get(family, "black"), pad=4 * scale)

    # Room for the stars and the intervals; a touch more on the right, which is
    # where the legend sits in both panels.
    lo = min(sub["ci_lo"].min(), 0.0)
    hi = max(sub["ci_hi"].max(), 0.0)
    span = hi - lo
    ax.set_xlim(lo - 0.08 * span, hi + 0.30 * span)
    ax.set_box_aspect(1)

    # Named directly: the manuscript calls these the original VANISH and
    # BloSSUM lists, and the derived ones by their own names. No category word.
    labels = [f"original {_display(lists[0])}"] + [_display(n) for n in lists[1:]]
    handles = [plt.Line2D([], [], marker="o", ms=4.2 * scale, lw=1.4 * scale,
                          color=colours.get(n, "#666"), markeredgecolor="black",
                          markeredgewidth=0.4 * scale, label=lab)
               for n, lab in zip(lists, labels)]
    # Which right-hand corner is free is a property of the family, not of the
    # layout: VANISH reaches furthest at the top of the panel and BloSSUM at
    # the bottom, so one fixed corner draws the legend over the data in
    # whichever of the two it does not suit.
    _reach = lambda c: float(sub[sub["Condition"] == c]["ci_hi"].max())
    loc = ("lower right" if _reach(CONDITIONS[-1]) <= _reach(CONDITIONS[0])
           else "upper right")
    ax.legend(handles=handles, loc=loc, frameon=False, borderpad=0.1,
              handlelength=1.1, labelspacing=0.22, handletextpad=0.45,
              borderaxespad=0.15,
              prop=dict(size=fs - 2.0 * scale, weight="bold"))


def _paired_panel(ax, paired, summary, scale=1.0, rich_labels=False):
    """Every original-list value against its two derived counterparts.

    `rich_labels` sets the list names in the axis labels in their own colours,
    which is what lets this panel be read without the key beside it. Figure 5H
    uses it; S31 keeps the plain labels, since its panels A and B carry the
    full key two inches away.
    """
    fs = FS * scale
    colours = config.CMH_LIST_COLORS
    # Headroom above the highest point, so the summary sits on empty canvas
    # rather than across the BloSSUM+ unhealthy marker.
    hi = max(paired["original_abs_z"].max(), paired["derived_abs_z"].max())
    lim = hi * 1.34

    ax.fill_between([0, lim], [0, lim], [lim, lim], color="#2F6B4F", alpha=0.05,
                    zorder=0, lw=0)
    ax.plot([0, lim], [0, lim], color="#8A8A8A", lw=0.9 * scale, ls=(0, (4, 3)),
            zorder=1)

    for role, marker, ms in (("plus", "o", 4.6), ("health", "^", 5.0)):
        sub = paired[paired["Role"] == role]
        ax.scatter(sub["original_abs_z"], sub["derived_abs_z"], marker=marker,
                   s=(ms * scale) ** 2,
                   c=[colours.get(n, "#666") for n in sub["derived_list"]],
                   edgecolors="black", linewidths=0.4 * scale, zorder=3)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_box_aspect(1)
    _scale_axes(ax, scale)
    if rich_labels:
        from persistence.plot_utils import rich_axis_label
        _lfs = fs - 0.3 * scale
        # Each axis names its lists in their own colours, and the derived axis
        # names both families' members in order -- "+/+" is VANISH+ then
        # BloSSUM+ -- so the reader can tell which marker belongs to which
        # list without crossing to another panel for the key.
        ax.set_xlabel(""); ax.set_ylabel("")
        rich_axis_label(ax, [[("|mean z|,  original ", None, _lfs),
                              ("VANISH", colours["VANISH"], _lfs),
                              ("/", None, _lfs),
                              ("BloSSUM", colours["BloSSUM"], _lfs)]],
                        axis="x", gap=0.035)
        # Two lines, not one: set on a single line this label is taller than
        # the square panel it belongs to and runs up into the title.
        rich_axis_label(ax, [[("|mean z|,", None, _lfs)],
                             [("+", colours["VANISH+"], _lfs),
                              ("/", None, _lfs),
                              ("+", colours["BloSSUM+"], _lfs),
                              (" and ", None, _lfs),
                              ("H+", colours["VanishH+"], _lfs),
                              ("/", None, _lfs),
                              ("H+", colours["BlossumH+"], _lfs),
                              (" lists", None, _lfs)]],
                        axis="y", gap=0.03, sep=_lfs * 0.3)
    else:
        ax.set_xlabel("|mean z|,  original VANISH/BloSSUM",
                      fontsize=fs - 0.3 * scale, labelpad=2 * scale)
        ax.set_ylabel("|mean z|,  + and H+ lists", fontsize=fs - 0.3 * scale,
                      labelpad=2 * scale)
    ax.tick_params(labelsize=fs - 1.0 * scale, pad=1.5 * scale)
    ax.set_title("Paired comparison", fontsize=fs + 0.4 * scale,
                 fontweight="bold", pad=4 * scale)

    # Both rows, because they differ: the + lists clear the originals in every
    # comparison and the H+ lists in all but one. The test is named in the
    # caption rather than here, which buys the room for the second line.
    _p = lambda v: ("p < 0.001" if v < 0.001 else f"p = {v:.3f}"
                    if np.isfinite(v) else "")
    ax.text(0.045, 0.965,
            f"+ lists  {summary['plus_exceeds']}/{summary['n_pairs']} above,  "
            f"{_p(summary.get('wilcoxon_p', float('nan')))}\n"
            f"H+ lists  {summary['health_exceeds']}/{summary['n_pairs']} above,  "
            f"{_p(summary.get('wilcoxon_p_health', float('nan')))}",
            transform=ax.transAxes, fontsize=fs - 1.6 * scale, va="top",
            ha="left", color="#2F6B4F", fontweight="bold", linespacing=1.3)

    handles = [
        plt.Line2D([], [], marker="o", ms=4.6 * scale, ls="none", color="#8C8C8C",
                   markeredgecolor="black", markeredgewidth=0.4 * scale,
                   label="+ lists"),
        plt.Line2D([], [], marker="^", ms=5.0 * scale, ls="none", color="#8C8C8C",
                   markeredgecolor="black", markeredgewidth=0.4 * scale,
                   label="H+ lists"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, borderpad=0.1,
              handlelength=1.0, labelspacing=0.22, handletextpad=0.45,
              borderaxespad=0.15, prop=dict(size=fs - 2.0 * scale, weight="bold"))
