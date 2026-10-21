#!/usr/bin/env python3
"""
make_metadata_summary_eqearth.py
================================
Companion to make_metadata_summary.py: re-renders the two world maps using
the Equal Earth projection (Šavrič, Patterson & Jenny, 2018) instead of
the default Plate Carrée. Equal Earth is a strict equal-area projection,
so the visual area of every country is proportional to its true area —
essential for a study explicitly framed around global microbial diversity,
where the default projection systematically inflates high-latitude
landmasses (Russia, Canada, Greenland) and shrinks the equatorial regions
that hold most H-G and Transitional samples.

Antarctica is cropped to give more room to inhabited continents.

All preprocessing and shared plotting helpers are imported from the sibling
script `make_metadata_summary.py` — keep both files in the same directory.

Outputs (in --out-dir, default ./metadata_summary/):
  metagenome_world_map_A_eqearth_full.png/pdf
  metagenome_world_map_B_eqearth_full.png/pdf

Usage
-----
  python make_metadata_summary_eqearth.py \
      --meta       metadata_Mar_16_2026.csv \
      --secondary  Chosen_population_samples.csv \
      --out-dir    ./metadata_summary/
"""

import argparse
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import box

# ── Re-use everything from the sibling script ────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from make_metadata_summary import (
    load_and_preprocess,
    _load_world,
    _build_country_grp,
    _make_lifestyle_facecolors,
    _draw_mixed_hatches,
    _add_lifestyle_colorbars,
    _make_circle_legend,
    _scale_bubbles,
    _fmt_sci,
    _MAP_LIFESTYLE_COLORS,
    _apply_display_text,
)

# ─── Display-name mapping ───────────────────────────────────────────────────
# Internal lifestyle codes ("H-G") stay unchanged everywhere — they're tied
# to the preprocessing pipeline and the colour dictionary. We only override
# what the *viewer* sees on legends and colorbars.
_LIFESTYLE_DISPLAY = {
    "Industrial":   "Industrial",
    "Transitional": "Agro-Pastoralist",
    "Agro-Pastoralist": "Agro-Pastoralist",
    "Agro-Pastorialist": "Agro-Pastoralist",
    "H-G":          "Hunter-Gatherer",
}

# Wrap the imported colorbar helper so the H-G bar's label reads
# "Hunter-Gatherer". The original function builds labels as
#   f"{ls}\n{title_text}"  where ls ∈ {"Industrial","Transitional","H-G"},
# so after it runs we walk the figure's axes and rewrite any ylabel whose
# first line is an internal code we want to rename.
#: Where the lifestyle colourbars sit, in figure fractions. Measured from the
#: render: the map's ink stops at ~0.11 and the first landmass begins at
#: ~0.163, so the bars and their rotated labels go in that gap.
_CB_INSET_X = 0.142

#: Anything whose projected box reaches west of this and east of its mirror has
#: been split by the antimeridian clip, so its western piece is a duplicate.
_AM_SLIVER_X = -1.60e7

_orig_add_lifestyle_colorbars = _add_lifestyle_colorbars

def _add_lifestyle_colorbars(fig, norms, title_text, fontsize=11, fixed_ticks=None):
    _orig_add_lifestyle_colorbars(fig, norms, title_text, fontsize, fixed_ticks)
    renamed_axes = []
    for ax in fig.axes:
        ylabel = ax.get_ylabel()
        if not ylabel:
            continue
        first_line, _, rest = ylabel.partition("\n")
        if first_line in _LIFESTYLE_DISPLAY and _LIFESTYLE_DISPLAY[first_line] != first_line:
            ax.set_ylabel(f"{_LIFESTYLE_DISPLAY[first_line]}\n{rest}",
                          fontsize=fontsize, labelpad=18, linespacing=1.18)
            renamed_axes.append(ax)

    # The manuscript label "Agro-Pastoralist" is longer than the historical
    # internal "Transitional" code, so the three vertical colorbar labels need
    # a little more breathing room in the Equal Earth layout.
    if len(renamed_axes) == 2:
        colorbar_axes = [
            ax for ax in fig.axes
            if ax.get_ylabel().split("\n", 1)[0]
            in {"Industrial", "Agro-Pastoralist", "Hunter-Gatherer"}
        ][-3:]
        if len(colorbar_axes) == 3:
            # Sit the three bars inside the frame, over the empty Pacific west
            # of the Americas, rather than in a margin outside the map. The
            # margin they used to occupy is then trimmed by the tight crop, so
            # the same printed width buys a larger map and larger type.
            for ax, bottom in zip(colorbar_axes, [0.56, 0.32, 0.08]):
                ax.set_position([_CB_INSET_X, bottom, 0.014, 0.155])

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")

# Equal Earth: EPSG:8857 is the official code (PROJ ≥ 6.3, 2019+).
EQEARTH_EPSG = "EPSG:8857"
EQEARTH_PROJ_FALLBACK = "+proj=eqearth +datum=WGS84 +units=m +no_defs"

# Antimeridian-safe clip box (keeps everything just inside ±180° so
# polygons that touch the date line don't smear horizontally when reprojected).
_AM_CLIP = box(-179.99, -90.0, 179.99, 90.0)


# ─────────────────────────────────────────────────────────────────────────────
#  World layer: load → drop Antarctica → trim antimeridian → reproject
# ─────────────────────────────────────────────────────────────────────────────

def _load_world_eqearth():
    """Natural Earth countries, Antarctica removed, projected to Equal Earth."""
    world = _load_world()  # WGS84 lat/lon, with name_norm column

    # Drop Antarctica
    if "continent" in world.columns:
        world = world[world["continent"] != "Antarctica"].copy()
    elif "name" in world.columns:
        world = world[~world["name"].astype(str)
                      .str.contains("Antarctica", case=False, na=False)].copy()

    # Trim everything to just inside ±180° before reprojection
    world["geometry"] = world.geometry.intersection(_AM_CLIP)
    world = world[~world.geometry.is_empty & world.geometry.notna()].copy()

    # Reproject to Equal Earth
    try:
        world = world.to_crs(EQEARTH_EPSG)
    except Exception:
        world = world.to_crs(EQEARTH_PROJ_FALLBACK)

    return world


# ─────────────────────────────────────────────────────────────────────────────
#  Shared plot body — A and B differ only in what drives shading vs. bubbles
# ─────────────────────────────────────────────────────────────────────────────

def _plot_world_map_eqearth(meta, out_stem, *,
                            value_col, value_label,
                            bubble_col, bubble_label,
                            bubble_fmt, bubble_refs, bubble_base,
                            hatch_value_cols,
                            cbar_fixed_ticks,
                            title,
                            grp=None,
                            show_title=True):
    # A missing geopandas used to print "[SKIP]" and return normally, so the
    # script exited 0 and a run "completed" with no Figure 1 (or a stale one
    # left in place). A map that cannot be drawn is a failure: raise.
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            f"ERROR {os.path.basename(out_stem)}: geopandas is not installed, so the "
            f"world map cannot be drawn. Install requirements-pinned.txt "
            f"(geopandas==1.0.1) and rerun; `python tools/preflight.py` checks this."
        ) from exc

    world = _load_world_eqearth()
    # `grp` may be handed in from a published data object, which is how the
    # map is redrawn without the metadata file.
    grp   = _build_country_grp(meta) if grp is None else grp

    # The figure's data object: the per-country table the map is drawn from.
    # The country outlines are a basemap, not data -- they come from the
    # bundled Natural Earth geometry and are the same for any figure that
    # uses this projection -- so what is published here is every number the
    # map encodes: which lifestyles are present, the shading value and the
    # bubble area, per country.
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from persistence import figure_source as _fsrc
        _slot = "F1" if os.path.basename(out_stem).startswith("figure1") else None
        if _slot:
            _fsrc.emit(
                os.path.dirname(out_stem) or ".", _slot,
                {"countries": grp},
                stem=os.path.basename(out_stem),
                reference="metadata",
                extras={"value_col": value_col, "bubble_col": bubble_col,
                        "bubble_base": bubble_base,
                        "bubble_refs": list(bubble_refs or []),
                        "hatch_value_cols": list(hatch_value_cols or []),
                        "cbar_fixed_ticks": cbar_fixed_ticks,
                        "title": title},
                notes=("Main Figure 1. One row per country: dominant "
                       "lifestyle, how many lifestyles are present, the "
                       "shading value and the bubble value. Country outlines "
                       "are the bundled Equal-Earth basemap, not data."))
    except Exception as _exc:
        print(f"  [data] could not write the Figure 1 data object: {_exc}")

    unmatched = sorted(set(grp["country_norm"]) - set(world["name_norm"]))
    if unmatched:
        print(f"  [{os.path.basename(out_stem)}] Unmatched ({len(unmatched)}): {unmatched}")

    merged = world.merge(grp, how="left", left_on="name_norm", right_on="country_norm")
    # Centroids in projected metres — must be computed AFTER reprojection
    merged["centroid"] = merged.geometry.representative_point()

    dom        = merged["dominant_lifestyle"].fillna("NA")
    values     = merged[value_col].fillna(np.nan).to_numpy()
    bubbles    = merged[bubble_col].fillna(0).to_numpy(dtype=float)
    mixed_mask = (merged["n_lifestyles_present"].fillna(0) >= 2).to_numpy()

    face, norms, _, _ = _make_lifestyle_facecolors(merged, values, dom)
    bubble_max = bubbles.max() if bubbles.max() > 0 else 1.0
    sizes = np.array([_scale_bubbles(x, bubble_max, bubble_base) for x in bubbles])

    fig, ax = plt.subplots(figsize=(14, 6.4), dpi=150)

    # Base: every country in pale grey
    merged.plot(ax=ax, color="#f5f5f5", edgecolor="#666666", linewidth=0.3)

    # Single-lifestyle countries: flat fill from the white→colour cmap
    nm = ~mixed_mask
    merged[nm].plot(
        ax=ax,
        color=[c for c, k in zip(face, nm) if k],
        edgecolor="#666666", linewidth=0.3,
    )

    # Mixed-lifestyle countries: primary fill + secondary hatch overlay
    _draw_mixed_hatches(ax, merged, norms, hatch_value_cols)

    # Bubbles
    has_bubble = bubbles > 0
    sel = merged.loc[has_bubble]
    ax.scatter(
        sel["centroid"].x, sel["centroid"].y,
        s=sizes[has_bubble],
        facecolor="none", edgecolor="#222", linewidth=0.9, alpha=0.88,
    )

    # Per-lifestyle colorbars (figure-relative coordinates — projection-agnostic)
    _add_lifestyle_colorbars(fig, norms, value_label, 11, cbar_fixed_ticks)

    # Lifestyle legend
    ax.legend(handles=[
        Line2D([0], [0], marker="s", color="w",
               label=_LIFESTYLE_DISPLAY.get(ls, ls),
               markerfacecolor=_MAP_LIFESTYLE_COLORS[ls], markersize=12)
        for ls in ["Industrial", "Transitional", "H-G"]
    ], loc="lower left", frameon=True, fontsize=11)

    # Bubble-size legend
    _make_circle_legend(ax, bubbles, bubble_label, "lower right",
                        bubble_fmt, bubble_refs, base=bubble_base)
    try:
        ax.get_legend().set_bbox_to_anchor((0.75, 0.07))
    except Exception:
        pass

    # Fit to the projected extent of the (Antarctica-less) world.
    #
    # Fiji straddles the antimeridian, so the ±180° clip leaves a sliver of it
    # at the far *west* of the projection as well as the main islands in the
    # east. That fragment is a few pixels wide at print size but it sets the
    # western edge of the map, holding roughly 5% of the figure's width for
    # something no reader can see -- and it is a duplicate: Fiji is drawn,
    # shaded and bubbled on the eastern side. Dropping the fragment lets the
    # map fill the column, in the same spirit as the Antarctica crop the title
    # already declares.
    xmin, ymin, xmax, ymax = merged.total_bounds
    _wrapped = merged[(merged.bounds["minx"] < _AM_SLIVER_X)
                      & (merged.bounds["maxx"] > -_AM_SLIVER_X)]
    if len(_wrapped):
        _mains = merged[merged.bounds["minx"] >= _AM_SLIVER_X]
        if len(_mains):
            xmin = float(_mains.bounds["minx"].min())
            print(f"  [extent] western antimeridian sliver dropped "
                  f"({', '.join(sorted(_wrapped['name'].astype(str))[:3])}); "
                  f"map starts at {xmin:,.0f} instead of "
                  f"{merged.total_bounds[0]:,.0f}")
    pad_x = (xmax - xmin) * 0.01
    pad_y = (ymax - ymin) * 0.02
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal")  # critical: preserves equal-area property visually
    ax.axis("off")

    # Figure 1 is published without a title -- the manuscript caption carries
    # it; the metadata-summary maps keep this descriptive one.
    if show_title:
        ax.set_title(f"{title}\nEqual Earth projection (equal-area, Antarctica cropped)",
                     pad=12)

    plt.tight_layout()
    _apply_display_text(fig)
    # Reduce to the printed column width here rather than leaving it to the
    # page. The layout is untouched -- axes sit at figure fractions -- and
    # everything specified in points is scaled with the canvas, so what is
    # written is the figure as drawn, photographically reduced.
    from persistence.figure_rescale import save_column_width
    save_column_width(fig, out_stem)
    for ext in (".png", ".pdf"):
        print(f"  Saved: {out_stem + ext}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Public per-map entry points — identical knobs to the originals in
#  make_metadata_summary.py, just routed through the Equal-Earth body.
# ─────────────────────────────────────────────────────────────────────────────

def plot_world_map_A_eqearth(meta, out_stem):
    """Map A: hue=Lifestyle • log-shading=avg bases/sample • circles=# samples."""
    _plot_world_map_eqearth(
        meta, out_stem,
        value_col        = "avg_bases_per_sample",
        value_label      = "bases\nper sample",
        bubble_col       = "total_samples",
        bubble_label     = "# samples",
        bubble_fmt       = "{:,.0f}",
        bubble_refs      = [10, 30, 100, 300, 1000],
        bubble_base      = 65.0,
        hatch_value_cols = {
            "Industrial":   "avg_bases_per_sample",
            "Transitional": "avg_bases_per_sample",
            "H-G":          "avg_bases_per_sample",
        },
        cbar_fixed_ticks = [3e9, 1e10, 3e10],
        title            = ("Map A — Hue: Lifestyle  •  "
                            "Log-shading: Avg bases per sample  •  "
                            "Circles: # samples"),
    )


def plot_world_map_B_eqearth(meta, out_stem, grp=None, show_title=True):
    """Map B: hue=Lifestyle • log-shading=# samples • circles=total bases."""
    _plot_world_map_eqearth(
        meta, out_stem,
        grp              = grp,
        show_title       = show_title,
        value_col        = "total_samples",
        value_label      = "# samples",
        bubble_col       = "total_bases",
        bubble_label     = "Total bases",
        bubble_fmt       = _fmt_sci,
        bubble_refs      = [1e11, 3e11, 1e12, 3e12, 1e13],
        bubble_base      = 75.0,
        hatch_value_cols = {
            "Industrial":   "l_ind",
            "Transitional": "l_tran",
            "H-G":          "l_hg",
        },
        cbar_fixed_ticks = None,
        title            = ("Map B — Hue: Lifestyle  •  "
                            "Log-shading: # samples  •  "
                            "Circles: Total bases (country sum)"),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _default_meta() -> str:
    """Metadata file from config, so a repointed cohort reaches this script.

    This was hard-coded to metadata_Mar_16_2026.csv, which silently kept the
    pre-2026-08-20 cohort while the analysis ran on the newer metadata.
    """
    import persistence.config as _c
    return _c.META_FILE


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", default=_default_meta(),
                    help="Input metadata CSV (default: config.META_FILE)")
    ap.add_argument("--secondary", default=None,
                    help="Optional secondary-run BC screening CSV")
    ap.add_argument("--out-dir", dest="out_dir", default="metadata_summary",
                    help="Output directory (default: metadata_summary/)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    meta = load_and_preprocess(args.meta, args.secondary)

    print("\nGenerating equal-area world maps ...")
    plot_world_map_A_eqearth(
        meta, os.path.join(args.out_dir, "metagenome_world_map_A_eqearth_full"))
    plot_world_map_B_eqearth(
        meta, os.path.join(args.out_dir, "metagenome_world_map_B_eqearth_full"))

    print("\nDone.")


if __name__ == "__main__":
    main()
