"""
Section 1 (tech): Within-sample technical replicate robustness.

For every primary sample that has one or more secondary runs listed in
``Secondary_run_IDs``, and whose secondary run rows are present in the
breadth/depth matrices, this module computes:

  1. Per-pair alpha-diversity comparison
       - Richness and Shannon on primary vs each secondary run
       - Fractional deviation:  |sec - pri| / pri
  2. Within-sample Bray-Curtis and Jaccard distances
       - Compared to the between-sample distribution from within each
         lifestyle (reference background from section1_within_variation)
  3. Summary statistics and ICC (intraclass correlation) for richness
  4. Diagnostic plots:
       a. Scatter: primary richness vs secondary richness, coloured by lifestyle
       b. Violin: fractional richness deviation by lifestyle
       c. Density overlay: within-sample BC vs within-lifestyle BC
       d. Heatmap of per-sample mean BC coloured by total_bp (depth proxy)

Outputs
-------
  results/<ref>/s1/supplement/
    tech_rep_pairs.csv              -- per-pair metrics
    tech_rep_summary.csv            -- per-lifestyle summary
    tech_rep_alpha_scatter.png
    tech_rep_richness_deviation.png
    tech_rep_bc_density.png
    tech_rep_depth_heatmap.png

Integration
-----------
Called from __main__.py as part of s1 / s1_tech / s1_within.
Gracefully skips if no secondary runs are available.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
from .plot_utils import (
    apply_oi_style,
    current_fs,
    page_figsize,
    plain_axes,
    square_figsize,
    panel_grid,
    savefig_dual,
    share_axis_labels,
)
apply_oi_style()
FS = current_fs()   # base font size; changes per preset in savefig_dual
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu

from . import config
from . import figure_source
from .detection import DetectionSpec, build_presence

# ── plot style (mirrors section1_within_variation) ───────────────────────────
plt.rcParams.update({
    "figure.dpi": 140,
    "axes.grid": True,
    "font.size": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
})

LIFESTYLE_ORDER = ["H-G", "Transitional", "Industrial"]


# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════

def _default_spec():
    return DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP)


def _icc_one_way(y_pri, y_sec):
    """
    One-way random-effects ICC (consistency) for n paired measurements.
    ICC = (MS_between - MS_within) / (MS_between + MS_within)
    Returns (icc, 95% CI lower, 95% CI upper) using the F-distribution.
    """
    n = len(y_pri)
    if n < 3:
        return np.nan, np.nan, np.nan

    grand = np.mean([*y_pri, *y_sec])
    row_means = (np.asarray(y_pri) + np.asarray(y_sec)) / 2.0

    ss_between = 2 * np.sum((row_means - grand) ** 2)
    ss_within  = np.sum((np.asarray(y_pri) - row_means) ** 2 +
                        (np.asarray(y_sec) - row_means) ** 2)

    ms_between = ss_between / (n - 1)
    ms_within  = ss_within  / n

    denom = ms_between + ms_within
    if denom == 0:
        return np.nan, np.nan, np.nan
    icc = (ms_between - ms_within) / denom

    # 95 % CI via F-distribution
    from scipy.stats import f as f_dist
    F = ms_between / max(ms_within, 1e-12)
    alpha = 0.05
    F_lo = F / f_dist.ppf(1 - alpha / 2, dfn=n - 1, dfd=n)
    F_hi = F * f_dist.ppf(1 - alpha / 2, dfn=n, dfd=n - 1)
    ci_lo = (F_lo - 1) / (F_lo + 1)
    ci_hi = (F_hi - 1) / (F_hi + 1)
    return float(icc), float(ci_lo), float(ci_hi)


def _between_lifestyle_bc(data, n_sample=1500, seed=42):
    """
    Sample within-lifestyle Bray-Curtis distances for the reference
    background distribution.  Returns {lifestyle: 1-D array of BC values}.
    """
    rng = np.random.default_rng(seed)
    result = {}
    for lf in LIFESTYLE_ORDER:
        ids = list(data.meta.index[data.meta["Lifestyle"] == lf])
        if len(ids) < 10:
            continue
        sub = rng.choice(ids, size=min(n_sample, len(ids)), replace=False).tolist()
        abund_g = np.nan_to_num(data.abundance.loc[sub].values, nan=0.0)
        from scipy.spatial.distance import pdist
        bc = pdist(abund_g, metric="braycurtis")
        result[lf] = np.nan_to_num(bc.ravel(), nan=1.0)
    return result


# ════════════════════════════════════════════════════════════════════
#  Core computation
# ════════════════════════════════════════════════════════════════════

def compute_tech_rep_pairs(data, spec=None):
    """
    Build the per-(primary, secondary) pair metrics table.

    For each primary sample `p` with secondary runs [s1, s2, ...]:
      - richness_primary, richness_secondary  (with the given DetectionSpec)
      - shannon_primary,  shannon_secondary
      - braycurtis(p, si)  and  jaccard(p, si)   on abundance vectors
      - frac_richness_dev = |rich_sec - rich_pri| / max(rich_pri, 1)

    Returns
    -------
    pd.DataFrame, one row per (primary, secondary) pair.
    Empty DataFrame if no sec_map entries survive.
    """
    if spec is None:
        spec = _default_spec()

    sec_map      = data.sec_map     or {}
    sec_breadth  = data.sec_breadth
    sec_depth    = data.sec_depth

    if not sec_map or sec_breadth is None:
        return pd.DataFrame()

    # Build presence/abundance for ALL secondary rows in one pass
    sgbs = data.breadth.columns.tolist()
    sec_b_aligned = sec_breadth.reindex(columns=sgbs, fill_value=0.0)
    sec_d_aligned = sec_depth.reindex(columns=sgbs, fill_value=0.0)

    # Relative abundance for secondary runs
    lengths_arr = data.lengths.reindex(sgbs, fill_value=0.0).values
    abund_raw   = sec_d_aligned.values * lengths_arr   # (n_sec, n_sgbs)
    row_sums    = abund_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = np.nan
    sec_abund_values = abund_raw / row_sums
    sec_abund_values = np.nan_to_num(sec_abund_values, nan=0.0)
    sec_abund_df = pd.DataFrame(sec_abund_values,
                                index=sec_breadth.index, columns=sgbs)

    # Secondary presence
    sec_pres_df = build_presence(sec_b_aligned, sec_d_aligned,
                                 data.lengths, spec=spec)

    # Shannon for secondary
    ap = np.clip(sec_abund_values, 1e-10, None)
    sec_shannon = -(ap * np.log(ap)).sum(axis=1)
    sec_shannon_s = pd.Series(sec_shannon, index=sec_breadth.index)
    sec_richness_s = sec_pres_df.sum(axis=1)

    rows = []
    for pri_id, sec_ids in sec_map.items():
        if pri_id not in data.meta.index:
            continue

        # Primary metrics
        pri_pres  = build_presence(
            data.breadth.loc[[pri_id]], data.depth.loc[[pri_id]],
            data.lengths, spec=spec)
        pri_rich  = int(pri_pres.sum(axis=1).iloc[0])
        pri_abund = data.abundance.loc[pri_id].values
        ap_pri    = np.clip(pri_abund, 1e-10, None)
        pri_shan  = float(-(ap_pri * np.log(ap_pri)).sum())

        lf = data.meta.loc[pri_id, "Lifestyle"] if "Lifestyle" in data.meta.columns else "Unknown"
        # A further library of the same BioSample (another extraction, not
        # just another run) -- see loading.load_reference.
        sib = str(data.meta.loc[pri_id].get("_sibling_library_ids", "") or "")
        sib_libs = set(x for x in sib.split(",") if x and x != "nan")
        pri_bp = float(data.total_bp.loc[pri_id]) if pri_id in data.total_bp.index else np.nan

        for sec_id in sec_ids:
            if sec_id not in sec_breadth.index:
                continue

            sec_rich = int(sec_richness_s.loc[sec_id])
            sec_shan = float(sec_shannon_s.loc[sec_id])
            sec_abund_vec = sec_abund_df.loc[sec_id].values

            # Distance metrics
            bc  = float(cdist(pri_abund[np.newaxis],
                               sec_abund_vec[np.newaxis],
                               metric="braycurtis")[0, 0])
            jac_p = (pri_pres.loc[pri_id].values > 0).astype(float)
            jac_s = (sec_pres_df.loc[sec_id].values > 0).astype(float)
            union = float(np.maximum(jac_p, jac_s).sum())
            jacc  = 1.0 - (float(np.minimum(jac_p, jac_s).sum()) / max(union, 1))

            frac_dev = abs(sec_rich - pri_rich) / max(pri_rich, 1)

            rows.append({
                "primary_id"        : pri_id,
                "secondary_id"      : sec_id,
                "replicate_type"    : "library" if sec_id in sib_libs else "run",
                "Lifestyle"         : lf,
                "richness_primary"  : pri_rich,
                "richness_secondary": sec_rich,
                "shannon_primary"   : round(pri_shan,  4),
                "shannon_secondary" : round(sec_shan,  4),
                "braycurtis"        : round(bc,   4),
                "jaccard"           : round(jacc, 4),
                "frac_richness_dev" : round(frac_dev, 4),
                "total_bp_primary"  : pri_bp,
            })

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
#  Summary statistics
# ════════════════════════════════════════════════════════════════════

def compute_summary(pairs_df, stats, spec, ref):
    """
    Compute per-lifestyle ICC, median BC, median fractional richness
    deviation.  Logs to stats and returns a summary DataFrame.
    """
    spec_tag = spec.short_label
    summary_rows = []

    for lf in LIFESTYLE_ORDER + ["All"]:
        if lf == "All":
            grp = pairs_df
        else:
            grp = pairs_df[pairs_df["Lifestyle"] == lf]
        if len(grp) < 2:
            continue

        y_pri = grp["richness_primary"].values.astype(float)
        y_sec = grp["richness_secondary"].values.astype(float)
        icc, ci_lo, ci_hi = _icc_one_way(y_pri, y_sec)

        med_bc       = float(np.median(grp["braycurtis"]))
        med_jacc     = float(np.median(grp["jaccard"]))
        med_frac_dev = float(np.median(grp["frac_richness_dev"]))
        n_pairs      = len(grp)
        n_primary    = grp["primary_id"].nunique()

        tag = lf.replace("-", "").replace(" ", "_").lower()
        stats.add(
            f"tech_rep_ICC_richness_{tag}_{spec_tag}", "s1", "supplement",
            f"Tech-rep ICC (richness, {lf}, {spec.description})",
            f"ICC={icc:.3f} 95%CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
            f"n_pairs={n_pairs}, n_primary={n_primary}",
            n1=n_primary, extra={"icc": icc, "ci_lo": ci_lo, "ci_hi": ci_hi})
        stats.add(
            f"tech_rep_BC_{tag}_{spec_tag}", "s1", "supplement",
            f"Tech-rep median Bray-Curtis ({lf}, {spec.description})",
            f"median={med_bc:.4f}  n_pairs={n_pairs}")
        stats.add(
            f"tech_rep_frac_dev_{tag}_{spec_tag}", "s1", "supplement",
            f"Tech-rep median fractional richness deviation ({lf})",
            f"median={med_frac_dev:.4f}")

        print(f"    {lf:15s}  n_pairs={n_pairs:4d}  n_primary={n_primary:4d}  "
              f"ICC={icc:.3f}[{ci_lo:.3f},{ci_hi:.3f}]  "
              f"BC={med_bc:.3f}  frac_dev={med_frac_dev:.3f}")

        summary_rows.append({
            "Lifestyle"          : lf,
            "n_primary_samples"  : n_primary,
            "n_pairs"            : n_pairs,
            # of which a second library of the BioSample, not a re-run
            "n_library_pairs"    : int((grp.get("replicate_type") == "library").sum()),
            "richness_ICC"       : round(icc,    4),
            "ICC_CI_lo"          : round(ci_lo,  4),
            "ICC_CI_hi"          : round(ci_hi,  4),
            "median_BC"          : round(med_bc,       4),
            "median_Jaccard"     : round(med_jacc,     4),
            "median_frac_richness_dev": round(med_frac_dev, 4),
        })

    return pd.DataFrame(summary_rows)


# ════════════════════════════════════════════════════════════════════
#  Plots
# ════════════════════════════════════════════════════════════════════

def _alpha_scatter_table(pairs_df):
    """Source rows behind Figure S2: one per technical-replicate pair.

    The ICC printed on each panel is computed from these two columns, so it is
    derivable from the table rather than carried beside it.
    """
    out = {}
    for lf in LIFESTYLE_ORDER:
        grp = pairs_df[pairs_df["Lifestyle"] == lf]
        if grp.empty:
            continue
        cols = {c: grp[c].values for c in
                ("richness_primary", "richness_secondary") if c in grp}
        df = pd.DataFrame(cols)
        for idcol in ("replicate_type", "Chosen_match_ID", "primary_id",
                      "secondary_id", "sample_id"):
            if idcol in grp:
                df.insert(0, idcol, grp[idcol].values)
        df.insert(0, "lifestyle", config.display_text(lf))
        out[config.display_text(lf)] = df.reset_index(drop=True)
    return out


def plot_alpha_scatter(tables, spec, outdir):
    """Draw Figure S2 from its tables and nothing else."""
    names = list(tables)
    if not names:
        return None
    _MG = dict(top=0.84)
    n_rows, n_cols = panel_grid(len(names))
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=square_figsize(n_cols=n_cols,
                                                    n_rows=n_rows,
                                                    margins=_MG))
    flat = axes.ravel()
    for ax in flat[len(names):]:
        ax.set_visible(False)

    for ax, name in zip(flat, names):
        df = tables[name]
        key = config.canonical_lifestyle(name) if hasattr(
            config, "canonical_lifestyle") else name
        color = config.LIFESTYLE_COLORS.get(key, "steelblue")
        x = df["richness_primary"].values.astype(float)
        y = df["richness_secondary"].values.astype(float)
        icc, _, _ = _icc_one_way(x, y)

        ax.scatter(x, y, c=color, alpha=0.5, s=18, edgecolors="none")
        lo = min(x.min(), y.min()) * 0.95
        hi = max(x.max(), y.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="y = x")

        icc_str = f"{icc:.3f}" if np.isfinite(icc) else "n/a"
        ax.set_title(name, fontsize=FS)
        ax.text(0.97, 0.06, f"n = {len(df):,}\nICC = {icc_str}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FS - 2,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="none", alpha=0.75))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

    fig._margin_overrides = _MG
    share_axis_labels(fig, "Primary run richness", "Secondary run richness")
    fig.suptitle(f"Technical replicates: richness concordance\n"
                 f"({spec.description})", fontsize=FS+2, fontweight="bold")
    fpath = os.path.join(outdir, f"tech_rep_alpha_scatter_{spec.short_label}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")
    return fpath


def _plot_alpha_scatter(pairs_df, spec, outdir):
    """Compute the figure's data object, publish it, then draw from it."""
    tables = _alpha_scatter_table(pairs_df)
    if not tables:
        return None
    # Only the main-text threshold ships as Figure S2; the other spec is a
    # sibling output of the same code and does not need its own data object.
    if spec.short_label == "10pct":
        figure_source.emit(outdir, "S2", tables,
                           stem=f"tech_rep_alpha_scatter_{spec.short_label}")
    return plot_alpha_scatter(tables, spec, outdir)

def _plot_richness_deviation(pairs_df, spec, outdir):
    """Violin of fractional richness deviation by lifestyle."""
    lfs_present = [lf for lf in LIFESTYLE_ORDER if lf in pairs_df["Lifestyle"].values]
    if not lfs_present:
        return

    fig, ax = plt.subplots(figsize=square_figsize())
    data_list  = [pairs_df[pairs_df["Lifestyle"] == lf]["frac_richness_dev"].values
                  for lf in lfs_present]
    colors     = [config.LIFESTYLE_COLORS.get(lf, "gray") for lf in lfs_present]
    positions  = list(range(1, len(lfs_present) + 1))

    parts = ax.violinplot(data_list, positions=positions,
                          showmedians=True, showextrema=False)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c); body.set_alpha(0.65)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.5)

    rng = np.random.default_rng(42)
    for i, (vals, c) in enumerate(zip(data_list, colors), 1):
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   c=c, alpha=0.35, s=8, edgecolors="none", zorder=3)

    ax.axhline(0.05, color="tomato", lw=1, ls="--", alpha=0.7,
               label="5% threshold")
    ax.set_xticks(positions)
    ax.set_xticklabels(lfs_present)
    ax.set_ylabel("Fractional richness deviation  |sec - pri| / pri")
    ax.set_title(f"Technical replicate richness deviation by lifestyle\n"
                 f"({spec.description})", fontsize=FS+1, fontweight="bold")
    ax.legend(fontsize=FS-1)
    plt.tight_layout()
    fpath = os.path.join(outdir, f"tech_rep_richness_deviation_{spec.short_label}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")


def _plot_bc_density(pairs_df, between_bc, spec, outdir):
    """
    Density overlay: within-sample BC (dashed, per lifestyle) vs
    between-sample within-lifestyle BC (solid).
    """
    from scipy.stats import gaussian_kde

    lfs_present = [lf for lf in LIFESTYLE_ORDER
                   if lf in pairs_df["Lifestyle"].values or lf in between_bc]
    if not lfs_present:
        return

    fig, ax = plt.subplots(figsize=square_figsize())
    x = np.linspace(0, 1, 500)

    for lf in lfs_present:
        color = config.LIFESTYLE_COLORS.get(lf, "gray")

        # Between-sample reference (solid)
        if lf in between_bc and len(between_bc[lf]) > 10:
            try:
                kde = gaussian_kde(between_bc[lf], bw_method=0.05)
                ax.plot(x, kde(x), color=color, lw=2,
                        label=f"{config.LIFESTYLE_DISPLAY.get(lf, lf)} between-sample")
            except Exception:
                pass

        # Within-sample tech-rep (dashed)
        grp_bc = pairs_df[pairs_df["Lifestyle"] == lf]["braycurtis"].values
        if len(grp_bc) >= 3:
            try:
                kde2 = gaussian_kde(grp_bc, bw_method=0.08)
                ax.plot(x, kde2(x), color=color, lw=2, ls="--",
                        label=f"{config.LIFESTYLE_DISPLAY.get(lf, lf)} within-sample (tech rep)")
            except Exception:
                pass

    # Log density. The technical-replicate curves are an order of magnitude
    # taller than the between-sample ones -- that is the result -- so on a
    # linear axis the between-sample distributions are pressed flat against the
    # baseline and their shape is unreadable. A log axis shows both.
    ax.set_yscale("log")
    _ymax = max((ln.get_ydata().max() for ln in ax.get_lines()
                 if len(ln.get_ydata())), default=1.0)
    ax.set_ylim(max(_ymax * 1e-4, 1e-3), _ymax * 2.2)
    # One label per decade, and say so on every decade in range. Without this
    # the axis showed a single tick and the scale was unreadable.
    from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2, 5), numticks=24))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Bray-Curtis dissimilarity")
    ax.set_ylabel("Density (log scale)")
    ax.set_title(f"Technical replicate vs between-sample Bray-Curtis\n"
                 f"({spec.description})", fontsize=FS+1, fontweight="bold")
    # The legend moves, the panel does not grow. Stretching the y axis to open
    # a gap for a key is paying in plot area for a placement problem: the curves
    # end up in the bottom two thirds of a panel that is mostly white. One
    # column and a smaller size make the key small enough to fit a corner, and
    # place_legends_clear then picks the corner that covers no curve.
    fig._plain_axes_kw = dict(x_ticks=5, y_ticks=4)
    ax.legend(fontsize=FS-3, ncol=1, framealpha=0.92)
    fpath = os.path.join(outdir, f"tech_rep_bc_density_{spec.short_label}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")


def _depth_vs_bc_table(pairs_df):
    """Source rows behind Figure S3: one per replicate pair, all lifestyles.

    A single-panel figure, so one table. The per-lifestyle counts in the key
    are group sizes within it rather than separate numbers.
    """
    df = pairs_df[pairs_df["total_bp_primary"].notna()].copy()
    if len(df) < 5:
        return {}
    cols = pd.DataFrame({
        "lifestyle": [config.display_text(x) for x in df["Lifestyle"]],
        "total_bp_primary": df["total_bp_primary"].values,
        "log10_total_bp_primary": np.log10(
            df["total_bp_primary"].clip(lower=1).values),
        "braycurtis": df["braycurtis"].values,
    })
    for idcol in ("replicate_type", "Chosen_match_ID", "primary_id",
                  "secondary_id", "sample_id"):
        if idcol in df:
            cols.insert(0, idcol, df[idcol].values)
    return {"all": cols.reset_index(drop=True)}


def plot_depth_vs_bc(tables, outdir, spec):
    """Draw Figure S3 from its table and nothing else."""
    if not tables or "all" not in tables:
        return None
    df = tables["all"]
    # Squat, not square: one scatter with a single trend carries little that
    # needs vertical room, and a square panel at page width spends most of it
    # on empty axis.
    fig, ax = plt.subplots(figsize=page_figsize(aspect=2.2))
    for name, sub in df.groupby("lifestyle", sort=False):
        key = config.canonical_lifestyle(name) if hasattr(
            config, "canonical_lifestyle") else name
        ax.scatter(sub["log10_total_bp_primary"], sub["braycurtis"],
                   c=config.LIFESTYLE_COLORS.get(key, "gray"),
                   alpha=0.45, s=14, edgecolors="none",
                   label=f"{name} (n={len(sub):,})")

    ax.set_xlabel("log$_{10}$(primary total_bp)")
    ax.set_ylabel("Bray-Curtis (primary vs secondary)")
    # Title to the caption. This one is an axes title, so the figure-level
    # strip never reached it.
    # Key above the panel, in one row. The deep second-library pairs fill the
    # right-hand side up to mid-height and the shallow outliers the left, so no
    # corner inside is clear; _fixed_loc keeps place_legends_clear off it.
    leg = ax.legend(fontsize=FS-1, loc="lower center", ncol=3,
                    bbox_to_anchor=(0.5, 1.01), frameon=False,
                    handletextpad=0.3, columnspacing=1.2)
    leg._fixed_loc = True
    fpath = os.path.join(outdir, f"tech_rep_depth_vs_bc_{spec.short_label}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fpath}")
    return fpath


def _plot_depth_vs_bc(pairs_df, outdir, spec):
    """Compute the figure's data object, publish it, then draw from it."""
    tables = _depth_vs_bc_table(pairs_df)
    if not tables:
        return None
    if spec.short_label == "10pct":
        figure_source.emit(outdir, "S3", tables,
                           stem=f"tech_rep_depth_vs_bc_{spec.short_label}")
    return plot_depth_vs_bc(tables, outdir, spec)


# ════════════════════════════════════════════════════════════════════
#  Public entry point
# ════════════════════════════════════════════════════════════════════

def run_tech_replicate_robustness(data, stats, specs=None):
    """
    Run the full within-sample technical replicate robustness analysis.

    Gracefully skips if:
      - data.sec_map is empty (no secondary runs available), or
      - sec_breadth / sec_depth are None.

    Parameters
    ----------
    data  : ReferenceData
    stats : StatsReport
    specs : list of DetectionSpec, or None (uses default 300kb flat-bp)
    """
    ref = data.ref_name
    print(f"\n{'='*60}")
    print(f"  Section 1 (tech replicates) [{ref}]")
    print(f"{'='*60}")

    # ── Guard: nothing to do ─────────────────────────────────────────
    if not data.sec_map:
        print("  [SKIP] No secondary run IDs available for this reference.")
        print("         Run build_final_metadata.py first to populate "
              "Secondary_run_IDs, then re-run the pipeline.")
        return

    if data.sec_breadth is None or data.sec_depth is None:
        print("  [SKIP] sec_breadth / sec_depth matrices not loaded "
              "(secondary run IDs present in metadata but not in matrix).")
        return

    n_primary_with_sec = len(data.sec_map)
    n_sec_total = sum(len(v) for v in data.sec_map.values())
    print(f"  Primary samples with secondary run(s): {n_primary_with_sec:,}")
    print(f"  Total secondary run rows available   : {n_sec_total:,}")

    outdir = config.section_dir(ref, "s1", "supplement")
    os.makedirs(outdir, exist_ok=True)

    if specs is None:
        specs = [
            DetectionSpec("flat_bp",     config.ACCUM_MAIN_TEXT_FLAT_BP),
            DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH),
        ]
    elif not isinstance(specs, (list, tuple)):
        specs = [specs]

    # Pre-compute between-lifestyle BC for reference (spec-independent)
    print("\n  Pre-computing between-sample BC reference distributions...")
    between_bc = _between_lifestyle_bc(data)

    for spec in specs:
        print(f"\n  [{spec.short_label}] detection: {spec.description}")

        # ── Build pair table ─────────────────────────────────────────
        pairs_df = compute_tech_rep_pairs(data, spec=spec)
        if pairs_df.empty:
            print(f"    [WARN] No valid pairs computed - skipping {spec.short_label}")
            continue

        n_pairs = len(pairs_df)
        n_pri   = pairs_df["primary_id"].nunique()
        print(f"    Pairs: {n_pairs:,}  ({n_pri:,} unique primary samples)")

        # ── Summary stats ────────────────────────────────────────────
        print(f"\n  Summary (per lifestyle):")
        summary_df = compute_summary(pairs_df, stats, spec, ref)

        # ── Save CSVs ────────────────────────────────────────────────
        pairs_path   = os.path.join(outdir,
            f"tech_rep_pairs_{spec.short_label}.csv")
        summary_path = os.path.join(outdir,
            f"tech_rep_summary_{spec.short_label}.csv")
        pairs_df.to_csv(pairs_path,   index=False)
        summary_df.to_csv(summary_path, index=False)
        print(f"    Saved: {pairs_path}")
        print(f"    Saved: {summary_path}")

        # ── Plots ────────────────────────────────────────────────────
        _plot_alpha_scatter(pairs_df, spec, outdir)
        _plot_richness_deviation(pairs_df, spec, outdir)
        _plot_bc_density(pairs_df, between_bc, spec, outdir)
        _plot_depth_vs_bc(pairs_df, outdir, spec)

    print(f"\n  Technical replicate robustness analysis complete [{ref}]")
