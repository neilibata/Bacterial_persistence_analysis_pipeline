"""Depth-bias analysis restricted to the Carter et al. (2023) meta-analysis cohort.

Additional Figures A6-A9 are the Carter-subset versions of the depth and GMW
analyses: depth distributions, the GMW richness correction, the threshold sweep,
and the per-sample weight diagnostics. They were originally produced by a
standalone script (`carter_microbiome_depth_analysis.py`) that was pinned to
UHGG with pre-August inputs and read its own copy of the sample list.

Two things are recovered here that the main pipeline never carried:

  * weight diagnostics  -- the per-sample importance weights against sequencing
    depth, which is the mechanism of the correction rather than its result.
  * detection bias      -- per-SGB detection difference against genome length,
    showing that short genomes drive the apparent richness gap.

Everything else reuses `depth_bias_ipw`, which the original script was written
to mirror; this module supplies the subset and the two missing figures rather
than reimplementing the estimator.

The cohort is `carter_cohort.find_carter_cohort` -- an identifier join against
Carter's published list, not a study-name match. See that module for why.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config
from .carter_cohort import find_carter_cohort, describe
from .detection import DetectionSpec, build_presence
from .depth_bias_ipw import (
    _BinnedKDE, _adaptive_n_boot, _ess, _geomean_weights, _make_main_figure,
    _run_one_spec, run_geomean_richness_pairwise, run_gmw_threshold_sweep,
)
import os
from . import figure_source
from .plot_utils import (
    apply_oi_style,
    style_for_figure,
    tighten_margins,
    current_fs,
    page_figsize,
    plain_axes,
    square_figsize,
    drawn_margins,
    savefig_dual,
    share_axis_labels,
)

apply_oi_style()

# Chimp is a negative control with no lifestyle metadata, so the Carter join is
# meaningless there. Every other reference gets its own run.
SUPPORTED_REFS = ("combined", "UHGG", "lev")


def _outdir(ref: str) -> str:
    d = os.path.join("results", ref, "carter_analysis")
    os.makedirs(d, exist_ok=True)
    return d


def _plot_weight_diagnostics(log_depth, is_ind, w_ind, w_ni, spec, out_path, ref):
    """Build this figure's table, publish it, then draw the figure from it.

    Depth and group determine the weights and the densities, and the effective
    sample sizes in the key are computed from the weights, so these three
    columns reproduce every number shown.
    """
    mask = np.asarray(is_ind, dtype=bool)
    weights = np.empty(mask.size, dtype=float)
    weights[mask] = np.asarray(w_ind, dtype=float)
    weights[~mask] = np.asarray(w_ni, dtype=float)
    tables = {"samples": pd.DataFrame({
        "log10_total_mapped_bp": np.asarray(log_depth, dtype=float),
        "group": np.where(mask, "Industrial", "Non-Industrial"),
        "gmw_weight": weights,
    })}
    if spec.short_label == "10pct":
        figure_source.emit(
            # S8 since the 2026-10-19 renumbering (was S9).
            os.path.dirname(out_path), "S8", tables,
            stem=os.path.basename(out_path).replace(".png", ""),
            reference=ref, notes=spec.description)
    return plot_weight_diagnostics(tables, out_path, ref, spec.description)


def plot_weight_diagnostics(tables, out_path, ref, description=""):
    """Per-sample GMW weights against depth -- the mechanism, not the result.

    The correction works by up-weighting samples where a group is thin on the
    depth axis and down-weighting where it is dense. Plotting the weights makes
    visible how much reweighting is applied and where, which a corrected point
    estimate cannot show.
    """
    df = tables["samples"]
    log_depth = df["log10_total_mapped_bp"].to_numpy(dtype=float)
    is_ind = (df["group"].astype(str) == "Industrial").to_numpy()
    _w_all = df["gmw_weight"].to_numpy(dtype=float)
    w_ind, w_ni = _w_all[is_ind], _w_all[~is_ind]

    # The gutter only has to clear the right panel's y tick labels and axis
    # label; the legends sit inside their panels. It is set through
    # _margin_overrides because the publication margin pass re-applies the
    # shared defaults and would otherwise undo a plain subplots_adjust. Size
    # the canvas from the margins as drawn (title stripped), not as authored.
    _MG = dict(top=0.84, wspace=0.20)
    fig, axes = plt.subplots(1, 2, squeeze=False,
                             figsize=square_figsize(n_cols=2,
                                                    margins=drawn_margins(_MG)))
    ax_w, ax_d = axes[0]
    FS = style_for_figure(fig)
    fig.subplots_adjust(wspace=_MG["wspace"])
    fig._margin_overrides = dict(_MG)

    ld_i, ld_n = log_depth[is_ind], log_depth[~is_ind]
    for ld, w, lab, col in ((ld_i, w_ind, "Industrial", config.OI["sky"]),
                            (ld_n, w_ni, "Non-Industrial", config.OI["orange"])):
        if len(ld) == 0:
            continue
        order = np.argsort(ld)
        ax_w.scatter(ld[order], w[order], s=7, alpha=0.35, color=col,
                     edgecolors="none", label=lab)
    ax_w.axhline(1.0, color="#555555", lw=1.0, ls="--", alpha=0.8)
    ax_w.set_yscale("log")
    # A weight of 1 is "unchanged", and up-weighting by 3x is the same size of
    # intervention as down-weighting by 3x. Letting matplotlib pick the limits
    # put 1 well above centre and spent half the panel on empty decades, which
    # also reads as though the correction were mostly downward. Pin the axis
    # symmetric about 1 on the log scale.
    allw = np.concatenate([w for w in (w_ind, w_ni) if len(w)])
    allw = allw[np.isfinite(allw) & (allw > 0)]
    if allw.size:
        # Span from the bulk of the weights, not the extremes: a handful of
        # samples in the thin tail of the depth distribution take very large
        # weights, and letting them set the axis spends most of the panel on
        # empty decades. The 99th percentile keeps the axis symmetric about 1
        # and still shows every point that carries the correction.
        span = float(np.percentile(np.abs(np.log10(allw)), 99)) * 1.10
        span = min(max(span, 0.25), 2.0)
        ax_w.set_ylim(10 ** -span, 10 ** span)
    ax_w.set_xlabel("log$_{10}$(total mapped bases)", fontsize=FS)
    ax_w.set_ylabel("GMW importance weight", fontsize=FS)
    # ESS rides in the key rather than the title: it describes the series, so
    # it belongs beside the swatch that identifies them.
    _h, _l = ax_w.get_legend_handles_labels()
    _ess_by = {"Industrial": (_ess(w_ind), len(w_ind)),
               "Non-Industrial": (_ess(w_ni), len(w_ni))}
    _l = [f"{lab} (ESS {_ess_by[lab][0]:.0f}/{_ess_by[lab][1]})"
          if lab in _ess_by else lab for lab in _l]
    # Bottom left: the weights rise to the right of this panel, so that is
    # the corner the data leaves free.
    _leg_w = ax_w.legend(_h, _l, fontsize=FS - 2, framealpha=0.85,
                         loc="lower left")
    _leg_w._fixed_loc = True
    ax_w.set_title("Weights per sample", fontsize=FS, fontweight="bold")

    # The depth densities the weights are derived from, plus the geometric-mean
    # target they are pulled toward.
    lo = float(np.min(log_depth)) - 0.25
    hi = float(np.max(log_depth)) + 0.25
    grid = np.linspace(lo, hi, 400)
    k_i, k_n = _BinnedKDE(ld_i), _BinnedKDE(ld_n)
    d_i, d_n = k_i(grid), k_n(grid)
    tgt = np.sqrt(np.maximum(d_i, 1e-300) * np.maximum(d_n, 1e-300))
    for d, lab, col in ((d_i, "Industrial", config.OI["sky"]),
                        (d_n, "Non-Industrial", config.OI["orange"])):
        ax_d.plot(grid, d / max(d.max(), 1e-300), color=col, lw=2.0, label=lab)
    ax_d.plot(grid, tgt / max(tgt.max(), 1e-300), color="black", lw=2.0,
              ls="--", label="GMW target")
    ax_d.set_xlabel("log$_{10}$(total mapped bases)", fontsize=FS)
    ax_d.set_ylabel("Normalised density", fontsize=FS)
    ax_d.legend(fontsize=FS - 2, framealpha=0.85, loc="best")
    ax_d.set_title("Depth densities and target", fontsize=FS, fontweight="bold")

    ess_i, ess_n = _ess(w_ind), _ess(w_ni)
    # One line, and placed clear of the panel titles: a two-line suptitle at
    # this width sat directly on top of them (100% overlap in the collision map).
    fig.suptitle(
        f"Carter cohort, {description}",
        fontsize=FS - 1, y=0.97)
    # Set the margins as this figure's own rather than calling the margin pass
    # here: savefig_dual calls it again with the shared numbers, so whichever
    # ran last won, and the title band reserved here was handed straight back.
    # That is why the suptitle ended up on the panel titles.
    fig._margin_overrides = _MG
    fig._plain_axes_kw = dict(x_ticks=5, y_ticks=4)
    savefig_dual(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"ess_ind": float(ess_i), "ess_ni": float(ess_n)}


def _plot_detection_bias(breadth, depth, lengths, is_ind, spec, out_path, ref):
    """Per-SGB detection difference against genome length.

    If the richness gap were a depth artefact alone it would fall on short
    genomes, which need proportionally more reads to clear a breadth threshold.
    Plotting the per-SGB difference against length tests that directly.
    """
    FS = current_fs()
    presence = build_presence(breadth, depth, lengths, spec=spec)
    prev_i = presence.loc[is_ind].mean(axis=0)
    prev_n = presence.loc[~is_ind].mean(axis=0)
    diff = (prev_i - prev_n).astype(float)

    L = lengths.reindex(diff.index).astype(float)
    ok = L.notna() & (L > 0) & diff.notna()
    if ok.sum() < 10:
        print("      [carter] too few SGBs with lengths - skipping detection bias")
        return None

    x = np.log10(L[ok].values)
    y = diff[ok].values

    fig, ax = plt.subplots(figsize=square_figsize())
    ax.scatter(x, y, s=6, alpha=0.30, color=config.OI["blue"], edgecolors="none")
    ax.axhline(0.0, color="#555555", lw=1.0, ls="--", alpha=0.8)

    # Median trend in length bins, which is what the eye should follow.
    bins = np.quantile(x, np.linspace(0, 1, 13))
    bins = np.unique(bins)
    if len(bins) > 2:
        centres, meds = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (x >= a) & (x < b)
            if m.sum() >= 5:
                centres.append(0.5 * (a + b)); meds.append(float(np.median(y[m])))
        if centres:
            ax.plot(centres, meds, color=config.OI["vermillion"], lw=2.2,
                    marker="o", ms=4, label="median per length bin")
            ax.legend(fontsize=FS - 2, framealpha=0.85, loc="best")

    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    ax.set_xlabel("log$_{10}$(genome length, bp)", fontsize=FS)
    ax.set_ylabel("Prevalence difference\n(Industrial $-$ Non-Industrial)", fontsize=FS)
    ax.set_title(
        f"Carter cohort: detection bias vs genome length [{ref}]\n"
        f"{spec.description}  |  Spearman $\\rho$ = {rho:+.3f}, p = {p:.2e}, "
        f"n = {int(ok.sum()):,} SGBs",
        fontsize=FS, fontweight="bold")
    savefig_dual(fig, out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"spearman_rho": float(rho), "spearman_p": float(p),
            "n_sgbs": int(ok.sum())}


def run_carter_analysis(data, stats, specs=None, n_boot=None, rng_seed=7):
    """Depth-bias figures restricted to Carter's meta-analysis cohort."""
    ref = data.ref_name
    if ref not in SUPPORTED_REFS:
        print(f"\n  Carter subset analysis: skipping [{ref}] "
              f"(supported: {', '.join(SUPPORTED_REFS)})")
        return None

    meta = data.meta
    mask, diag = find_carter_cohort(meta, strict=False)
    print(f"\n  Carter subset analysis [{ref}]")
    print(f"    {describe(diag)}")
    if diag["n"] == 0:
        print("    [WARN] no Carter samples matched - skipping")
        return None

    stats.add(f"carter_cohort_n_{ref}", "s1", "supplement",
              f"Carter meta-analysis samples present ({ref})", diag["n"])
    stats.add(f"carter_cohort_recovery_{ref}", "s1", "supplement",
              f"Fraction of Carter's 1,800 recovered ({ref})", diag["recovery"])

    valid = mask & meta["MacroGroup"].isin(["Industrial", "Non-Industrial"])
    idx = meta.index[valid]
    is_ind = (meta.loc[idx, "MacroGroup"] == "Industrial").values
    log_depth = np.log10(np.maximum(data.total_bp.loc[idx].values, 1.0))

    if is_ind.sum() < 10 or (~is_ind).sum() < 10:
        print(f"    [WARN] too few samples after grouping "
              f"({is_ind.sum()}/{(~is_ind).sum()}) - skipping")
        return None
    print(f"    Industrial={is_ind.sum():,}  Non-Industrial={(~is_ind).sum():,}")

    if specs is None:
        specs = [DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH),
                 DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP)]
    if n_boot is None:
        n_boot = _adaptive_n_boot(int(is_ind.sum()), int((~is_ind).sum()))

    out = _outdir(ref)
    rng = np.random.default_rng(rng_seed)
    results, rows = [], []

    for spec in specs:
        tag = spec.short_label
        print(f"\n    Spec: {spec.description}")
        presence = build_presence(data.breadth.loc[idx], data.depth.loc[idx],
                                  data.lengths, spec=spec)
        richness = presence.sum(axis=1).values.astype(float)

        r = _run_one_spec(log_depth, is_ind, richness, spec, n_boot, rng, stats)
        results.append(r)

        w = _plot_weight_diagnostics(
            log_depth, is_ind, r["w_ind"], r["w_ni"], spec,
            os.path.join(out, f"carter_gmw_weight_diagnostics_{tag}.png"), ref)
        db = _plot_detection_bias(
            data.breadth.loc[idx], data.depth.loc[idx], data.lengths, is_ind,
            spec, os.path.join(out, f"carter_detection_bias_{tag}.png"), ref)

        row = {"spec": spec.description, "mode": spec.mode,
               "threshold": spec.threshold,
               "n_ind": int(is_ind.sum()), "n_ni": int((~is_ind).sum()),
               "att": r["att"], "ci_lo": r["ci_att"][0], "ci_hi": r["ci_att"][1],
               "raw_pct": r["raw_pct"], "corr_pct": r["corr_pct"],
               "p_corr": r["p_corr"]}
        row.update(w or {})
        row.update(db or {})
        rows.append(row)

    pd.DataFrame(rows).to_csv(
        os.path.join(out, "carter_gmw_summary.csv"), index=False)
    _make_main_figure(results, os.path.join(out, "carter_gmw_richness_main.png"), rng)

    # A7: the three-way lifestyle comparison on this subset. The caption asks
    # for Industrial / Agro-pastoralist / Hunter-gatherer, which is the pairwise
    # routine, not the two-group one above.
    run_geomean_richness_pairwise(data, stats, specs=specs, sample_mask=mask,
                                  out_dir=out,
                                  stem="carter_richness_pairwise_corrected")

    # A8: the same gap across every detection threshold, on this subset. Reuses
    # the whole-cohort sweep with a mask rather than repeating the loop.
    run_gmw_threshold_sweep(data, stats, sample_mask=mask, out_dir=out,
                            label="  [Carter subset]")
    print(f"\n    Carter subset analysis complete [{ref}] -> {out}")
    return pd.DataFrame(rows)
