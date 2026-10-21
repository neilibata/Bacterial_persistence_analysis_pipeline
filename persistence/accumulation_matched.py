"""
Depth-matched accumulation curves.

Controls for sequencing-depth confounds by greedy 1:1 nearest-neighbour
matching (on log₁₀ total mapped bp) between Industrial and Non-Industrial
samples *before* computing accumulation curves.

Analyses:
  A. Pooled depth-matched: Industrial vs Non-Industrial (all samples)
  B. Per-continent depth-matched: top 2–3 continents by matched-pair count

This is complementary to the standard (unmatched) accumulation analysis —
the matched version has fewer samples but removes the depth bias, so any
difference in the curves is more convincingly biological.

Figures produced:
  - Accumulation curve overlays (matched Ind vs matched NI)
  - Depth-distribution QC plots (pre- vs post-match)
  - Parameter-vs-N_max sweep (same format as unmatched)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plot_utils import apply_oi_style, savefig_dual
apply_oi_style()
from scipy.spatial import cKDTree

from . import config
from .common import classify_group, classify_macro_group
from .accumulation import (
    bootstrap_curve, fit_central_curve, fit_mm_free,
    compute_r2, bootstrap_param_distribution,
    plot_accumulation_curves, plot_param_vs_N,
    _sweep_columns, _empty_sweep_row,
)
from math import isfinite


# ════════════════════════════════════════════════════════════════════
#  GREEDY NEAREST-NEIGHBOUR MATCHING
# ════════════════════════════════════════════════════════════════════

def greedy_depth_match(log_depths_g1, log_depths_g2, max_diff=None):
    """
    1:1 nearest-neighbour matching on log-depth, without replacement.

    Matches the smaller group against the larger, then greedily assigns
    pairs in order of increasing distance (best matches first).

    Parameters
    ----------
    log_depths_g1, log_depths_g2 : pd.Series
        Index = sample IDs, values = log₁₀(total mapped bp).
    max_diff : float, optional
        Maximum allowed log-depth difference for a pair.
        Default: None (no hard cutoff — rely on greedy ordering).

    Returns
    -------
    matched_g1, matched_g2 : list of sample IDs (same length)
    match_stats : dict with quality metrics
    """
    if len(log_depths_g1) == 0 or len(log_depths_g2) == 0:
        return [], [], {"n_pairs": 0}

    # Smaller group queries the larger
    if len(log_depths_g1) <= len(log_depths_g2):
        query, ref, q_is_g1 = log_depths_g1, log_depths_g2, True
    else:
        query, ref, q_is_g1 = log_depths_g2, log_depths_g1, False

    tree = cKDTree(ref.values.reshape(-1, 1))
    dists, indices = tree.query(query.values.reshape(-1, 1), k=1)

    # Greedy assignment: best matches first, no replacement
    used_ref = set()
    mq, mr, pair_dists = [], [], []
    order = np.argsort(dists.ravel())

    for qi in order:
        ri = int(indices[qi])
        if ri in used_ref:
            continue
        d = abs(float(query.values[qi]) - float(ref.values[ri]))
        if max_diff is not None and d > max_diff:
            continue
        used_ref.add(ri)
        mq.append(query.index[qi])
        mr.append(ref.index[ri])
        pair_dists.append(d)

    matched_g1, matched_g2 = (mq, mr) if q_is_g1 else (mr, mq)

    stats = {
        "n_pairs": len(matched_g1),
        "n_g1_input": len(log_depths_g1),
        "n_g2_input": len(log_depths_g2),
        "median_log_depth_diff": float(np.median(pair_dists)) if pair_dists else np.nan,
        "mean_log_depth_diff": float(np.mean(pair_dists)) if pair_dists else np.nan,
        "max_log_depth_diff": float(np.max(pair_dists)) if pair_dists else np.nan,
        "q95_log_depth_diff": float(np.percentile(pair_dists, 95)) if pair_dists else np.nan,
    }

    return matched_g1, matched_g2, stats


# ════════════════════════════════════════════════════════════════════
#  DEPTH-DISTRIBUTION QC PLOT
# ════════════════════════════════════════════════════════════════════

def plot_depth_qc(meta, ind_all, ni_all, ind_matched, ni_matched,
                  title, filepath):
    """
    Side-by-side histograms of log-depth before and after matching.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    bins = np.linspace(
        meta["_log_depth"].quantile(0.01),
        meta["_log_depth"].quantile(0.99),
        40)

    # Before matching
    ax = axes[0]
    ax.hist(meta.loc[ind_all, "_log_depth"].dropna(), bins=bins,
            alpha=0.5, color=config.OI["sky"], label=f"Industrial (n={len(ind_all)})",
            density=True)
    ax.hist(meta.loc[ni_all, "_log_depth"].dropna(), bins=bins,
            alpha=0.5, color=config.OI["orange"], label=f"Non-Industrial (n={len(ni_all)})",
            density=True)
    ax.set_xlabel("log₁₀(total mapped bp)")
    ax.set_ylabel("Density")
    ax.set_title("Before matching")
    ax.legend(fontsize=9)

    # After matching
    ax = axes[1]
    ax.hist(meta.loc[ind_matched, "_log_depth"].dropna(), bins=bins,
            alpha=0.5, color=config.OI["sky"],
            label=f"Industrial (n={len(ind_matched)})",
            density=True)
    ax.hist(meta.loc[ni_matched, "_log_depth"].dropna(), bins=bins,
            alpha=0.5, color=config.OI["orange"],
            label=f"Non-Industrial (n={len(ni_matched)})",
            density=True)
    ax.set_xlabel("log₁₀(total mapped bp)")
    ax.set_title("After depth-matching")
    ax.legend(fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig_dual(fig, filepath, dpi=200)
    plt.close(fig)
    print(f"    Saved: {filepath}")


# ════════════════════════════════════════════════════════════════════
#  MATCHED ACCUMULATION — SWEEP ENGINE
# ════════════════════════════════════════════════════════════════════

def _run_matched_sweep(series_inputs, thr, rng, out_dir, ref_name):
    """
    Run bootstrap accumulation + N_max sweep on depth-matched subsets.

    series_inputs: list of (name, block, color, ls) where block is
                   already the depth-matched subset of the breadth matrix.
    """
    N_points = sorted({
        int(round(x)) for x in np.logspace(
            np.log10(config.ACCUM_N_MIN),
            np.log10(config.ACCUM_N_MAX),
            config.ACCUM_N_NUM)
    })
    N_sweep = sorted({
        int(round(x)) for x in np.logspace(
            np.log10(config.ACCUM_SWEEP_N_MIN),
            np.log10(config.ACCUM_SWEEP_N_MAX),
            config.ACCUM_SWEEP_N_NUM)
    })

    sweep_rows = []
    curve_data = []

    for name, block, color, ls in series_inputs:
        n_samples = block.shape[0]
        if n_samples < 10:
            print(f"    ⚠️  Skipping {name}: only {n_samples} matched samples")
            continue

        Ns_curve = sorted({n for n in N_points if 1 <= n <= n_samples})
        if len(Ns_curve) < 5:
            # Fallback: generate a sensible grid for small pools
            Ns_curve = sorted({int(round(x)) for x in
                               np.logspace(np.log10(3),
                                           np.log10(n_samples),
                                           min(20, n_samples))})
        if len(Ns_curve) < 5:
            print(f"    ⚠️  Skipping {name}: insufficient N points "
                  f"({len(Ns_curve)} for {n_samples} samples)")
            continue

        (N_vec, mean_vec, std_vec, med_vec, q16_vec, q84_vec,
         y_center, scale_vec, n_mic, params, errs, r2,
         boot_matrix) = fit_central_curve(block, Ns_curve, thr, rng)

        if N_vec.size < 5 or boot_matrix.size == 0:
            continue

        curve_data.append({
            "name": name, "N_vec": N_vec, "y_center": y_center,
            "q16_vec": q16_vec, "q84_vec": q84_vec,
            "n_mic": n_mic, "params": params,
            "color": color, "ls": ls,
        })

        # N_max sweep
        Ns_sw = [n for n in N_sweep if 10 <= n <= n_samples]
        if not Ns_sw:
            # Use all available if sweep grid is too coarse
            Ns_sw = sorted({int(round(x)) for x in
                            np.logspace(np.log10(10),
                                        np.log10(n_samples),
                                        min(15, n_samples))})

        for N_max in Ns_sw:
            mask = N_vec <= N_max
            if mask.sum() < 5:
                continue

            N_sub = N_vec[mask]
            y_sub = y_center[mask]
            s_sub = scale_vec[mask]
            b_sub = boot_matrix[:, mask]

            params_sub, errs_sub = fit_mm_free(
                N_sub, y_sub, s_sub, n_mic,
                use_weights=config.ACCUM_USE_WEIGHTS)

            if params_sub is None:
                sweep_rows.append(_empty_sweep_row(
                    name, thr, N_max, n_samples, n_mic))
                continue

            a_sub, b_val, c_sub = params_sub
            r2_sub = compute_r2(N_sub, y_sub, n_mic, params_sub)

            qd = bootstrap_param_distribution(
                N_sub, b_sub, s_sub, n_mic, rng)

            sweep_rows.append([
                name, thr, N_max, n_samples, int(n_mic),
                a_sub, errs_sub[1] if isfinite(errs_sub[1]) else np.nan,
                b_val, c_sub,
                errs_sub[3] if isfinite(errs_sub[3]) else np.nan,
                r2_sub,
                qd["a_q16"], qd["a_q50"], qd["a_q84"],
                qd["b_q16"], qd["b_q50"], qd["b_q84"],
                qd["c_q16"], qd["c_q50"], qd["c_q84"],
                qd["a_mean"], qd["a_sd"],
                qd["b_mean"], qd["b_sd"],
                qd["c_mean"], qd["c_sd"],
                qd["p_a_below_nmic_minus_tol"], qd["tol_microbes"],
            ])

    sweep_df = pd.DataFrame()
    if sweep_rows:
        sweep_df = pd.DataFrame(sweep_rows, columns=_sweep_columns())

    return curve_data, sweep_df


# ════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def run_accumulation_matched(data, matched_idx=None):
    """
    Depth-matched accumulation curves for one reference database.

    For each breadth threshold:
      A. Pooled: 1:1 depth-match Industrial vs Non-Industrial, then
         run accumulation on each matched subset.
      B. Per-continent: depth-match within the top 2–3 continents that
         have sufficient data on both sides.

    Parameters
    ----------
    data : ReferenceData
    matched_idx : list, optional
        Not used (matching is done here). Kept for API compatibility.
    """
    ref = data.ref_name
    out = config.outdir(ref)
    meta = data.meta.copy()
    breadth = data.breadth

    rng = np.random.default_rng(123)   # different seed from unmatched

    print(f"\n  [{ref}] Depth-matched accumulation curves "
          f"(thresholds: {config.ACCUM_BREADTH_THRESHOLDS})")

    # ── Group classification ──
    meta["MacroGroup"] = meta.apply(classify_macro_group, axis=1)
    cont_col = "Continent" if "Continent" in meta.columns else None

    ind_mask = meta["MacroGroup"] == "Industrial"
    ni_mask  = meta["MacroGroup"] == "Non-Industrial"

    ind_ids = meta.index[ind_mask]
    ni_ids  = meta.index[ni_mask]

    print(f"    Industrial: {len(ind_ids):,}  |  Non-Industrial: {len(ni_ids):,}")

    # ════════════════════════════════════════════
    #  A. POOLED DEPTH-MATCHED
    # ════════════════════════════════════════════

    print(f"\n    ── Pooled depth-matching ──")
    d_ind = meta.loc[ind_ids, "_log_depth"].dropna()
    d_ni  = meta.loc[ni_ids, "_log_depth"].dropna()

    m_ind, m_ni, mstats = greedy_depth_match(d_ind, d_ni)

    print(f"    Matched pairs: {mstats['n_pairs']:,}  "
          f"(from {mstats['n_g1_input']:,} Ind + {mstats['n_g2_input']:,} NI)")
    print(f"    Median |Δlog₁₀ depth|: {mstats['median_log_depth_diff']:.4f}  "
          f"(95th pctile: {mstats['q95_log_depth_diff']:.4f})")

    # Save match stats
    stats_rows = [{"scope": "Pooled", **mstats}]

    if mstats["n_pairs"] >= 20:
        # Depth QC plot
        plot_depth_qc(
            meta, ind_ids, ni_ids, m_ind, m_ni,
            f"Depth matching QC — Pooled ({ref})",
            os.path.join(out, f"depth_match_qc_pooled_{ref}.png"))

        # Loop over thresholds
        for thr in config.ACCUM_BREADTH_THRESHOLDS:
            thr_label = str(thr).replace(".", "p")
            print(f"\n    ═══ Threshold {thr}% (pooled matched) ═══")

            series_inputs = [
                (f"Industrial (matched, n={len(m_ind)})",
                 breadth.loc[m_ind], config.OI["sky"], "-"),
                (f"Non-Industrial (matched, n={len(m_ni)})",
                 breadth.loc[m_ni], config.OI["orange"], "-"),
            ]

            curve_data, sweep_df = _run_matched_sweep(
                series_inputs, thr, rng, out, ref)

            if curve_data:
                plot_accumulation_curves(
                    curve_data, thr, ref, out,
                    suffix="_pooled_depthmatched")

            if not sweep_df.empty:
                csv_path = os.path.join(
                    out, f"accum_matched_pooled_fits_vsN_cov{thr_label}.csv")
                sweep_df.to_csv(csv_path, index=False)

                style_map = {s[0]: (s[2], s[3]) for s in series_inputs}
                plot_param_vs_N(
                    sweep_df, thr,
                    f"Depth-matched Pooled ({ref})",
                    style_map,
                    os.path.join(out,
                        f"accum_matched_pooled_params_vsN_cov{thr_label}.png"))
    else:
        print(f"    ⚠️  Too few matched pairs ({mstats['n_pairs']}) for "
              f"pooled analysis — skipping")

    # ════════════════════════════════════════════
    #  B. PER-CONTINENT DEPTH-MATCHED
    # ════════════════════════════════════════════

    if cont_col is None or meta[cont_col].isna().all():
        print(f"\n    [Continent matched] No continent data — skipping")
        _save_match_stats(stats_rows, out, ref)
        return

    print(f"\n    ── Per-continent depth-matching ──")

    # Find continents with enough data on BOTH sides
    continent_pairs = {}
    for cont in meta[cont_col].dropna().unique():
        c_ind = meta.index[ind_mask & (meta[cont_col] == cont)]
        c_ni  = meta.index[ni_mask & (meta[cont_col] == cont)]
        if len(c_ind) >= 15 and len(c_ni) >= 15:
            continent_pairs[cont] = (c_ind, c_ni)

    if not continent_pairs:
        print(f"    No continents with >=15 samples on both sides")
        _save_match_stats(stats_rows, out, ref)
        return

    # Rank by min(Industrial, Non-Industrial) → pick top 3
    ranked = sorted(continent_pairs.keys(),
                    key=lambda c: min(len(continent_pairs[c][0]),
                                      len(continent_pairs[c][1])),
                    reverse=True)
    top_continents = ranked[:3]

    print(f"    Top continents for matching: {top_continents}")
    for cont in ranked:
        c_ind, c_ni = continent_pairs[cont]
        marker = " ✓" if cont in top_continents else ""
        print(f"      {cont:20s}  Ind={len(c_ind):,}  NI={len(c_ni):,}{marker}")

    # Match within each selected continent
    continent_matched = {}
    for cont in top_continents:
        c_ind_ids, c_ni_ids = continent_pairs[cont]
        d1 = meta.loc[c_ind_ids, "_log_depth"].dropna()
        d2 = meta.loc[c_ni_ids, "_log_depth"].dropna()

        cm_ind, cm_ni, cm_stats = greedy_depth_match(d1, d2)
        cm_stats["continent"] = cont
        stats_rows.append({"scope": f"Continent: {cont}", **cm_stats})

        if cm_stats["n_pairs"] >= 15:
            continent_matched[cont] = (cm_ind, cm_ni, cm_stats)
            print(f"    {cont}: {cm_stats['n_pairs']} matched pairs  "
                  f"(median |Δ|={cm_stats['median_log_depth_diff']:.4f})")
        else:
            print(f"    {cont}: only {cm_stats['n_pairs']} pairs — skipping")

    if not continent_matched:
        print(f"    No continents with enough matched pairs")
        _save_match_stats(stats_rows, out, ref)
        return

    # Depth QC plot for continent matching
    _plot_continent_depth_qc(meta, ind_mask, ni_mask, continent_matched,
                             ref, out)

    # Run accumulation for each threshold
    for thr in config.ACCUM_BREADTH_THRESHOLDS:
        thr_label = str(thr).replace(".", "p")
        print(f"\n    ═══ Threshold {thr}% (continent matched) ═══")

        series_inputs = []
        style_map = {}

        for cont, (cm_ind, cm_ni, cm_stats) in continent_matched.items():
            color = config.CONTINENT_COLORS.get(cont, "gray")
            n = cm_stats["n_pairs"]

            label_ind = f"Industrial - {cont} (n={n})"
            label_ni  = f"Non-Industrial - {cont} (n={n})"

            series_inputs.append(
                (label_ind, breadth.loc[cm_ind], color, "-"))
            series_inputs.append(
                (label_ni, breadth.loc[cm_ni], color, "--"))
            style_map[label_ind] = (color, "-")
            style_map[label_ni]  = (color, "--")

        curve_data, sweep_df = _run_matched_sweep(
            series_inputs, thr, rng, out, ref)

        if curve_data:
            plot_accumulation_curves(
                curve_data, thr, ref, out,
                suffix="_continent_depthmatched")

        if not sweep_df.empty:
            csv_path = os.path.join(
                out, f"accum_matched_continent_fits_vsN_cov{thr_label}.csv")
            sweep_df.to_csv(csv_path, index=False)

            plot_param_vs_N(
                sweep_df, thr,
                f"Depth-matched by Continent ({ref})\n"
                f"Industrial (solid) vs Non-Industrial (dashed)",
                style_map,
                os.path.join(out,
                    f"accum_matched_continent_params_vsN_cov{thr_label}.png"))

    _save_match_stats(stats_rows, out, ref)
    print(f"\n    [{ref}] Depth-matched accumulation complete.")


# ════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════

def _save_match_stats(stats_rows, out_dir, ref_name):
    """Save depth-matching quality statistics."""
    df = pd.DataFrame(stats_rows)
    fpath = os.path.join(out_dir, f"depth_match_stats_{ref_name}.csv")
    df.to_csv(fpath, index=False)
    print(f"    Saved: {fpath}")


def _plot_continent_depth_qc(meta, ind_mask, ni_mask, continent_matched,
                              ref_name, out_dir):
    """Multi-panel depth QC for continent matching."""
    n_cont = len(continent_matched)
    fig, axes = plt.subplots(n_cont, 2, figsize=(14, 4 * n_cont),
                              squeeze=False)

    bins = np.linspace(
        meta["_log_depth"].quantile(0.01),
        meta["_log_depth"].quantile(0.99), 35)

    for row, (cont, (cm_ind, cm_ni, cm_stats)) in enumerate(
            continent_matched.items()):
        color = config.CONTINENT_COLORS.get(cont, "gray")
        cont_mask = meta.get("Continent") == cont

        # Before
        ax = axes[row, 0]
        all_ind = meta.index[ind_mask & cont_mask]
        all_ni  = meta.index[ni_mask & cont_mask]
        ax.hist(meta.loc[all_ind, "_log_depth"].dropna(), bins=bins,
                alpha=0.5, color=config.OI["sky"],
                label=f"Ind (n={len(all_ind)})", density=True)
        ax.hist(meta.loc[all_ni, "_log_depth"].dropna(), bins=bins,
                alpha=0.5, color=config.OI["orange"],
                label=f"NI (n={len(all_ni)})", density=True)
        ax.set_title(f"{cont} — before matching")
        ax.legend(fontsize=8)
        if row == n_cont - 1:
            ax.set_xlabel("log₁₀(total mapped bp)")

        # After
        ax = axes[row, 1]
        ax.hist(meta.loc[cm_ind, "_log_depth"].dropna(), bins=bins,
                alpha=0.5, color=config.OI["sky"],
                label=f"Ind (n={len(cm_ind)})", density=True)
        ax.hist(meta.loc[cm_ni, "_log_depth"].dropna(), bins=bins,
                alpha=0.5, color=config.OI["orange"],
                label=f"NI (n={len(cm_ni)})", density=True)
        ax.set_title(f"{cont} — after matching "
                     f"(median |Δ|={cm_stats['median_log_depth_diff']:.3f})")
        ax.legend(fontsize=8)
        if row == n_cont - 1:
            ax.set_xlabel("log₁₀(total mapped bp)")

    fig.suptitle(f"Depth-matching QC by Continent ({ref_name})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fpath = os.path.join(out_dir, f"depth_match_qc_continent_{ref_name}.png")
    savefig_dual(fig, fpath, dpi=200)
    plt.close(fig)
    print(f"    Saved: {fpath}")
