#!/usr/bin/env python3
"""Regenerate combined Figure 3 with fresh bootstraps and log insets."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln

# Ensure the repo root (parent of figures/) is importable when this script is
# run by path from figures/, e.g. `python figures/make_combined_figure3.py`.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from persistence import config
from persistence.accumulation import (
    _figure3_N_points,
    _figure3_asymptote,
    _figure3_boot_counts,
    _figure3_fit_grid,
    _figure3_model_y,
    _fit_figure3_asymptote_bootstrap,
    _fit_figure3_curve,
    _make_figure3_series,
    _median_fit_scale,
    _study_col,
    bootstrap_curve_by_study_variable_reps,
    bootstrap_curve_variable_reps,
)
# Reinstatement is part of cohort construction: load_reference alone returns
# the pre-reinstatement cohort, which is not what the paper's numbers use.
from persistence.loading import load_reference_analytic as load_reference
from persistence.plot_utils import (
    scale_point_rcparams,
    _hide_labels,
    _restore_labels,
    apply_oi_style,
)


REF_LABELS = {
    "UHGG": "Carter (Hadza + UHGG)",
    "lev": "Leviatan",
    "combined": "Combined (Carter + Leviatan)",
}

SIDE_LABELS = {
    "UHGG": r"$\bf{Carter}$ (Hadza + UHGG)",
    "lev": r"$\bf{Leviatan}$",
    "combined": r"$\bf{Combined}$ (Carter + Leviatan)",
}


def threshold_tag(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "p") + "pct"


def _log1mexp(log_x: float) -> float:
    """Stable log(1 - exp(log_x)) for log_x <= 0."""
    if log_x >= 0:
        return -np.inf
    if log_x < np.log(0.5):
        return float(np.log1p(-np.exp(log_x)))
    return float(np.log(-np.expm1(log_x)))


def _beta_binomial_logpmf(r, n, alpha, beta):
    r = np.asarray(r, dtype=float)
    n = np.asarray(n, dtype=float)
    return (
        gammaln(n + 1.0)
        - gammaln(r + 1.0)
        - gammaln(n - r + 1.0)
        + betaln(r + alpha, n - r + beta)
        - betaln(alpha, beta)
    )


def _beta_binomial_tail(n, k: int, alpha: float, beta: float):
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n, dtype=float)
    valid = n >= k
    if not np.any(valid):
        return out
    xv = n[valid]
    cdf = np.zeros_like(xv, dtype=float)
    for r in range(k):
        cdf += np.exp(_beta_binomial_logpmf(r, xv, alpha, beta))
    out[valid] = np.clip(1.0 - cdf, 0.0, 1.0)
    return out


def _fit_rsac_beta_binomial(counts, n_units: int, n_mic: int, k: int):
    """Fit a beta-binomial r-SAC from incidence frequencies."""
    counts = np.asarray(counts, dtype=int)
    observed = counts[counts > 0]
    s_obs = int(observed.size)
    if s_obs < 3 or n_units < max(k, 2):
        return None

    values, freqs = np.unique(observed, return_counts=True)
    prevalence_mean = max(float(observed.mean()) / float(n_units), 1.0 / max(n_units, 1))

    def nll(log_ab):
        alpha, beta = np.exp(log_ab)
        log_q0 = float(betaln(alpha, n_units + beta) - betaln(alpha, beta))
        log_seen = _log1mexp(log_q0)
        if not np.isfinite(log_seen):
            return 1e100
        log_q = _beta_binomial_logpmf(values, n_units, alpha, beta)
        val = -float(np.sum(freqs * (log_q - log_seen)))
        return val if np.isfinite(val) else 1e100

    starts = []
    for alpha0 in (0.05, 0.1, 0.3, 0.7, 1.0, 2.0):
        beta0 = alpha0 * max((1.0 - prevalence_mean) / prevalence_mean, 1.0)
        starts.append((alpha0, beta0))
    starts.extend([(0.1, 10.0), (0.5, 100.0), (1.0, 1000.0), (2.0, 10000.0)])

    best = None
    bounds = [(np.log(1e-5), np.log(1e5)), (np.log(1e-3), np.log(1e8))]
    for alpha0, beta0 in starts:
        x0 = np.log([alpha0, beta0])
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000})
        if res.success and np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return None

    alpha, beta = np.exp(best.x)
    log_q0 = float(betaln(alpha, n_units + beta) - betaln(alpha, beta))
    q0 = float(np.exp(log_q0)) if log_q0 > -745 else 0.0
    seen_prob = max(1.0 - q0, 1e-12)
    s_hat = min(float(n_mic), float(s_obs) / seen_prob)
    return {
        "S": s_hat,
        "alpha": float(alpha),
        "beta": float(beta),
        "k": int(k),
        "n_units": int(n_units),
        "s_obs": int(s_obs),
        "nll": float(best.fun),
    }


def _incidence_frequency(counts):
    values, freqs = np.unique(np.asarray(counts, dtype=int), return_counts=True)
    return {
        "counts": values.astype(int),
        "freqs": freqs.astype(int),
    }


def _rsac_model_y(x_fit, sd):
    params = sd.get("rsac_params")
    if not params:
        return None
    return params["S"] * _beta_binomial_tail(
        x_fit,
        int(params["k"]),
        float(params["alpha"]),
        float(params["beta"]),
    )


def _figure3_model_y_local(model, x_fit, sd):
    if model == "rsac":
        return _rsac_model_y(x_fit, sd)
    return _figure3_model_y(model, x_fit, sd)


def _figure3_asymptote_local(model, sd):
    if model == "rsac":
        params = sd.get("rsac_params")
        return np.nan if not params else float(params["S"])
    return _figure3_asymptote(model, sd)


def _figure3_series_for_threshold(data, threshold: float, match_existing_rng: bool = False):
    """Run the same bootstrap path as Figure3 and return the target threshold."""
    rng = np.random.default_rng(123)
    meta = data.meta.copy()
    style = {
        "Industrial": (config.OI["sky"], "-"),
        "Non-Industrial": (config.OI["orange"], "-"),
    }
    masks = {
        "Industrial": meta["Lifestyle"] == "Industrial",
        "Non-Industrial": meta["Lifestyle"].isin(["H-G", "Transitional"]),
    }
    study_col = _study_col(meta)

    if match_existing_rng:
        target_sample = target_study = None
        for thr in config.ACCUM_BREADTH_THRESHOLDS:
            print(f"    bootstrap {data.ref_name} {thr:g}%...", flush=True)
            _, sample_series, study_series = _make_figure3_series(
                data.breadth, meta, masks, style, study_col, thr, rng
            )
            if float(thr) == float(threshold):
                target_sample = sample_series
                target_study = study_series
                break
        if target_sample is None or target_study is None:
            raise ValueError(f"Threshold {threshold:g}% is not in config.ACCUM_BREADTH_THRESHOLDS")
        return target_sample, target_study

    print(f"    bootstrap {data.ref_name} {threshold:g}%...", flush=True)
    _, sample_series, study_series = _make_figure3_series(
        data.breadth, meta, masks, style, study_col, threshold, rng
    )
    return sample_series, study_series


def _series_from_bootstrap(
    name,
    color,
    ls,
    N_vec,
    q16_vec,
    q84_vec,
    med_vec,
    boot_matrix,
    n_mic,
    rsac_params=None,
    incidence_counts=None,
    unit_count=None,
    threshold=None,
    min_occurrences=None,
):
    y_center = med_vec
    fit_scale = _median_fit_scale(boot_matrix)
    scale_vec, hill_params, hill_r2, richards_params, richards_r2 = _fit_figure3_curve(
        N_vec, y_center, q16_vec, q84_vec, n_mic, fit_scale=fit_scale
    )
    hill_asymptote_ci = _fit_figure3_asymptote_bootstrap(
        N_vec, boot_matrix, scale_vec, n_mic, model="hill"
    )
    richards_asymptote_ci = _fit_figure3_asymptote_bootstrap(
        N_vec, boot_matrix, scale_vec, n_mic, model="richards"
    )
    return dict(
        name=name,
        N_vec=N_vec,
        y_center=y_center,
        q16_vec=q16_vec,
        q84_vec=q84_vec,
        n_mic=n_mic,
        hill_params=hill_params,
        hill_r2=hill_r2,
        richards_params=richards_params,
        richards_r2=richards_r2,
        hill_asymptote_ci=hill_asymptote_ci,
        richards_asymptote_ci=richards_asymptote_ci,
        color=color,
        ls=ls,
        rsac_params=rsac_params,
        boot_matrix=np.asarray(boot_matrix, dtype=np.float32),
        fit_scale=np.asarray(fit_scale, dtype=np.float32),
        incidence_counts=(
            None if incidence_counts is None else np.asarray(incidence_counts, dtype=np.uint16)
        ),
        incidence_frequency=(
            None if incidence_counts is None else _incidence_frequency(incidence_counts)
        ),
        unit_count=unit_count,
        threshold=threshold,
        min_occurrences=min_occurrences,
    )


def _summarize_boot_values(vals_by_N, Ns, n_mic):
    means, stds, meds, q16s, q84s = [], [], [], [], []
    for vals in vals_by_N:
        means.append(float(vals.mean()))
        stds.append(float(vals.std(ddof=1)) if vals.size > 1 else 0.0)
        meds.append(float(np.median(vals)))
        q16s.append(float(np.percentile(vals, 2.5)))
        q84s.append(float(np.percentile(vals, 97.5)))
    return (
        Ns.astype(float),
        np.array(means),
        np.array(stds),
        n_mic,
        np.array(meds),
        np.array(q16s),
        np.array(q84s),
    )


def _empty_bootstrap_result(Ns, n_mic):
    return (
        Ns.astype(float),
        np.array([]),
        np.array([]),
        n_mic,
        np.array([]),
        np.array([]),
        np.array([]),
        np.empty((0, 0), dtype=float),
    )


def _bootstrap_curve_variable_reps_min_occurrence(
    coverage_block,
    Ns,
    boot_counts,
    rng,
    threshold,
    min_occurrences,
):
    """Sample-count accumulation requiring detection in at least k samples."""
    if min_occurrences <= 1:
        return bootstrap_curve_variable_reps(
            coverage_block, Ns, boot_counts, rng, threshold, return_boot_matrix=True
        )

    present = coverage_block.values > threshold
    n_samples, n_mic = present.shape
    Ns = np.asarray(Ns, int)
    Ns = np.sort(np.unique(Ns[(Ns >= 1) & (Ns <= n_samples)]))
    if Ns.size == 0:
        return _empty_bootstrap_result(Ns, n_mic)

    n_boot_by_N = np.array(
        [max(1, int(boot_counts.get(int(N), config.ACCUM_NUM_INNER_BOOT))) for N in Ns],
        dtype=int,
    )
    max_boot = int(n_boot_by_N.max())
    vals_by_N = [np.empty(n_boot, dtype=float) for n_boot in n_boot_by_N]
    boot_matrix = np.full((max_boot, Ns.size), np.nan, dtype=float)
    fill_by_N = np.zeros(Ns.size, dtype=int)

    for r in range(max_boot):
        active = np.flatnonzero(r < n_boot_by_N)
        if active.size == 0:
            continue
        perm = rng.permutation(n_samples)
        present_count = np.zeros(n_mic, dtype=np.uint16)
        next_pos = 0
        for i in active:
            N = int(Ns[i])
            if N > next_pos:
                present_count += present[perm[next_pos:N]].sum(axis=0, dtype=np.uint16)
                next_pos = N
            value = int(np.count_nonzero(present_count >= min_occurrences))
            vals_by_N[i][fill_by_N[i]] = value
            boot_matrix[r, i] = value
            fill_by_N[i] += 1

    return _summarize_boot_values(vals_by_N, Ns, n_mic) + (boot_matrix,)


def _bootstrap_curve_by_study_variable_reps_min_occurrence(
    coverage_block,
    study_ids,
    Ns,
    boot_counts,
    rng,
    threshold,
    min_occurrences,
):
    """Study-count accumulation requiring detection in at least k studies."""
    if min_occurrences <= 1:
        return bootstrap_curve_by_study_variable_reps(
            coverage_block,
            study_ids,
            Ns,
            boot_counts,
            rng,
            threshold,
            return_boot_matrix=True,
        )

    present = coverage_block.values > threshold
    studies = pd.Series(study_ids, index=coverage_block.index).astype(str)
    unique_studies = np.array(sorted(studies.dropna().unique()))
    n_studies = len(unique_studies)
    n_mic = present.shape[1]
    Ns = np.asarray(Ns, int)
    Ns = np.sort(np.unique(Ns[(Ns >= 1) & (Ns <= n_studies)]))
    if Ns.size == 0:
        return _empty_bootstrap_result(Ns, n_mic)

    study_codes = pd.Categorical(studies, categories=unique_studies).codes
    study_present = np.zeros((n_studies, n_mic), dtype=bool)
    for i in range(n_studies):
        study_present[i] = present[study_codes == i].any(axis=0)

    n_boot_by_N = np.array(
        [max(1, int(boot_counts.get(int(N), config.ACCUM_NUM_INNER_BOOT))) for N in Ns],
        dtype=int,
    )
    max_boot = int(n_boot_by_N.max())
    vals_by_N = [np.empty(n_boot, dtype=float) for n_boot in n_boot_by_N]
    boot_matrix = np.full((max_boot, Ns.size), np.nan, dtype=float)
    fill_by_N = np.zeros(Ns.size, dtype=int)

    for r in range(max_boot):
        active = np.flatnonzero(r < n_boot_by_N)
        if active.size == 0:
            continue
        perm = rng.permutation(n_studies)
        present_count = np.zeros(n_mic, dtype=np.uint16)
        next_pos = 0
        for i in active:
            N = int(Ns[i])
            if N > next_pos:
                present_count += study_present[perm[next_pos:N]].sum(axis=0, dtype=np.uint16)
                next_pos = N
            value = int(np.count_nonzero(present_count >= min_occurrences))
            vals_by_N[i][fill_by_N[i]] = value
            boot_matrix[r, i] = value
            fill_by_N[i] += 1

    return _summarize_boot_values(vals_by_N, Ns, n_mic) + (boot_matrix,)


def _figure3_series_for_min_occurrence(data, threshold: float, min_occurrences: int):
    rng = np.random.default_rng(123)
    meta = data.meta.copy()
    style = {
        "Industrial": (config.OI["sky"], "-"),
        "Non-Industrial": (config.OI["orange"], "-"),
    }
    masks = {
        "Industrial": meta["Lifestyle"] == "Industrial",
        "Non-Industrial": meta["Lifestyle"].isin(["H-G", "Transitional"]),
    }
    study_col = _study_col(meta)
    sample_series = []
    study_series = []

    print(
        f"    bootstrap {data.ref_name} {threshold:g}%, min occurrence {min_occurrences}...",
        flush=True,
    )
    for name, mask in masks.items():
        ids = mask[mask].index.intersection(data.breadth.index)
        block = data.breadth.loc[ids]
        if block.shape[0] < config.ACCUM_MIN_SAMPLES:
            continue
        color, ls = style[name]
        Ns = _figure3_N_points(block.shape[0])
        boot_counts = _figure3_boot_counts(Ns)
        N_vec, _, _, n_mic, med_vec, q16_vec, q84_vec, boot_matrix = (
            _bootstrap_curve_variable_reps_min_occurrence(
                block, Ns, boot_counts, rng, threshold, min_occurrences
            )
        )
        if N_vec.size >= 5:
            present_counts = (block.values > threshold).sum(axis=0)
            rsac_params = _fit_rsac_beta_binomial(
                present_counts, block.shape[0], n_mic, min_occurrences
            )
            sample_series.append(
                _series_from_bootstrap(
                    name,
                    color,
                    ls,
                    N_vec,
                    q16_vec,
                    q84_vec,
                    med_vec,
                    boot_matrix,
                    n_mic,
                    rsac_params=rsac_params,
                    incidence_counts=present_counts,
                    unit_count=block.shape[0],
                    threshold=threshold,
                    min_occurrences=min_occurrences,
                )
            )

        if study_col is None:
            continue
        studies = meta.loc[ids, study_col]
        n_studies = studies.astype(str).nunique()
        if n_studies < 5:
            continue
        Ns_st = _figure3_N_points(n_studies)
        boot_counts = _figure3_boot_counts(Ns_st)
        present = block.values > threshold
        study_codes = pd.Categorical(studies.astype(str)).codes
        study_present = np.zeros((n_studies, present.shape[1]), dtype=bool)
        for i in range(n_studies):
            study_present[i] = present[study_codes == i].any(axis=0)
        N_vec, _, _, n_mic, med_vec, q16_vec, q84_vec, boot_matrix = (
            _bootstrap_curve_by_study_variable_reps_min_occurrence(
                block, studies, Ns_st, boot_counts, rng, threshold, min_occurrences
            )
        )
        if N_vec.size >= 5:
            study_counts = study_present.sum(axis=0)
            rsac_params = _fit_rsac_beta_binomial(
                study_counts, n_studies, n_mic, min_occurrences
            )
            study_series.append(
                _series_from_bootstrap(
                    name,
                    color,
                    ls,
                    N_vec,
                    q16_vec,
                    q84_vec,
                    med_vec,
                    boot_matrix,
                    n_mic,
                    rsac_params=rsac_params,
                    incidence_counts=study_counts,
                    unit_count=n_studies,
                    threshold=threshold,
                    min_occurrences=min_occurrences,
                )
            )
    return sample_series, study_series


def _draw_one_axis(
    ax,
    series_data,
    *,
    model: str,
    unit_label: str,
    y_label: str,
    row: int,
    show_xlabel: bool,
    show_ylabel_ticks: bool,
    show_xticks: bool,
    show_legend: bool,
):
    # Bottom raised from 0.11: the inset's own tick labels and x label sit
    # below it, and at the printed width that stack is a slightly larger
    # share of a shorter panel than it was at 19.6 in -- enough to push the
    # label under the panel's own axis. The inset's size is unchanged.
    inset = ax.inset_axes([0.36, 0.16, 0.61, 0.55])
    all_N = np.concatenate([np.asarray(sd["N_vec"], float) for sd in series_data])
    x_min = float(np.nanmin(all_N))
    x_max = float(np.nanmax(all_N))
    text_x = {
        "Industrial": 0.11,
        "Non-Industrial": 0.52,
    }

    for sd in series_data:
        N = np.asarray(sd["N_vec"], float)
        y = np.asarray(sd["y_center"], float)
        q16 = np.asarray(sd["q16_vec"], float)
        q84 = np.asarray(sd["q84_vec"], float)
        color = sd["color"]
        name = sd["name"]

        asym = _figure3_asymptote_local(model, sd)
        asym_q16, asym_q50, asym_q84 = sd.get(
            f"{model}_asymptote_ci", (np.nan, np.nan, np.nan)
        )
        asym_plot = asym_q50 if np.isfinite(asym_q50) else asym

        ax.fill_between(N, q16, q84, color=color, alpha=0.15, linewidth=0)
        ax.plot(N, y, "o", color=color, ms=_k(3.1), alpha=0.86)
        inset.fill_between(N, q16, q84, color=color, alpha=0.12, linewidth=0)
        inset.plot(N, y, "o", color=color, ms=_k(2.5), alpha=0.86, label=name)

        if np.isfinite(asym_q16) and np.isfinite(asym_q84):
            ax.axhspan(asym_q16, asym_q84, color=color, alpha=0.08, zorder=0)
            inset.axhspan(asym_q16, asym_q84, color=color, alpha=0.08, zorder=0)
        if np.isfinite(asym_plot):
            label = f"{name}: {asym_plot:.0f}"
            if np.isfinite(asym_q16) and np.isfinite(asym_q84):
                label = f"{name}: {asym_plot:.0f} [{asym_q16:.0f}-{asym_q84:.0f}]"
            ax.axhline(asym_plot, color=color, lw=_k(1.35, 0.5), ls=":", alpha=0.82)
            inset.axhline(asym_plot, color=color, lw=_k(1.1, 0.5), ls=":", alpha=0.82)
            ax.text(
                text_x.get(name, 0.15),
                asym_plot,
                label,
                transform=ax.get_yaxis_transform(),
                color=color,
                fontsize=_k(13, 5.0),
                ha="left",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.60, pad=_k(1.0)),
            )

        x_fit = _figure3_fit_grid(N)
        y_fit = _figure3_model_y_local(model, x_fit, sd)
        if y_fit is not None:
            ax.plot(x_fit, y_fit, color=color, lw=_k(2.4, 0.5))
            inset.plot(x_fit, y_fit, color=color, lw=_k(1.8, 0.5))

    ax.set_title("")
    ax.set_xlabel(f"Number of {unit_label} (linear scale)" if show_xlabel else "", fontsize=_k(16))
    ax.set_ylabel(y_label if show_ylabel_ticks else "", fontsize=_k(16))
    ax.grid(False)
    # Scaled: a literal here overrides the scaled rcParam and is what
    # kept the tick labels at their authored size on a canvas 2.7x
    # smaller, displacing the axis titles.
    ax.tick_params(labelsize=_k(14, 5.0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    y_ticks = [0, 900, 1800, 2700, 3600] if row < 2 else [0, 1200, 2400, 3600, 4800]
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):d}"))
    if not show_xticks:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    if not show_ylabel_ticks:
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    inset.set_xscale("log")
    inset.set_xlabel(f"Number of {unit_label} (log scale)", fontsize=_k(17), labelpad=_k(1))
    inset.set_ylabel("SGBs", fontsize=_k(17), labelpad=_k(1))
    inset.tick_params(labelsize=_k(16, 5.0))
    inset.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    inset.set_yticks(y_ticks[::2])
    inset.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):d}"))
    inset.grid(False)
    if show_legend:
        inset.legend(fontsize=_k(14, 5.0), loc="lower right", frameon=True)


def load_panels(threshold: float, match_existing_rng: bool = False):
    panels = {}
    for ref in ("UHGG", "lev", "combined"):
        print(f"\n== {REF_LABELS[ref]} ==", flush=True)
        data = load_reference(ref)
        panels[ref] = _figure3_series_for_threshold(
            data, threshold, match_existing_rng=match_existing_rng
        )
    return panels


def load_min_occurrence_panels(threshold: float, min_occurrences: int):
    panels = {}
    for ref in ("UHGG", "lev", "combined"):
        print(f"\n== {REF_LABELS[ref]} ==", flush=True)
        data = load_reference(ref)
        panels[ref] = _figure3_series_for_min_occurrence(data, threshold, min_occurrences)
    return panels


# Bumped when the cohort definition changes. "_reinstated" marks caches built
# after secondary-run reinstatement became part of the loader.
_CACHE_SCHEMA = "figure3_combined_panels_with_bootstrap_replicates_v3_input_signature"


def _cohort_fingerprint(panels):
    """Record what the cache was built on, so a stale one is visible on load."""
    out = {}
    for ref, panel in (panels or {}).items():
        try:
            obj = panel[0] if isinstance(panel, (list, tuple)) else panel
            out[ref] = {k: int(v["N_total"]) for k, v in obj.items()
                        if isinstance(v, dict) and "N_total" in v}
        except Exception:
            pass
    return out


def _input_signature():
    """Size and mtime of every matrix this figure is built from.

    The schema string alone cannot express "the data changed". A cohort
    fingerprint was already being stored, but only printed -- never compared --
    so a cache built before the Leviatan backfill was still served afterwards
    and the figure re-rendered the old cohort without any warning. Comparing
    input signatures makes replacing a matrix invalidate the cache, which is
    the same mtime rule the rest of the pipeline resumes on.
    """
    import os
    from persistence import config
    root = Path(__file__).resolve().parents[1]
    sig = {}
    for ref in ("UHGG", "lev", "combined"):
        spec = config.REFERENCES.get(ref, {})
        for key in ("depth_file", "breadth_file", "lengths_file"):
            name = spec.get(key)
            if not name:
                continue
            p = root / name
            try:
                st = p.stat()
                sig[name] = [int(st.st_size), int(st.st_mtime)]
            except OSError:
                sig[name] = None
    return sig


def load_or_build_panels(
    cache_path: Path,
    threshold: float,
    match_existing_rng: bool = False,
    min_occurrences: int = 1,
    rebuild: bool = False,
):
    # The cache is keyed by a schema string that encodes the cohort definition.
    # It used to be reused unconditionally, so a cache built before secondary-run
    # reinstatement was applied kept being served after the loader was fixed --
    # the figure silently re-rendered the wrong cohort. Bumping the schema
    # invalidates every pre-reinstatement cache automatically.
    if cache_path.exists() and not rebuild:
        print(f"Loading bootstrap cache: {cache_path}", flush=True)
        with cache_path.open("rb") as fh:
            payload = pickle.load(fh)
        schema = payload.get("schema") if isinstance(payload, dict) else None
        if schema == _CACHE_SCHEMA:
            cached_sig = payload.get("inputs")
            now_sig = _input_signature()
            if cached_sig == now_sig:
                cohort = payload.get("cohort")
                if cohort:
                    print(f"  cached cohort: {cohort}", flush=True)
                return payload["panels"]
            changed = sorted(
                k for k in set(cached_sig or {}) | set(now_sig)
                if (cached_sig or {}).get(k) != now_sig.get(k))
            print(f"  [stale cache] source matrices changed since it was built "
                  f"({', '.join(changed) or 'unknown'}) — rebuilding on the "
                  f"current data", flush=True)
        else:
            print(f"  [stale cache] schema {schema!r} != {_CACHE_SCHEMA!r} "
                  f"— rebuilding on the current cohort", flush=True)
    if min_occurrences > 1:
        panels = load_min_occurrence_panels(threshold, min_occurrences)
    else:
        panels = load_panels(threshold, match_existing_rng=match_existing_rng)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        payload = {
            "schema": _CACHE_SCHEMA,
            "threshold": float(threshold),
            "min_occurrences": int(min_occurrences),
            "cohort": _cohort_fingerprint(panels),
            "inputs": _input_signature(),
            "panels": panels,
        }
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved bootstrap cache: {cache_path}", flush=True)
    return panels


def savefig_dual_preserve_layout(fig, out_path: Path, dpi: int = 300):
    """Save as drawn, plus a no-suptitle twin.

    Kept separate from plot_utils.savefig_dual because this figure sets its own
    layout and must not have tight_layout run over it.
    """
    # Normalise the published labels. This save exists to skip the layout
    # passes, not the relabelling: without it "H-G" and "Transitional" reach
    # the page, which is the one thing every artefact is supposed to hide.
    try:
        from persistence.plot_utils import apply_display_text
        apply_display_text(fig)
    except Exception as exc:
        print(f"  [labels] could not normalise display text: {exc}")

    stem = out_path.with_suffix("")
    ext = out_path.suffix or ".png"

    main_path = Path(f"{stem}{ext}")
    # This script manages its own layout and so does not go through
    # savefig_dual; it still has to meet the same Science format.
    try:
        from persistence.plot_utils import prepare_for_publication
        # Inset axes carry deliberately smaller type than the panels
        # around them; one role-based size makes the inset labels
        # bigger than the inset.
        fig._skip_type_normalise = True
        prepare_for_publication(fig, Path(main_path).stem)
    except Exception as _exc:
        print(f'  [format] {_exc}')
    # Land on the printed column width exactly. The tight crop takes this
    # figure to a little under 7.24 in, so it would otherwise be placed
    # smaller than the column and its type reduced with it. Nothing in the
    # drawing is touched -- see persistence.figure_rescale.
    from persistence.figure_rescale import save_column_width
    _fmt = ext.lstrip(".") or "png"
    save_column_width(fig, str(stem), dpi=dpi, formats=(_fmt,), verbose=False)

    notitle_path = Path(f"{stem}_notitle{ext}")
    hidden = _hide_labels(fig)
    save_column_width(fig, f"{stem}_notitle", dpi=dpi, formats=(_fmt,),
                      verbose=False)
    _restore_labels(hidden)

    return main_path, notitle_path


def panels_from_tables(tables):
    """Rebuild the panel structure from the published table.

    Returns {reference: (sample_series, study_series)}, the shape make_figure
    expects. Colour and line style are presentation rather than data, so they
    are reassigned here from the series name.
    """
    import numpy as _np

    style = {
        "Industrial": (config.OI["sky"], "-"),
        "Non-Industrial": (config.OI["orange"], "-"),
    }
    acc = tables["accumulation"]
    fits = tables.get("fits")

    def _fit(ref, unit, series, model):
        if fits is None:
            return None, None, None
        m = fits[(fits["reference"] == ref) & (fits["unit"] == unit)
                 & (fits["series"] == series) & (fits["model"] == model)]
        if m.empty:
            return None, None, None
        r = m.iloc[0]
        par = tuple(float(v) for v in str(r["params"]).split(";") if v != "")
        ci = (r["asymptote_q16"], r["asymptote_q50"], r["asymptote_q84"])
        return par, r["r2"], ci

    panels = {}
    for ref, per_ref in acc.groupby("reference", sort=False):
        by_unit = []
        for unit in ("samples", "studies"):
            series_list = []
            sub_u = per_ref[per_ref["unit"] == unit]
            for name, sd in sub_u.groupby("series", sort=False):
                colour, ls = style.get(name, (config.OI["sky"], "-"))
                entry = dict(
                    name=name,
                    N_vec=_np.asarray(sd["n_units"], dtype=float),
                    y_center=_np.asarray(sd["sgbs"], dtype=float),
                    q16_vec=_np.asarray(sd["sgbs_q16"], dtype=float),
                    q84_vec=_np.asarray(sd["sgbs_q84"], dtype=float),
                    color=colour, ls=ls,
                )
                for model in ("hill", "richards"):
                    par, r2, ci = _fit(ref, unit, name, model)
                    entry[f"{model}_params"] = par
                    entry[f"{model}_r2"] = r2
                    entry[f"{model}_asymptote_ci"] = ci
                    if entry.get("n_mic") is None and fits is not None:
                        m = fits[(fits["reference"] == ref)
                                 & (fits["unit"] == unit)
                                 & (fits["series"] == name)]
                        if not m.empty:
                            entry["n_mic"] = m.iloc[0]["n_mic"]
                series_list.append(entry)
            by_unit.append(series_list)
        panels[ref] = tuple(by_unit)
    return panels


def plot_figure3(tables, out_dir, model="hill", threshold=10.0,
                 output_prefix=None, y_label="SGBs detected"):
    """Draw Figure 3 from its published table, and from nothing else."""
    from pathlib import Path as _P
    out_dir = _P(out_dir)
    if output_prefix is None:
        output_prefix = f"figure3_combined_regenerated_{threshold_tag(threshold)}"
    return make_figure(out_dir, threshold, model, panels_from_tables(tables),
                       output_prefix=output_prefix, y_label=y_label)


def rebuild_from_data_object(obj, out_dir, model="richards", threshold=10.0):
    """Redraw main-text Figure 3 from its published data object."""
    return plot_figure3(obj["panels"], out_dir, model=model,
                        threshold=threshold, output_prefix="figure3")


PRINT_WIDTH_IN = 7.24        # Science, three columns (18.4 cm)
AUTHORED_WIDTH_IN = 19.6     # the width this figure was composed at

#: Everything measured in points -- type, rules, markers, pads -- has to be
#: multiplied by this for the figure to reduce to the column width and look
#: exactly as it was composed. Scaling the canvas alone leaves the type and
#: the rules at their old absolute sizes, which is what wrecked the layout
#: the first time this was tried.
FIG3_K = PRINT_WIDTH_IN / AUTHORED_WIDTH_IN

#: Room at the top of the figure for the "Samples" / "Studies" column
#: headings, as a fraction of figure height, and the gap between a heading and
#: the panels underneath it.
COLUMN_HEADER_TOP = 0.947
COLUMN_HEADER_GAP = 0.009


def _k(v, floor=None):
    """Scale a point-valued quantity to the printed width."""
    out = v * FIG3_K
    return out if floor is None else max(out, floor)


def make_figure(
    out_dir: Path,
    threshold: float,
    model: str,
    panels,
    *,
    output_prefix: str,
    y_label: str = "SGBs detected",
    data_slot: str = "",
) -> Path:
    apply_oi_style()
    # Science caps a figure at 18.4 cm. This one was authored at 19.6 in, so
    # everything on it -- type, rules, markers -- is reduced to 37% on the
    # page: type landed near 6 pt and rules near 0.4 pt. Scaling the canvas
    # and every size on it by the same factor keeps the composition exactly
    # as it was tuned while making the printed sizes the ones written here.
    # Type and rules are floored at the Science minimums, which is the only
    # place the proportions change.
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
            # ── Figure 3 is tuned; hold it independent of the shared style ──
            # apply_oi_style() sets twenty rcParams and this figure only
            # overrode six, so a later change to the shared page-width policy
            # (smaller titles, tighter margins and label pads, an 8 pt legend)
            # would silently reshape a figure nobody meant to touch. These are
            # the values that were in force when it was tuned.
            "axes.titlesize": 14,
            "figure.titlesize": 14,
            "legend.fontsize": 12,
            "axes.labelpad": 4.0,
            "xtick.major.pad": 3.5,
            "ytick.major.pad": 3.5,
            "figure.subplot.left": 0.125,
            "figure.subplot.right": 0.9,
            "figure.subplot.top": 0.88,
            "figure.subplot.bottom": 0.11,
            "figure.subplot.wspace": 0.2,
            "figure.subplot.hspace": 0.2,
        }
    )

    # Authored at the printed width, so lines and symbols keep their
    # size through the reduction.
    # Every point-valued rcParam, not just the type: tick marks, spines,
    # pads and default marker size are in points as well, and leaving them
    # unscaled is what made the ticks tower over the shrunken panels.
    _moved = scale_point_rcparams(FIG3_K)
    print(f"  [scale] figure 3 to {PRINT_WIDTH_IN} in: "
          f"{len(_moved)} point-valued rcParams scaled by {FIG3_K:.3f}")

    fig, axes = plt.subplots(
        3, 2, figsize=(AUTHORED_WIDTH_IN * FIG3_K, 17.4 * FIG3_K),
        sharex=False, sharey=False)
    for row, ref in enumerate(("UHGG", "lev", "combined")):
        sample_series, study_series = panels[ref]
        _draw_one_axis(
            axes[row, 0],
            sample_series,
            model=model,
            unit_label="samples",
            y_label=y_label,
            row=row,
            show_xlabel=(row == 2),
            show_ylabel_ticks=True,
            show_xticks=(row == 2),
            show_legend=False,
        )
        _draw_one_axis(
            axes[row, 1],
            study_series,
            model=model,
            unit_label="studies",
            y_label=y_label,
            row=row,
            show_xlabel=(row == 2),
            show_ylabel_ticks=False,
            show_xticks=(row == 2),
            show_legend=(row == 0),
        )

    for label, ax in zip("ABCDEF", axes.flat):
        ax.text(0.012, 0.965, label, transform=ax.transAxes,
                fontsize=_k(23), fontweight="bold", va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=_k(0.8)))
    # The top of the rect is pulled down to open a band for the column
    # headings; everything else about the layout is left alone.
    fig.tight_layout(rect=(0.025, 0.025, 0.902, COLUMN_HEADER_TOP),
                     h_pad=0.05, w_pad=0.18)

    # What distinguishes the two columns is the unit the curve accumulates
    # over -- samples on the left, studies on the right. Saying it once at the
    # head of each column is clearer than repeating it in six axis labels.
    for _col, _label in ((0, "Samples"), (1, "Studies")):
        _pos = axes[0, _col].get_position()
        _hdr = fig.text(
            (_pos.x0 + _pos.x1) / 2.0,
            _pos.y1 + COLUMN_HEADER_GAP,
            _label,
            ha="center",
            va="bottom",
            fontsize=_k(21),
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.34", facecolor="#F2F3F5",
                      edgecolor="#C9CDD4", linewidth=_k(1.0, 0.5)),
        )
        # Layout, not a title: the no-title twin (the one placed as S15)
        # must keep it.
        _hdr._keep_in_notitle = True

    row_centers = []
    for row in range(3):
        pos = axes[row, 1].get_position()
        row_centers.append((pos.x1 + 0.018, (pos.y0 + pos.y1) / 2.0))
    for (x, y), ref in zip(row_centers, ("UHGG", "lev", "combined")):
        _side = fig.text(
            x,
            y,
            SIDE_LABELS[ref],
            ha="center",
            va="center",
            rotation=270,
            fontsize=_k(19),
        )
        _side._keep_in_notitle = True

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{output_prefix}_{model}.png"
    # The figure's data object: the accumulation curves the panels are fitted
    # to, per reference and series. The fitted asymptotes drawn on top are
    # derived from these points by the model named in the filename.
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from persistence import figure_source as _fsrc
        import pandas as _pd, numpy as _np
        _rows, _fits = [], []
        for _ref_name, _sets in (panels or {}).items():
            # Six panels, not three: each reference is drawn once against
            # sample count and once against study count, and they are
            # different curves. Tagging the unit keeps both in one table.
            _by_unit = (zip(("samples", "studies"), _sets)
                        if isinstance(_sets, (list, tuple))
                        else zip((), ()))
            for _unit, _series in _by_unit:
                for _sd in (_series or []):
                    _n = _np.asarray(_sd.get("N_vec", []), dtype=float)
                    # The shaded band is the 16th-84th percentile of the bootstrap
                    # at each sample count, so it travels with the centre line.
                    _rows.append(_pd.DataFrame({
                        "reference": _ref_name,
                        "unit": _unit,
                        "series": _sd.get("name", ""),
                        "n_units": _n,
                        "sgbs": _np.asarray(_sd.get("y_center", []), dtype=float),
                        "sgbs_q16": _np.asarray(_sd.get("q16_vec", []), dtype=float),
                        "sgbs_q84": _np.asarray(_sd.get("q84_vec", []), dtype=float),
                    }))
                    # The fitted curve and the asymptote quoted on each panel are
                    # not recoverable from the points alone, because the asymptote
                    # interval comes from refitting every bootstrap replicate.
                    for _m in ("hill", "richards"):
                        _par = _sd.get(f"{_m}_params")
                        _ci = _sd.get(f"{_m}_asymptote_ci") or (None, None, None)
                        if _par is None:
                            continue
                        _fits.append({
                            "reference": _ref_name,
                            "unit": _unit,
                            "series": _sd.get("name", ""),
                            "model": _m,
                            "r2": _sd.get(f"{_m}_r2"),
                            "n_mic": _sd.get("n_mic"),
                            "asymptote_q16": _ci[0],
                            "asymptote_q50": _ci[1],
                            "asymptote_q84": _ci[2],
                            "params": ";".join(f"{v:.10g}"
                                               for v in _np.asarray(_par, float).ravel()),
                        })
        if _rows:
            _t = {"accumulation": _pd.concat(_rows, ignore_index=True)}
            if _fits:
                _t["fits"] = _pd.DataFrame(_fits)
            if data_slot:
                # The main-text build lives outside the results tree, so its
                # path resolves to no slot; the caller that knows which figure
                # this is names the slot instead.
                _fsrc.emit(str(out_path.parent), data_slot, _t,
                           stem=out_path.stem, reference="combined")
            else:
                _fsrc.emit_for_path(str(out_path), _t)
    except Exception as _exc:
        print(f"  [data] Figure 3 table unavailable: {_exc}")

    savefig_dual_preserve_layout(fig, out_path, dpi=300)
    savefig_dual_preserve_layout(fig, out_path.with_suffix(".pdf"), dpi=300)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", type=Path)
    parser.add_argument("--out-dir", default=Path("results/Figure3_combined"), type=Path)
    parser.add_argument("--threshold", default=10.0, type=float)
    parser.add_argument(
        "--min-occurrences",
        default=1,
        type=int,
        help="Require each SGB to be detected in at least this many sampled samples/studies.",
    )
    parser.add_argument("--models", nargs="+", default=None, choices=["richards", "hill", "rsac"])
    parser.add_argument("--cache", default=None, type=Path)
    parser.add_argument(
        "--data-slot", default="",
        help="Force the figure-data slot for this build (e.g. F3). The "
             "main-text build writes outside results/, where the path-based "
             "slot lookup finds nothing.")
    parser.add_argument(
        "--match-existing-rng",
        action="store_true",
        help="Advance through earlier configured thresholds before 10%, matching run_figure3_accumulation's RNG state.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    tag = threshold_tag(args.threshold)
    if args.min_occurrences > 1:
        output_prefix = f"figure3_combined_regenerated_{tag}_min{args.min_occurrences}"
        y_label = f"SGBs detected in >= {args.min_occurrences}"
    else:
        output_prefix = f"figure3_combined_regenerated_{tag}"
        y_label = "SGBs detected"
    models = args.models if args.models is not None else (
        ["rsac"] if args.min_occurrences > 1 else ["richards", "hill"]
    )
    cache_suffix = (
        "_rsac_bootstrap_replicates_cache.pkl"
        if args.min_occurrences > 1
        else "_bootstrap_cache.pkl"
    )
    default_cache = out_dir / f"{output_prefix}{cache_suffix}"
    cache_path = args.cache if args.cache is not None else default_cache
    cache_path = cache_path if cache_path.is_absolute() else root / cache_path
    panels = load_or_build_panels(
        cache_path,
        args.threshold,
        match_existing_rng=args.match_existing_rng,
        min_occurrences=args.min_occurrences,
    )
    for model in models:
        print(f"\nRendering {model}...", flush=True)
        out_path = make_figure(
            out_dir,
            args.threshold,
            model,
            panels,
            output_prefix=output_prefix,
            y_label=y_label,
            data_slot=args.data_slot,
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
