#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, logsumexp


# Repo-relative, like the rest of the pipeline. These were absolute paths into
# the original working session and a home directory outside the repository, so
# the script read its bootstrap caches from one tree and wrote its profile CSVs
# into another. score_mixture_profiles_for_k.py imports both names from here,
# so fixing them here fixes the whole chain.
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK = ROOT / "results" / "Figure3_combined"


def beta_binom_logpmf(r, n, alpha, beta):
    r = np.asarray(r, dtype=float)
    n = np.asarray(n, dtype=float)
    return (
        gammaln(n + 1.0)
        - gammaln(r + 1.0)
        - gammaln(n - r + 1.0)
        + betaln(r + alpha, n - r + beta)
        - betaln(alpha, beta)
    )


def beta_binom_tail(n, k, alpha, beta):
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n, dtype=float)
    valid = n >= k
    if not np.any(valid):
        return out
    xv = n[valid]
    cdf = np.zeros_like(xv, dtype=float)
    for r in range(k):
        cdf += np.exp(beta_binom_logpmf(r, xv, alpha, beta))
    out[valid] = np.clip(1.0 - cdf, 0.0, 1.0)
    return out


def decode_theta(theta, h):
    theta = np.asarray(theta, dtype=float)
    if h == 1:
        weights = np.array([1.0])
        offset = 0
    else:
        logits = np.r_[theta[: h - 1], 0.0]
        logits -= np.max(logits)
        weights = np.exp(logits)
        weights /= weights.sum()
        offset = h - 1
    alphas = np.exp(theta[offset : offset + h])
    betas = np.exp(theta[offset + h : offset + 2 * h])
    return weights, alphas, betas


def encode_start(weights, alphas, betas):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    if weights.size == 1:
        logits = []
    else:
        logits = np.log(weights[:-1]) - np.log(weights[-1])
    return np.r_[logits, np.log(alphas), np.log(betas)]


def mix_logpmf(values, n_units, weights, alphas, betas):
    terms = []
    for w, a, b in zip(weights, alphas, betas):
        terms.append(np.log(w + 1e-300) + beta_binom_logpmf(values, n_units, a, b))
    return logsumexp(np.vstack(terms), axis=0)


def mix_tail(n_vec, k, weights, alphas, betas):
    out = np.zeros_like(np.asarray(n_vec, dtype=float), dtype=float)
    for w, a, b in zip(weights, alphas, betas):
        out += w * beta_binom_tail(n_vec, k, a, b)
    return out


def starts_for_h(h):
    component_sets = {
        1: [
            ([1.0], [0.1], [100.0]),
            ([1.0], [0.2], [5.0]),
            ([1.0], [0.7], [1.0]),
        ],
        2: [
            ([0.75, 0.25], [0.2, 1.2], [5.0, 1.0]),
            ([0.35, 0.65], [0.08, 0.7], [100.0, 2.0]),
            ([0.50, 0.50], [0.3, 2.0], [10.0, 5.0]),
            ([0.85, 0.15], [0.15, 3.0], [3.0, 20.0]),
        ],
        3: [
            ([0.55, 0.35, 0.10], [0.08, 0.5, 2.0], [100.0, 5.0, 2.0]),
            ([0.33, 0.34, 0.33], [0.05, 0.4, 1.5], [50.0, 3.0, 1.0]),
            ([0.70, 0.20, 0.10], [0.2, 1.0, 5.0], [5.0, 2.0, 10.0]),
        ],
    }
    return [encode_start(*spec) for spec in component_sets[h]]


def make_grid(s_positive, m_catalog, n=54):
    if s_positive >= m_catalog:
        return np.array([float(m_catalog)])
    linear = np.linspace(s_positive, m_catalog, n)
    near_cap = m_catalog - (m_catalog - s_positive) * np.geomspace(1.0, 1e-4, max(18, n // 3))
    grid = np.unique(np.r_[linear, near_cap, m_catalog])
    return grid[(grid >= s_positive) & (grid <= m_catalog)]


def interval_from_profile(profile, cutoff=3.84):
    good = profile[np.isfinite(profile["delta"]) & (profile["delta"] <= cutoff)]
    if good.empty:
        return np.nan, np.nan
    return float(good["S_inf"].min()), float(good["S_inf"].max())


def profile_sinf_mixbeta(counts, n_units, m_catalog, grid_s, h):
    counts = np.asarray(counts, dtype=int)
    values, freqs = np.unique(counts, return_counts=True)
    zero = values == 0
    nonzero = ~zero
    starts = starts_for_h(h)
    bounds = [(-8.0, 8.0)] * max(0, h - 1) + [
        (np.log(1e-5), np.log(1e5))
    ] * h + [(np.log(1e-3), np.log(1e9))] * h
    rows = []
    warm = None

    for s_inf in grid_s:
        rho = float(np.clip(s_inf / m_catalog, 1e-12, 1.0))
        log_rho = np.log(rho)
        log_absent = -np.inf if rho >= 1.0 else np.log1p(-rho)

        def nll(theta):
            weights, alphas, betas = decode_theta(theta, h)
            log_q = mix_logpmf(values, n_units, weights, alphas, betas)
            log_prob = np.empty_like(log_q, dtype=float)
            if np.any(zero):
                log_prob[zero] = np.logaddexp(log_absent, log_rho + log_q[zero])
            if np.any(nonzero):
                log_prob[nonzero] = log_rho + log_q[nonzero]
            val = -float(np.sum(freqs * log_prob))
            return val if np.isfinite(val) else 1e100

        candidates = []
        if warm is not None:
            candidates.append(warm)
        candidates.extend(starts)
        best = None
        for x0 in candidates:
            res = minimize(
                nll,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1500, "ftol": 1e-8},
            )
            if res.success and np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                best = res
        if best is None:
            rows.append({"S_inf": s_inf, "nll": np.nan})
            continue
        warm = best.x
        weights, alphas, betas = decode_theta(best.x, h)
        row = {
            "S_inf": float(s_inf),
            "rho": rho,
            "pi0": 1.0 - rho,
            "nll": float(best.fun),
        }
        for idx, (w, a, b) in enumerate(zip(weights, alphas, betas), start=1):
            row[f"w{idx}"] = float(w)
            row[f"alpha{idx}"] = float(a)
            row[f"beta{idx}"] = float(b)
        rows.append(row)

    out = pd.DataFrame(rows)
    out["delta"] = 2.0 * (out["nll"] - out["nll"].min())
    return out


def params_from_row(row, h):
    weights = np.array([row.get(f"w{i}", np.nan) for i in range(1, h + 1)], dtype=float)
    alphas = np.array([row.get(f"alpha{i}", np.nan) for i in range(1, h + 1)], dtype=float)
    betas = np.array([row.get(f"beta{i}", np.nan) for i in range(1, h + 1)], dtype=float)
    return weights, alphas, betas


def curve_metrics(sd, k, row, h):
    n_vec = np.asarray(sd["N_vec"], dtype=float)
    y_obs = np.asarray(sd["y_center"], dtype=float)
    weights, alphas, betas = params_from_row(row, h)
    shape = mix_tail(n_vec, k, weights, alphas, betas)
    y_fit = float(row["S_inf"]) * shape
    ss_res = float(np.sum((y_obs - y_fit) ** 2))
    ss_tot = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y_obs - y_fit) ** 2)))
    return r2, rmse


def fixed_shape_bootstrap_sinf(sd, k, row, h):
    n_vec = np.asarray(sd["N_vec"], dtype=float)
    boot = np.asarray(sd.get("boot_matrix"), dtype=float)
    if boot.ndim != 2 or boot.shape[1] != n_vec.size:
        return {}
    weights, alphas, betas = params_from_row(row, h)
    shape = mix_tail(n_vec, k, weights, alphas, betas)
    denom = float(np.sum(shape * shape))
    if denom <= 0:
        return {}
    m_catalog = float(sd["n_mic"])
    s_lower = float(np.count_nonzero(np.asarray(sd["incidence_counts"], dtype=int) > 0))
    s_vals = np.clip((boot @ shape) / denom, s_lower, m_catalog)
    return {
        "S_inf_boot_median_fixed_shape": float(np.median(s_vals)),
        "S_inf_boot_min_fixed_shape": float(np.min(s_vals)),
        "S_inf_boot_max_fixed_shape": float(np.max(s_vals)),
        "S_inf_boot_q16_fixed_shape": float(np.quantile(s_vals, 0.16)),
        "S_inf_boot_q84_fixed_shape": float(np.quantile(s_vals, 0.84)),
        "S_inf_boot_n": int(s_vals.size),
    }


def plot_r2(summary, out, k):
    panel_keys = summary[["ref", "unit", "series"]].drop_duplicates().to_dict("records")
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharey=True)
    for ax, key in zip(axes.ravel(), panel_keys):
        sub = summary[
            (summary["ref"] == key["ref"])
            & (summary["unit"] == key["unit"])
            & (summary["series"] == key["series"])
        ].sort_values("H")
        colors = ["#4E79A7", "#59A14F", "#E15759"]
        ax.bar(sub["H"].astype(str), sub["curve_r2_at_profile_mle"], color=colors[: len(sub)])
        best = sub.loc[sub["curve_r2_at_profile_mle"].idxmax()]
        ax.set_title(f"{key['ref']} {key['unit']}\n{key['series']}", fontsize=9)
        ax.set_xlabel("H")
        ax.text(
            0.02,
            0.05,
            f"best H={int(best['H'])}\nR2={best['curve_r2_at_profile_mle']:.4f}",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
        )
        ax.grid(axis="y", color="0.9", lw=0.8)
    for ax in axes[:, 0]:
        ax.set_ylabel("curve R2")
    fig.suptitle(f"k={k}: beta-mixture profile models, curve R2 by panel", fontsize=15)
    fig.tight_layout()
    fig.savefig(out / f"profile_mix_k{k}_r2_by_H.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_best_asymptotes(summary, out, k):
    best = (
        summary.sort_values(["ref", "unit", "series", "hist_bic_profile_params"])
        .groupby(["ref", "unit", "series"], as_index=False)
        .head(1)
        .copy()
    )
    best["label"] = best["ref"] + "\n" + best["unit"] + " " + best["series"]
    x = np.arange(len(best))
    fig, ax = plt.subplots(figsize=(16, 6))
    y = best["S_inf_boot_median_fixed_shape"].fillna(best["S_inf_mle"]).to_numpy(float)
    low = best["S_inf_boot_min_fixed_shape"].fillna(best["S_inf_profile95_lower"]).to_numpy(float)
    high = best["S_inf_boot_max_fixed_shape"].fillna(best["S_inf_profile95_upper"]).to_numpy(float)
    yerr = np.vstack([y - low, high - y])
    ax.errorbar(x, y, yerr=yerr, fmt="o", color="#4E79A7", ecolor="0.45", capsize=4, lw=1.5)
    ax.scatter(x, best["M_catalog"], marker="_", s=220, color="#D55E00", label="catalog M")
    ax.set_xticks(x)
    ax.set_xticklabels(best["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("S_inf")
    ax.set_title(f"k={k}: selected-H asymptote medians and bootstrap ranges")
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"profile_mix_k{k}_best_asymptote_ranges.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--refs", nargs="+", default=None,
        help="Limit the scan to these references. The bootstrap cache carries "
             "UHGG, lev and combined, and profiling all three costs three "
             "times as long; the Additional Figures need combined alone.")
    args = parser.parse_args()

    k = args.k
    cache = args.cache
    if cache is None:
        cache = ROOT / f"results/Figure3_combined/figure3_combined_regenerated_10pct_min{k}_rsac_bootstrap_replicates_cache.pkl"
    out = args.out or (DEFAULT_WORK / f"profile_sinf_mixture_min{k}")
    out.mkdir(parents=True, exist_ok=True)

    with cache.open("rb") as fh:
        payload = pickle.load(fh)
    panels = payload["panels"]
    cache_k = int(payload.get("min_occurrences", k))
    if cache_k != k:
        raise ValueError(f"Cache min_occurrences={cache_k}, requested k={k}")

    if args.refs:
        keep = set(args.refs)
        dropped = [r for r in panels if r not in keep]
        panels = {r: v for r, v in panels.items() if r in keep}
        if dropped:
            print(f"  scanning {sorted(panels)}; skipping {sorted(dropped)}",
                  flush=True)

    summary_rows = []
    for ref, (sample_series, study_series) in panels.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey=False)
        fig.suptitle(f"k={k} beta-mixture S_inf profiles: {ref}", fontsize=15)
        for col, (unit, series_list) in enumerate([("samples", sample_series), ("studies", study_series)]):
            for row_idx, sd in enumerate(series_list):
                ax = axes[row_idx, col]
                counts = np.asarray(sd["incidence_counts"], dtype=int)
                n_units = int(sd["unit_count"])
                m_catalog = int(sd["n_mic"])
                s_positive = int(np.count_nonzero(counts > 0))
                s_ge_k = int(np.count_nonzero(counts >= k))
                grid = make_grid(s_positive, m_catalog)

                for h in (1, 2, 3):
                    print(f"Profiling k={k} {ref} {unit} {sd['name']} H={h}", flush=True)
                    profile = profile_sinf_mixbeta(counts, n_units, m_catalog, grid, h)
                    profile["H"] = h
                    profile["ref"] = ref
                    profile["unit"] = unit
                    profile["series"] = sd["name"]
                    profile["k"] = k
                    profile["n_units"] = n_units
                    profile["M_catalog"] = m_catalog
                    profile["S_positive_observed"] = s_positive
                    profile["S_ge_k_observed"] = s_ge_k
                    safe_series = sd["name"].replace("-", "")
                    profile.to_csv(out / f"profile_mix_k{k}_H{h}_{ref}_{unit}_{safe_series}.csv", index=False)

                    best = profile.loc[profile["nll"].idxmin()]
                    lo95, hi95 = interval_from_profile(profile, 3.84)
                    delta_m = float(profile.iloc[(profile["S_inf"] - m_catalog).abs().argmin()]["delta"])
                    r2, rmse = curve_metrics(sd, k, best, h)
                    n_params_profile = (h - 1) + 2 * h
                    bic_hist = 2.0 * float(best["nll"]) + n_params_profile * np.log(m_catalog)
                    summary = {
                        "k": k,
                        "ref": ref,
                        "unit": unit,
                        "series": sd["name"],
                        "H": h,
                        "M_catalog": m_catalog,
                        "n_units": n_units,
                        "S_positive_observed": s_positive,
                        f"S_ge_{k}_observed": s_ge_k,
                        "S_inf_mle": float(best["S_inf"]),
                        "S_inf_mle_at_catalog_cap": bool(abs(float(best["S_inf"]) - m_catalog) < 1e-6),
                        "S_inf_profile95_lower": lo95,
                        "S_inf_profile95_upper": hi95,
                        "delta_at_M_catalog": delta_m,
                        "hist_nll": float(best["nll"]),
                        "hist_bic_profile_params": bic_hist,
                        "curve_r2_at_profile_mle": r2,
                        "curve_rmse_at_profile_mle": rmse,
                    }
                    summary.update(fixed_shape_bootstrap_sinf(sd, k, best, h))
                    for i in range(1, h + 1):
                        summary[f"w{i}"] = float(best.get(f"w{i}", np.nan))
                        summary[f"alpha{i}"] = float(best.get(f"alpha{i}", np.nan))
                        summary[f"beta{i}"] = float(best.get(f"beta{i}", np.nan))
                    summary_rows.append(summary)

                    ax.plot(profile["S_inf"], profile["delta"], lw=1.6, label=f"H={h}")

                ax.axhline(3.84, color="0.5", ls="--", lw=1, label="95% cutoff")
                ax.axvline(s_positive, color="#0072B2", ls=":", lw=1.3, label="S observed >0")
                ax.axvline(m_catalog, color="#D55E00", ls=":", lw=1.3, label="M catalog")
                ax.set_title(f"{unit}: {sd['name']}")
                ax.set_xlabel("S_inf")
                ax.set_ylabel("profile delta")
                ax.set_ylim(bottom=0)
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"profile_mix_k{k}_{ref}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary_path = out / f"profile_mix_k{k}_summary.csv"
    summary.to_csv(summary_path, index=False)
    ranked = summary.sort_values(["ref", "unit", "series", "hist_bic_profile_params"])
    ranked_path = out / f"profile_mix_k{k}_ranked_by_bic.csv"
    ranked.to_csv(ranked_path, index=False)
    plot_r2(summary, out, k)
    plot_best_asymptotes(summary, out, k)

    best = ranked.groupby(["ref", "unit", "series"], as_index=False).head(1)
    best_path = out / f"profile_mix_k{k}_best_by_bic_asymptotes.csv"
    best.to_csv(best_path, index=False)
    print(summary_path)
    print(ranked_path)
    print(best_path)
    keep = [
        "ref",
        "unit",
        "series",
        "H",
        "M_catalog",
        "S_inf_mle",
        "S_inf_profile95_lower",
        "S_inf_profile95_upper",
        "S_inf_boot_median_fixed_shape",
        "S_inf_boot_min_fixed_shape",
        "S_inf_boot_max_fixed_shape",
        "curve_r2_at_profile_mle",
        "delta_at_M_catalog",
    ]
    print(best[keep].to_string(index=False))


if __name__ == "__main__":
    main()
