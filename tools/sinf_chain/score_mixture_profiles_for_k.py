#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from profile_sinf_mixture import (
    DEFAULT_WORK,
    ROOT,
    curve_metrics,
    fixed_shape_bootstrap_sinf,
    interval_from_profile,
    plot_best_asymptotes,
    plot_r2,
)


def safe_series_name(name):
    return str(name).replace("-", "")


def profile_path(profiles_dir, h, ref, unit, series):
    pattern = f"profile_mix_*_H{h}_{ref}_{unit}_{safe_series_name(series)}.csv"
    matches = sorted(profiles_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No profile CSV found for H={h}, {ref}, {unit}, {series} in {profiles_dir}"
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
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

    summary_rows = []
    # Score only what was scanned. The bootstrap cache carries UHGG, lev and
    # combined, but the scan is normally limited to combined, so the others
    # have no profile CSVs to read. Without this the scorer wrote the combined
    # rows and then died on the first missing reference -- harmless only
    # because combined happened to come first in the cache.
    _avail = []
    for _ref in list(panels):
        if sorted(args.profiles_dir.glob(f"profile_mix_k{k}_H*_{_ref}_*.csv")):
            _avail.append(_ref)
    if _avail and len(_avail) < len(panels):
        print(f"  scoring {_avail}; no profiles on disk for "
              f"{[r for r in panels if r not in _avail]}", flush=True)
        panels = {r: v for r, v in panels.items() if r in _avail}

    for ref, (sample_series, study_series) in panels.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False, sharey=False)
        fig.suptitle(f"k={k} beta-mixture S_inf profiles: {ref}", fontsize=15)
        for col, (unit, series_list) in enumerate([("samples", sample_series), ("studies", study_series)]):
            for row_idx, sd in enumerate(series_list):
                ax = axes[row_idx, col]
                counts = np.asarray(sd["incidence_counts"], dtype=int)
                m_catalog = int(sd["n_mic"])
                s_positive = int(np.count_nonzero(counts > 0))
                s_ge_k = int(np.count_nonzero(counts >= k))

                for h in (1, 2, 3):
                    src = profile_path(args.profiles_dir, h, ref, unit, sd["name"])
                    profile = pd.read_csv(src)
                    profile["k"] = k
                    profile.to_csv(out / f"profile_mix_k{k}_H{h}_{ref}_{unit}_{safe_series_name(sd['name'])}.csv", index=False)
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
                        "n_units": int(sd["unit_count"]),
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
