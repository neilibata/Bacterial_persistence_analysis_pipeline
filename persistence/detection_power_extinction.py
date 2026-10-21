#!/usr/bin/env python3
"""
Detection-power-aware extinction diagnostics.

This module asks a narrower question than a raw accumulation curve:

    If catalogue taxon k were extant in group g, how likely was this study
    design to detect it at least once under the 10% breadth rule?

The output is taxon-specific Q_gk = P(miss taxon k in group g | extant),
pi_gk = 1 - Q_gk, Bayes factors for absence among missing taxa, and
posterior group asymptotes under a user-specified prior probability of
group-level extance.

The detection model is intentionally empirical:

1. Observed detection is breadth >= tau.
2. Weak sample-level evidence/presence is breadth > weak_tau and mean_depth > 0.
3. Conditional abundance values are estimated from the depth matrix as

       relative_abundance_ik = mean_depth_ik * genome_length_k / total_mapped_bp_i

4. A taxon passes a tau breadth rule when expected mean depth reaches

       c_tau = -log(1 - tau)

   because random coverage has expected breadth 1 - exp(-c).

5. For a target sample with total mapped bp D_i, taxon k and abundance a
   crosses the breadth rule when

       D_i * a / genome_length_k >= c_tau.

This separates:
    - sample number and depth distribution,
    - genome-length-dependent breadth-threshold censoring,
    - empirical sample-level prevalence,
    - empirical abundance when weakly present,
    - true group-level absence/posterior extance.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _import_pipeline_loader():
    package_dir = Path(__file__).resolve().parent
    project_dir = package_dir.parent
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    try:
        from persistence.loading import load_reference
        return load_reference
    except Exception as exc:
        raise RuntimeError(
            "Run from the full_pipeline_v10 directory, e.g. "
            "`/opt/anaconda3/bin/python -u persistence/detection_power_extinction.py`"
        ) from exc


def posterior_presence_if_missing(prior_extant: float, q_miss: np.ndarray) -> np.ndarray:
    rho = float(prior_extant)
    denom = (1.0 - rho) + rho * q_miss
    return np.divide(rho * q_miss, denom, out=np.zeros_like(q_miss, dtype=float), where=denom > 0)


def poisson_binomial_interval(probs: np.ndarray, rng: np.random.Generator, n_draws: int) -> tuple[float, float, float]:
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return 0.0, 0.0, 0.0
    draws = rng.random((n_draws, probs.size)) < probs
    totals = draws.sum(axis=1)
    return (
        float(np.mean(totals)),
        float(np.quantile(totals, 0.025)),
        float(np.quantile(totals, 0.975)),
    )


def group_index(meta: pd.DataFrame, group_col: str, values: list[str]) -> pd.Index:
    if group_col not in meta.columns:
        raise ValueError(f"Group column {group_col!r} not found in metadata")
    return meta.index[meta[group_col].isin(values)]


def empirical_detection_power_for_group(
    *,
    ref_name: str,
    group_name: str,
    target_ids: pd.Index,
    donor_ids: pd.Index,
    breadth: pd.DataFrame,
    depth: pd.DataFrame,
    lengths: pd.Series,
    total_bp: pd.Series,
    standard_total_bp: np.ndarray,
    threshold_pct: float,
    weak_breadth_pct: float,
    min_within_weak: int,
    prior_extant: float,
) -> pd.DataFrame:
    taxa = breadth.columns
    n_taxa = len(taxa)
    n_target = len(target_ids)

    breadth_g = breadth.loc[target_ids]
    depth_g = depth.loc[target_ids]
    breadth_donor = breadth.loc[donor_ids]
    depth_donor = depth.loc[donor_ids]

    total_bp_g = total_bp.reindex(target_ids).astype(float).values
    total_bp_donor = total_bp.reindex(donor_ids).astype(float)

    c_tau = -math.log(1.0 - threshold_pct / 100.0)
    observed_any = (breadth_g >= threshold_pct).any(axis=0).values
    weak_any = ((breadth_g > weak_breadth_pct) & (depth_g > 0)).any(axis=0).values

    rows = []
    for j, taxon in enumerate(taxa):
        genome_len = float(lengths.get(taxon, np.nan))
        if not np.isfinite(genome_len) or genome_len <= 0:
            rows.append({
                "reference": ref_name,
                "group": group_name,
                "taxon": taxon,
                "n_group_samples": n_target,
                "genome_length_bp": np.nan,
                "observed_detected": bool(observed_any[j]),
                "weak_evidence_in_group": bool(weak_any[j]),
                "model_source": "no_length",
                "source_n_samples": 0,
                "source_weak_count": 0,
                "q_hat_weak_prevalence": 0.0,
                "abundance_median_when_weak": np.nan,
                "Q_miss_if_extant": 1.0,
                "pi_detect_if_extant": 0.0,
                "Q_miss_if_extant_standard_design": 1.0,
                "pi_detect_if_extant_standard_design": 0.0,
                "BF_absent_vs_present_if_missing": 1.0,
                "posterior_present": 1.0 if observed_any[j] else prior_extant,
                "posterior_absent": 0.0 if observed_any[j] else 1.0 - prior_extant,
            })
            continue

        target_weak = ((breadth_g.iloc[:, j] > weak_breadth_pct) & (depth_g.iloc[:, j] > 0)).values
        if int(target_weak.sum()) >= min_within_weak:
            source_name = "within_group"
            source_depth = depth_g.iloc[:, j]
            source_total_bp = total_bp.reindex(target_ids)
            source_weak = pd.Series(target_weak, index=target_ids)
            source_n = n_target
        else:
            source_name = "donor_not_group"
            donor_weak = ((breadth_donor.iloc[:, j] > weak_breadth_pct) & (depth_donor.iloc[:, j] > 0))
            source_depth = depth_donor.iloc[:, j]
            source_total_bp = total_bp_donor
            source_weak = donor_weak
            source_n = len(donor_ids)

        source_weak_count = int(source_weak.sum())
        if source_n <= 0 or source_weak_count <= 0:
            q_hat = 0.0
            abund = np.array([], dtype=float)
        else:
            q_hat = (source_weak_count + 0.5) / (source_n + 1.0)
            abund = (
                source_depth.loc[source_weak.index[source_weak]].astype(float)
                * genome_len
                / source_total_bp.loc[source_weak.index[source_weak]].astype(float)
            ).replace([np.inf, -np.inf], np.nan).dropna().values
            abund = abund[abund > 0]

        def _miss_for_design(design_total_bp: np.ndarray) -> tuple[float, float]:
            if abund.size == 0 or q_hat <= 0:
                return 1.0, 0.0
            abund_sorted_local = np.sort(abund)
            needed_abund = c_tau * genome_len / np.clip(design_total_bp, 1.0, None)
            n_ge = abund_sorted_local.size - np.searchsorted(abund_sorted_local, needed_abund, side="left")
            alpha_i = n_ge.astype(float) / abund_sorted_local.size
            r_i = np.clip(q_hat * alpha_i, 0.0, 1.0 - 1e-12)
            log_q = np.log1p(-r_i).sum()
            q_local = float(np.exp(max(log_q, -745.0)))
            return q_local, 1.0 - q_local

        if abund.size == 0 or q_hat <= 0:
            q_miss = 1.0
            pi_detect = 0.0
            q_miss_std = 1.0
            pi_detect_std = 0.0
            abund_median = np.nan
        else:
            q_miss, pi_detect = _miss_for_design(total_bp_g)
            q_miss_std, pi_detect_std = _miss_for_design(standard_total_bp)
            abund_median = float(np.median(abund))

        if observed_any[j]:
            post_present = 1.0
        else:
            post_present = float(posterior_presence_if_missing(prior_extant, np.array([q_miss]))[0])
        post_absent = 1.0 - post_present
        bf = float(np.inf if q_miss <= 0 else 1.0 / q_miss)

        rows.append({
            "reference": ref_name,
            "group": group_name,
            "taxon": taxon,
            "n_group_samples": n_target,
            "genome_length_bp": genome_len,
            "observed_detected": bool(observed_any[j]),
            "weak_evidence_in_group": bool(weak_any[j]),
            "model_source": source_name,
            "source_n_samples": int(source_n),
            "source_weak_count": source_weak_count,
            "q_hat_weak_prevalence": float(q_hat),
            "abundance_median_when_weak": abund_median,
            "Q_miss_if_extant": q_miss,
            "pi_detect_if_extant": pi_detect,
            "Q_miss_if_extant_standard_design": q_miss_std,
            "pi_detect_if_extant_standard_design": pi_detect_std,
            "BF_absent_vs_present_if_missing": bf,
            "posterior_present": post_present,
            "posterior_absent": post_absent,
        })

    out = pd.DataFrame(rows)
    out["missing_but_power_80"] = (~out["observed_detected"]) & (out["pi_detect_if_extant"] >= 0.80)
    out["missing_but_power_95"] = (~out["observed_detected"]) & (out["pi_detect_if_extant"] >= 0.95)
    out["catalogue_taxa"] = n_taxa
    return out


def summarize_group_power(
    power: pd.DataFrame,
    *,
    total_bp: pd.Series,
    total_seq_bp: pd.Series,
    ids: pd.Index,
    rng: np.random.Generator,
    n_post_draws: int,
) -> dict:
    q = power["Q_miss_if_extant"].astype(float).values
    c_obs = int(power["observed_detected"].sum())
    n_cat = int(len(power))
    m_obs = n_cat - c_obs
    expected_missed = float(np.sum(q))
    var_missed = float(np.sum(q * (1.0 - q)))
    z = (m_obs - expected_missed) / math.sqrt(var_missed) if var_missed > 0 else np.nan

    p_post = power["posterior_present"].astype(float).values
    pi_std = power["pi_detect_if_extant_standard_design"].astype(float).values
    post_mean, post_lo, post_hi = poisson_binomial_interval(p_post, rng, n_post_draws)

    seq = total_seq_bp.reindex(ids).astype(float)
    mapped = total_bp.reindex(ids).astype(float)
    return {
        "reference": str(power["reference"].iloc[0]),
        "group": str(power["group"].iloc[0]),
        "n_samples": len(ids),
        "catalogue_taxa": n_cat,
        "observed_detected_taxa": c_obs,
        "observed_missing_taxa": m_obs,
        "weak_evidence_taxa": int(power["weak_evidence_in_group"].sum()),
        "expected_missing_under_extant_null": expected_missed,
        "var_missing_under_extant_null": var_missed,
        "z_missing_excess_vs_extant_null": z,
        "missing_with_power_ge_80": int(power["missing_but_power_80"].sum()),
        "missing_with_power_ge_95": int(power["missing_but_power_95"].sum()),
        "posterior_asymptote_mean": post_mean,
        "posterior_asymptote_q025": post_lo,
        "posterior_asymptote_q975": post_hi,
        "standard_design_expected_detected_posterior": float(np.sum(p_post * pi_std)),
        "standard_design_expected_detected_extant_null": float(np.sum(pi_std)),
        "median_total_mapped_bp": float(mapped.median()),
        "median_metadata_total_bases": float(seq.median()) if seq.notna().any() else np.nan,
        "mean_total_mapped_bp": float(mapped.mean()),
        "mean_metadata_total_bases": float(seq.mean()) if seq.notna().any() else np.nan,
    }


def compare_groups(power_a: pd.DataFrame, power_b: pd.DataFrame, rng: np.random.Generator, n_draws: int) -> tuple[pd.DataFrame, dict]:
    a = power_a.set_index("taxon")
    b = power_b.set_index("taxon")
    common = a.index.intersection(b.index)
    a = a.loc[common]
    b = b.loc[common]

    diff = pd.DataFrame({
        "reference": a["reference"],
        "taxon": common,
        "observed_A": a["observed_detected"].values,
        "observed_B": b["observed_detected"].values,
        "pi_A": a["pi_detect_if_extant"].values,
        "pi_B": b["pi_detect_if_extant"].values,
        "pi_A_standard_design": a["pi_detect_if_extant_standard_design"].values,
        "pi_B_standard_design": b["pi_detect_if_extant_standard_design"].values,
        "Q_A": a["Q_miss_if_extant"].values,
        "Q_B": b["Q_miss_if_extant"].values,
        "posterior_present_A": a["posterior_present"].values,
        "posterior_present_B": b["posterior_present"].values,
        "posterior_delta_A_minus_B": a["posterior_present"].values - b["posterior_present"].values,
        "missing_A_high_power": (~a["observed_detected"].values) & (a["pi_detect_if_extant"].values >= 0.95),
        "missing_B_high_power": (~b["observed_detected"].values) & (b["pi_detect_if_extant"].values >= 0.95),
    })

    pa = a["posterior_present"].astype(float).values
    pb = b["posterior_present"].astype(float).values
    draws_a = (rng.random((n_draws, pa.size)) < pa).sum(axis=1)
    draws_b = (rng.random((n_draws, pb.size)) < pb).sum(axis=1)
    delta = draws_a - draws_b
    std_delta = np.sum(
        a["posterior_present"].astype(float).values * a["pi_detect_if_extant_standard_design"].astype(float).values
        - b["posterior_present"].astype(float).values * b["pi_detect_if_extant_standard_design"].astype(float).values
    )
    summary = {
        "reference": str(a["reference"].iloc[0]),
        "group_A": str(a["group"].iloc[0]),
        "group_B": str(b["group"].iloc[0]),
        "posterior_delta_mean_A_minus_B": float(delta.mean()),
        "posterior_delta_q025": float(np.quantile(delta, 0.025)),
        "posterior_delta_q975": float(np.quantile(delta, 0.975)),
        "standard_design_expected_detected_delta_A_minus_B": float(std_delta),
        "taxa_missing_A_power95": int(diff["missing_A_high_power"].sum()),
        "taxa_missing_B_power95": int(diff["missing_B_high_power"].sum()),
    }
    return diff.reset_index(drop=True), summary


def analyze_reference(args: argparse.Namespace, ref_name: str) -> None:
    load_reference = _import_pipeline_loader()
    data = load_reference(ref_name)
    rng = np.random.default_rng(args.seed)

    if args.group_col == "MacroGroup" and "MacroGroup" not in data.meta.columns:
        raise ValueError("MacroGroup was not available after loading")

    ids_a = group_index(data.meta, args.group_col, args.group_a)
    ids_b = group_index(data.meta, args.group_col, args.group_b)
    if len(ids_a) == 0 or len(ids_b) == 0:
        raise ValueError(f"Empty group: {args.group_a} -> {len(ids_a)}, {args.group_b} -> {len(ids_b)}")

    outdir = Path(args.output_dir) / ref_name
    outdir.mkdir(parents=True, exist_ok=True)

    pooled_depth = data.total_bp.reindex(ids_a.union(ids_b)).astype(float).dropna().values
    standard_n = min(len(ids_a), len(ids_b))
    probs = (np.arange(standard_n) + 0.5) / standard_n
    standard_total_bp = np.quantile(pooled_depth, probs)
    print(
        f"[{ref_name}] Standardized design: {standard_n:,} samples using pooled "
        f"mapped-depth quantiles"
    )

    group_specs = [
        (args.label_a, ids_a, data.meta.index.difference(ids_a)),
        (args.label_b, ids_b, data.meta.index.difference(ids_b)),
    ]

    power_tables = []
    summaries = []
    for label, ids, donor_ids in group_specs:
        print(f"\n[{ref_name}] Estimating detection power for {label}: {len(ids):,} samples")
        power = empirical_detection_power_for_group(
            ref_name=ref_name,
            group_name=label,
            target_ids=ids,
            donor_ids=donor_ids,
            breadth=data.breadth,
            depth=data.depth,
            lengths=data.lengths,
            total_bp=data.total_bp,
            standard_total_bp=standard_total_bp,
            threshold_pct=args.threshold,
            weak_breadth_pct=args.weak_breadth_threshold,
            min_within_weak=args.min_within_weak,
            prior_extant=args.prior_extant,
        )
        power_tables.append(power)
        summaries.append(summarize_group_power(
            power,
            total_bp=data.total_bp,
            total_seq_bp=data.total_seq_bp,
            ids=ids,
            rng=rng,
            n_post_draws=args.posterior_draws,
        ))

    taxon_power = pd.concat(power_tables, ignore_index=True)
    diff, diff_summary = compare_groups(power_tables[0], power_tables[1], rng, args.posterior_draws)
    summary = pd.DataFrame(summaries)
    diff_summary_df = pd.DataFrame([diff_summary])

    taxon_power.to_csv(outdir / f"{ref_name}_taxon_detection_power.csv", index=False)
    diff.to_csv(outdir / f"{ref_name}_differential_extinction_taxa.csv", index=False)
    summary.to_csv(outdir / f"{ref_name}_group_detection_power_summary.csv", index=False)
    diff_summary_df.to_csv(outdir / f"{ref_name}_differential_extinction_summary.csv", index=False)

    print(f"\n[{ref_name}] Saved:")
    print(f"  {outdir / f'{ref_name}_taxon_detection_power.csv'}")
    print(f"  {outdir / f'{ref_name}_differential_extinction_taxa.csv'}")
    print(f"  {outdir / f'{ref_name}_group_detection_power_summary.csv'}")
    print(f"  {outdir / f'{ref_name}_differential_extinction_summary.csv'}")
    print(summary[[
        "group",
        "n_samples",
        "observed_detected_taxa",
        "missing_with_power_ge_95",
        "posterior_asymptote_mean",
        "posterior_asymptote_q025",
        "posterior_asymptote_q975",
        "standard_design_expected_detected_posterior",
    ]].to_string(index=False))
    print(diff_summary_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", nargs="+", default=["UHGG", "lev"], choices=["UHGG", "lev", "chimp"])
    parser.add_argument("--output-dir", default="results/detection_power_extinction")
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--weak-breadth-threshold", type=float, default=0.0)
    parser.add_argument("--min-within-weak", type=int, default=5)
    parser.add_argument("--prior-extant", type=float, default=0.95)
    parser.add_argument("--posterior-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--group-col", default="MacroGroup")
    parser.add_argument("--group-a", nargs="+", default=["Industrial"])
    parser.add_argument("--group-b", nargs="+", default=["Non-Industrial"])
    parser.add_argument("--label-a", default="Industrial")
    parser.add_argument("--label-b", default="Non-Industrial")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.threshold < 100.0):
        raise ValueError("--threshold must be between 0 and 100")
    if not (0.0 < args.prior_extant < 1.0):
        raise ValueError("--prior-extant must be between 0 and 1")
    for ref_name in args.refs:
        analyze_reference(args, ref_name)


if __name__ == "__main__":
    main()
