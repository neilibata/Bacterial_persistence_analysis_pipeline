"""
Within-group statistics plugin (Figure 2 stat placeholder filler).

Extends the existing within-group variation machinery in
section1_within_variation.py to write a *frozen* stats CSV and
stats-report entries that directly resolve the ``stat`` placeholders
in the Figure 2 section of the main manuscript.

Analyses
--------
WGS-1  Richness summary per lifestyle (n, mean, median, IQR, CV)
WGS-2  Jaccard *similarity* (= 1 - Jaccard distance) per lifestyle:
           mean, median, q95  — so the manuscript can say
           "samples share at most X% of species on average Y%"
WGS-3  Bray–Curtis distance per lifestyle: mean, median, IQR
WGS-4  Omnibus Kruskal–Wallis on per-sample richness (H-G / Trans / Ind)
WGS-5  Post-hoc pairwise Mann–Whitney U on richness with BH correction
WGS-6  Composition test: ANOSIM-style permutation on the within-group
       mean Jaccard similarity as the test statistic (avoids treating
       n_pairs distances as independent observations)

Outputs
-------
results/<ref>/section1_diversity/main/within_group_stats_<spec>.csv
    Frozen stats table.  One row per (Lifestyle, metric).

results/<ref>/stats_report.csv (via StatsReport)
    Stable tags for every manuscript-fill-in value.

results/<ref>/section1_diversity/main/manuscript_within_stats_<spec>.txt
    Ready-to-paste manuscript sentences.

Integration
-----------
Called from run_within_variation() in section1_within_variation.py
*after* the beta distances have been computed, OR standalone::

    from persistence.within_group_stats import run_within_group_stats
    run_within_group_stats(data, dm, stats, specs=None)
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import kruskal, mannwhitneyu

from . import config
from .detection import DetectionSpec, build_presence
from .section1_within_variation import (
    _lifestyle_split, LIFESTYLE_ORDER,
)
from .plot_utils import (  # module level: the helpers below are used
    page_figsize,          # inside functions that do their own local
    panel_grid,            # plot_utils imports, which would otherwise
    share_axis_labels,     # shadow these names out of scope
)


def _display_lifestyle(label: str) -> str:
    return config.LIFESTYLE_DISPLAY.get(label, label)


# ── BH correction (no statsmodels dependency) ────────────────────────────────

def _bh_correct(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg FDR correction.  Returns adjusted p-values."""
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    ranked = np.empty(n)
    for rank, idx in enumerate(order, 1):
        ranked[idx] = pvals[idx] * n / rank
    # Enforce monotonicity from the right
    for i in range(n - 2, -1, -1):
        ranked[order[i]] = min(ranked[order[i]], ranked[order[i + 1]])
    return [min(float(v), 1.0) for v in ranked]


def _star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ── ANOSIM-style permutation test ────────────────────────────────────────────

def _anosim_richness_permutation(
    richness: pd.Series,
    groups: dict[str, list],
    n_perm: int = 4999,
    rng_seed: int = 42,
) -> dict:
    """
    Permutation F-statistic (Kruskal–Wallis statistic) test on per-sample
    richness.  Uses the observed KW H as the test statistic; label-shuffles
    to build the null.

    Returns dict with keys: H_obs, p_perm, n_perm.
    """
    rng = np.random.default_rng(rng_seed)
    all_ids = [s for ids in groups.values() for s in ids]
    vals = richness.reindex(all_ids).dropna()
    labels = np.array(
        [lf for lf, ids in groups.items() for s in ids if s in vals.index]
    )
    vals = vals.values

    if len(vals) < 6 or len(groups) < 2:
        return {"H_obs": np.nan, "p_perm": np.nan, "n_perm": n_perm}

    # Observed KW H
    grouped = [vals[labels == lf] for lf in groups if (labels == lf).any()]
    grouped = [g for g in grouped if len(g) >= 2]
    if len(grouped) < 2:
        return {"H_obs": np.nan, "p_perm": np.nan, "n_perm": n_perm}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H_obs, _ = kruskal(*grouped)

    null = np.empty(n_perm)
    for i in range(n_perm):
        perm_labels = rng.permutation(labels)
        pg = [vals[perm_labels == lf] for lf in groups if (perm_labels == lf).any()]
        pg = [g for g in pg if len(g) >= 2]
        if len(pg) < 2:
            null[i] = 0.0
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h, _ = kruskal(*pg)
        null[i] = h

    p_perm = max(float(np.mean(null >= H_obs)), 1.0 / (n_perm + 1))
    return {"H_obs": float(H_obs), "p_perm": p_perm, "n_perm": n_perm}


def _composition_permanova_lite(
    jacc_dist_vecs: dict[str, np.ndarray],
    groups: dict[str, list],
    n_perm: int = 999,
    rng_seed: int = 42,
) -> dict:
    """
    Lightweight pseudo-PERMANOVA on within-group Jaccard distances.

    Test statistic: ratio of between-group mean distance to within-group
    mean distance (higher = more separation).  Permutes sample labels.

    Because we may have subsampled samples per group, we work from the
    *per-sample leave-one-out mean Jaccard distance* as the testing unit,
    keeping effective df ≈ n_samples rather than n_pairs.

    Returns dict with keys: F_obs, p_perm, n_perm.
    """
    rng = np.random.default_rng(rng_seed)

    # Build per-sample mean within-group Jaccard similarity
    # We use the mean of the jacc_dist row for each sample as its "score"
    lf_list = [lf for lf in LIFESTYLE_ORDER if lf in groups]
    if len(lf_list) < 2:
        return {"F_obs": np.nan, "p_perm": np.nan, "n_perm": n_perm}

    # For each lifestyle we only have the condensed-form distances; convert
    # to per-sample mean distance scores
    per_sample_scores = {}   # sample_id -> mean_within_jacc_dist
    per_sample_lf = {}

    for lf in lf_list:
        vec = jacc_dist_vecs.get(lf)
        if vec is None or len(vec) == 0:
            continue
        n = groups[lf] if isinstance(groups[lf], int) else len(groups[lf])
        if isinstance(groups[lf], list):
            ids = groups[lf]
        else:
            continue
        # Reconstruct square matrix from condensed form
        from scipy.spatial.distance import squareform
        sq = squareform(vec)
        for i, sid in enumerate(ids[: sq.shape[0]]):
            row = np.concatenate([sq[i, :i], sq[i, i + 1 :]])
            per_sample_scores[sid] = float(np.mean(row)) if len(row) else 0.0
            per_sample_lf[sid] = lf

    if len(per_sample_scores) < 6:
        return {"F_obs": np.nan, "p_perm": np.nan, "n_perm": n_perm}

    all_ids = list(per_sample_scores.keys())
    scores = np.array([per_sample_scores[s] for s in all_ids])
    labels = np.array([per_sample_lf[s] for s in all_ids])

    def _F(sc, lb):
        grand = sc.mean()
        ss_tot = np.sum((sc - grand) ** 2)
        ss_within = sum(
            np.sum((sc[lb == lf] - sc[lb == lf].mean()) ** 2)
            for lf in np.unique(lb)
            if (lb == lf).sum() > 1
        )
        ss_between = ss_tot - ss_within
        df_b = len(np.unique(lb)) - 1
        df_w = len(sc) - len(np.unique(lb))
        if df_w <= 0 or ss_within == 0:
            return np.nan
        return (ss_between / df_b) / (ss_within / df_w)

    F_obs = _F(scores, labels)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _F(scores, rng.permutation(labels))
        if np.isnan(null[i]):
            null[i] = 0.0

    if np.isnan(F_obs):
        return {"F_obs": np.nan, "p_perm": np.nan, "n_perm": n_perm}

    p_perm = max(float(np.mean(null >= F_obs)), 1.0 / (n_perm + 1))
    return {"F_obs": float(F_obs), "p_perm": p_perm, "n_perm": n_perm}


# ── Main computation ──────────────────────────────────────────────────────────

def run_within_group_stats(data, dm, stats, specs=None, max_per_group: int = 1500):
    """
    Compute all within-group statistics needed for Figure 2 manuscript fill-ins.

    Parameters
    ----------
    data          : ReferenceData
    dm            : DepthMatcher
    stats         : StatsReport
    specs         : list[DetectionSpec] | None
    max_per_group : int
        Subsample cap for distance computation (mirrors section1_within_variation).
    """
    if dm is None:
        print("  [within_group_stats] No depth-matching — skipping")
        return

    if specs is None:
        from .detection import DetectionSpec
        specs = [
            DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP),
            DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH),
        ]
    elif not isinstance(specs, (list, tuple)):
        specs = [specs]

    ref = data.ref_name
    outdir = config.section_dir(ref, "s1", "main")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Within-group stats plugin [{ref}]")
    print(f"{'='*60}")

    for spec in specs:
        _run_one_spec(data, dm, stats, spec, outdir, max_per_group)


def _run_one_spec(data, dm, stats, spec, outdir, max_per_group):
    spec_tag = spec.short_label
    ref = data.ref_name
    rng = np.random.default_rng(42)

    print(f"\n  [{spec_tag}] {spec.description}")

    ids_ind = dm.ind_vs_ni_ind
    ids_ni = dm.ind_vs_ni_ni
    if not ids_ind or not ids_ni:
        print("    [within_group_stats] No depth-matched pairs — skipping")
        return

    all_ids = list(ids_ind) + list(ids_ni)

    # ── Build matrices ──────────────────────────────────────────────────────
    presence = build_presence(
        data.breadth.loc[all_ids],
        data.depth.loc[all_ids],
        data.lengths,
        spec=spec,
    )
    richness = presence.sum(axis=1)

    abund = data.abundance.loc[all_ids]

    groups = _lifestyle_split(data, ids_ni)
    groups["Industrial"] = list(ids_ind)

    # ── WGS-1: Richness summary ─────────────────────────────────────────────
    print("\n    WGS-1  Richness summaries")
    richness_rows = []
    richness_by_lf = {}

    for lf in LIFESTYLE_ORDER:
        if lf not in groups:
            continue
        gids = groups[lf]
        vals = richness.reindex(gids).dropna().values
        if len(vals) < 3:
            continue
        richness_by_lf[lf] = vals

        n = len(vals)
        mean_r = float(np.mean(vals))
        med_r = float(np.median(vals))
        sd_r = float(np.std(vals, ddof=1))
        cv_r = sd_r / mean_r * 100 if mean_r > 0 else np.nan
        q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        iqr_r = q75 - q25

        richness_rows.append({
            "Lifestyle": lf,
            "Metric": "richness",
            "n_samples": n,
            "n_pairs": None,
            "mean": round(mean_r, 2),
            "median": round(med_r, 2),
            "sd": round(sd_r, 2),
            "cv_pct": round(cv_r, 2),
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "q95": round(float(np.percentile(vals, 95)), 2),
            "iqr": round(iqr_r, 2),
        })

        stats.add(
            f"within_richness_summary_{lf}_{spec_tag}",
            "s1", "main",
            f"Richness summary for {lf} (depth-matched, {spec.description})",
            f"n={n}, median={med_r:.0f}, mean={mean_r:.0f}, "
            f"CV={cv_r:.1f}%, IQR=[{q25:.0f}–{q75:.0f}]",
            n1=n,
        )
        print(f"      {lf}: n={n}, median={med_r:.0f}, CV={cv_r:.1f}%")

    # ── WGS-4: Omnibus KW richness ──────────────────────────────────────────
    print("\n    WGS-4  Omnibus Kruskal–Wallis on richness")
    kw_groups = [v for v in richness_by_lf.values() if len(v) >= 3]
    if len(kw_groups) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            H_kw, p_kw = kruskal(*kw_groups)
        stats.add(
            f"richness_kruskal_wallis_{spec_tag}",
            "s1", "main",
            f"Omnibus KW test on per-sample richness across lifestyles "
            f"({spec.description})",
            f"H={H_kw:.2f}, p={p_kw:.3e}, k={len(kw_groups)} groups",
            test="Kruskal–Wallis",
            p_value=p_kw,
            n1=sum(len(v) for v in kw_groups),
        )
        print(f"      KW H={H_kw:.2f}, p={p_kw:.3e}")

        # ── WGS-5: Pairwise MW-U with BH correction ─────────────────────────
        print("\n    WGS-5  Pairwise Mann–Whitney U (BH corrected)")
        pairs = []
        raw_ps = []
        for i, lf_a in enumerate(LIFESTYLE_ORDER):
            for lf_b in LIFESTYLE_ORDER[i + 1:]:
                if lf_a not in richness_by_lf or lf_b not in richness_by_lf:
                    continue
                u, p_raw = mannwhitneyu(
                    richness_by_lf[lf_a],
                    richness_by_lf[lf_b],
                    alternative="two-sided",
                )
                # Rank-biserial r
                n1, n2 = len(richness_by_lf[lf_a]), len(richness_by_lf[lf_b])
                r_rb = float(1.0 - 2.0 * u / max(n1 * n2, 1))
                pairs.append((lf_a, lf_b, u, p_raw, r_rb, n1, n2))
                raw_ps.append(p_raw)

        adj_ps = _bh_correct(raw_ps)
        for (lf_a, lf_b, u, p_raw, r_rb, n1, n2), p_adj in zip(pairs, adj_ps):
            tag = f"richness_mwu_{lf_a}_vs_{lf_b}_{spec_tag}"
            stats.add(
                tag, "s1", "main",
                f"Richness MW-U {lf_a} vs {lf_b} "
                f"(depth-matched, {spec.description})",
                f"U={u:.0f}, p_raw={p_raw:.3e}, p_BH={p_adj:.3e}, "
                f"rank-biserial r={r_rb:+.3f}",
                test="Mann–Whitney U (BH corrected)",
                p_value=p_adj,
                effect_size=r_rb,
                n1=n1, n2=n2,
            )
            print(f"      {lf_a} vs {lf_b}: U={u:.0f}, "
                  f"p_BH={p_adj:.3e} {_star(p_adj)}, r={r_rb:+.3f}")

    # ── WGS-2 & WGS-3: Jaccard similarity + Bray–Curtis ───────────────────
    print("\n    WGS-2/3  Jaccard similarity & Bray–Curtis per lifestyle")
    distance_rows = []
    jacc_dist_vecs: dict[str, np.ndarray] = {}

    for lf in LIFESTYLE_ORDER:
        if lf not in groups or len(groups[lf]) < 10:
            continue
        gids = list(groups[lf])

        # Subsample
        if len(gids) > max_per_group:
            gids = list(rng.choice(gids, size=max_per_group, replace=False))

        # Jaccard distance → similarity
        pres_g = presence.loc[gids].values.astype(float)
        jacc_d = pdist(pres_g, metric="jaccard")
        jacc_d = np.nan_to_num(jacc_d, nan=1.0)
        jacc_sim = 1.0 - jacc_d            # convert to similarity

        # Bray–Curtis
        abund_g = np.nan_to_num(abund.loc[gids].values, nan=0.0)
        bc = pdist(abund_g, metric="braycurtis")
        bc = np.nan_to_num(bc, nan=1.0)

        n = len(gids)
        n_pairs = len(jacc_d)

        jacc_dist_vecs[lf] = jacc_d   # kept for permutation test

        for metric_name, vals, direction in [
            ("jaccard_similarity", jacc_sim, "higher=more similar"),
            ("braycurtis_distance", bc, "higher=more dissimilar"),
        ]:
            mean_v = float(np.mean(vals))
            med_v = float(np.median(vals))
            q25_v = float(np.percentile(vals, 25))
            q75_v = float(np.percentile(vals, 75))
            q95_v = float(np.percentile(vals, 95))

            distance_rows.append({
                "Lifestyle": lf,
                "Metric": metric_name,
                "n_samples": n,
                "n_pairs": n_pairs,
                "mean": round(mean_v, 4),
                "median": round(med_v, 4),
                "sd": round(float(np.std(vals)), 4),
                "cv_pct": None,
                "q25": round(q25_v, 4),
                "q75": round(q75_v, 4),
                "q95": round(q95_v, 4),
                "iqr": round(q75_v - q25_v, 4),
            })

            tag = f"within_{metric_name}_{lf}_{spec_tag}"
            # Manuscript-friendly phrasing differs by metric
            if metric_name == "jaccard_similarity":
                msfriendly = (
                    f"mean within-group Jaccard similarity = {mean_v:.2f} "
                    f"(median = {med_v:.2f}, 95th pct = {q95_v:.2f}); "
                    f"samples share on average {mean_v*100:.0f}% of species "
                    f"and at most {q95_v*100:.0f}%"
                )
            else:
                msfriendly = (
                    f"mean within-group Bray–Curtis = {mean_v:.3f} "
                    f"(median = {med_v:.3f}, IQR = [{q25_v:.3f}–{q75_v:.3f}])"
                )

            stats.add(
                tag, "s1", "main",
                f"{metric_name} within {lf} "
                f"(depth-matched, {spec.description}; {direction})",
                msfriendly,
                n1=n,
                extra={"n_pairs": n_pairs},
            )

        print(
            f"      {lf}: n={n}, "
            f"Jacc sim mean={np.mean(jacc_sim):.3f} "
            f"median={np.median(jacc_sim):.3f} "
            f"q95={np.percentile(jacc_sim, 95):.3f}  |  "
            f"BC mean={np.mean(bc):.3f}"
        )

    # ── WGS-6: Composition permutation test ─────────────────────────────────
    print("\n    WGS-6  Composition permutation test (pseudo-F on per-sample means)")
    # Only compute if we have distances for all three lifestyles
    if len(jacc_dist_vecs) >= 2:
        perm_res = _composition_permanova_lite(
            jacc_dist_vecs, groups, n_perm=999, rng_seed=42
        )
        stats.add(
            f"composition_permanova_{spec_tag}",
            "s1", "main",
            f"Composition permutation pseudo-F on within-group mean Jaccard "
            f"distance (depth-matched, {spec.description}). "
            f"Testing unit = per-sample leave-one-out mean distance; "
            f"n_perm={perm_res['n_perm']}",
            f"F={perm_res['F_obs']:.3f}, p_perm={perm_res['p_perm']:.4f}",
            test=f"Permutation pseudo-F (n_perm={perm_res['n_perm']})",
            p_value=perm_res["p_perm"],
            effect_size=perm_res["F_obs"],
        )
        print(f"      F_obs={perm_res['F_obs']:.3f}, "
              f"p_perm={perm_res['p_perm']:.4f}")

    # ── Write frozen stats CSV ───────────────────────────────────────────────
    all_rows = richness_rows + distance_rows
    if all_rows:
        csv_path = os.path.join(
            outdir, f"within_group_stats_{spec_tag}.csv"
        )
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)
        print(f"\n    Saved: {csv_path}")

    # ── Write manuscript fill-in sentences ───────────────────────────────────
    # Look for 3-group GMW summary CSV written by _run_health_richness_gmw.
    # It lives in section3_health/ alongside the figure; may not exist yet
    # if S3 has not been run, in which case we fall back to raw medians.
    import glob as _glob
    _gmw3_csv_path = os.path.join(
        "results", ref, "section3_health", "main",
        f"health_richness_gmw_{ref}_summary_lifestyle_{spec_tag}.csv")
    gmw3_df = None
    if os.path.exists(_gmw3_csv_path):
        gmw3_df = pd.read_csv(_gmw3_csv_path)
        print(f"    [within_group_stats] Loaded GMW3 summary: {_gmw3_csv_path}")
    else:
        # CSV not found — compute 3-way GMW richness inline using depth_bias_ipw
        # primitives directly, with no cross-module import of section3_health.
        print(f"    [within_group_stats] GMW3 CSV not found — "
              f"computing 3-way GMW richness inline...")
        gmw3_df = _compute_gmw3_lifestyle(
            data, None, spec, spec_tag, ref, stats)

    _write_manuscript_sentences(
        richness_by_lf, jacc_dist_vecs,
        groups, outdir, spec_tag, ref,
        gmw3_df=gmw3_df,
    )


def _compute_gmw3_lifestyle(data, dm, spec, spec_tag, ref, stats,
                             n_boot: int = 500, rng_seed: int = 42):
    """
    Self-contained 3-way GMW depth-corrected richness for
    H-G / Transitional / Ind (all).

    Uses depth_bias_ipw primitives directly — no import of section3_health.
    Writes results/<ref>/section3_health/main/health_richness_gmw_<ref>_summary_lifestyle_<spec>.csv
    and returns a DataFrame with columns matching what _write_manuscript_sentences expects.
    """
    from .depth_bias_ipw import _BinnedKDE, _ess, _weighted_sem
    from .detection import build_presence

    rng = np.random.default_rng(rng_seed)
    meta = data.meta

    # ── Group membership ───────────────────────────────────────────────────────
    def _ids(lifestyle_list):
        return [s for s in meta.index
                if s in data.breadth.index
                and meta.loc[s, "Lifestyle"] in lifestyle_list]

    grp_ids = {
        "H-G":          _ids(["H-G"]),
        "Transitional": _ids(["Transitional"]),
        "Ind (all)":    _ids(["Industrial"]),
    }
    grp_names = ["H-G", "Transitional", "Ind (all)"]

    for g, ids in grp_ids.items():
        if len(ids) < 10:
            print(f"    [GMW3] {g}: only {len(ids)} samples — skipping")
            return None

    # ── Build richness ─────────────────────────────────────────────────────────
    all_ids = [s for ids in grp_ids.values() for s in ids]
    # Only build for samples present in breadth/depth
    all_ids = [s for s in all_ids if s in data.breadth.index and s in data.depth.index]
    presence = build_presence(
        data.breadth.loc[all_ids],
        data.depth.loc[all_ids],
        data.lengths,
        spec=spec,
    )
    rich_ser = presence.sum(axis=1)
    rich = {g: rich_ser.reindex(grp_ids[g]).dropna().values.astype(float)
            for g in grp_names}

    # ── Log depth per sample ───────────────────────────────────────────────────
    def _ld(ids):
        return np.log10(np.maximum(
            data.total_bp.reindex(ids).fillna(1.0).values, 1.0))

    ld = {g: _ld(grp_ids[g]) for g in grp_names}

    # ── KDEs ──────────────────────────────────────────────────────────────────
    kde = {g: _BinnedKDE(ld[g]) for g in grp_names}

    # ── 3-way GMW weights ────────────────────────────────────────────────────
    # w_x = (f_b * f_c / f_x^2)^(1/3), mean-normalised, clipped at 10
    def _gmw3_weights(x_vals, own_kde, other_kde_b, other_kde_c,
                      clip: float = 10.0):
        f_x = np.maximum(own_kde(x_vals), 1e-10)
        f_b = np.maximum(other_kde_b(x_vals), 1e-10)
        f_c = np.maximum(other_kde_c(x_vals), 1e-10)
        w = (f_b * f_c / f_x ** 2) ** (1.0 / 3.0)
        w = w / w.mean()
        return np.clip(w, 0.0, clip)

    ga, gb, gc = grp_names
    w = {
        ga: _gmw3_weights(ld[ga], kde[ga], kde[gb], kde[gc]),
        gb: _gmw3_weights(ld[gb], kde[gb], kde[ga], kde[gc]),
        gc: _gmw3_weights(ld[gc], kde[gc], kde[ga], kde[gb]),
    }

    # ── Weighted means ────────────────────────────────────────────────────────
    def _wmean(r_arr, w_arr):
        w_sum = w_arr.sum()
        return float(np.dot(w_arr, r_arr) / max(w_sum, 1e-300))

    corr_mu = {g: _wmean(rich[g], w[g]) for g in grp_names}

    # ── Bootstrap 95% CI ─────────────────────────────────────────────────────
    print(f"\n    [GMW3] Bootstrapping (n_boot={n_boot})...")
    boot = {}
    for g in grp_names:
        r_g = rich[g]
        w_g = w[g]
        n_g = len(r_g)
        boot_vals = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.choice(n_g, size=n_g, replace=True)
            boot_vals[i] = _wmean(r_g[idx], w_g[idx])
        boot[g] = boot_vals

    corr_ci = {g: [float(np.percentile(boot[g], 2.5)),
                   float(np.percentile(boot[g], 97.5))]
               for g in grp_names}
    corr_wsem = {g: _weighted_sem(rich[g], w[g], corr_mu[g])
                 for g in grp_names}

    # Print summary
    print(f"    [GMW3 summary_lifestyle] "
          f"{ga}: corr={corr_mu[ga]:.1f}  "
          f"{gb}: corr={corr_mu[gb]:.1f}  "
          f"{gc}: corr={corr_mu[gc]:.1f}")

    # ── Build output DataFrame ────────────────────────────────────────────────
    rows = []
    for g in grp_names:
        r_g = rich[g]
        rows.append(dict(
            spec=spec.description,
            set_label="summary_lifestyle",
            group=g,
            n=len(r_g),
            raw_mean=round(float(r_g.mean()), 2),
            raw_sem=round(float(r_g.std(ddof=1) / np.sqrt(len(r_g))), 3),
            corr_mean=round(corr_mu[g], 2),
            corr_ci_lo=round(corr_ci[g][0], 2),
            corr_ci_hi=round(corr_ci[g][1], 2),
            corr_wsem=round(corr_wsem[g], 3),
            ess=round(_ess(w[g]), 1),
        ))
    gmw3_df = pd.DataFrame(rows)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    out_dir = os.path.join("results", ref, "section3_health", "main")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(
        out_dir,
        f"health_richness_gmw_{ref}_summary_lifestyle_{spec_tag}.csv")
    gmw3_df.to_csv(csv_path, index=False)
    print(f"    [GMW3] Saved: {csv_path}")

    # ── Log to stats report ───────────────────────────────────────────────────
    for g in grp_names:
        tag = (f"gmw3_summary_lifestyle_{g}_{spec_tag}"
               .replace(" ", "_").replace("(", "").replace(")", ""))
        mu  = corr_mu[g]
        ci_ = corr_ci[g]
        stats.add(
            tag, "s1", "main",
            f"3-way GMW corrected richness: {g} "
            f"[summary_lifestyle, {spec.description}]",
            f"corr_mean={mu:.1f} [95%CI:{ci_[0]:.1f},{ci_[1]:.1f}] "
            f"wsem={corr_wsem[g]:.1f} ESS={_ess(w[g]):.0f} "
            f"n={len(rich[g]):,}",
            n1=len(rich[g]),
        )

    return gmw3_df


def _write_manuscript_sentences(
    richness_by_lf: dict,
    jacc_dist_vecs: dict,
    groups: dict,
    outdir: str,
    spec_tag: str,
    ref: str,
    gmw3_df=None,
) -> None:
    """
    Write ready-to-paste sentences for the Figure 2 placeholder slots.

    Produces three blocks:
      (A) Raw depth-matched medians/IQRs  — for the spread/variation claim
      (B) 3-way GMW corrected means ± CI  — for the depth-adjusted central
          tendency claim (read from health_richness_gmw_*_summary_lifestyle_*.csv
          if S3 has already run; otherwise flagged as pending)
      (C) Jaccard similarity within groups — for the composition claim

    Usage:
      Run --only s1_within first to get (A) and (C).
      Run --only s3 (or full pipeline) to get (B), then re-run s1_within
      so the fill-in file picks up the GMW3 CSV automatically.
    """
    import re as _re

    lines = [
        f"Manuscript within-group stat fill-ins — {ref} [{spec_tag}]",
        "=" * 70,
        "",
        "Paste these into the Figure 2 section where [stat] tokens appear.",
        "(A) = depth-matched raw medians for spread; (B) = GMW-corrected means",
        "for depth-adjusted central tendency; (C) = compositional sharing.",
        "",
    ]

    # ── (A) Raw depth-matched richness medians/IQRs ───────────────────────────
    lines.append("(A) RICHNESS — depth-matched raw medians and IQRs:")
    parts_a = []
    raw_median = {}
    for lf in LIFESTYLE_ORDER:
        if lf not in richness_by_lf:
            continue
        v = richness_by_lf[lf]
        med = np.median(v)
        q25, q75 = np.percentile(v, [25, 75])
        raw_median[lf] = med
        parts_a.append(
            f"{_display_lifestyle(lf)}: median {med:.0f} SGBs "
            f"(IQR {q25:.0f}\u2013{q75:.0f})"
        )
    lines.append("  " + ";  ".join(parts_a) if parts_a else "  [no data]")
    lines.append("")

    # ── (B) 3-way GMW corrected means ────────────────────────────────────────
    lines.append("(B) RICHNESS — 3-way GMW depth-corrected means + 95% CI:")
    gmw_corr = {}   # lf -> corr_mean float (for the paragraph template)
    gmw_display = {}  # lf -> full display string
    if gmw3_df is not None and not gmw3_df.empty:
        group_map = {"H-G": "H-G", "Transitional": "Transitional",
                     "Ind (all)": "Industrial"}
        for _, row in gmw3_df.iterrows():
            for csv_grp, lf_key in group_map.items():
                if row["group"] == csv_grp:
                    gmw_corr[lf_key] = float(row["corr_mean"])
                    gmw_display[lf_key] = (
                        f"corr. mean {row['corr_mean']:.0f} SGBs "
                        f"[95%CI {row['corr_ci_lo']:.0f}\u2013{row['corr_ci_hi']:.0f}] "
                        f"(\u00b1{row['corr_wsem']:.0f} wSEM, "
                        f"n={int(row['n']):,}, ESS={row['ess']:.0f})"
                    )
        gmw_parts = [f"{_display_lifestyle(lf)}: {gmw_display[lf]}"
                     for lf in LIFESTYLE_ORDER if lf in gmw_display]
        lines.append("  " + ";  ".join(gmw_parts) if gmw_parts else
                     "  [no matching groups in GMW3 CSV]")
    else:
        lines.append(
            "  [NOT YET AVAILABLE — run S3 first, then re-run --only s1_within]")
        lines.append(
            "  Expected file: results/" + ref +
            "/section3_health/health_richness_gmw_" + ref +
            "_summary_lifestyle_" + spec_tag + ".csv")
        # Provide raw means as a fallback marker
        for lf in LIFESTYLE_ORDER:
            if lf in richness_by_lf:
                lines.append(
                    f"  Fallback raw mean — {_display_lifestyle(lf)}: "
                    f"{np.mean(richness_by_lf[lf]):.0f} SGBs (depth-matched, uncorrected)")
    lines.append("")

    # ── (C) Compositional sharing ─────────────────────────────────────────────
    lines.append("(C) COMPOSITIONAL SHARING (Jaccard similarity; 1 = identical):")
    jacc_mean = {}
    jacc_q95 = {}
    for lf in LIFESTYLE_ORDER:
        if lf not in jacc_dist_vecs:
            continue
        sim = 1.0 - jacc_dist_vecs[lf]
        jacc_mean[lf] = float(np.mean(sim))
        jacc_q95[lf]  = float(np.percentile(sim, 95))
        lines.append(
            f"  {_display_lifestyle(lf)}: mean similarity = {jacc_mean[lf]:.3f} "
            f"({jacc_mean[lf]*100:.0f}% shared on average); "
            f"95th pct = {jacc_q95[lf]:.3f} "
            f"(at most {jacc_q95[lf]*100:.0f}% shared)"
        )
    lines.append("")

    # ── Suggested paragraph ───────────────────────────────────────────────────
    lines.append("SUGGESTED FIGURE 2 FILL-IN PARAGRAPH:")
    lines.append("-" * 68)

    # Richness line — prefer GMW-corrected if available, note raw IQR
    rich_sentences = []
    for lf in LIFESTYLE_ORDER:
        if lf not in richness_by_lf:
            continue
        v = richness_by_lf[lf]
        med = np.median(v)
        q25, q75 = np.percentile(v, [25, 75])
        if lf in gmw_corr:
            rich_sentences.append(
                f"{_display_lifestyle(lf)}: depth-corrected mean {gmw_corr[lf]:.0f} SGBs "
                f"(raw median {med:.0f}, IQR {q25:.0f}\u2013{q75:.0f})"
            )
        else:
            rich_sentences.append(
                f"{_display_lifestyle(lf)}: median {med:.0f} SGBs "
                f"(IQR {q25:.0f}\u2013{q75:.0f})")
    lines.append("  Richness — " + ";  ".join(rich_sentences) + ".")
    lines.append("")

    # Composition line
    if jacc_mean:
        min_mean = min(jacc_mean.values())
        max_mean = max(jacc_mean.values())
        max_q95  = max(jacc_q95.values())
        lines.append(
            f"  Individuals within each lifestyle share on average "
            f"{min_mean*100:.0f}\u2013{max_mean*100:.0f}% of their gut species "
            f"(within-group Jaccard similarity); even the most compositionally "
            f"similar pairs within a lifestyle share at most "
            f"{max_q95*100:.0f}% of species (95th percentile)."
        )
    lines.append("-" * 68)
    lines.append("")
    lines.append("NOTES:")
    lines.append("  (A)+(C) use depth-matched samples from the within_group_stats plugin.")
    lines.append("  (B) uses the 3-way GMW weights from _run_health_richness_gmw (S3).")
    lines.append("  For Figure 2 richness claims, cite (B) for the central tendency")
    lines.append("  and (A) IQR for the spread. Do not mix raw and corrected in the")
    lines.append("  same sentence without labelling them.")

    txt_path = os.path.join(outdir, f"manuscript_within_stats_{spec_tag}.txt")
    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"    Saved: {txt_path}")


# ════════════════════════════════════════════════════════════════════
#  DEPTH-BINNED RICHNESS CV  (residual variability after depth control)
# ════════════════════════════════════════════════════════════════════

def _pin_legend(leg):
    """Keep a legend at the size it was asked for.

    The shared type scale otherwise normalises every legend on the figure to
    one role size, which is what put these two keys back to 7.5 pt however
    large they were set. `_fixed_pt` is the opt-out it honours.
    """
    if leg is None:
        return leg
    for t in leg.get_texts():
        t._fixed_pt = True
    if leg.get_title() is not None:
        leg.get_title()._fixed_pt = True
    return leg


def run_depth_binned_richness_cv(data, stats, specs=None,
                                  n_bins: int = 40,
                                  min_samples_per_bin: int = 10,
                                  n_perm: int = 4999,
                                  rng_seed: int = 42):
    """
    Depth-binned richness coefficient of variation analysis.

    Answers the question: after controlling for sequencing depth, how much
    richness variability remains within each lifestyle, and does that
    variability differ between lifestyles?

    Method
    ------
    1. Bin all samples into ``n_bins`` equally log-spaced depth bins covering
       the full observed range of log10(total_mapped_bp).
    2. Within each occupied bin, compute per lifestyle:
         - n samples
         - median richness
         - SD richness
         - CV  = SD / median   (fractional ratio; the core statistic)
         - IQR / median        (non-parametric alternative)
    3. Use bins with >= ``min_samples_per_bin`` samples for inference.
    4. Test whether per-bin CV differs across lifestyles via:
         - Kruskal–Wallis on all per-bin CV values (one value per
           lifestyle × bin cell where n >= threshold)
         - Pairwise Mann–Whitney U with BH correction
    5. Also report: grand mean CV per lifestyle (depth-controlled proxy
       for depth-independent richness variability).

    Outputs
    -------
    results/<ref>/section1_diversity/main/depth_binned_richness_cv_<spec>.csv
        One row per (Lifestyle, bin_idx) with all statistics.

    results/<ref>/section1_diversity/main/depth_binned_richness_cv_<spec>.png
        Two-panel figure:
          Left  — Median richness per bin by lifestyle (ribbon = IQR)
          Right — CV (SD/median) per bin by lifestyle

    results/<ref>/stats_report.csv entries with tags:
        depthbin_cv_mean_<Lifestyle>_<spec>
        depthbin_cv_kw_<spec>
        depthbin_cv_mwu_<LifeA>_vs_<LifeB>_<spec>

    Can run standalone::

        python -m persistence --ref UHGG --only s1_depthbin
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plot_utils import (
        apply_oi_style,
        current_fs,
        page_figsize,
        panel_grid,
        savefig_dual,
        share_axis_labels,
    )
    apply_oi_style()

    ref = data.ref_name
    outdir = config.section_dir(ref, "s1", "main")
    os.makedirs(outdir, exist_ok=True)
    meta = data.meta
    rng = np.random.default_rng(rng_seed)

    if specs is None:
        specs = [
            DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP),
            DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH),
        ]
    elif not isinstance(specs, (list, tuple)):
        specs = [specs]

    print(f"\n{'='*60}")
    print(f"  Depth-binned richness CV [{ref}]")
    print(f"{'='*60}")

    for spec in specs:
        _run_depthbin_one_spec(
            data, meta, stats, spec, outdir, ref,
            n_bins=n_bins,
            min_n=min_samples_per_bin,
            n_perm=n_perm,
            rng=rng,
        )


def _run_depthbin_one_spec(data, meta, stats, spec, outdir, ref,
                            n_bins, min_n, n_perm, rng):
    """Inner loop for one detection spec."""
    import matplotlib.pyplot as plt
    from .plot_utils import savefig_dual, current_fs

    spec_tag = spec.short_label
    FS = current_fs()

    print(f"\n  [{spec_tag}] {spec.description}")

    # ── All valid samples (not depth-matched — use full post-reinstatement cohort) ──
    valid_ids = [s for s in meta.index
                 if s in data.breadth.index and s in data.depth.index
                 and meta.loc[s, "Lifestyle"] in LIFESTYLE_ORDER]
    if not valid_ids:
        print("    [depthbin] No valid samples — skipping")
        return

    presence = build_presence(
        data.breadth.loc[valid_ids],
        data.depth.loc[valid_ids],
        data.lengths,
        spec=spec,
    )
    richness = presence.sum(axis=1)
    total_bp = data.total_bp.reindex(valid_ids)
    log_depth = np.log10(total_bp.clip(lower=1))
    lifestyle  = meta.reindex(valid_ids)["Lifestyle"]

    # ── Build log-spaced depth bins ───────────────────────────────────────────
    lo = float(log_depth.min())
    hi = float(log_depth.max())
    edges = np.linspace(lo, hi, n_bins + 1)   # equally spaced in log10 space
    bin_idx = np.searchsorted(edges[1:-1], log_depth.values, side="right")
    # bin_idx is 0-based, range [0, n_bins-1]

    print(f"    Depth range: 10^{lo:.2f} – 10^{hi:.2f} bp  ({n_bins} bins)")

    # ── Per-bin, per-lifestyle statistics ────────────────────────────────────
    rows = []
    # Collect per-bin CV values per lifestyle for the KW test
    cv_by_lf: dict[str, list[float]] = {lf: [] for lf in LIFESTYLE_ORDER}

    for b in range(n_bins):
        bin_mask = bin_idx == b
        bin_lo_bp = 10 ** edges[b]
        bin_hi_bp = 10 ** edges[b + 1]

        for lf in LIFESTYLE_ORDER:
            lf_mask = lifestyle.values == lf
            sel = bin_mask & lf_mask
            n = int(sel.sum())

            rows.append({
                "Lifestyle": lf,
                "bin_idx": b,
                "bin_lo_log10bp": round(edges[b], 4),
                "bin_hi_log10bp": round(edges[b + 1], 4),
                "bin_mid_log10bp": round((edges[b] + edges[b + 1]) / 2, 4),
                "bin_lo_Gbp": round(bin_lo_bp / 1e9, 4),
                "bin_hi_Gbp": round(bin_hi_bp / 1e9, 4),
                "n": n,
                "median_richness": np.median(richness.values[sel]) if n >= 2 else np.nan,
                "mean_richness":   np.mean(richness.values[sel])   if n >= 2 else np.nan,
                "sd_richness":     np.std(richness.values[sel], ddof=1) if n >= 2 else np.nan,
                "q25_richness":    np.percentile(richness.values[sel], 25) if n >= 2 else np.nan,
                "q75_richness":    np.percentile(richness.values[sel], 75) if n >= 2 else np.nan,
                "cv_sd_median":    (np.std(richness.values[sel], ddof=1) /
                                    np.median(richness.values[sel]))
                                   if n >= min_n and np.median(richness.values[sel]) > 0
                                   else np.nan,
                "iqr_median_ratio": ((np.percentile(richness.values[sel], 75) -
                                      np.percentile(richness.values[sel], 25)) /
                                     np.median(richness.values[sel]))
                                    if n >= min_n and np.median(richness.values[sel]) > 0
                                    else np.nan,
            })

            if n >= min_n and np.isfinite(rows[-1]["cv_sd_median"]):
                cv_by_lf[lf].append(rows[-1]["cv_sd_median"])

    bin_df = pd.DataFrame(rows)

    # ── Grand mean CV per lifestyle ───────────────────────────────────────────
    print(f"\n    Grand mean CV (SD/median) per lifestyle "
          f"[bins with n >= {min_n}]:")
    cv_vals_for_kw = []
    cv_labels_for_kw = []
    grand_cv: dict[str, float] = {}
    # Every test this figure's caption quotes, gathered for its data object.
    # The `bins` frame carries what the panels are drawn from, but the
    # Kruskal-Wallis, the pairwise comparisons and the permutation test are
    # results, not inputs, and the submitted source data has to include them.
    stat_rows: list[dict] = []

    for lf in LIFESTYLE_ORDER:
        cvs = cv_by_lf[lf]
        if not cvs:
            print(f"      {lf}: no qualifying bins")
            continue
        grand = float(np.mean(cvs))
        grand_cv[lf] = grand
        print(f"      {lf}: mean CV = {grand:.3f}  "
              f"(median {np.median(cvs):.3f}, "
              f"n_bins = {len(cvs)})")
        stats.add(
            f"depthbin_cv_mean_{lf}_{spec_tag}",
            "s1", "main",
            f"Mean depth-bin CV (SD/median richness) for {lf} "
            f"[{spec.description}; bins with n>={min_n}]",
            f"mean_CV={grand:.4f}, median_CV={np.median(cvs):.4f}, "
            f"n_qualifying_bins={len(cvs)}",
            n1=len(cvs),
        )
        stat_rows.append({
            "quantity": "mean depth-bin CV", "group": lf, "comparison": "",
            "test": "", "statistic": np.nan, "p_value": np.nan,
            "effect_size": grand, "n_bins": len(cvs),
            "detail": f"median CV={float(np.median(cvs)):.6f}",
        })
        cv_vals_for_kw.extend(cvs)
        cv_labels_for_kw.extend([lf] * len(cvs))

    # ── Kruskal–Wallis across lifestyles ─────────────────────────────────────
    kw_groups = [cv_by_lf[lf] for lf in LIFESTYLE_ORDER if len(cv_by_lf[lf]) >= 3]
    if len(kw_groups) >= 2:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            H_kw, p_kw = kruskal(*kw_groups)
        stats.add(
            f"depthbin_cv_kw_{spec_tag}",
            "s1", "main",
            f"KW test on per-bin CV across lifestyles [{spec.description}]",
            f"H={H_kw:.3f}, p={p_kw:.3e}, k={len(kw_groups)} groups",
            test="Kruskal–Wallis",
            p_value=p_kw,
        )
        stat_rows.append({
            "quantity": "per-bin CV across lifestyles", "group": "",
            "comparison": f"{len(kw_groups)} lifestyle groups",
            "test": "Kruskal-Wallis", "statistic": float(H_kw),
            "p_value": float(p_kw), "effect_size": np.nan,
            # The bins the test actually used: a lifestyle with 1-2 qualifying
            # bins reaches cv_vals_for_kw but is excluded from kw_groups, so
            # that total would overstate the test's n.
            "n_bins": sum(len(g) for g in kw_groups), "detail": "",
        })
        print(f"\n    KW on per-bin CV: H={H_kw:.3f}, p={p_kw:.3e}")

        # Pairwise MWU with BH correction
        pairs, raw_ps = [], []
        for i, lf_a in enumerate(LIFESTYLE_ORDER):
            for lf_b in LIFESTYLE_ORDER[i + 1:]:
                if len(cv_by_lf[lf_a]) < 3 or len(cv_by_lf[lf_b]) < 3:
                    continue
                u, p_raw = mannwhitneyu(
                    cv_by_lf[lf_a], cv_by_lf[lf_b], alternative="two-sided")
                n1, n2 = len(cv_by_lf[lf_a]), len(cv_by_lf[lf_b])
                r_rb = float(1.0 - 2.0 * u / max(n1 * n2, 1))
                pairs.append((lf_a, lf_b, u, p_raw, r_rb, n1, n2))
                raw_ps.append(p_raw)

        adj_ps = _bh_correct(raw_ps)
        for (lf_a, lf_b, u, p_raw, r_rb, n1, n2), p_adj in zip(pairs, adj_ps):
            stats.add(
                f"depthbin_cv_mwu_{lf_a}_vs_{lf_b}_{spec_tag}",
                "s1", "main",
                f"Per-bin CV MWU {lf_a} vs {lf_b} [{spec.description}]",
                f"U={u:.0f}, p_BH={p_adj:.3e} {_star(p_adj)}, r={r_rb:+.3f}",
                test="Mann–Whitney U (BH corrected)",
                p_value=p_adj,
                effect_size=r_rb,
                n1=n1, n2=n2,
            )
            stat_rows.append({
                "quantity": "per-bin CV", "group": "",
                "comparison": f"{lf_a} vs {lf_b}",
                "test": "Mann-Whitney U, two-sided (BH corrected)",
                "statistic": float(u), "p_value": float(p_adj),
                "effect_size": r_rb, "n_bins": n1 + n2,
                "detail": f"p_raw={float(p_raw):.6g}, n1={n1}, n2={n2}",
            })
            print(f"      {lf_a} vs {lf_b}: p_BH={p_adj:.3e} "
                  f"{_star(p_adj)}, r={r_rb:+.3f}")

    # ── Permutation test: is mean CV homogeneous across lifestyles? ───────────
    # Test statistic: max pairwise difference in mean CV.
    # Permutes lifestyle labels across all qualifying (bin, sample) cells.
    if len(kw_groups) >= 2 and len(cv_vals_for_kw) >= 6:
        cv_arr = np.array(cv_vals_for_kw)
        lf_arr = np.array(cv_labels_for_kw)
        unique_lf = [lf for lf in LIFESTYLE_ORDER if lf in lf_arr]

        def _max_mean_diff(vals, labels):
            means = [vals[labels == lf].mean() for lf in unique_lf
                     if (labels == lf).sum() > 0]
            return max(means) - min(means) if len(means) >= 2 else 0.0

        obs_diff = _max_mean_diff(cv_arr, lf_arr)
        null_diffs = np.array([
            _max_mean_diff(cv_arr, rng.permutation(lf_arr))
            for _ in range(n_perm)
        ])
        p_perm = max(float(np.mean(null_diffs >= obs_diff)),
                     1.0 / (n_perm + 1))
        stats.add(
            f"depthbin_cv_perm_{spec_tag}",
            "s1", "main",
            f"Permutation test: max pairwise mean-CV difference across "
            f"lifestyles [{spec.description}; n_perm={n_perm}]",
            f"obs_diff={obs_diff:.4f}, p_perm={p_perm:.4f}",
            test=f"Label-permutation (n_perm={n_perm})",
            p_value=p_perm,
            effect_size=obs_diff,
        )
        stat_rows.append({
            "quantity": "max pairwise mean-CV difference", "group": "",
            "comparison": "lifestyle labels permuted",
            "test": f"label permutation (n_perm={n_perm}), one-sided",
            "statistic": float(obs_diff), "p_value": float(p_perm),
            "effect_size": float(obs_diff), "n_bins": len(cv_vals_for_kw),
            "detail": "p floored at 1/(n_perm+1)",
        })
        print(f"    Permutation test (max mean-CV diff): "
              f"obs={obs_diff:.4f}, p_perm={p_perm:.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(outdir, f"depth_binned_richness_cv_{spec_tag}.csv")
    bin_df.to_csv(csv_path, index=False)
    # The figure's data object: the binned frame the panels are drawn from,
    # and the tests its caption quotes.
    from . import figure_source
    _tables = {"bins": bin_df.reset_index(drop=True)}
    # The caption also quotes a per-lifestyle Spearman of depth against
    # richness. It is computed per sample in depth_dependence_audit, not here,
    # so it is read back rather than recomputed from the binned medians --
    # which would be a different statistic wearing the same name.
    #
    # Read from that step's CSV, not from the statistics registry. The registry
    # formats every field on the way in (stats_report._fmt/_fmt_p), so its p is
    # the string "< 1e-300" for three of the four lifestyles here and parses
    # back to nothing; it also holds only rows this *process* recorded, so a
    # resumed run that skips the depth audit would silently drop these. The CSV
    # carries rho and p as float64 and n as int64, and it is on disk either way.
    # Built the way depth_audit.py:68 builds it, rather than by walking up
    # from this figure's own directory.
    _audit = os.path.join("results", ref, "depth_audit",
                          f"depth_dependence_audit_{spec_tag}.csv")
    try:
        if os.path.exists(_audit):
            _a = pd.read_csv(_audit)
            _m = _a[(_a["x_var"] == "log10_total_bp")
                    & (_a["y_var"] == "richness")]
            for _r in _m.itertuples(index=False):
                _p = float(_r.p)
                stat_rows.append({
                    "quantity": "depth vs richness",
                    "group": str(_r.Lifestyle),
                    "comparison": "per sample",
                    "test": "Spearman rank correlation, two-sided",
                    "statistic": float(_r.rho),
                    "p_value": _p,
                    "effect_size": float(_r.rho),
                    "n_bins": np.nan,
                    "detail": f"n={int(_r.n)} samples; from "
                              f"depth_dependence_audit_{spec_tag}.csv"
                              + ("; p below double precision (< 1e-308)"
                                 if _p == 0.0 else ""),
                })
        else:
            print(f"    [data] {os.path.basename(_audit)} not found; S20's "
                  f"stats panel omits the depth-richness Spearman rows")
    except Exception as _exc:
        print(f"    [data] depth-richness Spearman lookup skipped: {_exc}")
    if stat_rows:
        _tables["stats"] = pd.DataFrame(stat_rows)
    figure_source.emit_for_path(csv_path.replace(".csv", ".png"), _tables)
    print(f"\n    Saved: {csv_path}")

    # ── Figure ────────────────────────────────────────────────────────────────
    _plot_depthbin(bin_df, grand_cv, spec, spec_tag, outdir, ref, min_n, FS)


def _plot_depthbin(bin_df, grand_cv, spec, spec_tag, outdir, ref,
                   min_n, FS):
    """
    Two-panel figure.

    Left  — Median richness per depth bin, by lifestyle.
             Ribbon = IQR (Q25–Q75).  Shows the strong depth→richness
             relationship and where the three lifestyle distributions sit.

    Right — CV (SD / median richness) per depth bin, by lifestyle.
             Each point is one occupied bin (n >= min_n).
             Dashed horizontal line = grand mean CV per lifestyle.
             This panel is the primary manuscript statistic: it shows
             that ~50% fractional variability persists at every depth,
             equally across lifestyles.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from .plot_utils import savefig_dual

    fig, (ax_r, ax_cv) = plt.subplots(1, 2, figsize=page_figsize(n_cols=2))

    for lf in LIFESTYLE_ORDER:
        color = config.LIFESTYLE_COLORS.get(lf, "gray")
        sub = bin_df[bin_df["Lifestyle"] == lf].copy()
        sub_ok = sub.dropna(subset=["median_richness"])

        if sub_ok.empty:
            continue

        xmid = sub_ok["bin_mid_log10bp"].values

        # ── Left: median + IQR ribbon ─────────────────────────────────────
        med = sub_ok["median_richness"].values
        q25 = sub_ok["q25_richness"].values
        q75 = sub_ok["q75_richness"].values
        n_valid = sub_ok["n"].values

        # Only show ribbon where n >= min_n
        ribbon_mask = n_valid >= min_n
        ax_r.plot(xmid, med, color=color, lw=2,
                  label=f"{_display_lifestyle(lf)} (n={int(sub_ok['n'].sum()):,})")
        if ribbon_mask.any():
            ax_r.fill_between(
                xmid[ribbon_mask], q25[ribbon_mask], q75[ribbon_mask],
                color=color, alpha=0.18)

        # ── Right: CV per bin ─────────────────────────────────────────────
        sub_cv = sub.dropna(subset=["cv_sd_median"])
        sub_cv = sub_cv[sub_cv["n"] >= min_n]
        if not sub_cv.empty:
            ax_cv.scatter(
                sub_cv["bin_mid_log10bp"].values,
                sub_cv["cv_sd_median"].values,
                color=color, alpha=0.65, s=28, edgecolors="none",
                label=f"{_display_lifestyle(lf)} bins",
                zorder=3,
            )
            # Grand mean line
            if lf in grand_cv:
                ax_cv.axhline(grand_cv[lf], color=color, ls="--",
                              lw=1.6, alpha=0.85,
                              label=f"{_display_lifestyle(lf)} mean CV = {grand_cv[lf]:.3f}")

    # ── Shared x-axis formatting ──────────────────────────────────────────
    # Tick positions at round Gbp values
    gbp_ticks = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    log_ticks  = [np.log10(v * 1e9) for v in gbp_ticks]
    for ax in (ax_r, ax_cv):
        ax.set_xticks(log_ticks)
        ax.set_xticklabels([str(v) for v in gbp_ticks],
                           rotation=30, fontsize=FS - 2)
        ax.set_xlabel("Sequencing depth (Gbp mapped)", fontsize=FS)
        ax.grid(alpha=0.25)

    ax_r.set_ylabel("Richness (SGBs detected)", fontsize=FS)
    ax_r.set_title(
        "Median richness per depth bin\n(ribbon = IQR)",
        fontsize=FS + 1, fontweight="bold")
    _pin_legend(ax_r.legend(fontsize=FS + 1))

    ax_cv.set_ylabel("CV = SD / median richness (per bin)", fontsize=FS)
    ax_cv.set_ylim(bottom=0)
    ax_cv.set_title(
        f"Richness CV by lifestyle\n(1 point = 1 bin, n ≥ {min_n})",
        fontsize=FS + 1, fontweight="bold")
    _pin_legend(ax_cv.legend(fontsize=FS + 1))

    fig.suptitle(
        f"Depth-binned richness variability — {ref}  [{spec.description}]\n"
        f"CV quantifies depth-independent richness spread within each lifestyle",
        fontsize=FS + 2, fontweight="bold")
    plt.tight_layout()

    png_path = os.path.join(outdir,
                            f"depth_binned_richness_cv_{spec_tag}.png")
    savefig_dual(fig, png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {png_path}")
