"""
Section 1 trace export: save all data needed to reproduce every figure.

Runs the full alpha + beta diversity analysis for every comparison and
detection threshold, writing structured CSVs into a ``traces/`` directory.

Directory structure
-------------------
    results/{ref}/section1_diversity/traces/
        matching/
            ind_vs_ni_pairs.csv
            ind_vs_trans_pairs.csv
            ind_vs_hg_pairs.csv
            triplet_ids.csv
            matching_summary.csv
        alpha/{comparison}_{threshold}.csv   (per-sample richness + Shannon)
        beta/{comparison}_{threshold}_pcoa.csv
        beta/{comparison}_{threshold}_permanova.csv
        beta/permanova_all.csv               (combined)
        within/{lifestyle}_{threshold}_richness.csv
        within/{lifestyle}_{threshold}_beta_summary.csv

Every CSV contains enough metadata (sample ID, lifestyle, health status,
sequencing depth) to rebuild any panel.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu

from . import config
from .detection import DetectionSpec, build_presence


# ╔══════════════════════════════════════════════════════════════════╗
# ║  CONSTANTS                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

SPECS = [
    DetectionSpec("flat_bp", config.ACCUM_MAIN_TEXT_FLAT_BP),
    DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH),
]

N_PCOA_AXES = config.DIVERSITY_PCOA_AXES
N_PERMANOVA = config.DIVERSITY_N_PERMANOVA


# ╔══════════════════════════════════════════════════════════════════╗
# ║  HELPERS                                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def _shannon(presence_row, abundance_row):
    """Shannon H' from abundance of detected SGBs."""
    det = presence_row.astype(bool)
    a = abundance_row[det].values.astype(float)
    a = a[a > 0]
    if len(a) == 0:
        return 0.0
    p = a / a.sum()
    return float(-np.sum(p * np.log(p)))


def _compute_richness_shannon(data, ids, spec):
    """Return DataFrame with richness + Shannon for sample IDs."""
    presence = build_presence(
        data.breadth.loc[ids], data.depth.loc[ids],
        data.lengths, spec=spec)
    richness = presence.sum(axis=1)

    abund = data.abundance.loc[ids]
    shannon = pd.Series(
        [_shannon(presence.loc[s], abund.loc[s]) for s in ids],
        index=ids, name="Shannon")

    df = pd.DataFrame({
        "sample_id": ids,
        "richness": richness.loc[ids].values,
        "shannon": shannon.values,
        "total_bp": data.total_bp.loc[ids].values,
        "log10_bp": np.log10(data.total_bp.loc[ids].values.astype(float)),
        "Lifestyle": data.meta.loc[ids, "Lifestyle"].values,
    })
    if "Unhealthy" in data.meta.columns:
        df["Health"] = data.meta.loc[ids, "Unhealthy"].values
    if "Population" in data.meta.columns:
        df["Population"] = data.meta.loc[ids, "Population"].values
    df.index = ids
    return df


def _pcoa(D_sq, n_axes=3):
    """Classical PCoA from a square distance matrix. Returns coords, var%."""
    D2 = D_sq ** 2
    n = D2.shape[0]
    row_m = D2.mean(axis=1)
    col_m = D2.mean(axis=0)
    B = -0.5 * (D2 - row_m[:, None] - col_m[None, :] + D2.mean())

    try:
        from scipy.linalg import eigh
        evals, evecs = eigh(B, subset_by_index=[n - n_axes, n - 1])
        evals = evals[::-1]
        evecs = evecs[:, ::-1]
    except (ImportError, TypeError):
        evals, evecs = np.linalg.eigh(B)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx[:n_axes]]
        evecs = evecs[:, idx[:n_axes]]

    total = max(np.trace(B), 1e-10)
    var_pct = np.array([max(ev, 0) / total * 100 for ev in evals])
    coords = evecs * np.sqrt(np.maximum(evals, 0))
    return coords, var_pct


def _permanova(D_sq, labels, n_perm=999):
    """PERMANOVA F-stat and permutation p-value."""
    rng = np.random.default_rng(42)
    n = D_sq.shape[0]
    groups = np.unique(labels)
    a = len(groups)

    # D_sq does not change across permutations -- only the labels do -- so both
    # the squared matrix and the total sum of squares are loop-invariant. The
    # previous form recomputed `D ** 2` inside every call, allocating a fresh
    # copy of the full matrix 1,000 times: 791 MB per call at n=9,945, on top of
    # two more copies for the largest group's submatrix. That is what killed the
    # 2026-08-26 run during the 10% breadth pass, after the 300 kb pass had
    # already completed on the same samples.
    D2 = np.asarray(D_sq, dtype=np.float64) ** 2
    ss_t = D2.sum() / (2 * n)

    def _ssw(labs):
        """Within-group sum of squares, without materialising submatrices.

        For a 0/1 indicator vector v, the sum of D2 over the group's block is
        v @ D2 @ v -- the same quantity as D2[ix_(idx, idx)].sum(), computed as
        two matrix-vector products in O(n) extra memory rather than O(k^2).
        """
        ss_w = 0.0
        for g in groups:
            v = (labs == g).astype(np.float64)
            k = v.sum()
            if k < 2:
                continue
            ss_w += float(v @ (D2 @ v)) / (2 * k)
        return ss_w

    def _ss(_D, labs):
        w = _ssw(labs)
        return ss_t - w, w

    ss_a, ss_w = _ss(D_sq, labels)
    if ss_w == 0:
        return np.inf, 1 / (n_perm + 1)
    F_obs = (ss_a / max(a - 1, 1)) / (ss_w / max(n - a, 1))

    n_ge = 1
    for _ in range(n_perm):
        plab = rng.permutation(labels)
        sa, sw = _ss(D_sq, plab)
        if sw > 0:
            Fp = (sa / max(a - 1, 1)) / (sw / max(n - a, 1))
        else:
            Fp = 0
        if Fp >= F_obs:
            n_ge += 1
    return F_obs, n_ge / (n_perm + 1)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PAIRWISE COMPARISON TRACE                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

def _run_pairwise_trace(data, ids_a, ids_b, label_a, label_b,
                        comparison_name, spec, trace_dir, stats,
                        permanova_rows):
    """
    Full alpha + beta trace for one pairwise depth-matched comparison.
    """
    spec_tag = spec.short_label
    tag = f"{comparison_name}_{spec_tag}"

    alpha_dir = _ensure(os.path.join(trace_dir, "alpha"))
    beta_dir = _ensure(os.path.join(trace_dir, "beta"))

    n_a, n_b = len(ids_a), len(ids_b)
    all_ids = list(ids_a) + list(ids_b)

    print(f"      [{tag}] {label_a}={n_a:,}, {label_b}={n_b:,}")

    # ── Alpha ──
    df_a = _compute_richness_shannon(data, ids_a, spec)
    df_b = _compute_richness_shannon(data, ids_b, spec)
    df_a["group"] = label_a
    df_b["group"] = label_b
    df_a["pair_idx"] = np.arange(n_a)
    df_b["pair_idx"] = np.arange(n_b)
    alpha_df = pd.concat([df_a, df_b], ignore_index=True)
    alpha_df["comparison"] = comparison_name
    alpha_df["detection"] = spec.description
    alpha_path = os.path.join(alpha_dir, f"{tag}.csv")
    alpha_df.to_csv(alpha_path, index=False)

    med_a = np.median(df_a["richness"])
    med_b = np.median(df_b["richness"])
    ratio = med_a / max(med_b, 1e-10)
    _, p = mannwhitneyu(df_a["richness"].values, df_b["richness"].values,
                        alternative="two-sided")

    stats.add(f"trace_richness_{tag}", "s1", "main",
              f"Richness: {label_a} vs {label_b} ({spec.description})",
              f"median {label_a}={med_a:.0f}, {label_b}={med_b:.0f}, "
              f"ratio={ratio:.2f}x",
              test="Mann-Whitney U", p_value=p, n1=n_a, n2=n_b)

    # ── Beta (Bray-Curtis PCoA + PERMANOVA) ──
    abund = data.abundance.loc[all_ids]
    vals = np.nan_to_num(abund.values, nan=0.0)

    t0 = time.time()
    D = pdist(vals, metric="braycurtis")
    D = np.nan_to_num(D, nan=0.0)
    D_sq = squareform(D)
    print(f"        Bray-Curtis: {time.time() - t0:.1f}s")

    coords, var_pct = _pcoa(D_sq, N_PCOA_AXES)

    pcoa_df = pd.DataFrame(
        coords, index=all_ids,
        columns=[f"PCo{i+1}" for i in range(N_PCOA_AXES)])
    pcoa_df["sample_id"] = all_ids
    pcoa_df["group"] = ([label_a] * n_a) + ([label_b] * n_b)
    pcoa_df["Lifestyle"] = data.meta.loc[all_ids, "Lifestyle"].values
    pcoa_df["total_bp"] = data.total_bp.loc[all_ids].values
    pcoa_df["comparison"] = comparison_name
    pcoa_df["detection"] = spec.description
    for i in range(min(N_PCOA_AXES, len(var_pct))):
        pcoa_df[f"var_pct_PCo{i+1}"] = var_pct[i]
    pcoa_path = os.path.join(beta_dir, f"{tag}_pcoa.csv")
    pcoa_df.to_csv(pcoa_path, index=False)

    print(f"        PCoA: {var_pct[0]:.1f}% / {var_pct[1]:.1f}%")

    # PERMANOVA
    labels = np.array([label_a] * n_a + [label_b] * n_b)
    F_obs, p_perm = _permanova(D_sq, labels, N_PERMANOVA)
    print(f"        PERMANOVA: F={F_obs:.2f}, p={p_perm:.4f}")

    stats.add(f"trace_permanova_{tag}", "s1", "main",
              f"PERMANOVA: {label_a} vs {label_b} ({spec.description})",
              f"F={F_obs:.2f}",
              test="PERMANOVA", p_value=p_perm, n1=n_a, n2=n_b)

    permanova_rows.append(dict(
        comparison=comparison_name, detection=spec.description,
        label_a=label_a, label_b=label_b, n_a=n_a, n_b=n_b,
        F=F_obs, p=p_perm,
        PCo1_var=var_pct[0], PCo2_var=var_pct[1],
        richness_ratio=ratio, richness_p=p))

    return alpha_df, pcoa_df


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TRIPLET TRACE                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def _run_triplet_trace(data, dm, spec, trace_dir, stats, permanova_rows):
    """Full alpha + beta trace for triplet-matched NI / IndH / IndU."""
    from scipy.stats import wilcoxon

    spec_tag = spec.short_label
    tag = f"triplet_{spec_tag}"

    alpha_dir = _ensure(os.path.join(trace_dir, "alpha"))
    beta_dir = _ensure(os.path.join(trace_dir, "beta"))

    ids_ni = dm.triplet_ni
    ids_h = dm.triplet_ind_h
    ids_u = dm.triplet_ind_u

    if not ids_ni:
        print(f"      [{tag}] No triplets — skipping")
        return

    n = len(ids_ni)
    print(f"      [{tag}] {n:,} triplets")

    # ── Alpha ──
    frames = []
    for ids, label in [(ids_ni, "Non-Industrial"),
                       (ids_h, "Ind-Healthy"),
                       (ids_u, "Ind-Unhealthy")]:
        df = _compute_richness_shannon(data, ids, spec)
        df["group"] = label
        df["triplet_idx"] = np.arange(n)
        frames.append(df)
    alpha_df = pd.concat(frames, ignore_index=True)
    alpha_df["comparison"] = "triplet"
    alpha_df["detection"] = spec.description
    alpha_df.to_csv(os.path.join(alpha_dir, f"{tag}.csv"), index=False)

    rni = frames[0]["richness"].values
    rh = frames[1]["richness"].values
    ru = frames[2]["richness"].values

    # Paired Wilcoxon tests
    for l1, l2, r1, r2 in [
        ("Ind-Healthy", "Ind-Unhealthy", rh, ru),
        ("Non-Industrial", "Ind-Healthy", rni, rh),
        ("Non-Industrial", "Ind-Unhealthy", rni, ru),
    ]:
        _, p = wilcoxon(r1, r2, alternative="two-sided")
        ratio = np.median(r1) / max(np.median(r2), 1e-10)
        stats.add(f"trace_triplet_{l1}_vs_{l2}_{spec_tag}", "s1", "main",
                  f"Triplet richness: {l1} vs {l2} ({spec.description})",
                  f"ratio={ratio:.2f}x",
                  test="Wilcoxon signed-rank (paired)", p_value=p,
                  n1=n, n2=n)
        print(f"        {l1} vs {l2}: ratio={ratio:.2f}x, p={p:.2e}")

    # ── Beta (PCoA on all 3 groups) ──
    all_ids = list(ids_ni) + list(ids_h) + list(ids_u)
    abund = data.abundance.loc[all_ids]
    vals = np.nan_to_num(abund.values, nan=0.0)

    t0 = time.time()
    D = pdist(vals, metric="braycurtis")
    D = np.nan_to_num(D, nan=0.0)
    D_sq = squareform(D)
    print(f"        Bray-Curtis: {time.time() - t0:.1f}s")

    coords, var_pct = _pcoa(D_sq, N_PCOA_AXES)

    labels = (["Non-Industrial"] * n +
              ["Ind-Healthy"] * n +
              ["Ind-Unhealthy"] * n)

    pcoa_df = pd.DataFrame(
        coords, index=all_ids,
        columns=[f"PCo{i+1}" for i in range(N_PCOA_AXES)])
    pcoa_df["sample_id"] = all_ids
    pcoa_df["group"] = labels
    pcoa_df["Lifestyle"] = data.meta.loc[all_ids, "Lifestyle"].values
    pcoa_df["total_bp"] = data.total_bp.loc[all_ids].values
    pcoa_df["triplet_idx"] = list(range(n)) * 3
    pcoa_df["comparison"] = "triplet"
    pcoa_df["detection"] = spec.description
    for i in range(min(N_PCOA_AXES, len(var_pct))):
        pcoa_df[f"var_pct_PCo{i+1}"] = var_pct[i]
    pcoa_df.to_csv(os.path.join(beta_dir, f"{tag}_pcoa.csv"), index=False)

    print(f"        PCoA: {var_pct[0]:.1f}% / {var_pct[1]:.1f}%")

    # PERMANOVA (3-group)
    lab_arr = np.array(labels)
    F_obs, p_perm = _permanova(D_sq, lab_arr, N_PERMANOVA)
    print(f"        PERMANOVA (3-group): F={F_obs:.2f}, p={p_perm:.4f}")

    stats.add(f"trace_permanova_{tag}", "s1", "main",
              f"PERMANOVA: triplet 3-group ({spec.description})",
              f"F={F_obs:.2f}",
              test="PERMANOVA", p_value=p_perm, n1=n, n2=n)

    permanova_rows.append(dict(
        comparison="triplet", detection=spec.description,
        label_a="NI/IndH/IndU", label_b="3-group", n_a=3 * n, n_b=n,
        F=F_obs, p=p_perm,
        PCo1_var=var_pct[0], PCo2_var=var_pct[1],
        richness_ratio=np.nan, richness_p=np.nan))

    # Pairwise PERMANOVAs (IndH vs IndU, NI vs IndH, NI vs IndU)
    for l1, l2, ids1, ids2 in [
        ("Ind-Healthy", "Ind-Unhealthy", ids_h, ids_u),
        ("Non-Industrial", "Ind-Healthy", ids_ni, ids_h),
        ("Non-Industrial", "Ind-Unhealthy", ids_ni, ids_u),
    ]:
        sub_ids = list(ids1) + list(ids2)
        sub_idx = [all_ids.index(s) for s in sub_ids]
        D_sub = D_sq[np.ix_(sub_idx, sub_idx)]
        lab_sub = np.array([l1] * n + [l2] * n)
        Fp, pp = _permanova(D_sub, lab_sub, N_PERMANOVA)
        print(f"        PERMANOVA {l1} vs {l2}: F={Fp:.2f}, p={pp:.4f}")

        permanova_rows.append(dict(
            comparison=f"triplet_{l1}_vs_{l2}", detection=spec.description,
            label_a=l1, label_b=l2, n_a=n, n_b=n,
            F=Fp, p=pp, PCo1_var=np.nan, PCo2_var=np.nan,
            richness_ratio=np.nan, richness_p=np.nan))


# ╔══════════════════════════════════════════════════════════════════╗
# ║  WITHIN-GROUP TRACES                                            ║
# ╚══════════════════════════════════════════════════════════════════╝

def _run_within_traces(data, dm, spec, trace_dir, stats):
    """
    Within-group richness + beta diversity for each lifestyle (depth-matched).
    """
    spec_tag = spec.short_label
    within_dir = _ensure(os.path.join(trace_dir, "within"))

    ids_ind = dm.ind_vs_ni_ind
    ids_ni = dm.ind_vs_ni_ni
    if not ids_ind:
        return

    all_ids = list(ids_ind) + list(ids_ni)
    meta = data.meta

    # Split NI by lifestyle
    lifestyles = {}
    for lf in ["H-G", "Transitional"]:
        gids = [s for s in ids_ni if meta.loc[s, "Lifestyle"] == lf]
        if gids:
            lifestyles[lf] = gids
    lifestyles["Industrial"] = ids_ind

    # ── Per-sample richness by lifestyle ──
    rich_df = _compute_richness_shannon(data, all_ids, spec)
    rich_df.to_csv(os.path.join(
        within_dir, f"within_richness_{spec_tag}.csv"), index=False)

    for lf, gids in lifestyles.items():
        vals = rich_df.loc[gids, "richness"].values
        if len(vals) < 2:
            continue
        med = np.median(vals)
        cv = np.std(vals, ddof=1) / np.mean(vals) * 100
        q25, q75 = np.percentile(vals, [25, 75])
        stats.add(f"trace_within_rich_{lf}_{spec_tag}", "s1", "main",
                  f"Within-group richness {lf} ({spec.description})",
                  f"n={len(vals)}, median={med:.0f}, "
                  f"CV={cv:.1f}%, IQR={q25:.0f}–{q75:.0f}",
                  n1=len(vals))

    # ── Within-group beta diversity ──
    presence = build_presence(
        data.breadth.loc[all_ids], data.depth.loc[all_ids],
        data.lengths, spec=spec)
    abund = data.abundance.loc[all_ids]

    rng = np.random.default_rng(42)
    max_per = 1500
    beta_rows = []

    for lf, gids in lifestyles.items():
        if len(gids) < 10:
            continue
        sub = gids
        if len(gids) > max_per:
            sub = list(rng.choice(gids, max_per, replace=False))

        pres_g = presence.loc[sub].values.astype(float)
        abund_g = np.nan_to_num(abund.loc[sub].values, nan=0.0)

        jacc = pdist(pres_g, metric="jaccard")
        jacc = np.nan_to_num(jacc, nan=1.0)
        bc = pdist(abund_g, metric="braycurtis")
        bc = np.nan_to_num(bc, nan=1.0)

        for metric, vals in [("jaccard", jacc), ("braycurtis", bc)]:
            m = np.median(vals)
            q25, q75 = np.percentile(vals, [25, 75])
            beta_rows.append(dict(
                Lifestyle=lf, metric=metric, detection=spec.description,
                n_samples=len(sub), n_pairs=len(vals),
                median=m, q25=q25, q75=q75, mean=np.mean(vals)))
            stats.add(
                f"trace_within_beta_{metric}_{lf}_{spec_tag}",
                "s1", "main",
                f"Within {metric} {lf} ({spec.description})",
                f"median={m:.3f}, IQR=[{q25:.3f}–{q75:.3f}]",
                n1=len(sub))

    # Between-group
    n_bw = min(1000, len(ids_ind), len(ids_ni))
    ind_sub = list(rng.choice(ids_ind, n_bw, replace=False))
    ni_sub = list(rng.choice(ids_ni, n_bw, replace=False))
    combined = ind_sub + ni_sub
    pres_c = presence.loc[combined].values.astype(float)
    abund_c = np.nan_to_num(abund.loc[combined].values, nan=0.0)
    jacc_sq = squareform(pdist(pres_c, metric="jaccard"))
    bc_sq = squareform(pdist(abund_c, metric="braycurtis"))
    # Extract between-group pairs only
    for metric, D in [("jaccard", jacc_sq), ("braycurtis", bc_sq)]:
        between = D[:n_bw, n_bw:]
        bv = between.ravel()
        m = np.median(bv)
        q25, q75 = np.percentile(bv, [25, 75])
        beta_rows.append(dict(
            Lifestyle="Between (Ind vs NI)", metric=metric,
            detection=spec.description,
            n_samples=2 * n_bw, n_pairs=len(bv),
            median=m, q25=q25, q75=q75, mean=np.mean(bv)))

    pd.DataFrame(beta_rows).to_csv(
        os.path.join(within_dir, f"within_beta_{spec_tag}.csv"), index=False)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  MATCHING METADATA                                              ║
# ╚══════════════════════════════════════════════════════════════════╝

def _save_matching_metadata(data, dm, trace_dir):
    """Save matched pair IDs and depths so matching can be audited."""
    match_dir = _ensure(os.path.join(trace_dir, "matching"))
    meta = data.meta
    total_bp = data.total_bp

    # Pairwise comparisons
    for comp_name, ids_a, ids_b, label_a, label_b in [
        ("ind_vs_ni", dm.ind_vs_ni_ind, dm.ind_vs_ni_ni,
         "Industrial", "Non-Industrial"),
        ("ind_vs_trans", dm.ind_vs_trans_ind, dm.ind_vs_trans_trans,
         "Industrial", "Transitional"),
        ("ind_vs_hg", dm.ind_vs_hg_ind, dm.ind_vs_hg_hg,
         "Industrial", "H-G"),
    ]:
        if not ids_a:
            continue
        n = len(ids_a)
        bp_a = total_bp.loc[ids_a].values.astype(float)
        bp_b = total_bp.loc[ids_b].values.astype(float)
        frac_diff = np.abs(bp_a - bp_b) / np.maximum(bp_a, bp_b)

        df = pd.DataFrame({
            "pair_idx": np.arange(n),
            f"{label_a}_id": ids_a,
            f"{label_b}_id": ids_b,
            f"{label_a}_bp": bp_a,
            f"{label_b}_bp": bp_b,
            "frac_diff": frac_diff,
        })
        df.to_csv(os.path.join(match_dir, f"{comp_name}_pairs.csv"),
                  index=False)
        print(f"      {comp_name}: {n:,} pairs")

    # Triplets
    if dm.triplet_ni:
        n = len(dm.triplet_ni)
        df = pd.DataFrame({
            "triplet_idx": np.arange(n),
            "NI_id": dm.triplet_ni,
            "IndH_id": dm.triplet_ind_h,
            "IndU_id": dm.triplet_ind_u,
            "NI_bp": total_bp.loc[dm.triplet_ni].values,
            "IndH_bp": total_bp.loc[dm.triplet_ind_h].values,
            "IndU_bp": total_bp.loc[dm.triplet_ind_u].values,
        })
        df.to_csv(os.path.join(match_dir, "triplet_ids.csv"), index=False)
        print(f"      triplets: {n:,}")

    # Summary
    rows = []
    for name, st in dm.stats.items():
        row = {"comparison": name}
        row.update(st)
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(match_dir, "matching_summary.csv"), index=False)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PUBLIC ENTRY POINT                                            ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_section1_traces(data, dm, stats):
    """
    Run full Section 1 diversity + composition analysis for both
    detection thresholds, saving all traces.

    Called from run_section1() or standalone via --only s1_traces.
    """
    ref = data.ref_name
    trace_dir = os.path.join("results", ref, "section1_diversity", "traces")
    _ensure(trace_dir)

    print(f"\n{'='*60}")
    print(f"  Section 1 traces [{ref}]")
    print(f"  Output: {trace_dir}/")
    print(f"{'='*60}")

    t0 = time.time()

    # ── 1. Save matching metadata ──
    print(f"\n  [1/4] Saving matching metadata …")
    _save_matching_metadata(data, dm, trace_dir)

    # ── 2. Pairwise comparisons ──
    print(f"\n  [2/4] Pairwise alpha + beta traces …")
    permanova_rows = []

    comparisons = [
        ("ind_vs_ni", dm.ind_vs_ni_ind, dm.ind_vs_ni_ni,
         "Industrial", "Non-Industrial"),
        ("ind_vs_trans", dm.ind_vs_trans_ind, dm.ind_vs_trans_trans,
         "Industrial", "Transitional"),
        ("ind_vs_hg", dm.ind_vs_hg_ind, dm.ind_vs_hg_hg,
         "Industrial", "H-G"),
    ]

    for comp_name, ids_a, ids_b, label_a, label_b in comparisons:
        if not ids_a or not ids_b:
            print(f"    {comp_name}: no matched pairs — skipping")
            continue
        for spec in SPECS:
            print(f"    {comp_name} [{spec.short_label}]:")
            _run_pairwise_trace(data, ids_a, ids_b, label_a, label_b,
                                comp_name, spec, trace_dir, stats,
                                permanova_rows)

    # ── 3. Triplet health analysis ──
    print(f"\n  [3/4] Triplet health traces …")
    for spec in SPECS:
        print(f"    triplet [{spec.short_label}]:")
        _run_triplet_trace(data, dm, spec, trace_dir, stats, permanova_rows)

    # Save combined PERMANOVA table
    beta_dir = _ensure(os.path.join(trace_dir, "beta"))
    pd.DataFrame(permanova_rows).to_csv(
        os.path.join(beta_dir, "permanova_all.csv"), index=False)

    # ── 4. Within-group variation ──
    print(f"\n  [4/4] Within-group variation traces …")
    for spec in SPECS:
        print(f"    within [{spec.short_label}]:")
        _run_within_traces(data, dm, spec, trace_dir, stats)

    elapsed = time.time() - t0
    print(f"\n  Section 1 traces complete [{ref}] ({elapsed / 60:.1f} min)")
    print(f"  Traces saved in: {trace_dir}/")

    # Print a summary of what was generated
    _print_trace_summary(trace_dir)


def _print_trace_summary(trace_dir):
    """Print a summary of all trace files generated."""
    print(f"\n  Trace file inventory:")
    total_size = 0
    for root, dirs, files in os.walk(trace_dir):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            total_size += size
            rel = os.path.relpath(fpath, trace_dir)
            print(f"    {rel:<55s} {size / 1024:>8.1f} kB")
    print(f"  Total: {total_size / 1024 / 1024:.1f} MB")
