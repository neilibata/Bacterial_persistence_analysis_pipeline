"""
Depth matching for the microbiome extinctions pipeline (v3).

Two matching modes
------------------
1. **1:1 nearest-neighbour matching** (Sections 1 & 2)
       For each sample in the smaller group, greedily assign the closest
       sample in the larger group by log₁₀(total_mapped_bp), without
       replacement, subject to a maximum tolerance.

2. **Three-way triplet matching** (Section 3)
       For the health analysis, we match triplets of samples — one each
       from Non-Industrial, Industrial_Healthy, and Industrial_Unhealthy —
       such that all three have at most a configurable fractional
       difference in total mapped bases.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


# ════════════════════════════════════════════════════════════════════
#  1.  EXPLICIT 1:1 NEAREST-NEIGHBOUR MATCHING
# ════════════════════════════════════════════════════════════════════

def match_samples_1to1(
    meta,
    total_bp,
    group_a_mask,
    group_b_mask,
    frac_tol: float = 0.03,
    label_a: str = "A",
    label_b: str = "B",
) -> Tuple[List[str], List[str], dict]:
    """
    Greedy 1:1 nearest-neighbour matching on sequencing depth.

    Samples with total_bp <= 0 are excluded before matching.
    The *smaller* group is the query; for each of its samples the
    closest unused sample in the larger group is assigned, provided
    the fractional depth difference does not exceed ``frac_tol``.

    Fractional difference = |D_a - D_b| / max(D_a, D_b)

    Returns (matched_a, matched_b, stats_dict).
    """
    idx_a = meta.index[group_a_mask]
    idx_b = meta.index[group_b_mask]

    # Get depth values and filter out zero/negative
    bp_a = total_bp.reindex(idx_a).astype(float)
    bp_b = total_bp.reindex(idx_b).astype(float)
    bp_a = bp_a[bp_a > 0]
    bp_b = bp_b[bp_b > 0]

    n_a_dropped = len(idx_a) - len(bp_a)
    n_b_dropped = len(idx_b) - len(bp_b)
    if n_a_dropped > 0 or n_b_dropped > 0:
        print(f"    [1:1 match] Excluded {n_a_dropped} {label_a} + "
              f"{n_b_dropped} {label_b} samples with 0 mapped bp")

    if len(bp_a) == 0 or len(bp_b) == 0:
        return [], [], {
            "n_pairs": 0,
            "n_a_input": int(group_a_mask.sum()),
            "n_b_input": int(group_b_mask.sum()),
            "label_a": label_a, "label_b": label_b,
            "n_a_valid": len(bp_a), "n_b_valid": len(bp_b),
            "frac_tol": frac_tol,
            "median_frac_diff": float("nan"), "max_frac_diff": float("nan"),
            "mean_frac_diff": float("nan"),
        }

    # Use log10 for efficient KDTree neighbor finding
    log_a = np.log10(bp_a)
    log_b = np.log10(bp_b)

    # Smaller group queries the larger
    if len(bp_a) <= len(bp_b):
        query_bp, ref_bp = bp_a, bp_b
        query_log, ref_log = log_a, log_b
        q_is_a = True
    else:
        query_bp, ref_bp = bp_b, bp_a
        query_log, ref_log = log_b, log_a
        q_is_a = False

    tree = cKDTree(ref_log.values.reshape(-1, 1))
    # Get multiple neighbors so we can find one within frac_tol
    # With strict tolerance (3%), need more candidates for greedy fallback
    k = min(50, len(ref_bp))
    dists, indices = tree.query(query_log.values.reshape(-1, 1), k=k)
    if k == 1:
        dists = dists.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # Greedy assignment: best matches first, without replacement
    # Score each query by its best log-distance
    best_log_dist = dists[:, 0]
    order = np.argsort(best_log_dist)

    used_ref = set()
    mq, mr, pair_frac_diffs = [], [], []

    for qi in order:
        q_val = float(query_bp.values[qi])
        matched = False

        # Try each of the k nearest neighbors
        for ki in range(k):
            ri = int(indices[qi, ki])
            if ri in used_ref:
                continue
            r_val = float(ref_bp.values[ri])
            frac_diff = abs(q_val - r_val) / max(q_val, r_val)
            if frac_diff <= frac_tol:
                used_ref.add(ri)
                mq.append(query_bp.index[qi])
                mr.append(ref_bp.index[ri])
                pair_frac_diffs.append(frac_diff)
                matched = True
                break

        # If none of the k neighbors worked, skip
        if not matched:
            pass

    matched_a, matched_b = (mq, mr) if q_is_a else (mr, mq)

    stats = {
        "n_pairs": len(matched_a),
        "n_a_input": int(group_a_mask.sum()),
        "n_b_input": int(group_b_mask.sum()),
        "n_a_valid": len(bp_a),
        "n_b_valid": len(bp_b),
        "label_a": label_a,
        "label_b": label_b,
        "frac_tol": frac_tol,
        "median_frac_diff": float(np.median(pair_frac_diffs)) if pair_frac_diffs else np.nan,
        "mean_frac_diff": float(np.mean(pair_frac_diffs)) if pair_frac_diffs else np.nan,
        "max_frac_diff": float(np.max(pair_frac_diffs)) if pair_frac_diffs else np.nan,
    }

    print(f"    [1:1 match] {label_a} (n={stats['n_a_valid']}) vs "
          f"{label_b} (n={stats['n_b_valid']}): "
          f"{stats['n_pairs']} pairs  "
          f"(median frac_diff={stats['median_frac_diff']:.4f}, "
          f"max frac_diff={stats['max_frac_diff']:.4f})")

    return matched_a, matched_b, stats


# ════════════════════════════════════════════════════════════════════
#  2.  THREE-WAY TRIPLET MATCHING (Section 3: NI ↔ IndH ↔ IndU)
# ════════════════════════════════════════════════════════════════════

def match_triplets(
    meta,
    total_bp,
    mask_ni,
    mask_ind_h,
    mask_ind_u,
    frac_tol: float = 0.01,
) -> Tuple[List[str], List[str], List[str], dict]:
    """
    Three-way greedy triplet matching on total mapped bases.

    Algorithm:
      1. Match Ind_Healthy to Ind_Unhealthy (1:1, greedy NN on log₁₀).
      2. For each H-U pair, find the closest unused NI sample.
      3. Accept the triplet if max pairwise fractional difference <= frac_tol.

    Parameters
    ----------
    frac_tol : float
        Maximum allowed fractional difference between any pair in a triplet.
        E.g. 0.01 means within 1% of each other.

    Returns
    -------
    (ids_ni, ids_ind_h, ids_ind_u, stats_dict)
        Three equal-length lists of matched sample IDs (triplet-aligned).
    """
    idx_ni = meta.index[mask_ni]
    idx_h = meta.index[mask_ind_h]
    idx_u = meta.index[mask_ind_u]

    bp_ni = total_bp.reindex(idx_ni).astype(float)
    bp_h = total_bp.reindex(idx_h).astype(float)
    bp_u = total_bp.reindex(idx_u).astype(float)

    # Remove zero/negative and non-finite
    bp_ni = bp_ni[(bp_ni > 0) & np.isfinite(bp_ni)]
    bp_h = bp_h[(bp_h > 0) & np.isfinite(bp_h)]
    bp_u = bp_u[(bp_u > 0) & np.isfinite(bp_u)]

    if len(bp_h) == 0 or len(bp_u) == 0 or len(bp_ni) == 0:
        return [], [], [], {"n_triplets": 0}

    log_ni = np.log10(bp_ni)
    log_h = np.log10(bp_h)
    log_u = np.log10(bp_u)

    # Step 1: match Ind_H to Ind_U (1:1, greedy NN)
    if len(log_h) <= len(log_u):
        query, ref, q_is_h = log_h, log_u, True
    else:
        query, ref, q_is_h = log_u, log_h, False

    tree_ref = cKDTree(ref.values.reshape(-1, 1))
    dists, indices = tree_ref.query(query.values.reshape(-1, 1), k=1)

    used_ref = set()
    pairs_h, pairs_u = [], []
    order = np.argsort(dists.ravel())

    for qi in order:
        ri = int(indices[qi])
        if ri in used_ref:
            remaining = [j for j in range(len(ref)) if j not in used_ref]
            if not remaining:
                break
            ri = min(remaining,
                     key=lambda j: abs(query.values[qi] - ref.values[j]))
        used_ref.add(ri)
        if q_is_h:
            pairs_h.append(query.index[qi])
            pairs_u.append(ref.index[ri])
        else:
            pairs_u.append(query.index[qi])
            pairs_h.append(ref.index[ri])

    # Step 2: for each H-U pair, find closest NI sample
    tree_ni = cKDTree(log_ni.values.reshape(-1, 1))
    used_ni = set()
    triplets_ni, triplets_h, triplets_u = [], [], []
    pair_frac_diffs = []

    # Sort pairs by the mean depth of the H-U pair (best matches first)
    pair_means = []
    for h_id, u_id in zip(pairs_h, pairs_u):
        pair_means.append(0.5 * (bp_h[h_id] + bp_u[u_id]))
    pair_order = np.argsort(
        [abs(bp_h[h] - bp_u[u]) for h, u in zip(pairs_h, pairs_u)])

    for pi in pair_order:
        h_id, u_id = pairs_h[pi], pairs_u[pi]
        mean_bp = 0.5 * (bp_h[h_id] + bp_u[u_id])
        log_mean = np.log10(mean_bp)

        # Find closest unused NI
        # Search multiple candidates in case closest is used
        k_search = min(20, len(log_ni))
        dists_ni, idxs_ni = tree_ni.query(
            np.array([[log_mean]]), k=k_search)
        dists_ni = dists_ni.ravel()
        idxs_ni = idxs_ni.ravel()

        found = False
        for j in range(k_search):
            ni_idx = int(idxs_ni[j])
            if ni_idx in used_ni:
                continue
            ni_id = log_ni.index[ni_idx]
            ni_bp = bp_ni[ni_id]

            # Check fractional tolerance for all three pairwise comparisons
            vals = [bp_h[h_id], bp_u[u_id], ni_bp]
            max_val = max(vals)
            min_val = min(vals)
            if max_val == 0:
                continue
            frac_diff = (max_val - min_val) / max_val

            if frac_diff <= frac_tol:
                used_ni.add(ni_idx)
                triplets_ni.append(ni_id)
                triplets_h.append(h_id)
                triplets_u.append(u_id)
                pair_frac_diffs.append(frac_diff)
                found = True
                break

        # If strict tolerance fails, skip this pair

    stats = {
        "n_triplets": len(triplets_ni),
        "n_ni_input": len(idx_ni),
        "n_ind_h_input": len(idx_h),
        "n_ind_u_input": len(idx_u),
        "frac_tol": frac_tol,
        "median_frac_diff": float(np.median(pair_frac_diffs)) if pair_frac_diffs else np.nan,
        "max_frac_diff": float(np.max(pair_frac_diffs)) if pair_frac_diffs else np.nan,
    }

    print(f"    [triplet match] NI (n={len(idx_ni)}) x "
          f"Ind-Healthy (n={len(idx_h)}) x "
          f"Ind-Unhealthy (n={len(idx_u)}): "
          f"{stats['n_triplets']} triplets  "
          f"(median frac_diff={stats['median_frac_diff']:.4f}, "
          f"max={stats['max_frac_diff']:.4f})")

    return triplets_ni, triplets_h, triplets_u, stats


# ════════════════════════════════════════════════════════════════════
#  3.  BUILD ALL MATCHED SETS FOR THE PIPELINE
# ════════════════════════════════════════════════════════════════════

def _is_positive(val):
    """Check if a value indicates disease-positive."""
    if pd.isna(val):
        return False
    if isinstance(val, (bool, np.bool_)):
        return val
    if isinstance(val, (int, float)):
        return val > 0
    s = str(val).strip().lower()
    return s in ("yes", "true", "1", "positive", "y")


def _stratify_unhealthy(meta, ind_unhealthy_mask, data):
    """
    Stratify Industrial_Unhealthy into 3 subcategories with equal n.

    Categories (from article):
      1. Obesity (BMI > 30)
      2. Gut & metabolic disease (T2D, IBD, CRC, etc.)
      3. Other disease

    Returns
    -------
    stratified_mask : pd.Series[bool] or None
        Boolean mask with exactly min_n x 3 True values, balanced across
        subcategories. None if stratification fails.
    stats : dict
        Per-subcategory counts.
    """
    from . import config

    idx_u = meta.index[ind_unhealthy_mask]
    if len(idx_u) < 30:
        return None, {}

    # Classify each unhealthy sample into a subcategory
    # Priority: obesity flag → named disease columns → other
    has_obesity = pd.Series(False, index=idx_u)
    if "Obesity_flag" in meta.columns:
        has_obesity = meta.loc[idx_u, "Obesity_flag"].fillna(False).astype(bool)

    # Gut & metabolic disease columns
    disease_cols = []
    for spec in config.DISEASE_NCD_MAP.values():
        if "cols" in spec:
            disease_cols.extend(spec["cols"])
    valid_dcols = [c for c in disease_cols if c in meta.columns]

    has_disease = pd.Series(False, index=idx_u)
    if valid_dcols:
        has_disease = meta.loc[idx_u, valid_dcols].apply(
            lambda row: any(_is_positive(v) for v in row), axis=1)

    # Assign subcategory (mutually exclusive, priority order)
    subcat = pd.Series("other", index=idx_u)
    subcat[has_disease] = "gut_metabolic"
    subcat[has_obesity] = "obesity"  # obesity takes priority

    counts = subcat.value_counts()
    print(f"    [stratify] Unhealthy subcategories: "
          f"{dict(counts.items())}")

    # Take equal numbers from each subcategory
    categories = ["obesity", "gut_metabolic", "other"]
    cat_counts = {c: int(counts.get(c, 0)) for c in categories}
    min_n = min(cat_counts.values())

    if min_n < 10:
        print(f"    [stratify] WARNING: smallest subcategory has only "
              f"{min_n} samples — using all unhealthy")
        return None, cat_counts

    # Randomly sample min_n from each subcategory
    rng = np.random.default_rng(42)
    selected = []
    for cat in categories:
        cat_ids = subcat[subcat == cat].index.tolist()
        chosen = rng.choice(cat_ids, size=min_n, replace=False).tolist()
        selected.extend(chosen)
        print(f"    [stratify] {cat}: {len(cat_ids)} available → "
              f"{min_n} selected")

    # Build mask
    stratified_mask = pd.Series(False, index=meta.index)
    stratified_mask[selected] = True

    stats = {
        "n_per_subcategory": min_n,
        "n_total": min_n * 3,
        **cat_counts,
    }
    print(f"    [stratify] Total: {min_n} x 3 = {min_n * 3} "
          f"stratified unhealthy samples")

    return stratified_mask, stats

@dataclass
class DepthMatchResult:
    """Container for all depth-matched sample sets."""

    # Sections 1 & 2: Ind ↔ Trans
    ind_vs_trans_ind: list = field(default_factory=list)
    ind_vs_trans_trans: list = field(default_factory=list)

    # Sections 1 & 2: Ind ↔ Non-Industrial
    ind_vs_ni_ind: list = field(default_factory=list)
    ind_vs_ni_ni: list = field(default_factory=list)

    # Section 1: Ind ↔ H-G
    ind_vs_hg_ind: list = field(default_factory=list)
    ind_vs_hg_hg: list = field(default_factory=list)

    # Section 3: triplets (NI ↔ IndHealthy ↔ IndUnhealthy)
    triplet_ni: list = field(default_factory=list)
    triplet_ind_h: list = field(default_factory=list)
    triplet_ind_u: list = field(default_factory=list)

    # Diagnostics
    stats: dict = field(default_factory=dict)

    @property
    def all_matched_ind_vs_ni(self):
        """Union of IDs from the Ind ↔ NI matching (for S2 differential)."""
        return list(set(self.ind_vs_ni_ind) | set(self.ind_vs_ni_ni))

    def summary(self):
        lines = ["Depth-matching summary:"]
        for name, st in self.stats.items():
            if "n_pairs" in st:
                lines.append(f"  {name}: {st['n_pairs']} pairs "
                             f"({st['label_a']} n={st['n_a_valid']}, "
                             f"{st['label_b']} n={st['n_b_valid']}, "
                             f"frac_tol={st['frac_tol']:.0%}, "
                             f"median_diff={st['median_frac_diff']:.4f}, "
                             f"max_diff={st['max_frac_diff']:.4f})")
            elif "n_triplets" in st:
                lines.append(f"  {name}: {st['n_triplets']} triplets "
                             f"(NI n={st['n_ni_input']}, "
                             f"IndH n={st['n_ind_h_input']}, "
                             f"IndU n={st['n_ind_u_input']})")
        return "\n".join(lines)


def build_depth_matched_sets(data, frac_tol=0.03,
                              health_frac_tol=0.01):
    """
    Build all depth-matched sample-ID sets needed by the pipeline.

    Parameters
    ----------
    data : ReferenceData
        Full pipeline data object (needed for subcategory classification).
    frac_tol : float
        Fractional tolerance for Sections 1/2 pairwise matching.
        E.g. 0.03 means matched samples differ by at most 3% in total bp.
    health_frac_tol : float
        Fractional tolerance for Section 3 triplet matching.
    """
    meta = data.meta
    total_bp = data.total_bp

    ind_mask = meta["Lifestyle"] == "Industrial"
    trans_mask = meta["Lifestyle"] == "Transitional"
    hg_mask = meta["Lifestyle"] == "H-G"
    ni_mask = trans_mask | hg_mask

    # ── Sections 1/2: Ind vs Transitional ──
    m_ind_t, m_trans, stats_it = match_samples_1to1(
        meta, total_bp, ind_mask, trans_mask,
        frac_tol=frac_tol,
        label_a="Industrial", label_b="Transitional")

    # ── Sections 1/2: Ind vs Non-Industrial ──
    m_ind_ni, m_ni, stats_ini = match_samples_1to1(
        meta, total_bp, ind_mask, ni_mask,
        frac_tol=frac_tol,
        label_a="Industrial", label_b="Non-Industrial")

    # ── Section 1: Ind vs H-G ──
    m_ind_hg, m_hg, stats_ihg = match_samples_1to1(
        meta, total_bp, ind_mask, hg_mask,
        frac_tol=frac_tol,
        label_a="Industrial", label_b="H-G")

    # ── Section 3: Three-way triplet matching ──
    # Article: "Within the Industrial_Unhealthy group, we depth-matched
    # 723 individuals in each of 3 smaller categories: obesity, gut and
    # metabolic disease, and other disease."
    #
    # Step 1: Stratify Ind_Unhealthy into 3 subcategories, take equal n
    # Step 2: Form triplets (NI x IndH x IndU) within 1% frac tolerance
    ind_healthy = ind_mask & (meta["Unhealthy"] == "Healthy")
    ind_unhealthy = ind_mask & (meta["Unhealthy"] == "Unhealthy")

    ind_u_stratified, subcat_stats = _stratify_unhealthy(
        meta, ind_unhealthy, data)

    if ind_u_stratified is not None:
        # Use the stratified mask (equal subcategory representation)
        t_ni, t_h, t_u, stats_triplet = match_triplets(
            meta, total_bp,
            ni_mask, ind_healthy, ind_u_stratified,
            frac_tol=health_frac_tol)
        stats_triplet["subcategory_stats"] = subcat_stats
    else:
        # Fallback: use all unhealthy
        t_ni, t_h, t_u, stats_triplet = match_triplets(
            meta, total_bp,
            ni_mask, ind_healthy, ind_unhealthy,
            frac_tol=health_frac_tol)

    return DepthMatchResult(
        ind_vs_trans_ind=m_ind_t,
        ind_vs_trans_trans=m_trans,
        ind_vs_ni_ind=m_ind_ni,
        ind_vs_ni_ni=m_ni,
        ind_vs_hg_ind=m_ind_hg,
        ind_vs_hg_hg=m_hg,
        triplet_ni=t_ni,
        triplet_ind_h=t_h,
        triplet_ind_u=t_u,
        stats={
            "Ind_vs_Trans": stats_it,
            "Ind_vs_NI": stats_ini,
            "Ind_vs_HG": stats_ihg,
            "Health_triplets": stats_triplet,
        },
    )
