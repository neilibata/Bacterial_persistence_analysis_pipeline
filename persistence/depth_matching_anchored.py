"""
Anchor-based 1-to-K depth matching for controlled multi-group comparisons.

Motivation
----------
The standard 1:1 matching (depth_matching.py) pairs the smaller group against
the larger.  This module implements a different strategy:

  • **Anchor on the smallest group** (e.g. H-G or Non-Industrial).
  • For each anchor sample, find *K* distinct depth-matched samples from each
    comparison group (Transitional, Industrial), all within a fractional
    tolerance of the anchor's depth.
  • K is uniform across anchors — every anchor maps to the same k_Trans
    Transitional and k_Ind Industrial samples.
  • All matched samples are unique (used exactly once).

This yields balanced, depth-controlled sets suitable for downstream alpha/beta
diversity and accumulation analyses where the anchor group drives the comparison.

Two analyses
------------
1. **H-G-anchored**:  Best 90% of H-G → k_Trans x Transitional + k_Ind x Industrial
2. **NI-anchored**:   Best 80% of Non-Industrial → k_Ind x Industrial

Algorithm
---------
1. For each (anchor, reference_group) pair, build a bipartite graph where
   edges connect anchor samples to reference samples within `frac_tol`.
2. Try K = 1, 2, 3, ... and find the largest subset of anchors such that
   every anchor has >=K available (unused) reference matches.  This is solved
   greedily: rank anchors by number of available matches, take the top
   `pct_keep` fraction, then greedily assign K matches per anchor (closest
   first, without replacement).
3. Across multiple reference groups, intersect the viable anchor subsets
   (an anchor must be viable for ALL groups simultaneously).
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


# ════════════════════════════════════════════════════════════════════
#  1.  CORE: GREEDY 1-to-K MATCHING
# ════════════════════════════════════════════════════════════════════

def _count_available_matches(
    anchor_bp: pd.Series,
    ref_bp: pd.Series,
    frac_tol: float,
    k_max: int = 20,
) -> Dict[str, int]:
    """
    For each anchor sample, count how many reference samples fall within
    `frac_tol` fractional depth difference.

    Returns dict: anchor_id → n_available_matches
    """
    if len(anchor_bp) == 0 or len(ref_bp) == 0:
        return {}

    log_anchor = np.log10(anchor_bp.values.astype(float))
    log_ref = np.log10(ref_bp.values.astype(float))

    tree = cKDTree(log_ref.reshape(-1, 1))

    # Search enough neighbors to cover the tolerance window
    k_search = min(k_max * 5, len(ref_bp))
    dists, indices = tree.query(log_anchor.reshape(-1, 1), k=k_search)
    if k_search == 1:
        dists = dists.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    counts = {}
    for i, aid in enumerate(anchor_bp.index):
        n_match = 0
        a_val = float(anchor_bp.values[i])
        for j in range(k_search):
            ri = int(indices[i, j])
            r_val = float(ref_bp.values[ri])
            frac_diff = abs(a_val - r_val) / max(a_val, r_val)
            if frac_diff <= frac_tol:
                n_match += 1
        counts[aid] = n_match

    return counts


def _greedy_assign_k(
    anchor_ids: List[str],
    anchor_bp: pd.Series,
    ref_bp: pd.Series,
    k: int,
    frac_tol: float,
) -> Tuple[List[str], Dict[str, List[str]], dict]:
    """
    Greedy 1-to-K assignment: for each anchor in `anchor_ids`, assign exactly
    `k` reference samples (closest first, no replacement).

    Returns
    -------
    successful_anchors : list of anchor IDs that got k matches
    assignments : dict  anchor_id → [ref_id_1, ..., ref_id_k]
    stats : dict
    """
    log_anchor = np.log10(anchor_bp.reindex(anchor_ids).values.astype(float))
    log_ref = np.log10(ref_bp.values.astype(float))

    tree = cKDTree(log_ref.reshape(-1, 1))
    k_search = min(k * 10, len(ref_bp))
    dists, indices = tree.query(log_anchor.reshape(-1, 1), k=k_search)
    if k_search == 1:
        dists = dists.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # Sort anchors by their best (tightest) match — give priority to
    # anchors with fewer options
    anchor_match_counts = []
    for i in range(len(anchor_ids)):
        a_val = float(anchor_bp[anchor_ids[i]])
        n = 0
        for j in range(k_search):
            ri = int(indices[i, j])
            r_val = float(ref_bp.values[ri])
            fd = abs(a_val - r_val) / max(a_val, r_val)
            if fd <= frac_tol:
                n += 1
        anchor_match_counts.append(n)

    # Process anchors with fewest options first (tightest constraint first)
    order = np.argsort(anchor_match_counts)

    used_ref = set()
    assignments = {}
    successful = []
    pair_frac_diffs = []

    for idx in order:
        aid = anchor_ids[idx]
        a_val = float(anchor_bp[aid])
        matched = []

        for j in range(k_search):
            ri = int(indices[idx, j])
            if ri in used_ref:
                continue
            r_val = float(ref_bp.values[ri])
            fd = abs(a_val - r_val) / max(a_val, r_val)
            if fd <= frac_tol:
                matched.append((ri, ref_bp.index[ri], fd))
                if len(matched) == k:
                    break

        if len(matched) == k:
            for ri, rid, fd in matched:
                used_ref.add(ri)
                pair_frac_diffs.append(fd)
            assignments[aid] = [rid for _, rid, _ in matched]
            successful.append(aid)

    stats = {
        "n_anchors_attempted": len(anchor_ids),
        "n_anchors_successful": len(successful),
        "n_ref_used": len(used_ref),
        "k": k,
        "frac_tol": frac_tol,
        "median_frac_diff": float(np.median(pair_frac_diffs)) if pair_frac_diffs else np.nan,
        "mean_frac_diff": float(np.mean(pair_frac_diffs)) if pair_frac_diffs else np.nan,
        "max_frac_diff": float(np.max(pair_frac_diffs)) if pair_frac_diffs else np.nan,
    }

    return successful, assignments, stats


# ════════════════════════════════════════════════════════════════════
#  2.  FIND BEST ANCHOR SUBSET + K FOR MULTIPLE REFERENCE GROUPS
# ════════════════════════════════════════════════════════════════════

def find_best_anchored_matching(
    meta,
    total_bp,
    anchor_mask,
    ref_group_masks: Dict[str, pd.Series],
    frac_tol: float = 0.05,
    pct_keep: float = 0.90,
    k_max_try: int = 10,
    anchor_label: str = "Anchor",
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, List[str]], dict]:
    """
    Find the best anchor subset and per-group K values.

    Algorithm:
      1. For each reference group, count per-anchor available matches.
      2. Intersect: keep only anchors viable across ALL groups.
      3. Take the best `pct_keep` fraction (ranked by min availability).
      4. For each reference group, find max K such that all selected anchors
         can be assigned K matches.
      5. Greedy-assign.

    Parameters
    ----------
    meta : DataFrame
    total_bp : Series
    anchor_mask : boolean mask for anchor group
    ref_group_masks : dict  group_name → boolean mask
    frac_tol : float  max fractional depth difference
    pct_keep : float  fraction of anchor samples to keep (e.g. 0.90)
    k_max_try : int  maximum K to attempt
    anchor_label : str  for logging

    Returns
    -------
    anchor_ids : list of selected anchor sample IDs
    ref_assignments : dict  group_name → flat list of matched ref IDs
                      (length = len(anchor_ids) * k_group)
    anchor_to_refs : dict  group_name → dict(anchor_id → [ref_ids])
    stats : dict  with per-group stats and overall summary
    """
    anchor_idx = meta.index[anchor_mask]
    anchor_bp = total_bp.reindex(anchor_idx).astype(float)
    anchor_bp = anchor_bp[anchor_bp > 0]

    n_anchor = len(anchor_bp)
    n_keep = int(np.ceil(pct_keep * n_anchor))

    print(f"\n    ── Anchored matching: {anchor_label} ──")
    print(f"    Anchor pool: {n_anchor} samples, keeping best {pct_keep:.0%} = {n_keep}")
    print(f"    Tolerance: {frac_tol:.0%} fractional depth difference")

    if n_anchor == 0 or n_keep == 0:
        print(f"    [SKIP] No {anchor_label} anchors available")
        empty_refs = {gname: [] for gname in ref_group_masks}
        empty_a2r = {gname: {} for gname in ref_group_masks}
        stats = {"anchor_label": anchor_label, "n_anchor_pool": n_anchor,
                 "n_anchor_selected": 0, "n_anchor_final": 0,
                 "pct_keep": pct_keep, "frac_tol": frac_tol}
        for gname in ref_group_masks:
            stats[f"{gname}_k"] = 0
            stats[f"{gname}_n_refs"] = 0
        return [], empty_refs, empty_a2r, stats

    # Step 1: Count available matches per anchor, per group
    group_counts = {}  # group → {anchor_id: count}
    group_bps = {}
    for gname, gmask in ref_group_masks.items():
        gidx = meta.index[gmask]
        gbp = total_bp.reindex(gidx).astype(float)
        gbp = gbp[gbp > 0]
        group_bps[gname] = gbp
        counts = _count_available_matches(anchor_bp, gbp, frac_tol)
        group_counts[gname] = counts
        print(f"    {gname}: {len(gbp)} ref samples, "
              f"median matches/anchor = "
              f"{np.median(list(counts.values())):.0f}")

    # Step 2: For each anchor, compute min availability across all groups
    min_avail = {}
    for aid in anchor_bp.index:
        mins = [group_counts[g].get(aid, 0) for g in ref_group_masks]
        min_avail[aid] = min(mins) if mins else 0

    # Step 3: Rank by min availability, take top pct_keep
    ranked = sorted(anchor_bp.index, key=lambda x: min_avail[x], reverse=True)
    selected_anchors = ranked[:n_keep]

    # Check that all selected anchors have >=1 match in every group
    selected_anchors = [a for a in selected_anchors if min_avail[a] >= 1]
    if len(selected_anchors) < n_keep:
        print(f"    WARNING: only {len(selected_anchors)}/{n_keep} anchors "
              f"have >=1 match in all groups")
    n_keep = len(selected_anchors)
    if n_keep == 0:
        print(f"    [SKIP] No {anchor_label} anchors have matches in all groups")
        empty_refs = {gname: [] for gname in ref_group_masks}
        empty_a2r = {gname: {} for gname in ref_group_masks}
        stats = {"anchor_label": anchor_label, "n_anchor_pool": n_anchor,
                 "n_anchor_selected": 0, "n_anchor_final": 0,
                 "pct_keep": pct_keep, "frac_tol": frac_tol}
        for gname in ref_group_masks:
            stats[f"{gname}_k"] = 0
            stats[f"{gname}_n_refs"] = 0
        return [], empty_refs, empty_a2r, stats

    # Step 4: For each group, find the max K achievable
    group_k = {}
    for gname in ref_group_masks:
        counts = group_counts[gname]
        anchor_avail = sorted([counts.get(a, 0) for a in selected_anchors])
        # K must be <= the minimum availability across all selected anchors
        # But we also need K * n_keep <= n_ref (can't use more refs than exist)
        n_ref = len(group_bps[gname])
        for k_try in range(k_max_try, 0, -1):
            # Check: enough reference samples?
            if k_try * n_keep > n_ref:
                continue
            # Check: every selected anchor has >=k_try matches?
            if anchor_avail[0] >= k_try:  # min across selected anchors
                group_k[gname] = k_try
                break
        else:
            group_k[gname] = 1
        print(f"    {gname}: achievable K = {group_k[gname]} "
              f"(needs {group_k[gname] * n_keep} of {n_ref} ref samples)")

    # Step 5: Greedy assignment per group
    ref_assignments = {}
    anchor_to_refs = {}
    all_stats = {"anchor_label": anchor_label, "n_anchor_pool": n_anchor,
                 "n_anchor_selected": n_keep, "pct_keep": pct_keep,
                 "frac_tol": frac_tol}

    final_anchors = set(selected_anchors)

    for gname in ref_group_masks:
        k = group_k[gname]
        successful, assignments, gstats = _greedy_assign_k(
            list(selected_anchors), anchor_bp, group_bps[gname],
            k=k, frac_tol=frac_tol)

        # Intersect: only keep anchors that succeeded in ALL groups
        final_anchors &= set(successful)

        anchor_to_refs[gname] = assignments
        all_stats[f"{gname}_k"] = k
        all_stats[f"{gname}_stats"] = gstats

        print(f"    {gname}: assigned K={k} to {len(successful)}/{n_keep} anchors "
              f"(median Δ={gstats['median_frac_diff']:.4f}, "
              f"max Δ={gstats['max_frac_diff']:.4f})")

    # Reconcile: only keep anchors successful in ALL groups
    final_anchor_list = [a for a in selected_anchors if a in final_anchors]

    # Rebuild flat ref lists for the final anchor set
    for gname in ref_group_masks:
        flat_refs = []
        for aid in final_anchor_list:
            if aid in anchor_to_refs[gname]:
                flat_refs.extend(anchor_to_refs[gname][aid])
        ref_assignments[gname] = flat_refs

        # Trim anchor_to_refs to final set
        anchor_to_refs[gname] = {
            a: anchor_to_refs[gname][a]
            for a in final_anchor_list
            if a in anchor_to_refs[gname]
        }

    all_stats["n_anchor_final"] = len(final_anchor_list)
    for gname in ref_group_masks:
        k = group_k[gname]
        all_stats[f"{gname}_n_refs"] = len(ref_assignments[gname])
        expected = len(final_anchor_list) * k
        actual = len(ref_assignments[gname])
        if expected != actual:
            print(f"    WARNING: {gname} expected {expected} refs, got {actual}")

    print(f"\n    Final: {len(final_anchor_list)} {anchor_label} anchors x "
          + " x ".join(f"K_{g}={group_k[g]}" for g in ref_group_masks))
    for gname in ref_group_masks:
        print(f"      {gname}: {len(ref_assignments[gname])} matched samples "
              f"(K={group_k[gname]})")

    return final_anchor_list, ref_assignments, anchor_to_refs, all_stats


# ════════════════════════════════════════════════════════════════════
#  3.  RESULT CONTAINER
# ════════════════════════════════════════════════════════════════════

@dataclass
class AnchoredMatchResult:
    """Container for anchored 1-to-K depth-matched sets."""

    # H-G-anchored three-way matching
    hg_anchor_ids: list = field(default_factory=list)
    hg_trans_ids: list = field(default_factory=list)    # flat: n_hg x k_trans
    hg_ind_ids: list = field(default_factory=list)      # flat: n_hg x k_ind
    hg_k_trans: int = 0
    hg_k_ind: int = 0
    hg_anchor_to_trans: dict = field(default_factory=dict)
    hg_anchor_to_ind: dict = field(default_factory=dict)

    # NI-anchored pairwise matching
    ni_anchor_ids: list = field(default_factory=list)
    ni_ind_ids: list = field(default_factory=list)      # flat: n_ni x k_ind
    ni_k_ind: int = 0
    ni_anchor_to_ind: dict = field(default_factory=dict)

    # Diagnostics
    hg_stats: dict = field(default_factory=dict)
    ni_stats: dict = field(default_factory=dict)

    def summary(self):
        lines = ["Anchored depth-matching summary:"]
        if self.hg_anchor_ids:
            lines.append(
                f"  H-G-anchored: {len(self.hg_anchor_ids)} H-G x "
                f"K_Trans={self.hg_k_trans} ({len(self.hg_trans_ids)} Trans) x "
                f"K_Ind={self.hg_k_ind} ({len(self.hg_ind_ids)} Ind)")
        if self.ni_anchor_ids:
            lines.append(
                f"  NI-anchored:  {len(self.ni_anchor_ids)} NI x "
                f"K_Ind={self.ni_k_ind} ({len(self.ni_ind_ids)} Ind)")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
#  4.  PUBLIC ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def build_anchored_matched_sets(
    data,
    hg_frac_tol: float = 0.05,
    hg_pct_keep: float = 0.90,
    ni_frac_tol: float = 0.05,
    ni_pct_keep: float = 0.80,
    k_max_try: int = 10,
) -> AnchoredMatchResult:
    """
    Build anchored 1-to-K depth-matched sample sets.

    Parameters
    ----------
    data : ReferenceData
    hg_frac_tol : float
        Max fractional depth difference for H-G-anchored matching.
    hg_pct_keep : float
        Fraction of H-G samples to retain (best-matched 90%).
    ni_frac_tol : float
        Max fractional depth difference for NI-anchored matching.
    ni_pct_keep : float
        Fraction of NI samples to retain (best-matched 80%).
    k_max_try : int
        Maximum K to attempt.
    """
    meta = data.meta
    total_bp = data.total_bp

    ind_mask = meta["Lifestyle"] == "Industrial"
    trans_mask = meta["Lifestyle"] == "Transitional"
    hg_mask = meta["Lifestyle"] == "H-G"
    ni_mask = trans_mask | hg_mask

    result = AnchoredMatchResult()

    # ── A. H-G-anchored: H-G → (Transitional, Industrial) ──
    print(f"\n{'─'*60}")
    print(f"  Anchored matching A: H-G → Transitional + Industrial")
    print(f"{'─'*60}")

    hg_anchors, hg_refs, hg_a2r, hg_stats = find_best_anchored_matching(
        meta, total_bp,
        anchor_mask=hg_mask,
        ref_group_masks={
            "Transitional": trans_mask,
            "Industrial": ind_mask,
        },
        frac_tol=hg_frac_tol,
        pct_keep=hg_pct_keep,
        k_max_try=k_max_try,
        anchor_label="H-G",
    )

    result.hg_anchor_ids = hg_anchors
    result.hg_trans_ids = hg_refs.get("Transitional", [])
    result.hg_ind_ids = hg_refs.get("Industrial", [])
    result.hg_k_trans = hg_stats.get("Transitional_k", 0)
    result.hg_k_ind = hg_stats.get("Industrial_k", 0)
    result.hg_anchor_to_trans = hg_a2r.get("Transitional", {})
    result.hg_anchor_to_ind = hg_a2r.get("Industrial", {})
    result.hg_stats = hg_stats

    # ── B. NI-anchored: Non-Industrial → Industrial ──
    print(f"\n{'─'*60}")
    print(f"  Anchored matching B: Non-Industrial → Industrial")
    print(f"{'─'*60}")

    ni_anchors, ni_refs, ni_a2r, ni_stats = find_best_anchored_matching(
        meta, total_bp,
        anchor_mask=ni_mask,
        ref_group_masks={
            "Industrial": ind_mask,
        },
        frac_tol=ni_frac_tol,
        pct_keep=ni_pct_keep,
        k_max_try=min(k_max_try, 5),  # NI is larger, K likely 2-3
        anchor_label="Non-Industrial",
    )

    result.ni_anchor_ids = ni_anchors
    result.ni_ind_ids = ni_refs.get("Industrial", [])
    result.ni_k_ind = ni_stats.get("Industrial_k", 0)
    result.ni_anchor_to_ind = ni_a2r.get("Industrial", {})
    result.ni_stats = ni_stats

    print(f"\n{result.summary()}")

    return result
