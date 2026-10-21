"""
Data-driven VANISH / BloSSUM definitions.

Instead of loading fixed lists from external files, this module derives
VANISH and BloSSUM taxa directly from the breadth/depth data using
prevalence-based criteria.

Two complementary approaches (from the sensitivity-analysis design document):

  Option A  — band-pass prevalence (hard thresholds):
      VANISH:  p_I <= r   AND  m <= p_N <= u
      BloSSUM: p_N <= r   AND  m <= p_I <= u

  Option B  — rarity-weighted log-odds gap (continuous score):
      S = |logit(p_I) - logit(p_N)| x sqrt(p_bar x (1 - p_bar))
      Still subject to one-sided rarity constraint, then ranked by S.

Both support optional study-aware prevalence aggregation to avoid
big-study domination.

Usage
-----
    from .vanish_blossum_definitions import (
        VBDefinition, compute_prevalence, define_vanish_blossum,
        PRESET_DEFINITIONS, sweep_definitions,
    )

    # Single definition
    vb = define_vanish_blossum(data, defn=VBDefinition(r=0.02, m=0.10, u=0.80))

    # Full sweep
    results = sweep_definitions(data, definitions=PRESET_DEFINITIONS)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set
from collections import OrderedDict

from . import config
from .detection import DetectionSpec, build_presence


# ════════════════════════════════════════════════════════════════════
#  DEFINITION DATACLASS
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VBDefinition:
    """
    A single VANISH/BloSSUM definition parameterisation.

    Parameters
    ----------
    r : float
        Rare cutoff — maximum prevalence in the "depleted" group (default 0.02).
    m : float
        Minimum prevalence in the "present" group (default 0.10).
    u : float
        Maximum prevalence in the "present" group (default 0.80).
    min_present_count : int
        Minimum number of presences in the "present" group (default 10).
    study_aware : bool
        If True, compute prevalence as median across study-level prevalences.
    breadth_threshold : float
        Breadth % threshold for declaring a taxon present (default 10.0).
    use_option_b_ranking : bool
        If True, rank taxa within VANISH/BloSSUM by the Option B score.
    label : str or None
        Human-readable label for this definition.
    """
    r: float = 0.02
    m: float = 0.10
    u: float = 0.80
    min_present_count: int = 10
    study_aware: bool = False
    breadth_threshold: float = 10.0
    use_option_b_ranking: bool = True
    label: Optional[str] = None

    @property
    def tag(self) -> str:
        """Filesystem-safe short tag for filenames."""
        if self.label:
            safe = self.label.replace(" ", "_").replace("/", "-")
            return safe
        parts = [f"r{self.r:.3f}", f"m{self.m:.2f}", f"u{self.u:.2f}"]
        if self.study_aware:
            parts.append("studyaware")
        return "_".join(parts)

    @property
    def description(self) -> str:
        """Human-readable description."""
        sa = " (study-aware)" if self.study_aware else ""
        lbl = f" [{self.label}]" if self.label else ""
        return (f"r<={self.r}, {self.m}<=p<={self.u}, "
                f"min_k={self.min_present_count}, "
                f"bt={self.breadth_threshold}%{sa}{lbl}")


@dataclass
class VBResult:
    """Result of a single VANISH/BloSSUM definition run."""
    definition: VBDefinition
    vanish_ids: List[str]
    blossum_ids: List[str]
    vanish_set: Set[str] = field(default_factory=set)
    blossum_set: Set[str] = field(default_factory=set)
    prevalence_df: Optional[pd.DataFrame] = None  # per-taxon prevalences + scores
    n_vanish: int = 0
    n_blossum: int = 0

    def __post_init__(self):
        self.vanish_set = set(self.vanish_ids)
        self.blossum_set = set(self.blossum_ids)
        self.n_vanish = len(self.vanish_ids)
        self.n_blossum = len(self.blossum_ids)


# ════════════════════════════════════════════════════════════════════
#  PREVALENCE COMPUTATION
# ════════════════════════════════════════════════════════════════════

def compute_prevalence(data, breadth_threshold=10.0, study_aware=False):
    """
    Compute per-taxon prevalence in Industrial and Non-Industrial groups.

    Parameters
    ----------
    data : ReferenceData
    breadth_threshold : float
        % breadth for calling a taxon present.
    study_aware : bool
        If True, compute median of within-study prevalences.

    Returns
    -------
    DataFrame with columns: taxon, prev_I, prev_N, k_I, k_N, n_I, n_N
    """
    meta = data.meta
    breadth = data.breadth

    ind_mask = meta["Lifestyle"] == "Industrial"
    ni_mask = meta["Lifestyle"].isin(["H-G", "Transitional"])

    taxa = breadth.columns.tolist()

    if study_aware and "_study" in meta.columns:
        return _compute_study_aware_prevalence(
            breadth, meta, ind_mask, ni_mask, taxa, breadth_threshold)
    else:
        return _compute_pooled_prevalence(
            breadth, meta, ind_mask, ni_mask, taxa, breadth_threshold)


def _compute_pooled_prevalence(breadth, meta, ind_mask, ni_mask,
                                taxa, threshold):
    """Simple pooled prevalence across all samples."""
    ind_idx = meta.index[ind_mask]
    ni_idx = meta.index[ni_mask]
    n_I = len(ind_idx)
    n_N = len(ni_idx)

    if n_I == 0 or n_N == 0:
        return pd.DataFrame(columns=["taxon", "prev_I", "prev_N",
                                      "k_I", "k_N", "n_I", "n_N"])

    present_I = (breadth.loc[ind_idx] >= threshold)
    present_N = (breadth.loc[ni_idx] >= threshold)

    k_I = present_I.sum(axis=0)
    k_N = present_N.sum(axis=0)

    rows = []
    for t in taxa:
        rows.append({
            "taxon": t,
            "prev_I": k_I[t] / n_I,
            "prev_N": k_N[t] / n_N,
            "k_I": int(k_I[t]),
            "k_N": int(k_N[t]),
            "n_I": n_I,
            "n_N": n_N,
        })
    return pd.DataFrame(rows)


def _compute_study_aware_prevalence(breadth, meta, ind_mask, ni_mask,
                                     taxa, threshold):
    """
    Compute prevalence as the median of within-study prevalences.

    This avoids big-study domination.
    """
    study_col = "_study"
    ind_idx = meta.index[ind_mask]
    ni_idx = meta.index[ni_mask]

    # Per-study prevalences
    study_prevs_I = {}
    study_prevs_N = {}

    for study, grp in meta.loc[ind_idx].groupby(study_col):
        if len(grp) < 5:  # skip tiny studies
            continue
        present = (breadth.loc[grp.index] >= threshold)
        study_prevs_I[study] = present.mean(axis=0)

    for study, grp in meta.loc[ni_idx].groupby(study_col):
        if len(grp) < 5:
            continue
        present = (breadth.loc[grp.index] >= threshold)
        study_prevs_N[study] = present.mean(axis=0)

    if not study_prevs_I or not study_prevs_N:
        return _compute_pooled_prevalence(
            breadth, meta, ind_mask, ni_mask, taxa, threshold)

    prevs_I_df = pd.DataFrame(study_prevs_I).T
    prevs_N_df = pd.DataFrame(study_prevs_N).T

    median_I = prevs_I_df.median(axis=0)
    median_N = prevs_N_df.median(axis=0)

    # Also compute pooled counts for the min_present_count guard
    present_I_all = (breadth.loc[ind_idx] >= threshold)
    present_N_all = (breadth.loc[ni_idx] >= threshold)
    k_I = present_I_all.sum(axis=0)
    k_N = present_N_all.sum(axis=0)

    rows = []
    for t in taxa:
        rows.append({
            "taxon": t,
            "prev_I": median_I.get(t, 0.0),
            "prev_N": median_N.get(t, 0.0),
            "k_I": int(k_I.get(t, 0)),
            "k_N": int(k_N.get(t, 0)),
            "n_I": len(ind_idx),
            "n_N": len(ni_idx),
            "n_studies_I": len(study_prevs_I),
            "n_studies_N": len(study_prevs_N),
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
#  OPTION B: RARITY-WEIGHTED LOG-ODDS SCORE
# ════════════════════════════════════════════════════════════════════

def _jeffreys_smooth(k, n):
    """Jeffreys prior smoothing to avoid zero/one prevalence."""
    return (k + 0.5) / (n + 1)


def logit(p):
    """Logit transform, clipped to avoid infinities."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def option_b_score(prev_I, prev_N):
    """
    Rarity-weighted log-odds gap score.

    S = |logit(p_I) - logit(p_N)| x sqrt(p_bar x (1 - p_bar))

    The sqrt(p_bar(1-p_bar)) term downweights taxa that are very rare
    in BOTH groups or ubiquitous in BOTH, and emphasises taxa with
    intermediate occupancy in the "present" group.
    """
    pI = np.clip(prev_I, 1e-12, 1 - 1e-12)
    pN = np.clip(prev_N, 1e-12, 1 - 1e-12)
    pbar = 0.5 * (pI + pN)
    logit_gap = np.abs(np.log(pI / (1 - pI)) - np.log(pN / (1 - pN)))
    weight = np.sqrt(pbar * (1 - pbar))
    return logit_gap * weight


# ════════════════════════════════════════════════════════════════════
#  MAIN DEFINITION FUNCTION
# ════════════════════════════════════════════════════════════════════

def define_vanish_blossum(data, defn=None, prev_df=None):
    """
    Derive VANISH and BloSSUM taxa from data using a VBDefinition.

    Parameters
    ----------
    data : ReferenceData
    defn : VBDefinition
        Threshold parameterisation. Defaults to VBDefinition().
    prev_df : DataFrame, optional
        Pre-computed prevalence table (from compute_prevalence).
        If None, computed on the fly.

    Returns
    -------
    VBResult
    """
    if defn is None:
        defn = VBDefinition()

    # Compute prevalences if not provided
    if prev_df is None:
        prev_df = compute_prevalence(
            data,
            breadth_threshold=defn.breadth_threshold,
            study_aware=defn.study_aware)

    r, m, u = defn.r, defn.m, defn.u
    min_k = defn.min_present_count

    # Option A: band-pass selection
    vanish_mask = (
        (prev_df["prev_I"] <= r) &
        (prev_df["prev_N"] >= m) &
        (prev_df["prev_N"] <= u) &
        (prev_df["k_N"] >= min_k)
    )
    blossum_mask = (
        (prev_df["prev_N"] <= r) &
        (prev_df["prev_I"] >= m) &
        (prev_df["prev_I"] <= u) &
        (prev_df["k_I"] >= min_k)
    )

    vanish_df = prev_df[vanish_mask].copy()
    blossum_df = prev_df[blossum_mask].copy()

    # Option B: score and rank
    if defn.use_option_b_ranking and not vanish_df.empty:
        vanish_df["score_B"] = option_b_score(
            vanish_df["prev_I"].values, vanish_df["prev_N"].values)
        vanish_df = vanish_df.sort_values("score_B", ascending=False)

    if defn.use_option_b_ranking and not blossum_df.empty:
        blossum_df["score_B"] = option_b_score(
            blossum_df["prev_I"].values, blossum_df["prev_N"].values)
        blossum_df = blossum_df.sort_values("score_B", ascending=False)

    vanish_ids = vanish_df["taxon"].tolist()
    blossum_ids = blossum_df["taxon"].tolist()

    # Build full prevalence table with scores for diagnostics
    full_prev = prev_df.copy()
    full_prev["score_B"] = option_b_score(
        full_prev["prev_I"].values, full_prev["prev_N"].values)
    full_prev["is_vanish"] = vanish_mask.values
    full_prev["is_blossum"] = blossum_mask.values

    return VBResult(
        definition=defn,
        vanish_ids=vanish_ids,
        blossum_ids=blossum_ids,
        prevalence_df=full_prev,
    )


# ════════════════════════════════════════════════════════════════════
#  PRESET DEFINITIONS  (for sensitivity sweep)
# ════════════════════════════════════════════════════════════════════

def _build_presets():
    """
    Build a list of preset VBDefinitions covering a sensitivity grid.

    Varies:
      r  ∈ {0.01, 0.02, 0.05}
      m  ∈ {0.05, 0.10, 0.20}
      u  ∈ {0.80}
      study_aware ∈ {False, True}
    Plus the "original" definition that mimics loading from files.
    """
    presets = []

    # Sensitivity grid
    for r in [0.01, 0.02, 0.05]:
        for m_val in [0.05, 0.10, 0.20]:
            for sa in [False, True]:
                sa_tag = "SA" if sa else "pool"
                label = f"r{r}_m{m_val}_u0.80_{sa_tag}"
                presets.append(VBDefinition(
                    r=r, m=m_val, u=0.80,
                    min_present_count=10,
                    study_aware=sa,
                    breadth_threshold=10.0,
                    use_option_b_ranking=True,
                    label=label,
                ))

    return presets


PRESET_DEFINITIONS = _build_presets()

# "Recommended" definition from the design document
DEFAULT_DEFINITION = VBDefinition(
    r=0.02, m=0.10, u=0.80,
    min_present_count=10,
    study_aware=False,
    breadth_threshold=10.0,
    use_option_b_ranking=True,
    label="default_r0.02_m0.10",
)

# Study-aware variant
DEFAULT_STUDY_AWARE = VBDefinition(
    r=0.02, m=0.10, u=0.80,
    min_present_count=10,
    study_aware=True,
    breadth_threshold=10.0,
    use_option_b_ranking=True,
    label="studyaware_r0.02_m0.10",
)


# ════════════════════════════════════════════════════════════════════
#  SWEEP FUNCTION
# ════════════════════════════════════════════════════════════════════

def sweep_definitions(data, definitions=None):
    """
    Run define_vanish_blossum for every definition in the list.

    Returns
    -------
    list of VBResult
    """
    if definitions is None:
        definitions = PRESET_DEFINITIONS

    results = []
    for defn in definitions:
        print(f"    VB definition: {defn.description}")
        vb = define_vanish_blossum(data, defn=defn)
        print(f"      → VANISH={vb.n_vanish}, BloSSUM={vb.n_blossum}")
        results.append(vb)

    return results


# ════════════════════════════════════════════════════════════════════
#  COMPARISON WITH ORIGINAL LISTS
# ════════════════════════════════════════════════════════════════════

def compare_with_original(vb_result, original_vanish_set, original_blossum_set):
    """
    Compare a data-driven VB result with the original file-based lists.

    Returns a dict of overlap statistics.
    """
    v_new = vb_result.vanish_set
    b_new = vb_result.blossum_set
    v_orig = original_vanish_set
    b_orig = original_blossum_set

    return {
        "defn_tag": vb_result.definition.tag,
        "n_vanish_new": len(v_new),
        "n_blossum_new": len(b_new),
        "n_vanish_orig": len(v_orig),
        "n_blossum_orig": len(b_orig),
        "vanish_overlap": len(v_new & v_orig),
        "vanish_jaccard": (len(v_new & v_orig) / len(v_new | v_orig)
                           if (v_new | v_orig) else 0.0),
        "blossum_overlap": len(b_new & b_orig),
        "blossum_jaccard": (len(b_new & b_orig) / len(b_new | b_orig)
                            if (b_new | b_orig) else 0.0),
        "vanish_only_new": len(v_new - v_orig),
        "vanish_only_orig": len(v_orig - v_new),
        "blossum_only_new": len(b_new - b_orig),
        "blossum_only_orig": len(b_orig - b_new),
    }
