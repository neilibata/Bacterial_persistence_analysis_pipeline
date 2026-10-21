#!/usr/bin/env python3
"""
three_group_occupancy.py
========================
Three-group (Industrial / Transitional / Hunter-Gatherer) per-SGB occupancy
analysis, with equal-population weighting, cluster bootstrap, and Bayesian
FDR-controlled category assignment.

This is a separate analysis that sits on top of the same preprocessing
pipeline used by figure_summary.py. It imports load_reference(),
estimate_group_model_bb(), and associated helpers directly from that module
so that breadth matrices, depth matrices, genome lengths, and metadata are
loaded identically.

Core estimands
--------------
For each SGB k and lifestyle group g ∈ {Industrial, Transitional, HG}:

  ω̄_k,g  = (1/P_g) Σ_{p: g(p)=g} ω_k,p     [equal-population mean occupancy]
  M_k,g(τ) = #{p: g(p)=g, ω_k,p > τ}          [population replication count]
  Λ_k,g←ref = Σ_{s∈g} h_ks(ω_k,ref)           [expected detections under ref occupancy]

Category definitions (per posterior draw)
------------------------------------------
  NI-core missing    : ω̄_I < ε  AND  M_T ≥ m_T  AND  M_HG ≥ m_HG
  HG-associated      : ω̄_I < ε  AND  M_HG ≥ m_HG  AND  M_T = 0
  Transitional-assoc : ω̄_I < ε  AND  M_T ≥ m_T    AND  M_HG = 0
  Industrial-assoc   : ω̄_I > τ  AND  M_T = 0      AND  M_HG = 0  AND  Λ_I←NI is large
  Shared             : all three groups have ω̄ > τ
  Depleted-Ind       : ω̄_I < ω̄_NI - δ but ω̄_I ≥ ε
  Unresolved         : insufficient power or ambiguous

Usage
-----
    python three_group_occupancy.py --ref UHGG --n-boot 200
    python three_group_occupancy.py --ref UHGG --run-stan

Run (from pipeline root):
    python -m persistence.three_group_occupancy
"""

import argparse
import os
import sys
import warnings
import tempfile
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.special import betainc, ndtr
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

try:
    from .plot_utils import apply_display_text
except ImportError:
    def apply_display_text(fig):
        for artist in fig.findobj(match=matplotlib.text.Text):
            text = artist.get_text()
            for old in ("Agro-Pastorialist", "Transitional"):
                text = str(text).replace(old, "Agro-Pastoralist")
            artist.set_text(text)
        return fig

# ════════════════════════════════════════════════════════════════════════════
#  IMPORT FROM PIPELINE — same preprocessing, no duplication
# ════════════════════════════════════════════════════════════════════════════

def _import_pipeline():
    """
    Import shared functions from figure_summary.py.
    Tries the package import first, then a sibling-file import.
    Returns the module or raises ImportError with a helpful message.
    """
    try:
        from . import figure_summary as fs
        return fs
    except ImportError:
        pass
    # Sibling-file fallback: look in same directory as this script
    import importlib.util, pathlib
    candidates = [
        pathlib.Path(__file__).parent / "figure_summary.py",
        pathlib.Path("figure_summary.py"),
        pathlib.Path("persistence/figure_summary.py"),
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("figure_summary", p)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(
        "Cannot find figure_summary.py. Run from the pipeline root or "
        "pass --fs-path /path/to/figure_summary.py")

FS = None   # set in main() after CLI parsing

# ════════════════════════════════════════════════════════════════════════════
#  STYLE
# ════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "legend.fontsize": 9, "figure.dpi": 300,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})

COL_GROUP = {
    "Industrial":   "#56B4E9",
    "Transitional": "#E69F00",
    "HG":           "#009E73",
}
COL_CAT = {
    "NI-core":          "#2166AC",
    "HG-assoc":         "#009E73",
    "T-assoc":          "#E69F00",
    "Industrial-assoc": "#D55E00",
    "Shared":           "#9970AB",
    "Depleted-Ind":     "#80CDC1",
    "Unresolved":       "#CCCCCC",
}

# ════════════════════════════════════════════════════════════════════════════
#  POPULATION IDENTIFICATION
# ════════════════════════════════════════════════════════════════════════════

# Population column priority: Society/cohort-level identifiers first,
# country-level only as last resort. StudyFolder beats Country because
# a study cohort is more meaningful than a national border — especially
# for HG where multiple societies may share a country.
POPULATION_COLS = [
    "Population", "Detailed location", "StudyFolder",
    "Author", "Country_norm", "Country",
]
_COUNTRY_LEVEL_COLS = {"Country_norm", "Country"}  # flag these as fallbacks


def _infer_population(meta):
    """
    Assign a population label to each sample using the best available column.

    Returns (pop, col_used) so callers can diagnose when country-level fallbacks
    were used and warn accordingly.
    """
    pop      = pd.Series("unknown", index=meta.index, dtype=str)
    col_used = pd.Series("none",    index=meta.index, dtype=str)
    for col in reversed(POPULATION_COLS):
        if col not in meta.columns:
            continue
        v = meta[col].astype(str).str.strip()
        valid = ~v.isin(["", "nan", "None", "unknown"])
        pop      = pop.where(~valid, v)
        col_used = col_used.where(~valid, col)
    return pop, col_used


def build_three_group_blocks(meta, breadth_df, threshold=10.0):
    """
    Split samples into Industrial / Transitional / HG with per-population labels
    and diagnostic output showing which column was used for each group.
    """
    lifestyle  = meta["Lifestyle"]
    population, col_used = _infer_population(meta)

    group_defs = {
        "Industrial":   lifestyle == "Industrial",
        "Transitional": lifestyle == "Transitional",
        "HG":           lifestyle == "H-G",
    }
    blocks = {}; pop_map = {}; group_pops = {}

    for gname, mask in group_defs.items():
        ids = mask[mask].index.intersection(breadth_df.index)
        if len(ids) == 0:
            print(f"    WARNING: {gname} has 0 samples after intersection")
        blocks[gname]     = breadth_df.loc[ids]
        pop_map[gname]    = population.loc[ids]
        group_pops[gname] = sorted(population.loc[ids].unique())
        n_pops = len(group_pops[gname])
        # Show label source breakdown and warn if country-level is dominant
        cu = col_used.loc[ids]
        src_counts = cu.value_counts().to_dict()
        n_country  = cu.isin(_COUNTRY_LEVEL_COLS).sum()
        warn = (f"  ⚠ {n_country:,}/{len(ids):,} samples use country as "
                f"population label — societies may be conflated"
                if n_country > 0 else "")
        print(f"    {gname}: {len(ids):,} samples  {n_pops} populations  "
              f"[label sources: {src_counts}]{warn}")

    return blocks, pop_map, group_pops


# ════════════════════════════════════════════════════════════════════════════
#  PER-POPULATION BB MODEL + EQUAL-POPULATION OCCUPANCY
# ════════════════════════════════════════════════════════════════════════════

def estimate_population_occupancies(breadth_block, depth_block, lengths,
                                     total_bp, pop_series, threshold=10.0,
                                     min_samples_per_pop=3, rng=None):
    """
    Estimate per-population BB-corrected HT occupancy (ψ_k,p) for every
    (SGB, population) pair within one lifestyle group.

    Equal-population mean is then:
        ω̄_k,g = (1/P) Σ_p ψ_k,p

    Parameters
    ----------
    breadth_block : (n_samples, n_sgb) DataFrame
    depth_block   : (n_samples, n_sgb) DataFrame  — mean depth per base
    lengths       : pd.Series  SGB → genome length
    total_bp      : pd.Series  sample → total mapped bp
    pop_series    : pd.Series  sample → population label
    threshold     : float      breadth % threshold (default 10%)
    min_samples_per_pop : int  populations with fewer samples are skipped

    Returns
    -------
    psi_pop   : dict  population → (n_sgb,) ψ̂_k,p array
    r_pop     : dict  population → (n_samples_p, n_sgb) r_mat
    omega_bar : (n_sgb,) equal-population mean occupancy
    omega_se  : (n_sgb,) between-population SE
    pops_used : list of population labels that contributed
    n_sgb     : int
    """
    if rng is None:
        rng = np.random.default_rng(42)

    sgbs = list(breadth_block.columns)
    n_sgb = len(sgbs)

    # Align indices
    shared = (breadth_block.index
              .intersection(depth_block.index)
              .intersection(total_bp.index)
              .intersection(pop_series.index))
    b_full   = breadth_block.loc[shared]
    d_full   = depth_block.loc[shared]
    tbp_full = total_bp.loc[shared]
    pop_full = pop_series.loc[shared]

    unique_pops = sorted(pop_full.unique())
    psi_pop    = {}
    r_pop      = {}
    pops_used  = []

    for pop in unique_pops:
        mask = pop_full == pop
        n_p  = int(mask.sum())
        if n_p < min_samples_per_pop:
            continue
        b_p   = b_full.loc[mask]
        d_p   = d_full.loc[mask]
        tbp_p = tbp_full.loc[mask]

        # Use the BB model from figure_summary
        bb = FS.estimate_group_model_bb(b_p, d_p, lengths, tbp_p, threshold)
        h_hat_p, psi_p, mu_p, sig_p, r_mat_p = bb[:5]

        psi_pop[pop] = psi_p          # (n_sgb,)
        r_pop[pop]   = r_mat_p        # (n_p, n_sgb)
        pops_used.append(pop)

    n_pops = len(pops_used)
    if n_pops == 0:
        return {}, {}, np.zeros(n_sgb), np.zeros(n_sgb), [], n_sgb

    # Equal-population mean and SE
    psi_mat   = np.stack([psi_pop[p] for p in pops_used], axis=0)   # (P, K)
    omega_bar = psi_mat.mean(0)
    omega_se  = psi_mat.std(0, ddof=min(1, n_pops-1)) / np.sqrt(n_pops) \
                if n_pops > 1 else np.zeros(n_sgb)

    return psi_pop, r_pop, omega_bar, omega_se, pops_used, n_sgb


# ════════════════════════════════════════════════════════════════════════════
#  REPLICATION COUNT  M_k,g(τ)
# ════════════════════════════════════════════════════════════════════════════

def replication_count(psi_pop, pops_used, tau=0.10):
    """
    M_k,g(τ) = number of populations where ψ_k,p > τ.

    Returns (n_sgb,) int array.
    """
    if not pops_used:
        return np.zeros(1, int)
    psi_mat = np.stack([psi_pop[p] for p in pops_used], axis=0)   # (P, K)
    return (psi_mat > tau).sum(0)                                   # (K,)


# ════════════════════════════════════════════════════════════════════════════
#  POWER METRIC  Λ_k,g←ref
# ════════════════════════════════════════════════════════════════════════════

def detection_power_lambda(r_pop_target, pops_target, omega_ref_k):
    """
    Expected detection count in the target group if an SGB had occupancy ω_ref:
        Λ_k,g←ref = Σ_{s∈g} h_ks(ω_ref_k) = Σ_{s∈g} ω_ref_k × r_ks

    Parameters
    ----------
    r_pop_target : dict  population → (n_samples_p, n_sgb) r_mat
    pops_target  : list  populations to sum over
    omega_ref_k  : (n_sgb,) reference occupancy vector

    Returns (n_sgb,) Λ array.
    """
    n_sgb = len(omega_ref_k)
    lam   = np.zeros(n_sgb, float)
    for pop in pops_target:
        if pop not in r_pop_target:
            continue
        r_mat_p = r_pop_target[pop]           # (n_p, n_sgb)
        lam    += (omega_ref_k[None, :] * r_mat_p).sum(0)
    return lam


# ════════════════════════════════════════════════════════════════════════════
#  THREE-GROUP CATEGORY CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════

CATEGORIES = [
    "NI-core",           # absent Industrial, present in ≥m_T Transitional AND ≥m_HG HG pops
    "HG-assoc",          # absent Industrial, present in ≥m_HG HG pops, absent Transitional
    "T-assoc",           # absent Industrial, present in ≥m_T Transitional pops, absent HG
    "Industrial-assoc",  # present Industrial, absent Transitional AND HG (with power)
    "Shared",            # all three groups present
    "Depleted-Ind",      # Industrial lower than NI but not near-absent
    "Unresolved",        # insufficient power or ambiguous pattern
]


def classify_three_group(omega_bar_I, omega_bar_T, omega_bar_HG,
                          M_I, M_T, M_HG,
                          lam_I_from_NI, lam_T_from_I, lam_HG_from_I,
                          eps=0.05, tau=0.10, delta=0.10,
                          m_T=2, m_HG=2,
                          lam_min=3.0):
    """
    Classify each SGB into one of seven biological categories.

    All inputs are (n_sgb,) arrays.

    Parameters
    ----------
    omega_bar_I/T/HG  : equal-population mean occupancy per group
    M_I/T/HG          : replication count (number of populations with ω > tau)
    lam_I_from_NI     : Λ_k,Ind←NI — expected detections in Ind under NI occupancy
    lam_T_from_I      : Λ_k,T←Ind — expected detections in T under Ind occupancy
    lam_HG_from_I     : Λ_k,HG←Ind — expected detections in HG under Ind occupancy
    eps               : near-absent threshold (default 0.05)
    tau               : meaningfully-present threshold (default 0.10)
    delta             : depletion threshold for Depleted-Ind (default 0.10)
    m_T, m_HG         : minimum population replication for category calls
    lam_min            : minimum expected detection count for power adequacy

    Returns
    -------
    categories : (n_sgb,) array of category strings
    power_ok_I : (n_sgb,) bool — Industrial has adequate power to detect NI-level SGBs
    """
    n = len(omega_bar_I)
    cats = np.full(n, "Unresolved", dtype=object)

    absent_I    = omega_bar_I < eps
    present_I   = omega_bar_I > tau
    present_T   = (omega_bar_T > tau) & (M_T >= m_T)
    present_HG  = (omega_bar_HG > tau) & (M_HG >= m_HG)
    absent_T    = (omega_bar_T < eps) | (M_T == 0)
    absent_HG   = (omega_bar_HG < eps) | (M_HG == 0)
    power_ok_I  = lam_I_from_NI >= lam_min     # can we trust an Industrial absence call?
    power_ok_T  = lam_T_from_I  >= lam_min
    power_ok_HG = lam_HG_from_I >= lam_min
    depleted_I  = (~absent_I) & (omega_bar_I < omega_bar_T - delta) \
                  & (omega_bar_I < omega_bar_HG - delta)

    # Shared: all three groups meaningfully present
    cats[present_I & present_T & present_HG] = "Shared"

    # Depleted in Industrial (present but substantially lower)
    cats[depleted_I & ~present_I] = "Depleted-Ind"

    # Industrial-associated: present in Industrial, absent in both NI groups
    # (require power to detect in T and HG if Industrial occupancy were absent)
    cats[present_I & absent_T & absent_HG & power_ok_T & power_ok_HG] = "Industrial-assoc"

    # Missing in Industrial (require power in Industrial direction)
    # NI-core: absent Industrial, present in both T and HG
    cats[absent_I & present_T & present_HG & power_ok_I] = "NI-core"

    # HG-associated: absent Industrial AND Transitional, present HG only
    cats[absent_I & absent_T & present_HG & power_ok_I] = "HG-assoc"

    # Transitional-associated: absent Industrial AND HG, present T only
    cats[absent_I & present_T & absent_HG & power_ok_I] = "T-assoc"

    # Everything else is Unresolved (ambiguous or low power)
    return cats, power_ok_I


# ════════════════════════════════════════════════════════════════════════════
#  CLUSTER BOOTSTRAP
# ════════════════════════════════════════════════════════════════════════════

def cluster_bootstrap_omega(breadth_block, depth_block, lengths,
                             total_bp, pop_series, threshold=10.0,
                             n_boot=200, min_samples_per_pop=3,
                             tau_for_M=0.10, rng=None):
    """
    Cluster bootstrap for equal-population mean occupancy ω̄_k,g.

    Fixes vs previous version:
    1. Bootstrap numerator uses the same sub-threshold augmented HT estimator
       as the BB point estimator (not strict threshold detections only).
       This ensures posterior p_cat is consistent with the point estimate.
    2. Population replication count M is bootstrapped properly by computing
       M_b = (psi_pop_b > tau).sum() from population-level ψ draws, not
       approximated from the group mean.

    Each replicate:
      1. Resample populations with replacement (inter-population variance)
      2. Within each selected population, resample samples with replacement
      3. Recompute ψ_k,p using BB-weighted HT numerator (n_full + w_sub)
      4. ω̄_k,g = mean of resampled population ψ estimates
      5. M_k,g = count of populations where ψ_k,p > tau

    Returns
    -------
    omega_boots : (n_boot, n_sgb) bootstrap ω̄
    M_boots     : (n_boot, n_sgb) bootstrap M (population replication count)
    pops_used   : list of population labels
    """
    if rng is None:
        rng = np.random.default_rng(42)

    shared = (breadth_block.index
              .intersection(depth_block.index)
              .intersection(total_bp.index)
              .intersection(pop_series.index))
    b_all   = breadth_block.loc[shared]
    d_all   = depth_block.loc[shared]
    tbp_all = total_bp.loc[shared]
    pop_all = pop_series.loc[shared]

    unique_pops = [p for p in sorted(pop_all.unique())
                   if (pop_all == p).sum() >= min_samples_per_pop]
    n_pops = len(unique_pops)
    n_sgb  = b_all.shape[1]

    if n_pops == 0:
        return np.zeros((n_boot, n_sgb)), np.zeros((n_boot, n_sgb), int), []

    # Pre-compute per-population BB model once (expensive step)
    print(f"      Pre-computing BB models for {n_pops} populations...",
          end=" ", flush=True)
    pop_psi     = {}   # pop → (n_sgb,) ψ̂ (BB point estimate)
    pop_r       = {}   # pop → (n_p, n_sgb) r_mat
    pop_num     = {}   # pop → (n_p, n_sgb) HT numerator (n_full + w_sub weighted)
    pop_nsamples = {}  # pop → n_p

    eps_pct = 0.5   # sub-threshold floor — must match estimate_group_model_bb default

    for pop in unique_pops:
        mask = pop_all == pop
        sids = mask[mask].index
        b_p  = b_all.loc[sids]
        d_p  = d_all.loc[sids]
        t_p  = tbp_all.loc[sids]
        bb   = FS.estimate_group_model_bb(b_p, d_p, lengths, t_p, threshold)
        psi_p, r_mat_p = bb[1], bb[4]    # psi_ht, r_mat

        # Reconstruct per-sample HT numerator from BB model outputs.
        # The BB model's psi_ht = (n_full + n_sub_wt) / sum_r.
        # We store the per-sample numerator contributions so bootstrap
        # resampling preserves the same weighting as the point estimate.
        # Per-sample numerator: matches BB point estimator's 0.90 sub-threshold weight
        b_vals   = b_p.values.astype(float)
        det_full = (b_vals >= threshold).astype(float)
        det_eps  = (b_vals >= eps_pct).astype(float)
        sub      = det_eps - det_full
        num_p    = det_full + 0.90 * sub    # (n_p, n_sgb)

        pop_psi[pop]      = psi_p
        pop_r[pop]        = r_mat_p
        pop_num[pop]      = num_p
        pop_nsamples[pop] = len(sids)
    print("done")

    omega_boots = np.zeros((n_boot, n_sgb), float)
    M_boots     = np.zeros((n_boot, n_sgb), int)

    for b_i in range(n_boot):
        # Level 1: resample populations with replacement
        pops_b = rng.choice(unique_pops, size=n_pops, replace=True)

        psi_b_list = []
        for pop in pops_b:
            n_p   = pop_nsamples[pop]
            s_idx = rng.integers(0, n_p, n_p)

            r_b   = pop_r[pop][s_idx]      # (n_p, n_sgb)
            num_b = pop_num[pop][s_idx]    # (n_p, n_sgb) — BB-weighted numerator

            sum_r  = r_b.sum(0)
            sum_num = num_b.sum(0)
            psi_bp = np.clip(
                np.where(sum_r > 0.1, sum_num / sum_r, 0.0), 0.0, 1.0)
            psi_b_list.append(psi_bp)

        psi_mat_b = np.stack(psi_b_list, axis=0)          # (n_pops, n_sgb)
        omega_boots[b_i] = psi_mat_b.mean(0)
        # M_b computed from population-level psi draws, using runtime tau
        M_boots[b_i] = (psi_mat_b > tau_for_M).sum(0)

    return omega_boots, M_boots, unique_pops


def leave_one_population_out(psi_pop, pops_used):
    """
    Leave-one-population-out (LOPO) sensitivity analysis.

    For each population p, compute ω̄_k,g excluding p.
    Returns an array of shape (n_pops, n_sgb) — the group mean when
    each population is removed.

    An SGB is LOPO-robust if its category assignment does not change
    when any single population is excluded. Used to flag categories
    that are driven by a single cohort (e.g., Hadza alone).
    """
    if not pops_used:
        return np.zeros((0, 1))
    psi_mat = np.stack([psi_pop[p] for p in pops_used], axis=0)   # (P, K)
    P = len(pops_used)
    lopo = np.zeros((P, psi_mat.shape[1]), float)
    for i in range(P):
        mask = np.ones(P, bool); mask[i] = False
        lopo[i] = psi_mat[mask].mean(0) if mask.sum() > 0 else np.zeros(psi_mat.shape[1])
    return lopo   # (P, K)


# ════════════════════════════════════════════════════════════════════════════
#  BAYESIAN FDR CONTROL
# ════════════════════════════════════════════════════════════════════════════

def bfdr_threshold(p_cat, target_bfdr=0.05):
    """
    Find the posterior probability cutoff that controls Bayesian FDR ≤ target.

    BFDR(S) = Σ_{k∈S} (1 - p_k) / |S|
    """
    thresholds = np.sort(np.unique(p_cat))[::-1]
    for thr in thresholds:
        sel  = p_cat >= thr
        n_s  = sel.sum()
        if n_s == 0:
            continue
        bfdr = (1.0 - p_cat[sel]).sum() / n_s
        if bfdr <= target_bfdr + 1e-9:
            E_N = p_cat[sel].sum()
            return float(thr), sel, float(bfdr), float(E_N)
    sel = p_cat >= thresholds[0]
    return float(thresholds[0]), sel, float((1-p_cat[sel]).sum()/max(sel.sum(),1)), float(p_cat[sel].sum())


def _sample_bootstrap_omega(breadth_block, depth_block, lengths,
                             total_bp, threshold, n_boot, rng):
    """
    Sample-level bootstrap for groups too small for cluster bootstrap (HG).
    Returns (n_boot, n_sgb) omega_boots.
    """
    shared = (breadth_block.index
              .intersection(depth_block.index)
              .intersection(total_bp.index))
    b_full   = breadth_block.loc[shared]
    d_full   = depth_block.loc[shared]
    tbp_full = total_bp.loc[shared]

    bb      = FS.estimate_group_model_bb(b_full, d_full, lengths, tbp_full, threshold)
    psi_ht  = bb[1]      # (n_sgb,) point estimate
    r_mat   = bb[4]      # (n_s, n_sgb)
    det_full = (b_full.values >= threshold).astype(float)
    n_s, n_k = det_full.shape

    omega_boots = np.zeros((n_boot, n_k), float)
    for b_i in range(n_boot):
        idx   = rng.integers(0, n_s, n_s)
        sum_r = r_mat[idx].sum(0)
        num   = det_full[idx].sum(0)
        omega_boots[b_i] = np.clip(
            np.where(sum_r > 0.1, num / sum_r, 0.0), 0.0, 1.0)
    return omega_boots


# ── Simplified category set (I vs T primary) ─────────────────────────────

CATEGORIES_SIMPLE = [
    "I-near-absent T-present",   # ω̄_I < ε, ω̄_T > τ, M_T ≥ m_T — NOT "absent", just near-absent at group level
    "I-near-absent T-absent",    # ω̄_I < ε, ω̄_T < ε
    "Industrial-assoc",          # ω̄_I > τ, M_I ≥ 1, absent Transitional
    "Shared",                    # ω̄_I > τ, M_I ≥ 1, ω̄_T > τ, M_T ≥ m_T
    "Depleted-Ind",              # ε ≤ ω̄_I, but substantially below Transitional
    "Unresolved",                # insufficient power or ambiguous
]

COL_CAT_SIMPLE = {
    "I-near-absent T-present": "#2166AC",
    "I-near-absent T-absent":  "#92C5DE",
    "Industrial-assoc":        "#D55E00",
    "Shared":                  "#9970AB",
    "Depleted-Ind":            "#80CDC1",
    "Unresolved":              "#CCCCCC",
}

HG_FLAGS = ["HG-enriched", "HG-absent", "HG-uncertain"]


# ════════════════════════════════════════════════════════════════════════════
#  HARD OBSERVATION FLAGS  (independent of occupancy thresholds)
# ════════════════════════════════════════════════════════════════════════════

def observed_in_group(breadth_block, pop_series, threshold=10.0,
                       kb_block=None, kb_thr=300_000):
    """
    Hard observation flags: was SGB k detected in ANY sample / how many
    populations had at least one detection?

    Uses the FULL detection rule (breadth ≥ threshold OR covered kb ≥ kb_thr)
    and operates on ALL samples before any population-size filtering, so no
    detection is lost due to small-population exclusion from the occupancy step.

    Parameters
    ----------
    breadth_block : (n_samples, n_sgb) DataFrame  — % breadth
    pop_series    : pd.Series  sample → population label
    threshold     : float      breadth % threshold (default 10%)
    kb_block      : (n_samples, n_sgb) DataFrame  — covered kb per SGB
                    If provided, satisfying kb ≥ kb_thr also counts as detected.
    kb_thr        : float      covered-kb threshold (default 200 kbp)

    Returns
    -------
    obs_sample : (n_sgb,) bool — detected in ≥1 sample
    obs_pop    : (n_sgb,) int  — number of populations with ≥1 detection
    """
    idx = breadth_block.index.intersection(pop_series.index)
    b   = breadth_block.loc[idx].values.astype(float)

    # Primary detection: breadth ≥ threshold
    det = b >= threshold

    # Secondary arm: covered kb ≥ kb_thr (if data available)
    if kb_block is not None:
        kb_idx = kb_block.index.intersection(idx)
        if len(kb_idx):
            kb_aligned = kb_block.reindex(index=idx, columns=breadth_block.columns).fillna(0.0)
            det |= (kb_aligned.values >= kb_thr)

    obs_sample = det.any(axis=0)   # (n_sgb,) bool

    # Per-population observation count
    pop_aligned = pop_series.loc[idx]
    obs_pop = np.zeros(det.shape[1], dtype=int)
    for pop in sorted(pop_aligned.unique()):
        mask = (pop_aligned == pop).values
        if mask.sum() == 0:
            continue
        obs_pop += det[mask].any(axis=0).astype(int)

    return obs_sample, obs_pop


def classify_two_group_with_hg_annotation(
        omega_I, omega_T, omega_HG, M_I, M_T,
        lam_I_from_T, lam_T_from_I,
        obs_I_pop=None, obs_T_pop=None,
        eps=0.05, tau=0.10, delta=0.10, m_I=1, m_T=2, lam_min=3.0):
    """
    Primary classification on Industrial vs Transitional axis.
    HG occupancy is an independent point-estimate annotation.

    **Observation vs occupancy are kept strictly separate.**
    **Categories describe group-level occupancy, not individual observations.**
    **"near-absent" means ω̄ < ε at the equal-population level — NOT "never seen".**

    Occupancy categories
    --------------------
    Shared              : ω̄_I > τ AND M_I ≥ m_I AND ω̄_T > τ AND M_T ≥ m_T
    Depleted-Ind        : ω̄_I between eps and tau, OR ω̄_I > tau but << ω̄_T - delta
                          (symmetric: captures both ε≤ω̄_I<τ and ω̄_I>τ but depleted)
    I-near-absent T-present: ω̄_I < ε AND ω̄_T > τ AND M_T ≥ m_T AND Λ_I←T ok
    I-near-absent T-absent : ω̄_I < ε AND ω̄_T < ε AND Λ_I←T ok
    Industrial-assoc    : ω̄_I > τ AND M_I ≥ m_I AND absent_T AND power_T
                          (power check: can Transitional detect Industrial-level?)
    Unresolved          : insufficient power or ambiguous

    Observation label (independent, printed separately)
    ---------------------------------------------------
    seen_I_pop{n}           : obs_I_pop = n (seen in n Industrial populations)
    seen_I_unresolved       : obs_I_pop ≥ 1 but Unresolved by occupancy model
    not_seen_I              : obs_I_pop = 0

    Returns cats, hg_flag, obs_label, power_I, power_T
    """
    n = len(omega_I)
    cats    = np.full(n, "Unresolved", dtype=object)
    hg_flag = np.full(n, "HG-uncertain", dtype=object)

    power_I = lam_I_from_T >= lam_min
    power_T = lam_T_from_I >= lam_min

    # Occupancy membership — symmetric replication requirements
    near_absent_I = omega_I < eps
    near_absent_T = (omega_T < eps) | (M_T == 0)
    present_I = (omega_I > tau) & (M_I >= m_I)   # now symmetric with present_T
    present_T = (omega_T > tau) & (M_T >= m_T)

    # Depleted: any Industrial that is substantially below Transitional.
    # Covers BOTH the ε≤ω_I<τ range AND ω_I>τ-but-much-lower-than-T.
    depleted = (~near_absent_I) & (omega_I < omega_T - delta)

    # Assign (most specific wins; order matters)
    cats[present_I  & present_T]                         = "Shared"
    cats[depleted   & ~present_I]                        = "Depleted-Ind"
    cats[near_absent_I & present_T & power_I]            = "I-near-absent T-present"
    cats[near_absent_I & near_absent_T & power_I]        = "I-near-absent T-absent"
    cats[present_I  & near_absent_T & power_T]           = "Industrial-assoc"

    # HG annotation (point estimate only)
    hg_flag[omega_HG > tau] = "HG-enriched"
    hg_flag[omega_HG < eps] = "HG-absent"

    # Observation label: entirely independent of occupancy model
    if obs_I_pop is not None:
        obs_label = np.where(obs_I_pop >= 1, "seen_I", "not_seen_I").astype(object)
        # Further split Unresolved so diagnostics are immediate
        obs_label[(obs_I_pop >= 1) & (cats == "Unresolved")] = "seen_I_unresolved"
        obs_label[(obs_I_pop == 0) & (cats == "Unresolved")] = "not_seen_I_unresolved"
    else:
        obs_label = np.full(n, "unknown", dtype=object)

    return cats, hg_flag, obs_label, power_I, power_T


# ════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS  (simplified: I vs T with HG annotation)
# ════════════════════════════════════════════════════════════════════════════

def run_three_group_analysis(rdat, threshold=10.0, ref_name="UHGG",
                              n_boot=200, min_pop=3, tau=0.10, eps=0.05,
                              delta=0.10, m_T=2, lam_min=3.0,
                              target_bfdr=0.05, rng=None):
    """
    Simplified three-group occupancy analysis.

    Primary axis  : Industrial vs Transitional — cluster bootstrap over ≥22
                    populations each. These have enough population-level
                    replication for cluster resampling to be valid.

    HG / Hadza    : Point estimate only, used as annotation on top of the
                    I vs T result. With only 4 HG populations (predominantly
                    Hadza), cluster bootstrap over HG is degenerate. A
                    sample-level bootstrap is run solely to characterise
                    ω̄_HG uncertainty — it does NOT drive category calls.

    HG enrichment flag: for each BFDR-selected SGB, report whether it is
                    present in HG (ω̄_HG > τ) as a secondary descriptor.
                    "I-absent T-present AND HG-enriched" is the strongest
                    evidence for a microbe that has been lost in Industrial
                    populations across lifestyle gradients.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    meta     = rdat["meta"]
    breadth  = rdat["breadth"]
    depth    = rdat["depth"]
    lengths  = rdat["lengths"]
    total_bp = rdat["total_bp"]

    print(f"\n{'='*70}")
    print(f"  THREE-GROUP ANALYSIS (simplified)  [{ref_name}]")
    print(f"  Primary: Industrial vs Transitional (cluster bootstrap)")
    print(f"  HG/Hadza: point estimate annotation only")
    print(f"{'='*70}")

    print("\n[1/6] Building three-group blocks...")
    blocks, pop_map, group_pops = build_three_group_blocks(meta, breadth, threshold)
    ref_sgbs = sorted(set(breadth.columns))
    n_sgb    = len(ref_sgbs)

    # ── Hard observation flags BEFORE any population dropping ────────────
    # These are computed over ALL samples so no detection is lost due to
    # small-population exclusion from the occupancy model.
    print("\n  Computing hard observation flags (all samples, before population filter)...")
    obs_I_sample, obs_I_pop = observed_in_group(
        blocks.get("Industrial",   pd.DataFrame()),
        pop_map.get("Industrial",  pd.Series(dtype=str)),
        threshold=threshold)
    obs_T_sample, obs_T_pop = observed_in_group(
        blocks.get("Transitional", pd.DataFrame()),
        pop_map.get("Transitional", pd.Series(dtype=str)),
        threshold=threshold)
    obs_HG_sample, obs_HG_pop = observed_in_group(
        blocks.get("HG",           pd.DataFrame()),
        pop_map.get("HG",          pd.Series(dtype=str)),
        threshold=threshold)
    # obs_I_pop counts POPULATIONS (not samples) with ≥1 detection
    print(f"  Ind: {obs_I_sample.sum():,} SGBs detected in ≥1 Industrial sample  "
          f"({(obs_I_pop >= 2).sum():,} detected in ≥2 Industrial populations)")
    print(f"  Tra: {obs_T_sample.sum():,} SGBs detected in ≥1 Transitional sample  "
          f"({(obs_T_pop >= 2).sum():,} detected in ≥2 Transitional populations)")
    print(f"  HG:  {obs_HG_sample.sum():,} SGBs detected in ≥1 HG sample  "
          f"({(obs_HG_pop >= 2).sum():,} detected in ≥2 HG populations)")
    # ── Industrial + Transitional: per-population BB ─────────────────────
    print("\n[2/6] Estimating per-population occupancies (Industrial + Transitional)...")
    group_results = {}
    for gname in ["Industrial", "Transitional"]:
        block = blocks.get(gname, pd.DataFrame())
        if len(block) == 0:
            print(f"  {gname}: SKIPPED")
            group_results[gname] = dict(psi_pop={}, r_pop={},
                omega_bar=np.zeros(n_sgb), omega_se=np.zeros(n_sgb),
                pops_used=[], n_sgb=n_sgb, block=block)
            continue
        print(f"  {gname} ({len(block):,} samples)...", end=" ", flush=True)
        d_block   = depth.loc[block.index]
        tbp_block = total_bp.loc[block.index]
        psi_pop, r_pop, omega_bar, omega_se, pops_used, _ = \
            estimate_population_occupancies(
                block, d_block, lengths.reindex(ref_sgbs),
                tbp_block, pop_map[gname], threshold=threshold,
                min_samples_per_pop=min_pop, rng=rng)
        print(f"{len(pops_used)} pops  median_ω̄={np.nanmedian(omega_bar):.4f}")
        group_results[gname] = dict(
            psi_pop=psi_pop, r_pop=r_pop, omega_bar=omega_bar,
            omega_se=omega_se, pops_used=pops_used, n_sgb=n_sgb,
            block=block, d_block=d_block, tbp_block=tbp_block)

    # ── HG: equal-population mean + sample-level CI only ──────────────
    hg_block  = blocks.get("HG", pd.DataFrame())
    n_hg_pops = len(group_pops.get("HG", []))
    print(f"\n  HG ({len(hg_block):,} samples, {n_hg_pops} populations) — "
          f"equal-pop mean + sample bootstrap CI...", end=" ", flush=True)
    if len(hg_block) > 0:
        hg_d   = depth.loc[hg_block.index]
        hg_tbp = total_bp.loc[hg_block.index]
        # Equal-population mean over HG populations (consistent with omega_I/omega_T)
        hg_psi_pop, _, hg_omega_bar, _, hg_pops_used, _ = \
            estimate_population_occupancies(
                hg_block, hg_d, lengths.reindex(ref_sgbs), hg_tbp,
                pop_map["HG"], threshold=threshold,
                min_samples_per_pop=min_pop, rng=rng)
        omega_HG = hg_omega_bar if len(hg_pops_used) > 0 else np.zeros(n_sgb)
        print(f"{len(hg_pops_used)} pops  median_ω̄_HG={np.nanmedian(omega_HG):.4f}")
        print(f"  HG populations used: {hg_pops_used}")
        # Sample bootstrap for CI only (not for category calls)
        hg_boots = _sample_bootstrap_omega(
            hg_block, hg_d, lengths.reindex(ref_sgbs),
            hg_tbp, threshold, n_boot=min(n_boot, 100), rng=rng)
        omega_HG_ci = np.percentile(hg_boots, [2.5, 97.5], axis=0)
    else:
        omega_HG = np.zeros(n_sgb)
        omega_HG_ci = np.zeros((2, n_sgb))
        hg_boots = np.zeros((1, n_sgb))
        hg_psi_pop = {}; hg_pops_used = []
        print("SKIPPED")

    omega_I = group_results["Industrial"]["omega_bar"]
    omega_T = group_results["Transitional"]["omega_bar"]

    # obs_I_pop_used / obs_T_pop_used: detection count restricted to populations
    # that made it into the occupancy model — enables the 2x2 diagnostic table.
    pops_used_I = set(group_results["Industrial"].get("pops_used", []))
    pops_used_T = set(group_results["Transitional"].get("pops_used", []))

    def _obs_used(block, pop_ser, pops_used_set, thr):
        if len(block) == 0: return np.zeros(n_sgb, int)
        idx = block.index.intersection(pop_ser.index)
        b   = block.loc[idx].values.astype(float)
        det = b >= thr
        pop_a = pop_ser.loc[idx]
        obs = np.zeros(det.shape[1], int)
        for pop in sorted(pops_used_set):
            mask = (pop_a == pop).values
            if mask.sum(): obs += det[mask].any(0).astype(int)
        return obs

    obs_I_pop_used = _obs_used(
        blocks.get("Industrial", pd.DataFrame()),
        pop_map.get("Industrial", pd.Series(dtype=str)), pops_used_I, threshold)
    obs_T_pop_used = _obs_used(
        blocks.get("Transitional", pd.DataFrame()),
        pop_map.get("Transitional", pd.Series(dtype=str)), pops_used_T, threshold)


    # ── Power metrics ────────────────────────────────────────────────────
    print("\n[3/6] Power metrics (Industrial ↔ Transitional)...")
    M_I = replication_count(group_results["Industrial"]["psi_pop"],
                             group_results["Industrial"]["pops_used"], tau=tau)
    M_T = replication_count(group_results["Transitional"]["psi_pop"],
                             group_results["Transitional"]["pops_used"], tau=tau)

    lam_I_from_T = detection_power_lambda(
        group_results["Industrial"]["r_pop"],
        group_results["Industrial"]["pops_used"], omega_T)
    lam_T_from_I = detection_power_lambda(
        group_results["Transitional"]["r_pop"],
        group_results["Transitional"]["pops_used"], omega_I)

    print(f"  Ind M: median={np.median(M_I):.0f}  max={M_I.max()}")
    print(f"  Tra M: median={np.median(M_T):.0f}  max={M_T.max()}")
    print(f"  Λ_I←T: median={np.median(lam_I_from_T):.1f}  "
          f"frac≥{lam_min}={np.mean(lam_I_from_T >= lam_min):.1%}")
    print(f"  Λ_T←I: median={np.median(lam_T_from_I):.1f}  "
          f"frac≥{lam_min}={np.mean(lam_T_from_I >= lam_min):.1%}")

    # ── Point-estimate classification ────────────────────────────────────
    print("\n[4/6] Classifying SGBs...")
    cats_point, hg_flag, obs_label, power_I, power_T = \
        classify_two_group_with_hg_annotation(
            omega_I, omega_T, omega_HG, M_I, M_T,
            lam_I_from_T, lam_T_from_I,
            obs_I_pop=obs_I_pop, obs_T_pop=obs_T_pop,
            eps=eps, tau=tau, delta=delta, m_I=1, m_T=m_T, lam_min=lam_min)

    cat_counts = {c: int((cats_point == c).sum()) for c in CATEGORIES_SIMPLE}
    obs_counts = {v: int((obs_label == v).sum()) for v in np.unique(obs_label)}

    for c, n_c in cat_counts.items():
        mask_c = cats_point == c
        n_hg        = int((hg_flag[mask_c] == "HG-enriched").sum()) if mask_c.any() else 0
        n_seen_all  = int((obs_I_pop[mask_c]      >= 1).sum()) if mask_c.any() else 0
        n_seen_used = int((obs_I_pop_used[mask_c]  >= 1).sum()) if mask_c.any() else 0
        suffix = ""
        if n_hg > 0:
            suffix += f"  {n_hg} HG-enriched"
        if n_seen_all > 0:
            # Show both: all populations and populations used in occupancy model
            suffix += (f"  seen in ≥1 Ind pop: {n_seen_all} (all) / "
                       f"{n_seen_used} (used in model)")
        print(f"  {c:<30}: {n_c:>5,}{suffix}")

    print(f"\n  Observation label counts:")
    for lbl, cnt in sorted(obs_counts.items()):
        print(f"    {lbl:<35}: {cnt:,}")

    # 2×2 diagnostic: obs_I_pop_used ≥ 1 × omega_I < eps
    # This is the single most useful table for diagnosing estimator vs data mismatch.
    seen_used   = obs_I_pop_used >= 1
    near_abs_I  = omega_I < eps
    n_11 = int(( seen_used &  near_abs_I).sum())  # seen in used pops, near-absent
    n_10 = int(( seen_used & ~near_abs_I).sum())  # seen in used pops, NOT near-absent
    n_01 = int((~seen_used &  near_abs_I).sum())  # not seen, near-absent
    n_00 = int((~seen_used & ~near_abs_I).sum())  # not seen, not near-absent
    print(f"\n  ── 2×2 diagnostic (obs_I_pop_used ≥1 × ω̄_I < ε={eps}) ──")
    print(f"                    ω̄_I ≥ ε    ω̄_I < ε")
    print(f"  seen in used pops: {n_10:>6,}    {n_11:>6,}  ← seen but near-absent: data/model mismatch")
    print(f"  not seen:          {n_00:>6,}    {n_01:>6,}")
    if n_11 > 0:
        print(f"\n  ⚠  {n_11:,} SGBs observed in ≥1 used Industrial population "
              f"but ω̄_I < ε = {eps}")
        print(f"     These are near-absent at GROUP level — present in data, low group-level occupancy")
        print(f"     Causes: rare/confined-to-one-population, BB model underestimation")

    # ── Cluster bootstrap: Industrial + Transitional only ────────────────
    print(f"\n[5/6] Cluster bootstrap I+T (n_boot={n_boot}, HG fixed)...")
    boot_cats = {c: np.zeros(n_sgb, float) for c in CATEGORIES_SIMPLE}
    omega_boots = {}
    M_boots_dict = {}

    for gname in ["Industrial", "Transitional"]:
        gr = group_results[gname]
        if len(gr.get("pops_used", [])) < 2:
            omega_boots[gname]   = np.tile(gr["omega_bar"], (n_boot, 1))
            M_boots_dict[gname]  = np.tile(
                replication_count(gr["psi_pop"], gr["pops_used"], tau=tau),
                (n_boot, 1))
            continue
        print(f"  {gname}...", end=" ", flush=True)
        ob, mb, _ = cluster_bootstrap_omega(
            gr["block"], gr["d_block"], lengths.reindex(ref_sgbs),
            gr["tbp_block"], pop_map[gname],
            threshold=threshold, n_boot=n_boot,
            min_samples_per_pop=min_pop, tau_for_M=tau, rng=rng)
        omega_boots[gname]   = ob
        M_boots_dict[gname]  = mb
        print("done")

    print("  Classifying bootstrap draws...")
    for b_i in range(n_boot):
        oI_b  = omega_boots["Industrial"][b_i]
        oT_b  = omega_boots["Transitional"][b_i]
        # Use properly bootstrapped M from population-level psi draws
        MI_b  = M_boots_dict["Industrial"][b_i]
        MT_b  = M_boots_dict["Transitional"][b_i]
        # HG is fixed (not resampled); obs flags fixed at point estimate
        cats_b, _, _, _, _ = classify_two_group_with_hg_annotation(
            oI_b, oT_b, omega_HG,
            MI_b, MT_b, lam_I_from_T, lam_T_from_I,
            obs_I_pop=obs_I_pop, obs_T_pop=obs_T_pop,
            eps=eps, tau=tau, delta=delta, m_T=m_T, lam_min=lam_min)
        for c in CATEGORIES_SIMPLE:
            boot_cats[c] += (cats_b == c).astype(float)

    p_cat = {c: boot_cats[c] / n_boot for c in CATEGORIES_SIMPLE}
    E_N   = {c: float(p_cat[c].sum()) for c in CATEGORIES_SIMPLE}
    omega_I_ci = np.percentile(omega_boots["Industrial"],   [2.5, 97.5], axis=0)
    omega_T_ci = np.percentile(omega_boots["Transitional"], [2.5, 97.5], axis=0)

    # Bootstrap variance diagnostic: fraction of SGBs where omega_boots actually varies
    # (if this is ~0 the bootstrap is not providing meaningful uncertainty)
    om_I_sd = omega_boots["Industrial"].std(0)
    om_T_sd = omega_boots["Transitional"].std(0)
    print(f"  Bootstrap SD: Ind median={np.median(om_I_sd):.4f}  "
          f"Tra median={np.median(om_T_sd):.4f}")
    print(f"  (SGBs with SD=0 in Industrial: "
          f"{(om_I_sd < 1e-6).sum():,}/{n_sgb:,} — these always land in same category)")

    print("  E[N] per category:")
    for c in CATEGORIES_SIMPLE:
        if E_N[c] >= 0.5:
            print(f"    {c:<30}: {E_N[c]:.1f}")

    # ── LOPO: test category invariance, not just omega SD ─────────────────
    # For each population p in each group, remove it and reclassify.
    # A category call is LOPO-robust if it survives ALL single-population removals.
    print("  Computing LOPO category robustness...")
    lopo_flag = np.zeros(n_sgb, bool)  # True = sensitive to single-pop removal

    for gname in ["Industrial", "Transitional"]:
        gr = group_results[gname]
        pops = gr.get("pops_used", [])
        if len(pops) < 2:
            continue
        psi_mat = np.stack([gr["psi_pop"][p] for p in pops], axis=0)  # (P, K)
        P = len(pops)
        for i in range(P):
            # omega_bar and M with population i removed
            mask = np.ones(P, bool); mask[i] = False
            omega_lopo = psi_mat[mask].mean(0)
            M_lopo     = (psi_mat[mask] > tau).sum(0).astype(int)
            if gname == "Industrial":
                cats_lopo, _, _, _, _ = classify_two_group_with_hg_annotation(
                    omega_lopo, omega_T, omega_HG, M_lopo, M_T,
                    lam_I_from_T, lam_T_from_I,
                    obs_I_pop=obs_I_pop, obs_T_pop=obs_T_pop,
                    eps=eps, tau=tau, delta=delta, m_I=1, m_T=m_T, lam_min=lam_min)
            else:
                cats_lopo, _, _, _, _ = classify_two_group_with_hg_annotation(
                    omega_I, omega_lopo, omega_HG, M_I, M_lopo,
                    lam_I_from_T, lam_T_from_I,
                    obs_I_pop=obs_I_pop, obs_T_pop=obs_T_pop,
                    eps=eps, tau=tau, delta=delta, m_I=1, m_T=m_T, lam_min=lam_min)
            # Flag any SGB whose category changed
            lopo_flag |= (cats_lopo != cats_point)

    n_lopo_sensitive = int(lopo_flag.sum())
    print(f"  LOPO: {n_lopo_sensitive:,} SGBs change category when any one population removed")

    # ── Bootstrap stability scores ────────────────────────────────────────
    # p_cat[c][k] = fraction of bootstrap draws landing in category c for SGB k.
    # This is a BOOTSTRAP STABILITY SCORE, not a Bayesian posterior probability.
    # 100% stability means every bootstrap draw agrees — which is expected for
    # extreme SGBs (consistently detected or not) and uninformative for borderline ones.
    # "Unresolved" is excluded: stability of Unresolved is not a useful selection criterion.
    SELECTABLE_CATS = [c for c in CATEGORIES_SIMPLE if c != "Unresolved"]

    print(f"\n[6/6] Bootstrap stability selection (stability ≥ {1-target_bfdr:.0%})...")
    print(f"  NOTE: 'Unresolved' excluded from selection — not a biological category.")
    print(f"  NOTE: 100% stability = always same category across all bootstrap draws.")
    print(f"        For SGBs near thresholds, check omega CI not just stability score.")
    bfdr_results = {}
    for c in SELECTABLE_CATS:
        if E_N.get(c, 0) < 1:
            continue
        cutoff, selected, bfdr_ach, e_n = bfdr_threshold(p_cat[c], target_bfdr)
        bfdr_results[c] = dict(cutoff=cutoff, selected=selected,
                                boot_stability=1-bfdr_ach, E_N=e_n)
        n_sel  = int(selected.sum())
        n_hg_s = int((hg_flag[selected] == "HG-enriched").sum()) if n_sel else 0
        # How many of the selected have omega CI straddling eps or tau?
        n_borderline_I = int(
            ((omega_I_ci[0][selected] < eps) & (omega_I_ci[1][selected] > eps)).sum()
        ) if n_sel else 0
        print(f"  {c:<30}: n={n_sel:,}  stability≥{cutoff:.0%}  "
              f"HG-enriched={n_hg_s} ({100*n_hg_s/max(n_sel,1):.0f}%)  "
              f"borderline_ε={n_borderline_I}")

    return dict(
        ref_name=ref_name, ref_sgbs=ref_sgbs, n_sgb=n_sgb,
        group_results=group_results,
        omega_I=omega_I, omega_T=omega_T, omega_HG=omega_HG,
        omega_I_ci=omega_I_ci, omega_T_ci=omega_T_ci,
        omega_HG_ci=omega_HG_ci,
        omega_boots=omega_boots, hg_boots=hg_boots,
        M_I=M_I, M_T=M_T,
        lam_I_from_T=lam_I_from_T, lam_T_from_I=lam_T_from_I,
        power_I=power_I, power_T=power_T,
        cats_point=cats_point, hg_flag=hg_flag, obs_label=obs_label,
        cat_counts=cat_counts,
        # Hard observation flags (all pops, pre-filter)
        obs_I_sample=obs_I_sample, obs_I_pop=obs_I_pop,
        obs_T_sample=obs_T_sample, obs_T_pop=obs_T_pop,
        obs_HG_sample=obs_HG_sample, obs_HG_pop=obs_HG_pop,
        # Observation flags restricted to populations used in occupancy model
        obs_I_pop_used=obs_I_pop_used, obs_T_pop_used=obs_T_pop_used,
        p_cat=p_cat, E_N=E_N,
        bfdr_results=bfdr_results,
        lopo_flag=lopo_flag, lopo_sensitive_n=n_lopo_sensitive,
        hg_psi_pop=hg_psi_pop, hg_pops_used=hg_pops_used,
        eps=eps, tau=tau, delta=delta, m_T=m_T, lam_min=lam_min,
        threshold=threshold,
    )



# ════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ════════════════════════════════════════════════════════════════════════════

def draw_three_group_figure(results, out_stem):
    """Four-panel: A=category counts, B=ω scatter, C=Λ, D=LOPO robustness."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), gridspec_kw={"wspace": 0.40})
    ax_A, ax_B, ax_C, ax_D = axes
    for lbl, ax_ in zip("ABCD", axes):
        ax_.text(-0.11, 1.04, lbl, transform=ax_.transAxes,
                 fontsize=14, fontweight="bold", va="bottom")

    E_N = results["E_N"]; p_cat = results["p_cat"]
    bfdr = results["bfdr_results"]; hg_flag = results["hg_flag"]
    cats_pt = results["cats_point"]

    # A: category counts
    cats_plot = [c for c in CATEGORIES_SIMPLE if E_N.get(c, 0) >= 0.5]
    y_en  = [E_N.get(c, 0) for c in cats_plot]
    y_sel = [int(bfdr[c]["selected"].sum()) if c in bfdr else 0 for c in cats_plot]
    y_hg  = [int((hg_flag[bfdr[c]["selected"]] == "HG-enriched").sum())
              if c in bfdr and bfdr[c]["selected"].sum() > 0 else 0 for c in cats_plot]
    x = np.arange(len(cats_plot)); w = 0.28
    ax_A.bar(x-w, y_en,  w, color=[COL_CAT_SIMPLE.get(c,"#AAA") for c in cats_plot], alpha=0.50, label="E[N]")
    ax_A.bar(x,   y_sel, w, color=[COL_CAT_SIMPLE.get(c,"#AAA") for c in cats_plot], alpha=0.90, label="BFDR≤5%")
    ax_A.bar(x+w, y_hg,  w, color=COL_GROUP["HG"], alpha=0.75, label="HG-enriched")
    for xi, (en, ns, nh) in enumerate(zip(y_en, y_sel, y_hg)):
        if en >= 1: ax_A.text(xi-w, en+1, f"{en:.0f}", ha="center", fontsize=7, color="dimgrey")
        if ns >= 1: ax_A.text(xi,   ns+1, f"{ns}",     ha="center", fontsize=7, fontweight="bold")
        if nh >= 1: ax_A.text(xi+w, nh+1, f"{nh}",     ha="center", fontsize=7, color=COL_GROUP["HG"], fontweight="bold")
    ax_A.set_xticks(x); ax_A.set_xticklabels(cats_plot, rotation=30, ha="right", fontsize=8)
    ax_A.set_ylabel("SGBs"); ax_A.set_title("Category counts")
    ax_A.legend(fontsize=8, frameon=False)
    ax_A.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{int(v):,}"))
    ax_A.grid(axis="y", ls="--", lw=0.3, alpha=0.4)

    # B: occupancy landscape
    oI = results["omega_I"]; oT = results["omega_T"]; oHG = results["omega_HG"]
    sc = ax_B.scatter(oI, oT, c=oHG, cmap="YlGn", vmin=0, vmax=1, s=6, alpha=0.55, edgecolors="none")
    plt.colorbar(sc, ax=ax_B, label="ω̄_HG", fraction=0.04, pad=0.02)
    for thresh in [results["eps"], results["tau"]]:
        ax_B.axvline(thresh, color="dimgrey", lw=0.7, ls="--", alpha=0.5)
        ax_B.axhline(thresh, color="dimgrey", lw=0.7, ls="--", alpha=0.5)
    ax_B.set_xlabel("ω̄_Industrial"); ax_B.set_ylabel("ω̄_Transitional")
    ax_B.set_xlim(-0.02, 1.02); ax_B.set_ylim(-0.02, 1.02)
    ax_B.set_title("Occupancy (colour=HG)"); ax_B.grid(ls="--", lw=0.25, alpha=0.3)
    ax_B.set_aspect("equal")

    # C: Λ distributions
    for lam_v, lbl, col in [(results["lam_I_from_T"], "Λ_Ind←T", COL_GROUP["Industrial"]),
                             (results["lam_T_from_I"], "Λ_T←Ind", COL_GROUP["Transitional"])]:
        finite = lam_v[(lam_v > 0) & np.isfinite(lam_v)]
        if len(finite): ax_C.hist(np.log10(np.maximum(finite, 1e-3)), bins=40, color=col, alpha=0.55, edgecolor="white", label=lbl)
    ax_C.axvline(np.log10(results["lam_min"]), color="black", lw=1.0, ls="--", label=f"Λ_min={results['lam_min']:.0f}")
    ax_C.set_xlabel("log₁₀(Λ)"); ax_C.set_ylabel("SGBs"); ax_C.set_title("Power metrics")
    ax_C.legend(fontsize=8, frameon=False)
    ax_C.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"$10^{{{int(round(v))}}}$"))
    ax_C.grid(axis="y", ls="--", lw=0.3, alpha=0.4)

    # D: LOPO robustness
    cats_lopo = [c for c in CATEGORIES_SIMPLE if results["cat_counts"].get(c, 0) > 0]
    lopo_frac = [(~results["lopo_flag"][cats_pt == c]).mean() if (cats_pt == c).sum() else 0.0
                 for c in cats_lopo]
    x2 = np.arange(len(cats_lopo))
    bars = ax_D.bar(x2, lopo_frac, color=[COL_CAT_SIMPLE.get(c,"#AAA") for c in cats_lopo], alpha=0.82, edgecolor="white")
    ax_D.axhline(0.80, color="black", lw=0.9, ls="--", alpha=0.6, label="80%")
    for bar_, f in zip(bars, lopo_frac):
        ax_D.text(bar_.get_x()+bar_.get_width()/2, f+0.01, f"{f:.0%}", ha="center", fontsize=8)
    ax_D.set_xticks(x2); ax_D.set_xticklabels(cats_lopo, rotation=30, ha="right", fontsize=8)
    ax_D.set_ylim(0, 1.12); ax_D.set_ylabel("Fraction LOPO-robust")
    ax_D.set_title("LOPO robustness (I+T only)"); ax_D.legend(fontsize=8, frameon=False)
    ax_D.grid(axis="y", ls="--", lw=0.3, alpha=0.4)

    fig.suptitle(f"Three-group [{results['ref_name']}] — Primary: I vs T | HG: annotation  |  "
                 f"ε={results['eps']}  τ={results['tau']}  m_T≥{results['m_T']}  Λ_min={results['lam_min']}",
                 fontsize=10, fontweight="bold", y=1.01)
    fig.canvas.draw()
    apply_display_text(fig)
    pdf_p = f"{out_stem}_three_group.pdf"; png_p = f"{out_stem}_three_group.png"
    fig.savefig(pdf_p, bbox_inches="tight", facecolor="white")
    tmp = os.path.join(tempfile.gettempdir(), os.path.basename(png_p))
    fig.savefig(tmp, dpi=150, bbox_inches="tight", facecolor="white")
    shutil.move(tmp, png_p); plt.close(fig)
    print(f"  Saved → {pdf_p}"); print(f"  Saved → {png_p}")

#  STAN JOINT MODEL SCAFFOLD
# ════════════════════════════════════════════════════════════════════════════

STAN_THREE_GROUP = r"""
data {
  int<lower=1> K;          // SGBs
  int<lower=1> S;          // samples (all groups pooled)
  int<lower=1> J_study;    // studies
  int<lower=1> J_pop;      // populations
  int<lower=1> J_cont;     // continents

  // Per-sample group membership (1=Industrial 2=Transitional 3=HG)
  array[S] int<lower=1,upper=3>      group;
  array[S] int<lower=1,upper=J_pop>  pop;
  array[S] int<lower=1,upper=J_study> study;
  array[S] int<lower=1,upper=J_cont> continent;

  vector[S] log_D;   // log(total_mapped_bp)
  vector[S] q_s;     // quality offset (log actual/expected)
  vector<lower=0>[K] G;      // genome lengths
  real<lower=0> read_len;
  real<lower=0,upper=1> breadth_thr;

  // Sparse observed read counts
  int<lower=0> N_obs;
  array[N_obs] int<lower=1,upper=S> obs_s;
  array[N_obs] int<lower=1,upper=K> obs_k;
  array[N_obs] int<lower=1>         obs_r;

  // Population → group mapping (for equal-population summary)
  int<lower=1> P_I;  int<lower=1> P_T;  int<lower=1> P_HG;
  array[J_pop] int<lower=1,upper=3> pop_group;
}

parameters {
  // Species-level occupancy intercept (NI baseline = group 3 = HG)
  vector[K] alpha_k_raw;
  real mu_alpha; real<lower=0> sigma_alpha;

  // Group contrasts (Industrial and Transitional vs HG)
  vector[K] beta_I_raw;   real mu_betaI; real<lower=0> sigma_betaI;
  vector[K] beta_T_raw;   real mu_betaT; real<lower=0> sigma_betaT;

  // Random effects
  vector[J_pop]  u_pop_raw;   real<lower=0> sigma_pop;
  vector[J_study] u_study_raw; real<lower=0> sigma_study;
  vector[J_cont]  u_cont_raw;  real<lower=0> sigma_cont;

  // Abundance layer
  vector[K] eta_k_raw; real mu_eta; real<lower=0> sigma_eta;
  vector[J_pop] v_pop_raw; real<lower=0> sigma_v_pop;
  vector<lower=0>[K] phi_k;
}

transformed parameters {
  vector[K] alpha_k = mu_alpha + sigma_alpha * alpha_k_raw;
  vector[K] beta_I  = mu_betaI + sigma_betaI * beta_I_raw;
  vector[K] beta_T  = mu_betaT + sigma_betaT * beta_T_raw;
  vector[J_pop]   u_pop   = sigma_pop   * u_pop_raw;
  vector[J_study] u_study = sigma_study * u_study_raw;
  vector[J_cont]  u_cont  = sigma_cont  * u_cont_raw;
  vector[K] eta_k = mu_eta + sigma_eta * eta_k_raw;
  vector[J_pop] v_pop = sigma_v_pop * v_pop_raw;
}

model {
  // Priors
  mu_alpha ~ normal(0, 2); sigma_alpha ~ normal(0, 1);
  mu_betaI ~ normal(0, 2); sigma_betaI ~ normal(0, 1);
  mu_betaT ~ normal(0, 2); sigma_betaT ~ normal(0, 1);
  sigma_pop ~ normal(0, 1); sigma_study ~ normal(0, 1);
  sigma_cont ~ normal(0, 0.5);
  mu_eta ~ normal(0, 3); sigma_eta ~ normal(0, 1);
  sigma_v_pop ~ normal(0, 1);
  alpha_k_raw ~ std_normal(); beta_I_raw ~ std_normal(); beta_T_raw ~ std_normal();
  u_pop_raw ~ std_normal(); u_study_raw ~ std_normal(); u_cont_raw ~ std_normal();
  eta_k_raw ~ std_normal(); v_pop_raw ~ std_normal();
  phi_k ~ normal(3, 2);

  // Likelihood: marginalise over Z_ks
  for (s in 1:S) {
    for (k in 1:K) {
      real g_effect = (group[s] == 1) ? beta_I[k] :
                      (group[s] == 2) ? beta_T[k] : 0.0;
      real logit_psi = alpha_k[k] + g_effect + u_pop[pop[s]]
                       + u_study[study[s]] + u_cont[continent[s]];
      target += log1m_inv_logit(logit_psi);   // Z=0 baseline
    }
  }
  for (n in 1:N_obs) {
    int s = obs_s[n]; int k = obs_k[n];
    real g_effect = (group[s] == 1) ? beta_I[k] :
                    (group[s] == 2) ? beta_T[k] : 0.0;
    real logit_psi = alpha_k[k] + g_effect + u_pop[pop[s]]
                     + u_study[study[s]] + u_cont[continent[s]];
    real log_lam = log_D[s] + eta_k[k] + v_pop[pop[s]] + q_s[s] - log(G[k]);
    real lp0 = log1m_inv_logit(logit_psi);
    real lp1 = log_inv_logit(logit_psi)
               + neg_binomial_2_log_lpmf(obs_r[n] | log_lam, phi_k[k]);
    target += log_sum_exp(lp1, lp0) - lp0;
  }
}

generated quantities {
  // Per-population occupancy for each SGB
  // Equal-population group means
  vector[K] omega_I;
  vector[K] omega_T;
  vector[K] omega_HG;
  for (k in 1:K) {
    real sum_I = 0; real sum_T = 0; real sum_HG = 0;
    for (p in 1:J_pop) {
      real op = inv_logit(alpha_k[k]
                + ((pop_group[p] == 1) ? beta_I[k] :
                   (pop_group[p] == 2) ? beta_T[k] : 0.0)
                + u_pop[p]);
      if (pop_group[p] == 1) sum_I  += op;
      if (pop_group[p] == 2) sum_T  += op;
      if (pop_group[p] == 3) sum_HG += op;
    }
    omega_I[k]  = sum_I  / P_I;
    omega_T[k]  = sum_T  / P_T;
    omega_HG[k] = sum_HG / P_HG;
  }
}
"""


def write_stan_model(out_dir="."):
    """Write the Stan model file and return the path."""
    path = os.path.join(out_dir, "three_group_occupancy.stan")
    with open(path, "w") as f:
        f.write(STAN_THREE_GROUP)
    print(f"  Stan model written → {path}")
    return path


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    global FS

    parser = argparse.ArgumentParser(
        description="Three-group occupancy analysis (Industrial/Transitional/HG).\n\n"
                    "Primary comparison: Industrial vs Transitional (cluster bootstrap).\n"
                    "HG/Hadza: point-estimate annotation only (too few populations).")
    parser.add_argument("--ref",       default="UHGG",
                        help="Reference metagenome key (UHGG or lev)")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Breadth %% threshold for presence call (default 10)")
    parser.add_argument("--n-boot",    type=int,   default=200,
                        help="Cluster bootstrap replicates for I+T (default 200)")
    parser.add_argument("--min-pop",   type=int,   default=3,
                        help="Min samples per population for BB model (default 3)")
    parser.add_argument("--tau",       type=float, default=0.10,
                        help="Meaningful-presence occupancy threshold (default 0.10)")
    parser.add_argument("--eps",       type=float, default=0.05,
                        help="Near-absent threshold (default 0.05)")
    parser.add_argument("--delta",     type=float, default=0.10,
                        help="Depletion gap (ω̄_T - ω̄_I > delta = Depleted-Ind)")
    parser.add_argument("--m-T",       type=int,   default=2,
                        help="Min Transitional populations required for T-present call")
    parser.add_argument("--lam-min",   type=float, default=3.0,
                        help="Min expected detection count Λ for power adequacy")
    parser.add_argument("--bfdr",      type=float, default=0.05,
                        help="Bayesian FDR target (default 0.05)")
    parser.add_argument("--out",       default="three_group_occupancy",
                        help="Output file stem")
    parser.add_argument("--fs-path",   default=None,
                        help="Path to figure_summary.py if not auto-detected")
    parser.add_argument("--write-stan", action="store_true",
                        help="Write Stan model .stan file and exit")
    parser.add_argument("--run-stan",   action="store_true",
                        help="Fit Stan model (requires cmdstanpy)")
    parser.add_argument("--stan-chains",  type=int, default=4)
    parser.add_argument("--stan-warmup",  type=int, default=500)
    parser.add_argument("--stan-samples", type=int, default=500)
    parser.add_argument("--export-csv",   action="store_true",
                        help="Export per-SGB results to CSV")
    args = parser.parse_args(argv)

    # ── Load pipeline module ─────────────────────────────────────────────
    if args.fs_path:
        import importlib.util
        spec = importlib.util.spec_from_file_location("figure_summary", args.fs_path)
        FS   = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(FS)
    else:
        FS = _import_pipeline()
    print(f"  Pipeline loaded from: {FS.__file__}")

    if args.write_stan:
        write_stan_model(os.path.dirname(args.out) or ".")
        return

    rng = np.random.default_rng(42)

    # ── Load data (identical preprocessing to figure_summary.py) ─────────
    print(f"\n=== Loading {args.ref} ===")
    rdat = FS.load_reference(args.ref)
    print(f"  {rdat['n_sgbs']:,} SGBs  {len(rdat['meta']):,} samples")

    # ── Run analysis ──────────────────────────────────────────────────────
    results = run_three_group_analysis(
        rdat, threshold=args.threshold, ref_name=args.ref,
        n_boot=args.n_boot, min_pop=args.min_pop,
        tau=args.tau, eps=args.eps, delta=args.delta,
        m_T=args.m_T, lam_min=args.lam_min,
        target_bfdr=args.bfdr, rng=rng)

    # ── Figure ────────────────────────────────────────────────────────────
    print("\n=== Drawing figure ===")
    draw_three_group_figure(results, args.out)

    # ── CSV export ────────────────────────────────────────────────────────
    if args.export_csv:
        print("\n=== Exporting per-SGB results ===")
        df = _export_results_csv(results, args.out)
        print(f"  {len(df):,} SGBs exported")

    # ── Optional Stan ─────────────────────────────────────────────────────
    if args.run_stan:
        print("\n=== Fitting Stan joint model ===")
        stan_path = write_stan_model(os.path.dirname(args.out) or ".")
        try:
            import cmdstanpy
            model = cmdstanpy.CmdStanModel(stan_file=stan_path)
            print("  Stan model compiled — prepare data dict from prepare_stan_data()")
            print("  in figure_summary.py, extended with group[], pop[], continent[] arrays.")
        except ImportError:
            print("  cmdstanpy not installed.")
            print("  pip install cmdstanpy && python -m cmdstanpy.install_cmdstan")

    print("\nDone.")


def _export_results_csv(results, out_stem):
    """
    Export per-SGB results to a tidy CSV for downstream analysis.

    Columns
    -------
    sgb              : SGB identifier
    omega_I, omega_T, omega_HG : equal-population mean occupancy (point estimates)
    omega_I_lo/hi    : 2.5/97.5 bootstrap CI for Industrial and Transitional
    omega_T_lo/hi
    omega_HG_lo/hi   : sample-bootstrap CI (not cluster — HG only)
    M_I, M_T         : population replication counts
    lam_I_from_T     : Λ_Ind←T (Industrial power to detect T-level SGBs)
    lam_T_from_I     : Λ_T←Ind
    power_I, power_T : bool — above lam_min threshold
    category         : point-estimate category
    hg_flag          : HG enrichment annotation
    lopo_flag        : True if category sensitive to single-population removal
    p_<cat>          : bootstrap posterior probability per category
    bfdr_selected_<cat>: bool — selected at BFDR ≤ 5% for that category
    """
    n = results["n_sgb"]
    df = pd.DataFrame({"sgb": results["ref_sgbs"]})

    df["omega_I"]   = results["omega_I"]
    df["omega_T"]   = results["omega_T"]
    df["omega_HG"]  = results["omega_HG"]

    if results["omega_I_ci"] is not None and results["omega_I_ci"].shape[1] == n:
        df["omega_I_lo"] = results["omega_I_ci"][0]
        df["omega_I_hi"] = results["omega_I_ci"][1]
    if results["omega_T_ci"] is not None and results["omega_T_ci"].shape[1] == n:
        df["omega_T_lo"] = results["omega_T_ci"][0]
        df["omega_T_hi"] = results["omega_T_ci"][1]
    if results["omega_HG_ci"] is not None and results["omega_HG_ci"].shape[1] == n:
        df["omega_HG_lo"] = results["omega_HG_ci"][0]
        df["omega_HG_hi"] = results["omega_HG_ci"][1]

    df["M_I"]          = results["M_I"]
    df["M_T"]          = results["M_T"]
    df["lam_I_from_T"] = results["lam_I_from_T"]
    df["lam_T_from_I"] = results["lam_T_from_I"]
    df["power_I"]      = results["power_I"]
    df["power_T"]      = results["power_T"]
    df["category"]     = results["cats_point"]
    df["hg_flag"]      = results["hg_flag"]
    df["obs_label"]    = results["obs_label"]
    df["lopo_flag"]    = results["lopo_flag"]
    # Hard observation flags — all populations (pre-filter)
    df["obs_I_sample"] = results.get("obs_I_sample", np.zeros(n, bool))
    df["obs_I_pop"]    = results.get("obs_I_pop",    np.zeros(n, int))
    df["obs_T_sample"] = results.get("obs_T_sample", np.zeros(n, bool))
    df["obs_T_pop"]    = results.get("obs_T_pop",    np.zeros(n, int))
    df["obs_HG_sample"]= results.get("obs_HG_sample",np.zeros(n, bool))
    df["obs_HG_pop"]   = results.get("obs_HG_pop",   np.zeros(n, int))
    # Restricted to populations used in occupancy model — use for 2×2 diagnostic
    df["obs_I_pop_used"] = results.get("obs_I_pop_used", np.zeros(n, int))
    df["obs_T_pop_used"] = results.get("obs_T_pop_used", np.zeros(n, int))

    for c in CATEGORIES_SIMPLE:
        safe = c.replace(" ", "_").replace("-", "_")
        df[f"p_{safe}"] = results["p_cat"].get(c, np.zeros(n))

    for c, bd in results["bfdr_results"].items():
        safe = c.replace(" ", "_").replace("-", "_")
        df[f"bfdr_selected_{safe}"] = bd["selected"]

    path = f"{out_stem}_per_sgb.csv"
    df.to_csv(path, index=False)
    print(f"  → {path}")
    return df


if __name__ == "__main__":
    main()
