"""
Per-sample read redistribution to correct genome-length competition bias.

Problem
-------
Industrial metagenomes are enriched for bacteria with long genomes at the HEAD
of the rank-abundance distribution.  These long-genome dominants consume a
disproportionate share of the sequencing budget (abundance = depth x L),
squeezing out detection of rarer, shorter-genome taxa near the threshold.

Solution — Direct Mechanistic Read Redistribution
---------------------------------------------------
For each sample we ask: "what if every SGB had length L_ref?"

The redistribution operates on ABUNDANCE (depth x genome_length), which
represents total read mass assigned to each genome — the quantity that
determines sequencing budget consumption.  Detection, however, is assessed
via MAPPED BASES ((breadth/100) x genome_length), which is the quantity
relevant to the flat-bp threshold.

Per-sample algorithm:

  1. For every SGB:
       mapped_bases_i = (breadth_i/100) x L_i  (for detection)
       abundance_i = depth_i x L_i             (for redistribution)

  2. Choose a reference genome length L_ref (default: global median).

  3. For each detected donor (L_i > L_ref):
       excess_i = depth_i x (L_i - L_ref)

  4. Total freed reads F = Σ excess_i over detected donors.

  5. Distribute F to receivers (L_i < L_ref, depth > 0):
       gain_i = F x absorption_i / Σ absorption_j
       where absorption_i = depth_i x (L_ref - L_i)

  6. Scale mapped_bases proportionally with abundance change:
       new_mapped_bases_i = mapped_bases_i x (new_abundance_i / abundance_i)

  7. Recount detections: corrected_richness = Σ (new_mapped_bases_i >= threshold)

Integration
-----------
This module is called from section1_diversity.py to produce "corrected richness"
alpha-diversity plots alongside the standard observed richness.  It operates on
the 1:1 depth-matched sample pairs, using only the flat-bp detection threshold.

NOTE: detection.py uses mapped_bases = (breadth/100) x L for the flat-bp
threshold.  This module uses abundance = depth x L for the redistribution
math, then converts back to mapped_bases for threshold application.
"""

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ════════════════════════════════════════════════════════════════════
#  Core per-sample correction
# ════════════════════════════════════════════════════════════════════

@dataclass
class SampleRedistResult:
    """Per-sample redistribution metrics."""
    richness_observed: int
    richness_corrected: int
    delta_N: int
    freed_reads: float
    freed_reads_pct: float
    total_mapped_bases: float
    n_donors: int
    n_receivers: int
    mean_L_detected_Mbp: float
    rw_mean_L_Mbp: float


def redistribute_single_sample(
    breadth_row: np.ndarray,
    depth_row: np.ndarray,
    lens_arr: np.ndarray,
    L_ref: float,
    flat_threshold_bp: float,
) -> SampleRedistResult:
    """
    Run genome-length redistribution for a single sample.

    Detection uses mapped_bases = (breadth%/100) x L (unique positions covered).
    Redistribution operates on abundance = depth x L (sequencing budget).
    After redistribution, mapped_bases are scaled proportionally with the
    abundance change to assess corrected detection.

    Parameters
    ----------
    breadth_row : array (n_sgbs,) — breadth coverage (%) per SGB
    depth_row : array (n_sgbs,) — mean depth per SGB
    lens_arr : array (n_sgbs,) — genome length per SGB (bp)
    L_ref : float — reference genome length (bp)
    flat_threshold_bp : float — mapped-bases threshold for detection

    Returns
    -------
    SampleRedistResult
    """
    # Detection via mapped bases (breadth-based)
    breadth_frac = breadth_row / 100.0
    mapped_bases = breadth_frac * lens_arr
    det_mask = mapped_bases >= flat_threshold_bp
    n_detected = int(det_mask.sum())
    total_det_mb = float(mapped_bases[det_mask].sum())

    # Abundance for redistribution math
    abundance = depth_row * lens_arr

    if n_detected < 2 or total_det_mb <= 0:
        return SampleRedistResult(
            richness_observed=n_detected,
            richness_corrected=n_detected,
            delta_N=0,
            freed_reads=0.0, freed_reads_pct=0.0,
            total_mapped_bases=total_det_mb,
            n_donors=0, n_receivers=0,
            mean_L_detected_Mbp=np.nan, rw_mean_L_Mbp=np.nan,
        )

    # Length deviation
    delta_L = lens_arr - L_ref
    long_mask = lens_arr > L_ref
    short_mask = lens_arr < L_ref

    # Donors: DETECTED SGBs with L > L_ref (abundance-based redistribution)
    donor_mask = det_mask & long_mask
    excess = depth_row * np.maximum(delta_L, 0.0)
    freed_reads = float(excess[donor_mask].sum())

    # Receivers: any SGB with L < L_ref and some signal
    has_signal = depth_row > 0
    receiver_mask = short_mask & has_signal
    absorption = depth_row * np.maximum(-delta_L, 0.0)
    total_absorption = float(absorption[receiver_mask].sum())

    # Redistribute abundance, then scale mapped_bases proportionally
    new_abundance = abundance.copy()
    if freed_reads > 0 and total_absorption > 0:
        new_abundance[donor_mask] -= excess[donor_mask]
        gains = np.zeros(len(lens_arr))
        gains[receiver_mask] = (
            freed_reads * absorption[receiver_mask] / total_absorption
        )
        new_abundance += gains

        # Scale mapped_bases by abundance ratio where abundance > 0
        new_mapped_bases = mapped_bases.copy()
        nonzero = abundance > 0
        new_mapped_bases[nonzero] = (
            mapped_bases[nonzero] * new_abundance[nonzero] / abundance[nonzero]
        )
        det_after = new_mapped_bases >= flat_threshold_bp
        n_after = int(det_after.sum())
    else:
        n_after = n_detected

    n_donors = int(donor_mask.sum())
    n_receivers = int(receiver_mask.sum())
    det_lens = lens_arr[det_mask]
    det_mb = mapped_bases[det_mask]
    mean_L_det = float(det_lens.mean()) / 1e6
    rw_mean_L = (float(np.average(det_lens, weights=det_mb)) / 1e6
                 if total_det_mb > 0 else mean_L_det)

    return SampleRedistResult(
        richness_observed=n_detected,
        richness_corrected=n_after,
        delta_N=n_after - n_detected,
        freed_reads=freed_reads,
        freed_reads_pct=freed_reads / max(total_det_mb, 1) * 100,
        total_mapped_bases=total_det_mb,
        n_donors=n_donors,
        n_receivers=n_receivers,
        mean_L_detected_Mbp=mean_L_det,
        rw_mean_L_Mbp=rw_mean_L,
    )


# ════════════════════════════════════════════════════════════════════
#  Batch redistribution for a set of samples
# ════════════════════════════════════════════════════════════════════

def redistribute_samples(
    breadth_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    lengths: pd.Series,
    sample_ids: List[str],
    flat_threshold_bp: float = 300_000,
    L_ref: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run redistribution for a list of samples.

    Parameters
    ----------
    breadth_df : DataFrame (samples x SGBs), breadth coverage (%)
    depth_df : DataFrame (samples x SGBs), mean depth
    lengths : Series (SGB → genome length in bp)
    sample_ids : list of sample IDs to process
    flat_threshold_bp : mapped-bases threshold for detection
    L_ref : reference genome length. None → global median of all SGB lengths.

    Returns
    -------
    DataFrame indexed by sample with columns:
        richness_observed, richness_corrected, delta_N,
        freed_reads, freed_reads_pct, total_mapped_bases,
        n_donors, n_receivers, mean_L_detected_Mbp, rw_mean_L_Mbp
    """
    sgbs = sorted(set(breadth_df.columns) & set(depth_df.columns)
                  & set(lengths.index))
    lens_arr = lengths[sgbs].values.astype(float)

    if L_ref is None:
        L_ref = float(np.median(lens_arr))

    rows = []
    for sid in sample_ids:
        if sid not in breadth_df.index or sid not in depth_df.index:
            continue
        breadth_row = breadth_df.loc[sid, sgbs].values.astype(float)
        depth_row = depth_df.loc[sid, sgbs].values.astype(float)
        res = redistribute_single_sample(
            breadth_row, depth_row, lens_arr, L_ref, flat_threshold_bp)
        rows.append({
            "sample": sid,
            "richness_observed": res.richness_observed,
            "richness_corrected": res.richness_corrected,
            "delta_N": res.delta_N,
            "freed_reads": res.freed_reads,
            "freed_reads_pct": res.freed_reads_pct,
            "total_mapped_bases": res.total_mapped_bases,
            "n_donors": res.n_donors,
            "n_receivers": res.n_receivers,
            "mean_L_detected_Mbp": res.mean_L_detected_Mbp,
            "rw_mean_L_Mbp": res.rw_mean_L_Mbp,
        })

    result = pd.DataFrame(rows).set_index("sample")
    return result, L_ref


# ════════════════════════════════════════════════════════════════════
#  Compute Shannon from redistributed bases
# ════════════════════════════════════════════════════════════════════

def redistribute_and_compute_shannon(
    breadth_df: pd.DataFrame,
    depth_df: pd.DataFrame,
    lengths: pd.Series,
    sample_ids: List[str],
    flat_threshold_bp: float = 300_000,
    L_ref: Optional[float] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Run redistribution AND compute Shannon diversity on redistributed
    relative abundances for each sample.

    Detection uses mapped_bases = (breadth%/100) x L.
    Redistribution operates on abundance = depth x L.
    Shannon is computed from abundance of detected SGBs.

    Returns (result_df, L_ref_used).
    result_df has columns: richness_observed, richness_corrected,
        shannon_observed, shannon_corrected, delta_N, ...
    """
    sgbs = sorted(set(breadth_df.columns) & set(depth_df.columns)
                  & set(lengths.index))
    lens_arr = lengths[sgbs].values.astype(float)

    if L_ref is None:
        L_ref = float(np.median(lens_arr))

    delta_L = lens_arr - L_ref
    long_mask = lens_arr > L_ref
    short_mask = lens_arr < L_ref

    rows = []
    for sid in sample_ids:
        if sid not in breadth_df.index or sid not in depth_df.index:
            continue
        breadth_row = breadth_df.loc[sid, sgbs].values.astype(float)
        depth_row = depth_df.loc[sid, sgbs].values.astype(float)

        # Detection via mapped bases (breadth-based)
        breadth_frac = breadth_row / 100.0
        mapped_bases = breadth_frac * lens_arr
        det_mask = mapped_bases >= flat_threshold_bp
        n_detected = int(det_mask.sum())

        # Abundance for redistribution and Shannon
        abundance = depth_row * lens_arr
        total_det_abund = float(abundance[det_mask].sum())

        # Observed Shannon (from abundance of detected SGBs)
        if n_detected > 0 and total_det_abund > 0:
            rel_obs = abundance[det_mask] / total_det_abund
            rel_obs = rel_obs[rel_obs > 0]
            shannon_obs = float(-np.sum(rel_obs * np.log(rel_obs)))
        else:
            shannon_obs = 0.0

        total_det_mb = float(mapped_bases[det_mask].sum())

        if n_detected < 2 or total_det_mb <= 0:
            rows.append({
                "sample": sid,
                "richness_observed": n_detected,
                "richness_corrected": n_detected,
                "shannon_observed": shannon_obs,
                "shannon_corrected": shannon_obs,
                "delta_N": 0,
                "freed_reads_pct": 0.0,
                "rw_mean_L_Mbp": np.nan,
            })
            continue

        # Redistribution (abundance-based)
        donor_mask = det_mask & long_mask
        excess = depth_row * np.maximum(delta_L, 0.0)
        freed = float(excess[donor_mask].sum())

        has_signal = depth_row > 0
        receiver_mask = short_mask & has_signal
        absorption = depth_row * np.maximum(-delta_L, 0.0)
        total_abs = float(absorption[receiver_mask].sum())

        new_abundance = abundance.copy()
        if freed > 0 and total_abs > 0:
            new_abundance[donor_mask] -= excess[donor_mask]
            gains = np.zeros(len(lens_arr))
            gains[receiver_mask] = freed * absorption[receiver_mask] / total_abs
            new_abundance += gains

        # Scale mapped_bases proportionally with abundance change
        new_mapped_bases = mapped_bases.copy()
        nonzero = abundance > 0
        new_mapped_bases[nonzero] = (
            mapped_bases[nonzero] * new_abundance[nonzero] / abundance[nonzero]
        )
        det_after = new_mapped_bases >= flat_threshold_bp
        n_after = int(det_after.sum())

        # Corrected Shannon (from redistributed abundance of newly-detected SGBs)
        total_corr_abund = float(new_abundance[det_after].sum())
        if n_after > 0 and total_corr_abund > 0:
            rel_corr = new_abundance[det_after] / total_corr_abund
            rel_corr = rel_corr[rel_corr > 0]
            shannon_corr = float(-np.sum(rel_corr * np.log(rel_corr)))
        else:
            shannon_corr = 0.0

        det_lens = lens_arr[det_mask]
        det_mb = mapped_bases[det_mask]
        rw_mean_L = (float(np.average(det_lens, weights=det_mb)) / 1e6
                     if total_det_mb > 0 else np.nan)

        rows.append({
            "sample": sid,
            "richness_observed": n_detected,
            "richness_corrected": n_after,
            "shannon_observed": shannon_obs,
            "shannon_corrected": shannon_corr,
            "delta_N": n_after - n_detected,
            "freed_reads_pct": freed / max(total_det_mb, 1) * 100,
            "rw_mean_L_Mbp": rw_mean_L,
        })

    return pd.DataFrame(rows).set_index("sample"), L_ref
