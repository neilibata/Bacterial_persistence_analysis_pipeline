"""
reinstate_secondary.py
======================
Identifies secondary runs that are compositionally distinct from their
primary sample (Bray-Curtis > BC_REINSTATE_THRESHOLD) and reinstates
them as independent samples in the pipeline.

Rationale
---------
Secondary run IDs were set aside during metadata construction because
they mapped to the same biosample as a primary run.  However, a high
Bray-Curtis dissimilarity between primary and secondary indicates that
the two runs captured genuinely different microbial communities — they
are distinct samples, not technical replicates — and should be analysed
as such.

Logic
-----
For every primary sample p with secondary runs [s1, s2, ...]:
  1. Compute Bray-Curtis between p (data.abundance) and si (derived
     from sec_breadth / sec_depth in the same way as primary abundance).
  2. If BC(p, si) > BC_REINSTATE_THRESHOLD (default 0.1):
       - Add si rows to data.breadth, data.depth, data.abundance,
         data.total_bp, data.total_seq_bp.
       - Copy p's metadata row for si, mark _reinstated_from = p.
       - Apply the same computed annotations (Country_norm, Continent,
         MacroGroup, Group_fine, Total_depth_bp, _depth, _log_depth,
         _study, flag columns).
       - Remove si from sec_map so it is no longer treated as a
         technical replicate.
  3. Write a CSV of all reinstated samples + their metadata.

The returned ReferenceData has:
  - All primary samples (unchanged)
  - All reinstated secondary samples appended
  - sec_map with reinstated IDs removed (remaining entries are genuine
    technical replicates with BC <= threshold)

Output
------
  results/<ref>/reinstated_secondary_samples.csv

Integration
-----------
Called from __main__.run_reference() immediately after run_depth_audit()
and before depth matching.  Runs unconditionally (not gated by --only).
"""

import os
import numpy as np
import pandas as pd

from . import config
from .loading import ReferenceData


# ── threshold ─────────────────────────────────────────────────────────────────
BC_REINSTATE_THRESHOLD = getattr(config, "BC_REINSTATE_THRESHOLD", 0.1)


# ── helpers ───────────────────────────────────────────────────────────────────

def _abundance_from_sec(sec_breadth, sec_depth, lengths, sgbs):
    """
    Compute relative abundance for secondary run rows.

    Mirrors the abundance calculation in loading.load_reference():
        abundance_raw = depth x genome_length
        abundance = abundance_raw / row_sum

    Returns
    -------
    abund_df  : DataFrame (secondary samples x SGBs), relative abundance
    total_bp  : Series   (secondary samples,), total mapped bp
    """
    sec_b = sec_breadth.reindex(columns=sgbs, fill_value=0.0)
    sec_d = sec_depth.reindex(columns=sgbs, fill_value=0.0)

    lengths_arr = lengths.reindex(sgbs, fill_value=0.0).values
    abund_raw   = sec_d.values * lengths_arr          # (n_sec, n_sgbs)
    total_bp    = pd.Series(abund_raw.sum(axis=1), index=sec_d.index)

    row_sums = total_bp.values.copy()
    row_sums[row_sums == 0] = np.nan
    abund_values = abund_raw / row_sums[:, np.newaxis]
    abund_df = pd.DataFrame(
        np.nan_to_num(abund_values, nan=0.0),
        index=sec_d.index, columns=sgbs,
    )
    return abund_df, total_bp


def _braycurtis(u, v):
    """Bray-Curtis dissimilarity between two 1-D abundance vectors."""
    denom = u.sum() + v.sum()
    if denom == 0:
        return np.nan
    return float(np.abs(u - v).sum() / denom)


def _build_reinstated_meta_row(primary_row, sec_id, pri_id,
                               sec_total_bp, sec_abund_row, data):
    """
    Build a metadata Series for a reinstated secondary sample.

    Copies the primary's metadata, then overwrites the fields that
    should reflect the secondary run's own matrix-derived values.
    """
    row = primary_row.copy()

    # Matrix-derived depth fields
    row["Total_depth_bp"] = float(sec_total_bp)

    # Propagate computed depth columns
    row["_depth"] = float(sec_total_bp)
    row["_log_depth"] = float(np.log10(sec_total_bp)) if sec_total_bp > 0 else np.nan

    # Mark provenance
    row["_reinstated_from"] = pri_id

    return row


# ── main entry point ──────────────────────────────────────────────────────────

def reinstate_distinct_secondaries(data, stats):
    """
    Reinstate secondary runs with Bray-Curtis > threshold as full samples.

    Parameters
    ----------
    data  : ReferenceData  (after depth audit)
    stats : StatsReport

    Returns
    -------
    ReferenceData with reinstated samples merged in, sec_map pruned.
    """
    ref    = data.ref_name
    outdir = config.outdir(ref)

    print(f"\n{'='*60}")
    print(f"  Secondary sample reinstatement [{ref}]")
    print(f"  BC threshold: {BC_REINSTATE_THRESHOLD}")
    print(f"{'='*60}")

    # ── Disabled by default ──────────────────────────────────────────────────
    # Every secondary is another sequencing run of the SAME metadata row, so
    # promoting one to an independent sample is only right where that row
    # conflates different specimens. It does not here: the pairs this fires on
    # are one participant across a diet intervention (ZeeviD), one specimen
    # under two DNA extraction protocols (CarterM Hadza_MoBio vs Hadza_PheChl),
    # and borderline MetaHIT resequencing (NielsenHB). See config for the full
    # audit. Left in place, and switchable, because the threshold sweep is a
    # supplementary sensitivity analysis.
    if not getattr(config, "REINSTATE_SECONDARIES", False):
        n_pairs = sum(len(v) for v in (data.sec_map or {}).values())
        print(f"  [OFF] config.REINSTATE_SECONDARIES is False — "
              f"{len(data.sec_map or {})} primaries with {n_pairs} secondary "
              f"run(s) stay technical replicates.")
        stats.add("reinstate_disabled", "cohort", "methods",
                  "Secondary runs retained as technical replicates rather than "
                  "reinstated as independent samples", n_pairs)
        return data

    # ── Guard ────────────────────────────────────────────────────────────────
    if not data.sec_map or data.sec_breadth is None or data.sec_depth is None:
        print("  [SKIP] No secondary runs available — nothing to reinstate.")
        return data

    sgbs = data.breadth.columns.tolist()

    # ── Pre-compute secondary abundances once ─────────────────────────────────
    sec_abund_df, sec_total_bp = _abundance_from_sec(
        data.sec_breadth, data.sec_depth, data.lengths, sgbs
    )

    # ── Screen every primary / secondary pair ─────────────────────────────────
    reinstated_ids   = []    # secondary IDs to add as full samples
    reinstated_from  = {}    # sec_id -> pri_id
    pair_bc_records  = []    # for the output CSV

    n_pairs_checked  = 0
    n_below_thresh   = 0

    for pri_id, sec_ids in data.sec_map.items():
        if pri_id not in data.abundance.index:
            continue
        pri_abund = data.abundance.loc[pri_id].values

        for sec_id in sec_ids:
            if sec_id not in sec_abund_df.index:
                continue

            n_pairs_checked += 1
            sec_abund = sec_abund_df.loc[sec_id].values
            bc = _braycurtis(pri_abund, sec_abund)

            pair_bc_records.append({
                "primary_id"   : pri_id,
                "secondary_id" : sec_id,
                "bray_curtis"  : round(bc, 4) if np.isfinite(bc) else np.nan,
                "reinstated"   : np.isfinite(bc) and bc > BC_REINSTATE_THRESHOLD,
            })

            if np.isfinite(bc) and bc > BC_REINSTATE_THRESHOLD:
                reinstated_ids.append(sec_id)
                reinstated_from[sec_id] = pri_id
            else:
                n_below_thresh += 1

    n_reinstated = len(reinstated_ids)
    print(f"  Pairs checked        : {n_pairs_checked:,}")
    print(f"  BC <= {BC_REINSTATE_THRESHOLD} (tech rep)  : {n_below_thresh:,}")
    print(f"  BC >  {BC_REINSTATE_THRESHOLD} (reinstated): {n_reinstated:,}")

    stats.add("sec_reinstatement_n_checked", "qc", "main",
              "Secondary runs screened for reinstatement",
              f"{n_pairs_checked:,} pairs, threshold BC={BC_REINSTATE_THRESHOLD}")
    stats.add("sec_reinstatement_n_reinstated", "qc", "main",
              "Secondary runs reinstated as independent samples",
              f"{n_reinstated:,}")

    # ── Write BC screening table ──────────────────────────────────────────────
    bc_csv = os.path.join(outdir, "secondary_bc_screening.csv")
    pd.DataFrame(pair_bc_records).to_csv(bc_csv, index=False)
    print(f"  BC screening table   : {bc_csv}")

    # ── If nothing to reinstate, return early ─────────────────────────────────
    if n_reinstated == 0:
        print("  No secondaries exceeded the BC threshold — data unchanged.")
        return data

    # ── Build new matrix rows for reinstated samples ──────────────────────────
    new_breadth  = data.sec_breadth.loc[reinstated_ids].reindex(columns=sgbs, fill_value=0.0)
    new_depth    = data.sec_depth.loc[reinstated_ids].reindex(columns=sgbs, fill_value=0.0)
    new_abund    = sec_abund_df.loc[reinstated_ids]
    new_total_bp = sec_total_bp.loc[reinstated_ids]

    # ── Build metadata rows for reinstated samples ────────────────────────────
    new_meta_rows = []
    for sec_id in reinstated_ids:
        pri_id  = reinstated_from[sec_id]
        pri_row = data.meta.loc[pri_id].copy()
        new_row = _build_reinstated_meta_row(
            pri_row, sec_id, pri_id,
            float(new_total_bp.loc[sec_id]),
            new_abund.loc[sec_id].values,
            data,
        )
        new_row.name = sec_id
        new_meta_rows.append(new_row)

    new_meta = pd.DataFrame(new_meta_rows)
    new_meta.index.name = "Chosen_match_ID"

    # ── Concatenate onto primary data ─────────────────────────────────────────
    merged_breadth  = pd.concat([data.breadth,   new_breadth],  axis=0)
    merged_depth    = pd.concat([data.depth,     new_depth],    axis=0)
    merged_abund    = pd.concat([data.abundance, new_abund],    axis=0)
    merged_total_bp = pd.concat([data.total_bp,  new_total_bp], axis=0)
    merged_meta     = pd.concat([data.meta,      new_meta],     axis=0)

    # total_seq_bp: NaN for reinstated (no raw FASTQ metadata row)
    new_seq_bp = pd.Series(np.nan, index=reinstated_ids, dtype=float)
    merged_seq_bp = pd.concat([data.total_seq_bp, new_seq_bp], axis=0)

    # ── Prune sec_map: remove reinstated IDs ─────────────────────────────────
    reinstated_set = set(reinstated_ids)
    new_sec_map = {}
    for pri_id, sec_ids in data.sec_map.items():
        remaining = [s for s in sec_ids if s not in reinstated_set]
        if remaining:
            new_sec_map[pri_id] = remaining

    # sec_breadth / sec_depth: drop reinstated rows
    remaining_sec_ids = [s for s in data.sec_breadth.index
                         if s not in reinstated_set]
    new_sec_breadth = data.sec_breadth.loc[remaining_sec_ids] if remaining_sec_ids else None
    new_sec_depth   = data.sec_depth.loc[remaining_sec_ids]   if remaining_sec_ids else None

    # ── Write reinstated-sample CSV ───────────────────────────────────────────
    # Include all metadata columns plus the _reinstated_from provenance column
    reinstated_csv = os.path.join(outdir, "reinstated_secondary_samples.csv")
    out_df = new_meta.copy().reset_index()
    out_df.to_csv(reinstated_csv, index=False)
    print(f"  Reinstated samples   : {reinstated_csv}  ({len(out_df):,} rows)")

    print(f"\n  Data size before reinstatement: {len(data.meta):,} samples")
    print(f"  Data size after  reinstatement: {len(merged_meta):,} samples")
    print(f"    Lifestyle: {merged_meta['Lifestyle'].value_counts(dropna=False).to_dict()}")
    print(f"    Health:    {merged_meta['Unhealthy'].value_counts(dropna=False).to_dict()}")

    return ReferenceData(
        ref_name=data.ref_name,
        breadth=merged_breadth,
        depth=merged_depth,
        abundance=merged_abund,
        meta=merged_meta,
        lengths=data.lengths,
        total_bp=merged_total_bp,
        total_seq_bp=merged_seq_bp,
        vanish_set=data.vanish_set,
        blossum_set=data.blossum_set,
        cov_matrix=merged_breadth,
        sec_breadth=new_sec_breadth,
        sec_depth=new_sec_depth,
        sec_map=new_sec_map,
    )
