"""Identify the Carter et al. (2023) meta-analysis cohort inside our metadata.

Carter 2023 is a meta-analysis of 1,800 metagenomes drawn from many studies. Its
samples are therefore scattered across our cohort under whatever identifiers each
source study used; they are NOT the samples Carter themselves sequenced, and they
do not share a study label.

The previous implementation matched `StudyFolder` against the string "carter".
That found Carter's own newly-sequenced samples -- 389 rows, 82% hunter-gatherer
-- and missed 1,526 of the 1,800. It also picked up 186 samples that were never
in Carter's study. The lifestyle composition came out close to inverted, so any
figure built on it was not showing what its caption claimed.

This module matches on identifiers instead, and on all of them: sample ids, SRS
accessions, and both run lists, tokenised so that a multi-run cell matches on any
of its runs. Recovery is checked against an expected band and a shortfall raises,
because silently analysing a fraction of the intended cohort is the failure mode
this replaces.
"""
from __future__ import annotations

import os
import re

import pandas as pd

# Carter's published sample list. Committed because it is the definition of the
# subset, not an intermediate: without it the cohort cannot be reconstructed.
CARTER_FILE = os.path.join("resources", "carter_sample_info.csv")

# Columns on each side that carry an identifier of some kind.
CARTER_ID_COLS = ("Sample ID", "SRS ID", "Paired SRR List", "Single SRR List")
META_ID_COLS = ("Master_ID", "Chosen_match_ID", "Secondary_run_IDs", "SRA_Accession",
                "Sample ID", "SRS ID", "Paired SRR List", "BioSample")

# Measured against metadata_Aug_26_2026_final.csv: 1,731 of 1,800 match, and of
# the 69 that do not, 53 are in the coverage matrix but absent from M1 and 16
# were never processed. Anything below this band means the join has broken, not
# that the cohort changed.
MIN_RECOVERY = 0.90

_SPLIT = re.compile(r"[,;|\s]+")


def _tokens(value) -> set[str]:
    """Identifier tokens in a cell, which may hold several run accessions."""
    if pd.isna(value):
        return set()
    out = set()
    for tok in _SPLIT.split(str(value)):
        tok = tok.strip().upper()
        # Bare integers are Study ID / Sample Number style row counters and would
        # collide with anything; short tokens are not accessions.
        if len(tok) > 3 and not tok.isdigit() and tok != "NAN":
            out.add(tok)
    return out


def _row_tokens(row, cols) -> set[str]:
    s = set()
    for c in cols:
        if c in row:
            s |= _tokens(row[c])
    return s


def find_carter_cohort(meta: pd.DataFrame, carter_file: str = None,
                       min_recovery: float = MIN_RECOVERY, strict: bool = True):
    """(mask over `meta`, diagnostic dict) for Carter's meta-analysis samples.

    Raises when recovery falls below `min_recovery`, unless `strict` is False.
    """
    path = carter_file or CARTER_FILE
    if not os.path.exists(path):
        msg = (f"Carter sample list not found at {path}. It defines the subset; "
               "without it the Carter figures cannot be built.")
        if strict:
            raise FileNotFoundError(msg)
        print(f"    [carter] {msg}")
        return pd.Series(False, index=meta.index), {"n": 0, "recovery": 0.0}

    carter = pd.read_csv(path, low_memory=False)

    # Every identifier Carter offers, in one pool.
    wanted = set()
    for col in CARTER_ID_COLS:
        if col in carter.columns:
            for v in carter[col]:
                wanted |= _tokens(v)

    present = [c for c in META_ID_COLS if c in meta.columns]
    mask = pd.Series(False, index=meta.index)
    per_col = {}
    for col in present:
        hit = meta[col].map(lambda v: bool(_tokens(v) & wanted))
        per_col[col] = int(hit.sum())
        mask |= hit

    # Recovery is measured Carter-side: what fraction of their samples we found.
    ours = set()
    for col in present:
        for v in meta.loc[mask, col]:
            ours |= _tokens(v)
    matched_rows = carter.apply(
        lambda r: bool(_row_tokens(r, CARTER_ID_COLS) & ours), axis=1)
    recovery = float(matched_rows.mean()) if len(carter) else 0.0

    diag = {
        "n": int(mask.sum()),
        "n_carter": int(len(carter)),
        "n_carter_matched": int(matched_rows.sum()),
        "recovery": round(recovery, 4),
        "per_column": per_col,
        "source": path,
    }
    if "Lifestyle" in meta.columns:
        diag["lifestyle"] = meta.loc[mask, "Lifestyle"].value_counts().to_dict()

    if recovery < min_recovery:
        msg = (f"Carter cohort recovery {recovery:.1%} is below the expected "
               f"{min_recovery:.0%}: matched {diag['n_carter_matched']} of "
               f"{diag['n_carter']} Carter samples to {diag['n']} of our rows. "
               "The identifier join has broken -- check that the metadata still "
               "carries SRS ID and the run lists.")
        if strict:
            raise RuntimeError(msg)
        print(f"    [carter] WARNING: {msg}")

    return mask, diag


def describe(diag: dict) -> str:
    """One-line summary for the run log."""
    ls = diag.get("lifestyle", {})
    parts = ", ".join(f"{k} {v}" for k, v in sorted(ls.items()))
    return (f"Carter cohort: {diag['n']:,} of our samples "
            f"({diag['n_carter_matched']:,}/{diag['n_carter']:,} of Carter's, "
            f"{diag['recovery']:.1%} recovered)"
            + (f" -- {parts}" if parts else ""))
