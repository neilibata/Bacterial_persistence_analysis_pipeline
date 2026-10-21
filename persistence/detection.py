"""
Unified SGB detection module.

Two detection modes
-------------------
1. **breadth_pct** (traditional):
       detected  ⟺  breadth_coverage >= threshold%
   Genome-length-DEPENDENT: longer genomes need more reads to reach
   the same % coverage, so this systematically undercounts SGBs with
   long genomes.

2. **flat_bp** (genome-length-independent):
       detected  ⟺  mapped_bases >= threshold_bp
   where  mapped_bases = (breadth% / 100) x genome_length
   i.e. the number of distinct genomic positions covered by >=1 read.
   This is genome-length-NEUTRAL by construction.

Both modes return the same shape: a boolean (samples x SGBs) DataFrame.
All downstream modules accept a DetectionSpec and call `build_presence()`
to get the appropriate presence matrix.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DetectionSpec:
    """Specifies how an SGB is declared 'present' in a sample."""

    mode: str               # "breadth_pct" or "flat_bp"
    threshold: float        # % for breadth_pct, bp for flat_bp
    label: Optional[str] = None  # human-readable short label

    def __post_init__(self):
        if self.mode not in ("breadth_pct", "flat_bp"):
            raise ValueError(f"Unknown detection mode: {self.mode}")

    @property
    def short_label(self):
        """Filesystem-safe short label for filenames."""
        if self.label:
            return self.label
        if self.mode == "breadth_pct":
            return f"{self.threshold:.0f}pct"
        else:
            if self.threshold >= 1_000_000:
                return f"{self.threshold/1_000_000:.0f}Mb"
            elif self.threshold >= 1_000:
                return f"{self.threshold/1_000:.0f}kb"
            else:
                return f"{self.threshold:.0f}bp"

    @property
    def description(self):
        """Human-readable description for titles and legends."""
        if self.mode == "breadth_pct":
            return f">={self.threshold:.0f}% breadth coverage"
        else:
            if self.threshold >= 1_000_000:
                return f">={self.threshold/1_000_000:.0f} Mb mapped bases"
            elif self.threshold >= 1_000:
                # kb, not kB. These are mapped BASES, so the unit is the
                # kilobase; kB is the kilobyte. `short_label` above already
                # wrote "300kb" into every filename, so the two disagreed
                # with each other and the supplement inherited both: Table
                # S12's headers read "300 kb" while Supplementary Text 6F
                # read "300 kB" for the same threshold.
                return f">={self.threshold/1_000:.0f} kb mapped bases"
            else:
                return f">={self.threshold:.0f} bp mapped bases"

    def __str__(self):
        return f"DetectionSpec({self.mode}, {self.short_label})"


# The presence matrix is the one computation every section shares, and it is
# rebuilt from scratch for each detection spec in each of them. Memoised, a
# format pass reuses it; a pipeline run recomputes it, because the cache is
# inert unless the figure tooling turns it on.
from .plotdata import memo as _memo


@_memo(name="presence")
def build_presence(breadth_df, depth_df=None, lengths=None, spec=None,
                   threshold=None, mode="breadth_pct"):
    """
    Build a boolean presence/absence matrix.

    Parameters
    ----------
    breadth_df : DataFrame  (samples x SGBs), values = % breadth coverage
    depth_df   : DataFrame  (samples x SGBs), values = mean depth
                 Not used for flat_bp mode; retained for API compatibility.
    lengths    : Series     (SGB -> genome length in bp)
                 Required for flat_bp mode.
    spec       : DetectionSpec  (preferred; overrides threshold & mode)
    threshold  : float      Threshold value (legacy interface)
    mode       : str        "breadth_pct" or "flat_bp" (legacy interface)

    Returns
    -------
    DataFrame of bool, same shape as breadth_df
    """
    if spec is not None:
        mode = spec.mode
        threshold = spec.threshold

    if threshold is None:
        raise ValueError("Must provide spec or threshold")

    if mode == "breadth_pct":
        return breadth_df >= threshold

    elif mode == "flat_bp":
        if lengths is None:
            raise ValueError("flat_bp mode requires lengths")
        # mapped_bases = (breadth% / 100) x genome_length
        # = number of distinct genomic positions covered by >=1 read
        common = sorted(set(breadth_df.columns) & set(lengths.index))
        breadth_frac = breadth_df[common] / 100.0
        mapped_bases = breadth_frac.mul(lengths[common], axis=1)
        # Reindex to match breadth_df columns (NaN for missing -> False)
        mapped_bases = mapped_bases.reindex(columns=breadth_df.columns,
                                            fill_value=0.0)
        return mapped_bases >= threshold

    else:
        raise ValueError(f"Unknown detection mode: {mode}")


def build_presence_from_data(data, spec):
    """
    Convenience wrapper: build presence matrix from a ReferenceData object.

    Parameters
    ----------
    data : ReferenceData namedtuple
    spec : DetectionSpec

    Returns
    -------
    DataFrame of bool (samples x SGBs)
    """
    return build_presence(
        breadth_df=data.breadth,
        depth_df=data.depth,
        lengths=data.lengths,
        spec=spec,
    )


def count_richness(presence_matrix):
    """Per-sample richness (number of detected SGBs)."""
    return presence_matrix.sum(axis=1)


# ════════════════════════════════════════════════════════════════════
#  STANDARD SPEC SETS
# ════════════════════════════════════════════════════════════════════

def breadth_pct_specs(thresholds=None):
    """Build a list of breadth-% DetectionSpecs."""
    if thresholds is None:
        thresholds = [1.0, 3.0, 10.0, 30.0, 50.0, 90.0]
    return [DetectionSpec("breadth_pct", t) for t in thresholds]


def flat_bp_specs(thresholds_bp=None):
    """Build a list of flat-bp DetectionSpecs."""
    if thresholds_bp is None:
        thresholds_bp = [10_000, 30_000, 100_000, 300_000, 500_000, 1_000_000]
    return [DetectionSpec("flat_bp", t) for t in thresholds_bp]


def all_specs(breadth_thresholds=None, flat_thresholds_bp=None):
    """Build the full list of specs for both modes."""
    return (breadth_pct_specs(breadth_thresholds) +
            flat_bp_specs(flat_thresholds_bp))
