#!/usr/bin/env python3
"""Write results/<ref>/cohort_samples.csv without running the full pipeline.

`run_reference` writes the manifest itself, so this tool is only needed in two
situations:

  * A run is already in flight. Its modules were imported before the writer
    existed, so it cannot emit the manifest for its own figure stage.
  * Only the figures need regenerating and a full analysis re-run is not worth
    the hours.

It reproduces the pipeline's cohort by *calling the same functions* in the same
order -- load_reference, the pre-reinstatement depth audit (which drops
zero-depth samples), then reinstate_distinct_secondaries -- rather than
reimplementing the exclusions, so the result is identical by construction.

The depth audit's plots and CSVs are redirected to a throwaway directory so this
can be run safely alongside a live pipeline without overwriting its outputs.

    python tools/write_cohort_manifest.py --ref UHGG
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.getcwd())

_TMP_AUDIT = "depth_audit_cohort_manifest_tmp"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="UHGG",
                    help="Reference to build the manifest for (default: UHGG)")
    ap.add_argument("--keep-audit-dir", action="store_true",
                    help="Keep the throwaway depth-audit output directory")
    args = ap.parse_args()

    from persistence import config
    from persistence.cohort_manifest import write_cohort_manifest
    from persistence.depth_audit import run_depth_audit
    from persistence.loading import load_reference
    from persistence.reinstate_secondary import reinstate_distinct_secondaries
    from persistence.stats_report import StatsReport

    if args.ref not in config.REFERENCES:
        print(f"ERROR unknown reference {args.ref!r}; "
              f"expected one of {sorted(config.REFERENCES)}")
        return 2

    # Stats go to a scratch root: this tool must never touch the live run's
    # stats_report output.
    stats = StatsReport(args.ref, out_root=os.path.join("results", "_scratch"))

    print(f"Loading {args.ref} ...")
    data = load_reference(args.ref)
    if data is None:
        print(f"ERROR inputs for {args.ref} are absent — nothing to do.")
        return 1
    print(f"  after load exclusions: {len(data.meta):,} samples")

    data = run_depth_audit(data, stats, outdir_override=_TMP_AUDIT)
    print(f"  after zero-depth removal: {len(data.meta):,} samples")

    data = reinstate_distinct_secondaries(data, stats)
    print(f"  after reinstatement: {len(data.meta):,} samples")

    path = write_cohort_manifest(data, stats)

    tmp = os.path.join("results", args.ref, _TMP_AUDIT)
    if not args.keep_audit_dir and os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    scratch = os.path.join("results", "_scratch")
    if os.path.isdir(scratch):
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
