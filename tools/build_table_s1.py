#!/usr/bin/env python3
"""Build Table S1 (study summary: samples, hosts, per-host age/BMI/Unhealthy %)
without running the analysis.

The full pipeline builds Table S1 inside run_reference (`--only table_s1a`),
after loading every matrix and running whichever stages precede it. This runs
only what Table S1 depends on, with the pipeline's own functions in the
pipeline's order -- load_reference, zero-depth removal (run_depth_audit),
reinstate_distinct_secondaries -- and then table_s1a.run_table_s1a, so the table
is the one a full run would write.

The depth audit's outputs go to a throwaway directory that is removed; the
table goes to --out (default results/<ref>/, where a full run puts it).

    python tools/build_table_s1.py                         # combined, as published
    python tools/build_table_s1.py --ref UHGG --out /tmp/s1
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.getcwd())

_TMP_AUDIT = "depth_audit_table_s1_tmp"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="combined",
                    help="reference (default: combined, the one Table S1 reports)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: results/<ref>/)")
    args = ap.parse_args()

    from persistence import config
    from persistence.depth_audit import run_depth_audit
    from persistence.loading import load_reference
    from persistence.reinstate_secondary import reinstate_distinct_secondaries
    from persistence.stats_report import StatsReport
    from persistence.table_s1a import run_table_s1a

    if args.ref not in config.REFERENCES:
        print(f"ERROR unknown reference {args.ref!r}; "
              f"expected one of {sorted(config.REFERENCES)}")
        return 2

    scratch = os.path.join("results", "_scratch_table_s1")
    stats = StatsReport(args.ref, out_root=scratch)
    data = load_reference(args.ref)
    data = run_depth_audit(data, stats, outdir_override=_TMP_AUDIT)
    data = reinstate_distinct_secondaries(data, stats)
    print(f"\ncohort: {len(data.meta):,} samples, "
          f"{data.meta['Subject_ID'].astype(str).nunique():,} hosts")

    run_table_s1a(data, stats, outdir=args.out)

    shutil.rmtree(os.path.join("results", args.ref, _TMP_AUDIT), ignore_errors=True)
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
