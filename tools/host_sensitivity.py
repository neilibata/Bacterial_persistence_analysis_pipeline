#!/usr/bin/env python3
"""Run the host-sensitivity stage without running the analysis.

A full run does this itself (stage `host_sensitivity`, after section 2). This
loads only the cohort -- load_reference, zero-depth removal, reinstatement, in
the pipeline's order -- and calls host_sensitivity.run_host_sensitivity, so the
numbers are the ones a full run writes: the Figure 2 richness ratio with all
samples (sample and host bootstrap) and with one sample per host, and
VANISH+/BloSSUM+/H+ membership with one sample per host and under sample versus
host bootstrap. The list check compares against results/<ref>/section2_differential
lists when they exist.

    python tools/host_sensitivity.py                      # combined -> results/combined/host_sensitivity/
    python tools/host_sensitivity.py --ref UHGG --out /tmp/hs
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.getcwd())

_TMP_AUDIT = "depth_audit_host_sens_tmp"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="combined")
    ap.add_argument("--out", default=None,
                    help="output directory (default results/<ref>/host_sensitivity/)")
    ap.add_argument("--boot-richness", type=int, default=None)
    ap.add_argument("--boot-lists", type=int, default=None)
    args = ap.parse_args()

    from persistence import host_sensitivity as hs
    from persistence.depth_audit import run_depth_audit
    from persistence.loading import load_reference
    from persistence.reinstate_secondary import reinstate_distinct_secondaries
    from persistence.stats_report import StatsReport

    scratch = os.path.join("results", "_scratch_host_sens")
    stats = StatsReport(args.ref, out_root=scratch)
    data = load_reference(args.ref)
    data = run_depth_audit(data, stats, outdir_override=_TMP_AUDIT)
    data = reinstate_distinct_secondaries(data, stats)
    hs.run_host_sensitivity(
        data, stats, outdir=args.out,
        n_boot_richness=args.boot_richness or hs.N_BOOT_RICHNESS,
        n_boot_lists=args.boot_lists or hs.N_BOOT_LISTS)
    shutil.rmtree(os.path.join("results", args.ref, _TMP_AUDIT), ignore_errors=True)
    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
