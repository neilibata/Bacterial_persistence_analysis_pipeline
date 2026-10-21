#!/usr/bin/env python3
"""Append a backfill shard to the published Leviatan matrices.

The shard is produced on the cluster by `combine_aggregated_stats.py` and holds
the 398 samples the Leviatan matrix was missing. Merging is by **column name**,
never by position: both files carry 7,189 columns and a reordered shard would
otherwise misalign every species silently, which is exactly the class of failure
this backfill exists to correct.

Refuses to merge unless the two column sets are identical, and refuses to
overwrite a sample already present unless asked, since a duplicate row would
change the cohort without any error.

    python tools/merge_lev_backfill.py \\
        --shard-depth   resources/depth_matrix_levbackfill_20260812.csv \\
        --shard-breadth resources/coverage_matrix_levbackfill_20260812.csv \\
        --suffix 20260812

Writes new versioned matrices; the published ones are never modified in place.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence import config  # noqa: E402


def header_of(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def merge_one(published: Path, shard: Path, out: Path,
              allow_replace: bool) -> int:
    ph, sh = header_of(published), header_of(shard)
    print(f"\n  published : {published.name}  ({len(ph):,} cols)")
    print(f"  shard     : {shard.name}  ({len(sh):,} cols)")

    pset, sset = set(ph[1:]), set(sh[1:])
    if pset != sset:
        only_p, only_s = sorted(pset - sset), sorted(sset - pset)
        print(f"  COLUMN MISMATCH: {len(only_p)} only in published, "
              f"{len(only_s)} only in shard")
        for c in only_p[:5]:
            print(f"    published only: {c}")
        for c in only_s[:5]:
            print(f"    shard only    : {c}")
        return 1
    print(f"  column sets identical ({len(pset):,} species columns)")
    if ph != sh:
        print("  note: column ORDER differs; shard will be reordered to match")

    new = pd.read_csv(shard, index_col=0)
    new.index = new.index.astype(str).str.strip()
    new = new[ph[1:]]                       # reorder to the published order
    print(f"  shard rows: {len(new):,}")

    existing: set[str] = set()
    for chunk in pd.read_csv(published, usecols=[0], chunksize=200_000, dtype=str):
        existing.update(chunk.iloc[:, 0].astype(str).str.strip())
    clash = sorted(set(new.index) & existing)
    if clash:
        print(f"  {len(clash):,} shard samples ALREADY in the published matrix "
              f"(e.g. {clash[:3]})")
        if not allow_replace:
            print("  refusing to merge; pass --allow-replace to drop the "
                  "published rows in favour of the shard")
            return 1
        print("  --allow-replace: published rows for these will be dropped")

    drop = set(clash) if allow_replace else set()
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out, "w", newline="") as fh:
        fh.write(",".join(ph) + "\n")
        for chunk in pd.read_csv(published, index_col=0, chunksize=4000):
            chunk.index = chunk.index.astype(str).str.strip()
            if drop:
                chunk = chunk[~chunk.index.isin(drop)]
            if len(chunk):
                chunk.to_csv(fh, header=False)
                written += len(chunk)
        new.to_csv(fh, header=False)
        written += len(new)
    print(f"  wrote {out.name}: {written:,} rows "
          f"({written - len(new):,} published + {len(new):,} backfilled)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-depth", required=True, type=Path)
    ap.add_argument("--shard-breadth", required=True, type=Path)
    ap.add_argument("--suffix", required=True,
                    help="version tag for the output filenames")
    ap.add_argument("--allow-replace", action="store_true")
    args = ap.parse_args()

    jobs = [
        (ROOT / config.REFERENCES["lev"]["depth_file"], args.shard_depth,
         ROOT / f"meandepth_and_std_matrix_high_lev_{args.suffix}.csv"),
        (ROOT / config.REFERENCES["lev"]["breadth_file"], args.shard_breadth,
         ROOT / f"meancoverage_and_std_matrix_high_lev_{args.suffix}.csv"),
    ]
    for pub, shard, out in jobs:
        for p in (pub, shard):
            if not p.exists():
                sys.exit(f"missing: {p}")
    rc = 0
    for pub, shard, out in jobs:
        rc |= merge_one(pub, shard, out, args.allow_replace)
    if rc == 0:
        print("\n  Next:")
        print("    1. point config.REFERENCES['lev'] at the new files")
        print("    2. python tools/build_combined_matrix.py --validate")
        print("    3. python -m persistence --ref lev combined --resume")
    return rc


if __name__ == "__main__":
    sys.exit(main())
