#!/usr/bin/env python3
"""Fill in missing GTDB lineages in existing section-2 result CSVs.

`_build_taxonomy_map` used to index only `GCA_*` and `MGYG*` identifiers, from a
UHGG-only crosswalk. On the combined reference that leaves every Leviatan-derived
`Rep_*` SGB, and several hundred UHGG ones, with an empty `taxonomy` column --
so the supplementary tables render a bare accession where they should name a
species. The map now also reads `combined_sgb_metadata.xlsx`
(`section2_pooling._taxonomy_from_combined_metadata`), but the CSVs already on
disk were written before that.

This backfills them in place rather than forcing a full section-2 rerun, which
would recompute identical numbers. Only the `taxonomy` column is touched, and
only where it is currently empty -- an existing annotation is never overwritten,
so this cannot silently change a published label.

    python tools/backfill_taxonomy.py --ref combined            # report only
    python tools/backfill_taxonomy.py --ref combined --apply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence.section2_pooling import _build_taxonomy_map  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="combined")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--apply", action="store_true",
                    help="Rewrite the CSVs (default is a dry run).")
    args = ap.parse_args()

    main_dir = args.results_dir / args.ref / "section2_differential" / "main"
    if not main_dir.exists():
        sys.exit(f"no such directory: {main_dir}")

    tax = _build_taxonomy_map(None)
    if not tax:
        sys.exit("taxonomy map is empty -- is combined_sgb_metadata.xlsx present?")

    files = sorted(p for p in main_dir.glob("*.csv"))
    total_filled, total_missing, still = 0, 0, set()
    touched = []

    for path in files:
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            # Expected: on lev/chimp the VANISH/BloSSUM lists are UHGG accessions
            # with no counterpart in the catalogue, so the pooling files are empty.
            continue
        if "taxon" not in df.columns or "taxonomy" not in df.columns:
            continue
        blank = df["taxonomy"].isna() | (df["taxonomy"].astype(str).str.strip() == "")
        if not blank.any():
            continue
        ids = df.loc[blank, "taxon"].astype(str).str.strip()
        filled = ids.map(tax)
        n_fill = int(filled.notna().sum())
        n_miss = int(filled.isna().sum())
        still |= set(ids[filled.isna()])
        total_filled += n_fill
        total_missing += n_miss
        if n_fill:
            touched.append((path, n_fill, n_miss))
            if args.apply:
                df.loc[blank, "taxonomy"] = filled.values
                df.to_csv(path, index=False)

    print(f"  taxonomy map     : {len(tax):,} identifiers")
    print(f"  files with gaps  : {len(touched)}")
    print(f"  rows fillable    : {total_filled:,}")
    print(f"  rows still empty : {total_missing:,}")
    if touched:
        print(f"\n  {'file':<52}{'filled':>8}{'left':>7}")
        for path, n_fill, n_miss in touched:
            print(f"  {path.name:<52}{n_fill:>8,}{n_miss:>7}")
    if still:
        print(f"\n  unresolved taxa ({len(still)}): {sorted(still)}")
        print("  These carry no gtdb_lineage in combined_sgb_metadata.xlsx and have "
              "no\n  cross-catalogue partner to borrow one from -- an upstream "
              "catalogue gap,\n  not a pipeline fault.")
    print(f"\n  {'rewritten' if args.apply else 'dry run -- pass --apply to write'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
