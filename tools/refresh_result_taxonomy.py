#!/usr/bin/env python3
"""Bring the taxonomy columns of the result CSVs up to GTDB r232.

The workbook was re-aligned (tools/apply_gtdbtk_taxonomy.py) and the figures
were redrawn from their data objects, but the result CSVs still carry the
lineages they were written with, so a table and the figure drawn from the same
numbers disagree about what a genome is called.

Only annotation moves. Nothing in this pipeline groups, weights or tests by
taxonomic rank, so every statistic in these files is left exactly as it was --
the files round-trip byte-identically when no name changes.

Columns rewritten, where a file has them:

    taxonomy / gtdb_taxonomy   the full lineage, from the workbook
    taxonomy_short             last two filled ranks  (section 3 drivers)
    lowest_rank                last filled rank       (section 2 hadza, S4)

The lineage comes from the pipeline's own resolver, so the values written here
are the ones a re-run produces rather than a second opinion about them.

    python tools/refresh_result_taxonomy.py --dry-run
    python tools/refresh_result_taxonomy.py

Originals are copied to archive/pre_gtdbtk_r232/results/ first.
"""
from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BACKUP = ROOT / "archive" / "pre_gtdbtk_r232" / "results"

#: the lineage column a file uses, in the order we prefer to find it
TAX_COLS = ("gtdb_taxonomy", "taxonomy")
#: the column naming the genome, in the order the pipeline writes them
KEY_COLS = ("taxon", "Taxon", "sgb_id", "sgb")


def load_map():
    """{genome id: lineage}, from the pipeline's own workbook resolver."""
    from persistence.section2_pooling import _taxonomy_from_combined_metadata
    return {k: v for k, v in _taxonomy_from_combined_metadata().items() if v}


def filled_ranks(lineage):
    return [p.strip() for p in str(lineage).split(";")
            if p.strip() and not p.strip().endswith("__")]


def short_label(lineage, n_ranks=2):
    """Mirrors section3_health._gtdb_short."""
    if not lineage or str(lineage) in ("", "nan"):
        return ""
    parts = filled_ranks(lineage)
    return "; ".join(parts[-n_ranks:]) if parts else ""


def lowest_rank(lineage):
    """Mirrors section2_differential._extract_lowest_rank."""
    if not lineage or str(lineage) in ("", "nan"):
        return ""
    parts = filled_ranks(lineage)
    return parts[-1] if parts else ""


def targets():
    out = []
    for p in sorted(glob.glob(str(ROOT / "results/**/*.csv"), recursive=True)):
        try:
            head = pd.read_csv(p, nrows=0)
        except Exception:
            continue
        cols = set(head.columns)
        if not (cols & set(TAX_COLS)):
            continue
        key = next((c for c in KEY_COLS if c in cols), None)
        if key:
            out.append((Path(p), key,
                        next(c for c in TAX_COLS if c in cols), cols))
    return out


def refresh(path, key, tax_col, cols, tmap, dry_run):
    # str throughout: the numbers are not ours to reformat, and a file with
    # no name changes then comes out byte-identical to the one that went in.
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    before = df[tax_col].astype(str).copy()
    new = df[key].astype(str).map(lambda k: tmap.get(k, ""))

    # Only overwrite where the workbook actually has a lineage; a value it
    # cannot supply came from the phylo lists and is left alone.
    keep = new == ""
    after = new.where(~keep, before)

    changed = int(((before != "") & (after != before)).sum())
    added = int(((before == "") & (after != "")).sum())
    if changed == 0 and added == 0:
        return dict(changed=0, added=0, derived=[])

    derived = []
    df[tax_col] = after
    if "taxonomy_short" in cols:
        df["taxonomy_short"] = after.map(short_label)
        derived.append("taxonomy_short")
    if "lowest_rank" in cols:
        df["lowest_rank"] = after.map(lowest_rank)
        derived.append("lowest_rank")

    if not dry_run:
        rel = path.relative_to(ROOT / "results")
        dest = BACKUP / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(path, dest)
        df.to_csv(path, index=False)
    return dict(changed=changed, added=added, derived=derived)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    tmap = load_map()
    print(f"  {len(tmap):,} lineages from combined_sgb_metadata.xlsx\n")

    files = targets()
    tot_c = tot_a = 0
    touched = []
    for path, key, tax_col, cols in files:
        r = refresh(path, key, tax_col, cols, tmap, a.dry_run)
        tot_c += r["changed"]
        tot_a += r["added"]
        if r["changed"] or r["added"]:
            touched.append(path)
            extra = f"  (+{', '.join(r['derived'])})" if r["derived"] else ""
            print(f"  {str(path.relative_to(ROOT)):74} "
                  f"{r['changed']:>4} changed {r['added']:>4} filled{extra}")

    print(f"\n  {len(files)} files scanned, {len(touched)} rewritten")
    print(f"  {tot_c:,} lineages changed, {tot_a:,} newly filled")
    if a.dry_run:
        print("  dry run; nothing written")
    else:
        print(f"  originals under {BACKUP.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
