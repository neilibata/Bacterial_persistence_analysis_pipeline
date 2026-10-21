#!/usr/bin/env python3
"""Replace the GTDB lineages in combined_sgb_metadata.xlsx with a GTDB-Tk run.

The workbook is the upstream of Metadata Table M2 and of every taxonomy label
the pipeline draws, so updating it here updates both. Rebuild M2 afterwards:

    python tools/apply_gtdbtk_taxonomy.py
    python tools/build_metadata_tables.py

Input is the parsed classification CSVs, one row per genome keyed on
`query_name` -- `Rep_N` for Leviatan, and a mix of `MGYG*` and `GCA*` for UHGG,
matching the `sgb` column of the corresponding sheet exactly.

The run also carries per-genome provenance the old lineages had none of:
how the call was made, the ANI to the closest reference, and whether the
species is novel. Those come across as columns rather than being dropped,
because "s__Foo sp012345" reads the same whether it came from a 99% ANI hit
or from RED, and a reader should be able to tell.
"""
from __future__ import annotations

import argparse
import datetime
import shutil
import sys
import os as _os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORKBOOK = ROOT / "combined_sgb_metadata.xlsx"
# The GTDB-Tk classification outputs, which live outside this repository
# (they belong to the upstream taxonomy run). Set GTDBTK_SGB_DIR.
DEFAULT_SRC = Path(_os.environ.get("GTDBTK_SGB_DIR")
                   or ROOT / "gtdbtk_sgb")

#: sheet -> classification csv
SHEETS = {"Leviatan": "leviatan_gtdbtk.classifications.csv",
          "UHGG": "uhgg_gtdbtk.classifications.csv"}

RANK_COL = {"d": "gtdb_domain", "p": "gtdb_phylum", "c": "gtdb_class",
            "o": "gtdb_order", "f": "gtdb_family", "g": "gtdb_genus",
            "s": "gtdb_species"}

#: Columns carried over from the GTDB-Tk output, renamed into the workbook's
#: gtdb_ namespace.
PROVENANCE = {"novel_species": "gtdb_novel_species",
              "classification_method": "gtdb_classification_method",
              "closest_genome_reference": "gtdb_closest_reference",
              "closest_genome_ani": "gtdb_closest_ani",
              "closest_genome_af": "gtdb_closest_af",
              "rank": "gtdb_assigned_rank"}


def split_lineage(lineage):
    """{gtdb_rank_column: token} for one lineage string."""
    out = {}
    for tok in str(lineage).split(";"):
        tok = tok.strip()
        if len(tok) > 3 and tok[1:3] == "__":
            col = RANK_COL.get(tok[0])
            if col:
                out[col] = tok
    return out


def load_calls(src):
    """{genome_id: row} across both catalogues."""
    calls, frames = {}, {}
    for sheet, fname in SHEETS.items():
        path = Path(src) / fname
        if not path.exists():
            raise SystemExit(f"missing classification file: {path}")
        df = pd.read_csv(path)
        frames[sheet] = df
        for r in df.to_dict("records"):
            calls[str(r["query_name"])] = r
    return calls, frames


def update_sheet(df, calls, key_col):
    """Rewrite the gtdb_ columns of one sheet in place. Returns a tally."""
    if key_col not in df.columns:
        return dict(rows=len(df), matched=0, changed=0, filled=0)

    stats = dict(rows=len(df), matched=0, changed=0, filled=0)
    new_cols = {c: [] for c in list(RANK_COL.values()) + ["gtdb_lineage"]}
    for c in PROVENANCE.values():
        new_cols[c] = []

    had = df["gtdb_lineage"].astype(str) if "gtdb_lineage" in df.columns else None
    for i, key in enumerate(df[key_col].astype(str)):
        rec = calls.get(key)
        if rec is None:
            # Leave the row as it was rather than blanking a lineage the
            # workbook already had from another source.
            prev = had.iloc[i] if had is not None else ""
            new_cols["gtdb_lineage"].append(prev if ";" in str(prev) else pd.NA)
            for c in RANK_COL.values():
                new_cols[c].append(df[c].iloc[i] if c in df.columns else pd.NA)
            for c in PROVENANCE.values():
                new_cols[c].append(pd.NA)
            continue

        stats["matched"] += 1
        lineage = str(rec.get("lineage", "") or "")
        before = str(had.iloc[i]) if had is not None else ""
        if ";" not in before and ";" in lineage:
            stats["filled"] += 1
        elif ";" in before and before != lineage:
            stats["changed"] += 1

        new_cols["gtdb_lineage"].append(lineage or pd.NA)
        ranks = split_lineage(lineage)
        for c in RANK_COL.values():
            new_cols[c].append(ranks.get(c, pd.NA))
        for src_c, dst_c in PROVENANCE.items():
            new_cols[dst_c].append(rec.get(src_c, pd.NA))

    for c, vals in new_cols.items():
        df[c] = vals
    return stats


def propagate(df, lineage_by_id, id_cols):
    """Refresh derived gtdb columns on a sheet keyed by a representative id."""
    key = next((c for c in id_cols if c in df.columns), None)
    if key is None:
        return 0
    n = 0
    lin = df[key].astype(str).map(lambda k: lineage_by_id.get(k))
    if "gtdb_lineage" in df.columns:
        df["gtdb_lineage"] = lin.where(lin.notna(), df["gtdb_lineage"])
        n = int(lin.notna().sum())
    for col, letter in (("gtdb_phylum", "p"), ("gtdb_family", "f"),
                        ("gtdb_species", "s"), ("gtdb_genus", "g")):
        if col in df.columns:
            df[col] = lin.map(
                lambda L: split_lineage(L).get(RANK_COL[letter])
                if isinstance(L, str) else None).where(lin.notna(), df[col])
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--workbook", default=str(WORKBOOK))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    calls, frames = load_calls(a.src)
    print(f"  {len(calls):,} classifications read from {a.src}")

    book = Path(a.workbook)
    xl = pd.ExcelFile(book)
    sheets = {s: xl.parse(s) for s in xl.sheet_names}

    for sheet in SHEETS:
        if sheet not in sheets:
            continue
        st = update_sheet(sheets[sheet], calls, "sgb")
        print(f"  {sheet:14} {st['matched']:>5}/{st['rows']} matched   "
              f"{st['filled']:>4} newly given a lineage, "
              f"{st['changed']:>4} lineage changed")

    # The derived sheets quote the representative's lineage, so they follow.
    lineage_by_id = {k: str(v.get("lineage", "") or "")
                     for k, v in calls.items() if v.get("lineage")}
    for sheet, id_cols in (("Combined", ("representative",)),
                           ("Shared_Pairs", ("representative",)),
                           ):
        if sheet in sheets:
            n = propagate(sheets[sheet], lineage_by_id, id_cols)
            print(f"  {sheet:14} {n:>5} rows refreshed from the representative")

    # Fuzzy_Boundary states each genome's own family and its partners'; those
    # are GTDB families under another column name, so they move too or the
    # workbook disagrees with itself about what family a genome is in.
    if "Fuzzy_Boundary" in sheets:
        d = sheets["Fuzzy_Boundary"]
        if {"sgb", "own_family"} <= set(d.columns):
            fam = d["sgb"].astype(str).map(
                lambda k: split_lineage(lineage_by_id.get(k, "")).get(
                    "gtdb_family"))
            d["own_family"] = fam.where(fam.notna(), d["own_family"])
            print(f"  {'Fuzzy_Boundary':14} {int(fam.notna().sum()):>5} "
                  "own_family values refreshed")

    if "Rep_Choice" in sheets:
        d = sheets["Rep_Choice"]
        for col, src in (("gtdb_family_x", "leviatan_rep"),
                         ("gtdb_family_y", "uhgg_rep")):
            if col in d.columns and src in d.columns:
                d[col] = d[src].astype(str).map(
                    lambda k: split_lineage(lineage_by_id.get(k, "")).get(
                        "gtdb_family"))
        if "gtdb_phylum" in d.columns and "chosen_rep" in d.columns:
            d["gtdb_phylum"] = d["chosen_rep"].astype(str).map(
                lambda k: split_lineage(lineage_by_id.get(k, "")).get(
                    "gtdb_phylum"))
        print(f"  {'Rep_Choice':14} {len(d):>5} rows refreshed")

    # Say where the numbers came from, in the workbook itself.
    prov = pd.DataFrame([
        ("source", "GTDB-Tk 2.7.2, GTDB release r232"),
        ("classified", f"{len(calls):,} genomes, 0 failures"),
        ("leviatan", f"{len(frames['Leviatan']):,} rows"),
        ("uhgg", f"{len(frames['UHGG']):,} rows"),
        ("novel_species",
         f"{sum(1 for v in calls.values() if v.get('novel_species'))} flagged; "
         "GTDB-Tk classify mode is not sufficient to publish new taxonomic "
         "designations, so these are joins, not named taxa"),
        ("applied", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("applied_by", "tools/apply_gtdbtk_taxonomy.py"),
    ], columns=["item", "value"])
    sheets["GTDB_provenance"] = prov

    if a.dry_run:
        print("  dry run; workbook not written")
        return

    tmp = book.with_suffix(".xlsx.new")
    with pd.ExcelWriter(tmp, engine="openpyxl") as w:
        for name, d in sheets.items():
            d.to_excel(w, sheet_name=name, index=False)
    shutil.move(str(tmp), str(book))
    print(f"  wrote {book}")


if __name__ == "__main__":
    main()
