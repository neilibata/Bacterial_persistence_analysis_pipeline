#!/usr/bin/env python3
"""Correct the one row whose `Sample ID` names the wrong sequencing run.

`Sample ID` 11358.1128 (CarterM_2023_TZA) mixes identifiers from two runs of one
BioSample, SAMEA4765588:

  ERR2674226        5,295,099 bp   library_name 11358.1128
  ERR7803603  209,223,196,400 bp   library_name Hadza_MoBio_hadza-A-D_C_6_1128

M1 recorded `Total bases` = 209,223,196,400 and `SRA_Accession` = ERR7803603 --
the deep run -- but took `Sample ID` from the 5 Mbp run. That made it the single
209 Gbp outlier in the cohort (the 99th percentile is 29.6 Gbp) and the only
Carter `Sample ID` with no matching ENA library, and it left the row named after
data it does not contain. `number_reads` x `median_read_length` confirms the
depth belongs to ERR7803603, so the DEPTH is right and the NAME was wrong.

`Chosen_match_ID` stays as the BioSample accession SAMEA4765588: ERR7803603 is
not in the matrices, SAMEA4765588 is, and 260 matrix rows are legitimately keyed
by a BioSample accession. Repointing it would orphan the row.

    python tools/fix_biosample_keyed_row.py            # report only
    python tools/fix_biosample_keyed_row.py --apply
"""
from __future__ import annotations
import argparse, io, shutil, sys, os, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
from persistence import config  # noqa: E402

WRONG_SAMPLE_ID = "11358.1128"
BIOSAMPLE = "SAMEA4765588"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    m = pd.read_csv(config.META_FILE, low_memory=False)
    sel = m["Sample ID"].astype(str).str.strip().eq(WRONG_SAMPLE_ID)
    if not sel.any():
        print(f"Sample ID {WRONG_SAMPLE_ID!r} not present -- already corrected"); return 0
    i = m.index[sel][0]
    print(f"row: Chosen_match_ID={m.at[i,'Chosen_match_ID']}  "
          f"SRA_Accession={m.at[i,'SRA_Accession']}  "
          f"Total bases={m.at[i,'Total bases']:,}")

    # resolve the true library from ENA rather than hardcoding it
    q = urllib.parse.urlencode({"result": "read_run",
        "query": f"sample_accession={BIOSAMPLE}",
        "fields": "run_accession,base_count,library_name", "format": "tsv",
        "limit": "0"})
    with urllib.request.urlopen(
            "https://www.ebi.ac.uk/ena/portal/api/search?" + q, timeout=60) as r:
        ena = pd.read_csv(io.StringIO(r.read().decode()), sep="\t", dtype=str)
    ena["base_count"] = pd.to_numeric(ena["base_count"], errors="coerce")
    print(f"\nENA runs for {BIOSAMPLE}:")
    for _, r in ena.sort_values("base_count", ascending=False).iterrows():
        print(f"  {r.run_accession}  {r.base_count:>18,.0f} bp  {r.library_name}")

    tb = pd.to_numeric(pd.Series([m.at[i, "Total bases"]]), errors="coerce").iloc[0]
    match = ena[ena.base_count.eq(tb)]
    if match.empty:
        # fall back to the deepest run, reporting the discrepancy
        match = ena.sort_values("base_count", ascending=False).head(1)
        print(f"\n  [note] no run matches Total bases exactly; using the deepest")
    run, lib = match.iloc[0].run_accession, match.iloc[0].library_name
    print(f"\n  Total bases matches {run} -> library {lib!r}")
    print(f"  correcting Sample ID: {WRONG_SAMPLE_ID!r} -> {lib!r}")
    print(f"  SRA_Accession stays {m.at[i,'SRA_Accession']} (already {run})")
    print(f"  Chosen_match_ID stays {BIOSAMPLE} -- {run} is not in the matrices")

    out = m.copy()
    out.at[i, "Sample ID"] = lib
    # Subject_ID embeds nothing from Sample ID for this row, but re-check
    print(f"\n  Sample ID now unique in its study: "
          f"{not out[out.StudyFolder.eq(out.at[i,'StudyFolder'])]['Sample ID'].duplicated().any()}")

    if a.apply:
        src = Path(config.META_FILE)
        bak = src.with_suffix(src.suffix + ".pre_biosample_row")
        if not bak.exists():
            shutil.copy2(src, bak); print(f"  backed up -> {bak}")
        out.to_csv(src, index=False); print(f"  applied to {src}")
    else:
        print("  (report only; rerun with --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
