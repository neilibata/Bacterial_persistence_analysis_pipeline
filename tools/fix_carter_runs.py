#!/usr/bin/env python3
"""Give each Carter library its own sequencing run in M1.

The defect. 21 M1 Carter rows share a `Chosen_match_ID` with another row, so two
rows resolve to one matrix row. loading.py already resolves this safely by
keeping the larger `Total bases`, but that discards a real sample.

The cause. Carter (PRJEB49206) extracted some stools by two protocols -- MoBio
and phenol-chloroform -- and the submitter registered both libraries under ONE
BioSample. ENA gives each library its own run and its own `library_name`, and
that name is exactly M1's `Sample ID`. M1 instead lumped both runs into one row's
`Paired SRR List` and pointed both rows at the same run, so the MoBio and PheChl
libraries were treated as replicates of one sample rather than two libraries.

The fix. Match `Sample ID` to ENA `library_name`, and point each row at the run
ENA assigns to that library, where that run is in the matrix. Rebuild the run
lists so a library claims only its own run. Normalise the 22 rows sitting in the
bare `CarterM_2023` folder into the country folder their own `Population ID`,
`Country` and `Lifestyle` already say they belong to -- all Carter folders share
`Study = 8`, and loading.py derives CMH strata from `Study`, so this changes no
stratification.

Rows whose library has no run in the matrix cannot be measured and are dropped,
with each one reported. That is 4 rows; without this fix loading.py drops 21.

Inputs, both cached in docs/ so this runs offline:
  docs/ena_runs_PRJEB49206.csv          ENA read_run table for the Carter project
  docs/matrix_sample_index_combined.txt sample ids present in the combined matrix

    python tools/fix_carter_runs.py            # report only
    python tools/fix_carter_runs.py --apply    # rewrite M1 (backs up first)
"""
from __future__ import annotations
import argparse, re, shutil, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)
from persistence import config  # noqa: E402

ENA_TABLE = ROOT / "docs" / "ena_runs_PRJEB49206.csv"
MATRIX_IDX = ROOT / "docs" / "matrix_sample_index_combined.txt"
PREFIX = "CarterM_2023"
# Population ID -> the folder that population belongs in
POP_FOLDER = {9: "CarterM_2023_NPL", 10: "CarterM_2023_TZA", 11: "CarterM_2023_USA"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    for p in (ENA_TABLE, MATRIX_IDX):
        if not p.exists():
            raise SystemExit(f"ERROR: missing input {p}")
    ena = pd.read_csv(ENA_TABLE, dtype=str)
    idx = {l.strip() for l in open(MATRIX_IDX) if l.strip()}
    m = pd.read_csv(config.META_FILE, low_memory=False)
    is_carter = m["StudyFolder"].astype(str).str.startswith(PREFIX)
    print(f"M1 {len(m):,} rows; Carter {int(is_carter.sum())}")
    print(f"ENA runs {len(ena):,}; distinct library_name {ena.library_name.nunique():,}")

    ena["base_count"] = pd.to_numeric(ena["base_count"], errors="coerce")
    ena["_in"] = ena.run_accession.isin(idx)
    # one run per library for this project; keep the deepest if that ever changes
    best = (ena[ena._in].sort_values("base_count", ascending=False)
                        .drop_duplicates("library_name"))
    lib2run = dict(zip(best.library_name, best.run_accession))

    sid = m.loc[is_carter, "Sample ID"].astype(str)
    matched = sid.map(lib2run)
    print(f"\n  Sample ID matched to a library with a matrix run: "
          f"{int(matched.notna().sum())} of {len(sid)}")

    # ---- report ----
    cur = m.loc[is_carter, "Chosen_match_ID"].astype(str)
    repoint = matched.notna() & matched.ne(cur)
    drop = matched.isna() & m.loc[is_carter, "Chosen_match_ID"].astype(str).duplicated(keep=False)
    unmatched = matched.isna() & ~drop
    print(f"  rows to repoint  : {int(repoint.sum())}")
    print(f"  rows to drop     : {int(drop.sum())}  (library has no run in the matrix")
    print(f"                       AND the row only resolves by sharing another's run)")
    print(f"  left as-is       : {int(unmatched.sum())}")
    if drop.any():
        print("\n  dropping:")
        for i in m.index[is_carter][drop.values]:
            print(f"    {m.at[i,'Sample ID']}  (was borrowing {m.at[i,'Chosen_match_ID']})")
    if unmatched.any():
        print("\n  no ENA library match, left untouched:")
        for i in m.index[is_carter][unmatched.values]:
            print(f"    {m.at[i,'Sample ID']!r}  Chosen={m.at[i,'Chosen_match_ID']}")

    # ---- apply ----
    out = m.copy()
    ci = out.index[is_carter]
    out.loc[ci[repoint.values], "Chosen_match_ID"] = matched[repoint].values
    # each library owns exactly its own run
    for i, lname in zip(ci, sid.values):
        r = lib2run.get(lname)
        if r is None:
            continue
        out.at[i, "SRA_Accession"] = r
        out.at[i, "Paired SRR List"] = r
        out.at[i, "Secondary_run_IDs"] = pd.NA
        out.at[i, "NumPairedRuns"] = 1
    # folder normalisation, driven by Population ID
    base = out["StudyFolder"].astype(str).eq(PREFIX)
    pop = pd.to_numeric(out["Population ID"], errors="coerce")
    moved = 0
    for p, folder in POP_FOLDER.items():
        sel = base & pop.eq(p)
        moved += int(sel.sum())
        out.loc[sel, "StudyFolder"] = folder
    print(f"\n  folder-normalised {moved} row(s) out of the bare '{PREFIX}' folder")
    out = out.drop(index=ci[drop.values])

    carter = out["StudyFolder"].astype(str).str.startswith(PREFIX)
    cm = out.loc[carter, "Chosen_match_ID"].astype(str)
    print(f"\n── after ──")
    print(f"  rows {len(m):,} -> {len(out):,}")
    print(f"  Carter rows {int(is_carter.sum())} -> {int(carter.sum())}")
    print(f"  Carter Chosen_match_ID collisions: {len(cm)-cm.nunique()}")
    print(f"  all Carter ids present in the matrix: "
          f"{int(cm.isin(idx).sum())} of {len(cm)}")
    print(f"  folders: {out.loc[carter,'StudyFolder'].value_counts().to_dict()}")
    allcm = out["Chosen_match_ID"].astype(str)
    print(f"  WHOLE-FILE Chosen_match_ID collisions: {len(allcm)-allcm.nunique()}")

    if args.apply:
        src = Path(config.META_FILE)
        bak = src.with_suffix(src.suffix + ".pre_carter_runs")
        if not bak.exists():
            shutil.copy2(src, bak); print(f"\nbacked up -> {bak}")
        out.to_csv(src, index=False); print(f"applied to {src}")
        print("NOTE: re-run tools/build_subject_ids.py --apply then "
              "tools/build_verified_m1.py --apply, since row set and folders changed.")
    else:
        print("\n(report only; rerun with --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
