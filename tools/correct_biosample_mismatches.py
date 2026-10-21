#!/usr/bin/env python3
"""Correct M1 BioSample accessions that disagree with ENA and would break the
one-sample-per-BioSample rule (config.ONE_SAMPLE_PER_BIOSAMPLE).

The rule trusts M1's BioSample column, so a row carrying another sample's
BioSample makes two people look like one specimen and one of them is dropped.
Each correction is listed, with its ENA evidence, in
docs/biosample_corrections.csv; a row is changed only while M1 still holds the
recorded wrong value. (tools/fix_biosample_accessions.py is a different job:
it replaces submitter sample names with accessions.)

    python tools/correct_biosample_mismatches.py            # report only
    python tools/correct_biosample_mismatches.py --apply    # rewrite M1 (backs up first)

Not corrected here: 216 HMP_2012_USA / NielsenHB_2014_DNK rows whose M1
BioSample differs from the one ENA now reports (re-registered accessions).
None of them shares a BioSample with another cohort row under either value, so
they do not affect the rule.
"""
from __future__ import annotations
import argparse, os, shutil, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from persistence import config  # noqa: E402

CORRECTIONS = ROOT / "docs" / "biosample_corrections.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    fix = pd.read_csv(CORRECTIONS, dtype=str)
    m = pd.read_csv(config.META_FILE, low_memory=False)
    col = config.BIOSAMPLE_COL
    n_changed = 0
    for _, r in fix.iterrows():
        rows = m.index[m["Chosen_match_ID"].astype(str).eq(r.Chosen_match_ID)]
        if len(rows) != 1:
            raise SystemExit(f"ERROR: {r.Chosen_match_ID} matches {len(rows)} M1 rows")
        cur = str(m.at[rows[0], col]).strip()
        if cur == r.BioSample_ENA:
            print(f"  {r.Chosen_match_ID}: already {cur}")
            continue
        if cur != r.BioSample_M1:
            raise SystemExit(f"ERROR: {r.Chosen_match_ID} has {cur}, expected "
                             f"{r.BioSample_M1} -- M1 changed; re-check against ENA")
        print(f"  {r.Chosen_match_ID} ({r['Sample ID']}): {cur} -> {r.BioSample_ENA}")
        m.at[rows[0], col] = r.BioSample_ENA
        n_changed += 1

    bs = m[col].astype(str).str.strip()
    print(f"  M1 rows sharing a BioSample after correction: "
          f"{int(bs.duplicated(keep=False).sum())} "
          f"(the loader resolves these; Carter's are expected)")

    if args.apply and n_changed:
        src = Path(config.META_FILE)
        bak = src.with_suffix(src.suffix + ".pre_biosample_fix")
        if not bak.exists():
            shutil.copy2(src, bak); print(f"\nbacked up -> {bak}")
        m.to_csv(src, index=False); print(f"applied to {src}")
    elif not args.apply:
        print("\n(report only; rerun with --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
