#!/usr/bin/env python3
"""Replace Tables S9 and S10 in the supplement with the combined-reference values.

Both tables were built on UHGG: the document's S10 reproduces
`results/UHGG/.../health_binomial_tests_10pct.csv` (Within-study CMH) exactly for
VANISH. On the combined basis 110 of S9's 171 cells and 57 of S10's 70 change, so
this rewrites every data cell rather than reporting a diff.

Two columns are never touched: S9's "Condition" and "Comparison". The pipeline does
not produce the comparison wording, and rewriting the condition labels would risk
reordering rows against the source table.

Cells are matched positionally -- the row order in the document is used to look up
the pipeline row, so a reordered document table would be silently mismatched. The
script therefore verifies the condition label of every row before writing anything
and aborts on the first disagreement.

Formatting survives because only the text of a cell's first run is replaced.

    python tools/patch_tables_s9_s10.py                 # report only
    python tools/patch_tables_s9_s10.py --apply         # backs up, then writes in place
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
# No default: the supplement is not part of the repository, and a path baked in
# here names one machine's copy of one dated draft -- which then gets silently
# patched when someone omits --docx. Require it.
DEFAULT_DOCX = None
HEALTH = ROOT / "results" / "combined" / "section3_health" / "main"

S9_TABLE_INDEX, S9_FIRST_ROW = 8, 2      # two header rows
S10_TABLE_INDEX, S10_FIRST_ROW = 9, 1    # one header row

# S9 columns the pipeline owns. 0 (Condition) and 1 (Comparison) are left alone.
S9_SKIP = {0, 1}


def fmt(value, integer: bool = False) -> str:
    """Match the document's existing style: integers bare, fractions 3 dp with
    trailing zeros stripped (it writes 0.73, not 0.730)."""
    if isinstance(value, str):
        return value
    if integer:
        return f"{int(round(float(value)))}"
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def set_cell(cell, text: str) -> None:
    paragraphs = cell.paragraphs
    first = paragraphs[0]
    if first.runs:
        first.runs[0].text = text
        for run in first.runs[1:]:
            run.text = ""
    else:
        first.add_run(text)
    # A cell edited by hand can carry extra paragraphs (stray line breaks); drop
    # their text so the cell reads as one line like every other cell in the table.
    for extra in paragraphs[1:]:
        for run in extra.runs:
            run.text = ""


def check_labels(rows, first_row, frame, label) -> list[str]:
    problems = []
    for offset in range(len(frame)):
        row = rows[first_row + offset]
        have = row.cells[0].text.strip()
        want = str(frame.iloc[offset]["Condition"]).strip()
        if not want.startswith(have[:14]) and not have.startswith(want[:14]):
            problems.append(f"{label} row {first_row + offset + 1}: "
                            f"document says {have!r}, pipeline says {want!r}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docx", type=Path, default=DEFAULT_DOCX, required=True,
                    help="the supplement .docx to read (and, with --apply, "
                         "patch in place)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    s9 = pd.read_csv(HEALTH / "table_s9_regenerated.csv")
    s10 = pd.read_csv(HEALTH / "table_s10_regenerated.csv")
    doc = Document(args.docx)
    t9, t10 = doc.tables[S9_TABLE_INDEX], doc.tables[S10_TABLE_INDEX]

    print(f"  document : {args.docx.name}")
    print(f"  S9  table[{S9_TABLE_INDEX}]  {len(t9.rows)}x{len(t9.columns)}   "
          f"pipeline {s9.shape}")
    print(f"  S10 table[{S10_TABLE_INDEX}] {len(t10.rows)}x{len(t10.columns)}   "
          f"pipeline {s10.shape}")

    problems = (check_labels(t9.rows, S9_FIRST_ROW, s9, "S9")
                + check_labels(t10.rows, S10_FIRST_ROW, s10, "S10"))
    if problems:
        print("\n  ROW MISMATCH -- nothing written:")
        for p in problems:
            print(f"    {p}")
        return 1

    edits = []

    for offset in range(len(s9)):
        cells = t9.rows[S9_FIRST_ROW + offset].cells
        record = s9.iloc[offset]
        for ci, col in enumerate(s9.columns):
            if ci in S9_SKIP:
                continue
            integer = col in ("Studies", "Controls", "Cases")
            want = fmt(record[col], integer=integer)
            have = cells[ci].text.strip()
            if have != want:
                edits.append(("S9", S9_FIRST_ROW + offset, ci, have, want, cells[ci]))

    for offset in range(len(s10)):
        cells = t10.rows[S10_FIRST_ROW + offset].cells
        record = s10.iloc[offset]
        for ci, col in enumerate(s10.columns):
            if ci == 0:
                continue
            want = str(record[col])
            have = cells[ci].text.strip()
            if have != want:
                edits.append(("S10", S10_FIRST_ROW + offset, ci, have, want, cells[ci]))

    by_table = {}
    for table, *_ in edits:
        by_table[table] = by_table.get(table, 0) + 1
    print(f"\n  cells to rewrite: S9 {by_table.get('S9', 0)}, S10 {by_table.get('S10', 0)}")

    if not args.apply:
        print(f"\n  {'row':>4} {'col':>4}  {'was':<22} -> now")
        for table, ri, ci, have, want, _ in edits[:20]:
            print(f"  {table} {ri + 1:>4} {ci:>4}  {have[:21]:<22} -> {want}")
        if len(edits) > 20:
            print(f"  ... and {len(edits) - 20} more")
        print("\n  (report only; pass --apply to write)")
        return 0

    backup = args.docx.with_suffix(f".pre_s9s10{args.docx.suffix}")
    shutil.copy2(args.docx, backup)
    for _, _, _, _, want, cell in edits:
        set_cell(cell, want)
    doc.save(args.docx)
    print(f"\n  backup  : {backup}")
    print(f"  written : {args.docx}  ({len(edits)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
