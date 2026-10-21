#!/usr/bin/env python3
"""Make Table S9 fit the page: drop derivable columns and set a fixed layout.

S9 declares 9.12in of width on an A4 portrait page that offers 6.27in -- 46% over,
with the font already at 7-7.5pt, so there is no rescue by shrinking type. Two
blocks are cheap to remove:

  * the four `Net` columns, which are exactly `Dep - Enr` and so carry nothing the
    reader cannot recover;
  * `Comparison`, 50 characters of near-boilerplate that is almost determined by
    `Condition` and reads better as a legend line.

That leaves 14 columns at 6.18in. The removed `Comparison` wording is printed so it
can be pasted into the legend -- it is not discarded silently.

Deleting a column is not just dropping a `<w:tc>`: row 0 merges each list's three
sub-columns with `gridSpan=3`, so a deletion inside a span must decrement the span
instead. `delete_column` handles both.

Word only honours explicit widths under a fixed layout, so this also sets
`tblLayout=fixed` and writes `tcW` on every cell; leaving autofit on would let Word
re-derive its own widths and undo the work.

    python tools/format_table_s9.py                 # report only
    python tools/format_table_s9.py --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

# No default: the supplement is not part of the repository, and a path baked in
# here names one machine's copy of one dated draft -- which then gets silently
# reformatted when someone omits --docx. Require it.
DEFAULT_DOCX = None
TABLE_INDEX = 8

# Logical column indices to remove, in the original 19-column numbering.
# 1 = Comparison; 9/12/15/18 = the four Net columns.
DROP = [1, 9, 12, 15, 18]

# Widths for the surviving 14 columns, in inches. Sum = 6.18, inside the 6.27
# usable width with a little slack for border rendering.
WIDTHS = [1.00,                     # Condition (wraps to two lines; that is fine)
          0.42, 0.48, 0.38,         # Studies, Controls, Cases
          0.48, 0.46,               # Med. Log Depth D, Mean ESS Frac
          *([0.37] * 8)]            # Dep/Enr x 4 lists

FONT_PT = 7.5          # uniform; the table currently mixes 7 and 7.5
CELL_MARGIN_IN = 0.05  # Word's default 0.08 wastes 0.48in across 14 columns


def delete_column(tbl, idx: int) -> None:
    """Remove logical column `idx`, honouring horizontal merges."""
    grid = tbl.find(qn("w:tblGrid"))
    cols = grid.findall(qn("w:gridCol"))
    grid.remove(cols[idx])

    for tr in tbl.findall(qn("w:tr")):
        cursor = 0
        for tc in tr.findall(qn("w:tc")):
            span_el = tc.find(qn("w:tcPr") + "/" + qn("w:gridSpan"))
            span = int(span_el.get(qn("w:val"))) if span_el is not None else 1
            if cursor <= idx < cursor + span:
                if span > 1:
                    span_el.set(qn("w:val"), str(span - 1))
                else:
                    tr.remove(tc)
                break
            cursor += span


def set_fixed_layout(tbl) -> None:
    tblPr = tbl.tblPr
    for tag in ("w:tblLayout",):
        for el in tblPr.findall(qn(tag)):
            tblPr.remove(el)
    layout = tblPr.makeelement(qn("w:tblLayout"), {qn("w:type"): "fixed"})
    tblPr.append(layout)


def set_cell_margins(tbl, inches: float) -> None:
    tblPr = tbl.tblPr
    for el in tblPr.findall(qn("w:tblCellMar")):
        tblPr.remove(el)
    mar = tblPr.makeelement(qn("w:tblCellMar"), {})
    for side in ("left", "right"):
        node = mar.makeelement(qn(f"w:{side}"),
                               {qn("w:w"): str(int(inches * 1440)),
                                qn("w:type"): "dxa"})
        mar.append(node)
    tblPr.append(mar)


def repeat_header(row) -> None:
    trPr = row._tr.get_or_add_trPr()
    if trPr.find(qn("w:tblHeader")) is None:
        trPr.append(trPr.makeelement(qn("w:tblHeader"), {}))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docx", type=Path, default=DEFAULT_DOCX, required=True,
                    help="the supplement .docx to read (and, with --apply, "
                         "reformat in place)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    doc = Document(args.docx)
    table = doc.tables[TABLE_INDEX]
    usable = (doc.sections[0].page_width - doc.sections[0].left_margin
              - doc.sections[0].right_margin) / 914400.0

    before = sum((c.width / 914400.0) for c in table.columns if c.width)
    print(f"  table S9   : {len(table.rows)}x{len(table.columns)}")
    print(f"  usable page: {usable:.2f}in")
    print(f"  current    : {before:.2f}in  (over by {before - usable:.2f}in)")
    print(f"  target     : {sum(WIDTHS):.2f}in over {len(WIDTHS)} columns")

    comparisons = [(r.cells[0].text.strip(), r.cells[1].text.strip())
                   for r in table.rows[2:]]
    print("\n  Comparison column, for the legend:")
    for cond, comp in comparisons:
        print(f"    {cond:<22}{comp}")

    if not args.apply:
        print("\n  (report only; pass --apply to write)")
        return 0

    backup = args.docx.with_suffix(f".pre_s9fmt{args.docx.suffix}")
    shutil.copy2(args.docx, backup)

    for idx in sorted(DROP, reverse=True):
        delete_column(table._tbl, idx)

    set_fixed_layout(table._tbl)
    set_cell_margins(table._tbl, CELL_MARGIN_IN)
    table.autofit = False

    grid = table._tbl.find(qn("w:tblGrid"))
    for col, width in zip(grid.findall(qn("w:gridCol")), WIDTHS):
        col.set(qn("w:w"), str(int(width * 1440)))
    for row in table.rows:
        cursor = 0
        for cell in row._tr.findall(qn("w:tc")):
            tcPr = cell.find(qn("w:tcPr"))
            if tcPr is None:
                tcPr = cell.makeelement(qn("w:tcPr"), {})
                cell.insert(0, tcPr)
            span_el = tcPr.find(qn("w:gridSpan"))
            span = int(span_el.get(qn("w:val"))) if span_el is not None else 1
            width = sum(WIDTHS[cursor:cursor + span])
            for el in tcPr.findall(qn("w:tcW")):
                tcPr.remove(el)
            tcPr.append(tcPr.makeelement(qn("w:tcW"),
                                         {qn("w:w"): str(int(width * 1440)),
                                          qn("w:type"): "dxa"}))
            cursor += span

    for row in table.rows:
        for cell in row.cells:
            for para in cell.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(FONT_PT)

    for row in table.rows[:2]:
        repeat_header(row)

    doc.save(args.docx)
    after = sum((c.width / 914400.0) for c in table.columns if c.width)
    print(f"\n  backup  : {backup}")
    print(f"  written : {args.docx}")
    print(f"  now     : {len(table.columns)} columns, {after:.2f}in "
          f"({'fits' if after <= usable else 'STILL OVER'} {usable:.2f}in)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
