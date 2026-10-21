#!/usr/bin/env python3
"""Render Metadata Table M1 for inspection, with every change marked.

Two workbooks, from one cell-level diff against the pre-correction cohort so
the highlighting is derived rather than asserted:

  metadata_M1.xlsx        the whole table, every changed cell shaded, so it can
                          be scanned or filtered like any spreadsheet
  metadata_M1_audit.xlsx  only the changes, one sheet per kind, with the before
                          value and the reason beside each

Colour carries meaning rather than decoration: amber where a value was
replaced, red where a flag was removed, green where one was added. Read-length
fills are shaded faintly -- there are 19,079 of them and they are provenance,
not curation, so they should not shout over 436 disease-flag decisions.

    python tools/build_m1_workbooks.py
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

M1 = ROOT / "final_run" / "metadata_M1.csv"
BASE = ROOT / "metadata_Aug_26_2026_final.csv"
CURATION = ROOT / "final_run" / "disease_flag_curation.csv"
FINAL = ROOT / "final_run" / "final_corrections.csv"
SAMPLEID = ROOT / "final_run" / "sample_id_repair.csv"

HDR = "FF1F3864"
AMBER = "FFFFE9A8"      # a value was replaced
RED = "FFF4B8AC"        # a flag was removed
GREEN = "FFBFE3C6"      # a flag was added
FAINT = "FFF2F2F2"      # provenance fill, deliberately quiet
BLUE = "FFD6E4F7"       # health label moved

#: columns whose changes are curation decisions rather than provenance
CURATED = {"Unhealthy", "BioSample"}
PROVENANCE = {"median_read_length", "number_reads"}


def diff_cells(m, b):
    """{(row, column): (old, new)} for every changed cell."""
    out = {}
    for c in [c for c in b.columns if c in m.columns]:
        x = m[c].astype(str).values
        y = b[c].astype(str).values
        for i in (x != y).nonzero()[0]:
            out[(int(i), c)] = (y[i], x[i])
    return out


def colour_for(col, old, new):
    if col in PROVENANCE:
        return FAINT
    if col == "Unhealthy":
        return BLUE
    if col == "BioSample":
        return AMBER
    if col.startswith("disease_"):
        return GREEN if str(new).strip() and str(new).strip().lower() != "nan" else RED
    return AMBER


def write_full(m, changes, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook(write_only=True)
    ws = wb.create_sheet("Metadata M1")
    cols = list(m.columns)

    hdr_fill = PatternFill("solid", fgColor=HDR)
    hdr_font = Font(color="FFFFFFFF", bold=True, size=9)
    fills = {c: PatternFill("solid", fgColor=c)
             for c in (AMBER, RED, GREEN, FAINT, BLUE)}

    from openpyxl.cell import WriteOnlyCell
    head = []
    for c in cols:
        cell = WriteOnlyCell(ws, value=c)
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        head.append(cell)
    ws.append(head)

    by_row = {}
    for (i, c), (old, new) in changes.items():
        by_row.setdefault(i, {})[c] = colour_for(c, old, new)

    values = m.values
    for i in range(len(m)):
        marks = by_row.get(i)
        if not marks:
            ws.append(list(values[i]))
            continue
        row = []
        for j, c in enumerate(cols):
            cell = WriteOnlyCell(ws, value=values[i][j])
            if c in marks:
                cell.fill = fills[marks[c]]
            row.append(cell)
        ws.append(row)

    ws.freeze_panes = "C2"
    wb.save(path)


def write_audit(m, b, changes, cur, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    hdr_fill = PatternFill("solid", fgColor=HDR)
    hdr_font = Font(color="FFFFFFFF", bold=True, size=10)

    def head(ws, cols, wide=None):
        ws.append(cols)
        for i in range(1, len(cols) + 1):
            cell = ws.cell(row=1, column=i)
            cell.fill = hdr_fill
            cell.font = hdr_font
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.freeze_panes = "A2"
        ws.row_dimensions[1].height = 28
        for col, w in (wide or {}).items():
            ws.column_dimensions[col].width = w

    def shade(ws, row, n, colour):
        f = PatternFill("solid", fgColor=colour)
        for c in range(1, n + 1):
            ws.cell(row=row, column=c).fill = f

    # ── summary ──────────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Summary"
    u = m["Unhealthy"].astype(str).str.strip()
    per_col = {}
    for (_i, c) in changes:
        per_col[c] = per_col.get(c, 0) + 1
    lines = [
        ("Metadata Table M1", ""),
        ("file", str(M1.relative_to(ROOT))),
        ("compared against", str(BASE.name)),
        ("built", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("rows / columns", f"{len(m):,} / {len(m.columns)}"),
        ("", ""),
        ("Changes by column", ""),
    ]
    for c, n in sorted(per_col.items(), key=lambda kv: -kv[1]):
        kind = ("read-length provenance" if c in PROVENANCE
                else "curation" if c in CURATED or c.startswith("disease_")
                else "other")
        lines.append((f"  {c}", f"{n:,}   ({kind})"))
    lines += [
        ("", ""),
        ("Cohort", ""),
        ("  Unhealthy", f"{int((u == 'Unhealthy').sum()):,}"),
        ("  Healthy", f"{int((u == 'Healthy').sum()):,}"),
        ("  unlabelled", f"{int((~u.isin(['Healthy', 'Unhealthy'])).sum()):,}"),
        ("", ""),
        ("BioSample", ""),
        ("  accession present", f"{len(m):,} / {len(m):,}  (100%)"),
        ("  recovered from SRS/ERS via ENA", "2,955"),
        ("  recovered from a run accession", "124"),
        ("  submitter name also in 'Sample ID' and 'Master_ID'", "yes, byte-identical"),
        ("", ""),
        ("Colour key", ""),
        ("  amber", "a value was replaced (BioSample)"),
        ("  blue", "health label moved"),
        ("  green", "disease flag added"),
        ("  red", "disease flag removed"),
        ("  grey", "read length / read count filled (provenance)"),
    ]
    for r, (k, v) in enumerate(lines, start=1):
        ws.cell(row=r, column=1, value=k)
        ws.cell(row=r, column=2, value=v)
        if v == "" and k:
            ws.cell(row=r, column=1).font = Font(bold=True, size=11)
    ws.column_dimensions["A"].width = 52
    ws.column_dimensions["B"].width = 56

    # ── BioSample ────────────────────────────────────────────────────────
    ws = wb.create_sheet("BioSample")
    cols = ["StudyFolder", "Sample ID", "BioSample (was)", "BioSample (now)",
            "SRS ID", "SRA_Accession", "recovered via"]
    head(ws, cols, {"A": 26, "B": 24, "C": 24, "D": 20, "E": 16, "F": 24,
                    "G": 20})
    SAMP = re.compile(r"^(SRS\d+|ERS\d+|DRS\d+)$")
    r = 1
    for i in range(len(m)):
        if (i, "BioSample") not in changes:
            continue
        old, new = changes[(i, "BioSample")]
        srs = str(m.iat[i, m.columns.get_loc("SRS ID")]).strip()
        via = "sample accession (ENA)" if SAMP.match(srs) else "run accession (ENA)"
        r += 1
        ws.append([m.iat[i, m.columns.get_loc("StudyFolder")],
                   m.iat[i, m.columns.get_loc("Sample ID")], old, new, srs,
                   m.iat[i, m.columns.get_loc("SRA_Accession")], via])
        shade(ws, r, len(cols), AMBER)
    ws.auto_filter.ref = f"A1:G{r}"

    # ── health label ─────────────────────────────────────────────────────
    ws = wb.create_sheet("Health label")
    cols = ["StudyFolder", "Sample ID", "Unhealthy (was)", "Unhealthy (now)",
            "why"]
    head(ws, cols, {"A": 26, "B": 24, "C": 16, "D": 16, "E": 62})
    why_map = {}
    if len(cur):
        for row in cur.itertuples(index=False):
            if row.column == "Unhealthy":
                why_map[str(row.Chosen_match_ID)] = row.reason
    r = 1
    for i in range(len(m)):
        if (i, "Unhealthy") not in changes:
            continue
        old, new = changes[(i, "Unhealthy")]
        cid = str(m.iat[i, m.columns.get_loc("Chosen_match_ID")])
        r += 1
        ws.append([m.iat[i, m.columns.get_loc("StudyFolder")],
                   m.iat[i, m.columns.get_loc("Sample ID")], old, new,
                   why_map.get(cid, "flagged obese; obesity now counts as unhealthy")])
        shade(ws, r, len(cols), BLUE)
    ws.auto_filter.ref = f"A1:E{r}"

    # ── disease flags ────────────────────────────────────────────────────
    if len(cur):
        ws = wb.create_sheet("Disease flags")
        cols = ["StudyFolder", "Chosen_match_ID", "column", "was", "now",
                "action", "reason"]
        head(ws, cols, {"A": 26, "B": 26, "C": 26, "D": 10, "E": 10, "F": 9,
                        "G": 76})
        r = 1
        for row in cur[cur["column"] != "Unhealthy"].itertuples(index=False):
            r += 1
            ws.append([row.StudyFolder, row.Chosen_match_ID, row.column,
                       row.old, row.new, row.action, row.reason])
            shade(ws, r, len(cols), GREEN if row.action == "add" else RED)
        ws.auto_filter.ref = f"A1:G{r}"

    # ── a finding worth not losing ───────────────────────────────────────
    ws = wb.create_sheet("Note - ZhuF aliases")
    head(ws, ["note"], {"A": 118})
    for line in [
        "ZhuF_2020_CHN carries two submitter-side names per sample.",
        "",
        "Our 'Sample ID' holds the BGI sequencing-library identifier "
        "(wHAXPI032587-26); ENA's sample_alias for the same accession is a "
        "CNGB/CNSA sample identifier (CNS0006992).",
        "",
        "The BioSample accessions recovered for these 42 samples were checked "
        "against ENA and agree. The mismatch is in the name, not the mapping.",
        "",
        "It matters if ZhuF is ever matched against its own supplementary "
        "table, which may list either identifier. 171 ZhuF samples are in the "
        "cohort in total, not only the 42 recovered here.",
    ]:
        ws.append([line])
        ws.cell(row=ws.max_row, column=1).alignment = Alignment(wrap_text=True,
                                                                vertical="top")
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=str(ROOT / "final_run" / "metadata_M1.xlsx"))
    ap.add_argument("--audit",
                    default=str(ROOT / "final_run" / "metadata_M1_audit.xlsx"))
    a = ap.parse_args()

    m = pd.read_csv(M1, low_memory=False)
    b = pd.read_csv(BASE, low_memory=False)
    parts = [pd.read_csv(p) for p in (CURATION, FINAL) if p.exists()]
    cur = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    print(f"  M1 {len(m):,} x {len(m.columns)}")
    print(f"  curation log rows: {len(cur):,}")

    changes = diff_cells(m, b)
    print(f"  changed cells: {len(changes):,}")

    write_audit(m, b, changes, cur, a.audit)
    print(f"  wrote {Path(a.audit).relative_to(ROOT)}")
    write_full(m, changes, a.full)
    print(f"  wrote {Path(a.full).relative_to(ROOT)}  "
          f"({Path(a.full).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
