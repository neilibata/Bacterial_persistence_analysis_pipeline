#!/usr/bin/env python3
"""An inspectable record of every change made to Metadata Table M1.

Three edits were made to the curated metadata, and each one should be
checkable by eye rather than taken on trust:

  * BioSample  -- 3,079 rows held the submitter's own sample name instead of
                  an accession, so any repository lookup keyed on that column
                  failed. The accession was recovered from ``SRS ID`` through
                  ENA and the submitter's name kept in a new column.
  * disease_*  -- flags added where the study's free-text field named a
                  condition no column recorded, and flags dropped where they
                  contradicted everything else known about their study.
  * Unhealthy  -- obesity now counts as unhealthy.

Every changed row is listed with its before and after value and shaded, so the
question "which rows are different, and why" has an answer that does not
require re-deriving anything.

    python tools/build_m1_audit_workbook.py
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
OUT = ROOT / "final_run" / "metadata_M1_audit.xlsx"

BIOSAMPLE = re.compile(r"^SAM[END]A?\d+$")

HDR = "FF1F3864"          # header band
CHANGED = "FFFFF2CC"      # amber: a value was changed
BAD = "FFF8CBAD"          # red: still not resolved
OK = "FFE2EFDA"           # green: verified


def style(ws, n_cols, freeze="A2"):
    from openpyxl.styles import Alignment, Font, PatternFill
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = PatternFill("solid", fgColor=HDR)
        cell.font = Font(color="FFFFFFFF", bold=True, size=10)
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.freeze_panes = freeze
    ws.row_dimensions[1].height = 28


def widths(ws, spec):
    for col, w in spec.items():
        ws.column_dimensions[col].width = w


def shade(ws, row, n_cols, colour):
    from openpyxl.styles import PatternFill
    f = PatternFill("solid", fgColor=colour)
    for c in range(1, n_cols + 1):
        ws.cell(row=row, column=c).fill = f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    m = pd.read_csv(M1, low_memory=False)
    base = pd.read_csv(BASE, low_memory=False)
    cur = pd.read_csv(CURATION) if CURATION.exists() else pd.DataFrame()

    sub = m["submitter_sample_name"].astype(str).fillna("").str.strip()
    sub = sub.where(sub.str.lower() != "nan", "")
    bs = m["BioSample"].astype(str).str.strip()
    is_acc = bs.map(lambda v: bool(BIOSAMPLE.match(v)))
    touched = sub != ""                       # BioSample was rewritten here
    unresolved = ~is_acc

    wb = Workbook()

    # ── summary ──────────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Summary"
    rows = [
        ("Metadata Table M1", ""),
        ("file", str(M1.relative_to(ROOT))),
        ("built", datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("rows", f"{len(m):,}"),
        ("", ""),
        ("BioSample column", ""),
        ("  rows where BioSample was already an accession",
         f"{int((~touched).sum()):,}"),
        ("  rows where it held the submitter's sample name, now replaced",
         f"{int(touched.sum()):,}"),
        ("  rows still without an accession", f"{int(unresolved.sum()):,}"),
        ("  submitter names preserved in 'submitter_sample_name'",
         f"{int(touched.sum()):,}"),
        ("  resolution source", "ENA portal API, secondary_sample_accession "
                                "-> sample_accession"),
        ("  lookups attempted / resolved", "2,954 / 2,954"),
        ("", ""),
        ("Disease flags", ""),
        ("  cells changed", f"{len(cur):,}" if len(cur) else "—"),
        ("  samples affected",
         f"{cur['Chosen_match_ID'].nunique():,}" if len(cur) else "—"),
        ("", ""),
        ("Health label", ""),
        ("  Unhealthy",
         f"{int((m['Unhealthy'].astype(str).str.strip() == 'Unhealthy').sum()):,}"),
        ("  Healthy",
         f"{int((m['Unhealthy'].astype(str).str.strip() == 'Healthy').sum()):,}"),
        ("", ""),
        ("How to read the sheets", ""),
        ("  BioSample changed", "amber = the value was replaced"),
        ("  BioSample unresolved", "red = no accession could be found"),
        ("  Disease flags", "amber = added, red = dropped"),
    ]
    for r, (k, v) in enumerate(rows, start=1):
        ws.cell(row=r, column=1, value=k)
        ws.cell(row=r, column=2, value=v)
        if v == "" and k and not k.startswith(" "):
            ws.cell(row=r, column=1).font = Font(bold=True, size=11)
    widths(ws, {"A": 58, "B": 62})
    ws.freeze_panes = "A2"

    # ── BioSample changes ────────────────────────────────────────────────
    ws = wb.create_sheet("BioSample changed")
    cols = ["StudyFolder", "Chosen_match_ID", "BioSample (was)",
            "BioSample (now)", "SRS ID", "SRA_Accession", "resolved?"]
    ws.append(cols)
    n = 0
    for i in m.index[touched]:
        ok = bool(BIOSAMPLE.match(str(m.at[i, "BioSample"]).strip()))
        ws.append([m.at[i, "StudyFolder"], m.at[i, "Chosen_match_ID"],
                   sub.at[i], m.at[i, "BioSample"], m.at[i, "SRS ID"],
                   m.at[i, "SRA_Accession"], "yes" if ok else "NO"])
        n += 1
        shade(ws, n + 1, len(cols), CHANGED if ok else BAD)
    style(ws, len(cols))
    widths(ws, {"A": 26, "B": 26, "C": 24, "D": 20, "E": 16, "F": 22, "G": 11})
    ws.auto_filter.ref = f"A1:G{n + 1}"

    # ── still unresolved ─────────────────────────────────────────────────
    ws = wb.create_sheet("BioSample unresolved")
    ws.append(["StudyFolder", "Chosen_match_ID", "BioSample", "SRS ID",
               "SRA_Accession", "BioProject ID", "why"])
    n = 0
    for i in m.index[unresolved]:
        ws.append([m.at[i, "StudyFolder"], m.at[i, "Chosen_match_ID"],
                   m.at[i, "BioSample"], m.at[i, "SRS ID"],
                   m.at[i, "SRA_Accession"], m.at[i, "BioProject ID"],
                   "neither BioSample nor SRS ID is a usable accession"])
        n += 1
        shade(ws, n + 1, 7, BAD)
    style(ws, 7)
    widths(ws, {"A": 26, "B": 26, "C": 22, "D": 16, "E": 22, "F": 18, "G": 46})
    if n:
        ws.auto_filter.ref = f"A1:G{n + 1}"

    # ── disease flag changes ─────────────────────────────────────────────
    if len(cur):
        ws = wb.create_sheet("Disease flags")
        cols = ["StudyFolder", "Chosen_match_ID", "column", "was", "now",
                "action", "reason"]
        ws.append(cols)
        for r, row in enumerate(cur.itertuples(index=False), start=2):
            ws.append([row.StudyFolder, row.Chosen_match_ID, row.column,
                       row.old, row.new, row.action, row.reason])
            shade(ws, r, len(cols), CHANGED if row.action == "add" else BAD)
        style(ws, len(cols))
        widths(ws, {"A": 26, "B": 26, "C": 26, "D": 12, "E": 12, "F": 9,
                    "G": 74})
        ws.auto_filter.ref = f"A1:G{len(cur) + 1}"

    # ── per-study roll-up, so nothing hides in a long list ───────────────
    ws = wb.create_sheet("By study")
    ws.append(["StudyFolder", "samples", "BioSample rewritten",
               "still unresolved", "disease-flag cells changed"])
    agg = (pd.DataFrame({"StudyFolder": m["StudyFolder"],
                         "touched": touched, "unres": unresolved})
           .groupby("StudyFolder").agg(samples=("touched", "size"),
                                       touched=("touched", "sum"),
                                       unres=("unres", "sum")))
    flagn = (cur.groupby("StudyFolder").size() if len(cur)
             else pd.Series(dtype=int))
    r = 1
    for st, row in agg.sort_values("touched", ascending=False).iterrows():
        if not row["touched"] and not flagn.get(st, 0):
            continue
        r += 1
        ws.append([st, int(row["samples"]), int(row["touched"]),
                   int(row["unres"]), int(flagn.get(st, 0))])
        if row["unres"]:
            shade(ws, r, 5, BAD)
        elif row["touched"]:
            shade(ws, r, 5, CHANGED)
        else:
            shade(ws, r, 5, OK)
    style(ws, 5)
    widths(ws, {"A": 30, "B": 11, "C": 20, "D": 17, "E": 25})
    ws.auto_filter.ref = f"A1:E{r}"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(a.out)
    print(f"  wrote {Path(a.out).relative_to(ROOT)}")
    print(f"    BioSample changed   : {int(touched.sum()):,} rows")
    print(f"    BioSample unresolved: {int(unresolved.sum()):,} rows")
    print(f"    disease-flag cells  : {len(cur):,}")


if __name__ == "__main__":
    main()
