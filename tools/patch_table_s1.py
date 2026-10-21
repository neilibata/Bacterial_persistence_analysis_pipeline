#!/usr/bin/env python3
"""Reconcile Table S1 in the supplement against the pipeline's derivation.

Table S1 can hold a different number of populated rows from the pipeline's
derivation, because a study spanning two lifestyle groups is one row in the
document and two in the derivation. Almost every other cell is already correct,
and a blanket regeneration would discard the Reference column, which the
pipeline does not produce.

So this compares cell by cell, keyed on (Author, Year, Country), reports exactly
what differs, and with --apply rewrites only those cells. Country names are
normalised for matching but a change of spelling is still reported: counting
distinct country *names* rather than codes is what inflates the country total
when one country is spelled several ways.

    python tools/patch_table_s1.py                          # report only
    python tools/patch_table_s1.py --apply --out <supplement>.docx

Formatting is preserved by editing the text of a cell's first run rather than
replacing the cell, so styling, fonts and table borders survive.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
# No default: the manuscript is not part of the repository, and a path
# baked in here names one machine's copy of one dated draft. Require it.
DEFAULT_DOCX = None
PIPE = ROOT / "results" / "UHGG" / "table_s1a_study_summary_formatted.csv"

# docx column index -> pipeline column name
COLMAP = {
    0: "Author",
    1: "Year",
    2: "Country_full",
    3: "N",
    4: "Age median [IQR]",
    5: "Gbp/sample median [IQR]",
    6: "Lifestyle",
    7: "unhealthy_pct",
    8: "conditions",
    # 9 = Reference: the pipeline does not produce it, never touched
}

COUNTRY_ALIASES = {
    "usa": "united states of america",
    "united states": "united states of america",
    "us": "united states of america",
    "uk": "united kingdom",
    "great britain": "united kingdom",
}

# The document writes "Agro-pastoralist"; the pipeline writes "Transitional".
# They denote the same group. Which label the manuscript should use is an
# editorial decision, not a data correction, so these are matched as equivalent
# and reported separately from real differences.
LIFESTYLE_ALIASES = {
    "agro-pastoralist": "transitional",
    "agro-pastoralist/agriculturalist": "transitional",
    "agricultural": "transitional",
    "agriculturalist": "transitional",
    "hunter-gatherer": "h-g",
    "hg": "h-g",
}


def norm_lifestyle(s: str) -> str:
    k = re.sub(r"\s+", " ", str(s).strip().lower())
    return LIFESTYLE_ALIASES.get(k, k)


def norm_country(s: str) -> str:
    k = re.sub(r"\s+", " ", str(s).strip().lower())
    return COUNTRY_ALIASES.get(k, k)


def norm_value(s) -> str:
    """Compare on content, not on incidental whitespace or dash style."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    t = str(s).strip()
    t = t.replace("–", "-").replace("—", "-").replace("−", "-")
    t = re.sub(r"\s+", " ", t)
    if re.fullmatch(r"-?\d+\.0", t):          # 8.0 == 8
        t = t[:-2]
    return t


def cell_text(cell) -> str:
    return cell.text.strip()


def set_cell(cell, text: str) -> None:
    """Replace a cell's text, keeping the first run's formatting."""
    paras = cell.paragraphs
    p = paras[0]
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.add_run(text)
    for extra in paras[1:]:
        for r in extra.runs:
            r.text = ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docx", type=Path, default=DEFAULT_DOCX,
                    required=True,
                    help="the manuscript .docx to read (and, with "
                         "--apply, edit in place)")
    ap.add_argument("--pipeline", type=Path, default=PIPE)
    ap.add_argument("--table-index", type=int, default=0)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.apply and not args.out:
        ap.error("--apply needs --out (the original is never modified in place)")

    pipe = pd.read_csv(args.pipeline, dtype=str).fillna("")
    doc = Document(args.docx)
    tbl = doc.tables[args.table_index]
    rows = tbl.rows[1:]
    print(f"  document : {args.docx.name}  table[{args.table_index}] "
          f"{len(rows)} data rows")
    print(f"  pipeline : {args.pipeline.name}  {len(pipe)} rows")

    # index the pipeline by (author, year, normalised country)
    pipe_key = {}
    for _, r in pipe.iterrows():
        k = (norm_value(r["Author"]).lower(), norm_value(r["Year"]),
             norm_country(r["Country_full"]))
        pipe_key.setdefault(k, []).append(r)

    edits, naming, unmatched_doc, blank = [], [], [], 0
    used = set()
    for ri, row in enumerate(rows, start=1):
        cells = row.cells
        author, year, country = (cell_text(cells[0]), cell_text(cells[1]),
                                 cell_text(cells[2]))
        if not any([author, year, country]):
            blank += 1
            continue
        k = (author.lower(), norm_value(year), norm_country(country))
        cand = pipe_key.get(k)
        if not cand:
            unmatched_doc.append((ri, author, year, country))
            continue
        # Several studies span two lifestyle groups and are already two rows in
        # the document. Disambiguate on lifestyle before concluding a row has no
        # match -- otherwise they look like structural changes when they are not.
        if len(cand) > 1:
            life = norm_lifestyle(cell_text(cells[6]))
            same = [r for r in cand if norm_lifestyle(r["Lifestyle"]) == life]
            if len(same) == 1:
                pr = same[0]
                used.add((k, norm_lifestyle(pr["Lifestyle"])))
            else:
                unmatched_doc.append(
                    (ri, author, year, f"{country} ({len(cand)} pipeline rows)"))
                continue
        else:
            pr = cand[0]
            used.add((k, norm_lifestyle(pr["Lifestyle"])))

        for ci, pcol in COLMAP.items():
            have, want = cell_text(cells[ci]), norm_value(pr[pcol])
            if pcol == "Lifestyle":
                # equivalent labels are not a difference
                if norm_lifestyle(have) == norm_lifestyle(want):
                    if norm_value(have) != want:
                        naming.append((ri, ci, author, year, country, have, want))
                    continue
            if norm_value(have) != want:
                edits.append((ri, ci, author, year, country, pcol, have, want))

    pipe_only = []
    for k, rowlist in pipe_key.items():
        for r in rowlist:
            if (k, norm_lifestyle(r["Lifestyle"])) not in used:
                pipe_only.append(r)

    print(f"\n  blank rows skipped        : {blank}")
    print(f"  rows matched 1:1          : {len(rows) - blank - len(unmatched_doc)}")
    print(f"  CELLS DIFFERING (data)    : {len(edits)}")
    print(f"  label-only differences    : {len(naming)}  (Agro-pastoralist vs Transitional)")
    print(f"  document rows unmatched   : {len(unmatched_doc)}")
    print(f"  pipeline rows unmatched   : {len(pipe_only)}")

    if edits:
        print(f"\n  === cell differences ===")
        print(f"  {'row':>4} {'column':<26} {'study':<28} {'document':<24} -> pipeline")
        for ri, ci, a, y, c, pcol, have, want in edits[:60]:
            study = f"{a} {y} {c}"[:27]
            print(f"  {ri:>4} {pcol:<26} {study:<28} {have[:23]:<24} -> {want[:28]}")
        if len(edits) > 60:
            print(f"  ... and {len(edits)-60} more")

    if unmatched_doc:
        print(f"\n  === document rows with no 1:1 pipeline match ===")
        for ri, a, y, c in unmatched_doc:
            print(f"  row {ri:>4}: {a} {y} {c}")

    if naming:
        print(f"\n  === label-only (editorial decision, NOT applied) ===")
        print(f"  {len(naming)} rows write 'Agro-pastoralist' where the pipeline writes "
              f"'Transitional'.")
        print(f"  These denote the same group. Pick one label and apply it globally.")

    if pipe_only:
        print(f"\n  === pipeline rows absent from the document (rows to ADD) ===")
        for r in pipe_only:
            print(f"    {r['Author']} {r['Year']} {r['Country_full']}  "
                  f"N={r['N']}  {r['Lifestyle']}")

    # Studies where the document and the pipeline disagree on table *structure*
    # rather than on a value. These are listed with their target rows rather
    # than rewritten, because inserting or re-assigning rows is not something to
    # do silently inside someone's manuscript.
    struct = {}
    for r in pipe_only:
        struct.setdefault((r["Author"], r["Year"], r["Country_full"]), []).append(r)
    for _, a, y, c in unmatched_doc:
        struct.setdefault((a, y, c), [])
    if struct or any(e[2] == "RubelMA" for e in edits):
        print(f"\n  === STRUCTURAL — apply by hand ===")
        for (a, y, c), rl in sorted(struct.items()):
            print(f"\n  {a} {y} {c}")
            if rl:
                for r in rl:
                    print(f"      -> N={r['N']:<5} {r['Lifestyle']:<14} "
                          f"age={r['Age median [IQR]']:<18} "
                          f"Gbp={r['Gbp/sample median [IQR]']:<20} "
                          f"unhealthy={r['unhealthy_pct']}")
            else:
                print(f"      (document row with no pipeline counterpart — "
                      f"check author/year/country spelling)")

    if args.apply:
        for ri, ci, *_ , want in [(e[0], e[1], None, None, None, None, None, e[7])
                                  for e in edits]:
            set_cell(rows[ri - 1].cells[ci], want)
        doc.save(args.out)
        print(f"\n  wrote {args.out}  ({len(edits)} cells rewritten)")
        print("  NOTE: split rows and additions are NOT applied automatically -- "
              "they change table structure and are listed above for manual insertion.")
    else:
        print(f"\n  (report only; pass --apply --out <file> to rewrite the {len(edits)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
