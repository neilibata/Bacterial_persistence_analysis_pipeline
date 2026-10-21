#!/usr/bin/env python3
"""Move the Supplement's old Table S12 to S13, freeing S12 for the new table.

Two tables were called S12. The document's own is the industrial-to-non-
industrial richness ratio under repeat sampling of hosts (caption at para 830,
cited at para 824). The pipeline's list-level method comparison is also S12:
persistence/table_methods.py writes table_s12_method_comparison_<ref>.csv,
tools/build_table_s12_docx.py renders docs/Table_S12.docx, and
tools/check_run_conditions.py names it. The pipeline keeps S12 and the
document's table becomes S13, so no output filename changes.

The new S12 is never referenced from the text: Note 6G.6 describes the five
list-level tests without pointing at the table that compares them. That
sentence is reported here and has to be written by hand, because where it goes
is an editorial choice.

Only whole-word "Table S12" and "Tables S12" runs are touched, and only in
body and table paragraphs -- not the bibliography, where "S12" cannot occur as
a table reference. Field codes are skipped, so a Zotero citation is never
rewritten.

Formatting survives: only the text of the runs that carry the match is
replaced, never the paragraph.

    python tools/renumber_table_s12.py <docx>            # report only
    python tools/renumber_table_s12.py <docx> --apply    # backs up, then writes
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from docx import Document

#: "Table S12" / "Tables S12" / "Table S12." but never "Table S120".
PAT = re.compile(r"\b(Tables?\s+S)12\b")

# No default: the manuscript is not part of the repository, and a path
# baked in here names one machine's copy of one dated draft. Require it.
DEFAULT_DOCX = None


def _paragraphs(doc):
    """Every body and table paragraph, with a location label."""
    for i, p in enumerate(doc.paragraphs):
        yield f"para {i}", p
    for ti, t in enumerate(doc.tables, start=1):
        for ri, row in enumerate(t.rows):
            for ci, cell in enumerate(row.cells):
                for p in cell.paragraphs:
                    yield f"tbl{ti}:r{ri}c{ci}", p


def _in_field(run):
    """True when this run is part of a Word field code, not visible prose."""
    return run._element.findall(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        "instrText") != []


def renumber(path, apply=False):
    doc = Document(path)
    hits, skipped = [], 0
    for where, p in _paragraphs(doc):
        if not PAT.search(p.text or ""):
            continue
        # The match can straddle runs ("Table S" in one, "12" in the next), so
        # rewrite run by run and fall back to reporting when it does.
        done = False
        for run in p.runs:
            if _in_field(run):
                continue
            if PAT.search(run.text or ""):
                new = PAT.sub(r"\g<1>13", run.text)
                hits.append((where, run.text.strip(), new.strip()))
                if apply:
                    run.text = new
                done = True
        if not done:
            skipped += 1
            hits.append((where, (p.text or "").strip()[:90],
                         "SPLIT ACROSS RUNS — edit by hand"))

    print(f"  {path}")
    print(f"  {len(hits)} run(s) carrying 'Table S12'"
          + (f", {skipped} split across runs" if skipped else "") + "\n")
    for where, before, after in hits:
        print(f"    [{where}]")
        print(f"       before: {before!r}")
        print(f"       after:  {after!r}")

    if apply and hits:
        bak = Path(path).with_suffix(".docx.bak")
        shutil.copy2(path, bak)
        doc.save(path)
        print(f"\n  backed up to {bak}")
        print(f"  wrote {path}")
    elif not apply:
        print("\n  report only — pass --apply to write")

    print("\n  Still to write by hand: Note 6G.6 ends without pointing at the")
    print("  new Table S12. One sentence after the five tests are defined, e.g.")
    print('    "Table S12 reports all five statistics for every')
    print('     condition-by-list cell, with the two detection rules combined')
    print('     by the harmonic mean p-value and an `eligible` flag marking')
    print("     the contrasts Figure 5 shows.\"")
    return len(hits)


def main():
    ap = argparse.ArgumentParser()
    # Required, not optional-with-a-default: str(None) would be accepted here
    # as a path and only fail later as "no such file: None".
    ap.add_argument("docx",
                    help="the manuscript .docx to read (and, with --apply, "
                         "renumber in place)")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if not Path(a.docx).exists():
        sys.exit(f"not found: {a.docx}")
    return 0 if renumber(a.docx, a.apply) else 1


if __name__ == "__main__":
    sys.exit(main())
