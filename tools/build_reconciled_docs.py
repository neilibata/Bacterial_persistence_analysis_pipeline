#!/usr/bin/env python3
"""Write new .docx files with pipeline-current numbers substituted in.

Closes the loop between the provenance maps and the manuscripts. Source files
are never modified: each input is copied to a new name and edited there.

Substitutions are driven by an explicit spec (see docs/substitutions_*.csv), not
by pattern matching over the text. A blind numeric find/replace is unsafe in this
manuscript -- citation years, journal page ranges and SGB identifiers collide
with statistic values -- so every edit is anchored to a distinctive phrase from
the target paragraph and refuses to fire when the anchor is ambiguous.

Spec columns:
    doc      main | supp | methods | addfigs
    anchor   distinctive substring identifying the paragraph (must match once)
    find     exact text to replace within that paragraph
    replace  replacement text
    tag      pipeline tag the value came from, for the audit trail
    note     free text

    python tools/build_reconciled_docs.py                    # dry run, report only
    python tools/build_reconciled_docs.py --apply
    python tools/build_reconciled_docs.py --figures figspec.csv --apply
"""
from __future__ import annotations

import argparse
import csv
import shutil
import os as _os
from pathlib import Path

import docx
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
# Where the manuscript documents live. They are not in the repository, so
# this is a setting rather than a default: point MANUSCRIPT_DOCS_DIR at the
# directory holding them, and name the four files with MANUSCRIPT_* if they
# are not the current set. Nothing here hardcodes one machine's copy of one
# dated draft, which is what made this tool silently target the wrong
# document after the drafts were renamed.
DOCS = Path(_os.environ.get("MANUSCRIPT_DOCS_DIR") or ROOT / "manuscript_docs")
INPUTS = {
    "main":    _os.environ.get("MANUSCRIPT_MAIN", "main_text.docx"),
    "supp":    _os.environ.get("MANUSCRIPT_SUPP", "supplementary.docx"),
    "methods": _os.environ.get("MANUSCRIPT_METHODS", "methods.docx"),
    "addfigs": _os.environ.get("MANUSCRIPT_ADDFIGS",
                               "additional_figures.docx"),
}


def _norm(t: str) -> str:
    """Word peppers prose with non-breaking spaces; match on normalised text.

    An anchor typed with ordinary spaces will not match "while\xa063 of the 63"
    otherwise, which silently looks like a missing paragraph rather than a
    whitespace difference.
    """
    return t.replace("\xa0", " ").replace("\u2009", " ").replace("\u202f", " ")


def _paragraphs(doc):
    """Every paragraph in the document, including those inside tables."""
    yield from doc.paragraphs
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                yield from cell.paragraphs


def _replace_in_paragraph(p, find: str, repl: str) -> bool:
    """Replace within a paragraph, preserving run formatting where possible.

    Word splits text across runs arbitrarily, so a value can straddle runs. Try
    the simple per-run case first; fall back to rewriting the paragraph's runs
    only when the match spans several, which loses intra-match formatting but
    keeps the paragraph's own style.
    """
    for run in p.runs:
        if find in run.text:
            run.text = run.text.replace(find, repl, 1)
            return True
    full = "".join(r.text for r in p.runs)
    if find not in full:
        return False
    new = full.replace(find, repl, 1)
    for r in p.runs[1:]:
        r.text = ""
    if p.runs:
        p.runs[0].text = new
        return True
    return False


def apply_substitutions(spec_rows, apply: bool, suffix: str):
    by_doc: dict[str, list[dict]] = {}
    for r in spec_rows:
        by_doc.setdefault(r["doc"], []).append(r)

    report = []
    for key, rows in by_doc.items():
        src = DOCS / INPUTS[key]
        if not src.exists():
            report.append((key, "-", "MISSING INPUT", str(src)))
            continue
        out = DOCS / f"{src.stem}{suffix}{src.suffix}"
        if apply:
            shutil.copy2(src, out)
            doc = docx.Document(out)
        else:
            doc = docx.Document(src)

        paras = list(_paragraphs(doc))
        for r in rows:
            anchor = _norm(r["anchor"])
            hits = [p for p in paras if anchor in _norm(p.text)]
            if len(hits) == 0:
                report.append((key, r["tag"], "ANCHOR NOT FOUND", r["anchor"][:52]))
                continue
            if len(hits) > 1:
                report.append((key, r["tag"], f"ANCHOR AMBIGUOUS ({len(hits)})", r["anchor"][:52]))
                continue
            p = hits[0]
            n = _norm(p.text).count(_norm(r["find"]))
            if n == 0:
                report.append((key, r["tag"], "VALUE NOT PRESENT", f"{r['find']!r} — already updated?"))
                continue
            if n > 1:
                report.append((key, r["tag"], f"VALUE APPEARS {n}x", f"{r['find']!r} — narrow the anchor"))
                continue
            if apply and not _replace_in_paragraph(p, r["find"], r["replace"]):
                report.append((key, r["tag"], "REPLACE FAILED", r["find"]))
                continue
            report.append((key, r["tag"], "OK", f"{r['find']} -> {r['replace']}"))

        if apply:
            doc.save(out)
            report.append((key, "-", "WROTE", str(out)))
    return report


def apply_figures(fig_rows, apply: bool, suffix: str):
    """Swap embedded image blobs by explicit relationship id."""
    report = []
    by_doc: dict[str, list[dict]] = {}
    for r in fig_rows:
        by_doc.setdefault(r["doc"], []).append(r)
    for key, rows in by_doc.items():
        src = DOCS / INPUTS[key]
        out = DOCS / f"{src.stem}{suffix}{src.suffix}"
        target = out if (apply and out.exists()) else src
        doc = docx.Document(target)
        for r in rows:
            rid, path = r["rId"], ROOT / r["source_file"]
            if not path.exists():
                report.append((key, rid, "SOURCE MISSING", r["source_file"])); continue
            part = doc.part.related_parts.get(rid)
            if part is None:
                report.append((key, rid, "rId NOT FOUND", "")); continue
            if apply:
                part._blob = path.read_bytes()
            report.append((key, rid, "OK", f"{r.get('paper_name','')} <- {r['source_file']}"))
        if apply:
            doc.save(out)
            report.append((key, "-", "WROTE", str(out)))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, default=ROOT / "docs" / "substitutions_082626_to_combined.csv")
    ap.add_argument("--figures", type=Path, default=None)
    ap.add_argument("--suffix", default="_combined_auto")
    ap.add_argument("--apply", action="store_true", help="write the new files (default: report only)")
    args = ap.parse_args()

    rows = []
    if args.spec.exists():
        with open(args.spec) as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("doc")]
    print(f"spec: {len(rows)} substitutions from {args.spec.name}")
    rep = apply_substitutions(rows, args.apply, args.suffix) if rows else []

    if args.figures and args.figures.exists():
        with open(args.figures) as fh:
            frows = [r for r in csv.DictReader(fh) if r.get("doc")]
        print(f"figures: {len(frows)} image swaps from {args.figures.name}")
        rep += apply_figures(frows, args.apply, args.suffix)

    ok = sum(1 for r in rep if r[2] == "OK")
    bad = [r for r in rep if r[2] not in ("OK", "WROTE")]
    print(f"\n{'doc':9s}{'tag':46s}{'status':22s}detail")
    for d, t, s, det in rep:
        if s in ("OK", "WROTE") and not args.apply and s == "WROTE":
            continue
        print(f"{d:9s}{t[:45]:46s}{s:22s}{det[:70]}")
    print(f"\n  {ok} ready" + (f", {len(bad)} need attention" if bad else ""))
    if not args.apply:
        print("  dry run — nothing written. Re-run with --apply.")


if __name__ == "__main__":
    main()
