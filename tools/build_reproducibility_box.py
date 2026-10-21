#!/usr/bin/env python3
"""Write the quantitative box for the metadata-standards comment piece.

The draft argues from principle; this supplies the arithmetic, all of it from one
meta-analysis of 88 public gut-metagenome studies, so every figure is a cost
actually paid rather than an estimate of one.

Design notes, because a box is read differently from body text. It is scanned
first and read second, so the layout has to answer "what is the point" before
the reader commits. Hence: one framing sentence, then a single shaded panel
carrying the headline contrast, then two compact tables, then four one-line
findings. Nothing is repeated between them.

The organising contrast -- archive-generated fields are complete, human-entered
fields are not -- is stated once, in the panel, and everything after it is
evidence. Provenance is marked per block, because three of the findings come from
our own curated file rather than from the repositories, and a reader who assumes
otherwise would draw the wrong conclusion about where the failure lies.

    python tools/build_reproducibility_box.py --out ~/Documents/Box1.docx
"""
from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

INK = RGBColor(0x14, 0x14, 0x16)
MUTED = RGBColor(0x60, 0x66, 0x70)
ACCENT = RGBColor(0x0F, 0x4C, 0x5C)
RULE = "C9D6DB"
PANEL = "EDF3F5"

COMPLETENESS = [
    ("Accession, layout, strategy, base count", "113,051", "100%"),
    ("Country", "72,915", "64.5%"),
    ("Collection date", "58,527", "51.8%"),
    ("Host species", "48,595", "43.0%"),
    ("Isolation source", "25,415", "22.5%"),
    ("Host sex", "16,679", "14.8%"),
    ("Host body site", "6,451", "5.7%"),
]

LADDER = [
    ("Sequence archive (ENA run + sample records)", "15,200", "84.2%"),
    ("+ a second repository (NCBI BioSample)", "15,210", "84.3%"),
    ("+ per-sample tables in the papers' supplements", "15,817", "87.6%"),
    ("Never deposited anywhere", "2,229", "12.4%"),
]

ARCHIVE_FINDINGS = [
    ("Layout", "100 of 101 runs declared single-end carry two reads per spot in "
     "the archive's own statistics. For 94, processed FASTA was uploaded instead "
     "of reads, so the archive reports zero bases and could not infer layout."),
    ("Body site", "610 runs with host = Homo sapiens and isolation source = "
     "feces are titled “sample from wastewater metagenome”, inherited from a "
     "project that sampled human and environmental habitats together."),
    ("Geography", "A four-city cohort (Toronto, Ann Arbor, Boston, Houston) is "
     "recorded as Canadian throughout: 52 of 79 country assignments wrong."),
]

CURATION_FINDINGS = [
    ("Identifiers", "Ten BioProject accessions carried trailing whitespace, "
     "silently breaking joins; five studies appeared under two spellings of "
     "their own name; ten country codes were not ISO 3166-1, including IND for "
     "Indonesia, which is the ISO code for India."),
]


def shade(cell_or_para, hexcolor: str) -> None:
    el = cell_or_para._tc if hasattr(cell_or_para, "_tc") else cell_or_para._p
    pr = el.get_or_add_tcPr() if hasattr(el, "get_or_add_tcPr") else el.get_or_add_pPr()
    sh = pr.makeelement(qn("w:shd"), {qn("w:val"): "clear",
                                      qn("w:color"): "auto",
                                      qn("w:fill"): hexcolor})
    pr.append(sh)


def border(paragraph, edge: str, size: int = 8, colour: str = RULE) -> None:
    pPr = paragraph._p.get_or_add_pPr()
    bd = pPr.find(qn("w:pBdr"))
    if bd is None:
        bd = pPr.makeelement(qn("w:pBdr"), {})
        pPr.append(bd)
    node = bd.makeelement(qn(f"w:{edge}"), {qn("w:val"): "single",
                                            qn("w:sz"): str(size),
                                            qn("w:space"): "6",
                                            qn("w:color"): colour})
    bd.append(node)


def run(p, text, *, size=9.5, bold=False, italic=False, colour=INK, caps=False):
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    r.font.color.rgb = colour
    r.font.all_caps = caps
    return r


def eyebrow(doc, text, provenance):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(2)
    run(p, text, size=8.5, bold=True, colour=ACCENT, caps=True)
    run(p, "   " + provenance, size=8, italic=True, colour=MUTED)
    border(p, "bottom")
    return p


def compact_table(doc, rows, widths, *, emphasise_first=True):
    t = doc.add_table(rows=0, cols=len(widths))
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    t.autofit = False
    for i, rec in enumerate(rows):
        cells = t.add_row().cells
        for j, (c, v) in enumerate(zip(cells, rec)):
            c.text = ""
            para = c.paragraphs[0]
            para.paragraph_format.space_after = Pt(1)
            para.paragraph_format.space_before = Pt(1)
            if j:
                para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            strong = emphasise_first and i == 0
            run(para, str(v), size=9,
                bold=strong or j == 1,
                colour=ACCENT if strong else INK)
            if strong:
                shade(c, PANEL)
    for row in t.rows:
        for c, w in zip(row.cells, widths):
            c.width = Inches(w)
    return t


def build(out: Path) -> None:
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1.1)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(9.5)
    normal.font.color.rgb = INK

    # ── title ────────────────────────────────────────────────────────────────
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(1)
    run(p, "Box 1", size=8.5, bold=True, colour=ACCENT, caps=True)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    run(p, "What poor metadata cost one meta-analysis", size=14, bold=True)
    border(p, "bottom", size=12)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(10)
    run(p, "18,046 gut metagenomes, 88 public studies, 113,051 sequencing runs, "
           "103 Tbp. Every figure below is a cost paid in assembling them.",
        size=9, italic=True, colour=MUTED)

    # ── the panel: the one idea ──────────────────────────────────────────────
    panel = doc.add_table(rows=1, cols=1)
    panel.autofit = False
    cell = panel.rows[0].cells[0]
    cell.width = Inches(6.3)
    cell.text = ""
    shade(cell, PANEL)
    q = cell.paragraphs[0]
    q.paragraph_format.space_before = Pt(6)
    q.paragraph_format.space_after = Pt(6)
    run(q, "Every field the archive generates is complete for all 113,051 runs. "
           "Every field a submitter must type is missing for 35% to 94% of them.",
        size=11, bold=True, colour=ACCENT)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

    # ── table 1 ──────────────────────────────────────────────────────────────
    eyebrow(doc, "Field completeness", "source: ENA, 113,051 runs")
    compact_table(doc, COMPLETENESS, [3.9, 1.2, 0.9])

    # ── table 2 ──────────────────────────────────────────────────────────────
    eyebrow(doc, "Recovering one variable: country of origin",
            "source: ENA, NCBI BioSample, 12 published supplements")
    compact_table(doc, LADDER, [4.2, 1.0, 0.8], emphasise_first=False)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    run(p, "Querying an entire second repository added 10 samples. The papers' "
           "own spreadsheets added 607.", size=8.5, italic=True, colour=MUTED)

    # ── findings, split by provenance ────────────────────────────────────────
    eyebrow(doc, "Errors no internal check can catch", "source: public archives")
    for label, body in ARCHIVE_FINDINGS:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.12)
        run(p, f"{label}. ", size=9, bold=True)
        run(p, body, size=9)

    eyebrow(doc, "And in our own curated file",
            "source: this study's metadata, not the repositories")
    for label, body in CURATION_FINDINGS:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.12)
        run(p, f"{label}. ", size=9, bold=True)
        run(p, body, size=9)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    run(p, "We report these against ourselves deliberately: curation performed "
           "downstream, by hand, under time pressure, reproduces exactly the "
           "failures it exists to correct.", size=8.5, italic=True, colour=MUTED)

    # ── the constructive close ───────────────────────────────────────────────
    eyebrow(doc, "What is already achievable", "")
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    run(p, "Re-deriving the cohort from the published inclusion criteria alone — "
           "paired-end shotgun sequencing of gut samples from non-infants, above "
           "a depth threshold, with assignable lifestyle — reproduces ")
    run(p, "98.0%", bold=True, colour=ACCENT)
    run(p, " of the curated sample list. The residual 2% is not analytical "
           "disagreement: it is the samples whose archive records are wrong or "
           "absent. An executable cohort definition is possible today; only "
           "deposition practice prevents it.")
    p = doc.add_paragraph()
    run(p, "Three requirements would have removed almost all of the above: "
           "per-sample geography and body site as structured fields rather than "
           "free text; run, sample, BioSample and BioProject accessions printed "
           "in the paper's own metadata table; and one controlled vocabulary, "
           "enforced at submission.", size=9)

    doc.save(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    # Repo-relative, and named for what it is rather than
                    # for one draft on one machine.
                    default=Path(__file__).resolve().parents[1]
                    / "paper-outputs" / "Box1_quantitative.docx")
    args = ap.parse_args()
    build(args.out)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
