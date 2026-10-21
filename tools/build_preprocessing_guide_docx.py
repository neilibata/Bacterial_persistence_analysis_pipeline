#!/usr/bin/env python3
"""Write the preprocessing-verification implementation guide as a .docx.

The guide tells an editor exactly what to add to the Supplement for the new
preprocessing and verification work: one new Note, five new tables, and the
cross-references that make them reachable from the main text and Methods. It is
generated rather than hand-written so the numbers in it stay tied to
`docs/preprocessing_methods.md` and the pipeline outputs.

    python tools/build_preprocessing_guide_docx.py
    python tools/build_preprocessing_guide_docx.py --out ~/Documents/guide.docx
"""
from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
# Repo-relative, not under the author's home: a published tool must
# write somewhere the reader of this repository can find.
DEFAULT_OUT = (ROOT / "paper-outputs"
               / "preprocessing_verification_guide.docx")

ACCENT = RGBColor(0x1F, 0x40, 0x60)
MUTED = RGBColor(0x55, 0x5F, 0x6A)
FLAG = RGBColor(0x8C, 0x2F, 0x1E)


def h(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = ACCENT
    return p


def para(doc, text, *, italic=False, size=10, colour=None, space_after=6):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.italic = italic
    if colour is not None:
        run.font.color.rgb = colour
    p.paragraph_format.space_after = Pt(space_after)
    return p


def bullet(doc, text, size=10):
    p = doc.add_paragraph(style="List Bullet")
    run = p.add_run(text)
    run.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(3)
    return p


def mono(doc, text, size=8.5):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(size)
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(6)
    return p


def table(doc, headers, rows, widths=None, size=8.5):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for cell, text in zip(t.rows[0].cells, headers):
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(size)
    for record in rows:
        cells = t.add_row().cells
        for cell, text in zip(cells, record):
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(text))
            run.font.size = Pt(size)
    if widths:
        for row in t.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def build(out: Path) -> None:
    doc = Document()
    for section in doc.sections:
        section.left_margin = section.right_margin = Inches(0.9)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10)

    doc.add_heading("Preprocessing verification — implementation guide", level=0)
    para(doc,
         "What to add to the Supplement for the reprocessing, backfill and verification work, "
         "and where to cite it. Everything below is new material; nothing here asks you to "
         "change an existing Note.",
         italic=True, colour=MUTED)
    para(doc,
         "Numbering assumes the Supplement currently ends at Note 7 and Table S13: S11 is the "
         "GMW-CMH count table, S12 the list-level method comparison and S13 the richness "
         "ratio under repeat host sampling. Source text for every section is "
         "docs/preprocessing_methods.md in the analysis repository.",
         size=9, colour=MUTED, space_after=12)

    # ---------------------------------------------------------------- part A
    h(doc, "A. New Supplementary Note 8", level=1)
    para(doc,
         "Add after Note 7 (“Robustness analyses and limitations”). Note 2 already describes "
         "how the pipeline works; Note 8 is the evidence that it was run correctly and that the "
         "deposited code reproduces it. Keeping them separate means Note 2 stays readable as a "
         "methods narrative.")
    para(doc, "Title: Note 8. Preprocessing verification, provenance and reproducibility", size=10)

    table(doc,
          ["§", "Section title", "Source in preprocessing_methods.md", "New table"],
          [["8.1", "What was verified, and why", "S.1 Overview + S.7 preamble", "—"],
           ["8.2", "Sample identity and run retrieval", "S.2", "—"],
           ["8.3", "Stage parameters and attrition", "S.3", "Table S14, S15"],
           ["8.4", "Matrix assembly and the Leviatan backfill", "S.5", "—"],
           ["8.5", "Cohort, reference and matrix integrity", "S.7.1–S.7.3", "Table S16"],
           ["8.6", "Concordance with the original processing", "S.7.4", "Table S17"],
           ["8.7", "Mapping rate varies by study, not by processing", "S.7.5", "—"],
           ["8.8", "Multi-run samples: a bounded shortfall", "S.7.6", "Table S18"],
           ["8.9", "Checks pending at submission", "S.7.7", "—"],
           ["8.10", "Reproducing the preprocessing", "S.8", "—"]],
          widths=[0.45, 2.35, 2.35, 0.95])

    # ---------------------------------------------------------------- part B
    doc.add_page_break()
    h(doc, "B. New supplementary tables S11–S15", level=1)
    para(doc,
         "Content for all five is already tabulated in docs/preprocessing_methods.md — paste "
         "from there rather than retyping. Column counts are given so you can size them before "
         "pasting; see Part E on width.")

    h(doc, "Table S14. Preprocessing stage parameters", level=2)
    para(doc, "4 columns × 8 rows. Source: §S.3 table (Stage / Tool / Invocation / Rationale).", size=9)
    para(doc, "Legend: “Every stage of the read-processing workflow, with the exact invocation "
              "used and the reason for each non-default choice. Both MAPQ tiers are retained so "
              "the discarded fraction is auditable.”", size=9, colour=MUTED)

    h(doc, "Table S15. Stage-by-stage attrition and read-volume accounting", level=2)
    para(doc, "Two blocks. Block 1: 4 columns × 4 rows (§S.3 attrition). "
              "Block 2: 6 columns × 9 rows (§S.3 read-volume accounting, three references "
              "× three lifestyle groups).", size=9)
    para(doc, "Legend: “Median, IQR and range of material retained at each stage, and total "
              "sequenced versus mapped bases per reference and lifestyle group. Mapped fraction "
              "differs between references because the catalogues differ in size and composition, "
              "not because processing differed.”", size=9, colour=MUTED)

    h(doc, "Table S16. Verification checks: cohort, reference and matrix integrity", level=2)
    para(doc, "2 columns (Check / Result) × 19 rows, in three labelled blocks from §S.7.1, "
              "§S.7.2 and §S.7.3. This is the table a reviewer will look for.", size=9)
    para(doc, "Legend: “All integrity checks run on the analytic matrices. Every count is "
              "regenerated by the tools listed in Note 8.10; none is asserted by hand.”",
         size=9, colour=MUTED)

    h(doc, "Table S17. Concordance between the deposited workflow and the original processing", level=2)
    para(doc, "2 columns × 4 rows (§S.7.4), plus the two named discordant samples in the legend.", size=9)
    # "the published matrices" was wrong: these are this study's own coverage
    # and depth matrices, and nothing here is published. Legend text goes into
    # the Supplement, so the word cannot stand.
    para(doc, "Legend: “Twenty-seven samples reprocessed from raw reads with the deposited "
              "workflow and compared against the matrices the analyses were run on. "
              "SRR061226 and ERR7746144 "
              "differ; SRR061226 because the original used one of its two runs. Both are retained "
              "and neither is excluded from any analysis.”", size=9, colour=MUTED)

    h(doc, "Table S18. Multi-run samples: bounded, one-directional shortfall", level=2)
    para(doc, "Source: §S.7.6. State the 106 candidate samples (0.6% of the cohort), the one "
              "confirmed case, and that the direction is conservative — a shortfall depresses "
              "detection, so it cannot manufacture the reported non-industrial excess.", size=9)

    # ---------------------------------------------------------------- part C
    doc.add_page_break()
    h(doc, "C. Cross-document references to insert", level=1)
    para(doc,
         "Without these the new Note is unreachable and a reader has no route from a claim to its "
         "evidence. Insert each citation at the point named.")

    table(doc,
          ["Document", "Location", "Insert", "Why"],
          [["Methods", "§ Read processing, reference catalogues… (end of first paragraph)",
            "“(Note 8 and Tables S11–S12 in Supplement)”",
            "Points to the exact invocations; currently only Note 2 is cited"],
           ["Methods", "§ Dataset assembly and analytic cohorts",
            "“(Note 8.5 and Table S16 in Supplement)”",
            "Cohort completeness and the Leviatan backfill"],
           ["Main text", "¶12, after the combined-reference sentence",
            "“(Note 8.4 in Supplement)”",
            "The 5,081-SGB merge is described only in the Supplement"],
           ["Main text", "Data availability / code statement",
            "“(Note 8.6 and Table S17 in Supplement)”",
            "Concordance is the claim that the deposited code is the code that was used"],
           ["Main text", "Limitations",
            "“(Note 8.8 and Table S18 in Supplement)”",
            "The multi-run shortfall belongs with the other stated limitations"],
           ["Supplement", "Note 2, closing line",
            "“Verification of every step described here is reported in Note 8.”",
            "Forward pointer so Note 2 readers find the evidence"]],
          widths=[0.75, 1.95, 1.95, 1.65])

    # ---------------------------------------------------------------- part D
    doc.add_page_break()
    h(doc, "D. The verification code", level=1)
    para(doc,
         "Everything in Tables S13–S15 is regenerated by these tools. Note 8.10 should list them "
         "so a reviewer can re-run any number in the Note. Paths are relative to the analysis "
         "repository root.")

    h(doc, "Analysis repository (tools/)", level=2)
    table(doc,
          ["Tool", "What it asserts", "Feeds"],
          [["reconcile_cohorts.py", "Curated metadata (M1) against each reference's breadth matrix",
            "Table S16, block 1"],
           ["find_samples_in_matrices.py", "Searches every matrix build for samples missing from the canonical one",
            "Table S16, block 1"],
           ["check_aggregated_coverage.py", "Which samples are already aggregated, matched on identity not filename",
            "Note 8.2"],
           ["concordance_check.py", "That the deposited workflow reproduces the original preprocessing",
            "Table S17"],
           ["parse_preprocessing_logs.py", "Read-volume cascade and multi-mapping rate from pipeline logs",
            "Table S15"],
           ["trim_coverage.py", "How much of the cohort the surviving Trim Galore reports cover",
            "Table S15"],
           ["mapped_fraction_outliers.py", "Samples whose mapped bases are implausibly low for their input volume",
            "Note 8.7"],
           ["build_multirun_worklist.py", "Emits the sample→runs worklist for reprocessing",
            "Table S18"],
           ["merge_lev_backfill.py", "Appends the backfill shard to the assembled Leviatan matrices",
            "Note 8.4"],
           ["build_combined_matrix.py", "Rebuilds the 5,081-SGB combined matrices from UHGG and Leviatan",
            "Note 8.4"],
           ["verify_reproducibility.py", "That the pipeline reproduces the reported numbers on a fresh machine",
            "Note 8.10"]],
          widths=[1.6, 3.4, 1.3])

    h(doc, "Preprocessing repository (workflow/scripts/)", level=2)
    table(doc,
          ["Script", "What it asserts", "Feeds"],
          [["fetch_runs.py", "Fetches every run of a sample and concatenates the paired-end ones; "
                             "ENA library_layout lookup with SRA fallback and md5 verification",
            "Note 8.2"],
           ["verify_completeness.py", "That every requested accession reached the final matrix",
            "Table S16, block 1"],
           ["merge_matrix_shards.py", "Merges dated matrix shards into the canonical matrix",
            "Note 8.4"],
           ["aggregate_coverage.py", "Contig→species aggregation with contig-length weighting",
            "Note 8.3"],
           ["validate_contig_map() in Snakefile", "That the contig map is species-level, not "
                                                  "contig-level — enforced at parse time",
            "Table S16, block 2"]],
          widths=[1.6, 3.4, 1.3])

    h(doc, "Re-running the whole verification set", level=2)
    mono(doc,
         "python tools/reconcile_cohorts.py\n"
         "python tools/concordance_check.py\n"
         "python tools/parse_preprocessing_logs.py\n"
         "python tools/mapped_fraction_outliers.py\n"
         "python tools/verify_reproducibility.py")
    para(doc,
         "verify_reproducibility.py is the one to cite in the Note: it is the single command that "
         "re-derives the reported numbers, and it is what a reviewer asking “does this code "
         "produce these results” should be pointed at.",
         size=9, colour=MUTED)

    # ---------------------------------------------------------------- part E
    doc.add_page_break()
    h(doc, "E. Issues found while auditing the current Supplement", level=1)
    para(doc,
         "These are separate from the new material and need decisions before submission.",
         italic=True, colour=MUTED)

    h(doc, "E.1 Two citations do not resolve to the table cited", level=2)
    para(doc, "¶30 cites Table S9 for richness effects by condition (“positive associations "
              "with obesity … large negative association with IBD”). Table S9 contains sample "
              "sizes, depth diagnostics and per-list depletion/enrichment fractions — no richness "
              "effect size. The data exist in results/<ref>/section3_health/main/"
              "health_richness_gmw_10pct.csv (columns diff, ci_lo, ci_hi, pval, sig). Either add a "
              "richness block to S9 or cite the figure alone.",
         colour=FLAG)
    para(doc, "¶32 cites Table S7 for BloSSUMH+ associations under prevalence matching. Table S7 "
              "is pooled-missing taxa for the original VANISH/BloSSUM lists; it contains no health "
              "data and no BloSSUMH+ column. The prevalence-matched results "
              "(health_per_taxon_prevmatched_10pct.csv) have no supplementary table at all — this "
              "is a missing table, not just a wrong number.",
         colour=FLAG)

    h(doc, "E.2 Mixed reference basis in one sentence", level=2)
    para(doc, "¶26 reads “the full industrial cohort defined in Table S3 (n = 14,246)” and then "
              "“115 of the 124 original VANISH taxa were detected”. 14,246 is the UHGG industrial "
              "count; 115/124 is the combined value, where the industrial cohort is 14,270. Table S3 "
              "also carries 14,246. Update both to 14,270 to match the combined basis now used "
              "throughout Tables S5–S10.", colour=FLAG)

    h(doc, "E.3 Uncited supplementary material", level=2)
    para(doc, "Tables S2 and S8, and Note 7, are never cited in the main text or Methods. Most "
              "journals require every supplementary item to be called out at least once.")

    h(doc, "E.4 Table widths", level=2)
    para(doc, "The usable text width is 6.27 in (A4, 1 in margins). Declared widths:")
    table(doc,
          ["Table", "Cols", "Declared", "Layout", "Effect"],
          [["S1", "10", "7.38 in", "fixed", "Overflows the page"],
           ["S6", "15", "7.08 in", "fixed", "Overflows the page"],
           ["S9", "14", "6.30 in", "fixed", "Over by 0.03 in — borderline"],
           ["S3", "8", "10.00 in", "auto", "Word rescales; severely cramped"],
           ["S2, S4, S7, S8, S10", "—", "6.30–7.18 in", "auto", "Word rescales; cramped"],
           ["S5", "9", "5.81 in", "auto", "Fits"]],
          widths=[1.5, 0.6, 1.0, 0.8, 2.4])
    para(doc, "S1 and S6 are the urgent ones: they are on a fixed layout and will print past the "
              "margin rather than being rescaled. S9 is 0.03 in over after the reformat — within "
              "border-rendering slop, but worth taking to 6.20 in.", colour=FLAG)

    doc.save(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    build(args.out)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
