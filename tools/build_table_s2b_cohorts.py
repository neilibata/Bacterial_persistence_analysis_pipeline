#!/usr/bin/env python3
"""Build Supplementary Table S2b — per-reference analytic cohort construction.

Each reference is filtered independently, so the analytic cohort differs between
them, structurally rather than by accident:

  * the >=30 Mb mapped-bases cut is **reference-dependent** — a sample maps more
    bases against the 5,081-genome combined catalogue than against UHGG's 3,971,
    so fewer samples fall below threshold;
  * the breadth/depth matrices cover slightly different sample sets.

Rows are samples (n_s) at each step, then the final cohort's hosts (n_h).

Run by the pipeline as the last `--figure full` task (figure_runner "table_s2b"),
so the table is regenerated with every run rather than transcribed:

    python tools/build_table_s2b_cohorts.py
    python tools/build_table_s2b_cohorts.py --out results/table_s2b_cohorts.csv

Values come from `results/<ref>/stats_report.csv` (tags `cohort_*`, written by
run_reference and cohort_manifest.write_cohort_manifest). A value the registry
cannot supply prints as "—"; nothing is filled in by hand.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"

REFS = [("UHGG", "Carter (Hadza + UHGG)"), ("lev", "Leviatan"),
        ("combined", "Combined (Carter + Leviatan)"), ("chimp", "Chimpanzee")]

# The role each reference plays (Supplementary Table S2, "Role in this study").
USED_BY = {
    "UHGG": "Non-industrial-enriched reference bound",
    "lev": "Industrial-enriched reference bound",
    "combined": "Primary reference for the reported results",
    "chimp": "Negative control only (Note 4.K, Table S5)",
}

ROWS = [
    ("matrix_samples",                   "Samples in the breadth matrix"),
    ("metadata_rows",                    "Rows in the curated metadata (M1)"),
    ("matched_to_matrix",                "Matched to the matrix by Chosen_match_ID"),
    ("dropped_unclassifiable_lifestyle", "Dropped: Lifestyle not Industrial/Non-Industrial"),
    ("dropped_excluded_samples",         "Dropped: amplicon exclusion list"),
    ("dropped_duplicate_biosample",      "Dropped: further library of an already-kept BioSample"),
    ("dropped_below_min_bp",             "Dropped: < 30 Mb mapped bases"),
    ("dropped_below_min_sgbs",           "Dropped: < 5 SGBs at >= 10% breadth"),
    ("_sec_runs",                        "Secondary runs retained as technical replicates"),
    ("_final",                           "**Analytic cohort**"),
    ("_hg",                              "  of which hunter-gatherer"),
    ("_trans",                           "  of which agro-pastoralist"),
    ("_ind",                             "  of which industrial"),
    ("_hosts",                           "**Hosts (n_h)**"),
    ("_hosts_hg",                        "  of which hunter-gatherer"),
    ("_hosts_trans",                     "  of which agro-pastoralist"),
    ("_hosts_ind",                       "  of which industrial"),
    ("sgbs",                             "SGBs in the catalogue"),
]


def _get(ref: str, tag: str):
    p = RESULTS / ref / "stats_report.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    h = d[d["tag"].astype(str) == tag]
    if not len(h):
        return None
    v = str(h.iloc[0]["value"]).strip()
    try:
        return int(float(v.replace(",", "")))
    except ValueError:
        return v


def column(ref: str) -> dict:
    col, derived = {}, []
    for key, _ in ROWS:
        if key.startswith("_"):
            continue
        col[key] = _get(ref, f"cohort_{key}")

    # Reinstatement is off (config.REINSTATE_SECONDARIES), so there is no
    # "before/after" split to report. What remains is how many secondary runs
    # were held back as technical replicates, which is the honest statement of
    # what happened to them.
    col["_sec_runs"] = _get(ref, "reinstate_disabled")
    col["_final"] = _get(ref, "n_total_samples")
    col["_hg"] = _get(ref, "n_samples_H-G")
    col["_trans"] = _get(ref, "n_samples_Transitional")
    col["_ind"] = _get(ref, "n_samples_Industrial")
    col["_hosts"] = _get(ref, "cohort_n_hosts")
    col["_hosts_hg"] = _get(ref, "cohort_n_hosts_H-G")
    col["_hosts_trans"] = _get(ref, "cohort_n_hosts_Transitional")
    col["_hosts_ind"] = _get(ref, "cohort_n_hosts_Industrial")

    # Fill gaps the registry cannot yet supply (pre-instrumentation reports).
    if col.get("sgbs") is None:
        col["sgbs"] = _get(ref, "n_sgbs_catalogue")
    col["_derived"] = derived
    return col


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results",
                    help="results root holding <ref>/stats_report.csv")
    ap.add_argument("--out", type=Path, default=None,
                    help="CSV (default <results>/table_s2b_cohorts.csv)")
    ap.add_argument("--md", type=Path, default=None,
                    help="Markdown (default <results>/table_s2b_cohorts.md)")
    args = ap.parse_args()
    global RESULTS
    RESULTS = args.results
    args.out = args.out or RESULTS / "table_s2b_cohorts.csv"
    args.md = args.md or RESULTS / "table_s2b_cohorts.md"

    cols = {ref: column(ref) for ref, _ in REFS}
    labels = dict(REFS)

    data = {"Step": [lab for _, lab in ROWS]}
    for ref, lab in REFS:
        data[lab] = [cols[ref].get(k) for k, _ in ROWS]
    df = pd.DataFrame(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    def fmt(v):
        return f"{v:,}" if isinstance(v, int) else ("—" if v is None else str(v))

    lines = [
        "# Table S2b — analytic cohort per reference catalogue", "",
        "Each reference is filtered independently, so the analytic cohort differs "
        "between them. The **≥30 Mb mapped-bases cut is reference-dependent**: a sample "
        "maps more bases against a larger catalogue, so fewer fall below threshold. "
        "Every sample is one biosample; secondary runs of a biosample are kept as "
        "technical replicates, never as further samples.", "",
        "| Step | " + " | ".join(lab for _, lab in REFS) + " |",
        "| --- | " + " | ".join("---:" for _ in REFS) + " |",
    ]
    for key, lab in ROWS:
        lines.append(f"| {lab} | " + " | ".join(fmt(cols[r].get(key)) for r, _ in REFS) + " |")
    lines += ["", "| Role | " + " | ".join(USED_BY[r] for r, _ in REFS) + " |",
              "| --- | " + " | ".join("---" for _ in REFS) + " |", ""]
    lines += [
        "", "## Samples and hosts", "",
        "n_s counts samples; n_h counts distinct hosts (`Subject_ID`). Several studies "
        "sampled the same people repeatedly, so n_s > n_h. Repeat samples are kept in the "
        "lifestyle, accumulation and prevalence analyses and reduced to one per host in "
        "the health analyses. A sample with no archived host identifier counts as one "
        "host, so n_h is an upper bound on the number of people.",
    ]
    # Coverage caveat, read from the reconciliation output if it has been run.
    rec = RESULTS / "cohort_reconciliation" / "summary.csv"
    if rec.exists():
        r = pd.read_csv(rec).set_index("reference")
        gaps = r[r["m1_classified_missing"] > 0]
        if len(gaps):
            lines += ["", "## Coverage caveat", ""]
            for ref, row in gaps.iterrows():
                lines.append(
                    f"**{labels.get(ref, ref)} does not cover the full metadata table.** "
                    f"{int(row['m1_classified_missing']):,} of the "
                    f"{int(row['m1_classified_missing'] + row['m1_classified_covered']):,} "
                    f"classified M1 samples have no row in its breadth matrix, so "
                    f"reference comparisons involving it are not run on identical cohorts.")
            lines += [
                "",
                "Every affected ID is listed in `results/cohort_reconciliation/` "
                "(`<ref>_in_m1_not_in_matrix.csv`), together with the reverse direction "
                "(matrix rows with no M1 entry) and the common-core ID list. Regenerate "
                "with `python tools/reconcile_cohorts.py`.",
            ]
    coll = RESULTS / "cohort_reconciliation" / "m1_id_collisions.csv"
    if coll.exists():
        c = pd.read_csv(coll, dtype=str)
        if len(c):
            studies = ", ".join(f"`{k}`" for k in c["Author"].value_counts().index[:3])
            life = dict(c["Lifestyle"].value_counts())
            lines += [
                "", "## Metadata key collisions", "",
                f"{c['_key'].nunique()} `Chosen_match_ID` values are shared by {len(c)} M1 "
                f"rows, from {studies} "
                f"({', '.join(f'{k} {v}' for k, v in life.items())}). `loading.py` keeps "
                f"one row per ID, so these need resolving upstream. Listed in "
                f"`m1_id_collisions.csv`.",
            ]

    lines += [
        "", "## Orphan matrix rows", "",
        "Each matrix carries rows with no M1 entry. These are **expected**: infants "
        "and non-gut samples were excluded during metadata curation, after the matrices "
        "were built, so the matrices retain rows the study does not analyse. They are "
        "filtered out on load and affect no result.",
    ]
    lines += ["", "Generated by `tools/build_table_s2b_cohorts.py` from "
              "`results/<ref>/stats_report.csv`."]
    derived = {r: cols[r]["_derived"] for r, _ in REFS if cols[r]["_derived"]}
    if derived:
        lines += ["", "> Some values were derived rather than read directly, because the "
                  "stats report predates the cohort instrumentation: "
                  + "; ".join(f"**{labels[r]}** ({', '.join(v)})" for r, v in derived.items())
                  + ". Re-run the pipeline to record them directly."]
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(lines) + "\n")

    print("\n".join(lines[:6 + len(ROWS)]))
    print(f"\nWrote {args.out} and {args.md}")


if __name__ == "__main__":
    main()
