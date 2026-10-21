#!/usr/bin/env python3
"""Emit the cohort metadata with audit columns, and a study-level audit table.

Two deliverables for the data-hygiene follow-up:

  metadata_audited.csv    every original column, unchanged, plus qc_* columns
                          carrying what the data says about each sample and how
                          that compares with what the archive declares. Flagged
                          samples are PRESENT and marked, never removed -- a
                          filtered file cannot document the errors it hides.

  study_audit_table.csv   one row per study, in the shape of Table S1: cohort
                          size, then per-field completeness, then counts of each
                          audit flag.

Audit columns come from three sources, kept distinct because their reliability
differs:

  measured   from the breadth and depth matrices, keyed on the matrix index --
             the pipeline's own linkage, so unambiguous
  declared   from ENA run records, joined by the documented multi-key rule
             (BioSample, SRS ID, Chosen_match_ID, SRA_Accession), with runs
             gathered from SRA_Accession and Secondary_run_IDs and split on both
             ';' and ','. Absent where no run resolves.
  derived    flags computed from the two above

    python tools/build_audited_metadata.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Fields whose completeness the follow-up reports on.
COMPLETENESS = ["Age", "Age group", "Gender", "BMI", "Country", "Lifestyle",
                "Unhealthy", "Diseased (all)", "Total bases", "BioSample",
                "SRA_Accession", "BioProject ID"]

AMPLICON_MAX_BREADTH_PCT = 0.02
AMPLICON_MIN_DPCB = 2.5
MIN_SGBS = 5


def _blank(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["nan", "", "none"])


def _runs_of(row) -> set[str]:
    """Every run accession recorded for a sample, however it is spelled."""
    out = set()
    for col in ("SRA_Accession", "Secondary_run_IDs"):
        for x in str(row.get(col) or "").replace(",", ";").split(";"):
            x = x.strip()
            if x:
                out.add(x)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "metadata_Aug_26_2026_final.csv",
                    help="the corrected cohort the pipeline actually reads")
    ap.add_argument("--readd-from", type=Path, default=ROOT / "metadata_Aug_26_2026_organised.csv",
                    help="pre-removal file; rows dropped from --src are re-added and flagged")
    ap.add_argument("--profile", type=Path, default=ROOT / "docs" / "sample_breadth_profile.csv")
    ap.add_argument("--consistency", type=Path, default=ROOT / "docs" / "breadth_depth_consistency.csv")
    ap.add_argument("--ena", type=Path, default=ROOT / "docs" / "ena_run_cache.tsv")
    ap.add_argument("--out-meta", type=Path, default=ROOT / "docs" / "metadata_audited.csv")
    ap.add_argument("--out-table", type=Path, default=ROOT / "docs" / "study_audit_table.csv")
    args = ap.parse_args()

    m = pd.read_csv(args.src, low_memory=False)
    if args.readd_from and args.readd_from.exists():
        # Build on the corrected cohort, then put the removed rows back so they
        # are documented rather than silently absent. Reading the pre-removal
        # file directly would also re-import corrections made after it was
        # written -- 24 Healthy/Unhealthy contradictions I had already fixed.
        pre = pd.read_csv(args.readd_from, low_memory=False)
        back = pre[~pre["Chosen_match_ID"].astype(str).str.strip().isin(
            m["Chosen_match_ID"].astype(str).str.strip())]
        if len(back):
            print(f"  re-adding {len(back)} removed row(s) so they can be flagged")
            m = pd.concat([m, back], ignore_index=True)
    n0 = len(m)
    key = m["Chosen_match_ID"].astype(str).str.strip()
    print(f"{args.src.name}: {n0:,} rows")

    # ── measured: keyed on the matrix index ──────────────────────────────
    prof = pd.read_csv(args.profile)
    prof["Sample_ID"] = prof.Sample_ID.astype(str).str.strip()
    cons = pd.read_csv(args.consistency)
    cons["Sample_ID"] = cons.Sample_ID.astype(str).str.strip()
    meas = prof[["Sample_ID", "richness_10pct", "max_breadth", "top1_share"]].merge(
        cons[["Sample_ID", "median_breadth_pct", "dpcb_median", "n_sgb_any"]],
        on="Sample_ID", how="outer")
    m = m.assign(_k=key).merge(meas, left_on="_k", right_on="Sample_ID", how="left")
    print(f"  measured stats attached for {int(m.richness_10pct.notna().sum()):,}")

    # ── declared: ENA, via the documented multi-key rule ─────────────────
    ena = pd.read_csv(args.ena, sep="\t", dtype=str)
    ena["run_accession"] = ena.run_accession.astype(str).str.strip()
    by_run = ena.set_index("run_accession")
    strat, sci, abase, complete, nfound = [], [], [], [], 0
    for _, r in m.iterrows():
        recorded = _runs_of(r)
        runs = [x for x in recorded if x in by_run.index]
        if not runs:
            strat.append(""); sci.append(""); abase.append(np.nan); complete.append(False); continue
        nfound += 1
        sub = by_run.loc[runs]
        if isinstance(sub, pd.Series):
            sub = sub.to_frame().T
        strat.append(";".join(sorted({str(x).upper() for x in sub.library_strategy.dropna()})))
        sci.append(";".join(sorted({str(x).lower() for x in sub.scientific_name.dropna()})))
        abase.append(float(pd.to_numeric(sub.base_count, errors="coerce").sum()))
        # only compare depths when every recorded run resolved, otherwise the
        # archive total is short by construction and the ratio is meaningless
        complete.append(len(runs) == len(recorded))
    m["qc_declared_strategy"] = strat
    m["qc_declared_scientific_name"] = sci
    m["qc_archive_bases"] = abase
    m["qc_all_runs_resolved"] = complete
    print(f"  ENA record resolved for {nfound:,} of {n0:,}")

    # ── derived flags ────────────────────────────────────────────────────
    m["qc_amplicon_suspect"] = ((m.median_breadth_pct < AMPLICON_MAX_BREADTH_PCT)
                                & (m.dpcb_median > AMPLICON_MIN_DPCB)).fillna(False)
    m["qc_low_richness"] = (m.richness_10pct < MIN_SGBS).fillna(False)
    m["qc_isolate_like"] = ((m.richness_10pct <= 3) & (m.top1_share >= 0.3)).fillna(False)
    m["qc_no_matrix_row"] = m.Sample_ID.isna()
    # declared WGS while the data says amplicon: the error the archive cannot see
    m["qc_strategy_contradicted"] = (m.qc_amplicon_suspect
                                     & m.qc_declared_strategy.str.contains("WGS", na=False))
    # ── accuracy: the declaration disagrees with the data, or the record
    #    disagrees with itself. Distinct from completeness, which asks only
    #    whether a value is present.
    GUT = ("metagenome",)
    declared_nonmeta = m.qc_declared_scientific_name.ne("") & ~m.qc_declared_scientific_name.str.contains(
        "|".join(GUT), na=False)
    # the converse error: an alarming label on data that is a normal community
    m["qc_label_misleading"] = declared_nonmeta & (m.richness_10pct.fillna(0) >= 20)

    def _pos(c):
        return m[c].astype(str).str.strip().str.lower().isin(["yes", "1", "1.0"]) if c in m.columns \
            else pd.Series(False, index=m.index)
    m["qc_health_contradiction"] = _pos("Healthy") & m["Unhealthy"].astype(str).str.strip().str.lower().eq("unhealthy")

    bmi = pd.to_numeric(m.get("BMI"), errors="coerce")
    m["qc_bmi_implausible"] = (bmi.notna() & ((bmi < 8) | (bmi > 90))).fillna(False)
    m["qc_bmi_obesity_mismatch"] = ((bmi.notna() & (bmi >= 30) & ~_pos("Obesity"))
                                    | (bmi.notna() & (bmi < 25) & _pos("Obesity"))).fillna(False)

    dis = [c for c in m.columns if c.startswith("disease_")]
    anyflag = pd.concat([_pos(c) for c in dis], axis=1).any(axis=1) if dis else pd.Series(False, index=m.index)
    m["qc_disease_not_rolled_up"] = anyflag & ~_pos("Diseased (all)")

    mb = pd.to_numeric(m.get("Total bases"), errors="coerce")
    ratio = mb / m.qc_archive_bases.replace(0, np.nan)
    m["qc_depth_disagrees_archive"] = (m.qc_all_runs_resolved
                                       & ratio.notna() & ((ratio > 1.5) | (ratio < 0.67))).fillna(False)

    dupe = m["Chosen_match_ID"].astype(str).str.strip()
    m["qc_duplicate_id"] = dupe.duplicated(keep=False)

    # Two severities, kept apart. Conflating them makes a study whose submitter
    # typed "Homo sapiens" look as compromised as one carrying amplicon libraries:
    # CosteaPI 2017 is 100% "flagged" on the combined measure and 0.4% on the one
    # that matters.
    EXCLUDE = ["qc_amplicon_suspect", "qc_low_richness", "qc_isolate_like",
               "qc_no_matrix_row"]                      # the data is not usable
    RECORD = ["qc_strategy_contradicted", "qc_label_misleading",
              "qc_health_contradiction", "qc_bmi_implausible",
              "qc_bmi_obesity_mismatch", "qc_disease_not_rolled_up",
              "qc_depth_disagrees_archive", "qc_duplicate_id"]   # the record is wrong
    flags = EXCLUDE + RECORD
    m["qc_flags"] = [";".join(f[3:] for f in flags if row[f]) for _, row in m[flags].iterrows()]
    m["qc_exclude_recommended"] = m[EXCLUDE].any(axis=1)
    m["qc_record_issue"] = m[RECORD].any(axis=1)
    m["qc_any_flag"] = m.qc_flags.ne("")

    m.drop(columns=["_k", "Sample_ID"]).to_csv(args.out_meta, index=False)
    print(f"\nwrote {args.out_meta.relative_to(ROOT)} ({len(m):,} rows, "
          f"{len(m.columns)-2} columns)")
    for f in flags + ["qc_exclude_recommended", "qc_record_issue", "qc_any_flag"]:
        print(f"    {f:26s} {int(m[f].sum()):5,d}")

    # ── study-level table, in the shape of Table S1 ──────────────────────
    rows = []
    for (a, y), g in m.groupby(["Author", "Year"], dropna=False):
        rec = {"Author": a, "Year": y, "N": len(g),
               "Countries": g["Country"].nunique(),
               "BioProject": ";".join(sorted({str(x) for x in g["BioProject ID"].dropna()})[:2])}
        for c in COMPLETENESS:
            if c in g.columns:
                rec[f"pct_{c.replace(' ', '_')}"] = round(100 * (~_blank(g[c])).mean(), 1)
        rec["median_richness_10pct"] = round(float(g.richness_10pct.median(skipna=True)), 1) \
            if g.richness_10pct.notna().any() else np.nan
        rec["median_breadth_pct"] = round(float(g.median_breadth_pct.median(skipna=True)), 4) \
            if g.median_breadth_pct.notna().any() else np.nan
        for f in flags:
            rec[f"n_{f[3:]}"] = int(g[f].sum())
        rec["n_exclude_recommended"] = int(g.qc_exclude_recommended.sum())
        rec["pct_exclude_recommended"] = round(100 * g.qc_exclude_recommended.mean(), 1)
        rec["n_record_issue"] = int(g.qc_record_issue.sum())
        rec["pct_record_issue"] = round(100 * g.qc_record_issue.mean(), 1)
        rows.append(rec)
    t = pd.DataFrame(rows).sort_values("N", ascending=False)
    t.to_csv(args.out_table, index=False)
    print(f"\nwrote {args.out_table.relative_to(ROOT)} ({len(t)} studies, {len(t.columns)} columns)")
    print("\n  studies with the most flagged samples:")
    print(t.nlargest(6, "n_exclude_recommended")[
        ["Author", "Year", "N", "n_exclude_recommended", "pct_exclude_recommended",
         "n_record_issue", "pct_record_issue"]].to_string(index=False))
    ACC = ["qc_strategy_contradicted", "qc_label_misleading", "qc_isolate_like",
           "qc_health_contradiction", "qc_bmi_obesity_mismatch", "qc_bmi_implausible",
           "qc_disease_not_rolled_up", "qc_depth_disagrees_archive", "qc_duplicate_id"]
    print("\n  accuracy: records that disagree with the data or with themselves")
    for f in ACC:
        n = int(m[f].sum())
        print(f"    {f[3:]:26s} {n:6,d}   {100*n/len(m):5.2f}%")
    print("\n  field completeness across the cohort:")
    for c in COMPLETENESS:
        col = f"pct_{c.replace(' ', '_')}"
        if col in t.columns:
            overall = 100 * (~_blank(m[c])).mean()
            n_studies_zero = int((t[col] == 0).sum())
            print(f"    {c:18s} {overall:5.1f}%   studies with none: {n_studies_zero}")


if __name__ == "__main__":
    main()
