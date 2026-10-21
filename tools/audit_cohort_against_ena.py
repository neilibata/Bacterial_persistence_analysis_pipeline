#!/usr/bin/env python3
"""Check every sample in the cohort against the archive's own description of it.

The inclusion criteria -- human gut, shotgun WGS, metagenomic, non-infant -- were
applied during curation, but nothing re-checks them against what ENA says the
sample actually is. Three isolate genomes of Bifidobacterium longum were sitting
in the hunter-gatherer group as Hadza gut metagenomes until a spot check found
them, so this does the check exhaustively.

Queries ENA once per BioProject (88 of them) and flags any sample whose archive
record disagrees with the criteria:

  not_gut_metagenome   scientific_name is not a gut/metagenome term
  not_wgs              library_strategy is not WGS (AMPLICON, RNA-Seq, ...)
  not_metagenomic      library_source is not METAGENOMIC
  not_found_in_ena     the accession returns nothing

Writes a per-sample report. Excludes nothing: what to do about a flagged sample
is a curation decision, and some flags are benign (a submitter labelling a stool
metagenome "Homo sapiens").

    python tools/audit_cohort_against_ena.py
    python tools/audit_cohort_against_ena.py --cache-only    # re-score without refetching
"""
from __future__ import annotations

import argparse, io, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEARCH = "https://www.ebi.ac.uk/ena/portal/api/search"
FIELDS = ["run_accession", "sample_accession", "scientific_name", "library_strategy",
          "library_source", "library_selection", "instrument_platform",
          "read_count", "base_count"]

GUT_OK = ("gut metagenome", "human gut metagenome", "feces metagenome",
          "fecal metagenome", "faecal metagenome", "metagenome")
# Submitters sometimes register a stool metagenome under the host species; that is
# sloppy but not disqualifying, so it is reported separately from a real mismatch.
HOST_LABELS = ("homo sapiens",)


def _fetch_project(prj: str) -> pd.DataFrame:
    q = urllib.parse.urlencode({"result": "read_run", "query": f"study_accession={prj}",
                                "fields": ",".join(FIELDS), "format": "tsv", "limit": 0})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{SEARCH}?{q}", timeout=120) as r:
                txt = r.read().decode("utf-8", "replace")
            if txt.strip():
                return pd.read_csv(io.StringIO(txt), sep="\t", dtype=str)
            return pd.DataFrame(columns=FIELDS)
        except Exception as exc:
            if attempt == 2:
                print(f"    {prj}: failed ({exc})")
                return pd.DataFrame(columns=FIELDS)
            time.sleep(2 * (attempt + 1))
    return pd.DataFrame(columns=FIELDS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "metadata_Aug_26_2026_organised.csv")
    ap.add_argument("--cache", type=Path, default=ROOT / "docs" / "ena_run_cache.tsv")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "cohort_ena_audit.csv")
    ap.add_argument("--cache-only", action="store_true")
    args = ap.parse_args()

    m = pd.read_csv(args.src, low_memory=False)
    projects = sorted({p for p in m["BioProject ID"].astype(str).str.strip()
                       if p not in ("nan", "", "none")})
    print(f"{len(m):,} samples across {len(projects)} BioProjects")

    if args.cache_only and args.cache.exists():
        ena = pd.read_csv(args.cache, sep="\t", dtype=str)
        print(f"  using cached {len(ena):,} run rows")
    else:
        frames = []
        for i, p in enumerate(projects, 1):
            d = _fetch_project(p)
            if len(d):
                frames.append(d)
            print(f"  {i}/{len(projects)} {p} -> {len(d):,} runs", end="\r")
        print()
        ena = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FIELDS)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        ena.to_csv(args.cache, sep="\t", index=False)
        print(f"  fetched {len(ena):,} run rows -> {args.cache.name}")

    # collapse runs to samples; a sample is WGS if any of its runs is
    ena["_sn"] = ena["scientific_name"].astype(str).str.strip().str.lower()
    ena["_ls"] = ena["library_strategy"].astype(str).str.strip().str.upper()
    ena["_src"] = ena["library_source"].astype(str).str.strip().str.upper()
    ena["_sel"] = ena["library_selection"].astype(str).str.strip().str.upper()
    g = ena.groupby("sample_accession").agg(
        scientific_name=("_sn", lambda s: s.mode().iloc[0] if len(s.mode()) else ""),
        strategies=("_ls", lambda s: ";".join(sorted(set(s)))),
        sources=("_src", lambda s: ";".join(sorted(set(s)))),
        selections=("_sel", lambda s: ";".join(sorted(set(s)))),
        n_runs=("run_accession", "nunique"),
        read_count=("read_count", lambda s: pd.to_numeric(s, errors="coerce").sum()),
        base_count=("base_count", lambda s: pd.to_numeric(s, errors="coerce").sum()),
    ).reset_index()

    # Join on the run accession, not BioSample: for several studies the BioSample
    # column holds an internal identifier (SID10042055_SF05) rather than a SAMN/SAMEA
    # accession, and joining on it loses 3,146 samples that are present in ENA.
    run_to_sample = dict(zip(ena["run_accession"].astype(str).str.strip(),
                             ena["sample_accession"].astype(str).str.strip()))
    # SRA_Accession may hold several runs for one sample, semicolon-delimited
    # ("ERR209908;ERR209907;..."). Match on any constituent run.
    def _first_known(cell: str):
        for r in str(cell).split(";"):
            r = r.strip()
            if r in run_to_sample:
                return run_to_sample[r]
        return None

    m["_run"] = m["SRA_Accession"].astype(str).str.strip()
    m["_sa"] = m["_run"].map(_first_known)
    m["_bs"] = m["BioSample"].astype(str).str.strip()
    m["_key"] = m["_sa"].fillna(m["_bs"])
    j = m[["Chosen_match_ID", "_key", "_run", "Author", "Year", "Country", "Lifestyle",
           "BioProject ID"]].merge(g, left_on="_key", right_on="sample_accession", how="left")

    def flags(r):
        f = []
        if pd.isna(r.sample_accession):
            return "not_found_in_ena"
        sn = str(r.scientific_name)
        if not any(k in sn for k in GUT_OK):
            f.append("host_labelled" if sn in HOST_LABELS else "not_gut_metagenome")
        st = str(r.strategies)
        if st and "WGS" not in st:
            f.append("not_wgs")
        if str(r.sources) and "METAGENOMIC" not in str(r.sources):
            f.append("not_metagenomic")
        return ";".join(f)

    j["qc_flags"] = j.apply(flags, axis=1)   # not "flags": DataFrame.flags is reserved
    j.drop(columns=["_key"]).to_csv(args.out, index=False)

    flagged = j[j["qc_flags"] != ""]
    print(f"\n  matched to ENA: {int(j.sample_accession.notna().sum()):,}/{len(j):,}")
    print(f"  flagged        : {len(flagged):,}\n")
    from collections import Counter
    c = Counter(f for s in flagged["qc_flags"] for f in s.split(";") if f)
    for k, v in c.most_common():
        print(f"    {k:22s} {v:5,d}")
    print(f"\n  wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
