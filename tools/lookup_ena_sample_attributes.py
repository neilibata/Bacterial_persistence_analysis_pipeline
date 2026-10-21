#!/usr/bin/env python3
"""Fetch host attributes and read counts from ENA for samples missing them.

The 260 samples added on 2026-08-20 arrived with sequencing linkage but no
demographics: Gender, Age, BMI and number_reads are blank for all of them, and
none is recoverable from within the file because they vary per person. This asks
ENA what the submitters actually deposited.

Writes a lookup table; it does not modify any metadata. Whether a returned value
should be trusted over a blank is a curation decision -- ENA host attributes are
free-text submitter fields of uneven quality, and 'age' may be a number, a range
or a life stage.

    python tools/lookup_ena_sample_attributes.py --ids-from-new
"""
from __future__ import annotations

import argparse, io, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENA = "https://www.ebi.ac.uk/ena/portal/api/filereport"
SAMPLE_FIELDS = ["sample_accession", "age", "sex", "host_sex", "submitted_host_sex",
                 "host_phenotype", "host_status", "dev_stage", "description"]
RUN_FIELDS = ["run_accession", "sample_accession", "read_count", "base_count",
              "library_layout", "instrument_model"]
NULLISH = {"nan", "", "none", "not collected", "missing", "not applicable",
           "n/a", "unknown", "not provided", "restricted access"}


def _fetch(accessions, result, fields, chunk=60):
    frames = []
    for i in range(0, len(accessions), chunk):
        part = [a for a in accessions[i:i + chunk] if a and a != "nan"]
        if not part:
            continue
        q = urllib.parse.urlencode({"accession": ",".join(part), "result": result,
                                    "fields": ",".join(fields), "format": "tsv"})
        txt = ""
        for attempt in range(3):
            try:
                with urllib.request.urlopen(f"{ENA}?{q}", timeout=90) as r:
                    txt = r.read().decode("utf-8", "replace")
                break
            except Exception as exc:
                if attempt == 2:
                    print(f"    chunk {i//chunk} failed: {exc}")
                else:
                    time.sleep(2 * (attempt + 1))
        if txt.strip():
            frames.append(pd.read_csv(io.StringIO(txt), sep="\t", dtype=str))
        print(f"    {min(i+chunk, len(accessions))}/{len(accessions)}", end="\r")
    print()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=fields)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "metadata_Aug_26_2026_organised.csv")
    ap.add_argument("--baseline", type=Path, default=ROOT / "archive" / "Metadata_M1_081426.csv")
    ap.add_argument("--ids-from-new", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "ena_sample_attributes.csv")
    args = ap.parse_args()

    m = pd.read_csv(args.src, low_memory=False)
    base = pd.read_csv(args.baseline, low_memory=False)
    added = set(m["Chosen_match_ID"].astype(str).str.strip()) - set(
        base["Chosen_match_ID"].astype(str).str.strip())
    sub = m[m["Chosen_match_ID"].astype(str).str.strip().isin(added)]
    biosamples = sub["BioSample"].dropna().astype(str).str.strip().unique().tolist()
    runs = sub["SRA_Accession"].dropna().astype(str).str.strip().unique().tolist()
    print(f"looking up {len(biosamples)} BioSamples and {len(runs)} runs")

    print("  sample attributes:")
    smp = _fetch(biosamples, "sample", SAMPLE_FIELDS)
    print("  run statistics:")
    run = _fetch(runs, "read_run", RUN_FIELDS)

    out = smp
    if len(run):
        agg = run.groupby("sample_accession").agg(
            read_count=("read_count", lambda s: pd.to_numeric(s, errors="coerce").sum()),
            base_count=("base_count", lambda s: pd.to_numeric(s, errors="coerce").sum()),
            n_runs=("run_accession", "nunique"),
            library_layout=("library_layout", "first"),
            instrument_model=("instrument_model", "first")).reset_index()
        out = out.merge(agg, on="sample_accession", how="outer")
    key = sub[["BioSample", "Chosen_match_ID", "Author", "Year", "Country"]].copy()
    key["BioSample"] = key["BioSample"].astype(str).str.strip()
    out = out.merge(key, left_on="sample_accession", right_on="BioSample", how="left")
    out.to_csv(args.out, index=False)

    print(f"\n  wrote {args.out.relative_to(ROOT)}  ({len(out)} rows)")
    print("\n  usable values returned by ENA:")
    for c in ["age", "sex", "host_sex", "submitted_host_sex", "host_phenotype",
              "dev_stage", "read_count", "base_count"]:
        if c in out.columns:
            v = out[c].astype(str).str.strip().str.lower()
            print(f"    {c:22s} {int((~v.isin(NULLISH)).sum()):4d}/{len(out)}")


if __name__ == "__main__":
    main()
