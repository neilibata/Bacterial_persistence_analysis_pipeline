#!/usr/bin/env python3
"""Per-sample read-length resolution from ENA for the hard cases.

Refines ``metadata_M1.csv`` for two groups that per-study median imputation handles
poorly:
  * studies with variable within-study read length (``--studies``): every sample is
    resolved individually and its imputed value overwritten with the exact one;
  * samples still missing read length (typically ones whose ``SRA_Accession`` is an
    ENA *sample* accession, ERS…, that SRA efetch cannot resolve) — these are looked
    up via their actual run accessions in ``Paired SRR List`` / sample accession.

For each sample the ENA portal API (filereport, result=read_run) returns every run's
``read_count`` / ``base_count`` / ``library_layout``; per-read length is
``base_count / read_count`` (halved for PAIRED, calibrated against known values), taken
from the sample's largest run. ``number_reads`` is recomputed as ``Total bases /
read_length``. Sources are recorded as ``ENA_perrun``.

Usage:
    python tools/resolve_read_length_ena.py --meta metadata_M1.csv \
        --studies ChengpingW_2017_CHN KiminaKM_2019_RUS Valles-ColomerM_2023_GNB_fastq
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ENA = "https://www.ebi.ac.uk/ena/portal/api/filereport"
RL, NR, TB = "median_read_length", "number_reads", "Total bases"
RUN_RE = re.compile(r"^[SED]RR\d+$")
SAMPLE_RE = re.compile(r"^([SED]RS\d+|SAM[END]\w+)$")
_cache: dict[str, list] = {}


def ena_runs(acc, delay=0.12, retries=3):
    acc = str(acc).strip()
    if not acc or acc in _cache:
        return _cache.get(acc, [])
    params = urllib.parse.urlencode({"accession": acc, "result": "read_run",
                                     "fields": "run_accession,read_count,base_count,library_layout",
                                     "format": "tsv"})
    rows = []
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{ENA}?{params}", timeout=45) as r:
                text = r.read().decode("utf-8", "replace")
            rdr = csv.DictReader(io.StringIO(text), delimiter="\t")
            for row in rdr:
                try:
                    bc = int(float(row.get("base_count") or 0))
                    rc = int(float(row.get("read_count") or 0))
                except ValueError:
                    continue
                if bc > 0 and rc > 0:
                    rows.append({"run": row.get("run_accession", ""), "bc": bc, "rc": rc,
                                 "layout": (row.get("library_layout") or "").strip().upper()})
            break
        except Exception:  # noqa: BLE001
            time.sleep(1.0 * (attempt + 1))
    time.sleep(delay)
    _cache[acc] = rows
    return rows


def resolve(row):
    """Return per-read length (rounded) for one metadata row, or None."""
    runs = []
    # 1) sample-level accession returns all of the sample's runs in one call
    for col in ("SRS ID", "SRA_Accession", "BioSample", "Index"):
        acc = str(row.get(col, "")).strip()
        if SAMPLE_RE.match(acc):
            runs = ena_runs(acc)
            if runs:
                break
    # 2) fall back to explicit run accessions
    if not runs:
        cands = []
        for tok in str(row.get("Paired SRR List", "")).split(";"):
            tok = tok.strip()
            if RUN_RE.match(tok):
                cands.append(tok)
        acc = str(row.get("SRA_Accession", "")).strip()
        if RUN_RE.match(acc):
            cands.append(acc)
        for ra in dict.fromkeys(cands):
            runs += ena_runs(ra)
    if not runs:
        return None
    rep = max(runs, key=lambda d: d["bc"])          # sample's largest run
    per = rep["bc"] / rep["rc"]
    if rep["layout"] == "PAIRED":
        per /= 2.0
    return round(per) if np.isfinite(per) and per > 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", type=Path, default=Path("metadata_M1.csv"))
    ap.add_argument("--studies", nargs="*", default=[])
    ap.add_argument("--summary", type=Path, default=Path("tools/metadata_explorer/readlen_perrun_summary.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.meta, low_memory=False)
    for c in (RL, NR, TB):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "median_read_length_source" not in df:
        df["median_read_length_source"] = np.where(df[RL].notna(), "original", "")
    if "number_reads_source" not in df:
        df["number_reads_source"] = np.where(df[NR].notna(), "original", "")

    target = (df["StudyFolder"].isin(args.studies)) | (df[RL].isna())
    idx = df.index[target].tolist()
    print(f"resolving {len(idx)} samples per-run via ENA "
          f"({int(df['StudyFolder'].isin(args.studies).sum())} in flagged studies + "
          f"{int(df[RL].isna().sum())} still-missing) …")

    resolved = 0
    for n, i in enumerate(idx, 1):
        rl = resolve(df.loc[i])
        if rl:
            df.at[i, RL] = rl
            df.at[i, "median_read_length_source"] = "ENA_perrun"
            if df.at[i, TB] and rl > 0:
                df.at[i, NR] = round(df.at[i, TB] / rl)
                df.at[i, "number_reads_source"] = "ENA_perrun"
            resolved += 1
        if n % 100 == 0:
            print(f"  {n}/{len(idx)} processed ({resolved} resolved)")

    df.to_csv(args.meta, index=False)
    # per-study summary of the now-exact values
    rows = []
    for s in args.studies:
        g = df[df["StudyFolder"] == s]
        rows.append({"StudyFolder": s, "n": len(g),
                     "read_len_min": g[RL].min(), "read_len_median": g[RL].median(),
                     "read_len_max": g[RL].max(), "read_len_sd": round(float(g[RL].std()), 1)})
    if rows:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.summary, index=False)
        print("\nper-study exact read length:")
        for r in rows:
            print(f"    {r['StudyFolder']}: n={r['n']} range {r['read_len_min']:.0f}–{r['read_len_max']:.0f} "
                  f"median {r['read_len_median']:.0f} sd {r['read_len_sd']}")

    print(f"\nresolved {resolved}/{len(idx)} · read length still missing: {int(df[RL].isna().sum())} "
          f"· number_reads still missing: {int(df[NR].isna().sum())}")
    print(f"updated {args.meta}")


if __name__ == "__main__":
    main()
