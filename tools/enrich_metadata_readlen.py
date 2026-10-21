#!/usr/bin/env python3
"""Fill missing read length and read count in the sample metadata (Table M1) from the SRA.

Read length is a per-sequencing-run property and is near-constant within a study
(StudyFolder). For every study with missing ``median_read_length`` we poll a few of
its still-missing runs from the NCBI SRA (efetch runinfo), derive the per-read length
from ``avgLength`` (spot length / 2 for PAIRED layouts), and impute the study's read
length. ``number_reads`` is then computed for every sample as ``Total bases / read
length`` — a relation verified against the samples that already carry both values.

Two transparency columns are added: ``median_read_length_source`` and
``number_reads_source`` (``original`` vs ``SRA_imputed`` / ``computed``); a per-study
summary is written alongside.

Usage:
    python tools/enrich_metadata_readlen.py \
        --meta metadata_Mar_16_2026.csv --out metadata_M1.csv --per-study 5
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
RL, NR, TB, SRR, STUDY = ("median_read_length", "number_reads", "Total bases",
                          "SRA_Accession", "StudyFolder")

#: NCBI asks every E-utilities caller to identify itself with a tool name and a
#: contact address, so that it can get in touch before throttling a client that
#: misbehaves. The address has to be whoever is running the query, which is why
#: it is read from the environment rather than written into the file: a
#: hard-coded address is published to everyone who reads the repository, and it
#: attributes a reader's requests to somebody who would never see the warning.
#: Leaving it unset is allowed -- NCBI serves anonymous callers at a lower rate
#: limit -- so this degrades instead of failing.
_EMAIL_VARS = ("NCBI_EMAIL", "ENTREZ_EMAIL")


def entrez_email() -> str:
    """The contact address to send to NCBI, from the environment ('' if unset)."""
    for var in _EMAIL_VARS:
        v = os.environ.get(var, "").strip()
        if v:
            return v
    return ""


def efetch_runinfo(accessions, batch=80, delay=0.4, tool="microbiome_metadata_explorer",
                   email=None, retries=3):
    """Return {run_accession: {avgLength, spots, layout}} from SRA runinfo."""
    out = {}
    if email is None:
        email = entrez_email()
    if not email:
        print(f"    [note] no contact address for NCBI; set "
              f"{_EMAIL_VARS[0]} to raise the E-utilities rate limit")
    accessions = list(dict.fromkeys(a for a in accessions if isinstance(a, str) and a))
    for i in range(0, len(accessions), batch):
        chunk = accessions[i:i + batch]
        query = {"db": "sra", "id": ",".join(chunk),
                 "rettype": "runinfo", "retmode": "text", "tool": tool}
        if email:
            query["email"] = email
        params = urllib.parse.urlencode(query)
        url = f"{EUTILS}?{params}"
        text = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    text = r.read().decode("utf-8", "replace")
                break
            except Exception as e:  # noqa: BLE001
                if attempt == retries - 1:
                    print(f"    [warn] batch {i//batch} failed: {e}")
                else:
                    time.sleep(1.5 * (attempt + 1))
        if not text:
            continue
        for row in csv.DictReader(io.StringIO(text)):
            run = (row.get("Run") or "").strip()
            if not run:
                continue
            try:
                avg = float(row.get("avgLength") or "nan")
            except ValueError:
                avg = float("nan")
            try:
                spots = int(float(row.get("spots") or 0))
            except ValueError:
                spots = 0
            out[run] = {"avgLength": avg, "spots": spots,
                        "layout": (row.get("LibraryLayout") or "").strip().upper()}
        print(f"    fetched {min(i+batch, len(accessions))}/{len(accessions)} runs")
        time.sleep(delay)
    return out


def read_len_from(info):
    avg, layout = info.get("avgLength"), info.get("layout")
    if avg is None or not np.isfinite(avg) or avg <= 0:
        return None
    return avg / 2.0 if layout == "PAIRED" else avg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("metadata_M1.csv"))
    ap.add_argument("--per-study", type=int, default=5, help="runs to poll per study")
    ap.add_argument("--summary", type=Path, default=Path("tools/metadata_explorer/readlen_imputation_summary.csv"))
    ap.add_argument("--dry-run-studies", type=int, default=0, help="limit to N studies (testing)")
    args = ap.parse_args()

    df = pd.read_csv(args.meta, low_memory=False)
    df[RL] = pd.to_numeric(df[RL], errors="coerce")
    df[NR] = pd.to_numeric(df[NR], errors="coerce")
    df[TB] = pd.to_numeric(df[TB], errors="coerce")

    # studies that still need read length
    need = [s for s, g in df.groupby(STUDY) if g[RL].isna().any()]
    if args.dry_run_studies:
        need = need[:args.dry_run_studies]
    print(f"{df[RL].isna().sum():,} samples missing read length across {len(need)} studies")

    # sample missing runs per study to poll
    to_poll, by_study = [], {}
    for s in need:
        g = df[df[STUDY] == s]
        miss = g[g[RL].isna()]
        picks = miss[SRR].dropna().head(args.per_study).tolist()
        by_study[s] = picks
        to_poll += picks
    print(f"polling {len(to_poll)} runs from SRA …")
    info = efetch_runinfo(to_poll)

    # impute per-study read length from polled runs + any existing values
    summary, rl_by_study = [], {}
    for s in need:
        g = df[df[STUDY] == s]
        polled = [read_len_from(info[a]) for a in by_study[s] if a in info]
        polled = [x for x in polled if x]
        existing = g[RL].dropna().tolist()
        vals = polled + existing
        rl = float(np.median(vals)) if vals else None
        rl_by_study[s] = rl
        summary.append({"StudyFolder": s, "n_polled": len(polled),
                        "read_len_polled_median": (round(np.median(polled), 1) if polled else None),
                        "read_len_existing_median": (round(np.median(existing), 1) if existing else None),
                        "read_len_polled_spread": (round(float(np.std(polled)), 1) if len(polled) > 1 else 0),
                        "imputed_read_len": (round(rl, 1) if rl else None),
                        "layouts": ",".join(sorted({info[a]["layout"] for a in by_study[s] if a in info}))})

    # fill read length (imputed) and record source
    df["median_read_length_source"] = np.where(df[RL].notna(), "original", "")
    for s, rl in rl_by_study.items():
        if rl is None:
            continue
        m = (df[STUDY] == s) & df[RL].isna()
        df.loc[m, RL] = round(rl)
        df.loc[m, "median_read_length_source"] = "SRA_imputed"

    # compute number_reads = Total bases / read length where missing
    df["number_reads_source"] = np.where(df[NR].notna(), "original", "")
    calc = df[NR].isna() & df[RL].notna() & df[TB].notna() & (df[RL] > 0)
    df.loc[calc, NR] = (df.loc[calc, TB] / df.loc[calc, RL]).round()
    df.loc[calc, "number_reads_source"] = "computed"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(args.summary, index=False)

    print(f"\nread length now missing: {df[RL].isna().sum():,} "
          f"(imputed {int((df['median_read_length_source']=='SRA_imputed').sum()):,})")
    print(f"number_reads now missing: {df[NR].isna().sum():,} "
          f"(computed {int((df['number_reads_source']=='computed').sum()):,})")
    print(f"wrote {args.out}  +  {args.summary}")
    flagged = [r for r in summary if r["read_len_polled_spread"] and r["read_len_polled_spread"] > 10]
    if flagged:
        print(f"\n[note] {len(flagged)} studies show variable polled read length (>10 bp SD) — "
              "per-run querying may be warranted:")
        for r in flagged[:10]:
            print(f"    {r['StudyFolder']}: {r['layouts']} spread={r['read_len_polled_spread']}")


if __name__ == "__main__":
    main()
