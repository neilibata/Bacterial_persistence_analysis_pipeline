#!/usr/bin/env python3
"""Put real BioSample accessions in the BioSample column.

For 3,079 of the 18,303 samples the ``BioSample`` column holds the submitter's
own sample name -- ``MH0001``, ``SID10042055_SF05``, ``PRIMM0001`` -- rather
than a BioSample accession. That is not missing data so much as a mislabelled
column: the accession is usually sitting in ``SRS ID`` as an SRA/ENA sample
accession (ERS/SRS/DRS), which is a different namespace from BioSample
(SAMN/SAMEA) but identifies the same sample.

The consequence is practical: any repository lookup keyed on ``BioSample``
fails for those rows, which is how an attempt to audit study metadata against
SRA/ENA came back empty for whole studies that are in fact well described.

This resolves ERS/SRS/DRS to their BioSample accession through ENA's portal
API, in batches, cached to disk so a re-run costs nothing. The submitter's name
is preserved in a new ``submitter_sample_name`` column rather than overwritten.

    python tools/fix_biosample_accessions.py --dry-run
    python tools/fix_biosample_accessions.py
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_IO = ROOT / "final_run" / "metadata_M1.csv"
CACHE = ROOT / "docs" / "ena_sample_accession_cache.tsv"
PORTAL = "https://www.ebi.ac.uk/ena/portal/api/search"

BIOSAMPLE = re.compile(r"^SAM[END]A?\d+$")
SRA_SAMPLE = re.compile(r"^(SRS\d+|ERS\d+|DRS\d+)$")


def load_cache():
    if not CACHE.exists():
        return {}
    with CACHE.open() as fh:
        return {r["secondary"]: r["biosample"]
                for r in csv.DictReader(fh, delimiter="\t")
                if r.get("secondary")}


def save_cache(mapping):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["secondary", "biosample"],
                           delimiter="\t")
        w.writeheader()
        for k in sorted(mapping):
            w.writerow(dict(secondary=k, biosample=mapping[k]))


def resolve(accessions, batch=60, delay=0.34, verbose=True):
    """{ERS/SRS/DRS: SAMN/SAMEA} from ENA, batched."""
    out, todo = {}, sorted(set(accessions))
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        q = " OR ".join(f"secondary_sample_accession={a}" for a in chunk)
        url = (f"{PORTAL}?result=sample&format=tsv&limit=0"
               "&fields=sample_accession,secondary_sample_accession&query="
               + urllib.parse.quote(q))
        try:
            txt = urllib.request.urlopen(url, timeout=60).read().decode()
            for line in txt.strip().splitlines()[1:]:
                parts = line.split("\t")
                if len(parts) >= 2 and BIOSAMPLE.match(parts[0].strip()):
                    out[parts[1].strip()] = parts[0].strip()
        except Exception as exc:
            if verbose:
                print(f"      batch {i//batch + 1}: {type(exc).__name__}")
        if verbose:
            print(f"      {min(i + batch, len(todo)):>5}/{len(todo)} queried, "
                  f"{len(out):>5} resolved", end="\r", flush=True)
        time.sleep(delay)
    if verbose:
        print()
    return out


def resolve_via_runs(runs, batch=1, delay=0.34, verbose=True):
    """{run accession: (biosample, sample_alias)} from ENA's run report.

    The fallback for samples that carry no sample accession at all: a run
    always knows which sample it came from. `sample_alias` comes back too,
    which is the submitter's own name -- so the mapping can be checked against
    what the project called the sample rather than taken on trust.
    """
    out = {}
    todo = sorted(set(runs))
    for i, acc in enumerate(todo, start=1):
        q = urllib.parse.urlencode({
            "accession": acc, "result": "read_run", "format": "tsv",
            "fields": "run_accession,sample_accession,sample_alias"})
        try:
            txt = urllib.request.urlopen(
                f"https://www.ebi.ac.uk/ena/portal/api/filereport?{q}",
                timeout=45).read().decode()
            lines = txt.strip().splitlines()
            if len(lines) > 1:
                parts = lines[1].split("\t")
                if len(parts) >= 2 and BIOSAMPLE.match(parts[1].strip()):
                    out[acc] = (parts[1].strip(),
                                parts[2].strip() if len(parts) > 2 else "")
        except Exception:
            pass
        if verbose and (i % 10 == 0 or i == len(todo)):
            print(f"      {i:>4}/{len(todo)} runs queried, {len(out):>4} resolved",
                  end="\r", flush=True)
        time.sleep(delay)
    if verbose:
        print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io", default=str(DEFAULT_IO))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-network", action="store_true",
                    help="use only what is already cached")
    a = ap.parse_args()

    m = pd.read_csv(a.io, low_memory=False)
    bs = m["BioSample"].astype(str).str.strip()
    srs = m["SRS ID"].astype(str).str.strip()
    bad = ~bs.map(lambda v: bool(BIOSAMPLE.match(v)))
    have_srs = srs.map(lambda v: bool(SRA_SAMPLE.match(v)))

    print(f"  {Path(a.io).name}: {len(m):,} rows")
    print(f"    BioSample already an accession : {int((~bad).sum()):,}")
    print(f"    holds a submitter name         : {int(bad.sum()):,}")
    print(f"      of which SRS ID is usable    : {int((bad & have_srs).sum()):,}")
    print(f"      neither usable               : {int((bad & ~have_srs).sum()):,}")

    cache = load_cache()
    need = sorted(set(srs[bad & have_srs]) - set(cache))
    print(f"\n  cache holds {len(cache):,} mappings; {len(need):,} still to look up")
    if need and not a.no_network:
        got = resolve(need)
        cache.update(got)
        print(f"    resolved {len(got):,} of {len(need):,}")
        if not a.dry_run:
            save_cache(cache)
            print(f"    cache -> {CACHE.relative_to(ROOT)}")

    mapped = srs.map(cache)
    fill = bad & have_srs & mapped.notna()

    # Fallback: a sample with no sample accession still has a run, and the run
    # knows its sample. This is what recovers the studies that carry only a
    # submitter name and a run.
    still = bad & ~fill
    run_map = {}
    if int(still.sum()) and not a.no_network:
        first_run = (m.loc[still, "SRA_Accession"].astype(str)
                     .str.split(";").str[0].str.strip())
        runs = [r for r in first_run.unique() if re.match(r"^[SDE]RR\d+$", r)]
        print(f"\n  {int(still.sum()):,} rows have no sample accession; "
              f"resolving {len(runs):,} run accessions instead")
        run_map = resolve_via_runs(runs)
        print(f"    resolved {len(run_map):,} of {len(runs):,}")
        alias_ok = sum(1 for r, (b, al) in run_map.items() if al)
        print(f"    of those, ENA also returned the submitter alias: {alias_ok:,}")
        m.loc[still, "_run_biosample"] = first_run.map(
            lambda r: run_map.get(r, ("", ""))[0])
        m.loc[still, "_run_alias"] = first_run.map(
            lambda r: run_map.get(r, ("", ""))[1])
        fill2 = still & m.get("_run_biosample", pd.Series("", index=m.index)).fillna("").ne("")
        print(f"    rows fillable from a run: {int(fill2.sum()):,}")
    else:
        fill2 = pd.Series(False, index=m.index)
    print(f"\n  rows that can be filled: {int(fill.sum()):,}")
    unresolved = bad & ~fill
    print(f"  rows left without an accession: {int(unresolved.sum()):,}")
    if int(unresolved.sum()):
        print("    ", dict(list(m.loc[unresolved, "StudyFolder"]
                                .value_counts().items())[:6]))

    if a.dry_run:
        print("\n  dry run; nothing written")
        return

    # Keep the submitter's name: it is how the source study's own tables refer
    # to the sample, which is exactly what a supplementary-table cross-check
    # needs.
    if "submitter_sample_name" not in m.columns:
        m["submitter_sample_name"] = ""
    m.loc[bad, "submitter_sample_name"] = bs[bad]
    m.loc[fill, "BioSample"] = mapped[fill]
    if int(fill2.sum()):
        m.loc[fill2, "BioSample"] = m.loc[fill2, "_run_biosample"]
        # Check the recovered accession against what the project called the
        # sample, where ENA gave us both.
        al = m.loc[fill2, "_run_alias"].astype(str).str.strip()
        old_name = bs[fill2].astype(str).str.strip()
        agree = int((al != "") .sum()), int(((al != "") & (al == old_name)).sum())
        print(f"    alias returned for {agree[0]:,}; matches the stored sample "
              f"name on {agree[1]:,}")
    for tmp in ("_run_biosample", "_run_alias"):
        if tmp in m.columns:
            m = m.drop(columns=[tmp])

    m.to_csv(a.io, index=False)
    final = m["BioSample"].astype(str).str.strip().map(
        lambda v: bool(BIOSAMPLE.match(v)))
    print(f"\n  wrote {Path(a.io).relative_to(ROOT)}")
    print(f"    BioSample is now an accession on {int(final.sum()):,}/{len(m):,} "
          f"rows ({100 * final.mean():.1f}%)")
    print(f"    submitter names preserved in 'submitter_sample_name': "
          f"{int((m['submitter_sample_name'].astype(str).str.strip() != '').sum()):,}")


if __name__ == "__main__":
    main()
