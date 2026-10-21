#!/usr/bin/env python3
"""Replace boilerplate Sample IDs with the submitter's real sample name.

Some samples carry a description where their identifier should be --
"Metagenome or environmental sample from human gut metagenome", repeated
across every sample in a study. The cause is visible in the repository record:
ENA exposes both a `sample_title` (boilerplate written by the submission tool)
and a `sample_alias` (what the study actually calls the sample), and the
ingestion took the title.

The alias is still there, so this fetches it and puts it back. It is not only
tidier: BrambleMS_2021_COD's aliases are `Konzo1`, `Konzo2`... which name the
condition the cohort was assembled to study, and BritoIL_2016_FJI's `W1.3.ST`
encodes village, household and sample type. That is the information a
supplementary-table cross-check needs.

Only values shared by more than one sample are touched -- a unique identifier,
however odd it looks, is left alone.

    python tools/repair_generic_sample_ids.py --dry-run
    python tools/repair_generic_sample_ids.py
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_IO = ROOT / "final_run" / "metadata_M1.csv"
CHANGELOG = ROOT / "final_run" / "sample_id_repair.csv"
CACHE = ROOT / "docs" / "ena_sample_alias_cache.tsv"
PORTAL = "https://www.ebi.ac.uk/ena/portal/api/search"
TAB = "\t"


def load_cache():
    if not CACHE.exists():
        return {}
    with CACHE.open() as fh:
        return {r["accession"]: r["alias"]
                for r in csv.DictReader(fh, delimiter=TAB) if r.get("accession")}


def save_cache(d):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["accession", "alias"], delimiter=TAB)
        w.writeheader()
        for k in sorted(d):
            w.writerow(dict(accession=k, alias=d[k]))


def fetch_aliases(accessions, batch=50, delay=0.34, verbose=True):
    out = {}
    todo = sorted(set(accessions))
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        q = " OR ".join(f"accession={a}" for a in chunk)
        url = (f"{PORTAL}?result=sample&format=tsv&limit=0"
               "&fields=sample_accession,sample_alias&query="
               + urllib.parse.quote(q))
        try:
            txt = urllib.request.urlopen(url, timeout=60).read().decode()
            for line in txt.strip().splitlines()[1:]:
                p = line.split(TAB)
                if len(p) >= 2 and p[1].strip():
                    out[p[0].strip()] = p[1].strip()
        except Exception as exc:
            if verbose:
                print(f"      batch {i // batch + 1}: {type(exc).__name__}")
        if verbose:
            print(f"      {min(i + batch, len(todo)):>5}/{len(todo)} queried, "
                  f"{len(out):>5} aliases", end="\r", flush=True)
        time.sleep(delay)
    if verbose:
        print()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io", default=str(DEFAULT_IO))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    m = pd.read_csv(a.io, low_memory=False)
    sid = m["Sample ID"].astype(str).str.strip()
    vc = sid.value_counts()
    shared = set(vc[vc > 1].index)

    # A description, not an identifier: shared between samples and written as
    # prose. A shared *short* value is usually a subject id, which is a
    # different problem and not this tool's business.
    generic = sid.isin(shared) & (sid.str.contains(" ") | (sid.str.len() > 40))
    print(f"  rows whose Sample ID is shared prose: {int(generic.sum()):,}")
    for st, n in m.loc[generic, "StudyFolder"].value_counts().items():
        print(f"      {str(st)[:30]:32}{n:>5}")

    accs = m.loc[generic, "BioSample"].astype(str).str.strip()
    cache = load_cache()
    need = sorted(set(accs) - set(cache))
    print(f"\n  cache holds {len(cache):,}; {len(need):,} aliases to fetch")
    if need:
        got = fetch_aliases(need)
        cache.update(got)
        print(f"    fetched {len(got):,}")
        if not a.dry_run:
            save_cache(cache)

    alias = accs.map(cache)
    # Refuse an alias that is itself boilerplate, or that repeats across
    # samples: that would swap one useless value for another.
    ok = alias.notna() & alias.str.strip().ne("")
    counts = alias[ok].value_counts()
    unique_alias = alias.map(lambda v: counts.get(v, 0) == 1 if pd.notna(v) else False)
    fill = generic & ok & unique_alias
    print(f"\n  aliases found        : {int(ok.sum()):,}")
    print(f"  usable (unique)      : {int(fill.sum()):,}")
    print(f"  not replaced         : {int((generic & ~fill).sum()):,}")

    if int(fill.sum()):
        print("\n  examples:")
        for i in list(m.index[fill])[:6]:
            print(f"    {m.at[i, 'StudyFolder']:24} "
                  f"{str(sid.at[i])[:38]:40} -> {alias.at[i]}")

    if a.dry_run:
        print("\n  dry run; nothing written")
        return

    rows = [dict(Chosen_match_ID=m.at[i, "Chosen_match_ID"],
                 StudyFolder=m.at[i, "StudyFolder"],
                 BioSample=m.at[i, "BioSample"],
                 old=sid.at[i], new=alias.at[i],
                 reason="Sample ID held ENA sample_title (boilerplate); "
                        "replaced with sample_alias")
            for i in m.index[fill]]
    m.loc[fill, "Sample ID"] = alias[fill]
    m.to_csv(a.io, index=False)
    pd.DataFrame(rows).to_csv(CHANGELOG, index=False)
    print(f"\n  wrote {Path(a.io).relative_to(ROOT)}")
    print(f"  wrote {CHANGELOG.relative_to(ROOT)}  ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
