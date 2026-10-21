#!/usr/bin/env python3
"""Close the ENA blind spot using NCBI BioSample `geo_loc_name`.

`verify_country_against_ena.py` leaves ~5,000 samples unverifiable because ENA's
run-level `country` field is empty for them -- notably PRJNA999635 (1,887),
PRJEB11532 (895) and PRJNA48479 (435). That absence is an artefact of where the
data was submitted, not of the sample: the same samples carry a populated
`geo_loc_name` in NCBI BioSample. PRJNA999635's records say `Honduras:Copan`.

So this walks the second source. Run -> sample accession comes from the ENA TSVs
already on disk; sample -> country comes from NCBI efetch, batched.

`geo_loc_name` is `Country:Region` by INSDC convention, which is better behaved
than ENA's free-text `country`, but the same guard applies: anything that does not
resolve to a country the pipeline recognises is unverifiable, not a disagreement.

Read-only with respect to the metadata; disagreements are written to a CSV.

    python tools/verify_country_via_biosample.py --ena-dir <dir> --cache <dir>
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence import config  # noqa: E402
from persistence.common import normalise_country  # noqa: E402
sys.path.insert(0, str(ROOT / 'tools'))
from geo_resolve import resolve as _resolve  # noqa: E402

KNOWN = set(config.COUNTRY_TO_CONTINENT)
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH = 300
# NCBI allows 3 requests/second unauthenticated; stay well under it.
PAUSE = 0.5


def fetch_batch(accessions: list[str]) -> dict[str, str]:
    # POST, not GET: 300 accessions in a query string is ~3.5 kB and NCBI answers
    # 414 Request-URI Too Long. efetch accepts the same parameters as a form body.
    data = urllib.parse.urlencode(
        {"db": "biosample", "id": ",".join(accessions), "rettype": "xml"}
    ).encode()
    req = urllib.request.Request(EFETCH, data=data,
                                 headers={"Content-Type":
                                          "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=180) as fh:
        blob = fh.read()
    out = {}
    try:
        root = ET.fromstring(blob)
    except ET.ParseError:
        return out
    for sample in root.iter("BioSample"):
        acc = sample.get("accession")
        geo = None
        for attr in sample.iter("Attribute"):
            if attr.get("harmonized_name") == "geo_loc_name" or \
               attr.get("attribute_name") in ("geo_loc_name", "geographic location"):
                geo = (attr.text or "").strip()
                break
        if acc and geo:
            out[acc] = geo
    return out



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=ROOT / "metadata_Mar_16_2026.csv")
    ap.add_argument("--ena-dir", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True,
                    help="Directory for the biosample->geo cache (resumable).")
    ap.add_argument("--out", type=Path, default=Path("biosample_country_check.csv"))
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    cache_file = args.cache / "geo_loc_name.tsv"

    # run/sample accessions and ENA's own country, from the already-fetched TSVs
    frames = []
    for path in sorted(args.ena_dir.glob("PRJ*.tsv")):
        try:
            df = pd.read_csv(path, sep="\t", dtype=str)
        except Exception:
            continue
        if {"run_accession", "sample_accession"} <= set(df.columns):
            frames.append(df)
    ena = pd.concat(frames, ignore_index=True)
    ena["ena_ok"] = ena.get("country", pd.Series(dtype=str)).map(
        lambda v: bool(_resolve(v)) if isinstance(v, str) else False)

    meta = pd.read_csv(args.metadata, dtype=str, low_memory=False)
    id_cols = [c for c in ("SRA_Accession", "Chosen_match_ID") if c in meta.columns]
    run2samp = dict(zip(ena["run_accession"], ena["sample_accession"]))
    run_ok = dict(zip(ena["run_accession"], ena["ena_ok"]))

    def sample_for(row):
        for col in id_cols:
            for key in str(row.get(col) or "").split(";"):
                key = key.strip()
                if key in run2samp and not run_ok.get(key, False):
                    return run2samp[key]
        # Runs absent from the ENA mirror still have a BioSample; fall back to the
        # accession the metadata itself records, when it looks like one.
        bs = str(row.get("BioSample") or "").strip()
        return bs if re.match(r"^SAM[ND]?[EA]?\d+$", bs) else None

    meta["_samp"] = meta.apply(sample_for, axis=1)
    wanted = sorted({s for s in meta["_samp"].dropna() if s})
    print(f"  samples needing a second source: {len(wanted):,}")

    cached = {}
    if cache_file.exists():
        for line in cache_file.read_text().splitlines():
            if "\t" in line:
                a, g = line.split("\t", 1)
                cached[a] = g
    todo = [a for a in wanted if a not in cached]
    print(f"  cached {len(cached):,}; fetching {len(todo):,} in batches of {BATCH}")

    with cache_file.open("a") as fh:
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            try:
                got = fetch_batch(chunk)
            except Exception as exc:
                print(f"    batch {i // BATCH}: {exc}")
                time.sleep(2)
                continue
            for acc, geo in got.items():
                cached[acc] = geo
                fh.write(f"{acc}\t{geo}\n")
            fh.flush()
            print(f"    {min(i + BATCH, len(todo)):>6}/{len(todo)}  "
                  f"(+{len(got)})", flush=True)
            time.sleep(PAUSE)

    meta["_geo"] = meta["_samp"].map(cached)
    meta["_truth"] = meta["_geo"].map(lambda v: _resolve(v) if isinstance(v, str) else None)
    meta["_ours"] = meta["Country"].map(
        lambda c: normalise_country(c) if pd.notna(c) else None)

    resolved = meta["_truth"].notna()
    disagree = resolved & (meta["_truth"] != meta["_ours"])
    print(f"\n  newly resolved via BioSample : {resolved.sum():,}")
    print(f"  DISAGREEMENTS                : {disagree.sum():,}")

    if disagree.any():
        grp = (meta[disagree].groupby(["BioProject ID", "_ours", "_truth"])
               .size().reset_index(name="n").sort_values("n", ascending=False))
        print(f"\n  {'BioProject':<14}{'ours':<26}{'BioSample':<26}{'n':>6}")
        for _, r in grp.iterrows():
            print(f"  {str(r['BioProject ID'])[:13]:<14}{str(r['_ours'])[:25]:<26}"
                  f"{str(r['_truth'])[:25]:<26}{r['n']:>6}")
        cols = [c for c in ("Chosen_match_ID", "BioProject ID", "Author",
                            "StudyFolder", "Country") if c in meta.columns]
        out = meta.loc[disagree, cols].copy()
        out["biosample"] = meta.loc[disagree, "_samp"]
        out["geo_loc_name"] = meta.loc[disagree, "_geo"]
        out.to_csv(args.out, index=False)
        print(f"\n  wrote {args.out}")

    still = meta[meta["_samp"].notna() & meta["_truth"].isna()]
    if len(still):
        print(f"\n  still unverifiable after both sources: {len(still):,}")
        for proj, n in still.groupby("BioProject ID").size().sort_values(
                ascending=False).head(8).items():
            print(f"    {proj:<16}{n:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
