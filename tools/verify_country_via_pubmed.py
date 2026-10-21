#!/usr/bin/env python3
"""Third-tier country check: does the source paper corroborate the cohort country?

ENA and BioSample together verify 81% of samples. The remainder are projects whose
submitters left the geography fields empty. For those the paper itself is the only
external record left.

This tier is deliberately weaker than the other two, and the difference matters:

  * ENA/BioSample give a country *per sample*. A paper gives one *per cohort*.
  * That is sufficient for a single-country study and insufficient for a
    multi-site one -- precisely the Hannigan failure mode, where a genuinely
    mixed USA/Canada cohort was labelled Canadian throughout and every
    internal check passed.

So only projects whose metadata declares a single country are checked here, and a
hit is recorded as `corroborated`, never as `verified`. Multi-country projects are
listed as out of scope rather than silently waved through.

Matching is on the country name, its adjectival form, and its major cities in the
title and abstract. Absence is reported as `no mention`, not as a contradiction: a
paper is free never to name its country in the abstract.

    python tools/verify_country_via_pubmed.py --projects unverified_projects.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PAUSE = 0.4

# country -> extra strings that corroborate it (adjectival form, major cities,
# institutions strongly tied to the cohort)
HINTS = {
    "Israel": ["Israeli", "Weizmann", "Tel Aviv", "Jerusalem"],
    "United States of America": ["United States", "U.S.", "US ", "American",
                                 "Houston", "Boston", "Ann Arbor", "Baltimore",
                                 "St. Louis", "Missouri", "Texas"],
    "China": ["Chinese", "Shanghai", "Beijing", "Shenzhen", "Hong Kong", "Shaanxi"],
    "Fiji": ["Fijian"],
    "Sweden": ["Swedish", "Gothenburg", "Stockholm"],
    "Finland": ["Finnish", "Helsinki", "Turku"],
    "Italy": ["Italian", "Rome", "Milan", "Bologna"],
    "United Kingdom": ["British", "England", "Scotland", "London", "Glasgow", "UK"],
    "Denmark": ["Danish", "Copenhagen"],
    "Spain": ["Spanish", "Madrid", "Barcelona"],
    "Germany": ["German", "Heidelberg", "Berlin"],
    "Japan": ["Japanese", "Tokyo", "Osaka"],
    "Netherlands": ["Dutch", "Groningen", "Amsterdam"],
    "Canada": ["Canadian", "Toronto", "Montreal"],
    "Austria": ["Austrian", "Vienna"],
    "France": ["French", "Paris"],
    "Peru": ["Peruvian", "Lima"],
    "Tanzania": ["Tanzanian", "Hadza"],
    "Russia": ["Russian", "Moscow"],
    "South Korea": ["Korean", "Seoul"],
    "Kazakhstan": ["Kazakh"],
    "Mongolia": ["Mongolian"],
    "Bangladesh": ["Bangladeshi", "Dhaka"],
    "Madagascar": ["Malagasy"],
    "Cameroon": ["Cameroonian"],
    "Honduras": ["Honduran", "Copan"],
}


def fetch_abstract(pmid: str) -> str:
    params = {"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "text"}
    url = f"{EFETCH}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as fh:
        return fh.read().decode("utf8", "replace")


def strip_affiliations(text: str) -> str:
    """Drop the `Author information:` block.

    Affiliations name the authors' institutions, not the cohort, and matching
    against them is worse than useless: PRJEB1690 "corroborated" South Korea
    purely on `Seoul National University` in the author list, which would have
    been recorded as evidence about the samples. The block starts at
    `Author information:` and runs while entries begin `(1)`, `(2)`, ...
    """
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        if lines[i].startswith("Author information:"):
            i += 1
            # Entries are numbered `(1)`, `(2)`, ... and wrap over several lines,
            # so a continuation cannot be recognised by its own content -- an
            # earlier attempt tested for a leading lowercase letter and let
            # `Hospital and Harvard Medical School, Boston, MA, USA.` through.
            # Consume until a blank line whose next non-blank line is not a new
            # numbered entry: that is where the abstract starts.
            while i < len(lines):
                if not lines[i].strip():
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j >= len(lines) or not re.match(r"^\(\d+\)", lines[j]):
                        break
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def corroborates(text: str, country: str) -> list[str]:
    found = []
    haystack = strip_affiliations(text).lower()
    for needle in [country] + HINTS.get(country, []):
        if needle.strip().lower() in haystack:
            found.append(needle.strip())
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", type=Path, required=True,
                    help="CSV from the coverage step: project,n_unverified,countries,single,pmid")
    ap.add_argument("--metadata", type=Path, default=ROOT / "metadata_Mar_16_2026.csv")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("pubmed_corroboration.csv"))
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    from persistence.common import normalise_country

    proj = pd.read_csv(args.projects, dtype=str)
    proj["n_unverified"] = proj["n_unverified"].astype(int)
    single = proj[proj["single"].astype(str).str.lower() == "true"]
    multi = proj[proj["single"].astype(str).str.lower() != "true"]

    print(f"  single-country projects to corroborate: {len(single)} "
          f"({single.n_unverified.sum():,} samples)")
    print(f"  multi-country, out of scope for this tier: {len(multi)} "
          f"({multi.n_unverified.sum():,} samples)\n")

    rows = []
    for _, r in single.sort_values("n_unverified", ascending=False).iterrows():
        pmids = re.findall(r"\d{6,9}", str(r["pmid"]))
        country = normalise_country(str(r["countries"]).split(",")[0])
        if not pmids:
            rows.append({**r.to_dict(), "pmid_used": "", "verdict": "no PMID",
                         "matched": ""})
            continue
        pmid = pmids[0]
        cache = args.cache / f"{pmid}.txt"
        if cache.exists():
            text = cache.read_text()
        else:
            try:
                text = fetch_abstract(pmid)
            except Exception as exc:
                rows.append({**r.to_dict(), "pmid_used": pmid,
                             "verdict": f"fetch failed: {exc}", "matched": ""})
                continue
            cache.write_text(text)
            time.sleep(PAUSE)
        hits = corroborates(text, country)
        rows.append({**r.to_dict(), "pmid_used": pmid,
                     "verdict": "corroborated" if hits else "no mention",
                     "matched": "; ".join(hits[:4]),
                     "title": text.strip().split("\n\n")[1][:90] if "\n\n" in text else ""})

    out = pd.DataFrame(rows)
    ok = out[out.verdict == "corroborated"]
    print(f"  {'project':<14}{'author':<18}{'n':>6}  {'country':<24}{'verdict':<14}matched")
    for _, r in out.iterrows():
        print(f"  {str(r['project'])[:13]:<14}{str(r.get('author',''))[:17]:<18}"
              f"{int(r['n_unverified']):>6}  {str(r['countries'])[:23]:<24}"
              f"{r['verdict']:<14}{str(r['matched'])[:40]}")
    print(f"\n  corroborated: {len(ok)}/{len(out)} projects, "
          f"{ok.n_unverified.sum():,} of {out.n_unverified.sum():,} samples")
    out.to_csv(args.out, index=False)
    print(f"  wrote {args.out}")

    if len(multi):
        print(f"\n  NOT corroborated by this tier -- multi-country, needs a "
              f"per-sample source:")
        for _, r in multi.iterrows():
            print(f"    {str(r['project']):<14}{int(r['n_unverified']):>6}  "
                  f"{r['countries']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
