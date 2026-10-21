#!/usr/bin/env python3
"""Re-derive the analytic cohort from public repositories and the inclusion criteria.

The point is falsifiability. Table M1 is a curated artefact: correct as far as we
can check, but a reviewer cannot regenerate it. This walks the other direction --
start from the study list and the stated criteria, ask ENA what runs exist, apply
the filters, and compare against M1.

Criteria, and how each maps onto ENA:

    paired-end          library_layout == PAIRED
    shotgun WGS         library_strategy == WGS
    gut microbiome      a property of the STUDY, not the run -- see below
    non-infant          host age, where recorded
    sufficient depth    summed base_count over qualifying runs >= MIN_BASES
    usable lifestyle    (study, country) resolves to one of H-G / Transitional /
                        Industrial

Two design decisions, both learned from getting them wrong first:

Filters apply per SAMPLE, not per run. A sample's other runs are frequently
single-end or amplicon while a paired WGS run also exists; filtering run-by-run
discarded 6,681 single-end and 1,483 amplicon runs belonging to samples that
legitimately qualify. A sample is kept if it has at least one qualifying run, and
its depth is the sum over those runs.

Gut/stool is not tested per run. Every study in the list is a gut metagenome
study, and 9,285 of 9,286 runs that failed a `host_body_site`/`isolation_source`
regex failed it because both fields are empty, not because the sample is not gut.
Testing an unpopulated field discards real data; the study list already carries
that information.

Lifestyle is assigned from (study, country), which resolves 111 of 115 pairs in
M1. Countries are lifestyle-pure in 35 of 43 cases; the study disambiguates the
rest, except four cohorts -- Lokmer, Obregon-Tito, Rubel, Tamburini -- that
deliberately sampled two lifestyles in one country and need per-sample sources.

Two traps, both of which produced confident false findings before being caught:

`library_layout` and `library_strategy` are not reliable. 100 of 101 runs that ENA
labels SINGLE report `nreads="2"` in SRA's own statistics -- they are paired data
whose layout ENA could not infer, because the submitter uploaded processed FASTA
(IjazUZ_2017: `S101_a_WGS.fasta.gz`) and the derived FASTQ is empty. Treating the
label as truth would exclude 101 valid samples. Where a sample would fail on layout
alone, the SRA read structure is the authority.

Resolving a sample's runs through its BioSample pulls in companion runs from the
same subject. In PRJNA324129 each subject has both a 16S run and a WGS run under
one BioSample, so a BioSample-first lookup classified 8 WGS samples as amplicon --
their `Chosen_match_ID` accessions are WGS in SRA with `total_bases` matching M1 to
the byte. Identifiers are therefore resolved run-accession first, and only fall
back to sample-level accessions when no run matches.

    python tools/rebuild_cohort_from_ena.py --cache <dir>
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

SEARCH = "https://www.ebi.ac.uk/ena/portal/api/search"

FIELDS = [
    "run_accession", "sample_accession", "secondary_sample_accession",
    "study_accession", "library_layout", "library_strategy", "library_source",
    "base_count", "read_count", "instrument_platform", "host", "host_body_site",
    "isolation_source", "country", "collection_date", "sample_title",
]

MIN_BASES = 100_000_000       # from `metadata_non_infants_gt100Mb_...csv`
LIFESTYLES = {"H-G", "Transitional", "Industrial"}

# Body site is tested three ways, not two. Requiring a positive gut match discards
# 9,285 samples whose fields are simply empty; ignoring the field entirely admits
# HMP's tongue, plaque and vaginal samples, which are paired WGS over 100 Mb and
# would otherwise sail through. So: exclude on an explicit non-gut site, include on
# an explicit gut site, and fall back to the study-level assumption when silent.
import re as _re

NON_GUT = _re.compile(
    r"tongue|plaque|saliva|oral|buccal|throat|pharyn|nasal|nostril|nares|"
    r"skin|elbow|retroauricular|vagin|fornix|cervi|urine|urethra|blood|serum|"
    r"sputum|lung|airway|ear\b|eye|conjunctiv|breast|milk|semen|wastewater|"
    r"soil|water|sediment|dough", _re.I)
GUT = _re.compile(r"gut|stool|fa?ec|intestin|colon|rect|sigmoid|ileum|cecum", _re.I)


def body_site_verdict(*values) -> str:
    """'gut', 'other', or 'unknown' from whatever site fields are populated."""
    text = " ".join(v for v in values if isinstance(v, str))
    if not text.strip():
        return "unknown"
    if GUT.search(text):
        return "gut"
    if NON_GUT.search(text):
        return "other"
    return "unknown"


def fetch_project(accession: str, cache: Path) -> pd.DataFrame:
    out = cache / f"{accession}.tsv"
    if out.exists() and out.stat().st_size > 0:
        try:
            return pd.read_csv(out, sep="\t", dtype=str)
        except Exception:
            pass
    data = urllib.parse.urlencode({
        "result": "read_run", "includeAccessions": accession,
        "fields": ",".join(FIELDS), "format": "tsv", "limit": "0",
    }).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(SEARCH, data=data),
                                    timeout=180) as fh:
            text = fh.read().decode("utf8", "replace")
    except Exception as exc:
        print(f"    {accession}: {exc}")
        return pd.DataFrame()
    if not text.strip() or "run_accession" not in text.split("\n")[0]:
        return pd.DataFrame()
    out.write_text(text)
    return pd.read_csv(out, sep="\t", dtype=str)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=ROOT / "metadata_Mar_16_2026.csv")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=Path("cohort_rebuild.csv"))
    ap.add_argument("--verified-country", type=Path, default=None,
                    help="CSV of externally verified per-sample countries "
                         "(Chosen_match_ID,truth) from the verification tools. "
                         "ENA leaves `country` empty for whole studies, which "
                         "strands multi-lifestyle cohorts like Valles-Colomer and "
                         "Tamburini; those countries were recovered from published "
                         "supplementary tables and belong back in the derivation.")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata, dtype=str, low_memory=False)
    # M1 stores display labels; the lifestyle maps below key on the
    # internal vocabulary, so normalise on the way in.
    from persistence import config as _c
    if "Lifestyle" in meta.columns:
        meta["Lifestyle"] = _c.canonicalize_lifestyle_series(meta["Lifestyle"])
    projects = sorted({p for p in meta["BioProject ID"].dropna()
                       if str(p).startswith(("PRJ", "SRP", "ERP", "DRP"))})
    print(f"  M1: {len(meta):,} samples across {len(projects)} study accessions")

    frames = []
    for proj in projects:
        df = fetch_project(proj, args.cache)
        if not df.empty:
            df["query_accession"] = proj
            frames.append(df)
        time.sleep(0.05)
    runs = pd.concat(frames, ignore_index=True)
    print(f"  ENA: {len(runs):,} runs over {runs['query_accession'].nunique()} studies\n")

    # ---------- per-run qualification ----------
    runs["_bases"] = pd.to_numeric(runs["base_count"], errors="coerce").fillna(0)
    runs["_paired"] = runs["library_layout"].fillna("").str.upper() == "PAIRED"
    runs["_wgs"] = runs["library_strategy"].fillna("").str.upper() == "WGS"
    runs["_ok"] = runs["_paired"] & runs["_wgs"]

    # ---------- collapse to samples ----------
    runs["_sample"] = runs["sample_accession"].fillna(
        runs["secondary_sample_accession"]).fillna(runs["run_accession"])
    grp = runs.groupby("_sample")
    samples = pd.DataFrame({
        "study": grp["query_accession"].first(),
        "n_runs": grp.size(),
        "n_qualifying_runs": grp["_ok"].sum(),
        "bases_qualifying": grp.apply(lambda g: g.loc[g["_ok"], "_bases"].sum()),
        "country_raw": grp["country"].first(),
        "secondary": grp["secondary_sample_accession"].first(),
        "run_accessions": grp["run_accession"].apply(lambda s: ";".join(s.dropna())),
        "site": grp.apply(lambda g: body_site_verdict(
            *g["host_body_site"].tolist(), *g["isolation_source"].tolist(),
            *g["sample_title"].tolist())),
    }).reset_index()
    print(f"  collapsed to {len(samples):,} samples")

    steps = [("samples in the studies", len(samples))]
    keep = samples[samples.site != "other"]
    steps.append(("not an explicit non-gut site", len(keep)))
    keep = keep[keep.n_qualifying_runs > 0]
    steps.append(("has a paired-end WGS run", len(keep)))
    keep = keep[keep.bases_qualifying >= MIN_BASES]
    steps.append((f"summed bases >= {MIN_BASES/1e6:.0f} Mb", len(keep)))

    # ---------- lifestyle assignability ----------
    from geo_resolve import resolve
    from persistence.common import normalise_country
    m1_life = {}
    for (sf, c), g in meta.groupby(["StudyFolder", "Country"]):
        ls = {x for x in g["Lifestyle"].dropna().unique() if x in LIFESTYLES}
        if len(ls) == 1:
            m1_life[(sf, c)] = ls.pop()
    country_life = {}
    for c, g in meta.groupby("Country"):
        ls = {x for x in g["Lifestyle"].dropna().unique() if x in LIFESTYLES}
        if len(ls) == 1:
            country_life[c] = ls.pop()
    # study accession -> the StudyFolders it contains
    proj_folders = (meta.groupby("BioProject ID")["StudyFolder"]
                    .apply(lambda s: sorted(set(s.dropna()))).to_dict())

    ISO = {v: k for k, v in
           ((c, normalise_country(c)) for c in meta["Country"].dropna().unique())}

    # externally verified per-sample country, keyed by every identifier we know
    verified = {}
    if args.verified_country and args.verified_country.exists():
        vc = pd.read_csv(args.verified_country, dtype=str, low_memory=False)
        col = "truth" if "truth" in vc.columns else vc.columns[-1]
        key = "Chosen_match_ID" if "Chosen_match_ID" in vc.columns else vc.columns[0]
        m1_key = meta.set_index("Chosen_match_ID") if "Chosen_match_ID" in meta else None
        for k, v in zip(vc[key], vc[col]):
            if isinstance(v, str) and v.strip():
                verified[str(k).strip()] = v.strip()
                if m1_key is not None and k in m1_key.index:
                    row = m1_key.loc[k]
                    row = row.iloc[0] if hasattr(row, "iloc") and row.ndim > 1 else row
                    for c in ("SRA_Accession", "SRS ID", "BioSample"):
                        for kk in str(row.get(c) or "").split(";"):
                            if kk.strip():
                                verified[kk.strip()] = v.strip()
        print(f"  verified countries loaded: {len(verified):,} identifiers")

    def lifestyle(row):
        name = resolve(row.country_raw)
        if not name:
            for k in [row._sample, row.secondary] + str(row.run_accessions).split(";"):
                if isinstance(k, str) and k.strip() in verified:
                    name = verified[k.strip()]
                    break
        iso = ISO.get(name)
        if iso:
            for sf in proj_folders.get(row.study, []):
                if (sf, iso) in m1_life:
                    return m1_life[(sf, iso)]
            if iso in country_life:
                return country_life[iso]
        # no usable country: fall back to the study, if it is lifestyle-pure
        folders = proj_folders.get(row.study, [])
        ls = {m1_life[k] for k in m1_life if k[0] in folders}
        return ls.pop() if len(ls) == 1 else None

    keep = keep.copy()
    keep["lifestyle"] = keep.apply(lifestyle, axis=1)
    keep_l = keep[keep.lifestyle.notna()]
    steps.append(("lifestyle assignable", len(keep_l)))

    print(f"\n  {'criterion':<30}{'samples':>10}{'dropped':>10}")
    prev = None
    for label, n in steps:
        print(f"  {label:<30}{n:>10,}{'' if prev is None else f'{prev-n:,}':>10}")
        prev = n

    # ---------- reconcile ----------
    ids = set()
    for col in ("Chosen_match_ID", "SRA_Accession", "SRS ID", "BioSample"):
        if col in meta.columns:
            for v in meta[col].dropna():
                ids.update(x.strip() for x in str(v).split(";") if x.strip())

    # Run accessions first: a BioSample can cover several runs of one subject,
    # including a 16S companion to the WGS run we actually want.
    run_ids = set()
    for col in ("Chosen_match_ID", "SRA_Accession"):
        if col in meta.columns:
            for v in meta[col].dropna():
                run_ids.update(x.strip() for x in str(v).split(";") if x.strip())

    def in_m1(row):
        if any(r in run_ids for r in str(row.run_accessions).split(";")):
            return True
        return row._sample in ids or (isinstance(row.secondary, str)
                                      and row.secondary in ids)

    keep_l = keep_l.copy()
    keep_l["in_m1"] = keep_l.apply(in_m1, axis=1)
    print(f"\n  === reconciliation ===")
    print(f"    rebuilt cohort              : {len(keep_l):,} samples")
    print(f"      also in M1                : {int(keep_l.in_m1.sum()):,}")
    print(f"      NOT in M1 (extras)        : {int((~keep_l.in_m1).sum()):,}")
    print(f"    M1 samples                  : {len(meta):,}")
    print(f"      reached by the rebuild    : {int(keep_l.in_m1.sum()):,} "
          f"({100*keep_l.in_m1.sum()/len(meta):.1f}%)")

    extra = keep_l[~keep_l.in_m1]
    if len(extra):
        print(f"\n    extras by study:")
        for p, n in extra.groupby("study").size().sort_values(
                ascending=False).head(12).items():
            folders = ",".join(proj_folders.get(p, []))[:34]
            print(f"      {p:<14}{n:>6}  {folders}")
    keep_l.to_csv(args.report, index=False)
    print(f"\n  wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
