#!/usr/bin/env python3
"""Check every sample's declared country against ENA's own record.

`PRJNA389927` (Hannigan CRC virome) turned out to be labelled Canadian for all 79
of its samples in our metadata, while ENA places 54 of them in Ann Arbor, Boston or
Houston. That kind of error -- a multi-centre cohort collapsed to the country of
one site -- is invisible to any internal consistency check, because the metadata
agrees with itself. Only an external source catches it.

So this compares against ENA rather than against ourselves. Both sides are pushed
through `normalise_country` and compared as full country names, which avoids
maintaining a second ISO3 table and makes `UK`/`GBR`/`United Kingdom` compare equal.

ENA's `country` field is free text of the form `USA: Ann Arbor`, `Canada:Toronto`
or bare `USA`; only the part before the colon is used. Runs with an empty country
are reported as unverifiable rather than as disagreements -- ENA not knowing is not
evidence that we are wrong.

Reads the per-project TSVs fetched into `--ena-dir` (see the module docstring of the
fetch step) and never writes to the metadata; corrections are emitted as a CSV for
review.

    python tools/verify_country_against_ena.py --metadata snapshot.csv --ena-dir ena/
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence import config  # noqa: E402
from persistence.common import normalise_country  # noqa: E402
sys.path.insert(0, str(ROOT / 'tools'))
from geo_resolve import resolve as _resolve  # noqa: E402

# The 44 countries the pipeline recognises; anything else in ENA's free-text
# country field is a city, a region, or a placeholder.
KNOWN_COUNTRIES = set(config.COUNTRY_TO_CONTINENT)


def load_ena(ena_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(ena_dir.glob("PRJ*.tsv")):
        try:
            df = pd.read_csv(path, sep="\t", dtype=str)
        except Exception:
            continue
        if "run_accession" not in df.columns:
            continue
        df["project"] = path.stem
        frames.append(df)
    if not frames:
        sys.exit(f"no ENA TSVs under {ena_dir}")
    return pd.concat(frames, ignore_index=True)



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=ROOT / "metadata_Mar_16_2026.csv")
    ap.add_argument("--ena-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("country_corrections.csv"))
    args = ap.parse_args()

    ena = load_ena(args.ena_dir)
    # `normalise_country`, which this module already imports; `ena_country`
    # was never defined anywhere and made the script a guaranteed NameError.
    ena["truth"] = (ena["country"].map(normalise_country)
                    if "country" in ena.columns else None)
    by_run = dict(zip(ena["run_accession"], ena["truth"]))
    by_sample = (dict(zip(ena["sample_accession"], ena["truth"]))
                 if "sample_accession" in ena.columns else {})
    print(f"  ENA: {len(ena):,} runs over {ena['project'].nunique()} projects; "
          f"{ena['truth'].notna().sum():,} carry a country")

    meta = pd.read_csv(args.metadata, dtype=str, low_memory=False)
    id_cols = [c for c in ("SRA_Accession", "Chosen_match_ID", "BioSample")
               if c in meta.columns]

    def resolve(row):
        for col in id_cols:
            key = row.get(col)
            if key in by_run and by_run[key]:
                return by_run[key]
            if key in by_sample and by_sample[key]:
                return by_sample[key]
        return None

    meta["_truth"] = meta.apply(resolve, axis=1)
    meta["_ours"] = meta["Country"].map(lambda c: normalise_country(c) if pd.notna(c) else None)

    matched = meta["_truth"].notna()
    disagree = matched & (meta["_truth"] != meta["_ours"])
    print(f"  metadata: {len(meta):,} rows; {matched.sum():,} resolved against ENA; "
          f"{(~matched).sum():,} unverifiable")
    print(f"  DISAGREEMENTS: {disagree.sum():,}\n")

    if disagree.any():
        grp = (meta[disagree].groupby(["BioProject ID", "_ours", "_truth"])
               .size().reset_index(name="n").sort_values("n", ascending=False))
        print(f"  {'BioProject':<14}{'ours':<28}{'ENA':<28}{'n':>6}")
        for _, r in grp.iterrows():
            print(f"  {str(r['BioProject ID'])[:13]:<14}{str(r['_ours'])[:27]:<28}"
                  f"{str(r['_truth'])[:27]:<28}{r['n']:>6}")

        cols = [c for c in ("Chosen_match_ID", "SRA_Accession", "BioProject ID",
                            "Author", "StudyFolder", "Country", "Country (full)",
                            "Population") if c in meta.columns]
        out = meta.loc[disagree, cols].copy()
        out["ena_country"] = meta.loc[disagree, "_truth"]
        out.to_csv(args.out, index=False)
        print(f"\n  wrote {args.out} ({len(out)} rows)")

    # A study where ENA knows the country for nobody is worth naming: it is the
    # blind spot where a Hannigan-style error could still be hiding.
    blind = (meta[~matched].groupby("BioProject ID").size()
             .sort_values(ascending=False))
    blind = blind[blind > 0]
    if len(blind):
        print(f"\n  projects with unverifiable samples (top 10):")
        for proj, n in blind.head(10).items():
            print(f"    {proj:<16}{n:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
