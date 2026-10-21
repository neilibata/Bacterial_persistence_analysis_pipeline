#!/usr/bin/env python3
"""Stress-test the candidate samples and emit a reprocessing worklist.

`build_candidate_samples.py` answers "which samples could be processed". This
answers "which should be", by applying the tests that only matter once compute is
about to be spent, and by writing the result in the format the preprocessing
Snakefile consumes (`sample_id <TAB> run1;run2`).

The extra tests, each added because the cheaper checks let something through:

  Cultured isolates, properly. The first pass matched Fiji's `MIGS Cultured...`
  title and missed Tanzania's `This sample represents a Bifidobacterium infantis
  genome isolated and sequenced from ERS2586086`, because `description` was not
  among the cached ENA fields. Depth is the corroborating signal: a single-genome
  library at 21 Gbp against a cohort median of 4.4 is not a community.

  Non-human hosts. Brito's Fiji project deposits 17 livestock faecal samples
  beside 224 human ones, and every structured ENA field on them is identical to
  the human samples -- 'gut metagenome', 'stool', host 'missing'. Only the alias
  suffix betrays them. Caught here because the consequence is severe and silent:
  they align at ~19% against a human gut catalogue and contribute ~4 detected
  SGBs each to the non-industrial arm.

  Longitudinal repeats. HMP2, Mehta and iMSMS follow subjects over time, so a
  naive count treats eight visits by one person as eight independent samples.
  Where `host_subject_id` is present, repeats beyond the first are flagged --
  they inflate n without adding an individual.

  Infants. The cohort is adult; `host_age` is recorded rarely but where present
  a sample below NON_INFANT_YEARS is flagged rather than silently included.

  Depth realism. The 100 Mb floor comes from the original file name and is far
  below what this cohort actually carries (Q1 = 2.43 Gbp). Samples are banded
  against the cohort's own distribution so the marginal ones are visible.

Nothing is dropped. Every candidate is written with its flags, and the worklist
is a filtered view, so a reader can see what was excluded and why.

None of this replaces the post-alignment check. The screens here read metadata,
and the Fiji livestock samples proved metadata can be uniformly correct and
still not say what you need; the alignment rate against the catalogue is the
only signal that separated them. Run `alignment_qc.py` in the preprocessing repo
on whatever this worklist produces before merging it into M1.

    python tools/build_reprocessing_worklist.py --enriched <tsv> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ISOLATE = re.compile(
    r"isolated and sequenced from|represents a[n]? .{0,60}genome|"
    r"cultured bacterial|cultured archaeal|MIGS Cultured|single[- ]colony|"
    r"genome (?:of|from) .{0,40}isolated|strain .{0,20}genome|"
    r"assembl(?:y|ed) (?:of|from)", re.I)

# Non-human hosts. The cohort is human gut, but a study that samples people and
# their livestock together deposits both under the same project with the same
# structured fields. In PRJNA217052 (Brito Fiji) the 17 animal faecal samples
# carry scientific_name='gut metagenome', isolation_source='stool' and
# host='missing' -- byte-identical to the 224 human stool samples. No structured
# field separates them; only the alias does, where humans get subject IDs
# (HUGE-05-12.ST) and animals get a village.species code (NV.HB.AST, M.P1.AST,
# NK.C2.AST -- cow, pig, horse, replicated per village).
#
# They are not a curation nicety. Aligned against a human gut catalogue they
# return ~19% where the cohort returns ~88%, and 4 detected SGBs where their
# human neighbours return 103, so including one contributes a near-empty column
# to a non-industrial cohort whose richness carries the main result.
NONHUMAN = re.compile(
    r"\.AST\b|"                                          # village.species.AST
    r"\b(?:animal|livestock|bovine|porcine|canine|cattle|cow|pig|swine|"
    r"poultry|chicken|goat|sheep|horse|dog)\b[^.]{0,30}"
    r"\b(?:stool|fa?ec(?:es|al)|gut|dung|manure|droppings)\b", re.I)

NON_INFANT_YEARS = 3.0
ISOLATE_DEPTH_GBP = 15.0     # corroborating signal, not a criterion on its own


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--enriched", type=Path, required=True)
    ap.add_argument("--sample-desc", type=Path, default=None,
                    help="sample-level descriptions for accessions the run "
                         "endpoint cannot resolve. The isolate wording lives on "
                         "the SAMPLE record, not the run: ENA's read_run returns "
                         "nothing at all for the Hadza Bifidobacterium entries, "
                         "which is itself a signal that they are not read data.")
    ap.add_argument("--metadata", type=Path, default=ROOT / "metadata_Mar_16_2026.csv")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cand = pd.read_csv(args.candidates, low_memory=False)
    clean = cand[cand.n_fail == 0].copy()
    enr = pd.read_csv(args.enriched, sep="\t", dtype=str, low_memory=False)
    enr["_bases"] = pd.to_numeric(enr.get("base_count"), errors="coerce").fillna(0)

    meta = pd.read_csv(args.metadata, dtype=str, low_memory=False)
    cohort = pd.to_numeric(meta["Total bases"], errors="coerce")
    q1, med = cohort.quantile(.25), cohort.median()

    # per-sample rollup of the enriched run table
    g = enr.groupby("sample_accession")
    text = g.apply(lambda x: " ".join(
        str(v) for v in list(x.get("description", pd.Series(dtype=str)).dropna())
        + list(x.get("sample_title", pd.Series(dtype=str)).dropna())
        + list(x.get("sample_alias", pd.Series(dtype=str)).dropna())))
    info = pd.DataFrame({
        "descr_text": text,
        "subject": g["host_subject_id"].first() if "host_subject_id" in enr else None,
        "age": pd.to_numeric(g["host_age"].first(), errors="coerce")
               if "host_age" in enr else None,
        "date": g["collection_date"].first() if "collection_date" in enr else None,
        "runs_enriched": g["run_accession"].apply(lambda s: ";".join(sorted(s.dropna()))),
        "bases_enriched": g["_bases"].sum(),
    })
    w = clean.merge(info, left_on="sample", right_index=True, how="left")
    w["bases"] = w["bases_enriched"].fillna(w["bases_qualifying"])
    w["runs_final"] = w["runs_enriched"].fillna(w["runs"])

    # ---- the additional tests ------------------------------------------------
    if args.sample_desc and args.sample_desc.exists():
        sd = pd.read_csv(args.sample_desc, sep="\t", dtype=str, low_memory=False)
        sd["_t"] = (sd.get("description", "").fillna("") + " "
                    + sd.get("sample_title", "").fillna(""))
        extra = dict(zip(sd["sample_accession"], sd["_t"]))
        w["descr_text"] = w["descr_text"].fillna("") + " " + w["sample"].map(extra).fillna("")
        # A candidate the run endpoint cannot resolve has no reads to fetch.
        w["flag_no_runs"] = ~w["sample"].isin(set(enr["sample_accession"].dropna()))
    else:
        w["flag_no_runs"] = False
    w["flag_isolate"] = (w.descr_text.fillna("").str.contains(ISOLATE)
                         | (w.bases / 1e9 >= ISOLATE_DEPTH_GBP))
    w["flag_nonhuman"] = w.descr_text.fillna("").str.contains(NONHUMAN)
    w["flag_infant"] = w.age.notna() & (w.age < NON_INFANT_YEARS)
    w["flag_repeat"] = False
    if "subject" in w and w.subject.notna().any():
        seen = {}
        order = w.sort_values("bases", ascending=False).index      # keep the deepest
        for i in order:
            s = w.at[i, "subject"]
            if not isinstance(s, str) or not s.strip():
                continue
            key = (w.at[i, "study"], s)
            if key in seen:
                w.at[i, "flag_repeat"] = True
            else:
                seen[key] = i
    w["depth_band"] = pd.cut(w.bases, [0, .5e9, 1e9, q1, med, 1e15],
                             labels=["<0.5 Gbp", "0.5-1 Gbp", f"1-{q1/1e9:.1f} Gbp",
                                     f"{q1/1e9:.1f}-{med/1e9:.1f} Gbp",
                                     f">{med/1e9:.1f} Gbp"])
    w["flag_shallow"] = w.bases < q1

    flags = ["flag_isolate", "flag_nonhuman", "flag_infant", "flag_repeat",
             "flag_shallow", "flag_no_runs"]
    w["n_flags"] = w[flags].sum(axis=1)

    print(f"  clean candidates entering the stress test : {len(w):,}\n")
    print(f"  {'additional test':<22}{'flagged':>9}")
    for f in flags:
        print(f"  {f.replace('flag_',''):<22}{int(w[f].sum()):>9,}")
    print(f"\n  {'flags':<8}{'samples':>9}")
    print(w.n_flags.value_counts().sort_index().to_string())

    survive = w[w.n_flags == 0]
    print(f"\n  ===> RECOMMENDED for reprocessing : {len(survive):,}")
    print(f"       lifestyle : {survive.lifestyle.value_counts().to_dict()}")
    print(f"       depth Gbp : median={survive.bases.median()/1e9:.2f} "
          f"Q1={survive.bases.quantile(.25)/1e9:.2f}")
    print(f"       total data: {survive.bases.sum()/1e12:.1f} Tbp to fetch and align")

    print(f"\n  {'study':<14}{'n':>6}{'median Gbp':>12}  lifestyle / country")
    for st, gg in survive.groupby("study"):
        print(f"  {st:<14}{len(gg):>6}{gg.bases.median()/1e9:>12.2f}  "
              f"{gg.lifestyle.mode().iloc[0] if gg.lifestyle.notna().any() else '?':<14}"
              f"{gg.country.dropna().iloc[0] if gg.country.notna().any() else '?'}")

    full = args.out_dir / "reprocessing_candidates_all.csv"
    w.drop(columns=["descr_text"]).to_csv(full, index=False)
    rec = args.out_dir / "reprocessing_recommended.csv"
    survive.drop(columns=["descr_text"]).to_csv(rec, index=False)
    # the Snakefile's own input format: sample_id <TAB> run1;run2
    wl = args.out_dir / "reprocessing_worklist.tsv"
    with wl.open("w") as fh:
        for _, r in survive.iterrows():
            fh.write(f"{r['sample']}\t{r['runs_final']}\n")
    print(f"\n  wrote {full}   ({len(w):,} rows, every flag retained)")
    print(f"  wrote {rec}   ({len(survive):,} rows)")
    print(f"  wrote {wl}   (sample_id<TAB>runs, ready for the Snakefile)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
