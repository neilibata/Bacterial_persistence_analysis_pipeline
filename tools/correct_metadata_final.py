#!/usr/bin/env python3
"""Final corrections to Metadata Table M1: konzo cases, and column cleanup.

Two unrelated things, done together because both edit the same file.

1. BrambleMS_2021_COD. ENA records ``host_disease`` per sample: 60 of the 180
   are ``konzo`` and 120 ``unaffected``. All 180 were labelled Healthy. Konzo
   is a primary disease -- a paralytic neuropathy -- so the 60 are cases, and a
   ``disease_konzo`` column is added to record it the way every other condition
   is recorded. Note the sample names are no guide: the samples called
   ``Konzo1``, ``Konzo2`` ... are the study's own numbering and are mostly
   *unaffected*, while the cases are named ``L2xx``.

2. Health columns. Eight columns describe health status, and only `Unhealthy`
   is the control/case split the statistics run on -- it is what Section 3
   compares, and the one the curation work corrects.

   The other seven are roll-ups and second opinions, but "no computation
   consults them" is only true of loading.py, which maps two of them to flags
   it then never uses. Looked at across the whole tree, five of the seven have
   live readers, so only three are dropped: `Index`, `Healthy` and
   `Unhealthy (check)`. DROP_COLUMNS records which reader keeps each of the
   others alive, and why they cannot simply be re-derived from the disease_*
   flags.

   `Secondary_run_IDs` is NOT dropped. The reason given here previously --
   that the reinstatement step uses it to decide cohort membership, so
   removing it would change N -- is no longer true: reinstatement has been off
   since config.REINSTATE_SECONDARIES was set False, and
   reinstate_distinct_secondaries now returns the data unchanged. N does not
   move either way.

   The real reason is that loading.py turns the column into `sec_map`,
   `sec_breadth` and `sec_depth` on the ReferenceData, and three shipped
   figures are built from those and from nothing else:

     S2  tech_rep_alpha_scatter   ) section1_tech_replicates.py: the
     S3  tech_rep_depth_vs_bc     ) technical-replicate concordance analysis
     S1  figure_s0_roadmap        secondary-run counts in the roadmap

   None of them depends on the reinstatement flag -- the replicate pairs are
   the same population whether or not any of them is promoted to an
   independent sample. Dropping the column would empty sec_map and silently
   reduce those three figures to their no-data branches.

    python tools/correct_metadata_final.py --dry-run
    python tools/correct_metadata_final.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_IO = ROOT / "final_run" / "metadata_M1.csv"
CHANGELOG = ROOT / "final_run" / "final_corrections.csv"
HOSTCACHE = ROOT / "docs" / "ena_host_disease_cache.json"

#: read by nothing; a roll-up, a second opinion, or a stale index
# Only columns nothing reads. The five `Diseased*` roll-ups were originally in
# this list on the grounds that no computation consults them -- that was wrong:
#   * visualization.build_country_summary reads three of them for its
#     per-country disease fractions (frac_gut_metab_disease,
#     frac_other_disease, frac_metab_or_obese);
#   * tools/metadata_explorer shows all four "Diseased (...)" as dashboard
#     proportions;
#   * tools/build_audited_metadata.py uses "Diseased (all)" for its
#     not-rolled-up QC check.
# Because _frac_yes returns NaN for a missing column, dropping them turned
# those fractions silently into NaN rather than raising. They also cannot be
# re-derived from the disease_* flags: the closest principled reconstruction
# from config.DISEASE_NCD_MAP agrees with the original columns on only 85-97%
# of rows, so substituting it would change published country-level numbers.
DROP_COLUMNS = [
    "Index",
    "Healthy",
    "Unhealthy (check)",
]

#: the one the analysis actually uses, kept
KEEP = "Unhealthy"


def host_disease_for(accessions, cache_path=HOSTCACHE, delay=0.2, verbose=True):
    """{accession: host_disease} from ENA's sample XML, cached."""
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    need = [a for a in accessions if a not in cache]
    for i, acc in enumerate(need, 1):
        try:
            xml = urllib.request.urlopen(
                f"https://www.ebi.ac.uk/ena/browser/api/xml/{acc}",
                timeout=25).read().decode("utf-8", "replace")
            tags = dict((t.strip(), v.strip()) for t, v in re.findall(
                r"<TAG>(.*?)</TAG>\s*<VALUE>(.*?)</VALUE>", xml, re.S))
            cache[acc] = tags.get("host_disease", "")
        except Exception:
            cache[acc] = None
        if verbose and (i % 20 == 0 or i == len(need)):
            print(f"      {i}/{len(need)} fetched", end="\r", flush=True)
        time.sleep(delay)
    if verbose and need:
        print()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=0, sort_keys=True))
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io", default=str(DEFAULT_IO))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-network", action="store_true")
    a = ap.parse_args()

    m = pd.read_csv(a.io, low_memory=False)
    print(f"  {Path(a.io).name}: {len(m):,} rows, {len(m.columns)} columns")
    rows = []

    # ── 1. konzo ─────────────────────────────────────────────────────────
    br = m["StudyFolder"] == "BrambleMS_2021_COD"
    print(f"\n  BrambleMS_2021_COD: {int(br.sum())} samples, "
          f"currently {dict(m.loc[br, KEEP].astype(str).value_counts())}")
    accs = [str(x).strip() for x in m.loc[br, "BioSample"]]
    hd = ({} if a.no_network and not HOSTCACHE.exists()
          else host_disease_for(accs))
    disease = pd.Series([str(hd.get(str(x).strip()) or "").strip().lower()
                         for x in m["BioSample"]], index=m.index)
    konzo = br & disease.eq("konzo")
    print(f"    host_disease=konzo      : {int(konzo.sum())}")
    print(f"    host_disease=unaffected : "
          f"{int((br & disease.eq('unaffected')).sum())}")

    if int(konzo.sum()):
        if "disease_konzo" not in m.columns:
            # Place it beside the other condition columns rather than at the end.
            pos = max((i for i, c in enumerate(m.columns)
                       if c.startswith("disease_")), default=len(m.columns) - 1)
            m.insert(pos + 1, "disease_konzo", "")
            print("    added column 'disease_konzo'")
        for i in m.index[konzo]:
            rows.append(dict(Chosen_match_ID=m.at[i, "Chosen_match_ID"],
                             StudyFolder="BrambleMS_2021_COD",
                             column="disease_konzo", old="", new="Yes",
                             reason="ENA sample attribute host_disease=konzo"))
            if str(m.at[i, KEEP]).strip() != "Unhealthy":
                rows.append(dict(Chosen_match_ID=m.at[i, "Chosen_match_ID"],
                                 StudyFolder="BrambleMS_2021_COD",
                                 column=KEEP, old=m.at[i, KEEP],
                                 new="Unhealthy",
                                 reason="konzo is a primary disease"))
        m.loc[konzo, "disease_konzo"] = "Yes"
        m.loc[konzo, KEEP] = "Unhealthy"

    # ── 2. columns ───────────────────────────────────────────────────────
    present = [c for c in DROP_COLUMNS if c in m.columns]
    print(f"\n  dropping {len(present)} health/index columns that nothing reads:")
    for c in present:
        v = m[c].astype(str).str.strip()
        n = int(((v != "") & (v.str.lower() != "nan")).sum())
        print(f"      {c:34} was populated on {n:>6,} rows")
    print(f"    keeping '{KEEP}' (the analysis split) and 'Secondary_run_IDs' "
          "(read by the replicate reinstatement)")

    u = m[KEEP].astype(str).str.strip()
    print(f"\n  Unhealthy {int((u == 'Unhealthy').sum()):,}   "
          f"Healthy {int((u == 'Healthy').sum()):,}   "
          f"unlabelled {int((~u.isin(['Healthy', 'Unhealthy'])).sum()):,}")

    if a.dry_run:
        print("\n  dry run; nothing written")
        return

    m = m.drop(columns=present)
    m.to_csv(a.io, index=False)
    if rows:
        pd.DataFrame(rows).to_csv(CHANGELOG, index=False)
    print(f"\n  wrote {Path(a.io).relative_to(ROOT)}  "
          f"({len(m):,} rows, {len(m.columns)} columns)")
    if rows:
        print(f"  wrote {CHANGELOG.relative_to(ROOT)}  ({len(rows):,} changes)")


if __name__ == "__main__":
    main()
