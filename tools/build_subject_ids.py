#!/usr/bin/env python3
"""Resolve a Subject_ID for every sample in Metadata M1, with provenance.

Why. Several studies contribute many samples per person, and M1 has no column
that says so. `Sample ID` cannot be trusted for this: in Lloyd-PriceJL_2019 it
*is* the participant id (1,125 samples / 104 people), while iMSMS_2022 assigns a
unique `Sample ID` per sample despite 1,133 samples coming from 611 people. Any
correction for non-independence needs a per-sample subject key first, and it
needs to say where each key came from so a reader can audit it.

Resolution order (first hit wins; the source is recorded per sample):
  1. cmd_subject_id   curatedMetagenomicData `subject_id`, joined through the
                      same three tiers as tools/build_cmd_health_audit.py
  2. archive_<tag>    a person-identifying attribute from the ENA/NCBI BioSample
                      record (host_subject_id, subject_id, participant, donor,
                      awigen_id, Isolate, ...), cached in
                      docs/archive_subject_ids.tsv
  3. sample_id_repeats  M1 `Sample ID`, used ONLY for studies where it is
                      demonstrably a subject key -- i.e. it repeats within the
                      study. Where `Sample ID` is unique per row it carries no
                      subject information and is not used.
  4. unresolved       no subject key could be established; Subject_ID falls back
                      to Chosen_match_ID so the column is always usable, and the
                      source column says `unresolved` so those rows can be
                      excluded from any subject-level analysis.

Per-sample tags that identify a *sample* rather than a person (sample_name,
alias, library_ID, aliquot) are deliberately NOT used: treating them as subject
keys would manufacture false independence.

    python tools/build_subject_ids.py                    # use the cache
    python tools/build_subject_ids.py --refresh-archive  # refetch ENA/NCBI
    python tools/build_subject_ids.py --apply            # add the columns to M1
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from persistence import config  # noqa: E402

# Run from the repo root regardless of where the tool was invoked: config.py
# names its inputs by bare filename relative to the working directory, so a tool
# launched from elsewhere would silently read nothing.
os.chdir(ROOT)


def _resolve_root():
    """External metadata sources, located by search (see docs/PORTING.md)."""
    _req = ("plots/MBIOTEMP_NCBI_metadata",)
    root = config.resolve_data_root(_req)
    if root is None:
        raise SystemExit("ERROR: " + config.describe_data_root_failure(_req))
    return root


DATA_ROOT = _resolve_root()
CMD_DIR = DATA_ROOT / "plots" / "MBIOTEMP_NCBI_metadata"
ARCHIVE_CACHE = ROOT / "docs" / "archive_subject_ids.tsv"
ENA_RUN_CACHE = ROOT / "docs" / "ena_run_cache.tsv"
OUT_MAP = ROOT / "final_run" / "subject_ids.csv"
OUT_XLSX = ROOT / "docs" / "subject_id_audit.xlsx"

NCBI = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        "?db=biosample&rettype=xml&id=")
ENA = "https://www.ebi.ac.uk/ena/browser/api/xml/"

# Tags that identify a PERSON. Ordered: earlier tags win.
PERSON_TAGS = [
    "host_subject_id", "host subject id", "host_individual_id", "subject_id",
    "subject id", "submitted_subject_id", "participant_id", "participant",
    "individual", "individual_name", "donor_id", "donor", "patient_id",
    "patient", "awigen_id", "host_id", "isolate",
]
# Tags that identify a SAMPLE, never a person. Listed so the exclusion is
# explicit and reviewable rather than implicit.
SAMPLE_ONLY_TAGS = {"sample_name", "alias", "library_id", "aliquot",
                    "sample_title", "biomaterial_provider"}

# Archive values that carry no information.
NULL_VALUES = {"", "na", "n/a", "none", "missing", "not applicable",
               "not collected", "unknown", "restricted access", "nan"}


def _clean(v: str) -> str:
    return (v or "").strip()


def _usable(v: str) -> bool:
    return _clean(v).lower() not in NULL_VALUES


def _parse_isolate(v: str) -> str:
    """BritoIL_2016 registers `Isolate` as M1.48.ST -- subject is M1.48."""
    v = _clean(v)
    return re.sub(r"\.(ST|SA|OR|NA)$", "", v, flags=re.I)


# ---------------------------------------------------------------- archive ----

def fetch_archive(accs: list[str]) -> pd.DataFrame:
    """Fetch person-identifying attributes for BioSample accessions."""
    rows = []
    ncbi = [a for a in accs if a.startswith("SAMN")]
    other = [a for a in accs if not a.startswith("SAMN")]
    for lst, use_ena in ((ncbi, False), (other, True)):
        B = 120
        for i in range(0, len(lst), B):
            chunk = lst[i:i + B]
            root = None
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(
                            (ENA if use_ena else NCBI) + ",".join(chunk),
                            timeout=120) as r:
                        root = ET.fromstring(r.read())
                    break
                except Exception:
                    if attempt == 2:
                        print(f"    [warn] batch at {i} failed", flush=True)
                    else:
                        time.sleep(4)
            if root is None:
                continue
            for bs in root.iter("BioSample"):
                acc = bs.get("accession")
                for a in bs.iter("Attribute"):
                    tag = (a.get("harmonized_name")
                           or a.get("attribute_name") or "").strip().lower()
                    if tag in SAMPLE_ONLY_TAGS or tag not in PERSON_TAGS:
                        continue
                    if _usable(a.text):
                        rows.append((acc, tag, _clean(a.text)))
            for s in root.iter("SAMPLE"):
                acc = s.get("accession")
                for a in s.iter("SAMPLE_ATTRIBUTE"):
                    tag = (a.findtext("TAG") or "").strip().lower()
                    if tag in SAMPLE_ONLY_TAGS or tag not in PERSON_TAGS:
                        continue
                    if _usable(a.findtext("VALUE")):
                        rows.append((acc, tag, _clean(a.findtext("VALUE"))))
            if (i // B) % 10 == 0:
                print(f"    fetched {i + len(chunk)}/{len(lst)}"
                      f"  rows {len(rows):,}", flush=True)
            time.sleep(0.35)
    return pd.DataFrame(rows, columns=["BioSample", "tag", "value"])


def load_archive(m: pd.DataFrame, refresh: bool) -> pd.DataFrame:
    # A missing cache used to trigger a silent ~18,000-record refetch, which on a
    # fresh machine looks like a hang and needs network the run may not have.
    # Make it an explicit choice instead.
    if not refresh and not ARCHIVE_CACHE.exists():
        raise SystemExit(
            f"ERROR: archive cache missing: {ARCHIVE_CACHE}\n"
            "  It ships with the transfer package (see docs/PORTING.md). Either\n"
            "  restore it, or pass --refresh-archive to refetch ~18,000 BioSample\n"
            "  records from ENA/NCBI (needs network, takes ~15-20 min).")
    if refresh or not ARCHIVE_CACHE.exists():
        accs = sorted(m["BioSample"].dropna().astype(str).str.strip().unique())
        print(f"  fetching archive attributes for {len(accs):,} BioSamples ...")
        df = fetch_archive(accs)
        ARCHIVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ARCHIVE_CACHE, sep="\t", index=False)
        print(f"  cached -> {ARCHIVE_CACHE}")
    else:
        df = pd.read_csv(ARCHIVE_CACHE, sep="\t", dtype=str)
        print(f"  archive cache: {len(df):,} attribute rows "
              f"({df['BioSample'].nunique():,} samples)")
    return df


# -------------------------------------------------------------------- cMD ----

def load_cmd_subjects(m: pd.DataFrame) -> dict[str, str]:
    """Chosen_match_ID -> cMD subject_id, via run / BioSample / ENA-cache."""
    files = sorted(glob.glob(str(CMD_DIR / "*.txt")))
    if not files:
        print(f"  [warn] no cMD files under {CMD_DIR}")
        return {}
    def strip(c): return re.sub(r"\s*\[MBIOTEMP_[^\]]+\]$", "", c).strip()
    fr = []
    for p in files:
        d = pd.read_csv(p, sep="\t", dtype=str)
        d.columns = [strip(c) for c in d.columns]
        cols = [c for c in ("NCBI_accession", "Source_ID", "subject_id") if c in d]
        if "subject_id" in cols:
            fr.append(d[cols])
    cmd = pd.concat(fr, ignore_index=True, sort=False)
    cmd = cmd[cmd["subject_id"].apply(_usable)]
    print(f"  cMD rows carrying subject_id: {len(cmd):,}")

    run_idx: dict[str, int] = {}
    for col in ("Chosen_match_ID", "SRA_Accession", "Secondary_run_IDs"):
        if col not in m.columns:
            continue
        for i, v in m[col].items():
            if pd.isna(v):
                continue
            for t in re.split(r"[;,]", str(v)):
                t = t.strip()
                if t:
                    run_idx.setdefault(t, i)
    bios = {}
    for i, v in m["BioSample"].items():
        if pd.notna(v):
            bios.setdefault(str(v).strip(), i)
    run2bs = {}
    if ENA_RUN_CACHE.exists():
        c = pd.read_csv(ENA_RUN_CACHE, sep="\t", dtype=str,
                        usecols=["run_accession", "sample_accession"]).dropna()
        run2bs = dict(zip(c["run_accession"].str.strip(),
                          c["sample_accession"].str.strip()))

    out: dict[str, str] = {}
    cmid = m["Chosen_match_ID"].astype(str)
    for _, r in cmd.iterrows():
        row = None
        acc = r.get("NCBI_accession")
        if pd.notna(acc):
            for t in re.split(r"[;,]", str(acc)):
                t = t.strip()
                if t in run_idx:
                    row = run_idx[t]; break
                b = run2bs.get(t)
                if b is not None and b in bios:
                    row = bios[b]; break
        if row is None and pd.notna(r.get("Source_ID")):
            sid = re.sub(r"\s*\(Source\)$", "", str(r["Source_ID"])).strip()
            row = bios.get(sid)
        if row is not None:
            out.setdefault(cmid.iat[row], _clean(r["subject_id"]))
    print(f"  cMD subject_id mapped onto {len(out):,} M1 samples")
    return out


# ------------------------------------------------------------------- main ----

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-archive", action="store_true")
    ap.add_argument("--apply", action="store_true",
                    help="add Subject_ID / Subject_ID_source to M1 (backs up first)")
    args = ap.parse_args()

    m = pd.read_csv(config.META_FILE, low_memory=False)
    print(f"M1: {len(m):,} rows, {m['StudyFolder'].nunique()} studies\n")

    cmd_map = load_cmd_subjects(m)
    arch = load_archive(m, args.refresh_archive)

    # best archive tag per sample, honouring PERSON_TAGS order
    # ENA registers the same tag with either spaces or underscores
    # ("host subject id" / "host_subject_id"), which otherwise produces two
    # source labels for one field. Normalise before ranking.
    arch["tag"] = arch["tag"].astype(str).str.strip().str.lower().str.replace(
        r"[ \-]+", "_", regex=True)
    rank = {t.replace(" ", "_"): i for i, t in enumerate(PERSON_TAGS)}
    arch = arch[arch["tag"].isin(rank)]
    arch["rank"] = arch["tag"].map(rank)
    arch = arch.sort_values("rank").drop_duplicates("BioSample")
    arch_val = dict(zip(arch["BioSample"], arch["value"]))
    arch_tag = dict(zip(arch["BioSample"], arch["tag"]))

    # studies where Sample ID repeats -> it is a subject key for that study
    rep = (m.groupby("StudyFolder")["Sample ID"]
             .agg(rows="size", uniq="nunique"))
    sid_studies = set(rep.index[(rep["rows"] / rep["uniq"].clip(lower=1)) > 1.02])

    cmid = m["Chosen_match_ID"].astype(str)
    bs = m["BioSample"].astype(str).str.strip()
    sid = m["Sample ID"].astype(str)

    # Scope the subject key by study ROOT, not StudyFolder. A trailing 3-letter
    # country code is a population split WITHIN one study, so folder scoping
    # would count one person twice: CarterM_2023 shares 11 bare subject ids with
    # CarterM_2023_TZA/_NPL. Checked across all 11 multi-folder study roots --
    # Carter is the only one with any overlap, so collapsing to the root is
    # correct there and a no-op everywhere else.
    sf = m["StudyFolder"].astype(str).str.replace(r"_[A-Z]{3}$", "", regex=True)

    subj, src = [], []
    for i in range(len(m)):
        k = cmid.iat[i]
        if k in cmd_map:
            subj.append(f"{sf.iat[i]}|{cmd_map[k]}"); src.append("cmd_subject_id"); continue
        b = bs.iat[i]
        if b in arch_val:
            tag = arch_tag[b]
            v = _parse_isolate(arch_val[b]) if tag == "isolate" else arch_val[b]
            subj.append(f"{sf.iat[i]}|{v}"); src.append(f"archive_{tag}"); continue
        if sf.iat[i] in sid_studies and _usable(sid.iat[i]):
            subj.append(f"{sf.iat[i]}|{sid.iat[i]}"); src.append("sample_id_repeats"); continue
        subj.append(f"{sf.iat[i]}|{k}"); src.append("unresolved")

    m["Subject_ID"] = subj
    m["Subject_ID_source"] = src

    # ---- per-study report ----
    g = m.groupby("StudyFolder").apply(
        lambda d: pd.Series({
            "samples": len(d),
            "subjects": d.loc[d.Subject_ID_source != "unresolved", "Subject_ID"].nunique(),
            "resolved": int((d.Subject_ID_source != "unresolved").sum()),
            "sources": ";".join(sorted(set(d.Subject_ID_source))),
        }), include_groups=False)
    g["resolved_%"] = (100 * g.resolved / g.samples).round(0)
    g["repeat_factor"] = (g.resolved / g.subjects.clip(lower=1)).round(2)
    g = g.sort_values(["repeat_factor", "samples"], ascending=False)

    res = g[g["resolved_%"] >= 80]
    unres = g[g["resolved_%"] < 80]
    dirty = g[(g["repeat_factor"] > 1.02) & (g["resolved_%"] >= 80)]
    print(f"\n── coverage ──")
    print(f"  samples with a subject key : {int((m.Subject_ID_source!='unresolved').sum()):,}"
          f" / {len(m):,}")
    print(f"  studies >=80% resolved     : {len(res)}")
    print(f"  studies <80% resolved      : {len(unres)}  "
          f"({int(unres.samples.sum()):,} samples)")
    print(f"  studies with repeats       : {len(dirty)}  "
          f"({int(dirty.samples.sum()):,} samples, "
          f"{int((dirty.samples - dirty.subjects).sum()):,} excess)")
    print(f"\n── source breakdown ──")
    for k, v in m["Subject_ID_source"].value_counts().items():
        print(f"  {k:<24}{v:>8,}")
    print(f"\n── studies with repeat_factor > 1.02 ──")
    print(dirty[["samples", "subjects", "repeat_factor", "resolved_%",
                 "sources"]].to_string())
    print(f"\n── studies still unresolved (<80%) ──")
    print(unres[["samples", "resolved", "resolved_%"]].head(40).to_string())

    OUT_MAP.parent.mkdir(parents=True, exist_ok=True)
    m[["Chosen_match_ID", "StudyFolder", "Sample ID", "BioSample",
       "Subject_ID", "Subject_ID_source"]].to_csv(OUT_MAP, index=False)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        g.reset_index().to_excel(xw, sheet_name="By study", index=False)
        dirty.reset_index().to_excel(xw, sheet_name="Repeated subjects", index=False)
        unres.reset_index().to_excel(xw, sheet_name="Unresolved studies", index=False)
        m[["Chosen_match_ID", "StudyFolder", "Sample ID", "BioSample",
           "Subject_ID", "Subject_ID_source"]].to_excel(
            xw, sheet_name="Per sample", index=False)
    print(f"\nwrote {OUT_MAP}\nwrote {OUT_XLSX}")

    if args.apply:
        src_path = Path(config.META_FILE)
        bak = src_path.with_suffix(src_path.suffix + ".pre_subject_ids")
        if not bak.exists():
            shutil.copy2(src_path, bak)
            print(f"backed up {src_path} -> {bak}")
        m.to_csv(src_path, index=False)
        print(f"added Subject_ID / Subject_ID_source to {src_path}")
    else:
        print("\n(not applied to M1; rerun with --apply to add the columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
