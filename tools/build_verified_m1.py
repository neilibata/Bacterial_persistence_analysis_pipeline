#!/usr/bin/env python3
"""Rebuild Metadata M1's health fields from the declared sources, with per-sample
provenance, so every value can be traced to where it came from.

Sources, in the order they are consulted:

  1. mdb   MicrobiomeDB download -- <data root>/plots/MBIOTEMP_NCBI_metadata/*.txt
           (65 study files; carries `disease`, `study_condition`, `subject_id`,
           age/gender/BMI). Column names arrive suffixed `[MBIOTEMP_*]`.
  2. dasilva  da Silva et al. 2026 supplement, reached through
           <data root>/plots/merged_with_media1.csv, whose per-condition columns
           are the `HS_ADD_COLS` of merge_media1_with_chosen.py.
  3. archive  ENA / NCBI BioSample attributes (host_disease, Case_status,
           study_ptorhc, host_subject_id, ...), cached under docs/.
  4. paper  a study's own published supplementary table, staged as
           docs/study_supplements/<StudyFolder>.csv with a `Chosen_match_ID` or
           `BioSample` key. Nothing is invented here: a study only contributes if
           its file is present.

Precedence is deliberately NOT a fixed ranking. Evidence runs both ways -- da
Silva is right about ChengpingW_2017 (ankylosing spondylitis, confirmed by the
paper title and the archive's `AS##` sample ids) where MicrobiomeDB says `RA`,
while MicrobiomeDB is right about FrankelAE/LomanNJ/LokmerA/PehrssonE/MehtaRS,
where M1 carries flags neither the archive nor the paper supports. So:

  * where exactly one source speaks, it is used and recorded;
  * where several agree, the value is used and marked `agree`;
  * where they disagree, the value from the highest-priority source is used but
    the row is written to the Conflicts sheet with every competing value, so the
    disagreement is adjudicated by a person rather than by precedence.

Writes a candidate file; it does not overwrite M1 until --apply is given.

    python tools/build_verified_m1.py
    python tools/build_verified_m1.py --apply
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
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
    _req = ("plots/MBIOTEMP_NCBI_metadata", "plots/merged_with_media1.csv")
    root = config.resolve_data_root(_req)
    if root is None:
        raise SystemExit("ERROR: " + config.describe_data_root_failure(_req))
    return root


DATA_ROOT = _resolve_root()
MDB_DIR = DATA_ROOT / "plots" / "MBIOTEMP_NCBI_metadata"
DASILVA = DATA_ROOT / "plots" / "merged_with_media1.csv"
SUPP_DIR = ROOT / "docs" / "study_supplements"
ARCH_SUBJ = ROOT / "docs" / "archive_subject_ids.tsv"
ARCH_DIS = ROOT / "docs" / "archive_disease.tsv"
ENA_RUN_CACHE = ROOT / "docs" / "ena_run_cache.tsv"

OUT_CSV = ROOT / "final_run" / "metadata_M1_verified.csv"
OUT_XLSX = ROOT / "docs" / "m1_verification_audit.xlsx"

# da Silva per-condition column -> M1 disease_* suffix
DASILVA_MAP = {
    "Ankylosing spondylitis": "ankylosing_spondylitis",
    # da Silva's "Bone disease" is RettedalAE_2021_NZL's osteoporosis cohort (54).
    # It used to land in `disease_BD`, which also held YeZ_2018_CHN's 20 Behcet's
    # samples -- two unrelated diseases in one flag. tools/split_disease_bd.py
    # separated them into disease_bone_disease and disease_Behcets.
    "Bone disease": "bone_disease",
    "Crohn's disease": "Crohn's",
    "Multiple sclerosis": "MS",
    "Myalgic encephalomyelitis": "myalgic_encephalomyelitis",
    "Parkinson's disease": "PD",
    "Rheumatoid arthritis": "RA",
    "Type 1 diabetes": "T1D",
    "Ulcerative colitis": "Ulcerative_colitis",
    "Diabetes": "T2D",
    "Colorectal Cancer": "CRC",
    "Liver Cirrhosis": "cirrhosis",
}
NULL_DISEASE = {"", "na", "none", "healthy", "control", "nan"}

# MicrobiomeDB carries TWO disease fields with different semantics:
#   `disease`         every condition the participant has, comorbidities included
#   `study_condition` the study's own case/control axis, which additionally
#                     distinguishes pre-disease states and intervention arms that
#                     `disease` collapses.
# Exploding `disease` alone therefore over-counts cases in six studies: it calls
# LiJ_2017's 56 pre-hypertensive participants hypertensive, and it calls
# LiSS_2016's and IaniroG_2022's post-FMT timepoints cases. Those samples are not
# controls either, so their flags are left factually intact and a
# `mdb_case_eligible` column marks them, letting the contrast machinery exclude
# them from case arms without deleting records.
PRE_DISEASE_PAT = ("pre-", "pre_", "prehyper", "impaired", "borderline",
                   "at risk", "atrisk", "latent", "remission", "recovered")
INTERVENTION_PAT = ("fmt", "placebo", "treated", "treatment", "post",
                    "intervention", "antibiotic", "probiotic", "supplement",
                    "donor", "recipient", "surgery", "surgery_history", "vaccin")


def choose_primary(df: pd.DataFrame) -> pd.Index:
    """One representative sample per Subject_ID, selected -- never merged.

    Merging a subject's timepoints is not an option: within-subject turnover
    means the union of two timepoints holds taxa neither holds alone, so pooled
    subjects would get systematically higher richness than singly-sampled ones --
    a worse bias than the pseudoreplication being corrected.

    Preference order within a subject:
      1. case-eligible over pre-disease / intervention arms, so an intervention
         study contributes its baseline rather than a treated timepoint;
      2. greatest `Total bases`, because richness is depth-sensitive and the
         downstream GMW step depth-corrects anyway.

    Accumulation curves must NOT use this: repeated samples there simply add
    fewer new taxa, so including them is conservative for catalogue coverage.
    """
    depth = pd.to_numeric(df.get("Total bases"), errors="coerce").fillna(-1)

    # Only consider rows the pipeline will actually see. loading.py resolves
    # ingestion duplicates -- the same run accession appearing twice under
    # inconsistent StudyFolder names, once per sequencing run -- by keeping the
    # largest `Total bases`, because the matrices carry the larger run. Choosing
    # a primary from the whole file instead left 4 subjects whose only primary
    # was the row loading.py discards, so those subjects vanished from every
    # subject-level contrast. Mirror the rule here.
    survives = pd.Series(True, index=df.index)
    if "Chosen_match_ID" in df.columns:
        cm = df["Chosen_match_ID"].astype(str).str.strip()
        if cm.duplicated().any():
            rank = pd.DataFrame({"_tb": depth, "_pos": np.arange(len(df))},
                                index=df.index).assign(_cm=cm)
            kept = (rank.sort_values(["_tb", "_pos"], ascending=[False, True])
                        .drop_duplicates(subset="_cm", keep="first").index)
            survives = df.index.isin(kept)
            survives = pd.Series(survives, index=df.index)
            n = int((~survives).sum())
            if n:
                print(f"  choose_primary: ignoring {n} row(s) that loading.py "
                      f"resolves away as ingestion duplicates")

    elig = df["mdb_case_eligible"].astype(str).str.startswith("No")
    order = pd.DataFrame({"gone": (~survives).astype(int),
                          "bad": elig.astype(int), "depth": -depth})
    keep = []
    for _, idx in df.groupby("Subject_ID", sort=False).groups.items():
        sub = order.loc[idx].sort_values(["gone", "bad", "depth"])
        keep.append(sub.index[0])
    return pd.Index(keep)


def case_eligibility(study_condition) -> str:
    """No for pre-disease or intervention arms, Yes otherwise, "" if unknown."""
    if study_condition is None or pd.isna(study_condition):
        return ""
    v = str(study_condition).strip().lower()
    if v in {"", "nan"}:
        return ""
    if any(t in v for t in PRE_DISEASE_PAT):
        return "No:pre-disease"
    if any(t in v for t in INTERVENTION_PAT):
        return "No:intervention"
    return "Yes"


# Study-level source overrides. Used only where a source has an entire cohort's
# condition wrong -- not for per-sample disagreements, which go to Conflicts.
STUDY_SOURCE_OVERRIDE = {
    # MicrobiomeDB labels this cohort `RA`, but the paper is "Quantitative
    # metagenomics reveals unique gut microbiome biomarkers in ankylosing
    # spondylitis" (Genome Biol 2017, PMID 28750650) and the archive registers
    # its samples as sample_ID AS01..AS##. da Silva's ankylosing_spondylitis
    # flag is correct; MicrobiomeDB's RA is not.
    "ChengpingW_2017_CHN": "dasilva",
}
YES = {"1", "1.0", "y", "yes", "true", "t"}


def is_yes(v) -> bool:
    return str(v).strip().lower() in YES


def sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


def parse_diseases(x) -> list[str]:
    """MicrobiomeDB `disease` -> condition list (explode_diseases.py semantics)."""
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s == "" or s.lower() in NULL_DISEASE:
        return []
    out = []
    for p in re.split(r"[;,]", s):
        p = p.strip()
        if p and p.lower() not in {"na", "none"}:
            out.append(p)
    return out


def strip_suffix(c: str) -> str:
    return re.sub(r"\s*\[MBIOTEMP_[^\]]+\]$", "", c).strip()


# ------------------------------------------------------------------ joins ----

def build_keys(m: pd.DataFrame):
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
    bios: dict[str, int] = {}
    for i, v in m["BioSample"].items():
        if pd.notna(v):
            bios.setdefault(str(v).strip(), i)
    run2bs: dict[str, str] = {}
    if ENA_RUN_CACHE.exists():
        c = pd.read_csv(ENA_RUN_CACHE, sep="\t", dtype=str,
                        usecols=["run_accession", "sample_accession"]).dropna()
        run2bs = dict(zip(c["run_accession"].str.strip(),
                          c["sample_accession"].str.strip()))
    return run_idx, bios, run2bs


def resolve_row(acc, source_id, run_idx, bios, run2bs):
    if pd.notna(acc):
        for t in re.split(r"[;,]", str(acc)):
            t = t.strip()
            if t in run_idx:
                return run_idx[t]
            b = run2bs.get(t)
            if b is not None and b in bios:
                return bios[b]
    if pd.notna(source_id):
        sid = re.sub(r"\s*\(Source\)$", "", str(source_id)).strip()
        if sid in bios:
            return bios[sid]
    return None


# ---------------------------------------------------------------- sources ----

def load_mdb(m, run_idx, bios, run2bs):
    """-> {m1_row: {'terms': [...], 'study_condition':…, 'subject_id':…, …}}"""
    files = sorted(glob.glob(str(MDB_DIR / "*.txt")))
    if not files:
        print(f"  [warn] no MicrobiomeDB files under {MDB_DIR}")
        return {}
    out = {}
    for p in files:
        d = pd.read_csv(p, sep="\t", dtype=str)
        d.columns = [strip_suffix(c) for c in d.columns]
        for _, r in d.iterrows():
            row = resolve_row(r.get("NCBI_accession"), r.get("Source_ID"),
                              run_idx, bios, run2bs)
            if row is None or row in out:
                continue
            out[row] = {
                "terms": parse_diseases(r.get("disease")),
                "disease_raw": r.get("disease"),
                "study_condition": r.get("study_condition"),
                "subject_id": r.get("subject_id"),
                "age": r.get("age"), "gender": r.get("gender"),
                "BMI": r.get("BMI"), "file": os.path.basename(p),
            }
    print(f"  MicrobiomeDB   -> {len(out):,} M1 rows")
    return out


def load_dasilva(m, run_idx, bios, run2bs):
    if not DASILVA.exists():
        print(f"  [warn] {DASILVA} absent")
        return {}
    ds = pd.read_csv(DASILVA, low_memory=False)
    out = {}
    for _, r in ds.iterrows():
        row = resolve_row(r.get("SRA_Accession"), r.get("SRS ID"),
                          run_idx, bios, run2bs)
        if row is None or row in out:
            continue
        flags = {}
        for col, suffix in DASILVA_MAP.items():
            if col in ds.columns and pd.notna(r.get(col)):
                v = str(r[col]).strip().lower()
                if v in {"yes", "no"}:
                    flags[suffix] = (v == "yes")
        out[row] = {"flags": flags, "Unhealthy": r.get("Unhealthy"),
                    "Healthy": r.get("Healthy"), "BMI": r.get("BMI")}
    print(f"  da Silva       -> {len(out):,} M1 rows")
    return out


def load_archive_disease(m):
    """docs/archive_disease.tsv: BioSample<TAB>tag<TAB>value (host_disease etc.)."""
    if not ARCH_DIS.exists():
        print(f"  [note] {ARCH_DIS} absent -- archive disease not consulted")
        return {}
    a = pd.read_csv(ARCH_DIS, sep="\t", dtype=str).dropna()
    by = defaultdict(dict)
    for _, r in a.iterrows():
        by[r["BioSample"]][r["tag"]] = r["value"]
    print(f"  archive disease-> {len(by):,} BioSamples")
    return by


def load_supplements():
    """docs/study_supplements/<StudyFolder>.csv, keyed by Chosen_match_ID or BioSample."""
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(SUPP_DIR / "*.csv")))
    out = {}
    for p in files:
        d = pd.read_csv(p, dtype=str)
        out[Path(p).stem] = d
    print(f"  supplements    -> {len(out)} study file(s) staged"
          + (f": {sorted(out)}" if out else " (none yet)"))
    return out


# ------------------------------------------------------------------- main ----

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    m = pd.read_csv(config.META_FILE, low_memory=False)
    dcols = [c for c in m.columns if c.startswith("disease_")]
    print(f"M1: {len(m):,} rows, {len(dcols)} disease_* columns\n")

    run_idx, bios, run2bs = build_keys(m)
    mdb = load_mdb(m, run_idx, bios, run2bs)
    ds = load_dasilva(m, run_idx, bios, run2bs)
    arch = load_archive_disease(m)
    supp = load_supplements()

    # MicrobiomeDB's condition vocabulary decides which flags it can speak to
    vocab = set()
    for v in mdb.values():
        vocab |= {sanitize(t) for t in v["terms"]}
    mdb_flags = {c[len("disease_"):] for c in dcols
                 if c[len("disease_"):] in vocab}
    ds_flags = set(DASILVA_MAP.values())
    print(f"\n  flags MicrobiomeDB can set : {len(mdb_flags)}")
    print(f"  flags da Silva can set     : {len(ds_flags)}")
    print(f"  flags neither can set      : "
          f"{sorted({c[len('disease_'):] for c in dcols} - mdb_flags - ds_flags)}")

    out = m.copy()
    prov, conflicts, changes = [], [], []
    bs = m["BioSample"].astype(str).str.strip()

    for i in range(len(m)):
        sample_src, srcs = {}, []
        sf_i = str(m.at[i, "StudyFolder"])
        override = STUDY_SOURCE_OVERRIDE.get(sf_i)
        if i in mdb and override != "dasilva":
            srcs.append("mdb")
            want = {sanitize(t) for t in mdb[i]["terms"]}
            for f in mdb_flags:
                sample_src[f] = (f in want, "mdb")
        if i in mdb and override == "dasilva":
            srcs.append("mdb(overridden)")
        if i in ds:
            srcs.append("dasilva")
            for f, val in ds[i]["flags"].items():
                if f in sample_src and sample_src[f][0] != val:
                    conflicts.append({
                        "Chosen_match_ID": m.at[i, "Chosen_match_ID"],
                        "StudyFolder": m.at[i, "StudyFolder"],
                        "flag": f,
                        "microbiomedb": sample_src[f][0],
                        "dasilva": val,
                        "mdb_disease_text": mdb[i]["disease_raw"] if i in mdb else "",
                        "current_M1": is_yes(m.at[i, f"disease_{f}"])
                        if f"disease_{f}" in m.columns else None,
                    })
                    # keep the higher-priority (mdb) value; conflict is recorded
                else:
                    sample_src.setdefault(f, (val, "dasilva"))
        a = arch.get(bs.iat[i], {})
        if a:
            srcs.append("archive")
        sf = sf_i
        if sf in supp:
            srcs.append("paper")

        for f, (val, src) in sample_src.items():
            col = f"disease_{f}"
            if col not in out.columns:
                continue
            old = is_yes(m.at[i, col])
            if old != val:
                changes.append({"Chosen_match_ID": m.at[i, "Chosen_match_ID"],
                                "StudyFolder": sf, "flag": f,
                                "was": old, "now": val, "source": src})
            out.at[i, col] = "Yes" if val else "No"
        prov.append(";".join(srcs) if srcs else "none")

    out["health_source"] = prov
    out["mdb_study_condition"] = [
        (mdb[i]["study_condition"] if i in mdb else "") for i in range(len(m))]
    out["mdb_case_eligible"] = [
        case_eligibility(mdb[i]["study_condition"]) if i in mdb else ""
        for i in range(len(m))]

    # one primary sample per subject (selection, never merging)
    if "Subject_ID" in out.columns:
        primary = choose_primary(out)
        out["subject_primary"] = "No"
        out.loc[primary, "subject_primary"] = "Yes"
        resolved = out["Subject_ID_source"].astype(str) != "unresolved"
        sec = out["subject_primary"].eq("No")
        print(f"\n  subject_primary: {len(primary):,} primaries, "
              f"{int(sec.sum()):,} secondary samples")
        print(f"    of the secondaries, {int((sec & resolved).sum()):,} have a real "
              f"subject key and {int((sec & ~resolved).sum()):,} do not")
        dd = out[sec].groupby("StudyFolder").size().sort_values(ascending=False)
        if len(dd):
            print(f"    secondaries by study (top 14):\n{dd.head(14).to_string()}")
    else:
        print("\n  [warn] no Subject_ID column; run tools/build_subject_ids.py first")
    ch = pd.DataFrame(changes)
    cf = pd.DataFrame(conflicts)
    print(f"\n── result ──")
    print(f"  samples with at least one source : "
          f"{int((out['health_source'] != 'none').sum()):,} / {len(out):,}")
    print(f"  samples with NO source           : "
          f"{int((out['health_source'] == 'none').sum()):,}")
    print(f"  flag values changed              : {len(ch):,}")
    print(f"  source conflicts to adjudicate   : {len(cf):,}")
    if len(ch):
        print(f"\n  changes by study (top 12):\n"
              f"{ch['StudyFolder'].value_counts().head(12).to_string()}")
        print(f"\n  changes by flag (top 12):\n"
              f"{ch['flag'].value_counts().head(12).to_string()}")
    if len(cf):
        print(f"\n  conflicts by flag:\n{cf['flag'].value_counts().to_string()}")

    print(f"\n  case eligibility (MicrobiomeDB study_condition):")
    for k, v in out["mdb_case_eligible"].replace("", "(no study_condition)"
                                                 ).value_counts().items():
        print(f"    {k:<24}{v:>8,}")
    ne = out[out["mdb_case_eligible"].astype(str).str.startswith("No")]
    if len(ne):
        print(f"  samples not case-eligible, by study:")
        print(ne.groupby(["StudyFolder", "mdb_case_eligible"]).size().to_string())

    print(f"\n  source coverage:\n{out['health_source'].value_counts().to_string()}")
    studies_nosrc = out.loc[out['health_source'] == 'none', 'StudyFolder']
    if len(studies_nosrc):
        print(f"\n  studies with NO health source (need a supplement):")
        print(studies_nosrc.value_counts().head(30).to_string())

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        (ch if len(ch) else pd.DataFrame({"note": ["no changes"]})
         ).to_excel(xw, sheet_name="Flag changes", index=False)
        (cf if len(cf) else pd.DataFrame({"note": ["no conflicts"]})
         ).to_excel(xw, sheet_name="Conflicts", index=False)
        out.groupby("StudyFolder")["health_source"].value_counts().unstack(
            fill_value=0).to_excel(xw, sheet_name="Source by study")
        out[["Chosen_match_ID", "StudyFolder", "health_source"]].to_excel(
            xw, sheet_name="Per sample", index=False)
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_XLSX}")
    if args.apply:
        src = Path(config.META_FILE)
        bak = src.with_suffix(src.suffix + ".pre_verified")
        if not bak.exists():
            shutil.copy2(src, bak); print(f"backed up -> {bak}")
        out.to_csv(src, index=False); print(f"applied to {src}")
    else:
        print("(candidate only; rerun with --apply to replace M1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
