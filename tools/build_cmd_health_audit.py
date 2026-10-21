#!/usr/bin/env python3
"""Paper trail for the health metadata: curatedMetagenomicData vs Metadata M1.

Why this exists. M1's `disease_*` flags are a union of two differently-sourced
families, and this workbook is where that extraction is written down. One
family is produced by exploding curatedMetagenomicData's
free-text `disease` field on ';'/','; the other is carried over from the
da Silva et al. 2026 per-condition columns. Because of that, a flag that cMD
does not support is *not* automatically an error -- it may be da Silva's. This
workbook records, per sample, exactly what cMD says and exactly what M1 says,
and marks which of the two families each flag could have come from, so a reader
can audit any sample rather than trust a summary.

Source. The 65 per-study files under

    <data root>/plots/MBIOTEMP_NCBI_metadata/*.txt

are curatedMetagenomicData exports (schema verbatim, including the per-row
`curator`). They are rebuilt here rather than read from a merged product so the
trail starts at the source.

Join, in three tiers. (1) cMD `NCBI_accession` (run, sometimes ';'-separated)
against M1 `Chosen_match_ID` / `SRA_Accession` / `Secondary_run_IDs`.
(2) `Source_ID` (BioSample) against M1 `BioSample`. (3) cMD runs through
`docs/ena_run_cache.tsv` (run_accession -> sample_accession) onto M1
`BioSample` -- needed for studies like ZellerG_2014 where cMD records runs but
M1 keys on sample accessions, so tiers 1 and 2 cannot meet.

    python tools/build_cmd_health_audit.py
    python tools/build_cmd_health_audit.py --out docs/cmd_health_audit.xlsx
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
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
DEFAULT_OUT = ROOT / "docs" / "cmd_health_audit.xlsx"

# cMD fields worth carrying into the trail. Everything else in the source files
# is assay chemistry or condition-specific labs, kept out to keep the sheet
# readable; rerun with --all-columns to include them.
CORE = ["study_name", "subject_id", "disease", "study_condition",
        "disease_subtype", "disease_stage", "age", "age_category", "gender",
        "BMI", "country", "non_westernized", "antibiotics_current_use",
        "PMID", "curator", "NCBI_accession", "Source_ID"]

# Tokens cMD uses for "no disease"; explode_diseases.py treats these as empty.
NULL_DISEASE = {"na", "none", "healthy", "control", "", "nan"}


def strip_suffix(c: str) -> str:
    return re.sub(r"\s*\[MBIOTEMP_[^\]]+\]$", "", c).strip()


def sanitize(name: str) -> str:
    """Column-name form used by explode_diseases.py."""
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


def parse_diseases(x) -> list[str]:
    """Verbatim reimplementation of explode_diseases.py:parse_diseases."""
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


def load_cmd(all_columns: bool = False) -> pd.DataFrame:
    files = sorted(glob.glob(str(CMD_DIR / "*.txt")))
    if not files:
        raise SystemExit(f"no cMD files under {CMD_DIR}")
    frames = []
    for p in files:
        d = pd.read_csv(p, sep="\t", dtype=str)
        d.columns = [strip_suffix(c) for c in d.columns]
        d["cmd_file"] = os.path.basename(p)
        frames.append(d)
    cmd = pd.concat(frames, ignore_index=True, sort=False)
    if not all_columns:
        keep = [c for c in CORE if c in cmd.columns] + ["cmd_file"]
        cmd = cmd[keep]
    return cmd


def build_run_index(m: pd.DataFrame) -> dict[str, int]:
    idx: dict[str, int] = {}
    for col in ("Chosen_match_ID", "SRA_Accession", "Secondary_run_IDs"):
        if col not in m.columns:
            continue
        for i, v in m[col].items():
            if pd.isna(v):
                continue
            for t in re.split(r"[;,]", str(v)):
                t = t.strip()
                if t:
                    idx.setdefault(t, i)
    return idx


def load_run_to_biosample() -> dict[str, str]:
    """run_accession -> sample_accession from the committed ENA cache."""
    cache = ROOT / "docs" / "ena_run_cache.tsv"
    if not cache.exists():
        print(f"  [warn] {cache} absent; skipping the run->BioSample join tier")
        return {}
    c = pd.read_csv(cache, sep="\t", dtype=str,
                    usecols=["run_accession", "sample_accession"])
    c = c.dropna(subset=["run_accession", "sample_accession"])
    return dict(zip(c["run_accession"].str.strip(),
                    c["sample_accession"].str.strip()))


def is_yes(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(
        {"1", "1.0", "y", "yes", "true", "t"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--all-columns", action="store_true",
                    help="carry every cMD field, not just the core health set")
    args = ap.parse_args()

    cmd = load_cmd(args.all_columns)
    m = pd.read_csv(config.META_FILE, low_memory=False)
    dcols = [c for c in m.columns if c.startswith("disease_")]
    print(f"cMD rows {len(cmd):,} from {cmd['cmd_file'].nunique()} studies")
    print(f"M1  rows {len(m):,} with {len(dcols)} disease_* columns")

    # ---- join -----------------------------------------------------------
    run_idx = build_run_index(m)
    bios = {}
    if "BioSample" in m.columns:
        for i, v in m["BioSample"].items():
            if pd.notna(v):
                bios.setdefault(str(v).strip(), i)

    def by_run(acc):
        if pd.isna(acc):
            return None
        for t in re.split(r"[;,]", str(acc)):
            t = t.strip()
            if t in run_idx:
                return run_idx[t]
        return None

    def by_biosample(sid):
        if pd.isna(sid):
            return None
        return bios.get(re.sub(r"\s*\(Source\)$", "", str(sid)).strip())

    m1row = cmd["NCBI_accession"].apply(by_run) if "NCBI_accession" in cmd else pd.Series(
        [None] * len(cmd), index=cmd.index)
    need = m1row.isna()
    m1row.loc[need] = cmd.loc[need, "Source_ID"].apply(by_biosample)

    # tier 3: cMD run -> ENA cache -> BioSample -> M1
    run2bs = load_run_to_biosample()

    def by_run_via_cache(acc):
        if pd.isna(acc) or not run2bs:
            return None
        for t in re.split(r"[;,]", str(acc)):
            b = run2bs.get(t.strip())
            if b is not None and b in bios:
                return bios[b]
        return None

    need = m1row.isna()
    if need.any() and "NCBI_accession" in cmd.columns:
        m1row.loc[need] = cmd.loc[need, "NCBI_accession"].apply(by_run_via_cache)
    cmd["m1_row"] = m1row
    joined = cmd[cmd["m1_row"].notna()].copy()
    joined["m1_row"] = joined["m1_row"].astype(int)
    print(f"joined {len(joined):,} ({100*len(joined)/len(cmd):.1f}%)")

    # ---- which flags could cMD even produce? ----------------------------
    vocab = set()
    for t in cmd["disease"].dropna():
        vocab |= {sanitize(d) for d in parse_diseases(t)}
    cmd_derivable = {c for c in dcols if c[len("disease_"):] in vocab}
    other_family = sorted(c for c in dcols if c not in cmd_derivable)
    print(f"disease_* resolvable from the cMD vocabulary : {len(cmd_derivable)}")
    print(f"disease_* NOT in the cMD vocabulary          : {len(other_family)}")

    # ---- per-sample comparison -----------------------------------------
    mi = m.loc[joined["m1_row"]].reset_index(drop=True)
    j = joined.reset_index(drop=True)
    expected = j["disease"].apply(lambda x: {sanitize(d) for d in parse_diseases(x)})
    flags_set, only_m1, only_cmd = [], [], []
    for k in range(len(j)):
        actual = {c[len("disease_"):] for c in dcols
                  if str(mi.at[k, c]).strip().lower() in {"1", "1.0", "y", "yes", "true", "t"}}
        exp = expected.iat[k]
        flags_set.append(";".join(sorted(actual)))
        only_m1.append(";".join(sorted(actual - exp)))
        only_cmd.append(";".join(sorted(exp - actual)))

    trail = pd.DataFrame({
        "cmd_file": j["cmd_file"],
        "cmd_study_name": j.get("study_name"),
        "NCBI_accession": j.get("NCBI_accession"),
        "Source_ID": j.get("Source_ID"),
        "M1_Chosen_match_ID": mi["Chosen_match_ID"],
        "M1_StudyFolder": mi.get("StudyFolder"),
        "M1_Study": mi.get("Study"),
        "cmd_disease": j["disease"],
        "cmd_study_condition": j.get("study_condition"),
        "cmd_disease_subtype": j.get("disease_subtype"),
        "cmd_disease_stage": j.get("disease_stage"),
        "M1_Unhealthy": mi.get("Unhealthy"),
        "M1_Lifestyle": mi.get("Lifestyle"),
        "M1_flags_set": flags_set,
        "flags_in_M1_not_in_cMD": only_m1,
        "flags_in_cMD_not_in_M1": only_cmd,
        "cmd_age": j.get("age"),
        "M1_Age": mi.get("Age"),
        "cmd_gender": j.get("gender"),
        "M1_Gender": mi.get("Gender"),
        "cmd_BMI": j.get("BMI"),
        "M1_BMI": mi.get("BMI"),
        "cmd_country": j.get("country"),
        "M1_Country": mi.get("Country"),
        "cmd_non_westernized": j.get("non_westernized"),
        "cmd_curator": j.get("curator"),
        "cmd_PMID": j.get("PMID"),
    })

    # every extra flag attributable to the other (da Silva) family?
    def classify(row):
        extra = [x for x in row["flags_in_M1_not_in_cMD"].split(";") if x]
        missing = [x for x in row["flags_in_cMD_not_in_M1"].split(";") if x]
        if not extra and not missing:
            return "flags agree with cMD"
        unexplained = [x for x in extra if f"disease_{x}" in cmd_derivable]
        bits = []
        if unexplained:
            bits.append("extra flag IS a cMD term but cMD does not say it here")
        elif extra:
            bits.append("extra flag not in cMD vocabulary (other family, e.g. da Silva)")
        if missing:
            bits.append("cMD names a condition M1 does not flag")
        return "; ".join(bits)

    trail["verdict"] = trail.apply(classify, axis=1)

    # health-label agreement, independent of the flags
    cond = trail["cmd_study_condition"].astype(str).str.strip().str.lower()
    dis = trail["cmd_disease"].astype(str).str.strip().str.lower()
    unh = trail["M1_Unhealthy"].astype(str).str.strip()
    cmd_healthy = cond.eq("control") & dis.isin(NULL_DISEASE)
    cmd_diseased = ~dis.isin(NULL_DISEASE)
    trail["health_label_check"] = "not comparable"
    trail.loc[cmd_healthy & unh.eq("Healthy"), "health_label_check"] = "agree (healthy)"
    trail.loc[cmd_diseased & unh.eq("Unhealthy"), "health_label_check"] = "agree (unhealthy)"
    trail.loc[cmd_healthy & unh.eq("Unhealthy"), "health_label_check"] = \
        "DISAGREE: cMD control/healthy, M1 Unhealthy"
    trail.loc[cmd_diseased & unh.eq("Healthy"), "health_label_check"] = \
        "DISAGREE: cMD names a disease, M1 Healthy"

    # ---- per-study rollup ----------------------------------------------
    by_study = (trail.assign(n=1)
                .groupby(["cmd_file", "M1_StudyFolder"], dropna=False)
                .agg(samples=("n", "sum"),
                     flags_agree=("verdict", lambda s: int((s == "flags agree with cMD").sum())),
                     label_disagree=("health_label_check",
                                     lambda s: int(s.astype(str).str.startswith("DISAGREE").sum())))
                .reset_index())
    by_study["flag_agreement_%"] = (100 * by_study["flags_agree"]
                                    / by_study["samples"]).round(1)
    by_study = by_study.sort_values("flag_agreement_%")

    disagree = trail[(trail["verdict"] != "flags agree with cMD")
                     | trail["health_label_check"].astype(str).str.startswith("DISAGREE")]
    unmatched = cmd[cmd["m1_row"].isna()][
        [c for c in ("cmd_file", "study_name", "disease", "study_condition",
                     "NCBI_accession", "Source_ID") if c in cmd.columns]]

    # M1 samples in cMD-covered studies that did not join
    covered_folders = set(trail["M1_StudyFolder"].dropna().astype(str))
    in_cov = m["StudyFolder"].astype(str).isin(covered_folders)
    joined_ids = set(trail["M1_Chosen_match_ID"].astype(str))
    m1_gap = m[in_cov & ~m["Chosen_match_ID"].astype(str).isin(joined_ids)][
        ["Chosen_match_ID", "StudyFolder", "Study", "Unhealthy", "Lifestyle"]]

    summary = pd.DataFrame({
        "Metadata health paper trail": [
            "source", "cMD study files", "cMD rows", "cMD rows joined to M1",
            "cMD rows not joined", "M1 rows", "M1 disease_* columns",
            "disease_* resolvable from cMD vocabulary",
            "disease_* from the other family (e.g. da Silva)",
            "samples whose flags agree with cMD",
            "samples with an extra flag that IS a cMD term",
            "samples where cMD names a condition M1 does not flag",
            "health-label disagreements",
            "M1 rows in covered studies that did not join",
            "", "caveat",
        ],
        "value": [
            str(CMD_DIR), cmd["cmd_file"].nunique(), len(cmd), len(trail),
            int(cmd["m1_row"].isna().sum()), len(m), len(dcols),
            len(cmd_derivable), len(other_family),
            int((trail["verdict"] == "flags agree with cMD").sum()),
            int(trail["verdict"].str.contains("IS a cMD term").sum()),
            int(trail["verdict"].str.contains("does not flag").sum()),
            int(trail["health_label_check"].astype(str).str.startswith("DISAGREE").sum()),
            len(m1_gap), "",
            "A flag cMD does not support is not automatically wrong: it may come "
            "from the da Silva et al. 2026 columns. Only rows whose verdict says "
            "the extra flag IS a cMD term lack any explanation from this source.",
        ],
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        trail.to_excel(xw, sheet_name="Paper trail", index=False)
        disagree.to_excel(xw, sheet_name="Disagreements", index=False)
        by_study.to_excel(xw, sheet_name="By study", index=False)
        pd.DataFrame({"disease_* not in cMD vocabulary": other_family}
                     ).to_excel(xw, sheet_name="Other-family flags", index=False)
        unmatched.to_excel(xw, sheet_name="cMD not in M1", index=False)
        m1_gap.to_excel(xw, sheet_name="M1 not in cMD", index=False)
    print(f"\nwrote {out}")
    print(f"  Paper trail      {len(trail):,} samples")
    print(f"  Disagreements    {len(disagree):,}")
    print(f"  flags agree      {int((trail['verdict']=='flags agree with cMD').sum()):,}")
    print(f"  label disagree   {int(trail['health_label_check'].astype(str).str.startswith('DISAGREE').sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
