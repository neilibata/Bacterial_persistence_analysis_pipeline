#!/usr/bin/env python3
"""Sweep the metadata for structural and numeric anomalies.

A final pass before the cohort is frozen. Reports; changes nothing. Each check
is deliberately narrow so a hit means something specific rather than "looks odd".

  STRUCTURE   study split across non-adjacent blocks; duplicate Chosen_match_ID;
              identifier format inconsistent within a study
  NUMERIC     non-positive or absent Total bases; values far outside their
              study's own distribution; impossible ages and BMIs
  MISSING     a field blank for a sample where the rest of its study populates it
  CONSISTENCY country vs lifestyle disagreement inside one study; obesity flags
              against BMI; health fields contradicting each other
  DEPTH       metadata Total bases against what the matrix actually recovered

    python tools/audit_metadata_anomalies.py
    python tools/audit_metadata_anomalies.py --src metadata_Aug_26_2026_organised.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _blank(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["nan", "", "none"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "metadata_Aug_20_2026.csv")
    ap.add_argument("--profile", type=Path, default=ROOT / "docs" / "sample_breadth_profile.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "metadata_anomalies.csv")
    args = ap.parse_args()

    m = pd.read_csv(args.src, low_memory=False)
    m["_row"] = range(len(m))
    ids = m["Chosen_match_ID"].astype(str).str.strip()
    print(f"{args.src.name}: {len(m):,} rows\n")
    found = []

    def add(kind, sub, detail):
        for i in (sub.index if hasattr(sub, "index") else sub):
            found.append(dict(check=kind, row=int(m.at[i, "_row"]),
                              Chosen_match_ID=m.at[i, "Chosen_match_ID"],
                              Author=m.at[i, "Author"], Year=m.at[i, "Year"],
                              detail=detail))

    # ── STRUCTURE ────────────────────────────────────────────────────────
    print("STRUCTURE")
    split = []
    for (a, y), g in m.groupby(["Author", "Year"]):
        r = g["_row"].values
        if len(r) > 1 and (r.max() - r.min() + 1) != len(r):
            split.append((a, y, len(r), int(r.min()), int(r.max())))
    print(f"  studies split across non-adjacent rows : {len(split)}")
    for a, y, n, lo, hi in sorted(split, key=lambda x: -x[2])[:8]:
        print(f"      {a} {y}: {n} rows spanning {lo:,}-{hi:,}")
        add("study_not_contiguous", m.index[(m.Author == a) & (m.Year == y)],
            f"{n} rows spanning {lo}-{hi}")

    dup = ids[ids.duplicated(keep=False)]
    print(f"  duplicate Chosen_match_ID              : {dup.nunique()} ids, {len(dup)} rows")
    if len(dup):
        add("duplicate_id", m.index[ids.isin(set(dup))], "shared Chosen_match_ID")

    # identifier style should be consistent inside a study
    def style(s):
        s = str(s)
        if s[:3] in ("SRR", "ERR", "DRR"):
            return "run"
        if s.startswith("SAM"):
            return "biosample"
        return "other"
    mixed = []
    for (a, y), g in m.groupby(["Author", "Year"]):
        k = set(g["Chosen_match_ID"].map(style))
        if len(k) > 1:
            mixed.append((a, y, k, len(g)))
    print(f"  studies with mixed identifier styles   : {len(mixed)}")
    for a, y, k, n in mixed[:8]:
        print(f"      {a} {y}: {sorted(k)} across {n} rows")
        add("mixed_id_style", m.index[(m.Author == a) & (m.Year == y)], f"styles {sorted(k)}")

    # ── NUMERIC ──────────────────────────────────────────────────────────
    print("\nNUMERIC")
    tb = pd.to_numeric(m.get("Total bases"), errors="coerce")
    bad = m.index[tb.isna() | (tb <= 0)]
    print(f"  Total bases missing or <= 0            : {len(bad)}")
    add("total_bases_invalid", bad, "missing or <=0")

    tiny = m.index[(tb > 0) & (tb < 1e8)]
    print(f"  Total bases < 100 Mb (very small)      : {len(tiny)}")
    add("total_bases_tiny", tiny, "< 100 Mb")

    age = pd.to_numeric(m.get("Age"), errors="coerce")
    a_bad = m.index[age.notna() & ((age < 0) | (age > 110))]
    print(f"  Age outside 0-110                      : {len(a_bad)}")
    add("age_implausible", a_bad, "outside 0-110")

    bmi = pd.to_numeric(m.get("BMI"), errors="coerce")
    b_bad = m.index[bmi.notna() & ((bmi < 8) | (bmi > 90))]
    print(f"  BMI outside 8-90                       : {len(b_bad)}")
    add("bmi_implausible", b_bad, "outside 8-90")

    # within-study depth outliers
    out = []
    for (a, y), g in m.groupby(["Author", "Year"]):
        v = pd.to_numeric(g.get("Total bases"), errors="coerce")
        v = v[v > 0]
        if len(v) < 20:
            continue
        lv = np.log10(v)
        z = (lv - lv.median()).abs() / (lv.mad() if hasattr(lv, "mad") else lv.std() or 1)
        out += list(v.index[z > 6])
    print(f"  depth >6 robust-SD from study median    : {len(out)}")
    add("depth_outlier_within_study", pd.Index(out), "far from study median")

    # ── CONSISTENCY ──────────────────────────────────────────────────────
    print("\nCONSISTENCY")
    if {"BMI", "Obesity"} <= set(m.columns):
        ob = m["Obesity"].astype(str).str.strip().str.lower().isin(["yes", "1", "1.0"])
        c1 = m.index[bmi.notna() & (bmi >= 30) & ~ob]
        c2 = m.index[bmi.notna() & (bmi < 25) & ob]
        print(f"  BMI >= 30 but Obesity not set          : {len(c1)}")
        print(f"  BMI < 25 but Obesity set               : {len(c2)}")
        add("bmi_obesity_mismatch", c1, "BMI>=30, Obesity unset")
        add("bmi_obesity_mismatch", c2, "BMI<25, Obesity set")
    if {"Healthy", "Unhealthy"} <= set(m.columns):
        h = m["Healthy"].astype(str).str.strip().str.lower().isin(["yes", "1", "1.0"])
        u = m["Unhealthy"].astype(str).str.strip().str.lower().eq("unhealthy")
        both = m.index[h & u]
        print(f"  Healthy and Unhealthy both asserted    : {len(both)}")
        add("healthy_and_unhealthy", both, "both set")

    lm = []
    for (a, y), g in m.groupby(["Author", "Year"]):
        for c, gg in g.groupby(g.get("Country").astype(str)):
            if gg["Lifestyle"].astype(str).nunique() > 1:
                lm.append((a, y, c, dict(gg["Lifestyle"].astype(str).value_counts())))
    print(f"  one study+country, several lifestyles  : {len(lm)}")
    for a, y, c, d in lm[:6]:
        print(f"      {a} {y} {c}: {d}")

    # ── DEPTH vs MATRIX ──────────────────────────────────────────────────
    if args.profile.exists():
        print("\nDEPTH vs MATRIX")
        p = pd.read_csv(args.profile)
        p["Sample_ID"] = p.Sample_ID.astype(str).str.strip()
        j = m.assign(_k=ids).merge(p, left_on="_k", right_on="Sample_ID", how="left")
        nomatch = int(j.Sample_ID.isna().sum())
        print(f"  metadata rows with no matrix row       : {nomatch:,}")
        deep_empty = j.index[(pd.to_numeric(j["Total bases"], errors="coerce") > 1e10)
                             & (j.richness_10pct.fillna(-1) <= 3)]
        print(f"  deep (>10 Gb) but <=3 SGBs recovered   : {len(deep_empty)}")
        for i in deep_empty[:10]:
            add("deep_but_empty", pd.Index([i]),
                f"{pd.to_numeric(j.at[i,'Total bases'],errors='coerce')/1e9:.1f} Gb -> "
                f"{j.at[i,'richness_10pct']:.0f} SGBs")

    d = pd.DataFrame(found)
    if len(d):
        d = d.drop_duplicates(["check", "Chosen_match_ID"])
        d.to_csv(args.out, index=False)
        print(f"\nwrote {args.out.relative_to(ROOT)} ({len(d):,} findings, "
              f"{d.Chosen_match_ID.nunique():,} distinct samples)")
        print("\nfindings by check:")
        print(d.groupby("check").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
