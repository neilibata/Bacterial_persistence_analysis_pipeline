#!/usr/bin/env python3
"""Derive the read-volume cascade and multi-mapping rate from pipeline logs.

Every number reported for preprocessing should come from the logs the run
actually produced, not from a figure typed into a manuscript. This parses the
Trim Galore reports and the three bowtie2 stages into a per-sample table plus a
cohort summary, so the supplementary numbers regenerate from deposited code.

Handles both layouts:
  Snakemake     logs/02_trim/<s>.log, logs/02_phix/<s>.log,
                logs/02_human/<s>.log, logs/03_align/<s>.log
  original run  studies/*/trimmed_reads/paired/<run>_?_trimming_report.txt

    python tools/parse_preprocessing_logs.py --logs /path/to/logs
    python tools/parse_preprocessing_logs.py --trim-glob \\
        '/n/netscratch/.../modular_pipeline/studies/*/trimmed_reads/paired/*_trimming_report.txt'

Coverage is reported explicitly: a cohort statistic drawn from a subset of
surviving logs is only meaningful alongside the number of samples it rests on.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

import pandas as pd

# ── Trim Galore / cutadapt ────────────────────────────────────────────────
RE_BP_IN = re.compile(r"Total basepairs processed:\s*([\d,]+)\s*bp")
RE_BP_OUT = re.compile(r"Total written \(filtered\):\s*([\d,]+)\s*bp")
RE_READS_IN = re.compile(r"Total reads processed:\s*([\d,]+)")

# ── bowtie2 summary (identical shape for phix / human / metagenome) ───────
RE_TOTAL = re.compile(r"^\s*([\d,]+)\s+reads; of these:", re.M)
RE_CONC_0 = re.compile(r"([\d,]+) \([\d.]+%\) aligned concordantly 0 times")
RE_CONC_1 = re.compile(r"([\d,]+) \([\d.]+%\) aligned concordantly exactly 1 time")
RE_CONC_M = re.compile(r"([\d,]+) \([\d.]+%\) aligned concordantly >1 times")
RE_RATE = re.compile(r"([\d.]+)% overall alignment rate")


def num(s: str | None) -> int | None:
    return int(s.replace(",", "")) if s else None


def first(rx: re.Pattern, text: str):
    m = rx.search(text)
    return m.group(1) if m else None


def parse_trim(path: Path) -> dict:
    t = path.read_text(errors="replace")
    return {"trim_bp_in": num(first(RE_BP_IN, t)),
            "trim_bp_out": num(first(RE_BP_OUT, t)),
            "trim_reads_in": num(first(RE_READS_IN, t))}


def parse_bt2(path: Path) -> dict:
    t = path.read_text(errors="replace")
    return {"total_pairs": num(first(RE_TOTAL, t)),
            "conc_0": num(first(RE_CONC_0, t)),
            "conc_1": num(first(RE_CONC_1, t)),
            "conc_multi": num(first(RE_CONC_M, t)),
            "rate": float(first(RE_RATE, t)) if first(RE_RATE, t) else None}


def collect(logs: Path) -> pd.DataFrame:
    rows: dict[str, dict] = {}

    def slot(sample: str) -> dict:
        return rows.setdefault(sample, {"sample": sample})

    for sub, fn, prefix in (("02_trim", parse_trim, ""),
                            ("02_phix", parse_bt2, "phix_"),
                            ("02_human", parse_bt2, "human_"),
                            ("03_align", parse_bt2, "align_")):
        d = logs / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.log")):
            s = p.stem
            try:
                got = fn(p)
            except Exception as exc:                       # noqa: BLE001
                print(f"  warn: {p}: {exc}", file=sys.stderr)
                continue
            slot(s).update({f"{prefix}{k}": v for k, v in got.items()})
    return pd.DataFrame(rows.values())


def collect_trim_glob(pattern: str) -> pd.DataFrame:
    """Trim Galore reports from the original staged execution.

    Named per run and per mate, so mates are summed and the run is reported
    under its own accession -- the sample mapping lives in metadata M1.
    """
    rows: dict[str, dict] = {}
    for f in glob.glob(pattern):
        p = Path(f)
        run = re.sub(r"_[12]\.fastq\.gz_trimming_report\.txt$", "", p.name)
        run = re.sub(r"\.fastq\.gz_trimming_report\.txt$", "", run)
        got = parse_trim(p)
        r = rows.setdefault(run, {"sample": run, "trim_bp_in": 0,
                                  "trim_bp_out": 0, "trim_reads_in": 0})
        for k in ("trim_bp_in", "trim_bp_out", "trim_reads_in"):
            if got.get(k):
                r[k] += got[k]
    return pd.DataFrame(rows.values())


def summarise(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n  samples with any log : {n:,}")

    def tot(c):
        return df[c].dropna().sum() if c in df else 0

    have_trim = df["trim_bp_in"].notna().sum() if "trim_bp_in" in df else 0
    if have_trim:
        bi, bo = tot("trim_bp_in"), tot("trim_bp_out")
        print(f"\n  TRIMMING            ({have_trim:,} samples)")
        print(f"    bases in          : {bi/1e12:.2f} Tbp")
        print(f"    bases surviving   : {bo/1e12:.2f} Tbp  ({bo/bi*100:.1f}%)")

    for stage, label in (("phix", "PHIX REMOVAL"), ("human", "HOST REMOVAL")):
        c0, ct = f"{stage}_conc_0", f"{stage}_total_pairs"
        if c0 not in df or df[c0].notna().sum() == 0:
            continue
        k = df[c0].notna().sum()
        kept, total = tot(c0), tot(ct)
        print(f"\n  {label}       ({k:,} samples)")
        print(f"    pairs in          : {total/1e9:.2f} G")
        print(f"    pairs retained    : {kept/1e9:.2f} G  ({kept/total*100:.1f}%)")

    if "align_conc_1" in df and df["align_conc_1"].notna().sum():
        k = df["align_conc_1"].notna().sum()
        u, m = tot("align_conc_1"), tot("align_conc_multi")
        rate = df["align_rate"].dropna()
        print(f"\n  ALIGNMENT           ({k:,} samples)")
        print(f"    pairs in          : {tot('align_total_pairs')/1e9:.2f} G")
        print(f"    concordant unique : {u/1e9:.3f} G")
        print(f"    concordant multi  : {m/1e9:.3f} G")
        if u + m:
            print(f"    MULTI-MAPPING     : {m/(u+m)*100:.1f}% of concordant pairs")
        if len(rate):
            print(f"    overall align rate: {rate.mean():.1f}% "
                  f"(median {rate.median():.1f}, n={len(rate):,})")
        print("\n    Note: pairs aligning concordantly >1 time are removed by the\n"
              "    MAPQ>=30 filter. This is the fraction that quantifies it.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path,
                    help="Snakemake logs/ directory (02_trim, 03_align, ...)")
    ap.add_argument("--trim-glob",
                    help="glob for original-run *_trimming_report.txt files")
    ap.add_argument("--out", type=Path,
                    default=Path("results/preprocessing_log_summary.csv"))
    args = ap.parse_args()
    if not args.logs and not args.trim_glob:
        ap.error("give --logs and/or --trim-glob")

    frames = []
    if args.logs:
        frames.append(collect(args.logs))
    if args.trim_glob:
        frames.append(collect_trim_glob(args.trim_glob))
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) \
        if any(len(f) for f in frames) else pd.DataFrame()
    if df.empty:
        sys.exit("no logs parsed -- check the paths")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    summarise(df)
    print(f"\n  wrote {args.out}  ({len(df):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
