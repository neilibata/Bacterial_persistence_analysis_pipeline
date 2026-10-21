#!/usr/bin/env python3
"""Find samples whose mapped bases are implausibly low for their input volume.

`SRR061226` reached the concordance subset by chance and turned out to have been
processed from one of its two sequencing runs, recovering half the sequence it
should have. That defect is invisible to presence checks -- the sample has a row,
the row has values, nothing errors -- and only shows up as an anomalously low
ratio of mapped bases to sequencing input.

This computes that ratio for every sample against every reference and flags the
low tail. `Total bases` in Metadata Table M1 is the input volume (validated
against the surviving Trim Galore reports to a median ratio of 1.000), and
mapped bases are Sum_SGB(mean_depth x genome_length) -- the same quantity
`loading.py` calls `total_bp`.

    python tools/mapped_fraction_outliers.py
    python tools/mapped_fraction_outliers.py --refs UHGG --z -3

Flagging is per study, not globally: mapping rate varies with read length,
sequencing platform and host contamination, so a global threshold would flag
whole studies rather than individual failures.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence import config  # noqa: E402

OUT = ROOT / "results" / "qc"
CLASSIFIED = ("H-G", "Transitional", "Industrial")


def lengths_of(ref: str) -> pd.Series:
    d = {}
    with open(ROOT / config.REFERENCES[ref]["lengths_file"]) as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 2:
                try:
                    d[f[0]] = float(f[1])
                except ValueError:
                    pass
    return pd.Series(d, dtype=float)


def mapped_bases(ref: str, lens: pd.Series) -> pd.Series:
    """Sum_SGB(mean_depth x genome_length), streamed in chunks."""
    path = ROOT / config.REFERENCES[ref]["depth_file"]
    head = pd.read_csv(path, nrows=0).columns.tolist()
    mean_cols = [c for c in head[1:] if c.endswith("_mean")]
    sgbs = [c[:-5] for c in mean_cols]
    missing = [s for s in sgbs if s not in lens.index]
    if missing:
        sys.exit(f"{ref}: {len(missing)} matrix SGBs absent from the lengths "
                 f"file (e.g. {missing[:3]})")
    w = lens.reindex(sgbs).to_numpy(float)
    out = {}
    for chunk in pd.read_csv(path, index_col=0, chunksize=2000,
                             usecols=[head[0]] + mean_cols):
        chunk.index = chunk.index.astype(str).str.strip()
        vals = chunk.to_numpy(float, na_value=0.0)
        for sid, bp in zip(chunk.index, vals @ w):
            out[sid] = bp
    return pd.Series(out, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", nargs="+", default=["UHGG", "lev", "combined"])
    ap.add_argument("--z", type=float, default=-3.0,
                    help="flag samples this many robust SDs below their study median")
    ap.add_argument("--min-study", type=int, default=10,
                    help="minimum samples for a study to be assessed")
    ap.add_argument("--multirun", action="store_true",
                    help="test multi-run samples for sub-multiple mapped fractions")
    ap.add_argument("--tol", type=float, default=0.12,
                    help="relative tolerance when matching a k/N sub-multiple")
    args = ap.parse_args()

    import re
    RUN = re.compile(r"[SEDR]RR\d+")
    m1 = pd.read_csv(ROOT / "metadata_M1.csv", low_memory=False, dtype=str)
    m1 = m1.drop_duplicates("Chosen_match_ID").copy()
    m1["_key"] = m1["Chosen_match_ID"].astype(str).str.strip()
    m1["_tb"] = pd.to_numeric(m1["Total bases"], errors="coerce")
    m1["_nruns"] = m1["Paired SRR List"].apply(
        lambda v: 0 if pd.isna(v) else len(set(RUN.findall(str(v)))))
    lab = m1["Lifestyle"].astype(str).str.strip()
    meta = m1[lab.isin(CLASSIFIED)].set_index("_key")

    OUT.mkdir(parents=True, exist_ok=True)
    all_flags = []
    for ref in args.refs:
        print(f"\n=== {ref} ===", flush=True)
        lens = lengths_of(ref)
        bp = mapped_bases(ref, lens)
        df = meta.join(bp.rename("mapped_bp"), how="inner")
        df = df[df["_tb"] > 0]
        df["frac"] = df["mapped_bp"] / df["_tb"]
        print(f"  samples assessed : {len(df):,}")
        print(f"  mapped/input     : median {df.frac.median():.4f}   "
              f"IQR {df.frac.quantile(.25):.4f}-{df.frac.quantile(.75):.4f}")

        # Robust per-study z: (x - median) / (1.4826 * MAD)
        g = df.groupby("StudyFolder")["frac"]
        med = g.transform("median")
        mad = g.transform(lambda s: np.median(np.abs(s - np.median(s))))
        n = g.transform("size")
        scale = (1.4826 * mad).replace(0, np.nan)
        df["study_median"] = med
        df["robust_z"] = (df["frac"] - med) / scale
        if args.multirun:
            # Partial processing of a multi-run sample leaves a specific
            # signature: the sample's mapped fraction falls to roughly k/N of
            # what a correctly processed sample in the same study achieves,
            # where N is its run count and k the number actually used. A
            # generic "low relative to study" rule cannot see this -- it misses
            # SRR061226, which used 1 of 2 runs and therefore sits at ~0.5x,
            # well inside normal study spread -- but a sub-multiple test is
            # specific to the mechanism.
            #
            # Baseline is single-run samples in the same study where enough
            # exist, since those cannot have this defect by construction.
            single = df[df["_nruns"] <= 1]
            base_single = single.groupby("StudyFolder")["frac"].median()
            base_all = df.groupby("StudyFolder")["frac"].median()
            counts = single.groupby("StudyFolder").size()
            baseline = base_single.where(counts >= 5, base_all)
            multi = df[df["_nruns"] >= 2].copy()
            multi["baseline"] = multi["StudyFolder"].map(baseline)
            multi = multi[multi["baseline"] > 0]
            multi["ratio"] = multi["frac"] / multi["baseline"]

            def submultiple(row):
                N = int(row["_nruns"])
                best = None
                for k in range(1, N):
                    tgt = k / N
                    if abs(row["ratio"] - tgt) <= args.tol * tgt:
                        if best is None or tgt > best[1]:
                            best = (k, tgt)
                return f"{best[0]}/{N}" if best else ""

            multi["submultiple"] = multi.apply(submultiple, axis=1)
            hit = multi[(multi.submultiple != "") & (multi.ratio < 0.9)]
            print(f"  multi-run samples : {len(multi):,}")
            print(f"  SUB-MULTIPLE hits : {len(hit):,}  "
                  f"(ratio matches k/N within {args.tol:.0%})")
            if len(hit):
                sh = hit.nsmallest(10, "ratio")
                print(f"    {'sample':<20}{'runs':>5}{'ratio':>8}{'k/N':>7}  study")
                for sid, r in sh.iterrows():
                    print(f"    {sid[:19]:<20}{int(r._nruns):>5}{r.ratio:>8.3f}"
                          f"{r.submultiple:>7}  {str(r.StudyFolder)[:26]}")
            hit_out = hit.copy(); hit_out["reference"] = ref
            all_flags.append(hit_out)
            continue

        flag = df[(df.robust_z <= args.z) & (n >= args.min_study)].copy()
        flag["reference"] = ref
        print(f"  FLAGGED (z <= {args.z}) : {len(flag):,}")
        if len(flag):
            cols = ["StudyFolder", "Lifestyle", "_tb", "mapped_bp", "frac",
                    "study_median", "robust_z"]
            show = flag.nsmallest(8, "robust_z")[cols]
            print(f"    {'sample':<20}{'study':<26}{'frac':>8}{'study med':>11}{'z':>8}")
            for sid, r in show.iterrows():
                print(f"    {sid[:19]:<20}{str(r.StudyFolder)[:25]:<26}"
                      f"{r.frac:>8.4f}{r.study_median:>11.4f}{r.robust_z:>8.1f}")
        all_flags.append(flag)

    if all_flags:
        allf = pd.concat(all_flags)
        p = OUT / "mapped_fraction_outliers.csv"
        allf.reset_index().rename(columns={"index": "Chosen_match_ID"}).to_csv(p, index=False)
        n_any = allf.reset_index()["index"].nunique() if "index" in allf.reset_index() else allf.index.nunique()
        print(f"\n  flagged in >=1 reference : {n_any:,}")
        # A genuine processing failure shows up against every catalogue; a
        # single-reference flag is more likely a reference-specific mapping quirk.
        counts = allf.groupby(allf.index).size()
        print(f"  flagged in ALL {len(args.refs)} references : "
              f"{int((counts == len(args.refs)).sum()):,}  <-- most likely genuine partial processing")
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
