#!/usr/bin/env python3
"""Rebuild the combined (5,081-SGB) matrices from the UHGG and Leviatan ones.

The combined reference is not a separate alignment. Each of the 5,081
non-redundant representatives keeps the values from whichever source catalogue
won the representative choice during ANI clustering, so the combined matrix is a
column-wise merge and rebuilding it costs minutes rather than a realignment.
The clustering result itself is frozen in `combined_sgb_metadata.xlsx`
(`Combined` sheet: `representative`, `rep_catalog`).

**Sample set is the intersection, not the union.** The original construction
took the union, so a sample present in UHGG but absent from Leviatan was carried
in with all 2,198 Leviatan-derived columns set to zero — indistinguishable from
a genuinely absent taxon, and systematically depressing that sample's richness
across 43% of the catalogue. 398 samples were affected. A sample lacking data in
either source is now excluded and reported.

    # verify the rebuild reproduces the published matrix, then
    python tools/build_combined_matrix.py --validate
    python tools/build_combined_matrix.py --out-suffix 20260812

Validation is the point: run against the current inputs, the builder must
reproduce the published combined matrix exactly for every sample present in
both sources. Only then is it trustworthy for the backfilled samples.
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

WORKBOOK = ROOT / "combined_sgb_metadata.xlsx"


def spec() -> pd.DataFrame:
    """representative -> source catalogue, from the frozen clustering result."""
    cmb = pd.read_excel(WORKBOOK, sheet_name="Combined")
    cmb["representative"] = cmb["representative"].astype(str).str.strip()
    cmb["rep_catalog"] = cmb["rep_catalog"].astype(str).str.strip().str.lower()
    bad = set(cmb["rep_catalog"]) - {"uhgg", "leviatan"}
    if bad:
        sys.exit(f"unexpected rep_catalog values: {bad}")
    return cmb[["representative", "rep_catalog"]]


def index_of(path: Path) -> list[str]:
    ids: list[str] = []
    for chunk in pd.read_csv(path, usecols=[0], chunksize=200_000, dtype=str):
        ids.extend(chunk.iloc[:, 0].astype(str).str.strip())
    return ids


def load_cols(path: Path, cols: list[str], keep: set[str]) -> pd.DataFrame:
    """Read only the needed columns, only the needed rows."""
    first = pd.read_csv(path, nrows=0).columns[0]
    frames = []
    for chunk in pd.read_csv(path, usecols=[first] + cols, index_col=0,
                             chunksize=4000, dtype={first: str}):
        chunk.index = chunk.index.astype(str).str.strip()
        hit = chunk[chunk.index.isin(keep)]
        if len(hit):
            frames.append(hit.astype(np.float64))
    if not frames:
        sys.exit(f"no requested samples found in {path.name}")
    return pd.concat(frames)


def source_paths(kind: str, override: dict) -> tuple[Path, Path]:
    """Where the UHGG and Leviatan values come from for this build.

    Defaults to config.REFERENCES, but accepts explicit overrides so a rebuilt
    pair can be combined without repointing the whole pipeline at them. The
    paths used are printed, because a combined matrix silently built from the
    wrong generation of inputs is indistinguishable from a correct one.
    """
    key = {"depth": "depth_file", "breadth": "breadth_file"}[kind]
    uh = Path(override.get(f"uhgg_{kind}") or ROOT / config.REFERENCES["UHGG"][key])
    lv = Path(override.get(f"lev_{kind}") or ROOT / config.REFERENCES["lev"][key])
    return uh, lv


def build(kind: str, order: list[str], sp: pd.DataFrame,
          keep: set[str], override: dict = None) -> pd.DataFrame:
    uh, lv = source_paths(kind, override or {})
    print(f"  from UHGG: {uh.name}")
    print(f"  from lev : {lv.name}")

    src = dict(zip(sp["representative"], sp["rep_catalog"]))
    u_cols = [c for c in order if src.get(c.rsplit("_", 1)[0]) == "uhgg"]
    l_cols = [c for c in order if src.get(c.rsplit("_", 1)[0]) == "leviatan"]
    print(f"  {kind}: {len(u_cols)} UHGG-derived, {len(l_cols)} Leviatan-derived")

    u = load_cols(uh, u_cols, keep)
    l = load_cols(lv, l_cols, keep)
    common = sorted(set(u.index) & set(l.index))
    out = pd.concat([u.loc[common], l.loc[common]], axis=1)[order]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="rebuild and compare to the published combined matrix")
    ap.add_argument("--out-suffix", default=None,
                    help="write combined_{depth,coverage}_matrix_<suffix>.csv")
    ap.add_argument("--limit", type=int, default=0,
                    help="validate on the first N shared samples only")
    for a in ("uhgg-depth", "uhgg-breadth", "lev-depth", "lev-breadth"):
        ap.add_argument(f"--{a}", default=None,
                        help=f"override the {a.replace('-', ' ')} input")
    args = ap.parse_args()
    if not args.validate and not args.out_suffix:
        ap.error("give --validate and/or --out-suffix")

    sp = spec()
    print(f"  specification: {len(sp):,} representatives "
          f"({(sp.rep_catalog=='uhgg').sum():,} UHGG, "
          f"{(sp.rep_catalog=='leviatan').sum():,} Leviatan)")

    published = ROOT / config.REFERENCES["combined"]["depth_file"]
    order = pd.read_csv(published, nrows=0).columns.tolist()[1:]
    print(f"  column order taken from {published.name}: {len(order):,}")

    override = {"uhgg_depth": args.uhgg_depth, "uhgg_breadth": args.uhgg_breadth,
                "lev_depth": args.lev_depth, "lev_breadth": args.lev_breadth}
    uh_ids = set(index_of(source_paths("depth", override)[0]))
    lv_ids = set(index_of(source_paths("depth", override)[1]))
    keep = uh_ids & lv_ids
    print(f"\n  UHGG samples      : {len(uh_ids):,}")
    print(f"  Leviatan samples  : {len(lv_ids):,}")
    print(f"  INTERSECTION      : {len(keep):,}")
    print(f"  dropped (UHGG-only): {len(uh_ids - lv_ids):,}   "
          f"(these were previously zero-filled)")
    if args.limit:
        keep = set(sorted(keep)[:args.limit])
        print(f"  limited to        : {len(keep):,}")

    for kind, stem in (("depth", "combined_depth_matrix"),
                       ("breadth", "combined_coverage_matrix")):
        print(f"\n{kind.upper()}")
        out = build(kind, order, sp, keep, override)
        print(f"  built {out.shape[0]:,} samples x {out.shape[1]:,} columns")

        if args.validate and kind == "depth":
            print("  validating against the published matrix ...")
            want = set(out.index)
            frames = []
            for chunk in pd.read_csv(published, index_col=0, chunksize=4000):
                chunk.index = chunk.index.astype(str).str.strip()
                hit = chunk[chunk.index.isin(want)]
                if len(hit):
                    frames.append(hit.astype(np.float64))
            pub = pd.concat(frames).loc[out.index, order]
            a, b = pub.to_numpy(), out.to_numpy()
            same = np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
            diff = np.nanmax(np.abs(a - b))
            print(f"    samples compared : {len(pub):,}")
            print(f"    identical        : {same}")
            print(f"    max abs diff     : {diff:.6g}")
            if not same:
                bad = np.argwhere(~np.isclose(a, b, rtol=1e-9, atol=1e-12))
                print(f"    differing cells  : {len(bad):,}")
                for i, j in bad[:5]:
                    print(f"      {pub.index[i]} / {order[j]}: "
                          f"published={a[i, j]} rebuilt={b[i, j]}")
                return 1

        if args.out_suffix:
            p = ROOT / f"{stem}_{args.out_suffix}.csv"
            out.to_csv(p)
            print(f"  wrote {p.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
