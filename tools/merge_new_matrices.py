#!/usr/bin/env python3
"""Fold reprocessed and newly processed samples into the full coverage matrices.

The old matrices are 2.4 GB and hold 23,936 rows; this run touches 447 of them.
So rather than load, concatenate and rewrite the whole thing -- which would push
every untouched value through a float parse and reformat, and silently change
things like trailing-zero formatting on 190 million cells -- the old file is
streamed line by line and any row not being replaced is copied through
BYTE-IDENTICAL. Only the 187 replaced rows and the 260 appended ones are
rendered from the new data.

Two matrices come out of this per reference, coverage and depth. Neither input
is modified; output goes to new dated filenames.

Column alignment is by NAME, never position. Both families are built as
`Sample_ID, {species}_mean, {species}_std`, but species order comes from a
contig map that is regenerated per run, so equal column counts prove nothing.
A species in the old header but missing from the new one is written as 0.0 and
reported; a species in the new header but not the old is dropped and reported.
Either being non-empty means the catalogues have diverged and the merge should
be looked at rather than trusted.

Scale is checked, not assumed. The 187 replaced rows describe the same samples
before and after, so their old and new values must be on the same scale; the
report prints the median new/old ratio over cells nonzero in both. Breadth is a
percentage in the published matrices (0-100), and a new matrix carrying
fractions would show up here as a ratio near 0.01 rather than as an error.
Reprocessed samples gained runs, so a ratio slightly above 1 is expected.

    python tools/merge_new_matrices.py --ref UHGG \
        --new-coverage <dir>/coverage_matrix_20260820.csv \
        --new-depth    <dir>/depth_matrix_20260820.csv \
        --suffix 20260820
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence import config as cfg              # noqa: E402

csv.field_size_limit(10_000_000)


def clean_stem(stem: str) -> str:
    """Strip build-history suffixes so the output carries one date, not two.

    These files are the Zenodo deposit, so the name is part of the artefact.
    The current inputs are `..._UHGG_merged_dedup_firstcol` and
    `..._lev_20260812`; naively appending would give `..._lev_20260812_20260820`,
    which reads as though the contents were from August 12.
    """
    for tail in ("_merged_dedup_firstcol", "_merged_dedup", "_merged", "_firstcol"):
        if stem.endswith(tail):
            stem = stem[: -len(tail)]
    if len(stem) > 9 and stem[-9] == "_" and stem[-8:].isdigit():
        stem = stem[:-9]                       # a trailing _YYYYMMDD
    return stem


def load_new(path: Path):
    """Read a new matrix fully: {sample_id: {column: value}} plus its header."""
    with path.open(newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        rows = {}
        for row in rd:
            rows[row[0]] = row
    return header, rows


def merge_one(old_path: Path, new_path: Path, out_path: Path, excluded: set,
              label: str):
    new_header, new_rows = load_new(new_path)
    new_pos = {c: i for i, c in enumerate(new_header)}

    with old_path.open(newline="") as fh:
        old_header = next(csv.reader(fh))

    missing = [c for c in old_header[1:] if c not in new_pos]
    extra = [c for c in new_header[1:] if c not in set(old_header)]
    # index into each new row, in OLD column order; None -> fill 0.0
    take = [new_pos.get(c) for c in old_header[1:]]

    keep = {s: r for s, r in new_rows.items() if s not in excluded}
    skipped = sorted(set(new_rows) - set(keep))

    def render(sample):
        r = keep[sample]
        return [sample] + [(r[i] if i is not None else "0.0") for i in take]

    replaced, written = set(), 0
    ratios = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with old_path.open(newline="") as fin, out_path.open("w", newline="") as fout:
        first = fin.readline()
        fout.write(first)                                   # header, untouched
        wtr = csv.writer(fout, lineterminator="\n")
        for line in fin:
            sid = line.split(",", 1)[0]
            if sid in keep:
                old_vals = next(csv.reader([line]))
                new_vals = render(sid)
                for a, b in zip(old_vals[1:], new_vals[1:]):
                    try:
                        fa, fb = float(a), float(b)
                    except ValueError:
                        continue
                    if fa > 0 and fb > 0:
                        ratios.append(fb / fa)
                wtr.writerow(new_vals)
                replaced.add(sid)
            else:
                fout.write(line)                            # byte-identical
            written += 1
        appended = sorted(set(keep) - replaced)
        for s in appended:
            wtr.writerow(render(s))

    ratios.sort()
    med = ratios[len(ratios) // 2] if ratios else float("nan")
    print(f"\n  {label}")
    print(f"    old rows                {written:>8,}")
    print(f"    replaced                {len(replaced):>8,}")
    print(f"    appended                {len(appended):>8,}")
    print(f"    excluded (not written)  {len(skipped):>8,}")
    print(f"    new total               {written + len(appended):>8,}")
    print(f"    columns: old {len(old_header):,}  new {len(new_header):,}  "
          f"missing-from-new {len(missing)}  extra-in-new {len(extra)}")
    if missing:
        print(f"      !! filled with 0.0: {missing[:5]}")
    if extra:
        print(f"      !! dropped: {extra[:5]}")
    print(f"    scale check: median new/old over {len(ratios):,} cells "
          f"nonzero in both = {med:.3f}")
    if ratios and not (0.2 < med < 20):
        print("      !! ratio far from 1 -- the two matrices are probably not "
              "on the same scale. Do NOT use this output until resolved.")
    return written + len(appended), missing, extra, med


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, choices=["UHGG", "lev", "chimp"])
    ap.add_argument("--new-coverage", type=Path, required=True)
    ap.add_argument("--new-depth", type=Path, required=True)
    ap.add_argument("--suffix", required=True, help="date stamp for the outputs")
    ap.add_argument("--exclude", type=Path,
                    default=ROOT / "tools" / "excluded_reprocessing_samples.tsv")
    ap.add_argument("--out-dir", type=Path, default=ROOT)
    args = ap.parse_args()

    excluded = set()
    if args.exclude.exists():
        for line in args.exclude.read_text().splitlines():
            if line and not line.startswith("#"):
                excluded.add(line.split("\t")[0])
    print(f"  exclusions loaded: {len(excluded)}")

    spec = cfg.REFERENCES[args.ref]
    jobs = [("coverage/breadth", Path(spec["breadth_file"]), args.new_coverage),
            ("depth",            Path(spec["depth_file"]),   args.new_depth)]

    results = []
    for label, old, new in jobs:
        if not old.exists():
            sys.exit(f"missing old matrix: {old}")
        if not new.exists():
            sys.exit(f"missing new matrix: {new}")
        out = args.out_dir / f"{clean_stem(old.stem)}_{args.suffix}.csv"
        results.append(merge_one(old, new, out, excluded,
                                 f"{args.ref} {label}  ->  {out.name}"))

    n_cov, n_dep = results[0][0], results[1][0]
    print(f"\n  row counts agree between coverage and depth: {n_cov == n_dep} "
          f"({n_cov:,} vs {n_dep:,})")
    bad = any(m or e for _, m, e, _ in results)
    print(f"  column sets identical in both merges: {not bad}")
    return 1 if (bad or n_cov != n_dep) else 0


if __name__ == "__main__":
    sys.exit(main())
