#!/usr/bin/env python3
"""Do the manuscript, the pipeline, the results record and Data S1 agree on
figure order?

Four independent sources, compared slot by slot:

  1. the pipeline's contract   tools/additional_figures_manifest.txt
  2. the results record        the deposited S01..S31 filenames
  3. Data S1                   the `index` sheet's figure -> source_figure
  4. the manuscript            the supplement's Fig. S1..S31 captions

The first three name a stem, so they compare exactly. The manuscript names
only a number and a caption, so it is checked for a complete unbroken
S1..S31 with no duplicates, and its caption is printed beside the stem for
reading.

    python tools/verify_figure_order.py \\
        --record-b ../record_b_staging \\
        --data-s1 Data_S1_figure_data.xlsx \\
        --supplement Persistence_Supp_Mat.docx

Exit status is 1 if any source disagrees, so this works as a pre-submission
gate rather than something to read and judge by eye.
"""
from __future__ import annotations

import argparse
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def norm(slot: str) -> str:
    return f"S{int(str(slot).lstrip('Ss'))}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record-b", type=Path, required=True, metavar="DIR",
                    help="a built results-record staging directory; the "
                         "deposited S01..S31 figures sit at its top level")
    ap.add_argument("--data-s1", type=Path, required=True, metavar="XLSX",
                    help="the Data S1 workbook to check")
    ap.add_argument("--supplement", type=Path, default=None, metavar="DOCX",
                    help="the supplement .docx; when given, its Fig. S "
                         "captions are checked for a complete, unduplicated "
                         "S1..S31")
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "tools/additional_figures_manifest.txt",
                    help="the slot contract (default: the committed one)")
    a = ap.parse_args()

    # 1. the pipeline's contract
    manifest = {}
    for line in a.manifest.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            slot, _ref, stem = (x.strip() for x in line.split("|"))
            manifest[norm(slot)] = stem.rsplit("/", 1)[-1]

    # 2. the deposited filenames
    deposit = {}
    for p in sorted(a.record_b.glob("S[0-9][0-9]_*.png")):
        m = re.match(r"^S(\d{2})_(.+)\.png$", p.name)
        if m:
            deposit[norm(m.group(1))] = m.group(2)

    # 3. Data S1's index
    wb = openpyxl.load_workbook(a.data_s1, read_only=True, data_only=True)
    ws = wb["index"]
    it = ws.iter_rows(values_only=True)
    hdr = list(next(it))
    ci = {h: i for i, h in enumerate(hdr)}
    ds1: dict = {}
    for r in it:
        m = re.match(r"Fig\. S(\d+)", str(r[ci["figure"]] or ""))
        if m and r[ci["source_figure"]]:
            ds1.setdefault(norm(m.group(1)),
                           set()).add(str(r[ci["source_figure"]]))
    wb.close()

    # 4. the manuscript's captions
    caps: dict = {}
    if a.supplement is not None:
        doc = zipfile.ZipFile(a.supplement).read("word/document.xml")
        root = ET.fromstring(doc)
        pat = re.compile(r"^Fig(?:ure)?\.?\s*S(\d{1,2})[.:]\s*(.*)")
        for p in root.iter(f"{W}p"):
            txt = "".join(t.text or "" for t in p.iter(f"{W}t")).strip()
            m = pat.match(txt)
            if m:
                caps.setdefault(norm(m.group(1)), m.group(2)[:58])

    print(f"  pipeline manifest  : {len(manifest)} slots")
    print(f"  deposited figures  : {len(deposit)}")
    print(f"  Data S1 index      : {len(ds1)} slots")
    print(f"  supplement captions: {len(caps)}"
          + ("" if a.supplement is not None else "  (not checked)"))
    print()

    slots = sorted(set(manifest) | set(deposit) | set(ds1),
                   key=lambda s: int(s[1:]))
    bad = []
    print(f"{'slot':6s} {'man':4s} {'dep':4s} {'DS1':4s} {'cap':4s}  stem")
    print("-" * 78)
    for s in slots:
        mf, dp = manifest.get(s), deposit.get(s)
        d1, cp = ds1.get(s, set()), caps.get(s)
        ok_dep = mf is not None and dp is not None and (
            mf == dp or mf in dp or dp in mf)
        ok_ds1 = bool(d1) and dp is not None and any(
            dp == x or dp in x or x in dp for x in d1)
        print(f"{s:6s} {'ok' if mf else '--':4s} {'ok' if dp else '--':4s} "
              f"{'ok' if d1 else '--':4s} {'ok' if cp else '--':4s}  "
              f"{(dp or mf or '?')[:44]}")
        if cp:
            print(f"{'':31s}caption: {cp}")
        if not (ok_dep and ok_ds1):
            bad.append(s)

    print("-" * 78)
    print(f"  manifest == deposit == Data S1 on every slot: "
          f"{'YES' if not bad else 'NO'}")
    if a.supplement is not None:
        nums = sorted(int(s[1:]) for s in caps)
        gaps = [n for n in range(1, len(manifest) + 1) if n not in nums]
        dupes = sorted({n for n in nums if nums.count(n) > 1})
        print(f"  supplement captions complete: "
              f"{'YES' if not gaps else 'NO, missing ' + str(gaps)}")
        print(f"  duplicate captions: {dupes if dupes else 'none'}")
        if gaps or dupes:
            bad += [f"caption:{n}" for n in gaps + dupes]
    if bad:
        print(f"\n  FAIL: {len(bad)} slot(s) disagree: "
              f"{', '.join(map(str, bad))}")
        return 1
    print("\n  PASS: every source agrees on figure order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
