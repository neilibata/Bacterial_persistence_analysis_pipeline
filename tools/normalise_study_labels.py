#!/usr/bin/env python3
"""Normalise `Country` to true ISO 3166-1 alpha-3 and `StudyFolder` to Author_YYYY_ISO3.

Two separate defects, both cosmetic today but both traps for anyone reading the
deposited metadata:

  * Ten `Country` codes are not ISO3. Most are merely non-standard (HON, NEP, SA),
    but `IND` is actively misleading -- it denotes Indonesia here, while ISO3 `IND`
    is India. `normalise_country` already resolves all of them, so nothing in the
    pipeline is wrong; a reader taking the column at face value would be.
  * `StudyFolder` follows Author_YYYY_ISO3 in only 37 of 111 cases. The rest drop
    the country, carry a stray `_fastq`, disagree with their own `Author`, or spell
    one country two ways.

Five renames collide, which is the point rather than a problem: each pair is one
study split by naming drift, verified by shared BioProject accession, identical
Author and Country, and zero overlapping sample IDs.

Folders spanning more than one country keep no country code -- inventing a single
code for a seven-country study would be worse than omitting it.

The metadata is rewritten field-by-field through the csv module, not by text
substitution. Codes like `UK`, `SA` and `IND` are short enough to occur inside
unrelated free-text fields, so a global replace would corrupt them. A csv
round-trip of this file is byte-identical apart from a trailing newline, which is
restored explicitly.

    python tools/normalise_study_labels.py            # report only
    python tools/normalise_study_labels.py --apply
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "metadata_Mar_16_2026.csv"

# non-ISO3 -> ISO3. Country (full) in the metadata confirms each reading.
ISO3 = {
    "BKF": "BFA",   # Burkina Faso
    "PHN": "PHL",   # Philippines
    "SA":  "ZAF",   # South Africa
    "UK":  "GBR",   # United Kingdom
    "IND": "IDN",   # Indonesia -- ISO3 IND is India
    "MAD": "MDG",   # Madagascar
    "NEP": "NPL",   # Nepal
    "HON": "HND",   # Honduras
    "DRC": "COD",   # DR Congo
    "LIB": "LBR",   # Liberia
}

# StudyFolder prefixes that disagree with their own Author column
PREFIX = {
    "Karlsson": "KarlssonFH",
    "KiminaKM": "KliminaKM",     # misspelling of Klimina
    "Liu": "LiuW",
    "Thomas": "ThomasAM",
    "Vangay": "VangayP",
    "Nagy-Szakal": "Nagy-SzakalD",
}

AUTHOR_FIX = {"Nagy-Szakal": "Nagy-SzakalD"}

_BOUNDARY = rb"[A-Za-z0-9_-]"


def token_pattern(token: str):
    return re.compile(rb"(?<!" + _BOUNDARY + rb")" + re.escape(token.encode())
                      + rb"(?!" + _BOUNDARY + rb")")


def read_rows(path: Path):
    raw = path.read_bytes()
    rows = list(csv.reader(io.StringIO(raw.decode("utf8"), newline="")))
    return rows, raw.endswith(b"\n")


def write_rows(path: Path, rows, trailing_newline: bool) -> None:
    buf = io.StringIO(newline="")
    csv.writer(buf, lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL).writerows(rows)
    out = buf.getvalue().encode("utf8")
    if not trailing_newline:
        out = out.rstrip(b"\r\n")
    path.write_bytes(out)


def build_folder_map(rows, header):
    i_folder = header.index("StudyFolder")
    i_country = header.index("Country")
    countries = defaultdict(set)
    for row in rows[1:]:
        if row[i_folder]:
            if row[i_country]:
                countries[row[i_folder]].add(row[i_country])

    mapping = {}
    for folder in sorted(countries) or []:
        new = folder[:-6] if folder.endswith("_fastq") else folder
        if folder == "USA_2012_HMP":
            new = "HMP_2012_USA"
        m = re.match(r"^(.*?)_(\d{4})(?:_(.+))?$", new)
        if not m:
            continue
        author, year, code = m.group(1), m.group(2), m.group(3)
        author = PREFIX.get(author, author)
        seen = sorted(countries[folder])
        if code:
            code = ISO3.get(code.upper(), code.upper())
        elif len(seen) == 1:
            code = ISO3.get(seen[0], seen[0])
        else:
            code = None
        mapping[folder] = f"{author}_{year}" + (f"_{code}" if code else "")
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=METADATA)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    path = args.metadata.resolve()
    rows, trailing = read_rows(path)
    header = rows[0]
    i_country, i_folder, i_author = (header.index("Country"),
                                     header.index("StudyFolder"),
                                     header.index("Author"))

    folder_map = build_folder_map(rows, header)
    changed_folders = {o: n for o, n in folder_map.items() if o != n}
    collisions = {n: [o for o in folder_map if folder_map[o] == n]
                  for n in set(folder_map.values())}
    collisions = {n: o for n, o in collisions.items() if len(o) > 1}

    counts = Counter()
    for row in rows[1:]:
        if row[i_country] in ISO3:
            counts["country"] += 1
        if row[i_folder] in changed_folders:
            counts["folder"] += 1
        if row[i_author] in AUTHOR_FIX:
            counts["author"] += 1

    print(f"  metadata : {path.name}  ({len(rows) - 1:,} rows)")
    print(f"  Country codes to fix   : {len(ISO3)} codes, {counts['country']:,} rows")
    print(f"  StudyFolder to rename  : {len(changed_folders)} folders, {counts['folder']:,} rows")
    print(f"  Author to fix          : {len(AUTHOR_FIX)}, {counts['author']:,} rows")
    print(f"\n  folders merging ({len(collisions)}):")
    for new, olds in sorted(collisions.items()):
        print(f"    {new:<28}<- {olds}")

    if not args.apply:
        print("\n  (report only; pass --apply to write)")
        return 0

    backup = path.with_suffix(".pre_normalise.csv")
    shutil.copy2(path, backup)
    for row in rows[1:]:
        if row[i_country] in ISO3:
            row[i_country] = ISO3[row[i_country]]
        if row[i_folder] in folder_map:
            row[i_folder] = folder_map[row[i_folder]]
        if row[i_author] in AUTHOR_FIX:
            row[i_author] = AUTHOR_FIX[row[i_author]]
    write_rows(path, rows, trailing)
    print(f"\n  backup  : {backup}")
    print(f"  written : {path}")

    # Derived artefacts carry the folder strings verbatim. Country codes are NOT
    # touched there: they are too short to replace safely outside a parsed column,
    # and no derived table is keyed on them.
    targets = [p for p in ROOT.rglob("*.csv")
               if ".git" not in p.parts and not any(
                   part.startswith(".rename_backup") for part in p.parts)]
    patterns = [(token_pattern(o), n.encode()) for o, n in changed_folders.items()]
    touched = 0
    for target in targets:
        if target.resolve() == path:
            continue
        try:
            blob = target.read_bytes()
        except OSError:
            continue
        if not any(o.encode() in blob for o in changed_folders):
            continue
        for pat, new in patterns:
            blob = pat.sub(new, blob)
        target.write_bytes(blob)
        touched += 1
    print(f"  derived CSVs updated: {touched}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
