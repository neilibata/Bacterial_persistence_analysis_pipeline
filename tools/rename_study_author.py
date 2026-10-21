#!/usr/bin/env python3
"""Rename a study's author label everywhere it appears.

`Davis-Richardson` is a mis-attribution: the dataset's paper is Brown et al. 2011
(PLoS ONE 6:e25792, doi:10.1371/journal.pone.0025792), where Davis-Richardson is
the second author. It is also one of only four Author values that break the
`SurnameInitials` convention the other 76 follow.

The rename touches the curated metadata (Table M1) and every derived artefact that
carries the label. It is purely cosmetic -- no sample, count or statistic changes --
so derived CSVs are patched in place rather than regenerated.

Substitution is done on raw bytes, not through pandas and not through text mode.
A pandas round-trip of the 116-column metadata would renormalise quoting, NaN
spelling and float formatting across 18,048 rows. Text mode is no safer: Python's
universal-newline handling silently rewrites CRLF to LF on read, which would touch
every line of a CRLF file while appearing to change only the label. Byte-level
replacement leaves every other byte alone, line endings included.

Matches are anchored to token boundaries -- a hit must not be flanked by
`[A-Za-z0-9_-]`. Bare substring replacement is unsafe here because these labels are
prefixes of one another: replacing `Nagy-Szakal` with `Nagy-SzakalD` would also hit
the `Nagy-Szakal` inside `Nagy-Szakal_2017`, and applying the pair list twice would
yield `Nagy-SzakalDD`. Anchoring makes each pair independent and the whole operation
idempotent, so a repeat run is a no-op rather than a corruption.

    python tools/rename_study_author.py                     # dry run
    python tools/rename_study_author.py --apply
    python tools/rename_study_author.py --old X --new Y --apply
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (old, new) ordered longest-first so compound labels are rewritten before the bare name
DEFAULT_PAIRS = [
    # Author column
    ("Nagy-Szakal", "Nagy-SzakalD"),
    # StudyFolder: prefix must match the Author it belongs to
    ("Nagy-Szakal_2017", "Nagy-SzakalD_2017"),
    ("Karlsson_2013_SWE_fastq", "KarlssonFH_2013_SWE_fastq"),
    ("KiminaKM_2019_RUS", "KliminaKM_2019_RUS"),          # misspelling of Klimina
    ("Liu_2016_MNG_fastq", "LiuW_2016_MNG_fastq"),        # merges 3 strays into 60
    ("Thomas_2019_ITA", "ThomasAM_2019_ITA"),
    ("Vangay_2018_THA_fastq", "VangayP_2018_THA_fastq"),
    ("USA_2012_HMP", "HMP_2012_USA"),                     # reversed field order
]

# A hit flanked by any of these is part of a longer token, not a match.
_BOUNDARY = rb"[A-Za-z0-9_-]"

# Extensions worth scanning. Binary artefacts (.docx, .xlsx, .png) are handled
# separately or by hand -- rewriting them as text would corrupt them.
TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".md", ".yaml", ".yml", ".py"}

SKIP_DIRS = {".git", "__pycache__", ".mplconfig", ".pipeline_state"}


def _pattern(token: str) -> "re.Pattern[bytes]":
    """Token-boundary-anchored matcher for `token`, on bytes."""
    return re.compile(rb"(?<!" + _BOUNDARY + rb")" + re.escape(token.encode())
                      + rb"(?!" + _BOUNDARY + rb")")


def iter_candidates(root: Path):
    me = Path(__file__).resolve()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        # This file names both labels in its docstring and in DEFAULT_PAIRS;
        # rewriting it would destroy the mapping it exists to apply.
        if path.resolve() == me:
            continue
        yield path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--old", default=None)
    ap.add_argument("--new", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup-dir", type=Path, default=None,
                    help="Where to copy originals (default: <root>/.rename_backup)")
    args = ap.parse_args()

    if bool(args.old) ^ bool(args.new):
        ap.error("--old and --new must be given together")
    pairs = ([(args.old, args.new)] if args.old
             else DEFAULT_PAIRS)

    backup_dir = args.backup_dir or (args.root / ".rename_backup")

    hits = []
    for path in iter_candidates(args.root):
        try:
            blob = path.read_bytes()
        except OSError:
            continue
        counts = {old: len(_pattern(old).findall(blob)) for old, _ in pairs}
        total = sum(counts.values())
        if total:
            hits.append((path, counts, total))

    if not hits:
        print("  nothing to rename")
        return 0

    print(f"  replacing: " + ", ".join(f"{o!r} -> {n!r}" for o, n in pairs))
    print(f"  files with occurrences: {len(hits)}\n")
    print(f"  {'file':<62}{'hits':>6}")
    grand = 0
    for path, counts, total in sorted(hits, key=lambda x: str(x[0])):
        rel = path.relative_to(args.root)
        # A symlinked input lives outside the tree; say so, because editing it
        # changes the shared data directory, not just this repository.
        tag = "  -> symlink" if path.is_symlink() else ""
        print(f"  {str(rel)[:60]:<62}{total:>6}{tag}")
        grand += total
    print(f"\n  total occurrences: {grand}")

    if not args.apply:
        print("\n  (dry run; pass --apply to write)")
        return 0

    backup_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path, _, _ in hits:
        resolved = path.resolve()
        rel = path.relative_to(args.root)
        dest = backup_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolved, dest)

        blob = resolved.read_bytes()
        for old, new in pairs:
            blob = _pattern(old).sub(new.encode(), blob)
        resolved.write_bytes(blob)
        written += 1

    print(f"\n  backups : {backup_dir}")
    print(f"  rewritten: {written} file(s)")
    print("\n  NOTE: the metadata is an input to the resume mtime check, so every\n"
          "  results/<ref>/.pipeline_state/*.done is now older than its input and a\n"
          "  --resume run will recompute every stage. The change is label-only, so the\n"
          "  recomputed numbers will be identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
