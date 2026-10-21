#!/usr/bin/env python3
"""Apply the publication display labels to the stats reports already on disk.

`StatsReport` writes a tag and a description for every number. The tag is an
identifier -- tools/verify_manuscript_numbers.py and the manuscript crosswalk
match on it -- so it keeps the internal condition key. The description is what
a reader sees, and Table S11 quotes it, so it takes the published name.

The source now does this at write time (section3_health.py, table_s7.py). This
brings the reports that were written before that change into line, without the
multi-hour re-run that would otherwise be needed to change a label.

Only description text moves. Tags, values, p-values and every other column are
untouched.

    python tools/refresh_stats_labels.py --dry-run
    python tools/refresh_stats_labels.py
"""
from __future__ import annotations

import argparse
import glob
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BACKUP = ROOT / "archive" / "pre_masld" / "results"


def replacements():
    """Only the condition rename.

    config.PLOT_TEXT_REPLACEMENTS also carries the lifestyle display labels.
    Those are deliberately left as the internal keys in the stats reports --
    relabelling them here would be a separate change to a separate set of
    numbers, and nobody asked for it.
    """
    from persistence import config
    return {k: v for k, v in config.PLOT_TEXT_REPLACEMENTS.items()
            if v == config.MASLD_DISPLAY}


def sub_text(s, reps):
    """Display replacement, skipping anything embedded in a stats tag.

    A tag spells the condition `cmh_VANISH_Fatty Liver_10pct` or
    `table_s6b_Fatty_Liver_VANISH_10pct` -- the key flanked by underscores.
    Leaving those alone keeps every tag matchable while the prose around them
    is relabelled.
    """
    out = str(s)
    for old, new in reps:
        if "\n" in old:                    # wrapped forms: no tag looks like this
            out = out.replace(old, new)
            continue
        out = re.sub(rf"(?<![_\w]){re.escape(old)}(?![_\w])", new, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # Longest first, so a key that contains another is rewritten whole.
    reps = sorted(replacements().items(), key=lambda kv: -len(kv[0]))
    print("  applying: " + ", ".join(f"{k!r}->{v!r}" for k, v in reps
                                     if "Fatty" in k))

    total = 0
    for ref_dir in sorted(glob.glob(str(ROOT / "results" / "*"))):
        ref = Path(ref_dir).name

        csv = Path(ref_dir) / "stats_report.csv"
        if csv.exists():
            d = pd.read_csv(csv, dtype=str, keep_default_na=False)
            if "description" in d.columns:
                before = d["description"].copy()
                # No tag lives in this column, so a plain replacement is safe.
                d["description"] = before.map(lambda s: sub_text(s, reps))
                n = int((d["description"] != before).sum())
                if n:
                    total += n
                    print(f"  {ref:10} stats_report.csv        "
                          f"{n:>4} descriptions")
                    if not a.dry_run:
                        dest = BACKUP / ref / csv.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if not dest.exists():
                            shutil.copy2(csv, dest)
                        d.to_csv(csv, index=False)

        for name in ("stats_report.txt", "manuscript_fill_ins.txt"):
            p = Path(ref_dir) / name
            if not p.exists():
                continue
            txt = p.read_text(errors="ignore")
            new = sub_text(txt, reps)
            if new == txt:
                continue
            n = sum(1 for x, y in zip(txt.splitlines(), new.splitlines())
                    if x != y)
            total += n
            print(f"  {ref:10} {name:23} {n:>4} lines")
            if not a.dry_run:
                dest = BACKUP / ref / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(p, dest)
                p.write_text(new)

    print(f"\n  {total} description lines relabelled")
    if a.dry_run:
        print("  dry run; nothing written")
    else:
        print(f"  originals under {BACKUP.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
