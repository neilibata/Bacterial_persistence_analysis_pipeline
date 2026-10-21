#!/usr/bin/env python3
"""For each Supplementary Figure: is the data behind it written out anywhere?

Science requires the tabulated data underlying every figure as data S1, keyed
to the figure panel. That is the *source* data -- sample identifiers, group
labels, the measured values, and the statistics quoted in annotations -- not
the coordinates of the drawn artists, which would let someone redraw the
picture without being able to check the analysis.

This pairs figures with tables by reading the code rather than by matching
filenames: it finds the function that saves each figure and reports the tables
that same function writes. Filename matching was tried first and produced
confident nonsense.

    python tools/audit_source_data.py
"""
from __future__ import annotations

import ast, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def manifest():
    out = []
    for line in (ROOT / "tools/additional_figures_manifest.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            slot, ref, stem = line.split("|")
            out.append((slot.strip(), ref.strip(), stem.strip()))
    return out


def _strings(node):
    """Every string constant appearing anywhere under this node."""
    return [n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def index_sources():
    """function -> (figures it saves, tables it writes), per module."""
    idx = {}
    files = [p for p in list((ROOT / "persistence").glob("*.py"))
             + list((ROOT / "figures").glob("*.py"))
             + list((ROOT / "tools").glob("*.py"))
             if not p.name.endswith(".bak")]
    for p in files:
        try:
            tree = ast.parse(p.read_text())
        except Exception:
            continue
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            # Save paths are almost always built into a variable, so a figure
            # name never appears inside the savefig call itself. Take every
            # string in the function and note whether it saves and what it
            # writes; the name will be in there somewhere.
            has_save = has_csv = False
            tables = []
            for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
                name = ""
                if isinstance(call.func, ast.Attribute):
                    name = call.func.attr
                elif isinstance(call.func, ast.Name):
                    name = call.func.id
                if name in ("savefig_dual", "savefig",
                            "savefig_dual_preserve_layout"):
                    has_save = True
                if name == "to_csv":
                    has_csv = True
            allstr = _strings(fn)
            if has_csv:
                tables = [t for t in allstr if ".csv" in t]
            if has_save:
                idx[(p.name, fn.name)] = (allstr, tables, fn)
    return idx


def main() -> int:
    idx = index_sources()
    man = manifest()
    print(f"  {'slot':<6}{'figure':<44}{'tables written by the same function'}")
    none = []
    for slot, ref, stem in man:
        leaf = stem.split("/")[-1]
        base = re.sub(r"_(10pct|300kb|\d+pct|\d+kb)$", "", leaf)
        hits = []
        for (mod, fnname), (saves, tables, _fn) in idx.items():
            blob = " ".join(saves)
            if base and (base in blob or leaf in blob):
                hits.append((mod, fnname, tables))
        if not hits:
            # fall back: the stem may be built from an f-string fragment
            for (mod, fnname), (saves, tables, _fn) in idx.items():
                if any(base[:18] in s for s in saves):
                    hits.append((mod, fnname, tables))
        if hits:
            tbls = sorted({t for _, _, ts in hits for t in ts if t.endswith(".csv")})
            src = hits[0][0]
            print(f"  {slot:<6}{leaf[:42]:<44}{src[:22]:<24}"
                  f"{(', '.join(tbls)[:46] if tbls else 'NONE')}")
            if not tbls:
                none.append((slot, leaf, src))
        else:
            print(f"  {slot:<6}{leaf[:42]:<44}{'-- generator not identified --'}")
            none.append((slot, leaf, "?"))
    print(f"\n  {len(man)-len(none)}/{len(man)} figures have a table written "
          f"alongside them by the same function")
    print(f"  {len(none)} need source data emitting:")
    for slot, leaf, src in none:
        print(f"    {slot:<6}{leaf[:44]:<46}{src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
