#!/usr/bin/env python3
"""Resolve manuscript placeholders to values the pipeline actually produced.

The pipeline already maintains a registry of every manuscript number: each
section calls ``stats.add(tag, section, placement, description, value)`` and the
result lands in ``results/<ref>/stats_report.csv``. This module makes that
registry addressable from prose, so a sentence can cite a computed value instead
of a transcribed one.

Placeholder syntax, used by tools/render_manuscript.py:

    {{UHGG:n_samples_Industrial}}          the `value` column, verbatim
    {{combined:vanish_overlap}}            "
    {{UHGG:...spearman_all_10pct|rho}}     a named field parsed out of `value`
    {{UHGG:n_countries|int}}               formatted as an integer
    {{UHGG:n_samples_Industrial|comma}}    1,234

Resolution is strict: an unknown ref, an unknown tag, or a field that cannot be
parsed raises rather than silently emitting a stale or empty number. That is the
whole point — a placeholder that cannot be resolved must stop the build.

    python tools/manuscript_values.py --list UHGG          # browse available tags
    python tools/manuscript_values.py --get UHGG:n_countries
"""
from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFS = ("UHGG", "lev", "combined", "chimp")


class UnresolvedPlaceholder(RuntimeError):
    """A placeholder could not be resolved — the build must not continue."""


@lru_cache(maxsize=None)
def _report(ref: str) -> pd.DataFrame:
    path = ROOT / "results" / ref / "stats_report.csv"
    if not path.exists():
        raise UnresolvedPlaceholder(
            f"no stats report for reference '{ref}' ({path.relative_to(ROOT)}). "
            f"Run: python -m persistence --ref {ref}")
    d = pd.read_csv(path)
    if "tag" not in d.columns:
        raise UnresolvedPlaceholder(f"{path} has no 'tag' column")
    return d


# Numbers embedded in a `value` string, e.g. "ρ=+0.706, p=0.000e+00, n=18114"
_FIELD_PATTERNS = {
    "rho":    r"ρ\s*=\s*([+-]?\d*\.?\d+)",
    "n":      r"\bn\s*=\s*([\d,]+)",
    "p":      r"\bp\s*=\s*([\d.eE+-]+)",
    "median": r"median\s*=\s*([\d.,]+)",
    "mean":   r"mean\s*=\s*([\d.,]+)",
    "ratio":  r"ratio\s*=\s*([\d.]+)",
    "ci_lo":  r"\[\s*95%\s*CI:\s*([\d.+-]+)",
    "ci_hi":  r"95%\s*CI:[^,]+,\s*([\d.+-]+)\s*\]",
    "cv":     r"CV\s*=\s*([\d.]+)%?",
    "pm_mean": r"^\s*([\d.,]+)\s*(?:\+/-|±)",
    "pm_se":   r"(?:\+/-|±)\s*([\d.,]+)",
}


def _apply_filter(raw: str, filt: str | None, where: str) -> str:
    if filt is None:
        return str(raw).strip()
    if filt in _FIELD_PATTERNS:
        m = re.search(_FIELD_PATTERNS[filt], str(raw))
        if not m:
            raise UnresolvedPlaceholder(
                f"{where}: no '{filt}' field in value {raw!r}")
        return m.group(1).replace(",", "")
    if filt in ("int", "comma"):
        m = re.search(r"[-+]?[\d,]*\.?\d+", str(raw))
        if not m:
            raise UnresolvedPlaceholder(f"{where}: no number in value {raw!r}")
        n = int(round(float(m.group(0).replace(",", ""))))
        return f"{n:,}" if filt == "comma" else str(n)
    if filt.startswith("round"):                       # round2, round3, ...
        digits = int(filt[5:] or 2)
        m = re.search(r"[-+]?[\d,]*\.?\d+", str(raw))
        if not m:
            raise UnresolvedPlaceholder(f"{where}: no number in value {raw!r}")
        return f"{float(m.group(0).replace(',', '')):.{digits}f}"
    raise UnresolvedPlaceholder(f"{where}: unknown filter '{filt}'")


def resolve(token: str) -> str:
    """Resolve one placeholder body, e.g. 'UHGG:n_countries|comma'."""
    body, _, filt = token.partition("|")
    ref, _, tag = body.partition(":")
    ref, tag, filt = ref.strip(), tag.strip(), (filt.strip() or None)
    if not tag:
        raise UnresolvedPlaceholder(f"malformed placeholder {{{{{token}}}}}")
    if ref not in REFS:
        raise UnresolvedPlaceholder(
            f"unknown reference '{ref}' in {{{{{token}}}}} (expected one of {', '.join(REFS)})")

    d = _report(ref)
    hit = d[d["tag"].astype(str) == tag]
    if not len(hit):
        near = [t for t in d["tag"].astype(str) if tag.lower() in t.lower()][:4]
        extra = f"; did you mean {near}?" if near else ""
        raise UnresolvedPlaceholder(f"tag '{tag}' not found for ref '{ref}'{extra}")
    return _apply_filter(hit.iloc[0]["value"], filt, f"{{{{{token}}}}}")


def describe(token: str) -> dict:
    """Resolved value plus the provenance a reader needs to check it."""
    body, _, filt = token.partition("|")
    ref, _, tag = body.partition(":")
    row = _report(ref.strip())
    row = row[row["tag"].astype(str) == tag.strip()].iloc[0]
    return {"token": token, "ref": ref.strip(), "tag": tag.strip(),
            "value": resolve(token), "raw": row["value"],
            "description": row.get("description", ""),
            "source": f"results/{ref.strip()}/stats_report.csv"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", metavar="REF", help="list available tags for a reference")
    g.add_argument("--get", metavar="TOKEN", help="resolve one placeholder, e.g. UHGG:n_countries")
    ap.add_argument("--grep", help="with --list, filter tags by substring")
    args = ap.parse_args()

    if args.get:
        try:
            info = describe(args.get)
        except UnresolvedPlaceholder as exc:
            sys.exit(f"unresolved: {exc}")
        print(f"  {info['value']}")
        print(f"    tag         {info['ref']}:{info['tag']}")
        print(f"    description {info['description']}")
        print(f"    raw value   {info['raw']}")
        print(f"    source      {info['source']}")
        return

    d = _report(args.list)
    rows = d[d["tag"].astype(str).str.contains(args.grep, case=False, na=False)] if args.grep else d
    print(f"{len(rows)} tag(s) for {args.list}"
          + (f" matching {args.grep!r}" if args.grep else "") + "\n")
    for _, r in rows.iterrows():
        print(f"  {r['tag']:<52} {str(r['value'])[:60]}")


if __name__ == "__main__":
    main()
