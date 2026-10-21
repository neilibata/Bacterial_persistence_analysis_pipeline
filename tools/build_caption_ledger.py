#!/usr/bin/env python3
"""Collate everything that has to move from a figure into its caption.

Two inputs, because there are two kinds of debt:

  * tools/caption_debt.csv   - written by hand as figures were edited, one row
                               per thing already taken out of a figure.
  * tools/figure_text_inventory.csv - harvested from the figures themselves by
                               tools/collect_figure_text.py, which is what
                               catches the text nobody remembered to record.

Science puts the figure's title in the caption and asks for explanatory text
in the figure to be minimised, so the harvest contributes a row for every
title and every multi-line label or panel heading: those are the strings the
formatting pass removes, and the caption then has to carry them.

    python tools/build_caption_ledger.py

Writes docs/figure_captions.md and refreshes tools/caption_debt.csv in place.
"""
from __future__ import annotations

import csv
import datetime
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import figure_sets                      # noqa: E402

DEBT = ROOT / "tools" / "caption_debt.csv"
INVENTORY = ROOT / "tools" / "figure_text_inventory.csv"
OUT_MD = ROOT / "docs" / "figure_captions.md"

FIELDS = ["slot", "figure", "removed", "caption_must_say", "source", "status",
          "origin"]

# Approved on 2026-09-14: abbreviations are allowed on the figures provided
# the caption expands them.
ABBREV = [("H-G", "Hunter-gatherer"), ("A-P", "Agro-pastoralist"),
          ("I", "Industrial"), ("NI", "Non-Industrial")]


def slot_num(slot):
    m = re.sub(r"\D", "", slot)
    return int(m) if m else 999


def load_rows(path, fields=None):
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def derive(inventory, stems):
    """Caption rows implied by the text still sitting on the figures."""
    out = []
    for r in inventory:
        slot, role, text = r["slot"], r["role"], r["text"]
        lines = int(r["lines"])
        fig = stems.get(slot, "")
        if role == "suptitle":
            head = text.split(" \\n ")[0]
            rest = " ".join(text.split(" \\n ")[1:]).strip()
            out.append(dict(
                slot=slot, figure=fig,
                removed=f'figure title: "{text}"',
                caption_must_say=(
                    f"Open the caption with the title: {head}."
                    + (f" Carry the qualifying line into the caption body: {rest}"
                       if rest else "")),
                source="figure", status="open", origin="auto:title"))
        elif role in ("axes_title",) and lines > 1:
            rest = " ".join(text.split(" \\n ")[1:]).strip()
            if rest:
                out.append(dict(
                    slot=slot, figure=fig,
                    removed=f'panel heading second line: "{rest}"',
                    caption_must_say=f"State per panel: {rest}",
                    source="figure", status="open", origin="auto:panel"))
        elif role in ("xlabel", "ylabel") and lines > 1:
            rest = " ".join(text.split(" \\n ")[1:]).strip()
            if rest:
                out.append(dict(
                    slot=slot, figure=fig,
                    removed=f'axis label qualifier: "{rest}"',
                    caption_must_say=f"Define the axis: {rest}",
                    source="figure", status="open", origin="auto:axis"))
    return out


def dedupe(rows):
    seen, out = set(), []
    for r in rows:
        k = (r["slot"], r.get("removed", "")[:120])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def main():
    stems = {slot: Path(st).name for slot, _r, st in figure_sets.manifest()}
    manual = [r for r in load_rows(DEBT) if r.get("origin") != "auto"
              and not str(r.get("origin", "")).startswith("auto:")]
    for r in manual:
        r.setdefault("origin", "manual")
    derived = derive(load_rows(INVENTORY), stems)
    rows = dedupe(manual + derived)
    rows.sort(key=lambda r: (slot_num(r["slot"]), r.get("origin", "")))

    with DEBT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    # ── the readable ledger ───────────────────────────────────────────────
    by = defaultdict(list)
    for r in rows:
        by[r["slot"]].append(r)
    inv = defaultdict(list)
    for r in load_rows(INVENTORY):
        inv[r["slot"]].append(r)

    n_open = sum(1 for r in rows if r.get("status") == "open")
    L = []
    L.append("# Caption ledger\n")
    L.append(f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by "
             "`tools/build_caption_ledger.py`.\n")
    L.append(f"**{len(rows)} items, {n_open} still open.** Every row is something "
             "that is being taken off a figure to meet the Science format, and "
             "therefore has to appear in that figure's caption instead. Rows "
             "marked `manual` were recorded by hand as figures were edited; rows "
             "marked `auto:` were harvested from the figures themselves, which is "
             "how the untracked titles and second lines were found.\n")
    L.append("## Applies to every figure\n")
    L.append("Science asks for the title in the caption rather than the figure, "
             "for explanatory text to be minimised in favour of the caption, and "
             "for keys to be as simple as possible. Three consequences run across "
             "the whole set:\n")
    L.append("- **Titles.** Every figure title below moves into the opening "
             "sentence of its caption. No figure keeps a title.")
    L.append("- **Abbreviations.** Where a figure uses the short forms, the "
             "caption must expand them: "
             + "; ".join(f"**{a}** = {b}" for a, b in ABBREV) + ".")
    L.append("- **Detection threshold.** The `>=10% breadth coverage` / "
             "`>=300 kB mapped bases` qualifier is repeated in almost every "
             "title. State the threshold once in the caption instead.\n")

    for slot in sorted(by, key=slot_num):
        items = by[slot]
        fig = next((i["figure"] for i in items if i.get("figure")), "")
        L.append(f"## {slot} — `{fig}`\n")
        title = next((i for i in items if i.get("origin") == "auto:title"), None)
        if title:
            t = re.sub(r'^figure title: "|"$', "", title["removed"])
            t = t.replace(" \\n ", " — ")
            L.append(f"**Title for the caption:** {t}\n")
        rest = [i for i in items if i is not title]
        if rest:
            L.append("| Taken off the figure | The caption must say | Status |")
            L.append("|---|---|---|")
            for i in rest:
                rm = i.get("removed", "").replace("|", "\\|")
                cm = i.get("caption_must_say", "").replace("|", "\\|")
                L.append(f"| {rm} | {cm} | {i.get('status','open')} |")
            L.append("")
        strings = inv.get(slot, [])
        keep = [s for s in strings if s["role"] not in ("xtick", "ytick")]
        if keep:
            worst = max(float(s["printed_pt"]) for s in keep)
            least = min(float(s["printed_pt"]) for s in keep)
            L.append(f"<sub>{len(strings)} strings on this figure; type prints "
                     f"between {least:.1f} and {worst:.1f} pt at 18.4 cm wide "
                     f"(Science asks for about 7 pt, never below 5).</sub>\n")

    OUT_MD.write_text("\n".join(L))
    print(f"  {DEBT}  ({len(rows)} rows, {n_open} open)")
    print(f"  {OUT_MD}")


if __name__ == "__main__":
    main()
