#!/usr/bin/env python3
"""Probe each Additional-figure slot before and after the sizing change.

"Before" is the retired preset output (<stem>_large_notitle.png and friends),
"after" is what the current policy writes (<stem>_notitle.png). Both are probed
at source resolution by tools/figure_layout_probe.py, so the two are measured
the same way; downsampled copies are not comparable.

    python tools/figure_ab_probe.py -o docs/layout_ab.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_layout_probe import probe, PLACEMENT_WIDTH_IN     # noqa: E402

NEW_SUFFIXES = ("_notitle", "")

# "Before" is not re-resolved from disk. Each slot shipped with a specific font
# preset, and that record lived in the manifest column that the sizing change
# retired; guessing it now picks _large for slots that shipped _medium and
# reports a figure nobody ever saw. Instead the before state is read from a
# probe taken before the change, which resolved the presets correctly.


def _resolve(root: Path, ref: str, stem: str, suffixes) -> Path | None:
    for suf in suffixes:
        p = root / ref / f"{stem}{suf}.png"
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path,
                    default=Path("tools/additional_figures_manifest.txt"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--width-in", type=float, default=PLACEMENT_WIDTH_IN)
    ap.add_argument("--thumb-px", type=int, default=900)
    ap.add_argument("--before", type=Path, required=True,
                    help="layout_probe.json captured before the sizing change")
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()

    rows = [l.split("|") for l in a.manifest.read_text().splitlines()
            if l.strip() and not l.startswith("#")]

    before = {f["name"].split("_")[0]: f
              for f in json.loads(a.before.read_text())["figures"]}

    slots = []
    for slot, ref, stem in rows:
        new_p = _resolve(a.results, ref, stem, NEW_SUFFIXES)
        entry = {"slot": slot, "stem": stem, "ref": ref,
                 "before": before.get(slot)}
        if entry["before"]:
            entry["before"].pop("thumb_bytes", None)
        if new_p is None:
            entry["after"] = None
        else:
            d = probe(new_p, a.width_in, a.thumb_px, new_p)
            d.pop("thumb_bytes", None)
            entry["after"] = d
        entry["regenerated"] = new_p is not None
        slots.append(entry)
        b = entry["before"]["summary"] if entry["before"] else None
        n = entry["after"]["summary"] if entry["after"] else None
        f = lambda s: f"{s['pt50']:>6} pt  g:t {s['gt']:>5}" if s else "        --        "
        print(f"  {slot:<6}{f(b)}   ->  {f(n)}"
              f"{'' if entry['regenerated'] else '   (not yet rebuilt)'}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"width_in": a.width_in, "slots": slots}))
    done = sum(1 for s in slots if s["regenerated"])
    print(f"\n  {done}/{len(slots)} rebuilt -> {a.out} "
          f"({a.out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
