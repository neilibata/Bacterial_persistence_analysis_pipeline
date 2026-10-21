#!/usr/bin/env python3
"""Are the internal lifestyle keys reaching the page anywhere?

The code compares against H-G and Transitional; the published artefacts must
say Hunter-gatherer and Agro-pastoralist. config.display_text bridges them and
savefig_dual applies it, so a figure only shows the wrong label if it saves by
some other route -- which is invisible in the source and only shows up in the
rendered text.

This reads the text out of each cached figure, which is exactly what was drawn.

    python tools/audit_labels.py
"""
from __future__ import annotations

import pickle, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")

BAD = re.compile(r"\bH-G\b|\bTransitional\b")


def texts_of(fig, spec):
    out = []
    for r in spec.get("axes", []):
        for key in ("title", "xlabel", "ylabel"):
            if r.get(key):
                out.append(r[key])
        if r.get("legend"):
            out.extend(r["legend"].get("labels") or [])
        for t in r.get("texts", []) + r.get("annotations", []):
            out.append(t.get("text", ""))
    for k in ("suptitle", "supxlabel", "supylabel"):
        if spec.get(k):
            out.append(spec[k]["text"])
    out.extend(t["text"] for t in spec.get("figtexts", []))
    for ax in fig.axes:
        out.extend(t.get_text() for t in ax.get_xticklabels())
        out.extend(t.get_text() for t in ax.get_yticklabels())
    return [t for t in out if t]


def main() -> int:
    caches = sorted(ROOT.glob("results/**/.figcache/*.pkl"))
    if not caches:
        print("  no cached figures yet -- run a generation pass first")
        return 0
    bad = 0
    print(f"  scanning {len(caches)} cached figures\n")
    for pkl in caches:
        try:
            with pkl.open("rb") as f:
                blob = pickle.load(f)
            hits = sorted({t for t in texts_of(blob["figure"], blob["annotations"])
                           if BAD.search(t)})
        except Exception as exc:
            print(f"  {pkl.stem[:46]:<48} unreadable ({exc})")
            continue
        if hits:
            bad += 1
            print(f"  {pkl.stem[:46]:<48} {hits}")
    print(f"\n  {bad} of {len(caches)} figures carry an internal lifestyle key")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
