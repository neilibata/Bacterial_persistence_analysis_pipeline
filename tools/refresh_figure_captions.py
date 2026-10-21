#!/usr/bin/env python3
"""Refresh tools/additional_figure_captions.json from the Word document.

The Word document is the source of truth for the supplementary figure
captions. This reads the captions out of it, keyed by slot number, and writes
them to `tools/additional_figure_captions.json`, which the caption-checking
tools read instead of reopening the document.

`--check` compares the document against the JSON and reports the differences
without writing, so it works as a gate: any caption edited in the document
shows up as a difference until the JSON is refreshed from it.

    python tools/refresh_figure_captions.py ~/Documents/Persistence_Additional_figures_100926.docx
    python tools/refresh_figure_captions.py <docx> --check     # report only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
CAP_RE = re.compile(r"^\s*Figure\s+S(\d+)\.\s", re.I)


def captions_from_docx(path: Path) -> dict[int, str]:
    """{slot number: full caption text}, read in document order."""
    root = ET.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    out: dict[int, str] = {}
    for p in root.iter(W + "p"):
        txt = "".join(t.text or "" for t in p.iter(W + "t")).strip()
        m = CAP_RE.match(txt)
        if m:
            out.setdefault(int(m.group(1)), re.sub(r"\s+", " ", txt))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("docx", type=Path)
    ap.add_argument("--json", type=Path,
                    default=Path(__file__).with_name("additional_figure_captions.json"))
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit non-zero; change nothing")
    a = ap.parse_args()

    caps = captions_from_docx(a.docx)
    if not caps:
        print(f"no 'Figure S<n>.' captions found in {a.docx}", file=sys.stderr)
        return 2
    old = json.loads(a.json.read_text()) if a.json.exists() else {}

    drift, missing = [], []
    for n in sorted(caps):
        prev = (old.get(str(n), {}) or {}).get("display", "")
        # The JSON predates the A->S rename, so compare on the body.
        norm = lambda s: re.sub(r"^\s*Figure\s+[AS]\s*\d+\.\s*", "", s or "").strip()
        if norm(prev) != norm(caps[n]):
            drift.append(n)
    for n in sorted(set(map(int, old)) - set(caps)):
        missing.append(n)

    print(f"{len(caps)} captions in {a.docx.name}; "
          f"{len(drift)} differ from the JSON"
          + (f"; {len(missing)} JSON slots not in the document: {missing}" if missing else ""))
    if drift:
        print("  drifted slots:", ", ".join(f"S{n}" for n in drift))
    if a.check:
        return 1 if (drift or missing) else 0

    a.json.write_text(json.dumps(
        {str(n): {"verbatim": caps[n], "display": caps[n]} for n in sorted(caps)},
        indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {a.json} ({len(caps)} slots, S-numbering)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
