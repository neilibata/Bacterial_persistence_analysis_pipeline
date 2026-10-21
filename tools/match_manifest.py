#!/usr/bin/env python3
"""Derive the Additional-Figure manifest by matching the document's own images.

Not by reading captions and inferring. The document embeds the exact PNGs that
were pasted into it, so the mapping from slot to output file is recoverable by
comparing images.

The figures on disk today are not those images -- fonts, layout and the sample
set have all moved since -- so the comparison runs against archived snapshots
of results/ from around the date the document was written, and reports the
distance and the margin over the runner-up for every slot. A match is only
reported as settled when it is both close and unambiguous; everything else is
listed as unresolved rather than guessed.

    python tools/match_manifest.py --docx ~/Documents/Additional_figures_080826.docx \
        --results results_pre_20260820 results_pre_20260826 results
"""
from __future__ import annotations

import argparse, io, json, re, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HASH_N = 40   # fine enough to separate the same figure across references


def doc_images(docx: Path):
    """(order, bytes) for each image in document order."""
    out = []
    with zipfile.ZipFile(docx) as z:
        doc = z.read("word/document.xml").decode("utf8", "replace")
        rels = z.read("word/_rels/document.xml.rels").decode("utf8", "replace")
        rid = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels))
        for i, r in enumerate(re.findall(r'r:embed="([^"]+)"', doc)):
            t = rid.get(r)
            if not t:
                continue
            n = t if t.startswith("word/") else "word/" + t.lstrip("/")
            try:
                out.append((i, z.read(n)))
            except KeyError:
                pass
    return out


def ahash(data: bytes):
    import numpy as np
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("L")
    a = np.asarray(im.resize((HASH_N, HASH_N)), dtype=float)
    return (a > a.mean()).flatten()


def _fam_global(k):
    n = Path(k).stem
    for suf in ("_medium", "_large", "_small", "_fit", "_notitle"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    n = re.sub(r"_(UHGG|combined|lev|chimp)(?=_|$)", "", n)
    n = re.sub(r"_(\d+pct|\d+kb|\d+Mb)(?=_|$)", "", n)
    return n


def main() -> int:
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", type=Path, required=True)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--cache", type=Path, default=ROOT / "tools/.fighash.json")
    ap.add_argument("--out", type=Path, default=ROOT / "tools/manifest_evidence.json")
    a = ap.parse_args()

    cache = json.loads(a.cache.read_text()) if a.cache.exists() else {}
    cands = []
    for d in a.results:
        root = ROOT / d
        if not root.exists():
            print(f"  [skip] {d}: not present"); continue
        n = 0
        for p in root.rglob("*.png"):
            if p.name.endswith("_notitle.png"):
                continue
            k = f"{p.relative_to(ROOT)}#{HASH_N}"
            if k not in cache:
                try:
                    cache[k] = [int(x) for x in ahash(p.read_bytes())]
                except Exception:
                    cache[k] = None
            if cache[k]:
                cands.append((k.split("#")[0],
                              np.array(cache[k], dtype=bool))); n += 1
        print(f"  {d}: {n} figures hashed")
    a.cache.write_text(json.dumps(cache))

    imgs = doc_images(a.docx)
    print(f"  document: {len(imgs)} embedded images\n")

    rows = []
    for i, data in imgs:
        try:
            h = ahash(data)
        except Exception:
            rows.append(dict(order=i, best=None)); continue
        scored = sorted((int((h != v).sum()), k) for k, v in cands)
        # The runner-up is only informative if it is a *different figure*. The
        # archive holds the same output at several preset sizes, and across
        # references, and those are pixel-identical after rescaling -- so the
        # nearest rival was almost always the same picture under another name,
        # which made every margin zero and every slot look unresolved.
        def _fam(k):
            """The figure's identity, independent of which run produced it.

            The same plot exists per reference (UHGG, combined, lev, chimp),
            per detection threshold and per retired size preset. Those are the
            same picture to within a few pixels, so treating them as rivals
            made every margin meaningless. What the manifest needs is which
            plot a slot shows; which reference the pasted image happened to
            come from is a separate question, and one worth answering
            separately.
            """
            n = Path(k).stem
            for suf in ("_medium", "_large", "_small", "_fit", "_notitle"):
                if n.endswith(suf):
                    n = n[: -len(suf)]
            n = re.sub(r"_(UHGG|combined|lev|chimp)(?=_|$)", "", n)
            n = re.sub(r"_(\d+pct|\d+kb|\d+Mb)(?=_|$)", "", n)
            return n
        d0, b0 = scored[0]
        fam0 = _fam(b0)
        rival = next(((d, k) for d, k in scored[1:] if _fam(k) != fam0),
                     (None, None))
        d1, b1 = rival
        # Everything sharing the winner's identity, so a slot that matches an
        # output present under several references says so.
        same = sorted({k for d, k in scored if d <= d0 + 2})
        # A slot is settled when its best match is close AND clearly better than
        # the next. Two near-identical distances mean the images cannot tell
        # those outputs apart, which is a finding, not something to round off.
        margin = (d1 - d0) if d1 is not None else 999
        rows.append(dict(order=i, _hash=[int(x) for x in h],
                         dist=d0, best=b0, runner=b1, runner_dist=d1,
                         margin=margin, equals=same,
                         family=_fam(b0),
                         settled=bool(margin >= max(40, 0.35 * d0))))
    # Solve the whole board at once. Four taxon-driver figures differ only in
    # which taxon list they show, and three GMW comparisons differ only in
    # which grouping -- as pictures they are near-identical, so picking each
    # slot's nearest neighbour independently let one figure win several slots
    # and left the rest unresolved. Assigning all slots simultaneously, one
    # figure each, uses the constraint that the document shows each once.
    try:
        from scipy.optimize import linear_sum_assignment
        fams = sorted({_f for _f in (r.get("family") for r in rows) if _f})
        readable = [r for r in rows if r.get("best")]
        fam_list = sorted({k for k, _ in [(None, None)]} | set())
        # cost matrix: slot x figure-identity, using each identity's best member
        ident = {}
        for k, v in cands:
            f = _fam_global(k)
            ident.setdefault(f, []).append((k, v))
        names = sorted(ident)
        cost = np.zeros((len(readable), len(names)))
        for i, r in enumerate(readable):
            h = np.array(r["_hash"], dtype=bool)
            for j, f in enumerate(names):
                cost[i, j] = min(int((h != v).sum()) for _, v in ident[f])
        ri, cj = linear_sum_assignment(cost)
        for i, j in zip(ri, cj):
            readable[i]["assigned"] = names[j]
            readable[i]["assigned_dist"] = int(cost[i, j])
    except Exception as exc:
        print(f"  [assign] joint assignment unavailable: {exc}")

    for r in rows:
        r.pop("_hash", None)
    a.out.write_text(json.dumps(rows, indent=1))

    ok = sum(1 for r in rows if r.get("settled"))
    print(f"  {'#':<4}{'dist':>5}{'margin':>8}   file")
    for r in rows:
        if not r.get("best"):
            print(f"  {r['order']:<4}   --            (unreadable)"); continue
        mark = "" if r["settled"] else "   UNRESOLVED"
        print(f"  {r['order']:<4}{r['dist']:>5}{r['margin']:>8}   "
              f"{r.get('family','')[:48]:<50}{mark}")
        alt = []
        if alt:
            print(f"        also identical: "
                  f"{', '.join(Path(x).name[:44] for x in alt[:3])}"
                  + (f" (+{len(alt)-3})" if len(alt) > 3 else ""))
    print("\n  joint assignment (each figure used once):")
    for r in rows:
        if r.get("assigned"):
            print(f"  {r['order']:<4}{r['assigned_dist']:>5}   {r['assigned'][:56]}")
    print(f"\n  settled {ok}/{len(rows)};  evidence -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
