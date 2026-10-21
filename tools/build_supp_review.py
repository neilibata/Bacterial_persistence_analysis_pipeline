#!/usr/bin/env python3
"""Build docs/supplementary_figures.html: every shipped figure, side by side
with the version redrawn from its published data object.

The grid shows downscaled previews so the page opens quickly; the full-
resolution PNG is one click away, which is what you need to judge type size
and collisions. Regenerate after a figure run:

    python tools/build_supp_review.py
"""
from __future__ import annotations

import argparse
import datetime
import html
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence import figure_source          # noqa: E402
from tools import figure_sets                     # noqa: E402

Image.MAX_IMAGE_PIXELS = None
PREVIEW_W = 900


def slot_num(slot):
    return int(re.sub(r"\D", "", slot))


def preview(src: Path, dest: Path, width=PREVIEW_W):
    """Write a downscaled JPEG preview; return its (w, h) or None."""
    if not src.exists():
        return None
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        with Image.open(dest) as im:
            return im.size
    with Image.open(src) as im:
        im = im.convert("RGB")
        h = max(1, round(width * im.size[1] / im.size[0]))
        im.resize((width, h), Image.LANCZOS).save(dest, "JPEG", quality=86,
                                                  optimize=True)
        return (width, h)


def compare(a_path: Path, b_path: Path):
    """Mean absolute pixel difference, or a note on why there isn't one."""
    if not (a_path.exists() and b_path.exists()):
        return None, "not compared"
    with Image.open(a_path) as ia, Image.open(b_path) as ib:
        if ia.size != ib.size:
            return None, f"canvas {ia.size[0]}x{ia.size[1]} vs {ib.size[0]}x{ib.size[1]}"
        a = np.asarray(ia.convert("RGB"), dtype=float)
        b = np.asarray(ib.convert("RGB"), dtype=float)
    return float(np.abs(a - b).mean()), ""


def badge(diff, note):
    if diff is None:
        return ("differs", note or "not compared")
    if diff == 0.0:
        return ("exact", "pixel-identical")
    if diff < 3.0:
        return ("close", f"{diff:.1f}/255")
    return ("loose", f"{diff:.1f}/255")


CSS = """
:root{
  --ink:#15171c; --ink-soft:#5a6070; --ink-faint:#8d94a5;
  --bg:#f6f5f2; --card:#fffefb; --line:#e2ded6;
  --accent:#8c4a2f; --accent-soft:#f0e4dd;
  --exact:#2f6d4f; --exact-bg:#e3efe8;
  --close:#7a6320; --close-bg:#f4eed9;
  --loose:#8c4a2f; --loose-bg:#f4e4dc;
  --shadow:0 1px 2px rgba(20,18,14,.05),0 6px 18px rgba(20,18,14,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme=light]){
    --ink:#e9e6df; --ink-soft:#a7a396; --ink-faint:#7c7869;
    --bg:#16171a; --card:#1e2024; --line:#2e3138;
    --accent:#d98b63; --accent-soft:#3a2a22;
    --exact:#7fc4a0; --exact-bg:#1f3329;
    --close:#d6bf72; --close-bg:#332d1a;
    --loose:#d98b63; --loose-bg:#33241d;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 6px 18px rgba(0,0,0,.28);
  }
}
:root[data-theme=dark]{
  --ink:#e9e6df; --ink-soft:#a7a396; --ink-faint:#7c7869;
  --bg:#16171a; --card:#1e2024; --line:#2e3138;
  --accent:#d98b63; --accent-soft:#3a2a22;
  --exact:#7fc4a0; --exact-bg:#1f3329;
  --close:#d6bf72; --close-bg:#332d1a;
  --loose:#d98b63; --loose-bg:#33241d;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 6px 18px rgba(0,0,0,.28);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1280px;margin:0 auto;padding:40px 28px 80px}
header{border-bottom:1px solid var(--line);padding-bottom:24px;margin-bottom:28px}
h1{margin:0 0 6px;font-size:26px;letter-spacing:-.015em;font-weight:640;text-wrap:balance}
.sub{color:var(--ink-soft);font-size:14px;margin:0}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;
  padding:9px 13px;font-size:13px;box-shadow:var(--shadow)}
.stat b{font-variant-numeric:tabular-nums;font-weight:650}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;
  margin:22px 0 4px;font-size:13px}
.controls label{display:flex;align-items:center;gap:6px;color:var(--ink-soft);
  cursor:pointer;user-select:none}
.controls input[type=search]{flex:1;min-width:180px;padding:8px 11px;
  border:1px solid var(--line);border-radius:7px;background:var(--card);
  color:var(--ink);font-size:13px}
.controls input[type=search]:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.grid{display:grid;gap:22px;margin-top:22px;
  grid-template-columns:repeat(auto-fill,minmax(340px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:11px;
  overflow:hidden;box-shadow:var(--shadow);display:flex;flex-direction:column}
.shot{display:block;background:#fff;border-bottom:1px solid var(--line);
  position:relative;line-height:0}
.shot img{width:100%;height:auto;display:block}
.shot:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.meta{padding:13px 15px 15px;display:flex;flex-direction:column;gap:9px;flex:1}
.row1{display:flex;align-items:center;gap:9px}
.slot{font-weight:680;font-size:14px;letter-spacing:-.01em}
.tag{font-size:11px;padding:2px 7px;border-radius:99px;font-weight:600;
  letter-spacing:.02em;white-space:nowrap}
.exact{color:var(--exact);background:var(--exact-bg)}
.close{color:var(--close);background:var(--close-bg)}
.loose,.differs{color:var(--loose);background:var(--loose-bg)}
.name{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;
  color:var(--ink-soft);word-break:break-all;line-height:1.45}
.tables{font-size:12px;color:var(--ink-faint);margin-top:auto;padding-top:4px;
  border-top:1px dashed var(--line)}
.tables b{color:var(--ink-soft);font-weight:600;font-variant-numeric:tabular-nums}
.which{font-size:11px;color:var(--ink-faint);display:flex;gap:8px;align-items:center}
.which button{font:inherit;font-size:11px;padding:2px 8px;border-radius:6px;
  border:1px solid var(--line);background:transparent;color:var(--ink-soft);cursor:pointer}
.which button[aria-pressed=true]{background:var(--accent-soft);color:var(--accent);
  border-color:var(--accent)}
.which button:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.missing{padding:36px 15px;text-align:center;color:var(--ink-faint);font-size:13px}
footer{margin-top:44px;padding-top:20px;border-top:1px solid var(--line);
  color:var(--ink-faint);font-size:12.5px}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;
  background:var(--accent-soft);color:var(--accent);padding:1px 5px;border-radius:4px}
@media (prefers-reduced-motion:no-preference){
  .card{transition:border-color .15s ease}
}
.card:hover{border-color:var(--ink-faint)}
"""

JS = """
const q = document.getElementById('q');
const onlyDiff = document.getElementById('onlyDiff');
function apply(){
  const t = (q.value||'').toLowerCase();
  const d = onlyDiff.checked;
  document.querySelectorAll('.card').forEach(c=>{
    const hay = (c.dataset.slot+' '+c.dataset.name+' '+c.dataset.tables).toLowerCase();
    const okText = !t || hay.includes(t);
    const okDiff = !d || c.dataset.fidelity !== 'exact';
    c.hidden = !(okText && okDiff);
  });
}
q.addEventListener('input', apply);
onlyDiff.addEventListener('change', apply);

document.querySelectorAll('.which').forEach(w=>{
  w.addEventListener('click', e=>{
    const b = e.target.closest('button'); if(!b) return;
    const card = w.closest('.card');
    const img = card.querySelector('.shot img');
    const link = card.querySelector('.shot');
    // data-previewship -> dataset.previewship: the attribute name is
    // lowercased, so the key has to be too.
    const which = b.dataset.which.toLowerCase();
    w.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed', String(x===b)));
    const src = card.dataset['preview'+which], full = card.dataset['full'+which];
    if(!src) return;
    img.src = src;
    if(full) link.href = full;
  });
});
"""


def build(out_path: Path, redraw_dir: Path, compare_label="redrawn"):
    assets = out_path.parent / "figure_review_assets"
    assets.mkdir(parents=True, exist_ok=True)

    cards, n_exact, n_close, n_loose, n_rows = [], 0, 0, 0, 0
    for slot, ref, stem in sorted(figure_sets.manifest(),
                                  key=lambda r: slot_num(r[0])):
        name = Path(stem).name
        shipped = ROOT / "results" / ref / (stem + ".png")
        redrawn = next((c for c in (redraw_dir / f"{slot}_{name}.png",
                                    redraw_dir / f"{name}.png") if c.exists()),
                       None)

        data_path = (ROOT / "results" / ref / Path(stem).parent
                     / figure_source.DATA_DIRNAME / f"Figure_{slot}_data.pkl")
        tables, rows = "-", 0
        if data_path.exists():
            try:
                obj = figure_source.load(str(data_path.parent.parent), slot)
                tables = ", ".join(obj["panels"])
                rows = sum(len(v) for v in obj["panels"].values())
            except Exception:
                pass
        n_rows += rows

        p_ship = preview(shipped, assets / f"{slot}_shipped.jpg")
        p_draw = preview(redrawn, assets / f"{slot}_redrawn.jpg") if redrawn else None

        diff, note = compare(shipped, redrawn) if redrawn else (None, "no redraw")
        kind, label = badge(diff, note)
        n_exact += kind == "exact"
        n_close += kind == "close"
        n_loose += kind in ("loose", "differs")

        if p_ship is None:
            body = '<div class="missing">figure not on disk</div>'
        else:
            body = (f'<a class="shot" href="{html.escape(str(shipped))}" '
                    f'target="_blank" rel="noopener" '
                    f'title="Open the full-resolution PNG">'
                    f'<img loading="lazy" src="figure_review_assets/{slot}_shipped.jpg" '
                    f'alt="Figure {slot}: {html.escape(name)}"></a>')

        toggle = ""
        if p_draw is not None:
            toggle = ('<div class="which">view'
                      '<button data-which="Ship" aria-pressed="true">current</button>'
                      f'<button data-which="Draw" aria-pressed="false">{compare_label}</button>'
                      '</div>')

        cards.append(f'''
<article class="card" data-slot="{slot}" data-name="{html.escape(name)}"
  data-tables="{html.escape(tables)}" data-fidelity="{kind}"
  data-previewship="figure_review_assets/{slot}_shipped.jpg"
  data-previewdraw="figure_review_assets/{slot}_redrawn.jpg"
  data-fullship="{html.escape(str(shipped))}"
  data-fulldraw="{html.escape(str(redrawn)) if redrawn else ''}">
  {body}
  <div class="meta">
    <div class="row1"><span class="slot">{slot}</span>
      <span class="tag {kind}">{html.escape(label)}</span></div>
    <div class="name">{html.escape(name)}</div>
    {toggle}
    <div class="tables"><b>{rows:,}</b> rows &middot; {html.escape(tables)}</div>
  </div>
</article>''')

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    page = f'''<title>Supplementary Figure Review</title>
<style>{CSS}</style>
<div class="wrap">
<header>
  <h1>Supplementary Figure Review</h1>
  <p class="sub">All 32 shipped figures, each beside its {compare_label} counterpart. Click a figure to open it at full resolution.</p>
  <div class="stats">
    <div class="stat"><b>32</b> figures</div>
    <div class="stat"><b>{n_rows:,}</b> published rows</div>
    <div class="stat"><b>{n_exact}</b> pixel-identical</div>
    <div class="stat"><b>{n_close}</b> under 3/255</div>
    <div class="stat"><b>{n_loose}</b> looser</div>
    <div class="stat">built {stamp}</div>
  </div>
  <div class="controls">
    <input id="q" type="search" placeholder="Filter by number, filename or table…"
           aria-label="Filter figures">
    <label><input id="onlyDiff" type="checkbox"> only those not pixel-identical</label>
  </div>
</header>
<div class="grid">{''.join(cards)}</div>
<footer>
  Previews are downscaled to {PREVIEW_W}px; the link opens the original PNG.
  Redraws come from <code>python tools/replot_from_data.py --all</code>.
  Figures that are not pixel-identical are the ones whose drawing draws random
  numbers &mdash; bootstrap intervals, weighted resampling, permuted statistics
  &mdash; recomputed from the published rows rather than stored.
</footer>
</div>
<script>{JS}</script>
'''
    out_path.write_text(page)
    return n_exact, n_close, n_loose, n_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "supplementary_figures.html"))
    ap.add_argument("--redrawn", default="/tmp/rp_v3",
                    help="directory holding the figures to compare against")
    ap.add_argument("--compare-label", default="redrawn",
                    help="what the comparison set is, e.g. 'before format'")
    a = ap.parse_args()
    ex, cl, lo, rows = build(Path(a.out), Path(a.redrawn),
                             a.compare_label)
    print(f"  {a.out}")
    print(f"  {ex} pixel-identical, {cl} under 3/255, {lo} looser; {rows:,} rows")


if __name__ == "__main__":
    main()
