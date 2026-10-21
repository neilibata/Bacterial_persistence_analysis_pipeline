#!/usr/bin/env python3
"""Build docs/figure_review.html: the measured figures and the caption debt.

One page, because they are one job. The sweep says which figures still have
collisions; the caption list says what was taken out of a figure and has to
reappear in its caption. Regenerate after any sweep:

    python tools/build_figure_review.py --sweep <log> [--out docs/figure_review.html]
"""
from __future__ import annotations
import argparse, csv, html, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_sweep(log_path: Path):
    log = log_path.read_text().splitlines()
    rows, in_tab = [], False
    for l in log:
        if l.startswith("figure "):
            in_tab = True; continue
        if in_tab:
            if not l.strip():
                in_tab = False; continue
            p = l.split()
            if len(p) >= 7:
                rows.append(dict(name=p[0], panels=int(p[1]), canvas=float(p[2]),
                                 saved=float(p[3]), base=float(p[4]),
                                 printed=float(p[5]), overlaps=int(p[6])))
    detail, cur = {}, None
    for l in log:
        m = re.match(r"^  ([a-z0-9_]+)$", l)
        if m:
            cur = m.group(1); detail.setdefault(cur, []); continue
        m = re.match(r"^\s+([0-9.]+)\s+'(.*)' x '(.*)'$", l)
        if m and cur:
            detail[cur].append((float(m.group(1)), m.group(2), m.group(3)))
    for r in rows:
        r["detail"] = detail.get(r["name"], [])
    seen, uniq = set(), []
    for r in sorted(rows, key=lambda r: -r["overlaps"]):
        if r["name"] in seen:
            continue
        seen.add(r["name"]); uniq.append(r)
    return uniq


def slot_map():
    m = {}
    f = ROOT / "tools/additional_figures_manifest.txt"
    for l in f.read_text().splitlines():
        if l and not l.startswith("#"):
            a, _ref, stem = l.split("|")
            m[stem.strip().split("/")[-1]] = a
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, nargs="+", required=True,
                    help="one or more sweep logs; later ones win per figure")
    ap.add_argument("--ref", default="combined")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/figure_review.html")
    a = ap.parse_args()

    # Several logs, newest last. A sweep of three sections should refresh those
    # three and leave the rest of the table standing, rather than shrinking the
    # page to whatever the last run happened to cover.
    merged, order = {}, []
    for log in a.sweep:
        for r in parse_sweep(log):
            if r["name"] not in merged:
                order.append(r["name"])
            merged[r["name"]] = r
    rows = [merged[n] for n in order]
    MAN = slot_map()

    def _slot(name):
        if name in MAN:
            return MAN[name]
        # The sweep table truncates long names, so a prefix match is needed --
        # but only for names it actually truncated. Applied to short names it
        # matched figure_robustness_alpha_hadza to figure_robustness_alpha and
        # reported three internal variants as the shipping figure.
        if len(name) >= 39:
            for stem, slot in MAN.items():
                if stem.startswith(name):
                    return slot
        return ""

    res = ROOT / "results" / a.ref
    for r in rows:
        hits = sorted(res.rglob(f"{r['name']}*.png"))
        hits = [h for h in hits if not h.name.endswith("_notitle.png")]
        r["path"] = (str(hits[0].relative_to(ROOT)) if hits else None)
        r["slot"] = _slot(r["name"])

    debt = list(csv.DictReader(open(ROOT / "tools/caption_debt.csv")))
    by_slot: dict = {}
    for d in debt:
        by_slot.setdefault(d["slot"], []).append(d)
    n_open = sum(1 for d in debt if d["status"] == "open")

    def esc(x): return html.escape(str(x))

    debt_rows = "".join(
        f'<tr class="{d["status"]}"><th scope="row">{esc(d["slot"])}</th>'
        f'<td class="mono">{esc(d["figure"])}</td>'
        f'<td>{esc(d["removed"])}</td>'
        f'<td class="say">{esc(d["caption_must_say"])}</td>'
        f'<td class="st">{esc(d["status"])}</td></tr>'
        for d in debt)

    # The run measures every figure a section writes, and most of them are
    # internal diagnostics that never reach the document. Listing 130 figures
    # buries the 30 that ship, so the manifest comes first and the rest goes
    # behind a fold.
    ship = [r for r in rows if r["slot"]]
    internal = [r for r in rows if not r["slot"]]
    ship.sort(key=lambda r: (-r["overlaps"], r["printed"]))
    internal.sort(key=lambda r: -r["overlaps"])

    cards = []
    for r in ship + internal:
        sev = "bad" if r["overlaps"] >= 10 else ("warn" if r["overlaps"] else "ok")
        psev = "bad" if r["printed"] < 5 else ("warn" if r["printed"] < 6 else "ok")
        grew = r["saved"] > r["canvas"] * 1.05
        dets = "".join(
            f'<li><span class="frac">{f:.0%}</span> <code>{esc(x)}</code>'
            f' <span class="x">over</span> <code>{esc(y)}</code></li>'
            for f, x, y in r["detail"][:6])
        mine = by_slot.get(r["slot"], [])
        capt = "".join(
            f'<li class="{d["status"]}">{esc(d["caption_must_say"])}</li>'
            for d in mine)
        cards.append(f'''<article id="{esc(r["name"])}">
  <header><h2>{esc(r["name"])}{f' <span class="slot">{esc(r["slot"])}</span>' if r["slot"] else ''}</h2>
  <dl>
    <div><dt>panels</dt><dd>{r["panels"]}</dd></div>
    <div><dt>canvas</dt><dd>{r["canvas"]} in</dd></div>
    <div><dt>saved</dt><dd class="{'warn' if grew else ''}">{r["saved"]} in</dd></div>
    <div><dt>base</dt><dd>{r["base"]} pt</dd></div>
    <div><dt>printed</dt><dd class="{psev}">{r["printed"]} pt</dd></div>
    <div><dt>overlaps</dt><dd class="{sev}">{r["overlaps"]}</dd></div>
  </dl></header>
  {'<p class="note">Saved %s in from a %s in canvas. Content is spilling outside the axes, so the figure shrinks by that ratio when placed at column width.</p>' % (r["saved"], r["canvas"]) if grew else ''}
  {f'<div class="capt"><h3>Caption must carry</h3><ul>{capt}</ul></div>' if capt else ''}
  {f'<ul class="hits">{dets}</ul>' if dets else ''}
  {f'<a href="../{r["path"]}" target="_blank"><img src="../{r["path"]}" loading="lazy" alt=""></a><p class="path"><code>{esc(r["path"])}</code></p>' if r["path"] else '<p class="note">No file found under results/.</p>'}
</article>''')

    n_ship_cards = len(ship)
    cards = (cards[:n_ship_cards]
             + ['<details><summary class="fold">Internal diagnostics '
                f'({len(internal)} figures not in the document)</summary>']
             + cards[n_ship_cards:] + ["</details>"]) if internal else cards
    clean = sum(1 for r in ship if r["overlaps"] == 0)
    doc = f'''<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Figure review</title>
<style>
:root{{--paper:#f6f5f1;--card:#fff;--ink:#1b211f;--ink2:#5c6360;--rule:#dedbd3;
 --bad:#9d3320;--warn:#a8620a;--ok:#3f7a52;--chip:#eceae3}}
@media (prefers-color-scheme:dark){{:root{{--paper:#14181a;--card:#1c2225;--ink:#e9eae6;
 --ink2:#9aa39f;--rule:#2b3337;--bad:#e08672;--warn:#e0a84a;--ok:#84c299;--chip:#242c30}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:36px 22px 80px}}
h1{{font-size:28px;margin:0 0 6px;letter-spacing:-.02em}}
.lede{{color:var(--ink2);margin:0 0 10px;max-width:72ch}}
h2.sec{{font-size:12px;letter-spacing:.13em;text-transform:uppercase;color:var(--ink2);
 margin:40px 0 14px;padding-bottom:8px;border-bottom:1px solid var(--rule)}}
article{{background:var(--card);border:1px solid var(--rule);margin-bottom:26px;padding:18px 20px}}
article h2{{font-size:16px;margin:0;font-family:ui-monospace,Menlo,monospace;font-weight:600}}
.slot{{background:var(--chip);color:var(--ink2);font-size:11px;padding:2px 7px;border-radius:3px;
 margin-left:8px;font-family:-apple-system,sans-serif;letter-spacing:.06em}}
dl{{display:flex;flex-wrap:wrap;gap:0 26px;margin:12px 0 0}}
dl div{{display:flex;gap:7px;align-items:baseline}}
dt{{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--ink2)}}
dd{{margin:0;font-family:ui-monospace,monospace;font-variant-numeric:tabular-nums;font-size:14px}}
dd.bad{{color:var(--bad);font-weight:700}} dd.warn{{color:var(--warn);font-weight:600}}
dd.ok{{color:var(--ok)}}
.note{{margin:12px 0 0;padding:9px 12px;background:var(--chip);font-size:13.5px;color:var(--ink2)}}
.capt{{margin-top:14px;padding:11px 14px;border-left:3px solid var(--warn);background:var(--chip)}}
.capt h3{{margin:0 0 6px;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--warn)}}
.capt ul{{margin:0;padding-left:18px;font-size:13.5px}}
.capt li{{padding:2px 0}} .capt li.resolved{{color:var(--ink2);text-decoration:line-through}}
.hits{{list-style:none;margin:12px 0 0;padding:0;font-size:13px}}
.hits li{{padding:3px 0;border-bottom:1px solid var(--rule)}} .hits li:last-child{{border-bottom:0}}
.frac{{display:inline-block;min-width:44px;font-family:ui-monospace,monospace;color:var(--bad);font-weight:600}}
.x{{color:var(--ink2);font-size:12px}}
code{{background:var(--chip);padding:.08em .34em;border-radius:3px;font-family:ui-monospace,monospace;font-size:.88em}}
img{{width:100%;margin-top:16px;border:1px solid var(--rule);background:#fff;display:block}}
.path{{margin:8px 0 0;font-size:11.5px;color:var(--ink2);word-break:break-all}}
table{{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--rule);font-size:13.5px}}
th,td{{padding:8px 11px;text-align:left;border-bottom:1px solid var(--rule);vertical-align:top}}
thead th{{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--ink2);
 border-bottom:1px solid var(--rule)}}
tbody th{{font-family:ui-monospace,monospace;font-weight:600;white-space:nowrap}}
td.mono{{font-family:ui-monospace,monospace;font-size:12px;color:var(--ink2)}}
td.say{{min-width:30ch}}
td.st{{font-size:11px;letter-spacing:.06em;text-transform:uppercase}}
tr.resolved{{color:var(--ink2)}} tr.resolved td.st{{color:var(--ok)}}
tr.open td.st{{color:var(--warn);font-weight:600}}
.fold{{cursor:pointer;font-size:13px;letter-spacing:.06em;text-transform:uppercase;
 color:var(--ink2);padding:14px 0;border-top:1px solid var(--rule);margin-top:20px}}
.count{{display:inline-block;background:var(--chip);padding:3px 10px;border-radius:3px;
 font-family:ui-monospace,monospace;font-size:13px;margin-right:8px}}
</style>
<div class="wrap">
<h1>Figure review</h1>
<p class="lede" style="font-size:13.5px">Measurements merged from: {", ".join(x.name for x in a.sweep)}.
Images are whatever is currently in <code>results/</code>, so a figure redrawn since its
last measurement shows the new picture beside the older numbers.</p>
<p class="lede">Two lists that belong together. What the sweep measured on each figure as the
pipeline drew it, and what was taken out of a figure to make it legible and now has to
reappear in its caption.</p>
<p><span class="count">{len(ship)} supplementary figures</span><span class="count">{clean} clean</span><span class="count">{len(internal)} internal</span>
<span class="count">{n_open} caption items open</span></p>

<h2 class="sec">Caption debt</h2>
<p class="lede">Text removed from a figure during the layout work. Each line is something a
reader needed and can no longer get from the figure itself.</p>
<table><thead><tr><th>Slot</th><th>Figure</th><th>Removed</th><th>Caption must say</th><th>Status</th></tr></thead>
<tbody>{debt_rows}</tbody></table>

<h2 class="sec">Measured figures</h2>
<p class="lede">Worst first. Printed size is what the type measures once the saved canvas is
placed in a 6.3 inch column. Click a figure to open it full size.</p>
{"".join(cards)}
</div>'''
    a.out.write_text(doc)
    print(f"  wrote {a.out}  ({len(ship)} supplementary figures, {clean} clean, "
          f"{len(internal)} internal, {n_open} open caption items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
