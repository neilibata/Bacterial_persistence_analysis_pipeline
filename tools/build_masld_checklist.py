#!/usr/bin/env python3
"""Track the NAFLD -> MASLD rename across the code, the figures and the drafts.

The rename is a display-label change: the internal key stays "Fatty Liver"
because it seeds DISEASE_NCD_MAP ordering and is a stored value in 100+ result
files. What moves is what a reader sees -- in figures, in Table S1A, and in the
manuscript text.

This regenerates docs/masld_rename_implementation.html from the live tree, so
the list of places still to change is current rather than a snapshot. Re-run it
after editing a draft or redrawing a figure.

    python tools/build_masld_checklist.py

Liver cirrhosis is deliberately out of scope: it is a separate condition with
its own metadata columns, it never reaches the Conditions column (the keys were
bound to names the metadata does not use), and its only appearance in the
manuscript is the title of a cited paper.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import html
import json
import pickle
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_HTML = ROOT / "docs" / "masld_rename_implementation.html"

#: (file, the string that proves the edit landed, what the edit is)
SOURCE_EDITS = [
    ("persistence/config.py", 'MASLD_DISPLAY = "MASLD"',
     "Declare the display constant beside the lifestyle ones"),
    ("persistence/config.py", '"Fatty Liver": MASLD_DISPLAY',
     "Add both forms to PLOT_TEXT_REPLACEMENTS, so every savefig_dual figure "
     "picks the label up at save time"),
    ("figures/figure5.py", '"Fatty Liver": "MASLD"',
     "Figure 5 saves with plt.savefig and never reaches the display layer, so "
     "its label map changes at source"),
    ("persistence/section3_health.py", '"Fatty Liver": "MASLD"',
     "The two wrap maps, so the label is right as drawn and not only as saved"),
    ("persistence/table_s1a.py", '"disease_fatty_liver":       "MASLD"',
     "The Table S1A condition label"),
]

#: things that must NOT change -- internal keys and metadata column names
KEEP = [
    ("persistence/config.py", "DISEASE_NCD_MAP key \"Fatty Liver\""),
    ("persistence/config.py", "disease_fatty_liver column name"),
    ("persistence/health_axis_projection.py", "\"Fatty Liver\" -> metabolic axis"),
    ("persistence/section3_health.py", "the condition list at line 2638"),
]

DOCS = {"main_text_v2": "Main text", "methods_v2": "Methods",
        "supplement_v2": "Supplementary Materials",
        "additional_figures_v2": "Additional figures"}

#: what still reads as the old name. Ordered longest-first so the fuller
#: phrases are reported rather than the bare noun inside them.
OLD = re.compile(
    r"non-alcoholic\s+fatty\s+liver\s+disease\s*\((?:metabolic|Metabolic)[^)]*\)"
    r"|non-alcoholic\s+fatty\s+liver(?:\s+disease)?"
    r"|NAFLD"
    r"|[Ff]atty\s*\\?n?\s*[Ll]iver(?:\s+disease)?",
    re.I)

#: The health block. S31/S32 were retired on 2026-10-09; tools/figure_sets.py
#: is the manifest and it ends at S30.
FIG_SLOTS = ("S25", "S26", "S27", "S28", "S29", "S30")


# ── state of the tree ─────────────────────────────────────────────────────
def source_state():
    rows = []
    for rel, proof, what in SOURCE_EDITS:
        p = ROOT / rel
        txt = p.read_text() if p.exists() else ""
        rows.append(dict(file=rel, what=what, done=proof in txt,
                         n=txt.count(proof)))
    return rows


def figure_state():
    """Which figures still print the old label, from the text harvest."""
    inv = ROOT / "tools" / "figure_text_inventory.csv"
    seen, shows = {}, set()
    if inv.exists():
        with inv.open() as fh:
            import csv
            for r in csv.DictReader(fh):
                if r.get("phase") != "final":
                    continue
                if re.search(r"fatty", r.get("text", ""), re.I):
                    seen.setdefault(r["slot"], []).append(r["text"][:60])
                if "MASLD" in r.get("text", ""):
                    shows.add(r["slot"])
    rows = []
    for slot in FIG_SLOTS:
        rows.append(dict(slot=slot, stale=seen.get(slot, []),
                         shows=slot in shows, measured=inv.exists()))
    return rows, inv.exists()


def data_objects():
    """Data objects that store the internal key (they should -- it is a key)."""
    def carries(o, d=0):
        if d > 6:
            return False
        if isinstance(o, str):
            return "Fatty" in o
        if isinstance(o, pd.DataFrame):
            if any("Fatty" in str(c) for c in o.columns):
                return True
            for c in o.columns:
                try:
                    if o[c].astype(str).str.contains("Fatty", na=False).any():
                        return True
                except Exception:
                    pass
            return False
        if isinstance(o, pd.Series):
            return o.astype(str).str.contains("Fatty", na=False).any()
        if isinstance(o, dict):
            return any(carries(k, d + 1) or carries(v, d + 1)
                       for k, v in o.items())
        if isinstance(o, (list, tuple, set)):
            return any(carries(x, d + 1) for x in o)
        return False

    hits = []
    for f in sorted(glob.glob(str(ROOT / "results/**/figure_data/*.pkl"),
                              recursive=True)):
        try:
            o = pickle.load(open(f, "rb"))
        except Exception:
            continue
        if carries(o):
            hits.append(str(Path(f).relative_to(ROOT)))
    return hits


def manuscript_occurrences():
    """Every remaining old-name mention, per document, in document order."""
    out = []

    md = ROOT / "manuscript" / "main_text.md"
    if md.exists():
        lines = md.read_text(errors="ignore").splitlines()
        for i, line in enumerate(lines, 1):
            for m in OLD.finditer(line):
                out.append(dict(doc="manuscript/main_text.md", where=f"line {i}",
                                text=" ".join(m.group(0).split()),
                                context=_ctx(line, m), cite=False))

    for key, label in DOCS.items():
        p = ROOT / "manuscript_check" / f"{key}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for pi, para in enumerate(d.get("paragraphs", [])):
            t = para if isinstance(para, str) else json.dumps(para)
            style = para.get("style", "") if isinstance(para, dict) else ""
            for m in OLD.finditer(t):
                ctx = _ctx(t, m)
                out.append(dict(doc=label, where=f"paragraph {pi}",
                                text=" ".join(m.group(0).split()),
                                context=ctx,
                                cite=is_citation(style, ctx)))
        for ti, tbl in enumerate(d.get("tables", [])):
            rows = tbl if isinstance(tbl, list) else tbl.get("rows", [])
            for ri, r in enumerate(rows):
                for ci, c in enumerate(r):
                    for m in OLD.finditer(str(c)):
                        out.append(dict(doc=label,
                                        where=f"table {ti + 1}, row {ri}",
                                        text=" ".join(m.group(0).split()),
                                        context=_ctx(str(c), m),
                                        cite=False))
    return out


def is_citation(style, context):
    """A reference-list entry: a published title that must not be reworded.

    The supplement carries at least two -- a cirrhosis paper and a NAFLD one --
    and relabelling either would misquote it.
    """
    if "bibliograph" in str(style).lower() or "reference" in str(style).lower():
        return True
    return bool(re.search(r"doi\.org|\b\d{4}\)\.|,\s*\d+\(\d+\),\s*\d+",
                          context))


def _ctx(s, m, pad=68):
    a, b = max(0, m.start() - pad), min(len(s), m.end() + pad)
    return " ".join(s[a:b].split())


def draft_docs():
    """Working drafts under docs/ that still carry the old name."""
    rows = []
    for p in sorted(glob.glob(str(ROOT / "docs/**/*"), recursive=True)):
        p = Path(p)
        if p.is_dir() or p.suffix.lower() not in (".md", ".html", ".csv"):
            continue
        if p.name == OUT_HTML.name:
            continue
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        n = len(OLD.findall(txt))
        if n:
            rows.append(dict(path=str(p.relative_to(ROOT)), n=n,
                             dated=bool(re.search(r"\d{6}", p.name))))
    return rows


# ── rendering ─────────────────────────────────────────────────────────────
CSS = """
:root{
  --ground:#FBFAF8; --panel:#FFFFFF; --sunk:#F4F2EE;
  --ink:#191C22; --body:#333A45; --muted:#5E6673; --faint:#8B93A1;
  --rule:#E4E1DA; --rule-strong:#CBC7BE;
  --accent:#274B6D; --accent-soft:#E6EDF4;
  --warn:#8A4B0F; --warn-soft:#FAF0E2; --warn-rule:#DFC08A;
  --old:#96351A; --old-soft:#FAEBE5;
  --new:#1C5E4A; --new-soft:#E3F1EB;
  --shadow:0 1px 2px rgba(25,28,34,.05);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0F1216; --panel:#161A20; --sunk:#12161B;
    --ink:#E8EBF0; --body:#C6CCD6; --muted:#98A1AF; --faint:#6D7686;
    --rule:#262C35; --rule-strong:#39414D;
    --accent:#7FAED4; --accent-soft:#17242F;
    --warn:#DFA55F; --warn-soft:#241C12; --warn-rule:#5A4526;
    --old:#DD8A6A; --old-soft:#26170F;
    --new:#68BC9C; --new-soft:#10231C;
    --shadow:none;
  }
}
:root[data-theme="dark"]{
  --ground:#0F1216; --panel:#161A20; --sunk:#12161B;
  --ink:#E8EBF0; --body:#C6CCD6; --muted:#98A1AF; --faint:#6D7686;
  --rule:#262C35; --rule-strong:#39414D;
  --accent:#7FAED4; --accent-soft:#17242F;
  --warn:#DFA55F; --warn-soft:#241C12; --warn-rule:#5A4526;
  --old:#DD8A6A; --old-soft:#26170F;
  --new:#68BC9C; --new-soft:#10231C;
  --shadow:none;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--body);
  font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,sans-serif;
  font-size:15px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:62rem;margin:0 auto;padding:3.2rem 1.5rem 7rem}
h1,h2,h3{font-family:"Source Serif 4",Georgia,serif;color:var(--ink);
  font-weight:600;text-wrap:balance;margin:0}
h1{font-size:2.35rem;line-height:1.12;letter-spacing:-.012em}
h2{font-size:1.42rem;margin:3.2rem 0 .5rem;padding-top:1.3rem;
  border-top:1px solid var(--rule)}
h3{font-size:1.04rem;margin:1.9rem 0 .45rem}
p{margin:.7rem 0;max-width:50rem}
.lede{font-size:1.07rem;color:var(--muted);max-width:46rem}
.eyebrow{font-size:.71rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);font-weight:600;margin-bottom:.55rem}
.phase{display:inline-block;font-family:"IBM Plex Mono";font-size:.78rem;
  color:var(--faint);margin-right:.5rem}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  font-size:.845em}
code{background:var(--sunk);padding:.09em .34em;border-radius:3px}
pre{background:var(--panel);border:1px solid var(--rule);border-radius:6px;
  padding:.85rem 1rem;overflow-x:auto;margin:.8rem 0;font-size:.82rem;
  line-height:1.55;box-shadow:var(--shadow)}
pre code{background:none;padding:0;font-size:1em}
.del{color:var(--old)} .ins{color:var(--new);font-weight:500}
.callout{background:var(--warn-soft);border:1px solid var(--warn-rule);
  border-left:3px solid var(--warn);border-radius:6px;padding:1.05rem 1.3rem;
  margin:1.5rem 0}
.callout h3{margin-top:0;color:var(--warn)}
.callout p, .note p{max-width:none}
.note{background:var(--panel);border:1px solid var(--rule);
  border-left:3px solid var(--accent);border-radius:6px;
  padding:.9rem 1.15rem;margin:1.1rem 0}
.note p{margin:.25rem 0}
.scroll{overflow-x:auto;margin:1rem 0}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th{text-align:left;font-size:.71rem;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);font-weight:600;padding:.45rem .6rem;
  border-bottom:1px solid var(--rule-strong);white-space:nowrap}
td{padding:.5rem .6rem;border-bottom:1px solid var(--rule);vertical-align:top}
tbody tr:hover{background:var(--sunk)}
td.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.tag{display:inline-block;font-size:.67rem;letter-spacing:.05em;font-weight:600;
  text-transform:uppercase;padding:.11rem .42rem;border-radius:3px;
  white-space:nowrap}
.tag.done{background:var(--new-soft);color:var(--new)}
.tag.todo{background:var(--old-soft);color:var(--old)}
.tag.keep{background:var(--accent-soft);color:var(--accent)}
ul,ol{max-width:50rem;padding-left:1.15rem}
li{margin:.34rem 0}
.meta{color:var(--faint);font-size:.8rem;margin-top:2.5rem}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(12rem,1fr));
  gap:.8rem;margin:1.4rem 0}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:6px;
  padding:.85rem .95rem;box-shadow:var(--shadow)}
.card b{display:block;font-family:"Source Serif 4",Georgia,serif;
  font-size:1.55rem;line-height:1.1;color:var(--ink);
  font-variant-numeric:tabular-nums}
.card span{font-size:.76rem;color:var(--muted);display:block;margin-top:.12rem}
"""


def esc(s):
    return html.escape(str(s))


def render(src, figs, measured, objs, occ, drafts):
    A = [].append
    L = []
    A = L.append
    done_src = sum(1 for r in src if r["done"])
    stale_figs = [f for f in figs if f["stale"]]

    A("<title>MASLD Nomenclature Migration</title>")
    A('<link rel="preconnect" href="https://fonts.googleapis.com">')
    A('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    A('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&'
      'family=IBM+Plex+Sans:wght@400;500;600&'
      'family=IBM+Plex+Mono:wght@400;500&display=swap">')
    A(f"<style>{CSS}</style>")
    A('<div class="wrap">')
    A('<div class="eyebrow">Implementation status · Section 3</div>')
    A("<h1>Renaming fatty liver disease to MASLD</h1>")
    A('<p class="lede">The condition drawn and tabulated as <b>Fatty Liver</b> '
      "becomes <b>MASLD</b>, expanded once at first use in each document and in "
      "the first figure that carries it. A labelling change end to end: no "
      "cohort, count, test or statistic moves. This page is regenerated from "
      "the tree, so the lists below are current.</p>")

    A('<div class="cols">')
    for big, lab in ((f"{done_src}/{len(src)}", "source edits applied"),
                     (f"{sum(1 for f in figs if f['shows'])}",
                      "figures printing the new label"),
                     (f"{len([o for o in occ if not o.get('cite')])}",
                      "manuscript mentions to change"),
                     ("0", "statistics affected")):
        A(f'<div class="card"><b>{big}</b><span>{lab}</span></div>')
    A("</div>")

    # ── design note ───────────────────────────────────────────────────────
    A("<h2><span class='phase'>00</span>Display label, not a rename</h2>")
    A("<p>The internal key stays <code>\"Fatty Liver\"</code>. It is a key in "
      "<code>config.DISEASE_NCD_MAP</code> (seeding dict ordering and colour "
      "lookups), an exact-match key in "
      "<code>health_axis_projection.py</code>, and a stored data value in more "
      "than a hundred result files. Renaming it would reorder panels silently "
      "and invalidate every one of those files. So the key stays and the "
      "display layer changes — the same split the codebase already uses for "
      "<code>Transitional</code> → <code>Agro-pastoralist</code>.</p>")
    A('<div class="note"><p><b>Unchanged by design:</b> '
      + "; ".join(f"<code>{esc(f)}</code> — {esc(w)}" for f, w in KEEP)
      + ". All internal keys or metadata column names.</p></div>")

    # ── 1 source ──────────────────────────────────────────────────────────
    A("<h2><span class='phase'>01</span>Source edits</h2>")
    A('<div class="scroll"><table><thead><tr><th></th><th>File</th>'
      "<th>What</th></tr></thead><tbody>")
    for r in src:
        tag = ('<span class="tag done">applied</span>' if r["done"]
               else '<span class="tag todo">pending</span>')
        n = f' <span class="mono" style="color:var(--faint)">×{r["n"]}</span>' \
            if r["n"] > 1 else ""
        A(f'<tr><td>{tag}</td><td class="mono">{esc(r["file"])}{n}</td>'
          f'<td>{esc(r["what"])}</td></tr>')
    A("</tbody></table></div>")
    A("<p>One entry in <code>PLOT_TEXT_REPLACEMENTS</code> reaches every "
      "supplementary Section 3 figure, because <code>section3_health.py</code> "
      "saves through <code>savefig_dual</code> 21 times and never raw — that "
      "catches the labels built straight from the condition name at lines 996, "
      "1178, 2192 and 2275, which no wrap map touches. "
      "<code>figures/figure5.py</code> is the exception: it calls "
      "<code>plt.savefig</code> directly, so the display pass never runs on it "
      "and its label map had to change at source.</p>")

    # ── 2 figures ─────────────────────────────────────────────────────────
    A("<h2><span class='phase'>02</span>Figures</h2>")
    if not measured:
        A('<div class="callout"><h3>Not measured</h3><p>'
          "<code>tools/figure_text_inventory.csv</code> is absent, so the table "
          "below cannot say what the figures actually print. Run "
          "<code>python tools/collect_figure_text.py --all</code> and "
          "regenerate.</p></div>")
    A('<div class="scroll"><table><thead><tr><th></th><th>Figure</th>'
      "<th>As printed</th></tr></thead><tbody>")
    for f in figs:
        if not measured:
            tag, note = '<span class="tag keep">unknown</span>', "not measured"
        elif f["stale"]:
            tag = '<span class="tag todo">still old</span>'
            note = "; ".join(esc(x) for x in f["stale"][:3])
        elif f["shows"]:
            tag, note = ('<span class="tag done">MASLD</span>',
                         "prints the new label")
        else:
            tag, note = ('<span class="tag keep">n/a</span>',
                         "prints no condition label at all, before or after "
                         "this change")
        A(f'<tr><td>{tag}</td><td class="mono">{esc(f["slot"])}</td>'
          f'<td class="mono" style="color:var(--muted)">{note}</td></tr>')
    A("</tbody></table></div>")
    A("<p><b>S12, S13 and S24 are not in this list</b> — they are keyed on "
      "lifestyle and health group, not condition, and carry no condition "
      "label. S25 and S27 are listed but print no condition name in the "
      "shipped figure either: the pre-change harvest shows the same, so that "
      "is how they have always been, not something this change removed.</p>")
    A("<p><b>Main-text Figure 5</b> sits outside the supplementary harvest, so "
      "it is checked by pulling the text back out of the saved PDF — which is "
      "how its second label site was caught, an axis built as "
      "<code>f\"{cond} − control\"</code> that bypassed the label map "
      "entirely:</p>")
    A("<pre><code>pdftotext main_text_figures/figure5.pdf - | grep -ci masld"
      "   # expect 2\n"
      "pdftotext main_text_figures/figure5.pdf - | grep -ci fatty"
      "   # expect 0</code></pre>")
    if objs:
        A(f'<div class="note"><p><b>{len(objs)} data objects still store '
          '"Fatty Liver".</b> That is correct and deliberate: the stored key '
          "matches the result CSVs and is the published record of the numbers. "
          "Only the rendering changes.</p></div>")
    A("<pre><code>python tools/replot_from_data.py S25   # one slot per call\n"
      "python figures/figure5.py                 # main text, standalone"
      "</code></pre>")

    # ── 3 manuscripts ─────────────────────────────────────────────────────
    A("<h2><span class='phase'>03</span>Places in the manuscripts</h2>")
    cites = [o for o in occ if o.get("cite")]
    todo = [o for o in occ if not o.get("cite")]
    if not occ:
        A('<div class="note"><p>No occurrence of the old name remains in any '
          "tracked document.</p></div>")
    else:
        A(f"<p>{len(todo)} mentions still read as the old name, in document "
          "order. The expansion belongs at the first mention of each document "
          "that is read on its own — the abstract, and the first figure caption "
          "that names the condition — with bare <code>MASLD</code> everywhere "
          "else. <code>manuscript/main_text.md</code> and <b>Main text</b> are "
          "the same document in two forms (the in-repo markdown and the "
          "extracted <code>.docx</code>); fix whichever you actually edit and "
          "the other follows.</p>")
        if cites:
            A('<div class="callout"><h3>Do not change these — they are '
              "citations</h3><p>A reference-list entry quotes a published "
              "title. Relabelling one misquotes it.</p><ul>")
            for o in cites:
                A(f'<li><b>{esc(o["doc"])}</b>, {esc(o["where"])}: '
                  f'<span class="mono">{esc(o["text"])}</span> — '
                  f'<span style="color:var(--muted)">…{esc(o["context"][:120])}…'
                  "</span></li>")
            A("</ul></div>")
        by = {}
        for o in todo:
            by.setdefault(o["doc"], []).append(o)
        for doc, items in by.items():
            A(f"<h3>{esc(doc)} — {len(items)}</h3>")
            A('<div class="scroll"><table><thead><tr><th>Where</th>'
              "<th>Reads</th><th>Context</th></tr></thead><tbody>")
            for o in items:
                A(f'<tr><td class="mono" style="white-space:nowrap">'
                  f'{esc(o["where"])}</td>'
                  f'<td class="mono del">{esc(o["text"])}</td>'
                  f'<td style="color:var(--muted)">…{esc(o["context"])}…</td>'
                  "</tr>")
            A("</tbody></table></div>")
        A('<div class="note"><p><b>The two half-done expansions.</b> Where the '
          "text already reads <i>“non-alcoholic fatty liver disease (metabolic "
          "dysfunction-Associated steatotic liver disease or MASLD)”</i>, the "
          "retired name still leads and the capitalisation is inconsistent "
          "between the two. Replace the whole construction with bare "
          "<code>MASLD</code>, and let the first-use expansion carry "
          "<i>Metabolic Dysfunction-Associated Steatotic Liver Disease "
          "(MASLD)</i>.</p></div>")

    # ── 4 drafts ──────────────────────────────────────────────────────────
    A("<h2><span class='phase'>04</span>Working documents under docs/</h2>")
    live = [d for d in drafts if not d["dated"]]
    dated = [d for d in drafts if d["dated"]]
    A(f"<p>{len(live)} live drafts and {len(dated)} dated notes mention the old "
      "name. The dated files record what was true when they were written and "
      "should not be retrofitted; the live ones need the change.</p>")
    A('<div class="scroll"><table><thead><tr><th></th><th>File</th>'
      "<th class='num'>Mentions</th></tr></thead><tbody>")
    for d in live + dated:
        tag = ('<span class="tag keep">dated, leave</span>' if d["dated"]
               else '<span class="tag todo">update</span>')
        A(f'<tr><td>{tag}</td><td class="mono">{esc(d["path"])}</td>'
          f'<td class="num">{d["n"]}</td></tr>')
    A("</tbody></table></div>")

    # ── 5 verification ────────────────────────────────────────────────────
    A("<h2><span class='phase'>05</span>Verification</h2>")
    A("<pre><code>python -m compileall -q persistence figures tools\n"
      "python -m persistence --help\n\n"
      "# no figure still PRINTS the old label. Filter to the final phase:\n"
      "# the authored phase is harvested before the display pass runs, so the\n"
      "# internal key legitimately appears there (S25 and S27 each hold one).\n"
      "python tools/collect_figure_text.py --all\n"
      "awk -F, '$2==\"final\"' tools/figure_text_inventory.csv \\\n"
      "     | grep -i fatty                             # expect nothing\n\n"
      "# the numbers did not move. results/ is gitignored, so git status\n"
      "# says nothing here -- ask the filesystem which files were rewritten,\n"
      "# and expect only figures and the stats reports (whose description\n"
      "# column carries the label) to appear.\n"
      "find results -newermt \"$(date +%Y-%m-%d)\" -name '*.csv' \\\n"
      "     | grep -v stats_report        # expect nothing\n\n"
      "# the Science formatting still holds with a shorter label\n"
      "python tools/audit_science_specs.py --skip S1</code></pre>")
    A('<div class="note"><p><b>The <code>_keep_text</code> opt-outs.</b> '
      "Artists flagged <code>_keep_text</code> are skipped by "
      "<code>apply_display_text</code> by design — the flag exists so the "
      "Science pass could stop short forms being expanded back. Four sit in "
      "<code>section3_health.py</code> (lines 4828, 4887, 4948, 4952); all "
      "carry health-group or detection-threshold text, not condition names. "
      "The harvest above is what proves it: a condition label hiding behind one "
      "of those flags would surface there as a live “Fatty”.</p></div>")

    A('<div class="callout"><h3>Liver cirrhosis is out of scope</h3>'
      "<p><code>disease_cirrhosis</code> is a separate variable from "
      "<code>disease_fatty_liver</code>, and cirrhosis is an end-stage outcome "
      "with several aetiologies rather than a synonym for MASLD. It reaches no "
      "analysis — the keys were bound to column names the metadata never uses "
      "(<code>table_s1a.py:68-72</code>), so those 132 samples never entered "
      "the Conditions column. Its only appearance in the manuscript is the "
      "title of a cited paper: Qin, N. <i>et al.</i> (2014), <i>Alterations of "
      "the human gut microbiome in liver cirrhosis</i>, <i>Nature</i> "
      "<b>513</b>, 59–64. Confirmed out of scope.</p></div>")

    A(f'<p class="meta">Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by '
      "<code>tools/build_masld_checklist.py</code> from the working tree. "
      "Re-run after editing a draft or redrawing a figure.</p>")
    A("</div>")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_HTML))
    a = ap.parse_args()

    src = source_state()
    figs, measured = figure_state()
    objs = data_objects()
    occ = manuscript_occurrences()
    drafts = draft_docs()

    Path(a.out).write_text(render(src, figs, measured, objs, occ, drafts))
    print(f"  {a.out}")
    print(f"    source edits applied: {sum(1 for r in src if r['done'])}"
          f"/{len(src)}")
    if measured:
        bad = [f['slot'] for f in figs if f['stale']]
        print(f"    figures still printing the old label: "
              f"{' '.join(bad) if bad else 'none'}")
    else:
        print("    figures: NOT MEASURED (run tools/collect_figure_text.py)")
    print(f"    manuscript mentions to change: "
          f"{len([o for o in occ if not o.get(chr(39)+chr(39)) and not o.get('cite')])}"
          f"  ({len([o for o in occ if o.get('cite')])} citations left alone)")
    print(f"    docs/ drafts mentioning it: {len(drafts)}")


if __name__ == "__main__":
    main()
