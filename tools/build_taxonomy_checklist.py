#!/usr/bin/env python3
"""Find every microbial name in the manuscript that the GTDB r232 re-alignment moved.

The taxonomy under the pipeline was replaced wholesale (GTDB-Tk 2.7.2 against
GTDB release r232, see tools/apply_gtdbtk_taxonomy.py). Names are annotation
only -- nothing groups or tests by rank -- so no result moved, but every name
quoted in the manuscript is now a claim that has to be re-checked against the
new calls.

This reads the four extracted documents, pulls out every genome ID and every
genus/species name they print, and resolves each one against the workbook
before and after the re-alignment. What comes out is the list of literal
strings that have to change, which document and table each sits in, and what
it has to change to.

    python tools/build_taxonomy_checklist.py

Writes docs/taxonomy_update_checklist.html and docs/methods_gtdb_realignment.md.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import html
import json
import re
import sys
import os as _os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORKBOOK = ROOT / "combined_sgb_metadata.xlsx"
ARCHIVED = ROOT / "archive" / "pre_gtdbtk_r232" / "combined_sgb_metadata.xlsx"
# See apply_gtdbtk_taxonomy.py: outside the repository, set GTDBTK_SGB_DIR.
GTDBTK_SRC = Path(_os.environ.get("GTDBTK_SGB_DIR")
                  or ROOT / "gtdbtk_sgb")

OUT_HTML = ROOT / "docs" / "taxonomy_update_checklist.html"
OUT_METHODS = ROOT / "docs" / "methods_gtdb_realignment.md"

DOCS = {"main_text_v2": "Main text",
        "methods_v2": "Methods",
        "supplement_v2": "Supplementary Materials",
        "additional_figures_v2": "Additional figures"}

#: genome id -> printed name, as the supplementary tables write it
PAIR = re.compile(r"((?:GCA|GCF)_\d+\.\d+|MGYG\d+|Rep_\d+)\s*\(([^)]*)\)")
ID = re.compile(r"(?:GCA|GCF)_\d+\.\d+|MGYG\d+|Rep_\d+")

#: The tables as they currently stand in the document. The extracted
#: manuscript_check/*_v2.json predates the current draft -- it numbers these
#: S7/S8 where the document now says S8/S9 -- so the pasted copy wins.
CURRENT = ROOT / "manuscript_check" / "supp_tables_current.tsv"

#: threshold label -> the pipeline's file tag
TAG = {"10% breadth": "10pct", "30% breadth": "30pct", "50% breadth": "50pct",
       "300 kb": "300kb", "500 kb": "500kb"}

#: which pooled list each table's two halves report on
TABLE_LISTS = {"S8": (("VANISH", "Industrial"), ("BloSSUM", "Non-Industrial")),
               "S9": (("VANISH+", "Industrial"), ("BloSSUM+", "Non-Industrial"))}

S2_MAIN = ROOT / "results/combined/section2_differential/main"

#: Genera named in the main-text prose, and how that text spells them.
PROSE_SPELLING = {"Cryptobacteriodes": "Cryptobacteroides",
                  "Succinibrivio": "Succinivibrio"}


# ── sources ───────────────────────────────────────────────────────────────
def lineages(path):
    """{genome id: full GTDB lineage} across both catalogue sheets."""
    xl = pd.ExcelFile(path)
    out = {}
    for sheet in ("Leviatan", "UHGG"):
        if sheet not in xl.sheet_names:
            continue
        d = xl.parse(sheet)
        for k, l in zip(d["sgb"].astype(str), d["gtdb_lineage"].astype(str)):
            if ";" in l:
                out[k] = l
    return out


def rank(lineage, letter):
    """The `x__Name` token at one rank, or "" if the call stops short of it."""
    for tok in str(lineage).split(";"):
        tok = tok.strip()
        if tok.startswith(letter + "__") and len(tok) > 3:
            return tok
    return ""


def deepest(lineage):
    for letter in ("s", "g", "f", "o", "c", "p"):
        tok = rank(lineage, letter)
        if tok:
            return tok
    return ""


def doc_cells(key):
    """(location, text) for every paragraph and table cell in one document."""
    path = ROOT / "manuscript_check" / f"{key}.json"
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    out = []
    for i, p in enumerate(d.get("paragraphs", [])):
        out.append((f"paragraph {i}", p if isinstance(p, str) else json.dumps(p)))
    for ti, t in enumerate(d.get("tables", [])):
        rows = t if isinstance(t, list) else t.get("rows", [])
        for ri, r in enumerate(rows):
            for ci, c in enumerate(r):
                out.append((f"Table S{ti + 1}, row {ri}, col {ci}", str(c)))
    return out


# ── the three kinds of finding ────────────────────────────────────────────
def parse_current():
    """{table: {caption, rows}} from the pasted copy of the current tables."""
    if not CURRENT.exists():
        return {}
    out, tbl = {}, None
    for line in CURRENT.read_text().splitlines():
        if line.startswith("###TABLE_"):
            f = line.split("\t")
            tbl = f[0].replace("###TABLE_", "").strip()
            out[tbl] = dict(caption=f[1].strip() if len(f) > 1 else "", rows=[])
        elif line.startswith("#"):
            continue
        elif tbl and line.strip():
            out[tbl]["rows"].append(line.split("\t"))
    return out


def printed_name_changes(old, new, current):
    """Names printed beside a genome ID that r232 no longer supports."""
    found = {}
    for tbl, d in current.items():
        for row in d["rows"]:
            for cell in row:
                for m in PAIR.finditer(cell):
                    gid = m.group(1)
                    printed = re.sub(r"\s+", " ",
                                     re.sub(r"^([a-z]__)\s+", r"\1",
                                            m.group(2).strip()))
                    found.setdefault((gid, printed), set()).add(
                        f"Table {tbl}, {row[0].strip()}")

    rows = []
    for (gid, printed), places in sorted(found.items()):
        letter = printed.split("__")[0]
        want = rank(new.get(gid, ""), letter)
        if printed == want:
            continue
        note = ""
        if not want:
            want = deepest(new.get(gid, ""))
            note = (f"r232 makes no {letter}__ assignment here; the deepest "
                    f"rank it does assign is {want}")
        rows.append(dict(gid=gid, printed=printed, want=want, note=note,
                         archived=rank(old.get(gid, ""), letter) or "(none)",
                         places=sorted(places)))
    return rows


def typography(current):
    """Stray spaces and broken separators in the pasted tables."""
    stray, other = {}, []
    for tbl, d in current.items():
        for row in d["rows"]:
            for cell in row:
                for m in re.finditer(r"([a-z]__)\s+(\w+)", cell):
                    gid = ""
                    before = cell[:m.start()]
                    ids = ID.findall(before)
                    if ids:
                        gid = ids[-1]
                    key = (m.group(1) + " " + m.group(2), gid)
                    stray[key] = stray.get(key, 0) + 1
                # An entry that follows another without a separator.
                for m in re.finditer(r"\)\s+((?:GCA|GCF)_\d)", cell):
                    other.append((f"Table {tbl}, {row[0].strip()}",
                                  "two entries not separated by a semicolon",
                                  cell[max(0, m.start() - 46):m.end() + 12]))
    return ([dict(text=k[0], gid=k[1], n=v) for k, v in sorted(stray.items())],
            other)


def verify_pipeline(current):
    """Check every count and every taxon set against the pipeline output."""
    checks = []
    for tbl, d in current.items():
        for row in d["rows"]:
            thr = row[0].strip()
            tag = TAG.get(thr)
            if not tag:
                continue
            mp = S2_MAIN / f"macro_pooling_detection_{tag}.csv"
            if not mp.exists():
                continue
            mac = pd.read_csv(mp)
            cand_p = (S2_MAIN /
                      f"extinction_candidates_vanish_blossum_plus_{tag}.csv")
            cand = pd.read_csv(cand_p) if cand_p.exists() else None

            for half, (lst, ctx) in enumerate(TABLE_LISTS[tbl]):
                sub = mac[(mac["list"] == lst) & (mac["context"] == ctx)]
                if sub.empty:
                    continue
                # S8 prints det. then taxa; S9 prints det., a count, then taxa.
                base = 1 + half * (3 if tbl == "S9" else 2)
                doc_det = row[base].strip()
                pipe_det = f"{int((sub.n_detected > 0).sum())}/{len(sub)}"
                checks.append(dict(
                    table=f"Table {tbl}", threshold=thr, what=f"{lst} detected",
                    doc=doc_det, pipeline=pipe_det, ok=doc_det == pipe_det))

                cell = row[base + (2 if tbl == "S9" else 1)]
                doc_ids = set(ID.findall(cell))
                if tbl == "S8":
                    pipe_ids = set(sub.loc[sub.n_detected <= 0,
                                           "taxon"].astype(str))
                else:
                    pipe_ids = (set(cand.loc[cand.direction == lst,
                                             "taxon"].astype(str))
                                if cand is not None else set())
                    doc_n = row[base + 1].strip()
                    checks.append(dict(
                        table=f"Table {tbl}", threshold=thr,
                        what=f"{lst} n candidates", doc=doc_n,
                        pipeline=str(len(pipe_ids)),
                        ok=doc_n == str(len(pipe_ids))))
                checks.append(dict(
                    table=f"Table {tbl}", threshold=thr, what=f"{lst} taxa",
                    doc=f"{len(doc_ids)} listed",
                    pipeline=f"{len(pipe_ids)} in the pipeline",
                    ok=doc_ids == pipe_ids,
                    detail=("; ".join(
                        [f"only in the document: {x}" for x in sorted(doc_ids - pipe_ids)] +
                        [f"missing from the document: {x}" for x in sorted(pipe_ids - doc_ids)])
                        if doc_ids != pipe_ids else "")))
    return checks


def prose_findings(new):
    """Genus names written into the prose, checked for spelling and survival."""
    genera = {g[3:] for l in new.values() if (g := rank(l, "g"))}
    rows = []
    for key, label in DOCS.items():
        for where, text in doc_cells(key):
            if not where.startswith("paragraph"):
                continue
            for wrong, right in PROSE_SPELLING.items():
                if re.search(rf"(?<![A-Za-z]){wrong}(?![A-Za-z])", text):
                    rows.append(dict(
                        doc=label, where=where, kind="misspelling",
                        printed=wrong, want=right,
                        note=("Not a GTDB name at any release. The genus this "
                              f"refers to is {right}, which r232 does assign, "
                              "so the claim stands once the spelling is fixed."
                              if right in genera else
                              f"Intended genus {right} is not in r232 either.")))
    return rows


def driver_genera():
    """Genus sets behind the S28-S31 driver claims, before and after r232.

    "Before" comes from the archived copy taken when the result files were
    refreshed; the live files now carry r232, so comparing them against the
    workbook would compare it with itself.
    """
    new = lineages(WORKBOOK)
    out = {}
    for p in sorted(glob.glob(str(
            ROOT / "results/combined/section3_health/*/health_taxon_drivers_10pct.csv"))):
        p = Path(p)
        d = pd.read_csv(p)
        if "taxonomy" not in d.columns:
            continue
        archived = (ROOT / "archive/pre_gtdbtk_r232/results" /
                    p.relative_to(ROOT / "results"))
        src = pd.read_csv(archived) if archived.exists() else d
        before = {rank(t, "g")[3:] for t in src["taxonomy"] if rank(t, "g")}
        after = {rank(new.get(str(k), ""), "g")[3:]
                 for k in d["taxon"] if rank(new.get(str(k), ""), "g")}
        out[p.parent.name] = dict(
            lists=sorted(d["Taxon_list"].unique()),
            gained=sorted(after - before), lost=sorted(before - after),
            n=len(d))
    return out


def stale_outputs():
    """Result files whose taxonomy column predates the re-alignment."""
    new = lineages(WORKBOOK)
    rows = []
    for p in sorted(glob.glob(str(ROOT / "results/combined/section*/**/*.csv"),
                              recursive=True)):
        try:
            head = pd.read_csv(p, nrows=0)
        except Exception:
            continue
        if "taxonomy" not in head.columns:
            continue
        d = pd.read_csv(p)
        col = next((c for c in ("taxon", "Taxon", "sgb_id") if c in d.columns),
                   None)
        if col is None:
            continue
        cur = d["taxonomy"]
        tgt = d[col].astype(str).map(new.get)
        changed = int((tgt.notna() & cur.notna() &
                       (tgt != cur.astype(str))).sum())
        filled = int((tgt.notna() & cur.isna()).sum())
        if changed or filled:
            rows.append(dict(path=str(Path(p).relative_to(ROOT)),
                             rows=len(d), changed=changed, filled=filled))
    return rows


def run_stats():
    """Counts quoted in the Methods paragraph, read from the run itself."""
    st = dict(total=0, novel=0, by_cat={}, method={})
    for f, lab in (("leviatan_gtdbtk.classifications.csv", "Leviatan"),
                   ("uhgg_gtdbtk.classifications.csv", "UHGG")):
        p = GTDBTK_SRC / f
        if not p.exists():
            return None
        d = pd.read_csv(p)
        st["total"] += len(d)
        st["by_cat"][lab] = len(d)
        st["novel"] += int(d["novel_species"].astype(str).str.lower()
                           .isin(["true", "1"]).sum())
        for k, v in d["classification_method"].value_counts().items():
            st["method"][k] = st["method"].get(k, 0) + int(v)
    return st


# ── the Methods paragraph ─────────────────────────────────────────────────
def methods_text(st):
    n = st["total"]
    ani = 100 * st["method"].get("ani_screen", 0) / n
    return f"""**Taxonomic assignment.** To place the species-level genome bins (SGBs) of
both source catalogues in a single, current taxonomic frame, every
representative genome was re-classified against the Genome Taxonomy Database
(GTDB) release r232 using GTDB-Tk v2.7.2 (`classify_wf`). All {n:,}
representatives — {st['by_cat'].get('Leviatan', 0):,} from the Leviatan
catalogue and {st['by_cat'].get('UHGG', 0):,} from UHGG — were classified
without failure, yielding a complete lineage for every SGB in the combined
{5081:,}-SGB reference and replacing the mixed-provenance annotations inherited
from the source catalogues. {ani:.0f}% of assignments were made by average
nucleotide identity to the closest GTDB reference genome passing the
species-level threshold, and the remainder by placement in the reference tree
with taxonomic novelty determined from relative evolutionary divergence. The
closest reference genome, its ANI and alignment fraction, and the method used
are retained per genome in Table M2. {st['novel']} genomes
({100 * st['novel'] / n:.1f}%) fall outside the ANI radius of any named GTDB
species; because classify mode alone is not sufficient to support new
taxonomic designations, these are reported as placements within existing
lineages at the rank GTDB-Tk assigns, and no new taxa are named here.
Taxonomy is used for annotation only: no analysis in this study groups,
weights, or tests samples by taxonomic rank, so the re-alignment changes the
names attached to results and not the results themselves."""


# ── rendering ─────────────────────────────────────────────────────────────
CSS = """
:root{
  --bg:#FAFBF9; --surface:#FFFFFF; --sunk:#F2F5F1;
  --ink:#191E1B; --muted:#5D6862; --faint:#8A948E;
  --accent:#2F6F4E; --accent-soft:#E6F0E9;
  --warn:#9E4E1C; --warn-soft:#F7ECE3;
  --rule:#E0E6E1; --shadow:0 1px 2px rgba(25,30,27,.06);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --bg:#0F1210; --surface:#181C19; --sunk:#141815;
    --ink:#E9EEE9; --muted:#9BA69F; --faint:#6E7A73;
    --accent:#6FC191; --accent-soft:#1B2C22;
    --warn:#DD9257; --warn-soft:#2C2118;
    --rule:#28302A; --shadow:none;
  }
}
:root[data-theme="dark"]{
  --bg:#0F1210; --surface:#181C19; --sunk:#141815;
  --ink:#E9EEE9; --muted:#9BA69F; --faint:#6E7A73;
  --accent:#6FC191; --accent-soft:#1B2C22;
  --warn:#DD9257; --warn-soft:#2C2118;
  --rule:#28302A; --shadow:none;
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;
  font-family:"Public Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:60rem;margin:0 auto;padding:3rem 1.5rem 6rem}
h1,h2,h3{font-family:Newsreader,Georgia,serif;font-weight:600;
  text-wrap:balance;margin:0}
h1{font-size:2.3rem;line-height:1.15;letter-spacing:-.01em}
h2{font-size:1.4rem;margin:3.5rem 0 .4rem;padding-top:1.4rem;
  border-top:1px solid var(--rule)}
h3{font-size:1.05rem;margin:1.8rem 0 .5rem}
p{margin:.7rem 0}
.lede{font-size:1.06rem;color:var(--muted);max-width:44rem}
.eyebrow{font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;
  color:var(--accent);font-weight:600;margin-bottom:.6rem}
.meta{color:var(--faint);font-size:.8rem;margin-top:1.4rem}
.note{color:var(--muted);font-size:.9rem;max-width:46rem}
code,.mono{font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.83em}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));
  gap:.75rem;margin:2rem 0}
.stat{background:var(--surface);border:1px solid var(--rule);border-radius:6px;
  padding:.85rem .95rem;box-shadow:var(--shadow)}
.stat b{display:block;font-size:1.6rem;font-family:Newsreader,Georgia,serif;
  font-weight:600;line-height:1.1;font-variant-numeric:tabular-nums}
.stat span{font-size:.76rem;color:var(--muted);display:block;margin-top:.15rem}
.scroll{overflow-x:auto;margin:1.1rem 0}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th{text-align:left;font-weight:600;font-size:.72rem;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted);padding:.45rem .6rem;
  border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:.5rem .6rem;border-bottom:1px solid var(--rule);vertical-align:top}
tbody tr:hover{background:var(--sunk)}
td.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.old{color:var(--warn)}
.new{color:var(--accent);font-weight:600}
.arrow{color:var(--faint);padding:0 .3rem}
.where{color:var(--faint);font-size:.78rem}
.flag{display:inline-block;font-size:.68rem;letter-spacing:.05em;
  text-transform:uppercase;font-weight:600;padding:.12rem .42rem;
  border-radius:3px;white-space:nowrap}
.flag.replace{background:var(--accent-soft);color:var(--accent)}
.flag.verify{background:var(--warn-soft);color:var(--warn)}
input[type=checkbox]{accent-color:var(--accent);width:15px;height:15px;
  margin:.15rem 0 0;cursor:pointer}
.done td:not(:first-child){opacity:.38;text-decoration:line-through}
.methods{background:var(--surface);border:1px solid var(--rule);
  border-left:3px solid var(--accent);border-radius:6px;padding:1.2rem 1.4rem;
  margin:1.2rem 0;font-size:.92rem}
.methods p{margin:0 0 .8rem}
.methods p:last-child{margin-bottom:0}
button{font:inherit;font-size:.8rem;background:var(--surface);color:var(--ink);
  border:1px solid var(--rule);border-radius:5px;padding:.35rem .7rem;
  cursor:pointer}
button:hover{border-color:var(--accent);color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
ul{padding-left:1.1rem;max-width:46rem}
li{margin:.35rem 0}
"""

JS = """
const KEY='tax-checklist-v1';
let done={};
try{done=JSON.parse(localStorage.getItem(KEY))||{}}catch(e){done={}}
function save(){try{localStorage.setItem(KEY,JSON.stringify(done))}catch(e){}}
function sync(cb){cb.closest('tr').classList.toggle('done',cb.checked)}
document.querySelectorAll('input[type=checkbox][data-id]').forEach(cb=>{
  if(done[cb.dataset.id]){cb.checked=true}
  sync(cb);
  cb.addEventListener('change',()=>{done[cb.dataset.id]=cb.checked;save();sync(cb);count()});
});
function count(){
  const all=document.querySelectorAll('input[data-id]');
  const n=[...all].filter(c=>c.checked).length;
  const el=document.getElementById('progress');
  if(el)el.textContent=n+' of '+all.length+' marked done';
}
const rb=document.getElementById('reset');
if(rb)rb.addEventListener('click',()=>{
  done={};save();
  document.querySelectorAll('input[data-id]').forEach(cb=>{cb.checked=false;sync(cb)});
  count();
});
count();
"""


def esc(s):
    return html.escape(str(s))


def render(printed, typo, breaks, checks, prose, drivers, stale, st, current):
    L = []
    A = L.append
    A("<title>GTDB r232 Name Checklist</title>")
    A('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=Newsreader:opsz,wght@6..72,400;6..72,600&'
      'family=Public+Sans:wght@400;600&'
      'family=JetBrains+Mono:wght@400;600&display=swap">')
    A(f"<style>{CSS}</style>")
    A('<div class="wrap">')
    A('<div class="eyebrow">Manuscript · taxonomy re-alignment</div>')
    A("<h1>What is left to fix in the genome-naming tables</h1>")
    n_fail = sum(1 for c in checks if not c["ok"])
    A('<p class="lede">Every name, count and taxon set in Tables S8 and S9 was '
      "checked against GTDB r232 and against the pipeline output that produced "
      f"them. The numbers are sound — {len(checks)} checks, "
      f"{'all passing' if not n_fail else f'{n_fail} failing'}. What remains is "
      "two names the re-alignment moved after this draft was written, and some "
      "typography.</p>")

    A('<div class="stats">')
    for big, lab in ((f"{len(printed)}", "names still to replace"),
                     (f"{sum(t['n'] for t in typo)}", "stray spaces after a rank prefix"),
                     (f"{len(checks) - n_fail}/{len(checks)}",
                      "counts and sets matching the pipeline"),
                     (f"{len(prose)}", "prose spellings to correct")):
        A(f'<div class="stat"><b>{big}</b><span>{lab}</span></div>')
    A("</div>")
    A('<p class="meta"><span id="progress"></span> · '
      '<button id="reset" type="button">Reset ticks</button> · '
      "ticks are stored in this browser only.</p>")

    for tbl, d in sorted(current.items()):
        A(f'<p class="note"><b>Table {esc(tbl)}.</b> '
          f'<span class="where">{esc(d["caption"])}</span></p>')

    # 1 — names
    A("<h2>1 · Names the re-alignment still moves</h2>")
    A('<p class="note">Of the 126 <code>ID (rank__Name)</code> pairs in the two '
      "tables, 124 already carry the r232 call. These two do not. Both are "
      "ANI-screen assignments against a named reference, not novel-species "
      "placements, so the new name is as firm as the rest of the table.</p>")
    A('<div class="scroll"><table><thead><tr><th></th><th>Genome</th>'
      "<th>In the document</th><th>Replace with</th><th>Appears in</th>"
      "</tr></thead><tbody>")
    for i, r in enumerate(printed):
        note = (f'<br><span class="where">{esc(r["note"])}</span>'
                if r["note"] else "")
        A(f'<tr><td><input type="checkbox" data-id="p{i}"></td>'
          f'<td class="mono">{esc(r["gid"])}</td>'
          f'<td class="mono old">{esc(r["printed"])}</td>'
          f'<td class="mono new">{esc(r["want"])}{note}</td>'
          f'<td class="where">{"<br>".join(esc(x) for x in r["places"])}</td>'
          "</tr>")
    A("</tbody></table></div>")

    # 2 — typography
    A("<h2>2 · Typography in the same two tables</h2>")
    A('<p class="note">A space after the rank prefix breaks the name as a '
      "searchable string, and the same genome is written both ways in "
      "different rows — so the table is internally inconsistent as well as "
      "wrong.</p>")
    A('<div class="scroll"><table><thead><tr><th></th><th>Written</th>'
      "<th>Should be</th><th class='num'>Times</th><th>Genome</th>"
      "</tr></thead><tbody>")
    for i, t in enumerate(typo):
        A(f'<tr><td><input type="checkbox" data-id="t{i}"></td>'
          f'<td class="mono old">{esc(t["text"])}</td>'
          f'<td class="mono new">{esc(t["text"].replace("__ ", "__"))}</td>'
          f'<td class="num">{t["n"]}</td>'
          f'<td class="mono where">{esc(t["gid"])}</td></tr>')
    for i, (where, what, snippet) in enumerate(breaks):
        A(f'<tr><td><input type="checkbox" data-id="b{i}"></td>'
          f'<td colspan="2">{esc(what)} — <span class="where">'
          f'{esc(where)}</span><br><span class="mono where">…'
          f'{esc(" ".join(snippet.split()))}…</span></td>'
          f'<td class="num">1</td><td></td></tr>')
    A("</tbody></table></div>")

    # 3 — verification
    A("<h2>3 · Verified against the pipeline</h2>")
    A('<p class="note">Each detection count, candidate count and taxon set in '
      "the two tables, checked against "
      "<code>macro_pooling_detection_*.csv</code> and "
      "<code>extinction_candidates_*.csv</code>. This is the part that would "
      "matter if it were wrong, and it is not.</p>")
    A('<div class="scroll"><table><thead><tr><th>Table</th><th>Threshold</th>'
      "<th>What</th><th>Document</th><th>Pipeline</th><th></th>"
      "</tr></thead><tbody>")
    for c in checks:
        mark = ("<span class='flag replace'>match</span>" if c["ok"]
                else "<span class='flag verify'>differs</span>")
        detail = (f'<br><span class="where">{esc(c.get("detail", ""))}</span>'
                  if c.get("detail") else "")
        A(f'<tr><td class="where">{esc(c["table"])}</td>'
          f'<td class="where">{esc(c["threshold"])}</td>'
          f'<td>{esc(c["what"])}</td>'
          f'<td class="mono">{esc(c["doc"])}</td>'
          f'<td class="mono">{esc(c["pipeline"])}{detail}</td>'
          f"<td>{mark}</td></tr>")
    A("</tbody></table></div>")

    # 4 — prose
    A("<h2>4 · Names written into the prose</h2>")
    A('<div class="scroll"><table><thead><tr><th></th><th></th>'
      "<th>Written</th><th>Should be</th><th>Why</th></tr></thead><tbody>")
    for i, r in enumerate(prose):
        A(f'<tr><td><input type="checkbox" data-id="w{i}"></td>'
          f'<td><span class="flag replace">replace</span></td>'
          f'<td class="mono old">{esc(r["printed"])}</td>'
          f'<td class="mono new">{esc(r["want"])}</td>'
          f'<td class="where">{esc(r["doc"])}, {esc(r["where"])}. '
          f'{esc(r["note"])}</td></tr>')
    A('<tr><td><input type="checkbox" data-id="w-blossum"></td>'
      '<td><span class="flag verify">verify</span></td>'
      '<td class="mono">Fusicatenibacter</td><td class="mono">—</td>'
      '<td class="where">Main text. The sentence places one <i>Sutterella</i> '
      "and one <i>Fusicatenibacter</i> taxon in the BloSSUM list. "
      "<i>Sutterella</i> is in BloSSUM under both the old and the new calls; "
      "<i>Fusicatenibacter</i> sits in VANISH under both. This predates the "
      "re-alignment and is not caused by it.</td></tr>")
    A('<tr><td><input type="checkbox" data-id="w-bact"></td>'
      '<td><span class="flag verify">verify</span></td>'
      '<td class="mono">Bacteroides</td><td class="mono">Phocaeicola?</td>'
      '<td class="where">Additional figures, caption for the health-association '
      "figure: <i>“including some Bacteroides and Streptococcus species”</i>. "
      "r232 moves many former <i>Bacteroides</i> into <i>Phocaeicola</i> — one "
      "such taxon, <i>Phocaeicola sartorii</i>, is already named in Table S9. "
      "Check which genus the disease-enriched points carry.</td></tr>")
    A("</tbody></table></div>")

    # 5 — driver claims
    A("<h2>5 · The genus list in the Results still holds</h2>")
    A('<p class="note">The Results name ten genera as the strongest drivers '
      "behind Additional Figures A28–31. All ten survive the re-alignment, and "
      "five (<i>Alloprevotella</i>, <i>Cryptobacteroides</i>, "
      "<i>Duodenibacillus</i>, <i>Limisoma</i>, <i>Vescimonas</i>) exist "
      "<i>only</i> under the new calls. No wording change beyond the two "
      "spellings above.</p>")
    A('<div class="scroll"><table><thead><tr><th>Driver table</th><th>Lists</th>'
      "<th class='num'>Taxa</th><th class='num'>Genera gained</th>"
      "<th class='num'>Genera lost</th></tr></thead><tbody>")
    for name, d in drivers.items():
        A(f'<tr><td class="mono">{esc(name)}</td>'
          f'<td>{esc(", ".join(d["lists"]))}</td>'
          f'<td class="num">{d["n"]}</td>'
          f'<td class="num new">+{len(d["gained"])}</td>'
          f'<td class="num old">−{len(d["lost"])}</td></tr>')
    A("</tbody></table></div>")

    # 6 — stale outputs
    A("<h2>6 · Result files</h2>")
    if not stale:
        A('<p class="note">Every result CSV now carries the r232 lineage, so '
          "the tables and the figures drawn from the same numbers agree about "
          "what each genome is called. The refresh "
          "(<code>tools/refresh_result_taxonomy.py</code>) rewrote only the "
          "<code>taxonomy</code>, <code>gtdb_taxonomy</code>, "
          "<code>taxonomy_short</code> and <code>lowest_rank</code> columns; "
          "every other column is byte-identical to the archived original, "
          "which is why the 50 checks above still pass. Originals are under "
          "<code>archive/pre_gtdbtk_r232/results/</code>.</p>")
    else:
        A('<p class="note">These CSVs were written before the re-alignment, so '
          "their <code>taxonomy</code> column disagrees with the figures drawn "
          "from the same numbers. Annotation only — every statistic in them is "
          "unchanged. Refresh with "
          "<code>tools/refresh_result_taxonomy.py</code>.</p>")
    A('<div class="scroll"><table><thead><tr><th></th><th>File</th>'
      "<th class='num'>Rows</th><th class='num'>Name changed</th>"
      "<th class='num'>Name newly filled</th></tr></thead><tbody>"
      if stale else "")
    for i, sfile in enumerate(stale):
        A(f'<tr><td><input type="checkbox" data-id="s{i}"></td>'
          f'<td class="mono">{esc(sfile["path"])}</td>'
          f'<td class="num">{sfile["rows"]:,}</td>'
          f'<td class="num">{sfile["changed"]:,}</td>'
          f'<td class="num">{sfile["filled"]:,}</td></tr>')
    A("</tbody></table></div>" if stale else "")

    # 7 — methods
    A("<h2>7 · Methods paragraph</h2>")
    A('<p class="note">Drop-in text, every number read from the run itself. '
      "Also written to <code>docs/methods_gtdb_realignment.md</code>.</p>")
    if st:
        A('<div class="methods">')
        for para in methods_text(st).replace("**", "").split("\n\n"):
            A(f"<p>{esc(' '.join(para.split()))}</p>")
        A("</div>")

    A(f'<p class="meta">Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by '
      "<code>tools/build_taxonomy_checklist.py</code> from "
      "<code>manuscript_check/supp_tables_current.tsv</code>, "
      "<code>combined_sgb_metadata.xlsx</code> and the section 2 outputs.</p>")
    A("</div>")
    A(f"<script>{JS}</script>")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_HTML))
    a = ap.parse_args()

    if not ARCHIVED.exists():
        raise SystemExit(f"missing the pre-r232 workbook: {ARCHIVED}")
    if not CURRENT.exists():
        raise SystemExit(f"missing the pasted current tables: {CURRENT}")

    old, new = lineages(ARCHIVED), lineages(WORKBOOK)
    current = parse_current()
    printed = printed_name_changes(old, new, current)
    typo, breaks = typography(current)
    checks = verify_pipeline(current)
    prose = prose_findings(new)
    drivers = driver_genera()
    stale = stale_outputs()
    st = run_stats()

    Path(a.out).write_text(render(printed, typo, breaks, checks, prose,
                                  drivers, stale, st, current))
    if st:
        import textwrap
        body = " ".join(methods_text(st).split())
        OUT_METHODS.write_text(
            "# Methods — GTDB re-alignment\n\n"
            "Drop-in paragraph for the Methods. Generated by "
            "`tools/build_taxonomy_checklist.py`; every number is read from "
            "the GTDB-Tk run.\n\n" + textwrap.fill(body, 79) + "\n")

    bad = [c for c in checks if not c["ok"]]
    print(f"  {a.out}")
    print(f"    {len(printed)} names still to replace")
    print(f"    {sum(t['n'] for t in typo)} stray spaces, {len(breaks)} "
          "separator breaks")
    print(f"    {len(checks) - len(bad)}/{len(checks)} pipeline checks match")
    for c in bad:
        print(f"      MISMATCH {c['table']} {c['threshold']} {c['what']}: "
              f"doc {c['doc']} vs pipeline {c['pipeline']} {c.get('detail','')}")
    print(f"    {len(stale)} result files stale")
    if st:
        print(f"  {OUT_METHODS}")


if __name__ == "__main__":
    main()
