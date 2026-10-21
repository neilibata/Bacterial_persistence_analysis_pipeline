#!/usr/bin/env python3
"""Pre-flight check for the final, from-scratch run of the whole pipeline.

The plan is that one run, from one curated metadata file and the current
coverage/depth matrices, produces every main-text and supplementary figure, and
that the metadata a reviewer is handed is the same file the pipeline read. This
checks that each of those premises actually holds before anything is launched,
because the run is measured in days rather than minutes.

Nothing here writes to results/ or starts a run. It reads the tree and writes
docs/final_run_plan.html.

    python tools/preflight_final_run.py
"""
from __future__ import annotations

import argparse
import datetime
import glob
import html
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_HTML = ROOT / "docs" / "final_run_plan.html"

#: Data_4 .. Data_11: the matrices the pipeline reads, in reference order.
MATRIX_ORDER = [
    ("UHGG", "breadth"), ("UHGG", "depth"),
    ("lev", "breadth"), ("lev", "depth"),
    ("combined", "breadth"), ("combined", "depth"),
    ("chimp", "breadth"), ("chimp", "depth"),
]


def fmt_ts(p):
    try:
        return datetime.datetime.fromtimestamp(
            os.path.getmtime(p)).strftime("%d %b %Y %H:%M")
    except OSError:
        return "—"


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


#: Where a reviewer-facing M1 would live once the submission bundle is
#: assembled (tools/build_data_directory.py, section 7 below). Deliberately
#: NOT the stale 2026-08-14 metadata_M1.csv at the repo root: comparing against
#: that is what made this page report "not in sync" for the very change that
#: brought M1 and the pipeline input into sync, and audit_final_run already
#: guards that nothing resolves to it.
DELIVERABLE_CANDIDATES = [
    "data/Data_1_metadata_M1.csv", "data/metadata_M1.csv",
    "submission/metadata_M1.csv", "deposit/metadata_M1.csv",
    "results_pkg/metadata_M1.csv",
]


def find_deliverable(explicit=None):
    """The M1 a reviewer is actually handed, or None if none exists yet."""
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    for rel in DELIVERABLE_CANDIDATES:
        p = ROOT / rel
        if p.exists():
            return p
    for p in sorted(ROOT.glob("data/Data_1*.csv")):
        return p
    return None


def metadata_check(deliverable=None):
    """Does the file a reviewer gets match the file the pipeline reads?

    Two separate questions, which the old version collapsed into one and so
    answered neither:

      1. does ``config.META_FILE`` resolve to the curated M1, or has it fallen
         back down ``META_FILE_CANDIDATES`` to a superseded file?
      2. is the M1 a reviewer is handed the same table as that input?

    (2) is the headline claim on this page. It used to load
    ``final_run/metadata_M1.csv`` as ``m1`` and ``config.META_FILE`` as
    ``cur`` -- and since 2026-09-24 ``config.META_FILE`` *is*
    ``final_run/metadata_M1.csv``, so the file was compared with itself:
    ``in_sync`` was unconditionally True and the card "OK / M1 matches the
    pipeline input" asserted nothing. Identity is not a comparison. It is now
    reported as identity, and the comparison runs against a genuinely separate
    deliverable copy when one exists.

    ``status`` is one of:
      "agree"     an independent deliverable copy was compared and matches
      "differ"    an independent deliverable copy was compared and does not
      "identity"  the deliverable path and the pipeline input are one file;
                  nothing independent was compared
      "absent"    no deliverable copy exists yet, so the claim is untested
    """
    from persistence import config
    live = Path(config.META_FILE)
    livep = ROOT / live

    # The curated M1: what config.META_FILE should resolve to, and the file the
    # submission bundle is to be built from.
    curated = ROOT / "final_run" / "metadata_M1.csv"
    cands = list(getattr(config, "META_FILE_CANDIDATES", []) or [])
    out = dict(live=live.name, live_rows=None, live_path=str(live),
               curated=str(curated.relative_to(ROOT)),
               curated_exists=curated.exists(),
               resolves_to_curated=(curated.exists() and livep.exists()
                                    and livep.resolve() == curated.resolve()),
               head_candidate=(cands[0] if cands else None),
               is_head_candidate=(bool(cands) and config.META_FILE == cands[0]),
               status="absent", in_sync=None)

    def _rel(p):
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    dp = find_deliverable(deliverable)
    out["deliverable"] = _rel(dp) if dp else None
    # Keep the old key: the HTML and the console summary both read M['m1'].
    m1p = dp or (curated if curated.exists() else None)
    out["m1"] = _rel(m1p) if m1p else None
    out["same_file"] = bool(m1p and livep.exists()
                            and m1p.resolve() == livep.resolve())
    if m1p is None or not livep.exists():
        return out
    if dp is None:
        # Nothing a reviewer is handed exists on disk, so there is nothing to
        # compare the input against. Loading the input twice and differencing
        # it with itself is what this branch replaced.
        out["status"] = "absent"
        return out
    if out["same_file"]:
        out["status"] = "identity"
        return out

    m1 = pd.read_csv(m1p, low_memory=False)
    cur = pd.read_csv(livep, low_memory=False)
    k = "Chosen_match_ID"
    a, b = set(m1[k].astype(str)), set(cur[k].astype(str))
    ua = m1.set_index(k)["Unhealthy"].astype(str).str.strip()
    ub = cur.set_index(k)["Unhealthy"].astype(str).str.strip()
    ua, ub = ua[~ua.index.duplicated()], ub[~ub.index.duplicated()]
    shared = [i for i in (a & b) if i in ua.index and i in ub.index]

    # Read length is a per-study property, so a sample missing from M1 can
    # inherit its study's value; that is the difference between a rebuild that
    # needs the network and one that does not.
    need = cur[cur[k].astype(str).isin(b - a)]
    studies = (set(need["StudyFolder"].astype(str))
               if "StudyFolder" in need.columns else set())
    known = (set(m1["StudyFolder"].astype(str))
             if "StudyFolder" in m1.columns else set())

    out.update(
        m1_rows=len(m1), live_rows=len(cur),
        only_m1=len(a - b), only_live=len(b - a),
        unhealthy_diff=int((ua.loc[shared] != ub.loc[shared]).sum()),
        m1_unhealthy=int((ua == "Unhealthy").sum()),
        live_unhealthy=int((ub == "Unhealthy").sum()),
        junk_cols=[c for c in m1.columns if c.startswith("Unnamed")],
        need_lookup=len(need), need_studies=len(studies),
        studies_covered=len(studies & known),
        in_sync=(len(a - b) == 0 and len(b - a) == 0
                 and int((ua.loc[shared] != ub.loc[shared]).sum()) == 0),
    )
    out["status"] = "agree" if out["in_sync"] else "differ"
    return out


def matrix_check():
    from persistence import config
    rows = []
    for i, (ref, kind) in enumerate(MATRIX_ORDER, start=4):
        spec = config.REFERENCES.get(ref, {})
        name = spec.get(f"{kind}_file")
        if not name:
            rows.append(dict(slot=f"Data_{i}", ref=ref, kind=kind,
                             name="(not configured)", ok=False, size="—",
                             note="no entry in config.REFERENCES"))
            continue
        p = ROOT / name
        link = os.path.islink(p)
        target = os.readlink(p) if link else None
        try:
            size = os.path.getsize(os.path.realpath(p))
            readable = True
        except OSError:
            size, readable = 0, False
        if readable:
            try:
                with open(p, "rb") as fh:
                    readable = bool(fh.read(64))
            except OSError:
                readable = False
        note = ""
        if link:
            note = "symlink → " + str(Path(target).parent.name) + "/"
        rows.append(dict(slot=f"Data_{i}", ref=ref, kind=kind, name=name,
                         ok=readable and size > 0,
                         size=human(size) if size else "—", note=note,
                         ts=fmt_ts(p)))
    return rows


def figure_coverage():
    from tools import figure_sets
    producer = {}
    for slot, ref, stem in figure_sets.manifest():
        producer.setdefault(stem.split("/")[0], []).append(slot)
    #: which --only key rebuilds each producer directory
    section_of = {
        "section1_diversity": "s1", "section2_differential": "s2",
        "section3_health": "s3", "carter_analysis": "carter",
        "depth_audit": "(always runs)", "depth_bias_ipw": "depth_bias",
        "figure_s0_roadmap": "--figure full",
    }
    rows = []
    for d, slots in sorted(producer.items(), key=lambda kv: -len(kv[1])):
        key = section_of.get(d)
        if key is None:
            key = "accum / figure3" if "accum" in d or "figure3" in d else "?"
        rows.append(dict(dir=d, slots=sorted(
            slots, key=lambda s: int("".join(c for c in s if c.isdigit()) or 0)),
            section=key))
    return rows


def runtime_evidence():
    out = []
    for p in sorted(glob.glob(str(ROOT / "logs" / "*.log")),
                    key=os.path.getmtime, reverse=True):
        try:
            txt = Path(p).read_text(errors="ignore")
        except OSError:
            continue
        m = re.findall(r"All done in ([0-9.]+) min", txt)
        if not m:
            continue
        scope = "--all" if "--all" in txt[:4000] else ""
        mo = re.search(r"--only ([a-z0-9_ ]+)", txt[:4000])
        if not scope and mo:
            scope = "--only " + mo.group(1).strip()
        out.append(dict(log=Path(p).name, mins=float(m[-1]),
                        scope=scope or "—", ts=fmt_ts(p)))
    out.sort(key=lambda r: -r["mins"])
    return out[:6]


CSS = """
:root{
  --ground:#FBFAF7; --panel:#FFFFFF; --sunk:#F3F1EC;
  --ink:#1A1915; --body:#3A382F; --muted:#63604F; --faint:#918D78;
  --rule:#E5E1D6; --rule-strong:#CBC5B2;
  --accent:#5B5030; --accent-soft:#F0ECDF;
  --stop:#8C3A2E; --stop-soft:#FAEBE7; --stop-rule:#DFB3A8;
  --go:#3C5F32; --go-soft:#ECF2E7;
  --wait:#8A6212; --wait-soft:#FBF3E0;
  --shadow:0 1px 2px rgba(26,25,21,.06);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#121210; --panel:#1A1A16; --sunk:#161613;
    --ink:#EFEDE4; --body:#CBC8BA; --muted:#9D9987; --faint:#736F5D;
    --rule:#2A2A24; --rule-strong:#3E3D33;
    --accent:#CBBE90; --accent-soft:#232116;
    --stop:#E0907F; --stop-soft:#2A1713; --stop-rule:#553028;
    --go:#8FBE7E; --go-soft:#16220F;
    --wait:#DDAE5C; --wait-soft:#241D0F;
    --shadow:none;
  }
}
:root[data-theme="dark"]{
  --ground:#121210; --panel:#1A1A16; --sunk:#161613;
  --ink:#EFEDE4; --body:#CBC8BA; --muted:#9D9987; --faint:#736F5D;
  --rule:#2A2A24; --rule-strong:#3E3D33;
  --accent:#CBBE90; --accent-soft:#232116;
  --stop:#E0907F; --stop-soft:#2A1713; --stop-rule:#553028;
  --go:#8FBE7E; --go-soft:#16220F;
  --wait:#DDAE5C; --wait-soft:#241D0F;
  --shadow:none;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--body);
  font-family:"Public Sans",-apple-system,BlinkMacSystemFont,sans-serif;
  font-size:15.5px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:64rem;margin:0 auto;padding:3rem 1.5rem 7rem}
h1,h2,h3{font-family:"Newsreader",Georgia,serif;color:var(--ink);font-weight:600;
  text-wrap:balance;margin:0}
h1{font-size:2.35rem;line-height:1.13;letter-spacing:-.012em}
h2{font-size:1.4rem;margin:3rem 0 .5rem;padding-top:1.25rem;
  border-top:1px solid var(--rule)}
h3{font-size:1.03rem;margin:1.7rem 0 .4rem}
p{margin:.68rem 0;max-width:50rem}
.lede{font-size:1.07rem;color:var(--muted);max-width:47rem}
.eyebrow{font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;
  color:var(--accent);font-weight:700;margin-bottom:.5rem}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:.85em}
code{background:var(--sunk);padding:.08em .34em;border-radius:3px}
pre{background:var(--panel);border:1px solid var(--rule);border-radius:6px;
  padding:.8rem 1rem;overflow-x:auto;margin:.75rem 0;font-size:.82rem;
  line-height:1.55;box-shadow:var(--shadow)}
pre code{background:none;padding:0}
.stop{background:var(--stop-soft);border:1px solid var(--stop-rule);
  border-left:3px solid var(--stop);border-radius:6px;padding:1.05rem 1.3rem;margin:1.5rem 0}
.stop h3{margin-top:0;color:var(--stop)}
.note{background:var(--panel);border:1px solid var(--rule);
  border-left:3px solid var(--accent);border-radius:6px;padding:.85rem 1.1rem;margin:1rem 0}
.note p,.stop p{max-width:none} .note p{margin:.2rem 0}
.scroll{overflow-x:auto;margin:.9rem 0}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th{text-align:left;font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);font-weight:700;padding:.42rem .6rem;
  border-bottom:1px solid var(--rule-strong);white-space:nowrap}
td{padding:.46rem .6rem;border-bottom:1px solid var(--rule);vertical-align:top}
tbody tr:hover{background:var(--sunk)}
td.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.tag{display:inline-block;font-size:.66rem;letter-spacing:.05em;font-weight:700;
  text-transform:uppercase;padding:.11rem .44rem;border-radius:3px;white-space:nowrap}
.tag.ok{background:var(--go-soft);color:var(--go)}
.tag.bad{background:var(--stop-soft);color:var(--stop)}
.tag.wait{background:var(--wait-soft);color:var(--wait)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));
  gap:.75rem;margin:1.4rem 0}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:6px;
  padding:.8rem .9rem;box-shadow:var(--shadow)}
.card b{display:block;font-family:"Newsreader",Georgia,serif;font-size:1.5rem;
  line-height:1.1;color:var(--ink);font-variant-numeric:tabular-nums}
.card span{font-size:.74rem;color:var(--muted);display:block;margin-top:.1rem}
ul,ol{max-width:50rem;padding-left:1.15rem} li{margin:.3rem 0}
.meta{color:var(--faint);font-size:.79rem;margin-top:2.4rem}
"""


def esc(s):
    return html.escape(str(s))


def render(M, mats, figs, runs):
    L = []; A = L.append
    total_fig = sum(len(f["slots"]) for f in figs)
    bad_mats = [m for m in mats if not m["ok"]]
    longest = runs[0]["mins"] if runs else None

    A("<title>Final Run Pre-flight</title>")
    A('<link rel="preconnect" href="https://fonts.googleapis.com">')
    A('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    A('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=Newsreader:opsz,wght@6..72,400;6..72,600&'
      'family=Public+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&'
      'display=swap">')
    A(f"<style>{CSS}</style>")
    A('<div class="wrap">')
    A('<div class="eyebrow">Pre-flight · final from-scratch run</div>')
    A("<h1>What has to be true before the final run starts</h1>")
    A('<p class="lede">One run, from one curated metadata file and the current '
      "coverage and depth matrices, producing every main-text and supplementary "
      "figure — and the metadata handed to a reviewer being the same file the "
      "pipeline read. Each of those premises is checked here. Nothing has been "
      "launched.</p>")

    # The card only reads OK when an independent copy was compared. "identity"
    # and "absent" are reported as what they are: nothing was compared, so the
    # claim is untested. The old card showed OK unconditionally.
    _m1_card = {"agree": ("OK", "deliverable M1 matches the pipeline input"),
                "differ": ("NO", "deliverable M1 differs from the input"),
                "identity": ("SAME FILE", "deliverable M1 *is* the input "
                                          "(nothing compared)"),
                "absent": ("—", "no deliverable M1 on disk to compare")}
    A('<div class="cards">')
    for big, lab in (
            _m1_card.get(M.get("status", "absent"), ("—", "M1 not checked")),
            (f"{len(mats) - len(bad_mats)}/{len(mats)}", "input matrices readable"),
            (f"{total_fig}", "shipped figures with a producer"),
            (f"~{longest/60:.0f} h" if longest else "—", "last full run, recorded")):
        A(f'<div class="card"><b>{esc(big)}</b><span>{lab}</span></div>')
    A("</div>")

    # ── 1 the metadata problem ────────────────────────────────────────────
    A("<h2>1 · The metadata a reviewer sees is not the metadata the pipeline reads</h2>")

    # Whatever the deliverable situation, there is one thing always worth
    # asserting and never asserted before: that config.META_FILE still
    # resolves to the curated M1 rather than having fallen down
    # META_FILE_CANDIDATES to a superseded file after a rename or a move.
    if M.get("resolves_to_curated"):
        A('<div class="note"><p><code>config.META_FILE</code> resolves to '
          f"<code>{esc(M['curated'])}</code>, the curated M1"
          + ("" if M.get("is_head_candidate") else
             " — but it is <b>not</b> the head of "
             "<code>META_FILE_CANDIDATES</code> (<code>"
             + esc(str(M.get("head_candidate"))) + "</code>), so the pipeline "
             "reached it by fallback")
          + ".</p></div>")
    else:
        A('<div class="stop"><h3>The pipeline is not reading the curated M1</h3>'
          f"<p><code>config.META_FILE</code> is <code>{esc(M['live_path'])}</code>, "
          f"not <code>{esc(M['curated'])}</code>"
          + ("" if M.get("curated_exists") else
             " (which does not exist on this tree)")
          + ". Everything below describes a different cohort from the one the "
          "submission is built on.</p></div>")

    if M.get("status") == "absent":
        A('<div class="stop"><h3>Not checked: there is no deliverable M1 yet</h3>'
          "<p>The claim this section exists to test — that the M1 a reviewer is "
          "handed is the table the pipeline read — <b>has not been tested</b>. "
          "No reviewer-facing copy exists at any of "
          f"<code>{esc(', '.join(DELIVERABLE_CANDIDATES))}</code>, and "
          "<code>tools/build_data_directory.py</code> (section 7) has not been "
          "written, so there is nothing to compare the input against. This page "
          "used to load the pipeline input twice and difference it with itself, "
          "which reported agreement unconditionally. Re-run with "
          "<code>--deliverable &lt;path&gt;</code> once the bundle exists.</p>"
          "</div>")
    elif M.get("status") == "identity":
        A('<div class="note"><p>The deliverable path and '
          "<code>config.META_FILE</code> are <b>one file</b> "
          f"(<code>{esc(M['m1'])}</code>): the submission artefact and the "
          "analysis input cannot disagree because they are the same object. "
          "Nothing was compared — this is identity, not a passed check. A "
          "separate copy in a submission bundle still needs comparing with "
          "<code>--deliverable &lt;path&gt;</code>.</p></div>")
    elif not M.get("in_sync"):
        A('<div class="stop"><h3>M1 is a parallel artefact, not the input</h3>')
        A("<p><code>metadata_M1.csv</code> was built from an earlier cohort and "
          "has drifted from the file <code>config.META_FILE</code> actually "
          "resolves to. The differences are not cosmetic:</p>")
        A('<div class="scroll"><table><thead><tr><th></th>'
          f"<th class='num'>{esc(M['m1'])}</th>"
          f"<th class='num'>{esc(M['live'])}</th></tr></thead><tbody>")
        for lab, x, y in (("rows", M["m1_rows"], M["live_rows"]),
                          ("samples labelled Unhealthy",
                           M["m1_unhealthy"], M["live_unhealthy"])):
            A(f'<tr><td>{lab}</td><td class="num">{x:,}</td>'
              f'<td class="num">{y:,}</td></tr>')
        A("</tbody></table></div>")
        A(f"<p>{M['only_live']:,} samples the pipeline analyses are absent from "
          f"M1 and {M['only_m1']:,} are in M1 but not analysed; on the samples "
          f"they share, <b>{M['unhealthy_diff']:,} health labels disagree</b>. "
          "M1 also carries "
          f"{len(M['junk_cols'])} stray unnamed column(s) from a CSV "
          "round-trip. So the obesity correction is not the only thing missing "
          "from it — it never had the August curation either.</p>")
        A("<p><b>The fix is structural, not another patch:</b> M1 becomes the "
          "curated file, and the pipeline reads M1. One file, one definition, "
          "the same object in the submission and in the run.</p></div>")
    else:
        A('<div class="note"><p>The deliverable '
          f"<code>{esc(M['m1'])}</code> and the pipeline input "
          f"<code>{esc(M['live'])}</code> are separate files, and they agree "
          f"on all {M['m1_rows']:,} samples and every health label.</p></div>")

    # Only meaningful when two files were actually differenced; with nothing to
    # compare, "the N samples missing from M1" is a count of nothing.
    if M.get("status") in ("agree", "differ"):
        A("<h3>Rebuilding M1 needs no network</h3>")
        A(f"<p>Read length is a per-study property, which is why the enrichment "
          "tool imputes it per study. The "
          f"{M.get('need_lookup', 0):,} samples missing from M1 fall in "
          f"{M.get('need_studies', 0)} studies, and "
          f"<b>{M.get('studies_covered', 0)} of those {M.get('need_studies', 0)} "
          "already have a read length in M1</b> — so they can inherit it and have "
          "<code>number_reads</code> computed, with no SRA calls and nothing to "
          "wait on.</p>")

    # ── 2 inputs ──────────────────────────────────────────────────────────
    A("<h2>2 · Inputs the run will read</h2>")
    A('<div class="scroll"><table><thead><tr><th></th><th>Slot</th>'
      "<th>Reference</th><th>Matrix</th><th class='num'>Size</th>"
      "<th>Modified</th></tr></thead><tbody>")
    for m in mats:
        tag = ('<span class="tag ok">readable</span>' if m["ok"]
               else '<span class="tag bad">missing</span>')
        note = f'<br><span style="color:var(--faint)">{esc(m["note"])}</span>' if m["note"] else ""
        A(f'<tr><td>{tag}</td><td class="mono">{esc(m["slot"])}</td>'
          f'<td>{esc(m["ref"])}</td>'
          f'<td class="mono">{esc(m["name"])}{note}</td>'
          f'<td class="num">{esc(m["size"])}</td>'
          f'<td class="mono">{esc(m.get("ts","—"))}</td></tr>')
    A("</tbody></table></div>")

    # ── 3 figure coverage ─────────────────────────────────────────────────
    A("<h2>3 · Every shipped figure has a producer</h2>")
    A(f"<p>All {total_fig} supplementary figures map to a section of the "
      "pipeline or to a standalone figure task, so a run with every section "
      "enabled plus <code>--figure full</code> leaves none of them behind. "
      "The five main-text figures are the figure tasks themselves.</p>")
    A('<div class="scroll"><table><thead><tr><th>Produced by</th>'
      "<th>Rebuilt by</th><th class='num'>N</th><th>Figures</th>"
      "</tr></thead><tbody>")
    for f in figs:
        A(f'<tr><td class="mono">{esc(f["dir"])}</td>'
          f'<td class="mono">{esc(f["section"])}</td>'
          f'<td class="num">{len(f["slots"])}</td>'
          f'<td class="mono" style="color:var(--muted)">'
          f'{esc(" ".join(f["slots"]))}</td></tr>')
    A("</tbody></table></div>")

    # ── 4 cost ────────────────────────────────────────────────────────────
    A("<h2>4 · What it costs, from this repository's own logs</h2>")
    A('<div class="scroll"><table><thead><tr><th>Log</th><th>Scope</th>'
      "<th class='num'>Minutes</th><th class='num'>Hours</th>"
      "</tr></thead><tbody>")
    for r in runs:
        A(f'<tr><td class="mono">{esc(r["log"])}</td>'
          f'<td class="mono">{esc(r["scope"])}</td>'
          f'<td class="num">{r["mins"]:,.0f}</td>'
          f'<td class="num">{r["mins"]/60:,.1f}</td></tr>')
    A("</tbody></table></div>")
    A('<div class="stop"><h3>A true from-scratch run is measured in days</h3>'
      "<p>The last recorded <code>--all</code> run took <b>"
      f"{longest:,.0f} minutes — about {longest/60:.0f} hours</b>. A single "
      "section, <code>s1_axis_angle</code>, accounts for roughly 50 of them on "
      "its own, and it cannot simply be dropped: it writes the axis-angle point "
      "cloud that <b>Figure 2 panel F embeds</b>.</p>"
      "<p>This is the number to agree on before starting, not after. If the "
      "goal is a defensible final run, it is a few days of machine time; if the "
      "goal is only that everything is consistent with the corrected health "
      "definition, the ordered subset in section 5 does that in a fraction of "
      "it.</p></div>" if longest else "")

    # ── 5 the plan ────────────────────────────────────────────────────────
    A("<h2>5 · The order, either way</h2>")
    A("<pre><code># 0  make M1 the single curated metadata, and the pipeline's input\n"
      "python tools/build_curated_m1.py            # to be written; see below\n"
      "#    then put metadata_M1.csv at the head of config.META_FILE_CANDIDATES\n\n"
      "# 1  full from-scratch run  (~3 days)\n"
      "python -m persistence --all --figure full\n\n"
      "# 1' or the consistency-only subset, per reference (~hours)\n"
      "python -m persistence --ref combined --only s2 s3 table_s1a accum\n\n"
      "# 2  regenerate the metadata tables that depend on the run\n"
      "python tools/build_metadata_tables.py       # M2, M3\n\n"
      "# 3  redraw from the data objects so figure-code edits are applied\n"
      "python tools/replot_from_data.py --all\n"
      "python tools/replot_main.py --all\n\n"
      "# 4  assemble the submission data directory\n"
      "python tools/build_data_directory.py        # to be written; see below\n\n"
      "# 5  verify\n"
      "python tools/preflight_final_run.py         # this page, re-run\n"
      "python tools/audit_science_specs.py --skip S1\n"
      "python tools/audit_pdf_type.py --all</code></pre>")

    # ── 6 data directory ──────────────────────────────────────────────────
    A("<h2>6 · The submission data directory</h2>")
    A("<p>Eleven pointers, in the order requested. M1 is the curated per-sample "
      "metadata and the pipeline's own input; M2 and M3 are regenerated after "
      "the run because M3's taxon lists are a Section 2 output.</p>")
    A('<div class="scroll"><table><thead><tr><th>Slot</th><th>Contents</th>'
      "<th>Source</th></tr></thead><tbody>")
    A('<tr><td class="mono">Data_1</td><td>Metadata M1 — curated per-sample '
      'metadata</td><td class="mono">metadata_M1.csv (rebuilt, and the '
      "pipeline's input)</td></tr>")
    A('<tr><td class="mono">Data_2</td><td>Metadata M2 — reference catalogues'
      '</td><td class="mono">metadata_M2.xlsx (regenerated)</td></tr>')
    A('<tr><td class="mono">Data_3</td><td>Metadata M3 — taxon lists</td>'
      '<td class="mono">metadata_M3.xlsx (regenerated after Section 2)</td></tr>')
    for m in mats:
        A(f'<tr><td class="mono">{esc(m["slot"])}</td>'
          f'<td>{esc(m["ref"])} {esc(m["kind"])} matrix</td>'
          f'<td class="mono">{esc(m["name"])}</td></tr>')
    A("</tbody></table></div>")
    A('<div class="note"><p>The matrices total about 15 GB. Whether '
      "<code>data/</code> holds copies or symlinks matters: symlinks keep the "
      "repository small but do not travel with it, and two of these already "
      "point into iCloud. For a submission bundle they have to be real files "
      "somewhere the journal can receive.</p></div>")

    # ── 7 what is still unwritten ─────────────────────────────────────────
    A("<h2>7 · What does not exist yet</h2>")
    A("<ul>"
      "<li><code>tools/build_curated_m1.py</code> — folds the health curation "
      "(disease flags, free-text field, obesity) into one idempotent pass and "
      "writes <code>metadata_M1.csv</code>, carrying read length across by "
      "study. Replaces the current chain of patched files.</li>"
      "<li><code>tools/build_data_directory.py</code> — assembles "
      "<code>data/Data_1..Data_11</code> with a manifest and checksums.</li>"
      "<li>A decision on full run versus consistency subset, and on copies "
      "versus symlinks in <code>data/</code>.</li></ul>")

    A(f'<p class="meta">Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by '
      "<code>tools/preflight_final_run.py</code>. Reads the tree only; starts "
      "nothing. Re-run immediately before launching.</p>")
    A("</div>")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_HTML))
    ap.add_argument("--deliverable", default=None,
                    help="the reviewer-facing M1 to compare the pipeline input "
                         "against (the submission bundle's copy). Without it "
                         "the comparison is reported as untested, not as "
                         "passed.")
    a = ap.parse_args()
    M = metadata_check(a.deliverable)
    mats = matrix_check()
    figs = figure_coverage()
    runs = runtime_evidence()
    Path(a.out).write_text(render(M, mats, figs, runs))

    # Three outcomes, printed as three different things. The old line said
    # "yes" whenever in_sync was truthy, and in_sync was computed by
    # differencing the pipeline input with itself, so it said "yes" always.
    verdict = {"agree": "yes (compared against an independent copy)",
               "differ": "NO",
               "identity": "CANNOT VERIFY — the deliverable path and the "
                           "pipeline input are one file; nothing was compared",
               "absent": "CANNOT VERIFY — no deliverable M1 exists to compare "
                         "(pass --deliverable)"}
    print(f"  {a.out}")
    print(f"    deliverable M1 matches the input   : "
          f"{verdict.get(M.get('status'), 'CANNOT VERIFY')}")
    if M.get("status") == "differ":
        print(f"      rows {M['m1_rows']:,} vs {M['live_rows']:,}; "
              f"{M['unhealthy_diff']:,} health labels differ")
    print(f"    pipeline reads the curated M1      : "
          f"{'yes' if M.get('resolves_to_curated') else 'NO'}"
          f"  ({M.get('live_path')})")
    print(f"    matrices readable                  : "
          f"{sum(1 for m in mats if m['ok'])}/{len(mats)}")
    print(f"    shipped figures with a producer    : "
          f"{sum(len(f['slots']) for f in figs)}")
    if runs:
        print(f"    longest recorded run               : "
              f"{runs[0]['mins']:,.0f} min ({runs[0]['mins']/60:.0f} h) "
              f"[{runs[0]['scope']}]")

    failed, unverified = [], []
    (failed if M.get("status") == "differ" else
     unverified if M.get("status") in ("identity", "absent") else
     []).append("the deliverable M1 equals the pipeline input")
    if not M.get("resolves_to_curated"):
        failed.append("config.META_FILE resolves to the curated M1")
    bad_mats = [m for m in mats if not m["ok"]]
    if bad_mats:
        failed.append(f"{len(bad_mats)} input matrix/matrices unreadable")
    if not runs:
        unverified.append("the run cost, from this repository's logs")
    for label, items in (("FAILED", failed), ("NOT VERIFIED", unverified)):
        for it in items:
            print(f"    {label:<12} {it}")
    # Exit status so this cannot be read as a go-ahead when it established
    # nothing: 1 a premise is false, 3 a premise is untested, 0 all verified.
    return 1 if failed else (3 if unverified else 0)


if __name__ == "__main__":
    raise SystemExit(main())
