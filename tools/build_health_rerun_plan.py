#!/usr/bin/env python3
"""Plan the regeneration that the corrected `Unhealthy` definition requires.

Obesity now counts as unhealthy (tools/apply_obesity_health_rule.py), which
moved 145 samples out of the Industrial Healthy group. `Unhealthy` is not a
label the pipeline only prints: it is the control/case split for Section 3 and,
less obviously, the group that Section 2 derives the VanishH+ and BlossumH+
lists from. So the change reaches further than the section it was made for.

This checks the tree and writes docs/health_definition_rerun_plan.html: what is
already consistent with the new definition, what is not, and the order the
remaining work has to happen in. Re-run it after each stage to see the status
move.

    python tools/build_health_rerun_plan.py
"""
from __future__ import annotations

import argparse
import datetime
import glob
import html
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_HTML = ROOT / "docs" / "health_definition_rerun_plan.html"

#: The lists Section 2 derives from Industrial Healthy, which Section 3 loads.
LIST_FILES = ["vanish_plus_ids", "blossum_plus_ids", "vanishH_ids",
              "blossumH_ids"]

#: Procedural / metastatic flags that do not by themselves mean disease.
NOISY = {
    "disease_metastases", "disease_metastases_nodes", "disease_metastases_lung",
    "disease_metastases_liver", "disease_metastases_bone", "disease_metastases_SQ",
    "disease_metastases_adrenal", "disease_melanoma_surgery",
    "disease_carcinoma_surgery_history", "disease_treatment_colitis",
}


def mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return None


def fmt(ts):
    return (datetime.datetime.fromtimestamp(ts).strftime("%d %b %H:%M")
            if ts else "—")


def health_facts():
    """What `Unhealthy` now means, and how the cohort splits."""
    from persistence import config
    from persistence.common import is_yes

    new = pd.read_csv(config.META_FILE, low_memory=False)
    old_p = ROOT / "metadata_Aug_26_2026_final.csv"
    old = pd.read_csv(old_p, low_memory=False) if old_p.exists() else None

    def split(m):
        u = m["Unhealthy"].astype(str).str.strip()
        out = {}
        for lf in sorted(m["Lifestyle"].dropna().unique()):
            s = u[m["Lifestyle"] == lf]
            out[lf] = (int((s == "Healthy").sum()),
                       int((s == "Unhealthy").sum()),
                       int((~s.isin(["Healthy", "Unhealthy"])).sum()))
        u_all = u
        out["TOTAL"] = (int((u_all == "Healthy").sum()),
                        int((u_all == "Unhealthy").sum()),
                        int((~u_all.isin(["Healthy", "Unhealthy"])).sum()))
        return out

    dz = [c for c in new.columns
          if c.startswith("disease_") and c not in NOISY]
    anydz = pd.concat([new[c].apply(is_yes) for c in dz], axis=1).any(axis=1)
    obc = [c for c in config.OBESITY_COLUMNS if c in new.columns]
    ob = pd.concat([new[c].apply(is_yes) for c in obc], axis=1).any(axis=1)
    if "BMI" in new.columns:
        ob = ob | (pd.to_numeric(new["BMI"], errors="coerce") >= 30).fillna(False)
    ow = (new["Overweight"].apply(is_yes) if "Overweight" in new.columns
          else pd.Series(False, index=new.index))
    unh = new["Unhealthy"].astype(str).str.strip() == "Unhealthy"

    composition = [
        ("a recorded disease flag", int((unh & anydz).sum())),
        ("obese, with no disease flag", int((unh & ~anydz & ob).sum())),
        ("overweight only", int((unh & ~anydz & ~ob & ow).sum())),
        ("named only in the free-text disease field",
         int((unh & ~anydz & ~ob & ~ow).sum())),
    ]

    return dict(new=split(new), old=split(old) if old is not None else None,
                n_disease_cols=len(dz), n_noisy=len(NOISY),
                composition=composition, n_unhealthy=int(unh.sum()),
                n_obese=int(ob.sum()),
                obese_all_unhealthy=int((ob & ~unh).sum()) == 0,
                meta=Path(config.META_FILE).name,
                meta_time=mtime(config.META_FILE))


def stage_status(meta_ts):
    """Which pipeline stages postdate the metadata correction."""
    s2 = ROOT / "results/combined/section2_differential/main"
    lists = [(f, mtime(s2 / f"{f}.txt")) for f in LIST_FILES]
    list_ts = max([t for _f, t in lists if t], default=None)

    rows = []
    rows.append(dict(
        stage="Section 2 — differential prevalence",
        detail=("Derives VANISH+/BloSSUM+ from Industrial-all vs NI, and "
                "VanishH+/BlossumH+ from <b>Industrial Healthy</b> vs NI. The "
                "H+ lists move when the Healthy group moves."),
        ts=list_ts, ok=bool(list_ts and meta_ts and list_ts >= meta_ts),
        cmd="python -m persistence --ref combined --only s2"))

    s3 = mtime(ROOT / "results/combined/section3_health/main/"
                      "health_per_condition_cmh_10pct.csv")
    # Staleness propagates: running Section 3 after Section 2 is not enough if
    # Section 2 itself predates the metadata. Being newer than a stale input
    # does not make an output current.
    s2_ok = rows[0]["ok"]
    rows.append(dict(
        stage="Section 3 — health associations",
        detail=("Uses the Healthy/Unhealthy split directly, and loads the four "
                "list files Section 2 writes. Correct only if it runs "
                "<i>after</i> a current Section 2."),
        ts=s3, ok=bool(s2_ok and s3 and list_ts and s3 >= list_ts
                       and meta_ts and s3 >= meta_ts),
        cmd="python -m persistence --ref combined --only s3"))

    t1 = mtime(ROOT / "results/combined/table_s1a_study_summary_formatted.csv")
    rows.append(dict(
        stage="Table S1 — study summary",
        detail="Counts healthy and unhealthy samples per study.",
        ts=t1, ok=bool(t1 and meta_ts and t1 >= meta_ts),
        cmd="python -m persistence --ref combined --only table_s1a"))

    acc = sorted(glob.glob(str(ROOT / "results/combined/**/accum*.csv"),
                           recursive=True))
    acc_ts = max([mtime(p) for p in acc], default=None) if acc else None
    rows.append(dict(
        stage="Accumulation",
        detail=("Builds Industrial-Healthy and Industrial-Unhealthy "
                "sub-groups for its curves."),
        ts=acc_ts, ok=bool(acc_ts and meta_ts and acc_ts >= meta_ts),
        cmd="python -m persistence --ref combined --only accum"))

    da = mtime(ROOT / "results/combined/depth_audit/depth_summary.csv")
    rows.append(dict(
        stage="Depth audit",
        detail="Reports cohort Ns by health group; runs on every invocation.",
        ts=da, ok=bool(da and meta_ts and da >= meta_ts),
        cmd="(runs automatically with any reference)"))

    return rows, lists, list_ts


def figure_status(list_ts, meta_ts, s2_ok, s3_ok):
    """Figures are current only if the stage that produced them is.

    A figure drawn after a stale table is still stale; only the reference it
    was built from can make it current.
    """
    from tools import figure_sets
    out = []
    for slot, ref, stem in sorted(
            figure_sets.manifest(),
            key=lambda r: int("".join(c for c in r[0] if c.isdigit()) or 0)):
        p = ROOT / "results" / ref / f"{stem}.png"
        ts = mtime(p)
        sec = stem.split("/")[0]
        health = sec in ("section3_health",) or slot in ("S4", "S23")
        derived = sec in ("section2_differential",) or slot in (
            "S21", "S22", "S25", "S26", "S27", "S28", "S29", "S30")
        ref_ts = list_ts if derived else meta_ts
        stage_ok = s2_ok if derived else s3_ok
        out.append(dict(slot=slot, section=sec, ts=ts,
                        depends=health or derived,
                        ok=bool(stage_ok and ts and ref_ts and ts >= ref_ts)))
    return out


def other_refs(meta_ts):
    rows = []
    for ref in ("UHGG", "lev", "chimp"):
        p = ROOT / f"results/{ref}/section3_health/main/health_per_condition_cmh_10pct.csv"
        lp = ROOT / f"results/{ref}/section2_differential/main/vanishH_ids.txt"
        rows.append(dict(ref=ref, s3=mtime(p), s2=mtime(lp)))
    return rows


# ── rendering ─────────────────────────────────────────────────────────────
CSS = """
:root{
  --ground:#FAFAFB; --panel:#FFFFFF; --sunk:#F1F2F6;
  --ink:#171A21; --body:#333845; --muted:#5C6375; --faint:#8A90A2;
  --rule:#E3E5EC; --rule-strong:#C7CAD6;
  --accent:#3A4A7A; --accent-soft:#E9ECF6;
  --stop:#8E3B34; --stop-soft:#FBECEA; --stop-rule:#E0B4AE;
  --go:#1F5E48; --go-soft:#E5F2ED;
  --wait:#8A5A12; --wait-soft:#FBF2E1;
  --shadow:0 1px 2px rgba(23,26,33,.06);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0E1014; --panel:#161922; --sunk:#12141B;
    --ink:#E9EBF2; --body:#C5C9D6; --muted:#98A0B4; --faint:#6C7488;
    --rule:#262A36; --rule-strong:#394052;
    --accent:#93A6DA; --accent-soft:#1A2034;
    --stop:#E0897F; --stop-soft:#2A1815; --stop-rule:#5A302A;
    --go:#6FC3A3; --go-soft:#10241D;
    --wait:#DCA85C; --wait-soft:#241C10;
    --shadow:none;
  }
}
:root[data-theme="dark"]{
  --ground:#0E1014; --panel:#161922; --sunk:#12141B;
  --ink:#E9EBF2; --body:#C5C9D6; --muted:#98A0B4; --faint:#6C7488;
  --rule:#262A36; --rule-strong:#394052;
  --accent:#93A6DA; --accent-soft:#1A2034;
  --stop:#E0897F; --stop-soft:#2A1815; --stop-rule:#5A302A;
  --go:#6FC3A3; --go-soft:#10241D;
  --wait:#DCA85C; --wait-soft:#241C10;
  --shadow:none;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--body);
  font-family:"Source Sans 3",-apple-system,BlinkMacSystemFont,sans-serif;
  font-size:15.5px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:63rem;margin:0 auto;padding:3rem 1.5rem 7rem}
h1,h2,h3{font-family:Spectral,Georgia,serif;color:var(--ink);font-weight:600;
  text-wrap:balance;margin:0}
h1{font-size:2.3rem;line-height:1.14;letter-spacing:-.011em}
h2{font-size:1.38rem;margin:3rem 0 .5rem;padding-top:1.25rem;
  border-top:1px solid var(--rule)}
h3{font-size:1.02rem;margin:1.7rem 0 .4rem}
p{margin:.68rem 0;max-width:50rem}
.lede{font-size:1.06rem;color:var(--muted);max-width:46rem}
.eyebrow{font-size:.7rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);font-weight:700;margin-bottom:.5rem}
.step{font-family:"IBM Plex Mono";font-size:.76rem;color:var(--faint);
  margin-right:.5rem}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  font-size:.845em}
code{background:var(--sunk);padding:.08em .34em;border-radius:3px}
pre{background:var(--panel);border:1px solid var(--rule);border-radius:6px;
  padding:.8rem 1rem;overflow-x:auto;margin:.75rem 0;font-size:.82rem;
  line-height:1.55;box-shadow:var(--shadow)}
pre code{background:none;padding:0}
.stop{background:var(--stop-soft);border:1px solid var(--stop-rule);
  border-left:3px solid var(--stop);border-radius:6px;padding:1.05rem 1.3rem;
  margin:1.5rem 0}
.stop h3{margin-top:0;color:var(--stop)}
.note{background:var(--panel);border:1px solid var(--rule);
  border-left:3px solid var(--accent);border-radius:6px;padding:.85rem 1.1rem;
  margin:1rem 0}
.note p,.stop p{max-width:none}
.note p{margin:.2rem 0}
.scroll{overflow-x:auto;margin:.9rem 0}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th{text-align:left;font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);font-weight:700;padding:.42rem .6rem;
  border-bottom:1px solid var(--rule-strong);white-space:nowrap}
td{padding:.48rem .6rem;border-bottom:1px solid var(--rule);vertical-align:top}
tbody tr:hover{background:var(--sunk)}
td.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.tag{display:inline-block;font-size:.66rem;letter-spacing:.05em;font-weight:700;
  text-transform:uppercase;padding:.11rem .44rem;border-radius:3px;
  white-space:nowrap}
.tag.ok{background:var(--go-soft);color:var(--go)}
.tag.stale{background:var(--stop-soft);color:var(--stop)}
.tag.wait{background:var(--wait-soft);color:var(--wait)}
.tag.na{background:var(--sunk);color:var(--faint)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));
  gap:.75rem;margin:1.4rem 0}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:6px;
  padding:.8rem .9rem;box-shadow:var(--shadow)}
.card b{display:block;font-family:Spectral,Georgia,serif;font-size:1.5rem;
  line-height:1.1;color:var(--ink);font-variant-numeric:tabular-nums}
.card span{font-size:.74rem;color:var(--muted);display:block;margin-top:.1rem}
.chain{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;margin:1rem 0;
  font-family:"IBM Plex Mono";font-size:.8rem}
.chain i{font-style:normal;background:var(--panel);border:1px solid var(--rule);
  border-radius:5px;padding:.35rem .6rem}
.chain u{text-decoration:none;color:var(--faint)}
.draft{background:var(--panel);border:1px solid var(--rule);
  border-left:3px solid var(--go);border-radius:6px;padding:1rem 1.25rem;
  margin:1rem 0;font-size:.94rem}
.draft p{margin:0 0 .7rem}
.draft p:last-child{margin:0}
ul,ol{max-width:50rem;padding-left:1.15rem}
li{margin:.3rem 0}
.meta{color:var(--faint);font-size:.79rem;margin-top:2.4rem}
"""


def esc(s):
    return html.escape(str(s))


def render(F, stages, lists, list_ts, figs, refs):
    L = []
    A = L.append
    meta_ts = F["meta_time"]
    blocked = not stages[0]["ok"]

    A("<title>Unhealthy Definition Rerun</title>")
    A('<link rel="preconnect" href="https://fonts.googleapis.com">')
    A('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
    A('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      'family=Spectral:wght@400;600&family=Source+Sans+3:wght@400;600&'
      'family=IBM+Plex+Mono:wght@400;500&display=swap">')
    A(f"<style>{CSS}</style>")
    A('<div class="wrap">')
    A('<div class="eyebrow">Cohort definition · regeneration plan</div>')
    A("<h1>Making every artefact agree on who is unhealthy</h1>")
    A('<p class="lede">Obesity now counts as unhealthy, which moved 145 samples '
      "out of the Industrial Healthy group. That group is not only the control "
      "arm of Section 3 — it is also what Section 2 derives the VanishH+ and "
      "BlossumH+ lists from, so the change reaches further than the section it "
      "was made for. This is what still has to be rebuilt, and in what order.</p>")

    n = F["new"]["TOTAL"]
    o = F["old"]["TOTAL"] if F["old"] else (0, 0, 0)
    A('<div class="cards">')
    for big, lab in ((f"{n[1]:,}", "samples now Unhealthy"),
                     (f"+{n[1] - o[1]:,}", "moved by the obesity rule"),
                     (f"{F['n_disease_cols']}", "disease flags that count"),
                     (f"{sum(1 for f in figs if f['depends'] and not f['ok'])}",
                      "shipped figures still stale")):
        A(f'<div class="card"><b>{big}</b><span>{lab}</span></div>')
    A("</div>")

    # ── the blocking finding ──────────────────────────────────────────────
    if blocked:
        A('<div class="stop"><h3>Order of operations: Section 3 has already '
          "been run on the wrong lists</h3>")
        A("<p>Section 2 Pass 3 derives <b>VanishH+</b> and <b>BlossumH+</b> by "
          "comparing <i>Industrial Healthy</i> against Non-Industrial, and "
          "writes them to four <code>*_ids.txt</code> files. Section 3 does not "
          "recompute them — it loads those files, by design, so that "
          "<code>--only s3</code> can run standalone.</p>")
        A("<p>Those files were written <b>" + fmt(list_ts) +
          "</b>; the corrected metadata dates from <b>" + fmt(meta_ts) +
          "</b>. So the Section 3 results currently on disk use the new "
          "Healthy/Unhealthy split for the case/control test while testing "
          "lists that were built from the <i>old</i> split. The two halves "
          "disagree about who was healthy.</p>")
        A("<p><b>Nothing downstream can be trusted until Section 2 is re-run "
          "first.</b> The Section 3 runs already completed — combined, and then "
          "UHGG/lev/chimp — have to be repeated after it.</p></div>")

    A('<div class="chain">'
      '<i>metadata</i><u>&rarr;</u><i>Section 2 &mdash; H+ lists</i>'
      '<u>&rarr;</u><i>Section 3</i><u>&rarr;</u><i>figures &amp; tables</i>'
      '<u>&rarr;</u><i>manuscript numbers</i></div>')

    # ── what Unhealthy means ──────────────────────────────────────────────
    A("<h2><span class='step'>01</span>What “Unhealthy” now means</h2>")
    A("<p>It is a curated label, not a rule evaluated at run time: it was set "
      "per sample from the recorded disease flags, corrected once against the "
      "free-text disease field, and now corrected again so that obesity counts. "
      f"Of the {F['n_unhealthy']:,} samples labelled Unhealthy:</p>")
    A('<div class="scroll"><table><thead><tr><th>Reason the sample is '
      "unhealthy</th><th class='num'>Samples</th></tr></thead><tbody>")
    for why, cnt in F["composition"]:
        A(f'<tr><td>{esc(why)}</td><td class="num">{cnt:,}</td></tr>')
    A("</tbody></table></div>")
    A(f"<p>{F['n_disease_cols']} <code>disease_*</code> columns count toward "
      f"the label; {F['n_noisy']} are excluded because they record a procedure "
      "or a metastatic site rather than a primary condition. All "
      f"{F['n_obese']:,} obese samples are now Unhealthy"
      + (", with none left labelled Healthy." if F["obese_all_unhealthy"]
         else " — <b>except some still labelled Healthy, which is a bug.</b>")
      + "</p>")

    A("<h3>Cohort split, before and after</h3>")
    A('<div class="scroll"><table><thead><tr><th>Lifestyle</th>'
      "<th class='num'>Healthy before</th><th class='num'>Healthy after</th>"
      "<th class='num'>Unhealthy before</th><th class='num'>Unhealthy after</th>"
      "<th class='num'>Unlabelled</th></tr></thead><tbody>")
    for lf in F["new"]:
        a = F["new"][lf]
        b = F["old"].get(lf, (0, 0, 0)) if F["old"] else (0, 0, 0)
        strong = ' style="font-weight:600"' if lf == "TOTAL" else ""
        A(f'<tr{strong}><td>{esc(lf)}</td>'
          f'<td class="num">{b[0]:,}</td><td class="num">{a[0]:,}</td>'
          f'<td class="num">{b[1]:,}</td><td class="num">{a[1]:,}</td>'
          f'<td class="num">{a[2]:,}</td></tr>')
    A("</tbody></table></div>")
    A('<div class="note"><p>The pipeline reports smaller numbers than these: '
      "it drops samples at loading and promotes technical replicates before "
      "any of it is counted. Quote the pipeline’s figures in the "
      "manuscript, not the raw metadata’s.</p></div>")

    # ── draft text ────────────────────────────────────────────────────────
    A("<h2><span class='step'>02</span>Text to add</h2>")
    A("<h3>Supplement — a paragraph defining the category</h3>")
    A('<div class="draft">')
    A("<p><b>Defining health status.</b> Each sample carries a binary health "
      "label used throughout Section 3 as the case/control split. A participant "
      "is counted as <i>unhealthy</i> if the source study recorded any primary "
      "disease, if the study’s free-text disease field names a condition, "
      "or if the participant was obese or overweight; everyone else is counted "
      f"as <i>healthy</i>. {F['n_disease_cols']} disease fields contribute, "
      "spanning gastrointestinal and metabolic disease (inflammatory bowel "
      "disease, Crohn’s disease, ulcerative colitis, colorectal cancer and "
      "adenoma, type 1 and type 2 diabetes, impaired glucose tolerance, "
      "metabolic dysfunction-associated steatotic liver disease, cirrhosis, "
      "hypertension) and disease elsewhere in the body (atherosclerotic "
      "cardiovascular disease, multiple sclerosis, Parkinson’s disease, "
      "rheumatoid arthritis, ankylosing spondylitis, melanoma, schizophrenia, "
      "myalgic encephalomyelitis). Flags that record a procedure or a "
      "metastatic site rather than a primary condition are excluded, since "
      "they describe the management of a disease already counted. Obesity is "
      "included because it is a disease state that alters the gut microbiome "
      "in its own right; treating it as health would have placed obese "
      "participants in the control group of every other condition tested. "
      f"Applying this definition leaves {n[1]:,} unhealthy and {n[0]:,} healthy "
      "samples in the cohort.</p>")
    A("</div>")

    A("<h3>Main text — one sentence</h3>")
    A('<div class="draft"><p>Participants were classified as unhealthy if the '
      "source study recorded any primary disease, or if they were obese or "
      "overweight, and as healthy otherwise (Note S<i>n</i>).</p></div>")
    A('<div class="note"><p>Both drafts quote counts that this page reads from '
      "the metadata. Regenerate the page after the reruns and take the numbers "
      "from the pipeline’s own cohort table rather than from here, so the "
      "manuscript and Table S1 cannot disagree.</p></div>")

    # ── the order ─────────────────────────────────────────────────────────
    A("<h2><span class='step'>03</span>What to run, in order</h2>")
    A('<div class="scroll"><table><thead><tr><th></th><th>Stage</th>'
      "<th>Why it depends on the definition</th><th>Last built</th>"
      "</tr></thead><tbody>")
    for st in stages:
        tag = ('<span class="tag ok">current</span>' if st["ok"]
               else '<span class="tag stale">rebuild</span>')
        A(f'<tr><td>{tag}</td><td><b>{esc(st["stage"])}</b><br>'
          f'<code>{esc(st["cmd"])}</code></td>'
          f'<td style="color:var(--muted)">{st["detail"]}</td>'
          f'<td class="mono">{fmt(st["ts"])}</td></tr>')
    A("</tbody></table></div>")
    A("<pre><code># the whole chain for the manuscript reference, in order\n"
      "python -m persistence --ref combined --only s2\n"
      "python -m persistence --ref combined --only s3\n"
      "python -m persistence --ref combined --only table_s1a accum\n\n"
      "# then the other three references\n"
      "python -m persistence --ref UHGG lev chimp --only s2 s3\n\n"
      "# then redraw the figures from the refreshed data objects, which is\n"
      "# what applies any figure-code changes made since the run started\n"
      "python tools/replot_from_data.py --all\n"
      "python tools/build_supp_review.py</code></pre>")
    A('<div class="note"><p><b>Why the redraw step is not optional.</b> A run '
      "imports the figure code once, at launch. Any edit made while it is "
      "running is not in the figures it writes — which is exactly what "
      "happened on the last pass, and why S27–S31 had to be redrawn "
      "afterwards. Redrawing from the data objects is seconds per figure and "
      "removes the question.</p></div>")

    # ── figures ───────────────────────────────────────────────────────────
    A("<h2><span class='step'>04</span>Shipped figures that depend on it</h2>")
    dep = [f for f in figs if f["depends"]]
    A(f"<p>{len(dep)} of {len(figs)} supplementary figures are built from the "
      "health split or from the lists derived through it. The rest are "
      "lifestyle-only and are unaffected.</p>")
    A('<div class="scroll"><table><thead><tr><th></th><th>Figure</th>'
      "<th>Section</th><th>Last built</th></tr></thead><tbody>")
    for f in dep:
        tag = ('<span class="tag ok">current</span>' if f["ok"]
               else '<span class="tag stale">rebuild</span>')
        A(f'<tr><td>{tag}</td><td class="mono">{esc(f["slot"])}</td>'
          f'<td style="color:var(--muted)">{esc(f["section"])}</td>'
          f'<td class="mono">{fmt(f["ts"])}</td></tr>')
    A("</tbody></table></div>")

    # ── other references ──────────────────────────────────────────────────
    A("<h2><span class='step'>05</span>The other three references</h2>")
    A("<p>Nothing in the manuscript is drawn from these, but they carry "
      "Section 3 outputs and the same ordering problem applies to each.</p>")
    A('<div class="scroll"><table><thead><tr><th>Reference</th>'
      "<th>Section 2 lists</th><th>Section 3</th><th></th></tr></thead><tbody>")
    for r in refs:
        good = bool(r["s2"] and r["s3"] and meta_ts and r["s2"] >= meta_ts
                    and r["s3"] >= r["s2"])
        tag = ('<span class="tag ok">consistent</span>' if good
               else '<span class="tag stale">rebuild</span>')
        A(f'<tr><td class="mono">{esc(r["ref"])}</td>'
          f'<td class="mono">{fmt(r["s2"])}</td>'
          f'<td class="mono">{fmt(r["s3"])}</td><td>{tag}</td></tr>')
    A("</tbody></table></div>")

    # ── verification ──────────────────────────────────────────────────────
    A("<h2><span class='step'>06</span>How to know it worked</h2>")
    A("<pre><code># 1. no obese sample is labelled healthy, and the run says so\n"
      "grep 'Obesity within Industrial Healthy' <run log>   # expect 0 / N\n\n"
      "# 2. the H+ lists were rebuilt after the metadata, not before\n"
      "ls -l results/combined/section2_differential/main/*_ids.txt\n\n"
      "# 3. the sensitivity comparison that is now empty says so rather than\n"
      "#    reporting a result\n"
      "grep 'obese healthy-only' <run log>                  # expect g2=0\n\n"
      "# 4. every figure postdates the lists it was drawn from\n"
      "python tools/build_health_rerun_plan.py              # this page\n\n"
      "# 5. the cohort table and the manuscript agree on the same N\n"
      "python -m persistence --ref combined --only table_s1a</code></pre>")
    A('<div class="note"><p><b>One result to expect, and not to mistake for a '
      "bug.</b> <code>Obese (healthy only)</code> is now empty by construction "
      "— it asks for samples that are both healthy and obese, of which there "
      "are none. It self-skips on the too-few-samples guard. That sensitivity "
      "analysis no longer exists and should not be cited.</p></div>")

    A(f'<p class="meta">Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by '
      "<code>tools/build_health_rerun_plan.py</code>, reading "
      f"<code>{esc(F['meta'])}</code> and the state of "
      "<code>results/</code>. Re-run it after each stage.</p>")
    A("</div>")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_HTML))
    a = ap.parse_args()

    F = health_facts()
    stages, lists, list_ts = stage_status(F["meta_time"])
    figs = figure_status(list_ts, F["meta_time"],
                         s2_ok=stages[0]["ok"], s3_ok=stages[1]["ok"])
    refs = other_refs(F["meta_time"])

    Path(a.out).write_text(render(F, stages, lists, list_ts, figs, refs))
    print(f"  {a.out}")
    print(f"    Unhealthy now {F['new']['TOTAL'][1]:,} "
          f"(was {F['old']['TOTAL'][1]:,})" if F["old"] else "")
    for st in stages:
        print(f"    {'OK    ' if st['ok'] else 'REBUILD'} {st['stage']}")
    dep = [f for f in figs if f["depends"]]
    print(f"    figures depending on the split: {len(dep)}, "
          f"{sum(1 for f in dep if not f['ok'])} stale")


if __name__ == "__main__":
    main()
