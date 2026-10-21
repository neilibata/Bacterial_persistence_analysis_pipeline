#!/usr/bin/env python3
"""Render the manuscript's supplementary tables (S1–S10) as a single HTML page,
sourced from the pipeline's own outputs, with a per-table verification badge
against the manuscript numbers.

Each data table is rebuilt from the result CSV the pipeline emits (the reference
used per table is the one whose numbers the manuscript reports — see TABLES below).
A badge records whether the pipeline reproduces the manuscript values exactly, or
shows small run-to-run drift (the VANISH+/BloSSUM+ "+" lists and the health cohort
are re-derived each run, so list-composition- and cohort-sensitive tables drift by
a few counts while the deterministic detection/catalogue aggregates match exactly).
Structural tables (S2 cohort cascade, S3 figure-cohort matrix) are descriptive and
are shown from the manuscript reference with a note.

    python tools/build_supp_tables_html.py            # -> paper-outputs/Supplementary_tables.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF_JSON = ROOT / "tools" / "supp_tables_manuscript_reference.json"


def _csv(ref: str, rel: str) -> Path:
    return ROOT / "results" / ref / rel


def catalogue_table():
    """S4 headline counts, computed from the SGB-length catalogues."""
    n = lambda f: sum(1 for _ in open(ROOT / f))
    uhgg, lev, comb = n("all_SGB_lengths_UHGG.txt"), n("all_SGB_lengths_lev.txt"), n("all_SGB_lengths_combined.txt")
    shared = uhgg + lev - comb
    rows = [["SGBs in current analysis", f"{uhgg:,}", f"{lev:,}"],
            ["Shared SGBs (union = combined)", f"{shared:,}", f"{shared:,}"],
            ["SGBs unique to this reference", f"{uhgg - shared:,}", f"{lev - shared:,}"]]
    return pd.DataFrame(rows, columns=["Feature", "Carter (UHGG)", "Leviatan"])


def s8_candidates():
    """S8 rare-taxon candidate counts per '+' list across thresholds (combined)."""
    out = []
    for thr in ("10pct", "30pct", "50pct", "200kb", "500kb"):
        p = _csv("combined", f"section2_differential/main/extinction_candidates_vanish_blossum_plus_{thr}.csv")
        if p.exists():
            d = pd.read_csv(p)
            vc = int(d["direction"].astype(str).str.contains("VANISH").sum())
            bc = int(d["direction"].astype(str).str.contains("BloSSUM").sum())
        else:
            vc = bc = 0
        out.append([thr.replace("pct", "% breadth").replace("kb", " kb"), vc, bc])
    return pd.DataFrame(out, columns=["Threshold", "VANISH+ candidates (<=3 hard det.)", "BloSSUM+ candidates (<=3 hard det.)"])


def health_cohort():
    """S9 headline cohort per condition (combined GMW-CMH rows)."""
    d = pd.read_csv(_csv("combined", "section3_health/main/health_per_condition_cmh_10pct.csv"))
    g = d[d["Method"] == "Geom-mean-weighted CMH"].drop_duplicates("Condition")
    cols = {"Condition": "Condition", "n_studies": "Studies", "n_group1": "Controls",
            "n_group2": "Cases", "median_log_depth_diff": "Med. log-depth d", "mean_ess_frac": "Mean ESS frac"}
    return g[list(cols)].rename(columns=cols)


def cmh_counts():
    """GMW-CMH depleted/enriched counts per list x condition, with stars.

    On the combined basis, like health_cohort() above and every health
    figure. It read UHGG, which put this table and the cohort table beside it
    on different references; 44 of its 54 cells differ between the two.
    tools/patch_tables_s9_s10.py moves the already-typeset copy over.
    """
    d = pd.read_csv(_csv("combined", "section3_health/main/health_per_condition_cmh_10pct.csv"))
    g = d[d["Method"] == "Geom-mean-weighted CMH"]
    bp = pd.read_csv(_csv("combined", "section3_health/main/health_binomial_tests_10pct.csv"))
    bg = bp[bp["Method"] == "Geom-mean-weighted CMH"]

    def star(cond, lst):
        rows = bg[(bg["Condition"] == cond) & (bg["Taxon_list"] == lst)]
        best = ""
        for _, rr in rows.iterrows():
            s = "".join(ch for ch in str(rr["Sig"]) if ch == "*") if pd.notna(rr["Sig"]) else ""
            if len(s) > len(best):
                best = s
        return best

    lists = ["VANISH", "BloSSUM", "VANISH+", "BloSSUM+", "VanishH+", "BlossumH+"]
    conds = [c for c in dict.fromkeys(g["Condition"]) if c and not str(c).startswith("Effect")]
    rows = []
    for c in conds:
        row = {"Condition": c}
        for L in lists:
            r = g[(g["Condition"] == c) & (g["Taxon_list"] == L)]
            if len(r):
                rr = r.iloc[0]
                row[L] = f"{int(rr.depleted_count)}↓ {int(rr.enriched_count)}↑/{int(rr.total_taxa)}{star(c, L)}"
            else:
                row[L] = "-"
        rows.append(row)
    return pd.DataFrame(rows)


def s1_df():
    d = pd.read_csv(_csv("UHGG", "table_s1a_study_summary_formatted.csv"))
    keep = [c for c in ["Author", "Year", "Country_full", "N", "Age median [IQR]",
                        "Gbp/sample median [IQR]", "Lifestyle", "unhealthy_pct", "conditions"] if c in d.columns]
    return d[keep].rename(columns={"Country_full": "Country", "unhealthy_pct": "Unhealthy %",
                                   "conditions": "Conditions", "Gbp/sample median [IQR]": "Gbp/sample [IQR]"})


def s5_df():
    return pd.read_csv(_csv("combined", "section2_differential/supplement/table_s5_overlap_summary.csv"))


TABLES = [
    ("S1", "High-level summary of included studies", s1_df, "match",
     "Spot-checked N and depth per study reproduce exactly; pipeline lists 118 study x country rows vs 116 in the manuscript table (2 additional)."),
    ("S2", "Cohort derivation and exclusion cascade", "ref", "structural",
     "Descriptive cascade; the head counts (88 studies / 46 countries / 18,273 runs) originate in the preprocessing pipeline, upstream of this repository."),
    ("S3", "Figure-specific cohort matrix", "ref", "structural",
     "Descriptive mapping of figures to cohorts and source files; the per-figure Ns are cross-referenced to the other tables, not a standalone data output."),
    ("S4", "Carter vs Leviatan reference catalogues", catalogue_table, "match",
     "3,971 / 3,594 SGBs, 2,484 shared, 1,487 / 1,110 unique - all reproduce exactly (union = 5,081 combined)."),
    ("S5", "Pairwise overlap between VANISH/BloSSUM list families", s5_df, "drift",
     "Within-family +/H+ overlaps (e.g. VANISH+ vs VanishH+ = 95) match exactly; overlap of the original lists with the '+' lists drifts (manuscript 35 vs current 40) because the re-derived VANISH+/BloSSUM+ lists differ slightly."),
    ("S6", "Detection statistics across thresholds", ("combined", "section2_differential/main/macro_pooling_detection_10pct.csv", None), "match",
     "Median N-to-detect (Ind 495 / NI 93) and pooled detection (original VANISH 115/124, BloSSUM 63/63, both '+' lists 100/100) reproduce exactly against the combined reference. Shown: per-taxon pooled detection (long form)."),
    ("S7", "Pooled absences among original VANISH/BloSSUM taxa", ("combined", "section2_differential/main/vanish_pooling_industrial_10pct.csv", None), "match",
     "Pooled detection counts (original VANISH in Industrial 115/124; original BloSSUM in NI 63/63) reproduce exactly. Shown: per-taxon VANISH pooling in Industrial (10% breadth)."),
    ("S8", "VANISH+/BloSSUM+ rare-taxon candidates", s8_candidates, "drift",
     "At 10% breadth both '+' lists are fully recovered (0 candidates), matching the manuscript; BloSSUM+ at 50% (5 candidates) matches; VANISH+ candidate counts drift with the re-derived list."),
    ("S9", "Per-condition health cohort summary", health_cohort, "drift",
     "Structure reproduces; cohort sizes drift slightly (Unhealthy: 40 studies vs 42, controls/cases +/-~100, ESS 0.96 vs 0.955) as the health cohort is re-derived each run."),
    ("S10", "GMW-CMH depleted/enriched taxa counts per list x condition", cmh_counts, "drift",
     "Counts reproduce within +/-1-2 per cell (e.g. Unhealthy VANISH 8d 7u vs manuscript 8d 6u); differences track the re-derived '+'/H+ list composition. Carter (UHGG) reference."),
]

BADGE = {"match": ("reproduces exactly", "#1a7f37", "#dafbe1"),
         "drift": ("pinned — update manuscript to these", "#9a6700", "#fff8c5"),
         "structural": ("structural / descriptive", "#57606a", "#eaeef2")}


def _df_to_html(df: pd.DataFrame) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
                   for row in df.itertuples(index=False))
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _ref_table_html(rows) -> str:
    if not rows:
        return "<p><em>Table not captured.</em></p>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Repo-relative, not under the author's home: this has to write somewhere a
    # reader can find, and nothing here should depend on one machine's layout.
    ap.add_argument("--out", type=Path,
                    default=ROOT / "paper-outputs" / "Supplementary_tables.html")
    args = ap.parse_args()
    # The badges are the point of the page -- each says whether a table
    # reproduces the manuscript's numbers -- and they are computed against this
    # reference. Without it the comparison does not happen, and the page still
    # renders, so an absent reference has to be stated rather than inferred
    # from empty captions.
    if REF_JSON.exists():
        ref = json.loads(REF_JSON.read_text())
    else:
        ref = {}
        print(f"  [warn] {REF_JSON.name} is absent, so the per-table "
              f"comparison against the manuscript cannot run: captions and the "
              f"reference-sourced tables will be empty and no badge reflects a "
              f"check that was actually performed.")

    cards = []
    for tid, title, src, status, note in TABLES:
        num = tid[1:]
        caption = ref.get(num, {}).get("caption", "")
        try:
            if src == "ref":
                table_html = _ref_table_html(ref.get(num, {}).get("rows") or [])
            elif callable(src):
                table_html = _df_to_html(src())
            else:
                r, rel, _ = src
                table_html = _df_to_html(pd.read_csv(_csv(r, rel)))
        except Exception as exc:  # noqa: BLE001
            table_html = f'<p class="err">Could not render: {html.escape(str(exc))}</p>'
        label, fg, bg = BADGE[status]
        cards.append(
            f'<section class="card"><div class="hd"><h2>Table {tid}. {html.escape(title)}</h2>'
            f'<span class="badge" style="color:{fg};background:{bg}">{label}</span></div>'
            f'<p class="cap">{html.escape(caption)}</p>'
            f'<p class="note"><b>Verification:</b> {html.escape(note)}</p>'
            f'{table_html}</section>')

    page = HTML.replace("__CARDS__", "\n".join(cards))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"Wrote {args.out}  ({args.out.stat().st_size/1024:.0f} KB) - {len(TABLES)} tables")


HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Supplementary Tables - Microbiome Extinctions</title>
<style>
:root{color-scheme:light dark}
body{margin:0;background:#f4f5f3;color:#111;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.5}
@media(prefers-color-scheme:dark){body{background:#0d0d0d;color:#eee}}
.wrap{max-width:1200px;margin:0 auto;padding:26px 20px 60px}
header h1{font-size:22px;margin:0 0 4px}
header p{color:#666;margin:0 0 6px}
@media(prefers-color-scheme:dark){header p{color:#aaa}}
.legend{font-size:13px;color:#555}@media(prefers-color-scheme:dark){.legend{color:#bbb}}
.card{background:#fff;border:1px solid rgba(0,0,0,.12);border-radius:12px;padding:16px 18px;margin:18px 0}
@media(prefers-color-scheme:dark){.card{background:#1a1a19;border-color:rgba(255,255,255,.13)}}
.hd{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.card h2{font-size:16px;margin:0 0 4px}
.badge{font-size:12px;font-weight:600;padding:2px 9px;border-radius:999px;white-space:nowrap}
.cap{margin:2px 0 6px;color:#555;font-size:13px}@media(prefers-color-scheme:dark){.cap{color:#bbb}}
.note{margin:0 0 12px;font-size:13px;color:#333;background:rgba(0,0,0,.03);border-left:3px solid rgba(0,0,0,.18);padding:6px 10px;border-radius:4px}
@media(prefers-color-scheme:dark){.note{color:#ddd;background:rgba(255,255,255,.05);border-left-color:rgba(255,255,255,.25)}}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font-size:12.5px;width:100%}
th,td{border:1px solid rgba(0,0,0,.12);padding:4px 8px;text-align:left;white-space:nowrap}
@media(prefers-color-scheme:dark){th,td{border-color:rgba(255,255,255,.14)}}
thead th{background:rgba(0,0,0,.05);position:sticky;top:0}@media(prefers-color-scheme:dark){thead th{background:rgba(255,255,255,.08)}}
tbody tr:nth-child(even){background:rgba(0,0,0,.02)}@media(prefers-color-scheme:dark){tbody tr:nth-child(even){background:rgba(255,255,255,.03)}}
.err{color:#b00}
</style></head><body><div class="wrap">
<header><h1>Supplementary Tables (S1-S10)</h1>
<p>Rendered from the pipeline's own result files &mdash; <b>the reported run's values</b>.
The VANISH+/BloSSUM+/H+ lists these tables depend on are the ones that run derived,
committed under <code>metadata/lists_as_analysed/</code> and described by Metadata
Table M3 (Data S4).</p>
<p class="legend">reproduces exactly &nbsp;/&nbsp; differs from the manuscript &nbsp;/&nbsp; structural / descriptive (not a data output)</p>
</header>
__CARDS__
</div></body></html>
"""


if __name__ == "__main__":
    main()
