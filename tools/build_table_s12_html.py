"""Preview of the Table S12 Word design, and a browser-paste alternative."""
from __future__ import annotations
import os, sys, html as H
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_table_s12_docx import (fp, stars, ORDER, LISTS, DISPLAY, COLS,
                                  CAPTION, FOOTNOTES, DIR_FROM)

def build(csv_path, out_path):
    d = pd.read_csv(csv_path)
    d["_c"] = d["Condition"].map(lambda c: ORDER.index(c) if c in ORDER else 99)
    d["_l"] = d["Taxon_list"].map(lambda l: LISTS.index(l) if l in LISTS else 99)
    d = d.sort_values(["_l", "_c"]).reset_index(drop=True)

    head = ('<tr class="grp"><th colspan="%d">Two-sided, referenced to the '
            'background</th><th colspan="%d">One-sided, in the dominant '
            'direction</th></tr><tr>' % (DIR_FROM, len(COLS) - DIR_FROM)) + \
        "".join(f'<th>{H.escape(n).replace(chr(10), "<br>")}</th>'
                for n, _, _ in COLS) + "</tr>"
    body, last = [], None
    for _, r in d.iterrows():
        elig = str(r.get("eligible", "True")).lower() == "true"
        lname = DISPLAY.get(r["Taxon_list"], r["Taxon_list"])
        if lname != last:
            body.append(f'<tr class="band"><td colspan="{len(COLS)}">'
                        f'{H.escape(lname)}</td></tr>')
            last = lname
        z1, z2 = r.get("mean_z_10pct"), r.get("mean_z_300kb")
        hmp = r.get("p_meanz_hmp")
        sig = pd.notna(hmp) and hmp < 0.05
        dagger = "" if bool(r.get("sign_agree", True)) else \
            '<span class="dag">†</span>'
        agree = "" if pd.isna(r.get("n_methods_tested")) else \
            f"{int(r['n_methods_sig'])}/{int(r['n_methods_tested'])}"
        cells = [
            H.escape(DISPLAY.get(r["Condition"], r["Condition"])),
            H.escape(lname),
            "" if pd.isna(r.get("n_list_10pct")) else str(int(r["n_list_10pct"])),
            (f"{z1:+.2f} / {z2:+.2f}" if pd.notna(z1) and pd.notna(z2)
             else "—") + dagger,
            f"{fp(r.get('p_meanz_10pct'))} / {fp(r.get('p_meanz_300kb'))}",
            (f'<b>{fp(hmp)} {stars(hmp)}</b>' if sig else fp(hmp)),
            fp(r.get("p_energy_hmp")),
            agree,
            f'<i>{H.escape(str(r.get("direction","") or "").replace("nan",""))}</i>',
            fp(r.get("p_iw_fisher_hmp")),
            fp(r.get("p_iw_directional_hmp")),
        ]
        tds = "".join(
            f'<td class="{al}">{v}</td>'
            for v, (_, _, al) in zip(cells, COLS))
        body.append(f'<tr class="{"inelig" if not elig else ""}">{tds}</tr>')

    n_sign = int((~d.get("sign_agree", pd.Series([True] * len(d)))
                  .astype(bool)).sum())
    css = open(os.path.join(HERE, "_section6_page.css")).read()
    page = f"""<title>Table S12 Design</title>
{css}
<style>
.s12{{border-collapse:collapse;font-size:.72rem;width:100%;
  font-variant-numeric:tabular-nums}}
.s12 th{{background:var(--sunk);font-weight:700;text-transform:none;
  letter-spacing:0;font-size:.68rem;padding:.35rem .4rem;
  border-top:1.5px solid var(--ink);border-bottom:1.5px solid var(--ink);
  vertical-align:bottom;white-space:normal}}
.s12 td{{padding:.26rem .4rem;border:0}}
.s12 tr.band td{{background:var(--sunk);font-weight:700;
  border-top:1px solid var(--rule-strong);padding-top:.34rem}}
.s12 tr.inelig td{{color:var(--faint);font-style:italic}}
.s12 td.center{{text-align:center}} .s12 td.left{{text-align:left}}
.s12 tbody tr:last-child td{{border-bottom:1.5px solid var(--ink)}}
.dag{{color:var(--stop);font-weight:700}}
.s12 tr.grp th{{border-top:1.5px solid var(--ink);border-bottom:1px solid var(--rule-strong);
  font-size:.66rem;font-style:italic;font-weight:600}}
.cap{{font-size:.8rem;color:var(--body);max-width:none;margin:0 0 .6rem}}
.fn{{font-size:.72rem;color:var(--muted);max-width:none;margin:.6rem 0 0}}
</style>
<div class="wrap">
<h1>Table S12, as it will look in Word</h1>
<p class="lede">The CSV has 31 columns and 54 rows, which is a data file rather than a table. This is the same content collapsed to the comparison the table exists to make, at the size and weight a supplement prints at.</p>

<div class="note">
<p><b>Two ways to get it into Word.</b> Either open <code>Table_S12.docx</code>, click the table's move handle to select it, copy, and paste into the supplement &mdash; Word keeps the borders, shading, fonts and the landscape column widths. Or select the table below in the browser and paste with <b>Paste Special &rarr; Keep Source Formatting</b>; that carries the structure and the bold but not the page orientation.</p>
<p>The table is landscape. If the supplement is portrait throughout, insert a continuous section break around it rather than shrinking the type below 8&nbsp;pt.</p>
</div>

<h2>The design, and why</h2>
<ul>
<li><b>Two specifications paired inside one cell</b> (<code>+1.44 / +1.46</code>) rather than given a column each. That is what takes it from 31 columns to 10, and it puts the two numbers a reader is comparing next to each other instead of eight columns apart.</li>
<li><b>Grouped by taxon list</b>, conditions within each group in the order Figure 5 plots them, then the four excluded contrasts.</li>
<li><b>Excluded contrasts kept, in grey italic</b>, not dropped. A reader comparing methods most wants the contrasts that were excluded for want of studies rather than for want of signal.</li>
<li><b>Stars only on the combined primary p-value.</b> Starring every column invites reading across the row as five independent results, which they are not: they are five views of one comparison.</li>
<li><b>The bootstrap floor is shown, not hidden</b> &mdash; <code>&le;0.0005</code>, never <code>&lt;0.001</code>, because 2,000 replicates cannot resolve below 1/2,000 and writing <code>0.000</code> would claim they could.</li>
<li><b>Two column groups, labelled.</b> The mean z and the energy distance are two-sided and referenced to the background, so counting how many agree is meaningful. The two importance-weighted tests are one-sided in whichever direction dominates the list's raw counts, so they are reported beside the others and not counted with them &mdash; see the warning below.</li>
<li>Three rules only, no vertical lines, tabular figures. Standard journal table style.</li>
</ul>

<h2>The table</h2>
<p class="cap"><b>Table S12.</b> {H.escape(CAPTION)}</p>
<div class="scroll"><table class="s12"><thead>{head}</thead>
<tbody>
{"".join(body)}
</tbody></table></div>
<p class="fn">{H.escape(FOOTNOTES.format(n_sign=n_sign))}</p>

<div class="stop"><h3>A one-sided p near 1 is not "no effect"</h3>
<p>It means the list departs from its background in the direction <i>opposite</i> to the one tested. BloSSUM_H+ under Unhealthy has mean&nbsp;z <b>+1.44</b> at p&nbsp;&le;&nbsp;0.0005 and an IW-directional p of <b>1.000</b>, because its dominant raw direction is depletion while it is <i>less</i> depleted than equally prevalent background taxa. Same finding, opposite tail.</p>
<p>That is why <b>Agree</b> counts only the two-sided tests. Counting all four would have reported those cells as catastrophic method disagreements when the methods do not disagree at all, and it is the reason the <b>Dom. dir.</b> column is in the table rather than left in the CSV.</p></div>
</div>
"""
    open(out_path, "w", encoding="utf-8").write(page)
    print(f"wrote {out_path} {len(page)} bytes")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
