"""
Section 2 — extended pooling and detection analyses (2E–2G).

Fills the analytical gaps required by the manuscript text:

  2E. Detection-with-pooling for **all four** taxa lists
      (VANISH, VANISH+, BloSSUM, BloSSUM+) plus a reference curve
      for "a typical SGB" (median over all SGBs).
      → "requiring 20x more samples" / "5x for a typical SGB"

  2F. Continent x lifestyle group detection with full pooling.
      → "all VANISH taxa except ________ were detected in
         industrial populations"

  2G. Extinction candidates from VANISH+ / BloSSUM+ taxa.
      → Taxa in <= 3 opposite-context samples at the detection threshold.
      → Annotated with GTDB taxonomy when available.

Main-text outputs  (all in  section2_differential/main/)
-----------------------------------------------------------------
  - {vanish,vanish_plus}_pooling_industrial_{spec}.csv
  - {blossum,blossum_plus}_pooling_ni_{spec}.csv
  - all_sgb_pooling_{industrial,ni}_{spec}.csv
  - continent_lifestyle_detection_{spec}.csv
  - macro_pooling_detection_{spec}.csv
  - extinction_candidates_vanish_blossum_plus_{spec}.csv
  - Stats → manuscript_fill_ins.txt  via StatsReport
"""

import os
import numpy as np
import pandas as pd

from . import config
from .common import get_continent
from .detection import DetectionSpec, build_presence
from .plot_utils import (  # module level: the helpers below are used
    page_figsize,          # inside functions that do their own local
    panel_grid,            # plot_utils imports, which would otherwise
    share_axis_labels,     # shadow these names out of scope
)


# ════════════════════════════════════════════════════════════════════
#  POOLING HELPERS
# ════════════════════════════════════════════════════════════════════

def _n_needed_analytic(prevalence, n_total):
    """Smallest N where  P(detect) = 1 - (1-p)^N  >=  0.5."""
    if prevalence <= 0:
        return n_total
    if prevalence >= 1.0:
        return 1
    n = int(np.ceil(np.log(0.5) / np.log(1.0 - prevalence)))
    return min(n, n_total)


def _pooling_for_taxon_list(presence_df, taxon_ids, label=""):
    """Per-taxon prevalence and N-samples-needed-to-detect.

    Returns DataFrame [taxon, prevalence, n_detected, n_needed, n_total].
    """
    n_total = presence_df.shape[0]
    rows = []
    for taxon in taxon_ids:
        if taxon not in presence_df.columns:
            continue
        n_det = int(presence_df[taxon].sum())
        prev  = n_det / n_total if n_total > 0 else 0.0
        rows.append({
            "taxon":      taxon,
            "prevalence":  prev,
            "n_detected":  n_det,
            "n_needed":    _n_needed_analytic(prev, n_total),
            "n_total":     n_total,
        })
    df = pd.DataFrame(rows)
    if label and not df.empty:
        print(f"    {label}: {len(df)} taxa, "
              f"median prev = {df['prevalence'].median():.4f}, "
              f"median N = {df['n_needed'].median():.0f}")
    return df


# ════════════════════════════════════════════════════════════════════
#  2E.  POOLING FOR ALL LISTS + FOLD-CHANGE STATS
# ════════════════════════════════════════════════════════════════════

def compute_pooling_all_lists(data, vanish_plus_ids, blossum_plus_ids,
                              stats, spec=None):
    """
    Pooling for VANISH, VANISH+, BloSSUM, BloSSUM+, and all SGBs.

    Industrial context  → VANISH / VANISH+ / All SGBs
    Non-Industrial      → BloSSUM / BloSSUM+ / All SGBs
    """
    ref = data.ref_name
    s2  = config.section_dir(ref, "s2", "main")
    if spec is None:
        spec = DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)
    tag = spec.short_label

    meta     = data.meta
    breadth  = data.breadth
    presence = build_presence(breadth, data.depth, data.lengths, spec=spec)

    vanish   = data.vanish_set
    blossum  = data.blossum_set
    vp_set   = set(vanish_plus_ids)  if vanish_plus_ids  else set()
    bp_set   = set(blossum_plus_ids) if blossum_plus_ids else set()
    all_sgbs = sorted(breadth.columns)

    # ── Industrial ──
    ind_mask = meta["Lifestyle"] == "Industrial"
    ind_pres = presence.loc[ind_mask]
    n_ind    = int(ind_mask.sum())
    print(f"\n    === Pooling in Industrial (n={n_ind:,}) ===")

    pool_vanish  = _pooling_for_taxon_list(ind_pres,
        sorted(v for v in vanish  if v in breadth.columns), "VANISH in Industrial")
    pool_vp      = _pooling_for_taxon_list(ind_pres,
        sorted(v for v in vp_set  if v in breadth.columns), "VANISH+ in Industrial")
    pool_all_ind = _pooling_for_taxon_list(ind_pres, all_sgbs,
        "All SGBs in Industrial")

    pool_vanish.to_csv( os.path.join(s2, f"vanish_pooling_industrial_{tag}.csv"),      index=False)
    pool_vp.to_csv(     os.path.join(s2, f"vanish_plus_pooling_industrial_{tag}.csv"), index=False)
    pool_all_ind.to_csv(os.path.join(s2, f"all_sgb_pooling_industrial_{tag}.csv"),     index=False)

    # ── Non-Industrial ──
    ni_mask = meta["Lifestyle"].isin(["H-G", "Transitional"])
    ni_pres = presence.loc[ni_mask]
    n_ni    = int(ni_mask.sum())
    print(f"\n    === Pooling in Non-Industrial (n={n_ni:,}) ===")

    pool_blossum = _pooling_for_taxon_list(ni_pres,
        sorted(b for b in blossum if b in breadth.columns), "BloSSUM in Non-Industrial")
    pool_bp      = _pooling_for_taxon_list(ni_pres,
        sorted(b for b in bp_set  if b in breadth.columns), "BloSSUM+ in Non-Industrial")
    pool_all_ni  = _pooling_for_taxon_list(ni_pres, all_sgbs,
        "All SGBs in Non-Industrial")

    pool_blossum.to_csv(os.path.join(s2, f"blossum_pooling_ni_{tag}.csv"),      index=False)
    pool_bp.to_csv(     os.path.join(s2, f"blossum_plus_pooling_ni_{tag}.csv"), index=False)
    pool_all_ni.to_csv( os.path.join(s2, f"all_sgb_pooling_ni_{tag}.csv"),      index=False)

    # ── Fold-change stats ──
    _log_fold_stats(pool_vanish, pool_vp, pool_all_ind,
                    pool_blossum, pool_bp, pool_all_ni, stats, tag)

    # ── Min detection at the spec threshold ──
    _log_min_detection(data,
        sorted(v for v in vanish  if v in breadth.columns),
        sorted(b for b in blossum if b in breadth.columns),
        sorted(v for v in vp_set  if v in breadth.columns),
        sorted(b for b in bp_set  if b in breadth.columns),
        presence, stats, tag)

    return (pool_vanish, pool_vp, pool_all_ind,
            pool_blossum, pool_bp, pool_all_ni)


def _log_fold_stats(pv, pvp, pai, pb, pbp, pni, stats, tag):
    """Log fold-change in N-needed for manuscript fill-ins."""
    def _med(df):
        return df["n_needed"].median() if not df.empty else np.nan

    med_ai = _med(pai)
    med_ni = _med(pni)

    for label, pool, ref_med in [
        ("VANISH",   pv,  med_ai),
        ("VANISH+",  pvp, med_ai),
        ("BloSSUM",  pb,  med_ni),
        ("BloSSUM+", pbp, med_ni),
    ]:
        med  = _med(pool)
        fold = med / ref_med if ref_med and ref_med > 0 else np.nan
        ctx  = "Industrial" if "VANISH" in label else "Non-Industrial"
        safe = label.replace("+", "_plus")

        stats.add(f"{safe}_median_n_needed_{ctx}_{tag}", "s2", "main",
                  f"Median N {ctx} samples to detect a {label} taxon",
                  f"{med:.0f}",
                  extra={"fold_vs_typical": f"{fold:.1f}\u00d7",
                         "typical_sgb_median": f"{ref_med:.0f}"})

    stats.add(f"typical_sgb_n_needed_ind_{tag}", "s2", "main",
              "Median N Industrial samples for a typical SGB", f"{med_ai:.0f}")
    stats.add(f"typical_sgb_n_needed_ni_{tag}",  "s2", "main",
              "Median N Non-Industrial samples for a typical SGB", f"{med_ni:.0f}")

    print(f"\n    \u2500\u2500 Manuscript fill-in: samples-to-detect \u2500\u2500")
    print(f"    VANISH  in Ind:  N={_med(pv):.0f}  "
          f"({_med(pv)/med_ai:.1f}\u00d7 typical @ {med_ai:.0f})")
    print(f"    VANISH+ in Ind:  N={_med(pvp):.0f}  "
          f"({_med(pvp)/med_ai:.1f}\u00d7)")
    print(f"    BloSSUM  in NI:  N={_med(pb):.0f}  "
          f"({_med(pb)/med_ni:.1f}\u00d7 typical @ {med_ni:.0f})")
    print(f"    BloSSUM+ in NI:  N={_med(pbp):.0f}  "
          f"({_med(pbp)/med_ni:.1f}\u00d7)")


def _log_min_detection(data, vanish_ids, blossum_ids, vp_ids, bp_ids,
                       presence, stats, tag):
    """Min detection counts at the spec threshold for each list."""
    meta = data.meta
    ind_mask = meta["Lifestyle"] == "Industrial"
    ni_mask  = meta["Lifestyle"].isin(["H-G", "Transitional"])

    for name, ids, mask, ctx in [
        ("VANISH",   vanish_ids,  ind_mask, "Industrial"),
        ("VANISH+",  vp_ids,      ind_mask, "Industrial"),
        ("BloSSUM",  blossum_ids, ni_mask,  "Non-Industrial"),
        ("BloSSUM+", bp_ids,      ni_mask,  "Non-Industrial"),
    ]:
        valid = [t for t in ids if t in presence.columns]
        if not valid:
            continue
        n_det = presence.loc[mask, valid].sum(axis=0)
        safe  = name.replace("+", "_plus")

        stats.add(f"{safe}_min_det_{tag}_{ctx}", "s2", "main",
                  f"Min # {ctx} samples detecting any {name} taxon @ {tag}",
                  int(n_det.min()),
                  extra={"n_taxa": len(valid)})

        hard = n_det.nsmallest(5)
        stats.add(f"{safe}_hardest_{tag}_{ctx}", "s2", "main",
                  f"Hardest-to-find {name} taxa @ {tag} in {ctx}",
                  "; ".join(f"{t}={v}" for t, v in hard.items()))

        print(f"    {name} in {ctx}: min@{tag}={int(n_det.min())}, "
              f"n_taxa={len(valid)}")


# ════════════════════════════════════════════════════════════════════
#  2F.  CONTINENT x LIFESTYLE GROUP DETECTION
# ════════════════════════════════════════════════════════════════════

def compute_continent_lifestyle_detection(data, vanish_plus_ids,
                                          blossum_plus_ids, stats,
                                          spec=None, taxonomy_df=None):
    """Per continent x lifestyle detection and macro-level pooling report.

    Fills in: "all VANISH taxa except ________ were detected in
    industrial populations"
    """
    ref = data.ref_name
    s2  = config.section_dir(ref, "s2", "main")
    if spec is None:
        spec = DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)
    tag = spec.short_label

    meta, breadth = data.meta, data.breadth
    presence      = build_presence(breadth, data.depth, data.lengths, spec=spec)

    vanish  = data.vanish_set
    blossum = data.blossum_set
    vp_set  = set(vanish_plus_ids)  if vanish_plus_ids  else set()
    bp_set  = set(blossum_plus_ids) if blossum_plus_ids else set()

    lists = [("VANISH", vanish), ("VANISH+", vp_set),
             ("BloSSUM", blossum), ("BloSSUM+", bp_set)]

    # ── Per-group table ──
    meta_c = meta.copy()
    if "Continent" not in meta_c.columns:
        meta_c["Continent"] = meta_c["Country_norm"].apply(get_continent)
    groups = meta_c.groupby(["Continent", "Lifestyle"]).groups
    print(f"\n    Continent \u00d7 Lifestyle groups: {len(groups)}")

    rows = []
    for (cont, life), idx in groups.items():
        if pd.isna(cont) or pd.isna(life) or len(idx) < 5:
            continue
        for lname, lset in lists:
            tv = sorted(t for t in lset if t in breadth.columns)
            if not tv:
                continue
            det = sum(1 for t in tv if presence.loc[idx, t].any())
            rows.append({
                "continent": cont, "lifestyle": life,
                "n_samples": len(idx), "list": lname,
                "n_taxa": len(tv),
                "n_detected": det,
                "frac_detected": det / len(tv),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(s2, f"continent_lifestyle_detection_{tag}.csv"),
        index=False)

    # ── Macro pooling ──
    _report_macro_pooling(data, presence, lists, stats, tag, s2,
                          taxonomy_df=taxonomy_df)

    return pd.DataFrame(rows)


def _report_macro_pooling(data, presence, lists, stats, tag, s2,
                          taxonomy_df=None):
    """Pool all Industrial / NI / H-G samples; report missing taxa."""
    meta, breadth = data.meta, data.breadth
    ind_mask = meta["Lifestyle"] == "Industrial"
    ni_mask  = meta["Lifestyle"].isin(["H-G", "Transitional"])
    hg_mask  = meta["Lifestyle"] == "H-G"

    # Build taxonomy map for annotating missing taxa
    tax_map = _build_taxonomy_map(taxonomy_df)

    report_rows = []
    combos = [
        ("VANISH",   lists[0][1], ind_mask, "Industrial"),
        ("VANISH+",  lists[1][1], ind_mask, "Industrial"),
        ("BloSSUM",  lists[2][1], ni_mask,  "Non-Industrial"),
        ("BloSSUM+", lists[3][1], ni_mask,  "Non-Industrial"),
        ("BloSSUM",  lists[2][1], hg_mask,  "H-G"),
        ("BloSSUM+", lists[3][1], hg_mask,  "H-G"),
    ]

    for lname, lset, mask, ctx in combos:
        tv = sorted(t for t in lset if t in breadth.columns)
        if not tv:
            continue

        detected = [t for t in tv if presence.loc[mask, t].any()]
        missing  = sorted(set(tv) - set(detected))

        safe = lname.replace("+", "_plus").replace(" ", "_")
        sctx = ctx.replace("-", "")

        # Annotate missing taxa with taxonomy
        miss_annot = []
        for t in missing:
            tax = tax_map.get(t, "")
            lr  = _last_rank(tax)
            miss_annot.append(f"{t} ({lr})" if lr else t)

        stats.add(f"{safe}_pooled_{sctx}_{tag}", "s2", "main",
                  f"{lname} detected in pooled {ctx} @ {tag}",
                  f"{len(detected)}/{len(tv)}",
                  extra={"missing": ", ".join(miss_annot) or "(all detected)"})

        print(f"\n    {lname} in pooled {ctx}:  "
              f"{len(detected)}/{len(tv)} @ {tag}")
        if missing:
            for m in miss_annot:
                print(f"      Missing: {m}")

        for t in tv:
            report_rows.append({
                "list": lname, "context": ctx, "taxon": t,
                "n_detected": int(presence.loc[mask, t].sum()),
                "n_total_samples": int(mask.sum()),
                "taxonomy": tax_map.get(t, ""),
            })

    pd.DataFrame(report_rows).to_csv(
        os.path.join(s2, f"macro_pooling_detection_{tag}.csv"), index=False)


# ════════════════════════════════════════════════════════════════════
#  2G.  EXTINCTION CANDIDATES (VANISH+ / BloSSUM+)
# ════════════════════════════════════════════════════════════════════

def identify_extinction_candidates_plus(data, vanish_plus_ids,
                                        blossum_plus_ids, stats,
                                        spec=None, taxonomy_df=None):
    """VANISH+ / BloSSUM+ taxa in <= 3 opposite-context samples at the
    spec detection threshold.

    Annotates with GTDB taxonomy when taxonomy_df or phylo list files
    are available.
    """
    ref = data.ref_name
    s2  = config.section_dir(ref, "s2", "main")
    meta, breadth = data.meta, data.breadth

    if spec is None:
        spec = DetectionSpec("breadth_pct", config.ACCUM_MAIN_TEXT_BREADTH)
    tag = spec.short_label

    presence = build_presence(breadth, data.depth, data.lengths, spec=spec)

    ind_mask = meta["Lifestyle"] == "Industrial"
    ni_mask  = meta["Lifestyle"].isin(["H-G", "Transitional"])

    tax_map = _build_taxonomy_map(taxonomy_df)

    vp = set(vanish_plus_ids)  if vanish_plus_ids  else set()
    bp = set(blossum_plus_ids) if blossum_plus_ids else set()

    cands = []
    for label, taxa, mask, ctx in [
        ("VANISH+",  vp, ind_mask, "Industrial"),
        ("BloSSUM+", bp, ni_mask,  "Non-Industrial"),
    ]:
        print(f"\n    Extinction candidates \u2014 {label} in {ctx} @ {tag}:")
        for t in sorted(taxa):
            if t not in presence.columns:
                continue
            n_det = int(presence.loc[mask, t].sum())
            if n_det <= 3:
                tax = tax_map.get(t, "")
                cands.append({
                    "taxon": t, "direction": label,
                    "context_absent": ctx,
                    "n_detected": n_det,
                    "detection_spec": tag,
                    "n_total_samples": int(mask.sum()),
                    "taxonomy": tax,
                })
                print(f"      {t}: {n_det} samples @ {tag}  {_last_rank(tax)}")

    cdf = pd.DataFrame(cands)
    if not cdf.empty:
        cdf.to_csv(os.path.join(s2,
            f"extinction_candidates_vanish_blossum_plus_{tag}.csv"), index=False)
        for d in cdf["direction"].unique():
            sub = cdf[cdf["direction"] == d]
            safe = d.replace("+", "_plus")
            stats.add(f"n_extinction_{safe}_{tag}", "s2", "main",
                      f"{d} taxa in \u2264 3 opposite-context samples @ {tag}",
                      len(sub))
            stats.add(f"extinction_list_{safe}_{tag}", "s2", "main",
                      f"{d} extinction candidate taxa @ {tag}",
                      "; ".join(sub["taxon"]))
    else:
        print(f"    No extinction candidates @ {tag} (all taxa in \u2265 4 samples)")
        stats.add(f"n_extinction_candidates_{tag}", "s2", "main",
                  f"Total extinction candidates from VANISH+/BloSSUM+ @ {tag}", 0)
    return cdf


_COMBINED_TAX_CACHE = None


def _taxonomy_from_combined_metadata(path=None):
    """{genome_id: GTDB lineage} from combined_sgb_metadata.xlsx.

    The per-catalogue sheets carry `gtdb_lineage` keyed on `sgb`; the Combined
    sheet keys on `representative`. A representative that is itself unannotated
    can still be resolved through its cross-catalogue partner, since membership
    of a combined cluster means >=95% ANI and so the same GTDB species. That
    recovers the 16 `shared` representatives whose UHGG row has no lineage.

    Returns {} if the workbook is absent -- this is a fallback, not a
    requirement, and references other than `combined` do not need it.
    """
    global _COMBINED_TAX_CACHE
    if _COMBINED_TAX_CACHE is not None:
        return _COMBINED_TAX_CACHE

    path = path or getattr(config, "COMBINED_SGB_METADATA_FILE",
                           "combined_sgb_metadata.xlsx")
    tax = {}
    if not os.path.exists(path):
        _COMBINED_TAX_CACHE = tax
        return tax

    try:
        book = pd.ExcelFile(path)
    except Exception as exc:                       # unreadable workbook
        print(f"    Taxonomy: could not read {path} ({exc})")
        _COMBINED_TAX_CACHE = tax
        return tax

    def _clean(v):
        s = str(v).strip()
        return s if s and s.lower() != "nan" else ""

    # Per-catalogue sheets first: they are the primary annotation.
    for sheet in ("Leviatan", "UHGG"):
        if sheet not in book.sheet_names:
            continue
        df = book.parse(sheet)
        if "gtdb_lineage" not in df.columns:
            continue
        for _, row in df.iterrows():
            lineage = _clean(row.get("gtdb_lineage"))
            if not lineage:
                continue
            for col in ("sgb", "combined_id"):
                gid = _clean(row.get(col))
                if gid:
                    tax.setdefault(gid, lineage)

    if "Combined" in book.sheet_names:
        comb = book.parse("Combined")
        if "gtdb_lineage" in comb.columns:
            for _, row in comb.iterrows():
                lineage = _clean(row.get("gtdb_lineage"))
                if not lineage:
                    continue
                for col in ("representative", "combined_id"):
                    gid = _clean(row.get(col))
                    if gid:
                        tax.setdefault(gid, lineage)

            # Second pass: borrow a partner's lineage for representatives that
            # still have none.
            filled = 0
            for _, row in comb.iterrows():
                rep = _clean(row.get("representative"))
                if not rep or rep in tax:
                    continue
                members = []
                for col in ("leviatan_members", "uhgg_members"):
                    raw = _clean(row.get(col))
                    if raw:
                        members += [m.strip() for m in raw.replace(";", ",").split(",")
                                    if m.strip()]
                for member in members:
                    if member in tax:
                        tax[rep] = tax[member]
                        filled += 1
                        break
            if filled:
                print(f"    Taxonomy: {filled} representative(s) resolved via a "
                      f"cross-catalogue partner")

    print(f"    Taxonomy: {len(tax):,} entries from {os.path.basename(path)}")
    _COMBINED_TAX_CACHE = tax
    return tax


def _build_taxonomy_map(taxonomy_df):
    """Build {genome_id: GTDB_taxonomy_string} from available sources.

    Precedence, highest first:

      1. `combined_sgb_metadata.xlsx` — the catalogue-wide annotation, the
         only source covering both the Leviatan Rep_* SGBs and the UHGG ones;
         see `_taxonomy_from_combined_metadata`.
      2. `taxonomy_df`, the Hadza_UHGG_SGB_matching.csv crosswalk, where
         column 0 has long assembly names, columns 1-2 have GCA IDs
         (with/without suffix) and the last column has GTDB taxonomy. We
         index EVERY column holding GCA_*, MGYG* or Rep_* identifiers. It
         covers UHGG only, and on an older GTDB release.
      3. The VANISH/BloSSUM phylo files named in config.

    This is the same precedence `section3_health._load_taxonomy_map_s3` uses,
    so Tables S8/S9 and Figs. S26-S30 cannot name the same accession two
    different ways.
    """
    import re
    _ID_RE = re.compile(r'^(GCA_\d+\.\d+|MGYG\d+|Rep_\d+)$')
    tax = {}

    # The combined-catalogue workbook FIRST. It is the catalogue-wide
    # annotation, it covers Rep_* as well as the UHGG ids, and it is the one
    # `section3_health._load_taxonomy_map_s3` consults first -- which is the
    # whole point of the ordering. Until 2026-10-09 the crosswalk below was
    # read first and overwrote it, so the same accession could be printed two
    # ways in one submission: GCA_934700615.1 was `g__RC9` in Tables S8/S9 and
    # `Cryptobacteroides sp934700615` in Fig. S28, and fourteen others
    # disagreed the same way, the crosswalk carrying an older GTDB release.
    # Everything below now fills gaps only, so nothing loses a name.
    for gid, lineage in _taxonomy_from_combined_metadata().items():
        tax.setdefault(gid, lineage)
    _n_workbook = len(tax)

    # Then the explicit DataFrame (Hadza_UHGG_SGB_matching etc.), for the
    # accessions the workbook does not annotate.
    if taxonomy_df is not None and not taxonomy_df.empty:
        # Find taxonomy column
        tcol = next((c for c in taxonomy_df.columns
                     if any(k in c.lower() for k in ("gtdb", "tax", "lineage"))),
                    None)
        if tcol is None:
            # Last resort: rightmost column if it contains semicolons
            for c in reversed(list(taxonomy_df.columns)):
                vals = taxonomy_df[c].dropna().astype(str)
                if vals.str.contains(";").mean() > 0.5:
                    tcol = c
                    break

        if tcol is not None:
            # Index every column that contains GCA/MGYG-style IDs
            id_cols = []
            for c in taxonomy_df.columns:
                if c == tcol:
                    continue
                vals = taxonomy_df[c].dropna().astype(str)
                # Check if >=20% of values look like SGB IDs
                matches = vals.apply(lambda v: bool(_ID_RE.match(v.strip())))
                if matches.mean() >= 0.2:
                    id_cols.append(c)

            # If no columns auto-detected, fall back to column 0
            if not id_cols:
                id_cols = [taxonomy_df.columns[0]]

            for _, row in taxonomy_df.iterrows():
                t = str(row[tcol]).strip()
                if not t or t == "nan":
                    continue
                for ic in id_cols:
                    gid = str(row[ic]).strip()
                    if gid and gid != "nan" and _ID_RE.match(gid):
                        tax.setdefault(gid, t)

            if id_cols:
                print(f"    Taxonomy: using ID columns {id_cols} "
                      f"→ {len(tax) - _n_workbook} further entries from "
                      f"the crosswalk (workbook supplied {_n_workbook})")

    # Fall-back: phylo list files referenced in config
    for fpath in [config.VANISH_PHYLO_FILE, config.BLOSSUM_PHYLO_FILE]:
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath, sep=None, engine="python")
            tcols = [c for c in df.columns
                     if any(k in c.lower() for k in ("gtdb", "tax", "lineage"))]
            if df.shape[1] >= 2 and tcols:
                for _, row in df.iterrows():
                    gid = str(row.iloc[0]).strip()
                    tax.setdefault(gid, str(row[tcols[0]]).strip())
        except Exception:
            pass

    if tax:
        print(f"    Taxonomy lookup: {len(tax)} entries total")
    return tax


# ════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def run_section2_pooling(data, vanish_plus_ids, blossum_plus_ids,
                         stats, main_specs=None, taxonomy_df=None):
    """Run all Section 2 extended pooling analyses (2E–2G).

    Call *after* the core Section 2 differential analysis which
    produces vanish_plus_ids and blossum_plus_ids.
    """
    ref = data.ref_name
    print(f"\n{'='*60}")
    print(f"  Section 2 Extended: Pooling & Extinction [{ref}]")
    print(f"{'='*60}")

    if main_specs is None:
        main_specs = [DetectionSpec("breadth_pct",
                                     config.ACCUM_MAIN_TEXT_BREADTH)]

    for spec in main_specs:
        print(f"\n  \u2500\u2500 {spec.description} \u2500\u2500")

        print("\n  2E. Detection with pooling (all lists)")
        compute_pooling_all_lists(data, vanish_plus_ids,
                                  blossum_plus_ids, stats, spec=spec)

        print("\n  2F. Continent \u00d7 lifestyle group detection")
        compute_continent_lifestyle_detection(
            data, vanish_plus_ids, blossum_plus_ids, stats, spec=spec,
            taxonomy_df=taxonomy_df)

        print("\n  2G. Extinction candidates (VANISH+ / BloSSUM+)")
        cdf = identify_extinction_candidates_plus(
            data, vanish_plus_ids, blossum_plus_ids, stats,
            spec=spec, taxonomy_df=taxonomy_df)

    print(f"\n  Section 2 Extended complete [{ref}]")

    # ── Supplementary figures ──
    print(f"\n  Section 2: generating supplementary figures ...")
    plot_section2_figures(data, vanish_plus_ids, blossum_plus_ids,
                          main_specs,
                          config.section_dir(ref, "s2", "main"),
                          stats, taxonomy_df=taxonomy_df)

    return cdf


# ════════════════════════════════════════════════════════════════════
#  SECTION 2 SUPPLEMENTARY FIGURES
# ════════════════════════════════════════════════════════════════════

_LIST_COLORS = {
    "VANISH":   "#E69F00",   # OI orange
    "VANISH+":  "#D55E00",   # OI vermillion
    "BloSSUM":  "#56B4E9",   # OI sky
    "BloSSUM+": "#0072B2",   # OI blue
}
_ALL_SGB_COLOR = "#999999"


def _last_rank(tax_str):
    """Extract the last non-empty GTDB rank from a taxonomy string."""
    if not tax_str or str(tax_str) == "nan":
        return ""
    for p in reversed(str(tax_str).split(";")):
        p = p.strip()
        if p and not p.endswith("__"):
            return p
    return ""


class _HeatmapSpec:
    """Carries the one field the heatmap title reads off a DetectionSpec."""

    def __init__(self, description=""):
        self.description = description


def plot_continent_lifestyle_heatmap(tables, out_c, description="", FS=None):
    """Draw the continent x lifestyle detection heatmap from its table.

    The table is the plotted matrix: rows are continent-by-lifestyle
    groups, columns are taxon lists, values are the fraction of each list
    detected. Every cell annotation is that value formatted.
    """
    import matplotlib.pyplot as plt
    from .plot_utils import savefig_dual, apply_oi_style, current_fs

    if FS is None:
        # Take the type scale from the shared style, as the pipeline does;
        # a different base here would change the saved canvas width and so
        # the printed point size.
        apply_oi_style()
        FS = current_fs()

    pivot = tables["detection"]
    if "group" in pivot.columns:
        pivot = pivot.set_index("group")
    n_rows, n_cols = pivot.shape
    spec = _HeatmapSpec(description)

    # The cells carry one number each and the row labels carry the continent
    # and the lifestyle and the sample count, so the labels are worth more of
    # the page than the squares are. Sizing the axes from the label block
    # rather than the grid is what lets the type grow at a fixed page width.
    fig, ax = plt.subplots(
        figsize=page_figsize(height_in=max(2.4, n_rows * 0.22 + 1.0)))
    # Leave the left third to the row labels.
    # As the figure's own margins, not a bare subplots_adjust: savefig_dual
    # runs a shared margin pass that would otherwise overwrite these and clip
    # the row labels off the left edge.
    _MG = dict(left=0.30, right=0.91, top=0.95, bottom=0.12)
    fig.subplots_adjust(**_MG)
    fig._margin_overrides = _MG

    # Okabe-Ito, like the rest of the set: white through yellow and sky to
    # blue. Monotonic in lightness, so it survives greyscale, and it avoids
    # the red-green pairing the guidelines rule out.
    from .plot_utils import okabe_ito_sequential
    cmap = okabe_ito_sequential()
    im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=1,
                   aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns, fontsize=FS, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index, fontsize=FS - 1)

    # Annotate cells. The per-group sample counts are row labels on this
    # figure, so they are published alongside the matrix rather than looked
    # up from the analysis frame.
    _gs = tables.get("group_sizes")
    n_samp = (_gs.set_index("group")["n_samples"]
              if _gs is not None and not _gs.empty
              else pd.Series(dtype=float))
    for ri, grp in enumerate(pivot.index):
        for ci, lst in enumerate(pivot.columns):
            val = pivot.loc[grp, lst]
            if pd.isna(val):
                ax.text(ci, ri, "n/a", ha="center", va="center",
                        fontsize=FS - 2, color="grey")
            else:
                # The scale runs white to blue, so only the dark end needs
                # light type.
                txt_col = "white" if val > 0.72 else "black"
                ax.text(ci, ri, f"{val:.0%}", ha="center", va="center",
                        fontsize=FS - 1, color=txt_col, fontweight="bold")

    # Add n_samples annotation on y-axis
    # Abbreviate the lifestyle half of each row label. Spelled out, these run
    # wider than the third of the page reserved for them, and a tight bounding
    # box then grows the saved canvas -- which shrinks every glyph on the
    # figure when it is reduced to the column width. The caption defines them.
    # Both the published names and the internal keys, because the labels
    # arrive here as keys and the display pass would otherwise expand them
    # back to the long form after this runs.
    _short = {"Industrial": "I", "Agro-pastoralist": "A-P",
              "Hunter-gatherer": "H-G", "Non-Industrial": "NI",
              "Transitional": "A-P", "Hunter-Gatherer": "H-G",
              "Agro-Pastoralist": "A-P", "H-G": "H-G"}

    def _row_label(g):
        text = str(g)
        for long, short in _short.items():
            text = text.replace(long, short)
        if g in n_samp.index:
            return f"{text}  (n={int(n_samp[g]):,})"
        return text

    # A group with no published count loses the count, not gains a wrong one:
    # labelling every row "n=0" would read as a real measurement.
    y_labels = [_row_label(g) for g in pivot.index]
    for _lab in ax.set_yticklabels(y_labels, fontsize=FS - 1):
        # Already abbreviated on purpose; keep the display pass off them.
        _lab._keep_text = True

    # Separate Industrial from NI rows
    # The published matrix is already in row order, so its index is the order.
    n_ind_rows = sum(1 for g in pivot.index if "Industrial" in str(g))
    if 0 < n_ind_rows < n_rows:
        ax.axhline(n_ind_rows - 0.5, color="black", lw=1.5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Fraction of list detected", fontsize=FS - 1)
    # Title and method note belong to the caption, not the figure. This one
    # is an axes title rather than a suptitle, which is why the figure-level
    # strip did not catch it.
    plt.tight_layout()



    # Cropped to what is drawn. The fixed margins keep the row labels and
    # the colourbar inside the canvas, so the tight box can only shrink it --
    # it used to grow it to 9 in, before the lifestyle labels were
    # abbreviated -- and saving the whole canvas left a band of white below
    # the column labels and beside the colourbar.
    savefig_dual(fig, out_c, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"    Saved: {out_c}")
    return out_c


def plot_section2_figures(data, vanish_plus_ids, blossum_plus_ids,
                          main_specs, s2_main, stats,
                          taxonomy_df=None):
    """
    Generate four supplementary figures for Section 2.

    Fig S2A — Detection-burden dotplot
        N-samples-needed for 50% detection probability per taxon per list.
        Shows VANISH/VANISH+ vs all-SGB baseline (Industrial), and
        BloSSUM/BloSSUM+ vs all-SGB baseline (Non-Industrial).
        Key message: these taxa are not particularly hard to detect.

    Fig S2B — Prevalence scatter: Industrial vs Non-Industrial
        Each point = one SGB; colour = list membership.
        Diagonal = equal prevalence.  VANISH cluster below diagonal;
        BloSSUM cluster above.
        Key message: the two lists occupy distinct prevalence quadrants.

    Fig S2C — Continent × lifestyle detection heatmap
        Rows = continent × lifestyle; columns = VANISH/BloSSUM/+.
        Colour = fraction detected in fully pooled context samples.
        Annotated with n_samples.
        Key message: VANISH+ / BloSSUM+ are detected broadly.

    Fig S2D — Overlap bar chart + prevalence comparison
        How many VANISH overlap VANISH+?  BloSSUM ↔ BloSSUM+?
        Side-by-side bars + prevalence medians per subgroup.
        Key message: partial but meaningful overlap.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import matplotlib.cm as mcm
    from .plot_utils import (
        add_post_layout_hook,
        apply_oi_style,
        current_fs,
        page_figsize,
        panel_grid,
        square_figsize,
        savefig_dual,
        share_axis_labels,
    )

    apply_oi_style()
    FS = current_fs()

    # Use the first (main text) spec for the core figures
    spec = main_specs[0] if main_specs else DetectionSpec("breadth_pct", 10)
    tag  = spec.short_label

    meta     = data.meta
    breadth  = data.breadth
    presence = build_presence(breadth, data.depth, data.lengths, spec=spec)

    vanish_ids  = sorted(v for v in (data.vanish_set  or set()) if v in breadth.columns)
    blossum_ids = sorted(b for b in (data.blossum_set or set()) if b in breadth.columns)
    vp_ids      = sorted(v for v in (vanish_plus_ids  or set()) if v in breadth.columns)
    bp_ids      = sorted(b for b in (blossum_plus_ids or set()) if b in breadth.columns)
    all_sgbs    = sorted(breadth.columns)

    ind_mask = meta["Lifestyle"] == "Industrial"
    ni_mask  = meta["Lifestyle"].isin(["H-G", "Transitional"])
    hg_mask  = meta["Lifestyle"] == "H-G"

    # ── Build per-taxon prevalence tables ──────────────────────────────
    def _prev_table(mask, ids):
        n = int(mask.sum())
        rows = []
        for t in ids:
            if t not in presence.columns:
                continue
            n_det = int(presence.loc[mask, t].sum())
            prev  = n_det / n if n > 0 else 0.0
            rows.append({"taxon": t, "n_detected": n_det,
                         "prevalence": prev, "n_total": n,
                         "n_needed": _n_needed_analytic(prev, n)})
        return pd.DataFrame(rows)

    pv   = _prev_table(ind_mask, vanish_ids)
    pvp  = _prev_table(ind_mask, vp_ids)
    pai  = _prev_table(ind_mask, all_sgbs)
    pb   = _prev_table(ni_mask,  blossum_ids)
    pbp  = _prev_table(ni_mask,  bp_ids)
    pni  = _prev_table(ni_mask,  all_sgbs)

    # ────────────────────────────────────────────────────────────────────
    #  Fig S2A: Detection-burden dotplot
    # ────────────────────────────────────────────────────────────────────
    fig, (ax_ind, ax_ni) = plt.subplots(1, 2, figsize=page_figsize(n_cols=2))

    def _detection_panel(ax, pools, labels, colors, all_pool, context, n_ctx):
        """
        Horizontal strip plot: each dot = one taxon, x = n_needed (log).
        Reference median for 'all SGBs' shown as vertical band.
        """
        import matplotlib.ticker as ticker

        med_all = all_pool["n_needed"].median() if not all_pool.empty else np.nan
        if np.isfinite(med_all):
            ax.axvspan(med_all * 0.7, med_all * 1.43, color=_ALL_SGB_COLOR,
                       alpha=0.12, zorder=0)
            ax.axvline(med_all, color=_ALL_SGB_COLOR, lw=1.5, ls="--",
                       alpha=0.7, zorder=1,
                       label=f"Typical SGB median (N={med_all:.0f})")

        rng = np.random.default_rng(42)
        for yi, (pool, lbl, col) in enumerate(zip(pools, labels, colors)):
            if pool.empty:
                continue
            jitter = rng.uniform(-0.22, 0.22, len(pool))
            ax.scatter(pool["n_needed"].values, yi + jitter,
                       color=col, s=22, alpha=0.55, zorder=3,
                       edgecolors="none", rasterized=True)
            med = pool["n_needed"].median()
            ax.plot([med, med], [yi - 0.38, yi + 0.38],
                    color=col, lw=2.5, zorder=4)
            ax.text(med * 1.08, yi + 0.42, f"N={med:.0f}",
                    color=col, fontsize=FS - 2, va="bottom", fontweight="bold")

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=FS)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xlabel(f"N {context} samples for 50% detection probability",
                      fontsize=FS)
        ax.set_title(f"{context} populations (n={n_ctx:,})\n"
                     f"Grey band = typical SGB range  |  Tick = median",
                     fontsize=FS)
        ax.grid(axis="x", alpha=0.2, which="both")
        ax.legend(fontsize=FS - 2, loc="upper right")

    _detection_panel(ax_ind,
        [pv, pvp], ["VANISH\n(n=124)", "VANISH+\n(n=100)"],
        [_LIST_COLORS["VANISH"], _LIST_COLORS["VANISH+"]],
        pai, "Industrial", int(ind_mask.sum()))

    _detection_panel(ax_ni,
        [pb, pbp], ["BloSSUM\n(n=63)", "BloSSUM+\n(n=100)"],
        [_LIST_COLORS["BloSSUM"], _LIST_COLORS["BloSSUM+"]],
        pni, "Non-Industrial", int(ni_mask.sum()))

    fig.suptitle(
        f"Detection-burden: samples needed for 50% probability of detecting "
        f"each taxon ({spec.description})\n"
        "Each dot = one taxon  |  Vertical tick = median  |  "
        "Grey band = typical SGB (all SGBs) ±0.5 log unit around median",
        fontsize=FS + 1, fontweight="bold")
    plt.tight_layout()
    out_a = os.path.join(s2_main, f"s2a_detection_burden_{tag}.png")
    savefig_dual(fig, out_a, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_a}")

    # ────────────────────────────────────────────────────────────────────
    #  Fig S2B: Prevalence scatter Industrial vs Non-Industrial
    # ────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=square_figsize(margins=dict(top=0.90)))
    fig._margin_overrides = dict(top=0.90)

    # Per-taxon prevalence in both contexts
    def _prev_map(mask):
        n = int(mask.sum())
        return {t: int(presence.loc[mask, t].sum()) / n
                for t in breadth.columns if t in presence.columns}

    prev_ind = _prev_map(ind_mask)
    prev_ni  = _prev_map(ni_mask)

    # Classify each SGB
    vanish_set  = set(vanish_ids)
    blossum_set = set(blossum_ids)
    vp_set      = set(vp_ids)
    bp_set      = set(bp_ids)

    layer_defs = [
        ("Other SGBs",    "#dddddd", 10, 0.3,
         set(all_sgbs) - vanish_set - blossum_set - vp_set - bp_set),
        ("BloSSUM only",  _LIST_COLORS["BloSSUM"],  30, 0.7,
         blossum_set - vp_set - bp_set),
        ("BloSSUM+",      _LIST_COLORS["BloSSUM+"], 40, 0.85,
         (bp_set - blossum_set)),
        ("BloSSUM ∩ BloSSUM+", "#003F7F", 50, 0.9,
         blossum_set & bp_set),
        ("VANISH only",   _LIST_COLORS["VANISH"],   30, 0.7,
         vanish_set - vp_set - bp_set),
        ("VANISH+",       _LIST_COLORS["VANISH+"],  40, 0.85,
         (vp_set - vanish_set)),
        ("VANISH ∩ VANISH+", "#7B2200", 50, 0.9,
         vanish_set & vp_set),
    ]

    for label, col, ms, alpha, taxa in layer_defs:
        xs = [prev_ind.get(t, 0) * 100 for t in taxa if t in prev_ind]
        ys = [prev_ni.get(t, 0)  * 100 for t in taxa if t in prev_ni]
        if xs:
            ax.scatter(xs, ys, s=ms, color=col, alpha=alpha,
                       edgecolors="none" if ms < 35 else "white",
                       linewidths=0.4, zorder=3, label=f"{label} (n={len(xs)})",
                       rasterized=True)

    # Diagonal = equal prevalence
    ax.plot([0, 100], [0, 100], color="black", lw=0.8, ls="--", alpha=0.4,
            zorder=1, label="Equal prevalence")
    ax.set_xlabel("Prevalence in Industrial (%)", fontsize=FS)
    ax.set_ylabel("Prevalence in Non-Industrial (%)", fontsize=FS)
    ax.set_xlim(-1, 101); ax.set_ylim(-1, 101)
    ax.set_title(
        f"SGB prevalence: Industrial vs Non-Industrial ({spec.description})\n"
        "VANISH cluster below diagonal; BloSSUM cluster above",
        fontsize=FS + 1, fontweight="bold")
    ax.legend(fontsize=FS - 2, loc="upper left",
              framealpha=0.9, markerscale=1.2)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    out_b = os.path.join(s2_main, f"s2b_prevalence_scatter_{tag}.png")
    savefig_dual(fig, out_b, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_b}")

    # ────────────────────────────────────────────────────────────────────
    #  Fig S2C: Continent × lifestyle detection heatmap
    # ────────────────────────────────────────────────────────────────────
    # Load the precomputed CSV if it exists, else compute inline
    cl_csv = os.path.join(s2_main, f"continent_lifestyle_detection_{tag}.csv")
    if os.path.exists(cl_csv):
        cl_df = pd.read_csv(cl_csv)
    else:
        cl_df = compute_continent_lifestyle_detection(
            data, vanish_plus_ids, blossum_plus_ids, stats, spec=spec,
            taxonomy_df=taxonomy_df)

    if not cl_df.empty:
        # Pivot: rows = continent × lifestyle, columns = list
        cl_df["group"] = cl_df["continent"] + " / " + cl_df["lifestyle"]
        list_order = ["VANISH", "VANISH+", "BloSSUM", "BloSSUM+"]
        present_lists = [l for l in list_order if l in cl_df["list"].unique()]
        pivot = cl_df.pivot_table(index="group", columns="list",
                                   values="frac_detected", aggfunc="mean")
        pivot = pivot.reindex(columns=present_lists)

        # Sort rows: Industrial groups first, then H-G / Transitional
        def _row_sort(g):
            life = g.split(" / ")[-1].strip()
            return (0 if "Industrial" in life else
                    1 if "Transitional" in life else 2, g)
        row_order = sorted(pivot.index, key=_row_sort)
        pivot = pivot.loc[row_order]

        n_rows, n_cols = pivot.shape
        out_c = os.path.join(s2_main, f"s2c_continent_lifestyle_heatmap_{tag}.png")
        from . import figure_source
        _tables = {
            "detection": pivot.reset_index(),
            "group_sizes": (cl_df.groupby("group", as_index=False)["n_samples"]
                            .first()),
        }
        figure_source.emit_for_path(out_c, _tables)
        plot_continent_lifestyle_heatmap(_tables, out_c, spec.description, FS)

    # ────────────────────────────────────────────────────────────────────
    #  Fig S2D: VANISH ↔ VANISH+ and BloSSUM ↔ BloSSUM+ overlap
    #  Shows subgroup sizes (bars) and GMW-weighted prevalence (diamonds)
    #  with bootstrap 95% CIs.  A grey reference line marks the median
    #  GMW-weighted prevalence of all background SGBs in each context.
    # ────────────────────────────────────────────────────────────────────
    from .depth_bias_ipw import (_BinnedKDE, _geomean_weights, _ess)

    rng_s2d = np.random.default_rng(42)
    N_BOOT  = 500

    def _gmw_prev_stats(taxon_ids, presence_ctx, w_ctx, rng, n_boot):
        """
        Bootstrap GMW-weighted prevalence (median + 95% CI) for a set of taxa.

        GMW-weighted prevalence of taxon t =
            Σ_i w_i * presence[i,t]  /  Σ_i w_i

        Bootstrap: resample samples with replacement (weights preserved by
        resampling both the presence column and the weight vector together),
        compute per-resample weighted prevalence, take percentiles.

        Returns (median_pct, ci_lo_pct, ci_hi_pct).
        """
        valid = [t for t in taxon_ids if t in presence_ctx.columns]
        if not valid:
            return np.nan, np.nan, np.nan
        w = w_ctx / w_ctx.sum()
        n = len(w)
        # Observed weighted prevalence per taxon
        pmat = presence_ctx[valid].values.astype(float)   # (n_samples, n_taxa)
        obs_prev = (w @ pmat) * 100.0                     # (n_taxa,)
        med_obs  = float(np.median(obs_prev))
        # Bootstrap distribution of the MEDIAN across taxa
        boot_meds = np.empty(n_boot)
        for b in range(n_boot):
            idx  = rng.choice(n, size=n, replace=True)
            wb   = w[idx]; wb /= wb.sum()
            boot_meds[b] = float(np.median(wb @ pmat[idx]))  * 100.0
        ci_lo = float(np.percentile(boot_meds, 2.5))
        ci_hi = float(np.percentile(boot_meds, 97.5))
        return med_obs, ci_lo, ci_hi

    # Compute GMW weights once per context
    def _gmw_weights(mask):
        idx   = meta.index[mask]
        ld    = np.log10(np.maximum(data.total_bp.loc[idx].values, 1.0))
        other = meta.index[ind_mask if mask.equals(ni_mask) else ni_mask]
        ld_o  = np.log10(np.maximum(data.total_bp.loc[other].values, 1.0))
        kde_s = _BinnedKDE(ld); kde_o = _BinnedKDE(ld_o)
        return idx, ld, _geomean_weights(ld, kde_s, kde_o)

    try:
        ind_idx, ld_ind, w_ind = _gmw_weights(ind_mask)
        ni_idx,  ld_ni,  w_ni  = _gmw_weights(ni_mask)
        pres_ind = presence.loc[ind_idx]
        pres_ni  = presence.loc[ni_idx]
        gmw_ok   = True
    except Exception as _e:
        print(f"    [S2D] GMW weight computation failed: {_e}; falling back to raw")
        gmw_ok = False

    if gmw_ok:
        # Background median: all SGBs not in any list
        list_union = set(vanish_ids) | set(blossum_ids) | set(vp_ids) | set(bp_ids)
        bg_ids_ind = [t for t in all_sgbs if t not in list_union]
        bg_ids_ni  = bg_ids_ind   # same background set
        bg_med_ind, bg_lo_ind, bg_hi_ind = _gmw_prev_stats(
            bg_ids_ind, pres_ind, w_ind, rng_s2d, N_BOOT)
        bg_med_ni,  bg_lo_ni,  bg_hi_ni  = _gmw_prev_stats(
            bg_ids_ni,  pres_ni,  w_ni,  rng_s2d, N_BOOT)

    _MG4 = dict(top=0.80, wspace=0.34)
    fig, axes = plt.subplots(1, 2,
                             figsize=square_figsize(n_cols=2, margins=_MG4))
    fig._margin_overrides = _MG4

    panels = [
        (vanish_ids, vp_ids,   "VANISH",  "VANISH+",  "Industrial",
         ind_mask, _LIST_COLORS["VANISH"],  _LIST_COLORS["VANISH+"],
         pres_ind if gmw_ok else None, w_ind if gmw_ok else None,
         bg_med_ind if gmw_ok else np.nan,
         bg_lo_ind  if gmw_ok else np.nan,
         bg_hi_ind  if gmw_ok else np.nan),
        (blossum_ids, bp_ids, "BloSSUM", "BloSSUM+", "Non-Industrial",
         ni_mask, _LIST_COLORS["BloSSUM"], _LIST_COLORS["BloSSUM+"],
         pres_ni  if gmw_ok else None, w_ni  if gmw_ok else None,
         bg_med_ni  if gmw_ok else np.nan,
         bg_lo_ni   if gmw_ok else np.nan,
         bg_hi_ni   if gmw_ok else np.nan),
    ]

    for ax, (orig_ids, plus_ids, orig_lbl, plus_lbl, context, mask,
             orig_col, plus_col, pres_ctx, w_ctx,
             bg_med, bg_lo, bg_hi) in zip(axes, panels):

        orig_set  = set(orig_ids)
        plus_set  = set(plus_ids)
        overlap   = orig_set & plus_set
        orig_only = orig_set - plus_set
        plus_only = plus_set - orig_set

        subgroups = [
            (f"{orig_lbl} only", list(orig_only), orig_col),
            ("Overlap",          list(overlap),   "#555555"),
            (f"{plus_lbl} only", list(plus_only), plus_col),
        ]
        labels = [s[0] for s in subgroups]
        n_vals = [len(s[1]) for s in subgroups]
        colors = [s[2] for s in subgroups]
        x      = np.arange(len(subgroups))

        # GMW-weighted prevalence + bootstrap CIs per subgroup
        prev_med, prev_lo, prev_hi = [], [], []
        for _, ids, _ in subgroups:
            if gmw_ok and ids:
                med, lo, hi = _gmw_prev_stats(
                    ids, pres_ctx, w_ctx, rng_s2d, N_BOOT)
            else:
                # fallback: raw prevalence median
                n_ctx = int(mask.sum())
                prevs = [float(presence.loc[mask, t].sum()) / n_ctx * 100
                         for t in ids if t in presence.columns]
                med = float(np.median(prevs)) if prevs else np.nan
                lo = hi = np.nan
            prev_med.append(med); prev_lo.append(lo); prev_hi.append(hi)

        # ── Bars (n_taxa) ──
        ax.bar(x, n_vals, color=colors, alpha=0.72,
               edgecolor="white", lw=0.5, width=0.55)
        for i, (n, col) in enumerate(zip(n_vals, colors)):
            ax.text(i, n + 0.4, str(n), ha="center", va="bottom",
                    fontsize=FS, fontweight="bold", color=col)

        # ── Twinx: GMW-weighted prevalence ──
        ax2 = ax.twinx()

        # Background reference band
        if np.isfinite(bg_med):
            ax2.axhspan(bg_lo, bg_hi, color="#bbbbbb", alpha=0.25, zorder=0)
            ax2.axhline(bg_med, color="#888888", lw=1.5, ls="--",
                        zorder=1, label=f"Background (n={len(bg_ids_ind):,})")

        # Prevalence diamonds + CI bars per subgroup
        for i, (med, lo, hi, col) in enumerate(
                zip(prev_med, prev_lo, prev_hi, colors)):
            if not np.isfinite(med):
                continue
            yerr_lo = max(0, med - lo) if np.isfinite(lo) else 0
            yerr_hi = max(0, hi - med) if np.isfinite(hi) else 0
            ax2.errorbar(i, med,
                         yerr=[[yerr_lo], [yerr_hi]],
                         fmt="D", color=col, ecolor=col,
                         markersize=9, capsize=4, lw=1.5,
                         markeredgecolor="black", markeredgewidth=0.6,
                         zorder=5)

        # Connect medians with dotted line
        valid_i = [i for i, v in enumerate(prev_med) if np.isfinite(v)]
        if len(valid_i) >= 2:
            ax2.plot([x[i] for i in valid_i], [prev_med[i] for i in valid_i],
                     color="#333333", lw=1.2, ls=":", alpha=0.5, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FS - 1)
        ax.set_ylabel("Number of taxa", fontsize=FS)
        y_label = (f"GMW prevalence, {context} (%)"
                   if gmw_ok else f"Raw prevalence, {context} (%)")
        ax2.set_ylabel(y_label, fontsize=FS - 1, color="#333333")
        ax2.legend(fontsize=FS - 2, loc="upper left", framealpha=0.85)
        ax.set_title(f"{orig_lbl} \u2194 {plus_lbl}",
                     fontsize=FS + 1, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0, max(n_vals) * 1.28)

    weighting_note = "GMW depth-corrected" if gmw_ok else "raw"
    fig.suptitle(
        f"Original vs depth-matched lists ({spec.description})",
        fontsize=FS + 1, fontweight="bold")
    # The key belongs once, in small type under the title -- not repeated in
    # every panel heading at title size, which is what forced this figure to
    # 10.2 in and collided the labels.
    # The key line lives in the band the margin pass reserves for the title,
    # between the suptitle above it and the panel titles below. Measuring it
    # against the panels put it inside them, on their legends; a fixed place
    # inside a band that is itself reserved cannot land on anything.
    fig.text(0.5, 0.885,
             f"bar = taxa  \u00b7  diamond = {weighting_note} median "
             "prevalence \u00b1 95% CI  \u00b7  grey band = background SGBs",
             ha="center", va="center", fontsize=FS - 2, color="#555555")
    out_d = os.path.join(s2_main, f"s2d_list_overlap_{tag}.png")
    savefig_dual(fig, out_d, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_d}")
