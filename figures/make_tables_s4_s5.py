#!/usr/bin/env python3
"""
make_tables_s4_s5.py
--------------------
Standalone script that reads pipeline output files and writes:

    Table S4  — table_s4_list_membership.csv
    Table S5  — table_s5_overlap_summary.csv

No pipeline imports required.  All inputs are flat CSV / TXT files
that Section 2 writes during a normal pipeline run.

Input files read
----------------
From the results directory (results/{ref}/):
    prevalence_gmw_ind_all_vs_ni.csv       columns: microbe, prev_g1, prev_g2,
                                                     prevalence_diff, ...
    prevalence_gmw_ind_healthy_vs_ni.csv   same schema (prev_g1 = Ind-healthy)
    abundance_ind_vs_nonind.csv            columns: microbe, clr_diff, ...

    (If --bayesian was used, prefix the above with "bayes_")

From the section2 main dir (results/{ref}/section2_differential/main/):
    vanish_plus_ids.txt    }
    blossum_plus_ids.txt   }  one SGB ID per line
    vanishH_ids.txt        }
    blossumH_ids.txt       }

    (Again prefix with "bayes_" if using Bayesian lists)

From the working directory (or --data-dir):
    Vanish_list.txt        original VANISH IDs (one per line)
    Blossum_list.txt       original BloSSUM IDs (one per line)

Taxonomy (optional):
    Hadza_UHGG_SGB_matching.csv  or any CSV with an ID column and a
                                  column whose name contains 'gtdb', 'tax',
                                  or 'lineage'.

Usage
-----
    # Standard (hard-threshold) lists
    python make_tables_s4_s5.py --ref UHGG

    # Bayesian lists
    python make_tables_s4_s5.py --ref UHGG --bayesian

    # Explicit paths for everything
    python make_tables_s4_s5.py \\
        --ref UHGG \\
        --results-dir results \\
        --data-dir /path/to/data \\
        --taxonomy-file Hadza_UHGG_SGB_matching.csv \\
        --out-dir results/UHGG/section2_differential/supplement
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
#  FILE LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_id_set(path: str) -> set:
    """Load a one-ID-per-line text file. Returns an empty set if missing."""
    if not path or not os.path.exists(path):
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s:
                ids.add(s)
    return ids


def _load_gmw_csv(path: str) -> pd.DataFrame:
    """
    Load a prevalence_gmw_*.csv written by run_differential_gmw.

    Expected columns (from differential.test_prevalence_weighted):
        microbe, prev_g1, prev_g2, prevalence_diff, chi2, pval,
        log10_pval, neg_log10_p_bonf

    Returns an empty DataFrame if the file is missing.
    """
    if not path or not os.path.exists(path):
        print(f"  [warn] not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalise column name for microbe identity
    for alias in ("microbe", "sgb", "sgb_id", "SGB"):
        if alias in df.columns:
            df = df.rename(columns={alias: "microbe"})
            break
    return df


def _load_abundance_csv(path: str) -> pd.DataFrame:
    """
    Load abundance_ind_vs_nonind.csv written by run_differential.

    Expected columns: microbe, clr_diff, ...
    """
    if not path or not os.path.exists(path):
        print(f"  [warn] not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    for alias in ("microbe", "sgb", "sgb_id", "SGB"):
        if alias in df.columns:
            df = df.rename(columns={alias: "microbe"})
            break
    return df


import re as _re
_ID_RE = _re.compile(r'^(GCA_\d+\.\d+|MGYG\d+)$')


# Maximum file size to attempt parsing as a taxonomy source (bytes).
# Files larger than this are almost certainly data matrices, not taxonomy tables.
_MAX_TAX_FILE_BYTES = 5 * 1024 * 1024   # 5 MB
# Rows to read when sniffing — enough to detect column content, fast to load.
_TAX_SNIFF_ROWS = 10


def _load_taxonomy_file(path: str) -> dict:
    """
    Load one taxonomy CSV/TSV into a {sgb_id: taxonomy_string} dict.

    Handles two formats:
      1. Standard: ID column + taxonomy column with 'gtdb'/'tax'/'lineage' in name.
      2. Hadza_UHGG_SGB_matching format: multiple GCA/MGYG ID columns + rightmost
         column containing semicolon-separated lineages.

    Returns an empty dict if the file is missing, too large, or has no
    taxonomy column.
    """
    if not path or not os.path.exists(path):
        return {}

    # Skip files that are too large to be taxonomy tables.
    try:
        if os.path.getsize(path) > _MAX_TAX_FILE_BYTES:
            return {}
    except OSError:
        return {}

    # Sniff the first few rows with a fast engine to check column structure
    # before committing to a full parse.
    try:
        head = pd.read_csv(path, nrows=_TAX_SNIFF_ROWS,
                           sep=None, engine="python")
    except Exception:
        return {}

    if head.empty or head.shape[1] < 2:
        return {}

    # Quick reject: none of the column names or first-row values look like
    # taxonomy or SGB IDs — skip expensive full load.
    col_text = " ".join(str(c).lower() for c in head.columns)
    has_tax_col  = any(k in col_text for k in ("gtdb", "tax", "lineage"))
    has_semicols = any(
        head[c].astype(str).str.contains(";").mean() > 0.3
        for c in head.columns
    )
    has_id_col = head.apply(
        lambda col: col.astype(str).apply(
            lambda v: bool(_ID_RE.match(v.strip()))
        ).mean() >= 0.15
    ).any()

    if not (has_tax_col or has_semicols) or not has_id_col:
        return {}

    # Full parse — we are now confident this is a taxonomy-like file.
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception as e:
        print(f"  [warn] could not read {os.path.basename(path)}: {e}")
        return {}

    if df.empty or df.shape[1] < 2:
        return {}

    # Find taxonomy column: prefer explicit keyword match, then rightmost semicolon col
    tax_col = next(
        (c for c in df.columns
         if any(k in c.lower() for k in ("gtdb", "tax", "lineage"))),
        None,
    )
    if tax_col is None:
        for c in reversed(list(df.columns)):
            vals = df[c].dropna().astype(str)
            if len(vals) > 0 and vals.str.contains(";").mean() > 0.3:
                tax_col = c
                break

    if tax_col is None:
        return {}

    # Find ID columns: any column whose values look like GCA/MGYG accessions
    id_cols = []
    for c in df.columns:
        if c == tax_col:
            continue
        vals = df[c].dropna().astype(str)
        if len(vals) > 0 and vals.apply(lambda v: bool(_ID_RE.match(v.strip()))).mean() >= 0.15:
            id_cols.append(c)
    if not id_cols:
        # Last resort: named ID-like columns
        id_cols = [c for c in df.columns if c != tax_col and
                   any(k in c.lower() for k in ("sgb", "genome", "id", "accession", "unnamed"))]
    if not id_cols:
        id_cols = [df.columns[0]]

    tax_map = {}
    for _, row in df.iterrows():
        t = str(row[tax_col]).strip()
        if not t or t == "nan":
            continue
        for ic in id_cols:
            gid = str(row[ic]).strip()
            if gid and gid != "nan":
                tax_map.setdefault(gid, t)

    return tax_map


def _load_taxonomy(path: str) -> dict:
    """Thin wrapper kept for backward compatibility."""
    tax = _load_taxonomy_file(path)
    if tax:
        print(f"  Taxonomy: {len(tax)} entries from {os.path.basename(path)}")
    return tax


def _discover_taxonomy(data_dir: str, results_dir: str = None, ref: str = "") -> dict:
    """
    Search a range of candidate paths for taxonomy information and merge
    all sources together.  Later sources do not overwrite earlier ones
    (first-found wins per SGB ID).

    Search order:
      1. Common named files in data_dir
      2. Phylo list files (Vanish_phylo.txt, Blossom_phylo.txt) — these are
         standard pipeline files that _build_taxonomy_map also reads
      3. Any CSV/TSV in data_dir whose name contains 'tax', 'gtdb',
         'phylo', 'match', or 'uggh' / 'uhgg'
      4. results_dir scan (cached outputs from previous pipeline runs)
    """
    tax_map: dict = {}

    def _merge(new: dict):
        for k, v in new.items():
            tax_map.setdefault(k, v)

    # 0: the combined-catalogue workbook, before anything else -- the same
    # precedence section2_pooling._build_taxonomy_map and
    # section3_health._load_taxonomy_map_s3 settled on. This function is a
    # third, independent implementation that the 2026-10-09 precedence fix did
    # not reach: it starts from Hadza_UHGG_SGB_matching.csv, an older GTDB
    # release, and because _merge is first-found-wins those names stuck. The
    # result was 51 accessions named two ways in one submission -- Tables
    # S8/S9 printing g__Ruminococcus_A, g__RC9 and g__CAG-83 where Figs.
    # S26-S30 printed Oliverpabstia, Cryptobacteroides and Vescimonas for the
    # same genome, and every Rep_* carrying the placeholder g__Rep_NNNN rather
    # than a genus name at all.
    # The repo root goes on the path first: this file advertises itself as
    # needing no pipeline imports, so it is run by path and sys.path[0] is
    # `figures/`. Without this the import below fails rather than the workbook
    # being missing, and the except branch hides it -- the fix then looks
    # applied and changes nothing.
    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from persistence.section2_pooling import (
            _taxonomy_from_combined_metadata)
        _merge({k: v for k, v in _taxonomy_from_combined_metadata().items()
                if v})
    except Exception as _exc:              # workbook absent or unreadable
        print(f"  [warn] combined-catalogue taxonomy unavailable ({_exc}); "
              f"falling back to the crosswalk, which names some accessions "
              f"on an older GTDB release than the figures do")

    # 1 & 2: explicit candidates
    explicit = [
        os.path.join(data_dir, "Hadza_UHGG_SGB_matching.csv"),
        os.path.join(data_dir, "Hadza_UHGG_SGB_matching.tsv"),
        os.path.join(data_dir, "taxonomy.csv"),
        os.path.join(data_dir, "taxonomy.tsv"),
        os.path.join(data_dir, "Vanish_phylo.txt"),
        os.path.join(data_dir, "Blossom_phylo.txt"),
        os.path.join(data_dir, "Vanish_phylo.csv"),
        os.path.join(data_dir, "Blossum_phylo.txt"),
        "Hadza_UHGG_SGB_matching.csv",
        "Vanish_phylo.txt",
        "Blossom_phylo.txt",
        "Blossum_phylo.txt",
    ]
    for p in explicit:
        if p and os.path.exists(p) and p not in (None, ""):
            _merge(_load_taxonomy_file(p))

    # 3: glob data_dir for taxonomy-looking CSVs
    _TAX_KEYWORDS = ("tax", "gtdb", "phylo", "match", "uhgg", "uggh", "lineage", "sgb")
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            fl = fname.lower()
            if any(k in fl for k in _TAX_KEYWORDS) and (fl.endswith(".csv") or fl.endswith(".tsv") or fl.endswith(".txt")):
                fp = os.path.join(data_dir, fname)
                if fp not in explicit:
                    _merge(_load_taxonomy_file(fp))

    # 4: results directory scan
    if results_dir and ref:
        for subdir in [
            os.path.join(results_dir, ref),
            os.path.join(results_dir, ref, "section2_differential", "main"),
            os.path.join(results_dir, ref, "section2_differential", "supplement"),
        ]:
            if not os.path.isdir(subdir):
                continue
            for fname in os.listdir(subdir):
                fl = fname.lower()
                if any(k in fl for k in _TAX_KEYWORDS) and (fl.endswith(".csv") or fl.endswith(".tsv")):
                    _merge(_load_taxonomy_file(os.path.join(subdir, fname)))

    if tax_map:
        print(f"  Taxonomy: {len(tax_map)} entries total (merged from all sources)")
    else:
        print("  Taxonomy: none found — taxonomy columns will be empty")
        print("  Hint: place Vanish_phylo.txt / Blossom_phylo.txt / "
              "Hadza_UHGG_SGB_matching.csv in the working directory, or")
        print("        pass --taxonomy-file explicitly.")
    return tax_map


def _last_rank(tax_str: str) -> str:
    """Extract the lowest non-empty GTDB rank from a taxonomy string."""
    if not tax_str or tax_str == "nan":
        return ""
    for part in reversed(tax_str.split(";")):
        part = part.strip()
        if part and not part.endswith("__"):
            return part
    return ""


# ══════════════════════════════════════════════════════════════════════════════
#  TABLE S4
# ══════════════════════════════════════════════════════════════════════════════

def build_table_s4(
    vanish_set:       set,
    blossum_set:      set,
    vanish_plus_ids:  set,
    blossum_plus_ids: set,
    vanish_h_ids:     set,
    blossum_h_ids:    set,
    gmw_all:          pd.DataFrame,
    gmw_healthy:      pd.DataFrame,
    abund_df:         pd.DataFrame,
    tax_map:          dict,
) -> pd.DataFrame:
    """
    Build Table S4: one row per SGB in any of the six lists.

    Columns
    -------
    sgb_id
    taxonomy              full GTDB lineage string
    lowest_rank           lowest non-empty rank (convenience column)
    original_VANISH       0/1
    original_BloSSUM      0/1
    VANISH_plus           0/1
    BloSSUM_plus          0/1
    VanishH_plus          0/1
    BlossumH_plus         0/1
    prev_I_gmw_all        GMW-weighted prevalence in Industrial (all), contrast 2
    prev_NI_gmw_all       GMW-weighted prevalence in Non-Industrial, contrast 2
    prev_IndH_gmw_healthy GMW-weighted prevalence in Healthy Industrial, contrast 3
    delta_p_all_vs_NI     prev_I_gmw_all − prev_NI_gmw_all  (Δp contrast 2)
    delta_p_healthy_vs_NI prev_IndH_gmw_healthy − prev_NI  (Δp contrast 3)
    clr_diff              CLR abundance difference (Industrial − NI, depth-matched)
    neg_log10_p_bonf_all  Bonferroni-corrected significance for contrast 2
    """

    # Union of all SGBs appearing in any list
    all_sgbs = (
        vanish_set | blossum_set |
        vanish_plus_ids | blossum_plus_ids |
        vanish_h_ids | blossum_h_ids
    )
    if not all_sgbs:
        print("  [warn] all lists are empty — Table S4 will be empty")
        return pd.DataFrame()

    # Build lookup dicts from the GMW and abundance CSVs
    def _col(df, col):
        if df.empty or col not in df.columns:
            return {}
        return dict(zip(df["microbe"], df[col]))

    prev_I_all   = _col(gmw_all,     "prev_g1")
    prev_NI_all  = _col(gmw_all,     "prev_g2")
    delta_all    = _col(gmw_all,     "prevalence_diff")
    p_bonf_all   = _col(gmw_all,     "neg_log10_p_bonf")
    prev_I_hlt   = _col(gmw_healthy, "prev_g1")
    prev_NI_hlt  = _col(gmw_healthy, "prev_g2")   # same NI group, different Industrial
    delta_hlt    = _col(gmw_healthy, "prevalence_diff")
    clr_d        = _col(abund_df,    "clr_diff")

    def _f(d, k):
        v = d.get(k)
        return round(float(v), 4) if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan

    rows = []
    for sgb in sorted(all_sgbs, key=str):
        tax = tax_map.get(sgb, "")
        rows.append({
            "sgb_id":               sgb,
            "taxonomy":             tax,
            "lowest_rank":          _last_rank(tax),
            # List membership (0 / 1)
            "original_VANISH":      int(sgb in vanish_set),
            "original_BloSSUM":     int(sgb in blossum_set),
            "VANISH_plus":          int(sgb in vanish_plus_ids),
            "BloSSUM_plus":         int(sgb in blossum_plus_ids),
            "VanishH_plus":         int(sgb in vanish_h_ids),
            "BlossumH_plus":        int(sgb in blossum_h_ids),
            # Prevalences
            "prev_I_gmw_all":           _f(prev_I_all,  sgb),
            "prev_NI_gmw_all":          _f(prev_NI_all, sgb),
            "prev_IndH_gmw_healthy":    _f(prev_I_hlt,  sgb),
            # Δp for both contrasts (positive = enriched in Industrial)
            "delta_p_all_vs_NI":        _f(delta_all,   sgb),
            "delta_p_healthy_vs_NI":    _f(delta_hlt,   sgb),
            # CLR abundance
            "clr_diff_abundance":       _f(clr_d,       sgb),
            # Significance
            "neg_log10_p_bonf_all":     _f(p_bonf_all,  sgb),
        })

    df = pd.DataFrame(rows)

    # Sort: VANISH family first (by flag sum descending), then BloSSUM, then sgb_id
    df["_v"] = (df["original_VANISH"] * 4 + df["VANISH_plus"] * 2 + df["VanishH_plus"])
    df["_b"] = (df["original_BloSSUM"] * 4 + df["BloSSUM_plus"] * 2 + df["BlossumH_plus"])
    df["_fam"] = np.where(df["_v"] >= df["_b"], 0, 1)
    df = (df.sort_values(["_fam", "_v", "_b", "sgb_id"],
                         ascending=[True, False, False, True])
            .drop(columns=["_v", "_b", "_fam"])
            .reset_index(drop=True))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  TABLE S5
# ══════════════════════════════════════════════════════════════════════════════

def _overlap_row(family, name_a, ids_a, name_b, ids_b) -> dict:
    a, b  = set(ids_a), set(ids_b)
    inter = a & b
    union = a | b
    # recall_A_in_B: what fraction of A appears in B  (A is the "reference" list)
    # recall_B_in_A: what fraction of B appears in A
    # These are more interpretable than Jaccard when the lists have very different
    # sizes (e.g. VANISH has 124 taxa; VANISH+ is always exactly 100 by construction).
    # Jaccard is retained for completeness but should be read alongside the recalls.
    recall_a = round(len(inter) / len(a), 4) if a else 0.0
    recall_b = round(len(inter) / len(b), 4) if b else 0.0
    return {
        "family":         family,
        "list_A":         name_a,
        "list_B":         name_b,
        "n_A":            len(a),
        "n_B":            len(b),
        "overlap":        len(inter),
        "A_only":         len(a - b),
        "B_only":         len(b - a),
        "recall_A_in_B":  recall_a,   # |A∩B| / |A|  — fraction of A recovered in B
        "recall_B_in_A":  recall_b,   # |A∩B| / |B|  — fraction of B that was in A
        "Jaccard":        round(len(inter) / len(union), 4) if union else 0.0,
    }


def build_table_s5(
    vanish_set:       set,
    blossum_set:      set,
    vanish_plus_ids:  set,
    blossum_plus_ids: set,
    vanish_h_ids:     set,
    blossum_h_ids:    set,
) -> pd.DataFrame:
    """
    Build Table S5: pairwise overlap summary (6 rows — 3 per family).

    Rows
    ----
    VANISH family:
      original VANISH  vs  VANISH+
      VANISH+          vs  VanishH+
      original VANISH  vs  VanishH+

    BloSSUM family:
      original BloSSUM  vs  BloSSUM+
      BloSSUM+          vs  BlossumH+
      original BloSSUM  vs  BlossumH+
    """
    rows = [
        _overlap_row("VANISH",  "original VANISH",  vanish_set,       "VANISH+",    vanish_plus_ids),
        _overlap_row("VANISH",  "VANISH+",           vanish_plus_ids,  "VanishH+",   vanish_h_ids),
        _overlap_row("VANISH",  "original VANISH",   vanish_set,       "VanishH+",   vanish_h_ids),
        _overlap_row("BloSSUM", "original BloSSUM",  blossum_set,      "BloSSUM+",   blossum_plus_ids),
        _overlap_row("BloSSUM", "BloSSUM+",          blossum_plus_ids, "BlossumH+",  blossum_h_ids),
        _overlap_row("BloSSUM", "original BloSSUM",  blossum_set,      "BlossumH+",  blossum_h_ids),
    ]
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def _print_s5(df: pd.DataFrame):
    """Pretty-print Table S5 to stdout."""
    sep = "─" * 80
    print(f"\n{sep}")
    print("  Table S5 — List family overlap summary")
    print(sep)
    a_only_hdr = "A\\B"
    b_only_hdr = "B\\A"
    hdr = (f"  {'Family':<8}  {'List A':<22}  {'List B':<16}"
           f"  {'nA':>5}  {'nB':>5}  {'∩':>5}  {a_only_hdr:>5}  {b_only_hdr:>5}"
           f"  {'RecA':>6}  {'RecB':>6}  {'Jac':>6}")
    print(hdr)
    print(f"  {'─'*90}")
    print(f"  (RecA = |A∩B|/|A|, RecB = |A∩B|/|B|, Jac = |A∩B|/|A∪B|)")
    print(f"  {'─'*90}")
    prev_fam = None
    for _, r in df.iterrows():
        if prev_fam and r["family"] != prev_fam:
            print()
        prev_fam = r["family"]
        print(f"  {r['family']:<8}  {r['list_A']:<22}  {r['list_B']:<16}"
              f"  {r['n_A']:>5}  {r['n_B']:>5}  {r['overlap']:>5}"
              f"  {r['A_only']:>5}  {r['B_only']:>5}"
              f"  {r['recall_A_in_B']:>6.3f}  {r['recall_B_in_A']:>6.3f}  {r['Jaccard']:>6.3f}")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ref", default="UHGG",
                   help="Reference name (default: UHGG). Used to locate results.")
    p.add_argument("--results-dir", default="results",
                   help="Top-level results directory (default: results).")
    p.add_argument("--data-dir", default=".",
                   help="Directory containing Vanish_list.txt / Blossum_list.txt "
                        "(default: current directory).")
    p.add_argument("--taxonomy-file", default=None,
                   help="Optional CSV/TSV with SGB IDs and GTDB taxonomy. "
                        "Searched automatically if not supplied.")
    p.add_argument("--out-dir", default=None,
                   help="Output directory for the two CSVs. "
                        "Defaults to {results-dir}/{ref}/section2_differential/supplement/.")
    p.add_argument("--bayesian", action="store_true", default=False,
                   help="Read 'bayes_'-prefixed list files and GMW CSVs "
                        "produced by the Bayesian pipeline run.")
    return p.parse_args()


def main():
    args  = parse_args()
    ref   = args.ref
    pfx   = "bayes_" if args.bayesian else ""

    # ── Directory layout ───────────────────────────────────────────────────
    ref_dir  = os.path.join(args.results_dir, ref)
    s2_main  = os.path.join(ref_dir, "section2_differential", "main")
    out_dir  = args.out_dir or os.path.join(ref_dir, "section2_differential", "supplement")
    os.makedirs(out_dir, exist_ok=True)

    mode = "Bayesian" if args.bayesian else "standard"
    print(f"\n{'='*60}")
    print(f"  Tables S4 / S5  [{ref}]  [{mode} detection]")
    print(f"{'='*60}")

    # ── Load original VANISH / BloSSUM ─────────────────────────────────────
    vanish_file  = os.path.join(args.data_dir, "Vanish_list.txt")
    blossum_file = os.path.join(args.data_dir, "Blossum_list.txt")
    vanish_set   = _load_id_set(vanish_file)
    blossum_set  = _load_id_set(blossum_file)
    print(f"\n  Original lists:  VANISH={len(vanish_set)}  BloSSUM={len(blossum_set)}")
    if not vanish_set:
        print(f"  [warn] Vanish_list.txt not found at {vanish_file}")
    if not blossum_set:
        print(f"  [warn] Blossum_list.txt not found at {blossum_file}")

    # ── Load derived list IDs ──────────────────────────────────────────────
    def _s2(fname):
        return os.path.join(s2_main, fname)

    vanish_plus_ids  = _load_id_set(_s2(f"{pfx}vanish_plus_ids.txt"))
    blossum_plus_ids = _load_id_set(_s2(f"{pfx}blossum_plus_ids.txt"))
    vanish_h_ids     = _load_id_set(_s2(f"{pfx}vanishH_ids.txt"))
    blossum_h_ids    = _load_id_set(_s2(f"{pfx}blossumH_ids.txt"))

    print(f"  Derived lists:   VANISH+={len(vanish_plus_ids)}  "
          f"BloSSUM+={len(blossum_plus_ids)}  "
          f"VanishH+={len(vanish_h_ids)}  "
          f"BlossumH+={len(blossum_h_ids)}")

    for name, ids, path in [
        ("VANISH+",    vanish_plus_ids,  _s2(f"{pfx}vanish_plus_ids.txt")),
        ("BloSSUM+",   blossum_plus_ids, _s2(f"{pfx}blossum_plus_ids.txt")),
        ("VanishH+",   vanish_h_ids,     _s2(f"{pfx}vanishH_ids.txt")),
        ("BlossumH+",  blossum_h_ids,    _s2(f"{pfx}blossumH_ids.txt")),
    ]:
        if not ids:
            print(f"  [warn] {name} list empty — file not found: {path}")

    # ── Load GMW prevalence CSVs ───────────────────────────────────────────
    print(f"\n  Loading GMW prevalence tables ...")
    gmw_all     = _load_gmw_csv(os.path.join(ref_dir, f"{pfx}prevalence_gmw_ind_all_vs_ni.csv"))
    gmw_healthy = _load_gmw_csv(os.path.join(ref_dir, f"{pfx}prevalence_gmw_ind_healthy_vs_ni.csv"))
    abund_df    = _load_abundance_csv(os.path.join(ref_dir, f"{pfx}abundance_ind_vs_nonind.csv"))

    if not gmw_all.empty:
        print(f"    GMW all:     {len(gmw_all)} SGBs")
    if not gmw_healthy.empty:
        print(f"    GMW healthy: {len(gmw_healthy)} SGBs")
    if not abund_df.empty:
        print(f"    Abundance:   {len(abund_df)} SGBs")

    # ── Load taxonomy ──────────────────────────────────────────────────────
    if args.taxonomy_file:
        tax_map = _load_taxonomy(args.taxonomy_file)
    else:
        tax_map = _discover_taxonomy(args.data_dir, args.results_dir, ref)

    # ── Build tables ───────────────────────────────────────────────────────
    print("\n  Building Table S4 ...")
    t4 = build_table_s4(
        vanish_set, blossum_set,
        vanish_plus_ids, blossum_plus_ids,
        vanish_h_ids, blossum_h_ids,
        gmw_all, gmw_healthy, abund_df,
        tax_map,
    )
    p4 = os.path.join(out_dir, f"{pfx}table_s4_list_membership.csv")
    t4.to_csv(p4, index=False)
    print(f"  Table S4: {len(t4)} SGBs  →  {p4}")

    print("\n  Building Table S5 ...")
    t5 = build_table_s5(
        vanish_set, blossum_set,
        vanish_plus_ids, blossum_plus_ids,
        vanish_h_ids, blossum_h_ids,
    )
    p5 = os.path.join(out_dir, f"{pfx}table_s5_overlap_summary.csv")
    t5.to_csv(p5, index=False)
    print(f"  Table S5: {len(t5)} rows  →  {p5}")
    _print_s5(t5)

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
