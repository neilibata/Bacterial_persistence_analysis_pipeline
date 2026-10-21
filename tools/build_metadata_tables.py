#!/usr/bin/env python3
"""Build the manuscript's Metadata Tables.

    M1  curated per-sample metadata        metadata_M1.csv         (built by
                                                                    tools/enrich_metadata_readlen.py)
    M2  reference-catalogue metadata       metadata_M2.xlsx        (this script)
    M3  VANISH/BloSSUM taxon-list metadata metadata_M3.xlsx        (this script)

M2 covers all three catalogues used by the pipeline in one table (Carter/UHGG,
Leviatan, Combined; plus the chimp side reference where only lengths exist).
M3 covers the six taxon-list families (VANISH, BloSSUM, VANISH+, BloSSUM+,
VanishH+, BlossumH+) in both reference ID spaces.

List IDs are resolved into each reference's ID space with the pipeline's own
crosswalk (``persistence.loading._load_combined_taxon_crosswalk``), so M3
membership matches what Sections 2-3 actually analysed.

    python tools/build_metadata_tables.py            # -> metadata_M2.xlsx, metadata_M3.xlsx
    python tools/build_metadata_tables.py --bundle   # + metadata_M1-M3.xlsx (embeds M1)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence import config  # noqa: E402
from persistence.loading import _load_combined_taxon_crosswalk  # noqa: E402

META_XLSX = ROOT / "combined_sgb_metadata.xlsx"

GTDB_RANKS = ["gtdb_domain", "gtdb_phylum", "gtdb_class", "gtdb_order",
              "gtdb_family", "gtdb_genus", "gtdb_species"]

# Catalogues in M2: display name -> (config reference key, metadata sheet or None)
CATALOGUES = [
    ("Carter (UHGG)", "UHGG", "UHGG"),
    ("Leviatan", "lev", "Leviatan"),
    ("Combined", "combined", "Combined"),
    ("Chimp", "chimp", None),
]

# The six taxon-list families of Section 2. ``derived`` files live per
# reference under the --lists-dir tree; ``curated`` files are
# reference-agnostic root lists in the UHGG ID space.
LIST_FAMILIES = [
    ("VANISH", "curated", "Vanish_list.txt",
     "Curated non-industrial-associated taxa (VANISH; Sonnenburg & Sonnenburg 2019)"),
    ("BloSSUM", "curated", "Blossum_list.txt",
     "Curated industrial-associated taxa (BloSSUM)"),
    ("VANISH+", "pinned", "vanish_plus_ids.txt",
     "Top-100 differentially-prevalent non-industrial-enriched SGBs (data-derived)"),
    ("BloSSUM+", "pinned", "blossum_plus_ids.txt",
     "Top-100 differentially-prevalent industrial-enriched SGBs (data-derived)"),
    ("VanishH+", "pinned", "vanishH_ids.txt",
     "VANISH+ restricted to the health-analysis cohort"),
    ("BlossumH+", "pinned", "blossumH_ids.txt",
     "BloSSUM+ restricted to the health-analysis cohort"),
]

LIST_REFS = ["UHGG", "combined"]

PHYLO_FALLBACK = {"VANISH": "Vanish_phylo.txt", "BloSSUM": "Blossom_phylo.txt"}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def read_lengths(ref: str) -> pd.Series:
    """SGB -> genome length (bp) for one reference, or an empty Series."""
    p = ROOT / config.REFERENCES[ref]["lengths_file"]
    if not p.exists():
        return pd.Series(dtype="int64")
    d = pd.read_csv(p, sep=r"\s+", header=None, names=["sgb", "genome_length_bp"])
    return d.set_index(d.sgb.astype(str).str.strip())["genome_length_bp"].astype("int64")


def matrix_columns(ref: str) -> set[str]:
    """SGB IDs present as columns of a reference's breadth matrix (header only)."""
    p = ROOT / config.REFERENCES[ref]["breadth_file"]
    if not p.exists():
        return set()
    with open(p) as fh:
        header = fh.readline().rstrip("\n").split(",")
    out = set()
    for col in header[1:]:
        for suffix in ("_mean", "_std"):
            if col.endswith(suffix):
                out.add(col[: -len(suffix)])
                break
    return out


def read_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    seen, out = set(), []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def phylo_fallback(name: str) -> dict[str, str]:
    """GTDB lineage from the curated *_phylo.txt sidecars (id<TAB>lineage)."""
    fname = PHYLO_FALLBACK.get(name)
    if not fname or not (ROOT / fname).exists():
        return {}
    out = {}
    for line in (ROOT / fname).read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip():
            out[parts[0].strip()] = parts[1].strip()
    return out


def split_lineage(lineage: str) -> dict[str, str]:
    ranks = {}
    for tok in str(lineage).split(";"):
        tok = tok.strip()
        if len(tok) > 3 and tok[1:3] == "__":
            key = {"d": "gtdb_domain", "p": "gtdb_phylum", "c": "gtdb_class",
                   "o": "gtdb_order", "f": "gtdb_family", "g": "gtdb_genus",
                   "s": "gtdb_species"}.get(tok[0])
            if key:
                ranks[key] = tok
    return ranks


# ---------------------------------------------------------------------------
# M2 - reference catalogues
# ---------------------------------------------------------------------------

def build_m2() -> tuple[dict[str, pd.DataFrame], set[str]]:
    """M2 = combined_sgb_metadata.xlsx carried through verbatim, plus the columns
    and one sheet that tie the catalogues to this analysis' inputs.

    The clustering metadata already lives in combined_sgb_metadata.xlsx, so M2
    does not restate it: every sheet keeps its original name, row order and
    columns. The only additions are ``genome_length_bp`` / ``in_breadth_matrix``
    on the three per-catalogue sheets (from the length files and matrix headers,
    which are not in the source workbook) and an ``Analysis_link`` sheet naming
    the exact input file per reference.
    """
    # header=None keeps the Summary sheet's report layout intact.
    sheets = pd.read_excel(META_XLSX, sheet_name=None)
    raw = {}
    if "Summary" in sheets:
        raw["Summary"] = pd.read_excel(META_XLSX, sheet_name="Summary", header=None)

    lengths = {ref: read_lengths(ref) for _, ref, _ in CATALOGUES}
    cols = {ref: matrix_columns(ref) for _, ref, _ in CATALOGUES}

    out: dict[str, pd.DataFrame] = {}
    for name, df in sheets.items():
        out[name] = raw.get(name, df).copy()

    id_col_for = {"UHGG": "sgb", "Leviatan": "sgb", "Combined": "representative"}
    for disp, ref, sheet in CATALOGUES:
        if sheet is None or sheet not in out:
            continue
        d, idc = out[sheet], id_col_for[sheet]
        ids = d[idc].astype(str).str.strip()
        d["genome_length_bp"] = ids.map(lengths[ref])
        d["in_breadth_matrix"] = ids.isin(cols[ref]) if cols[ref] else pd.NA

    # ---- Analysis_link: the only genuinely new tabular content --------------
    rows = []
    for disp, ref, sheet in CATALOGUES:
        L, C = lengths[ref], cols[ref]
        cfg = config.REFERENCES[ref]
        rows.append({
            "Catalogue": disp,
            "Reference key (--ref)": ref,
            "M2 sheet": sheet or "—  (lengths only; not in combined_sgb_metadata.xlsx)",
            "SGBs": len(L) if len(L) else pd.NA,
            "SGBs present as breadth-matrix columns": len(C) if C else pd.NA,
            "Total catalogue bp": int(L.sum()) if len(L) else pd.NA,
            "Median genome length (bp)": int(L.median()) if len(L) else pd.NA,
            "Min genome length (bp)": int(L.min()) if len(L) else pd.NA,
            "Max genome length (bp)": int(L.max()) if len(L) else pd.NA,
            "Genome-length file": cfg["lengths_file"],
            "Breadth matrix": cfg["breadth_file"],
            "Depth matrix": cfg["depth_file"],
        })
    out["Analysis_link"] = pd.DataFrame(rows)

    order = (["Summary", "Analysis_link"]
             + [s for s in ("Combined", "Leviatan", "UHGG", "Shared_Pairs",
                            "Rep_Choice", "Fuzzy_Boundary") if s in out])
    ordered = {k: out[k] for k in order if k in out}
    ordered.update({k: v for k, v in out.items() if k not in ordered})
    return ordered, set(raw)


# ---------------------------------------------------------------------------
# M3 - taxon lists
# ---------------------------------------------------------------------------

# Where the derived VANISH+/BloSSUM+/VanishH+/BlossumH+ lists are read from.
# Set by main() so the caller can point it elsewhere. They are not
# interchangeable: a cohort change moves a few members of each ranked top-100,
# so M3 must name which membership it describes.
#
# The default is the membership the reported run produced, committed under
# metadata/lists_as_analysed/, because M3 is Data S4 and Data S4 has to be the
# membership the paper's numbers were computed from.
#
# Until 2026-10-20 it was a frozen snapshot of an earlier derivation, which
# was wrong and provably so: Table S6 reports the pairwise overlaps 40, 96,
# 42, 38 and 82 for the combined reference, that snapshot yielded 40, 95, 42,
# 39 and 86, and the reported run's own lists yield all five. The snapshot has
# been removed; tools/make_submission.py rebuilds this directory from results/
# so it stays the run's.
LISTS_DIR = ROOT / "metadata" / "lists_as_analysed"


#: Where a run leaves its derived lists, relative to results/<ref>/.
_RUN_LIST_SUBDIR = Path("section2_differential") / "main"


def list_path(family: str, kind: str, fname: str, ref: str) -> Path:
    """Resolve one list file under LISTS_DIR.

    Two layouts are accepted, because the lists have two homes and M3 must be
    buildable straight from either. A flat `<ref>/<file>.txt` is the committed
    snapshot under metadata/lists_as_analysed/; `<ref>/section2_differential/
    main/<file>.txt` is where a run writes them. Pointing --lists-dir at a
    results tree used to match nothing and silently build M3 from the curated
    lists alone, which is how a wrong Data S4 got as far as it did.
    """
    if kind == "curated":
        return ROOT / fname
    flat = LISTS_DIR / ref / fname
    if flat.exists():
        return flat
    from_run = LISTS_DIR / ref / _RUN_LIST_SUBDIR / fname
    return from_run if from_run.exists() else flat


#: The committed copy of the lists M3 describes.
SNAPSHOT_DIR = ROOT / "metadata" / "lists_as_analysed"


def _sync_snapshot() -> None:
    """Make the committed list snapshot a by-product of the build.

    M3 is Data S4, so the lists it describes have to be the ones the reported
    run derived. Keeping a separate hand-updated copy of those lists is what
    let Data S4 describe a membership the paper never analysed: the snapshot
    was a run behind, and nothing compared them.

    So whenever M3 is built from anywhere other than the snapshot, the files
    actually read are written back into it. The snapshot then cannot disagree
    with the workbook, because the same build produced both.
    """
    if LISTS_DIR == SNAPSHOT_DIR:
        return
    import shutil
    copied = 0
    for ref in LIST_REFS:
        for family, kind, fname, _desc in LIST_FAMILIES:
            if kind == "curated":
                continue
            src = list_path(family, kind, fname, ref)
            if not src.exists():
                continue
            dst = SNAPSHOT_DIR / ref / fname
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or dst.read_bytes() != src.read_bytes():
                shutil.copy2(src, dst)
            copied += 1
    if copied:
        print(f"\n  synced {copied} list file(s) into "
              f"{SNAPSHOT_DIR.relative_to(ROOT)}, so the committed snapshot "
              f"matches the workbook just written")


def build_m3(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    crosswalk = _load_combined_taxon_crosswalk(META_XLSX)
    lengths = {ref: read_lengths(ref) for ref in LIST_REFS}
    cols = {ref: matrix_columns(ref) for ref in LIST_REFS}

    # taxonomy lookups, keyed in each reference's own ID space
    tax = {}
    uh = sheets["UHGG"].copy()
    tax["UHGG"] = uh.set_index(uh["sgb"].astype(str).str.strip())
    cb = sheets["Combined"].copy()
    tax["combined"] = cb.set_index(cb["representative"].astype(str).str.strip())
    combined_id_by_rep = dict(zip(cb["representative"].astype(str).str.strip(),
                                  cb["combined_id"].astype(str)))

    members: list[dict] = []
    resolved: dict[tuple[str, str], set[str]] = {}
    summary: list[dict] = []

    for family, kind, fname, definition in LIST_FAMILIES:
        fb = phylo_fallback(family)
        for ref in LIST_REFS:
            p = list_path(family, kind, fname, ref)
            ids = read_ids(p)
            if not ids:
                continue
            valid = cols[ref] or set(lengths[ref].index)
            keep: set[str] = set()
            n_remapped = n_absent = 0
            for src in ids:
                rid = crosswalk.get(src, src) if ref == "combined" else src
                present = rid in valid
                if ref == "combined" and rid != src:
                    n_remapped += 1
                if not present:
                    n_absent += 1
                else:
                    keep.add(rid)
                row = {"reference": ref, "list": family, "source_id": src,
                       "resolved_id": rid, "remapped": rid != src,
                       "in_reference": present,
                       "combined_id": combined_id_by_rep.get(rid, pd.NA)
                       if ref == "combined" else pd.NA,
                       "genome_length_bp": lengths[ref].get(rid, pd.NA)}
                t = tax[ref]
                if rid in t.index:
                    r = t.loc[rid]
                    r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
                    for rank in GTDB_RANKS:
                        row[rank] = r.get(rank, pd.NA)
                    row["gtdb_lineage"] = r.get("gtdb_lineage", pd.NA)
                else:
                    row.update({rank: pd.NA for rank in GTDB_RANKS})
                    row["gtdb_lineage"] = pd.NA
                if pd.isna(row.get("gtdb_lineage")) and src in fb:
                    row["gtdb_lineage"] = fb[src]
                    row.update(split_lineage(fb[src]))
                members.append(row)

            resolved[(ref, family)] = keep
            summary.append({
                "reference": ref, "list": family, "definition": definition,
                "source_file": str(p.relative_to(ROOT)
                                   if p.is_relative_to(ROOT) else p),
                "n_source_ids": len(ids),
                "n_remapped_to_representative": n_remapped,
                "n_absent_from_reference": n_absent,
                "n_analysed": len(keep),
            })

    mem = pd.DataFrame(members)
    cols_order = ["reference", "list", "source_id", "resolved_id", "remapped",
                  "in_reference", "combined_id", "genome_length_bp",
                  "gtdb_lineage"] + GTDB_RANKS
    mem = mem[[c for c in cols_order if c in mem.columns]]

    # ---- Membership matrix (one row per reference x SGB) --------------------
    families = [f for f, _, _, _ in LIST_FAMILIES]
    wide_rows = []
    for ref in LIST_REFS:
        universe = sorted(set().union(*[resolved.get((ref, f), set()) for f in families]))
        sub = mem[(mem.reference == ref) & mem.in_reference]
        first = sub.drop_duplicates("resolved_id").set_index("resolved_id")
        for sgb in universe:
            row = {"reference": ref, "sgb": sgb}
            row.update({f: sgb in resolved.get((ref, f), set()) for f in families})
            row["n_lists"] = sum(row[f] for f in families)
            if sgb in first.index:
                r = first.loc[sgb]
                row["genome_length_bp"] = r.get("genome_length_bp", pd.NA)
                row["gtdb_phylum"] = r.get("gtdb_phylum", pd.NA)
                row["gtdb_family"] = r.get("gtdb_family", pd.NA)
                row["gtdb_species"] = r.get("gtdb_species", pd.NA)
            wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)

    # ---- Pairwise overlaps (reproduces Table S5 for the combined reference) --
    ov = []
    for ref in LIST_REFS:
        for i, a in enumerate(families):
            for b in families[i + 1:]:
                A, B = resolved.get((ref, a), set()), resolved.get((ref, b), set())
                if not A or not B:
                    continue
                inter, union = len(A & B), len(A | B)
                ov.append({"reference": ref, "list_A": a, "list_B": b,
                           "n_A": len(A), "n_B": len(B), "overlap": inter,
                           "A_only": len(A) - inter, "B_only": len(B) - inter,
                           "Jaccard": round(inter / union, 4) if union else 0.0})
    overlaps = pd.DataFrame(ov)

    return {"Summary": pd.DataFrame(summary), "Members": mem,
            "Membership": wide, "Overlaps": overlaps}


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

README_M2 = [
    ("Table", "Metadata Table M2 - reference-catalogue metadata"),
    ("Relationship to combined_sgb_metadata.xlsx",
     "M2 IS that workbook. Every sheet is carried through verbatim - same names, "
     "row order and columns - so the two never diverge. M2 adds only what the "
     "source workbook does not contain (below)."),
    ("Added: genome_length_bp",
     "Appended to Combined / Leviatan / UHGG from all_SGB_lengths_*.txt "
     "(keyed on 'representative' for Combined, 'sgb' for the catalogue sheets)"),
    ("Added: in_breadth_matrix",
     "Whether the SGB appears as a column of that reference's breadth matrix"),
    ("Added: Analysis_link sheet",
     "One row per reference: SGB and bp totals, genome-length distribution, and "
     "the exact length/breadth/depth input filenames - including the chimp side "
     "reference, which has lengths only and no sheet in the source workbook"),
    ("Summary", "The source workbook's own summary report, unmodified"),
    ("Source", "combined_sgb_metadata.xlsx + all_SGB_lengths_*.txt"),
    ("Built by", "tools/build_metadata_tables.py"),
]

README_M3 = [
    ("Table", "Metadata Table M3 - VANISH/BloSSUM taxon-list metadata"),
    ("Scope", "Six list families x two reference ID spaces (Carter/UHGG and Combined)"),
    ("Summary", "Per list x reference: definition, source file, ID counts, "
                "how many IDs were remapped to a combined representative, and how "
                "many were absent from that reference"),
    ("Members", "One row per source ID: resolved ID, GTDB lineage, genome length"),
    ("Membership", "One row per reference x SGB with a boolean per list family"),
    ("Overlaps", "Pairwise overlap and Jaccard between list families "
                 "(the combined rows reproduce Supplementary Table S5)"),
    ("ID resolution", "persistence.loading._load_combined_taxon_crosswalk - the same "
                      "crosswalk Sections 2-3 use, so membership matches the analysis"),
    ("Source", "Vanish_list.txt, Blossum_list.txt, *_phylo.txt, "
               "metadata/lists_as_analysed/<ref>/*.txt"),
    ("Built by", "tools/build_metadata_tables.py"),
]


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame], readme: list[tuple],
                   raw: set[str] | None = None):
    """``raw`` names sheets written without a header row (preserved report layouts)."""
    raw = raw or set()
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        pd.DataFrame(readme, columns=["Item", "Description"]).to_excel(
            xl, sheet_name="README", index=False)
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name[:31], index=False,
                        header=name not in raw)
    shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    print(f"Wrote {shown}  "
          f"({len(sheets) + 1} sheets: README, {', '.join(sheets)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", action="store_true",
                    help="also write metadata_M1-M3.xlsx embedding M1 (large)")
    ap.add_argument("--m1", type=Path, default=None,
                    help="per-sample metadata for the bundle. Defaults to the "
                         "file the pipeline itself loads (config.META_FILE), "
                         "not the stale metadata_M1.csv.")
    ap.add_argument("--lists-dir", type=Path, default=None,
                    help="directory of per-reference derived taxon lists "
                         "(<ref>/vanish_plus_ids.txt etc). Default: "
                         "metadata/lists_as_analysed, the membership the "
                         "reported run produced. Pass a results tree to "
                         "rebuild from a run directly.")
    ap.add_argument("--out-dir", type=Path, default=ROOT,
                    help="where to write the workbooks (default: repo root). "
                         "Point elsewhere to rebuild without touching the committed copies.")
    args = ap.parse_args()

    # Resolve the two inputs that decide what these tables describe, and say so
    # out loud: pointing either at a stale file produces a table that looks
    # right and documents a cohort nobody analysed.
    global LISTS_DIR
    if args.lists_dir is not None:
        LISTS_DIR = args.lists_dir
    if args.m1 is None:
        from persistence import config as _cfg
        args.m1 = ROOT / _cfg.META_FILE
    print(f"  per-sample metadata : {args.m1.name}")
    print(f"  derived taxon lists : {LISTS_DIR.relative_to(ROOT) if LISTS_DIR.is_relative_to(ROOT) else LISTS_DIR}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sheets = pd.read_excel(META_XLSX, sheet_name=None)

    m2, m2_raw = build_m2()
    write_workbook(out_dir / "metadata_M2.xlsx", m2, README_M2, raw=m2_raw)

    m3 = build_m3(sheets)
    write_workbook(out_dir / "metadata_M3.xlsx", m3, README_M3)
    _sync_snapshot()

    print("\nM3 list sizes (analysed):")
    for _, r in m3["Summary"].iterrows():
        print(f"  {r['reference']:>8}  {r['list']:<10} {r['n_analysed']:>4}"
              f"   (source {r['n_source_ids']}, remapped {r['n_remapped_to_representative']},"
              f" absent {r['n_absent_from_reference']})")

    if args.bundle:
        if not args.m1.exists():
            print(f"\n! {args.m1.name} not found - skipping the bundled workbook.")
            return
        m1 = pd.read_csv(args.m1, low_memory=False)
        bundle = {"M1_samples": m1}
        bundle.update({f"M2_{k}": v for k, v in m2.items()})
        bundle.update({f"M3_{k}": v for k, v in m3.items()})
        readme = ([("Table", "Metadata Tables M1-M3 (bundled)"),
                   ("M1_samples", f"Curated per-sample metadata - {len(m1):,} samples "
                                  f"x {m1.shape[1]} columns (from {args.m1.name})")]
                  + [(f"M2_{k}", v) for k, v in README_M2] + [(f"M3_{k}", v) for k, v in README_M3])
        write_workbook(out_dir / "metadata_M1-M3.xlsx", bundle, readme,
                       raw={f"M2_{k}" for k in m2_raw})


if __name__ == "__main__":
    main()
