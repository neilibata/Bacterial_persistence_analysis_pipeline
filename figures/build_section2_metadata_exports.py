#!/usr/bin/env python3
"""Build annotated Section 2 differential metadata exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("results/combined/section2_differential/main")
OUT = BASE / "metadata_exports"
COMBINED_META = Path("combined_sgb_metadata.xlsx")

LISTS = [
    ("top100_NI_enriched_prevalence", "top100_NI_enriched_prevalence.txt", "Top 100 Non-Industrial enriched by prevalence"),
    ("top100_Ind_enriched_prevalence", "top100_Ind_enriched_prevalence.txt", "Top 100 Industrial enriched by prevalence"),
    ("vanish_plus_ids", "vanish_plus_ids.txt", "VANISH+ IDs"),
    ("blossum_plus_ids", "blossum_plus_ids.txt", "BloSSUM+ IDs"),
    ("vanishH_ids", "vanishH_ids.txt", "VANISH-H IDs"),
    ("blossumH_ids", "blossumH_ids.txt", "BloSSUM-H IDs"),
]

LIST_LABELS = {
    "top100_NI_enriched_prevalence": "Top100 NI-enriched",
    "top100_Ind_enriched_prevalence": "Top100 Industrial-enriched",
    "vanish_plus_ids": "VANISH+",
    "blossum_plus_ids": "BloSSUM+",
    "vanishH_ids": "VanishH+",
    "blossumH_ids": "BlossumH+",
}

TAXONOMY_FIELDS = [
    "gtdb_lineage",
    "gtdb_domain",
    "gtdb_phylum",
    "gtdb_class",
    "gtdb_order",
    "gtdb_family",
    "gtdb_genus",
    "gtdb_species",
]


def is_blank(value) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "na"}


def read_id_list(path: Path) -> pd.DataFrame:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame({"rank": range(1, len(ids) + 1), "microbe": ids})


def load_list_memberships() -> dict[str, list[str]]:
    memberships: dict[str, list[str]] = {}
    for slug, filename, _description in LISTS:
        label = LIST_LABELS[slug]
        for microbe in read_id_list(BASE / filename)["microbe"]:
            memberships.setdefault(microbe, []).append(label)
    return memberships


def load_combined_metadata() -> pd.DataFrame:
    xl = pd.ExcelFile(COMBINED_META)
    frames = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(COMBINED_META, sheet_name=sheet)
        df["metadata_sheet"] = sheet
        frames.append(df)

    candidates = []
    for df in frames:
        id_cols = [c for c in df.columns if c in {"combined_id", "representative", "sgb", "microbe"}]
        taxonomy_cols = [
            c for c in df.columns
            if any(token in c.lower() for token in ["tax", "species", "genus", "family", "order", "class", "phylum"])
        ]
        keep = ["metadata_sheet"] + id_cols + taxonomy_cols
        if id_cols:
            candidates.append(df.loc[:, list(dict.fromkeys(keep))].copy())

    rows = []
    for df in candidates:
        id_cols = [c for c in df.columns if c in {"combined_id", "representative", "sgb", "microbe"}]
        for id_col in id_cols:
            tmp = df.copy()
            tmp["microbe"] = tmp[id_col].astype(str).str.strip()
            tmp["metadata_match_field"] = id_col
            rows.append(tmp)

    if not rows:
        return pd.DataFrame({"microbe": []})

    meta = pd.concat(rows, ignore_index=True)
    meta = meta[meta["microbe"].notna() & (meta["microbe"] != "") & (meta["microbe"].str.lower() != "nan")]
    meta = meta.drop_duplicates(subset=["microbe"], keep="first")
    preferred = ["microbe", "metadata_sheet", "metadata_match_field"]
    rest = [c for c in meta.columns if c not in preferred]
    return meta[preferred + rest]


def build_exports() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prevalence = pd.read_csv(BASE / "prevalence_ind_vs_nonind.csv")
    metadata = load_combined_metadata()
    list_memberships = load_list_memberships()

    overview_rows = []
    tables = {}
    for slug, filename, description in LISTS:
        ids = read_id_list(BASE / filename)
        table = ids.merge(prevalence, on="microbe", how="left")
        table = table.merge(metadata, on="microbe", how="left")
        table["list_memberships"] = table["microbe"].map(
            lambda m: "; ".join(list_memberships.get(m, [])))
        table["category_7"] = table["list_memberships"]
        table.insert(1, "list_name", slug)
        table.insert(2, "description", description)
        table.to_csv(OUT / f"{slug}_metadata.csv", index=False)
        tables[slug] = table
        overview_rows.append(
            {
                "list_name": slug,
                "description": description,
                "source_file": filename,
                "n_ids": len(ids),
                "n_with_prevalence_stats": int(table["prev_g1"].notna().sum()) if "prev_g1" in table else 0,
                "n_with_metadata_match": int(table["metadata_sheet"].notna().sum()) if "metadata_sheet" in table else 0,
                "n_need_taxonomy_recheck": int(build_taxonomy_recheck(table).shape[0]),
            }
        )

    overview = pd.DataFrame(overview_rows)
    overview.to_csv(OUT / "section2_differential_metadata_export_overview.csv", index=False)

    taxonomy_recheck = build_taxonomy_recheck(pd.concat(tables.values(), ignore_index=True))
    taxonomy_recheck.to_csv(OUT / "taxa_missing_taxonomy_recheck.csv", index=False)

    workbook = OUT / "section2_differential_combined_metadata_lists.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="Overview", index=False)
        taxonomy_recheck.to_excel(writer, sheet_name="Taxonomy recheck", index=False)
        for slug, table in tables.items():
            table.to_excel(writer, sheet_name=slug[:31], index=False)

        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.style = "Headline 4"
            for col in ws.columns:
                header = str(col[0].value or "")
                max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col[:101])
                width = min(max(max_len + 2, len(header) + 2, 10), 45)
                ws.column_dimensions[col[0].column_letter].width = width


def build_taxonomy_recheck(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()

    rows = []
    for microbe, grp in table.groupby("microbe", dropna=False):
        missing_fields = []
        if "metadata_sheet" not in grp or grp["metadata_sheet"].isna().all():
            missing_fields.append("metadata_match")
        for field in TAXONOMY_FIELDS:
            if field in grp.columns and grp[field].apply(is_blank).all():
                missing_fields.append(field)

        if not missing_fields:
            continue

        first = grp.iloc[0]
        rows.append({
            "microbe": microbe,
            "lists": "; ".join(sorted(grp["list_name"].dropna().astype(str).unique())),
            "missing_or_blank_fields": "; ".join(missing_fields),
            "metadata_sheet": first.get("metadata_sheet", ""),
            "metadata_match_field": first.get("metadata_match_field", ""),
            "gtdb_lineage": first.get("gtdb_lineage", ""),
            "gtdb_domain": first.get("gtdb_domain", ""),
            "gtdb_phylum": first.get("gtdb_phylum", ""),
            "gtdb_class": first.get("gtdb_class", ""),
            "gtdb_order": first.get("gtdb_order", ""),
            "gtdb_family": first.get("gtdb_family", ""),
            "gtdb_genus": first.get("gtdb_genus", ""),
            "gtdb_species": first.get("gtdb_species", ""),
            "prev_g1": first.get("prev_g1", ""),
            "prev_g2": first.get("prev_g2", ""),
            "prevalence_diff": first.get("prevalence_diff", ""),
            "neg_log10_p_bonf": first.get("neg_log10_p_bonf", ""),
            "category_7": first.get("category_7", ""),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "microbe", "lists", "missing_or_blank_fields",
            "metadata_sheet", "metadata_match_field",
            *TAXONOMY_FIELDS,
            "prev_g1", "prev_g2", "prevalence_diff",
            "neg_log10_p_bonf", "category_7",
        ])

    return (
        pd.DataFrame(rows)
        .sort_values(["missing_or_blank_fields", "microbe"])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    build_exports()
