#!/usr/bin/env python3
"""Build a self-contained interactive world-map metadata explorer.

Audits the curated sample metadata, groups the relevant columns, and aggregates
each by country (and study x country) — computing distribution-aware summaries
(mean +/- SD for ~normal variables; geometric mean x// geometric SD for ~lognormal
ones), per-country histograms, and the missingness of every field — then emits a
single self-contained, colour-blind-safe HTML explorer with:

  * an Equal-Earth world map (same geometry as manuscript Figure 1), light by default;
  * a two-level group -> variable menu spanning all informative columns;
  * a Central / Spread / Missingness metric toggle (viridis for value, magma for missing);
  * hover tooltips with the distribution summary + a mini-histogram + top studies;
  * a Map <-> Table view toggle and a light/dark theme toggle.

Run with the pipeline's env (pandas, numpy, geopandas):
    python tools/metadata_explorer/build_explorer.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The path setup has to precede the config import, not follow it: running this
# as a script puts tools/metadata_explorer on sys.path, not the repo root, so
# `import persistence.config` at the top of the file only resolved when the
# caller had already arranged for it.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "figures"))

import persistence.config as _cfg  # noqa: E402

STUDY_COL = "StudyFolder"

#: Column that joins the curated table to a run's `cohort_samples.csv`.
_COHORT_KEY = "Chosen_match_ID"
NBINS = 14
DISEASE_MIN_POS = 30  # keep a disease flag if >= this many positives dataset-wide

# ── The health categories the analysis actually reports ───────────────────────
# Figure 5D-E and every health supplementary figure show one pooled contrast
# (Unhealthy) and the four per-condition contrasts that clear the
# >=CMH_MIN_STRATUM / >=MIN_STRATA_TO_PLOT rule. The `Specific diseases` group
# below lists every `disease_*` flag with enough positives to be worth plotting,
# and the `Health status` group lists the superseded `Diseased (...)` roll-ups,
# so neither tells a reader which conditions the paper is actually about. This
# field does.
#
# The column lists come from config rather than being restated here: the
# explorer and the analysis must not be able to disagree about what "IBD" means,
# and restating them is exactly how this file came to describe a different
# cohort from the pipeline once before.
HEALTH_CATEGORY_COL = "Health category"
HEALTH_FLAG_PREFIX = "Has: "

#: Named categories, in precedence order, each as (label, config key, colour).
#: Precedence settles the samples that carry more than one: a specific
#: diagnosis outranks the obesity flag, which is partly a BMI threshold and
#: co-occurs with all three diagnoses, and among the diagnoses cancer outranks
#: glucose intolerance. Only the Glucose-Intolerance/CRC pair and the obesity
#: overlaps are affected; build() reports how many samples the rule moved.
HEALTH_CATEGORIES = [
    ("IBD",                 "IBD",                 "#4a3aa7"),
    ("CRC / Adenoma",       "CRC / Adenoma",       "#e34948"),
    ("Glucose Intolerance", "Glucose Intolerance", "#eda100"),
    ("Obesity",             "Obese",               "#eb6834"),
]
HEALTH_HEALTHY_LABEL = "Healthy"
HEALTH_OTHER_LABEL = "Other Unhealthy"
HEALTH_CAT_COLORS = {HEALTH_HEALTHY_LABEL: "#1baf7a"}
HEALTH_CAT_COLORS.update({lab: col for lab, _, col in HEALTH_CATEGORIES})
HEALTH_CAT_COLORS[HEALTH_OTHER_LABEL] = "#8c8c86"

# Curated groups of columns that "go together". Each entry:
#   (col, label, kind, opts)   kind = numeric | proportion | categorical
# opts: transform (gbp|millions), dist (force normal|lognormal), fmt, unit
CURATED_GROUPS: list[tuple[str, list]] = [
    ("Sequencing effort", [
        ("Total bases",        "Sequencing depth (Gbp)",  "numeric",     {"transform": "gbp", "dist": "lognormal", "fmt": "num2"}),
        ("number_reads",       "Read count (millions)",   "numeric",     {"transform": "millions", "dist": "lognormal", "fmt": "num1"}),
        ("median_read_length", "Median read length (bp)", "numeric",     {"dist": "normal", "fmt": "int"}),
        ("NumPairedRuns",      "Sequencing runs / sample","numeric",     {"dist": "lognormal", "fmt": "num1"}),
        ("LineType",           "Run structure",           "categorical", {}),
    ]),
    ("Demographics", [
        ("Age",       "Age (years)", "numeric",     {"dist": "normal", "fmt": "num1"}),
        # Not the raw `Age group` column: it is a case-inconsistent mix of
        # Adult/adult, Elderly/senior/Senior, Child/child/schoolage that
        # renders as nine overlapping categories. `Age band` below is derived
        # from numeric Age by the same code the analysis bands with
        # (health_cmh.age_band_series), so the explorer and the demographic
        # supplementary figure cut age at the same places.
        ("Age band",  "Age band",    "categorical", {}),
        ("Gender",    "% Female",    "proportion",  {"true_value": "Female"}),
        ("BMI",       "BMI",         "numeric",     {"dist": "normal", "fmt": "num1"}),
    ]),
    ("Lifestyle", [
        ("Lifestyle", "Lifestyle", "categorical", {}),
        # "% Non-Industrial" is added synthetically from the Lifestyle composition below.
    ]),
    # The conditions the paper reports, first, because every other health group
    # here is either every flag in the table or a superseded roll-up.
    ("Health categories (as analysed)", [
        (HEALTH_CATEGORY_COL, "Health category", "categorical",
         {"catcolors": HEALTH_CAT_COLORS}),
        (f"{HEALTH_FLAG_PREFIX}Obesity", "% Obesity", "proportion",
         {"true_value": "Yes"}),
        (f"{HEALTH_FLAG_PREFIX}Glucose Intolerance", "% Glucose Intolerance",
         "proportion", {"true_value": "Yes"}),
        (f"{HEALTH_FLAG_PREFIX}IBD", "% IBD", "proportion",
         {"true_value": "Yes"}),
        (f"{HEALTH_FLAG_PREFIX}CRC / Adenoma", "% CRC / Adenoma", "proportion",
         {"true_value": "Yes"}),
    ]),
    ("Health status", [
        ("Unhealthy",                      "% Unhealthy",                 "proportion", {"true_value": "Unhealthy"}),
        ("Diseased (all)",                 "% Diseased (any)",            "proportion", {"true_value": "Yes"}),
        ("Diseased (gut and metabolic)",   "% Gut/metabolic disease",     "proportion", {"true_value": "Yes"}),
        ("Diseased (metabolic) or obese",  "% Metabolic disease/obese",   "proportion", {"true_value": "Yes"}),
        ("Diseased (other)",               "% Other disease",             "proportion", {"true_value": "Yes"}),
    ]),
    ("Obesity & weight", [
        ("Obesity",    "% Obese",      "proportion", {"true_value": "Yes"}),
        ("Overweight", "% Overweight", "proportion", {"true_value": "Yes"}),
    ]),
]

# Friendly names for specific-disease flags (fallback prettifies the column).
DISEASE_LABELS = {
    "disease_CRC": "Colorectal cancer", "disease_adenoma": "Adenoma",
    "disease_T2D": "Type 2 diabetes", "disease_IGT": "Glucose intolerance",
    "disease_IBD": "IBD", "disease_Crohn's": "Crohn's disease",
    "disease_Ulcerative_colitis": "Ulcerative colitis", "disease_fatty_liver": "Fatty liver (MASLD)",
    "disease_cirrhosis": "Cirrhosis", "disease_hepatitis": "Hepatitis",
    "disease_ACVD": "Atherosclerotic CVD", "disease_hypertension": "Hypertension",
    "disease_MS": "Multiple sclerosis", "disease_PD": "Parkinson's disease",
    "disease_RA": "Rheumatoid arthritis", "disease_ankylosing_spondylitis": "Ankylosing spondylitis",
    "disease_bone_disease": "Bone disease (osteoporosis/osteopenia)",
    "disease_Behcets": "Behcet's disease", "disease_myalgic_encephalomyelitis": "ME/CFS",
    "disease_ME_CFS": "ME/CFS (alt)", "disease_schizofrenia": "Schizophrenia",
    "disease_melanoma": "Melanoma", "disease_metastases": "Metastatic cancer",
    "disease_CDI": "C. difficile infection", "disease_STH": "Soil-transmitted helminth",
    "disease_ascites": "Ascites", "disease_acute_diarrhoea": "Acute diarrhoea",
    "disease_STEC": "STEC infection", "disease_treatment_colitis": "Treatment colitis",
}

CAT_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#008300", "#4a3aa7", "#e34948"]
# Lifestyle colours: Okabe-Ito, matching the manuscript (config.py) and colour-blind-safe.
# Raw metadata value -> (display label, colour).
# Labels come from config, not restated here: the published display names
# changed on 2026-08-14 (Transitional -> Agro-pastoralist, Hunter-Gatherer ->
# Hunter-gatherer) and this file kept the old ones, so the explorer disagreed
# with Figure 1 and Table M1 about what the groups are called.
LIFE_ORDER = [
    ("Industrial",   _cfg.LIFESTYLE_DISPLAY.get("Industrial", "Industrial"),
     "#56B4E9"),
    (_cfg.MIDDLE_LIFESTYLE_KEY, _cfg.MIDDLE_LIFESTYLE_DISPLAY, "#F0E442"),
    (_cfg.HG_LIFESTYLE_KEY,     _cfg.HG_LIFESTYLE_DISPLAY,     "#D55E00"),
]
LIFE_LABEL = {raw: lab for raw, lab, _ in LIFE_ORDER}
LIFE_COLOR = {lab: col for _, lab, col in LIFE_ORDER}   # keyed by display label
LIFE_RAW3 = ["Industrial", "Transitional", "H-G"]


def _load_world_paths(simplify_m=30000.0, width=1000.0):
    from make_metadata_summary import _load_world
    world = _load_world().copy()
    if "name" in world.columns:
        world = world[~world["name"].astype(str).str.contains("Antarctica", case=False, na=False)]
    world = world.to_crs("EPSG:8857")
    world = world[world.geometry.notna() & ~world.geometry.is_empty].copy()
    world["geometry"] = world.geometry.simplify(simplify_m)
    minx, miny, maxx, maxy = world.total_bounds
    sx = width / (maxx - minx)
    height = (maxy - miny) * sx

    def to_path(geom):
        parts = []
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            if poly.is_empty:
                continue
            for ring in [poly.exterior, *poly.interiors]:
                xs, ys = ring.coords.xy
                pts = [f"{(x-minx)*sx:.1f} {(maxy-y)*sx:.1f}" for x, y in zip(xs, ys)]
                if pts:
                    parts.append("M" + "L".join(pts) + "Z")
        return "".join(parts)

    paths = {}
    for _, r in world.iterrows():
        d = to_path(r.geometry)
        if d:
            paths[str(r["name_norm"])] = d
    return paths, round(width, 1), round(height, 1)


def _obesity_flag(raw):
    """The analysis's obesity flag, derived the way loading.py derives it.

    Any of `config.OBESITY_COLUMNS` reading yes, **or** BMI >= 30. Both halves
    matter: the columns are only partly populated and BMI reaches samples they
    miss, which is why `loading.py` ORs them rather than preferring one.
    """
    from persistence.common import is_yes

    cols = [c for c in _cfg.OBESITY_COLUMNS if c in raw.columns]
    flag = (raw[cols].apply(lambda r: any(is_yes(v) for v in r), axis=1)
            if cols else pd.Series(False, index=raw.index))
    if "BMI" in raw.columns:
        flag = flag | (pd.to_numeric(raw["BMI"], errors="coerce") >= 30).fillna(False)
    return flag.astype(bool)


def _condition_flag(raw, key):
    """Boolean flag for one `config.DISEASE_NCD_MAP` entry, by its config key."""
    from persistence.common import is_yes

    spec = _cfg.DISEASE_NCD_MAP.get(key)
    if spec is None:
        raise KeyError(
            f"config.DISEASE_NCD_MAP has no {key!r}. HEALTH_CATEGORIES names "
            f"the conditions the manuscript reports; if one was renamed in "
            f"config, rename it here too rather than letting this field "
            f"silently drop a category the figures still show. Available: "
            f"{list(_cfg.DISEASE_NCD_MAP)}")
    if spec.get("type") == "obese":
        return _obesity_flag(raw)
    cols = [c for c in spec.get("cols", []) if c in raw.columns]
    if not cols:
        return pd.Series(False, index=raw.index)
    return raw[cols].apply(lambda r: any(is_yes(v) for v in r), axis=1).astype(bool)


def add_health_categories(raw):
    """Add `Health category` plus one `Has: <condition>` flag per category.

    The categorical column is what a reader wants on the map -- one value per
    sample, so the per-country counts add up -- but assigning one value means
    choosing for the samples that carry two conditions. The `Has:` flags are
    the same information without that choice, so the overlap stays visible
    rather than being resolved away silently.

    Samples whose curated `Unhealthy` label is absent are left unset, not
    folded into Healthy: "not recorded" and "recorded as healthy" are
    different, and the explorer already reports missingness per field.
    """
    if "Unhealthy" not in raw.columns:
        print("    [warn] no `Unhealthy` column; skipping the health categories")
        return raw, {}

    unh = raw["Unhealthy"].astype(str).str.strip()
    is_unhealthy = unh.eq("Unhealthy")
    is_healthy = unh.eq("Healthy")

    flags, named = {}, pd.Series(False, index=raw.index)
    for label, key, _col in HEALTH_CATEGORIES:
        f = _condition_flag(raw, key)
        flags[label] = f
        named = named | f
        raw[f"{HEALTH_FLAG_PREFIX}{label}"] = np.where(f, "Yes", "No")

    cat = pd.Series(pd.NA, index=raw.index, dtype="object")
    cat[is_healthy] = HEALTH_HEALTHY_LABEL
    cat[is_unhealthy & ~named] = HEALTH_OTHER_LABEL
    # Reverse precedence order so the earliest entry wins the last write.
    for label, _key, _col in reversed(HEALTH_CATEGORIES):
        cat[flags[label]] = label
    raw[HEALTH_CATEGORY_COL] = cat

    multi = sum(f.astype(int) for f in flags.values())
    stats = {
        "counts": {lab: int((cat == lab).sum())
                   for lab in HEALTH_CAT_COLORS},
        "unset": int(cat.isna().sum()),
        "in_more_than_one": int((multi > 1).sum()),
        "named_outside_unhealthy": int((named & ~is_unhealthy).sum()),
    }
    shown = "  ".join(f"{lab} {n:,}" for lab, n in stats["counts"].items() if n)
    print(f"    Health categories: {shown}")
    print(f"      {stats['unset']:,} with no curated health label (left unset); "
          f"{stats['in_more_than_one']:,} carry more than one condition and "
          f"follow the precedence rule")
    if stats["named_outside_unhealthy"]:
        # Every named condition sat inside the Unhealthy label when this was
        # written. If that stops being true the categorical silently relabels
        # Healthy samples as diseased, so it is worth a line rather than a
        # surprise.
        print(f"      [note] {stats['named_outside_unhealthy']:,} sample(s) "
              f"carry a named condition without the Unhealthy label")
    return raw, stats


def _norm_gender(s):
    m = {"female": "Female", "male": "Male"}
    return s.astype(str).str.strip().str.lower().map(m)


def _transform(x, opts):
    x = pd.to_numeric(x, errors="coerce")
    t = opts.get("transform")
    if t == "gbp":
        return x / 1e9
    if t == "millions":
        return x / 1e6
    return x


def build(meta_path: Path, cohort_path: Path | None = None):
    from persistence.common import normalise_country, get_continent

    raw = pd.read_csv(meta_path, low_memory=False)
    # Table M1 stores the manuscript's display labels (Agro-pastoralist,
    # Hunter-gatherer); analysis code compares against the internal keys.
    if "Lifestyle" in raw.columns:
        raw["Lifestyle"] = _cfg.canonicalize_lifestyle_series(raw["Lifestyle"])

    n_source = len(raw)
    n_outside = 0
    cohort_used = None

    # The curated table is a superset of what the analysis reports on: it holds
    # every sample that was linked to a matrix, while the reported cohort is
    # the subset that also cleared the depth and coverage requirements. The gap
    # is small but it moves the headline numbers, including two whole
    # categories -- one country and one study survive in the table and in no
    # analysis. A reader comparing this page against the paper has no way to
    # tell a filtered-out sample from a disagreement, so restrict to the
    # cohort the run actually wrote whenever that file is at hand, and say in
    # the footer which it is.
    if cohort_path is not None and Path(cohort_path).is_file():
        ch = pd.read_csv(cohort_path)
        if _COHORT_KEY in ch.columns and _COHORT_KEY in raw.columns:
            keep = raw[_COHORT_KEY].isin(set(ch[_COHORT_KEY]))
            n_outside = int((~keep).sum())
            raw = raw[keep].reset_index(drop=True)
            cohort_used = Path(cohort_path)
            print(f"  Restricting to the analysed cohort in "
                  f"{_rel(cohort_path)}: {len(raw):,} of {n_source:,} rows "
                  f"({n_outside:,} outside it)")
        else:
            print(f"  [warn] {_rel(cohort_path)} has no {_COHORT_KEY} column; "
                  f"showing the full table instead")

    # Match the pipeline: samples whose Lifestyle resolves to neither Industrial
    # nor Non-Industrial are dropped at load (config.DROP_UNCLASSIFIED_LIFESTYLE),
    # so the explorer must not show them either or its totals disagree with the
    # manuscript's. Restricting to the cohort above already removed these, so
    # this reports nothing on a cohort-restricted build.
    dropped_life = {}
    if "Lifestyle" in raw:
        life = raw["Lifestyle"].astype(str).str.strip()
        unclassified = life.notna() & ~life.isin(LIFE_RAW3) & (life != "nan")
        if unclassified.any():
            dropped_life = {str(k): int(v) for k, v in
                            life[unclassified].value_counts().items()}
            print(f"  Dropping {int(unclassified.sum()):,} samples with an "
                  f"unclassifiable Lifestyle: {dropped_life}")
            raw = raw[~unclassified].reset_index(drop=True)

    # Derived here rather than expected in the file: it is a presentation of
    # `Age`, and deriving it keeps the bands identical to the analysis's.
    try:
        from persistence.health_cmh import age_band_series
        raw["Age band"] = age_band_series(raw)
    except Exception as exc:                       # pragma: no cover
        print(f"  [WARN] Age band not derived ({exc}); falling back to "
              "the raw Age group column")
        raw["Age band"] = raw.get("Age group")

    # After the cohort and lifestyle filters, so the category counts describe
    # exactly the rows the rest of the page describes.
    raw, health_stats = add_health_categories(raw)

    raw["Country_norm"] = raw["Country (full)"].map(
        lambda v: normalise_country(v) if pd.notna(v) else None)
    if "Gender" in raw:
        raw["Gender"] = _norm_gender(raw["Gender"])
    if "Lifestyle" not in raw:
        raw["Lifestyle"] = None

    # ---- assemble the field list (curated groups + auto disease group) ----
    groups = [g for g in CURATED_GROUPS]
    disease_cols = []
    for c in raw.columns:
        if not c.startswith("disease_"):
            continue
        present = raw[c].dropna().astype(str).str.strip()
        npos = int((present == "Yes").sum())
        if npos >= DISEASE_MIN_POS:
            disease_cols.append((c, npos))
    disease_cols.sort(key=lambda t: -t[1])
    disease_specs = [(c, DISEASE_LABELS.get(c, c.replace("disease_", "").replace("_", " ").capitalize()),
                      "proportion", {"true_value": "Yes"}) for c, _ in disease_cols]
    groups = groups + [("Specific diseases", disease_specs)]

    # flat field defs + global metadata (dist, domains, hist edges, cat colors)
    field_defs = []
    field_index = {}
    for gname, specs in groups:
        for (col, label, kind, opts) in specs:
            if col not in raw.columns:
                continue
            fd = {"key": col, "label": label, "group": gname, "kind": kind,
                  "fmt": opts.get("fmt", "num1")}
            if kind == "numeric":
                x = _transform(raw[col], opts).dropna().astype(float)
                dist = opts.get("dist") or _auto_dist(x)
                fd["dist"] = dist
                fd["hist"] = _global_bins(x, dist)  # {edges, log}
            elif kind == "categorical":
                cats = list(raw[col].dropna().astype(str).str.strip().value_counts().index)
                if col == "Lifestyle":
                    fd["catcolors"] = {LIFE_LABEL.get(c, c): LIFE_COLOR.get(LIFE_LABEL.get(c, c), "#9a9a94")
                                       for c in cats}
                elif opts.get("catcolors"):
                    # An explicit map, kept in its own order: the legend is
                    # rendered from catcolors, and the palette fallback orders
                    # by frequency, which would reshuffle the legend whenever
                    # the cohort changed. Categories absent from this country
                    # are dropped so the legend matches what is on screen.
                    fd["catcolors"] = {k: v for k, v in opts["catcolors"].items()
                                       if k in set(cats)}
                else:
                    fd["catcolors"] = {c: CAT_PALETTE[i % len(CAT_PALETTE)] for i, c in enumerate(cats)}
            elif kind == "proportion":
                fd["true_value"] = opts.get("true_value", "Yes")
            field_defs.append(fd)
            field_index[col] = fd

    # synthetic proportion derived from the Lifestyle composition (all non-Industrial)
    field_defs.append({"key": "pct_non_industrial", "label": "% Non-Industrial",
                       "group": "Lifestyle", "kind": "proportion", "fmt": "num1"})

    # ---- per-country aggregation ----
    countries = {}
    for cn, g in raw.groupby("Country_norm", dropna=True):
        cn = str(cn)
        if cn in ("", "nan", "None"):
            continue
        n = len(g)
        n_studies = int(g[STUDY_COL].nunique()) if STUDY_COL in g else 0
        n_pop = int(g["Population"].nunique()) if "Population" in g else 0
        life_raw = {k: int((g["Lifestyle"] == k).sum()) for k in LIFE_RAW3}
        life = {LIFE_LABEL[k]: v for k, v in life_raw.items()}
        dom = max(life, key=life.get) if any(life.values()) else None

        fields = {}
        for fd in field_defs:
            col, kind = fd["key"], fd["kind"]
            if col not in g.columns:
                continue
            if kind == "numeric":
                x = _transform(g[col], _opts_for(fd)).astype(float)
                miss = float(x.isna().mean())
                fields[col] = _numeric_stats(x.dropna(), fd) | {"missing": miss}
            elif kind == "proportion":
                s = g[col].dropna().astype(str).str.strip()
                miss = float(g[col].isna().mean())
                frac = float((s == fd["true_value"]).mean()) if s.size else None
                fields[col] = {"frac": frac, "n": int(s.size), "missing": miss}
            elif kind == "categorical":
                s = g[col].dropna().astype(str).str.strip()
                miss = float(g[col].isna().mean())
                relab = LIFE_LABEL if col == "Lifestyle" else {}
                counts = {}
                for k, v in s.value_counts().items():
                    kk = relab.get(k, k)
                    counts[kk] = counts.get(kk, 0) + int(v)
                fields[col] = {"counts": counts,
                               "dominant": (max(counts, key=counts.get) if counts else None),
                               "n": int(s.size), "missing": miss}

        n_ni = life_raw["Transitional"] + life_raw["H-G"]
        fields["pct_non_industrial"] = {"frac": (n_ni / n) if n else None, "n": n,
                                        "missing": float(g["Lifestyle"].isna().mean())}

        studies = []
        if STUDY_COL in g:
            for st, sg in g.groupby(STUDY_COL):
                slc = {k: int((sg["Lifestyle"] == k).sum()) for k in LIFE_RAW3}
                sdom = LIFE_LABEL[max(slc, key=slc.get)] if any(slc.values()) else None
                sf, smp = {}, {}     # sf: scalar per study (table);  smp: sample-level payload
                for fd in field_defs:
                    ck, kd = fd["key"], fd["kind"]
                    if ck == "pct_non_industrial":
                        nni = slc["Transitional"] + slc["H-G"]
                        sf[ck] = round(nni / len(sg), 4) if len(sg) else None
                        smp[ck] = [int(nni), int(len(sg))]           # [positive, present]
                        continue
                    if ck not in sg.columns:
                        continue
                    if kd == "numeric":
                        xx = _transform(sg[ck], _opts_for(fd)).dropna().astype(float)
                        edges = np.asarray(fd["hist"]["edges"])
                        if xx.size:
                            v = (float(np.exp(np.log(xx).mean()))
                                 if (fd.get("dist") == "lognormal" and (xx > 0).all()) else float(xx.mean()))
                            sf[ck] = round(v, 3)
                            smp[ck] = _hist_with_overflow(xx, edges)
                        else:
                            sf[ck] = None
                            smp[ck] = [0] * (len(edges) + 1)
                    elif kd == "proportion":
                        ss = sg[ck].dropna().astype(str).str.strip()
                        npos, npres = int((ss == fd["true_value"]).sum()), int(ss.size)
                        sf[ck] = (round(npos / npres, 4) if npres else None)
                        smp[ck] = [npos, npres]
                    elif kd == "categorical":
                        ss = sg[ck].dropna().astype(str).str.strip()
                        relab = LIFE_LABEL if ck == "Lifestyle" else {}
                        cnt = {}
                        for k2, v2 in ss.value_counts().items():
                            kk = relab.get(k2, k2)
                            cnt[kk] = cnt.get(kk, 0) + int(v2)
                        sf[ck] = (max(cnt, key=cnt.get) if cnt else None)
                        smp[ck] = cnt
                auth = (str(sg["Author"].dropna().iloc[0]) if "Author" in sg and sg["Author"].notna().any() else "")
                yr = (int(sg["Year"].dropna().iloc[0]) if "Year" in sg and sg["Year"].notna().any() else None)
                studies.append({"study": str(st), "author": auth, "year": yr,
                                "n": int(len(sg)), "lifestyle": sdom, "fields": sf, "smp": smp})
            studies.sort(key=lambda d: -d["n"])

        iso = str(g["Country"].dropna().iloc[0]) if "Country" in g and g["Country"].notna().any() else ""
        region = get_continent(cn) or "Other"
        countries[cn] = {"name": cn, "iso3": iso, "region": region, "n_samples": n,
                         "n_studies": n_studies, "n_populations": n_pop, "lifestyle_counts": life,
                         "dominant_lifestyle": dom, "fields": fields, "studies": studies}

    # ---- global domains for central & spread (per numeric/proportion field) ----
    for fd in field_defs:
        col, kind = fd["key"], fd["kind"]
        if kind == "numeric":
            cvals = [countries[c]["fields"][col].get("central") for c in countries]
            svals = [countries[c]["fields"][col].get("spread") for c in countries]
            fd["dom_central"] = _minmax([v for v in cvals if v is not None])
            fd["dom_spread"] = _minmax([v for v in svals if v is not None])
        elif kind == "proportion":
            fvals = [countries[c]["fields"][col].get("frac") for c in countries]
            fd["dom_central"] = _minmax([v for v in fvals if v is not None]) or [0.0, 1.0]

    overview = [
        {"key": "n_samples", "label": "Sample count", "group": "Overview", "kind": "count"},
        {"key": "n_studies", "label": "Study count", "group": "Overview", "kind": "count"},
        {"key": "n_populations", "label": "Population count", "group": "Overview", "kind": "count"},
        {"key": "dominant_lifestyle", "label": "Dominant lifestyle", "group": "Overview",
         "kind": "categorical", "catcolors": dict(LIFE_COLOR)},
    ]
    field_defs = overview + field_defs

    paths, w, h = _load_world_paths()
    unmatched = sorted(set(countries) - set(paths))
    n_sc = sum(len(c["studies"]) for c in countries.values())   # study×country combinations
    totals = {"n_samples": int(len(raw)),
              "n_studies": int(raw["Study"].nunique()) if "Study" in raw else 0,           # distinct studies
              "n_studyfolders": int(raw[STUDY_COL].nunique()) if STUDY_COL in raw else 0,   # author_year_country groupings
              "n_countries": len(countries),
              "n_study_country": int(n_sc)}
    life_legend = [{"label": lab, "color": col} for _, lab, col in LIFE_ORDER]
    # n_studies is the distinct-study count the manuscript reports, not the
    # StudyFolder count -- the two differ (one study can span several
    # author_year_country folders), and reading the folder count here put a
    # study total in the footer that no table in the paper agrees with.
    cohort = {"n_source": int(n_source), "n_shown": int(len(raw)),
              "dropped_lifestyle": dropped_life,
              "n_outside_cohort": int(n_outside),
              "cohort_file": _rel(cohort_used) if cohort_used else None,
              "n_studies": totals["n_studies"],
              "n_studyfolders": totals["n_studyfolders"]}
    cohort["health"] = health_stats
    return {"cohort": cohort,
            "fields": field_defs, "countries": countries, "paths": paths,
            "viewBox": [w, h], "totals": totals, "unmatched": unmatched,
            "lifestyle": life_legend}, unmatched


def _opts_for(fd):
    return {"transform": "gbp" if fd["key"] == "Total bases"
            else "millions" if fd["key"] == "number_reads" else None}


def _auto_dist(x):
    x = x.dropna().astype(float)
    if (x > 0).all() and len(x) > 10:
        if abs(np.log(x).skew()) < abs(x.skew()) and x.skew() > 1:
            return "lognormal"
    return "normal"


def _global_bins(x, dist, nbins=NBINS, qlo=0.01, qhi=0.99):
    """Inner bin edges over a robust 1-99% range; values outside go to overflow bins."""
    x = x.dropna().astype(float)
    if x.empty:
        return {"edges": [0, 1], "log": False}
    lo, hi = float(x.quantile(qlo)), float(x.quantile(qhi))
    if not (hi > lo):  # degenerate (few unique values) -> full range
        lo, hi = float(x.min()), float(x.max())
        if not (hi > lo):
            hi = lo + 1
    if dist == "lognormal" and lo > 0:
        edges = np.exp(np.linspace(np.log(lo), np.log(hi), nbins + 1))
        return {"edges": [float(e) for e in edges], "log": True}
    edges = np.linspace(lo, hi, nbins + 1)
    return {"edges": [float(e) for e in edges], "log": False}


def _hist_with_overflow(x, edges):
    """counts = [ <edges[0] , inner bins... , >edges[-1] ]  (length = len(edges)+1)."""
    inner = np.histogram(x, bins=edges)[0]
    under = int((x < edges[0]).sum())
    over = int((x > edges[-1]).sum())
    return [under] + [int(v) for v in inner] + [over]


def _numeric_stats(x, fd):
    x = x.astype(float)
    if x.empty:
        return {"n": 0, "central": None, "spread": None, "stats": None, "hist": [0] * NBINS}
    dist = fd["dist"]
    mean, sd = float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else 0.0
    med = float(x.median())
    q1, q3 = float(x.quantile(.25)), float(x.quantile(.75))
    st = {"n": int(x.size), "mean": mean, "sd": sd, "median": med,
          "q1": q1, "q3": q3, "min": float(x.min()), "max": float(x.max())}
    if dist == "lognormal" and (x > 0).all() and x.size > 1:
        lx = np.log(x)
        st["gmean"] = float(np.exp(lx.mean()))
        st["gsd"] = float(np.exp(lx.std(ddof=1)))
        central, spread = st["gmean"], st["gsd"]
    else:
        central, spread = mean, sd
    edges = np.asarray(fd["hist"]["edges"])
    return {"n": int(x.size), "central": central, "spread": spread,
            "stats": st, "hist": _hist_with_overflow(x, edges)}


def _minmax(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    return [float(lo), float(hi if hi > lo else lo + 1)]


# ── HTML template ─────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Metadata Explorer — Microbiome Extinctions</title>
<style>
:root{
  --surface:#ffffff; --plane:#f4f5f3; --ink:#111; --ink2:#555; --muted:#8a8a84;
  --grid:#e5e5df; --hair:rgba(0,0,0,0.12); --ocean:#f0f3f6; --nodata:#e9e9e3;
  --stroke:#b9b9b2; --stroke-nd:#cfcfc8;
  color-scheme:light;
}
:root[data-theme="dark"]{
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7; --muted:#8a8a84;
  --grid:#2c2c2a; --hair:rgba(255,255,255,0.13); --ocean:#191d22; --nodata:#2a2a28;
  --stroke:#3a3a37; --stroke-nd:#333330; color-scheme:dark;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:14px;line-height:1.45}
.wrap{max-width:1200px;margin:0 auto;padding:22px 20px 40px}
header h1{font-size:21px;margin:0 0 3px;font-weight:650}
header p{margin:0;color:var(--ink2);font-size:13px;max-width:70ch}
.chips{display:flex;gap:10px;margin:14px 0 4px;flex-wrap:wrap}
.chip{background:var(--surface);border:1px solid var(--hair);border-radius:9px;padding:7px 12px;min-width:92px}
.chip b{display:block;font-size:19px;font-weight:650;font-variant-numeric:tabular-nums}
.chip span{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.controls{display:flex;gap:14px;align-items:flex-end;flex-wrap:wrap;margin:16px 0 10px}
.ctl label{display:block;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}
select{font:inherit;background:var(--surface);color:var(--ink);border:1px solid var(--hair);border-radius:8px;padding:7px 9px}
#group{min-width:170px} #field{min-width:230px}
.seg{display:inline-flex;border:1px solid var(--hair);border-radius:8px;overflow:hidden}
.seg button{font:inherit;background:var(--surface);color:var(--ink2);border:0;padding:7px 13px;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--surface)}
.seg button:disabled{opacity:.35;cursor:not-allowed}
.spacer{flex:1}
.themebtn{font:inherit;background:var(--surface);color:var(--ink2);border:1px solid var(--hair);border-radius:8px;padding:7px 11px;cursor:pointer}
.panel{background:var(--surface);border:1px solid var(--hair);border-radius:12px;padding:8px;position:relative;overflow:hidden}
svg.map{display:block;width:100%;height:auto;background:var(--ocean);border-radius:8px}
svg.map path.country{stroke:var(--stroke);stroke-width:.5;transition:fill .16s,opacity .12s}
svg.map path.country.nodata{fill:var(--nodata);stroke:var(--stroke-nd)}
svg.map path.country:hover{opacity:.85;stroke:var(--ink);stroke-width:1.2;cursor:pointer}
.legend{display:flex;align-items:center;gap:12px;margin:10px 4px 2px;flex-wrap:wrap}
.legbar{height:12px;width:230px;border-radius:6px;border:1px solid var(--hair)}
.legend .lab{font-size:12px;color:var(--ink2);font-variant-numeric:tabular-nums}
.legend .title{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.swatch{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--ink2)}
.swatch i{width:13px;height:13px;border-radius:3px;display:inline-block;border:1px solid var(--hair)}
.tip{position:fixed;pointer-events:none;z-index:20;background:var(--surface);border:1px solid var(--hair);
  border-radius:10px;padding:10px 12px;max-width:300px;box-shadow:0 8px 26px rgba(0,0,0,.18);opacity:0;transition:opacity .1s;font-size:12.5px}
.tip h4{margin:0 0 5px;font-size:14px}
.tip .row{display:flex;justify-content:space-between;gap:14px;color:var(--ink2)}
.tip .row b{color:var(--ink);font-variant-numeric:tabular-nums}
.tip .bars{display:flex;height:6px;border-radius:3px;overflow:hidden;margin:6px 0 3px}
.hist{display:flex;align-items:flex-end;gap:1px;height:34px;margin:7px 0 2px}
.hist i{flex:1;background:var(--histbar,#21918c);border-radius:1px;min-height:1px;opacity:.9}
.tip .studies{margin-top:7px;padding-top:6px;border-top:1px solid var(--hair);color:var(--muted);font-size:11.5px}
.tbl-wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--grid);white-space:nowrap}
th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:var(--surface)}
th{color:var(--ink2);cursor:pointer;user-select:none;font-weight:600}
th:hover{color:var(--ink)}
td{font-variant-numeric:tabular-nums;color:var(--ink2)}
td.name{color:var(--ink);font-variant-numeric:normal}
.dot{width:9px;height:9px;border-radius:2px;display:inline-block;margin-right:6px;vertical-align:-1px;border:1px solid var(--hair)}
footer{color:var(--muted);font-size:11.5px;margin-top:14px}
svg.map path.country.sel{stroke:var(--ink);stroke-width:1.6}
.detail{margin-top:12px;padding:12px 14px}
.detail-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap}
.detail-head h3{margin:0 0 2px;font-size:17px}
.detail .mini{color:var(--ink2);font-size:12.5px;margin:0 0 6px}
.detail .bars{max-width:300px;height:8px}
.detail .lifeleg{display:flex;gap:14px;flex-wrap:wrap;margin-top:7px}
.detail-body{display:grid;grid-template-columns:minmax(210px,1fr) 2fr;gap:20px;margin-top:14px}
@media(max-width:720px){.detail-body{grid-template-columns:1fr}}
.subttl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:7px}
.detail-studies table{font-size:12.5px}
.detail-studies th,.detail-studies td{padding:6px 9px}
header h1{cursor:pointer}
.studies-controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:2px 4px 12px}
.studies-controls select{min-width:132px}
.muted-lab{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.colpick{position:relative}
.colmenu{position:absolute;top:38px;left:0;z-index:15;background:var(--surface);border:1px solid var(--hair);
  border-radius:10px;padding:10px 12px;max-height:360px;overflow:auto;box-shadow:0 8px 26px rgba(0,0,0,.18);width:264px}
.colmenu .colgrp{margin-bottom:9px}
.colmenu .colgrp-h{font-size:10.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;
  margin-bottom:3px;display:flex;justify-content:space-between;align-items:center}
.colmenu .grptog{color:#2a78d6;cursor:pointer;text-transform:none;letter-spacing:0;font-size:11px}
.colmenu label{display:flex;gap:7px;align-items:center;font-size:12.5px;padding:2px 0;cursor:pointer}
.charts-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:16px;padding:6px 2px}
.chartcard{border:1px solid var(--hair);border-radius:10px;padding:10px 12px}
.chart{margin:6px 0 2px}
.chart-yl{font-size:10px;color:var(--muted);margin-bottom:2px}
.chart-plot{display:flex;gap:6px;align-items:stretch;height:66px}
.y-ax{display:flex;flex-direction:column;justify-content:space-between;font-size:10px;color:var(--muted);text-align:right;min-width:26px;font-variant-numeric:tabular-nums}
.bars2{display:flex;align-items:flex-end;gap:1px;flex:1;border-left:1px solid var(--grid);border-bottom:1px solid var(--grid);padding:0 1px}
.bars2 .hb{flex:1;background:#2c728e;border-radius:1px 1px 0 0;min-height:1px}
.bars2 .hbof{background:var(--muted);opacity:.85}
.bars2 .hb:hover{opacity:.55;cursor:crosshair}
.x-ax{display:flex;justify-content:space-between;font-size:9.5px;color:var(--muted);margin:3px 0 0 32px;font-variant-numeric:tabular-nums}
.x-title{font-size:11px;color:var(--ink2);text-align:center;margin-top:3px;margin-left:32px}
.barchart{display:flex;flex-direction:column;gap:4px;margin:6px 0}
.bcrow{display:grid;grid-template-columns:96px 1fr 34px;align-items:center;gap:8px;font-size:12px}
.bclab{color:var(--ink2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bcbar{height:12px;border-radius:3px;min-width:2px;border:1px solid var(--hair)}
.bcval{color:var(--muted);text-align:right;font-variant-numeric:tabular-nums}
.prov{margin-top:.6rem;font-size:.82rem;line-height:1.55;opacity:.72;
  border-left:3px solid var(--edge,#cbd2d9);padding-left:.7rem}
.prov code{font-size:.8rem}
.hidden{display:none}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Study metadata across countries</h1>
    <p>Distributions and data completeness for the human gut metagenome dataset, aggregated by
       country and study. Pick a variable group and variable; colour by central value, spread
       (SD, or geometric SD for lognormal variables), or missingness. Hover a country for its
       distribution; click to pin &amp; zoom. The <b>Studies</b> view is a filterable study × country
       table; the title or <b>Home</b> resets the view.</p>
    <p class="prov">__PROVENANCE__</p>
  </header>
  <div class="chips" id="chips"></div>
  <div class="controls">
    <div class="ctl"><label for="group">Variable group</label><select id="group"></select></div>
    <div class="ctl"><label for="field">Variable</label><select id="field"></select></div>
    <div class="ctl"><label>Colour by</label>
      <div class="seg" id="metric">
        <button data-v="central" aria-pressed="true">Central</button>
        <button data-v="spread">Spread</button>
        <button data-v="missing">Missing</button>
      </div>
    </div>
    <div class="ctl"><label>View</label>
      <div class="seg" id="view">
        <button data-v="map" aria-pressed="true">Map</button>
        <button data-v="country">Country</button>
        <button data-v="studies">Studies</button>
      </div>
    </div>
    <div class="spacer"></div>
    <button class="themebtn" id="home">⌂ Home</button>
    <button class="themebtn" id="theme">◐ Theme</button>
  </div>
  <div class="panel" id="mapPanel">
    <svg class="map" id="map" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="legend" id="legend"></div>
  </div>
  <div class="panel tbl-wrap hidden" id="tablePanel"><table id="tbl"><thead></thead><tbody></tbody></table></div>
  <div class="panel studies hidden" id="studiesPanel">
    <div class="studies-controls">
      <select id="fRegion" title="Filter by region"></select>
      <select id="fLife" title="Filter by lifestyle"></select>
      <select id="fCountry" title="Filter by country"></select>
      <div class="colpick"><button class="themebtn" id="colBtn">Columns ▾</button>
        <div class="colmenu hidden" id="colMenu"></div></div>
      <select id="sortMode" title="Row sort order">
        <option value="AYC">Sort: Author, Year, Country</option>
        <option value="CAY">Sort: Country, Author, Year</option>
        <option value="col">Sort: by column ▾</option>
      </select>
      <div class="seg" id="studyMode"><button data-v="table" aria-pressed="true">Table</button><button data-v="charts">Charts</button></div>
      <div class="seg" id="chartUnit"><button data-v="sample" aria-pressed="true">Per sample</button><button data-v="study">Per study</button></div>
      <span class="spacer"></span>
      <span id="studyCount" class="muted-lab"></span>
    </div>
    <div class="tbl-wrap" id="studyTableWrap"><table id="stbl"><thead></thead><tbody></tbody></table></div>
    <div id="studyCharts" class="charts-grid hidden"></div>
  </div>
  <div class="panel detail hidden" id="detail"></div>
  <footer id="foot"></footer>
</div>
<div class="tip" id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
// perceptually-uniform, CVD-safe ramps
const VIRIDIS = ['#440154','#472d7b','#3b528b','#2c728e','#21918c','#28ae80','#5ec962','#addc30','#fde725'];
const MAGMA   = ['#fcfdbf','#fed799','#fdac7b','#f77f6f','#e75263','#c43c75','#9c2e7f','#6b1d81','#3b0f70','#000004'];
const byKey = k => DATA.fields.find(f=>f.key===k);
const LIFECOL = {}; (DATA.lifestyle||[]).forEach(l=>{LIFECOL[l.label]=l.color;});
const state = {group:'Sequencing effort', field:'Total bases', metric:'central', view:'map', selected:null,
  sort:{key:'n_samples',dir:-1},
  studyFilters:{region:'',lifestyle:'',country:''},
  studyCols:['Total bases','median_read_length','Age','BMI','Gender','Unhealthy','pct_non_industrial'],
  studySort:{key:'n',dir:-1}, studySortMode:'AYC', studyMode:'table', chartUnit:'sample'};

const fmt = {
  int:  v=>v==null?'—':Math.round(v).toLocaleString(),
  num1: v=>v==null?'—':(+v).toLocaleString(undefined,{maximumFractionDigits:1}),
  num2: v=>v==null?'—':(+v).toLocaleString(undefined,{maximumFractionDigits:2}),
  pct:  v=>v==null?'—':(100*v).toFixed(0)+'%',
};
function hex2rgb(h){return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)];}
function lerp(a,b,t){return Math.round(a+(b-a)*t);}
function rampColor(t, ramp){
  t=Math.max(0,Math.min(1,t)); const x=t*(ramp.length-1); const i=Math.floor(x), f=x-i;
  if(i>=ramp.length-1) return ramp[ramp.length-1];
  const a=hex2rgb(ramp[i]), b=hex2rgb(ramp[i+1]);
  return `rgb(${lerp(a[0],b[0],f)},${lerp(a[1],b[1],f)},${lerp(a[2],b[2],f)})`;
}

// ── metric value + colour ──
function metricVal(c,f){
  const fl=c.fields[f.key];
  if(state.metric==='missing') return fl? fl.missing : null;
  if(f.kind==='count') return c[f.key];
  if(f.kind==='numeric') return state.metric==='spread'? fl.spread : fl.central;
  if(f.kind==='proportion') return state.metric==='spread'? null : fl.frac;
  return null; // categorical handled via colour map
}
function domainFor(f){
  if(state.metric==='missing') return {lo:0,hi:1,log:false};
  if(f.kind==='count'){const v=Object.values(DATA.countries).map(c=>c[f.key]); return {lo:Math.min(...v),hi:Math.max(...v),log:true};}
  if(f.kind==='numeric'){const d=state.metric==='spread'?f.dom_spread:f.dom_central;
     const log=(f.dist==='lognormal'&&d&&d[0]>0); return {lo:d?d[0]:0,hi:d?d[1]:1,log};}
  if(f.kind==='proportion'){return {lo:0,hi:Math.max(0.001,(f.dom_central||[0,1])[1]),log:false};}
  return {lo:0,hi:1,log:false};
}
function fillFor(c,f,dom){
  if(f.kind==='categorical' && state.metric!=='missing'){
    const fl=c.fields? c.fields[f.key] : null;
    const cat = f.key==='dominant_lifestyle'? c.dominant_lifestyle : (fl?fl.dominant:null);
    return (cat && f.catcolors[cat]) ? f.catcolors[cat] : null;
  }
  const v=metricVal(c,f);
  if(v==null||!isFinite(v)) return null;
  const ramp = state.metric==='missing'?MAGMA:VIRIDIS;
  let t = dom.log ? (Math.log(Math.max(v,dom.lo))-Math.log(dom.lo))/(Math.log(dom.hi)-Math.log(dom.lo))
                  : (v-dom.lo)/(dom.hi-dom.lo);
  return rampColor(t, ramp);
}

// ── menus ──
function buildGroupMenu(){
  const gsel=document.getElementById('group');
  const groups=[...new Set(DATA.fields.map(f=>f.group))];
  gsel.innerHTML=groups.map(g=>`<option>${g}</option>`).join('');
  gsel.value=state.group;
  gsel.onchange=()=>{state.group=gsel.value; const first=DATA.fields.find(f=>f.group===state.group); state.field=first.key; buildFieldMenu(); syncMetric(); render();};
  buildFieldMenu();
}
function buildFieldMenu(){
  const fsel=document.getElementById('field');
  const fs=DATA.fields.filter(f=>f.group===state.group);
  fsel.innerHTML=fs.map(f=>`<option value="${f.key}">${f.label}</option>`).join('');
  fsel.value=state.field;
  fsel.onchange=()=>{state.field=fsel.value; syncMetric(); render();};
}
function syncMetric(){
  const f=byKey(state.field);
  const av={central:true, spread:f.kind==='numeric', missing:!(f.kind==='count'||f.key==='dominant_lifestyle')};
  document.querySelectorAll('#metric button').forEach(b=>{b.disabled=!av[b.dataset.v];});
  if(!av[state.metric]) { state.metric='central'; }
  setSeg('metric',state.metric);
}
function setSeg(id,v){document.querySelectorAll('#'+id+' button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.v===v));}
function wireSeg(id,key,cb){document.querySelectorAll('#'+id+' button').forEach(b=>b.onclick=()=>{if(b.disabled)return;state[key]=b.dataset.v;setSeg(id,b.dataset.v);cb&&cb();render();});}

function renderChips(){const t=DATA.totals;document.getElementById('chips').innerHTML=
  [['Samples',t.n_samples,''],['Studies',t.n_studies,t.n_studyfolders+' study×cohort folders'],
   ['Countries',t.n_countries,''],['Study × country',t.n_study_country,'']]
  .map(([l,v,sub])=>`<div class="chip" ${sub?`title="${sub}"`:''}><b>${v.toLocaleString()}</b><span>${l}</span></div>`).join('');}

// ── map ──
// categorical composition -> proportional diagonal hatching (stripe width ∝ group size)
function composition(c,f){
  const cts = f.key==='dominant_lifestyle'? c.lifestyle_counts : (c.fields&&c.fields[f.key]?c.fields[f.key].counts:null);
  if(!cts) return null;
  const es=Object.entries(cts).filter(([,v])=>v>0); const tot=es.reduce((a,[,v])=>a+v,0);
  if(!tot) return null; es.sort((a,b)=>b[1]-a[1]);
  return es.map(([k,v])=>[f.catcolors[k]||'#c9c9c2', v/tot]);
}
function makePattern(id,comp){
  const W=7; let x=0,r='';
  for(const [color,frac] of comp){ const w=Math.max(.001,frac*W);
    r+=`<rect x="${x.toFixed(3)}" width="${w.toFixed(3)}" height="${W}" fill="${color}"/>`; x+=w; }
  return `<pattern id="${id}" width="${W}" height="${W}" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">${r}</pattern>`;
}
function renderMap(){
  const [w,h]=DATA.viewBox, svg=document.getElementById('map'); svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
  const f=byKey(state.field), dom=domainFor(f);
  const cat = f.kind==='categorical' && state.metric!=='missing';
  let defs='',s='',pid=0;
  for(const [name,d] of Object.entries(DATA.paths)){
    const c=DATA.countries[name]; let fill=null;
    if(c){
      if(cat){ const comp=composition(c,f); if(comp){ const id='pat'+(pid++); defs+=makePattern(id,comp); fill=`url(#${id})`; } }
      else fill=fillFor(c,f,dom);
    }
    s+=`<path class="country${fill?'':' nodata'}" d="${d}"${fill?` fill="${fill}"`:''} data-c="${encodeURIComponent(name)}"></path>`;
  }
  svg.innerHTML=(defs?`<defs>${defs}</defs>`:'')+s;
  svg.querySelectorAll('path.country').forEach(p=>{
    const nm=decodeURIComponent(p.dataset.c);
    if(nm===state.selected) p.classList.add('sel');
    p.addEventListener('mousemove',ev=>showTip(ev,nm));
    p.addEventListener('mouseleave',hideTip);
    p.addEventListener('click',()=>selectCountry(nm));
  });
  if(state.selected && DATA.countries[state.selected]) applyZoom(state.selected);
  renderLegend(f,dom);
}
function labVal(f,v){return f.kind==='proportion'?fmt.pct(v):(fmt[f.fmt]||fmt.num1)(v);}
function renderLegend(f,dom){
  const el=document.getElementById('legend');
  if(f.kind==='categorical' && state.metric!=='missing'){
    el.innerHTML=`<span class="title">${f.label} · diagonal hatch width ∝ share</span>`+
      Object.entries(f.catcolors).map(([k,c])=>`<span class="swatch"><i style="background:${c}"></i>${k}</span>`).join('')+
      `<span class="swatch"><i style="background:var(--nodata)"></i>no data</span>`; return;
  }
  const ramp=state.metric==='missing'?MAGMA:VIRIDIS;
  const grad=`linear-gradient(90deg,${ramp.join(',')})`;
  const lo=state.metric==='missing'?'0%':labVal(f,dom.lo), hi=state.metric==='missing'?'100%':labVal(f,dom.hi);
  const title=state.metric==='missing'?`Missing: ${f.label}`:(state.metric==='spread'?`Spread: ${f.label}`:f.label);
  el.innerHTML=`<span class="title">${title}</span><span class="lab">${lo}</span>`+
    `<div class="legbar" style="background:${grad}"></div><span class="lab">${hi}</span>`+
    `<span class="swatch"><i style="background:var(--nodata)"></i>no data</span>`;
}

// ── tooltip ──
function row(k,v){return `<div class="row"><span>${k}</span><b>${v}</b></div>`;}
function histSvg(counts, edges, o){
  const nIn=edges.length-1, hasOF=counts.length===nIn+2, F=fmt[o.fmt]||fmt.num1;
  const mx=Math.max(1,Math.max.apply(null,counts));
  const bar=(c,r,of)=>`<i class="hb${of?' hbof':''}" style="height:${(100*c/mx).toFixed(1)}%" data-r="${r}" data-c="${c}"></i>`;
  let bars='';
  if(hasOF) bars+=bar(counts[0],'&lt; '+F(edges[0]),true);
  for(let i=0;i<nIn;i++) bars+=bar(counts[hasOF?i+1:i], F(edges[i])+' – '+F(edges[i+1]), false);
  if(hasOF) bars+=bar(counts[nIn+1],'&gt; '+F(edges[nIn]),true);
  const ticks=[0,Math.round(nIn/4),Math.round(nIn/2),Math.round(3*nIn/4),nIn].map(i=>`<span>${F(edges[Math.min(i,nIn)])}</span>`).join('');
  return `<div class="chart"><div class="chart-yl">${o.ylab||'count'}</div><div class="chart-plot">`+
    `<div class="y-ax"><span>${mx.toLocaleString()}</span><span>0</span></div><div class="bars2">${bars}</div></div>`+
    `<div class="x-ax">${ticks}</div><div class="x-title">${o.label}${o.log?' · log scale':''}${hasOF?' · outer bars = tails':''}</div></div>`;
}
function histChart(fl,f){
  if(!fl||!fl.hist||!fl.hist.some(v=>v>0)) return '';
  return histSvg(fl.hist, f.hist.edges, {label:f.label, fmt:f.fmt, log:f.hist.log, ylab:'samples'});
}
function histFromValues(vals,f){
  vals=vals.filter(v=>v!=null&&isFinite(v)); if(!vals.length) return '<div class="muted-lab">no data</div>';
  const log=(f.dist==='lognormal'&&vals.every(v=>v>0));
  const sorted=[...vals].sort((a,b)=>a-b), q=p=>sorted[Math.min(sorted.length-1,Math.max(0,Math.floor(p*(sorted.length-1))))];
  let lo=q(0.01),hi=q(0.99); if(!(hi>lo)){lo=sorted[0];hi=sorted[sorted.length-1];} if(!(hi>lo))hi=lo+(Math.abs(lo)||1)*0.01;
  const nb=Math.min(14,Math.max(6,Math.round(Math.sqrt(vals.length))));
  const edges=[]; for(let i=0;i<=nb;i++) edges.push(log?Math.exp(Math.log(lo)+(Math.log(hi)-Math.log(lo))*i/nb):lo+(hi-lo)*i/nb);
  const counts=new Array(nb+2).fill(0);
  vals.forEach(v=>{ if(v<lo){counts[0]++;return;} if(v>hi){counts[nb+1]++;return;}
    const t=log?(Math.log(v)-Math.log(lo))/(Math.log(hi)-Math.log(lo)):(v-lo)/(hi-lo);
    counts[1+Math.max(0,Math.min(nb-1,Math.floor(t*nb)))]++;});
  return histSvg(counts,edges,{label:f.label,fmt:f.kind==='proportion'?'pct':f.fmt,log,ylab:'studies'});
}
function barChart(counts,f){
  const es=Object.entries(counts).sort((a,b)=>b[1]-a[1]); const mx=Math.max(1,...es.map(e=>e[1]));
  return '<div class="barchart">'+es.map(([k,v])=>`<div class="bcrow"><span class="bclab">${k}</span>`+
    `<span class="bcbar" style="width:${(100*v/mx).toFixed(0)}%;background:${(f.catcolors&&f.catcolors[k])||'#2c728e'}"></span>`+
    `<span class="bcval">${v}</span></div>`).join('')+'</div>';
}
function showBinTip(ev,r,c){const tip=document.getElementById('tip');
  tip.innerHTML=`<div class="row"><span>Range</span><b>${r}</b></div><div class="row"><span>Count</span><b>${(+c).toLocaleString()}</b></div>`;
  tip.style.opacity=1; const pad=14; let x=ev.clientX+pad,y=ev.clientY+pad;
  if(x+tip.offsetWidth>innerWidth)x=ev.clientX-tip.offsetWidth-pad; if(y+tip.offsetHeight>innerHeight)y=ev.clientY-tip.offsetHeight-pad;
  tip.style.left=x+'px';tip.style.top=y+'px';}
function wireHist(container){container.querySelectorAll('.hb').forEach(b=>{
  b.addEventListener('mousemove',ev=>{ev.stopPropagation();showBinTip(ev,b.dataset.r,b.dataset.c);});
  b.addEventListener('mouseleave',hideTip);});}
function lifeBar(c){const lc=c.lifestyle_counts;
  return '<div class="bars">'+DATA.lifestyle.map(l=>`<span style="flex:${lc[l.label]||0};background:${l.color}"></span>`).join('')+'</div>';}
function fieldSummaryHTML(c,f,fl){
  let body='';
  if(f.kind==='numeric' && fl && fl.stats){
    const S=fl.stats, F=fmt[f.fmt]||fmt.num1;
    if(f.dist==='lognormal' && S.gmean!=null){
      body=row('Geo. mean',F(S.gmean))+row('Geo. SD (×/÷)',fmt.num2(S.gsd))+row('Median',F(S.median))+row('IQR',`${F(S.q1)} – ${F(S.q3)}`);
    }else{
      body=row('Mean ± SD',`${F(S.mean)} ± ${F(S.sd)}`)+row('Median',F(S.median))+row('IQR',`${F(S.q1)} – ${F(S.q3)}`);
    }
    body+=row('Range',`${F(S.min)} – ${F(S.max)}`)+row('n',S.n.toLocaleString())+histChart(fl,f);
  }else if(f.kind==='proportion'){
    body=row(f.label,fl&&fl.frac!=null?fmt.pct(fl.frac):'—')+row('n with data',(fl?fl.n:0).toLocaleString());
  }else if(f.kind==='categorical'){
    const dom=f.key==='dominant_lifestyle'?c.dominant_lifestyle:(fl?fl.dominant:null);
    body=row('Dominant',dom||'—');
    const cts=f.key==='dominant_lifestyle'?c.lifestyle_counts:(fl?fl.counts:{});
    body+=Object.entries(cts).filter(([k,v])=>v>0).slice(0,6).map(([k,v])=>row(k,v.toLocaleString())).join('');
  }else{ body=row(f.label,(c[f.key]||0).toLocaleString()); }
  if(fl && f.kind!=='count'){ body+=row('Missing',fmt.pct(fl.missing)); }
  return body;
}
function showTip(ev,name){
  const c=DATA.countries[name], tip=document.getElementById('tip'); if(!c){tip.style.opacity=0;return;}
  const f=byKey(state.field), fl=c.fields?c.fields[f.key]:null;
  tip.innerHTML=`<h4>${c.name} <span style="color:var(--muted);font-weight:400">${c.iso3}</span></h4>`+
    row('Samples',c.n_samples.toLocaleString())+row('Studies',c.n_studies)+lifeBar(c)+
    `<div style="margin-top:6px">${fieldSummaryHTML(c,f,fl)}</div>`+
    `<div class="studies">${state.selected===name?'Selected — see panel below':'Click to pin &amp; zoom · '+c.studies.length+' studies'}</div>`;
  tip.style.opacity=1;
  const pad=14,tw=tip.offsetWidth,th=tip.offsetHeight; let x=ev.clientX+pad,y=ev.clientY+pad;
  if(x+tw>innerWidth)x=ev.clientX-tw-pad; if(y+th>innerHeight)y=ev.clientY-th-pad;
  tip.style.left=x+'px'; tip.style.top=y+'px';
}
// ── selection / zoom / pinned detail ──
function pathBBox(d){const n=d.match(/-?\d+(?:\.\d+)?/g).map(Number);
  let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
  for(let i=0;i+1<n.length;i+=2){x0=Math.min(x0,n[i]);x1=Math.max(x1,n[i]);y0=Math.min(y0,n[i+1]);y1=Math.max(y1,n[i+1]);}
  return [x0,y0,x1,y1];}
function applyZoom(name){const d=DATA.paths[name];if(!d)return;
  let[x0,y0,x1,y1]=pathBBox(d);const pad=Math.max(x1-x0,y1-y0)*0.45+12;x0-=pad;y0-=pad;x1+=pad;y1+=pad;
  document.getElementById('map').setAttribute('viewBox',`${x0.toFixed(1)} ${y0.toFixed(1)} ${(x1-x0).toFixed(1)} ${(y1-y0).toFixed(1)}`);}
function selectCountry(name){
  if(!DATA.countries[name]) return;
  state.selected = (state.selected===name? null : name);
  if(state.selected){ state.view='map'; setSeg('view','map'); }
  hideTip(); render();
  if(state.selected) document.getElementById('detail').scrollIntoView({behavior:'smooth',block:'nearest'});
}
function studyColLabel(f){ return f.kind==='numeric'?(f.dist==='lognormal'?'Geo. mean':'Mean')
  : f.kind==='proportion'?f.label : f.kind==='categorical'?'Dominant' : f.label; }
function studyVal(f,s){ const v=s.fields?s.fields[f.key]:null;
  if(f.kind==='proportion') return v==null?'—':fmt.pct(v);
  if(f.kind==='numeric') return (fmt[f.fmt]||fmt.num1)(v);
  if(f.kind==='categorical') return v||'—';
  return '—'; }
function renderDetail(){
  const el=document.getElementById('detail');
  if(!state.selected || !DATA.countries[state.selected]){ el.classList.add('hidden'); el.innerHTML=''; return; }
  el.classList.remove('hidden');
  const c=DATA.countries[state.selected], f=byKey(state.field), fl=c.fields?c.fields[f.key]:null;
  const lifeleg=DATA.lifestyle.map(l=>`<span class="swatch"><i style="background:${l.color}"></i>${l.label} (${(c.lifestyle_counts[l.label]||0).toLocaleString()})</span>`).join('');
  const rows=c.studies.map(s=>`<tr><td class="name">${s.study}</td><td>${s.n.toLocaleString()}</td>`+
    `<td><span class="dot" style="background:${LIFECOL[s.lifestyle]||'var(--nodata)'}"></span>${s.lifestyle||'—'}</td>`+
    `<td>${studyVal(f,s)}</td></tr>`).join('');
  el.innerHTML=`<div class="detail-head"><div><h3>${c.name} <span style="color:var(--muted);font-weight:400">${c.iso3}</span></h3>`+
    `<div class="mini">${c.n_samples.toLocaleString()} samples · ${c.n_studies} studies · ${c.n_populations} populations</div>`+
    lifeBar(c)+`<div class="lifeleg">${lifeleg}</div></div>`+
    `<button class="themebtn" id="detailClose">✕ close</button></div>`+
    `<div class="detail-body"><div><div class="subttl">${f.label}${state.metric==='missing'?' — missingness':''}</div>${fieldSummaryHTML(c,f,fl)}</div>`+
    `<div class="detail-studies"><div class="subttl">Breakdown by study (${c.studies.length})</div><div class="tbl-wrap">`+
    `<table><thead><tr><th>Study</th><th>n</th><th>Lifestyle</th><th>${studyColLabel(f)}</th></tr></thead><tbody>${rows}</tbody></table></div></div></div>`;
  document.getElementById('detailClose').onclick=()=>selectCountry(state.selected);
  wireHist(el);
}
function hideTip(){document.getElementById('tip').style.opacity=0;}

// ── table ──
function renderTable(){
  const f=byKey(state.field), F=fmt[f.fmt]||fmt.num1;
  const cols=[{k:'name',t:'Country',get:c=>c.name},
    {k:'n_samples',t:'Samples',get:c=>c.n_samples},
    {k:'n_studies',t:'Studies',get:c=>c.n_studies},
    {k:'lifestyle',t:'Dominant',get:c=>c.dominant_lifestyle||'—'}];
  if(f.kind==='numeric'){
    cols.push({k:'central',t:f.dist==='lognormal'?'Geo. mean':'Mean',get:c=>c.fields[f.key].central},
      {k:'spread',t:f.dist==='lognormal'?'Geo. SD':'SD',get:c=>c.fields[f.key].spread},
      {k:'median',t:'Median',get:c=>c.fields[f.key].stats?.median});
  }else if(f.kind==='proportion'){ cols.push({k:'frac',t:f.label,get:c=>c.fields[f.key].frac}); }
  else if(f.kind==='count'){ cols.push({k:'val',t:f.label,get:c=>c[f.key]}); }
  if(f.kind!=='count') cols.push({k:'missing',t:'% missing',get:c=>c.fields?c.fields[f.key]?.missing:null});

  const rows=Object.values(DATA.countries), sk=state.sort.key, dir=state.sort.dir;
  const sc=cols.find(c=>c.k===sk)||cols[1];
  rows.sort((a,b)=>{let va=sc.get(a),vb=sc.get(b);
    if(typeof va==='string'||typeof vb==='string'){return dir*String(va).localeCompare(String(vb));}
    va=(va==null||!isFinite(va))?-Infinity:va; vb=(vb==null||!isFinite(vb))?-Infinity:vb; return dir*(va-vb);});
  const disp=(col,c)=>{const v=col.get(c);
    if(col.k==='name')return `<span class="dot" style="background:${LIFECOL[c.dominant_lifestyle]||'var(--nodata)'}"></span>${v}`;
    if(col.k==='missing'||col.k==='frac')return fmt.pct(v);
    if(col.k==='central'||col.k==='median'||col.k==='val')return (fmt[f.fmt]||fmt.num1)(v);
    if(col.k==='spread')return f.dist==='lognormal'?fmt.num2(v):(fmt[f.fmt]||fmt.num1)(v);
    if(typeof v==='number')return v.toLocaleString(); return v??'—';};
  const thead=document.querySelector('#tbl thead');
  thead.innerHTML='<tr>'+cols.map(c=>`<th data-k="${c.k}">${c.t}${sk===c.k?(dir<0?' ▾':' ▴'):''}</th>`).join('')+'</tr>';
  thead.querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;
    state.sort=state.sort.key===k?{key:k,dir:-state.sort.dir}:{key:k,dir:(k==='name'?1:-1)}; renderTable();});
  document.querySelector('#tbl tbody').innerHTML=rows.map(c=>'<tr>'+cols.map(col=>`<td class="${col.k==='name'?'name':''}">${disp(col,c)}</td>`).join('')+'</tr>').join('');
}

// ── study × country table (Table S1-plus) ──
const STUDY_ROWS = (()=>{const rows=[];
  for(const [name,c] of Object.entries(DATA.countries)){
    for(const s of c.studies){ rows.push({country:name,iso:c.iso3,region:c.region||'Other',study:s.study,
      author:s.author||'',year:s.year,n:s.n,lifestyle:s.lifestyle,fields:s.fields,smp:s.smp}); }
  } return rows;})();
function colFields(){return DATA.fields.filter(f=>f.kind!=='count'&&f.key!=='dominant_lifestyle');}
function populateStudyFilters(){
  const uniq=k=>[...new Set(STUDY_ROWS.map(r=>r[k]))].filter(Boolean).sort();
  const O=(v,cur)=>`<option value="${v}" ${v===cur?'selected':''}>${v}</option>`;
  const R=document.getElementById('fRegion'),L=document.getElementById('fLife'),C=document.getElementById('fCountry');
  R.innerHTML=`<option value="">All regions</option>`+uniq('region').map(v=>O(v,state.studyFilters.region)).join('');
  L.innerHTML=`<option value="">All lifestyles</option>`+DATA.lifestyle.map(l=>O(l.label,state.studyFilters.lifestyle)).join('');
  C.innerHTML=`<option value="">All countries</option>`+uniq('country').map(v=>O(v,state.studyFilters.country)).join('');
  R.onchange=e=>{state.studyFilters.region=e.target.value;renderStudies();};
  L.onchange=e=>{state.studyFilters.lifestyle=e.target.value;renderStudies();};
  C.onchange=e=>{state.studyFilters.country=e.target.value;renderStudies();};
}
function buildColMenu(){
  const groups={}; colFields().forEach(f=>{(groups[f.group]=groups[f.group]||[]).push(f);});
  let html=''; for(const g in groups){
    html+=`<div class="colgrp"><div class="colgrp-h"><span>${g}</span><span class="grptog" data-grp="${g}">toggle all</span></div>`+
      groups[g].map(f=>`<label><input type="checkbox" data-k="${f.key}" ${state.studyCols.includes(f.key)?'checked':''}>${f.label}</label>`).join('')+`</div>`;
  }
  const m=document.getElementById('colMenu'); m.innerHTML=html;
  m.querySelectorAll('input').forEach(cb=>cb.onchange=()=>{const k=cb.dataset.k;
    if(cb.checked){if(!state.studyCols.includes(k))state.studyCols.push(k);} else state.studyCols=state.studyCols.filter(x=>x!==k); renderStudies();});
  m.querySelectorAll('.grptog').forEach(a=>a.onclick=()=>{const ks=colFields().filter(f=>f.group===a.dataset.grp).map(f=>f.key);
    const all=ks.every(k=>state.studyCols.includes(k));
    state.studyCols = all ? state.studyCols.filter(k=>!ks.includes(k)) : [...new Set(state.studyCols.concat(ks))];
    buildColMenu(); renderStudies();});
}
function filteredStudyRows(){const F=state.studyFilters;
  return STUDY_ROWS.filter(r=>(!F.region||r.region===F.region)&&(!F.lifestyle||r.lifestyle===F.lifestyle)&&(!F.country||r.country===F.country));}
function cmp(a,b){return String(a==null?'':a).localeCompare(String(b==null?'':b));}
function sortStudyRows(rows,cols){
  const m=state.studySortMode;
  if(m==='AYC') rows.sort((a,b)=>cmp(a.author,b.author)||((a.year||0)-(b.year||0))||cmp(a.country,b.country));
  else if(m==='CAY') rows.sort((a,b)=>cmp(a.country,b.country)||cmp(a.author,b.author)||((a.year||0)-(b.year||0)));
  else { const sc=cols.find(c=>c.k===state.studySort.key)||cols[0], dir=state.studySort.dir;
    rows.sort((a,b)=>{let va=sc.get(a),vb=sc.get(b);
      if(sc.txt||typeof va==='string') return dir*cmp(va,vb);
      va=(va==null||!isFinite(va))?-Infinity:va; vb=(vb==null||!isFinite(vb))?-Infinity:vb; return dir*(va-vb);});}
}
function renderStudies(){
  const isTable=state.studyMode==='table';
  document.getElementById('studyTableWrap').classList.toggle('hidden',!isTable);
  document.getElementById('studyCharts').classList.toggle('hidden',isTable);
  const rows=filteredStudyRows(), varCols=state.studyCols.map(k=>byKey(k)).filter(Boolean);
  document.getElementById('studyCount').textContent=`${rows.length.toLocaleString()} study × country rows · ${varCols.length} variable${varCols.length===1?'':'s'}`;
  if(!isTable){ renderStudyCharts(rows,varCols); return; }
  const base=[{k:'author',t:'Author',txt:1,get:r=>r.author},{k:'year',t:'Year',get:r=>r.year},
    {k:'country',t:'Country',txt:1,get:r=>r.country},{k:'region',t:'Region',txt:1,get:r=>r.region},
    {k:'lifestyle',t:'Lifestyle',txt:1,dot:1,get:r=>r.lifestyle},{k:'n',t:'N',get:r=>r.n}];
  const cols=base.concat(varCols.map(f=>({k:f.key,t:f.label,f:f,get:r=>r.fields?r.fields[f.key]:null})));
  sortStudyRows(rows,cols);
  const disp=(col,r)=>{const v=col.get(r);
    if(col.dot)return `<span class="dot" style="background:${LIFECOL[v]||'var(--nodata)'}"></span>${v||'—'}`;
    if(col.txt)return v==null?'—':v;
    if(col.f)return studyVal(col.f,r);
    return typeof v==='number'?v.toLocaleString():(v==null?'—':v);};
  const act=state.studySortMode==='col'?state.studySort.key:null;
  const thead=document.querySelector('#stbl thead');
  thead.innerHTML='<tr>'+cols.map(c=>`<th data-k="${c.k}">${c.t}${act===c.k?(state.studySort.dir<0?' ▾':' ▴'):''}</th>`).join('')+'</tr>';
  thead.querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;
    if(state.studySortMode==='col'&&state.studySort.key===k) state.studySort.dir*=-1;
    else state.studySort={key:k,dir:(['author','country','region','lifestyle'].includes(k)?1:-1)};
    state.studySortMode='col'; document.getElementById('sortMode').value='col'; renderStudies();});
  document.querySelector('#stbl tbody').innerHTML=rows.map(r=>'<tr>'+cols.map(c=>`<td class="${c.txt?'name':''}">${disp(c,r)}</td>`).join('')+'</tr>').join('');
}
function sumHist(rows,key,nb){const a=new Array(nb).fill(0);
  rows.forEach(r=>{const h=r.smp&&r.smp[key]; if(Array.isArray(h))for(let i=0;i<nb;i++)a[i]+=(h[i]||0);}); return a;}
function sumProp(rows,key){let pos=0,pres=0;
  rows.forEach(r=>{const p=r.smp&&r.smp[key]; if(Array.isArray(p)){pos+=p[0]||0;pres+=p[1]||0;}}); return [pos,pres];}
function sumCat(rows,key){const c={};
  rows.forEach(r=>{const o=r.smp&&r.smp[key]; if(o&&!Array.isArray(o))for(const k in o)c[k]=(c[k]||0)+o[k];}); return c;}
function propBar(pos,pres){const frac=pres?pos/pres:0;
  return `<div class="barchart"><div class="bcrow"><span class="bclab">positive</span>`+
    `<span class="bcbar" style="width:${(100*frac).toFixed(0)}%;background:#2c728e"></span><span class="bcval">${fmt.pct(frac)}</span></div></div>`+
    `<div class="muted-lab" style="margin-top:2px">${pos.toLocaleString()} / ${pres.toLocaleString()} samples positive</div>`;}
function renderStudyCharts(rows,varCols){
  const cont=document.getElementById('studyCharts');
  if(!varCols.length){ cont.innerHTML='<div class="muted-lab" style="padding:14px">Add variables with <b>Columns ▾</b> to chart their distribution.</div>'; return; }
  const persample=state.chartUnit==='sample';
  cont.innerHTML=varCols.map(f=>{
    const tag=persample?'per sample':'per study';
    if(persample){
      if(f.kind==='numeric'){ const nb=f.hist.edges.length+1, counts=sumHist(rows,f.key,nb), tot=counts.reduce((a,b)=>a+b,0);
        return `<div class="chartcard"><div class="subttl">${f.label} <span class="muted-lab">· ${tot.toLocaleString()} samples</span></div>`+
               `${histSvg(counts,f.hist.edges,{label:f.label,fmt:f.fmt,log:f.hist.log,ylab:'samples'})}</div>`; }
      if(f.kind==='proportion'){ const [pos,pres]=sumProp(rows,f.key);
        return `<div class="chartcard"><div class="subttl">${f.label} <span class="muted-lab">· ${tag}</span></div>${propBar(pos,pres)}</div>`; }
      const counts=sumCat(rows,f.key);
      return `<div class="chartcard"><div class="subttl">${f.label} <span class="muted-lab">· samples by category</span></div>${barChart(counts,f)}</div>`;
    }
    if(f.kind==='numeric'||f.kind==='proportion'){ const vals=rows.map(r=>r.fields?r.fields[f.key]:null).filter(v=>v!=null&&isFinite(v));
      return `<div class="chartcard"><div class="subttl">${f.label} <span class="muted-lab">· ${vals.length} studies</span></div>${histFromValues(vals,f)}</div>`; }
    const counts={}; rows.forEach(r=>{const v=r.fields?r.fields[f.key]:null; if(v!=null)counts[v]=(counts[v]||0)+1;});
    return `<div class="chartcard"><div class="subttl">${f.label} <span class="muted-lab">· studies by dominant</span></div>${barChart(counts,f)}</div>`;
  }).join('');
  wireHist(cont);
}
function goHome(){
  state.selected=null; state.view='map'; state.group='Sequencing effort'; state.field='Total bases'; state.metric='central';
  setSeg('view','map'); document.getElementById('group').value='Sequencing effort'; buildFieldMenu(); syncMetric();
  render(); window.scrollTo({top:0,behavior:'smooth'});
}
function render(){
  const v=state.view;
  document.getElementById('mapPanel').classList.toggle('hidden',v!=='map');
  document.getElementById('tablePanel').classList.toggle('hidden',v!=='country');
  document.getElementById('studiesPanel').classList.toggle('hidden',v!=='studies');
  if(v==='map')renderMap(); else if(v==='country')renderTable(); else renderStudies();
  renderDetail();
  const f=byKey(state.field);
  document.getElementById('foot').textContent=
    `${f.group} › ${f.label} · coloured by ${state.metric==='missing'?'data completeness':state.metric} · `+
    `${f.kind==='numeric'?('distribution treated as '+f.dist+'; spread = '+(f.dist==='lognormal'?'geometric SD':'SD')+' · '):''}`+
    `${DATA.totals.n_countries} countries · Equal Earth`+
    (DATA.unmatched.length?` · unmatched: ${DATA.unmatched.join(', ')}`:'');
}

buildGroupMenu(); renderChips();
wireSeg('metric','metric'); wireSeg('view','view'); syncMetric();
populateStudyFilters(); buildColMenu();
wireSeg('studyMode','studyMode'); wireSeg('chartUnit','chartUnit');
document.getElementById('sortMode').onchange=e=>{state.studySortMode=e.target.value; if(state.view==='studies')renderStudies();};
document.getElementById('colBtn').onclick=e=>{e.stopPropagation();document.getElementById('colMenu').classList.toggle('hidden');};
document.addEventListener('click',e=>{ if(!e.target.closest('.colpick')) document.getElementById('colMenu').classList.add('hidden'); });
document.getElementById('home').onclick=goHome;
document.querySelector('header h1').onclick=goHome;
document.getElementById('theme').onclick=()=>{const r=document.documentElement;
  r.setAttribute('data-theme', r.getAttribute('data-theme')==='dark'?'light':'dark'); render();};
render();
</script>
</body>
</html>
"""


def _rel(path: Path) -> str:
    """Repo-relative path if possible, else enough of it to be unambiguous.

    The bare name is not always enough to identify a file: every reference
    writes its own `cohort_samples.csv`, and a `metadata_M1.csv` sits both at
    the repo root and under `final_run/`. So when the path is outside the repo
    -- an extracted run package, say -- keep it from the last `results/` or
    `final_run/` segment rather than reducing it to a basename that names
    several files.
    """
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        pass
    parts = p.parts
    for anchor in ("results", "final_run"):
        if anchor in parts:
            return str(Path(*parts[len(parts) - 1 - parts[::-1].index(anchor):]))
    return p.name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Follow whatever the pipeline itself reads, so the explorer and the
    # analysis can never describe different cohorts. Hardcoding
    # `metadata_M1.csv` at the repo root left this built from the 2026-08-14
    # file long after config had moved on to the curated final_run/ one.
    default_meta = REPO_ROOT / _cfg.META_FILE
    ap.add_argument("--meta", type=Path, default=default_meta)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "metadata_explorer.html")
    # The reported cohort, as the run wrote it. Default to the reference the
    # manuscript reports; --cohort "" shows the whole curated table instead,
    # which is a superset and will not match the paper's totals.
    # A plain string, not type=Path: argparse would turn the empty string into
    # Path('.'), which exists, is not a file, and reaches read_csv as a
    # directory.
    ap.add_argument("--cohort",
                    default=str(REPO_ROOT / "results" / "combined" /
                                "cohort_samples.csv"),
                    help="a run's cohort_samples.csv; pass an empty string to "
                         "show every row of --meta")
    args = ap.parse_args()
    if not args.meta.exists():
        sys.exit(f"metadata not found: {args.meta} (link it via setup_symlinks.sh)")
    cohort_path = Path(args.cohort) if str(args.cohort).strip() else None
    if cohort_path is not None and not cohort_path.is_file():
        print(f"  [warn] no cohort file at {_rel(cohort_path)}; showing every "
              f"row of the curated table, which is a superset of the reported "
              f"cohort and will not match the manuscript's totals")
        cohort_path = None

    print(f"Reading {args.meta} …")
    data, unmatched = build(args.meta, cohort_path)
    nfields = len([f for f in data["fields"] if f["kind"] != "count"]) + 1
    print(f"  {len(data['countries'])} countries · {len(data['fields'])} fields across "
          f"{len(set(f['group'] for f in data['fields']))} groups"
          + (f" · unmatched: {', '.join(unmatched)}" if unmatched else ""))
    # A companion to a data deposit has to say which version of the data it is
    # a companion to. Without this the page is undatable: two builds from
    # different metadata look identical.
    c = data.get("cohort", {})
    dropped = "; ".join(f"{v:,} {k}" for k, v in
                        (c.get("dropped_lifestyle") or {}).items())
    # Say which subset is on screen, in the terms the reader can check. On a
    # cohort-restricted build every total here is one the manuscript reports;
    # on a full-table build none of them are, and the sentence has to admit it
    # rather than leave the reader to discover the gap by arithmetic.
    if c.get("cohort_file"):
        scope = (f"&nbsp;·&nbsp; <b>Samples:</b> {c.get('n_shown', 0):,} "
                 f"(the analysed cohort, from "
                 f"<code>{c['cohort_file']}</code>; "
                 f"{c.get('n_outside_cohort', 0):,} of the "
                 f"{c.get('n_source', 0):,} curated rows fall outside it and "
                 f"are not shown) ")
    else:
        scope = (f"&nbsp;·&nbsp; <b>Samples:</b> {c.get('n_shown', 0):,} of "
                 f"{c.get('n_source', 0):,} curated rows "
                 + (f"({dropped} dropped as unclassifiable lifestyle) "
                    if dropped else "")
                 + "&mdash; the whole curated table, which is a superset of "
                   "the analysed cohort, so these totals are larger than the "
                   "manuscript's ")
    # Say what the health categories are and, plainly, that a sample carrying
    # two conditions is shown under one of them. A reader who adds the
    # per-condition percentages and gets more than the category counts should
    # find the reason here rather than conclude the page is inconsistent.
    h = c.get("health") or {}
    if h.get("counts"):
        named = ", ".join(lab for lab, _k, _col in HEALTH_CATEGORIES)
        health_note = (
            f"<b>Health category</b> is the reported set &mdash; {named}, and "
            f"<i>{HEALTH_OTHER_LABEL}</i> for an Unhealthy sample with none of "
            f"them &mdash; the four being the per-condition contrasts that "
            f"clear the reporting rule in Figure 5D&ndash;E and the health "
            f"supplementary figures. "
            + (f"{h['in_more_than_one']:,} samples carry more than one "
               f"condition and are counted under whichever comes first in that "
               f"list, so the <i>%&nbsp;condition</i> fields in the same group "
               f"sum higher than the categories do. "
               if h.get("in_more_than_one") else "")
            + (f"{h['unset']:,} samples have no curated health label and are "
               f"left out of this field, counting as missing. "
               if h.get("unset") else ""))
    else:
        health_note = ""
    prov = (
        # Repo-relative, not the basename: a stale metadata_M1.csv also sits
        # at the repo root, so "metadata_M1.csv" alone names two files.
        f"<b>Source:</b> <code>{_rel(args.meta)}</code> "
        f"&nbsp;·&nbsp; <b>Built:</b> {_dt.date.today():%Y-%m-%d} "
        + scope
        + f"&nbsp;·&nbsp; <b>Studies:</b> {c.get('n_studies', 0):,} "
        f"(in {c.get('n_studyfolders', 0):,} study&times;country folders) "
        f"&nbsp;·&nbsp; <b>Countries:</b> {len(data['countries']):,}<br>"
        "Health status is the curated <b>Healthy / Unhealthy</b> label, which "
        "is the case/control split used throughout the analysis. "
        + health_note
        + "The <i>Diseased&nbsp;(…)</i> fields are earlier roll-ups kept for "
          "reference only and are not used as a contrast anywhere, and "
          "<i>Specific diseases</i> lists every flag in the table rather than "
          "the reported set. "
          "Lifestyle groups follow Figure 1: Industrial, Agro-pastoralist, "
          "Hunter-gatherer."
    )
    html = (HTML_TEMPLATE
            .replace("__DATA__", json.dumps(data, allow_nan=False))
            .replace("__PROVENANCE__", prov))
    args.out.write_text(html)
    print(f"Wrote {args.out}  ({args.out.stat().st_size/1024:.0f} KB, self-contained)")


if __name__ == "__main__":
    main()
