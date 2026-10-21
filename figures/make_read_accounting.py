#!/usr/bin/env python3
"""
Figure S1 -- read accounting and reference recovery across the dataset.

The main text reports one pooled number for the cost of the mapping pipeline:
"41% of these bases survived trimming, removal of host reads, and alignment at
MAPQ >= 30". That number is not flat across lifestyles, and the direction it
leans matters for every richness comparison in the paper, so it is broken out
here.

  A  what each lifestyle's sequenced bases come to after mapping, as a share
     of its own total, with the pooled figure beneath for reference
  B  the same rate for each reference catalogue separately, on one shared set
     of samples, with the chimpanzee catalogue as the non-human floor
  C  the per-sample distribution behind (A), with each study's median drawn on
     top, so the gap is visibly a property of studies and not of samples
  D  the same gap within deciles of sequencing depth, which is the confound it
     would otherwise be attributed to

The residual industrial advantage is the point of the figure. Non-industrial
samples lose a larger share of their sequenced bases to the reference than
industrial samples do, which means the industrial richness deficit reported in
Section 1 is measured against an ascertainment asymmetry that runs the other
way. Combining the two human catalogues raises recovery for every lifestyle but
does not close that asymmetry; it widens it slightly.

Inputs, all already written by a full run -- nothing here re-reads a matrix:

  results/<ref>/cohort_samples.csv   mapped bases per sample, per reference
                                     (persistence/cohort_manifest.py)
  config.META_FILE                   "Total bases", the sequenced total, and
                                     the study each sample belongs to

    python figures/make_read_accounting.py
    python figures/make_read_accounting.py --out /tmp/preview.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))     # figures/
ROOT = os.path.dirname(HERE)                          # repo root
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from persistence import config, cohort_manifest, figure_source  # noqa: E402
from persistence.plot_utils import (                            # noqa: E402
    add_panel_letters,
    apply_oi_style,
    current_fs,
    label_pad,
    savefig_dual,
    square_figsize,
)

apply_oi_style()

#: The three human catalogues, in the order the main text introduces them,
#: with the labels it uses. `REF_LABELS` in figure_summary.py is the same
#: mapping; the internal key for the Carter catalogue is "UHGG".
HUMAN_REFS = [("UHGG", "Carter"), ("lev", "Leviatan"), ("combined", "Combined")]

#: The non-human control. Processed over a subset of the cohort only (see
#: Supplementary Text 4K), so it never shares an axis basis with the human
#: catalogues and is drawn as a floor rather than as a fourth bar.
CONTROL_REF = ("chimp", "Chimpanzee")

#: Internal lifestyle keys, in the order every other figure uses them.
LIFE = ["H-G", "Transitional", "Industrial"]

#: Catalogue sizes, for the (B) legend. Read from the length files rather than
#: hard-coded: these are the same counts the main text quotes.
_LENGTH_FILES = {"UHGG": "all_SGB_lengths_UHGG.txt",
                 "lev": "all_SGB_lengths_lev.txt",
                 "combined": "all_SGB_lengths_combined.txt",
                 "chimp": "all_SGB_lengths_chimp.txt"}

#: Panel (B) bar colours. Okabe-Ito members, so the style check passes; the
#: combined catalogue is the one the paper uses, so it gets the dark bar.
REF_COLORS = {"Carter": config.OI["pink"], "Leviatan": config.OI["blue"],
              "Combined": "#333333"}

N_DEPTH_BINS = 10
MIN_STUDY_N = 10          # studies below this are not given a median point


# ── inputs ────────────────────────────────────────────────────────────────
def _catalogue_size(ref: str, root: str) -> tuple:
    """(n_SGBs, total_bp) for a reference, or (None, None) if absent."""
    path = os.path.join(root, _LENGTH_FILES[ref])
    if not os.path.exists(path):
        return None, None
    lens = pd.read_csv(path, sep=r"\s+", header=None, usecols=[1],
                       names=["_id", "len"])["len"]
    lens = pd.to_numeric(lens, errors="coerce").dropna()
    return int(len(lens)), float(lens.sum())


def results_dir(root: str = ROOT, results_root: str = None) -> str:
    """The results tree to read from and write into.

    `results_root` is a path to a `results/` directory, the same thing
    `tools/replot_from_data.py --results-root` takes, so a finished run
    package can be refreshed without copying anything by hand.
    """
    return os.path.abspath(results_root or os.path.join(root, "results"))


def load_inputs(root: str = ROOT, results_root: str = None) -> dict:
    """Per-sample mapped bases for every reference, with sequenced bases.

    Returns {"mapped": DataFrame indexed by sample with one column per
    reference, "meta": DataFrame with Study / Lifestyle / seq_bp,
    "catalogues": {label: (n_sgb, total_bp)}}.
    """
    root = os.path.abspath(root)
    res = results_dir(root, results_root)

    meta_path = config.META_FILE
    if not os.path.isabs(meta_path):
        meta_path = os.path.join(root, meta_path)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata not found: {meta_path}")
    meta = pd.read_csv(meta_path, low_memory=False,
                       dtype={"Chosen_match_ID": str})
    keep = ["Chosen_match_ID", "Study", "Total bases"]
    missing = [c for c in keep if c not in meta.columns]
    if missing:
        raise KeyError(f"{os.path.basename(meta_path)} lacks {missing}")
    meta = (meta[keep].rename(columns={"Total bases": "seq_bp"})
            .drop_duplicates("Chosen_match_ID")
            .set_index("Chosen_match_ID"))
    meta["seq_bp"] = pd.to_numeric(meta["seq_bp"], errors="coerce")

    mapped, lifestyle, catalogues = {}, None, {}
    for ref, label in HUMAN_REFS + [CONTROL_REF]:
        man_path = os.path.join(res, ref, cohort_manifest.FILENAME)
        if not os.path.exists(man_path):
            print(f"  [skip] no cohort manifest for {ref}")
            continue
        man = pd.read_csv(man_path, dtype={"Chosen_match_ID": str})
        man = man.drop_duplicates("Chosen_match_ID").set_index("Chosen_match_ID")
        mapped[label] = pd.to_numeric(man["mapped_bp"], errors="coerce")
        catalogues[label] = _catalogue_size(ref, root)
        # The combined manifest defines the analytic cohort for the paper, so
        # its lifestyle column is the one every panel is grouped by.
        if ref == "combined":
            lifestyle = man["Lifestyle"].astype(str)
        elif lifestyle is None:
            lifestyle = man["Lifestyle"].astype(str)

    # Loud about the three the comparison rests on, quiet about the control.
    # A silently dropped catalogue would still draw a plausible panel B, which
    # is the worst of the three possible outcomes.
    absent = [lab for _, lab in HUMAN_REFS if lab not in mapped]
    if absent:
        raise FileNotFoundError(
            f"panel B compares the human catalogues and {absent} have no "
            f"cohort_samples.csv under {res} -- run those references first")
    if CONTROL_REF[1] not in mapped:
        print(f"  [note] no {CONTROL_REF[0]} cohort manifest; drawing without "
              f"the non-human control line")

    mapped = pd.DataFrame(mapped)
    meta = meta.reindex(mapped.index)
    meta["Lifestyle"] = lifestyle.reindex(mapped.index)
    return {"mapped": mapped, "meta": meta, "catalogues": catalogues}


# ── panels ────────────────────────────────────────────────────────────────
def panels_from_inputs(inp: dict) -> dict:
    """The four tidy tables the figure draws, and nothing else."""
    mapped, meta = inp["mapped"], inp["meta"]
    cats = inp["catalogues"]
    main = "Combined" if "Combined" in mapped.columns else mapped.columns[0]

    # (A) one row per lifestyle plus the pooled row, on the analytic cohort of
    # the reference the paper reports.
    base = pd.DataFrame({"Study": meta["Study"], "Lifestyle": meta["Lifestyle"],
                         "seq_bp": meta["seq_bp"],
                         "mapped_bp": mapped[main]}).dropna(
                             subset=["seq_bp", "mapped_bp"])
    rows = []
    for life in LIFE + ["All"]:
        sub = base if life == "All" else base[base["Lifestyle"] == life]
        if not len(sub):
            continue
        rows.append({"Lifestyle": life,
                     "display": ("All samples" if life == "All"
                                 else config.display_lifestyle(life)),
                     "n_samples": int(len(sub)),
                     "sequenced_bp": float(sub["seq_bp"].sum()),
                     "mapped_bp": float(sub["mapped_bp"].sum()),
                     "mapped_pct": float(100 * sub["mapped_bp"].sum()
                                         / sub["seq_bp"].sum())})
    read_accounting = pd.DataFrame(rows)

    # (B) every catalogue on its OWN analytic cohort, which is what makes the
    # combined bar here the same number as (A)'s.
    #
    # A reference's cohort is a property of its own matrix -- a sample maps
    # more bases against a bigger catalogue, so the >=30 Mb floor keeps
    # slightly different samples per reference. Restricting to the samples all
    # three retain is the stricter comparison and it was drawn that way first,
    # but it put 46.1 in this panel beside 46.0 in (A) for the same reference
    # and the same lifestyle, which reads as an error rather than as a
    # difference of basis. The stricter figures are computed anyway and
    # published in the data object as `shared_basis_pct`: no rate moves by
    # more than 0.1 pp, so nothing in the comparison turns on the choice, and
    # the caption says so.
    human = [lab for _, lab in HUMAN_REFS if lab in mapped.columns]
    shared = mapped[human].dropna().index.intersection(
        meta["seq_bp"].dropna().index)
    rows = []
    for lab in human:
        own = mapped[lab].dropna().index.intersection(
            meta["seq_bp"].dropna().index)
        sub = pd.DataFrame({"Lifestyle": meta.loc[own, "Lifestyle"],
                            "seq_bp": meta.loc[own, "seq_bp"],
                            "mapped_bp": mapped.loc[own, lab]})
        shr = pd.DataFrame({"Lifestyle": meta.loc[shared, "Lifestyle"],
                            "seq_bp": meta.loc[shared, "seq_bp"],
                            "mapped_bp": mapped.loc[shared, lab]})
        n_sgb, cat_bp = cats.get(lab, (None, None))
        for life in LIFE:
            s = sub[sub["Lifestyle"] == life]
            if not len(s):
                continue
            t = shr[shr["Lifestyle"] == life]
            rows.append({"reference": lab, "Lifestyle": life,
                         "display": config.display_lifestyle(life),
                         "n_samples": int(len(s)),
                         "n_sgb": n_sgb, "catalogue_bp": cat_bp,
                         "sequenced_bp": float(s["seq_bp"].sum()),
                         "mapped_bp": float(s["mapped_bp"].sum()),
                         "mapped_pct": float(100 * s["mapped_bp"].sum()
                                             / s["seq_bp"].sum()),
                         "n_samples_shared": int(len(t)),
                         "shared_basis_pct": (
                             float(100 * t["mapped_bp"].sum()
                                   / t["seq_bp"].sum()) if len(t) else np.nan),
                         "basis": "own analytic cohort"})
    # The control, on its own subset and industrial only -- 85 non-industrial
    # samples were processed against it, which Supplementary Text 4K already
    # declines to report.
    ctl = CONTROL_REF[1]
    if ctl in mapped.columns:
        idx = mapped[ctl].dropna().index.intersection(
            meta["seq_bp"].dropna().index)
        s = pd.DataFrame({"Lifestyle": meta.loc[idx, "Lifestyle"],
                          "seq_bp": meta.loc[idx, "seq_bp"],
                          "mapped_bp": mapped.loc[idx, ctl]})
        s = s[s["Lifestyle"] == "Industrial"]
        n_sgb, cat_bp = cats.get(ctl, (None, None))
        if len(s):
            rows.append({"reference": ctl, "Lifestyle": "Industrial",
                         "display": config.display_lifestyle("Industrial"),
                         "n_samples": int(len(s)),
                         "n_sgb": n_sgb, "catalogue_bp": cat_bp,
                         "sequenced_bp": float(s["seq_bp"].sum()),
                         "mapped_bp": float(s["mapped_bp"].sum()),
                         "mapped_pct": float(100 * s["mapped_bp"].sum()
                                             / s["seq_bp"].sum()),
                         "n_samples_shared": int(len(s)),
                         "shared_basis_pct": float(
                             100 * s["mapped_bp"].sum() / s["seq_bp"].sum()),
                         "basis": "own subset, industrial only"})
    catalogue_rate = pd.DataFrame(rows)

    # (C) the per-sample distribution. A mapped total above the sequenced total
    # is impossible; where it happens the metadata's "Total bases" is
    # under-reported for that accession, so those samples carry no ratio. The
    # count is published in `excluded_frac_gt_1` and stated in the caption.
    samp = base.copy()
    samp["frac"] = samp["mapped_bp"] / samp["seq_bp"]
    n_bad = int((~np.isfinite(samp["frac"])
                 | (samp["frac"] <= 0) | (samp["frac"] > 1)).sum())
    samp = samp[np.isfinite(samp["frac"]) & (samp["frac"] > 0)
                & (samp["frac"] <= 1)].copy()
    samp["group"] = np.where(samp["Lifestyle"] == "Industrial",
                             "Industrial", "Non-industrial")
    sample_fraction = samp.reset_index().rename(
        columns={"index": "Chosen_match_ID"})

    study_fraction = (samp.groupby(["Study", "Lifestyle"])
                      .agg(n_samples=("frac", "size"),
                           median_frac=("frac", "median"))
                      .reset_index())
    study_fraction["group"] = np.where(
        study_fraction["Lifestyle"] == "Industrial",
        "Industrial", "Non-industrial")
    study_fraction["display"] = [config.display_lifestyle(v)
                                 for v in study_fraction["Lifestyle"]]

    # (D) the same contrast inside deciles of sequenced depth. Equal-count
    # bins, so each point rests on the same number of samples rather than on
    # whatever happens to fall in a fixed-width band.
    samp["bin"] = pd.qcut(np.log10(samp["seq_bp"]), N_DEPTH_BINS, labels=False)
    rows = []
    for b, grp in samp.groupby("bin"):
        # float, not the int64 the column arrives as: the geometric midpoint
        # of the widest decile multiplies two ~1e11 counts, which overflows
        # int64 silently and leaves the x axis with NaN positions.
        edge_lo = float(grp["seq_bp"].min())
        edge_hi = float(grp["seq_bp"].max())
        for g in ("Industrial", "Non-industrial"):
            s = grp[grp["group"] == g]
            rows.append({"bin": int(b), "group": g, "n_samples": int(len(s)),
                         "median_frac": (float(s["frac"].median())
                                         if len(s) else np.nan),
                         "seq_bp_lo": float(edge_lo),
                         "seq_bp_hi": float(edge_hi),
                         "seq_bp_mid": float(np.sqrt(edge_lo * edge_hi))})
    depth_bins = pd.DataFrame(rows)

    return {"read_accounting": read_accounting,
            "catalogue_rate": catalogue_rate,
            "sample_fraction": sample_fraction,
            "study_fraction": study_fraction,
            "depth_bins": depth_bins}, n_bad


# ── the statistics the panels annotate ────────────────────────────────────
def summarise(panels: dict) -> dict:
    """Every number the figure prints, derived from the frames it plots.

    Kept separate from the drawing so a replot from the data object computes
    the same annotations rather than reading cached strings.
    """
    from scipy import stats as sps

    out = {}
    ra = panels["read_accounting"].set_index("Lifestyle")
    if "All" in ra.index:
        out["pooled_pct"] = float(ra.loc["All", "mapped_pct"])

    cr = panels["catalogue_rate"]
    human = cr[cr["basis"] == "own analytic cohort"]
    if len(human):
        if "shared_basis_pct" in human:
            d = (human["mapped_pct"] - human["shared_basis_pct"]).abs()
            out["max_basis_shift_pp"] = float(d.max())
            out["shared_n"] = int(human["n_samples_shared"].sum()
                                  / max(human["reference"].nunique(), 1))
        piv = human.pivot(index="reference", columns="Lifestyle",
                          values="mapped_pct")
        if "Combined" in piv.index:
            singles = piv.drop(index="Combined", errors="ignore")
            if len(singles):
                gain = piv.loc["Combined"] - singles.max()
                out["combined_gain_pp"] = {k: float(v) for k, v in gain.items()}
            for life in ("H-G", "Industrial"):
                pass
            if {"H-G", "Industrial"} <= set(piv.columns):
                out["gap_combined_pp"] = float(piv.loc["Combined", "Industrial"]
                                               - piv.loc["Combined", "H-G"])
                best = singles["Industrial"].idxmax() if len(singles) else None
                if best is not None:
                    out["gap_best_single_pp"] = float(
                        singles.loc[best, "Industrial"]
                        - singles.loc[best, "H-G"])
    ctl = cr[cr["basis"] != "own analytic cohort"]
    if len(ctl):
        out["control_pct"] = float(ctl["mapped_pct"].iloc[0])
        out["control_n"] = int(ctl["n_samples"].iloc[0])

    sf = panels["study_fraction"]
    sf = sf[sf["n_samples"] >= MIN_STUDY_N]
    ind = sf[sf["group"] == "Industrial"]["median_frac"].dropna()
    non = sf[sf["group"] == "Non-industrial"]["median_frac"].dropna()
    if len(ind) and len(non):
        u = sps.mannwhitneyu(ind, non, alternative="two-sided")
        out.update(study_n_ind=int(len(ind)), study_n_non=int(len(non)),
                   study_med_ind=float(ind.median()),
                   study_med_non=float(non.median()),
                   study_p=float(u.pvalue))

    db = panels["depth_bins"].pivot(index="bin", columns="group",
                                    values="median_frac")
    if {"Industrial", "Non-industrial"} <= set(db.columns):
        gap = (db["Industrial"] - db["Non-industrial"]) * 100
        gap = gap.dropna()
        out.update(n_bins=int(len(gap)),
                   n_bins_industrial_higher=int((gap > 0).sum()),
                   median_gap_pp=float(gap.median()))

    samp = panels["sample_fraction"]
    for g, sub in samp.groupby("group"):
        r = sps.spearmanr(sub["seq_bp"], sub["frac"])
        key = g.lower().replace('-', '_')
        out[f"rho_{key}"] = float(r.statistic)
        # The caption quotes rho alone, but the submitted source data should
        # carry the test's p-value and n beside it.
        out[f"rho_p_{key}"] = float(r.pvalue)
        out[f"rho_n_{key}"] = int(len(sub))
    return out


def stats_table(summ: dict) -> pd.DataFrame:
    """The tests this figure's caption quotes, as a tidy frame.

    These numbers were already computed into the summary, but a summary dict
    rides in the data object's `extras` and so never reaches the submitted
    spreadsheet, which is written from the panels alone. Published as a panel
    they land in Data S1 beside the values they were computed from.
    """
    nan = float("nan")
    rows = []
    if "study_p" in summ:
        rows.append({
            "quantity": "study median mapped fraction",
            "group": "Industrial vs Non-industrial",
            "test": "Mann-Whitney U, two-sided",
            "statistic": nan,
            "p_value": float(summ["study_p"]),
            "n": int(summ.get("study_n_ind", 0))
                 + int(summ.get("study_n_non", 0)),
            "detail": f"median industrial="
                      f"{summ.get('study_med_ind', nan):.4f} "
                      f"(n={summ.get('study_n_ind', '')} studies), "
                      f"non-industrial={summ.get('study_med_non', nan):.4f} "
                      f"(n={summ.get('study_n_non', '')} studies); "
                      f"studies with >= {MIN_STUDY_N} samples",
        })
    for key, label in (("industrial", "Industrial"),
                       ("non_industrial", "Non-industrial")):
        if f"rho_{key}" in summ:
            rows.append({
                "quantity": "sequenced bases vs mapped fraction",
                "group": label,
                "test": "Spearman rank correlation, two-sided",
                "statistic": float(summ[f"rho_{key}"]),
                # A summary written before these keys existed carries neither;
                # report them missing rather than as a zero-sample test.
                "p_value": float(summ.get(f"rho_p_{key}", nan)),
                "n": int(summ[f"rho_n_{key}"]) if f"rho_n_{key}" in summ
                     else nan,
                "detail": "per sample",
            })
    if "n_bins" in summ:
        # Two separate facts, so two rows: the statistic column must name the
        # same thing the quantity column does.
        rows.append({
            "quantity": "median industrial - non-industrial gap in mapped "
                        "fraction, across depth bins",
            "group": "Industrial vs Non-industrial",
            "test": "",
            "statistic": float(summ.get("median_gap_pp", nan)),
            "p_value": nan,
            "n": int(summ["n_bins"]),
            "detail": "percentage points; descriptive, not a test",
        })
        rows.append({
            "quantity": "depth bins in which industrial maps a higher share",
            "group": "Industrial vs Non-industrial",
            "test": "",
            "statistic": float(summ.get("n_bins_industrial_higher", nan)),
            "p_value": nan,
            "n": int(summ["n_bins"]),
            "detail": "count of bins out of n",
        })
    return pd.DataFrame(rows)


def _p_text(p: float) -> str:
    if p < 1e-4:
        return "$p$ < 10$^{-4}$"
    return f"$p$ = {p:.3g}"


# ── drawing ───────────────────────────────────────────────────────────────
def _panel_A(ax, df, fs):
    """Sequenced bases against what survives mapping, as a share of each."""
    df = df.iloc[::-1].reset_index(drop=True)      # pooled row at the bottom
    y = np.arange(len(df))
    for i, row in df.iterrows():
        life = row["Lifestyle"]
        colour = ("#555555" if life == "All"
                  else config.LIFESTYLE_COLORS.get(life, "gray"))
        ax.barh(i, 100, color="#E8E8E8", edgecolor="white", height=0.66,
                linewidth=0.4, zorder=1)
        ax.barh(i, row["mapped_pct"], color=colour, edgecolor="white",
                height=0.66, linewidth=0.4, zorder=2)
        ax.text(row["mapped_pct"] - 1.5, i, f"{row['mapped_pct']:.1f}%",
                ha="right", va="center", fontsize=fs - 1, color="white",
                fontweight="bold", zorder=3)
        # Inside the unmapped remainder, not past the frame: at page width
        # anything drawn beyond x=100 lands under the next panel's y label.
        ax.text(row["mapped_pct"] + 2.0, i,
                f"{row['mapped_bp']/1e12:.1f} of "
                f"{row['sequenced_bp']/1e12:.1f} Tb   $n$ = {row['n_samples']:,}",
                ha="left", va="center", fontsize=fs - 2, color="#555555",
                zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["display"], fontsize=fs - 0.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_xlabel("sequenced bases retained after mapping (%)",
                  fontsize=fs, labelpad=label_pad())
    ax.axhline(0.5, color="#BBBBBB", lw=0.6, ls=(0, (3, 2)))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _panel_B(ax, df, summ, fs):
    """One rate per catalogue, on one set of samples, with the non-human floor."""
    human = df[df["basis"] == "own analytic cohort"]
    refs = [lab for _, lab in HUMAN_REFS if lab in set(human["reference"])]
    lives = [l for l in LIFE if l in set(human["Lifestyle"])]
    width = 0.78 / max(len(refs), 1)
    x = np.arange(len(lives))

    for j, ref in enumerate(refs):
        sub = human[human["reference"] == ref].set_index("Lifestyle")
        vals = [sub.loc[l, "mapped_pct"] if l in sub.index else np.nan
                for l in lives]
        off = (j - (len(refs) - 1) / 2) * width
        # Catalogue sizes go in the caption, not the legend: spelled out here
        # the three labels will not fit on one row at panel width, and a
        # stacked legend has to sit over the bars.
        ax.bar(x + off, vals, width * 0.92, label=ref,
               color=REF_COLORS.get(ref, "gray"), edgecolor="white",
               linewidth=0.5, zorder=2)
        for xi, v in zip(x + off, vals):
            if np.isfinite(v):
                # Inside the bar and turned, not above it: the publication
                # type pass brings every small string up towards 9 pt, and at
                # that size two horizontal labels on adjacent bars read as one
                # number ("38.938.3").
                ax.text(xi, v - 0.8, f"{v:.1f}", ha="center", va="top",
                        rotation=90, fontsize=fs - 2, color="white",
                        fontweight="bold", zorder=3)

    handles = [Patch(facecolor=REF_COLORS.get(r, "gray"), edgecolor="white",
                     label=r) for r in refs]

    ctl = df[df["basis"] != "own analytic cohort"]
    if len(ctl):
        v = float(ctl["mapped_pct"].iloc[0])
        ax.axhline(v, color="#777777", lw=1.0, ls=(0, (4, 2)), zorder=1)
        # The control goes in the legend rather than on the line: every x
        # position in this panel has a bar standing on it, so a label beside
        # the line is a label over the data wherever it is put.
        handles.append(Line2D([], [], color="#777777", lw=1.0, ls=(0, (4, 2)),
                              label=f"Chimpanzee ({v:.1f}%)"))

    ax.set_xticks(x)
    ax.set_xticklabels([config.display_lifestyle(l) for l in lives],
                       fontsize=fs - 0.5)
    ax.set_ylabel("bases mapped (%)", fontsize=fs, labelpad=label_pad())
    ax.set_ylim(0, max(58.0, float(human["mapped_pct"].max()) + 12))
    leg = ax.legend(handles=handles, loc="upper center", ncol=2,
                    fontsize=fs - 1.5, frameon=False, handlelength=1.2,
                    handletextpad=0.45, columnspacing=1.3, borderaxespad=0.0,
                    bbox_to_anchor=(0.5, 1.02))
    leg._fixed_loc = True          # keep it off the bars; see place_legends_clear
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _panel_C(ax, samp, study, summ, fs):
    """Per-sample fractions, with each study's median on top."""
    lives = [l for l in LIFE if l in set(samp["Lifestyle"])]
    rng = np.random.default_rng(42)
    for i, life in enumerate(lives):
        v = samp.loc[samp["Lifestyle"] == life, "frac"].to_numpy()
        colour = config.LIFESTYLE_COLORS.get(life, "gray")
        if not len(v):
            continue
        parts = ax.violinplot([v], positions=[i], widths=0.72,
                              showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(colour)
            body.set_alpha(0.40)
            body.set_edgecolor("none")
        q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
        ax.plot([i, i], [q1, q3], color="#333333", lw=1.1, zorder=3,
                solid_capstyle="butt")
        ax.plot([i], [med], marker="o", ms=3.4, color="white",
                markeredgecolor="#333333", markeredgewidth=0.9, zorder=4)

        s = study[(study["Lifestyle"] == life)
                  & (study["n_samples"] >= MIN_STUDY_N)]
        if len(s):
            jit = rng.uniform(-0.17, 0.17, len(s))
            ax.scatter(i + 0.30 + jit, s["median_frac"], s=7.0,
                       facecolor=colour, edgecolor="#333333", linewidth=0.35,
                       alpha=0.95, zorder=5)
    ax.set_xticks(np.arange(len(lives)))
    ax.set_xticklabels([config.display_lifestyle(l) for l in lives],
                       fontsize=fs - 0.5)
    ax.set_ylabel("mapped fraction of sequenced bases",
                  fontsize=fs, labelpad=label_pad())
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # The statistics live in the caption. What the panel cannot do without
    # is saying what the points are, because nothing else on it does.
    ax.text(0.5, 0.985, "points: study medians",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=fs - 2, color="#555555")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _panel_D(ax, df, summ, fs):
    """The same gap inside deciles of sequencing depth."""
    piv = df.pivot(index="bin", columns="group", values="median_frac")
    mid = df.groupby("bin")["seq_bp_mid"].first() / 1e9
    style = {"Industrial": (config.LIFESTYLE_COLORS["Industrial"], "o"),
             "Non-industrial": (config.LIFESTYLE_COLORS["Non-Industrial"], "s")}
    if {"Industrial", "Non-industrial"} <= set(piv.columns):
        ax.fill_between(mid.loc[piv.index], piv["Non-industrial"],
                        piv["Industrial"], color="#BBBBBB", alpha=0.35,
                        linewidth=0, zorder=1)
    for g in ("Non-industrial", "Industrial"):
        if g not in piv.columns:
            continue
        colour, marker = style[g]
        ax.plot(mid.loc[piv.index], piv[g], marker=marker, ms=3.6, lw=1.3,
                color=colour, markeredgecolor="white", markeredgewidth=0.5,
                label=g, zorder=3)
    ax.set_xscale("log")
    # Decade ticks alone leave two labels on this range; name the decile
    # positions a reader is actually looking at.
    from matplotlib.ticker import FixedLocator, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10, 30]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:g}" if v >= 1 else f"{v:.1f}"))
    ax.set_xlabel("sequenced bases per sample (Gb, decile midpoint)",
                  fontsize=fs, labelpad=label_pad())
    ax.set_ylabel("median mapped fraction", fontsize=fs,
                  labelpad=label_pad())
    lo = float(np.nanmin(piv.to_numpy())) if piv.size else 0.3
    hi = float(np.nanmax(piv.to_numpy())) if piv.size else 0.5
    # Room under the series for the legend only; the annotation it used to
    # carry is in the caption. Limits snapped to 0.05 so the locator lands on
    # round tick values rather than on 0.24 / 0.32 / 0.40.
    ax.set_ylim(max(0.0, np.floor((lo - 0.11) * 20) / 20),
                np.ceil((hi + 0.05) * 20) / 20)
    leg = ax.legend(loc="lower center", ncol=2, fontsize=fs - 1.5,
                    frameon=False, handlelength=1.4, handletextpad=0.5,
                    columnspacing=1.4, borderaxespad=0.2)
    leg._fixed_loc = True
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw(panels: dict, outdir: str, stem: str = "read_accounting") -> str:
    """Draw the figure from its tables, and from nothing else."""
    summ = summarise(panels)
    fs = current_fs()
    _MG = dict(top=0.92, bottom=0.075, left=0.095, right=0.975,
               wspace=0.30, hspace=0.34)
    fig, axes = plt.subplots(2, 2, figsize=square_figsize(
        n_cols=2, n_rows=2, margins=_MG, extra_h_in=0.25))

    _panel_A(axes[0, 0], panels["read_accounting"], fs)
    _panel_B(axes[0, 1], panels["catalogue_rate"], summ, fs)
    _panel_C(axes[1, 0], panels["sample_fraction"], panels["study_fraction"],
             summ, fs)
    _panel_D(axes[1, 1], panels["depth_bins"], summ, fs)

    fig._margin_overrides = _MG
    fig._plain_axes_kw = dict(x_ticks=5, y_ticks=5)
    add_panel_letters(fig)
    fig.suptitle("Read accounting and reference recovery across the dataset",
                 fontsize=fs + 1, fontweight="bold")

    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir, f"{stem}.png")
    savefig_dual(fig, fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fpath}")
    return fpath


# ── CLI ───────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="combined",
                    help="reference whose results directory receives the "
                         "figure; the panels always compare all catalogues")
    ap.add_argument("--results-root", default=None,
                    help="a results/ directory to read from and write into, "
                         "as tools/replot_from_data.py takes, so a finished "
                         "run package can be refreshed in place")
    ap.add_argument("--out", default=None,
                    help="write the preview here instead of into results/")
    a = ap.parse_args()

    inp = load_inputs(ROOT, results_root=a.results_root)
    panels, n_bad = panels_from_inputs(inp)
    summ = summarise(panels)

    print(f"  pooled mapping rate: {summ.get('pooled_pct', float('nan')):.1f}%")
    print(f"  combined-catalogue gain over the better single catalogue: "
          + ", ".join(f"{k} {v:+.1f} pp" for k, v in
                      summ.get("combined_gain_pp", {}).items()))
    print(f"  industrial - H-G gap: {summ.get('gap_best_single_pp', float('nan')):+.1f} pp "
          f"best single -> {summ.get('gap_combined_pp', float('nan')):+.1f} pp combined")
    print(f"  samples without a usable ratio (mapped > sequenced): {n_bad}")

    out_dir = (os.path.dirname(os.path.abspath(a.out)) if a.out else
               os.path.join(results_dir(ROOT, a.results_root), a.ref,
                            "section1_diversity", "supplement"))
    stem = (os.path.splitext(os.path.basename(a.out))[0] if a.out
            else f"read_accounting_{a.ref}")

    if not a.out:
        # emit_for_path, not emit("S1"): the slot comes from the manifest, so
        # a future renumbering moves the data object with the figure instead
        # of leaving a correctly drawn figure beside a wrongly named pickle.
        _st = stats_table(summ)
        _panels = dict(panels)
        if not _st.empty:
            _panels["stats"] = _st
        figure_source.emit_for_path(
            os.path.join(out_dir, f"{stem}.png"), _panels, reference=a.ref,
            extras={"summary": summ, "excluded_frac_gt_1": n_bad,
                    "min_study_n": MIN_STUDY_N,
                    "n_depth_bins": N_DEPTH_BINS,
                    "jitter_seed": 42},
            notes="Sequenced bases from the metadata 'Total bases' column; "
                  "mapped bases are sum(mean_depth x genome_length) per "
                  "sample, as published in results/<ref>/cohort_samples.csv. "
                  "Panel B is computed on each reference's own analytic "
                  "cohort, so its combined/industrial rate matches panel A; "
                  "shared_basis_pct gives the same rate on the three-way "
                  "intersection and differs by at most 0.03 pp. The "
                  "chimpanzee control is on its own industrial-only subset.")

    draw(panels, out_dir, stem=stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
