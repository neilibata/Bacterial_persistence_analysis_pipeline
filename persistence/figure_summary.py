#!/usr/bin/env python3
"""
figure_summary.py
=================
Publication-quality summary figure (panels A–F) for gut microbiome
SGB accumulation analysis.

Layout (3 rows × 2 cols):
  Row 0:  A = Carter (UHGG) accumulation   B = Leviatan (lev) accumulation
  Row 1:  C = Carter asymptote sweep        D = Leviatan asymptote sweep
  Row 2:  E = Carter excl. Hadza-exclusive  F = Hill-MM world extrapolation

Two figure variants are produced:
  figure_summary_Richards.pdf/png   — Richards(logN) model
  figure_summary_StretchedExp.pdf/png — Stretched-exponential model

Run (from pipeline root):
    python -m persistence.figure_summary
    python -m persistence.figure_summary --n-boot 200 --out results/fig3
"""

import persistence.config as _cfg
import argparse
import os
import shutil      # used by the atomic save below; was missing
import tempfile    # ditto
import warnings
from scipy.special import erf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

try:
    from .plot_utils import apply_display_text
except ImportError:
    from matplotlib.text import Text

    def apply_display_text(fig):
        for artist in fig.findobj(match=lambda item: isinstance(item, Text)):
            text = artist.get_text()
            for old in ("Agro-Pastorialist", "Transitional"):
                text = str(text).replace(old, "Agro-Pastoralist")
            artist.set_text(text)

# ════════════════════════════════════════════════════════════════════════════
#  GLOBAL STYLE
# ════════════════════════════════════════════════════════════════════════════

FS_TITLE  = 12
FS_LABEL  = 11
FS_TICK   = 10
FS_LEGEND = 9
FS_ANNOT  = 9
FS_PANEL  = 14

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          FS_TICK,
    "axes.labelsize":     FS_LABEL,
    "axes.titlesize":     FS_TITLE,
    "xtick.labelsize":    FS_TICK,
    "ytick.labelsize":    FS_TICK,
    "legend.fontsize":    FS_LEGEND,
    "axes.linewidth":     0.7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.minor.size":   2,
    "ytick.minor.size":   2,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "lines.linewidth":    1.2,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# Colorblind-safe (Wong 2011) — consistent across ALL panels A–E
COL = {
    "Industrial":     "#56B4E9",   # sky blue
    "Non-Industrial": "#E69F00",   # orange
    "world":          "#555555",
    "band":           "#CC79A7",
    "fit_F":          "#CC79A7",
}

REF_LABELS = {"UHGG": "Carter", "lev": "Leviatan"}

# ════════════════════════════════════════════════════════════════════════════
#  PIPELINE-MIRRORED GEOGRAPHY TABLES  (verbatim from config.py)
# ════════════════════════════════════════════════════════════════════════════

COUNTRY_ALIASES = {
    "SWE":"Sweden","Sweden":"Sweden","DNK":"Denmark","DEN":"Denmark","Denmark":"Denmark",
    "DEU":"Germany","GER":"Germany","Germany":"Germany",
    "FRA":"France","France":"France","NLD":"Netherlands","Netherlands":"Netherlands",
    "GBR":"United Kingdom","UK":"United Kingdom","United Kingdom":"United Kingdom",
    "ESP":"Spain","Spain":"Spain","AUT":"Austria","AUS":"Austria","Austria":"Austria",
    "ITA":"Italy","Italy":"Italy","KAZ":"Kazakhstan","Kazakhstan":"Kazakhstan",
    "FIN":"Finland","Finland":"Finland","BEL":"Belgium","Belgium":"Belgium",
    "ISL":"Iceland","Iceland":"Iceland","RUS":"Russia","Russia":"Russia",
    "USA":"United States of America","United States":"United States of America",
    "United States of America":"United States of America",
    "CAN":"Canada","Canada":"Canada","BRA":"Brazil","Brazil":"Brazil",
    "ARG":"Argentina","Argentina":"Argentina","PER":"Peru","Peru":"Peru",
    "COL":"Colombia","Colombia":"Colombia",
    "SLV":"El Salvador","El Salvador":"El Salvador",
    "HND":"Honduras","HON":"Honduras","Honduras":"Honduras",
    "ZAF":"South Africa","SA":"South Africa","South Africa":"South Africa",
    "ETH":"Ethiopia","Ethiopia":"Ethiopia","GHA":"Ghana","Ghana":"Ghana",
    "TZA":"Tanzania","Tanzania":"Tanzania",
    "GNB":"Guinea-Bissau","Guinea-Bissau":"Guinea-Bissau",
    "MDG":"Madagascar","MAD":"Madagascar","Madagascar":"Madagascar",
    "CMR":"Cameroon","Cameroon":"Cameroon",
    "LBR":"Liberia","LIB":"Liberia","Liberia":"Liberia",
    "BFA":"Burkina Faso","BKF":"Burkina Faso","Burkina Faso":"Burkina Faso",
    "COD":"Democratic Republic of the Congo",
    "DRC":"Democratic Republic of the Congo",
    "DRC Congo":"Democratic Republic of the Congo",
    "CHN":"China","CHI":"China","China":"China",
    "JPN":"Japan","Japan":"Japan","NPL":"Nepal","NEP":"Nepal","Nepal":"Nepal",
    "THA":"Thailand","Thailand":"Thailand","MNG":"Mongolia","Mongolia":"Mongolia",
    "IDN":"Indonesia","IND":"Indonesia","Indonesia":"Indonesia",
    "ISR":"Israel","Israel":"Israel","KOR":"South Korea","South Korea":"South Korea",
    "BGD":"Bangladesh","Bangladesh":"Bangladesh",
    "PHL":"Philippines","PHN":"Philippines","Philippines":"Philippines",
    "FJI":"Fiji","Fiji":"Fiji","NZL":"New Zealand","New Zealand":"New Zealand",
}

COUNTRY_TO_CONTINENT = {
    "Sweden":"Europe","Denmark":"Europe","Germany":"Europe","France":"Europe",
    "Netherlands":"Europe","United Kingdom":"Europe","Spain":"Europe",
    "Austria":"Europe","Italy":"Europe","Kazakhstan":"Europe","Finland":"Europe",
    "Belgium":"Europe","Iceland":"Europe","Russia":"Europe",
    "United States of America":"North America","Canada":"North America",
    "El Salvador":"North America","Honduras":"North America",
    "Brazil":"South America","Peru":"South America",
    "Argentina":"South America","Colombia":"South America",
    "South Africa":"Africa","Ethiopia":"Africa","Ghana":"Africa",
    "Tanzania":"Africa","Guinea-Bissau":"Africa","Madagascar":"Africa",
    "Cameroon":"Africa","Liberia":"Africa","Burkina Faso":"Africa",
    "Democratic Republic of the Congo":"Africa",
    "Israel":"Asia","China":"Asia","Japan":"Asia","Nepal":"Asia",
    "Thailand":"Asia","Mongolia":"Asia","Indonesia":"Asia",
    "South Korea":"Asia","Bangladesh":"Asia","Philippines":"Asia",
    "Fiji":"Oceania","New Zealand":"Oceania",
}

# ════════════════════════════════════════════════════════════════════════════
#  PIPELINE DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

REFERENCES = {
    "UHGG": {
        "breadth_file": "meancoverage_and_std_matrix_high_UHGG_merged_dedup_firstcol.csv",
        "depth_file":   "meandepth_and_std_matrix_high_UHGG_merged_dedup_firstcol.csv",
        "lengths_file": "all_SGB_lengths_UHGG.txt",
    },
    "lev": {
        "breadth_file": "meancoverage_and_std_matrix_high_lev_merged_dedup_firstcol.csv",
        "depth_file":   "meandepth_and_std_matrix_high_lev_merged_dedup_firstcol.csv",
        "lengths_file": "all_SGB_lengths_lev.txt",
    },
}

META_FILE_CANDIDATES = [
    "metadata_Mar_16_2026.csv",
    "metadata_non_infants_gt100Mb_sra_by_biosample_best_meandepth_fullmeta_dedup_by_Chosen_match_ID_filled_v2.csv",
    "metadata_non_infants_gt100Mb_sra_by_biosample_best_meandepth_fullmeta_dedup_by_Chosen_match_ID.csv",
    "diag_depth_abundance_meta_filtered_ge_10000000_no_newborn.csv",
    "Chosen_population_samples.csv",
]

EXCLUDE_STUDIES = frozenset({
    "Bengtsson-PalmeJ","DhakanDB","FerrettiP",
    "Heintz-BuschartA","KaurK","UKN","eKmeta",
})
STUDY_EXCLUSION_COLS = [
    "StudyFolder","study_name [MBIOTEMP_study_name]","Author",
]
MIN_TOTAL_BP = 3e7


def _find_meta():
    for c in META_FILE_CANDIDATES:
        if os.path.isfile(c):
            return c
    return META_FILE_CANDIDATES[0]


def _normalise_country(s):
    if pd.isna(s): return None
    return COUNTRY_ALIASES.get(str(s).strip(), str(s).strip())


def _get_continent(c):
    if c is None: return None
    result = COUNTRY_TO_CONTINENT.get(str(c).strip())
    if result: return result
    return COUNTRY_TO_CONTINENT.get(_normalise_country(c))


def _classify_macro(row):
    life = row.get("Lifestyle")
    if life == "Industrial": return "Industrial"
    if life in ("H-G","Transitional"): return "Non-Industrial"
    return None


def _apply_study_exclusion(meta):
    mask = pd.Series(False, index=meta.index)
    for col in STUDY_EXCLUSION_COLS:
        if col not in meta.columns: continue
        s = meta[col].astype(str).str.strip()
        if col == "Author":
            mask |= s.isin(EXCLUDE_STUDIES)
        else:
            for pfx in EXCLUDE_STUDIES:
                mask |= s.str.startswith(pfx)
    return meta[~mask].copy(), int(mask.sum())


def load_reference(ref_name):
    """
    Mirror of loading.py — tries pipeline import first, falls back inline.
    """
    try:
        from . import loading as _loading
        print(f"\n  Loading {ref_name} via pipeline loading.py...")
        data = _loading.load_reference(ref_name)
        return {"breadth": data.breadth, "meta": data.meta,
                "n_sgbs": data.breadth.shape[1]}
    except ImportError:
        pass

    cfg = REFERENCES[ref_name]
    print(f"\n  Loading {ref_name} (standalone)...")

    def _load_matrix(filepath, tag):
        df = pd.read_csv(filepath, index_col=0, low_memory=False)
        df.index = df.index.astype(str).str.strip()
        mean_cols = [c for c in df.columns if c.endswith("_mean")]
        df = df[mean_cols].copy()
        df.columns = [c.rsplit("_mean", 1)[0] for c in mean_cols]
        return df.fillna(0.0)

    breadth_df = _load_matrix(cfg["breadth_file"], "breadth")
    depth_df   = _load_matrix(cfg["depth_file"],   "depth")
    common     = sorted(set(breadth_df.columns) & set(depth_df.columns))
    breadth_df = breadth_df[common]; depth_df = depth_df[common]

    length_df  = pd.read_csv(cfg["lengths_file"], sep=r"\s+", header=None,
                              names=["microbe","length"])
    length_df["microbe"] = length_df["microbe"].astype(str).str.strip()
    ldict = dict(zip(length_df["microbe"], length_df["length"]))
    keep  = [s for s in common if s in ldict]
    if not keep:
        raise ValueError(
            f"0 SGBs matched lengths for {ref_name}.\n"
            f"  Matrix cols[:5]: {common[:5]}\n"
            f"  Lengths keys[:5]: {list(ldict.keys())[:5]}")
    breadth_df = breadth_df[keep]; depth_df = depth_df[keep]
    lengths = pd.Series({s: ldict[s] for s in keep})
    print(f"    {len(keep):,} SGBs with lengths")

    total_bp = depth_df.mul(lengths, axis=1).sum(axis=1)

    meta_file = _find_meta()
    meta = pd.read_csv(meta_file, low_memory=False)
    # Table M1 stores the manuscript's display labels (Agro-pastoralist,
    # Hunter-gatherer); analysis code compares against the internal keys.
    if "Lifestyle" in meta.columns:
        meta["Lifestyle"] = _cfg.canonicalize_lifestyle_series(meta["Lifestyle"])
    if "Chosen_match_ID" not in meta.columns:
        raise ValueError(f"'Chosen_match_ID' not in {meta_file}; columns: {list(meta.columns[:8])}")
    meta["Chosen_match_ID"] = meta["Chosen_match_ID"].astype(str).str.strip()
    meta = meta[meta["Chosen_match_ID"].isin(breadth_df.index)].copy()
    meta = meta.drop_duplicates("Chosen_match_ID", keep="first").set_index("Chosen_match_ID")
    breadth_df = breadth_df.loc[meta.index]; total_bp = total_bp.loc[meta.index]

    meta["Country_norm"] = meta["Country"].apply(_normalise_country)
    if "Country (full)" in meta.columns:
        known   = set(COUNTRY_TO_CONTINENT.keys())
        no_cont = meta["Country_norm"].isna() | ~meta["Country_norm"].isin(known)
        if no_cont.any():
            meta.loc[no_cont,"Country_norm"] = \
                meta.loc[no_cont,"Country (full)"].apply(_normalise_country)
    meta["Continent"]  = meta["Country_norm"].apply(_get_continent)
    meta["MacroGroup"] = meta.apply(_classify_macro, axis=1)

    valid = meta["Lifestyle"].notna()
    meta = meta[valid]; breadth_df = breadth_df.loc[meta.index]; total_bp = total_bp.loc[meta.index]

    low = total_bp < MIN_TOTAL_BP
    if low.any():
        meta = meta[~low]; breadth_df = breadth_df.loc[meta.index]; total_bp = total_bp.loc[meta.index]

    meta, n_excl = _apply_study_exclusion(meta)
    breadth_df = breadth_df.loc[meta.index]
    if n_excl: print(f"    {n_excl:,} samples removed by study exclusion")

    nonind_na = meta["Lifestyle"].isin(["H-G","Transitional"]) & meta["Unhealthy"].isna()
    if nonind_na.any(): meta.loc[nonind_na,"Unhealthy"] = "Healthy"

    print(f"    {len(meta):,} samples after all filters")
    return {"breadth": breadth_df, "meta": meta, "n_sgbs": len(keep)}


# ════════════════════════════════════════════════════════════════════════════
#  GROUP MASKS
# ════════════════════════════════════════════════════════════════════════════

def build_group_blocks(meta, breadth_df, threshold=10.0):
    lifestyle = meta["Lifestyle"]
    blocks = {}
    for name, mask in [("Industrial (all)",     lifestyle == "Industrial"),
                       ("Non-Industrial (all)", lifestyle.isin(["H-G","Transitional"]))]:
        ids   = mask[mask].index.intersection(breadth_df.index)
        blocks[name] = breadth_df.loc[ids]
        print(f"    {name}: {len(ids):,} samples")
    return blocks


# ════════════════════════════════════════════════════════════════════════════
#  BOOTSTRAP ACCUMULATION
# ════════════════════════════════════════════════════════════════════════════

def bootstrap_curve(breadth_block, threshold, n_boot=100, rng=None, n_points=60):
    if rng is None: rng = np.random.default_rng(42)
    present  = (breadth_block.values > threshold).astype(bool)
    n_samples, n_mic = present.shape
    Ns = np.unique(np.concatenate([
        np.arange(1, min(25, n_samples)+1),
        np.round(np.logspace(np.log10(5), np.log10(n_samples), n_points)).astype(int),
        np.round(np.linspace(n_samples*0.60, n_samples, 20)).astype(int),
    ]))
    Ns = np.array(sorted({int(n) for n in Ns if 1 <= n <= n_samples}), dtype=int)
    boot = np.empty((n_boot, len(Ns)), dtype=float)
    for r in range(n_boot):
        perm = rng.permutation(n_samples); present_any = np.zeros(n_mic, dtype=bool); nxt = 0
        for i, N in enumerate(Ns):
            while nxt < N: present_any |= present[perm[nxt]]; nxt += 1
            boot[r, i] = present_any.sum()
    mean_v = boot.mean(0)
    std_v  = boot.std(0, ddof=1) if n_boot > 1 else np.zeros(len(Ns))
    q16, q84 = np.percentile(boot, [16, 84], axis=0)
    return (Ns.astype(float), mean_v, std_v, q16, q84, n_mic, n_boot, boot)


# ════════════════════════════════════════════════════════════════════════════
#  MODELS
# ════════════════════════════════════════════════════════════════════════════

def richards_logN(N, a, b, c, nu):
    logN = np.log(np.maximum(N, 1e-12))
    val  = 1.0 + np.exp(-float(c)*(logN - np.log(max(float(b),1e-9))))
    return a / np.power(np.maximum(val, 1e-9), 1.0/max(float(nu),1e-9))


def stretched_exp(N, S_max, lam, beta):
    """S(N) = S_max * (1 - exp(-(N/lam)^beta))"""
    return S_max * (1.0 - np.exp(-np.power(np.maximum(N,1e-12)/max(float(lam),1e-9), max(float(beta),1e-9))))


def hill_mm_panel_f(N, a, b, c):
    return a * N**c / (b**c + N**c)


MODEL_VARIANTS = {
    "Richards(logN)": dict(
        func=richards_logN,
        pnames=["a","b","c","ν"],
        p0_fn=lambda S,N: [S.max()*1.05, np.median(N), 1.5, 0.5],
        bounds=([10,1,0.1,0.05],[1e5,1e6,10,5.0]),
        color="#2b2d42", lw=1.4, ls="-", zorder=7,
        label_short="Richards",
    ),
    "StretchedExp": dict(
        func=stretched_exp,
        pnames=["S_max","λ","β"],
        p0_fn=lambda S,N: [S.max()*1.2, np.median(N)*0.5, 0.6],
        bounds=([10,1e-3,0.05],[1e5,1e8,3.0]),
        color="#D55E00", lw=1.4, ls="-", zorder=7,
        label_short="Str.Exp",
    ),
}


def fit_one(mname, N_vec, S_vec, sem_vec=None):
    spec = MODEL_VARIANTS[mname]
    n  = len(N_vec)
    k  = len(spec["pnames"])
    w  = np.maximum(sem_vec, 0.1) if sem_vec is not None and np.isfinite(sem_vec).all() else None
    try:
        popt, _ = curve_fit(
            spec["func"], N_vec, S_vec,
            p0=spec["p0_fn"](S_vec, N_vec),
            bounds=(spec["bounds"][0], spec["bounds"][1]),
            sigma=w, absolute_sigma=(w is not None),
            maxfev=40000, ftol=1e-12, xtol=1e-12,
        )
        yhat = spec["func"](N_vec, *popt)
        ssr  = float(np.sum((S_vec-yhat)**2))
        ss_t = float(np.sum((S_vec-S_vec.mean())**2))
        r2   = 1.0 - ssr/ss_t if ss_t > 0 else np.nan
        aic  = n*np.log(max(ssr/n,1e-20)) + 2*k
        return dict(popt=popt, r2=r2, aic=aic, failed=False)
    except Exception:
        return dict(popt=None, r2=np.nan, aic=np.nan, failed=True)


# ════════════════════════════════════════════════════════════════════════════
#  ASYMPTOTE SWEEP
# ════════════════════════════════════════════════════════════════════════════

def asymptote_sweep(mname, N_vec, boot_matrix, sem_vec, min_pts=10):
    spec = MODEL_VARIANTS[mname]
    n_boot, n_N = boot_matrix.shape
    results = []
    for k in range(min_pts, n_N+1):
        N_k = N_vec[:k]; S_k = boot_matrix[:, :k].mean(0); sem_k = sem_vec[:k]
        fd  = fit_one(mname, N_k, S_k, sem_k)
        if fd["failed"] or fd["popt"] is None: continue
        a_val = fd["popt"][0]
        a_boots = []
        for bi in range(min(n_boot, 200)):
            S_bi = boot_matrix[bi, :k]
            try:
                po, _ = curve_fit(
                    spec["func"], N_k, S_bi,
                    p0=spec["p0_fn"](S_bi, N_k),
                    bounds=(spec["bounds"][0], spec["bounds"][1]),
                    maxfev=8000, ftol=1e-9,
                )
                a_boots.append(po[0])
            except Exception:
                pass
        if len(a_boots) < 10: continue
        a_lo, a_hi = np.percentile(a_boots, [2.5, 97.5])
        results.append((N_vec[k-1], a_val, a_lo, a_hi))
    if not results:
        return np.array([]), np.array([]), np.array([]), np.array([])
    arr = np.array(results)
    return arr[:,0], arr[:,1], arr[:,2], arr[:,3]


# ════════════════════════════════════════════════════════════════════════════
#  HADZA EXCLUSION
# ════════════════════════════════════════════════════════════════════════════

def _hadza_mask(meta):
    mask = pd.Series(False, index=meta.index)
    for col in ["Population","Detailed location","Sample ID","StudyFolder","Author"]:
        if col in meta.columns:
            mask |= meta[col].astype(str).str.contains("hadza", case=False, na=False)
    if not mask.any():
        for cc in ["Country","Country (full)","Country_norm"]:
            if cc in meta.columns:
                tz   = meta[cc].astype(str).str.contains("TZA|Tanzania", case=False, na=False)
                mask = tz & (meta["Lifestyle"] == "H-G")
                if mask.any(): break
    return mask


def hadza_exclusive_sgbs(breadth_df, meta, threshold=10.0, hadza_min=0.01):
    """SGBs detected in Hadza (≥hadza_min prevalence) and completely absent elsewhere."""
    hg_ids    = meta.index[meta["Lifestyle"] == "H-G"].intersection(breadth_df.index)
    hadza_ids = hg_ids[_hadza_mask(meta.loc[hg_ids])]
    other_ids = hg_ids[~_hadza_mask(meta.loc[hg_ids])].union(
        meta.index[meta["Lifestyle"] == "Industrial"].intersection(breadth_df.index))
    if len(hadza_ids) == 0: return set()
    pres     = (breadth_df >= threshold)
    had_prev = pres.loc[hadza_ids].mean(0) if len(hadza_ids) else pd.Series(0.0, index=breadth_df.columns)
    oth_prev = pres.loc[other_ids].mean(0) if len(other_ids) else pd.Series(0.0, index=breadth_df.columns)
    excl = set(had_prev.index[(had_prev >= hadza_min) & (oth_prev == 0.0)])
    print(f"    Hadza-exclusive SGBs (≥{hadza_min:.0%} Hadza, 0% elsewhere): {len(excl):,}")
    return excl


# ════════════════════════════════════════════════════════════════════════════
#  PANEL F DATA  (Hill-MM extrapolation — unchanged from fig3g_sgb_extrapolation.py)
# ════════════════════════════════════════════════════════════════════════════

N_DATA = np.array([10,30,100,300,1000,3000,10000,30000,53545], dtype=float)
K_DATA = np.array([34.,89.,209.,382.,694.,1147.,1893.5,2891.,3594.])
K_Q025 = np.array([17.,56.475,158.,326.95,622.95,1091.475,1841.475,2838.425,3594.])
K_Q975 = np.array([82.,132.,248.025,441.05,766.575,1225.15,1947.,2937.05,3594.])
K_VAR  = np.array([291.978,404.767,631.303,824.353,1286.654,1158.729,887.218,735.844,0.])


def panel_f_data():
    sigma = np.sqrt(K_VAR); sigma[sigma==0] = 1.0
    popt, pcov = curve_fit(hill_mm_panel_f, N_DATA, K_DATA,
                           p0=[5000,500,0.7], sigma=sigma, absolute_sigma=True,
                           maxfev=50000, bounds=([0,0,0],[np.inf,np.inf,5]))
    perr   = np.sqrt(np.diag(pcov))
    K_pred = hill_mm_panel_f(N_DATA, *popt)
    R2     = 1 - np.sum((K_DATA-K_pred)**2)/np.sum((K_DATA-K_DATA.mean())**2)
    N_world = 8.2e9; S_world = hill_mm_panel_f(N_world, *popt)
    rng2 = np.random.default_rng(42)
    pb   = rng2.multivariate_normal(popt, pcov, 100_000)
    pb   = pb[np.all(pb>0, axis=1)]
    S_b  = np.array([hill_mm_panel_f(N_world,*p) for p in pb])
    S_q025, S_q975 = np.percentile(S_b, [2.5,97.5])
    N_c  = np.logspace(np.log10(5), np.log10(2e10), 2000)
    S_c  = hill_mm_panel_f(N_c, *popt)
    S_bd = np.array([hill_mm_panel_f(N_c,*p) for p in pb[:5000]])
    return dict(popt=popt, perr=perr, R2=R2,
                N_world=N_world, S_world=S_world, S_q025=S_q025, S_q975=S_q975,
                N_curve=N_c, S_curve=S_c,
                S_lo95=np.percentile(S_bd,2.5,axis=0),
                S_hi95=np.percentile(S_bd,97.5,axis=0),
                yerr_lo=K_DATA-K_Q025, yerr_hi=K_Q975-K_DATA)


# ════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _staggered_asym_labels(ax, entries, y_ceil, x_lo_frac=0.05, x_hi_frac=0.40,
                            fs=FS_ANNOT-1, is_log=False):
    """
    Draw horizontal asymptote lines and staggered labels.
    entries: list of (y_val, label_str, colour)
    Labels are spread horizontally over [x_lo_frac, x_hi_frac] * x_max
    and pushed vertically apart so they never overlap.
    """
    if not entries: return
    xl = ax.get_xlim()
    x_max = xl[1]
    entries_sorted = sorted(entries, key=lambda e: e[0])  # sort by y value
    n    = len(entries_sorted)
    x_pos = np.linspace(x_lo_frac, x_hi_frac, n) * x_max

    MIN_GAP = y_ceil * 0.028
    y_prev  = -1e9
    for i, (y_v, lbl, col) in enumerate(entries_sorted):
        ax.axhline(y_v, color=col, lw=0.8, ls="--", alpha=0.55, zorder=2)
        y_lbl = y_v + y_ceil * 0.005
        if y_lbl < y_prev + MIN_GAP:
            y_lbl = y_prev + MIN_GAP
        y_prev = y_lbl
        ax.text(x_pos[i], y_lbl, lbl,
                ha="center", va="bottom", fontsize=fs, color=col, alpha=0.90)


def draw_accum_panel(ax_main, data_by_group, mname, threshold=10.0,
                     title="", N_LOG_MAX=1e5, n_total_sgbs=None):
    """
    Linear main axes + log-space inset (bottom-right, 2/3 × 2/3).
    data_by_group: {label: (accum_8tuple, fit_dict, colour)}
    mname: which model to draw (must be key in MODEL_VARIANTS)
    n_total_sgbs: total SGBs in reference (for dotted ceiling line)
    """
    spec   = MODEL_VARIANTS[mname]
    ax_ins = ax_main.inset_axes([0.33, 0.05, 0.66, 0.65])

    all_a      = []
    n_samp_max = 1

    for gname, (accum, fd, col) in data_by_group.items():
        N_vec, mean_v, std_v, q16_v, q84_v = accum[0], accum[1], accum[2], accum[3], accum[4]
        n_boot_ = accum[6]
        n_samp  = int(N_vec[-1]); n_samp_max = max(n_samp_max, n_samp)
        sem_v   = std_v / np.sqrt(max(n_boot_, 1)); sem95 = sem_v * 1.96
        step    = max(1, len(N_vec)//40)
        N_lin   = np.linspace(1, n_samp*1.05, 500)
        N_log   = np.logspace(np.log10(1), np.log10(max(n_samp*2.5, N_LOG_MAX)), 600)

        if not fd["failed"] and fd["popt"] is not None:
            all_a.append((fd["popt"][0], gname, col))

        for ax_, is_log, N_plot in [(ax_main, False, N_lin), (ax_ins, True, N_log)]:
            ax_.fill_between(N_vec, q16_v, q84_v, color=col, alpha=0.16, zorder=1)
            ax_.plot(N_vec, mean_v, color=col, lw=0.8, alpha=0.35, zorder=1)
            ax_.errorbar(N_vec[::step], mean_v[::step], yerr=sem95[::step],
                         fmt="o", color=col, ms=3, capsize=2, lw=0,
                         elinewidth=0.9, alpha=0.80, zorder=10)
            if not fd["failed"]:
                yp = spec["func"](N_plot, *fd["popt"])
                ax_.plot(N_plot, yp, color=col, lw=spec["lw"],
                         ls=spec["ls"], alpha=0.92, zorder=spec["zorder"])

    # Asymptote ceiling
    a_values = [a for a, _, _ in all_a]
    y_ceil = float(np.ceil((max(a_values + ([float(n_total_sgbs)] if n_total_sgbs else [])) + 150) / 50) * 50) \
             if a_values else 5000

    # Linear axes: y-top = S(n_samp_max) from best fit * 1.08
    best_s = max(float(accum[1][-1]) for accum, _, _ in data_by_group.values())
    y_top_lin = best_s * 1.10

    ax_main.set_xscale("linear")
    ax_main.set_xlim(0, n_samp_max * 1.05)
    ax_main.set_ylim(0, y_top_lin)
    ax_main.set_ylabel(f"SGBs detected (breadth ≥ {threshold:.0f}%)", fontsize=FS_LABEL)
    ax_main.set_xlabel("Number of samples (N)", fontsize=FS_LABEL)
    ax_main.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=5)
    ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax_main.grid(True, ls="--", lw=0.3, alpha=0.35)

    # Asymptote lines on linear axes (only those below y_top_lin)
    lin_entries = [(a, f"{g.split(' ')[0]}: a={a:.0f}", c)
                   for a, g, c in all_a if a <= y_top_lin * 1.02]
    if n_total_sgbs and n_total_sgbs <= y_top_lin:
        lin_entries.append((float(n_total_sgbs), f"all SGBs={n_total_sgbs:,}", "dimgrey"))
    ax_main.set_xlim(0, n_samp_max * 1.05)  # ensure xlim set before labels
    _staggered_asym_labels(ax_main, lin_entries, y_top_lin,
                           x_lo_frac=0.05, x_hi_frac=0.45, fs=FS_ANNOT-1, is_log=False)

    # Log inset
    ax_ins.set_xscale("log")
    ax_ins.set_xlim(0.8, max(n_samp_max*2.5, N_LOG_MAX))
    ax_ins.set_ylim(0, y_ceil)
    ax_ins.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax_ins.tick_params(labelsize=FS_TICK-2, pad=1)
    ax_ins.set_xlabel("log N", fontsize=FS_TICK-1, labelpad=2)
    ax_ins.set_ylabel("SGBs", fontsize=FS_TICK-1, labelpad=2)
    ax_ins.grid(True, which="both", ls="--", lw=0.22, alpha=0.32)

    # Asymptote lines on inset (all, including total SGBs)
    ins_entries = [(a, f"{g.split(' ')[0]}: a={a:.0f}", c) for a, g, c in all_a]
    if n_total_sgbs:
        ins_entries.append((float(n_total_sgbs), f"all={n_total_sgbs:,}", "dimgrey"))
    ax_ins.set_xlim(0.8, max(n_samp_max*2.5, N_LOG_MAX))  # ensure before labels
    _staggered_asym_labels(ax_ins, ins_entries, y_ceil,
                           x_lo_frac=0.04, x_hi_frac=0.42, fs=FS_ANNOT-2.5)

    # Connector lines from linear top to matching inset position
    for y_m in [y_top_lin, 0.0]:
        y_f = float(np.clip(y_m/y_ceil, 0, 1))
        con = ConnectionPatch(
            xyA=(1.0, y_f), coordsA="axes fraction", axesA=ax_ins,
            xyB=(n_samp_max*1.04, y_m), coordsB="data", axesB=ax_main,
            color="dimgrey", lw=0.6, alpha=0.40, ls="-", arrowstyle="-",
        )
        ax_main.add_artist(con)
    for sp in ax_ins.spines.values():
        sp.set_edgecolor("dimgrey"); sp.set_linewidth(0.5); sp.set_alpha(0.5)

    # Group colour legend on main axes (no model name — avoids duplicate with asymptote labels)
    handles = [Line2D([0],[0], color=col, lw=2.0, label=gname)
               for gname, (_, _, col) in data_by_group.items()]
    ax_main.legend(handles=handles, fontsize=FS_LEGEND, frameon=False, loc="upper left")


def draw_sweep_panel(ax, sweep_by_group, title="", n_total_sgbs=None):
    """Asymptote-vs-N sweep with 95% CI band + error bars."""
    all_lo, all_hi = [], []
    for gname, (N_s, a_s, a_lo, a_hi, col, ls) in sweep_by_group.items():
        if len(N_s) == 0: continue
        ax.fill_between(N_s, a_lo, a_hi, color=col, alpha=0.20, zorder=1)
        step = max(1, len(N_s)//25)
        ax.errorbar(N_s[::step], a_s[::step],
                    yerr=[np.maximum(a_s[::step]-a_lo[::step], 0),
                          np.maximum(a_hi[::step]-a_s[::step], 0)],
                    fmt="none", color=col, elinewidth=0.8, capsize=2.5, capthick=0.7,
                    alpha=0.55, zorder=2)
        ax.plot(N_s, a_s, color=col, lw=1.4, ls=ls, zorder=3,
                label=f"{gname}  a={a_s[-1]:,.0f} [{a_lo[-1]:,.0f}–{a_hi[-1]:,.0f}]")
        all_lo.append(a_lo); all_hi.append(a_hi)

    if n_total_sgbs:
        ax.axhline(n_total_sgbs, color="dimgrey", lw=1.0, ls=":", alpha=0.65, zorder=2)
        ax.text(ax.get_xlim()[0]*1.5, n_total_sgbs * 1.02,
                f"total SGBs = {n_total_sgbs:,}",
                fontsize=FS_ANNOT-1, color="dimgrey", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Total samples included ($N_{total}$)", fontsize=FS_LABEL)
    ax.set_ylabel("Asymptote $a$ (SGBs)", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    if all_lo:
        ymin = min(v.min() for v in all_lo)*0.88
        ymax = max(v.max() for v in all_hi)*1.12
        if n_total_sgbs: ymax = max(ymax, n_total_sgbs*1.12)
        ax.set_ylim(max(0, ymin), ymax)
    leg = ax.legend(fontsize=FS_LEGEND, frameon=True, loc="upper right",
                    framealpha=0.9, edgecolor="#cccccc")
    leg.get_frame().set_linewidth(0.4)
    ax.grid(True, which="major", ls="--", lw=0.3, alpha=0.35)


def draw_panel_f(ax, pf):
    """Panel F: Hill-MM world extrapolation."""
    cF = COL["fit_F"]; cB = COL["band"]; cW = COL["world"]
    ax.fill_between(pf["N_curve"], pf["S_lo95"], pf["S_hi95"],
                    alpha=0.12, color=cB, linewidth=0, label="95% CI")
    mask_obs = pf["N_curve"] <= 6e4; mask_ext = pf["N_curve"] >= 5e4
    ax.plot(pf["N_curve"][mask_obs], pf["S_curve"][mask_obs], color=cF, lw=1.4,
            label=f"Hill–MM fit (R²={pf['R2']:.4f})", zorder=3)
    ax.plot(pf["N_curve"][mask_ext], pf["S_curve"][mask_ext], color=cF, lw=1.4,
            ls="--", label="Extrapolation", zorder=3)
    ax.errorbar(N_DATA, K_DATA, yerr=[pf["yerr_lo"],pf["yerr_hi"]],
                fmt="o", color="#333333", markersize=4, markerfacecolor="white",
                markeredgewidth=0.9, elinewidth=0.8, capsize=3, capthick=0.7,
                label="Observed (median ± 95% CI)", zorder=5)
    ax.axvline(pf["N_world"], color=cW, ls="-", lw=0.8, alpha=0.6, zorder=2)
    ax.text(pf["N_world"]*0.55, 1400, "World\npopulation\n(2026)",
            fontsize=FS_ANNOT-1, color=cW, ha="right", va="bottom",
            fontstyle="italic", linespacing=1.3)
    ax.axhline(pf["S_world"], color=cW, ls="-", lw=0.8, alpha=0.6, zorder=2)
    ax.text(5e5, pf["S_world"]+300,
            f"Predicted: {pf['S_world']:,.0f} SGBs\n"
            f"(95% CI: {pf['S_q025']:,.0f}–{pf['S_q975']:,.0f})",
            fontsize=FS_ANNOT-1, color=cW, va="bottom", ha="center",
            fontstyle="italic", linespacing=1.3)
    ax.plot(pf["N_world"], pf["S_world"], "*", color=cW, markersize=9,
            markeredgecolor="white", markeredgewidth=0.5, zorder=6)
    ax.set_xscale("log"); ax.set_xlim(5, 5e10); ax.set_ylim(0, 14000)
    ax.set_xlabel("Number of individuals sampled ($N$)", fontsize=FS_LABEL)
    ax.set_ylabel("SGBs detected", fontsize=FS_LABEL)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.tick_params(which="both", top=True, right=True)
    leg = ax.legend(loc="center left", bbox_to_anchor=(0.0,0.48),
                    frameon=True, fancybox=False, edgecolor="#cccccc",
                    framealpha=0.95, borderpad=0.5,
                    handlelength=1.8, handletextpad=0.5, labelspacing=0.4,
                    fontsize=FS_LEGEND)
    leg.get_frame().set_linewidth(0.4)


# ════════════════════════════════════════════════════════════════════════════
#  BUILD ONE FIGURE
# ════════════════════════════════════════════════════════════════════════════

def build_figure(mname, accum_results, fits_by_mname, boot_matrices,
                 sweep_results, accum_E, fits_E, pf,
                 n_sgbs_by_ref, threshold, n_boot, out_stem):
    """
    Build and save one figure variant for a given model (mname).
    Layout:
      Row 0: A=Carter accum    B=Leviatan accum
      Row 1: C=Carter sweep    D=Leviatan sweep
      Row 2: E=Carter excl.Hadza  F=world extrapolation
    """
    print(f"\n=== Generating figure: {mname} ===")
    FIG_W, FIG_H = 18, 16

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs  = gridspec.GridSpec(
        3, 2, figure=fig,
        hspace=0.32, wspace=0.28,
        left=0.08, right=0.97, top=0.93, bottom=0.06,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    ax_E = fig.add_subplot(gs[2, 0])
    ax_F = fig.add_subplot(gs[2, 1])

    for lbl, ax_ in [("A",ax_A),("B",ax_B),("C",ax_C),("D",ax_D),("E",ax_E),("F",ax_F)]:
        ax_.text(-0.11, 1.04, lbl, transform=ax_.transAxes,
                 fontsize=FS_PANEL, fontweight="bold", va="bottom")

    REFS = list(accum_results.keys())
    ref_A, ref_B = REFS[0], REFS[min(1, len(REFS)-1)]

    # ── Panels A / B ─────────────────────────────────────────────────────
    for ax_ab, ref, panel_lbl in [(ax_A, ref_A, REF_LABELS.get(ref_A, ref_A)),
                                   (ax_B, ref_B, REF_LABELS.get(ref_B, ref_B))]:
        data_by_g = {}
        for gname in ["Industrial (all)","Non-Industrial (all)"]:
            col = COL["Industrial"] if "Industrial" in gname else COL["Non-Industrial"]
            acc = accum_results[ref][gname]
            fd  = fits_by_mname[ref][gname][mname]
            data_by_g[gname] = (acc, fd, col)
        try:
            draw_accum_panel(ax_ab, data_by_g, mname,
                             threshold=threshold, title=panel_lbl,
                             n_total_sgbs=n_sgbs_by_ref.get(ref))
        except Exception as exc:
            import traceback
            print(f"  ERROR in panel {panel_lbl}: {exc}")
            traceback.print_exc()
            ax_ab.set_title(f"{panel_lbl} [ERROR]", color="red")

    # ── Panels C / D ─────────────────────────────────────────────────────
    for ax_cd, ref, panel_lbl in [(ax_C, ref_A, REF_LABELS.get(ref_A, ref_A)),
                                   (ax_D, ref_B, REF_LABELS.get(ref_B, ref_B))]:
        try:
            draw_sweep_panel(ax_cd, sweep_results[ref][mname],
                             title=panel_lbl,
                             n_total_sgbs=n_sgbs_by_ref.get(ref))
        except Exception as exc:
            import traceback
            print(f"  ERROR in sweep {panel_lbl}: {exc}")
            traceback.print_exc()
            ax_cd.set_title(f"{panel_lbl} sweep [ERROR]", color="red")

    # ── Panel E ───────────────────────────────────────────────────────────
    data_E = {}
    for gname_lbl, acc in accum_E.items():
        col = COL["Industrial"] if "Industrial" in gname_lbl else COL["Non-Industrial"]
        fd  = fits_E[gname_lbl][mname]
        data_E[gname_lbl] = (acc, fd, col)
    try:
        draw_accum_panel(ax_E, data_E, mname,
                         threshold=threshold,
                         title=f"{REF_LABELS.get(ref_A, ref_A)} — excl. Hadza-exclusive SGBs",
                         n_total_sgbs=n_sgbs_by_ref.get(ref_A))
    except Exception as exc:
        import traceback
        print(f"  ERROR in panel E: {exc}")
        traceback.print_exc()
        ax_E.set_title("Panel E [ERROR]", color="red")

    # ── Panel F ───────────────────────────────────────────────────────────
    try:
        draw_panel_f(ax_F, pf)
    except Exception as exc:
        import traceback
        print(f"  ERROR in panel F: {exc}")
        traceback.print_exc()
        ax_F.set_title("Panel F [ERROR]", color="red")

    spec = MODEL_VARIANTS[mname]
    fig.suptitle(
        f"Human gut SGB accumulation — {spec['label_short']} model   |   "
        f"breadth ≥ {threshold:.0f}%   |   {n_boot} bootstrap replicates",
        fontsize=FS_TITLE+1, fontweight="bold", y=0.975,
    )

    # Force canvas render before saving
    fig.canvas.draw()

    for ext in ("pdf", "png"):
        path = f"{out_stem}_{mname.replace('(','').replace(')','').replace(' ','_')}.{ext}"
        # Save to a local temp file first to avoid iCloud Drive / network-FS timeouts,
        # then atomically move to the final destination.
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, os.path.basename(path))
        apply_display_text(fig)
        fig.savefig(tmp_path, dpi=150, bbox_inches="tight", facecolor="white")
        shutil.move(tmp_path, path)
        print(f"  Saved → {path}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate summary figure (Richards + StretchedExp).")
    parser.add_argument("--refs",   nargs="+", default=["UHGG","lev"])
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--n-boot",    type=int,   default=100)
    parser.add_argument("--out",       default="figure_summary")
    args = parser.parse_args(argv)

    THRESHOLD = args.threshold; N_BOOT = args.n_boot
    rng = np.random.default_rng(42)

    # ── Load ─────────────────────────────────────────────────────────────
    print("\n=== Loading data ===")
    ref_data = {ref: load_reference(ref) for ref in args.refs}
    n_sgbs_by_ref = {ref: d["n_sgbs"] for ref, d in ref_data.items()}

    # ── Accumulation + fit (both models) ─────────────────────────────────
    print("\n=== Bootstrap accumulation & fitting ===")
    accum_results  = {}  # [ref][group] = accum 8-tuple
    fits_by_mname  = {}  # [ref][group][mname] = fit_dict
    boot_matrices  = {}  # [ref][group] = boot matrix
    for ref, rdat in ref_data.items():
        accum_results[ref] = {}; fits_by_mname[ref] = {}; boot_matrices[ref] = {}
        blocks = build_group_blocks(rdat["meta"], rdat["breadth"], THRESHOLD)
        for gname, block in blocks.items():
            print(f"  {ref} / {gname}...", end=" ", flush=True)
            acc   = bootstrap_curve(block, THRESHOLD, n_boot=N_BOOT, rng=rng)
            sem_v = np.maximum(acc[2]/np.sqrt(max(N_BOOT,1)), 0.1)
            accum_results[ref][gname] = acc
            boot_matrices[ref][gname] = acc[-1]
            fits_by_mname[ref][gname] = {}
            for mn in MODEL_VARIANTS:
                fits_by_mname[ref][gname][mn] = fit_one(mn, acc[0], acc[1], sem_v)
                r2 = fits_by_mname[ref][gname][mn].get("r2", np.nan)
            print(f"Richards R²={fits_by_mname[ref][gname]['Richards(logN)']['r2']:.4f}  "
                  f"StretchedExp R²={fits_by_mname[ref][gname]['StretchedExp']['r2']:.4f}")

    # ── Asymptote sweep (both models) ────────────────────────────────────
    print("\n=== Asymptote sweep ===")
    # sweep_results[ref][mname][gname] = (N_s, a_s, a_lo, a_hi, col, ls)
    sweep_results = {}
    for ref, rdat in ref_data.items():
        sweep_results[ref] = {mn: {} for mn in MODEL_VARIANTS}
        blocks = build_group_blocks(rdat["meta"], rdat["breadth"], THRESHOLD)
        for gname, block in blocks.items():
            acc  = accum_results[ref][gname]
            sem_v = np.maximum(acc[2]/np.sqrt(max(N_BOOT,1)), 0.1)
            bm   = boot_matrices[ref].get(gname)
            if bm is None or bm.size == 0: continue
            col = COL["Industrial"] if "Industrial" in gname else COL["Non-Industrial"]
            ls  = "-" if "Industrial" in gname else "--"
            for mn in MODEL_VARIANTS:
                print(f"  {ref} / {gname} / {mn}...", end=" ", flush=True)
                N_s, a_s, a_lo, a_hi = asymptote_sweep(mn, acc[0], bm, sem_v)
                sweep_results[ref][mn][gname] = (N_s, a_s, a_lo, a_hi, col, ls)
                print(f"done ({len(N_s)} pts)")

    # ── Panel E (both models) ─────────────────────────────────────────────
    print("\n=== Panel E: Hadza exclusion ===")
    ref_e  = args.refs[0]
    rdat_e = ref_data[ref_e]
    excl   = hadza_exclusive_sgbs(rdat_e["breadth"], rdat_e["meta"], THRESHOLD)
    blocks_e = build_group_blocks(rdat_e["meta"], rdat_e["breadth"], THRESHOLD)
    accum_E = {}; fits_E = {}
    for gname, block in blocks_e.items():
        if "Non-Industrial" in gname and len(excl) > 0:
            block     = block[[c for c in block.columns if c not in excl]]
            gname_lbl = f"{gname}\n(excl. Hadza-exclusive)"
        else:
            gname_lbl = gname
        print(f"  {gname_lbl}...", end=" ", flush=True)
        acc   = bootstrap_curve(block, THRESHOLD, n_boot=N_BOOT, rng=rng)
        sem_v = np.maximum(acc[2]/np.sqrt(max(N_BOOT,1)), 0.1)
        accum_E[gname_lbl] = acc
        fits_E[gname_lbl]  = {mn: fit_one(mn, acc[0], acc[1], sem_v) for mn in MODEL_VARIANTS}
        print("done")

    # ── Panel F ───────────────────────────────────────────────────────────
    print("\n=== Panel F ===")
    pf = panel_f_data()
    print(f"  S(8.2e9) = {pf['S_world']:,.0f}  [{pf['S_q025']:,.0f}–{pf['S_q975']:,.0f}]")

    # ── Build one figure per model ────────────────────────────────────────
    for mn in MODEL_VARIANTS:
        build_figure(
            mname=mn,
            accum_results=accum_results,
            fits_by_mname=fits_by_mname,
            boot_matrices=boot_matrices,
            sweep_results=sweep_results,
            accum_E=accum_E,
            fits_E=fits_E,
            pf=pf,
            n_sgbs_by_ref=n_sgbs_by_ref,
            threshold=THRESHOLD,
            n_boot=N_BOOT,
            out_stem=args.out,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
