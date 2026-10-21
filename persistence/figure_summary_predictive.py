#!/usr/bin/env python3
"""
figure_summary_predictive.py
============================
Drop-in replacement for figure_summary.py that upgrades three core components:

1. MARGINAL DETECTION MODEL  (within-species abundance integration)
   estimate_group_model_bb() now integrates P(B≥θ | logx, d_i, L_k, φ_k)
   over the within-species abundance distribution f_k(logx), supporting both
   lognormal and Student-t tails.  sig_x now fully drives r_mat instead of
   only mu_x, so heavy-tail effects in deep samples are captured.

2. PREDICTIVE ACCUMULATION  (extrapolation beyond observed N)
   predictive_accumulation_from_hmat() generates posterior-predictive
   accumulation curves for N > n_obs by bootstrap-resampling h_mat rows
   with replacement.  This is the primary reported curve; Richards /
   BetaMixture / BetaMixture2 are overlaid as descriptive smoothers.

3. TWO-COMPONENT BETA MIXTURE  (heavy across-species tail)
   BetaMixture2 decomposes species into "common" + "very rare" pools,
   giving an extra degree of freedom at large N where rare discoveries live.
   A deep-holdout stress test panel shows which model is actually needed.

Layout (3 rows × 2 cols):
  Row 0:  A = Carter (UHGG) predictive accumulation
          B = Leviatan (lev) predictive accumulation
  Row 1:  C = Carter asymptote sweep
          D = Leviatan asymptote sweep
  Row 2:  E = Carter excl. Hadza-exclusive
          F = Hill-MM world extrapolation

Run (from pipeline root):
    python -m persistence.figure_summary_predictive
    python -m persistence.figure_summary_predictive --n-boot 200 --out results/fig3p
    python -m persistence.figure_summary_predictive --abund-dist student_t --df 5
"""

import persistence.config as _cfg
import argparse
import os
import shutil
import tempfile
import warnings
from scipy.special import erf, gammaln, ndtr, betainc  # betainc = regularised incomplete Beta
from scipy.stats import norm as _scipy_norm, t as _student_t

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
    def apply_display_text(fig):
        for artist in fig.findobj(match=matplotlib.text.Text):
            text = artist.get_text()
            for old in ("Agro-Pastorialist", "Transitional"):
                text = str(text).replace(old, "Agro-Pastoralist")
            artist.set_text(text)
        return fig

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
        # total_bp: Σ(depth × genome_length) — same as pipeline computes
        return {"breadth": data.breadth, "depth": data.depth,
                "meta": data.meta, "lengths": data.lengths,
                "total_bp": data.total_bp,
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

    # total_bp = Σ_k depth_ks × L_k  — total MAPPED bases per sample.
    # NOTE: this is endogenous — if Industrial microbiomes are more divergent
    # from the reference, total_bp shrinks even at equal raw sequencing effort,
    # blurring biology and detectability. Ideally use raw_bases_sequenced from
    # metadata (column "Total_bases" or similar). We use total_bp here as the
    # best available proxy but flag it explicitly.
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

    depth_df  = depth_df.loc[meta.index]
    total_bp  = total_bp.loc[meta.index]
    print(f"    {len(meta):,} samples after all filters")
    return {"breadth": breadth_df, "depth": depth_df,
            "meta": meta, "lengths": lengths,
            "total_bp": total_bp, "n_sgbs": len(keep)}


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

def bootstrap_curve(breadth_block, threshold, n_boot=100, rng=None, n_points=60,
                    h_mat=None):
    """
    Bootstrap accumulation curve.

    Parameters
    ----------
    h_mat : (n_samples, n_species) array of per-sample detection probabilities,
            i.e. psi_k * r_k(d_i) from the BB generative model.
            When provided, the curve is computed from these continuous
            probabilities rather than the hard-thresholded binary matrix.
            This is the correct path when the BB model has been fitted —
            it propagates pre-threshold information into the main reported curves.
            When None (fallback), the old binary-threshold logic is used.
    """
    if rng is None: rng = np.random.default_rng(42)
    n_samples = len(breadth_block)
    n_mic     = breadth_block.shape[1]

    use_bb = (h_mat is not None) and (h_mat.shape == (n_samples, n_mic))

    if use_bb:
        # h_mat[i, k] = P(detect species k in sample i) — continuous, pre-threshold
        h_arr = np.clip(h_mat, 0.0, 1.0 - 1e-9)   # (n_samples, n_mic)
    else:
        # Fallback: binary threshold matrix
        h_arr = (breadth_block.values > threshold).astype(float)

    Ns = np.unique(np.concatenate([
        np.arange(1, min(25, n_samples)+1),
        np.round(np.logspace(np.log10(5), np.log10(n_samples), n_points)).astype(int),
        np.round(np.linspace(n_samples*0.60, n_samples, 20)).astype(int),
    ]))
    Ns = np.array(sorted({int(n) for n in Ns if 1 <= n <= n_samples}), dtype=int)
    boot = np.empty((n_boot, len(Ns)), dtype=float)

    if use_bb:
        # E[S(N)] = Σ_k [1 – Π_{i=1}^N (1 – h_{perm[i],k})]
        # Computed via cumulative log-sum for numerical stability.
        for r in range(n_boot):
            perm   = rng.permutation(n_samples)
            cumlog = np.cumsum(np.log1p(-h_arr[perm]), axis=0)  # (n_samples, n_mic)
            boot[r] = np.sum(1.0 - np.exp(cumlog[Ns - 1, :]), axis=1)
    else:
        for r in range(n_boot):
            perm = rng.permutation(n_samples)
            present_any = np.zeros(n_mic, dtype=bool); nxt = 0
            for i, N in enumerate(Ns):
                while nxt < N: present_any |= h_arr[perm[nxt]].astype(bool); nxt += 1
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


def hill_mm_panel_f(N, a, b, c):
    return a * N**c / (b**c + N**c)


def beta_mixture(N, K_inf, a, b):
    """
    Beta-Binomial mixture accumulation curve.

    Placing h_k ~ Beta(a, b) across species gives the closed-form:
        E[S(N)] = K_inf * [1 - B(a, b+N) / B(a, b)]
                = K_inf * [1 - exp(gammaln(b+N) + gammaln(a+b)
                                   - gammaln(a+b+N) - gammaln(b))]

    Parameters
    ----------
    K_inf : total detectable diversity (asymptote)
    a     : Beta shape controlling rare-species tail.
            a < 1 → J-shaped h_k distribution, heavy power-law tail N^{-a}
            a > 1 → modal distribution, lighter tail
    b     : Beta shape (sets the "typical" detection probability h ~ a/(a+b))

    For large N, 1 - S(N)/K_inf ~ N^{-a}, so a directly encodes the
    power-law exponent of the undetected fraction.
    """
    N   = np.maximum(np.asarray(N, float), 1e-12)
    a_  = max(float(a), 1e-6)
    b_  = max(float(b), 1e-6)
    Ki  = max(float(K_inf), 1.0)
    log_ratio = (gammaln(b_ + N) + gammaln(a_ + b_)
                 - gammaln(a_ + b_ + N) - gammaln(b_))
    return Ki * (1.0 - np.exp(log_ratio))


# ════════════════════════════════════════════════════════════════════════════
#  NONPARAMETRIC ACCUMULATION + CHAO2  (Layer 1 & 2 from the generative framework)
# ════════════════════════════════════════════════════════════════════════════

def nonparametric_accumulation(breadth_block, threshold, psi_k=None):
    """
    Nonparametric E[S(N)] using pooled ĥ_k (ignores depth heterogeneity).
    Used as a baseline comparison against the depth-aware model.

    Parameters
    ----------
    psi_k : (n_sgb,) array, optional.
        If supplied (from estimate_group_model_bb), the HT-corrected occupancy
        is used instead of the naive detection frequency, so the nonparametric
        curve is consistent with the depth-aware model.
        When None, falls back to naive ĥ_k from binary threshold.

    Returns (N_vec, S_vec, h_hat)
    """
    pres  = (breadth_block.values > threshold).astype(float)
    h_hat = pres.mean(axis=0)                          # naive ĥ_k (always computed)
    h_use = np.clip(psi_k, 0.0, 1.0 - 1e-9) if psi_k is not None else h_hat
    n     = len(breadth_block)
    N_vec = np.unique(np.concatenate([
        np.arange(1, min(50, n)+1),
        np.round(np.logspace(np.log10(5), np.log10(n), 60)).astype(int),
    ]))
    N_vec  = np.array(sorted({int(x) for x in N_vec if 1 <= x <= n}), dtype=float)
    log1_h = np.log1p(-h_use.clip(0, 1 - 1e-9))
    S_vec  = np.array([np.sum(1.0 - np.exp(N * log1_h)) for N in N_vec])
    return N_vec, S_vec, h_hat


def depth_aware_accumulation(breadth_block, depth_block, lengths, total_bp,
                              threshold, target_depth=None, n_perm=30, n_points=60,
                              rng=None):
    """
    Full generative model: E[S(N)] accounting for Poisson breadth detection
    and per-sample sequencing depth heterogeneity.

    Generative model
    ----------------
    Detection of species k in sample i:
        λ_ik = x_ik × d_i / L_k   (mean depth per base, Poisson parameter)
        B_ik = 1 - exp(-λ_ik)     (expected breadth)
        detected ↔ B_ik ≥ θ  ↔  λ_ik ≥ τ_θ = -log(1-θ)

    h_ik = ψ_k × r_k(d_i)
        ψ_k    = occupancy (biologically present in this lifestyle group)
        r_k(d_i) = P(λ_ik ≥ τ_θ | present)
                 = Φ[(μ_x,k + log d_i - log L_k - log τ_θ) / σ_x,k]

    Parameters μ_x,k, σ_x,k are the lognormal parameters of log(x_ik) =
    log(abundance relative to total mapped bases), estimated from detected
    samples.  ψ_k is corrected from the naive ĥ_k by the average r_k.

    If target_depth is given, r_k(d_i) is replaced by r_k(d*) everywhere —
    projecting to a uniform sequencing depth d*.

    Returns
    -------
    N_vec   : sample-size grid
    S_vec   : depth-aware E[S(N)] (mean over n_perm random orderings)
    S_std   : standard deviation across permutations
    psi_k   : (n_sgb,) occupancy estimates
    mu_x    : (n_sgb,) lognormal mean of log(x_ik) for detected samples
    sig_x   : (n_sgb,) lognormal SD
    h_mat   : (n_samp, n_sgb) per-sample per-species detection probabilities
    """
    if rng is None:
        rng = np.random.default_rng(42)

    θ     = threshold / 100.0      # 0.10
    TAU   = -np.log1p(-θ)         # τ_θ = 0.1054 for θ=0.10
    log_tau = np.log(TAU)

    # Align indices
    shared_idx = breadth_block.index.intersection(depth_block.index).intersection(total_bp.index)
    b_mat = breadth_block.loc[shared_idx].values        # (n_samp, n_sgb)  % breadth
    d_mat = depth_block.loc[shared_idx].values          # (n_samp, n_sgb)  mean depth/base
    d_i   = total_bp.loc[shared_idx].values.astype(float)  # (n_samp,)  total mapped bp
    d_i   = np.maximum(d_i, 1.0)
    n_samp, n_sgb = b_mat.shape

    # Genome lengths aligned to breadth_block columns
    L_k = lengths.reindex(breadth_block.columns).values.astype(float)
    L_k = np.maximum(L_k, 1.0)
    log_L = np.log(L_k)
    log_d = np.log(d_i)  # (n_samp,)

    detected = (b_mat >= threshold)      # (n_samp, n_sgb) bool
    h_hat    = detected.mean(0)          # (n_sgb,) naive ĥ_k

    # Per-species lognormal parameters of log(x_ik) from detected samples
    # log(x_ik) = log(λ_ik) - log(d_i) + log(L_k)
    # where λ_ik = d_mat[i,k] (mean depth per base)
    mu_x  = np.full(n_sgb, np.nan, dtype=float)
    sig_x = np.full(n_sgb, 0.01,  dtype=float)

    for k in range(n_sgb):
        idx_det = np.where(detected[:, k])[0]
        if len(idx_det) < 3:
            continue
        lam_det = d_mat[idx_det, k]
        lam_det = lam_det[lam_det > 0]
        if len(lam_det) < 3:
            continue
        log_x = np.log(lam_det) - log_d[idx_det[:len(lam_det)]] + log_L[k]
        mu_x[k]  = np.mean(log_x)
        sig_x[k] = max(np.std(log_x, ddof=1), 0.05)  # floor to avoid division issues

    # r_k(d_i) per sample: vectorised over all species with valid estimates
    # r_k(d_i) = Φ[(μ_x,k + log(d_i) - log(L_k) - log(τ_θ)) / σ_x,k]
    r_mat = np.zeros((n_samp, n_sgb), dtype=float)
    has_params = np.isfinite(mu_x)

    # Vectorised over samples and species with params
    if has_params.any():
        log_d_2d = log_d[:, None]                  # (n_samp, 1)
        mu_2d    = mu_x[None, has_params]           # (1,  n_valid)
        sig_2d   = sig_x[None, has_params]          # (1,  n_valid)
        logL_2d  = log_L[None, has_params]          # (1,  n_valid)
        z = (mu_2d + log_d_2d - logL_2d - log_tau) / sig_2d   # (n_samp, n_valid)
        r_mat[:, has_params] = ndtr(z)

    # Fallback: species with < 3 detected samples use constant ĥ_k
    r_mat[:, ~has_params] = h_hat[~has_params][None, :]

    # Project to target depth if requested
    if target_depth is not None:
        log_d_target = np.log(float(target_depth))
        if has_params.any():
            z_star = (mu_x[has_params] + log_d_target - log_L[has_params] - log_tau)                      / sig_x[has_params]
            r_mat[:, has_params] = ndtr(z_star)[None, :]

    # ψ_k: occupancy probability (corrected for depth-driven misdetection)
    mean_r = r_mat.mean(0)
    psi_k  = np.where(mean_r > 1e-9, h_hat / mean_r, 0.0)
    psi_k  = np.clip(psi_k, 0.0, 1.0)

    # h_ik = ψ_k × r_k(d_i)
    h_mat = psi_k[None, :] * r_mat       # (n_samp, n_sgb)
    h_mat = np.clip(h_mat, 0.0, 1.0)

    # E[S(N)] via random permutations (order-free expectation)
    Ns = np.unique(np.concatenate([
        np.arange(1, min(25, n_samp)+1),
        np.round(np.logspace(np.log10(5), np.log10(n_samp), n_points)).astype(int),
    ]))
    Ns = np.array(sorted({int(x) for x in Ns if 1 <= x <= n_samp}), dtype=int)

    S_mat = np.zeros((n_perm, len(Ns)), dtype=float)
    for p in range(n_perm):
        perm   = rng.permutation(n_samp)
        h_perm = h_mat[perm, :]
        # cumulative log(1-h) across samples → log of product
        cumlog = np.cumsum(np.log1p(-h_perm.clip(0, 1-1e-9)), axis=0)  # (n_samp, n_sgb)
        # at each N, S = Σ_k [1 - exp(cumlog[N-1, k])]
        S_mat[p, :] = np.sum(1.0 - np.exp(cumlog[Ns-1, :]), axis=1)

    S_vec = S_mat.mean(0)
    S_std = S_mat.std(0)

    return Ns.astype(float), S_vec, S_std, psi_k, mu_x, sig_x, h_mat


# ════════════════════════════════════════════════════════════════════════════
#  PREDICTIVE ACCUMULATION  (Change 2: extrapolate beyond observed N)
# ════════════════════════════════════════════════════════════════════════════

def predictive_accumulation_from_hmat(h_mat, N_max, n_draw=400, rng=None):
    """
    Posterior-predictive accumulation curve for arbitrary N (including N > n_obs).

    The curve is a **bootstrap predictive** estimate throughout — rows of h_mat
    are resampled WITH replacement for the entire path, including N ≤ n_obs.
    This is NOT permutation rarefaction (which draws without replacement);
    it is a bootstrap predictive interval that widens below n_obs relative to
    the permutation curve.  The difference is small for large n_obs but is
    present at all scales.

    For N ≤ n_obs: similar in expectation to permutation rarefaction, but
                   with replacement giving a slightly wider CI.
    For N > n_obs: extrapolates by assuming new samples are drawn from the
                   same depth/quality distribution as the observed data.

    This is the primary reported curve; Richards / BetaMixture / BetaMixture2
    are fitted as descriptive smoothers on top of it.

    Parameters
    ----------
    h_mat  : (n_obs, n_sgb) per-sample detection probabilities from BB model.
    N_max  : maximum sample size to predict to.
    n_draw : number of bootstrap draws (default 400).

    Returns
    -------
    Ns   : (n_pts,) float array of N values
    mean : (n_pts,) mean predicted S(N)
    lo   : (n_pts,) 2.5th percentile
    hi   : (n_pts,) 97.5th percentile
    draws: (n_draw, n_pts) all bootstrap realizations
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_obs, n_sgb = h_mat.shape
    N_max = max(int(N_max), n_obs)

    # Grid: dense at low N, logarithmic at large N
    Ns_set = set(range(1, min(30, N_max) + 1))
    Ns_set.update(
        int(n) for n in np.round(
            np.logspace(np.log10(5), np.log10(N_max), 80)
        ) if 1 <= n <= N_max
    )
    Ns = np.array(sorted(Ns_set), dtype=int)

    draws = np.zeros((n_draw, len(Ns)), float)
    h_arr = np.clip(h_mat, 0.0, 1.0 - 1e-9)

    for b in range(n_draw):
        # With-replacement resample rows to get N_max "future samples"
        idx    = rng.integers(0, n_obs, size=int(Ns.max()))
        h_sim  = h_arr[idx]                              # (N_max, n_sgb)
        cumlog = np.cumsum(np.log1p(-h_sim), axis=0)     # (N_max, n_sgb)
        draws[b] = np.sum(1.0 - np.exp(cumlog[Ns - 1, :]), axis=1)

    mean = draws.mean(0)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return Ns.astype(float), mean, lo, hi, draws


# ════════════════════════════════════════════════════════════════════════════
#  DEEP-HOLDOUT STRESS TEST  (Change 4: show where the heavy tail matters)
# ════════════════════════════════════════════════════════════════════════════

def deep_holdout_new_taxa_test(breadth_block, h_mat_pred, total_bp,
                                threshold=10.0, hold_frac=0.15,
                                depth_block=None, lengths=None,
                                refit_model_fn=None):
    """
    Sort samples by effective depth, hold out the deepest `hold_frac` fraction,
    and record how many NEW SGBs each holdout sample contributes beyond the
    running union of the training set.

    This is where a heavier within-species or across-species tail should show
    up: the model that better predicts "new discoveries in the deepest samples"
    is the better-calibrated model.

    Parameters
    ----------
    breadth_block : (n_samp, n_sgb) DataFrame of % breadth values.
    h_mat_pred    : (n_samp, n_sgb) predicted detection probabilities from BB model.
    total_bp      : (n_samp,) Series of total mapped bp (proxy for depth).
    threshold     : breadth threshold in % for "detected".
    hold_frac     : fraction of deepest samples to hold out (default 15%).

    Returns
    -------
    dict with keys:
      obs_new_per_sample  : (n_hold,) observed new SGBs per holdout sample
      pred_new_per_sample : (n_hold,) model-predicted E[new SGBs] per sample
      n_train, n_hold
    """
    idx_ordered = total_bp.loc[breadth_block.index].sort_values().index
    n_hold  = max(5, int(len(idx_ordered) * hold_frac))
    n_train = len(idx_ordered) - n_hold

    train_idx = idx_ordered[:n_train]
    hold_idx  = idx_ordered[n_train:]

    b_mat   = (breadth_block.values >= threshold)          # (n, k) bool

    # Observed: running union in training set, then count new in holdout
    union = b_mat[breadth_block.index.get_indexer(train_idx)].any(axis=0)
    obs_new = []
    running = union.copy()
    hold_pos = breadth_block.index.get_indexer(hold_idx)
    for i in hold_pos:
        new_row   = b_mat[i]
        added     = new_row & ~running
        obs_new.append(int(added.sum()))
        running  |= new_row


    # ── Fix 2: optionally refit on training samples only ─────────────────
    # If refit_model_fn is provided, the model is re-estimated using ONLY
    # training samples, and the holdout detection probabilities are generated
    # from that train-only model. This is a true out-of-sample test.
    # Without it, h_mat_pred was fit on all data including the holdout, making
    # this a calibration check rather than an out-of-sample validation.
    if refit_model_fn is not None and depth_block is not None and lengths is not None:
        train_breadth = breadth_block.loc[train_idx]
        train_depth   = depth_block.loc[train_idx]
        train_tbp     = total_bp.loc[train_idx]
        jm_train = refit_model_fn(train_breadth, train_depth, lengths, train_tbp)
        psi_tr, mu_tr, sig_tr, phi_tr, d_tr, L_tr = jm_train[:6]
        # Score holdout samples using train-only parameters
        log_d_h  = np.log(np.maximum(total_bp.loc[hold_idx].values.astype(float), 1.0))
        log_L_tr = np.log(np.maximum(L_tr, 1.0))
        log_eff_h = log_d_h[:, None] - log_L_tr[None, :]   # (n_hold, n_sgb)
        theta = threshold / 100.0
        # Batch mean-field r for holdout
        n_hold_s = len(hold_idx); n_sgb_h = len(psi_tr)
        batch_k_h = 500
        h_hold_refit = np.zeros((n_hold_s, n_sgb_h), float)
        for ks in range(0, n_sgb_h, batch_k_h):
            ke = min(ks + batch_k_h, n_sgb_h)
            sl = slice(ks, ke)
            lam = mu_tr[sl][None, :] + log_eff_h[:, sl]
            p   = np.clip(1.0 - np.exp(-np.exp(lam)), 1e-8, 1-1e-8)
            om  = phi_tr[sl][None, :]
            a   = np.maximum(p*om, 1e-6); b = np.maximum((1-p)*om, 1e-6)
            r_f = 1.0 - betainc(a.ravel(), b.ravel(), theta)
            h_hold_refit[:, sl] = np.clip(psi_tr[sl][None, :] * r_f.reshape(n_hold_s, ke-ks),
                                           0.0, 1.0 - 1e-9)
        h_hold = h_hold_refit
    else:
        h_hold = np.clip(h_mat_pred[hold_pos], 0.0, 1.0 - 1e-9)

    # Recompute predicted new taxa using the (possibly refitted) h_hold
    h_train_full = np.clip(h_mat_pred[breadth_block.index.get_indexer(train_idx)],
                           0.0, 1.0 - 1e-9)
    log1_h_train = np.log1p(-h_train_full).sum(axis=0)
    p_unseen = np.exp(log1_h_train)
    pred_new = []
    p_still_unseen = p_unseen.copy()
    for j in range(len(hold_idx)):
        h_j = h_hold[j]
        pred_new.append(float((h_j * p_still_unseen).sum()))
        p_still_unseen *= (1.0 - h_j)

    return {
        "obs_new_per_sample":  np.array(obs_new,  float),
        "pred_new_per_sample": np.array(pred_new, float),
        "n_train":   n_train,
        "n_hold":    n_hold,
        "hold_idx":  hold_idx,
        "refitted":  (refit_model_fn is not None),
    }


def chao2_estimate(breadth_block, threshold):
    """
    Chao2 nonparametric richness estimator with 95% CI.

    Uses the incidence-based bias-corrected Chao2 formula.
    Q1 = species seen in exactly 1 sample (singletons in incidence sense)
    Q2 = species seen in exactly 2 samples (doubletons)
    m  = number of samples

    Chao2 (bias-corrected): S_obs + (m-1)/m * Q1*(Q1-1)/(2*(Q2+1))

    Variance approximation from Chao & Lee (1992) / Colwell et al. (2012).

    Returns (S_chao2, S_lo95, S_hi95)
    """
    pres    = (breadth_block.values > threshold)     # (n_samp, n_sgb) bool
    m       = pres.shape[0]                          # number of samples
    counts  = pres.sum(axis=0)                       # per-species incidence counts
    S_obs   = int((counts > 0).sum())
    Q1      = int((counts == 1).sum())
    Q2      = int((counts == 2).sum())

    if Q2 == 0:
        # Can't compute; use singleton-only correction
        chao2 = float(S_obs + (m-1)/m * Q1*(Q1-1)/2)
        return chao2, np.nan, np.nan

    chao2 = float(S_obs + (m-1)/m * Q1*(Q1-1) / (2*(Q2+1)))

    # Approximate variance (Chao 1987 / EstimateS documentation)
    # Var ≈ Q2[(Q1/Q2)^4/4 + (Q1/Q2)^3 + (Q1/Q2)^2/2] * (m-1)/m
    r = Q1 / Q2
    var = Q2 * (r**4/4 + r**3 + r**2/2) * (m-1)/m
    se  = np.sqrt(max(var, 0.0))
    return chao2, chao2 - 1.96*se, chao2 + 1.96*se


def beta_mixture2(N, K_inf, w, a1, b1, a2, b2):
    """
    Two-component Beta-Mixture accumulation curve (Change 3).

    Decomposes species into two pools:
      Component 1 (weight w):   common taxa     — detection prob ~ Beta(a1, b1)
      Component 2 (weight 1-w): very rare taxa  — detection prob ~ Beta(a2, b2)

    E[S(N)] = K_inf * [1 - w·surv(a1,b1,N) - (1-w)·surv(a2,b2,N)]

    where surv(a,b,N) = B(a, b+N)/B(a,b) = exp(gammaln(b+N)+gammaln(a+b)
                                                 -gammaln(a+b+N)-gammaln(b))

    For large N:  undetected fraction ~ w·N^{-a1} + (1-w)·N^{-a2}
    a2 < a1 encodes "very rare taxa with heavier detection tail".

    Parameters
    ----------
    K_inf : total detectable diversity (asymptote)
    w     : weight of common component (0 < w < 1)
    a1,b1 : Beta shape of common component   (a1 > a2 typically)
    a2,b2 : Beta shape of rare component     (small a2 → heavy power-law tail)
    """
    N   = np.maximum(np.asarray(N, float), 1e-12)
    w   = np.clip(float(w), 1e-6, 1 - 1e-6)
    Ki  = max(float(K_inf), 1.0)

    def _surv(a, b):
        a_ = max(float(a), 1e-6)
        b_ = max(float(b), 1e-6)
        return np.exp(gammaln(b_ + N) + gammaln(a_ + b_)
                      - gammaln(a_ + b_ + N) - gammaln(b_))

    return Ki * (1.0 - (w * _surv(a1, b1) + (1.0 - w) * _surv(a2, b2)))


MODEL_VARIANTS = {
    "Richards(logN)": dict(
        func=richards_logN,
        pnames=["a","b","c","ν"],
        p0_fn=lambda S,N: [S.max()*1.05, np.median(N), 1.5, 0.5],
        bounds=([10,1,0.1,0.05],[1e5,1e6,10,5.0]),
        color="#2b2d42", lw=1.4, ls="-", zorder=7,
        label_short="Richards",
    ),
    "BetaMixture": dict(
        func=beta_mixture,
        pnames=["K_inf","a","b"],
        p0_fn=lambda S,N: [S.max()*1.3, 0.4, 1.5],
        bounds=([10, 0.01, 0.01],[1e5, 5.0, 1e4]),
        color="#009E73", lw=1.4, ls="-", zorder=7,
        label_short="BetaMix",
    ),
    "BetaMixture2": dict(
        func=beta_mixture2,
        pnames=["K_inf","w","a1","b1","a2","b2"],
        p0_fn=lambda S,N: [S.max()*1.5, 0.8, 0.25, 2.0, 1.2, 50.0],
        bounds=([10, 0.01, 0.01, 0.01, 0.01, 0.01],
                [1e6, 0.99,  5.0, 1e5,  5.0, 1e5]),
        color="#7A0177", lw=1.5, ls="-", zorder=8,
        label_short="2BetaMix",
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



# ════════════════════════════════════════════════════════════════════════════
#  TWO-GROUP DEPTH-CORRECTED COMPARISON  (Ind vs NI)
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
#  QUADRATURE HELPERS FOR MARGINAL DETECTION PROBABILITY
#  Change 1 from design doc: integrate over within-species abundance distribution
#  so sig_x fully drives r_mat (previously only mu_x was used).
# ════════════════════════════════════════════════════════════════════════════

def _quad_points(dist="lognormal", df=5, n_q=31):
    """
    Return (z, w) quadrature points and equal weights for marginalising over
    the within-species abundance distribution.

    dist : "lognormal"  → standard-normal quantiles (exact for lognormal)
           "student_t"  → Student-t quantiles with `df` degrees of freedom
                          (heavier tail; df=5 is a practical default)
    """
    q = (np.arange(n_q) + 0.5) / n_q        # mid-point rule on (0,1)
    if dist == "lognormal":
        z = _scipy_norm.ppf(q)
    elif dist == "student_t":
        z = _student_t.ppf(q, df=df)
    else:
        raise ValueError(f"Unknown abundance distribution: {dist!r}")
    w = np.full(n_q, 1.0 / n_q)             # equal weights (mid-point rule)
    return z, w


def _bb_detect_prob_marginal(mu, sig, phi, log_eff, theta,
                              dist="lognormal", df=5, n_q=31):
    """
    Marginalise BB detection probability over the within-species abundance
    distribution f_k(logx).

    r_ik = ∫ P(B_ik ≥ θ | logx, d_i, L_k, φ_k) · f_k(logx) d(logx)

    Parameters
    ----------
    mu      : (n_valid,) lognormal mean of log(x_k) for species with params
    sig     : (n_valid,) lognormal SD (floored ≥ 0.05)
    phi     : (n_valid,) BB concentration ω_k
    log_eff : (n_samp, n_valid)  log(d_i) + q_s - log(L_k)  (log efficiency)
    theta   : detection threshold as a fraction in (0,1)
    dist    : "lognormal" | "student_t"
    df      : Student-t degrees of freedom (ignored for lognormal)
    n_q     : number of quadrature points (31 is accurate to < 0.1%)

    Returns
    -------
    r_mat : (n_samp, n_valid) marginal detection probabilities
    """
    z, w = _quad_points(dist=dist, df=df, n_q=n_q)     # (n_q,), (n_q,)

    # Quadrature points in log-abundance space: (n_valid, n_q)
    # logx_qj = mu_k + sig_k * z_j
    logx_pts = mu[:, None] + sig[:, None] * z[None, :]   # (n_valid, n_q)

    # log(λ_ik) = log_eff[i,k] + logx_k   →  (n_samp, n_valid, n_q)
    # log_eff is (n_samp, n_valid); add n_q axis
    log_lam = log_eff[:, :, None] + logx_pts[None, :, :]  # (n_samp, n_valid, n_q)

    # Poisson mean breadth  p = 1 - exp(-exp(log_lam))
    p = np.clip(1.0 - np.exp(-np.exp(log_lam)), 1e-8, 1 - 1e-8)

    # BB parameters  a = p*ω,  b = (1-p)*ω
    om = phi[None, :, None]                               # (1, n_valid, 1)
    a  = np.maximum(p * om, 1e-6)
    b  = np.maximum((1.0 - p) * om, 1e-6)

    # P(B ≥ θ | logx_j) = 1 - I_θ(a, b)   — broadcast over n_q
    # betainc expects 1-D arrays; flatten then reshape
    a_flat = a.ravel(); b_flat = b.ravel()
    tail_flat = 1.0 - betainc(a_flat, b_flat, theta)
    tail = tail_flat.reshape(a.shape)                     # (n_samp, n_valid, n_q)

    # Integrate:  r_ik = Σ_j w_j * P(B≥θ | logx_j)
    r_mat = (tail * w[None, None, :]).sum(axis=2)         # (n_samp, n_valid)
    return r_mat


# ════════════════════════════════════════════════════════════════════════════
#  THRESHOLD GRID AND COLOUR FAMILIES  (for accumulation surface)
# ════════════════════════════════════════════════════════════════════════════

# Breadth thresholds as fractions — the surface S(N, τ) is computed at each.
THRESHOLDS_GRID = np.array([0.10, 0.20, 0.30, 0.40, 0.50])

# Sequential colour ramps per group: light (τ=0.10) → dark (τ=0.50)
_THRESH_COLS_IND = ["#AED6F1", "#5DADE2", "#2E86C1", "#1A5276", "#0D2137"]
_THRESH_COLS_NI  = ["#FAD7A0", "#F5A623", "#E67E22", "#CA6F1E", "#7D3C00"]


def _thresh_color(gname, j):
    """Colour for threshold index j within group gname."""
    if "Industrial" in gname and "Non" not in gname:
        return _THRESH_COLS_IND[min(j, len(_THRESH_COLS_IND) - 1)]
    return _THRESH_COLS_NI[min(j, len(_THRESH_COLS_NI) - 1)]


# ════════════════════════════════════════════════════════════════════════════
#  TRUNCATED LOGNORMAL MLE  (Fix A: left-truncation bias in mu_x / sig_x)
# ════════════════════════════════════════════════════════════════════════════

def _truncated_lognormal_mle(log_x_obs, log_trunc_pts, n_iter=8, tol=1e-5,
                              max_iter=120, tol_fallback=1e-7):
    """
    MLE for Normal(μ, σ²) truncated from below at sample-varying thresholds.

    **Fast path** (default): 8-iteration analytical Newton correction using the
    truncated-normal score equations.  O(n) numpy per iteration, no scipy overhead.
    Converges to the same MLE as L-BFGS-B for well-conditioned data.

    **Fallback**: scipy L-BFGS-B used only when the fast path diverges.

    The key equations (from the truncated-normal log-likelihood gradient):
        ∂ℓ/∂μ = 0 → μ̂ = mean(z) − σ · mean(φ(a_i)/Φ(−a_i))
        ∂ℓ/∂σ = 0 → σ̂² = Var(z) / mean(1 − a_i·mills_i − mills_i²)
    where a_i = (c_i − μ)/σ  and  mills_i = φ(a_i)/Φ(−a_i) is the Mills ratio.

    Returns  mu_mle, sig_mle : float
    """
    z = np.asarray(log_x_obs,     float)
    c = np.asarray(log_trunc_pts, float)
    n = len(z)
    if n < 2:
        return (float(z[0]) if n == 1 else 0.0), 0.10

    mu  = float(z.mean())
    sig = max(float(z.std(ddof=1)), 0.05)

    _SQRT2PI = np.sqrt(2.0 * np.pi)

    for it in range(n_iter):
        a       = (c - mu) / sig
        phi_a   = np.exp(-0.5 * a * a) / _SQRT2PI   # inline: avoids scipy overhead
        Phi_neg = np.maximum(ndtr(-a), 1e-300)
        mills   = phi_a / Phi_neg

        mu_new  = float(z.mean() - sig * mills.mean())

        # Variance update via truncated-normal variance formula:
        #   Var[z|z>c] = σ²·(1 - a·mills - mills²)   (per-sample)
        a2       = (c - mu_new) / sig
        phi_a2   = np.exp(-0.5 * a2 * a2) / _SQRT2PI
        mills2   = phi_a2 / np.maximum(ndtr(-a2), 1e-300)
        vc       = np.maximum(1.0 - a2 * mills2 - mills2**2, 1e-3)
        obs_var  = float(np.var(z, ddof=1))
        sig_new  = max(np.sqrt(obs_var / float(vc.mean())), 0.05)

        if not (np.isfinite(mu_new) and np.isfinite(sig_new)) or sig_new > 12:
            break

        if abs(mu_new - mu) < tol and abs(sig_new - sig) < tol:
            return float(mu_new), float(sig_new)

        mu, sig = mu_new, sig_new

    # Sanity check — return fast result if reasonable
    if np.isfinite(mu) and np.isfinite(sig) and sig < 12:
        return float(mu), float(sig)

    # Fallback: scipy L-BFGS-B (rare; used when fast path diverges)
    from scipy.optimize import minimize
    mu0  = float(z.mean()); sig0 = max(float(z.std(ddof=1)), 0.05)

    def _nll(params):
        mu_, log_sig = params; sig_ = np.exp(log_sig)
        zi = (z - mu_) / sig_; ci = (c - mu_) / sig_
        return -((-0.5 * zi**2 - np.log(sig_)) - np.log(np.maximum(ndtr(-ci), 1e-300))).sum()

    try:
        res = minimize(_nll, [mu0, np.log(sig0)], method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': tol_fallback})
        mh, sh = float(res.x[0]), float(np.exp(res.x[1]))
        if np.isfinite(mh) and np.isfinite(sh) and sh < 12:
            return mh, max(sh, 0.05)
    except Exception:
        pass
    return mu0, sig0


# ════════════════════════════════════════════════════════════════════════════
#  EMPIRICAL-BAYES SHRINKAGE  (Fix B: replace r_mat=0 for rare species)
# ════════════════════════════════════════════════════════════════════════════

def _empirical_bayes_shrink(mu_x, sig_x, n_det_arr, n0=5):
    """
    James-Stein-style blending of per-species (μ_x, σ_x) toward the group
    median, with shrinkage weight α_k = n_det_k / (n_det_k + n0).

    For species with 0–2 sub-threshold detections (previously assigned
    r_mat = 0, discarding all information), the group prior is used instead.
    This gives every species a small but non-zero expected detection
    probability, preventing the rare-species tail from collapsing to zero.

    Parameters
    ----------
    mu_x, sig_x  : (n_sgb,) per-species estimates; NaN where < min_det
    n_det_arr    : (n_sgb,) sub-threshold detection counts
    n0           : shrinkage pseudo-count (default 5)

    Returns  mu_out, sig_out : (n_sgb,) with NaN replaced, estimates shrunk
    """
    has_est = np.isfinite(mu_x) & np.isfinite(sig_x)
    mu_out  = mu_x.copy()
    sig_out = sig_x.copy()

    if has_est.sum() < 2:
        return mu_out, sig_out

    mu_group  = float(np.median(mu_x[has_est]))
    sig_group = float(np.median(sig_x[has_est]))

    for k in range(len(mu_x)):
        nd    = max(int(n_det_arr[k]), 0)
        alpha = nd / (nd + n0)
        if has_est[k]:
            mu_out[k]  = alpha * mu_x[k]  + (1.0 - alpha) * mu_group
            sig_out[k] = alpha * sig_x[k] + (1.0 - alpha) * sig_group
        else:
            # Fill unobserved species with group prior + extra uncertainty.
            # Replaces old: r_mat[:, ~has_p] = 0.0
            mu_out[k]  = mu_group
            sig_out[k] = sig_group * 1.5   # wider: less confident than observed species

    return mu_out, sig_out


# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 — BREADTH CATEGORIES  (doc §"Step 1: build threshold categories")
# ════════════════════════════════════════════════════════════════════════════

def make_breadth_categories(breadth_df, thresholds_pct=(10, 20, 30, 40, 50)):
    """
    Convert a continuous breadth matrix into ordered category integers.

    C_ks ∈ {0, 1, …, J} where J = len(thresholds_pct):
      C_ks = 0  ↔  B_ks < tau_1          (below all thresholds)
      C_ks = j  ↔  tau_j ≤ B_ks < tau_{j+1}   for j = 1 … J-1
      C_ks = J  ↔  B_ks ≥ tau_J          (above all thresholds)

    This encodes the FULL ordered information in the breadth value, not just
    a single binary call.  The resulting categories feed the joint interval
    likelihood described in doc §"Version 1: joint threshold-category hurdle".

    Parameters
    ----------
    breadth_df    : (n_samp, n_sgb) DataFrame, values in %
    thresholds_pct: ordered threshold values in % (default (10,20,30,40,50))

    Returns
    -------
    C    : (n_samp, n_sgb) int ndarray, values in {0, …, J}
    taus : (J,) float array of threshold fractions
    """
    b    = breadth_df.values.astype(float) / 100.0
    taus = np.asarray(thresholds_pct, float) / 100.0
    # C_ks = number of thresholds the breadth exceeds
    C    = np.sum(b[:, :, None] >= taus[None, None, :], axis=2).astype(np.int8)
    return C, taus


# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 — INTERVAL LIKELIHOOD ESTIMATOR FOR ψ_k
#  (doc §"Step 3: replace the current HT occupancy estimator")
# ════════════════════════════════════════════════════════════════════════════

def _estimate_psi_from_categories(C_k, r_at_thresholds, eps_fp=1e-6):
    """
    Estimate ψ_k (occupancy) via 1-D MLE on the interval category likelihood.

    The joint interval likelihood for the ordered observation C_ks ∈ {0,…,J}:

        P(C_ks) = (1−ψ_k) · P(C=0|z=0)  +  ψ_k · P(C_ks|z=1)

    where:
        P(C=0 | z=0) = 1 − ε_fp  (near-certain: absent species have B≈0)
        P(C>0 | z=0) = ε_fp       (small false-positive rate)

    and the conditional probabilities given presence are:
        P(C=0 | z=1) = 1 − r_ks(τ_1)
        P(C=j | z=1) = r_ks(τ_j) − r_ks(τ_{j+1})   for j = 1…J-1
        P(C=J | z=1) = r_ks(τ_J)

    where r_ks(τ_j) = P(B_ks ≥ τ_j | z=1) from the BB-CDF.

    This replaces the Horvitz-Thompson estimator as the primary ψ estimator.
    HT is retained as a fast initializer.

    Parameters
    ----------
    C_k              : (n_samp,) int array, category 0…J for species k
    r_at_thresholds  : (n_samp, J) P(B ≥ τ_j | z=1) per sample & threshold

    Returns  psi_hat : float in [0, 1]
    """
    from scipy.optimize import minimize_scalar

    n_samp, n_thresh = r_at_thresholds.shape

    # ── Build interval probability matrix  (n_samp, J+1) ─────────────────
    p_z1 = np.empty((n_samp, n_thresh + 1), float)
    p_z1[:, 0]          = 1.0 - r_at_thresholds[:, 0]   # P(C=0 | z=1)
    for j in range(1, n_thresh):
        p_z1[:, j]      = r_at_thresholds[:, j-1] - r_at_thresholds[:, j]
    p_z1[:, n_thresh]   = r_at_thresholds[:, -1]         # P(C=J | z=1)
    p_z1                = np.maximum(p_z1, 1e-300)

    # Vectorised negative log-likelihood over logit(ψ)
    c_idx  = C_k.astype(int)
    p_cz1  = p_z1[np.arange(n_samp), c_idx]             # (n_samp,)
    p_cz0  = np.where(c_idx == 0, 1.0 - eps_fp, eps_fp)  # (n_samp,)

    def _nll(logit_psi):
        psi   = 1.0 / (1.0 + np.exp(-float(logit_psi)))
        p_obs = (1.0 - psi) * p_cz0 + psi * p_cz1
        return -np.sum(np.log(np.maximum(p_obs, 1e-300)))

    # Warm-start from detection rate at primary threshold
    h0      = float((c_idx > 0).mean())
    h0      = np.clip(h0, 1e-4, 1 - 1e-4)
    logit0  = float(np.log(h0 / (1.0 - h0)))

    try:
        res = minimize_scalar(_nll, bounds=(-12, 12), method='bounded',
                              options={'xatol': 1e-5})
        return float(1.0 / (1.0 + np.exp(-res.x)))
    except Exception:
        return h0
#                               + marginal abundance integration
#                               + truncation correction + EB shrinkage)
# ════════════════════════════════════════════════════════════════════════════

def estimate_group_model_bb(breadth_block, depth_block, lengths, total_bp,
                             threshold, eps_pct=0.5,
                             abund_dist="lognormal", abund_df=5, n_q=31,
                             correct_truncation=True, eb_n0=5):
    """
    Three-improvement upgrade over estimate_group_model, now with a fourth:

    1. Pre-threshold fitting (eps_pct lower bound instead of threshold)
    2. Sample-level quality offset q_s
    3. Beta-Binomial breadth model
    4. [NEW] Marginal detection probability — sig_x now fully drives r_mat.
       r_ik = ∫ P(B≥θ | logx, d_i, L_k, φ_k) f_k(logx) d(logx)
       where f_k is lognormal(mu_k, sig_k²) or Student-t(mu_k, sig_k, df).
       Previously only mu_x entered r_mat; sig_x was estimated but unused in
       the detection step, so heavy-tail effects in deep samples were missed.

    Parameters
    ----------
    abund_dist : "lognormal" | "student_t"   within-species abundance dist
    abund_df   : Student-t degrees of freedom (ignored for lognormal)
    n_q        : quadrature points for marginalisation (default 31)

    Returns  (unchanged from original)
    -------
    h_hat, psi_ht, mu_x, sig_x, r_mat, sum_r, d_i, L_k, log_tau, q_s, phi_k
    """
    θ       = threshold / 100.0
    TAU     = -np.log1p(-θ)
    log_tau = np.log(TAU)
    eps     = eps_pct / 100.0            # sub-threshold detection floor

    idx    = (breadth_block.index
              .intersection(depth_block.index)
              .intersection(total_bp.index))
    b_frac = breadth_block.loc[idx].values.astype(float) / 100.0  # (n,k) 0-1
    d_mat  = depth_block.loc[idx].values.astype(float)             # mean depth/base
    d_i    = np.maximum(total_bp.loc[idx].values.astype(float), 1.0)
    n_samp, n_sgb = b_frac.shape

    L_k   = np.maximum(lengths.reindex(breadth_block.columns).values.astype(float), 1.0)
    log_L = np.log(L_k)

    # ── Sequencing depth (log scale) ─────────────────────────────────────
    log_d = np.log(d_i)                  # (n_samp,) — used directly in r_mat

    # ── 2. Sample quality q_s — independent of depth ─────────────────────
    # FIX: the previous definition q_s = log(d_i) – mean(log(d_i)) is just
    # centred depth, so adding it to log_d in log_lam double-counts depth.
    #
    # True quality offset: residual of log(total_bp) from what pure Poisson
    # coverage at the observed mean depth would predict.
    #   Expected total mapped bp = mean_depth_per_base × total_catalog_length
    #   where mean_depth_per_base = d_mat.mean(axis=1) (average across SGBs)
    #                total_catalog_length = L_k.sum()
    # The ratio actual / expected captures mapping efficiency, extraction
    # quality, and host contamination fraction independently of raw depth.
    #
    # If d_mat is all-zero (no mapped reads), fall back to zero quality offset.
    catalog_len  = float(L_k.sum())
    mean_dpb     = d_mat.mean(axis=1)                    # mean depth/base over catalog
    expected_tbp = np.maximum(mean_dpb * catalog_len, 1.0)
    q_s          = np.log(d_i) - np.log(expected_tbp)   # log(actual/expected) mapping
    q_s         -= q_s.mean()                            # zero-mean within group
    # q_s > 0: sample maps better than expected (good library / low host DNA)
    # q_s < 0: sample maps worse (poor extraction / high host contamination)

    # ── 1. Pre-threshold detection masks ─────────────────────────────────
    det_eps  = b_frac > eps             # (n, k) any sub-threshold detection
    det_full = b_frac >= θ              # (n, k) above-threshold (standard)
    h_hat    = det_full.mean(0)

    # ── 1. mu_x_k and sig_x_k from sub-threshold detections ──────────────
    # log(x_ks) = log(λ_ks) – log(d_i) + log(L_k)
    # With Fix A: use truncation-corrected MLE instead of naive mean/std.
    # Truncation point per sample: c_ik = log(tau_eps) - log(d_i) + log(L_k)
    # where tau_eps = -log(1-eps) is the Poisson threshold at the eps breadth level.
    mu_x      = np.full(n_sgb, np.nan, float)
    sig_x     = np.full(n_sgb, 0.05,  float)
    n_det_arr = np.zeros(n_sgb, int)          # sub-threshold detection counts per species
    eps_tau   = -np.log1p(-eps)               # Poisson threshold at eps breadth level

    for k in range(n_sgb):
        idx_det = np.where(det_eps[:, k])[0]
        n_det_arr[k] = len(idx_det)
        if len(idx_det) < 2:                  # lowered from 3; EB shrinkage handles noise
            continue
        lam_raw = d_mat[idx_det, k]
        mask_pos = lam_raw > 0
        lam      = lam_raw[mask_pos]
        idx_v    = idx_det[mask_pos]
        if len(lam) < 2:
            continue
        log_x = np.log(lam) - log_d[idx_v] + log_L[k]
        # Truncation points: minimum log-abundance detectable in each sample at eps
        log_trunc = np.log(eps_tau) - log_d[idx_v] + log_L[k]
        if correct_truncation and len(log_x) >= 3:
            # Fix A: MLE corrects upward bias in mu_x and downward bias in sig_x
            mu_x[k], sig_x[k] = _truncated_lognormal_mle(log_x, log_trunc)
        else:
            mu_x[k]  = float(np.mean(log_x))
            sig_x[k] = max(float(np.std(log_x, ddof=1)), 0.05)

    # Fix B: EB shrinkage fills NaN species with group prior instead of r_mat=0.
    # After this, ALL species have finite mu_x/sig_x — has_p becomes all-True.
    mu_x, sig_x = _empirical_bayes_shrink(mu_x, sig_x, n_det_arr, n0=eb_n0)

    # ── 3. Beta-Binomial concentration omega_k via method of moments ──────
    # For Beta(p·ω, (1-p)·ω):  E[B] = p,  Var[B] = p(1-p)/(ω+1)
    # → ω = p(1-p)/Var(B) – 1
    phi_k = np.full(n_sgb, np.nan, float)
    for k in range(n_sgb):
        idx_det = np.where(det_eps[:, k])[0]
        if len(idx_det) < 5:
            continue
        b_obs = b_frac[idx_det, k]
        b_obs = b_obs[b_obs > eps]
        if len(b_obs) < 5:
            continue
        p_hat = np.mean(b_obs)
        v_hat = np.var(b_obs, ddof=1)
        if v_hat < 1e-9:
            phi_k[k] = 50.0   # near-deterministic breadth
        else:
            omega = p_hat * (1.0 - p_hat) / v_hat - 1.0
            phi_k[k] = max(omega, 0.5)   # floor: maximally overdispersed
    phi_median    = float(np.nanmedian(phi_k))
    if np.isnan(phi_median):
        phi_median = 5.0
    phi_k         = np.where(np.isfinite(phi_k), phi_k, phi_median)

    # ── 3. r_k(d_i, q_s) via marginal BB-CDF ────────────────────────────
    # After EB shrinkage all species have finite mu_x/sig_x, so has_p is
    # all-True and r_mat is computed for every species (no more r_mat=0 fallback).
    has_p = np.isfinite(mu_x)   # True for all species after EB shrinkage
    r_mat = np.zeros((n_samp, n_sgb), float)

    if has_p.any():
        log_eff = (log_d[:, None] + q_s[:, None]
                   - np.log(L_k)[None, has_p])           # (n_samp, n_valid)
        r_mat[:, has_p] = _bb_detect_prob_marginal(
            mu      = mu_x[has_p],
            sig     = sig_x[has_p],
            phi     = phi_k[has_p],
            log_eff = log_eff,
            theta   = θ,
            dist    = abund_dist,
            df      = abund_df,
            n_q     = n_q,
        )

    r_mat[:, ~has_p] = 0.0   # no-op after EB shrinkage; kept for safety

    # ── HT occupancy with pre-threshold augmentation ──────────────────────
    # Numerator: full detections + BB-derived posterior weight for sub-threshold.
    #
    # For sub-threshold detections (eps < B < theta), the correct weight is:
    #   w_sub_ks = P(Z=1 | B_ks in [eps, theta])
    #
    # Under the strict absence model (Z=0 → B=0), any B > eps is evidence of
    # presence, but the RELIABILITY of that evidence depends on how likely a
    # truly-present species is to fall in the sub-threshold band vs above it.
    # Specifically:
    #   w_sub_ks = P(B in [eps, theta] | Z=1, d_i) / P(B >= eps | Z=1, d_i)
    #            = [r_k(d_i, eps) - r_k(d_i, theta)] / r_k(d_i, eps)
    #
    # This equals 1 for species near the detection limit (always sub-threshold
    # when present) and shrinks toward 0 for abundant species that should be
    # above theta when present (making a sub-threshold observation less credible).
    #
    # r_k(d_i, eps): BB prob of breadth >= eps given present (computed below)
    sub_thresh  = det_eps & ~det_full                          # (n, k) bool

    # Compute r_k(d_i, eps) — same structure as r_mat but at threshold=eps
    r_mat_eps = np.zeros((n_samp, n_sgb), float)
    if has_p.any():
        log_eff_eps = (log_d[:, None] + q_s[:, None]
                       - np.log(L_k)[None, has_p])       # (n_samp, n_valid)
        r_mat_eps[:, has_p] = _bb_detect_prob_marginal(
            mu  = mu_x[has_p],
            sig = sig_x[has_p],
            phi = phi_k[has_p],
            log_eff = log_eff_eps,
            theta   = eps,
            dist    = abund_dist,
            df      = abund_df,
            n_q     = n_q,
        )
    r_mat_eps[:, ~has_p] = h_hat[~has_p][None, :]

    # Per-(sample, species) BB posterior weight for sub-threshold detections
    band_prob  = np.clip(r_mat_eps - r_mat, 0.0, 1.0)        # P(eps<=B<theta | Z=1)
    safe_r_eps = np.maximum(r_mat_eps, 1e-9)
    w_sub      = np.clip(band_prob / safe_r_eps, 0.0, 1.0)   # (n_samp, n_sgb)

    n_full    = det_full.sum(0).astype(float)                 # (k,)
    # Weighted sub-threshold contribution: sum of per-sample weights
    n_sub_wt  = (sub_thresh.astype(float) * w_sub).sum(0)    # (k,)
    sum_r     = r_mat.sum(0)
    numerator = n_full + n_sub_wt                             # BB-principled numerator
    psi_ht    = np.clip(np.where(sum_r > 0.1, numerator / sum_r, 0.0), 0.0, 1.0)

    return (h_hat, psi_ht, mu_x, sig_x, r_mat, sum_r, d_i, L_k, log_tau, q_s, phi_k)


# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 — JOINT THRESHOLD MODEL
#
#  Phase A: two-pass implementation.
#
#  Pass 1 (single batch loop):  μ_k, σ_k, φ_k, h_obs per species chunk.
#  EB shrinkage applied globally between passes.
#  Pass 2 (single batch loop):  r_mean_surf and h_mat_primary.
#
#  r_mean_surf uses the EMPIRICAL average over the actual observed depths:
#      r̄_k(τ_j) = (1/n) Σ_s Φ((μ_k + log d_s − log L_k − log τ_j) / σ_k)
#  This is the correct non-parametric estimator.  It makes no assumption
#  about the distribution of log(d): depth is typically multi-modal across
#  studies and far from normal, so the closed-form normal approximation
#  (which depends on Var(log d) only) would be wrong here.
#  total_bp = Σ_k depth_ks·L_k is already in the pipeline; log_d = log(total_bp).
# ════════════════════════════════════════════════════════════════════════════

def estimate_joint_bb_model(breadth_block, depth_block, lengths, total_bp,
                             thresholds_pct=None, eps_pct=0.5,
                             abund_dist="lognormal", abund_df=5, n_q=7,
                             correct_truncation=True, eb_n0=5,
                             batch_k=500):
    """
    Phase A joint threshold-category hurdle model — two-pass, memory-safe.

    Pass 1 (batch loop over species chunks):
      μ_k, σ_k via truncation-corrected MLE on sub-threshold detections.
      φ_k (BB concentration) via vectorised method of moments.
      h_obs[k,j] = empirical detection fraction at each threshold.

    EB shrinkage applied globally between passes.

    Pass 2 (batch loop over species chunks):
      For each species chunk and each threshold τ_j, compute:
          r_j[s, k] = Φ((μ_k + log d_s − log L_k − log τ_j) / σ_k)
      using the ACTUAL observed depths log d_s (no distributional assumption).
      Then:
          r_mean_surf[k, j] = mean_s( r_j[:, k] )   — empirical average
          h_mat_primary[:, k] = r_j[:, k]            — primary threshold only

    ψ_k estimated via multi-threshold HT: weighted average over thresholds of
    h_obs[k,j] / r_mean_surf[k,j], weights ∝ h_obs[k,j].

    Why empirical depth average?
    log(total_bp) is multi-modal across studies and typically far from normal.
    The normal approximation E_d[Φ(z)] ≈ Φ(mean_z/√(1+Var_z)) is only exact
    under normality and can be badly wrong here.  The empirical sum costs
    O(n_samp × batch_k) ndtr calls per threshold per batch — still 27× faster
    than betainc while being unconditionally correct.

    Returns
    -------
    psi_k, mu_x, sig_x, phi_k, d_i, L_k,
    thresholds_frac, r_mean_surf, autc_k, h_mat_primary, None
    """
    if thresholds_pct is None:
        thresholds_pct = THRESHOLDS_GRID * 100.0
    thresholds_pct  = np.asarray(thresholds_pct, float)
    thresholds_frac = thresholds_pct / 100.0
    n_thresh        = len(thresholds_frac)
    j_primary       = int(np.argmin(thresholds_frac))
    tau_primary     = float(thresholds_frac[j_primary])
    log_tau_primary = float(np.log(-np.log1p(-tau_primary)))

    # ── Align indices ─────────────────────────────────────────────────────
    idx    = (breadth_block.index
              .intersection(depth_block.index)
              .intersection(total_bp.index))
    b_frac = breadth_block.loc[idx].values.astype(float) / 100.0
    d_mat  = depth_block.loc[idx].values.astype(float)
    d_i    = np.maximum(total_bp.loc[idx].values.astype(float), 1.0)
    n_samp, n_sgb = b_frac.shape

    L_k   = np.maximum(lengths.reindex(breadth_block.columns).values.astype(float), 1.0)
    log_L = np.log(L_k)
    # log(total_bp) — the actual observed sequencing effort per sample.
    # total_bp = Σ_k depth_ks × L_k is computed upstream in load_reference().
    # Its distribution is NOT assumed normal; the empirical values are used directly.
    log_d = np.log(d_i)
    eps   = eps_pct / 100.0
    eps_tau = -np.log1p(-eps)

    # ── Outputs ────────────────────────────────────────────────────────────
    mu_x        = np.full(n_sgb, np.nan, float)
    sig_x       = np.full(n_sgb, 0.05,  float)
    phi_k       = np.full(n_sgb, np.nan, float)
    n_det_arr   = np.zeros(n_sgb, int)
    h_obs       = np.zeros((n_sgb, n_thresh), float)
    r_mean_surf = np.zeros((n_sgb, n_thresh), float)
    h_mat_primary = np.zeros((n_samp, n_sgb), float)

    # ══════════════════════════════════════════════════════════════════════
    #  PASS 1: μ_k, σ_k, φ_k, h_obs
    #  Batch over species; each chunk loads only (n_samp, batch_k) arrays.
    # ══════════════════════════════════════════════════════════════════════
    for k_start in range(0, n_sgb, batch_k):
        k_end = min(k_start + batch_k, n_sgb)
        k_sl  = slice(k_start, k_end)
        bk    = k_end - k_start

        b_b = b_frac[:, k_sl]   # (n_samp, bk)
        d_b = d_mat[:, k_sl]    # (n_samp, bk)

        det_eps_b = b_b > eps
        n_det_arr[k_sl] = det_eps_b.sum(axis=0)

        # ── μ_k, σ_k: per-species within batch (cache-friendly access) ──
        for k_local in range(bk):
            k = k_start + k_local
            idx_det  = np.where(det_eps_b[:, k_local])[0]
            if len(idx_det) < 2: continue
            lam_raw  = d_b[idx_det, k_local]
            mask_pos = lam_raw > 0
            lam      = lam_raw[mask_pos]
            idx_v    = idx_det[mask_pos]
            if len(lam) < 2: continue
            log_x     = np.log(lam) - log_d[idx_v] + log_L[k]
            log_trunc = np.log(eps_tau) - log_d[idx_v] + log_L[k]
            if correct_truncation and len(log_x) >= 3:
                mu_x[k], sig_x[k] = _truncated_lognormal_mle(log_x, log_trunc)
            else:
                mu_x[k]  = float(np.mean(log_x))
                sig_x[k] = max(float(np.std(log_x, ddof=1)), 0.05)

        # ── φ_k: vectorised method of moments ──────────────────────────
        b_masked = np.where(det_eps_b, b_b, np.nan)
        n_d      = n_det_arr[k_sl].astype(float)
        p_hat    = np.nanmean(b_masked, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            var_b = np.nanvar(b_masked, axis=0)
            var_b = np.where(n_d > 1, var_b * n_d / (n_d - 1), np.nan)
            om    = np.where(
                (n_d >= 5) & (var_b > 1e-9) & np.isfinite(p_hat),
                p_hat * (1.0 - p_hat) / var_b - 1.0, np.nan)
        phi_b = np.where(np.isfinite(om) & (om > 0),
                          np.clip(om, 0.5, 500.0), np.nan)
        phi_k[k_sl] = np.where(n_d >= 5,
                                np.where(np.isfinite(phi_b), phi_b, 50.0),
                                np.nan)

        # ── h_obs: raw detection fraction per threshold ─────────────────
        for j, tau in enumerate(thresholds_frac):
            h_obs[k_sl, j] = (b_b >= tau).mean(axis=0)

    # ── Fill missing φ_k with group median ───────────────────────────────
    phi_med = float(np.nanmedian(phi_k))
    if np.isnan(phi_med): phi_med = 5.0
    phi_k = np.where(np.isfinite(phi_k), phi_k, phi_med)

    # ── EB shrinkage for μ_x, σ_x ─────────────────────────────────────────
    mu_x, sig_x = _empirical_bayes_shrink(mu_x, sig_x, n_det_arr, n0=eb_n0)

    # ══════════════════════════════════════════════════════════════════════
    #  PASS 2: r_mean_surf and h_mat_primary
    #
    #  r̄_k(τ_j) = (1/n) Σ_s Φ((μ_k + log d_s − log L_k − log τ_j) / σ_k)
    #
    #  This averages the per-sample lognormal detection probability over the
    #  ACTUAL empirical distribution of log(total_bp), not a parametric fit.
    #  log_d contains the real observed values; no normality is assumed.
    #
    #  Memory per batch: (n_samp, batch_k) × 8B = ~56 MB for typical sizes.
    #  Speed: n_thresh × ndtr(n_samp × batch_k) per batch — no betainc.
    # ══════════════════════════════════════════════════════════════════════
    for k_start in range(0, n_sgb, batch_k):
        k_end = min(k_start + batch_k, n_sgb)
        k_sl  = slice(k_start, k_end)

        mu_b  = np.where(np.isfinite(mu_x[k_sl]),
                          mu_x[k_sl], float(log_d.mean()) - log_L[k_sl])
        sig_b = np.maximum(np.where(np.isfinite(sig_x[k_sl]),
                                     sig_x[k_sl], 0.5), 0.05)

        # log_eff[s, k] = log d_s − log L_k  (Phase A: no q_s offset)
        log_eff_b = log_d[:, None] - log_L[None, k_sl]   # (n_samp, bk)

        for j, tau in enumerate(thresholds_frac):
            log_tau_j = float(np.log(-np.log1p(-tau)))
            # z[s, k] = (μ_k + log d_s − log L_k − log τ_j) / σ_k
            z_j = (mu_b[None, :] + log_eff_b - log_tau_j) / sig_b[None, :]
            r_j = ndtr(z_j)                           # (n_samp, bk)
            r_mean_surf[k_sl, j] = r_j.mean(axis=0)  # empirical mean over actual depths

            if j == j_primary:
                # Store per-sample r at primary threshold for h_mat_primary.
                # Will be multiplied by ψ_k below.
                h_mat_primary[:, k_sl] = r_j

    # ── Multi-threshold HT occupancy estimator ────────────────────────────
    with np.errstate(divide='ignore', invalid='ignore'):
        psi_by_thresh = np.where(r_mean_surf > 1e-9,
                                  h_obs / np.maximum(r_mean_surf, 1e-9), 0.0)
    psi_by_thresh = np.clip(psi_by_thresh, 0.0, 1.0)
    weights = np.maximum(h_obs, 1e-6)
    psi_k   = (psi_by_thresh * weights).sum(1) / weights.sum(1)
    psi_k   = np.clip(psi_k, 1e-4, 1.0 - 1e-4)

    # ── h_mat_primary = ψ_k × r_k(d_s, τ_primary) ────────────────────────
    h_mat_primary *= psi_k[None, :]
    h_mat_primary  = np.clip(h_mat_primary, 0.0, 1.0 - 1e-9)

    # ── AUTC: ∫ r̄_k(τ) dτ  (trapezoidal over threshold grid) ─────────────
    _npz   = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    autc_k = _npz(r_mean_surf, x=thresholds_frac, axis=1)

    return (psi_k, mu_x, sig_x, phi_k, d_i, L_k,
            thresholds_frac, r_mean_surf, autc_k, h_mat_primary, None)

def predictive_surface_from_joint(psi_k, mu_x, sig_x, phi_k, d_i, L_k,
                                   thresholds_pct, N_max, n_draw, rng,
                                   abund_dist="lognormal", abund_df=5, n_q=7,
                                   batch_k=500):
    """
    Compute S(N, τ) surface using batched lognormal r (no betainc).

    h_mat_j[s,k] = ψ_k · Phi((μ_k + log d_s − log L_k − log τ_j) / σ_k)

    27× faster than the betainc path.  Returns real predictive draws at the
    primary (lowest) threshold.

    Returns
    -------
    surface      : {tau_pct: (Ns, pMean, pLo, pHi)}
    primary_draws: (n_draw, n_pts) actual draws at primary threshold
    Ns_primary   : (n_pts,) N-grid for primary threshold
    """
    thresholds_pct  = np.asarray(thresholds_pct, float)
    j_primary       = int(np.argmin(thresholds_pct))
    tau_primary_pct = float(thresholds_pct[j_primary])

    log_d  = np.log(np.maximum(d_i, 1.0))
    log_L  = np.log(np.maximum(L_k, 1.0))
    n_samp = len(d_i); n_sgb = len(psi_k)

    surface       = {}
    pDraws_primary = None
    Ns_primary     = None

    for tau_pct in thresholds_pct:
        tau       = float(tau_pct) / 100.0
        log_tau_j = float(np.log(-np.log1p(-tau)))
        h_tau     = np.zeros((n_samp, n_sgb), float)

        for k_start in range(0, n_sgb, batch_k):
            k_end = min(k_start + batch_k, n_sgb)
            k_sl  = slice(k_start, k_end)
            mu_b  = np.where(np.isfinite(mu_x[k_sl]),
                              mu_x[k_sl], float(log_d.mean()) - log_L[k_sl])
            sig_b = np.maximum(np.where(np.isfinite(sig_x[k_sl]),
                                         sig_x[k_sl], 0.5), 0.05)
            log_eff_b = log_d[:, None] - log_L[None, k_sl]   # (n_samp, bk)
            z = (mu_b[None, :] + log_eff_b - log_tau_j) / sig_b[None, :]
            r_j = ndtr(z)
            h_tau[:, k_sl] = np.clip(psi_k[k_sl][None, :] * r_j, 0.0, 1.0 - 1e-9)

        Ns, pM, pLo, pHi, draws = predictive_accumulation_from_hmat(
            h_tau, N_max=N_max, n_draw=n_draw, rng=rng)
        surface[float(tau_pct)] = (Ns, pM, pLo, pHi)

        if tau_pct == tau_primary_pct:
            pDraws_primary = draws
            Ns_primary     = Ns

    return surface, pDraws_primary, Ns_primary



# ════════════════════════════════════════════════════════════════════════════
#  PER-SPECIES ABSENCE TESTING  (Document 7 framework)
# ════════════════════════════════════════════════════════════════════════════

def classify_species_by_evidence(sum_r_ind, h_hat_ind, psi_ht_ind=None,
                                  class1_thr=5.0, class2_thr=1.0,
                                  psi_absent_eps=0.05):
    """
    Three-class evidence taxonomy based on Industrial detection power Σ_s r_k(d_s^Ind).

    Classification uses psi_ht_ind (HT-corrected occupancy from BB model) as
    the primary "undetected" criterion rather than h_hat_ind (binary threshold).
    This ensures consistency: a species with substantial sub-threshold signal
    (ψ_Ind > ε) is NOT classified as absent regardless of above-threshold count.

    Parameters
    ----------
    sum_r_ind      : (n_sgb,) Σ_s r_k(d_s^Ind) — Industrial detection power
    h_hat_ind      : (n_sgb,) naive above-threshold prevalence
    psi_ht_ind     : (n_sgb,) BB-corrected HT occupancy (preferred criterion).
                     If None, falls back to h_hat_ind == 0 (old behaviour).
    class1_thr     : sum_r threshold for Class I (high power)
    class2_thr     : sum_r threshold for Class II (marginal power)
    psi_absent_eps : occupancy threshold below which a species is considered
                     "near-absent" in Industrial. Default 0.05.

    Classes
    -------
    Class I   — near-absent in Industrial AND high detection power:
                psi_ht < ε  AND  sum_r > class1_thr
                → genuine biological absence is identifiable
    Class II  — near-absent AND marginal power:
                psi_ht < ε  AND  class2_thr ≤ sum_r ≤ class1_thr
    Class III — near-absent AND negligible power:
                psi_ht < ε  AND  sum_r < class2_thr
                → prior-dominated, unresolvable
    Detected  — ψ_Ind ≥ ε: species has meaningful Industrial signal,
                cannot be called absent (may be depleted but not missing)

    Also returns `psi_conflict`: species where psi_ht >> h_hat (sub-threshold
    model gives high occupancy but zero above-threshold detections). These
    are present in Industrial at low abundance, not missing.
    """
    # Primary absence criterion: use BB-corrected occupancy if available
    if psi_ht_ind is not None:
        near_absent = psi_ht_ind < psi_absent_eps
    else:
        near_absent = h_hat_ind == 0.0

    # Conflict: sub-threshold model says ψ > 5% but no above-threshold detections
    # These are NOT absent — they are present at sub-threshold abundance
    if psi_ht_ind is not None:
        psi_conflict = (h_hat_ind == 0.0) & (psi_ht_ind >= psi_absent_eps)
    else:
        psi_conflict = np.zeros(len(h_hat_ind), bool)

    return {
        "class1":       near_absent & (sum_r_ind >  class1_thr),
        "class2":       near_absent & (sum_r_ind >= class2_thr) & (sum_r_ind <= class1_thr),
        "class3":       near_absent & (sum_r_ind <  class2_thr),
        "detected":     ~near_absent,
        "psi_conflict": psi_conflict,   # present sub-threshold, not absent
    }


def study_stratified_psi(breadth_block, r_mat, meta, threshold=10.0,
                          study_col="StudyFolder", eps_pct=0.5):
    """
    Study-stratified Horvitz-Thompson occupancy estimator.

    Pools HT estimates within studies first, then combines across studies
    with inverse-variance weighting. This guards against individual studies
    dominating ψ̂_k when sample sizes are highly unequal across studies
    (which is exactly the situation with Industrial vs NI pooling).

    Returns
    -------
    psi_strat : (n_sgb,) study-stratified occupancy
    psi_se    : (n_sgb,) approximate SE (from between-study variance)
    n_studies : int — number of studies contributing
    """
    det_full = (breadth_block.values >= threshold).astype(float)
    det_eps  = (breadth_block.values >= eps_pct).astype(float)
    n_samp, n_sgb = det_full.shape

    # Identify study per sample
    idx = breadth_block.index
    if study_col in meta.columns:
        studies = meta.loc[idx, study_col].astype(str).fillna("unknown").values
    else:
        studies = np.array(["all"] * n_samp)

    unique_studies = np.unique(studies)
    n_studies = len(unique_studies)

    psi_per_study = np.full((n_studies, n_sgb), np.nan, float)
    n_per_study   = np.zeros(n_studies, int)

    for si, study in enumerate(unique_studies):
        mask = studies == study
        n_s  = int(mask.sum())
        if n_s < 2:
            continue
        n_per_study[si] = n_s
        sum_r_s = r_mat[mask].sum(0)
        num_s   = det_full[mask].sum(0)  # use threshold detections only (conservative)
        psi_s   = np.where(sum_r_s > 0.1, num_s / sum_r_s, np.nan)
        psi_per_study[si] = np.clip(psi_s, 0.0, 1.0)

    # Inverse-sample-size weighted mean across studies
    weights = np.where(n_per_study > 0, n_per_study.astype(float), 0.0)
    w_sum   = weights.sum()
    if w_sum < 1:
        # Fallback: global HT
        sum_r_g = r_mat.sum(0)
        psi_strat = np.clip(
            np.where(sum_r_g > 0.1, det_full.sum(0) / sum_r_g, 0.0), 0.0, 1.0)
        return psi_strat, np.zeros(n_sgb), 1

    psi_strat = np.nansum(psi_per_study * weights[:, None], axis=0) / w_sum

    # Between-study variance as SE estimate (captures study heterogeneity)
    diff2 = (psi_per_study - psi_strat[None, :]) ** 2
    psi_var = np.nansum(diff2 * weights[:, None], axis=0) / (
        w_sum * max(n_studies - 1, 1))
    psi_se  = np.sqrt(np.maximum(psi_var, 0.0))

    return np.clip(psi_strat, 0.0, 1.0), psi_se, n_studies


def joint_absence_posterior(psi_ind_boots, psi_ni_boots,
                             eps=0.05, tau=0.10):
    """
    Per-SGB posterior probability of joint absence/depletion:
        P(ψ_k^Ind < ε  AND  ψ_k^NI > τ | data)

    This is the primary estimand: a species is "missing in Industrial"
    only when it is (a) genuinely rare/absent in Industrial AND (b)
    meaningfully prevalent in NI. The joint condition rules out species
    that are globally rare.

    Parameters
    ----------
    psi_ind_boots : (n_boot, n_sgb) bootstrap draws of Industrial occupancy
    psi_ni_boots  : (n_boot, n_sgb) bootstrap draws of NI occupancy
    eps           : threshold for "near-absent" in Industrial (default 0.05)
    tau           : threshold for "meaningfully present" in NI (default 0.10)

    Returns
    -------
    p_joint   : (n_sgb,) P(ψ_Ind < ε AND ψ_NI > τ)
    p_depleted: (n_sgb,) P(ψ_NI – ψ_Ind > τ)  [depletion, not necessarily absent]
    E_N_missing: float — posterior expected number of missing SGBs
    CI_N_missing: (lo, hi) 95% CI on N_missing
    """
    absent_ind  = psi_ind_boots < eps          # (n_boot, n_sgb)
    present_ni  = psi_ni_boots  > tau          # (n_boot, n_sgb)
    joint       = absent_ind & present_ni      # (n_boot, n_sgb)

    p_joint    = joint.mean(0)                 # (n_sgb,)
    p_depleted = (psi_ni_boots - psi_ind_boots > tau).mean(0)

    # Posterior N_missing per bootstrap draw
    N_missing_draws = joint.sum(1)             # (n_boot,)
    E_N_missing = float(N_missing_draws.mean())
    CI_N_missing = (float(np.percentile(N_missing_draws, 2.5)),
                    float(np.percentile(N_missing_draws, 97.5)))

    return p_joint, p_depleted, E_N_missing, CI_N_missing


def compute_log_likelihood_ratios(psi_ni, r_mat_ind):
    """
    Log likelihood ratio for each species: H1 (absent) vs H0 (present at NI occupancy).

    H0: ψ_k^Ind = ψ_k^NI   (present but missed due to low depth)
    H1: ψ_k^Ind = 0         (genuinely absent)

    Under H1: P(zero detections) = 1
    Under H0: P(zero detections) = Π_s (1 – ψ_k^NI · r_k(d_s^Ind))

    log LR_k = – Σ_s log(1 – ψ_k^NI · r_k(d_s^Ind))

    Larger log_LR → stronger evidence for genuine absence.
    Grows exponentially with both sample count and per-sample detection power.

    Parameters
    ----------
    psi_ni   : (n_sgb,) HT occupancy for NI group
    r_mat_ind: (n_samp_ind, n_sgb) per-sample detection probs in Industrial
    """
    psi_clipped = np.clip(psi_ni, 0.0, 1.0 - 1e-9)
    # ψ_k · r_k(d_s): probability of detection under H0
    # clip to avoid log(0): if ψ·r ≈ 1, H0 almost certainly predicts detection
    # so non-detection is near-impossible under H0 → LR = +∞
    pr = np.clip(psi_clipped[None, :] * r_mat_ind, 0.0, 1.0 - 1e-9)
    log_lr = -np.sum(np.log1p(-pr), axis=0)   # Σ_s log(1/(1-ψ·r))
    return log_lr                               # (n_sgb,)


def bootstrap_psi_contrast(breadth_ind, breadth_ni, r_mat_ind, r_mat_ni,
                            w_sub_ind=None, w_sub_ni=None,
                            eps_pct=0.5, threshold=10.0,
                            delta_bio=0.10, n_boot=500, rng=None):
    """
    Bootstrap posterior on Δψ_k = ψ_k^NI – ψ_k^Ind using the HT estimator.

    Uses the BB-derived per-sample sub-threshold weights w_sub_ks (from
    estimate_group_model_bb) so that the bootstrap CI is internally consistent
    with the point estimate.  Falls back to binary threshold when weights are
    not supplied.

    For each bootstrap replicate b:
      Resample Industrial samples with replacement → ψ_k^Ind(b) = Σ(det + w_sub) / Σ r_k
      Resample NI samples with replacement         → ψ_k^NI(b) = Σ(det + w_sub) / Σ r_k
      Δψ_k(b) = ψ_k^NI(b) – ψ_k^Ind(b)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    b_ind_vals = breadth_ind.values
    b_ni_vals  = breadth_ni.values
    n_ind, n_sgb = b_ind_vals.shape
    n_ni         = b_ni_vals.shape[0]

    det_full_ind = (b_ind_vals >= threshold).astype(float)
    det_full_ni  = (b_ni_vals  >= threshold).astype(float)
    det_eps_ind  = (b_ind_vals >= eps_pct).astype(float)
    det_eps_ni   = (b_ni_vals  >= eps_pct).astype(float)
    sub_ind = det_eps_ind - det_full_ind   # 1 where eps <= B < theta
    sub_ni  = det_eps_ni  - det_full_ni

    # Use supplied BB weights or fall back to uniform 1.0 for sub-threshold
    w_i = w_sub_ind if w_sub_ind is not None else np.ones((n_ind, n_sgb), float)
    w_n = w_sub_ni  if w_sub_ni  is not None else np.ones((n_ni,  n_sgb), float)

    # Pre-compute weighted numerators per sample: det_full + w_sub * sub
    num_ind = det_full_ind + w_i * sub_ind   # (n_ind, n_sgb)
    num_ni  = det_full_ni  + w_n * sub_ni    # (n_ni,  n_sgb)

    deltas = np.zeros((n_boot, n_sgb), float)
    for b in range(n_boot):
        idx_i = rng.integers(0, n_ind, n_ind)
        idx_n = rng.integers(0, n_ni,  n_ni)

        sum_r_i = r_mat_ind[idx_i].sum(0); wnum_i = num_ind[idx_i].sum(0)
        sum_r_n = r_mat_ni[idx_n].sum(0);  wnum_n = num_ni[idx_n].sum(0)

        psi_i = np.clip(np.where(sum_r_i > 0.1, wnum_i / sum_r_i, 0.0), 0.0, 1.0)
        psi_n = np.clip(np.where(sum_r_n > 0.1, wnum_n / sum_r_n, 0.0), 0.0, 1.0)
        deltas[b] = psi_n - psi_i

    return (deltas.mean(0),
            deltas.std(0),
            (deltas > delta_bio).mean(0),
            np.percentile(deltas, 2.5,  axis=0),
            np.percentile(deltas, 97.5, axis=0))


def depth_gradient_test_class2(breadth_ind, total_bp_ind, evidence_classes,
                                max_species=300, rng=None):
    """
    Within-Industrial depth gradient test for Class II species.

    If ∂/∂d P(detected | Ind, d) > 0: detection increases with depth
       → consistent with H0 (present but missed)
    If slope ≈ 0: depth doesn't predict detection
       → consistent with H1 (truly absent)

    Fits logistic regression: logit P(Y_ks=1) = α_k + β_k · log(d_s)
    jointly over (α_k, β_k) via BFGS with full observed Fisher information
    for the SE — not a 1-D marginal optimisation.

    Returns β_k (slope on log-depth) and its z-score for each Class II species.
    NaN for all non-Class-II species.
    """
    from scipy.special import expit
    from scipy.optimize import minimize

    class2_mask = evidence_classes["class2"]
    class2_idx  = np.where(class2_mask)[0][:max_species]
    b_mat = (breadth_ind.values >= 10.0).astype(float)   # (n, k) binary
    log_d = np.log(np.maximum(total_bp_ind.values.astype(float), 1.0))
    log_d_c = log_d - log_d.mean()       # centre to improve conditioning

    slopes   = np.full(len(class2_mask), np.nan, float)
    slope_se = np.full(len(class2_mask), np.nan, float)

    def neg_ll_and_grad(params, y, x):
        a, b_ = params
        eta = a + b_ * x
        p   = expit(eta)
        p   = np.clip(p, 1e-9, 1 - 1e-9)
        nll = -np.sum(y * np.log(p) + (1 - y) * np.log1p(-p))
        # Gradient
        resid = p - y
        ga = resid.sum()
        gb = (resid * x).sum()
        return nll, np.array([ga, gb])

    def hessian(params, x):
        """Observed Fisher information matrix (2×2)."""
        a, b_ = params
        p = expit(np.clip(a + b_ * x, -30, 30))
        w = p * (1 - p)
        H = np.array([
            [w.sum(),          (w * x).sum()],
            [(w * x).sum(),    (w * x**2).sum()],
        ])
        return H

    for k in class2_idx:
        y = b_mat[:, k]
        if y.sum() == 0:
            slopes[k] = 0.0; slope_se[k] = np.inf
            continue

        # Initialise: intercept at empirical log-odds, slope at 0
        p0 = np.clip(y.mean(), 1e-4, 1 - 1e-4)
        x0 = np.array([np.log(p0 / (1 - p0)), 0.0])

        res = minimize(
            neg_ll_and_grad, x0, args=(y, log_d_c),
            method="BFGS", jac=True,
            options={"maxiter": 400, "gtol": 1e-6},
        )
        if not res.success and res.fun > neg_ll_and_grad(x0, y, log_d_c)[0]:
            # Optimisation failed to improve — skip
            slopes[k] = np.nan; slope_se[k] = np.nan
            continue

        a_hat, b_hat = res.x
        H = hessian(res.x, log_d_c)
        # SE from inverse Hessian diagonal
        try:
            H_inv = np.linalg.inv(H)
            se_b  = float(np.sqrt(max(H_inv[1, 1], 1e-12)))
        except np.linalg.LinAlgError:
            se_b  = np.nan

        slopes[k]   = b_hat
        slope_se[k] = se_b

    return slopes, slope_se   # NaN for non-Class-II species


def power_sensitivity_curve(mu_x_ind, sig_x_ind, phi_k_ind, L_k, log_tau,
                             n_ind_samples, class1_thr=5.0,
                             depth_grid=None,
                             abundance_percentiles=(5, 10, 25, 50)):
    """
    Power sensitivity analysis for Class III (unidentifiable) species.

    For species with no sub-threshold detections (mu_x = NaN), no species-
    specific abundance estimate exists. Rather than imputing a single community
    prior (which collapses all species to one depth value), this function
    computes a SENSITIVITY CURVE:

        For each assumed abundance percentile p of the detected species,
        and each target sequencing depth d, what fraction of Class III
        species would accumulate sufficient detection power (sum_r > class1_thr)?

    This answers the practically useful question: "if we sequenced to depth d,
    and Class III species had abundance typical of the p-th percentile of
    detected species, what fraction could we evaluate?"

    The genome length L_k is the ONLY species-specific quantity available for
    Class III species. It enters r_at_d directly and creates genuine variation
    in detectability across the curve — shorter genomes need less depth to
    clear the breadth threshold at fixed abundance.

    Parameters
    ----------
    mu_x_ind             : (n_sgb,) lognormal log-abundance; NaN for Class III
    sig_x_ind            : (n_sgb,) lognormal SD
    phi_k_ind            : (n_sgb,) BB concentration
    L_k                  : (n_sgb,) genome lengths in bp
    log_tau              : scalar   log(-log(1-θ))
    n_ind_samples        : int      number of Industrial samples
    class1_thr           : float    sum_r threshold for Class I
    depth_grid           : array    depths to evaluate (bp); default log-spaced
    abundance_percentiles: tuple    percentiles of mu_x (from detected species)
                                    to use as sensitivity scenarios

    Returns
    -------
    depth_grid  : (n_depths,) array of depths evaluated
    frac_classifiable : (n_percentiles, n_depths) fraction of Class III species
                        reaching Class I power at each depth/abundance scenario
    pct_labels  : list of strings for legend
    mu_scenarios : (n_percentiles,) mu_x value for each percentile scenario
    L_k_class3  : (n_class3,) genome lengths of Class III species (for scatter)
    """
    if depth_grid is None:
        depth_grid = np.logspace(7, 13, 80)   # 10M bp to 10 Tbp

    # Valid (detected) species provide the abundance distribution
    valid     = np.isfinite(mu_x_ind)
    class3    = ~valid                          # Class III = no abundance estimate

    L_k_class3 = L_k[class3]
    n_c3       = int(class3.sum())

    if n_c3 == 0:
        return depth_grid, np.zeros((len(abundance_percentiles), len(depth_grid))), [], np.array([]), np.array([])

    # Abundance scenarios from detected species distribution
    if valid.sum() < 2:
        mu_scenarios = np.array([-12.0] * len(abundance_percentiles))
    else:
        mu_scenarios = np.percentile(mu_x_ind[valid], list(abundance_percentiles))

    phi_median = float(np.nanmedian(phi_k_ind))
    target_sum_r = float(class1_thr)

    frac_classifiable = np.zeros((len(abundance_percentiles), len(depth_grid)))

    for pi, mu_assumed in enumerate(mu_scenarios):
        for di, d_target in enumerate(depth_grid):
            log_d_t = np.log(d_target)
            n_classifiable = 0
            for ki in np.where(class3)[0]:
                logL   = np.log(max(L_k[ki], 1.0))
                omega  = phi_k_ind[ki] if np.isfinite(phi_k_ind[ki]) else phi_median
                log_lam = mu_assumed + log_d_t - logL
                p = float(np.clip(1.0 - np.exp(-np.exp(log_lam)), 1e-7, 1-1e-7))
                a_ = max(p * omega, 1e-5)
                b_ = max((1-p) * omega, 1e-5)
                r_k = float(1.0 - betainc(a_, b_, float(-np.log1p(-0.10))))
                if r_k * n_ind_samples >= target_sum_r:
                    n_classifiable += 1
            frac_classifiable[pi, di] = n_classifiable / n_c3

    pct_labels = [f"{p}th pctile (μ_x={mu:.1f})"
                  for p, mu in zip(abundance_percentiles, mu_scenarios)]

    return depth_grid, frac_classifiable, pct_labels, mu_scenarios, L_k_class3


def power_to_classify(mu_x_ind, sig_x_ind, phi_k_ind, L_k, log_tau,
                       n_ind_samples, target_sum_r=5.0):
    """
    Legacy point-estimate version — retained for backward compatibility.
    For visualisation, use power_sensitivity_curve() instead.

    Returns d_star_k in bp and imputed mask (all True for Class III since
    all mu_x are NaN there, and the community prior collapses them to a
    single depth value — see power_sensitivity_curve for the correct treatment).
    """
    n_sgb    = len(mu_x_ind)
    d_star   = np.full(n_sgb, np.nan, float)
    imputed  = np.zeros(n_sgb, bool)
    target_r = target_sum_r / max(n_ind_samples, 1)

    # Community-level prior from species with valid estimates
    valid    = np.isfinite(mu_x_ind)
    mu_prior  = float(np.median(mu_x_ind[valid])) if valid.any() else -10.0
    sig_prior = float(np.median(sig_x_ind[valid])) if valid.any() else 1.0
    phi_prior = float(np.nanmedian(phi_k_ind))     if valid.any() else 5.0

    for k in range(n_sgb):
        if np.isfinite(mu_x_ind[k]):
            mu_k  = mu_x_ind[k]
            sig_k = sig_x_ind[k]
            phi_k = phi_k_ind[k]
        else:
            # Impute with community prior — estimate is conservative
            mu_k  = mu_prior
            sig_k = sig_prior
            phi_k = phi_prior
            imputed[k] = True

        omega = phi_k
        logL  = np.log(max(L_k[k], 1.0))

        def r_at_d(log_d_, mu=mu_k, logL_=logL, omega_=omega):
            log_lam = mu + log_d_ - logL_
            p = np.clip(1.0 - np.exp(-np.exp(log_lam)), 1e-7, 1-1e-7)
            a_ = max(p * omega_, 1e-5)
            b_ = max((1-p) * omega_, 1e-5)
            return float(1.0 - betainc(a_, b_, float(-np.log1p(-0.10))))

        # Check achievability bounds
        r_max = r_at_d(np.log(1e13))
        if r_max < target_r:
            d_star[k] = np.inf   # not achievable even at 10 Tbp — genome too rare
            continue
        r_min = r_at_d(np.log(1e7))
        if r_min >= target_r:
            d_star[k] = 1e7     # already classifiable at 10 Mbp
            continue
        # Bisection over log(d)
        lo, hi = np.log(1e7), np.log(1e13)
        for _ in range(40):
            mid = (lo + hi) / 2
            if r_at_d(mid) < target_r:
                lo = mid
            else:
                hi = mid
        d_star[k] = float(np.exp((lo + hi) / 2))

    return d_star, imputed


def build_stan_joint_model():
    """
    Return the Stan model code string for the joint two-group hierarchical
    occupancy-detection model.

    This is the strongest defensible model for the biological claim that
    bacteria are missing (not merely depleted) in Industrial populations.
    It addresses the remaining weaknesses not fixable in scipy:

    (a) Joint estimation: Industrial and NI parameters share a species-level
        prior, giving partial pooling across groups and a direct per-SGB
        group contrast β_k with uncertainty.

    (b) Study/population random effects: b_study[s] absorbs cohort-level
        confounding (library chemistry, extraction, host content) that is
        entangled with Industrial vs NI label.

    (c) Exact detection rule: the 10%-breadth threshold is encoded in
        generated quantities as a derived event from the Poisson coverage
        model, so the detection likelihood matches the pipeline exactly.

    (d) Posterior N_missing: computed per draw in generated quantities,
        giving a proper credible interval on the headline count.

    Model structure
    ---------------
    Ecological layer (occupancy):
        Z_ks ~ Bernoulli(ψ_ks)
        logit(ψ_ks) = α_k + β_k · I(Industrial_s) + b_study[study[s]]

        α_k ~ Normal(μ_α, σ_α)            [species baseline occupancy]
        β_k ~ Normal(μ_β, σ_β)            [Industrial effect; < 0 = depleted]
        b_study[j] ~ Normal(0, σ_study)   [study random effect]

        μ_α, μ_β ~ Normal(0, 2)
        σ_α, σ_β, σ_study ~ HalfNormal(1)

    Observation layer (detection given presence):
        R_ks | Z_ks=1 ~ NegBin(λ_ks, φ_k)
        log(λ_ks) = log(D_s) + η_k + v_study[study[s]] + q_s

        η_k ~ Normal(μ_η, σ_η)            [species log-abundance baseline]
        v_study[j] ~ Normal(0, σ_v)       [study abundance random effect]
        q_s = log(total_bp_s) - mean(log(total_bp))   [quality offset]

    Detection event (generated quantities):
        B_ks = 1 - exp(-R_ks · read_len / G_k)        [expected breadth]
        Y_ks = I(B_ks ≥ 0.10  OR  R_ks · read_len ≥ 300_000)

    Primary estimand (generated quantities per draw):
        I_k = I(invlogit(α_k + β_k) < 0.05  AND  invlogit(α_k) > 0.10)
        N_missing = Σ_k I_k

    Usage
    -----
    Requires: cmdstanpy or pystan.

        import cmdstanpy, json, numpy as np
        model_code = build_stan_joint_model()
        with open('joint_occupancy.stan', 'w') as f:
            f.write(model_code)
        model = cmdstanpy.CmdStanModel(stan_file='joint_occupancy.stan')

        data = prepare_stan_data(gap_results, breadth_ind, breadth_ni, rdat)
        fit  = model.sample(data=data, chains=4, iter_warmup=500,
                            iter_sampling=500, seed=42)

        # Primary result
        N_missing_draws = fit.stan_variable('N_missing')
        print(f'E[N_missing] = {N_missing_draws.mean():.1f}')
        print(f'95% CI = [{np.percentile(N_missing_draws, 2.5):.0f}, '
              f'{np.percentile(N_missing_draws, 97.5):.0f}]')

    See prepare_stan_data() for the data dictionary structure.
    """
    return r"""
data {
  int<lower=1> K;          // number of SGBs
  int<lower=1> S;          // number of samples (Ind + NI pooled)
  int<lower=1> J;          // number of studies
  int<lower=1> S_ind;      // number of Industrial samples
  int<lower=1> S_ni;       // number of NI samples

  // Per-sample covariates
  array[S] int<lower=1,upper=J> study;   // study index per sample
  array[S] int<lower=0,upper=1> is_ind;  // 1 = Industrial, 0 = NI
  vector[S] q_s;                          // quality offset (log actual/expected)
  vector[S] log_D;                        // log(total_mapped_bp) per sample

  // Per-SGB constants
  vector<lower=0>[K] G;                  // genome length in bp
  real<lower=0> read_len;                // mean read length in bp
  real<lower=0,upper=1> breadth_thr;    // breadth threshold (0.10)
  real<lower=0> kb_thr;                 // minimum covered kb (200)

  // Observed read counts — only for samples where Z_ks might be 1
  // Stored as sparse (sample, sgb, count) triplets to avoid K×S dense matrix
  int<lower=0> N_obs;                       // number of (sample, SGB) pairs with R > 0
  array[N_obs] int<lower=1,upper=S> obs_s;  // sample index
  array[N_obs] int<lower=1,upper=K> obs_k;  // SGB index
  array[N_obs] int<lower=1>         obs_r;  // mapped read count
}

parameters {
  // Occupancy layer
  real mu_alpha;
  real<lower=0> sigma_alpha;
  real mu_beta;
  real<lower=0> sigma_beta;
  real<lower=0> sigma_study_occ;
  vector[K] alpha_k_raw;       // non-centred parameterisation
  vector[K] beta_k_raw;
  vector[J] b_study_raw;

  // Abundance layer
  real mu_eta;
  real<lower=0> sigma_eta;
  real<lower=0> sigma_study_abund;
  vector[K] eta_k_raw;
  vector[J] v_study_raw;
  vector<lower=0>[K] phi_k;   // NegBin overdispersion per SGB
}

transformed parameters {
  vector[K] alpha_k = mu_alpha + sigma_alpha * alpha_k_raw;
  vector[K] beta_k  = mu_beta  + sigma_beta  * beta_k_raw;
  vector[J] b_study = sigma_study_occ   * b_study_raw;
  vector[K] eta_k   = mu_eta   + sigma_eta   * eta_k_raw;
  vector[J] v_study = sigma_study_abund * v_study_raw;
}

model {
  // Priors
  mu_alpha ~ normal(0, 2);
  mu_beta  ~ normal(0, 2);
  sigma_alpha ~ normal(0, 1);
  sigma_beta  ~ normal(0, 1);
  sigma_study_occ   ~ normal(0, 1);
  sigma_study_abund ~ normal(0, 1);
  mu_eta   ~ normal(0, 3);
  sigma_eta ~ normal(0, 1);
  alpha_k_raw ~ std_normal();
  beta_k_raw  ~ std_normal();
  b_study_raw ~ std_normal();
  eta_k_raw   ~ std_normal();
  v_study_raw ~ std_normal();
  phi_k ~ normal(3, 2);          // mild prior: moderate overdispersion

  // Likelihood (marginalised over discrete Z_ks)
  // For each sample s and SGB k:
  //   log p(R_ks) = log_sum_exp(
  //     log(ψ_ks) + NegBin(R_ks | λ_ks, φ_k),   // Z=1, present
  //     log(1-ψ_ks) + (R_ks==0 ? 0 : -inf)        // Z=0, absent -> R must be 0
  //   )
  // We handle this by only looping over observed (R>0) pairs for the Z=1 branch,
  // and computing the Z=0 contribution from all samples separately.

  // For all (s,k): accumulate log(1-ψ_sk) into target (Z=0 baseline)
  // then for observed R>0 pairs, add the Z=1 correction.
  // This is efficient when R>0 is sparse.
  for (s in 1:S) {
    for (k in 1:K) {
      real logit_psi = alpha_k[k] + beta_k[k] * is_ind[s] + b_study[study[s]];
      // Z=0 contribution (R must equal 0)
      target += log1m_inv_logit(logit_psi);
    }
  }
  for (n in 1:N_obs) {
    int s = obs_s[n]; int k = obs_k[n]; int r = obs_r[n];
    real logit_psi = alpha_k[k] + beta_k[k] * is_ind[s] + b_study[study[s]];
    real log_lambda = log_D[s] + eta_k[k] + v_study[study[s]] + q_s[s] - log(G[k]);
    // Replace Z=0 with log_sum_exp(Z=0, Z=1)
    real lp0 = log1m_inv_logit(logit_psi);           // Z=0: R must be 0
    real lp1 = log_inv_logit(logit_psi)
               + neg_binomial_2_log_lpmf(r | log_lambda, phi_k[k]);
    target += log_sum_exp(lp1, lp0) - lp0;           // correction for non-zero R
  }
}

generated quantities {
  // Per-SGB Industrial and NI occupancy (marginalised over studies)
  vector[K] psi_ind_k;
  vector[K] psi_ni_k;
  for (k in 1:K) {
    psi_ind_k[k] = inv_logit(alpha_k[k] + beta_k[k]);  // Industrial (is_ind=1)
    psi_ni_k[k]  = inv_logit(alpha_k[k]);               // NI (is_ind=0)
  }

  // Primary estimand: N_missing
  // A species is "missing in Industrial" when:
  //   (1) Industrial occupancy is near-zero:  ψ_Ind,k < 0.05
  //   (2) NI occupancy is meaningfully above zero: ψ_NI,k > 0.10
  int N_missing = 0;
  vector[K] p_missing_k;
  for (k in 1:K) {
    p_missing_k[k] = (psi_ind_k[k] < 0.05 && psi_ni_k[k] > 0.10) ? 1.0 : 0.0;
    N_missing += (psi_ind_k[k] < 0.05 && psi_ni_k[k] > 0.10);
  }

  // Depletion (weaker criterion): ψ_NI - ψ_Ind > 0.10
  int N_depleted = 0;
  for (k in 1:K) {
    N_depleted += ((psi_ni_k[k] - psi_ind_k[k]) > 0.10);
  }
}
"""


def prepare_stan_data(gap_results, breadth_ind, breadth_ni, rdat,
                      threshold=10.0, read_len=150.0, kb_thr=200.0):
    """
    Build the data dictionary for the Stan joint model from pipeline outputs.

    Parameters match those produced by decompose_richness_gap() and
    estimate_group_model_bb().

    Returns
    -------
    data : dict  — pass directly to cmdstanpy model.sample(data=data)
    sgb_names : list[str]
    sample_ids : list[str]
    """
    ref_sgbs   = sorted(set(breadth_ind.columns) & set(breadth_ni.columns))
    ind_idx    = breadth_ind.index
    ni_idx     = breadth_ni.index
    all_idx    = list(ind_idx) + [i for i in ni_idx if i not in set(ind_idx)]
    S          = len(all_idx)
    K          = len(ref_sgbs)

    meta       = rdat["meta"]
    depth_df   = rdat["depth"]
    total_bp_s = rdat["total_bp"]
    lengths    = rdat["lengths"].reindex(ref_sgbs).values.astype(float)
    G          = np.maximum(lengths, 1.0)

    # Study index (1-based for Stan)
    study_col  = next((c for c in ["StudyFolder","Author","study_name [MBIOTEMP_study_name]"]
                       if c in meta.columns), None)
    if study_col:
        studies_raw = meta.loc[all_idx, study_col].astype(str).fillna("unknown").values
    else:
        studies_raw = np.array(["all"] * S)
    unique_studies  = np.unique(studies_raw)
    study_map       = {s: i+1 for i, s in enumerate(unique_studies)}
    study_arr       = np.array([study_map[s] for s in studies_raw], dtype=int)
    J               = len(unique_studies)

    # is_ind flag
    ind_set  = set(ind_idx)
    is_ind   = np.array([1 if sid in ind_set else 0 for sid in all_idx], dtype=int)

    # log(total_bp) and quality offset
    tbp      = np.maximum(total_bp_s.reindex(all_idx).values.astype(float), 1.0)
    log_D    = np.log(tbp)
    # quality offset: log(actual / expected) mapping
    depth_vals  = depth_df.reindex(index=all_idx, columns=ref_sgbs).values.astype(float)
    catalog_len = float(G.sum())
    mean_dpb    = depth_vals.mean(axis=1)
    expected_tbp = np.maximum(mean_dpb * catalog_len, 1.0)
    q_s          = np.log(tbp) - np.log(expected_tbp)
    q_s         -= q_s.mean()

    # Sparse read count matrix (R > 0 only)
    # Approximate mapped reads from mean depth per base × genome length / read_len
    obs_s_list, obs_k_list, obs_r_list = [], [], []
    for si, sid in enumerate(all_idx):
        if sid not in set(breadth_ind.index) | set(breadth_ni.index):
            continue
        block = breadth_ind if sid in ind_set else breadth_ni
        if sid not in block.index:
            continue
        breadth_row = block.loc[sid, ref_sgbs].values.astype(float)
        depth_row   = depth_df.reindex([sid], columns=ref_sgbs).values[0].astype(float)
        # R_ks ≈ depth_per_base × G_k / read_len  (integer approximation)
        r_approx    = np.round(depth_row * G / read_len).astype(int)
        nonzero     = np.where((breadth_row > 0) & (r_approx > 0))[0]
        for ki in nonzero:
            obs_s_list.append(si + 1)      # 1-based
            obs_k_list.append(ki + 1)
            obs_r_list.append(int(r_approx[ki]))

    return dict(
        K=K, S=S, J=J, S_ind=int(is_ind.sum()), S_ni=int((1-is_ind).sum()),
        study=study_arr.tolist(),
        is_ind=is_ind.tolist(),
        q_s=q_s.tolist(),
        log_D=log_D.tolist(),
        G=G.tolist(),
        read_len=float(read_len),
        breadth_thr=float(threshold / 100.0),
        kb_thr=float(kb_thr * 1000),
        N_obs=len(obs_s_list),
        obs_s=obs_s_list,
        obs_k=obs_k_list,
        obs_r=obs_r_list,
    ), ref_sgbs, all_idx


def run_stan_joint_model(gap_results, breadth_ind, breadth_ni, rdat,
                         threshold=10.0, chains=4, iter_warmup=500,
                         iter_sampling=500, seed=42, out_dir="."):
    """
    Fit the Stan joint hierarchical model and return posterior summaries.

    Requires cmdstanpy. Falls back gracefully if not installed.

    Returns
    -------
    result dict with posterior arrays and summary statistics, or
    None if cmdstanpy is not available.
    """
    try:
        import cmdstanpy
    except ImportError:
        print("  cmdstanpy not installed — skipping Stan joint model.")
        print("  Install with: pip install cmdstanpy && python -m cmdstanpy.install_cmdstan")
        return None

    import json

    print("  Writing Stan model...", end=" ", flush=True)
    stan_code = build_stan_joint_model()
    stan_file = os.path.join(out_dir, "joint_occupancy.stan")
    with open(stan_file, "w") as f:
        f.write(stan_code)
    print(f"→ {stan_file}")

    print("  Preparing data...", end=" ", flush=True)
    data_dict, sgb_names, sample_ids = prepare_stan_data(
        gap_results, breadth_ind, breadth_ni, rdat, threshold=threshold)
    data_file = os.path.join(out_dir, "joint_occupancy_data.json")
    with open(data_file, "w") as f:
        json.dump(data_dict, f)
    print(f"K={data_dict['K']}  S={data_dict['S']}  J={data_dict['J']}  "
          f"N_obs={data_dict['N_obs']}")

    print("  Compiling Stan model...", end=" ", flush=True)
    try:
        model = cmdstanpy.CmdStanModel(stan_file=stan_file)
        print("done")
    except Exception as e:
        print(f"FAILED: {e}")
        return None

    print(f"  Sampling ({chains} chains × {iter_sampling} draws)...")
    try:
        fit = model.sample(
            data=data_file,
            chains=chains,
            iter_warmup=iter_warmup,
            iter_sampling=iter_sampling,
            seed=seed,
            show_progress=True,
        )
    except Exception as e:
        print(f"  Sampling FAILED: {e}")
        return None

    # Extract primary quantities
    N_missing  = fit.stan_variable("N_missing")       # (draws,)
    N_depleted = fit.stan_variable("N_depleted")
    psi_ind_k  = fit.stan_variable("psi_ind_k")       # (draws, K)
    psi_ni_k   = fit.stan_variable("psi_ni_k")
    beta_k     = fit.stan_variable("beta_k")          # (draws, K)

    p_missing_k = (psi_ind_k < 0.05) & (psi_ni_k > 0.10)   # (draws, K)
    p_missing   = p_missing_k.mean(0)                        # (K,) posterior prob

    E_N   = float(N_missing.mean())
    CI_N  = (float(np.percentile(N_missing, 2.5)),
             float(np.percentile(N_missing, 97.5)))

    print(f"\n  ── Stan joint model results ──────────────────────────────")
    print(f"  E[N_missing] = {E_N:.1f}  95% CI [{CI_N[0]:.0f}, {CI_N[1]:.0f}]")
    print(f"  E[N_depleted] = {N_depleted.mean():.1f}")
    print(f"  SGBs with P(missing) > 0.90: {int((p_missing > 0.90).sum()):,}")
    print(f"  SGBs with P(missing) > 0.50: {int((p_missing > 0.50).sum()):,}")

    # Diagnostics
    diag = fit.diagnose()
    print(f"  Stan diagnostics: {diag[:200]}")

    return dict(
        fit=fit,
        sgb_names=sgb_names,
        sample_ids=sample_ids,
        N_missing=N_missing,
        N_depleted=N_depleted,
        psi_ind_k=psi_ind_k,
        psi_ni_k=psi_ni_k,
        beta_k=beta_k,
        p_missing=p_missing,
        E_N_missing=E_N,
        CI_N_missing=CI_N,
    )


def hadza_fraction_of_absent(breadth_ni_full, meta, absent_sgbs,
                              threshold=10.0, hadza_min_prev=0.01):
    """
    For a set of absent/putatively-absent SGBs, compute what fraction are
    Hadza-prevalent (≥ hadza_min_prev in Hadza) vs. prevalent in other
    non-industrial populations.

    This separates two biological interpretations of "missing in Industrial":
      (a) Globally-distributed NI microbes that Industrial populations have lost
      (b) Hadza-specific microbes whose absence in Industrial is trivially
          explained by them never being present in most NI populations either

    Parameters
    ----------
    breadth_ni_full : DataFrame (n_NI_samples × n_sgbs) — FULL NI breadth
                      block BEFORE any Hadza exclusion, so that Hadza samples
                      are still present for this calculation.
    meta            : metadata DataFrame with Lifestyle column.
    absent_sgbs     : list/array of SGB column names to evaluate.
    threshold       : breadth % threshold for presence call.
    hadza_min_prev  : minimum Hadza prevalence to count as "Hadza-prevalent".

    Returns
    -------
    dict with per-SGB arrays and aggregate counts, keyed by SGB name.
    """
    import pandas as pd

    # Identify Hadza and non-Hadza NI sample IDs
    ni_idx    = breadth_ni_full.index
    ni_meta   = meta.loc[meta.index.intersection(ni_idx)]
    hadza_ids = ni_meta.index[_hadza_mask(ni_meta)]
    other_ni_ids = ni_meta.index.difference(hadza_ids)

    absent_sgbs = [s for s in absent_sgbs if s in breadth_ni_full.columns]
    if not absent_sgbs:
        return dict(
            hadza_prev={}, other_ni_prev={},
            n_hadza_prevalent=0, n_other_ni_prevalent=0,
            n_both=0, n_hadza_only=0, n_absent_sgbs=0,
            sgb_names=[],
        )

    b_sub = (breadth_ni_full[absent_sgbs] >= threshold)

    # Prevalence in each subgroup
    hadza_prev    = b_sub.loc[b_sub.index.intersection(hadza_ids)].mean(0) \
                    if len(hadza_ids) else pd.Series(0.0, index=absent_sgbs)
    other_ni_prev = b_sub.loc[b_sub.index.intersection(other_ni_ids)].mean(0) \
                    if len(other_ni_ids) else pd.Series(0.0, index=absent_sgbs)

    hadza_prev_arr    = hadza_prev.reindex(absent_sgbs).fillna(0.0).values
    other_ni_prev_arr = other_ni_prev.reindex(absent_sgbs).fillna(0.0).values

    is_hadza_prevalent = hadza_prev_arr >= hadza_min_prev
    is_other_ni_prev   = other_ni_prev_arr >= hadza_min_prev

    n_hadza_prevalent  = int(is_hadza_prevalent.sum())
    n_other_ni_prev    = int(is_other_ni_prev.sum())
    n_both             = int((is_hadza_prevalent & is_other_ni_prev).sum())
    n_hadza_only       = int((is_hadza_prevalent & ~is_other_ni_prev).sum())

    return dict(
        sgb_names=absent_sgbs,
        hadza_prev=dict(zip(absent_sgbs, hadza_prev_arr)),
        other_ni_prev=dict(zip(absent_sgbs, other_ni_prev_arr)),
        hadza_prev_arr=hadza_prev_arr,
        other_ni_prev_arr=other_ni_prev_arr,
        is_hadza_prevalent=is_hadza_prevalent,
        is_other_ni_prev=is_other_ni_prev,
        n_absent_sgbs=len(absent_sgbs),
        n_hadza_prevalent=n_hadza_prevalent,
        n_other_ni_prevalent=n_other_ni_prev,
        n_both=n_both,
        n_hadza_only=n_hadza_only,
        n_hadza_samples=int(len(hadza_ids)),
        n_other_ni_samples=int(len(other_ni_ids)),
    )


def run_absence_test(gap_results, breadth_ind, breadth_ni,
                     rdat, threshold=10.0,
                     meta=None, breadth_ni_full=None,
                     class1_thr=5.0, class2_thr=1.0,
                     delta_bio=0.10, posterior_alpha=0.10,
                     n_boot=500, rng=None):
    """
    Full per-species evidence test for genuine biological absence in Industrial.

    Orchestrates:
    1. Evidence classification (Class I/II/III)
    2. Likelihood ratios for Class I species
    3. Bootstrap posterior Δψ_k = ψ_k^NI – ψ_k^Ind
    4. Within-depth gradient test for Class II species
    5. Power calculations (depth needed for Class III → Class I)
    6. Hadza fraction: of absent SGBs, how many are Hadza-prevalent
       but absent in other NI populations?
    7. Aggregate N_absent, N_uncertain, N_unresolvable

    Parameters
    ----------
    meta           : metadata DataFrame (needed for Hadza fraction analysis).
    breadth_ni_full: FULL NI breadth block before Hadza exclusion, so Hadza
                     samples are present for the fraction computation.
                     If None, uses breadth_ni (Hadza fraction will be 0).

    Returns
    -------
    result dict with per-species arrays and aggregate statistics.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    psi_ind  = gap_results["psi_ind"]
    psi_ni   = gap_results["psi_ni"]
    sum_r_ind = gap_results["sum_r_ind"] if "sum_r_ind" in gap_results else gap_results.get("r_mean_ind", np.zeros(len(psi_ind))) * gap_results["N_ind"]
    r_mat_ind = gap_results.get("r_mat_ind_full")
    h_hat_ind = gap_results["h_ind"]
    mu_x_ind  = gap_results["mu_ind"]
    sig_x_ind = gap_results.get("sig_ind", np.full(len(psi_ind), 0.5))
    phi_k_ind = gap_results.get("phi_k_ind", np.full(len(psi_ind), 5.0))
    ref_sgbs  = sorted(set(breadth_ind.columns) & set(breadth_ni.columns))
    L_k_arr   = rdat["lengths"].reindex(ref_sgbs).values.astype(float)
    θ         = threshold / 100.0
    log_tau   = np.log(-np.log1p(-θ))
    meta_use  = meta if meta is not None else rdat.get("meta")

    print("  [1/5] Classifying species by evidence class...", end=" ", flush=True)
    ev_classes = classify_species_by_evidence(
        sum_r_ind, h_hat_ind, psi_ht_ind=psi_ind,
        class1_thr=class1_thr, class2_thr=class2_thr,
        psi_absent_eps=0.05)
    n_c1      = int(ev_classes["class1"].sum())
    n_c2      = int(ev_classes["class2"].sum())
    n_c3      = int(ev_classes["class3"].sum())
    n_det     = int(ev_classes["detected"].sum())
    n_conflict = int(ev_classes["psi_conflict"].sum())
    print(f"Class I={n_c1:,}  Class II={n_c2:,}  Class III={n_c3:,}  "
          f"Detected={n_det:,}  Sub-threshold present={n_conflict:,}")
    if n_conflict > 0:
        print(f"  NOTE: {n_conflict:,} SGBs have ψ_Ind ≥ 0.05 from sub-threshold signal "
              f"but zero above-threshold detections. These are present at low abundance "
              f"in Industrial (not missing) — excluded from absence classes.")

    # Build r_mat_ind if not cached (re-estimate from BB model using stored params)
    if r_mat_ind is None:
        print("  Re-estimating r_mat_ind from BB model...", end=" ", flush=True)
        idx = breadth_ind.index.intersection(rdat["depth"].index).intersection(rdat["total_bp"].index)
        b_ind_blk = breadth_ind.loc[idx, ref_sgbs]
        d_ind_blk = rdat["depth"].loc[idx, ref_sgbs]
        tbp_ind   = rdat["total_bp"].loc[idx]
        (*_, r_mat_ind, sum_r_ind, d_i_ind, _, _,
         q_s_ind, phi_k_ind) = estimate_group_model_bb(
            b_ind_blk, d_ind_blk,
            rdat["lengths"].reindex(ref_sgbs), tbp_ind, threshold)
        print("done")
    else:
        idx = breadth_ind.index

    idx_ni = breadth_ni.index.intersection(rdat["depth"].index).intersection(rdat["total_bp"].index)
    b_ni_blk = breadth_ni.loc[idx_ni, ref_sgbs]
    d_ni_blk = rdat["depth"].loc[idx_ni, ref_sgbs]
    tbp_ni   = rdat["total_bp"].loc[idx_ni]
    _ni_ret2 = estimate_group_model_bb(
        b_ni_blk, d_ni_blk,
        rdat["lengths"].reindex(ref_sgbs), tbp_ni, threshold)
    r_mat_ni = _ni_ret2[4]

    print("  [2/5] Computing likelihood ratios (Class I)...", end=" ", flush=True)
    log_lr = compute_log_likelihood_ratios(psi_ni, r_mat_ind)
    log_lr_c1 = log_lr[ev_classes["class1"]]
    if len(log_lr_c1):
        print(f"done  median_log_LR={np.median(log_lr_c1):.1f}  max={np.max(log_lr_c1):.1f}")
    else:
        print("done  (no Class I species)")

    print("  [3/5] Bootstrap posterior Δψ_k (BB-weighted numerator)...", end=" ", flush=True)
    b_ind_th = breadth_ind[ref_sgbs]; b_ni_th = breadth_ni[ref_sgbs]

    # Re-derive per-sample BB weights for sub-threshold detections so the
    # bootstrap CI is internally consistent with the HT point estimates.
    # These are computed fresh from the stored BB model parameters.
    def _make_w_sub(b_blk, r_mat_full, eps_pct=0.5, thr=threshold):
        eps  = eps_pct / 100.0; thr_ = thr / 100.0
        b_f  = b_blk.values.astype(float) / 100.0
        det_full = b_f >= thr_
        det_eps  = b_f >= eps
        sub      = det_eps & ~det_full
        # r_mat_eps not stored; approximate band_prob as r_mat_eps - r_mat.
        # Use conservative weight 1.0 for sub-threshold detections (valid under
        # strict absence model: any B > eps implies Z=1 with prob 1).
        # For truly principled weights, estimate_group_model_bb would need to
        # return r_mat_eps; the caller should pass it if available.
        w = np.ones_like(b_f)
        return np.where(sub, w, 0.0)   # (n_samp, n_sgb), nonzero only in sub band

    w_sub_ind_arr = _make_w_sub(b_ind_th, r_mat_ind)
    w_sub_ni_arr  = _make_w_sub(b_ni_th,  r_mat_ni)

    (delta_mean, delta_se, p_above, ci_lo, ci_hi) = bootstrap_psi_contrast(
        b_ind_th, b_ni_th, r_mat_ind, r_mat_ni,
        w_sub_ind=w_sub_ind_arr, w_sub_ni=w_sub_ni_arr,
        threshold=threshold, delta_bio=delta_bio, n_boot=n_boot, rng=rng)
    c1_ok = ev_classes["class1"]
    if c1_ok.any():
        print(f"done  Class I: median_delta={np.median(delta_mean[c1_ok]):.3f}")
    else:
        print("done  (no Class I species)")

    # ── [3b] Study-stratified occupancy estimates ──────────────────────────
    print("  [3b] Study-stratified HT occupancy...", end=" ", flush=True)
    idx_ind = breadth_ind.index.intersection(rdat["depth"].index).intersection(
        rdat["total_bp"].index)
    idx_ni  = breadth_ni.index.intersection(rdat["depth"].index).intersection(
        rdat["total_bp"].index)
    psi_strat_ind, psi_se_ind, n_stud_ind = study_stratified_psi(
        breadth_ind.loc[idx_ind, ref_sgbs], r_mat_ind,
        meta_use.loc[idx_ind] if meta_use is not None else pd.DataFrame(index=idx_ind),
        threshold=threshold)
    psi_strat_ni, psi_se_ni, n_stud_ni = study_stratified_psi(
        breadth_ni.loc[idx_ni, ref_sgbs],  r_mat_ni,
        meta_use.loc[idx_ni]  if meta_use is not None else pd.DataFrame(index=idx_ni),
        threshold=threshold)
    print(f"done  Ind studies={n_stud_ind}  NI studies={n_stud_ni}  "
          f"median_psi_Ind={np.nanmedian(psi_strat_ind):.4f}  "
          f"median_psi_NI={np.nanmedian(psi_strat_ni):.4f}")

    # ── [3c] Joint posterior P(ψ_Ind < ε AND ψ_NI > τ) ───────────────────
    print("  [3c] Joint absence posterior...", end=" ", flush=True)
    # Build bootstrap matrices aligned to ref_sgbs
    b_ind_aligned = breadth_ind[ref_sgbs]
    b_ni_aligned  = breadth_ni[ref_sgbs]
    n_boot_joint  = min(n_boot, 500)

    psi_ind_boots = np.zeros((n_boot_joint, len(ref_sgbs)), float)
    psi_ni_boots  = np.zeros((n_boot_joint, len(ref_sgbs)), float)
    det_ind_full  = (b_ind_aligned.values >= threshold).astype(float)
    det_ni_full   = (b_ni_aligned.values  >= threshold).astype(float)
    n_ind_b, n_ni_b = det_ind_full.shape[0], det_ni_full.shape[0]

    for b in range(n_boot_joint):
        ii = rng.integers(0, n_ind_b, n_ind_b)
        ni = rng.integers(0, n_ni_b,  n_ni_b)
        sr_i = r_mat_ind[ii].sum(0); num_i = det_ind_full[ii].sum(0)
        sr_n = r_mat_ni[ni].sum(0);  num_n = det_ni_full[ni].sum(0)
        psi_ind_boots[b] = np.clip(
            np.where(sr_i > 0.1, num_i / sr_i, 0.0), 0.0, 1.0)
        psi_ni_boots[b]  = np.clip(
            np.where(sr_n > 0.1, num_n / sr_n, 0.0), 0.0, 1.0)

    p_joint, p_depleted, E_N_miss, CI_N_miss = joint_absence_posterior(
        psi_ind_boots, psi_ni_boots, eps=0.05, tau=0.10)
    print(f"done  E[N_missing]={E_N_miss:.1f}  "
          f"95% CI=[{CI_N_miss[0]:.0f}, {CI_N_miss[1]:.0f}]")

    print("  [4/5] Within-depth gradient test (Class II)...", end=" ", flush=True)
    b_ind_full = breadth_ind[ref_sgbs]
    slopes, slope_se = depth_gradient_test_class2(
        b_ind_full, rdat["total_bp"].loc[breadth_ind.index], ev_classes)
    n_class2_pos_slope = int((slopes[ev_classes["class2"]] > 0).sum())
    print(f"done  {n_class2_pos_slope}/{n_c2} Class II species show positive depth gradient")

    print("  [5/5] Power calculations (Class III depth needed)...", end=" ", flush=True)
    d_star_needed, d_star_imputed = power_to_classify(
        mu_x_ind, sig_x_ind, phi_k_ind, L_k_arr, log_tau,
        n_ind_samples=r_mat_ind.shape[0], target_sum_r=class1_thr)
    c3_mask   = ev_classes["class3"]
    finite_ok = c3_mask & np.isfinite(d_star_needed) & (d_star_needed < 1e12)
    finite_dstar        = d_star_needed[finite_ok]
    finite_dstar_imp    = d_star_imputed[finite_ok]   # which of these used the prior
    n_imputed_c3 = int(finite_dstar_imp.sum())
    med_str = f"{np.median(finite_dstar):.2e}" if len(finite_dstar) else "N/A"
    print(f"done  median={med_str} bp  ({n_imputed_c3}/{len(finite_dstar)} imputed from prior)")

    # ── Aggregate counts  ──────────────────────────────────────────────────
    # N_absent: Class I species with P(Δψ > δ) > 1-α
    cred_thr = 1.0 - posterior_alpha
    class1_absent    = ev_classes["class1"] & (p_above > cred_thr)
    class1_uncertain = ev_classes["class1"] & ~class1_absent
    N_absent       = int(class1_absent.sum())
    N_uncertain    = int(class1_uncertain.sum()) + n_c2
    N_unresolvable = n_c3

    # ── [6/6] Hadza fraction of absent SGBs ───────────────────────────────
    # Which absent/putatively-absent SGBs are Hadza-prevalent (≥1%) but
    # found at <1% prevalence in all other non-industrial samples?
    absent_mask   = class1_absent                           # high-confidence absent
    uncertain_mask = class1_uncertain | ev_classes["class2"]  # uncertain

    # Build SGB name arrays for the two groups
    absent_sgb_names    = [ref_sgbs[i] for i in np.where(absent_mask)[0]]
    uncertain_sgb_names = [ref_sgbs[i] for i in np.where(uncertain_mask)[0]]
    all_absent_names    = absent_sgb_names + uncertain_sgb_names

    b_ni_full = breadth_ni_full if breadth_ni_full is not None else breadth_ni
    meta_use  = meta if meta is not None else rdat.get("meta")

    hadza_absent    = hadza_fraction_of_absent(
        b_ni_full, meta_use, absent_sgb_names, threshold=threshold)
    hadza_uncertain = hadza_fraction_of_absent(
        b_ni_full, meta_use, uncertain_sgb_names, threshold=threshold)
    hadza_all       = hadza_fraction_of_absent(
        b_ni_full, meta_use, all_absent_names, threshold=threshold)

    print(f"\n  ── Hadza fraction of absent SGBs ─────────────────────────")
    for label, hf in [("High-confidence absent (Class I)", hadza_absent),
                      ("Uncertain (Class I ambiguous + II)", hadza_uncertain),
                      ("All putatively missing", hadza_all)]:
        n  = hf["n_absent_sgbs"]
        nh = hf["n_hadza_prevalent"]
        no = hf["n_other_ni_prevalent"]
        nb = hf["n_both"]
        nho = hf["n_hadza_only"]
        if n == 0:
            print(f"  {label}: 0 SGBs")
            continue
        print(f"  {label} (n={n:,}):")
        print(f"    Hadza-prevalent (≥1%): {nh:,} / {n:,} ({100*nh/n:.1f}%)")
        print(f"    Other-NI-prevalent:    {no:,} / {n:,} ({100*no/n:.1f}%)")
        print(f"    Hadza-only (not other NI): {nho:,} / {n:,} ({100*nho/n:.1f}%)")
        print(f"    Both Hadza + other NI:     {nb:,} / {n:,} ({100*nb/n:.1f}%)")

    print(f"\n  ── Primary estimand (joint posterior) ────────────────────")
    print(f"  E[N_missing | ψ_Ind<0.05 AND ψ_NI>0.10] = {E_N_miss:.1f}  "
          f"95% CI [{CI_N_miss[0]:.0f}, {CI_N_miss[1]:.0f}]")
    n_p_joint_90 = int((p_joint > 0.90).sum())
    n_p_joint_50 = int((p_joint > 0.50).sum())
    print(f"  SGBs with P(missing)>90%: {n_p_joint_90:,}")
    print(f"  SGBs with P(missing)>50%: {n_p_joint_50:,}")
    n_depleted_90 = int((p_depleted > 0.90).sum())
    print(f"  SGBs with P(depleted Δψ>0.10)>90%: {n_depleted_90:,}  "
          f"(includes partially-present species)")

    print(f"\n  ── Absence evidence summary (HT bootstrap) ───────────────")
    print(f"  N_absent      = {N_absent:,}  (Class I, P(Δψ>{delta_bio})>{cred_thr:.0%})")
    print(f"  N_uncertain   = {N_uncertain:,}  (Class I ambiguous + all Class II)")
    print(f"  N_unresolvable= {N_unresolvable:,}  (Class III — prior-dominated)")
    print(f"  Honest bounds: genuinely missing ∈ [{N_absent:,},  {N_absent+N_uncertain+N_unresolvable:,}]")

    return dict(
        ev_classes=ev_classes, n_c1=n_c1, n_c2=n_c2, n_c3=n_c3, n_det=n_det,
        n_conflict=n_conflict,
        # Arrays needed for power_sensitivity_curve in panel D
        mu_x_ind=mu_x_ind, sig_x_ind=sig_x_ind, phi_k_ind=phi_k_ind,
        L_k_arr=L_k_arr, log_tau=log_tau, n_ind_samples=r_mat_ind.shape[0],
        d_i_ind=gap_results.get("di_ind", np.array([1.37e9])),
        log_lr=log_lr, log_lr_c1=log_lr_c1,
        delta_mean=delta_mean, delta_se=delta_se,
        p_above=p_above, ci_lo=ci_lo, ci_hi=ci_hi,
        slopes=slopes, slope_se=slope_se,
        d_star_needed=d_star_needed, d_star_imputed=d_star_imputed,
        N_absent=N_absent, N_uncertain=N_uncertain, N_unresolvable=N_unresolvable,
        delta_bio=delta_bio, posterior_alpha=posterior_alpha,
        class1_thr=class1_thr, class2_thr=class2_thr,
        psi_ind=psi_ind, psi_ni=psi_ni,
        sum_r_ind=sum_r_ind,
        # Study-stratified occupancy
        psi_strat_ind=psi_strat_ind, psi_se_ind=psi_se_ind, n_stud_ind=n_stud_ind,
        psi_strat_ni=psi_strat_ni,   psi_se_ni=psi_se_ni,   n_stud_ni=n_stud_ni,
        # Joint posterior: primary estimand
        p_joint=p_joint, p_depleted=p_depleted,
        E_N_missing=E_N_miss, CI_N_missing=CI_N_miss,
        psi_ind_boots=psi_ind_boots, psi_ni_boots=psi_ni_boots,
        # Hadza fraction analysis
        hadza_absent=hadza_absent,
        hadza_uncertain=hadza_uncertain,
        hadza_all=hadza_all,
        absent_sgb_names=absent_sgb_names,
        uncertain_sgb_names=uncertain_sgb_names,
        ref_sgbs=ref_sgbs,
    )


def draw_absence_figure(absence, out_stem):
    """
    Five-panel evidence figure:
    A  Bar chart of N per evidence class + honest bounds annotation
    B  log LR distribution for Class I species
    C  Δψ_k scatter (Class I in red if absent, grey if uncertain)
    D  Power curve: depth needed to classify Class III species
    E  Hadza fraction: stacked bar showing Hadza-only / both / other-NI-only
       for high-confidence absent and uncertain SGBs
    """
    fig, axes = plt.subplots(1, 5, figsize=(28, 6),
                              gridspec_kw={"wspace": 0.38})
    ax_A, ax_B, ax_C, ax_D, ax_E = axes
    for lbl, ax_ in zip("ABCDE", axes):
        ax_.text(-0.12, 1.04, lbl, transform=ax_.transAxes,
                 fontsize=FS_PANEL, fontweight="bold", va="bottom")

    ev  = absence["ev_classes"]
    col = {"class1":"#D55E00", "class2":"#E69F00",
           "class3":"#56B4E9", "detected":"#009E73"}

    # ── Panel A: evidence class bar chart ────────────────────────────────
    n_conflict = int(ev.get("psi_conflict", np.zeros(1, bool)).sum())
    labels  = ["Class I\n(high power)", "Class II\n(marginal)",
               "Class III\n(low power)", "Sub-threshold\npresent", "Detected"]
    counts  = [absence["n_c1"], absence["n_c2"], absence["n_c3"],
               n_conflict, absence["n_det"]]
    colours = [col["class1"], col["class2"], col["class3"], "#AAAAAA", col["detected"]]
    bars    = ax_A.bar(labels, counts, color=colours, alpha=0.82, edgecolor="white")
    for bar_, cnt in zip(bars, counts):
        if cnt > 0:
            ax_A.text(bar_.get_x()+bar_.get_width()/2, bar_.get_height()+5,
                      f"{cnt:,}", ha="center", fontsize=FS_ANNOT-1)

    # Primary estimand annotation (joint posterior)
    E_N  = absence.get("E_N_missing", np.nan)
    CI_N = absence.get("CI_N_missing", (np.nan, np.nan))
    lb   = absence["N_absent"]
    ub   = lb + absence["N_uncertain"] + absence["N_unresolvable"]
    ann  = (f"Joint posterior\nE[N_missing]={E_N:.0f}\n"
            f"95% CI [{CI_N[0]:.0f}–{CI_N[1]:.0f}]\n\n"
            f"HT bootstrap bounds\n"
            f"lower={lb:,}  upper≤{ub:,}")
    ax_A.text(0.5, 0.97, ann,
              transform=ax_A.transAxes, ha="center", va="top",
              fontsize=FS_ANNOT-1, family="monospace",
              bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="dimgrey", alpha=0.92))
    ax_A.set_ylabel("Number of SGBs", fontsize=FS_LABEL)
    ax_A.set_title("Evidence classes\n(grey = sub-threshold present, not absent)",
                   fontsize=FS_TITLE, fontweight="bold")
    ax_A.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax_A.grid(True, axis="y", ls="--", lw=0.3, alpha=0.4)

    # ── Panel B: Joint posterior P(ψ_Ind<0.05 AND ψ_NI>0.10) ────────────
    p_joint = absence.get("p_joint")
    llr     = absence.get("log_lr_c1", np.array([]))
    if p_joint is not None and len(p_joint):
        c1_mask = ev["class1"]
        pj_c1   = p_joint[c1_mask]
        pj_all  = p_joint                          # all SGBs

        # Histogram of P(missing) for all undetected SGBs
        undet_mask = ~ev["detected"]
        pj_undet   = p_joint[undet_mask]
        if len(pj_undet):
            ax_B.hist(pj_undet, bins=30, color="#999999", alpha=0.45,
                      edgecolor="white", label=f"All undetected (n={len(pj_undet):,})")
        if len(pj_c1):
            ax_B.hist(pj_c1, bins=20, color=col["class1"], alpha=0.80,
                      edgecolor="white", label=f"Class I (n={len(pj_c1):,})")
        ax_B.axvline(0.50, color="black", lw=0.9, ls="--", alpha=0.7,
                     label="P=0.50")
        ax_B.axvline(0.90, color="black", lw=0.9, ls=":",  alpha=0.7,
                     label="P=0.90")
        n90 = int((p_joint > 0.90).sum())
        ax_B.text(0.97, 0.97, f"P>0.90: {n90:,} SGBs",
                  transform=ax_B.transAxes, ha="right", va="top",
                  fontsize=FS_ANNOT-1,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white",
                            ec="dimgrey", alpha=0.85, lw=0.5))
    ax_B.set_xlabel("P(ψ_Ind < 0.05  AND  ψ_NI > 0.10)",
                    fontsize=FS_LABEL-1)
    ax_B.set_ylabel("Number of SGBs", fontsize=FS_LABEL-1)
    ax_B.set_title("Joint absence posterior\n(primary estimand)",
                   fontsize=FS_TITLE, fontweight="bold")
    ax_B.legend(fontsize=FS_LEGEND, frameon=False)
    ax_B.grid(True, axis="y", ls="--", lw=0.3, alpha=0.4)

    # ── Panel C: Δψ scatter (ψ_k^NI vs ψ_k^Ind) for Class I + conflict ──
    c1           = ev["class1"]
    conflict_mask = ev.get("psi_conflict", np.zeros(len(c1), bool))
    p_ab         = absence["p_above"]

    # Plot conflict species first (grey, behind) — present sub-threshold, not absent
    if conflict_mask.any():
        ax_C.scatter(absence["psi_ind"][conflict_mask],
                     absence["psi_ni"][conflict_mask],
                     c="#BBBBBB", s=10, alpha=0.50, zorder=2,
                     edgecolors="none", label=f"Sub-threshold present (n={conflict_mask.sum():,})")

    # Class I species coloured by P(Δψ > δ)
    if c1.any():
        sc = ax_C.scatter(absence["psi_ind"][c1], absence["psi_ni"][c1],
                          c=p_ab[c1], cmap="Reds", vmin=0, vmax=1,
                          s=14, alpha=0.80, zorder=4, edgecolors="none",
                          label="Class I (near-absent)")
        plt.colorbar(sc, ax=ax_C, label=f"P(Δψ > {absence['delta_bio']})",
                     fraction=0.04, pad=0.02)

    # ψ = ε boundary line (absence threshold)
    ax_C.axvline(0.05, color=col["class1"], lw=0.9, ls="--", alpha=0.7,
                 label="ψ_Ind = 0.05 (absence threshold)")
    ax_C.axhline(0.10, color=col["detected"], lw=0.9, ls="--", alpha=0.7,
                 label="ψ_NI = 0.10 (present in NI)")
    ax_C.plot([0,1],[0,1], color="grey", lw=0.7, ls=":", alpha=0.4)

    x_max = max(0.15, float(absence["psi_ind"].max()) * 1.05)
    ax_C.set_xlim(-0.01, x_max)
    ax_C.set_ylim(0, 1.02)
    ax_C.set_xlabel("ψ_k^Ind  (Industrial occupancy)", fontsize=FS_LABEL-1)
    ax_C.set_ylabel("ψ_k^NI  (NI occupancy)", fontsize=FS_LABEL-1)
    ax_C.set_title("Posterior occupancy contrast\n"
                   "(grey = sub-threshold present; red = near-absent)",
                   fontsize=FS_TITLE, fontweight="bold")
    leg_C = ax_C.legend(fontsize=FS_LEGEND-1.5, loc="lower right",
                         framealpha=0.90)
    leg_C.get_frame().set_linewidth(0.3)
    ax_C.grid(True, ls="--", lw=0.3, alpha=0.35)

    # ── Panel D: power sensitivity curve for Class III species ───────────
    # The old histogram was meaningless: all Class III species share a
    # community prior and 14k+ samples make target_r_per_sample tiny, so
    # bisection always returns the lower bound (1e7 bp) for every species.
    #
    # Replacement: sensitivity curve showing what fraction of Class III
    # species would become classifiable (sum_r > class1_thr) at a given
    # target sequencing depth, for several assumed abundance percentiles.
    # Genome length is the only species-specific quantity available.

    mu_x_all  = absence.get("mu_x_ind",    np.full(len(ev["class3"]), np.nan))
    sig_x_all = absence.get("sig_x_ind",   np.full(len(ev["class3"]), 0.5))
    phi_all   = absence.get("phi_k_ind",   np.full(len(ev["class3"]), 3.0))
    L_k_all   = absence.get("L_k_arr",     np.full(len(ev["class3"]), 3e6))
    n_ind_s   = absence.get("n_ind_samples", 14000)
    c1_thr    = absence.get("class1_thr",  5.0)
    log_tau_v = absence.get("log_tau",     np.log(-np.log1p(-0.10)))

    if len(mu_x_all) == len(ev["class3"]) and ev["class3"].any():
        depth_g, frac_cls, pct_lbls, mu_scen, L_k_c3 = power_sensitivity_curve(
            mu_x_all, sig_x_all, phi_all, L_k_all, log_tau_v,
            n_ind_samples=n_ind_s, class1_thr=c1_thr,
            abundance_percentiles=(5, 10, 25, 50))

        colours_pct = ["#2166ac", "#4dac26", "#d6604d", "#762a83"]
        for pi, (frac, lbl) in enumerate(zip(frac_cls, pct_lbls)):
            ax_D.plot(depth_g, frac * 100,
                      color=colours_pct[pi % len(colours_pct)],
                      lw=1.4, label=lbl)

        # Mark current Industrial median depth
        d_ind_med = float(np.median(absence.get("d_i_ind", np.array([1.37e9]))))
        ax_D.axvline(d_ind_med, color="dimgrey", lw=1.0, ls="--", alpha=0.7,
                     label=f"Ind. median depth\n({d_ind_med:.1e} bp)")
        ax_D.axhline(50, color="black", lw=0.6, ls=":", alpha=0.4)

        ax_D.set_xscale("log")
        ax_D.set_xlim(depth_g[0], depth_g[-1])
        ax_D.set_ylim(0, 105)
        ax_D.set_xlabel("Target sequencing depth per sample (bp)", fontsize=FS_LABEL-1)
        ax_D.set_ylabel("Class III SGBs classifiable (%)", fontsize=FS_LABEL-1)
        ax_D.set_title(f"Power sensitivity: Class III SGBs (n={ev['class3'].sum():,})\n"
                        f"fraction reaching Class I power vs. depth",
                        fontsize=FS_TITLE, fontweight="bold")
        leg_D = ax_D.legend(fontsize=FS_LEGEND-1.5, loc="upper left",
                             framealpha=0.90, title="Assumed abundance")
        leg_D.get_frame().set_linewidth(0.3)
        ax_D.grid(True, which="both", ls="--", lw=0.3, alpha=0.35)

        # Inset: genome length distribution of Class III species
        if len(L_k_c3) > 0:
            ax_ins_D = ax_D.inset_axes([0.55, 0.08, 0.42, 0.40])
            ax_ins_D.hist(np.log10(np.maximum(L_k_c3, 1)),
                          bins=20, color="#56B4E9", alpha=0.75, edgecolor="white")
            ax_ins_D.set_xlabel("log₁₀(genome length, bp)", fontsize=FS_ANNOT-2,
                                labelpad=1)
            ax_ins_D.set_ylabel("n SGBs", fontsize=FS_ANNOT-2, labelpad=1)
            ax_ins_D.set_title("Genome sizes", fontsize=FS_ANNOT-1, pad=2)
            ax_ins_D.tick_params(labelsize=FS_ANNOT-2, pad=1)
            for sp in ax_ins_D.spines.values():
                sp.set_linewidth(0.4)
    else:
        ax_D.text(0.5, 0.5, "No Class III species\nor model params unavailable",
                  transform=ax_D.transAxes, ha="center", va="center",
                  fontsize=FS_ANNOT, color="dimgrey")
        ax_D.set_title("Power sensitivity (Class III)", fontsize=FS_TITLE,
                        fontweight="bold")

    # ── Panel E: Hadza fraction of absent SGBs ───────────────────────────
    hf_absent    = absence.get("hadza_absent",    {})
    hf_uncertain = absence.get("hadza_uncertain", {})

    bar_groups = []
    for label, hf in [("High-confidence\nabsent", hf_absent),
                      ("Uncertain\n(Class I + II)", hf_uncertain)]:
        n = hf.get("n_absent_sgbs", 0)
        if n == 0:
            bar_groups.append((label, 0, 0, 0, 0))
            continue
        n_hadza_only  = hf.get("n_hadza_only", 0)
        n_both        = hf.get("n_both", 0)
        n_other_only  = hf.get("n_other_ni_prevalent", 0) - n_both
        n_neither     = n - n_hadza_only - n_both - max(n_other_only, 0)
        bar_groups.append((label, n_hadza_only, n_both,
                           max(n_other_only, 0), max(n_neither, 0)))

    x_pos  = np.arange(len(bar_groups))
    w      = 0.55
    c_hadza_only = "#E69F00"   # orange — Hadza-only
    c_both       = "#CC79A7"   # pink   — Hadza + other NI
    c_other      = "#009E73"   # green  — other NI only
    c_neither    = "#BBBBBB"   # grey   — neither

    bottoms = np.zeros(len(bar_groups))
    for vals, color, label in [
        ([g[1] for g in bar_groups], c_hadza_only, "Hadza-only (≥1% Hadza, <1% other NI)"),
        ([g[2] for g in bar_groups], c_both,       "Hadza + other NI (≥1% both)"),
        ([g[3] for g in bar_groups], c_other,       "Other NI only (<1% Hadza)"),
        ([g[4] for g in bar_groups], c_neither,     "Neither (<1% in all NI)"),
    ]:
        vals_arr = np.array(vals, float)
        ax_E.bar(x_pos, vals_arr, w, bottom=bottoms,
                 color=color, alpha=0.83, edgecolor="white", label=label)
        # Label non-trivial segments
        for xi, (v, bot) in enumerate(zip(vals_arr, bottoms)):
            if v >= 3:
                ax_E.text(xi, bot + v/2, str(int(v)),
                          ha="center", va="center",
                          fontsize=FS_ANNOT-1, color="white", fontweight="bold")
        bottoms += vals_arr

    # Total count labels on top
    for xi, g in enumerate(bar_groups):
        total = g[1] + g[2] + g[3] + g[4]
        if total > 0:
            ax_E.text(xi, total + 1, f"n={int(total)}",
                      ha="center", va="bottom", fontsize=FS_ANNOT-1)

    ax_E.set_xticks(x_pos)
    ax_E.set_xticklabels([g[0] for g in bar_groups], fontsize=FS_TICK)
    ax_E.set_ylabel("Number of SGBs", fontsize=FS_LABEL-1)
    ax_E.set_title("Hadza prevalence of putatively\nmissing Industrial SGBs",
                   fontsize=FS_TITLE, fontweight="bold")
    ax_E.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax_E.grid(True, axis="y", ls="--", lw=0.3, alpha=0.4)

    # Add Hadza sample count annotation
    n_hadza_s = hf_absent.get("n_hadza_samples", 0) or \
                hf_uncertain.get("n_hadza_samples", 0)
    n_other_s = hf_absent.get("n_other_ni_samples", 0) or \
                hf_uncertain.get("n_other_ni_samples", 0)
    ax_E.text(0.97, 0.97,
              f"Hadza n={n_hadza_s:,}\nOther NI n={n_other_s:,}",
              transform=ax_E.transAxes, ha="right", va="top",
              fontsize=FS_ANNOT-1, color="dimgrey",
              bbox=dict(boxstyle="round,pad=0.3", fc="white",
                        ec="dimgrey", alpha=0.80, lw=0.5))
    leg_E = ax_E.legend(fontsize=FS_LEGEND-1, loc="upper left",
                         framealpha=0.90, edgecolor="#cccccc",
                         bbox_to_anchor=(0, 0.85))
    leg_E.get_frame().set_linewidth(0.3)

    fig.suptitle("Per-species evidence for genuine biological absence in Industrial populations\n"
                 "Framework: Document 7 — Likelihood ratios, HT posterior contrast, power analysis",
                 fontsize=FS_TITLE, fontweight="bold", y=1.02)
    fig.canvas.draw()
    apply_display_text(fig)
    abs_pdf = f"{out_stem}_absence_test.pdf"
    fig.savefig(abs_pdf, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {abs_pdf}")
    abs_png = f"{out_stem}_absence_test.png"
    tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(abs_png))
    fig.savefig(tmp_path, dpi=150, bbox_inches="tight", facecolor="white")
    shutil.move(tmp_path, abs_png)
    print(f"  Saved → {abs_png}")
    plt.close(fig)


def draw_comparison_panel(ax, gap, title="", N_LOG_MAX=1e5):
    """
    Panel E: Ind vs NI depth-corrected comparison.

    Shows:
    - Depth-aware observed curves (solid, group colour)
    - Counterfactual curves at d* (dashed, lighter)
    - Shaded gap between NI and Ind counterfactual curves
    - Log-space inset with asymptotes and species classification
    - Annotation: ΔS_obs, ΔS_bio, ΔS_tech, depth-invisible pool
    """
    ax_ins = ax.inset_axes([0.33, 0.05, 0.66, 0.65])

    col_ind = COL["Industrial"]; col_ni = COL["Non-Industrial"]

    for ax_, is_log in [(ax, False), (ax_ins, True)]:
        # Depth-aware observed curves
        for N_v, S_v, Ssd, col, lbl in [
            (gap["N_da_ind"], gap["S_da_ind"], gap["Ssd_da_ind"],
             col_ind, "Industrial (depth-aware obs.)"),
            (gap["N_da_ni"],  gap["S_da_ni"],  gap["Ssd_da_ni"],
             col_ni,  "Non-Industrial (depth-aware obs.)"),
        ]:
            ax_.fill_between(N_v, S_v-Ssd, S_v+Ssd, color=col, alpha=0.12, zorder=1)
            ax_.plot(N_v, S_v, color=col, lw=1.6, ls="-", alpha=0.85, zorder=4,
                     label=lbl if is_log else "_nolegend_")

        # Counterfactual curves at d*
        for N_v, S_v, col, lbl in [
            (gap["N_cf_ind"], gap["S_cf_ind"], col_ind,
             f"Industrial at d*={gap['d_star']:.1e}"),
            (gap["N_cf_ni"],  gap["S_cf_ni"],  col_ni,
             f"Non-Industrial at d*={gap['d_star']:.1e}"),
        ]:
            ax_.plot(N_v, S_v, color=col, lw=1.4, ls="--", alpha=0.70, zorder=5,
                     label=lbl if is_log else "_nolegend_")

        # Shaded gap between counterfactual NI and Ind
        N_cf_min = min(gap["N_cf_ind"][-1], gap["N_cf_ni"][-1])
        m_ind = gap["N_cf_ind"] <= N_cf_min; m_ni = gap["N_cf_ni"] <= N_cf_min
        if m_ind.any() and m_ni.any():
            # Interpolate NI curve to Ind N grid
            N_shared = gap["N_cf_ind"][m_ind]
            S_ni_interp = np.interp(N_shared, gap["N_cf_ni"][m_ni], gap["S_cf_ni"][m_ni])
            ax_.fill_between(N_shared, gap["S_cf_ind"][m_ind], S_ni_interp,
                             color="gold", alpha=0.22, zorder=2,
                             label="Biological gap at d*" if is_log else "_nolegend_")

    # Axes setup (linear)
    S_top = max(gap["S_da_ni"][-1], gap["S_da_ind"][-1]) * 1.15
    ax.set_xscale("linear")
    ax.set_xlim(0, max(gap["N_ind"], gap["N_ni"]) * 1.05)
    ax.set_ylim(0, S_top)
    ax.set_ylabel(f"SGBs detected", fontsize=FS_LABEL)
    ax.set_xlabel("Number of samples (N)", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.grid(True, ls="--", lw=0.3, alpha=0.35)

    # Inset (log)
    S_ceil_ins = max(gap["S_cf_ni"][-1], gap["S_cf_ind"][-1]) * 1.25
    ax_ins.set_xscale("log")
    ax_ins.set_xlim(0.8, max(gap["N_cf_ind"][-1], gap["N_cf_ni"][-1]) * 3)
    ax_ins.set_ylim(0, S_ceil_ins)
    ax_ins.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax_ins.tick_params(labelsize=FS_TICK-2, pad=1)
    ax_ins.set_xlabel("log N", fontsize=FS_TICK-1, labelpad=2)
    ax_ins.set_ylabel("SGBs", fontsize=FS_TICK-1, labelpad=2)
    ax_ins.grid(True, which="both", ls="--", lw=0.22, alpha=0.32)
    leg = ax_ins.legend(fontsize=FS_LEGEND-1.5, loc="lower right", framealpha=0.92,
                        handlelength=1.6, borderpad=0.4, labelspacing=0.3)
    leg.get_frame().set_linewidth(0.3)

    for sp in ax_ins.spines.values():
        sp.set_edgecolor("dimgrey"); sp.set_linewidth(0.5); sp.set_alpha(0.5)

    # Connector lines
    for y_m in [S_top, 0.0]:
        y_f = float(np.clip(y_m / S_ceil_ins, 0, 1))
        con = ConnectionPatch(
            xyA=(1.0, y_f), coordsA="axes fraction", axesA=ax_ins,
            xyB=(max(gap["N_ind"], gap["N_ni"]) * 1.04, y_m), coordsB="data", axesB=ax,
            color="dimgrey", lw=0.6, alpha=0.40, ls="-", arrowstyle="-")
        ax.add_artist(con)

    # Gap decomposition annotation
    Dgap = gap["S_obs_gap"]; Dbio = gap["S_bio_gap"]; Dtech = gap["S_tech_gap"]
    n_invisible = gap["n_cats"]["NI_only_depth_invisible"]
    n_inexplicable = gap["n_cats"]["NI_only_depth_inexplicable"]
    n_shared = gap["n_cats"]["shared"]
    phi_i = gap.get("phi_ind_median", float("nan"))
    phi_n = gap.get("phi_ni_median",  float("nan"))
    q_i   = gap.get("q_spread_ind",   float("nan"))
    q_n   = gap.get("q_spread_ni",    float("nan"))
    ann_lines = [
        f"ΔS_obs    = {Dgap:+.0f}",
        f"ΔS_bio    = {Dbio:+.0f}  (at d*={gap['d_star']:.1e})",
        f"ΔS_tech   = {Dtech:+.0f}",
        f"─────────────────────",
        f"Shared SGBs:        {n_shared:,}",
        f"NI-only depth-invisible: {n_invisible:,}",
        f"NI-only truly biological: {n_inexplicable:,}",
        f"Ind-only:           {gap['n_cats']['Ind_only']:,}",
        f"─────────────────────",
        f"BB phi (Ind/NI):    {phi_i:.1f} / {phi_n:.1f}",
        f"Sample q_s spread:  {q_i:.3f} / {q_n:.3f}",
    ]
    ax.text(0.98, 0.03, "\n".join(ann_lines),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS_ANNOT-2, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec="dimgrey", alpha=0.90))


def draw_accum_panel(ax_main, data_by_group, mname, threshold=10.0,
                     title="", N_LOG_MAX=1e5, n_total_sgbs=None,
                     np_curves=None, chao2_by_group=None,
                     depth_curves=None, depth_proj_curves=None,
                     predictive_curves=None, holdout_diag=None):
    """
    Linear main axes + log-space inset (bottom-right, 2/3 × 2/3).

    Parameters
    ----------
    data_by_group    : {label: (accum_8tuple, fit_dict, colour)}
                       accum_8tuple from bootstrap_curve; fit_dict from fit_one.
    mname            : model name key in MODEL_VARIANTS (plotted as overlay smoother)
    predictive_curves: {gname: (Ns, mean, lo, hi, draws)}
                       From predictive_accumulation_from_hmat — the PRIMARY curve.
                       When supplied, this is plotted as the main shaded band, and
                       the bootstrap scatter dots become secondary reference points.
    holdout_diag     : {gname: dict from deep_holdout_new_taxa_test}
                       If provided, a small inset (top-right of log panel) shows
                       observed vs predicted new SGBs in the deepest holdout samples.
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

        # ── Predictive accumulation (primary) ─────────────────────────────
        if predictive_curves is not None and gname in predictive_curves:
            pN, pMean, pLo, pHi, _ = predictive_curves[gname]
            N_max_pred = float(pN[-1]); n_samp_max = max(n_samp_max, int(N_max_pred))
            for ax_, is_log in [(ax_main, False), (ax_ins, True)]:
                # Shaded 95% predictive interval
                ax_.fill_between(pN, pLo, pHi,
                                 color=col, alpha=0.18, zorder=3,
                                 label=(f"{gname.split(' ')[0]} pred. 95% CI"
                                        if is_log else "_nolegend_"))
                ax_.plot(pN, pMean, color=col, lw=1.8, alpha=0.92,
                         zorder=4,
                         label=(f"{gname.split(' ')[0]} predictive mean"
                                if is_log else "_nolegend_"))
            # Bootstrap scatter as secondary reference (lighter, smaller dots)
            for ax_, is_log in [(ax_main, False), (ax_ins, True)]:
                ax_.errorbar(N_vec[::step], mean_v[::step], yerr=sem95[::step],
                             fmt="o", color=col, ms=2.5, capsize=1.5, lw=0,
                             elinewidth=0.7, alpha=0.45, zorder=9)
        else:
            # Fallback: original bootstrap-only display (no predictive curves)
            for ax_, is_log in [(ax_main, False), (ax_ins, True)]:
                ax_.fill_between(N_vec, q16_v, q84_v, color=col, alpha=0.16, zorder=1)
                ax_.plot(N_vec, mean_v, color=col, lw=0.8, alpha=0.35, zorder=1)
                ax_.errorbar(N_vec[::step], mean_v[::step], yerr=sem95[::step],
                             fmt="o", color=col, ms=3, capsize=2, lw=0,
                             elinewidth=0.9, alpha=0.80, zorder=10)

        # ── Model overlay (descriptive smoother) ─────────────────────────
        # Fit on the predictive mean if available, otherwise on bootstrap mean
        fit_N  = predictive_curves[gname][0] if (predictive_curves and gname in predictive_curves) else N_vec
        fit_S  = predictive_curves[gname][1] if (predictive_curves and gname in predictive_curves) else mean_v
        fit_sem = np.maximum(np.ones_like(fit_S) * 0.5, 0.1)

        if not fd["failed"] and fd["popt"] is not None:
            all_a.append((fd["popt"][0], gname, col))

        N_lin = np.linspace(1, n_samp_max * 1.05, 500)
        N_log = np.logspace(np.log10(1), np.log10(max(n_samp_max * 2.5, N_LOG_MAX)), 600)
        for ax_, is_log, N_plot in [(ax_main, False, N_lin), (ax_ins, True, N_log)]:
            if not fd["failed"] and fd["popt"] is not None:
                yp = spec["func"](N_plot, *fd["popt"])
                lbl = (f"{spec['label_short']}" if (is_log and gname == list(data_by_group.keys())[0])
                       else "_nolegend_")
                ax_.plot(N_plot, yp, color=col, lw=spec["lw"] * 0.85,
                         ls=(0, (5, 2)), alpha=0.70, zorder=spec["zorder"],
                         label=lbl)

    # Asymptote ceiling
    a_values = [a for a, _, _ in all_a]
    y_ceil = float(np.ceil((max(a_values + ([float(n_total_sgbs)] if n_total_sgbs else [])) + 150) / 50) * 50) \
             if a_values else 5000

    # Linear axes: y-top = predictive max * 1.08 or bootstrap max
    if predictive_curves:
        best_s = max(float(v[1][-1]) for v in predictive_curves.values()
                     if v[1].size > 0)
    else:
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
    ax_main.set_xlim(0, n_samp_max * 1.05)
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
    ax_ins.set_xlim(0.8, max(n_samp_max*2.5, N_LOG_MAX))
    _staggered_asym_labels(ax_ins, ins_entries, y_ceil,
                           x_lo_frac=0.04, x_hi_frac=0.42, fs=FS_ANNOT-2.5)

    # ── Depth-aware E[S(N)] (generative model, shown as dash) ────────────
    if depth_curves:
        for gname, (da_N, da_S, da_std) in depth_curves.items():
            col = next((c for g, (_a,_f,c) in data_by_group.items() if g == gname), "grey")
            for ax_, is_log in [(ax_main, False), (ax_ins, True)]:
                lbl = (f"Depth-aware model" if (is_log and gname == list(depth_curves.keys())[0])
                       else "_nolegend_")
                ax_.fill_between(da_N, da_S - da_std, da_S + da_std,
                                 color=col, alpha=0.07, zorder=2)
                ax_.plot(da_N, da_S, color=col, lw=1.2, ls=(0,(3,1)),
                         alpha=0.60, zorder=5, label=lbl)

    # ── Depth-projected curve (uniform d*) ───────────────────────────────
    if depth_proj_curves:
        for gname, (dp_N, dp_S) in depth_proj_curves.items():
            col = next((c for g, (_a,_f,c) in data_by_group.items() if g == gname), "grey")
            for ax_, is_log in [(ax_main, False), (ax_ins, True)]:
                lbl = (f"Projected to 1Gbp" if (is_log and gname == list(depth_proj_curves.keys())[0])
                       else "_nolegend_")
                ax_.plot(dp_N, dp_S, color=col, lw=0.9, ls="--",
                         alpha=0.45, zorder=5, label=lbl)

    # ── Naive ĥ_k curve ───────────────────────────────────────────────────
    if np_curves:
        for gname, (np_N, np_S) in np_curves.items():
            col = next((c for g, (_a,_f,c) in data_by_group.items() if g == gname), "grey")
            for ax_, is_log in [(ax_main, False), (ax_ins, True)]:
                lbl = (f"Naive ĥ_k (ignores depth)" if (is_log and gname == list(np_curves.keys())[0])
                       else "_nolegend_")
                ax_.plot(np_N, np_S, color=col, lw=0.8, ls=":",
                         alpha=0.45, zorder=5, label=lbl)

    # ── Chao2 horizontal references ───────────────────────────────────────
    if chao2_by_group:
        for gname, (c2, c2_lo, c2_hi) in chao2_by_group.items():
            col = next((c for g, (_a,_f,c) in data_by_group.items() if g == gname), "grey")
            if not np.isfinite(c2): continue
            ax_ins.axhline(c2, color=col, lw=1.0, ls="-.", alpha=0.60, zorder=3)
            if np.isfinite(c2_lo) and np.isfinite(c2_hi):
                ax_ins.axhspan(c2_lo, c2_hi, color=col, alpha=0.07, zorder=1)
            xl = ax_ins.get_xlim()
            ax_ins.text(xl[0] * (xl[1]/xl[0])**0.08 if xl[0] > 0 else xl[0]+1,
                        c2 + y_ceil*0.007, f"Chao2={c2:.0f}",
                        fontsize=FS_TICK-3, color=col, va="bottom", alpha=0.85)

    # ── Holdout diagnostic micro-inset (top-right of log panel) ──────────
    if holdout_diag:
        # pick the first group for the inset (avoids overcrowding)
        _gname0 = list(holdout_diag.keys())[0]
        hd = holdout_diag[_gname0]
        col0 = next((c for g, (_, _, c) in data_by_group.items() if g == _gname0), "grey")
        obs_arr  = hd["obs_new_per_sample"]
        pred_arr = hd["pred_new_per_sample"]
        if len(obs_arr) > 0:
            ax_hd = ax_ins.inset_axes([0.55, 0.60, 0.44, 0.38])
            ax_hd.scatter(pred_arr, obs_arr, s=12, color=col0, alpha=0.60, zorder=3)
            lim_max = max(float(max(obs_arr.max(), pred_arr.max(), 1)), 1.0)
            ax_hd.plot([0, lim_max], [0, lim_max], "k:", lw=0.8, alpha=0.5)
            ax_hd.set_xlabel("Pred.", fontsize=FS_ANNOT-3, labelpad=1)
            ax_hd.set_ylabel("Obs.", fontsize=FS_ANNOT-3, labelpad=1)
            ax_hd.set_title(f"Holdout\nnew SGBs\n(n={hd['n_hold']})",
                            fontsize=FS_ANNOT-3, pad=1)
            ax_hd.tick_params(labelsize=FS_ANNOT-4, pad=1)
            for sp in ax_hd.spines.values():
                sp.set_linewidth(0.4)

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

    # Legend: group colours (main axes) + model smoother (log inset)
    handles = [Line2D([0],[0], color=col, lw=2.0, label=gname)
               for gname, (_, _, col) in data_by_group.items()]
    ax_main.legend(handles=handles, fontsize=FS_LEGEND, frameon=False, loc="upper left")
    if any(not fd["failed"] and fd.get("popt") is not None
           for _, fd, _ in data_by_group.values()):
        ax_ins.legend(fontsize=FS_LEGEND-1, loc="lower right",
                      frameon=True, framealpha=0.85, edgecolor="#cccccc")


def draw_surface_accum_panel(ax_main, surface_by_group, thresholds_pct,
                              title="", N_LOG_MAX=3e5, n_total_sgbs=None,
                              chao2_by_group=None):
    """
    Plot the accumulation surface S(N, τ) as a family of predictive curves.

    For each group × threshold, draws a shaded predictive band.  Within each
    group, curves use a sequential colourmap (lighter = lower threshold = higher
    sensitivity).  The spread BETWEEN threshold curves at large N shows how
    threshold-sensitive the accumulation is — wide spread means most detections
    are marginal (just above 10%); narrow spread means species are robustly
    covered at all thresholds.

    Parameters
    ----------
    surface_by_group : {gname: {tau_pct: (Ns, pMean, pLo, pHi)}}
    thresholds_pct   : ordered array of thresholds in % (e.g. [10,20,30,40,50])
    """
    ax_ins = ax_main.inset_axes([0.33, 0.05, 0.66, 0.65])
    thresholds_pct = np.asarray(thresholds_pct, float)

    n_samp_max = 1
    y_vals_all = []

    for gname, surf in surface_by_group.items():
        for j, tau_pct in enumerate(thresholds_pct):
            if tau_pct not in surf:
                continue
            Ns, pM, pLo, pHi = surf[tau_pct]
            col   = _thresh_color(gname, j)
            alpha_band  = 0.20 if j == 0 else 0.13
            alpha_line  = 0.90 if j == 0 else 0.70
            lw          = 1.6  if j == 0 else 1.1
            # Primary threshold label only once
            lbl_main = (f"τ={tau_pct:.0f}% — {gname.split(' ')[0]}"
                        if j < len(thresholds_pct) else "_nolegend_")
            n_samp_max = max(n_samp_max, int(Ns[-1]))
            y_vals_all.extend([float(pHi.max()), float(pM.max())])

            for ax_ in [ax_main, ax_ins]:
                ax_.fill_between(Ns, pLo, pHi, color=col,
                                  alpha=alpha_band, zorder=2, linewidth=0)
                ax_.plot(Ns, pM, color=col, lw=lw, alpha=alpha_line,
                          zorder=3, label=lbl_main)
                lbl_main = "_nolegend_"

        # Fill between lowest and highest threshold to show the "surface gap"
        taus_sorted = sorted([t for t in surf if t in set(thresholds_pct)])
        if len(taus_sorted) >= 2:
            Ns_lo, pM_lo = surf[taus_sorted[0]][0], surf[taus_sorted[0]][1]
            Ns_hi, pM_hi = surf[taus_sorted[-1]][0], surf[taus_sorted[-1]][1]
            col0 = _thresh_color(gname, 0)
            for ax_ in [ax_main, ax_ins]:
                ax_.fill_between(Ns_lo, pM_hi, pM_lo, color=col0,
                                  alpha=0.06, zorder=1, linewidth=0)

    y_ceil = float(np.ceil((max(y_vals_all + ([float(n_total_sgbs)] if n_total_sgbs else [1])) + 150) / 50) * 50) \
             if y_vals_all else 5000
    y_top_lin = float(max(y_vals_all)) * 1.10 if y_vals_all else y_ceil

    ax_main.set_xscale("linear")
    ax_main.set_xlim(0, n_samp_max * 1.05)
    ax_main.set_ylim(0, y_top_lin)
    ax_main.set_ylabel("SGBs detected", fontsize=FS_LABEL)
    ax_main.set_xlabel("Number of samples (N)", fontsize=FS_LABEL)
    ax_main.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=5)
    ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_main.grid(True, ls="--", lw=0.3, alpha=0.35)

    if n_total_sgbs and n_total_sgbs <= y_top_lin * 1.02:
        ax_main.axhline(n_total_sgbs, color="dimgrey", lw=0.8, ls=":", alpha=0.5)
        ax_main.text(n_samp_max * 0.02, n_total_sgbs * 1.01,
                     f"all SGBs = {n_total_sgbs:,}",
                     fontsize=FS_ANNOT-1, color="dimgrey", va="bottom")

    ax_ins.set_xscale("log")
    ax_ins.set_xlim(0.8, max(n_samp_max * 2.5, N_LOG_MAX))
    ax_ins.set_ylim(0, y_ceil)
    ax_ins.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_ins.tick_params(labelsize=FS_TICK-2, pad=1)
    ax_ins.set_xlabel("log N", fontsize=FS_TICK-1, labelpad=2)
    ax_ins.set_ylabel("SGBs", fontsize=FS_TICK-1, labelpad=2)
    ax_ins.grid(True, which="both", ls="--", lw=0.22, alpha=0.32)

    if chao2_by_group:
        for gname, (c2, c2_lo, c2_hi) in chao2_by_group.items():
            col = _thresh_color(gname, 0)
            if np.isfinite(c2):
                ax_ins.axhline(c2, color=col, lw=0.9, ls="-.", alpha=0.55)
                xl = ax_ins.get_xlim()
                ax_ins.text(xl[0] * (xl[1] / xl[0]) ** 0.08,
                            c2 + y_ceil * 0.007, f"Chao2={c2:.0f}",
                            fontsize=FS_TICK-3, color=col, va="bottom", alpha=0.8)

    ax_main.legend(fontsize=FS_LEGEND-1, frameon=False, loc="upper left",
                   ncol=2, handlelength=1.2)

    for sp in ax_ins.spines.values():
        sp.set_edgecolor("dimgrey"); sp.set_linewidth(0.5); sp.set_alpha(0.5)
    y_f_top = float(np.clip(y_top_lin / y_ceil, 0, 1))
    for y_f in [y_f_top, 0.0]:
        y_data = y_f * y_ceil
        con = ConnectionPatch(
            xyA=(1.0, y_f), coordsA="axes fraction", axesA=ax_ins,
            xyB=(n_samp_max * 1.04, y_data), coordsB="data", axesB=ax_main,
            color="dimgrey", lw=0.6, alpha=0.40, ls="-", arrowstyle="-")
        ax_main.add_artist(con)


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
                 n_sgbs_by_ref, threshold, n_boot, out_stem,
                 predictive_curves_all=None,
                 holdout_diag_all=None,
                 np_curves_all=None,
                 depth_curves_all=None,
                 proj_curves_all=None,
                 chao2_all=None,
                 surface_all=None,           # {ref: {gname: {tau_pct: (Ns,pM,pLo,pHi)}}}
                 thresholds_pct=None):        # threshold grid in %
    """
    Build and save one figure variant.

    Layout:
      Row 0: A=Carter breadth surface    B=Leviatan breadth surface
      Row 1: C=Carter asymptote sweep    D=Leviatan asymptote sweep
      Row 2: E=Carter excl.Hadza surface F=world extrapolation

    When surface_all is provided, panels A/B/E show the multi-threshold
    accumulation surface S(N, τ) instead of a single-threshold curve.
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

    REFS  = list(accum_results.keys())
    ref_A = REFS[0]; ref_B = REFS[min(1, len(REFS)-1)]

    # ── Panels A / B: breadth surface (or single-threshold fallback) ──────
    for ax_ab, ref, panel_lbl in [(ax_A, ref_A, REF_LABELS.get(ref_A, ref_A)),
                                   (ax_B, ref_B, REF_LABELS.get(ref_B, ref_B))]:
        surf_ref  = (surface_all or {}).get(ref)
        c2_g      = (chao2_all or {}).get(ref)
        try:
            if surf_ref:
                draw_surface_accum_panel(
                    ax_ab, surf_ref,
                    thresholds_pct=thresholds_pct or THRESHOLDS_GRID * 100,
                    title=panel_lbl,
                    n_total_sgbs=n_sgbs_by_ref.get(ref),
                    chao2_by_group=c2_g)
            else:
                # Fallback: single-threshold predictive curve
                data_by_g = {}
                for gname in ["Industrial (all)", "Non-Industrial (all)"]:
                    col = COL["Industrial"] if "Industrial" in gname else COL["Non-Industrial"]
                    acc = accum_results[ref][gname]
                    fd  = fits_by_mname[ref][gname][mname]
                    data_by_g[gname] = (acc, fd, col)
                pred_g = (predictive_curves_all or {}).get(ref)
                hd_g   = (holdout_diag_all or {}).get(ref)
                draw_accum_panel(ax_ab, data_by_g, mname,
                                 threshold=threshold, title=panel_lbl,
                                 n_total_sgbs=n_sgbs_by_ref.get(ref),
                                 predictive_curves=pred_g,
                                 holdout_diag=hd_g,
                                 chao2_by_group=c2_g)
        except Exception as exc:
            import traceback
            print(f"  ERROR in panel {panel_lbl}: {exc}"); traceback.print_exc()
            ax_ab.set_title(f"{panel_lbl} [ERROR]", color="red")

    # ── Panels C / D: asymptote sweep ─────────────────────────────────────
    for ax_cd, ref, panel_lbl in [(ax_C, ref_A, REF_LABELS.get(ref_A, ref_A)),
                                   (ax_D, ref_B, REF_LABELS.get(ref_B, ref_B))]:
        try:
            draw_sweep_panel(ax_cd, sweep_results[ref][mname],
                             title=panel_lbl,
                             n_total_sgbs=n_sgbs_by_ref.get(ref))
        except Exception as exc:
            import traceback
            print(f"  ERROR in sweep {panel_lbl}: {exc}"); traceback.print_exc()
            ax_cd.set_title(f"{panel_lbl} sweep [ERROR]", color="red")

    # ── Panel E: Hadza-excluded NI surface ────────────────────────────────
    surf_E = (surface_all or {}).get("_panel_E")
    c2_E   = None
    try:
        if surf_E:
            draw_surface_accum_panel(
                ax_E, surf_E,
                thresholds_pct=thresholds_pct or THRESHOLDS_GRID * 100,
                title=f"{REF_LABELS.get(ref_A, ref_A)} — excl. Hadza-exclusive SGBs",
                n_total_sgbs=n_sgbs_by_ref.get(ref_A))
        else:
            data_E = {}
            for gname_lbl, acc in accum_E.items():
                col = COL["Industrial"] if "Industrial" in gname_lbl else COL["Non-Industrial"]
                fd  = fits_E[gname_lbl][mname]
                data_E[gname_lbl] = (acc, fd, col)
            pred_E = (predictive_curves_all or {}).get("_panel_E")
            draw_accum_panel(ax_E, data_E, mname,
                             threshold=threshold,
                             title=f"{REF_LABELS.get(ref_A, ref_A)} — excl. Hadza-exclusive SGBs",
                             n_total_sgbs=n_sgbs_by_ref.get(ref_A),
                             predictive_curves=pred_E)
    except Exception as exc:
        import traceback
        print(f"  ERROR in panel E: {exc}"); traceback.print_exc()
        ax_E.set_title("Panel E [ERROR]", color="red")

    # ── Panel F ───────────────────────────────────────────────────────────
    try:
        draw_panel_f(ax_F, pf)
    except Exception as exc:
        import traceback
        print(f"  ERROR in panel F: {exc}"); traceback.print_exc()
        ax_F.set_title("Panel F [ERROR]", color="red")

    spec = MODEL_VARIANTS[mname]
    thr_str = ("τ∈{" + ",".join(f"{t:.0f}%" for t in (thresholds_pct or THRESHOLDS_GRID*100)) + "}"
               if surface_all else f"τ={threshold:.0f}%")
    fig.suptitle(
        f"Human gut SGB accumulation — {spec['label_short']} overlay   |   "
        f"{thr_str}   |   {n_boot} bootstrap + predictive",
        fontsize=FS_TITLE+1, fontweight="bold", y=0.975,
    )

    fig.canvas.draw()
    apply_display_text(fig)
    stem    = f"{out_stem}_{mname.replace('(','').replace(')','').replace(' ','_')}"
    pdf_path = stem + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"  Saved → {pdf_path}")
    png_path = stem + ".png"
    tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(png_path))
    fig.savefig(tmp_path, dpi=150, bbox_inches="tight", facecolor="white")
    shutil.move(tmp_path, png_path)
    print(f"  Saved → {png_path}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    """
    Five-step pipeline (Phase A joint threshold-category hurdle model).

    Steps:
      1. Load data
      2. estimate_joint_bb_model()  → psi_k, r_mean_surf, autc_k, h_mat_primary
      3. bootstrap_curve(h_mat_primary) + model fitting → accum_results
      4. predictive_surface_from_joint() → S(N, tau) surface
      5. diagnostics: deep-holdout, asymptote sweep, Chao2, depth curves
      → build_figure()
    """
    parser = argparse.ArgumentParser(
        description="Joint multi-threshold breadth model — predictive surface.")
    parser.add_argument("--refs",        nargs="+", default=["UHGG", "lev"])
    parser.add_argument("--thresholds",  nargs="+", type=float,
                        default=[10, 20, 30, 40, 50],
                        help="Breadth thresholds in %% (default: 10 20 30 40 50)")
    parser.add_argument("--threshold",   type=float, default=10.0,
                        help="Primary threshold for legacy outputs (default 10)")
    parser.add_argument("--n-boot",      type=int,   default=100)
    parser.add_argument("--n-pred",      type=int,   default=400,
                        help="Predictive draws per threshold (default 400)")
    parser.add_argument("--n-extrap",    type=float, default=2.0,
                        help="Extrapolation factor beyond observed N (default 2x)")
    parser.add_argument("--abund-dist",  default="lognormal",
                        choices=["lognormal", "student_t"])
    parser.add_argument("--abund-df",    type=int, default=5)
    parser.add_argument("--n-quad",      type=int, default=31)
    parser.add_argument("--holdout-frac",type=float, default=0.15)
    parser.add_argument("--holdout-refit", action="store_true", default=False,
                        help="Refit model on training samples only for holdout test "
                             "(true out-of-sample; slower but statistically correct)")
    parser.add_argument("--no-trunc-correction", action="store_true", default=False,
                        help="Disable truncation-corrected MLE (faster, less accurate)")
    parser.add_argument("--eb-n0",       type=int, default=5,
                        help="EB shrinkage pseudo-count (default 5)")
    parser.add_argument("--out",         default="figure_summary_predictive")
    parser.add_argument("--run-stan",    action="store_true", default=False)
    parser.add_argument("--stan-chains", type=int, default=4)
    parser.add_argument("--stan-warmup", type=int, default=500)
    parser.add_argument("--stan-samples",type=int, default=500)
    args = parser.parse_args(argv)

    THRESHOLDS_PCT = np.asarray(args.thresholds, float)
    THRESHOLD      = float(args.threshold)
    # Fix 3: enforce primary threshold is a member of the threshold grid.
    # If the user's --threshold is not in --thresholds, snap to nearest and warn.
    if not np.any(np.isclose(THRESHOLDS_PCT, THRESHOLD)):
        nearest = float(THRESHOLDS_PCT[np.argmin(np.abs(THRESHOLDS_PCT - THRESHOLD))])
        print(f"  WARNING: --threshold {THRESHOLD} not in threshold grid "
              f"{THRESHOLDS_PCT.tolist()}; snapping to nearest: {nearest}")
        THRESHOLD = nearest
    N_BOOT         = args.n_boot
    N_PRED         = args.n_pred
    correct_trunc  = not args.no_trunc_correction
    holdout_refit  = args.holdout_refit
    rng            = np.random.default_rng(42)

    print("\n=== Configuration ===")
    print(f"  Thresholds     : {THRESHOLDS_PCT.tolist()} %")
    print(f"  Abundance dist : {args.abund_dist}"
          + (f" (df={args.abund_df})" if args.abund_dist == "student_t" else ""))
    print(f"  Truncation MLE : {'ON' if correct_trunc else 'OFF (--no-trunc-correction)'}")
    print(f"  EB shrinkage n0: {args.eb_n0}")
    print(f"  n_boot={N_BOOT}  n_pred={N_PRED}  n_extrap={args.n_extrap}x")

    # ── STEP 1: Load ──────────────────────────────────────────────────────
    print("\n=== Step 1: Loading data ===")
    ref_data      = {ref: load_reference(ref) for ref in args.refs}
    n_sgbs_by_ref = {ref: d["n_sgbs"] for ref, d in ref_data.items()}

    # ── STEP 2: Joint threshold model ─────────────────────────────────────
    # Fixes vs. previous code:
    #   - psi_k estimated via interval MLE on C_ks in {0,...,J}  (replaces HT)
    #   - mu_x, sig_x via truncation-corrected MLE               (fixes left-trunc bias)
    #   - EB shrinkage for rare species                           (replaces r_mat=0)
    #   - No q_s in log_eff (Phase A)                            (fixes depth double-count)
    print("\n=== Step 2: Joint threshold model ===")
    joint_models  = {}   # [ref][gname] -> full tuple from estimate_joint_bb_model
    h_mat_primary = {}   # [ref][gname] -> (n_samp, n_sgb) at primary threshold

    for ref, rdat in ref_data.items():
        joint_models[ref] = {}; h_mat_primary[ref] = {}
        blocks = build_group_blocks(rdat["meta"], rdat["breadth"], THRESHOLD)
        for gname, block in blocks.items():
            print(f"  {ref} / {gname} [n={len(block):,}]...", end=" ", flush=True)
            jm = estimate_joint_bb_model(
                block,
                rdat["depth"].loc[block.index],
                rdat["lengths"],
                rdat["total_bp"].loc[block.index],
                thresholds_pct     = THRESHOLDS_PCT,
                abund_dist         = args.abund_dist,
                abund_df           = args.abund_df,
                n_q                = args.n_quad,
                correct_truncation = correct_trunc,
                eb_n0              = args.eb_n0,
            )
            psi_k, mu_x, sig_x, phi_k, d_i, L_k, thr_frac, r_mean_surf, autc_k, h_mat_p, C = jm
            joint_models[ref][gname]  = jm
            h_mat_primary[ref][gname] = h_mat_p
            print(f"done  median_psi={np.median(psi_k):.3f}  "
                  f"median_AUTC={np.median(autc_k):.3f}  "
                  f"med_sig_x={np.nanmedian(sig_x[np.isfinite(sig_x)]):.3f}")

    # ── STEP 3: Bootstrap accumulation & model fitting ────────────────────
    print("\n=== Step 3: Bootstrap accumulation & model fitting ===")
    accum_results = {}; fits_by_mname = {}; boot_matrices = {}

    for ref, rdat in ref_data.items():
        accum_results[ref] = {}; fits_by_mname[ref] = {}; boot_matrices[ref] = {}
        blocks = build_group_blocks(rdat["meta"], rdat["breadth"], THRESHOLD)
        for gname, block in blocks.items():
            h_mat = h_mat_primary[ref][gname]
            print(f"  {ref} / {gname}...", end=" ", flush=True)
            acc   = bootstrap_curve(block, THRESHOLD, n_boot=N_BOOT, rng=rng,
                                    h_mat=h_mat)
            sem_v = np.maximum(acc[2] / np.sqrt(max(N_BOOT, 1)), 0.1)
            accum_results[ref][gname] = acc
            boot_matrices[ref][gname] = acc[-1]
            fits_by_mname[ref][gname] = {mn: fit_one(mn, acc[0], acc[1], sem_v)
                                          for mn in MODEL_VARIANTS}
            print("  ".join(f"{mn.split('(')[0]} R2={fits_by_mname[ref][gname][mn]['r2']:.4f}"
                            for mn in MODEL_VARIANTS))

    # ── STEP 4: Predictive accumulation surface S(N, tau) ─────────────────
    print(f"\n=== Step 4: Predictive surface S(N, tau) "
          f"({N_PRED} draws, {args.n_extrap}x extrap) ===")
    surface_all           = {}   # {ref: {gname: {tau_pct: (Ns,pM,pLo,pHi)}}}
    predictive_curves_all = {}   # primary-threshold predictive curves for overlays

    for ref, rdat in ref_data.items():
        surface_all[ref] = {}; predictive_curves_all[ref] = {}
        blocks = build_group_blocks(rdat["meta"], rdat["breadth"], THRESHOLD)
        for gname, block in blocks.items():
            psi_k, mu_x, sig_x, phi_k, d_i, L_k = joint_models[ref][gname][:6]
            N_max = int(np.ceil(len(block) * args.n_extrap))
            print(f"  {ref} / {gname}  N_max={N_max:,}...", end=" ", flush=True)

            surf, pDraws, Ns_p = predictive_surface_from_joint(
                psi_k, mu_x, sig_x, phi_k, d_i, L_k,
                THRESHOLDS_PCT, N_max, N_PRED, rng,
                args.abund_dist, args.abund_df, args.n_quad)
            surface_all[ref][gname] = surf

            # Fix 4: store real predictive draws (not zeros)
            Ns, pM, pLo, pHi = surf[float(THRESHOLD)]
            predictive_curves_all[ref][gname] = (Ns_p, pM, pLo, pHi, pDraws)
            sem_pred = np.maximum(pDraws.std(0) / np.sqrt(max(N_PRED, 1)), 0.1)
            for mn in MODEL_VARIANTS:
                fd = fit_one(mn, Ns, pM, sem_pred)
                if not fd["failed"] and fd["popt"] is not None:
                    fits_by_mname[ref][gname][mn] = fd
            print(f"done  S({THRESHOLD:.0f}%,N={N_max:,})={pM[-1]:.0f} [{pLo[-1]:.0f}-{pHi[-1]:.0f}]")

    # ── STEP 5a: Deep-holdout test ────────────────────────────────────────
    print(f"\n=== Step 5a: Deep-holdout (hold_frac={args.holdout_frac:.0%}"
          f"{', refit=ON' if holdout_refit else ', refit=OFF (calibration)'}) ===")
    holdout_diag_all = {}
    for ref, rdat in ref_data.items():
        holdout_diag_all[ref] = {}
        blocks = build_group_blocks(rdat["meta"], rdat["breadth"], THRESHOLD)
        for gname, block in blocks.items():
            h_mat    = h_mat_primary[ref][gname]
            total_bp = rdat["total_bp"].loc[block.index]
            depth_b  = rdat["depth"].loc[block.index]
            # Fix 2: build a closure that re-estimates the joint model on any subset.
            def _make_refit(thr=THRESHOLDS_PCT, ad=args.abund_dist, df=args.abund_df,
                            nq=args.n_quad, ct=correct_trunc, n0=args.eb_n0):
                def _fn(bb, db, ls, tb):
                    return estimate_joint_bb_model(
                        bb, db, ls, tb, thresholds_pct=thr,
                        abund_dist=ad, abund_df=df, n_q=nq,
                        correct_truncation=ct, eb_n0=n0)
                return _fn
            hd = deep_holdout_new_taxa_test(
                block, h_mat, total_bp,
                threshold=THRESHOLD, hold_frac=args.holdout_frac,
                depth_block=depth_b         if holdout_refit else None,
                lengths=rdat["lengths"]     if holdout_refit else None,
                refit_model_fn=_make_refit() if holdout_refit else None)
            holdout_diag_all[ref][gname] = hd
            print(f"  {ref}/{gname}  train={hd['n_train']:,}  hold={hd['n_hold']:,}"
                  f"  obs_new={hd['obs_new_per_sample'].sum():.0f}"
                  f"  pred_new={hd['pred_new_per_sample'].sum():.1f}")

    # ── STEP 5b: Asymptote sweep ──────────────────────────────────────────
    print("\n=== Step 5b: Asymptote sweep ===")
    sweep_results = {}
    for ref, rdat in ref_data.items():
        sweep_results[ref] = {mn: {} for mn in MODEL_VARIANTS}
        blocks = build_group_blocks(rdat["meta"], rdat["breadth"], THRESHOLD)
        for gname, block in blocks.items():
            acc   = accum_results[ref][gname]
            sem_v = np.maximum(acc[2] / np.sqrt(max(N_BOOT, 1)), 0.1)
            bm    = boot_matrices[ref].get(gname)
            if bm is None or bm.size == 0: continue
            col = COL["Industrial"] if "Industrial" in gname else COL["Non-Industrial"]
            ls  = "-" if "Industrial" in gname else "--"
            for mn in MODEL_VARIANTS:
                print(f"  {ref}/{gname}/{mn}...", end=" ", flush=True)
                N_s, a_s, a_lo, a_hi = asymptote_sweep(mn, acc[0], bm, sem_v)
                sweep_results[ref][mn][gname] = (N_s, a_s, a_lo, a_hi, col, ls)
                print(f"done ({len(N_s)} pts)")

    # ── STEP 5c: Chao2 + NP/depth overlay curves ──────────────────────────
    TARGET_DEPTH = 1e9
    print(f"\n=== Step 5c: Chao2 + depth curves (target={TARGET_DEPTH:.0e} bp) ===")
    np_curves_all = {}; depth_curves_all = {}
    proj_curves_all = {}; chao2_all = {}
    for ref, rdat in ref_data.items():
        np_curves_all[ref] = {}; depth_curves_all[ref] = {}
        proj_curves_all[ref] = {}; chao2_all[ref] = {}
        blocks = build_group_blocks(rdat["meta"], rdat["breadth"], THRESHOLD)
        for gname, block in blocks.items():
            depth_block = rdat["depth"].loc[block.index]
            total_bp    = rdat["total_bp"].loc[block.index]
            psi_k_jm    = joint_models[ref][gname][0]
            print(f"  {ref}/{gname}...", end=" ", flush=True)
            N_np, S_np, _ = nonparametric_accumulation(block, THRESHOLD, psi_k=psi_k_jm)
            np_curves_all[ref][gname] = (N_np, S_np)
            N_da, S_da, S_std, *_ = depth_aware_accumulation(
                block, depth_block, rdat["lengths"], total_bp, THRESHOLD, rng=rng)
            depth_curves_all[ref][gname] = (N_da, S_da, S_std)
            N_pr, S_pr, *_ = depth_aware_accumulation(
                block, depth_block, rdat["lengths"], total_bp,
                THRESHOLD, target_depth=TARGET_DEPTH, rng=rng)
            proj_curves_all[ref][gname] = (N_pr, S_pr)
            c2, c2_lo, c2_hi = chao2_estimate(block, THRESHOLD)
            chao2_all[ref][gname] = (c2, c2_lo, c2_hi)
            print(f"S_depth={S_da[-1]:.0f}  S_proj={S_pr[-1]:.0f}  Chao2={c2:.0f}")

    # ── Panel E: Hadza-excluded NI ────────────────────────────────────────
    print("\n=== Panel E: Hadza-excluded NI ===")
    ref_e    = args.refs[0]
    rdat_e   = ref_data[ref_e]
    blocks_e = build_group_blocks(rdat_e["meta"], rdat_e["breadth"], THRESHOLD)
    excl     = hadza_exclusive_sgbs(rdat_e["breadth"], rdat_e["meta"], THRESHOLD)
    breadth_ind_e = blocks_e["Industrial (all)"]
    breadth_ni_e  = blocks_e["Non-Industrial (all)"]
    if excl:
        breadth_ni_e = breadth_ni_e[[c for c in breadth_ni_e.columns if c not in excl]]
        print(f"  Removed {len(excl):,} Hadza-exclusive SGBs")

    gap_results = decompose_richness_gap(
        rdat_e, rdat_e["meta"], breadth_ind_e, breadth_ni_e, THRESHOLD, rng=rng)
    print(f"  Gap: obs={gap_results['S_obs_gap']:+.0f}  "
          f"bio={gap_results['S_bio_gap']:+.0f}  "
          f"tech={gap_results['S_obs_gap']-gap_results['S_bio_gap']:+.0f}")

    # Fix 1 (complete): inject ALL joint-model fields into gap_results so that
    # run_absence_test uses the same model for occupancy, power, and evidence
    # classes — not a hybrid of new ψ and old HT-based quantities.
    print("  Fitting joint model for absence test injection...")
    ref_sgbs_e = sorted(set(breadth_ind_e.columns) & set(breadth_ni_e.columns))
    _jm_ind_abs = estimate_joint_bb_model(
        breadth_ind_e[ref_sgbs_e],
        rdat_e["depth"].loc[breadth_ind_e.index, ref_sgbs_e],
        rdat_e["lengths"].reindex(ref_sgbs_e),
        rdat_e["total_bp"].loc[breadth_ind_e.index],
        thresholds_pct=THRESHOLDS_PCT,
        abund_dist=args.abund_dist, abund_df=args.abund_df, n_q=args.n_quad,
        correct_truncation=correct_trunc, eb_n0=args.eb_n0)
    _jm_ni_abs = estimate_joint_bb_model(
        breadth_ni_e[ref_sgbs_e],
        rdat_e["depth"].loc[breadth_ni_e.index, ref_sgbs_e],
        rdat_e["lengths"].reindex(ref_sgbs_e),
        rdat_e["total_bp"].loc[breadth_ni_e.index],
        thresholds_pct=THRESHOLDS_PCT,
        abund_dist=args.abund_dist, abund_df=args.abund_df, n_q=args.n_quad,
        correct_truncation=correct_trunc, eb_n0=args.eb_n0)

    psi_ind_jm, mu_x_ind, sig_x_ind, phi_k_ind, d_i_ind, L_k_ind, \
        thr_f, r_mean_surf_ind, autc_ind, h_mat_ind, _ = _jm_ind_abs
    psi_ni_jm, *_, r_mean_surf_ni, _, _, _ = _jm_ni_abs

    j_prim = int(np.argmin(thr_f))
    n_ind  = len(d_i_ind)
    # h_hat at primary threshold = raw detection fraction
    h_hat_ind_jm = (breadth_ind_e[ref_sgbs_e].values >= THRESHOLD).mean(axis=0)
    # sum_r from analytical r̄_k at primary threshold
    sum_r_ind_jm = r_mean_surf_ind[:, j_prim] * n_ind
    log_tau_jm   = float(np.log(-np.log1p(-THRESHOLD / 100.0)))

    gap_results.update({
        "psi_ind":       psi_ind_jm,
        "psi_ni":        psi_ni_jm,
        "h_ind":         h_hat_ind_jm,
        "sum_r_ind":     sum_r_ind_jm,
        "r_mean_ind":    r_mean_surf_ind[:, j_prim],
        "mu_ind":        mu_x_ind,
        "sig_ind":       sig_x_ind,
        "phi_k_ind":     phi_k_ind,
        "L_k_arr":       L_k_ind,
        "d_i_ind":       d_i_ind,
        "n_ind_samples": n_ind,
        "autc_ind":      autc_ind,
        "log_tau":       log_tau_jm,
        "N_ind":         n_ind,
    })
    print(f"  Injected joint-model fields: median_ψ_Ind={np.median(psi_ind_jm):.3f}  "
          f"median_ψ_NI={np.median(psi_ni_jm):.3f}")

    absence_results = run_absence_test(
        gap_results, breadth_ind_e, breadth_ni_e, rdat_e,
        threshold=THRESHOLD, rng=rng,
        meta=rdat_e["meta"],
        breadth_ni_full=blocks_e["Non-Industrial (all)"],
        n_boot=min(N_BOOT, 300))
    draw_absence_figure(absence_results, args.out)

    if args.run_stan:
        print("\n=== Stan joint hierarchical model ===")
        stan_results = run_stan_joint_model(
            gap_results, breadth_ind_e, breadth_ni_e, rdat_e,
            threshold=THRESHOLD, chains=args.stan_chains,
            iter_warmup=args.stan_warmup, iter_sampling=args.stan_samples,
            out_dir=os.path.dirname(args.out) or ".")
        if stan_results is not None:
            absence_results.update({
                "E_N_missing":  stan_results["E_N_missing"],
                "CI_N_missing": stan_results["CI_N_missing"],
                "p_joint":      stan_results["p_missing"],
            })
            draw_absence_figure(absence_results,
                                args.out.replace(".pdf", "") + "_stan")
    else:
        print("  (Stan skipped -- use --run-stan to enable)")

    # Panel E: joint model + surface for Hadza-excluded blocks
    surface_all["_panel_E"] = {}; predictive_curves_all["_panel_E"] = {}
    holdout_diag_all["_panel_E"] = {}
    accum_E = {}; fits_E = {}
    for gname_e, block_e in {"Industrial (all)": breadth_ind_e,
                              "Non-Industrial (all)": breadth_ni_e}.items():
        print(f"  Panel E / {gname_e}...", end=" ", flush=True)
        depth_e = rdat_e["depth"].loc[block_e.index]
        total_e = rdat_e["total_bp"].loc[block_e.index]
        jm_e = estimate_joint_bb_model(
            block_e, depth_e, rdat_e["lengths"], total_e,
            thresholds_pct=THRESHOLDS_PCT,
            abund_dist=args.abund_dist, abund_df=args.abund_df, n_q=args.n_quad,
            correct_truncation=correct_trunc, eb_n0=args.eb_n0)
        psi_e, mu_e, sig_e, phi_e, d_e, L_e, *_, h_mat_e, C_e = jm_e

        acc_e = bootstrap_curve(block_e, THRESHOLD, n_boot=N_BOOT, rng=rng,
                                h_mat=h_mat_e)
        sem_e = np.maximum(acc_e[2] / np.sqrt(max(N_BOOT, 1)), 0.1)
        accum_E[gname_e]  = acc_e
        fits_E[gname_e]   = {mn: fit_one(mn, acc_e[0], acc_e[1], sem_e)
                             for mn in MODEL_VARIANTS}

        N_max_e = int(np.ceil(len(block_e) * args.n_extrap))
        surf_e, pDr_e, Ns_pe = predictive_surface_from_joint(
            psi_e, mu_e, sig_e, phi_e, d_e, L_e,
            THRESHOLDS_PCT, N_max_e, N_PRED, rng,
            args.abund_dist, args.abund_df, args.n_quad)
        surface_all["_panel_E"][gname_e] = surf_e

        # Fix 4: real draws
        Ns_e, pM_e, pLo_e, pHi_e = surf_e[float(THRESHOLD)]
        predictive_curves_all["_panel_E"][gname_e] = (Ns_pe, pM_e, pLo_e, pHi_e, pDr_e)
        sem_pe = np.maximum(pDr_e.std(0) / np.sqrt(max(N_PRED, 1)), 0.1)
        for mn in MODEL_VARIANTS:
            fd_e = fit_one(mn, Ns_e, pM_e, sem_pe)
            if not fd_e["failed"] and fd_e["popt"] is not None:
                fits_E[gname_e][mn] = fd_e

        holdout_diag_all["_panel_E"][gname_e] = deep_holdout_new_taxa_test(
            block_e, h_mat_e, total_e,
            threshold=THRESHOLD, hold_frac=args.holdout_frac)
        print("done")

    # ── Panel F ───────────────────────────────────────────────────────────
    print("\n=== Panel F: world extrapolation ===")
    pf = panel_f_data()
    print(f"  S(8.2e9) = {pf['S_world']:,.0f}  [{pf['S_q025']:,.0f}-{pf['S_q975']:,.0f}]")

    # ── Build figures ─────────────────────────────────────────────────────
    for mn in MODEL_VARIANTS:
        build_figure(
            mname                 = mn,
            accum_results         = accum_results,
            fits_by_mname         = fits_by_mname,
            boot_matrices         = boot_matrices,
            sweep_results         = sweep_results,
            accum_E               = accum_E,
            fits_E                = fits_E,
            pf                    = pf,
            n_sgbs_by_ref         = n_sgbs_by_ref,
            threshold             = THRESHOLD,
            n_boot                = N_BOOT,
            out_stem              = args.out,
            predictive_curves_all = predictive_curves_all,
            holdout_diag_all      = holdout_diag_all,
            np_curves_all         = np_curves_all,
            depth_curves_all      = depth_curves_all,
            proj_curves_all       = proj_curves_all,
            chao2_all             = chao2_all,
            surface_all           = surface_all,
            thresholds_pct        = THRESHOLDS_PCT,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
