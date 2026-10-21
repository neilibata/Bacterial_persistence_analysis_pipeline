"""
Shared utilities used by multiple pipeline modules.

Functions:
    normalise_country, get_continent      — geography
    classify_macro_group, classify_group   — lifestyle x health classification
    is_yes, clr_transform                 — data helpers
    load_id_list, load_phylo_list,
    load_top100_list                       — file loaders
    assign_volcano_category,
    assign_volcano_category_7              — volcano colouring
"""

import os
import numpy as np
import pandas as pd

from . import config


# ════════════════════════════════════════════════════════════════════
#  GEOGRAPHY
# ════════════════════════════════════════════════════════════════════

def normalise_country(s):
    """Normalise raw country string using the alias table in config."""
    if pd.isna(s):
        return None
    return config.COUNTRY_ALIASES.get(str(s).strip(), str(s).strip())


def get_continent(country):
    """Map a country name (already normalised or raw) to its continent.

    Tries a direct lookup first, then falls back to running normalise_country
    in case an un-normalised code/alias was passed in directly.
    """
    if pd.isna(country) or country is None:
        return None
    s = str(country).strip()
    # Direct lookup (fast path — country is usually already normalised)
    result = config.COUNTRY_TO_CONTINENT.get(s)
    if result is not None:
        return result
    # Fallback: try normalising first (handles raw codes passed in directly)
    norm = normalise_country(s)
    return config.COUNTRY_TO_CONTINENT.get(norm)


# ════════════════════════════════════════════════════════════════════
#  LIFESTYLE x HEALTH CLASSIFICATION
# ════════════════════════════════════════════════════════════════════

def classify_macro_group(row):
    """Industrial / Non-Industrial (ignoring health status)."""
    life = row.get("Lifestyle", None)
    if life == "Industrial":
        return "Industrial"
    elif life in ("H-G", "Transitional"):
        return "Non-Industrial"
    return None


def classify_group(row):
    """Four-way: Industrial / Industrial Unhealthy / Non-Industrial / Non-Industrial Unhealthy."""
    uh   = row.get("Unhealthy", None)
    life = row.get("Lifestyle", None)
    if uh == "Unhealthy":
        if life in ("H-G", "Transitional"):
            return "Non-Industrial Unhealthy"
        elif life == "Industrial":
            return "Industrial Unhealthy"
    elif uh == "Healthy":
        if life in ("H-G", "Transitional"):
            return "Non-Industrial"
        elif life == "Industrial":
            return "Industrial"
    return None


# ════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ════════════════════════════════════════════════════════════════════

def is_yes(val):
    """Flexible truth test for metadata yes/no columns.

    Matches: 1, 1.0, yes, y, true, obese, overweight, positive.
    Also treats any non-zero numeric as True.
    """
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    if s in ("1", "1.0", "yes", "y", "true", "positive",
             "obese", "overweight"):
        return True
    # Try numeric: any value > 0 is True
    try:
        return float(s) > 0
    except (ValueError, TypeError):
        return False


def clr_transform(df, pseudocount=None):
    """
    Centred log-ratio transform on a samples x taxa DataFrame.

    Adds a small pseudocount, takes log, then subtracts the row-wise
    geometric mean (= arithmetic mean in log-space).
    """
    if pseudocount is None:
        pseudocount = config.CLR_PSEUDOCOUNT
    log_df = np.log(df + pseudocount)
    return log_df.subtract(log_df.mean(axis=1), axis=0)


# ════════════════════════════════════════════════════════════════════
#  FILE LOADERS
# ════════════════════════════════════════════════════════════════════

def load_id_list(filepath):
    """Load a simple one-ID-per-line file, return a set."""
    if not os.path.exists(filepath):
        return set()
    ids = set()
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line)
    return ids


def load_phylo_list(filepath):
    """Load a phylo list CSV/TSV with at least an 'ID' column. Returns DataFrame."""
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=["ID"])
    try:
        df = pd.read_csv(filepath, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(filepath, sep="\t")
    # Normalise column names
    for c in df.columns:
        if c.strip().lower() in ("id", "sgb", "sgb_id", "microbe"):
            df = df.rename(columns={c: "ID"})
            break
    if "ID" not in df.columns and df.shape[1] >= 1:
        df = df.rename(columns={df.columns[0]: "ID"})
    df["ID"] = df["ID"].astype(str).str.strip()
    return df


def load_top100_list(filepath):
    """Load a top-100 SGB ID list (one per line)."""
    return sorted(load_id_list(filepath))


# ════════════════════════════════════════════════════════════════════
#  VOLCANO CATEGORY ASSIGNMENT
# ════════════════════════════════════════════════════════════════════

def assign_volcano_category(microbe, vanish_set, blossum_set):
    """Legacy 5-category assignment for backward compatibility."""
    in_v = microbe in vanish_set
    in_b = microbe in blossum_set
    if in_v and in_b:
        return "Both"
    if in_v:
        return "VANISH"
    if in_b:
        return "BloSSUM"
    return "Other"


def assign_volcano_category_7(microbe, vanish_set, blossum_set,
                               vanish_plus_set, blossum_plus_set):
    """
    Seven-category assignment for updated volcano plots.

    Categories (in draw order, back to front):
        1. None                 — not in any list
        2. VANISH only          — in VANISH but NOT in VANISH+
        3. VANISH+ only         — in VANISH+ but NOT in VANISH
        4. VANISH  &  VANISH+     — in both VANISH and VANISH+
        5. BloSSUM only         — in BloSSUM but NOT in BloSSUM+
        6. BloSSUM+ only        — in BloSSUM+ but NOT in BloSSUM
        7. BloSSUM  &  BloSSUM+   — in both BloSSUM and BloSSUM+

    If a microbe is in both VANISH-family AND BloSSUM-family,
    BloSSUM family takes precedence (rarer, more interesting).
    """
    in_v  = microbe in vanish_set
    in_vp = microbe in vanish_plus_set
    in_b  = microbe in blossum_set
    in_bp = microbe in blossum_plus_set

    # BloSSUM family takes priority when overlapping with VANISH family
    if in_b and in_bp:
        return "BloSSUM ∩ BloSSUM+"
    if in_bp and not in_b:
        return "BloSSUM+ only"
    if in_b and not in_bp:
        return "BloSSUM only"

    # VANISH family
    if in_v and in_vp:
        return "VANISH ∩ VANISH+"
    if in_vp and not in_v:
        return "VANISH+ only"
    if in_v and not in_vp:
        return "VANISH only"

    return "No list"
