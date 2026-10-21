"""Check that no ineligible contrast reached a figure in a finished run.

The eligibility rule is a study count, not a list: a study contributes a
stratum only where both arms carry at least config.CMH_MIN_STRATUM samples,
and a contrast is shown only where at least config.MIN_STRATA_TO_PLOT such
strata exist. The tables deliberately keep every contrast and flag it with an
`eligible` column; the figures must not show the excluded ones.

A figure's data object carries the panels it was built from, which are a
superset: the curve is fitted for every contrast and filtered at draw time.
So the panels alone cannot answer the question. Each health figure therefore
records `extras["drawn_conditions"]`, which is what it actually drew, and this
script checks that.

    python tools/check_run_conditions.py <results-root>

Exit code is the number of problems, so it can gate a run.
"""
from __future__ import annotations

import os
import pickle
import sys
import glob
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from persistence import config  # noqa: E402

HEALTH_SLOTS = {"S24", "S25", "S26", "S27", "S28", "S29", "S30", "F5"}


def eligible_set(results_root: str, ref: str = "combined"):
    """The contrasts the rule admits, read from the run's own CMH table."""
    import numpy as np
    pat = os.path.join(results_root, ref, "section3_health", "main",
                       "health_per_condition_cmh_*.csv")
    hits = sorted(f for f in glob.glob(pat) if "combined" not in os.path.basename(f))
    if not hits:
        return None, None
    cmh = pd.read_csv(hits[0])
    if "n_studies" not in cmh.columns:
        return None, None
    k = cmh.groupby("Condition")["n_studies"].max()
    ok = {c for c, v in k.items()
          if np.isfinite(v) and v >= config.MIN_STRATA_TO_PLOT}
    return set(k.index), ok


def main(results_root: str) -> int:
    judged, ok = eligible_set(results_root)
    if ok is None:
        print("  no CMH table found — cannot judge eligibility")
        return 1
    print(f"  rule: >= {config.CMH_MIN_STRATUM} samples per arm in "
          f">= {config.MIN_STRATA_TO_PLOT} studies")
    print(f"  admits {len(ok)}: {', '.join(sorted(ok))}")
    print(f"  excludes {len(judged - ok)}: {', '.join(sorted(judged - ok))}\n")

    problems, checked, unknown = 0, 0, []
    for f in sorted(glob.glob(os.path.join(results_root, "**", "*_data.pkl"),
                              recursive=True)
                    + glob.glob(os.path.join(os.path.dirname(results_root),
                                             "main_text_figures", "**",
                                             "*_data.pkl"), recursive=True)):
        try:
            with open(f, "rb") as fh:
                obj = pickle.load(fh)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        slot = str(obj.get("slot", ""))
        drawn = (obj.get("extras") or {}).get("drawn_conditions")
        name = os.path.relpath(f, results_root)
        if drawn is None:
            if slot in HEALTH_SLOTS:
                unknown.append((slot, name))
            continue
        checked += 1
        bad = sorted(set(map(str, drawn)) & (judged - ok))
        if bad:
            problems += 1
            print(f"  LEAK  {slot:4s} {name}")
            print(f"        drew {bad}")
        else:
            print(f"  ok    {slot:4s} drew {len(drawn)}: "
                  f"{', '.join(map(str, drawn))}")

    for slot, name in unknown:
        print(f"  ?     {slot:4s} {name} — no drawn_conditions recorded")
    if unknown:
        print("\n  A health figure with no drawn_conditions was built by code "
              "older than this check; rerun that figure or inspect it by eye.")

    # The tables must keep every contrast and flag it, not silently drop rows.
    tables = sorted(glob.glob(os.path.join(
        results_root, "**", "table_s6*_*.csv"), recursive=True)
        + glob.glob(os.path.join(
            results_root, "**", "table_s12_*.csv"), recursive=True))
    for t in tables:
        d = pd.read_csv(t)
        if "eligible" not in d.columns:
            problems += 1
            print(f"  TABLE {os.path.relpath(t, results_root)} has no "
                  f"`eligible` column — the supplement table cannot be "
                  f"filtered without hand-editing")

    print(f"\n  {checked} figure(s) checked, {problems} problem(s)")
    return problems


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
