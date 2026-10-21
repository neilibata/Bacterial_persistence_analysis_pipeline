"""
Centralized statistics reporter.

Every quantitative claim in the manuscript is logged here with:
  - a unique tag (matching [RC*] or descriptive)
  - the section it belongs to
  - whether it goes in main text or supplement
  - the computed value(s)
  - the test/method used

At the end of the pipeline run, this writes:
  1. stats_report.txt           — human-readable summary
  2. stats_report.csv           — machine-readable table
  3. manuscript_fill_ins.txt    — blanks filled in for the manuscript
"""

import os
import csv
import numpy as np
import pandas as pd
from collections import OrderedDict


class StatsReport:
    """Accumulates statistics across the pipeline and writes them out."""

    def __init__(self, ref_name, out_root="results"):
        self.ref_name = ref_name
        self.out_root = os.path.join(out_root, ref_name)
        os.makedirs(self.out_root, exist_ok=True)
        self._rows = []  # list of dicts

    def add(self, tag, section, placement, description, value,
            test=None, p_value=None, ci_lo=None, ci_hi=None,
            n1=None, n2=None, effect_size=None, extra=None):
        """
        Record a statistic.

        Parameters
        ----------
        tag         : str   Unique identifier, e.g. "depth_mean_HG"
        section     : str   "s1", "s2", "s3"
        placement   : str   "main" or "supplement"
        description : str   Human-readable description
        value       : any   The statistic value (will be str-ified)
        test        : str   Statistical test used (optional)
        p_value     : float (optional)
        ci_lo, ci_hi: float (optional) confidence interval bounds
        n1, n2      : int   Group sample sizes (optional)
        effect_size : float (optional)
        extra       : dict  Any additional info (optional)
        """
        row = OrderedDict([
            ("tag", tag),
            ("section", section),
            ("placement", placement),
            ("description", description),
            ("value", _fmt(value)),
            ("test", test or ""),
            ("p_value", _fmt_p(p_value)),
            ("ci_lo", _fmt(ci_lo)),
            ("ci_hi", _fmt(ci_hi)),
            ("n1", _fmt(n1)),
            ("n2", _fmt(n2)),
            ("effect_size", _fmt(effect_size)),
        ])
        if extra:
            for k, v in extra.items():
                row[k] = _fmt(v)
        self._rows.append(row)
        # Also print to console
        p_str = f"  p={_fmt_p(p_value)}" if p_value is not None else ""
        print(f"  [STAT] {tag}: {_fmt(value)}{p_str}  — {description}")

    def _merge_with_existing(self, df):
        """Carry forward tags from a previous run that this run did not produce.

        The report is written from scratch each invocation, so a partial run
        (`--only s2`) used to discard every statistic from the sections it did
        not execute — silently truncating the accumulated report from a full
        run. Tags this run produced always win; anything else is preserved, so
        a partial re-run updates its own section and leaves the rest standing.
        """
        csv_path = os.path.join(self.out_root, "stats_report.csv")
        if not os.path.exists(csv_path):
            return df
        try:
            prev = pd.read_csv(csv_path)
        except Exception as exc:                      # corrupt/empty: start clean
            print(f"  [StatsReport] could not read existing report ({exc}); overwriting")
            return df
        if "tag" not in prev.columns or not len(prev):
            return df

        fresh = set(df["tag"]) if "tag" in df.columns else set()
        carried = prev[~prev["tag"].isin(fresh)]
        if not len(carried):
            return df
        print(f"  [StatsReport] carried {len(carried):,} statistic(s) forward from the "
              f"previous report; {len(fresh):,} updated by this run")
        return pd.concat([df, carried], ignore_index=True, sort=False)

    def save(self):
        """Write stats_report.txt, stats_report.csv, and manuscript_fill_ins.txt."""
        if not self._rows:
            print("  [StatsReport] No statistics recorded.")
            return

        df = self._merge_with_existing(pd.DataFrame(self._rows))

        # ── CSV ──
        csv_path = os.path.join(self.out_root, "stats_report.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        # ── Human-readable TXT ──
        txt_path = os.path.join(self.out_root, "stats_report.txt")
        with open(txt_path, "w") as f:
            f.write(f"Statistics Report — {self.ref_name}\n")
            f.write("=" * 70 + "\n\n")

            current_section = None
            for _, row in df.iterrows():
                sec = row.get("section", "")
                if sec != current_section:
                    current_section = sec
                    sec_label = {
                        "s1": "Section 1: Diversity & Composition",
                        "s2": "Section 2: Differential Prevalence",
                        "s3": "Section 3: Health Associations",
                    }.get(sec, sec)
                    f.write(f"\n{'─'*60}\n")
                    f.write(f"  {sec_label}\n")
                    f.write(f"{'─'*60}\n\n")

                place = "[MAIN]" if row.get("placement") == "main" else "[SUPP]"
                f.write(f"  {place} {row['tag']}\n")
                f.write(f"    {row['description']}\n")
                f.write(f"    Value: {row['value']}\n")
                if row.get("test"):
                    f.write(f"    Test:  {row['test']}\n")
                if row.get("p_value"):
                    f.write(f"    p:     {row['p_value']}\n")
                if row.get("ci_lo") or row.get("ci_hi"):
                    f.write(f"    CI:    [{row.get('ci_lo', '?')}, "
                            f"{row.get('ci_hi', '?')}]\n")
                if row.get("n1"):
                    f.write(f"    n:     {row.get('n1', '')}"
                            f"{' vs ' + str(row.get('n2', '')) if row.get('n2') else ''}\n")
                if row.get("effect_size"):
                    f.write(f"    ES:    {row['effect_size']}\n")
                f.write("\n")

        print(f"  Saved: {txt_path}")

        # ── Manuscript fill-ins ──
        fill_path = os.path.join(self.out_root, "manuscript_fill_ins.txt")
        with open(fill_path, "w") as f:
            f.write(f"Manuscript Fill-ins — {self.ref_name}\n")
            f.write("=" * 70 + "\n")
            f.write("Copy-paste these into the manuscript where blanks appear.\n\n")

            for _, row in df.iterrows():
                tag = row["tag"]
                val = row["value"]
                p = row.get("p_value", "")
                desc = row["description"]
                f.write(f"[{tag}]: {val}")
                if p:
                    f.write(f" (p = {p})")
                f.write(f"\n  → {desc}\n\n")

        print(f"  Saved: {fill_path}")

    def get_df(self):
        return pd.DataFrame(self._rows)


# ── Formatting helpers ──

def _fmt(val):
    if val is None:
        return ""
    if isinstance(val, float):
        if np.isnan(val):
            return ""
        if abs(val) < 0.001 and val != 0:
            return f"{val:.2e}"
        if abs(val) > 1e6:
            return f"{val:.4e}"
        return f"{val:.4f}"
    return str(val)


def _fmt_p(val):
    if val is None:
        return ""
    if isinstance(val, float):
        if np.isnan(val):
            return ""
        if val < 1e-300:
            return "< 1e-300"
        if val < 0.001:
            return f"{val:.2e}"
        return f"{val:.4f}"
    return str(val)
