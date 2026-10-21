#!/usr/bin/env python3
"""Pre-flight audit for the final pipeline run.

Checks the curated metadata and every element added for the demographic
analysis, and reports PASS/FAIL per check so the decision to launch an ~84
hour run rests on something other than recollection.

Three groups:
  1. metadata   -- M1's integrity, and that the pipeline will actually read it
  2. additions  -- the demographic analysis, its figure, and its registrations
  3. readiness  -- the tree compiles, the CLI parses, nothing was clobbered

    python tools/audit_final_run.py
    python tools/audit_final_run.py --verbose
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results" / "combined" / "section3_health" / "main"
SUPP = ROOT / "results" / "combined" / "section3_health" / "supplement"

_rows: list[tuple[str, str, bool | None, str]] = []


def _code_changed_at(relpath):
    """When the code in `relpath` last changed.

    The file's mtime is not that: unpacking a code archive (git archive stamps
    every file with the commit time) or a fresh checkout makes unchanged code
    look newer than tables the run wrote, and a table then reads STALE when it
    is not. The time of the last commit that touched the file is the real
    answer; fall back to mtime where there is no git history. A local edit
    that is not committed yet still counts, through the mtime.
    """
    import subprocess
    p = ROOT / relpath
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%ct",
                              "--", relpath], capture_output=True, text=True,
                             timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                                "--", relpath], capture_output=True, text=True,
                               timeout=20).stdout.strip()
        if out and not dirty:
            return float(out)
    except (OSError, subprocess.SubprocessError):
        pass
    return p.stat().st_mtime


def _called_name(node):
    """The bare name a Call node invokes: `f(...)` -> f, `m.f(...)` -> f."""
    import ast
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _module_functions(relpath):
    """{name: [FunctionDef, ...]} for every function defined in `relpath`.

    Nested definitions are included under their own name, so a helper hidden
    inside another function is still found.
    """
    import ast
    tree = ast.parse((ROOT / relpath).read_text())
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(n.name, []).append(n)
    return out


def _reachable_functions(relpath, root_fn):
    """Every function in `relpath` that `root_fn` can reach, by name.

    A name-matched call graph over one module, which is all this file needs:
    section3_health.py calls its own helpers by bare name throughout. It is
    deliberately over-inclusive -- a nested def's body counts as part of its
    parent, and an attribute call `x.f()` matches a module function named
    `f` -- because the checks built on it assert that something is NOT
    reachable, and over-inclusion can only make such a check stricter.

    None when `root_fn` is not defined in the module, so a caller can fail
    rather than silently pass on a renamed entry point.
    """
    import ast
    bodies = _module_functions(relpath)
    if root_fn not in bodies:
        return None
    seen, stack = set(), [root_fn]
    while stack:
        fn = stack.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for node in bodies.get(fn, []):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    nm = _called_name(sub)
                    if nm and nm in bodies and nm not in seen:
                        stack.append(nm)
    return seen


def _string_literals(node):
    """The str constants of a literal, or of a literal tuple/list/set."""
    import ast
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return {e.value for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    return set()


def _excluding_test(test, token):
    """Line of a sub-test of `test` that excludes `token`, else None.

    Accepts the four ways the tree writes an exclusion: `x not in (...)`,
    `x != "tok"`, `not (x in (...))` and `not (x == "tok")`.
    """
    import ast
    for sub in ast.walk(test):
        if isinstance(sub, ast.Compare):
            for op, cmp_ in zip(sub.ops, sub.comparators):
                if (isinstance(op, (ast.NotIn, ast.NotEq))
                        and token in _string_literals(cmp_)):
                    return sub.lineno
        if isinstance(sub, ast.UnaryOp) and isinstance(sub.op, ast.Not):
            for inner in ast.walk(sub.operand):
                if isinstance(inner, ast.Compare):
                    for op, cmp_ in zip(inner.ops, inner.comparators):
                        if (isinstance(op, (ast.In, ast.Eq))
                                and token in _string_literals(cmp_)):
                            return inner.lineno
    return None


def _guards_against(relpath, fn, token):
    """Does `fn` branch on a test that keeps `token` off the panel?

    Reading the whole file for the token passed on prose. Both files explain
    the guard in a comment beside it, and section3_health.py also ASSIGNS
    `star = "untested"`, so `'"untested"' in txt` was satisfied four times
    over in a file whose guard had been deleted -- which is the defect this
    defends against: the literal word "untested" printed where a significance
    star belongs, in Figure 5 panel D and in the richness panels behind
    S12/S13/S24.

    So: find the named function in the AST, and require an exclusion test
    that is the `test` of a branch, so it governs the drawing rather than
    sitting in a dead expression, a comment or a docstring.

    Returns (ok, detail).
    """
    import ast
    try:
        defs = _module_functions(relpath).get(fn, [])
    except (OSError, SyntaxError) as exc:
        return False, f"{relpath}: {type(exc).__name__}: {exc}"
    if not defs:
        return False, f"{fn}() not found in {relpath} (renamed or deleted)"
    for node in defs:
        for sub in ast.walk(node):
            if not isinstance(sub, (ast.If, ast.IfExp, ast.While)):
                continue
            ln = _excluding_test(sub.test, token)
            if ln is not None:
                return True, f"guarded at {relpath}:{ln}"
    return False, (f'no branch inside {fn}() excludes "{token}" -- the word '
                   "itself would be drawn where the star goes")


def _tests_not_flag(test, flag):
    """True if `test` requires `flag` to be false: `not flag`, `flag is
    False`, `flag == False`."""
    import ast
    for sub in ast.walk(test):
        if (isinstance(sub, ast.UnaryOp) and isinstance(sub.op, ast.Not)
                and isinstance(sub.operand, ast.Name)
                and sub.operand.id == flag):
            return True
        if isinstance(sub, ast.Compare) and isinstance(sub.left, ast.Name) \
                and sub.left.id == flag:
            for op, cmp_ in zip(sub.ops, sub.comparators):
                if (isinstance(op, (ast.Is, ast.Eq))
                        and isinstance(cmp_, ast.Constant)
                        and cmp_.value is False):
                    return True
    return False


def _calls_guarded_by_not(relpath, fn, callee, flag):
    """Is every call to `callee` inside `fn` under a branch requiring
    `not flag`?

    Returns (ok, detail). A call that is not under such a branch, and a
    `callee` that `fn` does not call at all, both fail: the first is the
    guard having been removed, the second the call having moved somewhere
    this check can no longer see.
    """
    import ast
    defs = _module_functions(relpath).get(fn, [])
    if not defs:
        return False, f"{fn}() not found in {relpath} (renamed or deleted)"
    spans, calls = [], []
    for node in defs:
        for sub in ast.walk(node):
            if isinstance(sub, ast.If) and _tests_not_flag(sub.test, flag):
                for st in sub.body:
                    spans.append((st.lineno,
                                  getattr(st, "end_lineno", st.lineno)))
            if isinstance(sub, ast.Call) and _called_name(sub) == callee:
                calls.append(sub.lineno)
    if not calls:
        return False, f"{fn}() no longer calls {callee}()"
    loose = [ln for ln in calls
             if not any(lo <= ln <= hi for lo, hi in spans)]
    if loose:
        return False, (f"{callee}() called at {relpath}:{loose} outside any "
                       f"`not {flag}` branch")
    return True, f"{len(calls)} call(s), all under `not {flag}`"


def check(group, name, ok, detail=""):
    _rows.append((group, name, ok, str(detail)))
    return ok


def audit_metadata():
    g = "metadata"
    from persistence import config

    meta_path = ROOT / config.META_FILE
    check(g, "config.META_FILE resolves to the curated M1",
          config.META_FILE == "final_run/metadata_M1.csv", config.META_FILE)
    if not meta_path.exists():
        check(g, "metadata file exists", False, str(meta_path))
        return None
    m = pd.read_csv(meta_path, low_memory=False)
    # 18,299 = 18,303 minus the 4 Carter rows whose library has no run in the
    # matrix (tools/fix_carter_runs.py). Pin the number so an unexplained change
    # still fails, but record why it moved rather than leaving a bare literal.
    _EXPECTED_ROWS = 18299
    check(g, f"row count is {_EXPECTED_ROWS:,}", len(m) == _EXPECTED_ROWS, f"{len(m):,}")
    _cm = m["Chosen_match_ID"].astype(str)
    check(g, "Chosen_match_ID has no collisions",
          _cm.nunique() == len(_cm), f"{len(_cm) - _cm.nunique()} duplicated")
    # The original 115 curated columns must all survive; added columns are fine.
    # An exact-count assertion fails the moment a column is deliberately added
    # (Subject_ID, subject_primary, mdb_case_eligible, mdb_study_condition,
    # health_source, Subject_ID_source), which is a stale check rather than a
    # defect -- so assert the invariant that matters: nothing was dropped.
    _BASE_COLS = 115
    _missing = []
    _bak = Path(config.META_FILE + ".pre_subject_ids")
    if _bak.exists():
        _orig = pd.read_csv(_bak, nrows=0).columns
        _missing = [c for c in _orig if c not in m.columns]
    # These five were dropped deliberately by tools/clean_m1_columns.py:
    # `Diseased` and `Diseased (all)` contradicted the authoritative `Unhealthy`
    # on 2,037 and 1,816 samples and were unread by the analysis;
    # `Obesity (old)` was fully redundant (the obesity union is 1,719 with or
    # without it, and no sample loses obese status); `Summary totals` and
    # `Which column?` were 0.1%-populated scratch columns. Anything ELSE going
    # missing is still a defect.
    # `disease_BD` was SPLIT, not lost: it conflated RettedalAE_2021_NZL's
    # osteoporosis cohort (54) with YeZ_2018_CHN's Behcet's cohort (20), so
    # tools/split_disease_bd.py replaced it with disease_bone_disease and
    # disease_Behcets. Require both replacements to exist, so a genuine loss
    # still fails.
    _INTENTIONAL_DROPS = {"Diseased", "Diseased (all)", "Obesity (old)",
                          "Summary totals", "Which column?", "disease_BD"}
    _unexpected = [c for c in _missing if c not in _INTENTIONAL_DROPS]
    check(g, "no curated column dropped beyond the documented set",
          not _unexpected, _unexpected or f"only the {len(_INTENTIONAL_DROPS)} intended")
    _SPLIT_HEIRS = ["disease_bone_disease", "disease_Behcets"]
    _absent = [c for c in _SPLIT_HEIRS if c not in m.columns]
    check(g, "disease_BD's two replacements both present", not _absent,
          _absent or "bone_disease + Behcets")
    if not _absent:
        _b = int((m["disease_bone_disease"].astype(str).str.strip() == "Yes").sum())
        _h = int((m["disease_Behcets"].astype(str).str.strip() == "Yes").sum())
        check(g, "the BD split conserved all 74 positives", _b + _h == 74,
              f"{_b} bone + {_h} Behcets = {_b + _h}")
    check(g, f"column count >= {_BASE_COLS}", len(m.columns) >= _BASE_COLS,
          len(m.columns))

    # identity
    cm = m["Chosen_match_ID"].astype(str)
    check(g, "Chosen_match_ID fully populated", cm.notna().all() and
          (cm.str.strip() != "").all())
    check(g, "BioSample 100% populated",
          m["BioSample"].notna().all(), f"{m['BioSample'].notna().sum():,}")

    # health label -- the one the analysis contrasts on
    uh = m["Unhealthy"].astype(str).str.strip()
    vals = set(uh.unique()) - {"nan"}
    check(g, "Unhealthy has only Healthy/Unhealthy", vals == {"Healthy", "Unhealthy"},
          sorted(vals))
    check(g, "Unhealthy counts match the curation (11,459 / 6,496)",
          int((uh == "Healthy").sum()) == 11459 and
          int((uh == "Unhealthy").sum()) == 6496,
          f"{int((uh=='Healthy').sum()):,} / {int((uh=='Unhealthy').sum()):,}")

    # obesity rule: no obese sample left labelled Healthy
    ob = m[[c for c in config.OBESITY_COLUMNS if c in m.columns]].apply(
        lambda r: any(str(v).strip().lower() in ("yes", "1", "1.0") for v in r),
        axis=1)
    bmi = pd.to_numeric(m["BMI"], errors="coerce")
    obese = ob | bmi.ge(30)
    check(g, "no obese sample is still labelled Healthy",
          int((obese & uh.eq("Healthy")).sum()) == 0,
          f"{int((obese & uh.eq('Healthy')).sum())} violations")

    # lifestyle
    life = config.canonicalize_lifestyle_series(m["Lifestyle"])
    keys = set(life.dropna().unique())
    check(g, "Lifestyle canonicalises to the internal keys",
          keys <= {"Industrial", "Transitional", "H-G", "Unclear"}, sorted(keys))

    # the five roll-ups restored today, byte-identical to their source
    src_p = ROOT / "metadata_Sep_21_2026_obesity_unhealthy.csv"
    # The three roll-ups still shipped. `Diseased (gut and metabolic)` is now
    # DERIVED from the disease flags rather than inherited, so it is checked for
    # presence but excluded from the byte-identical comparison below.
    # All three surviving roll-ups are now DERIVED from the disease flags by
    # tools/_disease_categories.py, so "byte-identical to the source" is no
    # longer the right invariant -- the source values were the drifted ones
    # (`Diseased (other)` was 802 rows out, 741 of its Yes carrying no flag).
    # Check instead that each column still equals its derivation: that is what
    # stops a flag correction leaving a roll-up stale.
    RESTORED = ["Diseased (gut and metabolic)",
                "Diseased (metabolic) or obese", "Diseased (other)"]
    check(g, "the 3 retained roll-up columns are present",
          all(c in m.columns for c in RESTORED),
          [c for c in RESTORED if c not in m.columns] or "all present")
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        import _disease_categories as _dcat
        _drift = []
        for _c, _v in _dcat.derive_rollups(m, config.OBESITY_COLUMNS).items():
            if _c in m.columns:
                _cur = m[_c].astype(str).str.strip().str.lower().eq("yes")
                if not (_cur == _v).all():
                    _drift.append(f"{_c} ({int((_cur != _v).sum())} rows)")
        check(g, "roll-ups match their derivation from the flags",
              not _drift, _drift or "all three consistent")
    except Exception as _e:
        check(g, "roll-ups match their derivation from the flags", False,
              f"{type(_e).__name__}: {_e}")

    check(g, "disease_konzo present with 60 positives",
          "disease_konzo" in m.columns and
          int((m.get("disease_konzo").astype(str).str.strip() == "Yes").sum()) == 60,
          int((m["disease_konzo"].astype(str).str.strip() == "Yes").sum())
          if "disease_konzo" in m.columns else "absent")
    check(g, "the 3 unread columns stay dropped",
          not any(c in m.columns for c in ("Index", "Healthy", "Unhealthy (check)")),
          [c for c in ("Index", "Healthy", "Unhealthy (check)") if c in m.columns]
          or "none present")
    check(g, "Secondary_run_IDs retained (feeds S1/S2/S3 via sec_map)",
          "Secondary_run_IDs" in m.columns,
          f"{int(m['Secondary_run_IDs'].notna().sum()):,} populated"
          if "Secondary_run_IDs" in m.columns else "MISSING")

    # The stale 2026-08-14 metadata_M1.csv at the repo root has now misled
    # two separate tools into reading the wrong cohort (the metadata explorer
    # defaulted to it for six weeks; preflight_final_run compared against it
    # and declared the curated file out of sync). Nothing should resolve to it
    # any more, so check that nothing does.
    # None of the three is gated on the stale file still being there. All
    # three used to sit under `if root_m1.exists():`, and
    # docs/the 2026-09-28 session record recommends removing that file -- so
    # doing the recommended cleanup silently DELETED the checks instead of
    # passing them, and the two that actually protect against silent
    # repointing (preflight's path and the explorer's) are not about the root
    # file at all. The audit then printed three fewer PASS lines and said
    # nothing about having skipped them.
    root_m1 = ROOT / "metadata_M1.csv"
    if root_m1.exists():
        n_stale = sum(1 for _ in open(root_m1)) - 1
        _where = (f"root file still present with {n_stale:,} rows "
                  "(consider archiving)")
    else:
        _where = "root file gone, which is the recommended end state"
    check(g, "no tool resolves to the stale root metadata_M1.csv",
          config.META_FILE != "metadata_M1.csv", _where)
    from tools import preflight_final_run as _pf
    src = Path(_pf.__file__).read_text()
    check(g, "preflight compares against the curated M1",
          'ROOT / "final_run" / "metadata_M1.csv"' in src)
    ex = (ROOT / "tools/metadata_explorer/build_explorer.py").read_text()
    check(g, "explorer follows config.META_FILE",
          "REPO_ROOT / _cfg.META_FILE" in ex)

    # demographic fields the new analysis needs
    from persistence.health_cmh import (age_band_series, bmi_band_series,
                                           sex_series)
    ab, sx, bb = age_band_series(m), sex_series(m), bmi_band_series(m)
    check(g, "Age bands derive (5 bands)",
          set(ab.dropna().unique()) == {b for b, _l, _h in config.AGE_BANDS},
          f"{ab.notna().sum():,} banded")
    check(g, "Sex normalises to Female/Male",
          set(sx.dropna().unique()) == {"Female", "Male"},
          f"{sx.notna().sum():,} resolved")
    check(g, "BMI bands derive (4 bands)",
          set(bb.dropna().unique()) == {b for b, _l, _h in config.BMI_BANDS},
          f"{bb.notna().sum():,} banded")
    return m


def audit_additions(m):
    g = "additions"
    from persistence import config
    from persistence.health_cmh import build_group_masks
    from persistence.section3_health import _health_comparisons, _out_prefix

    check(g, "DEMOGRAPHIC_MAP has 7 contrasts",
          len(config.DEMOGRAPHIC_MAP) == 7, len(config.DEMOGRAPHIC_MAP))
    if m is not None:
        mm = m.copy()
        mm["Lifestyle"] = config.canonicalize_lifestyle_series(mm["Lifestyle"])
        bad = []
        with contextlib.redirect_stdout(io.StringIO()):
            for label, ctype in config.DEMOGRAPHIC_MAP.items():
                a, b = build_group_masks(mm, ctype)
                if (a & b).any() or a.sum() < 10 or b.sum() < 10:
                    bad.append(label)
        check(g, "every demographic mask: arms disjoint and >=10 per arm",
              not bad, bad or "all 7 ok")

    # the refactor must not have changed the disease comparison set
    #
    # "Obese (healthy only)" used to head this list. It was a sensitivity
    # contrast and was deliberately removed -- config.HEALTH_PLOT_SKIP's
    # comment records it as "a contrast that no longer exists" -- so expecting
    # it made this check FAIL for the very change that was intended. The
    # builder still carries the obese_healthy_only branch, which is what keeps
    # the removal reversible; it is simply not iterated.
    from collections import OrderedDict
    expect = OrderedDict([("Unhealthy", ("unhealthy", None)),
                          ("Obese", ("obese", None))])
    for cl, cs in config.DISEASE_NCD_MAP.items():
        t, d = cs["type"], cs.get("cols", None)
        if t in ("obese", "overweight"):
            continue
        expect[cl] = (t, d)
    _act = list(_health_comparisons(False).items())
    check(g, "disease comparison set unchanged by the refactor",
          _act == list(expect.items()),
          f"{len(_act)} comparisons"
          + ("" if _act == list(expect.items()) else
             f"; expected {[k for k in expect]}, got {[k for k, _ in _act]}"))
    check(g, "output prefixes separate the two runs",
          (_out_prefix(False), _out_prefix(True)) == ("health", "demographic"))

    # registrations
    man = (ROOT / "tools/additional_figures_manifest.txt").read_text()
    # The within-clade contrast figure is S26 since the 2026-10-19
    # renumbering (S32, then S30 from 2026-10-09). It must be a live manifest
    # line, not the commented demographic one it replaced. Assert the pairing
    # rather than that both strings occur somewhere: the looser form went on
    # passing after the figure moved to S26, because S30 was by then a live
    # line for a different figure.
    check(g, "S26 is the within-clade contrast manifest line",
          any(l.startswith("S26|") and "within_clade" in l
              for l in man.splitlines()))
    from persistence import figure_source
    # Deliberately NOT a manifest slot since 2026-09-27: assert the
    # withdrawal instead, so a silent re-addition is caught.
    check(g, "demographic figure is not a manifest slot",
          figure_source.slot_for(
              "results/combined/section3_health/supplement/"
              "demographic_associations_combined_10pct.png") is None)
    from persistence import figure_runner
    check(g, "demographics task registered and in the full set",
          "demographics" in figure_runner.TASKS and
          "demographics" in figure_runner.FULL_TASKS)
    rp = (ROOT / "tools/replot_from_data.py").read_text()
    # The demographic drawer must not be wired to a slot: it is not one.
    check(g, "the demographic drawer is not registered as a slot",
          '": _draw_demographic' not in rp and '"S32": _draw_S32' not in rp)
    check(g, "S26 has a drawer, so it can be redrawn from its pickle",
          '"S26": _draw_within_clade' in rp)
    check(g, "s3demo section key registered",
          "s3demo" in (ROOT / "persistence/__main__.py").read_text())

    # outputs from the demographic run
    expected = ["demographic_richness_gmw_10pct.csv",
                "demographic_per_taxon_gmw_10pct.csv",
                "demographic_per_taxon_prevmatched_10pct.csv",
                "demographic_table_s10a_unadjusted_10pct.csv",
                "demographic_table_s10b_energy_first_10pct.csv",
                "demographic_cohorts.csv", "demographic_field_coverage.csv"]
    missing = [f for f in expected if not (RESULTS / f).exists()]
    check(g, "all 7 demographic outputs written", not missing,
          missing or "all present")

    # the energy columns must not be NaN -- they are the table's primary ones
    p = RESULTS / "demographic_table_s10b_energy_first_10pct.csv"
    if p.exists():
        t = pd.read_csv(p)
        check(g, "S10b energy_distance populated (not NaN)",
              t["energy_distance"].notna().any(),
              f"{int(t['energy_distance'].notna().sum())}/{len(t)} rows")
        check(g, "S10b energy_p populated (not NaN)",
              t["energy_p"].notna().any(),
              f"{int(t['energy_p'].notna().sum())}/{len(t)} rows")

    # cohort table internal consistency
    cp = RESULTS / "demographic_cohorts.csv"
    if cp.exists():
        c = pd.read_csv(cp)
        check(g, "cohort table has 7 contrasts", len(c) == 7, len(c))
        bmi = c[c["Metadata_field"] == "BMI"]
        check(g, "both BMI contrasts draw the same pool",
              bmi["n_total"].nunique() == 1, sorted(bmi["n_total"].unique()))
        check(g, "BMI contrasts are not restricted to healthy",
              (~bmi["Restricted_to_healthy"]).all())
        check(g, "age and sex contrasts are restricted to healthy",
              c[c["Metadata_field"] != "BMI"]["Restricted_to_healthy"].all())


def audit_stratification():
    """Panel D must be study-stratified, and its p-values must be real."""
    g = "stratification"
    import numpy as np
    from persistence import config

    src = (ROOT / "persistence/section3_health.py").read_text()
    check(g, "richness comparison uses study strata",
          "_stratified_richness(" in src and "_get_study_strata" in src)
    check(g, "uncertainty is not a cluster bootstrap over studies",
          "robust variance estimation" in src and "_st.t.sf" in src)
    # The check names a function, so it has to look inside that function. It
    # used to read the whole file and assert `'"untested"' in txt`, which the
    # comment explaining the guard satisfied on its own -- and in
    # section3_health.py so did a docstring, a comment and the assignment
    # `star = "untested"`. Deleting either guard left it passing.
    for f, fn in (("figures/figure5.py", "draw_D"),
                  ("persistence/section3_health.py",
                   "_plot_richness_comparison")):
        _ok, _detail = _guards_against(f, fn, "untested")
        check(g, f"'untested' not drawn as a star in {fn}", _ok, _detail)

    # what the panel actually plots must follow the evidence, not a list
    f5 = (ROOT / "figures/figure5.py").read_text()
    check(g, "panel D drops single-study contrasts from the data",
          "MIN_STRATA_TO_PLOT" in f5 and "n_strata" in f5)
    # The set is empty now, and the check has to survive that. It used to
    # split on the literal "SKIP    = {" and read the names inside; once the
    # set was emptied to `SKIP: set = set()` there was no brace to split on,
    # so this raised IndexError and the whole audit stopped before reporting
    # anything -- a crash where a PASS was intended. Assert the thing that
    # matters instead: nothing is excluded by name, so eligibility is left to
    # MIN_STRATA_TO_PLOT above.
    _skip = re.search(r"^SKIP[^=\n]*=\s*(.+)$", f5, re.M)
    check(g, "no condition is hidden by a hardcoded SKIP set",
          bool(_skip) and _skip.group(1).strip() in ("set()", "{}"),
          _skip.group(1).strip() if _skip else "no SKIP assignment found")
    s32 = (ROOT / "figures/make_demographic_associations.py").read_text()
    check(g, "demographic panel A uses the same rule",
          "MIN_STRATA_TO_PLOT" in s32)
    # Panel labels report the analysed cohort (Tables S3/S10), not every
    # Industrial sample in the arm before study filtering.
    check(g, "panel D labels the study-stratified cohort",
          "_cohort_note" in f5 and "n_g2_strat" in f5)
    # The background curve shows the conditions Figure 5 shows; it used to
    # hardcode its own list.
    s3h = (ROOT / "persistence/section3_health.py").read_text()
    check(g, "background-curve conditions follow the Figure 5 study rule",
          "config.MIN_STRATA_TO_PLOT" in s3h
          and '"Glucose Intolerance", "Fatty Liver", "IBD")' not in s3h)
    # Repeat samples of one host: the run re-derives the Figure 2 ratio and the
    # list membership with one sample per host (Supplementary Text #5, #7C).
    _hs = ROOT / "results" / "combined" / "host_sensitivity"
    if (_hs / "list_host_sensitivity_summary.csv").exists():
        _ls = pd.read_csv(_hs / "list_host_sensitivity_summary.csv")
        check(g, "host sensitivity rebuilds section 2's lists from all samples",
              _ls["matches_section2_list"].astype(str).eq("True").all(),
              dict(zip(_ls["list"], _ls["matches_section2_list"])))
        check(g, "host sensitivity: richness ratio written",
              (_hs / "richness_host_sensitivity.csv").exists())
    else:
        check(g, "host sensitivity rebuilds section 2's lists from all samples", None,
              "results/combined/host_sensitivity/ not written yet")

    # Table S2 is built by the run, with hosts beside samples.
    from persistence import figure_runner as _fr
    check(g, "Table S2b is a final task of --figure full",
          "table_s2b" in _fr.FULL_TASKS and "table_s2b" in _fr.FINAL_TASKS)
    _s2b = ROOT / "results" / "table_s2b_cohorts.csv"
    if _s2b.exists():
        _t = pd.read_csv(_s2b)
        _h = _t[_t["Step"].astype(str).str.contains("Hosts")]
        check(g, "Table S2b reports hosts for every reference",
              len(_h) and _h.iloc[0, 1:].notna().all(),
              "no Hosts row" if not len(_h) else _h.iloc[0, 1:].to_dict())
    else:
        check(g, "Table S2b reports hosts for every reference", None,
              "results/table_s2b_cohorts.csv not written yet")

    for tag, fname in (("disease", "health_richness_gmw_10pct.csv"),
                       ("demographic", "demographic_richness_gmw_10pct.csv")):
        pth = RESULTS / fname
        if not pth.exists():
            check(g, f"{tag} richness table present", False, str(pth))
            continue
        d = pd.read_csv(pth)
        # Full column contract, not a sample of it. A crashed regeneration
        # leaves the previous CSV in place, so checking two or three columns
        # passed happily against a stale file written before the code changed
        # -- which is exactly what happened when a NameError killed the run.
        EXPECTED = {"Condition", "comp_type", "n_g1", "n_g2", "ess_g1",
                    "ess_g2", "mu_g1", "mu_g2", "diff", "se_g1", "se_g2",
                    "ci_lo", "ci_hi", "pval", "sig", "n_strata",
                    "n_g1_strat", "n_g2_strat", "frac_studies_same_sign",
                    "frac_strata_depth_weighted", "se_diff", "diff_pooled",
                    "mu_g1_pooled", "mu_g2_pooled"}
        missing = sorted(EXPECTED - set(d.columns))
        check(g, f"{tag}: every expected column present", not missing,
              missing or f"{len(d.columns)} columns")
        # and the invalid pooled inference must be gone
        gone = [c for c in ("pval_pooled", "ci_lo_pooled", "ci_hi_pooled")
                if c in d.columns]
        check(g, f"{tag}: unclustered pooled inference not recorded",
              not gone, gone or "absent")
        # every stratum depth-corrected, which was the whole point of the
        # binned GMW: previously 17-80% of strata were silently unadjusted
        if "frac_strata_depth_weighted" in d.columns:
            worst = float(d["frac_strata_depth_weighted"].min())
            check(g, f"{tag}: every stratum is depth-corrected",
                  worst >= 1.0, f"worst contrast {worst:.2f}")
        if "depth_method_mix" in d.columns:
            nones = [r.Condition for r in d.itertuples()
                     if "none" in str(r.depth_method_mix)]
            check(g, f"{tag}: no stratum falls back to unadjusted",
                  not nones, nones or "none do")

        # the file must postdate the last change to the module that writes it
        src_mt = _code_changed_at("persistence/section3_health.py")
        check(g, f"{tag}: table is not older than the code that writes it",
              pth.stat().st_mtime >= src_mt,
              "STALE - regenerate" if pth.stat().st_mtime < src_mt else "fresh")
        if "n_strata" not in d.columns:
            continue
        # a single-stratum contrast has no between-study variance: it must
        # carry no p-value rather than a fabricated one
        one = d[d["n_strata"] <= 1]
        check(g, f"{tag}: single-study rows carry no p-value",
              bool(one["pval"].isna().all()) if len(one) else True,
              f"{len(one)} single-study rows")
        check(g, f"{tag}: single-study rows marked untested",
              bool((one["sig"].astype(str) == "untested").all())
              if len(one) else True)
        # the old failure: every p pinned to the bootstrap floor
        tested = d[d["pval"].notna()]
        floor_hits = int((tested["pval"].round(6) == 0.002).sum())
        check(g, f"{tag}: p-values not pinned to a bootstrap floor",
              floor_hits == 0, f"{floor_hits} at exactly 0.002")
        check(g, f"{tag}: p-values span a real range",
              len(tested) == 0 or
              (tested["pval"].max() - tested["pval"].min()) > 0.1,
              f"{tested['pval'].min():.4f}-{tested['pval'].max():.4f}"
              if len(tested) else "none tested")


def audit_readiness():
    g = "readiness"
    r = subprocess.run([sys.executable, "-m", "compileall", "-q",
                        "persistence", "figures", "tools"],
                       cwd=ROOT, capture_output=True, text=True)
    check(g, "source tree compiles", r.returncode == 0,
          (r.stdout + r.stderr).strip()[:160] or "clean")

    r = subprocess.run([sys.executable, "-m", "persistence", "--help"],
                       cwd=ROOT, capture_output=True, text=True)
    check(g, "CLI parses", r.returncode == 0)
    check(g, "s3demo offered by --only", "s3demo" in r.stdout)

    r = subprocess.run([sys.executable,
                        "figures/make_demographic_associations.py", "--help"],
                       cwd=ROOT, capture_output=True, text=True)
    check(g, "demographic figure script runs standalone", r.returncode == 0)

    # Nothing the demographic pass could have clobbered.
    #
    # Both checks here were tautologies. The PNG was compared against a
    # sibling named demographic_prevmatched_..., which no code in the tree
    # ever writes, so `not demo.exists() or bytes differ` was always true;
    # and the pickle's was `ok = p.exists()` evaluated inside `if
    # p.exists():`. Neither could report a problem, while the risk they were
    # written for is real: _plot_prevalence_matched hardcodes S25's stem
    # ("health_prevmatched_{ref}_{spec_tag}") and the slot "S25", so a
    # demographic pass that ever reached it would overwrite S25's PNG and
    # republish S25's data object from the age/sex/BMI contrasts.
    #
    # What keeps the two passes apart is the call graph, not the filenames.
    _S3H = "persistence/section3_health.py"
    _reach = _reachable_functions(_S3H, "run_demographic_associations")
    if _reach is None:
        check(g, "S25's drawer is unreachable from the demographic pass",
              False, "run_demographic_associations() not found in " + _S3H)
    else:
        _wired = sorted(d for d in ("_plot_prevalence_matched",
                                    "run_prevmatch_plots")
                        if d in _reach)
        check(g, "S25's drawer is unreachable from the demographic pass",
              not _wired,
              f"reaches {_wired} -- they write health_prevmatched_* and "
              "emit slot S25" if _wired else
              f"{len(_reach)} functions reachable, none of them the drawer")

    # S25's turn. Its only drawing call is guarded by `not demographic`;
    # remove that guard and the demographic pass writes over the same
    # health_background_curve_* path, which is precisely the case comparing
    # against a "demographic_" filename could not see.
    _ok, _detail = _calls_guarded_by_not(
        _S3H, "_run_matched_background_curve", "_plot_background_curve",
        "demographic")
    check(g, "the background curve is drawn only by the health pass",
          _ok, _detail)

    # And the published object itself: the health and demographic contrast
    # labels are disjoint sets, so S25's own panels say which pass wrote it.
    _s25 = RESULTS / "figure_data" / "Figure_S25_data.pkl"
    if _s25.exists():
        try:
            import pickle
            from persistence.section3_health import _health_comparisons
            with _s25.open("rb") as _fh:
                _obj = pickle.load(_fh)
            _labels = set()
            for _fr in (_obj.get("panels") or {}).values():
                for _c in ("Condition", "condition"):
                    if hasattr(_fr, "columns") and _c in _fr.columns:
                        _labels |= set(_fr[_c].astype(str).unique())
            _demo = sorted(_labels & set(_health_comparisons(True)))
            _dis = sorted(_labels & set(_health_comparisons(False)))
            check(g, "S25's data object carries the disease contrasts",
                  not _demo and bool(_dis),
                  f"demographic contrasts present: {_demo}" if _demo else
                  (f"{len(_dis)} disease contrasts, no demographic one"
                   if _dis else f"no recognised contrast in {sorted(_labels)}"))
        except Exception as _e:
            check(g, "S25's data object carries the disease contrasts", False,
                  f"{type(_e).__name__}: {_e}")
    else:
        # Absent is not a pass: say so, the way the other on-disk checks do.
        check(g, "S25's data object carries the disease contrasts", None,
              f"{_s25} not written yet")

    # The demographic figure. Withdrawn from the publishable set on
    # 2026-09-27, so it is no longer slot S32 -- but the pipeline still
    # builds it and it must still publish a data object, or it stops being
    # reproducible the moment anyone wants it back.
    check(g, "demographic figure still rendered",
          (SUPP / "demographic_associations_combined_10pct.png").exists())
    # Its own pickle, by name: Figure_S32_data.pkl in this directory is the
    # within-clade figure's object now, so accepting it would pass this check
    # on the wrong file.
    check(g, "demographic figure still publishes a data object",
          (SUPP / "figure_data" / "Figure_demographic_data.pkl").exists())
    for _name, _p in (
            ("S26 (within-clade contrasts) rendered",
             SUPP / "health_within_clade_contrasts_combined_10pct.png"),
            ("S26 publishes a data object",
             SUPP / "figure_data" / "Figure_S26_data.pkl")):
        check(g, _name, _p.exists() if SUPP.exists() else None,
              "" if _p.exists() else "absent")
    # Not "S32 is absent" any more -- there is no S32: the within-clade
    # figure is S26 since 2026-10-19 (S30 from 2026-10-09 before that). What
    # must hold is that the DEMOGRAPHIC figure is not a numbered slot.
    _man_txt = (ROOT / "tools/additional_figures_manifest.txt").read_text()
    check(g, "demographic figure is out of the shipped manifest",
          not any(l.strip().startswith("S") and "demographic" in l
                  for l in _man_txt.splitlines()))
    # Check the OUTCOME, in the reviewer-facing view rather than in results/:
    # the demographic figure may be generated and may keep its data object, but
    # it must never appear in the shipped set. (This used to glob the results
    # supplement directory, where the figure legitimately exists, so it failed
    # as soon as a combined run was present.)
    _view = ROOT / "supplementary_figures"
    _leaked = sorted(p.name for p in _view.glob("*")
                     if p.name != "figure_data"
                     and "demographic" in p.name.lower()) if _view.exists() else []
    check(g, "no demographic artifact in supplementary_figures/",
          not _leaked,
          _leaked or (f"{len(list(_view.glob('S*')))} slots, none demographic"
                      if _view.exists() else "view not built"))
    src = (ROOT / "figures/make_demographic_associations.py").read_text()
    check(g, "demographic figure pins its rcParams",
          "rc_context(_RC)" in src and "font.sans-serif" in src
          and "xtick.major.pad" in src)

    # the disease outputs must still exist alongside the demographic ones
    for f in ("health_richness_gmw_10pct.csv", "health_per_taxon_gmw_10pct.csv",
              "health_per_taxon_prevmatched_10pct.csv"):
        check(g, f"disease output intact: {f}", (RESULTS / f).exists())

    # ── Figure 1 must report the analytic cohort, not a metadata approximation ──
    # Both pipeline exclusions are matrix-derived (>=30 Mb of *mapped* bases, and
    # a floor on SGBs detected), so neither is reproducible from Table M1.
    # figure1 therefore intersects against results/<ref>/cohort_samples.csv. The
    # old code cut on the metadata "Total bases" column instead -- sequenced
    # bases, ~100x larger -- and so reported ~200 samples more than any analysis
    # used. These checks are the outcome: the manifest exists, it agrees with the
    # canonical post-reinstatement audit, and figure1 no longer cuts on M1.
    from persistence import config as _cfgm
    _ref = _cfgm.MAIN_COHORT_REF
    _man = ROOT / "results" / _ref / "cohort_samples.csv"
    if check(g, f"cohort manifest written ({_ref})", _man.exists(),
             str(_man.relative_to(ROOT))):
        _mdf = pd.read_csv(_man)
        _ds = (ROOT / "results" / _ref / "depth_audit_post_reinstatement"
               / "depth_summary.csv")
        if _ds.exists():
            _sum = pd.read_csv(_ds)
            _hit = _sum.loc[_sum["group"].astype(str) == "ALL", "n"]
            _n = int(_hit.iloc[0]) if len(_hit) else -1
            check(g, "cohort manifest N == post-reinstatement audit N",
                  len(_mdf) == _n,
                  f"manifest {len(_mdf):,} vs audit {_n:,}")
        else:
            check(g, "cohort manifest N == post-reinstatement audit N", None,
                  "post-reinstatement audit not written yet")
        check(g, "cohort manifest carries sample IDs",
              "Chosen_match_ID" in _mdf.columns and _mdf["Chosen_match_ID"].is_unique,
              f"{len(_mdf):,} rows")

    # ── One sample per BioSample ──
    # A second library of one specimen is a re-measurement, not an observation.
    # Carter's MoBio/PheChl pairs slipped back in once already, when giving each
    # library its own run made them stop looking like secondaries.
    from persistence import config as _cfg
    _m1 = pd.read_csv(_cfg.META_FILE, low_memory=False,
                      usecols=["Chosen_match_ID", _cfg.BIOSAMPLE_COL])
    _m1["Chosen_match_ID"] = _m1["Chosen_match_ID"].astype(str).str.strip()
    _m1 = _m1.drop_duplicates("Chosen_match_ID").set_index("Chosen_match_ID")
    for _r in ("UHGG", "lev", "combined", "chimp"):
        _mf = ROOT / "results" / _r / "cohort_samples.csv"
        if not _mf.exists():
            check(g, f"one sample per BioSample ({_r})", None, "no cohort manifest yet")
            continue
        _ids = pd.read_csv(_mf)["Chosen_match_ID"].astype(str)
        _sub = _m1.reindex(_ids)
        _bs = _sub[_cfg.BIOSAMPLE_COL].astype(str).str.strip()
        _nd = int(_bs.duplicated(keep=False).sum())
        check(g, f"one sample per BioSample ({_r})", _nd == 0,
              f"{_nd} cohort rows share a BioSample")

    # Figure 1 exists and was drawn from the current cohort. A run once finished
    # "cleanly" with no Figure 1: the basemap lookup failed inside a task the
    # runner tolerates. Absent or older than the main cohort manifest = stale.
    _fig1 = ROOT / "main_text_figures" / "figure1.png"
    _man1 = _man
    if _man1.exists():
        check(g, "Figure 1 drawn after the cohort was written",
              _fig1.exists() and _fig1.stat().st_mtime >= _man1.stat().st_mtime,
              "missing" if not _fig1.exists() else "older than results/UHGG/cohort_samples.csv")
    _status = ROOT / "results" / "figure_task_status.csv"
    if _status.exists():
        _st = pd.read_csv(_status)
        _bad = _st[_st["status"] != "ok"]
        check(g, "every figure task produced output", _bad.empty,
              "; ".join(f"{r.task}: {r.status}" for r in _bad.itertuples()) or "all ok")
    else:
        check(g, "every figure task produced output", None,
              "results/figure_task_status.csv not written yet")
    _mm = (ROOT / "figures/make_metadata_summary.py").read_text()
    check(g, "basemap is the bundled Natural Earth copy",
          (ROOT / "figures/basemap/naturalearth_lowres/naturalearth_lowres.shp").exists()
          and '"basemap"' in _mm)

    _f1 = (ROOT / "figures/make_metadata_summary.py").read_text()
    check(g, "figure1 intersects the cohort manifest",
          "_apply_pipeline_cohort" in _f1 and "read_cohort_manifest" in _f1)
    # The sequenced-bases floor still exists, but only inside the fallback -- so
    # assert the fallback is loud rather than that the expression is gone.
    check(g, "figure1's metadata-only fallback warns loudly",
          "no cohort manifest at" in _f1 and "OVER-COUNTS" in _f1)
    check(g, "load_and_preprocess applies the cohort",
          "meta = _apply_pipeline_cohort(meta, cohort_ref)" in _f1)


def audit_background_method():
    """The corrected prevalence background (2026-10-06).

    Figure 5E and S25-S32 all rest on two quantities: a study-stratified
    case-control shift, and a background curve fitted on taxa of the same
    commonness. These checks assert the pipeline is wired to those and not to
    the importance-weighted background they replaced -- in the code, in the
    manifest, and in what a finished run wrote.
    """
    g = "corrected prevalence background"
    from persistence import config as cfg
    from persistence import health_cmh, health_background, section3_health

    check(g, "HEALTH_BG_METHOD is the corrected curve",
          getattr(cfg, "HEALTH_BG_METHOD", "") == "curve",
          getattr(cfg, "HEALTH_BG_METHOD", "unset"))
    check(g, "the stratified estimator exists",
          callable(getattr(health_cmh, "stratified_shifts_per_taxon", None)))
    check(g, "both per-taxon estimators share one set of stratum weights",
          "build_weighted_strata(strata, meta, min_stratum)" in
          (ROOT / "persistence/health_cmh.py").read_text())
    check(g, "the shift accessor prefers the stratified shift",
          section3_health.SHIFT_COLUMNS[0] == "rd_mh")
    check(g, "the weight accessor prefers the matched weight",
          section3_health.BG_WEIGHT_COLUMNS[0] == "bg_weight")

    # The IW path is kept reachable on purpose, to reproduce the 29 Sep
    # figures -- but only from the one function that config can select.
    s3 = (ROOT / "persistence/section3_health.py").read_text()
    # Count code references, by the quoted column name -- prose that explains
    # the switch mentions it too. Five are expected: the accessor tuple, the
    # label helper (which only names the background a frame carries), the S25
    # redraw of a published object, the IW diagnostic, and the IW writer.
    # Anything beyond that is a consumer computing with the old weights.
    _iw_lines = [l for l in s3.splitlines()
                 if '"importance_weight"' in l and not l.strip().startswith("#")]
    check(g, "importance_weight is read only by the superseded IW path",
          len(_iw_lines) <= 5, f"{len(_iw_lines)} code references")
    _cfg_txt = (ROOT / "persistence/config.py").read_text()
    check(g, "the IW background is still reproducible",
          "_run_prevalence_matched_background_iw" in s3
          and 'HEALTH_BG_METHOD' in _cfg_txt and '"iw"' in _cfg_txt)

    # No floor, clip or rounding survives in the method that is now in use.
    hb = (ROOT / "persistence/health_background.py").read_text()
    check(g, "the matched background rounds nothing away",
          "np.floor" not in hb and "match_counts" not in hb)
    check(g, "the bootstrap refits the curve",
          "curves(bx[ib], by[ib]" in hb)
    check(g, "sigma is floored at each taxon's own SE",
          "np.maximum(sig, lse)" in hb)

    man = (ROOT / "tools/additional_figures_manifest.txt").read_text()
    check(g, "S25 is the background-curve figure, base lists",
          "S25|combined|section3_health/supplement/"
          "health_background_curve_base_combined_10pct" in man)
    # The renumbering: 31 slots, contiguous, no S32. The bound was left at 30
    # when S31 (the analytic roadmap) was added, so this check failed on a
    # correct manifest until 2026-10-19. The 2026-10-19 rotation does not
    # change the count -- it permutes slots within S5..S14 and S26..S30 -- so
    # a contiguous S1..S31 is exactly what it must still produce.
    import re as _re
    _slots = sorted(int(m) for m in
                    _re.findall(r"^S(\d+)\|", man, flags=_re.M))
    check(g, "31 contiguous Additional Figure slots",
          _slots == list(range(1, 32)),
          f"{len(_slots)} slots, {_slots[-1] if _slots else 0} highest")

    # Slots the pipeline hard-codes, cross-checked against the manifest. The
    # 2026-10-19 rotation moved figures between slot numbers, and a literal
    # left behind does not fail loudly -- it writes a data object under
    # another figure's name, which is how a renumbering silently attaches the
    # wrong table to a figure. Each entry is (what it is, the slot the code
    # hard-codes, the manifest stem that slot must carry).
    from tools import figure_sets as _fs
    _by_stem = {stem: slot for slot, _r, stem in _fs.manifest()}
    from persistence import health_driver_family_panels as _hdfp
    _hard = [
        ("within-clade emit, section3_health",
         "S26", "section3_health/supplement/"
                "health_within_clade_contrasts_combined_10pct"),
        ("Carter weight diagnostics emit, carter_analysis",
         "S8", "carter_analysis/carter_gmw_weight_diagnostics_10pct"),
        ("GMW richness lifestyle_all, section3_health",
         "S11", "section3_health/main/"
                "health_richness_gmw_combined_lifestyle_all_10pct"),
        ("GMW richness lifestyle3, section3_health",
         "S12", "section3_health/main/"
                "health_richness_gmw_combined_lifestyle3_10pct"),
        ("driver panel, VANISH family",
         _hdfp.FAMILIES["VANISH family"][0],
         "section3_health/main/"
         "health_taxon_drivers_VANISH_family_combined_10pct"),
        ("driver panel, BloSSUM family",
         _hdfp.FAMILIES["BloSSUM family"][0],
         "section3_health/main/"
         "health_taxon_drivers_BloSSUM_family_combined_10pct"),
        ("driver panel, VANISH original",
         _hdfp.FAMILIES["VANISH"][0],
         "section3_health/supplement/"
         "health_taxon_drivers_VANISH_combined_10pct"),
        ("driver panel, BloSSUM original",
         _hdfp.FAMILIES["BloSSUM"][0],
         "section3_health/supplement/"
         "health_taxon_drivers_BloSSUM_combined_10pct"),
    ]
    _wrong = [f"{what}: code says {slot}, manifest says {_by_stem.get(stem)}"
              for what, slot, stem in _hard if _by_stem.get(stem) != slot]
    check(g, "every hard-coded slot agrees with the manifest",
          not _wrong, "; ".join(_wrong) or f"{len(_hard)} checked")
    # And that each driver panel is written into the directory the manifest
    # puts it in, which SLOT_DIR duplicates.
    _dirmis = [f"{s} -> {d}, manifest has {_by_stem}"
               for s, d in _hdfp.SLOT_DIR.items()
               if not any(st.split("/")[1] == d and sl == s
                          for st, sl in _by_stem.items()
                          if st.startswith("section3_health/"))]
    check(g, "SLOT_DIR matches the manifest's directories",
          not _dirmis, "; ".join(x[:60] for x in _dirmis)
          or f"{len(_hdfp.SLOT_DIR)} checked")

    # What a finished run must have written.
    for f, what in (
            ("health_per_taxon_stratified_10pct.csv", "stratified shifts"),
            ("health_background_curve_summary_10pct.csv", "curve summary"),
            ("health_background_curve_taxa_10pct.csv", "per-taxon z"),
            ("health_clade_within_10pct.csv", "within-clade contrasts"),
            ("../supplement/health_background_curve_base_combined_10pct.png",
             "S25"),
    ):
        p_ = RESULTS / f
        check(g, f"run wrote the {what}", p_.exists() if RESULTS.exists() else None,
              "" if p_.exists() else "absent")

    # And that the figures quote the corrected statistic.
    f5 = (ROOT / "figures/figure5.py").read_text()
    check(g, "Figure 5E prints the mean distance from the background",
          "mean-z statistics" in f5 and "curve_row" in f5)
    # Panel E is the background curve since 2026-10-07, and S25 is the same
    # figure for the base lists. Two implementations of this comparison
    # drifted apart in type size, line weight and annotation once already,
    # so the check is that there is exactly one cell renderer behind both.
    pu = (ROOT / "persistence/plot_utils.py").read_text()
    check(g, "one grid renderer behind Figure 5E and S25",
          "def draw_background_curve_grid" in pu
          and "draw_background_curve_grid" in f5
          and "draw_background_curve_grid" in s3)
    # The sizes are declared once, in printed points, and each caller passes
    # the magnification of its own canvas. This is what makes the two match:
    # Figure 5 is authored 24 in wide and reduced on save, S25 at the column
    # width, so any size set in a call site means two different things.
    check(g, "panel sizes are declared once in printed points",
          "def background_curve_style" in pu
          and "background_curve_style(1.0)" in s3
          and "background_curve_style(FIG_W / 7.24)" in f5)
    check(g, "S25 opts out of the shared type scale, as Figure 5 does",
          "_skip_type_normalise = True" in s3)
    check(g, "the density form of panel E is gone",
          "def _plot_prevmatched_distributions" not in s3
          and "health_prevdist" not in man)
    check(g, "Figure 5E carries the background-curve panels in its data object",
          "panel_e_curve_taxa" in f5 and "panel_e_all_taxa" in f5)
    # And that a completed run actually took that path. If the curve tables
    # are absent at figure time, build_figure5 warns loudly and then draws
    # the SUPERSEDED density form -- a warning in a long log is easy to miss,
    # so the published artefact is checked instead.
    _f5pkl = ROOT / "main_text_figures/figure_data/Figure_F5_data.pkl"
    if _f5pkl.exists():
        try:
            import pickle
            _panels = pickle.load(open(_f5pkl, "rb")).get("panel_order", [])
            check(g, "the run's Figure 5 used the background curve, not the "
                     "density fallback",
                  "panel_e_curve_taxa" in _panels and
                  "panel_e_all_taxa" in _panels,
                  f"panel_order = {list(_panels)}")
        except Exception as _exc:
            check(g, "Figure 5's data object is readable", False, repr(_exc))
    else:
        check(g, "the run's Figure 5 used the background curve, not the "
                 "density fallback", None, "absent")
    # Science: lettering about 7 pt, never below 5, at final size.
    check(g, "Figure 5's type clears the 5 pt floor at final size",
          "FS      = 23" in f5 or "FS = 23" in f5,
          "FS sets the printed size: FS * 7.24 / 24")
    t6 = (ROOT / "persistence/table_s7.py").read_text()
    check(g, "Table S6B leads with mean z",
          '"mean_z":' in t6 and '"z_CI_lo":' in t6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    m = audit_metadata()
    audit_additions(m)
    audit_stratification()
    audit_background_method()
    audit_readiness()

    width = max(len(n) for _g, n, _o, _d in _rows) + 2
    cur = None
    npass = nfail = 0
    for grp, name, ok, detail in _rows:
        if grp != cur:
            print(f"\n  ── {grp} " + "─" * (width + 18 - len(grp)))
            cur = grp
        mark = "PASS" if ok else ("----" if ok is None else "FAIL")
        if ok:
            npass += 1
        elif ok is False:
            nfail += 1
        line = f"  {mark}  {name:<{width}}"
        if detail and (a.verbose or not ok):
            line += f"  {detail}"
        print(line)
    print(f"\n  {npass} passed, {nfail} failed, {len(_rows)} checks\n")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
