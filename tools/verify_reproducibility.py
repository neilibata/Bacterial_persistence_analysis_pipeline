#!/usr/bin/env python3
"""Check that this pipeline reproduces its published numbers on your machine.

Runs in tiers, cheapest first. Each tier makes a strictly stronger claim:

  tier 0  inputs        the committed inputs and the downloaded data are the
                        ones the paper used                      (seconds, no compute)
  tier 1  derived       Metadata Tables M1/M2/M3 rebuild identically from those
                        inputs                                   (~1 min)
  tier 2  determinism   a SEEDED, STOCHASTIC pipeline section re-runs to the
                        same numbers       (~20 min, needs the matrices)
  tier 3  full          the whole pipeline reproduces            (hours; not automated here)

    python tools/verify_reproducibility.py              # tiers 0-1
    python tools/verify_reproducibility.py --tier 2     # + re-run a section
    python tools/verify_reproducibility.py --update     # author: rewrite the manifest

What the deposited record alone can check. Tiers 0 and 1 run from the input
record that ``download_data.sh`` fetches, and are the useful ones for a
reader. The ``figures``, ``supp_figures`` and ``outputs`` groups compare files
at the paths the pipeline writes them to, and the results record deposits
figures only as its flat, renumbered ``S01_*.png`` set -- it carries tables
and data objects, not PNGs at their source paths. So those groups report
``missing`` against an unpacked deposit, which is not a failure of the data:
they need a local figure run, or the deposited figures copied back to the
paths ``tools/additional_figures_manifest.txt`` maps each slot to.

Two hashes are recorded per file. ``sha256`` is over the raw bytes and is the
strictest check. ``value_hash`` is over the parsed table with floats rounded to
``SIGFIGS`` significant digits, so a platform that formats floats differently
still verifies — a file that fails ``sha256`` but passes ``value_hash`` is
numerically identical, which is usually what a reader actually cares about.

Exit codes, so a caller can tell the three outcomes apart rather than reading
"PASS" off a run that checked almost nothing:

  0  every recorded file in every group it was asked to check was compared and
     agreed
  1  FAIL        at least one file differs in value, or the manifest records a
                 group that no pass compares
  2  INCOMPLETE  files are missing, so their claim is untested
  3  UNVERIFIED  files differ and the difference could not be adjudicated, or a
                 tier could not run. Not a pass: nothing was established.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# Run as a script, sys.path[0] is tools/, so `from tools import figure_sets`
# and `from persistence import config` both fail. The group builders below
# used to swallow that and return empty lists, and the manifest silently
# covered 52 artifacts instead of 113 -- a shortfall invisible in its own
# output. Put the repo root on the path before anything imports from it.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / "verification_manifest.json"
#: Files larger than this get sha256 only, no parsed value hash.
VALUE_HASH_MAX_BYTES = 200 * 1024 * 1024
SIGFIGS = 8

# ── what gets verified ───────────────────────────────────────────────────────
# Committed inputs: must match byte-for-byte, else everything downstream is moot.
INPUTS = [
    "Vanish_list.txt", "Blossum_list.txt", "Vanish_phylo.txt", "Blossom_phylo.txt",
    "all_SGB_lengths_UHGG.txt", "all_SGB_lengths_lev.txt",
    "all_SGB_lengths_combined.txt", "all_SGB_lengths_chimp.txt",
    "combined_sgb_metadata.xlsx",
]

# Rebuilt by tier 1 from the inputs above (matrix headers only — no full load).
# Data S2 is here rather than among the inputs because the workbook is a
# rendering of final_run/metadata_M1.csv, not a separate source: the CSV is
# what the pipeline reads and what the input record deposits. Having it
# rebuilt and compared is the only thing that stops the submitted workbook
# drifting from that table, which is exactly what happened to the copy this
# replaced -- 18,303 rows by 110 columns against the current 18,299 by 118.
DERIVED = ["metadata_M2.xlsx", "metadata_M3.xlsx",
           "Data_S2_metadata_M1.xlsx"]

# Main-text figures. PDFs are compared with their embedded dates stripped, so a
# re-render of an unchanged figure verifies instead of always looking different.
FIGURES = [f"main_text_figures/figure{i}.pdf" for i in range(1, 6)]

# Manuscript-critical pipeline outputs.
OUTPUTS = [
    "results/UHGG/stats_report.csv",
    "results/combined/stats_report.csv",
    "results/UHGG/table_s1a_study_summary_formatted.csv",
    "results/combined/section2_differential/supplement/table_s5_overlap_summary.csv",
    "results/combined/section3_health/main/health_per_condition_cmh_10pct.csv",
    "results/combined/section3_health/main/health_binomial_tests_10pct.csv",
    "results/chimp/negative_control_asymptotes.csv",
]
for _ref in ("UHGG", "combined"):
    for _l in ("vanish_plus_ids", "blossum_plus_ids", "vanishH_ids", "blossumH_ids"):
        OUTPUTS.append(f"results/{_ref}/section2_differential/main/{_l}.txt")

# Tier 2 re-runs this and re-checks the outputs it writes.
#
# It used to re-run `--only table_s1a`, which contains no stochastic code at
# all: no default_rng, no bootstrap, no permutation. The tier advertises
# "determinism", so re-running a stage with nothing to seed would have
# reported success unchanged even if every bootstrap and permutation in the
# pipeline were unseeded. s1_axis_stats is the cheapest stage whose numbers
# come out of an RNG -- the Jonckheere-Terpstra and PERMANOVA permutation
# nulls, seeded from config -- so re-running it is the claim the tier makes.
# TIER2_MODULES is asserted below to contain that randomness, which is what
# stops the tier quietly reverting to a deterministic stage again.
TIER2_STAGE = "s1_axis_stats"
TIER2_CMD = [sys.executable, "-m", "persistence", "--ref", "UHGG",
             "--only", TIER2_STAGE]
TIER2_CHECKS = ["results/UHGG/section1_diversity/main/axis_angle_v2_stats_10pct.csv"]
#: The modules the tier-2 stage runs. Scanned for RNG use before the tier is
#: allowed to claim it checked determinism.
TIER2_MODULES = ["persistence/axis_angle_stats.py"]
#: Tokens that mark a line as drawing on the pipeline's randomness.
_RNG_TOKENS = ("default_rng", "rng.permutation", "rng.choice", "rng.integers",
               "rng.normal", "permutation(", "bootstrap", "shuffle",
               "random_state")


# ── hashing ──────────────────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _round_sig(x, sig=SIGFIGS):
    """Round elementwise to `sig` significant figures.

    np.round takes a scalar `decimals`, so the per-element decimal count has to
    be applied as a scale factor rather than passed in directly.

    A value small enough that the scale factor would overflow is left at full
    precision rather than rounded. At sig=8 that is anything below about
    1e-302, which this analysis does reach: the register records P values of
    `< 1e-300`. Scaling them overflowed to inf and the value became nan, so
    two different tiny numbers hashed the same -- and `value_hash` is the
    adjudicator that decides whether a file failing `sha256` is nonetheless
    numerically identical. Leaving them unrounded can only make that
    comparison stricter.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    nz = np.isfinite(x) & (x != 0)
    if np.any(nz):
        with np.errstate(over="ignore", invalid="ignore"):
            factor = 10.0 ** (sig - 1 - np.floor(np.log10(np.abs(x[nz]))))
        vals = x[nz].copy()
        usable = np.isfinite(factor) & (factor != 0)
        if np.any(usable):
            with np.errstate(over="ignore", invalid="ignore"):
                vals[usable] = (np.round(vals[usable] * factor[usable])
                                / factor[usable])
        out[nz] = vals
    out[~np.isfinite(x)] = np.nan
    return out


def _frame_hash(df: pd.DataFrame) -> str:
    """Hash a table by content, insensitive to float formatting."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = _round_sig(d[c].to_numpy())
    payload = "|".join(map(str, d.columns)) + "\n" + d.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


_PDF_DATE = re.compile(rb"/(?:CreationDate|ModDate)\s*\([^)]*\)")


def value_hash(path: Path) -> str | None:
    """Content hash, insensitive to formatting/metadata; None if unparseable.

    Figures need this as much as tables do: a PDF embeds its creation date, so
    re-rendering an unchanged figure always produces different bytes. Stripping
    the date fields makes "did this figure actually change?" answerable. PNGs
    are hashed on pixels, which ignores their metadata chunks.
    """
    # Above this, only the raw sha256 is recorded. The value hash exists so a
    # re-formatted table still verifies, which is worth a parse for a results
    # table of a few thousand rows and pointless for a 2.8 GB input matrix:
    # parsing 24,173 x 5,081 cells and rounding each one ran for 42 minutes of
    # CPU without finishing, and an external input is verified byte-for-byte
    # anyway -- nothing re-formats it.
    if path.stat().st_size > VALUE_HASH_MAX_BYTES:
        return None
    try:
        if path.suffix == ".csv":
            return _frame_hash(pd.read_csv(path, low_memory=False))
        if path.suffix == ".xlsx":
            sheets = pd.read_excel(path, sheet_name=None)
            return hashlib.sha256("".join(
                f"{n}:{_frame_hash(df)}" for n, df in sorted(sheets.items())).encode()).hexdigest()
        if path.suffix == ".txt":
            lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
            return hashlib.sha256("\n".join(lines).encode()).hexdigest()
        if path.suffix == ".pdf":
            return hashlib.sha256(_PDF_DATE.sub(b"", path.read_bytes())).hexdigest()
        if path.suffix == ".pkl":
            # Figure data objects carry a "created" timestamp, so their raw
            # bytes change on every regeneration even when the data does not
            # -- exactly the case value_hash exists for. Hash the panels and
            # the extras, in a fixed order, and leave the timestamp out.
            import pickle
            with path.open("rb") as fh:
                obj = pickle.load(fh)
            parts = []
            for name in sorted((obj.get("panels") or {})):
                parts.append(f"{name}:{_frame_hash(obj['panels'][name])}")
            ex = obj.get("extras") or {}
            for name in sorted(ex):
                parts.append(f"x/{name}:{repr(ex[name])}")
            parts.append(f"slot:{obj.get('slot')}")
            return hashlib.sha256("|".join(parts).encode()).hexdigest()
        if path.suffix == ".png":
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(path) as im:
                return hashlib.sha256(im.convert("RGBA").tobytes()).hexdigest()
    except Exception:
        return None
    return None


def describe(rel: str) -> dict | None:
    p = ROOT / rel
    if not p.exists():
        return None
    return {"path": rel, "bytes": p.stat().st_size,
            "sha256": sha256(p), "value_hash": value_hash(p)}


# ── manifest ─────────────────────────────────────────────────────────────────

# ── External inputs: hashed here, distributed elsewhere ──────────────────────
# The multi-GB matrices and the per-sample metadata are not in git -- the
# matrices for size, M1 because it is human-subject data. Hashing them anyway
# is the point: a reader who obtains them from the data deposit can confirm
# byte-for-byte that they hold the same files this pipeline ran on, which is
# what closes the chain at the input end. Recording the hash distributes
# nothing.
def _external_inputs():
    rels = []
    for pat in ("combined_coverage_matrix_*.csv", "combined_depth_matrix_*.csv"):
        rels += sorted(str(q.relative_to(ROOT)) for q in ROOT.glob(pat))
    try:
        from persistence import config
        if (ROOT / config.META_FILE).exists():
            rels.append(config.META_FILE)
    except Exception as exc:                 # pragma: no cover
        print(f"  [WARN] config (M1 will be omitted): {exc}")
    return rels


# ── Figure data objects ──────────────────────────────────────────────────────
# One pickle per shipped figure. These matter more than the images: a figure
# can be redrawn from its data object, so hashing the object verifies the
# numbers behind the picture rather than the rendering of it.
def _data_objects():
    try:
        from tools import figure_sets
    except Exception as exc:                     # pragma: no cover
        print(f"  [WARN] data objects omitted from the manifest: {exc}")
        return []
    rels = []
    for slot, ref, stem in figure_sets.manifest():
        q = (ROOT / "results" / ref / Path(stem).parent / "figure_data"
             / f"Figure_{slot}_data.pkl")
        if q.exists():
            rels.append(str(q.relative_to(ROOT)))
    for n in range(1, 6):
        q = ROOT / "main_text_figures" / "figure_data" / f"Figure_F{n}_data.pkl"
        if q.exists():
            rels.append(str(q.relative_to(ROOT)))
    return rels


def _supp_figures():
    try:
        from tools import figure_sets
    except Exception as exc:                     # pragma: no cover
        print(f"  [WARN] supp figures omitted from the manifest: {exc}")
        return []
    return [f"results/{ref}/{stem}.png" for _s, ref, stem
            in figure_sets.manifest()
            if (ROOT / f"results/{ref}/{stem}.png").exists()]


# The stratified richness tables and the demographic analysis, added
# 2026-09-25. Section 3 rewrites all of these, so they belong in the same
# chain as the rest of the manuscript-critical output.
for _sp in ("10pct",):
    for _f in ("health_richness_gmw", "health_per_taxon_gmw",
               "health_per_taxon_prevmatched", "demographic_richness_gmw",
               "demographic_per_taxon_gmw", "demographic_per_taxon_prevmatched",
               "demographic_table_s10a_unadjusted",
               "demographic_table_s10b_energy_first"):
        OUTPUTS.append(
            f"results/combined/section3_health/main/{_f}_{_sp}.csv")
for _f in ("demographic_cohorts.csv", "demographic_field_coverage.csv"):
    OUTPUTS.append(f"results/combined/section3_health/main/{_f}")

# The tier-2 target has to be a recorded output or there is nothing to compare
# the re-run against; the tier reports "cannot verify" if it is absent from the
# manifest, which is what an author who forgets --update after moving the tier
# would otherwise never see.
OUTPUTS.extend(c for c in TIER2_CHECKS if c not in OUTPUTS)

GROUPS = {"inputs": INPUTS, "external_inputs": _external_inputs(),
          "derived": DERIVED,
          "outputs": OUTPUTS, "figures": FIGURES,
          "supp_figures": _supp_figures(), "data_objects": _data_objects()}

# ── which pass compares which group ──────────────────────────────────────────
# main() used to call check_group for inputs/derived/outputs/figures
# only. external_inputs, supp_figures and data_objects were recorded and never
# compared -- 74 of 113 files, including the five downloaded matrices, M1,
# every supplementary figure and every figure data object -- so a reader with
# different matrices, a different M1 or regenerated supplementary numbers still
# read "PASS" off this script. Declaring the passes here and asserting below
# that every recorded group is named in one of them turns that silent omission
# into a reported failure.
TIER0_GROUPS = ("inputs", "external_inputs")
TIER1_GROUPS = ("derived",)
ONDISK_GROUPS = ("outputs", "figures", "supp_figures", "data_objects")


def undeclared_groups(man: dict) -> list[str]:
    """Groups that exist but that no pass in main() compares."""
    declared = set(TIER0_GROUPS) | set(TIER1_GROUPS) | set(ONDISK_GROUPS)
    recorded = {g for g in man if not g.startswith("_") and g != "sigfigs"}
    return sorted((recorded | set(GROUPS)) - declared)


def stochastic_evidence(rels: list[str]) -> list[str]:
    """Lines in those modules that draw on the pipeline's randomness.

    Empty means the tier-2 stage has nothing to seed, so re-running it says
    nothing about determinism -- the defect this list exists to catch.
    """
    hits = []
    for rel in rels:
        p = ROOT / rel
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            code = line.split("#")[0]
            if any(t in code for t in _RNG_TOKENS):
                hits.append(f"{rel}:{i}  {code.strip()[:70]}")
    return hits


def build_manifest() -> dict:
    man = {"_generated_on": {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip() or None,
    }, "sigfigs": SIGFIGS}
    for grp, rels in GROUPS.items():
        entries = {}
        for rel in rels:
            e = describe(rel)
            if e is None:
                print(f"  ! missing, not recorded: {rel}")
            else:
                entries[rel] = e
        man[grp] = entries
    return man


def check_group(man: dict, grp: str, only: list[str] | None = None,
                at: Path | None = None) -> tuple[int, int, int, int, int]:
    """-> (exact, numeric-only, mismatched, missing, unverifiable)

    .xlsx files embed a build timestamp in their zip container, so their bytes
    change on every rebuild even when every cell is identical. For those the
    value hash, not sha256, is the meaningful check.
    """
    entries = man.get(grp, {})
    if only is not None:
        entries = {k: v for k, v in entries.items() if k in only}
    if not entries:
        # Returning silent zeros is how a pass that compared nothing looked
        # like a pass that compared everything. Say so instead.
        if only is not None:
            print(f"\n  [{grp}]  none of the {len(only)} requested path(s) is "
                  "recorded in the manifest — nothing compared")
        else:
            print(f"\n  [{grp}]  nothing recorded in the manifest — "
                  "nothing compared")
        return (0, 0, 0, 0, 0)
    exact = numeric = bad = missing = unver = 0
    print(f"\n  [{grp}]  {len(entries)} file(s)")
    for rel, want in sorted(entries.items()):
        # `at` redirects the lookup to a scratch build, leaving the repo untouched.
        p = (at / Path(rel).name) if at is not None else (ROOT / rel)
        if not p.exists():
            print(f"    MISSING      {rel}")
            missing += 1
            continue
        if sha256(p) == want["sha256"]:
            exact += 1
            continue
        got_val, want_val = value_hash(p), want.get("value_hash")
        if got_val is not None and want_val is not None:
            if got_val == want_val:
                note = {
                    ".xlsx": "rebuilt; every cell identical",
                    ".pdf": "re-rendered; identical once the embedded date is stripped",
                    ".png": "re-rendered; every pixel identical",
                }.get(p.suffix, f"bytes differ, values identical to {SIGFIGS} s.f.")
                print(f"    numeric      {rel}  ({note})")
                numeric += 1
            else:
                print(f"    MISMATCH     {rel}  (values differ)")
                bad += 1
        else:
            # Bytes differ and the difference cannot be adjudicated. This is
            # NOT a pass: a results table that was truncated or corrupted
            # since the manifest was written lands here (bytes differ, pandas
            # can no longer parse it), and so does any format value_hash does
            # not handle. main() counts it towards exit status 3.
            why = ("no value hash was recorded (file over "
                   f"{VALUE_HASH_MAX_BYTES // (1024*1024)} MB, or the format "
                   "was unparseable when the manifest was built)"
                   if want_val is None else
                   "the file on disk could not be parsed to compare values")
            print(f"    UNVERIFIED   {rel}  (bytes differ; {why})")
            unver += 1
    return (exact, numeric, bad, missing, unver)


# ── tiers ────────────────────────────────────────────────────────────────────

def run(cmd: list[str], label: str) -> bool:
    print(f"\n  running: {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ! {label} failed (exit {r.returncode})")
        print("   ", (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else "")
        return False
    return True


def main():
    global MANIFEST
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", type=int, default=1, choices=[0, 1, 2],
                    help="highest tier to run (default 1)")
    ap.add_argument("--update", action="store_true",
                    help="author-only: regenerate verification_manifest.json")
    # A verifier has to be testable against a manifest whose answers are known,
    # or the only way to find out whether it can fail is to damage the repo.
    ap.add_argument("--manifest", default=None,
                    help="read (or with --update write) this manifest instead "
                         "of verification_manifest.json")
    args = ap.parse_args()
    if args.manifest:
        MANIFEST = Path(args.manifest).expanduser()

    if args.update:
        man = build_manifest()
        MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
        n = sum(len(man[g]) for g in GROUPS)
        print(f"\nWrote {MANIFEST}  ({n} files across {len(GROUPS)} groups)")
        return

    if not MANIFEST.exists():
        sys.exit(f"no manifest at {MANIFEST} — generate it with --update")
    man = json.loads(MANIFEST.read_text())
    gen = man.get("_generated_on", {})
    print(f"Manifest generated on python {gen.get('python')} / numpy {gen.get('numpy')} / "
          f"pandas {gen.get('pandas')}\n  commit {gen.get('git_commit')}")
    print(f"This machine:      python {platform.python_version()} / numpy {np.__version__} / "
          f"pandas {pd.__version__}")

    tot = [0, 0, 0, 0, 0]
    # Groups this invocation actually compared, and the reason any was not, so
    # the summary can state coverage instead of implying it.
    visited: list[str] = []
    skipped: dict[str, str] = {}
    cannot: list[str] = []          # a claim this run tried and failed to establish
    scope: list[str] = []           # a claim this run was not asked to make
    t2tot = [0, 0, 0, 0, 0]         # tier 2 re-checks a group already counted

    # A group recorded in the manifest that no pass names is unverifiable by
    # construction, and no amount of output will show it. Fail on it here.
    stray = undeclared_groups(man)
    if stray:
        print("\nFAIL — the manifest records group(s) that no pass compares: "
              + ", ".join(stray)
              + "\n  Add them to TIER0_GROUPS / TIER1_GROUPS / ONDISK_GROUPS.")
        sys.exit(1)

    print("\n── tier 0: committed inputs and downloaded inputs ──")
    for grp in TIER0_GROUPS:
        tot = [a + b for a, b in zip(tot, check_group(man, grp))]
        visited.append(grp)

    if args.tier >= 1:
        print("\n── tier 1: Metadata Tables M1/M2/M3 rebuild from those inputs ──")
        # Build into a scratch directory: verifying must never modify the repo,
        # or every reader who runs this ends up with a dirty working tree.
        with tempfile.TemporaryDirectory(prefix="verify-m2m3-") as tmp:
            built = run([sys.executable, "tools/build_metadata_tables.py",
                         "--out-dir", tmp], "build_metadata_tables")
            # Data S2 comes from its own tool, and both have to succeed before
            # the group means anything: comparing two of three recorded tables
            # and reporting PASS is the failure mode this tier exists to avoid.
            if built:
                built = run([sys.executable, "tools/build_data_s2.py",
                             "--out", str(Path(tmp) / "Data_S2_metadata_M1.xlsx")],
                            "build_data_s2")
            if built:
                for grp in TIER1_GROUPS:
                    tot = [a + b for a, b in zip(
                        tot, check_group(man, grp, at=Path(tmp)))]
                    visited.append(grp)
            else:
                for grp in TIER1_GROUPS:
                    skipped[grp] = "the M1/M2/M3 rebuild did not run"
                cannot.append("tier 1: M1/M2/M3 did not rebuild, so nothing "
                              "was compared against the recorded tables")
    else:
        for grp in TIER1_GROUPS:
            skipped[grp] = "tier 1 not requested"
        scope.append("tier 1: M1/M2/M3 were not rebuilt (pass --tier 1)")

    print("\n── existing pipeline outputs, figures and figure data objects ──")
    for grp in ONDISK_GROUPS:
        tot = [a + b for a, b in zip(tot, check_group(man, grp))]
        visited.append(grp)

    # The figure manifest can be renumbered under this one: the 2026-10-07 pass
    # retired S32 and moved S25-S31 up, so every recorded path for a retired
    # slot now reads MISSING with no hint why. Name it once, or a reader spends
    # the afternoon looking for a figure that no longer exists.
    try:
        from tools import figure_sets
        live_slots = {s for s, _r, _st in figure_sets.manifest()}
        live_pngs = {f"results/{r}/{st}.png" for _s, r, st in figure_sets.manifest()}
        retired = sorted({m.group(0) for rel in man.get("data_objects", {})
                          if (m := re.search(r"S\d+", Path(rel).name))
                          and m.group(0) not in live_slots})
        renamed = sorted(rel for rel in man.get("supp_figures", {})
                         if rel not in live_pngs)
        if retired or renamed:
            print("\n  note: the manifest was written against a different "
                  "figure manifest —")
            if retired:
                print(f"        data objects for retired slot(s): "
                      f"{', '.join(retired)}")
            if renamed:
                print(f"        {len(renamed)} recorded figure path(s) no "
                      f"longer named by tools/additional_figures_manifest.txt, "
                      f"e.g.\n          {renamed[0]}")
            print("        regenerate with --update on a complete tree.")
    except Exception as exc:                             # pragma: no cover
        print(f"  [WARN] could not compare the manifest's slots with the "
              f"figure manifest: {exc}")

    if args.tier >= 2:
        print("\n── tier 2: re-run a seeded, stochastic section and re-check ──")
        # The tier claims determinism, so the stage it re-runs must contain
        # randomness. Assert that before running it: a tier pointed at a
        # deterministic stage passes whatever the seeding does.
        rng_hits = stochastic_evidence(TIER2_MODULES)
        print(f"  stage: --only {TIER2_STAGE}  ({', '.join(TIER2_MODULES)})")
        if not rng_hits:
            print("  CANNOT VERIFY — no randomness found in the tier-2 stage, so "
                  "re-running it\n    says nothing about determinism. Point "
                  "TIER2_CMD/TIER2_MODULES at a seeded stage.")
            cannot.append(f"tier 2: --only {TIER2_STAGE} contains no randomness; "
                          "the determinism claim was not tested")
        elif not all(c in man.get("outputs", {}) for c in TIER2_CHECKS):
            missing_rec = [c for c in TIER2_CHECKS
                           if c not in man.get("outputs", {})]
            print("  CANNOT VERIFY — the tier-2 target is not recorded in the "
                  "manifest, so a\n    re-run has nothing to be compared "
                  "against: " + ", ".join(missing_rec)
                  + "\n    (regenerate it with --update on a complete tree)")
            cannot.append("tier 2: the re-run target is unrecorded, so the "
                          "determinism claim was not tested")
        else:
            print(f"  randomness confirmed: {len(rng_hits)} RNG site(s), "
                  f"e.g. {rng_hits[0]}")
            # A partial run rewrites results/<ref>/stats_report.* from scratch, keeping
            # only the sections that ran — so verifying would silently truncate the
            # accumulated report from a full run. Snapshot everything the re-run can
            # touch and put it back afterwards: verification must not damage results.
            collateral = [ROOT / "results/UHGG/stats_report.csv",
                          ROOT / "results/UHGG/stats_report.txt",
                          ROOT / "results/UHGG/manuscript_fill_ins.txt"]
            targets = [ROOT / c for c in TIER2_CHECKS]
            # The stage prints a message and writes nothing when its own input
            # CSVs are absent, and the process still exits 0. Fingerprint the
            # targets so a re-run that produced nothing cannot be read as a
            # re-run that reproduced.
            before = {p: (p.stat().st_mtime_ns, p.stat().st_size)
                      if p.exists() else None for p in targets}
            with tempfile.TemporaryDirectory(prefix="verify-restore-") as tmp:
                saved = {}
                for p in collateral + targets:
                    if p.exists():
                        saved[p] = Path(tmp) / p.name
                        shutil.copy2(p, saved[p])
                try:
                    if not run(TIER2_CMD, "tier-2 re-run"):
                        print("  CANNOT VERIFY — the re-run did not complete "
                              "(the matrices are needed for this tier).")
                        cannot.append("tier 2: the re-run failed, so the "
                                      "determinism claim was not tested")
                    else:
                        after = {p: (p.stat().st_mtime_ns, p.stat().st_size)
                                 if p.exists() else None for p in targets}
                        rewritten = [p for p in targets
                                     if before[p] != after[p] and after[p]]
                        if not rewritten:
                            print("  CANNOT VERIFY — the stage exited 0 but "
                                  "rewrote none of its targets\n    "
                                  + ", ".join(str(p.relative_to(ROOT))
                                              for p in targets)
                                  + "\n    (it skips silently when its own "
                                  "input CSVs are absent).")
                            cannot.append("tier 2: the stage wrote nothing, so "
                                          "the determinism claim was not tested")
                        else:
                            t2tot = [a + b for a, b in zip(
                                t2tot, check_group(man, "outputs",
                                                   only=TIER2_CHECKS))]
                finally:
                    for orig, backup in saved.items():
                        shutil.copy2(backup, orig)
                    if saved:
                        print(f"  restored {len(saved)} file(s) the re-run overwrote")
    else:
        scope.append("tier 2: determinism was not tested (pass --tier 2)")

    group_compared = sum(tot)
    exact, numeric, bad, missing, unver = [a + b for a, b in zip(tot, t2tot)]
    recorded_all = sum(len(man[g]) for g in GROUPS if g in man)
    recorded_visited = sum(len(man.get(g, {})) for g in visited)
    print("\n" + "─" * 64)
    print(f"  files recorded         : {recorded_all}")
    print(f"  files compared         : {group_compared}"
          f"  (groups: {', '.join(visited)})")
    for grp, why in sorted(skipped.items()):
        print(f"    not compared         : {grp} "
              f"({len(man.get(grp, {}))} file(s)) — {why}")
    print(f"  exact (byte-identical) : {exact}")
    print(f"  numerically identical  : {numeric}")
    print(f"  MISMATCHED             : {bad}")
    print(f"  missing                : {missing}")
    print(f"  UNVERIFIED             : {unver}")
    # A pass that compared fewer files than its groups record means
    # check_group and the manifest disagree about what is in a group -- it
    # must not read as a clean run.
    if group_compared != recorded_visited:
        print(f"\nFAIL — {recorded_visited} file(s) are recorded in the groups "
              f"this run visited but {group_compared} were compared.")
        sys.exit(1)
    if cannot:
        print("\n  claims this run TRIED and FAILED to establish:")
        for c in cannot:
            print(f"    · {c}")
    if scope:
        print("\n  outside the scope of this run:")
        for c in scope:
            print(f"    · {c}")
    if bad:
        print("\nFAIL — some outputs differ in value. If you changed inputs or code this is\n"
              "expected; otherwise please open an issue with this output.")
        sys.exit(1)
    if missing:
        print("\nINCOMPLETE — files above were not found. Matrices come from the Zenodo\n"
              "deposit (see download_data.sh); results/ is produced by running the pipeline.")
        sys.exit(2)
    if unver:
        print("\nUNVERIFIED — the files above differ from the manifest and the difference\n"
              "could not be adjudicated. This is not a pass: their reproduction is untested.")
        sys.exit(3)
    if cannot:
        print("\nUNVERIFIED — every file compared agreed, but the claims listed above were\n"
              "attempted and not established. This is not 'everything reproduces'.")
        sys.exit(3)
    print(f"\nPASS — all {group_compared} recorded file(s) in "
          f"{len(visited)} group(s) were compared and reproduce"
          + (";\n  not claimed: " + "; ".join(scope) if scope else "") + ".")


if __name__ == "__main__":
    main()
