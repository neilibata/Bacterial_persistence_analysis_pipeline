#!/usr/bin/env python3
"""Check, before a 12-hour run, that this machine can produce every output.

Each check below is a way a past run "completed cleanly" and still came back
wrong or incomplete:

  * Python and the pinned library versions (requirements-pinned.txt). An
    existing conda env is not re-installed by `conda activate`, so it can be
    missing packages the current pipeline needs.
  * The map stack: geopandas, shapely and plotly import, and the bundled Natural
    Earth basemap reads with country names. Figure 1 used to find its basemap
    inside pyogrio's test fixtures, which a fresh install may not have; the run
    then finished with no Figure 1.
  * The eight reference matrices are present and resolve (symlinks included).
  * Table M1 is the one the package was built with (sha256 against
    ../TRANSFER_MANIFEST.txt when run from an unpacked package) and has the
    expected 18,299 rows.
  * The external health-metadata sources resolve.
  * Disk space for results/.

Exits 1 if anything that would break or silently degrade the run fails; version
drift on a pinned package is a warning.

    python tools/preflight.py
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

EXPECTED_M1_ROWS = 18_299
MIN_FREE_GB = 50

errors: list[str] = []
warnings: list[str] = []


def ok(msg):   print(f"  OK    {msg}")
def warn(msg): print(f"  WARN  {msg}"); warnings.append(msg)
def fail(msg): print(f"  FAIL  {msg}"); errors.append(msg)


def check_python():
    print("\nPython")
    v = sys.version_info
    if (v.major, v.minor) == (3, 11):
        ok(f"Python {v.major}.{v.minor}.{v.micro} ({sys.executable})")
    else:
        fail(f"Python {v.major}.{v.minor}.{v.micro}; the pipeline is validated on 3.11 "
             f"(conda create -n microbiome python=3.11)")


def check_packages():
    print("\nPinned packages (requirements-pinned.txt)")
    for line in (ROOT / "requirements-pinned.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name, want = line.split("==", 1)
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            fail(f"{name} not installed (pip install -r requirements-pinned.txt)")
            continue
        if have == want:
            ok(f"{name} {have}")
        else:
            warn(f"{name} {have} installed, {want} pinned")


def check_maps():
    print("\nMap stack (Figure 1, metadata maps)")
    try:
        import geopandas  # noqa: F401
        import shapely    # noqa: F401
        import plotly     # noqa: F401
        ok("geopandas, shapely, plotly import")
    except ImportError as exc:
        fail(f"map dependency missing: {exc}")
        return
    try:
        sys.path.insert(0, str(ROOT / "figures"))
        import make_metadata_summary as mms
        path = mms._find_ne_path()
        world = mms._load_world()
        if "name" not in world.columns or len(world) < 150:
            fail(f"basemap {path} has {len(world)} rows / no 'name' column")
        elif "figures/basemap" not in path.replace(os.sep, "/"):
            warn(f"basemap read from {path}, not the bundled copy")
        else:
            ok(f"basemap: {len(world)} countries from {Path(path).relative_to(ROOT)}")
    except Exception as exc:  # noqa: BLE001 -- report whatever stops the map
        fail(f"basemap cannot be loaded: {exc}")


def check_matrices():
    print("\nReference matrices")
    from persistence import config
    for ref, cfg in config.REFERENCES.items():
        for key in ("breadth_file", "depth_file"):
            f = cfg.get(key)
            if not f:
                continue
            p = Path(f)
            if p.exists():
                ok(f"{ref:<9s} {key:<12s} {p.name} ({p.resolve().stat().st_size / 1e9:.2f} GB)")
            elif p.is_symlink():
                fail(f"{ref} {key}: {p.name} is a broken symlink -> {os.readlink(p)}")
            else:
                fail(f"{ref} {key}: {p.name} not in the repo root "
                     f"(symlink ../matrices/*.csv here; docs/PORTING.md §2)")


def check_m1():
    print("\nMetadata table M1")
    from persistence import config
    m1 = Path(config.META_FILE)
    if not m1.exists():
        fail(f"{m1} not found")
        return
    with open(m1, "rb") as fh:
        rows = sum(1 for _ in fh) - 1
    (ok if rows == EXPECTED_M1_ROWS else fail)(
        f"{m1} has {rows:,} rows (expected {EXPECTED_M1_ROWS:,})")
    manifest = ROOT.parent / "TRANSFER_MANIFEST.txt"
    if manifest.exists():
        want = next((l.split()[1] for l in manifest.read_text().splitlines()
                     if l.startswith("M1:")), None)
        have = hashlib.sha256(m1.read_bytes()).hexdigest()
        if want is None:
            warn("TRANSFER_MANIFEST.txt records no M1 hash")
        elif have == want:
            ok(f"M1 sha256 matches the package manifest ({have[:12]}…)")
        else:
            fail(f"M1 sha256 {have[:12]}… differs from the package manifest {want[:12]}…")


#: Pinned checksums of the nine inputs the reported run consumed. Written by
#: hand once (see docs/INPUT_CHECKSUMS.tsv for how each line is attested) so
#: that a reviewer can confirm the inputs without trusting any packaging
#: artifact that happens to be lying beside the repo.
CHECKSUM_FILE = ROOT / "docs" / "INPUT_CHECKSUMS.tsv"


def _pinned_inputs():
    """[(path, bytes, md5, sha256, attested_by)] from the pinned table."""
    if not CHECKSUM_FILE.exists():
        return []
    out = []
    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#") or line.startswith("path\t"):
            continue
        v = line.split("\t")
        if len(v) >= 5:
            out.append((v[0], int(v[1]), v[2], v[3], v[4]))
    return out


def check_inputs_match_the_run(full: bool = False):
    """Are these the inputs that produced the published numbers?

    Size and presence are not enough. Six files called metadata_M1.csv have
    existed in this tree -- repo root, metadata/, final_run/, a rename
    backup, and a _verified variant -- and they are different cohorts, not
    different spellings. config.py picks the first candidate that exists, so
    a reviewer with the wrong copy in the right place gets a complete run
    and different results.

    The curated metadata is hashed every time: it is 10 MB, it is the file
    most easily confused, and the cost is nothing. The multi-GB matrices are
    checked by size by default and hashed under --checksums, because hashing
    15.9 GB takes minutes and a reviewer should be able to run preflight
    freely.
    """
    print("\nInputs vs the reported run")
    pinned = _pinned_inputs()
    if not pinned:
        warn(f"{CHECKSUM_FILE.name} is missing; cannot confirm the inputs "
             f"are the ones that produced the published results")
        return
    for rel, nbytes, want_md5, want_sha, attested in pinned:
        p = Path(rel)
        if not p.exists():
            fail(f"{rel} not present")
            continue
        size = p.resolve().stat().st_size
        if size != nbytes:
            fail(f"{rel}: {size:,} bytes, expected {nbytes:,} -- this is not "
                 f"the file the reported run used")
            continue
        big = nbytes > 100e6
        if big and not full:
            ok(f"{rel:<52s} size matches ({size/1e9:.2f} GB; --checksums to "
               f"hash it)")
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 23), b""):
                h.update(b)
        got = h.hexdigest()
        if got == want_sha:
            note = "" if attested == "run_manifest" else "  (local attestation)"
            ok(f"{rel:<52s} sha256 matches{note}")
        else:
            fail(f"{rel}: sha256 {got[:16]}... expected {want_sha[:16]}... -- "
                 f"this is NOT the file the reported run used")

    # The silent-fallback trap, stated explicitly.
    from persistence import config
    first = config.META_FILE_CANDIDATES[0]
    if config.META_FILE != first:
        fail(f"config resolved the metadata to {config.META_FILE!r}, not "
             f"{first!r}. That is a different cohort and will not reproduce "
             f"the published numbers.")
    else:
        ok(f"config resolves the metadata to {first}")


def check_sources():
    """The MicrobiomeDB and da Silva sources behind the curated metadata.

    A warning, not a failure. These are inputs to *building* M1
    (`persistence/build_final_metadata.py`, `tools/build_subject_ids.py`,
    `tools/build_cmd_health_audit.py`); nothing on the run path reads them,
    because the run consumes the finished `final_run/metadata_M1.csv`, which
    `download_data.sh` fetches from the deposit alongside the matrices.

    Failing here blocked the one case the deposit exists for: a reviewer who
    downloads the record and runs the pipeline, without the third-party
    MicrobiomeDB export that only M1's own provenance needs.
    """
    print("\nExternal health-metadata sources")
    from persistence import config
    root = config.resolve_data_root()
    if root is None:
        warn("not found -- needed only to rebuild metadata_M1.csv from its "
             "sources, not to run the pipeline, which reads the finished "
             "final_run/metadata_M1.csv. "
             + config.describe_data_root_failure())
    else:
        ok(f"found under {root}")


def check_disk():
    print("\nDisk")
    free = shutil.disk_usage(ROOT).free / 1e9
    (ok if free >= MIN_FREE_GB else fail)(f"{free:,.0f} GB free (need ~{MIN_FREE_GB} GB)")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checksums", action="store_true",
                    help="also hash the multi-GB matrices, not just the "
                         "metadata. Slower (reads ~16 GB) but proves every "
                         "input is the one the reported run consumed.")
    a = ap.parse_args()

    print(f"Preflight for {ROOT}")
    checks = [check_python, check_packages, check_maps, check_matrices,
              check_m1,
              lambda: check_inputs_match_the_run(full=a.checksums),
              check_sources, check_disk]
    for check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001
            name = getattr(check, "__name__", "check_inputs_match_the_run")
            fail(f"{name} raised {type(exc).__name__}: {exc}")
    print()
    if errors:
        print(f"PREFLIGHT FAILED: {len(errors)} problem(s), {len(warnings)} warning(s). "
              f"Do not start the run.")
        return 1
    print(f"PREFLIGHT PASSED ({len(warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
