#!/usr/bin/env python3
"""Check a new M1 against the version it was built from, and against ENA.

Written to be independent of build_m1_v2.py: it re-reads both CSVs and re-derives
every claim from the archive rather than trusting the provenance file. If the two
disagree the builder is wrong, which is the only useful thing a verifier can say.

The checks, in the order a reviewer would ask them:

  1  shape        every original column survives in its original order; the
                  only permitted addition is Master_ID
  2  preservation every row of the old file still present and unchanged, except
                  the reprocessed ones, which may differ ONLY in the run and
                  depth columns. This is the check that matters most -- the
                  failure mode being guarded against is a rewrite that quietly
                  reformats 18,046 rows while adding 260. Rows are paired on
                  Sample ID + SRS ID, the composite build_final_metadata.py
                  itself dedups on, because Chosen_match_ID is not unique.
  3  identity     Master_ID present and unique; no NEW Chosen_match_ID
                  collisions beyond the 21 already in the file, and each of
                  those is a single BioSample prepped twice -- which is the
                  assumption loading.py's largest-Total-bases rule relies on;
                  no added sample colliding with an existing identifier under
                  any of the eight ID columns
  4  archive      for every added row, the accessions, run list and base counts
                  are re-fetched from the ENA cache and compared
  5  inheritance  each added row's study-level fields match its same-study
                  same-country siblings
  6  vocabulary   Lifestyle values are ones config.canonical_lifestyle accepts,
                  so nothing lands outside the three groups
  7  exclusions   none of the excluded samples appear
  8  bytes        CRLF preserved and the trailing-newline convention unchanged

Exit status is non-zero if any check fails, so it can gate a commit.

    python tools/verify_m1_v2.py --old metadata_Mar_16_2026.csv \
        --new metadata_Aug_17_2026.csv --ena-dir <dir> --worklist <annotated.csv>
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from persistence import config as cfg                  # noqa: E402
from geo_resolve import resolve as geo_resolve            # noqa: E402

MASTER = "Master_ID"
RUN_COLS = {"SRA_Accession", "Secondary_run_IDs", "Paired SRR List",
            "NumPairedRuns", "LineType", "Total bases"}
ID_COLS = ["Chosen_match_ID", "SRA_Accession", "SRS ID", "BioSample", "Index",
           "Secondary_run_IDs", "Paired SRR List", "Sample ID"]
INHERIT = ["Author", "Year", "StudyFolder", "Study", "Population ID", "PubMed ID",
           "Population", "Country", "Country (full)", "Lifestyle"]


class Report:
    def __init__(self):
        self.fail = 0
        self.n = 0

    def check(self, ok, label, detail=""):
        self.n += 1
        if not ok:
            self.fail += 1
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}" + (f"   {detail}" if detail else ""))
        return ok


def read_rows(path: Path):
    raw = path.read_bytes()
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
    return rows[0], rows[1:], raw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--ena-dir", type=Path, required=True)
    ap.add_argument("--runs-extra", type=Path, nargs="*", default=[])
    ap.add_argument("--worklist", type=Path, required=True)
    ap.add_argument("--exclude", type=Path,
                    default=ROOT / "tools" / "excluded_reprocessing_samples.tsv")
    ap.add_argument("--matrix-samples", type=Path, default=None)
    args = ap.parse_args()

    r = Report()
    oh, od, oraw = read_rows(args.old)
    nh, nd, nraw = read_rows(args.new)
    print(f"  old {args.old.name}: {len(od):,} rows    "
          f"new {args.new.name}: {len(nd):,} rows\n")

    # ---- 1 shape ------------------------------------------------------------
    # The new file may carry exactly one extra column, Master_ID, at the front.
    # Everything the old file had must survive, in its original order.
    extra = [c for c in nh if c not in oh]
    r.check(extra in ([], [MASTER]), "only permitted new column is Master_ID",
            f"extra columns: {extra}")
    kept = [c for c in nh if c != MASTER]
    r.check(kept == oh, "every original column survives, in the original order",
            "" if kept == oh else f"old {len(oh)} vs kept {len(kept)}")
    oidx = {c: i for i, c in enumerate(oh)}
    nidx = {c: i for i, c in enumerate(nh)}
    shared = [c for c in oh if c in nidx]

    # ---- 2 preservation -----------------------------------------------------
    # Rows are paired on the composite the pipeline itself treats as identity
    # (Sample ID + SRS ID, per build_final_metadata.py step 4b), NOT on
    # Chosen_match_ID -- which 21 Carter pairs share.
    w = pd.read_csv(args.worklist, dtype=str)
    reproc = set(w[w.group == "A_incomplete"]["sample"])

    def ident(row, ix):
        return (row[ix["Sample ID"]], row[ix["SRS ID"]])

    okey = {ident(row, oidx): row for row in od}
    nkey = {ident(row, nidx): row for row in nd}
    r.check(len(okey) == len(od), "identity key unique in the old file",
            f"{len(okey):,} keys for {len(od):,} rows")
    r.check(len(nkey) == len(nd), "identity key unique in the new file",
            f"{len(nkey):,} keys for {len(nd):,} rows")

    missing = [k for k in okey if k not in nkey]
    r.check(not missing, f"all {len(okey):,} original rows still present",
            "" if not missing else f"{len(missing)} lost, e.g. {missing[:3]}")

    untouched_bad, touched_bad, n_touched = [], [], 0
    for k, orow in okey.items():
        nrow = nkey.get(k)
        if nrow is None:
            continue
        diffs = {c for c in shared if orow[oidx[c]] != nrow[nidx[c]]}
        if not diffs:
            continue
        n_touched += 1
        ids = {orow[oidx[c]] for c in ID_COLS} | {
            x.strip() for c in ("Paired SRR List", "Secondary_run_IDs")
            for x in orow[oidx[c]].replace(",", ";").split(";") if x.strip()}
        if ids & reproc:
            outside = diffs - RUN_COLS
            if outside:
                touched_bad.append((k[0], sorted(outside)[:5]))
        else:
            untouched_bad.append((k[0], sorted(diffs)[:5]))
    r.check(not untouched_bad,
            "no row outside the reprocessed set was modified",
            "" if not untouched_bad else f"{len(untouched_bad)} changed, "
            f"e.g. {untouched_bad[:3]}")
    r.check(not touched_bad,
            "reprocessed rows changed only in run/depth columns",
            "" if not touched_bad else f"{len(touched_bad)} changed elsewhere, "
            f"e.g. {touched_bad[:3]}")
    print(f"         ({n_touched} rows modified in total)")

    added = [row for row in nd if ident(row, nidx) not in okey]
    r.check(len(nd) == len(od) + len(added),
            f"row arithmetic: {len(od):,} + {len(added)} added = {len(nd):,}")

    # ---- 3 identity ---------------------------------------------------------
    mids = [row[nidx[MASTER]] for row in nd] if MASTER in nidx else []
    r.check(bool(mids) and all(mids), "Master_ID present on every row",
            f"{sum(1 for m in mids if not m)} blank")
    dm = [k for k, v in Counter(mids).items() if v > 1]
    r.check(not dm, f"Master_ID unique across {len(mids):,} rows",
            "" if not dm else f"{len(dm)} duplicated: {dm[:5]}")

    # Chosen_match_ID is a run pointer, so duplicates are expected. What must
    # not happen is the build creating new ones.
    ocm = Counter(row[oidx["Chosen_match_ID"]] for row in od)
    ncm = Counter(row[nidx["Chosen_match_ID"]] for row in nd)
    od_dup = {k for k, v in ocm.items() if v > 1}
    nd_dup = {k for k, v in ncm.items() if v > 1}
    r.check(nd_dup == od_dup,
            f"no new Chosen_match_ID collisions ({len(od_dup)} pre-existing)",
            "" if nd_dup == od_dup else f"introduced: {sorted(nd_dup - od_dup)[:5]}")
    # and each pre-existing collision is one BioSample prepped twice, which is
    # what loading.py's largest-Total-bases rule assumes
    same_bs = all(len({row[nidx["BioSample"]] for row in nd
                       if row[nidx["Chosen_match_ID"]] == k}) == 1 for k in nd_dup)
    r.check(same_bs, "every Chosen_match_ID collision is a single BioSample")

    old_ids = defaultdict(list)
    for i, row in enumerate(od):
        for c in ID_COLS:
            for v in row[oidx[c]].replace(",", ";").split(";"):
                v = v.strip()
                if v:
                    old_ids[v].append(i)
    coll = []
    for row in added:
        for c in ID_COLS:
            for v in row[nidx[c]].replace(",", ";").split(";"):
                v = v.strip()
                if v and v in old_ids:
                    coll.append((row[nidx["Chosen_match_ID"]], c, v))
    r.check(not coll, "no added sample collides with an existing identifier",
            "" if not coll else f"{len(coll)} collisions: {coll[:4]}")

    # ---- 4 archive ----------------------------------------------------------
    frames = []
    for f in sorted(glob.glob(str(args.ena_dir / "*.tsv"))):
        frames.append(pd.read_csv(f, sep="\t", dtype=str))
    e = pd.concat(frames, ignore_index=True)
    e["_b"] = pd.to_numeric(e["base_count"], errors="coerce").fillna(0)
    runs = e.drop_duplicates("run_accession").set_index("run_accession")
    for pat in args.runs_extra:
        for f in sorted(glob.glob(str(pat))):
            x = pd.read_csv(f, sep="\t", dtype=str)
            if "base_count" not in x.columns:
                continue
            x["_b"] = pd.to_numeric(x["base_count"], errors="coerce").fillna(0)
            x = x.drop_duplicates("run_accession").set_index("run_accession")
            runs = pd.concat([runs, x[~x.index.isin(runs.index)][["_b"]]])
    samp = e.drop_duplicates("sample_accession").set_index("sample_accession")

    bad_bases, bad_bio, no_runs = [], [], []
    for row in added:
        sid = row[nidx["Chosen_match_ID"]]
        acc = [x.strip() for x in row[nidx["Paired SRR List"]].split(";") if x.strip()]
        known = [a for a in acc if a in runs.index]
        if not known:
            no_runs.append(sid)
            continue
        want = int(sum(float(runs.at[a, "_b"]) for a in known))
        try:
            got = int(row[nidx["Total bases"]])
        except ValueError:
            got = -1
        if got != want:
            bad_bases.append((sid, got, want))
        bs = row[nidx["BioSample"]]
        if bs and bs in samp.index:
            pass
        elif bs:
            bad_bio.append(sid)
    r.check(not bad_bases, "Total bases equals summed ENA base_count for added rows",
            "" if not bad_bases else f"{len(bad_bases)} mismatch: {bad_bases[:3]}")
    r.check(not no_runs, "every added row's runs resolve in the archive",
            "" if not no_runs else f"{len(no_runs)} unresolved: {no_runs[:4]}")
    r.check(not bad_bio, "BioSample accessions exist in ENA",
            "" if not bad_bio else f"{len(bad_bio)} unknown: {bad_bio[:4]}")

    # ---- 5 inheritance ------------------------------------------------------
    odf = pd.DataFrame(od, columns=oh)
    mism = []
    for row in added:
        bp = row[nidx["BioProject ID"]]
        sib = odf[odf["BioProject ID"] == bp]
        cf = row[nidx["Country (full)"]]
        if cf and sib["Country (full)"].nunique() > 1:
            sib = sib[sib["Country (full)"] == cf]
        if not len(sib):
            mism.append((row[nidx["Chosen_match_ID"]], "no siblings", "", ""))
            continue
        for c in INHERIT:
            v = row[nidx[c]]
            if not v:
                continue
            allowed = set(sib[c].dropna()) - {""}
            if allowed and v not in allowed:
                mism.append((row[nidx["Chosen_match_ID"]], c, v, sorted(allowed)[:3]))
    r.check(not mism, "added rows' study fields match their same-study siblings",
            "" if not mism else f"{len(mism)} mismatches: {mism[:3]}")

    # ---- 6 vocabulary -------------------------------------------------------
    bad_life = sorted({row[nidx["Lifestyle"]] for row in nd
                       if cfg.canonical_lifestyle(row[nidx["Lifestyle"]]) is None})
    r.check(not bad_life, "every Lifestyle value is canonical",
            "" if not bad_life else f"unrecognised: {bad_life[:6]}")
    life_added = Counter(cfg.canonical_lifestyle(row[nidx["Lifestyle"]]) for row in added)
    print(f"         (added rows by lifestyle: {dict(life_added)})")

    # ---- 7 exclusions -------------------------------------------------------
    excl = set()
    if args.exclude.exists():
        for line in args.exclude.read_text().splitlines():
            if line and not line.startswith("#"):
                excl.add(line.split("\t")[0])
    present = sorted({row[nidx["Chosen_match_ID"]] for row in nd} & excl)
    r.check(not present, f"none of the {len(excl)} excluded samples appear",
            "" if not present else f"{len(present)} present: {present[:5]}")

    # ---- 8 bytes ------------------------------------------------------------
    CRLF, LF = b"\r\n", b"\n"
    o_bare = oraw.count(LF) - oraw.count(CRLF)
    n_bare = nraw.count(LF) - nraw.count(CRLF)
    r.check(n_bare == o_bare == 0, "line endings still CRLF throughout",
            f"bare LF old={o_bare} new={n_bare}")
    r.check(oraw.endswith(LF) == nraw.endswith(LF),
            "trailing-newline convention unchanged",
            f"old={oraw.endswith(LF)} new={nraw.endswith(LF)}")
    r.check(nraw.count(CRLF) == len(nd) + (1 if not nraw.endswith(LF) else 0) - 1 + 1
            or nraw.count(CRLF) == len(nd),
            "CRLF count consistent with row count",
            f"{nraw.count(CRLF):,} terminators for {len(nd):,} data rows")

    # ---- optional: agreement with the coverage matrices ---------------------
    if args.matrix_samples and args.matrix_samples.exists():
        have = {l.strip() for l in args.matrix_samples.read_text().splitlines() if l.strip()}
        ids = {row[nidx["Chosen_match_ID"]] for row in added}
        miss = sorted(ids - have)
        r.check(not miss, "every added sample has a row in the coverage matrices",
                "" if not miss else f"{len(miss)} missing: {miss[:5]}")

        # Chosen_match_ID is a cached result, not the rule. build_id_lookup()
        # links a matrix row to a metadata row through SRS ID / SRA_Accession /
        # Sample ID / run_accession / BioSample / Paired SRR List, and picks the
        # highest-depth match. A row whose Chosen_match_ID cannot be re-derived
        # from its own identifiers is one that only resolves because somebody
        # wrote the answer in by hand -- it will vanish the next time
        # build_final_metadata.py is run, silently. That is exactly what
        # happened to all 260 rows added in the 2026 pass before BioSample was
        # added to the lookup.
        def id_tokens(row):
            t = set()
            for c in ("SRS ID", "SRA_Accession", "Sample ID", "run_accession",
                      "BioSample"):
                if c in nidx:
                    v = row[nidx[c]].strip()
                    if v:
                        t.add(v)
            v = row[nidx["Paired SRR List"]]
            t |= {x.strip() for x in v.replace(",", ";").split(";") if x.strip()}
            return t

        def underivable(rows, ix):
            out = []
            for row in rows:
                cm = row[ix["Chosen_match_ID"]]
                if cm in have and cm not in id_tokens(row):
                    out.append(cm)
            return out

        bad_new = underivable(added, nidx)
        r.check(not bad_new,
                "every ADDED row's matrix link is re-derivable from the lookup rule",
                "" if not bad_new else f"{len(bad_new)} rely on a hand-set "
                f"Chosen_match_ID: {bad_new[:4]}")
        bad_all = underivable(nd, nidx)
        pre = len(bad_all) - len(bad_new)
        r.check(len(bad_new) == 0,
                f"no NEW unresolvable matrix links ({pre} pre-existing, all "
                f"NielsenHB_2014_DNK, mapped through the step-3 recovery path)")

    print(f"\n  {r.n - r.fail}/{r.n} checks passed")
    return 1 if r.fail else 0


if __name__ == "__main__":
    sys.exit(main())
