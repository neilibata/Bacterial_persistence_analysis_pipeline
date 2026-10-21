#!/usr/bin/env python3
"""Add the reprocessed samples to M1, writing a new version beside the old one.

The existing file is never opened for writing. A new dated CSV is produced and
the old one is copied, byte for byte, into archive/ -- so the previous version
survives both as its original path and as an explicit snapshot.

Three kinds of row come out of this:

  unchanged      17,859 rows nothing in this run touches
  updated        187 rows whose samples were reprocessed with runs the original
                 pass missed. Only the run-list and depth columns move; every
                 other cell is carried across untouched, and each change is
                 written to the provenance sidecar with its before and after.
  added          260 rows for samples that were never in M1

It also adds one column, Master_ID. Chosen_match_ID could not serve as the
metadata's identity key -- it is assigned downstream as whichever matrix row
carried the highest depth score, so 21 Carter 2023 pairs share one. Master_ID
anchors on the BioSample accession instead, suffixed by Sample ID rank where a
BioSample legitimately carries several rows. Chosen_match_ID keeps its job as
the join key into the coverage matrices; the two are not interchangeable.

Where each column of an added row comes from:

  archive     read off ENA -- accessions, run lists, base counts, instrument
  inherited   copied from M1 rows of the same study *and the same country*.
              BioProject alone is not a safe key: PRJEB49206 spans Nepal,
              Tanzania and the USA at three different lifestyles, and
              PRJEB17632 spans Germany and Kazakhstan at two. Inheriting on
              BioProject alone would have labelled Kazakh samples Industrial.
  constant    a study-level value every sibling agrees on (all of these
              cohorts code 'Diseased (other)' as No, for instance). Only used
              where the siblings are unanimous; a split forces blank.
  blank       no source. ENA carries no age for these samples and sex for two
              of 281, so the clinical columns are simply empty -- which is what
              `is_yes()` already treats as No, and what loading.py already
              recodes to Healthy for non-industrial samples.

Nothing is guessed. A column with no archive value and no unanimous sibling is
left empty rather than filled with a plausible default, and every populated
cell in every added row is written to the provenance file with the rule and the
evidence that produced it.

    python tools/build_m1_v2.py --worklist <annotated.csv> --ena-dir <dir> \
        --exclude tools/excluded_reprocessing_samples.tsv --out-dir .
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from geo_resolve import resolve as geo_resolve            # noqa: E402

# Columns rewritten for a reprocessed (group A) sample. Everything else on those
# rows is carried across byte-identical.
RUN_COLS = ["SRA_Accession", "Secondary_run_IDs", "Paired SRR List",
            "NumPairedRuns", "LineType", "Total bases"]

# Study-level identity. Inherited as a block from same-study same-country rows.
INHERIT = ["Author", "Year", "StudyFolder", "Study", "Population ID", "PubMed ID",
           "Population", "Country", "Country (full)", "Lifestyle",
           "study_name [MBIOTEMP_study_name]", "non_westernized"]

# Inherited only when every sibling agrees.
CONSTANT_IF_UNANIMOUS = ["Diseased (all)", "Diseased (gut and metabolic)",
                         "Diseased (other)", "disease [MBIOTEMP_disease]"]

# Modal sibling agreement below this is reported rather than assumed silent.
WEAK_AGREEMENT = 0.99

# Cohort-level annotations: filled from a unanimous sibling vote so the added
# rows carry the same curation as the rest of their study.
#
# This is an ALLOWLIST on purpose. "Every sibling agrees" is not on its own a
# licence to copy a value, because it cannot tell a cohort-level convention from
# a per-individual measurement that happens to be uniform in a small sample. The
# Bramble DRC study makes the difference concrete: its six curated rows are all
# 13-14 year olds, so `Age group` is unanimously 'Teenager'. Inheriting that
# would assert an age for 174 Congolese individuals on the evidence of six --
# whereas `disease_T2D = No` across the same six is a statement about how the
# study was annotated, not about any one person.
#
# So age, sex, BMI, obesity and overweight are never inherited and stay blank.
# Nothing is lost by that: `is_yes()` reads blank as False, so a blank obesity
# flag behaves exactly as 'No' downstream without asserting a body composition
# nobody measured.
COHORT_FLAG_PREFIX = "disease_"
COHORT_FLAG_EXTRA = ["Diseased (all)", "Unhealthy"]
MIN_FLAG_SIBLINGS = 5


MASTER_COL = "Master_ID"


def master_ids(rows, idx):
    """A stable per-sample identity key, independent of which run won.

    Chosen_match_ID cannot serve as one. build_final_metadata.py maps matrix
    rows to metadata through SRS ID / SRA_Accession / Sample ID / Paired SRR
    List, dedups on Sample ID + SRS ID, and only then sets Chosen_match_ID to
    whichever matrix row carried the highest depth score. It is a pointer to a
    run, so two rows can share it: 21 Carter 2023 pairs do, being the same
    BioSample prepped two ways (Hadza_MoBio_* and Hadza_PheChl_*). Keying
    anything on it silently merges those samples.

    BioSample is the archive's own sample-level accession and is the natural
    anchor. Where one BioSample legitimately carries several rows, the alias in
    Sample ID separates them, so the suffix is the rank of Sample ID sorted
    alphabetically within that BioSample -- deterministic, and unchanged by row
    order, depth, or a later rebuild.
    """
    def anchor(row):
        for c in ("BioSample", "SRS ID", "Sample ID", "Chosen_match_ID"):
            v = row[idx[c]].strip() if c in idx else ""
            if v:
                return v
        return ""

    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[anchor(row)].append(i)
    out = [""] * len(rows)
    for a, members in groups.items():
        if len(members) == 1:
            out[members[0]] = a
            continue
        order = sorted(members, key=lambda i: (rows[i][idx["Sample ID"]], i))
        for k, i in enumerate(order, start=1):
            out[i] = f"{a}.{k}"
    return out


def read_csv_bytes(path: Path):
    """Read a CSV preserving the exact row values, and note the trailing newline.

    M1 is CRLF throughout with no terminator on the final line. Writing it back
    through pandas would silently normalise both, so the whole round trip goes
    through the csv module on decoded bytes and the terminator is restored by
    hand. An earlier pass through read_text/write_text converted all 18,048 line
    endings and had to be restored from backup; hence the care here.
    """
    raw = path.read_bytes()
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
    return rows, raw.endswith(b"\n")


def write_csv_bytes(path: Path, rows, trailing_newline: bool) -> int:
    buf = io.StringIO(newline="")
    csv.writer(buf, lineterminator="\r\n").writerows(rows)
    out = buf.getvalue().encode("utf-8")
    if not trailing_newline and out.endswith(b"\r\n"):
        out = out[:-2]
    path.write_bytes(out)
    return len(out)


def load_ena(ena_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(ena_dir / "*.tsv")))
    if not files:
        sys.exit(f"no ENA tsv files in {ena_dir}")
    e = pd.concat([pd.read_csv(f, sep="\t", dtype=str) for f in files],
                  ignore_index=True)
    for c in ("base_count", "read_count"):
        e["_" + c] = pd.to_numeric(e.get(c), errors="coerce").fillna(0)
    return e


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, default=ROOT / "metadata_Mar_16_2026.csv")
    ap.add_argument("--worklist", type=Path, required=True)
    ap.add_argument("--ena-dir", type=Path, required=True)
    ap.add_argument("--runs-extra", type=Path, nargs="*", default=[],
                    help="further run-level TSVs carrying run_accession and "
                         "base_count. The reprocessed samples come from studies "
                         "outside the added samples' BioProjects, so their runs "
                         "are absent from --ena-dir; without this their depth "
                         "columns silently fail to update.")
    ap.add_argument("--exclude", type=Path, default=ROOT / "tools" / "excluded_reprocessing_samples.tsv")
    ap.add_argument("--out", type=Path, required=True, help="new M1 path")
    ap.add_argument("--archive-dir", type=Path, default=ROOT / "archive")
    ap.add_argument("--provenance", type=Path, default=None)
    ap.add_argument("--matrix-samples", type=Path, default=None,
                    help="optional: one sample id per line, the rows actually "
                         "present in the coverage matrices. Added samples "
                         "missing from it are reported, not dropped.")
    args = ap.parse_args()

    if args.out.resolve() == args.metadata.resolve():
        sys.exit("refusing to overwrite the input; --out must be a new path")

    rows, trailing = read_csv_bytes(args.metadata)
    header, data = rows[0], rows[1:]
    idx = {c: i for i, c in enumerate(header)}
    print(f"  read {args.metadata.name}: {len(data):,} rows x {len(header)} cols")

    # ---- archive the old version, byte for byte -----------------------------
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    snap = args.archive_dir / args.metadata.name
    if snap.exists():
        print(f"  archive snapshot already present: {snap}")
    else:
        shutil.copy2(args.metadata, snap)
    assert snap.read_bytes() == args.metadata.read_bytes(), "archive copy differs"
    print(f"  archived -> {snap}  (verified byte-identical)")

    # ---- inputs -------------------------------------------------------------
    w = pd.read_csv(args.worklist, dtype=str)
    e = load_ena(args.ena_dir)
    excl = {}
    if args.exclude.exists():
        with args.exclude.open() as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                p = line.rstrip("\n").split("\t")
                excl[p[0]] = p[1] if len(p) > 1 else "excluded"
    print(f"  worklist {len(w):,} samples   ENA runs {len(e):,}   exclusions {len(excl)}")

    key = {}
    for c in ("sample_accession", "secondary_sample_accession", "run_accession"):
        for v, sa in zip(e[c], e["sample_accession"]):
            if isinstance(v, str) and v:
                key.setdefault(v, sa)
    by_sample = {k: g for k, g in e.groupby("sample_accession")}
    run_frames = [e[["run_accession", "_base_count", "_read_count"]]]
    for pat in args.runs_extra:
        for f in sorted(glob.glob(str(pat))):
            x = pd.read_csv(f, sep="\t", dtype=str)
            if "run_accession" not in x.columns or "base_count" not in x.columns:
                continue
            x["_base_count"] = pd.to_numeric(x["base_count"], errors="coerce").fillna(0)
            x["_read_count"] = (pd.to_numeric(x["read_count"], errors="coerce").fillna(0)
                                if "read_count" in x.columns else 0.0)
            run_frames.append(x[["run_accession", "_base_count", "_read_count"]])
    by_run = (pd.concat(run_frames, ignore_index=True)
                .dropna(subset=["run_accession"])
                .drop_duplicates("run_accession").set_index("run_accession"))
    print(f"  run-level records available: {len(by_run):,} "
          f"({len(run_frames)} source(s))")

    meta_df = pd.DataFrame(data, columns=header)

    # ---- calibrate the optional archive-derived columns ----------------------
    # number_reads is an MBIOTEMP column, half empty and possibly defined
    # differently from ENA's read_count. Fill it only if the two agree on the
    # rows where both already exist.
    cal = meta_df[meta_df["number_reads"].astype(str).str.strip() != ""].head(3000)
    agree = tot = 0
    for _, r in cal.iterrows():
        accs = {x.strip() for c in ("Paired SRR List", "SRA_Accession")
                for x in str(r[c]).replace(",", ";").split(";") if x.strip()}
        hit = [by_run.at[a, "_read_count"] for a in accs if a in by_run.index]
        if not hit:
            continue
        tot += 1
        try:
            if abs(float(r["number_reads"]) - sum(hit)) < 1:
                agree += 1
        except ValueError:
            pass
    fill_reads = tot >= 50 and agree / tot > 0.9
    print(f"  number_reads vs ENA read_count: {agree}/{tot} agree -> "
          f"{'fill from archive' if fill_reads else 'leave blank (definitions differ)'}")

    # ---- split the worklist -------------------------------------------------
    w["_ena"] = w["sample"].map(key)
    grpA = w[w.group == "A_incomplete"]
    grpB = w[w.group == "B_new_nonind"]
    addB = grpB[~grpB["sample"].isin(excl)]
    print(f"  group A (update) {len(grpA)}   group B (add) {len(grpB)} "
          f"- {len(grpB) - len(addB)} excluded = {len(addB)}")

    prov = []       # (sample, column, value, source, evidence)
    weak = []       # inherited on a less-than-unanimous sibling vote

    # ---- update the reprocessed rows ---------------------------------------
    m1_pos = {}
    for i, row in enumerate(data):
        for c in ("Chosen_match_ID", "SRA_Accession", "SRS ID", "BioSample", "Index",
                  "Secondary_run_IDs", "Paired SRR List"):
            for v in row[idx[c]].replace(",", ";").split(";"):
                v = v.strip()
                if v:
                    m1_pos.setdefault(v, i)
    n_upd = n_found = 0
    no_row, no_runs = [], []
    for _, r in grpA.iterrows():
        i = m1_pos.get(r["sample"])
        if i is None:
            no_row.append(r["sample"])
            continue
        n_found += 1
        runs = sorted({x.strip() for x in str(r["runs"]).split(";") if x.strip()})
        known = [a for a in runs if a in by_run.index]
        if not known:
            no_runs.append(r["sample"])
            continue
        total = int(sum(by_run.at[a, "_base_count"] for a in known))
        # Secondary_run_IDs must be EMPTY for a reprocessed sample. The column
        # means "runs of this biosample that exist as their own matrix rows",
        # and reprocessing concatenated every run into the single merged row, so
        # there are none left. Writing runs[1:] here was wrong twice over:
        #
        #   Chosen_match_ID is not necessarily runs[0], so for 37 samples the
        #   primary listed its own accession as a secondary and loading.py
        #   dutifully treated the sample as a technical replicate of itself.
        #
        #   53 of the listed runs still exist as per-run rows from the original
        #   per-run processing. Those reads are now also inside the merged row,
        #   so the reinstatement step could promote a strict subset of a sample
        #   to an independent observation of it.
        #
        # Clearing the field fixes both: the self-references vanish, and the 53
        # stale rows become orphans with no M1 entry, which loading.py already
        # drops when it intersects metadata against the matrix. Paired SRR List
        # still records every run, so no provenance is lost.
        new = {"SRA_Accession": runs[0],
               "Secondary_run_IDs": "",
               "Paired SRR List": ";".join(runs),
               "NumPairedRuns": str(len(runs)),
               "LineType": "Single paired run" if len(runs) == 1 else "Multiple paired runs",
               "Total bases": str(total)}
        changed = False
        for c, v in new.items():
            old = data[i][idx[c]]
            if old != v:
                prov.append((r["sample"], c, v, "updated", f"was {old!r}"))
                data[i][idx[c]] = v
                changed = True
        n_upd += changed
    print(f"  group A: {n_found}/{len(grpA)} located in M1, "
          f"{n_upd} rows actually changed, {len(grpA) - n_found - len(no_runs)} "
          f"already current")
    if no_row:
        print(f"  !! {len(no_row)} group-A samples not found in M1: {no_row[:6]}")
    if no_runs:
        print(f"  !! {len(no_runs)} group-A samples have no run record "
              f"(pass --runs-extra): {no_runs[:6]}")

    # ---- build the added rows ----------------------------------------------
    meta_df["_bp"] = meta_df["BioProject ID"]
    added = []
    unresolved = []
    for _, r in addB.iterrows():
        sample = r["sample"]
        es = r["_ena"]
        g = by_sample.get(es)
        if g is None:
            unresolved.append(sample)
            continue
        first = g.iloc[0]
        runs = sorted({x.strip() for x in str(r["runs"]).split(";") if x.strip()})
        known = [a for a in runs if a in by_run.index]
        bp = str(first.get("study_accession") or r["study"])
        raw_country = first.get("country")
        country = geo_resolve(raw_country) if isinstance(raw_country, str) else None

        # siblings: same study, and same country when the study spans several
        sib = meta_df[meta_df._bp == bp]
        basis = f"BioProject {bp}"
        if country is not None and sib["Country (full)"].nunique(dropna=True) > 1:
            narrowed = sib[sib["Country (full)"] == country]
            if len(narrowed):
                sib, basis = narrowed, f"BioProject {bp} + country {country}"
            else:
                unresolved.append(f"{sample} (no sibling for {country!r} in {bp})")
                continue
        if not len(sib):
            unresolved.append(f"{sample} (no sibling rows for {bp})")
            continue

        out = {c: "" for c in header}

        # --- archive-derived
        arc = {
            "Chosen_match_ID": sample,
            "SRA_Accession": runs[0] if runs else "",
            "Secondary_run_IDs": ";".join(runs[1:]),
            "Paired SRR List": ";".join(runs),
            "NumPairedRuns": str(len(runs)),
            "LineType": "Single paired run" if len(runs) == 1 else "Multiple paired runs",
            "Total bases": str(int(sum(by_run.at[a, "_base_count"] for a in known))) if known else "",
            "BioSample": str(first.get("sample_accession") or ""),
            "SRS ID": str(first.get("secondary_sample_accession") or ""),
            "BioProject ID": bp,
            "Sample ID": str(first.get("sample_title") or first.get("sample_alias") or ""),
        }
        if fill_reads and known:
            arc["number_reads"] = str(int(sum(by_run.at[a, "_read_count"] for a in known)))
        for c, v in arc.items():
            if c in out and v != "":
                out[c] = v
                prov.append((sample, c, v, "archive", "ENA read_run"))

        # --- inherited study identity
        # Take the modal sibling value and record how solid it was, rather than
        # demanding nnanimity. Carter's Tanzania rows split 298 CarterM_2023_TZA
        # to 21 CarterM_2023 -- the inconsistent-folder-name artefact that
        # build_final_metadata.py documents, not a genuine disagreement -- and a
        # unanimity rule left StudyFolder blank on those samples, which matters
        # because StudyFolder drives study exclusion and grouping in 13 modules.
        # Anything below WEAK_AGREEMENT is listed at the end so a thin majority
        # is visible instead of silently accepted.
        for c in INHERIT:
            vals = [x for x in sib[c].tolist() if isinstance(x, str) and x.strip()]
            if not vals:
                continue
            top, n = Counter(vals).most_common(1)[0]
            frac = n / len(vals)
            out[c] = top
            prov.append((sample, c, top, "inherited",
                         f"{basis} ({n}/{len(vals)} = {frac:.0%})"))
            if frac < WEAK_AGREEMENT:
                weak.append((sample, c, top, f"{n}/{len(vals)}"))

        # --- study constants, only when unanimous
        for c in CONSTANT_IF_UNANIMOUS:
            vals = {x for x in sib[c].tolist() if isinstance(x, str) and x.strip()}
            if len(vals) == 1:
                v = vals.pop()
                out[c] = v
                prov.append((sample, c, v, "constant", f"unanimous across {len(sib)} siblings"))

        # --- cohort-level annotations, from a unanimous sibling vote
        flag_cols = [c for c in header if c.startswith(COHORT_FLAG_PREFIX)]
        flag_cols += [c for c in COHORT_FLAG_EXTRA if c in out]
        for c in flag_cols:
            if out.get(c):
                continue
            vals = [x for x in sib[c].tolist() if isinstance(x, str) and x.strip()]
            if len(vals) < MIN_FLAG_SIBLINGS:
                continue
            uniq = set(vals)
            if len(uniq) != 1:
                continue
            v = uniq.pop()
            out[c] = v
            prov.append((sample, c, v, "cohort_flag",
                         f"{basis}, unanimous across {len(vals)} siblings"))

        # --- clinical evidence that IS in the archive
        src = " ".join(str(first.get(k) or "") for k in
                       ("isolation_source", "sample_description", "sample_title"))
        if "worm-infected" in src.lower():
            for c, v in (("disease_Worm-Infected", "Yes"), ("Unhealthy", "Unhealthy"),
                         ("Diseased (all)", "Yes"), ("Diseased (gut and metabolic)", "Yes")):
                if c in out:
                    out[c] = v
                    prov.append((sample, c, v, "archive",
                                 f"isolation_source={first.get('isolation_source')!r}"))

        # --- detailed location, only when ENA gives a sub-national part
        if isinstance(raw_country, str) and ":" in raw_country:
            out["Detailed location"] = raw_country
            prov.append((sample, "Detailed location", raw_country, "archive", "ENA country"))

        added.append([out[c] for c in header])

    print(f"  built {len(added)} added rows")
    if unresolved:
        print(f"  !! {len(unresolved)} could not be built:")
        for u in unresolved[:10]:
            print(f"       {u}")

    # ---- master identity ----------------------------------------------------
    all_rows = data + added
    mids = master_ids(all_rows, idx)
    n_suffixed = sum(1 for m in mids if "." in m and m.rsplit(".", 1)[-1].isdigit())
    dupes = len(mids) - len(set(mids))
    print(f"  {MASTER_COL}: {len(set(mids)):,} distinct over {len(mids):,} rows "
          f"({n_suffixed} needed a disambiguating suffix, {dupes} collisions)")
    if dupes:
        sys.exit(f"  ABORT: {MASTER_COL} is not unique; refusing to write")
    out_header = [MASTER_COL] + header
    out_rows = [[m] + r for m, r in zip(mids, all_rows)]

    # ---- write --------------------------------------------------------------
    n = write_csv_bytes(args.out, [out_header] + out_rows, trailing)
    print(f"\n  wrote {args.out}  ({len(all_rows):,} rows, {n:,} bytes)")

    pv = args.provenance or args.out.with_name(args.out.stem + "_provenance.csv")
    pd.DataFrame(prov, columns=["sample", "column", "value", "source", "evidence"]
                 ).to_csv(pv, index=False)
    print(f"  wrote {pv}  ({len(prov):,} cell-level records)")

    if weak:
        agg = Counter((c, v, n) for _, c, v, n in weak)
        print(f"\n  inherited on a split sibling vote ({len(weak)} cells) "
              f"-- modal value taken, shown for review:")
        for (c, v, n), k in agg.most_common(8):
            print(f"     {c:<22}{v!r:26}{n:>10}   x{k}")

    src_counts = Counter(p[3] for p in prov)
    print(f"\n  provenance by source: {dict(src_counts)}")
    if added:
        filled = sum(1 for v in added[0] if v != "")
        print(f"  a typical added row populates {filled} of {len(header)} columns; "
              f"the rest have no archive source and are left empty")

    if args.matrix_samples and args.matrix_samples.exists():
        have = {l.strip() for l in args.matrix_samples.read_text().split("\n") if l.strip()}
        ids = {row[idx["Chosen_match_ID"]] for row in added}
        missing = sorted(ids - have)
        print(f"\n  added samples absent from the coverage matrices: {len(missing)}")
        for m in missing[:10]:
            print(f"       {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
