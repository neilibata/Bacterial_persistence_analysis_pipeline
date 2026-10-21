#!/usr/bin/env python3
"""Generate docs/code_map.md: which code produces which published item.

A reader arriving at this repository can see 75 modules under
`persistence/` and 160-odd scripts under `tools/` and have no way to tell
which of them drew Figure S26 or wrote Table S6. This builds that map, and
builds it from the files that already know rather than from a hand-kept list,
so it cannot drift quietly:

  tools/additional_figures_manifest.txt   slot -> the output path it lands at
  tools/replot_from_data.py (DRAW)        slot -> the function that redraws it
  persistence/figure_runner.py (TASKS) task -> the script that first makes it
  tools/paper_manifest.csv                tables and metadata -> source outputs
  tools/audit_artifact_provenance.py      post-run artifacts -> their builders

The drawers in DRAW import the plotting code inside the function body rather
than at module scope, so the module that actually draws a figure is recovered
by reading each drawer's source for its imports. That is why this reports a
drawer AND a module: the drawer is the entry point you call, the module is
where the drawing lives.

    python tools/build_provenance_map.py            # write docs/code_map.md
    python tools/build_provenance_map.py --check    # report gaps, write nothing
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Not `PROVENANCE.md`, which is the obvious name and the wrong one. A
#: case-insensitive filesystem -- this one sets `core.ignorecase = true` --
#: does not distinguish it from a `docs/provenance.md`, so writing
#: `PROVENANCE.md` silently overwrites that file rather than creating a second
#: one. That is how 54 lines of hand-written assembly history were lost once
#: already. The two answer different questions, where the code came from and
#: where each published item comes from, so they get names that cannot
#: collide on any filesystem.
OUT_MD = ROOT / "docs" / "code_map.md"


def _replot():
    spec = importlib.util.spec_from_file_location(
        "replot", ROOT / "tools" / "replot_from_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: `from persistence.section3_health import ...`, `from persistence import x`,
#: `from figures.foo import ...` -- the deferred imports inside a drawer body.
_IMP = re.compile(
    r"from\s+((?:persistence|figures|tools)(?:\.[A-Za-z_][\w]*)*)\s+import\s+([^\n#]+)")

#: Five drawers reach a standalone figure script by bare name, after putting
#: its directory on sys.path -- `from make_read_accounting import draw`. The
#: name alone does not say where the file is, so resolve it against the three
#: places a script can live.
_BARE = re.compile(r"^\s*from\s+([A-Za-z_]\w*)\s+import\s", re.M)

#: Search order matters, and it is the runtime's, not a convenient one. Each
#: of these drawers does `sys.path.insert(0, root/"figures")` or
#: `root/"tools"` before importing, so those directories shadow the
#: repository root. Searching the root first named the stale root duplicates
#: of figure3.py, make_combined_figure3.py and make_accumulation_sinf_figures
#: -- copies that are months behind, one of which does not even import -- as
#: the producers of S15, S16 and S17, which is the opposite of the truth.
_SEARCH_DIRS = ("figures", "tools", "")


def _resolve_bare(name: str) -> str | None:
    for d in _SEARCH_DIRS:
        p = (ROOT / d / f"{name}.py") if d else (ROOT / f"{name}.py")
        if p.is_file():
            return str(p.relative_to(ROOT))
    return None


def drawer_modules(fn) -> list[str]:
    """The modules a drawer imports from, in source order."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return []
    seen, out = set(), []

    def add(path):
        if path and path not in seen:
            seen.add(path)
            out.append(path)

    for mod, names in _IMP.findall(src):
        # `from persistence import figure_source` names the module in the
        # import list, not in the dotted path.
        if mod in ("persistence", "figures", "tools"):
            # One item per comma, and the module is that item's first token:
            # `health_driver_family_panels as hdp` names one module, not three.
            # Splitting the whole list on whitespace instead put the keyword
            # `as` and every alias in the map, so the code map credited five
            # figure slots to persistence/as.py, persistence/m.py and
            # persistence/hdp.py -- files a reader greps for and never finds.
            for item in names.replace("(", " ").replace(")", " ").split(","):
                tok = item.split()
                if tok and tok[0].isidentifier():
                    add(f"{mod}/{tok[0]}.py")
        else:
            add(mod.replace(".", "/") + ".py")
    for name in _BARE.findall(src):
        add(_resolve_bare(name))
    return out


def figure_rows():
    """One row per supplementary figure slot."""
    from tools import figure_sets
    rp = _replot()
    man = {s: (r, stem) for s, r, stem in figure_sets.manifest()}
    rows = []
    for slot in sorted(man, key=lambda s: int(s.lstrip("Ss"))):
        ref, stem = man[slot]
        fn = rp.DRAW.get(slot)
        mods = drawer_modules(fn) if fn else []
        rows.append({
            "slot": slot,
            "output": f"results/{ref}/{stem}.png",
            "drawer": getattr(fn, "__name__", "—") if fn else "— none —",
            "modules": mods,
        })
    return rows


def task_rows():
    from persistence import figure_runner as fr
    return [{"task": t.name, "script": t.script, "what": t.description}
            for t in (fr.TASKS[k] for k in sorted(fr.TASKS))]


def manifest_rows(categories):
    rows = []
    p = ROOT / "tools" / "paper_manifest.csv"
    for line in p.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 6 or parts[0] == "item_id":
            continue
        if parts[1] in categories:
            rows.append({"item": parts[0], "source": parts[3],
                         "what": parts[5]})
    return rows


def declared_rows():
    spec = importlib.util.spec_from_file_location(
        "prov", ROOT / "tools" / "audit_artifact_provenance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    seen, rows = set(), []
    for name, who in sorted(mod.DECLARED.items()):
        if name in seen:
            continue
        seen.add(name)
        rows.append({"item": name, "built_by": who})
    return rows


def render(figs, tasks, tables, metadata, declared) -> str:
    L = []
    w = L.append
    w("# Where things come from\n")
    w("<!-- Generated by tools/build_provenance_map.py. Do not edit by hand:")
    w("     rebuild it instead, so it stays tied to the manifest and the")
    w("     task table rather than drifting from them. -->\n")
    w("Which code produces which published item. Everything here is read out")
    w("of the files that already define it, so this map cannot disagree with")
    w("the pipeline without the generator disagreeing too.\n")

    w("## Running the pipeline\n")
    w("```")
    w("python -m persistence --all --resume        # the whole analysis")
    w("python -m persistence --ref combined --only s3   # one section")
    w("python tools/replot_from_data.py S26 --results-root RUN/results")
    w("```\n")
    w("`persistence/` is the analysis package: `__main__.py` is the entry")
    w("point and dispatches to the section modules below. `tools/` holds")
    w("standalone scripts run before or after it -- builders, audits and")
    w("one-off checks. `figures/` holds the main-text figure scripts.\n")

    w("## Supplementary figures\n")
    w("`drawer` is the function `tools/replot_from_data.py` calls to redraw a")
    w("figure from its stored data object; `drawn by` is where the plotting")
    w("code actually lives.\n")
    w("| Slot | Output | Drawer | Drawn by |")
    w("| --- | --- | --- | --- |")
    for r in figs:
        mods = "<br>".join(f"`{m}`" for m in r["modules"]) or "—"
        w(f"| **{r['slot']}** | `{r['output']}` | `{r['drawer']}` | {mods} |")
    w("")

    w("## Figure tasks\n")
    w("What `python -m persistence --figure full` runs, and what each")
    w("task is for. These produce a figure the first time; the table above")
    w("is how to redraw one afterwards.\n")
    w("| Task | Script | Produces |")
    w("| --- | --- | --- |")
    for r in tasks:
        w(f"| `{r['task']}` | `{r['script']}` | {r['what']} |")
    w("")

    w("## Supplementary tables\n")
    w("| Item | Built from | What it is |")
    w("| --- | --- | --- |")
    for r in tables:
        src = f"`{r['source']}`" if r["source"] else "—"
        w(f"| {r['item']} | {src} | {r['what']} |")
    w("")

    w("## Metadata tables\n")
    w("| Item | Built from | What it is |")
    w("| --- | --- | --- |")
    for r in metadata:
        src = f"`{r['source']}`" if r["source"] else "—"
        w(f"| {r['item']} | {src} | {r['what']} |")
    w("")

    w("## Post-run artifacts\n")
    w("Built by standalone tools after the run, not by")
    w("`python -m persistence`.\n")
    w("| Item | Built by |")
    w("| --- | --- |")
    for r in declared:
        w(f"| `{r['item']}` | `{r['built_by']}` |")
    w("")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="report slots with no drawer or no module, write nothing")
    a = ap.parse_args()

    figs = figure_rows()
    tasks = task_rows()
    tables = manifest_rows({"supp_table"})
    metadata = manifest_rows({"metadata"})
    declared = declared_rows()

    gaps = [r["slot"] for r in figs if not r["modules"]]
    print(f"  {len(figs)} figure slots, {len(tasks)} figure tasks, "
          f"{len(tables)} tables, {len(metadata)} metadata items, "
          f"{len(declared)} post-run artifacts")
    if gaps:
        print(f"  [WARN] {len(gaps)} slot(s) whose drawer names no module: "
              + ", ".join(gaps))
    if a.check:
        return 1 if gaps else 0

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render(figs, tasks, tables, metadata, declared))
    print(f"  wrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
