"""One data object per figure: everything needed to redraw it and to check it.

Named `Figure_S<n>_data`. The journal calls these Supplementary Figures, so the
identifier is S, not the A the working manifest inherited.

One file per figure, not one per panel. A panel-per-file scheme multiplies:
32 figures with three or twelve panels each is a directory nobody can audit,
and nothing ties the panels of one figure together. The object carries every
panel keyed by name, in draw order, with the reference and detection threshold
it was built at.

    emit(out_dir, "S4", {"Hunter-gatherer": df1, ...}, stem="depth_by_lifestyle")
    obj = load(out_dir, "S4")
    plot_depth_by_lifestyle(obj["panels"], out_dir)

The pickle is what the replot loop reads. `to_workbook()` renders the same
objects as the multi-sheet spreadsheet the journal asks for, so the submitted
data and the data the figures are drawn from are the same thing rather than two
exports that can disagree.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime, timezone

DATA_DIRNAME = "figure_data"
SCHEMA = 1


def _dir_for(out_dir: str) -> str:
    d = os.path.join(out_dir, DATA_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def path_for(out_dir: str, slot: str) -> str:
    return os.path.join(_dir_for(out_dir), f"Figure_{slot}_data.pkl")


def emit(out_dir: str, slot: str, panels: dict, stem: str = "",
         reference: str = "combined", notes: str = "", extras: dict = None
         ) -> str:
    """Write the one data object behind a figure.

    `panels` maps panel name to a tidy DataFrame, in the order they are drawn.
    Everything the figure shows -- including the numbers in its annotations --
    must be derivable from these frames, or the object is incomplete.

    `extras` carries the parts of a figure that are not a table: a panel that
    embeds an already-rendered image, a fitted parameter set, the seed a
    bootstrap ran under. They are kept out of `panels` so the submitted
    spreadsheet stays one sheet per plotted table, but they travel in the same
    pickle, because a figure that cannot be redrawn without them is not
    reproducible from its data object.

    Set `FIGURE_DATA_READONLY` in the environment to make this a no-op. The
    layout and typography tools (`tools/figure_size_sweep.py`,
    `tools/calibrate_figure_fonts.py`) re-invoke the real section entry points
    to measure what they draw, which means they re-emit every data object they
    touch -- with a fresh, empty statistics registry, and with stdout
    redirected so even the log line is swallowed. A panel assembled from that
    registry is silently emptied by such a pass. These are measuring tools;
    they have no business rewriting submitted data, so they set this.
    """
    if os.environ.get("FIGURE_DATA_READONLY"):
        p = os.path.join(out_dir, DATA_DIRNAME, f"Figure_{slot}_data.pkl")
        print(f"    [data] Figure_{slot}_data.pkl left as-is "
              f"(FIGURE_DATA_READONLY)")
        return p
    obj = {
        "schema": SCHEMA,
        "slot": slot,
        "stem": stem,
        "reference": reference,
        "panel_order": list(panels),
        "panels": {k: v.copy() for k, v in panels.items()},
        "extras": dict(extras or {}),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    }
    p = path_for(out_dir, slot)
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)
    rows = sum(len(v) for v in panels.values())
    extra_note = f", {len(obj['extras'])} extra(s)" if obj["extras"] else ""
    print(f"    [data] Figure_{slot}_data.pkl  {len(panels)} panel(s), "
          f"{rows:,} rows{extra_note}")
    return p


def load(out_dir: str, slot: str):
    with open(path_for(out_dir, slot), "rb") as f:
        obj = pickle.load(f)
    # Objects written before extras existed have no such key; give every
    # caller the same shape so none of them has to guess.
    obj.setdefault("extras", {})
    return obj


def to_workbook(objs, out_path: str) -> str:
    """The submitted spreadsheet: one sheet per figure panel.

    Sheet names carry the figure and panel so a reader can tie a sheet to what
    it plots, which is what the journal asks for.
    """
    import pandas as pd
    with pd.ExcelWriter(out_path) as xl:
        index = []
        for obj in objs:
            for name in obj["panel_order"]:
                sheet = f"{obj['slot']}_{name}"[:31]
                obj["panels"][name].to_excel(xl, sheet_name=sheet, index=False)
                index.append(dict(figure=obj["slot"], panel=name, sheet=sheet,
                                  rows=len(obj["panels"][name]),
                                  reference=obj.get("reference", ""),
                                  source_figure=obj.get("stem", "")))
        pd.DataFrame(index).to_excel(xl, sheet_name="index", index=False)
    print(f"  wrote {out_path}  ({len(index)} sheets)")
    return out_path


# ── Which figure is this? ──────────────────────────────────────────────────
_SLOT_CACHE = None


def slot_for(out_path: str):
    """The S-slot for an output path, or None if it is not a shipped figure.

    Several plotting functions serve more than one figure -- the same code
    draws the Carter cohort and the full one, differing only in where it saves
    -- so hard-coding a slot at the save site would be wrong. Ask the manifest
    instead, which is the one place that already knows.
    """
    global _SLOT_CACHE
    if _SLOT_CACHE is None:
        _SLOT_CACHE = {}
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        mf = os.path.join(here, "tools", "additional_figures_manifest.txt")
        try:
            with open(mf) as f:
                for line in f:
                    if line.strip() and not line.startswith("#"):
                        slot, _ref, stem = line.strip().split("|")
                        # Keyed on the last two path components: two figures
                        # share the basename gmw_threshold_sweep and differ
                        # only by directory (the Carter cohort vs the full
                        # one), so a basename key silently gave both the same
                        # slot.
                        _SLOT_CACHE["/".join(stem.split("/")[-2:])] = slot
        except Exception:
            pass
    parts = str(out_path).replace("\\", "/").split("/")
    stem = parts[-1]
    for suffix in (".png", ".pdf", "_notitle"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    key = "/".join(parts[-2:-1] + [stem])
    return _SLOT_CACHE.get(key) or _SLOT_CACHE.get(stem)


def emit_for_path(out_path: str, panels: dict, **kw):
    """Emit only if this output is a shipped figure. Returns the slot or None."""
    slot = slot_for(out_path)
    if not slot:
        return None
    emit(os.path.dirname(str(out_path)), slot, panels,
         stem=os.path.basename(str(out_path)).replace(".png", ""), **kw)
    return slot
