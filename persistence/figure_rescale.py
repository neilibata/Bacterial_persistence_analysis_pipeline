#!/usr/bin/env python3
"""Reduce a finished figure to a printed column width without moving anything.

Science asks for figures at the width they will print (5.7, 12.1 or 18.4 cm)
and at 300 dpi. The main-text figures are authored much wider than that, so on
the page they are reduced -- and everything specified in points is reduced with
them, which is why a 17 pt label prints at 5 pt.

This does that reduction in the file instead of leaving it to the page. Axes
positions are figure fractions, so shrinking the canvas preserves the layout
exactly; what has to move with it is everything specified in points -- type,
rules, markers, tick marks, pads, legend spacing. Multiply all of those by the
same factor and the result is the figure as drawn, photographically reduced.

The layout is therefore untouched by construction: no axes are repositioned, no
text is re-wrapped, nothing is re-fitted. A pixel-for-pixel check that the
reduced figure matches the original downscaled is in tools/verify_rescale.py.

    from persistence.figure_rescale import save_column_width
    save_column_width(fig, "main_text_figures/figure1")   # writes .png + .pdf
"""
from __future__ import annotations

import datetime
import math
import os

import numpy as np

#: Science column widths, inches (5.7 / 12.1 / 18.4 cm)
COLUMN_1_IN = 2.24
COLUMN_2_IN = 4.76
COLUMN_3_IN = 7.24
SUBMISSION_DPI = 300

#: Below this the reduction is doing more harm than the spec is worth; the
#: caller is told rather than the figure being silently re-laid out.
TYPE_FLOOR_PT = 5.0


def _scale_text(artist, k):
    try:
        artist.set_fontsize(artist.get_fontsize() * k)
    except Exception:
        return
    # A text box is padded in points too, and its rule is a line width.
    bbox = getattr(artist, "get_bbox_patch", lambda: None)()
    if bbox is not None:
        try:
            bbox.set_mutation_scale(bbox.get_mutation_scale() * k)
            bbox.set_linewidth(bbox.get_linewidth() * k)
        except Exception:
            pass


def _scale_line(ln, k):
    try:
        ln.set_linewidth(ln.get_linewidth() * k)
    except Exception:
        pass
    try:
        if ln.get_marker() not in (None, "None", "", " ", "none"):
            ln.set_markersize(ln.get_markersize() * k)
            ln.set_markeredgewidth(ln.get_markeredgewidth() * k)
    except Exception:
        pass
    # Dash patterns are in points; leaving them makes a reduced hairline
    # read as a solid rule.
    try:
        offset, seq = ln.get_dashes() if hasattr(ln, "get_dashes") else (None, None)
        if seq:
            ln.set_dashes([max(v * k, 0.1) for v in seq])
    except Exception:
        pass


def _scale_collection(col, k):
    try:
        lw = np.atleast_1d(col.get_linewidths())
        col.set_linewidths([float(v) * k for v in lw])
    except Exception:
        pass
    # scatter sizes are an AREA in points squared, so they take k squared.
    try:
        s = np.atleast_1d(col.get_sizes())
        if len(s):
            col.set_sizes([float(v) * k * k for v in s])
    except Exception:
        pass


def _scale_patch(p, k):
    try:
        p.set_linewidth(p.get_linewidth() * k)
    except Exception:
        pass
    for meth in ("get_mutation_scale", "set_mutation_scale"):
        if not hasattr(p, meth):
            break
    else:
        try:
            p.set_mutation_scale(p.get_mutation_scale() * k)
        except Exception:
            pass


def _scale_title_pad(ax, k):
    """The gap between a panel and its title is in points, like everything else.

    matplotlib keeps it inside a ScaledTranslation rather than as a number, so
    it is read back out of the transform. Missing it leaves the title sitting
    proportionally further from the axes after a reduction, which is the one
    thing that showed up as vertical drift when this was first checked.
    """
    tr = getattr(ax, "titleOffsetTrans", None)
    t = getattr(tr, "_t", None)
    if t is None or not hasattr(ax, "_set_title_offset_trans"):
        return
    try:
        ax._set_title_offset_trans(float(t[1]) * 72.0 * k)
    except Exception:
        pass


def _scale_axis(ax, k):
    for axis in (ax.xaxis, ax.yaxis):
        for which in ("major", "minor"):
            try:
                tk = axis._get_tick(major=(which == "major"))
            except Exception:
                tk = None
            kw = {}
            try:
                t0 = (axis.get_major_ticks() if which == "major"
                      else axis.get_minor_ticks())
                if t0:
                    kw = dict(length=t0[0].tick1line.get_markersize() * k,
                              width=t0[0].tick1line.get_markeredgewidth() * k,
                              pad=t0[0].get_pad() * k)
            except Exception:
                kw = {}
            if kw:
                try:
                    ax.tick_params(axis="x" if axis is ax.xaxis else "y",
                                   which=which, **kw)
                except Exception:
                    pass
        # Axis label padding is in points.
        try:
            axis.labelpad = axis.labelpad * k
        except Exception:
            pass
        # The 1e6-style offset text carries its own size.
        try:
            _scale_text(axis.get_offset_text(), k)
        except Exception:
            pass


def _scale_legend(leg, k):
    for attr in ("borderpad", "labelspacing", "handlelength", "handleheight",
                 "handletextpad", "borderaxespad", "columnspacing"):
        try:
            setattr(leg, attr, getattr(leg, attr) * k)
        except Exception:
            pass
    try:
        leg.get_frame().set_linewidth(leg.get_frame().get_linewidth() * k)
    except Exception:
        pass
    for t in leg.get_texts():
        _scale_text(t, k)
    if leg.get_title() is not None:
        _scale_text(leg.get_title(), k)
    # Legend handles are Line2D/Patch copies and are not reached by the axes
    # walk, so they are scaled here or the key keeps full-size symbols.
    for h in getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])):
        _scale_artist(h, k)


def _scale_artist(a, k):
    import matplotlib.collections as mcoll
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.text as mtext

    if isinstance(a, mtext.Text):
        _scale_text(a, k)
    elif isinstance(a, mlines.Line2D):
        _scale_line(a, k)
    elif isinstance(a, mcoll.Collection):
        _scale_collection(a, k)
    elif isinstance(a, mpatches.Patch):
        _scale_patch(a, k)


def rescale_figure(fig, k):
    """Multiply every point-specified property in the figure by k.

    Positions are untouched: axes sit at figure fractions and data sits in
    data coordinates, so both survive the canvas change unaltered.
    """
    if not math.isfinite(k) or k <= 0:
        raise ValueError(f"bad scale factor {k!r}")

    if getattr(fig, "_suptitle", None) is not None:
        _scale_text(fig._suptitle, k)
    for t in fig.texts:
        _scale_text(t, k)
    for a in fig.lines:
        _scale_line(a, k)
    for a in fig.patches:
        _scale_patch(a, k)
    for leg in getattr(fig, "legends", []):
        _scale_legend(leg, k)

    for ax in fig.get_axes():
        for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
            _scale_text(t, k)
        for t in ax.texts:
            _scale_text(t, k)
        for ln in ax.lines:
            _scale_line(ln, k)
        for col in ax.collections:
            _scale_collection(col, k)
        for p in ax.patches:
            _scale_patch(p, k)
        for im in ax.images:
            try:
                im.set_linewidth(im.get_linewidth() * k)
            except Exception:
                pass
        for sp in ax.spines.values():
            try:
                sp.set_linewidth(sp.get_linewidth() * k)
            except Exception:
                pass
        for tl in (ax.get_xticklabels(which="both")
                   + ax.get_yticklabels(which="both")):
            _scale_text(tl, k)
        _scale_axis(ax, k)
        _scale_title_pad(ax, k)
        leg = ax.get_legend()
        if leg is not None:
            _scale_legend(leg, k)
        # Grid lines carry a width even when the spec says not to use them.
        for gl in (ax.get_xgridlines() + ax.get_ygridlines()):
            _scale_line(gl, k)

    w, h = fig.get_size_inches()
    fig.set_size_inches(w * k, h * k, forward=True)
    return fig


def type_sizes(fig):
    """[(points, role, text)] for every string the figure will print.

    Everything that carries a size, not just the obvious artists: legend
    titles, the exponent an axis parks in its corner, minor tick labels and
    figure-level legends all print at their own size, and each of them has
    been the smallest thing on a figure at some point.
    """
    out = []

    def add(role, t):
        if t is None:
            return
        try:
            s = str(t.get_text())
        except Exception:
            return
        if not s.strip():
            return
        try:
            out.append((float(t.get_fontsize()), role, s.replace("\n", " ")))
        except Exception:
            pass

    def add_legend(leg, where):
        if leg is None:
            return
        add(f"{where} legend title", leg.get_title())
        for t in leg.get_texts():
            add(f"{where} legend", t)

    add("suptitle", getattr(fig, "_suptitle", None))
    for t in fig.texts:
        add("figure text", t)
    for leg in getattr(fig, "legends", []):
        add_legend(leg, "figure")

    for i, ax in enumerate(fig.get_axes()):
        if not ax.get_visible():
            continue
        add(f"ax{i} title", ax.title)
        add(f"ax{i} xlabel", ax.xaxis.label)
        add(f"ax{i} ylabel", ax.yaxis.label)
        for t in ax.texts:
            add(f"ax{i} annotation", t)
        for which in (False, True):
            kind = "minor" if which else "major"
            for t in ax.get_xticklabels(minor=which):
                add(f"ax{i} x{kind} tick", t)
            for t in ax.get_yticklabels(minor=which):
                add(f"ax{i} y{kind} tick", t)
        for axis in (ax.xaxis, ax.yaxis):
            try:
                add(f"ax{i} offset text", axis.get_offset_text())
            except Exception:
                pass
        add_legend(ax.get_legend(), f"ax{i}")
    return out


def smallest_type_pt(fig):
    """The smallest type size left on the figure, in points."""
    sizes = [v for v, _r, _t in type_sizes(fig)]
    return min(sizes) if sizes else float("nan")


def _tight_width_in(fig, **savekw):
    """Width the figure would be saved at, in inches."""
    try:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        bb = fig.get_tightbbox(r)
        pad = float(savekw.get("pad_inches", 0.1) or 0.0)
        return float(bb.width) + 2 * pad
    except Exception:
        return float(fig.get_size_inches()[0])


def scale_pdf_page(path, k, verbose=False):
    """Scale a vector PDF's page uniformly by k, content and all.

    A PostScript `scale` before the page content is an exact affine transform:
    every coordinate, line width and glyph is multiplied by the same number.
    Nothing is re-laid out, because there is no layout engine involved -- it is
    the same drawing on a smaller page. Returns True when the page was scaled.
    """
    import shutil as _sh
    import subprocess
    import tempfile

    if not _sh.which("gs"):
        return False
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096)
        import re as _re
        m = _re.search(rb"/MediaBox\s*\[\s*([\d.+-]+)\s+([\d.+-]+)\s+"
                       rb"([\d.+-]+)\s+([\d.+-]+)", head)
        if not m:
            return False
        x0, y0, x1, y1 = (float(v) for v in m.groups())
        w_pt, h_pt = (x1 - x0) * k, (y1 - y0) * k
    except Exception:
        return False

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tmp = tf.name
    cmd = ["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
           "-dAutoRotatePages=/None",
           f"-dDEVICEWIDTHPOINTS={w_pt:.4f}",
           f"-dDEVICEHEIGHTPOINTS={h_pt:.4f}",
           "-dFIXEDMEDIA", "-dPDFSETTINGS=/prepress",
           f"-sOutputFile={tmp}",
           "-c", f"<</BeginPage{{{k:.8f} {k:.8f} scale}}>> setpagedevice",
           "-f", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0 or not os.path.getsize(tmp):
            os.unlink(tmp)
            return False
        _sh.move(tmp, str(path))
        return True
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


def _record(stem, k, width_in, dpi, min_pt, pdf_scaled):
    """Append the reduction to a CSV beside the figures.

    The printed type size is the one number that decides whether a figure
    meets the spec, and it is only knowable at the moment of the reduction.
    Writing it down means the claim can be checked later without re-running
    the generator, which for some of these is a ten-minute job.
    """
    import csv
    out = os.path.join(os.path.dirname(stem) or ".", "column_width_report.csv")
    name = os.path.basename(stem)
    fields = ["figure", "scale_k", "width_in", "dpi", "min_type_pt",
              "meets_5pt_floor", "pdf_page_scaled", "written"]
    rows = {}
    if os.path.exists(out):
        try:
            with open(out) as fh:
                rows = {r["figure"]: r for r in csv.DictReader(fh)}
        except Exception:
            rows = {}
    rows[name] = dict(
        figure=name, scale_k=f"{k:.4f}",
        width_in=(f"{width_in:.3f}" if width_in else ""),
        dpi=f"{dpi:g}", min_type_pt=f"{min_pt:.2f}",
        meets_5pt_floor=str(bool(min_pt >= TYPE_FLOOR_PT)).lower(),
        pdf_page_scaled=str(bool(pdf_scaled)).lower(),
        written=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    try:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for key in sorted(rows):
                w.writerow({f: rows[key].get(f, "") for f in fields})
    except Exception:
        pass


def adopt_report_row(build_dir, build_stem, out_dir, final_name):
    """Copy a build's reduction row into the final figure's report.

    Figures 3 and 4 are produced under a build name and then copied to
    figureN.png, so their measurements land in the build directory under a
    name nobody looks up. This re-files the row under the name the figure is
    published as, keeping one report for all five.
    """
    import csv
    src = os.path.join(str(build_dir), "column_width_report.csv")
    if not os.path.exists(src):
        return False
    try:
        with open(src) as fh:
            rows = {r["figure"]: r for r in csv.DictReader(fh)}
    except Exception:
        return False
    row = rows.get(build_stem)
    if row is None:
        return False
    dst = os.path.join(str(out_dir), "column_width_report.csv")
    out = {}
    if os.path.exists(dst):
        try:
            with open(dst) as fh:
                out = {r["figure"]: r for r in csv.DictReader(fh)}
        except Exception:
            out = {}
    row = dict(row, figure=final_name)
    out[final_name] = row
    fields = list(row)
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for k in sorted(out):
            w.writerow({f: out[k].get(f, "") for f in fields})
    return True


def save_column_width(fig, stem, target_w=COLUMN_3_IN, dpi=SUBMISSION_DPI,
                      formats=("png", "pdf"), verbose=True, **savekw):
    """Write the figure at `target_w` inches without touching the figure.

    The layout is hand-tuned, so nothing in it is moved, resized or re-fitted.
    What changes is only how large the result is declared to be:

      * the PNG is rendered at exactly ``target_w * dpi`` pixels across -- by
        raising the render resolution, not by altering the drawing -- and its
        metadata then declares ``dpi``, so it prints at ``target_w`` inches at
        that resolution;
      * the PDF is vector, so its page is scaled by the same factor with an
        exact affine transform.

    Because the drawing is untouched, type that measured N points on the
    authored canvas measures ``N * k`` on the page. That is the same reduction
    the journal would apply; doing it here just makes the file say so. Where
    that lands below the 5 pt floor the caller is told, since fixing it means
    changing the layout, which is not this function's business.

    Returns {"k", "width_in", "min_pt_printed", "paths"}.
    """
    # Leave pad_inches to the caller's default: changing it reframes the
    # figure, which is a layout change by another name and showed up as a
    # whole-image mismatch the first time it was tried.
    savekw.setdefault("bbox_inches", "tight")

    w_in = _tight_width_in(fig, **savekw)
    k = target_w / w_in
    target_px = int(round(target_w * dpi))
    render_dpi = target_px / w_in          # = dpi / k

    paths = []
    png_path = f"{stem}.png"
    for fmt in formats:
        pth = f"{stem}.{fmt}"
        os.makedirs(os.path.dirname(pth) or ".", exist_ok=True)
        if fmt == "png":
            fig.savefig(pth, dpi=render_dpi, **savekw)
        else:
            fig.savefig(pth, **savekw)
        paths.append(pth)

    # Stamp the print size on the raster and scale the vector page.
    width_in = None
    if os.path.exists(png_path):
        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(png_path) as im:
                im.load()
                px = im.size[0]
                im.save(png_path, dpi=(dpi, dpi))
            width_in = px / float(dpi)
        except Exception:
            pass
    pdf_path = f"{stem}.pdf"
    pdf_scaled = (scale_pdf_page(pdf_path, k) if "pdf" in formats
                  and os.path.exists(pdf_path) else False)

    _all = sorted(type_sizes(fig))
    min_pt = (_all[0][0] * k) if _all else float("nan")
    _record(stem, k, width_in, dpi, min_pt, pdf_scaled)
    if verbose and _all and min_pt < TYPE_FLOOR_PT:
        print(f"    [type] {os.path.basename(stem)}: strings under the "
              f"{TYPE_FLOOR_PT:g} pt floor --")
        _seen = set()
        for _pt, _role, _txt in _all:
            if _pt * k >= TYPE_FLOOR_PT:
                break
            _key = (round(_pt, 2), _role.split()[-1])
            if _key in _seen:
                continue
            _seen.add(_key)
            print(f"           {_pt * k:.2f} pt  {_role}: {_txt[:44]!r}")
    if verbose:
        bits = [f"x{k:.4f}"]
        if width_in:
            bits.append(f"{width_in:.2f} in at {dpi:g} dpi ({px} px)")
        bits.append(f"smallest type prints at {min_pt:.1f} pt")
        if min_pt == min_pt and min_pt < TYPE_FLOOR_PT:
            bits.append(f"BELOW the {TYPE_FLOOR_PT:g} pt floor")
        if "pdf" in formats:
            bits.append("pdf page scaled" if pdf_scaled else "pdf left at size")
        print(f"    [column] {os.path.basename(stem)}: " + ", ".join(bits))
    return dict(k=k, width_in=width_in, min_pt_printed=min_pt, paths=paths,
                pdf_scaled=pdf_scaled)
