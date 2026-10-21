"""
Pipeline-wide plotting utilities.

Figure sizing policy
====================
Every figure is authored at the width it will be printed at. That is the whole
policy, and it replaces the medium/large preset pair and the auto-fitting layer
that used to sit here.

The reason is arithmetic. A figure authored W inches wide and placed in the
manuscript at PAGE_WIDTH_IN is scaled by PAGE_WIDTH_IN / W, and every point size
in it scales with the canvas. A figure drawn 30 inches wide therefore needs ~55 pt
type to arrive at 12 pt on the page, and type that size consumes the canvas the
panels were supposed to occupy. Measuring the shipped figures showed exactly that:
the only ones that came out legible were the three authored near page width, and
the starved ones were all authored at three to six times page width.

So: author at page width, set type in real points, and give whatever is left to
the panels. Graphics get the space first; text gets as large as it can without
intruding on them. Never shrink a panel to make a label fit -- cut the label.

Use page_figsize() to get a figure size that obeys this, and check_page_width()
warns when a figure is authored wide enough for the old failure to recur.

savefig_dual() writes two files:
    <stem>.png
    <stem>_notitle.png   (figure-level suptitles hidden, for in-document use)

The name is kept because ~130 call sites use it; nothing reads its return value.
"""

import os
import matplotlib as _mpl
import matplotlib.pyplot as _plt

# ── Page geometry and type scale ───────────────────────────────────────────
# Science prints at three fixed widths: 5.7 cm (1 column), 12.1 cm (2 columns)
# and 18.4 cm (3 columns). Figures are authored at the width they will print
# at, so nominal point size and printed point size are the same number and the
# 7 pt target can be checked by reading the code rather than measuring the PNG.
COLUMN_1_IN: float = 2.24      # 5.7 cm
COLUMN_2_IN: float = 4.76      # 12.1 cm
COLUMN_3_IN: float = 7.24      # 18.4 cm
PAGE_WIDTH_IN: float = COLUMN_3_IN

# "please save images at a resolution of 300 dots per inch (dpi) for
# initial submission". Figures were being written at 180, 220, 300 and
# 360; the three below 300 were not submittable.
SUBMISSION_DPI: int = 300

# Science asks for the figure's title in the caption, not in the figure. Set
# False only to inspect a figure with its title attached; the shipped output
# never has one, and the text is recorded in tools/caption_debt.csv instead.
PUBLICATION: bool = True

# Base type size, in points, as printed.
#
# Measured, not chosen: at 10 pt the median printed text lands at 8.5 pt, which
# is just above the 7.4-8.5 pt band of the figures that were already legible,
# and leaves graphics ink at 3.8-4.3x text ink. Pushing to 12 pt buys 2.5 pt of
# type and costs nearly half the graphics share (down to 2.2-2.6x) -- text grows
# into the space the panels need. An earlier pass targeted 12 pt printed and
# that is what produced the colliding, squeezed figures it was meant to fix.
#
# Ticks and annotations step down from here.
# Science: "a reduced size of about 7 points (2.5 mm) high, and not smaller
# than 5 points". Authoring at the printed width makes this literal -- 7 here
# is 7 on the page. Ticks and annotations step down from here, so the floor
# matters: the step-down must not take anything below 5.
BASE_FS: int = 7
MIN_FS:  int = 5          # Science hard floor; nothing ships below this

# Figure code sets explicit sizes as FS + offset. Differences within a figure
# are wanted -- an in-axes note should not compete with a panel title -- but the
# range has to stay inside what survives placement. FS-5.5 used to arrive at
# 4.6 pt on a shrunk canvas, and FS+6 was part of what caused the shrink.
# Prefer the rcParams above to an explicit size: they already encode the roles.
FS_OFFSET_MIN: float = -3.0
FS_OFFSET_MAX: float = +2.0

# Authoring much wider than the page reintroduces the shrink this policy exists
# to remove. Beyond this multiple, check_page_width() complains.
WIDTH_WARN_RATIO: float = 1.35


# ── Type size as a function of figure complexity ───────────────────────────
# A single panel at page width can carry 11 pt comfortably; a 3x3 grid of the
# same width gives each panel a third of the room and cannot. Holding one base
# size across both is what forced the choice between illegible small panels and
# colliding large ones -- the right size depends on how much the figure shows.
#
# Small text in a dense figure is not a defect. The main-text figures read well
# at sizes that would look cramped on a single-panel plot, because the reader is
# closer to a panel that carries less.
PANEL_FS = (
    (1,  11.0),
    (2,  10.5),
    (4,  10.0),
    (6,   9.0),
    (9,   8.5),
    (12,  8.0),
)
PANEL_FS_FLOOR = 7.5


def fs_for_panels(n_panels: int) -> float:
    """Base type size for a figure with this many panels, at page width."""
    for cap, fs in PANEL_FS:
        if n_panels <= cap:
            return fs
    return PANEL_FS_FLOOR


def style_for_figure(fig=None, n_panels: int = None) -> float:
    """Set the type scale from a figure's own panel count. Returns the base size.

    Call once the axes exist and before drawing text:

        fig, axes = plt.subplots(3, 3, figsize=page_figsize(n_cols=3, n_rows=3))
        FS = style_for_figure(fig)
    """
    if n_panels is None:
        n_panels = len([a for a in (fig.axes if fig is not None else [])
                        if a.get_visible()]) or 1
    fs = fs_for_panels(n_panels)
    apply_oi_style(base=fs)
    return fs


# Explicit figure margins, in figure fractions. These exist because
# `tight_layout` sizes the margins to fit the text, so the axes get whatever is
# left: on a 2x3 it hands the panels 46% of the canvas at 12 pt type and 58% at
# 9 pt. Fixing the margins instead holds the plot area constant whatever the
# type size -- 59% on a 2x3, 67% on a single row -- and lets font size be a
# typographic decision rather than a layout one.
MARGINS = {
    "left": 0.085, "right": 0.985, "top": 0.93, "bottom": 0.105,
    "wspace": 0.16, "hspace": 0.28,
}
# A figure with a suptitle needs the top margin pulled down to clear it.
MARGIN_TOP_WITH_SUPTITLE = 0.88
# A figure-level legend needs its own band. Squeezing it into the panels is
# what puts it on the data; giving it margin costs a little height and nothing
# else.
MARGIN_BOTTOM_WITH_FIGLEGEND = 0.17
# Where the top margin goes once strip_title_for_publication() has taken the
# suptitle off: a hair of room so ascenders on the top row are not clipped.
MARGIN_TOP_TITLE_STRIPPED = 0.975


def drawn_margins(margins: dict = None, titled: bool = True) -> dict:
    """The margins a figure is actually saved with, for sizing it up front.

    A figure authored with a suptitle reserves a top margin for it, and the
    publication path then strips the title and raises that margin. A canvas
    sized from the authored margins (square_figsize) is therefore too tall for
    panels with a fixed box aspect, and the surplus lands as blank bands
    between the rows. Pass the result to square_figsize instead.
    """
    m = dict(margins or {})
    if titled and PUBLICATION:
        m["top"] = max(float(m.get("top", MARGINS["top"])),
                       MARGIN_TOP_TITLE_STRIPPED)
    return m


def add_post_layout_hook(fig, fn) -> None:
    """Register a callable to run at the end of savefig_dual's fitting pass.

    fit_tick_labels() and place_legends_clear() both change how much room text
    takes, so a figure that positions an axes *from* where its own labels land
    has to be told again once those have settled. Registering the placement
    here is the difference between aligning to the labels as authored and
    aligning to the labels as saved.
    """
    fig._post_layout_hooks = list(getattr(fig, "_post_layout_hooks", ())) + [fn]


def tighten_margins(fig, **overrides) -> None:
    """Give the canvas to the axes rather than to the margins.

    Set `fig._skip_tighten_margins = True` on a figure whose layout is managed
    some other way (a GridSpec with explicit ratios, an inset, a colorbar).
    Skipping this pass is rarely what a dense grid wants, though: the shared
    margins are what crops the canvas down to the panels, and a figure that
    opts out keeps every inch of the whitespace its GridSpec defaults leave.
    Prefer `fig._margin_overrides = {...}` -- same pass, this figure's numbers.
    """
    if getattr(fig, "_skip_tighten_margins", False):
        return
    m = dict(MARGINS)
    m.update(getattr(fig, "_margin_overrides", None) or {})
    sup = getattr(fig, "_suptitle", None)
    if sup is not None:
        # Reserve what the title needs, not a fixed fraction. 0.88 is about
        # right on a 4 in figure and leaves two and a half inches of white on a
        # 22 in one, which is where the band of empty space above the top row
        # came from.
        try:
            fig.canvas.draw()
            h_in = sup.get_window_extent(
                renderer=fig.canvas.get_renderer()).height / fig.dpi
            need = h_in / max(float(fig.get_size_inches()[1]), 1e-6)
            m["top"] = min(m["top"], 1.0 - (need * 2.2))
        except Exception:
            m["top"] = min(m["top"], MARGIN_TOP_WITH_SUPTITLE)
    m.update(overrides)
    try:
        fig.subplots_adjust(**m)
    except Exception:
        pass


# The size apply_oi_style() last applied. current_fs() returned the BASE_FS
# constant regardless of what was applied, so every explicit `fontsize=FS-1` in
# the figure code stayed at the default however the style was set -- font sweeps
# moved only the rcParam-driven text and left the rest untouched.
_ACTIVE_FS: float = float(BASE_FS)
# Has any caller asked for a specific size? Import-time calls must not overwrite
# one that has.
_EXPLICIT_FS: bool = False


def current_fs() -> float:
    """Base font size, in points, as last applied by apply_oi_style()."""
    return _ACTIVE_FS


# There is deliberately no height limit. These figures are placed in a Word
# document at a fixed width, and height costs nothing but page space -- a tall
# figure is scrolled, not shrunk. An earlier version of this file clamped height
# to a page, which silently forced every multi-row grid into landscape and left
# the panels squat; squat panels waste the vertical room that lets type grow.
# Height follows from the content.

# Below about 1.9 in a panel cannot carry an axis and a readable tick label, so
# three columns is the most a 6.3 in page takes. More panels go to more rows.
MAX_PANEL_COLS: int = 3

# Panel aspect (width / height), measured on figures carrying realistic content
# (per-panel titles and annotations, not bare axes). Titles and axis furniture
# occupy a fixed band of height whatever the panel size, so taller panels spend
# a smaller share of the canvas on text. That trend does not stop at a sensible
# layout, though: a three-panel row keeps improving on the graphics-to-text
# ratio right down to panels 2 in wide and 8 in tall, which is a better number
# and a worse figure. These are the tallest panels that still read as the shape
# the data wants, and the ratio is allowed to be what it is at that shape.
#
# The corollary: on a dense multi-panel figure, a low graphics-to-text ratio is
# a content problem, not a geometry one. Nine per-panel titles are a lot of text
# at any size, and the fix is to cut them, not to stretch the canvas.
ASPECT_SINGLE: float = 1.30
ASPECT_GRID:   float = 0.80
ASPECT_ROW:    float = 0.70
# A single row of panels is sized by the height the whole strip should occupy,
# not by a fixed panel aspect: holding the aspect makes two wide panels portrait
# and five narrow ones a sliver. A floor keeps a five-across strip from becoming
# a sliver; there is no ceiling, for the reason above.
ROW_HEIGHT_MIN: float = 2.6


#: Where a richness panel's y-axis label sits, in axes fractions. Fixed rather
#: than left to matplotlib, which places the label outside the tick labels: a
#: panel whose top tick reads 1200 pushes its label further left than one
#: reading 900, and across a grid that reads as a wobble.
RICHNESS_YLABEL_X = -0.20


def pin_ylabel(ax, text, x: float = RICHNESS_YLABEL_X, **kw):
    """Set a y-axis label at a fixed horizontal position.

    Use where the same label appears on several panels whose tick labels are
    different widths, so the labels line up down the figure instead of
    drifting with the number of digits beside them.
    """
    ax.set_ylabel(text, **kw)
    if str(text).strip():
        ax.yaxis.set_label_coords(x, 0.5)
    return ax


def square_panels(*axes):
    """Give each axes a 1:1 plotting box.

    The box, not the data: these panels carry different units on the two axes,
    so `set_aspect` would be meaningless, while a square box makes the panels
    comparable by eye across a grid.
    """
    for ax in axes:
        if ax is None:
            continue
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass


def page_figsize(aspect: float = None, width_in: float = None,
                 n_cols: int = 1, n_rows: int = 1,
                 height_in: float = None,
                 max_height_in: float = None) -> tuple:
    """
    A figure size at page width, shaped by `aspect` (width / height).

    Panels are what the height is for, so a grid gets taller rather than the
    figure getting wider: width is pinned to the page, and each row is given
    the height one panel of the requested aspect needs.

        page_figsize()                        -> a single panel
        page_figsize(n_cols=3)                -> a one-row strip of three
        page_figsize(n_cols=3, n_rows=3)      -> a 3x3 grid

    Omit `aspect` to get the measured default for that shape.
    """
    w = float(width_in if width_in is not None else PAGE_WIDTH_IN)
    panel_w = w / max(n_cols, 1)
    if aspect is None:
        if n_rows == 1 and n_cols > 1:
            h = max(panel_w / ASPECT_ROW, ROW_HEIGHT_MIN)
            aspect = None
        else:
            # Aspect follows the panel's width, not the grid's shape. A grid
            # panel is narrow, so it wants to be taller than wide; a full-width
            # panel in a single column does not, and applying the grid aspect to
            # it produced a 23 in figure for three stacked plots.
            aspect = ASPECT_SINGLE if n_cols == 1 else ASPECT_GRID
            h = (panel_w / aspect) * max(n_rows, 1)
    else:
        h = (panel_w / max(aspect, 1e-6)) * max(n_rows, 1)
    if height_in is not None:
        h = float(height_in)
    # A figure taller than the page has to be shrunk to fit it, which is the
    # same defeat as authoring too wide -- just on the other axis.
    # Only clamped if the caller explicitly asks for a ceiling.
    if max_height_in is not None:
        h = min(h, float(max_height_in))
    return (round(w, 2), round(h, 2))


def panel_grid(n_panels: int, max_cols: int = None) -> tuple:
    """
    (n_rows, n_cols) for n_panels, wrapped so no panel is narrower than usable.

    Laying seven conditions out as one row of seven is what produced the
    36-inch canvases: at page width those panels are 0.9 in across, so the code
    widened the figure instead. Wrapping to a grid keeps the page fixed and
    spends height, which a document can scroll and a page cannot widen.
    """
    n = max(int(n_panels), 1)
    cols = max(1, min(n, int(max_cols if max_cols is not None else MAX_PANEL_COLS)))
    rows = -(-n // cols)                      # ceil
    return rows, cols


def square_figsize(n_cols: int = 1, n_rows: int = 1,
                   width_in: float = None, margins: dict = None,
                   extra_h_in: float = 0.0) -> tuple:
    """Page-width figure whose panels come out square *as drawn*.

    A panel taller than it is wide spends its height on the gap between the
    data and the frame, and its title has to reach across a narrow column,
    which is where the title-into-title overlaps come from.

    The panel is what has to be square, not the canvas, and the two are not the
    same: the margins and the inter-panel gutters take their cut first. Dividing
    the page width by the column count and calling that the height ignores all
    of it -- measured, a three-across row sized that way came out at 0.73, still
    visibly portrait. Solve for the height that makes the drawn panel square:

        panel_w = W (right - left) / (cols + (cols-1) wspace)
        H       = panel_w (rows + (rows-1) hspace) / (top - bottom)

    Pass the same `margins` the figure will be drawn with (its
    `_margin_overrides`), or the shared defaults are assumed.
    """
    m = dict(MARGINS)
    m.update(margins or {})
    w = float(width_in if width_in is not None else PAGE_WIDTH_IN)
    cols, rows = max(int(n_cols), 1), max(int(n_rows), 1)
    axes_w = w * (m["right"] - m["left"])
    panel_w = axes_w / (cols + (cols - 1) * m["wspace"])
    axes_h = panel_w * (rows + (rows - 1) * m["hspace"])
    h = axes_h / max(m["top"] - m["bottom"], 1e-6)
    return (round(w, 2), round(h + float(extra_h_in), 2))


def label_pad(scale: float = 0.55) -> float:
    """Axis-label padding, in points, proportional to the current type scale.

    A constant pad is a constant only in points: written for an 8.5 pt figure
    it is a third of the right gap on a 26 pt one, and the axis label lands back
    on its own tick labels. Scale it with the type and it holds everywhere.
    """
    return max(6.0, current_fs() * float(scale))


def plain_axes(fig, x_ticks: int = 4, y_ticks: int = 4, axes=None,
               prune: str = None) -> None:
    """Drop the gridlines and cap the tick count on every axes of a figure.

    Both are ink that competes with the data rather than supporting it: a
    gridline crossing a marker is one more line to read, and a tick every
    interval puts six numbers down an axis where three carry the scale. Removing
    them buys more legibility than growing the type does, and buys headroom for
    growing it afterwards.

    apply_oi_style() turns grids on globally, so this has to run on the axes
    after they exist rather than in the rcParams.

    `prune` is passed to the locator. "lower" drops the first tick on each
    axis, which is what stops the x and y labels meeting at the origin corner
    -- two "0.00" strings a few pixels apart, on every panel of a grid.
    """
    from matplotlib.ticker import MaxNLocator
    for ax in (axes if axes is not None else fig.axes):
        try:
            ax.grid(False)
            # MaxNLocator picks evenly spaced values, which is meaningless on a
            # log axis: applied to one it replaced the decade ticks with a
            # couple of arbitrary numbers, and a density plot came out with a
            # single label on its y axis and no readable scale. Leave any
            # non-linear axis to its own locator.
            if (x_ticks and ax.get_xscale() == "linear" and not isinstance(
                    ax.xaxis.get_major_locator(), _mpl.ticker.FixedLocator)):
                ax.xaxis.set_major_locator(
                    MaxNLocator(nbins=x_ticks, prune=prune))
            if (y_ticks and ax.get_yscale() == "linear" and not isinstance(
                    ax.yaxis.get_major_locator(), _mpl.ticker.FixedLocator)):
                ax.yaxis.set_major_locator(
                    MaxNLocator(nbins=y_ticks, prune=prune))
        except Exception:
            continue


def check_page_width(fig, name: str = "") -> bool:
    """Warn when a figure is authored wide enough to reintroduce the shrink."""
    w = float(fig.get_size_inches()[0])
    # A figure may be authored wide on purpose, with its type scaled to match,
    # so that its panels are usable. That is a decision, not a mistake.
    if getattr(fig, "_wide_on_purpose", False):
        return True
    if w > PAGE_WIDTH_IN * WIDTH_WARN_RATIO:
        shrink = w / PAGE_WIDTH_IN
        print(f"    [size] {name or 'figure'} authored {w:.1f} in for a "
              f"{PAGE_WIDTH_IN} in column: text will print at 1/{shrink:.1f} of "
              f"its nominal size. Use page_figsize().")
        return False
    return True


# ── Pipeline-wide style ────────────────────────────────────────────────────

def apply_oi_style(preset: str = None, base: int = None) -> None:
    """
    Okabe-Ito colour cycle plus the printed type scale.

    Called once at module import in each section file. `preset` is accepted and
    ignored: it selected the old medium/large pair, which no longer exists.
    Pass `base` to shift the whole scale for a figure that carries unusual
    amounts of text.

    A call with no `base` establishes the default *only if nothing has asked for
    a size yet*. Fourteen modules make that call at import, and a lazy import of
    any of them part-way through a build used to reset the whole scale under the
    figure being drawn -- silently, since the call that set the size and the call
    that undid it live in different files. The figure then came out with default
    ticks and axis labels while every explicit `fontsize=FS-n`, captured before
    the reset, stayed large. To go back to the default deliberately, ask for it:
    `apply_oi_style(base=BASE_FS)`.
    """
    from . import config as _cfg
    global _ACTIVE_FS, _EXPLICIT_FS
    if base is None and _EXPLICIT_FS:
        return
    fs = float(base if base is not None else BASE_FS)
    _ACTIVE_FS = fs
    _EXPLICIT_FS = base is not None
    oi_cycle = [_cfg.OI[k] for k in
                ("orange", "sky", "green", "yellow", "blue",
                 "vermillion", "pink", "black")]
    _mpl.rcParams.update({
        "axes.prop_cycle":   _mpl.cycler(color=oi_cycle),
        # A real hierarchy, not a flat scale. Tick labels are read by glancing
        # at a position, not by reading a word, so they can sit well below the
        # body size; giving them near-body size spends canvas on digits and is
        # what crowds dense panels. Titles and axis labels carry the meaning and
        # stay large. Varying size within a figure is deliberate.
        "font.size":         fs,
        "figure.titlesize":  fs + 2,          # 12 at base 10
        "axes.titlesize":    fs,              # 10
        "axes.labelsize":    max(fs - 1, MIN_FS),      # 9
        # Weight is deliberately NOT part of the hierarchy here. Bolding titles
        # and axis labels via rcParams turns most of the text in a ten-panel
        # figure bold, and bold only reads as emphasis while it stays rare --
        # so the panel letters carry it on their own and everything else is
        # regular. Size and position do the rest of the work.
        "legend.fontsize":   max(fs - 2.5, MIN_FS),    # 7.5
        "xtick.labelsize":   max(fs - 3, MIN_FS),      # 7
        "ytick.labelsize":   max(fs - 3, MIN_FS),      # 7
        # An axis label belongs to the ticks beside it. matplotlib's default
        # gap leaves it drifting between panels, close enough to the neighbour
        # to look like its label; pulling it in ties it to its own axis.
        "axes.labelpad":     2.0,
        "xtick.major.pad":   2.0,
        "ytick.major.pad":   2.0,
        "axes.grid":         True,
        "grid.alpha":        0.2,
        "figure.autolayout": False,
        # Panels first. These are the margins applied by tighten_margins(); the
        # rcParams matter for figures that never call it.
        "figure.subplot.left":   MARGINS["left"],
        "figure.subplot.right":  MARGINS["right"],
        "figure.subplot.top":    MARGINS["top"],
        "figure.subplot.bottom": MARGINS["bottom"],
        "figure.subplot.wspace": MARGINS["wspace"],
        "figure.subplot.hspace": MARGINS["hspace"],
        # A list here rather than "sans-serif" enables matplotlib's per-glyph
        # font fallback (>=3.6). A single family renders any character it lacks
        # -- U+2192 arrow, U+25CB circle, U+221D proportional-to -- as a tofu
        # box with only a warning. Listing families lets those glyphs fall
        # through to DejaVu Sans while everything else keeps the first family.
        #
        # Arial leads, not Helvetica, and that is the whole reason fontweight
        # works. macOS ships Helvetica as a .ttc collection; matplotlib cannot
        # select a face inside one, so it returns Helvetica.ttc for every weight
        # and `fontweight="bold"` renders as regular -- identical ink, no
        # warning. Measured on this machine: Helvetica bold/regular ink ratio
        # 1.00, Arial 1.28. Arial is metrically compatible (a 40-character
        # panel title differs by 0.4% in width), so leading with it makes bold
        # real without moving any layout.
        "font.family":       ["Arial", "Helvetica", "DejaVu Sans"],
        "font.sans-serif":   ["Arial", "Helvetica",
                              "DejaVu Sans", "Liberation Sans", "FreeSans"],
    })


def share_axis_labels(fig, xlabel: str = None, ylabel: str = None,
                      axes=None) -> None:
    """
    Replace per-panel axis labels with one label for the whole figure.

    This is the single largest win available on a multi-panel figure. Nine
    panels each captioned "Total mapped bases (Gb)" say one thing nine times,
    and measured on a 3x3 that repetition costs about four fifths of the
    graphics share: dropping to one shared pair took graphics ink from 1.1x
    the text to 6.0x, and cut 188 text runs to 27, with no information lost.

    Call it after the panels are drawn. Pass `axes` to limit it to a subset.
    """
    for ax in (axes if axes is not None else fig.axes):
        ax.set_xlabel("")
        ax.set_ylabel("")
    sx = fig.supxlabel(xlabel) if xlabel else None
    sy = fig.supylabel(ylabel) if ylabel else None

    def _clear_of_ticks(_fig=fig, _sx=sx, _sy=sy):
        """Put the shared labels outside the tick labels, not on top of them.

        matplotlib pins supxlabel and supylabel to fixed figure fractions, which
        knows nothing about where the tick labels ended up. On a short figure
        the shared x label lands squarely on them -- measured at 100% overlap,
        the axis label completely covering the numbers it belongs to. Measure
        the ticks and sit just beyond them.
        """
        try:
            _fig.canvas.draw()
            _r = _fig.canvas.get_renderer()
            _h, _w = _fig.bbox.height, _fig.bbox.width
            if _sx is not None:
                lo = min((t.get_window_extent(renderer=_r).y0
                          for a in _fig.axes for t in a.get_xticklabels()
                          if t.get_visible() and t.get_text().strip()),
                         default=None)
                if lo is not None:
                    pad = _sx.get_window_extent(renderer=_r).height * 0.45
                    _sx.set_y(max(0.004, (lo - pad) / _h))
                    _sx.set_va("top")
            if _sy is not None:
                lf = min((t.get_window_extent(renderer=_r).x0
                          for a in _fig.axes for t in a.get_yticklabels()
                          if t.get_visible() and t.get_text().strip()),
                         default=None)
                if lf is not None:
                    pad = _sy.get_window_extent(renderer=_r).width * 0.45
                    _sy.set_x(max(0.004, (lf - pad) / _w))
                    _sy.set_ha("right")
        except Exception as exc:
            print(f"    [labels] could not place shared labels: {exc}")

    add_post_layout_hook(fig, _clear_of_ticks)


# ── Text artists ───────────────────────────────────────────────────────────

def _collect_text_artists(fig):
    """Return every Text artist in a figure including tick labels."""
    artists = list(fig.texts)
    for ax in fig.get_axes():
        for obj in (ax.title, ax.xaxis.label, ax.yaxis.label):
            artists.append(obj)
        # A twinx shares its x axis with the parent and matplotlib hides that
        # axis rather than deleting its tick labels, so the labels stay on the
        # artist list, still flagged visible, sitting exactly on the parent's.
        # Counting them reported every twinned figure as having its x labels
        # perfectly overlapped by an invisible copy of itself.
        if ax.xaxis.get_visible():
            artists.extend(ax.get_xticklabels(which="both"))
        if ax.yaxis.get_visible():
            artists.extend(ax.get_yticklabels(which="both"))
        artists.extend(ax.texts)
        leg = ax.get_legend()
        if leg is not None:
            artists.extend(leg.get_texts())
            t = leg.get_title()
            if t is not None:
                artists.append(t)
    return [a for a in artists if hasattr(a, "get_fontsize")]


def apply_display_text(fig):
    """Normalize publication display labels on every text artist in a figure.

    Tick labels need the formatter rewritten, not the Text artist. A tick's
    text is regenerated from its formatter on every draw, so setting it here
    was silently undone -- which is why "H-G" and "Transitional" survived to
    the page on exactly those figures that carry the lifestyle as a tick label,
    and nowhere else. Rewriting the FixedFormatter's sequence makes it stick.

    Legend entries need the HANDLE relabelled, for the same reason. Renaming
    the legend's Text artists works until something downstream rebuilds the
    legend from `get_legend_handles_labels()`, which reads each handle's own
    `_label` and so restores the original string -- and this module's own
    legend-fitting pass does exactly that, a few lines after this function
    runs. The effect was that every rename reached titles, axis labels,
    annotations and ticks but silently missed the keys, which is where the
    group names mostly appear.
    """
    from . import config as _cfg

    def _rename(old):
        new = _cfg.display_text(old)
        return new if new != old else None

    # Tick formatters first. set_text on a tick's Text artist is regenerated
    # from the formatter on the next draw, so it only stuck here because a
    # later pass in savefig_dual happened to rewrite the labels again. Doing
    # it at source means the function keeps its promise on its own, including
    # for the callers that use it outside savefig_dual.
    # Which formatter `set_xticklabels` installs depends on the matplotlib
    # version -- FixedFormatter on older ones, a FuncFormatter closing over
    # the list on 3.10 -- so both are handled, and a numeric axis (Scalar,
    # Log, ...) is left alone.
    #
    # The opt-out has to be honoured here too. `_keep_text` is set on the Text
    # artists that `set_yticklabels` returns -- four figures abbreviate their
    # row labels on purpose and flag them -- so rewriting the formatter under
    # them would expand exactly the abbreviations the flag exists to protect.
    # If any tick on an axis is flagged, that axis keeps its formatter.
    import matplotlib.ticker as _mticker
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            try:
                if any(getattr(t, "_keep_text", False)
                       for t in axis.get_ticklabels(which="both")):
                    continue
            except Exception:
                pass
            fmt = axis.get_major_formatter()
            if isinstance(fmt, _mticker.FixedFormatter):
                seq = list(getattr(fmt, "seq", []) or [])
                new_seq = [_cfg.display_text(s) if isinstance(s, str) else s
                           for s in seq]
                if new_seq != seq:
                    axis.set_major_formatter(_mticker.FixedFormatter(new_seq))
            elif isinstance(fmt, _mticker.FuncFormatter):
                _inner = fmt.func
                if getattr(_inner, "_display_wrapped", False):
                    continue                     # already wrapped; don't nest
                def _wrapped(x, pos, _f=_inner):
                    return _cfg.display_text(_f(x, pos))
                _wrapped._display_wrapped = True
                # Keep the original so freeze_display_ticks can put it back on
                # an axis it cannot pin. The wrapper is a local function and
                # makes the whole figure unpicklable, which silently disables
                # the figure cache.
                _wrapped._display_inner = _inner
                axis.set_major_formatter(_mticker.FuncFormatter(_wrapped))

    # Handles next, so a later legend rebuild carries the display name.
    for ax in fig.get_axes():
        try:
            handles, _labels = ax.get_legend_handles_labels()
        except Exception:
            continue
        for h in handles:
            if getattr(h, "_keep_text", False):
                continue
            lab = h.get_label()
            # A label starting with "_" is matplotlib's opt-out from the
            # legend; leave it alone rather than risk changing that meaning.
            if not isinstance(lab, str) or lab.startswith("_"):
                continue
            new = _rename(lab)
            if new is not None:
                h.set_label(new)

    for artist in _collect_text_artists(fig):
        # A string that is deliberately abbreviated opts out: this pass exists
        # to stop internal keys reaching a reader, and "H-G" set on purpose,
        # with the caption defining it, is not that. Without the opt-out it
        # expands the abbreviation straight back to the long form the figure
        # had no room for.
        if getattr(artist, "_keep_text", False):
            continue
        new = _rename(artist.get_text())
        if new is not None:
            artist.set_text(new)
    return fig


def _report_overflow(fig, name: str, bbox_inches) -> float:
    """Warn when text outside the axes will grow the saved canvas past the page.

    Returns the overflow in inches (0.0 when the figure fits). Only meaningful
    when saving with bbox_inches="tight", which is what grows the canvas.
    """
    if bbox_inches != "tight":
        return 0.0
    try:
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
    except Exception:
        return 0.0
    fig_w = float(fig.get_size_inches()[0])
    over = float(bb.width) - fig_w
    if over > 0.25:
        n_text = sum(1 for a in _collect_text_artists(fig) if a.get_text().strip())
        print(f"    [overflow] {name}: text outside the axes widens the save to "
              f"{bb.width:.1f} in from {fig_w:.1f} in, so everything prints at "
              f"1/{bb.width / PAGE_WIDTH_IN:.1f} size. {n_text} text artists -- "
              f"cut labels, do not widen the figure.")
    return max(over, 0.0)


# ── Save ───────────────────────────────────────────────────────────────────

# ── Collision mapping ──────────────────────────────────────────────────────
# Overlap is the failure that matters and the one that is cheapest to detect:
# every text artist knows its own rendered box. Measuring at draw time, before
# saving, means a figure can report exactly which strings hit each other rather
# than leaving it to be spotted by eye afterwards.

OVERLAP_MIN_FRAC: float = 0.12   # of the smaller box, before it counts


def _offscreen_tick_labels(fig) -> set:
    """Tick labels matplotlib keeps but never draws.

    A tick whose location falls outside its axis limits keeps a label artist
    with get_visible() True; it is simply not rendered. Counting those as text
    produced collisions against labels that are not on the page -- a '1000'
    that no reader ever sees, reported against a real one -- and a font sweep
    that chased them settled on a smaller size than the figure needed.
    """
    dead = set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = (lim[0], lim[1]) if lim[0] <= lim[1] else (lim[1], lim[0])
            try:
                ticks = axis.get_major_ticks()
                locs = axis.get_majorticklocs()
            except Exception:
                continue
            for tick, loc in zip(ticks, locs):
                if lo - 1e-9 <= loc <= hi + 1e-9:
                    continue
                for lab in (getattr(tick, "label1", None),
                            getattr(tick, "label2", None)):
                    if lab is not None:
                        dead.add(id(lab))
    return dead


def _text_boxes(fig, renderer):
    """(artist, bbox, text) for every visible, non-empty text artist."""
    out = []
    dead = _offscreen_tick_labels(fig)
    for art in _collect_text_artists(fig):
        if id(art) in dead:
            continue
        try:
            if not art.get_visible() or not art.get_text().strip():
                continue
            bb = art.get_window_extent(renderer=renderer)
            if bb.width <= 0 or bb.height <= 0:
                continue
            out.append((art, bb, art.get_text()))
        except Exception:
            continue
    return out


def find_collisions(fig, min_frac: float = OVERLAP_MIN_FRAC):
    """Pairs of text artists whose rendered boxes overlap.

    Returns [(text_a, text_b, overlap_fraction), ...], worst first. The fraction
    is of the smaller box, so a short label swallowed by a long one scores 1.0.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = _text_boxes(fig, r)
    hits = []
    for i in range(len(boxes)):
        _, bi, ti = boxes[i]
        ai = max(bi.width * bi.height, 1e-9)
        for j in range(i + 1, len(boxes)):
            _, bj, tj = boxes[j]
            ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
            oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
            if ox <= 0 or oy <= 0:
                continue
            aj = max(bj.width * bj.height, 1e-9)
            frac = (ox * oy) / min(ai, aj)
            if frac >= min_frac:
                hits.append((ti, tj, round(float(frac), 3)))
    hits.sort(key=lambda h: -h[2])
    return hits


def report_collisions(fig, name: str = "", limit: int = 6) -> int:
    """Print the colliding text in a figure. Returns how many pairs collide."""
    hits = find_collisions(fig)
    if not hits:
        return 0
    one = lambda t: " ".join(t.split())[:38]
    print(f"    [collide] {name or 'figure'}: {len(hits)} overlapping text pairs")
    for a, b, f in hits[:limit]:
        print(f"                {f:>5.0%}  {one(a)!r} x {one(b)!r}")
    if len(hits) > limit:
        print(f"                ... and {len(hits) - limit} more")
    return len(hits)


# ── Fitting text to the space it actually has ──────────────────────────────
# Panel count is a proxy, and a poor one. What decides whether type is too big
# is whether a given string fits the width it is given: a two-inch panel with
# six tick labels allows each label a third of an inch, whatever the grid shape.
# Measure that directly and shrink only what does not fit.
#
# This only ever reduces. An earlier version searched upward for the largest
# size that stayed clean, which made every figure a negotiation and never
# settled; shrinking to a measured constraint converges in one pass.

FIT_MIN_FS: float = 5.5          # below this a label is not worth keeping
FIT_PAD: float = 1.04            # a little air between neighbours


def _fits_scale(texts, avail_px, renderer):
    """Largest scale at which the widest of `texts` fits `avail_px`."""
    need = 0.0
    for t in texts:
        try:
            if not t.get_visible() or not t.get_text().strip():
                continue
            need = max(need, t.get_window_extent(renderer=renderer).width)
        except Exception:
            continue
    if need <= 0 or avail_px <= 0:
        return 1.0
    return min(1.0, avail_px / (need * FIT_PAD))


def fit_tick_labels(fig, min_fs: float = FIT_MIN_FS) -> dict:
    """Shrink tick labels that do not fit the space between their ticks.

    Returns {axes_index: new_size} for whatever was changed. Each axis is
    treated separately: a crowded x-axis does not shrink the y-axis with it.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    changed = {}
    for ai, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        try:
            bb = ax.get_window_extent(renderer=r)
        except Exception:
            continue
        for axis, span in (("x", bb.width), ("y", bb.height)):
            labs = (ax.get_xticklabels() if axis == "x"
                    else ax.get_yticklabels())
            labs = [t for t in labs if t.get_visible() and t.get_text().strip()]
            if len(labs) < 2:
                continue
            # x labels sit side by side, so each gets its share of the width;
            # y labels stack, so width is not the binding constraint -- their
            # height is, and it is rarely close.
            if axis == "x":
                avail = span / len(labs)
                sc = _fits_scale(labs, avail, r)
            else:
                avail = span / len(labs)
                heights = [t.get_window_extent(renderer=r).height for t in labs]
                need = max(heights) if heights else 0
                sc = 1.0 if need <= 0 else min(1.0, avail / (need * FIT_PAD))
            if os.environ.get("FIT_DEBUG"):
                print(f"    [fitdbg] ax{ai}.{axis} n={len(labs)} span={span:.0f} "
                      f"avail={avail:.1f} need={(need if axis=='y' else 0):.1f} "
                      f"sc={sc:.3f} cur={labs[0].get_fontsize():.1f}")
            if sc >= 0.995:
                continue
            cur = labs[0].get_fontsize()
            new = max(min_fs, cur * sc)
            if new < cur - 0.05:
                for t in labs:
                    t.set_fontsize(new)
                changed[f"ax{ai}.{axis}"] = round(new, 1)
    return changed


# Order matters: the first placement that covers no data wins, so this is a
# preference list, not just a search space. Lower corners first -- a legend at
# the foot of a panel reads as a key, one pinned top-right reads as a label on
# the plot and competes with the title.
# Corners only. A legend that lands mid-panel scores well on "covers no data"
# -- it slips into a gap between markers -- and reads as a box dropped on the
# figure. A reader looks for a key in a corner, so those are the candidates.
LEGEND_LOCS = ("lower right", "lower left", "upper left", "upper right")
LEGEND_MIN_FS: float = 5.0
LEGEND_SCALES = (1.0, 0.88, 0.78, 0.68, 0.60, 0.52)
LEGEND_WRAP: int = 18          # characters before a label gets a second line


def _wrap_labels(labels, width: int = LEGEND_WRAP):
    """Break long legend entries over two lines.

    A tall narrow legend fits corners that a wide flat one cannot, so wrapping
    buys placement room without shrinking the type.
    """
    import textwrap
    out = []
    for lab in labels:
        t = str(lab)
        # Never wrap mathtext. textwrap counts characters, not tokens, so it
        # happily breaks \hat{f}_{Ind} into \hat{f}_{In\nd} and the label stops
        # rendering as an equation.
        if "$" in t or len(t) <= width:
            out.append(t)
            continue
        out.append("\n".join(textwrap.wrap(t, width)))
    return out


def _legend_occlusion(ax, renderer):
    leg = ax.get_legend()
    if leg is None:
        return 1.0
    try:
        bb = leg.get_window_extent(renderer=renderer)
    except Exception:
        return 1.0
    # A legend that reaches the panel title is as wrong as one on the data, and
    # the title is not data so nothing else here would notice it.
    try:
        t = ax.title
        if t.get_text().strip():
            tb = t.get_window_extent(renderer=renderer)
            if not (bb.x1 <= tb.x0 or bb.x0 >= tb.x1 or
                    bb.y1 <= tb.y0 or bb.y0 >= tb.y1):
                return 1.0
    except Exception:
        pass
    n_in, n_tot = _data_points_in(ax, bb, renderer)
    frac = (n_in / n_tot) if n_tot else 0.0
    # A curve reaching into the box, or running under it, is occlusion however
    # few vertices it spends there. Without this the corner holding the tallest
    # peak scores as the emptiest, which is how the density legend kept landing
    # on the technical-replicate peak.
    if _curve_intrudes(ax, bb):
        frac = max(frac, 1.0)
    return frac


def place_legends_clear(fig, max_frac: float = 0.01,
                        min_fs: float = LEGEND_MIN_FS) -> list:
    """Place each legend inside its axes and clear of the data.

    Panels showing the same series should carry their key in the same corner --
    one bottom left and its neighbour top right reads as carelessness. Panels
    showing different things have no such tie and are free to use whatever
    corner suits their own data.

    Axes are grouped by their legend labels, which is the available signal for
    "these two plot the same curves": same key, same placement; different key,
    independent choice.

    Returns [(axes_index, loc, fontsize, covered_fraction), ...].
    """
    axes_with = []
    for ai, ax in enumerate(fig.axes):
        leg = ax.get_legend()
        # A key deliberately placed outside its axes opts out: this pass finds
        # the clearest corner *inside* the panel, which is the wrong answer
        # when the figure has already decided there is no clear corner.
        if leg is not None and getattr(leg, "_fixed_loc", False):
            continue
        if leg is None or not leg.get_visible():
            continue
        handles = list(getattr(leg, "legend_handles", None)
                       or getattr(leg, "legendHandles", None) or [])
        labels = [t.get_text() for t in leg.get_texts()]
        if not handles or len(handles) != len(labels):
            handles, labels = ax.get_legend_handles_labels()
        if not handles:
            continue
        base_fs = (leg.get_texts()[0].get_fontsize()
                   if leg.get_texts() else _mpl.rcParams["legend.fontsize"])
        axes_with.append((ai, ax, handles, _wrap_labels(labels), base_fs))
    if not axes_with:
        return []

    # Group by legend content: identical keys mean identical series.
    groups = {}
    for entry in axes_with:
        groups.setdefault(tuple(entry[3]), []).append(entry)

    def _apply(group, loc, scale):
        """Draw this group's legends at one corner; return the worst coverage."""
        for _, ax, h, l, bfs in group:
            try:
                ax.legend(h, l, loc=loc, fontsize=max(min_fs, bfs * scale),
                          framealpha=0.9, borderpad=0.3, labelspacing=0.3,
                          handlelength=1.4, handletextpad=0.5)
            except Exception:
                return 1.0
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        return max(_legend_occlusion(ax, r) for _, ax, *_ in group)

    out = []
    chosen = []                       # (group, loc, scale) decided per group
    for group in groups.values():
        best = None                   # (worst_frac, loc, scale)
        for scale in LEGEND_SCALES:
            for loc in LEGEND_LOCS:
                worst = _apply(group, loc, scale)
                if best is None or worst < best[0]:
                    best = (worst, loc, scale)
                if worst <= max_frac:
                    break
            if best and best[0] <= max_frac:
                break
        worst, loc, scale = best
        chosen.append((group, loc, scale))

    # One size for the whole figure. Groups may sit in different corners -- they
    # show different things -- but a 9 pt key beside a 7 pt key reads as an
    # accident, so every legend takes the smallest size any of them needed.
    if chosen:
        uniform = min(s_ for _, _, s_ in chosen)
        for group, loc, _ in chosen:
            _apply(group, loc, uniform)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for group, loc, _ in chosen:
            for ai, ax, _, _, bfs in group:
                out.append((ai, loc, round(max(min_fs, bfs * uniform), 1),
                            round(_legend_occlusion(ax, r), 3)))
    return sorted(out)

def _data_points_in(ax, bbox, renderer):
    """How much of an axes' drawn data falls inside a display-space box.

    Returns (n_inside, n_total) over the vertices of every line and the offsets
    of every scatter/collection in the axes. Sampling vertices rather than
    rasterising keeps this cheap enough to run on every figure the pipeline
    draws.
    """
    import numpy as _np
    inside = total = 0
    for ln in ax.get_lines():
        try:
            if not ln.get_visible():
                continue
            xy = ln.get_xydata()
            if xy is None or len(xy) == 0:
                continue
            disp = ax.transData.transform(xy)
            disp = disp[_np.isfinite(disp).all(axis=1)]
            if len(disp) == 0:
                continue
            total += len(disp)
            inside += int(((disp[:, 0] >= bbox.x0) & (disp[:, 0] <= bbox.x1) &
                           (disp[:, 1] >= bbox.y0) & (disp[:, 1] <= bbox.y1)).sum())
        except Exception:
            continue
    for col in ax.collections:
        try:
            if not col.get_visible():
                continue
            off = col.get_offsets()
            if off is None or len(off) == 0:
                continue
            disp = col.get_offset_transform().transform(off)
            disp = disp[_np.isfinite(disp).all(axis=1)]
            if len(disp) == 0:
                continue
            total += len(disp)
            inside += int(((disp[:, 0] >= bbox.x0) & (disp[:, 0] <= bbox.x1) &
                           (disp[:, 1] >= bbox.y0) & (disp[:, 1] <= bbox.y1)).sum())
        except Exception:
            continue
    return inside, total


OCCLUSION_MIN_FRAC: float = 0.02   # of an axes' data hidden, before it counts


def _curve_intrudes(ax, bbox) -> bool:
    """Does a plotted curve pass through this box, or run beneath it?

    Counting vertices is the wrong measure for a line. A kernel density curve
    carries five hundred vertices, almost all of them strung along a flat tail
    near the axis, and only a couple of dozen at its peak -- so by vertex count
    the corner containing the tallest peak in the figure looks like the emptiest
    place on the panel. That is exactly where the legend kept being sent.

    Ask the question the reader would instead: within the horizontal span of
    the box, does the curve reach the box, or is the box sitting over the area
    under the curve? Either way it is on the data.
    """
    import numpy as _np
    for ln in ax.get_lines():
        try:
            if not ln.get_visible():
                continue
            xy = ln.get_xydata()
            if xy is None or len(xy) < 2:
                continue
            disp = ax.transData.transform(xy)
            disp = disp[_np.isfinite(disp).all(axis=1)]
            span = disp[(disp[:, 0] >= bbox.x0) & (disp[:, 0] <= bbox.x1)]
            if len(span) == 0:
                continue
            # The curve enters the box, or passes above its lower edge (so the
            # box covers the region between curve and axis).
            if float(span[:, 1].max()) >= bbox.y0:
                return True
        except Exception:
            continue
    return False


def find_occlusions(fig, min_frac: float = OCCLUSION_MIN_FRAC):
    """Legends and in-axes text boxes sitting on top of plotted data.

    Text-against-text is only half the problem: a legend parked on the curve it
    describes collides with nothing and still hides the result. Returns
    [(what, axes_index, fraction_of_data_covered), ...], worst first.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hits = []
    for ai, ax in enumerate(fig.axes):
        boxes = []
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            try:
                boxes.append(("legend", leg.get_window_extent(renderer=r)))
            except Exception:
                pass
        for t in ax.texts:
            try:
                if not t.get_visible() or not t.get_text().strip():
                    continue
                # Only boxed notes occlude; a bare number sits between points.
                if t.get_bbox_patch() is None:
                    continue
                boxes.append((f"note {t.get_text()[:24]!r}",
                              t.get_window_extent(renderer=r)))
            except Exception:
                continue
        if not boxes:
            continue
        for what, bb in boxes:
            n_in, n_tot = _data_points_in(ax, bb, r)
            frac = (n_in / n_tot) if n_tot else 0.0
            if _curve_intrudes(ax, bb):
                frac = max(frac, 1.0)
            if frac >= min_frac:
                hits.append((what, ai, round(float(frac), 3)))
    hits.sort(key=lambda h: -h[2])
    return hits


def report_layout(fig, name: str = "", limit: int = 6) -> dict:
    """Collisions and occlusions in one report. Returns counts."""
    coll = find_collisions(fig)
    occ = find_occlusions(fig)
    # Always emit the line, including when it is clean. A silent pass is
    # indistinguishable from a run that never drew the figure, and "no problems
    # found" has to mean something was actually looked at.
    print(f"    [layout] {name or 'figure'}: "
          f"{len(coll)} text overlaps, {len(occ)} data occlusions")
    if coll or occ:
        one = lambda t: " ".join(t.split())[:34]
        for a, b, f in coll[:limit]:
            print(f"               overlap {f:>5.0%}  {one(a)!r} x {one(b)!r}")
        for what, ai, f in occ[:limit]:
            print(f"               occlude {f:>5.0%}  ax{ai} {what}")
    return {"collisions": len(coll), "occlusions": len(occ)}


def check_style_coherence(fig, name: str = "") -> bool:
    """Do the figure's tick labels agree with the size the code thinks it set?

    Explicit sizes in figure code are written as `current_fs() + offset`, while
    ticks and axis labels come from the rcParams. Those two can drift apart
    without any error: anything that resets the style between the size being
    chosen and the axes being created leaves a figure whose annotations are one
    scale and whose ticks are another. That is invisible in review and survives
    a font sweep, because the usual cause is a lazy import that fires only on
    the first figure in a process.

    One comparison catches it. Returns True when the figure is coherent.
    """
    want = _mpl.rcParams.get("xtick.labelsize")
    if not isinstance(want, (int, float)):
        return True
    sizes = set()
    for ax in fig.axes:
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if t.get_visible() and t.get_text().strip():
                sizes.add(round(float(t.get_fontsize()), 1))
    if not sizes:
        return True
    expect = round(float(want), 1)
    # Deliberate per-axis sizes are fine; a figure drawn under a different scale
    # entirely is not, so compare against the size the scale would give.
    stray = {v for v in sizes if v < expect * 0.6}
    if stray and expect > MIN_FS + 0.5:
        print(f"    [style] {name or 'figure'}: tick labels at "
              f"{sorted(stray)} pt but the type scale is {current_fs():g} pt "
              f"(expects {expect:g}). Something reset the style after the size "
              f"was chosen -- most likely an import inside the builder.")
        return False
    return True


FIGCACHE_DIRNAME = ".figcache"


def _cache_figure(fig, stem: str) -> bool:
    """Store the drawn figure so its layout can be changed without recomputing.

    Caching the *data* a figure plots needs one bespoke entry per figure, and
    in several sections the maths and the drawing are the same function, so
    there is nothing to hand over. Caching the figure object needs none of
    that: everything the picture is made of is already in it, and restyling --
    type size, canvas, margins, ticks, grids, legend placement -- operates on
    exactly those objects.

    Costs about 0.2 MB and 0.03 s per figure, so it runs unconditionally.
    Closures do not pickle, and this module attaches post-layout hooks that are
    closures, so they come off first; they have already run by this point.
    """
    import pickle as _pk
    try:
        d = os.path.join(os.path.dirname(stem) or ".", FIGCACHE_DIRNAME)
        os.makedirs(d, exist_ok=True)
        # The annotation layer is described, not stored. Baking legends,
        # titles, notes and arrows into the cache would preserve exactly the
        # things that keep needing to move -- a cached figure would carry its
        # own layout problems forward and restyling could only nudge them.
        # Strip them to a spec and rebuild from it, so every replot lays the
        # annotations out afresh at whatever size the new style asks for.
        spec = capture_annotations(fig)
        strip_annotations(fig)
        hooks = getattr(fig, "_post_layout_hooks", None)
        fig._post_layout_hooks = []
        try:
            blob = _pk.dumps({"figure": fig, "annotations": spec,
                              "base_fs": current_fs()},
                             protocol=_pk.HIGHEST_PROTOCOL)
        finally:
            if hooks is not None:
                fig._post_layout_hooks = hooks
        with open(os.path.join(d, os.path.basename(stem) + ".pkl"), "wb") as f:
            f.write(blob)
        # Put the live figure back the way it was found. Caching is a read, as
        # far as the caller is concerned, and a caller that draws again or
        # inspects the figure afterwards should not find it stripped bare.
        apply_annotations(fig, spec)
        return True
    except Exception as exc:
        print(f"    [figcache] {os.path.basename(stem)}: not cacheable ({exc})")
        return False


def capture_annotations(fig) -> dict:
    """Everything that is written on a figure rather than plotted in it.

    Legend *handles* are deliberately not captured: the data artists keep their
    own labels, so a legend can be rebuilt from the axes itself. What has to be
    remembered is that there was one, and how it was asked to sit.
    """
    out = {"suptitle": None, "supxlabel": None, "supylabel": None,
           "figtexts": [], "axes": []}
    sup = getattr(fig, "_suptitle", None)
    if sup is not None and sup.get_text().strip():
        out["suptitle"] = dict(text=sup.get_text(),
                               fontsize=sup.get_fontsize(),
                               weight=sup.get_fontweight())
    for name in ("supxlabel", "supylabel"):
        t = getattr(fig, "_" + name, None)
        if t is not None and t.get_text().strip():
            out[name] = dict(text=t.get_text(), fontsize=t.get_fontsize())
    known = {id(x) for x in (sup, getattr(fig, "_supxlabel", None),
                             getattr(fig, "_supylabel", None)) if x is not None}
    for t in fig.texts:
        if id(t) in known or not t.get_text().strip():
            continue
        out["figtexts"].append(dict(
            text=t.get_text(), xy=t.get_position(), fontsize=t.get_fontsize(),
            ha=t.get_ha(), va=t.get_va(), color=t.get_color()))
    for ax in fig.axes:
        leg = ax.get_legend()
        rec = dict(
            title=ax.get_title(), title_fs=ax.title.get_fontsize(),
            xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(),
            label_fs=ax.xaxis.label.get_fontsize(),
            legend=None, texts=[], annotations=[])
        if leg is not None and leg.get_visible():
            rec["legend"] = dict(
                loc=getattr(leg, "_loc", 0),
                ncol=getattr(leg, "_ncols", getattr(leg, "_ncol", 1)),
                fontsize=(leg.get_texts()[0].get_fontsize()
                          if leg.get_texts() else None),
                labels=[t.get_text() for t in leg.get_texts()])
        for t in ax.texts:
            if not t.get_text().strip():
                continue
            item = dict(text=t.get_text(), xy=t.get_position(),
                        fontsize=t.get_fontsize(), ha=t.get_ha(),
                        va=t.get_va(), color=t.get_color(),
                        axes_coords=(t.get_transform() is ax.transAxes),
                        weight=t.get_fontweight())
            arrow = getattr(t, "arrow_patch", None)
            if arrow is not None or getattr(t, "xyann", None) is not None:
                item["xyann"] = getattr(t, "xyann", None)
                item["anchor"] = getattr(t, "xy", None)
                item["has_arrow"] = arrow is not None
            (rec["annotations"] if item.get("has_arrow") else rec["texts"]).append(item)
        out["axes"].append(rec)
    return out


def strip_annotations(fig) -> None:
    """Remove the annotation layer, leaving the panels and the data."""
    sup = getattr(fig, "_suptitle", None)
    if sup is not None:
        sup.set_text("")
    for name in ("_supxlabel", "_supylabel"):
        t = getattr(fig, name, None)
        if t is not None:
            t.set_text("")
    for t in list(fig.texts):
        try:
            t.remove()
        except Exception:
            pass
    for ax in fig.axes:
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        for t in list(ax.texts):
            try:
                t.remove()
            except Exception:
                pass


def apply_annotations(fig, spec: dict, scale: float = 1.0) -> None:
    """Put the annotation layer back, sized for the style now in force."""
    def _fs(v, default):
        return max(MIN_FS, (v if v else default) * scale)
    if spec.get("suptitle"):
        d = spec["suptitle"]
        fig.suptitle(d["text"], fontsize=_fs(d["fontsize"], current_fs() + 2),
                     fontweight=d.get("weight") or "normal")
    if spec.get("supxlabel"):
        fig.supxlabel(spec["supxlabel"]["text"],
                      fontsize=_fs(spec["supxlabel"]["fontsize"], current_fs()))
    if spec.get("supylabel"):
        fig.supylabel(spec["supylabel"]["text"],
                      fontsize=_fs(spec["supylabel"]["fontsize"], current_fs()))
    for d in spec.get("figtexts", []):
        fig.text(d["xy"][0], d["xy"][1], d["text"],
                 fontsize=_fs(d["fontsize"], current_fs() - 2),
                 ha=d["ha"], va=d["va"], color=d["color"])
    for ax, rec in zip(fig.axes, spec.get("axes", [])):
        if rec["title"]:
            ax.set_title(rec["title"], fontsize=_fs(rec["title_fs"], current_fs()))
        if rec["xlabel"]:
            ax.set_xlabel(rec["xlabel"], fontsize=_fs(rec["label_fs"], current_fs()),
                          labelpad=label_pad())
        if rec["ylabel"]:
            ax.set_ylabel(rec["ylabel"], fontsize=_fs(rec["label_fs"], current_fs()),
                          labelpad=label_pad())
        for d in rec.get("texts", []):
            ax.text(d["xy"][0], d["xy"][1], d["text"],
                    transform=ax.transAxes if d.get("axes_coords") else ax.transData,
                    fontsize=_fs(d["fontsize"], current_fs() - 2),
                    ha=d["ha"], va=d["va"], color=d["color"],
                    fontweight=d.get("weight") or "normal")
        for d in rec.get("annotations", []):
            try:
                ax.annotate(d["text"], xy=d.get("anchor"),
                            xytext=d.get("xyann"), textcoords="data",
                            fontsize=_fs(d["fontsize"], current_fs() - 2),
                            ha=d["ha"], va=d["va"], color=d["color"],
                            arrowprops=dict(arrowstyle="-|>", color="gray", lw=0.8))
            except Exception:
                pass
        if rec.get("legend"):
            h, l = ax.get_legend_handles_labels()
            if h:
                ax.legend(h, l, ncol=rec["legend"].get("ncol") or 1,
                          fontsize=_fs(rec["legend"].get("fontsize"),
                                       current_fs() - 2),
                          framealpha=0.9)


def freeze_display_ticks(fig) -> int:
    """Rewrite tick labels to their published form and pin them there.

    Returns how many labels were changed.
    """
    from . import config as _cfg
    from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter
    changed = 0
    try:
        fig.canvas.draw()
    except Exception:
        return 0
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            try:
                # An axis apply_display_text wrapped but that this pass cannot
                # pin -- no materialised labels, or an opt-out below -- gets the
                # original formatter back. Leaving the wrapper on costs the
                # figure cache: it is a local function, so the figure stops
                # pickling and _cache_figure drops it.
                _f = getattr(axis.get_major_formatter(), "func", None)
                _inner = getattr(_f, "_display_inner", None)

                ticklabels = list(axis.get_ticklabels())
                labs = [t.get_text() for t in ticklabels]
                if not labs or not any(l.strip() for l in labs):
                    if _inner is not None:
                        axis.set_major_formatter(FuncFormatter(_inner))
                    continue
                # An axis whose labels were deliberately abbreviated opts out:
                # this pass exists to publish the display form, and re-running
                # it over a chosen short form expands it back to the long one.
                if any(getattr(t, "_keep_text", False) for t in ticklabels):
                    continue
                new = [_cfg.display_text(l) for l in labs]
                # An axis apply_display_text wrapped is pinned even when the
                # rename is a no-op. The wrapper is a local function, so the
                # figure stops pickling and the figure cache drops it -- the
                # last run reported "not cacheable" for 126 figures for exactly
                # this reason. Pinning the drawn strings here replaces the
                # wrapper with a plain list, which both renders identically and
                # pickles.
                if new == labs and not getattr(
                        getattr(axis.get_major_formatter(), "func", None),
                        "_display_wrapped", False):
                    continue
                locs = list(axis.get_majorticklocs())
                if len(locs) != len(new):
                    continue
                axis.set_major_locator(FixedLocator(locs))
                axis.set_major_formatter(FixedFormatter(new))
                changed += sum(1 for a, b in zip(labs, new) if a != b)
            except Exception:
                continue
    return changed


# One type scale for every figure, in printed points. Science asks for about
# 7 pt and warns against wide variation within a figure; the greater risk here
# was variation *between* figures, where a reader re-calibrates on every page,
# and inconsistent legends *within* one figure, which reads as carelessness.
# Roles, not call sites, decide size.
# Type sizes are relative to the panel, not to the page. A fixed printed size
# reads as too small on a figure of three wide panels and too large on one of
# fourteen narrow ones, because what the eye compares type against is the panel
# beside it. These are multiples of the median printed panel width in inches,
# then clamped into the Science band.
# One base size per figure, set by the panel it sits beside, then the roles as
# ratios of that base. Clamping each role separately instead flattened the
# hierarchy on any figure with large panels -- title, label and tick all hit
# the ceiling and came out the same size.
TYPE_BASE_PER_IN = 4.0          # points of body type per printed inch of panel
TYPE_BASE_MIN, TYPE_BASE_MAX = 5.2, 8.9
TYPE_RATIO = {
    "axes_title": 1.15,
    "label":      1.00,     # axis labels, and shared figure-level axis labels
    "tick":       0.86,
    "legend":     0.86,     # a key is reference, not reading matter
    "annotation": 0.98,
}
TYPE_MIN, TYPE_MAX = 5.2, 9.5   # Science: about 7 pt, never below 5

# One family, resolved the same way everywhere. Several modules asked for
# Helvetica first and this one for Arial; on macOS those resolve to different
# files and the figures came out visibly different from each other. Arial
# leads because the macOS Helvetica .ttc has no usable bold face, which is how
# bold panel letters silently stopped being bold.
FONT_STACK = ["Arial", "Helvetica", "DejaVu Sans"]


#: Every rcParam measured in points, grouped by what floor applies. A figure
#: composed at one width and printed at another has to scale all of these, not
#: just the font sizes: tick marks, spines and pads are in points too, and
#: leaving them behind makes the ticks tower over the panels they belong to.
_RC_FONT_PT = (
    "font.size", "axes.labelsize", "axes.titlesize", "figure.titlesize",
    "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
    "legend.title_fontsize",
)
_RC_WIDTH_PT = (
    "axes.linewidth", "lines.linewidth", "lines.markeredgewidth",
    "grid.linewidth", "patch.linewidth", "hatch.linewidth",
    "xtick.major.width", "xtick.minor.width",
    "ytick.major.width", "ytick.minor.width",
    "boxplot.boxprops.linewidth", "boxplot.capprops.linewidth",
    "boxplot.whiskerprops.linewidth", "boxplot.medianprops.linewidth",
    "boxplot.meanprops.linewidth", "boxplot.flierprops.linewidth",
    "boxplot.flierprops.markeredgewidth",
)
_RC_LENGTH_PT = (
    "xtick.major.size", "xtick.minor.size",
    "ytick.major.size", "ytick.minor.size",
    "xtick.major.pad", "xtick.minor.pad",
    "ytick.major.pad", "ytick.minor.pad",
    "axes.labelpad", "axes.titlepad",
    "lines.markersize", "boxplot.flierprops.markersize",
    "boxplot.meanprops.markersize",
)


def _png_width_in(path, dpi):
    """Width of a written PNG, in inches. None if it cannot be read."""
    try:
        from PIL import Image as _I
        with _I.open(path) as im:
            return im.size[0] / float(dpi)
    except Exception:
        return None


def shrink_to_page_width(fig, target_in=None, tries=4, floor=0.70):
    """Bring a figure's saved width down to the printed width.

    `bbox_inches="tight"` sizes the saved file to the content, and artists
    that sit outside the axes -- a key below the panels, a rotated row label,
    a long tick label -- push it past the column the figure is placed in.
    Anything wider is reduced on the page, which takes the rules and symbols
    with it.

    Shrinks the canvas and re-measures, because type keeps its point size when
    the canvas changes: the bounding box does not shrink proportionally, so
    one pass undershoots. Stops at `floor` of the original width rather than
    chasing a figure whose overflow is mostly text.

    Returns (before_in, after_in).
    """
    target_in = PAGE_WIDTH_IN if target_in is None else target_in
    try:
        r = fig.canvas.get_renderer()
        before = float(fig.get_tightbbox(r).width)
    except Exception:
        return (None, None)
    w0 = float(fig.get_size_inches()[0])
    cur = before
    for _ in range(tries):
        if cur <= target_in + 0.02:
            break
        size = fig.get_size_inches()
        k = max(target_in / cur, floor * w0 / float(size[0]))
        if k >= 0.999:
            break
        fig.set_size_inches(size[0] * k, size[1] * k, forward=True)
        try:
            fig.canvas.draw()
            cur = float(fig.get_tightbbox(fig.canvas.get_renderer()).width)
        except Exception:
            break
    return (before, cur)


def scale_point_rcparams(k, font_floor=None, width_floor=0.5):
    """Multiply every point-valued rcParam by `k`.

    For a figure composed at one width and printed at another. Fonts are held
    at the Science 5 pt floor and rules at 0.5 pt; lengths and pads scale
    freely, since a short tick is not a legibility problem.

    Returns {param: (before, after)} for the ones that moved, so a caller can
    say what it did rather than changing the whole style silently.
    """
    font_floor = MIN_FS if font_floor is None else font_floor
    moved = {}
    for keys, floor in ((_RC_FONT_PT, font_floor),
                        (_RC_WIDTH_PT, width_floor),
                        (_RC_LENGTH_PT, None)):
        for key in keys:
            try:
                before = _mpl.rcParams[key]
            except KeyError:
                continue
            if not isinstance(before, (int, float)) or isinstance(before, bool):
                continue
            after = before * k
            if floor is not None:
                after = max(after, floor)
            if abs(after - before) > 1e-9:
                _mpl.rcParams[key] = after
                moved[key] = (before, after)
    return moved


def okabe_ito_sequential(name="oi_seq"):
    """A sequential colormap built from the Okabe-Ito palette.

    Light yellow through sky to blue: monotonic in lightness, so it survives
    greyscale and every common colour deficiency, and it keeps the figure in
    the same palette as the rest of the set rather than importing a second
    colour language for one panel.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from . import config as _cfg
    stops = ["#FFFFFF", _cfg.OI["yellow"], _cfg.OI["sky"], _cfg.OI["blue"]]
    return LinearSegmentedColormap.from_list(name, stops)


def okabe_ito_diverging(name="oi_div"):
    """Orange-white-blue, the two Okabe-Ito hues that read apart for everyone."""
    from matplotlib.colors import LinearSegmentedColormap
    from . import config as _cfg
    return LinearSegmentedColormap.from_list(
        name, [_cfg.OI["orange"], "#FFFFFF", _cfg.OI["blue"]])


def okabe_ito_sequential_r(name="oi_seq_r"):
    return okabe_ito_sequential(name).reversed()


PANEL_LETTER_PT = 10.0   # Science: part labels are 10 pt bold, as printed


def add_panel_letters(fig, axes=None, start=0, offset=(-0.02, 1.02)):
    """Label the panels A, B, C ... in the upper left of each.

    Science asks for capital part labels at 10 pt bold in the final figure,
    placed upper-left. The size is in printed points, so it is converted
    against the saved width the same way the rest of the type scale is; a
    fixed 10 here would print at whatever the reduction happened to make it.

    Marked _fixed_pt so the shared type scale leaves them alone: they are
    deliberately larger than the body text, which is the one place a figure
    should vary.
    """
    axes = list(axes) if axes is not None else [a for a in fig.axes
                                                if a.get_visible()]
    # A twinned axis occupies the same rectangle as the one it twins, so
    # lettering every axis puts two letters in the same corner. Keep the
    # first axis at each position.
    seen, unique = set(), []
    for ax in axes:
        try:
            key = tuple(round(v, 4) for v in ax.get_position().bounds)
        except Exception:
            key = id(ax)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ax)
    axes = unique
    if not axes:
        return []
    try:
        r = fig.canvas.get_renderer()
        w_in = float(fig.get_tightbbox(r).width)
    except Exception:
        w_in = float(fig.get_size_inches()[0])
    pt = PANEL_LETTER_PT / (PAGE_WIDTH_IN / max(w_in, 1e-6))

    out = []
    for i, ax in enumerate(axes):
        letter = chr(ord("A") + start + i)
        t = ax.text(offset[0], offset[1], letter, transform=ax.transAxes,
                    fontsize=pt, fontweight="bold", va="bottom", ha="right",
                    clip_on=False)
        t._fixed_pt = True
        out.append(t)
    return out


def normalise_type_scale(fig):
    """Put every string on the figure onto the shared type scale.

    Sets size by role and family for all of them, so a figure is internally
    consistent and consistent with its neighbours. Set `artist._fixed_pt` to
    keep a particular string out of this -- a mathematical symbol that has to
    match its neighbour, say.

    Runs before the fitting passes, so a figure that cannot hold its labels at
    these sizes still gets shrunk to fit rather than clipped; what it must not
    do is start from a size nobody chose.
    """
    # Some figures carry inset axes, or a hand-tuned relationship between
    # panel size and type size that a single role-based scale destroys: the
    # inset's labels are small *on purpose*, and setting them to the body size
    # makes them larger than the inset. Those figures opt out.
    if getattr(fig, "_skip_type_normalise", False):
        return
    try:
        r = fig.canvas.get_renderer()
        w_in = float(fig.get_tightbbox(r).width)
    except Exception:
        w_in = float(fig.get_size_inches()[0])
    scale = PAGE_WIDTH_IN / max(w_in, 1e-6)

    # The median drawn panel width, in printed inches.
    widths = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        try:
            bb = ax.get_window_extent(renderer=r)
            widths.append(bb.width / fig.dpi)
        except Exception:
            pass
    panel_w = (sorted(widths)[len(widths) // 2] if widths else w_in) * scale
    # A figure that has spare room around its panels can ask for larger type
    # than the panel width alone would give it.
    boost = float(getattr(fig, "_type_boost", 1.0) or 1.0)
    base = min(max(panel_w * TYPE_BASE_PER_IN * boost,
                   TYPE_BASE_MIN), TYPE_BASE_MAX)
    sizes = {k: min(max(v * base, TYPE_MIN), TYPE_MAX)
             for k, v in TYPE_RATIO.items()}

    def put(artist, role):
        if artist is None or getattr(artist, "_fixed_pt", False):
            return
        try:
            if not str(artist.get_text()).strip():
                return
            artist.set_fontsize(sizes[role] / scale)
            artist.set_fontfamily(FONT_STACK)
        except Exception:
            pass

    if getattr(fig, "_suptitle", None) is not None:
        put(fig._suptitle, "axes_title")
    for t in fig.texts:
        put(t, "label")
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        put(ax.title, "axes_title")
        put(ax.xaxis.label, "label")
        put(ax.yaxis.label, "label")
        for t in ax.texts:
            put(t, "annotation")
        for axis in (ax.xaxis, ax.yaxis):
            for lab in axis.get_ticklabels():
                put(lab, "tick")
        leg = ax.get_legend()
        if leg is not None:
            put(leg.get_title(), "legend")
            for t in leg.get_texts():
                put(t, "legend")
    for leg in getattr(fig, "legends", []) or []:
        put(leg.get_title(), "legend")
        for t in leg.get_texts():
            put(t, "legend")


def prepare_for_publication(fig, name=""):
    """Title off, type inside the band. For figures that do not use savefig_dual.

    Three figure scripts save with fig.savefig directly because they manage
    their own layout. They still have to meet the same format, so they call
    this instead of reimplementing it.
    """
    if not PUBLICATION:
        return
    strip_title_for_publication(fig)
    normalise_type_scale(fig)
    # These figures never reach savefig_dual, so they were missing the two
    # passes that keep a key off the data and the axes uncluttered.
    if not getattr(fig, "_keep_grid", False):
        plain_axes(fig, **(getattr(fig, "_plain_axes_kw", None) or {}))
    moved = place_legends_clear(fig)
    up, down = enforce_type_scale(fig)
    if (up or down or moved) and name:
        print(f"    [type] {name}: {down} capped, {up} raised to the floor, "
              f"{len(moved) if moved else 0} legends moved")


def strip_title_for_publication(fig):
    """Take the figure's title off and give the band back to the panels.

    Hiding the suptitle is not enough: the modules reserve room for it with a
    top margin, so a hidden title leaves a strip of white space where the
    title was. Drop the text, then relax the reserved margin.

    The removed string is stashed on the figure so a caller can record it;
    tools/collect_figure_text.py reads the figures before this runs, which is
    where the caption ledger gets the titles from.
    """
    sup = getattr(fig, "_suptitle", None)
    if sup is None or not str(sup.get_text()).strip():
        return None
    text = str(sup.get_text())
    fig._caption_title = text
    sup.set_text("")
    sup.set_visible(False)
    over = dict(getattr(fig, "_margin_overrides", None) or {})
    # Leave a hair of room so ascenders on the top row are not clipped.
    if float(over.get("top", 0.0)) < MARGIN_TOP_TITLE_STRIPPED:
        over["top"] = MARGIN_TOP_TITLE_STRIPPED
    fig._margin_overrides = over
    return text


def enforce_type_scale(fig, lo=None, hi=None):
    """Bring every string on the figure inside the printed-size band.

    Science asks for about 7 pt at the printed size, never below 5, and warns
    against wide variation in type size within one figure. Figures here set
    sizes as BASE_FS + offset, and a figure whose base was tuned when the
    canvas was a different width ends up with axis labels at 15 pt beside
    annotations at 4 pt -- both wrong, in opposite directions.

    Clamping rather than rescaling: a linear remap would shrink the readable
    middle to make room for the outliers, and it is the outliers that are the
    error. Returns the count of artists moved at each end so the caller can
    say so rather than changing the figure silently.
    """
    # A margin inside the legal band. The width this is measured against is
    # the tight bounding box before the save, and the saved box can differ by
    # a few hundredths of an inch, which is enough to push a string set to
    # exactly 5.0 pt back under the floor.
    if getattr(fig, "_skip_type_normalise", False):
        return 0, 0
    lo = (MIN_FS * 1.06) if lo is None else lo
    hi = ((BASE_FS + 2) * 0.96) if hi is None else hi
    try:
        r = fig.canvas.get_renderer()
        w_in = float(fig.get_tightbbox(r).width)
    except Exception:
        w_in = float(fig.get_size_inches()[0])
    # printed = set size x (print width / saved width)
    scale = PAGE_WIDTH_IN / max(w_in, 1e-6)

    raised = lowered = 0

    def fix(artist):
        nonlocal raised, lowered
        try:
            pt = float(artist.get_fontsize())
        except Exception:
            return
        printed = pt * scale
        if printed > hi:
            artist.set_fontsize(hi / scale)
            lowered += 1
        elif printed < lo:
            artist.set_fontsize(lo / scale)
            raised += 1

    if getattr(fig, "_suptitle", None) is not None:
        fix(fig._suptitle)
    for t in fig.texts:
        fix(t)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
            fix(t)
        for t in ax.texts:
            fix(t)
        for axis in (ax.xaxis, ax.yaxis):
            for lab in axis.get_ticklabels():
                fix(lab)
        leg = ax.get_legend()
        if leg is not None:
            if leg.get_title() is not None:
                fix(leg.get_title())
            for t in leg.get_texts():
                fix(t)
    for leg in getattr(fig, "legends", []) or []:
        if leg.get_title() is not None:
            fix(leg.get_title())
        for t in leg.get_texts():
            fix(t)
    return raised, lowered


def savefig_dual(fig, out_path: str, dpi: int = 300, **kwargs) -> tuple:
    """
    Save a figure and a no-suptitle twin of it.

    Outputs:
        <stem>.png            as drawn
        <stem>_notitle.png    figure-level suptitles hidden, for placement in
                              a document that carries its own caption

    Returns (path, notitle_path). No caller reads it; the tuple is kept so the
    signature is unchanged from the version that returned medium/large paths.
    """
    kwargs.setdefault("bbox_inches", "tight")
    stem, ext = os.path.splitext(out_path)
    ext = ext or ".png"

    apply_display_text(fig)
    # Before the margin pass, so the space the title occupied is returned to
    # the panels rather than left blank.
    if PUBLICATION:
        strip_title_for_publication(fig)
        normalise_type_scale(fig)
    check_page_width(fig, os.path.basename(stem))
    check_style_coherence(fig, os.path.basename(stem))

    # Grids off and tick counts capped, for every figure rather than the
    # handful that remembered to ask. Calling it per figure reached 6 of 25 and
    # left gridlines and eight-tick axes across the rest. Set
    # `fig._keep_grid = True` on a figure whose grid is load-bearing.
    if not getattr(fig, "_keep_grid", False):
        plain_axes(fig, **(getattr(fig, "_plain_axes_kw", None) or {}))

    # Margins before text. tight_layout sizes the margins to fit the labels,
    # which hands the canvas to whichever text is longest; fixed margins hand it
    # to the panels and leave overflow visible instead of silently absorbed.
    if not getattr(fig, "_skip_savefig_tight_layout", False):
        tighten_margins(fig)

    # Then make the text fit the space the panels have, and get the legends off
    # the data. Both measure the figure as drawn and only ever shrink or move --
    # nothing here searches for a bigger size, which is what made the previous
    # attempt oscillate.
    _fit = fit_tick_labels(fig)
    _moved = place_legends_clear(fig)
    if _fit:
        print(f"    [fit] {os.path.basename(stem)}: shrank " +
              ", ".join(f"{k}->{v}pt" for k, v in list(_fit.items())[:4]) +
              (f" (+{len(_fit)-4} more)" if len(_fit) > 4 else ""))
    if _moved:
        print(f"    [fit] {os.path.basename(stem)}: legends " +
              ", ".join(f"ax{a}->{loc}@{fs}pt({frac:.0%})"
                        for a, loc, fs, frac in _moved[:4]))

    # Placement that depends on the fitted text goes last.
    for _hook in getattr(fig, "_post_layout_hooks", ()):
        try:
            _hook()
        except Exception as _exc:
            print(f"    [layout] post-layout hook failed: {_exc}")

    # bbox_inches="tight" trims surrounding whitespace, which is wanted, but it
    # also GROWS the canvas to fit anything drawn outside the axes -- a suptitle
    # wider than the figure, a legend pushed out, an over-long panel title. That
    # growth silently defeats the whole page-width policy: the figure is authored
    # at 6.3 in, saved at 9, and shrunk back to 6.3 on placement, taking every
    # point size down with it. Measure it and say so, naming it as the content
    # problem it is.
    _report_overflow(fig, os.path.basename(stem), kwargs.get("bbox_inches"))
    report_layout(fig, os.path.basename(stem))

    # Tick labels last. A tick's text comes from its formatter on every draw,
    # so rewriting the Text artist earlier is undone by the next draw -- which
    # is why the internal lifestyle keys survived onto exactly those figures
    # that carry the lifestyle as a tick label. Freeze the corrected strings
    # now that the layout has settled, with a plain list rather than a closure
    # so the figure still pickles for the cache.
    freeze_display_ticks(fig)

    # Last thing before the save, so it sees the sizes the fitting passes
    # settled on rather than the ones the figure was authored with.
    if PUBLICATION:
        _up, _down = enforce_type_scale(fig)
        if _up or _down:
            print(f"    [type] {os.path.basename(stem)}: "
                  f"{_down} strings capped at {BASE_FS + 2} pt, "
                  f"{_up} raised to the {MIN_FS} pt floor")

    main_path = f"{stem}{ext}"
    fig.savefig(main_path, dpi=dpi, **kwargs)

    # Then measure what was actually written and, if it is wider than the
    # column, shrink and write it again. The tight bounding box computed from
    # the live figure disagrees with what savefig produces whenever artists
    # sit outside the axes -- it read 6.93 in for a file that came out at
    # 8.56 -- so the file is the only trustworthy measurement.
    if (PUBLICATION and kwargs.get("bbox_inches") == "tight"
            and not getattr(fig, "_wide_on_purpose", False)
            and main_path.lower().endswith(".png")):
        _w0 = _png_width_in(main_path, dpi)
        for _ in range(3):
            _w = _png_width_in(main_path, dpi)
            if _w is None or _w <= PAGE_WIDTH_IN + 0.02:
                break
            _sz = fig.get_size_inches()
            _k = max(PAGE_WIDTH_IN / _w, 0.70)
            fig.set_size_inches(_sz[0] * _k, _sz[1] * _k, forward=True)
            fig.savefig(main_path, dpi=dpi, **kwargs)
        _w1 = _png_width_in(main_path, dpi)
        if _w0 and _w1 and _w0 - _w1 > 0.05:
            print(f"    [width] {os.path.basename(stem)}: "
                  f"{_w0:.2f} in -> {_w1:.2f} in to fit the column")

    notitle_path = f"{stem}_notitle{ext}"
    hidden = _hide_labels(fig)
    fig.savefig(notitle_path, dpi=dpi, **kwargs)
    _restore_labels(hidden)

    # Last, because caching removes the annotation layer from the live figure
    # and both images have to be written from the annotated one first.
    _cache_figure(fig, stem)

    return main_path, notitle_path


def _hide_labels(fig):
    """Hide only figure-level titles; keep axes titles, labels, ticks, legends, and notes.

    A figure-level text that is part of the layout rather than a title (column
    headings, row labels) sets `_keep_in_notitle = True`; without that the
    no-title twin loses it, which is how S15 lost Figure 3's "Samples" /
    "Studies" headings and its reference labels.

    supxlabel and supylabel are kept unconditionally. They are axis labels that
    happen to be stored figure-level, not titles, and a caption cannot stand in
    for them: hiding them is how S2, S4, S10 and S23 reached the staged set --
    the images a document actually places -- with bare tick numbers and no
    statement of what was counted or in what units. `share_axis_labels` exists
    to move a repeated per-panel label up to the figure, so every figure that
    took that advice was losing its labels by taking it.
    """
    keep = {id(t) for t in (getattr(fig, "_supxlabel", None),
                            getattr(fig, "_supylabel", None)) if t is not None}
    hidden = []
    for t in fig.texts:
        if not t.get_text():
            continue
        if id(t) in keep or getattr(t, "_keep_in_notitle", False):
            continue
        hidden.append((t, t.get_visible()))
        t.set_visible(False)
    return hidden


def _restore_labels(hidden):
    """Restore visibility of hidden artists."""
    for obj, vis in hidden:
        obj.set_visible(vis)


# ══════════════════════════════════════════════════════════════════════════════
#  THE LIST-vs-BACKGROUND PREVALENCE CELL  (Figure 5E and Figure S27)
# ══════════════════════════════════════════════════════════════════════════════
#
# Figure 5E and Figure S27 draw the same comparison -- a taxon list's
# case-control prevalence shift against the shift of taxa that are equally
# common -- and they used to draw it with two copies of the same forty lines.
# They had already drifted apart in what they printed on the panel. One
# renderer, used by both, so "S27 is formatted like 5E" stays true without
# anyone having to remember it.


def weighted_kde(vals, weights, x_eval):
    """Gaussian KDE of `vals` under `weights`, evaluated on `x_eval`.

    Falls back to the unweighted density, then to zeros, so a panel with one
    usable taxon draws empty rather than raising.
    """
    import numpy as np
    from scipy.stats import gaussian_kde
    vals = np.asarray(vals, dtype=float)
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    mask = np.isfinite(vals) & (weights > 0)
    vals, weights = vals[mask], weights[mask]
    if len(vals) < 2:
        return np.zeros_like(np.asarray(x_eval, dtype=float))
    try:
        return np.maximum(
            gaussian_kde(vals, weights=weights / weights.sum(),
                         bw_method="scott")(x_eval), 0.0)
    except Exception:
        try:
            return np.maximum(gaussian_kde(vals, bw_method="scott")(x_eval), 0.0)
        except Exception:
            return np.zeros_like(np.asarray(x_eval, dtype=float))


def weighted_median(vals, weights):
    """Median of `vals` under `weights`."""
    import numpy as np
    vals = np.asarray(vals, dtype=float)
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if len(vals) == 0 or weights.sum() <= 0:
        return float("nan")
    order = np.argsort(vals)
    cum = np.cumsum(weights[order]) / weights.sum()
    return float(vals[order[np.searchsorted(cum, 0.5)]])


def style_cell(ax, fs, box=True):
    """The panel furniture both figures use: thin box, small ticks, no grid."""
    ax.grid(False)
    ax.set_axisbelow(False)
    ax.tick_params(axis="both", labelsize=fs - 1, length=3)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)
        sp.set_visible(True)
    if not box:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def draw_prevalence_cell(ax, list_vals, bg_vals, bg_weights, colour, bg_colour,
                         fs, curve_row=None, fallback_label=None,
                         show_n_bg=True):
    """One list-vs-background cell, drawn the way Figure 5E draws it.

    `list_vals` and `bg_vals` are case - control prevalence shifts in points
    (positive = enriched in disease); `bg_weights` weights the background taxa
    so the comparison is at the list's own prevalence.

    `curve_row` is the list's row from the background-curve summary. When it is
    present the panel prints the mean distance from the curve and its bootstrap
    CI -- the statistic the manuscript quotes. `fallback_label` is what to print
    instead when it is not (an older data object, where the energy-distance
    star is all there is).

    Returns dict(ovl, median_list, median_bg, xmax).
    """
    import numpy as np
    _trap = getattr(np, "trapezoid", np.trapz)

    list_vals = np.asarray(list_vals, dtype=float)
    bg_vals = np.asarray(bg_vals, dtype=float)
    bg_weights = np.asarray(bg_weights, dtype=float)

    style_cell(ax, fs)

    all_v = np.concatenate([list_vals, bg_vals])
    q = max(abs(np.percentile(all_v, 0.5)), abs(np.percentile(all_v, 99.5)), 1.0)
    xmax = q * 1.30
    xr = np.linspace(-xmax, xmax, 400)
    dx = xr[1] - xr[0]

    kde_l = weighted_kde(list_vals, np.ones(len(list_vals)), xr)
    kde_b = weighted_kde(bg_vals, bg_weights, xr)
    ovlap = np.minimum(kde_l, kde_b)
    ovl = float(_trap(ovlap, dx=dx))

    ax.fill_between(xr, ovlap, alpha=0.55, color="#bbbbbb", zorder=1)
    ax.fill_between(xr, kde_b, alpha=0.20, color=bg_colour, zorder=2)
    ax.fill_between(xr, kde_l, alpha=0.38, color=colour, zorder=3)
    ax.plot(xr, kde_l, color=colour, lw=1.8, zorder=4)
    ax.plot(xr, kde_b, color=bg_colour, lw=1.4, ls="--", zorder=4)

    med_l = float(np.median(list_vals))
    med_b = weighted_median(bg_vals, bg_weights)
    ax.axvline(med_l, color=colour, lw=1.8, alpha=0.90, zorder=5)
    if np.isfinite(med_b):
        ax.axvline(med_b, color=bg_colour, lw=1.4, ls="--", alpha=0.85, zorder=5)
    ax.axvline(0, color="black", lw=0.5, ls=":", alpha=0.25)

    def _star(p):
        if not np.isfinite(p):
            return ""
        return ("***" if p < 0.001 else "**" if p < 0.01
                else "*" if p < 0.05 else "ns")

    if curve_row is not None and np.isfinite(curve_row.get("mean_z", np.nan)):
        p_z = float(curve_row.get("boot_p", np.nan))
        sig = np.isfinite(p_z) and p_z < 0.05
        ax.text(0.97, 0.97, f"z {curve_row['mean_z']:+.2f}{_star(p_z)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=fs - 1,
                fontweight="bold" if sig else "normal",
                color="black" if sig else "#888888")
        ax.text(0.97, 0.76,
                f"[{curve_row['ci_lo']:+.2f}, {curve_row['ci_hi']:+.2f}]",
                transform=ax.transAxes, ha="right", va="top", fontsize=fs - 1,
                color="#555555")
    elif fallback_label:
        text, sig = fallback_label
        ax.text(0.97, 0.97, text, transform=ax.transAxes, ha="right", va="top",
                fontsize=fs - 1, fontweight="bold" if sig else "normal",
                color="black" if sig else "#888888")
        ax.text(0.97, 0.76, f"OVL {ovl:.0%}", transform=ax.transAxes,
                ha="right", va="top", fontsize=fs - 1, color="#555555")

    if show_n_bg:
        ax.text(0.03, 0.97, f"n_bg={len(bg_vals)}", transform=ax.transAxes,
                ha="left", va="top", fontsize=fs - 1,
                color=_desaturate(colour, desat=0.7, bright=0.7))

    # The largest round step that still lands inside the panel. A fixed
    # +/-10% left the narrow panels showing nothing but zero.
    r_tick = max([t for t in (1, 2, 5, 10, 20, 50) if t <= xmax * 0.80],
                 default=0)
    ax.set_xticks([-r_tick, 0, r_tick] if r_tick else [0])
    ax.xaxis.set_major_formatter(_plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.tick_params(axis="x", labelsize=fs - 1, rotation=0)
    ax.set_xlim(-xmax * 1.18, xmax * 1.18)
    ax.set_yticks([])
    return dict(ovl=ovl, median_list=med_l, median_bg=med_b, xmax=xmax)


def _desaturate(hex_colour, desat=0.35, bright=1.20):
    """A paler, lighter twin of a colour, for the background series."""
    import colorsys
    h = hex_colour.lstrip("#")
    r, g, b = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    hh, s, v = colorsys.rgb_to_hsv(r, g, b)
    nr, ng, nb = colorsys.hsv_to_rgb(hh, s * desat, min(v * bright, 1.0))
    return "#{:02x}{:02x}{:02x}".format(int(nr * 255), int(ng * 255), int(nb * 255))


# ── the background-curve figure ──────────────────────────────────────────────
#
# Main-text Figure 5E and supplementary S26 show the same comparison for
# different taxon lists, and they must be indistinguishable in formatting.
# The reason they kept drifting is that they are drawn on different canvases:
# S26 is authored at the printed column width, Figure 5 is authored 24 inches
# wide and reduced to that width on save. Every size therefore means something
# different in the two call sites, and matching them by hand failed each time.
#
# So sizes are declared ONCE here, in PRINTED points, and each caller passes
# the magnification of its own canvas (canvas width / column width). Point
# sizes scale by mag, marker areas by mag squared. Nothing outside this block
# sets a size for these panels.

PRINTED_PT = {
    "annot":     6.4,   # the mean-z box
    "tick":      6.0,
    "axis":      6.5,
    "title":     7.0,   # condition heading
    "legend":    6.5,
}
# Science: no rule below 0.5 pt at final size. "zero" and "hollow" are drawn
# rules -- the y=0 reference line, and the outline that marks a taxon with no
# comparable background -- and both sat at 0.4, which would print under the
# floor. "edge" is not a rule: it is a hairline white separator so adjacent
# filled markers stay countable, carries no information of its own, and is
# deliberately left below the floor.
PRINTED_LW = {"curve": 0.9, "zero": 0.5, "hollow": 0.5, "edge": 0.15,
              "spine": 0.6, "tick": 0.6}
PRINTED_S  = {"marker": 5.0, "bg": 1.4, "legend_ms": 6.0, "tick_len": 2.4}
# Height / width of each cell's DATA BOX, forced with set_box_aspect so the
# two figures' cells are the same shape whatever their surrounding layout
# does. Figure 5's gridspec gave 1.02 and S26's tight_layout gave 0.885 --
# the cells were simply different rectangles, which no amount of matching
# type sizes could hide.
CELL_ASPECT = 1.0


# Gaps between cells, as gridspec fractions. Shared for the same reason the
# sizes are: the module-level MARGINS carry wspace 0.16 / hspace 0.28 and
# tighten_margins applies them to any figure that does not override, so S25
# was being reset to the pipeline default while Figure 5's own gridspec kept
# these. Both figures draw the same 2 x N grid, so they must take the same
# two numbers or they stop looking like the same figure.
#
# Tightened from 0.26 / 0.17 on 2026-10-16. The horizontal gap is still the
# wider of the two, and cannot go much below this: every column carries its
# own y range and so its own tick labels, and at 0.10 the end labels of
# adjacent columns collided ("100%0.01%"). The vertical gap had far more to
# give, because the upper row carries no x-tick labels at all -- that space
# was simply blank.
CELL_WSPACE = 0.18
CELL_HSPACE = 0.09


def background_curve_figsize(width_in, n_cols, n_rows, bottom_in,
                             margins=None, titled=True):
    """Figure height that makes each gridspec cell square at the shared
    spacing, using the margins the figure is actually SAVED with.

    set_box_aspect forces the drawn box square, but if the gridspec cell is
    not square the box shrinks inside it and the surplus shows as whitespace
    -- horizontally in a wide short figure, vertically in a tall one. Sizing
    the cell right removes the surplus instead of hiding it.

    The top margin has to come from drawn_margins(): a figure authored with a
    suptitle reserves room for it and the publication path then strips the
    title and raises the margin to 0.975, so a height computed from the
    authored 0.90 leaves exactly the blank band between rows that this is
    meant to avoid.

    Returns (height_in, cell_w_in).
    """
    m = drawn_margins(margins or {"left": MARGINS["left"],
                                  "right": MARGINS["right"],
                                  "top": MARGINS["top"]}, titled=titled)
    left = float(m.get("left", MARGINS["left"]))
    right = float(m.get("right", MARGINS["right"]))
    top = float(m.get("top", MARGINS["top"]))
    avail_w = (right - left) * width_in
    cell_w = avail_w / (n_cols + (n_cols - 1) * CELL_WSPACE)
    axes_h = (n_rows + (n_rows - 1) * CELL_HSPACE) * cell_w * CELL_ASPECT
    return (axes_h + bottom_in) / top, cell_w


def background_curve_style(mag=1.0):
    """Drawing sizes for the background-curve panels on a canvas `mag` times
    the printed column width. See the note above: this is the only place the
    figure's type, line weights and marker sizes are set."""
    st = {f"fs_{k}": v * mag for k, v in PRINTED_PT.items()}
    st.update({f"lw_{k}": v * mag for k, v in PRINTED_LW.items()})
    st["s_marker"] = PRINTED_S["marker"] * mag ** 2
    st["s_bg"] = PRINTED_S["bg"] * mag ** 2
    st["legend_ms"] = PRINTED_S["legend_ms"] * mag
    st["tick_len"] = PRINTED_S["tick_len"] * mag
    st["mag"] = mag
    return st


def style_background_curve_axes(ax, st):
    """Spines, ticks and box shape for one cell.

    These are the parts drawn AROUND the data, and they were the last thing
    still set per figure: Figure 5 applied its own _style() with an absolute
    0.7 pt spine and 3 pt ticks on a 24 in canvas, which print at 0.21 and
    0.9 pt, against S26's rcParam defaults at the column width. Same place,
    same printed sizes, for both.
    """
    ax.grid(False)
    ax.set_axisbelow(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(st["lw_spine"])
    ax.tick_params(axis="both", which="major",
                   labelsize=st["fs_tick"], width=st["lw_tick"],
                   length=st["tick_len"])
    ax.tick_params(axis="both", which="minor", length=0)
    ax.set_box_aspect(CELL_ASPECT)
    return ax


GENERAL_CONTRASTS = ("Unhealthy", "Obese")


def display_order(conds):
    """Condition order for figures: the pooled contrast goes last.

    Every health figure takes its column order from the order the contrasts
    were computed in, and the registries list the pooled "Unhealthy" contrast
    first because it is what the others are compared against. On the page that
    reads backwards, so it moves to the end -- for DISPLAY only; the tables
    keep computation order.

    Lives here because both section3_health and figures/figure5 need it, and
    a copy in each is how two implementations of one rule drift apart.
    """
    conds = list(conds)
    last = [c for c in conds if c == "Unhealthy"]
    return [c for c in conds if c != "Unhealthy"] + last


def group_dividers(conds):
    """x positions for the dashed rule between pooled and specific contrasts.

    A boundary wherever the group changes. The old `count(pooled) - 0.5` was
    right only while the pooled contrasts led the row; display_order moves
    Unhealthy to the end, so they are no longer contiguous.
    """
    grp = [c in GENERAL_CONTRASTS for c in conds]
    return [i + 0.5 for i in range(len(grp) - 1) if grp[i] != grp[i + 1]]


def boot_star(p, n_boot=None):
    """Stars for a bootstrap p-value, never finer than the bootstrap's own
    resolution.

    A percentile bootstrap cannot return a two-sided p below 1/n_boot, so a
    ladder whose finest tier sits at or below that floor advertises a result
    the method can never produce: at the 200 replicates this figure runs,
    "***" (p < 0.001) was unreachable and every cell that reached the floor
    printed "**". Tiers the resolution cannot distinguish are dropped rather
    than shown and never earned. Raising n_boot to make "***" reachable costs
    about three hours per detection spec, because each replicate refits the
    curve; the resolution is reported in the caption instead.
    """
    tiers = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
    if n_boot:
        floor = 1.0 / float(n_boot)
        tiers = [(t, star) for t, star in tiers if t > floor]
    for t, star in tiers:
        if p < t:
            return star
    return "n.s."


def draw_background_curve_cell(ax, frame, members, grid, li, row, colour,
                               style=None, floor=1e-4, stats=True,
                               n_boot=None):
    """One cell: a taxon list against the prevalence background.

    Draws, in the order they must stack: every background taxon as a faint
    point, the fitted curve's +/-1 SD band, the curve itself, then the list
    taxa on top. The curve goes above the list markers because it is what the
    panel is read against and must never be the thing they hide.
    """
    import numpy as np
    st = style or background_curve_style()
    if frame is not None and len(frame):
        bgm = ~frame.index.isin(set(members))
        ax.scatter(frame["x"].values[bgm], frame["shift"].values[bgm],
                   s=st["s_bg"], color="0.83", alpha=0.45, lw=0,
                   rasterized=True)
    if grid is not None and len(grid):
        ax.fill_between(grid["x"], grid["mu"] - grid["sd"],
                        grid["mu"] + grid["sd"],
                        color="0.55", alpha=0.25, lw=0)
        ax.plot(grid["x"], grid["mu"], color="0.15", lw=st["lw_curve"],
                zorder=5)
    ax.axhline(0, color="k", lw=st["lw_zero"])

    sup = li["supported"].values.astype(bool)
    lx = np.log10(np.maximum(li["prev"].values, floor))
    ax.scatter(lx[sup], li["shift"].values[sup], s=st["s_marker"],
               color=colour, edgecolor="white", lw=st["lw_edge"], zorder=3)
    if (~sup).any():
        ax.scatter(lx[~sup], li["shift"].values[~sup], s=st["s_marker"],
                   facecolor="none", edgecolor=colour, lw=st["lw_hollow"],
                   zorder=3)

    if stats and row is not None:
        star = boot_star(float(row["boot_p"]), n_boot)
        # "mean z" and the interval stay regular; the value carries the
        # emphasis, and the stars only when they say something. Built from
        # separate text runs rather than one bold string, because one string
        # cannot carry two weights.
        from matplotlib.offsetbox import (TextArea, HPacker, VPacker,
                                          AnchoredOffsetbox)
        fs = st["fs_annot"]
        reg = dict(fontsize=fs)
        bold = dict(fontsize=fs, fontweight="bold")
        runs = [TextArea("mean z ", textprops=reg),
                TextArea(f"{row['mean_z']:+.2f}", textprops=bold)]
        if star:
            runs.append(TextArea(" " + star,
                                 textprops=bold if star != "n.s." else reg))
        box = VPacker(
            children=[HPacker(children=runs, align="baseline", pad=0, sep=0),
                      TextArea(f"[{row['ci_lo']:+.2f}, {row['ci_hi']:+.2f}]",
                               textprops=reg)],
            align="left", pad=0, sep=0.18 * fs)
        anc = AnchoredOffsetbox(loc="upper left", child=box, frameon=True,
                                pad=0.15, borderpad=0.25,
                                bbox_to_anchor=(0.0, 1.0),
                                bbox_transform=ax.transAxes)
        anc.patch.set(facecolor="white", edgecolor="none", alpha=0.85)
        anc.set_zorder(6)
        ax.add_artist(anc)

    if frame is not None and len(frame):
        lo, hi = np.percentile(frame["shift"].values, [0.3, 99.7])
        lo = min(lo, float(np.min(li["shift"].values)))
        hi = max(hi, float(np.max(li["shift"].values)))
        pad = (hi - lo) * 0.06
        ax.set_ylim(lo - pad, hi + (hi - lo) * 0.22)
    ax.set_xlim(-4.15, 0.15)


X_TICKS = [-4, -2, 0]
X_LABELS = ["0.01%", "1%", "100%"]


def text_outline(colour, size, max_lum=0.70):
    """A thin dark edge for a pale glyph, or None where none is needed.

    Coloured label text has to be legible on white *and* has to be the same
    colour as the thing it names, or it stops working as a key. Those pull
    against each other at the pale end of the palette: VANISH+ yellow
    (#F0E442) set as text on white is close to invisible. Darkening it buys
    the first by breaking the second, so the glyph keeps its exact colour and
    takes an edge instead -- which is what the marker it names already has.

    Only the genuinely pale entries qualify. VanishH+ orange and BlossumH+ sky
    sit below the threshold and print plain, in exactly the colour of the
    points they label.
    """
    r, g, b = _mpl.colors.to_rgb(colour)
    if 0.2126 * r + 0.7152 * g + 0.0722 * b <= max_lum:
        return None
    from matplotlib import patheffects as _pe
    return [_pe.withStroke(linewidth=max(0.7, float(size) * 0.05),
                           foreground="#333333")]


def rich_axis_label(ax, lines, axis="x", gap=0.02, sep=2.0, align_group=None):
    """An axis label whose words can carry their own colour and size.

    matplotlib gives a label one colour and one size. Where part of a label
    names the series it belongs to -- "VanishH+", "BloSSUM" -- that word is
    doing the work of a key, and setting it in the series' own colour spares
    the reader the lookup; the rest of the label stays as it was.

    `lines` is a list of lines, each a list of (text, colour or None, size or
    None) pieces. A piece with no colour or size takes the axis-label default,
    and a coloured piece is bold. On a y axis the text is rotated and so reads
    bottom to top, which is why a line's pieces are packed in reverse.

    The label is anchored outside the axes and reserves no space of its own,
    exactly like the label it replaces: the caller's margins must already hold
    it. Returns the artist.

    `align_group` is a shared list passed to several calls that must line up.
    Each label is otherwise placed just clear of ITS OWN tick labels, so two
    rows of the same column part company whenever their ticks differ in width
    -- Figure 5E's upper row reads "0, 2, 4, 6" and its lower row "0.0, 2.5,
    ... 10.0", which pushed the lower label visibly further left. Members of a
    group are placed at the innermost of their positions instead, so the wider
    row's label moves in to meet the narrower one rather than the other way
    round; nothing moves outward, so no label can collide with its own ticks.
    """
    from matplotlib.offsetbox import (TextArea, HPacker, VPacker,
                                      AnchoredOffsetbox)
    base_fs = _mpl.rcParams["axes.labelsize"]
    base_c = _mpl.rcParams.get("axes.labelcolor", "black")
    rot = 90 if axis == "y" else 0

    def _piece(p):
        text = p[0]
        colour = p[1] if len(p) > 1 else None
        size = (p[2] if len(p) > 2 else None) or base_fs
        props = dict(color=colour or base_c, fontsize=size, rotation=rot,
                     fontweight="bold" if colour else "normal")
        if colour:
            edge = text_outline(colour, size)
            if edge:
                props["path_effects"] = edge
        return TextArea(text, textprops=props)

    if axis == "y":
        packed = [VPacker(children=[_piece(p) for p in reversed(line)],
                          align="center", pad=0, sep=0) for line in lines]
        box = HPacker(children=packed, align="center", pad=0, sep=sep)
        anchor = AnchoredOffsetbox(
            loc="center right", child=box, pad=0, borderpad=0, frameon=False,
            bbox_to_anchor=(-gap, 0.5), bbox_transform=ax.transAxes)
    else:
        packed = [HPacker(children=[_piece(p) for p in line],
                          align="baseline", pad=0, sep=0) for line in lines]
        box = VPacker(children=packed, align="center", pad=0, sep=sep)
        anchor = AnchoredOffsetbox(
            loc="upper center", child=box, pad=0, borderpad=0, frameon=False,
            bbox_to_anchor=(0.5, -gap), bbox_transform=ax.transAxes)
    anchor.set_clip_on(False)
    ax.add_artist(anchor)

    # An ordinary axis label is positioned clear of the tick labels; an
    # anchored box is not, and a fraction of the axes is the wrong unit for
    # the job -- what it has to clear is a strip of digits whose width depends
    # on the numbers and the type size, not on how wide the panel is. So the
    # offset is measured at draw time, when the tick labels have been laid
    # out, and `gap` is only the small space left beyond them.
    state = {"placed": False}

    def _clear_ticks(event=None):
        if state["placed"]:
            return
        r = getattr(event, "renderer", None)
        if r is None:
            try:
                r = ax.figure.canvas.get_renderer()
            except AttributeError:
                return
        axb = ax.get_window_extent(r)
        getter = ax.get_yticklabels if axis == "y" else ax.get_xticklabels
        boxes = [t.get_window_extent(r) for t in getter() if t.get_text()]
        if axis == "y":
            edge = min([b.x0 for b in boxes], default=axb.x0)
            frac = (edge - axb.x0) / max(axb.width, 1.0) - gap
            if align_group is None:
                anchor.set_bbox_to_anchor((frac, 0.5), transform=ax.transAxes)
            else:
                # Hold the innermost (largest, i.e. least negative) position of
                # the group. Re-applied to every member each time one places,
                # so the group converges once the last of them has been laid
                # out, whatever order the draw visits them in.
                align_group.append((anchor, ax, frac))
                best = max(f for _, _, f in align_group)
                for a, a_ax, _ in align_group:
                    a.set_bbox_to_anchor((best, 0.5),
                                         transform=a_ax.transAxes)
        else:
            edge = min([b.y0 for b in boxes], default=axb.y0)
            anchor.set_bbox_to_anchor(
                (0.5, (edge - axb.y0) / max(axb.height, 1.0) - gap),
                transform=ax.transAxes)
        state["placed"] = True

    ax.figure.canvas.mpl_connect("draw_event", _clear_ticks)
    return anchor


def draw_background_curve_grid(axes, cond_order, focus_lists, frames, taxa,
                               grid, summary, colours, labels, style=None,
                               floor=1e-4, display=None, titles=True,
                               n_boot=None, colour_row_labels=False):
    """The whole 2 x N grid, including every title, label and tick.

    `axes[i][j]` is the cell for focus_lists[i] x cond_order[j]. Both callers
    hand their own axes in and this function does the rest, so neither can
    set a size of its own. `summary` maps (condition, list) to the row with
    mean_z / ci_lo / ci_hi / boot_p; `display` renames a condition or list
    for the page.
    """
    st = style or background_curve_style()
    disp = display or (lambda s: s)
    #: Shared by every row label so they line up with each other; see
    #: rich_axis_label's `align_group`.
    _row_label_group = []
    for j, cond in enumerate(cond_order):
        frame = frames.get(cond)
        for i, list_name in enumerate(focus_lists):
            ax = axes[i][j]
            li = taxa[(taxa["Condition"] == cond)
                      & (taxa["Taxon_list"] == list_name)]
            if li.empty:
                ax.axis("off")
                continue
            g = grid[(grid["Condition"] == cond)
                     & (grid["Taxon_list"] == list_name)] if grid is not None \
                else None
            draw_background_curve_cell(
                ax, frame, set(li["taxon"]), g, li,
                summary.get((cond, list_name)),
                colours.get(list_name, "#333333"), style=st, floor=floor,
                n_boot=n_boot)
            style_background_curve_axes(ax, st)
            ax.set_xticks(X_TICKS)
            last_row = (i == len(focus_lists) - 1)
            ax.set_xticklabels(X_LABELS if last_row else [])
            if last_row:
                ax.set_xlabel("Control prevalence", fontsize=st["fs_axis"],
                              labelpad=2)
            # `titles=False` for main-text Figure 5E: panel D sits directly
            # above it and already heads each column, with the cohort note.
            # Repeating them would label every column twice.
            if i == 0 and titles:
                ax.set_title(disp(cond), fontsize=st["fs_title"],
                             fontweight="bold", pad=3)
            if j == 0 and colour_row_labels:
                # The list name is the row's key, so it is set in the row's
                # own colour and larger than the unit it shares the label
                # with. Figure 5E asks for this; the supplementary twin keeps
                # the plain label, where the row is already titled.
                #
                # Every row label shares one x, via `_row_label_group`: the
                # rows' y-ticks differ in width, and placing each clear of its
                # own ticks left the two names out of line down the page.
                rich_axis_label(
                    ax,
                    [[(disp(list_name), colours.get(list_name, "#333333"),
                       st["fs_axis"] * 1.45)],
                     [("case $-$ control (%)", None, st["fs_axis"])]],
                    axis="y", gap=0.03, sep=st["fs_axis"] * 0.35,
                    align_group=_row_label_group)
            elif j == 0:
                ax.set_ylabel(f"{disp(list_name)}\ncase $-$ control (%)",
                              fontsize=st["fs_axis"], labelpad=2)


def background_curve_legend(focus_lists, colours, labels, style=None):
    """Legend handles for the background-curve figure, in drawing order.

    Built rather than harvested: harvesting from one panel returns the first
    list's marker and silently drops the second's. The hollow "no comparable
    taxa" entry was removed on 2026-10-07 -- it explained a marker that is
    rare on the page and cost more attention than it returned.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    st = style or background_curve_style()
    ms = st["legend_ms"]
    handles = [Line2D([], [], marker="o", ls="", color="0.75", ms=ms,
                      label="background taxa"),
               Patch(facecolor="0.55", alpha=0.25,
                     label="background mean $\\pm$ 1 SD")]
    for name in focus_lists:
        handles.append(Line2D([], [], marker="o", ls="",
                              color=colours.get(name, "#333333"),
                              mec="white", mew=0.3, ms=ms,
                              label=labels.get(name, name)))
    return handles


