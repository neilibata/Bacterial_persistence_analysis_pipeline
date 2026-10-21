# How the figures are sized and typeset

Every figure is authored at the width it will be printed at. That is the whole
policy; everything below follows from it.

## Why

A figure authored *W* inches wide and placed in a column *C* inches wide is
scaled by *C* / *W*, and every point size scales with it. A figure drawn 30
inches wide and printed at 7 needs about 30 pt type to arrive at 7 pt on the
page — and type that size consumes the canvas the panels were meant to occupy.

Measuring the shipped figures showed exactly that: the only ones that came out
legible were those authored near their printed width, and every starved one had
been authored at several times it.

Authoring at the printed width also makes the type specification literal. A
figure written with 7 pt type is 7 pt on the page, so compliance can be checked
by reading the code instead of measuring the PNG.

## The three widths

Science publishes figures at one of three column widths, and
`persistence/plot_utils.py` names all three:

| Constant | Inches | Centimetres |
| --- | ---: | ---: |
| `COLUMN_1_IN` | 2.24 | 5.7 |
| `COLUMN_2_IN` | 4.76 | 12.1 |
| `COLUMN_3_IN` = `PAGE_WIDTH_IN` | 7.24 | 18.4 |

`page_figsize()` returns a size that respects the chosen width:

```python
from .plot_utils import page_figsize, panel_grid, share_axis_labels

fig, ax   = plt.subplots(figsize=page_figsize())                      # one panel
fig, axes = plt.subplots(1, 3, figsize=page_figsize(n_cols=3))        # a row
fig, axes = plt.subplots(3, 3, figsize=page_figsize(n_cols=3, n_rows=3))
nr, nc    = panel_grid(n_conditions)    # wrap many panels rather than widening
```

It takes `width_in`, `height_in`, `max_height_in` and `aspect` as well. Width
is the page; only height varies with content. `panel_grid()` wraps a long row
of panels into a grid instead of letting the canvas grow sideways.

## Type

`BASE_FS = 7`, because Science asks for type at "a reduced size of about 7
points (2.5 mm) high, and not smaller than 5 points". Ticks and annotations
step down from `BASE_FS`, so the floor matters: `MIN_FS = 5` is a hard limit
and nothing ships below it.

Figure code sets explicit sizes as `BASE_FS + offset`. Differences within one
figure are wanted — an in-axes note should not compete with a panel title — but
the offsets are relative to the base so that changing the base moves everything
together.

## Resolution

`SUBMISSION_DPI = 300`, from "please save images at a resolution of 300 dots
per inch (dpi) for initial submission". Figures had been written at 180, 220,
300 and 360; the three below 300 were not submittable.

## Order of priority

Graphics get the space first; text gets as large as it can without intruding.
Never shrink a panel to make a label fit — cut the label.

`share_axis_labels()` is the cheapest cut available. On a 3 × 3 grid, replacing
nine repeated x-labels and nine repeated y-labels with one shared pair moved
graphics from 1.07× the text to 5.97× and cut 188 text runs to 27, losing
nothing.

## Two widths, two purposes

Do not confuse the **authoring** width with the **staging** width.

`savefig_dual()` writes `<stem>.png` and `<stem>_notitle.png` at the authoring
width and 300 dpi. The no-title twin is the one that goes into the manuscript,
which carries its own captions: `tools/stage_figures_for_docx.py` prefers it
and falls back to `<stem>.png`, while `tools/link_supplementary_figures.py`
takes `<stem>.png` only.

The figures already embedded in the supplementary Word document are 220 dpi at
about 1378 px — a natural size of 6.3 in. Dropping raw pipeline output in
beside them would place it at a different scale, so staging resamples to match:

```bash
python tools/stage_figures_for_docx.py --out staged_figures --width-in 6.3
```

That 6.3 in is a property of the existing document, not of the figure design.

The `_medium` / `_large` presets and the `MICROBIOME_AUTOFIT` layer were
retired. Both existed to compensate for a shrink that authoring at printed
width removes.

## Checking a figure

`tools/figure_layout_probe.py` measures rendered figures without re-plotting
them: text runs with their printed point size, the graphics-to-text ink ratio,
how far the graphics reach across the page, and overlapping labels.

```bash
python tools/figure_layout_probe.py --from-manifest \
       --manifest tools/additional_figures_manifest.txt -o docs/layout_probe.json
```

Probe the full-resolution sources, not the staged copies. Downsampling loses
thin data strokes and drops sub-3-pixel glyphs below the noise floor, which
biases both the ink budget and the median point size.
