# `accumulation_sinf_inputs/` — fitted curves for Figures S16 and S17

Figures S16 and S17 ask whether the taxa that differ between lifestyles are
genuinely absent from industrial populations or merely less prevalent there.
Both plot **repeated-detection accumulation**: how many catalogued SGBs are
found as samples are pooled, when a taxon must appear in more than one sample
to count, or must clear a stricter per-sample coverage threshold. The quantity
of interest is the asymptote, **S∞** — what the curve converges to as pooling
continues.

This directory holds the fitted curves those two figures are drawn from.
`tools/make_accumulation_sinf_figures.py` reads them and draws the figures;
`tools/sinf_chain/` is what produces them.

The method is **Supplementary Text 4** of the manuscript — §4D for why
repeated detection needs a different model from first detection, §4E–4F for the
beta-binomial and beta-mixture models, §4H for the bootstrap and the asymptote
intervals.

## The two figures

| Slot | Output file | What it varies |
| --- | --- | --- |
| **S16** | `figure_a17_accumulation_sinf_mink_medium.png` | Detection in at least *k* samples, for k = 1, 2, 3, 5 |
| **S17** | `figure_a16_accumulation_sinf_thresholds_medium.png` | Per-sample breadth threshold, at ≥10%, ≥30% and ≥50% |

Both are written to `results/Figure3_combined/`.

Each is a grid of rows by three columns: the **accumulation curve**; the
**support for S∞**, a profile-likelihood landscape with the χ²₁ 95% cutoff; and
**bootstrap S∞**, a histogram of the asymptote across replicates.

> The filenames carry an earlier numbering and are inverted against the slots:
> `figure_a17_…` is Figure S16, and `figure_a16_…` is Figure S17.
> `tools/additional_figures_manifest.txt` is the authority on which file fills
> which slot.

## What is here

`caches/` holds accumulation curves and bootstrap replicates, written as pickle
protocol 4 so they load under numpy 1.x and 2.x alike:

| File | Feeds |
| --- | --- |
| `figure3_combined_regenerated_{10,30,50}pct_bootstrap_cache.pkl` | S17's three threshold rows |
| `figure3_combined_regenerated_10pct_min{2,3,5}_rsac_bootstrap_replicates_cache.pkl` | S16's k ≥ 2 rows |
| `k1_combined_10pct_bootstrap_replicates.pkl` | S16's k = 1 row, an exact sample-level bootstrap |

`profiles/` holds the beta-mixture fits, one per *k*, selected by BIC and read
by `fit_repeated`: `profile_mix_k{2,3,5}_best_by_bic_asymptotes.csv`.

S17's ≥30% and ≥50% rows also re-bootstrap their k = 1 support and histogram
columns directly from the combined coverage matrix
(`combined_meancoverage_matrix_5081_SGBs_best.csv`, an ordinary pipeline
input), caching that under `results/Figure3_combined/sinf_cache/`.

## Why the fits are stored rather than recomputed

The fitting chain is heavy and stochastic, so storing its output lets the two
figures redraw exactly and in seconds.

The stored fits encode the sample grid they were fitted on, and
`make_accumulation_sinf_figures.py` checks that grid before drawing: a cohort
change means the fits must be rebuilt, and the script stops rather than fit a
curve on one cohort and plot bootstrap replicates from another.
[`../tools/sinf_chain/README.md`](../tools/sinf_chain/README.md) is the rebuild
procedure.
