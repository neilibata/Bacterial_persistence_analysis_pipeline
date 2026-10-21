# `tools/sinf_chain/` — rebuilding the S∞ fits behind Figures S16 and S17

The three stages that produce the fits in
[`../../accumulation_sinf_inputs/`](../../accumulation_sinf_inputs/), which
Figures S16 and S17 are drawn from. Run these to rebuild those fits; to redraw
the figures from fits that already exist, run
`tools/make_accumulation_sinf_figures.py` on its own.

The models are **Supplementary Text 4** of the manuscript: §4E–4F define the
beta-binomial and beta-mixture forms, §4H the bootstrap and the asymptote
intervals these stages compute.

## The chain

1. **Bootstrap the accumulation curves.**
   `figures/make_combined_figure3.py`, at breadth thresholds 10 / 30 / 50% and
   at `--min-occurrences {2,3,5}`.

2. **Select the mixture order.**
   `score_mixture_profiles_for_k.py --k {2,3,5}` — profile-likelihood scans
   over mixture orders, BIC selection, writing
   `profile_mix_k{k}_best_by_bic_asymptotes.csv`. `fit_repeated` reads the
   selected order *H* and the fitted shape parameters directly out of that
   file, so without this stage there are no k ≥ 2 rows to draw.

3. **Draw.** `tools/make_accumulation_sinf_figures.py`.

`profile_sinf_mixture.py` takes its work path as an argument and defaults to
the directory it was written in, so pass the repository path explicitly when
running it here.

## When a rebuild is necessary

The caches encode the sample grid they were fitted on. Stage 3 re-bootstraps
the k = 1 rows for the 30% and 50% thresholds from the live coverage matrix and
**refuses to run when that grid does not match what the caches hold** — it will
not fit a curve on one cohort and plot bootstrap replicates from another.

So a cohort change means rebuilding from stage 1. The cached fits currently in
the repository were produced on the cohort the manuscript reports; if a run
changes the sample set, expect stage 3 to stop rather than to draw something
inconsistent.

## Files

| File | Stage | What it does |
| --- | --- | --- |
| `score_mixture_profiles_for_k.py` | 2 | Profile-likelihood scan and BIC selection per *k* |
| `profile_sinf_mixture.py` | 2 | The mixture fit that the scan calls |
| `combined_summary_two_figures.py` | — | The original combined analysis the chain was derived from, kept for reference |
