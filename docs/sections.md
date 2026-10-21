# Pipeline sections: what each one computes

The analysis runs in three sections, plus a sensitivity sweep and figure
assembly. This document gives the statistics in enough detail to reimplement
them. The README covers why the pipeline is structured this way, and
`docs/code_map.md` says which module produces which published item.

Section numbering matches the manuscript's Supplementary Text.

## Common machinery

### Detection

A taxon is present in a sample when it passes one of two thresholds:

| Mode | Rule | Constant |
| --- | --- | --- |
| breadth (main text) | ≥ 10% of the genome covered | `BREADTH_THRESHOLD = 10.0` |
| flat | ≥ 300 kb of mapped bases | `ACCUM_MAIN_TEXT_FLAT_BP = 300_000` |

`--detection-mode both` runs them side by side. Every headline result is
reported under both, because a detection rule is a modelling choice, and a
result that survives only one of them is a result about the rule.

Sequencing effort is measured in **mapped bases**, never reads: read length
varies between studies, so read counts are not comparable across them.

### Geometric-mean-target weighting (GMW)

Depth drives richness — Spearman ρ = 0.682 against mapped bases — and depth
differs systematically between lifestyle groups, so a raw comparison confounds
biology with sequencing effort. Rather than rarefy, which discards data and is
ill-suited to the rare tail this study is about, samples are reweighted toward
a shared target depth distribution.

For groups *A* and *B* with kernel-density depth distributions `f_A` and `f_B`
over log₁₀(mapped bases), the target is their normalised geometric mean:

```
f*(d)  ∝  sqrt( f_A(d) · f_B(d) )                 two groups
f*(d)  ∝  ( f_HG(d) · f_T(d) · f_I(d) )^(1/3)     three groups
```

The geometric mean is not arbitrary: it is the distribution equidistant from
the groups in Kullback–Leibler divergence, so neither group's empirical depth
distribution is privileged.

A sample *i* in group *A* at depth *d_i* gets weight

```
w_i = f*(d_i) / f_A(d_i)
```

mean-normalised within its group so mean(*w*) = 1, then hard-clipped at
`WEIGHT_CLIP = 10.0` so that no single sample dominates.

Densities come from a binned KDE (`depth_bias_ipw._BinnedKDE`): 512 histogram
bins, Silverman-rule bandwidth, Gaussian convolution by FFT. A naive KDE over
groups of 3,000–14,000 samples is prohibitively slow.

The corrected group estimate is the weighted mean; the quantity of interest is
the difference between two of them.

**Bootstrap inference.** Replicates resample samples, and the KDEs are held
fixed. That is deliberate: the weights define the estimand — both groups
standardised to the same reference depth distribution — rather than estimating
a nuisance parameter. Refitting them per replicate would move the reference, so
each replicate would answer a slightly different question. The replicate count
scales with the smaller group.

**Effective sample size.** The Kish ESS, `(Σw)² / Σw²`. A low ESS fraction
means the correction is expensive: weight dispersion has bought depth balance
at the cost of precision.

**Did the correction matter?** Per group, the raw unweighted mean richness is
compared against the bootstrap distribution of the corrected mean as a z-score.
If the depth distributions were already balanced, the weights sit near 1 and
the two agree. A significant result — the bracket on each violin in Figs. S6,
S11–S12 and S24 — means the correction moved the estimate, which is the
evidence that it was doing work.

---

## Section 1 — richness and composition (`s1`)

**Richness.** Depth-corrected per-sample richness is compared across lifestyles
by bootstrap permutation under GMW. Because that bootstrap resamples *samples*,
and 3,451 samples are repeats of a host already in the cohort, the comparison
is repeated two further ways: resampling *hosts* with their samples kept
together, and retaining one sample per host. All three are reported.

**Composition.** Jaccard (presence/absence) and Bray–Curtis (abundance)
dissimilarities, computed on 1:1 depth-matched sample pairs rather than on the
raw cohort.

**Intermediacy.** To test whether agro-pastoralist microbiomes sit
compositionally between hunter-gatherer and industrial ones, samples are binned
and projected onto the hunter-gatherer-to-industrial axis. `s1_axis_angle`
computes the projection, `s1_axis_stats` its statistics.

## Section 2 — accumulation, and the taxon lists (`s2`, `accum`)

### Finite-catalogue accumulation

Population-level diversity is posed as a **discovery problem**: how many
catalogued SGBs are detected as samples or studies are pooled, and what the
asymptote is. The asymptote is the inferential target — it is the quantity that
widespread extinction would have to move.

Samples or studies are pooled incrementally across bootstrap replicates, and
two kinds of curve are fitted:

- **First detection (k = 1).** Three-parameter Hill-type Michaelis–Menten and
  four-parameter Richards sigmoid. Uncertainty in the asymptote comes from
  likelihood profiling over the bootstrap replicates.
- **Repeated detection (k ≥ 2).** A first-detection model is wrong here:
  requiring detection in several samples makes the per-taxon detection
  probability heterogeneous in a way one rate cannot absorb. Beta-binomial and
  beta-mixture models are used instead, with mixture order chosen by BIC.

`tools/sinf_chain/` rebuilds those mixture fits; `accumulation_sinf_inputs/`
holds the cached result.

### VANISH and BloSSUM

Six list families across two reference ID spaces:

| List | Definition | n |
| --- | --- | ---: |
| VANISH | Published Carter et al. list, unchanged | 124 |
| BloSSUM | Published Carter et al. list, unchanged | 63 |
| VANISH+ | Most non-industrial-enriched by GMW prevalence difference | 100 |
| BloSSUM+ | Most industrial-enriched, same ranking | 100 |
| VANISH_H+ | As VANISH+, on healthy-industrial vs non-industrial | 100 |
| BloSSUM_H+ | As BloSSUM+, on the same contrast | 100 |

For taxon *t* in group *g*, weighted prevalence is the weight-share of samples
in which *t* is detected:

```
prev(t,g) = Σ_{i∈g} w_i · 1[breadth(t,i) ≥ θ]  /  Σ_{i∈g} w_i
Δ(t)      = prev(t, I) − prev(t, NI)
```

The 100 most negative Δ are VANISH+, the 100 most positive BloSSUM+.
Accompanying p-values come from weighted 2×2 tables and are used for the
volcano plots and descriptive calibration — **not** as the selection criterion,
which is rank on Δ alone.

These lists are data-dependent, so membership moves with the cohort. Rebuilt
one-sample-per-host they retain 97/100 VANISH+ and 94/100 BloSSUM+, with
Spearman ρ = 0.994–0.996 on the full 5,081-SGB ranking.
`metadata/lists_as_analysed/` is the committed copy of those lists.

## Section 3 — health associations (`s3`)

Tested **within industrial cohorts only**: case-control studies of
non-industrial populations are too few. Four conditions carry at least three
independent studies — obesity, glucose intolerance, IBD and colorectal cancer —
alongside a pooled `Unhealthy` group spanning 26 recorded conditions.

### GMW-CMH, the taxon-level model

Pooling cases and controls across studies would confound health with study
membership, since studies differ in population, sequencing depth and extraction
protocol. The test is therefore **study-stratified Cochran–Mantel–Haenszel**,
extended to equalise depth within each stratum.

Each eligible study contributes a 2×2 table of taxon detected/absent by
control/case. A study is eligible with at least `CMH_MIN_STRATUM = 5` cases
**and** 5 controls; a contrast is reported when at least
`MIN_STRATA_TO_PLOT = 3` strata qualify.

Within each stratum, GMW weights are computed as above but on the case/control
split, and the integer cells are replaced by weighted effective counts:

```
a_s = Σ_{i∈ctrl_s} w_i · 1[detected]    b_s = Σ_{i∈ctrl_s} w_i · 1[absent]
c_s = Σ_{i∈case_s} w_i · 1[detected]    d_s = Σ_{i∈case_s} w_i · 1[absent]
```

Those weighted tables pass unchanged into the standard CMH aggregator, with
Yates' continuity correction, giving a χ²₁ statistic and a two-sided p-value. A
stratum with fewer than five samples per arm, or fewer than three distinct
depth values in an arm, falls back to all-weights-1 — an unweighted CMH
contribution rather than an excluded stratum.

Two effect measures are reported per taxon:

- the **Mantel–Haenszel common log-odds ratio**, and
- the **MH common risk difference** on the prevalence scale, with the
  **Sato (1989)** variance estimator, which stays consistent in the sparse-data
  limit — the relevant regime, since many strata contribute only a few samples
  per arm.

Sign convention: **positive means more prevalent in controls**, hence depleted
in cases.

### The prevalence-matched background

Counting significant taxa per list is incomplete. It discards subthreshold
direction, and it ignores that baseline prevalence bounds how far a taxon can
move at all: one present in 95% of controls cannot gain much, one in 2% cannot
lose much.

So each list is compared not against the raw background pool but against a
**fitted expectation for taxa of the same prevalence**:

1. For every non-list taxon, compute a study-aware control-prevalence
   coordinate — prevalence within each eligible stratum, collapsed by weighting
   each stratum by its control count.
2. Regress the study-stratified case-control prevalence difference on that
   coordinate across all background taxa, and estimate the local spread.
3. Score each list taxon by its distance from the fitted curve, in units of
   that local spread: **z**.

Two constraints. The local spread is floored at the taxon's own Sato standard
error, so a distance is never expressed in units smaller than the taxon's own
sampling noise. And a taxon detected in no sample of either arm is excluded: a
zero shift there reflects absence of data, not absence of effect.

**The list-level statistic is the mean z** over supported list taxa, with a 95%
interval from a bootstrap that resamples the background taxa and *refits the
curve* on each resample. That refit is the expensive part, and the reason a
full run takes two days (`HEALTH_BG_N_BOOT = 2000`).

An energy distance between the list and an importance-weighted reference
background is reported alongside it, as a descriptive shape statistic sensitive
to differences in spread and tails that a mean is not. It accompanies the mean
z rather than testing it.

### Multiplicity

FDR is controlled **per list** by Benjamini–Hochberg: `section3_health` groups
by `Taxon_list` and applies `health_background._bh` within each group. So one
taxon's q-value differs between the lists it belongs to, and the
best-across-lists q is not the within-list q — a distinction worth holding onto
when reading a per-taxon table.

## Sensitivity sweep (`sensitivity`)

Re-derives VANISH and BloSSUM from prevalence under alternative criteria, then
reruns Sections 2 and 3 for each definition, so that no conclusion rests on one
definition of the lists.

| Set | Criteria |
| --- | --- |
| `core` | Default compact grid |
| `tight` | Stricter one-sided rarity |
| `loose` | More permissive |
| `full` | Full grid over rare cutoff, min-present cutoff and study-aware prevalence |

```bash
python -m persistence --ref combined --only sensitivity --sensitivity-defs full
```

Two derivations are available (`persistence/vanish_blossum_definitions.py`):
band-pass prevalence with hard thresholds — `p_I ≤ r` and `m ≤ p_NI ≤ u` for
VANISH, reversed for BloSSUM — and a rarity-weighted log-odds gap scored
continuously,

```
S = |logit(p_I) − logit(p_NI)| · sqrt( p̄ (1 − p̄) )
```

still subject to the one-sided rarity constraint, then ranked by *S*. Both can
aggregate prevalence study-aware, so that one large study cannot define a list.

## All section keys

| Key | What it runs |
| --- | --- |
| `s1` | Diversity, richness and accumulation |
| `s1_alpha` | Alpha-diversity subset |
| `s1_traces` | Rarefaction traces |
| `s1_within` | Within-group variation |
| `s1_tech` | Technical-replicate robustness |
| `s1_pcoa3` | Three-way GMW-weighted PCoA |
| `s1_axis_angle` | Projection onto the non-industrial-to-industrial axis |
| `s1_axis_stats` | Statistics for those projections |
| `s1_depthbin` | Depth-binned richness |
| `s1_figure` | Section 1 figure assembly |
| `s2` | Differential prevalence |
| `s3` | Health associations |
| `s3_axis` | Health conditions projected onto the lifestyle axis |
| `s3demo` | Demographic associations |
| `accum` | Accumulation curves |
| `carter` | Carter-cohort analyses |
| `depth_bias` | Depth-bias diagnostics |
| `gmw_richness` | GMW richness panels |
| `health_richness` | Health richness panels |
| `prevmatch` | Prevalence-matched background |
| `host_sens` | One-sample-per-host sensitivity |
| `figure3` | Figure 3 assembly |
| `sensitivity` | VANISH/BloSSUM definition sweep |
| `table_s1a` | Per-study metadata summary |
| `figures` | Figure and table assembly |
| `visualization` | Standalone visualisations |
| `graphical-abstract` | Graphical abstract |

`python -m persistence --help` lists them all with their options.

## Output layout

```
results/<ref>/
  cohort_samples.csv            The samples that survived, for this reference
  section1_diversity/{main,supplement}/
  section2_differential/{main,supplement}/
  section3_health/{main,supplement}/
  depth_audit/  depth_bias_ipw/  carter_analysis/
  sensitivity_sweep/
  figure_data/Figure_S<n>_data.pkl   One per supplementary figure
  stats_report.csv              Every reported number, with its tag
  .pipeline_state/              --resume checkpoints
main_text_figures/              Figures 1-5
```

Every section records its reported numbers through `StatsReport`
(`stats.add(tag, section, placement, description, value)`), so
`stats_report.csv` is the trail from a manuscript number back to the code that
produced it.
