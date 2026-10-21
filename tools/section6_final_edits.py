# Section 6 of the supplement as it stood when this page was built -- read
# with tracked insertions applied and equations counted.
#
# READ THIS FIRST IF YOU ARE REGENERATING THIS PAGE. python-docx's
# Paragraph.text concatenates only runs that are DIRECT children of w:p. Every
# run inside a tracked insertion sits under w:ins and is invisible to it, and
# OMML equations are dropped entirely, so a document edited with track-changes
# on reads back as a sequence of blank paragraphs. An earlier version of this
# page was built that way and reported Section 6 as gutted when in fact almost
# all of it had been written. Use tools/read_docx_tracked.py.

RUN = "microbiome_results_intermediate_20261008_1454"

# ── Implemented, verified against the code. Not asked for again. ────────────
DONE = [
 ("6A, overview and naming (671, 673)",
  "Both paragraphs are in. The three confounds are named and addressed in order, "
  "and CMH/MH are separated as test and estimators."),
 ("6A, the 2×2 table (675–676)",
  "Introduced by a sentence and displayed as an equation. This is the table your "
  "advisor asked for, and it is no longer a stray copy — see section 2."),
 ("6B, cohorts, eligible strata, estimand (683, 684, 686)",
  "The five comparisons, the >=3 studies / >=5 per arm rule, ESS 0.842–0.958, and "
  "the Mantel–Haenszel risk difference with Sato variance as the primary estimand, "
  "with Simpson's paradox named."),
 ("6E, the background curve (701–703, 705)",
  "The fitted expectation, the bandwidth, the local scale, z, the SE floor, the "
  "support rule, sibling exclusion, the high-prevalence caveat and the refitting "
  "bootstrap are all in and all match the code."),
 ("6E/6F, the demotion (706, 709)",
  "Mean z is named the primary list-level test and the HMP is applied to its "
  "bootstrap p-values — which is exactly what the pipeline now computes."),
 ("6F, driver taxa (712)",
  "Ranked on z against the curve, uncorrected on the figure and BH in the "
  "companion panel."),
 ("6G.3, the risk difference (745–754)",
  "A new subsection with rd_MH, the Sato variance and both P and Q terms, the "
  "sparse-data consistency argument and the single-stratum limit."),
 ("6G.5 Step 2, the background curve (769)",
  "The density-ratio step is replaced by the kernel regression, with the grid, "
  "the kernel weight and the support rule."),
 ("6G.6, the primary test (779–780)",
  "Mean z listed first of four, with 2000 replicates, the 1/2000 floor and the "
  "pointer to 6G.9."),
 ("6G.10, within-clade contrasts (808, 810)",
  "A new subsection, with Fig. S31 cited — the figure previously had neither a "
  "description nor a citation anywhere."),
 ("Subsection numbering and headings",
  "6G.1–6G.10 are numbered correctly, and 6G.9 now reads “Threshold combination "
  "across detection specifications”."),
 ("Typos",
  "artifact, Control-prevalence, geometric-mean-target, energy distance, "
  "“a shift in center”, CRC / Adenoma and the Section 4.x cross-references."),
]

# ── The one unfilled placeholder ────────────────────────────────────────────
PLACEHOLDER = ("686",
 "disagree in sign for [[n]]% of taxon-by-condition comparisons",
 "12.7",
 "3,111 of 24,536 taxon-by-condition rows, comparing rd_mh with pooled_diff in "
 "health_per_taxon_stratified_10pct.csv. 207 of them differ by more than five "
 "percentage points. Under the 300 kB rule it is 12.3%. If you want the sentence "
 "to carry the weight: “… disagree in sign for 12.7% of taxon-by-condition "
 "comparisons (3,111 of 24,536), and for 207 of those the pooled difference "
 "exceeds five percentage points.”")

# ── Where the text and the code still disagree ──────────────────────────────
# (anchor still in the document, para, what is wrong, what to do)
CONFLICTS = [

("where k is the number of detection specifications combined", "803",
 "6G.9 now contradicts itself. 801 says the HMP is applied to the mean-z "
 "analysis; 803, directly under the equation, still says p_i is the permutation "
 "p-value of the energy-distance test and that the mean energy distance is the "
 "companion effect-size summary. The pipeline combines the mean-z bootstrap "
 "p-values. “Valid under positive dependence” also understates the result: the "
 "HMP is valid under arbitrary dependence, which is the reason it is the right "
 "choice here.",
 """where k is the number of detection specifications combined (here k = 2: the 10% genome-relative breadth threshold and the 300,000 mapped-bases flat threshold), p_i is the bootstrap p-value of the mean z under specification i for a given condition and focal list, and p_HMP is the resulting threshold-robust combined p-value. The two components are weighted equally, so the general weighted form of the harmonic mean p-value reduces to the expression above. The HMP is used because it remains valid under arbitrary dependence between the component tests (Wilson 2019), which is what two detection rules applied to the same taxa in the same samples require. The combined value is written to health_background_curve_meta.csv alongside each specification's own mean z and interval, and it is the value quoted in the main text. Because the bootstrap floors each p_i at 1/2,000, the HMP inherits that floor. The same combination is applied to the energy-distance permutation p-values as a descriptive check, and the mean of the mean-z values across specifications is reported as a companion effect-size summary.

Agreement between the two specifications is tabulated separately rather than folded into the combined p-value, since the HMP measures the strength of the combined evidence and not the concordance of its parts: across the 54 condition-by-list cells the two rules agree in the sign of the mean z for 50, and all four disagreements have |mean z| below 0.12 and are non-significant under either rule. The figures annotate one specification and not the combination — Figure 5E and Fig. S26 print the mean z, its interval and its bootstrap p under the primary rule, breadth ≥ 10% — so that the star in each panel matches the interval drawn beside it. In this run the primary rule and the combined value agree on significance at the 5% level for every condition-by-list cell."""),

("the distribution being compared is the distribution of", "705",
 "The sentence says the compared distribution is the distribution of prev_diff "
 "values. It is not: every list-level comparison runs on the study-stratified "
 "shift derived from rd_MH, and prev_diff is the pooled quantity that 686 has "
 "just finished saying is used for no inference. As written, 6E contradicts 6B.",
 """In the present application, the distribution being compared is the distribution of study-stratified case–control shifts, so the energy distance analysis asks whether focal-list taxa show a collectively unusual pattern of depletion or enrichment relative to equally prevalent taxa outside the list."""),

("an importance-weighted directional permutation test asks whether the mean", "706",
 "Same substitution: the directional permutation test is described as acting on "
 "the mean prev_diff of the focal list.",
 """First, an importance-weighted directional permutation test asks whether the mean stratified shift of the focal list is more extreme than expected under repeated draws from the matched background."""),

("Pass 1: Cross-condition consistency z-scores", "791",
 "6G.7 still describes Pass 1 as comparing each taxon's prev_diff to the "
 "importance-weighted background. Paragraph 712 already says drivers are ranked "
 "on z against the fitted curve, which is what the pipeline does, so the "
 "appendix and the body disagree.",
 """Pass 1: Cross-condition consistency. Within each disease condition, each focal-list taxon is scored by its own distance from the fitted background curve, z, as defined in 6E. These condition-level scores are then aggregated across diseases to summarise (i) directional consistency, defined as the fraction of conditions in which the taxon shifts in its dominant direction, and (ii) average magnitude across conditions. Per-taxon significance is reported both uncorrected and after Benjamini–Hochberg correction within each list."""),

("This null draw is repeated", "786",
 "The energy-distance null is stated as 5,000–10,000 draws. The code uses 5,000 "
 "exactly (_iw_energy_test, n_perm=5000); the directional test's 2,000 at 782 is "
 "correct.",
 """This null draw is repeated 5,000 times. The empirical p-value is the fraction of null energy distances that are at least as large as the observed statistic, floored at 1/(n+1)."""),

("we construct a prevalence-matched background of non-list taxa through importance weighting", "764–765",
 "The 6G.5 heading and its opening sentence still frame the whole subsection as "
 "importance weighting, but Step 2 is now a kernel regression and importance "
 "weights survive only in the secondary tests and in the Table S10B direction "
 "columns. Retitle the subsection and rewrite the lead.",
 """6G.5. Background construction

For each disease condition and focal taxon list, we reference the list to taxa of the same baseline prevalence. Step 1 fixes the matching coordinate. Step 2 fits the expectation used by the primary statistic. Importance weighting survives only in the secondary tests of 6G.6, where a matched reference set rather than a fitted curve is required."""),
]

# ── Procedures the document still describes that the pipeline does not run ──
DEL = [
 ("711", "6F, “Where retained, we additionally place the observed list-level statistics within explicit null distributions…”",
  "No such procedure is computed. The paragraph already hedges with “Where retained”"),
 ("777", "6G.5 Step 3, largest-remainder replication of weights into integer rows",
  "Nothing now requires replicated background rows; with Step 2 replaced by the curve there is nothing for it to serve"),
 ("789", "6G.7, the explicit null-distribution check over 500 importance-weighted draws",
  "Not computed. Same reason as 711"),
 ("792–795", "6G.7 Pass 2, the leave-one-out energy distance δE, its equation and its closing summary",
  "Not computed in the current pipeline. With it goes the “two passes” framing of 790, which becomes a single pass"),
]

# ── Figure cross-references ─────────────────────────────────────────────────
XREF = [
 ("680", "Figs. S24–S31", "Figs. S23–S31",
  "683 cites S23 for the raw depth difference, so the range that opens the section should include it"),
 ("707", "driver analyses are shown in Figs. S28–S31", "Figs. S27–S30",
  "S27 and S28 are the VANISH_H+ and BloSSUM_H+ driver figures, S29 and S30 the VANISH and BloSSUM ones. S31 is the within-clade figure, correctly cited at 810"),
 ("712", "Figs. S28–31", "Figs. S27–S30", "the same slip"),
 ("703", "Figure S25", "correct, no change",
  "S25 carries the prevalence-coverage row, which is the quality control for this step"),
 ("707", "Fig. S26", "correct, no change",
  "S26 is the background curve for the base VANISH and BloSSUM lists, which is what robustness across list families means here"),
]

# ── Small textual ───────────────────────────────────────────────────────────
FR = [
 ("with the variance of Sato (90) We write it",
  "with the variance of Sato (90). We write it", "686, missing full stop"),
 ("(87, 88)(88)(89, 90)", "delete",
  "673, a row of stray citation markers left over beside the inserted paragraph, which now carries those citations inline"),
 ("(90)(91)", "delete", "686, the same leftover"),
]

# ── Numbers the text could carry, with the run value ────────────────────────
NUMBERS = [
 ("Comparisons clearing the rule", "5", "5", "n_studies >= 3 in the CMH table", "match"),
 ("Smallest / largest cohort", "979 (IBD) / 9,238 (unhealthy)", "253+726 / 4,094+5,144", "Figure 5D cohort labels", "match"),
 ("GMW ESS fraction range", "0.842–0.958", "IBD 0.842, Obese 0.958", "mean_ess_frac, five plotted conditions", "match"),
 ("Detection specifications", "10% breadth, 300,000 bp", "10.0, 300000", "config", "match"),
 ("List sizes", "124 / 63 / 100 each", "124 / 63 / 100 each", "curve taxa table", "match"),
 ("Bootstrap replicates", "2000, floor 1/2000", "2000, seed 17", "config.HEALTH_BG_N_BOOT", "match"),
 ("Directional permutations", "2,000", "2000", "_iw_directional_test", "match"),
 ("Kernel bandwidth, grid", "h = 0.12, 400 points over [1e-4, 1]", "same", "config.HEALTH_BG_BANDWIDTH", "match"),
 ("Support rule", ">=15 within 0.25 log10", "same", "config.HEALTH_BG_MIN_SUPPORT", "match"),
 ("High-prevalence limit", "85%", "0.85", "config.HEALTH_BG_HIGHPREV_LIMIT", "match"),
 ("Clade null", "5,000 placebo clades, min 5 taxa, >=3 and >=3", "same", "config.HEALTH_BG_CLADE_NPERM", "match"),
 ("Energy-distance null draws", "5,000–10,000", "5000", "_iw_energy_test", "FIX — see 786"),
 ("Simpson's disagreement", "[[n]]%", "12.7% (3,111 of 24,536)", "rd_mh vs pooled_diff", "FILL"),
 ("Kernel reach", "not stated", "509 effective background taxa at 1% prevalence, 60 at 50%", "Kish effective n of the kernel weights", "could add"),
 ("Sibling overlap", "“overlap substantially”", "98/100 VANISH_H+, 83/100 BloSSUM_H+", "list membership", "could add"),
 ("High-prevalence taxa", "not counted", "4 in BloSSUM+ and BloSSUM_H+, 3 in BloSSUM, 0 in every VANISH list (max 47%)", "control prevalence >= 0.85", "could add"),
 ("High-prevalence sensitivity", "not stated", "unhealthy BloSSUM_H+ +1.44 -> +1.24, IBD +1.39 -> +1.23, no sign change anywhere", "mean z excluding prev >= 0.85", "could add"),
 ("Taxa failing the support rule", "“plotted but excluded”", "none: 0 of 5,283 list-taxon scores", "curve taxa table", "could add"),
 ("Cross-spec sign agreement", "not stated", "50 of 54 cells; all 4 disagreements |z| <= 0.12 and n.s.", "health_background_curve_meta.csv", "could add"),
]

# ── Optional additions, as paste-ready sentences ────────────────────────────
ADDITIONS = [
 ("703, after “overlap substantially”",
  "the overlap is the reason for the sibling exclusion, and it is lopsided",
  """Because the six focal lists overlap heavily — 98 of the 100 VANISH_H+ taxa and 83 of the 100 BloSSUM_H+ taxa also belong to at least one other list — fitting the curve on all non-list taxa would fit it partly on near-duplicates of the list being scored, pulling the expectation toward that list and shrinking z. The overlap is not symmetric between the two families, so the attenuation differed between them. The background for every list is therefore the set of taxa belonging to none of the six, which makes the comparison identical in construction for each."""),
 ("703, after the support-rule sentence",
  "how far the kernel reaches, and that the rule rejected nothing here",
  """The kernel draws on a large but prevalence-local pool, with an effective 509 background taxa near 1% control prevalence falling to 60 near 50%, where the catalogue thins. In the present run no taxon in any list failed the support rule (0 of 5,283 list-taxon scores)."""),
 ("703, after the high-prevalence sentence",
  "the high-prevalence caveat is currently asserted without its size",
  """Four taxa in each of BloSSUM+ and BloSSUM_H+, and three in BloSSUM, lie in this range; the VANISH-family lists contain none, their most prevalent member reaching 47%. Excluding them would move the pooled unhealthy BloSSUM_H+ mean z from +1.44 to +1.24 and the IBD value from +1.39 to +1.23, with no change of sign in any condition or list."""),
]

DETERMINISM = """Yes, and it is worth stating how far that holds.

Every random procedure in the health path is constructed as np.random.default_rng() with an explicit integer seed: the background bootstrap (17), the clade permutations (41), the energy-distance and directional nulls (42), the label-shuffle permutations (42) and the figure's own sampler (42). There are no unseeded draws in health_background.py, health_cmh.py, section3_health.py or figure5.py.

Seeds alone are not sufficient, because Python randomises string hashing per process, so a set iterated in a different order can change a result. That was a real defect here — an abundance routine keyed a sampler on abs(hash(name)) and was changed to a CRC. To check the rest, the background curve was recomputed under three different PYTHONHASHSEED values. The mean z, the bootstrap standard error and both confidence limits were bit-identical to full double precision across all three, for both focal lists.

So a final run should reproduce the present numbers exactly, given the same inputs, the same seeds and the same library versions. Two caveats. The bootstrap replicate count was raised from 200 to 2,000 on 8 October, so p-values from runs before that change will not reproduce; at 200 the floor was 0.005 and the IBD verdict moved between seeds, which is why it was raised. Point estimates are unaffected. And the final run will contain one table the present results do not, health_background_curve_meta.csv, carrying the combined statistic."""

PIPELINE_TODO = """Done, in the way you asked for. The tables keep every contrast and gain an `eligible` column computed from the same rule the figures use, so the five-row supplement tables can be produced by sorting rather than by hand-filtering a nine-row CSV after every run.

Nothing leaks into a figure. Every figure data object in the 8 October run was scanned and the rendered Figure 5, S25 and S26 re-read: all three show exactly the five eligible contrasts. The five health figure paths each filtered independently and correctly, but nothing recorded the result, so each figure now writes the conditions it actually drew into its data object. tools/check_run_conditions.py checks that after a run, re-deriving the eligible set from that run's own CMH table.

One thing to run tonight, after the pipeline finishes: python tools/check_run_conditions.py <results-root>. Its exit code is the number of problems, so it can gate."""

TABLE_NOTE = """Your advisor's request and the note that said to delete paragraph 670 were never in conflict, and the document has already resolved it — better than my note proposed.

The objection was not to the table. It was that a bare copy of the 6G.1 display sat in the overview with no sentence introducing it and no definition of its cells, while the identical display appeared properly in 6G.1 with both. In the current draft the overview carries an introducing sentence at 675, “The key object in the CMH test is the per-study 2×2 table of taxon detection versus absence by control versus case status”, and the table follows it at 676. That is the table your advisor asked for, and it is now introduced. Nothing further is needed.

Two things worth tightening while it is open. First, 6G.1 at 716–718 displays the same table a second time; now that the overview introduces it, 6G.1 can refer back to it instead — give the overview copy a number, Table 6.1 or equation (6.1), and have 6G.1 cite that number. Second, the document carries two notations for the same object: 6G.1 names the cells control-present, control-absent, case-present and case-absent, while the new 6G.3 writes the risk difference in weighted arm sizes and detection counts. Fix one set of symbols on the overview table and use it in both places, so a reader moving between 6G.1 and 6G.3 does not have to re-derive which cell is which."""

# ── New supplementary table: every list-level p-value side by side ──────────
NEWTABLE_WHY = """Four list-level tests are computed for every condition-by-list cell, and until now three of them appeared nowhere a reader could check: the ESS-adjusted IW-Fisher and the IW-directional permutation p-values set the F and U stars on Fig. S25 and were never written to a file, and the energy distance was tabulated only in its own Table S6B. The threshold-robustness claim in 6F and 6G.9 therefore rested on numbers no reader could see.

table_s12_method_comparison_<ref>.csv now puts all of them in one frame, one row per condition and list: each detection rule's mean z with its bootstrap interval and p-value, its energy distance and permutation p, the two importance-weighted tests, and the harmonic mean across rules for each. Every contrast is kept, flagged by `eligible`, because a reader comparing methods most wants the contrasts that were excluded for want of studies rather than for want of signal.

The result is a table that makes the paper's two central methodological claims auditable rather than asserted: that the two detection rules agree, and that demoting the energy distance to a descriptive statistic does not manufacture the result."""

NEWTABLE_CAPTION = """Table S12. List-level tests compared across methods and detection specifications. One row per disease condition and taxon list. For each of the two detection specifications — a 10% genome-relative breadth threshold and a flat 300,000 mapped-bases threshold — the table reports the mean distance from the fitted prevalence background (mean z) with its bootstrap 95% interval and p-value, the importance-weighted energy distance with its permutation p-value, the ESS-adjusted IW-Fisher p-value, and the IW-directional permutation p-value. For each test, the p_*_hmp column is the harmonic mean p-value across the two specifications (Wilson 2019), which is the threshold-robust value quoted in the main text. mean_z_combined is the mean of the two mean-z values and sign_agree records whether they share a sign. n_methods_sig counts how many of the four tests reach p < 0.05 on their combined value, out of the n_methods_tested that could be computed; the energy distance is not computable where the matched background is too small, which is why some rows test fewer. The eligible column marks the contrasts shown in Figure 5 and the supplementary figures, namely those resting on at least three studies with at least five cases and five controls. Contrasts failing that rule are retained here rather than dropped. The two importance-weighted tests are computed only for eligible contrasts, since they are evaluated in the procedure that draws Fig. S25."""

NEWTABLE_PARA = ("6F, after the threshold-combination paragraph (709), or as the "
 "closing paragraph of 6G.9",
 """Table S12 reports every list-level test side by side, for both detection specifications and for their harmonic mean, so that the threshold-robustness claim and the demotion of the energy distance can both be checked directly. Across the 30 eligible condition-by-list cells the primary mean z and the descriptive energy distance agree on significance at the 5% level in 26. Of the four disagreements, three are cells in which the mean z is significant and the energy distance is not, which is the expected direction: the energy distance spends power on differences in spread and in the tails that a shift in centre does not produce. The fourth, obesity against BloSSUM+, is the reverse, and is the case the energy distance exists to catch — a list that differs from its background in spread rather than in centre.""")

NEWTABLE_NUMBERS = [
 ("Cells in the table", "54 (9 conditions x 6 lists)"),
 ("Eligible cells", "30"),
 ("Significant by the combined mean z", "19 of 30 eligible, 34 of 54 overall"),
 ("Mean z and energy distance agree", "26 of 30 eligible"),
 ("Every tested method agrees it is significant", "16 of 30 eligible"),
 ("No method significant", "10 of 30 eligible"),
 ("Sign agreement across detection rules", "29 of 30 eligible, 50 of 54 overall"),
]
