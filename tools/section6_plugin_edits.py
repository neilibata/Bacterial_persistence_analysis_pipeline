# Plug-in edits to Supplementary Section 6. Minimal diffs against the shipped
# text: keep the existing wording and mathematics wherever they are still
# correct, change only what the corrected pipeline makes wrong.

# --- A. find / replace, applied in Word -------------------------------------
FR = [
 ("artfact", "artefact", "699"),
 ("at least three studies obesity", "at least three studies (obesity",
  "36 -- Methods front matter; the opening bracket is missing, so the list of "
  "four conditions runs on from the clause that introduces it"),
 ("geometric-meantarget", "geometric-mean target", "685"),
 ("Controlprevalence", "Control prevalence", "693"),
 ("differenceprev_diff", "difference prev_diff", "686"),
 ("prevalence difference,prev_diff", "prevalence difference, prev_diff", "679"),
 ("energydistance", "energy distance", "695, 699, 702, 759, 768, 777 and the 6E/6F headings"),
 ("a shift in cent,", "a shift in centre,", "695, 761"),
 ("CRC / denoma", "CRC / adenoma", "681"),
 ("renormalied", "renormalised", "753"),
 ("given in Section 4.6", "given in Section 6G.2", "685"),
 ("the list-level summaries in Section 4.3", "the list-level summaries in Section 6D", "686"),
 ("the prevalence-matched background analyses in Section 4.4",
  "the background-curve analyses in Section 6E", "686"),
 ("robustness analyses in Section 4.5", "robustness analyses in Section 6F", "686"),
 ("6.8 Threshold combination across detection specifications",
  "6G.8 Threshold combination across detection specifications", "775 heading"),
 ("6E Prevalence-matched background and energy-distance test",
  "6E Prevalence background and the list-level statistic", "691 heading"),
 ("6G.4 Importance-weighted background construction",
  "6G.4 Background construction", "742 heading"),
 ("6G.1 Standard CMH and Mantel–Haenszel common log-odds ratio",
  "6G.1 The CMH test and the Mantel–Haenszel common odds ratio", "705 heading"),
]

# --- B. paragraph revisions: (para, note, original, revised) -----------------
# `original` is quoted from the shipped document so the diff is exact.
REV = [
(669, "complete the broken sentence and absorb the orphan fragment at 671",
 "6 describes the within-industrial health analyses underlying Figure 5. These comparisons are vulnerable to three confounds: sequencing-depth imbalance between case and control samples, heterogeneity across studies, and baseline-prevalence effects that constrain how much a taxon can change between health groups. To address these simultaneously, we",
 "6 describes the within-industrial health analyses underlying Figure 5. These comparisons are vulnerable to three confounds: sequencing-depth imbalance between case and control samples, heterogeneity across studies, and baseline-prevalence effects that constrain how much a taxon can change between health groups. To address these simultaneously, we equalise depth within each study stratum by geometric-mean weighting, combine strata by the Mantel–Haenszel construction, and reference each taxon to a fitted expectation for taxa of the same baseline prevalence. We refer to the weighted stratified construction as the geometric-mean-weighted Cochran–Mantel–Haenszel test (GMW-CMH)."),

(679, "the estimand is the risk difference, not the log-odds ratio",
 "The primary taxon-level estimand in the health analyses is the common within-study log-odds ratio of taxon detection in controls versus cases after reweighting both groups to a shared geometric-mean target depth distribution within each study stratum. For interpretation, we also report the weighted prevalence difference,prev_diff = prev_control − prev_case, so that positive values denote taxa that are more prevalent in controls and therefore depleted in disease.",
 "The primary taxon-level estimand in the health analyses is the Mantel–Haenszel common risk difference in taxon detection between controls and cases, after reweighting both groups to a shared geometric-mean target depth distribution within each study stratum, with the variance of Sato (1989). We write it rd_MH = prev_control − prev_case, so that positive values denote taxa that are more prevalent in controls and therefore depleted in disease. The Mantel–Haenszel common log-odds ratio is retained and reported alongside it. A second difference is also stored for each taxon, obtained by pooling every control across studies, pooling every case, and then subtracting; this is the quantity earlier drafts reported as prev_diff. It is not a within-study contrast and is used for no inference. Because studies differ both in their ratio of cases to controls and in how common a taxon is at baseline, subtracting after pooling allows study composition to masquerade as a health effect. This is Simpson's paradox (Simpson 1951): a taxon equally prevalent in cases and controls within every single study can still show a large pooled difference, and even one of the opposite sign, if the studies in which it is common happen to have contributed most of the controls. In this dataset the pooled and within-study differences disagree in sign for [[n]]% of taxon-by-condition comparisons."),

(686, "add the risk-difference outputs; the rest of the paragraph stands",
 "For each taxon, GMW-CMH yields a common log-odds ratio of detection in controls versus cases, evaluated at the shared within-study depth target, together with the weighted prevalence in controls and cases, their differenceprev_diff = prev_ctrl − prev_case, and a two-sided p-value.",
 "For each taxon, the weighted stratified tables yield the Mantel–Haenszel common risk difference with its Sato standard error, and the Mantel–Haenszel common log-odds ratio, both evaluated at the shared within-study depth target, together with the weighted prevalence in controls and cases, their difference prev_diff = prev_ctrl − prev_case, and the two-sided p-value of the CMH test. A Wald statistic and p-value for the risk difference are stored for completeness but carry no inference: the counts in Section 6D use the CMH p-value, and the list-level inference of Section 6E comes from the bootstrap described there."),

(690, "one sentence added for complete separation; the rest stands",
 "Because some individually significant taxa are expected by chance under a p < 0.05 threshold, we test whether the observed number of depleted or enriched taxa exceeds the 5% null expectation using a one-sided binomial excess test.",
 "Because some individually significant taxa are expected by chance under a p < 0.05 threshold, we test whether the observed number of depleted or enriched taxa exceeds the 5% null expectation using a one-sided binomial excess test. A taxon detected in every control and no case, or the reverse, yields a valid table with a small p-value but an odds ratio of zero or infinity; these are counted in the direction given by the sign of the Mantel–Haenszel numerator rather than discarded, since discarding them would remove the most extreme taxa from every count."),

(692, "only the last sentence changes",
 "We therefore compare each focal list to a prevalence-matched background of non-list taxa rather than to the raw background pool.",
 "We therefore reference each focal list to a fitted expectation for taxa of the same baseline prevalence, rather than to the raw background pool."),

(693, "keep the first three sentences; replace the density-ratio half",
 "We then fit kernel density estimates to the prevalence distribution of the focal list and to the prevalence distribution of the candidate background pool. Each background taxon receives an importance weight proportional to the ratio of these two densities, after regularizing the background density in sparse prevalence regions and clipping extreme weights. The resulting weighted background is therefore matched to the focal list in prevalence space rather than selected ad hoc by nearest neighbors.",
 "On that coordinate we then fit, to the background taxa alone, the expected case-control shift as a function of baseline prevalence: a local-linear regression with a Gaussian kernel of fixed bandwidth h = 0.12 log10 units, evaluated on a 400-point grid (Fan and Gijbels 1996). This is a fixed-bandwidth kernel regression rather than LOWESS (Cleveland 1979): the bandwidth is the same at every prevalence instead of spanning a fixed fraction of nearest neighbours, and no robustness iterations are performed. The local spread is obtained from the local mean absolute residual about that fit, scaled by sqrt(pi/2). Each list taxon is then scored by its distance from the fitted curve in units of that local spread, z = (observed shift − expected shift) / local spread. A background taxon enters the fit in proportion to how close its own control prevalence is to the taxon being scored, and according to nothing else: its standard error, its arm sizes and its number of contributing strata play no part in that weight."),

(694, "the fit has no OVL; replace with the two constraints that do govern it",
 "Matching quality is assessed using the overlap coefficient (OVL) between the focal-list prevalence density and the reweighted background density. Fig. S25 is the primary quality-control figure for this step: it shows the raw prevalence mismatch, the implied importance-weight function, and the post-weighting overlap for each list and condition. High OVL confirms that the weighted background provides a valid reference for asking whether focal-list taxa behave differently than equally prevalent non-list taxa.",
 "Two constraints govern the fit. The local spread is floored at the taxon's own Sato standard error, so that a taxon's distance from the curve is never expressed in units smaller than its own sampling noise. And a taxon is scored only if at least 15 background taxa lie within 0.25 log10 of its control prevalence; taxa failing this support rule are plotted but excluded from the list mean. Because the six focal lists overlap substantially, the background for every list is the set of taxa belonging to none of them, so that each list is compared against an identically constructed reference. Taxa at or above 85% control prevalence are retained but their expected shift is a one-sided extrapolation, since above that prevalence the support window reaches only taxa less prevalent than they are. Figure A25 is the quality-control figure for this step."),

(695, "mean z becomes primary; the energy distance is demoted, not removed",
 "Using this matched background, we compare the distribution of per-taxon health effects in the focal list to the distribution expected for equally prevalent background taxa. Our primary list-level statistic is the energy distance between the observed list and an importance-weighted reference background.",
 "Using this fitted expectation, we compare the per-taxon health effects in the focal list to what equally prevalent background taxa would show. Our primary list-level statistic is the mean of z over supported list taxa, with a 95% interval from a bootstrap that resamples the background taxa, refits the curve on each resample, and rescores the resampled list taxa against it. Refitting is the essential step: it propagates the uncertainty in the background expectation itself. The energy distance between the observed list and an importance-weighted reference background is retained as a descriptive shape statistic."),

(696, "demote the energy distance; the two secondary tests are unchanged and still computed",
 "These two tests are useful for interpretation and figure annotation, but throughout the note and the main Figure 5 narrative, energy distance is the primary list-level test, whereas the directional permutation and IW-Fisher statistics are supportive summaries.",
 "These two tests are useful for interpretation and figure annotation, but throughout the note and the main Figure 5 narrative, the mean z against the fitted background curve is the primary list-level test, whereas the directional permutation, IW-Fisher and energy-distance statistics are supportive summaries."),

(699, "the combined quantity changes; the HMP rationale stands",
 "For the primary prevalence-matched energydistance analysis, we therefore combine evidence across thresholds using the harmonic mean p-value (HMP), and we report the mean energy distance across specifications as a complementary effect-size summary.",
 "For the primary prevalence-background analysis, we therefore combine evidence across thresholds using the harmonic mean p-value (HMP), and we report the mean of z across specifications as a complementary effect-size summary. The HMP is valid under arbitrary dependence (Wilson 2019), which is what the two detection rules require of it: they are applied to the same taxa in the same samples, so their bootstrap p-values are strongly positively dependent and a method assuming independence would be anti-conservative. The quantities combined are the bootstrap p-values of the mean z; the energy distance is combined the same way as a descriptive check. Agreement between the two specifications is reported alongside the combination rather than in place of it, since the HMP measures the strength of the combined evidence and not the concordance of its parts."),

(702, "Pass 1 changes metric; Pass 2 is no longer computed",
 "Finally, to identify the taxa most responsible for a list-level signal, we perform two complementary driver analyses. The first summarizes cross-condition consistency by comparing each taxon’s prev_diff to the matched-background distribution within each condition and then aggregating both the direction and magnitude of excess across diseases. The second is a leave-one-out energydistance analysis, in which the change in list-level energy distance after removing one taxon (δE) quantifies that taxon’s contribution to the overall distributional divergence.",
 "Finally, to identify the taxa most responsible for a list-level signal, we rank each list taxon by its own distance from the background curve, z, within each condition, and aggregate across conditions to summarise directional consistency and average magnitude. Per-taxon significance is reported both uncorrected and after Benjamini–Hochberg correction within each list; the figures annotate the uncorrected value, so that a star and a confidence interval crossing zero cannot disagree, and encode the corrected value in the companion panel."),

(747, "6G.4 Step 2: the density-ratio weights are replaced by the curve; Step 1 and its equation are unchanged",
 "Step 2: Density-ratio importance weighting. Kernel density estimates are fitted to the scalar control prevalences of the focal-list taxa and to the scalar control prevalences of the candidate background pool. Each background taxon is then assigned a continuous importance weight:",
 "Step 2: The background curve. Let x_i = log10 p_i be the coordinate from Step 1 and y_i the case-control shift of taxon i in percentage points. Fitted on the background taxa alone, the expected shift is the local-linear Gaussian-kernel estimate with fixed bandwidth h = 0.12, evaluated on a 400-point grid over [1e-4, 1]; the kernel weight of background taxon j at a point x is exp(−(x_j − x)^2 / 2h^2) and depends on nothing else about that taxon. The local scale is sqrt(pi/2) times the local-linear fit of the absolute residuals, floored at the focal taxon's Sato standard error. Each list taxon is scored z_i = (y_i − mu(x_i)) / sigma(x_i), and is supported, and so enters the list mean, only if at least 15 background taxa lie within 0.25 log10 of x_i. Background taxa are those belonging to none of the six focal lists."),

(759, "6G.5: the energy distance is demoted from primary to descriptive",
 "IW-energy distance test (primary list-level test). Our primary list-level statistic is the energy distance between the observed focal-list prev_diff values and a fixed importance-weighted reference background.",
 "IW-energy distance (descriptive shape statistic). The energy distance between the observed focal-list values and a fixed importance-weighted reference background is retained as a descriptive summary, because it is sensitive to differences in spread and tails that a mean is not. It is no longer the primary list-level test: it treats the reweighted background as known, which understates uncertainty at the effective sample sizes available here."),

(777, "6G.8: only the name of the combined statistic changes; the equation is unchanged",
 "For the primary prevalence-matched energydistance analysis, evidence across thresholds is combined using the harmonic mean p-value (HMP):",
 "For the primary prevalence-background analysis, evidence across thresholds is combined using the harmonic mean p-value (HMP):"),
]

# --- C. insertions ----------------------------------------------------------
INS = [
("after 669, as a new paragraph in 6A", "the naming convention, which is the source of the confusion",
 """Two related names are used throughout and are distinguished here once. Cochran–Mantel–Haenszel (CMH) names the stratified test of conditional independence, which supplies the per-taxon p-value (Cochran 1954; Mantel and Haenszel 1959). Mantel–Haenszel (MH) names the two stratified estimators taken from the same tables: the common odds ratio (Mantel and Haenszel 1959) and the common risk difference (Greenland and Robins 1985; Sato 1989). We write GMW-CMH for the weighted stratified construction as a whole, CMH when we mean the test, and MH when we mean an estimator."""),

("as a new 6G.3, before the present 6G.3", "the risk difference and its variance, which the document does not define",
 """6G.3 The Mantel–Haenszel common risk difference

On the prevalence scale, let n1k and n2k be the weighted control and case arm sizes in stratum k, A_k and C_k the weighted detection counts, and N_k = n1k + n2k. With W_k = n1k n2k / N_k,

rd_MH = Σ_k W_k ( A_k/n1k − C_k/n2k ) / Σ_k W_k.

Its variance follows Sato (1989):

Var(rd_MH) = ( rd_MH Σ_k P_k + Σ_k Q_k ) / ( Σ_k W_k )²,

P_k = ( n1k² C_k − n2k² A_k + n1k n2k (n2k − n1k) / 2 ) / N_k²,
Q_k = ( A_k (n2k − C_k) + C_k (n1k − A_k) ) / ( 2 N_k ).

This estimator is consistent in the sparse-data limit as well as the large-strata limit (Greenland and Robins 1985; Robins, Breslow and Greenland 1986), which is the relevant regime here because many strata contribute only a few samples per arm. In the single-stratum limit it reduces exactly to p1(1−p1)/n1 + p2(1−p2)/n2."""),

("in 6G.5, before the IW-Fisher paragraph", "the primary statistic now needs its own implementation note",
 """Mean z against the background curve (primary list-level test). The statistic is the mean of z over supported list taxa. Its interval is obtained by resampling the background taxa with replacement and refitting the curve on each resample, resampling the supported list taxa and rescoring them against that refitted curve, and recording the mean. We take [[n_boot]] replicates under a fixed seed and report the 2.5th and 97.5th percentiles, with a two-sided p-value as twice the smaller tail proportion. A percentile bootstrap cannot resolve a tail finer than one replicate, so the p-value is floored at 1/[[n_boot]] and no significance band finer than that floor is used. This per-specification p-value is the quantity combined across detection rules in Section 6G.8."""),

("as a new 6G.9, after the driver procedures", "the within-clade figure has no description at all",
 """6G.9 Within-clade contrasts

To test whether the list-level signal survives holding taxonomy fixed, list members are compared with their own relatives. Within each genus or family containing at least five catalogue taxa, the background curve is refitted with that clade excluded — otherwise a large clade is partly its own control — and every clade member is scored against the refitted curve. The clade is summarised by its mean z, against a null of 5,000 placebo clades whose members are drawn from non-clade taxa within one bandwidth of each real member's control prevalence, so that a clade cannot score merely by being unusually common or rare. Then, within each clade holding at least three list taxa and three non-list taxa, we take the mean z of the list members minus that of the remaining members, with a p-value from permuting the list labels inside the clade. Taxa at or above 85% control prevalence are excluded here, since the clade-excluded refit requires a neighbourhood on both sides. Contrasts are corrected across clades by the Benjamini–Hochberg procedure."""),
]

# --- D. deletions -----------------------------------------------------------
DEL = [
("670", "a stray copy of the 6G.1 2×2 table equation, dropped into the overview where no table has been introduced"),
("671", "the orphan fragment “geometric-mean-weighted Cochran-Mantel-Haenszel test (GMW-CMH)”, absorbed into the revised 669"),
("678", "a bare “(87)” standing as its own paragraph — a stray citation marker; restore the citation it was meant to carry, or delete"),
("701", "the explicit null-distribution check over 500 importance-weighted draws. The paragraph already hedges with “Where retained”; no such procedure is computed in the current pipeline"),
("754", "6G.4 Step 3, largest-remainder replication of weights into integer rows. No procedure now requires replicated background rows"),
("765", "6G.6 null-distribution paragraph, for the same reason as 701"),
("768–771", "6G.6 Pass 2, the leave-one-out energy distance δE. Not computed in the current pipeline"),
]

KEEP = [
 ("6G.1", "The CMH statistic, its Yates correction, the variance of a_k, and the MH common odds ratio. All correct; only the heading changes, to name it as the test."),
 ("6G.2", "The whole of GMW weighting — the geometric-mean target, the per-sample weights, mean-normalisation, the clip at 10, the small-stratum fallback, and the weighted cell equations. Unchanged."),
 ("6G.3 (present)", "Per-taxon outputs and the Kish ESS. Keep, and add the risk-difference columns to the stored list."),
 ("6G.4 Step 1", "Stratum-aware control prevalence and its equation. This is still exactly how the matching coordinate is computed."),
 ("6G.5 first two tests", "IW-Fisher (ESS-adjusted) and the IW-directional permutation test. Both are still computed and still reported; only their ranking relative to the energy distance changes."),
 ("6G.6 Pass 1", "Cross-condition consistency. Keep the structure; the statistic compared is now z rather than prev_diff."),
 ("6G.7", "Mundlak within–between GLMM and the Hausman diagnostic. Unchanged."),
 ("6G.8 equation", "The harmonic mean p-value expression. Unchanged; only what p_i denotes changes."),
 ("Table S10, Table S11", "Captions and structure stand. The values move with the rerun."),
]


# --- anchors: what to SEARCH FOR in Word. The document is being edited while
# these notes are written -- it gained 8 paragraphs between two reads -- so
# paragraph numbers drift and only quoted text is a stable address.
ANCHOR = {
 679: "The primary taxon-level estimand in the health analyses",
 686: "yields a common log-odds ratio of detection in controls versus cases",
 690: "we test whether the observed number of depleted or enriched taxa exceeds",
 692: "rather than to the raw background pool",
 693: "We then fit kernel density estimates to the prevalence distribution",
 694: "Matching quality is assessed using the overlap coefficient",
 695: "Using this matched background, we compare the distribution",
 696: "energy distance is the primary list-level test",
 699: "we therefore combine evidence across thresholds using the harmonic mean",
 702: "we perform two complementary driver analyses",
 747: "Step 2: Density-ratio importance weighting",
 759: "IW-energy distance test (primary list-level test)",
 777: "evidence across thresholds is combined using the harmonic mean p-value",
}

# 6A's body is empty in the current file: the overview prose is gone and only a
# stray "6" remains, so this is an insertion rather than a revision.
INS.insert(0, ("6A, as the section's opening paragraph",
 "the body of 6A is currently empty -- only a stray \"6\" remains where the overview was",
 """6 describes the within-industrial health analyses underlying Figure 5. These comparisons are vulnerable to three confounds: sequencing-depth imbalance between case and control samples, heterogeneity across studies, and baseline-prevalence effects that constrain how much a taxon can change between health groups. To address these simultaneously, we equalise depth within each study stratum by geometric-mean weighting, combine strata by the Mantel\u2013Haenszel construction, and reference each taxon to a fitted expectation for taxa of the same baseline prevalence. We refer to the weighted stratified construction as the geometric-mean-weighted Cochran\u2013Mantel\u2013Haenszel test (GMW-CMH)."""))

# drop the 669 revision: there is no longer a paragraph to revise there
REV[:] = [r for r in REV if r[0] != 669]

DEL.insert(0, ("the stray \"6\" in 6A",
  "a bare \"6\" standing where the overview paragraph used to be; replaced by the insertion above"))

REV.append((691, "name the phenomenon the paragraph already describes",
 "To avoid this pooling fallacy, we estimate taxon-level health associations within studies rather than across a single pooled dataset.",
 "To avoid this pooling fallacy \u2014 Simpson's paradox in its classic form, in which an association present within every stratum reverses or vanishes on aggregation (Simpson 1951) \u2014 we estimate taxon-level health associations within studies rather than across a single pooled dataset."))
ANCHOR[691] = "To avoid this pooling fallacy"

INS.append(("at the end of 6G.8", "say what p_i is, where the combined value is written, and why the figures still show one specification",
 """Here p_i is the bootstrap p-value of the mean z under detection specification i for a given condition and focal list, and k = 2 is the number of specifications. The combined value is written to health_background_curve_meta.csv alongside each specification's own mean z and interval, and it is the value quoted in the main text. Because the bootstrap floors each p_i at 1/[[n_boot]], the HMP inherits that floor. The same combination is applied to the energy-distance permutation p-values as a descriptive check and is reported in the diagnostic figure health_prevdist_meta. Sign agreement between specifications is tabulated separately rather than folded into the combined p-value: a cell can carry a small HMP while the two rules disagree about the sign of a mean z close to zero. The figures annotate one specification and not the combination \u2014 Figure 5E and Figure A26 print the mean z, its interval and its bootstrap p under the primary rule, breadth \u2265 10%, so that the star in each panel matches the interval drawn beside it. In the present run the primary rule and the combined value agree on significance at the 5% level for every condition-by-list cell, and the two rules agree in the sign of the mean z for [[n]]% of them."""))
