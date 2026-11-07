# Population-density hypothesis (user, 2026-10-26) — pre-registered before any run

User's idea (paraphrased): treat population size like literal population density. The fewer minds, the less variety and the more stagnation; with enough minds (up to the order of magnitude of humanity's effective population, ~10,000) variety is sustained and EXPERTS can emerge.

## Design (fixed here)
* Independent variable: population size N in {500, 1000, 2000, 5000, 10000} (start with 500, 1000, 2000 as the pilot ladder). Everything else fixed: the 24-problem task, free self-modification, cull = 0.9 * u^6, random-collection refill (copy 25 / variation 25 / crossover 20 / blend 10 / scramble 10 / fresh 10), brain size cap 96, task code on, 3 seeds, G generations (pilot 1,000; full 3,000).
* Note: "density" here means well-mixed population SIZE (no space). A spatial version (brains on a grid, local competition) is a later extension and is not tested here.
* Comparison condition for the cull rule (labelled): MEAN = rank by average accuracy over all problems (the user's rule, primary); NICHE = rank by the best normalised accuracy on ANY single problem (keeps specialists). Reported separately, never pooled.

## Measures (all computed on fixed reference test data, never used for selection)
* Variety V (four measures): effective lineages (inverse Simpson), distinct functions on a probe set, mean pairwise genotype distance, SD of scores.
* Stagnation: the generation after which the best population-mean score gains less than 0.001 per 100 generations (reported as "stagnates at generation g*", or "still improving" if it never does), plus final score.
* Expertise (per checkpoint, using per-problem accuracy of every brain): 
  E1 = number of DISTINCT brains that are the best on at least one problem (1 = one champion does everything; up to 24 = full division of labour);
  E2 = mean over problems of [accuracy of that problem's best brain - accuracy of the overall best (champion) brain on that problem], the "expert advantage";
  E3 = number of problems where the problem's best brain beats the champion by at least 5 percentage points.
  Chance level of E1/E2 is estimated from a SHUFFLE control (per-problem accuracies shuffled across brains, same marginals) so that noise alone is not read as expertise.

## Predictions (all must hold for the idea to stand)
P1  Variety increases with N at the end of the run (all four measures, Spearman rank correlation with N >= 0.8 across the ladder, per seed pooled).
P2  Stagnation is earlier and final score lower at small N: final score increases with N, and g* is earlier at N=500 than at the largest N.
P3  Expertise increases with N: E1 and E2 (relative to the shuffle control) are higher at larger N; under MEAN selection a threshold N may exist below which E1 stays at the shuffle level.
P4  Under NICHE selection experts appear at smaller N than under MEAN selection (the rule matters as well as the size).
## What falsifies it
* Variety does not increase with N (then size is not the driver of variety here); or score/stagnation do not depend on N; or E1/E2 stay at the shuffle level for all N under both rules (no experts emerge in this regime); or the effect is non-monotone (then the 'more minds is better' reading fails and the interior value is reported).
## Reporting rule
Whatever happens is reported with numbers; the ladder, seeds and measures are fixed above; changing them after seeing results requires a new dated pre-registration.

## AMENDMENT 1 (2026-10-26, made BEFORE any run of this study): expertise is cross-validated; the shuffle control is withdrawn
Reason: shuffling each problem's accuracies across brains makes every problem's best brain differ by construction, so a shuffled population would score as maximal division of labour; it is not a valid null. Selecting the best brain per problem on a noisy evaluation also over-states its ability (winner's curse).
Replacement (unbiased): evaluate every brain twice with independent randomness (A and B). Pick each problem's best brain and the overall champion on A; judge both on B.
  E1 = number of distinct brains that are the best on at least one problem (on A);  E2_cv = mean over problems [B-accuracy of the problem's best brain - B-accuracy of the champion on that problem];
  E3_cv = number of problems where that cross-validated advantage is at least 0.05.
Null: with no real specialisation E2_cv is about 0 (or negative if the champion is truly better everywhere); positive E2_cv/E3_cv is evidence of experts. E1 is reported but only interpreted together with E2_cv.
Predictions P3/P4 now refer to E2_cv and E3_cv. Everything else is unchanged.

## AMENDMENT 2 (2026-10-26, made BEFORE any run of this study): expertise must be measured against baselines, and evaluation B must use different data
Found in a 21-generation test: a population of purely RANDOM brains (generation 0) already shows E1 = 23 distinct best brains and E2_cv = +25 percentage points. Cause: among N random brains there are always lucky ones that happen to fit one problem well (an extreme-value effect that grows with N), so raw E1/E2/E3 conflate "luck of random variety" with "expertise that evolution produced", and the effect depends on N, the very variable under test.
Also, evaluations A and B used the same 32 test rows per problem (only the stochastic node noise differed), so B was not independent data.
Fixes (both before any run):
1. Evaluation B now uses DIFFERENT, independently drawn reference-test rows from A (different random subset of each problem's reference test set).
2. Expertise is reported as EXCESS over two baselines computed at the SAME population size N: (a) the generation-0 random population, and (b) the no-selection control (cull 0, same self-modification and refill, same number of generations). "Experts emerged" means the measure exceeds BOTH baselines, and the champion's mean accuracy is reported alongside (a population whose champion is good everywhere is a generalist, not a population of experts, even if E1 is high).
3. Added P5 (prediction): the excess expertise over the no-selection control rises with N and with the niche rule. Falsified if excess expertise is zero or negative at every N under both rules.
