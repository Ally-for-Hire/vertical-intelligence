# Probability net: attempt log

Goal (user, 2026-10-24): drop neural-net machinery; binary on/off cells that decide for themselves by probability; an estimator that
computes the possibility of succeeding (0..1); the system can only SUGGEST, cells self-decide; output = a group in a >=55% majority;
connections get cheaper with repeated use; works on the car and on a 50+ problem set. Document every attempt, including failures.

Reference (reference.json, 68 problems in 9 families): chance mean 0.570; plain neural net shown the right answers 0.965.
The probability net only ever sees success/fail, so the net reference is an easier setting.

## Attempt 0 (v0, net.py): dense stochastic net + REINFORCE on cell coin flips, energy cost, pruning
Result: at chance on AND/XOR/NAND/... (e.g. AND 0.74 vs chance 0.75). 'NOT A' (a copy of an input) only solved in 1 of 2 seeds at lr=1; other lr 0.3-30 failed.
Why (diagnosed): success depends on a 9-cell majority, so one cell's coin flip says almost nothing about success (credit is diluted); the net settles on the
constant best answer and stops.

## Attempt 1 (v1, net2_v1_snapshot.py): success estimator + suggester + self-deciding output cells
Estimator = per-option logistic model over (input bits, hidden cell states) -> P(success) in (0,1). Suggester picks an option with probability proportional
to the odds P/(1-P) and pushes that option's output cells up and the others down by logit(P) (never sets a cell). Output weights imitate states that preceded a
success. Hidden cells = random fixed wiring.
Result on 'NOT A', 6 seeds: 3 solve it (0.98-0.99), 3 lock onto one constant answer (0.51). Bimodal: success-only imitation also reinforces lucky constant answers.

## Attempt 1b (net2.py): learn from surprise (success minus the estimator's expectation) instead of success alone
Result on 'NOT A', 6 seeds: 0 of 6 solve it. Diagnosed: with the estimator at 0.5 the push logit(0.5)=0 so the net never tries the other option, so the estimator
never sees it. The suggestion needs a floor (exploration). Proposed floor: derive it from the 55% rule (weakest push that lets a suggested group reach a majority
reliably). NOT RUN: the user redirected before this edit was applied.

## Redirect (user, same session): grid substrate, one random seed neuron per problem, then prediction and growth
See attempts 2+ below.

## Attempt 2 (v2a, grid.py snapshot grid_v2a_snapshot.py): grid substrate, 1 random seed neuron, estimator + suggester, growth when progress stalls
Suite (68 problems, 3 seeds, 1500 batches of 64, all 24 cores, 95 s): mean accuracy 0.664 vs chance 0.570 vs plain neural net 0.965.
Beats chance by >3 points on 31/68 problems, within 10 points of the plain net on 22/68. By family (pn / chance / net):
bool2 .88/.67/1.0, bool3 .75/.64/1.0, boolN .45/.58/.98, distractor .80/.57/1.0, arith .38/.48/1.0, classify .57/.50/.87, linear .86/.78/1.0,
pattern .59/.44/1.0, sequence .87/.57/1.0. Solves linearly separable things (NOT, OR, MAJ3, copy-a-bit-from-the-past); fails XOR/parity
and multi-class arithmetic, where it guesses all the time. With the suggestion switched off the accuracy falls to chance on most problems
(mean 0.542): the success ESTIMATOR (a logistic model over cell states) did the work, not the cells' own weights.
Car (3 seeds, 3000 ticks): grade 0.14-0.18 = random steering (0.155). Diagnosis: all free cells drifted to 'on' (63%), all 3 option groups passed
the 55% majority at once (counted as 'not unique' = guess), estimator weights blew up (max 34), net grew to fill the whole grid (168 cells).

## Attempt 3 (v2b): strongest-majority tie-break, estimator L2, activity energy in the advantage
Suite: mean 0.367, beats chance on 0/68 (worse than attempt 2: the energy term + tie-break destroyed learning, below chance on the multi-class problems). Car: 0.065-0.148.
Not diagnosed further: the user redirected to operant conditioning (SNARC).

## Redirect (user): operant conditioning, heavily SNARC-based, with a NeuroBlock substructure
SNARC (Minsky and Edmonds, 1951): ~40 'Hebb synapses', each a probability that a signal passing is passed on; when the machine succeeds, the synapses
that were recently active are reinforced (operant conditioning of a maze runner). Mapping used from here (snarc.py): stimulus cells = NeuroBlock-style
prototype specialists that spawn on unexplained input; each cell holds SNARC synapse probabilities over the K responses (starts uniform = purely random);
every matching cell votes by sampling its own synapses; the answer is the response with a >=55% weighted majority; reinforcement touches only the
cells that voted for the emitted response.

## Attempt 4 (v4a = snarc.py, `SnarcNet`): SNARC operant conditioning + NeuroBlock substructure  [WORKS; first real result]
Design: see the module docstring. One purely random cell at the start; cells spawn on unexplained input and split (half radius, inherit synapses) when
they fail with >=10 visits; children not better than their parent are merged away; unused cells are pruned. Step size 1/(visits+2) (running average).
No estimator network, no gradient, no learning-rate knob; only success/fail reaches it.
Suite (68 problems, 3 seeds, 8000 trials each, 24 cores, 25 s total): mean accuracy 0.893 vs chance 0.570 vs plain neural net (shown the answers) 0.965.
Beats chance on 62/68, within 10 points of the net on 56/68 (v2a: 31 and 22). By family (snarc / chance / net, mean cells):
bool2 1.00/.67/1.0 (8), bool3 1.00/.64/1.0 (12), boolN .96/.58/.98 (89), distractor .57/.57/1.0 (50), arith .99/.48/1.0 (43), classify .84/.50/.87 (231),
linear .83/.78/1.0 (42), pattern .61/.44/1.0 (63), sequence 1.00/.57/1.0 (14).
XOR, parity, 7-way ADD and noisy XOR (hits its 0.91 ceiling) are solved with a handful of cells.
Failures (12 problems >10 points under the net): every problem with irrelevant input bits (the matching radius counts junk bits, so no cell ever
generalises: 'ONE bit, 10 junk bits' 0.50, 'AND of 2, 6 junk' 0.68, 'XOR of 2, 6 junk' 0.55, 'MAJ3, 5 junk' 0.55, THRESHOLD 3 of 8 matter 0.68), wide pictures
(LEFT vs RIGHT heavier 0.49, MIRROR SYMMETRIC 0.65, 3x3 LINE 0.73, CORNERS count 0.35, 5x5 SHAPES 0.82) and 3D OCTANT PARITY 0.62.
Reading: it is a growing nearest-prototype table whose answers are shaped by reward; strong when the number of distinct situations is small, weak when
generalising across many irrelevant dimensions needs feature selection (which the plain net does by weighting inputs).
Car (3 seeds, 3000 ticks x 16 cars on training tracks, graded on 8 held-out tracks): 0.870 / 0.906 / 0.852; random steering 0.155; hand controller 0.962.
Cells at the end: 166 / 71 / 440. Success signal (set by me, a constant of the task): on the road and (lateral error not growing or < quarter of the half-width).
Caveats: held-out tracks come from the same family of seeded loops; the earlier evolution-strategies car test used the score itself as fitness, so the
comparison is only indicative.

## Open problems / next attempts
1. Feature relevance: let each cell learn which inputs to ignore (per-cell radius per dimension, grown only along dimensions where outcomes vary) -> junk-bit problems.
2. Constants still hand-set (CONSTANTS.md): n_min=10, depth=6, cap=600, trials=8000, 55% weights = 1/radius.
3. Use the NeuroBlock map (positions, travel time, energy) rather than only prototypes + radii; add energy cost to cells that fire often.
4. Compare against a tabular/nearest-neighbour baseline with the same reward-only signal to see what the SNARC synapses add.
5. Non-stationary tests (change the car's drag mid-run), the property operant conditioning is meant to give.

## Attempt 5 (v4b, snarc_rel_v4b_snapshot.py): cells learn which inputs matter (relevance test on the trial log)  [mixed]
Test: per response, chi-square of success rate across an input's two values (alone) and across pairs (counted only if the pair beats either input alone, which flags
XOR pairs but not a junk bit paired with a real one); Bonferroni threshold x > -2 ln(alpha/tests), alpha 0.01; cells match only on relevant inputs; identical cells merge.
Suite mean 0.880 (v4a 0.893); beats chance 65/68 (v4a 62). Distractor family 0.57 -> 0.97, pattern 0.61 -> 0.78, linear 0.83 -> 0.91. But PARITY4 1.0 -> 0.5, EVEN ONES 0.94 -> 0.55.
Cause: a chance false alarm shrank the mask early (3% per look x 20 looks), and a later look with nothing significant KEPT the shrunken mask instead of restoring all inputs.

## Attempt 6 (v4c = snarc_rel.py now): alpha spending over repeated looks (alpha/(k(k+1))) and 'nothing significant = all inputs relevant'  [trade-off, not a win]
Suite mean 0.887 vs v4a 0.893 (beats chance 65/68, within 10 points of the net 49/68 vs 56). Distractor 0.91, pattern 0.77, linear 0.88, but ~20 problems lose 5-20 points vs v4a
(MUX 2-select 0.93 -> 0.71, RING 0.71 -> 0.53, COMPARE 2-bit 1.0 -> 0.82, STRIPES 1.0 -> 0.86 ...). Likely cause: every mask change merges cells destructively and the
smaller alpha lowers power, so cells are rebuilt and junk-bit problems reach 0.88-0.91 rather than 1.0. Not tested.
Next idea: do not merge destructively; keep the full-resolution cells and add reduced-input cells next to them, letting the more specific / better-performing ones vote.

## Attempt 7: world-change test (drift_test.py), 4 seeds, 3000 ticks of training then 3000 ticks in the changed world, graded on 8 held-out tracks
Arms: keeps learning vs a frozen copy made at the moment of change.
- drag x2.5: NO EFFECT on SnarcNet (0.869 -> 0.879; frozen 0.879). The sensors are normalised by top speed, so this change did not test adaptation. Uninformative.
- steering reversed (mirror): SnarcNet 0.869 -> 0.080 at the change; frozen stays 0.082; learning: 0.28 (250 ticks), 0.70 (500), 0.58 (750), 0.71, 0.85 (1250), then 0.85-0.87 to the end of the 3000 ticks.
  SnarcRel: before 0.725, 0.082 at the change, 0.30 -> 0.71 -> 0.86 within 750 ticks, then oscillating 0.68-0.86, ending 0.82-0.83.
Reading: reward-driven synapses do re-learn a reversed world to about the old level within ~1000 ticks (x16 cars), without being told anything changed. A harder, more
informative drift (e.g. steering authority halved, throttle response changed, or a change mid-lap on one car) is still to be done.
