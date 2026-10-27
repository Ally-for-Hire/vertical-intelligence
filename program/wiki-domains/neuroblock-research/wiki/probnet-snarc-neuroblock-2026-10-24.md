---
title: Probability net: SNARC operant conditioning with a NeuroBlock substructure
domain: neuroblock-research
kind: experiment
tags: [probnet, snarc, operant-conditioning, neuroblock, 68-problem-suite, car]
updated: 2026-10-24
---

# Probability net (probnet): attempts and first working design

Code, full attempt log and constants list: `~/neuroblock-research/probnet/` (`LOG.md`, `CONSTANTS.md`, commit 4c9b92e).

## User's spec (2026-10-24, evolved over the session)
Drop neural-net machinery. On/off cells decide by probability; a system may only suggest, cells self-decide; a success estimate in 0..1 starts near chance; output when a group holds a 55% majority; repeated connections cost less energy; must work on the car and on a 50+ problem set. Redirect 1: grid substrate, one purely random neuron per problem, then prediction and growth. Redirect 2 (final): learn by operant conditioning, heavily SNARC-based (Minsky and Edmonds 1951, stochastic synapses reinforced when the machine succeeds; description from memory, unverified against a source), with a NeuroBlock substructure. The user also lifted the 1-thread compute limit for these runs ("give it my entire computer"; 24 logical cores, session-scoped).

## Suite and references
`problems.py`: 68 problems in 9 families (2-/3-/N-input boolean, distractor bits, small arithmetic, 2D/3D classification with thermometer bits, linear thresholds, pictures, next-bit sequences). `reference.json`: chance (best constant) mean 0.570; a 32-unit plain net shown the right answers 0.965. Probability nets only see success/fail.

## Attempts
- v0 dense stochastic net + REINFORCE on cell flips: at chance (credit diluted over a 9-cell majority).
- v1 estimator + suggester + self-deciding cells: bimodal on a copy task (3/6 seeds), then 0/6 with surprise weighting; cause: no exploration floor.
- v2a grid substrate, one random neuron, estimator, growth: mean 0.664, beats chance on 31/68; the estimator (a logistic model) did the work, with the suggestion off it fell to chance; car = random steering.
- v2b energy term + tie-break: worse (0.367).
- **v4a SNARC + NeuroBlock (`snarc.py`)**: stimulus cells = NeuroBlock-style prototypes with radius, spawned on unexplained input and split on failure (half radius, inherit synapses), merged/pruned when redundant; each cell holds SNARC synapse probabilities over responses (start uniform); all matching cells vote by sampling, 55% weighted majority emits; only cells that voted for the emitted response are reinforced, step 1/(visits+2).
  Suite: mean 0.893 (chance 0.570, net 0.965), beats chance 62/68, within 10 points of net 56/68; XOR/parity/ADD solved with a handful of cells. Fails on irrelevant-bit problems (radius counts junk dimensions) and wide pictures.
  Car (held-out tracks, reward-only): 0.870 / 0.906 / 0.852 vs random 0.155, hand controller 0.962.

## Follow-ups (same day)
- **Feature relevance (v4b/v4c, `snarc_rel.py`)**: cells learn which inputs matter from the trial log (alone + pair chi-square, alpha spending). Big gain on junk-bit problems (distractor family 0.57 -> 0.91, pictures 0.61 -> 0.77) but ~20 problems lose 5-20 points; suite mean 0.887 vs 0.893 for v4a, so a trade-off. An early version failed on PARITY4 (1.0 -> 0.5) because a false alarm shrank the mask and a null result kept it; fixed by alpha spending and 'nothing significant = all inputs'. Idea not yet tried: keep full cells and add reduced cells beside them.
- **World change (`drift_test.py`, 4 seeds)**: drag x2.5 did not affect the car (uninformative). Steering reversed: 0.87 -> 0.08, frozen copy stays 0.08, the learning net recovers to 0.70 within 500 ticks and 0.85-0.87 within ~1250 ticks (x16 cars). Operant conditioning relearns a reversed world with no signal that the world changed.

## Caveats
Toy tasks; 3 seeds; success signal for the car is my own definition; it is a growing nearest-prototype table shaped by reward, so it will not generalise across many irrelevant dimensions without feature selection; constants still hand-set (list in CONSTANTS.md). Next: per-cell feature relevance, NeuroBlock map/energy, tabular baseline with the same signal, drag-change test.

Related: [[wiki/neuroblock-project-state]], [[wiki/car-steering-test-2026-10-23]].
