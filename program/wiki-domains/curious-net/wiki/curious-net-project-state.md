---
title: Curious-net: project state
domain: curious-net
kind: project
tags: [curiosity, uncertainty, boolean-network, quantum-state, mdl, car-track, research]
updated: 2026-10-22
---

# Curious-net: project state

## Summary

User goal (raw: `raw/2026-10-22-user-brief.md`): one algorithm for any problem with inputs and outputs; units start Boolean, neuron count self-expands;
primary drive = reduce uncertainty, hidden goal = task score; it runs its own experiments; theorem + test + figures + paper cycle; **mid-session the user added that the Boolean representation should be converted into continuous quantum states** (interpreted as classical simulation of qubit states; unconfirmed).
Project `~/curious-net` (git; `LOG.md` is the resume point, `paper/theory.md` holds theorems and the hypothesis scoreboard). Plan authored by Fable 5.1 (read-only architecture-planner run). Compute rule inherited: 1 thread, below-normal priority.

## Design (v0, built)

* Env: continuous car on a 48x48 tile track (seeded closed spline), actions = steer L/S/R x throttle; off-track drag x4, no resets. Agent sees a 7x7 car-frame Boolean tile view plus named scalars (v, omega, sin, cos, lateral, heading error, on-track, drag, score) plus classed distractors (noise, clock, noisy copy of v, action lamp, lagged score).
* Encoding: each scalar -> 4 evenly spaced thermometer bits over the seen range + a delta bit. Each bit at t+1 is a target predicted from <=4 parent bits at t (incl. 3 action bits) by a KT count table; uncertainty = prequential bits/tick.
* Growth: every 500 ticks, MDL-gated greedy parent addition (gain must beat log2(#inputs) + 2^p * 0.5 * log2 W). Parents never removed; tables rebuilt from full history (no forgetting).
* Curiosity: learning-progress-weighted expected-information heuristic; exploit: 8-step sampled rollouts of the bit model.

## Results (4 seeds, toy world)

* Uncertainty falls ~45 -> 6 bits/tick at 0 distractors in ~1k ticks, then plateaus. 64 distractors: ~105-112 bits/tick, 11-13 false links to independent variables (0 at n=0).
* **Curiosity ties random actions** on action-link discovery (random uniform is already near-optimal for independent action bits). Its real test (state coverage) is not built.
* **Primary theory not supported yet:** model-rollout driving scores ~0.05/tick regardless of training length (random 0.048, hand PD controller 0.976).
* Steering->omega links mostly missed (1 of 3 action links at n=0); suspected resolution loss of Boolean thresholds (theory T3) -> the reason for the soft-bit and quantum-state tests.

## Gotchas

* Edges absent from a hand-written truth are not necessarily false: delta bits create self-links and copies of v / lamps legitimately predict each other. Use links touching independent-by-construction variables (noise, clock) as false positives.
* Warm-up thresholds from random actions made the score bits near-constant; calibrate over the seen range with a throttle-biased warm-up.
* Result JSONs are overwritten by re-runs; the pre-fix score of ~0.015 survives only in `LOG.md`.

## Open / next

Soft-bit control, then Q-mode (qubit states, von Neumann entropy beliefs, vs soft-bit control so quantum is not credited for continuity); state-seeking curiosity; regime-switch forgetting test; coherent joint rollouts for exploit; tabular-Q baseline; n=512 scaling. Confirm with the user: simulated vs real quantum.

## v2 (same day, later): two timescales, sleep, constants register

User directions: awake = fast self-correction only; **sleep (internal ticks, world frozen, nothing new measured) defines importances and makes every permanent connection from the stored day**; per-item complexity stored; per-item update rate that decays on correct predictions and slides back up; every unbacked number marked as a problem; goal framed as a model that fills a gap neural networks leave (not yet shown).
Design: permanent counts written only in sleep, fast counts + trust g=2^-S awake (S = running excess surprise), complexity in bits, sleep = full-history MDL grow/prune/2-input concept units to a fixed point, sleep only when awake progress is flat (BIC) and history has doubled. `src/constants.py` registers every number (CONSTANTS.md; 0 undeclared literals; 39 ARBITRARY = open problems). v1 and its dead ends are in `src/legacy/`.
Results (8 seeds): without sleep no structure forms (76.5 vs 14.2 bits/tick on causal variables); self-chosen sleep schedule matches sleeping every 250 ticks at ~6% of the sleep compute; trust speeds recovery after a physics change (47.1 vs 54.9 bits/tick); permanent memory survives a change and return; the tuned online MLP is worse overall (39 vs 31 bits/tick) but shows no catastrophic forgetting in this test, so the gap claim is unproven.
Open: trust miscalibrated (mean g 0.3-0.5 when stationary); awake vs sleep compute units differ; dreaming and latent-regime units unbuilt; soft/qubit not re-run on v2; 39 constants unswept.
Details: `~/curious-net/LOG.md`, `paper/theory.md`.

## Direction change and first results (2026-10-22, later)

User's refined goal: sleep is parked (to be studied separately as it affects neural-network learning); the subject is now a SELF-LEARNING NEURAL NETWORK: given a new problem with no context it performs identically to a plain neural network (verified: bitwise-identical at zero experience); experience changes how fast it works out WHAT AFFECTS ITS ENVIRONMENT, not re-use of weights. Built `worlds.py`, `nn.py`, `selfnn.py`; experiments K (headroom), L/L2, M.
Findings: knowing the true inputs is worth ~2x error / ~6x time at 200 distractors; a learned relevance prior (soft input gating, trained on the agent's own permutation-importance labels) improves monotonically with experience (K=3: 23% of the oracle headroom, K=30: 57%, K=100-300: 61%), one scenario of experience hurts, and a fixed statistical screen gives almost nothing on the diverse distribution. Not yet done: neural meta-learning baselines (in-context model, Reptile, learned optimiser), group-lasso/dropout, closed-loop experiment policy. Fable's warning: such baselines also speed up in-distribution, so the claim that NNs cannot do this is probably false as stated.
Continuous count-table learners (core3/core4) remain as diagnostic arms: core4 handles the drag-change example (first 100 ticks 37 vs 72 bits/tick for a fresh learner). Details: `~/curious-net/LOG.md`.

## Research loop results so far (2026-10-22, iterations 1-2b)
Dossier (`~/curious-net/paper/`: LEDGER.md claims, LITERATURE.md, PAPER.md, UNDERSTANDING.md, QUEUE.md; figures regenerated by `experiments/make_figures.py`). Findings: transformer headroom +0.294 (oracle attention mask); learned relevance prior recovers ~50% after ~10 scenarios and plateaus; the zero-experience wrapper is bitwise identical to the plain net (MLP and transformer); recursive self-labelling over 150 generations improves to a plateau (+0.166, ~49% of the oracle gain) and does not collapse because each generation's labels come from fresh real data; fly connectome (FlyWire v783) statistics: half of a neuron's input from a median of 3 partners, 90% from 9, heavy tail, so the generator's 1-3 parents match the dominant inputs only; teacher-student toy: adaptive teaching needs 30 interactions vs 120 random, trusting students beat skeptical ones against random teacher errors. Details and caveats: ledger rows C9-C13, C21-C28.



## Current state after iterations 3-13 (2026-10-22, written after the independent review)

**What holds (toy, one-day sprint, not replicated):** knowing the useful inputs halves an online MLP's early error under 200 distractors (C9); a small learned per-input gate fitted on ~10 earlier worlds gives +0.128 +- 0.012 (oracle hard mask +0.209), saturates by K = 10 and transfers under mild generator shifts (C10, C11, C14); no collapse over 150 self-labelled generations because each generation trains a fresh net on fresh real data (C13); fly-like and real-connectome worlds: the gate still helps (C22, C29).
**What it is not:** not causal-structure learning (oracle-label priors worse than own-label priors; AUROC for true parents 0.56-0.73; C23 NOT SUPPORTED). The hard oracle mask is not an upper bound (beaten in expS2 and q19), so percentages of it are not meaningful.
**Negative / limiting results (settled):** curiosity vs random for independent immediate effects (C3); quantum-state predictor wins only on QQ-structured data (C16); structured randomness never beats a single prior (C17); out-of-library law detection limited by noise vs library flexibility (C19, Q12b/c); oracle/noisy-teacher toys (C24-C26) show a perfectly trusting student is optimal only when the teacher is perfect.
**Corrections in iteration 13:** Q12 curious-at-n=3 was an artefact; hard-mask decay within noise; theory T4 calibration claim false; constants audit fails (76 literals); q19 truth rate 0.25 held-out.
**Next:** QUEUE R3 (fair soft-screen baseline), E1-E5 (oracle-taught student: calibration, edge detection, claim verdicts with abstention, contradicting data, non-oracle sources), R2 constants, L1 literature.

**R3 (iteration 13):** a two-constant soft screen gives +0.082 vs the learned prior +0.117 (paired +0.035 +- 0.005): ~70% of the gate's gain needs no cross-problem experience (C30). **Generator drift:** same seed != same world across iterations before/after 3b; compare only within an experiment.

**E1/E2 (iteration 14):** oracle-taught student calibrated in-family (ECE 0.007) and under mild shift; confidently wrong on connectome worlds (0.013 vs 0.236, AUROC 0.48); own-uncertainty novelty scores fail (inverted); input-distance novelty works 0.71-0.86 but misses fly-like; marginal conformal useless for rare positives (C31).

**R6/X1 (iteration 15):** perfect measurement: 5 own drops give 100% law identification, 100% out-of-library flags, 0 false flags, 1%-off claims caught; design irrelevant. 3% noise: resolution limit returns; chosen design helps early (C32).

**Pieces tested so far (iterations 17-31, rigor rules):** B1 memory+replay PARTIAL (trade-off at equal compute); B7 surprise PARTIAL (reset+boost pair); P5 mastery NOT YET (recognition is the unsolved core; 2/24 predictions); B9 sleep-merging NOT SUPPORTED as built; P3 curiosity PARTIAL (information gain avoids the noisy TV; nothing beats random allocation); **B11 wiring energy SUPPORTED and replicated** (-12.8% shifted error at 5% weight mass), activity energy not. Prediction register: paper/PREDICTIONS.md.

**Iterations 32-34b:** A3 scan ingredients SUPPORTED as a ranking (signs >> intrinsic ~ strengths > neuromodulation; 8/8 predictions); T3 rune universe SUPPORTED with a clean dose-response (block library: 1.00x with unrelated problems -> 7.5x with fully shared blocks; 6/7 predictions).

**Iterations 59-61 (context inference, C46):** a memory (episodic) router plus asking only where its 5 nearest memories disagree solves the Self's core problem in the P6 world: at test 99.5% routing and 1.3-1.4 x oracle (from 5-10 x); during learning parts reach 1.12 x oracle with questions on 6.8% of ticks, beating 1.5 x more random labels; end-to-end 1.29 x. Replicated. Surprise: a more accurate memory vote without asking did not give better parts.

**Iterations 62-69 (context, continued):** confident misroutes are 40-46 x more damaging than error-based ones (confirmation bias, R); a no-question surprise gate gets ~70-75% of the way to asking (S); questions at shocks + disagreements give essentially perfect parts (1.02 x, R); asking on shocks transfers to a TIME-context world (oracle-level routing with 0.34% questions) while input-memory recognition does not; a meta-recogniser judged by predictive consequences picks the right recognition method in both kinds of world (S). Two design flaws caught and recorded (no manipulation check in CTX11; a '3 x efficiency' overstatement in CTX9c, corrected by CTX10).

**Synthesis (iteration 37, C45):** across mastery and Self (13 variants) the unsolved core is latent-context inference: given the right memory/part everything works (practice halves error; specialists 20x better); inferring it from errors alone fails every time. Candidate core job of the Self. Next: PIECE-CTX.
