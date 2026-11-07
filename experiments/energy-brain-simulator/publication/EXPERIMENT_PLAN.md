# Experiment plan: turning the claims into publishable evidence

Principle: every experiment below states its question, design, success/failure criteria and compute BEFORE it is run (as `VARIETY_HYPOTHESIS.md` already does for the variety window). Seeds >= 10 for any comparison; report spread; negative results are reported.
Compute numbers are estimates from measured speeds on the local RTX 5070 laptop GPU (10,000 brains ~ 3.5-7 s/generation) or Modal L4 (~$0.80/h).

## E0. Engineering that makes results publishable (do first)
* **Stateless, block-keyed random numbers** (the "ticker tape"): every random draw is a pure function of (run seed, generation, stream, block of brains) using the counter-based Philox generator, re-seeded per fixed block. Result: a run is bit-for-bit repeatable, independent of execution order, and the population can be split across GPUs or machines with identical outcomes. Deliverable: a determinism test (same seed twice => identical story file; split run == single run).
* **Performance pass** (profile, then fix the top costs): candidates are the three diagnostic forward passes per generation (measure every k generations), TF32/bf16 matmul (validate the numerics), skipping unused activation variants, fused kernels (torch.compile on the Linux/Modal side). Target: >= 3x faster with unchanged dynamics (checked by comparing score trajectories over 10 seeds).
* Deliverable: `reproduce.py` runs every figure/table from saved data in minutes.

## E1. Cull theory on realistic populations (Paper A)
* Question: do the identity and the shared-batch selection law (C1-C4) hold for stochastic networks WITH hidden units, on several problems, and under other selection rules (tournament, rank-proportional)?
* Design: 6 problems x {truncation, tournament} x 4 noise levels x 10 seeds x 100 generations. Instrument z exactly where possible, else by high-sample Monte Carlo (estimation error reported).
* Success: identity residual below estimation error everywhere; closed-form median error < 15% at moderate culls; interior optimum reproduced. Failure is reported and narrows the claim.
* Compute: small (hours on the local GPU).

## E2. Variety-window hypothesis (the central theory; Paper B)
* Question and rules: `selfscore/VARIETY_HYPOTHESIS.md` (frozen).
* Step 1: finish the pre-registered grid (162 runs; ~2-4 GPU hours).
* Step 2 (replication at scale): sweep variety in the 24-problem, 10,000-brain setting by changing the cull power and the refill mix (copy / variation / crossover / scramble / fresh), measuring variety four ways (effective lineages, distinct functions, genotype distance, accuracy SD) and emergent complexity as tasks solved at 90%+ that need hidden units.
* Success: a window with a lower limit, an upper limit and a plateau on >= 2 independent variety measures. Failure: monotone or single-peaked response, or only one measure shows it => reported as such.
* Compute: Step 1 small; Step 2 about 30 configurations x 3 seeds x 1,000 generations ~ 100-300 GPU-hours (~$80-240 on L4) unless the performance pass lands.

## E3. Random-cull multi-problem brains vs strong controls (Paper B/C)
* Question: does "free random self-modification + skewed cull + random-collection refill" produce brains that solve many problems, and how do they compare with conventional training on the same tasks?
* Conditions (10 seeds each, 3,000 generations): proposed method; no cull (drift control); fixed 50% cull; gated inner life; task-code removed; and a gradient-trained MLP on all 24 problems (upper reference, reported separately because it uses labels).
* Metrics: mean accuracy and chance-adjusted accuracy over all problems, tasks solved at 90%+, size of brain (nodes), generations to first hidden-unit solution.
* Success: significantly above the no-cull control on mean accuracy with the effect surviving the task-code ablation.
* Compute: 10 seeds x 6 conditions x ~6 h = ~360 GPU-hours (~$290 on L4). Run a 2-seed pilot first (~$20) to size the real study.

## E4. External validity
* Re-run the strongest result on one or two public benchmark datasets (small tabular / image subsets) with a standard split, so the work is not only on problems we wrote ourselves.

## Order and decision points
E0 -> E1 (gives Paper A its evidence) -> E2 step 1 (cheap, tests the user's central theory) -> decide on E2 step 2 / E3 by what step 1 shows. Do not buy large compute before the E2 step 1 and the E3 pilot results exist.

## E5. FUTURE (user request 2026-10-26, "not for today"): shared knowledge of how to solve problems
A repeat of the random-cull multi-problem experiment in which the brains also have SHARED KNOWLEDGE: shared category weights and biases, i.e. parameter blocks that every brain reads (and possibly writes), organised by problem category (for example Boolean logic, 2-D shapes, multi-class blobs), in addition to each brain's own weights.
Questions to design before running: do shared blocks let a small population match a large one (knowledge standing in for population size)? do they speed the emergence of experts? what is the update rule for shared blocks (read-only prior, averaged across survivors, or written by the best brain in a category)? Controls: same experiment without sharing, at the same population sizes. Not designed or run yet.
