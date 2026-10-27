# Random-cull brains on 24 problems — what worked and what did not (series closed 2026-10-26)

Closed by the user ("reframe the whole test series ... stop everything else and document our successes and failures"). Details, numbers and every revision are in `RESEARCH_LOG.md`; every code revision is kept as a `*_vN_snapshot.py` file; code is committed (git 3496613, 855943b).

## Setup that all these runs shared
Self-growing recurrent stochastic brains (`pop2.py`: any-direction edges, T internal steps, six node types, self-adaptive mutation genes), scored on 24 toy problems through one shared 16-line input bus plus a 5-line task code (`multi_big.py`). Every brain modifies itself randomly every generation and keeps the change; an external cull removes brains; empty slots are refilled.

## Successes (engineering, all verified)
* **Speed:** 3.8x end to end at 1,000 brains (300 generations: 127 s -> 33 s); forward pass ~6x on evolved populations with outputs bit-identical to the reference (fp32, `test_fast.py`, difference 0.0); mutation ~2x with the same random distribution (`test_mut.py`); 10,000 brains ~0.6 s/generation early in a run.
* **Safety:** the process can use at most 50% of GPU memory (`NB_GPU_FRAC`), after two PC crashes (bugcheck 0x20001 HYPERVISOR_ERROR at 17:51 and 20:06; cause NOT established).
* **Resume from crash:** rolling state file every 2 minutes, atomic; a stopped and resumed run equals an uninterrupted one in every generation.
* **Reproducibility:** fixed seeds; diagnostics no longer consume the run's random numbers; two runs with the same seed are identical in every generation (same hardware/software).
* **Diagnosis tool:** `ceiling.py` (gradient-trained reference models on the same inputs).

## Failures (scientific: the brains never got past a low plateau)
| what was tried | result |
|---|---|
| 10,000 brains, random cull skewed small, mixed refill | mean 0.47, best 0.57 at generation 30-40; stopped for changes |
| uniform cull 0-75%, then 20-60% | collapse to 1 lineage by ~generation 100-300; plateau ~0.59 |
| 1,000 brains, 20-60% cull, 3,000 generations | end: 1 lineage, 95 hidden nodes, mean 0.588, 0/24 solved at 90% |
| cosine wave 15-50% (period 50, then 1,000), blend-only refill | plateau 0.56-0.61; 1 lineage, 5-16% distinct functions |
| lineage-improvement cull (50% and 15%), random refill | stayed at the random level 0.452-0.455 |
| near-copy culling (threshold derived from random brains) | almost never fired (0-1 per generation) when refill was random |
| sliding-window 15th-percentile death, all-random brains, survival score | stayed at 0.47 for 450 generations |
| untrained-NN bar | bar = chance (0.37-0.51), almost nobody died; mean 0.53 |
| learning-NN bar (never drops) | bar beat every brain at generation 2; 462 restarts from random in 475 generations |
| ReLU cull ramp 15-50% per 100 generations, 3,000 brains | mean 0.600 at generation 850, T collapsed to 1 |
| wave peak 0.10 (peak sweep, 1 of 24 runs finished) | final 0.555 |

## Why (measured, `ceiling.py` and the diagnostic run)
1. **The plateau is the ceiling of a one-step (linear) brain.** A gradient-trained shared linear model on the same inputs reaches 0.549 with 1/24 solved; the evolved brains sat at 0.55-0.61. A small gradient-trained neural net reaches 0.973 (22-23/24), so the task is solvable; the search did not find it.
2. **Depth was selected away.** Mean internal steps T fell from 2.0 to 1.0 within ~50 generations in every score-ranked run; at T=1 hidden nodes cannot affect the output, and they then grew to the 96 cap as dead weight. Forcing T >= 2 raised the plateau only to 0.62-0.645.
3. **Variety collapsed** to 1-2 lineages before any hidden-unit circuit could be assembled (XOR/parity need several coordinated weights: a fitness valley).
4. **Lesson (user): no repopulating from random.** Every run that refilled from fresh random brains stayed at the random baseline.

## What the next series changes (user, 2026-10-26)
Start from the minimum: pure random weights and biases, internal randomness, external culling, ONE problem, and continue until something with neural-net behaviour emerges at a measurable efficiency. See `emergent/README.md`.
