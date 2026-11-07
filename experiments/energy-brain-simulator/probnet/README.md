# probnet: a probability net that learns by operant conditioning (SNARC + NeuroBlock cells)

One purely random cell per problem; it learns only from success or failure; it grows, splits and tidies itself. Full attempt history: `LOG.md`.
Constants and which are still hand-set: `CONSTANTS.md`. Wiki page: `Knowledge-Wiki/domains/neuroblock-research/wiki/probnet-snarc-neuroblock-2026-10-24.md`.

## The design that works (`snarc.py`, class `SnarcNet`)
- **Cell** = NeuroBlock-style specialist: a prototype, a radius, and SNARC-style synapse probabilities over the K possible responses (start uniform = random).
- **Trial**: every cell whose radius contains the input fires and votes by sampling its own probabilities (specific cells weigh more). A response holding a
  55% weighted majority is emitted; otherwise the net is "unsure" and uses the most specific cell's draw.
- **Reinforcement**: cells that voted for the emitted response move toward it on success and away on failure, step 1/(visits+2). Nothing else changes.
- **Growth**: a cell that fails with >= 10 visits spawns a child at the failing input (half radius, inherited synapses). Children not better than their parent are merged
  away; cells that never fire are pruned. Inputs no cell covers spawn a new random cell.
- `snarc_rel.py` (`SnarcRel`) adds a statistical test for which inputs matter (helps junk-input problems, costs accuracy on about 20 others; see `LOG.md` attempts 5-6).
- Superseded attempts kept for the record: `net.py` (dense stochastic net), `net2.py` / `net2_v1_snapshot.py` (estimator + suggester), `grid.py` (grid substrate).

## Results (all numbers are in `results_*.json`; 3 seeds unless noted)
| Test | Chance / random | Plain neural net (sees the answers) | Probability net |
| --- | --- | --- | --- |
| 68-problem suite, mean accuracy (`SnarcNet`) | 0.570 | 0.965 | **0.893** (beats chance 62/68; within 10 points of the net 56/68) |
| same, grid + estimator version (`GridNet`, attempt 2) | 0.570 | 0.965 | 0.664 |
| same, with relevance test (`SnarcRel`) | 0.570 | 0.965 | 0.887 (distractor family 0.57 -> 0.91) |
| Car, held-out tracks, reward-only (`SnarcNet`) | 0.155 | not like-for-like | **0.870 / 0.906 / 0.852** (hand controller 0.962) |
| Car after steering is reversed, keeps learning | frozen copy 0.08 | | 0.28 -> 0.70 (500 ticks) -> 0.85-0.87 (1250 ticks) |

Weak spots: problems with many irrelevant inputs (fixed only partly by `SnarcRel`), wide pictures, and anything needing feature selection. It is a growing
nearest-prototype table shaped by reward, not a general learner. Results are on toy tasks and 3-4 seeds; the car's success signal is a definition of mine.

## Files
| File | What |
| --- | --- |
| `problems.py` | the 68-problem suite in 9 families (`python problems.py` lists them) |
| `ref_mlp.py` | chance and plain-net reference scores -> `reference.json` |
| `run_suite.py` | train a net class on every problem in parallel -> `results_<tag>.json` |
| `car_task.py`, `car_env.py` | the car as a 3-option problem (steer -1/0/+1, throttle on); `--cls SnarcNet` |
| `drift_test.py` | train, change the world (drag or steering reversal), keep training, compare with a frozen copy |
| `LOG.md`, `CONSTANTS.md` | every attempt (including failures) and the constants audit |

## Reproduce (Windows, Python 3.11, numpy 2.4; workers default to 24, set `--procs` for fewer cores)
```
python ref_mlp.py                                                   # references, ~15 s
python run_suite.py v3a --cls SnarcNet --batches 125 --seeds 3      # the main suite result, ~25 s on 24 cores
python run_suite.py v4c --cls SnarcRel --batches 125 --seeds 3      # with relevance test
python car_task.py --cls SnarcNet --ticks 3000 --seeds 3 --tag car_v3a
python drift_test.py --seeds 4                                      # world-change test (steering reversal, drag)
```
Run from this folder with plain `python` (not `python -I`). The project default compute limit is 1 thread at low priority (see `../CLAUDE.md`); the user lifted it for
these runs on 2026-10-24, so pass `--procs 1` if that limit should apply again. Results change slightly between runs only through the seeds listed; same seeds give the same numbers.
