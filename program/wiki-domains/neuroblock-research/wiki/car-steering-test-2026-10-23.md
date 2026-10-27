---
title: NeuroBlock car-steering test (first run)
domain: neuroblock-research
kind: experiment
tags: [neuroblock, car, steering, evolution-strategies, grading]
updated: 2026-10-23
---

# NeuroBlock car-steering test (first run, 2026-10-23)

## Idea (user's words, condensed)
Run the default NeuroBlock with all its systems, and grade the whole thing by how well it steers a car. A small neural net at the very bottom makes the choices; the block is the persistent "brain" across generations, the net trains on top of it.

## Built (`~/neuroblock-research/car/`, uncommitted work elsewhere in repo left alone)
- `env.py`: copy of curious-net's continuous `CarEnv` (48x48 track, score = on-track x speed/vmax x centredness). Actions: steer idx 0:-1, 1:0, 2:+1, +3 for throttle.
- `agent.py`: 7 sensors (lateral, heading error, speed, on-track, track centre at 3 look-ahead depths from the tile view). `CarBlock(Brain2)` with the sweep default config (rho 0.5, move/merge/prune on, fast firing) learns situation prototypes online and stores the reward that followed each (in `lab`). Policy net = tanh MLP (16 hidden) reading sensors + winner prototype + novelty + remembered outcome + log n.
- `train.py`: per generation, evolution strategies on the net (12 mirrored pairs, block frozen, 3 train tracks), then one learning life for the block; grade = mean score on 8 held-out tracks (seeds 100-107), 600 ticks. Arms: `block` vs `plain` (same net, sensors only, no block). 40 gens x 3 seeds, 1 thread, 577 s. Raw: `car/results.json`, `car/run.log`.

## Results (held-out grade, final generation; 3 seeds, so only a rough read)
- Reference: hand PD controller 0.962; random net 0.039.
- plain: 0.629 / 0.656 / 0.712 (best-over-generations ~0.90 for all three), never off track.
- block: 0.824 / 0.128 / 0.757 (best ~0.89 / 0.24 / 0.92). Seed 1 failed: spent 66% off track, 13 specialists.
- Block thought time 12-17 us; n = 2-13 specialists.

## Reading and caveats
- No clear win for the block: two seeds beat plain at the end, one collapsed. n=3 and noisy ES; final-gen numbers swing a lot (plain best 0.90 vs final 0.63-0.71), so selecting/keeping the best theta was not done and would change the picture.
- Persistent block + moving policy is a non-stationary loop: the block learns from a policy that keeps changing, and the net's inputs shift when the block restructures (merge/prune). Seed 1 is consistent with that but the cause is unverified.
- Not done: more seeds, elitism/best-theta retention, block-ablation of individual inputs (prototype vs novelty vs remembered outcome), time-cost of thinking in the control loop, distractor/changing-drag worlds.
- Latency (block `d`) is reported but does not delay actions yet.

Related: [[wiki/neuroblock-project-state]], curious-net car env (`domains/curious-net`).
