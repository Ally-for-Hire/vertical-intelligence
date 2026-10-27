---
title: Neuron-block research: project state
domain: neuroblock-research
kind: project
tags: [neural-architecture, simulation, numpy, research, ifs, art]
updated: 2026-10-18
---

# Neuron-block research: project state

## Summary

A 2D-graph sandbox where specialist "neurons" spawn on unexplained input (the expansion instinct), signal latency is
`edge length x 40 us`, and the network condenses itself (move, merge, prune). Imported on 2026-10-18 from a handoff zip made in
another chat; the original handoff is in `raw/2026-10-18-handoff-readme.md`. Project folder (local git repo, own `CLAUDE.md`):
`~/neuroblock-research`.

## Reproduced results (2026-10-18, Windows, Python 3.11.9, numpy 2.4.1, 8 seeds, toy Gaussian clusters)

- `experiments3.py` (7 s) matches the handoff: no condensing ~94 us mean thought; moving nodes ~31 us (mean edge 1.0 -> 0.4);
  merge+prune 27 -> 11 neurons with accuracy 98.3% -> 99.5%.
- Expansion drive is inert under merge/prune: ~25 accepted expansions per run with merge+prune but net neuron count barely
  moves (14.5 vs 24.5 without expand); without merge/prune ~5 accepted. Cause: merge/prune remove the new neurons and the
  same ones get re-proposed.
- Handoff-only (not re-run here): fastest-match firing 84% vs best-fit 97% on the hard task for ~7 us more; gentle hardening
  (c0 ~30-300) is the better balance than early hard hardening (c0=3 grew 38 vs 21 neurons).

## Caveats

- Differences under ~2 accuracy points are noise at 8 seeds. The ~30 us floor is partly the `min_sep=0.3` parameter. The
  40 us/unit and the task are assumptions. Why hardening slows thoughts is unverified.
- `python -I` breaks the scripts (they import siblings from their own folder). `experiments2.py` overwrites `results2.json`.

## Next steps (from the handoff)

1. Harder task where existing categories cannot explain input, plus a grace period protecting newborn neurons from merge/prune.
2. General graph with shortcut edges and battery (delayed-fire) nodes.
3. Self-style coordinator and a "negative for the whole" check on spawning.
4. Sensor-trust decay and a self-as-biased-data loop (predict own next error).
5. Logging of births, deaths, vigilance, reward curves so runs stay comparable.
