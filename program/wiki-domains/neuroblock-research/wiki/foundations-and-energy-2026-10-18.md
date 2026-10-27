---
title: Neuron-block foundations and the energy economy (2026-10-18)
domain: neuroblock-research
kind: project
tags: [foundations, energy, self, purpose, mastery, compute-limit]
updated: 2026-10-18
---

# Neuron-block foundations and the energy economy (2026-10-18)

## Summary

The user (Rose/Ally) stated the project's foundations over one session; full text, proposals (marked) and open questions are in
`~/neuroblock-research/FOUNDATIONS.md`. Energy results are in `ENERGY_FINDINGS.md`. Code: `lab/plugins/energy_brain.py`, `lean_brain.py`.

## Foundations (user's statements are givens)

* The brain converts the body's senses into abstract understanding and lowers uncertainty; it is split into at least five parts (emotional converter, intellectual converter, Self named);
  **building the Self is the main focus.** The end result is a general abstract problem-solver that improves itself without limit, efficiently.
* **Energy is foundational:** no action is free (every cost has a positive minimum; sensing and being alive cost energy; a shut-down brain still pays a tiny rest cost). Overload/empty pool = *forced shutdown*
  (not death); sustained high use lowers the overload threshold (fatigue).
* **Infinite growth should come from free solving:** what is known costs ~nothing; store little, reuse; each new problem should connect to all previous ones; understand each problem so well that repeating it costs nothing.
* Today's simulator is roughly an intellectual converter with a body; the Self and the emotional converter do not exist.

## Results worth keeping

* Prediction-error self-tuning of the matching radius (grade each candidate radius on predictions made *before* learning) rescues worlds where a fixed radius collapses: 32 inputs 15% -> 100% accuracy and 1,084 -> 8 neurons;
  noise 0.2: 83% -> 96%. A first version that graded radii on recently-learned experiences failed (every experience had just become a neuron at its own location, so every radius looked perfect).
* Natural economy (a move only if its expected saving beats its price) cut energy about 36x vs cost-blind squeezing at equal accuracy. Diversified income was good ~90% of the time; growth-only income only 26-37% ("stagnation is death").
* Automatic neurons (answering with a well-known neuron costs a small floor share): 30 -> 10 E per 1000 experiences at equal accuracy (3-run probe).
* Caution: ranking searches by energy rewards cheap *prices* (world physics) rather than strategy; hold prices fixed to study strategy.

## Gotchas

* The user's compute limit is **1 thread, low priority** (they game on this PC): enforced in `lab/limits.json`; do not raise it unprompted.
* PowerShell: `R` is an alias for `Invoke-History`; do not name helper functions `R`. `Start-Sleep` followed by more commands is blocked by the harness; use background commands.
* A patch script that asserts on every replacement and writes only if all succeed is safer than many separate edits, but one wrong anchor aborts the whole file: re-run only the failed file.

## Open

Cross-problem transfer (lifelong run), a problem frame for arbitrary problems, the emotional converter and Self v0, consolidation of neurons into rules, a meaning-of-life criterion (candidates in FOUNDATIONS.md).
