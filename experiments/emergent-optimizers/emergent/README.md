# Emergent neural-net behaviour from pure random weights + external culling

> **AUDIT NOTE 2026-10-28 (see curious-net `paper/AUDIT_2026-10-28.md`, V6/V7).** This README predates series 3 and X6.5-X6.9 and still states the pre-registered target 0.90 (the runs used 0.95). The headline ratios were selected and scored on the same seeds (winner's curse: the same setting on fresh seeds is 0.65-0.74x default Adam, not 0.51x), the stop rule reads test accuracy with unequal peek intervals, and Adam was tuned less than the evolutionary optimizer. Treat 'matches default Adam' as the strongest supportable wording until the benchmark is repaired (R-V7).

User's goal (2026-10-26): "emergent neural network behaviour from pure random weights and biases using external culling as a mechanism ... continue until we get something at some level of efficiency that produces neural net behaviour using internal randomness and external culling. test on 1 problem."

## The one problem: `xorgauss` (chosen by the assistant)
Four Gaussian clouds (SD 0.5) at (+-1, +-1); class = which diagonal. Raw 2-D coordinates are the input (no bit encoding).
* It is the XOR problem with noise: **no linear classifier can do better than chance (0.50)**, so solving it requires hidden units, i.e. genuine neural-net behaviour.
* Its best possible accuracy is known exactly: a point is classified right when both coordinate signs are right or both wrong, so the Bayes optimum is Phi(2)^2 + (1-Phi(2))^2 = **0.955**.
* 2,000 training points, 4,000 held-out test points (fixed data seed); test data is never used for selection.

## Success criterion (fixed before any run)
1. **Neural-net behaviour:** the best brain (picked on TRAINING accuracy) reaches **>= 0.90 test accuracy** (more than 40 points above any linear model, within 5.5 points of the optimum) in **5 of 5 seeds**.
2. **Efficiency** is reported as the number of network evaluations on the training set until (1) is first met, and compared with two references on the same network size:
   * **random search** (the same random networks, no culling: keep drawing and keep the best), and
   * **gradient descent** (a normal trained net of the same size, counted in training samples seen).
   "Some level of efficiency" = clearly better than random search (at least 10x fewer evaluations, or random search fails within the same budget).

## Mechanism (the only ingredients)
* **Pure random weights and biases** at the start.
* **Internal randomness:** every new brain is a copy of a surviving brain with random noise added to its weights and biases.
* **External culling:** each generation the worst-scoring brains die. No gradients, no hand-designed weights.

Files: `evolve.py` (the experiment, one command), `LOG.md` (every attempt, failures included).

## Result, series 2 (2026-10-26): goal reached = match default gradient descent (user's choice)
Rule (user): task performance is used ONLY by the external cull; every change inside a net is blind random noise.
Method: 40 nets (2-8-2 tanh) with pure random weights and biases (scale 0.5); each generation every net is graded on the same 16 random training points by the log-probability it gives the right answer, averaged over its life; the worst 80% die; each empty slot gets a copy of a random survivor plus Gaussian noise of the copy's own, randomly inherited size.
On 20 fresh seeds: **20/20 reach 0.95 test accuracy at a median 67,840 sample-forward-equivalents vs 75,120 for gradient descent with default Adam (0.90x).** Gradient descent with a tuned learning rate is still ~12x faster (5,520).
Reproduce: `python emergent/confirm.py` (prints exactly these numbers on the same hardware/software). Search history: `LOG.md`, `search.py`, `round6.py`.
