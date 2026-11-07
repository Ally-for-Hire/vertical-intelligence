---
title: The author's request - train a network on its ability to increase complexity
date: 2026-11-01
kind: raw (verbatim user words, append-only)
---

> claude. we are on the frontier of neural networking evolution. we need to train a neural network on it's ability to increase complexity. we must do it. figure it out. this is related to vertical learning friend

## Assistant reading (not the author's words)

- "Train on its ability", not "train on complexity": the object being trained is the network's LEARNING MACHINERY (how it learns), and
  the signal is how fast it raises the complexity it holds. That is a meta-learning outer objective, not a new loss on one task.
- "Complexity" must be the structure the network holds ABOUT THE WORLD (bits on unseen points), not raw complexity of the network or its
  outputs: raw complexity is maximised by memorising noise (the trap; tested as RAWCX). Ties: U12 (growth, not without limit), U9
  (boredom), I15/I16 (depth ladder, bits of real structure), X1.4 (an exact structure signal alone does the work for choosing what to
  study; X1.6 asks whether it also works for shaping the learner's weights).
- "Vertical": the definition proposed in the protocol (each new problem solved faster by connecting to earlier ones) makes the test
  rungs NEVER trained on (extrapolation up the ladder).
- First test: X1.6 (curious-net `vi/x1_6.py`, P376-P380).
