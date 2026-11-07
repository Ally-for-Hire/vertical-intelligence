---
title: The author's request - a linear base rung before the complexity ladder climbs
date: 2026-11-04
kind: raw (verbatim user words, append-only)
---

> i want to continue from "neural network complexity training" i mean each output is a linear function, even underlying it should be linear, and just see if the eural network can map the entire thing before increasing in complexity

## Assistant reading (not the author's words)

- "Neural network complexity training" = X1.6 (I20, curious-net `vi/x1_6.py`), whose ladder started at rung 1 (one bend).
- "Each output is a linear function, even underlying it should be linear" = a multi-output task y = A x where the hidden generator
  is itself a stack of linear layers, so it is linear at every layer, not only at the output.
- "Map the entire thing" = learn the whole rule, scored in range AND out of range (a learner holding the line, not the cloud of
  training points, must pass both). "Before increasing in complexity" = a mastery gate for the ladder.
- Test: X1.47 (curious-net `vi/x1_47.py`, P563-P567).
