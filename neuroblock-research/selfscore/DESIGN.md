# Self-scored growing networks (user request 2026-10-26)

User spec (paraphrased): an algorithm scored only on itself, with external scoring acting as STIMULI. Use the problem suite and test a variety of methods on a
self-growing base neural network. It is scored in parallel generations with random changes and a HARD pass/fail. Primary goal: grow larger and more complex.
Secondary goal: grow optimized and correct in its output. Try combinations of those; try weaning it off external scoring. Full machine and GPU, no limits.

## Honest framing (read first)
- Information cannot come from nowhere. A network that is never told anything about the task cannot become correct on it. "External scoring as stimuli" is
  implemented as: the environment's feedback bit (was the action I took correct?) is an INPUT SIGNAL the network can predict, never an objective that selection sees.
- "Scored on itself" is implemented as three internal scores computed from the network and its own stimulus stream only:
  Q_pred = how well it predicted its own feedback stimuli (self-supervised, log-likelihood); C = effective complexity of its own functional structure (primary);
  later variants: self-consistency, compression.
- Correct OUTPUT behaviour is then a by-product: the network's outputs are its predictions of P(feedback=1 | input, action); it acts on the action it predicts
  will be rewarded. If the self-model is accurate, the action is correct. Nothing selects on task accuracy except the labelled reference rule `ext`.
- Weaning = progressively withholding fresh feedback stimuli. The network keeps scoring itself on its own stored memory (replay of past stimuli). Weaning removes
  NEW information, so we expect retention, not further improvement; this is tested, not assumed.

## Genome (the "base network")
Feed-forward DAG over nodes [inputs d | hidden H slots | outputs K]; masked weight matrix W (P,N,N), biases, alive flags for hidden slots; hidden = tanh, outputs = sigmoid.
Starts with NO hidden nodes and 2 random input edges per output. Mutations (random, 1-3 per child per generation): weight noise, add edge, remove edge, add node,
remove node, DUPLICATE node (gene duplication then drift; from the cell-handoff next steps). All populations are evaluated in one batched GPU pass.

## Generation loop (parallel)
P parents, each makes m children by mutation. Parents and children are evaluated on the SAME stimulus batch with the same exploration noise (paired). A child passes or
fails a HARD rule against its own parent. A parent slot is replaced by one passing child (picked randomly, or by largest C, or by best Q); with no passer the parent stays.

## Hard rules (pass criteria, child vs parent)
- ext: external accuracy of the greedy action improves (REFERENCE ONLY: uses labels as an objective).
- pred: Q_pred does not drop.                                         (quality only)
- grow: C increases.                                                  (growth only; control, expected to produce size without competence)
- grow_and_pred: C increases AND Q_pred does not drop.                (growth primary, quality hard second gate)
- nonreg: C and Q_pred both non-decreasing, at least one strictly up.
- pred_pickC: pass = Q_pred does not drop; among passers choose the one with the largest C.   (quality gate, growth chooses)
- grow_pickQ: pass = C increases; among passers choose the one with the best Q_pred.         (growth gate, quality chooses)

## Weaning schedules for the feedback stimulus: p_fb(g)
full (1 throughout) | linear (1 -> 0 over the first half) | step (1 until half, then 0) | none (0 from the start, control)

## Measures
Held-out ACCURACY of the greedy action (reporting only, never used for selection except by `ext`), effective complexity C, effective nodes/edges, pass rate,
correlation between internal score Q_pred and held-out accuracy across the population, accuracy retained/lost after weaning.

## Pre-registered predictions (written before any run)
P1  `grow` produces the largest C and chance-level accuracy (growth alone does not create competence).
P2  `pred` (self-prediction only, no labels in selection) reaches >= 90% of the `ext` reference accuracy on >= 70% of problems.
P3  `grow_and_pred` and `pred_pickC` end with >= 1.5x the effective complexity of `pred` while keeping accuracy within 5 points of `pred`.
P4  Weaning with replay (`linear`, `step`) retains accuracy within 3 points of the pre-weaning level; after weaning there is no further improvement.
P5  Without a quality gate after weaning (`grow` after step-weaning), accuracy decays toward chance (drift).
SUCCESS for the user's goal: a rule that uses no labels in selection (only Q_pred and C), including a weaning schedule, reaches >= 90% of `ext` accuracy on >= 70% of
problems AND has >= 1.5x the effective complexity of `pred`. Anything less is reported as a partial result with the numbers.

## Revision 1 (2026-10-26, user): hard 50% cull at the actual problem; individuals trained only on their own growth
Replaces the per-parent gate loop as the MAIN design (the per-parent rules in run.py are kept as the legacy mode; `runs/screen_*` are its screening results).
- INNER LIFE (parallel per individual, hard pass/fail): random structural change kept iff it passes the inner rule; primary = effective complexity grew
  (`grow`); optional second gate = no regression of the individual's own stimulus prediction (`grow_and_pred`); `none` = no inner growth (plain evolution control).
- OUTER CULL: every generation all individuals are scored on the actual problem; the bottom 50% are pruned (hard); survivors are multiplied and the copies slightly
  randomised (weight noise only). Individuals are never trained on the cull; accuracy is not an objective of any individual.
- Weaning: the cull ranks by an internal score (Q = self-prediction of own stimuli, or C = complexity) with probability lam(g) rising 0 -> 1 (linear / step / all).
- Files: selfscore/cull.py (snapshot of the first version: cull_v1_snapshot.py).

## Revision 2 (2026-10-26, user): base model switched from deterministic networks to a base STOCHASTIC model of neural networks
Interpretation (an assumption, to be confirmed by the user): stochastic binary neurons. Hidden units SAMPLE a state in {-1,+1} with P(+1) = sigmoid(beta * input);
beta is an evolvable per-genome temperature gene (the network can evolve how random it is); a prediction is the mean of S stochastic passes (default S=4); the
single-pass accuracy is reported alongside. Deterministic mode kept as `--stoch` off (snapshot pop_v1_snapshot.py). Structure, mutations, complexity measure unchanged.
First check: parity4 after 50 generations: mean-of-4 accuracy .677, single pass .634, mean beta 1.00 -> 1.52 (networks became less random).
