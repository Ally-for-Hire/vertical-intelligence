# Variety window hypothesis (user, 2026-10-26) — pre-registered before any run

User's hypothesis (paraphrased): evolution requires SOME amount of variety; beyond that, life (emergent complexity) innately emerges. Emergence should appear within a
band of variety, and fail outside it, either because variety was not reduced enough or because it was reduced too much. The user is most interested in the UPPER limit
(the maximum variety at which emergence still works).

## Operationalisation
* Variety V is MEASURED, four ways (the hypothesis must hold on at least two independent measures):
  N_lin   effective number of surviving lineages (inverse Simpson index of founder-lineage frequencies, 1..P)
  N_func  number of distinct input->output functions in the population (phenotypic variety)
  D_gen   mean pairwise genotype distance between individuals (normalised weight/structure distance)
  sd_z    standard deviation of accuracy across the population
* Variety is CONTROLLED by two knobs: M = number of distinct survivors kept each generation (cull = 1 - M/P, from 1 = single winner to P = no culling) and
  eps = size of the random change applied to each copy ("slight randomisation").
* Emergence is FUNCTIONAL, not mere size: a task is solved that requires hidden units. Tasks: xor2, parity3, xor2 with 4 distractor bits (no-hidden-unit ceiling 0.75 on all;
  solved = held-out accuracy >= 0.95 by the best individual). Success probability E(V) = fraction of seeds that solve it within G generations; also time to solve.
  (Effective complexity C alone is not emergence: the inner growth drive raises it regardless.)

## Predictions (all must hold for the hypothesis to stand)
H1  LOWER LIMIT: E(V) is near 0 below some V_lo (too little variety: stagnation, premature convergence).
H2  UPPER LIMIT: E(V) is near 0 above some V_hi (too much variety: selection cannot accumulate).
H3  PLATEAU: for V_lo <= V <= V_hi, E(V) varies by less than 15 percentage points (flat band, 'innately emerges'), and each edge goes from 10% to 90% of the plateau
    within one decade of V (sharp thresholds).
## What falsifies it
* E(V) monotone in V (no window); or smooth single-peaked rather than a plateau (H3 fails); or no upper cutoff within the largest variety reachable (H2 untestable -> reported as such);
* the window appearing for one measure of V but not for others (then 'variety' is not a single quantity).
## Reporting rule
Whatever happens is reported with the numbers; the grid is fixed here: P=128 individuals, M in {1,3,8,26,64,128}, eps in {0.01,0.1,0.5}, 3 seeds, 300 generations,
tasks xor2 / parity3 / xor2_d4, inner rule `grow`, model v2 (any-direction growth). A pilot on one cell only checks that the pipeline works (no tuning of the grid after seeing results).
