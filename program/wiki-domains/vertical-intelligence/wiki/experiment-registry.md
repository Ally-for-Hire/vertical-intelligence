---
title: Experiment registry - every experiment of the vertical-intelligence program, by concept, with feasibility verdicts
domain: vertical-intelligence
kind: index
tags: [experiments, registry, feasibility, audit, families]
updated: 2026-11-07
---

# Experiment registry (one program, many experiment families)

Read this if: you want the list of everything that has been tried for vertical intelligence (VI), grouped by the concept it tests, and whether each test could even have worked.

Five-line summary: **every experiment is a vertical-intelligence (VI) experiment, filed by the family of VI concepts it tests** (author, 2026-11-07); there are no sub-projects. Each experiment below tests a VI concept or branches off one; folder names in paths are storage detail only. Counts per experiment are from the 2026-11-07 feasibility audit (`paper/FEASIBILITY_AUDIT_2026-11-07.md`, with per-row verdicts in `paper/feasibility/block{A,B,C,D}.md`): columns A = could pass or fail and informs, B = cannot fail, C = impossible by construction, D = noise-dominated or does not measure the claim, E/N = never run, withdrawn or conditional. Overall: 377 rows, 60% A, 16% B, 3% C, 14% D, 7% E/N; the goal-level questions (deep discovery from nothing, 'vertical' in X1.10-X1.17, graded doubt X7.7) were mostly not answerable by their design.

Where the pieces sit: concept sections and open questions in [[findings-map]]; planned tests in [[experiment-plan]]; run queue `paper/QUEUE.md`; predictions `paper/PREDICTIONS.md`; code `experiments/ and core/` (X-ids), `experiments/self-gate-and-calibration/experiments/` (older), `experiments/emergent-optimizers/emergent/` (X6.x) and `neuroblock-research/` (brain simulator).

Rule (author, 2026-11-07): new work is filed here and in [[findings-map]] under the family and concept it tests; the folder or script it lives in is not a category.

## 1. Value and the self; drives (X0.x, X1.1-X1.2, E9c)

Does value defined through the agent's own needs give need-dependent valence, safety, attention to every need, a hidden need only depth can satisfy? Code `core/x0_*.py`, `x1_1.py`, `x1_2.py`; findings-map section 0.

| Experiment | rows | A | B | C | D | N/A | One-line verdict |
|---|---|---|---|---|---|---|---|
| X0.1 | 6 | 4 | 2 | 0 | 0 | 0 | Safety/valence rows real (two failed honestly); the headline "for me" need-slope row cannot fail because the comparison arm's reward ignores state |
| X0.1b | 5 | 3 | 1 | 0 | 1 | 0 | Foresight question answerable and answered (no); need-slope row repeats the X0.1 tautology |
| X0.6 | 7 | 7 | 0 | 0 | 0 | 0 | Best-designed block: oracle-feasibility check, probes>=30, real failures. Answerable question; answer was mostly "no" |
| X0.7 | 7 | 6 | 1 | 0 | 0 | 0 | Real, two-world design with feasibility check; only the need-slope row is structural |
| X0.2 | 6 | 3 | 1 | 1 | 1 | 0 | The secure-base signature is fixed by world geometry (threat penalty 0.25 > every patch value 0.05); one baseline bar sits above what the tuned arm can reach |
| X1.1 | 6 | 4 | 2 | 0 | 0 | 0 | Dark room and noisy TV are true by reward definition; LP-vs-random slowness is a real finding |
| X1.2 | 5 | 4 | 1 | 0 | 0 | 0 | Registered as expected confirmations of a derived fix; could still fail, so A; NOVO row is tautological |
| X0.8 | 6 | 5 | 1 | 0 | 0 | 0 | Real predictions, three failed; but the design confounded unit with rate, so the headline question was NOT answerable (X0.9 admits this) |
| X0.9 | 5 | 3 | 1 | 0 | 1 | 0 | Diagnosis experiment; two rows near-expected by construction, one confounded, one fixed by rates |
| X0.10 | 9 | 3 | 4 | 0 | 2 | 0 | Headline "ALL finds depth / fills hidden need" passed by bars that RANDOM also clears; the tuned ALL-IGN gap is an IGN-handicap effect |
| E9c | 4 | 4 | 0 | 0 | 0 | 0 | Real, but a data repair of a toy (library contains Kepler form); informs about data, not VI |

## 2. Evolution and emergent optimizers (X6.6-X6.9, BENCH2)

Blind variation + culling versus gradient descent, energy as viability, stepping stones, growth. Code `experiments/emergent-optimizers/emergent/`. X6.x rows were classified from register text only.

| Experiment | rows | A | B | C | D | N/A | One-line verdict |
|---|---|---|---|---|---|---|---|
| X6.6 | 6 | 5 | 0 | 0 | 0 | 1 | Informative (null). Gene-movement check mis-targeted for the duplication claim |
| X6.7 | 6 | 2 | 2 | 1 | 0 | 1 | Literal world was infeasible (extinct 20/20); two rows are controls that cannot fail |
| X6.8 | 5 | 5 | 0 | 0 | 0 | 0 | Informative, real fails |
| X6.9 | 6 | 4 | 0 | 0 | 1 | 1 | Informative; "not the cheapest of four" row is noise-dominated |
| BENCH2 | 6 | 5 | 1 | 0 | 0 | 0 | Genuinely informative; two predictions FAILED (P283, P286), i.e. could fail. Best-designed block here. |

## 3. Prediction-only life and boredom (X1.3, X1.4)

An entity whose existence is prediction and internal models; how exact must boredom be. `experiments/boredom-and-prediction/x1_3.py`, `x1_4.py`.

| Experiment | rows | A | B | C | D | N/R | One-line verdict |
|---|---|---|---|---|---|---|---|
| X1.3 prediction-only life | 8 | 2 | 5 | 0 | 1 | 0 | Mostly reward-definition outcomes (IGN's reward excludes the only variable that distinguishes rooms); predictions were also written after seeing seed 97000 in the predicted direction. Only "ALL vs LP-family / chance" carries news. |
| X1.4 exactness of boredom | 7 | 4 | 1 | 2 | 0 | 0 | Headline (does noise/lag in boredom matter?) was unanswerable by the registered median metric: P289/P293 cannot pass (V). Unregistered means answered it. |

## 4. The Self, sensors and doubt (X7.x, X8.x)

A Self that audits networks from the inside, learns when not to answer, closed-form model checks. `vi/x7_*.py`, `x8_*.py`.

| Experiment | rows | A | B | C | D | N/R | One-line verdict |
|---|---|---|---|---|---|---|---|
| X7.1 Mirror | 6 | 0 | 0 | 0 | 0 | 6 | Never run (superseded). Nothing to score. |
| X7.2 Self audits sensors | 7 | 5 | 1 | 1 | 0 | 0 | Good: two real FAILs (P303, P305) changed the program. P305 was impossible-ish (no doubt signal ever trained). |
| X7.3 many sensors + life goal | 7 | 5 | 1 | 0 | 1 | 0 | Informative; P311 "learned distrust" passed for a different reason (measures learning the blackout, not doubt). |
| X7.4 can Self learn doubt | 6 | 5 | 1 | 0 | 0 | 0 | Informative; mostly FAILED because of memorisation (diagnosed). |
| X7.5 taming insides | 6 | 4 | 1 | 0 | 1 | 0 | Informative; P324 vacuous (conditional on a failed prediction). |
| X7.6 refresh | 6 | 5 | 1 | 0 | 0 | 0 | Feasible, but P327/P329 are close to guaranteed by X8.0/P338 arithmetic (refreshed labels => abstain); only held/novel generalisation was open. |
| X8.0 proof loop | 9 | 0 | 8 | 0 | 1 | 0 | Algebra-vs-Monte-Carlo checks, stated as circular. Caught 2 real algebra gaps (P335 renewal, P337 checker bug). Say nothing about learners. |
| X8.1 signature recogniser | 6 | 5 | 0 | 0 | 1 | 0 | Informative; P345 bar is borrowed from an unrelated world. |
| X8.1b create routines | 6 | 5 | 0 | 0 | 1 | 0 | Informative failures; P348 count bar satisfiable by merging everything. |
| X8.1c | 4 | 4 | 0 | 0 | 0 | 0 | Informative (over-creation); manipulation P353 honestly failed. |
| X8.1d | 4 | 4 | 0 | 0 | 0 | 0 | Informative; FRESH control finally added and shows library worse than no memory. |
| X8.1e | 3 | 3 | 0 | 0 | 0 | 0 | Informative. |
| X7.7 look-alikes | 6 | 3 | 0 | 3 | 0 | 0 | Design flawed: graded levels not readable (P367/P368 impossible, V in LOG CORRECTION), negative control geometry mismatch (P365 impossible). Whole experiment's headline (graded doubt) was unanswerable by design. |
| X7.7b matched control | 6 | 5 | 1 | 0 | 0 | 0 | Repairs X7.7; PENDING (no result file; log shows only "workers 5"). Feasible by design. |

## 5. Learning to learn: connection learners and the 'vertical' test (X1.6-X1.28b)

Meta-learned climb, connecting nets, trees, flat connection learner, digger/refiner. `experiments/learning-to-learn/x1_6.py` ... `x1_28b.py`. Block C's X1.8 (withdrawn) has no rows.

| Experiment | rows | A | B | C | D | N/R | One-line verdict |
|---|---|---|---|---|---|---|---|
| X1.6 meta-SGD climb | 5 | 3 | 1 | 0 | 1 | 0 | Real FAILs (P376, P377, P379). P378 trap passes by construction. |
| X1.7 connecting net | 7 | 4 | 1 | 0 | 2 | 0 | Core claim (P381, P387) feasible; P385 strawman arm; P382/P383 "pass" only after the effect-size filter was removed post hoc. |
| X1.9 | 8 | 5 | 2 | 0 | 1 | 0 | Solid: minimal-training arms are real contrasts; DENSE and TRAP wins are free; P398 is a fragile 0.012 effect |
| X1.10 | 7 | 4 | 2 | 0 | 1 | 0 | Real arm contrasts (tree lost to flat); vertical confounded by depth drift |
| X1.11 | 6 | 3 | 1 | 0 | 2 | 0 | Precision reward changes little (reward only reranks vs fixed cost); vertical confounded |
| X1.12 | 7 | 5 | 1 | 0 | 1 | 0 | Generalization test was a real new measurement; informative negatives |
| X1.13 | 6 | 4 | 1 | 0 | 1 | 0 | Distance reward feasible, failed; informative |
| X1.14 | 7 | 5 | 1 | 0 | 1 | 0 | Same |
| X1.15 | 6 | 3 | 1 | 0 | 2 | 0 | Tuned vs untuned default: a near-certain win, 4 rows are one measurement |
| X1.16 | 6 | 3 | 0 | 0 | 3 | 0 | Fair-vertical rows confounded by world drift; P447 duplicates P442 |
| X1.17 | 6 | 3 | 1 | 0 | 2 | 0 | World expanded during run; author voided; trend rows measured world growth |
| X1.18 | 6 | 5 | 1 | 0 | 0 | 0 | BEST DESIGN: finite world + no-memory control gives the first clean vertical (+0.039) |
| X1.19 | 4 | 4 | 0 | 0 | 0 | 0 | Feasible; only the loop-vs-no-memory row passed |
| X1.20 | 4 | 4 | 0 | 0 | 0 | 0 | Same |
| X1.21 | 4 | 4 | 0 | 0 | 0 | 0 | Same |
| X1.22 | 4 | 2 | 2 | 0 | 0 | 0 | Diagnostic; two rows foregone |
| X1.23 | 2 | 2 | 0 | 0 | 0 | 0 | Informative (depth gap +0.60) |
| X1.24 | 4 | 4 | 0 | 0 | 0 | 0 | Feasible, all null/negative: informative |
| X1.25 | 3 | 3 | 0 | 0 | 0 | 0 | Feasible; passes are +0.005 (trivial in size) |
| X1.26 | 4 | 2 | 0 | 0 | 2 | 0 | Diagnostics; P486 underpowered, P488 compares to another seed block's point value |
| X1.27 | 4 | 3 | 0 | 1 | 0 | 0 | P492 tests a mechanism that was a no-op |
| X1.28 | 6 | 4 | 0 | 1 | 1 | 0 | P497 at floor (0.0 everywhere); other rows feasible |
| X1.28b | 0 | - | - | - | - | - | Descriptive route batch, no verdicts (fine by design) |

## 6. Discovery from nothing: the deep-composition world (X1.29-X1.47)

The author's ultimate goal (2026-11-03): infinitely deepening discoveries from nothing via one simple rule. `experiments/deep-discovery/wrec.py`, `cell.py`, `x1_29.py` ... `x1_47.py`.

| Experiment | #P | A | B | C | D | E | One-line verdict |
|---|---|---|---|---|---|---|---|
| X1.29 huge compute (P499-502) | 4 | 3 | 1 | 0 | 0 | 0 | Real run, 12 worlds; "more compute helps" near-certain a priori, P501 (depth-1 recovery, 6/120) the only informative one |
| X1.30 simple goal (P503-505) | 3 | 1 | 2 | 0 | 0 | 0 | Frontier is informative; "readings differ / ranks disagree" cannot fail |
| X1.31 rule-making (P506-509) | 4 | 0 | 0 | 0 | 0 | 4 | Withdrawn, never evaluated |
| X1.32 whole-env exam (P510-513) | 4 | 0 | 1 | 0 | 0 | 3 | Never run (no result file); P513 is certain by design |
| X1.33 limit ladder (P514-517) | 4 | 0 | 1 | 0 | 0 | 3 | Never run; oracle_all falls with L by construction (60 labels vs 100+ concepts) |
| X1.33r one-rule ladder (P518-520) | 3 | 0 | 1 | 0 | 2 | 0 | Never run; floor effect at g=2.0 makes "share crosses 0.5" meaningless |
| X1.34 one rule from nothing (P521-524) | 4 | 0 | 0 | 0 | 0 | 4 | Withdrawn; its own pilot said P524 would hold at 60 labels (impossible) |
| X1.35 puzzle room (P525-528) | 4 | 2 | 1 | 0 | 1 | 0 | Honest, but "simplification" at g>=1.2 is mostly not-learning |
| X1.36 shared cell (P529-532) | 4 | 2 | 0 | 0 | 0 | 2 | Compute helps only at g=0.8; P530/P531 never evaluated |
| X1.37 labels (P533-536) | 4 | 3 | 0 | 0 | 1 | 0 | Clean negative: labels are not the limit (tanh cell) |
| X1.38 capacity (P537-539) | 3 | 3 | 0 | 0 | 0 | 0 | Clean but n=3 worlds, non-monotone in size |
| X1.39 sin vs tanh (P540-542) | 3 | 1 | 0 | 0 | 2 | 0 | P540 is the one real finding (family was the limit); gives the answer's family |
| X1.40 mixed cell (P543-545) | 3 | 0 | 0 | 0 | 3 | 0 | n=3, window (0.1) is below the noise; cell contains sin |
| X1.41 L1 chooser (P546-548) | 3 | 0 | 0 | 0 | 3 | 0 | n=3, thresholds are single 3-world medians |
| X1.42 depth (P549-551) | 3 | 0 | 0 | 0 | 2 | 1 | Stopped after ONE world with a row cap that changes results; the depth question is untested |
| X1.43 sin g=2 N=2000 (P552-553) | 2 | 2 | 0 | 0 | 0 | 0 | Informative negative (but n=3) |
| X1.44/44b (P554-557) | 4 | 1 | 0 | 0 | 3 | 0 | Bimodal outcomes + median of 3 = meaningless statistic; replication had ~40% power |
| X1.45 restarts (P558-559) | 2 | 1 | 0 | 0 | 1 | 0 | Existence claim fine; per-world disagreement n=3 |
| X1.46 choose family (P560-562) | 3 | 0 | 1 | 0 | 2 | 0 | Pilot picks sin because it is on the list and fits; 3 of 6 worlds are exact replays of X1.39 cells |
| X1.47 linear rung (P563-567) | 5 | 4 | 1 | 0 | 0 | 0 | Best-designed block: fair controls, clear answer; P567 tautological |

## 7. Older experiment families, not covered by the 2026-11-07 feasibility pass

Also VI experiments; their predictions (P1-P198, C1-C57, and the earlier simulator and emergent series) were audited for design on 2026-10-28 ([[audit-2026-10-28]]) but not row by row for feasibility.

| Family | VI concept it tests | Where (paths only) |
|---|---|---|
| Self-gate, oracle teacher, calibration and abstention (E1-E9, Q-series) | skepticism, calibrated knowledge, law derivation from data (THE DREAM benchmark Q-PRE1900) | `experiments/self-gate-and-calibration/experiments/`, `paper/LEDGER.md` |
| Context inference, sleep, motivation (C-series, CTX) | two timescales, importance, sleep (parked) | `LOG.md`; voids in [[audit-2026-10-28]] |
| Energy-limited brain simulator (energy, parts, Self, sleep, mastery) | the Self, energy-limited learning, foundational blocks | `experiments/energy-brain-simulator/FOUNDATIONS.md`, `FINDINGS.md` |
| Emergent optimizer series 1-3 | blind variation + culling vs gradient descent | `experiments/emergent-optimizers/emergent/`; benchmark repaired in BENCH2 |

Next step for these: run the same four-question feasibility gate (best arm clears the bar, no-skill arm does not, every arm sees the manipulated variable, n and bar give power) over the older registers.
