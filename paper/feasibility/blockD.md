# Audit block D: PREDICTIONS.md X1.29 to X1.47 (P499-P567, 69 rows)

Method: read PREDICTIONS.md rows, LOG.md results, HANDOFF.md s3/s5, vi/wrec.py, vi/cell.py, vi/x1_35.py, vi/x1_39.py, wstream.py. Numeric checks (verified, in `audit/_chk*.py`) use the exact `wrec.py` rule. Everything marked (inf) is inference. Register PASSED/FAILED labels were re-checked against LOG.md; two register statuses are weaker than they look (P561, P554, see notes).

Verdicts: A feasible-informative, B tautological, C impossible-by-construction, D underpowered/untestable, E unknown (never run or withdrawn).

## 1. Per-experiment summary

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
| **Total** | **69** | **23** | **9** | **0** | **20** | **17** | |

C is 0 only because most "impossible" targets were registered as predictions of failure (P518, P524, P528b) or as descriptive existence claims. The programme's headline targets (exact deep recovery at g=2.0 with 60 labels, depth to 20+ at corr^2>=0.99) are effectively C; see section 4.

## 2. Per-prediction table

| P | V | Reason |
|---|---|---|
| P499 | A | 9/10 arms up (+0.03..+0.10); MDL_mix flat so it can fail; sign near-certain a priori |
| P500 | B | 'auto' keeps the better of two copies by validation error, so >= each by selection; only GROW cleared zero |
| P501 | A | Existence of depth-1 recovery; 6 of 120 arm-worlds (3 of 12 worlds), none depth 2+; interim "disproof" was wrongly called then corrected |
| P502 | A | FLAT_BIG - FLAT = +0.018 [+0.003,+0.029]; tiny but real; sign near-certain |
| P503 | A | 63 weights at R^2 0.50; threshold 0.5 and 150 weights arbitrary but frontier is informative |
| P504 | B | Pareto front over ~20 arms spanning 2 readings is near-certain (any two readings share FLAT/REUSE/MDL points) |
| P505 | B | Rankings by weights/functions/links/seconds disagreeing "somewhere" cannot realistically fail |
| P506-P509 | E | X1.31 withdrawn by author 2026-11-03; never evaluated |
| P510 | E | X1.32 never run (no results file); feasible in principle |
| P511 | E | Never run; X1.29 saw 0/120 recoveries at depth 2+ in same generic world, so depth 2+ clause likely near-impossible (inf) |
| P512 | E | Never run; "links fall while accuracy does not fall in every arm" is a conjunction of 5+ arms (inf: fragile) |
| P513 | B | "Best arm leaves part of 24-concept world unexplained": certain with 60 labels and depth to 5 |
| P514 | E | Never run (x1_33 cells absent); "climb until no gain" stop rule on 4 worlds is noise-prone |
| P515 | E | Never run; net limit grows with world: depends on noisy stop points |
| P516 | E | Never run |
| P517 | B | oracle_all (60 labels, L=96..384 true concepts) falls with L by linear algebra; deepest recovered << max depth known from X1.29 (depth 1) |
| P518 | B | reach << L: even level 1 at g=2 failed from the exact parent at 60 labels (X1.34 pilot, 0.04); predicts a certain failure |
| P519 | D | One-rule world at g=2 vs multi-rule world differ in many things besides rule count; confounded |
| P520 | D | At L=12 g=2.0 exam is already 0.07-0.15 (X1.35); "falls below 0.5 of smallest-world value" is floor noise |
| P521 | E | X1.34 withdrawn |
| P522 | E | Withdrawn |
| P523 | E | Withdrawn; "steeper reaches less" near-certain |
| P524 | E | Withdrawn; its own pilot says it holds by impossibility at 60 labels |
| P525 | A | ALL > NONE every room at g 0.8,1.2 (smallest low end +0.003); NONE collapses at depth, so partly measures NONE's overfit |
| P526 | D | Withdrawn: first 10 vs last 10 differ in tier difficulty (register admits it) |
| P527 | A | g=0.8: links 7-8 -> ~2; weights/level -74 [-129,+63] (null, not shrink). At g>=1.2 links fall because nothing is learned |
| P528 | B | Exam falls with depth in every arm is certain (fewer problems per level); second clause (g=2 no recovery past level 1) holds by identifiability at 60 labels |
| P529 | A | g=0.8 up (+0.11..+0.18, decided); g=1.2 not shown; informative split |
| P530 | E | LOG says "not evaluated here"; weights are lower (187-485 vs ~850/level) but exam difference not reported |
| P531 | E | Not evaluated explicitly; only n90 2/3/6/13 at refit 600 reported |
| P532 | A | FAILED: no exact deep concept at any rung (tanh cell); g=2 clause passes (exam 0.02-0.11) |
| P533 | A | tanh cell, N to 2000: exam falls at large N; 0 exact. Informative negative: labels are not the limit (cell family is) |
| P534 | A | Same at g=1.2 (saturated N=600) |
| P535 | A | g=2.0, N to 6000: corr^2 stays 0.06-0.08. Negative informative but tanh cell likely cannot reach 0.99 through learned-parent amplification (inf) |
| P536 | D | "Monotone in N at every g" = ~12 consecutive increments over 3 noisy worlds; cannot realistically hold |
| P537 | A | Bigger cell lower at g=0.8; informative (non-monotone capacity) |
| P538 | A | No exact deep concept at any capacity: shows family, not capacity, binds |
| P539 | A | Same at g=1.2 |
| P540 | A | sin cell: 2/3 worlds exact at g=1.2 hid 24; real finding that family was limiting. Note: hands over the rule's family |
| P541 | D | Medians of 3 worlds per cell; per-world exam spread 0.69-0.99 (SD ~0.16, SE of 3-mean ~0.09) |
| P542 | D | 0.26/0.21 vs a 0.2 bar chosen after tanh results; 3 worlds, bimodal; passes by 0.01 |
| P543 | D | 1 of 3 worlds; cannot distinguish from chance; MIX includes sin units, so "hint not required" is not tested (sin still present) |
| P544 | D | 0.698 vs 0.708 on 3 worlds: noise |
| P545 | D | A 0.1 window is about 1 SE of a 3-world mean (0.09): cannot be passed or failed reliably |
| P546 | D | Compares one 3-world mean (0.669) to 0.698; noise |
| P547 | D | 0/3 worlds; the sine reference 2/3 is itself within noise of 1/3 |
| P548 | D | Same window problem as P545 |
| P549 | D | Stopped after one world; row cap changes results (world 960001: 0.989 -> 0.668 capped) |
| P550 | E | No tanh arm run |
| P551 | D | n90 stayed 5,5,5,5 in one world; one world and capped refit cannot decide scaling |
| P552 | A | N=2000 sin g=2: corr^2 fell 0.21 -> 0.073; informative (3 worlds) |
| P553 | A | 0 of 3; consistent with the later 2-in-9 lottery |
| P554 | D | Median of 3 on a bimodal outcome (0.99, 0.10, 0.03): median ignores the one world that landed |
| P555 | A | 1 of 3 exact; first at g=2.0; sets up the lottery question |
| P556 | D | Power: if true rate is 2/9, P(>=2 of 6) = 0.40; a coin flip, not a replication test |
| P557 | D | 9-world median vs 3-world default median on bimodal data; incomparable |
| P558 | A | 1/12 restarts reach 0.978; informative that failed worlds are not intrinsically unlearnable |
| P559 | D | Per-world disagreement judged on 3 worlds x 4 restarts, mixed outcome (1 of 3) |
| P560 | B | sin is on the 3-item list and is the generating family; it beats tanh/relu on the first 10 problems (0.99 vs 0.92-0.95) by construction |
| P561 | D | Worlds 960000-960002 reproduce X1.39 exactly (same seeds, same cell); only 3 new worlds; threshold equals X1.39's count |
| P562 | D | Window 0.1 vs spread 0.65-0.99; NO driven by the 3 new worlds |
| P563 | A | CONV 0.55 in range: informative failure of a 4000-step tanh net on pure linear |
| P564 | A | 0.00 out of range for all tanh arms; expected by saturation but a real bar to pass |
| P565 | A | LINNET 1.00/1.00 and LSQ 1.00: a positive control; passes only if optimiser converges; attribution is informative |
| P566 | A | META10 0.85 in range (prediction <0.5 FAILED); meta helps +3.12 bits |
| P567 | B | Product of d linear layers is one matrix with outputs normalised; depth cannot matter: +0.00 [-0.13,+0.00] |
| M-X1.47 | B | LSQ masters all: sanity control, correct to include |

## 3. Recurring design flaws

1. **Prediction-of-failure / certain-direction rows (B).** The register frequently predicts what cannot fail: "readings differ" (P504, P505), "even the best leaves part unexplained" (P513), "exam falls with depth" (P528), "oracle_all falls with L" (P517), "reach << L" (P518), depth-irrelevance of a linear stack (P567). Passing them tells nothing.
2. **n=3 worlds with bimodal outcomes (D).** X1.38-X1.46 run 3 worlds per cell; outcomes are bimodal (exam 0.11-0.48 or 0.95-0.98 at g=2; 0.69/0.96/0.99 at g=1.2). Medians, 0.1 windows and "at least k of 3" are statistically empty: P541-P549, P554, P556, P557, P559, P562. The log says "ranges not computed" repeatedly.
3. **Windows below the noise (D).** SE of a 3-world exam mean at g=1.2 is about 0.09, yet P545/P548/P562 test "within 0.1". P556 has about 40% power under its own favourable assumption.
4. **Bar above or at the learner-family optimum (C-ish).** corr^2>=0.99 is the exactness bar for a cell with ONE parent, but the true rule also ties each concept to the mean of all older concepts (0.1 term). The one-parent cell cannot express it; the best chain-only unit reaches 0.961 (g=2, level 2) and 0.986 (g=1.2, level 2) (verified). Observed 0.99s at level 2 therefore come from multi-unit fits, but the bar is at the edge of what the architecture can do at g>=1.2 and error compounds (section 4).
5. **Family and structure hard-wired into the learner (vacuous "discovery from nothing").**
   - `cell.py` shares one theta across rotations `np.roll(W1x, -s)` with `m=8`: this IS the world's `x[n mod 8]` rule (verified in `fwd_np`); parent topology is one parent, matching `wrec._raw`.
   - Room 1 curriculum (k=1) presents ten problems of level 0, then ten of level 1, ... in dependency order, so the parent is always already in the library (x1_35.room_specs).
   - The world gives each problem a single exact concept (`S=[level]`, a=[1.0]), no mixing; labels noise-free.
   - Activations: sin is the rule's family (P540, P552-P557); X1.40/X1.41/X1.46 candidate lists contain sin (P543-P548, P560-P562); P560 picks sin because it fits best.
   - Result: the programme measures "can a net shaped like the rule fit the rule", not discovery from nothing. The author's own logs flag the sine caveat, but not the rotation/parent/curriculum hints.
6. **Depth never tested.** Cell experiments X1.36-X1.46 use `Lv=5` (levels 0-4); only X1.42 goes further, one world, capped refit, and shows no recovery past level 4 (corr^2>=0.9 stays at 5 levels in rooms 1-4). The goal ("infinitely deepening") has 0 valid tests; all "deep" claims are depth 1-4.
7. **Metric confusion in "simplification".** At g>=1.2 fewer links per problem and higher exam than NONE comes from linking nothing (exam 0.02-0.15): P525/P527 pass for g=0.8 only in substance (LOG X1.35 caveat).
8. **Never-run or withdrawn rows counted in the register** (17 of 69): X1.31, X1.32, X1.33, X1.33r, X1.34 and half of X1.42. These inflate the apparent size of the programme but contain no evidence.
9. **Post-hoc and moving thresholds.** P542 bar 0.2 was set after tanh stayed at 0.06-0.08; P554 uses 0.21 from a different experiment's median; P557 compares to a different-sample median. P561's threshold equals the replayed 2-of-3 count.
10. **Run-to-run confound with caps.** `refit_rows` cap changes some worlds a lot (0.989 -> 0.668), so X1.42 compared to X1.39 is not like-for-like (P549-P551).

## 4. Identifiability of the deep world (verified numerics, rule from `vi/wrec.py`, dim 8, noise-free)

**Complexity of concept n as a function of x (g = 0.8 / 1.2 / 2.0).**
- Turning points along one axis line over [-3,3], depth 0..4: g=2.0: 4, 11, 23.6, 46.5, 87; g=1.2: 2, 4.7, 7.5, 10.2, 13.7; g=0.8: 2, 3.1, 3.7, 3.6, 3.6 (saturates ~3.5).
- RMS gradient norm by depth: g=2.0 doubles per level (2.1, 4.6, 9.4, 19, 38, 77, 154, 312); g=1.2 grows x1.3 per level (1.3 -> 13.7 at depth 9); g=0.8 flat (1.1 -> 1.8).
- Nonparametric (tanh-net) recovery of level 4 at g=2 would need >=2 samples per half-wave on each of the ~5-8 active axes: about (2 x 87)^k points, k>=3, far beyond the register's 60-6000 labels. Level 0-1 at g=2 (4-11 turns/axis) is borderline at 60 labels (the X1.34 pilot found even the exact parent gives R^2 0.04 at 60 labels, ~300 labels needed for one step).
- So at g=2.0, "recover deep concepts at corr^2>=0.99 from labels" is impossible without the sine/parent structure (C, effectively). With the rule family and an exact parent, one step has ~10 parameters and is identifiable from a few hundred labels; the obstacle then is the periodic loss landscape (optimisation), which is what the 2-in-9 lottery (X1.44b, X1.45 1-in-12) shows. (inf)
- g=0.8 is nearly depth-flat (3.6 turns/axis from depth 2 on): nonparametrically learnable at any depth, so it tests nothing about depth. g=1.2 is borderline (13.7 turns at depth 4, 27 at depth 9).

**Error amplification (parent corr^2 -> child corr^2 when the child is built from an imperfect parent).**
- Parent 0.99 -> child 0.993 / 0.986 / 0.960 for g = 0.8 / 1.2 / 2.0.
- Parent 0.95 -> child 0.967 / 0.932 / 0.820.
- Per level the world amplifies error by about g (derived from gradient growth: x2 per level at g=2). g=0.8 is contractive (errors die out: infinite depth feasible), g=1.2 is mildly expansive, g=2.0 doubles: to hit 0.99 at level n via a learned chain, the level-0 error must be about 0.1/2^n (n=4: corr^2 ~ 0.9999). So "a chain of learned concepts at corr^2>=0.99 to infinite depth" is impossible at g=2 for any finite-precision learner without end-to-end correction at each level, and only the contractive regime (g<1) is infinitely extendable. (inf for the end-to-end-correction clause; the numbers are verified)

**Architecture ceiling for the bar 0.99.**
- Chain-only sine unit with exact parent: corr^2 with true concept at levels 1..4: g=0.8: 1.00/0.994/0.995/0.996; g=1.2: 1.00/0.986/0.990/0.991; g=2.0: 1.00/0.961/0.979/0.985. Free-weight nonlinear fit from true init: g=1.2: 0.9956/0.9962/0.9963; g=2.0: 0.9663/0.9804/0.9856 (levels 2-4; a local fit, so a lower bound on the single-unit optimum).
- So with one parent and one sine unit, the 0.99 bar is above the ceiling at g=2 for levels 2-4 and at the margin at g=1.2. The observed 0.99 at level 2 for hid 24 comes from extra units. This is why exact deep recovery is rare and lottery-like at g=2 and why P549-P551 could not show "more depth".

**Could the VI ultimate goal be tested in this world?** No.
1. Depth > 4 never run in the cell (Lv=5), single-world X1.42 shows no growth.
2. g=0.8 is flat in complexity, so recovery there is not "deep"; g=2.0 is not identifiable at the register's label budgets without both family and exact parents.
3. The learner is given the rule's shape (cyclic variable rotation, one-parent chain, ordered curriculum, sine units or a sine-containing list), so success does not mean discovery from nothing; and failure at g=2.0 does not mean the idea fails (bar above optimum / amplification).
4. A valid test needs: a contractive world (g around 1), per-level ties the cell can represent, a family-free cell without ordered curriculum, depth >= 20 with replication (>= 30 worlds), and a bar relative to the architecture ceiling (e.g. 0.95 of the oracle-chain corr^2 rather than an absolute 0.99).

**Answers to the three questions.**
(1) At g=2.0 recovering deep concepts at corr^2>=0.99 is not identifiable nonparametrically at 60-6000 labels (87 turns/axis by depth 4; gradient x2/level) and compounds through learned parents (child 0.96 from a 0.99 parent). The "exam" bar at deep levels is unreachable except through exact sine structure.
(2) "Never recovered a concept built on another" (X1.33 ladder) is not a result: the ladder was never run. In X1.29 it was observed at depth 1 only (6/120) in the generic world, a consequence of 60 labels versus mixed 1-6 concept targets plus the 0.99 bar. In the deep world, the exactness failure for tanh cells is partly an architecture-ceiling and error-amplification effect, so it is a design consequence (also optimisation lottery at g=2), not evidence against the learner idea.
(3) Yes, the sin-containing cell (X1.39-X1.45) and sin-containing candidate lists (X1.40, X1.41, X1.46) make "discovery from nothing" vacuous; the rotation-tied cell, the one-parent topology and the level-ordered curriculum add further hints. The only unvacuous parts are X1.47 (linear base rung) and X1.37/X1.38 (labels/capacity do not help the tanh cell).
