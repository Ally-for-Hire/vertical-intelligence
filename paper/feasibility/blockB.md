# Audit block B: PREDICTIONS.md lines 410-700 (X1.3, X1.4, BENCH2, X7.1-X7.7b, X8.0-X8.1e, X1.6, X1.7 to P387)

Scope notes. The file rows in this range run X1.3 -> BENCH2 -> X1.4 -> X7.1..X7.6 -> X8.0 -> X8.1/b/c/d/e -> X7.7 -> X7.7b -> X1.6 -> X1.7 (P381-P387), 115 rows incl. manipulation checks (M-rows). X1.5 has NO rows in PREDICTIONS.md (nothing to audit). M-X1.6 / M-X1.7 are described only in registration notes (not table rows) and are not counted. Labels: V = verified by reading code/result files or recomputing; I = inference. I did not run experiments; short numpy recomputes from saved results only.

Verdict key: A feasible-informative, B tautological, C impossible-by-construction, D underpowered/untestable (bar or metric does not measure the claim), N/R never run (X7.1 VOID-unrun).

## 1. Per-experiment summary

| Experiment | rows | A | B | C | D | N/R | One-line verdict |
|---|---|---|---|---|---|---|---|
| X1.3 prediction-only life | 8 | 2 | 5 | 0 | 1 | 0 | Mostly reward-definition outcomes (IGN's reward excludes the only variable that distinguishes rooms); predictions were also written after seeing seed 97000 in the predicted direction. Only "ALL vs LP-family / chance" carries news. |
| BENCH2 | 6 | 5 | 1 | 0 | 0 | 0 | Genuinely informative; two predictions FAILED (P283, P286), i.e. could fail. Best-designed block here. |
| X1.4 exactness of boredom | 7 | 4 | 1 | 2 | 0 | 0 | Headline (does noise/lag in boredom matter?) was unanswerable by the registered median metric: P289/P293 cannot pass (V). Unregistered means answered it. |
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
| X1.6 meta-SGD climb | 5 | 3 | 1 | 0 | 1 | 0 | Real FAILs (P376, P377, P379). P378 trap passes by construction. |
| X1.7 connecting net | 7 | 4 | 1 | 0 | 2 | 0 | Core claim (P381, P387) feasible; P385 strawman arm; P382/P383 "pass" only after the effect-size filter was removed post hoc. |
| TOTAL | 115 | 71 | 23 | 6 | 9 | 6 | |

E (unknown): 0.

## 2. Per-prediction table

| P# | V | Reason (<=25 words) |
|---|---|---|
| M-X1.3 | B | Checks coded mechanics: dark room gives H no refill, noise has no accuracy; ORACLE hand-written to pass. Sanity only. (V: x1_3.py) |
| P275 | B | IGN reward = dV only, blind to H, which is refilled only by learning; dark room maximises dV, so boredom death follows by definition. Author admits "by design". |
| P276 | B | IGN cannot see H; dark room has accuracy 1 so V never dies; only H can kill. 100% boredom deaths is arithmetic. |
| P277 | B | ALL's reward contains dH and H rises only through learning; ALL-IGN in H (0.99 vs 0.26) is fixed by the reward definition. |
| P278 | A | ALL vs random could fail, but seed-97000 look showed ALL 0.0 vs 21.7 before registration (pre-seen direction). |
| P279 | A | UR vs LP: could go either way; effect is 1-2 deaths in 6000 steps, resolution 1.67 (median quantised). Note UR ~ affine of IGN reward (I). |
| P280 | D | Bar "upper end < 10 deaths per 10k" vs a field where every learned arm is <= 8.3 and chance 21.7; passes for nearly any learned arm. |
| P281 | B | Same cause as P275: IGN's reward prefers the room with max accuracy (dark); ALL pays for learning. Share 91% vs 31% by design. |
| M-BENCH2 | B | selftest checks determinism and seed disjointness; these hold by code. Valid hygiene, not a finding. |
| P282 | A | Tuned evolution vs tuned Adam; could have gone either way; ratio 12.5 over 6 problems. |
| P283 | A | FAILED: evolution still faster than default Adam (0.45). Informative (Adam default badly set). |
| P284 | A | Winner's-curse >=10%: measured 1.48x, could have been 1.0x. |
| P286 | A | Old-rule flattery <=0.87: FAILED at 0.88; knife-edge but feasible. |
| P287 | A | Evolution solves <half of checker3: 0/20. Feasible; consistent with earlier series. |
| M-X1.4 | B | Same world/controls as M-X1.3; mechanics by construction. |
| P288 | A | FAILED: BITS has 0 deaths in 48/48 worlds. Feasible; result tells vitality adds nothing for survival. Ceiling at 0 limits resolution (V). |
| P289 | C | Median paired difference with range >0 needs >50% of worlds worse; ALLN10 differs from ALL in only 42% of worlds, 58% death-free (V recompute: frac N10>ALL = 0.417, <0 = 0.0). Median ceiling at 0. |
| P290 | A | ALLN10 vs IGN: IGN 7.3 vs ALLN10 0.87; easy but feasible (noise could have destroyed the signal). |
| P291 | A | Tuned LP windows vs untuned: medians both 3.3, resolution 1.67; feasible, FAILED, quantised. |
| P292 | A | ALL vs tuned LPO: -3.33 [-3.33,-2.5]; real comparison, designer drive given tuning. |
| P293 | C | Chain of medians ALL<=ALLN2<=ALLN5<=ALLN10 with ALLN10>ALL; medians all 0 (V: ALL, ALLN2 frac0 = 1.0, ALLN5 0.98, ALLN10 0.58). Registered metric hid the dose-response. |
| P294 (M-X7.1) | N/R | Never run. |
| P295-P299 | N/R | Never run (VOID-unrun, superseded). |
| M-X7.2 (P300) | B | Sensors are trained only on 2 contexts, so unseen-context accuracy is low by construction (0.30 vs 0.86). Oracle 0.99 vs best single 0.65 is world design. |
| P301 | A | Self (trained on 1500 labels from the test contexts) vs label-free rules; built-in advantage but could fail. 0.917 vs 0.841. |
| P302 | A | Insides vs answers-only +0.031 real; "> SELF_NOISE" clause near-guaranteed (noise adds nothing). |
| P303 | A | FAILED honestly: raw-input Self ties; showed 1500 labels let it re-solve. Led to X7.3. |
| P304 | A | Mostly implied by P301 (net fixes follows from beating CONF_PICK by 7 pts); precision > 0.5 is the extra part. Redundant but not vacuous. |
| P305 | C | Self trained only on contexts 0-3, no abstain option in this version, no training case needing doubt; context 4 is never signalled. Cannot learn distrust from its information (I). |
| P306 | A | Gap closure 0.51 vs bar 0.5; feasible, knife-edge. |
| M-X7.3 (P307) | B | Same world mechanics as X7.2 (junk = random relabel => chance by construction). |
| P308 | A | More sensors helps: could have plateaued; K=8 minus K=1 +0.169. |
| P309 | A | Junk costs nothing; junk is consistently wrong (reverse oracle possible), so lenient but feasible. |
| P310 | A | Checking beats raw re-solving at 300 labels; raw MLP could have won. |
| P311 | D | Passed because Self trained on labelled blackout cases and re-solved them (0% abstain); does not measure the stated claim "learned distrust" (author notes this). |
| P312 | A | FAILED -1.13: no abstain training signal for unfamiliarity; feasible in principle (sensor disagreement), but predictable from P305. |
| P313 | A | Insides at K=12: +0.090 real. |
| M-X7.4 (P314) | B | Coin-flip context gives chance accuracy by construction (0.25). |
| P315 | A | FAILED: SET memorised 300 cases (training utility 1.000); ANS12 shows doubt learnable (0.77). |
| P316 | A | FAILED: competence damaged by memorised noise. |
| P317 | A | FAILED: novel context abstain 0.00. |
| P318 | A | FAILED, reversed by memorisation channel. |
| P319 | A | SET > RAW on seen; passes though both degraded. |
| M-X7.5 (P320) | B | Reproduces X7.4 failure with identical code and settings; passes by determinism. |
| P321 | A | FAILED by 0.01 (abstain 0.49 vs 0.5); feasible. |
| P322 | A | Memorisation shrinks with data: 1.000 -> 0.505, real result. |
| P323 | A | FAILED: none of dropout/bottleneck/decay restored doubt; untuned single settings (disclosed). |
| P324 | D | Conditional on P323 passing; none passed, so the prediction is vacuous (no best arm). |
| P325 | A | FAILED on held clause; large finding (insides pay on seen, answers-only becomes near-total doubter). |
| M-X7.6 (P326) | B | Replicates SET memorisation with same code; by determinism. |
| P327 | A | Held-context abstain 0.69 vs 0.5. Close to guaranteed by T15 arithmetic (P338) but generalisation to unseen distortion was open (I). |
| P328 | A | FAILED (0.34 > 0.3); feasible, thresholds arbitrary. |
| P329 | A | Novel context abstain 0.67; generalisation open, passed. |
| P330 | A | "worth 20x data": FAILED on 0.2 seen bar; arbitrary bars but feasible. |
| P331 | A | Insides pay in doubting Self: +0.207 vs ANS; feasible. |
| M-X8.0 (P332) | B | Checks that wrong settings lose to optimum in a simulation of the formulas; bite check on arithmetic. |
| P333 | B | Closed form vs MC of the process it was derived for (stated circular). |
| P334 | D | FAILED at 3.075 SE vs bar 3.0 over 125 points; expected max of 125 normals ~2.7-3.2, so the bar ignored multiplicity: verdict noise-dominated. |
| P335 | B | Closed-form check; real algebra gap (renewal) found, 6.8% vs 5%. Arithmetic, says nothing about learners. |
| P336 | B | T13 closed form vs sim; crossing 0.8942 vs 0.8940. |
| P337 | B | Capacity/break-even formula vs sim; first run had checker bug. |
| P338 | B | Free logits trained on redrawn labels abstain under util_loss: optimum of the loss by construction. |
| P339 | B | Autograd gradient equals product formula: identity. Local-update accuracy within 1 pt is mild empirical. |
| P340 | B | Post-hoc refined formula tested on fresh seeds; formula was fitted after failure, then verified by arithmetic. |
| P341 (X8.1 manip) | A | Whether input signatures identify problems depends on generator; SIG_TRUE 0.988 vs scrambled 0.075. Could have failed. |
| P342 | A | Audit verdict equals truth in 96%; bar 90%, feasible. |
| P343 | A | SIG_AUD beats ERR by 0.050 vs bar 0.05; knife-edge but feasible. |
| P344 | A | Gap to true labels 0.019 (<=0.05); largely implied by P342 but not vacuous. |
| P345 | D | Bar 0.894 is T13's floor from unrelated reference prices (world has no priced goal); comparison does not measure the claim. |
| P346 | A | FAILED: 3% misroutes cost 5.7x oracle. Informative. |
| P347 | A | Oracle practice helps (ratio 0.601 <= 0.7); world could have been memoryless. |
| P348 | D | "<= 15 routines" is satisfied by merging problems (8 < 10 true); the count bar can be passed by degeneracy. Second clause (fewer than ERR) is real. |
| P349 | A | FAILED. |
| P350 | A | FAILED: 41.7% misreuse; mechanism diagnosed. |
| P351 | A | FAILED. |
| P352 | A | FAILED (4/24). |
| P353 | A | Manipulation that honestly failed: sampling variance far below episode spread. |
| P354 | A | FAILED: over-creation (81 routines). |
| P355 | A | FAILED. |
| P356 | A | FAILED by one set; note SIG ~ no-memory learner (no FRESH control then). |
| P357 | A | True-label calibration 0.024 and ORACLE/FRESH 0.65: both could fail. |
| P358 | A | 14.0 routines in 8-14; edge pass. Amendment: pre-scoring selftest found SIG_CREATE identical to ERR_CREATE (75 routines each); caught before scoring. |
| P359 | A | FAILED (11.7% another's). |
| P360 | A | FAILED: worse than FRESH in 15/24 sets; informative. |
| P361 | A | Veto can detect misroutes in 82%; could have failed. |
| P362 | A | FAILED (own reuse 0.50). |
| P363 | A | FAILED. |
| P364 | A | Manipulation: look-alike confidence 0.77 vs 0.78 on answerable; measured property, could fail. |
| P365 | C | Control compares look-alike (copies context 3) with answerable pool of contexts 0-3: geometry mismatch makes gap 0.23 > 0.15 unavoidable (V: stated in X7.7b CORRECTION). |
| P366 | A | VOID but feasible; numbers show it would have passed (0.45 [0.40,0.49]). |
| P367 | C | Graded levels s at cue 3 identically distributed inputs: no sensor/state can carry s (V: LOG CORRECTION 2026-10-31). Cannot grade by construction. |
| P368 | C | Same: abstain curve flat 0.58-0.59 because s unreadable; non-increasing + >=0.5 at s=0 and <=0.3 at s=1 is contradictory with flat. |
| P369 | A | Refresh + more labels recovers competence: feasible; 0.568 vs 0.372. |
| P370 (X7.7b) | B | Matched negative control: look-alike at cue 0 and context-3 answerable cases have the same input distribution; exact expectation 0. PENDING. |
| P371 | A | Feasible; PENDING. Held look-alike geometry versus matched answerable. |
| P372 | A | Conjunction of 4 conditions with readable tell; feasible, PENDING. |
| P373 | A | GRADED vs SET on held levels; feasible, PENDING. |
| P374 | A | Skeptic bar vs always-abstain; feasible, PENDING. |
| P375 | A | Insides vs answers-only, graded; feasible, PENDING. |
| P376 | A | FAILED -0.251; climb objective could have helped extrapolation; instead hurt. Note CLIMB_U held_r4/5 ~ FRESH (V). |
| P377 | D | CLIMB vs CLIMB_U: rung shares 0.43/0.31/0.27 barely differ from uniform; held_r4 0.215 vs 0.211; lever barely moves, difference 0.003 bits. |
| P378 | B | RAWCX objective = bits of fit on training points, so noise_fit rises by construction (0.531 vs 0.213); "trap" demonstrates the objective. |
| P379 | A | FAILED; GOAL actually holds MORE on rungs 3-5 (0.74/0.515/0.424 vs MIX 0.737/0.492/0.406). Four-rung conjunction; feasible. |
| P380 | A | Skeptic bar: noise held bits not lower; noise bits unbounded below, so can fail. +0.033. |
| P381 | A | FULL vs HELP later bits +0.160; could have been zero. |
| P382 | D | Selectivity +0.002; 24-world selftest FAILED it under 0.02 floor; floor removed post hoc. Passes only through 1080+ worlds of power. |
| P383 | D | NEW-HELP +0.004, same post-hoc filter removal. |
| P384 | A | Units clause is by construction (bonus paid on existence adds units); late-bits clause (-0.206) is the real test. |
| P385 | B | ANY adds 2 units per problem for 40 problems, connects all to every head with ridge 1e-3 on 60 train points: p > n by problem ~30 (V: constants). Strawman; -1.19 inevitable. |
| P386 | A | Second-half > first-half in FULL (+0.085) vs FRESH (+0.002); world reuses concepts, so any reusing arm passes. Feasible, low news. |
| P387 | A | FULL vs DENSE on "late" (right after learning, so readout drift not an issue); attack list notes freezing confound. |

## 3. Recurring design flaws (named)

1. **Reward-definition outcome (arm blind to the variable the world uses).** The ignoring arm (IGN) is given a reward that excludes the only variable that distinguishes rooms (H), so ALL-IGN differences are fixed. X1.3 P275, P276, P277, P281; also X1.4 P290 (partly). Count 4-5 B.
2. **Median-of-zeros ceiling.** Best arms have zero deaths in all or most worlds, so a paired-median test cannot pass. X1.4 P289, P293 (V: ALL, ALLN2, BITS, ORACLE 48/48 death-free; ALLN10 58%). Same ceiling makes P288 resolve only to "0.0 [0.0,0.0]". The registered metric hid a dose-response the means showed.
3. **Information the learner is never given.** Self trained only on answerable cases is asked to doubt unfamiliar inputs: X7.2 P305, near-case X7.3 P312. Learned: doubt needs unanswerable training data (X7.4-X7.6 found this).
4. **Control/graded arm cannot see the manipulated variable.** X7.7 graded levels s were identically distributed (P367, P368), and the cue-0 control compared mismatched geometries (P365). Honest CORRECTION in LOG; X7.7b is the repair (still PENDING).
5. **Conditional/vacuous predictions and borrowed bars.** P324 (conditional on a failed prediction), P345 (floor from other world's prices), P334 (3.0 SE bar ignores multiplicity of 125 points).
6. **Degenerate-satisfiable counts.** P348 "<=15 routines" satisfiable by merging all problems; P280 "<10 deaths" satisfiable by nearly any learned arm.
7. **Strawman arms and by-objective traps.** ANY (P385: p>n by construction), RAWCX (P378: fits noise because that is its objective), TRAP units clause (P384). These are demonstrations of the framework, not experiments that could surprise.
8. **Closed-form checks with shared assumptions (B by author's own statement).** All of X8.0 except P334. Useful for algebra bugs (two found), but pass/fail cannot say anything about learners.
9. **Post-hoc bar removal or direction pre-seen.** X1.7 effect-size filter removed after selftest failed (P382, P383); X1.3 disclosure shows seed 97000 run for every arm before P275-P281 were registered; X1.6 a 4-world selftest with verdicts seen. Wins reported as "tiny but significant" with 1080-3600 worlds.
10. **Knife-edge bars with arbitrary thresholds.** P286 (0.88 vs 0.87), P306 (0.51 vs 0.5), P321 (0.49 vs 0.5), P343 (0.050 vs 0.05), P356 (17 vs 18 sets). Feasible but verdict is a coin flip relative to noise.

## Overall answer to the author's worry for this block

Roughly 20% of rows (23 B + 6 C = 29/115) could not fail or could not pass; another 8% (9 D) are underpowered or measure something else; 71 (62%) are feasible and informative, and the X7.2-X7.6 and X8.1x series in particular produced many real FAILs that redirected the work. The impossible ones are concentrated in X1.3/X1.4 (survival worlds with reward-defined outcomes, median ceiling), X7.7 (unreadable grade), and X8.0 (arithmetic). Experiments whose headline question was unanswerable by design: X1.4's dose-response (registered metric), X7.7's graded-doubt claim (fixed in X7.7b), and X8.0's relation to real models.
