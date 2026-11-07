# Audit block C: PREDICTIONS.md X1.9 - X1.28b (P395-P498; X1.8 = P388-P394 withdrawn, not audited)

Range found: X1.8 rows start line 681; X1.8 was withdrawn before any result, so the audited block is P395-P498 (104 predictions, lines 690-838) plus the descriptive X1.28b. Ends before `### X1.29` (line 842).
Labels: [V] verified by code/recompute here; [R] taken from register/LOG text; [I] inference.

## Headline
104 predictions: A 72, B 13, C 2, D 17, E 0. Most predictions in this block COULD pass or fail. The author's fear ("a vast majority could never pass") is NOT borne out here. What is true: (a) ~13 are near-free wins (B), (b) the "vertical" quantity was confounded with world drift (D), (c) none of the block could in principle show progress on the stated ultimate goal (see section 4).

## 1. Per-experiment table
| Exp | #pred | A | B | C | D | E | One-line verdict |
|---|---|---|---|---|---|---|---|
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
| TOTAL | 104 | 72 | 13 | 2 | 17 | 0 | |

## 2. Per-prediction table
| P# | Verdict | Reason (<=25 words) |
|---|---|---|
| P395 | A | FULL vs FULLFZ (never trains): could lose; passed +0.206 |
| P396 | A | Backward change measured per arm; small but could be negative |
| P397 | A | FULLALL trains more; FULL could have held more; it did not (-0.253) |
| P398 | D | Effect +0.012; flipped PASSED/FAILED between batches [R]: noise-dominated |
| P399 | B | DENSE has 32 shared units total [V x17_dense_h] vs ~79 nodes with per-problem heads in FULL; +0.965 expected |
| P400 | A | SPEC vs HELP bonus; genuine ablation |
| P401 | B | TRAP pays depth on existence by design (adds useless nodes); lower bits is built in. Negative control, not a test |
| P402 | A | X1.7 world has no depth [V]; late-minus-early feasible, passed +0.122 |
| P403 | A | Tree vs flat, real contest; tree lost |
| P404 | A | Depth reward ablation; failed |
| P405 | A | Application weighting ablation |
| P406 | A | Update-used vs freeze |
| P407 | B | TRAP guard again, by construction |
| P408 | B | DENSE 32 units vs tree; +0.53 free |
| P409 | D | World now has depth: mean concept depth rises ~0.5 -> ~0.8, max 1.0->1.55 across stream [V, 6 seeds]; learner (not oracle) difficulty rises |
| P410 | A | Precision reward vs unshaped; failed |
| P411 | D | +0.012 [+0.002,+0.023]: lower bound at noise floor, not a meaningful pass |
| P412 | A | Self-weighting ablation |
| P413 | A | Tree vs flat |
| P414 | D | Vertical in depth-drifting world, confounded (see section 3) |
| P415 | B | DENSE handicap |
| P416 | A | Confirmation vs not; passed +0.053 |
| P417 | A | Generalization of confirmed links; failed opposite |
| P418 | A | Frontier bonus; failed |
| P419 | A | Frontier bonus on generalization; null |
| P420 | A | Tree vs flat generalization; failed |
| P421 | B | DENSE (32 units, connect-only) handicap; +0.375 |
| P422 | D | Vertical, depth-drift confounded |
| P423 | A | Distance reward; failed -0.155 |
| P424 | A | Same, later problems |
| P425 | A | Raw distance reward |
| P426 | A | Confirmation on top |
| P427 | B | DENSE handicap |
| P428 | D | Vertical, confounded |
| P429 | A | Decay rescue; null |
| P430 | A | Memory rescue; passed (not trivial) |
| P431 | A | vs plain flat |
| P432 | A | Decay alone |
| P433 | A | Decay alone, later |
| P434 | B | DENSE handicap |
| P435 | D | Vertical, confounded |
| P436 | D | Composite: tuning objective = mean of P437-P439, redundant; FLAT0 also leaves decision points unused at scoring (attack 2 [R]) |
| P437 | A | Tuned vs default on fresh worlds could fail by overfitting; passed +0.295 (expected direction) |
| P438 | A | Same, end-of-stream |
| P439 | A | Same, generalization |
| P440 | B | DENSE handicap, +0.87 |
| P441 | D | Vertical; author's own CORRECTION says ceilings equal [V: 3.30 vs 3.31], yet drop persists via learner-side depth difficulty |
| P442 | A | Depth on stagnation; null (+0.001) |
| P443 | A | Same, generalization |
| P444 | A | STAG vs ALWAYS; real contrast (ALWAYS raises evidence bar) |
| P445 | D | Share-of-ceiling vertical: fixes oracle ceiling but not learner-side difficulty (depth, pool growth) |
| P446 | D | Same |
| P447 | D | Share = bits/ceiling and ceilings equal [R], so this is P442 restated; redundant |
| P448 | D | Trend in a world whose library grows during run; author voided |
| P449 | A | Confirm vs growing charge (paired, same world); feasible though VOID |
| P450 | A | Same |
| P451 | A | Same |
| P452 | B | DENSE handicap (and VOID) |
| P453 | D | Trend in expanding world, voided |
| P454 | A | Fixed finite world; feasible; passed +0.039 |
| P455 | A | Memory vs fresh; could fail (X1.17 style) |
| P456 | A | Control: no-memory trend; passes if lower CI <= 0 [V code]; weak but could fail if world drifts |
| P457 | A | Trend difference; passed |
| P458 | A | Prior knowledge helps new concepts; plausible fail |
| P459 | B | DENSE handicap |
| P460 | A | Loop learner vs no memory |
| P461 | A | Within-stage improvement; null |
| P462 | A | Advantage grows per expansion: the closest test to the ultimate goal; borderline null |
| P463 | A | End-of-stage share rises across stages; null |
| P464 | A | As P460, longer stages |
| P465 | A | |
| P466 | A | Goal-relevant slope; null |
| P467 | A | |
| P468 | A | |
| P469 | A | |
| P470 | A | Goal-relevant slope; null |
| P471 | A | |
| P472 | A | Late-improvement test; null |
| P473 | B | More labels (120 vs 60) raising share is a statistical near-certainty; only magnitude informs |
| P474 | A | Memory x scarcity interaction; came out opposite |
| P475 | B | Bar 0.5 set after X1.18/X1.21 had already shown ~0.4 (LOG: "about a third"); foregone, result 0.422 |
| P476 | A | Base vs deep gap; measured +0.597 (expected from tanh limits but not guaranteed) |
| P477 | A | Memory helps more on deep |
| P478 | A | Compositional nodes; null |
| P479 | A | |
| P480 | A | |
| P481 | A | |
| P482 | A | Consolidation; +0.005 passes statistically, trivial size |
| P483 | A | Same |
| P484 | A | Keeps improving; null |
| P485 | A | Span base>deep; passed +0.187, expected but could fail |
| P486 | D | CI [-0.09,+0.21]: few worlds with enough deep concepts; undecidable |
| P487 | A | Fit-from-1000-points representability; 0.148 vs 0.987, a measured fact not a learner test |
| P488 | D | Compared against X1.23's point values from a different seed block with no paired control; more-data effect already known (P473) |
| P489 | A | Concept nodes; null |
| P490 | A | |
| P491 | A | |
| P492 | C | Consolidation "effectively a no-op" [R, +0.000 [+0.000,+0.001]]; mechanism did not act, cannot pass |
| P493 | A | Digger vs flat; digger lost |
| P494 | A | Input choice helps; but pool is i.i.d. Gaussian [V code], so little to choose; feasible, tiny prior |
| P495 | A | GROW vs MDL; passed |
| P496 | A | GROW vs budget; passed |
| P497 | C | Floor: median recovery of depth>=1 is 0.0 in every arm at both time points [V x1_28_run.json]; bar corr^2>=0.99 on a mixture target unreachable |
| P498 | D | +0.014 [-0.005,+0.058]: wide CI, 48 worlds; undecidable |

## 3. Recurring design flaws
1. **Handicapped baseline (DENSE) = free win (B).** DENSE is a 32-unit shared net trained online [V x1_7 run_dense, x17_dense_h=32] versus growing arms with 70-300 nodes and per-problem readouts. Nine rows compare against it (P399, 408, 415, 421, 427, 434, 440, 452, 459); all PASSED by +0.37 to +0.97 bits. They are repeated confirmations of one fact, not tests. Same shape: TRAP "guard" arms (P401, P407) whose loss is built in.
2. **The "vertical" (later minus earlier) quantity is confounded with world drift (D).** Verified recompute on 6 x1_10 worlds: the later half has higher mean concept depth (0.47->0.80, 1.00->1.33, ...) and larger max depth, with oracle ceilings equal (3.33 vs 3.28 etc.). The X1.16 correction ("ceilings equal, so the drop is real") only fixes oracle difficulty; the learner's difficulty rises with depth (X1.23: base minus deep share = +0.60; X1.26: one tanh node reaches R^2 0.148 on deep concepts). Where there is no depth (X1.9, x1_7 world) the quantity was feasible and PASSED (+0.122, P402). In depth worlds (P409, 414, 422, 428, 435, 441, 445, 446) it could not be positive for a flat-tanh learner without solving the depth problem, so those tests mostly measured the world; near-C. [V for depth drift and ceilings; I for "mostly measured the world".] The first clean version is X1.18 (finite fixed world plus a no-memory control).
3. **Floor/ceiling and no-op mechanisms.** P497 deep recovery is 0.0 everywhere (even depth 1). P492 consolidation was a no-op. P475, P473 foregone given earlier numbers.
4. **Weak power or tiny effects counted as findings.** P398 (+0.012, flipped), P411 (+0.012, lower bound 0.002), P482/P483 (+0.005), P486, P498, P488.
5. **Redundant rows / composite metrics.** P436 is the mean of P437-P439; P447 restates P442 (share = bits/ceiling, ceilings equal); P445/P446 are the vertical twice.
6. **Timing worlds seen before registration** (X1.9 seed 750000, X1.10 770000, X1.12, X1.13, X1.14, X1.16...). Disclosed honestly, but several directions were effectively known before "prediction" (X1.10 TREE below FLAT; X1.13 FLAT_G worse; X1.14 FLAT_GU worse).
7. **Unvalidated drift assumption:** X1.17's world expanded during the run; author voided it. The X1.19-X1.21 stagnation rule makes the stage schedule near-fixed (attack 1 in the notes), but these rows are still feasible.

## 4. Did anything address the ultimate goal ("infinitely deepening discoveries from nothing via one simple rule")?
- None could in principle show it fully. Every learner here is handed labelled problems from a fixed generator (tanh concept families, 60 labels, Gaussian inputs); there is no "from nothing" discovery and no self-generated problems. The unit family (tanh) is given. Depth growth was never demonstrated: tree depth ~1 (X1.10), deep nodes 5 per stream (X1.16), 0 deep concepts recovered (X1.28), concept nodes/composition null (X1.24, X1.27).
- Closest in-principle: X1.18-X1.21 (expansion loop; P460-P463, P462/P466/P470 "advantage grows with each expansion") and X1.24/X1.27/X1.28 (depth builders). They were feasible and returned null for the depth part: informative negatives.
- Could not: X1.9, X1.13, X1.14, X1.15 (tuning), X1.22 (diagnostic), X1.25: fixed-pool connection learning, no route to depth.

## 5. Per-experiment: did the headline question have an answerable outcome?
Yes for all except where noted. Answerable-and-answered: X1.9-X1.16, X1.18-X1.25, X1.27, X1.28 (each had arms that could differ; most came out negative). Headline unanswerable by design: X1.17 (world expanded during run; voided); the vertical headline in X1.10-X1.16 (D). X1.28 headline "dig until stone" is not answerable at 150 problems/60 labels (recovery floor 0).

## 6. Caveats
Not rerun: results JSONs only read for X1.28; recompute for depth/ceilings was done on generated worlds (seeds 760000-760005 for depth/ceiling; 760000-760039 for seen-fraction: 0.78 early vs 0.84 late, so pool dilution is NOT a confound). Verdict text labels in the register were checked against code for P456 and against X1.28 json; others taken from the register [R]. I did not test whether a tree learner could be positive on the vertical in a depth world, so D (not C) is the conservative call.
