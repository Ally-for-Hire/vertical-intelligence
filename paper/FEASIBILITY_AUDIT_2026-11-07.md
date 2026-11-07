# Feasibility audit of the vertical-intelligence predictions (2026-11-07)

Question (author, 2026-11-07): "go through all previous tests and identify whether they could have even
succeeded theoretically; I feel a vast majority of our project has been tests that could never have been passed."

Scope: every registered row P199-P567 plus manipulation checks (377 rows; X0.1 to X1.47, X6.6-X6.9, X7.x,
X8.x, BENCH2, E9c) in `paper/PREDICTIONS.md`. NOT covered: P1-P198 and the older simulator/emergent families before X6.6 (partly covered by
`AUDIT_2026-10-28.md`). Method: four independent read-only auditors, one per
block, each reading register text + experiment script + recorded result and recomputing from saved results where
cheap (no experiments run). Per-block evidence with a verdict and reason for every row:
`paper/feasibility/blockA.md` (P199-P274), `blockB.md` (X1.3-X1.7, X7, X8), `blockC.md` (X1.9-X1.28b),
`blockD.md` (X1.29-X1.47). Verdicts: **A** feasible and informative; **B** tautological (cannot fail);
**C** impossible by construction; **D** underpowered, noise-dominated, or does not measure the claim;
**E/N** never run, withdrawn, or conditional on a failed row.

## Result in one table

| Block | rows | A | B | C | D | E/N |
|---|---|---|---|---|---|---|
| A: X0.1-X0.10, X1.1-X1.2, X6.6-X6.9, E9c | 89 | 62 | 16 | 2 | 6 | 3 |
| B: X1.3-X1.4, BENCH2, X7.x, X8.x, X1.6-X1.7 | 115 | 71 | 23 | 6 | 9 | 6 |
| C: X1.9-X1.28b | 104 | 72 | 13 | 2 | 17 | 0 |
| D: X1.29-X1.47 | 69 | 23 | 9 | 0 | 20 | 17 |
| **Total** | **377** | **228 (60%)** | **61 (16%)** | **10 (3%)** | **52 (14%)** | **26 (7%)** |

## Answer to the worry

1. **Row by row, the worry is not borne out.** 60% of rows could genuinely pass or fail, and many did fail
   honestly. Only 3% are strictly impossible. 16% cannot fail and 14% are noise-dominated, so about a third of the
   register carries little or no information.
2. **At the goal level, the worry is largely right.** The questions the author cares most about were mostly NOT
   answerable by the experiment that asked them:
   - **"Discoveries from nothing / infinitely deepening" (X1.29-X1.47):** zero valid tests. The learner is handed the
     rule's structure (the cell rotates inputs exactly as the world's `x[n mod 8]` rule does; one parent per concept;
     problems shown in dependency order; sine units or candidate lists that contain sine, the generating family).
     Depth was never tested past level 4. At g=2.0 the target (corr^2 >= 0.99 at depth) is effectively unreachable at
     the label budgets used (block D section 4).
   - **"Vertical" (later problems learned better than earlier) in X1.10-X1.17:** confounded with depth drift in the
     generated world (mean concept depth rises 0.5 -> 0.8 across the stream) and with a learner that gets worse with
     depth, so the quantity could not have come out positive for reasons unrelated to the idea. The first clean
     positive was X1.18 (finite world + no-memory control, +0.039). Block C section 3.
   - **"Graded doubt" (X7.7), "boredom exactness" (X1.4), "unit vs rate" (X0.8):** headline unanswerable by design.
     The first two were repaired or re-measured by unregistered means; X0.9 redoes X0.8.
3. **Where the design was good** (informative, could fail, and did): X0.6, X0.7, BENCH2, X7.2-X7.6, X8.1-X8.1e,
   X1.18-X1.21, X1.24, X1.37-X1.38, X1.47. Several of these changed the program.

## Recurring design flaws (named, so they are avoided next time)

1. **Foil arm that cannot see the variable** (X0.1/X0.1b/X0.7 need-slope rows; X1.3 IGN; X7.7 look-alikes). Outcome fixed by reward/input definition.
2. **Bar a no-skill arm already clears** (X0.10: RANDOM 0.54 vs bar 0.5; X0.2 geometry-fixed retreat).
3. **Handicapped baseline** (DENSE 32 shared units vs 70+ nodes with per-problem heads: nine rows, all "won" by +0.37 to +0.97 bits; TRAP arms lose by design; IGN's tuned discount produces the ALL-IGN gap).
4. **Certain-direction or restatement rows** ("readings differ", "exam falls with depth", "linear stack depth irrelevant", closed-form checks X8.0 that are arithmetic on shared assumptions).
5. **Median/threshold tests on tiny, bimodal samples** (3 worlds; SE of a 3-world exam mean ~0.09 vs a 0.1 window; "at least 2 of 6" replication with ~40% power).
6. **Metric confounded with world drift** (see above) or at a floor (deep recovery 0.0 in every arm, P497).
7. **Learner given the answer's structure/family** (vacuous "from nothing").
8. **Thresholds set or borrowed after seeing data** (P542, P554, P557, P345; X1.7 P382/P383 pass only after a post-hoc filter removal).
9. **Never-run or withdrawn rows counted in the register** (X1.31-X1.34, X1.8, X7.1; 26 rows).

## What to do differently (proposed gates, not yet in the protocol)

- Before registering a row, run the **feasibility gate**: (a) an oracle/least-squares/true-structure arm must clear the bar; (b) a no-skill arm must NOT; (c) every compared arm must see the manipulated variable; (d) n and bar must give at least 80% power for the stated effect, with the sample's bimodality checked.
- Count only A-rows in the "predictions passed/failed" headline; show B/D separately. The public at-a-glance line (352 predictions, 168 passed) currently mixes all of them.
- New-direction worlds (I30 pattern-finder) must pass an oracle gate and must not hand the learner the generating family.

## Caveats

- Auditors classified from reading code and saved results; none re-ran experiments. X6.x rows were classified from register text only. P216, P257, P259, P262 are inferred from register numbers plus the x0_1 reward definition. Treat borderline B/D calls as judgment, and spot-check any row before acting on it.
- The 10 C rows and ~20 highest-impact D rows are listed in the block reports; the consolidated numbers above are sums of the four tables.
- Nothing in the register was edited; per the "never delete a result" rule, status relabels should be added as a CORRECTION block, not applied here.
