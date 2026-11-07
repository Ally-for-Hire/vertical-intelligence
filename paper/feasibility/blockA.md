# Block A audit: paper/PREDICTIONS.md lines 210-410 (P199-P274 + manipulation checks)

Method: read the register text, the scripts (vi/x0_1.py, x0_2.py, x1_1.py, constants.py, x0_6.py head; results/x0_10_run.json recomputed), and used the recorded numbers. Verified = I read the code or recomputed; Inferred = from register text only. X6.x (neuroblock) classified briefly from register text only (inferred). Register PASSED/FAILED labels were not trusted; they were checked where noted. P253 does not exist in the register (numbering gap). 89 rows (75 P rows + 14 M-checks).

Verdicts: A feasible-informative, B tautological, C impossible-by-construction, D underpowered/not measuring the claim, N/A conditional prediction never evaluated.

## 1. Per-experiment summary

| Experiment | rows | A | B | C | D | N/A | One-line verdict |
|---|---|---|---|---|---|---|---|
| X0.1 | 6 | 4 | 2 | 0 | 0 | 0 | Safety/valence rows real (two failed honestly); the headline "for me" need-slope row cannot fail because the comparison arm's reward ignores state |
| X0.1b | 5 | 3 | 1 | 0 | 1 | 0 | Foresight question answerable and answered (no); need-slope row repeats the X0.1 tautology |
| X0.6 | 7 | 7 | 0 | 0 | 0 | 0 | Best-designed block: oracle-feasibility check, probes>=30, real failures. Answerable question; answer was mostly "no" |
| X0.7 | 7 | 6 | 1 | 0 | 0 | 0 | Real, two-world design with feasibility check; only the need-slope row is structural |
| X6.6 | 6 | 5 | 0 | 0 | 0 | 1 | Informative (null). Gene-movement check mis-targeted for the duplication claim |
| X6.7 | 6 | 2 | 2 | 1 | 0 | 1 | Literal world was infeasible (extinct 20/20); two rows are controls that cannot fail |
| X6.8 | 5 | 5 | 0 | 0 | 0 | 0 | Informative, real fails |
| X6.9 | 6 | 4 | 0 | 0 | 1 | 1 | Informative; "not the cheapest of four" row is noise-dominated |
| X0.2 | 6 | 3 | 1 | 1 | 1 | 0 | The secure-base signature is fixed by world geometry (threat penalty 0.25 > every patch value 0.05); one baseline bar sits above what the tuned arm can reach |
| X1.1 | 6 | 4 | 2 | 0 | 0 | 0 | Dark room and noisy TV are true by reward definition; LP-vs-random slowness is a real finding |
| X1.2 | 5 | 4 | 1 | 0 | 0 | 0 | Registered as expected confirmations of a derived fix; could still fail, so A; NOVO row is tautological |
| X0.8 | 6 | 5 | 1 | 0 | 0 | 0 | Real predictions, three failed; but the design confounded unit with rate, so the headline question was NOT answerable (X0.9 admits this) |
| X0.9 | 5 | 3 | 1 | 0 | 1 | 0 | Diagnosis experiment; two rows near-expected by construction, one confounded, one fixed by rates |
| X0.10 | 9 | 3 | 4 | 0 | 2 | 0 | Headline "ALL finds depth / fills hidden need" passed by bars that RANDOM also clears; the tuned ALL-IGN gap is an IGN-handicap effect |
| E9c | 4 | 4 | 0 | 0 | 0 | 0 | Real, but a data repair of a toy (library contains Kepler form); informs about data, not VI |
| TOTAL | 89 | 62 | 16 | 2 | 6 | 3 | E (unknown) = 0 |

Experiment-level answerability (could any outcome have told the author something new?):
- Answerable and informative: X0.1b (foresight), X0.6, X0.7, X6.6, X6.8, X6.9, X1.1 (LP slower than random), X0.8 (negative surprises), E9c (data).
- Headline unanswerable by design: X0.8 (unit and rate confounded; the raw sum "won" because scale also scaled dynamics; admitted and redone in X0.9). X0.10's headline "listening to boredom enables finding depth" (RANDOM reaches level 4 faster, 2072 steps vs ALL 3004 steps; RANDOM H_late 0.54 > bar 0.5): the world tests sustaining a hidden need, not discovery (the register's own critique, verified in results/x0_10_run.json). X0.2's headline "one signal explains secure-base behaviour": the severe-world half is geometry-fixed.
- Headline answerable but the main tested contrast tautological: X0.1 (P200 "for me" signature).

## 2. Per-prediction table

| P | V | Reason (<=25 words) |
|---|---|---|
| P199 | B | Author says near-certain by construction: B0 reward ignores integrity. Verified in x0_1.py reward(). |
| P200 | B | Verified: SUM/B0 reward is state-independent (eff.sum() if eat), HOMEO's is drive reduction. Need slope is fixed by reward form; SUM 0.137 is leakage. |
| P201 | A | HOMEO vs SUM deaths; could go either way; failed honestly (7.0 vs 2.0). |
| P202 | A | Valence at neutral state; setpoint 0.8 vs neutral 0.5 makes it genuinely open; failed 0.62 vs 0.93. |
| P203 | A | Post-swap deaths counts are tiny (0 vs 4 medians) so weak, but outcome not fixed. |
| M-X0.1 | A | Correlation <0.9 check is a real (loose) void test. |
| P204 | A | TD gamma vs gamma=0; failed borderline [-1.67, 0.0]. |
| P205 | A | Non-inferiority vs SUM; failed. |
| P206 | B | Same structure as P200: SUM slope 0.056 fixed by state-blind reward; HOMEO0 0.725 shows foresight irrelevant. |
| P207 | A | Valence improves with foresight; failed (+0.009). |
| M-X0.1b | D | Tuned gamma 0.99>0 does not show foresight mattered (P204 failed); check cannot void anything. |
| P208 | A | VIAB vs B0 deaths; failed (16.0 vs 9.3). |
| P209 | A | VIAB vs DEATH; passed 16.0 vs 38.3. |
| P210 | A | Non-inferiority vs HOMEO; failed. |
| P211 | A | VIAB vs SUM slope; SUM near 0 structurally, so reduces to VIAB>0; failed (CI includes 0). |
| P212 | A | Oracle agreement VIAB vs HOMEO; ~0.68 vs 0.67; open, failed. |
| P213 | A | VIAB vs DEATH agreement; borderline fail. |
| M-X0.6 | A | Oracle<others, Pearson>0.5, probes>=30: real, passed (0.572 marginal). |
| P214 | A | MARGIN vs VIAB, orth world; passed with feasibility verified. |
| P215 | A | Non-inferiority vs HOMEO; passed (MARGIN safer). |
| P216 | B | MARGIN state-dependent reward vs SUM state-blind reward; slope gap by reward form (same as P200). |
| P217 | A | MARGIN vs SUM deaths in trade world; failed (14.7 vs 11.3). |
| P218 | A | Non-inferiority in trade world; failed. |
| P219 | A | Oracle agreement vs SUM; CI spans 0 and the gap is ~0.02. Open. |
| M-X0.7 | A | ORACLE<0.5 RANDOM feasibility and trade-bites check: real. |
| P220 | A | Inherited direction gene; failed 51/70 vs 60/70 (inferred). |
| P221 | A | Recombination gene; failed 1.07x vs bar 1.25x (inferred). |
| P222 | A | Duplication helps at impasse; numeric clauses met but VOID by gene check (inferred). |
| P223 | A | Combination arm; failed (inferred). |
| P224 | N/A | Conditional on a judge pass; nothing passed. |
| M-X6.6 | A | Gene p<0.01 per gene is a real check; mis-targeted for u (benefit was capacity, not evolved rate). |
| P225 | C | Literal energy world went extinct 20/20 (change summed over 16 points vs thresholds 0 and 2); bug-check already showed extinction. Could not pass as specified. Inferred. |
| P226 | N/A | Conditional on (b) passing. |
| P227 | B | Random search 0/20 on a 3e6 budget; control essentially cannot reach 0.95. Inferred. |
| P228 | A | Per-point average world; 17/20 vs bar 18 (bar arbitrary but open). |
| P229 | B | Unlimited food (pays 2 per correct answer) explodes by construction, 20/20. |
| M-X6.7 | A | Spearman and conservation checks real; clause (2) is an OR that (c) trivially meets. |
| P230 | A | Self-paced ramp; failed 1.19x. |
| P231 | A | Fixed ramp; failed (4/20). |
| P232 | A | Mixed stages; failed (0/20). |
| P233 | A | Ramp cracks checker3; failed 0/10 for all. Low prior (no arm in X6.5-6.9 ever solved it) but open. |
| M-X6.8 | A | Impasse exists (10/20 direct fails); ramp reaches full problem 20/20. |
| P234 | A | Fixed-rate duplication; failed 1.23x vs 1.25x. |
| P235 | A | Spirals concentration; failed 1.43x vs 1.5x. |
| P236 | D | "u=1.0 not cheapest of four" has about 75% chance under null; costs differ ~5% (100,127 vs 105,902). |
| P237 | A | Checker3 null again (0/10); low prior but open. |
| P238 | N/A | Conditional on a judge pass. |
| M-X6.9 | A | Width>=17 check real; passed (23-30). |
| P239 | B | Verified in x0_2.py: severe delta 0.25 vs mu_hi 0.05, so away is never worth it under threat; explore_threat=0 and threat_away=0 for ONE and ORACLE alike. explore_calm>0.03 is Thompson property. |
| P240 | C | Verified: B0 explore_threat <= eps*(K/(K+1)) and tuning grid is (0.01, 0.03, 0.1, 0.3); chosen eps=0.03 makes the >0.03 bar unreachable (observed 0.021). ONE clauses are geometry-fixed (B). |
| P241 | A | Regret non-inferiority vs per-world-tuned TWOS*; failed by ~0.0002 regret (tiny effect, still open). |
| P242 | A | ONE vs tuned B0 regret; passed. |
| P243 | D | Coarse tuning grid (w, eps) returned the same optimum in all worlds (difference exactly 0.0), so the test could not discriminate; transfer was likely from the grid. |
| M-X0.2 | A | Oracle behaviour, threat share (p_on/(p_on+p_off)=0.167 by construction), ONE<RANDOM: sound sanity checks. |
| P244 | B | Verified in x1_1.py: UR reward=-err, dark error 0, v starts 0 so greedy picks dark by definition. |
| P245 | B | Verified: NOV reward=err, noise err~0.94 is max; picks noise by definition. |
| P246 | A | LP combined trap share 0.528 vs UR 0.967; random 0.667; real contrast (LP only slightly under random). |
| P247 | A | LP t_l2 vs UR; partial fail, CI includes 0. |
| P248 | A | LP slower than random (388 vs 64.5): real finding, caused by v init 0 (pessimistic). |
| M-X1.1 | A | Noise unlearnable, dark error 0 by construction, oracle <100: bug detectors, fine. |
| P249 | A | LPO vs LP; derived fix, could have failed (e.g. optimism washing out); passed with huge margin. |
| P250 | A | LPO <= 1.5x RANDOM; open, passed. |
| P251 | A | Trap share <0.6 and below UR/NOV; open, passed. |
| P252 | B | NOVO reward = err, noise err max; optimism cannot change that. |
| M-X1.2 | A | Parity with X1.1 results: real code check. |
| P254 | A | SUM-SUMN with differing scales; failed reversed (-19.3). |
| P255 | A | MARGIN non-inferiority vs SUMN; failed. |
| P256 | A | MARGIN vs SUM; failed reversed. |
| P257 | B | B0 ignores needs; deaths rise with N (205 to 324), and RANDOM too (279 to 460): a world property. |
| P258 | A | Deaths concentrate on small-scale variables; failed (18%). |
| M-X0.8 | A | REF<0.5 RANDOM: feasibility; passed (13 vs 460). |
| P259 | A | Registered as near-expected; 10x unit spread makes SUM weighting arbitrary but Adam scale-invariance could have saved it. Low info. |
| P260 | A | MARGIN vs SUM in U1; passes tuned but default-setting gives the opposite sign (+12.7): setting-sensitive. |
| P261 | D | Cross-world difference of differences is confounded (SUMN also worse in U2: 131 vs 89), as register admits. |
| P262 | B | In U2 smallest unit = fastest variable; in X0.8 every arm piled 41-53% of deaths on the fastest, so >40% passes without any SUM property. |
| M-X0.9 | A | Parity (15/15), REF<0.5 RANDOM: real. |
| M-X0.10 | A | REF0 dies 100%, REF 0%: real feasibility check. |
| P263 | D | Tuned ALL-IGN gap +0.40 comes from IGN's tuned discount 0.5 (IGN default H_late 0.93 vs ALL 0.99: gap +0.06); arm handicapped by its own tuning objective. |
| P264 | B | Verified: ALL0 cannot reach tasks beyond circle; REF0 dies of H in 100% of worlds; ALL0 fails (22.7 vs 5.3) by world design. |
| P265 | B | Verified from results: RANDOM H_late median 0.54 >= bar 0.5, so any arm that eats and tasks passes. |
| P266 | B | Verified: RANDOM reaches level 4 (top=4, t3=2072) faster than ALL (3004), 5 of 7 actions are tasks; passes for any arm, as the register concedes. |
| P267 | D | ALL-IGN -89 steps [-368, -8.5] out of 3,000, while RANDOM is faster than both; measures nothing about learned discovery. |
| P268 | B | REF is a hand policy with 0.0 deaths; "learner worse than informed reference" is near-certain. |
| P269 | A | Hidden-need deaths ALL vs IGN; failed borderline [-3.3, 0.0]. |
| P270 | A | Default-setting replication; passed but gap only +0.06. |
| P271 | A | Sun/Jupiter ratio within 2%; real, passed (1050.1, +0.26%). |
| P272 | A | Moon exponent 1.5; library contains the form, so passes on real data only; data-checking, not VI. |
| P273 | A | BIC difference >= 40 (156.5); open. |
| P274 | A | NSSDC residuals smaller than JPL (0.024% vs 0.464%); data repair, could fail. |

## 3. Recurring design flaws

1. **State-blind comparison arm (the "for me" tautology).** The arm used as foil has a reward that cannot depend on internal state, so the "need-dependent valence" contrast is fixed by the reward definition (verified x0_1.py reward(): SUM = eff.sum() if eat). Examples P200, P206, P216; also P211 reduces to "VIAB>0". Informative only against an arm that could be state-dependent.
2. **Behaviour fixed by reward definition (drive tautology).** UR dark room, NOV noise, NOVO noise (P244, P245, P252) and the severe-threat retreat (P239, ONE and ORACLE both exactly 0; penalty 0.25 > patch max 0.05). The outcome is mathematically fixed; only the LP/LPO rows carry information.
3. **Bar a no-skill arm already clears (non-discriminating thresholds).** P265 (RANDOM 0.54 >= 0.5), P266 (RANDOM reaches level 4 faster than ALL), P262 (every arm puts 41-53% of deaths on the fastest variable vs bar 40%), P267, P268 (REF=0 deaths). Fix: require beating RANDOM/null arm on the same metric.
4. **Handicapped or restricted comparison arms.** P264 (ALL0 cannot access the ladder; REF0 dies of H 100%), P263 (IGN tuned to discount 0.5; default gap +0.06). Gap reflects the handicap/tuning rather than the mechanism.
5. **Bar tied to the tuning grid / threshold against own parameter.** P240 (bar 0.03 equals B0's chosen eps; max reachable < eps), P243 (coarse grid gives identical optimum, difference exactly 0).
6. **Confounded world makes the headline unanswerable.** X0.8 (unit multiplied rate and units together), P261 (cross-world difference confounded). Later diagnosed in X0.9, which is the proper repair.
7. **Infeasible-as-literally-specified world.** P225 (energy world extinct 20/20; bug-check already showed it). A feasibility pre-check on the pre-registered arm, not just controls, is needed before registering a pass bar.
8. **Noise-level directional rows.** P236 ("not the cheapest of four"), P203 (counts of 0 vs 4 deaths), P243. Under 5-6% effects with no power analysis.
9. **Registered "expected confirmations" after seeing the bug check.** X1.2 (all four) and X0.9 P259/P261/P262 were registered after a one-seed look. They can technically fail (A) but have low information, and the register itself labels them diagnoses.

Positive notes: M-checks of the form "oracle/reference survives, random does not" (M-X0.6, M-X0.7, M-X0.8, M-X0.9, M-X0.10) are good feasibility gates and prevented C-type failures in those experiments; the honest failures (P201, P202, P208, P210, P217, P254-P256, P258) are the highest-information rows in the block.

Limits: X6.x rows classified from register text only; I did not read x0_7/x0_8/x0_9 code beyond headers, so P216/P257/P259/P262 are inferred from register numbers plus the x0_1 reward definition.
