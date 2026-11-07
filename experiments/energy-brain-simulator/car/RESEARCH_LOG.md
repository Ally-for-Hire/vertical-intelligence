# NeuroBlock car / steering line: research log

Rule (user, 2026-10-23): nothing is deleted. Old designs are kept as legacy files; new designs are added beside them; every
design and result is written down here, including failures. Compute: 1 thread, low priority.

## Versions (oldest first)

| ver | file(s) | idea | status |
|---|---|---|---|
| v1 | `agent.py`, `train.py`, `env.py`, `results.json`, `run.log` | Brain2 situation prototypes feed a small tanh policy net, evolved (ES). `env.py` is the shared track simulator. | LEGACY, results kept. Plain net reached ~0.65-0.82 grade on held-out tracks; the Brain2-augmented net 0.13-0.82 (3 seeds, 40 gens, unstable). PD reference 0.94-0.96; random 0.04. |
| v2 | `neuro.py` | Boolean blocks (flip, receive, send, connect): `state' = bias XOR parity(inputs)`. Starts with 12 sensor bits + 3 outputs (forward/right/left); grows by mutate-and-keep. | LEGACY. Only smoke-tested (3 generations, scores 0.14); the long run was never executed (user stopped it). `neuro.py` still supplies `SENSORS`, `sense_bits` to later versions. |
| v3 | `neurogrid.py`, `neurogrid_results.json` (last run only) | v2 blocks on a 16x16 grid, wire delay = 1 + dist//4 ticks, blocks grow next to active ones, local reward-gated rule (eligibility traces, reward-prediction error), no teacher. | LEGACY, FAILED. 150 laps x 2-3 seeds: held-out score 0.07-0.16 (about "drive forward without steering", 0.12); bloated to 185 blocks / 3,000 wires. Adding wire upkeep (decay of unused wires) cut wires to 56-1,000, score unchanged. Untested hypotheses: reward too noisy for the number of wires; dense parity networks are chaotic. |
| v4 | `voter.py`, `voter_results.json`, `voter_run_2026-10-23_a.log` | "The poll": one neuron at grid centre; each tick every neuron votes yes/no/abstain on each of forward/right/left; majority wins, tie = coin flip; each neuron keeps reward-when-ON vs reward-when-OFF per action for its situation and votes only on a significant gap; splits into two children on a sensor bit where an action's effect differs. | ACTIVE but stalled on the car. 300 laps x 3 seeds: held-out 0.15-0.19 vs coin flip 0.126; grew to 29-90 neurons; only the "forward" question was ever decided (right/left tied 67% of polls = every neuron abstained on steering). Diagnosis NOT yet measured (the check was interrupted): leading suspects are (a) score is exactly 0 off-track so random exploration gives no gradient, (b) steering's effect is delayed and interacts with the situation. |
| v5 | `cascade.py`, `cascade_results.json`, `cascade_run_2026-10-23_a.log` | "Cascade of trust": neuron i sees inputs + the states of neurons 1..i-1; opinion head p_i and trust head t_i; on-probability `t*p + (1-t)*0.5`; decision by poll (majority) or last neuron; REINFORCE; grows a neuron when reward plateaus. Tested on toy problems, not the car. | ACTIVE, first results below. |

## v5 results (2026-10-23, 20 seeds, 20,000 training ticks, fresh 20,000-sample eval, networks still sample stochastically at eval)

| problem | mode | best constant | Bayes ceiling | accuracy (mean, sd) | verdict |
|---|---|---|---|---|---|
| mirror (Y = A) | poll / last | 0.501 | 1.000 | 0.894 (.005) / 0.955 (.004) | beats coin flip; not converged |
| A AND B + 3 noise bits | poll / last | 0.752 | 1.000 | 0.821 / 0.894 | beats best constant |
| ends in 5 (from even, x5, x10 bits) | poll / last | 0.899 | 1.000 | 0.886 / 0.893 | beats coin flip, does NOT beat "always no" |
| multiple of 4 (impossible) | poll / last | 0.750 | 0.751 | 0.729 / 0.739 | near ceiling, no faking |
| greater than 50 (impossible) | poll / last | 0.501 | 0.502 | 0.501 / 0.499 | correct: nothing to learn, stays at chance |
| XOR | poll / last | 0.502 | 1.000 | 0.531 / 0.653 | beats chance only weakly; fails to solve |

Findings (measured, run 2026-10-23):
1. Objectively better than a coin flip on every learnable problem (z of 200-550 over 20 seeds for the easy ones), and it stays at
   chance on the unlearnable ones. The "manage its own randomness" part is visible: neuron 1 trust is 0.8-0.9 where the input carries
   signal and 0.5 on `gt50` (no signal), but see 3.
2. 'last' mode beats 'poll' mode on 5 of 6 problems. The poll is diluted by later neurons that never become trustworthy.
3. Defect: every run grew to the 10-neuron cap, including `gt50` where there is nothing to learn. The growth rule (reward not
   improved by 2 standard errors) cannot tell "plateau at the ceiling" from "needs more capacity", and there is no cost or pruning.
   Later neurons sit at trust 0.27-0.3 (their starting value), i.e. they are mostly coin flips that lower the poll accuracy.
4. XOR is mostly unsolved, as the toy-problem design review predicted: no single bit has a marginal effect.
5. Not yet done: ends5 should reach 1.0; it plateaus near the always-no rate, so the learning signal for the rare class is weak
   (10% of ticks). Learning rate / exploration were not tuned (and per the user's standing preference should be derived, not hand-tuned).

## Toy-problem ladder (from a Fable consult, 2026-10-23; arithmetic of the number problem re-checked by hand)
Number problem 1-100, bits (even, x5, x10): only 4 of 8 patterns occur: (1,1,1) 10 numbers, (1,0,0) 40, (0,1,0) 10, (0,0,0) 40.
Ceilings: mult4 75%, mult20 95%, gt50 50%. Ladder: mirror -> AND+noise -> number problem -> rule flip -> delayed reward -> XOR.
Rule-flip and delayed-reward problems are not implemented yet. Fable's expected tick counts are estimates, not measurements.

## Open questions / next
- Why v4 never learns to steer (measure: root-neuron on/off reward means for right/left; share of on-track ticks under random driving).
- Growth that stops: add a cost per neuron / prune neurons whose trust stays at its birth value.
- Run v4 on the toy ladder, then bring the trust/cascade idea back to the car.
- User's reference "Walt in 42 / a family member, my dad's 42 project" could not be identified; not used.

## v5 run b: unlimited neuron cap (2026-10-23, user request; `python car/cascade.py --tag unlimited`, log `cascade_run_2026-10-23_b_unlimited.log`)
`MAXN` is now unlimited by default (`--maxn 10` reproduces run a). Same 20 seeds / 20,000 ticks. The net still grew on every window
with no improvement, so the real limit was training length: 25-30 neurons on every problem, including ones with nothing to learn.

| problem | poll (cap 10 -> unlimited) | last (cap 10 -> unlimited) | best constant |
|---|---|---|---|
| mirror | 0.894 -> 0.829 | 0.955 -> 0.504 | 0.501 |
| A AND B + noise | 0.821 -> 0.762 | 0.894 -> 0.504 | 0.752 |
| ends in 5 | 0.886 -> 0.859 | 0.893 -> 0.569 (sd 0.11) | 0.899 |
| multiple of 4 | 0.729 -> 0.706 | 0.739 -> 0.506 | 0.750 |
| greater than 50 | 0.501 -> 0.499 | 0.499 -> 0.500 | 0.501 |
| XOR | 0.531 -> 0.515 | 0.653 -> 0.499 | 0.502 |

Findings: more neurons made it WORSE everywhere. Every neuron after the first stays at trust ~0.27-0.3 (its birth value), so the extras
are coin flips. Poll accuracy falls as coin-flip voters outnumber the one informed neuron (mirror: 1 informed of ~28). 'Last' mode
collapses to chance because the last neuron is always a newborn that has not had time to learn. Conclusion: capacity was never the
bottleneck; the growth trigger (no improvement) and the lack of any cost for useless neurons are. Next fix candidates: grow only if the
newborn's trust rises (otherwise retire it), per-neuron cost, or give a newborn a protected learning period.

## v5 run c: exponential neuron cost (2026-10-23, user request; default now, `--nocost` reproduces run b; log `cascade_run_2026-10-23_c_cost.log`)
Rule: when reward plateaus a newborn neuron is added ON TRIAL. After 2 windows it is kept only if the window accuracy rose by more
than its cost, else it is retired. Cost of the k-th neuron = `COST0 * 2^(k-2)` (neuron 1 free, neuron 2 costs COST0 and each next one
doubles), with `COST0` = 2 standard errors of a window's accuracy (~0.04), derived rather than hand-set. Same 20 seeds / 20,000 ticks.

| problem | poll acc | last acc | best const | ceiling | mean neurons kept | retired per run |
|---|---|---|---|---|---|---|
| mirror | 0.970 | 0.959 | 0.501 | 1.000 | 1.2 | 7.3 |
| A AND B + noise | 0.935 | 0.908 | 0.752 | 1.000 | 1.1 | 7.8 |
| ends in 5 | 0.922 | 0.909 | 0.899 | 1.000 | 1.1 | 7.2 |
| multiple of 4 | 0.741 | 0.737 | 0.750 | 0.751 | 1.1 | 8.0 |
| greater than 50 | 0.499 | 0.501 | 0.501 | 0.502 | 1.7 | 9.3 |
| XOR | 0.655 (sd .06) | 0.560 | 0.502 | 1.000 | 1.4 | 9.0 |

Findings: best results so far on every problem (run a was 0.894/0.821/0.886/0.729/0.501/0.531 for poll). "Ends in 5" now beats the always-no
rate (0.922 vs 0.899); impossible targets stay at their ceilings (no faking); XOR improved (0.53 -> 0.66) but is unsolved with high spread.
Caveat (important): almost all trial neurons are retired, so the surviving network is ~1 neuron. The gain comes mainly from removing the
coin-flip dilution, not from grown structure. The cascade has not yet shown a case where extra neurons are KEPT and needed (XOR is the
candidate: 1.4 kept on average). The 2-SE acceptance test still lets a few useless neurons through (gt50 keeps 1.7). The car task has not been re-run.

## v6 consensus brain (`consensus.py`, 2026-10-23, user spec + Fable review loop)
User spec: ONE persistent brain for every problem (no reset, no problem id); random neurons that re-read each other until >=70% agree, then a
poll; the brain manages itself from its own statistics (no outside teacher). Problems share one input vector with disjoint slots (mirror,
AND+noise, ends-in-5, XOR); each is trained alone in sequence (12,000 ticks), then all mixed. Eval: 4,000 fresh samples/problem, still stochastic.
`consensus_r0_snapshot.py` = frozen round-0 code (legacy); `--enc onehot --vote mix` reproduces round 0 inside the current file.

Round 0 (one-hot encoding, trust = noise mixing, consensus over all neurons; 10 seeds): final mirror .974, AND .964, ends5 .961, XOR .555
(sd .05), ~4.7 neurons. Log `consensus_run_2026-10-23_r0.log`.

Fable review (agent a905e..., read code + ran probes) found:
1. One-hot encoding gives the brain an implicit problem id (constant number of lit slots per problem + shared bias = free per-problem prior).
2. Every neuron was forced to be an output voter, so no hidden representation could form; XOR was structurally unlearnable (its probe: .50;
   our .555 was a cascade-ordering artefact).
3. Growth was mostly churn (trial on every plateau, ~30 retirements/run); cost base 2 and window sizes are experimenter constants.
4. Bug: retiring a neuron left other neurons' weights on its column, so the next newborn inherited them.
Fable's proposed change (adopted): trust = probability of VOTING (non-voters are readable "observers"/hidden units); bit drawn from opinion
alone; consensus and poll over voters only; REINFORCE on both draws. Plus signed encoding (absent slot = 0) and the retire fix.
Prediction was XOR .55-.80 after its phase, .70-.90 final mixed (10 seeds).

Round 1 (`--enc pm --vote abstain`, 10 seeds, mixed 24,000): final mirror .968, AND .951, ends5 .949, XOR .844 (sd .08). Prediction held.
Round 1 long (30 seeds, mixed 96,000; `consensus_run_2026-10-23_r1_long.log`): FINAL mirror .989, AND+noise .981, ends5 .982, XOR .964 (sd .034),
3.6 neurons, 1.3-2.0 sweeps to reach consensus, 0.0-1.0% unresolved, ~68 neurons tried and retired per run. Best constants .50/.75/.90/.50.
All four problems are well above chance and above the best constant, from ONE brain with no problem id.

Caveats: (a) FORGETTING during the sequential phase is large (mirror .97 -> .80 after later problems; recovers in the mixed phase), so this is
not yet continual learning without interleaving. (b) 48k sequential + 96k mixed ticks is far more data than the toy problems need; XOR needed the long stream.
(c) Experimenter constants remain: 70% consensus (user's), cost base 2, WINDOW/B, LR, trial length, unresolved>10% trigger, newborn trust -1.
(d) Toy problems only; the car has not been re-run with v6. (e) Eval is stochastic (the brain samples), so accuracy includes its own randomness.
Next: reduce sequential forgetting (e.g. per-neuron baselines, or neurons that specialise by active slots), drop the experimenter constants
by deriving them from the brain's own statistics, then connect v6 to the car sensors.

## v6 on the car (`drive_consensus.py`, 2026-10-23)
Same brain class as consensus.py (options `window`, `own_cost`, `norm_adv`; snapshot of the earlier file `consensus_v6a_snapshot.py`). 12 sensor bits + 3
question slots, three polls per tick (forward/right/left), reward = env.score only, 8 training cars in parallel, grade on 8 held-out tracks.
Success rule fixed in advance: held-out >= 0.5 on 3 evaluations in a row. References: coin .122, forward-straight .162, PD controller .943.
Result: all 5 seeds met it, at tick 18,000 / 9,000 / 19,500 / 75,000 / 9,000 (about 1-9 minutes each). 5-7 neurons, held-out 0.62-0.81 at the stop.

## v6 continued while it keeps improving (`drive_plateau.py`, resumed from those brains; stop = smoothed score sets no new best for 8 evals)
| seed | evals | first-5 mean | last-5 mean | best / worst single eval | share of evals >= 0.5 |
|---|---|---|---|---|---|
| 0 | 26 | .45 | .14 | .83 / .14 | .54 |
| 1 | 13 | .66 | .53 | .80 / .23 | .77 |
| 2 | 22 | .59 | .68 | .81 / .34 | .95 |
| 3 | 20 | .50 | .52 | .76 / .16 | .60 |
| 4 | 13 | .70 | .35 | .79 / .24 | .38 |
Finding: NO consistent improvement. The policy-gradient brain keeps drifting under constant-rate learning and exploration; individual evaluations swing
0.14-0.83 and 3 of 5 seeds end lower than they started (seed 0 collapsed to 0.14). Fix candidates (untested): keep the best checkpoint, decay the learning rate.

## v7 predictive agent (`predict.py`, `benchmarks.py` problems; user spec 2026-10-23, limits lifted for this test)
Replaces the poll: each sense predicts P(success) for every action from its own "shape" (Table = random subset of input bits with Beta counts;
RNet = fixed random tanh layer + trained logistic head); senses are pooled best-first, the agent takes the most probable action, thinks (consults more
senses) until the best action reaches TAU, otherwise explores by probability matching and spawns new random senses; useless senses retire.
Methods: A table + evidence-weighted average ("assign values to actions"), B random nets + average, C both + a learned logistic calibrator (stacking).
Bandit feedback only (it never sees the right answer, only whether its action succeeded). 10 seeds per cell; ECE = calibration error.

Held-out accuracy / ECE (TAU 0.7 | TAU 0.9). Method A = value tables, C = mixed + calibrator, B = random nets:
| problem | A | B | C |
|---|---|---|---|
| parity4 | 1.00 / .004 \| 1.00 / .003 | .66 \| .69 | 1.00 \| 1.00 |
| parity6 | 1.00 \| 1.00 | .50 \| .50 | 1.00 \| 1.00 |
| majority5 | .87 \| .99 | .95 \| 1.00 (but 4 of 10 reach TAU .9 only 1/10, ECE .16) | .90 \| 1.00 |
| mux6 | .97 \| .98 | .79 \| .81 | .96 \| 1.00 |
| adder2 (7 classes) | 1.00 \| 1.00 | .25 \| .25 | .95 \| 1.00 |
| quadrants | .89 \| .91 | .78 \| .86 | .89 \| .91 |
| moons | .89 \| .97 | .95 \| .97 | .94 \| .95 |
| circles | .79 \| .94 | .88 \| .99 | .86 \| .92 |
| spirals | .72 \| .87 | .72 \| .69 | .78 \| .84 |
| blobs3 | .88 \| .94 | .62 \| .64 | .92 \| .94 |
Findings: (1) tables solve exact logic (parity, adder, multiplexer) with near-perfect calibration (ECE .001-.007 on those); (2) random nets win on smooth
shapes (moons, circles) but cannot do parity/adder and are miscalibrated at TAU .9 (e.g. circles .99 accuracy while stating .76); (3) the mixed method
with the learned calibrator is the best all-rounder; (4) with TAU .7 the agent stops improving once it is 70% sure, so accuracy stalls (majority5
.87-.90 at TAU .7 vs .99-1.0 at TAU .9): the threshold is a stopping rule for learning too. Logs `predict_run_2026-10-23_tau70.log`, `_tau90.log`.

## v7 on the car (`predict_drive.py`; 4 seeds, 30-min cap, same plateau stop)
Context = 12 sensor bits, 6 car actions as one choice. SUCCESS event = mean score over the next 5 ticks beats the agent's own long-run average score.
B at TAU .7 was the steadiest: all 4 seeds reached smoothed best .73-.88 and ended at .79 (last-5 mean). A at TAU .9 seed 0 climbed steadily .72 -> .88 with
stated confidence .90-.94 against actual success .85-.95 (ECE about .05). But many A/C runs never took off (stuck at 0.06-0.16 = always-forward-straight: once
"sure" it stops exploring) or fell after a peak (best single eval 0.9, last-5 as low as .06-.15), and the plateau rule stopped runs that had not yet taken off.
Compare v6 policy gradient: first success within minutes in 5/5 seeds, but no steady improvement afterwards. Neither method is yet "consistent".
Caveats: the success definition (beat own average) makes stated probabilities partly depend on a moving baseline; the plateau stop is too eager for runs
that have not taken off; ECE on the car uses overlapping windows (optimistic). Next: best-checkpoint keeping, a minimum exploration floor below TAU,
patience that starts only after take-off, and a fixed (not moving) success target.

## "Purely better than an equivalent neural network" (user, 2026-10-24): predictive agent vs tuned MLPs
Harness `compare.py` (earlier version kept as `compare_it0_snapshot.py`; `predict_v7a_snapshot.py`, `predict_v7b_snapshot.py`, `benchmarks_v1_snapshot.py` are frozen
earlier code). Identical conditions for every learner: same problems, same sample stream (batch 32), BANDIT feedback only (success/failure of the action taken),
fixed budget 128,000 samples, same held-out test, 10 seeds. Rivals: (pg) policy-gradient MLP and (q) value-based MLP that predicts P(success) per action and learns
from the chosen action's outcome (the closest equivalent to the agent), each tuned PER PROBLEM on separate tuning seeds over lr/entropy-or-epsilon/hidden size {32,128}.
The agent uses one global setting. REF = same MLP trained WITH labels; level L = 0.95 x REF accuracy.
Criterion declared in advance (PURELY BETTER): on every problem (1) final accuracy >= best NN - 0.01, (2) samples to reach L <= best NN's, (3) training CPU seconds to L <= best NN's.

Iteration 0 (method C: tables + random nets + stacking, vs policy-gradient MLP only): agent won parity4/6, mux6, adder2, quadrants; lost moons, circles, spirals, blobs3, majority5.
Iteration 1: new sense `RLSArm` (per-action recursive least squares over raw bits + random tanh features, exact Woodbury updates) = method D; stronger rival (value-based MLP) added.
  D won 7/10, lost parity6 (samples), mux6 (acc .990), adder2 (one stuck seed at .45: once "sure" it stopped exploring).
Iteration 2: method E = D + (a) exploration at the rate the agent's own "sure" decisions turn out wrong (self-derived floor) + (b) check for new senses every 3 steps.
  DEVELOPMENT SET (10 problems, `compare_run_it2dev.log`): PURELY BETTER on all 10 (acc, samples, CPU).
  HELD-OUT SET (11 NEW problems never used for any design decision, agent frozen as of iteration 2; `compare_run_it2holdout.log`): PURELY BETTER on all 11:
  parity3of6, cmp3, addmod4, and_or10, majority7, checker3, sine, blobs4, rings3, xorgauss, moons_noisy15 (label noise 15%). E final accuracy equals or beats the best tuned MLP
  everywhere (e.g. addmod4 1.000 vs pg .669; checker3 .966 vs q .964; xorgauss .902 vs q .902); samples-to-L 480-1280 vs q 480-2240.
  Bug fixed on the way: workers on Windows re-import modules, so held-out problems must be looked up from BM.HOLDOUT inside `compare.problem`.
Honest limits: (1) toy problems only (<=20 input bits, <=4 classes, 2-D continuous after quantisation); no images or language; (2) many "samples to L" sit at the
480-sample measurement floor (3 evaluations x 160 samples), so they are ties with a fast MLP rather than measured gaps; (3) parameter count is NOT better than the smallest MLP
(agent stores 450-12,000 numbers; pg MLP 1.3-1.7k; the tuned value MLP is 17-19k when it picks 128 hidden units); (4) CPU numbers are training seconds from
process_time and near timer resolution; inference cost per decision not yet measured; (5) the MLP gets the same quantised bits, not raw continuous inputs.

## Car: agent E vs an equivalent value-based MLP (2026-10-24)
Both use the same 12 sensor bits, 6 actions, bandit feedback, 8 training cars, plateau stop and held-out grade (8 tracks x 300 ticks). MLP = in-h-h-6 tanh, BCE on the chosen
action's outcome, epsilon-greedy, 8 configs (lr .003/.01, eps .05/.2, h 32/128) x 2 seeds; the agent has ONE setting x 4 seeds. The best MLP config is chosen from its own
results (generous to the MLP). Three rounds:
1. success = "beat my long-run average" (moving target): agent best smoothed .59 (one seed never started), best MLP .79; BOTH decayed late (last-5 far below best).
2. success = "beat what usually happens in THIS sensor pattern" (`--success ctx`, per-context baseline, `CtxBase`): agent .889 (all seeds .86-.90), best MLP .843, all-16-config mean .67.
   Diagnosis (`diag_dips.py`): the agent drove at ~.90 for long stretches, then collapsed to "throttle off everywhere" (score 0, cars stand still): the start-from-rest context is almost
   never visited in continuous training, so its predictions there are poorly learned and can flip.
3. + episodic training (`--episode 300`: training cars reset to the start every 300 ticks, matching the held-out runs), both learners:
   | | smoothed best | last-5 | share of evals >= .5 | worst eval | samples to .5 | samples to .8 |
   | agent E (1 setting, 4 seeds) | .896 | .883 | .97 | .53 | 12k on all 4 seeds | 12k / 24k / 36k / 12k |
   | best MLP config (chosen in hindsight, 2 seeds) | .879 | .728 | .95 | .33 | 48k / 12k | 60k / 12k |
   | all 16 MLP runs | .667 | .478 | .55 | .08 | | |
   Rare single-evaluation collapses remain in 2 of 4 agent seeds (worst .15 and .35).
Files: predict_drive.py (`--success ctx --episode 300`), nn_drive.py, snapshots `predict_drive_v1_snapshot.py`, `nn_drive_v1_snapshot.py`, logs `predict_drive_E_*`, `nn_drive_nnq_*`.

## Inference and memory cost (`inference_cost.py`, wall time, 1 thread, quiet machine)
| problem | agent senses | single decision: agent / MLP32 | batch of 1000: agent / MLP32 | agent learned numbers / extra state / MLP params |
| moons | 6 | .07 / .03 ms | 1.05 / .14 ms | 576 / 14,406 / 1,666 |
| parity6 | 6 | .07 / .03 ms | .15 / .04 ms | 578 / 12,168 / 1,346 |
| mux6 | 12 | .11 / .03 ms | .39 / .06 ms | 952 / 12,168 / 1,346 |
| spirals | 60 | .56 / .03 ms | 12.5 / .13 ms | 6,578 / 196,882 / 1,666 |
The agent is 2-100x SLOWER per decision than a 32-unit MLP and keeps far more state (the RLS covariance matrices are 12k-197k numbers).

## VERDICT on "purely better than an equivalent neural network" (as of 2026-10-24)
Declared criterion (accuracy, samples-to-level, training CPU to level, 10 seeds): MET on all 10 development and all 11 held-out benchmark problems against tuned policy-gradient
and value-based MLPs (agent frozen before the held-out run). On the car it matches or beats the best hindsight-tuned MLP on score, stability and speed on average, and beats the
typical MLP by a wide margin, but it does NOT dominate on every axis and seed (one MLP seed reaches .8 as fast; the agent still has rare collapses).
NOT met: run-time decision cost and memory (see above); parameter count vs a small MLP. So the honest status is "better in sample efficiency, end accuracy, calibration, tuning
robustness; worse in inference speed and memory", not "purely better". Scope: toy problems and one control task, no images or language.
Next: speed up inference (compile/vectorise senses, drop the RLS covariance to a diagonal or low-rank form, prune senses by measured gain), remove the rare collapses (optimism for rarely
visited contexts), and test on a larger external task.

## Consistency: what makes it fail, and a version that never gets worse (2026-10-24)
User goal: "something that ALWAYS improves". Tools: `stability_probe.py` (snapshot `stability_probe_v1_snapshot.py`), `predict.py` method F (snapshot `predict_v7c_snapshot.py`).
(An earlier probe launch crashed at the first evaluation because my fixed probe set was still growing; the watcher waited 50 min for files that never came. Fixed: probe frozen after tick 6000; watchers now also stop if the processes die.)

Diagnosis, agent E (ctx success, episodic 300), 12 seeds x 90k ticks, logged at every held-out evaluation (`stability_probe_E_seed*.json`):
- All seeds reach .89-.91 and end .85-.90, but 18 of 648 evaluations (2.8%) after warmup score < .6; every dip lasts exactly one evaluation (longest streak 1) and self-recovers.
- Dip types: 8/18 cars standing still (throttle off everywhere), 2 steering failures with throttle on, 8 partial.
- NOT sense growth: births and retirements were 0 in every evaluation after warmup (the agent sits at the 60-sense cap from early on).
- Not predictable from the previous evaluation (churn .34 vs .40, calibrator movement .48 vs .47, sure rate .66 vs .69).
- What differs AT a dip: the greedy policy flipped on 65% of a fixed probe set of situations (39% in good evaluations), throttle-on .38 vs 1.00, stated confidence .54 vs .67.
- Root cause 1: the learned calibrator weights moved ~.5 (L2) per 1,500 ticks forever (constant step size 0.05), so near-tied actions flip on noise; one flip of "throttle at rest" is catastrophic.
- Root cause 2 (design inconsistency, NOT yet fixed): the agent is "unsure" on 99.5% of training decisions. Under the per-context success definition the best action succeeds only ~65%, so the
  tau = 0.9 target is unreachable; it therefore explores by probability matching all the time and always spawns (hence the 60-sense cap). tau should be tied to what is attainable.

Fixes tested on 12 FRESH seeds (20-31), paired E vs F, 90k ticks:
- Method F = E + calibrator step size decaying as 1/sqrt(1 + t/2000).
- Validation guard (champion/challenger): every evaluation the learner is run greedily on 16 validation tracks (seeds 200-215; never trained on, never in the test score); a deep copy of the
  learner is promoted to champion ONLY if its validation score is strictly higher. The deployed policy is the champion.
| | raw E | raw F | guarded champion (E or F) |
| evaluations < .6 | 16/648 | 1/648 | 0/648 |
| evaluations that dropped > .02 | 117 | 40 | 0 |
| biggest single drop | .90 | .58 | .01 / .00 |
| worst evaluation | .00 | .32 | .85 / .82 |
| final score | .892 | .886 | .902 / .903 |
Guarded champion reached .8 within 1,500-9,000 ticks on all 24 runs. Its validation score is non-decreasing by construction; its TEST score (tracks never used for promotion) also never dropped by more than .01 in 24/24 runs.
Caveats: the guarantee is exact only on the validation tracks, the test result is empirical (24 runs); "improves" means never-worse, not always-better (the score plateaus near .90 against .94 for the hand controller);
the guard costs a deep copy of the agent and 16 extra evaluation episodes per 1,500 training ticks (about .4x the training simulation cost); raw learner F still has 1 dip in 648; the benchmark problems
(compare.py) were not re-tested with the guard; tau unreachable under ctx success is still open.
