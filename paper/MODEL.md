# The Model: exactly how the pieces combine (proof loop, started 2026-10-30)

Author's request (2026-10-30): "theorems and good, grounded expectations ... a complete, entire model of EXACTLY how we will combine these elements so by the time we need insane GPU we can just use it to measure results, not test."

**How to read this file.** Section 1 is the model in plain words. Section 2 is the controller: every quantity it uses is either a price (physics) or measured online, so nothing is hand-tuned. Section 3 is the theorems, each with its proof and its status. Section 4 is what the GPU stage measures. Section 5 is where the theory is most likely to break. Status words: **PROVEN** (mathematics, under the stated assumptions), **CHECKED** (a registered simulation agreed with the closed form; see `PREDICTIONS.md`), **EXPECTATION** (reasoned, not proven), **OPEN**.

## 1. The model in plain words

A life is a stream of cases. Each case belongs to a hidden **class** (a kind of situation). Five parts, all paying for everything they do (author's given 4: no action is free):

| Part | Job | Cost per case | Built from (proven pieces) |
|---|---|---|---|
| **Sensors** `S_1..S_K` | networks of any kind, read as complete internal states (answer + hooks on every leaf layer) | `c_S` (all sensors together) | X7.2-X7.6 |
| **Self** | reads the sensors, answers or abstains, to maximise the life goal (+1 right, -`lam` wrong, 0 abstain) | inside `c_S` | X7.3, X7.6 |
| **Recogniser** | names the class of the case from the case alone | inside `c_A` | memory-based router (CTX5), T8 |
| **Automatic path** `A_z` | one cheap routine per class (the "muscle memory"): a small local-rule network | `c_A` << `c_S` | NeuroBlock automatic neurons, C51 |
| **Audit** | with probability `q`, a case is ALSO run through the slow path; disagreement is the error signal | `q c_S` | X7.3 (double-checking), three-factor tag C51 |

**One case, step by step**
1. The recogniser names the class `z-hat`.
2. If `z-hat` is not automatic, run the slow path (sensors, Self) and act on its answer (or abstain). The case is a practice case for class `z-hat`.
3. If `z-hat` is automatic, act on `A_z-hat`'s answer. With probability `q`, also run the slow path, act on the slow answer instead, and log agree/disagree.
4. Disagreements are the third factor (broadcast error): they train `A_z-hat` by a local rule and they train the recogniser (the audit is a TEACHER that unsupervised self-recognition lacked, the P5 failure).
5. The hand-off controller (section 2) decides, per class, whether it becomes automatic, stays automatic, or is demoted to slow (a detected shift).
6. Cases the Self flags as unanswerable (no usable signal) are abstained on and never trained on a fixed label (T15), so they cannot be memorised.

## 2. The controller (no free knobs)

Prices (physics, fixed by the world or the goal): `c_S`, `c_A`, `c_T` (one-off cost to train an automatic routine), `c_R` (cost to retrain after a shift), `lam` (cost of a wrong answer, in the same units as `c_S`).
Measured online, per class: `e_S`, `e_A` (error rates), `d` (disagreement rate between A and S in audits), `rho` (shifts per case; counted from detections), `m` (disagreements needed to call a shift; the audit's own noise sets it).

Derived quantities (all closed forms below):

| Quantity | Formula | Meaning |
|---|---|---|
| answer threshold `tau` | `lam/(1+lam)` (2/3 at `lam`=2) | Self answers only above this confidence (T9) |
| audit rate `q*` | `sqrt(rho lam m kappa / c_S)` (capped at 1; `kappa` = 1 unless failures are shared, T12) | how often to double-check an automatic path (T12) |
| value of automaticity `G(q)` | `(1-q)c_S - c_A - lam(1-q)(e_A-e_S) - rho[lam m kappa(1-q)/q + c_R]` | cost saved per case versus staying slow (T11+T12) |
| **hand-off rule** | make class automatic iff posterior mean of `G(q*)` > 0 | risk-neutral Bayes: `G` is linear in the error rates, so the posterior mean decides and no confidence level is needed |
| error budget `delta` | `c_S/lam - c_A/(lam(1-q))` | how much worse than the slow path an automatic path may be (T11) |
| practice before hand-off `n*` | typical `n > (1-2 delta)/(delta - d)`; with `d`=0, `n* = floor(1/delta - 2) + 1` | how many checked repeats are needed (T11) |
| recognition floor `a*` | `1 - (delta + e_S - e_r)/(e_w - e_r)` | below this recogniser accuracy nothing becomes automatic (T13) |
| stability limit `rho_max` | `(sqrt(c_S) - sqrt(c_A'))^2 / (lam m kappa)` with `c_A' = c_A + lam(e_A - e_S)`, exact for `c_R`=0 | in a world that shifts faster than this, automaticity never pays (T12) |
| break-even recurrences `R_be` | `n* + c_T / G` | a class must recur this often for training the routine to pay (T14) |

## 3. Theorems

**T9 (answer threshold).** *Status: PROVEN, CHECKED (X8.0, P333 passed).* A Self with a calibrated confidence `p` (probability its top answer is right), utility +1 right, -`lam` wrong, -`kappa` for abstaining (0 in X7), maximises expected utility by answering iff `p >= (lam - kappa)/(1 + lam)`. Proof: answering yields `p - lam(1-p)`; abstaining yields `-kappa`; the first is at least the second iff `p(1+lam) >= lam - kappa`. For any other threshold `t` the regret is `integral over p between t and tau of |(1+lam)p - (lam-kappa)| f(p) dp` (f the density of confidence), which is quadratic in `|t - tau|` near `tau`. QED. Consequence: the abstain rule needs no tuning; X7.3's `rule_util` already uses it.

**T10 (information never hurts the ideal Self; junk is free in the limit).** *Status: PROVEN for the Bayes Self; EXPECTATION for learned Selves (observed: K=8 to K=12 with 4 junk sensors, +0.010, X7.3).* Adding a sensor adds a component to the observation; the Bayes decision rule on the larger observation can ignore it, so its optimal expected utility cannot fall (Blackwell). A learned Self with `n` labels pays an estimation cost that grows with the readout dimension it must learn to ignore, shrinking like `1/sqrt(n)` for a linear-in-features head [standard learning theory, not derived here for attention pooling]. The memorisation edge (complete state at 300-1500 labels) is the same estimation cost seen from the other side and is covered by T15, not by this theorem.

**T11 (hand-off rule and practice length).** *Status: PROVEN (static case rho=0), CHECKED (X8.0: all signs and exact `n*` agree; P334 FAILED as registered by 0.075 SE because its bar ignored 125-fold multiplicity).*
Per-case cost, slow only: `J_S = c_S + lam e_S`. Automatic with audit rate `q` (an audited case uses the slow answer):
`J_A = c_A + q c_S + lam[(1-q) e_A + q e_S]`.
`J_S - J_A = (1-q)c_S - c_A - lam(1-q)(e_A - e_S)`, so hand off iff `e_A - e_S < delta`, where
`delta = c_S/lam - c_A/(lam(1-q))`.
Two readings: (i) a free automatic path may be worse than the slow one by at most `c_S/lam` in error rate: the price of one slow computation buys that many error points; (ii) the dearer a mistake is relative to thinking (`lam/c_S`), the smaller `delta` and the longer practice must last.
Measuring it: `e_A - e_S <= d` (if A is wrong and S right they disagree), so `d` is a conservative stand-in. With a uniform prior the posterior mean of `d` after `n` audits with `k` disagreements is `(k+1)/(n+2)`. Hand off at the first `n` with `(k+1)/(n+2) < delta`. With no disagreements: `n > 1/delta - 2`, i.e. **practice length is about `lam/c_S`, a pure function of prices.** With disagreement rate `d`, the typical path crosses at `n ~ (1-2 delta)/(delta - d)`; early crossings by luck (few disagreements by chance) make the median lie between the `d`=0 value and this one.
Proof: algebra above. QED.

**T12 (square-root audit law and stability limit).** *Status: PROVEN (under assumptions A2-A4), CHECKED at the optimum and for the stability limit (X8.0); the overhead formula below is the short-delay approximation: P335 FAILED as registered (6-7% off at audit rates away from `q*`), see the CORRECTION after the proof.*
Assume shifts arrive at rate `rho` per case, a shift raises A's error (and its disagreement with S) by `Delta` and nothing else, and a shift is declared at the `m`-th audited disagreement. The number of cases until then is negative-binomial with mean `m/(q Delta)`; extra errors accrue only in unaudited cases (fraction `1-q`) at rate `Delta`. Expected extra loss per shift: `lam Delta (1-q) m/(q Delta) = lam m (1-q)/q`, **independent of `Delta`** (a small shift is slow to find but cheap while undiscovered; a big one the reverse). If part of the new failure is SHARED with the slow path (both wrong together, so no disagreement), the audit sees only a disagreement jump `Delta_d < Delta_e`, and the loss is multiplied by `kappa = Delta_e/Delta_d >= 1`; `kappa` = 1 is the clean case. Overhead per case: `Phi(q) = q c_S + rho[lam m kappa (1-q)/q + c_R]`. With `s^2 = rho lam m kappa`: `Phi(q) = q c_S + s^2/q - s^2 + rho c_R`, minimised at `q* = s/sqrt(c_S)`, giving `Phi(q*) = 2 s sqrt(c_S) - s^2 + rho c_R`. Automaticity beats staying slow iff `c_A' + Phi(q*) < c_S` (`c_A' = c_A + lam(e_A - e_S)`). For `c_R` = 0 this is `s^2 - 2 s sqrt(c_S) + (c_S - c_A') > 0` fails, i.e. `s < sqrt(c_S) - sqrt(c_A')`, so **`rho_max = (sqrt(c_S) - sqrt(c_A'))^2 / (lam m kappa)`**, with `q*(rho_max) = 1 - sqrt(c_A'/c_S) < 1` (the cap on `q` never binds first). Reference world (`c_S`=0.1, `c_A`=0.005, `lam`=2, `m`=`kappa`=1): `rho_max` = 0.030, i.e. one shift per 33 cases. QED. Valid while the detection delay is short against the gap between shifts: `rho/(q Delta_d) << 1`; the checks run in that regime and say so.
**CORRECTION (X8.0, 2026-10-30; the statement above is kept as first written).** Time spent shifted lowers the effective shift rate. The renewal-reward form is `Phi(q) = q c_S + rho[lam m kappa (1-q)/q + c_R] / (1 + rho D)` with `D = m/(q Delta_d)` the mean cases to detection. It is within 0.5% of simulation in all 48 cells on fresh seeds (P340), against up to 6.4% for the short-delay form. `q*` and `rho_max` above are the short-delay values; they stayed correct in the checks (`q*` was the best grid point in all 16 settings; the stability check separated winners from losers) and shift slightly in fast-shift worlds.
Expectation this encodes (the author's muscle-memory claim, quantified): **skills pay in proportion to the stability of the world**; the audit rate should fall as `sqrt(rho)`; and a mind in a world that changes faster than `rho_max` should stay deliberate.

**T13 (recognition floor).** *Status: PROVEN given T8, CHECKED (X8.0).* If the recogniser names the right class with probability `a`, a correctly routed case has error `e_r` and a misrouted one `e_w`, then `e_A = a e_r + (1-a) e_w` (T8). Hand-off needs `e_A - e_S < delta`, i.e. `a > a* = 1 - (delta + e_S - e_r)/(e_w - e_r)`. Reference numbers (`c_S`=0.1, `lam`=2, `c_A`=0.005, `q`=0.05, `e_S`=0.05, `e_r`=0.02, `e_w`=0.75): `delta` = 0.0474, `a*` = 0.894. **The P5 failure (self-recognition 50% of repeats) sits far below this floor, which is why its automatic path could never pay.** EXPECTATION (not proven): audits turn recognition into supervised learning (the slow answer is a label for "A was right here"), so the floor is reachable; X8.1 must test this on the P5 world. **X8.1 RESULT (2026-10-30):** with the routines GIVEN, a recogniser trained only on audit verdicts reached 0.969 top-1 at w=50 (error-window rule 0.894; true-label ceiling 0.988), so the floor is reachable (P341-P345 passed). But the 3% misroutes still cost 5.7x the oracle error (P346 failed): in this world a wrong routine is catastrophic, so the floor must be computed from the measured e_w, not reference prices. Routine CREATION (the P5 fragmentation failure) is untested: X8.1b.

**T14 (what the freed budget buys; break-even).** *Status: PROVEN (accounting), CHECKED (X8.0); the "expanded vision" claim is EXPECTATION.* With a fraction `f` of cases automatic, compute per case is `f(c_A + q c_S) + (1-f)c_S`, so the same budget serves `M = c_S / (f(c_A + q c_S) + (1-f)c_S)` times as many cases, or leaves a fraction `phi = f(1 - c_A/c_S - q)` of compute for new problems. Example: `f`=0.9, `c_A/c_S`=0.05, `q`=0.05 gives `phi`=0.81 and `M`=5.3. A class needs `n*` slow practice cases, then training `c_T`, then saves `G` per case, so total savings turn positive after `R_be = n* + c_T/G` recurrences. The claim that freed budget IMPROVES performance on new problems is not implied by the accounting: it needs the new problems to be learnable with the extra compute (EXPECTATION; X8.2).

**T15 (doubt that cannot be memorised).** *Status: PROVEN for the X7.6 loss; the observed rates are CHECKED in X7.6.* In `util_loss`, a case whose label is redrawn uniformly over the `R`=4 classes each time it is seen has expected loss `-P_cls(1 - lam(R-1))/R + (CE term)` where `P_cls` is the total probability on class outputs; with `lam`=2, `R`=4 the coefficient of `P_cls` is `+5/4 > 0` in the loss, so the minimum is `P_cls` = 0 (all mass on abstain). The CE term acts on the class logits only (it is computed on `o[:, :R]`), so it cannot move the abstain probability. Hence on flagged cases the optimum is to abstain, and no label can be memorised. X7.6 shows this transfers to unseen contexts (novel abstain 0.67) but costs competence (seen abstain 0.34); how far it transfers is EXPECTATION, not part of the proof. X8.0 checks the proof on the real loss with free logits.

**T16 (a local rule suffices for the readout).** *Status: PROVEN for a linear-softmax readout; OPEN for deeper automatic paths.* For a softmax readout `p = softmax(W x)` trained by cross-entropy against the slow answer `t`, `dL/dW_{ij} = (p_i - t_i) x_j`: the product of the presynaptic activity `x_j` and a per-output error `(p_i - t_i)` broadcast from the audit. That is a three-factor rule (presynaptic, error, and a gate that is on only in audited cases), needing no backpropagation. The same holds for the last layer of any deeper path; credit through hidden layers needs a further local approximation (feedback alignment or similar [unverified for this setting]) and is OPEN. This is the sense in which the automatic path can live on a "biological substrate".

## 4. What the GPU stage measures (so it measures, not tests)

For each real model pair (the big model as sensors+Self, a small distilled model as `A`), and each task family, measure:

| Quantity | How | Used by |
|---|---|---|
| `c_S`, `c_A`, `c_T`, `c_R` | tokens/FLOPs/seconds per case, training cost of the small model | T11, T12, T14 |
| `lam` | fixed by the task's stated goal | T9-T12 |
| `e_S`, `e_A(n)`, `d(n)` | accuracy vs reference answers; the distilled model's learning curve | T11, T13 |
| `rho`, `Delta`, `m` | a stream with known shifts injected; detections counted | T12 |
| `a`, `e_r`, `e_w` | recogniser accuracy; error when the wrong routine fires | T13 |
| `f` | share of the stream that recurs | T14 |

The theory then predicts, BEFORE the run: whether the hand-off pays, `n*`, `q*`, the total cost `J`, and the capacity multiplier `M`. The GPU run passes if measured `J` and `n*` lie within the registered tolerance. A miss is a result about an assumption in section 5, not a tuning problem.

## 5. Assumptions, and where they are most likely to break

- **A1. The audit's truth is the slow path.** Disagreement with S only catches errors where A and S differ. A small model distilled from S shares S's mistakes. For the hand-off rule this is harmless (`e_A - e_S <= d` still holds), but (i) the combined system can never beat `e_S` through the audit alone (the Self's multi-sensor double-checking is what lowers `e_S`), and (ii) a shift that damages S and A together is invisible: that is `kappa` in T12, and `kappa` large makes the audit law blow up. The single largest threat on real LLMs; the GPU stage must measure `kappa` (the ratio of a shift's true error jump to its disagreement jump).
- **A2. Errors of A are independent of the audit draw.** Fine by construction when audits are random; an adversarial or periodic stream breaks it.
- **A3. Shifts are rare and abrupt (Poisson, `rho Delta`-detectable).** Gradual drift is detected late; the audit law then needs a sequential change-point test (EXPECTATION: same `sqrt` scaling with a worse constant).
- **A4. After a shift A is fully retrained at cost `c_R`.** Partial retraining or catastrophic interference (P5: one network carried across problems was 5x worse) would add a cost not in `Phi`.
- **A5. Class structure exists and is nameable from the case.** If the world has no recurring classes, `f` = 0 and the model reduces to X7.6.
- **A6. Costs are additive and measured in one currency.** True in the simulator; on real hardware latency and energy can disagree, so the currency must be chosen once and kept.
- **A7. `lam` is known.** The goal is given; learning the goal is outside this model.

**What the simulations can and cannot show.** The X8.x checks run the model's own assumptions in code, so agreement proves the algebra and the logic of the controller, not that real language models obey A1-A7. Only the GPU stage can confirm A1 (shared failures, `kappa`), A3 (shifts are abrupt), A4 (retraining is clean) and A5 (recurring, nameable classes); those are the assumptions to measure first, and a miss there is a finding about real models, not a bug in the theorems.

## 6. Proof-loop status

| Step | State |
|---|---|
| 1 Model and theorems T9-T16 written | done (this file) |
| 2 Closed forms checked by simulation (X8.0, P332-P340) | done 2026-10-30: 7 of 9 predictions passed as registered or on rerun; P334 and P335 failed as registered, both explained (multiplicity of the bar; omitted renewal factor), see `LOG.md` |
| 3 Substrate version: automatic path as a local-rule network on the P5 world with audits as teacher (X8.1) | queued |
| 4 Freed-budget claim: does the compute saved improve results on new problems (X8.2) | queued |
| 5 Real-model measurement plan (section 4) turned into a registered GPU protocol | after 3 and 4 |

Corrections to this file follow the project rule: never silently edited; a failed check gets a CORRECTION entry in `LOG.md`.
