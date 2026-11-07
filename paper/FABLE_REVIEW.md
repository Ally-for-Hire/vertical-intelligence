# Critical review of the curious-net project (Fable 5.1, 2026-10-22)

Scope: everything in `~/curious-net` (code, experiments, results, `paper/`, `LOG.md`, `CONSTANTS.md`) and the wiki domain `Knowledge-Wiki/domains/curious-net`. Read-only: no file other than this one was created or changed, no experiment was run, nothing was committed. Numbers quoted as "recomputed" were recomputed from `results/*.json|pkl` with read-only scripts; the rest are quoted from the project's documents.

Tags used below: **[verified]** = I read the code or recomputed the number; **[inference]** = my reasoning from verified facts; **[memory]** = from my own knowledge of the literature, not checked against a source in this session (treat exactly as the project treats its own `[unverified]`).

Context that colours everything: every result file carries a modification time between 01:00 and 08:06 on 2026-10-22 **[verified]**. The whole project, from the Boolean car world to the connectome worlds, is a one-day sprint. Nothing has been replicated on a second day, a second machine, or a second prior-fit seed. That is not a criticism of the work rate; it is the first caveat any reader should attach to every number.

---

## 1. Ideas and findings worth keeping, with their robustness

### 1.1 The headroom result (C9) — robust, but partly a statement about the baseline
Knowing the true inputs cuts the first-1000-tick error of an online MLP from 0.84 to 0.40 at 200 distractors (`expK_headroom.py`, recomputed: plain 0.837 ± 0.029, oracle 0.399 ± 0.025, n = 24) and the gain grows monotonically with distractors (0 / 10 / 50 / 200: oracle even slightly *worse* at 0 distractors, 0.457 vs 0.423, within noise) **[verified]**. Hyperparameters were tuned separately per arm on validation worlds (`expK_headroom.py:44-54`) **[verified]**, so this is fair. Robust. Two caveats: (a) it is a statement about an online, batch-1 Adam MLP with 32-128 hidden units; it measures how badly *that* learner handles irrelevant inputs, which is well known (sample complexity of dense regression grows with input count; sparse methods pay `s log B`); (b) the transformer headroom (0.794 -> 0.500) is against a plain transformer trained online at batch 1 for 1500 ticks with attention over up to 213 tokens (`tfm.py:44-87`) — an undertrained baseline, so part of the "headroom" is optimisation difficulty, not information **[inference]**.

### 1.2 The learned relevance prior (C10) — real effect, wrong description
A 2x32 tanh network over eight hand-built pair descriptors (`selfnn.py:19-57`), trained on the agent's own permutation-importance labels, applied as a soft input gate, gives +0.128 ± 0.012 over plain (oracle +0.209) at K = 100 on 48 held-out worlds (`expM`, recomputed) **[verified]**. The gain persists in ticks 1000-1500 (+0.090 vs oracle +0.116, recomputed), so the plain net has not caught up by the end of the run **[verified]**. Train/validation/test seed ranges (5000+, 1000+, 9000+) do not overlap; chain seeds (20000+, 30000+) are separate **[verified]**. The effect is real.

What it is **not**: it is not "learning what affects what". Three pieces of the project's own evidence say the gate learns *which inputs are predictively useful to a small-data MLP*, not the causal graph:
- Ground-truth (causal) labels train a *worse* gate than own labels: `nn_oracle_soft100` +0.101 vs `nn_soft100` +0.128 (`expM`, recomputed) **[verified]**.
- Own-label positive rate is 0.10-0.20 against a true-parent rate of 0.06-0.07 (`expN.txt`, `expS2.txt`, recomputed 0.067) **[verified]**; the extra positives are noisy copies and other proxies (`worlds.py:91-92`: a "copy" distractor is `core_new + 0.3*noise`, i.e. a legitimately predictive input).
- On connectome worlds the prior gives 95-110% of the "oracle" gain while its AUROC for true direct parents is 0.56 / 0.63 (`q19.pkl`, recomputed) **[verified]**.

So the honest one-line claim is: *a small network can learn, from ~10 earlier problems, a calibrated map from cheap early statistics to "attend to this input", and that map transfers across worlds from the same generator.* That is useful and should be kept, but the words "what affects what" and "causal" should be removed from the headline (see 2.1 and 6.1).

### 1.3 Soft gating beats hard masking — keep, and understand why
`expL2` (recomputed): hard threshold 0.5 gives +0.114 with precision 0.62 / recall 0.70; soft gating +0.149 with precision 0.19 / recall 0.61 **[verified]**. Under Adam, scaling input `i` by `p_i` (`nn.py:19-23`) scales its first-layer gradient by `p_i`, which Adam's per-parameter normalisation undoes; the effective weight `p_i * W1[i]` therefore moves at rate `~p_i * lr` and starts at `p_i` times its initial size **[inference, mathematical]**. The "relevance prior" is thus a **learned per-input learning rate and initial scale**. This (a) explains why false negatives are cheap (a true parent with p = 0.2 still learns, five times slower), (b) explains why the gate can beat the hard oracle mask (1.4), and (c) places the method squarely next to Meta-SGD / learned per-parameter learning rates and ARD (section 4). It is a good idea; it is not a new one.

### 1.4 The recursive loop does not collapse (C13, exp N2 / O) — robust in this setting, and the reason is structural
Eight chains x 150 generations: +0.142 ± 0.007 (g5) to +0.166 ± 0.001 (g150), chain SD falling from 0.019 to 0.003 (recomputed) **[verified]**. The explanation in the ledger is correct and important: every generation retrains a fresh network on real data; only a 2x32 gate carries over; the labels are a verifier output on real data. This is a design lesson worth keeping (small transferable state + fresh real data + soft gates = no collapse channel). It is also known: Gerstgrasser et al. 2024 show collapse is avoided when real data keeps accumulating **[memory]**; cite it.

### 1.5 Three negative results that should be kept as settled
- **Quantum encoding (C16, `q9_quantum_order.py`)** — clean, correct, and the conclusion ("a hypothesis about the world, not a learner improvement; wins by thrift only on QQ-structured data") is exactly right **[verified]**. Close it.
- **Emotion = structured randomness (C17, exp S/S2, q10c/d)** — the population never beats the single prior; proposal randomness only insures against an inverted prior **[verified]**. The reinterpretation as an exploration dial is reasonable. Close it, with the caveat in 2.9.
- **Curiosity vs random for independent immediate action effects (C3)** — correct and expected; random actions are a near-optimal design there.

### 1.6 Q12 ball-drop agent — the right *shape* of test for the user's vision
Hypothesis library + BIC + a 3-sigma claim check + a lack-of-fit test for "none of my laws fit" (`q12*.py`) is the right micro-benchmark for "skeptical, knows its limits". The resolution lesson from Q12b/c ("you can only detect a gap bigger than your noise relative to your library's flexibility") is a genuine, transferable limit statement and should be the template for the new direction. Caveats in 2.6.

### 1.7 FlyWire statistics (C22, Q18) — real data, sound measurement
`results/q18_flywire_stats.json` **[verified]**: 138,639 neurons, 54.49M synapses, in-degree median 9 at >= 5 synapses, effective-parents-50% median 3, 90% median 9. Keep as a reference table. Its implication (the generator's 1-3 parents matches the dominant inputs only) is correctly stated.

---

## 2. Mis-steps and methodological problems

### 2.1 "Share of the oracle gain" is not a bounded quantity — the central yardstick is unsound
The "oracle" arm is a *hard mask to the true direct parents*. It is beaten by the learned soft prior in two experiments: pure-interaction worlds, prior +0.090 vs oracle +0.080 (paired -0.010 ± 0.004, `expS2.pkl` recomputed), and connectome worlds, connectome-prior +0.108 vs oracle +0.098 (`q19.pkl`) **[verified]**. Hard masking removes predictive non-parents (copies, proxies, children) and forces a small-data MLP to rely on parents that are weak predictors when there are hidden variables. Therefore "50-60% of the oracle benefit" (UNDERSTANDING §2.2, PAPER abstract, C10, C11, C14, C22, C29) is a ratio to a reference that is neither an upper bound nor the Bayes-optimal gate. Report absolute gains with SEs and, if a reference is wanted, use the best of {oracle-all, oracle-top-k, soft oracle with p = 1 on parents and a tuned floor elsewhere}, and report the ratio's SE.

### 2.2 The no-experience baselines are straw men; the "experience" may be mostly a calibration constant
- The "fixed screen" is a **hard** Bonferroni mask over `n_in * n_out` tests on **one** descriptor (`selfnn.py:116-117`) **[verified]**. It is handicapped by the gating rule (hard) that the project itself showed is fragile (1.3), not by lack of experience. A fair no-experience competitor is a *soft* gate `p = sigmoid(a * f1 + b)` with `(a, b)` tuned on the validation worlds (two numbers), or `p = 1 - p_value`. Nobody has run it. If it recovers most of +0.128, the "experience across problems" narrative reduces to "two calibration constants".
- The ridge-logistic prior has nine parameters and reaches +0.087 at K = 3 and saturates by K = 10 (`expM`, recomputed) **[verified]**. The amount of "experience" being used is ~10 worlds x ~1000 pairs of labelled descriptor vectors to fit nine (or ~1,300 for the 2x32 net) parameters. The saturation is the Bayes error of these eight features for the label, not a property of experience **[inference]**. "Gets faster and faster with experience" (user) is therefore true only between K = 3 and K = 10 and then false by construction; the ledger says "saturates" but the understanding guide still leads with the experience story.
- Group-lasso was tuned on the first-1000-tick error (`expP*.py`) and applied as a proximal step at every tick with a fixed strength; a *decaying* or lr-coupled strength, or a lasso/OMP screen on the first-100-tick regression (p > n, so lasso) was not tried. Input dropout as a sparsity baseline is weak by construction. Missing strong no-experience baselines: SIS/sure-independence screening with soft weights, knockoff/stability-selection screens, Granger/VAR-lasso, random-forest importance on the 100-tick window, Meta-SGD-style online per-input learning-rate adaptation (hypergradient) **[memory for the names]**.
- Missing meta-learned baselines are acknowledged (Q2d/e) but the "REFUTED in-distribution" verdict on C14 was reached without them; the project's own raw-window CNN is the only alternative learned design.

### 2.3 Statistics
- **Unexplained between-experiment offset.** Same protocol, same 24 test worlds, same plain baseline: exp N gives +0.156 ± 0.005 at generation 10 (8 chains, chain SD 0.0135), exp O "real" gives +0.128 ± 0.008 (5 chains) (recomputed) **[verified]**. That is roughly 2 chain-SDs. The ledger reports within-experiment SEs of ± 0.001-0.002 at late generations, which is 15-30x smaller than this replication gap. Until the offset is explained (different training-world seed ranges are the only known difference, `expN_transformer.py:58` vs `expO_breakpoints.py:40`), the late-generation SEs are not credible as uncertainty on the effect.
- **"Hard mask decays" (C13, UNDERSTANDING) is not established.** Recomputed per-chain: +0.063 ± 0.016, +0.060 ± 0.012, +0.056 ± 0.012, +0.049 ± 0.013 (n = 5 chains) **[verified]**. The trend is within one SE at every step. The label-rate halving (0.057 -> 0.030) is the robust diagnostic; the gain decay is not. The ledger row gives no SE for these numbers.
- **Single prior fit per K** (`expM`, `expN`: `NNPrior(seed=K)`) — acknowledged (Q1) but K = 1, 3 numbers are still quoted in text.
- **Autocorrelated samples in the label test.** `perm_labels` (`selfnn.py:146-163`) runs a paired z-test over 300 *consecutive* ticks of a dynamical system with AR(1) distractors at 0.5-0.95 persistence (`worlds.py:48`) **[verified]**. The test's nominal Bonferroni level is meaningless; the positive-label rate of 2-3x the true rate is partly this. This matters for the oracle-teacher direction: the labels are the "truth" the loop teaches itself.
- **Mastery cadence.** `q24_teacher_student.py:65-68` checks mastery only every 5 interactions **[verified]**, so "adaptive 30 (IQR 30-30)" is quantised; the true minimum may be lower than 30 and the IQR is an artefact.
- **No majority-class baselines in the claim-checking experiments.** In the q25b worlds the true-claim rate is 0.24, so "say every claim is false" scores 0.76 (recomputed from the generator) **[verified]**. `prior_only` 0.864, `verify_all` 0.79, and the systematic-teacher `credulous` 0.807 are all within 0.03-0.10 of that. The ledger quotes none of this.

### 2.4 The "curious" design result in Q12 is an artefact
`q12_law_discovery.py:56-57`: for fewer than three drops the "curious" strategy returns a **random** height, alternating between log-uniform(1, 100) and log-uniform(10, 100) **[verified]**. At n = 3 all three drops are therefore random with a high-height bias, so "curious gives the best claim-check accuracy at n = 3 (0.983 vs 0.931 random)" (C19, LOG iteration 8) compares two random designs. The reported advantage of curiosity at small n is not evidence about curiosity.

### 2.5 Leakage of the noise model into the claim checker
`q12_law_discovery.py:83` and `q12b_out_of_library.py:74,118` use the generator's `NOISE_REL`/`NOISE_ABS` constants inside the agent's tolerance and inside the post-expansion lack-of-fit test **[verified]**. The agent is told its own measurement noise. The 0.93-0.999 verdict accuracies are therefore upper bounds for an agent that must estimate its noise (Q12b does estimate it from replicates for the *flag*, but not for the verdict tolerance).

### 2.6 Q25b/Q30 experimental "intervention" is idealised
`q25b_fact_checking.py:35-56`: an own experiment re-creates the world from its seed (`sample_world(seed)`), i.e. the student can reset the world to its initial state and run a clean pulse **[verified]**. Real agents cannot reset; the 79% single-experiment reliability is optimistic for the same experiment run in the live stream. Also, the systematic teacher's errors are placed with knowledge of the truth table (`claim[jstar,:] = True; claim[:,astar] = False`), which makes "disagreement with the prior" an unusually clean signal.

### 2.7 Generator artefacts the results may depend on
- Every core variable always includes itself as a parent (`worlds.py:19,112`) and most kinds are strongly persistent (`_f`: 0.95 s + ..., 0.9 s + ...) **[verified]**. Predicting the *change* removes most of this, but the descriptor `f4` (association after removing own state) and the "self flag" feature give the prior an easy, always-correct positive. Part of every gain is "attend to yourself".
- The `copy` distractors are the only correlated distractors; there are no *children* of core variables among the observed non-core inputs, no common causes, no confounders. A gate that learns "copies are useful" is rewarded here and would be punished in a world where copies are collider-like.
- The `sin` distractor is a deterministic sinusoid, trivially predictable from itself; `wn` is pure noise. Distractor "difficulty" is bimodal.
- Delayed-effects worlds are scored against lag-t truth (noted in LOG). The "oracle" there is partly wrong, so the 57% share is against a wrong reference.
- Connectome subgraphs are grown greedily by total connection weight from a hub seed (`q19_prepare.py:42-55`) **[verified]**: they are the densest local neighbourhoods, which is why the true-parent rate is 0.25-0.33 and the headroom is small. A degree-matched random control (Q19b) is essential before any claim that "real wiring" matters.

### 2.8 Evaluation window dilution
Gains are averaged over ticks 0-1000 but the gate switches on at t = 100 (`selfnn.py:113`), so 10% of every "first-1000" gain window is identical across arms **[verified]**. The "ticks 100-1000" gains are ~10% larger (recomputed: `nn_soft100` +0.142 vs +0.128). Not wrong, but state it.

### 2.9 The q10c/q10d success metric penalises larger budgets by construction
Success requires the true pair to be *the single best-scoring* proposal *and* to pass a Bonferroni threshold that tightens with the number of proposals (`q10c_proposals.py:68,90`) **[verified]**. The "ceiling ~0.46 even with exhaustive search" is partly this metric, not only the 500-tick test's power.

### 2.10 Over-claims in prose (examples)
- UNDERSTANDING §2.4: "With zero experience it is exactly a plain network. Not approximately: bit for bit." The parity holds because `mode="plain"` and `mode="prior"` with `w=None` execute the same code path with a mask of ones (`selfnn.py:95-101,121-124`) **[verified]**. It is a tautology of the design, not a finding, and every meta-learner with an identity-initialised modulation (MAML at its init, a zero attention bias) has the same property. It does not "let any later gain be attributed to experience" more than any other ablation would.
- UNDERSTANDING Q19 appendix / C29: "about as much as knowing the true parents" — see 2.1.
- UNDERSTANDING "Does the learned gate survive problems that look different": "kept roughly half to 60% ... in every case" — true for the descriptor prior; the CNN prior kept 37% and 42% in two families (`expQ`, recomputed) **[verified]**.
- C25 "the adaptive teacher is optimal here" — mastery cadence (2.3) means this was not measured at the resolution needed to say "optimal".

---

## 3. Documentation errors and stale claims (every inconsistency found)

| where | problem | evidence |
|---|---|---|
| `CONSTANTS.md:3-5` | "0 undeclared literals ... audit fails if any numeric literal in src/ is not routed through the registry" | Running `src/audit_constants.py` now reports **76 undeclared numeric literals** (worlds.py 40, selfnn.py 15, nn.py 9, seqnet.py 8, tfm.py 4) **[verified]**. The new line never went through the registry; W = 100, last = 300, floor 1e-3, 2x32/400 steps, hidden 32, ridge 1, Bonferroni 1/n_in, SEL = 200, sigma = 1, K = 8 are all unregistered. This violates the user's standing rule (memory: "unbacked numbers must be declared as problems"). |
| `paper/theory.md:30` (T4) | Still states "S is a reflected random walk (g stays O(1))" with "Status: mechanism verified" | `LOG.md` (Fable review item 3) records that this is false: a zero-drift reflected walk grows like sqrt(t). theory.md was never corrected. |
| `paper/PAPER.md:6` abstract | "Whether the recursive loop improves, plateaus or degrades across generations is under test (C13)" | exp N2 and O are done (ledger C13). |
| `paper/PAPER.md:32` §5 | "no meta-learned neural baseline yet; ... evaluation worlds drawn from the training distribution; no out-of-family test yet" | exp P/P2 (learned CNN prior, group-lasso, dropout) and exp Q (six families) exist and are cited in §4 of the same file. |
| `paper/PAPER.md:35` | "Next: Q1-Q17" | queue runs to Q31 + Q12d + QO1-4. |
| `paper/LEDGER.md:4` | "Last updated: 2026-10-22 (loop iteration 0)" | twelve iterations later. |
| `LEDGER.md` C23 | "UNTESTED ... Q19" | Q19 is done; its AUROC 0.56/0.63 answers C23 **negatively** (no edge recovery). C23 should be NOT SUPPORTED (in this setting), cross-referenced to C29. |
| `LEDGER.md` C27 | "exp N2 (a different loop: priors from own labels) pending" | N2 finished at iteration 2b. |
| `LEDGER.md` C29, `UNDERSTANDING.md` Q19 appendix, `LOG.md` iteration 12 | "true-parent rate is 0.33" / "parents are 33% of inputs" | The 40 held-out worlds have mean truth rate **0.251** (`q19.pkl`, recomputed). 0.33 is the training-subgraph figure printed by `q19_connectome_world.py:81`. |
| `LEDGER.md` C13 | "HARD mask ... decaying" | within 1 SE at every step (2.3); no SE given in the row. |
| `LEDGER.md` C19 | "curious gives the best claim-check accuracy at n = 3" | artefact (2.4). |
| `LEDGER.md` C4 | SUPPORTED (toy) | `LOG.md` Fable review item 6: the ranker's top variable is the lagged-score distractor, so "self-ranking finds the task variables" is partly trivial. Status not revised. |
| `LEDGER.md` C7 | PARTIAL on 1 seed | fine as PARTIAL, but UNDERSTANDING/wiki quote the 37 vs 72 bits/tick number without the "1 seed" caveat (wiki project-state, "core4 handles the drag-change example"). |
| `LEDGER.md` C1-C8 generally | statuses rest on the Boolean/count-table line whose evaluation flaws (fixed start, respawn during eval, training-track evaluation, compute units) are documented in LOG but not reflected in statuses | LOG 2026-10-22 "review by Fable 5.1 of v2", items 1-7. |
| `paper/QUEUE.md` Q19 row | columns are shifted: the "item" cell holds "DONE ... Q19b" and the "why" cell holds the original item text | read the row. |
| `paper/QUEUE.md` Q20 | "todo" | Q19 already trained a connectome-statistics prior vs a synthetic one and compared them (C29). Q20 is done or must be redefined. |
| `paper/QUEUE.md` Q2 | "(e) Reptile-style init on a FIXED layout" still open, and C14 says "REFUTED in-distribution" | a verdict was reached before the queued competitor ran. |
| `UNDERSTANDING.md` §5 | "The recursive-improvement test (C13) is not finished." | finished. |
| `UNDERSTANDING.md` §2.2 | "A plain statistical screen with no experience recovers almost none" | 4% on the diverse distribution, but 22% on the expL2 distribution (`expL2`, recomputed: screen +0.058 of +0.259); the sentence should say "on the diverse distribution". |
| `UNDERSTANDING.md` §2.3 / C11 | "experience can hurt at first" | distribution-dependent: K = 1 gives +0.149 on expL2 and -0.14 on expM. Say so. |
| `UNDERSTANDING.md` "Correction and result" (fly-like) | "the prior learned from earlier problems recovered about 64% of the full gain, the same as in the simpler worlds" | fine, but "full gain" is the hard oracle-all mask (2.1). |
| `LITERATURE.md` Shiu et al. | "journal venue not confirmed" | Nature 2024 (vol. 634, pp. 210-219) **[memory]**; confirm and fix. |
| `LITERATURE.md` Wang & Busemeyer | "'70 surveys' not confirmed" | the 70-survey test is Wang, Solloway, Shiffrin & Busemeyer, PNAS 2014 **[memory]**; cite that instead. |
| `LITERATURE.md` "Zanichelli et al., State of Brain Emulation Report 2025, arXiv:2510.15745" and "Lebedev & Khrennikov arXiv:1811.00045" | I could not confirm these identifiers from memory | treat as [unverified] until the arXiv pages are read. |
| wiki `index.md` | describes the project as "Curious Boolean-to-quantum-state network ... tested on a car-on-tracks world" | the project has been the self-learning NN line for 11 of 12 iterations. |
| wiki `curious-net-project-state.md` | last substantive section is "iterations 1-2b" | P, P2, Q, R, S, S2, q10, q12, q19, q25b, q30, q9 absent. |
| wiki `vision-and-roadmap.md` "Staged roadmap" | "Now (in progress): transformer base ... Then neural meta-learning baselines" | stale by ten iterations; the redirect paragraph at the bottom contradicts the "Now" paragraph. |
| `results/expS.json` | 44-byte file left from the failed JSON serialisation (LOG iteration 10) | harmless but misleading next to `expS.pkl`. |
| `experiments/expN2_analysis.py:25` and `make_figures.py:95` | hard-code the reference "+0.148" / K = 100 non-recursive line from exp N1 | if N1 is rerun the figure lies silently. |

---

## 4. Literature problems

### 4.1 Citations that look fine
The verified rows in `LITERATURE.md` match my memory on venue, year and content for: Herbster & Warmuth 1998; Krichevsky & Trofimov 1981; Ortega et al. 2019 (arXiv:1905.03030); Mikulik et al. NeurIPS 2020; Ke et al. 2019 (arXiv:1910.01075); Shumailov et al. Nature 631 (2024); Houlsby et al. 2011; Guo et al. ICML 2017; Rissanen 1978; Nichol et al. 2018; McClelland et al. 1995; Schmidhuber 2010; Brunton et al. PNAS 2016; Udrescu & Tegmark 2020; Ellis et al. PLDI 2021; Zelikman et al. 2022; FlyWire (Dorkenwald et al., Nature 2024, 139,255 neurons); Lin et al. 2024; Scheffer et al. eLife 2020; Lappalainen et al. Nature 634 (2024); Sandberg & Bostrom 2008; Warstadt et al. 2023; Eldan & Li 2023; Hermann et al. 2017; Chevalier-Boisvert et al. 2019; Lazaridou et al. 2017; Zhu 2015; Kirby, Cornish & Smith 2008; Saffran et al. 1996; Busemeyer & Bruza 2012; Wang & Busemeyer 2013 **[memory]**. The project's habit of marking "abstract only" is good and should continue.

### 4.2 Mischaracterisations
- Ke et al. 2019 is cited as "the closest published neighbour of the relevance prior". It is not the closest. The closest are the **amortised structure-inference** papers that train one network across many synthetic graphs to output edge probabilities from data: Lorch et al., "Amortized Inference for Causal Structure Learning" (AVICI, NeurIPS 2022) and Ke et al., "Learning to Induce Causal Structure" (CSIvA, ICLR 2023) **[memory]**. The project's prior is a tiny, descriptor-based AVICI. Also directly relevant: prior-data fitted networks (Müller et al. 2022; TabPFN, Hollmann et al. 2023) — "experience across synthetic problems, used in-context on a new one with no retraining" is exactly their setting **[memory]**.
- "Meta-learning is amortised Bayesian inference" (UNDERSTANDING §3) is used to explain why NN meta-learners match the wrapper, but the same fact is the ceiling argument against the user's "faster and faster with experience": a meta-learner converges to the Bayes-optimal in-family prior and then stops improving. The guide states the first use and not the second.
- Model collapse (Shumailov) is cited as "the main risk for C13", but the project's own finding (no collapse when real data flows and the carried state is small) is the already-published result of Gerstgrasser et al. 2024 ("Is model collapse inevitable?") **[memory]**; the project's "surprise" at exp O is a rediscovery.

### 4.3 Known prior work that already answers project questions (all **[memory]**, verify before citing)
| project question | prior work |
|---|---|
| which inputs matter, with a controlled false-discovery rate (C2, H5, the screen) | lasso (Tibshirani 1996); sure independence screening (Fan & Lv 2008) = the project's "fixed screen"; stability selection (Meinshausen & Bühlmann 2010); model-X knockoffs (Candès et al. 2018) for FDR-controlled selection; ARD (MacKay 1992, Neal 1996) for learned per-input relevance inside a network |
| the soft gate as a learned per-input rate (1.3) | Meta-SGD (Li et al. 2017), hypergradient descent (Baydin et al. 2018), learned optimisers (Andrychowicz et al. 2016, already queued) |
| structure from time series with distractors and lags (worlds with delays, C23) | Granger causality, VAR-lasso (Basu & Michailidis 2015), PCMCI (Runge et al. 2019), DYNOTEARS (Pamfil et al. 2020), transfer entropy |
| why activity does not recover wiring in a recurrent network with hidden units (Q19, AUROC 0.56) | Das & Fiete 2020, Nature Neuroscience, "Systematic errors in connectivity inferred from activity in strongly recurrent networks"; GLM coupling inference (Pillow et al. 2008) |
| choosing experiments where hypotheses disagree (Q12 "curious") | T-optimal design for model discrimination (Atkinson & Fedorov 1975; Box & Hill 1967); Bayesian experimental design (Lindley 1956; Chaloner & Verdinelli 1995); for structure: Tong & Koller 2001, Murphy 2001 (active learning of Bayesian networks) |
| "none of my hypotheses fits" | classical lack-of-fit tests (Draper & Smith) — already used; Bayesian model criticism / posterior predictive checks (Gelman et al.); Bayes factors with an "other" model |
| calibration and abstention (C15, QO2) | Platt scaling; Guo 2017 (cited); deep ensembles (Lakshminarayanan et al. 2017); MC dropout (Gal & Ghahramani 2016); calibration under shift (Ovadia et al. 2019: it degrades); selective prediction (Geifman & El-Yaniv 2017, SelectiveNet 2019); conformal prediction (Vovk, Gammerman & Shafer 2005; Angelopoulos & Bates 2021); OOD detection baselines (Hendrycks & Gimpel 2017) and its impossibility results (Fang et al. 2022, "Is OOD detection learnable?") |
| an oracle teacher producing overconfident students (QO1-QO3) | Gekhman et al. 2024 "Does fine-tuning LLMs on new knowledge encourage hallucinations?"; Kang et al. 2024 "Unfamiliar fine-tuning examples control how language models hallucinate"; R-Tuning (Zhang et al. 2023: teach the model to say "I don't know" for what it cannot ground); Kadavath et al. 2022 "Language models (mostly) know what they know"; Lin, Hilton & Evans 2022 (verbalised uncertainty) |
| curriculum / teaching (C25) | Bengio et al. 2009 (queued); teacher-student curriculum learning (Matiisen et al. 2017); knowledge distillation (Hinton et al. 2015, queued); Goldman & Kearns teaching dimension (cited) |
| recursive self-improvement without collapse (C13) | Gerstgrasser et al. 2024 (above); Bertrand et al. 2023 "On the stability of iterative retraining" |
| iterated learning (C27) | Kirby, Tamariz, Cornish & Smith 2015 (compression vs expressivity) |
| quantum cognition critique (C16) | Pothos & Busemeyer 2013 (BBS target article and commentaries) |

### 4.4 The user's "neural networks cannot do this" belief
The ledger correctly refutes it in-distribution. It should also say plainly that the project's own learner *is* a neural network trained across problems (a 2x32 tanh net), and that the remaining claim, "exact parity at zero experience", is satisfied by any zero-initialised modulation, including an additive attention bias in a standard transformer (`tfm.py:37`). There is no architectural novelty claim left to defend; the defensible contribution is the measurement protocol (parity + headroom + own-label loop + breakpoints) and its negative results.

---

## 5. Better targets, especially for the oracle-teacher + calibrated-skepticism direction

### 5.0 The first thing to notice: the project already ran the oracle-teacher experiment, and the oracle lost
`expM` arm `nn_oracle_soft` is a student taught the *ground-truth* causal graph by an exact teacher; it is beaten by the student that learned from its own experience (+0.101 vs +0.128 at K = 100, recomputed) **[verified]**. Two lessons for QO1: (1) "objective truth" as the teacher defines it (direct parents) is not the same as what the student needs (predictive relevance at its data budget); the oracle must teach in the student's terms or the student must be allowed to reweight the lesson; (2) an oracle-taught student whose lessons are exact has **zero** residual uncertainty about the taught facts and therefore no mechanism for skepticism later. Q25's "credulous student wins" (`q25_noisy_teacher.py:18`, `se = 1e-6`) is the same point from the other side: after one wrong utterance that student is confidently wrong (posterior on the true concept at e^-13.8) until outvoted **[verified]**. The mastery metric hides this; a calibration metric would not. The new direction therefore has to separate three things the project currently conflates: *knowing the facts*, *knowing the boundary of the taught set*, and *knowing the reliability of a source*.

### 5.1 Concrete experiments (each with prediction and failure condition)

**E1. Calibration of the taught student, in and out of distribution (QO2).**
Student: the descriptor prior (ridge-logistic and 2x32 net) trained on oracle labels, and the same trained on own labels, K = 100. Test: reliability diagrams, ECE, Brier, and accuracy-coverage curves of `P(i affects j)` on (a) 48 standard worlds, (b) the six exp Q families, (c) pure-interaction worlds, (d) the 40 connectome worlds. Also conformal prediction sets (split-conformal on 24 calibration worlds, coverage measured on the rest).
Prediction: the logistic prior is near-calibrated in-distribution (ECE < 0.05); the 2x32 net is overconfident (ECE 0.05-0.10); under shift ECE rises with the AUROC drop and is worst on pure-interaction and connectome worlds; conformal coverage holds in-distribution (0.90 ± 0.03) and fails under shift.
Failure condition: if ECE stays < 0.05 under every shift, "knowing the edge" is easier than feared and E2 can be skipped.

**E2. Knowing the edge of what it was taught (the "realistic expectations" test).**
Fit a novelty score on the taught worlds' descriptor distribution: (i) k-NN distance in standardised descriptor space, (ii) disagreement of an ensemble of five priors with different seeds, (iii) the mean entropy of the prior's outputs. Test AUROC of each score for "this world is from a family I was taught" vs the four out-of-family sets.
Prediction: ensemble disagreement AUROC 0.7-0.8 for pure-interaction and connectome worlds, ~0.55 for the mild exp Q shifts.
Failure: AUROC ~0.5 everywhere means the eight descriptors do not carry family identity; then edge detection needs the raw windows (SeqPrior) or a generative model of the descriptors.

**E3. Three-way claim verdicts with abstention (C15, QO3, Q7).**
Use the Q12 agent after oracle teaching (library = truth) and present claims in three classes: in-scope true, in-scope false (factors 0.9-1.1 as well as 0.4-2.5), and out-of-scope (quantities the library cannot predict: bounce height, horizontal drift, a different object). Required answers: consistent / inconsistent / no-connection. Metrics: 3x3 confusion, false-accept, false-reject, abstention on out-of-scope, and ECE of the consistency probability.
Prediction: near-perfect on factor-2 claims; false-accept rises sharply for 0.9-1.1 claims at 3% noise (resolution limit); abstention on out-of-scope is 100% if "no connection" is implemented as "the claim's variables are not in my model", and ~0% if abstention is only triggered by lack-of-fit.
Failure: if the agent abstains on in-scope claims near the noise floor, that is the correct behaviour and the metric must reward it (selective accuracy, not accuracy).

**E4. Contradicting data after oracle teaching (the planet change, done properly).**
Teach free fall with g = 9.8 exactly. Then stream drop data from g in {9.8, 8.8, 5, 1.6} with 3% noise. Measure (a) how many samples until the student declares "the taught law no longer holds here" (sequential Bayes factor or fixed-share posterior between "teacher right + noise" and "regime changed"), (b) the false-declaration rate on g = 9.8, (c) whether a *single* contradicting claim (not data) is rejected while *consistent contradicting data* is accepted.
Prediction: detection in 3-5 samples for g = 5, ~15-30 for g = 8.8, never for changes below the resolution; false alarms at the chosen alpha. (c) is the operational definition of "healthy skepticism": claims are weighed by source reliability and prior, data by likelihood.
Failure: a student that either never revises (oracle trust absolute) or revises on the first sample (no prior) — both are miscalibration and both are likely with the current Bayesian updates unless a source model is added.

**E5. Does an oracle-taught student over-trust a non-oracle?**
After oracle teaching, present the same claim types from a "stranger" source with 20% systematic errors. A student trained only against an oracle has no source variable and will treat the stranger as an oracle.
Prediction: credulous accuracy = stranger accuracy (0.80); with a source-identity feature and a handful of verified stranger claims it recovers to ~0.9. This is the demoted Dawid-Skene question in its simplest form and it cannot actually be dropped: without *some* source model, "healthy skepticism" is undefined.

**E6. Re-run the gate experiments with the oracle-label student as the main arm** and with the fair soft-screen baseline (2.2). The oracle-direction headline number must be "oracle-taught student vs own-taught student vs calibrated no-experience screen", all with absolute gains and ECE.

**E7. Curriculum ordering from the oracle.** For the batch-fitted descriptor prior, order cannot matter; only an *online* student (e.g. the transformer gate refit every 5 scenarios) can benefit. Prediction: easy-to-hard by distractor count gives no measurable gain at K >= 30. Run only if an online student is adopted; otherwise drop.

### 5.2 What to drop or deprioritise
- C16 quantum, C17 emotion populations, Q9b, Q10e: closed; do not spend more compute.
- Q30b (correlated Dawid-Skene) and Q24b: demoted by the user; keep only E5's minimal source variable.
- Q19b connectome edge recovery: Das & Fiete-type limits make edge recovery from 50 observed of 300 recurrent neurons a losing proposition without interventions; keep the connectome worlds as a *shift* test (E1/E2), not as a wiring-recovery test.
- Q26 iterated chains, Q11 part/sub-part spawning, Q15 scaling law, Q4 BN comparison, Q5/Q3 legacy count-table items: park; they do not serve the new direction.
- Q6 closed-loop experiment policy: keep, but only as the verifier inside E3/E4 (an experiment that checks a claim), not as a general curiosity study.
- The Boolean/count-table line (C1-C8): archive with statuses marked "legacy, unreviewed" rather than SUPPORTED.

---

## 6. Other matters, including limits of the mathematics the user should understand

6.1 **Identifiability.** From observational time series with hidden variables, direct causes are not identifiable in general; proxies of a parent are statistically indistinguishable from the parent at finite samples. The project's own data show it (AUROC 0.56-0.73 for true parents in Q19/S2; copies labelled positive). A gate that improves prediction is compatible with learning the wrong graph. "Learns what affects what" needs interventions and a structure learner; the project has neither in its main line.

6.2 **The prior's ceiling is an information ceiling.** The gate's input is eight statistics of a 100-tick window. Its best possible accuracy is the Bayes error of those statistics for the label; more worlds cannot move it. The only ways up are more informative inputs (interventional pulse responses, longer windows, raw windows with a bigger model) or a different label.

6.3 **Bayes-optimal meta-learners stop improving.** In-family, a meta-learner converges to the optimal prior for the family (Ortega; Mikulik). "Faster and faster with experience, indefinitely" is impossible within a fixed family; it requires the family itself to keep growing (open-ended curriculum), which is a statement about the teacher, not the student.

6.4 **Calibration is distribution-relative.** A model calibrated on the taught distribution carries no guarantee under shift (Ovadia et al.); conformal coverage is marginal and requires exchangeability, which time-series data within a world violate (across worlds is fine). There is no OOD detector that works for every shift (Fang et al. 2022). "Never wrong without knowing it" is therefore unattainable in principle, not just in practice; the attainable target is: calibrated inside the taught support, abstaining outside a *detectable* boundary, with the resolution statement of Q12c attached.

6.5 **Resolution.** Q12c's result is general: with noise sigma and a library able to imitate alternatives to within delta, no test can flag a gap smaller than ~delta/sigma per sqrt(n). An honest agent reports its resolution, not a yes/no.

6.6 **Self-training is information-free.** A student trained on labels produced from its own outputs cannot gain information about the world (data-processing inequality); exp O's "synth" arm did not collapse because the *labels were computed on real data by a fresh network*, which is new information every generation. If the weights themselves are trained on generated data (Q29b), expect the standard collapse.

6.7 **The oracle teacher and skepticism are in tension by construction.** A perfect teacher drives the student's posterior to a point mass; skepticism requires residual uncertainty or a source model. The user should decide which: either the curriculum includes lessons *about sources and limits* (meta-knowledge taught by the oracle), or the student keeps its own verifier (experiments) and a source variable. Teaching facts only will produce a confidently credulous student; the project's Q25 numbers already show it.

6.8 **Generation-count efficiency.** The user's metric is fine as long as compute is reported alongside (C20 says so). But note that the gate experiments use ~10 worlds of "experience" to fit nine parameters; the metric flatters any small learned component.

6.9 **Engineering notes.** `tfm.py:83` `err[t] = float(loss)` on a tensor with grad triggers warnings (`results/expN.txt`) and should be `loss.item()`; `NNPrior.fit` sets `torch.set_num_threads(4)` inside worker-spawned fits while `common.py` pins BLAS to 1 (harmless but inconsistent); `expN2_analysis.py` and `make_figures.py` hard-code reference numbers; `results/expS.json` is a stub.

---

## Prioritised top-10 actions

1. **Replace the "share of oracle" yardstick.** Report absolute gains ± SE everywhere; if a reference is needed, define a proper upper bound (best of oracle-all / top-k / soft oracle) and give the ratio an SE. Rewrite C10, C11, C14, C22, C29, UNDERSTANDING §2 and PAPER accordingly (2.1).
2. **Run the fair no-experience baseline:** a soft gate from the single descriptor `f1` with two constants tuned on validation worlds, plus a lasso/SIS screen on the 100-tick window. If it recovers most of +0.128, change the headline from "experience" to "soft gating of a calibrated screen" (2.2).
3. **Resolve the exp N vs exp O offset** by re-running the "real" chains with exp N's seeds, and stop quoting ± 0.001 late-generation SEs until between-experiment replication agrees; add SEs to every "decay" statement and downgrade "hard mask decays" to "label rate halves; gain trend within noise" (2.3).
4. **Fix the documentation table in section 3** (CONSTANTS.md audit claim, theory.md T4, PAPER abstract/§5/§6, LEDGER header/C23/C27/C29 0.33, QUEUE Q19 row/Q20, UNDERSTANDING §5, wiki index/project-state/roadmap).
5. **Register the new line's constants** (W, last, floor, prior architecture and steps, SEL, sigma, K, Bonferroni levels) per the user's rule, and sweep W and the floor; the audit currently fails with 76 undeclared literals.
6. **Repair Q12's "curious" design** so the first drops use the same random design as the random arm, and remove the generator's noise constants from the agent's tolerance; re-report C19.
7. **Add majority-class and "trust the prior only" baselines to every claim-checking table** (q25b, q30) and report selective accuracy rather than accuracy where abstention exists.
8. **Make E1-E3 the first deliverables of the oracle direction**, with the oracle-label student from expM as the primary arm and ECE / accuracy-coverage / conformal coverage as the primary metrics; state the predictions in section 5.1 in the queue before running.
9. **Literature:** add and verify AVICI, CSIvA, TabPFN/PFNs, Meta-SGD, SIS/knockoffs/stability selection, PCMCI/VAR-lasso, Das & Fiete 2020, Gerstgrasser 2024, Ovadia 2019, conformal prediction, selective prediction, Gekhman 2024 / Kang 2024 / R-Tuning, T-optimal design; fix the Shiu venue and the Wang 2014 citation; mark the two arXiv identifiers I could not confirm.
10. **Rewrite the one-paragraph description of what has been shown** (UNDERSTANDING §1-2, PAPER abstract) as: a small learned per-input gate, fitted on ~10 earlier worlds, recovers a sizeable part of the benefit of knowing the useful inputs for an online MLP/transformer on this generator, transfers under mild shifts, and does not collapse when retrained on its own labels from fresh real data; it does not learn causal structure (AUROC 0.56-0.73), and the project's strongest results are its negative ones (C3, C16, C17, Q12b resolution).
