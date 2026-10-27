---
title: Vertical Intelligence - Experiment plan (one test per item in the findings map)
domain: vertical-intelligence
kind: plan
tags: [research-program, experiments, plan, building-blocks, shared-world]
updated: 2026-10-26
---

# Experiment plan

Read this if: you are choosing the next experiment, or wondering how the pieces will ever be
combined. Every item in [[findings-map]] gets a direct test here. All tests run in **one shared
world (W0)** against **one shared baseline**, so each building block is measured on the same
ground and the passing blocks can be joined into one agent at the end (X9).

Five-line summary: build W0 and the baseline first (phase A). Then run section 0, the "for me"
(phase B, the current direction). Then one block at a time from sections 1-8 (phase C). Then
the integration test (phase D). There are 36 experiments (30 in the base plan plus 6 from the literature check); 11 absorb existing P-drafts. Each is still a *draft* until it is registered with seeds and a threshold under
[[research-protocol]] s3. Published prior experiments for each section are in
[[literature-map]].

## The shared world W0 (why the pieces can be combined)

A small CPU toy world, one file, which every experiment configures rather than rebuilds:

- **Internal variables** the agent must keep in range (energy, integrity). They decay over time.
  This is where "for me" lives: in W0, value can be defined *only* by these variables.
- **Resources and harms** whose cues share hidden structure across **seasons** (regimes or
  contexts that switch), so transfer, context inference and parts all have something to bite on.
- **Three kinds of region:** learnable, unlearnable noise, and a **dark corner** of zero surprise
  with no food.
- **Threats** that appear at random, used to test safety, narrowing and the secure base.
- **Impasse problems** solvable only after re-representing the input (for example, a rule that
  is linear in a rotated basis), used to test the wall and representation change.
- **Optional modules:** an oracle teacher (curious-net E8 style), a real-data module (E9 style).

**Shared baseline B0:** a plain gradient-descent network (Adam, tuned learning rate *and*
default, per method lesson s10 of the map) given an external reward equal to food eaten, with
the same inputs and the same compute. It has **exact parity at zero experience** (same
initialisation and outputs before learning).

**Default decision rule (replaces hand-set thresholds):** 48 worlds, paired against B0 on the
same seeds. Pass = the paired median difference has a 95% bootstrap CI excluding zero in the
predicted direction. Effect sizes are reported both absolute and as ratios. Every test also
needs a **manipulation check** (the varied factor must measurably change the agent in W0), or
it is VOID. Fresh seed block per test. Budget: CPU, <= 10 min per arm.

## Phase A - infrastructure (no science claims)

- **A1** build W0 with the modules above, plus a `--check` that verifies each module does what
  it claims (for example, the noise region is truly unpredictable).
- **A2** build B0 and the zero-experience parity check.
- **A3** a constants audit: route every W0 and agent number through `constants.py` (curious-net
  rule).

## Phase B - section 0: value and the self ("for me")

| Id | Question | Arms | Main metric | Serves |
|---|---|---|---|---|
| X0.1 | Can value come *only* from keeping yourself intact? | (a) homeostatic value: reward = reduction in the gap of the internal variables (Keramati and Gutkin style); (b) B0 external food reward; (c) random | survival time; **learned valence** for novel cues it was never told about; recovery when the seasons swap which cues are food | Q10, disconnect 1 |
| X0.2 | Are comfort and growth one hunger (U17)? | (a) ONE valence signal ("better/worse for me") driving both safety and exploration; (b) TWO drives (safety, curiosity) with a hand-weighted switch; (c) B0 | exploration vs threat curve: does (a) show the secure-base pattern (explore when safe, retreat under threat, resume after) with **no switching rule**? Also total learning, and the count of ARBITRARY constants each arm needs | U17, U10 |
| X0.3 | Does a learner discover what is "itself"? | (a) learner that partitions its variables into self/world by contingency (what its actions control); (b) no partition; (c) oracle partition | partition accuracy; learning speed after partition vs oracle | Q10, Markov blanket idea |
| X0.4 | Does ignoring a part's warning cause delayed breakdown (U16)? | one internal variable's signal is (a) heard by the controller, (b) masked from the controller while still affecting viability, (c) masked, then unmasked mid-life | time to collapse; size of the "warning" signal over time; recovery after unmasking | U16, U11 |
| X0.5 | Does a receding set point keep learning alive (light dims, never off)? | (a) fixed set point; (b) set point that rises with competence; (c) no set point (pure task reward) | learning still happening late in life; stagnation episodes; energy collapse (burnout) | U12, U9, U10 |

## Phase C - one block per section

### Section 1: drive

| Id | Question | Arms | Main metric | Serves |
|---|---|---|---|---|
| X1.1 (= P-draft 9) | Which drive avoids the dark room and the noise? | uncertainty reduction; novelty; learning progress; homeostatic (X0.1) | time in dark corner / noise; fraction of learnable region learned | U12, Q9 |
| X1.2 | Is boredom a useful signal (U9)? | (a) boredom = drive high AND learning progress ~ 0 -> switch task or representation; (b) no boredom; (c) fixed-schedule switching | total learning; time stuck | U9 |
| X1.3 | Does drive past the wall pay, and how (U15 + Ohlsson)? | on impasse problems: (a) stop at wall; (b) persist, same representation; (c) persist with a **background re-representation process** (incubation) | solve rate vs compute; what changed at the breakthrough | U15, U13 |
| X1.4 | Does external reward crowd out self-driven learning (overjustification)? | intrinsic drive alone; intrinsic + external reward added then removed; external only | self-driven exploration after the reward is removed, vs before | U3, Q3 |
| X1.5 | Can drive itself be learned (U15 "mostly learned")? | persistence fixed vs adapted from past payoff; early "rich" vs "poor" seasons | persistence and solve rate later in life | U15, I7 |

### Section 2: representation

| Id | Question | Arms | Main metric | Serves |
|---|---|---|---|---|
| X2.1 | Same knowledge, different representation: does layout decide cost (U13)? | identical data given as (a) a spatial map encoding, (b) a flat list encoding, (c) a learner that can choose its encoding | sample cost on new transformation tasks (rotation-like) | U13, disconnect 3 |
| X2.2 (= I6) | Local, use-only learning (U13/I6) | (a) connections with strength, cost and length, updated only when used; (b) B0 backprop; (c) (a) without cost | forgetting, new learning, energy per solved problem | I6, disconnect 5 |
| X2.3 (= I7) | Does fed curiosity widen the connections that survive pruning? | rich early curriculum vs poor, then a pruning phase, then new problems | transfer speed on new problems; surviving connection diversity | I7, C43 |

### Section 3: parts and the Self

| Id | Question | Arms | Main metric | Serves |
|---|---|---|---|---|
| X3.1 (= P-draft 4) | Is harmony superadditive? | all 8 on/off subsets of Sensors / Emotions / Self | joint gain vs sum of single gains | U4, Q1 |
| X3.2 (= P-draft 8) | Is the Self a predictor or a holder? | highest-weight part takes over; skill-free arbiter by confidence; arbiter that predicts the parts | mixed-season error; recovery when one part goes wrong | U7, Q7 |
| X3.3 | Is the Self a shared representation? | parts that write to one shared workspace; isolated parts; one monolith | transfer across parts; conflicts resolved | Q7 (Dialogue s6 candidate) |
| X3.4 (= P-draft 10) | Does threat narrow learning, and does wholeness multiply it? | threat routes plasticity to one part; uniform plasticity; no threat | total learning across parts; recovery after the threat | U10 |

### Section 4: memory over time

| Id | Question | Arms | Main metric | Serves |
|---|---|---|---|---|
| X4.1 (= P-draft 5) | Nothing thrown away? | archive + selective retrieval; forgetting memory; replay (C35); archive + pruned wiring | returning-knowledge error, new-learning error | U5, s7 item 7 |
| X4.2 (= P-draft 11) | Frozen part: express, then learn (U11) | train around; unfreeze directly; reactivate then unfreeze | repair of the frozen part; collateral damage | U11 |
| X4.3 | Brakes and reopening: does plasticity need brakes that can be released? | always plastic; brake after mastery (never released); brake released by surprise | long-run plasticity across 100+ seasons (loss-of-plasticity test); forgetting | U11, disconnect 5 |

### Section 5: learning from the world

| Id | Question | Arms | Main metric | Serves |
|---|---|---|---|---|
| X5.1 (= P-draft 1) | Goal-only vs self-driven data | goal-only; self-driven (own experiments, questions); mixed | out-of-set confident-wrong rate, calibration | U3 |
| X5.2 (= E8, X1) | Teaching modes and own experiments | teacher exposes its limits vs not; no / random / chosen experiments | skepticism tests from curious-net | oracle line |
| X5.3 | Secure-base teacher | teacher available on request; absent; constantly intervening | exploration breadth, calibration, learning | U10 |

### Section 6: randomness, variation and selection

| Id | Question | Arms | Main metric | Serves |
|---|---|---|---|---|
| X6.1 (= P-draft 7) | Directed vs undirected variation | none; blind noise; variation aimed by predicted learning gain + selection | out-of-training solve rate at equal compute | U6, Q8 |
| X6.2 | Can selection keep depth and diversity? | score-ranked cull (selfscore failure); novelty / quality-diversity cull; variety-window band (C9) | depth retained (T), distinct lineages, hidden-unit solutions found | U6, C9 |
| X6.3 (= I5) | Spontaneous thought | idle internal simulation kept if it improves the model; idle nothing; replay only | learning gained per idle tick | U8, U15 |
| X6.4 | Randomness + a predictor (learning within a life, variation across lives) | variation only; gradient only; both (Baldwin-effect style) | solve rate and cost on W0 impasse problems | U6 |

**Nature's optimizer before gradients (user, 2026-10-27; P-draft 12; code `neuroblock-research/emergent`, log `emergent/LOG.md`).**
User's question: how could nature produce life without gradient descent, which did not exist
at the dawn of life? "You are the stepping stones for improving any algorithm the way nature
does it." Verbatim: `raw/2026-10-27-emergent-optimizer-direction.md`. Series rule (user,
2026-10-26): task performance is used only from outside (the cull or the environment); every
change inside a net is blind. Adam is a yardstick only, never the target. Current state (series
3, commit 22e05ce): blind noise + 90% truncation cull solves 59/70 runs of a 7-problem suite at
0.51x default Adam, 9.4x tuned Adam; knob tuning has stalled, so per protocol s6 the question
changes. The assistant's reading of what separates the current system from early life: (1)
nature had no grader, only persistence under limited resources; (2) nature's variation machinery
(mutation rate, sex, gene duplication) evolved; (3) hard traits arose through stepping stones.
One test per gap, plus a benchmark repair first. Run order X6.5 -> X6.6 -> X6.7 -> X6.8 (user
may reorder; X6.7 is the most direct test of the question). Each <= 30 min GPU, one GPU job at a
time, after X0.7 finishes.

| Id | Question | Arms | Main metric, threshold (set before running) | Serves |
|---|---|---|---|---|
| X6.5 (infra) | Repair the benchmark: on spirals and checker3 most runs hit the 3e6 budget and are scored AT the budget, so the suite score hides their real cost | re-run series-3 round 0 (cull 80%) and the best (cull 90%) at budget 1e7, same seeds 0-9 | report solved/70 FIRST, then the geometric mean with unsolved runs flagged; parity check: every run solved under 3e6 must give a bit-identical cost (extending the budget must change nothing else) | benchmark validity |
| X6.6 | **Can the variation machinery evolve itself?** (nature improving its own algorithm, with no outer loop) | (a) current best, only noise size heritable; (b) + inherited direction: child change = m x parent's last change + sigma x fresh noise, with the gene m in [0,1] random at start, inherited, blindly mutated; (c) + recombination gene r: chance a child is a per-parameter mix of two survivors; (d) + unit duplication: a hidden unit is copied into a spare slot with outgoing weights halved (function-preserving), so nets grow; (e) b+c+d | manipulation check: across runs the evolved gene values move away from their random start (sign test over 70 runs, p < 0.01), else the gene was not under selection and that arm is VOID. Pass: an arm solves >= as many runs as (a) AND is >= 1.25x cheaper (geometric mean), on judge seeds 400-409, then confirmed on 20 fresh seeds 410-429. Evolvability: improvement per unit cost in the second half of each run vs the first half, (a) vs the arm. Prediction (assistant): (b) passes; (d) helps only on spirals/checker3 | U6, U1, Q8 |
| X6.7 | **Does selection work with no grader?** (early-life selection: whatever keeps being copied while resources are limited) | xorgauss first (as series 1). (a) graded cull, series-2 best; (b) ENERGY WORLD: each step B = 16 training points are shown; food per point = P_cap (= 40, parity with (a)) shared equally by the nets that answer it correctly; living costs 1 per point per net; a net with energy >= 2 splits into itself + a blind-noise copy (energy halved); energy <= 0 dies; no ranking, no cull fraction; (c) energy world with unlimited food (a fixed payment per correct answer, no competition) = the density-dependence check; (d) random search | pass: (b) reaches 0.95 test (best possible 0.955) in >= 18/20 seeds (seeds 500-519) within 3e6; cost reported vs (a) and default Adam with no threshold (prediction: 1-3x slower than (a)). Manipulation check: the correlation between a net's training accuracy and its number of offspring is > 0 in (b), and (c) either explodes or fails to learn; else VOID. The units (cost 1, split at 2, food = P_cap) are a choice of units plus one parity constant, not tuned | U1, Q4, Q10 (the 'for me' as energy) |
| X6.8 | **Do stepping stones reach hard problems sooner?** | targets spirals and checker3 (same nets, same targets). (a) direct; (b) fixed ramp: the problem grows (spiral turns 0.25 -> full; checkerboard 2x2 -> 3x3) on a fixed schedule; (c) self-paced ramp: the next stage starts when the population's best net reaches the stage target (external, allowed); (d) all stages mixed at once | cost to the full-problem target counting every stage, 10 seeds (600-609), budget 1e7. Pass: (b) or (c) solves more of the 20 runs than (a), or is >= 1.5x cheaper on the median. Manipulation check: in (a) at least 5 of 20 runs fail, else there is no impasse to cross and the test is VOID. Ties: Stanley & Lehman stepping stones / novelty search, POET (Wang et al. 2019), curriculum learning (Bengio et al. 2009) [u] | U1, Q4, C53 |

**X6.5 result (2026-10-27): DONE, parity PASSED.** 60/70 solved for both settings (was 58 and 59); all 10 unsolved are checker3. Geometric mean 0.60x default Adam / 11.2x tuned for the best (90% cull), 0.87x / 16.2x for the 80% cull. Series 3's 0.51x / 9.4x is superseded. All 58 and 59 runs solved under 3e6 reproduced bit-identically. Next in order: X6.6.

**X6.6 result (2026-10-27): DONE; no arm passed.** Judge seeds 400-409, 70 runs per arm: solved a 60,
b 51, c 60, d 60, e 57; cost vs (a): b 0.79x, c 1.07x, d 1.22x, e 1.07x (pass line 1.25x). The
manipulation check failed for every arm (genes not under selection at p < 0.01; only m in (e)
moved significantly, downward), so b, c, d and e are VOID. Duplication (d) still solved spirals 1.7x
faster with nets growing to width 28: the capacity helps, its evolved rate did not. No confirmation
run (nothing passed). Next in order: X6.7. Follow-ups parked in [[inbox]]: I8 (meta-evolution from
outside, now unparked) and I10 (fixed-rate duplication).

**X6.7 result (2026-10-27): DONE; the registered pass rule was missed.** Seeds 500-519, xorgauss,
target 0.95, budget 3e6. Graded cull 20/20 (median 51,520); default Adam 20/20 (54,240), tuned
Adam 3,840; random search 0/20; unlimited food exploded in 20/20; the energy world as literally
specified (b, energy summed over the step) went EXTINCT in 20/20 (0/20 solved); the per-point
sensitivity arm (e, registered before running because the plan does not fix the energy granularity)
solved 17/20 (needs 18) at a median 1,367,392, 26.5x the graded cull. Manipulation checks passed
(accuracy predicts offspring in every run; unlimited food explodes). Next in order: X6.8.

Parked, I8: a meta-evolution of the optimizer's settings (an outer population culled on suite
speed). Worth testing if X6.6 shows the machinery cannot evolve from inside; cost 2-4 h GPU
(over budget, ask first).

### Section 7: energy and substrate

| Id | Question | Arms | Main metric | Serves |
|---|---|---|---|---|
| X7.1 | Dale's principle while *learning* (beyond C42) | fixed excitatory/inhibitory neurons; free signs; fixed signs with a wiring cost | transfer, forgetting, calibration | U13, disconnect 8 |
| X7.2 | Neuromodulatory broadcast from viability | one global value (from X0.1's internal variables) gating local three-factor plasticity; B0 backprop | learning, forgetting, energy | U13, Q10, disconnect 8 |
| X7.3 (= P-drafts 2, 3) | Noise floor; growth on demand | as registered | as registered | U1, U2 |

### Section 8: time and order

| Id | Question | Arms | Main metric | Serves |
|---|---|---|---|---|
| X8.1 | Speed or order (U14)? | slow vs fast internal clock with order kept; same speed with order shuffled | causal-structure learning in W0 | U14, disconnect 9 |

## Changes from the literature check ([[literature-map]], 2026-10-26)

**New experiments (gaps no one seems to have tested):**

| Id | Question | Arms | Main metric | Serves |
|---|---|---|---|---|
| X0.6 | **Can a learner discover its own setpoints?** (the deepest "for me" test) | (a) essential variables whose viable ranges are found only by what kills or weakens the agent; (b) designer setpoints (Keramati-Gutkin); (c) B0 | does a stable "better/worse for me" form without given setpoints; survival; transfer to new seasons | Q10 |
| X0.7 | Ownership by consequences | an external "avatar" object whose damage (a) affects the agent's viability, (b) does not; sensing held equal | does the agent learn to protect it only in (a) | Q10 (rubber-hand analogue) |
| X1.6 | Wanting separate from learning (Berridge) | (a) a wanting variable kept alive while progress is flat; (b) one combined learning-progress signal | persistence through temporarily unlearnable stretches; time wasted on truly unlearnable ones | U9, U15 |
| X4.4 | Does a safety signal reopen frozen learning? (Nardou, Sale) | a "safety" input that lowers plasticity brakes vs none vs brakes always off | relearning in locked parts where C49 failed | U10, U11 |
| X5.4 | Abstention as a learned faculty | an explicit "unknown" output trained from the learner's own errors; max-softmax; temperature scaling | abstention on unseen outside regions | U3, C31 |
| X5.5 | Is the curriculum saving order, or self-selection? | ambiguity-driven self-chosen queries; fixed easy-to-hard; random | cost to criterion (vs C53's 2.4x) | U3, U15 |

**Status (2026-10-27):** X0.1, X0.1b, X0.6 and X0.7 are DONE (results in [[findings-map]] section 0; X0.7: a weakest-link margin works without designer numbers, and ignoring a need is what kills).
New from X0.6:

| Id | Question | Arms | Main metric | Serves |
|---|---|---|---|---|
| X0.7 | **Is a graded warning with no designer values enough?** | MARGIN (drive on the weakest essential variable, min of the variables; only given: 0 = death) vs learned VIAB, HOMEO, SUM, DEATH, ORACLE, in the orthogonal W0 and in a variant where one-variable trade-offs are deadly | deaths per 10k, need-dependent valence, oracle agreement where survival is at stake | Q10, U10, U16 |

**Modified experiments:**

- **X0.1** is built in the style of Egbert et al. 2010: metabolism sets behaviour directly, with no
  designer labels for food or poison.
- **X1.3** gains an **offline arm** (no data, noisy replay or weight noise; Wagner 2004, Sio and
  Ormerod 2009) and a **re-chunking mutation arm** (Knoblich 1999).
- **X3.2** gains a **COIN-style holder** arm (context posteriors only, no prediction) and a
  takeover arm without load balancing (Shazeer 2017), to test U7 directly.
- **X4.1** gains a **turnover** arm (Dohare 2024 continual backprop: reset little-used units) as
  the strongest rival to U5.
- **X4.2** is reframed: what must happen at reactivation is **a mismatch** (prediction error,
  Sevenster 2013), not just full expression. Arms: reactivate with mismatch; reactivate without
  mismatch; no reactivation.
- **X4.3** uses the Dohare long-sequence protocol as its world.
- **X6.2** uses MAP-Elites niches and novelty search as the diversity arms, on a deceptive task.
- **X7.1** matches weight **spectra** across sign-constrained and free arms (Li et al. 2023), so
  a sign effect can be told apart from a spectrum effect.

## Phase D - combination

- **X9.1 integration lattice.** Take every block that passed in phases B and C, put them in one
  agent in W0, and run the ablation lattice (P-draft 4 generalised). The program's main claim is
  harmony: the whole is **superadditive** (joint gain > sum of single gains), and removing any
  block hurts.
- **X9.2 final-exam ladder.** The integrated agent on the I3 toy pre-quantum exam, then the real
  pre-quantum benchmark (Q-PRE1900).

## Order and dependencies

1. A1-A3 (needed by everything).
2. **X0.1, then X0.2** (current direction; X0.1's value signal feeds X0.2-X0.5, X1.1 and X7.2).
3. X1.1 (resolves protocol s7 item 8), X6.1, X3.2: the three tests whose results most change
   the design of the others.
4. The rest of phase C in any order; one registered run per iteration (protocol s6).
5. X9.1 once at least one block from each of sections 0, 1, 2 and 3 has passed.

Every result goes back into [[findings-map]] under its section, including failures.
