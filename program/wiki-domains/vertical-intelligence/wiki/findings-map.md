---
title: Vertical Intelligence - Findings map (everything sorted by what it tells us)
domain: vertical-intelligence
kind: synthesis
tags: [research-program, map, building-blocks, gradient-descent, findings]
updated: 2026-10-26
---

# Findings map

Read this if: you want to know what the program has actually learned, or where a new
experiment belongs. Every result, theorem, question and planned test from both code lines
(curious-net, neuroblock-research) and Dialogue 1 is sorted here by **what it tells us about the
building blocks of intelligence**, not by score or by date. Each section ends with the
**gradient-descent disconnect**: where a plain network trained by gradient descent misses.

Five-line summary: the program is searching for the building blocks of intelligence and for the
places where gradient descent falls short. Nine building-block sections (0-8) hold the science.
Section 9 holds engineering results, which tell us nothing about brains. Section 10 holds
lessons about method, including the user's rule (2026-10-26): **stop tuning numbers to move a
mean accuracy; change the question instead.** The current direction is section 0, the "for me"
(Q10).

**Rule for new work:** every test names its section here and the disconnect it probes. A stalled
number is a cue to change the question, not the knobs.

Confidence letters (curious-net): **R** replicated, **S** supported, **P** partial, **D**
diagnosis only, **N** not supported as tested. NeuroBlock uses VERIFIED / PARTIAL / UNTESTED
(`neuroblock-research/publication/CLAIMS_LEDGER.md`). All results come from toy worlds:
principles in miniature, not facts about brains or LLMs. Ids point to `curious-net/LOG.md`,
`paper/PIECES.md` and the NeuroBlock logs.

## 0. Value and the self: what makes something "better or worse for me" (current direction)

- **Givens:** U1 (we came from random), U16 (intuition as warning from a pushed-aside part), U17
  (comfort and growth are one hunger, grown from one better/worse-for-me signal), U10
  (wholeness first).
- **What experiments tell us:**
  - **X0.1 (2026-10-26, 48 worlds; curious-net `vi/x0_1.py`, P199-P203):** one gradient-descent
    learner, with only the value signal varied. Value defined through the self (drive reduction,
    HOMEO) gives **need-dependent valence**: it takes energy-for-damage food when starving and
    refuses it when full (need slope 0.86 vs 0.14 linear, 0.01 food-only; P200 PASSED). It beats
    the food-only baseline on survival (7.0 vs 28.7 deaths per 10k; P199 PASSED) and after season
    swaps (P203 PASSED). But a **linear two-channel reward survives better** (2.0; P201 FAILED), and
    HOMEO's valence is poor at states it rarely visits (0.62; P202 FAILED). Untested diagnoses:
    one-step myopia (gamma = 0), a harder value function, designer setpoints. Confidence: S for
    the need signature (one run, fresh seeds, pre-registered); N for 'self-value is safer'.
  - **X0.1b (2026-10-27, 48 fresh worlds; P204-P207):** adding foresight (TD value, gamma tuned
    to 0.99 for the self-defined arm) does **not** fix the safety gap or the valence accuracy, so
    **myopia is rejected as the cause**. The need-dependent signature survives unchanged (need slope
    0.70 vs 0.06 for the linear reward; P206 PASSED), so it is robust to foresight and tuning.
    Open diagnoses: the neutral probe state is rarely visited (possible probe artifact); the drive
    has no barrier at the death boundary; the setpoints are designer-given. Next: X0.6.
  - **X0.6 (2026-10-27, 48 fresh worlds, exact survival oracle as ceiling; P208-P213):** an agent
    that learns its own viability from deaths alone, with **no designer setpoint or weights**, has
    a self-model that **halves deaths relative to the raw death signal** (16.0 vs 38.3 per 10k;
    P209 PASSED). But it is worse than food-only (9.3), than designer setpoints (3.3) and than the
    summed reward (0.67), shows **no need-dependent valence** (0.07 vs 0.84 with setpoints), and
    agrees with the oracle no better than the setpoint arm (P208, P210-P213 FAILED). It is
    data-starved: about 24 deaths per lifetime to learn the boundary from (untested diagnosis).
  - **Correction to the X0.1 reading:** the need-dependent valence came from the designer
    setpoints (distance to a target gives graded urgency), not from self-defined value in the
    abstract. What is missing without a designer is a **graded warning before death**. Biology
    supplies it by evolved interoceptive signals; the next candidate with no setpoint and no
    weights is the margin of the weakest essential variable (X0.7), which also formalises "no
    part left behind" (U10) and the neglected part's warning (U16).
  - **X0.7 (2026-10-27, 48 worlds per world type; P214-P219):** a **weakest-link margin** (reward =
    change in the lowest essential variable; no setpoints, no weights, only that 0 means death)
    gives what X0.6 lacked, a graded warning: it beats the viability learned from deaths (2.0 vs 18.7
    deaths per 10k), is at least as safe as designer setpoints (2.0 vs 4.0) and shows need-dependent
    valence (0.51 vs 0.02 for a plain sum; P214-P216 PASSED). But in a world where needs conflict it
    is **not** safer than a plain sum or than setpoints (14.7 vs 11.3 vs 10.7; P217-P219 FAILED).
  - **What decides survival is attending to every need, not how they are combined:** the learner
    that ignores one need (food only) dies 11 to 15 times more often when needs conflict (155 vs
    11 to 15 deaths per 10k) and 4.7 times more often in the gentle world. The data side of "no part
    left behind" (U10) and of a neglected part's warning (U16). Untested: how this scales with
    many variables of different scales (idea I9).
  - Before X0.1 there was nothing direct. The nearest result is NeuroBlock's energy
  economy (`ENERGY_FINDINGS.md`): brains with diversified income stayed viable 90% of the time,
  while growth-only income starved (26-37% good, offline half the time), which the log calls
  "stagnation is death". That is a viability result: value defined by staying alive.
- **Disconnect:** a gradient-descent network has no "me". Its value is an external loss. There
  is nothing it keeps intact, so nothing is better or worse *for it*.
- **Next:** **Q10.** (1) a sourced literature map (autopoiesis; Damasio; homeostatic RL;
  Markov blankets; minimal self); (2) a viability toy where value is defined only by keeping
  the learner's own variables in range. Related: P-draft 6, I1 (Self v0 paid in energy).

## 1. Drive: why a learner learns at all (curiosity, boredom, motivation)

- **Givens:** U9 (boredom = want + inability to learn), U12 (not uncertainty reduction), U15
  (genius = drive past the wall, mostly learned), U3 (goal-trained becomes single-minded).
- **What experiments tell us:**
  - Curiosity's proven job is **avoiding the unlearnable**: it spends only 3-6% of its time on
    noise traps, but it does not choose well among learnable things and loses to random (C40,
    S/P).
  - **Motivation plus curiosity** beats either alone: filter by learnability, then persist on
    the hardest learnable thing (C52, R). This is the closest existing result to U15.
  - NeuroBlock's "always try to expand" drive did nothing: 25 expansions per run, no gain
    (sweep, contradicts). A drive with no signal of where growth pays is noise.
- **Disconnect:** gradient descent has no drive. It learns whatever the loss demands, cannot
  choose what to learn next, and never gets bored.
- **Next:** P-draft 9 (dark-room test: uncertainty reduction vs novelty vs learning progress),
  P-draft 6 (internal score), I7 (fed curiosity widens connections). Conflict: protocol s7
  item 8.

## 2. Representation and building blocks: how knowledge is laid out

- **Givens:** U13 (the missing link is in representation, not knowledge), Oct 22 foundational
  blocks.
- **What experiments tell us:**
  - **Transfer is proportional to shared hidden structure**: 1.00x on unrelated problems, up to
    7.5x when all problems share blocks; a block learner rediscovers most hidden blocks (C43, R).
  - **Few connections matter**: a wiring budget (C41), a focused forward model (C47) and sparse
    dendrites (C50) all win; real fly wiring is sparse the same way (C22) (R).
  - NeuroBlock: the matching radius (ART-style vigilance) is the dial that matters most. When the
    brain is wrong, **the wrong neuron answered**, not a missing one: a best-fit rule lifts 79->95%
    (sweep). A fixed radius breaks in noisier or wider worlds and cannot loosen itself.
  - The learned gate is exactly a per-input re-initialisation plus a learning-rate change
    (theorem T7; C30, C33, C34; R): it learns *useful* inputs, not causes.
- **Disconnect:** gradient descent learns dense, entangled representations with no wiring cost,
  and the representation is fixed by the architecture chosen in advance.
- **Next:** I6 (local rule: strength, cost, length, changed only by use). Q8 (part or loop).

## 3. Parts, routing and the Self: which part handles which situation

- **Givens:** U4 (Self / Sensors / Emotions in harmony), U7 (parts with weights can take over),
  U10, U11; Q1 (harmony), Q7 (what is the Self).
- **What experiments tell us:**
  - **Correct parts are 20x better** than one net in a world of distinct regimes, but the system
    cannot yet assign them itself (C44, D).
  - **Context inference, "which situation am I in?", is the shared bottleneck** of mastery,
    parts and memory protection (C45, C46, CTX2; D/S). Context mistakes cost more the more the
    rules differ (T8).
  - **Check when unsure**: checking only where remembered cases disagree (about 9% of points)
    cuts errors 65-73% (CTX6, R) and beats 1.5x more random labels (CTX7, R). A meta-recogniser
    judged by consequences matches the best single recogniser in each world (CTX12b). The full
    context stack works with one setting in both worlds (CTX13).
  - NeuroBlock: one persistent brain solves several problems by 70% consensus voting with no
    problem id (C8, VERIFIED), but forgets when trained in sequence (0.97 -> 0.80).
- **Disconnect:** one monolithic network averages incompatible situations, and has no notion of
  which situation it is in or when to ask.
- **Next:** P-draft 4 (harmony ablation), P-draft 8 (blending vs arbitration: is the Self a
  predictor or a holder?), I1.

## 4. Memory and change over time: what stays, what changes, what freezes

- **Givens:** U5 (nothing thrown away), U11 (frozen trauma part: express, then learn).
- **What experiments tell us:**
  - Fast store plus replay is a **trade-off**: replay protects returning knowledge (-12%) but
    costs about 5% on new learning (C35, P).
  - A surprise signal detects change at 8x chance, but only the **pair** (forget stale history +
    boost learning) helps (C36, P).
  - **Locking weights fights relearning** (C49, N): protection works only when skills live in
    separate weights. This is the closest result to U11.
  - Sleep replay is no better than awake replay at equal compute (C54-C56, N), unless waking
    learning saturates, where a cost-benefit bedtime rule times sleep within 1.2% (R).
  - Mastery: practice with the right memory halves error; recognising which memory applies is
    unsolved (C37-C39, D).
  - NeuroBlock: pruning unused neurons is free (25 -> 11, no loss); hardening slowed thinking
    with no steadiness gain (sweep, contradicts); a champion/challenger guard removes collapses
    (C7, VERIFIED).
- **Disconnect:** every gradient step touches almost every weight, used or not, causing
  catastrophic forgetting. Plasticity is not gated by use and cannot be frozen and reopened.
- **Next:** P-draft 5 (archive everything), P-draft 11 (frozen part: reactivate, then update),
  I6.

## 5. Learning from the world: teachers, own experiments, skepticism

- **Givens:** U3; the oracle teacher (Oct 22); the student must run its own experiments; Q2-Q3.
- **What experiments tell us:**
  - A student taught only truth is **honest inside, confidently wrong outside**, and its
    self-doubt fails; "this looks unfamiliar" mostly works (C31, S).
  - **Teacher order matters**: easy-to-hard is 2.4x cheaper than random (C53, R). Right innate
    knowledge buys early speed; wrong innate knowledge hurts more and longer (C48, R).
  - A few self-chosen experiments identify one-variable laws and catch claims 1% off (C32, S).
  - **Kepler's third law from real NASA data**: power law selected, p = 1.4978, bad theories
    rejected, a data inconsistency found (E9/C57).
  - NeuroBlock: a predictive agent matches or beats tuned MLPs under bandit feedback on 11/11
    held-out problems, though it is slower per decision (C6, VERIFIED).
- **Disconnect:** a network trained on a goal is confidently wrong outside it, never runs its
  own experiments, and does not reject theories.
- **Next:** P-draft 1 (goal-only vs self-driven data), E8 (teaching modes), X1 (experiment arms),
  I3 (pre-quantum exam).

## 6. Randomness, variation and selection: what randomness can build

- **Givens:** U1, U6 (intelligence uses randomness to its benefit).
- **What experiments tell us:**
  - Populations of randomised variants never beat the single best (C17, N); quantum-style
    beliefs help only on quantum-like data (C16, N).
  - **But blind variation plus external culling does build structure.** Random weights + noise
    + a 50% cull solve noisy XOR 5/5 (linear: 0.515), with at least 280x fewer evaluations than
    random search, which never got there (emergent attempts 1-2). With a smooth score it matches
    **default** Adam on fresh seeds (0.90x), but is still about **12x slower than tuned Adam**
    (series 2). Series 3 reported 0.51x default / 9.4x tuned on 7 problems at a 3e6 budget; **X6.5 repaired the benchmark (budget 1e7, 70 runs per setting): 60/70 solved, best setting 0.60x default Adam / 11.2x tuned (80% cull: 0.87x / 16.2x); all 10 unsolved runs are checker3, which neither setting solves even at 1e7 (its cost is a lower bound); default Adam also fails spirals and checker3 in 20,000 steps, so those two ratios compare censored costs; parity: all 58 and 59 runs solved under 3e6 reproduce bit-identically.**
  - Greedy rules (undo + barred change types, the separate stack) are **worse** than noise on
    all weights: they stall where no single change helps (attempts 3-4).
  - Random-cull multi-problem brains plateaued at the **linear ceiling** (0.55-0.61 vs 0.973 for a
    small trained net): **depth was selected away** (T 2 -> 1), **variety collapsed** to 1-2
    lineages, and refilling from random always stayed at baseline (selfscore series, closed).
  - Selection math: Price identity exact (C1, not novel); interior optimum cull 75-95% under
    noisy scoring (C4); the multi-generation law was refuted (C5).
  - **X6.6 (2026-10-27, 70 runs per arm, budget 1e7; P220-P224):** nature's variation machinery
    does **not** evolve itself here. Three blind, heritable mechanisms (a direction gene, a
    recombination gene, a duplication gene) were each tried alone and together; **no arm passed**
    (pass = as many solved and 1.25x cheaper) and the genes drifted without detectable direction
    (sign tests p = 0.19, 0.72, 0.47); the one that moved significantly, momentum, was selected
    AGAINST (p = 0.0035). The one benefit: unit duplication solved spirals 1.7x faster and grew the
    nets to width 28, with a gene that did not evolve, so the capacity to grow helps but its rate
    was not evolved. Unparks meta-evolution from outside (I8) and a fixed-rate duplication test (I10).
- **Reading (assistant):** randomness *with selection* can build what randomness alone cannot.
  Blind variation is far less efficient than gradient, and score-ranked selection destroys
  depth and diversity. This sharpens U6: randomness used well must be **aimed**, not blind.
- **Disconnect:** gradient descent has no exploration of its own. It follows the slope from one
  start and cannot jump valleys, but on a slope it is 12x more efficient than blind variation.
- **Next:** P-draft 7 (directed vs undirected variation), C9 (variety-window hypothesis,
  UNTESTED), I5 (spontaneous thought), I4 (rocks to thinkers).

## 7. Energy and substrate: what biology's constraints buy

- **Givens:** U2 (brain as bare-minimum machine), U13 (substrate aspect).
- **What experiments tell us:**
  - **A wiring budget is useful, not just a cost**: -12.8% shifted-situation error at 5% of the
    weight mass (C41, R). A firing cost alone only costs.
  - **E/I signs are non-negotiable**: first in 27 of 27 regimes; a wiring diagram gets you about
    halfway (C42, R, FlyWire). This is the first evidence for U13 (Dale's principle).
  - A **local three-factor rule** (fading tag + later signal) solves delayed credit when the tag
    lasts about as long as the delay (C51, R). Biologically this is neuromodulation.
  - NeuroBlock energy economy: moving only when the expected saving beats the price cuts energy
    36x at 100% accuracy; automatic neurons make known things almost free (30 -> 10 E per 1000);
    paying only for firing neurons is slightly better; shutdown/fatigue rules cost viability.
- **Disconnect:** gradient descent pays nothing for connections or activity, lets any weight
  take either sign, and uses a global error signal instead of local three-factor learning.
- **Next:** P-draft 2 (noise floor), P-draft 3 (grow on demand), I2 (join the two halves).

## 8. Time and order

- **Givens:** U14 (time is not required for intelligence).
- **What experiments tell us:** C51 (the eligibility tag must last as long as the delay: order and
  delay matter for credit); NeuroBlock's thought-speed floor is a geometry setting, not a law.
- **Reading (assistant):** absolute speed is not required; relative order (what came before what)
  probably is.
- **Disconnect:** gradient descent learns from shuffled batches, discarding order unless the
  architecture puts it back.
- **Next:** none registered.

## 9. Engineering only (tells us nothing about brains)

Speedups (3.8x end to end, forward pass 6x bit-identical), GPU memory cap after two crashes,
crash-resume, reproducibility, W&B tracking, the Lab, `ceiling.py`. Useful infrastructure; keep
them out of science claims.

## 10. Method lessons (what the experiments tell us about how to experiment)

- **Metric chasing is a trap (user, 2026-10-26).** The selfscore series and the series 2/3
  speed tuning spent many runs moving a mean accuracy or a speed ratio. The selfscore result
  that mattered was a **diagnosis** (linear ceiling, depth selected away, variety collapse), not
  a number. Rule: when a number stalls, change the question.
- **Winner's curse:** 5-seed winners were partly luck. Fresh 20-seed confirmation cut 0.48x to
  1.76x (series 2).
- **Fair baselines:** compare against *tuned* gradient descent as well as default. "Matches
  Adam" meant default Adam; tuned was 12x faster.
- **Test design can fail before the method does:** a 0.90 XOR target was reachable by luck
  (emergent attempt 0); manipulation checks voided CTX11b at noise 0.05; scoring after seeing
  the answer (CTX11, CORRECTION iteration 70).
- **Smooth scores beat 0/1 scores** for selection (series 2 round 3: 13x -> 1.7x).

## Cross-cutting: where gradient descent ALONE falls short (the disconnect list)

Framing (U18, user): gradient descent is not wrong. It is probably the fastest learner and will
be inside the final system. Each row is something a plain gradient-descent learner does not do
*by itself*, which a manager around it (drive, parts, Self, value) would have to supply.

| # | Disconnect | Section | Evidence |
|---|---|---|---|
| 1 | No "me": value is external, nothing to keep intact | 0 | none yet (Q10) |
| 2 | No drive: cannot choose what to learn; no boredom | 1 | C40, C52 |
| 3 | Dense representations, no wiring cost, fixed layout | 2, 7 | C41, C43, C47, C50 |
| 4 | One network averages incompatible situations; no context, no asking | 3 | C44, CTX6-7 |
| 5 | Global updates touch every weight: forgetting; no use-gated plasticity | 4 | C35, C49, C8 |
| 6 | Confidently wrong outside its goal; no own experiments | 5 | C31, C32, E9 |
| 7 | No exploration of its own: cannot jump valleys (but 12x more efficient on a slope) | 6 | emergent, selfscore |
| 8 | Any weight, any sign; global error instead of local three-factor learning | 7 | C42, C51 |
| 9 | Shuffled batches discard order | 8 | C51 (indirect) |

**Literature check of the disconnect list** ([[literature-map]], 2026-10-26):

- **1** (no "me"): untested anywhere with learned setpoints. Open, and the program's best shot
  at something new.
- **2** (no drive): learning-progress curiosity exists in ML and has human evidence. The gap is
  wanting kept separate from learning.
- **3** (dense representations) is **weakened**: Othello-GPT, TEM and Lake and Baroni show
  gradient descent *can* build structured world models. The gap may be in data, objective and
  cost rather than in the learning rule.
- **5** (forgetting) is **strengthened**: deep nets lose the ability to learn over long sequences
  (Dohare 2024).
- **7** (no exploration): diversity-keeping search beats objective search on deceptive tasks.
  This also **challenges C17 as stated**.
- **8** (any sign): Dale-compatible nets can learn as well as free ones, and the weight
  spectrum may be what matters. C42 needs a spectrum control.

The rest are consistent with the literature, with no reversal found.

## Credit

Sorting, section readings and the disconnect list are the assistant's synthesis of the
registered results (2026-10-26), for the user to correct. The givens, the program and the
method rule are the user's.
