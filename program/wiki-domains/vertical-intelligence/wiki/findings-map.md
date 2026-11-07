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
(`experiments/energy-brain-simulator/publication/CLAIMS_LEDGER.md`). All results come from toy worlds:
principles in miniature, not facts about brains or LLMs. Ids point to `LOG.md`,
`paper/PIECES.md` and the NeuroBlock logs.

## 0. Value and the self: what makes something "better or worse for me" (current direction)

- **Givens:** U1 (we came from random), U16 (intuition as warning from a pushed-aside part), U17
  (comfort and growth are one hunger, grown from one better/worse-for-me signal), U10
  (wholeness first).
- **What experiments tell us:**
  - **Energy as viability (X6.7, 2026-10-27):** a population whose only pressure is staying alive on limited food found a solution without any grader (17/20); too little energy buffer against the noise of one step drove it extinct in 20/20 (see section 6).
  - **X0.1 (2026-10-26, 48 worlds; curious-net `experiments/value-and-self/x0_1.py`, P199-P203):** one gradient-descent
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
  - **[FLAGGED: NEEDS MORE STUDY (author, 2026-10-27); see the inbox list and I9]**
  - **X0.7 (2026-10-27, 48 worlds per world type; P214-P219):** a **weakest-link margin** (reward =
    change in the lowest essential variable; no setpoints, no weights, only that 0 means death)
    gives what X0.6 lacked, a graded warning: it beats the viability learned from deaths (2.0 vs 18.7
    deaths per 10k), is at least as safe as designer setpoints (2.0 vs 4.0) and shows need-dependent
    valence (0.51 vs 0.02 for a plain sum; P214-P216 PASSED). But in a world where needs conflict it
    is **not** safer than a plain sum or than setpoints (14.7 vs 11.3 vs 10.7; P217-P219 FAILED).
  - **What decides survival is attending to every need, not how they are combined:** the learner
    that ignores one need (food only) dies 11 to 15 times more often when needs conflict (155 vs
    11 to 15 deaths per 10k) and 4.7 times more often in the gentle world. The data side of "no part
    left behind" (U10) and of a neglected part's warning (U16). Scaling with many variables: X0.8 below.
  - **X0.8 (= I9, 2026-10-27; 48 worlds, N = 2, 4, 6 essential variables with scales spread 10x;
    P254-P258):** ignoring needs costs more as they multiply (food only 205 -> 324 deaths per 10k;
    P257 PASSED), but three theory predictions about HOW to combine needs FAILED, two of them reversed:
    the raw equal-weight sum (109 at N = 6) was better than the sum normalised by each variable's scale
    (131; P254), and the weakest-need-first margin, which needs no designer numbers, was worse than both
    (153; P255, P256). With default settings the margin ties food-only at N = 6 (229 vs 221): it pays only
    when the object touches the weakest variable, so with many needs its signal is sparse. **Confound
    found:** in this world a variable's scale multiplies both the units of its change AND its dynamics, so
    the raw sum weights the fast, dangerous variables most (SUM spread its deaths over all variables;
    the others piled 41-53% on the fastest). Units vs rates was separated in X0.9 (2026-10-28, 48 worlds x N = 2, 4, 6; P259-P262): when units are unrelated to urgency the raw sum is the WORST combiner (141 vs 89 deaths per 10k with units divided out, at 6 needs), so X0.8's 'raw sum beat the normalised sum' was an artefact of unit-urgency coupling. The weakest-need-first rule needs no units and beats the raw sum when units mislead (-30 [-43, -13]) but trails the unit-removed sum and designer setpoints at 6 needs and is setting-sensitive. P261 (raw sum hurt more when units are anti-coupled) FAILED, reversed, confounded by rate heterogeneity hurting every arm. What holds in all three worlds: attending to every need beats ignoring them.
  - **X1.3 (2026-10-28, the author's interest: an entity whose whole existence is prediction and internal models; 48 worlds; P275-P281):** with no food or goal, an entity whose only needs are prediction-based (vitality that dies if predictions fail; a hidden need refilled only by real structure learned, felt as boredom, that dies at zero) and whose reward is the relief of those needs **avoids both the dark room and the noise** (0 deaths in 48 of 48 worlds vs 6.7 for the same entity ignoring boredom, 8.3 for surprise minimisation, 21.7 for chance; dark-room share 31% vs 91%). Entities that care only about being right retreat to the dark room and die of boredom: U12 in miniature. **Hostile reading:** the boredom signal is exact (the world hands it the bits learned) while the designer learning-progress arms estimate progress from noisy errors, so 'beats the given drive' shows signal quality, not framing; depth is cheap to find (chance learns more bits than the informed reference). Next: X1.4 (exact-bits designer control, noisy boredom, sustained-streak depth).
  - **X1.4 (2026-10-28, 48 worlds; P288-P293) resolves X1.3's confound in an unexpected direction:** a designer arm rewarded only by the EXACT bits learned (no vitality, no boredom) survives as well as the vitality-plus-boredom entity (0 deaths in 48 of 48 worlds); so in this world the signal's exactness, not the framing, does the work. Noisy, 10-step-averaged boredom degrades gracefully: death-free worlds 100% / 98% / 58% at noise sd 0.02 / 0.05 / 0.10 (medians hid it; P293 failed as registered), and every level still beats ignoring boredom. Tuned learning-progress windows did not help (the estimated-progress drives stay at 3-5 deaths per 10k). Next: X1.5, larger noise and a boredom computed from the entity's own prediction errors.
    The genuinely *hidden* need (a variable whose level the learner cannot observe) remains untested.
  - **X0.10 (2026-10-28, the author's request; 48 worlds; P263-P270):** a hidden need that only
    non-survival *depth* can fill, with the entity given only a rising boredom warning. The entity
    that treats boredom relief as reward **stays alive and keeps the hidden need full** (5 deaths per
    10k vs 102 for chance and 23 for a circle-only entity; hidden level 1.00 vs 0.46 for the same
    learner ignoring boredom). Circle-only entities, learned or informed, die of the hidden need in
    every world, so depth is genuinely required. **Hostile reading:** the gap to the ignore-boredom
    learner is +0.40 at tuned settings but only +0.06 at default settings (direction robust, size
    not); and chance finds the ladder faster than either learner, so this world shows boredom relief
    *sustaining* the hidden need, not learned *discovery* of depth. P269 (fewer hidden-need deaths)
    failed just short. Next: X0.11, depth that must be searched. First experiment on W&B.
  - **X0.2 (2026-10-27, 48 worlds per world type; P239-P243) tests U17 (comfort and growth are one
    hunger):** a single quantity, expected change for me with uncertainty-driven sampling and a known
    threat penalty, **is sufficient to produce the secure-base pattern** with no weight and no switch:
    it explores while calm (10.4% of ticks), retreats under severe threat (0.0% exploring), resumes at
    once, and retreats in proportion to the stakes (it keeps exploring under mild threat, 9.8%, where
    staying out still pays). It beats random exploration in all three worlds. It is **not better** than
    a two-drive agent with one tuned weight and a hand-built switch (regret 0.0018 vs 0.0015 to 0.0016
    in the medium and severe worlds), and the predicted advantage of needing no tuning did not appear:
    the best weight (2.0) was the same in every world. So U17 is not falsified and not shown superior;
    the difference is parsimony (0 constants vs 1 to 2). Confounded with Thompson sampling vs a bonus.
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
  - **X1.40-X1.42 (family-free cell, depth; 2026-11-03/04).** A cell whose units mix tanh/sin/relu/identity recovers an exact deep concept in 1 of 3 worlds at g = 1.2 (the sine cell: 2 of 3) but loses the sine cell's accuracy (exam 0.70 vs 0.96); an L1 penalty on the output weights to let the learner choose its own activations did not recover it (0 of 3 worlds). So the X1.39 win is, so far, about handing over the rule's family, not discovering it. In one world (capped refit) the sine cell recovers the same 2 exact levels in rooms 1-4 (5 to 20 levels) and nothing new in deeper rooms (exam 0.97 to 0.68): no depth scaling seen; a memory-light refit that does not change results is needed to replicate. Detail: curious-net LOG.md.
  - **X1.37-X1.39 (labels, capacity, family; 2026-11-03).** More labels per problem (60 to 6000) and a bigger cell (6 to 96 hidden units) did NOT give exact deep recovery; the tanh cell fits better without finding the concepts (g = 2.0 corr^2 stays ~0.06). Putting a sine inside the cell (the world's rule is a sine of a sum) at g = 1.2, 24 hidden units gave exam 0.96 and exact deep concepts (corr^2 >= 0.99) in 2 of 3 worlds, 5 of 5 levels at corr^2 >= 0.9: the first exact recovery. Caveat: sin hands over the rule's family; a family-free version (choose among activations by measured help) is the open test. g = 2.0 only reaches corr^2 ~0.2-0.26. Detail: curious-net LOG.md.
  - **X1.36 (one shared tied cell, compute ladder, 2026-11-03).** One tiny net reused at every depth and refit jointly on all its uses: at g = 0.8 exam R^2 rises from ~0.73 (no refit) to 0.87-0.90 (30-600 refit steps) with ~390 weights in the deepest room (decided up); at g = 1.2 and 2.0 compute does nothing because the first step from the parent is not learnable at 60 labels, so the cell never starts. No concept recovered at corr^2 >= 0.99. Detail: curious-net LOG.md.
  - **X1.35 (puzzle room, one-rule deep world with thin ties, 2026-11-03).** Choosing a parent by measured help (try every known function) or reusing a learned computation with a new parent and input rotation (clone) cuts the functions linked per problem from 7-8 to about 2 while raising the exam score (g = 0.8: 0.86 vs 0.80 in the deepest room). Correlation-chosen parents do nothing; pooled refinement adds accuracy (up to +0.2 at g = 1.2, room 1) but not simplification. At g = 2.0 the routes that 'simplify' do so by not learning. No route recovers an exact concept beyond level 0: gains are better approximations, not discovered rules. Many-candidate routes need an evidence bar that grows with the number of candidates (else ~300 chance functions are accepted). Detail: curious-net LOG.md.
- **X1.29 (huge compute, random as a choice, 2026-11-03).** 10x compute lifts every arm (+0.02 to +0.10 R^2, 9 of 10 clear) and closes the gap between 'bigger is better' and the flat learner (0.797 vs 0.770), but concepts built on other concepts appear only by luck: 6 of 120 arm-world cases, depth 1 only, 3 of 12 worlds, none deeper. Compute alone does not produce depth. All-random labels are the right default; learning each problem both ways and keeping the better ('auto') adds +0.03 on 'bigger is better'; choosing inputs never helps. Detail: curious-net LOG.md.
- **X1.30 (a goal of SIMPLE, 2026-11-03).** Simple is cheap down to about 100 weights and 1.6 links per problem (R^2 0.56-0.59, 75% of the 8000-weight model), then falls off a cliff; charges of 4+ bits per weight give an empty model. No single reading of simple wins: the trade-off front spans FLAT, REUSE, MDL, NODE and LINKS arms, and rank orders by weights, links and speed disagree. Detail: curious-net LOG.md.
- **X1.28b (every route of the digger, 2026-11-03).** Restraint costs accuracy and freedom helps: starting budget 1200 beats 600 by +0.040 R^2; all-random inputs beat 40 chosen by +0.014; 50 chosen is worse (-0.009); smaller concept nodes (3 hidden) match the default at 42% fewer weights; a heavier per-weight charge (1x) is worse (-0.012). Parents 0/2/4 and hidden 12 do not differ. No route beats the flat learner (0.743). Interactions untested. Detail: curious-net LOG.md.
- **X1.28 (2026-11-02, 48 noise-free worlds; P495-P496 PASSED, 4 FAILED): dig until stone (author I22), all three readings of 'total
    model size'. Bigger-is-better 0.71 R^2 > budget 0.65 > smallest description 0.59; the plain flat learner 0.74. NO arm recovers a single
    concept built on another (0%): stone is hit at the first composed concept.** Choosing inputs by disagreement hurts (-0.014).
  - **X1.27 (2026-11-02, 48 fixed worlds; P489-P492 all FAILED): concept nodes (tiny 2-layer nets) are linked ~1,600 times per stream
    but deep problems stay at ~28%.** A 2-layer node CAN express a deep concept (X1.26), but cannot LEARN it from one problem's 60
    points of a mixture; consolidation was a no-op. Points to identifiability per problem, the target of X1.28 (noise-free, chosen inputs).
  - **X1.26 (2026-11-02, 48 fixed worlds; 3 of 4 PASSED): THE DEPTH GAP IS AT THE NODE. Fitted directly on clean data, one flat node
    expresses a 2+-deep concept at R^2 0.15, a tiny 2-layer net at 0.99; the learner only approximates deep concepts by many nodes (pool
    R^2 0.82).** Reuse count does not help; 4x data lifts deep problems 0.29 -> 0.40. Next: nodes that are tiny 2-layer nets.
  - **X1.25 (2026-11-02, 48 fixed worlds; P482-P483 PASSED tiny, P484 FAILED): consolidation (author's 'sleep': all nodes re-trained on
    every stored problem that uses them, every 30 problems) lifts deep problems by only +0.5 points (0.278 -> 0.283).** Node type, data per
    update and consolidation all fail to explain the deep gap; next: measure whether deep concepts are identifiable at all.
  - **X1.24 (2026-11-02, 48 fixed worlds; P478-P481 all FAILED): products and bends built on any existing nodes are linked heavily (most
    links) but deep problems stay at ~29% of the ceiling; offered every step they cost base problems.** The plateau is not a missing node
    type; candidate cause: nodes are only ever trained on one problem's 60 points. Next: consolidation (the author's 'sleep') across
    all stored problems that use a node.
  - **X1.23 (2026-11-02, 48 fixed worlds; P476, P477 PASSED): THE PLATEAU IS DEPTH. After the plateau, problems on base concepts reach
    88% of their ceiling; problems with a concept 2+ levels deep (two thirds of all problems) only 29%.** Memory helps most where depth
    is needed (+12 vs +5 points). Next: depth built on any existing nodes (products, bends, mixes), chosen by measured help.
  - **X1.22 (2026-11-02, 48 fixed worlds x 240; P473, P475 PASSED): in a fixed world the learner climbs for ~120 problems then flattens
    at ~40% of the ceiling (47% with twice the examples); the plateau arrives at the same time at both data sizes, so it is likely
    representational (one layer of nodes vs concepts built on concepts).** Memory's advantage grows with more data.
  - **X1.21 (2026-11-02, 48 loops x 400; P468 PASSED, P469-P471 FAILED): a fixed internal probe (re-score the stage's first 20 problems)
    makes stagnation detectable: stages 40-150. First stage climbs 27.7% -> 36.5% of the ceiling; later stages start higher and plateau
    ~35-37%. Memory advantage ~+9 points, not growing.** Next: what sets the plateau. Worlds: `curious-net/worlds/x1_21/`.
  - **X1.20 (2026-11-02, 48 loops x 400; P464 PASSED, P465-P467 FAILED): stagnation window 30 -> stages ~60; whole-loop share 0.35 vs
    no memory 0.26 (+9 points); advantage jumps after the first expansion then holds ~+10.** Detector still fires at its minimum (per-problem
    noise); no in-stage trend. Next: a low-noise internal progress signal. Worlds: `curious-net/worlds/x1_20/`.
  - **X1.19 (2026-11-02, 48 loops x 240 problems; P460 PASSED, P461-P463 FAILED): constant train-and-expand loop (expand by 12 concepts when
    the learner's own progress stalls). The learner holds a constant ~30% of the ceiling while the world grows 24 -> 96 concepts; no-memory
    control drifts 26% -> 22%; advantage +5 -> +8 points after expansions (slope n.s.).** Stages ~30 problems (noisy detector) are too short
    for in-stage mastery. Worlds saved for reuse: `curious-net/worlds/x1_19/` (wstream.load).
  - **X1.18 (2026-11-02, 360 worlds; P454-P459 ALL PASSED): FIRST CLEAN VERTICAL RESULT. In a finite fixed world (24 concepts, nothing
    new for 60 problems) the flat learner reaches more of its ceiling later (32.6% -> 37.3%) while the same learner with no memory stays
    flat (27.6% -> 27.7%): experience effect +3.8 points.** After one expansion (12 concepts built on old ones) prior knowledge absorbs the new
    concepts better (+7 points). Generalization 0.66 vs no memory 0.35 vs plain network 0.24. The X1.16/X1.17 declines came from the worlds.
  - **X1.17 VOID (author ruling: the world must not expand during a run).** Kept for the record: **(2026-11-02, 360 worlds; 1 of 6 PASSED): the near-infinite world (`core/winf.py`: nothing ever repeats, endless
    compositional concepts) is much harder** (flat later 1.17, generalization 0.61; plain network 0.67 / 0.19). Every learner reaches less
    of its ceiling later (-6 points) even with no growing evidence charge, so that cause is rejected; a confirm-only bar is worse.
    Next diagnostic: does a fresh learner also decline (world) or only experienced ones (experience hurts)?
  - **X1.16 (2026-11-02, 360 worlds; P442-P447, 1 of 6 PASSED): depth grown only on stagnation is rarely used and adds nothing; depth
    always on offer hurts.** FAIR vertical (share of each problem's own ceiling): ceilings are equal early and late, yet every learner
    reaches LESS later (flat 0.57 -> 0.54, plain network 0.41 -> 0.33): no learner here gets better with experience (corrects X1.15).
    Suspect: the evidence bar log2(#candidates) grows with what the learner knows (untested; X1.17).
  - **X1.15 (2026-11-02, 360 fresh worlds; P436-P440 PASSED, P441 FAILED): the flat learner optimised (`core/flat.py`, `OPT`).**
    Tuned on separate worlds: +0.30 bits on later, end and generalization (0.93 -> 1.26, +34%); vs plain network +0.87. Gain is mostly
    statistics (readout ridge 0.001 -> 3, patience, best-of-4 new nodes); specificity bonus and old-problem re-checks confirmed essential.
    Vertical drop is real, not confounding (X1.16 correction). New base for growing complexity.
  - **X1.14 (2026-11-02, 540 worlds; P429-P435, 2 of 7 PASSED): decreasing reward for consistently used nodes (2^-recent use).** With
    the distance reward: more nodes (28 vs 19), old problems +0.16, generalization unchanged. Alone: hurts the flat learner (-0.08
    generalization) by cutting reuse of nodes that are used often because they are useful. Flat X1.9 still best (0.96).
  - **X1.13 (2026-11-02, 360 worlds; P423-P428, 1 of 6 PASSED): paying more the less direct the assistance makes links reach twice as
    far, but the learner then shares 20 nodes across unrelated problems instead of building 84, they get retrained in opposite directions,
    and everything drops** (final 1.05 vs 1.61, generalization 0.77 vs 0.93). Lesson X1.9-X1.13: many specific nodes reused only where
    they truly help generalize best; next: let distant problems USE a node but never retrain it (X1.13b).
  - **X1.12 (2026-11-02, 720 worlds; P416-P422, 2 of 7 PASSED): confirmed links (help on two separate halves) halve the nodes and add
    +0.05 bits, but generalize WORSE to new combinations (0.76 vs 0.85): correct-for-now is not useful-for-later.** Frontier bonus: +33
    nodes, no effect. The one-layer learner generalizes best (0.94); every tree drifts over the stream (shared trunk retrained by every
    problem). All beat the plain network (0.40).
  - **X1.11 (2026-11-02, 360 worlds; P410-P415, 3 of 6 PASSED): a precision reward (1/error, exponential near 100%) behaves like
    the depth reward**: ~19 nodes per problem, -0.10 bits vs the unshaped tree. A bigger reward only lowers the bar against a fixed cost,
    so more lucky gains get in. Self-weighting positive for the third time (+0.03). Lever to test next: confirm gains on fresh points.
  - **X1.10 (2026-11-02, 360 worlds; P403-P409, 4 of 7 PASSED): the author's tree (enter at the base, grow, only used nodes learn,
    reward x 2^(nodes used) with no ceiling, internal self-weighting).** The uncapped depth reward makes problems use ~21 nodes where ~9
    do better (later bits 1.27 vs 1.40) and later problems get worse: tiny measured gains on few points are mostly luck. Self-weighting
    (+0.04) and use-only updates (+0.03) help a little. Beats the plain network (+0.53) but loses to X1.9's flat learner (-0.20): it forgets.
  - **X1.9 (2026-11-02, 2880 worlds; P395-P402, 7 of 8 PASSED; X1.8 withdrawn): training only the fewest nodes (author) protects old
    problems, but barely (+0.03 bits), and costs 0.25 bits against training every linked node** (final 1.74 vs 2.01; none trained 1.47).
    With stored examples and re-fitted readouts, moving shared nodes hardly hurts old problems: learning, not forgetting, is the limit
    here. With equal re-fits for all arms the rule still beats the plain network by ~1 bit (X1.7 correction does not overturn it).
  - **X1.7 (2026-11-02, 1440 worlds; P381-P387 all PASSED; author's rule I21): growing by connecting, with every link paid by its exact
    measured help, works, and the specificity bonus does the work.** Later-problem bits: rule 1.39 vs help only 1.23 vs no reuse 1.02 vs plain
    network 1.16 (which forgot: 0.51 at the end vs 1.47). Newcomer bonus alone ~0. Bonuses paid on existence: 3.6x the nodes, worse;
    connect-everything collapses. Gets better at new problems with experience and old problems improve as nodes arrive. Nodes did NOT
    become cleaner concepts (match fell slightly): the bonus wins by linking more. Open: better links or just more links (X1.7b).
  - **X1.47 (2026-11-04, 48 worlds; P563-P567; author's request I29): the X1.6 learner does NOT map a purely linear world, so
    X1.6's ladder had no mastered base rung.** 6 outputs, each linear, hidden rule a stack of 1-5 linear layers, no noise. The
    tanh learner gets close in range (worst-output R^2 ~0.99; 0.85 of tasks mastered after 10 meta-learned steps, 0.55 after
    4000 Adam steps) and masters NONE out of range (input sd 3: R^2 0.77-0.82). Same net without tanh: 1.00 in and out; least
    squares 1.00. Depth of the linear rule does not matter. Lesson: gate each rung (in AND out of range) before climbing; the
    tanh units, not the optimizer, are what stop the network holding the line. Next candidate: linear skip path, re-gate.
  - **X1.6 (2026-11-01, 48 worlds; P376-P380; author's request I20): training a network's learning machinery on its rate of
    gaining structure is NOT a climber.** Meta-SGD, equal tasks and compute. Rewarding bits held on unseen points raised
    structure overall by piling it on the easiest rung (rung 1: 2.80 vs 1.72 bits) and gave up harder and never-trained rungs
    (rungs 4-5: 0.17 vs 0.44, no better than untrained; P376 FAILED). Rewarding 'complexity' on the points it saw is a trap: it
    memorises noise and loses real structure (P378 PASSED). One fixed-goal rung transferred as well as all rungs (P379 FAILED, U3
    not supported here). Lesson: a growth drive needs a headroom term (what is LEFT to gain), i.e. U12 and boredom, or it drifts
    down to cheap structure. Next candidate X1.6b.
  - **X1.1 (2026-10-27, 48 worlds; P244-P248): the dark room is real for a surprise-minimising drive.**
    With a perfectly predictable room, an unlearnable noise region and learnable regions (one opening
    halfway), minimising surprise (the program's first design rule) parked in the dark room 93% of
    the time; error-seeking sat in the noise 92%; **learning progress** (the slope of the error) avoided
    both (53% of the first half in the traps vs 96 to 97%) and spent 69% of the second half on the
    learnable regions. But it was **6 times slower than random** at learning the newly opened region
    (389 vs 65 ticks): an unvisited region has no value and a one-shot learner shows no progress until a
    first pass is done, so it needs optimism toward untried regions. Supports U12 and Q9 (the edge is
    where error is falling) and bears on protocol s7 item 8; the learning-progress drive alone is not enough.
  - **X1.2 (2026-10-27, 48 fresh worlds; P249-P252): an optimistic start is the missing ingredient for
    learning progress.** Untried regions start at the largest reward the drive can pay (1.0, derived, not
    tuned). Learning progress with optimism found and learned the newly opened region in 16 ticks (plain
    learning progress 336, random 58.5, the perfect learner 16) while keeping the trap avoidance (53% of
    the first half in the dark room or noise vs 95 to 97%). It does not rescue error-seeking: that drive
    finds the new region in 17 ticks and then camps in the noise (91%): optimism governs FINDING, the drive
    governs where the agent STAYS. Together: stay where the error is falling, move on when it stops, try
    what you have not tried (the toy form of Q9 and of 'curiosity is part of the answer, not the answer').
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
    (series 2). Series 3 reported 0.51x default / 9.4x tuned on 7 problems at a 3e6 budget; X6.5 then reported 0.60x default / 11.2x tuned at 1e7. **SUPERSEDED (audit 2026-10-28, [[audit-2026-10-28]]): series 2/3 and X6.5 stopped on the test set, tuned the two sides unequally and quoted tuning-seed numbers (winner's curse); they are VOID as headlines. The repaired benchmark BENCH2 (2026-10-28, P282-P287; validation-based equal-interval stop, equal tuning grids, held-out seeds) gives: tuned evolution about 12.5x costlier than tuned Adam on six problems; default vs default 0.69x; evolution unable on checker3; evolution's tuning-seed cost inflated 1.48x. The X6.6/X6.9 gains of 1.2-1.3x are indistinguishable from seed-set noise under the same standard.**
  - Greedy rules (undo + barred change types, the separate stack) are **worse** than noise on
    all weights: they stall where no single change helps (attempts 3-4).
  - Random-cull multi-problem brains plateaued at the **linear ceiling** (0.55-0.61 vs 0.973 for a
    small trained net): **depth was selected away** (T 2 -> 1), **variety collapsed** to 1-2
    lineages, and refilling from random always stayed at baseline (selfscore series, closed).
  - Selection math: Price identity exact (C1, not novel); interior optimum cull 75-95% under
    noisy scoring (C4); the multi-generation law was refuted (C5).
  - **X6.8 (2026-10-27, 20 runs per arm, budget 1e7 including every stage; P230-P233): stepping
    stones do NOT help.** The direct optimizer solved 10/20 (spirals 10/10, checkerboard 0/10); the
    self-paced ramp also 10/20 (faster at the median on spirals, 0.95M vs 1.35M, but heavy-tailed,
    0.5M to 6.0M), the fixed ramp 4/20 and the mixed stages 0/20 (both worse); **the checkerboard
    was unsolved in all 40 ramp runs**. A plausible reason (untested): a population converged on an
    easy stage has lost the variety and noise size it needs to move on.
  - **X6.9 = I10 (2026-10-27, 70 runs per arm, budget 1e7; P234-P238): the capacity to grow gives a
    modest, consistent gain.** Function-preserving unit duplication at a fixed rate with no gene
    (u = 0.1, 0.3, 0.5, 1.0): all arms solved 60/70 like the baseline; cost 1.16x, 1.03x, **1.23x**,
    1.17x cheaper overall (pass line 1.25x, missed by 0.02) and 1.16x to 1.43x on spirals; nets grew to
    width 23 to 30; the checkerboard stayed unsolved at every rate. Same gain as the heritable-gene arm
    of X6.6 (1.22x): growth helps a little, it is not the missing ingredient, and the differences between
    rates are within noise at 10 seeds.
  - **Reading across X6.5 to X6.8 (assistant; for the author to correct):** none of the things
    added to blind variation plus culling closed the gap to gradient descent: its tricks did not
    evolve themselves (X6.6), selection without any grader works but at about 26 times the cost
    (X6.7), and stepping stones did not help (X6.8); the repaired benchmark is about 11 times behind
    tuned gradient descent (X6.5). This fits U18: gradient descent is probably the fastest way to pick
    up a problem, and what is missing is what manages it. What the evolutionary side does show is
    that selection alone, with no grader, is enough to find solutions, so the grader is a convenience,
    not a requirement.
  - **X6.7 (2026-10-27, 20 seeds, budget 3e6; P225-P229): selection with NO grader works, slowly.**
    A population with nothing ranking anything (food shared among the nets that answer correctly,
    a living cost, split at energy 2, die at 0) solved a noisy XOR to 0.95 in **17 of 20 runs**
    (needs 18), at a median cost of 1.37M, **about 26 times the graded cull** (51.5k) and 25 times
    default Adam (54.2k). Random search solved 0/20; unlimited food made the population explode
    in 20/20 (so competition for the limited resource is what selects); accuracy predicted offspring
    in every run (Spearman 0.30). **The units matter:** with the energy change summed over a whole
    step against a 0-to-2 band, the population went extinct in 20/20 runs. Energy as a budget must be
    large relative to the noise of one step.
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
