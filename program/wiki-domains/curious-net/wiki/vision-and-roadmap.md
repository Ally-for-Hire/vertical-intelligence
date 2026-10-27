---
title: Curious-net: end-goal vision and roadmap (user's statements, 2026-10-22)
domain: curious-net
kind: project
tags: [vision, roadmap, llm, recursive-self-improvement, calibration, curiosity, skepticism, parked-ideas]
updated: 2026-10-22
---

# Curious-net: end-goal vision and roadmap

Source: the user's own words in the 2026-10-22 session (raw capture: `raw/2026-10-22-vision-statement.md`). Statements are the user's givens; "Note" lines are the assistant's comments and are labelled as such. **Everything not in the immediate scope is parked here so it is not lost.**

## End goal (user)

A **large language model that uses this algorithm to recursively self-improve**, ideally with a literal physical (hardware) brain that scales the way biology and psychology do. Given internet access it would self-research and self-expand until it has no threads left to pull. The user explains abstract concepts to it; it connects each to everything previously learned and discussed, understands it **immediately and completely**, and then applies it to future problems. Given an understanding of physics it should be able to **discover quantum theory**: know where its understanding ends, research and experiment in its own way, derive the theorem (the theorem "exists in the universe"; it can be derived from knowledge that can be attained). Efficiency is measured in **generation count** (not necessarily time or compute): each applied problem makes the next one faster to understand.

Advantage over current AI (user): not correctness but the ability to throw the whole human library at a task. Target: an AI that **never gives a wrong answer without knowing it is wrong**: it knows its own limits, knows when an answer is "not the complete picture" (missing variables), and says so.

**Implicitly curious and skeptical (user):** every input is treated as subjective and is checked against all previous inputs; if no connection can be found, it wants to learn more.

**Understand the situation first, then answer.** Contrast (user's understanding): a transformer LLM predicts the next word. This one should first build an understanding of the situation in its entirety and then use it to interpret and explain.

**Worked example (user):** "A dropped ball hits the ground in 5 seconds." The system should put that into its brain and connect it with physics, gravity and the world, or conclude that it cannot explain it and must research until it can; then answer "dropped from 15 m, how long to fall?" by understanding, not by next-word statistics. (Note: free fall from 15 m takes about 1.75 s at g = 9.8 m/s^2, so the claim of 5 s is inconsistent unless something else is going on (air, a different planet, a wrong height); a skeptical system should notice that.)

**Working method (user):** self-run existing LLMs that monitor it offline and teach it with minimal human involvement; the user can "sit and let it grind"; then talk to it, have it ask random questions, and get complete answers.

## Principles to test (turned into measurable properties)

1. Parity at zero experience: with no experience it behaves exactly like a plain network (done: bitwise).
2. Experience improves how fast it identifies what affects what (done for MLP, diverse worlds; transformer next).
3. Recursive self-improvement across **generations**: each generation's own outputs (importance labels, experiments) train the next generation's prior. Metric: error / identification speed vs generation count, plus whether it plateaus or degrades (self-confirmation bias).
4. **Calibration and abstention** (the testable form of "never wrong"): it should output a confidence that matches its accuracy and say "not enough variables" when out of its depth; metric: accuracy vs coverage curve, expected calibration error, rate of confident-wrong answers. Note: literally never wrong is not attainable; "knows when it might be wrong" is.
5. **Skeptical claim checking**: treat a statement as a hypothesis, check it against the learned world model, return one of {consistent, inconsistent (with the evidence), no connection (needs research)}. Metric: accuracy on planted true/false/unrelated claims.
6. **Curiosity**: when a claim has no connection, choose experiments/sources that reduce the uncertainty fastest (the closed-loop experiment policy).

## Staged roadmap

- **Done (iterations 1-12):** transformer and MLP bases with a relevance gate; headroom, experience curve, generations (no collapse), parity, baselines, out-of-family, fly-like and connectome worlds, law discovery, teacher toys, quantum and emotion null tests.
- **Now (from iteration 13, after the independent review):** a fair no-experience baseline, then the ORACLE-TEACHER direction: calibration of a taught student in and out of what it was taught, detecting the edge of the taught set, three-way claim verdicts with abstention, contradicting data, non-oracle sources (QUEUE E1-E5).
- **Next:** closed-loop experiment policy (learn how to manipulate the environment to expose structure); claim-checking micro-benchmark in simulated physics worlds (the ball-drop example generalised: claims about worlds the agent has experienced); calibration/abstention measurement.
- **Later (parked):** symbolic/natural-language grounding (map text claims onto the learned world models); LLM teachers/monitors that run offline and curate lessons; a corpus pipeline; internet self-research loop (retrieval, source skepticism, provenance); long-run "grind" infrastructure (checkpoints, budgets, stop conditions); discovery tasks (derive a law from data the agent collects: e.g. re-derive free-fall law, then harder: blackbody/quantum-like anomalies); hardware/neuromorphic realisation of the algorithm; sleep as a separate study of NEURAL-NETWORK learning (user: isolate variable; earlier sleep results in LOG.md); conversational interface; combining everything into an LLM.

## Cautions (assistant notes, not user statements)

- Unattended internet self-expansion needs sandboxing, rate limits, source provenance and a kill switch; "until it has no threads to pull" has no natural stopping point and can run away in compute and in acquiring junk; define budgets and review points before building it.
- Recursive self-improvement loops that train on their own labels can confirm their own mistakes; every generation must be evaluated on held-out ground truth the loop never saw.
- The user's belief that neural networks cannot get faster across problems is, in general, false (meta-learning, in-context learning do this in-distribution); the project claims should be: exact parity at zero experience, explicit effect structure, continual improvement without retraining, and behaviour on out-of-family / intervention-only problems; each needs its own test.
- "It should never be wrong" is not a measurable target; "its confidence matches its accuracy and it abstains when out of its depth" is.

## Parked: a second algorithm based on the IFS (Internal Family Systems) view of the brain (user, 2026-10-22)

The user wants, eventually, a new algorithm built on the IFS understanding of the brain with three components as explored in NeuroBlock: **the sensors, the emotions (emotional converter), and the Self**. "Just something to think about"; not started. Pointers for whoever picks it up: `~/neuroblock-research/FOUNDATIONS.md` (the brain is split into parts; building the Self is the main focus; emotional converter v0 = a few signals computed from the input stream and body that the Self reads: alarm = surprise rate, relief/interest = surprise falling, strain = low energy / high fatigue, curiosity = surprise it can afford to chase; the Self picks among explore / consolidate / etc. and pays for being a Self; a self-model predicts the Self's own next error) and wiki pages `domains/neuroblock-research/wiki/foundations-and-energy-2026-10-18.md`. Open questions recorded there: which parts besides emotion, intellectual converter and Self; whether emotions are only signals the Self reads or also change other parts' behaviour; whether the Self has goals beyond lowering uncertainty cheaply. Natural bridges to the current line: "curiosity = learning progress" and "skepticism = unexplained input" are emotion-like signals; the claim-checker and the experiment policy are Self-like controllers; the relevance prior is part of the sensor/attention hub.

## Parked: the three-brain architecture and recursive part generation (user, 2026-10-22; the user's wording, condensed)

Needed: **three different pieces.**
1. **Intellectual side**: takes all the connections the system has and creates connections it did NOT previously have, from its knowledge and understanding of the situation (not necessarily new ideas: completed/inferred links).
2. **Emotional side**: where *new ideas* come from. "Somewhat random weights and biases" that produce thought processes individualised to maximise learning and to find deeper connections than are inherently obvious (structured randomness / individual temperaments that explore differently).
3. **Sensor side**: understands the world and is used to try to understand it.
**The Self sits at the base and has NO parts of its own (user correction).** Parts and sub-parts are **generated as situations require more context and understanding**, recursively: a part can create sub-parts. This allows continuous learning and growth in complexity; it does not immediately create new thoughts, but a sufficient level of complexity should produce thoughts at a somewhat human level.

Assistant notes (not user statements), how this lines up with what exists and what could be tested:
- Intellectual side ~ link completion over the learned causal graph: given edges and tables, propose and score missing edges (abduction); testable as held-out link prediction accuracy and, in the claim-checker, "no connection found" vs "connection inferred".
- Emotional side ~ a population of differently-initialised/biased explorers (random priors over which relevance to try, random experiment templates) whose diversity is itself measured; testable: does a diverse population find a hidden structure faster than a single best-tuned learner (the usual ensemble / quality-diversity question), and does the diversity have to be *structured* rather than pure noise.
- Sensor side ~ the relevance-gated attention and descriptors built so far (what affects what).
- Self with no fixed parts + on-demand part generation ~ the spawn-on-unexplained-input mechanism of NeuroBlock and of the count-table learner (a new neuron/table is created only when the current model cannot explain the input, charged by an MDL cost); recursion = a part that itself spawns when its own sub-problem is unexplained. Open design question: what triggers a part vs a sub-part (a persistent unexplained residual inside an existing part).

## Parked / future: brain scan -> code -> optimise, with the fruit-fly connectome as the stepping stone (user, 2026-10-22)

User's belief: the future of AI is a human brain scan so precise that a brain can be coded, after which optimisation begins; the user wants to "get a jump on" that future and to treat this project as an expansion of the fruit-fly brain scans.
Facts established (searches; see `paper/LITERATURE.md` in the project): FlyWire whole-brain connectome (Nature 2024: ~139k neurons, ~5x10^7 synapses, public data, CC BY 4.0); hemibrain (eLife 2020); a connectome-based whole-brain spiking model predicts taste/feeding circuits (Shiu et al.); connectome-constrained optic-lobe network with parameters fitted by deep learning matches 26 studies and works better with sparse connectivity (Lappalainen et al. 2024); Whole Brain Emulation roadmap (Sandberg & Bostrom 2008) and the State of Brain Emulation Report 2025. Known gaps: synaptic weights, neuromodulation, gap junctions, plasticity; the extent to which wiring predicts function is debated.
Plan in measurable terms (queue Q18-Q21): measure real wiring statistics against the project's generator assumptions; build connectome-derived simulated worlds and test structure recovery from activity and interventions; compare a prior learned from synthetic worlds with one built from connectome statistics. The project's learner is positioned as the inference step a scan cannot supply (what affects what, and how strongly), validated first where the answer is known (the fly).
Assistant notes (not user statements): "code a brain from a scan" is plausible at circuit scale and unproven at whole-brain scale; the scan is necessary but not sufficient; the 2008 roadmap and the 2025 update are the right yardsticks; a human-brain-scale scan is orders of magnitude beyond the fly (the fly has ~1.4x10^5 neurons, a human ~8.6x10^10 [unverified figure]).

## Product requirement: continuous teaching by an offline LLM parent, starting from literally zero (user, 2026-10-22)

The user wants the large-language-model product to be **taught continuously by an offline LLM, starting from literally zero (not even an understanding of human language), with the offline LLM acting as parent and teacher.**
Literature tied in (searches; `paper/LITERATURE.md`): TinyStories (LLM-written child-vocabulary curriculum trains tiny LMs), BabyLM (children < 100M words), Hermann et al. 2017 (grounded word learning from minimal prior knowledge, accelerating), BabyAI (teacher bot + curriculum; methods still sample-inefficient), Lazaridou et al. 2017 (emergent protolanguage in referential games), Zhu 2015 machine teaching (optimal training set / teaching dimension), Kirby-Cornish-Smith 2008 (iterated learning increases structure), Saffran et al. 1996 (statistical word segmentation in infants).
Assistant notes (not user statements): (1) "zero" has to begin with a pre-linguistic, grounded, statistical stage; (2) a student can only exceed its parent if it has its own experiments/verifier (this is the curious-and-skeptical principle made mechanical); otherwise it inherits the parent's errors, and a student-becomes-teacher chain can collapse (Shumailov 2024) or gain structure (iterated learning), depending on bottleneck and verification; (3) the teacher LLM's own errors, biases and sycophancy are part of the system; (4) an "offline" parent means a locally-run open-weights model: choosing and downloading one is parked for the user (hardware, licence); a scripted teacher is used first. Claims C24-C28 and queue Q23-Q28 in the project's paper/ directory.



## Redirect 2026-10-22: oracle teacher
USER REDIRECT 2026-10-22 (iteration 12): the teacher is an ORACLE (objective truth of our understanding of the world). Priority is now: train the student on oracle-taught truth from zero, THEN measure whether it ends with healthy skepticism and realistic expectations (calibration of confidence, knowing the edge of what it was taught, abstention, correct handling of claims/data that contradict or exceed the taught knowledge). Discovering uncertainty on its own is secondary. Noisy-teacher / reliability-fusion work (Q24b, Q30b, Dawid-Skene) is demoted.


## User answer 2026-10-22: the oracle and its limits
USER ANSWER 2026-10-22 (after iteration 13) to the oracle-vs-skepticism question: the oracle should teach about its own limits, OR necessarily teach in a way that EXPOSES them; which of the two is unclear and is itself something to work on. The goal: a learner that learns from REAL data, can poke holes in bad theories, and can derive laws from data. Tests: QUEUE E8 (facts only vs stated scope vs teaching that exposes the edges) and E9 (real planetary data: derive Kepler's third law, reject bad theories).


## User requirement 2026-10-22: the student runs its own experiments
USER REQUIREMENT 2026-10-22: the student must be able to RUN ITS OWN EXPERIMENTS; this is vital. Consequence: every oracle-teaching test (E3, E4, E5, E8, E9 where the data allow) gets the student's own experiment budget as a standard capability, and the student chooses WHICH experiment to run (design by expected discrimination between the taught theory and its rivals: T-optimal / expected information gain), not just random checks. Q6 (closed-loop experiment policy) is un-parked and merged into these tests: measured as verdict accuracy and revision speed per experiment spent, chosen experiments vs random experiments vs no experiments.


## User direction 2026-10-22: perfect instruments by default
USER DIRECTION 2026-10-22: do not assume bad measurement tools. The student's own experiments are PERFECT (noise-free measurement) by default; the learner must still be ABLE to process noisy data when it has to (real data sets, E9). Implementation: X1/E3/E4/E8 run noise-free as the main condition; noisy measurement is a robustness arm, and the learner never reads the noise level from the generator (it estimates it, and an estimate of zero must be allowed). Assistant notes (to test, not assumptions): (1) with exact measurements the Q12c resolution limit disappears for the measured quantity: any deviation from a law is detectable with enough distinct experiments, so 'poking holes' becomes a question of choosing the right experiment and of the hypothesis library, not of statistics; (2) measurement noise is different from INTRINSIC randomness of the world (hidden variables, stochastic dynamics such as the causal worlds' process noise); perfect instruments do not remove the second, so the causal-world tests keep their process noise; (3) floating-point precision is the remaining floor (relative ~1e-16), so 'exact' tests need a tolerance tied to numerical precision, declared as a constant.


## THE DREAM (user, 2026-10-22; supersedes earlier summaries of the end goal)
THE DREAM (user, 2026-10-22, correcting earlier summaries): a large language model that can DERIVE QUANTUM PHYSICS from our understanding of science before quantum physics. It is given only pre-quantum (classical) physics and the measurements, never told quantum theory exists, and must propose that something like quantum theory exists (it need not be an in-depth theorem). More broadly: an LLM that can actually learn and create semi-new things. All other components (what-affects-what, experiments, oracle teacher, skepticism, connectome) serve this.
Benchmark: QUEUE Q-PRE1900 (blackbody spectra, hydrogen lines, photoelectric effect; pass = flags classical failure and proposes quantisation / a new constant).


## THE DREAM, capital D (user, 2026-10-22, supersedes the 'pre-quantum LLM' wording as the end goal)
THE DREAM (capital D), user 2026-10-22: a better transformer, a new kind of AI model that changes how AI operates, based on biology, psychology, and our current understanding of intelligence, knowledge and wisdom; the pre-quantum-physics test is its proof milestone, not the Dream itself; all curious-net and NeuroBlock work = small experiments for this. Zoomed-out synthesis: paper/THE_DREAM.md and wiki the-dream. See [[the-dream]].


## User direction 2026-10-22: rigor phase before the next arc
USER DIRECTION 2026-10-22 (after the Dream synthesis): 'we aren't done diving into the architecture of the learning process yet. I want to refine and perfect every individual part of this based on as MUCH data as we can possibly get, so we FULLY understand every single result and can make accurate theorems and predictions that hold up.' -> RIGOR PHASE: the merge/Self/transformer arc waits. Every part of the learning process is re-measured at scale, every constant swept, every mechanism given a theorem with a prediction registered BEFORE its test (paper/PREDICTIONS.md), and a result is called SUPPORTED only after a pre-registered prediction passes on fresh data.


## User direction 2026-10-22 (iteration 17): all pieces first, then individual rigorous tests
USER DIRECTION 2026-10-22 (iteration 17): do not hyper-dive on one angle; look into ALL relevant pieces and science; figure out which parts and aspects of the brain actually matter in a perfect self-curated learning machine. Plan: FIRST figure out all the relevant pieces, THEN test them individually and rigorously until we have refined, evidence-backed versions of all the baseline and foundational ideas. User's baseline assumptions (givens): A1 human brains are only as smart as they had to be for evolution; A2 the perfect thinking machine is based on some fundamental understanding we don't have yet; A3 the start of this will be a perfect human brain scan. Inventory written: paper/PIECES.md (3 assumptions + 26 pieces in brain / psychology / theory / AI, each with evidence status and an individual test; ranked order). The rigor rules apply to every piece; RIGOR items continue as the rigor treatment of piece B4 only.
