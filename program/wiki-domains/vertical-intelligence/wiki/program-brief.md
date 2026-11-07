---
title: Vertical Intelligence - Program brief (two pages for sponsors, advisors and collaborators)
domain: vertical-intelligence
kind: brief
tags: [brief, sponsors, pitch, summary, results]
updated: 2026-10-30
---

# Vertical Intelligence: program brief

*Two pages for a reader deciding whether to advise, join or fund this work. Every number traces to a
registered prediction (P-number) or result id in the public repository. Updated 2026-10-30.*

## 1. The question

Modern machine learning trains a network on one objective by gradient descent. It works, and it is
probably the fastest learner we have. But a plain gradient-descent learner has no stake in anything
("better or worse for *me*" does not exist for it), no drive of its own (it cannot get bored or choose
what to learn next), no notion of which situation it is in, no own experiments, and no way to say "I
am out of my depth". This program asks: **what does gradient descent alone lack, and can each missing
piece be isolated, tested fairly, and then combined?** The ideas come from biology (evolution,
energy, parts of the brain), psychology (motivation, curiosity, the Self and its parts) and the
author's own theorems; the tests are small, controlled computer experiments.

The long goal is a model that learns and creates semi-new things. The final exam we have set: given
only pre-quantum physics and the measurements of that era, the model must notice that the classical
laws fail and propose that something like quantum theory must exist. Nothing here claims that yet; the
exam is the direction, and the work is a ladder of smaller, provable steps toward it.

## 2. How the work is run (the part that is unusual)

- **Pre-registration.** Every success rule is committed to git before the scored run. 352
  predictions so far: 168 passed, 170 failed, 14 voided or resolution rules. Failures are reported
  with the same prominence as passes.
- **Fair comparison.** 48 independent worlds per experiment, paired bootstrap confidence intervals,
  baselines tuned on their own objective with the same budget as the method, defaults also reported,
  manipulation checks, held-out seeds for every headline number.
- **Self-audit.** On 2026-10-28 every past result was audited against those rules; several were marked
  VOID and are being redone; the evolution benchmark was rebuilt (BENCH2) and its headline reversed.
- **Earn every component.** No mechanism enters the combined model until it has passed its own test
  alone; combinations are then tested against their parts.
- **Readable instrumentation.** All runs are on Weights & Biases with plain-language names so a
  non-specialist can read the data.

## 3. What has held up (toy worlds; principles in miniature)

1. **Attend to every need.** An agent whose value is its own viability learns to attend to all its
   needs; ignoring one is 4 to 15 times deadlier than any rule for combining them, in five different
   worlds (X0.7-X0.10, X1.3).
2. **A prediction-only life.** An entity with no food or goal, whose only needs are "my predictions
   must not fail" and a hidden need refilled only by real learning (felt as rising boredom), stays
   alive and avoids both the dark room (perfect predictability) and unlearnable noise: 0 deaths in 48
   worlds, against 6.7 to 21.7 per 10k for entities that ignore boredom, minimise surprise or act at
   random (X1.3). An exact "bits learned" signal alone does the same; a noisy boredom signal degrades
   gracefully (X1.4).
3. **Curiosity needs optimism.** Learning progress avoids the noise trap but is 6x slower than chance
   at finding newly learnable regions; an optimistic prior for untried options fixes it and matches an
   oracle (X1.1, X1.2).
4. **One hunger is enough for the secure base.** A single "expected change for me" signal produces
   explore-when-safe, retreat-under-threat, in proportion to the stakes, with nothing tuned (X0.2).
5. **Evolution versus gradient, fairly.** Under a validation-based stop, equal tuning and held-out
   seeds, tuned evolution is about 12.5x costlier than tuned Adam on six small problems; selection with
   no grader at all still finds solutions, at about 26x the cost (BENCH2, X6.7).
6. **A Self over sensors.** A small supervisor that reads several sensor networks' complete internal
   states beats the best single sensor and the best hand rule (0.917 vs 0.841), ignores junk sensors,
   and gains from reading insides rather than answers; with few labels it memorises and loses doubt,
   and refreshing the labels of unanswerable cases restores doubt at a competence cost (X7.2-X7.6).
   Recognition of which stored routine to reuse can be learned from audit verdicts alone (96%; X8.1),
   but creating routines from scratch only reaches parity with no memory (X8.1b-e, parked).
7. **Laws from real data.** Kepler's third law recovered from NASA planetary and moon data by model
   selection (exponent 1.4978 / 1.5002), competing forms rejected, a 0.4-0.7% disagreement between
   two NASA sources found along the way (E9, E9c).
8. **Smaller findings that travel:** a learned relevance gate beats a tuned soft screen by +0.035
   (C30/C33); easy-to-hard curricula are 2.4x cheaper (C53); transfer is proportional to shared
   structure (C43); replay protects old knowledge at about 5% cost to new (C35).

**Negative results kept:** locking weights fights relearning; sleep replay is null unless waking
learning saturates; populations of random variants never beat the single best; stepping-stone
curricula did not help evolution; three default regularisers did not stop a Self from memorising.

## 4. Where it is going

- **The proof ladder** (each rung tested alone with a planted null before use): dependence beyond
  correlation; cause versus correlation by self-run interventions; functional form outside a fixed
  library; flagging anomalies at a fixed false-alarm rate; inventing a missing constant; proposing a
  principle. Rung 3 on real historical data (the Balmer series) is the next real-data step.
- **The Self line**, now the centre: a controller over sensor networks that holds a goal, reads
  their internal states, knows when not to answer, and manages a library of routines; next stages use
  real data and real small models, then LLM sensors (needs GPU).
- **An inner state richer than weights** (author, 2026-10-29): a persistent, integrated, self-sensed
  state that the entity acts on; first test is a felt-state ablation against a scrambled control.

## 5. What we are asking for

An **advisor of record** at the author's institution; **GPU time** (one 24-48 GB GPU, or cluster
hours) for the real-model stage; **two to five collaborators** with defined first tasks (see the
collaborator page); and, with an advisor, applications for compute credits and a small grant. In
return: a public, reproducible, pre-registered program with a working onboarding path and co-authorship
on the preprint.

## 6. Honest limits

All results are from small simulated worlds on one laptop; one real-data result exists. The Self line
is weeks old. The program has one author and AI coding assistants; no publications yet. The long goal
is far away and is stated as a direction, not a claim.

Repository: github.com/Ally-for-Hire/vertical-intelligence (start at `README.md`; agents start at
`AGENTS.md`). Charter: `program/wiki-domains/vertical-intelligence/wiki/research-protocol.md`.
