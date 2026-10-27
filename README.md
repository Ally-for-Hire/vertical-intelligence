# Vertical Intelligence

A deep dive into **evolution and the nature of intelligence**, with the express goal of applying
what it finds to **machine learning**. The question underneath everything: how did nature
produce minds that get *better at learning itself*, not just better at one task, and what would
a machine learner built on those principles look like? The work combines biology (evolution,
how brains work), psychology (motivation, curiosity, the parts of the self) and computer
science, and tests each idea with small, carefully controlled computer experiments.

> **"Vertical" intelligence** means growth in *how well* a learner learns: each new problem is
> solved faster and with less effort because it connects to everything learned before, and the
> learner knows where its own understanding ends. Most of today's machine learning is trained
> hard on one goal; this program asks what evolution and the brain do that is missing from that picture.

The long-term milestone is a model that, given only the science known before 1900 and the
measurements of that time, works out on its own that something like quantum theory must exist.

**Stage: early and exploratory.** Every experiment here runs in a small simulated "toy world"
on a laptop. The results are principles shown in miniature, not claims about real brains or
about large AI systems.

## What has been found so far

Each result has an id (like C43) and a confidence letter: **R** = replicated on fresh data,
**S** = supported once, **P** = partly supported, **D** = a diagnosis only, **N** = not supported.

- **Learning transfers in proportion to shared structure.** A learner that has seen related
  problems solves new ones up to 7.5x faster; on unrelated problems it gains nothing, as it should (C43, R).
- **Less is more, repeatedly.** Giving a learner an "energy budget" for its connections cuts its
  error in unfamiliar situations by 12.8% while using 5% of the connections (C41, R). Focused
  models and sparse connections win in other tests too (C47, C50).
- **Curiosity's proven job is avoiding what cannot be learned**, rather than choosing between
  learnable things (C40). **Motivation plus curiosity** beats either one alone (C52, R).
- **Order of teaching matters.** Easy-to-hard lessons are 2.4x cheaper than random order (C53, R).
  A student taught only correct facts is confidently wrong outside them (C31, S).
- **Knowing "which situation am I in?"** is the shared bottleneck behind several abilities:
  mastery, using specialised parts, and memory (C44-C46, D).
- **A brain's wiring diagram gets you about halfway.** Using the real fruit-fly connectome,
  whether each connection excites or inhibits matters most (C42, R).
- **Laws from real data.** Kepler's third law was rediscovered from real NASA planetary data,
  and deliberately wrong theories were rejected (C57).
- **Evolution without calculus.** Starting from pure random networks, blind random changes plus
  "survival of the fittest" solve learning problems with no gradient descent (the calculus-based
  method that trains modern AI). On seven test problems it reached its targets at about half the
  cost of gradient descent with standard settings, though still about 9x behind well-tuned
  gradient descent. Current question: how could nature have optimized life before anything
  like gradient descent existed?
- **The sense of "good or bad for me."** A simulated creature that learns a model of its own
  survival dies half as often as one that only learns from deaths (X0.6).

Failures are kept and reported alongside successes; the logs record every attempt.

## How to explore it

If you are reading for the ideas, start here:

1. [`program/wiki-domains/vertical-intelligence/wiki/research-protocol.md`](program/wiki-domains/vertical-intelligence/wiki/research-protocol.md):
   the program's charter. Section 1 lists the author's founding statements (U1-U18) and the
   main open questions (Q1-Q10).
2. [`program/wiki-domains/vertical-intelligence/wiki/findings-map.md`](program/wiki-domains/vertical-intelligence/wiki/findings-map.md):
   what has been learned, organised by topic.
3. [`program/wiki-domains/vertical-intelligence/wiki/dialogue-prediction-is-not-the-whole-story.md`](program/wiki-domains/vertical-intelligence/wiki/dialogue-prediction-is-not-the-whole-story.md):
   a long argument about whether prediction is all there is to intelligence (randomness,
   imagination, boredom, wholeness).
4. [`program/wiki-domains/vertical-intelligence/wiki/experiment-plan.md`](program/wiki-domains/vertical-intelligence/wiki/experiment-plan.md):
   every planned experiment and what would count as success.
5. [`neuroblock-research/emergent/README.md`](neuroblock-research/emergent/README.md): the
   "evolution without calculus" experiments in one page.

If you want to see the raw work, each line's `LOG.md` records every experiment in order, failures included.

## How it is organized

| Folder | What is inside |
|---|---|
| `program/` | The research notebook: charter, plan, findings, literature, and the author's own statements in `raw/` (word for word) |
| `curious-net/` | The main experimental code: curiosity, teaching, self-doubt, discovering laws from data, and `vi/` (the newest experiments, X0.x) |
| `neuroblock-research/` | The earlier line: a brain-like simulator with parts, sleep and mastery, plus the evolution experiments in `emergent/` |
| `tools/` | `sync.py`, which refreshes this repository from the working copies |

## Words used in the notes

| Term | Meaning |
|---|---|
| U1, U2, ... | The author's founding statements ("givens"), taken as starting points |
| Q1, Q2, ... | The program's open research questions |
| P-draft, P-number | A proposed test; it becomes a numbered prediction once its pass/fail rule is written down *before* it runs |
| C1, C2, ... / X0.1, X6.6 ... | Ids of finished results / of experiments in the current plan |
| Toy world | A small simulated environment built to test one idea cleanly |
| Seed | One independent random run; results are reported across many seeds |
| Gradient descent, Adam | The standard way modern AI is trained (Adam is its most common version) |
| Pre-registered | The success rule was fixed before the experiment ran, so it cannot be bent afterwards |

## Running the code

Python 3.11 with numpy, scipy, torch and matplotlib. Run scripts from their folder; the top of
each script says how (for example `python emergent/confirm.py` inside `neuroblock-research/`
reproduces the main evolution result in a couple of minutes). Raw third-party data is not
included; sources are listed in `curious-net/paper/LITERATURE.md`. AI agents working on the
repository should start at [`AGENTS.md`](AGENTS.md).

## Credits

The research direction, the founding statements and every redirect are the author's. Most of
the code, experiment write-ups and logs were produced by AI coding assistants (Anthropic's
Claude) working under the author's direction. Details and data sources are in
[`CREDITS.md`](CREDITS.md). [`SNAPSHOT.md`](SNAPSHOT.md) records when this copy was last refreshed.
