---
title: Vertical Intelligence - Work on this (roles, first tasks, how to start)
domain: vertical-intelligence
kind: guide
tags: [recruiting, collaborators, onboarding, roles, first-tasks]
updated: 2026-10-30
---

# Work on this

Read this if: you are thinking about joining, or you are onboarding someone. The program needs a
small team; each role below has a first task that is finishable in about two weeks, has a written
success rule, and lands in the public register under your name ([[credits]]).

Five-line summary: six roles; every first task is a bounded, pre-specified piece of work (most come
from the 2026-10-28 audit's redo list, which is a set of well-defined experiments with known
baselines). Day one: reproduce one result with one command. You need Python; you do not need to have
published. The rules you work under are short and the same for everyone: "Defend every result" and
"Earn every component" ([[research-protocol]] section 5).

## What you would be joining

A pre-registered experimental program on what gradient descent alone lacks (value, drive, parts, a
Self, own experiments, skepticism), run in small simulated worlds with fair baselines and a public
audit trail, heading toward a model that finds laws and principles in data. The two-page summary is
[[program-brief]]; the map of everything found is [[findings-map]].

## Roles and first tasks

| Role | You bring | First task (two weeks) | Success rule |
|---|---|---|---|
| **Experiment engineer** | Python, numpy; torch helps | Redo item R-V2 from the audit: motivation as a learning booster (C52) with the controls it lacked (equal-compute baseline, tuned persistence), 48 worlds | registered predictions, verdicts recorded either way |
| **Adversarial reviewer** | statistics, a sceptical temperament | Write the attack list and fairness check for one recorded result (start with X1.4 or BENCH2); run the one control the author did not | a CORRECTION or a confirmed result in the register |
| **Theory and literature** | cognitive science, neuroscience, or RL theory | Fill the gaps in [[literature-map]] for sections 0 and 1 (homeostatic RL, learning-progress curiosity, interoception): paper, claim, what it changes for us | each entry one line, `[u]` if not read in full |
| **ML systems** | PyTorch internals, hooks, a GPU | Self line rung 2 (inbox I18): sensors are real small CNNs trained on corrupted subsets of a standard image set; the Self reads their complete internal state | reproduces the toy Self result's structure, or shows where it breaks |
| **Science data** | physics or data wrangling | Find, verify and document the 1880s Balmer/Rydberg line measurements and the blackbody spectra for the final exam, with provenance | a data note with sources, units, uncertainties, and a loader |
| **Writing and figures** | clear prose, matplotlib | The X1.3/X1.4 figure for the README to the repo's figure standard (title states the finding, one accent hue, light and dark, alt text, data table) | merged into the README |

If none fits, say what you can do; the audit redo list ([[audit-2026-10-28]] section "Redo order") has
more bounded tasks.

## Day one

1. Clone `github.com/Ally-for-Hire/vertical-intelligence`. Python 3.11, numpy, scipy, torch, matplotlib.
2. Reproduce one result with one command (for example the evolution benchmark or an X0 world; each
   script's header says how). Compare your numbers to the register.
3. Read [[research-protocol]] sections 3 to 5 (intake, autonomy, definition of done) and the
   "Defend every result" rule. They are short.
4. Pick a first task above; write its prediction before running anything; the author or a reviewer
   registers it with you.

## How we work

- Predictions before runs, in git. Failures are kept. No metric chasing: a stalled number means change
  the question, not the knobs.
- Every result carries a fairness proof, an attack list and an own-goal check. Expect your work to be
  attacked, politely, and attack ours.
- Toy results are "principles in miniature", never facts about brains or large models.
- Credit is explicit: every registered item names who designed, ran and reviewed it.
- Time: most first tasks are evenings-and-weekends sized; the compute is a laptop CPU unless the task
  says GPU.

## Contact

Through the author (see the repository README). Do not send data or drafts that are not yours to share.
