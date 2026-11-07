# Experiment queue

Each file here is one **proposal**: an experiment idea from a key-holder, written by the contributor
portal (`portal/README.md`) as `<date>-<slug>.md` and committed. A proposal is a request, not a
result. **Nothing here runs until the author registers it as a P-number in
`paper/PREDICTIONS.md`**; the portal never executes anything.

## Frontmatter
```
---
title: Short name of the idea
status: proposed
proposed_by: alice
date: 2026-10-30
section: 1
---
```
- `title`: one line. `proposed_by`: the key's name. `date`: when it was submitted.
- `section`: findings-map section `0`-`8`, or `method`.
- `status`: see below.

## Status vocabulary (changed only by the author, by editing the file)
| Status | Meaning |
|---|---|
| `proposed` | Submitted, not yet reviewed |
| `registered` | Accepted and given a P-number with its pass/fail rule written down |
| `running` | The experiment is in progress |
| `done` | Finished; the result is in the register and logs (pass or fail) |
| `declined` | Not taken up, with a reason in the file |

## P-draft shape (from the research protocol)
The body should have these headings:
1. `## Question and U-theorem`: one sentence, and the founding statement or parked idea it serves
2. `## Arms`: including a plain baseline with exact parity at zero experience, and a ceiling/oracle arm where one exists
3. `## Metric, threshold, direction, seeds`: written down before any run
4. `## Manipulation check`: evidence that the varied factor matters in this world
5. `## Budget`: expected wall time and hardware (toy-world default: CPU, <= 10 min per arm, >= 48 seeds)
6. `## Fairness notes`: equal tuning and compute across arms, held-out seeds, what a sceptic would attack
