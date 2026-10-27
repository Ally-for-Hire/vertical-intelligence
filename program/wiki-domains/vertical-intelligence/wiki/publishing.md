---
title: Publishing the public repo (standing grant, procedure, quality bar, the private archive)
domain: vertical-intelligence
kind: workflow
tags: [publishing, public-repo, sponsors, sync, readme, figures]
updated: 2026-10-27
---

# Publishing the public repo

Read this if: you have recorded a result and must update the public repo, you are changing its front
page, or you wonder what the older private repo is for.

Five-line summary: the public repo is the sponsor-facing face of the program and is kept current under
a standing grant from the author (2026-10-27). Edit the canonical repos and this wiki first, then run
`python tools/publish.py` in the public clone: it syncs, scans, and (with `--push`) pushes only to the
public remote. The front page (`README.md`) carries a "Latest update" block and figures that are
refreshed at every publish. The older private repo is a redundant unredacted archive and the only
offsite copy of the code and results; it is not retired.

## 1. Standing grant (user, 2026-10-27)

"Keep the public repo actively updated with progress. This is the public-facing section of our work,
and needs to be maintained, shiny, and beautiful for potential sponsors."

Allowed without asking: sync, scan, commit and push to the public remote's `main` via
`tools/publish.py`. Still ask first: force pushes or history rewrites; changing a repo's visibility;
creating or deleting repos or branches; any push to the private archive; adding personal or
third-party material to the public tree. The charter ([[research-protocol]] section 4) records the same.

## 2. The two repos

| Repo | Visibility | Role | Local |
|---|---|---|---|
| `vertical-intelligence` | **public** | The program's public face: charter, plan, findings, code, figures. Redacted snapshot | the public clone's branch `public`, remote `public`, branch `main` |
| `vertical-learning` | private | Unredacted archive from the first snapshot (2026-10-26) | the same clone's branch `main`, remote `origin`; never push the `public` branch there and never push `main` to the public remote |

The canonical sources are the working repos (curious-net, neuroblock-research) and this wiki's
vertical-intelligence, curious-net and neuroblock-research domains. Everything in the public repo except
four root documents (README.md, AGENTS.md, CREDITS.md, CLAUDE.md) and `tools/` is copied by
`tools/sync.py`, redacted, and never edited by hand.

## 3. Procedure (each publish)

1. **Record first** in the canonical places ([[research-protocol]] section 2): LOG, PREDICTIONS, QUEUE,
   [[findings-map]], [[experiment-plan]], [[inbox]], wiki log. Publish after a result is recorded, never
   mid-run (partial run files would be copied).
2. **Figures.** `python curious-net/vi/figures.py` (builds from the result files). Open every PNG, light
   and dark, and look at it: clipped text, label collisions, hairline artifacts.
3. **README "Latest update" block** (public clone): the at-a-glance line from
   `python curious-net/vi/register_counts.py`, one figure per new milestone, what finished, what is
   running, what is next. A claim that a new result supersedes is corrected in the README in the same
   publish, and the commit message says so.
4. **Dry run:** `python tools/publish.py`. It refuses unless the branch, the remote and its PUBLIC
   visibility check out, and it scans the new or changed files for private terms (from the gitignored
   `tools/redact.local.json`, never printed), secret-looking values, e-mail addresses, local paths and
   oversize files. Fix findings at the source (edit the canonical file or add a redaction rule).
5. **Push:** `python tools/publish.py --push "one-line summary"`. It never forces; a rejected push stops.
6. **Verify:** `gh api repos/Ally-for-Hire/vertical-intelligence/commits/main` shows the new commit.
   Add a wiki log line.

## 4. Quality bar for the front page

- Lead with the question and the stage ("early and exploratory", toy worlds), never with a claim about
  real brains or large AI systems.
- **Failures are part of the product.** The at-a-glance line states passed, failed and awaiting counts.
  Corrections are acknowledged where they happen (for example the repaired evolution benchmark).
- Every figure states its finding in the title, uses one accent hue plus grays (emphasis), thin marks,
  direct labels, light and dark variants, alt text and a collapsible data table; the numbers come from
  the result files, never typed. Palette and checks follow the dataviz method.
- No private details, no overclaiming, no unexplained jargon: a term used on the page is defined in the
  glossary table.
- The page must read well to a sponsor who stops after one screen, and still point an agent to
  `AGENTS.md`.

## 5. What never goes public

Credentials, account tokens and e-mail addresses; family names and private terms (the redaction list);
local paths; third-party drafts (the `private` list in `redact.local.json`); raw third-party data.
Personal statements of the author are the author's decision: see the log for any open question.

## 6. The private archive (`vertical-learning`)

What it holds: the first unredacted snapshot (2026-10-26) plus later refreshes of the same files. A
file-by-file comparison on 2026-10-27 found **no file in it that is not also in a canonical source**,
apart from its six root files (README, AGENTS, CREDITS, CLAUDE, SNAPSHOT, `tools/sync.py`), which now
live in the public repo. What was folded into this wiki: the credits ([[credits]]), the publishing
procedure (this page), the findings summary ([[findings-map]], curious-net `synthesis`).

**Not retired, on purpose.** curious-net and neuroblock-research have no git remote, so this private
repo is currently the only offsite copy of their code and raw results. Its `main` is out of date by
every commit since 2026-10-27. Before it could be archived or deleted, those two repos would need
their own private remotes (a decision for the author). Do not delete, archive or rewrite it without
asking.

## 7. Cadence

Publish after each experiment that is recorded (verdicts in the register), after each correction, and
whenever the front page is out of date. Several experiments finished in one session go in one publish.
