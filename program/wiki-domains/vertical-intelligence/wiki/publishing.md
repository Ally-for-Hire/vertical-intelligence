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

## 2. The repos (reorganized 2026-11-07)

| Repo | Visibility | Role | Local |
|---|---|---|---|
| `vertical-intelligence-private` | private | **Canonical, unredacted**: all code (experiments by family under `experiments/`, `core/`), results, register (`paper/`), and a copy of this wiki domain in `program/`. Full git history of the former curious-net and neuroblock-research repos | `~/vertical-intelligence` (branch `main`, remote `origin`) |
| `vertical-intelligence` | **public** | The program's public face. Redacted snapshot of the repo above | `~/vertical-intelligence-public` (branch `public`, remote `public`, branch `main`) |
| `curious-net`, `neuroblock-research`, `vertical-learning` | private | Read-only archives of the former separate repos (kept, not retired) | their old local folders |

Edit the canonical repo (`vertical-intelligence`) and this wiki first; refresh its `program/` copy with `python tools/sync.py --program`. Everything in the public repo is copied by `tools/sync.py` from the canonical repo and this domain, redacted, and never edited by hand; `README.md`, `AGENTS.md`, `CREDITS.md`, `CLAUDE.md` are written in the canonical repo and copied with the rest. Never push the public clone's `main` branch (it tracks the old archive) and never push `public` to a private remote.

## 3. Procedure (each publish)

1. **Record first** in the canonical places ([[research-protocol]] section 2): LOG, PREDICTIONS, QUEUE,
   [[findings-map]], [[experiment-plan]], [[inbox]], wiki log. Publish after a result is recorded, never
   mid-run (partial run files would be copied).
2. **Figures.** `python core/figures.py` (builds from the result files). Open every PNG, light
   and dark, and look at it: clipped text, label collisions, hairline artifacts.
3. **README "Latest update" block** (public clone): the at-a-glance line from
   `python core/register_counts.py`, one figure per new milestone, what finished, what is
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
Personal statements of the author are summarized for the public, not quoted (author's decision,
2026-10-27): the verbatim transcript of Dialogue 1 stays in the private wiki and is on the redaction
`private` list; its public stand-in is `raw/2026-10-26-dialogue-summary.md`, which describes what led to
each claim only at the level needed. New raw statements that contain personal reflections get the same
treatment (verbatim in the private wiki, summary in the public tree). Older commits of the public repo
still contain the earlier verbatim file; clearing them needs a history rewrite and a force-push, which
is done only on the author's explicit request.

## 6. The archives

Before 2026-11-07 there were three private repos: `curious-net` and `neuroblock-research` (created 2026-10-27 at the author's request, branch `master`) and `vertical-learning` (the first unredacted snapshot, 2026-10-26, branch `main`; it held no file that was not in a canonical source apart from its root files). They were merged into `vertical-intelligence-private` with history kept (both histories imported as subtree merges, then every file `git mv`'d into its family folder, so `git log --follow` works across the move). The three old repos are left as they were and are not to be deleted, archived or rewritten without asking. Uncommitted working files are not backed up anywhere: commit work you want kept.

## 7. Cadence

Publish after each experiment that is recorded (verdicts in the register), after each correction, and
whenever the front page is out of date. Several experiments finished in one session go in one publish.
