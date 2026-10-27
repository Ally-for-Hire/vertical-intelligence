# AGENTS.md: start here (for LLM agents)

The vertical-intelligence research program in one repo: the charter and plan (`program/`) and
two code lines (`curious-net/`, `neuroblock-research/`). This file is the only entry point; it
holds pointers, not results. Current state always lives in the files it points to.

## Read order (stop when you have what you need)

1. `program/wiki-domains/vertical-intelligence/wiki/research-protocol.md`: the charter. Givens
   U1-U18 (the author's theorems, authoritative), questions Q1-Q10, rules, autonomy grant and stop lines.
2. `program/wiki-domains/vertical-intelligence/wiki/inbox.md`: the **"Next up"** line (what runs
   next), open theorems, tests (P-drafts) and ideas.
3. `curious-net/paper/QUEUE.md`: the ordered run queue (head = next experiment).
4. Only for the area you touch: `program/wiki-domains/vertical-intelligence/wiki/experiment-plan.md`
   (every planned test with arms and thresholds), `findings-map.md` (results by section), and the
   line's `LOG.md` (newest work).

## Where each kind of record lives

| Record | Path |
|---|---|
| Author's words, verbatim (append-only) | `program/wiki-domains/*/raw/` |
| Charter, givens, open conflicts | `program/wiki-domains/vertical-intelligence/wiki/research-protocol.md` |
| Intake: theorems, P-drafts, ideas, questions | `.../vertical-intelligence/wiki/inbox.md` |
| Planned experiments (arms, metrics, thresholds) | `.../vertical-intelligence/wiki/experiment-plan.md` |
| Results by topic | `.../vertical-intelligence/wiki/findings-map.md`; `program/wiki-domains/curious-net/wiki/synthesis.md` |
| Run queue | `curious-net/paper/QUEUE.md` |
| Pre-registered predictions (P-numbers) and verdicts | `curious-net/paper/PREDICTIONS.md` |
| Per-iteration narrative, newest first | `curious-net/LOG.md`; `neuroblock-research/emergent/LOG.md` and each series folder's `LOG.md` |
| Current experiment code (plan X-ids) | `curious-net/vi/` (`x0_7.py` = experiment X0.7); constants registry `curious-net/vi/constants.py` |
| Older curious-net experiments | `curious-net/experiments/`, results `curious-net/results/` |
| Emergent optimizer (blind variation + culling) | `neuroblock-research/emergent/` (`README.md` first) |
| NeuroBlock brain simulator (older line) | `neuroblock-research/` (`README_HANDOFF.md`, `FOUNDATIONS.md`, `FINDINGS.md`) |
| What this snapshot was built from | `SNAPSHOT.md` |

**Wiki links.** Pages under `program/` were written for a wiki: `[[page]]` means
`program/wiki-domains/vertical-intelligence/wiki/page.md`, and `[[domains/<d>/wiki/page]]` means
`program/wiki-domains/<d>/wiki/page.md`. A link with no matching file points to a page that was not
copied, or not written yet.

## Rules every agent follows

- The author's statements (U-givens, `raw/` files) are givens; anything marked "(assistant)" is a
  proposal the author may overrule. Never change a given silently.
- Register before running: question, arms (with a plain baseline), metric, threshold, seeds and
  a manipulation check go in the plan or `PREDICTIONS.md` before any run; score as registered.
- Never delete a result. Failures, voids and corrections stay; a wrong number gets a CORRECTION
  entry, not an edit.
- No hand-tuned constants: derive a value or measure it with a recorded sweep (`curious-net/CONSTANTS.md`, `vi/constants.py`).
- Compute: one GPU job at a time (the author's PC crashed twice during GPU runs, cause unknown); ask before any job
  over 30 min of GPU. CPU toy-world defaults are in the protocol, section 3.
- Report outcome first, then evidence, in plain language. Toy-world results are principles in
  miniature, not facts about brains or LLMs.
- Leave it easier than you found it: update the surrounding docs a change affects (this file,
  READMEs, indexes, queue, logs), and fix stale paths, wrong commands or misleading pointers at
  their source when you find them.
- Commit only when asked; push only when asked.

## Editing: this repo is a snapshot

The canonical working copies are separate local repos on the author's machine (`curious-net`,
`neuroblock-research`, and the private Knowledge-Wiki domains). **Edit those**, then run
`python tools/sync.py` here to refresh this repo (it copies, never deletes, and rewrites
`SNAPSHOT.md`). **This repo is public:** the sync replaces private terms and local paths using
`tools/redact.local.json` (gitignored, on the author's machine only) and refuses to run without
it. Read `SNAPSHOT.md` before committing; never commit the redaction file, credentials, or
session details. If you only have this repo, work on a branch here and say so; the author merges it back.
Line-specific notes (`neuroblock-research/CLAUDE.md`) describe the author's local setup (dashboard,
thread limits); the machine-specific steps apply only on that machine.

## Running code

Python 3.11. curious-net: numpy, scipy, torch, matplotlib. neuroblock-research: pure NumPy, except
`emergent/` and `selfscore/` (torch, CUDA if present). Run scripts from their line's folder; each
script's docstring gives its command (for example `python vi/x0_7.py tune|run`,
`python emergent/confirm.py`). Raw third-party data (`curious-net/data/`) is not included; see
`curious-net/paper/LITERATURE.md` for sources.
