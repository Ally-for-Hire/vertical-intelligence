# AGENTS.md: start here (for LLM agents)

The vertical-intelligence (VI) research program in one repo. Every experiment is a VI experiment, filed
by the family of VI concepts it tests; there are no sub-projects. This file is the only entry point; it holds
pointers, not results. Current state always lives in the files it points to.

## Layout

| Folder | What is in it |
|---|---|
| `program/` | the charter, plan and wiki pages (copied from the author's wiki; see "Editing") |
| `paper/` | prediction register (`PREDICTIONS.md`), run queue, ledger, write-ups, feasibility audit |
| `core/` | shared code: constants registry, worlds, learners, run control (`ctl.py`), `vipaths.py` (import path), `where.py` (find a script) |
| `experiments/<family>/` | code per family: `value-and-self`, `boredom-and-prediction`, `self-and-doubt`, `learning-to-learn`, `deep-discovery`; each has a `README.md`. Older families that keep their own internal layout: `self-gate-and-calibration`, `emergent-optimizers`, `energy-brain-simulator` |
| `results/` | run outputs of the vi experiments (one shared folder: experiments read each other's files); `results/figures/` |
| `worlds/` | saved worlds that replay bit-exactly |
| `tools/`, `portal/`, `contrib/` | publishing tools, contributor portal, queued proposals |

Older text (logs, the register, wiki pages) says `vi/x1_47.py`, `curious-net/...` or `neuroblock-research/...`: those are the
pre-2026-11-07 locations. `python core/where.py x1_47` gives the current path; folder names then were storage, not projects.

## Read order (stop when you have what you need)

1. `program/wiki-domains/vertical-intelligence/wiki/research-protocol.md`: the charter. Givens
   U1-U18 (the author's theorems, authoritative), questions Q1-Q10, rules, autonomy grant and stop lines.
2. `program/wiki-domains/vertical-intelligence/wiki/inbox.md`: the **"Next up"** line (what runs
   next), open theorems, tests (P-drafts) and ideas.
3. `paper/QUEUE.md`: the ordered run queue (head = next experiment).
4. Only for the area you touch: `program/wiki-domains/vertical-intelligence/wiki/experiment-plan.md`
   (every planned test with arms and thresholds), `findings-map.md` (results by section), and the
   line's `LOG.md` (newest work).

## Where each kind of record lives

| Record | Path |
|---|---|
| Author's words, verbatim (append-only); personal reflections are summarized, verbatim kept private | `program/wiki-domains/*/raw/` |
| Charter, givens, open conflicts | `program/wiki-domains/vertical-intelligence/wiki/research-protocol.md` |
| Intake: theorems, P-drafts, ideas, questions | `.../vertical-intelligence/wiki/inbox.md` |
| Every experiment by concept, with feasibility verdicts (2026-11-07 audit) | `.../vertical-intelligence/wiki/experiment-registry.md`; details `paper/FEASIBILITY_AUDIT_2026-11-07.md` |
| Planned experiments (arms, metrics, thresholds) | `.../vertical-intelligence/wiki/experiment-plan.md` |
| Results by topic | `.../vertical-intelligence/wiki/findings-map.md`; `program/wiki-domains/vertical-intelligence/wiki/synthesis.md` |
| Run queue | `paper/QUEUE.md` |
| Pre-registered predictions (P-numbers) and verdicts | `paper/PREDICTIONS.md` |
| Per-iteration narrative, newest first | `LOG.md`; `experiments/emergent-optimizers/emergent/LOG.md` and each series folder's `LOG.md` |
| Hand-off for the next session | `HANDOFF.md` |
| Current experiment code (plan X-ids) | `experiments/<family>/` (`x0_7.py` = experiment X0.7; `python core/where.py x0_7`); constants registry `core/constants.py` |
| Self-gate, oracle-teacher, calibration family (older) | `experiments/self-gate-and-calibration/` (`experiments/`, `src/`, `results/`) |
| Emergent optimizer (blind variation + culling) | `experiments/emergent-optimizers/emergent/` (`README.md` first) |
| Energy-limited brain-simulator family (older) | `experiments/energy-brain-simulator/` (`README_HANDOFF.md`, `FOUNDATIONS.md`, `FINDINGS.md`) |
| What the public copy was built from | `SNAPSHOT.md` (public repo only) |

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
- No hand-tuned constants: derive a value or measure it with a recorded sweep (`CONSTANTS.md`, `core/constants.py`).
- Compute (author, 2026-10-27): machine-usage limits are lifted, but keep a 15% margin because the author's PC
  crashed twice during GPU runs (cause unknown): total CPU <= 85%, free RAM >= 15% of total, GPU memory <= 85%.
  `core/budget.py` sizes worker pools from the live load. Toy-world defaults are in the protocol, section 3.
- Report outcome first, then evidence, in plain language. Toy-world results are principles in
  miniature, not facts about brains or LLMs.
- Leave it easier than you found it: update the surrounding docs a change affects (this file,
  READMEs, indexes, queue, logs), and fix stale paths, wrong commands or misleading pointers at
  their source when you find them.
- Publishing: on the author's machine the public repo is updated only through `python tools/publish.py`
  (a dry run), then `--push "message"` (a standing grant from the author, 2026-10-27; see
  `program/wiki-domains/vertical-intelligence/wiki/publishing.md`). Anywhere else, or for anything
  other than that, commit and push only when asked.

## Editing and publishing

This repo is the canonical working copy of the code, results and register (private, unredacted). The wiki pages
under `program/` are copies of the author's Knowledge-Wiki domain `vertical-intelligence` (edit the wiki, not the copy). The
**public** repo (`Ally-for-Hire/vertical-intelligence`) is a redacted snapshot of this one, produced by `python tools/publish.py`
(a dry run), then `--push "message"` (standing grant from the author, 2026-10-27; see
`program/wiki-domains/vertical-intelligence/wiki/publishing.md`). The sync replaces private terms and local paths using
`tools/redact.local.json` (gitignored, on the author's machine only) and refuses to run without it. Never commit the
redaction file, credentials or session details. Anywhere else, or for anything other than that, commit and push only when asked.

## Running code

Python 3.11: numpy, scipy, torch, matplotlib; the brain-simulator family is pure NumPy except `selfscore/` (torch, CUDA if present).
Run a script from its own folder; each script's docstring gives its command (for example
`cd experiments/value-and-self && python x0_7.py tune|run`; run control `python core/ctl.py status x1_47`;
emergent optimizers `cd experiments/emergent-optimizers/emergent && python confirm.py`). Raw third-party data
(`experiments/self-gate-and-calibration/data/`) is not included in the public copy; sources are in `paper/LITERATURE.md`.
