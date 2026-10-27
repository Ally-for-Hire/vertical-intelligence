# Neuron-block research

Continuing a brain-like architecture research project handed off from another chat. **Read `README_HANDOFF.md` first**
(vision, design principles, results, caveats, "Next steps"), then `FOUNDATIONS.md` (the user's definition of what the brain IS: parts, the Self as the main focus, uncertainty and energy; user statements are givens, proposals are marked), then `FINDINGS.md` (the 2026-10-18 one-variable-at-a-time sweep:
what lined up with the principles and what did not).

## Start of every session
1. Read `lab/bookmarks/BOOKMARKS.md`: the user's bookmarked results and the notes they want to discuss. Offer to go through them.
2. The live dashboard (http://127.0.0.1:8765) and the Lab (http://127.0.0.1:8765/lab.html) should be running; log notable steps with
   `~/.claude/tools/live-dashboard/live-log.ps1` so the user can watch. If the server is down, start the scheduled task
   "Claude Live Dashboard".

## How to work here
- **Compute limit (user's instruction, 2026-10-18): 1 thread, low priority.** The Lab enforces it (`lab/limits.json`). Do not raise it or run heavy multi-process work unless the user says so; the user plays games on this PC.
- Everything is a measured experiment: answer questions by running them, not by guessing. Use 8+ seeds (24 for sweeps), report spread,
  and treat differences under ~2 accuracy points as noise. "Real effect" = paired z > 3.
- The user is a college junior who knows the basics: explain results in plain language (see the dashboard glossary in
  `~\claude-live\viz.js`; add entries there for any new setting or metric).
- Pure NumPy (Python 3.11, numpy 2.4). Run scripts from this folder with plain `python <script>.py` (NOT `python -I`).
- `experiments2.py` overwrites `results2.json`. `sweep.py` writes `sweeps/` and streams to the dashboard; `analyze.py` makes graphs and
  tables; `write_findings.py` publishes the findings.
- Prefer the Lab for new experiments (`lab/README.md`): it queues jobs, uses all cores, and keeps results the user can bookmark.
  New brain designs and tasks go in `lab/plugins/` (see `lab/plugins/README.md`); verify with a Compare job.
- Results are from a toy Gaussian-cluster task. They say nothing about real brains; keep claims proportionate.
- Known open questions (from the sweep): the matching radius `rho` is fixed by hand and dominates everything (a self-adjusting radius is
  a candidate for principle 8); `expand` and `rigid` (hardening) did not help; the fitness score F undervalues accuracy; accuracy is
  frequency-weighted (use a balanced accuracy before drawing conclusions about rare classes).

## probnet (2026-10-24)
A probability net (SNARC-style operant conditioning with NeuroBlock cells) lives in `probnet/`. Read `probnet/README.md` first, then `probnet/LOG.md` (all attempts, failures included) and `probnet/CONSTANTS.md`.
It was run on all cores by the user's permission for that work; the 1-thread rule above still applies to the Lab and other work unless the user says otherwise.
