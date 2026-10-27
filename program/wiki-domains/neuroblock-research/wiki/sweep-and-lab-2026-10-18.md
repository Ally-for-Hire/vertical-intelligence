---
title: Neuron-block sweep findings and the Lab (2026-10-18)
domain: neuroblock-research
kind: audit
tags: [simulation, sweep, lab, dashboard, plugins, bookmarks]
updated: 2026-10-18
---

# Neuron-block sweep findings and the Lab (2026-10-18)

## Summary

A one-variable-at-a-time sweep (28 settings; 4 task conditions with all features on plus 2 bare-brain conditions; 24 seeds;
9,744 simulated lifetimes, 16 minutes on 16 threads) and a Lab for running user-designed experiments. Full write-up:
`~/neuroblock-research/FINDINGS.md`; raw data `sweeps/raw_runs.csv`; tables `sweeps/analysis_tables.md`.
"Real effect" = paired z > 3 (paired by seed against a reference setting).

## Findings that change what to do next

- **Matching radius rho dominates everything** (acc 65% at 0.2, 99.3% at 0.5, 64% at 1.0; neurons 700 -> 6). Inputs D and noise sigma
  next. With fixed rho, D = 32 or sigma = 0.3 collapse the brain (15% / 40% accuracy, ~1,000 neurons): needs a self-adjusting radius.
- **Errors are the wrong neuron answering, not missing neurons**: coverage ~99% yet accuracy 79-88% on hard tasks. `select='best'`
  gives +8.8 (hard/shift: 88.4 -> 97.2) to +15.5 points (hard/replace: 79.4 -> 94.9) for ~7-10 us.
- **Movement 3x faster for free** (94 -> 31 us, no accuracy change). **Prune halves neurons** (25 -> 11). Merge: borderline.
- **Speed floor is geometry**: `min_sep` 0 / 0.3 / 0.8 -> 5 / 31 / 84 us with identical accuracy and size. `us_per_unit` scales time exactly linearly.
- **Expansion drive inert** in all four tasks, even with headroom (hard/replace 79.4% either way); +10 neurons on a bare brain.
- **Hardening not helpful**: +10-15 us, no accuracy gain, run-to-run variance NOT halved (handoff claim not reproduced); c0 = 3 bloats (25 vs 11 neurons) - that claim reproduced.
- Measurement cautions: accuracy is frequency-weighted (zipf 2 inflates it); the original fitness F rates best-fit selection no better than fastest.

## Gotchas

- `python -I` breaks the scripts (sibling imports). `experiments2.py` overwrites `results2.json`.
- The lab engine `lab/lab_core.simulate` reproduces `sweep.run_ext` and the original `neuroblock.run` bit for bit at the defaults (self-test).
- Windows: child processes inherit the parent's priority class; `SetPriorityClass` via ctypes needs declared argtypes/restype or it silently fails
  (verified with `Win32_Process.Priority`: 10 = above normal, 8 = normal, 6 = below normal).
- PowerShell 5.1 `Set-Content -Encoding utf8` adds a BOM and `Start-Process -ArgumentList` does not quote array elements; write scripts to files.
- Local tool `sweep.py` summaries show `z` capped at 99; paired z needs >= 5 seeds.

## The Lab (http://127.0.0.1:8765/lab.html)

Server `~/.claude/tools/live-dashboard/serve.py` (scheduled task "Claude Live Dashboard", logon). Experiment types: sweep, grid, compare
(incl. two brain designs), watch a brain. CPU threads up to all cores, priority, keep-awake; verified 86-98% CPU at 24 threads, cancel in
~3 s. Bookmarks + notes stored under `neuroblock-research/lab/bookmarks` with a readable `BOOKMARKS.md`. Plugins in `lab/plugins/`
(example `CautiousBrain`). Security: 127.0.0.1 only, Host/Origin/X-Lab checks, jobs are validated data only (no code from the browser).
See `lab/README.md`.

## Open

Interactions between settings are untested (the Lab grid is built for that). Principles 6-8 (sensor trust, "negative for the whole"
spawn check, uncertainty-driven effort) are not implemented. The example plugin's first 3-seed look (patience 4 halves neurons at similar
accuracy) is untested noise until a real Compare job says otherwise.
