# HANDOFF (written 2026-11-07, for the compaction / next session)

**Layout changed 2026-11-07 (later the same day):** curious-net and neuroblock-research were merged into this single repo (`vertical-intelligence`, history kept), organized by experiment family. Old `vi/xN_M.py` paths in the text below and in older logs: use `python core/where.py xN_M`. The old repos remain as read-only archives. See `AGENTS.md` for the layout.

Read this first, then `LOG.md` (newest first), `paper/PREDICTIONS.md`, and the wiki domain `vertical-intelligence`
(`~/Documents/Knowledge-Wiki/domains/vertical-intelligence/`: `wiki/research-protocol.md`, `wiki/findings-map.md`, `wiki/inbox.md`).
Routing rule from the user's global instructions: the FIRST tool call in a research session is reading `Knowledge-Wiki/AGENTS.md`.

## -1. Added 2026-11-07 (same day, after the section below)
- Author ruling: there are no sub-projects. Everything is a vertical-intelligence experiment in some family; folder names are storage only. One list by family/concept: wiki `vertical-intelligence/wiki/experiment-registry.md`.
- Feasibility audit of P199-P567: `paper/FEASIBILITY_AUDIT_2026-11-07.md` (+ `paper/feasibility/`). Row by row ~60% could really pass/fail; the goal-level tests (deep discovery from nothing, 'vertical' X1.10-X1.17, graded doubt X7.7) were mostly unanswerable by design. Before registering anything for the new I30 world, apply the four-question feasibility gate in that file (oracle clears the bar; no-skill arm does not; every arm sees the manipulated variable; n and bar give power) and do not hand the learner the generating family.

## 0. State right now
- NOTHING IS RUNNING (checked 2026-11-07: ctl.py x1_47 FINISHED, no experiment processes). All results through X1.47 are committed; the last curious-net commit is `8869ace` (X1.47). Only `results/info_history.jsonl` (churn) is modified and `results/x1_42_run.log` is an old untracked log.
- Disk: C: is 98% full (57 GB free). Do not launch big caches without checking `df`.
- Public repo (Ally-for-Hire/vertical-intelligence) update via `tools/publish.py` is still NOT authorised: do not push or publish; keep reporting it.
- No `.codegraph/` in the repo: use normal tools; indexing is opt-in.

## 1. The NEW direction (user, 2026-11-07) - this is what to do next
User: "I want to make a neural network that learns to identify patterns in functions before we give it this. I don't think this is the best problem. We should have a series of seemingly unrelated outputs that are linear in nature and easily predictable at a base level, but built of lots of parts that run underneath."
I restated it and asked one question; the user answered: **the SECOND reading**:
- Every output is (roughly) a LINEAR function of the inputs, so it is easy to predict at a base level (a plain linear fit already scores well);
- underneath, each output also contains SMALLER NONLINEAR PARTS drawn from a SHARED POOL of hidden parts: the same parts recur inside outputs that look unrelated on the surface;
- the network's job (stage 0, before any deep-composition world): identify those shared parts and which outputs use them ("patterns in functions").
Proposed tests (mine, not yet registered or built): (a) TRANSFER: a new output built from already-known parts should be learned from far fewer labels than from scratch; (b) RECOVERY: learned parts match the true hidden parts (corr^2, as in X1.39-46; exact = corr^2 >= 0.99); (c) the linear base must not be lost out of range (see X1.47 below).
First steps: write it into the wiki inbox as I30 (done in the same commit as this file), register predictions BEFORE any run (paper/PREDICTIONS.md), build a small world generator + one cheap test, minutes of compute not hours.
Open design decisions to settle by arms, not by asking: number of hidden parts, how many parts per output, size of the nonlinear part relative to the linear base, input dimension, how parts are shared, noise (default none).

## 2. Link to the user's earlier request I29 / X1.47 (done 2026-11-04, from a session not in this context)
Author: "each output is a linear function, even underlying it should be linear, and just see if the neural network can map the entire thing before increasing in complexity."
X1.47 (`experiments/deep-discovery/x1_47.py`, P563-P567): the X1.6 tanh learner does NOT pass the linear-base gate: it masters only 0.55 of tasks in range and 0.00 out of range (input sd 3); the same network WITHOUT tanh masters 1.00/1.00; least squares 1.00/1.00; 10 meta-learned steps (0.85 in range) beat 4000 Adam steps (0.55). Cause = the activation (tanh bends outside the training cloud), not data or optimizer.
Consequence for the new direction: the new network needs an EXACT LINEAR PATH (linear skip from inputs to outputs beside the nonlinear parts) and must pass the X1.47 gate (in AND out of range) BEFORE hidden parts are layered on top. This is also the logical rung 0 -> rung 1 of the X1.6 ladder.

## 3. What this session's long run found (X1.36-X1.46, all in LOG.md; the deep-composition world, `experiments/deep-discovery/wrec.py`)
World: one rule, concept n = sin(g * (concept n-1 + x[n mod 8] + thin ties)), g in {0.8, 1.2, 2.0}; room = 50 problems; exam = 40 fresh problems with frozen library; exact = corr^2 >= 0.99.
1. X1.36 shared tied cell + refit ladder: helps at g=0.8 (exam 0.73 -> 0.87-0.90), nothing at 1.2/2.0.
2. X1.37 labels 60-6000: no exact recovery; g=0.8 got worse at 2000.
3. X1.38 capacity 6/24/96: 24 helps at g=1.2 (0.48 -> 0.71); 96 hurts; refit 150 = 600.
4. X1.39 sine inside the cell: g=1.2, hid 24: exam 0.96, exact deep concepts in 2 of 3 worlds (FIRST exact recovery; sin names the rule's family).
5. X1.40 mixed-activation cell: 1 of 3 exact, exam 0.70. 6. X1.41 L1 on output weights: 0 of 3.
7. X1.42 depth (sine cell, rooms 1-4, ONE world, capped refit): same 2 exact levels, nothing new in deeper rooms, exam 0.97 -> 0.68; not decided (P549-P551).
8. X1.43 sine cell g=2.0 with 2000 labels: no. 9. X1.44/X1.44b slow optimizer settings g=2.0: 2 of 9 worlds near-full recovery (lottery). 10. X1.45 restarts: 1 of 12.
11. X1.46 pilot picks activation (tanh|sin|relu): sin in 6 of 6 worlds, 2 of 6 exact, median exam 0.81 (list contains the true family).
Conclusion kept honest: exact deep recovery appears only when the cell has (or is steered to) the rule's sine family; g=2.0 is a lottery; depth scaling is unproven. The user now thinks this is not the best problem (section 1).

## 4. Rules and preferences in force (also in memory MEMORY.md and the wiki protocol)
- Pre-register predictions before running; log EVERY look; no min-effect filters; "wins are wins"; try every route as an arm, never ask the author to pick; one idea + one question per turn in dialogues; earn every component; defend every result (sceptic checklist); mark arbitrary constants.
- Least compute for ALL the information; stop a run when 5 minutes pass with no new information (30 s snapshots); stop when the registered predictions are decided (>= 6 units, 99% interim range, log in results/decide_log.txt).
- Compute margin: CPU <= 85%, RAM free >= 15%, GPU <= 85%. RAM fell to 1%/4% twice this session (hid-96 cells and deep rooms): cap workers by memory, `budget.workers(per_worker_gb=...)`.
- Runs must be pingable/stoppable/restartable: `python core/ctl.py status|ping|stop|start|restart <name>` (default name is the last experiment; edit `names = ... or ["x1_.."]` in ctl.py when a new experiment starts). Heartbeats via `x1_28.hb`. Cells are cached in results/<exp>_cells/ so reruns resume.
- Final self-review loop over the whole chat before delivering a concrete deliverable; commit after changes (curious-net and Knowledge-Wiki); push only when asked.

## 5. Tooling gotchas that cost time (fix at source if you touch them)
- `experiments/deep-discovery/x1_39.py` is a multi-purpose runner: `python x1_39.py run <acts> <gains> <hids> <N>` plus env vars X139_LR / X139_PAT / X139_REFIT / X139_SEED_OFFSET / X139_SEEDS / X139_ONLY / X139_INITS. Result-file names depend on those (X1.44 once overwrote x1_40_run.json; fixed). Prefer a new script per experiment.
- `experiments/deep-discovery/cell.py` options: act (tanh|sin|relu|mix), l1, refit_rows (memory cap that CHANGES results in some worlds: 0.989 -> 0.668 in one world), init, n_lab, hid; `Learner.activate()` must be called before using a learner when several learners with different activations share a process (module-global ACT).
- Cells have no per-room checkpoint (X1.42 lost hours when stopped); the exam step prints no heartbeat (looks stalled for 10-20 minutes at large N: check the cell cache before killing).
- Monitors expire after 30 minutes; ScheduleWakeup is the fallback; the loop skill was stopped at the end of the last session (PushNotification sent).
- Windows/Git Bash: Bash heredocs with quotes break; use Write/Edit for scripts; `\\` paths in python need raw strings.

## 6. Open items (not started)
Handoff section 1 (the new world); gate the new network with the X1.47 linear test; candidate list without the true family (X1.46 follow-up); memory-light refit + per-room checkpoint to replicate depth; warm-up from smaller g or a different optimizer for g=2.0; held X1.32/X1.33 ladders; wiki findings-map/inbox should get the new-direction entry (I30) and X1.47 is already there.
