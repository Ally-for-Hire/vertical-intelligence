"""Information watcher (author 2026-11-03: stop the tests if we ever stop learning something new, and record it).
Every SNAP seconds (30; compared with the last snapshot that carried new information) snapshot the live heartbeats (results/progress/*.json) + finished worlds. Per arm: mean running R^2 over active workers, plus the
arm ranking. A snapshot is NEW INFORMATION if any of: a world finished, the arm ranking changed, any arm's mean R^2 moved by >= EPS, a worker reached a new arm.
STALE (emit, then the controller stops the run and logs it as a failed route) after PATIENCE snapshots (10 x 30 s = 5 min) in a row with none of those.
EPS, SNAP, PATIENCE are ARBITRARY (declared, not backed). History is appended to results/info_history.jsonl."""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import os, sys, json, time, glob
R = os.path.join(str(vipaths.RESULTS)); SNAP, EPS, PATIENCE = 30, 0.002, 10   # CONST: info_watch (ARBITRARY)
prev, quiet, t_start, last_print = None, 0, time.time(), 0.0
while True:
    now = time.time(); rows = []
    for f in glob.glob(os.path.join(R, "progress", "*.json")):
        try: j = json.load(open(f))
        except (OSError, ValueError): continue
        if now - j["updated"] < 900: rows.append(j)
    done = len(glob.glob(os.path.join(R, "x1_29_worlds", "*.json"))) + len(glob.glob(os.path.join(R, "x1_34_cells", "*.json"))) + len(glob.glob(os.path.join(R, "x1_35_cells", "*.json"))) + len(glob.glob(os.path.join(R, "x1_36_cells", "*.json"))) + len(glob.glob(os.path.join(R, "x1_33_cells", "*.json")))
    for j in rows: j["arm"] = j.get("exp", "?") + ":" + j["arm"]
    arms = sorted({j["arm"] for j in rows}); mean = {a: sum(j["r2"] for j in rows if j["arm"] == a) / sum(1 for j in rows if j["arm"] == a) for a in arms}
    rank = sorted(mean, key=lambda a: -mean[a]); snap = dict(time=now, done=done, workers=len(rows), mean=mean, rank=rank, arms=arms)
    open(os.path.join(R, "info_history.jsonl"), "a").write(json.dumps(snap) + "\n")
    if prev is None or not rows: new = "start" if prev is None else None
    else:
        new = ("world finished" if done > prev["done"] else "ranking changed" if rank != prev["rank"] and set(arms) == set(prev["arms"]) else
               "new arm reached" if set(arms) - set(prev["arms"]) else
               next((f"{a} moved {mean[a] - prev['mean'][a]:+.4f}" for a in arms if a in prev["mean"] and abs(mean[a] - prev["mean"][a]) >= EPS), None))
    if new: quiet = 0
    if new and (now - last_print >= 600 or new in ('start',)): last_print = now; print(f"NEW INFO ({new}) | worlds done {done}, {len(rows)} workers, best {rank[0] if rank else '-'} {mean.get(rank[0], 0):.3f}" if rank else f"NEW INFO ({new})", flush=True)
    elif rows or (prev and prev["workers"]):
        quiet += 1
        if quiet >= PATIENCE: print(f"STALE no new information for {quiet * SNAP // 60} min; recommend stop + record as failed route", flush=True); quiet = 0
    if new or prev is None: prev = snap      # baseline moves only when something new appeared, so slow drift still counts
    time.sleep(SNAP)
