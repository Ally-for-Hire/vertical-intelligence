"""Exp H: does sleep matter, and does the self-chosen schedule beat fixed ones? Arms: never / self (flat progress + history doubled) / sched at 3 periods.
All arms: same env seeds, curiosity awake, score never in learning. Compute is reported separately for awake and sleep."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

CK = (1000, 2000, 4000, 8000, 16000)
ARMS = [("never", None), ("self", None), ("sched", 250), ("sched", 1000), ("sched", 4000)]


def job(a):
    import agent2, constants
    arm, period, seed, n_d = a
    if period:
        constants.REG["sleep_scheduled_period"]["value"] = period
    env, enc, net, rk, table, res, _ = agent2.train(seed, n_d, 16000, "hard", arm, ck=CK)
    res["expert"] = agent2.expert_drive(env, seed=seed)
    res["arm"], res["period"], res["seed"], res["n_d"] = arm, period, seed, n_d
    res["log"]["hist_keys"] = None
    return res


if __name__ == "__main__":
    jobs = [(arm, per, s, 8) for (arm, per) in ARMS for s in range(8)]
    t0 = time.time(); out = []
    with Pool(20) as p:
        for r in p.imap_unordered(job, jobs):
            out.append(r)
            print(len(out), len(jobs), r["arm"], r["period"], r["seed"], "sleeps", r["n_sleeps"], "edges", r["log"]["edges"][-1], "evals", {k: (None if v is None else round(v, 2)) for k, v in r["evals"].items()}, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open("../results/expH.json", "w"), default=float)
