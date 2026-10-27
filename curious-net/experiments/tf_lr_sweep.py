"""validation sweep for the transformer's online learning rate (separate worlds 1000..1007), plain and oracle-masked attention"""
import os, sys
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool


def job(a):
    import torch
    torch.set_num_threads(1)
    from worlds import sample_world, rollout
    from tfm import run_scenario_tf
    seed, lr, mode = a
    w = sample_world(seed)
    X, Y = rollout(w, 1500, np.random.default_rng(seed + 7))
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    r = run_scenario_tf(X, Y, cp, w.n_act, 1500, 100, seed, mode, truth_mask=w.truth()[cp], lr=lr)
    return (lr, mode, float(r["err"][:300].mean()), float(r["err"][:1000].mean()), float(r["err"][-300:].mean()))


if __name__ == "__main__":
    jobs = [(1000 + s, lr, m) for s in range(8) for lr in (1e-4, 3e-4, 1e-3, 3e-3) for m in ("plain", "oracle")]
    with Pool(20) as p:
        R = p.map(job, jobs)
    for m in ("plain", "oracle"):
        for lr in (1e-4, 3e-4, 1e-3, 3e-3):
            v = np.array([r[2:] for r in R if r[0] == lr and r[1] == m])
            print(m, lr, "first300 %.3f first1000 %.3f last300 %.3f" % tuple(v.mean(0)))
