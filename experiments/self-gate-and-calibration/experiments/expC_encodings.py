"""Exp C: Boolean vs soft vs qubit encoding. Same env/seeds/policy (curiosity), score never used in learning.
Measures: score of model-driven exploit policy at training checkpoints, action-link recall, false links, neuron/edge counts."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np
import common
from multiprocessing import Pool

CK = (500, 1000, 2000, 4000, 8000, 16000)
CONFIGS = [("hard", 0.3, ("sample", "meanfield")), ("soft", 0.1, ("meanfield",)), ("soft", 0.3, ("meanfield",)),
           ("soft", 1.0, ("meanfield",)), ("qubit", 0.3, ("meanfield",)), ("qubit", 1.0, ("meanfield",))]
N_D, SEEDS = 8, 10


def job(a):
    import run, evalpolicy as ev
    from cbn import ACTIONS, ACT_BITS
    mode, width, rolls, seed = a
    env, enc, m = run.make(seed, N_D, CK[-1], mode=mode, width=width)
    x = np.zeros(m.Bi + 1, np.float32); x[:m.Bs] = enc.encode(env.view(), env.scalars())
    snaps, structure = {}, {}
    for t in range(CK[-1]):
        a_ = m.choose_action(x)
        x[m.Bs:m.Bs + ACT_BITS] = ACTIONS[a_]
        env.step(a_)
        y = enc.encode(env.view(), env.scalars())
        m.observe(x, y); x[:m.Bs] = y
        if (t + 1) % 500 == 0: m.grow()
        if (t + 1) in CK:
            snaps[t + 1] = ev.snapshot(m)
            prec, rec, fp = run.struct_metrics(env, m.edges())
            structure[t + 1] = dict(rec=rec, prec=prec, fp=fp, edges=m.n_edges(), neurons=m.n_neurons())
    out = dict(mode=mode, width=width, seed=seed, structure=structure, score={})
    for r in rolls:
        out["score"][r] = {ck: ev.eval_policy(env, enc, s, "exploit", 600, seed, rollout=r)[0] for ck, s in snaps.items()}
    out["random"] = ev.eval_policy(env, enc, snaps[500], "random", 600, seed)[0]
    out["expert"] = ev.eval_policy(env, enc, snaps[500], "expert", 600, seed)[0]
    return out


if __name__ == "__main__":
    jobs = [(m, w, r, s) for (m, w, r) in CONFIGS for s in range(SEEDS)]
    t0 = time.time()
    res = []
    with Pool(common.n_workers() - 2) as p:
        for i, o in enumerate(p.imap_unordered(job, jobs)):
            res.append(o)
            print(i + 1, len(jobs), o["mode"], o["width"], o["seed"], {k: round(v[16000], 3) for k, v in o["score"].items()}, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open("../results/expC.json", "w"), default=float)
