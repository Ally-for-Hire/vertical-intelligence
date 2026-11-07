"""Exp K: HEADROOM. How much faster does an online MLP learn a world when it is told which inputs matter (oracle causal mask), as a function of the number of
distractor inputs? If the oracle mask gives little speed-up, the premise 'knowing what affects the environment makes learning faster' has no room.
Targets: next-step change of the core variables (standardised online); inputs: every observed variable + the actions (standardised online).
Plain MLP sees all inputs; masked MLP sees only the true parents of the core outputs (same width, same optimiser). Hyperparameters swept per arm on
validation worlds. Metric: prequential normalised MSE on core outputs; area under the curve over the first 1000 ticks; ticks to reach 1.5x the masked net's final error."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

T = 3000


def run_world(args):
    seed, n_d, arm, hidden, lr = args
    from worlds import World, rollout
    from nn import AdamMLP, RunStd
    w = World(seed, n_state=6, n_act=2, n_distract=n_d)
    rng = np.random.default_rng(seed + 7)
    X, Y = rollout(w, T, rng)
    ns, na = w.n_state, w.n_act
    truth = w.truth()
    core_pos = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]          # observed positions of the core variables
    D = Y - X[:, :ns]                                                                     # change of each variable (target = delta)
    Din = X
    sx, sy = RunStd(ns + na), RunStd(len(core_pos))
    mask = None
    if arm == "oracle":
        mask = truth[core_pos].any(0).astype(float)                                        # union of true parents of the core outputs
    net = AdamMLP(ns + na, len(core_pos), hidden, lr, seed, mask)
    err = np.zeros(T)
    for t in range(T):
        x = X[t]; y = D[t][core_pos]
        sx.update(x); sy.update(y)
        p = net.step(sx(x), sy(y))
        err[t] = np.mean((p - sy(y)) ** 2)
    return dict(seed=seed, n_d=n_d, arm=arm, hidden=hidden, lr=lr, err=err.tolist(), n_in_masked=int(mask.sum()) if mask is not None else ns + na)


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        # 1. fair tuning on validation worlds (seeds 1000..1005), n_d=50
        grid = [(h, lr) for h in (32, 128) for lr in (3e-4, 1e-3, 3e-3, 1e-2)]
        best = {}
        for arm in ("plain", "oracle"):
            jobs = [(1000 + s, 50, arm, h, lr) for s in range(6) for (h, lr) in grid]
            res = p.map(run_world, jobs)
            sc = {}
            for r in res: sc.setdefault((r["hidden"], r["lr"]), []).append(float(np.mean(r["err"][:1000])))
            sc = {k: float(np.mean(v)) for k, v in sc.items()}
            best[arm] = min(sc, key=sc.get)
            print(arm, "validation (mean first-1000 err):", {str(k): round(v, 3) for k, v in sc.items()}, "-> chose", best[arm], flush=True)
        jobs = [(s, nd, arm, *best[arm]) for s in range(24) for nd in (0, 10, 50, 200) for arm in ("plain", "oracle")]
        out = p.map(run_world, jobs)
    json.dump(dict(res=out, best={k: list(v) for k, v in best.items()}, T=T), open("../results/expK.json", "w"), default=float)
    print("done", round(time.time() - t0), "s")
