"""PIECE-CTX5: a MEMORY-BASED router (hippocampus-style episodic index: route an input by its k = 5 nearest remembered inputs whose context was labelled) vs the learned parametric router (softmax on 64 random
Fourier features), with the SAME specialist parts. P6 world, 48 seeds. Labels (the true region) on a fraction q of training ticks, q in {1, 0.25}; unlabelled ticks train the best-predicting part (as CTX).
The memory stores only labelled ticks (it never sees labels it was not given).
PREDICTIONS (registered before running): P155 at q = 1 the memory router sends >= 98% of test points to the right part (parametric: 93.6%); P156 at q = 1 test error with the memory router <= 2 x oracle; P157 at q = 0.25 memory-routed test error <= 3 x oracle."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
from scipy.spatial import cKDTree
import pp6_self as P6

NF, RLR, K = 64, 0.05, 5


def run(seed):
    from nn import AdamMLP
    rng = np.random.default_rng(seed); region, f = P6.world(rng)
    xs, ys = [], []
    for t in range(P6.T):
        x = P6.sample_x(rng, min(P6.NR, 1 + t // P6.UNLOCK), region); xs.append(x); ys.append(f(x) + 0.05 * rng.normal())
    test = [P6.sample_x(rng, P6.NR, region) for _ in range(2000)]; ty = np.array([f(x) for x in test]); tr = np.array([region(x) for x in test])
    frng = np.random.default_rng(seed + 4242); Wf = frng.normal(0, 3, (2, NF)); bf = frng.uniform(0, 2 * np.pi, NF)
    phi = lambda x: np.cos(x @ Wf + bf) * np.sqrt(2 / NF)
    # oracle reference: parts routed by the true region at train and test
    op = [AdamMLP(2, 1, P6.HP, P6.LR, seed * 100 + i) for i in range(P6.NR)]
    for t in range(P6.T):
        op[region(xs[t])].step(xs[t], np.array([ys[t]]))
    oracle = float(np.mean((np.array([op[region(x)].predict(x)[0][0] for x in test]) - ty) ** 2))
    out = dict(oracle=oracle)
    for q in (1.0, 0.25):
        crng = np.random.default_rng(seed + 777)
        parts = [AdamMLP(2, 1, P6.HP, P6.LR, seed * 100 + i) for i in range(P6.NR)]; R = np.zeros((P6.NR, NF)); memX, memY = [], []
        for t in range(P6.T):
            x, y = xs[t], ys[t]; ph = phi(x); lg = R @ ph; pr_ = np.exp(lg - lg.max()); pr_ /= pr_.sum()
            if crng.random() < q:
                k = region(x); memX.append(x); memY.append(k)
            else:
                k = int(np.argmin([(p.predict(x)[0][0] - y) ** 2 for p in parts]))
            parts[k].step(x, np.array([y])); tg = np.zeros(P6.NR); tg[k] = 1.0; R += RLR * np.outer(tg - pr_, ph)
        tree = cKDTree(np.array(memX)); memY = np.array(memY)
        _, idx = tree.query(np.array(test), k=K)
        mem_route = np.array([np.bincount(memY[i], minlength=P6.NR).argmax() for i in idx])
        par_route = np.array([int(np.argmax(R @ phi(x))) for x in test])
        Pall = np.array([[p.predict(x)[0][0] for p in parts] for x in test])
        for name, rt in (("memory", mem_route), ("parametric", par_route)):
            out[f"{name}_q{q}"] = dict(acc=float(np.mean(rt == tr)), err=float(np.mean((Pall[np.arange(len(test)), rt] - ty) ** 2)))
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(48))
    json.dump(res, open("../results/pctx5_memrouter.json", "w"))
    for q in (1.0, 0.25):
        for name in ("memory", "parametric"):
            k = f"{name}_q{q}"
            print(f"q={q} {name:10s} routing accuracy median {np.median([r[k]['acc'] for r in res]):.3f} | test error x oracle median {np.median([r[k]['err'] / r['oracle'] for r in res]):.2f}")
    print("done", round(time.time() - t0), "s")
