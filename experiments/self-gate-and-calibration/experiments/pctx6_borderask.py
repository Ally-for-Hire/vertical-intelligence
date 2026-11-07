"""PIECE-CTX6: memory router + CHECKING AT BORDERS. As CTX5 (P6 world, 48 seeds, the same parts trained as CTX5), but at test time the memory router ASKS for the true context whenever its k = 5 nearest
labelled memories are not unanimous (disagreement among memories marks a border). Diagnostic arm 'true': the same parts routed by the true region, which isolates part error from routing error (T8 decomposition).
PREDICTIONS (registered before running): P158 at q = 1 border-asking routes >= 99.4% correctly, asking on <= 15% of test points; P159 at q = 1 border-asking error <= 1.5 x 'true'; P160 at q = 0.25 border-asking error <= 0.5 x memory."""
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
    op = [AdamMLP(2, 1, P6.HP, P6.LR, seed * 100 + i) for i in range(P6.NR)]
    for t in range(P6.T):
        op[region(xs[t])].step(xs[t], np.array([ys[t]]))
    oracle = float(np.mean((np.array([op[region(x)].predict(x)[0][0] for x in test]) - ty) ** 2))
    out = dict(oracle=oracle)
    for q in (1.0, 0.25):
        crng = np.random.default_rng(seed + 777)
        parts = [AdamMLP(2, 1, P6.HP, P6.LR, seed * 100 + i) for i in range(P6.NR)]; memX, memY = [], []
        for t in range(P6.T):
            x, y = xs[t], ys[t]
            if crng.random() < q:
                k = region(x); memX.append(x); memY.append(k)
            else:
                k = int(np.argmin([(p.predict(x)[0][0] - y) ** 2 for p in parts]))
            parts[k].step(x, np.array([y]))
        memY = np.array(memY); _, idx = cKDTree(np.array(memX)).query(np.array(test), k=K)
        votes = memY[idx]; mem = np.array([np.bincount(v, minlength=P6.NR).argmax() for v in votes])
        border = ~np.all(votes == votes[:, :1], axis=1); ask = np.where(border, tr, mem)
        Pall = np.array([[p.predict(x)[0][0] for p in parts] for x in test]); I = np.arange(len(test))
        for name, rt in (("memory", mem), ("ask", ask), ("true", tr)):
            out[f"{name}_q{q}"] = dict(acc=float(np.mean(rt == tr)), err=float(np.mean((Pall[I, rt] - ty) ** 2)))
        out[f"askrate_q{q}"] = float(border.mean())
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(int(os.environ.get("OFF", "0")), int(os.environ.get("OFF", "0")) + 48))
    json.dump(res, open("../results/pctx6_borderask%s.json" % os.environ.get("OFF", ""), "w"))
    for q in (1.0, 0.25):
        print(f"q={q}: ask rate median {np.median([r[f'askrate_q{q}'] for r in res]):.3f}")
        for name in ("memory", "ask", "true"):
            k = f"{name}_q{q}"
            print(f"  {name:7s} routing {np.median([r[k]['acc'] for r in res]):.4f} | error x oracle {np.median([r[k]['err'] / r['oracle'] for r in res]):.2f}")
        print(f"  ask / true (per world median) {np.median([r[f'ask_q{q}']['err'] / r[f'true_q{q}']['err'] for r in res]):.2f} | ask / memory {np.median([r[f'ask_q{q}']['err'] / r[f'memory_q{q}']['err'] for r in res]):.2f}")
    print("done", round(time.time() - t0), "s")
