"""PIECE-CTX7: LEARNING-time context inference. P6 world, 48 seeds (OFF env shifts seeds). Labels (true region) on 25% of training ticks (same draws as CTX5/6). Unlabelled ticks are routed to a part by:
  best     : the best-predicting part (as CTX5/6)
  memvote  : the vote of the 5 nearest labelled memories (fallback best-predicting while fewer than 5 memories)
  memask   : as memvote, but where the 5 memories are not unanimous it ASKS for the true region if the question budget allows (asks <= 10% of ticks so far); answers are stored as memories
  random35 : control with the same extra-label budget spent at random: labels on 35% of ticks (a superset of the 25% draws), rest best-predicting
Test: 'true' = parts routed by the true region (part quality); 'ask' = memory router + border asking at test (as CTX6). EXPERIMENT settings: K = 5, budget 10%.
PREDICTIONS (registered before running): P161 memvote true-routed error <= 0.6 x best; P162 memask true-routed error <= 0.8 x random35; P163 memask with test border asking <= 2 x oracle."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pp6_self as P6

K, BUDGET = 5, 0.10
ARMS = ("best", "memvote", "memask", "random35")


def vote(memX, memY, n, x):
    d = ((memX[:n] - x) ** 2).sum(1); idx = np.argpartition(d, K - 1)[:K]; v = memY[idx]
    return int(np.bincount(v, minlength=P6.NR).argmax()), bool(np.all(v == v[0]))


def run(seed):
    from nn import AdamMLP
    rng = np.random.default_rng(seed); region, f = P6.world(rng)
    xs, ys = [], []
    for t in range(P6.T):
        x = P6.sample_x(rng, min(P6.NR, 1 + t // P6.UNLOCK), region); xs.append(x); ys.append(f(x) + 0.05 * rng.normal())
    test = np.array([P6.sample_x(rng, P6.NR, region) for _ in range(2000)]); ty = np.array([f(x) for x in test]); tr = np.array([region(x) for x in test])
    op = [AdamMLP(2, 1, P6.HP, P6.LR, seed * 100 + i) for i in range(P6.NR)]
    for t in range(P6.T):
        op[region(xs[t])].step(xs[t], np.array([ys[t]]))
    oracle = float(np.mean((np.array([op[region(x)].predict(x)[0][0] for x in test]) - ty) ** 2))
    out = dict(oracle=oracle)
    for arm in ARMS:
        crng = np.random.default_rng(seed + 777); q = 0.35 if arm == "random35" else 0.25
        parts = [AdamMLP(2, 1, P6.HP, P6.LR, seed * 100 + i) for i in range(P6.NR)]
        memX = np.zeros((P6.T, 2)); memY = np.zeros(P6.T, int); n = 0; asks = 0; wrong = 0; unl = 0
        for t in range(P6.T):
            x, y = xs[t], ys[t]
            if crng.random() < q:
                k = region(x); memX[n], memY[n] = x, k; n += 1
            else:
                unl += 1
                if arm in ("memvote", "memask") and n >= K:
                    k, unan = vote(memX, memY, n, x)
                    if arm == "memask" and not unan and asks < BUDGET * (t + 1):
                        k = region(x); asks += 1; memX[n], memY[n] = x, k; n += 1
                else:
                    k = int(np.argmin([(p.predict(x)[0][0] - y) ** 2 for p in parts]))
                wrong += k != region(x)
            parts[k].step(x, np.array([y]))
        Pall = np.array([[p.predict(x)[0][0] for p in parts] for x in test]); I = np.arange(len(test))
        tv = [vote(memX, memY, n, x) for x in test]; mem = np.array([v[0] for v in tv]); border = np.array([not v[1] for v in tv])
        ask = np.where(border, tr, mem)
        out[arm] = dict(true=float(np.mean((Pall[I, tr] - ty) ** 2)), ask=float(np.mean((Pall[I, ask] - ty) ** 2)), asks=asks / P6.T, labels=n / P6.T, trainwrong=wrong / max(unl, 1), testask=float(border.mean()))
    return out


if __name__ == "__main__":
    t0 = time.time(); off = int(os.environ.get("OFF", "0"))
    with Pool(20) as p:
        res = p.map(run, range(off, off + 48))
    json.dump(res, open("../results/pctx7_learnask%s.json" % os.environ.get("OFF", ""), "w"))
    for a in ARMS:
        m = lambda key: np.median([r[a][key] for r in res])
        print(f"{a:9s} true-routed x oracle {np.median([r[a]['true'] / r['oracle'] for r in res]):.2f} | test-ask x oracle {np.median([r[a]['ask'] / r['oracle'] for r in res]):.2f} | labels {m('labels'):.3f} (asked {m('asks'):.3f}) | unlabelled ticks misrouted {m('trainwrong'):.3f} | test ask rate {m('testask'):.3f}")
    ratio = lambda a, b: np.median([r[a]["true"] / r[b]["true"] for r in res])
    print("memvote / best %.2f | memask / random35 %.2f | memask / best %.2f" % (ratio("memvote", "best"), ratio("memask", "random35"), ratio("memask", "best")))
    print("done", round(time.time() - t0), "s")
