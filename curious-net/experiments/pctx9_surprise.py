"""PIECE-CTX9: a SURPRISE GATE, no questions. Same world, seeds and 25% label draws as CTX7/8. Unlabelled ticks:
  best     : best-predicting part
  memvote  : vote of the 5 nearest labelled memories
  hybrid   : vote where the 5 memories agree, best-predicting where they disagree
  gate     : vote; if the receiving part's error on the point > G x its running mean error on points it accepted (exponential average, rate 0.01; the gate is off for a part's first 20 updates),
             hand the point to the best-predicting part instead (CTX8: those assignments are nearly harmless)
  gateh    : hybrid + gate
EXPERIMENT settings: G = 4, rate 0.01, warm-up 20 (unbacked; see PREDICTIONS P167-P169)."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pp6_self as P6
from pctx7_learnask import vote, K

G, RATE, WARM = float(os.environ.get("G", "4")), 0.01, 20
ARMS = ("best", "memvote", "hybrid", "gate", "gateh")


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
        crng = np.random.default_rng(seed + 777)
        parts = [AdamMLP(2, 1, P6.HP, P6.LR, seed * 100 + i) for i in range(P6.NR)]
        memX = np.zeros((P6.T, 2)); memY = np.zeros(P6.T, int); n = 0; avg = np.zeros(P6.NR); cnt = np.zeros(P6.NR); gated = 0; wrong = 0; unl = 0
        for t in range(P6.T):
            x, y = xs[t], ys[t]
            errs = np.array([(p.predict(x)[0][0] - y) ** 2 for p in parts])
            if crng.random() < 0.25:
                k = region(x); memX[n], memY[n] = x, k; n += 1
            else:
                unl += 1; k = int(np.argmin(errs))
                if arm != "best" and n >= K:
                    v, unan = vote(memX, memY, n, x)
                    if arm in ("memvote", "gate") or unan:
                        if arm.startswith("gate") and cnt[v] >= WARM and errs[v] > G * avg[v]:
                            gated += 1
                        else:
                            k = v
                wrong += k != region(x)
            avg[k] = errs[k] if cnt[k] == 0 else (1 - RATE) * avg[k] + RATE * errs[k]; cnt[k] += 1
            parts[k].step(x, np.array([y]))
        Pall = np.array([[p.predict(x)[0][0] for p in parts] for x in test]); I = np.arange(len(test))
        out[arm] = dict(true=float(np.mean((Pall[I, tr] - ty) ** 2)), gated=gated / P6.T, wrong=wrong / max(unl, 1))
    return out


if __name__ == "__main__":
    t0 = time.time(); off = int(os.environ.get("OFF", "0"))
    with Pool(20) as p:
        res = p.map(run, range(off, off + 48))
    json.dump(res, open("../results/pctx9_surprise%s_G%s.json" % (os.environ.get("OFF", ""), G), "w"))
    for a in ARMS:
        print(f"{a:8s} parts x oracle {np.median([r[a]['true'] / r['oracle'] for r in res]):.2f} | / best {np.median([r[a]['true'] / r['best']['true'] for r in res]):.2f} | unlabelled misrouted {np.median([r[a]['wrong'] for r in res]):.3f} | gated {np.median([r[a]['gated'] for r in res]):.3f}")
    print("done", round(time.time() - t0), "s")
