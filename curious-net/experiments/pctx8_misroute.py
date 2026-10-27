"""PIECE-CTX8: WHY did the memory vote's fewer misroutes (7% vs 48%) give no better parts (CTX7, P161 failed)? Same world, seeds and 25% label draws as CTX7.
For every misrouted unlabelled training tick we record the receiving part's squared error BEFORE its update (the update size is proportional to it). Arms: best, memvote (as CTX7), voteskip (vote; where the
5 memories disagree, no update at all, no question). PREDICTIONS (registered before running): P164 mean receiving-part error on misroutes >= 3 x larger for memvote than best; P165 total misroute damage
(sum) memvote / best >= 1 (per-world median); P166 voteskip true-routed part error <= 0.7 x best."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pp6_self as P6
from pctx7_learnask import vote, K

ARMS = ("best", "memvote", "voteskip")


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
        memX = np.zeros((P6.T, 2)); memY = np.zeros(P6.T, int); n = 0; dmg = []; ok = []; skipped = 0
        for t in range(P6.T):
            x, y = xs[t], ys[t]
            if crng.random() < 0.25:
                k = region(x); memX[n], memY[n] = x, k; n += 1
            else:
                if arm != "best" and n >= K:
                    k, unan = vote(memX, memY, n, x)
                    if arm == "voteskip" and not unan:
                        skipped += 1; continue
                else:
                    k = int(np.argmin([(p.predict(x)[0][0] - y) ** 2 for p in parts]))
                e = float((parts[k].predict(x)[0][0] - y) ** 2)
                (dmg if k != region(x) else ok).append(e)
            parts[k].step(x, np.array([y]))
        Pall = np.array([[p.predict(x)[0][0] for p in parts] for x in test]); I = np.arange(len(test))
        out[arm] = dict(true=float(np.mean((Pall[I, tr] - ty) ** 2)), nmis=len(dmg), mean_mis=float(np.mean(dmg)) if dmg else 0.0, sum_mis=float(np.sum(dmg)),
                        mean_ok=float(np.mean(ok)) if ok else 0.0, skipped=skipped / P6.T)
    return out


if __name__ == "__main__":
    t0 = time.time(); off = int(os.environ.get("OFF", "0"))
    with Pool(20) as p:
        res = p.map(run, range(off, off + 48))
    json.dump(res, open("../results/pctx8_misroute%s.json" % os.environ.get("OFF", ""), "w"))
    for a in ARMS:
        m = lambda key: np.median([r[a][key] for r in res])
        print(f"{a:9s} parts x oracle {np.median([r[a]['true'] / r['oracle'] for r in res]):.2f} | misroutes {m('nmis'):.0f} | mean error on misroutes {m('mean_mis'):.4f} (on correct routes {m('mean_ok'):.4f}) | total misroute damage {m('sum_mis'):.1f} | skipped {m('skipped'):.3f}")
    rr = lambda a, b, key: np.median([r[a][key] / max(r[b][key], 1e-12) for r in res])
    print("memvote/best: mean misroute error %.2f | total damage %.2f | voteskip/best parts %.2f" % (rr("memvote", "best", "mean_mis"), rr("memvote", "best", "sum_mis"), rr("voteskip", "best", "true")))
    print("done", round(time.time() - t0), "s")
