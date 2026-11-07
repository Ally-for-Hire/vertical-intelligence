"""PIECE-CTX9c: (a) replicate the G = 8 surprise gate on fresh seeds 9500-9547 (P171); (b) WHERE to spend questions: on memory disagreement (CTX7), on shocks, or on either (P172, P173).
Same world and 25% label draws as CTX7/9. Question budget 10% of ticks so far (answers stored as memories). Shock: receiving part's error > G x its running mean (rate 0.01, warm-up 20), G = 8.
Arms: best; gate8 (no questions; shocked points go to the best-predicting part); ALL ask arms sit on top of gate8 (an unasked shocked point goes to the best-predicting part), so they differ only in WHERE questions go: askdis; askshock (accept the vote unless shocked, then ask; if the budget is spent, fall back to gate8 behaviour); askboth."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pp6_self as P6
from pctx7_learnask import vote, K

G, RATE, WARM, BUDGET = 8.0, 0.01, 20, float(os.environ.get("BUDGET", "0.10"))
ARMS = ("best", "gate8", "askdis", "askshock", "askboth")


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
        memX = np.zeros((P6.T, 2)); memY = np.zeros(P6.T, int); n = 0; avg = np.zeros(P6.NR); cnt = np.zeros(P6.NR); asks = 0
        for t in range(P6.T):
            x, y = xs[t], ys[t]
            errs = np.array([(p.predict(x)[0][0] - y) ** 2 for p in parts])
            if crng.random() < 0.25:
                k = region(x); memX[n], memY[n] = x, k; n += 1
            else:
                k = int(np.argmin(errs))
                if arm != "best" and n >= K:
                    v, unan = vote(memX, memY, n, x)
                    shock = cnt[v] >= WARM and errs[v] > G * avg[v]
                    trig = {"askdis": not unan, "askshock": shock, "askboth": (not unan) or shock}.get(arm, False)
                    if trig and asks < BUDGET * (t + 1):
                        k = region(x); asks += 1; memX[n], memY[n] = x, k; n += 1
                    elif not shock:
                        k = v
            avg[k] = errs[k] if cnt[k] == 0 else (1 - RATE) * avg[k] + RATE * errs[k]; cnt[k] += 1
            parts[k].step(x, np.array([y]))
        Pall = np.array([[p.predict(x)[0][0] for p in parts] for x in test]); I = np.arange(len(test))
        out[arm] = dict(true=float(np.mean((Pall[I, tr] - ty) ** 2)), asks=asks / P6.T)
    return out


if __name__ == "__main__":
    t0 = time.time(); off = int(os.environ.get("OFF", "9500"))
    with Pool(20) as p:
        res = p.map(run, range(off, off + 48))
    json.dump(res, open("../results/pctx9c_shockask%d_b%s.json" % (off, BUDGET), "w"))
    for a in ARMS:
        print(f"{a:9s} parts x oracle {np.median([r[a]['true'] / r['oracle'] for r in res]):.2f} | / best {np.median([r[a]['true'] / r['best']['true'] for r in res]):.2f} | / askdis {np.median([r[a]['true'] / r['askdis']['true'] for r in res]):.2f} | asked {np.median([r[a]['asks'] for r in res]):.3f}")
    print("done", round(time.time() - t0), "s")
