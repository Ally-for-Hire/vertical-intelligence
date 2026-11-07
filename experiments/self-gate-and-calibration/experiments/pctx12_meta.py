"""PIECE-CTX12: one recogniser for BOTH kinds of context? Two recognisers, two worlds:
  input   : vote of the 5 nearest labelled INPUTS (works when context is a property of the input: place)
  recent  : the part(-set) with the lowest RECENT error (exponential average, rate 0.1, over all parts every tick; works when context persists in time)
  meta    : on every labelled tick, BEFORE using the label, score both recognisers' guesses (accuracy averages, rate 0.05, start 0.5); on unlabelled ticks use the currently more accurate one
Worlds: P6 (place; seeds 0-47, labels 25%, score = true-routed part error, as CTX7) and the B1 high-contrast noise-free stream (time; seeds 9200-9247, labels 1%, score = prequential error, as CTX11c).
No asking, no gate. PREDICTIONS P186-P188 (registered before running)."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pp6_self as P6
from pctx7_learnask import vote, K

REC, MR = 0.1, 0.05


def choose(arm, gi, gr, acc, ce=(0, 0)):
    return gi if arm == "input" or (arm == "meta" and acc[0] >= acc[1]) or (arm == "metac" and ce[0] <= ce[1]) else gr


def run_p6(seed):
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
    for arm in ("input", "recent", "meta", "metac"):
        crng = np.random.default_rng(seed + 777)
        parts = [AdamMLP(2, 1, P6.HP, P6.LR, seed * 100 + i) for i in range(P6.NR)]
        memX = np.zeros((P6.T, 2)); memY = np.zeros(P6.T, int); n = 0; rec = np.zeros(P6.NR); acc = np.array([0.5, 0.5]); ce = np.zeros(2); right = 0; unl = 0; use_in = 0
        for t in range(P6.T):
            x, y = xs[t], ys[t]
            gi = vote(memX, memY, n, x)[0] if n >= K else 0; gr = int(np.argmin(rec))
            if crng.random() < 0.25:
                k = region(x); acc = (1 - MR) * acc + MR * np.array([gi == k, gr == k]); memX[n], memY[n] = x, k; n += 1
            else:
                k = choose(arm, gi, gr, acc, ce); unl += 1; right += k == region(x); use_in += choose(arm, 1, 0, acc, ce)
            errs = np.array([(p.predict(x)[0][0] - y) ** 2 for p in parts]); rec = (1 - REC) * rec + REC * errs; ce = (1 - MR) * ce + MR * np.array([errs[gi], errs[gr]])
            parts[k].step(x, np.array([y]))
        Pall = np.array([[p.predict(x)[0][0] for p in parts] for x in test]); I = np.arange(len(test))
        out[arm] = dict(score=float(np.mean((Pall[I, tr] - ty) ** 2)), acc=right / unl, frac_input=use_in / unl)
    return out


def run_b1(seed):
    from nn import AdamMLP, RunStd
    import pctx11b_contrast  # noqa: F401  (installs the high-contrast regimes)
    import pb1_memory as B
    X, Y, cp = B.stream(seed); T, n_in = X.shape; n_out = len(cp); D = (Y - X[:, :Y.shape[1]])[:, cp]
    reg = np.repeat(B.ORDER, B.BLOCK)
    out = {}
    for arm in ("oracle", "lastlabel", "input", "recent", "meta", "metac"):
        sx, sy = RunStd(n_in), RunStd(n_out); crng = np.random.default_rng(seed + 777)
        sets = [[AdamMLP(n_in, 1, B.HID, B.LR, seed + j) for j in range(n_out)] for _ in range(3)]
        cur = 0; rec = np.zeros(3); acc = np.array([0.5, 0.5]); ce = np.zeros(2); memX, memY = [], []; err = 0.0; right = 0; use_in = 0; unl = 0
        for t in range(T):
            sx.update(X[t]); sy.update(D[t]); xs, ys = sx(X[t]), sy(D[t])
            e = np.array([np.mean([(n.predict(xs)[0][0] - ys[j]) ** 2 for j, n in enumerate(s)]) for s in sets])
            if len(memX) >= K:
                d = ((np.array(memX) - xs) ** 2).sum(1); idx = np.argpartition(d, K - 1)[:K]; gi = int(np.bincount(np.array(memY)[idx], minlength=3).argmax())
            else:
                gi = cur
            gr = int(np.argmin(rec))
            if arm == "oracle":
                k = reg[t]
            elif crng.random() < 0.01:
                acc = (1 - MR) * acc + MR * np.array([gi == reg[t], gr == reg[t]]); cur = k = reg[t]; memX.append(xs); memY.append(reg[t])
            else:
                k = cur if arm == "lastlabel" else choose(arm, gi, gr, acc, ce); unl += 1; use_in += arm != "lastlabel" and choose(arm, 1, 0, acc, ce)
            err += e[k]; right += k == reg[t]
            for j, n in enumerate(sets[k]):
                n.step(xs, ys[j:j + 1])
            rec = (1 - REC) * rec + REC * e; ce = (1 - MR) * ce + MR * np.array([e[gi], e[gr]])
        out[arm] = dict(score=float(err / T), acc=right / T, frac_input=use_in / max(unl, 1))
    return out


if __name__ == "__main__":
    os.environ["NOISE"] = "0"
    t0 = time.time()
    with Pool(20) as p:
        o6, ob = int(os.environ.get("OFF6", "0")), int(os.environ.get("OFFB", "9200"))
        r6 = p.map(run_p6, range(o6, o6 + 48)); rb = p.map(run_b1, range(ob, ob + 48))
    json.dump(dict(p6=r6, b1=rb), open("../results/pctx12_meta%s.json" % os.environ.get("OFF6", ""), "w"))
    print("P6 (place), part error x oracle | unlabelled routing accuracy | meta uses input")
    for a in ("input", "recent", "meta", "metac"):
        print(f"  {a:7s} {np.median([r[a]['score'] / r['oracle'] for r in r6]):.2f} | {np.median([r[a]['acc'] for r in r6]):.3f} | {np.median([r[a]['frac_input'] for r in r6]):.2f}")
    print("B1 contrast (time), prequential x oracle | / lastlabel | regime accuracy")
    for a in ("oracle", "lastlabel", "input", "recent", "meta", "metac"):
        print(f"  {a:9s} {np.median([r[a]['score'] / r['oracle']['score'] for r in rb]):.2f} | {np.median([r[a]['score'] / r['lastlabel']['score'] for r in rb]):.2f} | {np.median([r[a]['acc'] for r in rb]):.3f} | uses input {np.median([r[a]['frac_input'] for r in rb]):.2f}")
    best6 = lambda r: min(r["input"]["score"], r["recent"]["score"]); bestb = lambda r: min(r["input"]["score"], r["recent"]["score"])
    print("meta / better single (per world median): P6 %.2f | B1 %.2f" % (np.median([r["meta"]["score"] / best6(r) for r in r6]), np.median([r["meta"]["score"] / bestb(r) for r in rb])))
    print("metac / better single (per world median): P6 %.2f | B1 %.2f" % (np.median([r["metac"]["score"] / best6(r) for r in r6]), np.median([r["metac"]["score"] / bestb(r) for r in rb])))
    print("done", round(time.time() - t0), "s")
