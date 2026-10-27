"""PIECE-CTX13: the FULL context stack, ONE setting for both worlds, end to end, fresh seeds.
Stack: recognisers 'input' (vote of 5 nearest labelled inputs) and 'recent' (lowest recent error, rate 0.1); META choice by consequences (running error of each recogniser's pick, rate 0.05; CTX12b);
SURPRISE GATE (G = 8, running mean rate 0.01, warm-up 20; CTX9); QUESTIONS (budget 5% of ticks so far): first on a shock, else on input-memory disagreement when the input recogniser is the trusted one (CTX9c/10).
Shocked and unasked: P6 hands the point to the best-predicting part (CTX9); B1 switches to the recent recogniser's set. Arms: metac (recognition only), metac_gate (+ gate), stack (+ questions).
Scores: P6 (place; seeds 6000+, 25% labels) true-routed part error on a held-out test set; B1 (time; high-contrast noise-free, seeds 9400+, 1% labels) prequential error of the COMMITTED prediction (iteration 70 rule).
PREDICTIONS P191-P193 (registered before running)."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pp6_self as P6
from pctx7_learnask import vote, K

REC, MR, G, RATE, WARM, BUDGET = 0.1, 0.05, 8.0, 0.01, 20, 0.05
ARMS = ("metac", "metac_gate", "stack")


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
    for arm in ARMS:
        crng = np.random.default_rng(seed + 777)
        parts = [AdamMLP(2, 1, P6.HP, P6.LR, seed * 100 + i) for i in range(P6.NR)]
        memX = np.zeros((P6.T, 2)); memY = np.zeros(P6.T, int); n = 0; rec = np.zeros(P6.NR); ce = np.zeros(2); avg = np.zeros(P6.NR); cnt = np.zeros(P6.NR); asks = 0
        for t in range(P6.T):
            x, y = xs[t], ys[t]
            errs = np.array([(p.predict(x)[0][0] - y) ** 2 for p in parts])
            gi, unan = vote(memX, memY, n, x) if n >= K else (0, True); gr = int(np.argmin(rec)); trust_in = ce[0] <= ce[1]
            if crng.random() < 0.25:
                k = region(x); memX[n], memY[n] = x, k; n += 1
            else:
                k = gi if trust_in else gr
                shock = arm != "metac" and cnt[k] >= WARM and errs[k] > G * avg[k]
                can = arm == "stack" and asks < BUDGET * (t + 1)
                if can and (shock or (trust_in and not unan)):
                    k = region(x); asks += 1; memX[n], memY[n] = x, k; n += 1
                elif shock:
                    k = int(np.argmin(errs))
            avg[k] = errs[k] if cnt[k] == 0 else (1 - RATE) * avg[k] + RATE * errs[k]; cnt[k] += 1
            rec = (1 - REC) * rec + REC * errs; ce = (1 - MR) * ce + MR * np.array([errs[gi], errs[gr]])
            parts[k].step(x, np.array([y]))
        Pall = np.array([[p.predict(x)[0][0] for p in parts] for x in test]); I = np.arange(len(test))
        out[arm] = dict(score=float(np.mean((Pall[I, tr] - ty) ** 2)), asks=asks / P6.T)
    return out


def run_b1(seed):
    from nn import AdamMLP, RunStd
    import pctx11b_contrast  # noqa: F401  (installs the high-contrast regimes)
    import pb1_memory as B
    X, Y, cp = B.stream(seed); T, n_in = X.shape; n_out = len(cp); D = (Y - X[:, :Y.shape[1]])[:, cp]
    reg = np.repeat(B.ORDER, B.BLOCK)
    out = {}
    for arm in ("oracle", "single") + ARMS:
        sx, sy = RunStd(n_in), RunStd(n_out); crng = np.random.default_rng(seed + 777)
        S = 1 if arm == "single" else 3
        sets = [[AdamMLP(n_in, 1, B.HID, B.LR, seed + j) for j in range(n_out)] for _ in range(S)]
        rec = np.zeros(S); ce = np.zeros(2); avg = np.zeros(S); cnt = np.zeros(S); memX, memY = [], []; err = 0.0; asks = 0; right = 0
        for t in range(T):
            sx.update(X[t]); sy.update(D[t]); xs, ys = sx(X[t]), sy(D[t])
            e = np.array([np.mean([(n.predict(xs)[0][0] - ys[j]) ** 2 for j, n in enumerate(s)]) for s in sets])
            if len(memX) >= K:
                d = ((np.array(memX) - xs) ** 2).sum(1); idx = np.argpartition(d, K - 1)[:K]; v = np.array(memY)[idx]
                gi, unan = int(np.bincount(v, minlength=3).argmax()), bool(np.all(v == v[0]))
            else:
                gi, unan = 0, True
            gr = int(np.argmin(rec)); trust_in = ce[0] <= ce[1]
            lab = crng.random() < 0.01
            if arm == "single":
                kp = k = 0
            elif arm == "oracle" or lab:
                kp = k = reg[t]
                if lab:
                    memX.append(xs); memY.append(reg[t])
            else:
                kp = k = gi if trust_in else gr                                # committed prediction
                shock = arm != "metac" and cnt[kp] >= WARM and e[kp] > G * avg[kp]
                can = arm == "stack" and asks < BUDGET * (t + 1)
                if can and (shock or (trust_in and not unan)):
                    k = reg[t]; asks += 1; memX.append(xs); memY.append(reg[t])
                elif shock:
                    k = gr if gr != kp else int(np.argsort(rec)[1])
            err += e[kp]; right += kp == reg[t]
            for j, n in enumerate(sets[k]):
                n.step(xs, ys[j:j + 1])
            avg[k] = e[k] if cnt[k] == 0 else (1 - RATE) * avg[k] + RATE * e[k]; cnt[k] += 1
            rec = (1 - REC) * rec + REC * e
            if S == 3:
                ce = (1 - MR) * ce + MR * np.array([e[gi], e[gr]])
        out[arm] = dict(score=float(err / T), asks=asks / T, acc=right / T)
    return out


if __name__ == "__main__":
    os.environ["NOISE"] = "0"
    t0 = time.time()
    with Pool(20) as p:
        r6 = p.map(run_p6, range(6000, 6048)); rb = p.map(run_b1, range(9400, 9448))
    json.dump(dict(p6=r6, b1=rb), open("../results/pctx13_stack.json", "w"))
    print("PLACE (P6): part error x oracle | asked")
    for a in ARMS:
        print(f"  {a:10s} {np.median([r[a]['score'] / r['oracle'] for r in r6]):.2f} | {np.median([r[a]['asks'] for r in r6]):.3f}")
    print("TIME (B1 contrast): prequential x oracle | regime accuracy | asked")
    for a in ("oracle", "single") + ARMS:
        print(f"  {a:10s} {np.median([r[a]['score'] / r['oracle']['score'] for r in rb]):.2f} | {np.median([r[a]['acc'] for r in rb]):.3f} | {np.median([r[a]['asks'] for r in rb]):.4f}")
    print("stack / metac (per world median): P6 %.2f | B1 %.2f" % (np.median([r["stack"]["score"] / r["metac"]["score"] for r in r6]), np.median([r["stack"]["score"] / r["metac"]["score"] for r in rb])))
    print("done", round(time.time() - t0), "s")
