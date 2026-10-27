"""PIECE-B7: a learning rate set by surprise, after the neuromodulator account of Yu & Dayan 2005 (acetylcholine = expected uncertainty, noradrenaline = unexpected uncertainty, e.g. a context switch).
Same stream as PIECE-B1 (one world, 3 hidden regimes, A B C A B C A B C, 600 ticks each, 48 worlds).
Learner: per-output online MLPs (Adam). Each output keeps a SLOW error average (expected uncertainty, window 200) and a FAST one (window 10). Surprise s = fast / slow.
Arms: adam (base lr); neuromod (lr x clip(s, 1, 8): learn faster while surprised); neuromod+reset (also resets Adam's moment estimates when s > 3, an 'interrupt' that drops stale gradient history);
lr2 and lr4 (constant 2x and 4x lr: is it just a bigger learning rate?). Windows, cap 8 and threshold 3 are EXPERIMENT settings, not tuned.
Metrics: NEW = first 100 ticks after switching into an unseen regime; RETURN = first 100 ticks after switching back; STEADY = last 300 ticks of each block; TOTAL.
PREDICTIONS (registered before running): P13 neuromod cuts NEW error >= 10% vs adam; P14 neuromod raises STEADY error by <= 3% vs adam; P15 neuromod has lower TOTAL error than both lr2 and lr4;
P16 the reset adds a further >= 5% NEW-error cut over neuromod alone."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pb1_memory as B

ARMS = ("adam", "neuromod", "neuromod_reset", "reset_only", "lr2", "lr4")
SLOW, FAST, CAP, THR = 200, 10, 8.0, 3.0


def run(seed):
    from nn import AdamMLP, RunStd
    X, Y, cp = B.stream(seed)
    T, n_in = X.shape; n_out = len(cp)
    D = (Y - X[:, :Y.shape[1]])[:, cp]
    sx, sy = RunStd(n_in), RunStd(n_out)
    nets = {a: [AdamMLP(n_in, 1, B.HID, B.LR * {"lr2": 2, "lr4": 4}.get(a, 1), seed + j) for j in range(n_out)] for a in ARMS}
    es = {a: np.ones(n_out) for a in ARMS}; ef = {a: np.ones(n_out) for a in ARMS}
    errs = {a: np.zeros(T) for a in ARMS}; resets = 0; near = 0
    for t in range(T):
        sx.update(X[t]); sy.update(D[t])
        xs, ys = sx(X[t]), sy(D[t])
        for a in ARMS:
            e_t = np.zeros(n_out)
            for j, net in enumerate(nets[a]):
                if a.startswith("neuromod") or a == "reset_only":
                    s = ef[a][j] / (es[a][j] + 1e-9)
                    if a.startswith("neuromod"):
                        net.lr = B.LR * float(np.clip(s, 1.0, CAP))
                    if a in ("neuromod_reset", "reset_only") and s > THR and t > SLOW:
                        for q in range(4):
                            net.m[q][...] = 0; net.v[q][...] = 0
                        net.t = 0
                        if a == "reset_only":
                            resets += 1; near += int((t % B.BLOCK) < 50)
                p = net.step(xs, ys[j:j + 1])[0]
                e_t[j] = (p - ys[j]) ** 2
            es[a] += (e_t - es[a]) / SLOW; ef[a] += (e_t - ef[a]) / FAST
            errs[a][t] = e_t.mean()
    out = {}
    for a, e in errs.items():
        blk = e.reshape(len(B.ORDER), B.BLOCK)
        out[a] = dict(new=float(blk[[1, 2], :100].mean()), ret=float(blk[3:, :100].mean()), steady=float(blk[:, -300:].mean()), total=float(e.mean()))
    out["resets"] = resets; out["near"] = near
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(9000, 9048))
    json.dump(res, open("../results/pb7_neuromod.json", "w"))
    for m in ("new", "ret", "steady", "total"):
        base = np.array([r["adam"][m] for r in res]); line = []
        for a in ARMS:
            rel = np.array([r[a][m] for r in res]) / base - 1
            line.append(f"{a} {100 * rel.mean():+.1f}% +- {100 * rel.std(ddof=1) / np.sqrt(len(rel)):.1f}")
        print(f"{m:6s} (vs adam, negative = better): " + " | ".join(line))
    nm = np.array([r["neuromod"]["new"] for r in res]); nr = np.array([r["neuromod_reset"]["new"] for r in res]); rel = nr / nm - 1
    print("reset vs neuromod on NEW: %+.1f%% +- %.1f; reset_only: mean resets per world %.0f, share within 50 ticks after a switch %.2f (chance 0.083)" % (100 * rel.mean(), 100 * rel.std(ddof=1) / np.sqrt(len(rel)), np.mean([r["resets"] for r in res]), np.sum([r["near"] for r in res]) / max(np.sum([r["resets"] for r in res]), 1)))
    for a in ("lr2", "lr4"):
        d = np.array([r["neuromod"]["total"] for r in res]) / np.array([r[a]["total"] for r in res]) - 1
        print(f"neuromod TOTAL vs {a}: {100 * d.mean():+.1f}% +- {100 * d.std(ddof=1) / np.sqrt(len(d)):.1f}")
    print("done", round(time.time() - t0), "s")
