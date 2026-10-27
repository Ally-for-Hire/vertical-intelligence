"""PIECE-CTX11: does the context stack TRANSFER to a world where context is TIME, not place? B1 stream: one causal world whose dynamics switch between 3 regimes, A B C A B C A B C, 600 ticks each, 48 worlds.
One part-set (per-output online MLPs as B1) per regime. The true regime is given on 1% of ticks (Q). Settings carried over unchanged from CTX9: G = 8, running-mean rate 0.01, warm-up 20; recent-error window: exponential, rate 0.1.
Arms (each tick: predict with the chosen set, score, then train that set):
  single    : one set, no context (B1's slow learner)
  oracle    : the true regime every tick
  lastlabel : keep the most recently labelled regime
  inputmem  : vote of the 5 nearest labelled INPUTS (CTX5-7's recogniser; expected to fail, since regimes share the input space)
  gate      : lastlabel; on a shock (chosen set's error > G x its running mean) switch to the set with the lowest recent error
  ask       : as gate, but on a shock ASK for the true regime while extra questions < 1% of ticks so far (else as gate)
Score: prequential error (normalised squared error, before each update) over all ticks; regime accuracy of the routing.
PREDICTIONS (registered before running): P177 inputmem accuracy <= 45%; P178 gate error <= 0.8 x lastlabel; P179 ask error <= 1.3 x oracle."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pb1_memory as B

Q, G, RATE, WARM, REC, BUDGET, K = 0.01, 8.0, 0.01, 20, 0.1, 0.01, 5
FIX = os.environ.get("SCOREFIX", "1") == "1"   # iteration 70 correction: on a shock tick, score the part-set the prediction was COMMITTED with (before the outcome), not the one switched to after seeing it. SCOREFIX=0 reproduces the flawed CTX11/11c scoring
ARMS = ("single", "oracle", "lastlabel", "inputmem", "gate", "ask")


def run(seed):
    from nn import AdamMLP, RunStd
    X, Y, cp = B.stream(seed); T, n_in = X.shape; n_out = len(cp); D = (Y - X[:, :Y.shape[1]])[:, cp]
    reg = np.repeat(B.ORDER, B.BLOCK)
    out = {}
    for arm in ARMS:
        sx, sy = RunStd(n_in), RunStd(n_out); crng = np.random.default_rng(seed + 777)
        sets = [[AdamMLP(n_in, 1, B.HID, B.LR, seed + j) for j in range(n_out)] for _ in range(3 if arm != "single" else 1)]
        S = len(sets); cur = 0; avg = np.zeros(S); cnt = np.zeros(S); rec = np.zeros(S); asks = 0; err = 0.0; right = 0; shocks = 0
        memX, memY = [], []
        for t in range(T):
            sx.update(X[t]); sy.update(D[t]); xs, ys = sx(X[t]), sy(D[t])
            e = np.array([np.mean([(n.predict(xs)[0][0] - ys[j]) ** 2 for j, n in enumerate(s)]) for s in sets])
            lab = crng.random() < Q; ks = None
            if arm == "single":
                k = 0
            elif arm == "oracle":
                k = reg[t]
            elif lab:
                cur = k = reg[t]; memX.append(xs); memY.append(reg[t])
            elif arm == "inputmem":
                if len(memX) >= K:
                    d = ((np.array(memX) - xs) ** 2).sum(1); idx = np.argpartition(d, K - 1)[:K]
                    cur = int(np.bincount(np.array(memY)[idx], minlength=3).argmax())
                k = cur
            else:
                if arm in ("gate", "ask") and cnt[cur] >= WARM and e[cur] > G * avg[cur]:
                    shocks += 1; ks = cur                                  # the prediction was already committed with 'cur' (SCOREFIX)
                    if arm == "ask" and asks < BUDGET * (t + 1):
                        cur = reg[t]; asks += 1
                    else:
                        cur = int(np.argmin(rec))
                k = cur
            err += e[ks if (FIX and ks is not None) else k]; right += (k == reg[t]) if arm != "single" else 0
            for j, n in enumerate(sets[k]):
                n.step(xs, ys[j:j + 1])
            avg[k] = e[k] if cnt[k] == 0 else (1 - RATE) * avg[k] + RATE * e[k]; cnt[k] += 1
            rec = (1 - REC) * rec + REC * e
        out[arm] = dict(err=float(err / T), acc=right / T, asks=asks / T, shocks=shocks / T)
    return out


if __name__ == "__main__":
    t0 = time.time(); off = int(os.environ.get("OFF", "9000"))
    with Pool(20) as p:
        res = p.map(run, range(off, off + 48))
    json.dump(res, open("../results/pctx11_transfer%d.json" % off, "w"))
    for a in ARMS:
        print(f"{a:9s} error {np.median([r[a]['err'] for r in res]):.4f} | x oracle {np.median([r[a]['err'] / r['oracle']['err'] for r in res]):.2f} | / lastlabel {np.median([r[a]['err'] / r['lastlabel']['err'] for r in res]):.2f} | regime accuracy {np.median([r[a]['acc'] for r in res]):.3f} | shocks {np.median([r[a]['shocks'] for r in res]):.4f} | asked {np.median([r[a]['asks'] for r in res]):.4f}")
    print("done", round(time.time() - t0), "s")
