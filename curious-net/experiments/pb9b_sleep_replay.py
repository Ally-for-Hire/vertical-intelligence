"""PIECE-B9b: sleep as OFFLINE replay at equal compute (the user's sleep idea: consolidate during internal ticks with the world frozen; B1b suggested replay during rest might keep the protection without the plasticity cost).
Stream: B1's regime switches (A B C A B C A B C, 600 ticks, 48 worlds). Every arm makes exactly 2 weight updates per awake tick on average:
  slow2       : 2 updates per tick on the current sample (no replay)
  interleaved : 1 update on the current sample + 1 replayed past sample per tick (B1's replay)
  sleep       : awake: 1 update per tick on the current sample; after every 100 awake ticks, a SLEEP of 100 replay updates from the reservoir buffer (no new input, no predictions; world frozen)
Reservoir buffer of 2,000 as B1. Metrics on awake ticks: NEW, RETURN (first 100 ticks after a switch), TOTAL.
PREDICTIONS (registered before running): P138 sleep RETURN error <= 0.95 x interleaved RETURN; P139 sleep TOTAL error <= interleaved TOTAL; P140 sleep RETURN error <= 0.9 x slow2 RETURN."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pb1_memory as B

SLEEP_EVERY, SLEEP_LEN = 100, 100


def run(seed):
    from nn import AdamMLP, RunStd
    X, Y, cp = B.stream(seed)
    T, n_in = X.shape; n_out = len(cp); D = (Y - X[:, :Y.shape[1]])[:, cp]
    sx, sy = RunStd(n_in), RunStd(n_out); rng = np.random.default_rng(seed + 99)
    arms = ("slow2", "interleaved", "sleep")
    nets = {a: [AdamMLP(n_in, 1, B.HID, B.LR, seed + j) for j in range(n_out)] for a in arms}
    bufX = np.zeros((B.BUF, n_in)); bufY = np.zeros((B.BUF, n_out)); nb = 0; seen = 0
    errs = {a: np.zeros(T) for a in arms}; upd = {a: 0 for a in arms}
    for t in range(T):
        sx.update(X[t]); sy.update(D[t]); xs, ys = sx(X[t]), sy(D[t])
        for a in arms:
            e = 0.0
            for j, n in enumerate(nets[a]):
                p = n.step(xs, ys[j:j + 1])[0]; e += (p - ys[j]) ** 2
            errs[a][t] = e / n_out; upd[a] += 1
        for j, n in enumerate(nets["slow2"]):
            n.step(xs, ys[j:j + 1])
        upd["slow2"] += 1
        if nb > 0:
            q = int(rng.integers(nb))
            for j, n in enumerate(nets["interleaved"]):
                n.step(bufX[q], bufY[q, j:j + 1])
            upd["interleaved"] += 1
        seen += 1
        if nb < B.BUF:
            bufX[nb], bufY[nb] = xs, ys; nb += 1
        else:
            k = int(rng.integers(seen))
            if k < B.BUF:
                bufX[k], bufY[k] = xs, ys
        if (t + 1) % SLEEP_EVERY == 0:                                   # sleep: offline replay, world frozen
            for _ in range(SLEEP_LEN):
                q = int(rng.integers(nb))
                for j, n in enumerate(nets["sleep"]):
                    n.step(bufX[q], bufY[q, j:j + 1])
                upd["sleep"] += 1
    out = {}
    for a, e in errs.items():
        blk = e.reshape(len(B.ORDER), B.BLOCK)
        out[a] = dict(new=float(blk[[1, 2], :100].mean()), ret=float(blk[3:, :100].mean()), total=float(e.mean()), upd=upd[a])
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(9000, 9048))
    json.dump(res, open("../results/pb9b_sleep_replay.json", "w"))
    for m in ("new", "ret", "total"):
        print(f"{m:5s}: " + " | ".join(f"{a} {np.median([r[a][m] for r in res]):.3f}" for a in ("slow2", "interleaved", "sleep")) +
              f" | sleep/interleaved {np.median([r['sleep'][m] / r['interleaved'][m] for r in res]):.3f} | sleep/slow2 {np.median([r['sleep'][m] / r['slow2'][m] for r in res]):.3f}")
    print("updates per arm (mean):", {a: np.mean([r[a]['upd'] for r in res]) for a in ("slow2", "interleaved", "sleep")})
    print("done", round(time.time() - t0), "s")
