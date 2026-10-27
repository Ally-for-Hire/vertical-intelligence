"""PIECE-B8: three-factor plasticity (eligibility trace + a later third factor; Gerstner et al. 2018 [v]) for credit assignment under DELAYED reward.
Task: contextual bandit, 4 contexts (one-hot) x 4 arms; the correct arm per context is fixed; reward 1 for the correct arm else 0, but it ARRIVES d steps later (d in 0, 3, 10), while new choices keep being made.
Policy: softmax(W x), W 4x4, learning rate 0.1, reward baseline = running mean.
Learners: oracle (each reward is applied to the gradient of the choice that caused it: exact credit); trace (three-factor: eligibility e <- lam e + grad log pi of the current choice; each arriving reward updates W by
(r - baseline) x e; lam = 0.8, an EXPERIMENT setting); no_trace (each arriving reward is applied to the CURRENT choice's gradient: no memory of what caused it). 3,000 steps, 200 seeds.
Score: accuracy (correct arm chosen) over the last 1,000 steps; chance = 0.25.
PREDICTIONS (registered before running): P121 at d = 0 all three are within 0.02 of each other; P122 at d = 3 the trace keeps >= 70% of the oracle's above-chance accuracy, no_trace <= 20%; P123 at d = 10 the trace keeps <= 50% (credit by trace degrades with delay)."""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool

NC, NA, T, LR, LAM = 4, 4, 3000, 0.1, 0.8


def run(args):
    seed, d = args
    rng0 = np.random.default_rng(seed); correct = rng0.integers(0, NA, NC)
    out = {}
    kinds = ("oracle", "trace", "multi") if os.environ.get("B8B") else ("oracle", "trace", "no_trace")
    for lr_kind in kinds:
        rng = np.random.default_rng(seed + 1); W = np.zeros((NA, NC)); e = np.zeros_like(W); base = 0.0
        es = [np.zeros_like(W) for _ in (0.5, 0.8, 0.95)]
        pending = []; hits = []
        for t in range(T):
            c = int(rng.integers(NC)); x = np.zeros(NC); x[c] = 1.0
            z = W @ x; p = np.exp(z - z.max()); p /= p.sum()
            a = int(rng.choice(NA, p=p)); r = float(a == correct[c]); hits.append(r)
            g = -np.outer(p, x); g[a] += x                                  # grad of log pi(a | x)
            e = LAM * e + g
            es = [l * ee + g for l, ee in zip((0.5, 0.8, 0.95), es)]
            pending.append((t + d, r, g))
            arrived = [q for q in pending if q[0] == t]; pending = [q for q in pending if q[0] > t]
            for _, rr, gg in arrived:
                adv = rr - base; base += 0.01 * (rr - base)
                if lr_kind == "oracle":
                    W += LR * adv * gg
                elif lr_kind == "trace":
                    W += LR * adv * e
                elif lr_kind == "multi":
                    W += LR * adv * sum(ee * (1 - l) for l, ee in zip((0.5, 0.8, 0.95), es)) / 3 * 5
                else:
                    W += LR * adv * g
        out[lr_kind] = float(np.mean(hits[-1000:]))
    return d, out


if __name__ == "__main__" and os.environ.get("B8B"):
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, [(s, d) for d in range(1, 16) for s in range(60)])
    short = [o for d, o in res if d <= 3]; allo = [o for d, o in res]
    for name, grp in (("delays 1-15", allo), ("delays 1-3", short), ("delays 10-15", [o for d, o in res if d >= 10])):
        print(f"{name}: oracle {np.mean([o['oracle'] for o in grp]):.3f} | fixed trace 0.8 {np.mean([o['trace'] for o in grp]):.3f} | multi-timescale {np.mean([o['multi'] for o in grp]):.3f}")
    json.dump([[d, o] for d, o in res], open("../results/pb8b_threefactor.json", "w"))
    print("done", round(time.time() - t0), "s"); raise SystemExit
if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, [(s, d) for d in (0, 3, 10) for s in range(int(os.environ.get("OFF", "0")), int(os.environ.get("OFF", "0")) + 200)])
    allr = {}
    for d in (0, 3, 10):
        r = [o for dd, o in res if dd == d]; allr[d] = r
        m = {k: np.mean([o[k] for o in r]) for k in ("oracle", "trace", "no_trace")}
        keep = {k: (m[k] - 0.25) / max(m["oracle"] - 0.25, 1e-9) for k in ("trace", "no_trace")}
        print(f"delay {d:2d}: accuracy oracle {m['oracle']:.3f} | trace {m['trace']:.3f} (keeps {keep['trace']:.0%} of oracle's above-chance) | no_trace {m['no_trace']:.3f} (keeps {keep['no_trace']:.0%})")
    json.dump({str(k): v for k, v in allr.items()}, open("../results/pb8_threefactor%s.json" % os.environ.get("OFF", ""), "w"))
    print("done", round(time.time() - t0), "s")
