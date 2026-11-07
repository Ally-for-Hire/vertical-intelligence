"""PIECE-P6: Self v0 with parts generated on demand (the user: the Self has no parts; parts and sub-parts are generated as situations require more context).
World: inputs x in [-1, 1]^2; the plane is split into 6 hidden REGIONS (angular sectors), each with its own rule y = tanh(a . x + b) * c + d (+ noise 0.05). Regions UNLOCK over time: only region 1 is
sampled at first, one more every 1,000 ticks (6 by tick 5,000); 6,000 ticks in total. The learner never sees region labels.
Learners (all online, Adam, batch 1; parts are 2 -> 8 -> 1 nets):
  small  : one 8-unit net
  wide   : one 48-unit net (same total hidden units as 6 parts)
  oracle : 6 parts, each sample routed to the part of its TRUE region (upper reference)
  self   : starts with ONE part. Each part keeps the centroid of the inputs it handles and its own error baseline. Inputs are routed to the nearest centroid. If the last 100 errors of a part are
           > SPAWN x its baseline error (the mean of its errors between its ticks 200-400, i.e. once it had settled), a new part is spawned at the centroid of that part's worst recent inputs. SPAWN = 2 (EXPERIMENT setting).
Score: test error over all 6 regions at the end (2,000 points); prequential error; parts at the end; weight-update compute (sum of hidden units updated per tick).
PREDICTIONS (registered before running): P78 self ends with 6 +- 2 parts (median over 48 seeds); P79 self final test error <= 0.8 x wide; P80 self final test error <= 1.2 x oracle; P81 self uses <= 60% of wide's update compute."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

T, NR, UNLOCK, SPAWN, HP, LR = 6000, 6, 1000, 2.0, 8, 3e-3


def world(rng):
    params = [(rng.normal(0, 2, 2), rng.normal(0, 0.5), rng.uniform(0.5, 1.5), rng.normal(0, 0.5)) for _ in range(NR)]
    def region(x):
        return int(((np.arctan2(x[1], x[0]) + np.pi) / (2 * np.pi) * NR)) % NR
    def f(x):
        a, b, c, d = params[region(x)]
        return float(np.tanh(a @ x + b) * c + d)
    return region, f


def sample_x(rng, n_open, region):
    while True:
        x = rng.uniform(-1, 1, 2)
        if region(x) < n_open:
            return x


class Part:
    def __init__(self, seed, centroid):
        from nn import AdamMLP
        self.net = AdamMLP(2, 1, HP, LR, seed); self.c = np.array(centroid, float); self.n = 0
        self.errs = []; self.base = None; self.recent_x = []

    def step(self, x, y):
        p = self.net.step(x, np.array([y]))[0]; e = (p - y) ** 2
        self.n += 1; self.c += (x - self.c) / min(self.n, 200)
        self.errs.append(e); self.recent_x.append((e, x))
        self.recent_x = self.recent_x[-100:]
        if self.base is None and self.n == 400:
            self.base = float(np.mean(self.errs[200:400]))
        return e


def run(seed):
    from nn import AdamMLP
    rng = np.random.default_rng(seed); region, f = world(rng)
    xs, ys = [], []
    for t in range(T):
        x = sample_x(rng, min(NR, 1 + t // UNLOCK), region); xs.append(x); ys.append(f(x) + 0.05 * rng.normal())
    test = [sample_x(rng, NR, region) for _ in range(2000)]; ty = np.array([f(x) for x in test])
    out = {}
    for arm in ("small", "wide"):
        net = AdamMLP(2, 1, HP if arm == "small" else HP * NR, LR, seed)
        pe = [(net.step(xs[t], np.array([ys[t]]))[0] - ys[t]) ** 2 for t in range(T)]
        pr = np.array([net.predict(x)[0][0] for x in test])
        out[arm] = dict(test=float(np.mean((pr - ty) ** 2)), preq=float(np.mean(pe)), parts=1, compute=float(T * (HP if arm == "small" else HP * NR)))
    parts = [Part(seed * 10 + k, [0, 0]) for k in range(NR)]
    pe = [parts[region(xs[t])].step(xs[t], ys[t]) for t in range(T)]
    pr = np.array([parts[region(x)].net.predict(x)[0][0] for x in test])
    out["oracle"] = dict(test=float(np.mean((pr - ty) ** 2)), preq=float(np.mean(pe)), parts=NR, compute=float(T * HP))
    parts = [Part(seed * 100, xs[0])]; pe = []; spawns = []
    for t in range(T):
        x = xs[t]; k = int(np.argmin([np.sum((p.c - x) ** 2) for p in parts]))
        pe.append(parts[k].step(x, ys[t]))
        p = parts[k]
        if p.base is not None and len(p.errs) >= 100 and np.mean(p.errs[-100:]) > SPAWN * p.base:
            worst = sorted(p.recent_x, key=lambda z: -z[0])[:20]
            parts.append(Part(seed * 100 + len(parts), np.mean([w[1] for w in worst], 0)))
            p.errs = p.errs[:-100] + [p.base] * 0; p.recent_x = []; spawns.append(t)
            p.errs = p.errs[-100:] if len(p.errs) > 100 else p.errs
            p.errs = [p.base] * 100                                     # reset its recent window so one burst spawns one part
    pr = np.array([parts[int(np.argmin([np.sum((q.c - x) ** 2) for q in parts]))].net.predict(x)[0][0] for x in test])
    out["self"] = dict(test=float(np.mean((pr - ty) ** 2)), preq=float(np.mean(pe)), parts=len(parts), compute=float(T * HP), spawns=spawns)
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(48))
    json.dump(res, open("../results/pp6_self.json", "w"), default=float)
    w = np.array([r["wide"]["test"] for r in res]); o = np.array([r["oracle"]["test"] for r in res])
    print("arm    | final test error (median) | ratio to wide (median) | ratio to oracle (median) | prequential | parts (median, range) | update compute vs wide")
    for a in ("small", "wide", "oracle", "self"):
        v = np.array([r[a]["test"] for r in res]); pts = [r[a]["parts"] for r in res]
        print(f"{a:6s} | {np.median(v):.4f} | {np.median(v / w):.2f} | {np.median(v / o):.2f} | {np.median([r[a]['preq'] for r in res]):.4f} | {np.median(pts):.0f} ({min(pts)}-{max(pts)}) | {np.mean([r[a]['compute'] for r in res]) / np.mean([r['wide']['compute'] for r in res]):.2f}")
    print("done", round(time.time() - t0), "s")
