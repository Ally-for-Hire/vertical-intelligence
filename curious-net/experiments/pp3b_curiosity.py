"""PIECE-P3b: follow-up to P3. Does a better progress estimate make learning-progress curiosity beat random? Variants of the progress signal:
  lp10      : last 10 vs previous 10 samples (P3's version)
  lp30      : last 30 vs previous 30
  lp30abs   : |change| over 30 vs 30 (Oudeyer's absolute learning progress: also attends to activities that got WORSE, i.e. changed)
Conditions: budget 600 or 2400 picks; 2 noisy TVs of 6 activities (as P3) or 4 noisy TVs of 8 (2 easy, 2 hard, 4 noise). 200 seeds each. Score as P3 (median error ratio to random on learnable activities).
PREDICTIONS (registered before running): P48 lp30abs reaches <= 1.0 x random at budget 600 (2 of 6 noise); P49 at budget 2400 the best progress variant is <= 0.9 x random (2 of 6 noise);
P50 with 4 of 8 noisy TVs, the best progress variant beats random (<= 0.9 x) at BOTH budgets (more traps = more value in avoiding them)."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

WARM, EPS = 20, 0.1
VARIANTS = ("random", "lp10", "lp30", "lp30abs")


def make(rng, n_noise):
    acts = []
    for k in range(2):
        a, b = rng.uniform(-1.5, 1.5, 2); acts.append(("easy", lambda x, a=a, b=b: a * x + b, 0.1))
    for k in range(2):
        w, c = rng.uniform(1.5, 3.0), rng.uniform(-0.5, 0.5); acts.append(("hard", lambda x, w=w, c=c: np.sin(w * x) + c * x * x, 0.1))
    for k in range(n_noise):
        acts.append(("noise", lambda x: 0.0 * x, 3.0))
    return acts


def run(args):
    seed, budget, n_noise = args
    from nn import AdamMLP
    out = {}
    for v in VARIANTS:
        rng = np.random.default_rng(seed)
        acts = make(rng, n_noise); na = len(acts)
        nets = [AdamMLP(1, 1, 16, 3e-3, seed * 10 + k) for k in range(na)]
        hist = [[] for _ in range(na)]; picks = np.zeros(na, int); srng = np.random.default_rng(seed + 12345)
        win = 10 if v == "lp10" else 30
        def sample(k):
            x = rng.uniform(-2, 2); kind, f, sd = acts[k]
            y = f(x) + sd * rng.normal()
            p = nets[k].step(np.array([x]), np.array([y]))[0]
            hist[k].append((p - y) ** 2); picks[k] += 1
        for r in range(max(WARM, 2 * win)):
            for k in range(na):
                sample(k)
        picks[:] = 0
        for t in range(budget):
            if v == "random" or srng.random() < EPS:
                k = int(srng.integers(na))
            else:
                lp = [np.mean(h[-2 * win:-win]) - np.mean(h[-win:]) for h in hist]
                k = int(np.argmax(np.abs(lp) if v == "lp30abs" else lp))
            sample(k)
        xt = np.linspace(-2, 2, 200); err = []
        for k in range(4):
            pr = np.array([nets[k].predict(np.array([x]))[0][0] for x in xt]); err.append(np.mean((pr - acts[k][1](xt)) ** 2))
        out[v] = dict(err=float(np.mean(err)), noise_share=float(picks[4:].sum() / budget), hard_share=float(picks[2:4].sum() / budget))
    return out


if __name__ == "__main__":
    t0 = time.time()
    allres = {}
    with Pool(20) as p:
        for budget in (600, 2400):
            for n_noise in (2, 4):
                res = p.map(run, [(s, budget, n_noise) for s in range(200)])
                allres[f"{budget}_{n_noise}"] = res
                base = np.array([r["random"]["err"] for r in res])
                print(f"budget {budget}, {n_noise} noisy TVs of {4 + n_noise}:")
                for v in VARIANTS:
                    ratio = np.array([r[v]["err"] for r in res]) / base
                    print(f"   {v:8s} noise {np.mean([r[v]['noise_share'] for r in res]):.3f} hard {np.mean([r[v]['hard_share'] for r in res]):.3f} | error ratio to random: median {np.median(ratio):.3f}")
    json.dump(allres, open("../results/pp3b_curiosity.json", "w"))
    print("done", round(time.time() - t0), "s")
