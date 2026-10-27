"""PIECE-P3c: EPISTEMIC curiosity ('study where more data would change my mind') vs random and learning progress, with noisy TVs.
Every strategy uses the SAME learner per activity: an ensemble of 5 small nets (1 -> 16 -> 1, Adam), each trained on a sample with probability 0.5 (online bootstrap, for diversity); the prediction = ensemble mean.
Only the choice rule differs: random; lp30 (learning progress of the ensemble-mean error, 30 vs 30 samples); epistemic (largest ensemble disagreement = variance of the 5 predictions averaged over 20 probe x values).
10% random picks for the two curious rules (EXPERIMENT setting). Conditions as P3b: budget 600 / 2400; 2 of 6 or 4 of 8 noisy TVs. 100 seeds per cell. Score = final ensemble-mean error on the 4 learnable activities, median ratio to random.
PREDICTIONS (registered before running): P51 epistemic spends <= 25% of picks on noisy TVs (budget 600, 2 of 6); P52 epistemic ends <= 0.9 x random (budget 600, 2 of 6); P53 with 4 of 8 noisy TVs at budget 2400, epistemic ends <= 0.8 x random."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool
import pp3b_curiosity as B

ENS, EPS, PROBE = 5, 0.1, np.linspace(-2, 2, 20)
RULES = ("random", "motivated", "motivated2") if os.environ.get("P4B") else ("random", "lp30", "epistemic", "infogain", "motivated") if os.environ.get("P4M") else ("random", "lp30", "epistemic", "infogain")


def run(args):
    seed, budget, n_noise = args
    from nn import AdamMLP
    out = {}
    for rule in RULES:
        rng = np.random.default_rng(seed)
        acts = B.make(rng, n_noise); na = len(acts)
        ens = [[AdamMLP(1, 1, 16, 3e-3, seed * 100 + 10 * k + m) for m in range(ENS)] for k in range(na)]
        hist = [[] for _ in range(na)]; picks = np.zeros(na, int); srng = np.random.default_rng(seed + 12345); brng = np.random.default_rng(seed + 999)
        def pred(k, x):
            return np.array([n.predict(np.array([x]))[0][0] for n in ens[k]])
        def sample(k):
            x = rng.uniform(-2, 2); kind, f, sd = acts[k]
            y = f(x) + sd * rng.normal()
            p = pred(k, x).mean()
            for n in ens[k]:
                if brng.random() < 0.5:
                    n.step(np.array([x]), np.array([y]))
            hist[k].append((p - y) ** 2); picks[k] += 1
        for r in range(60):
            for k in range(na):
                sample(k)
        picks[:] = 0
        for t in range(budget):
            if rule == "random" or srng.random() < EPS:
                k = int(srng.integers(na))
            elif rule == "lp30":
                k = int(np.argmax([np.mean(h[-60:-30]) - np.mean(h[-30:]) for h in hist]))
            elif rule == "epistemic":
                k = int(np.argmax([np.mean([pred(q, x).var() for x in PROBE]) for q in range(na)]))
            elif rule == "motivated2":                                   # P4b: keep activities whose information gain is >= 25% of the best one's
                ig, err = [], []
                for q in range(na):
                    dis = np.mean([pred(q, x).var() for x in PROBE]); nv = max(np.mean(hist[q][-30:]) - dis, 1e-6)
                    ig.append(0.5 * np.log1p(dis / nv)); err.append(np.mean(hist[q][-30:]) - nv)
                keep = [q for q in range(na) if ig[q] >= 0.25 * max(ig)]
                k = int(max(keep, key=lambda q: err[q]))
            elif rule == "motivated":                                    # P4: drop the unlearnable (information gain below the median), then work on the worst remaining activity
                ig, err = [], []
                for q in range(na):
                    dis = np.mean([pred(q, x).var() for x in PROBE]); nv = max(np.mean(hist[q][-30:]) - dis, 1e-6)
                    ig.append(0.5 * np.log1p(dis / nv)); err.append(np.mean(hist[q][-30:]) - nv)
                keep = [q for q in range(na) if ig[q] >= np.median(ig)]
                k = int(max(keep, key=lambda q: err[q]))
            else:                                                        # P3d: expected information gain ~ 0.5 log(1 + model uncertainty / noise variance)
                ig = []
                for q in range(na):
                    dis = np.mean([pred(q, x).var() for x in PROBE])
                    nv = max(np.mean(hist[q][-30:]) - dis, 1e-6)        # residual error of the ensemble mean minus its own uncertainty = estimated noise variance
                    ig.append(0.5 * np.log1p(dis / nv))
                k = int(np.argmax(ig))
            sample(k)
        xt = np.linspace(-2, 2, 200); err = []
        for k in range(4):
            pr = np.array([pred(k, x).mean() for x in xt]); err.append(np.mean((pr - acts[k][1](xt)) ** 2))
        out[rule] = dict(err=float(np.mean(err)), noise_share=float(picks[4:].sum() / budget), hard_share=float(picks[2:4].sum() / budget))
    return out


if __name__ == "__main__":
    t0 = time.time()
    allres = {}
    with Pool(20) as p:
        for budget in (600, 2400):
            for n_noise in (2, 4):
                res = p.map(run, [(s, budget, n_noise) for s in (range(100, 200) if os.environ.get("P4B") else range(100))])
                allres[f"{budget}_{n_noise}"] = res
                base = np.array([r["random"]["err"] for r in res])
                print(f"budget {budget}, {n_noise} noisy TVs of {4 + n_noise}:")
                for v in RULES:
                    ratio = np.array([r[v]["err"] for r in res]) / base
                    print(f"   {v:9s} noise {np.mean([r[v]['noise_share'] for r in res]):.3f} hard {np.mean([r[v]['hard_share'] for r in res]):.3f} | error ratio to random: median {np.median(ratio):.3f}")
    json.dump(allres, open("../results/pp4b_motivation.json" if os.environ.get("P4B") else "../results/pp4_motivation.json" if os.environ.get("P4M") else "../results/pp3d_infogain.json", "w"))
    print("done", round(time.time() - t0), "s")
