"""PIECE-P3: curiosity as LEARNING PROGRESS vs curiosity as UNCERTAINTY vs random, with a 'noisy TV' (Oudeyer & Kaplan 2007; Schmidhuber 2010; Burda et al. 2019 noisy-TV problem [u]).
The agent studies 6 sub-problems ('activities'). Each tick it picks one, samples x ~ U(-2, 2), observes y, and trains that activity's own small online network (1 -> 16 -> 1, Adam).
Activities: 2 easy (linear, noise 0.1), 2 hard (sin(w x) + c x^2, noise 0.1), 2 NOISY TVs (y = pure noise, sd 3: nothing to learn). Budget 600 picks after a 20-pick warm-up per activity.
Strategies: random; round-robin; uncertainty (pick the activity with the highest recent error: 'what do I understand least?'); learning progress (pick the activity whose recent error is falling fastest:
'where am I improving?', error of the last 10 vs the previous 10 samples); both greedy with 10% random picks (an EXPERIMENT setting).
Score: final test error on the 4 LEARNABLE activities (noise-free targets, 200 test points each), relative to random. 200 seeds.
PREDICTIONS (registered before running): P44 uncertainty spends >= 40% of picks on the noisy TVs; P45 learning progress spends <= 25% (uniform = 33%); P46 learning progress ends with learnable error <= 0.9 x random;
P47 uncertainty ends with learnable error >= 1.1 x random."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

NA, WARM, BUDGET, WIN, EPS = 6, 20, 600, 10, 0.1
STRATS = ("random", "roundrobin", "uncertainty", "progress")


def make(rng):
    acts = []
    for k in range(2):
        a, b = rng.uniform(-1.5, 1.5, 2); acts.append(("easy", lambda x, a=a, b=b: a * x + b, 0.1))
    for k in range(2):
        w, c = rng.uniform(1.5, 3.0), rng.uniform(-0.5, 0.5); acts.append(("hard", lambda x, w=w, c=c: np.sin(w * x) + c * x * x, 0.1))
    for k in range(2):
        acts.append(("noise", lambda x: 0.0 * x, 3.0))
    return acts


def run(seed):
    from nn import AdamMLP
    out = {}
    for strat in STRATS:
        rng = np.random.default_rng(seed)
        acts = make(rng)
        nets = [AdamMLP(1, 1, 16, 3e-3, seed * 10 + k) for k in range(NA)]
        hist = [[] for _ in range(NA)]; picks = np.zeros(NA, int)
        srng = np.random.default_rng(seed + 12345)
        def sample(k):
            x = rng.uniform(-2, 2); kind, f, sd = acts[k]
            y = f(x) + sd * rng.normal()
            p = nets[k].step(np.array([x]), np.array([y]))[0]
            hist[k].append((p - y) ** 2); picks[k] += 1
        for r in range(WARM):
            for k in range(NA):
                sample(k)
        picks[:] = 0
        for t in range(BUDGET):
            if strat == "random" or (strat in ("uncertainty", "progress") and srng.random() < EPS):
                k = int(srng.integers(NA))
            elif strat == "roundrobin":
                k = t % NA
            elif strat == "uncertainty":
                k = int(np.argmax([np.mean(h[-WIN:]) for h in hist]))
            else:
                k = int(np.argmax([np.mean(h[-2 * WIN:-WIN]) - np.mean(h[-WIN:]) for h in hist]))
            sample(k)
        xt = np.linspace(-2, 2, 200)
        err = []
        for k in range(4):
            f = acts[k][1]
            pr = np.array([nets[k].predict(np.array([x]))[0][0] for x in xt])
            err.append(np.mean((pr - f(xt)) ** 2))
        out[strat] = dict(err=float(np.mean(err)), noise_share=float(picks[4:].sum() / BUDGET), hard_share=float(picks[2:4].sum() / BUDGET))
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(200))
    json.dump(res, open("../results/pp3_curiosity.json", "w"))
    base = np.array([r["random"]["err"] for r in res])
    print("strategy     | share of picks on noisy TVs | on hard activities | final learnable error (median ratio to random; mean ratio +- SE)")
    for s in STRATS:
        e = np.array([r[s]["err"] for r in res]); ratio = e / base
        print(f"{s:12s} | {np.mean([r[s]['noise_share'] for r in res]):.3f} | {np.mean([r[s]['hard_share'] for r in res]):.3f} | {np.median(ratio):.3f}; {ratio.mean():.3f} +- {ratio.std(ddof=1) / np.sqrt(len(ratio)):.3f}")
    print("done", round(time.time() - t0), "s")
