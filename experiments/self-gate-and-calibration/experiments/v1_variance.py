"""RIGOR-1 (rigor phase, user 2026-10-22: 'refine and perfect every individual part based on as much data as we can, so we fully understand every result and can make theorems and predictions that hold up').
Question: how much of the learned gate's gain, and of its variation, comes from (a) which test world, (b) which earlier problems it learned from (training subset), (c) the prior network's random seed, (d) the amount of experience K?
Design: pool of 300 training worlds (seeds 5000-5299, own permutation labels, generator v2); for each K in {1,3,10,30,100} draw 5 random training subsets; for each subset fit the 2x32 prior with 2 seeds
-> 50 priors; evaluate every prior on 96 test worlds (seeds 9000-9095) against the plain net. Metric: gain in ticks 0-1000 (exp M convention). Variance components by a nested random-effects decomposition.
PREDICTIONS (stated before running): P1 the mean gain at K >= 10 is flat within +-0.01; P2 at K = 1 the between-subset spread is large enough that some subsets give a NEGATIVE mean gain; P3 prior-seed variance is
smaller than subset variance at every K; P4 world-to-world variance dominates every other component by an order of magnitude; P5 gain(K) fits g_inf - a K^-b with b in [0.5, 1.5]."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import expM_curriculum_scaling as M

KS = (1, 3, 10, 30, 100)
NSUB, NSEED = 5, 2
TRAIN = range(5000, 5300)
TEST = range(9000, 9096)


def test_job(args):
    seed, specs = args
    from selfnn import run_scenario
    X, Y, cp, truth, nd, na = M.build(seed)
    out = {}
    for name, mode, prior in specs:
        out[name] = float(run_scenario(X, Y, cp, na, M.T, M.HID, M.LR, M.W, seed, mode, prior=prior)["err"][:1000].mean())
    return seed, out


if __name__ == "__main__":
    from selfnn import NNPrior
    t0 = time.time()
    with Pool(20) as p:
        tr = p.map(M.train_job, TRAIN)
        print("training pool done", round(time.time() - t0), "s", flush=True)
        rng = np.random.default_rng(77)
        priors, meta = [], []
        for K in KS:
            for s in range(NSUB):
                idx = rng.choice(len(tr), K, replace=False)
                F = np.concatenate([tr[i]["desc"].reshape(-1, tr[i]["desc"].shape[-1]) for i in idx])
                y = np.concatenate([np.asarray(tr[i]["lab"], float).reshape(-1) for i in idx])
                for q in range(NSEED):
                    m = NNPrior(seed=1000 * K + 10 * s + q); m.fit(F, y)
                    priors.append((f"K{K}_s{s}_q{q}", "prior", m)); meta.append((K, s, q))
        specs = [("plain", "plain", None)] + priors
        res = p.map(test_job, [(s, specs) for s in TEST])
    names = [n for n, _, _ in priors]
    G = np.array([[r["plain"] - r[n] for n in names] for _, r in res])          # (worlds, priors)
    json.dump(dict(names=names, meta=meta, gain=G.tolist(), plain=[r["plain"] for _, r in res], seeds=[s for s, _ in res]), open("../results/v1_variance.json", "w"))
    meta = np.array(meta)
    print(f"{len(TEST)} test worlds x {len(names)} priors; done {round(time.time() - t0)} s")
    print("K  | mean gain +- SE(worlds) | subset means (avg over seeds and worlds) | seed diff (mean |q0-q1|) | var components: world, subset, seed, residual")
    out = {}
    for K in KS:
        cols = np.nonzero(meta[:, 0] == K)[0]
        g = G[:, cols]                                                          # worlds x (subset, seed)
        gs = g.reshape(len(TEST), NSUB, NSEED)
        world_means = gs.mean((1, 2))
        sub_means = gs.mean((0, 2))
        seed_diff = np.abs(gs[:, :, 0] - gs[:, :, 1]).mean(0).mean()
        grand = gs.mean()
        v_world = world_means.var(ddof=1)
        v_sub = sub_means.var(ddof=1)
        v_seed = (gs.mean(0)[:, 0] - gs.mean(0)[:, 1]).var(ddof=1) / 2 if NSEED == 2 else float("nan")
        resid = gs - world_means[:, None, None] - (sub_means - grand)[None, :, None]
        v_res = resid.var(ddof=1)
        se = world_means.std(ddof=1) / np.sqrt(len(TEST))
        out[K] = dict(mean=grand, se=se, sub=sub_means.tolist(), v=[v_world, v_sub, v_seed, v_res])
        print(f"{K:3d} | {grand:+.3f} +- {se:.3f} | " + " ".join(f"{x:+.3f}" for x in sub_means) + f" | {seed_diff:.3f} | {v_world:.5f} {v_sub:.5f} {v_seed:.5f} {v_res:.5f}")
    from scipy.optimize import curve_fit
    k = np.array(KS, float); m = np.array([out[K]["mean"] for K in KS]); s = np.array([max(out[K]["se"], 1e-4) for K in KS])
    try:
        (gi, a, b), cov = curve_fit(lambda x, gi, a, b: gi - a * x ** (-b), k, m, p0=[0.12, 0.05, 1.0], sigma=s, maxfev=20000)
        print(f"fit gain(K) = {gi:.3f} - {a:.3f} K^-{b:.2f}  (b SE {np.sqrt(cov[2, 2]):.2f})")
    except Exception as e:
        print("fit failed", e)
