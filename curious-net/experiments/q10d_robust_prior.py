"""Q10d: is randomness useful as INSURANCE against a wrong prior? Same pure_interaction worlds and pair test as q10c (m = 200 proposals per output, 500 test ticks). The learned prior p is corrupted:
p' = (1 - lam) * p + lam * U(0,1), lam in {0, 0.5, 0.9, 1.0} (lam = 1: the prior carries no information). Strategies: uniform random pairs; prior-weighted random sampling; deterministic top-k (all pairs among the t highest
p' inputs); epsilon-mix (80% of the budget deterministic top-k, 20% uniform random pairs). Success = the true pair is the best proposal and passes Bonferroni (alpha 0.01)."""
import os, sys, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
from scipy.stats import norm
import expS2_anti_prior as S2

NTEST, M = 500, 200
LAMS = (0.0, 0.5, 0.9, 1.0, "anti")
STRATS = ("uniform", "prior_sample", "topk", "eps_mix")


def score(pairs, Xs, d, true_pair):
    best, bz = None, -1.0
    for (a, b) in pairs:
        z = abs(np.mean(Xs[:, a] * Xs[:, b] * d)) * np.sqrt(NTEST)
        if z > bz:
            best, bz = (a, b), z
    zc = norm.isf(0.01 / (2 * max(len(pairs), 1)))
    return best == true_pair and bz > zc


def topk_pairs(cand, pj, m):
    t = 2
    while t * (t - 1) // 2 < m:
        t += 1
    top = cand[np.argsort(-pj)[:t]]
    return [(min(a, b), max(a, b)) for ai, a in enumerate(top) for b in top[ai + 1:]][:m]


def rand_pairs(cand, w, m, rng):
    w = w / w.sum()
    pairs, tries = set(), 0
    while len(pairs) < m and tries < 20 * m:
        a, b = rng.choice(cand, 2, replace=False, p=w)
        pairs.add((min(a, b), max(a, b)))
        tries += 1
    return list(pairs)


def world_job(args):
    seed, base = args
    from worlds import sample_world_family, rollout
    from selfnn import run_scenario
    w = sample_world_family(seed, "pure_interaction")
    X, Y = rollout(w, 1500, np.random.default_rng(seed + 7))
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    truth = w.truth()[cp]
    na = w.n_act
    r = run_scenario(X, Y, cp, na, 300, S2.HID, S2.LR, S2.W, seed, "plain")
    desc = r["desc"]
    n_out, n_in = truth.shape
    p0 = base.prob(desc.reshape(-1, desc.shape[-1])).reshape(n_out, n_in)
    D = Y - X[:, :w.n_state]
    rng = np.random.default_rng(seed)
    Xs = (X[:NTEST] - X[:NTEST].mean(0)) / (X[:NTEST].std(0) + 1e-9)
    out = {(lam, s): [] for lam in LAMS for s in STRATS}
    for j in range(n_out):
        par = [i for i in np.nonzero(truth[j])[0] if i != cp[j]]
        if len(par) != 2:
            continue
        true_pair = tuple(sorted(par))
        cand = np.array([i for i in range(n_in) if i != cp[j]])
        d = D[:NTEST, cp[j]]; d = (d - d.mean()) / (d.std() + 1e-9)
        for lam in LAMS:
            if lam == "anti":                                # confidently BACKWARDS prior: the inverted ranking
                pj = (1.0 - p0[j])[cand]
            else:
                pj = ((1 - lam) * p0[j] + lam * rng.uniform(size=n_in))[cand]
            out[(lam, "uniform")].append(score(rand_pairs(cand, np.ones(len(cand)), M, rng), Xs, d, true_pair))
            out[(lam, "prior_sample")].append(score(rand_pairs(cand, pj + 1e-6, M, rng), Xs, d, true_pair))
            out[(lam, "topk")].append(score(topk_pairs(cand, pj, M), Xs, d, true_pair))
            mix = topk_pairs(cand, pj, int(0.8 * M)) + rand_pairs(cand, np.ones(len(cand)), int(0.2 * M), rng)
            out[(lam, "eps_mix")].append(score(list(dict.fromkeys(mix)), Xs, d, true_pair))
    return out


if __name__ == "__main__":
    from selfnn import NNPrior
    t0 = time.time()
    with Pool(20) as p:
        std = p.map(S2.train_job, [(s, "standard") for s in range(5000, 5100)])
        base = NNPrior(seed=1)
        base.fit(np.concatenate([t[0] for t in std]), np.concatenate([t[1] for t in std]))
        res = p.map(world_job, [(s, base) for s in range(9000, 9096)])
    print(f"success rate with {M} proposals per output, 96 worlds, by prior corruption lam (0 = learned prior as is, 1 = prior carries no information)")
    print("lam    " + "  ".join(f"{s:>12s}" for s in STRATS))
    for lam in LAMS:
        row = []
        for s in STRATS:
            v = np.concatenate([r[(lam, s)] for r in res]).astype(float)
            row.append(f"{v.mean():.3f}±{v.std(ddof=1)/np.sqrt(len(v)):.3f}")
        print(f"{str(lam):>5s}  " + "  ".join(f"{c:>12s}" for c in row))
    print("done", round(time.time() - t0), "s")
