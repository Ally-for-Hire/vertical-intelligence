"""Q10c: 'emotion proposes new connections, the intellectual side tests them'. In pure_interaction worlds each core output changes with the product of exactly two parents (invisible to marginal association).
The test of a proposed pair (a, b) = z-statistic of the correlation between the product feature x_a * x_b and the output's next change over the first 500 ticks; the pair is 'found' if the true pair is the
best-scoring proposal and passes a Bonferroni threshold over the proposals made (alpha 0.01). Strategies (budget m proposals per output): uniform random pairs; prior-weighted pairs (inputs drawn with
probability proportional to the learned relevance prior p_ji^1 or p_ji^0.5, a 'tempered' version); exhaustive (all n(n-1)/2 pairs). Reported: whether the prior even ranks the true parents highly
(AUROC), success rate vs m, and the cost of exhaustive search."""
import os, sys, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
from scipy.stats import norm
import expS2_anti_prior as S2

NTEST = 500


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
    p = base.prob(desc.reshape(-1, desc.shape[-1])).reshape(n_out, n_in)
    ns = w.n_state
    D = Y - X[:, :ns]
    rng = np.random.default_rng(seed)
    Xs = (X[:NTEST] - X[:NTEST].mean(0)) / (X[:NTEST].std(0) + 1e-9)
    out = dict(auc=[], succ={}, exh_pairs=n_in * (n_in - 1) // 2, n_in=n_in)
    budgets = (50, 200, 1000, 5000)
    for m in budgets:
        for s in ("uniform", "prior", "prior_sqrt", "topk"):
            out["succ"][(s, m)] = []
    for j in range(n_out):
        # true product parents of this output = the two parents that are not its own variable
        par = [i for i in np.nonzero(truth[j])[0] if i != cp[j]]
        if len(par) != 2:
            continue
        true_pair = tuple(sorted(par))
        # does the prior rank the true parents? (AUROC of p over the non-self inputs)
        cand = np.array([i for i in range(n_in) if i != cp[j]])
        lab = np.isin(cand, par)
        pj = p[j, cand]
        ranks = pj.argsort().argsort()
        auc = (ranks[lab].mean() - (lab.sum() - 1) / 2) / (len(cand) - lab.sum())
        out["auc"].append(float(auc))
        d = D[:NTEST, cp[j]]
        d = (d - d.mean()) / (d.std() + 1e-9)
        for m in budgets:
            for s in ("uniform", "prior", "prior_sqrt", "topk"):
                if s == "topk":                                   # deterministic: all pairs among the t highest-prior inputs, t(t-1)/2 >= m
                    t = 2
                    while t * (t - 1) // 2 < m:
                        t += 1
                    top = cand[np.argsort(-pj)[:t]]
                    pairs = [(min(a, b), max(a, b)) for ai, a in enumerate(top) for b in top[ai + 1:]][:m]
                    best, bz = None, -1.0
                    for (a, b) in pairs:
                        z = abs(np.mean(Xs[:, a] * Xs[:, b] * d)) * np.sqrt(NTEST)
                        if z > bz:
                            best, bz = (a, b), z
                    zc = norm.isf(0.01 / (2 * max(len(pairs), 1)))
                    out["succ"][(s, m)].append(best == true_pair and bz > zc)
                    continue
                if s == "uniform":
                    w_i = np.ones(len(cand))
                elif s == "prior":
                    w_i = pj + 1e-6
                else:
                    w_i = np.sqrt(pj + 1e-6)
                w_i = w_i / w_i.sum()
                pairs = set()
                tries = 0
                while len(pairs) < m and tries < 20 * m:
                    a, b = rng.choice(cand, 2, replace=False, p=w_i)
                    pairs.add((min(a, b), max(a, b)))
                    tries += 1
                pairs = list(pairs) + [true_pair] * 0
                best, bz = None, -1.0
                for (a, b) in pairs:
                    z = abs(np.mean(Xs[:, a] * Xs[:, b] * d)) * np.sqrt(NTEST)
                    if z > bz:
                        best, bz = (a, b), z
                zc = norm.isf(0.01 / (2 * max(len(pairs), 1)))
                out["succ"][(s, m)].append(best == true_pair and bz > zc)
    return out


if __name__ == "__main__":
    from selfnn import NNPrior
    t0 = time.time()
    with Pool(20) as p:
        std = p.map(S2.train_job, [(s, "standard") for s in range(5000, 5100)])
        base = NNPrior(seed=1)
        base.fit(np.concatenate([t[0] for t in std]), np.concatenate([t[1] for t in std]))
        res = p.map(world_job, [(s, base) for s in range(9000, 9048)])
    auc = np.concatenate([r["auc"] for r in res])
    print("prior ranking of the true product parents among non-self inputs: AUROC %.3f +- %.3f (0.5 = blind) over %d outputs" % (auc.mean(), auc.std(ddof=1) / np.sqrt(len(auc)), len(auc)))
    n_in = np.mean([r["n_in"] for r in res]); ex = np.mean([r["exh_pairs"] for r in res])
    print("mean inputs per world %.0f; exhaustive pair search = %.0f proposals per output" % (n_in, ex))
    print("success rate (true pair is the best proposal and passes Bonferroni) by proposals per output:")
    for s in ("uniform", "prior", "prior_sqrt", "topk"):
        row = []
        for m in (50, 200, 1000, 5000):
            v = np.concatenate([r["succ"][(s, m)] for r in res]).astype(float)
            row.append(f"m={m}: {v.mean():.2f}")
        print(f"  {s:11s} " + "  ".join(row))
    print("done", round(time.time() - t0), "s")
