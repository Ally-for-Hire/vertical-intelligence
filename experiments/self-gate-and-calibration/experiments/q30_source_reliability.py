"""Q30: learn how much to trust each source (teacher claim, own learned prior, own experiment) with Dawid-Skene EM, instead of doubting by default.
Stage 1 (parallel): per world, cache the ground truth of every (action, output) claim, the prior's belief p, and the verdict of an own experiment on every claim (so a student can 'request' it
later; the cost is counted when it does). Teacher claims are generated as in q25b (honest / random 10% / systematic over- and under-connection).
Stage 2 (cheap, sequential, 30 random orderings of the 48 worlds): a student that keeps one Dawid-Skene model per teacher (binary latent truth, class prior, sensitivity and specificity for
each of the 3 sources, Beta(1,1) pseudo-counts), refit by EM on all claims seen so far. For each new world: posterior of every claim from teacher + prior; claims with posterior in
[0.1, 0.9] are tested by an own experiment (experiment-setting thresholds: marked ARBITRARY) and the verdict is fused; belief = posterior > 0.5; then the world's claims are added and EM refits.
Compared with: credulous, prior-only, and the q25b 'targeted' rule (experiment where teacher and prior disagree, verdict overrides)."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
from scipy.stats import norm
import q25b_fact_checking as Q


def cache_world(args):
    seed, prior = args
    w, X, Y, cp, truth, na = Q.build(seed)
    n_core = len(cp)
    n_in = X.shape[1]
    act_cols = list(range(n_in - na, n_in))
    T_true = truth[:, act_cols]
    from selfnn import run_scenario
    r = run_scenario(X, Y, cp, na, 300, Q.HID, Q.LR, Q.W, seed, "plain")
    p = prior.prob(r["desc"].reshape(-1, r["desc"].shape[-1])).reshape(n_core, n_in)[:, act_cols]
    rng = np.random.default_rng(seed)
    zc = norm.isf(1.0 / (2 * T_true.size))
    exp = np.zeros(T_true.shape, bool)
    for j in range(n_core):
        for a in range(na):
            exp[j, a] = Q.run_experiment(seed, a, cp[j], rng) > zc
    return dict(seed=seed, T=T_true, p=p, exp=exp)


def teacher_claims(T_true, kind, rng):
    claim = T_true.copy()
    if kind == "random":
        claim = np.where(rng.random(claim.shape) < 0.10, ~claim, claim)
    elif kind == "systematic":
        n_core, na = claim.shape
        claim[int(rng.integers(n_core)), :] = True
        claim[:, int(rng.integers(na))] = False
    return claim


class DS:
    """binary Dawid-Skene over sources S (columns); observations may be missing (nan). params: class prior pi, sens[s]=P(obs=1|z=1), spec[s]=P(obs=0|z=0)."""
    def __init__(self, n_src, init=0.7):
        self.pi = 0.5
        self.sens = np.full(n_src, init)                      # ASSUMPTION (flagged): sources start better than chance; this orients the latent labels
        self.spec = np.full(n_src, init)
        self.obs = []                                         # rows of length n_src with 0/1/nan

    def posterior(self, o):
        l1, l0 = np.log(self.pi), np.log(1 - self.pi)
        for s, v in enumerate(o):
            if np.isnan(v):
                continue
            l1 += np.log(self.sens[s] if v == 1 else 1 - self.sens[s])
            l0 += np.log(1 - self.spec[s] if v == 1 else self.spec[s])
        return 1.0 / (1.0 + np.exp(np.clip(l0 - l1, -50, 50)))

    def refit(self, iters=10):
        if not self.obs:
            return
        O = np.array(self.obs)
        for _ in range(iters):
            post = np.array([self.posterior(o) for o in O])
            self.pi = (post.sum() + 1) / (len(post) + 2)
            for s in range(O.shape[1]):
                m = ~np.isnan(O[:, s])
                o = O[m, s]; pz = post[m]
                self.sens[s] = ((pz * (o == 1)).sum() + 1) / (pz.sum() + 2)
                self.spec[s] = (((1 - pz) * (o == 0)).sum() + 1) / ((1 - pz).sum() + 2)
        if np.mean((self.sens + self.spec) / 2) < 0.5:        # label-swap guard: on average the sources are better than chance
            self.sens, self.spec = 1 - self.spec, 1 - self.sens
            self.pi = 1 - self.pi


def simulate(cache, kind, order, rng, init=0.7):
    ds = DS(3, init)                                                # sources: teacher, prior (p>=0.5), experiment
    res = dict(fused=[], credulous=[], prior_only=[], targeted=[], n_exp_fused=[], n_exp_targeted=[], claims=[])
    for idx in order:
        c = cache[idx]
        T_true, p, ex = c["T"], c["p"], c["exp"]
        claim = teacher_claims(T_true, kind, rng)
        flat = [(j, a) for j in range(T_true.shape[0]) for a in range(T_true.shape[1])]
        belief = np.zeros(T_true.shape, bool); nexp = 0; rows = []
        for j, a in flat:
            o = np.array([float(claim[j, a]), float(p[j, a] >= 0.5), np.nan])
            post = ds.posterior(o)
            if 0.1 <= post <= 0.9:                            # uncertain -> run an own experiment and fuse the verdict
                o[2] = float(ex[j, a]); nexp += 1
                post = ds.posterior(o)
            belief[j, a] = post > 0.5
            rows.append(o)
        ds.obs.extend(rows)
        ds.refit()
        n = T_true.size
        res["fused"].append(float((belief == T_true).mean())); res["n_exp_fused"].append(nexp)
        res["credulous"].append(float((claim == T_true).mean()))
        res["prior_only"].append(float(((p >= 0.5) == T_true).mean()))
        tb = claim.copy(); ne = 0
        for j, a in flat:
            if claim[j, a] != (p[j, a] >= 0.5):
                tb[j, a] = ex[j, a]; ne += 1
        res["targeted"].append(float((tb == T_true).mean())); res["n_exp_targeted"].append(ne)
        res["claims"].append(n)
    return res, ds


if __name__ == "__main__":
    from selfnn import NNPrior
    t0 = time.time()
    with Pool(20) as pool:
        trn = pool.map(Q.train_job, range(5000, 5100))
        prior = NNPrior(seed=100)
        prior.fit(np.concatenate([t[0] for t in trn]), np.concatenate([t[1] for t in trn]))
        cache = pool.map(cache_world, [(s, prior) for s in range(9000, 9048)])
    print("cache built", round(time.time() - t0), "s", flush=True)
    import pickle
    pickle.dump(cache, open("../results/q30_cache.pkl", "wb"))
    out = {}
    for kind in ("honest", "random", "systematic"):
        allr = []
        for rep in range(30):
            rng = np.random.default_rng(1000 + rep)
            order = rng.permutation(len(cache))
            r, ds = simulate(cache, kind, order, rng)
            allr.append(r)
        out[kind] = allr
        def mean(key, sl=slice(None)):
            return float(np.mean([np.mean(r[key][sl]) for r in allr]))
        print(f"-- teacher {kind}")
        for key, nk in (("credulous", None), ("prior_only", None), ("targeted", "n_exp_targeted"), ("fused", "n_exp_fused")):
            late = mean(key, slice(10, None))
            ne = mean(nk, slice(10, None)) if nk else 0.0
            print(f"   {key:11s} accuracy (worlds 11-48) {late:.3f}   experiments per world {ne:4.1f}   (all worlds: {mean(key):.3f})")
    json.dump(out, open("../results/q30.json", "w"), default=float)
    print("done", round(time.time() - t0), "s")
