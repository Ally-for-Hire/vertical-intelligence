"""Exp S (queue Q10): 'emotion as structured randomness'. A population of K=8 explorers whose relevance gates are random perturbations of the learned prior (STRUCTURED: p * exp(sigma xi),
sigma = 1, xi standard normal per pair) or arbitrary random gates (UNSTRUCTURED: p ~ U(0,1) independent of the prior), each learning the same stream; selection is causal: choose the explorer with the
lowest prequential error over ticks [W, W+200) (the others are dropped) and score only ticks 300..1000 (after selection). Compared with: the single learner using the prior as is (K=1) and the plain
net, on four families (standard, many_parents_4_6, delayed_effects, fly_like), 24 worlds each. Also reported: the oracle-selection upper bound (best explorer by the SAME scored window) and the compute multiplier
(selection costs K x training for the first 200 ticks after W; after that one learner). MLP per-output framework. Experiment settings: K=8, sigma=1, selection window 200 (EXPERIMENT status)."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

T, W, HID, LR = 1500, 100, 32, 3e-3
K, SIGMA, SEL = 8, 1.0, 200
FAMS = ["standard", "many_parents_4_6", "delayed_effects", "fly_like"]


class Perturbed:
    windows = False
    rule, tau, y_min_p, w = "soft", 0.5, 0.0, True

    def __init__(self, base, kind, seed):
        self.base, self.kind, self.rng = base, kind, np.random.default_rng(seed)

    def prob(self, F):
        p = self.base.prob(F)
        if self.kind == "structured":
            return np.clip(p * np.exp(SIGMA * self.rng.normal(size=p.shape)), 1e-3, 1.0)
        return self.rng.uniform(0.0, 1.0, size=p.shape)


def build(seed, fam):
    from worlds import sample_world_family, rollout
    w = sample_world_family(seed, fam)
    X, Y = rollout(w, T, np.random.default_rng(seed + 7))
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    return X, Y, cp, w.n_act


def train_job(seed):
    from selfnn import run_scenario, perm_labels
    X, Y, cp, na = build(seed, "standard")
    r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "plain")
    lab = perm_labels(r["nets"], r["Xs"], r["Ds"], seed=seed)
    return r["desc"].reshape(-1, r["desc"].shape[-1]), lab.reshape(-1)


def world_job(args):
    seed, fam, base = args
    from selfnn import run_scenario
    X, Y, cp, na = build(seed, fam)
    out = dict(seed=seed, fam=fam)
    plain = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "plain")["err"]
    single = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "prior", prior=base)["err"]
    out["plain"], out["single"] = plain, single
    for kind in ("structured", "unstructured"):
        errs = [run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "prior", prior=Perturbed(base, kind, seed * 100 + v))["err"] for v in range(K)]
        errs = np.array(errs)
        sel_scores = errs[:, W:W + SEL].mean(1)
        chosen = int(np.argmin(sel_scores))
        out[kind] = dict(chosen=errs[chosen], oracle=errs[np.argmin(errs[:, W + SEL:1000].mean(1))], mean=errs.mean(0))
    return out


if __name__ == "__main__":
    from selfnn import NNPrior
    t0 = time.time()
    with Pool(20) as p:
        trn = p.map(train_job, range(5000, 5100))
        base = NNPrior(seed=100)
        base.fit(np.concatenate([t[0] for t in trn]), np.concatenate([t[1] for t in trn]))
        print("prior fitted", round(time.time() - t0), "s", flush=True)
        res = p.map(world_job, [(s, f, base) for f in FAMS for s in range(9000, 9024)])
    import pickle
    pickle.dump(res, open("../results/expS.pkl", "wb"))
    sl = slice(W + SEL, 1000)
    print("gain over plain in ticks 300-1000 (after selection), mean over 24 worlds per family; selection cost = K x compute during ticks 100-300")
    print(f"{'family':20s} {'single prior':>13s} {'structured pop':>15s} {'unstructured pop':>17s} {'struct. oracle-sel':>19s} {'mean of struct. members':>24s}")
    for f in FAMS:
        r = [x for x in res if x["fam"] == f]
        pl = np.array([np.mean(x["plain"][sl]) for x in r])
        def g(key, sub=None):
            v = np.array([np.mean((x[key] if sub is None else x[key][sub])[sl]) for x in r])
            return (pl - v).mean(), (pl - v).std(ddof=1) / np.sqrt(len(v))
        a, b, c, d, e = g("single"), g("structured", "chosen"), g("unstructured", "chosen"), g("structured", "oracle"), g("structured", "mean")
        print(f"{f:20s} {a[0]:+.3f}±{a[1]:.3f}  {b[0]:+.3f}±{b[1]:.3f}   {c[0]:+.3f}±{c[1]:.3f}    {d[0]:+.3f}±{d[1]:.3f}    {e[0]:+.3f}±{e[1]:.3f}")
