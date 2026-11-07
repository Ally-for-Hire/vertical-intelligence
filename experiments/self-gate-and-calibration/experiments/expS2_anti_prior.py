"""Exp S2 (queue Q10b): structured randomness where the learned prior is WRONG. Family 'pure_interaction': each core output changes with the PRODUCT of exactly two parents (no marginal
association for lag-1 descriptors to see). 48 worlds. Arms (MLP per-output framework, gain over plain in ticks 300-1000): oracle mask; single prior trained on STANDARD worlds (blind to interactions);
structured population (K=8, sigma=1 perturbations of that prior, causal selection by the lowest error in ticks [100, 300)); unstructured population (random gates); an IN-FAMILY prior (trained on 100
pure_interaction scenarios with own labels). Reports whether the standard prior hurts here and whether diversity rescues it."""
import os, sys, pickle, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool
import expS_diversity as S

T, W, HID, LR, K, SEL = S.T, S.W, S.HID, S.LR, S.K, S.SEL


def build(seed, fam):
    from worlds import sample_world_family, rollout
    w = sample_world_family(seed, fam)
    X, Y = rollout(w, T, np.random.default_rng(seed + 7))
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    return X, Y, cp, w.truth()[cp], w.n_act


def train_job(args):
    seed, fam = args
    from selfnn import run_scenario, perm_labels
    X, Y, cp, truth, na = build(seed, fam)
    r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "plain")
    lab = perm_labels(r["nets"], r["Xs"], r["Ds"], seed=seed)
    return r["desc"].reshape(-1, r["desc"].shape[-1]), lab.reshape(-1), float(lab.mean()), float(truth.mean())


def world_job(args):
    seed, base, infam = args
    from selfnn import run_scenario
    X, Y, cp, truth, na = build(seed, "pure_interaction")
    out = dict(seed=seed)
    out["plain"] = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "plain")["err"]
    out["oracle"] = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "oracle", truth_mask=truth)["err"]
    out["single"] = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "prior", prior=base)["err"]
    out["infamily"] = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "prior", prior=infam)["err"]
    for kind in ("structured", "unstructured"):
        errs = np.array([run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "prior", prior=S.Perturbed(base, kind, seed * 100 + v))["err"] for v in range(K)])
        out[kind] = errs[int(np.argmin(errs[:, W:W + SEL].mean(1)))]
        out[kind + "_oracle"] = errs[int(np.argmin(errs[:, W + SEL:1000].mean(1)))]
    return out


if __name__ == "__main__":
    from selfnn import NNPrior
    t0 = time.time()
    with Pool(20) as p:
        std = p.map(train_job, [(s, "standard") for s in range(5000, 5100)])
        fam = p.map(train_job, [(s, "pure_interaction") for s in range(5000, 5100)])
        base = NNPrior(seed=1); base.fit(np.concatenate([t[0] for t in std]), np.concatenate([t[1] for t in std]))
        infam = NNPrior(seed=2); infam.fit(np.concatenate([t[0] for t in fam]), np.concatenate([t[1] for t in fam]))
        print("label positive rate: standard worlds %.3f (true-parent rate %.3f), pure-interaction worlds %.3f (true-parent rate %.3f)" % (
            np.mean([t[2] for t in std]), np.mean([t[3] for t in std]), np.mean([t[2] for t in fam]), np.mean([t[3] for t in fam])), flush=True)
        res = p.map(world_job, [(s, base, infam) for s in range(9000, 9048)])
    pickle.dump(res, open("../results/expS2.pkl", "wb"))
    sl = slice(W + SEL, 1000)
    pl = np.array([np.mean(x["plain"][sl]) for x in res])
    print("pure_interaction worlds, 48: plain error in ticks 300-1000: %.3f" % pl.mean())
    for key in ("oracle", "single", "infamily", "structured", "structured_oracle", "unstructured", "unstructured_oracle"):
        v = np.array([np.mean(x[key][sl]) for x in res]); d = pl - v
        print(f"  {key:20s} gain {d.mean():+.3f} +- {d.std(ddof=1)/np.sqrt(len(d)):.3f}")
