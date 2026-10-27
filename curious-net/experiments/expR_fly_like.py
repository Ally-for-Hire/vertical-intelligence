"""Exp R (queue Q22): fly-like wiring. Worlds whose core outputs have ~4-16 inputs (median 8) with power-law strengths (top 3 hold ~half the weight), 10-18 core variables,
30-200 distractors. Questions: (1) how much headroom does knowing ALL true inputs give, and how much does knowing only the 3 strongest give (what sparse recovery can realistically
find)? (2) does a prior trained on the standard worlds still help? (3) does a prior trained IN-family (fly-like, K=100) help more? MLP per-output framework, 48 worlds (seeds 9000-9047),
metric: gain over plain in first-1000-tick error. Training scenarios for the in-family prior: seeds 5000-5099 generated as fly-like."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

T, W, HID, LR = 1500, 100, 32, 3e-3


def build(seed, fam):
    from worlds import sample_world_family, rollout
    w = sample_world_family(seed, fam)
    X, Y = rollout(w, T, np.random.default_rng(seed + 7))
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    return X, Y, cp, w.truth()[cp], w.truth_top(3)[cp], w.n_d, w.n_act


def train_job(args):
    seed, fam = args
    from selfnn import run_scenario, perm_labels
    X, Y, cp, truth, top3, nd, na = build(seed, fam)
    r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "plain")
    lab = perm_labels(r["nets"], r["Xs"], r["Ds"], seed=seed)
    return r["desc"].reshape(-1, r["desc"].shape[-1]), lab.reshape(-1)


def test_job(args):
    seed, specs = args
    from selfnn import run_scenario
    X, Y, cp, truth, top3, nd, na = build(seed, "fly_like")
    out = dict(seed=seed, nd=nd, n_parents=float(truth.sum(1).mean()), arms={})
    for name, mode, prior, mask in specs:
        m = {"all": truth, "top3": top3}.get(mask)
        r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, mode, prior=prior, truth_mask=m)
        out["arms"][name] = r["err"].tolist()
    return out


if __name__ == "__main__":
    from selfnn import NNPrior
    t0 = time.time()
    with Pool(20) as p:
        std = p.map(train_job, [(s, "standard") for s in range(5000, 5100)])
        fly = p.map(train_job, [(s, "fly_like") for s in range(5000, 5100)])
        pri_std = NNPrior(seed=1); pri_std.fit(np.concatenate([t[0] for t in std]), np.concatenate([t[1] for t in std]))
        pri_fly = NNPrior(seed=2); pri_fly.fit(np.concatenate([t[0] for t in fly]), np.concatenate([t[1] for t in fly]))
        print("priors fitted", round(time.time() - t0), "s", flush=True)
        specs = [("plain", "plain", None, None), ("oracle_all", "oracle", None, "all"), ("oracle_top3", "oracle", None, "top3"), ("screen", "screen", None, None),
                 ("prior_std", "prior", pri_std, None), ("prior_fly", "prior", pri_fly, None)]
        te = p.map(test_job, [(s, specs) for s in range(9000, 9048)])
    json.dump(dict(test=te), open("../results/expR.json", "w"), default=float)
    print("done", round(time.time() - t0), "s")
