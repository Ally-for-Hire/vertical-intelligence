"""Exp M: does the self-learning network get better with MORE experience? Diverse scenario distribution (4-10 core variables, 1-3 actions, 10-200 distractors of 4 kinds
including noisy COPIES of core variables, noise 0.02-0.2, 1-3 parents). Training scenarios K in {1,3,10,30,100,300}; 48 held-out test scenarios.
Arms: plain (per-output, all inputs), oracle mask, fixed screen (no learning), lr_soft(K) (ridge logistic prior, soft gate), nn_soft(K) (neural prior, soft gate),
nn_oracle_soft(K) (labels from ground truth = label-quality upper bound). Experiment settings (EXPERIMENT status): T=1500, W=100, hidden=32, lr=3e-3, ridge=1."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

T, W, HID, LR, RIDGE = 1500, 100, 32, 3e-3, 1.0
KS = (1, 3, 10, 30, 100, 300)


def build(seed):
    from worlds import sample_world, rollout
    w = sample_world(seed)
    X, Y = rollout(w, T, np.random.default_rng(seed + 7))
    core_pos = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    return X, Y, core_pos, w.truth()[core_pos], w.n_d, w.n_act


def train_job(seed):
    from selfnn import run_scenario, perm_labels
    X, Y, cp, truth, nd, na = build(seed)
    r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "plain")
    return dict(seed=seed, nd=nd, desc=r["desc"], lab=perm_labels(r["nets"], r["Xs"], r["Ds"], seed=seed), truth=truth)


def test_job(args):
    seed, specs = args
    from selfnn import run_scenario
    X, Y, cp, truth, nd, na = build(seed)
    out = dict(seed=seed, nd=nd, arms={})
    for name, mode, prior in specs:
        r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, mode, prior=prior, truth_mask=truth)
        out["arms"][name] = dict(err=r["err"].tolist())
    return out


if __name__ == "__main__":
    from selfnn import PriorModel, NNPrior
    t0 = time.time()
    with Pool(20) as p:
        tr = p.map(train_job, range(5000, 5000 + max(KS)))
        print("training done", round(time.time() - t0), "s; label positive rate", round(float(np.mean([t["lab"].mean() for t in tr])), 3), "true", round(float(np.mean([t["truth"].mean() for t in tr])), 3), flush=True)
        priors = {}
        for K in KS:
            for nm, key, cls in (("lr_soft", "lab", "lr"), ("nn_soft", "lab", "nn"), ("nn_oracle_soft", "truth", "nn")):
                F = np.concatenate([t["desc"].reshape(-1, t["desc"].shape[-1]) for t in tr[:K]])
                y = np.concatenate([np.asarray(t[key], float).reshape(-1) for t in tr[:K]])
                m = PriorModel(RIDGE, "soft") if cls == "lr" else NNPrior(seed=K)
                m.fit(F, y)
                priors[f"{nm}{K}"] = m
        print("priors fitted", round(time.time() - t0), "s", flush=True)
        specs = [("plain", "plain", None), ("oracle", "oracle", None), ("screen", "screen", None)] + [(k, "prior", m) for k, m in priors.items()]
        te = p.map(test_job, [(s, specs) for s in range(9000, 9048)])
    json.dump(dict(test=te, T=T, W=W), open("../results/expM.json", "w"), default=float)
    print("done", round(time.time() - t0), "s")
