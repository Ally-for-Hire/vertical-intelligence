"""Exp N: the self-learning TRANSFORMER. Variable-token transformer (tfm.py), online, lr=1e-3 (tuned on validation worlds). Experience = a learned relevance prior used as an
attention bias (log p, floor 1e-3; zero bias = the plain transformer exactly).
Part 1 (non-recursive): priors trained on own-experience labels from PLAIN transformers, K in {1,3,10,30,100,300}; 48 held-out worlds; arms plain, oracle mask, prior(K), prior_oracle(K).
Part 2 (RECURSIVE generations): 8 independent chains; in a chain scenario k is learned with the prior fitted on the labels of scenarios 0..k-1, where those labels were produced by
networks that THEMSELVES used the previous prior (the loop that trains on its own outputs); the prior is refit every 5 scenarios; checkpoints at generation counts 0,5,10,20,40,80,150.
Each checkpoint is evaluated on 24 held-out worlds. Experiment settings (EXPERIMENT status): T=1500, W=100, refit period 5, NN prior 2x32 tanh, 400 Adam steps."""
import os, sys, json, time, pickle
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

T, W, LR = 1500, 100, 1e-3
KS = (1, 3, 10, 30, 100, 300)
GENS = (0, 5, 10, 20, 40, 80, 150)
REFIT = 5


def build(seed):
    from worlds import sample_world, rollout
    w = sample_world(seed)
    X, Y = rollout(w, T, np.random.default_rng(seed + 7))
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    return X, Y, cp, w.truth()[cp], w.n_d, w.n_act


def train_job(seed):
    import torch
    torch.set_num_threads(1)
    from tfm import run_scenario_tf, perm_labels_tf
    X, Y, cp, truth, nd, na = build(seed)
    r = run_scenario_tf(X, Y, cp, na, T, W, seed, "plain", lr=LR)
    return dict(seed=seed, desc=r["desc"], lab=perm_labels_tf(r["model"], r["bias"], r["Xs"], r["Ds"], seed=seed), truth=truth)


def test_job(args):
    import torch
    torch.set_num_threads(1)
    from tfm import run_scenario_tf
    seed, specs = args
    X, Y, cp, truth, nd, na = build(seed)
    out = dict(seed=seed, nd=nd, arms={})
    for name, mode, prior in specs:
        r = run_scenario_tf(X, Y, cp, na, T, W, seed, mode, prior=prior, truth_mask=truth, lr=LR)
        out["arms"][name] = r["err"].tolist()
    return out


def chain_job(chain):
    import torch
    torch.set_num_threads(1)
    from tfm import run_scenario_tf, perm_labels_tf
    from selfnn import NNPrior
    prior, D, L, saved, pl = None, [], [], {}, []
    for k in range(max(GENS) + 1):
        if k in GENS:
            saved[k] = prior
        seed = 20000 + chain * 1000 + k
        X, Y, cp, truth, nd, na = build(seed)
        r = run_scenario_tf(X, Y, cp, na, T, W, seed, "prior" if prior is not None else "plain", prior=prior, lr=LR)
        lab = perm_labels_tf(r["model"], r["bias"], r["Xs"], r["Ds"], seed=seed)
        D.append(r["desc"].reshape(-1, r["desc"].shape[-1]))
        L.append(lab.reshape(-1))
        pl.append(float(lab.mean()))
        if (k + 1) % REFIT == 0:
            prior = NNPrior(seed=k)
            prior.fit(np.concatenate(D), np.concatenate(L))
    return dict(chain=chain, priors=saved, label_rate=pl)


if __name__ == "__main__":
    from selfnn import NNPrior
    t0 = time.time()
    with Pool(20) as p:
        tr = p.map(train_job, range(5000, 5000 + max(KS)))
        print("part 1 training done", round(time.time() - t0), "s; label rate", round(float(np.mean([t["lab"].mean() for t in tr])), 4), "true rate", round(float(np.mean([t["truth"].mean() for t in tr])), 4), flush=True)
        priors = {}
        for K in KS:
            for nm, key in (("prior", "lab"), ("prior_oracle", "truth")):
                if nm == "prior_oracle" and K not in (10, 100, 300):
                    continue
                F = np.concatenate([t["desc"].reshape(-1, t["desc"].shape[-1]) for t in tr[:K]])
                y = np.concatenate([np.asarray(t[key], float).reshape(-1) for t in tr[:K]])
                m = NNPrior(seed=K)
                m.fit(F, y)
                priors[f"{nm}{K}"] = m
        specs = [("plain", "plain", None), ("oracle", "oracle", None)] + [(k, "prior", m) for k, m in priors.items()]
        te1 = p.map(test_job, [(s, specs) for s in range(9000, 9048)])
        print("part 1 done", round(time.time() - t0), "s", flush=True)
        json.dump(dict(test=te1), open("../results/expN1.json", "w"), default=float)
        chains = p.map(chain_job, range(8))
        print("part 2 chains done", round(time.time() - t0), "s", flush=True)
        specs2 = [("plain", "plain", None)]
        jobs = []
        for c in chains:
            sp = [(f"c{c['chain']}_g{g}", "prior", pr) for g, pr in c["priors"].items() if pr is not None]
            for s in range(9000, 9024):
                jobs.append((s, sp))
        te2 = p.map(test_job, jobs)
    json.dump(dict(test=te2, label_rates=[c["label_rate"] for c in chains]), open("../results/expN2.json", "w"), default=float)
    print("done", round(time.time() - t0), "s")
