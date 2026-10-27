"""Exp O (queue Q29): where does the recursive self-labelling loop break? Same loop as exp N part 2 (transformer, relevance prior refit every 5 scenarios, each scenario learned with the
previous prior) with four variants, 6 chains x 60 scenarios each, checkpoints at generations 10, 20, 40, 60, each evaluated on 24 held-out worlds (seeds 9000-9023):
 real    : labels from fresh REAL data (reference; = exp N protocol)
 synth   : labels computed on data SIMULATED by the learner's own model (model-collapse setting; no new world information enters)
 hard    : real labels, but the prior is applied as a hard mask (p < 0.5 excludes the input; an excluded input can no longer influence the net, so its importance cannot be re-measured)
 noisy20 : real labels with 20% of the label entries flipped each generation
Experiment settings (EXPERIMENT status): T=1500, W=100, refit period 5, NN prior 2x32 tanh, 400 Adam steps, lr 1e-3."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

T, W, LR = 1500, 100, 1e-3
GENS = (10, 20, 40, 60)
NSC = 60
REFIT = 5
VARIANTS = ("real", "synth", "hard", "noisy20")


def build(seed):
    from worlds import sample_world, rollout
    w = sample_world(seed)
    X, Y = rollout(w, T, np.random.default_rng(seed + 7))
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    return X, Y, cp, w.truth()[cp], w.n_d, w.n_act


def chain_job(args):
    import torch
    torch.set_num_threads(1)
    from tfm import run_scenario_tf, perm_labels_tf, synth_labels_tf
    from selfnn import NNPrior
    variant, chain = args
    prior, D, L, saved, rates, true_agree = None, [], [], {}, [], []
    rng = np.random.default_rng(chain * 7 + 1)
    for k in range(NSC):
        if k in GENS:
            saved[k] = prior
        seed = 30000 + chain * 1000 + k
        X, Y, cp, truth, nd, na = build(seed)
        r = run_scenario_tf(X, Y, cp, na, T, W, seed, "prior" if prior is not None else "plain", prior=prior, lr=LR, gate="hard" if variant == "hard" else "soft")
        if variant == "synth":
            lab = synth_labels_tf(r["model"], r["bias"], r["sx"], r["sy"], r["Xs"], r["Ds"], cp, seed=seed)
        else:
            lab = perm_labels_tf(r["model"], r["bias"], r["Xs"], r["Ds"], seed=seed)
        if variant == "noisy20":
            flip = rng.random(lab.shape) < 0.20
            lab = np.where(flip, 1.0 - lab, lab)
        D.append(r["desc"].reshape(-1, r["desc"].shape[-1]))
        L.append(lab.reshape(-1))
        rates.append(float(lab.mean()))
        true_agree.append(float((lab == truth).mean()))                       # label agreement with ground truth (diagnostic only)
        if (k + 1) % REFIT == 0:
            prior = NNPrior(seed=k)
            prior.fit(np.concatenate(D), np.concatenate(L))
    saved[NSC] = prior
    return dict(variant=variant, chain=chain, priors=saved, label_rate=rates, truth_agree=true_agree)


def test_job(args):
    import torch
    torch.set_num_threads(1)
    from tfm import run_scenario_tf
    seed, specs = args
    X, Y, cp, truth, nd, na = build(seed)
    out = dict(seed=seed, arms={})
    for name, prior, gate in specs:
        r = run_scenario_tf(X, Y, cp, na, T, W, seed, "prior", prior=prior, lr=LR, gate=gate)
        out["arms"][name] = r["err"].tolist()
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        chains = p.map(chain_job, [(v, c) for v in VARIANTS for c in range(5)], chunksize=1)
        print("chains done", round(time.time() - t0), "s", flush=True)
        jobs = []
        for c in chains:
            sp = [(f"{c['variant']}_c{c['chain']}_g{g}", pr, "hard" if c["variant"] == "hard" else "soft") for g, pr in c["priors"].items() if pr is not None and g in GENS + (NSC,)]
            for s in range(9000, 9024):
                jobs.append((s, sp))
        te = p.map(test_job, jobs, chunksize=1)
    json.dump(dict(test=te, chains=[dict(variant=c["variant"], chain=c["chain"], label_rate=c["label_rate"], truth_agree=c["truth_agree"]) for c in chains]), open("../results/expO.json", "w"), default=float)
    print("done", round(time.time() - t0), "s")
