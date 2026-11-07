"""Exp Q (queue Q16): out-of-family generalisation. Priors are trained on the STANDARD scenario distribution (K=100 scenarios, seeds 5000-5099, own-experience labels) and then
used, unchanged, on six families (24 worlds each, seeds 9000-9023): standard (control), kinds_tanh_product, many_parents_4_6 (outside the training range 1-3), high_noise_0.4,
copy_distractors_only, delayed_effects (parents act with a one-step delay, invisible to lag-1 statistics). Arms: plain, oracle mask, fixed screen, nn_soft (hand-built descriptors),
seq_long (raw-window CNN). Metric: gain over the plain per-output MLP in first-1000-tick error; a NEGATIVE gain means the prior hurts."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

T, W, HID, LR = 1500, 100, 32, 3e-3
FAMS = ["standard", "kinds_tanh_product", "many_parents_4_6", "high_noise_0.4", "copy_distractors_only", "delayed_effects"]


def build(seed, fam="standard"):
    from worlds import sample_world_family, rollout
    w = sample_world_family(seed, fam)
    X, Y = rollout(w, T, np.random.default_rng(seed + 7))
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    return X, Y, cp, w.truth()[cp], w.n_d, w.n_act


def train_job(seed):
    import torch
    torch.set_num_threads(1)
    from selfnn import run_scenario, perm_labels, SeqPrior
    X, Y, cp, truth, nd, na = build(seed)
    r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "plain")
    lab = perm_labels(r["nets"], r["Xs"], r["Ds"], seed=seed)
    S, Fl = SeqPrior.pairs(r["Xs"][:W], r["Ds"][:W], cp, na)
    return dict(desc=r["desc"].reshape(-1, r["desc"].shape[-1]), lab=lab.reshape(-1), S=S, Fl=Fl)


def test_job(args):
    import torch
    torch.set_num_threads(1)
    from selfnn import run_scenario
    seed, fam, specs = args
    X, Y, cp, truth, nd, na = build(seed, fam)
    out = dict(seed=seed, fam=fam, arms={})
    for name, mode, prior in specs:
        r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, mode, prior=prior, truth_mask=truth)
        out["arms"][name] = r["err"].tolist()
    return out


if __name__ == "__main__":
    from selfnn import NNPrior, SeqPrior
    t0 = time.time()
    with Pool(20) as p:
        trn = p.map(train_job, range(5000, 5100))
        F = np.concatenate([t["desc"] for t in trn]); y = np.concatenate([t["lab"] for t in trn])
        nn = NNPrior(seed=100); nn.fit(F, y)
        sp = SeqPrior(steps=1500, seed=100); sp.fit(np.concatenate([t["S"] for t in trn]), np.concatenate([t["Fl"] for t in trn]), y)
        print("priors fitted", round(time.time() - t0), "s", flush=True)
        specs = [("plain", "plain", None), ("oracle", "oracle", None), ("screen", "screen", None), ("nn_soft", "prior", nn), ("seq_long", "prior", sp)]
        te = p.map(test_job, [(s, f, specs) for f in FAMS for s in range(9000, 9024)])
    json.dump(dict(test=te), open("../results/expQ.json", "w"), default=float)
    print("done", round(time.time() - t0), "s")
