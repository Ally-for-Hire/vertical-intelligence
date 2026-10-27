"""Exp P (queue Q2): can no-experience or alternative learned methods match the learned relevance prior? MLP per-output framework, same worlds as exp M.
Baselines tuned on SEPARATE validation worlds (seeds 1000-1011): group-lasso strength on the first-layer input rows (proximal), input dropout. Learned arms: NNPrior (hand-built
descriptors, 2x32 tanh) and SeqPrior (1D CNN over the raw first-100-tick window of each input-output pair), each trained on own-experience labels from K training scenarios
(K=10, 100; seeds 5000..). Test: 48 held-out worlds (seeds 9000-9047). Metric: gain over the plain per-output net in prequential error over the first 1000 ticks."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

T, W, HID, LR = 1500, 100, 32, 3e-3


def build(seed):
    from worlds import sample_world, rollout
    w = sample_world(seed)
    X, Y = rollout(w, T, np.random.default_rng(seed + 7))
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    return X, Y, cp, w.truth()[cp], w.n_d, w.n_act


def tune_job(a):
    from selfnn import run_scenario
    seed, kind, val = a
    X, Y, cp, truth, nd, na = build(seed)
    gl, dr = (val, 0.0) if kind == "gl" else ((0.0, val) if kind == "drop" else (0.0, 0.0))
    r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "plain", gl=gl, drop=dr)
    return (kind, val, float(np.mean(r["err"][:1000])))


def train_job(seed):
    import torch
    torch.set_num_threads(1)
    from selfnn import run_scenario, perm_labels, SeqPrior
    X, Y, cp, truth, nd, na = build(seed)
    r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "plain")
    lab = perm_labels(r["nets"], r["Xs"], r["Ds"], seed=seed)
    S, Fl = SeqPrior.pairs(r["Xs"][:W], r["Ds"][:W], cp, na)
    return dict(seed=seed, desc=r["desc"].reshape(-1, r["desc"].shape[-1]), lab=lab.reshape(-1), S=S, Fl=Fl)


def test_job(args):
    import torch
    torch.set_num_threads(1)
    from selfnn import run_scenario
    seed, specs, gl, dr = args
    X, Y, cp, truth, nd, na = build(seed)
    out = dict(seed=seed, nd=nd, arms={})
    for name, mode, prior in specs:
        if name == "gl":
            r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "plain", gl=gl)
        elif name == "drop":
            r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "plain", drop=dr)
        else:
            r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, mode, prior=prior, truth_mask=truth)
        out["arms"][name] = r["err"].tolist()
    return out


if __name__ == "__main__":
    from selfnn import NNPrior, SeqPrior
    t0 = time.time()
    with Pool(20) as p:
        grid = [("gl", v) for v in (1e-3, 3e-3, 1e-2, 3e-2, 0.1)] + [("drop", v) for v in (0.1, 0.3, 0.5)] + [("none", 0.0)]
        tr = p.map(tune_job, [(1000 + s, k, v) for s in range(12) for (k, v) in grid])
        sc = {}
        for k, v, e in tr:
            sc.setdefault((k, v), []).append(e)
        sc = {k: float(np.mean(v)) for k, v in sc.items()}
        print("validation first-1000 error:", {str(k): round(v, 3) for k, v in sc.items()}, flush=True)
        gl = min([k for k in sc if k[0] == "gl"], key=lambda k: sc[k])[1]
        dr = min([k for k in sc if k[0] == "drop"], key=lambda k: sc[k])[1]
        print("chosen group-lasso", gl, "dropout", dr, flush=True)
        trn = p.map(train_job, range(5000, 5100))
        print("training data done", round(time.time() - t0), "s", flush=True)
        priors = {}
        for K in (10, 100):
            F = np.concatenate([t["desc"] for t in trn[:K]]); y = np.concatenate([t["lab"] for t in trn[:K]])
            m = NNPrior(seed=K); m.fit(F, y); priors[f"nn_soft{K}"] = m
            S = np.concatenate([t["S"] for t in trn[:K]]); Fl = np.concatenate([t["Fl"] for t in trn[:K]])
            sp = SeqPrior(seed=K); sp.fit(S, Fl, y); priors[f"seq_soft{K}"] = sp
        print("priors fitted", round(time.time() - t0), "s", flush=True)
        specs = [("plain", "plain", None), ("oracle", "oracle", None), ("screen", "screen", None), ("gl", "plain", None), ("drop", "plain", None)] + [(k, "prior", m) for k, m in priors.items()]
        te = p.map(test_job, [(s, specs, gl, dr) for s in range(9000, 9048)])
    json.dump(dict(test=te, gl=gl, dr=dr, val={str(k): v for k, v in sc.items()}), open("../results/expP.json", "w"), default=float)
    print("done", round(time.time() - t0), "s")
