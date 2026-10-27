"""Exp L: self-learning network v0. Learns, across scenarios, a prior over which inputs affect which outputs; with no experience it IS the plain per-output net.
Training scenarios: 100 (generator seeds 5000..), n_distract drawn from {10,50,200}; held-out test scenarios: 24 (seeds 9000..), same distribution.
Arms on each test scenario: joint (one plain MLP), plain (per-output, all inputs), oracle (true parents), screen (fixed Bonferroni correlation screen on the first W ticks;
NO learning), prior(K) for K in {1,3,10,30,100} training scenarios (own-experience labels), prior_oracleK (labels from ground truth: an upper bound on label quality).
Experiment settings (EXPERIMENT status): T=1500 ticks, W=100, hidden=32, lr=3e-3 (the tuned plain-net setting of Exp K), ridge=1."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

T, W, HID, LR, RIDGE = 1500, 100, 32, 3e-3, 1.0
KS = (1, 3, 10, 30, 100)


def build(seed):
    from worlds import World, rollout
    rng = np.random.default_rng(seed)
    nd = int(rng.choice([10, 50, 200]))
    w = World(seed, n_state=6, n_act=2, n_distract=nd)
    X, Y = rollout(w, T, np.random.default_rng(seed + 7))
    core_pos = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    truth = w.truth()[core_pos]                       # (n_out, n_in) true parents of the core outputs
    return X, Y, core_pos, truth, nd, w.n_act


def train_job(seed):
    from selfnn import run_scenario, perm_labels
    X, Y, cp, truth, nd, na = build(seed)
    r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "plain")
    lab = perm_labels(r["nets"], r["Xs"], r["Ds"], seed=seed)
    return dict(seed=seed, nd=nd, desc=r["desc"], lab=lab, truth=truth, final=float(r["err"][-300:].mean()))


def test_job(args):
    seed, specs = args
    from selfnn import run_scenario
    X, Y, cp, truth, nd, na = build(seed)
    out = dict(seed=seed, nd=nd, arms={})

    def prf(masks):
        tp = float((masks * truth).sum())
        return dict(prec=tp / max(masks.sum(), 1), rec=tp / max(truth.sum(), 1), kept=float(masks.sum(1).mean()))
    for name, mode, prior in specs:
        r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, mode, prior=prior, truth_mask=truth)
        out["arms"][name] = dict(err=r["err"].tolist(), **prf(r["masks"]))
    return out


if __name__ == "__main__":
    from selfnn import PriorModel
    t0 = time.time()
    with Pool(20) as p:
        tr = p.map(train_job, range(5000, 5100))
        print("training scenarios done", round(time.time() - t0), "s; positive-label rate (own labels):", round(float(np.mean([t["lab"].mean() for t in tr])), 4),
              " true-parent rate:", round(float(np.mean([t["truth"].mean() for t in tr])), 4), flush=True)
        priors = {}
        for K in KS:
            for nm, labkey in (("prior", "lab"), ("prior_oracle", "truth")):
                F = np.concatenate([t["desc"].reshape(-1, t["desc"].shape[-1]) for t in tr[:K]])
                y = np.concatenate([np.asarray(t[labkey], float).reshape(-1) for t in tr[:K]])
                for rule, tau in (("hard", 0.5), ("hard", 0.1), ("hard", 0.02), ("zeromiss", 0), ("soft", 0)):
                    m = PriorModel(RIDGE, rule, tau)
                    m.fit(F, y)
                    priors[f"{nm}{K}_{rule}{tau if rule == 'hard' else ''}"] = m
        priors["noexp"] = PriorModel(RIDGE)                    # no experience: w=None -> keeps everything (parity arm)
        specs = [("joint", "joint", None), ("plain", "plain", None), ("oracle", "oracle", None), ("screen", "screen", None)] + [(k, "prior", m) for k, m in priors.items()]
        te = p.map(test_job, [(s, specs) for s in range(9000, 9024)])
    json.dump(dict(test=te, train_summary=[dict(seed=t["seed"], nd=t["nd"], final=t["final"]) for t in tr], T=T, W=W), open("../results/expL2.json", "w"), default=float)
    print("done", round(time.time() - t0), "s")
