"""Q19: does the learned relevance prior help on dynamics whose wiring is a REAL fly connectome subgraph (FlyWire v783, 60 subgraphs x 300 neurons, pairs >= 5 synapses, sign from dominant transmitter)?
World: rate network s' = (1-leak) s + leak * tanh(gain * sum_i sign_i w_ij s_i / in_j + action input) + noise, over all 300 neurons. The agent sees only the first N_OBS neurons (growth order = the densely
connected core) plus 3 action channels (each injected into one observed neuron); the other neurons are hidden. Truth for an observed output j = itself + observed presynaptic neurons (w >= 5 synapses) + any action injected into j.
Arms (gain over the plain per-output MLP in ticks 300-1000, same protocol as expS2): oracle mask; prior trained on 100 SYNTHETIC 'standard' worlds (expS2 setup); prior trained on CONNECTOME worlds (20 held-in subgraphs,
own permutation labels); evaluated on the 40 other subgraphs. Also AUROC of each prior for true parents. Leak/gain/noise are simulation settings (declared here, not tuned)."""
import os, sys, time, pickle
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import expS2_anti_prior as S2

N_OBS, N_ACT, LEAK, GAIN, NOISE, T = 50, 3, 0.3, 2.0, 0.05, 1500
Z = np.load("../results/q19_subgraphs.npz")


def build(g):
    W, sgn = Z["W"][g], Z["sign"][g]
    rng = np.random.default_rng(g)
    n = W.shape[0]
    Wn = (W * sgn[:, None]) / (W.sum(0, keepdims=True) + 1e-9)             # signed, each neuron's inputs normalised to total weight 1
    inj = rng.choice(N_OBS, N_ACT, replace=False)
    s = rng.normal(size=n) * 0.1
    X = np.zeros((T, N_OBS + N_ACT)); Y = np.zeros((T, N_OBS))
    a = np.zeros(N_ACT)
    for t in range(T):
        if t % 5 == 0:
            a = rng.integers(-1, 2, size=N_ACT).astype(float)
        X[t, :N_OBS] = s[:N_OBS]; X[t, N_OBS:] = a
        drive = s @ Wn
        drive[inj] += a
        s = (1 - LEAK) * s + LEAK * np.tanh(GAIN * drive) + NOISE * rng.normal(size=n)
        Y[t] = s[:N_OBS]
    truth = np.zeros((N_OBS, N_OBS + N_ACT), bool)
    truth[:, :N_OBS] = (W[:N_OBS, :N_OBS].T > 0)
    truth[np.arange(N_OBS), np.arange(N_OBS)] = True
    for k, j in enumerate(inj):
        truth[j, N_OBS + k] = True
    return X, Y, list(range(N_OBS)), truth, N_ACT


def auroc(p, truth):
    lab = truth.ravel(); pj = p.ravel()
    r = pj.argsort().argsort()
    return float((r[lab].mean() - (lab.sum() - 1) / 2) / (len(lab) - lab.sum()))


def train_job(g):
    from selfnn import run_scenario, perm_labels
    X, Y, cp, truth, na = build(g)
    r = run_scenario(X, Y, cp, na, T, S2.HID, S2.LR, S2.W, g, "plain")
    lab = perm_labels(r["nets"], r["Xs"], r["Ds"], seed=g)
    return r["desc"].reshape(-1, r["desc"].shape[-1]), lab.reshape(-1), float(lab.mean()), float(truth.mean())


def world_job(args):
    g, synth, conn = args
    from selfnn import run_scenario
    X, Y, cp, truth, na = build(g)
    out = dict(g=g, truth_rate=float(truth.mean()))
    out["plain"] = run_scenario(X, Y, cp, na, T, S2.HID, S2.LR, S2.W, g, "plain")["err"]
    out["oracle"] = run_scenario(X, Y, cp, na, T, S2.HID, S2.LR, S2.W, g, "oracle", truth_mask=truth)["err"]
    rs = run_scenario(X, Y, cp, na, T, S2.HID, S2.LR, S2.W, g, "prior", prior=synth)
    rc = run_scenario(X, Y, cp, na, T, S2.HID, S2.LR, S2.W, g, "prior", prior=conn)
    out["synth"], out["conn"] = rs["err"], rc["err"]
    d = rs["desc"].reshape(-1, rs["desc"].shape[-1])
    out["auc_synth"] = auroc(synth.prob(d).reshape(truth.shape), truth)
    out["auc_conn"] = auroc(conn.prob(d).reshape(truth.shape), truth)
    return out


if __name__ == "__main__":
    from selfnn import NNPrior
    t0 = time.time()
    with Pool(20) as p:
        std = p.map(S2.train_job, [(s, "standard") for s in range(5000, 5100)])
        synth = NNPrior(seed=1); synth.fit(np.concatenate([t[0] for t in std]), np.concatenate([t[1] for t in std]))
        tr = p.map(train_job, range(20))
        conn = NNPrior(seed=2); conn.fit(np.concatenate([t[0] for t in tr]), np.concatenate([t[1] for t in tr]))
        print("connectome worlds: own-label positive rate %.3f, true-parent rate %.3f" % (np.mean([t[2] for t in tr]), np.mean([t[3] for t in tr])), flush=True)
        res = p.map(world_job, [(g, synth, conn) for g in range(20, 60)])
    pickle.dump(res, open("../results/q19.pkl", "wb"))
    sl = slice(S2.W + S2.SEL, 1000)
    pl = np.array([np.mean(x["plain"][sl]) for x in res])
    print("40 held-out connectome worlds: plain error in ticks 300-1000 %.3f" % pl.mean())
    gains = {}
    for key in ("oracle", "synth", "conn"):
        v = np.array([np.mean(x[key][sl]) for x in res]); d = pl - v; gains[key] = d
        print(f"  {key:7s} gain {d.mean():+.3f} +- {d.std(ddof=1)/np.sqrt(len(d)):.3f}   worlds improved {int((d > 0).sum())}/{len(d)}")
    for key in ("synth", "conn"):
        print(f"  share of oracle gain, {key}: {gains[key].mean()/gains['oracle'].mean():.2f}")
    print("  AUROC for true parents: synthetic-trained %.3f, connectome-trained %.3f" % (np.mean([x["auc_synth"] for x in res]), np.mean([x["auc_conn"] for x in res])))
    print("done", round(time.time() - t0), "s")
