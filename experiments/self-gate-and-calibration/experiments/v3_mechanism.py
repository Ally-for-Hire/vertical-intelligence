"""RIGOR-3a: what IS the soft gate, mechanically?
Theorem (derivation in paper/theory.md, T7): for a first-layer input scaled by p_i, the effective weight is u_i = p_i W1_i. Its gradient is p_i times the ungated one, and Adam's update m/sqrt(v) is
invariant to a constant gradient scale. So after the switch, du_i = p_i * (Adam step of the ungated problem): the gate is EXACTLY a first-layer re-initialisation u_i <- p_i W1_i plus a per-input learning
rate p_i * lr. This holds up to Adam's epsilon and a transient while Adam's moments still hold pre-switch gradients.
Arms (same worlds, same prior, the gate switches on at t = W as in run_scenario): plain; GATED (the project's soft gate); BOTH (no gate; at t = W rows W1_i *= p_i and per-row lr = p_i lr: the theorem's equivalent);
LR-ONLY (per-row lr = p_i lr, weights untouched); INIT-ONLY (W1_i *= p_i at t = W, normal lr).
PREDICTIONS (registered in PREDICTIONS.md before running): P6 BOTH matches GATED: |paired mean gain difference| <= 0.005 and per-world correlation of gains >= 0.95;
P7 LR-ONLY recovers >= 70% of the GATED gain, INIT-ONLY <= 30%."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import expM_curriculum_scaling as M


def run_variant(X, Y, cp, na, seed, prior, variant):
    from nn import AdamMLP, RunStd
    from selfnn import pair_descriptors, FEATS
    T, W = M.T, M.W
    ns = Y.shape[1]; n_in = X.shape[1]; D = Y - X[:, :ns]
    sx, sy = RunStd(n_in), RunStd(len(cp)); n_out = len(cp)
    nets = [AdamMLP(n_in, 1, M.HID, M.LR, seed + j, np.ones(n_in)) for j in range(n_out)]
    rowlr = [None] * n_out
    err = np.zeros(T); Xs = np.zeros((T, n_in)); Ds = np.zeros((T, n_out))
    for t in range(T):
        x = X[t]; y = D[t][cp]
        sx.update(x); sy.update(y)
        xs, ys = sx(x), sy(y)
        Xs[t], Ds[t] = xs, ys
        if t == W and variant != "plain":
            desc = pair_descriptors(Xs[:W], Ds[:W], cp, na)
            p = prior.prob(desc.reshape(-1, FEATS)).reshape(n_out, n_in)
            for j in range(n_out):
                if variant == "gated":
                    nets[j].mask = p[j]
                if variant in ("both", "init"):
                    nets[j].W1 *= p[j][:, None]
                if variant in ("both", "lr"):
                    rowlr[j] = p[j][:, None]
        e = 0.0
        for j, net in enumerate(nets):
            if rowlr[j] is None:
                pj = net.step(xs, ys[j:j + 1])
            else:                                                  # Adam step with a per-input-row learning rate on W1 (other layers at the base lr)
                W1b = net.W1.copy()
                pj = net.step(xs, ys[j:j + 1])
                net.W1 = W1b + (net.W1 - W1b) * rowlr[j]
            e += (pj[0] - ys[j]) ** 2
        err[t] = e / n_out
    return err


def job(args):
    seed, prior = args
    X, Y, cp, truth, nd, na = M.build(seed)
    return {v: float(run_variant(X, Y, cp, na, seed, prior, v)[:1000].mean()) for v in ("plain", "gated", "both", "lr", "init")}


if __name__ == "__main__":
    from selfnn import NNPrior
    t0 = time.time()
    with Pool(20) as p:
        tr = p.map(M.train_job, range(5000, 5100))
        F = np.concatenate([t["desc"].reshape(-1, t["desc"].shape[-1]) for t in tr]); y = np.concatenate([np.asarray(t["lab"], float).reshape(-1) for t in tr])
        prior = NNPrior(seed=100); prior.fit(F, y)
        res = p.map(job, [(s, prior) for s in range(9000, 9048)])
    json.dump(res, open("../results/v3_mechanism.json", "w"))
    pl = np.array([r["plain"] for r in res])
    g = {v: pl - np.array([r[v] for r in res]) for v in ("gated", "both", "lr", "init")}
    for v in g:
        print(f"{v:6s} gain {g[v].mean():+.4f} +- {g[v].std(ddof=1) / np.sqrt(len(pl)):.4f}")
    d = g["both"] - g["gated"]
    print("BOTH - GATED paired: %+.4f +- %.4f; per-world correlation %.3f" % (d.mean(), d.std(ddof=1) / np.sqrt(len(d)), np.corrcoef(g["both"], g["gated"])[0, 1]))
    print("share of gated gain: lr-only %.2f, init-only %.2f" % (g["lr"].mean() / g["gated"].mean(), g["init"].mean() / g["gated"].mean()))
    print("done", round(time.time() - t0), "s")
