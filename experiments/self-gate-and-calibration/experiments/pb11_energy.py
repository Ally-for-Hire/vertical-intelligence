"""PIECE-B11: is ENERGY a useful constraint or only a cost? (the user's assumption A1: brains are only as smart as evolution needed, under an energy budget; Attwell & Laughlin 2001 [v]: signalling dominates brain energy, favouring sparse codes)
World: exp M causal worlds (generator v2, 48 test seeds 9000-9047). Learner: per-output online MLPs (32 hidden, Adam, lr 3e-3), fixed input/output standardisation from the first 300 ticks (shared by all arms).
Arms (identical except for the energy cost):
  plain
  act_e   : ACTIVITY energy (spikes): + lam * mean |h| on the hidden units, lam in {1e-4, 1e-3, 1e-2}
  syn_e   : SYNAPSE energy (wiring):  proximal L1 shrink of all weights by lr * lam each step, lam in {1e-4, 1e-3, 1e-2}
Training: 1,500 ticks online (prequential error = in-distribution learning). Test (frozen nets): 500 ticks of a SHIFTED rollout of the same world (actions held 25 ticks instead of 5, new start state) = generalisation.
Energy actually spent: mean |hidden activity| over training, and final sum |weights|.
PREDICTIONS (registered before running): P56 at least one activity-energy strength cuts SHIFTED test error by >= 5% vs plain (paired, 48 worlds); P57 that same strength keeps in-distribution (training) error within +2% of plain;
P58 the act_e arm at lam 1e-3 ends with <= 50% of plain's mean hidden activity; P59 synapse energy at some strength also cuts shifted error by >= 5%."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

T, TEST, HID, LR = 1500, 500, 32, 3e-3
LAMS = (1e-4, 1e-3, 1e-2)
ACT_LAMS, SYN_LAMS = ((0.03, 0.1, 0.3, 1.0, 3.0), (0.03, 0.1, 0.3)) if os.environ.get("B11B") else (LAMS, LAMS)
if os.environ.get("B11C"):
    ACT_LAMS, SYN_LAMS = (), (0.1,)


class EMLP:
    def __init__(self, n_in, seed, act_lam=0.0, syn_lam=0.0):
        from nn import AdamMLP
        self.n = AdamMLP(n_in, 1, HID, LR, seed)
        self.al, self.sl = act_lam, syn_lam
        self.act_sum = 0.0; self.steps = 0

    def predict(self, x):
        return self.n.predict(x)[0][0]

    def step(self, x, y):
        n = self.n
        p, h, xm = n.predict(x)
        e = p - y
        gW2 = np.outer(h, e); gb2 = e
        dh = (e @ n.W2.T) + self.al * np.sign(h) / HID                 # activity energy: d/dh of lam * mean|h|
        dh = dh * (1 - h * h)
        gW1 = np.outer(xm, dh); gb1 = dh
        n.t += 1
        for i, (P, g) in enumerate(zip((n.W1, n.b1, n.W2, n.b2), (gW1, gb1, gW2, gb2))):
            n.m[i] = 0.9 * n.m[i] + 0.1 * g
            n.v[i] = 0.999 * n.v[i] + 0.001 * g * g
            mh = n.m[i] / (1 - 0.9 ** n.t); vh = n.v[i] / (1 - 0.999 ** n.t)
            P -= LR * mh / (np.sqrt(vh) + 1e-8)
        if self.sl > 0:                                                 # synapse energy: proximal L1 shrink
            for P in (n.W1, n.W2):
                P[...] = np.sign(P) * np.maximum(np.abs(P) - LR * self.sl, 0.0)
        self.act_sum += float(np.mean(np.abs(h))); self.steps += 1
        return p[0]


def rollout(w, n, rng, hold):
    ns, na = w.n_state, w.n_act
    X = np.zeros((n, ns + na)); Y = np.zeros((n, ns)); a = np.zeros(na); obs = w.observe()
    for t in range(n):
        if t % hold == 0:
            a = rng.integers(-1, 2, size=na).astype(float)
        X[t, :ns] = obs; X[t, ns:] = a
        w.step(w.act_to_world(a)); obs = w.observe(); Y[t] = obs
    return X, Y


def run(seed):
    from worlds import sample_world
    w = sample_world(seed)
    X, Y = rollout(w, T, np.random.default_rng(seed + 7), 5)
    w2 = sample_world(seed); w2.s = np.random.default_rng(seed + 99).normal(0, 1, w2.n_state)
    Xt, Yt = rollout(w2, TEST, np.random.default_rng(seed + 8), 25)
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    D, Dt = (Y - X[:, :Y.shape[1]])[:, cp], (Yt - Xt[:, :Yt.shape[1]])[:, cp]
    mx, sx = X[:300].mean(0), X[:300].std(0) + 1e-6; md, sd = D[:300].mean(0), D[:300].std(0) + 1e-6
    Xs, Ds, Xts, Dts = (X - mx) / sx, (D - md) / sd, (Xt - mx) / sx, (Dt - md) / sd
    arms = [("plain", 0.0, 0.0)] + [(f"act_{l:g}", l, 0.0) for l in ACT_LAMS] + [(f"syn_{l:g}", 0.0, l) for l in SYN_LAMS]
    out = {}
    for name, al, sl in arms:
        nets = [EMLP(X.shape[1], seed + j, al, sl) for j in range(len(cp))]
        tr = np.mean([np.mean([(nets[j].step(Xs[t], Ds[t, j:j + 1]) - Ds[t, j]) ** 2 for j in range(len(cp))]) for t in range(T)])
        te = np.mean([(nets[j].predict(Xts[t]) - Dts[t, j]) ** 2 for t in range(TEST) for j in range(len(cp))])
        out[name] = dict(train=float(tr), test=float(te), act=float(np.mean([n.act_sum / n.steps for n in nets])), wsum=float(np.mean([np.abs(n.n.W1).sum() + np.abs(n.n.W2).sum() for n in nets])))
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(9100, 9148) if os.environ.get("B11C") else range(9000, 9048))
    json.dump(res, open("../results/pb11c_energy.json" if os.environ.get("B11C") else "../results/pb11b_energy.json" if os.environ.get("B11B") else "../results/pb11_energy.json", "w"))
    pl = {m: np.array([r["plain"][m] for r in res]) for m in ("train", "test", "act", "wsum")}
    print("arm          | training (in-distribution) error vs plain | SHIFTED test error vs plain (mean of per-world ratio - 1, +- SE; median) | mean |activity| / plain | sum|weights| / plain")
    for name in res[0]:
        line = []
        for m in ("train", "test"):
            rr = np.array([r[name][m] for r in res]) / pl[m] - 1
            line.append(f"{100 * rr.mean():+.1f}% +- {100 * rr.std(ddof=1) / np.sqrt(len(rr)):.1f} (median {100 * np.median(rr):+.1f}%)")
        print(f"{name:12s} | " + " | ".join(line) + f" | {np.mean([r[name]['act'] for r in res]) / pl['act'].mean():.2f} | {np.mean([r[name]['wsum'] for r in res]) / pl['wsum'].mean():.2f}")
    print("done", round(time.time() - t0), "s")
