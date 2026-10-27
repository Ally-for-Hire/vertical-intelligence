"""PIECE-B10: dendritic computation (a cortical neuron ~ a 5-8 layer network, Beniaguev et al. 2021 [v]).
Per-output online learners (Adam, batch 1) at roughly EQUAL parameter count:
  point    : plain MLP, 32 tanh hidden units (= the project's base learner)
  dendrite : 8 'neurons' x 4 dendritic branches; branch = tanh(w_b . x + c_b) over all inputs; neuron = tanh(sum of its 4 branches x v + bias); output = linear over neurons
  sparse   : as dendrite, but each branch sees only a random 25% of the inputs (clustered synapses), with 4x more branches per neuron (16) to keep parameters equal
Worlds: exp M standard worlds and pure-interaction worlds (output change = product of two parents), 48 each, generator v2. Score: prequential error over 1,500 ticks, ratio to point.
PREDICTIONS (registered before running): P114 on pure-interaction worlds dendrite error <= 0.95 x point; P115 on standard worlds dendrite within +-5% of point; P116 sparse branches beat dense dendrites on standard worlds (<= 0.97 x dendrite), because each branch sees fewer distractors."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

T, LR = 1500, 3e-3


class Adam:
    def __init__(self, params):
        self.p = params; self.m = [np.zeros_like(x) for x in params]; self.v = [np.zeros_like(x) for x in params]; self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (P, g) in enumerate(zip(self.p, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g; self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            P -= LR * (self.m[i] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[i] / (1 - 0.999 ** self.t)) + 1e-8)


class Dendritic:
    def __init__(self, n_in, n_neur, n_br, seed, frac=1.0):
        r = np.random.default_rng(seed)
        nb = n_neur * n_br
        self.mask = (r.random((n_in, nb)) < frac).astype(float) if frac < 1 else np.ones((n_in, nb))
        fan = np.maximum(self.mask.sum(0), 1)
        self.Wb = r.normal(0, 1, (n_in, nb)) / np.sqrt(fan) * self.mask; self.cb = np.zeros(nb)
        self.V = r.normal(0, 1 / np.sqrt(n_br), nb); self.bn = np.zeros(n_neur)
        self.w2 = r.normal(0, 1 / np.sqrt(n_neur), n_neur); self.b2 = np.zeros(1)
        self.n_neur, self.n_br = n_neur, n_br
        self.opt = Adam([self.Wb, self.cb, self.V, self.bn, self.w2, self.b2])

    def step(self, x, y):
        a = np.tanh(x @ self.Wb + self.cb)                               # branches
        z = (a * self.V).reshape(self.n_neur, self.n_br).sum(1) + self.bn
        u = np.tanh(z)                                                   # neurons
        p = u @ self.w2 + self.b2[0]
        e = p - y
        gw2 = e * u; gb2 = np.array([e]); du = e * self.w2 * (1 - u * u)
        gbn = du; gV = np.repeat(du, self.n_br) * a
        da = np.repeat(du, self.n_br) * self.V * (1 - a * a)
        gWb = np.outer(x, da) * self.mask; gcb = da
        self.opt.step([gWb, gcb, gV, gbn, gw2, gb2])
        self.Wb *= self.mask
        return p


class Deep3:
    """depth-matched POINT network: in -> 32 tanh -> 8 tanh -> out, every unit of layer 2 sees all of layer 1 (a dense 'V')"""
    def __init__(self, n_in, seed):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 1 / np.sqrt(n_in), (n_in, 32)); self.b1 = np.zeros(32)
        self.W2 = r.normal(0, 1 / np.sqrt(32), (32, 8)); self.b2 = np.zeros(8)
        self.w3 = r.normal(0, 1 / np.sqrt(8), 8); self.b3 = np.zeros(1)
        self.opt = Adam([self.W1, self.b1, self.W2, self.b2, self.w3, self.b3])

    def step(self, x, y):
        a = np.tanh(x @ self.W1 + self.b1); u = np.tanh(a @ self.W2 + self.b2); p = u @ self.w3 + self.b3[0]; e = p - y
        du = e * self.w3 * (1 - u * u); da = (self.W2 @ du) * (1 - a * a)
        self.opt.step([np.outer(x, da), da, np.outer(a, du), du, e * u, np.array([e])])
        return p


def run(args):
    seed, fam = args
    from worlds import sample_world, sample_world_family, rollout
    from nn import AdamMLP, RunStd
    w = sample_world(seed) if fam == "standard" else sample_world_family(seed, "pure_interaction")
    X, Y = rollout(w, T, np.random.default_rng(seed + 7))
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    D = (Y - X[:, :Y.shape[1]])[:, cp]; n_in = X.shape[1]
    out = {}
    for arm in (("point", "dendrite", "sparse", "deep3") if os.environ.get("B10B") else ("point", "dendrite", "sparse")):
        sx, sy = RunStd(n_in), RunStd(len(cp))
        if arm == "point":
            nets = [AdamMLP(n_in, 1, 32, LR, seed + j) for j in range(len(cp))]
        elif arm == "dendrite":
            nets = [Dendritic(n_in, 8, 4, seed + j) for j in range(len(cp))]
        elif arm == "deep3":
            nets = [Deep3(n_in, seed + j) for j in range(len(cp))]
        else:
            nets = [Dendritic(n_in, 8, 16, seed + j, frac=0.25) for j in range(len(cp))]
        err = 0.0
        for t in range(T):
            sx.update(X[t]); sy.update(D[t]); xs, ys = sx(X[t]), sy(D[t])
            for j, n in enumerate(nets):
                p = n.step(xs, ys[j:j + 1])[0] if arm == "point" else n.step(xs, ys[j])
                err += (p - ys[j]) ** 2
        out[arm] = float(err / (T * len(cp)))
    return out


if __name__ == "__main__":
    t0 = time.time(); allr = {}
    with Pool(20) as p:
        for fam in ("standard", "pure_interaction"):
            res = p.map(run, [(s, fam) for s in (range(9100, 9148) if os.environ.get("B10R") else range(9000, 9048))]); allr[fam] = res
            base = np.array([r["point"] for r in res])
            print(f"{fam}: point {np.median(base):.3f} | " + " | ".join(f"{a} / point median {np.median(np.array([r[a] for r in res]) / base):.3f}" for a in ("dendrite", "sparse")) +
                  f" | sparse / dendrite median {np.median([r['sparse'] / r['dendrite'] for r in res]):.3f}" + (f" | deep3 / point {np.median([r['deep3'] / r['point'] for r in res]):.3f} | dendrite / deep3 {np.median([r['dendrite'] / r['deep3'] for r in res]):.3f} | sparse / deep3 {np.median([r['sparse'] / r['deep3'] for r in res]):.3f}" if "deep3" in res[0] else ""))
    json.dump(allr, open("../results/pb10b_dendrites.json" if os.environ.get("B10B") else "../results/pb10r_dendrites.json" if os.environ.get("B10R") else "../results/pb10_dendrites.json", "w"))
    print("done", round(time.time() - t0), "s")
