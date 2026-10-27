"""PIECE-B12: synaptic consolidation (protect the weights that mattered for old knowledge; Kirkpatrick et al. 2017 EWC [v]), on PIECE-B1's regime-switching stream (A B C A B C A B C, 600 ticks each, 48 worlds).
Arms (per-output online MLPs, Adam, same settings as B1; no extra training updates, so compute is ~equal to 'slow'):
  slow          : plain online net
  ewc_known     : at every regime switch (boundaries GIVEN), store anchors w* and a diagonal importance F (mean squared gradient over the last 200 ticks, normalised to mean 1, accumulated);
                  every tick after the Adam step, pull weights toward the anchors: w -= lr * lam * F * (w - w*)
  ewc_surprise  : the same, but consolidation is triggered by B7's surprise signal (fast/slow error ratio > 3, at most once per 100 ticks), so no boundaries are given
lam = 1 (EXPERIMENT setting, untuned). Metrics as B1: NEW (first 100 ticks in an unseen regime), RETURN (first 100 ticks back in a seen regime), TOTAL.
PREDICTIONS (registered before running): P111 ewc_known cuts RETURN error by >= 10% vs slow; P112 ewc_known raises NEW error by <= 5% vs slow; P113 ewc_surprise keeps >= half of ewc_known's RETURN benefit."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pb1_memory as B

LAM, FWIN = 1.0, 200


class Cons:
    def __init__(self, net):
        self.net = net; self.F = None; self.anchor = None; self.g2 = []

    def record(self, x, y):
        n = self.net
        p, h, xm = n.predict(x); e = p - y
        dh = (e @ n.W2.T) * (1 - h * h)
        g = [np.outer(xm, dh), dh, np.outer(h, e), e]
        self.g2.append([gi * gi for gi in g]); self.g2 = self.g2[-FWIN:]

    def consolidate(self):
        if not self.g2:
            return
        Fn = [np.mean([g[i] for g in self.g2], 0) for i in range(4)]
        m = np.mean(np.concatenate([f.ravel() for f in Fn])) + 1e-12
        Fn = [f / m for f in Fn]
        self.F = Fn if self.F is None else [a + b for a, b in zip(self.F, Fn)]
        self.anchor = [P.copy() for P in (self.net.W1, self.net.b1, self.net.W2, self.net.b2)]

    def pull(self):
        if self.F is None:
            return
        for P, F, A in zip((self.net.W1, self.net.b1, self.net.W2, self.net.b2), self.F, self.anchor):
            P -= self.net.lr * LAM * np.minimum(F, 1.0 / (self.net.lr * LAM)) * (P - A)


def run(seed):
    from nn import AdamMLP, RunStd
    X, Y, cp = B.stream(seed)
    T, n_in = X.shape; n_out = len(cp); D = (Y - X[:, :Y.shape[1]])[:, cp]
    sx, sy = RunStd(n_in), RunStd(n_out)
    arms = ("slow", "ewc_known", "ewc_surprise")
    nets = {a: [AdamMLP(n_in, 1, B.HID, B.LR, seed + j) for j in range(n_out)] for a in arms}
    cons = {a: [Cons(n) for n in nets[a]] for a in arms}
    errs = {a: np.zeros(T) for a in arms}; es = np.ones(n_out); ef = np.ones(n_out); last_c = -1000; ncons = 0
    for t in range(T):
        sx.update(X[t]); sy.update(D[t]); xs, ys = sx(X[t]), sy(D[t])
        if t > 0 and t % B.BLOCK == 0:
            for c in cons["ewc_known"]:
                c.consolidate()
        for a in arms:
            e_t = np.zeros(n_out)
            for j, n in enumerate(nets[a]):
                if a != "slow":
                    cons[a][j].record(xs, ys[j:j + 1])
                p = n.step(xs, ys[j:j + 1])[0]; e_t[j] = (p - ys[j]) ** 2
                if a != "slow":
                    cons[a][j].pull()
            errs[a][t] = e_t.mean()
            if a == "ewc_surprise":
                es += (e_t - es) / 200; ef += (e_t - ef) / 10
                if t > 200 and np.max(ef / (es + 1e-9)) > 3 and t - last_c > 100:
                    for c in cons[a]:
                        c.consolidate()
                    last_c = t; ncons += 1
    out = {}
    for a, e in errs.items():
        blk = e.reshape(len(B.ORDER), B.BLOCK)
        out[a] = dict(new=float(blk[[1, 2], :100].mean()), ret=float(blk[3:, :100].mean()), total=float(e.mean()))
    out["ncons"] = ncons
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(9000, 9048))
    json.dump(res, open("../results/pb12_consolidation.json", "w"))
    for m in ("new", "ret", "total"):
        base = np.array([r["slow"][m] for r in res])
        print(f"{m:5s} vs slow (negative = better): " + " | ".join(f"{a} {100 * (np.array([r[a][m] for r in res]) / base - 1).mean():+.1f}% +- {100 * (np.array([r[a][m] for r in res]) / base - 1).std(ddof=1) / np.sqrt(len(res)):.1f}" for a in ("ewc_known", "ewc_surprise")))
    print("surprise-triggered consolidations per world: %.1f (true switches: 8)" % np.mean([r["ncons"] for r in res]))
    print("done", round(time.time() - t0), "s")
