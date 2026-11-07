"""cell.py: ONE shared learned pattern, reused at every depth, trained jointly on all its uses (author 2026-11-03: 'patterns IN the patterns and so on and so forth
as long as we get it compute'; the goal: infinitely scaling depth from nothing with one simple rule).
The library holds ONE tiny net theta (a 'cell') and a list of INSTANCES (s, p): instance k = cell(inputs rotated by s, output of instance p), p = -1 meaning no parent.
A problem is solved by linking instances (measured help, confirmed on both halves of the decision points, evidence bar log2(#candidates+1)); an instance that is not
in the library yet is created on the spot (it costs no new weights: only a wiring). Because every instance shares theta, every problem that uses any instance
teaches the same cell: theta is re-fitted on ALL stored problems jointly (each keeps its own readout, refit exactly) and replaces the old cell only if pooled
decision-point error drops; the number of refit steps is the compute knob (cfg['refit']). If nothing helps (the cell is still blank), the best-looking wirings are
proposed, the cell trained on them, and kept only if they then help; otherwise everything is reverted.
Exam: exam(problems) freezes theta and the instances; only readouts are fitted on existing instances."""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import math
import numpy as np
import torch
import flat
from flat import vbits

torch.set_num_threads(1)
def _mix(lib, ops):
    """Family-free cell (X1.40): hidden unit j uses activation j % 4 of tanh / sin / relu / identity; nothing says which one the world needs."""
    def f(pre):
        H = pre.shape[-1]; out = []
        for j in range(H):
            x = pre[..., j]; k = j % 4; out.append(ops[0](x) if k == 0 else ops[1](x) if k == 1 else ops[2](x) if k == 2 else x)
        return lib.stack(out, -1)
    return f


ACT = dict(np=np.tanh, th=torch.tanh)       # inner activation of the cell; Learner(cfg act="sin") swaps in sin (X1.39: matches the world's sine rule, a family hint, so it is its own arm)
DEFAULT = dict(hid=6, m=8, ridge=3.0, max_links=8, refit=30, lr=0.03, propose=3, patience=10, n_lab=60, act="tanh", l1=0.0, refit_rows=0, init=0)   # CONST: x136_cfg


def fwd_np(theta, X, nodes, upto=None):
    """Outputs of instances 0..upto-1 on X (numpy)."""
    W1x, wp, b1, w2, b2 = theta; m = X.shape[1]; XW = [X @ np.roll(W1x, -s, 0) for s in range(m)]; K = len(nodes) if upto is None else upto
    H = np.zeros((len(X), K))
    for k in range(K):
        s, p = nodes[k]; pre = XW[s] + (H[:, p:p + 1] * wp if p >= 0 else 0.0) + b1; H[:, k] = np.tanh(ACT["np"](pre) @ w2 + b2)
    return H, XW


class Learner:
    def __init__(self, seed, cfg=None):
        self.cfg = dict(DEFAULT, **(cfg or {})); self.activate(); self.rng = np.random.default_rng([seed, 136, self.cfg["init"]] if self.cfg["init"] else [seed, 136]); m, H = self.cfg["m"], self.cfg["hid"]
        self.theta = [self.rng.standard_normal((m, H)) / math.sqrt(m + 1), self.rng.standard_normal(H) / math.sqrt(m + 1), np.zeros(H), self.rng.standard_normal(H) / math.sqrt(H), np.zeros(())]
        self.nodes, self.index, self.P, self.heads = [], {}, [], []     # P: stored problems (xtr, ytr, xva, yva)

    def activate(self):
        """Set the module-wide inner activation to this learner's (several learners with different activations can share a process: call before using one)."""
        ACT.update(np=lambda x: np.maximum(x, 0.0), th=torch.relu) if self.cfg["act"] == "relu" else ACT.update(np=np.sin, th=torch.sin) if self.cfg["act"] == "sin" else ACT.update(np=_mix(np, (np.tanh, np.sin, lambda x: np.maximum(x, 0.0))), th=_mix(torch, (torch.tanh, torch.sin, torch.relu))) if self.cfg["act"] == "mix" else ACT.update(np=np.tanh, th=torch.tanh)

    def size(self): return sum(a.size for a in self.theta) + 2 * len(self.nodes)          # one cell + a wiring (s, p) per instance

    def candidates(self, X, Hn):
        """Outputs of every possible instance (s, p) given the current library outputs Hn: columns ordered by (p, s); p in -1..K-1."""
        W1x, wp, b1, w2, b2 = self.theta; m = X.shape[1]; K = Hn.shape[1]; cols, specs = [], []
        for s in range(m):
            XW = X @ np.roll(W1x, -s, 0)
            par = np.hstack([np.zeros((len(X), 1)), Hn])                       # p = -1 is the zero parent
            pre = XW[:, None, :] + par[:, :, None] * wp + b1
            cols.append(np.tanh(ACT["np"](pre) @ w2 + b2)); specs += [(s, p) for p in range(-1, K)]
        return np.hstack(cols), specs

    def link(self, xtr, ytr, xva, yva, allow_new=True):
        cfg = self.cfg; rid = cfg["ridge"]; nva = len(yva); h = nva // 2; E, new = [], []
        Htr, _ = fwd_np(self.theta, xtr, self.nodes); Hva, _ = fwd_np(self.theta, xva, self.nodes)
        for _ in range(cfg["max_links"]):
            F = np.hstack([np.ones((len(ytr), 1)), Htr[:, E]]); Fv = np.hstack([np.ones((nva, 1)), Hva[:, E]])
            Gi = np.linalg.inv(F.T @ F + rid * np.eye(F.shape[1])); beta = Gi @ (F.T @ ytr); r = ytr - F @ beta; pv = Fv @ beta
            if allow_new:
                Zc, specs = self.candidates(xtr, Htr); Zcv, _ = self.candidates(xva, Hva); Z, Zv, sp = Zc, Zcv, specs
                have = [i for i in range(len(self.nodes)) if i not in E]
                Z = np.hstack([Htr[:, have], Z]); Zv = np.hstack([Hva[:, have], Zv]); sp = [("old", i) for i in have] + specs
            else:
                have = [i for i in range(len(self.nodes)) if i not in E]
                if not have: break
                Z, Zv, sp = Htr[:, have], Hva[:, have], [("old", i) for i in have]
            K = Z.shape[1]
            Wc = Gi @ (F.T @ Z); sch = (Z ** 2).sum(0) + rid - (Z * (F @ Wc)).sum(0); cf = (Z.T @ r) / np.maximum(sch, 1e-12)
            err = yva[:, None] - (pv[:, None] + cf * (Zv - Fv @ Wc)); hlp = vbits(err, nva) - float(vbits(yva - pv, nva))
            hA = vbits(err[:h], h) - float(vbits(yva[:h] - pv[:h], h)); hB = vbits(err[h:], nva - h) - float(vbits(yva[h:] - pv[h:], nva - h))
            ok = (hA > 0) & (hB > 0) & (hlp > math.log2(K + 1))
            if not ok.any(): break
            k = int(np.argmax(np.where(ok, hA, -np.inf))); spec = sp[k]
            if spec[0] == "old": j = spec[1]
            else:
                j = self.index.get(spec)
                if j is None:       # a new instance: only a wiring, no new weights; extend the library outputs
                    j = len(self.nodes); self.nodes.append(spec); self.index[spec] = j; new.append(j)
                    Htr, _ = fwd_np(self.theta, xtr, self.nodes); Hva, _ = fwd_np(self.theta, xva, self.nodes)
            E.append(j)
        return E, new

    def refit(self, steps):
        """Joint re-fit of the ONE cell on every stored problem; accepted only if the pooled decision-point error drops."""
        if steps <= 0 or not self.nodes or not self.P: return
        cfg = self.cfg; rid = cfg["ridge"]; R = cfg["refit_rows"]      # refit_rows > 0: each stored problem contributes at most R train and R//2 validation rows to the joint refit (memory: X1.42 hit 1% free RAM at room 3 with all rows)
        PP = [(p[0][:R], p[1][:R], p[2][:max(R // 2, 1)], p[3][:max(R // 2, 1)]) if R else p for p in self.P]
        X = np.vstack([np.vstack([p[0], p[2]]) for p in PP]); off = np.cumsum([0] + [len(p[0]) + len(p[2]) for p in PP])
        Xt = torch.tensor(X); th = [torch.tensor(a, requires_grad=True) for a in self.theta]; nodes = self.nodes; m = X.shape[1]
        Y = [(torch.tensor(p[1]), torch.tensor(p[3]), len(p[0])) for p in PP]

        def run(th):
            W1x, wp, b1, w2, b2 = th; XW = [Xt @ torch.roll(W1x, -s, 0) for s in range(m)]; cols = []
            for k, (s, p) in enumerate(nodes):
                pre = XW[s] + (cols[p][:, None] * wp if p >= 0 else 0.0) + b1; cols.append(torch.tanh(ACT["th"](pre) @ w2 + b2))
            return torch.stack(cols, 1)

        def losses(Hall, fit_beta=True):
            tr, va = 0.0, 0.0
            for i, ((ytr, yva, ntr), E) in enumerate(zip(Y, self.heads)):
                a, b = off[i], off[i + 1]; Fa = torch.cat([torch.ones(b - a, 1, dtype=torch.float64), Hall[a:b][:, E]], 1) if E else torch.ones(b - a, 1, dtype=torch.float64)
                F, Fv = Fa[:ntr], Fa[ntr:]; beta = torch.linalg.solve(F.T @ F + rid * torch.eye(F.shape[1], dtype=torch.float64), (F.T @ ytr).detach() if False else F.T @ ytr)
                tr = tr + torch.mean((F @ beta - ytr) ** 2); va = va + torch.mean((Fv @ beta.detach() - yva) ** 2)
            return tr, va
        with torch.no_grad(): best = float(losses(run(th))[1])
        bestth = [t.detach().clone() for t in th]; opt = torch.optim.Adam(th, lr=cfg["lr"]); since = 0
        for _ in range(steps):
            tr, _ = losses(run(th)); tr = tr + cfg["l1"] * th[3].abs().sum(); opt.zero_grad(); tr.backward(); opt.step()      # l1 on the output weights: units that do not help are switched off (X1.41: the learner chooses its activations)
            with torch.no_grad(): v = float(losses(run(th))[1])
            if v < best - 1e-12: best, bestth, since = v, [t.detach().clone() for t in th], 0
            else:
                since += 1
                if since >= cfg["patience"]: break
        self.theta = [t.numpy().copy() for t in bestth]

    def step(self, d):
        cfg = self.cfg; xtr, ytr = d["tr"][0][:cfg["n_lab"]], d["tr"][1][:cfg["n_lab"]]; xva, yva = d["va"]
        self.P.append((xtr, ytr, xva, yva)); self.heads.append([]); snap = (self.theta, list(self.nodes), dict(self.index))
        E, new = self.link(xtr, ytr, xva, yva)
        if not E:      # the cell is blank or wrong for this problem: propose the best-looking wirings, train on them, keep only if they then help
            Htr, _ = fwd_np(self.theta, xtr, self.nodes)
            Zc, specs = self.candidates(xtr, Htr); r = ytr - ytr.mean(); sd = Zc.std(0) + 1e-9
            rc = np.abs((Zc - Zc.mean(0)).T @ r) / (sd * r.std() * len(r) + 1e-9)
            for k in np.argsort(-rc)[:cfg["propose"]]:
                if specs[k] not in self.index: self.index[specs[k]] = len(self.nodes); self.nodes.append(specs[k])
            self.heads[-1] = [self.index[specs[k]] for k in np.argsort(-rc)[:cfg["propose"]]]
            self.refit(max(cfg["refit"], 60)); self.heads[-1] = []
            E, new = self.link(xtr, ytr, xva, yva)
            if not E: self.theta, self.nodes, self.index = snap[0], snap[1], snap[2]
        self.heads[-1] = E
        if E: self.refit(cfg["refit"])
        return self.score(E, xtr, ytr, xva, yva, d["te"])

    def score(self, E, xtr, ytr, xva, yva, te):
        x, y = te; xf = np.vstack([xtr, xva]); yf = np.concatenate([ytr, yva])
        Hf, _ = fwd_np(self.theta, xf, self.nodes); Ht, _ = fwd_np(self.theta, x, self.nodes)
        F = np.hstack([np.ones((len(yf), 1)), Hf[:, E]]); Ft = np.hstack([np.ones((len(y), 1)), Ht[:, E]])
        b = np.linalg.solve(F.T @ F + self.cfg["ridge"] * np.eye(F.shape[1]), F.T @ yf); return dict(r2=1.0 - float(np.mean((y - Ft @ b) ** 2)) / float(np.var(y)))

    def exam(self, problems):
        out = []
        for d in problems:
            idx = self.rng.choice(len(d["tr"][1]), min(self.cfg["n_lab"], len(d["tr"][1])), replace=False); xtr, ytr = d["tr"][0][idx], d["tr"][1][idx]
            E, _ = self.link(xtr, ytr, d["va"][0], d["va"][1], allow_new=False); out.append((self.score(E, xtr, ytr, d["va"][0], d["va"][1], d["te"])["r2"], len(E)))
        return out

    def recovered(self, xp, V, cols):
        if not self.nodes: return {j: 0.0 for j in cols}
        H, _ = fwd_np(self.theta, xp, self.nodes); Hz = (H - H.mean(0)) / (H.std(0) + 1e-9)
        return {j: float(np.max(np.abs(Hz.T @ ((V[:, j] - V[:, j].mean()) / (V[:, j].std() + 1e-9))) / len(xp))) for j in cols}
