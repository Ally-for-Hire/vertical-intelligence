"""flat.py: the flat connection learner (X1.9's best arm, 'all linked nodes train'), standalone, config-driven and fast; the base to grow from.

Learner, per problem in a stream:
  1. LINK: greedy, exact measured help (bits added on the problem's decision points, rank-one ridge updates): candidates = every existing node
     + n_props freshly trained nodes (fit to the current residual); value = help x (1 + spec x log2((P+1)/(uses+1))) x (1 + new) for a new node;
     accept the best if it beats cost_scale x log2(#candidates + 1); prune links that stop paying; repeat up to max_steps.
  2. TRAIN: every node this problem links to, and its readout, train by Adam on the problem's train points (best decision-error step kept,
     stop after ft_patience steps without improvement).
  3. RE-CHECK: if new nodes were made, every old problem re-runs the link step over the new nodes (no new nodes).
Scoring: readouts re-fitted (on train points, or train + decision points if refit_all) and scored on test points never used for decisions.
DEFAULT reproduces X1.9 FULLALL / X1.13 FLAT bit-exactly (vi/x1_15.py parity).
API: run(data, gens, seed, cfg) -> dict(after, final, gen, nodes, links). Use cfg=OPT (X1.15) unless reproducing older results."""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import math
import numpy as np

DEFAULT = dict(spec=1.0, new=1.0, max_steps=8, n_props=1, unit_steps=150, unit_lr=0.03, stall=0.01, check=10, a0=0.1,   # CONST: flat_default
               ft_steps=200, ft_patience=10, ft_lr=0.01, ridge=1e-3, cost_scale=1.0, sweep=True, refit_all=False)   # CONST: flat_default
# X1.15 tuned settings (tuning worlds 820000+, confirmed on fresh worlds: +0.30 bits on every measure vs DEFAULT). Start here.   # CONST: flat_default
OPT = dict(DEFAULT, n_props=4, refit_all=True, ridge=3.0, ft_lr=0.03, ft_patience=30, unit_lr=0.01)   # CONST: flat_default
B1, B2, EPS, FLOOR = 0.9, 0.999, 1e-8, 1e-6                            # Adam standard; log floor (same values as vi/constants.py)   # CONST: flat_default
STREAM, GEN_STREAM = 11, 2                                             # random-stream ids used by X1.9 FULLALL and the X1.13 generalization test   # CONST: flat_default


class Cache:
    """Each node's activations on one problem's train and decision points, computed once (preallocated, grows by doubling)."""
    def __init__(self, d):
        self.xtr, self.ytr = d["tr"]; self.xva, self.yva = d["va"]; self.n = 0
        self._tr = np.zeros((len(self.ytr), 16)); self._va = np.zeros((len(self.yva), 16))   # CONST: flat_default

    @property
    def tr(self): return self._tr[:, :self.n]

    @property
    def va(self): return self._va[:, :self.n]

    def add(self, u, b):
        if self.n == self._tr.shape[1]:
            self._tr = np.hstack([self._tr, np.zeros_like(self._tr)]); self._va = np.hstack([self._va, np.zeros_like(self._va)])
        self._tr[:, self.n] = np.tanh(self.xtr @ u + b); self._va[:, self.n] = np.tanh(self.xva @ u + b); self.n += 1

    def set(self, i, u, b):
        self._tr[:, i] = np.tanh(self.xtr @ u + b); self._va[:, i] = np.tanh(self.xva @ u + b)


def vbits(err, n):
    return n * 0.5 * np.log2(1.0 / np.maximum(np.mean(err ** 2, 0), FLOOR))


def solve(c, E, ridge):
    F = np.hstack([np.ones((len(c.ytr), 1)), c.tr[:, E]]); Fv = np.hstack([np.ones((len(c.yva), 1)), c.va[:, E]])
    Gi = np.linalg.inv(F.T @ F + ridge * np.eye(F.shape[1]))
    return F, Fv, Gi, Gi @ (F.T @ c.ytr)


def train_unit(rng, x, r, cfg):
    n, m = x.shape
    p = np.concatenate([rng.standard_normal(m) / math.sqrt(m), [0.0, cfg["a0"]]])
    X1 = np.hstack([x, np.ones((n, 1))])
    mo = np.zeros(m + 2); ve = np.zeros(m + 2); g = np.zeros(m + 2)
    lr = cfg["unit_lr"]; last = float(np.mean(r ** 2)); ck = cfg["check"]; k = 2.0 / n
    for s in range(1, cfg["unit_steps"] + 1):
        h = np.tanh(X1 @ p[:-1]); e = p[-1] * h - r
        g[:-1] = X1.T @ (e * (p[-1] * k) * (1 - h * h)); g[-1] = k * (e @ h)
        mo = B1 * mo + (1 - B1) * g; ve = B2 * ve + (1 - B2) * g * g
        p = p - lr * (mo / (1 - B1 ** s)) / (np.sqrt(ve / (1 - B2 ** s)) + EPS)
        if s % ck == 0:
            cur = float(e @ e) / n
            if cur > last * (1 - cfg["stall"]): break
            last = cur
    return (p[:m], float(p[m]))


def train_units(rng, x, r, cfg):
    """n_props new nodes; n_props = 1 is the exact single-node path; more are trained together (vectorised), each stopping on its own stall."""
    K = cfg["n_props"]
    if K == 1: return [train_unit(rng, x, r, cfg)]
    n, m = x.shape
    P = np.stack([np.concatenate([rng.standard_normal(m) / math.sqrt(m), [0.0, cfg["a0"]]]) for _ in range(K)])
    X1 = np.hstack([x, np.ones((n, 1))])
    mo = np.zeros_like(P); ve = np.zeros_like(P); G = np.zeros_like(P); act = np.ones(K, bool)
    lr = cfg["unit_lr"]; last = np.full(K, float(np.mean(r ** 2))); ck = cfg["check"]; k = 2.0 / n
    for s in range(1, cfg["unit_steps"] + 1):
        H = np.tanh(X1 @ P[:, :-1].T); E = P[:, -1] * H - r[:, None]
        G[:, :-1] = ((E * (P[:, -1] * k)) * (1 - H * H)).T @ X1; G[:, -1] = k * (E * H).sum(0)
        mo = B1 * mo + (1 - B1) * G; ve = B2 * ve + (1 - B2) * G * G
        step = lr * (mo / (1 - B1 ** s)) / (np.sqrt(ve / (1 - B2 ** s)) + EPS)
        P = P - step * act[:, None]
        if s % ck == 0:
            cur = (E * E).sum(0) / n
            act &= ~(cur > last * (1 - cfg["stall"])); last = np.where(act, cur, last)
            if not act.any(): break
    return [(P[i, :m].copy(), float(P[i, m])) for i in range(K)]


def adjusted(help_, isnew, deg, P, cfg):
    spec = np.where(isnew, math.log2(P + 1), np.log2((P + 1) / (deg + 1.0)))
    return np.where(help_ > 0, help_ * (1 + cfg["spec"] * spec) * (1 + np.where(isnew, cfg["new"], 0)), help_)


def fit_head(units, deg, caches, t, E, P, rng, allow_new, cfg, new_from=0, created=None):
    c = caches[t]; nva = len(c.yva); rid = cfg["ridge"]
    for _ in range(cfg["max_steps"]):
        F, Fv, Gi, beta = solve(c, E, rid)
        r = c.ytr - F @ beta; pv = Fv @ beta; b0 = float(vbits(c.yva - pv, nva))
        idx = np.array([i for i in range(new_from, len(units)) if i not in E], int)
        Z, Zv = c.tr[:, idx], c.va[:, idx]
        props = train_units(rng, c.xtr, r, cfg) if allow_new else []
        if props:
            Z = np.hstack([Z] + [np.tanh(c.xtr @ u + b)[:, None] for u, b in props]); Zv = np.hstack([Zv] + [np.tanh(c.xva @ u + b)[:, None] for u, b in props])
        K = Z.shape[1]
        if K == 0: break
        cost = cfg["cost_scale"] * math.log2(K + 1)
        Wc = Gi @ (F.T @ Z); sch = (Z ** 2).sum(0) + rid - (Z * (F @ Wc)).sum(0); cf = (Z.T @ r) / sch
        err = c.yva[:, None] - (pv[:, None] + cf * (Zv - Fv @ Wc))
        help_ = vbits(err, nva) - b0
        isnew = np.zeros(K, bool); isnew[len(idx):] = True; dg = np.zeros(K)
        if len(idx): dg[:len(idx)] = np.asarray(deg, float)[idx]
        if cfg.get("bar") == "confirm":                                    # X1.17: no charge that grows with the number of nodes; instead a link
            h = nva // 2                                                     # chosen on one half of the decision points must ALSO help on the other
            hA = vbits(err[:h], h) - float(vbits(c.yva[:h] - pv[:h], h)); hB = vbits(err[h:], nva - h) - float(vbits(c.yva[h:] - pv[h:], nva - h))
            v = np.where(hB > 0, adjusted(hA, isnew, dg, P, cfg), -np.inf); cost = 0.0
        else:
            v = adjusted(help_, isnew, dg, P, cfg)
        k = int(np.argmax(v))
        if not np.isfinite(v[k]) or v[k] <= cost: break
        if isnew[k]:
            nu = props[k - len(idx)]; units.append(nu); deg.append(0); j = len(units) - 1
            for cx in caches:
                if cx is not None: cx.add(*nu)
            if created is not None: created.append(j)
        else: j = int(idx[k])
        E.append(j); deg[j] += 1
        while len(E) > 1:
            F, Fv, Gi, beta = solve(c, E, rid); bf = float(vbits(c.yva - Fv @ beta, nva))
            jj = np.arange(1, len(E) + 1)
            Bm = beta[:, None] - Gi[:, jj] * (beta[jj] / Gi[jj, jj])
            hl = bf - vbits(c.yva[:, None] - Fv @ Bm, nva)
            av = adjusted(hl, np.zeros(len(E), bool), np.array([deg[e] - 1 for e in E], float), P, cfg)
            w = int(np.argmin(av))
            if av[w] > cost: break
            deg[E[w]] -= 1; E.pop(w)
    return E


def finetune(units, E, d, cfg):
    if not E: return
    (xtr, ytr), (xva, yva) = d["tr"], d["va"]
    n, m = xtr.shape; k = len(E)
    X1, V1 = np.hstack([xtr, np.ones((n, 1))]), np.hstack([xva, np.ones((len(yva), 1))])
    U = np.array([np.concatenate([units[i][0], [units[i][1]]]) for i in E])
    H = np.tanh(X1 @ U.T); F = np.hstack([np.ones((n, 1)), H])
    beta = np.linalg.solve(F.T @ F + cfg["ridge"] * np.eye(k + 1), F.T @ ytr)
    P = [U, beta]; mo = [np.zeros_like(U), np.zeros_like(beta)]; ve = [np.zeros_like(U), np.zeros_like(beta)]
    lr = cfg["ft_lr"]
    val = lambda U_, b_: float(np.mean((b_[0] + np.tanh(V1 @ U_.T) @ b_[1:] - yva) ** 2))
    best, bestP, since = val(U, beta), (U.copy(), beta.copy()), 0
    for s in range(1, cfg["ft_steps"] + 1):
        H = np.tanh(X1 @ P[0].T); e = (P[1][0] + H @ P[1][1:] - ytr) * (2.0 / n)
        gb = np.concatenate([[e.sum()], H.T @ e])
        gU = ((e[:, None] * P[1][1:][None, :]) * (1 - H * H)).T @ X1
        for i, g in enumerate((gU, gb)):
            mo[i] = B1 * mo[i] + (1 - B1) * g; ve[i] = B2 * ve[i] + (1 - B2) * g * g
            P[i] = P[i] - lr * (mo[i] / (1 - B1 ** s)) / (np.sqrt(ve[i] / (1 - B2 ** s)) + EPS)
        v = val(P[0], P[1])
        if v < best: best, bestP, since = v, (P[0].copy(), P[1].copy()), 0
        else:
            since += 1
            if since >= cfg["ft_patience"]: break
    for j, i in enumerate(E): units[i] = (bestP[0][j, :m].copy(), float(bestP[0][j, m]))


def head_bits(units, E, d, cfg):
    xtr, ytr = d["tr"]
    if cfg["refit_all"]: xtr, ytr = np.vstack([xtr, d["va"][0]]), np.concatenate([ytr, d["va"][1]])
    x, y = d["te"]
    act = lambda z: np.tanh(z @ np.array([units[i][0] for i in E]).T + np.array([units[i][1] for i in E])) if E else np.zeros((len(z), 0))
    F = np.hstack([np.ones((len(ytr), 1)), act(xtr)]); Ft = np.hstack([np.ones((len(y), 1)), act(x)])
    beta = np.linalg.solve(F.T @ F + cfg["ridge"] * np.eye(F.shape[1]), F.T @ ytr)
    mse = float(np.mean((y - Ft @ beta) ** 2))
    return len(y) * 0.5 * math.log2(1.0 / max(mse, FLOOR)) / len(y)


def run(data, gens, seed, cfg=None):
    cfg = dict(DEFAULT, **(cfg or {}))
    rng = np.random.default_rng([seed, STREAM])
    units, deg, heads, after, caches = [], [], [], [], [None] * len(data)
    for t, d in enumerate(data):
        caches[t] = Cache(d)
        for u in units: caches[t].add(*u)
        n0 = len(units); created = []
        E = fit_head(units, deg, caches, t, [], t, rng, True, cfg, created=created)
        if E:
            finetune(units, E, d, cfg)
            for c in caches:
                if c is not None:
                    for i in E: c.set(i, *units[i])
        if cfg["sweep"] and created:
            for s in range(t): heads[s] = fit_head(units, deg, caches, s, heads[s], t, rng, False, cfg, new_from=n0)
        heads.append(E); after.append(head_bits(units, E, d, cfg))
    final = [head_bits(units, E, data[s], cfg) for s, E in enumerate(heads)]
    gb, P, grng = [], len(data), np.random.default_rng([seed, GEN_STREAM])
    for d in gens:
        c = Cache(d)
        for u in units: c.add(*u)
        cs = [None] * (P + 1); cs[P] = c
        E = fit_head(list(units), [0] * len(units), cs, P, [], P, grng, False, cfg)
        gb.append(head_bits(units, E, d, cfg))
    return dict(after=after, final=final, gen=gb, nodes=len(units), links=sum(len(E) for E in heads))


class Learner:
    """The same learner as run(), one problem at a time (for open-ended loops). step(d) returns (test bits after learning d, own bits on d's
    decision points: the learner's INTERNAL measure of how well it did, available without any test data)."""
    def __init__(self, seed, cfg=None):
        self.cfg = dict(DEFAULT, **(cfg or {})); self.rng = np.random.default_rng([seed, STREAM])
        self.units, self.deg, self.heads, self.caches, self.data = [], [], [], [], []

    def step(self, d):
        cfg, units, caches = self.cfg, self.units, self.caches
        t = len(self.data); self.data.append(d)
        c = Cache(d)
        for u in units: c.add(*u)
        caches.append(c)
        n0 = len(units); created = []
        E = fit_head(units, self.deg, caches, t, [], t, self.rng, True, cfg, created=created)
        if E:
            finetune(units, E, d, cfg)
            for cx in caches:
                for i in E: cx.set(i, *units[i])
        if cfg["sweep"] and created:
            for s in range(t): self.heads[s] = fit_head(units, self.deg, caches, s, self.heads[s], t, self.rng, False, cfg, new_from=n0)
        self.heads.append(E)
        F, Fv, Gi, beta = solve(c, E, cfg["ridge"])
        own = float(vbits(c.yva - Fv @ beta, len(c.yva))) / len(c.yva)
        return head_bits(units, E, d, cfg), own
