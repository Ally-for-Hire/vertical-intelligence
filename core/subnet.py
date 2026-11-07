"""subnet.py: flat.OPT plus CONCEPT NODES: a node that is itself a tiny 2-layer net, h(x) = tanh(w2 . tanh(x W1 + b1) + b2) (X1.26: one flat node
expresses a 2+-deep concept at R^2 0.15, a tiny 2-layer net at 0.99).

Nodes: ('f', u, b) flat, or ('s', theta) concept node (theta packs W1 [m x H], b1 [H], w2 [H], b2). Candidates per step: n_props flat nodes as in
flat.OPT; concept nodes (x127_n_sub of them, trained on the residual) offered every step ('always'), only when no flat candidate pays ('stag'), or
never ('off' = flat.OPT exactly). Linked nodes of both kinds train together after each problem (Adam through both layers; best decision-error step).
Optional consolidation ('sleep' every x125_every problems): all nodes and readouts trained jointly on all stored problems.
API: Learner(seed, cfg).step(d) -> test bits per point; .kinds() -> counts of node types made and linked."""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import math
import numpy as np
import flat
from flat import vbits, adjusted, B1, B2, EPS, FLOOR

HID = 6                                                                # CONST: x127_hidden
OPT = dict(flat.OPT, sub_mode="off", n_sub=2, sleep_every=0, sleep_steps=100)   # CONST: x127_hidden,x125_every


def unpack(th, m):
    W1 = th[:m * HID].reshape(m, HID); b1 = th[m * HID:m * HID + HID]; w2 = th[m * HID + HID:m * HID + 2 * HID]; b2 = th[-1]
    return W1, b1, w2, b2


def act(nd, x):
    if nd[0] == "f": return np.tanh(x @ nd[1] + nd[2])
    W1, b1, w2, b2 = unpack(nd[1], x.shape[1]); return np.tanh(np.tanh(x @ W1 + b1) @ w2 + b2)


def grad_node(nd, x, dh):
    """Gradient of sum(dh * h(x)) wrt the node's parameters, as a flat vector (flat nodes: [u, b])."""
    if nd[0] == "f":
        h = np.tanh(x @ nd[1] + nd[2]); dz = dh * (1 - h * h); return np.concatenate([x.T @ dz, [dz.sum()]])
    m = x.shape[1]; W1, b1, w2, b2 = unpack(nd[1], m)
    a = np.tanh(x @ W1 + b1); h = np.tanh(a @ w2 + b2); dz = dh * (1 - h * h)
    da = np.outer(dz, w2) * (1 - a * a)
    return np.concatenate([(x.T @ da).ravel(), da.sum(0), a.T @ dz, [dz.sum()]])


def params(nd): return np.concatenate([nd[1], [nd[2]]]) if nd[0] == "f" else nd[1]


def rebuild(nd, p): return ("f", p[:-1].copy(), float(p[-1])) if nd[0] == "f" else ("s", p.copy())


class Cache(flat.Cache):
    def add(self, *nd):
        if len(nd) == 2 and not isinstance(nd[0], str): nd = ("f", nd[0], nd[1])
        if self.n == self._tr.shape[1]:
            self._tr = np.hstack([self._tr, np.zeros_like(self._tr)]); self._va = np.hstack([self._va, np.zeros_like(self._va)])
        self._tr[:, self.n] = act(nd, self.xtr); self._va[:, self.n] = act(nd, self.xva); self.n += 1

    def setnd(self, i, nd): self._tr[:, i] = act(nd, self.xtr); self._va[:, i] = act(nd, self.xva)


def train_sub(rng, x, r, cfg):
    """A concept node fitted to the residual: a * h(x) ~ r, Adam, stall rule as flat nodes."""
    n, m = x.shape; out = []
    for _ in range(cfg["n_sub"]):
        th = np.concatenate([rng.standard_normal(m * HID) / math.sqrt(m), np.zeros(HID), rng.standard_normal(HID) / math.sqrt(HID), [0.0]])
        a = cfg["a0"]; P = [th, np.array([a])]; mo = [np.zeros_like(p) for p in P]; ve = [np.zeros_like(p) for p in P]
        last = float(np.mean(r ** 2))
        for s in range(1, cfg["unit_steps"] + 1):
            nd = ("s", P[0]); h = act(nd, x); e = P[1][0] * h - r
            g = [grad_node(nd, x, e * P[1][0] * (2.0 / n)), np.array([2.0 * np.mean(e * h)])]
            for i in range(2):
                mo[i] = B1 * mo[i] + (1 - B1) * g[i]; ve[i] = B2 * ve[i] + (1 - B2) * g[i] * g[i]
                P[i] = P[i] - cfg["unit_lr"] * (mo[i] / (1 - B1 ** s)) / (np.sqrt(ve[i] / (1 - B2 ** s)) + EPS)
            if s % cfg["check"] == 0:
                cur = float(e @ e) / n
                if cur > last * (1 - cfg["stall"]): break
                last = cur
        out.append(("s", P[0].copy()))
    return out


def fit_head(units, deg, caches, t, E, P, rng, allow_new, cfg, new_from=0, created=None):
    c = caches[t]; nva = len(c.yva); rid = cfg["ridge"]; mode = cfg["sub_mode"]
    for _ in range(cfg["max_steps"]):
        F, Fv, Gi, beta = flat.solve(c, E, rid)
        r = c.ytr - F @ beta; pv = Fv @ beta; b0 = float(vbits(c.yva - pv, nva))
        idx = np.array([i for i in range(new_from, len(units)) if i not in E], int)

        def evaluate(props):
            Z, Zv = c.tr[:, idx], c.va[:, idx]
            if props: Z = np.hstack([Z] + [act(nd, c.xtr)[:, None] for nd in props]); Zv = np.hstack([Zv] + [act(nd, c.xva)[:, None] for nd in props])
            K = Z.shape[1]
            if K == 0: return None
            cost = cfg["cost_scale"] * math.log2(K + 1)
            Wc = Gi @ (F.T @ Z); sch = (Z ** 2).sum(0) + rid - (Z * (F @ Wc)).sum(0); cf = (Z.T @ r) / np.maximum(sch, 1e-12)
            help_ = vbits(c.yva[:, None] - (pv[:, None] + cf * (Zv - Fv @ Wc)), nva) - b0
            isnew = np.zeros(K, bool); isnew[len(idx):] = True; dg = np.zeros(K)
            if len(idx): dg[:len(idx)] = np.asarray(deg, float)[idx]
            v = adjusted(help_, isnew, dg, P, cfg); return v, int(np.argmax(v)), cost
        props = [("f", u, b) for u, b in flat.train_units(rng, c.xtr, r, cfg)] if allow_new else []
        if allow_new and mode == "always": props = props + train_sub(rng, c.xtr, r, cfg)
        res = evaluate(props)
        if (res is None or res[0][res[1]] <= res[2]) and allow_new and mode == "stag":
            props = props + train_sub(rng, c.xtr, r, cfg); res = evaluate(props)
        if res is None: break
        v, k, cost = res
        if v[k] <= cost: break
        if k >= len(idx):
            nd = props[k - len(idx)]; units.append(nd); deg.append(0); j = len(units) - 1
            for cx in caches:
                if cx is not None: cx.add(*nd)
            if created is not None: created.append(j)
        else: j = int(idx[k])
        E.append(j); deg[j] += 1
        while len(E) > 1:
            F, Fv, Gi, beta = flat.solve(c, E, rid); bf = float(vbits(c.yva - Fv @ beta, nva))
            jj = np.arange(1, len(E) + 1)
            Bm = beta[:, None] - Gi[:, jj] * (beta[jj] / Gi[jj, jj])
            hl = bf - vbits(c.yva[:, None] - Fv @ Bm, nva)
            av = adjusted(hl, np.zeros(len(E), bool), np.array([deg[e] - 1 for e in E], float), P, cfg)
            w = int(np.argmin(av))
            if av[w] > cost: break
            deg[E[w]] -= 1; E.pop(w)
    return E


def joint_train(units, groups, cfg, steps, patience):
    """Adam on the nodes used by any group and every group's readout; groups = [(E, xtr, ytr, xva, yva)]; best summed decision error kept."""
    used = sorted({e for g in groups for e in g[0]})
    if not used: return
    Pn = {i: params(units[i]) for i in used}
    betas = []
    for E, x, y, xv, yv in groups:
        H = np.column_stack([act(units[e], x) for e in E]); F = np.hstack([np.ones((len(y), 1)), H])
        betas.append(np.linalg.solve(F.T @ F + cfg["ridge"] * np.eye(len(E) + 1), F.T @ y))
    mo = {i: np.zeros_like(p) for i, p in Pn.items()}; ve = {i: np.zeros_like(p) for i, p in Pn.items()}
    mb = [np.zeros_like(b) for b in betas]; vb = [np.zeros_like(b) for b in betas]
    nd = lambda i: rebuild(units[i], Pn[i])
    val = lambda: sum(float(np.mean((b[0] + np.column_stack([act(nd(e), xv) for e in E]) @ b[1:] - yv) ** 2)) for (E, x, y, xv, yv), b in zip(groups, betas))
    best, bestP, since = val(), {i: p.copy() for i, p in Pn.items()}, 0
    for s in range(1, steps + 1):
        G = {i: np.zeros_like(p) for i, p in Pn.items()}
        for j, (E, x, y, xv, yv) in enumerate(groups):
            b = betas[j]; H = np.column_stack([act(nd(e), x) for e in E]); e_ = (b[0] + H @ b[1:] - y) * (2.0 / len(y))
            gb = np.concatenate([[e_.sum()], H.T @ e_])
            for k, i in enumerate(E): G[i] += grad_node(nd(i), x, e_ * b[1 + k])
            mb[j] = B1 * mb[j] + (1 - B1) * gb; vb[j] = B2 * vb[j] + (1 - B2) * gb * gb
            betas[j] = b - cfg["ft_lr"] * (mb[j] / (1 - B1 ** s)) / (np.sqrt(vb[j] / (1 - B2 ** s)) + EPS)
        for i in Pn:
            mo[i] = B1 * mo[i] + (1 - B1) * G[i]; ve[i] = B2 * ve[i] + (1 - B2) * G[i] * G[i]
            Pn[i] = Pn[i] - cfg["ft_lr"] * (mo[i] / (1 - B1 ** s)) / (np.sqrt(ve[i] / (1 - B2 ** s)) + EPS)
        v = val()
        if v < best: best, bestP, since = v, {i: p.copy() for i, p in Pn.items()}, 0
        else:
            since += 1
            if since >= patience: break
    for i, p in bestP.items(): units[i] = rebuild(units[i], p)
    return used


def head_bits(units, E, d, cfg):
    xtr, ytr = d["tr"]; x, y = d["te"]
    if cfg["refit_all"]: xtr, ytr = np.vstack([xtr, d["va"][0]]), np.concatenate([ytr, d["va"][1]])
    F = np.hstack([np.ones((len(ytr), 1))] + [act(units[i], xtr)[:, None] for i in E]); Ft = np.hstack([np.ones((len(y), 1))] + [act(units[i], x)[:, None] for i in E])
    beta = np.linalg.solve(F.T @ F + cfg["ridge"] * np.eye(F.shape[1]), F.T @ ytr)
    return 0.5 * math.log2(1.0 / max(float(np.mean((y - Ft @ beta) ** 2)), FLOOR))


class Learner:
    def __init__(self, seed, cfg=None):
        self.cfg = dict(OPT, **(cfg or {})); self.rng = np.random.default_rng([seed, flat.STREAM])
        self.units, self.deg, self.heads, self.caches, self.data = [], [], [], [], []

    def _refresh(self, idx):
        for c in self.caches:
            for i in idx: c.setnd(i, self.units[i])

    def step(self, d):
        cfg, units, caches = self.cfg, self.units, self.caches
        t = len(caches); c = Cache(d)
        for nd in units: c.add(*nd)
        caches.append(c); self.data.append(d); n0 = len(units); created = []
        E = fit_head(units, self.deg, caches, t, [], t, self.rng, True, cfg, created=created)
        if E:
            if all(units[i][0] == "f" for i in E):                     # all flat: the exact flat.finetune path (parity with flat.OPT)
                tmp = [(units[i][1], units[i][2]) for i in E]; flat.finetune(tmp, list(range(len(E))), d, cfg)
                for j, i in enumerate(E): units[i] = ("f", tmp[j][0], tmp[j][1])
            else:
                joint_train(units, [(E, d["tr"][0], d["tr"][1], d["va"][0], d["va"][1])], cfg, cfg["ft_steps"], cfg["ft_patience"])
            self._refresh(E)
        if cfg["sweep"] and created:
            for s in range(t): self.heads[s] = fit_head(units, self.deg, caches, s, self.heads[s], t, self.rng, False, cfg, new_from=n0)
        self.heads.append(E)
        if cfg["sleep_every"] and (t + 1) % cfg["sleep_every"] == 0:
            groups = [(Es, dd["tr"][0], dd["tr"][1], dd["va"][0], dd["va"][1]) for Es, dd in zip(self.heads, self.data) if Es]
            used = joint_train(units, groups, cfg, cfg["sleep_steps"], cfg["ft_patience"])
            if used: self._refresh(used)
        return head_bits(units, E, d, cfg)

    def kinds(self):
        made = {"f": 0, "s": 0}; linked = {"f": 0, "s": 0}
        for nd in self.units: made[nd[0]] += 1
        for E in self.heads:
            for e in E: linked[self.units[e][0]] += 1
        return made, linked
