"""comp.py: flat.OPT plus COMPOSITIONAL depth built on ANY existing nodes (X1.23: the plateau is depth; X1.16's deep candidates only saw the current
problem's links).

Node = (u, b, par, v): v None = flat tanh(u.x + b); v array = 'bend' tanh(u.x + v.h_par + b); v == "prod" = h_p * h_q (no parameters).
Extra candidates per step (mode "always": every step; "stag": only when no flat candidate pays, i.e. on stagnation; "off": none = flat.OPT):
  * products of every pair among the x124_topk existing nodes whose outputs best match the current residual (exact, nothing to train);
  * n_props bend nodes trained on [x, outputs of the 2 best-matching existing nodes].
All judged by exact measured help against the same evidence bar (now over the larger candidate set). Linked parametric nodes train as before; product
nodes have no weights; activations are recomputed in node order after every change.
API: run(data, seed, cfg) -> dict(after, final, nodes, prod_nodes, bend_nodes, kinds_linked)"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import math
import numpy as np
import flat
from flat import vbits, adjusted, B1, B2, EPS, FLOOR

TOPK = 5                                                               # CONST: x124_topk
OPT = dict(flat.OPT, deep_mode="off")


class Cache(flat.Cache):
    def __init__(self, d, units):
        super().__init__(d); self.units = units

    def add(self, *nd):
        if self.n == self._tr.shape[1]:
            self._tr = np.hstack([self._tr, np.zeros_like(self._tr)]); self._va = np.hstack([self._va, np.zeros_like(self._va)])
        self._col(self.n, nd); self.n += 1

    def _col(self, i, nd):
        self._tr[:, i], self._va[:, i] = cols(nd, self.xtr, self.xva, self._tr, self._va)

    def refresh(self, idx):
        for i in idx: self._col(i, self.units[i])


def dirty(units, E):
    """The changed nodes and every node built on them, in node order (parents always come first)."""
    D = set(E)
    for i in range(min(E), len(units)):
        if set(norm(units[i])[2]) & D: D.add(i)
    return sorted(D)


def norm(nd):
    return nd if len(nd) == 4 else (nd[0], nd[1], (), None)


def cols(nd, xtr, xva, Htr, Hva):
    u, b, par, v = norm(nd)
    if isinstance(v, str): return Htr[:, par[0]] * Htr[:, par[1]], Hva[:, par[0]] * Hva[:, par[1]]
    if par: return np.tanh(xtr @ u + Htr[:, list(par)] @ v + b), np.tanh(xva @ u + Hva[:, list(par)] @ v + b)
    return np.tanh(xtr @ u + b), np.tanh(xva @ u + b)


def node_act(units, x, need):
    seen, st = set(), list(need)
    while st:
        i = st.pop()
        if i in seen: continue
        seen.add(i); st += list(norm(units[i])[2])
    out = {}
    for i in sorted(seen):
        u, b, par, v = norm(units[i])
        if isinstance(v, str): out[i] = out[par[0]] * out[par[1]]
        elif par: out[i] = np.tanh(x @ u + np.column_stack([out[p] for p in par]) @ v + b)
        else: out[i] = np.tanh(x @ u + b)
    return out


def comp_props(rng, c, r, units, cfg):
    n = len(units)
    if n < 2: return []
    rc = np.abs((c.tr[:, :n] - c.tr[:, :n].mean(0)).T @ (r - r.mean())) / (c.tr[:, :n].std(0) * r.std() * len(r) + 1e-9)
    top = list(np.argsort(-rc)[:TOPK])
    props = [(None, 0.0, (int(min(p, q)), int(max(p, q))), "prod") for a_, p in enumerate(top) for q in top[a_ + 1:]]
    par = tuple(sorted(int(p) for p in top[:2])); xa = np.hstack([c.xtr, c.tr[:, list(par)]]); m = c.xtr.shape[1]
    props += [(w[:m].copy(), b, par, w[m:].copy()) for w, b in flat.train_units(rng, xa, r, cfg)]
    return props


def fit_head(units, deg, caches, t, E, P, rng, allow_new, cfg, new_from=0, created=None):
    c = caches[t]; nva = len(c.yva); rid = cfg["ridge"]; mode = cfg["deep_mode"]
    for _ in range(cfg["max_steps"]):
        F, Fv, Gi, beta = flat.solve(c, E, rid)
        r = c.ytr - F @ beta; pv = Fv @ beta; b0 = float(vbits(c.yva - pv, nva))
        idx = np.array([i for i in range(new_from, len(units)) if i not in E], int)

        def evaluate(props):
            Z, Zv = c.tr[:, idx], c.va[:, idx]
            if props:
                cc = [cols(nd, c.xtr, c.xva, c.tr, c.va) for nd in props]
                Z = np.hstack([Z] + [a[:, None] for a, _ in cc]); Zv = np.hstack([Zv] + [bv[:, None] for _, bv in cc])
            K = Z.shape[1]
            if K == 0: return None
            cost = cfg["cost_scale"] * math.log2(K + 1)
            Wc = Gi @ (F.T @ Z); sch = (Z ** 2).sum(0) + rid - (Z * (F @ Wc)).sum(0); cf = (Z.T @ r) / np.maximum(sch, 1e-12)
            help_ = vbits(c.yva[:, None] - (pv[:, None] + cf * (Zv - Fv @ Wc)), nva) - b0
            isnew = np.zeros(K, bool); isnew[len(idx):] = True; dg = np.zeros(K)
            if len(idx): dg[:len(idx)] = np.asarray(deg, float)[idx]
            v = adjusted(help_, isnew, dg, P, cfg); k = int(np.argmax(v))
            return v, k, cost
        props = [(u, b, (), None) for u, b in flat.train_units(rng, c.xtr, r, cfg)] if allow_new else []
        if allow_new and mode == "always": props = props + comp_props(rng, c, r, units, cfg)
        res = evaluate(props)
        if (res is None or res[0][res[1]] <= res[2]) and allow_new and mode == "stag":
            props = props + comp_props(rng, c, r, units, cfg); res = evaluate(props)
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


def finetune(units, E, d, cfg, cache):
    """Linked parametric nodes train (parents' outputs held as inputs); product nodes are fixed features."""
    if not E: return
    if all(norm(units[i])[2] == () for i in E):                          # all flat: the exact flat path
        tmp = [(units[i][0], units[i][1]) for i in E]; flat.finetune(tmp, list(range(len(E))), d, cfg)
        for j, i in enumerate(E): units[i] = (tmp[j][0], tmp[j][1], (), None)
        return
    (xtr, ytr), (xva, yva) = d["tr"], d["va"]; n, m = xtr.shape; k = len(E)
    trn = [j for j, i in enumerate(E) if not isinstance(norm(units[i])[3], str)]
    Xs, Vs, W = {}, {}, {}
    for j in trn:
        u, b, par, v = norm(units[E[j]])
        Xs[j] = np.hstack([xtr, cache.tr[:, list(par)], np.ones((n, 1))]); Vs[j] = np.hstack([xva, cache.va[:, list(par)], np.ones((len(yva), 1))])
        W[j] = np.concatenate([u, v if par else [], [b]])
    fixed_tr = {j: cache.tr[:, E[j]].copy() for j in range(k) if j not in W}; fixed_va = {j: cache.va[:, E[j]].copy() for j in range(k) if j not in W}
    Htr = lambda W_: np.column_stack([np.tanh(Xs[j] @ W_[j]) if j in W_ else fixed_tr[j] for j in range(k)])
    Hva = lambda W_: np.column_stack([np.tanh(Vs[j] @ W_[j]) if j in W_ else fixed_va[j] for j in range(k)])
    H = Htr(W); F = np.hstack([np.ones((n, 1)), H]); beta = np.linalg.solve(F.T @ F + cfg["ridge"] * np.eye(k + 1), F.T @ ytr)
    mo = {j: np.zeros_like(W[j]) for j in W}; ve = {j: np.zeros_like(W[j]) for j in W}; mb = np.zeros_like(beta); vb = np.zeros_like(beta)
    val = lambda W_, b_: float(np.mean((b_[0] + Hva(W_) @ b_[1:] - yva) ** 2))
    best, bestW, since = val(W, beta), {j: w.copy() for j, w in W.items()}, 0
    for s in range(1, cfg["ft_steps"] + 1):
        H = Htr(W); e = (beta[0] + H @ beta[1:] - ytr) * (2.0 / n)
        gb = np.concatenate([[e.sum()], H.T @ e])
        for j in W:
            g = Xs[j].T @ (e * beta[1 + j] * (1 - H[:, j] ** 2))
            mo[j] = B1 * mo[j] + (1 - B1) * g; ve[j] = B2 * ve[j] + (1 - B2) * g * g
            W[j] = W[j] - cfg["ft_lr"] * (mo[j] / (1 - B1 ** s)) / (np.sqrt(ve[j] / (1 - B2 ** s)) + EPS)
        mb = B1 * mb + (1 - B1) * gb; vb = B2 * vb + (1 - B2) * gb * gb
        beta = beta - cfg["ft_lr"] * (mb / (1 - B1 ** s)) / (np.sqrt(vb / (1 - B2 ** s)) + EPS)
        vv = val(W, beta)
        if vv < best: best, bestW, since = vv, {j: w.copy() for j, w in W.items()}, 0
        else:
            since += 1
            if since >= cfg["ft_patience"]: break
    for j, w in bestW.items():
        par = norm(units[E[j]])[2]
        units[E[j]] = (w[:m].copy(), float(w[-1]), par, w[m:m + len(par)].copy() if par else None)


def head_bits(units, E, d, cfg):
    xtr, ytr = d["tr"]; x, y = d["te"]
    if cfg["refit_all"]: xtr, ytr = np.vstack([xtr, d["va"][0]]), np.concatenate([ytr, d["va"][1]])
    A, At = node_act(units, xtr, E), node_act(units, x, E)
    Ftr = np.column_stack([A[i] for i in E]) if E else np.zeros((len(ytr), 0)); Fte = np.column_stack([At[i] for i in E]) if E else np.zeros((len(y), 0))
    F = np.hstack([np.ones((len(ytr), 1)), Ftr]); Ft = np.hstack([np.ones((len(y), 1)), Fte])
    beta = np.linalg.solve(F.T @ F + cfg["ridge"] * np.eye(F.shape[1]), F.T @ ytr)
    return 0.5 * math.log2(1.0 / max(float(np.mean((y - Ft @ beta) ** 2)), FLOOR))


class Learner:
    def __init__(self, seed, cfg):
        self.cfg = dict(OPT, **cfg); self.rng = np.random.default_rng([seed, flat.STREAM])
        self.units, self.deg, self.heads, self.caches = [], [], [], []

    def step(self, d):
        cfg, units, caches = self.cfg, self.units, self.caches
        t = len(caches); c = Cache(d, units)
        for nd in units: c.add(*nd)
        caches.append(c); n0 = len(units); created = []
        E = fit_head(units, self.deg, caches, t, [], t, self.rng, True, cfg, created=created)
        if E:
            finetune(units, E, d, cfg, c)
            D = dirty(units, E)
            for cx in caches: cx.refresh(D)
        if cfg["sweep"] and created:
            for s in range(t): self.heads[s] = fit_head(units, self.deg, caches, s, self.heads[s], t, self.rng, False, cfg, new_from=n0)
        self.heads.append(E)
        return head_bits(units, E, d, cfg)

    def kinds(self):
        k = {"flat": 0, "bend": 0, "prod": 0}
        for nd in self.units:
            v = norm(nd)[3]; k["prod" if isinstance(v, str) else ("bend" if v is not None else "flat")] += 1
        lk = {"flat": 0, "bend": 0, "prod": 0}
        for E in self.heads:
            for e in E:
                v = norm(self.units[e])[3]; lk["prod" if isinstance(v, str) else ("bend" if v is not None else "flat")] += 1
        return k, lk
