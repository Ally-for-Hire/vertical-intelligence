"""deep.py: flat.OPT plus nodes that can build ON other nodes, grown when progress stagnates (author, 2026-11-02: 'grow it in complexity';
'no penalties except for stagnation').

A node is (u, b, parents, v): h = tanh(u.x + v.h_parents + b); flat nodes have no parents. A deep candidate is trained on the input plus the outputs
of the nodes the current problem already links to ('builds on learned behaviour').
deep_mode:
  "off"    flat.OPT exactly (bit-exact parity, vi/x1_16.py parity)
  "stag"   STAGNATION drives growth: each step first tries the flat candidates; only if none pays (the learner has stagnated) is the search
           escalated to n_props deep candidates built on the linked nodes, scored against the bar for the WHOLE enlarged candidate set; if that
           also fails the problem stops. No other penalty or bonus is added.
  "always" deep candidates are offered at every step alongside the flat ones.
Simplification (disclosed): when linked nodes train, a deep node trains its own weights with its parents' outputs held as inputs (no gradient into
the parents through it); all activations are recomputed afterwards in node order.
API: run(data, gens, seed, cfg) -> dict(after, final, gen, nodes, deep_nodes, deep_links, links, rel_after, rel_gen)"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import math
import numpy as np
import flat
from flat import vbits, adjusted, B1, B2, EPS, FLOOR

OPT = dict(flat.OPT, deep_mode="off")


class Cache(flat.Cache):
    def __init__(self, d, units):
        super().__init__(d); self.units = units

    def add(self, u, b, par=(), v=None):
        if self.n == self._tr.shape[1]:
            self._tr = np.hstack([self._tr, np.zeros_like(self._tr)]); self._va = np.hstack([self._va, np.zeros_like(self._va)])
        self._col(self.n, u, b, par, v); self.n += 1

    def _col(self, i, u, b, par, v):
        if par:
            self._tr[:, i] = np.tanh(self.xtr @ u + self._tr[:, list(par)] @ v + b); self._va[:, i] = np.tanh(self.xva @ u + self._va[:, list(par)] @ v + b)
        else:
            self._tr[:, i] = np.tanh(self.xtr @ u + b); self._va[:, i] = np.tanh(self.xva @ u + b)

    def refresh_from(self, first):
        for i in range(first, self.n): self._col(i, *self.units[i])


def node_act(units, x, need):
    """Activations of the nodes in `need` (and their ancestors) on inputs x; dict index -> column."""
    order, seen, st = [], set(), list(need)
    while st:
        i = st.pop()
        if i in seen: continue
        seen.add(i); st += list(units[i][2])
    out = {}
    for i in sorted(seen):
        u, b, par, v = units[i]
        out[i] = np.tanh(x @ u + (np.column_stack([out[p] for p in par]) @ v if par else 0) + b)
    return out


def props_for(rng, c, r, E, cfg, deep):
    """n_props candidates: flat (input x) or deep (input x + outputs of the linked nodes E)."""
    if not deep: return [(u, b, (), None) for u, b in flat.train_units(rng, c.xtr, r, cfg)]
    par = tuple(sorted(E)); xa = np.hstack([c.xtr, c.tr[:, list(par)]]); m = c.xtr.shape[1]
    return [(w[:m].copy(), b, par, w[m:].copy()) for w, b in flat.train_units(rng, xa, r, cfg)]


def cand_cols(c, nd):
    u, b, par, v = nd
    if par: return np.tanh(c.xtr @ u + c.tr[:, list(par)] @ v + b), np.tanh(c.xva @ u + c.va[:, list(par)] @ v + b)
    return np.tanh(c.xtr @ u + b), np.tanh(c.xva @ u + b)


def fit_head(units, deg, caches, t, E, P, rng, allow_new, cfg, new_from=0, created=None):
    c = caches[t]; nva = len(c.yva); rid = cfg["ridge"]; mode = cfg["deep_mode"]
    for _ in range(cfg["max_steps"]):
        F, Fv, Gi, beta = flat.solve(c, E, rid)
        r = c.ytr - F @ beta; pv = Fv @ beta; b0 = float(vbits(c.yva - pv, nva))
        idx = np.array([i for i in range(new_from, len(units)) if i not in E], int)

        def evaluate(props):
            Z, Zv = c.tr[:, idx], c.va[:, idx]
            if props:
                cols = [cand_cols(c, nd) for nd in props]
                Z = np.hstack([Z] + [a[:, None] for a, _ in cols]); Zv = np.hstack([Zv] + [bv[:, None] for _, bv in cols])
            K = Z.shape[1]
            if K == 0: return None
            cost = cfg["cost_scale"] * math.log2(K + 1)
            Wc = Gi @ (F.T @ Z); sch = (Z ** 2).sum(0) + rid - (Z * (F @ Wc)).sum(0); cf = (Z.T @ r) / sch
            help_ = vbits(c.yva[:, None] - (pv[:, None] + cf * (Zv - Fv @ Wc)), nva) - b0
            isnew = np.zeros(K, bool); isnew[len(idx):] = True; dg = np.zeros(K)
            if len(idx): dg[:len(idx)] = np.asarray(deg, float)[idx]
            v = adjusted(help_, isnew, dg, P, cfg); k = int(np.argmax(v))
            return v, k, cost
        props = props_for(rng, c, r, E, cfg, False) if allow_new else []
        if allow_new and mode == "always" and E: props = props + props_for(rng, c, r, E, cfg, True)
        res = evaluate(props)
        if (res is None or res[0][res[1]] <= res[2]) and allow_new and mode == "stag" and E:   # stagnated: escalate to building on linked nodes
            props = props + props_for(rng, c, r, E, cfg, True); res = evaluate(props)
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
    """flat.finetune generalised: each linked node trains (u, v, b) with its parents' outputs as fixed inputs (from the cache)."""
    if not E: return
    if all(not units[i][2] for i in E):                                  # all flat: the exact flat path (parity)
        U = [(units[i][0], units[i][1]) for i in E]; tmp = list(U)
        flat.finetune(tmp, list(range(len(E))), d, cfg)
        for j, i in enumerate(E): units[i] = (tmp[j][0], tmp[j][1], (), None)
        return
    (xtr, ytr), (xva, yva) = d["tr"], d["va"]; n, m = xtr.shape; k = len(E)
    Xs = [np.hstack([xtr, cache.tr[:, list(units[i][2])], np.ones((n, 1))]) for i in E]
    Vs = [np.hstack([xva, cache.va[:, list(units[i][2])], np.ones((len(yva), 1))]) for i in E]
    W = [np.concatenate([units[i][0], units[i][3] if units[i][2] else [], [units[i][1]]]) for i in E]
    H = np.column_stack([np.tanh(Xs[j] @ W[j]) for j in range(k)]); F = np.hstack([np.ones((n, 1)), H])
    beta = np.linalg.solve(F.T @ F + cfg["ridge"] * np.eye(k + 1), F.T @ ytr)
    mo = [np.zeros_like(w) for w in W] + [np.zeros_like(beta)]; ve = [np.zeros_like(w) for w in W] + [np.zeros_like(beta)]
    val = lambda W_, b_: float(np.mean((b_[0] + np.column_stack([np.tanh(Vs[j] @ W_[j]) for j in range(k)]) @ b_[1:] - yva) ** 2))
    best, bestW, since = val(W, beta), ([w.copy() for w in W], beta.copy()), 0
    for s in range(1, cfg["ft_steps"] + 1):
        H = np.column_stack([np.tanh(Xs[j] @ W[j]) for j in range(k)]); e = (beta[0] + H @ beta[1:] - ytr) * (2.0 / n)
        gs = [Xs[j].T @ (e * beta[1 + j] * (1 - H[:, j] ** 2)) for j in range(k)] + [np.concatenate([[e.sum()], H.T @ e])]
        P = W + [beta]
        for q, g in enumerate(gs):
            mo[q] = B1 * mo[q] + (1 - B1) * g; ve[q] = B2 * ve[q] + (1 - B2) * g * g
            P[q] = P[q] - cfg["ft_lr"] * (mo[q] / (1 - B1 ** s)) / (np.sqrt(ve[q] / (1 - B2 ** s)) + EPS)
        W, beta = P[:k], P[k]
        vv = val(W, beta)
        if vv < best: best, bestW, since = vv, ([w.copy() for w in W], beta.copy()), 0
        else:
            since += 1
            if since >= cfg["ft_patience"]: break
    for j, i in enumerate(E):
        par = units[i][2]; w = bestW[0][j]
        units[i] = (w[:m].copy(), float(w[-1]), par, w[m:m + len(par)].copy() if par else None)


def head_bits(units, E, d, cfg, feats=None):
    xtr, ytr = d["tr"]; x, y = d["te"]
    if cfg["refit_all"]: xtr, ytr = np.vstack([xtr, d["va"][0]]), np.concatenate([ytr, d["va"][1]])
    if feats is None:
        A, At = node_act(units, xtr, E), node_act(units, x, E)
        Ftr = np.column_stack([A[i] for i in E]) if E else np.zeros((len(ytr), 0)); Fte = np.column_stack([At[i] for i in E]) if E else np.zeros((len(y), 0))
        rid = cfg["ridge"]
    else:
        Ftr, Fte = feats; rid = cfg["ceiling_ridge"]
    F = np.hstack([np.ones((len(ytr), 1)), Ftr]); Ft = np.hstack([np.ones((len(y), 1)), Fte])
    beta = np.linalg.solve(F.T @ F + rid * np.eye(F.shape[1]), F.T @ ytr)
    mse = float(np.mean((y - Ft @ beta) ** 2))
    return len(y) * 0.5 * math.log2(1.0 / max(mse, FLOOR)) / len(y)


def ceiling(d, cfg):
    """Bits a readout given the TRUE concepts of this problem reaches with the same examples: the problem's own ceiling."""
    Ctr, Cva, Cte = d["Ctrue"]
    Ftr = np.vstack([Ctr, Cva]) if cfg["refit_all"] else Ctr
    return head_bits(None, None, d, cfg, feats=(Ftr, Cte))


def run(data, gens, seed, cfg):
    cfg = dict(OPT, ceiling_ridge=1e-3, **cfg)
    rng = np.random.default_rng([seed, flat.STREAM])
    units, deg, heads, after, caches = [], [], [], [], [None] * len(data)
    for t, d in enumerate(data):
        caches[t] = Cache(d, units)
        for nd in units: caches[t].add(*nd)
        n0 = len(units); created = []
        E = fit_head(units, deg, caches, t, [], t, rng, True, cfg, created=created)
        if E:
            finetune(units, E, d, cfg, caches[t])
            first = min(E)
            for c in caches:
                if c is not None: c.refresh_from(first)
        if cfg["sweep"] and created:
            for s in range(t): heads[s] = fit_head(units, deg, caches, s, heads[s], t, rng, False, cfg, new_from=n0)
        heads.append(E); after.append(head_bits(units, E, d, cfg))
    final = [head_bits(units, E, data[s], cfg) for s, E in enumerate(heads)]
    gb, P, grng = [], len(data), np.random.default_rng([seed, flat.GEN_STREAM])
    for d in gens:
        c = Cache(d, units)
        for nd in units: c.add(*nd)
        cs = [None] * (P + 1); cs[P] = c
        E = fit_head(list(units), [0] * len(units), cs, P, [], P, grng, False, cfg)
        gb.append(head_bits(units, E, d, cfg))
    ceil_a = [ceiling(d, cfg) for d in data]; ceil_g = [ceiling(d, cfg) for d in gens]
    dn = [i for i, nd in enumerate(units) if nd[2]]
    return dict(after=after, final=final, gen=gb, nodes=len(units), deep_nodes=len(dn), links=sum(len(E) for E in heads),
                deep_links=sum(1 for E in heads for e in E if units[e][2]),
                rel_after=[a / max(cv, FLOOR) for a, cv in zip(after, ceil_a)], rel_gen=[g / max(cv, FLOOR) for g, cv in zip(gb, ceil_g)],
                ceil_after=ceil_a)
