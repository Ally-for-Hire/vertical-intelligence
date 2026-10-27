"""Sleep-time policy analysis: rank variables by how much they help predict the scored outcome of a HELD action, then fit a joint value table
over the chosen variables. Runs only during sleep, from the stored log (the day's runs + history). Numbers come from constants.py."""
import numpy as np
from constants import C
from core import Net

OUT = C("n_outcome_classes")


def _ll_cat(ctx, y, ncell):
    n = np.bincount(ctx * OUT + y, minlength=ncell * OUT).reshape(ncell, OUT).astype(float)
    tot = n.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.where(n > 0, n * np.log2(n / tot), 0).sum())


def held_samples(A, R, h):
    T = len(A)
    ok = np.ones(T - h - 1, bool)
    for k in range(1, h):
        ok &= A[k:T - h - 1 + k] == A[:T - h - 1]
    t0 = np.nonzero(ok)[0]
    cs = np.concatenate([[0], np.cumsum(R)])
    fut = (cs[t0 + h + 1] - cs[t0 + 1]) / h
    return t0, fut


class Table:
    """Joint value table over chosen variables x actions; bins are quantiles of what the agent has actually seen."""
    def __init__(self, var_idx, hist, n_act):
        self.vi = list(var_idx)
        self.nb = C("table_bins").get(len(self.vi), C("rank_bins"))
        self.edges = [np.unique(np.quantile(hist[:, v], np.linspace(0, 1, self.nb + 1)[1:-1])) for v in self.vi]
        self.sz = self.nb ** len(self.vi)
        self.sum = np.zeros((self.sz, n_act))
        self.n = np.zeros((self.sz, n_act))
        self.prior = 0.0

    def state(self, s):
        i = 0
        for v, e in zip(self.vi, self.edges):
            i = i * self.nb + int(min(np.searchsorted(e, s[v]), self.nb - 1))
        return i

    def fit(self, S, A, R, h):
        t0, fut = held_samples(A, R, h)
        self.prior = float(R.mean())
        for t, r in zip(t0, fut):
            i = self.state(S[t])
            self.sum[i, A[t]] += r
            self.n[i, A[t]] += 1
        return len(t0)

    def act(self, s, rng):
        q = (self.sum[self.state(s)] + self.prior * C("table_prior_strength")) / (self.n[self.state(s)] + C("table_prior_strength"))
        return int(np.argmax(q + rng.random(len(q)) * 1e-9))


class TaskRanker:
    """p_u: importance of variable u for the scored outcome; set only in sleep; rate alpha=1/(1+m) with consistency-based m (Net._rate_update)."""
    def __init__(self, names, target="score"):
        self.names = names
        self.cand = [i for i, n in enumerate(names) if n != target]
        self.p = np.full(len(names), C("p_init"))
        self.m = np.zeros(len(names))
        self.hist = []

    def levels(self, S):
        out = np.zeros(S.shape, np.int64)
        for v in self.cand:
            e = np.unique(np.quantile(S[:, v], np.linspace(0, 1, C("rank_bins") + 1)[1:-1]))
            out[:, v] = np.minimum(np.searchsorted(e, S[:, v]), C("rank_bins") - 1)
        return out

    def update(self, t, S, A, R, n_act):
        h = C("hold_ticks")
        if len(A) < 3 * h:
            return
        t0, fut = held_samples(A, R, h)
        if len(t0) < 2:
            return
        y = np.searchsorted(np.quantile(fut, np.linspace(0, 1, OUT + 1)[1:-1]), fut)
        Lv = self.levels(S[:len(A)])[t0]
        ctx = A[t0].astype(np.int64)
        N = len(t0)
        base = _ll_cat(ctx, y, n_act)
        g1 = np.zeros(len(self.names))
        nb = C("rank_bins")
        for u in self.cand:
            g1[u] = max(0.0, _ll_cat(ctx * nb + Lv[:, u], y, n_act * nb) - base) / N
        pos = g1[g1 > 0]
        c = np.median(pos) if len(pos) else 1.0
        r = g1 / (g1 + c) if len(pos) else g1
        a_used = 1.0 / (1.0 + self.m)
        self.p, self.m = Net._rate_update(self.p, self.m, r)
        self.hist.append((t, a_used, self.p.copy()))

    def select(self, S, A, R, n_act):
        """walk variables in the agent's own importance order; keep one only if it adds >= redundancy_ratio of its marginal gain beyond what is chosen"""
        h = C("hold_ticks")
        t0, fut = held_samples(A, R, h)
        y = np.searchsorted(np.quantile(fut, np.linspace(0, 1, OUT + 1)[1:-1]), fut)
        Lv = self.levels(S[:len(A)])[t0]
        ctx0 = A[t0].astype(np.int64)
        nb = C("rank_bins")
        base0 = _ll_cat(ctx0, y, n_act)
        chosen, ctx, ncell, base = [], ctx0, n_act, base0
        for u in [i for i in np.argsort(-self.p) if i in self.cand]:
            if len(chosen) >= C("select_k"):
                break
            marg = _ll_cat(ctx0 * nb + Lv[:, u], y, n_act * nb) - base0
            cond = _ll_cat(ctx * nb + Lv[:, u], y, ncell * nb) - base
            if marg > 0 and cond > C("redundancy_ratio") * marg:
                chosen.append(int(u))
                ctx = ctx * nb + Lv[:, u]
                ncell *= nb
                base = _ll_cat(ctx, y, ncell)
        return chosen
