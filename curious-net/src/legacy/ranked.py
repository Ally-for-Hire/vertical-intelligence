"""Ranking-selected value tables: the agent's own importance ranking picks WHICH variables go into a small joint table
(3 bins per variable x 6 actions) that scores 'what happens if I keep this action for h ticks'. Built offline from the logged experiment stream."""
import numpy as np
from env import CarEnv

NBIN = 3


class Table:
    """Joint value table over chosen variables x 6 actions; bins = quantiles of what the agent has actually seen (self-calibrating)."""
    def __init__(self, var_idx, hist, h=10):
        self.vi, self.h = list(var_idx), h
        self.nb = {0: 1, 1: 9, 2: 9, 3: 5}.get(len(self.vi), 3)
        self.edges = [np.unique(np.quantile(hist[:, v], np.linspace(0, 1, self.nb + 1)[1:-1])) for v in self.vi]
        self.sz = self.nb ** len(self.vi)
        self.sum = np.zeros((self.sz, 6)); self.n = np.zeros((self.sz, 6)); self.prior = 0.0

    def state(self, s):
        i = 0
        for v, e in zip(self.vi, self.edges):
            i = i * self.nb + int(min(np.searchsorted(e, s[v]), self.nb - 1))
        return i

    def fit(self, S, A, R):
        T, h = len(A), self.h
        ok = np.ones(T - h - 1, bool)
        for k in range(1, h):
            ok &= A[k:T - h - 1 + k] == A[:T - h - 1]
        t0 = np.nonzero(ok)[0]
        self.prior = float(R.mean())
        cs = np.concatenate([[0], np.cumsum(R)])
        for t in t0:
            r = (cs[t + h + 1] - cs[t + 1]) / h
            i = self.state(S[t]); a = A[t]
            self.sum[i, a] += r; self.n[i, a] += 1
        return len(t0)

    def act(self, s, rng):
        i = self.state(s)
        q = (self.sum[i] + self.prior * 2) / (self.n[i] + 2) + rng.random(6) * 1e-9
        return int(np.argmax(q))


def drive_table(env_like, tab, ticks=600, seed=0):
    env = CarEnv(seed=seed, n_distract=env_like.n_d, drag=env_like.drag, track=env_like.track)
    rng = np.random.default_rng(seed + 9)
    tot = 0.0
    for t in range(ticks):
        env.step(tab.act(env.scalars(), rng))
        tot += env.score
    return tot / ticks


def pick(names, ranks, k, exclude=("drag",), reverse=False):
    """ranks: list of (name, p) best first. keep scalar variables only."""
    cand = [n for n, p in ranks if n in names and n not in exclude]
    if reverse:
        cand = cand[::-1]
    return [names.index(n) for n in cand[:k]]


NCLS = 4


def _ll_cat(ctx, y, ncell):
    n = np.bincount(ctx * NCLS + y, minlength=ncell * NCLS).reshape(ncell, NCLS).astype(float)
    tot = n.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.where(n > 0, n * np.log2(n / tot), 0).sum())


class TaskRanker:
    """Online self-ranking of variables by how much they help predict the scored outcome of a held action.
    gain(u) = bits saved by adding variable u (3 bins) to the context {action} (greedy, MDL-penalised). p_u <- p_u + alpha_n (r_u - p_u):
    starts near 0, volatile early (alpha 0.8), smooth later (alpha floor 0.05)."""
    def __init__(self, names, q, h=10, exclude=("drag", "score"), alpha0=0.8, alpha_min=0.05, tau=3.0):
        self.names, self.q, self.h = names, q, h
        self.cand = [i for i, n in enumerate(names) if n not in exclude]
        self.p = np.full(len(names), 0.01)
        self.n_upd = 0
        self.a0, self.am, self.tau = alpha0, alpha_min, tau
        self.chosen = []
        self.hist = []          # (t, alpha, p copy)

    def levels(self, S):
        out = np.zeros(S.shape, np.int64)
        for v in self.cand:
            e = np.unique(np.quantile(S[:, v], [1 / 3, 2 / 3]))
            out[:, v] = np.minimum(np.searchsorted(e, S[:, v]), 2)
        return out

    def update(self, t, S, A, R):
        T, h = len(A), self.h
        if T < 400:
            return
        ok = np.ones(T - h - 1, bool)
        for k in range(1, h):
            ok &= A[k:T - h - 1 + k] == A[:T - h - 1]
        t0 = np.nonzero(ok)[0]
        if len(t0) < 100:
            return
        cs = np.concatenate([[0], np.cumsum(R)])
        fut = (cs[t0 + h + 1] - cs[t0 + 1]) / h
        edges = np.quantile(fut, [0.25, 0.5, 0.75])
        y = np.searchsorted(edges, fut)
        Lv = self.levels(S[:T])[t0]
        ctx = A[t0].astype(np.int64); ncell = 6
        N = len(t0)
        base = _ll_cat(ctx, y, ncell)
        g1 = np.zeros(len(self.names))
        chosen = []
        for step in range(4):
            best, bg, bi = None, 0.0, None
            for u in self.cand:
                if u in chosen:
                    continue
                g = _ll_cat(ctx * 3 + Lv[:, u], y, ncell * 3) - base
                if step == 0:
                    g1[u] = max(g, 0.0)
                pen = np.log2(len(self.cand)) + ncell * 2 * (NCLS - 1) * 0.5 * np.log2(N)
                if g - pen > bg:
                    best, bg, bi = g, g - pen, u
            if bi is None:
                break
            chosen.append(bi)
            ctx = ctx * 3 + Lv[:, bi]; ncell *= 3
            base = _ll_cat(ctx, y, ncell)
        gu = g1 / N
        pos = gu[gu > 0]
        c = np.median(pos) if len(pos) else 1.0
        r = gu / (gu + c) if len(pos) else gu
        r[[i for i in range(len(self.names)) if i not in self.cand]] = 0
        a = self.am + (self.a0 - self.am) / (1.0 + self.n_upd / self.tau)
        self.p += a * (r - self.p)
        self.n_upd += 1
        self.chosen = chosen
        self.hist.append((t, a, self.p.copy(), list(chosen)))

    def top(self, k=4):
        order = [i for i in np.argsort(-self.p) if i in self.cand]
        return order[:k]


def filtered_select(rk, p, S, A, R, k=3, ratio=0.2):
    """Walk variables in the agent's own importance order p; keep one only if it adds >= ratio of its own marginal gain beyond what is
    already chosen (drops redundant proxies such as lagged or noisy copies)."""
    T, h = len(A), rk.h
    ok = np.ones(T - h - 1, bool)
    for kk in range(1, h):
        ok &= A[kk:T - h - 1 + kk] == A[:T - h - 1]
    t0 = np.nonzero(ok)[0]
    cs = np.concatenate([[0], np.cumsum(R)])
    fut = (cs[t0 + h + 1] - cs[t0 + 1]) / h
    y = np.searchsorted(np.quantile(fut, [0.25, 0.5, 0.75]), fut)
    Lv = rk.levels(S[:T])[t0]
    ctx0 = A[t0].astype(np.int64)
    base0 = _ll_cat(ctx0, y, 6)
    chosen, ctx, ncell = [], ctx0, 6
    base = base0
    for u in [i for i in np.argsort(-p) if i in rk.cand]:
        if len(chosen) >= k:
            break
        marg = _ll_cat(ctx0 * 3 + Lv[:, u], y, 18) - base0
        cond = _ll_cat(ctx * 3 + Lv[:, u], y, ncell * 3) - base
        if marg > 0 and cond > ratio * marg:
            chosen.append(int(u)); ctx = ctx * 3 + Lv[:, u]; ncell *= 3; base = _ll_cat(ctx, y, ncell)
    return chosen
