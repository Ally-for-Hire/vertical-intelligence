"""Continuous lifelong learner (v4): ONE online algorithm, no sleep phase, no stored history.
- Bits live in a global registry keyed by NAME; a new problem registers its new names (the network grows) and reuses neurons for shared names.
- Each neuron (one per target bit) has one count table per CONTEXT (problem / physics regime), a fresh table, and shares a table across contexts
  where it does not differ (`alias`). A GLOBAL posterior over contexts is updated every tick from the summed likelihood of all active neurons (fixed-share
  switching rate 1/t); the prediction is the posterior mixture, and the stored-context mass is the TRUST that the past applies now.
- A new context forks when the fresh tables have predicted better than every stored context by more than the code length of naming it and locating
  the change in time (prequential CUSUM, log2 t + log2(R+1) bits). Summing over many neurons makes a real regime change visible in a few ticks while
  local noise stays far below the threshold.
- Tables learn in proportion to the context posterior (online EM), so a table only learns from data it explains: old regimes are not overwritten.
- Structure is maintained incrementally: per tick a few neurons are examined; a candidate parent is accepted when its shadow-table evidence (exact KT
  Bayesian code length, which charges the cells itself) beats the code length of naming it; a parent is dropped when marginalising the table is cheaper.
Memory is O(neurons); nothing is replayed. Every free choice is in constants.py."""
import numpy as np
from scipy.special import gammaln
from constants import C
from core2 import ce_bits, kl_bits

LN2 = np.log(2.0)


def kt_ll(T):
    """exact KT (Jeffreys) log2 marginal likelihood of counts T (..., 2), summed over the leading axes (charges its own parameter cost)"""
    n0, n1 = T[..., 0].astype(np.float64), T[..., 1].astype(np.float64)
    kt = C("kt_prior")
    return float(((gammaln(n0 + kt) + gammaln(n1 + kt) - gammaln(n0 + n1 + 2 * kt) - 2 * gammaln(kt) + gammaln(2 * kt)) / LN2).sum())


class Prob:
    pass


class CNet:
    def __init__(self, seed=0):
        self.k = C("parent_cap")
        self.cells = 2 ** self.k
        self.cap = C("registry_capacity")
        self.R = C("variant_buffer")                       # max stored contexts
        self.M = C("screen_top_m")
        self.PA = C("pair_candidates")
        self.rng = np.random.default_rng(seed)
        cap, K, R, cells, M = self.cap, self.k, self.R, self.cells, self.M
        self.names, self.idx = [], {}
        self.par = np.full((cap, K), -1, np.int32)
        self.npar = np.zeros(cap, np.int32)
        self.cV = np.zeros((cap, R, cells, 2), np.float32)           # table slots; context r reads slot alias[j, r]
        self.alias = np.tile(np.arange(R, dtype=np.int32), (cap, 1))
        self.cF = np.zeros((cap, cells, 2), np.float32)
        self.ctx_used = np.zeros(R, bool)
        self.pi = np.zeros(R + 1)
        self.pi[R] = 1.0
        self.dLg = 0.0
        self.fresh_mass_last = 0.0
        self.nupd = np.zeros(cap, np.int64)
        self.is_target = np.zeros(cap, bool)
        self.C11 = np.zeros((cap, cap), np.float32)
        self.N1 = np.zeros((cap, cap), np.float32)
        self.Y1 = np.zeros(cap)
        self.NW = np.zeros(cap)
        self.scand = np.full((cap, M), -1, np.int32)
        self.SH = np.zeros((cap, M, cells, 2, 2), np.float32)
        self.pcand = np.full((cap, 3 * self.PA, 2), -1, np.int32)
        self.SHP = np.zeros((cap, 3 * self.PA, cells, 2, 2, 2), np.float32)
        self.probs = []
        self.rr = 0
        self.t = 0
        self.forks = self.merges = self.adds = self.prunes = self.absorbs = 0
        self.fork_blocked = 0
        self.cur = None

    # ------------------------------------------------------------------ registry
    def _reg(self, name):
        if name not in self.idx:
            self.idx[name] = len(self.names)
            self.names.append(name)
            assert len(self.names) < self.cap
        return self.idx[name]

    def add_problem(self, key, state_names, act_names, T_max=None):
        p = Prob()
        p.key = key
        p.gin = np.array([self._reg(n) for n in state_names])
        p.gact = np.array([self._reg(n) for n in act_names])
        self.is_target[p.gin] = True
        p.Bs, p.Ba = len(p.gin), len(p.gact)
        p.n = 0
        self.probs.append(p)
        self.cur = len(self.probs) - 1
        nu = int(self.ctx_used.sum())
        if nu:                                              # prior over contexts at the start of a problem: stored share by evidence, fresh 1/(n+1)
            ev = np.array([self.cV[:, r].sum() for r in range(self.R)]) * self.ctx_used
            self.pi[:self.R] = ev / max(ev.sum(), 1e-9) * (1 - 1.0 / (nu + 1))
            self.pi[self.R] = 1.0 / (nu + 1)
            self.dLg = 0.0
        return self.cur

    # ------------------------------------------------------------------ the only phase
    def _w(self, xg, g):
        P = self.par[g]
        w = np.ones((len(g), 1), np.float32)
        for i in range(self.k):
            xi = xg[P[:, i]][:, None]
            w = np.concatenate([w * (1 - xi), w * xi], 1)
        return w

    def _preds(self, g, w):
        """per-context predictions (B, R+1): stored contexts read their slot (falling back to the neuron's best table where it has none); last = fresh"""
        kt = C("kt_prior")
        slots = self.alias[g]                                            # (B,R)
        T = self.cV[g[:, None], slots]                                   # (B,R,cells,2)
        mass = T.sum((2, 3))                                             # (B,R)
        pc = (T[..., 1] + kt) / (T[..., 0] + T[..., 1] + 2 * kt)
        p = np.einsum("bc,brc->br", w, pc)
        best = np.argmax(mass, 1)
        pb = p[np.arange(len(g)), best]
        has_any = mass.max(1) > 0
        p = np.where(mass > 0, p, pb[:, None])
        f = self.cF[g]
        pf = (w * ((f[..., 1] + kt) / (f[..., 0] + f[..., 1] + 2 * kt))).sum(1)
        return np.concatenate([p, pf[:, None]], 1), has_any

    def observe(self, pid, x, y):
        pr = self.probs[pid]
        g = pr.gin
        R = self.R
        xg = np.zeros(self.cap + 1, np.float32)
        xg[pr.gin] = x[:pr.Bs]
        xg[pr.gact] = x[pr.Bs:]
        w = self._w(xg, g)
        pe, has_any = self._preds(g, w)
        # neurons with no stored table anywhere do not vote among stored contexts (their stored predictions equal the fresh one)
        pe[:, :R] = np.where(has_any[:, None], pe[:, :R], pe[:, R:R + 1])
        pi = self.pi
        p = (pi[None, :] * pe).sum(1)
        loss = kl_bits(y, p)
        ce = ce_bits(y[:, None], pe)                                      # (B,R+1)
        L = -ce.sum(0)                                                    # summed log2-likelihood per context
        post = pi * np.power(2.0, L - L.max())
        post /= max(post.sum(), 1e-300)
        trust = np.full(len(g), post[:R].sum())
        s = 1.0 / (self.t + 1.0)
        avail = np.concatenate([self.ctx_used, [True]]).astype(float)
        self.pi = (1 - s) * post + s * avail / avail.sum()
        if self.ctx_used.any():                                           # prequential evidence that the fresh tables beat every stored context (CUSUM)
            Lb = np.max(np.where(self.ctx_used, L[:R], -np.inf))
            self.dLg = max(0.0, self.dLg + L[R] - Lb)
        yv = y.astype(np.float32)
        wy = np.stack([w * (1 - yv)[:, None], w * yv[:, None]], -1)       # (B,cells,2)
        for r in range(R):
            if self.ctx_used[r] and post[r] > 1e-6:
                sl = self.alias[g, r]
                self.cV[g, sl] += np.float32(post[r]) * wy
        self.cF[g] += np.float32(post[R]) * wy
        self.nupd[g] += 1
        wm = np.float32(post[:R].max() if self.ctx_used.any() else 1.0)
        gl = np.concatenate([pr.gin, pr.gact])
        xl = xg[gl]
        sub = np.ix_(gl, g)
        self.C11[sub] += wm * np.outer(xl, yv)
        self.N1[sub] += wm * np.outer(xl, np.ones(len(g), np.float32))
        self.Y1[g] += wm * yv
        self.NW[g] += wm
        wmc = w * wm
        sc = self.scand[g]
        xc = xg[np.where(sc >= 0, sc, self.cap)]
        cvw = np.stack([1 - xc, xc], -1)
        yw = np.stack([1 - yv, yv], -1)
        self.SH[g] += wmc[:, None, :, None, None] * cvw[:, :, None, :, None] * yw[:, None, None, None, :]
        pc = self.pcand[g]
        xa = xg[np.where(pc[..., 0] >= 0, pc[..., 0], self.cap)]
        xcc = xg[np.where(pc[..., 1] >= 0, pc[..., 1], self.cap)]
        aw = np.stack([1 - xa, xa], -1)
        cw = np.stack([1 - xcc, xcc], -1)
        self.SHP[g] += wmc[:, None, :, None, None, None] * aw[:, :, None, :, None, None] * cw[:, :, None, None, :, None] * yw[:, None, None, None, None, :]
        pr.n += 1
        self.t += 1
        self._contexts()
        for _ in range(C("maint_per_tick")):
            self._maintain(pr)
        return loss, trust

    # ------------------------------------------------------------------ contexts (global)
    def _contexts(self):
        R = self.R
        g = self.probs[self.cur].gin
        fm = float(self.cF[g].sum())
        if not self.ctx_used.any():
            if fm >= 1.0:                                                  # first context: the data seen so far
                self.cV[:, 0] = self.cF
                self.ctx_used[0] = True
                self.cF[:] = 0
                self.pi[:] = 0
                self.pi[0] = 1.0 - 1.0 / (self.t + 2.0)
                self.pi[R] = 1.0 / (self.t + 2.0)
                self.dLg = 0.0
                self.fresh_mass_last = 0.0
            return
        cost = np.log2(self.ctx_used.sum() + 1.0) + np.log2(self.t + 1.0)      # name the context + locate the change in time (MDL)
        if self.dLg > cost:
            free = np.nonzero(~self.ctx_used)[0]
            if len(free):
                r = int(free[0])
                self.alias[:, r] = r
                self.cV[:, r] = 0
                self.cV[g, r] = self.cF[g]
                none = self.cF[g].sum((1, 2)) < 1.0                        # neurons with no fresh data inherit the leading context's table
                lead = int(np.argmax(self.pi[:R] * self.ctx_used))
                self.alias[g[none], r] = self.alias[g[none], lead]
                self.ctx_used[r] = True
                self.cF[:] = 0
                self.pi = np.zeros(R + 1)
                self.pi[r] = 1.0 - 1.0 / (self.t + 2.0)
                self.pi[R] = 1.0 / (self.t + 2.0)
                self.dLg = 0.0
                self.fresh_mass_last = 0.0
                self.forks += 1
            else:
                self.fork_blocked += 1
                self.dLg = 0.0
        elif fm >= 2 * self.fresh_mass_last and self.fresh_mass_last > 0:
            b = int(np.argmax(self.pi[:R] * self.ctx_used))            # evidence doubled without a decisive difference: absorb into the leading context
            sl = self.alias[g, b]
            self.cV[g, sl] += self.cF[g]
            self.cF[g] = 0
            self.fresh_mass_last = 0.0
            self.absorbs += 1
        elif self.fresh_mass_last == 0 and fm > 0:
            self.fresh_mass_last = fm

    def _slots(self, j):
        return sorted({int(self.alias[j, r]) for r in range(self.R) if self.ctx_used[r]})

    def _main_slot(self, j):
        sl = self._slots(j)
        return max(sl, key=lambda s: self.cV[j, s].sum()) if sl else None

    def _condense(self, j):
        """neuron j: contexts whose tables are indistinguishable share one slot"""
        us = [r for r in range(self.R) if self.ctx_used[r]]
        if len(us) < 2:
            return
        slots = {}
        for r in us:
            slots.setdefault(int(self.alias[j, r]), []).append(r)
        ss = list(slots)
        for a in range(len(ss)):
            for b in range(a + 1, len(ss)):
                T1, T2 = self.cV[j, ss[a]], self.cV[j, ss[b]]
                if T1.sum() < 1 or T2.sum() < 1:
                    continue
                if kt_ll(T1) + kt_ll(T2) - kt_ll(T1 + T2) < np.log2(len(ss)):
                    self.cV[j, ss[a]] += T2
                    self.cV[j, ss[b]] = 0
                    for r in slots[ss[b]]:
                        self.alias[j, r] = ss[a]
                    self.merges += 1
                    return

    # ------------------------------------------------------------------ incremental structure maintenance
    def _maintain(self, pr):
        g = pr.gin
        j = int(g[self.rr % len(g)])
        self.rr += 1
        if self.nupd[j] < 2 or not self.ctx_used.any():
            return
        self._condense(j)
        self._refresh_candidates(j, pr)
        if not self._prune(j):
            self._grow(j)

    def _refresh_candidates(self, j, pr):
        G = len(self.names)
        N = self.NW[j]
        if N < 2:
            return
        c11, n1 = self.C11[:G, j], self.N1[:G, j]
        y1 = self.Y1[j]
        c10, c01, c00 = n1 - c11, y1 - c11, N - n1 - y1 + c11
        with np.errstate(divide="ignore", invalid="ignore"):
            def xl(n, d):
                return np.where(n > 1e-9, n * np.log2(n / d), 0.0)
            gain = np.nan_to_num(xl(c11, n1) + xl(c10, n1) + xl(c01, N - n1) + xl(c00, N - n1) - xl(y1, N) - xl(N - y1, N))
        for q in self.par[j][self.par[j] >= 0]:
            gain[q] = -1e9
        M = min(self.M, G - 1)
        top = [int(c) for c in np.argpartition(-gain, M)[:M] if gain[c] >= C("screen_min_gain_bits")]
        top.sort(key=lambda c: -gain[c])
        keep = [int(c) for c in self.scand[j] if c >= 0 and c in top]
        new = [c for c in top if c not in keep][:M - len(keep)]
        cur = (keep + new + [-1] * M)[:M]
        old = {int(c): self.SH[j, m].copy() for m, c in enumerate(self.scand[j]) if c >= 0}
        self.SH[j] = 0
        for m, c in enumerate(cur):
            if c >= 0 and c in old and c in keep:
                self.SH[j, m] = old[c]
        self.scand[j] = cur
        pc = []
        for a in pr.gact:
            if int(a) in self.par[j]:
                continue
            for c in top[:self.PA]:
                if c != a:
                    pc.append((int(a), int(c)))
        pc = (pc + [(-1, -1)] * (3 * self.PA))[:3 * self.PA]
        oldp = {tuple(p): self.SHP[j, i].copy() for i, p in enumerate(self.pcand[j]) if p[0] >= 0}
        self.SHP[j] = 0
        for i, p in enumerate(pc):
            if p in oldp:
                self.SHP[j, i] = oldp[p]
        self.pcand[j] = np.array(pc, np.int32)

    def _split(self, j, shadow, nb, main):
        """refine every table of neuron j after adding nb parents. shadow[cell, v, y]: counts observed with the new parent values. The main slot uses
        them where it has data (the dependence that justified the parent is exactly what the table holds); other tables and unseen cells are split by
        the shadow's share of v (no dependence is invented)."""
        p = int(self.npar[j])
        nc = 2 ** p
        kt = C("kt_prior")
        tot = shadow.sum(-1)
        share = (tot + kt) / (tot.sum(1, keepdims=True) + 2 ** nb * kt)
        seen = shadow.sum((1, 2)) >= 1.0
        arrs = [(self.cV[j, s], s == main) for s in range(self.R)] + [(self.cF[j], False)]
        for t, is_main in arrs:
            old = t[:nc].copy()
            t[:] = 0
            for v in range(2 ** nb):
                blk = old * share[:, v][:, None]
                if is_main:
                    blk = np.where(seen[:, None], shadow[:, v, :], blk)
                t[v * nc:(v + 1) * nc] = blk

    def _grow(self, j):
        p = int(self.npar[j])
        if p >= self.k:
            return
        G = len(self.names)
        nc = 2 ** p
        best, bg, kind = None, 0.0, None
        for m, c in enumerate(self.scand[j]):
            if c < 0:
                continue
            T = self.SH[j, m, :nc]
            if float(T.sum()) < 2:
                continue
            g = kt_ll(T) - kt_ll(T.sum(1))
            if g - np.log2(G) > bg:
                best, bg, kind = (m, int(c)), g - np.log2(G), "s"
        if p <= self.k - 2:
            for i, (a, c) in enumerate(self.pcand[j]):
                if a < 0:
                    continue
                T = self.SHP[j, i, :nc]
                if float(T.sum()) < 2:
                    continue
                Tj = T.transpose(0, 2, 1, 3).reshape(nc, 4, 2)            # index a + 2c, the order in which the parents are appended
                g = kt_ll(Tj) - kt_ll(Tj.sum(1))
                if g - 2 * np.log2(G) > bg:
                    best, bg, kind = (i, int(a), int(c)), g - 2 * np.log2(G), "p"
        if best is None:
            return
        main = self._main_slot(j)
        if kind == "s":
            m, c = best
            self._split(j, self.SH[j, m, :nc].copy(), 1, main)
            self.par[j, p] = c
            self.npar[j] += 1
        else:
            i, a, c = best
            self._split(j, self.SHP[j, i, :nc].transpose(0, 2, 1, 3).reshape(nc, 4, 2).copy(), 2, main)
            self.par[j, p], self.par[j, p + 1] = a, c
            self.npar[j] += 2
        self.SH[j] = 0
        self.SHP[j] = 0
        self.scand[j] = -1
        self.pcand[j] = -1
        self.adds += 1

    def _prune(self, j):
        p = int(self.npar[j])
        main = self._main_slot(j)
        if p == 0 or main is None:
            return False
        G = len(self.names)
        Tm = self.cV[j, main, :2 ** p]
        if float(Tm.sum()) < 2:
            return False
        full = kt_ll(Tm)
        best, bv = None, 0.0
        sets = [(i,) for i in range(p)] + [(i, k) for i in range(p) for k in range(i + 1, p)]
        for st in sets:
            g = full - kt_ll(self._marg(Tm, p, st))
            cost = len(st) * np.log2(G)
            if g - cost < bv:
                best, bv = st, g - cost
        if best is None:
            return False
        keep = [i for i in range(p) if i not in best]
        arrs = [self.cV[j, s] for s in range(self.R)] + [self.cF[j]]
        for t in arrs:
            nt = self._marg(t[:2 ** p], p, best)
            t[:] = 0
            t[:len(nt)] = nt
        pl = [int(self.par[j, i]) for i in keep]
        self.par[j, :] = -1
        for q, v in enumerate(pl):
            self.par[j, q] = v
        self.npar[j] = len(pl)
        self.SH[j] = 0
        self.SHP[j] = 0
        self.scand[j] = -1
        self.pcand[j] = -1
        self.prunes += 1
        return True

    @staticmethod
    def _marg(T, p, drop):
        keep = [i for i in range(p) if i not in drop]
        out = np.zeros((2 ** len(keep),) + T.shape[1:], T.dtype)
        for idx in range(2 ** p):
            new = 0
            for q, i in enumerate(keep):
                new |= ((idx >> i) & 1) << q
            out[new] += T[idx]
        return out

    # ------------------------------------------------------------------ readouts
    def complexity(self):
        G = len(self.names)
        tot = 0.0
        for j in np.nonzero(self.is_target)[0]:
            tot += self.npar[j] * np.log2(G)
            for s in self._slots(j):
                cells = max(int((self.cV[j, s].sum(-1) > 0).sum()), 1)
                tot += cells * C("param_cost_bits") * np.log2(self.cV[j, s].sum() + 2.0)
        return float(tot)

    def n_variants(self):
        tg = np.nonzero(self.is_target)[0]
        return int(sum(len(self._slots(j)) for j in tg)), int(len(tg))

    def n_edges(self):
        return int(self.npar.sum())

    def edges(self):
        return {(self.names[q], self.names[j]) for j in np.nonzero(self.is_target)[0] for q in self.par[j, :self.npar[j]]}
