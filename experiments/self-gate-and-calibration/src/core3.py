"""Continuous lifelong learner (v3): ONE online algorithm, no separate sleep phase, no stored history.
Every tick, for the active neurons: predict with a posterior mixture over that neuron's stored VARIANTS (regimes) and a spare FRESH table
(fixed-share switching rate 1/n; the stored-variant mass is the neuron's TRUST g); update every table in proportion to its responsibility
(online EM), so a table only learns from the data it explains; update running sufficient statistics (pair statistics for screening and per-neuron
SHADOW tables for candidate parents). Maintenance is incremental: each tick a few neurons are examined and, by the same MDL rule used before
(gain in code length must beat the added complexity), may gain a parent (from shadow counts), lose one (by marginalising the table, exact), fork the
fresh table into a new variant, or merge variants that are indistinguishable. Memory is O(neurons): nothing is replayed.
Every free choice is declared in constants.py."""
import numpy as np
from constants import C
from scipy.special import gammaln
from core2 import ce_bits, kl_bits, ll_counts

LN2 = np.log(2.0)


def kt_ll(T):
    """exact KT (Bayesian, Jeffreys) log2 marginal likelihood of counts T (..., 2), summed over the leading axes. Charges the parameter cost itself."""
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
        self.NV = C("variant_buffer")
        self.M = C("screen_top_m")
        self.PA = C("pair_candidates")
        self.rng = np.random.default_rng(seed)
        cap, K, NV, cells, M = self.cap, self.k, self.NV, self.cells, self.M
        self.names, self.idx = [], {}
        self.par = np.full((cap, K), -1, np.int32)
        self.npar = np.zeros(cap, np.int32)
        self.cV = np.zeros((cap, NV, cells, 2), np.float32)
        self.used = np.zeros((cap, NV), bool)
        self.cF = np.zeros((cap, cells, 2), np.float32)
        self.pi = np.zeros((cap, NV + 1))
        self.pi[:, NV] = 1.0
        self.nupd = np.zeros(cap, np.int64)
        self.is_target = np.zeros(cap, bool)
        # screening statistics, weighted by the responsibility of the neuron's main variant
        self.C11 = np.zeros((cap, cap), np.float32)
        self.N1 = np.zeros((cap, cap), np.float32)
        self.Y1 = np.zeros(cap)
        self.NW = np.zeros(cap)
        # shadow tables for candidate parents: [cell, candidate value, y]
        self.scand = np.full((cap, M), -1, np.int32)
        self.SH = np.zeros((cap, M, cells, 2, 2), np.float32)
        self.pcand = np.full((cap, 3 * self.PA, 2), -1, np.int32)
        self.SHP = np.zeros((cap, 3 * self.PA, cells, 2, 2, 2), np.float32)
        self.fresh_last = np.zeros(cap)
        self.dL = np.zeros(cap)                 # prequential evidence (bits) that the fresh table predicts better than the main stored variant
        self.main = np.full(cap, NV, np.int32)
        self.probs = []
        self.rr = 0
        self.t = 0
        self.forks = self.merges = self.adds = self.prunes = 0
        self.fork_blocked = 0

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
        for j in p.gin:                                     # a neuron returning to work starts with its stored variants and a fresh share
            nu = int(self.used[j].sum())
            if nu:
                ev = self.cV[j].sum((1, 2)) * self.used[j]
                self.pi[j, :self.NV] = ev / max(ev.sum(), 1e-9) * (1 - 1.0 / (nu + 1))
                self.pi[j, self.NV] = 1.0 / (nu + 1)
        return self.cur

    # ------------------------------------------------------------------ awake = the only phase
    def _w(self, xg, g):
        P = self.par[g]
        w = np.ones((len(g), 1), np.float32)
        for i in range(self.k):
            xi = xg[P[:, i]][:, None]
            w = np.concatenate([w * (1 - xi), w * xi], 1)
        return w

    def observe(self, pid, x, y):
        pr = self.probs[pid]
        g = pr.gin
        NV = self.NV
        xg = np.zeros(self.cap + 1, np.float32)
        xg[pr.gin] = x[:pr.Bs]
        xg[pr.gact] = x[pr.Bs:]
        w = self._w(xg, g)
        kt = C("kt_prior")
        n1, n0 = self.cV[g][..., 1], self.cV[g][..., 0]
        pcV = (n1 + kt) / (n0 + n1 + 2 * kt)
        f1, f0 = self.cF[g][..., 1], self.cF[g][..., 0]
        pcF = (f1 + kt) / (f0 + f1 + 2 * kt)
        pe = np.concatenate([np.einsum("bc,bec->be", w, pcV), (w * pcF).sum(1)[:, None]], 1)
        pi = self.pi[g]
        p = (pi * pe).sum(1)
        loss = kl_bits(y, p)
        post = pi * np.power(2.0, -ce_bits(y[:, None], pe))
        post /= np.maximum(post.sum(1, keepdims=True), 1e-300)
        trust = post[:, :NV].sum(1)
        s = 1.0 / (self.nupd[g] + 1.0)
        avail = np.concatenate([self.used[g], np.ones((len(g), 1), bool)], 1).astype(float)
        self.pi[g] = (1 - s)[:, None] * post + s[:, None] * avail / avail.sum(1, keepdims=True)
        mg = self.main[g]
        hasv = mg < NV
        dl = ce_bits(y, pe[np.arange(len(g)), np.minimum(mg, NV)]) - ce_bits(y, pe[:, NV])
        self.dL[g] = np.where(hasv, np.maximum(0.0, self.dL[g] + dl), 0.0)
        yv = y.astype(np.float32)
        wy = np.stack([w * (1 - yv)[:, None], w * yv[:, None]], -1)                  # (B,cells,2)
        self.cV[g] += post[:, :NV, None, None].astype(np.float32) * wy[:, None]
        self.cF[g] += post[:, NV, None, None].astype(np.float32) * wy
        self.nupd[g] += 1
        # statistics weighted by the main variant's responsibility (all of it when the neuron has no stored variant)
        mj = self.main[g]
        wm = np.where(mj < NV, post[np.arange(len(g)), np.minimum(mj, NV)], post[:, NV]).astype(np.float32)
        gl = np.concatenate([pr.gin, pr.gact])
        xl = xg[gl]
        sub = np.ix_(gl, g)
        self.C11[sub] += np.outer(xl, yv * wm)
        self.N1[sub] += np.outer(xl, wm)
        self.Y1[g] += yv * wm
        self.NW[g] += wm
        wmc = (w * wm[:, None])
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
        self.SHP[g] += (wmc[:, None, :, None, None, None] * aw[:, :, None, :, None, None] * cw[:, :, None, None, :, None] * yw[:, None, None, None, None, :])
        pr.n += 1
        self.t += 1
        for _ in range(C("maint_per_tick")):
            self._maintain(pr)
        return loss, trust

    # ------------------------------------------------------------------ incremental maintenance
    def _maintain(self, pr):
        g = pr.gin
        j = int(g[self.rr % len(g)])
        self.rr += 1
        if self.nupd[j] < 2:
            return
        self._variants(j)
        self._refresh_candidates(j, pr)
        if not self._prune(j):
            self._grow(j)

    def _mass(self, j):
        return float((self.cV[j].sum((1, 2)) * self.used[j]).sum() + self.cF[j].sum())

    def _variants(self, j):
        NV = self.NV
        Tf = self.cF[j]
        nf = float(Tf.sum())
        if nf < 1.0:
            return
        if not self.used[j].any():
            self.cV[j, 0] = Tf
            self.used[j, 0] = True
            self.cF[j] = 0
            self.main[j] = 0
            self.fresh_last[j] = 0.0
            return
        ev = self.cV[j].sum((1, 2)) * self.used[j]
        b = int(np.argmax(ev))
        self.main[j] = b
        Tb = self.cV[j, b]
        cost = np.log2(self.used[j].sum() + 1.0) + np.log2(self.nupd[j] + 1.0)      # name the variant + locate the change in time (MDL)
        if self.dL[j] > cost:                                      # the fresh table has PREDICTED better than the stored one by more than naming cost
            free = np.nonzero(~self.used[j])[0]
            if len(free):
                e = int(free[0])
                self.cV[j, e] = Tf
                self.used[j, e] = True
                self.cF[j] = 0
                self.fresh_last[j] = 0.0
                self.dL[j] = 0.0
                self.forks += 1
            else:
                self.fork_blocked += 1
        elif nf >= 2 * self.fresh_last[j] and self.fresh_last[j] > 0:
            self.cV[j, b] += Tf                                  # evidence has doubled and the fresh table is still indistinguishable: absorb
            self.cF[j] = 0
            self.fresh_last[j] = 0.0
            self.merges += 1
        elif self.fresh_last[j] == 0:
            self.fresh_last[j] = nf
        # merge stored variants that are indistinguishable (condensation)
        us = np.nonzero(self.used[j])[0]
        for a in range(len(us)):
            for c in range(a + 1, len(us)):
                if not (self.used[j, us[a]] and self.used[j, us[c]]):
                    continue
                T1, T2 = self.cV[j, us[a]], self.cV[j, us[c]]
                g12 = kt_ll(T1) + kt_ll(T2) - kt_ll(T1 + T2)
                saved = np.log2(len(us))
                if g12 < saved:
                    self.cV[j, us[a]] += T2
                    self.cV[j, us[c]] = 0
                    self.used[j, us[c]] = False
                    self.merges += 1

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
        gain[j] = gain[j]
        M = min(self.M, G - 1)
        top = [int(c) for c in np.argpartition(-gain, M)[:M] if gain[c] >= C("screen_min_gain_bits")]
        top.sort(key=lambda c: -gain[c])
        keep = [int(c) for c in self.scand[j] if c >= 0 and c in top]
        new = [c for c in top if c not in keep][:M - len(keep)]
        cur = keep + new
        cur = (cur + [-1] * M)[:M]
        # keep accumulated shadows for retained candidates (slots may move): rebuild slot order
        old = {int(c): self.SH[j, m].copy() for m, c in enumerate(self.scand[j]) if c >= 0}
        self.SH[j] = 0
        for m, c in enumerate(cur):
            if c >= 0 and c in old and c in keep:
                self.SH[j, m] = old[c]
        self.scand[j] = cur
        # pair shadows: each action bit of this problem x the 4 best single candidates
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
        """refine every table of neuron j after adding nb parents. shadow[cell, v, y]: counts actually observed with the new parent values.
        The main variant uses the shadow counts where it has data (so the dependence that justified the parent is exactly what the table holds);
        other tables, and cells the shadow never saw, are split by the shadow's share of v (no dependence information is invented)."""
        p = int(self.npar[j])
        nc = 2 ** p
        kt = C("kt_prior")
        tot = shadow.sum(-1)                                             # (cells, v)
        share = (tot + kt) / (tot.sum(1, keepdims=True) + 2 ** nb * kt)
        seen = shadow.sum((1, 2)) >= 1.0
        arrs = [(self.cV[j, e], e == main) for e in range(self.NV)] + [(self.cF[j], False)]
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
            T = self.SH[j, m, :nc]                                  # (cells,2cv,2y)
            mass = float(T.sum())
            if mass < 2:
                continue
            g = kt_ll(T) - kt_ll(T.sum(1))
            pen = np.log2(G)
            if g - pen > bg:
                best, bg, kind = (m, int(c)), g - pen, "s"
        if p <= self.k - 2:
            for i, (a, c) in enumerate(self.pcand[j]):
                if a < 0:
                    continue
                T = self.SHP[j, i, :nc]                              # (cells,2a,2c,2y)
                mass = float(T.sum())
                if mass < 2:
                    continue
                Tj = T.transpose(0, 2, 1, 3).reshape(nc, 4, 2)         # index a + 2c, the order in which the parents are appended
                g = kt_ll(Tj) - kt_ll(Tj.sum(1))
                pen = 2 * np.log2(G)
                if g - pen > bg:
                    best, bg, kind = (i, int(a), int(c)), g - pen, "p"
        if best is None:
            return
        if kind == "s":
            m, c = best
            T = self.SH[j, m, :nc].copy()
            self._split(j, T, 1, int(self.main[j]))
            self.par[j, p] = c
            self.npar[j] += 1
        else:
            i, a, c = best
            T = self.SHP[j, i, :nc].transpose(0, 2, 1, 3).reshape(nc, 4, 2).copy()
            self._split(j, T, 2, int(self.main[j]))
            self.par[j, p], self.par[j, p + 1] = a, c
            self.npar[j] += 2
        self.SH[j] = 0
        self.SHP[j] = 0
        self.scand[j] = -1
        self.pcand[j] = -1
        self.adds += 1

    def _prune(self, j):
        p = int(self.npar[j])
        if p == 0:
            return False
        G = len(self.names)
        ev = self.cV[j].sum((1, 2)) * self.used[j]
        if not self.used[j].any():
            return False
        b = int(np.argmax(ev))
        Tm = self.cV[j, b, :2 ** p]
        N = float(Tm.sum())
        if N < 2:
            return False
        full = kt_ll(Tm)
        best, bv = None, 0.0
        sets = [(i,) for i in range(p)] + [(i, k) for i in range(p) for k in range(i + 1, p)]
        for st in sets:
            Tr = self._marg(Tm, p, st)
            g = full - kt_ll(Tr)                                   # evidence the parents add (KT charges their cells)
            cost = len(st) * np.log2(G)                            # their naming cost
            if g - cost < bv:
                best, bv = st, g - cost
        if best is None:
            return False
        keep = [i for i in range(p) if i not in best]
        arrs = [self.cV[j, e] for e in range(self.NV)] + [self.cF[j]]
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
        """sum a table over the parent bits in `drop`, keeping the rest in order"""
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
            for e in np.nonzero(self.used[j])[0]:
                cells = max(int((self.cV[j, e].sum(-1) > 0).sum()), 1)
                tot += cells * C("param_cost_bits") * np.log2(self.cV[j, e].sum() + 2.0)
        return float(tot)

    def n_variants(self):
        return int(self.used[self.is_target].sum()), int(self.is_target.sum())

    def n_edges(self):
        return int(self.npar.sum())

    def edges(self):
        return {(self.names[q], self.names[j]) for j in np.nonzero(self.is_target)[0] for q in self.par[j, :self.npar[j]]}
