"""Curious Boolean/soft/qubit network v2: two timescales.
AWAKE (fast, local): count tables with fast counts, trust in permanent memory g=2^-S that slides both ways; NO structure change, NO importance.
ASLEEP (slow, permanent): analyse the stored day + all history; add/remove parents by MDL; create 2-input concept units; rebuild permanent
counts; define importances. Sleep reads stored data only: the world does not advance and nothing new is measured.
Every free choice is declared in constants.py; this file contains no free numbers."""
import numpy as np
from collections import Counter
from constants import C
from encode import ACT_BITS, ACTIONS

LOG2 = np.log(2.0)


def _clip(p):
    eps = C("kl_clip")
    return np.clip(np.asarray(p, np.float64), eps, 1 - eps)


def kl_bits(y, p):
    p = _clip(p)
    y = np.asarray(y, np.float64)
    yc = np.clip(y, 1e-9, 1 - 1e-9)           # NUMERICAL
    return y * np.log2(yc / p) + (1 - y) * np.log2((1 - yc) / (1 - p))


def ce_bits(y, p):
    p = _clip(p)
    y = np.asarray(y, np.float64)
    return -(y * np.log2(p) + (1 - y) * np.log2(1 - p))


def hbin(p):
    p = _clip(p)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class Net:
    def __init__(self, enc, T_max, seed=0, use_trust=True):
        self.use_trust = use_trust      # ablation switch for the sliding trust in permanent memory
        self.enc = enc
        self.Bs = enc.n_state
        self.Bi = self.Bs + ACT_BITS
        self.k = C("parent_cap")
        self.cells = 2 ** self.k
        self.Bt = self.Bs
        self.rng = np.random.default_rng(seed)
        self.par = np.full((self.Bt, self.k), -1, np.int32)          # -1 = dummy (reads the constant-zero slot)
        self.npar = np.zeros(self.Bt, np.int32)
        self.cL = np.zeros((self.Bt, self.cells, 2), np.float32)     # permanent counts: written only in sleep
        self.cF = np.zeros((self.Bt, self.cells, 2), np.float32)     # fast counts: awake self-correction
        self.S = np.zeros(self.Bt)                                   # excess-surprise state per neuron
        self.hid = []                                                # concept units: (a, b, op)
        self.X = np.zeros((T_max, self.Bi), np.float32)
        self.Y = np.zeros((T_max, self.Bt), np.float32)
        self.t = 0
        self.t_sleep = 0
        self.last_sleep_changed = True
        self.t_last_sleep_T = 0
        self.names_in = list(enc.var_of_in)
        self.vnames = sorted(set(self.names_in))
        self.vi = {n: i for i, n in enumerate(self.vnames)}
        self.vid_in = np.array([self.vi[n] for n in self.names_in])
        self.vid_t = np.array([self.vi[n] for n in enc.var_of_bit])
        V = len(self.vnames)
        self.p_unc = np.full(V, C("p_init"))
        self.m_unc = np.zeros(V)
        self.p_hist = []
        self.sleeps = 0
        self.ops_awake = 0.0
        self.ops_sleep = 0.0
        self.cx = np.zeros(self.Bt)
        self._reset_stretch()

    # ------------------------------------------------------------------ inputs with hidden units
    def _hcol(self, X, a, b, op):
        return (X[..., a] * X[..., b]) if op == "AND" else (X[..., a] + X[..., b] - 2 * X[..., a] * X[..., b])

    def ext(self, x):
        h = [self._hcol(x, a, b, op) for a, b, op in self.hid]
        return np.concatenate([x[:self.Bi], np.array(h, np.float32), [0.0]]).astype(np.float32)

    def ext_hist(self, T):
        X = self.X[:T]
        cols = [X] + [self._hcol(X, a, b, op)[:, None] for a, b, op in self.hid]
        return np.concatenate(cols, 1)

    @property
    def Be(self):
        return self.Bi + len(self.hid)

    def _w(self, xe):
        w = np.ones((self.Bt, 1), np.float32)
        for i in range(self.k):
            xi = xe[self.par[:, i]][:, None]                 # par=-1 -> last slot (0)
            w = np.concatenate([w * (1 - xi), w * xi], 1)
        return w

    def trust(self):
        return np.power(2.0, -self.S).astype(np.float32)

    def predict(self, xe):
        w = self._w(xe)
        g = self.trust()[:, None]
        n1 = self.cL[:, :, 1] * g + self.cF[:, :, 1]
        n0 = self.cL[:, :, 0] * g + self.cF[:, :, 0]
        kt = C("kt_prior")
        pc = (n1 + kt) / (n0 + n1 + 2 * kt)
        return (w * pc).sum(1), w, (n0 + n1)

    # ------------------------------------------------------------------ AWAKE
    def observe(self, x, y):
        xe = self.ext(x)
        p, w, _ = self.predict(xe)
        loss = kl_bits(y, p)
        excess = ce_bits(y, p) - hbin(p)                    # >0 when the neuron is more surprised than it expects to be
        if self.use_trust:
            self.S = np.maximum(0.0, self.S + excess)
        self.cF[:, :, 1] += w * y[:, None]
        self.cF[:, :, 0] += w * (1 - y)[:, None]
        t = self.t
        self.X[t] = x[:self.Bi]
        self.Y[t] = y
        self.t += 1
        self.ops_awake += self.Bt * self.k
        self._stretch_add(float(loss.sum()))
        return loss

    def alpha(self):
        """per-neuron learning rate = 1/(1+effective evidence): falls with every correct prediction, rises when trust collapses"""
        n = self.cL.sum((1, 2)) * self.trust() + self.cF.sum((1, 2))
        return 1.0 / (1.0 + n)

    # ---- awake progress test (derived BIC, online)
    def _reset_stretch(self):
        self._n = 0
        self._sx = self._sxx = self._sy = self._sxy = self._syy = 0.0

    def _stretch_add(self, y):
        x = float(self._n)
        self._n += 1
        self._sx += x
        self._sxx += x * x
        self._sy += y
        self._sxy += x * y
        self._syy += y * y

    def awake_flat(self):
        """True when a constant model of per-tick loss beats a sloped one by BIC (no measurable learning progress this stretch)."""
        n = self._n
        if n < 3:
            return False
        mx = self._sx / n
        my = self._sy / n
        sxx = self._sxx - n * mx * mx
        sxy = self._sxy - n * mx * my
        syy = self._syy - n * my * my
        slope = sxy / sxx if sxx > 0 else 0.0
        rss0 = max(syy, 1e-12)
        rss1 = max(syy - slope * sxy, 1e-12)
        bic0 = n * np.log(rss0 / n) + 1 * np.log(n)
        bic1 = n * np.log(rss1 / n) + 2 * np.log(n)
        return bool(bic0 <= bic1 or slope >= 0)

    def should_sleep(self):
        """Derived: (a) awake learning progress is flat (BIC), and (b) the history has doubled since the last sleep (keeps total sleep compute
        within a constant factor of awake compute). The first sleep needs only enough samples for one parent to pay for itself."""
        new = self.t - self.t_sleep
        pen_min = np.log2(self.Be) + C("param_cost_bits") * np.log2(max(new, 2))
        need = max(pen_min, self.t_last_sleep_T)
        return new >= need and self.awake_flat()

    # ------------------------------------------------------------------ curiosity (awake; no importance, no long-term goal)
    def choose_action(self, x, eps=None):
        eps = C("random_floor") if eps is None else eps
        if self.rng.random() < eps:
            return int(self.rng.integers(len(ACTIONS)))
        xe = self.ext(x)
        sc = np.zeros(len(ACTIONS))
        for a in range(len(ACTIONS)):
            xe[self.Bs:self.Bs + ACT_BITS] = ACTIONS[a]
            _, w, n = self.predict(xe)
            n_cell = (w * n).sum(1)
            sc[a] = (C("param_cost_bits") * np.log2(1.0 + 1.0 / (n_cell + 1.0))).sum()      # info gain of one more sample (MDL derivative)
        return int(np.argmax(sc + self.rng.random(len(sc)) * 1e-9))

    # ------------------------------------------------------------------ SLEEP
    @staticmethod
    def _ll(w, y):
        n = w.sum(0)
        n1 = w.T @ y
        n0 = n - n1
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.where(n1 > 1e-9, n1 * np.log2(n1 / n), 0).sum() + np.where(n0 > 1e-9, n0 * np.log2(n0 / n), 0).sum())

    @staticmethod
    def _wm(Xe, plist):
        w = np.ones((Xe.shape[0], 1), np.float32)
        for q in plist:
            xi = Xe[:, q][:, None]
            w = np.concatenate([w * (1 - xi), w * xi], 1)
        return w

    def _plist(self, j):
        return [int(q) for q in self.par[j, :self.npar[j]]]

    def _setp(self, j, plist):
        self.par[j, :] = -1
        for q, v in enumerate(plist):
            self.par[j, q] = v
        self.npar[j] = len(plist)

    def _cx(self, p, N, Be):
        return (2 ** p) * C("param_cost_bits") * np.log2(max(N, 2)) + p * np.log2(Be)

    def complexity(self):
        N = max(self.t, 2)
        self.cx = np.array([self._cx(int(p), N, self.Be) for p in self.npar])
        return float(self.cx.sum())

    def sleep(self):
        T = self.t
        N = T
        Y = self.Y[:T]
        added = removed = units = 0
        ops0 = self.ops_sleep
        while True:                                   # ends when a full pass changes nothing: every accepted step lowers D = code + complexity
            Xe = self.ext_hist(T)
            a = self._grow(Xe, Y, N)
            r = self._prune(Xe, Y, N)
            u = self._concepts(Xe, Y, N)
            added += a
            removed += r
            units += u
            if a + r + u == 0:
                break
        Xe = self.ext_hist(T)
        self._rebuild_all(Xe, Y)
        self._importance(Xe, Y, N)
        self.cF[:] = 0
        self.S[:] = 0
        changed = (added + removed + units) > 0
        self.t_sleep = T
        self.t_last_sleep_T = T
        self.last_sleep_changed = changed
        self.sleeps += 1
        self._reset_stretch()
        self.complexity()
        return dict(added=added, removed=removed, units=units, ops=self.ops_sleep - ops0, changed=changed)

    def _grow(self, Xe, Y, N):
        Be = Xe.shape[1]
        n1 = Xe.sum(0)
        c11 = Xe.T @ Y
        self.ops_sleep += Xe.shape[0] * Be * self.Bt
        y1 = Y.sum(0)[None, :]
        c1 = n1[:, None]
        c10, c01, c00 = c1 - c11, y1 - c11, N - c1 - y1 + c11
        with np.errstate(divide="ignore", invalid="ignore"):
            def xl(n, d):
                return np.where(n > 1e-9, n * np.log2(n / d), 0.0)
            gain = np.nan_to_num(xl(c11, c1) + xl(c10, c1) + xl(c01, N - c1) + xl(c00, N - c1) - xl(y1, N) - xl(N - y1, N))
        for j in range(self.Bt):
            pl = self._plist(j)
            if pl:
                gain[pl, j] = -1e9                      # NUMERICAL: exclude existing parents
        M = min(C("screen_top_m"), Be - 1)
        top = np.argpartition(-gain, M, axis=0)[:M]
        added = 0
        for j in range(self.Bt):
            p = int(self.npar[j])
            if p >= self.k:
                continue
            pl = self._plist(j)
            w = self._wm(Xe, pl)
            yj = Y[:, j]
            base = self._ll(w, yj)
            pen = np.log2(Be) + (2 ** p) * C("param_cost_bits") * np.log2(N)
            best, bg, pair = -1, 0.0, None
            for c in top[:, j]:
                if gain[c, j] < C("screen_min_gain_bits"):
                    continue
                xc = Xe[:, c][:, None]
                g = self._ll(np.concatenate([w * (1 - xc), w * xc], 1), yj) - base
                self.ops_sleep += N * w.shape[1] * 2
                if g - pen > bg:
                    best, bg = int(c), g - pen
            if best < 0 and p <= self.k - 2 and not any(self.Bs <= q < self.Bs + ACT_BITS for q in pl):
                pen2 = 2 * np.log2(Be) + 3 * (2 ** p) * C("param_cost_bits") * np.log2(N)       # joint (action x variable) interaction screen
                for ac in range(self.Bs, self.Bs + ACT_BITS):
                    xa = Xe[:, ac][:, None]
                    wa = np.concatenate([w * (1 - xa), w * xa], 1)
                    for c in top[:, j]:
                        if c == ac or gain[c, j] <= 0:
                            continue
                        xc = Xe[:, c][:, None]
                        g2 = self._ll(np.concatenate([wa * (1 - xc), wa * xc], 1), yj) - base
                        self.ops_sleep += N * wa.shape[1] * 2
                        if g2 - pen2 > bg:
                            best, bg, pair = int(c), g2 - pen2, ac
            if best >= 0 and pair is not None:
                self._setp(j, pl + [pair, best])
                added += 2
            elif best >= 0:
                self._setp(j, pl + [best])
                added += 1
        return added

    def _prune(self, Xe, Y, N):
        """Remove the parent set (single or pair, the same units growth adds) whose loss in code length is less than the complexity it frees."""
        Be = Xe.shape[1]
        removed = 0
        for j in range(self.Bt):
            pl = self._plist(j)
            p = len(pl)
            if not p:
                continue
            yj = Y[:, j]
            full = self._ll(self._wm(Xe, pl), yj)
            best, bv = None, 0.0
            sets = [(i,) for i in range(p)] + [(i, k) for i in range(p) for k in range(i + 1, p)]
            for st in sets:
                rest = [q for i, q in enumerate(pl) if i not in st]
                g = full - self._ll(self._wm(Xe, rest), yj)
                self.ops_sleep += N * (2 ** p) * 2
                cost = len(st) * np.log2(Be) + (2 ** p - 2 ** (p - len(st))) * C("param_cost_bits") * np.log2(N)     # complexity freed
                if g - cost < bv:
                    best, bv = rest, g - cost
            if best is not None:
                self._setp(j, best)
                removed += 1
        return removed

    def _concepts(self, Xe, Y, N):
        """Name a recurring pair of parents as a hidden unit when sharing it lowers total description length (one unit per call)."""
        Be = Xe.shape[1]
        cnt = Counter()
        for j in range(self.Bt):
            pl = [q for q in self._plist(j) if q < self.Bi]
            for a in range(len(pl)):
                for b in range(a + 1, len(pl)):
                    cnt[(min(pl[a], pl[b]), max(pl[a], pl[b]))] += 1
        for (a, b), c in cnt.items():
            if c < 2 or any(h[0] == a and h[1] == b for h in self.hid):
                continue
            for op in C("concept_unit_ops"):
                hcol = self._hcol(Xe, a, b, op)
                Xh = np.concatenate([Xe, hcol[:, None]], 1)
                tot, plan = 0.0, []
                for j in range(self.Bt):
                    pl = self._plist(j)
                    if a in pl and b in pl:
                        rest = [q for q in pl if q not in (a, b)]
                        old = self._ll(self._wm(Xe, pl), Y[:, j]) - self._cx(len(pl), N, Be)
                        new = self._ll(self._wm(Xh, rest + [Be]), Y[:, j]) - self._cx(len(rest) + 1, N, Be)
                        self.ops_sleep += N * (2 ** len(pl)) * 4
                        if new > old:
                            tot += new - old
                            plan.append((j, rest))
                if len(plan) >= 2 and tot > 2 * np.log2(Be):                     # unit definition costs two input names
                    h_idx = self.Bi + len(self.hid)
                    self.hid.append((a, b, op))
                    for j, rest in plan:
                        self._setp(j, rest + [h_idx])
                    return 1
        return 0

    def _rebuild_all(self, Xe, Y):
        self.cL[:] = 0
        for j in range(self.Bt):
            w = self._wm(Xe, self._plist(j))
            n1 = w.T @ Y[:, j]
            n = w.sum(0)
            self.cL[j, :w.shape[1], 1] = n1
            self.cL[j, :w.shape[1], 0] = n - n1
        self.ops_sleep += Xe.shape[0] * self.Bt * self.cells

    def _var_of_ext(self, q):
        if q < self.Bi:
            return [int(self.vid_in[q])]
        a, b, _ = self.hid[q - self.Bi]
        return [int(self.vid_in[a]), int(self.vid_in[b])]

    def _importance(self, Xe, Y, N):
        """Define importances from the day's analysis: bits/sample each variable's edges save (leave-one-out), then
        p <- p + alpha (r - p), alpha = 1/(1+m), m <- (m+1) 2^-excess (Gaussian consistency of the new measurement with p)."""
        G = np.zeros(len(self.vnames))
        for j in range(self.Bt):
            pl = self._plist(j)
            if not pl:
                continue
            yj = Y[:, j]
            full = self._ll(self._wm(Xe, pl), yj)
            cv = self.vid_t[j]
            for i, q in enumerate(pl):
                g = max(0.0, full - self._ll(self._wm(Xe, pl[:i] + pl[i + 1:]), yj)) / N
                self.ops_sleep += N * (2 ** len(pl)) * 2
                vs = self._var_of_ext(q)
                for v in vs:
                    if v != cv:
                        G[v] += g / len(vs)
        pos = G[G > 0]
        c = np.median(pos) if len(pos) else 1.0                      # DERIVED: half-saturation = median positive contribution (scale-free)
        r = G / (G + c) if len(pos) else G
        a_used = 1.0 / (1.0 + self.m_unc)
        self.p_unc, self.m_unc = self._rate_update(self.p_unc, self.m_unc, r)
        self.p_hist.append((self.t, a_used, self.p_unc.copy()))

    @staticmethod
    def _rate_update(p, m, r):
        var = np.clip(p * (1 - p), C("kl_clip"), None) / (m + 1.0)
        z2 = (r - p) ** 2 / var
        excess = np.maximum(0.0, (z2 - 1.0) / (2 * LOG2))            # bits of surprise beyond the expected 1/(2 ln2)
        a = 1.0 / (1.0 + m)
        return p + a * (r - p), (m + 1.0) * np.power(2.0, -excess)

    # ------------------------------------------------------------------ readouts
    def edges(self):
        E = set()
        for j in range(self.Bt):
            for q in self._plist(j):
                for v in self._var_of_ext(q):
                    E.add((self.vnames[v], self.vnames[self.vid_t[j]]))
        return E

    def n_edges(self):
        return int(self.npar.sum())

    def n_neurons(self):
        return int((self.npar > 0).sum())

    def ranking(self):
        return [(self.vnames[i], float(self.p_unc[i])) for i in np.argsort(-self.p_unc)]
