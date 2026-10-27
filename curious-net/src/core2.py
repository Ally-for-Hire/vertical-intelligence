"""Lifelong network (v3). One learner applied to a sequence of different problems.
- Bits live in a global registry keyed by NAME ('<variable>#<role>', 'grid#i', action-bit names). A new problem registers new names (the network
  grows) and reuses neurons for names it shares. Sharing a name is an offer of shared meaning: trust decides if it is accepted.
- Each neuron (one per target bit) has a library of permanent table VARIANTS (one per regime it has seen) plus a fresh table. Awake, the
  prediction is a posterior mixture over {variants, fresh} with fixed-share switching rate 1/n (derived; no constants). The total posterior
  weight on stored variants is the neuron's TRUST g: the derived signal for 'is past data relevant here'.
- Sleep (stored data only): per neuron, fork a new variant if the day's fresh table differs from its best stored match by more than the code
  length of a new table, otherwise absorb it; merge variants that are statistically indistinguishable (condensation); then MDL grow/prune of the
  shared parent sets over all problems' history (variant-aware), and rebuild of the permanent tables.
All free choices are in constants.py."""
import numpy as np
from constants import C

LN2 = np.log(2.0)


def _clipp(p):
    e = C("kl_clip")
    return np.clip(np.asarray(p, np.float64), e, 1 - e)


def ce_bits(y, p):
    p = _clipp(p)
    y = np.asarray(y, np.float64)
    return -(y * np.log2(p) + (1 - y) * np.log2(1 - p))


def kl_bits(y, p):
    p = _clipp(p)
    y = np.asarray(y, np.float64)
    yc = np.clip(y, 1e-9, 1 - 1e-9)
    return y * np.log2(yc / p) + (1 - y) * np.log2((1 - yc) / (1 - p))


def ll_counts(T):
    """plug-in code length (bits, as a log-likelihood) of counts T (..., cells, 2)"""
    n = T.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(T[..., 1] > 1e-9, T[..., 1] * np.log2(T[..., 1] / n), 0.0)
        b = np.where(T[..., 0] > 1e-9, T[..., 0] * np.log2(T[..., 0] / n), 0.0)
    return (a + b).sum(-1)


class Prob:
    pass


class LNet:
    def __init__(self, seed=0, T_hint=None):
        self.k = C("parent_cap")
        self.cells = 2 ** self.k
        self.cap = C("registry_capacity")
        self.NV = C("variant_buffer")
        self.rng = np.random.default_rng(seed)
        self.names, self.idx = [], {}
        cap, K, NV, cells = self.cap, self.k, self.NV, self.cells
        self.par = np.full((cap, K), -1, np.int32)
        self.npar = np.zeros(cap, np.int32)
        self.cV = np.zeros((cap, NV, cells, 2), np.float32)
        self.used = np.zeros((cap, NV), bool)
        self.cF = np.zeros((cap, cells, 2), np.float32)
        self.cDay = np.zeros((cap, NV + 1, cells, 2), np.float32)
        self.pi = np.zeros((cap, NV + 1))
        self.pi[:, NV] = 1.0
        self.nupd = np.zeros(cap, np.int64)
        self.day_n = np.zeros(cap, np.int64)
        self.is_target = np.zeros(cap, bool)
        self.probs = []
        self.t_sleep_total = 0
        self.t_total = 0
        self.sleeps = 0
        self.fork_blocked = 0
        self._reset_stretch()
        self.cur = None
        self.rows_at_sleep = 0

    # ------------------------------------------------------------------ registry / problems
    def _reg(self, name):
        if name not in self.idx:
            self.idx[name] = len(self.names)
            self.names.append(name)
            assert len(self.names) < self.cap, "registry_capacity exceeded"
        return self.idx[name]

    def add_problem(self, key, state_names, act_names, T_max):
        p = Prob()
        p.key = key
        p.gin = np.array([self._reg(n) for n in state_names])
        p.gact = np.array([self._reg(n) for n in act_names])
        self.is_target[p.gin] = True
        p.Bs, p.Ba = len(p.gin), len(p.gact)
        p.X = np.zeros((T_max, p.Bs + p.Ba), np.float32)
        p.Y = np.zeros((T_max, p.Bs), np.float32)
        p.Z = np.zeros((T_max, p.Bs), np.int8)
        p.n = 0
        p.n_sleep = 0
        p.g2l = np.full(self.cap + 1, -1, np.int32)
        p.g2l[np.concatenate([p.gin, p.gact])] = np.arange(p.Bs + p.Ba)
        p.t2l = np.full(self.cap + 1, -1, np.int32)
        p.t2l[p.gin] = np.arange(p.Bs)
        self.probs.append(p)
        self.new_day_priors(p.gin)
        return len(self.probs) - 1

    def start_problem(self, pid):
        """the sleep schedule (history-doubling) restarts for a new problem; the learner is NOT told anything else about it"""
        self.cur = pid
        self.rows_at_sleep = 0
        self._reset_stretch()

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
        n = self._n
        if n < 3:
            return False
        mx, my = self._sx / n, self._sy / n
        sxx, sxy, syy = self._sxx - n * mx * mx, self._sxy - n * mx * my, self._syy - n * my * my
        slope = sxy / sxx if sxx > 0 else 0.0
        rss0, rss1 = max(syy, 1e-12), max(syy - slope * sxy, 1e-12)
        return bool(n * np.log(rss0 / n) + np.log(n) <= n * np.log(rss1 / n) + 2 * np.log(n) or slope >= 0)

    def should_sleep(self):
        pr = self.probs[self.cur]
        new = pr.n - self.rows_at_sleep
        pen_min = np.log2(len(self.names)) + C("param_cost_bits") * np.log2(max(new, 2))
        return new >= max(pen_min, self.rows_at_sleep) and self.awake_flat()

    def new_day_priors(self, g):
        """posterior prior at the start of a day: stored variants share mass in proportion to their evidence, fresh gets 1/(n_used+1)"""
        for j in g:
            u = self.used[j]
            if u.any():
                ev = self.cV[j].sum((1, 2)) * u
                share = ev / max(ev.sum(), 1e-9)
                nu = u.sum()
                self.pi[j, :self.NV] = share * (1 - 1.0 / (nu + 1))
                self.pi[j, self.NV] = 1.0 / (nu + 1)
            else:
                self.pi[j, :] = 0
                self.pi[j, self.NV] = 1.0

    # ------------------------------------------------------------------ awake
    def _w(self, xg, g):
        P = self.par[g]
        w = np.ones((len(g), 1), np.float32)
        for i in range(self.k):
            xi = xg[P[:, i]][:, None]
            w = np.concatenate([w * (1 - xi), w * xi], 1)
        return w

    def _xg(self, pr, x):
        xg = np.zeros(self.cap + 1, np.float32)          # last slot stays 0: par=-1 reads it
        xg[pr.gin] = x[:pr.Bs]
        xg[pr.gact] = x[pr.Bs:]
        return xg

    def _experts(self, g, w):
        kt = C("kt_prior")
        n1, n0 = self.cV[g][..., 1], self.cV[g][..., 0]
        pcV = (n1 + kt) / (n0 + n1 + 2 * kt)                             # (B,NV,cells)
        f1, f0 = self.cF[g][..., 1], self.cF[g][..., 0]
        pcF = (f1 + kt) / (f0 + f1 + 2 * kt)
        pV = np.einsum("bc,bec->be", w, pcV)
        pF = (w * pcF).sum(1)[:, None]
        return np.concatenate([pV, pF], 1)

    def observe(self, pid, x, y):
        pr = self.probs[pid]
        g = pr.gin
        xg = self._xg(pr, x)
        w = self._w(xg, g)
        pe = self._experts(g, w)                                          # (B,NV+1)
        pi = self.pi[g]
        p = (pi * pe).sum(1)
        loss = kl_bits(y, p)
        post = pi * np.power(2.0, -ce_bits(y[:, None], pe))
        post /= np.maximum(post.sum(1, keepdims=True), 1e-300)
        trust = post[:, :self.NV].sum(1)
        s = 1.0 / (self.nupd[g] + 1.0)
        avail = np.concatenate([self.used[g], np.ones((len(g), 1), bool)], 1).astype(float)
        self.pi[g] = (1 - s)[:, None] * post + s[:, None] * avail / avail.sum(1, keepdims=True)
        wy = np.stack([w * (1 - y)[:, None], w * y[:, None]], -1)         # (B,cells,2)
        self.cF[g] += wy
        self.cDay[g] += post[:, :, None, None] * wy[:, None]
        self.nupd[g] += 1
        self.day_n[g] += 1
        t = pr.n
        pr.X[t] = x
        pr.Y[t] = y
        pr.Z[t] = np.argmax(post, 1)
        pr.n += 1
        self.t_total += 1
        self._stretch_add(float(loss.sum()))
        return loss, trust

    def choose_action(self, pid, x, actions, eps=None):
        eps = C("random_floor") if eps is None else eps
        if self.rng.random() < eps:
            return int(self.rng.integers(len(actions)))
        pr = self.probs[pid]
        g = pr.gin
        xg = self._xg(pr, x)
        sc = np.zeros(len(actions))
        nE = np.concatenate([self.cV[g].sum(-1), self.cF[g].sum(-1)[:, None]], 1)   # (B,NV+1,cells)
        for a in range(len(actions)):
            xg[pr.gact] = actions[a]
            w = self._w(xg, g)
            n_cell = np.einsum("be,bec,bc->b", self.pi[g], nE, w)
            sc[a] = (C("param_cost_bits") * np.log2(1.0 + 1.0 / (n_cell + 1.0))).sum()
        return int(np.argmax(sc + self.rng.random(len(sc)) * 1e-9))

    # ------------------------------------------------------------------ SLEEP
    def sleep(self):
        info = dict(forks=0, merges=0, added=0, removed=0)
        self._variants(info)
        while True:
            a = self._grow()
            r = self._prune()
            info["added"] += a
            info["removed"] += r
            if a + r == 0:
                break
        self._rebuild_changed()
        for pr in self.probs:
            pr.n_sleep = pr.n
        self.day_n[:] = 0
        self.cF[:] = 0
        self.cDay[:] = 0
        for pr in self.probs:
            self.new_day_priors(pr.gin)
        self.sleeps += 1
        self.t_sleep_total = self.t_total
        self.rows_at_sleep = self.probs[self.cur].n if self.cur is not None else 0
        self._reset_stretch()
        return info

    # ---- variant fork / absorb / merge (per neuron, from the day's responsibilities)
    def _variants(self, info):
        NV = self.NV
        for j in np.nonzero(self.day_n > 0)[0]:
            rel = {}                                              # fresh code -> final variant id
            for e in range(NV):
                if self.used[j, e]:
                    self.cV[j, e] += self.cDay[j, e]
            Tf = self.cDay[j, NV]
            nf = float(Tf.sum())
            if nf < 1.0:
                continue
            if not self.used[j].any():
                self.cV[j, 0] = Tf
                self.used[j, 0] = True
                dest = 0
            else:
                ev = self.cV[j].sum((1, 2)) * self.used[j]
                b = int(np.argmax(ev))
                Tb = self.cV[j, b]
                gain = ll_counts(Tf) + ll_counts(Tb) - ll_counts(Tf + Tb)
                cells_f = int((Tf.sum(-1) > 0).sum())
                cost = cells_f * C("param_cost_bits") * np.log2(nf + 1.0) + np.log2(self.used[j].sum() + 1.0)
                free = np.nonzero(~self.used[j])[0]
                if gain > cost and len(free):
                    dest = int(free[0])
                    self.cV[j, dest] = Tf
                    self.used[j, dest] = True
                    info["forks"] += 1
                else:
                    if gain > cost:
                        self.fork_blocked += 1
                    self.cV[j, b] += Tf
                    dest = b
            for pr in self.probs:
                lj = pr.t2l[j]
                if lj >= 0 and pr.n > pr.n_sleep:
                    z = pr.Z[pr.n_sleep:pr.n, lj]
                    z[z == NV] = dest
                    pr.Z[pr.n_sleep:pr.n, lj] = z
            self._merge(j, info)

    def _merge(self, j, info):
        NV = self.NV
        while True:
            us = np.nonzero(self.used[j])[0]
            best = None
            for a in range(len(us)):
                for b in range(a + 1, len(us)):
                    T1, T2 = self.cV[j, us[a]], self.cV[j, us[b]]
                    sep = ll_counts(T1) + ll_counts(T2)
                    gain = sep - ll_counts(T1 + T2)
                    saved = float((T2.sum(-1) > 0).sum()) * C("param_cost_bits") * np.log2(T2.sum() + 1.0) + np.log2(len(us))
                    if gain < saved and (best is None or gain - saved < best[0]):
                        best = (gain - saved, us[a], us[b])
            if best is None:
                return
            _, e1, e2 = best
            self.cV[j, e1] += self.cV[j, e2]
            self.cV[j, e2] = 0
            self.used[j, e2] = False
            for pr in self.probs:
                lj = pr.t2l[j]
                if lj >= 0:
                    z = pr.Z[:pr.n, lj]
                    z[z == e2] = e1
                    pr.Z[:pr.n, lj] = z
            info["merges"] += 1

    # ---- data access for structure search (variant-aware)
    def _rows(self, j, cols):
        Xs, ys, zs = [], [], []
        for pr in self.probs:
            lj = pr.t2l[j]
            if lj < 0 or pr.n == 0:
                continue
            loc = pr.g2l[np.asarray(cols, int)] if len(cols) else np.zeros(0, int)
            blk = np.zeros((pr.n, len(cols)), np.float32)
            ok = loc >= 0
            if ok.any():
                blk[:, ok] = pr.X[:pr.n][:, loc[ok]]
            Xs.append(blk)
            ys.append(pr.Y[:pr.n, lj])
            zs.append(pr.Z[:pr.n, lj])
        if not Xs:
            return np.zeros((0, len(cols)), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int8)
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(zs)

    @staticmethod
    def _wm(X):
        w = np.ones((X.shape[0], 1), np.float32)
        for q in range(X.shape[1]):
            xi = X[:, q][:, None]
            w = np.concatenate([w * (1 - xi), w * xi], 1)
        return w

    @staticmethod
    def _llv(w, y, z):
        tot = 0.0
        for e in np.unique(z):
            m = z == e
            n = w[m].sum(0)
            n1 = w[m].T @ y[m]
            n0 = n - n1
            with np.errstate(divide="ignore", invalid="ignore"):
                tot += float(np.where(n1 > 1e-9, n1 * np.log2(n1 / n), 0).sum() + np.where(n0 > 1e-9, n0 * np.log2(n0 / n), 0).sum())
        return tot

    def _plist(self, j):
        return [int(q) for q in self.par[j, :self.npar[j]]]

    def _setp(self, j, pl):
        self.par[j, :] = -1
        for q, v in enumerate(pl):
            self.par[j, q] = v
        self.npar[j] = len(pl)
        self._changed.add(int(j))

    _changed = set()

    def _grow(self):
        self._changed = getattr(self, "_changed", set())
        G = len(self.names)
        gain = np.zeros((G, G), np.float32)
        for pr in self.probs:
            if pr.n < 2:
                continue
            X, Y = pr.X[:pr.n], pr.Y[:pr.n]
            N = pr.n
            c11 = X.T @ Y
            n1 = X.sum(0)[:, None]
            y1 = Y.sum(0)[None, :]
            c10, c01, c00 = n1 - c11, y1 - c11, N - n1 - y1 + c11
            with np.errstate(divide="ignore", invalid="ignore"):
                def xl(n, d):
                    return np.where(n > 1e-9, n * np.log2(n / d), 0.0)
                gp = np.nan_to_num(xl(c11, n1) + xl(c10, n1) + xl(c01, N - n1) + xl(c00, N - n1) - xl(y1, N) - xl(N - y1, N))
            gl = np.concatenate([pr.gin, pr.gact])
            sub = np.ix_(gl, pr.gin)
            gain[sub] = np.maximum(gain[sub], gp.astype(np.float32))
        M = min(C("screen_top_m"), G - 1)
        added = 0
        acts = set(int(a) for pr in self.probs for a in pr.gact)
        for j in np.nonzero(self.is_target)[0]:
            p = int(self.npar[j])
            if p >= self.k:
                continue
            pl = self._plist(j)
            col = gain[:G, j].copy()
            col[pl] = -1e9
            col[j] = col[j]
            top = np.argpartition(-col, M)[:M]
            top = [int(c) for c in top if col[c] >= C("screen_min_gain_bits")]
            if not top:
                continue
            cols = pl + top
            X, y, z = self._rows(j, cols)
            if len(y) < 2:
                continue
            N = len(y)
            w = self._wm(X[:, :len(pl)])
            base = self._llv(w, y, z)
            pen = np.log2(G) + (2 ** p) * C("param_cost_bits") * np.log2(N)
            best, bg, pair = -1, 0.0, None
            for i, c in enumerate(top):
                xc = X[:, len(pl) + i][:, None]
                g = self._llv(np.concatenate([w * (1 - xc), w * xc], 1), y, z) - base
                if g - pen > bg:
                    best, bg = c, g - pen
            if best < 0 and p <= self.k - 2 and not any(q in acts for q in pl):
                pen2 = 2 * np.log2(G) + 3 * (2 ** p) * C("param_cost_bits") * np.log2(N)
                acols = [a for a in acts if gain[a, j] > 0 and a not in pl][:C("screen_top_m")]
                if acols:
                    Xa, _, _ = self._rows(j, acols)
                    for ai, a in enumerate(acols):
                        xa = Xa[:, ai][:, None]
                        wa = np.concatenate([w * (1 - xa), w * xa], 1)
                        for i, c in enumerate(top):
                            xc = X[:, len(pl) + i][:, None]
                            g2 = self._llv(np.concatenate([wa * (1 - xc), wa * xc], 1), y, z) - base
                            if g2 - pen2 > bg:
                                best, bg, pair = c, g2 - pen2, a
            if best >= 0 and pair is not None:
                self._setp(j, pl + [pair, best])
                added += 2
            elif best >= 0:
                self._setp(j, pl + [best])
                added += 1
        return added

    def _prune(self):
        G = len(self.names)
        removed = 0
        for j in np.nonzero(self.is_target & (self.npar > 0))[0]:
            pl = self._plist(j)
            p = len(pl)
            X, y, z = self._rows(j, pl)
            if len(y) < 2:
                continue
            N = len(y)
            full = self._llv(self._wm(X), y, z)
            best, bv = None, 0.0
            sets = [(i,) for i in range(p)] + [(i, k) for i in range(p) for k in range(i + 1, p)]
            for st in sets:
                keep = [i for i in range(p) if i not in st]
                g = full - self._llv(self._wm(X[:, keep]), y, z)
                cost = len(st) * np.log2(G) + (2 ** p - 2 ** (p - len(st))) * C("param_cost_bits") * np.log2(N)
                if g - cost < bv:
                    best, bv = [pl[i] for i in keep], g - cost
            if best is not None:
                self._setp(j, best)
                removed += 1
        return removed

    def _rebuild_changed(self):
        for j in list(self._changed):
            pl = self._plist(j)
            X, y, z = self._rows(j, pl)
            self.cV[j] = 0
            if len(y) == 0:
                continue
            w = self._wm(X)
            for e in np.unique(z):
                if not self.used[j, e]:
                    self.used[j, e] = True
                m = z == e
                n1 = w[m].T @ y[m]
                n = w[m].sum(0)
                self.cV[j, e, :w.shape[1], 1] = n1
                self.cV[j, e, :w.shape[1], 0] = n - n1
        self._changed = set()

    # ------------------------------------------------------------------ readouts
    def complexity(self):
        tot = 0.0
        G = len(self.names)
        for j in np.nonzero(self.is_target)[0]:
            N = max(int(sum(pr.n for pr in self.probs if pr.t2l[j] >= 0)), 2)
            tot += self.npar[j] * np.log2(G)
            for e in np.nonzero(self.used[j])[0]:
                cells = max(int((self.cV[j, e].sum(-1) > 0).sum()), 1)
                tot += cells * C("param_cost_bits") * np.log2(N)
        return float(tot)

    def n_variants(self):
        return int(self.used[self.is_target].sum()), int(self.is_target.sum())

    def edges(self, pid=None):
        out = set()
        for j in np.nonzero(self.is_target)[0]:
            for q in self._plist(j):
                out.add((self.names[q], self.names[j]))
        return out
