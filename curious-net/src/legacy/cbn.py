"""Curious network v1: Boolean baseline, soft-bit control and qubit-style encoding share ONE learner.

Learner: each target bit at t+1 is predicted from <=K parent bits at t (incl. 3 action bits) by a count table.
Inputs/targets may be fractional in [0,1] ("soft"): the table cell weights are the multilinear product of the parents' probabilities and
counts are updated by weight*y, so hard 0/1 bits reduce EXACTLY to the Boolean count-table learner (Boolean = the collapsed limit).
Uncertainty = prequential KL(y || p) in bits per tick (= code length for hard bits).

Encodings per scalar (4 thresholds, evenly spaced over the seen range):
  hard : 4 thermometer bits + 1 delta bit
  soft : same 5 bits but smooth: sigmoid((v-q)/width)
  qubit: per threshold a qubit with polar angle theta = pi*sigmoid((v-q)/width) (z-readout p=sin^2(theta/2): the threshold bit) and azimuth set by the
         rate of change: x-readout p = (1 + sin(theta)*tanh(dv/scale))/2. Near a threshold the qubit is in superposition and its x-readout carries the
         sign/size of change; far from a threshold the x-readout is maximally uncertain (0.5). 8 readouts per scalar.
"""
import numpy as np

K = 4
GRID_BITS = 49
ACT_BITS = 3
ACTIONS = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1]], np.float32)  # steer L/S/R x thr 0/1
BITS_PER_MODE = {"hard": 8, "soft": 8, "qubit": 8, "hard5": 5, "soft5": 5}


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class Encoder:
    def __init__(self, names, mode="hard", width=0.3):
        self.names, self.mode, self.width = names, mode, width
        self.bp = BITS_PER_MODE[mode]
        self.q = None
        self.prev = None
        self.var_of_bit = ["grid"] * GRID_BITS + [n for n in names for _ in range(self.bp)]
        self.n_state = len(self.var_of_bit)
        self.var_of_in = self.var_of_bit + ["a_L", "a_R", "a_T"]

    def calibrate(self, S):
        S = np.asarray(S)
        lo, hi = np.percentile(S, 1, axis=0), np.percentile(S, 99, axis=0)
        self.sp = np.maximum(hi - lo, 0.0) / 5.0                                 # bin spacing
        self.q = lo[:, None] + (hi - lo)[:, None] * (np.arange(1, 5) / 5.0)[None, :]
        d = np.diff(S, axis=0)
        self.dsc = d.std(0) * 0.5 + 1e-9
        dlo, dhi = np.percentile(d, 1, axis=0), np.percentile(d, 99, axis=0)
        self.dsp = np.maximum(dhi - dlo, 0.0) / 5.0
        self.dq = dlo[:, None] + (dhi - dlo)[:, None] * (np.arange(1, 5) / 5.0)[None, :]   # thresholds on the INCREMENT

    def encode(self, grid, s):
        dv = s - (self.prev if self.prev is not None else s)
        self.prev = s.copy()
        g = grid.astype(np.float32)
        if self.mode == "hard5":
            thr = (s[:, None] > self.q).astype(np.float32)
            return np.concatenate([g, np.concatenate([thr, (dv > 1e-9).astype(np.float32)[:, None]], 1).ravel()])
        if self.mode == "hard":
            thr = (s[:, None] > self.q).astype(np.float32)
            inc = (dv[:, None] > self.dq).astype(np.float32)
            return np.concatenate([g, np.concatenate([thr, inc], 1).ravel()])
        w = np.maximum(self.sp * self.width, 1e-12)[:, None]
        zs = np.where(self.sp[:, None] > 1e-9, sig((s[:, None] - self.q) / w), (s[:, None] > self.q).astype(float))
        if self.mode == "soft5":
            d = sig(dv / (0.25 * self.dsc))[:, None]
            return np.concatenate([g, np.concatenate([zs, d], 1).ravel()]).astype(np.float32)
        if self.mode == "soft":
            wd = np.maximum(self.dsp * self.width, 1e-12)[:, None]
            inc = np.where(self.dsp[:, None] > 1e-9, sig((dv[:, None] - self.dq) / wd), (dv[:, None] > self.dq).astype(float))
            return np.concatenate([g, np.concatenate([zs, inc], 1).ravel()]).astype(np.float32)
        th = np.pi * zs
        pz = np.sin(th / 2) ** 2
        px = 0.5 * (1 + np.sin(th) * np.tanh(dv / self.dsc)[:, None])
        return np.concatenate([g, np.concatenate([pz, px], 1).ravel()]).astype(np.float32)


def kl_bits(y, p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(y, np.float64)
    yc = np.clip(y, 1e-9, 1 - 1e-9)
    return y * np.log2(yc / p) + (1 - y) * np.log2((1 - yc) / (1 - p))


class CBN:
    def __init__(self, enc, T_max, seed=0, k=K, window=3000, gain_scale=1.0, gate=False, tnames=None, joint=True):
        self.enc = enc
        self.Bs = enc.n_state
        self.Bi = self.Bs + ACT_BITS
        self.tnames = list(tnames) if tnames is not None else list(enc.var_of_bit)
        self.Bt = len(self.tnames)
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.par = np.full((self.Bt, k), self.Bi, np.int32)        # Bi == constant-zero dummy input
        self.npar = np.zeros(self.Bt, np.int32)
        self.cnt = np.zeros((self.Bt, 2 ** k, 2), np.float32)
        self.X = np.zeros((T_max, self.Bi + 1), np.float32)
        self.Y = np.zeros((T_max, self.Bt), np.float32)
        self.t = 0
        self.W = window
        self.gain_scale = gain_scale
        self.fast = np.zeros(self.Bt)
        self.slow = np.zeros(self.Bt)
        self.act_n = np.ones(8)
        # ---- self-ranking of importance (per variable) ----
        self.vnames = sorted(set(enc.var_of_in))
        vi = {n: i for i, n in enumerate(self.vnames)}
        for n in self.tnames:
            if n not in vi:
                vi[n] = len(vi); self.vnames.append(n)
        self.vid_in = np.array([vi[n] for n in enc.var_of_in] + [-1])           # -1 = dummy input
        self.vid_t = np.array([vi[n] for n in self.tnames])
        V = len(self.vnames)
        self.p_unc = np.full(V, 0.01)       # importance for reducing uncertainty (the drive)
        self.p_task = np.full(V, 0.01)      # relevance to the scored variables (monitored only; never used in learning)
        self.n_upd = 0
        self.alpha0, self.alpha_min, self.tau = 0.8, 0.05, 3.0
        self.gate = gate
        self.joint = joint
        self.exempt = np.array([n.startswith("a_") or n == "grid" for n in self.vnames])
        self.task_vars = [vi["score"]] if "score" in vi else []   # all the agent is told: this sensor is what is scored; the rest is learned
        self.p_hist = []
        self.n_search = 0

    # ---------------------------------------------------------------- core
    def weights(self, x):
        """x: (Bi+1,) -> (Bs, 2^k) cell weights (multilinear in parent probabilities)."""
        w = np.ones((self.Bt, 1), np.float32)
        for i in range(self.k):
            xi = x[self.par[:, i]][:, None]
            w = np.concatenate([w * (1 - xi), w * xi], axis=1)
        return w

    def p1(self, x):
        w = self.weights(x)
        c = self.cnt
        pc = (c[:, :, 1] + 0.5) / (c[:, :, 0] + c[:, :, 1] + 1.0)
        return (w * pc).sum(1), w

    def observe(self, x, y):
        p, w = self.p1(x)
        loss = kl_bits(y, p)
        self.cnt[:, :, 1] += w * y[:, None]
        self.cnt[:, :, 0] += w * (1 - y)[:, None]
        t = self.t
        self.X[t] = x
        self.Y[t] = y
        self.t += 1
        self.fast += (loss - self.fast) / 100.0
        self.slow += (loss - self.slow) / 1000.0
        return loss

    # ---------------------------------------------------------------- structure growth
    @staticmethod
    def _ll(w, y):
        n = w.sum(0)
        n1 = w.T @ y
        n0 = n - n1
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.where(n1 > 1e-9, n1 * np.log2(n1 / n), 0).sum() + np.where(n0 > 1e-9, n0 * np.log2(n0 / n), 0).sum())

    def _wmat(self, Xw, j, upto=None):
        w = np.ones((Xw.shape[0], 1), np.float32)
        for i in range(self.npar[j] if upto is None else upto):
            xi = Xw[:, self.par[j, i]][:, None]
            w = np.concatenate([w * (1 - xi), w * xi], axis=1)
        return w

    def grow(self):
        T = self.t
        if T < 400:
            return 0
        W = min(self.W, T)
        Xw = self.X[T - W:T, :self.Bi]
        Yw = self.Y[T - W:T]
        n1 = Xw.sum(0)
        c11 = Xw.T @ Yw
        y1 = Yw.sum(0)[None, :]
        c1 = n1[:, None]
        c10, c01, c00 = c1 - c11, y1 - c11, W - c1 - y1 + c11
        with np.errstate(divide="ignore", invalid="ignore"):
            def xl(n, d):
                return np.where(n > 1e-9, n * np.log2(n / d), 0.0)
            gain = np.nan_to_num(xl(c11, c1) + xl(c10, c1) + xl(c01, W - c1) + xl(c00, W - c1) - xl(y1, W) - xl(W - y1, W))
        for j in range(self.Bt):
            gain[self.par[j][self.par[j] < self.Bi], j] = -1e9
        self.n_search += 1
        parked = np.zeros(len(self.vnames), bool)
        if self.gate and self.n_upd >= 6:
            parked = (self.p_unc < 0.03) & ~self.exempt
            if self.n_search % 4 == 0:           # periodic re-check so a parked variable can come back
                parked[:] = False
            gain[parked[self.vid_in[:self.Bi]], :] = -1e9
        tgt_ok = ~parked[self.vid_t]
        M = 8
        top = np.argpartition(-gain, M, axis=0)[:M]
        Xf = self.X[T - W:T]
        added = 0
        self.update_importance(Xf, Yw, W)
        for j in range(self.Bt):
            p = int(self.npar[j])
            if p >= self.k or not tgt_ok[j]:
                continue
            w = self._wmat(Xf, j)
            yj = Yw[:, j]
            base = self._ll(w, yj)
            best, bg, pair = -1, 0.0, None
            pen = np.log2(self.Bi) + (2 ** p) * 0.5 * np.log2(W)
            for c in top[:, j]:
                if gain[c, j] < 1.0:
                    continue
                xc = Xf[:, c][:, None]
                g = self._ll(np.concatenate([w * (1 - xc), w * xc], 1), yj) - base
                if g > self.gain_scale * pen and g - pen > bg:
                    best, bg = int(c), g - pen
            if best < 0 and p <= self.k - 2 and self.joint:
                # interaction screen: an action bit often matters only jointly with another variable (e.g. steer x heading error)
                has_act = ((self.par[j] >= self.Bs) & (self.par[j] < self.Bi)).any()
                if not has_act:
                    pen2 = 2 * np.log2(self.Bi) + 3 * (2 ** p) * 0.5 * np.log2(W)
                    for ac in range(self.Bs, self.Bs + ACT_BITS):
                        xa = Xf[:, ac][:, None]
                        wa = np.concatenate([w * (1 - xa), w * xa], 1)
                        for c in top[:, j]:
                            if c == ac or gain[c, j] < 0.0:
                                continue
                            xc = Xf[:, c][:, None]
                            g2 = self._ll(np.concatenate([wa * (1 - xc), wa * xc], 1), yj) - base
                            if g2 > self.gain_scale * pen2 and g2 - pen2 > bg:
                                best, bg, pair = int(c), g2 - pen2, ac
            if best >= 0 and pair is not None:
                self.par[j, p] = pair
                self.par[j, p + 1] = best
                self.npar[j] += 2
                self._rebuild(j)
                added += 2
            elif best >= 0:
                self.par[j, p] = best
                self.npar[j] += 1
                self._rebuild(j)
                added += 1
        return added

    def update_importance(self, Xf, Yw, W):
        """Measure each variable's contribution to progress: bits/tick saved by its edges (leave-one-out), then
        p <- p + alpha_n (r - p) with alpha_n decaying from alpha0 to alpha_min: volatile early, smooth later."""
        V = len(self.vnames)
        G = np.zeros((V, V))
        for j in range(self.Bt):
            n = int(self.npar[j])
            if n == 0:
                continue
            w = self._wmat(Xf, j)
            full = self._ll(w, Yw[:, j])
            for i in range(n):
                ps = [self.par[j, q] for q in range(n) if q != i]
                wi = np.ones((Xf.shape[0], 1), np.float32)
                for pidx in ps:
                    xi = Xf[:, pidx][:, None]
                    wi = np.concatenate([wi * (1 - xi), wi * xi], axis=1)
                pv, cv = self.vid_in[self.par[j, i]], self.vid_t[j]
                if pv != cv:
                    G[pv, cv] += max(0.0, full - self._ll(wi, Yw[:, j])) / W
        gu = G.sum(1)
        pos = gu[gu > 0]
        c = np.median(pos) if len(pos) else 1.0
        r_unc = gu / (gu + c)
        # task relevance: share of each child's explained bits flowing back from the scored variables (3 hops, decay 0.5)
        base = np.zeros(V); base[self.task_vars] = 1.0
        Gn = G / np.maximum(G.sum(0, keepdims=True), 1e-12)
        rel = base.copy()
        for _ in range(3):
            rel = base + 0.5 * (Gn @ rel)
        r_task = rel / (1.0 + rel)
        a = self.alpha_min + (self.alpha0 - self.alpha_min) / (1.0 + self.n_upd / self.tau)
        self.p_unc += a * (r_unc - self.p_unc)
        self.p_task += a * (r_task - self.p_task)
        self.n_upd += 1
        self.p_hist.append((self.t, a, self.p_unc.copy(), self.p_task.copy()))

    def ranking(self, which="unc"):
        p = self.p_unc if which == "unc" else self.p_task
        return [(self.vnames[i], float(p[i])) for i in np.argsort(-p)]

    def _rebuild(self, j):
        T = self.t
        w = self._wmat(self.X[:T], j, upto=self.k)
        n1 = w.T @ self.Y[:T, j]
        n = w.sum(0)
        self.cnt[j, :, 1] = n1
        self.cnt[j, :, 0] = n - n1

    # ---------------------------------------------------------------- readouts
    def edges(self):
        E = set()
        for j in range(self.Bt):
            for p in self.par[j, :self.npar[j]]:
                E.add((self.enc.var_of_in[p], self.tnames[j]))
        return E

    def n_edges(self):
        return int(self.npar.sum())

    def n_neurons(self):
        return int((self.npar > 0).sum())

    # ---------------------------------------------------------------- curiosity
    def choose_action(self, x_state, eps=0.05):
        if self.rng.random() < eps:
            return int(self.rng.integers(6))
        lp = np.maximum(self.slow - self.fast, 0.0) + 1e-3
        has_act = ((self.par >= self.Bs) & (self.par < self.Bi)).any(1)
        open_ = (~has_act).astype(np.float64)
        score = np.zeros(6)
        x = x_state.copy()
        ntot = self.cnt.sum(2)
        for a in range(6):
            x[self.Bs:self.Bs + ACT_BITS] = ACTIONS[a]
            w = self.weights(x)
            n = (w * ntot).sum(1)
            score[a] = (lp / (n + 1.5)).sum() + (lp * open_).sum() / np.sqrt(self.act_n[a])
        a = int(np.argmax(score + self.rng.random(6) * 1e-9))
        self.act_n[a] += 1
        return a

    # ---------------------------------------------------------------- exploit
    def exploit_action(self, x_state, score_bits, depth=8, samples=4, rollout="meanfield"):
        """Constant-action rollouts of the learned bit model. meanfield: propagate predicted probabilities (coherent, no sampling noise);
        sample: Bernoulli draws (hard bit worlds)."""
        best, ba = -1e9, 0
        for a in range(6):
            tot = 0.0
            reps = 1 if rollout == "meanfield" else samples
            for _ in range(reps):
                x = x_state.copy()
                x[self.Bs:self.Bs + ACT_BITS] = ACTIONS[a]
                for d in range(depth):
                    p, _ = self.p1(x)
                    nxt = p.astype(np.float32) if rollout == "meanfield" else (self.rng.random(self.Bs) < p).astype(np.float32)
                    x[:self.Bs] = nxt
                    tot += float(nxt[score_bits].sum()) / 4.0
            tot = tot / reps + self.rng.random() * 1e-6
            if tot > best:
                best, ba = tot, a
        return ba


class FutureHeads:
    """Predicts the scored variable's level bits h ticks ahead, from the current state + action (a general value-function style prediction).
    Same learner, same MDL growth; trained from a delayed queue. The agent uses it to compare actions by predicted future score."""
    def __init__(self, enc, score_bits, horizons=(5, 10), T_max=20000, seed=0):
        self.h = horizons
        self.sb = score_bits
        self.heads = [CBN(enc, T_max, seed + i, tnames=[f"score@{h}"] * 4) for i, h in enumerate(horizons)]
        self.q, self.qa = [], []

    def observe(self, x, y_now, a):
        """x: inputs at t (copy it), a: action index taken at t; y_now: encoded state at t+1.
        Head h learns 'score after KEEPING this action for h ticks': a sample is used only if the last h actions were identical."""
        self.q.append(x.copy()); self.qa.append(a)
        t = len(self.q)
        for hd, h in zip(self.heads, self.h):
            if t >= h and all(v == a for v in self.qa[t - h:]):
                hd.observe(self.q[t - h], y_now[self.sb])
        if t > max(self.h) + 2:
            self.q.pop(0); self.qa.pop(0)

    def grow(self):
        for hd in self.heads:
            hd.grow()

    def value(self, x):
        v = 0.0
        for hd in self.heads:
            p, _ = hd.p1(x)
            v += float(p.sum()) / 4.0
        return v / len(self.heads)

    def best_action(self, x_state, rng):
        Bs = self.heads[0].Bs
        best, ba = -1e9, 0
        for a in range(6):
            x = x_state.copy()
            x[Bs:Bs + ACT_BITS] = ACTIONS[a]
            v = self.value(x) + rng.random() * 1e-6
            if v > best:
                best, ba = v, a
        return ba
