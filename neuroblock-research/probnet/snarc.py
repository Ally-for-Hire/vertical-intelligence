"""Probability net v3: a SNARC with a NeuroBlock substructure, learning by operant conditioning.

SNARC (Minsky and Edmonds, 1951) was a maze-running machine of ~40 "Hebb synapses". Each synapse holds a probability that a signal
reaching it is passed on. When the machine reaches its goal, the synapses that were just active are made more likely to pass signals
again. Nothing is ever told the right answer; behaviour is shaped by reward alone (operant conditioning).

Here:
  stimulus cells  = NeuroBlock specialists. A cell has a prototype, a radius and, like a SNARC unit, one synapse probability per
                    possible response (K numbers that sum to 1). The first cell is purely random (all responses equally likely).
  a trial         = every cell whose radius contains the input fires and passes its signal to ONE response, drawn from its own synapse
                    probabilities. More specific cells (small radius) carry more weight. The response holding a >=55% weighted
                    majority is emitted; if none does, the system is unsure and emits the most specific cell's own draw.
  reinforcement   = success moves the synapses of the cells that voted for the emitted response toward that response; failure moves
                    them away (the moved mass spreads over the other responses). Step size 1/(visits+2): a cell's probabilities are the
                    running average of what worked for it, so no learning-rate knob.
  growth          = a cell that fails while it has enough evidence spawns a child at the failing input with half the radius and its parent's
                    synapses (already pushed away from the failed response). The input was 'unexplained' at that resolution.
  condensing      = like the NeuroBlock: a child that is not better than its parent is merged away; cells that never fire are pruned.
"""
import numpy as np

MAJORITY = 0.55


class SnarcNet:
    def __init__(s, n_in, K, rng, cap=600, depth=6, n_min=10, **_):
        s.D, s.K, s.rng, s.cap, s.n_min = n_in, K, rng, cap, n_min
        s.r0 = np.sqrt(n_in) / 2 + 1e-9                  # reaches every corner of the unit cube from its centre
        s.W = np.zeros((cap, n_in)); s.r = np.zeros(cap); s.p = np.zeros((cap, K))
        s.n = np.zeros(cap); s.succ = np.zeros(cap); s.par = np.full(cap, -1); s.alive = np.zeros(cap, bool)
        s.fired = np.zeros(cap)
        s.count = 0
        s._new(np.full(n_in, 0.5), s.r0, np.full(K, 1.0 / K), -1)      # the one random cell
        s.min_r = s.r0 / 2 ** depth
        s.trials = 0
        s.unsure = 0.0
        s.pe = 1.0 / K

    @property
    def guess_rate(s):
        return s.unsure

    @property
    def n_neurons(s):
        return int(s.alive.sum())

    def _dist(s, x):
        return np.linalg.norm(s.W[:s.count] - x, axis=1)

    def _new(s, w, r, p, par):
        if s.count >= s.cap:
            free = np.where(~s.alive[:s.count])[0]
            if len(free) == 0:
                return -1
            i = free[0]
        else:
            i = s.count; s.count += 1
        s.W[i], s.r[i], s.p[i], s.par[i] = w, r, p, par
        s.n[i] = s.succ[i] = s.fired[i] = 0
        s.alive[i] = True
        return i

    # ------------------------------------------------------------------ one trial: decide
    def decide(s, x, learn=False, greedy=False):
        a = s.alive[:s.count]
        d = s._dist(x)
        m = np.where(a & (d <= s.r[:s.count]))[0]
        if len(m) == 0:                                      # unexplained input: a new, purely random specialist
            i = s._new(x.copy(), s.r0 / 2, np.full(s.K, 1.0 / s.K), 0 if s.alive[0] else -1)
            m = np.array([i]) if i >= 0 else np.array([int(np.argmin(np.where(a, d, np.inf)))])
        w = 1.0 / (s.r[m] + 1e-9)
        if greedy:
            share = (w[:, None] * s.p[m]).sum(0) / w.sum()
            return int(share.argmax()), m, None
        votes = np.array([s.rng.choice(s.K, p=s.p[i]) for i in m])
        share = np.bincount(votes, weights=w, minlength=s.K) / w.sum()
        if share.max() >= MAJORITY:
            act, sure = int(share.argmax()), True
        else:
            act, sure = int(votes[np.argmin(s.r[m])]), False
        return act, m, (votes, sure)

    # ------------------------------------------------------------------ one trial: reinforce
    def reinforce(s, x, act, m, votes, R):
        for i, v in zip(m, votes):
            s.fired[i] += 1
            if v != act:
                continue                                      # only the synapses that carried the emitted response
            alpha = 1.0 / (s.n[i] + 2)
            e = np.zeros(s.K); e[act] = 1.0
            if R > 0:
                s.p[i] += alpha * (e - s.p[i])
            else:
                rest = (1.0 - e) / max(s.K - 1, 1)
                s.p[i] += alpha * (rest - s.p[i]) if s.K > 1 else 0
            s.p[i] = np.clip(s.p[i], 1e-3, None); s.p[i] /= s.p[i].sum()
            s.n[i] += 1; s.succ[i] += R
            if R == 0 and s.n[i] >= s.n_min and s.r[i] / 2 >= s.min_r:   # failing with evidence: this resolution is too coarse here
                s._split(i, x, act)

    def _split(s, i, x, act):
        if np.any(s.alive[:s.count] & (s.par[:s.count] == i) & (s._dist(x) <= s.r[i] / 2)):
            return                                            # already has a finer cell here
        p = s.p[i].copy()
        j = s._new(x.copy(), s.r[i] / 2, p, i)
        if j >= 0:
            s.n[j] = 0

    def condense(s):
        """merge away children that are not better than their parent; prune cells that never fire"""
        for i in range(s.count):
            if not s.alive[i] or i == 0:
                continue
            if s.fired[i] == 0 and s.trials > 0:
                s.alive[i] = False
                continue
            pa = s.par[i]
            if pa >= 0 and s.alive[pa] and s.n[i] >= s.n_min and s.n[pa] >= s.n_min:
                mi, mp = s.succ[i] / s.n[i], s.succ[pa] / s.n[pa]
                se = np.sqrt(mi * (1 - mi) / s.n[i] + mp * (1 - mp) / s.n[pa] + 1e-9)
                if mi - mp < 2 * se and s.n[i] >= 4 * s.n_min:
                    s.alive[i] = False
        s.fired[:] = 0

    # ------------------------------------------------------------------ batch interface used by the suite and the car
    def present(s, X, y=None, learn=False, suggest=True):
        B = len(X)
        acts = np.zeros(B, int); info = []
        for b in range(B):
            a, m, v = s.decide(np.asarray(X[b], float), greedy=(not suggest))
            acts[b] = a; info.append((m, v))
        if y is None:
            return acts, np.array([(v is not None and not v[1]) for _, v in info])
        R = y(acts) if callable(y) else (acts == y).astype(float)
        if learn:
            for b in range(B):
                m, v = info[b]
                s.reinforce(np.asarray(X[b], float), acts[b], m, v[0], R[b])
                s.trials += 1
                s.unsure = 0.99 * s.unsure + 0.01 * float(not v[1])
                s.pe = 0.99 * s.pe + 0.01 * R[b]
            if s.trials // 200 > (s.trials - B) // 200:
                s.condense()
        return acts, R, np.array([not v[1] for _, v in info])

    def evaluate(s, X, y):
        a1, u1 = s.present(X)
        a0, _ = s.present(X, suggest=False)
        return float((a1 == y).mean()), float(u1.mean()), float((a0 == y).mean()), s.n_neurons
