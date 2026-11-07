"""Probability net v2: stochastic cells on a grid substrate that start as ONE random neuron and grow.

Substrate: a G x G lattice. Input bits sit on fixed sensor sites in a block at the centre. Cells talk only to cells within `R` sites.
Start: the sensors plus ONE free neuron that votes for option 0. Its output is a coin flip, so the answer is a random guess (chance).
Every presentation:
  - T ticks of binary cells; each free cell turns on with probability sigmoid(weights from active local cells + bias + suggestion).
  - an option's voters are the neurons assigned to it (new neurons are assigned round-robin: option 0..K-1, then a non-voting
    "hidden" role). An option is chosen when its voters hold a >=55% majority and no other option does. If that is not the case
    the net does not know, and it guesses at random (counted as 'guess rate', its visible uncertainty).
  - the ESTIMATOR reads every active cell and gives the possibility of success per option (0..1). The SUGGESTER nudges voters toward
    the option picked with probability proportional to the odds; cells still decide for themselves.
Prediction and growth: when the recent success rate stops improving by more than chance fluctuation and is not yet near 1, the net grows
new neurons on free sites next to existing cells. Connections reinforce themselves from use; used connections cost less energy, unused
costly ones fade and are cut.
"""
import numpy as np
from math import comb

MAJORITY = 0.55


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class GridNet:
    def __init__(s, n_in, K, rng, G=None, R=2, T=8, lr=0.5, lr_est=0.05, wd=0.001, prune_eps=0.02, prune_every=20,
                 grow_every=20, hebb=0.0, w_scale=0.7, tiebreak=True, est_l2=1e-3, lam=0.1, **_):
        s.n_in, s.K, s.rng, s.T, s.R = n_in, K, rng, T, R
        s.G = int(np.clip(8 + np.ceil(np.sqrt(n_in)), 10, 14)) if G is None else G
        G = s.G; S = G * G; s.S = S
        s.tiebreak, s.est_l2, s.lam, s.E_ref = tiebreak, est_l2, lam, None
        s.lr, s.lr_est, s.wd, s.prune_eps, s.prune_every, s.grow_every, s.hebb, s.w_scale = lr, lr_est, wd, prune_eps, prune_every, grow_every, hebb, w_scale
        rr, cc = np.divmod(np.arange(S), G)
        s.rr, s.cc = rr, cc
        s.loc = (np.abs(rr[:, None] - rr[None, :]) <= R) & (np.abs(cc[:, None] - cc[None, :]) <= R) & ~np.eye(S, dtype=bool)
        d = np.hypot(rr - (G - 1) / 2, cc - (G - 1) / 2) + rng.random(S) * 1e-3
        order = np.argsort(d)
        s.sensors = order[:n_in]
        s.alive = np.zeros(S, bool); s.alive[order[:n_in + 1]] = True
        s.is_sensor = np.zeros(S, bool); s.is_sensor[s.sensors] = True
        s.role = np.full(S, -1)                    # -1 none/sensor, 0..K-1 voter for that option, K hidden
        seed = order[n_in]
        s.role[seed] = 0
        s.next_role = 1
        s.W = np.zeros((S, S)); s.b = np.zeros(S); s.U = np.zeros((S, S))
        s.est = np.zeros((K, S + 1)); s.est_v = np.full((K, S + 1), 1e-3)
        s._wire(seed)
        s.batches = 0
        s.win = []
        s.guess_rate = 1.0
        # smallest push (logits) that lets a group of 9 voters reach the 55% majority 9 times out of 10 (derived from the majority rule)
        g = 9; need = int(np.ceil(MAJORITY * g))
        for p_ in np.linspace(0.5, 0.99, 491):
            if sum(comb(g, k) * p_ ** k * (1 - p_) ** (g - k) for k in range(need, g + 1)) >= 0.9:
                break
        s.c0 = float(np.log(p_ / (1 - p_)))

    # ------------------------------------------------------------------ structure
    def M(s):
        return s.loc & s.alive[:, None] & s.alive[None, :] & ~s.is_sensor[None, :]

    def _wire(s, i):
        """give a new neuron random incoming and outgoing connections to the living cells in range"""
        nb = s.loc[i] & s.alive
        n = max(int(nb.sum()), 1)
        s.W[nb, i] = s.rng.normal(0, s.w_scale / np.sqrt(n), int(nb.sum()))
        s.W[i, nb] = s.rng.normal(0, s.w_scale / np.sqrt(n), int(nb.sum())) * ~s.is_sensor[nb]
        s.b[i] = s.rng.normal(0, 0.3)

    def grow(s, m):
        cand = np.where(~s.alive & (s.loc & s.alive[None, :]).any(1))[0]
        if len(cand) == 0:
            return 0
        pick = s.rng.choice(cand, size=min(m, len(cand)), replace=False)
        for i in pick:
            s.alive[i] = True
            s.role[i] = s.next_role % (s.K + 1) if s.next_role % (s.K + 1) < s.K else s.K
            s.next_role += 1
            s._wire(i)
        return len(pick)

    @property
    def n_neurons(s):
        return int((s.alive & ~s.is_sensor).sum())

    def cost(s):
        M = s.M()
        Ubar = (s.U * M).sum() / max(M.sum(), 1)
        return 1.0 / (1.0 + s.U / (1.0 + Ubar))

    # ------------------------------------------------------------------ one presentation
    def present(s, X, y=None, learn=False, suggest=True):
        B, S, K = len(X), s.S, s.K
        rng = s.rng
        M = s.M()
        Wm = s.W * M
        free = s.alive & ~s.is_sensor
        St = (rng.random((B, S)) < 0.5) * free
        St = St.astype(float)
        St[:, s.sensors] = X
        h = np.zeros((B, S))
        voters = [(s.role == k) & s.alive for k in range(K)]
        nv = np.array([v.sum() for v in voters])
        Tfree = s.T // 2
        trace = []; dU = np.zeros((S, S))
        energy = np.zeros(B); rowcost = (s.cost() * M).sum(1) if learn else None
        f = Ph = None
        for t in range(s.T):
            if t == Tfree:
                f = np.concatenate([St, np.ones((B, 1))], 1)
                Ph = np.clip(sig(f @ s.est.T), 0.01, 0.99)
                if suggest:
                    odds = Ph / (1 - Ph)
                    ks = (rng.random(B)[:, None] > np.cumsum(odds / odds.sum(1, keepdims=True), 1)).sum(1).clip(0, K - 1)
                    c = np.maximum(s.c0, np.log(Ph[np.arange(B), ks] / (1 - Ph[np.arange(B), ks])))
                    for k in range(K):
                        if nv[k]:
                            h[:, voters[k]] = np.where(ks == k, c, -c)[:, None]
            Z = St @ Wm + s.b
            P = sig(Z + h)
            Snew = ((rng.random((B, S)) < P) * free).astype(float)
            Snew[:, s.sensors] = X
            if learn:
                trace.append((St, Snew, sig(Z)))
                dU += St.T @ Snew
                energy += St @ rowcost
            St = Snew
        frac = np.stack([(St[:, v].mean(1) if v.any() else np.zeros(B)) for v in voters], 1)
        passing = (frac >= MAJORITY) & (nv > 0)[None, :]
        known = (passing.sum(1) >= 1) if s.tiebreak else (passing.sum(1) == 1)
        guess = rng.integers(0, K, B)
        act = np.where(known, np.where(passing, frac, -1).argmax(1), guess)
        if y is None:
            return act, ~known
        R = y(act) if callable(y) else (act == y).astype(float)       # y may be a function: act the answers in the world, return success
        if learn:
            idx = np.arange(B)
            g_ = (R - Ph[idx, act])[:, None] * f
            for k in range(K):
                m = act == k
                if m.any():
                    gk = g_[m].mean(0)
                    s.est_v[k] = 0.9 * s.est_v[k] + 0.1 * gk * gk
                    s.est[k] += s.lr_est * gk / (np.sqrt(s.est_v[k]) + 1e-8)
            if s.E_ref is None:
                s.E_ref = max(energy.mean(), 1e-9)
            Adv = R - Ph[idx, act] - s.lam * (energy / s.E_ref - 1)         # surprise, minus energy spent above the starting level
            s.est *= 1 - s.est_l2
            dW = np.zeros_like(s.W); db = np.zeros(S)
            for Sp, Sn, P0 in trace[Tfree:]:
                D = (Sn - P0) * Adv[:, None]
                D[:, ~free] = 0
                dW += Sp.T @ D
                db += D.sum(0)
                if s.hebb:                      # connections reinforce themselves from use (prediction-error Hebb), every tick
                    Dh = (Sn - P0); Dh[:, ~free] = 0
                    dW += s.hebb * (Sp.T @ Dh)
            s.W += s.lr * dW / B * M
            s.b += s.lr * db / B
            s.U += dU / B * M
            s.W -= s.wd * s.cost() * s.W * M
            s.batches += 1
            s.win.append(R.mean())
            s.guess_rate = 0.9 * s.guess_rate + 0.1 * float((~known).mean())
            if s.batches % s.prune_every == 0:
                s.W[(np.abs(s.W) < s.prune_eps)] = 0
            if s.batches % s.grow_every == 0:
                s._maybe_grow(B)
        return act, R, ~known

    def _maybe_grow(s, B):
        n = s.grow_every
        if len(s.win) < 2 * n:
            return
        now, before = np.mean(s.win[-n:]), np.mean(s.win[-2 * n:-n])
        se = np.sqrt(2 * max(now * (1 - now), 1e-3) / (n * B))             # chance fluctuation of the difference
        if now - before < 2 * se and now < 1 - 2 * se:
            s.grow(1 + s.n_neurons // 4)

    def evaluate(s, X, y):
        a1, g1 = s.present(X)
        a0, _ = s.present(X, suggest=False)
        return float((a1 == y).mean()), float(g1.mean()), float((a0 == y).mean()), s.n_neurons
