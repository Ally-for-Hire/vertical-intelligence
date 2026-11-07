"""Probability net: binary on/off cells that decide for themselves by probability.

Cells: inputs (clamped to the input bits) + hidden cluster + K output groups of g cells. Every tick each free cell turns on with
probability sigmoid(sum of weights from active cells + bias [+ suggestion]). After T ticks the answer is the output group that
holds a >=55% majority (exactly one group must; otherwise the net gives no answer, which counts as a failure).
The only teaching signal is success (1) or failure (0), never the correct answer.

Learning (v0): reward-modulated Hebbian = REINFORCE on the cells' own coin flips:  dW_ij ~ (success - expected success) * (s_j - p_j) * s_i.
Energy: an active connection costs c_ij = 1/(1 + usage_ij/(1+mean usage)); used connections get cheaper. Costly connections decay and
are cut when tiny, so the network sheds what it does not use.
"""
import numpy as np

MAJORITY = 0.55   # given by the user


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class ProbNet:
    # every number below is a constant that is NOT derived (see CONSTANTS.md); they are open problems
    def __init__(s, n_in, K, rng, H=None, g=9, T=4, lr=0.3, lam=0.1, wd=0.002, prune_eps=0.02, prune_every=20,
                 base_decay=0.95, **_):
        s.n_in, s.K, s.rng, s.g, s.T = n_in, K, rng, g, T
        s.H = n_in if H is None else H
        s.N = n_in + s.H + K * g
        s.out0 = n_in + s.H
        s.need = int(np.ceil(MAJORITY * g))          # 5 of 9 = 55.6%
        s.lr, s.lam, s.wd, s.prune_eps, s.prune_every, s.base_decay = lr, lam, wd, prune_eps, prune_every, base_decay
        N = s.N
        s.W = rng.normal(0, 1.0 / np.sqrt(N), (N, N))
        s.M = np.ones((N, N), bool)
        s.M[:, :n_in] = False
        np.fill_diagonal(s.M, False)
        s.b = np.zeros(N)
        s.U = np.zeros((N, N))
        s.base = 0.0
        s.E_ref = None
        s.batches = 0

    # ------------------------------------------------------------------ economics
    def cost(s):
        Ubar = (s.U * s.M).sum() / max(s.M.sum(), 1)
        return 1.0 / (1.0 + s.U / (1.0 + Ubar))

    # ------------------------------------------------------------------ one presentation of B inputs
    def present(s, X, y=None, learn=False):
        B, N, n_in = len(X), s.N, s.n_in
        rng = s.rng
        S = (rng.random((B, N)) < 0.5).astype(float)
        S[:, :n_in] = X
        Wm = s.W * s.M
        cost = s.cost() * s.M
        rowcost = cost.sum(1)
        energy = np.zeros(B)
        trace = []
        dU = np.zeros((N, N))
        for t in range(s.T):
            P = sig(S @ Wm + s.b)
            Snew = (rng.random((B, N)) < P).astype(float)
            Snew[:, :n_in] = X
            if learn:
                D = Snew - P
                D[:, :n_in] = 0
                trace.append((S, D))
                dU += S.T @ Snew
            energy += S @ rowcost
            S = Snew
        counts = S[:, s.out0:].reshape(B, s.K, s.g).sum(2)
        passing = counts >= s.need
        valid = passing.sum(1) == 1
        act = np.where(valid, passing.argmax(1), -1)
        if y is None:
            return act, energy
        R = (act == y).astype(float)
        if learn:
            if s.E_ref is None:
                s.E_ref = max(energy.mean(), 1e-9)
            Reff = R - s.lam * energy / s.E_ref
            A = Reff - s.base
            s.base = s.base_decay * s.base + (1 - s.base_decay) * Reff.mean()
            dW = np.zeros_like(s.W)
            db = np.zeros(N)
            for Sp, D in trace:
                G = D * A[:, None]
                dW += Sp.T @ G
                db += G.sum(0)
            s.W += s.lr * dW / B * s.M
            s.b += s.lr * db / B
            s.U += dU / B * s.M
            s.W -= s.wd * s.cost() * s.W * s.M       # costly connections fade
            s.batches += 1
            if s.batches % s.prune_every == 0:
                s.M &= np.abs(s.W) >= s.prune_eps
        return act, R, energy

    def evaluate(s, X, y):
        act, energy = s.present(X)
        return float((act == y).mean()), float((act == -1).mean()), float(energy.mean()), int(s.M.sum())
