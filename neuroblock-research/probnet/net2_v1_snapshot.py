"""Probability net v1: success estimator + suggestion + self-deciding cells.

Per presentation (T ticks of binary cells, as in v0):
  1. first half of the ticks run freely (cells flip coins from their own weights).
  2. THE ESTIMATOR reads everything so far (input bits + hidden cell states) and computes, for each option k, the possibility of
     succeeding P(success | state, k), a number between 0 and 1 (about the chance rate at the start).
  3. THE SUGGESTER picks an option with probability proportional to its odds P/(1-P) (probability matching on odds: it sharpens
     by itself as the estimates approach 1) and pushes the output cells: suggested group +logit(P), other groups -logit(P).
     It cannot set any cell; each cell still flips its own coin with the pushed probability.
  4. the remaining ticks run; the answer is the output group with a >=55% majority (exactly one), else no answer.
Learning: the estimator learns from the answers actually given (success or not). The output cells' own weights learn to reproduce
the states they were in when the answer succeeded (reward-weighted delta rule), so with experience they decide without the push.
Connections into outputs cost energy that shrinks with repeated use; costly ones fade and are cut.
"""
import numpy as np

MAJORITY = 0.55


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class ProbNet2:
    def __init__(s, n_in, K, rng, H=None, g=9, T=6, lr=0.5, lr_est=0.05, wd=0.001, prune_eps=0.02, prune_every=20,
                 hid_scale=2.0, suggest=True, **_):
        s.n_in, s.K, s.rng, s.g, s.T = n_in, K, rng, g, T
        s.H = int(np.clip(4 * n_in, 16, 64)) if H is None else H
        s.N = n_in + s.H + K * g
        s.out0 = n_in + s.H
        s.need = int(np.ceil(MAJORITY * g))
        s.lr, s.lr_est, s.wd, s.prune_eps, s.prune_every = lr, lr_est, wd, prune_eps, prune_every
        s.suggest = suggest
        N = s.N
        s.W = np.zeros((N, N))
        # hidden cells: random fixed wiring, sharp enough that cells are mostly decided by their inputs
        s.W[:, n_in:s.out0] = rng.normal(0, hid_scale / np.sqrt(n_in + 1), (N, s.H))
        s.W[s.out0:, n_in:s.out0] = 0                      # outputs do not feed hidden cells
        s.W[:, s.out0:] = rng.normal(0, 0.1, (N, K * s.g))
        s.M = np.ones((N, N), bool)
        s.M[:, :n_in] = False
        s.M[s.out0:, n_in:s.out0] = False
        np.fill_diagonal(s.M, False)
        s.b = np.zeros(N)
        s.b[n_in:s.out0] = rng.normal(0, 0.5, s.H)
        s.U = np.zeros((N, N))
        s.F = n_in + s.H + 1
        s.est = np.zeros((K, s.F))
        s.est_v = np.full((K, s.F), 1e-3)                   # running squared gradient (step normalisation)
        s.batches = 0

    def cost(s):
        Ubar = (s.U * s.M).sum() / max(s.M.sum(), 1)
        return 1.0 / (1.0 + s.U / (1.0 + Ubar))

    def features(s, S):
        return np.concatenate([S[:, :s.out0], np.ones((len(S), 1))], 1)

    def present(s, X, y=None, learn=False, suggest=None):
        suggest = s.suggest if suggest is None else suggest
        B, N, n_in, K = len(X), s.N, s.n_in, s.K
        rng = s.rng
        S = (rng.random((B, N)) < 0.5).astype(float)
        S[:, :n_in] = X
        Wm = s.W * s.M
        h = np.zeros((B, N))
        Tfree = s.T // 2
        trace = []
        dU = np.zeros((N, N))
        f = Ph = None
        for t in range(s.T):
            if t == Tfree:
                f = s.features(S)
                Ph = np.clip(sig(f @ s.est.T), 0.01, 0.99)            # (B, K): possibility of success per option
                if suggest:
                    odds = Ph / (1 - Ph)
                    ks = (rng.random(B)[:, None] > np.cumsum(odds / odds.sum(1, keepdims=True), 1)).sum(1).clip(0, K - 1)
                    c = np.maximum(0.0, np.log(Ph[np.arange(B), ks] / (1 - Ph[np.arange(B), ks])))
                    grp = np.repeat(np.arange(K), s.g)
                    h[:, s.out0:] = np.where(grp[None, :] == ks[:, None], c[:, None], -c[:, None])
            Z = S @ Wm + s.b
            P = sig(Z + h)
            Snew = (rng.random((B, N)) < P).astype(float)
            Snew[:, :n_in] = X
            if learn:
                trace.append((S, Snew, sig(Z)))
                dU += S.T @ Snew
            S = Snew
        counts = S[:, s.out0:].reshape(B, K, s.g).sum(2)
        passing = counts >= s.need
        valid = passing.sum(1) == 1
        act = np.where(valid, passing.argmax(1), -1)
        if y is None:
            return act
        R = (act == y).astype(float)
        if learn:
            # estimator: logistic update of the chosen option's success probability
            if valid.any():
                idx = np.where(valid)[0]
                a = act[idx]
                g_ = (R[idx] - Ph[idx, a])[:, None] * f[idx]
                for k in range(K):
                    m = a == k
                    if m.any():
                        gk = g_[m].mean(0)
                        s.est_v[k] = 0.9 * s.est_v[k] + 0.1 * gk * gk
                        s.est[k] += s.lr_est * gk / (np.sqrt(s.est_v[k]) + 1e-8)
            # output cells: reproduce the states they were in when the answer succeeded
            dW = np.zeros_like(s.W)
            db = np.zeros(N)
            for Sp, Sn, P0 in trace[Tfree:]:
                D = (Sn - P0) * R[:, None]
                D[:, :s.out0] = 0
                dW += Sp.T @ D
                db += D.sum(0)
            s.W += s.lr * dW / B * s.M
            s.b += s.lr * db / B
            s.U += dU / B * s.M
            fade = s.wd * s.cost() * s.W * s.M
            fade[:, :s.out0] = 0
            s.W -= fade
            s.batches += 1
            if s.batches % s.prune_every == 0:
                dead = (np.abs(s.W) < s.prune_eps)
                dead[:, :s.out0] = False
                s.M &= ~dead
        return act, R

    def evaluate(s, X, y):
        a1 = s.present(X)
        a0 = s.present(X, suggest=False)
        return float((a1 == y).mean()), float((a1 == -1).mean()), float((a0 == y).mean()), int(s.M.sum())
