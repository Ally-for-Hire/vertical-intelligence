"""Probability net v4b: SNARC + NeuroBlock cells that learn WHICH INPUTS MATTER from reward alone.

v4a matched inputs on every dimension, so junk inputs blew the table up (2^junk cells) and nothing generalised.
Here the net keeps a log of its recent trials (input, response given, success). Every RELEVANCE_EVERY trials it asks, for each input
dimension: does knowing this input change the success rate of a response, on its own or together with one partner input?
  alone:  chi-square of success rate across the two values of the input (per response)
  pair:   chi-square across the four value combinations of (input i, input j), counted only as evidence for BOTH inputs when it beats
          what either input explains alone (this keeps a junk bit paired with a real one from being flagged, and still catches XOR,
          whose two inputs say nothing alone and everything together)
A dimension is relevant when the evidence beats a Bonferroni threshold: x > -2 ln(alpha / tests), alpha = 0.01.
Cells then match only on the relevant dimensions; cells that become identical are merged (synapse probabilities and counts pooled).
If nothing is significant, all dimensions are relevant. Repeated looks spend alpha/(k(k+1)) so chance false alarms stay rare.
"""
import numpy as np
from snarc import SnarcNet

ALPHA = 0.01


class SnarcRel(SnarcNet):
    def __init__(s, n_in, K, rng, every=400, log=800, **kw):
        super().__init__(n_in, K, rng, **kw)
        s.rel = np.ones(n_in, bool)
        s.every, s.nlog = every, log
        s.LX = np.zeros((log, n_in)); s.La = np.zeros(log, int); s.LR = np.zeros(log); s.nl = 0
        s.D_full = n_in
        s.calls = 0

    def _dist(s, x):
        return np.linalg.norm((s.W[:s.count] - x)[:, s.rel], axis=1)

    # ------------------------------------------------------------------ relevance test
    def _relevance(s):
        n = min(s.nl, s.nlog)
        L = s.LX[:n]
        Xb = (L > 0.5).astype(float) if np.all(np.isin(L, [0, 1])) else (L > np.median(L, 0)[None, :]).astype(float)
        D = s.D_full
        s.calls += 1
        tests = max(D * (D + 1) // 2 * s.K, 1)
        crit = -2.0 * np.log(ALPHA / (s.calls * (s.calls + 1)) / tests)      # alpha spending over repeated looks: alpha / (k (k+1)) sums to alpha
        single = np.zeros(D); pair = np.zeros((D, D))
        for a in range(s.K):
            m = (s.La[:n] == a).astype(float)
            if m.sum() < 20:
                continue
            Rm = s.LR[:n] * m
            pbar = max(min(Rm.sum() / m.sum(), 1 - 1e-3), 1e-3)
            v = pbar * (1 - pbar)
            Xn = 1 - Xb
            # single input
            n1, n0 = (Xb * m[:, None]).sum(0), (Xn * m[:, None]).sum(0)
            s1, s0 = (Xb * Rm[:, None]).sum(0), (Xn * Rm[:, None]).sum(0)
            ps = np.where(n1 > 0, s1 / np.maximum(n1, 1), pbar); p0 = np.where(n0 > 0, s0 / np.maximum(n0, 1), pbar)
            st = (n1 * (ps - pbar) ** 2 + n0 * (p0 - pbar) ** 2) / v
            single = np.maximum(single, st)
            # pairs
            tot = np.zeros((D, D))
            for A, B in ((Xb, Xb), (Xb, Xn), (Xn, Xb), (Xn, Xn)):
                cnt = (A * m[:, None]).T @ B
                suc = (A * Rm[:, None]).T @ B
                pg = np.where(cnt > 0, suc / np.maximum(cnt, 1), pbar)
                tot += cnt * (pg - pbar) ** 2 / v
            pair = np.maximum(pair, tot)
        gain = pair - np.maximum(single[:, None], single[None, :])
        np.fill_diagonal(gain, 0)
        rel = single > crit
        both = gain > crit
        rel |= both.any(1) | both.any(0)
        return rel

    def _set_rel(s, rel):
        if not rel.any() or np.array_equal(rel, s.rel):
            return
        scale = np.sqrt(rel.sum() / s.rel.sum())
        s.rel = rel
        s.r0 = s.r0 * scale
        s.r[:s.count] *= scale
        s.min_r = s.r0 / 2 ** 6
        # merge cells that are now identical (same projected prototype and radius)
        seen = {}
        for i in range(s.count):
            if not s.alive[i]:
                continue
            key = (tuple(np.round(s.W[i][rel], 6)), round(float(s.r[i]), 6))
            if key in seen:
                j = seen[key]
                wj, wi = s.n[j] + 1, s.n[i] + 1
                s.p[j] = (s.p[j] * wj + s.p[i] * wi) / (wi + wj); s.p[j] /= s.p[j].sum()
                s.n[j] += s.n[i]; s.succ[j] += s.succ[i]; s.fired[j] += s.fired[i]
                s.alive[i] = False
            else:
                seen[key] = i

    def present(s, X, y=None, learn=False, suggest=True):
        out = super().present(X, y, learn, suggest)
        if learn and y is not None:
            acts, R = out[0], out[1]
            for b in range(len(X)):
                k = s.nl % s.nlog
                s.LX[k], s.La[k], s.LR[k] = X[b], acts[b], R[b]
                s.nl += 1
                if s.nl >= 200 and s.nl % s.every == 0:
                    rel = s._relevance(); s._set_rel(rel if rel.any() else np.ones(s.D_full, bool))
        return out
