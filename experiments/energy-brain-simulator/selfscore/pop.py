"""Batched self-growing networks (genomes) evaluated together on the GPU. See DESIGN.md.

Node order (topological): [0, d) inputs | [d, d+H) hidden slots | [d+H, N) outputs. Edge i->j exists only for i < j (mask M, weight W).
"""
import torch

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OPS = ('weight', 'add_edge', 'rm_edge', 'add_node', 'rm_node', 'dup_node')
OP_P = torch.tensor([0.40, 0.20, 0.10, 0.12, 0.06, 0.12])


class Pop:
    def __init__(s, P, d, K, H=32, init_edges=2, tensors=None, stochastic=False, S=1):
        """stochastic=True: hidden units are binary neurons that SAMPLE their state, P(on) = sigmoid(beta * input); beta is an evolvable per-genome temperature
        gene (how random the network is). A prediction is the mean of S stochastic passes. stochastic=False is the original deterministic tanh network (legacy)."""
        s.P, s.d, s.K, s.H = P, d, K, H
        s.N = d + H + K; s.o0 = d + H; s.stochastic = stochastic; s.S = S
        if tensors is not None:
            s.W, s.M, s.b, s.alive = tensors[:4]
            s.beta = tensors[4] if len(tensors) > 4 else torch.ones(P, device=dev); return
        s.beta = torch.ones(P, device=dev)
        s.W = torch.zeros(P, s.N, s.N, device=dev); s.M = torch.zeros(P, s.N, s.N, dtype=torch.bool, device=dev)
        s.b = torch.zeros(P, s.N, device=dev); s.alive = torch.zeros(P, H, dtype=torch.bool, device=dev)
        ar = torch.arange(P, device=dev)[:, None]
        for k in range(K):
            if init_edges <= 0:                                                  # full connectivity: every output reads every input
                s.M[:, :d, s.o0 + k] = True; s.W[:, :d, s.o0 + k] = torch.randn(P, d, device=dev) * 0.5; continue
            idx = torch.randint(0, d, (P, init_edges), device=dev)
            s.M[ar, idx, s.o0 + k] = True; s.W[ar, idx, s.o0 + k] = torch.randn(P, init_edges, device=dev) * 0.5

    # ---- bookkeeping ---------------------------------------------------------------------------------------------------------
    def take(s, idx):
        return Pop(len(idx), s.d, s.K, s.H, tensors=(s.W[idx].clone(), s.M[idx].clone(), s.b[idx].clone(), s.alive[idx].clone(), s.beta[idx].clone()), stochastic=s.stochastic, S=s.S)

    @staticmethod
    def cat(a, b):
        return Pop(a.P + b.P, a.d, a.K, a.H, tensors=tuple(torch.cat([x, y]) for x, y in zip((a.W, a.M, a.b, a.alive, a.beta), (b.W, b.M, b.b, b.alive, b.beta))), stochastic=a.stochastic, S=a.S)

    # ---- forward -------------------------------------------------------------------------------------------------------------
    def forward(s, x, want_act=False):
        """x: (B,d) shared by every genome -> q (P,B,K) predicted P(feedback=1 | x, action k).
        Stochastic mode: x is replicated S times, hidden units sample binary states, outputs are the mean over the S passes."""
        P, B, d, H, o0 = s.P, x.shape[0], s.d, s.H, s.o0
        S = s.S if s.stochastic else 1
        xr = x.repeat(S, 1) if S > 1 else x                                   # (S*B, d), sample-major
        act = torch.zeros(P, xr.shape[0], s.N, device=dev); act[:, :, :d] = xr
        Wm = s.W * s.M
        anyalive = s.alive.any(0)
        beta = s.beta[:, None]
        for h in range(H):
            if not bool(anyalive[h]): continue
            j = d + h
            z = torch.bmm(act, Wm[:, :, j:j + 1])[:, :, 0] + s.b[:, j][:, None]
            if s.stochastic:
                act[:, :, j] = (torch.rand_like(z) < torch.sigmoid(beta * z)).float().mul_(2).sub_(1) * s.alive[:, h][:, None]     # sampled binary state in {-1, +1}
            else:
                act[:, :, j] = torch.tanh(z) * s.alive[:, h][:, None]
        zq = torch.bmm(act, Wm[:, :, o0:]) + s.b[:, o0:][:, None, :]
        q = torch.sigmoid(beta[:, :, None] * zq if s.stochastic else zq)
        if S > 1: q = q.view(P, S, B, s.K).mean(1)
        return (q, act) if want_act else q

    # ---- effective complexity of the functional structure ------------------------------------------------------------------------
    def complexity(s, thr=0.05):
        """C = effective edges + effective hidden nodes. Effective = reachable from an input AND reaching an output with |w| > thr. Dead structure does not count."""
        P, N, d, o0 = s.P, s.N, s.d, s.o0
        Mw = (s.M & (s.W.abs() > thr))
        Mf = Mw.float()
        fin = torch.zeros(P, N, device=dev); fin[:, :d] = 1
        for j in range(d, N):
            fin[:, j] = (Mf[:, :, j] * fin).amax(1)
        rto = torch.zeros(P, N, device=dev); rto[:, o0:] = 1
        for i in range(o0 - 1, -1, -1):
            rto[:, i] = (Mf[:, i, :] * rto).amax(1)
        hid = s.alive.float() * fin[:, d:o0] * rto[:, d:o0]
        edges = (Mf * fin[:, :, None] * rto[:, None, :]).sum((1, 2))
        # an edge into/out of a dead hidden slot never counts: fin/rto of a dead slot are propagated only through live edges, so also mask by alive
        alive_full = torch.ones(P, N, device=dev); alive_full[:, d:o0] = s.alive.float()
        edges = (Mf * (fin * alive_full)[:, :, None] * (rto * alive_full)[:, None, :]).sum((1, 2))
        return edges + hid.sum(1), hid.sum(1), edges

    # ---- mutation (in place, vectorised) ----------------------------------------------------------------------------------------
    def mutate(s, sigma=0.3):
        P, N, d, H, o0 = s.P, s.N, s.d, s.H, s.o0
        nmut = torch.randint(1, 4, (P,), device=dev)
        for r in range(3):
            active = nmut > r
            op = torch.multinomial(OP_P.to(dev).expand(P, -1), 1)[:, 0]
            for t, name in enumerate(OPS):
                idx = torch.nonzero(active & (op == t))[:, 0]
                if len(idx): getattr(s, '_' + name)(idx, sigma)

    def _weight(s, idx, sigma):
        n = len(idx); sel = (torch.rand(n, s.N, s.N, device=dev) < 0.15) & s.M[idx]
        s.W[idx] += torch.randn(n, s.N, s.N, device=dev) * sigma * sel
        s.b[idx] += torch.randn(n, s.N, device=dev) * sigma * 0.3 * (torch.rand(n, s.N, device=dev) < 0.15)
        if s.stochastic:                                                                  # the temperature gene drifts too (log-scale)
            s.beta[idx] = (s.beta[idx] * torch.exp(torch.randn(n, device=dev) * 0.2 * (torch.rand(n, device=dev) < 0.15))).clamp(0.25, 8.0)

    def _add_edge(s, idx, sigma):
        n = len(idx); i = (torch.rand(n, device=dev) * s.o0).long()
        j = i + 1 + (torch.rand(n, device=dev) * (s.N - i - 1).float()).long().clamp(max=s.N - 1)
        j = j.clamp(max=s.N - 1); ok = j > i
        i, j, idx = i[ok], j[ok], idx[ok]
        s.M[idx, i, j] = True; s.W[idx, i, j] = torch.randn(len(idx), device=dev) * 0.5

    def _rm_edge(s, idx, sigma):
        Mf = s.M[idx].flatten(1).float(); ok = Mf.sum(1) > 0
        idx, Mf = idx[ok], Mf[ok]
        if not len(idx): return
        pick = torch.multinomial(Mf, 1)[:, 0]
        s.M[idx, pick // s.N, pick % s.N] = False

    def _add_node(s, idx, sigma):
        n = len(idx); dead = ~s.alive[idx]; has = dead.any(1)
        idx, dead = idx[has], dead[has]; n = len(idx)
        if not n: return
        h = dead.float().argmax(1); slot = s.d + h
        src = (torch.rand(n, 3, device=dev) * slot[:, None].float()).long()
        tgt = slot[:, None] + 1 + (torch.rand(n, 2, device=dev) * (s.N - slot[:, None] - 1).float()).long()
        tgt = tgt.clamp(max=s.N - 1)
        ar = idx[:, None]
        s.alive[idx, h] = True; s.b[idx, slot] = 0
        s.M[ar.expand(-1, 3), src, slot[:, None].expand(-1, 3)] = True
        s.W[ar.expand(-1, 3), src, slot[:, None].expand(-1, 3)] = torch.randn(n, 3, device=dev) * 0.5
        s.M[ar.expand(-1, 2), slot[:, None].expand(-1, 2), tgt] = True
        s.W[ar.expand(-1, 2), slot[:, None].expand(-1, 2), tgt] = torch.randn(n, 2, device=dev) * 0.3

    def _rm_node(s, idx, sigma):
        a = s.alive[idx].float(); ok = a.sum(1) > 0
        idx, a = idx[ok], a[ok]
        if not len(idx): return
        h = torch.multinomial(a, 1)[:, 0]; slot = s.d + h
        s.alive[idx, h] = False
        s.M[idx, :, slot] = False; s.M[idx, slot, :] = False

    def _dup_node(s, idx, sigma):
        a = s.alive[idx].float(); ok = a.sum(1) > 0
        idx, a = idx[ok], a[ok]
        if not len(idx): return
        k = torch.multinomial(a, 1)[:, 0]                                    # hidden index to copy
        harange = torch.arange(s.H, device=dev)[None, :]
        free = (~s.alive[idx]) & (harange > k[:, None])
        ok = free.any(1); idx, k, free = idx[ok], k[ok], free[ok]
        if not len(idx): return
        h2 = free.float().argmax(1); sk, s2 = s.d + k, s.d + h2
        s.alive[idx, h2] = True
        s.M[idx, :, s2] = s.M[idx, :, sk]; s.W[idx, :, s2] = s.W[idx, :, sk] + torch.randn(len(idx), s.N, device=dev) * 0.05 * s.M[idx, :, sk]
        cols = torch.arange(s.N, device=dev)[None, :] > s2[:, None]
        s.M[idx, s2, :] = s.M[idx, sk, :] & cols; s.W[idx, s2, :] = (s.W[idx, sk, :] + torch.randn(len(idx), s.N, device=dev) * 0.05) * (s.M[idx, sk, :] & cols)
        s.b[idx, s2] = s.b[idx, sk]
