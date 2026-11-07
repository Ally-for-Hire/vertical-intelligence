"""Unit tests for pop2.py: hand-built hidden-unit parity4 (feed-forward and recurrent), split_edge preserves function, capacity growth preserves function."""
import itertools, sys, os, copy, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.pop2 import Pop2, dev
X = np.array(list(itertools.product([0, 1], repeat=4)), float); y = (X.sum(1) % 2).astype(int)
xs = torch.tensor(X * 2 - 1, dtype=torch.float32, device=dev)
def build(stoch):
    p = Pop2(1, 4, 2, 8, stochastic=stoch, S=16); p.W.zero_(); p.M.zero_(); p.b.zero_(); p.alive.zero_(); p.typ.zero_(); p.beta.fill_(1.0)
    thr = [-2.5, -0.5, 1.5, 3.5]; g = 6.0
    for k in range(4):
        j = 4 + k; p.alive[0, k] = True
        for i in range(4): p.M[0, i, j] = True; p.W[0, i, j] = g
        p.b[0, j] = -g * thr[k]
    o0 = p.o0
    for k, sg in enumerate([1, -1, 1, -1]): p.M[0, 4 + k, o0] = True; p.W[0, 4 + k, o0] = 4.0 * sg
    p.b[0, o0] = -1.0; p.T[0] = 2                      # step 1: hidden computed, step 2: outputs see them
    return p
def acc(p): return float(((p.forward(xs)[0, :, 0] > 0.5).long().cpu().numpy() == y).mean())
for stoch in (False, True):
    p = build(stoch); print(f'hand-built parity4 (recurrent engine, T=2) stochastic={stoch}: acc {acc(p):.3f}')
p = build(False)
q0 = p.forward(xs).clone()
# split_edge on an existing edge must not change the function when T is large enough for the extra hop
p2 = build(False); p2.T[0] = 3; q_ref = p2.forward(xs).clone()
torch.manual_seed(0)
n_before = int(p2.alive.sum()); p2._split_edge(torch.tensor([0], device=dev)); n_after = int(p2.alive.sum())
print(f'split_edge: hidden nodes {n_before} -> {n_after}; max change of output on parity4 inputs (T=3 so the extra hop settles): {float((p2.forward(xs) - q_ref).abs().max()):.4f}')
# capacity growth preserves function
p3 = build(False); ref = p3.forward(xs).clone(); H0 = p3.H; p3.ensure_capacity(min_free=100, step=32)
print(f'ensure_capacity: H {H0} -> {p3.H}, N {p3.N}; max output change {float((p3.forward(xs) - ref).abs().max()):.6f}')
# complexity counts the engineered net's structure and survives cycles
p4 = build(False); p4._add_loop(torch.tensor([0], device=dev)); print('complexity (edges+nodes):', [float(v) for v in p4.complexity()[0]], '(expect 4 hidden + 4*4+4 edges + loop => about 25-26)')
# mutation runs and capacity grows under pressure
pp = Pop2(64, 4, 2, 8, stochastic=True, S=4)
for g in range(60):
    pp.ensure_capacity(min_free=4, step=16); k = pp.take(torch.arange(64, device=dev)); k.mutate(); pp = Pop2.cat(pp.take(torch.arange(32, device=dev)), k.take(torch.arange(32, device=dev)))
print('after 60 random generations: capacity', pp.H, 'stats', {k: v for k, v in pp.stats().items() if k != 'op_probs'})
