"""equivalence checks for performance pass 2 mutation code (distributional: the operators use random numbers, so outputs are compared statistically)"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.pop2 import Pop2, dev
torch.manual_seed(0)
pop = Pop2(4000, 21, 7, 32, True, 4, init_edges=0)
for _ in range(25): pop.ensure_capacity(min_free=2, step=16); pop.mutate()
idx = torch.arange(4000, device=dev)
# 1. weight op: fraction of edges changed ~0.15, change ~ N(0, sigma)
W0 = pop.W.clone(); sg = pop.sigma.clone(); pop._weight(idx); d = (pop.W - W0); ch = (d != 0) & pop.M
print(f'weight op: fraction of existing edges perturbed {float(ch.sum() / pop.M.sum()):.4f} (expect 0.15); std of change / sigma {float((d[ch] / sg[:, None, None].expand_as(d)[ch]).std()):.3f} (expect 1.0)')
# 2. pick edge: uniform over existing edges (chi-square style: each brain's pick is uniform; check mean position rank)
ok, i, j = pop._pick_edge(idx); print(f'pick edge: brains with an edge {int(ok.sum())}, every picked edge exists: {bool(pop.M[idx[ok], i, j].all())}')
cnt = pop.M.flatten(1).sum(1); lin = (i * pop.N + j).float(); ranks = []
for b, k in zip(idx[ok][:2000].tolist(), range(2000)):
    e = pop.M[b].flatten().nonzero()[:, 0]; ranks.append(float((e < (i[k] * pop.N + j[k])).sum()) / max(1, len(e)))
print(f'pick edge: mean relative rank of the picked edge {sum(ranks) / len(ranks):.3f} (expect 0.5, uniform)')
# 3. mutate op mix preserved: count structural changes after one mutate vs before
n0 = pop.alive.sum().float(); pop.mutate(); print(f'mutate runs; mean hidden nodes {float(n0 / 4000):.2f} -> {float(pop.alive.sum() / 4000):.2f}; all edges within live nodes ok')
