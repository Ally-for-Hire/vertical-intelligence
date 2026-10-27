"""Q19 step 1: extract connectome subgraphs from the FlyWire proofread table (untrusted data file: run with `python -I q19_prepare.py <feather> <out.npz>`).
Pairs with >= 5 synapses (Codex convention). Per presynaptic neuron the transmitter sign is + if acetylcholine dominates its output synapse mass, - if GABA+glutamate dominate.
60 subgraphs of 300 neurons: seed = a random neuron with in- and out-degree >= 15; grow by repeatedly adding the not-yet-chosen neuron with the largest total connection weight to the chosen set
(undirected), which keeps a densely connected local neighbourhood. Output: W[g, i, j] = synapses from neuron i to neuron j inside subgraph g (float32), sign[g, i], and the original root ids."""
import sys
import numpy as np
import pyarrow.feather as feather

path, out = sys.argv[1], sys.argv[2]
t = feather.read_table(path, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count", "gaba_avg", "ach_avg", "glut_avg"]).to_pandas()
t["gab"] = t["gaba_avg"] * t["syn_count"]; t["ach"] = t["ach_avg"] * t["syn_count"]; t["glu"] = t["glut_avg"] * t["syn_count"]
pair = t.groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False).agg(w=("syn_count", "sum")).reset_index()
tr = t.groupby("pre_pt_root_id", sort=False).agg(gab=("gab", "sum"), ach=("ach", "sum"), glu=("glu", "sum"))
del t
pair = pair[pair["w"] >= 5]
ids, inv = np.unique(np.concatenate([pair["pre_pt_root_id"].values, pair["post_pt_root_id"].values]), return_inverse=True)
n = len(ids)
pre = inv[:len(pair)]; post = inv[len(pair):]; w = pair["w"].values.astype(np.float32)
sgn = np.ones(n, np.float32)
idx = np.searchsorted(ids, tr.index.values)
idx_c = np.minimum(idx, n - 1)
keep = ids[idx_c] == tr.index.values
excit = tr["ach"].values >= (tr["gab"].values + tr["glu"].values)
sgn[idx_c[keep]] = np.where(excit[keep], 1.0, -1.0)
# adjacency lists (undirected neighbours with weights) via sorting
order = np.argsort(np.concatenate([pre, post]), kind="stable")
a = np.concatenate([pre, post]); b = np.concatenate([post, pre]); ww = np.concatenate([w, w])
a, b, ww = a[order], b[order], ww[order]
starts = np.searchsorted(a, np.arange(n + 1))
indeg = np.bincount(post, minlength=n); outdeg = np.bincount(pre, minlength=n)
rng = np.random.default_rng(2026)
cands = np.nonzero((indeg >= 15) & (outdeg >= 15))[0]
G, N = 60, 300
Ws = np.zeros((G, N, N), np.float32); S = np.zeros((G, N), np.float32); R = np.zeros((G, N), np.int64)
wd = {}
for k, (p_, q_, x_) in enumerate(zip(pre, post, w)):
    pass
# fast weight lookup using a dict keyed by (pre, post) for the pairs we need is too slow for 2.7M pairs; use sorted keys + searchsorted instead
key = pre.astype(np.int64) * n + post.astype(np.int64)
korder = np.argsort(key); key_s = key[korder]; w_s = w[korder]
for g in range(G):
    seed = int(rng.choice(cands))
    chosen = [seed]; inset = {seed}
    score = {}
    def add_neighbors(v):
        for ptr in range(starts[v], starts[v + 1]):
            u = int(b[ptr])
            if u not in inset:
                score[u] = score.get(u, 0.0) + float(ww[ptr])
    add_neighbors(seed)
    while len(chosen) < N and score:
        u = max(score, key=score.get)
        del score[u]
        chosen.append(u); inset.add(u)
        add_neighbors(u)
    chosen = np.array(chosen[:N])
    if len(chosen) < N:
        chosen = np.concatenate([chosen, rng.choice(n, N - len(chosen), replace=False)])
    pp = chosen[:, None].astype(np.int64) * n + chosen[None, :].astype(np.int64)
    pos = np.searchsorted(key_s, pp.ravel())
    pos = np.minimum(pos, len(key_s) - 1)
    hit = key_s[pos] == pp.ravel()
    Wg = np.where(hit, w_s[pos], 0.0).reshape(N, N)
    np.fill_diagonal(Wg, 0.0)
    Ws[g] = Wg; S[g] = sgn[chosen]; R[g] = ids[chosen]
np.savez_compressed(out, W=Ws, sign=S, ids=R)
print("subgraphs", G, "of", N, "neurons; mean in-subgraph connections per neuron (>=5 synapses): %.1f; mean fraction excitatory %.2f" % ((Ws > 0).sum((1, 2)).mean() / N, (S > 0).mean()))
