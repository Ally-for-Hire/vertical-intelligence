"""Q18: wiring statistics of the FlyWire connectome (proofread connections, v783) vs the project's generator assumptions.
Data file is untrusted input: run with `python -I q18_flywire_stats.py <path-to-feather>`; this script only reads columns and computes statistics.
Outputs results/q18_flywire_stats.json and results/figures/fig_connectome_degrees.png."""
import sys, json, os
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]
t = feather.read_table(path, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count", "gaba_avg", "ach_avg", "glut_avg"]).to_pandas()
print("rows (pair x neuropil):", len(t))
# synapse-weighted transmitter mass per pair
for c in ("gaba_avg", "ach_avg", "glut_avg"):
    t[c] = t[c] * t["syn_count"]
pair = t.groupby(["pre_pt_root_id", "post_pt_root_id"], sort=False).agg(w=("syn_count", "sum"), gaba=("gaba_avg", "sum"), ach=("ach_avg", "sum"), glut=("glut_avg", "sum")).reset_index()
del t
n_syn = int(pair["w"].sum())
print("distinct neuron pairs:", len(pair), " total synapses:", n_syn)
neurons = np.union1d(pair["pre_pt_root_id"].unique(), pair["post_pt_root_id"].unique())
N = len(neurons)
out = dict(n_neurons_in_table=int(N), n_pairs_all=int(len(pair)), total_synapses=n_syn)


def degree_stats(p, label):
    indeg = p.groupby("post_pt_root_id").size().reindex(neurons, fill_value=0).values
    outdeg = p.groupby("pre_pt_root_id").size().reindex(neurons, fill_value=0).values
    d = {}
    for nm, v in (("in", indeg), ("out", outdeg)):
        d[nm] = dict(mean=float(v.mean()), median=float(np.median(v)), p90=float(np.percentile(v, 90)), p99=float(np.percentile(v, 99)), max=int(v.max()),
                     frac_le3=float((v <= 3).mean()), frac_zero=float((v == 0).mean()))
    # concentration of synapses in the top 1% of neurons by in/out strength
    ins = p.groupby("post_pt_root_id")["w"].sum().values
    outs = p.groupby("pre_pt_root_id")["w"].sum().values
    d["top1pct_share_of_synapses_in"] = float(np.sort(ins)[::-1][: max(1, len(ins) // 100)].sum() / ins.sum())
    d["top1pct_share_of_synapses_out"] = float(np.sort(outs)[::-1][: max(1, len(outs) // 100)].sum() / outs.sum())
    # reciprocity: fraction of directed pairs whose reverse also exists
    key = p["pre_pt_root_id"].values.astype(np.int64) * 1000003 + (p["post_pt_root_id"].values.astype(np.int64) % 1000003)
    rev = p["post_pt_root_id"].values.astype(np.int64) * 1000003 + (p["pre_pt_root_id"].values.astype(np.int64) % 1000003)
    d["reciprocity"] = float(np.isin(rev, key).mean())
    # effective parents: smallest number of presynaptic partners holding 50% / 90% of a neuron's incoming synapses
    q = p.sort_values(["post_pt_root_id", "w"], ascending=[True, False]).copy()
    g = q.groupby("post_pt_root_id")["w"]
    q["share"] = g.cumsum() / g.transform("sum")
    q["rank"] = g.cumcount() + 1
    for thr in (0.5, 0.9):
        k = q[q["share"] >= thr].groupby("post_pt_root_id")["rank"].min().values
        d[f"effective_parents_{int(thr*100)}"] = dict(median=float(np.median(k)), mean=float(k.mean()), p90=float(np.percentile(k, 90)), frac_le3=float((k <= 3).mean()), frac_le10=float((k <= 10).mean()))
    d["n_pairs"] = int(len(p))
    out[label] = d
    return indeg, outdeg


ind_all, out_all = degree_stats(pair, "all_pairs")
p5 = pair[pair["w"] >= 5]
ind5, out5 = degree_stats(p5, "pairs_ge5_synapses")
# transmitter mix by synapse mass
tot = pair[["gaba", "ach", "glut"]].sum()
out["transmitter_share_of_synapse_mass"] = {k: float(v / tot.sum()) for k, v in tot.items()}
out["generator_comparison"] = {
    "generator_parents_per_output": "1-3 causal parents (+ own variable), from B = 14..210 inputs; fraction of inputs that are parents ~ 1-30%",
    "fly_effective_parents_90_median": out["pairs_ge5_synapses"]["effective_parents_90"]["median"],
    "fly_inputs_B": int(N),
    "fly_parent_fraction_of_inputs": float(out["pairs_ge5_synapses"]["in"]["median"] / N),
}
os.makedirs("../results/figures", exist_ok=True)
json.dump(out, open("../results/q18_flywire_stats.json", "w"), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for v, lab, c in ((ind5, "in-degree (>=5 synapses)", "#1f6fb4"), (out5, "out-degree (>=5 synapses)", "#b4561f")):
    h, b = np.histogram(v[v > 0], bins=np.unique(np.logspace(0, np.log10(max(v.max(), 2)), 40).astype(int)))
    ax[0].loglog(b[:-1], h, "o-", ms=3, color=c, label=lab)
ax[0].set_xlabel("degree"); ax[0].set_ylabel("neurons"); ax[0].set_title("FlyWire v783: degree distributions"); ax[0].legend(fontsize=7)
q = p5.sort_values(["post_pt_root_id", "w"], ascending=[True, False])
g = q.groupby("post_pt_root_id")["w"]
share = g.cumsum() / g.transform("sum")
rank = g.cumcount() + 1
k90 = q[share >= 0.9].assign(rank=rank[share >= 0.9]).groupby("post_pt_root_id")["rank"].min().values
ax[1].hist(np.minimum(k90, 60), bins=np.arange(1, 62), color="#2ca02c")
ax[1].set_xlabel("effective parents (inputs holding 90% of synapses), capped at 60"); ax[1].set_ylabel("neurons"); ax[1].set_title("How many inputs a neuron effectively listens to")
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("../results/figures/fig_connectome_degrees.png", dpi=140)
print(json.dumps(out, indent=1))
