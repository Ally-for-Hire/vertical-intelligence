import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = json.load(open("../results/expJ.json")); order = D["order"]; N = len(order)
arms = {}
for r in D["net"] + D["mlp"]:
    arms.setdefault(r["arm"], []).append(r)
def stat(arm, k, a, b):
    v = np.array([np.mean(r["res"][k]["loss"][a:b]) for r in arms[arm]]); return v.mean(), v.std(ddof=1) / np.sqrt(len(v))
print("problem | first-500-tick bits/tick (mean±se) per arm | last-500")
for k in range(N):
    row = f"{k+1} {order[k]:11s} first: " + "  ".join(f"{a}={stat(a,k,0,5)[0]:6.1f}±{stat(a,k,0,5)[1]:.1f}" for a in ("life", "fresh", "mlp_ft", "mlp_replay", "mlp_fresh"))
    row += " | last: " + "  ".join(f"{a}={stat(a,k,35,40)[0]:5.1f}" for a in ("life", "fresh", "mlp_ft", "mlp_fresh"))
    print(row)
print("\nforward transfer = first-500 bits(fresh) - first-500 bits(arm), paired by seed (positive = helped)")
for a in ("life", "mlp_ft", "mlp_replay"):
    ref = "fresh" if a == "life" else "mlp_fresh"
    out = []
    for k in range(1, N):
        d = np.array([np.mean(f["res"][k]["loss"][:5]) - np.mean(r["res"][k]["loss"][:5]) for f, r in zip(sorted(arms[ref], key=lambda x: x["seed"]), sorted(arms[a], key=lambda x: x["seed"]))])
        out.append(f"{order[k]}={d.mean():+.1f}±{d.std(ddof=1)/np.sqrt(len(d)):.1f}")
    print(a, " ".join(out))
print("\nbackward transfer on return to the first world: first-500 loss on return minus last-500 on first exposure (positive = forgot)")
for a in ("life", "mlp_ft", "mlp_replay", "mlp_fresh", "fresh"):
    d = np.array([np.mean(r["res"][-1]["loss"][:5]) - np.mean(r["res"][0]["loss"][-5:]) for r in arms[a]]); print(f"{a:11s} {d.mean():+.1f}±{d.std(ddof=1)/np.sqrt(len(d)):.1f}")
# relevance: AUROC of trust against truth-equality
def auroc(pos, neg):
    if not len(pos) or not len(neg): return np.nan
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
pos, neg, items = [], [], []
for r in arms["life"]:
    meta = r and None
for r in arms["life"]:
    meta = [m for m in D["net"] if m["arm"] == "life" and m["seed"] == r["seed"]][0]["meta"]
    seen = {}
    for k, m in enumerate(meta):
        names = m["names"]; truth = m["truth"]
        vb = {}
        for i, n in enumerate(names):
            v = n.split("#")[0]
            if v == "grid": continue
            vb.setdefault(v, []).append(i)
        for v, idx in vb.items():
            tr = r["res"][k]["bit_trust_first"]
            g = float(np.mean([tr[i] for i in idx]))
            if v in seen and v in truth:
                same = set(truth[v]) == set(seen[v])
                (pos if same else neg).append(g); items.append((order[k], v, same, g))
        for v in vb:
            if v in truth: seen[v] = truth[v]
print("\nrelevance detection: trust (first 500 ticks) of variables whose NAME appeared before; same true parents (n=%d) vs different (n=%d)" % (len(pos), len(neg)))
print("  mean trust same: %.2f  different: %.2f  AUROC %.2f" % (np.mean(pos), np.mean(neg), auroc(pos, neg)))
from collections import defaultdict
bykey = defaultdict(list)
for o, v, s, g in items: bykey[(o, v, s)].append(g)
for k_, v_ in sorted(bykey.items()): 
    if k_[0] in ("cart", "pend", "car_mirror", "rbn"): print("   ", k_, round(np.mean(v_), 2))
print("\ncondensation: LNet complexity (bits) after k problems vs sum of separate fresh nets")
cl = [np.mean([r["res"][k]["complexity"] for r in arms["life"]]) for k in range(N)]
cf = [np.mean([r["res"][k]["complexity"] for r in arms["fresh"]]) for k in range(N)]
for k in range(N): print(f"  {k+1} {order[k]:11s} life {cl[k]:8.0f}   sum fresh {sum(cf[:k+1]):8.0f}   ratio {cl[k]/sum(cf[:k+1]):.2f}   variants {np.mean([r['res'][k]['variants'] for r in arms['life']]):.0f}")
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
w = 0.16; col = {"life": "#1f6fb4", "fresh": "#9a9a9a", "mlp_ft": "#b4561f", "mlp_replay": "#d98a55", "mlp_fresh": "#e8c3a5"}
for i, a in enumerate(col):
    m = [stat(a, k, 0, 5)[0] for k in range(N)]; e = [stat(a, k, 0, 5)[1] for k in range(N)]
    ax[0].bar(np.arange(N) + (i - 2) * w, m, w, yerr=e, color=col[a], label=a)
ax[0].set_xticks(range(N)); ax[0].set_xticklabels(order, rotation=30, fontsize=7); ax[0].set_title("Surprise in the first 500 ticks of each problem (bits/tick)"); ax[0].legend(fontsize=6)
for a in ("life", "fresh"):
    L = np.array([[np.mean(r["res"][k]["trust"][:5]) for k in range(N)] for r in arms[a]]); ax[1].plot(range(N), L.mean(0), "o-", color=col[a], label=a)
ax[1].set_xticks(range(N)); ax[1].set_xticklabels(order, rotation=30, fontsize=7); ax[1].set_title("Mean trust in stored variants, first 500 ticks"); ax[1].legend(fontsize=7)
ax[2].plot(range(N), cl, "o-", color=col["life"], label="one lifelong network"); ax[2].plot(range(N), np.cumsum(cf), "s--", color=col["fresh"], label="separate network per problem (sum)")
ax[2].set_xticks(range(N)); ax[2].set_xticklabels(order, rotation=30, fontsize=7); ax[2].set_title("Model complexity (bits)"); ax[2].legend(fontsize=7)
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("../results/fig6_lifelong.png", dpi=140)
