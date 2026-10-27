import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
R = json.load(open("../results/expE.json"))
CK = ["1000", "2000", "4000", "8000", "16000", "32000"]
M = ["ranked_top3", "ranked_filtered", "greedy_mdl", "oracle", "random3", "bottom3"]
col = {"ranked_top3": "#1f6fb4", "ranked_filtered": "#2ca02c", "greedy_mdl": "#9467bd", "oracle": "#b4561f", "random3": "#7a7a7a", "bottom3": "#c9c9c9"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
print("method | mean score by checkpoint", CK)
for k in M:
    mat = np.array([[np.nan if r["ck"][c][k] is None else r["ck"][c][k] for c in CK] for r in R])
    mu = np.nanmean(mat, 0); se = np.nanstd(mat, 0) / np.sqrt(len(R))
    ax[0].errorbar([int(c) for c in CK], mu, se, color=col[k], label=k, marker="o", ms=3, capsize=2)
    print(f"{k:16s}", mu.round(2), "se", se.round(2))
ex = np.mean([r["expert"] for r in R]); rp = np.mean([r["random_policy"] for r in R])
ax[0].axhline(ex, color="k", ls="--", lw=.8); ax[0].text(1100, ex + .01, "hand-written controller", fontsize=7)
ax[0].set_xscale("log"); ax[0].set_ylim(0, 1.05); ax[0].set_xlabel("training ticks"); ax[0].set_ylabel("driving score (500 ticks, 3 starts)")
ax[0].set_title("Variables picked by the agent's own ranking"); ax[0].legend(fontsize=6)
# paired ranked vs random
d = np.array([[r["ck"][c]["ranked_top3"] - r["ck"][c]["random3"] for c in CK] for r in R])
print("paired ranked_top3 - random3 mean", d.mean(0).round(2), "t", (d.mean(0) / (d.std(0, ddof=1) / np.sqrt(len(R)))).round(1))
d2 = np.array([[r["ck"][c]["ranked_top3"] - r["ck"][c]["bottom3"] for c in CK] for r in R])
print("paired ranked_top3 - bottom3 mean", d2.mean(0).round(2), "t", (d2.mean(0) / (d2.std(0, ddof=1) / np.sqrt(len(R)))).round(1))
# importance trajectories
names = R[0]["names"]
cls = {}
for i, n in enumerate(names):
    cls[i] = ("causal:" + n) if not n.startswith("d") or n == "drag" else "distractor:" + n.split("_", 1)[1]
groups = {}
for i, c in cls.items():
    groups.setdefault(c, []).append(i)
focus = ["causal:lat", "causal:herr", "causal:ontrack", "causal:v", "causal:omega", "distractor:noise", "distractor:clock", "distractor:lamp", "distractor:copy_v", "distractor:lag_score"]
cm = plt.get_cmap("tab10")
T = [h[0] for h in R[0]["p_hist"]]
for gi, g in enumerate(focus):
    P = np.array([[np.mean([h[2][i] for i in groups[g]]) for h in r["p_hist"]] for r in R])
    ax[1].plot(T, P.mean(0), color=cm(gi), label=g.split(":")[1] + (" (d)" if g.startswith("dis") else ""), lw=1.5, ls="--" if g.startswith("dis") else "-")
ax[1].set_xscale("log"); ax[1].set_xlabel("training ticks"); ax[1].set_ylabel("importance p (mean of 10 seeds)")
ax[1].set_title("Self-assigned importance p: starts ~0"); ax[1].legend(fontsize=6, ncol=2)
# volatility
vol_all = []
for r in R:
    P = np.array([h[2] for h in r["p_hist"]]); vol_all.append(np.abs(np.diff(P, axis=0)).mean(1))
vol = np.mean(vol_all, 0)
ax[2].plot(T[1:], vol, "o-", color="#1f6fb4", label="mean |change in p| per update")
ax[2].plot(T, [h[1] for h in R[0]["p_hist"]], "s--", color="#b4561f", label="update rate alpha")
ax[2].set_xscale("log"); ax[2].set_xlabel("training ticks"); ax[2].set_title("Volatile at first, smooth later"); ax[2].legend(fontsize=7)
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("../results/fig3_ranking_and_importance.png", dpi=140)
print("volatility early(first 4 updates) %.3f late(last 8) %.3f ratio %.1f" % (vol[:4].mean(), vol[-8:].mean(), vol[:4].mean() / vol[-8:].mean()))
# rank stability
rs_e, rs_l = [], []
for r in R:
    P = np.array([h[2] for h in r["p_hist"]])
    rho = [spearmanr(P[i], P[i + 1])[0] for i in range(len(P) - 1)]
    rs_e.append(np.nanmean(rho[:4])); rs_l.append(np.nanmean(rho[-8:]))
print("rank stability (Spearman consecutive updates): early %.2f late %.2f" % (np.mean(rs_e), np.mean(rs_l)))
# what does it rank on top at the end
from collections import Counter
c = Counter()
for r in R:
    P = np.array(r["p_hist"][-1][2]); top = [names[i] for i in np.argsort(-P)[:4]]
    for n in top: c[n.split("_", 1)[1] if n.startswith("d") and "_" in n else n] += 1
print("top-4 by p at end (counts over 10 seeds):", c.most_common())
print("expert", round(ex, 3), "random policy", round(rp, 3))
