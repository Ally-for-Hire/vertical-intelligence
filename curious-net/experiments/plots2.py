import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = json.load(open("../results/expH.json")); I = json.load(open("../results/expI.json"))
col = {"never": "#9a9a9a", "self": "#1f6fb4", "sched250": "#b4561f", "sched1000": "#d98a55", "sched4000": "#e8c3a5"}
lab = {"never": "never sleeps", "self": "self-chosen sleep", "sched250": "every 250 ticks", "sched1000": "every 1000", "sched4000": "every 4000"}
def key(r): return r["arm"] + (str(r["period"]) if r["period"] else "")
keys = ["never", "self", "sched250", "sched1000", "sched4000"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k in keys:
    v = [r for r in H if key(r) == k]
    ax[0].bar(keys.index(k), np.mean([r["log"]["bits_causal"][-1] for r in v]), yerr=np.std([r["log"]["bits_causal"][-1] for r in v]) / np.sqrt(len(v)), color=col[k])
    ax[1].bar(keys.index(k), max(np.mean([r["ops_sleep"] for r in v]), 1), color=col[k])
ax[0].set_xticks(range(5)); ax[0].set_xticklabels([lab[k] for k in keys], rotation=25, fontsize=7); ax[0].set_title("Uncertainty on causal variables (bits/tick)")
ax[1].set_yscale("log"); ax[1].set_xticks(range(5)); ax[1].set_xticklabels([lab[k] for k in keys], rotation=25, fontsize=7); ax[1].set_title("Sleep compute (measured ops, log)")
CK = ["1000", "2000", "4000", "8000", "16000"]
for k in keys[1:]:
    v = [r for r in H if key(r) == k]
    M = np.array([[np.nan if r["evals"][c] is None else r["evals"][c] for c in CK] for r in v])
    ax[2].errorbar([int(c) for c in CK], np.nanmean(M, 0), np.nanstd(M, 0) / np.sqrt(len(v)), color=col[k], label=lab[k], marker="o", ms=3, capsize=2)
ax[2].axhline(np.mean([r["expert"] for r in H]), color="k", ls="--", lw=.8); ax[2].set_xscale("log"); ax[2].set_title("Driving score from the sleep-built policy"); ax[2].legend(fontsize=6)
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("../results/fig4_sleep_arms.png", dpi=140)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
BLK, TA, TB = I["BLK"], I["TA"], I["TB"]
colr = {"full": "#1f6fb4", "notrust": "#9467bd", "daysleep": "#2ca02c", "mlp": "#b4561f"}
names = {"full": "model (self sleep + trust)", "notrust": "model without trust", "daysleep": "model, sleep only at phase ends", "mlp": "online MLP (tuned)"}
for arm in colr:
    rs = [r for r in (I["net"] + I["mlp"]) if r["arm"] == arm]
    M = np.array([r["tot"] for r in rs]); t = (np.arange(M.shape[1]) + 1) * BLK
    ax[0].plot(t, M.mean(0), color=colr[arm], label=names[arm], lw=1.3)
ax[0].axvspan(TA, TA + TB, color="#eeeeee"); ax[0].set_ylim(25, 70); ax[0].set_title("Surprise through a physics change (shaded) and return"); ax[0].set_xlabel("tick"); ax[0].set_ylabel("bits/tick"); ax[0].legend(fontsize=6)
s = [r for r in H if r["arm"] == "self"]
t = s[0]["log"]["t"]
ax[1].plot(t, np.mean([r["log"]["alpha_mean"] for r in s], 0), color="#1f6fb4", label="mean learning rate 1/(1+evidence)")
ax[1].plot(t, np.mean([r["log"]["g_mean"] for r in s], 0), color="#b4561f", label="mean trust in permanent memory")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].legend(fontsize=7); ax[1].set_title("Awake rates (self-sleep arm)"); ax[1].set_xlabel("tick")
vn = s[0]["names"]
for nm, c in (("lat", "#1f6fb4"), ("herr", "#ff7f0e"), ("ontrack", "#2ca02c"), ("v", "#d62728"), ("a_L", "#9467bd")):
    P = np.array([[h[2][sorted(set(vn + ["grid", "a_L", "a_R", "a_T"])).index(nm)] for h in r["p_hist"]] for r in s if len(r["p_hist"]) == len(s[0]["p_hist"])])
    ax[2].plot([h[0] for h in s[0]["p_hist"]], P.mean(0), color=c, label=nm)
al = [h[1] for h in s[0]["p_hist"]]
ax[2].set_xscale("log"); ax[2].set_title("Importance p, set only in sleep"); ax[2].set_xlabel("tick"); ax[2].legend(fontsize=7)
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("../results/fig5_regime_and_rates.png", dpi=140)
print("ok")
