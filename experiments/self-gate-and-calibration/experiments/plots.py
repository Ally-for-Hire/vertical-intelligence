import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
A = json.load(open("../results/expA.json")); B = json.load(open("../results/expB.json"))
col = {"random": "#7a7a7a", "curious": "#1f6fb4"}
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
rows = []
for n_d, ls in ((0, "-"), (8, "--"), (64, ":")):
    for pol in ("random", "curious"):
        L = [A[f"{n_d}|{pol}|{s}"]["log"] for s in range(4)]
        t = L[0]["t"]
        bits = np.mean([l["bits"] for l in L], 0)
        ax[0].plot(t, bits, ls, color=col[pol], label=f"{pol}, {n_d} distractors")
        ax[1].plot(t, np.mean([l["junk"] for l in L], 0), ls, color=col[pol])
        ax[2].plot(t, np.mean([l["rec"] for l in L], 0), ls, color=col[pol])
        rows.append((n_d, pol, np.mean([l["bits"][-1] for l in L]), np.std([l["bits"][-1] for l in L]),
                     np.mean([l["junk"][-1] for l in L]), np.mean([l["rec"][-1] for l in L]), np.mean([l["prec"][-1] for l in L]),
                     np.mean([A[f"{n_d}|{pol}|{s}"]["edges"] for s in range(4)])))
ax[0].set_yscale("log"); ax[0].set_title("Uncertainty (bits/tick, lower = better)"); ax[0].set_xlabel("tick"); ax[0].legend(fontsize=6)
ax[1].set_title("False links to independent variables"); ax[1].set_xlabel("tick")
ax[2].set_title("Action-link recall"); ax[2].set_xlabel("tick")
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("../results/fig1_uncertainty_structure.png", dpi=140)
print("n_d policy bits(sd) fp rec prec edges")
for r in rows: print(r[0], r[1], f"{r[2]:.1f}±{r[3]:.1f}", f"{r[4]:.1f}", f"{r[5]:.2f}", f"{r[6]:.2f}", f"{r[7]:.0f}")

ck = [500, 1000, 2000, 4000, 8000, 12000, 16000]
S = np.array([[B[str(s)][str(c)]["score"] for c in ck] for s in range(3)])
rnd = np.mean([B[str(s)]["random"] for s in range(3)]); ex = np.mean([B[str(s)]["expert"] for s in range(3)])
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(ck, S.mean(0), "o-", color="#1f6fb4", label="model-driven (curiosity-trained, score never used in learning)")
ax.fill_between(ck, S.min(0), S.max(0), color="#1f6fb4", alpha=.15)
ax.axhline(rnd, color="#7a7a7a", ls="--", label="random actions"); ax.axhline(ex, color="#b4561f", ls="--", label="hand-written PD controller")
ax.set_xscale("log"); ax.set_xlabel("training ticks"); ax.set_ylabel("score per tick"); ax.set_title("Does understanding become score? (v0: no)")
ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("../results/fig2_score_vs_training.png", dpi=140)
print("exploit score by checkpoint", dict(zip(ck, S.mean(0).round(3))), "random", round(rnd, 3), "expert", round(ex, 3))
