"""Regenerates the living figure set from results/*.json. Safe to re-run; skips what is missing. Output: results/figures/*.png"""
import json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "../results"
OUT = os.path.join(R, "figures")
os.makedirs(OUT, exist_ok=True)


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) > 0 else None


def gain(te, a, base="plain", k=1000):
    d = np.array([np.mean(r["arms"][base][:k]) - np.mean(r["arms"][a][:k]) if isinstance(r["arms"][a], list) else
                  np.mean(r["arms"][base]["err"][:k]) - np.mean(r["arms"][a]["err"][:k]) for r in te])
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def fig_headroom():
    K = load("expK.json")
    N1 = load("expN1.json")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if K:
        for arm, c in (("plain", "#9a9a9a"), ("oracle", "#1f6fb4")):
            m, e = [], []
            nds = (0, 10, 50, 200)
            for nd in nds:
                v = np.array([np.mean(r["err"][:1000]) for r in K["res"] if r["n_d"] == nd and r["arm"] == arm])
                m.append(v.mean()); e.append(v.std(ddof=1) / np.sqrt(len(v)))
            ax[0].errorbar(range(4), m, e, marker="o", color=c, label=("MLP, all inputs" if arm == "plain" else "MLP told the true inputs"), capsize=2)
        ax[0].set_xticks(range(4)); ax[0].set_xticklabels(nds); ax[0].set_xlabel("distractor inputs"); ax[0].set_ylabel("prediction error, first 1000 ticks"); ax[0].set_title("MLP headroom"); ax[0].legend(fontsize=7)
    if N1:
        te = N1["test"]
        nd = np.array([r["nd"] for r in te])
        for arm, c, lab in (("plain", "#9a9a9a", "transformer, all inputs"), ("oracle", "#1f6fb4", "told the true inputs"), ("prior300", "#2ca02c", "learned prior (300 scenarios)")):
            m = [np.mean([np.mean(r["arms"][arm][:1000]) for r in te if r["nd"] == n]) for n in sorted(set(nd))]
            ax[1].plot(range(len(m)), m, "o-", color=c, label=lab)
        ax[1].set_xticks(range(len(set(nd)))); ax[1].set_xticklabels(sorted(set(nd))); ax[1].set_xlabel("distractor inputs"); ax[1].set_title("Transformer headroom and what experience recovers"); ax[1].legend(fontsize=7)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_headroom.png"), dpi=140); plt.close()


def fig_experience():
    M = load("expM.json")
    N1 = load("expN1.json")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if M:
        te = M["test"]
        og = gain(te, "oracle")[0]
        for nm, c in (("lr_soft", "#1f6fb4"), ("nn_soft", "#2ca02c"), ("nn_oracle_soft", "#b4561f")):
            Ks = (1, 3, 10, 30, 100, 300)
            g = [gain(te, f"{nm}{K}") for K in Ks]
            ax[0].errorbar(range(6), [x[0] for x in g], [x[1] for x in g], marker="o", color=c, capsize=2, label=nm)
        ax[0].axhline(og, color="k", ls="--", lw=.8); ax[0].text(0, og, " oracle mask", fontsize=7, va="bottom")
        ax[0].set_xticks(range(6)); ax[0].set_xticklabels((1, 3, 10, 30, 100, 300)); ax[0].set_xlabel("previous scenarios K"); ax[0].set_ylabel("gain over plain net (error, first 1000 ticks)"); ax[0].set_title("MLP: gain vs experience"); ax[0].legend(fontsize=7)
    if N1:
        te = N1["test"]
        og = gain(te, "oracle")[0]
        for nm, c, Ks in (("prior", "#2ca02c", (1, 3, 10, 30, 100, 300)), ("prior_oracle", "#b4561f", (10, 100, 300))):
            g = [gain(te, f"{nm}{K}") for K in Ks]
            ax[1].errorbar([ (1, 3, 10, 30, 100, 300).index(K) for K in Ks], [x[0] for x in g], [x[1] for x in g], marker="o", color=c, capsize=2, label=nm)
        ax[1].axhline(og, color="k", ls="--", lw=.8); ax[1].text(0, og, " oracle mask", fontsize=7, va="bottom")
        ax[1].set_xticks(range(6)); ax[1].set_xticklabels((1, 3, 10, 30, 100, 300)); ax[1].set_xlabel("previous scenarios K"); ax[1].set_title("Transformer: gain vs experience"); ax[1].legend(fontsize=7)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_experience.png"), dpi=140); plt.close()


def fig_generations():
    N2 = load("expN2.json")
    N1 = load("expN1.json")
    if not N2:
        return
    te = N2["test"]
    plain = {r["seed"]: np.mean(r["arms"]["plain"][:1000]) for r in N1["test"]} if N1 else {}
    gens = sorted({int(a.split("_g")[1]) for r in te for a in r["arms"] if "_g" in a})
    chains = sorted({int(a.split("_g")[0][1:]) for r in te for a in r["arms"] if "_g" in a})
    fig, ax = plt.subplots(figsize=(6, 4))
    mat = []
    for c in chains:
        row = []
        for g in gens:
            key = f"c{c}_g{g}"
            v = [plain[r["seed"]] - np.mean(r["arms"][key][:1000]) for r in te if key in r["arms"] and r["seed"] in plain]
            row.append(np.mean(v) if v else np.nan)
        mat.append(row)
    mat = np.array(mat)
    base = 0.0
    for row in mat:
        ax.plot(range(len(gens)), row, color="#2ca02c", alpha=.3)
    ax.errorbar(range(len(gens)), np.nanmean(mat, 0), np.nanstd(mat, 0, ddof=1) / np.sqrt(len(chains)), color="#1f6fb4", marker="o", capsize=2, label="mean over chains")
    ax.axhline(0.148, color="#b4561f", ls="--", lw=.8); ax.text(0, 0.150, " non-recursive prior (exp N1, K=100)", fontsize=7)
    ax.set_xticks(range(len(gens))); ax.set_xticklabels(gens); ax.set_xlabel("generations (scenarios whose own labels trained the prior)"); ax.set_ylabel("gain over plain transformer")
    ax.set_title("Recursive self-improvement"); ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_generations.png"), dpi=140); plt.close()


def fig_fair_screen():
    D = load("r3_fair_screen.json")
    if not D:
        return
    te = D["test"]
    pl = np.array([np.mean(t["plain"][:1000]) for t in te])
    arms = (("hard_screen", "hard screen\n(old baseline)"), ("pvalue_screen", "soft p-value\nscreen (0 const.)"), ("tuned_screen", "tuned soft screen\n(2 const.)"),
            ("lr_soft100", "learned prior\nlogistic, K=100"), ("nn_soft100", "learned prior\n2x32 net, K=100"), ("oracle", "hard mask on\ntrue parents"))
    m, e = [], []
    for k, _ in arms:
        d = pl - np.array([np.mean(t[k][:1000]) for t in te]); m.append(d.mean()); e.append(d.std(ddof=1) / np.sqrt(len(d)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(arms)), m, yerr=e, capsize=3, color=["#9a9a9a", "#9a9a9a", "#9a9a9a", "#2ca02c", "#2ca02c", "#1f6fb4"])
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([a[1] for a in arms], fontsize=7)
    ax.set_ylabel("gain over plain MLP (error, ticks 0-1000)"); ax.set_title("R3: no-experience screens (grey) vs learned priors (green), 48 worlds")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_fair_screen.png"), dpi=140); plt.close()


def fig_calibration():
    D = load("e1_calibration.json")
    if not D:
        return
    sets = ["standard_test", "kinds_tanh_product", "high_noise_0.4", "delayed_effects", "copy_distractors_only", "many_parents_4_6", "fly_like", "pure_interaction", "connectome"]
    lab = [s.replace("_", "\n", 1) for s in sets]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(sets))
    for k, (sn, c) in enumerate((("oracle_net", "#1f6fb4"), ("own_net", "#2ca02c"))):
        mp = [D["metrics"][f"{sn}|{s}"]["mean_p"] for s in sets]
        ax[0].bar(x + (k - 0.5) * 0.27, mp, 0.27, color=c, label=f"{sn}: mean predicted P(parent)")
    ax[0].plot(x, [D["metrics"][f"oracle_net|{s}"]["pos_rate"] for s in sets], "k_", ms=18, mew=2, label="actual rate of true parents")
    ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=6); ax[0].set_ylabel("probability"); ax[0].set_title("E1: does the taught student's confidence match the truth? (non-self pairs)"); ax[0].legend(fontsize=7)
    for k, (nm, c) in enumerate((("ensemble_disagreement", "#9a9a9a"), ("nn_distance", "#b4561f"), ("entropy", "#6a3d9a"))):
        ax[1].bar(np.arange(len(sets) - 1) + (k - 1) * 0.27, [D["novelty"][s][nm] for s in sets[1:]], 0.27, color=c, label=nm)
    ax[1].axhline(0.5, color="k", lw=.8, ls="--"); ax[1].set_xticks(np.arange(len(sets) - 1)); ax[1].set_xticklabels(lab[1:], fontsize=6)
    ax[1].set_ylabel("AUROC: taught family vs this family"); ax[1].set_title("E2: can it tell it is outside what it was taught? (0.5 = no)"); ax[1].set_ylim(0, 1.05); ax[1].legend(fontsize=7, loc="upper left", ncol=3)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_calibration.png"), dpi=140); plt.close()


def fig_energy():
    b = load("pb11b_energy.json")
    if not b:
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for kind, lams, c, lab in (("act", (0.03, 0.1, 0.3, 1.0, 3.0), "#b4561f", "activity (spike) energy"), ("syn", (0.03, 0.1, 0.3), "#1f6fb4", "synapse (wiring) energy")):
        xs, ys, es = [], [], []
        for l in lams:
            k = f"{kind}_{l:g}"
            rr = np.array([r[k]["test"] / r["plain"]["test"] - 1 for r in b])
            en = np.mean([r[k]["act"] for r in b]) / np.mean([r["plain"]["act"] for r in b])
            xs.append(en); ys.append(100 * rr.mean()); es.append(100 * rr.std(ddof=1) / np.sqrt(len(rr)))
        ax.errorbar(xs, ys, es, marker="o", color=c, capsize=2, label=lab)
        for x, y, l in zip(xs, ys, lams):
            ax.annotate(f"lam {l:g}", (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("mean hidden activity relative to the plain net (lower = less energy)"); ax.set_ylabel("shifted-test error vs plain (%)")
    ax.set_title("B11: wiring cost helps generalisation; activity cost only costs (48 worlds)"); ax.legend(fontsize=8); ax.invert_xaxis()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_energy.png"), dpi=140); plt.close()


def fig_context():
    c7, c9, c9c = load("pctx7_learnask.json"), load("pctx9_surprise_G4.0.json"), load("pctx9c_shockask9500.json")
    bud = [(b, load(f"pctx9c_shockask9600_b{b}.json")) for b in ("0.01", "0.02", "0.05", "0.1")]
    rel = lambda res, a: [r[a]["true"] / r["oracle"] for r in res]
    rows = [("best-predicting", rel(c7, "best")), ("memory vote", rel(c7, "memvote")), ("random labels 35%", rel(c7, "random35")),
            ("surprise gate (no questions)", rel(c9, "gate")), ("ask on disagreement", rel(c7, "memask")), ("ask on either (CTX9c)", rel(c9c, "askboth"))]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    med = [np.median(v) for _, v in rows]; lo = [np.percentile(v, 25) for _, v in rows]; hi = [np.percentile(v, 75) for _, v in rows]
    y = np.arange(len(rows))[::-1]
    ax[0].barh(y, med, color="#8a9bb0"); ax[0].errorbar(med, y, xerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="none", ecolor="k", lw=1)
    ax[0].set_yticks(y); ax[0].set_yticklabels([n for n, _ in rows], fontsize=8); ax[0].axvline(1, color="k", lw=.8, ls="--")
    ax[0].set_xlabel("part error x oracle (median, IQR; 48 worlds, 25% labels)"); ax[0].set_title("CTX7-9c: learning-time context")
    xb = [float(b) * 100 for b, _ in bud]
    for a, lab in (("askdis", "ask on disagreement"), ("askshock", "ask on shock"), ("askboth", "ask on either")):
        ax[1].plot(xb, [np.median(rel(r, a)) for _, r in bud], marker="o", label=lab)
    ax[1].axhline(np.median(rel(bud[0][1], "gate8")), color="grey", ls=":", label="gate only (0 questions)"); ax[1].axhline(1, color="k", lw=.8, ls="--")
    ax[1].set_xscale("log"); ax[1].set_xticks(xb); ax[1].set_xticklabels([f"{v:g}%" for v in xb])
    ax[1].set_xlabel("question budget (% of training ticks)"); ax[1].set_ylabel("part error x oracle (median)"); ax[1].set_title("CTX10: question budget (seeds 9600+)"); ax[1].legend(fontsize=8)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_context.png"), dpi=140); plt.close()


if __name__ == "__main__":
    for f in (fig_headroom, fig_experience, fig_generations, fig_fair_screen, fig_calibration, fig_energy, fig_context):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:
            print("skip", f.__name__, repr(e)[:120])
