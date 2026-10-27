import json, numpy as np
D = json.load(open("../results/expO.json"))
te = D["test"]
N1 = json.load(open("../results/expN1.json"))["test"]
plain = {r["seed"]: np.mean(r["arms"]["plain"][:1000]) for r in N1}
variants = ["real", "synth", "hard", "noisy20"]
gens = [10, 20, 40, 60]
print("gain over plain transformer (first 1000 ticks), mean over 24 worlds, mean +- se over chains")
print(f"{'variant':9s} " + " ".join(f"g={g:<10d}" for g in gens))
table = {}
for v in variants:
    chains = sorted({int(a.split("_c")[1].split("_g")[0]) for r in te for a in r["arms"] if a.startswith(v + "_c")})
    cells = []
    for g in gens:
        per_chain = []
        for c in chains:
            key = f"{v}_c{c}_g{g}"
            xs = [plain[r["seed"]] - np.mean(r["arms"][key][:1000]) for r in te if key in r["arms"] and r["seed"] in plain]
            if xs:
                per_chain.append(np.mean(xs))
        if per_chain:
            table[(v, g)] = (np.mean(per_chain), np.std(per_chain, ddof=1) / np.sqrt(len(per_chain)), len(per_chain))
            cells.append(f"{np.mean(per_chain):+.3f}±{np.std(per_chain, ddof=1)/np.sqrt(len(per_chain)):.3f}")
        else:
            cells.append("      n/a   ")
    print(f"{v:9s} " + " ".join(f"{c:12s}" for c in cells))
print("\nlabel diagnostics per variant (mean over chains): positive-label rate and agreement of the labels with ground truth, early vs late generations")
for v in variants:
    cs = [c for c in D["chains"] if c["variant"] == v]
    lr = np.array([c["label_rate"] for c in cs]); ta = np.array([c["truth_agree"] for c in cs])
    print(f"  {v:8s} rate gens 0-9 {lr[:, :10].mean():.3f} -> gens 50-59 {lr[:, 50:].mean():.3f}   truth agreement {ta[:, :10].mean():.3f} -> {ta[:, 50:].mean():.3f}")
print("\nreference: true-parent rate ~0.059; always-negative labels would agree with truth ~0.94 (so agreement alone is not informative; use gain)")
