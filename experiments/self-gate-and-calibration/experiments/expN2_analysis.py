import json, numpy as np
D = json.load(open("../results/expN2.json"))
te = D["test"]
chains = sorted({int(a.split("_g")[0][1:]) for r in te for a in r["arms"] if "_g" in a})
gens = sorted({int(a.split("_g")[1]) for r in te for a in r["arms"] if "_g" in a})
N1 = json.load(open("../results/expN1.json"))["test"]
plain = {r["seed"]: np.mean(r["arms"]["plain"][:1000]) for r in N1}   # same worlds/seeds and code path as exp N part 1 (plain arm is deterministic given the world seed)
print("test jobs", len(te), "chains", chains, "generations", gens)
# per-chain per-generation gain over plain, averaged over the 24 test worlds
G = np.full((len(chains), len(gens)), np.nan)
for i, c in enumerate(chains):
    for j, g in enumerate(gens):
        key = f"c{c}_g{g}"
        v = [plain[r["seed"]] - np.mean(r["arms"][key][:1000]) for r in te if key in r["arms"] and r["seed"] in plain]
        if v:
            G[i, j] = np.mean(v)
print("\ngain over plain transformer (first 1000 ticks), mean over 24 worlds, per chain:")
print("gen      " + "  ".join(f"{g:>6d}" for g in gens))
for i, c in enumerate(chains):
    print(f"chain {c}  " + "  ".join(f"{x:+.3f}" for x in G[i]))
m = np.nanmean(G, 0); se = np.nanstd(G, 0, ddof=1) / np.sqrt(len(chains))
print("mean      " + "  ".join(f"{x:+.3f}" for x in m))
print("se        " + "  ".join(f"{x:.3f}" for x in se))
# non-recursive reference (labels from plain networks), exp N1 same worlds are different (9000..9047); report N1 numbers for context
print("\nreference (exp N1, non-recursive, own labels from PLAIN nets, 48 worlds): K=10 +0.133, K=30 +0.149, K=100 +0.146, K=300 +0.148; oracle mask +0.294")
# label positive-rate drift along each chain (labels produced by networks that used the previous prior)
lr = np.array(D["label_rates"])
print("\nlabel positive rate (own permutation-importance labels) by scenario index within a chain, mean over chains:")
for a, b in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 150), (140, 150)):
    print(f"  scenarios {a:3d}-{b:3d}: {lr[:, a:b].mean():.4f}")
print("  true-parent rate in the generator ~0.059; plain-net label rate was 0.1045")
# paired: later generations vs g=5 (within chain)
print("\nwithin-chain change relative to generation 5 (paired over chains):")
for j, g in enumerate(gens[1:], 1):
    d = G[:, j] - G[:, 0]
    print(f"  gen {g:3d} - gen 5: {np.nanmean(d):+.3f} +- {np.nanstd(d, ddof=1)/np.sqrt(len(chains)):.3f}")
