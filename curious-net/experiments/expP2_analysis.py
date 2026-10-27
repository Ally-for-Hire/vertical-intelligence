import json, numpy as np
D2 = json.load(open("../results/expP2.json"))
D1 = json.load(open("../results/expP.json"))
print("validation (wide group-lasso grid):", {k: round(v, 3) for k, v in D2["val"].items()}, "-> chosen", D2["gl"])
te2, te1 = D2["test"], D1["test"]
plain = np.array([np.mean(r["arms"]["plain"][:1000]) for r in te2])
og = (np.array([np.mean(r["arms"]["plain"][:1000]) for r in te1]) - np.array([np.mean(r["arms"]["oracle"][:1000]) for r in te1])).mean()
print("oracle gain (exp P, same worlds) %+.3f" % og)
for a in ("gl", "seq_long100", "seq_long300"):
    d = plain - np.array([np.mean(r["arms"][a][:1000]) for r in te2])
    print(f"  {a:12s} gain {d.mean():+.3f} +- {d.std(ddof=1)/np.sqrt(len(d)):.3f}  ({100*d.mean()/og:.1f}% of oracle)")
nn = np.array([np.mean(r["arms"]["plain"][:1000]) - np.mean(r["arms"]["nn_soft100"][:1000]) for r in te1])
print("  nn_soft100 (hand-built descriptors, exp P) gain %+.3f (%.1f%% of oracle)" % (nn.mean(), 100 * nn.mean() / og))
for a in ("seq_long100", "seq_long300"):
    x = np.array([np.mean(r["arms"][a][:1000]) for r in te2]) - np.array([np.mean(r["arms"]["nn_soft100"][:1000]) for r in te1])
    print(f"  paired error {a} - nn_soft100: {x.mean():+.3f} +- {x.std(ddof=1)/np.sqrt(len(x)):.3f}  (positive = descriptors better)")
