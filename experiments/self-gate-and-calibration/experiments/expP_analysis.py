import json, numpy as np
D = json.load(open("../results/expP.json"))
te = D["test"]
print("chosen group-lasso", D["gl"], "dropout", D["dr"])
plain = np.array([np.mean(r["arms"]["plain"][:1000]) for r in te])
og = None
print("48 test worlds, gain over plain per-output MLP in first-1000-tick error (higher = better):")
for a in te[0]["arms"]:
    if a == "plain":
        continue
    d = plain - np.array([np.mean(r["arms"][a][:1000]) for r in te])
    d3 = np.array([np.mean(r["arms"]["plain"][:300]) - np.mean(r["arms"][a][:300]) for r in te])
    if a == "oracle":
        og = d.mean()
    print(f"  {a:12s} first-1000 gain {d.mean():+.3f} +- {d.std(ddof=1)/np.sqrt(len(d)):.3f}   first-300 gain {d3.mean():+.3f}")
print("\nshare of oracle headroom (first-1000):")
for a in te[0]["arms"]:
    if a in ("plain", "oracle"):
        continue
    d = (plain - np.array([np.mean(r["arms"][a][:1000]) for r in te])).mean()
    print(f"  {a:12s} {100*d/og:5.1f}%")
# paired differences between the main contenders
def diff(a, b):
    x = np.array([np.mean(r["arms"][a][:1000]) - np.mean(r["arms"][b][:1000]) for r in te])
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))
print("\npaired error differences (negative = first is better):")
for a, b in (("nn_soft100", "seq_soft100"), ("nn_soft100", "gl"), ("seq_soft100", "gl"), ("nn_soft10", "seq_soft10"), ("nn_soft100", "screen")):
    m, s = diff(a, b)
    print(f"  {a} - {b}: {m:+.3f} +- {s:.3f}")
