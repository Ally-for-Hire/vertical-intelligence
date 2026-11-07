import json, numpy as np
te = json.load(open("../results/expR.json"))["test"]
plain = np.array([np.mean(r["arms"]["plain"][:1000]) for r in te])
print("fly-like worlds: 48 worlds; mean true parents per output %.1f; plain first-1000 error %.3f" % (np.mean([r["n_parents"] for r in te]), plain.mean()))
og = None
print("gain over plain per-output MLP in first-1000-tick error (higher = better):")
for a in ("oracle_all", "oracle_top3", "screen", "prior_std", "prior_fly"):
    d = plain - np.array([np.mean(r["arms"][a][:1000]) for r in te])
    d3 = np.array([np.mean(r["arms"]["plain"][:300]) - np.mean(r["arms"][a][:300]) for r in te])
    if a == "oracle_all":
        og = d.mean()
    print(f"  {a:12s} first-1000 {d.mean():+.3f} +- {d.std(ddof=1)/np.sqrt(len(d)):.3f}   first-300 {d3.mean():+.3f}   share of oracle_all {100*d.mean()/og:5.1f}%")
nd = np.array([r["nd"] for r in te])
print("\nby number of distractors (oracle_all / oracle_top3 / prior_std / prior_fly gain):")
for n in sorted(set(nd)):
    m = nd == n
    print(n, int(m.sum()), ["%+.3f" % (plain[m] - np.array([np.mean(r["arms"][a][:1000]) for r, k in zip(te, m) if k])).mean() for a in ("oracle_all", "oracle_top3", "prior_std", "prior_fly")])
x = np.array([np.mean(r["arms"]["prior_fly"][:1000]) - np.mean(r["arms"]["prior_std"][:1000]) for r in te])
print("\npaired prior_fly - prior_std error: %+.3f +- %.3f (negative = in-family training better)" % (x.mean(), x.std(ddof=1) / np.sqrt(len(x))))
y = np.array([np.mean(r["arms"]["oracle_top3"][:1000]) - np.mean(r["arms"]["oracle_all"][:1000]) for r in te])
print("oracle_top3 - oracle_all error: %+.3f +- %.3f (positive = the weak inputs matter for accuracy)" % (y.mean(), y.std(ddof=1) / np.sqrt(len(y))))
