import json, numpy as np
N1 = json.load(open("../results/expN1.json"))["test"]
sub = [r for r in N1 if 9000 <= r["seed"] < 9024]
plain = np.array([np.mean(r["arms"]["plain"][:1000]) for r in sub])
print("same 24 worlds (seeds 9000-9023):")
for a in ("oracle", "prior10", "prior30", "prior100", "prior300", "prior_oracle100", "prior_oracle300"):
    d = plain - np.array([np.mean(r["arms"][a][:1000]) for r in sub])
    print(f"  {a:16s} gain {d.mean():+.3f} +- {d.std(ddof=1)/np.sqrt(len(d)):.3f}")
D = json.load(open("../results/expN2.json"))["test"]
for g in (20, 40, 150):
    per_world = []
    for r in D:
        ks = [k for k in r["arms"] if k.endswith(f"_g{g}")]
        if ks:
            per_world.append((r["seed"], np.mean([np.mean(r["arms"][k][:1000]) for k in ks])))
    # average over chains per world then paired against plain
    byseed = {}
    for s, v in per_world:
        byseed.setdefault(s, []).append(v)
    d = np.array([plain[i] - np.mean(byseed[r["seed"]]) for i, r in enumerate(sub)])
    print(f"  recursive gen {g:3d} (mean of 8 chains per world) gain {d.mean():+.3f} +- {d.std(ddof=1)/np.sqrt(len(d)):.3f}")
