import json, numpy as np
res = json.load(open("../results/q25b.json"))
print("paired accuracy differences over 48 worlds (second minus first):")
for teacher in ("honest", "random", "systematic"):
    for a, b in (("credulous", "targeted"), ("prior_only", "targeted"), ("verify_all", "targeted"), ("credulous", "verify_all")):
        d = np.array([r[teacher]["res"][b][0] - r[teacher]["res"][a][0] for r in res])
        print(f"  teacher {teacher:10s} {b:10s} - {a:10s}: {d.mean():+.3f} +- {d.std(ddof=1)/np.sqrt(len(d)):.3f}")
# accuracy gained per experiment for the targeted student relative to credulous, systematic teacher
d = np.array([r["systematic"]["res"]["targeted"][0] - r["systematic"]["res"]["credulous"][0] for r in res])
n = np.array([r["systematic"]["res"]["targeted"][1] for r in res])
print("\nsystematic teacher: targeted student gains %.3f accuracy with %.1f experiments per world (= %.3f accuracy per experiment); claims per world %.1f"
      % (d.mean(), n.mean(), d.mean() / n.mean(), np.mean([r["systematic"]["n_claims"] for r in res])))
# how good are the experiments themselves? accuracy of verify_all under an honest teacher is the experiment accuracy
print("single-experiment accuracy (verify_all, any teacher): %.3f" % np.mean([r["honest"]["res"]["verify_all"][0] for r in res]))
