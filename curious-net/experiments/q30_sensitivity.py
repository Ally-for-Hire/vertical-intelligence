"""Sensitivity of the fused student to its orientation assumption (initial source accuracy) using the cached worlds from q30_source_reliability.py"""
import sys, pickle
sys.path.insert(0, ".")
import numpy as np
import q30_source_reliability as Q30

cache = pickle.load(open("../results/q30_cache.pkl", "rb"))
for init in (0.55, 0.7, 0.9):
    print(f"-- initial source accuracy {init}")
    for kind in ("honest", "random", "systematic"):
        accs, nexp, ta = [], [], []
        for rep in range(30):
            rng = np.random.default_rng(1000 + rep)
            order = rng.permutation(len(cache))
            r, ds = Q30.simulate(cache, kind, order, rng, init)
            accs.append(np.mean(r["fused"][10:])); nexp.append(np.mean(r["n_exp_fused"][10:]))
            ta.append((ds.sens[0] + ds.spec[0]) / 2)
        print(f"   teacher {kind:10s} fused accuracy {np.mean(accs):.3f} +- {np.std(accs, ddof=1)/np.sqrt(len(accs)):.3f}  experiments/world {np.mean(nexp):.1f}  learned teacher reliability {np.mean(ta):.3f}")
