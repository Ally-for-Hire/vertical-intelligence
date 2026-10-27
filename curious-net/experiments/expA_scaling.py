"""Exp A: structure + uncertainty vs number of distractor variables, curious vs random actions."""
import sys, json, time
sys.path.insert(0, "../src")
import run

out = {}
T = 15000
t0 = time.time()
for n_d in (0, 8, 64):
    for pol in ("random", "curious"):
        for seed in range(4):
            env, enc, m, log = run.run_learn(seed, n_d, T, pol, log_every=500)
            out[f"{n_d}|{pol}|{seed}"] = dict(log=log, edges=m.n_edges(), neurons=m.n_neurons(), Bs=m.Bs)
            print(n_d, pol, seed, "bits", round(log["bits"][-1], 1), "fp", log["junk"][-1], "aprec", round(log["prec"][-1], 2),
                  "arec", round(log["rec"][-1], 2), "edges", m.n_edges(), f"{time.time()-t0:.0f}s", flush=True)
json.dump(out, open("../results/expA.json", "w"), default=float)
