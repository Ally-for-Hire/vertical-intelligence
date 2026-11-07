"""Exp P2: remove the two fairness caveats of exp P. (1) group-lasso strength tuned over a WIDE grid on the validation worlds (seeds 1000-1011); (2) the raw-window CNN prior
trained 5x longer (1500 steps) and on K=100 and K=300 training scenarios. Test on the same 48 worlds (9000-9047) as exp P."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool
import expP_baselines as P


def main():
    from selfnn import SeqPrior
    t0 = time.time()
    with Pool(20) as p:
        grid = [("gl", v) for v in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)] + [("none", 0.0)]
        tr = p.map(P.tune_job, [(1000 + s, k, v) for s in range(12) for (k, v) in grid])
        sc = {}
        for k, v, e in tr:
            sc.setdefault((k, v), []).append(e)
        sc = {k: float(np.mean(v)) for k, v in sc.items()}
        print("validation first-1000 error (wide group-lasso grid):", {str(k): round(v, 3) for k, v in sc.items()}, flush=True)
        gl = min([k for k in sc if k[0] == "gl"], key=lambda k: sc[k])[1]
        print("chosen group-lasso", gl, flush=True)
        trn = p.map(P.train_job, range(5000, 5300))
        print("training data done", round(time.time() - t0), "s", flush=True)
        priors = {}
        for K in (100, 300):
            y = np.concatenate([t["lab"] for t in trn[:K]])
            S = np.concatenate([t["S"] for t in trn[:K]]); Fl = np.concatenate([t["Fl"] for t in trn[:K]])
            sp = SeqPrior(steps=1500, seed=K); sp.fit(S, Fl, y); priors[f"seq_long{K}"] = sp
        print("seq priors fitted", round(time.time() - t0), "s", flush=True)
        specs = [("plain", "plain", None), ("gl", "plain", None)] + [(k, "prior", m) for k, m in priors.items()]
        te = p.map(P.test_job, [(s, specs, gl, 0.0) for s in range(9000, 9048)])
    json.dump(dict(test=te, gl=gl, val={str(k): v for k, v in sc.items()}), open("../results/expP2.json", "w"), default=float)
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
