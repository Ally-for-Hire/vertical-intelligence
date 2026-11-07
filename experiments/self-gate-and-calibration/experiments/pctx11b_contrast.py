"""PIECE-CTX11b: CTX11 with HIGH rule contrast between regimes, and a registered manipulation check (P180: context-free >= 1.5 x true-regime routing; else P181-P182 void).
Regimes (same wiring): 0 = the base coefficients, 1 = all coefficients sign-flipped, 2 = a random half of the coefficients flipped; gains redrawn per regime as B1. Everything else as CTX11 (same arms, seeds 9000-9047)."""
import os, sys, json, time, copy
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np
from multiprocessing import Pool
import pb1_memory as B
import pctx11_transfer as C


def make_contrast(seed):
    from worlds import World
    w = World(seed, n_state=6, n_act=2, n_distract=10, noise=float(os.environ.get("NOISE", "0.05")), max_parents=2)
    r = np.random.default_rng(seed + 31337)
    regimes = []
    for k in range(3):
        coef = []
        for c in w.coef:
            c = np.array(c, dtype=float)
            s = np.ones(len(c)) if k == 0 else (-np.ones(len(c)) if k == 1 else np.where(r.random(len(c)) < 0.5, -1.0, 1.0))
            coef.append(c * s)
        regimes.append((coef, [float(r.uniform(0.1, 0.6)) for _ in w.gain]))
    return w, regimes


B.make = make_contrast

if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(C.run, range(int(os.environ.get("OFF", "9000")), int(os.environ.get("OFF", "9000")) + 48))
    json.dump(res, open("../results/pctx11b_contrast_n%s_%d%s.json" % (os.environ.get("NOISE", "0.05"), int(os.environ.get("OFF", "9000")), "_fix" if C.FIX else ""), "w"))
    for a in C.ARMS:
        print(f"{a:9s} x oracle {np.median([r[a]['err'] / r['oracle']['err'] for r in res]):.2f} | / lastlabel {np.median([r[a]['err'] / r['lastlabel']['err'] for r in res]):.2f} | regime accuracy {np.median([r[a]['acc'] for r in res]):.3f} | shocks {np.median([r[a]['shocks'] for r in res]):.4f} | asked {np.median([r[a]['asks'] for r in res]):.4f}")
    print("done", round(time.time() - t0), "s")
