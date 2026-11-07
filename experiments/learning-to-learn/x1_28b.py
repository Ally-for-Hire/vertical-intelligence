"""X1.28b: every route of the digger's design choices (author rule 2026-11-02: 'whenever a decision must be made, you must try every single route
and record outcomes'). One choice varied at a time around X1.28's defaults, same seeds as X1.28 (960000+), same world and measures.
Routes: chosen inputs (of 60) 0/20/40/50 [grow]; parents of a new function 0/2/4 [grow]; hidden units per concept node 3/6/12 [grow];
MDL charge per weight 0.25/0.5/1 x log2(n) [mdl]; starting budget 300/600/1200 weights [budget]. Interactions are NOT covered (one-at-a-time).
Usage: python x1_28b.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import digger, wstream, x1_28

OUT = os.path.join(str(vipaths.RESULTS))
ROUTES = {   # NOTE: since 2026-11-03 digger.DEFAULT is all-random; these routes were run with the earlier default rand=20,chosen=40 (pass them explicitly to reproduce)
          "choose_0": dict(size="grow", rand=60, chosen=0), "choose_20": dict(size="grow", rand=40, chosen=20), "choose_40": dict(size="grow"),   # CONST: x128_cfg
          "choose_50": dict(size="grow", rand=10, chosen=50), "par_0": dict(size="grow", n_par=0), "par_4": dict(size="grow", n_par=4),          # CONST: x128_cfg
          "hid_3": dict(size="grow", hid=3), "hid_12": dict(size="grow", hid=12), "mdl_025": dict(size="mdl", mdl_w=0.25), "mdl_05": dict(size="mdl"),   # CONST: x128_cfg
          "mdl_1": dict(size="mdl", mdl_w=1.0), "budget_300": dict(size="budget", budget0=300), "budget_600": dict(size="budget"), "budget_1200": dict(size="budget", budget0=1200)}   # CONST: x128_cfg


def world(seed):
    t0 = time.time(); T = C("x128_problems"); bs = C("x122_block")
    W = wstream.Stream(seed, dict(noise=0.0, n_tr=C("x128_pool"))); data = [W.next() for _ in range(T)]
    out = {}
    for a, cfg in ROUTES.items():
        L = digger.Learner(seed, cfg); r2 = []
        for t, d in enumerate(data): r2.append(L.step(d)["r2"]); x1_28.hb(seed, a, t + 1, T, r2)
        out[a] = dict(blocks=[float(np.mean(r2[i:i + bs])) for i in range(0, T, bs)], last=float(np.mean(r2[-bs:])), size=L.size(), nodes=len(L.units))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run():
    import multiprocessing as mp, budget
    from x0_1 import boot_ci
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x128_seed0"); t0 = time.time(); seeds = list(range(seed0, seed0 + C("x119_worlds")))
    with mp.get_context("spawn").Pool(budget.workers()) as pool: res = pool.map(world, seeds, chunksize=1)
    per = dict(zip(seeds, res)); g = lambda a, k: np.array([per[s][a][k] for s in seeds], float)
    base = {"choose": "choose_40", "par": "choose_40", "hid": "choose_40", "mdl": "mdl_05", "budget": "budget_600"}
    S = dict(worlds=len(seeds), med={a: dict(last=float(np.median(g(a, "last"))), size=float(np.median(g(a, "size"))), nodes=float(np.median(g(a, "nodes")))) for a in ROUTES},
             vs_default={a: boot_ci(g(a, "last") - g(base[a.split("_")[0]], "last"), seed0) for a in ROUTES if a != base[a.split("_")[0]]})
    out = {"per_world": per, "summary": S}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_28b_run.json"), "w"), indent=1, default=float)
    for a in ROUTES:
        d = S["vs_default"].get(a); print(f"{a:12s} last R2 {S['med'][a]['last']:.3f} size {S['med'][a]['size']:.0f} nodes {S['med'][a]['nodes']:.0f}" + (f"  vs default {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]" if d else "  (default)"))
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
