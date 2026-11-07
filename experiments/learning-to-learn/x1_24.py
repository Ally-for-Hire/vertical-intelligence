"""X1.24: does compositional depth built on any existing nodes (vi/comp.py) break the depth plateau found in X1.23?
Same long fixed worlds as X1.22/X1.23 (wstream, 24 concepts, never expanded, 240 problems, 60 train points). Arms: OPT (flat), COMP (product and bend
candidates every step), COMP_STAG (only when flat candidates stop paying). Scored on the last 120 problems by concept depth, as X1.23.
Usage: python x1_24.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import flat, comp, deep, wstream
import x1_19 as L19

OUT = L19.OUT
ARMS = {"OPT": "off", "COMP": "always", "COMP_STAG": "stag"}


def world(seed):
    t0 = time.time(); W = wstream.Stream(seed); data = [W.next() for _ in range(C("x122_problems"))]
    depth = [max(W.lib.c[s]["depth"] for s in d["S"]) for d in data]; ceil = [deep.ceiling(d, L19.CEIL) for d in data]
    out = {}
    for a, mode in ARMS.items():
        L = comp.Learner(seed, {"deep_mode": mode}); sh = [L.step(d) / c for d, c in zip(data, ceil)]
        tail = list(range(len(data) - C("x123_tail"), len(data)))
        g = lambda lo, hi: [sh[i] for i in tail if lo <= depth[i] <= hi]
        k, lk = L.kinds()
        out[a] = dict(all=float(np.mean([sh[i] for i in tail])), d0=float(np.mean(g(0, 0))) if g(0, 0) else float("nan"),
                      d1=float(np.mean(g(1, 1))) if g(1, 1) else float("nan"), d2=float(np.mean(g(2, 99))) if g(2, 99) else float("nan"),   # CONST: x123_tail
                      early=float(np.mean(sh[:C("x122_block")])), nodes=k, linked=lk)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run():
    import multiprocessing as mp, budget
    from x0_1 import boot_ci
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x124_seed0"); t0 = time.time(); seeds = list(range(seed0, seed0 + C("x119_worlds")))
    with mp.get_context("spawn").Pool(budget.workers()) as pool: res = pool.map(world, seeds, chunksize=1)
    per = dict(zip(seeds, res))
    g = lambda a, k: np.array([per[s][a][k] for s in seeds], float)
    pair = lambda a, b, k: boot_ci((g(a, k) - g(b, k))[~np.isnan(g(a, k) - g(b, k))], seed0)
    S = dict(worlds=len(seeds), med={a: {k: float(np.nanmedian(g(a, k))) for k in ("all", "d0", "d1", "d2", "early")} for a in ARMS},
             kinds={a: {kk: float(np.median([per[s][a]["linked"][kk] for s in seeds])) for kk in ("flat", "bend", "prod")} for a in ARMS},
             d2_comp=pair("COMP", "OPT", "d2"), all_comp=pair("COMP", "OPT", "all"), d2_stag=pair("COMP_STAG", "OPT", "d2"), stag_vs_comp=pair("COMP_STAG", "COMP", "all"))
    f = lambda k: f"{S[k][0]:+.4f} [{S[k][1]:+.4f}, {S[k][2]:+.4f}]"; up = lambda k: S[k][1] > 0
    V = [("P478", "Compositional depth on any nodes raises the share of ceiling on problems with a concept 2+ levels deep: COMP minus OPT, above 0", "PASSED" if up("d2_comp") else "FAILED", f("d2_comp")),
         ("P479", "...and on all post-plateau problems: COMP minus OPT, above 0", "PASSED" if up("all_comp") else "FAILED", f("all_comp")),
         ("P480", "Depth added only on stagnation raises the deep-problem share: COMP_STAG minus OPT, above 0", "PASSED" if up("d2_stag") else "FAILED", f("d2_stag")),
         ("P481", "Depth only on stagnation beats depth always on offer, all post-plateau problems: COMP_STAG minus COMP, above 0", "PASSED" if up("stag_vs_comp") else "FAILED", f("stag_vs_comp"))]
    out = {"per_world": per, "summary": S, "verdicts": V}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_24_run.json"), "w"), indent=1, default=float)
    for a in ARMS: print(a, {k: round(v, 3) for k, v in S["med"][a].items()}, "linked", S["kinds"][a])
    for v in V: print(v[0], v[2], "|", v[3])
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
