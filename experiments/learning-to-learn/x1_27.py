"""X1.27: do CONCEPT NODES (tiny 2-layer nets, vi/subnet.py) close the depth gap? Fixed worlds and scoring as X1.23 (240 problems, last 120 by depth).
Arms: OPT (flat), SUB (concept nodes offered every step), SUB_STAG (only on stagnation), SUB_SLEEP (every step + consolidation every 30 problems).
Usage: python x1_27.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import subnet, deep, wstream
import x1_19 as L19

OUT = L19.OUT
ARMS = {"OPT": {"sub_mode": "off"}, "SUB": {"sub_mode": "always"}, "SUB_STAG": {"sub_mode": "stag"}, "SUB_SLEEP": {"sub_mode": "always", "sleep_every": C("x125_every")}}


def world(seed):
    t0 = time.time(); W = wstream.Stream(seed); data = [W.next() for _ in range(C("x122_problems"))]
    depth = [max(W.lib.c[s]["depth"] for s in d["S"]) for d in data]; ceil = [deep.ceiling(d, L19.CEIL) for d in data]
    tail = list(range(len(data) - C("x123_tail"), len(data))); out = {}
    for a, cfg in ARMS.items():
        L = subnet.Learner(seed, cfg); sh = [L.step(d) / c for d, c in zip(data, ceil)]
        g = lambda lo, hi: [sh[i] for i in tail if lo <= depth[i] <= hi]
        made, linked = L.kinds()
        out[a] = dict(all=float(np.mean([sh[i] for i in tail])), d0=float(np.mean(g(0, 0))) if g(0, 0) else float("nan"),
                      d1=float(np.mean(g(1, 1))) if g(1, 1) else float("nan"), d2=float(np.mean(g(2, 99))) if g(2, 99) else float("nan"),   # CONST: x123_tail
                      sub_made=made["s"], sub_linked=linked["s"])
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run():
    import multiprocessing as mp, budget
    from x0_1 import boot_ci
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x127_seed0"); t0 = time.time(); seeds = list(range(seed0, seed0 + C("x119_worlds")))
    with mp.get_context("spawn").Pool(budget.workers()) as pool: res = pool.map(world, seeds, chunksize=1)
    per = dict(zip(seeds, res)); g = lambda a, k: np.array([per[s][a][k] for s in seeds], float)
    pair = lambda a, b, k: boot_ci((g(a, k) - g(b, k))[~np.isnan(g(a, k) - g(b, k))], seed0)
    S = dict(worlds=len(seeds), med={a: {k: float(np.nanmedian(g(a, k))) for k in ("all", "d0", "d1", "d2", "sub_made", "sub_linked")} for a in ARMS},
             d2_sub=pair("SUB", "OPT", "d2"), all_sub=pair("SUB", "OPT", "all"), d2_stag=pair("SUB_STAG", "OPT", "d2"), d2_sleep=pair("SUB_SLEEP", "SUB", "d2"),
             all_stag=pair("SUB_STAG", "OPT", "all"), all_sleep=pair("SUB_SLEEP", "OPT", "all"))
    f = lambda k: f"{S[k][0]:+.4f} [{S[k][1]:+.4f}, {S[k][2]:+.4f}]"; up = lambda k: S[k][1] > 0
    V = [("P489", "Concept nodes raise the share of ceiling on post-plateau problems with a 2+-deep concept: SUB minus OPT, above 0", "PASSED" if up("d2_sub") else "FAILED", f("d2_sub")),
         ("P490", "...and on all post-plateau problems: SUB minus OPT, above 0", "PASSED" if up("all_sub") else "FAILED", f("all_sub")),
         ("P491", "Concept nodes offered only on stagnation raise the deep-problem share: SUB_STAG minus OPT, above 0", "PASSED" if up("d2_stag") else "FAILED", f("d2_stag")),
         ("P492", "Consolidation helps concept nodes on deep problems: SUB_SLEEP minus SUB, above 0", "PASSED" if up("d2_sleep") else "FAILED", f("d2_sleep"))]
    out = {"per_world": per, "summary": S, "verdicts": V}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_27_run.json"), "w"), indent=1, default=float)
    for a in ARMS: print(a, {k: round(v, 3) for k, v in S["med"][a].items()})
    print("all: STAG-OPT", f("all_stag"), "SLEEP-OPT", f("all_sleep"))
    for v in V: print(v[0], v[2], "|", v[3])
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
