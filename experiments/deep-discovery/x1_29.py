"""X1.29: HUGE COMPUTE (author 2026-11-03: 'make random a choice and give the models huge compute and time so they can really work. we want to see the
absolute best out of every idea'). Same noise-free world and seeds as X1.28 (first x129_worlds), every idea with ~10x the search/training budget
(BIG cfg, x129_big) and random input selection as one of the learner's CHOICES: per problem the learner learns it both ways (60 random, or 20 random + 40
disagreement-chosen) on copies and keeps the one with the lower decision-point error ('auto'). Every route is an arm (author rule): size reading
{mdl, grow, budget} x pick {rand, mix, auto} + FLAT_BIG. Pairing: each arm vs its default-compute X1.28 result on the same world (does compute help?).
Usage: python x1_29.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_28

OUT = x1_28.OUT
BIG = dict(n_sub=8, n_par=4, n_props=16, unit_steps=600, max_steps=16, ft_steps=1000, ft_patience=100, stall=0.001)   # CONST: x129_big
ARMS = {f"{sz.upper()}_{pk}": dict(BIG, size=sz, pick=pk, rand=20, chosen=40) for sz in ("mdl", "grow", "budget") for pk in ("rand", "mix", "auto")}
ARMS["FLAT_BIG"] = {"flat": {k: BIG[k] for k in ("n_props", "unit_steps", "max_steps", "ft_steps", "ft_patience", "stall")}}


CACHE = os.path.join(OUT, "x1_29_worlds"); PROG = os.path.join(OUT, "x1_29_progress.json")


def world(seed):
    """One world, cached on disk: a stopped/restarted run (vi/ctl.py) resumes without redoing finished worlds."""
    f = os.path.join(CACHE, f"{seed}.json")
    if os.path.exists(f): return json.load(open(f))
    x1_28.ARMS = ARMS; r = x1_28.world(seed); os.makedirs(CACHE, exist_ok=True); json.dump(r, open(f, "w"), default=float); return r


def run():
    import multiprocessing as mp, budget
    from x0_1 import boot_ci
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x128_seed0"); t0 = time.time(); seeds = list(range(seed0, seed0 + C("x129_worlds")))
    res = {}
    with mp.get_context("spawn").Pool(budget.workers()) as pool:
        for sd, r in zip(seeds, pool.imap(world, seeds, chunksize=1)):      # imap keeps order; progress is written as worlds finish
            res[sd] = r; json.dump(dict(done=len(res), total=len(seeds), seconds=round(time.time() - t0), updated=time.time()), open(PROG, "w"))
    per = res; g = lambda a, k: np.array([per[s][a][k] for s in seeds], float)
    old = json.load(open(os.path.join(OUT, "x1_28_run.json")))["result"]["per_world"]
    o = lambda a, k: np.array([old[str(s)][a][k] for s in seeds], float)
    pairs = {"MDL": "DIG_MDL", "GROW": "DIG_GROW", "BUDGET": "DIG_BUDGET"}
    S = dict(worlds=len(seeds), med={a: {k: float(np.nanmedian(g(a, k))) for k in ("last", "solved", "rec_end_d0", "rec_end_d1", "rec_end_d2", "size", "nodes")} for a in ARMS},
             vs_default_compute={a: boot_ci(g(a, "last") - o(pairs[a.split("_")[0]], "last"), seed0) for a in ARMS if a.split("_")[0] in pairs},
             flat_vs_default=boot_ci(g("FLAT_BIG", "last") - o("FLAT", "last"), seed0),
             auto_vs_rand={s: boot_ci(g(f"{s}_auto", "last") - g(f"{s}_rand", "last"), seed0) for s in ("MDL", "GROW", "BUDGET")},
             mix_vs_rand={s: boot_ci(g(f"{s}_mix", "last") - g(f"{s}_rand", "last"), seed0) for s in ("MDL", "GROW", "BUDGET")},
             best_vs_flat={a: boot_ci(g(a, "last") - g("FLAT_BIG", "last"), seed0) for a in ARMS if a != "FLAT_BIG"},
             curve={a: [float(np.median([per[s][a]["blocks"][i] for s in seeds])) for i in range(len(per[seeds[0]][a]["blocks"]))] for a in ARMS})
    out = {"per_world": per, "summary": S}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_29_run.json"), "w"), indent=1, default=float)
    f = lambda r: f"{r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]"
    for a in ARMS: print(a, {k: round(v, 3) for k, v in S["med"][a].items()}, "curve", " ".join(f"{x:.3f}" for x in S["curve"][a]))
    print("compute effect (arm minus its default-compute X1.28 arm):"); [print(" ", a, f(r)) for a, r in S["vs_default_compute"].items()]; print("  FLAT_BIG", f(S["flat_vs_default"]))
    print("auto minus rand:", {k: f(v) for k, v in S["auto_vs_rand"].items()}); print("mix minus rand:", {k: f(v) for k, v in S["mix_vs_rand"].items()})
    print("vs FLAT_BIG:"); [print(" ", a, f(r)) for a, r in S["best_vs_flat"].items()]
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
