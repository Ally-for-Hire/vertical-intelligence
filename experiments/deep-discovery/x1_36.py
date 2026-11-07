"""X1.36: ONE shared cell, reused at every depth and trained jointly on all its uses, in the puzzle room (author 2026-11-03: 'patterns IN the patterns and so on and so
forth as long as we get it compute'; goal: infinitely scaling depth from nothing with one simple rule). Same world, rooms, exam and seeds as X1.35 (vi/x1_35.py:
room_specs, one-rule deep world with thin ties, 5 tiers x 10 problems, rooms 1-4 = 5/10/15/20 levels, g = 0.8 / 1.2 / 2.0), so every cell pairs per world with the
X1.35 routes. Learner vi/cell.py: ONE tiny net + a wiring (rotation, parent) per instance; the compute knob is the number of joint refit steps per problem.
Arms (compute ladder, every rung an arm): CELL_0 (no joint refit: the cell stays as first proposed) | CELL_30 | CELL_150 | CELL_600.
Grades per room (library frozen): exam R^2 by tier, deepest level recovered (corr^2 >= 0.99), levels with corr^2 >= 0.9, functions linked per exam problem, total weights.
Usage: python x1_36.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_28, x1_35

OUT = x1_28.OUT; CACHE = os.path.join(OUT, "x1_36_cells"); GAINS = x1_35.GAINS; REFITS = (0, 30, 150, 600)   # CONST: x136_cfg


def cell_run(task):
    g, refit, seed = task; f = os.path.join(CACHE, f"g{g}_r{refit}_{seed}.json")
    if os.path.exists(f): return json.load(open(f))
    import cell, wstream, wrec
    wrec.GAIN = g; t0 = time.time(); NE = C("x132_exam"); rooms = C("x135_rooms"); L = cell.Learner(seed, dict(refit=refit)); out = dict(g=g, refit=refit, seed=seed, rooms=[])
    for k in range(1, rooms + 1):
        Lv = 5 * k; W = wstream.Stream(seed, dict(noise=0.0, n_tr=C("x128_pool"), lib_size=Lv, lib="rec")); data = [wstream.build(W.lib, sp, seed, W.cfg) for sp in x1_35.room_specs(k, k)]; r2 = []
        for i, d in enumerate(data): r2.append(L.step(d)["r2"]); x1_28.hb(seed, f"g{g}_r{refit}_room{k}", i + 1, 50, r2)
        rng = np.random.default_rng([seed, 7, k]); lvls = rng.integers(0, Lv, NE)
        exam = [wstream.build(W.lib, dict(t=90000 + k * 100 + j, S=[int(lv)], a=[1.0], stage=0, has_new=False), seed, W.cfg) for j, lv in enumerate(lvls)]
        ex = L.exam(exam); e = np.array([x[0] for x in ex]); tier = np.minimum(lvls // k, 4)
        xp = rng.standard_normal((C("x126_probe"), W.cfg["dim"])); V = W.lib.values(xp); rec = L.recovered(xp, V, list(range(Lv))); c2 = np.array([rec[j] ** 2 for j in range(Lv)])
        ok = [j for j in range(Lv) if c2[j] >= C("x128_recovered")]
        out["rooms"].append(dict(room=k, levels=Lv, exam=float(e.mean()), tiers=[float(e[tier == t].mean()) if (tier == t).any() else None for t in range(5)], solved=float(np.mean(e >= C("x128_recovered"))),
                                 reach=(max(ok) if ok else -1), n_rec=len(ok), n90=int((c2 >= 0.9).sum()), corr2_mean=float(c2.mean()), links=float(np.mean([x[1] for x in ex])), nodes=len(L.nodes),
                                 weights=L.size(), weights_per_level=L.size() / Lv, seconds=round(time.time() - t0, 1)))
    os.makedirs(CACHE, exist_ok=True); json.dump(out, open(f, "w")); return out


def run():
    import multiprocessing as mp, budget
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x128_seed0"); seeds = list(range(seed0, seed0 + C("x135_worlds"))); t0 = time.time()
    tasks = [(g, r, s) for g in GAINS for r in REFITS for s in seeds]
    with mp.get_context("spawn").Pool(budget.workers(per_worker_gb=1.0)) as pool: res = pool.map(cell_run, tasks, chunksize=1)
    med = lambda xs: float(np.median([x for x in xs if x is not None]))
    print(f"{'g':>4s} {'refit':>5s} {'room':>4s} {'exam':>6s} {'reach':>5s} {'n99':>4s} {'n90':>4s} {'c2mean':>6s} {'links':>5s} {'nodes':>5s} {'weights':>7s} {'sec':>6s}")
    for g in GAINS:
        for r in REFITS:
            rs = [x for x in res if x["g"] == g and x["refit"] == r]
            for k in range(C("x135_rooms")):
                R = [x["rooms"][k] for x in rs]
                print(f"{g:4.1f} {r:5d} {k + 1:4d} {med([x['exam'] for x in R]):6.3f} {med([x['reach'] for x in R]):5.0f} {med([x['n_rec'] for x in R]):4.0f} {med([x['n90'] for x in R]):4.0f} {med([x['corr2_mean'] for x in R]):6.2f} {med([x['links'] for x in R]):5.2f} {med([x['nodes'] for x in R]):5.0f} {med([x['weights'] for x in R]):7.0f} {med([x['seconds'] for x in R]):6.0f}")
    json.dump(dict(per_cell=res, seconds=round(time.time() - t0, 1)), open(os.path.join(OUT, "x1_36_run.json"), "w"), indent=1, default=float)
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
