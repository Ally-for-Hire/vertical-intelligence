"""X1.37: UNCAPPED LABELS (author 2026-11-03: 'give more data ... uncapped testing data for our current algorithm until it begins to understand; I think our issue is the equipment not the experiment').
Same learner as X1.36 (vi/cell.py, shared tied cell, refit 150) and same one-rule world / room 1 (50 problems, levels 0-4) / exam, but each problem now has N labels (and N/2 validation points)
instead of 60. N climbs a ladder 60 -> 200 -> 600 -> 2000 -> 6000; per steepness g the climb STOPS when the cell has understood (a majority of worlds has a concept at level >= 1 with corr^2 >= 0.99)
or has saturated (exam gain < 0.02 and no new levels at corr^2 >= 0.9 over the previous rung): both are results, a stalled rung is a failed route. Rungs run in order so compute is spent only where needed.
Usage: python x1_37.py run | python x1_37.py run 2.0   (the second form = continuation of one steepness; saturation needs 2 flat rungs in a row, since one rung at n=3 worlds is noisy; cached rungs are reused)"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_28, x1_35

OUT = x1_28.OUT; CACHE = os.path.join(OUT, "x1_37_cells"); GAINS = x1_35.GAINS; LADDER = (60, 200, 600, 2000, 6000); SEEDS = 3; REFIT = 150   # CONST: x137_cfg


def cell_run(task):
    g, N, seed = task; f = os.path.join(CACHE, f"g{g}_n{N}_{seed}.json")
    if os.path.exists(f): return json.load(open(f))
    import cell, wstream, wrec
    wrec.GAIN = g; t0 = time.time(); NE = C("x132_exam"); k = 1; Lv = 5 * k; L = cell.Learner(seed, dict(refit=REFIT, n_lab=N))
    W = wstream.Stream(seed, dict(noise=0.0, n_tr=N, n_va=max(60, N // 2), lib_size=Lv, lib="rec")); data = [wstream.build(W.lib, sp, seed, W.cfg) for sp in x1_35.room_specs(k, k)]; r2 = []
    for i, d in enumerate(data): r2.append(L.step(d)["r2"]); x1_28.hb(seed, f"g{g}_n{N}", i + 1, 50, r2)
    rng = np.random.default_rng([seed, 7, k]); lvls = rng.integers(0, Lv, NE)
    exam = [wstream.build(W.lib, dict(t=90000 + j, S=[int(lv)], a=[1.0], stage=0, has_new=False), seed, W.cfg) for j, lv in enumerate(lvls)]
    e = np.array([x[0] for x in L.exam(exam)]); xp = rng.standard_normal((C("x126_probe"), W.cfg["dim"])); V = W.lib.values(xp); rec = L.recovered(xp, V, list(range(Lv))); c2 = np.array([rec[j] ** 2 for j in range(Lv)])
    out = dict(g=g, N=N, seed=seed, exam=float(e.mean()), c2=[float(x) for x in c2], n99=int((c2 >= 0.99).sum()), deep99=int((c2[1:] >= 0.99).sum()), n90=int((c2 >= 0.9).sum()), nodes=len(L.nodes), weights=L.size(), seconds=round(time.time() - t0, 1))
    os.makedirs(CACHE, exist_ok=True); json.dump(out, open(f, "w")); return out


def run():
    import multiprocessing as mp, budget
    for kk in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(kk, "1")
    seed0 = C("x128_seed0"); seeds = list(range(seed0, seed0 + SEEDS)); only = [float(a) for a in sys.argv[2:]]; PAT = 2 if only else 1; flat = {g: 0 for g in GAINS}; active = {g: (g in only if only else True) for g in GAINS}; prev = {}; t0 = time.time(); res = []
    for N in LADDER:
        gs = [g for g in GAINS if active[g]]
        if not gs: break
        with mp.get_context("spawn").Pool(budget.workers(per_worker_gb=2.0 if N >= 2000 else 1.0)) as pool: R = pool.map(cell_run, [(g, N, s) for g in gs for s in seeds], chunksize=1)
        res += R
        for g in gs:
            r = [x for x in R if x["g"] == g]; ex = float(np.median([x["exam"] for x in r])); n90 = float(np.median([x["n90"] for x in r])); deep = sum(x["deep99"] > 0 for x in r)
            print(f"g {g} N {N}: exam {ex:.3f} n90 {n90:.0f} worlds with a deep concept at corr2>=0.99: {deep}/{len(r)} median max corr2 {np.median([max(x['c2'][1:]) for x in r]):.3f} {np.median([x['seconds'] for x in r]):.0f}s", flush=True)
            if deep * 2 > len(r): active[g] = False; print(f"  g {g}: UNDERSTOOD at N={N}", flush=True)
            elif g in prev and ex - prev[g][0] < 0.02 and n90 <= prev[g][1]:
                flat[g] += 1
                if flat[g] >= PAT: active[g] = False; print(f"  g {g}: SATURATED at N={N} (failed route)", flush=True)
            else: flat[g] = 0
            prev[g] = (ex, n90)
    json.dump(dict(per_cell=res, seconds=round(time.time() - t0, 1)), open(os.path.join(OUT, ("x1_37b_run.json" if only else "x1_37_run.json")), "w"), indent=1, default=float)
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
