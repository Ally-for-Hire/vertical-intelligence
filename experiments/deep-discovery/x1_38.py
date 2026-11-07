"""X1.38: is the limit the cell's CAPACITY / optimizer budget, not the data? (X1.37: up to 2000 labels per problem did not give an exact deep concept at any steepness.)
Same cell learner, world, room 1 and exam as X1.37 at N=600 labels; arms: hidden units 6 / 24 / 96 x refit steps 150 / 600; g = 0.8 and 1.2; 3 worlds. Grades: exam, deepest-level corr^2 (max over levels >= 1), levels at corr^2 >= 0.9 / 0.99.
Usage: python x1_38.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_28, x1_35

OUT = x1_28.OUT; CACHE = os.path.join(OUT, "x1_38_cells"); N = 600; HIDS = (6, 24, 96); REFITS = (150, 600); GS = (0.8, 1.2); SEEDS = 3   # CONST: x138_cfg


def cell_run(task):
    g, hid, refit, seed = task; f = os.path.join(CACHE, f"g{g}_h{hid}_r{refit}_{seed}.json")
    if os.path.exists(f): return json.load(open(f))
    import cell, wstream, wrec
    wrec.GAIN = g; t0 = time.time(); NE = C("x132_exam"); k = 1; Lv = 5; L = cell.Learner(seed, dict(refit=refit, n_lab=N, hid=hid))
    W = wstream.Stream(seed, dict(noise=0.0, n_tr=N, n_va=N // 2, lib_size=Lv, lib="rec")); data = [wstream.build(W.lib, sp, seed, W.cfg) for sp in x1_35.room_specs(k, k)]; r2 = []
    for i, d in enumerate(data): r2.append(L.step(d)["r2"]); x1_28.hb(seed, f"g{g}_h{hid}_r{refit}", i + 1, 50, r2)
    rng = np.random.default_rng([seed, 7, k]); lvls = rng.integers(0, Lv, NE)
    exam = [wstream.build(W.lib, dict(t=90000 + j, S=[int(lv)], a=[1.0], stage=0, has_new=False), seed, W.cfg) for j, lv in enumerate(lvls)]
    e = np.array([x[0] for x in L.exam(exam)]); xp = rng.standard_normal((C("x126_probe"), W.cfg["dim"])); V = W.lib.values(xp); rec = L.recovered(xp, V, list(range(Lv))); c2 = np.array([rec[j] ** 2 for j in range(Lv)])
    out = dict(g=g, hid=hid, refit=refit, seed=seed, exam=float(e.mean()), c2=[float(x) for x in c2], deep99=int((c2[1:] >= 0.99).sum()), n90=int((c2 >= 0.9).sum()), weights=L.size(), seconds=round(time.time() - t0, 1))
    os.makedirs(CACHE, exist_ok=True); json.dump(out, open(f, "w")); return out


def run():
    import multiprocessing as mp, budget
    for kk in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(kk, "1")
    seed0 = C("x128_seed0"); t0 = time.time(); tasks = [(g, h, r, s) for g in GS for h in HIDS for r in REFITS for s in range(seed0, seed0 + SEEDS)]
    # least compute: refit 150 and 600 gave identical results at hid 6 and 24, and hid 96 costs ~40 min a cell; keep hid 96 only at refit 150, g=1.2 (plus any hid-96 cell already cached)
    tasks = [t for t in tasks if t[1] < 96 or os.path.exists(os.path.join(CACHE, f"g{t[0]}_h{t[1]}_r{t[2]}_{t[3]}.json")) or (t[0] == 1.2 and t[2] == 150)]
    res = []
    for sel, gb, cap in ((lambda t: t[1] < 96, 3.0, 6), (lambda t: t[1] >= 96, 12.0, 1)):      # hid 96 holds a very large autograd graph (RAM fell to 4% with several at once): one at a time
        with mp.get_context("spawn").Pool(max(1, min(budget.workers(per_worker_gb=gb), cap))) as pool: res += pool.map(cell_run, [t for t in tasks if sel(t)], chunksize=1)
    for g in GS:
        for h in HIDS:
            for r in REFITS:
                x = [q for q in res if q["g"] == g and q["hid"] == h and q["refit"] == r]
                print(f"g {g} hid {h} refit {r}: exam {np.median([q['exam'] for q in x]):.3f} n90 {np.median([q['n90'] for q in x]):.0f} deep99 worlds {sum(q['deep99'] > 0 for q in x)}/{len(x)} median max corr2 {np.median([max(q['c2'][1:]) for q in x]):.3f} {np.median([q['seconds'] for q in x]):.0f}s", flush=True)
    json.dump(dict(per_cell=res, seconds=round(time.time() - t0, 1)), open(os.path.join(OUT, "x1_38_run.json"), "w"), indent=1, default=float)
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
