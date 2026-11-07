"""X1.42: does the exact recovery SCALE WITH DEPTH? (X1.39: a sine cell at g=1.2, hid 24 got exact deep concepts in room 1 = levels 0-4.) Same learner and worlds as X1.36 rooms 1-4 (5/10/15/20 levels, 50 problems each,
library kept across rooms) with the X1.39 cell: hid 24, N=600 labels, refit 150, g=1.2; arms: inner activation SIN | TANH (the comparison), 3 worlds. Grades per room: exam, deepest level at corr^2>=0.99, count of levels at corr^2>=0.9 / 0.99.
Memory: the joint refit uses at most ROWS=300 train / 150 validation rows per stored problem (room 3 with all 600/300 rows drove RAM free to 1%); both arms use the same cap. Honest framing: sin names the rule's family (see X1.39/X1.40). Usage: python x1_42.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_28, x1_35

OUT = x1_28.OUT; CACHE = os.path.join(OUT, "x1_42_cells"); N = 600; HID = 24; REFIT = 150; G = 1.2; ACTS = ("sin", "tanh"); ROWS = 300; SEEDS = 3   # CONST: x142_cfg


def cell_run(task):
    act, seed = task; f = os.path.join(CACHE, f"{act}_{seed}.json")
    if os.path.exists(f): return json.load(open(f))
    import cell, wstream, wrec
    wrec.GAIN = G; t0 = time.time(); NE = C("x132_exam"); L = cell.Learner(seed, dict(refit=REFIT, n_lab=N, hid=HID, act=act, refit_rows=ROWS)); out = dict(act=act, seed=seed, rooms=[])
    for k in range(1, C("x135_rooms") + 1):
        Lv = 5 * k; W = wstream.Stream(seed, dict(noise=0.0, n_tr=N, n_va=N // 2, lib_size=Lv, lib="rec")); data = [wstream.build(W.lib, sp, seed, W.cfg) for sp in x1_35.room_specs(k, k)]; r2 = []
        for i, d in enumerate(data): r2.append(L.step(d)["r2"]); x1_28.hb(seed, f"{act}_room{k}", i + 1, 50, r2)
        rng = np.random.default_rng([seed, 7, k]); lvls = rng.integers(0, Lv, NE)
        exam = [wstream.build(W.lib, dict(t=90000 + k * 100 + j, S=[int(lv)], a=[1.0], stage=0, has_new=False), seed, W.cfg) for j, lv in enumerate(lvls)]
        e = np.array([x[0] for x in L.exam(exam)]); xp = rng.standard_normal((C("x126_probe"), W.cfg["dim"])); V = W.lib.values(xp); rec = L.recovered(xp, V, list(range(Lv))); c2 = np.array([rec[j] ** 2 for j in range(Lv)])
        ok = [j for j in range(Lv) if c2[j] >= 0.99]
        out["rooms"].append(dict(room=k, levels=Lv, exam=float(e.mean()), c2=[float(x) for x in c2], reach=(max(ok) if ok else -1), n99=len(ok), n90=int((c2 >= 0.9).sum()), nodes=len(L.nodes), weights=L.size(), seconds=round(time.time() - t0, 1)))
        os.makedirs(CACHE, exist_ok=True); json.dump(out, open(f + ".part", "w"))          # partial progress is visible while the cell runs
    json.dump(out, open(f, "w")); return out


def run():
    import multiprocessing as mp, budget
    for kk in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(kk, "1")
    seed0 = C("x128_seed0"); t0 = time.time(); tasks = [(a, s) for a in ACTS for s in range(seed0, seed0 + SEEDS)]
    with mp.get_context("spawn").Pool(max(1, min(budget.workers(per_worker_gb=8.0), 2))) as pool: res = pool.map(cell_run, tasks, chunksize=1)
    for a in ACTS:
        for k in range(C("x135_rooms")):
            R = [r["rooms"][k] for r in res if r["act"] == a]
            print(f"{a} room {k + 1} ({R[0]['levels']} levels): exam {np.median([x['exam'] for x in R]):.3f} reach {np.median([x['reach'] for x in R]):.0f} n99 {np.median([x['n99'] for x in R]):.0f} n90 {np.median([x['n90'] for x in R]):.0f} worlds with n99>0: {sum(x['n99'] > 0 for x in R)}/{len(R)} weights {np.median([x['weights'] for x in R]):.0f} {np.median([x['seconds'] for x in R]):.0f}s", flush=True)
    json.dump(dict(per_cell=res, seconds=round(time.time() - t0, 1)), open(os.path.join(OUT, "x1_42_run.json"), "w"), indent=1, default=float)
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
