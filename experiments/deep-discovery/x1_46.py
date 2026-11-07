"""X1.46: CHOOSE the activation family by measured help instead of naming it (author goal: discover from nothing; X1.39: a sine inside the cell worked at g=1.2, but sin names the rule's family; X1.40/X1.41 mixing and L1 did not recover it).
Pilot selection: three copies of the cell (inner activation tanh | sin | relu) each learn the first PILOT=10 problems of room 1; the copy with the best mean running R^2 over those problems (the same R^2 the run prints while learning; it is the test split of the problem being learned, NOT the exam) continues alone through the remaining 40 problems and is graded on the exam. The three candidate families are fixed in advance; nothing says which fits the world.
Setup otherwise as X1.39 (g=1.2, hid 24, N=600, refit 150, room 1, default settings), 6 worlds (960000-960005); worlds 960000-960002 have cached tanh and sin cells to compare with.
Usage: python x1_46.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_28, x1_35

OUT = x1_28.OUT; CACHE = os.path.join(OUT, "x1_46_cells"); N = 600; HID = 24; REFIT = 150; G = 1.2; ACTS = ("tanh", "sin", "relu"); PILOT = 10; SEEDS = 6   # CONST: x146_cfg


def cell_run(seed):
    f = os.path.join(CACHE, f"{seed}.json")
    if os.path.exists(f): return json.load(open(f))
    import cell, wstream, wrec
    wrec.GAIN = G; t0 = time.time(); NE = C("x132_exam"); Lv = 5; k = 1
    W = wstream.Stream(seed, dict(noise=0.0, n_tr=N, n_va=N // 2, lib_size=Lv, lib="rec")); data = [wstream.build(W.lib, sp, seed, W.cfg) for sp in x1_35.room_specs(k, k)]
    Ls = {a: cell.Learner(seed, dict(refit=REFIT, n_lab=N, hid=HID, act=a)) for a in ACTS}; pil = {a: [] for a in ACTS}
    for i in range(PILOT):
        for a in ACTS: Ls[a].activate(); pil[a].append(Ls[a].step(data[i])["r2"]); x1_28.hb(seed, f"pilot_{a}", i + 1, PILOT, pil[a])
    score = {a: float(np.mean(pil[a])) for a in ACTS}; best = max(score, key=score.get); L = Ls[best]; L.activate(); r2 = list(pil[best])
    for i in range(PILOT, len(data)): r2.append(L.step(data[i])["r2"]); x1_28.hb(seed, f"chosen_{best}", i + 1, 50, r2)
    rng = np.random.default_rng([seed, 7, k]); lvls = rng.integers(0, Lv, NE)
    exam = [wstream.build(W.lib, dict(t=90000 + k * 100 + j, S=[int(lv)], a=[1.0], stage=0, has_new=False), seed, W.cfg) for j, lv in enumerate(lvls)]
    e = np.array([x[0] for x in L.exam(exam)]); xp = rng.standard_normal((C("x126_probe"), W.cfg["dim"])); V = W.lib.values(xp); rec = L.recovered(xp, V, list(range(Lv))); c2 = np.array([rec[j] ** 2 for j in range(Lv)])
    out = dict(seed=seed, chosen=best, pilot_scores=score, exam=float(e.mean()), c2=[float(x) for x in c2], deep99=int((c2[1:] >= 0.99).sum()), n90=int((c2 >= 0.9).sum()), seconds=round(time.time() - t0, 1))
    os.makedirs(CACHE, exist_ok=True); json.dump(out, open(f, "w")); return out


def run():
    import multiprocessing as mp, budget
    for kk in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(kk, "1")
    seed0 = C("x128_seed0"); t0 = time.time()
    with mp.get_context("spawn").Pool(max(1, min(budget.workers(per_worker_gb=4.0), 3))) as pool: res = pool.map(cell_run, list(range(seed0, seed0 + SEEDS)), chunksize=1)
    for r in res: print(f"world {r['seed']}: chose {r['chosen']} (pilot {', '.join(f'{a} {v:.2f}' for a, v in r['pilot_scores'].items())}) exam {r['exam']:.3f} corr2 {[round(x, 2) for x in r['c2']]} deep99 {r['deep99']}", flush=True)
    print(f"chose sin in {sum(r['chosen'] == 'sin' for r in res)}/{len(res)} worlds; median exam {np.median([r['exam'] for r in res]):.3f}; worlds with an exact deep concept {sum(r['deep99'] > 0 for r in res)}/{len(res)}", flush=True)
    json.dump(dict(per_cell=res, seconds=round(time.time() - t0, 1)), open(os.path.join(OUT, "x1_46_run.json"), "w"), indent=1, default=float)
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
