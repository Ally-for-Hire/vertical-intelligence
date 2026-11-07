"""X1.34: ONE simple rule, from nothing, in a world that is extremely simple in nature and extremely deep (author 2026-11-03; replaces X1.31).
World (vi/wrec.py): ONE rule, concept n = sin(g (concept n-1 + x[n mod 8])); each problem is ONE level (noise-free, 60 random labels); levels appear equally often;
the stream is finite while learning and every level is graded exactly. The steepness g is a world axis (every setting an arm): g = 0.8, 1.2, 1.6, 2.0.
The learner starts with NOTHING and has ONE rule, no stages, no schedules: a new function is a small net of the environment's variables and at most one known
function; kept if it measurably helps on both halves of the decision points ('grow'); inputs are random labels. Routes for how the known function is chosen
(every route an arm): NONE (never use one: the flat baseline) | CORR (the known function most correlated with what is left unexplained) | ALL (try every known
function, keep the one that helps) | CORR_POOL (CORR plus pooled refinement of everything a problem touched, vi/refiner.py).
Grades, after T = x134_t_per * L problems with the library then FROZEN: exam on x132_exam fresh problems drawn from the whole world (R^2 by level), the DEEPEST level
recovered (a function with corr^2 >= 0.99 to the true concept), and the same reach measured at T/3 and 2T/3 (does depth keep growing with experience, or is there
a ceiling?). Resumable: every (g, arm, world) cell is cached. Usage: python x1_34.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_28

OUT = x1_28.OUT; CACHE = os.path.join(OUT, "x1_34_cells"); GAINS = (0.8, 1.2, 1.6, 2.0)   # CONST: x134_cfg
B = dict(size="grow", flat_off=True, n_par=1, hid=6)
ARMS = {"NONE": dict(B, parent_pick="none"), "CORR": dict(B, parent_pick="corr"), "ALL": dict(B, parent_pick="all"), "CORR_POOL": dict(B, parent_pick="corr", pool="local")}   # CONST: x134_cfg


def cell(task):
    g, arm, seed = task; f = os.path.join(CACHE, f"g{g}_{arm}_{seed}.json")
    if os.path.exists(f): return json.load(open(f))
    import refiner, wstream, wrec
    wrec.GAIN = g; t0 = time.time(); Lv = C("x134_levels"); T = C("x134_t_per") * Lv; NE = C("x132_exam")
    W = wstream.Stream(seed, dict(noise=0.0, n_tr=C("x128_pool"), lib_size=Lv, lib="rec", mix_extra=0.0)); data = [W.next() for _ in range(T)]; exam = [W.next() for _ in range(NE)]
    rng = np.random.default_rng([seed, C("x126_stream")]); xp = rng.standard_normal((C("x126_probe"), W.cfg["dim"])); V = W.lib.values(xp)
    L = refiner.Learner(seed, ARMS[arm]); r2, reach = [], {}
    for t, d in enumerate(data):
        r2.append(L.step(d)["r2"]); x1_28.hb(seed, f"g{g}_{arm}", t + 1, T, r2)
        if t + 1 in (T // 3, 2 * T // 3, T):
            rec = L.recovered(xp, V, list(range(Lv))); ok = [j for j in range(Lv) if rec[j] ** 2 >= C("x128_recovered")]; reach[t + 1] = (max(ok) if ok else -1, len(ok))
    ex = L.exam(exam); lv = np.array([d["S"][0] for d in exam]); e = np.array([x[0] for x in ex]); m = lambda lo, hi: float(e[(lv >= lo) & (lv <= hi)].mean()) if ((lv >= lo) & (lv <= hi)).any() else None
    out = dict(g=g, arm=arm, seed=seed, reach_third=reach[T // 3], reach_two_thirds=reach[2 * T // 3], reach=reach[T], exam=float(e.mean()), exam_l0=m(0, 0), exam_l1=m(1, 1), exam_l2_3=m(2, 3), exam_l4p=m(4, 99),
               solved=float(np.mean(e >= C("x128_recovered"))), units=len(L.units), size=L.size(), seconds=round(time.time() - t0, 1))
    os.makedirs(CACHE, exist_ok=True); json.dump(out, open(f, "w")); return out


def run():
    import multiprocessing as mp, budget
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x128_seed0"); seeds = list(range(seed0, seed0 + C("x134_worlds"))); t0 = time.time()
    tasks = [(g, a, s) for g in GAINS for a in ("ALL", "CORR_POOL", "CORR", "NONE") for s in seeds]      # slowest first
    with mp.get_context("spawn").Pool(budget.workers()) as pool: res = pool.map(cell, tasks, chunksize=1)
    med = lambda rs, k: float(np.median([r[k] for r in rs if r[k] is not None])) if any(r[k] is not None for r in rs) else float("nan")
    table = {}
    print(f"{'g':>4s} {'arm':10s} {'reach T/3':>9s} {'2T/3':>5s} {'T':>4s} {'exam':>6s} {'l0':>5s} {'l1':>5s} {'l2-3':>5s} {'l4+':>5s} {'solved':>6s} {'units':>6s} {'sec':>5s}")
    for g in GAINS:
        for a in ARMS:
            rs = [r for r in res if r["g"] == g and r["arm"] == a]
            row = dict(reach_third=float(np.median([r["reach_third"][0] for r in rs])), reach_two_thirds=float(np.median([r["reach_two_thirds"][0] for r in rs])), reach=float(np.median([r["reach"][0] for r in rs])),
                       **{k: med(rs, k) for k in ("exam", "exam_l0", "exam_l1", "exam_l2_3", "exam_l4p", "solved", "units", "seconds")}); table[f"{g}_{a}"] = row
            print(f"{g:4.1f} {a:10s} {row['reach_third']:9.0f} {row['reach_two_thirds']:5.0f} {row['reach']:4.0f} {row['exam']:6.3f} {row['exam_l0']:5.2f} {row['exam_l1']:5.2f} {row['exam_l2_3']:5.2f} {row['exam_l4p']:5.2f} {row['solved']:6.2f} {row['units']:6.0f} {row['seconds']:5.0f}")
    json.dump(dict(table=table, per_cell=res, seconds=round(time.time() - t0, 1)), open(os.path.join(OUT, "x1_34_run.json"), "w"), indent=1, default=float)
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
