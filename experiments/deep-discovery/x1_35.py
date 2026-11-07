"""X1.35: the PUZZLE ROOM (author 2026-11-03: '50 problems grouped by vertical complexity and connection to each other. think of it as a puzzle room. one that we make
more and more complex and are trying to get a network to get good at solving, simplifying everything'). Replaces X1.34.
World (vi/wrec.py, the one-rule deep world: concept n = sin(g (concept n-1 + x[n mod 8]))). A ROOM = 50 problems, noise-free, 60 random labels each, in 5 TIERS of
10 problems (tier t = vertical complexity: the levels t*s .. (t+1)*s-1, s = the room's step). Inside a tier the problems are CONNECTED: they come in dependency order
(level n, then n+1, ...; each is built from the one before it). Room k has step s = k: room 1 = levels 0-4, room 2 = levels 0-9, room 3 = levels 0-14, ... : every room is
deeper than the last and gives each level fewer problems (50 / levels). The SAME learner walks through the rooms in order, keeping everything it has built.
Learner: ONE rule from nothing (no stages): a new function is a small net of the environment's variables and at most one known function, kept if it measurably helps on
both halves of the decision points. Routes for choosing the known function (every route an arm): NONE | CORR | ALL (try every known function) | CORR_POOL | CORR_CLONE / ALL_CLONE (patterns in patterns: reuse a learned computation with a new parent and a new input wiring, no training; needs one parented function to start from, so it rides on CORR or ALL). The many-candidate routes (ALL, the clone routes) use an evidence bar that grows with the number of candidates tried (help > log2(K+1) bits): without it the clone route accepted a lucky candidate nearly every step (297 functions vs 44, 28x slower, no better score). Smoke test: clone with NONE is identical to NONE by construction (no parented template ever exists), so that arm is dropped.
Grades after each room (library frozen): exam on 40 fresh problems drawn over ALL the room's levels (R^2 by tier); the deepest level recovered; and SIMPLIFYING: functions
linked per exam problem, and the weights and functions the library holds per level covered. Plus 'getting good': R^2 of the last 10 training problems of the room vs the first 10.
Axis: world steepness g (every setting an arm). Resumable: every (g, arm, world) cell is cached. Usage: python x1_35.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_28

OUT = x1_28.OUT; CACHE = os.path.join(OUT, "x1_35_cells"); GAINS = (0.8, 1.2, 2.0)   # CONST: x135_cfg
B = dict(size="grow", flat_off=True, n_par=1, hid=6)
ARMS = {"NONE": dict(B, parent_pick="none"), "CORR": dict(B, parent_pick="corr"), "ALL": dict(B, parent_pick="all", bar=True), "CORR_POOL": dict(B, parent_pick="corr", pool="local"),
        "CORR_CLONE": dict(B, parent_pick="corr", clone=True, bar=True), "ALL_CLONE": dict(B, parent_pick="all", clone=True, bar=True)}   # CONST: x135_cfg


def room_specs(k, room_idx):
    """The 50 problems of room k: 5 tiers x 10 problems; tier t covers levels [t*k, (t+1)*k); inside a tier the problems follow the dependency order."""
    specs = []
    for t in range(5):
        lv = list(range(t * k, (t + 1) * k))
        for i in range(10): specs.append(dict(t=room_idx * 1000 + t * 10 + i, S=[lv[i % k]], a=[1.0], stage=0, has_new=False, tier=t))
    return specs


def cell(task):
    g, arm, seed = task; f = os.path.join(CACHE, f"g{g}_{arm}_{seed}.json")
    if os.path.exists(f): return json.load(open(f))
    import refiner, wstream, wrec
    wrec.GAIN = g; t0 = time.time(); NE = C("x132_exam"); rooms = C("x135_rooms"); L = refiner.Learner(seed, ARMS[arm]); out = dict(g=g, arm=arm, seed=seed, rooms=[]); done = 0
    for k in range(1, rooms + 1):
        Lv = 5 * k; W = wstream.Stream(seed, dict(noise=0.0, n_tr=C("x128_pool"), lib_size=Lv, lib="rec"))
        data = [wstream.build(W.lib, sp, seed, W.cfg) for sp in room_specs(k, k)]; r2 = []
        for i, d in enumerate(data): r2.append(L.step(d)["r2"]); done += 1; x1_28.hb(seed, f"g{g}_{arm}_room{k}", i + 1, 50, r2)
        rng = np.random.default_rng([seed, 7, k]); lvls = rng.integers(0, Lv, NE)
        exam = [wstream.build(W.lib, dict(t=90000 + k * 100 + j, S=[int(lv)], a=[1.0], stage=0, has_new=False), seed, W.cfg) for j, lv in enumerate(lvls)]
        ex = L.exam(exam); e = np.array([x[0] for x in ex]); tier = np.minimum(lvls // k, 4)
        xp = rng.standard_normal((C("x126_probe"), W.cfg["dim"])); V = W.lib.values(xp); rec = L.recovered(xp, V, list(range(Lv))); ok = [j for j in range(Lv) if rec[j] ** 2 >= C("x128_recovered")]
        out["rooms"].append(dict(room=k, levels=Lv, exam=float(e.mean()), tiers=[float(e[tier == t].mean()) if (tier == t).any() else None for t in range(5)], solved=float(np.mean(e >= C("x128_recovered"))),
                                 reach=(max(ok) if ok else -1), n_rec=len(ok), links=float(np.mean([x[1] for x in ex])), units=len(L.units), weights=L.size(),
                                 weights_per_level=L.size() / Lv, first10=float(np.mean(r2[:10])), last10=float(np.mean(r2[-10:])), seconds=round(time.time() - t0, 1)))
    os.makedirs(CACHE, exist_ok=True); json.dump(out, open(f, "w")); return out


def run():
    import multiprocessing as mp, budget
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x128_seed0"); seeds = list(range(seed0, seed0 + C("x135_worlds"))); t0 = time.time()
    cheap = [(g, a, s) for g in GAINS for a in ("NONE", "CORR", "CORR_CLONE", "CORR_POOL") for s in seeds]
    heavy = [(g, a, s) for g in GAINS for a in ("ALL", "ALL_CLONE") for s in seeds]      # the try-every-known-function routes hold ~1.4 GB per worker and grow
    res = []
    for tasks, gb in ((cheap, 1.0), (heavy, 2.0)):
        with mp.get_context("spawn").Pool(budget.workers(per_worker_gb=gb)) as pool: res += pool.map(cell, tasks, chunksize=1)
    med = lambda xs: float(np.median([x for x in xs if x is not None])) if any(x is not None for x in xs) else float("nan")
    print(f"{'g':>4s} {'arm':10s} {'room':>4s} {'lvls':>4s} {'exam':>6s} {'t0':>5s} {'t1':>5s} {'t2':>5s} {'t3':>5s} {'t4':>5s} {'solved':>6s} {'reach':>5s} {'links':>5s} {'units':>5s} {'w/lvl':>6s} {'1st10':>6s} {'last10':>6s}")
    for g in GAINS:
        for a in ARMS:
            rs = [r for r in res if r["g"] == g and r["arm"] == a]
            for k in range(C("x135_rooms")):
                R = [r["rooms"][k] for r in rs]; T = [med([x["tiers"][t] for x in R]) for t in range(5)]
                print(f"{g:4.1f} {a:10s} {k + 1:4d} {R[0]['levels']:4d} {med([x['exam'] for x in R]):6.3f} " + " ".join(f"{v:5.2f}" for v in T) +
                      f" {med([x['solved'] for x in R]):6.2f} {med([x['reach'] for x in R]):5.0f} {med([x['links'] for x in R]):5.2f} {med([x['units'] for x in R]):5.0f} {med([x['weights_per_level'] for x in R]):6.0f} {med([x['first10'] for x in R]):6.2f} {med([x['last10'] for x in R]):6.2f}")
    json.dump(dict(per_cell=res, seconds=round(time.time() - t0, 1)), open(os.path.join(OUT, "x1_35_run.json"), "w"), indent=1, default=float)
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
