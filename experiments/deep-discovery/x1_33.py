"""X1.33: find the LIMIT (author 2026-11-03: a world with an extremely profound level of depth that can be graded; the perfect neural net should not
physically be able to discover all the rules; continuously scale the net then scale the world until we find the limit).
World ladder: the finite noise-free world with L concepts (L = 12, 24, 48, 96, 192, ...; depth grows 3 -> 12 as L grows; every concept's true values are known, so
every concept is graded) and T = x133_t_per * L training problems. Net ladder (three ways to scale the net, every route a family):
  CAP_base  a hard cap on total weights, doubling   150 -> 19200   (digger 'grow')
  CAP_pool  the same cap with pooled refinement (vi/refiner.py)
  HID_base  hidden units per concept node, 2 -> 48 (no cap)
For each (world L, family): climb the net ladder until a bigger net no longer raises the median exam score (any gain continues), then scale the world.
Grades (exam = x132_exam fresh problems from the whole world, library frozen): exam R^2; SHARE of the ceiling (true-concept readout, same 60 labels); concepts
recovered (corr^2 >= 0.99) by depth; deepest depth with any concept recovered; oracle_all = the best any net could do from 60 labels even holding every true
concept (the data limit). The world keeps scaling until the best exam share (any family) falls below x133_stop of its value at the smallest world, or a
world will not finish. Resumable: every (L, family, rung, seed) cell is cached. Usage: python x1_33.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_28

OUT = x1_28.OUT; CACHE = os.path.join(OUT, "x1_33_cells"); WORLDS = (12, 24, 48, 96, 192, 384)   # CONST: x133_cfg
LADDER = {"CAP_base": [150 * 2 ** i for i in range(8)], "CAP_pool": [150 * 2 ** i for i in range(8)], "HID_base": [2, 3, 6, 12, 24, 48]}


def cfg_of(fam, rung):
    base = dict(size="grow"); v = LADDER[fam][rung]
    return (dict(base, cap=v) if fam.startswith("CAP") else dict(base, hid=v)), (dict(pool="local") if fam == "CAP_pool" else {})


def cell(task):
    kind, L, fam, rung, seed = task; f = os.path.join(CACHE, f"{kind}_L{L}_{fam}_{rung}_{seed}.json")
    if os.path.exists(f): return json.load(open(f))
    import refiner, wstream
    t0 = time.time(); T = C("x133_t_per") * L; NE = C("x132_exam")
    W = wstream.Stream(seed, dict(noise=0.0, n_tr=C("x128_pool"), lib_size=L, lib=kind)); data = [W.next() for _ in range(T)]; exam = [W.next() for _ in range(NE)]
    rng = np.random.default_rng([seed, C("x126_stream")]); xp = rng.standard_normal((C("x126_probe"), W.cfg["dim"])); V = W.lib.values(xp); dep = np.array([c["depth"] for c in W.lib.c])
    c1, c2 = cfg_of(fam, rung); L_ = refiner.Learner(seed, dict(c1, **c2)); r2 = []
    for t, d in enumerate(data): r2.append(L_.step(d)["r2"]); x1_28.hb(seed, f"{kind}_L{L}_{fam}_{LADDER[fam][rung]}", t + 1, T, r2)
    ex = L_.exam(exam); rec = L_.recovered(xp, V, list(range(L))); ok = np.array([rec[j] ** 2 >= C("x128_recovered") for j in range(L)])
    # ceilings: readout on the TRUE concepts of the problem (S) and on ALL true concepts, same 60 labels, ridge as the learner's readout
    rid = 3.0; cS, cA = [], []
    for d in exam:
        x, y = d["te"]; xl, yl = d["tr"][0][:60], d["tr"][1][:60]; Vl, Vt = W.lib.values(xl), W.lib.values(x)
        for cols, acc in ((d["S"], cS), (list(range(L)), cA)):
            Fl = np.hstack([np.ones((60, 1)), Vl[:, cols]]); Ft = np.hstack([np.ones((len(y), 1)), Vt[:, cols]])
            b = np.linalg.solve(Fl.T @ Fl + rid * np.eye(Fl.shape[1]), Fl.T @ yl); acc.append(1 - float(np.mean((y - Ft @ b) ** 2) / np.var(y)))
    deep_ok = [int(dep[j]) for j in range(L) if ok[j]]
    out = dict(kind=kind, L=L, fam=fam, rung=rung, size_param=LADDER[fam][rung], seed=seed, exam=float(np.mean([e[0] for e in ex])), ceil_S=float(np.mean(cS)), oracle_all=float(np.mean(cA)),
               rec_all=float(ok.mean()), rec_d0=float(ok[dep == 0].mean()), rec_d1=float(ok[dep == 1].mean()) if (dep == 1).any() else None, rec_d2p=float(ok[dep >= 2].mean()) if (dep >= 2).any() else None,
               reach=max(deep_ok) if deep_ok else -1, max_depth=int(dep.max()), train_last=float(np.mean(r2[-30:])), size=L_.size(), nodes=len(L_.units), exam_links=float(np.mean([e[1] for e in ex])),
               seconds=round(time.time() - t0, 1))
    os.makedirs(CACHE, exist_ok=True); json.dump(out, open(f, "w")); return out


def run(kind="mix"):
    import multiprocessing as mp, budget
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x128_seed0"); seeds = list(range(seed0, seed0 + C("x133_seeds"))); t0 = time.time(); table, first = {}, None
    med = lambda rs, k: float(np.median([r[k] for r in rs]))
    with mp.get_context("spawn").Pool(budget.workers()) as pool:
        for L in WORLDS:
            rung, active, res = {f: 0 for f in LADDER}, set(LADDER), {f: [] for f in LADDER}
            while active:     # one round = the next net size for every family still climbing, all seeds at once
                tasks = [(kind, L, f, rung[f], s) for f in sorted(active) for s in seeds]; out = pool.map(cell, tasks, chunksize=1)
                for f in sorted(active):
                    rs = [r for r in out if r["fam"] == f]; res[f].append(rs)
                    print(f"[{kind}] L={L:3d} {f:9s} size {LADDER[f][rung[f]]:6d}: exam {med(rs, 'exam'):.3f} share {med(rs, 'exam') / max(med(rs, 'ceil_S'), 1e-9):.2f} rec {med(rs, 'rec_all'):.2f} reach {med(rs, 'reach'):.0f}/{rs[0]['max_depth']} weights {med(rs, 'size'):.0f}  ({time.time() - t0:.0f} s)", flush=True)
                    gain = len(res[f]) == 1 or med(rs, "exam") > med(res[f][-2], "exam")     # any gain continues
                    if gain and rung[f] + 1 < len(LADDER[f]): rung[f] += 1
                    else: active.discard(f)
            best = max((med(rs, "exam") / max(med(rs, "ceil_S"), 1e-9), f, i) for f in LADDER for i, rs in enumerate(res[f]))
            top = max(range(len(res[best[1]])), key=lambda i: med(res[best[1]][i], "exam"))
            table[L] = dict(best_share=best[0], best_family=best[1], best_rung=best[2], oracle_all=med(res[best[1]][best[2]], "oracle_all"), reach=med(res[best[1]][best[2]], "reach"),
                            max_depth=res[best[1]][0][0]["max_depth"], limit_net={f: LADDER[f][max(range(len(res[f])), key=lambda i: med(res[f][i], "exam"))] for f in LADDER},
                            curves={f: [(LADDER[f][i], med(rs, "exam"), med(rs, "rec_all")) for i, rs in enumerate(res[f])] for f in LADDER})
            first = first or best[0]
            print(f"== [{kind}] world L={L}: best share {best[0]:.2f} ({best[1]}, rung {best[2]}), data limit (oracle on all concepts) exam {table[L]['oracle_all']:.2f}, deepest recovered {table[L]['reach']:.0f} of {table[L]['max_depth']}", flush=True)
            json.dump(table, open(os.path.join(OUT, f"x1_33{'' if kind == 'mix' else '_' + kind}_run.json"), "w"), indent=1, default=float)
            if best[0] < C("x133_stop") * first: print(f"LIMIT FOUND at L={L}: best share fell below {C('x133_stop')} of the smallest world"); break
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run(sys.argv[2] if len(sys.argv) > 2 else "mix")
