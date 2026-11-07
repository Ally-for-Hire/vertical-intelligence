"""X1.32: no arbitrary stages; formulas of ever-increasing accuracy that make things simple; tested on the WHOLE environment at the end
(author 2026-11-03). One continuous rule (vi/refiner.py): link by confirmed help, then re-fit every touched function (and what it is built from) jointly on ALL
stored problems that use it, accepted only if pooled decision-point error drops, repeated until nothing improves. Environment: the finite noise-free world
(24 concepts, depth up to 5) - deeper than the learner reaches. After T training problems the library is FROZEN and examined on x132_exam fresh problems drawn
from the whole world (readout only), and on all 24 concepts (best single function, corr^2 >= 0.99 = recovered), by depth.
Arms (author rule: every route): BASE (digger 'grow', no pooling) | POOL_local (touched functions + ancestors) | POOL_all (every function equally malleable)
| POOL_local+RULE (pooled + one distilled rule per solved problem). Usage: python x1_32.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_28

OUT = x1_28.OUT
G = dict(size="grow")
ARMS = {"BASE": G, "POOL_local": dict(G, pool="local"), "POOL_all": dict(G, pool="all"), "POOL_local+RULE": dict(G, pool="local", rule="best")}   # CONST: x132_cfg


def world(seed):
    import refiner, wstream
    t0 = time.time(); T = C("x128_problems"); bs = C("x122_block"); NE = C("x132_exam")
    W = wstream.Stream(seed, dict(noise=0.0, n_tr=C("x128_pool"))); data = [W.next() for _ in range(T)]; exam = [W.next() for _ in range(NE)]
    rng = np.random.default_rng([seed, C("x126_stream")]); xp = rng.standard_normal((C("x126_probe"), W.cfg["dim"])); V = W.lib.values(xp)
    dep = np.array([c["depth"] for c in W.lib.c]); out = {}
    for a, cfg in ARMS.items():
        ta = time.time(); L = refiner.Learner(seed, cfg); r2 = []
        for t, d in enumerate(data): r2.append(L.step(d)["r2"]); x1_28.hb(seed, a, t + 1, T, r2)
        ex = L.exam(exam); rec = L.recovered(xp, V, list(range(len(W.lib.c)))); ok = np.array([rec[j] ** 2 >= C("x128_recovered") for j in range(len(dep))])
        out[a] = dict(train_last=float(np.mean(r2[-bs:])), train_first=float(np.mean(r2[:bs])), links_first=float(np.mean([len(h) for h in L.heads[:bs]])), links_last=float(np.mean([len(h) for h in L.heads[-bs:]])),
                      exam_r2=float(np.mean([e[0] for e in ex])), exam_links=float(np.mean([e[1] for e in ex])), exam_solved=float(np.mean([e[0] >= C("x128_recovered") for e in ex])),
                      rec_all=float(ok.mean()), rec_d0=float(ok[dep == 0].mean()), rec_d1=float(ok[dep == 1].mean()), rec_d2=float(ok[dep >= 2].mean()), size=L.size(), nodes=len(L.units),
                      seconds=round(time.time() - ta, 1))
    return out


def run():
    import multiprocessing as mp, budget
    from x0_1 import boot_ci
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x128_seed0"); t0 = time.time(); seeds = list(range(seed0, seed0 + C("x132_worlds")))
    with mp.get_context("spawn").Pool(budget.workers()) as pool: res = pool.map(world, seeds, chunksize=1)
    per = dict(zip(seeds, res)); g = lambda a, k: np.array([per[s][a][k] for s in seeds], float)
    keys = ("train_last", "links_first", "links_last", "exam_r2", "exam_links", "exam_solved", "rec_all", "rec_d0", "rec_d1", "rec_d2", "size", "nodes", "seconds")
    med = {a: {k: float(np.nanmedian(g(a, k))) for k in keys} for a in ARMS}
    S = dict(worlds=len(seeds), med=med, exam_vs_base={a: boot_ci(g(a, "exam_r2") - g("BASE", "exam_r2"), seed0) for a in ARMS if a != "BASE"},
             rec_vs_base={a: boot_ci(g(a, "rec_all") - g("BASE", "rec_all"), seed0) for a in ARMS if a != "BASE"},
             simpler={a: boot_ci((g(a, "links_first") - g(a, "links_last")), seed0) for a in ARMS})
    json.dump({"seconds": round(time.time() - t0, 1), "result": {"per_world": per, "summary": S}}, open(os.path.join(OUT, "x1_32_run.json"), "w"), indent=1, default=float)
    print(f"{'arm':16s} {'trainR2':>7s} {'examR2':>7s} {'solved':>6s} {'xLinks':>6s} {'rec all':>7s} {'d0':>5s} {'d1':>5s} {'d2+':>5s} {'links 1st->last':>16s} {'size':>7s}  exam dR2 vs BASE")
    for a in ARMS:
        m = med[a]; d = S["exam_vs_base"].get(a)
        print(f"{a:16s} {m['train_last']:7.3f} {m['exam_r2']:7.3f} {m['exam_solved']:6.2f} {m['exam_links']:6.2f} {m['rec_all']:7.2f} {m['rec_d0']:5.2f} {m['rec_d1']:5.2f} {m['rec_d2']:5.2f} {m['links_first']:7.2f}->{m['links_last']:5.2f} {m['size']:9.0f}" + (f"  {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else ""))
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
