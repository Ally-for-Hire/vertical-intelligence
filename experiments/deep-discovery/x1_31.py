"""X1.31: solve the problem, THEN make a repeatable rule (author 2026-11-03: 'it both solves it AND then makes a repeatable rule/function to solve it,
with emphasis on the 2nd'). After each solved problem the learner distils its solution into ONE reusable function (fitted to its own predictions on all pool
inputs, no extra labels) that later problems can link in a single step. Every way of making the rule is an arm (author rule):
 NONE (no rule; reference) | RULE_flat | RULE_sub (function of inputs + known functions) | RULE_best (better of the two) | RULE_sub_f90 (kept only if fidelity >= 0.9)
 | RULE_sub_min1 (also from one-part solutions) | PRUNE_02 / PRUNE_10 (shrink the solution first, tolerance 2% / 10%) | PRUNE_10+RULE_best.
Emphasis on the rule = the main measures are about reuse: share of later problems/links that use a rule, links per problem, last-block R^2, plus fidelity and size.
Noise-free worlds, all-random labels, size reading 'grow' (best in X1.28), default compute. Usage: python x1_31.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_28

OUT = x1_28.OUT
G = dict(size="grow")
ARMS = {"NONE": G, "RULE_flat": dict(G, rule="flat"), "RULE_sub": dict(G, rule="sub"), "RULE_best": dict(G, rule="best"), "RULE_sub_f90": dict(G, rule="sub", rule_fid=0.9),
        "RULE_sub_min1": dict(G, rule="sub", rule_min=1), "PRUNE_02": dict(G, rule="best", prune_tol=0.02, rule_min=99), "PRUNE_10": dict(G, rule="best", prune_tol=0.10, rule_min=99),
        "PRUNE_10+RULE": dict(G, rule="best", prune_tol=0.10)}   # CONST: x131_cfg


def world(seed):
    import digger, wstream
    t0 = time.time(); T = C("x128_problems"); bs = C("x122_block")
    W = wstream.Stream(seed, dict(noise=0.0, n_tr=C("x128_pool"))); data = [W.next() for _ in range(T)]; out = {}
    for a, cfg in ARMS.items():
        ta = time.time(); L = digger.Learner(seed, cfg); r2 = []
        for t, d in enumerate(data): r2.append(L.step(d)["r2"]); x1_28.hb(seed, a, t + 1, T, r2)
        R = getattr(L, "rules", {}); tail = range(T - bs, T); uses = {j: sum(j in L.heads[u] for u in range(t + 1, T)) for j, (t, f) in R.items()}
        nl = sum(len(L.heads[u]) for u in tail)
        out[a] = dict(last=float(np.mean(r2[-bs:])), links=float(np.mean([len(L.heads[u]) for u in tail])), size=L.size(), nodes=len(L.units), rules=len(R),
                      fid=float(np.mean([f for t, f in R.values()])) if R else float("nan"), reused=float(np.mean([u > 0 for u in uses.values()])) if R else float("nan"),
                      uses=float(np.mean(list(uses.values()))) if R else float("nan"), rule_share=float(sum(j in L.heads[u] for u in tail for j in R) / max(nl, 1)),
                      seconds=round(time.time() - ta, 1))
    return out


def run():
    import multiprocessing as mp, budget
    from x0_1 import boot_ci
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x128_seed0"); t0 = time.time(); seeds = list(range(seed0, seed0 + C("x131_worlds")))
    with mp.get_context("spawn").Pool(budget.workers()) as pool: res = pool.map(world, seeds, chunksize=1)
    per = dict(zip(seeds, res)); g = lambda a, k: np.array([per[s][a][k] for s in seeds], float)
    keys = ("last", "links", "size", "nodes", "rules", "fid", "reused", "uses", "rule_share", "seconds")
    med = {a: {k: float(np.nanmedian(g(a, k))) if not np.all(np.isnan(g(a, k))) else float("nan") for k in keys} for a in ARMS}
    S = dict(worlds=len(seeds), med=med, vs_none={a: boot_ci(g(a, "last") - g("NONE", "last"), seed0) for a in ARMS if a != "NONE"},
             links_vs_none={a: boot_ci(g(a, "links") - g("NONE", "links"), seed0) for a in ARMS if a != "NONE"})
    json.dump({"seconds": round(time.time() - t0, 1), "result": {"per_world": per, "summary": S}}, open(os.path.join(OUT, "x1_31_run.json"), "w"), indent=1, default=float)
    print(f"{'arm':14s} {'R2':>6s} {'links':>6s} {'rules':>6s} {'fid':>5s} {'reused':>6s} {'uses':>5s} {'rule%links':>10s} {'size':>7s}  dR2 vs NONE | dLinks")
    for a in ARMS:
        m = med[a]; d = S["vs_none"].get(a); l = S["links_vs_none"].get(a)
        print(f"{a:14s} {m['last']:6.3f} {m['links']:6.2f} {m['rules']:6.0f} {m['fid']:5.2f} {m['reused']:6.2f} {m['uses']:5.1f} {m['rule_share']:10.2f} {m['size']:7.0f}" + (f"  {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | {l[0]:+.2f} [{l[1]:+.2f}, {l[2]:+.2f}]" if d else ""))
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
