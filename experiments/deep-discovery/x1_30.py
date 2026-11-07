"""X1.30: an AI whose goal is to make everything SIMPLE (author 2026-11-03: 'not certain, simple. everything should be simple, easy'). 'Simple' has several
readings, so EVERY reading is an arm (author rule), each swept over how much error it tolerates ('not certain'): the same noise-free world and seeds as
X1.28/29 (first x130_worlds), all-random labels, default compute (least compute for the information: this is a map, not a best-shot run).
Readings: MDL (charge per weight 0.5/1/2/4/8 x log2 n), NODE (charge per new function 5/20/80 bits, any size), LINKS (at most 1/2/4 functions per problem),
FLATONLY (no multi-layer functions, MDL 0.5), REUSE (use known functions first, new ones only for the remainder, MDL 0.5); references: GROW (no simplicity pressure), FLAT.
Measures: last-block test R^2 (how right), weights, functions, links per problem, seconds (how easy to learn), Pareto front of (R^2 up, weights down).
Usage: python x1_30.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_28

OUT = x1_28.OUT
ARMS = {"GROW": dict(size="grow"), "FLAT": None, "REUSE": dict(size="mdl", reuse_first=True), "FLATONLY": dict(size="mdl", n_sub=0)}
for w in (0.5, 1, 2, 4, 8): ARMS[f"MDL_{w}"] = dict(size="mdl", mdl_w=w)
for b in (5, 20, 80): ARMS[f"NODE_{b}"] = dict(size="mdl", mdl_w=0.0, node_bits=float(b))
for k in (1, 2, 4): ARMS[f"LINKS_{k}"] = dict(size="mdl", max_steps=k)                                            # CONST: x130_cfg


def world(seed):
    import digger, wstream, flat
    t0 = time.time(); T = C("x128_problems"); bs = C("x122_block")
    W = wstream.Stream(seed, dict(noise=0.0, n_tr=C("x128_pool"))); data = [W.next() for _ in range(T)]; out = {}
    for a, cfg in ARMS.items():
        ta = time.time(); r2 = []
        if cfg is None:
            L = flat.Learner(seed, flat.OPT)
            for t, d in enumerate(data):
                d60 = dict(d, tr=(d["tr"][0][:60], d["tr"][1][:60])); L.step(d60); x, y = d["te"]; E = L.heads[-1]
                xf = np.vstack([d60["tr"][0], d["va"][0]]); yf = np.concatenate([d60["tr"][1], d["va"][1]])
                act = lambda z: np.column_stack([np.tanh(z @ L.units[i][0] + L.units[i][1]) for i in E]) if E else np.zeros((len(z), 0))
                F = np.hstack([np.ones((len(yf), 1)), act(xf)]); Ft = np.hstack([np.ones((len(y), 1)), act(x)])
                beta = np.linalg.solve(F.T @ F + flat.OPT["ridge"] * np.eye(F.shape[1]), F.T @ yf); r2.append(1 - float(np.mean((y - Ft @ beta) ** 2) / np.var(y))); x1_28.hb(seed, a, t + 1, T, r2)
            size, nodes, links = sum(len(u) + 1 for u, b in L.units), len(L.units), float(np.mean([len(h) for h in L.heads[-bs:]]))
        else:
            L = digger.Learner(seed, cfg)
            for t, d in enumerate(data): r2.append(L.step(d)["r2"]); x1_28.hb(seed, a, t + 1, T, r2)
            size, nodes, links = L.size(), len(L.units), float(np.mean([len(h) for h in L.heads[-bs:]]))
        out[a] = dict(last=float(np.mean(r2[-bs:])), size=size, nodes=nodes, links=links, seconds=round(time.time() - ta, 1))
    return out


def run():
    import multiprocessing as mp, budget
    from x0_1 import boot_ci
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x128_seed0"); t0 = time.time(); seeds = list(range(seed0, seed0 + C("x130_worlds")))
    with mp.get_context("spawn").Pool(budget.workers()) as pool: res = pool.map(world, seeds, chunksize=1)
    per = dict(zip(seeds, res)); g = lambda a, k: np.array([per[s][a][k] for s in seeds], float)
    med = {a: {k: float(np.median(g(a, k))) for k in ("last", "size", "nodes", "links", "seconds")} for a in ARMS}
    front = [a for a in ARMS if not any(med[b]["last"] >= med[a]["last"] and med[b]["size"] <= med[a]["size"] and (med[b]["last"] > med[a]["last"] or med[b]["size"] < med[a]["size"]) for b in ARMS)]
    rk = lambda k, rev: {a: i + 1 for i, a in enumerate(sorted(ARMS, key=lambda a: med[a][k], reverse=rev))}
    ranks = {"accuracy": rk("last", True), "weights": rk("size", False), "functions": rk("nodes", False), "links": rk("links", False), "seconds": rk("seconds", False)}
    S = dict(worlds=len(seeds), med=med, pareto=front, ranks=ranks, vs_grow={a: boot_ci(g(a, "last") - g("GROW", "last"), seed0) for a in ARMS if a != "GROW"})
    json.dump({"seconds": round(time.time() - t0, 1), "result": {"per_world": per, "summary": S}}, open(os.path.join(OUT, "x1_30_run.json"), "w"), indent=1, default=float)
    print(f"{'arm':10s} {'R2':>6s} {'weights':>8s} {'funcs':>6s} {'links':>6s} {'sec':>6s} {'R2/1000w':>9s}  vs GROW")
    for a in ARMS:
        m = med[a]; d = S["vs_grow"].get(a); print(f"{a:10s} {m['last']:6.3f} {m['size']:8.0f} {m['nodes']:6.0f} {m['links']:6.2f} {m['seconds']:6.0f} {m['last'] / max(m['size'], 1) * 1000:9.2f}" + (f"  {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else ""))
    print("Pareto front (more right AND fewer weights):", front)
    for k, r in ranks.items(): print("rank by", k, ":", sorted(r, key=r.get)[:6])
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
