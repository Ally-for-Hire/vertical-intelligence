"""X1.25: CONSOLIDATION ('sleep'): every x125_every problems, every node and every stored problem's readout are trained TOGETHER on all stored problems'
train points (each problem weighted equally), keeping the step with the lowest summed decision error (stop after x18-style patience). A concept
used by many problems is then estimated from all their data, not 60 points at a time (X1.24's candidate cause of the depth plateau).
Worlds and scoring as X1.23/X1.24 (fixed 24-concept world, 240 problems, last 120 by concept depth).
Usage: python x1_25.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import flat, deep, wstream
import x1_19 as L19

OUT = L19.OUT
B1, B2, EPS = flat.B1, flat.B2, flat.EPS


def sleep(L):
    """Joint Adam on all nodes and all readouts over every stored problem; best summed decision error kept."""
    cfg = L.cfg; m = L.units[0][0].shape[0] if L.units else 0
    P = [(s, E, np.hstack([d["tr"][0], np.ones((len(d["tr"][1]), 1))]), d["tr"][1], np.hstack([d["va"][0], np.ones((len(d["va"][1]), 1))]), d["va"][1])
         for s, (E, d) in enumerate(zip(L.heads, L.data)) if E]
    if not P: return 0
    U = np.array([np.concatenate([u, [b]]) for u, b in L.units])
    betas = []
    for s, E, X1, y, V1, yv in P:
        H = np.tanh(X1 @ U[E].T); F = np.hstack([np.ones((len(y), 1)), H])
        betas.append(np.linalg.solve(F.T @ F + cfg["ridge"] * np.eye(len(E) + 1), F.T @ y))
    mU, vU = np.zeros_like(U), np.zeros_like(U); mb = [np.zeros_like(b) for b in betas]; vb = [np.zeros_like(b) for b in betas]
    val = lambda U_, B_: sum(float(np.mean((b[0] + np.tanh(V1 @ U_[E].T) @ b[1:] - yv) ** 2)) for (s, E, X1, y, V1, yv), b in zip(P, B_))
    best, bestU, since, lr = val(U, betas), U.copy(), 0, cfg["ft_lr"]
    for st in range(1, C("x125_steps") + 1):
        gU = np.zeros_like(U)
        for j, (s, E, X1, y, V1, yv) in enumerate(P):
            b = betas[j]; H = np.tanh(X1 @ U[E].T); e = (b[0] + H @ b[1:] - y) * (2.0 / len(y))
            gb = np.concatenate([[e.sum()], H.T @ e])
            gU[E] += ((e[:, None] * b[1:][None, :]) * (1 - H * H)).T @ X1
            mb[j] = B1 * mb[j] + (1 - B1) * gb; vb[j] = B2 * vb[j] + (1 - B2) * gb * gb
            betas[j] = b - lr * (mb[j] / (1 - B1 ** st)) / (np.sqrt(vb[j] / (1 - B2 ** st)) + EPS)
        mU = B1 * mU + (1 - B1) * gU; vU = B2 * vU + (1 - B2) * gU * gU
        U = U - lr * (mU / (1 - B1 ** st)) / (np.sqrt(vU / (1 - B2 ** st)) + EPS)
        v = val(U, betas)
        if v < best: best, bestU, since = v, U.copy(), 0
        else:
            since += 1
            if since >= cfg["ft_patience"]: break
    L.units[:] = [(bestU[i, :m].copy(), float(bestU[i, m])) for i in range(len(L.units))]
    for c in L.caches:
        for i in range(len(L.units)): c.set(i, *L.units[i])
    return st


def world(seed):
    t0 = time.time(); W = wstream.Stream(seed); data = [W.next() for _ in range(C("x122_problems"))]
    depth = [max(W.lib.c[s]["depth"] for s in d["S"]) for d in data]; ceil = [deep.ceiling(d, L19.CEIL) for d in data]
    out = {}
    for a in ("OPT", "SLEEP"):
        L = flat.Learner(seed, flat.OPT); sh = []; steps = []
        for t, (d, c) in enumerate(zip(data, ceil)):
            sh.append(L.step(d)[0] / c)
            if a == "SLEEP" and (t + 1) % C("x125_every") == 0: steps.append(sleep(L))
        tail = list(range(len(data) - C("x123_tail"), len(data)))
        g = lambda lo, hi: [sh[i] for i in tail if lo <= depth[i] <= hi]
        bs = C("x122_block")
        out[a] = dict(all=float(np.mean([sh[i] for i in tail])), d0=float(np.mean(g(0, 0))) if g(0, 0) else float("nan"),
                      d1=float(np.mean(g(1, 1))) if g(1, 1) else float("nan"), d2=float(np.mean(g(2, 99))) if g(2, 99) else float("nan"),   # CONST: x123_tail
                      blocks=[float(np.mean(sh[i:i + bs])) for i in range(0, len(sh), bs)], sleep_steps=float(np.mean(steps)) if steps else 0.0)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run():
    import multiprocessing as mp, budget
    from x0_1 import boot_ci
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x125_seed0"); t0 = time.time(); seeds = list(range(seed0, seed0 + C("x119_worlds")))
    with mp.get_context("spawn").Pool(budget.workers()) as pool: res = pool.map(world, seeds, chunksize=1)
    per = dict(zip(seeds, res))
    g = lambda a, k: np.array([per[s][a][k] for s in seeds], float)
    pair = lambda k: boot_ci((g("SLEEP", k) - g("OPT", k))[~np.isnan(g("SLEEP", k) - g("OPT", k))], seed0)
    blk = lambda a, i: np.array([per[s][a]["blocks"][i] for s in seeds])
    S = dict(worlds=len(seeds), med={a: {k: float(np.nanmedian(g(a, k))) for k in ("all", "d0", "d1", "d2", "sleep_steps")} for a in ("OPT", "SLEEP")},
             curve={a: [float(np.median(blk(a, i))) for i in range(len(per[seeds[0]][a]["blocks"]))] for a in ("OPT", "SLEEP")},
             d2=pair("d2"), all=pair("all"), d0=pair("d0"), rise=boot_ci(blk("SLEEP", -1) - blk("SLEEP", C("x122_mid_block")), seed0))
    f = lambda k: f"{S[k][0]:+.4f} [{S[k][1]:+.4f}, {S[k][2]:+.4f}]"; up = lambda k: S[k][1] > 0
    V = [("P482", "Consolidation raises the share of ceiling on post-plateau problems with a concept 2+ levels deep: SLEEP minus OPT, above 0", "PASSED" if up("d2") else "FAILED", f("d2")),
         ("P483", "...and on all post-plateau problems, above 0", "PASSED" if up("all") else "FAILED", f("all")),
         ("P484", "With consolidation the learner keeps improving after 120 problems: last block minus block 91-120, above 0", "PASSED" if up("rise") else "FAILED", f("rise"))]
    out = {"per_world": per, "summary": S, "verdicts": V}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_25_run.json"), "w"), indent=1, default=float)
    for a in ("OPT", "SLEEP"): print(a, {k: round(v, 3) for k, v in S["med"][a].items()}, "curve", " ".join(f"{x:.3f}" for x in S["curve"][a]))
    print("base-only SLEEP-OPT", f("d0"))
    for v in V: print(v[0], v[2], "|", v[3])
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
