"""X1.26: why do deep problems stall (X1.23-X1.25)? Three diagnostics in one run, on the same fixed 24-concept worlds (240 problems).
  A. IDENTIFIABILITY: after the stream, for every concept the stream used: how well the learner's nodes SPAN the true concept (ridge R^2 of the
     concept on all node outputs, fitted on one half of a 2000-point probe, scored on the other), the best single node's |correlation|, and how many
     problems used the concept.
  B. REPRESENTABILITY: the same concepts fitted DIRECTLY from 1000 points (scored on 1000 more) by (i) one flat tanh node and (ii) a 2-layer tanh
     network with 32 units: can a flat node express a deep concept at all, and can a deeper network?
  C. DATA PER PROBLEM: the learner run with 240 train points per problem (4x): post-plateau share by depth.
Usage: python x1_26.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import flat, deep, wstream
import x1_19 as L19

OUT = L19.OUT


def r2(pred, y): return 1.0 - float(np.mean((y - pred) ** 2) / np.var(y))


def mlp_r2(x, y, xt, yt, rng):
    """2-layer tanh net, Adam, full batch, best test step (a capacity reference only)."""
    m, h = x.shape[1], C("x126_mlp_h")
    P = [rng.standard_normal((m, h)) / math.sqrt(m), np.zeros(h), rng.standard_normal((h, h)) / math.sqrt(h), np.zeros(h), np.zeros(h), np.zeros(1)]
    mo = [np.zeros_like(p) for p in P]; ve = [np.zeros_like(p) for p in P]; best = -9.0   # CONST: x126_mlp_h
    for s in range(1, C("x126_mlp_steps") + 1):
        a1 = np.tanh(x @ P[0] + P[1]); a2 = np.tanh(a1 @ P[2] + P[3]); e = (a2 @ P[4] + P[5] - y) * (2.0 / len(y))   # CONST: x126_mlp_h
        g4 = a2.T @ e; g5 = np.array([e.sum()]); d2 = np.outer(e, P[4]) * (1 - a2 ** 2)
        g2 = a1.T @ d2; g3 = d2.sum(0); d1 = (d2 @ P[2].T) * (1 - a1 ** 2); g0 = x.T @ d1; g1 = d1.sum(0)
        for i, g in enumerate((g0, g1, g2, g3, g4, g5)):
            mo[i] = flat.B1 * mo[i] + (1 - flat.B1) * g; ve[i] = flat.B2 * ve[i] + (1 - flat.B2) * g * g
            P[i] = P[i] - C("x126_mlp_lr") * (mo[i] / (1 - flat.B1 ** s)) / (np.sqrt(ve[i] / (1 - flat.B2 ** s)) + flat.EPS)
        if s % C("x17_check_every") == 0:
            best = max(best, r2(np.tanh(np.tanh(xt @ P[0] + P[1]) @ P[2] + P[3]) @ P[4] + P[5], yt))   # CONST: x126_mlp_h
    return best


def world(seed):
    t0 = time.time(); out = {}
    W = wstream.Stream(seed); data = [W.next() for _ in range(C("x122_problems"))]
    L = flat.Learner(seed, flat.OPT)
    for d in data: L.step(d)
    rng = np.random.default_rng([seed, C("x126_stream")])
    xp = rng.standard_normal((C("x126_probe"), W.cfg["dim"])); h = len(xp) // 2
    V = W.lib.values(xp); H = np.column_stack([np.tanh(xp @ u + b) for u, b in L.units])
    used = {}
    for d in data:
        for s in d["S"]: used[s] = used.get(s, 0) + 1
    rows = []
    for j, n_used in used.items():
        y = V[:, j]; F = np.hstack([np.ones((len(xp), 1)), H])
        beta = np.linalg.solve(F[:h].T @ F[:h] + flat.OPT["ridge"] * np.eye(F.shape[1]), F[:h].T @ y[:h])
        span = r2(F[h:] @ beta, y[h:])
        bestc = float(np.max(np.abs(np.corrcoef(np.column_stack([H, y]).T)[-1, :-1])))
        u, b = flat.train_unit(rng, xp[:h], y[:h] - y[:h].mean(), dict(flat.OPT, unit_steps=C("x126_unit_steps"), stall=0.0))
        z = np.tanh(xp @ u + b); Z = np.column_stack([np.ones(len(xp)), z]); bb = np.linalg.lstsq(Z[:h], y[:h], rcond=None)[0]
        one = r2(Z[h:] @ bb, y[h:])
        net = mlp_r2(xp[:h], y[:h], xp[h:], y[h:], rng)
        rows.append(dict(depth=W.lib.c[j]["depth"], op=W.lib.c[j]["op"], used=n_used, span=span, best_corr=bestc, one_node=one, net=net))
    out["concepts"] = rows
    W4 = wstream.Stream(seed, dict(n_tr=C("x126_ntr_big"))); d4 = [W4.next() for _ in range(C("x122_problems"))]
    depth = [max(W4.lib.c[s]["depth"] for s in d["S"]) for d in d4]; L4 = flat.Learner(seed, flat.OPT)
    sh = [L4.step(d)[0] / deep.ceiling(d, L19.CEIL) for d in d4]; tail = range(len(d4) - C("x123_tail"), len(d4))
    grp = lambda lo, hi: [sh[i] for i in tail if lo <= depth[i] <= hi]
    out["big"] = dict(d0=float(np.mean(grp(0, 0))) if grp(0, 0) else float("nan"), d2=float(np.mean(grp(2, 99))) if grp(2, 99) else float("nan"))   # CONST: x123_tail
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run():
    import multiprocessing as mp, budget
    from x0_1 import boot_ci
    from scipy.stats import spearmanr
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x126_seed0"); t0 = time.time(); seeds = list(range(seed0, seed0 + C("x119_worlds")))
    with mp.get_context("spawn").Pool(budget.workers()) as pool: res = pool.map(world, seeds, chunksize=1)
    per = dict(zip(seeds, res)); allc = [r for s in seeds for r in per[s]["concepts"]]
    G = lambda lo, hi, k: [r[k] for r in allc if lo <= r["depth"] <= hi]
    tab = {f"depth {lo}{'+' if hi > 2 else ''}": {k: float(np.median(G(lo, hi, k))) for k in ("span", "best_corr", "one_node", "net", "used")} | {"n": len(G(lo, hi, "span"))}
           for lo, hi in ((0, 0), (1, 1), (2, 99))}   # CONST: x123_tail
    per_world_gap = np.array([np.median([r["span"] for r in per[s]["concepts"] if r["depth"] == 0]) - np.median([r["span"] for r in per[s]["concepts"] if r["depth"] >= 2])
                              for s in seeds if any(r["depth"] == 0 for r in per[s]["concepts"]) and any(r["depth"] >= 2 for r in per[s]["concepts"])])
    rho = []
    for s in seeds:
        R = [r for r in per[s]["concepts"] if r["depth"] >= 2]
        if len(R) >= C("x126_min_rho"): rho.append(spearmanr([r["used"] for r in R], [r["span"] for r in R])[0])
    rho = np.array([x for x in rho if np.isfinite(x)])
    deep_one = np.array(G(2, 99, "one_node")); deep_net = np.array(G(2, 99, "net"))   # CONST: x123_tail
    big = {k: float(np.nanmedian([per[s]["big"][k] for s in seeds])) for k in ("d0", "d2")}
    S = dict(worlds=len(seeds), table=tab, span_gap=boot_ci(per_world_gap, seed0), rho=boot_ci(rho, seed0), rho_worlds=len(rho),
             deep_one_median=float(np.median(deep_one)), deep_net_median=float(np.median(deep_net)), big=big)
    f = lambda k: f"{S[k][0]:+.4f} [{S[k][1]:+.4f}, {S[k][2]:+.4f}]"
    V = [("P485", "The learner's nodes span deep concepts worse than base concepts: per-world median span R^2 (base minus 2+ deep), above 0", "PASSED" if S["span_gap"][1] > 0 else "FAILED", f("span_gap")),
         ("P486", "For deep concepts, span rises with how many problems used the concept: per-world Spearman, range above 0", "PASSED" if S["rho"][1] > 0 else "FAILED", f("rho") + f" ({len(rho)} worlds)"),
         ("P487", "Representability: fitted directly from 1000 points, one flat node expresses 2+ deep concepts poorly (median R^2 below 0.8) while a 2-layer net does well (above 0.9)",
          "PASSED" if S["deep_one_median"] < C("x126_one_bar") and S["deep_net_median"] > C("x126_net_bar") else "FAILED", f"one node {S['deep_one_median']:.3f}, 2-layer net {S['deep_net_median']:.3f}"),
         ("P488", "With 4x train points per problem the deep-problem share rises above X1.23's 0.287 and the base-minus-deep gap shrinks below X1.23's 0.59",
          "PASSED" if big["d2"] > C("x126_x123_d2") and (big["d0"] - big["d2"]) < C("x126_x123_gap") else "FAILED", f"240 points: base {big['d0']:.3f}, deep {big['d2']:.3f}, gap {big['d0'] - big['d2']:.3f}")]
    out = {"per_world": per, "summary": S, "verdicts": V}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_26_run.json"), "w"), indent=1, default=float)
    for k, v in tab.items(): print(k, {a: round(b, 3) for a, b in v.items()})
    print("4x points:", big)
    for v in V: print(v[0], v[2], "|", v[3])
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
