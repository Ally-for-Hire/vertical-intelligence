"""X1.28: dig until stone (inbox I22). A NOISE-FREE fixed 24-concept world (wstream, noise 0, a pool of 300 candidate inputs per problem); the digger
(vi/digger.py) chooses which 60 to label and builds functions from the input and its own functions; three readings of 'training on total model
size' (author: 'try all the types and record findings').
Arms: DIG_MDL, DIG_GROW, DIG_BUDGET (vi/digger.py), DIG_GROW_RAND (grow, but its 60 inputs are random: does choosing help?), FLAT (flat.OPT on 60 random).
Measures: test R^2 per block of 30 problems; problems explained to R^2 >= 0.99 in the last block; true concepts RECOVERED (some function with
corr^2 >= 0.99 on a probe) by depth, at mid-stream and at the end; whether the last block still beats the one before (still digging or at stone).
Usage: python x1_28.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import flat, digger, wstream

OUT = os.path.join(str(vipaths.RESULTS))
ARMS = {"DIG_MDL": {"size": "mdl", "pick": "mix"}, "DIG_GROW": {"size": "grow", "pick": "mix"}, "DIG_BUDGET": {"size": "budget", "pick": "mix"},   # "mix" = 20 random + 40 chosen (the default until 2026-11-03)
         "DIG_GROW_RAND": {"size": "grow", "rand": 60, "chosen": 0}, "FLAT": None}   # CONST: x128_cfg


def hb(seed, arm, t, T, r2):
    """Live heartbeat (read by vi/ctl.py status): which world/arm/problem a worker is on and its running R^2."""
    try:
        d = os.path.join(OUT, "progress"); os.makedirs(d, exist_ok=True)
        json.dump(dict(exp=os.path.splitext(os.path.basename(sys.argv[0]))[0], seed=seed, arm=arm, t=t, T=T, r2=float(np.mean(r2[-30:])) if r2 else 0.0, updated=time.time()), open(os.path.join(d, f"{os.path.basename(sys.argv[0])}_{seed}.json"), "w"))
    except OSError: pass


def world(seed):
    t0 = time.time(); T = C("x128_problems"); bs = C("x122_block")
    W = wstream.Stream(seed, dict(noise=0.0, n_tr=C("x128_pool"))); data = [W.next() for _ in range(T)]
    rng = np.random.default_rng([seed, C("x126_stream")]); xp = rng.standard_normal((C("x126_probe"), W.cfg["dim"])); V = W.lib.values(xp)
    out = {}
    for a, cfg in ARMS.items():
        r2, rec = [], {}
        if cfg is None or "flat" in cfg:
            L = flat.Learner(seed, flat.OPT if cfg is None else dict(flat.OPT, **cfg["flat"]))
            for t, d in enumerate(data):
                d60 = dict(d, tr=(d["tr"][0][:C("x128_labels")], d["tr"][1][:C("x128_labels")])); L.step(d60)
                x, y = d["te"]; E = L.heads[-1]
                xf = np.vstack([d60["tr"][0], d["va"][0]]); yf = np.concatenate([d60["tr"][1], d["va"][1]])
                act = lambda z: np.column_stack([np.tanh(z @ L.units[i][0] + L.units[i][1]) for i in E]) if E else np.zeros((len(z), 0))
                F = np.hstack([np.ones((len(yf), 1)), act(xf)]); Ft = np.hstack([np.ones((len(y), 1)), act(x)])
                beta = np.linalg.solve(F.T @ F + flat.OPT["ridge"] * np.eye(F.shape[1]), F.T @ yf); r2.append(1 - float(np.mean((y - Ft @ beta) ** 2) / np.var(y))); hb(seed, a, t + 1, T, r2)
                if t + 1 in (T // 2, T):
                    used = sorted({s for dd in data[:t + 1] for s in dd["S"]}); H = np.column_stack([np.tanh(xp @ u + b) for u, b in L.units])
                    Hz = (H - H.mean(0)) / (H.std(0) + 1e-9)
                    rec[t + 1] = {j: float(np.max(np.abs(Hz.T @ ((V[:, j] - V[:, j].mean()) / V[:, j].std())) / len(xp))) for j in used}
            size = sum(len(u) + 1 for u, b in L.units); nodes, sub = len(L.units), 0
        else:
            L = digger.Learner(seed, cfg)
            for t, d in enumerate(data):
                r2.append(L.step(d)["r2"]); hb(seed, a, t + 1, T, r2)
                if t + 1 in (T // 2, T): rec[t + 1] = L.recovered(xp, V, sorted({s for dd in data[:t + 1] for s in dd["S"]}))
            size, nodes, sub = L.size(), len(L.units), sum(u[0] == "s" for u in L.units)
        blocks = [float(np.mean(r2[i:i + bs])) for i in range(0, T, bs)]
        dep = lambda j: W.lib.c[j]["depth"]
        frac = lambda tt, lo, hi: (lambda js: float(np.mean([rec[tt][j] ** 2 >= C("x128_recovered") for j in js])) if js else float("nan"))([j for j in rec[tt] if lo <= dep(j) <= hi])
        out[a] = dict(blocks=blocks, last=blocks[-1], prev=blocks[-2], solved=float(np.mean([x >= C("x128_recovered") for x in r2[-bs:]])),
                      rec_mid_d0=frac(T // 2, 0, 0), rec_mid_d2=frac(T // 2, 2, 99), rec_end_d0=frac(T, 0, 0), rec_end_d1=frac(T, 1, 1), rec_end_d2=frac(T, 2, 99),   # CONST: x123_tail
                      size=size, nodes=nodes, sub=sub)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run():
    import multiprocessing as mp, budget
    from x0_1 import boot_ci
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x128_seed0"); t0 = time.time(); seeds = list(range(seed0, seed0 + C("x119_worlds")))
    with mp.get_context("spawn").Pool(budget.workers()) as pool: res = pool.map(world, seeds, chunksize=1)
    per = dict(zip(seeds, res)); g = lambda a, k: np.array([per[s][a][k] for s in seeds], float)
    pair = lambda a, b, k: boot_ci((g(a, k) - g(b, k))[~np.isnan(g(a, k) - g(b, k))], seed0)
    keys = ("last", "prev", "solved", "rec_mid_d0", "rec_mid_d2", "rec_end_d0", "rec_end_d1", "rec_end_d2", "size", "nodes", "sub")
    S = dict(worlds=len(seeds), med={a: {k: float(np.nanmedian(g(a, k))) for k in keys} for a in ARMS},
             curve={a: [float(np.median([per[s][a]["blocks"][i] for s in seeds])) for i in range(len(per[seeds[0]][a]["blocks"]))] for a in ARMS},
             grow_flat=pair("DIG_GROW", "FLAT", "last"), choose=pair("DIG_GROW", "DIG_GROW_RAND", "last"), grow_mdl=pair("DIG_GROW", "DIG_MDL", "last"),
             grow_budget=pair("DIG_GROW", "DIG_BUDGET", "last"),
             still=boot_ci(g("DIG_GROW", "last") - g("DIG_GROW", "prev"), seed0))
    S["deep_rise"] = boot_ci((g("DIG_GROW", "rec_end_d2") - g("DIG_GROW", "rec_mid_d2"))[~np.isnan(g("DIG_GROW", "rec_end_d2") - g("DIG_GROW", "rec_mid_d2"))], seed0)
    f = lambda k: f"{S[k][0]:+.4f} [{S[k][1]:+.4f}, {S[k][2]:+.4f}]"; up = lambda k: S[k][1] > 0
    V = [("P493", "In a noise-free world the digger ('bigger is better') explains more of later problems than the flat learner: last-block test R^2, DIG_GROW minus FLAT, above 0", "PASSED" if up("grow_flat") else "FAILED", f("grow_flat")),
         ("P494", "Choosing its own inputs helps: DIG_GROW minus DIG_GROW_RAND, last-block R^2, above 0", "PASSED" if up("choose") else "FAILED", f("choose")),
         ("P495", "'Bigger is better' beats 'smallest total description': DIG_GROW minus DIG_MDL, last-block R^2, above 0", "PASSED" if up("grow_mdl") else "FAILED", f("grow_mdl")),
         ("P496", "'Bigger is better' beats a growing size budget: DIG_GROW minus DIG_BUDGET, above 0", "PASSED" if up("grow_budget") else "FAILED", f("grow_budget")),
         ("P497", "Basics first, then deeper: DIG_GROW recovers more 2+-deep concepts at the end than at mid-stream (fraction with corr^2 >= 0.99), above 0", "PASSED" if up("deep_rise") else "FAILED", f("deep_rise")),
         ("P498", "Still digging at the end (not at stone): DIG_GROW's last block beats the one before, above 0", "PASSED" if up("still") else "FAILED", f("still"))]
    out = {"per_world": per, "summary": S, "verdicts": V}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_28_run.json"), "w"), indent=1, default=float)
    for a in ARMS: print(a, {k: round(v, 3) for k, v in S["med"][a].items()}, "curve", " ".join(f"{x:.3f}" for x in S["curve"][a]))
    for v in V: print(v[0], v[2], "|", v[3])
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
