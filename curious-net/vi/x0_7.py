"""X0.7: is a GRADED WARNING with no designer values enough? (follows X0.6, where a viability learned from
deaths alone was data-starved and showed no need-dependent valence.)

MARGIN: reward = change in the margin of the WEAKEST essential variable, min_i h_i. No setpoints, no
variable weights, nothing learned: the only given is that a variable at 0 means death (Ashby essential
variables). It formalises "no part left behind" (U10): the neglected part dominates.

Two worlds (W0 with orthogonal effect directions, exact survival oracle valid in both):
  orth   energy and integrity effects independent (rho = 0), the world of X0.6;
  trade  effects anti-correlated (rho = x07_rho): food that helps one variable tends to harm the other, so
         which food is good depends on which variable is weak. A state-blind sum should suffer here.
Arms: B0, SUM, HOMEO (designer setpoints), MARGIN, VIAB (learned from deaths; X0.6 hyperparameters,
not retuned), ORACLE (ceiling), RANDOM. One TD gradient-descent learner in every learning arm; lr and
discount tuned per arm and world on its own objective (B0, SUM, HOMEO in the orth world reuse the X0.6
tuning, same world type).
Usage: python x0_7.py tune | run          (writes ../results/x0_7_<mode>.json)"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x0_1 as X
import x0_6 as Z

ARMS = ("B0", "SUM", "HOMEO", "MARGIN", "VIAB", "ORACLE", "RANDOM")
TUNE_ARMS = {"orth": ("MARGIN",), "trade": ("B0", "SUM", "HOMEO", "MARGIN")}
OUT = X.OUT


def worlds():
    return {"orth": 0.0, "trade": C("x07_rho")}


def grid():
    return [(lr, g, C("lr_v_default")) for lr in C("lr_grid") for g in C("gamma_grid")]


def main(mode):
    X.low_priority()
    t0 = time.time()
    if mode == "tune":
        out = {}
        for wn, rho in worlds().items():
            out[wn] = {}
            for a in TUNE_ARMS[wn]:
                scores = {}
                for lr, g, lv in grid():
                    rs = [Z.run06(C("x07_tune_seed0") + i, a, lr, g, lv, C("x01_ticks"), rho) for i in range(C("x01_tune_worlds"))]
                    scores[f"{lr}|{g}|{lv}"] = float(np.median([Z.objective(a, r) for r in rs]))
                best = max(scores, key=scores.get).split("|")
                out[wn][a] = {"scores": scores, "best": [float(b) for b in best]}
    else:
        t6 = json.load(open(os.path.join(OUT, "x0_6_tune.json")))["result"]
        t7 = json.load(open(os.path.join(OUT, "x0_7_tune.json")))["result"]
        hyper = {}
        for wn in worlds():
            hyper[wn] = {a: t6[a]["best"] for a in ("B0", "SUM", "HOMEO", "VIAB")} if wn == "orth" else {"VIAB": t6["VIAB"]["best"]}
            hyper[wn].update({a: v["best"] for a, v in t7[wn].items()})
        out = {"hyper": hyper, "per_world": {}}
        for wn, rho in worlds().items():
            for setting in ("tuned", "default"):
                pw = {a: [] for a in ARMS}
                for i in range(C("x01_worlds")):
                    s = C("x07_seed0") + i
                    for a in ARMS:
                        if setting == "tuned":
                            lr, g, lv = hyper[wn].get(a, [C("lr_default"), 0.0, C("lr_v_default")])
                        else:
                            lr, g, lv = C("lr_default"), C("gamma_default"), C("lr_v_default")
                        pw[a].append(Z.run06(s, a, lr, g, lv, C("x01_ticks"), rho))
                out["per_world"][f"{wn}/{setting}"] = pw
        out["summary"] = summarize(out["per_world"])
    out = {"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f"x0_7_{mode}.json"), "w"), indent=1, default=float)
    print(json.dumps(out["result"]["summary"] if mode == "run" else out["result"], indent=1, default=float))


def summarize(pw):
    S = {}
    for key, d in pw.items():
        g = lambda a, k: np.array([r.get(k, np.nan) for r in d[a]], float)
        s = {}
        for k in ("deaths_per_10k", "post_swap_deaths", "need_slope", "agree_danger", "n_danger"):
            s[k] = {a: float(np.nanmedian(g(a, k))) for a in ARMS if not np.isnan(g(a, k)).all()}
            for b in ("B0", "SUM", "HOMEO", "VIAB"):
                dd = g("MARGIN", k) - g(b, k)
                s[k][f"MARGIN-{b}"] = X.boot_ci(dd[~np.isnan(dd)], C("x07_seed0"))
        S[key] = s
    return S


if __name__ == "__main__":
    main(sys.argv[1])
