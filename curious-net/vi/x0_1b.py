"""X0.1b: X0.1 with MULTI-STEP value. Same world, same learner network, same value signals; the only
change is that the learning target is a TD target r + gamma * max_a' Q(z', a') (terminal at death)
instead of the one-step reward. Tests the myopia diagnosis of X0.1's failed P201/P202: the
equivalence of reward seeking and physiological stability (Keramati & Gutkin 2014) needs gamma > 0.
Arms: B0, SUM, HOMEO with TD (lr and gamma tuned per arm on its own objective); HOMEO0 = HOMEO with
gamma = 0 (the X0.1 learner, lr tuned) as an in-experiment control on the same worlds; RANDOM.
Usage: python x0_1b.py tune | run          (writes ../results/x0_1b_<mode>.json)"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
from w0 import W0
from learner import ValueNet
import x0_1 as X

ARMS = ("B0", "SUM", "HOMEO", "HOMEO0", "RANDOM")
VALUE = {"B0": "B0", "SUM": "SUM", "HOMEO": "HOMEO", "HOMEO0": "HOMEO", "RANDOM": "B0"}
OUT = X.OUT


def run_td(seed, arm, lr, gamma, ticks):
    w = W0(seed)
    net = ValueNet(w.obs_dim() + 2, 2 * seed + 1, lr)
    h = np.full(2, C("w0_respawn"))
    deaths, food, pend = [], 0.0, None
    for t in range(ticks):
        obs, eff = w.next_object()
        z = np.r_[obs, h]
        if pend is not None and arm != "RANDOM":
            zp, ap, rp, term = pend
            tgt = rp if term else rp + gamma * float(np.max(net.q(z)[0]))
            net.learn(zp, ap, tgt)
        a = int(net.rng.integers(2)) if arm == "RANDOM" else net.act(z)
        h1, died, pre = W0.apply(h, a == 1, eff)
        pend = (z, a, X.reward(VALUE[arm], h, pre, a == 1, eff), died)
        food += float(eff[0]) if a == 1 else 0.0
        if died:
            deaths.append(t)
        h = h1
    L, W = C("w0_season_len"), C("x01_window")
    res = {"deaths_per_10k": len(deaths) * C("per_10k") / ticks,
           "post_swap_deaths": sum(1 for d in deaths if d >= L and d % L < W), "food": food}
    res.update(X.novel_test(net, w, seed + C("x01_test_offset")))
    return res


def main(mode):
    X.low_priority()
    t0 = time.time()
    if mode == "tune":
        out = {}
        for a in ("B0", "SUM", "HOMEO", "HOMEO0"):
            gams = (0.0,) if a == "HOMEO0" else C("gamma_grid")
            scores = {}
            for g in gams:
                for lr in C("lr_grid"):
                    rs = [run_td(C("x01b_tune_seed0") + i, a, lr, g, C("x01_ticks")) for i in range(C("x01_tune_worlds"))]
                    scores[f"{lr}|{g}"] = float(np.median([X.objective(VALUE[a], r) for r in rs]))
            best = max(scores, key=scores.get)
            out[a] = {"scores": scores, "best_lr": float(best.split("|")[0]), "best_gamma": float(best.split("|")[1])}
    else:
        tuned = json.load(open(os.path.join(OUT, "x0_1b_tune.json")))["result"]
        out = {"hyper": {a: (tuned[a]["best_lr"], tuned[a]["best_gamma"]) for a in tuned}, "per_world": {}}
        out["hyper"]["RANDOM"] = (C("lr_default"), 0.0)
        for setting in ("tuned", "default"):
            pw = {a: [] for a in ARMS}
            for i in range(C("x01_worlds")):
                s = C("x01b_seed0") + i
                for a in ARMS:
                    lr, g = out["hyper"][a] if setting == "tuned" else (C("lr_default"), 0.0 if a == "HOMEO0" else C("gamma_default"))
                    pw[a].append(run_td(s, a, lr, g, C("x01_ticks")))
            out["per_world"][setting] = pw
        out["summary"] = summarize(out["per_world"])
    out = {"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f"x0_1b_{mode}.json"), "w"), indent=1)
    print(json.dumps(out["result"]["summary"] if mode == "run" else out["result"], indent=1))


def summarize(pw):
    S = {}
    for setting, d in pw.items():
        g = lambda a, k: np.array([r[k] for r in d[a]])
        s = {}
        for k in ("deaths_per_10k", "post_swap_deaths", "valence_acc", "netharm_accept", "need_slope"):
            s[k] = {a: float(np.median(g(a, k))) for a in ARMS}
            for b in ("HOMEO0", "SUM", "B0"):
                s[k][f"HOMEO-{b}"] = X.boot_ci(g("HOMEO", k) - g(b, k), C("x01b_seed0"))
        S[setting] = s
    return S


if __name__ == "__main__":
    main(sys.argv[1])
