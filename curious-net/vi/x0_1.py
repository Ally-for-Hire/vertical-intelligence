"""X0.1 (experiment-plan, section 0): can value come ONLY from keeping yourself intact?

Every arm uses the same gradient-descent learner (learner.ValueNet), the same initial weights, and the
same object stream; only the VALUE SIGNAL differs:
  B0      external task reward: energy gained from food (the plain goal-trained baseline)
  SUM     both channels, linear: energy + integrity change from the object (more reward, no self)
  HOMEO   value defined by the self: reduction of the drive D = |setpoint - internal state|
          (Keramati & Gutkin 2014), so the same object is worth different amounts in different states
  RANDOM  random actions, no learning
Learning rate: tuned per arm on its OWN objective on a separate seed block, plus every arm at the
Adam default (method lesson: report tuned and default). Primary analysis = tuned.

Usage: python x0_1.py pilot | tune | run       (writes ../results/x0_1_<mode>.json)"""
import sys, os, json, time, ctypes
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
from w0 import W0
from learner import ValueNet

ARMS = ("B0", "SUM", "HOMEO", "RANDOM")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def low_priority():
    try:
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), C("win_below_normal"))
    except Exception:
        pass


def drive(h):
    sp = np.array(C("setpoint"))
    return float(np.sum(np.abs(sp - h) ** C("drive_power")) ** (1 / C("drive_power")))


def reward(arm, h0, pre, eat, eff):
    if arm == "B0":
        return float(eff[0]) if eat else 0.0
    if arm == "SUM":
        return float(eff.sum()) if eat else 0.0
    return drive(h0) - drive(pre)  # HOMEO: drive reduction, both actions, death included via pre


def novel_test(net, w, seed):
    """Greedy choices on novel objects drawn from the CURRENT season's structure (never seen exactly)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((C("x01_novel"), C("w0_feat_dim")))
    E = np.array([w.effects(x, False) for x in X])
    good, bad = (E > 0).all(1), (E < 0).all(1)
    netharm = (E[:, 0] > 0) & (E.sum(1) < 0)
    trade = (E[:, 0] > 0) & (E[:, 1] < 0)

    def eats(h):
        Z = np.c_[X, np.zeros(len(X)), np.tile(h, (len(X), 1))]
        return np.array([np.argmax(net.q(z)[0]) == 1 for z in Z])
    neutral = eats(np.full(2, C("w0_respawn")))
    starving = eats(np.array(C("x01_state_starving")))
    sated = eats(np.array(C("x01_state_sated")))
    acc = np.r_[neutral[good], ~neutral[bad]].mean()
    return {"valence_acc": float(acc), "netharm_accept": float(neutral[netharm].mean()),
            "need_slope": float(starving[trade].mean() - sated[trade].mean())}


def run(seed, arm, lr, ticks):
    w = W0(seed)
    n_in = w.obs_dim() + 2
    net = ValueNet(n_in, 2 * seed + 1, lr)
    h = np.full(2, C("w0_respawn"))
    deaths, food, r_arm, r_sum = [], 0.0, [], []
    for t in range(ticks):
        obs, eff = w.next_object()
        z = np.r_[obs, h]
        a = int(net.rng.integers(2)) if arm == "RANDOM" else net.act(z)
        h1, died, pre = W0.apply(h, a == 1, eff)
        r = reward(arm, h, pre, a == 1, eff)
        if arm == "HOMEO":
            r_arm.append(r); r_sum.append(reward("SUM", h, pre, a == 1, eff))
        if arm != "RANDOM":
            net.learn(z, a, r)
        food += float(eff[0]) if a == 1 else 0.0
        if died:
            deaths.append(t)
        h = h1
    L, W = C("w0_season_len"), C("x01_window")
    post = sum(1 for d in deaths if d >= L and d % L < W)
    res = {"deaths_per_10k": len(deaths) * C("per_10k") / ticks, "post_swap_deaths": post, "food": food}
    res.update(novel_test(net, w, seed + C("x01_test_offset")))
    if r_arm:
        res["manip_corr_homeo_vs_sum"] = float(np.corrcoef(r_arm, r_sum)[0, 1])
    return res


def boot_ci(d, seed):
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    meds = np.median(rng.choice(d, (C("boot"), d.size)), axis=1)
    return [float(np.median(d)), float(np.percentile(meds, C("ci_lo"))), float(np.percentile(meds, C("ci_hi")))]


def objective(arm, res):
    return {"B0": res["food"], "SUM": -res["deaths_per_10k"], "HOMEO": -res["deaths_per_10k"]}[arm]


def main(mode):
    low_priority()
    t0 = time.time()
    if mode == "pilot":
        out = {a: run(C("x01_pilot_seed"), a, C("lr_default"), C("x01_ticks")) for a in ARMS}
    elif mode == "tune":
        out = {}
        for a in ARMS[:3]:
            scores = {}
            for lr in C("lr_grid"):
                rs = [run(C("x01_tune_seed0") + i, a, lr, C("x01_ticks")) for i in range(C("x01_tune_worlds"))]
                scores[str(lr)] = float(np.median([objective(a, r) for r in rs]))
            out[a] = {"scores": scores, "best_lr": float(max(scores, key=scores.get))}
    else:
        tuned = json.load(open(os.path.join(OUT, "x0_1_tune.json")))["result"]
        out = {"lr": {a: tuned[a]["best_lr"] for a in ARMS[:3]}, "per_world": {}}
        out["lr"]["RANDOM"] = C("lr_default")
        for setting in ("tuned", "default"):
            pw = {a: [] for a in ARMS}
            for i in range(C("x01_worlds")):
                s = C("x01_seed0") + i
                for a in ARMS:
                    lr = out["lr"][a] if setting == "tuned" else C("lr_default")
                    pw[a].append(run(s, a, lr, C("x01_ticks")))
            out["per_world"][setting] = pw
        out["summary"] = summarize(out["per_world"])
    out = {"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f"x0_1_{mode}.json"), "w"), indent=1)
    print(json.dumps(out if mode != "run" else out["result"]["summary"], indent=1))


def summarize(pw):
    S = {}
    for setting, d in pw.items():
        g = lambda a, k: np.array([r[k] for r in d[a]])
        s = {}
        for k in ("deaths_per_10k", "post_swap_deaths", "valence_acc", "netharm_accept", "need_slope"):
            s[k] = {a: float(np.median(g(a, k))) for a in ARMS}
            s[k]["HOMEO-B0"] = boot_ci(g("HOMEO", k) - g("B0", k), C("x01_seed0"))
            s[k]["HOMEO-SUM"] = boot_ci(g("HOMEO", k) - g("SUM", k), C("x01_seed0"))
        s["manip_corr_homeo_vs_sum"] = float(np.median(g("HOMEO", "manip_corr_homeo_vs_sum")))
        S[setting] = s
    return S


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot")
