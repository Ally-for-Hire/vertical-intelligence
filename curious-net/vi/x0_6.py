"""X0.6: can value come from keeping yourself alive WITHOUT any designer-given setpoint or weighting?
(experiment-plan section 0; follows X0.1/X0.1b, whose self-defined arm used designer setpoints.)

World: W0 with orthogonal directions (exact survival oracle valid). One gradient-descent value learner
(learner.ValueNet, TD target, lr and gamma tuned per arm on its own objective). Arms differ ONLY in the
value signal:
  B0      food energy gained                         (goal-trained baseline)
  SUM     energy + integrity change                  (designer: equal weights, linear)
  HOMEO   drive reduction vs DESIGNER setpoints      (X0.1)
  DEATH   -1 at death, else 0                        (raw survival signal; sparse; the control)
  VIAB    the agent LEARNS its own viability V(h) = -E[gamma^(time to death)] from deaths alone (TD on
          a small net of its internal state), and is rewarded by the change in that learned viability.
          No setpoint, no variable weighting: only that dying is bad.
  ORACLE  exact DP survival policy that knows each object's effects (ceiling; oracle.py)
  RANDOM  random actions
Valence is now STATE-AWARE: agreement of an arm greedy choice with the oracle choice on novel objects at
states sampled uniformly over the viable range, counting only decisions where the oracle has a real
preference (the state-blind valence accuracy of X0.1 scored against a designer notion of good, not
against survival; a smoke run showed the oracle is indifferent at safe states, so agreement is only
defined where survival is at stake).
Usage: python x0_6.py smoke | tune | run     (writes ../results/x0_6_<mode>.json)"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
from w0 import W0
from learner import ValueNet
from oracle import Oracle, interp
import x0_1 as X

ARMS = ("B0", "SUM", "HOMEO", "DEATH", "VIAB", "ORACLE", "RANDOM")
NETARMS = ("B0", "SUM", "HOMEO", "DEATH", "VIAB")
OUT = X.OUT
_ORACLE = {}


def oracle(rho=0.0):
    if rho not in _ORACLE:
        _ORACLE[rho] = Oracle(rho)
    return _ORACLE[rho]


class VNet:
    """Scalar viability net V(h): 2 -> v_hidden (tanh) -> 1, Adam, TD updates on the agent own transitions."""
    def __init__(self, seed, lr):
        rng = np.random.default_rng(seed)
        H = C("v_hidden")
        self.p = {"W1": rng.standard_normal((2, H)) * C("init_scale") / np.sqrt(2), "b1": np.zeros(H),
                  "W2": rng.standard_normal(H) * C("init_scale") / np.sqrt(H), "b2": np.zeros(())}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.lr, self.t = lr, 0

    def __call__(self, h):
        return float(np.tanh(h @ self.p["W1"] + self.p["b1"]) @ self.p["W2"] + self.p["b2"])

    def update(self, h, target):
        a = np.tanh(h @ self.p["W1"] + self.p["b1"])
        err = float(a @ self.p["W2"] + self.p["b2"]) - target
        g_a = err * self.p["W2"] * (1 - a * a)
        g = {"W2": err * a, "b2": np.array(err), "W1": np.outer(h, g_a), "b1": g_a}
        self.t += 1
        b1, b2, eps = C("adam_b1"), C("adam_b2"), C("adam_eps")
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh, vh = self.m[k] / (1 - b1 ** self.t), self.v[k] / (1 - b2 ** self.t)
            self.p[k] = self.p[k] - self.lr * mh / (np.sqrt(vh) + eps)


def agreement(net, w, states, seed, rho=0.0):
    """Fraction of informative (oracle not indifferent) probes where the arm greedy choice == the oracle choice."""
    rng = np.random.default_rng(seed)
    o, n_ok, n = oracle(rho), 0, 0
    for h in states:
        for _ in range(C("x06_objs")):
            x = rng.standard_normal(C("w0_feat_dim"))
            e = w.effects(x, False)
            qd = o.qdiff(h, e)
            if abs(qd) < C("x06_indiff"):
                continue
            choice = int(np.argmax(net.q(np.r_[x, 0.0, h])[0]))
            n += 1
            n_ok += int(choice == int(qd > 0))
    return (n_ok / n) if n else float("nan"), n


def run06(seed, arm, lr, gamma, lr_v, ticks, rho=0.0):
    w = W0(seed, orthogonal=True, corr=rho)
    net = None if arm in ("ORACLE", "RANDOM") else ValueNet(w.obs_dim() + 2, 2 * seed + 1, lr)
    vn = VNet(2 * seed + 2, lr_v) if arm == "VIAB" else None
    rng = np.random.default_rng(2 * seed + 3)
    h = np.full(2, C("w0_respawn"))
    deaths, food, pend = [], 0.0, None
    for t in range(ticks):
        obs, eff = w.next_object()
        z = np.r_[obs, h]
        if pend is not None and net is not None:
            zp, ap, rp, term = pend
            net.learn(zp, ap, rp if term else rp + gamma * float(np.max(net.q(z)[0])))
        if arm == "ORACLE":
            a = int(oracle(rho).qdiff(h, eff) > 0)
        elif arm == "RANDOM":
            a = int(rng.integers(2))
        else:
            a = net.act(z)
        h1, died, pre = W0.apply(h, a == 1, eff)
        if arm in ("B0", "SUM"):
            r = X.reward(arm, h, pre, a == 1, eff)
        elif arm == "HOMEO":
            r = X.reward("HOMEO", h, pre, a == 1, eff)
        elif arm == "DEATH":
            r = -1.0 if died else 0.0
        elif arm == "MARGIN":
            r = float(pre.min() - h.min())  # weakest-link margin: the only given is that 0 means death
        elif arm == "VIAB":
            if died:
                r = -1.0 - vn(h)
                vn.update(h, -1.0)
            else:
                r = vn(h1) - vn(h)
                vn.update(h, C("gamma_v") * vn(h1))
        else:
            r = 0.0
        pend = (z, a, r, died)
        food += float(eff[0]) if a == 1 else 0.0
        if died:
            deaths.append(t)
        h = h1
    L, W = C("w0_season_len"), C("x01_window")
    res = {"deaths_per_10k": len(deaths) * C("per_10k") / ticks,
           "post_swap_deaths": sum(1 for d in deaths if d >= L and d % L < W), "food": food}
    if net is not None:
        res.update(X.novel_test(net, w, seed + C("x01_test_offset")))
        srng = np.random.default_rng(seed + C("x01_test_offset") + 1)
        states = srng.uniform(C("vgrid_lo"), C("vgrid_hi"), (C("x06_states"), 2))
        res["agree_danger"], res["n_danger"] = agreement(net, w, states, seed + 2, rho)
    if vn is not None:
        ax = np.linspace(C("vgrid_lo"), C("vgrid_hi"), C("vgrid"))
        P = np.array([[a, b] for a in ax for b in ax])
        learned = np.array([vn(p) for p in P])
        true = interp(oracle(rho).V, P)
        res["v_corr"] = float(np.corrcoef(learned, true)[0, 1]) if learned.std() > 0 and true.std() > 0 else 0.0
    return res


def hyper_grid(arm):
    if arm == "VIAB":
        return [(lr, g, lv) for lr in C("lr_grid") for g in C("gamma_grid") for lv in C("lr_v_grid")]
    return [(lr, g, C("lr_v_default")) for lr in C("lr_grid") for g in C("gamma_grid")]


def objective(arm, r):
    return r["food"] if arm == "B0" else -r["deaths_per_10k"]


def main(mode):
    X.low_priority()
    t0 = time.time()
    if mode == "smoke":
        out = {a: run06(C("x06_smoke_seed"), a, C("lr_default"), C("gamma_default"), C("lr_v_default"), C("x01_ticks") // 5) for a in ARMS}
    elif mode == "tune":
        out = {}
        for a in NETARMS:
            scores = {}
            for lr, g, lv in hyper_grid(a):
                rs = [run06(C("x06_tune_seed0") + i, a, lr, g, lv, C("x01_ticks")) for i in range(C("x01_tune_worlds"))]
                scores[f"{lr}|{g}|{lv}"] = float(np.median([objective(a, r) for r in rs]))
            best = max(scores, key=scores.get).split("|")
            out[a] = {"scores": scores, "best": [float(b) for b in best]}
    else:
        tuned = json.load(open(os.path.join(OUT, "x0_6_tune.json")))["result"]
        out = {"hyper": {a: tuned[a]["best"] for a in tuned}, "per_world": {}}
        for setting in ("tuned", "default"):
            pw = {a: [] for a in ARMS}
            for i in range(C("x01_worlds")):
                s = C("x06_seed0") + i
                for a in ARMS:
                    if setting == "tuned":
                        lr, g, lv = out["hyper"].get(a, [C("lr_default"), 0.0, C("lr_v_default")])
                    else:
                        lr, g, lv = C("lr_default"), C("gamma_default"), C("lr_v_default")
                    pw[a].append(run06(s, a, lr, g, lv, C("x01_ticks")))
            out["per_world"][setting] = pw
        out["summary"] = summarize(out["per_world"])
    out = {"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f"x0_6_{mode}.json"), "w"), indent=1, default=float)
    print(json.dumps(out["result"]["summary"] if mode == "run" else out["result"], indent=1, default=float))


def summarize(pw):
    S = {}
    for setting, d in pw.items():
        g = lambda a, k: np.array([r.get(k, np.nan) for r in d[a]], float)
        s = {}
        for k in ("deaths_per_10k", "post_swap_deaths", "need_slope", "agree_danger", "n_danger"):
            s[k] = {a: float(np.nanmedian(g(a, k))) for a in ARMS if not np.isnan(g(a, k)).all()}
            for b in ("B0", "DEATH", "HOMEO", "SUM"):
                dd = g("VIAB", k) - g(b, k)
                s[k][f"VIAB-{b}"] = X.boot_ci(dd[~np.isnan(dd)], C("x06_seed0"))
        s["v_corr_median"] = float(np.median(g("VIAB", "v_corr")))
        s["oracle_deaths_vs_median_other"] = [s["deaths_per_10k"]["ORACLE"], float(np.median([s["deaths_per_10k"][a] for a in ARMS if a != "ORACLE"]))]
        S[setting] = s
    return S


if __name__ == "__main__":
    main(sys.argv[1])
