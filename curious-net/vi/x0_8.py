"""X0.8 (= idea I9, the follow-up the author flagged 'needs more study' after X0.7): how does the way needs are combined scale with MANY
essential variables whose scales and rates differ?

World WN: N essential variables, each in [0, 1], dying at 0 (all reset on a death). Variable i has a relative scale r_i (log-spaced over a 10x
range): its decay per tick is 0.08 x its scale and an object's effect on it is scale x (direction_i . features), so every variable behaves like
W0's energy at its own scale, but a raw sum of changes is dominated by the large-scale variables. Effect directions drift at season swaps as in W0.
Each tick an object appears; the agent eats or skips it. One TD gradient-descent learner for every arm; arms differ ONLY in the value signal:
  B0      gain in ONE variable (the one whose scale is nearest 1): food only, ignores the others
  SUM     the RAW sum of the changes of all variables (equal weights in raw units; breaks when units differ)
  SUMN    the sum of changes each divided by its variable's scale (a sum done properly; needs the designer to know every scale)
  HOMEO   drive reduction against designer setpoints (0.8 for every variable), Euclidean distance in level units
  MARGIN  change in the weakest variable's LEVEL (levels are in [0, 1] whatever the scale: no designer numbers)
  REF     NOT learned: eats iff that improves the lowest level after the tick's drift, using the TRUE effects (an informed reference, not an
          exact optimum: no exact oracle exists for N > 2)
  RANDOM
lr and discount are tuned per arm and per N on their own objective (seeds apart from the scored ones); the default setting is also reported.
Metrics: deaths per 10,000 ticks, deaths after season swaps, share of deaths caused by each variable, mean level of the weakest variable.
Usage: python x0_8.py check | tune | run [N workers]      (writes ../results/x0_8_<mode>.json)"""
import sys, os, json, time
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
from learner import ValueNet
import x0_1 as X

ARMS = ("B0", "SUM", "SUMN", "HOMEO", "MARGIN", "REF", "RANDOM")
LEARN = ("B0", "SUM", "SUMN", "HOMEO", "MARGIN")
OUT = X.OUT


def rel_scales(N):
    return np.geomspace(1 / np.sqrt(C("wn_scale_range")), np.sqrt(C("wn_scale_range")), N)


class WN:
    def __init__(self, seed, N):
        self.rng = np.random.default_rng(seed); self.N = N; r = rel_scales(N)
        self.s = C("w0_effect_scale") * r; self.d = C("wn_decay_ratio") * self.s
        U = self.rng.standard_normal((N, C("wn_feat_dim"))); self.U = U / np.linalg.norm(U, axis=1, keepdims=True); self.t = 0

    def _drift(self):
        k = C("w0_season_keep"); fresh = self.rng.standard_normal(self.U.shape)
        fresh = fresh - (fresh * self.U).sum(1, keepdims=True) * self.U; fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
        U = k * self.U + np.sqrt(1 - k * k) * fresh; self.U = U / np.linalg.norm(U, axis=1, keepdims=True)

    def next_object(self):
        if self.t > 0 and self.t % C("w0_season_len") == 0: self._drift()
        self.t += 1
        x = self.rng.standard_normal(C("wn_feat_dim")); noise = self.rng.random() < C("w0_noise_frac"); nz = self.rng.standard_normal(self.N) * C("wn_noise_rel") * self.s
        eff = nz if noise else np.clip(self.s * (self.U @ x), -C("wn_clip_mult") * self.s, C("wn_clip_mult") * self.s)
        return np.append(x, float(noise)), eff


def drive(h):
    return float(np.sqrt(np.sum((C("x08_setpoint") - h) ** 2)))


def reward(arm, h0, pre, eat, eff, s, i0):
    if arm == "B0": return float(eff[i0]) if eat else 0.0
    if arm == "SUM": return float(eff.sum()) if eat else 0.0
    if arm == "SUMN": return float((eff / s).sum()) if eat else 0.0
    if arm == "HOMEO": return drive(h0) - drive(pre)
    return float(pre.min() - h0.min())                        # MARGIN


def run(seed, arm, N, lr=0.0, gamma=0.0, ticks=None):
    ticks = ticks or C("x01_ticks"); w = WN(seed, N); i0 = int(np.argmin(np.abs(rel_scales(N) - 1.0)))
    n_in = C("wn_feat_dim") + 1 + N
    net = ValueNet(n_in, 2 * seed + 1, lr) if arm in LEARN else None
    ra = np.random.default_rng(2 * seed + 3); h = np.full(N, C("w0_respawn")); pend = None
    deaths, cause, L, W = [], np.zeros(N), C("w0_season_len"), C("x01_window"); minlev = 0.0
    for t in range(ticks):
        obs, eff = w.next_object(); z = np.r_[obs, h]
        if pend is not None and net is not None:
            zp, ap, rp, term = pend; net.learn(zp, ap, rp if term else rp + gamma * float(np.max(net.q(z)[0])))
        if arm == "RANDOM": a = int(ra.integers(2))
        elif arm == "REF": a = int(np.min(h + eff - w.d) > np.min(h - w.d))
        else: a = net.act(z)
        after = np.clip(h + (eff if a == 1 else 0) - w.d, 0, 1); died = bool((after <= 0).any()); pre = after.copy()
        if died: cause[int(np.argmin(after))] += 1; deaths.append(t); after = np.full(N, C("w0_respawn"))
        if arm in LEARN: pend = (z, a, reward(arm, h, pre, a == 1, eff, w.s, i0), died)
        minlev += float(pre.min()); h = after
    return dict(deaths_per_10k=len(deaths) * C("per_10k") / ticks, post_swap=sum(1 for d in deaths if d >= L and d % L < W), cause=cause.tolist(), mean_min=minlev / ticks)


def job(a):
    tag, seed, arm, N, lr, g = a
    return tag, seed, arm, N, lr, g, run(seed, arm, N, lr, g)


def pool_map(jobs, n):
    with mp.get_context("spawn").Pool(n) as p:
        return p.map(job, jobs, chunksize=4)


def tune(n):
    jobs = []
    for N in C("wn_ns"):
        for arm in LEARN:
            for lr in C("x08_lr_grid"):
                for g in C("x08_gamma_grid"):
                    for i in range(C("x08_tune_worlds")): jobs.append(("tune", C("x08_tune_seed0") + i, arm, N, lr, g))
    res = {}
    for _, seed, arm, N, lr, g, m in pool_map(jobs, n):
        obj = -m["deaths_per_10k"]; res.setdefault((arm, N, lr, g), []).append(obj)
    best = {}
    for N in C("wn_ns"):
        for arm in LEARN:
            cand = {k: float(np.median(v)) for k, v in res.items() if k[0] == arm and k[1] == N}
            k = max(cand, key=cand.get); best[f"{arm}@{N}"] = dict(lr=k[2], gamma=k[3], score=cand[k])
    return best


def main(mode, n):
    X.low_priority(); t0 = time.time()
    if mode == "check":
        out = {}
        for N in C("wn_ns"):
            out[str(N)] = {a: run(C("x08_check_seed"), a, N, C("lr_default"), C("gamma_default")) for a in ("REF", "RANDOM", "B0", "SUM", "SUMN", "MARGIN")}
    elif mode == "tune":
        out = tune(n)
    else:
        tuned = json.load(open(os.path.join(OUT, "x0_8_tune.json")))["result"]; jobs = []
        for N in C("wn_ns"):
            for i in range(C("x01_worlds")):
                s = C("x08_seed0") + i
                for arm in ARMS:
                    if arm in LEARN:
                        b = tuned[f"{arm}@{N}"]; jobs.append(("tuned", s, arm, N, b["lr"], b["gamma"])); jobs.append(("default", s, arm, N, C("lr_default"), C("gamma_default")))
                    else: jobs.append(("tuned", s, arm, N, 0.0, 0.0))
        per = {}
        for tag, seed, arm, N, lr, g, m in pool_map(jobs, n): per.setdefault(f"{tag}|{N}", {}).setdefault(arm, {})[seed] = m
        out = {"tuned": tuned, "per_world": per, "summary": summarize(per)}
    out = {"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, f"x0_8_{mode}.json"), "w"), indent=1, default=float)
    print(json.dumps(out["result"] if mode != "run" else out["result"]["summary"], indent=1, default=float))


def summarize(per):
    S = {}
    for key, arms in per.items():
        g = lambda a, k: np.array([arms[a][s][k] for s in sorted(arms[a])], float); s = {}
        for k in ("deaths_per_10k", "post_swap", "mean_min"): s[k] = {a: float(np.median(g(a, k))) for a in arms}
        N = int(key.split("|")[1]); rank = np.argsort(rel_scales(N))
        causes = {a: np.sum([arms[a][sd]["cause"] for sd in arms[a]], axis=0) for a in arms}
        s["death_share_by_scale_rank"] = {a: (causes[a][rank] / max(1.0, causes[a].sum())).tolist() for a in arms}
        for b in ("SUM", "SUMN", "HOMEO", "B0"): s[f"MARGIN-{b}"] = X.boot_ci(g("MARGIN", "deaths_per_10k") - g(b, "deaths_per_10k"), C("x08_seed0"))
        s["SUM-SUMN"] = X.boot_ci(g("SUM", "deaths_per_10k") - g("SUMN", "deaths_per_10k"), C("x08_seed0"))
        S[key] = s
    return S


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if n == 0:
        import budget as BU; n = min(BU.workers(per_worker_gb=C("x08_worker_gb")), 14)
    main(sys.argv[1], n)
