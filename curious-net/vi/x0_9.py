"""X0.9 (diagnosis of the X0.8 failures): are the UNITS of a need's reported change, or its RATE of change, what decided the combiners?

In X0.8 a variable's scale multiplied both its units and its dynamics, so the raw sum weighted the fast variables most. Here each variable i
has a UNIT factor u_i (log-spaced over a 10x range) that multiplies ONLY the change the SUM signal sees, and its own RATE (decay, effect size and
noise in level units) set separately:
  U1  rates all equal (W0's energy), units differ        -> units unrelated to urgency
  U2  rate_i = 1/u_i                                      -> the largest unit sits on the SLOWEST variable (anti-coupled)
  X8  rate_i = u_i                                        -> the X0.8 world; used ONLY for the parity check against x0_8.py
Arms are those of X0.8 with SUM = sum of u_i x change, SUMN = sum of change (a sum done properly: units divided out), HOMEO and MARGIN on unit-free
levels (unchanged), B0 = the variable whose unit is nearest 1. lr and discount are tuned per arm, N and world on their own objective.
Usage: python x0_9.py check | tune | run [N workers]      (writes ../results/x0_9_<mode>.json)"""
import sys, os, json, time
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
from learner import ValueNet
import x0_1 as X
import x0_8 as E8
from x0_8 import ARMS, LEARN, OUT, drive

WORLDS = ("U1", "U2")


def units(N):
    r = C("x09_unit_range"); return np.geomspace(1 / np.sqrt(r), np.sqrt(r), N)


def rates(world, N):
    u = units(N); return {"U1": np.ones(N), "U2": 1 / u, "X8": u}[world]


class WN9(E8.WN):
    def __init__(self, seed, N, world):
        super().__init__(seed, N)
        self.s = C("w0_effect_scale") * rates(world, N); self.d = C("wn_decay_ratio") * self.s; self.u = units(N)


def reward(arm, h0, pre, eat, eff, u, i0):
    if arm == "B0": return float(eff[i0]) if eat else 0.0
    if arm == "SUM": return float((u * eff).sum()) if eat else 0.0
    if arm == "SUMN": return float(eff.sum()) if eat else 0.0
    if arm == "HOMEO": return drive(h0) - drive(pre)
    return float(pre.min() - h0.min())                        # MARGIN


def run(seed, arm, N, world, lr=0.0, gamma=0.0, ticks=None):
    ticks = ticks or C("x01_ticks"); w = WN9(seed, N, world); i0 = int(np.argmin(np.abs(w.u - 1.0)))
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
        if arm in LEARN: pend = (z, a, reward(arm, h, pre, a == 1, eff, w.u, i0), died)
        minlev += float(pre.min()); h = after
    return dict(deaths_per_10k=len(deaths) * C("per_10k") / ticks, post_swap=sum(1 for d in deaths if d >= L and d % L < W), cause=cause.tolist(), mean_min=minlev / ticks)


def job(a):
    tag, seed, arm, N, world, lr, g = a
    return tag, seed, arm, N, world, lr, g, run(seed, arm, N, world, lr, g)


def pool_map(jobs, n):
    with mp.get_context("spawn").Pool(n) as p:
        return p.map(job, jobs, chunksize=4)


def tune(n):
    jobs = [("tune", C("x09_tune_seed0") + i, arm, N, wd, lr, g) for wd in WORLDS for N in C("wn_ns") for arm in LEARN
            for lr in C("x08_lr_grid") for g in C("x08_gamma_grid") for i in range(C("x08_tune_worlds"))]
    res = {}
    for _, seed, arm, N, wd, lr, g, m in pool_map(jobs, n): res.setdefault((arm, N, wd, lr, g), []).append(-m["deaths_per_10k"])
    best = {}
    for wd in WORLDS:
        for N in C("wn_ns"):
            for arm in LEARN:
                cand = {k: float(np.median(v)) for k, v in res.items() if k[:3] == (arm, N, wd)}
                k = max(cand, key=cand.get); best[f"{arm}@{N}@{wd}"] = dict(lr=k[3], gamma=k[4], score=cand[k])
    return best


def check():
    s = C("x09_check_seed"); out = {"parity_vs_x08": {}, "world": {}}
    for N in C("wn_ns"):                                   # X8 world with x0_9's run must equal x0_8's run, bit for bit
        for a in ("B0", "HOMEO", "MARGIN", "REF", "RANDOM"):    # unit-free arms only: SUM and SUMN differ by design (X0.8 feeds them raw level changes)
            m9 = run(s, a, N, "X8", C("lr_default"), C("gamma_default")); m8 = E8.run(s, a, N, C("lr_default"), C("gamma_default"))
            out["parity_vs_x08"][f"{a}@{N}"] = bool(m9 == m8)
    for wd in WORLDS:
        for N in C("wn_ns"):
            out["world"][f"{wd}@{N}"] = {a: run(s, a, N, wd, C("lr_default"), C("gamma_default"))["deaths_per_10k"] for a in ARMS}
    out["rates_equal_U1"] = bool(np.all(rates("U1", 6) == 1.0)); out["rate_x_unit_const_U2"] = bool(np.allclose(rates("U2", 6) * units(6), 1.0))
    out["unit_ratio"] = float(units(6).max() / units(6).min()); out["all_parity"] = all(out["parity_vs_x08"].values())
    return out


def summarize(per):
    S = {}
    for key, arms in per.items():
        g = lambda a, k: np.array([arms[a][s][k] for s in sorted(arms[a])], float); s = {}
        for k in ("deaths_per_10k", "post_swap", "mean_min"): s[k] = {a: float(np.median(g(a, k))) for a in arms}
        causes = {a: np.sum([arms[a][sd]["cause"] for sd in arms[a]], axis=0) for a in arms}
        s["death_share_by_unit_rank"] = {a: (causes[a] / max(1.0, causes[a].sum())).tolist() for a in arms}   # index 0 = smallest unit (= fastest in U2)
        for b in ("SUM", "SUMN", "HOMEO", "B0"): s[f"MARGIN-{b}"] = X.boot_ci(g("MARGIN", "deaths_per_10k") - g(b, "deaths_per_10k"), C("x09_seed0"))
        s["SUM-SUMN"] = X.boot_ci(g("SUM", "deaths_per_10k") - g("SUMN", "deaths_per_10k"), C("x09_seed0"))
        S[key] = s
    for tag in ("tuned", "default"):                       # P261: the same seeds, U2 vs U1
        for N in C("wn_ns"):
            a, b = per.get(f"{tag}|{N}|U2"), per.get(f"{tag}|{N}|U1")
            if a and b:
                d = lambda p: np.array([p["SUM"][s]["deaths_per_10k"] - p["SUMN"][s]["deaths_per_10k"] for s in sorted(p["SUM"])], float)
                S[f"{tag}|{N}|U2-minus-U1 of SUM-SUMN"] = X.boot_ci(d(a) - d(b), C("x09_seed0"))
    return S


def deaths_med(S, key, arm): return S[key]["deaths_per_10k"][arm]


def judge(S, parity_ok):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE tuning or the scored run. Returns [(id, statement, verdict, evidence)]."""
    f = lambda c: f"{c[0]:+.3g} [{c[1]:+.3g}, {c[2]:+.3g}]"; V = []
    ok = parity_ok
    for wd in WORLDS:
        for N in C("wn_ns"):
            k = f"tuned|{N}|{wd}"; ok = ok and deaths_med(S, k, "REF") < 0.5 * deaths_med(S, k, "RANDOM")
    V.append(("M-X0.9", "The worlds are fair tests: the informed reference dies less than half as often as chance in both worlds at every number of needs, the unit-free arms are bit-identical to the earlier X0.8 code, units differ 10x, and in world U2 rate times unit is constant",
              "PASSED" if ok else "FAILED (all other verdicts VOID)", f"REF/RANDOM deaths per 10k at N=6: U1 {deaths_med(S, 'tuned|6|U1', 'REF'):.1f}/{deaths_med(S, 'tuned|6|U1', 'RANDOM'):.1f}, U2 {deaths_med(S, 'tuned|6|U2', 'REF'):.1f}/{deaths_med(S, 'tuned|6|U2', 'RANDOM'):.1f}; parity {parity_ok}"))

    def v(pid, stmt, good, ev): V.append((pid, stmt, ("PASSED" if good else "FAILED") if ok else "VOID", ev))
    u1, u2 = S["tuned|6|U1"], S["tuned|6|U2"]
    v("P259", "When units differ but the needs are equally urgent (world U1), the raw sum breaks: it has more deaths than the sum with units divided out (6 needs, paired difference above 0)", u1["SUM-SUMN"][1] > 0, f"SUM minus SUMN = {f(u1['SUM-SUMN'])} deaths per 10k")
    v("P260", "In that world the weakest-need-first rule, which needs no units, beats the raw sum (6 needs, paired difference below 0)", u1["MARGIN-SUM"][2] < 0, f"MARGIN minus SUM = {f(u1['MARGIN-SUM'])}")
    v("P261", "When the largest unit sits on the slowest need (world U2) the raw sum is hurt more than in U1 (difference of the SUM-minus-SUMN gaps above 0)", S["tuned|6|U2-minus-U1 of SUM-SUMN"][1] > 0, f"(SUM-SUMN in U2) minus (in U1) = {f(S['tuned|6|U2-minus-U1 of SUM-SUMN'])}")
    sh = u2["death_share_by_unit_rank"]["SUM"][0]
    v("P262", "In world U2 the fastest need (smallest unit) causes more than 40% of the raw sum's deaths", sh > 0.40, f"share {sh:.0%}")
    return V


ARM9 = {
    "B0": ("Food only", "Rewarded only for one survival need (the one whose unit is nearest 1); ignores the others."),
    "SUM": ("Raw sum", "Rewarded for the sum of changes in every need, each multiplied by its own reporting unit (units differ 10x)."),
    "SUMN": ("Sum with units divided out", "Rewarded for the plain sum of changes in every need's level: the designer knows and removes every unit."),
    "HOMEO": ("Setpoints", "Rewarded for moving every need toward a designer-chosen target level (0.8)."),
    "MARGIN": ("Weakest need first", "Rewarded only for raising the lowest need's level; needs no units or targets."),
    "REF": ("Informed reference", "Not a learner: eats exactly when that raises the lowest level, knowing the true effects. A reference, not an optimum."),
    "RANDOM": ("Random actions", "Eats or skips at random."),
}
WORLD9 = ("An entity with N survival needs (N = 2, 4 or 6) that all fall every step; it meets one object per step and chooses to eat it or skip it; eating changes each need by an "
          "amount that depends on the object, and some objects are harmful. It dies if any need reaches zero. The value signals differ in how they combine the needs. "
          "Each need is reported in its own UNIT (the units differ by a factor of 10 between the smallest and the largest). World U1: all needs fall at the same speed, so the "
          "units tell nothing about urgency. World U2: the need with the largest unit is the SLOWEST, so a sum in raw units over-weights the least urgent need. "
          "(The earlier experiment X0.8 tied unit to speed, which is why its result could not say whether units or urgency mattered.)")
READ9 = "Deaths are counted per 10,000 steps (lower is better). Differences are 'arm A minus arm B' on the same simulated worlds."


def log_wandb(out):
    from track import Tracker
    S, per = out["summary"], out["per_world"]
    tk = Tracker("X0.9", "Units versus urgency when combining many needs", "When an entity has many needs reported in different units, does a plain sum of changes break, and does it depend on whether units track urgency?",
                 WORLD9, ARM9, READ9, dict(worlds=C("x01_worlds"), needs=list(C("wn_ns")), unit_range=C("x09_unit_range"), tuned=out["tuned"]), tags=["needs", "units", "diagnosis of X0.8"])
    rows = []
    for key in sorted(S):
        if "deaths_per_10k" in S[key]:
            tag, N, wd = key.split("|"); rows.append([tag, int(N), wd] + [round(S[key]["deaths_per_10k"].get(a, float("nan")), 2) for a in ARMS])
    tk.table("Deaths per 10,000 steps, median over worlds (lower is better)", ["Settings", "Needs", "World"] + [tk.name(a) for a in ARMS], rows)
    gaps = [[k, *[round(x, 3) for x in S[k][g]]] for k in sorted(S) if "deaths_per_10k" in S[k] for g in ("SUM-SUMN", "MARGIN-SUM", "MARGIN-SUMN")]
    gaps = [[f"{k} | {g}", *[round(x, 3) for x in S[k][g]]] for k in sorted(S) if "deaths_per_10k" in S[k] for g in ("SUM-SUMN", "MARGIN-SUM", "MARGIN-SUMN")]
    tk.table("Paired differences with 95% ranges (same worlds, arm A minus arm B)", ["Settings | needs | world | comparison", "Median difference", "Lower end", "Upper end"], gaps)
    tk.table("Which need caused the deaths (share of deaths by need, smallest unit first; in world U2 the first entry is the fastest need)", ["Settings | needs | world | arm", "Share by need, smallest unit first"],
             [[f"{k} | {tk.name(a)}", ", ".join(f"{x:.0%}" for x in S[k]["death_share_by_unit_rank"][a])] for k in sorted(S) if "death_share_by_unit_rank" in S[k] and k.startswith("tuned") for a in ARMS])
    seeds = sorted(per["tuned|6|U1"]["SUM"])
    tk.table("Results per world, 6 needs, tuned settings (every number behind the medians)", ["World seed", "World", "Arm", "Deaths per 10,000 steps"],
             [[s, wd, tk.name(a), round(per[f"tuned|6|{wd}"][a][s]["deaths_per_10k"], 2)] for wd in WORLDS for s in seeds for a in ARMS])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode, n):
    X.low_priority(); t0 = time.time()
    if mode == "check": out = check()
    elif mode == "tune": out = tune(n)
    elif mode == "log":
        d = json.load(open(os.path.join(OUT, "x0_9_run.json")))["result"]; log_wandb(d); return
    else:
        tuned = json.load(open(os.path.join(OUT, "x0_9_tune.json")))["result"]; jobs = []
        for wd in WORLDS:
            for N in C("wn_ns"):
                for i in range(C("x01_worlds")):
                    s = C("x09_seed0") + i
                    for arm in ARMS:
                        if arm in LEARN:
                            b = tuned[f"{arm}@{N}@{wd}"]; jobs.append(("tuned", s, arm, N, wd, b["lr"], b["gamma"])); jobs.append(("default", s, arm, N, wd, C("lr_default"), C("gamma_default")))
                        else: jobs.append(("tuned", s, arm, N, wd, 0.0, 0.0))
        per = {}
        for tag, seed, arm, N, wd, lr, g, m in pool_map(jobs, n): per.setdefault(f"{tag}|{N}|{wd}", {}).setdefault(arm, {})[seed] = m
        S = summarize(per); parity = json.load(open(os.path.join(OUT, "x0_9_check.json")))["result"]["all_parity"]
        out = {"tuned": tuned, "per_world": per, "summary": S, "verdicts": judge(S, parity)}
    out = {"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, f"x0_9_{mode}.json"), "w"), indent=1, default=float)
    if mode == "run":
        log_wandb(out["result"])
        for v in out["result"]["verdicts"]: print(v[0], v[2], "|", v[3])


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if n == 0:
        import budget as BU; n = min(BU.workers(per_worker_gb=C("x08_worker_gb")), 14)
    main(sys.argv[1], n)
