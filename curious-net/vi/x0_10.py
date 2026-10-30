"""X0.10 (the author's request, 2026-10-28; raw/2026-10-28-hidden-need-depth-test.md): a HIDDEN need that only non-survival depth can satisfy.

World WH: N_s = 2 visible survival variables (W0 energy dynamics, as X0.9 world U1) plus ONE hidden essential variable H in [0, 1] that decays by
x10_dH per tick and KILLS at 0 like the others. H is refilled only by TASKS, never by food. Task level k (k = 0 .. K-1) is a ladder of depth:
  gain to H per task tick = G_k x c_k,  G_k = x10_g0 x x10_gm^k  (deeper = more),  c_k = competence in [0, 1]
  c_0 = 1 always (walking in a circle: tiny gain);  practising level k raises c_k by eta_k (1 - c_k) c_{k-1}, eta_k = x10_eta0 x x10_etam^(k-1)
  (a level only progresses once the one below is mastered; deeper levels are slower to master). Competence is the entity's skill: it survives a death.
Each tick one object arrives and the agent picks ONE action: skip, eat the object, or do task k. A task tick therefore costs a foraging tick.
The learner sees: the object, its two visible levels, and BOREDOM b = 1 - H (a rising warning as H falls). It never sees H as a reward, the task
gains, the competences, or which action refills what. Value signals (one TD learner, differing only in the signal):
  ALL  sum of the changes of every need's level, the hidden one through boredom (-delta b): the unit-free 'every need counts' sum
  IGN  the same without the boredom term (the entity ignores its warning)
  REF  NOT learned: an informed reference that knows the ladder; REF0 the same restricted to level 0 (the control showing depth is REQUIRED)
  RANDOM, ALL0 (ALL restricted to skip / eat / level 0: the circle-only learner)
Usage: python x0_10.py check | explore [N workers]   (writes ../results/x0_10_<mode>.json)"""
import sys, os, json, time
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x0_1 as X
import x0_9 as E9
from x0_8 import drive
from learner import ValueNet

OUT = X.OUT
ARMS = ("ALL", "IGN", "ALL0", "REF", "REF0", "RANDOM")
LEARN = ("ALL", "IGN", "ALL0")
WORLDS = ("WH", "WHF")


class ValueNetK(ValueNet):
    """The shared learner with n_act actions (learner.py is fixed at 2 and left untouched); identical update rule."""
    def __init__(self, n_in, n_act, seed, lr):
        super().__init__(n_in, seed, lr); self.n = n_act
        rng = np.random.default_rng(seed + 7); H = self.p["W1"].shape[1]
        self.p["W2"] = rng.standard_normal((H, n_act)) * C("init_scale") / np.sqrt(H); self.p["b2"] = np.zeros(n_act)
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}; self.v = {k: np.zeros_like(v) for k, v in self.p.items()}

    def act_among(self, z, allowed):
        if self.rng.random() < C("epsilon"): return int(allowed[int(self.rng.integers(len(allowed)))])
        q = self.q(z)[0]; return int(allowed[int(np.argmax(q[allowed]))])

    def learn(self, z, a, r):
        out, h = self.q(z)
        g_out = np.zeros(self.n); g_out[a] = out[a] - r
        g = {"W2": np.outer(h, g_out), "b2": g_out}
        g_h = (self.p["W2"] @ g_out) * (1 - h * h)
        g["W1"], g["b1"] = np.outer(z, g_h), g_h
        self.t += 1
        b1, b2, eps = C("adam_b1"), C("adam_b2"), C("adam_eps")
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]; self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            self.p[k] -= self.lr * (self.m[k] / (1 - b1 ** self.t)) / (np.sqrt(self.v[k] / (1 - b2 ** self.t)) + eps)


def ladder():
    K = C("x10_K"); k = np.arange(K)
    G = C("x10_g0") * C("x10_gm") ** k
    eta = np.r_[0.0, C("x10_eta0") * C("x10_etam") ** (k[1:] - 1)]
    return G, eta


def ref_action(arm, h, H, c, eff, d, G, K):
    nS = len(h)
    prac = 0 if arm == "REF0" else max(k for k in range(K) if k == 0 or c[k - 1] >= C("x10_unlock"))   # deepest level whose prerequisite is mastered
    forage = int(np.min(h + eff - d) > np.min(h - d))
    if np.min(h) < C("x10_ref_food") or H >= C("x10_ref_task"): return forage                        # 0 = skip, 1 = eat
    return 2 + prac


def run(seed, arm, lr=0.0, gamma=0.0, ticks=None, world="WH", trace=False):
    ticks = ticks or C("x10_ticks"); K = C("x10_K"); nS = C("x10_ns"); G, eta = ladder()
    w = E9.WN9(seed, nS, "U1"); n_act = 2 + K; n_in = C("wn_feat_dim") + 1 + nS + 1
    net = ValueNetK(n_in, n_act, 2 * seed + 1, lr) if arm in LEARN else None
    allowed = list(range(n_act)) if arm != "ALL0" else [0, 1, 2]
    ra = np.random.default_rng(2 * seed + 3); h = np.full(nS, C("w0_respawn")); H = C("w0_respawn"); c = np.zeros(K); c[0] = 1.0
    pend = None; deaths, cause = [], np.zeros(nS + 1); act = np.zeros(n_act); sumH = 0.0; third = ticks * 2 // 3; sumH_late = 0.0; first = {}; csum = np.zeros(K); tr = []
    fade = C("x10_fade") * eta if world == "WHF" else np.zeros(K)                   # WHF: an unpractised skill fades by fade_k x c_k per tick
    for t in range(ticks):
        obs, eff = w.next_object(); z = np.r_[obs, h, 1.0 - H]
        if pend is not None and net is not None:
            zp, ap, rp, term = pend; net.learn(zp, ap, rp if term else rp + gamma * float(np.max(net.q(z)[0][allowed])))
        if arm == "RANDOM": a = int(ra.integers(n_act))
        elif arm in ("REF", "REF0"): a = ref_action(arm, h, H, c, eff, w.d, G, K)
        else: a = net.act_among(z, allowed)
        act[a] += 1; gain = 0.0; kp = a - 2 if a >= 2 else -1
        c[1:] -= np.where(np.arange(1, K) == kp, 0.0, fade[1:] * c[1:])
        if a >= 2:
            k = a - 2; gain = G[k] * c[k]
            if k > 0: c[k] += eta[k] * (1 - c[k]) * c[k - 1]
        after = np.clip(h + (eff if a == 1 else 0) - w.d, 0, 1); Hn = float(np.clip(H + gain - C("x10_dH"), 0, 1))
        died = bool((after <= 0).any() or Hn <= 0); pre, Hpre = after.copy(), Hn
        if died:
            cause[int(np.argmin(after)) if (after <= 0).any() else nS] += 1; deaths.append(t)
            after = np.full(nS, C("w0_respawn")); Hn = C("w0_respawn")
        if arm in LEARN:
            r = float((pre - h).sum()) + (0.0 if arm == "IGN" else float(Hpre - H))                  # change in b is minus change in H
            pend = (z, a, r, died)
        sumH += Hpre
        if t >= third: sumH_late += Hpre; csum += c
        for m in range(1, K):
            if m not in first and c[m] >= C("x10_master"): first[m] = t
        h, H = after, Hn
        if trace and t % C("x10_trace_every") == 0: tr.append((t, float(H), float(h.min()), c.tolist()))
    share = act / ticks
    return dict(trace=tr, deaths_per_10k=len(deaths) * C("per_10k") / ticks, cause=cause.tolist(), mean_H=sumH / ticks, mean_H_late=sumH_late / (ticks - third),
                comp=c.tolist(), comp_late=(csum / (ticks - third)).tolist(), first_master={str(k): v for k, v in first.items()}, action_share=share.tolist(),
                top_level=int(max([k for k in range(K) if c[k] >= C("x10_master")] or [0])), depth_late=int(max([k for k in range(K) if csum[k] / (ticks - third) >= C("x10_hold")] or [0])))


def job(a):
    tag, seed, arm, lr, g, world = a
    return tag, seed, arm, lr, g, world, run(seed, arm, lr, g, world=world)


def pool_map(jobs, n):
    with mp.get_context("spawn").Pool(n) as p:
        return p.map(job, jobs, chunksize=2)


def tune(n):
    """Each learned arm's learning rate and discount are tuned on ITS OWN objective (fewest deaths of any cause), on seeds apart from the scored ones."""
    jobs = [("tune", C("x10_tune_seed0") + i, arm, lr, g, "WH") for arm in LEARN for lr in C("x08_lr_grid") for g in C("x08_gamma_grid") for i in range(C("x08_tune_worlds"))]
    res = {}
    for _, seed, arm, lr, g, w, m in pool_map(jobs, n): res.setdefault((arm, lr, g), []).append(-m["deaths_per_10k"])
    best = {}
    for arm in LEARN:
        cand = {k: float(np.median(v)) for k, v in res.items() if k[0] == arm}; k = max(cand, key=cand.get); best[arm] = dict(lr=k[1], gamma=k[2], score=cand[k])
    return best


def hidden_deaths(m): return m["cause"][-1] * C("per_10k") / C("x10_ticks")


METRICS = {   # key: (plain name with unit and direction, function of one run)
    "deaths": ("Deaths per 10,000 steps, any cause (lower is better)", lambda m: m["deaths_per_10k"]),
    "hid_deaths": ("Deaths from the hidden need per 10,000 steps (lower is better)", hidden_deaths),
    "H_late": ("How full the hidden need was kept in the last third of life, 0 to 1 (higher is better)", lambda m: m["mean_H_late"]),
    "top": ("Deepest skill level mastered (0 = only walking in a circle; higher is deeper)", lambda m: m["top_level"]),
    "t3": ("Steps until skill level 3 was mastered (lower is faster; blank = never)", lambda m: m["first_master"].get("3", float("nan"))),
}


def summarize(per):
    """per[arm][seed] = metrics. Returns medians per arm and paired 95% bootstrap ranges of the differences the predictions name."""
    seeds = sorted(per["ALL"]); S = {"median": {}, "paired": {}, "frac": {}}
    g = lambda arm, key: np.array([METRICS[key][1](per[arm][s]) for s in seeds], float)
    for arm in per: S["median"][arm] = {k: float(np.nanmedian(g(arm, k))) for k in METRICS}
    pair = lambda a, b, key: X.boot_ci(g(a, key) - g(b, key), C("x10_seed0"))
    S["paired"]["H_late ALL-IGN"] = pair("ALL", "IGN", "H_late"); S["paired"]["deaths ALL-ALL0"] = pair("ALL", "ALL0", "deaths")
    S["paired"]["deaths ALL-RANDOM"] = pair("ALL", "RANDOM", "deaths"); S["paired"]["hid_deaths ALL-IGN"] = pair("ALL", "IGN", "hid_deaths")
    S["paired"]["deaths ALL-REF"] = pair("ALL", "REF", "deaths")
    both = np.isfinite(g("ALL", "t3")) & np.isfinite(g("IGN", "t3")); S["n_both_reach_3"] = int(both.sum())
    S["paired"]["t3 ALL-IGN"] = X.boot_ci((g("ALL", "t3") - g("IGN", "t3"))[both], C("x10_seed0")) if both.sum() > 3 else [float("nan")] * 3
    S["frac"]["ALL top>=3"] = float(np.mean(g("ALL", "top") >= 3)); S["frac"]["REF0 dies of hidden need"] = float(np.mean(g("REF0", "hid_deaths") > 0))
    S["frac"]["REF dies of hidden need"] = float(np.mean(g("REF", "hid_deaths") > 0))
    return S


def judge(S):
    """The verdict rules, fixed in code and registered in PREDICTIONS.md BEFORE the scored run. Returns [(id, statement, verdict, evidence)]."""
    m, pr, fr = S["median"], S["paired"], S["frac"]; ci = lambda k: pr[k]; fmt = lambda k: f"{pr[k][0]:+.3g} [{pr[k][1]:+.3g}, {pr[k][2]:+.3g}]"
    V = []
    manip = (fr["REF0 dies of hidden need"] >= 0.9) and (fr["REF dies of hidden need"] <= 0.1) and (m["REF"]["deaths"] <= 0.5 * m["RANDOM"]["deaths"])
    V.append(("M-X0.10", "The world is a fair test: an informed entity that only walks in circles dies of the hidden need in at least 90% of worlds, an informed entity that learns the skill ladder almost never does, and it survives at least twice as well as chance",
              "PASSED" if manip else "FAILED (all other verdicts VOID)", f"circles-only informed dies of hidden need in {fr['REF0 dies of hidden need']:.0%} of worlds; skill-ladder informed in {fr['REF dies of hidden need']:.0%}; deaths per 10k {m['REF']['deaths']:.1f} vs chance {m['RANDOM']['deaths']:.1f}"))

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P263", "Listening to boredom fills the hidden need better than ignoring it (hidden-need level in the last third of life, paired difference above 0)", ci("H_late ALL-IGN")[1] > 0, f"ALL minus IGN = {fmt('H_late ALL-IGN')}")
    v("P264", "The entity that listens to boredom survives better than the one that can only walk in circles (deaths per 10k, paired difference below 0)", ci("deaths ALL-ALL0")[2] < 0, f"ALL minus ALL0 = {fmt('deaths ALL-ALL0')}")
    v("P265", "The entity that listens to boredom keeps the hidden need at least half full in the last third of life (median)", m["ALL"]["H_late"] >= 0.5, f"median {m['ALL']['H_late']:.2f}")
    v("P266", "It finds depth: in at least 90% of worlds it masters skill level 3 or deeper", fr["ALL top>=3"] >= 0.9, f"{fr['ALL top>=3']:.0%} of worlds")
    v("P267", "It finds depth sooner than the entity that ignores boredom (steps until level 3 is mastered, worlds where both get there, paired difference below 0)", ci("t3 ALL-IGN")[2] < 0, f"ALL minus IGN = {fmt('t3 ALL-IGN')} steps in {S['n_both_reach_3']} worlds")
    v("P268", "It is still worse than the informed reference (deaths per 10k, paired difference above 0): learning from boredom alone does not fully match knowing the ladder", ci("deaths ALL-REF")[1] > 0, f"ALL minus REF = {fmt('deaths ALL-REF')}")
    v("P269", "It dies of the hidden need less often than the entity that ignores boredom (paired difference below 0)", ci("hid_deaths ALL-IGN")[2] < 0, f"ALL minus IGN = {fmt('hid_deaths ALL-IGN')}")
    v("P270", "The P263 result is not an artefact of tuning: with the learners' DEFAULT settings the hidden-level difference ALL - IGN is also above 0", ci("H_late ALL-IGN default settings")[1] > 0, f"ALL minus IGN (default settings) = {fmt('H_late ALL-IGN default settings')}")
    return V


ARM_TEXT = {
    "ALL": ("Listens to boredom", "A learning entity whose reward is the change in ALL its needs, the hidden one through its boredom warning (boredom going down counts as good)."),
    "IGN": ("Ignores boredom", "The same learner with the same inputs, but its reward counts only the survival needs it can see; boredom is shown to it but never rewarded."),
    "ALL0": ("Listens, but only circles", "Same as 'Listens to boredom' but it is only allowed to eat, skip or walk in a circle: shows whether depth is really needed."),
    "REF": ("Informed reference", "Not a learner: a hand-written policy that knows the skill ladder and practises the deepest unlocked level when bored. A reference point, not an optimum."),
    "REF0": ("Informed, circles only", "The informed policy restricted to walking in a circle: the control that proves the hidden need cannot be filled without depth."),
    "RANDOM": ("Random actions", "Picks one of the 7 actions uniformly at random each step."),
}
WORLD_TEXT = ("The entity lives in a small simulated world for 15,000 steps. It has two survival needs you can see (like energy and health); they fall every step and "
              "are refilled by eating objects that drift past, but some objects are harmful. It also has ONE HIDDEN need that it can never see. The hidden need falls slowly "
              "and the entity dies if it reaches zero, exactly like the others. Food never refills it. Only practising TASKS refills it: a ladder of five tasks, from walking in "
              "a circle (almost no refill) to deep tasks that refill a lot, but deeper tasks take longer to learn and each one only starts to improve once the one below is mastered. "
              "Each step the entity does ONE thing (skip, eat the object, or practise a task), so practising costs foraging time. The only thing the entity feels about the hidden "
              "need is BOREDOM, a warning that rises as the hidden need falls. Nothing tells it which action fills it.")
READ_TEXT = ("'Deaths' count how often the entity died and restarted (lower is better). 'Hidden-need level' is how full the secret need was kept (1 = full, 0 = dead; higher is "
             "better). 'Skill level' counts the rungs of the ladder mastered. The informed reference is a ceiling for comparison, not a competitor.")


def log_wandb(out, tuned):
    from track import Tracker
    cfg = dict(question="Can an entity that only feels boredom find the deep activity that keeps a hidden need alive, while also staying alive?", seeds_scored=f"{C('x10_seed0')}..{C('x10_seed0') + C('x01_worlds') - 1}", worlds=C("x01_worlds"),
               steps_per_life=C("x10_ticks"), task_levels=C("x10_K"), tuned=tuned)
    tk = Tracker("X0.10", "Hidden need that only depth can satisfy", cfg["question"], WORLD_TEXT, ARM_TEXT, READ_TEXT, cfg, tags=["hidden need", "boredom", "depth"])
    S = out["summary"]; per = out["per_world"]
    tk.table("Summary by arm (median over worlds)", ["Arm"] + [METRICS[k][0] for k in METRICS],
             [[tk.name(a)] + [round(S["median"][a][k], 3) for k in METRICS] for a in ARMS])
    seeds = sorted(per["ALL"])
    tk.table("Results per world (every number behind the medians)", ["World seed", "Arm"] + [METRICS[k][0] for k in METRICS],
             [[s, tk.name(a)] + [round(float(METRICS[k][1](per[a][s])), 3) for k in METRICS] for s in seeds for a in ARMS])
    tk.table("Paired differences with 95% ranges (same worlds, arm A minus arm B)", ["Comparison", "Median difference", "Lower end", "Upper end"],
             [[k, *[round(x, 3) for x in v]] for k, v in S["paired"].items()])
    tk.table("Same learners with their default settings (median over worlds)", ["Arm"] + [METRICS[k][0] for k in METRICS],
             [[tk.name(a)] + [round(S["median_default"][a][k], 3) for k in METRICS] for a in S["median_default"]])
    tk.verdicts(out["verdicts"])
    sh = out["showcase"]                                   # showcase: one world's life, step by step; ALL arms logged together at each step
    for i in range(len(sh[ARMS[0]])):
        row = {}
        for arm in ARMS:
            t_, H, mh, c = sh[arm][i]
            row.update({f"Life story | Hidden need level, 1 = full, 0 = dead | {tk.name(arm)}": H, f"Life story | Boredom warning felt, higher = hungrier | {tk.name(arm)}": 1 - H,
                        f"Life story | Lowest survival need level | {tk.name(arm)}": mh,
                        f"Life story | Deepest skill rung at least half learned | {tk.name(arm)}": max([k for k in range(len(c)) if c[k] >= 0.5] or [0])})
        tk.log(row, step=int(sh[ARMS[0]][i][0]))
    tk.summary(**{f"Median {METRICS[k][0]} | {tk.name(a)}": S["median"][a][k] for a in ARMS for k in ("deaths", "H_late")})
    tk.finish()


def main(mode, n):
    X.low_priority(); t0 = time.time()
    if mode == "check":
        s = C("x10_check_seed"); out = {w: {a: run(s, a, C("lr_default"), C("gamma_default"), world=w) for a in ARMS} for w in WORLDS}
    elif mode == "explore":                                # a few non-scored worlds, default settings
        jobs = [("explore", C("x10_check_seed") + i, a, C("lr_default"), C("gamma_default"), w) for w in WORLDS for i in range(C("x10_explore_worlds")) for a in ARMS]
        out = {}
        for tag, seed, a, lr, g, w, m in pool_map(jobs, n): out.setdefault(w, {}).setdefault(a, {})[seed] = m
    elif mode == "tune":
        out = tune(n)
    elif mode == "log":                                    # re-log the saved scored run to W&B (no recomputation)
        d = json.load(open(os.path.join(OUT, "x0_10_run.json")))["result"]; log_wandb(d, d["tuned"]); return
    else:                                                  # run: the scored experiment
        tuned = json.load(open(os.path.join(OUT, "x0_10_tune.json")))["result"]; jobs = []
        for i in range(C("x01_worlds")):
            s = C("x10_seed0") + i
            for a in ARMS:
                b = tuned.get(a, dict(lr=0.0, gamma=0.0)); jobs.append(("tuned", s, a, b["lr"], b["gamma"], "WH"))
                if a in LEARN: jobs.append(("default", s, a, C("lr_default"), C("gamma_default"), "WH"))
        per, per_def = {}, {}
        for tag, seed, a, lr, g, w, m in pool_map(jobs, n): m.pop("trace", None); (per if tag == "tuned" else per_def).setdefault(a, {})[seed] = m
        S = summarize(per); show = {}
        sd = sorted(per_def["ALL"]); S["paired"]["H_late ALL-IGN default settings"] = X.boot_ci(
            np.array([per_def["ALL"][s]["mean_H_late"] - per_def["IGN"][s]["mean_H_late"] for s in sd]), C("x10_seed0"))
        S["median_default"] = {a: {k: float(np.nanmedian([METRICS[k][1](per_def[a][s]) for s in sd])) for k in METRICS} for a in per_def}
        for a in ARMS:
            b = tuned.get(a, dict(lr=0.0, gamma=0.0)); show[a] = run(C("x10_seed0"), a, b["lr"], b["gamma"], world="WH", trace=True)["trace"]
        out = {"tuned": tuned, "per_world": per, "per_world_default": per_def, "summary": S, "verdicts": judge(S), "showcase": show}
        log_wandb(out, tuned)
    out = {"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, f"x0_10_{mode}.json"), "w"), indent=1, default=float)
    if mode == "run":
        for v in out["result"]["verdicts"]: print(v[0], v[2], "|", v[3])


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(sys.argv[1], n)
