"""X0.2 (experiment-plan Phase B): are comfort and growth ONE hunger (the author's theorem U17)?

World W1: a safe HOME and K patches AWAY. Each tick the agent picks home (no gain, no risk) or one patch (gain: the patch's hidden mean
for this season plus noise). Patch values are redrawn every season, so what was learned goes stale and exploring keeps paying. A visible
THREAT comes and goes (two-state Markov chain); being away during a threat costs delta per tick. Three worlds differ in delta: mild
(staying out still pays), medium, severe (home is the only sensible place). The world stream (patch values, threats) is the same for
every arm; each arm has its own noise and sampling stream.

Agents share ONE estimator (a forgetting mean and standard deviation per patch) and know the threat penalty (an innate danger signal);
they differ only in how the decision is formed:
  ONE    one quantity, expected change for me: a SAMPLE of each patch's value (Thompson: uncertainty drives exploring) minus the threat
         penalty; home is 0; take the best. No weight, no switch.
  TWOF   two drives, no switch: mean + w x uncertainty (a curiosity bonus) minus the threat penalty; w tuned on the medium world, then frozen.
  TWOS   two drives with a hand-built switch: the curiosity bonus is turned OFF while a threat is on; w tuned on the medium world, then frozen.
  B0     the plain baseline: greedy on the mean minus the threat penalty, with a random action at rate eps (tuned on the medium world, frozen).
  ORACLE knows this season's true patch values and the threat (ceiling).   RANDOM  random actions.
Metrics (after a warm-up; noise-free): regret per tick = oracle's expected net gain - the chosen action's expected net gain;
calm_away / threat_away = share of calm / threat ticks spent away; explore_calm / explore_threat = share of calm / threat ticks spent visiting a
patch other than the one currently believed best (an exploratory visit); resume = ticks from the end of a threat to the first move away.
The arms tuned on the medium world are also tuned per world ('TWOS*', 'TWOF*', 'B0*') to show what retuning would buy them.
Usage: python x0_2.py check | tune | run [--workers N]     (writes ../results/x0_2_<mode>.json)"""
import sys, os, json, time
import multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x0_1 as X

ARMS = ("ONE", "TWOF", "TWOS", "B0", "ORACLE", "RANDOM")
OUT = X.OUT
IDX = {a: i for i, a in enumerate(ARMS)}


def make_world(seed, delta):
    T, L, K = C("x02_ticks"), C("w1_season_len"), C("w1_patches")
    rw = np.random.default_rng(seed)
    mu = rw.uniform(C("w1_mu_lo"), C("w1_mu_hi"), (T // L + 1, K))
    u = rw.random(T); th = np.zeros(T, bool); s = False
    for t in range(T):
        s = bool(u[t] < C("w1_p_on")) if not s else bool(u[t] >= C("w1_p_off"))
        th[t] = s
    return mu, th


def run(seed, arm, delta, w=0.0, eps=0.0):
    T, L, K, W = C("x02_ticks"), C("w1_season_len"), C("w1_patches"), C("x02_warmup")
    mu, th = make_world(seed, delta)
    ra = np.random.default_rng(seed + C("x02_stream_stride") * (IDX[arm] + 1))
    s = np.zeros(K); n = np.zeros(K); r = C("w0_respawn"); deaths = 0
    away = np.zeros(T, bool); eg = np.zeros(T); expl = np.zeros(T, bool)
    lam, n0, sa = C("ag_lambda"), C("ag_n0"), C("ag_sigma")
    for t in range(T):
        tt = bool(th[t]); sea = t // L; pen = delta * tt
        n *= lam; s *= lam
        m = s / (n + n0); sd = sa / np.sqrt(n + n0)
        gk = int(np.argmax(mu[sea] if arm == "ORACLE" else m - pen))     # the patch believed best right now (the oracle: truly best)
        if arm == "RANDOM":
            a = int(ra.integers(K + 1))
        elif arm == "B0" and ra.random() < eps:
            a = int(ra.integers(K + 1))
        else:
            if arm == "ONE": v = ra.normal(m, sd) - pen
            elif arm == "TWOF": v = m + w * sd - pen
            elif arm == "TWOS": v = m + w * sd * (0.0 if tt else 1.0) - pen
            elif arm == "B0": v = m - pen
            else: v = mu[sea] - pen
            k = int(np.argmax(v)); a = k + 1 if v[k] > 0 else 0
        if a > 0:
            k = a - 1; y = mu[sea, k] + ra.normal(0, C("w1_noise")) - pen
            n[k] += 1; s[k] += y + pen; r += y; away[t] = True; eg[t] = mu[sea, k] - pen; expl[t] = k != gk
        r -= C("w0_energy_decay"); r = min(r, 1.0)
        if r <= 0: deaths += 1; r = C("w0_respawn")
    best = mu[np.arange(T) // L].max(1); og = np.maximum(0.0, best - delta * th)
    sl = slice(W, T); thw = th[sl]; aw = away[sl]; ex = expl[sl]
    ends = [t for t in range(max(W, 1), T) if th[t - 1] and not th[t]]
    lags = []
    for e in ends:
        nxt = np.nonzero(away[e:e + C("x02_resume_cap")])[0]
        lags.append(int(nxt[0]) if len(nxt) else C("x02_resume_cap"))
    return dict(regret=float((og[sl] - eg[sl]).mean()), calm_away=float(aw[~thw].mean()), explore_calm=float(ex[~thw].mean()), explore_threat=float(ex[thw].mean()) if thw.any() else float("nan"), threat_away=float(aw[thw].mean()) if thw.any() else float("nan"),
                resume=float(np.median(lags)) if lags else float("nan"), deaths=deaths * C("per_10k") / T, threat_share=float(thw.mean()))


def job(args):
    seed, arm, delta, w, eps, tag = args
    return tag, seed, delta, run(seed, arm, delta, w, eps)


def pool_map(jobs, n):
    with mp.get_context("spawn").Pool(n) as p:
        return p.map(job, jobs, chunksize=4)


def tune(n):
    deltas = C("w1_deltas"); res = {}
    jobs = []
    for d in deltas:
        for i in range(C("x01_tune_worlds")):
            s = C("x02_tune_seed0") + i
            for w in C("w_grid"):
                jobs += [(s, "TWOF", d, w, 0.0, f"TWOF|{w}"), (s, "TWOS", d, w, 0.0, f"TWOS|{w}")]
            for e in C("eps_grid"):
                jobs.append((s, "B0", d, 0.0, e, f"B0|{e}"))
    out = {}
    for tag, seed, d, m in pool_map(jobs, n):
        out.setdefault((d, tag), []).append(m["regret"])
    best = {}
    for d in deltas:
        for arm in ("TWOF", "TWOS", "B0"):
            cand = {tag: float(np.mean(v)) for (dd, tag), v in out.items() if dd == d and tag.startswith(arm + "|")}
            tag = min(cand, key=cand.get); best[f"{arm}@{d}"] = dict(param=float(tag.split("|")[1]), regret=cand[tag], all=cand)
    return best


def main(mode, n):
    X.low_priority(); t0 = time.time()
    if mode == "check":
        out = {}
        for d in C("w1_deltas"):
            out[str(d)] = {a: run(C("x02_check_seed"), a, d, 1.0, 0.1) for a in ARMS}
    elif mode == "tune":
        out = tune(n)
    else:
        tuned = json.load(open(os.path.join(OUT, "x0_2_tune.json")))["result"]
        mid = C("x02_tune_delta"); deltas = C("w1_deltas"); jobs = []
        for d in deltas:
            for i in range(C("x01_worlds")):
                s = C("x02_seed0") + i
                for a in ("ONE", "ORACLE", "RANDOM"): jobs.append((s, a, d, 0.0, 0.0, a))
                for a, key in (("TWOF", "TWOF"), ("TWOS", "TWOS"), ("B0", "B0")):
                    p = tuned[f"{key}@{mid}"]["param"]; pw = tuned[f"{key}@{d}"]["param"]
                    jobs.append((s, a, d, p if a != "B0" else 0.0, p if a == "B0" else 0.0, a))
                    jobs.append((s, a, d, pw if a != "B0" else 0.0, pw if a == "B0" else 0.0, a + "*"))
        per = {}
        for tag, seed, d, m in pool_map(jobs, n):
            per.setdefault(str(d), {}).setdefault(tag, {})[seed] = m
        out = {"tuned": tuned, "per_world": per, "summary": summarize(per)}
    out = {"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, f"x0_2_{mode}.json"), "w"), indent=1, default=float)
    print(json.dumps(out["result"] if mode != "run" else out["result"]["summary"], indent=1, default=float))


def summarize(per):
    S = {}
    for d, arms in per.items():
        g = lambda a, k: np.array([arms[a][s][k] for s in sorted(arms[a])], float)
        s = {}
        for k in ("regret", "calm_away", "threat_away", "explore_calm", "explore_threat", "resume", "deaths", "threat_share"):
            s[k] = {a: float(np.nanmedian(g(a, k))) for a in arms}
        for k in ("regret",):
            for b in ("TWOS", "TWOF", "B0", "TWOS*", "TWOF*", "B0*", "RANDOM"):
                dd = g("ONE", k) - g(b, k); s[f"ONE-{b}"] = X.boot_ci(dd, C("x02_seed0"))
            for b in ("TWOS", "TWOF", "B0"):                      # what retuning for the world buys the hand-weighted arms
                dd = g(b, k) - g(b + "*", k); s[f"{b}-{b}*"] = X.boot_ci(dd, C("x02_seed0"))
        S[d] = s
    return S


if __name__ == "__main__":
    ap_n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if ap_n == 0:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import budget as BU
        ap_n = min(BU.workers(per_worker_gb=C("x02_worker_gb")), 6)
    main(sys.argv[1], ap_n)
