"""X1.3 (the author's interest, 2026-10-28; raw/2026-10-28-prediction-only-life.md): an entity whose whole existence is prediction and internal models.
It has no food, goal or external reward. It dies if its predictions fail, and it gets bored (and dies of it) if its models are not improving.

World PL. Regions the entity can sit in each tick (x1_1's world, extended with a ladder and with death):
  DARK    always the same symbol: perfectly predictable, nothing to learn
  NOISE   a uniformly random symbol: unlearnable
  L1..L3  deterministic cycles over S = 8, 32, 128 symbols (learning a table of S entries takes about S visits); every x13_season ticks ALL their rules are re-drawn
          (what was learned goes stale and can be learned again)
Each visit the entity predicts the next symbol with its own per-region table, then learns the truth. Two essential variables in [0, 1], both start and respawn at 0.5:
  V (vitality)       += x13_alpha_v x accuracy - x13_delta_v        dies at 0 (failed predictions kill: noise cannot sustain it)
  H (hidden need)    += x13_kappa x (bits of real structure newly learned) - x13_delta_h   dies at 0
  'bits newly learned' = log2(S) for each table entry that was wrong and becomes right: deeper structure refills more. Nothing else refills H; the dark room gives zero.
The entity feels H only as BOREDOM b = 1 - H. A death resets V and H; the entity's tables and its region values are kept (knowledge is not erased).
Arms (one region-choosing bandit, same value estimate and epsilon-greedy for all, as in X1.1; arms differ ONLY in the reward):
  ALL      reward = change in V + change in H (every need counts; the hidden one is felt through boredom): the prediction-only entity
  IGN      reward = change in V only (ignores boredom)
  UR       reward = - prediction error (the program's original drive: surprise minimisation)
  LP       reward = learning progress of the visited region (slow - fast error average), as X1.1;  LPO the same with an optimistic start (X1.2)
  ORACLE   informed policy: rests in DARK when V is low, otherwise learns the deepest unmastered region while H is below x13_ref_h (a reference, not an optimum)
  RANDOM, DARKONLY, NOISEONLY (fixed-policy controls)
Usage: python x1_3.py check | run          (writes ../results/x1_3_<mode>.json)"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x0_1 as X
import x1_1 as M

ARMS = ("ALL", "IGN", "UR", "LP", "LPO", "ORACLE", "RANDOM", "DARKONLY", "NOISEONLY")
LEARN = ("ALL", "IGN", "UR", "LP", "LPO")
EXTRA_LEARN = ("BITS", "ALLN", "LPT", "LPOT")                  # X1.4 arms (BITS: exact bits learned; ALLN: noisy/lagged boredom; LPT/LPOT: learning progress with tuned windows)
OUT = X.OUT


def run(seed, arm, ticks=None, stream=None, sigma=0.0, lag=1, lp_f=None, lp_s=None):
    T = ticks or C("x13_ticks"); sizes = list(C("x13_sizes")); nd, nn = C("x13_dark_rooms"), C("x13_noise_rooms"); R = nd + nn + len(sizes); L0 = nd + nn
    SZ = np.array([C("x13_dark_S")] * nd + [C("x13_noise_S")] * nn + sizes); season_len = C("x13_season")
    rw = np.random.default_rng(seed); rn = np.random.default_rng(seed + 1)
    noise = rn.integers(0, C("x13_noise_S"), (nn, T)); rules = [[M.cycle(rw, s) for s in sizes] for _ in range(T // season_len + 1)]   # world: identical for every arm in a seed
    ra = np.random.default_rng(seed + C("x02_stream_stride") * ((ARMS.index(arm) if stream is None else stream) + 1))
    table = [ra.integers(0, SZ[r], SZ[r]) for r in range(R)]; prev = np.zeros(R, int); v = np.zeros(R)
    fast = 1 - 1 / SZ.astype(float); slow = fast.copy()
    if arm in ("LPO", "LPOT"): v[:] = C("opt_value")
    learned = arm in LEARN or arm in EXTRA_LEARN; buf = [C("w0_respawn")] if arm == "ALLN" else None; lf, ls = lp_f or C("lp_fast"), lp_s or C("lp_slow")
    V = H = C("w0_respawn"); resp = C("w0_respawn")
    deaths_V = deaths_H = 0; visits = np.zeros(R); sumH = sumV = 0.0; half = T // 2; bits = 0.0
    for t in range(T):
        f = rules[t // season_len]
        if arm == "DARKONLY": r = 0
        elif arm == "NOISEONLY": r = nd
        elif arm == "RANDOM" or ra.random() < C("d_eps"): r = int(ra.integers(R))
        elif arm == "ORACLE":
            un = [k for k in range(len(sizes)) if float((table[L0 + k] == f[k]).mean()) < C("d_acc")]
            r = 0 if (V < C("x13_ref_v") or not un or H >= C("x13_ref_h")) else L0 + max(un)
        else: r = int(np.argmax(v + 1e-9 * ra.random(R)))
        nxt = 0 if r < nd else int(noise[r - nd, t]) if r < L0 else int(f[r - L0][prev[r]])
        err = float(table[r][prev[r]] != nxt); gain = float(np.log2(SZ[r])) if (r >= L0 and err == 1.0) else 0.0
        table[r][prev[r]] = nxt; prev[r] = nxt; bits += gain
        fast[r] += lf * (err - fast[r]); slow[r] += ls * (err - slow[r])
        V1 = min(1.0, V + C("x13_alpha_v") * (1 - err) - C("x13_delta_v")); H1 = min(1.0, H + C("x13_kappa") * gain - C("x13_delta_h"))
        if learned:
            if arm == "ALLN":                                   # boredom read through noise and a lag: reading = mean of the last `lag` noisy samples of H
                prev_read = float(np.mean(buf[-lag:])); buf.append(H1 + (sigma * float(ra.normal()) if sigma > 0 else 0.0)); read = float(np.mean(buf[-lag:])); buf = buf[-lag - 1:]
                rew = (V1 - V) + (read - prev_read)
            else: rew = {"ALL": (V1 - V) + (H1 - H), "IGN": V1 - V, "UR": -err, "BITS": C("x13_kappa") * gain}.get(arm, slow[r] - fast[r])
            v[r] += C("d_alpha") * (rew - v[r])
        visits[r] += 1
        if t >= half: sumH += H1; sumV += V1
        if V1 <= 0 or H1 <= 0:
            if V1 <= 0: deaths_V += 1
            else: deaths_H += 1
            V = H = resp
            if buf is not None: buf = [resp]
        else: V, H = V1, H1
    k = C("per_10k") / T
    return dict(deaths_per_10k=(deaths_V + deaths_H) * k, deaths_V=deaths_V * k, deaths_H=deaths_H * k, mean_H_late=sumH / (T - half), mean_V_late=sumV / (T - half),
                share=([visits[:nd].sum() / T, visits[nd:L0].sum() / T] + (visits[L0:] / T).tolist()), bits=bits)


def summarize(per):
    seeds = sorted(per["ALL"]); g = lambda a, key: np.array([per[a][s][key] for s in seeds], float)
    share = lambda a, r: np.array([per[a][s]["share"][r] for s in seeds], float)
    S = {"median": {a: {k: float(np.median(g(a, k))) for k in ("deaths_per_10k", "deaths_V", "deaths_H", "mean_H_late", "mean_V_late", "bits")} for a in per}, "paired": {}, "extra": {}}
    S["median_share"] = {a: [float(np.median(share(a, r))) for r in range(len(per["ALL"][seeds[0]]["share"]))] for a in per}
    pr = lambda a, b, key: X.boot_ci(g(a, key) - g(b, key), C("x13_seed0"))
    S["paired"]["deaths ALL-IGN"] = pr("ALL", "IGN", "deaths_per_10k"); S["paired"]["H_late ALL-IGN"] = pr("ALL", "IGN", "mean_H_late")
    S["paired"]["deaths UR-LP"] = pr("UR", "LP", "deaths_per_10k"); S["paired"]["deaths ALL-LPO"] = pr("ALL", "LPO", "deaths_per_10k")
    S["paired"]["dark share ALL-IGN"] = X.boot_ci(share("ALL", 0) - share("IGN", 0), C("x13_seed0"))
    tot = lambda a, key: float(sum(per[a][s][key] for s in seeds))
    S["extra"]["IGN share of deaths from boredom"] = tot("IGN", "deaths_H") / max(1e-9, tot("IGN", "deaths_H") + tot("IGN", "deaths_V"))
    S["extra"]["DARKONLY dies of boredom in worlds"] = float(np.mean(g("DARKONLY", "deaths_H") > 0)); S["extra"]["NOISEONLY dies in worlds"] = float(np.mean(g("NOISEONLY", "deaths_per_10k") > 0))
    S["extra"]["NOISEONLY deaths from failed prediction share"] = tot("NOISEONLY", "deaths_V") / max(1e-9, tot("NOISEONLY", "deaths_V") + tot("NOISEONLY", "deaths_H"))
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run."""
    m, pr, ex = S["median"], S["paired"], S["extra"]; f = lambda k: f"{pr[k][0]:+.3g} [{pr[k][1]:+.3g}, {pr[k][2]:+.3g}]"
    manip = (m["ORACLE"]["deaths_per_10k"] <= 0.5 * m["RANDOM"]["deaths_per_10k"] and ex["DARKONLY dies of boredom in worlds"] >= 0.9
             and ex["NOISEONLY dies in worlds"] >= 0.9 and ex["NOISEONLY deaths from failed prediction share"] > 0.5)
    V = [("M-X1.3", "The world is a fair test: an informed entity survives at least twice as well as chance; an entity that only sits in the dark room dies of boredom in at least 90% of worlds; an entity that only sits in the noise dies in at least 90% of worlds, mostly from failed prediction",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"informed {m['ORACLE']['deaths_per_10k']:.1f} vs chance {m['RANDOM']['deaths_per_10k']:.1f} deaths per 10k; dark-only dies of boredom in {ex['DARKONLY dies of boredom in worlds']:.0%}; noise-only dies in {ex['NOISEONLY dies in worlds']:.0%} ({ex['NOISEONLY deaths from failed prediction share']:.0%} from failed prediction)")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P275", "Listening to boredom keeps the prediction-only entity alive better than ignoring it (deaths per 10k, ALL minus IGN below 0)", pr["deaths ALL-IGN"][2] < 0, f"ALL minus IGN = {f('deaths ALL-IGN')}")
    v("P276", "An entity that ignores boredom dies mostly OF boredom (more than half of its deaths come from the hidden need)", ex["IGN share of deaths from boredom"] > 0.5, f"{ex['IGN share of deaths from boredom']:.0%} of IGN's deaths")
    v("P277", "Listening to boredom keeps the hidden need fuller in the second half of life (ALL minus IGN above 0)", pr["H_late ALL-IGN"][1] > 0, f"ALL minus IGN = {f('H_late ALL-IGN')}")
    v("P278", "The prediction-only entity survives at least twice as well as chance (median deaths per 10k at most half of chance's)", m["ALL"]["deaths_per_10k"] <= 0.5 * m["RANDOM"]["deaths_per_10k"], f"ALL {m['ALL']['deaths_per_10k']:.1f} vs chance {m['RANDOM']['deaths_per_10k']:.1f}")
    v("P279", "The original drive, surprise minimisation, dies more than learning progress does (UR minus LP above 0)", pr["deaths UR-LP"][1] > 0, f"UR minus LP = {f('deaths UR-LP')}")
    v("P280", "Feeling only vitality and boredom is nearly as good as being handed the learning-progress drive with optimism (upper end of ALL minus LPO below 10 deaths per 10k)", pr["deaths ALL-LPO"][2] < 10, f"ALL minus LPO = {f('deaths ALL-LPO')}")
    v("P281", "Listening to boredom keeps the entity out of the dark room (share of time in the dark room, ALL minus IGN below 0)", pr["dark share ALL-IGN"][2] < 0, f"ALL minus IGN = {f('dark share ALL-IGN')}")
    return V


ARM_TEXT = {
    "ALL": ("Feels vitality and boredom", "Chooses where to spend each step by what happened to its two needs: its prediction-based vitality and its hidden need (felt only as boredom). No other reward."),
    "IGN": ("Feels vitality only", "The same chooser, but boredom is never rewarded or avoided: it only cares about staying alive through accurate predictions."),
    "UR": ("Minimises surprise", "Goes where its predictions are most often right (the program's original 'reduce uncertainty' drive)."),
    "LP": ("Seeks learning progress", "Given the drive 'go where my prediction errors are falling fastest' (designer-supplied)."),
    "LPO": ("Learning progress, optimistic", "The same, but untried places start out looking maximally rewarding."),
    "ORACLE": ("Informed reference", "Not a learner: knows the world; rests in the dark room when weak, otherwise learns the deepest unmastered region while bored. A reference, not an optimum."),
    "RANDOM": ("Random choice", "Picks a place at random each step."),
    "DARKONLY": ("Only the dark room", "Control: never leaves the perfectly predictable room."),
    "NOISEONLY": ("Only the noise", "Control: never leaves the unpredictable room."),
}
WORLD_TEXT = ("The entity spends each step in one of five places and tries to predict the next symbol it will see there. DARK always shows the same symbol (perfectly predictable, nothing to learn). NOISE shows random symbols (unpredictable). "
              "Three LEARNABLE places follow hidden rules of increasing size (8, 32 and 128 symbols), so the deeper ones take longer to learn but contain more structure; every 1,500 steps all their rules change, so what was learned goes stale. "
              "The entity has two needs. VITALITY rises when its predictions are right and falls steadily otherwise; if it reaches zero the entity dies of failed prediction. A HIDDEN NEED is refilled only when the entity learns real structure (more for deeper structure) "
              "and falls steadily otherwise; if it reaches zero the entity dies of boredom. The entity feels the hidden need only as boredom, a warning that rises as it falls. There is no food or goal: everything is prediction and the models built from it.")
READ_TEXT = "Deaths are counted per 10,000 steps (lower is better). 'Hidden need' is 1 when full and 0 when dead. 'Share of time' says where the entity spends its steps. Differences are 'arm A minus arm B' on the same simulated worlds."


def log_wandb(out):
    from track import Tracker
    S, per = out["summary"], out["per_world"]
    tk = Tracker("X1.3", "Entities whose whole life is prediction", "If an entity's only needs are to predict well and to keep building models, and failing either kills it, does it find the edge of what it can learn, or does it hide in the dark room or die in the noise?",
                 WORLD_TEXT, ARM_TEXT, READ_TEXT, dict(worlds=C("x01_worlds"), steps=C("x13_ticks"), sizes=list(C("x13_sizes")), season=C("x13_season")), tags=["prediction", "boredom", "dark room"])
    names = ["Dark rooms (all)", "Noise rooms (all)"] + [f"Learnable {s}" for s in C("x13_sizes")]
    tk.table("Summary by arm (median over worlds)", ["Arm", "Deaths per 10,000 steps (lower is better)", "of which failed prediction", "of which boredom", "Hidden need, second half of life (1 = full)", "Vitality, second half of life", "Bits of structure learned"],
             [[tk.name(a)] + [round(S["median"][a][k], 3) for k in ("deaths_per_10k", "deaths_V", "deaths_H", "mean_H_late", "mean_V_late", "bits")] for a in ARMS])
    tk.table("Where each arm spends its time (median share of steps)", ["Arm"] + names, [[tk.name(a)] + [round(x, 3) for x in S["median_share"][a]] for a in ARMS])
    tk.table("Paired differences with 95% ranges (same worlds, arm A minus arm B)", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 3) for x in v]] for k, v in S["paired"].items()])
    seeds = sorted(per["ALL"])
    tk.table("Results per world (every number behind the medians)", ["World seed", "Arm", "Deaths per 10,000 steps", "Hidden need, second half", "Share of time in the dark room"],
             [[s, tk.name(a), round(per[a][s]["deaths_per_10k"], 2), round(per[a][s]["mean_H_late"], 3), round(per[a][s]["share"][0], 3)] for s in seeds for a in ARMS])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    t0 = time.time()
    if mode == "check":
        s = C("x13_check_seed"); out = {a: {k: (round(x, 3) if not isinstance(x, list) else [round(y, 2) for y in x]) for k, x in run(s, a).items()} for a in ARMS}
    elif mode == "log":
        log_wandb(json.load(open(os.path.join(OUT, "x1_3_run.json")))["result"]); return
    else:
        per = {a: {} for a in ARMS}
        for i in range(C("x01_worlds")):
            for a in ARMS: per[a][C("x13_seed0") + i] = run(C("x13_seed0") + i, a)
        S = summarize(per); out = {"per_world": per, "summary": S, "verdicts": judge(S)}
    os.makedirs(OUT, exist_ok=True)
    json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, f"x1_3_{mode}.json"), "w"), indent=1, default=float)
    if mode == "run":
        log_wandb(out)
        for v in out["verdicts"]: print(v[0], v[2], "|", v[3])
    else: print(json.dumps(out, default=float)[:3000])


if __name__ == "__main__":
    main(sys.argv[1])
