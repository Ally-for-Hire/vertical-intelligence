"""X1.4: how much of X1.3's result is the boredom FRAMING, and how much is the EXACTNESS of the boredom signal? (the confound found in X1.3's hostile reading)

Same world PL as X1.3 (vi/x1_3.py): 1 dark room, 6 noise rooms, 3 learnable rooms (8, 32, 128 symbols) re-drawn every 400 steps; vitality V from accuracy (dies at 0), a hidden need H refilled only by
bits of real structure newly learned (dies at 0), felt only as boredom. New arms on the same worlds and the same shared bandit (alpha, epsilon identical for every arm):
  BITS    designer arm: reward = the EXACT bits learned this step (x13_kappa x gain). No vitality term: tests whether an exact structure signal alone is enough
  ALLN2 / ALLN5 / ALLN10   ALL with the boredom reading corrupted: each reading of H gets Gaussian noise of sd 0.02 / 0.05 / 0.10 and the reward uses the mean of the last 10 readings
  LPT / LPOT   learning progress / optimistic learning progress with their two averaging windows TUNED on separate tuning worlds (x1_3 left them at X1.1's values, set for a 16-symbol world)
Reference arms from X1.3 on these same worlds: ALL (exact boredom), IGN (vitality only), LP, LPO (untuned windows), ORACLE, RANDOM, DARKONLY, NOISEONLY.
Usage: python x1_4.py tune | run | log          (writes ../results/x1_4_<mode>.json)"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x0_1 as X
import x1_3 as E

OUT = X.OUT
ARMS = ("ALL", "IGN", "LP", "LPO", "BITS", "ALLN2", "ALLN5", "ALLN10", "LPT", "LPOT", "ORACLE", "RANDOM", "DARKONLY", "NOISEONLY")
NOISE_SD = {"ALLN2": 0.02, "ALLN5": 0.05, "ALLN10": 0.10}
LP_GRID = [(0.3, 0.05)] + [(f, s) for f in (0.1, 0.3, 0.5) for s in (0.01, 0.02, 0.05) if (f, s) != (0.3, 0.05)]      # the X1.1 default first, so ties keep it


def one(seed, arm, windows=None):
    if arm in NOISE_SD: return E.run(seed, "ALLN", stream=20 + ARMS.index(arm), sigma=NOISE_SD[arm], lag=C("x14_lag"))
    if arm in ("LPT", "LPOT"):
        f, s = windows[arm]; return E.run(seed, arm, stream=20 + ARMS.index(arm), lp_f=f, lp_s=s)
    if arm == "BITS": return E.run(seed, "BITS", stream=20 + ARMS.index(arm))
    return E.run(seed, arm)                                              # X1.3 arms keep their X1.3 random streams


def tune():
    res = {}
    for arm in ("LPT", "LPOT"):
        score = {}
        for (f, s) in LP_GRID:
            score[(f, s)] = float(np.median([-E.run(C("x14_tune_seed0") + i, arm, stream=20 + ARMS.index(arm), lp_f=f, lp_s=s)["deaths_per_10k"] for i in range(C("x08_tune_worlds"))]))
        best = max(score, key=score.get); res[arm] = dict(lp_fast=best[0], lp_slow=best[1], score=score[best], default_score=score[(0.3, 0.05)])
    return res


def summarize(per):
    seeds = sorted(per["ALL"]); g = lambda a, k: np.array([per[a][s][k] for s in seeds], float); share = lambda a, r: np.array([per[a][s]["share"][r] for s in seeds], float)
    S = {"median": {a: {k: float(np.median(g(a, k))) for k in ("deaths_per_10k", "deaths_V", "deaths_H", "mean_H_late", "mean_V_late", "bits")} for a in per}, "paired": {}, "extra": {}}
    S["median_share"] = {a: [float(np.median(share(a, r))) for r in range(len(per["ALL"][seeds[0]]["share"]))] for a in per}
    pr = lambda a, b: X.boot_ci(g(a, "deaths_per_10k") - g(b, "deaths_per_10k"), C("x14_seed0"))
    for a, b in (("BITS", "ALL"), ("ALLN10", "ALL"), ("ALLN10", "IGN"), ("ALLN5", "IGN"), ("ALLN2", "IGN"), ("LPOT", "LPO"), ("LPT", "LP"), ("ALL", "LPOT"), ("ALL", "IGN")): S["paired"][f"deaths {a}-{b}"] = pr(a, b)
    tot = lambda a, k: float(sum(per[a][s][k] for s in seeds))
    S["extra"]["BITS share of deaths from failed prediction"] = tot("BITS", "deaths_V") / max(1e-9, tot("BITS", "deaths_V") + tot("BITS", "deaths_H"))
    S["extra"]["DARKONLY dies of boredom in worlds"] = float(np.mean(g("DARKONLY", "deaths_H") > 0)); S["extra"]["NOISEONLY dies in worlds"] = float(np.mean(g("NOISEONLY", "deaths_per_10k") > 0))
    S["extra"]["NOISEONLY deaths from failed prediction share"] = tot("NOISEONLY", "deaths_V") / max(1e-9, tot("NOISEONLY", "deaths_V") + tot("NOISEONLY", "deaths_H"))
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run."""
    m, pr, ex = S["median"], S["paired"], S["extra"]; f = lambda k: f"{pr[k][0]:+.3g} [{pr[k][1]:+.3g}, {pr[k][2]:+.3g}]"
    manip = (m["ORACLE"]["deaths_per_10k"] <= 0.5 * m["RANDOM"]["deaths_per_10k"] and ex["DARKONLY dies of boredom in worlds"] >= 0.9 and ex["NOISEONLY dies in worlds"] >= 0.9 and ex["NOISEONLY deaths from failed prediction share"] > 0.5)
    V = [("M-X1.4", "The world is a fair test (as in X1.3): the informed reference survives at least twice as well as chance, the dark-only entity dies of boredom and the noise-only entity of failed prediction in at least 90% of worlds",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"informed {m['ORACLE']['deaths_per_10k']:.1f} vs chance {m['RANDOM']['deaths_per_10k']:.1f}; dark-only boredom {ex['DARKONLY dies of boredom in worlds']:.0%}; noise-only {ex['NOISEONLY dies in worlds']:.0%}")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P288", "An exact structure-learned signal alone is NOT enough: the entity rewarded only by exact bits learned dies more than the one that also feels vitality and boredom (BITS minus ALL above 0), mostly from failed prediction", pr["deaths BITS-ALL"][1] > 0 and ex["BITS share of deaths from failed prediction"] > 0.5, f"BITS minus ALL = {f('deaths BITS-ALL')}; {ex['BITS share of deaths from failed prediction']:.0%} of BITS deaths from failed prediction")
    v("P289", "A noisy, lagged boredom reading hurts: ALLN10 dies more than the exact-boredom entity (ALLN10 minus ALL above 0)", pr["deaths ALLN10-ALL"][1] > 0, f"ALLN10 minus ALL = {f('deaths ALLN10-ALL')}")
    v("P290", "Even the noisiest reading (sd 0.10, lag 10) still beats ignoring boredom (ALLN10 minus IGN below 0)", pr["deaths ALLN10-IGN"][2] < 0, f"ALLN10 minus IGN = {f('deaths ALLN10-IGN')}")
    v("P291", "Tuning the learning-progress windows helps the optimistic learning-progress drive (LPOT minus LPO below 0)", pr["deaths LPOT-LPO"][2] < 0, f"LPOT minus LPO = {f('deaths LPOT-LPO')}")
    v("P292", "Even against tuned optimistic learning progress, feeling vitality and exact boredom is still better (ALL minus LPOT below 0)", pr["deaths ALL-LPOT"][2] < 0, f"ALL minus LPOT = {f('deaths ALL-LPOT')}")
    med = [m[a]["deaths_per_10k"] for a in ("ALL", "ALLN2", "ALLN5", "ALLN10")]
    v("P293", "Dose-response: median deaths do not decrease as the boredom noise grows (ALL <= ALLN2 <= ALLN5 <= ALLN10) and ALLN10 is above ALL", all(a <= b for a, b in zip(med, med[1:])) and med[-1] > med[0], "medians " + ", ".join(f"{x:.2f}" for x in med))
    return V


ARM_TEXT = dict(E.ARM_TEXT)
ARM_TEXT.update({"BITS": ("Rewarded by exact structure learned", "A designer-supplied curiosity: rewarded only by the exact number of bits of real structure it just learned. No sense of vitality or boredom."),
                 "ALLN2": ("Feels needs, slightly noisy boredom", "Like the main entity but its boredom reading is blurred (noise 0.02, averaged over 10 steps)."),
                 "ALLN5": ("Feels needs, noisy boredom", "Boredom reading noise 0.05, averaged over 10 steps."),
                 "ALLN10": ("Feels needs, very noisy boredom", "Boredom reading noise 0.10, averaged over 10 steps."),
                 "LPT": ("Learning progress, tuned", "Learning progress with its two averaging windows tuned on separate worlds."),
                 "LPOT": ("Learning progress, optimistic and tuned", "Optimistic learning progress with tuned windows."),
                 "ALL": ("Feels vitality and exact boredom", "Reward = change in vitality plus change in the (exactly read) hidden need. The main entity of X1.3.")})


def log_wandb(out):
    from track import Tracker
    S, per = out["summary"], out["per_world"]
    tk = Tracker("X1.4", "How much does boredom's exactness matter?", "In the prediction-only life, is the result due to feeling vitality and boredom, or only to the boredom signal being an exact count of what was learned? What happens when boredom is noisy and delayed?",
                 E.WORLD_TEXT, ARM_TEXT, E.READ_TEXT, dict(worlds=C("x01_worlds"), noise_levels=NOISE_SD, lag=C("x14_lag"), tuned_windows=out["tuned"]), tags=["boredom", "noise", "control"])
    tk.table("Summary by arm (median over worlds)", ["Arm", "Deaths per 10,000 steps (lower is better)", "of which failed prediction", "of which boredom", "Hidden need, second half of life (1 = full)", "Vitality, second half of life", "Bits of structure learned"],
             [[tk.name(a)] + [round(S["median"][a][k], 3) for k in ("deaths_per_10k", "deaths_V", "deaths_H", "mean_H_late", "mean_V_late", "bits")] for a in ARMS])
    names = ["Dark rooms (all)", "Noise rooms (all)"] + [f"Learnable {s}" for s in C("x13_sizes")]
    tk.table("Where each arm spends its time (median share of steps)", ["Arm"] + names, [[tk.name(a)] + [round(x, 3) for x in S["median_share"][a]] for a in ARMS])
    tk.table("Paired differences with 95% ranges (same worlds, arm A minus arm B)", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 3) for x in v]] for k, v in S["paired"].items()])
    seeds = sorted(per["ALL"])
    tk.table("Results per world (every number behind the medians)", ["World seed", "Arm", "Deaths per 10,000 steps", "Hidden need, second half", "Share of time in the dark room"],
             [[s, tk.name(a), round(per[a][s]["deaths_per_10k"], 2), round(per[a][s]["mean_H_late"], 3), round(per[a][s]["share"][0], 3)] for s in seeds for a in ARMS])
    tk.table("Tuned learning-progress windows", ["Arm", "Fast window", "Slow window", "Tuning score (minus deaths)", "Score with the untuned windows"], [[tk.name(a), v["lp_fast"], v["lp_slow"], round(v["score"], 2), round(v["default_score"], 2)] for a, v in out["tuned"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    t0 = time.time()
    if mode == "tune": out = tune()
    elif mode == "log":
        log_wandb(json.load(open(os.path.join(OUT, "x1_4_run.json")))["result"]); return
    else:
        tuned = json.load(open(os.path.join(OUT, "x1_4_tune.json")))["result"]; windows = {a: (v["lp_fast"], v["lp_slow"]) for a, v in tuned.items()}
        per = {a: {} for a in ARMS}
        for i in range(C("x01_worlds")):
            for a in ARMS: per[a][C("x14_seed0") + i] = one(C("x14_seed0") + i, a, windows)
        S = summarize(per); out = {"tuned": tuned, "per_world": per, "summary": S, "verdicts": judge(S)}
    os.makedirs(OUT, exist_ok=True)
    json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, f"x1_4_{mode}.json"), "w"), indent=1, default=float)
    if mode == "run":
        log_wandb(out)
        for v in out["verdicts"]: print(v[0], v[2], "|", v[3])
    else: print(json.dumps(out, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
