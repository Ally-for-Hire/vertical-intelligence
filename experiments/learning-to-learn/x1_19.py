"""X1.19: a constant loop of training and expanding (author, 2026-11-02: 'loop in a constant process of training the learner and expanding the
complexity. save these worlds for future experimentation').

One persistent learner (flat.Learner, flat.OPT) in a wstream.Stream world, finite at every moment. After every problem the learner reports its OWN
bits on that problem's decision points (internal; no test data). When the mean of its last x119_win problems is not above the mean of the x119_win
before (progress has STAGNATED), and the current stage has lasted at least 2 x x119_win problems, the world expands by x119_expand concepts built on
the existing ones. The loop runs for x119_problems problems. A no-memory control (the same learner, new for every problem) is scored on every
problem. Each realized world is saved to worlds/x1_19/world_<seed>.json (replay: wstream.load).
Usage: python x1_19.py run | log"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import flat, deep, wstream

OUT = os.path.join(str(vipaths.RESULTS))
WORLDS = os.path.join(str(vipaths.WORLDS), "x1_19")
CEIL = dict(deep.OPT, ceiling_ridge=C("x116_ceiling_ridge"))


def world(seed):
    t0 = time.time()
    W = wstream.Stream(seed); L = flat.Learner(seed, flat.OPT); win, nexp = C("x119_win"), C("x119_expand")
    rows, own, start = [], [], 0
    for t in range(C("x119_problems")):
        d = W.next()
        bits, ob = L.step(d); own.append(ob)
        fresh = flat.run([d], [], seed, flat.OPT)["after"][0]; c = deep.ceiling(d, CEIL)
        rows.append(dict(t=t, stage=d["stage"], has_new=d["has_new"], share=bits / c, fresh=fresh / c, own=ob))
        if t + 1 - start >= 2 * win and np.mean(own[-win:]) <= np.mean(own[-2 * win:-win]):    # stagnation -> expand
            W.expand(nexp); start = t + 1
    W.save(os.path.join(WORLDS, f"world_{seed}.json"))
    return dict(rows=rows, events=W.events, concepts=len(W.lib.c), nodes=len(L.units), seconds=round(time.time() - t0, 1))


def per_stage(r):
    """Per stage: mean share in its first x119_first problems (absorption) for learner and control, and the in-stage trend."""
    out, rows, nf = [], r["rows"], C("x119_first")
    for s in sorted({x["stage"] for x in rows}):
        R = [x for x in rows if x["stage"] == s]
        if len(R) < 2 * nf: continue
        h = len(R) // 2
        out.append(dict(stage=s, n=len(R), first=float(np.mean([x["share"] for x in R[:nf]])), first_fresh=float(np.mean([x["fresh"] for x in R[:nf]])),
                        trend=float(np.mean([x["share"] for x in R[h:]]) - np.mean([x["share"] for x in R[:h]])),
                        end=float(np.mean([x["share"] for x in R[-nf:]]))))
    return out


def summarize(per, seed0):
    from x0_1 import boot_ci
    seeds = sorted(per)
    adv = np.array([np.mean([x["share"] - x["fresh"] for x in per[s]["rows"]]) for s in seeds])
    trend = np.array([np.mean([st["trend"] for st in per_stage(per[s])]) for s in seeds])

    def slope(s, key):
        P = per_stage(per[s]); P = [p for p in P if p["stage"] >= 1]
        if len(P) < 2: return np.nan
        x = np.array([p["stage"] for p in P], float); y = np.array([p[key] for p in P])
        return float(np.polyfit(x, y, 1)[0])
    absorb = np.array([slope(s, "adv") if False else np.nan for s in seeds])
    sl_adv = []
    for s in seeds:
        P = [p for p in per_stage(per[s]) if p["stage"] >= 1]
        sl_adv.append(np.polyfit([p["stage"] for p in P], [p["first"] - p["first_fresh"] for p in P], 1)[0] if len(P) >= 2 else np.nan)
    sl_adv = np.array(sl_adv, float); sl_end = np.array([slope(s, "end") for s in seeds])
    ok_a, ok_e = ~np.isnan(sl_adv), ~np.isnan(sl_end)
    exps = np.array([len(per[s]["events"]) for s in seeds])
    stages_all = [p for s in seeds for p in per_stage(per[s])]
    by_stage = {}
    for p in stages_all: by_stage.setdefault(p["stage"], []).append(p)
    S = dict(worlds=len(seeds), expansions_median=float(np.median(exps)), expansions_min=int(exps.min()), concepts_median=float(np.median([per[s]["concepts"] for s in seeds])),
             adv=boot_ci(adv, seed0), trend=boot_ci(trend, seed0), slope_adv=boot_ci(sl_adv[ok_a], seed0), slope_end=boot_ci(sl_end[ok_e], seed0),
             worlds_with_slope=int(ok_a.sum()), share_all=float(np.median([np.mean([x["share"] for x in per[s]["rows"]]) for s in seeds])),
             fresh_all=float(np.median([np.mean([x["fresh"] for x in per[s]["rows"]]) for s in seeds])),
             by_stage={k: dict(worlds=len(v), first=float(np.median([p["first"] for p in v])), first_fresh=float(np.median([p["first_fresh"] for p in v])),
                               end=float(np.median([p["end"] for p in v])), length=float(np.median([p["n"] for p in v]))) for k, v in sorted(by_stage.items())})
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run. A win = 95% range on the predicted side of 0."""
    f = lambda k: f"{S[k][0]:+.4f} [{S[k][1]:+.4f}, {S[k][2]:+.4f}]"; up = lambda k: S[k][1] > 0
    manip = S["expansions_min"] >= 1 and S["expansions_median"] >= C("x119_min_expansions")
    V = [("M-X1.19", "Fair test: the loop actually expands (every world at least once, median at least 3 expansions)",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"expansions median {S['expansions_median']:.0f}, min {S['expansions_min']}; concepts {S['concepts_median']:.0f}; worlds with >=2 expansions scored for slopes {S['worlds_with_slope']}")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P460", "Across the whole loop the persistent learner beats the same learner with no memory: mean share of ceiling minus control, above 0", up("adv"), f("adv"))
    v("P461", "Within each finite stage the learner improves: mean over stages of (later half minus earlier half share), above 0", up("trend"), f("trend"))
    v("P462", "VERTICAL ACROSS EXPANSIONS: the advantage over no memory on the first problems after an expansion grows with each expansion (slope over stages, above 0)", up("slope_adv"), f("slope_adv"))
    v("P463", "The learner keeps up with growing complexity: share at the end of each stage does not fall across stages (slope above 0)", up("slope_end"), f("slope_end"))
    return V


def run():
    import multiprocessing as mp, budget
    from x0_1 import boot_ci
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    os.makedirs(WORLDS, exist_ok=True)
    seed0 = C("x119_seed0"); t0 = time.time(); seeds = list(range(seed0, seed0 + C("x119_worlds")))
    with mp.get_context("spawn").Pool(budget.workers()) as pool: res = pool.map(world, seeds, chunksize=1)
    per = dict(zip(seeds, res)); S = summarize(per, seed0); V = judge(S)
    out = {"per_world": per, "summary": S, "verdicts": V}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_19_run.json"), "w"), indent=1, default=float)
    log_wandb(out)
    print(json.dumps({k: S[k] for k in ("expansions_median", "concepts_median", "share_all", "fresh_all")}, default=float))
    for k, b in S["by_stage"].items(): print(f"stage {k}: worlds {b['worlds']} length {b['length']:.0f} first {b['first']:.3f} (no memory {b['first_fresh']:.3f}) end {b['end']:.3f}")
    for v in V: print(v[0], v[2], "|", v[3])
    print(f"{time.time() - t0:.0f} s")


def log_wandb(out):
    from track import Tracker
    S = out["summary"]
    tk = Tracker("X1.19", "Train, master, expand: a constant loop", "When the world expands only after the learner's own progress stalls, does each expansion get absorbed faster than the last?",
                 "A world that is finite at every moment; whenever the learner stops improving (by its own measure) it grows by 12 concepts built on the old ones. 240 problems per world; every example is new.",
                 {"LEARNER": ("Persistent flat learner", "Keeps everything it learns, across every expansion."), "CONTROL": ("Same learner, no memory", "Starts from nothing on every problem.")},
                 "Share of ceiling = bits held on unseen examples divided by what the true concepts allow with the same examples.", dict(worlds=S["worlds"]), tags=["loop", "expansion", "vertical"])
    tk.table("By stage (median over worlds)", ["Stage", "Worlds", "Problems in stage", "First problems: share", "First problems: no-memory share", "End of stage: share"],
             [[k, b["worlds"], round(b["length"], 1), round(b["first"], 4), round(b["first_fresh"], 4), round(b["end"], 4)] for k, b in S["by_stage"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


if __name__ == "__main__":
    {"run": run, "log": lambda: log_wandb(json.load(open(os.path.join(OUT, "x1_19_run.json")))["result"])}[sys.argv[1]]()
