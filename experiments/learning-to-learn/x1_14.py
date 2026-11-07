"""X1.14: decreasing rewards for consistently used nodes (author, 2026-11-02; same world as X1.10-X1.13).

Each node keeps a fading count of recent use: after every problem, use <- x114_use_keep x use, +1 if the problem used the node. Reusing a node pays
its measured help x 2^(-use): a node used by nearly every recent problem pays almost nothing, a rarely used one nearly full, a new node full.
Built on X1.13 (vi/x1_13.py): flat learner, optional distance reward for less direct assistance; generalization test as X1.12.
Arms: FLAT (X1.9 flat)  FLAT_G (X1.13 distance reward)  FLAT_GU (distance + decreasing reward for consistent use)  FLAT_U (decreasing reward alone)  DENSE.
Usage: python x1_14.py run | log"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_7 as B
import x1_10 as X
import x1_12 as G
import x1_13 as Q

OUT = B.OUT
ARMS = ("FLAT", "FLAT_G", "FLAT_GU", "FLAT_U", "DENSE")
Q.ARMS = ARMS
MODE = {"FLAT": ("FLAT", False), "FLAT_G": ("FLAT_G", False), "FLAT_GU": ("FLAT_G", True), "FLAT_U": ("FLAT", True)}


def world(seed):
    t0 = time.time()
    data, _ = X.make_world(seed); gens = G.gen_problems(seed, data)
    res = {}
    for a in ARMS:
        if a == "DENSE":
            Q.CTX.update(mode="FLAT", links={}, origin={}, far=[], usage=False, use={})
            r = G.run_dense_gen(data, gens, seed); res[a] = dict(after=r["after"], final=r["final"], nodes=r["nodes"], gen=r["gen"], far=0.0)
        else:
            mode, usage = MODE[a]; res[a] = Q.run_flat(a, data, gens, seed, mode=mode, usage=usage)
    half = len(data) // 2
    out = {"arms": {a: {"late": float(np.mean(r["after"][half:])), "early": float(np.mean(r["after"][:half])), "final_all": float(np.mean(r["final"])),
                        "gen": float(np.mean(r["gen"])), "nodes": r["nodes"], "far": r["far"]} for a, r in res.items()}}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(per, seed0):
    from x0_1 import boot_ci
    seeds = sorted(per)
    g = lambda a, k: np.array([per[s]["arms"][a][k] for s in seeds], float)
    S = {"median": {a: {k: float(np.median(g(a, k))) for k in per[seeds[0]]["arms"][a]} for a in ARMS}, "paired": {}, "worlds": len(seeds)}
    for a, b, k in (("FLAT_GU", "FLAT_G", "nodes"), ("FLAT_U", "FLAT", "nodes"), ("FLAT_GU", "FLAT_G", "gen"), ("FLAT_GU", "FLAT_G", "final_all"),
                    ("FLAT_U", "FLAT", "gen"), ("FLAT_GU", "FLAT", "gen"), ("FLAT_U", "FLAT", "late"), ("FLAT_GU", "DENSE", "gen"), ("FLAT_U", "FLAT", "final_all")):
        S["paired"][f"{k} {a}-{b}"] = boot_ci(g(a, k) - g(b, k), seed0)
    S["vertical FLAT_GU late-early"] = boot_ci(g("FLAT_GU", "late") - g("FLAT_GU", "early"), seed0)
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run. A win = paired 95% range on the predicted side of 0."""
    pr = S["paired"]
    f = lambda k: f"{pr[k][0]:+.4f} [{pr[k][1]:+.4f}, {pr[k][2]:+.4f}]"
    up = lambda k: pr[k][1] > 0
    manip = up("nodes FLAT_GU-FLAT_G")
    V = [("M-X1.14", "Fair test: the decreasing reward acts: with it the distance-rewarded learner builds more nodes (FLAT_GU minus FLAT_G nodes, range above 0)",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"nodes FLAT_GU-FLAT_G {f('nodes FLAT_GU-FLAT_G')}; FLAT_U-FLAT {f('nodes FLAT_U-FLAT')}")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P429", "Decreasing rewards for consistently used nodes rescue the distance reward on generalization: FLAT_GU minus FLAT_G, above 0", up("gen FLAT_GU-FLAT_G"), f("gen FLAT_GU-FLAT_G"))
    v("P430", "They also rescue its memory of old problems: FLAT_GU minus FLAT_G, all problems at the end, above 0", up("final_all FLAT_GU-FLAT_G"), f("final_all FLAT_GU-FLAT_G"))
    v("P431", "Distance + decreasing reward beats the plain flat learner on generalization: FLAT_GU minus FLAT, above 0", up("gen FLAT_GU-FLAT"), f("gen FLAT_GU-FLAT"))
    v("P432", "The decreasing reward alone helps the flat learner generalize: FLAT_U minus FLAT, above 0", up("gen FLAT_U-FLAT"), f("gen FLAT_U-FLAT"))
    v("P433", "The decreasing reward alone helps later problems: FLAT_U minus FLAT, above 0", up("late FLAT_U-FLAT"), f("late FLAT_U-FLAT"))
    v("P434", "Distance + decreasing reward generalizes better than the plain network: FLAT_GU minus DENSE, above 0", up("gen FLAT_GU-DENSE"), f("gen FLAT_GU-DENSE"))
    v("P435", "Vertical: FLAT_GU later minus earlier bits, above 0", S["vertical FLAT_GU late-early"][1] > 0,
      f"{S['vertical FLAT_GU late-early'][0]:+.4f} [{S['vertical FLAT_GU late-early'][1]:+.4f}, {S['vertical FLAT_GU late-early'][2]:+.4f}]")
    return V


ARM_TEXT = {"FLAT": Q.ARM_TEXT["FLAT"], "FLAT_G": Q.ARM_TEXT["FLAT_G"],
            "FLAT_GU": ("Distant help + tiring of overused nodes", "Pays more for distant help, and a node used by many recent problems pays less and less."),
            "FLAT_U": ("Tiring of overused nodes", "A node used by many recent problems pays less and less; no distance reward."), "DENSE": Q.ARM_TEXT["DENSE"]}


def log_wandb(out):
    from track import Tracker
    S = out["summary"]
    tk = Tracker("X1.14", "Overused nodes pay less", "If consistently used nodes earn decreasing rewards, does the learner build more specific nodes and generalize better (with and without the distance reward)?",
                 X.WORLD_TEXT, ARM_TEXT, B.READ_TEXT + " 'Generalization' = bits on new combinations of known concepts.", dict(worlds=S["worlds"]), tags=["generalization", "habituation"])
    keys = [("late", "Bits per point, later problems"), ("final_all", "Bits per point at the end, all problems"), ("gen", "Generalization (higher is better)"), ("nodes", "Nodes in total"), ("far", "Mean distance of reused links")]
    tk.table("Summary by arm (median over worlds)", ["Arm"] + [k[1] for k in keys], [[tk.name(a)] + [round(S["median"][a][k], 4) for k, _ in keys] for a in ARMS])
    tk.table("Paired differences with 95% ranges (same worlds, A minus B)", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    import multiprocessing as mp, budget
    t0 = time.time()
    if mode == "log":
        log_wandb(json.load(open(os.path.join(OUT, "x1_14_run.json")))["result"]); return
    seed0 = C("x114_seed0"); nw = C("x110_max_worlds")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    per, history, bs, V, S = {}, [], C("x110_batch"), None, None
    with mp.get_context("spawn").Pool(budget.workers()) as pool:
        for start in range(0, nw, bs):
            seeds = [seed0 + i for i in range(start, min(nw, start + bs))]
            for s_, r in zip(seeds, pool.map(world, seeds, chunksize=1)): per[s_] = r
            S = summarize(per, seed0); V = judge(S); history.append([v[2] for v in V])
            print(f"{len(per)} worlds, {time.time() - t0:.0f} s:", " ".join(f"{v[0]}={v[2].split()[0]}" for v in V), flush=True)
            if len(history) >= 2 and history[-1] == history[-2]: break
    S["stopped_at_worlds"] = len(per)
    out = {"per_world": per, "summary": S, "verdicts": V, "verdict_history": history}
    os.makedirs(OUT, exist_ok=True)
    json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, f"x1_14_{mode}.json"), "w"), indent=1, default=float)
    if mode == "run": log_wandb(out)
    for v in V: print(v[0], v[2], "|", v[3])


if __name__ == "__main__":
    main(sys.argv[1])
