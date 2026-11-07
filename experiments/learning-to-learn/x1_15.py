"""X1.15: optimise the flat learner (author, 2026-11-02: 'let's start off the flat learner. optimize that literally as best you can so we can grow it
in complexity'). Learner: vi/flat.py. World: X1.10's (concepts on concepts) + X1.12's generalization test (new combinations, 20 points, connect only).

  parity   flat.py at DEFAULT reproduces X1.13's FLAT arm (= X1.9 FULLALL) bit-exactly on 3 worlds, and reports the speed of both
  tune     coordinate search over every free setting on TUNING worlds only (seeds x115_tune_seed0+, never scored): objective per world =
           mean of (later-problem bits, end-of-stream bits, generalization bits); a setting is changed only if the paired median gain over
           the current best is above 0 with its 95% range above 0; two rounds over all knobs. Writes ../results/x1_15_tune.json
  run      scored on FRESH worlds: FLAT0 (default) vs FLAT_OPT (tuned) vs DENSE; batches of 180, stop when two consecutive batches agree.
Usage: python x1_15.py parity | tune | run | log"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import flat
import x1_10 as X
import x1_12 as G

OUT = os.path.join(str(vipaths.RESULTS))
GRID = [("n_props", [1, 2, 4, 8]), ("refit_all", [False, True]), ("spec", [0.0, 0.5, 1.0, 2.0, 4.0]), ("new", [0.0, 1.0, 3.0]),   # CONST: x115_grid
        ("max_steps", [8, 12, 16]), ("cost_scale", [0.5, 1.0, 1.5, 2.0]), ("ridge", [1e-3, 1e-2, 1e-1, 1.0]),   # CONST: x115_grid
        ("unit_steps", [150, 300, 600]), ("stall", [0.01, 0.003, 0.0]), ("unit_lr", [0.01, 0.03, 0.1]),   # CONST: x115_grid
        ("ft_lr", [0.003, 0.01, 0.03]), ("ft_patience", [10, 30]), ("ft_steps", [200, 500]), ("sweep", [True, False])]   # CONST: x115_grid


def score(r):
    h = len(r["after"]) // 2
    late, final, gen = float(np.mean(r["after"][h:])), float(np.mean(r["final"])), float(np.mean(r["gen"]))
    return dict(late=late, early=float(np.mean(r["after"][:h])), final=final, gen=gen, J=(late + final + gen) / 3, nodes=r["nodes"], links=r["links"])


def eval_cfg(seed, cfg):
    data, _ = X.make_world(seed); gens = G.gen_problems(seed, data)
    t0 = time.time(); r = flat.run(data, gens, seed, cfg); s = score(r); s["sec"] = time.time() - t0
    return s


def parity():
    import x1_13 as Q
    for seed in (801000, 811000, 821000):                                  # CONST: x115_parity_seeds
        data, _ = X.make_world(seed); gens = G.gen_problems(seed, data)
        t0 = time.time(); a = Q.run_flat("FLAT", data, gens, seed); ta = time.time() - t0
        t0 = time.time(); b = flat.run(data, gens, seed); tb = time.time() - t0
        diff = max(abs(x - y) for x, y in zip(a["after"] + a["final"] + a["gen"], b["after"] + b["final"] + b["gen"]))
        print(f"seed {seed}: max diff {diff:.2e}; nodes {a['nodes']} vs {b['nodes']}; seconds old {ta:.2f} new {tb:.2f}")


def tune(pool):
    from x0_1 import boot_ci
    seeds = [C("x115_tune_seed0") + i for i in range(C("x115_tune_worlds"))]
    best = dict(flat.DEFAULT); cur = pool.starmap(eval_cfg, [(s, best) for s in seeds]); log = []
    J = lambda rs: np.array([r["J"] for r in rs])
    print(f"default: median J {np.median(J(cur)):.4f}, {np.median([r['sec'] for r in cur]):.2f} s per world", flush=True)
    for rnd in range(C("x115_rounds")):
        changed = False
        for knob, vals in GRID:
            for v in vals:
                if v == best[knob]: continue
                cfg = dict(best, **{knob: v}); rs = pool.starmap(eval_cfg, [(s, cfg) for s in seeds])
                ci = boot_ci(J(rs) - J(cur), C("x115_tune_seed0"))
                log.append(dict(round=rnd, knob=knob, value=v, gain=ci, J=float(np.median(J(rs))), sec=float(np.median([r["sec"] for r in rs]))))
                print(f"r{rnd} {knob}={v}: gain {ci[0]:+.4f} [{ci[1]:+.4f}, {ci[2]:+.4f}]  J {np.median(J(rs)):.4f}  {np.median([r['sec'] for r in rs]):.2f}s", flush=True)
                if ci[0] > 0 and ci[1] > 0:
                    best, cur, changed = cfg, rs, True; print(f"   -> keep {knob}={v}", flush=True)
        if not changed: break
    out = {"best": best, "log": log, "J_best": float(np.median(J(cur)))}
    json.dump(out, open(os.path.join(OUT, "x1_15_tune.json"), "w"), indent=1, default=float)
    print(json.dumps(best, default=float))


def world(seed, best):
    data, _ = X.make_world(seed); gens = G.gen_problems(seed, data)
    out = {}
    for a, cfg in (("FLAT0", flat.DEFAULT), ("FLAT_OPT", best)):
        t0 = time.time(); r = flat.run(data, gens, seed, cfg); s = score(r); s["sec"] = time.time() - t0; out[a] = s
    import x1_13 as Q
    Q.CTX.update(mode="FLAT", links={}, origin={}, far=[], usage=False, use={})
    r = G.run_dense_gen(data, gens, seed); r["nodes"] = r["nodes"]; r["links"] = 0; s = score(r); s["sec"] = 0.0; out["DENSE"] = s
    return out


ARMS = ("FLAT0", "FLAT_OPT", "DENSE")


def summarize(per, seed0):
    from x0_1 import boot_ci
    seeds = sorted(per); g = lambda a, k: np.array([per[s][a][k] for s in seeds], float)
    S = {"median": {a: {k: float(np.median(g(a, k))) for k in per[seeds[0]][a]} for a in ARMS}, "paired": {}, "worlds": len(seeds)}
    for k in ("late", "final", "gen", "J"): S["paired"][f"{k} FLAT_OPT-FLAT0"] = boot_ci(g("FLAT_OPT", k) - g("FLAT0", k), seed0)
    S["paired"]["gen FLAT_OPT-DENSE"] = boot_ci(g("FLAT_OPT", "gen") - g("DENSE", "gen"), seed0)
    S["vertical FLAT_OPT late-early"] = boot_ci(g("FLAT_OPT", "late") - g("FLAT_OPT", "early"), seed0)
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run. A win = paired 95% range on the predicted side of 0."""
    pr = S["paired"]; f = lambda k: f"{pr[k][0]:+.4f} [{pr[k][1]:+.4f}, {pr[k][2]:+.4f}]"; up = lambda k: pr[k][1] > 0
    V = []

    def v(pid, stmt, ok, ev): V.append((pid, stmt, "PASSED" if ok else "FAILED", ev))
    v("P436", "The tuned flat learner beats the default on fresh worlds in the tuning objective (mean of later, end and generalization bits): FLAT_OPT minus FLAT0, above 0", up("J FLAT_OPT-FLAT0"), f("J FLAT_OPT-FLAT0"))
    v("P437", "...on later problems", up("late FLAT_OPT-FLAT0"), f("late FLAT_OPT-FLAT0"))
    v("P438", "...on all problems at the end of the stream", up("final FLAT_OPT-FLAT0"), f("final FLAT_OPT-FLAT0"))
    v("P439", "...on generalization to new combinations", up("gen FLAT_OPT-FLAT0"), f("gen FLAT_OPT-FLAT0"))
    v("P440", "The tuned flat learner generalizes better than the plain network: FLAT_OPT minus DENSE, above 0", up("gen FLAT_OPT-DENSE"), f("gen FLAT_OPT-DENSE"))
    v("P441", "Vertical: FLAT_OPT later minus earlier bits, above 0", S["vertical FLAT_OPT late-early"][1] > 0,
      f"{S['vertical FLAT_OPT late-early'][0]:+.4f} [{S['vertical FLAT_OPT late-early'][1]:+.4f}, {S['vertical FLAT_OPT late-early'][2]:+.4f}]")
    return V


def run(pool):
    best = json.load(open(os.path.join(OUT, "x1_15_tune.json")))["best"]
    seed0 = C("x115_seed0"); per, history, bs, t0 = {}, [], C("x110_batch"), time.time()
    for start in range(0, C("x110_max_worlds"), bs):
        seeds = list(range(seed0 + start, seed0 + start + bs))
        for s_, r in zip(seeds, pool.starmap(world, [(s_, best) for s_ in seeds])): per[s_] = r
        S = summarize(per, seed0); V = judge(S); history.append([v[2] for v in V])
        print(f"{len(per)} worlds, {time.time() - t0:.0f} s:", " ".join(f"{v[0]}={v[2]}" for v in V), flush=True)
        if len(history) >= 2 and history[-1] == history[-2]: break
    S["stopped_at_worlds"] = len(per); S["best"] = best
    out = {"per_world": per, "summary": S, "verdicts": V}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_15_run.json"), "w"), indent=1, default=float)
    log_wandb(out)
    for v in V: print(v[0], v[2], "|", v[3])


def log_wandb(out):
    from track import Tracker
    S = out["summary"]
    tk = Tracker("X1.15", "Optimising the flat learner", "How good can the flat connection learner get when every free setting is tuned on separate worlds, and does the gain hold on fresh ones?",
                 X.WORLD_TEXT, {"FLAT0": ("Flat learner, default settings", "X1.9's best arm as it was."), "FLAT_OPT": ("Flat learner, tuned", f"Settings: {json.dumps(S['best'], default=float)}"),
                                "DENSE": ("Plain neural network", "32 shared units.")},
                 "Bits per point on unseen examples; generalization = new combinations of known concepts, 20 examples, connect only.", dict(worlds=S["worlds"], best=S["best"]), tags=["flat", "tuning"])
    keys = [("late", "Later problems"), ("final", "All problems at the end"), ("gen", "Generalization"), ("J", "Tuning objective"), ("nodes", "Nodes"), ("sec", "Seconds per world")]
    tk.table("Summary by arm (median over worlds)", ["Arm"] + [k[1] for k in keys], [[tk.name(a)] + [round(S["median"][a][k], 4) for k, _ in keys] for a in ARMS])
    tk.table("Paired differences with 95% ranges", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    import multiprocessing as mp, budget
    if mode == "parity": parity(); return
    if mode == "log": log_wandb(json.load(open(os.path.join(OUT, "x1_15_run.json")))["result"]); return
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    with mp.get_context("spawn").Pool(budget.workers()) as pool:
        tune(pool) if mode == "tune" else run(pool)


if __name__ == "__main__":
    main(sys.argv[1])
