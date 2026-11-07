"""X1.16: grow the optimised flat learner in complexity, driven only by STAGNATION, with a fair 'better with experience' measure (author, 2026-11-02).

Fair vertical measure: each problem is scored RELATIVE TO ITS OWN CEILING (bits / bits of a readout given the problem's true concepts with the same
examples), so later problems being harder (deeper concepts arrive later in this world) no longer biases 'later minus earlier'.
Arms (vi/deep.py): OPT (flat.OPT, no depth)  DEEP_STAG (deep nodes tried only when flat progress stagnates)  DEEP_ALWAYS (deep nodes offered every
step)  DENSE (plain network). World and generalization test as X1.15.
Usage: python x1_16.py parity | run | log"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import flat, deep
import x1_10 as X
import x1_12 as G

OUT = os.path.join(str(vipaths.RESULTS))
ARMS = ("OPT", "DEEP_STAG", "DEEP_ALWAYS", "DENSE")
MODES = {"OPT": "off", "DEEP_STAG": "stag", "DEEP_ALWAYS": "always"}


def parity():
    for seed in (801000, 811000, 821000):                                  # CONST: x115_parity_seeds
        data, _ = X.make_world(seed); gens = G.gen_problems(seed, data)
        t0 = time.time(); a = flat.run(data, gens, seed, flat.OPT); ta = time.time() - t0
        t0 = time.time(); b = deep.run(data, gens, seed, {"deep_mode": "off"}); tb = time.time() - t0
        diff = max(abs(x - y) for x, y in zip(a["after"] + a["final"] + a["gen"], b["after"] + b["final"] + b["gen"]))
        print(f"seed {seed}: max diff {diff:.2e}; nodes {a['nodes']} vs {b['nodes']}; seconds flat {ta:.2f} deep(off) {tb:.2f}")


def stats(r, half):
    ra = r["rel_after"]
    return dict(late=float(np.mean(r["after"][half:])), early=float(np.mean(r["after"][:half])), final=float(np.mean(r["final"])), gen=float(np.mean(r["gen"])),
                rel_late=float(np.mean(ra[half:])), rel_early=float(np.mean(ra[:half])), rel_gen=float(np.mean(r["rel_gen"])),
                nodes=r["nodes"], deep_nodes=r.get("deep_nodes", 0), deep_links=r.get("deep_links", 0), links=r.get("links", 0),
                ceil_late=float(np.mean(r["ceil_after"][half:])), ceil_early=float(np.mean(r["ceil_after"][:half])))


def world(seed):
    t0 = time.time()
    data, _ = X.make_world(seed); gens = G.gen_problems(seed, data); half = len(data) // 2
    out = {}
    for a in ARMS:
        if a == "DENSE":
            import x1_13 as Q
            Q.CTX.update(mode="FLAT", links={}, origin={}, far=[], usage=False, use={})
            r = G.run_dense_gen(data, gens, seed)
            ca = [deep.ceiling(d, deep.OPT | {"ceiling_ridge": 1e-3}) for d in data]; cg = [deep.ceiling(d, deep.OPT | {"ceiling_ridge": 1e-3}) for d in gens]   # CONST: x116_ceiling_ridge
            r.update(rel_after=[x / c for x, c in zip(r["after"], ca)], rel_gen=[x / c for x, c in zip(r["gen"], cg)], ceil_after=ca)
        else:
            r = deep.run(data, gens, seed, {"deep_mode": MODES[a]})
        out[a] = stats(r, half)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(per, seed0):
    from x0_1 import boot_ci
    seeds = sorted(per); g = lambda a, k: np.array([per[s][a][k] for s in seeds], float)
    S = {"median": {a: {k: float(np.median(g(a, k))) for k in per[seeds[0]][a]} for a in ARMS}, "paired": {}, "worlds": len(seeds)}
    for a, b, k in (("DEEP_STAG", "OPT", "late"), ("DEEP_STAG", "OPT", "gen"), ("DEEP_STAG", "DEEP_ALWAYS", "gen"), ("DEEP_STAG", "OPT", "rel_late"),
                    ("DEEP_STAG", "OPT", "final"), ("DEEP_ALWAYS", "OPT", "gen"), ("DEEP_STAG", "OPT", "deep_nodes")):
        S["paired"][f"{k} {a}-{b}"] = boot_ci(g(a, k) - g(b, k), seed0)
    for a in ("OPT", "DEEP_STAG", "DENSE"):
        S["paired"][f"vertical {a} rel_late-rel_early"] = boot_ci(g(a, "rel_late") - g(a, "rel_early"), seed0)
    S["paired"]["ceiling late-early"] = boot_ci(g("OPT", "ceil_late") - g("OPT", "ceil_early"), seed0)
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run. A win = paired 95% range on the predicted side of 0."""
    m, pr = S["median"], S["paired"]
    f = lambda k: f"{pr[k][0]:+.4f} [{pr[k][1]:+.4f}, {pr[k][2]:+.4f}]"; up = lambda k: pr[k][1] > 0
    manip = up("deep_nodes DEEP_STAG-OPT") and m["DEEP_STAG"]["deep_links"] > 0
    V = [("M-X1.16", "Fair test: stagnation-driven growth actually builds and links deep nodes (DEEP_STAG has more deep nodes than OPT, and links to some)",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"deep nodes {f('deep_nodes DEEP_STAG-OPT')}; deep links {m['DEEP_STAG']['deep_links']:.0f}; ceiling later-earlier {f('ceiling late-early')}")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P442", "Growing deeper when stagnant helps later problems: DEEP_STAG minus OPT, later bits, above 0", up("late DEEP_STAG-OPT"), f("late DEEP_STAG-OPT"))
    v("P443", "Growing deeper when stagnant helps generalization: DEEP_STAG minus OPT, above 0", up("gen DEEP_STAG-OPT"), f("gen DEEP_STAG-OPT"))
    v("P444", "Depth only on stagnation generalizes better than depth always on offer: DEEP_STAG minus DEEP_ALWAYS, above 0", up("gen DEEP_STAG-DEEP_ALWAYS"), f("gen DEEP_STAG-DEEP_ALWAYS"))
    v("P445", "Fair vertical: the optimised flat learner reaches a larger share of each problem's ceiling on later problems than earlier ones, above 0", up("vertical OPT rel_late-rel_early"), f("vertical OPT rel_late-rel_early"))
    v("P446", "Fair vertical for the stagnation-grown learner, above 0", up("vertical DEEP_STAG rel_late-rel_early"), f("vertical DEEP_STAG rel_late-rel_early"))
    v("P447", "Relative to each problem's ceiling, the stagnation-grown learner does better on later problems than the flat one: DEEP_STAG minus OPT, above 0", up("rel_late DEEP_STAG-OPT"), f("rel_late DEEP_STAG-OPT"))
    return V


def run():
    import multiprocessing as mp, budget
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x116_seed0"); per, history, bs, t0 = {}, [], C("x110_batch"), time.time()
    with mp.get_context("spawn").Pool(budget.workers()) as pool:
        for start in range(0, C("x110_max_worlds"), bs):
            seeds = list(range(seed0 + start, seed0 + start + bs))
            for s_, r in zip(seeds, pool.map(world, seeds, chunksize=1)): per[s_] = r
            S = summarize(per, seed0); V = judge(S); history.append([v[2] for v in V])
            print(f"{len(per)} worlds, {time.time() - t0:.0f} s:", " ".join(f"{v[0]}={v[2].split()[0]}" for v in V), flush=True)
            if len(history) >= 2 and history[-1] == history[-2]: break
    S["stopped_at_worlds"] = len(per)
    out = {"per_world": per, "summary": S, "verdicts": V}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_16_run.json"), "w"), indent=1, default=float)
    log_wandb(out)
    for v in V: print(v[0], v[2], "|", v[3])


def log_wandb(out):
    from track import Tracker
    S = out["summary"]
    arms = {"OPT": ("Optimised flat learner (X1.15)", "No deep nodes."), "DEEP_STAG": ("Grows deeper when stuck", "Tries ordinary nodes first; only when none helps does it build nodes on top of the nodes it already uses."),
            "DEEP_ALWAYS": ("Deep nodes always on offer", "Deep candidates are offered at every step."), "DENSE": ("Plain neural network", "32 shared units.")}
    tk = Tracker("X1.16", "Growing in complexity when stuck", "Does building nodes on nodes, only when ordinary progress stalls, help? And does the learner get better with experience when each problem is judged against its own ceiling?",
                 X.WORLD_TEXT, arms, "Bits per point on unseen examples; 'share of ceiling' = bits divided by what a readout given the true hidden concepts reaches with the same examples.",
                 dict(worlds=S["worlds"]), tags=["depth", "stagnation", "vertical"])
    keys = [("late", "Later problems (bits)"), ("final", "All problems at the end"), ("gen", "Generalization"), ("rel_early", "Share of ceiling, earlier"), ("rel_late", "Share of ceiling, later"),
            ("nodes", "Nodes"), ("deep_nodes", "Deep nodes"), ("deep_links", "Links to deep nodes")]
    tk.table("Summary by arm (median over worlds)", ["Arm"] + [k[1] for k in keys], [[tk.name(a)] + [round(S["median"][a][k], 4) for k, _ in keys] for a in ARMS])
    tk.table("Paired differences with 95% ranges", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


if __name__ == "__main__":
    {"parity": parity, "run": run, "log": lambda: log_wandb(json.load(open(os.path.join(OUT, "x1_16_run.json")))["result"])}[sys.argv[1]]()
