"""X1.18: a FINITE fixed world first, then ONE deliberate expansion (author ruling 2026-11-02; X1.17 is VOID because its world expanded during the run).

World: winf.make_finite_world(FINITE): phase A = 60 problems from a fixed 24-concept library (nothing new appears); phase B = the library grows once
by 12 concepts built on the old ones, then 30 more problems (half include a new concept). Inputs and mixing weights are always fresh: memorising is
useless, mastering the finite library is possible.
Arms: OPT (flat.OPT)  FRESH (flat.OPT with NO memory: a new learner for every problem; the control that separates experience from the world)  DENSE.
Measures: share of each problem's own ceiling; vertical A = later half of phase A minus earlier half; phase B on problems with new concepts.
Usage: python x1_18.py run | log"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import flat, deep, winf
import x1_13 as Q
import x1_12 as G

OUT = os.path.join(str(vipaths.RESULTS))
ARMS = ("OPT", "FRESH", "DENSE")
CEIL = dict(deep.OPT, ceiling_ridge=C("x116_ceiling_ridge"))


def stats(after, final, gen, data, ca, cg):
    rel = np.array([x / c for x, c in zip(after, ca)]); A = [i for i, d in enumerate(data) if not d["phase"]]; h = len(A) // 2
    Bn = [i for i, d in enumerate(data) if d["phase"] and d["has_new"]]; Bo = [i for i, d in enumerate(data) if d["phase"] and not d["has_new"]]
    m = lambda idx: float(np.mean(rel[idx])) if len(idx) else float("nan")
    return dict(relA_early=m(A[:h]), relA_late=m(A[h:]), relB_new=m(Bn), relB_old=m(Bo), late_bits=float(np.mean(np.array(after)[A[h:]])),
                final=float(np.mean(final)), gen=float(np.mean(gen)), rel_gen=float(np.mean([x / c for x, c in zip(gen, cg)])))


def world(seed):
    t0 = time.time()
    data, gens, info = winf.make_finite_world(seed, winf.FINITE)
    ca = [deep.ceiling(d, CEIL) for d in data]; cg = [deep.ceiling(d, CEIL) for d in gens]
    seen = {np.round(x, 12).tobytes() for d in data + gens for x in np.vstack([d["tr"][0], d["va"][0]])}   # CONST: winf_cfg
    rep = sum(np.round(x, 12).tobytes() in seen for d in data + gens for x in d["te"][0])                  # CONST: winf_cfg
    out = {"world": dict(info, repeats=int(rep), A_only_original=bool(max(max(d["S"]) for d in data if not d["phase"]) < info["concepts_A"]))}
    r = flat.run(data, gens, seed, flat.OPT); out["OPT"] = stats(r["after"], r["final"], r["gen"], data, ca, cg); out["OPT"]["nodes"] = r["nodes"]
    fa = [flat.run([d], [], seed, flat.OPT)["after"][0] for d in data]
    out["FRESH"] = stats(fa, fa, [flat.run([d], [], seed, flat.OPT)["after"][0] for d in gens], data, ca, cg); out["FRESH"]["nodes"] = 0
    Q.CTX.update(mode="FLAT", links={}, origin={}, far=[], usage=False, use={})
    r = G.run_dense_gen(data, gens, seed); out["DENSE"] = stats(r["after"], r["final"], r["gen"], data, ca, cg); out["DENSE"]["nodes"] = 32   # CONST: x17_dense_h
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(per, seed0):
    from x0_1 import boot_ci
    seeds = sorted(per); g = lambda a, k: np.array([per[s][a][k] for s in seeds], float)
    S = {"median": {a: {k: float(np.nanmedian(g(a, k))) for k in per[seeds[0]][a]} for a in ARMS}, "paired": {}, "worlds": len(seeds)}
    S["world"] = dict(repeats_total=int(sum(per[s]["world"]["repeats"] for s in seeds)), A_only_original=all(per[s]["world"]["A_only_original"] for s in seeds),
                      concepts_A=float(np.median([per[s]["world"]["concepts_A"] for s in seeds])), concepts=float(np.median([per[s]["world"]["concepts"] for s in seeds])),
                      used_A=float(np.median([per[s]["world"]["used_A"] for s in seeds])))
    for a in ARMS: S["paired"][f"vertical A {a}"] = boot_ci(g(a, "relA_late") - g(a, "relA_early"), seed0)
    S["paired"]["vertical A OPT minus FRESH"] = boot_ci((g("OPT", "relA_late") - g("OPT", "relA_early")) - (g("FRESH", "relA_late") - g("FRESH", "relA_early")), seed0)
    S["paired"]["relA_late OPT-FRESH"] = boot_ci(g("OPT", "relA_late") - g("FRESH", "relA_late"), seed0)
    ok = ~np.isnan(g("OPT", "relB_new"))
    S["paired"]["relB_new OPT-FRESH"] = boot_ci((g("OPT", "relB_new") - g("FRESH", "relB_new"))[ok], seed0)
    S["paired"]["gen OPT-DENSE"] = boot_ci(g("OPT", "gen") - g("DENSE", "gen"), seed0)
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run. A win = paired 95% range on the predicted side of 0."""
    pr, w = S["paired"], S["world"]; f = lambda k: f"{pr[k][0]:+.4f} [{pr[k][1]:+.4f}, {pr[k][2]:+.4f}]"; up = lambda k: pr[k][1] > 0
    manip = w["repeats_total"] == 0 and w["A_only_original"] and w["concepts"] > w["concepts_A"]
    V = [("M-X1.18", "Fair test: no test point ever shown before; phase A uses only the original fixed library; the expansion adds concepts",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"repeats {w['repeats_total']}; phase A original only {w['A_only_original']}; concepts {w['concepts_A']:.0f} -> {w['concepts']:.0f}; used in A {w['used_A']:.0f}")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P454", "In a finite world the flat learner gets better with experience: share of ceiling, later half of phase A minus earlier half, above 0", up("vertical A OPT"), f("vertical A OPT"))
    v("P455", "Experience beats no memory: OPT minus FRESH, share of ceiling on the later half of phase A, above 0", up("relA_late OPT-FRESH"), f("relA_late OPT-FRESH"))
    v("P456", "Control: a learner with no memory does NOT improve over phase A (its trend's range is not above 0)", pr["vertical A FRESH"][1] <= 0, f("vertical A FRESH"))
    v("P457", "The experience effect: OPT's phase-A trend minus FRESH's, above 0", up("vertical A OPT minus FRESH"), f("vertical A OPT minus FRESH"))
    v("P458", "After the expansion, prior knowledge helps absorb new concepts: OPT minus FRESH on phase-B problems with a new concept, above 0", up("relB_new OPT-FRESH"), f("relB_new OPT-FRESH"))
    v("P459", "The flat learner generalizes better than the plain network: OPT minus DENSE, above 0", up("gen OPT-DENSE"), f("gen OPT-DENSE"))
    return V


def run():
    import multiprocessing as mp, budget
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x118_seed0"); per, history, bs, t0 = {}, [], C("x110_batch"), time.time()
    with mp.get_context("spawn").Pool(budget.workers()) as pool:
        for start in range(0, C("x110_max_worlds"), bs):
            seeds = list(range(seed0 + start, seed0 + start + bs))
            for s_, r in zip(seeds, pool.map(world, seeds, chunksize=1)): per[s_] = r
            S = summarize(per, seed0); V = judge(S); history.append([v[2] for v in V])
            print(f"{len(per)} worlds, {time.time() - t0:.0f} s:", " ".join(f"{v[0]}={v[2].split()[0]}" for v in V), flush=True)
            if len(history) >= 2 and history[-1] == history[-2]: break
    S["stopped_at_worlds"] = len(per)
    out = {"per_world": per, "summary": S, "verdicts": V}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_18_run.json"), "w"), indent=1, default=float)
    log_wandb(out)
    for v in V: print(v[0], v[2], "|", v[3])


def log_wandb(out):
    from track import Tracker
    S = out["summary"]
    arms = {"OPT": ("Optimised flat learner", "Keeps everything it learns."), "FRESH": ("Same learner, no memory", "Starts from nothing on every problem (control)."),
            "DENSE": ("Plain neural network", "32 shared units.")}
    tk = Tracker("X1.18", "A finite world, then one expansion", "In a fixed, finite world does the learner get better with experience (against a no-memory control), and does that knowledge help once the world expands?",
                 "Phase A: 60 problems from a fixed library of 24 hidden concepts (nothing new appears). Phase B: the library grows once by 12 concepts built on the old ones; 30 more problems. Every example is new.",
                 arms, "Share of ceiling = bits held on unseen examples divided by what the true concepts allow with the same examples.", dict(worlds=S["worlds"]), tags=["finite-world", "vertical", "expansion"])
    keys = [("relA_early", "Phase A earlier: share of ceiling"), ("relA_late", "Phase A later: share of ceiling"), ("relB_new", "Phase B, new concepts: share"), ("relB_old", "Phase B, old only: share"),
            ("gen", "Generalization (bits)"), ("final", "All problems at the end (bits)")]
    tk.table("Summary by arm (median over worlds)", ["Arm"] + [k[1] for k in keys], [[tk.name(a)] + [round(S["median"][a][k], 4) for k, _ in keys] for a in ARMS])
    tk.table("Paired differences with 95% ranges", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


if __name__ == "__main__":
    {"run": run, "log": lambda: log_wandb(json.load(open(os.path.join(OUT, "x1_18_run.json")))["result"])}[sys.argv[1]]()
