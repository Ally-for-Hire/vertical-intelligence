"""X1.17: the optimised flat learner in the near-infinite world (vi/winf.py), and an evidence bar that does not grow with what the learner knows.

World: winf (nothing ever repeats; an endless, compositional concept library with popularity-driven reuse); 60 problems per stream.   # CONST: winf_cfg
Arms: OPT (flat.OPT: a link must beat log2(#candidates + 1) bits, a charge that grows with the number of nodes)
      OPT_CONF (flat.OPT with bar='confirm': a link chosen on one half of the decision points must also help on the other half; no charge)
      DENSE (plain network).
Fair vertical: share of each problem's own ceiling (true-concept readout, same examples), later half minus earlier half.
Usage: python x1_17.py run | log"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import flat, deep, winf
import x1_13 as Q
import x1_12 as G

OUT = os.path.join(str(vipaths.RESULTS))
ARMS = ("OPT", "OPT_CONF", "DENSE")
CEIL = dict(deep.OPT, ceiling_ridge=C("x116_ceiling_ridge"))


def stats(r, ca, cg, half):
    ra = [x / c for x, c in zip(r["after"], ca)]
    return dict(late=float(np.mean(r["after"][half:])), early=float(np.mean(r["after"][:half])), final=float(np.mean(r["final"])), gen=float(np.mean(r["gen"])),
                rel_late=float(np.mean(ra[half:])), rel_early=float(np.mean(ra[:half])), rel_gen=float(np.mean([x / c for x, c in zip(r["gen"], cg)])),
                nodes=r["nodes"], links=r.get("links", 0))


def world(seed):
    t0 = time.time()
    data, gens, info = winf.make_world(seed); half = len(data) // 2
    ca = [deep.ceiling(d, CEIL) for d in data]; cg = [deep.ceiling(d, CEIL) for d in gens]
    tr = {np.round(x, 12).tobytes() for d in data + gens for x in np.vstack([d["tr"][0], d["va"][0]])}   # CONST: winf_cfg
    repeats = sum(np.round(x, 12).tobytes() in tr for d in data + gens for x in d["te"][0])   # CONST: winf_cfg
    out = {"world": dict(info, repeats=int(repeats), ceil_late=float(np.mean(ca[half:])), ceil_early=float(np.mean(ca[:half])))}
    out["OPT"] = stats(flat.run(data, gens, seed, flat.OPT), ca, cg, half)
    out["OPT_CONF"] = stats(flat.run(data, gens, seed, dict(flat.OPT, bar="confirm")), ca, cg, half)
    Q.CTX.update(mode="FLAT", links={}, origin={}, far=[], usage=False, use={})
    out["DENSE"] = stats(G.run_dense_gen(data, gens, seed), ca, cg, half)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(per, seed0):
    from x0_1 import boot_ci
    seeds = sorted(per); g = lambda a, k: np.array([per[s][a][k] for s in seeds], float)
    S = {"median": {a: {k: float(np.median(g(a, k))) for k in per[seeds[0]][a]} for a in ARMS}, "paired": {}, "worlds": len(seeds)}
    S["world"] = {k: float(np.median([per[s]["world"][k] for s in seeds])) for k in ("concepts", "max_depth", "mean_depth", "ceil_late", "ceil_early")}
    S["world"]["repeats_total"] = int(sum(per[s]["world"]["repeats"] for s in seeds))
    for k in ("late", "gen", "final", "rel_late"): S["paired"][f"{k} OPT_CONF-OPT"] = boot_ci(g("OPT_CONF", k) - g("OPT", k), seed0)
    S["paired"]["gen OPT-DENSE"] = boot_ci(g("OPT", "gen") - g("DENSE", "gen"), seed0)
    for a in ARMS: S["paired"][f"vertical {a}"] = boot_ci(g(a, "rel_late") - g(a, "rel_early"), seed0)
    S["paired"]["vertical OPT_CONF minus vertical OPT"] = boot_ci((g("OPT_CONF", "rel_late") - g("OPT_CONF", "rel_early")) - (g("OPT", "rel_late") - g("OPT", "rel_early")), seed0)
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run. A win = paired 95% range on the predicted side of 0."""
    pr, w = S["paired"], S["world"]; f = lambda k: f"{pr[k][0]:+.4f} [{pr[k][1]:+.4f}, {pr[k][2]:+.4f}]"; up = lambda k: pr[k][1] > 0
    manip = w["repeats_total"] == 0 and w["concepts"] > 2 * C("x117_pool0_ref")
    V = [("M-X1.17", "Fair test: no test point in any world was ever shown before (memorising is useless), and the concept library keeps growing (median concepts per stream > 2 x the starting pool)",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"repeats {w['repeats_total']}; concepts {w['concepts']:.0f}; max depth {w['max_depth']:.0f}; ceilings later/earlier {w['ceil_late']:.3f}/{w['ceil_early']:.3f}")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P448", "With the confirm bar the learner gets better with experience in the infinite world: OPT_CONF share of ceiling, later minus earlier, above 0", up("vertical OPT_CONF"), f("vertical OPT_CONF"))
    v("P449", "The confirm bar improves the experience trend over the growing charge: (OPT_CONF trend) minus (OPT trend), above 0", up("vertical OPT_CONF minus vertical OPT"), f("vertical OPT_CONF minus vertical OPT"))
    v("P450", "The confirm bar helps later problems: OPT_CONF minus OPT, above 0", up("late OPT_CONF-OPT"), f("late OPT_CONF-OPT"))
    v("P451", "The confirm bar helps generalization: OPT_CONF minus OPT, above 0", up("gen OPT_CONF-OPT"), f("gen OPT_CONF-OPT"))
    v("P452", "In the infinite world the flat learner still generalizes better than the plain network: OPT minus DENSE, above 0", up("gen OPT-DENSE"), f("gen OPT-DENSE"))
    v("P453", "The optimised flat learner (growing charge) gets better with experience in the infinite world: OPT trend above 0", up("vertical OPT"), f("vertical OPT"))
    return V


def run():
    import multiprocessing as mp, budget
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x117_seed0"); per, history, bs, t0 = {}, [], C("x110_batch"), time.time()
    with mp.get_context("spawn").Pool(budget.workers()) as pool:
        for start in range(0, C("x110_max_worlds"), bs):
            seeds = list(range(seed0 + start, seed0 + start + bs))
            for s_, r in zip(seeds, pool.map(world, seeds, chunksize=1)): per[s_] = r
            S = summarize(per, seed0); V = judge(S); history.append([v[2] for v in V])
            print(f"{len(per)} worlds, {time.time() - t0:.0f} s:", " ".join(f"{v[0]}={v[2].split()[0]}" for v in V), flush=True)
            if len(history) >= 2 and history[-1] == history[-2]: break
    S["stopped_at_worlds"] = len(per)
    out = {"per_world": per, "summary": S, "verdicts": V}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_17_run.json"), "w"), indent=1, default=float)
    log_wandb(out)
    for v in V: print(v[0], v[2], "|", v[3])


def log_wandb(out):
    from track import Tracker
    S = out["summary"]
    arms = {"OPT": ("Optimised flat learner", "A link must beat a charge that grows with how many nodes it chose from."),
            "OPT_CONF": ("Optimised flat learner, confirm bar", "A link must help on two separate halves of the held-back examples; no charge."),
            "DENSE": ("Plain neural network", "32 shared units.")}
    tk = Tracker("X1.17", "A near-infinite world, and a bar that does not punish knowing more",
                 "In a world where nothing ever repeats and concepts keep growing, does the learner get better with experience, and does replacing the growing evidence charge help?",
                 "An endless stream of prediction problems built from an ever-growing library of hidden concepts, most built on earlier ones; every example is new, so memorising is useless.",
                 arms, "Bits per point on unseen examples; 'share of ceiling' = bits divided by what the true hidden concepts allow with the same examples.", dict(worlds=S["worlds"]), tags=["infinite-world", "vertical"])
    keys = [("late", "Later problems (bits)"), ("final", "All problems at the end"), ("gen", "Generalization"), ("rel_early", "Share of ceiling, earlier"), ("rel_late", "Share of ceiling, later"), ("nodes", "Nodes"), ("links", "Links")]
    tk.table("Summary by arm (median over worlds)", ["Arm"] + [k[1] for k in keys], [[tk.name(a)] + [round(S["median"][a][k], 4) for k, _ in keys] for a in ARMS])
    tk.table("Paired differences with 95% ranges", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


if __name__ == "__main__":
    {"run": run, "log": lambda: log_wandb(json.load(open(os.path.join(OUT, "x1_17_run.json")))["result"])}[sys.argv[1]]()
