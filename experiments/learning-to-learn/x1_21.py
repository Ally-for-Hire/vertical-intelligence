"""X1.21: the train-and-expand loop with a LOW-NOISE internal progress signal (X1.19-X1.20's per-problem comparison could not resolve slow progress).

Probe = the first x121_probe problems of the current stage, fixed for the stage. After each later problem the learner re-scores itself on the probe:
mean bits on the probe problems' decision points with its CURRENT nodes and links (readouts re-fitted on their train points). Internal only (no test
data, no ceilings). The world expands when the probe score is not above its value x121_lag problems earlier and the stage has lasted at least
x121_probe + x121_lag problems; a stage ends at x120_max_stage. Everything else as X1.19/X1.20.
Usage: python x1_21.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import flat, deep, wstream
import x1_19 as L19

OUT = L19.OUT
WORLDS = os.path.join(str(vipaths.WORLDS), "x1_21")


def probe_score(L, idx):
    tot = 0.0
    for s in idx:
        c = L.caches[s]; F, Fv, Gi, beta = flat.solve(c, L.heads[s], L.cfg["ridge"])
        tot += float(flat.vbits(c.yva - Fv @ beta, len(c.yva))) / len(c.yva)
    return tot / len(idx)


def world(seed):
    t0 = time.time()
    W = wstream.Stream(seed); L = flat.Learner(seed, flat.OPT)
    npb, lag, nexp, mx = C("x121_probe"), C("x121_lag"), C("x119_expand"), C("x120_max_stage")
    rows, start, score = [], 0, {}
    for t in range(C("x120_problems")):
        d = W.next()
        bits, ob = L.step(d)
        fresh = flat.run([d], [], seed, flat.OPT)["after"][0]; c = deep.ceiling(d, L19.CEIL)
        n = t + 1 - start
        if n >= npb: score[t] = probe_score(L, list(range(start, start + npb)))
        rows.append(dict(t=t, stage=d["stage"], has_new=d["has_new"], share=bits / c, fresh=fresh / c, own=ob, probe=score.get(t)))
        if (n >= npb + lag and score[t] <= score[t - lag]) or n >= mx:
            W.expand(nexp); start = t + 1; score = {}
    W.save(os.path.join(WORLDS, f"world_{seed}.json"))
    return dict(rows=rows, events=W.events, concepts=len(W.lib.c), nodes=len(L.units), seconds=round(time.time() - t0, 1))


def run():
    import multiprocessing as mp, budget
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    os.makedirs(WORLDS, exist_ok=True)
    seed0 = C("x121_seed0"); t0 = time.time(); seeds = list(range(seed0, seed0 + C("x119_worlds")))
    with mp.get_context("spawn").Pool(budget.workers()) as pool: res = pool.map(world, seeds, chunksize=1)
    per = dict(zip(seeds, res)); S = L19.summarize(per, seed0)
    ren = {"X1.19": "X1.21", "P460": "P468", "P461": "P469", "P462": "P470", "P463": "P471"}
    V = []
    for v in L19.judge(S):
        a = v[0]
        for k_, r_ in ren.items(): a = a.replace(k_, r_)
        V.append((a,) + tuple(v[1:]))
    S["stage_length_median"] = float(np.median([b["length"] for b in S["by_stage"].values()]))
    S["stage_lengths_all"] = [int(sum(1 for x in per[s]["rows"] if x["stage"] == k)) for s in per for k in range(len(per[s]["events"]))]
    S["probe_gain_per_stage"] = float(np.median([per[s]["rows"][i]["probe"] - next(x["probe"] for x in per[s]["rows"] if x["stage"] == per[s]["rows"][i]["stage"] and x["probe"] is not None)
                                                 for s in per for i in range(1, len(per[s]["rows"]))
                                                 if per[s]["rows"][i]["probe"] is not None and (i + 1 == len(per[s]["rows"]) or per[s]["rows"][i + 1]["stage"] != per[s]["rows"][i]["stage"])]))
    out = {"per_world": per, "summary": S, "verdicts": V}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_21_run.json"), "w"), indent=1, default=float)
    print(json.dumps({k: S[k] for k in ("expansions_median", "concepts_median", "share_all", "fresh_all", "stage_length_median", "probe_gain_per_stage")}, default=float))
    print("stage lengths: min", min(S["stage_lengths_all"]), "median", int(np.median(S["stage_lengths_all"])), "max", max(S["stage_lengths_all"]))
    for k, b in S["by_stage"].items(): print(f"stage {k}: worlds {b['worlds']} length {b['length']:.0f} first {b['first']:.3f} (no memory {b['first_fresh']:.3f}) end {b['end']:.3f}")
    for v in V: print(v[0], v[2], "|", v[3])
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
