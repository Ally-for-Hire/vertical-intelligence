"""X1.20: the X1.19 loop with a stagnation detector that needs more evidence, so stages last until progress truly stops (X1.19's fired near its
30-problem minimum and left no time for mastery).

Rule: expand when the mean of the learner's OWN decision-point bits over the last x120_win problems is not above the mean over the x120_win before,
and the stage has lasted >= 2 x x120_win problems; a stage also ends at x120_max_stage problems. Everything else as X1.19 (vi/x1_19.py).
Usage: python x1_20.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import flat, deep, wstream
import x1_19 as L19

OUT = L19.OUT
WORLDS = os.path.join(str(vipaths.WORLDS), "x1_20")


def world(seed):
    t0 = time.time()
    W = wstream.Stream(seed); L = flat.Learner(seed, flat.OPT); win, nexp, mx = C("x120_win"), C("x119_expand"), C("x120_max_stage")
    rows, own, start = [], [], 0
    for t in range(C("x120_problems")):
        d = W.next()
        bits, ob = L.step(d); own.append(ob)
        fresh = flat.run([d], [], seed, flat.OPT)["after"][0]; c = deep.ceiling(d, L19.CEIL)
        rows.append(dict(t=t, stage=d["stage"], has_new=d["has_new"], share=bits / c, fresh=fresh / c, own=ob))
        n = t + 1 - start
        if (n >= 2 * win and np.mean(own[-win:]) <= np.mean(own[-2 * win:-win])) or n >= mx:
            W.expand(nexp); start = t + 1
    W.save(os.path.join(WORLDS, f"world_{seed}.json"))
    return dict(rows=rows, events=W.events, concepts=len(W.lib.c), nodes=len(L.units), seconds=round(time.time() - t0, 1))


def run():
    import multiprocessing as mp, budget
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    os.makedirs(WORLDS, exist_ok=True)
    seed0 = C("x120_seed0"); t0 = time.time(); seeds = list(range(seed0, seed0 + C("x119_worlds")))
    with mp.get_context("spawn").Pool(budget.workers()) as pool: res = pool.map(world, seeds, chunksize=1)
    per = dict(zip(seeds, res)); S = L19.summarize(per, seed0)
    V = [(v[0].replace("X1.19", "X1.20").replace("P460", "P464").replace("P461", "P465").replace("P462", "P466").replace("P463", "P467"),) + tuple(v[1:]) for v in L19.judge(S)]
    S["stage_length_median"] = float(np.median([b["length"] for b in S["by_stage"].values()]))
    out = {"per_world": per, "summary": S, "verdicts": V}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_20_run.json"), "w"), indent=1, default=float)
    print(json.dumps({k: S[k] for k in ("expansions_median", "concepts_median", "share_all", "fresh_all", "stage_length_median")}, default=float))
    for k, b in S["by_stage"].items(): print(f"stage {k}: worlds {b['worlds']} length {b['length']:.0f} first {b['first']:.3f} (no memory {b['first_fresh']:.3f}) end {b['end']:.3f}")
    for v in V: print(v[0], v[2], "|", v[3])
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
