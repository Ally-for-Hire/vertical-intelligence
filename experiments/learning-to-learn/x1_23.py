"""X1.23: is the X1.22 plateau representational? Break the post-plateau share of ceiling down by how deep each problem's concepts are.
Same long fixed world as X1.22 (wstream, 24 concepts, never expanded, 240 problems, 60 train points). For the last x123_tail problems: groups by the
deepest concept a problem uses (0 = only concepts built directly on the input; 2+ = at least one concept built on concepts built on concepts).
Usage: python x1_23.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import flat, deep, wstream
import x1_19 as L19

OUT = L19.OUT


def world(seed):
    t0 = time.time()
    W = wstream.Stream(seed); L = flat.Learner(seed, flat.OPT); rows = []
    for t in range(C("x122_problems")):
        d = W.next(); b, _ = L.step(d); f = flat.run([d], [], seed, flat.OPT)["after"][0]; c = deep.ceiling(d, L19.CEIL)
        rows.append(dict(depth=max(W.lib.c[s]["depth"] for s in d["S"]), share=b / c, fresh=f / c))
    tail = rows[-C("x123_tail"):]
    grp = lambda lo, hi: [r for r in tail if lo <= r["depth"] <= hi]
    out = {}
    for name, (lo, hi) in (("d0", (0, 0)), ("d1", (1, 1)), ("d2", (2, 99))):   # CONST: x123_tail
        G = grp(lo, hi); out[name] = dict(n=len(G), share=float(np.mean([r["share"] for r in G])) if G else float("nan"),
                                          fresh=float(np.mean([r["fresh"] for r in G])) if G else float("nan"))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run():
    import multiprocessing as mp, budget
    from x0_1 import boot_ci
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x123_seed0"); t0 = time.time(); seeds = list(range(seed0, seed0 + C("x119_worlds")))
    with mp.get_context("spawn").Pool(budget.workers()) as pool: res = pool.map(world, seeds, chunksize=1)
    per = dict(zip(seeds, res))
    ok = [s for s in seeds if per[s]["d0"]["n"] and per[s]["d2"]["n"]]
    g = lambda grp, k: np.array([per[s][grp][k] for s in ok])
    S = dict(worlds=len(seeds), worlds_with_both=len(ok),
             med={grp: dict(n=float(np.median([per[s][grp]["n"] for s in seeds])), share=float(np.nanmedian([per[s][grp]["share"] for s in seeds])),
                            fresh=float(np.nanmedian([per[s][grp]["fresh"] for s in seeds]))) for grp in ("d0", "d1", "d2")})
    S["deep_harder"] = boot_ci(g("d0", "share") - g("d2", "share"), seed0)
    S["memory_deep_vs_base"] = boot_ci((g("d2", "share") - g("d2", "fresh")) - (g("d0", "share") - g("d0", "fresh")), seed0)
    f = lambda k: f"{S[k][0]:+.4f} [{S[k][1]:+.4f}, {S[k][2]:+.4f}]"
    V = [("P476", "After the plateau, problems built only on base concepts reach a larger share of their ceiling than problems with a concept 2+ levels deep: base minus deep, above 0", "PASSED" if S["deep_harder"][1] > 0 else "FAILED", f("deep_harder")),
         ("P477", "Memory helps more on deep problems: (learner minus no memory on deep) minus (on base), above 0", "PASSED" if S["memory_deep_vs_base"][1] > 0 else "FAILED", f("memory_deep_vs_base"))]
    out = {"per_world": per, "summary": S, "verdicts": V}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_23_run.json"), "w"), indent=1, default=float)
    for grp in ("d0", "d1", "d2"): print(grp, S["med"][grp])
    print("worlds with both groups", len(ok))
    for v in V: print(v[0], v[2], "|", v[3])
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
