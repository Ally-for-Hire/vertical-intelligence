"""X1.22: what sets the ~36% plateau (X1.21)? A long FIXED finite world (24 concepts, never expanded, 240 problems), learning curves in blocks of 30,
at 60 and at 120 train points per problem (same seeds, same library, same problems and weights: wstream draws each problem's concepts and weights
from the world stream and its points from the problem's own seed), each with a no-memory control.
Usage: python x1_22.py run"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import flat, deep, wstream
import x1_19 as L19

OUT = L19.OUT


def curve(seed, n_tr):
    W = wstream.Stream(seed, dict(n_tr=n_tr)); L = flat.Learner(seed, flat.OPT); rows = []
    for t in range(C("x122_problems")):
        d = W.next(); b, _ = L.step(d); f = flat.run([d], [], seed, flat.OPT)["after"][0]; c = deep.ceiling(d, L19.CEIL)
        rows.append((b / c, f / c))
    return np.array(rows)


def world(seed):
    t0 = time.time(); out = {}
    for n in C("x122_ntr"):
        R = curve(seed, n); bs = C("x122_block")
        out[str(n)] = dict(blocks=[float(R[i:i + bs, 0].mean()) for i in range(0, len(R), bs)], fresh_blocks=[float(R[i:i + bs, 1].mean()) for i in range(0, len(R), bs)],
                           last=float(R[-2 * bs:, 0].mean()), last_fresh=float(R[-2 * bs:, 1].mean()))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run():
    import multiprocessing as mp, budget
    from x0_1 import boot_ci
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seed0 = C("x122_seed0"); t0 = time.time(); seeds = list(range(seed0, seed0 + C("x119_worlds")))
    with mp.get_context("spawn").Pool(budget.workers()) as pool: res = pool.map(world, seeds, chunksize=1)
    per = dict(zip(seeds, res)); a, b = (str(n) for n in C("x122_ntr"))
    g = lambda n, k: np.array([per[s][n][k] for s in seeds])
    blk = lambda n, i: np.array([per[s][n]["blocks"][i] for s in seeds])
    S = dict(worlds=len(seeds), curve={n: [float(np.median([per[s][n]["blocks"][i] for s in seeds])) for i in range(len(per[seeds[0]][n]["blocks"]))] for n in (a, b)},
             fresh_curve={n: [float(np.median([per[s][n]["fresh_blocks"][i] for s in seeds])) for i in range(len(per[seeds[0]][n]["blocks"]))] for n in (a, b)})
    S["rise_late"] = boot_ci(blk(a, -1) - blk(a, C("x122_mid_block")), seed0)
    S["more_data"] = boot_ci(g(b, "last") - g(a, "last"), seed0)
    S["memory_vs_data"] = boot_ci((g(a, "last") - g(a, "last_fresh")) - (g(b, "last") - g(b, "last_fresh")), seed0)
    S["last_median"] = float(np.median(g(a, "last")))
    f = lambda k: f"{S[k][0]:+.4f} [{S[k][1]:+.4f}, {S[k][2]:+.4f}]"
    V = [("P472", "In a long fixed world the learner is still improving late: last block (problems 211-240) minus block 4 (91-120), share of ceiling, above 0", "PASSED" if S["rise_late"][1] > 0 else "FAILED", f("rise_late")),
         ("P473", "More examples per problem raise the level: share on the last 60 problems at 120 train points minus at 60, above 0", "PASSED" if S["more_data"][1] > 0 else "FAILED", f("more_data")),
         ("P474", "Memory matters more when examples are scarce: (memory advantage at 60) minus (at 120), last 60 problems, above 0", "PASSED" if S["memory_vs_data"][1] > 0 else "FAILED", f("memory_vs_data")),
         ("P475", "The plateau is far from the ceiling: median share on the last 60 problems at 60 train points below 0.5", "PASSED" if S["last_median"] < C("x122_far") else "FAILED", f"{S['last_median']:.3f}")]
    out = {"per_world": per, "summary": S, "verdicts": V}
    json.dump({"seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x1_22_run.json"), "w"), indent=1, default=float)
    for n in (a, b): print(f"n_tr {n}: learner", " ".join(f"{x:.3f}" for x in S["curve"][n]), "| no memory", " ".join(f"{x:.3f}" for x in S["fresh_curve"][n]))
    for v in V: print(v[0], v[2], "|", v[3])
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
