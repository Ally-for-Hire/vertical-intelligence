"""PIECE-P5c: how much evidence does recognition need? Diagnostic for the P5/P5b/B9 failures.
For each world set (24), train one 'expert' chunk per problem (10 problems) on 5 episodes of that problem (identity known). Then, on 5 FRESH episodes of each problem, score all 10 experts by their
cumulative prediction error (agent units, frozen) over the first w ticks, w in {10, 25, 50, 100, 200, 300}, and record (a) top-1 identification accuracy (is the right expert the best?),
(b) how often the right expert beats a fresh network trained on those same w ticks (the P5b acceptance test).
PREDICTIONS (registered before running): P30 identification accuracy at w = 50 is >= 90% (if it fails, error-based recognition from short windows is the bottleneck, not fragmentation);
P31 accuracy rises monotonically with w and reaches >= 98% by w = 300; P32 at w = 50 the right expert beats a fresh net in >= 90% of episodes."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pp5_mastery as P

WS = (10, 25, 50, 100, 200, 300)


def run(world_set):
    pseeds = [20000 + 100 * world_set + i for i in range(P.NP)]
    refs = [P.ref_sd(ps) for ps in pseeds]
    experts = []
    for i, ps in enumerate(pseeds):
        ch = P.Chunk(18, 6, 1000 * world_set + i)
        for ep in range(5):
            X, D = P.episode_data(ps, 777000 + 100 * world_set + 10 * i + ep)
            for t in range(P.EP):
                ch.step(X[t], D[t], refs[i])
        experts.append(ch)
    out = {w: dict(right=0, beats_fresh=0, right_raw=0, beats_raw=0, right_ev=0, beats_ev=0, n=0) for w in WS}
    for i, ps in enumerate(pseeds):
        for ep in range(5):
            X, D = P.episode_data(ps, 888000 + 100 * world_set + 10 * i + ep)
            sce = np.zeros((P.NP, P.EP)); fre = np.zeros(P.EP); sc = np.zeros((P.NP, P.EP)); fr = np.zeros(P.EP); scr = np.zeros((P.NP, P.EP)); frr = np.zeros(P.EP)
            fresh = P.Chunk(18, 6, 5000 + i)
            from nn import RunStd
            dv = RunStd(6)                                               # P5d: ONE shared, data-based scale for every candidate (the agent's own running variance of this episode's targets)
            for t in range(P.EP):
                dv.update(D[t]); v = np.maximum(dv.s2, 1e-6) if t > 0 else np.ones(6)
                for j, c in enumerate(experts):
                    sc[j, t], sce[j, t] = P.chunk_error_only(c, X[t], D[t], refs[i])
                    pr = c._raw(np.array([n.predict(c.sx(X[t]))[0][0] for n in c.nets]))
                    scr[j, t] = float(np.mean((pr - D[t]) ** 2 / v))
                fresh.sx.update(X[t]); fresh.sy.update(D[t])
                xs, ys = fresh.sx(X[t]), fresh.sy(D[t])
                pf = np.array([n.step(xs, ys[q:q + 1])[0] for q, n in enumerate(fresh.nets)])
                fr[t] = float(np.mean((pf - ys) ** 2))
                frr[t] = float(np.mean((fresh._raw(pf) - D[t]) ** 2 / v))
                fre[t] = float(np.mean(((fresh._raw(pf) - D[t]) / refs[i]) ** 2))
            cs, cf = np.cumsum(sc, 1), np.cumsum(fr); csr, cfr = np.cumsum(scr, 1), np.cumsum(frr); cse, cfe = np.cumsum(sce, 1), np.cumsum(fre)
            for w in WS:
                out[w]["right"] += int(np.argmin(cs[:, w - 1]) == i)
                out[w]["beats_fresh"] += int(cs[i, w - 1] < cf[w - 1])
                out[w]["right_raw"] += int(np.argmin(csr[:, w - 1]) == i)
                out[w]["beats_raw"] += int(csr[i, w - 1] < cfr[w - 1])
                out[w]["right_ev"] += int(np.argmin(cse[:, w - 1]) == i)
                out[w]["beats_ev"] += int(cse[i, w - 1] < cfe[w - 1])
                out[w]["n"] += 1
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(24))
    json.dump(res, open("../results/pp5c_recognition%s.json" % ("_burn" if os.environ.get("BURN") else ""), "w"), default=int)
    print("window w | OWN-SCALE scoring: top-1 identification (chance 0.10) | right expert beats fresh || SHARED DATA SCALE (P5d): top-1 | beats fresh || EVALUATOR SCALE (diagnostic): top-1 | beats fresh")
    for w in WS:
        r = sum(x[w]["right"] for x in res); b = sum(x[w]["beats_fresh"] for x in res); n = sum(x[w]["n"] for x in res)
        rr = sum(x[w]["right_raw"] for x in res); br = sum(x[w]["beats_raw"] for x in res)
        re_ = sum(x[w]["right_ev"] for x in res); be = sum(x[w]["beats_ev"] for x in res)
        print(f"  {w:4d} | {r / n:.3f} | {b / n:.3f} || {rr / n:.3f} | {br / n:.3f} || {re_ / n:.3f} | {be / n:.3f}   (n={n})")
    print("done", round(time.time() - t0), "s")
