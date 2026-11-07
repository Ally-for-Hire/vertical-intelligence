"""PIECE-B5: forward models (predict the consequences of one's own actions; cerebellum; the user: 'learn how its own effects on variables affect its surroundings').
Task: in a causal world (generator v2, exp M family, 48 worlds) keep the world's TARGET variable at its setpoint (score = -|s_target|). Actions: every combination of {-1, 0, +1} per action channel.
Controllers (1,000 steps each):
  random      : uniform random actions
  model_free  : tabular Q-learning; state = (target error in 9 bins, sign of its last change); reward = -|s_target|; alpha 0.1, gamma 0.9, epsilon 0.1
  model_based : learns a FORWARD MODEL (online MLP: observed state + action -> change of the target's observed value), then picks the action whose predicted next target value is closest to the setpoint;
                10% random actions (EXPERIMENT setting)
  oracle      : plans one step ahead with the TRUE noise-free world model (upper reference)
Score: mean |s_target| over steps 0-500 (early) and 500-1000 (late).
PREDICTIONS (registered before running): P95 model_based early error <= 0.7 x model_free early; P96 model_based late error <= 1.3 x oracle late; P97 model_free late error <= 0.8 x random late."""
import os, sys, json, time, copy, itertools
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

STEPS = 1000


def run(seed):
    from worlds import sample_world
    from nn import AdamMLP
    out = {}
    for ctrl in ("random", "model_free", "model_based", "oracle"):
        w = sample_world(seed); rng = np.random.default_rng(seed + 5)
        na = w.n_act; acts = [np.array(a, float) for a in itertools.product((-1, 0, 1), repeat=na)]
        pos = int(np.nonzero(w.perm == w.target)[0][0]); goal = float(w.offset[pos]); sc = float(w.scale[pos])
        errs = []; Q = np.zeros((9, 2, len(acts))); prev_s = None; prev_obs_t = None
        net = AdamMLP(w.n_state + na, 1, 32, 3e-3, seed) if ctrl == "model_based" else None
        mx = None
        for t in range(STEPS):
            obs = w.observe(); e = (obs[pos] - goal) / sc
            if ctrl == "random":
                k = int(rng.integers(len(acts)))
            elif ctrl == "model_free":
                st = (int(np.clip(np.floor(e + 4.5), 0, 8)), int(prev_obs_t is not None and obs[pos] > prev_obs_t))
                if prev_s is not None:
                    r = -abs(e); Q[prev_s + (prev_k,)] += 0.1 * (r + 0.9 * Q[st].max() - Q[prev_s + (prev_k,)])
                k = int(rng.integers(len(acts))) if rng.random() < 0.1 else int(np.argmax(Q[st]))
                prev_s, prev_k = st, k
            elif ctrl == "model_based":
                if mx is None:
                    mx = np.abs(obs) + 1.0
                if rng.random() < 0.1 or t < 20:
                    k = int(rng.integers(len(acts)))
                else:
                    preds = [obs[pos] + net.predict(np.concatenate([obs / mx, a]))[0][0] * sc for a in acts]
                    k = int(np.argmin([abs(p - goal) for p in preds]))
            else:
                best, bk = np.inf, 0
                for kk, a in enumerate(acts):
                    w2 = copy.deepcopy(w); w2.noise = 0.0; w2.step(w2.act_to_world(a))
                    v = abs(w2.s[w.target])
                    if v < best:
                        best, bk = v, kk
                k = bk
            prev_obs_t = obs[pos]
            w.step(w.act_to_world(acts[k]))
            if ctrl == "model_based":
                net.step(np.concatenate([obs / mx, acts[k]]), np.array([(w.observe()[pos] - obs[pos]) / sc]))
            errs.append(abs(w.s[w.target]))
        out[ctrl] = dict(early=float(np.mean(errs[:500])), late=float(np.mean(errs[500:])))
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(9000, 9048))
    json.dump(res, open("../results/pb5_forward.json", "w"))
    print("controller   | mean |target| early (0-500) | late (500-1000) | late / oracle (median ratio)")
    o = np.array([r["oracle"]["late"] for r in res])
    for c in ("random", "model_free", "model_based", "oracle"):
        e = np.array([r[c]["early"] for r in res]); l = np.array([r[c]["late"] for r in res])
        print(f"{c:12s} | {np.median(e):.3f} | {np.median(l):.3f} | {np.median(l / o):.2f}")
    mb = np.array([r["model_based"]["early"] / r["model_free"]["early"] for r in res]); mf = np.array([r["model_free"]["late"] / r["random"]["late"] for r in res])
    print("model_based / model_free early: median %.2f; model_free / random late: median %.2f" % (np.median(mb), np.median(mf)))
    print("done", round(time.time() - t0), "s")
