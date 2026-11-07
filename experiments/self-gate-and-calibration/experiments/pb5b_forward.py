"""PIECE-B5b: forward models re-test with the controllability restriction PRE-REGISTERED.
Worlds: the first 48 seeds from 9000 upward whose TARGET is a core variable with an ACTION among its direct parents (decided from the generator's truth table before any controller runs).
3,000 steps. Controllers: random; model_free (as B5); model_based_1 (as B5: one-step greedy on a learned target-change model); model_based_3 (learns a FULL state-change model, observed state + action -> all
observed changes, and plans by random shooting: 30 random 3-step action sequences rolled through the learned model, executes the first action of the best); oracle (one-step plan with the true model).
Score: mean |s_target| early (steps 0-1000) and late (2000-3000).
PREDICTIONS (registered before running): P98 model_based_3 late error <= 0.8 x model_free late; P99 model_based_3 early error <= 0.8 x model_free early; P100 model_based_3 late error <= 1.5 x oracle late."""
import os, sys, json, time, copy, itertools
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

STEPS, SHOOT, H = 3000, 30, 3


def controllable(seed):
    from worlds import sample_world
    w = sample_world(seed)
    return w.target < w.n_core and any(t == "a" for (t, j) in w.parents[w.target])


def run(seed):
    from worlds import sample_world
    from nn import AdamMLP
    out = {}
    for ctrl in ("random", "model_free", "model_based_1", "model_based_3", "oracle"):
        w = sample_world(seed); rng = np.random.default_rng(seed + 5)
        na, ns = w.n_act, w.n_state; acts = [np.array(a, float) for a in itertools.product((-1, 0, 1), repeat=na)]
        pos = int(np.nonzero(w.perm == w.target)[0][0]); goal = float(w.offset[pos]); sc = float(w.scale[pos])
        errs = []; Q = np.zeros((9, 2, len(acts))); prev_s = None; prev_obs_t = None
        net1 = AdamMLP(ns + na, 1, 32, 3e-3, seed); netF = AdamMLP(ns + na, ns, 64, 3e-3, seed + 1); mx = None
        for t in range(STEPS):
            obs = w.observe(); e = (obs[pos] - goal) / sc
            if mx is None:
                mx = np.abs(obs) + 1.0
            if ctrl == "random":
                k = int(rng.integers(len(acts)))
            elif ctrl == "model_free":
                st = (int(np.clip(np.floor(e + 4.5), 0, 8)), int(prev_obs_t is not None and obs[pos] > prev_obs_t))
                if prev_s is not None:
                    Q[prev_s + (prev_k,)] += 0.1 * (-abs(e) + 0.9 * Q[st].max() - Q[prev_s + (prev_k,)])
                k = int(rng.integers(len(acts))) if rng.random() < 0.1 else int(np.argmax(Q[st]))
                prev_s, prev_k = st, k
            elif ctrl == "model_based_1":
                if rng.random() < 0.1 or t < 20:
                    k = int(rng.integers(len(acts)))
                else:
                    k = int(np.argmin([abs(obs[pos] + net1.predict(np.concatenate([obs / mx, a]))[0][0] * sc - goal) for a in acts]))
            elif ctrl == "model_based_3":
                if rng.random() < 0.1 or t < 20:
                    k = int(rng.integers(len(acts)))
                else:
                    best, k = np.inf, 0
                    for s_ in range(SHOOT):
                        seq = rng.integers(len(acts), size=H); o = obs.copy(); cost = 0.0
                        for h in seq:
                            o = o + netF.predict(np.concatenate([o / mx, acts[h]]))[0] * mx
                            cost += abs(o[pos] - goal)
                        if cost < best:
                            best, k = cost, int(seq[0])
            else:
                best, k = np.inf, 0
                for kk, a in enumerate(acts):
                    w2 = copy.deepcopy(w); w2.noise = 0.0; w2.step(w2.act_to_world(a))
                    if abs(w2.s[w.target]) < best:
                        best, k = abs(w2.s[w.target]), kk
            prev_obs_t = obs[pos]
            w.step(w.act_to_world(acts[k])); nobs = w.observe()
            if ctrl == "model_based_1":
                net1.step(np.concatenate([obs / mx, acts[k]]), np.array([(nobs[pos] - obs[pos]) / sc]))
            if ctrl == "model_based_3":
                netF.step(np.concatenate([obs / mx, acts[k]]), (nobs - obs) / mx)
            errs.append(abs(w.s[w.target]))
        out[ctrl] = dict(early=float(np.mean(errs[:1000])), late=float(np.mean(errs[2000:])))
    return out


if __name__ == "__main__":
    t0 = time.time()
    seeds = [s for s in range(9000, 9600) if controllable(s)][:48]
    with Pool(20) as p:
        res = p.map(run, seeds)
    json.dump(dict(seeds=seeds, res=res), open("../results/pb5b_forward.json", "w"))
    o = np.array([r["oracle"]["late"] for r in res])
    print(f"{len(seeds)} pre-registered controllable worlds (seeds {seeds[0]}..{seeds[-1]})")
    print("controller     | early (0-1000) | late (2000-3000) | late / oracle (median)")
    for c in ("random", "model_free", "model_based_1", "model_based_3", "oracle"):
        print(f"{c:14s} | {np.median([r[c]['early'] for r in res]):.3f} | {np.median([r[c]['late'] for r in res]):.3f} | {np.median(np.array([r[c]['late'] for r in res]) / o):.2f}")
    for m in ("late", "early"):
        print(f"model_based_3 / model_free {m}: median {np.median([r['model_based_3'][m] / r['model_free'][m] for r in res]):.2f}")
    print("done", round(time.time() - t0), "s")
