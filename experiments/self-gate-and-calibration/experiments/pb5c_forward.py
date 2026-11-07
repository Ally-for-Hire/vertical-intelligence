"""PIECE-B5c: focused vs full forward models, and the wiring-energy cost (C41) as the fix for a full model (pieces B5 + B11 combined).
Same pre-registered controllable worlds and settings as B5b (48 worlds, 3,000 steps). Controllers: model_free; focused_1 (B5b's model_based_1); full_1 (full-state model, one-step plan);
full_3 (full-state model, 3-step random shooting, = B5b's model_based_3); fullL1_1 and fullL1_3 (the same full models with the replicated wiring cost: proximal L1 shrink, lam = 0.1, as C41); oracle.
PREDICTIONS (registered before running): P101 focused_1 late error <= 0.85 x full_3 late (replicates the B5b exploratory 0.74); P102 focused_1 early error <= 0.85 x model_free early (replicates 0.77);
P103 the wiring cost helps the full 3-step planner: fullL1_3 late <= 0.9 x full_3 late; P104 fullL1_1 late within 10% of focused_1 late (a wiring-budgeted full model learns to focus by itself)."""
import os, sys, json, time, copy, itertools
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pb5b_forward as B

LAM = 0.1
CTRLS = ("model_free", "focused_1", "full_1", "full_3", "fullL1_1", "fullL1_3", "oracle")


def l1(net):
    for P in (net.W1, net.W2):
        P[...] = np.sign(P) * np.maximum(np.abs(P) - net.lr * LAM, 0.0)


def run(seed):
    from worlds import sample_world
    from nn import AdamMLP
    out = {}
    for ctrl in CTRLS:
        w = sample_world(seed); rng = np.random.default_rng(seed + 5)
        na, ns = w.n_act, w.n_state; acts = [np.array(a, float) for a in itertools.product((-1, 0, 1), repeat=na)]
        pos = int(np.nonzero(w.perm == w.target)[0][0]); goal = float(w.offset[pos]); sc = float(w.scale[pos])
        errs = []; Q = np.zeros((9, 2, len(acts))); prev_s = None; prev_obs_t = None; mx = None
        net1 = AdamMLP(ns + na, 1, 32, 3e-3, seed); netF = AdamMLP(ns + na, ns, 64, 3e-3, seed + 1)
        H = 3 if ctrl.endswith("_3") else 1
        for t in range(B.STEPS):
            obs = w.observe(); e = (obs[pos] - goal) / sc
            if mx is None:
                mx = np.abs(obs) + 1.0
            if ctrl == "model_free":
                st = (int(np.clip(np.floor(e + 4.5), 0, 8)), int(prev_obs_t is not None and obs[pos] > prev_obs_t))
                if prev_s is not None:
                    Q[prev_s + (prev_k,)] += 0.1 * (-abs(e) + 0.9 * Q[st].max() - Q[prev_s + (prev_k,)])
                k = int(rng.integers(len(acts))) if rng.random() < 0.1 else int(np.argmax(Q[st])); prev_s, prev_k = st, k
            elif ctrl == "oracle":
                best, k = np.inf, 0
                for kk, a in enumerate(acts):
                    w2 = copy.deepcopy(w); w2.noise = 0.0; w2.step(w2.act_to_world(a))
                    if abs(w2.s[w.target]) < best:
                        best, k = abs(w2.s[w.target]), kk
            elif rng.random() < 0.1 or t < 20:
                k = int(rng.integers(len(acts)))
            elif ctrl == "focused_1":
                k = int(np.argmin([abs(obs[pos] + net1.predict(np.concatenate([obs / mx, a]))[0][0] * sc - goal) for a in acts]))
            elif H == 1:
                k = int(np.argmin([abs(obs[pos] + netF.predict(np.concatenate([obs / mx, a]))[0][pos] * mx[pos] - goal) for a in acts]))
            else:
                best, k = np.inf, 0
                for s_ in range(B.SHOOT):
                    seq = rng.integers(len(acts), size=H); o = obs.copy(); cost = 0.0
                    for h in seq:
                        o = o + netF.predict(np.concatenate([o / mx, acts[h]]))[0] * mx; cost += abs(o[pos] - goal)
                    if cost < best:
                        best, k = cost, int(seq[0])
            prev_obs_t = obs[pos]
            w.step(w.act_to_world(acts[k])); nobs = w.observe()
            if ctrl == "focused_1":
                net1.step(np.concatenate([obs / mx, acts[k]]), np.array([(nobs[pos] - obs[pos]) / sc]))
            elif ctrl.startswith("full"):
                netF.step(np.concatenate([obs / mx, acts[k]]), (nobs - obs) / mx)
                if ctrl.startswith("fullL1"):
                    l1(netF)
            errs.append(abs(w.s[w.target]))
        out[ctrl] = dict(early=float(np.mean(errs[:1000])), late=float(np.mean(errs[2000:])))
    return out


if __name__ == "__main__":
    t0 = time.time()
    seeds = [s for s in range(9110, 9800) if B.controllable(s)][:48] if os.environ.get("B5D") else [s for s in range(9000, 9600) if B.controllable(s)][:48]
    with Pool(20) as p:
        res = p.map(run, seeds)
    json.dump(dict(seeds=seeds, res=res), open("../results/pb5d_forward.json" if os.environ.get("B5D") else "../results/pb5c_forward.json", "w"))
    o = np.array([r["oracle"]["late"] for r in res])
    print("controller | early | late | late x oracle (medians)")
    for c in CTRLS:
        print(f"{c:10s} | {np.median([r[c]['early'] for r in res]):.3f} | {np.median([r[c]['late'] for r in res]):.3f} | {np.median(np.array([r[c]['late'] for r in res]) / o):.2f}")
    m = lambda a, b, k: np.median([r[a][k] / r[b][k] for r in res])
    print("focused_1/full_3 late %.2f | focused_1/model_free early %.2f | fullL1_3/full_3 late %.2f | fullL1_1/focused_1 late %.2f" % (m("focused_1", "full_3", "late"), m("focused_1", "model_free", "early"), m("fullL1_3", "full_3", "late"), m("fullL1_1", "focused_1", "late")))
    print("done", round(time.time() - t0), "s")
