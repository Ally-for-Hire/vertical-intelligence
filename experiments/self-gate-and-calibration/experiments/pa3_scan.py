"""PIECE-A3: what must a brain scan contain? (user assumption A3: the start is a perfect human brain scan; Bargmann & Marder 2013 [v]: a connectome is necessary, not sufficient)
'True brain' on each of 40 real FlyWire subgraphs (300 neurons; q19_subgraphs.npz, held-out set 20-59):
   s_i(t+1) = (1 - leak) s_i(t) + leak * tanh( m(t) * g_i * (sum_j sign_j w_ji s_j(t) / in_i) + b_i + inject_i(t) )
with ALL ingredients: wiring (who connects), synapse strengths (counts), signs (dominant transmitter), intrinsic per-neuron gain g_i and bias b_i (heterogeneous), and a hidden NEUROMODULATORY state m(t)
(slow random drift around 1, shared by all neurons). Inputs: 3 action channels injected into 3 neurons, held 5 ticks, as in q19.
'Scans' (emulations) run on the same inputs, each MISSING one ingredient (set to its default): full; no_signs (all excitatory); binary (wiring only, all strengths equal); no_intrinsic (g = 2, b = 0 for all);
no_neuromod (m = 1); plus measurement error on a full scan (5% and 20% multiplicative noise on strengths). Fidelity = mean over the 50 core 'readout' neurons of the correlation between emulated and true activity over 500 ticks
(deterministic: no process noise), averaged over 40 subgraphs. Constants (leak 0.3, gain 2, heterogeneity sd 0.3, neuromod drift) are simulation settings, untuned.
PREDICTIONS (registered before running): P63 no_signs loses >= 50% of the full scan's fidelity; P64 binary loses >= 20%; P65 no_intrinsic loses >= 20%; P66 no_neuromod loses >= 10%; P67 the full scan with 5% strength error keeps >= 90% of full fidelity."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
import numpy as np
from multiprocessing import Pool

Z = np.load("../results/q19_subgraphs.npz")
N_OBS, N_ACT, LEAK, GAIN, T = 50, 3, 0.3, 2.0, 500


def simulate(Wn, g, b, m, inj, acts, s0):
    s = s0.copy(); out = np.zeros((T, N_OBS))
    for t in range(T):
        drive = m[t] * g * (s @ Wn) + b
        drive[inj] += acts[t]
        s = (1 - LEAK) * s + LEAK * np.tanh(drive)
        out[t] = s[:N_OBS]
    return out


def fidelity(a, b):
    c = [np.corrcoef(a[:, i], b[:, i])[0, 1] for i in range(N_OBS) if a[:, i].std() > 1e-9 and b[:, i].std() > 1e-9]
    return float(np.mean(c)) if c else 0.0


def run(args):
    gi, HET, NM, GN = args if isinstance(args, tuple) else (args, 0.3, 0.03, GAIN)
    W, sgn = Z["W"][gi].astype(float), Z["sign"][gi].astype(float)
    rng = np.random.default_rng(gi)
    n = W.shape[0]
    norm = W.sum(0, keepdims=True) + 1e-9
    Wtrue = (W * sgn[:, None]) / norm
    g = GN * np.exp(HET * rng.normal(size=n)); b = HET * rng.normal(size=n)
    m = np.ones(T); x = 0.0
    for t in range(T):
        x = 0.98 * x + NM * rng.normal(); m[t] = 1.0 + x
    inj = rng.choice(N_OBS, N_ACT, replace=False)
    acts = np.zeros((T, N_ACT)); a = np.zeros(N_ACT)
    for t in range(T):
        if t % 5 == 0:
            a = rng.integers(-1, 2, size=N_ACT).astype(float)
        acts[t] = a
    s0 = rng.normal(size=n) * 0.1
    true = simulate(Wtrue, g, b, m, inj, acts, s0)
    binW = (W > 0).astype(float)
    scans = {
        "full": (Wtrue, g, b, m),
        "no_signs": (W / norm, g, b, m),
        "binary": ((binW * sgn[:, None]) / (binW.sum(0, keepdims=True) + 1e-9), g, b, m),
        "no_intrinsic": (Wtrue, np.full(n, GN), np.zeros(n), m),
        "no_neuromod": (Wtrue, g, b, np.ones(T)),
    }
    for e in (0.05, 0.2):
        Wn = W * np.exp(e * rng.normal(size=W.shape))
        scans[f"full_err{int(e * 100)}"] = ((Wn * sgn[:, None]) / (Wn.sum(0, keepdims=True) + 1e-9), g, b, m)
    out = {}
    for k, (Wn, gg, bb, mm) in scans.items():
        out[k] = fidelity(true, simulate(Wn, gg, bb, mm, inj, acts, s0))
    out["true_std"] = float(true.std(0).mean())
    return out


if __name__ == "__main__" and os.environ.get("A3B"):
    t0 = time.time(); allr = {}
    with Pool(20) as p:
        for het in (0.1, 0.3, 0.6):
            for nm in (0.01, 0.03, 0.1):
                for gn in (1.0, 2.0, 4.0):
                    res = p.map(run, [(g, het, nm, gn) for g in range(20, 60)])
                    f = np.mean([r["full"] for r in res])
                    sh = {k: float(np.mean([r[k] for r in res]) / f) for k in ("no_signs", "binary", "no_intrinsic", "no_neuromod", "full_err5", "full_err20")}
                    allr[f"{het}_{nm}_{gn}"] = sh
                    order = sorted(("no_signs", "binary", "no_intrinsic", "no_neuromod"), key=lambda k: sh[k])
                    print(f"het {het} nm {nm} gain {gn}: " + " ".join(f"{k} {sh[k]:.2f}" for k in sh) + " | most damaging first: " + " > ".join(order))
    json.dump(allr, open("../results/pa3b_scan.json", "w"))
    print("done", round(time.time() - t0), "s")
    raise SystemExit
if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(0, 20) if os.environ.get("A3REP") else range(20, 60))
    json.dump(res, open("../results/pa3rep_scan.json" if os.environ.get("A3REP") else "../results/pa3_scan.json", "w"))
    full = np.array([r["full"] for r in res])
    print("scan            | fidelity (mean corr with the true brain's readout activity) | share of the full scan's fidelity")
    for k in res[0]:
        if k == "true_std":
            continue
        v = np.array([r[k] for r in res])
        print(f"{k:15s} | {v.mean():.3f} +- {v.std(ddof=1) / np.sqrt(len(v)):.3f} | {v.mean() / full.mean():.2f}")
    print("mean activity sd of the true brain's readout neurons: %.3f" % np.mean([r["true_std"] for r in res]))
    print("done", round(time.time() - t0), "s")
