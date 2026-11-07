"""PIECE-B9e: sleep under an AWAKE CONSTRAINT, and self-timed sleep (the user's NeuroBlock idea: sleep when pressure builds; synaptic homeostasis, Tononi & Cirelli 2014 [v]).
Constraint (modelled, stated openly): learning saturates. Each output network accumulates potentiation load L (sum of absolute weight changes); its effective learning rate is lr / (1 + L / L0). Only SLEEP resets L,
and a sleep lasts 50 ticks during which the network does not learn (it still predicts with frozen weights; those errors count). L0 = 2 (EXPERIMENT setting, chosen so saturation bites within a few hundred ticks).
Because only sleep restores plasticity, 'sleep beats no sleep' is built in; the real questions are timing.
World: exp M causal worlds, 3,000 ticks, 48 worlds. Arms: never (no sleep); fixed schedules sleeping every 250 / 500 / 1000 / 2000 awake ticks; self-timed (sleep when the network's own learning-rate factor
1 / (1 + L / L0) falls below 0.5: 'sleep pressure', no knowledge of the schedule grid). Score: mean prequential error over all 3,000 ticks.
PREDICTIONS (registered before running): P146 never-sleeping is >= 1.2 x the best fixed schedule (manipulation check: the constraint bites); P147 self-timed sleep is <= 1.05 x the best fixed schedule (it finds a near-best timing by itself)."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import expM_curriculum_scaling as M

T, SLEEP = 3000, 50
L0 = float(os.environ.get("L0", "2.0"))
ARMS = ("never", "every50", "every100", "every150", "every250", "every500", "marginal") if os.environ.get("B9F") else ("never", "every250", "every500", "every1000", "every2000", "self")


def run(seed):
    from worlds import sample_world, rollout
    from nn import AdamMLP, RunStd
    w = sample_world(seed); X, Y = rollout(w, T, np.random.default_rng(seed + 7))
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]; D = (Y - X[:, :Y.shape[1]])[:, cp]
    out = {}
    for arm in ARMS:
        sx, sy = RunStd(X.shape[1]), RunStd(len(cp))
        nets = [AdamMLP(X.shape[1], 1, M.HID, M.LR, seed + j) for j in range(len(cp))]
        L = np.zeros(len(cp)); asleep = 0; awake_since = 0; err = 0.0; sleeps = 0; deficit = 0.0
        for t in range(T):
            sx.update(X[t]); sy.update(D[t]); xs, ys = sx(X[t]), sy(D[t])
            for j, n in enumerate(nets):
                if asleep > 0:
                    p = n.predict(xs)[0][0]
                else:
                    W = [n.W1.copy(), n.W2.copy()]
                    n.lr = M.LR / (1 + L[j] / L0)
                    p = n.step(xs, ys[j:j + 1])[0]
                    L[j] += np.abs(n.W1 - W[0]).sum() + np.abs(n.W2 - W[1]).sum()
                err += (p - ys[j]) ** 2
            if asleep > 0:
                asleep -= 1
                if asleep == 0:
                    L[:] = 0.0
                continue
            awake_since += 1
            deficit += 1.0 - float(np.mean(1 / (1 + L / L0)))           # B9f: learning lost to saturation since waking, in tick units
            go = (arm.startswith("every") and awake_since >= int(arm[5:])) or (arm == "self" and np.mean(1 / (1 + L / L0)) < 0.5) or (arm == "marginal" and deficit >= SLEEP)
            if go:
                asleep = SLEEP; awake_since = 0; sleeps += 1; deficit = 0.0
        out[arm] = dict(err=float(err / (T * len(cp))), sleeps=sleeps)
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(9100, 9148) if os.environ.get("B9REP") else range(9000, 9048))
    json.dump(res, open("../results/pb9f_sleep_marginal%s.json" % (("_rep_L0_" + os.environ.get("L0", "2")) if os.environ.get("B9REP") else "") if os.environ.get("B9F") else "../results/pb9e_sleep_pressure.json", "w"))
    for a in ARMS:
        print(f"{a:10s} error median {np.median([r[a]['err'] for r in res]):.4f} | sleeps {np.median([r[a]['sleeps'] for r in res]):.0f}")
    best = np.array([min(r[a]["err"] for a in ARMS if a.startswith("every")) for r in res])
    other = "marginal" if os.environ.get("B9F") else "self"
    print("never / best fixed (per world, median) %.3f | %s / best fixed %.3f" % (np.median(np.array([r["never"]["err"] for r in res]) / best), other, np.median(np.array([r[other]["err"] for r in res]) / best)))
    fx = [a for a in ARMS if a.startswith("every")]
    print("best fixed interval per world (counts):", {a: int(sum(min(fx, key=lambda b: r[b]["err"]) == a for r in res)) for a in fx})
    print("done", round(time.time() - t0), "s")
