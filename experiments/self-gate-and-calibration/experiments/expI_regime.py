"""Exp I: change of physics and return (does permanent memory survive; does awake self-correction work) vs an online MLP.
All arms see the IDENTICAL stream (random held actions, drag x2 for phase B, back for phase C), so differences are the learner's.
Experiment sizes (EXPERIMENT status): phase A 8000 ticks, B 3000, C 3000; drag multiplier 2 (WORLD change). MLP hyperparameters are tuned by sweep on phase A only."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

TA, TB, TC, MULT, BLK = 8000, 3000, 3000, 2.0, 100
T = TA + TB + TC


def stream(seed, n_d=8):
    import agent2
    from encode import ACTIONS, ACT_BITS
    env, enc, net = agent2.make(seed, n_d, T)
    rng = np.random.default_rng(seed + 123)
    d1 = env.drag
    Bs = enc.n_state
    X = np.zeros((T, Bs + ACT_BITS), np.float32); Y = np.zeros((T, Bs), np.float32)
    x = np.zeros(Bs + ACT_BITS, np.float32); x[:Bs] = enc.encode(env.view(), env.scalars())
    a = 0
    for t in range(T):
        if t == TA: env.set_drag(d1 * MULT)
        if t == TA + TB: env.set_drag(d1)
        if t % 10 == 0: a = int(rng.integers(6))
        x[Bs:] = ACTIONS[a]
        X[t] = x
        env.step(a)
        y = enc.encode(env.view(), env.scalars())
        Y[t] = y; x[:Bs] = y
    vmask = np.array([v == "v" for v in enc.var_of_bit])
    return enc, X, Y, vmask


def run_net(args):
    arm, seed = args
    import agent2
    from core import Net
    enc, X, Y, vmask = stream(seed)
    net = Net(enc, T + 10, seed, use_trust=(arm != "notrust"))
    tot = np.zeros(T); vv = np.zeros(T)
    for t in range(T):
        loss = net.observe(X[t], Y[t])
        tot[t] = loss.sum(); vv[t] = loss[vmask].sum()
        sleep = (arm == "daysleep" and (t + 1) in (TA, TA + TB)) or (arm in ("full", "notrust") and net.should_sleep())
        if sleep:
            net.sleep()
    return dict(arm=arm, seed=seed, tot=tot.reshape(-1, BLK).mean(1).tolist(), v=vv.reshape(-1, BLK).mean(1).tolist(), sleeps=net.sleeps)


def run_mlp(args):
    seed, hidden, lr, upto = args
    from baselines import OnlineMLP
    enc, X, Y, vmask = stream(seed)
    m = OnlineMLP(X.shape[1], Y.shape[1], hidden, lr, seed)
    n = T if upto is None else upto
    tot = np.zeros(n); vv = np.zeros(n)
    for t in range(n):
        l = m.step(X[t], Y[t]); tot[t] = l.sum(); vv[t] = l[vmask].sum()
    return dict(arm="mlp", seed=seed, hidden=hidden, lr=lr, tot=tot, v=vv)


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        sweep = [(s, h, lr, TA) for s in range(3) for h in (32, 128) for lr in (0.003, 0.01, 0.03, 0.1)]
        sw = p.map(run_mlp, sweep)
        best = {}
        for r in sw:
            key = (r["hidden"], r["lr"]); best.setdefault(key, []).append(float(r["tot"][-1000:].mean()))
        scores = {k: float(np.mean(v)) for k, v in best.items()}
        hbest, lrbest = min(scores, key=scores.get)
        print("MLP sweep (mean prequential bits/tick, last 1000 ticks of phase A):", {str(k): round(v, 2) for k, v in scores.items()}, "-> chose", hbest, lrbest, flush=True)
        jobs_net = [(arm, s) for arm in ("full", "notrust", "daysleep") for s in range(8)]
        jobs_mlp = [(s, hbest, lrbest, None) for s in range(8)]
        res_net = p.map(run_net, jobs_net)
        res_mlp = p.map(run_mlp, jobs_mlp)
    for r in res_mlp:
        r["tot"] = np.asarray(r["tot"]).reshape(-1, BLK).mean(1).tolist(); r["v"] = np.asarray(r["v"]).reshape(-1, BLK).mean(1).tolist()
    json.dump(dict(net=res_net, mlp=res_mlp, sweep={str(k): v for k, v in scores.items()}, chosen=[hbest, lrbest], TA=TA, TB=TB, TC=TC, BLK=BLK), open("../results/expI.json", "w"), default=float)
    print("done", round(time.time() - t0), "s")
