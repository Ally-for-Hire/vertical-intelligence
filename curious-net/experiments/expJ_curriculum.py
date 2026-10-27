"""Exp J: lifelong curriculum. Problems (7, 4000 ticks each, identical streams for all arms):
 car -> cart -> pendulum -> car with mirrored steering -> random boolean net -> car (new track, 64 distractors incl. AR(1)) -> car (the first world again).
Arms: life (one LNet), fresh (new LNet per problem), mlp_ft (fine-tuned), mlp_replay, mlp_fresh. 8 seeds. MLP hyperparameters swept on problem 1 only.
Experiment sizes are EXPERIMENT status."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

T = 4000
SPECS = [("car", dict(n_d=8)), ("cart", {}), ("pend", {}), ("car", dict(n_d=8, mirror=True, key="car_mirror")), ("rbn", {}),
         ("car", dict(n_d=64, classes=("noise", "clock", "copy_v", "lamp", "lag_score", "ar1"), key="car_new64", kind=(3, 4))), ("car", dict(n_d=8, key="car_return"))]
SEEDS = {"car": 0, "cart": 0, "pend": 0, "car_mirror": 0, "rbn": 0, "car_new64": 1, "car_return": 0}   # car_return = the SAME world as problem 1


def streams_for(seed):
    import lifelong as L
    out = []
    for spec in SPECS:
        key = spec[1].get("key", spec[0])
        out.append(L.make_stream(spec, SEEDS[key] + 100 * seed, T))
    return out


def job_net(a):
    arm, seed = a
    import lifelong as L
    S = streams_for(seed)
    R = L.run_life(S, arm, seed)
    return dict(arm=arm, seed=seed, res=[dict(key=r["key"], loss=r["loss"].reshape(-1, 100).mean(1).tolist(), trust=r["trust"].reshape(-1, 100).mean(1).tolist(),
                bit_loss_first=r["bit_loss_first"].tolist(), bit_trust_first=r["bit_trust_first"].tolist(), complexity=r["complexity"], variants=r["variants"],
                sleeps=r["sleeps"], edges=r["edges"]) for r in R],
                meta=[dict(key=s["key"], names=s["names"], truth={k: sorted(v) for k, v in s["truth"].items()}, scalars=s["scalars"], bits_per=s["bits_per"], grid_n=s["grid_n"]) for s in S])


def job_mlp(a):
    arm, seed, hidden, lr, replay, only = a
    from baselines2 import GlobalMLP
    S = streams_for(seed) if only is None else streams_for(seed)[:only]
    reg = {}
    for st in S:
        for n in st["names"] + st["act_names"]:
            reg.setdefault(n, len(reg))
    res = []
    m = None
    for k, st in enumerate(S):
        if m is None or arm == "mlp_fresh":
            m = GlobalMLP(len(reg), hidden, lr, seed, replay)
        xin = np.array([reg[n] for n in st["names"] + st["act_names"]]); yout = np.array([reg[n] for n in st["names"]])
        loss = np.zeros(len(st["X"]))
        for t in range(len(st["X"])):
            loss[t] = m.step(xin, st["X"][t].astype(np.float64), yout, st["Y"][t].astype(np.float64)).sum()
        res.append(dict(key=st["key"], loss=loss.reshape(-1, 100).mean(1).tolist()))
    return dict(arm=arm, seed=seed, res=res, hp=(hidden, lr, replay))


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        grid = [(h, lr) for h in (64, 256) for lr in (0.003, 0.01, 0.03)]
        sw = p.map(job_mlp, [("mlp_ft", s, h, lr, 0, 1) for s in range(3) for (h, lr) in grid])
        sc = {}
        for r in sw:
            sc.setdefault(tuple(r["hp"][:2]), []).append(float(np.mean(r["res"][0]["loss"][-10:])))
        sc = {k: float(np.mean(v)) for k, v in sc.items()}
        hb, lb = min(sc, key=sc.get)
        print("MLP sweep on problem 1 (mean last-1000-tick bits/tick):", {str(k): round(v, 1) for k, v in sc.items()}, "-> chose", hb, lb, flush=True)
        jobs = [(arm, s) for arm in ("life", "fresh") for s in range(8)]
        net = p.map_async(job_net, jobs)
        mjobs = [(arm, s, hb, lb, rp, None) for (arm, rp) in (("mlp_ft", 0), ("mlp_replay", 2000), ("mlp_fresh", 0)) for s in range(8)]
        mlp = p.map(job_mlp, mjobs)
        net = net.get()
    json.dump(dict(net=net, mlp=mlp, sweep={str(k): v for k, v in sc.items()}, chosen=[hb, lb], T=T, order=[s[1].get("key", s[0]) for s in SPECS]), open("../results/expJ.json", "w"), default=float)
    print("done", round(time.time() - t0), "s")
