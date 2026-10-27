"""Exp F: importance gating (park low-p variables) at many distractors. Exp G: Boolean vs soft vs qubit encodings on the fixed learner."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool


def job(a):
    import agent, run
    kind, n_d, mode, gate, seed, T = a
    env, enc, m, fh, log, snaps, rk = agent.train(seed, n_d, T, mode, gate=gate)
    prec, rec, fp = run.struct_metrics(env, m.edges())
    names = env.names
    S, A, R = log["hist"]["S"], log["hist"]["A"], log["hist"]["R"]
    import ranked
    from env import CarEnv
    p = rk.p
    vs = ranked.filtered_select(rk, p, S, A, R)
    tab = ranked.Table(vs, S); tab.fit(S, A, R)
    drv = float(np.mean([ranked.drive_table(env, tab, 500, sd) for sd in range(3)]))
    return dict(kind=kind, n_d=n_d, mode=mode, gate=gate, seed=seed, T=T, secs=log["secs"], fp=fp, rec=rec, prec=prec, edges=m.n_edges(),
                bits_causal=log["bits_causal"][-1], bits=log["bits"][-1], drive=drv, chosen=[names[i] for i in vs],
                top_unc=[(n, round(p_, 2)) for n, p_ in m.ranking("unc")[:6]], parked=int(((m.p_unc < 0.03) & ~m.exempt).sum()), V=len(m.vnames))


if __name__ == "__main__":
    jobs = []
    for n_d in (64, 256):
        for gate in (False, True):
            for s in range(4):
                jobs.append(("F", n_d, "hard", gate, s, 12000))
    for mode in ("hard", "soft", "qubit"):
        for s in range(8):
            jobs.append(("G", 8, mode, False, s, 16000))
    t0 = time.time(); res = []
    with Pool(20) as p:
        for o in p.imap_unordered(job, jobs):
            res.append(o); print(len(res), len(jobs), o["kind"], o["n_d"], o["mode"], o["gate"], o["seed"], "secs", round(o["secs"]), "fp", o["fp"], "rec", round(o["rec"], 2), "drive", round(o["drive"], 2), f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open("../results/expFG.json", "w"), default=float)
