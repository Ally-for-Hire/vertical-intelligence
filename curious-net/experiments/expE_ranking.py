"""Exp E: does the agent's own importance ranking pick the variables that make a good driver? (curiosity-trained, score never in learning
except via the 'what is scored' sensor used by the ranker). 10 seeds, 8 distractors, T=32k, checkpoints."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

CK = (1000, 2000, 4000, 8000, 16000, 32000)


def job(args):
    import agent, ranked
    seed, n_d, mode = args
    env, enc, m, fh, log, snaps, rk = agent.train(seed, n_d, CK[-1], mode, ck=CK)
    names = env.names; S, A, R = log["hist"]["S"], log["hist"]["A"], log["hist"]["R"]
    rng = np.random.default_rng(seed)
    pool = [i for i, n in enumerate(names) if n not in ("score", "drag")]
    out = dict(seed=seed, n_d=n_d, mode=mode, ck={}, names=names)
    for ck, s in snaps.items():
        p = s["p"]
        sets = {"ranked_top3": s["top"][:3], "ranked_filtered": ranked.filtered_select(rk, p, S[:ck], A[:ck], R[:ck]),
                "greedy_mdl": s["chosen"] or s["top"][:1],
                "oracle": [names.index(n) for n in ("herr", "lat", "ontrack")],
                "bottom3": [i for i in np.argsort(p) if names[i] not in ("drag", "score")][:3]}
        res = {}
        for k, vs in sets.items():
            if not vs:
                res[k] = None; continue
            tab = ranked.Table(vs, S[:ck]); tab.fit(S[:ck], A[:ck], R[:ck])
            res[k] = float(np.mean([ranked.drive_table(env, tab, 500, sd) for sd in range(3)]))
        rr = []
        for _ in range(5):
            vs = list(rng.choice(pool, 3, replace=False))
            tab = ranked.Table(vs, S[:ck]); tab.fit(S[:ck], A[:ck], R[:ck])
            rr.append(float(np.mean([ranked.drive_table(env, tab, 500, sd) for sd in range(3)])))
        res["random3"] = float(np.mean(rr))
        res["sets"] = {k: [names[i] for i in v] for k, v in sets.items()}
        out["ck"][ck] = res
    out["p_hist"] = [(h[0], float(h[1]), [float(x) for x in h[2]]) for h in rk.hist]
    out["expert"] = agent.drive(env, enc, None, "expert", 500, seed)
    out["random_policy"] = agent.drive(env, enc, None, "random", 500, seed)
    out["unc_rank"] = [(n, float(p)) for n, p in m.ranking("unc")]
    out["train_secs"] = log["secs"]
    return out


if __name__ == "__main__":
    jobs = [(s, 8, "hard") for s in range(10)]
    t0 = time.time(); res = []
    with Pool(10) as p:
        for o in p.imap_unordered(job, jobs):
            res.append(o); print(o["seed"], {k: (round(v["ranked_top3"], 2), round(v["random3"], 2)) for k, v in o["ck"].items()}, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open("../results/expE.json", "w"), default=float)
