"""Exp B (H3): does driving score follow model accuracy? Curiosity-only training; hidden score never used in learning."""
import sys, json, time
sys.path.insert(0, "../src")
import numpy as np, run, evalpolicy as ev

CK = (500, 1000, 2000, 4000, 8000, 12000, 16000)
res = {}
t0 = time.time()
for seed in range(3):
    env, enc, m, snaps = None, None, None, {}
    env, enc, m = run.make(seed, 8, 16000)
    # manual training loop with snapshots
    from cbn import ACTIONS, ACT_BITS
    x = np.zeros(m.Bi + 1, np.uint8); x[:m.Bs] = enc.encode(env.view(), env.scalars())
    for t in range(16000):
        a = m.choose_action(x)
        x[m.Bs:m.Bs + ACT_BITS] = ACTIONS[a]
        env.step(a)
        y = enc.encode(env.view(), env.scalars())
        m.observe(x, y); x[:m.Bs] = y
        if (t + 1) % 500 == 0: m.grow()
        if (t + 1) in CK: snaps[t + 1] = ev.snapshot(m)
    row = {}
    for ck, s in snaps.items():
        sc, hb = ev.eval_policy(env, enc, s, "exploit", 800, seed)
        row[ck] = dict(score=sc, score_bits=hb)
    row["random"] = ev.eval_policy(env, enc, snaps[500], "random", 800, seed)[0]
    row["expert"] = ev.eval_policy(env, enc, snaps[500], "expert", 800, seed)[0]
    res[seed] = row
    print(seed, {k: (round(v["score"], 3) if isinstance(v, dict) else round(v, 3)) for k, v in row.items()}, f"{time.time()-t0:.0f}s", flush=True)
json.dump(res, open("../results/expB.json", "w"), default=float)
