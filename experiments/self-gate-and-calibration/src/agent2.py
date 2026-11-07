"""Runner for the two-timescale agent. Arms: 'self' (sleep when awake learning progress is flat), 'sched' (fixed-period sleep, ablation),
'never' (no sleep: no structure ever forms). Evaluation drives with the permanent value table that sleep built; the score is never used
by awake learning."""
import time
import numpy as np
import common  # noqa: F401
from constants import C
from env import CarEnv
from encode import Encoder, ACTIONS, ACT_BITS, GRID_BITS
from core import Net
import ranked2

REPORT_EVERY = 500   # CONST: report_every


def make(seed, n_d, T, mode="hard", respawn=True):
    env = CarEnv(seed=seed, n_distract=n_d)
    rng = np.random.default_rng(seed + C("seed_offset_warmup"))
    S = []
    for _ in range(C("warmup_ticks")):
        env.step(int(rng.integers(3)) + 3 * int(rng.random() < C("warmup_throttle_bias")))
        S.append(env.scalars())
    enc = Encoder(env.names, mode)
    enc.calibrate(S)
    enc.encode(env.view(), env.scalars())
    return env, enc, Net(enc, T + C("history_slack"), seed)


def struct_metrics(env, E):
    truth = env.truth_edges()
    indep = {n for n, c in zip(env.names[len(env.CAUSAL):], env.dist_class) if c in ("noise", "clock")}
    act_true = {e for e in truth if e[0].startswith("a_")}
    act_found = {e for e in E if e[0].startswith("a_")}
    tp = len(act_found & act_true)
    fp = sum(1 for p, c in E if p != c and (p in indep or c in indep))
    return tp / max(1, len(act_found)), tp / max(1, len(act_true)), fp


def drive_table(env_like, tab, ticks=None, seed=0):
    ticks = ticks or C("eval_ticks")
    env = CarEnv(seed=seed, n_distract=env_like.n_d, drag=env_like.drag, track=env_like.track)
    rng = np.random.default_rng(seed + C("seed_offset_eval"))
    tot = 0.0
    for _ in range(ticks):
        env.step(tab.act(env.scalars(), rng))
        tot += env.score
    return tot / ticks


def expert_drive(env_like, ticks=None, seed=0):
    ticks = ticks or C("eval_ticks")
    env = CarEnv(seed=seed, n_distract=env_like.n_d, drag=env_like.drag, track=env_like.track)
    tot = 0.0
    for _ in range(ticks):
        st = env.herr + C("expert_gain") * env.lat
        env.step({-1: 0, 0: 1, 1: 2}[-1 if st > C("expert_deadband") else (1 if st < -C("expert_deadband") else 0)] + 3)
        tot += env.score
    return tot / ticks


def train(seed=0, n_d=8, T=None, mode="hard", arm="self", ck=(), verbose=False):
    T = T or C("run_length")
    env, enc, net = make(seed, n_d, T, mode)
    names = env.names
    nact = len(ACTIONS)
    rk = ranked2.TaskRanker(names)
    rng = np.random.default_rng(seed + C("seed_offset_policy"))
    hold = C("hold_ticks")
    x = np.zeros(net.Bi, np.float32)
    x[:net.Bs] = enc.encode(env.view(), env.scalars())
    HS = np.zeros((T, len(names)), np.float32)
    HA = np.zeros(T, np.int64)
    HR = np.zeros(T)
    log = dict(t=[], bits=[], bits_causal=[], score=[], edges=[], neurons=[], hid=[], rec=[], fp=[], cx=[], g_mean=[], alpha_mean=[])
    sleeps, evals = [], {}
    table = None
    acc = np.zeros(3)
    causal = np.array([i for i, v in enumerate(enc.var_of_bit) if v in env.CAUSAL or v == "grid"])
    t0 = time.time()
    a = 0
    for t in range(T):
        if t % hold == 0:
            a = net.choose_action(x)
        x[net.Bs:net.Bs + ACT_BITS] = ACTIONS[a]
        HS[t] = env.scalars()
        HA[t] = a
        env.step(a)
        HR[t] = env.score
        y = enc.encode(env.view(), env.scalars())
        loss = net.observe(x, y)
        acc += (loss.sum(), loss[causal].sum(), env.score)
        x[:net.Bs] = y
        do_sleep = (arm == "self" and net.should_sleep()) or (arm == "sched" and (t + 1) % C("sleep_scheduled_period") == 0)
        if do_sleep:
            info = net.sleep()
            rk.update(t + 1, HS[:t + 1], HA[:t + 1], HR[:t + 1], nact)
            vs = rk.select(HS[:t + 1], HA[:t + 1], HR[:t + 1], nact) if rk.hist else []
            if vs:
                table = ranked2.Table(vs, HS[:t + 1], nact)
                table.fit(HS[:t + 1], HA[:t + 1], HR[:t + 1], hold)
                info["chosen"] = [names[i] for i in vs]
            info["t"] = t + 1
            sleeps.append(info)
        if (t + 1) % REPORT_EVERY == 0:
            prec, rec, fp = struct_metrics(env, net.edges())
            for k, v in zip(("bits", "bits_causal", "score"), acc / REPORT_EVERY):
                log[k].append(float(v))
            log["t"].append(t + 1)
            log["edges"].append(net.n_edges())
            log["neurons"].append(net.n_neurons())
            log["hid"].append(len(net.hid))
            log["rec"].append(rec)
            log["fp"].append(fp)
            log["cx"].append(net.complexity())
            log["g_mean"].append(float(net.trust().mean()))
            log["alpha_mean"].append(float(net.alpha().mean()))
            acc[:] = 0
        if (t + 1) in ck:
            evals[t + 1] = None if table is None else float(np.mean([drive_table(env, table, seed=s) for s in range(C("eval_starts"))]))
    log["secs"] = time.time() - t0
    res = dict(log=log, sleeps=sleeps, evals=evals, ops_awake=net.ops_awake, ops_sleep=net.ops_sleep, n_sleeps=net.sleeps,
               ranking=net.ranking()[:8], task_ranking=[(names[i], float(rk.p[i])) for i in np.argsort(-rk.p)[:8]],
               p_hist=[(h[0], [float(v) for v in h[1]] if hasattr(h[1], "__len__") else float(h[1]), [float(v) for v in h[2]]) for h in net.p_hist],
               hid=net.hid, names=names)
    return env, enc, net, rk, table, res, (HS, HA, HR)


if __name__ == "__main__":
    import sys
    arm = sys.argv[1] if len(sys.argv) > 1 else "self"
    env, enc, net, rk, table, res, _ = train(0, 8, C("demo_length"), "hard", arm, ck=(2000, 6000))
    print(arm, "secs", round(res["log"]["secs"]), "sleeps", res["n_sleeps"], "edges", net.n_edges(), "hid", len(net.hid), "bits", round(res["log"]["bits"][-1], 1),
          "evals", res["evals"], "ops awake/sleep", int(res["ops_awake"]), int(res["ops_sleep"]))
    print([(s["t"], s["added"], s["removed"], s["units"]) for s in res["sleeps"]][:30])
    print("ranking", [(n, round(p, 2)) for n, p in res["ranking"]])
