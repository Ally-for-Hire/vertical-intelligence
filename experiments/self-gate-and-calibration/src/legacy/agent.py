import copy, time
import numpy as np
import run, ranked
from cbn import FutureHeads, ACTIONS, ACT_BITS, Encoder, kl_bits
from env import CarEnv


def snap_heads(fh):
    c = copy.copy(fh)
    c.heads = []
    for h in fh.heads:
        s = copy.copy(h); s.cnt, s.par, s.npar = h.cnt.copy(), h.par.copy(), h.npar.copy(); s.X = s.Y = None
        c.heads.append(s)
    c.q = []
    return c


def train(seed=0, n_d=8, T=16000, mode="hard", width=0.3, gate=False, policy="curious", ck=(), G=500, log_every=500, hold=10):
    env, enc, m = run.make(seed, n_d, T, mode=mode, width=width, gate=gate)
    sb = run.score_bit_idx(enc)
    fh = FutureHeads(enc, sb, T_max=T + 10, seed=seed)
    rng = np.random.default_rng(seed + 5)
    x = np.zeros(m.Bi + 1, np.float32); x[:m.Bs] = enc.encode(env.view(), env.scalars())
    log = dict(t=[], bits=[], bits_causal=[], score=[], n_edges=[], neurons=[], rec=[], fp=[], head_edges=[])
    snaps = {}
    nv = len(env.names)
    HS = np.zeros((T, nv), np.float32); HA = np.zeros(T, np.int64); HR = np.zeros(T)
    rk = ranked.TaskRanker(env.names, enc.q)
    acc = np.zeros(3)
    causal_bits = np.array([i for i, v in enumerate(enc.var_of_bit) if v in env.CAUSAL or v == "grid"])
    t0 = time.time()
    for t in range(T):
        if t % hold == 0:
            a = int(rng.integers(6)) if policy == "random" else m.choose_action(x)
        x[m.Bs:m.Bs + ACT_BITS] = ACTIONS[a]
        HS[t] = env.scalars(); HA[t] = a
        env.step(a)
        HR[t] = env.score
        y = enc.encode(env.view(), env.scalars())
        loss = m.observe(x, y)
        fh.observe(x, y, a)
        acc += (loss.sum(), loss[causal_bits].sum(), env.score)
        x[:m.Bs] = y
        if (t + 1) % G == 0:
            m.grow(); fh.grow(); rk.update(t + 1, HS[:t + 1], HA[:t + 1], HR[:t + 1])
        if (t + 1) % log_every == 0:
            prec, rec, fp = run.struct_metrics(env, m.edges())
            for k, v in zip(("bits", "bits_causal", "score"), acc / log_every):
                log[k].append(float(v))
            log["t"].append(t + 1); log["n_edges"].append(m.n_edges()); log["neurons"].append(m.n_neurons())
            log["rec"].append(rec); log["fp"].append(fp)
            log["head_edges"].append(int(sum(h.n_edges() for h in fh.heads)))
            acc[:] = 0
        if (t + 1) in ck:
            snaps[t + 1] = dict(top=rk.top(4), chosen=list(rk.chosen), p=rk.p.copy(), n=t + 1)
    log["secs"] = time.time() - t0
    log["hist"] = dict(S=HS, A=HA, R=HR)
    return env, enc, m, fh, log, snaps, rk


def drive(env_like, enc, fh_snap, kind="heads", ticks=600, seed=0, eps=0.0):
    env = CarEnv(seed=seed, n_distract=env_like.n_d, drag=env_like.drag, track=env_like.track)
    e2 = Encoder(enc.names, enc.mode, enc.width)
    e2.q, e2.sp, e2.dsc, e2.dq, e2.dsp = enc.q, enc.sp, enc.dsc, enc.dq, enc.dsp
    rng = np.random.default_rng(seed + 9)
    Bs = enc.n_state
    x = np.zeros(Bs + ACT_BITS + 1, np.float32); x[:Bs] = e2.encode(env.view(), env.scalars())
    tot = 0.0
    for t in range(ticks):
        if kind == "random" or rng.random() < eps:
            a = int(rng.integers(6))
        elif kind == "expert":
            st = env.herr + 0.35 * env.lat
            a = {-1: 0, 0: 1, 1: 2}[-1 if st > 0.05 else (1 if st < -0.05 else 0)] + 3
        else:
            a = fh_snap.best_action(x, rng)
        x[Bs:Bs + ACT_BITS] = ACTIONS[a]
        env.step(a)
        x[:Bs] = e2.encode(env.view(), env.scalars())
        tot += env.score
    return tot / ticks


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "hard"
    env, enc, m, fh, log, snaps, rk = train(0, 8, 8000, mode, ck=(2000, 4000, 8000))
    print(mode, "secs", round(log["secs"]), "bits", round(log["bits"][-1], 1), "head edges", log["head_edges"][-1])
    for k, s in snaps.items():
        print(k, "heads-driven", round(drive(env, enc, s), 3), "random", round(drive(env, enc, s, "random"), 3), "expert", round(drive(env, enc, s, "expert"), 3))
