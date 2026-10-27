import copy
import numpy as np
import run
from env import CarEnv
from cbn import Encoder, ACTIONS, ACT_BITS, kl_bits


def snapshot(m):
    s = copy.copy(m)
    s.cnt, s.par, s.npar = m.cnt.copy(), m.par.copy(), m.npar.copy()
    s.X = s.Y = None
    return s


def fresh_env(like, seed):
    return CarEnv(seed=like.seed_ if hasattr(like, "seed_") else seed, n_distract=like.n_d, drag=like.drag, track=like.track)


def eval_policy(env_like, enc, snap, kind, ticks=1000, seed=0, rollout="meanfield"):
    """kind: 'exploit' (model lookahead) | 'random' | 'expert' (hand PD controller: reference upper bound)."""
    env = CarEnv(seed=seed, n_distract=env_like.n_d, drag=env_like.drag, track=env_like.track)
    e2 = Encoder(enc.names, enc.mode, enc.width); e2.q, e2.sp, e2.dsc, e2.dq, e2.dsp = enc.q, enc.sp, enc.dsc, enc.dq, enc.dsp
    rng = np.random.default_rng(seed + 9)
    x = np.zeros(snap.Bi + 1, np.float32)
    x[:snap.Bs] = e2.encode(env.view(), env.scalars())
    sb = run.score_bit_idx(enc)
    tot = 0.0; hb = 0.0
    snap.rng = np.random.default_rng(seed + 3)
    for t in range(ticks):
        if kind == "random":
            a = int(rng.integers(6))
        elif kind == "expert":
            steer = -1 if (env.herr + 0.35 * env.lat) > 0.05 else (1 if (env.herr + 0.35 * env.lat) < -0.05 else 0)
            steer = {-1: 0, 0: 1, 1: 2}[steer]
            a = steer + 3
        else:
            a = snap.exploit_action(x, sb, rollout=rollout)
        x[snap.Bs:snap.Bs + ACT_BITS] = ACTIONS[a]
        env.step(a)
        y = e2.encode(env.view(), env.scalars())
        p, _ = snap.p1(x)
        hb += float(kl_bits(y, p)[sb].sum())
        tot += env.score
        x[:snap.Bs] = y
    return tot / ticks, hb / ticks
