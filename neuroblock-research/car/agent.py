"""NeuroBlock steering agent: a default Brain2 (situation prototypes + remembered outcome) feeding a small policy net.

The block is the persistent "brain": it learns online, across every generation, which situations exist and how good each has been.
The net at the very bottom reads the sensors plus what the block makes of them, and chooses steer (L/S/R) and throttle.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from brain2 import Brain2
from car.env import CarEnv, K_FWD, K_LAT

# the default NeuroBlock (sweep.py BASE_BRAIN): condensing on, fast firing, no hardening, no expand drive
BLOCK_CFG = dict(rho=0.5, move=True, merge=True, prune=True, select='fast', lr=0.1, min_sep=0.3, spawn_len=1.0,
                 eta=0.5, spawn_cost=50.0, rigid=False, expand=False)
D_SENSE = 7
WIN = 200   # ticks between condensing, same as the sweep's window


def sense(env):
    """7 numbers in about [0,1]: lateral offset, heading error, speed, on-track, and where the track centre sits at 3 look-ahead depths."""
    hw = env.track.width / 2
    v = env.view().reshape(K_FWD, K_LAT)
    lj = np.arange(K_LAT) - K_LAT // 2
    cs = []
    for r in (2, 4, 6):
        row = v[r]
        cs.append(0.5 + (row @ lj) / row.sum() / (2 * (K_LAT // 2)) if row.any() else 0.5)
    return np.array([np.clip(env.lat / hw, -2, 2) / 4 + 0.5, env.herr / (2 * np.pi) + 0.5, env.v / env.vmax,
                     env.ontrack, *cs])


class CarBlock(Brain2):
    """Brain2 whose specialists also remember the reward that followed them (stored in lab: col 0 = reward sum, col 1 = visits)."""

    def look(s, x):
        """read-only firing: (winner prototype, novelty 0..1, remembered outcome, thought time us)"""
        if s.n == 0:
            return x, 1.0, 0.0, s.spawn_cost
        dist = np.linalg.norm(s.w[1:] - x, axis=1)
        m = dist < s.rho
        if m.any():
            idx = np.where(m)[0] + 1
            win = idx[np.argmin(s.d[idx])]
            return s.w[win], float(dist[win - 1] / s.rho), float(s.lab[win, 0] / max(s.lab[win, 1], 1)), float(s.d[win])
        j = int(np.argmin(dist)) + 1
        return s.w[j], 1.0, 0.0, float(s.d.max() + s.spawn_cost)

    def learn(s, x):
        """one experience: fire/spawn as usual, then credit the reward that followed to the specialist that fired"""
        t, win, spawned = s.step(x, 1)
        s._pending = win
        return t

    def credit(s, r):
        w = getattr(s, '_pending', None)
        if w is not None and w < len(s.lab):
            s.lab[w, 0] += r


def features(block, x):
    if block is None:
        return x, 0.0
    w, nov, val, t = block.look(x)
    return np.concatenate([x, w, [nov, val, np.log1p(block.n) / 4]]), t


class Net:
    """tiny tanh MLP; weights live in one flat vector so evolution can perturb them directly"""

    def __init__(self, n_in, n_hid=16, n_out=5):
        self.shape = (n_in, n_hid, n_out)
        self.size = n_in * n_hid + n_hid + n_hid * n_out + n_out

    def act(self, theta, f):
        a, h, o = self.shape
        W1 = theta[:a * h].reshape(a, h); b1 = theta[a * h:a * h + h]
        k = a * h + h
        W2 = theta[k:k + h * o].reshape(h, o); b2 = theta[k + h * o:]
        y = np.tanh(f @ W1 + b1) @ W2 + b2
        return int(np.argmax(y[:3])) + 3 * int(y[3] > y[4])   # env action = steer index (0:-1, 1:0, 2:+1) + 3*throttle


def rollout(theta, net, block, seed, T=600, learn=False, drag=None):
    """one lifetime on one track. returns dict of grades. If learn, the block learns online (and condenses every WIN ticks)."""
    env = CarEnv(seed=seed, drag=drag)
    score = off = lat = 0.0
    times = []
    for t in range(T):
        x = sense(env)
        f, tt = features(block, x)
        times.append(tt)
        if learn and block is not None:
            block.learn(x)
        env.step(net.act(theta, f))
        score += env.score; off += 1 - env.ontrack; lat += env.lat_d
        if learn and block is not None:
            block.credit(env.score)
            if (t + 1) % WIN == 0:
                block.condense()
    return dict(score=score / T, off=off / T, lat=lat / T, think=float(np.mean(times)) if block is not None else 0.0)


def pd_rollout(seed, T=600, drag=None):
    """hand-written reference controller (what 'steers well' looks like)"""
    env = CarEnv(seed=seed, drag=drag)
    score = off = lat = 0.0
    for t in range(T):
        u = 2.0 * env.herr + 0.5 * env.lat          # >0: heading/offset to the left of the track, so steer right (action 0)
        env.step((0 if u > 0.05 else 2 if u < -0.05 else 1) + 3)
        score += env.score; off += 1 - env.ontrack; lat += env.lat_d
    return dict(score=score / T, off=off / T, lat=lat / T, think=0.0)
