"""NeuroBlock v2: a grid of Boolean blocks that builds itself, scored only on driving the track.

Every block is one neuron with exactly four abilities:
  flip   - a bias: it can flip itself on for a tick
  receive- an incoming signal flips it (odd number of active inputs = flipped)
  send   - while on, it sends a signal along each of its connections
  connect- it can hold connections to any other neuron
So one tick is:  state[t+1] = bias XOR parity(active inputs at t).   Nothing else exists.

Start: sensor blocks (clamped by the world), 3 output blocks (forward / left / right), no connections, no hidden blocks.
The network grows by trying one change at a time (new block, new connection, cut a connection, flip a bias) and keeping
the change only if the car drives at least as well. The grade is env.score (on track, centred, moving) on held-out tracks.

python car/neuro.py [--gens 60] [--seeds 3]   (1 thread, low priority)
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from car.env import CarEnv, K_FWD, K_LAT

try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass

SENSORS = ('lat_L', 'lat_R', 'herr_L', 'herr_R', 'ahead2_L', 'ahead2_R', 'ahead4_L', 'ahead4_R', 'ahead6_L', 'ahead6_R', 'ontrack', 'fast')
NS = len(SENSORS)
OUTS = ('forward', 'right', 'left')          # right = env steer index 0 (-1), left = index 2 (+1), both/neither = straight
NO = len(OUTS)
MAXN = 64
TRAIN, TEST, T = [0, 1, 2], list(range(100, 108)), 300


def sense_bits(env):
    hw = env.track.width / 2
    v = env.view().reshape(K_FWD, K_LAT)
    lj = np.arange(K_LAT) - K_LAT // 2
    b = [env.lat < -0.3 * hw, env.lat > 0.3 * hw, env.herr < -0.15, env.herr > 0.15]
    for r in (2, 4, 6):
        row = v[r]
        c = (row @ lj) / row.sum() if row.any() else 0.0
        b += [c < -0.3, c > 0.3]
    b += [env.ontrack > 0.5, env.v > 0.5 * env.vmax]
    return np.array(b, np.int8)


class Net:
    """Neurons 0..NS-1 sensors, then NO outputs, then hidden. A[i, j] = 1 means j sends a signal to i."""

    def __init__(s):
        s.n = NS + NO
        s.A = np.zeros((MAXN, MAXN), np.int8)
        s.bias = np.zeros(MAXN, np.int8)

    def copy(s):
        c = Net(); c.n = s.n; c.A = s.A.copy(); c.bias = s.bias.copy()
        return c

    def edges(s):
        return int(s.A[:s.n].sum())

    def tick(s, state, sens):
        n = s.n
        nxt = ((s.A[:n, :n] @ state[:n]) & 1) ^ s.bias[:n]          # receive + self flip
        nxt[:NS] = sens                                              # sensors are clamped by the world
        return nxt.astype(np.int8)

    def mutate(s, rng):
        """one structural try. sensors have no inputs; outputs have no outgoing edges."""
        n, r = s.n, rng.random()
        tgt = lambda: int(rng.integers(NS, n))                       # any non-sensor
        src = lambda: int(rng.choice(np.r_[np.arange(NS), np.arange(NS + NO, n)]))   # sensor or hidden
        if r < 0.55:                                                  # connect
            i, j = tgt(), src()
            if i != j: s.A[i, j] ^= 1
        elif r < 0.75 and n < MAXN:                                   # grow: new block spliced into a random connection or hung off a sensor
            k = n; s.n += 1
            i, j = tgt(), src()
            s.A[k, j] = 1
            if i != k: s.A[i, k] = 1
        elif r < 0.9:                                                 # flip a bias
            s.bias[tgt()] ^= 1
        else:                                                         # cut
            ii, jj = np.nonzero(s.A[:n, :n])
            if len(ii):
                m = int(rng.integers(len(ii))); s.A[ii[m], jj[m]] = 0

    def prune(s):
        """drop hidden blocks that cannot reach an output (dead weight). returns the pruned copy"""
        n = s.n
        reach = np.zeros(n, bool); reach[NS:NS + NO] = True
        for _ in range(n):
            new = reach | s.A[:n, :n][reach].any(0)                  # anything that sends to a reached block
            if (new == reach).all(): break
            reach = new
        keep = np.r_[np.arange(NS + NO), [k for k in range(NS + NO, n) if reach[k]]].astype(int)
        c = Net(); m = len(keep); c.n = m
        c.A[:m, :m] = s.A[np.ix_(keep, keep)]; c.bias[:m] = s.bias[keep]
        return c


def drive(net, seed, T=T, drag=None):
    env = CarEnv(seed=seed, drag=drag)
    state = np.zeros(MAXN, np.int8)
    score = off = lat = 0.0
    for _ in range(T):
        state[:net.n] = net.tick(state, sense_bits(env))
        f, rt, lf = state[NS:NS + NO]
        env.step((1 if rt == lf else 0 if rt else 2) + 3 * int(f))
        score += env.score; off += 1 - env.ontrack; lat += env.lat_d
    return dict(score=score / T, off=off / T, lat=lat / T)


def grade(net, tracks):
    r = [drive(net, s) for s in tracks]
    return {k: float(np.mean([x[k] for x in r])) for k in r[0]}


def evolve(seed, gens, kids, log):
    rng = np.random.default_rng(seed)
    best = Net(); fit = grade(best, TRAIN)['score']
    hist = []
    for g in range(gens):
        cands = []
        for _ in range(kids):
            c = best.copy()
            for _ in range(int(rng.integers(1, 4))): c.mutate(rng)
            cands.append((grade(c, TRAIN)['score'], c))
        f, c = max(cands, key=lambda x: x[0])
        if f >= fit:
            fit, best = f, c
        if (g + 1) % 10 == 0: best = best.prune()
        gr = grade(best, TEST)
        gr.update(gen=g, n=best.n - NS - NO, edges=best.edges(), fit=fit)
        hist.append(gr)
        if g % 10 == 0 or g == gens - 1:
            log(f'neuro seed{seed} gen{g:3d} test {gr["score"]:.3f} off {gr["off"]:.2f} lat {gr["lat"]:.2f} hidden {gr["n"]} edges {gr["edges"]} train {fit:.3f}')
    return hist, best


def pd_bits_ref():
    from car.agent import pd_rollout
    return float(np.mean([pd_rollout(s, T=T)['score'] for s in TEST]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--gens', type=int, default=60); ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--kids', type=int, default=8)
    a = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neuro_results.json')
    t0 = time.time()
    res = {'ref_pd': pd_bits_ref(), 'ref_empty': grade(Net(), TEST)['score'], 'runs': {}, 'nets': {}}
    print('reference: pd', res['ref_pd'], 'empty net', res['ref_empty'], flush=True)
    for sd in range(a.seeds):
        h, net = evolve(sd, a.gens, a.kids, lambda m: print(m, flush=True))
        res['runs'][f'neuro{sd}'] = h
        res['nets'][f'neuro{sd}'] = dict(n=net.n, edges=[[int(i), int(j)] for i, j in zip(*np.nonzero(net.A[:net.n, :net.n]))],
                                         bias=net.bias[:net.n].tolist(), sensors=list(SENSORS), outs=list(OUTS))
        json.dump(res, open(out, 'w'))
    print(f'done in {time.time() - t0:.0f}s')
