"""NeuroBlock v3: Boolean blocks on a 2D grid that wire and grow themselves while driving. Scored only on steering.

Each block has exactly four abilities:  flip (bias), receive (an incoming signal flips it), send (while on, along each
connection), connect (to any block it has a connection to).  state[t+1] = bias XOR parity(signals arriving at t).
Geometry: blocks sit on grid cells; a wire's delay is 1 + distance//DELAY_CELLS ticks, so far wires are slow.

Start: 12 sensor blocks down the left edge, 3 output blocks (forward / right / left) down the right edge, nothing else.
Self-building (no teacher, no global mutate-and-keep; only the car's score):
  * every wire keeps an eligibility trace (it recently carried a signal)
  * reward-prediction error = score now - running average of score
  * better than expected: recently-used wires strengthen; worse: they weaken, and die at zero
  * while the average score is low the net explores: new wires between active blocks, new blocks grown in a free cell
    next to an active block, bias flips. Exploration fades as the average score rises.
python car/neurogrid.py [--laps 60] [--seeds 3]   (1 thread, low priority)
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from car.env import CarEnv
from car.neuro import SENSORS, OUTS, NS, NO, sense_bits

try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass

G, DELAY_CELLS, DMAX = 16, 4, 6
TRAIN, TEST, T = [0, 1, 2], list(range(100, 108)), 300
LR, TRACE, AVG_RATE, EXPLORE_EVERY, UPKEEP = 0.15, 0.9, 0.02, 20, 0.03


class Grid:
    def __init__(s, rng):
        s.rng = rng
        ys = np.linspace(1, G - 2, NS).round().astype(int)
        oy = np.linspace(G // 2 - 3, G // 2 + 3, NO).round().astype(int)
        s.pos = [(0, int(y)) for y in ys] + [(G - 1, int(y)) for y in oy]
        s.bias = [0] * (NS + NO)
        s.src, s.dst, s.w = [], [], []              # wire k: src -> dst with strength w
        s.hist = np.zeros((DMAX + 1, 256), np.int8)  # hist[d] = state d ticks ago
        s.trace = []
        s._pack()

    @property
    def n(s):
        return len(s.pos)

    def _pack(s):
        s.S, s.D = np.array(s.src, int), np.array(s.dst, int)
        s.W, s.B = np.array(s.w, float), np.array(s.bias, np.int8)
        s.dly = np.array([1 + int(np.hypot(s.pos[a][0] - s.pos[b][0], s.pos[a][1] - s.pos[b][1])) // DELAY_CELLS
                          for a, b in zip(s.src, s.dst)], int)
        if len(s.trace) < len(s.src): s.trace += [0.0] * (len(s.src) - len(s.trace))
        s.T = np.array(s.trace, float)

    def reset_state(s):
        s.hist[:] = 0

    def tick(s, sens):
        n = s.n
        sig = np.zeros(len(s.S))
        if len(s.S):
            sig = s.hist[np.minimum(s.dly, DMAX) - 1, s.S].astype(float)     # state of src `delay` ticks ago
            cnt = np.bincount(s.D, weights=sig, minlength=n).astype(int)
        else:
            cnt = np.zeros(n, int)
        nxt = ((cnt & 1) ^ s.B[:n]).astype(np.int8)
        nxt[:NS] = sens
        s.hist = np.roll(s.hist, 1, axis=0); s.hist[0, :n] = nxt
        if len(s.T): s.T = TRACE * s.T + sig
        return nxt, sig

    # ---- self-building ------------------------------------------------------------------------
    def learn(s, rpe, active, explore):
        """rpe: reward prediction error; active: indices of blocks on now; explore in 0..1"""
        if len(s.T):
            s.W += LR * np.clip(rpe, -1, 1) * np.minimum(s.T, 3.0) / 3.0 - UPKEEP * (s.T < 0.5)
            keep = s.W > 0
            if not keep.all():
                s.src = [a for a, k in zip(s.src, keep) if k]; s.dst = [a for a, k in zip(s.dst, keep) if k]
                s.w = list(s.W[keep]); s.trace = list(s.T[keep])
                s._pack(); return
            s.w = list(s.W); s.trace = list(s.T)
        r = s.rng
        if r.random() < explore:
            kind = r.random()
            act = list(active)
            if kind < 0.5 and act:                                    # wire an active block to anything non-sensor
                a, b = int(r.choice(act)), int(r.integers(NS, s.n))
                if a != b and (a, b) not in set(zip(s.src, s.dst)):
                    s.src.append(a); s.dst.append(b); s.w.append(0.6); s.trace.append(0.0)
            elif kind < 0.8 and act and s.n < 200:                    # grow a block in a free cell next to an active one
                a = int(r.choice(act)); ax, ay = s.pos[a]
                for _ in range(8):
                    c = (int(np.clip(ax + r.integers(-2, 3), 0, G - 1)), int(np.clip(ay + r.integers(-2, 3), 0, G - 1)))
                    if c not in s.pos:
                        k = s.n; s.pos.append(c); s.bias.append(0)
                        o = int(r.integers(NS, s.n))
                        s.src += [a, k]; s.dst += [k, o if o != k else NS]; s.w += [0.6, 0.6]; s.trace += [0.0, 0.0]
                        break
            else:                                                     # flip a bias
                b = int(r.integers(NS, s.n)); s.bias[b] ^= 1
            s._pack()


def drive(net, seed, T=T, learn=False, avg0=0.0):
    env = CarEnv(seed=seed)
    net.reset_state()
    avg, score, off, lat = avg0, 0.0, 0.0, 0.0
    for t in range(T):
        st, _ = net.tick(sense_bits(env))
        f, rt, lf = st[NS:NS + NO]
        env.step((1 if rt == lf else 0 if rt else 2) + 3 * int(f))
        score += env.score; off += 1 - env.ontrack; lat += env.lat_d
        if learn:
            if t and t % EXPLORE_EVERY == 0:
                net.learn(env.score - avg, np.nonzero(st[:net.n])[0], float(np.clip(1 - 1.5 * avg, 0.05, 1)))
            avg += AVG_RATE * (env.score - avg)
    return dict(score=score / T, off=off / T, lat=lat / T), avg


def grade(net, tracks):
    r = [drive(net, s)[0] for s in tracks]
    return {k: float(np.mean([x[k] for x in r])) for k in r[0]}


def live(seed, laps, log):
    net = Grid(np.random.default_rng(seed))
    avg, hist = 0.0, []
    for lap in range(laps):
        _, avg = drive(net, TRAIN[lap % len(TRAIN)], learn=True, avg0=avg)
        if lap % 5 == 4 or lap == laps - 1:
            gr = grade(net, TEST)
            gr.update(lap=lap, blocks=net.n - NS - NO, wires=len(net.S), avg=avg)
            hist.append(gr)
            log(f'grid seed{seed} lap{lap:3d} test {gr["score"]:.3f} off {gr["off"]:.2f} lat {gr["lat"]:.2f} hidden {gr["blocks"]} wires {gr["wires"]} avg {avg:.2f}')
    return hist, net


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--laps', type=int, default=60); ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'neurogrid_results.json')
    t0, res = time.time(), {'runs': {}}
    for sd in range(a.seeds):
        res['runs'][f'grid{sd}'] = live(sd, a.laps, lambda m: print(m, flush=True))[0]
        json.dump(res, open(out, 'w'))
    print(f'done in {time.time() - t0:.0f}s')
