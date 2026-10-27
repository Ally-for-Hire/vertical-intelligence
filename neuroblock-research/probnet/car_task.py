"""The car as a problem for the probability net: 3 options (steer -1 / 0 / +1), throttle fixed on, success = did the step keep the car
on the road and not drifting outward. Inputs: 7 sensor numbers, each thermometer-coded to 4 bits.   python car_task.py [--ticks N]"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from car_env import CarEnv, K_FWD, K_LAT
import grid, snarc
from problems import thermo

try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass

TRAIN = list(range(16)); TEST = list(range(100, 108))


def sense(env):
    hw = env.track.width / 2
    v = env.view().reshape(K_FWD, K_LAT)
    lj = np.arange(K_LAT) - K_LAT // 2
    cs = [0.5 + (v[r] @ lj) / v[r].sum() / (2 * (K_LAT // 2)) if v[r].any() else 0.5 for r in (2, 4, 6)]
    return np.array([np.clip(env.lat / hw, -2, 2) / 4 + 0.5, env.herr / (2 * np.pi) + 0.5, env.v / env.vmax, env.ontrack, *cs])


RAW = False


def observe(envs):
    X = np.clip(np.stack([sense(e) for e in envs]), 0, 0.999)
    return X if RAW else thermo(X)


def success_fn(envs):
    def f(act):
        out = np.zeros(len(envs))
        for i, (e, a) in enumerate(zip(envs, act)):
            before = e.lat_d
            e.step(int(a) + 3)
            out[i] = float(e.ontrack > 0 and (e.lat_d <= before or e.lat_d < 0.25 * e.track.width / 2))
        return out
    return f


def grade(net, tracks=TEST, T=600):
    envs = [CarEnv(seed=s) for s in tracks]
    score = np.zeros(len(envs)); off = np.zeros(len(envs))
    for t in range(T):
        act, _ = net.present(observe(envs))
        for e, a in zip(envs, act):
            e.step(int(a) + 3)
        score += [e.score for e in envs]; off += [1 - e.ontrack for e in envs]
    return float(score.mean() / T), float(off.mean() / T)


def pd_grade(tracks=TEST, T=600):
    out = []
    for s in tracks:
        e = CarEnv(seed=s); sc = 0
        for t in range(T):
            u = 2.0 * e.herr + 0.5 * e.lat
            e.step((0 if u > 0.05 else 2 if u < -0.05 else 1) + 3); sc += e.score
        out.append(sc / T)
    return float(np.mean(out))


def random_grade(tracks=TEST, T=600):
    rng = np.random.default_rng(0); out = []
    for s in tracks:
        e = CarEnv(seed=s); sc = 0
        for t in range(T):
            e.step(int(rng.integers(0, 3)) + 3); sc += e.score
        out.append(sc / T)
    return float(np.mean(out))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--ticks', type=int, default=3000); ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--tag', default='car_v2a'); ap.add_argument('--kw', default='{}'); ap.add_argument('--cls', default='GridNet')
    a = ap.parse_args()
    cls = getattr(grid, a.cls, None) or getattr(snarc, a.cls)
    RAW = a.cls == 'SnarcNet'
    res = dict(ref=dict(pd=pd_grade(), random=random_grade()), runs=[])
    print('reference', res['ref'], flush=True)
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        envs = [CarEnv(seed=s) for s in TRAIN]
        net = cls(7 if RAW else 28, 3, rng, **json.loads(a.kw))
        curve = []
        t0 = time.time()
        for t in range(1, a.ticks + 1):
            net.present(observe(envs), success_fn(envs), learn=True)
            if t % 500 == 0:
                g, off = grade(net)
                curve.append(dict(tick=t, score=g, off=off, neurons=net.n_neurons, guess=net.guess_rate))
                print(f'seed{seed} tick{t} grade {g:.3f} off {off:.2f} neurons {net.n_neurons} guess {net.guess_rate:.2f} ({time.time() - t0:.0f}s)', flush=True)
        res['runs'].append(curve)
    json.dump(res, open(os.path.join(HERE, f'results_{a.tag}.json'), 'w'))
