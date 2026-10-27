"""Generations: evolve the policy net (block frozen), then let the block live a life and learn. Brain and net persist across generations.

python car/train.py [--gens 40] [--seeds 3]   (1 thread, low priority)
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from car.agent import CarBlock, BLOCK_CFG, D_SENSE, Net, rollout, pd_rollout

try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass

TRAIN = [0, 1, 2]             # tracks used for fitness and for the block's life
TEST = list(range(100, 108))  # held-out tracks: the grade
POP, SIGMA, LR = 12, 0.15, 0.1


def grade(theta, net, block, tracks):
    r = [rollout(theta, net, block, s) for s in tracks]
    return {k: float(np.mean([x[k] for x in r])) for k in r[0]}


def run(mode, seed, gens, log):
    rng = np.random.default_rng(seed)
    block = CarBlock(D_SENSE, 2, rng=np.random.default_rng(seed + 1000), **BLOCK_CFG) if mode == 'block' else None
    n_in = D_SENSE if block is None else 2 * D_SENSE + 3
    net = Net(n_in)
    theta = rng.normal(0, 0.3, net.size)
    hist = []
    for g in range(gens):
        eps = rng.normal(0, 1, (POP, net.size))
        fit = np.empty(2 * POP)
        for i in range(POP):
            for j, sgn in enumerate((1, -1)):
                fit[2 * i + j] = np.mean([rollout(theta + sgn * SIGMA * eps[i], net, block, s)['score'] for s in TRAIN])
        rk = (np.argsort(np.argsort(fit)) / (len(fit) - 1)) - 0.5
        adv = rk[0::2] - rk[1::2]
        theta = theta + LR / (POP * SIGMA) * (adv @ eps) * 1.0
        if block is not None:
            rollout(theta, net, block, TRAIN[g % len(TRAIN)], learn=True)    # the block's life this generation
        gr = grade(theta, net, block, TEST)
        gr.update(gen=g, n=(block.n if block is not None else 0), fit_best=float(fit.max()))
        hist.append(gr)
        log(f'{mode:6s} seed{seed} gen{g:3d} grade {gr["score"]:.3f} off {gr["off"]:.2f} lat {gr["lat"]:.2f} n {gr["n"]} think {gr["think"]:.0f}us')
    return hist


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--gens', type=int, default=40); ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
    ref = dict(pd=float(np.mean([pd_rollout(s)['score'] for s in TEST])))
    from car.agent import features  # noqa
    ref['random'] = float(np.mean([rollout(np.random.default_rng(s).normal(0, 1, Net(D_SENSE).size), Net(D_SENSE), None, 100 + s)['score'] for s in range(8)]))
    print('reference grades:', ref, flush=True)
    res = {'ref': ref, 'runs': {}}
    t0 = time.time()
    for mode in ('block', 'plain'):
        for seed in range(a.seeds):
            res['runs'][f'{mode}{seed}'] = run(mode, seed, a.gens, lambda m: print(m, flush=True) if m.split('gen')[1].split()[0] in ('0', str(a.gens - 1)) or int(m.split('gen')[1].split()[0]) % 10 == 0 else None)
            json.dump(res, open(out, 'w'))
    print(f'done in {time.time() - t0:.0f}s')
