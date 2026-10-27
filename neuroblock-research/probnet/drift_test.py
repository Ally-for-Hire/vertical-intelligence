"""Does reward-driven learning keep up when the world changes?  Train the car, then change the world and keep training.

Scenarios:  drag  = every car's drag is multiplied by 2.5 (it takes more to move and the top speed falls)
            mirror = steering is reversed (steer-left now turns right)
Arms: the net that keeps learning, and a frozen copy of the same net that stops learning at the moment of the change (control).
Grades are on 8 held-out tracks with the changed world, every 250 ticks.   python drift_test.py [--seeds N]
"""
import os, sys, json, copy, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import car_task as ct
from car_env import CarEnv
import snarc, snarc_rel

ct.RAW = True


def make_envs(tracks, scenario, changed):
    envs = [CarEnv(seed=s) for s in tracks]
    if changed:
        for e in envs:
            if scenario == 'drag':
                e.set_drag(e.drag * 2.5)
            else:
                e.mirror = True
    return envs


def grade(net, scenario, changed, T=600):
    envs = make_envs(ct.TEST, scenario, changed)
    score = 0.0
    for t in range(T):
        act, _ = net.present(ct.observe(envs))
        for e, a in zip(envs, act):
            e.step(int(a) + 3)
        score += np.mean([e.score for e in envs])
    return score / T


def run(cls, scenario, seed, pre=3000, post=3000, every=250):
    rng = np.random.default_rng(seed)
    net = cls(7, 3, rng)
    envs = make_envs(ct.TRAIN, scenario, False)
    for t in range(pre):
        net.present(ct.observe(envs), ct.success_fn(envs), learn=True)
    out = dict(before=grade(net, scenario, False), right_after=grade(net, scenario, True))
    frozen = copy.deepcopy(net)
    envs = make_envs(ct.TRAIN, scenario, True)
    learn_curve, frozen_curve = [], []
    for t in range(1, post + 1):
        net.present(ct.observe(envs), ct.success_fn(envs), learn=True)
        if t % every == 0:
            learn_curve.append(grade(net, scenario, True))
    out['learning'] = learn_curve
    out['frozen'] = grade(frozen, scenario, True)
    out['cells'] = net.n_neurons
    return out


def job(j):
    c, sc, sd = j
    cls = getattr(snarc, c, None) or getattr(snarc_rel, c)
    return j, run(cls, sc, sd)


if __name__ == '__main__':
    import multiprocessing as mp
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=3); ap.add_argument('--tag', default='drift')
    a = ap.parse_args()
    jobs = [(c, sc, sd) for c in ('SnarcNet', 'SnarcRel') for sc in ('drag', 'mirror') for sd in range(a.seeds)]
    with mp.Pool(min(len(jobs), 24)) as pool:
        res = pool.map(job, jobs)
    ref = {sc: dict(random=float(np.mean([0.0]))) for sc in ('drag', 'mirror')}
    summary = {}
    for (c, sc, sd), r in res:
        summary.setdefault(f'{c}/{sc}', []).append(r)
    for k, rs in summary.items():
        print(f"{k:18s} before {np.mean([r['before'] for r in rs]):.3f}  right after change {np.mean([r['right_after'] for r in rs]):.3f}  "
              f"frozen {np.mean([r['frozen'] for r in rs]):.3f}  learning (every 250 ticks) {np.round(np.mean([r['learning'] for r in rs], 0), 2).tolist()}")
    json.dump({k: v for k, v in summary.items()}, open(os.path.join(HERE, f'results_{a.tag}.json'), 'w'))
