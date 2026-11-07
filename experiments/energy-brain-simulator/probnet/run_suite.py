"""Train the probability net on every problem and compare to chance and the plain neural net.

python run_suite.py TAG [--only name] [--batches N] [--seeds S] [--cls ProbNet] [--kw '{"lr":0.5}']
"""
import os, sys, json, time, argparse
import multiprocessing as mp
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from problems import SUITE
from ref_mlp import test_set
import net as netmod, net2 as netmod2, grid as netmod3, snarc as netmod4, snarc_rel as netmod5



def train(p, seed, batches, B, cls, kw, every=None):
    every = every or max(batches // 10, 1)
    rng = np.random.default_rng(seed * 1000 + p['id'])
    n = cls(p['n_in'], p['K'], rng, **kw)
    Xt, yt = test_set(p)
    curve = []
    for b in range(1, batches + 1):
        X, y = p['gen'](rng, B)
        n.present(X, y, learn=True)
        if b % every == 0:
            curve.append(n.evaluate(Xt, yt))
    last = np.array(curve[-3:])
    return dict(acc=float(last[:, 0].mean()), abstain=float(last[:, 1].mean()), third=float(last[:, 2].mean()), edges=int(last[-1, 3]),
                curve=[c[0] for c in curve])


def job(args):
    i, sd, batches, B, clsname, kw = args
    cls = getattr(netmod, clsname, None) or getattr(netmod2, clsname, None) or getattr(netmod3, clsname, None) or getattr(netmod4, clsname, None) or getattr(netmod5, clsname)
    return (i, sd), train(SUITE[i], sd, batches, B, cls, kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tag', nargs='?', default='v0'); ap.add_argument('--only'); ap.add_argument('--batches', type=int, default=1500)
    ap.add_argument('--seeds', type=int, default=2); ap.add_argument('--B', type=int, default=64); ap.add_argument('--cls', default='ProbNet')
    ap.add_argument('--kw', default='{}'); ap.add_argument('--procs', type=int, default=24)
    a = ap.parse_args()
    ref = json.load(open(os.path.join(HERE, 'reference.json')))
    cls = getattr(netmod, a.cls, None) or getattr(netmod2, a.cls, None) or getattr(netmod3, a.cls, None) or getattr(netmod4, a.cls, None) or getattr(netmod5, a.cls); kw = json.loads(a.kw)
    res, t0 = {}, time.time()
    todo = [(p['id'], sd) for p in SUITE if not (a.only and a.only.lower() not in p['name'].lower()) for sd in range(a.seeds)]
    with mp.Pool(a.procs) as pool:
        R_ = dict(pool.imap_unordered(job, [(i, sd, a.batches, a.B, a.cls, kw) for i, sd in todo]))
    for p in SUITE:
        if a.only and a.only.lower() not in p['name'].lower():
            continue
        runs = [R_[(p['id'], s)] for s in range(a.seeds)]
        r = ref[p['name']]
        res[p['name']] = dict(family=p['family'], acc=float(np.mean([x['acc'] for x in runs])), accs=[x['acc'] for x in runs],
                              abstain=float(np.mean([x['abstain'] for x in runs])), third=float(np.mean([x['third'] for x in runs])), edges=int(np.mean([x['edges'] for x in runs])),
                              chance=r['chance'], mlp=r['mlp'], curve=runs[0]['curve'])
        print(f"{p['id']:2d} {p['name']:34s} pn {res[p['name']]['acc']:.2f} (chance {r['chance']:.2f} mlp {r['mlp']:.2f}) abstain {res[p['name']]['abstain']:.2f} third {res[p['name']]['third']:.2f} edges {res[p['name']]['edges']}", flush=True)
    if not a.only:
        v = list(res.values())
        gap = [(x['acc'] - x['chance']) / max(x['mlp'] - x['chance'], 1e-9) for x in v]
        summ = dict(tag=a.tag, kw=kw, cls=a.cls, mean_acc=float(np.mean([x['acc'] for x in v])), mean_chance=float(np.mean([x['chance'] for x in v])),
                    mean_mlp=float(np.mean([x['mlp'] for x in v])), beats_chance=int(sum(x['acc'] > x['chance'] + 0.03 for x in v)),
                    within_10pts_of_mlp=int(sum(x['acc'] >= x['mlp'] - 0.10 for x in v)), mean_fraction_of_headroom=float(np.mean(np.clip(gap, 0, 1))),
                    seconds=time.time() - t0)
        json.dump(dict(summary=summ, problems=res), open(os.path.join(HERE, f'results_{a.tag}.json'), 'w'))
        print(json.dumps(summ))


if __name__ == '__main__':
    main()
