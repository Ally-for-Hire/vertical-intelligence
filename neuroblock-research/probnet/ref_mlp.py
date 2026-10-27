"""Reference scores for every problem: chance (best constant answer) and a small plain neural net that is shown the RIGHT ANSWERS.

The probability net only gets success/fail feedback, so this reference is a deliberately easier setting: it tells us what is learnable.
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import SUITE

TEST_N = 1000


def test_set(p):
    return p['gen'](np.random.default_rng(10_000 + p['id']), TEST_N)


def mlp_score(p, hid=32, steps=1500, bs=64, lr=0.01):
    rng = np.random.default_rng(p['id'])
    d, K = p['n_in'], p['K']
    W1 = rng.normal(0, 1 / np.sqrt(d), (d, hid)); b1 = np.zeros(hid)
    W2 = rng.normal(0, 1 / np.sqrt(hid), (hid, K)); b2 = np.zeros(K)
    params = [W1, b1, W2, b2]; m = [np.zeros_like(q) for q in params]; v = [np.zeros_like(q) for q in params]
    for t in range(1, steps + 1):
        X, y = p['gen'](rng, bs); X = X.astype(float) * 2 - 1
        h = np.maximum(0, X @ W1 + b1); z = h @ W2 + b2
        e = np.exp(z - z.max(1, keepdims=True)); P = e / e.sum(1, keepdims=True)
        P[np.arange(bs), y] -= 1; P /= bs
        gW2 = h.T @ P; gb2 = P.sum(0); gh = (P @ W2.T) * (h > 0); gW1 = X.T @ gh; gb1 = gh.sum(0)
        for i, g in enumerate([gW1, gb1, gW2, gb2]):
            m[i] = .9 * m[i] + .1 * g; v[i] = .999 * v[i] + .001 * g * g
            params[i] -= lr * (m[i] / (1 - .9 ** t)) / (np.sqrt(v[i] / (1 - .999 ** t)) + 1e-8)
    X, y = test_set(p); X = X.astype(float) * 2 - 1
    z = np.maximum(0, X @ W1 + b1) @ W2 + b2
    return float((z.argmax(1) == y).mean())


def chance(p):
    return float(np.bincount(test_set(p)[1], minlength=p['K']).max() / TEST_N)


if __name__ == '__main__':
    out = {}
    t0 = time.time()
    for p in SUITE:
        out[p['name']] = dict(chance=chance(p), mlp=mlp_score(p), ceiling=p['ceiling'])
        print(f"{p['id']:2d} {p['name']:34s} chance {out[p['name']]['chance']:.2f} mlp {out[p['name']]['mlp']:.2f}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference.json'), 'w'), indent=1)
    print('mean chance %.3f mean mlp %.3f (%.0fs)' % (np.mean([v['chance'] for v in out.values()]), np.mean([v['mlp'] for v in out.values()]), time.time() - t0))
