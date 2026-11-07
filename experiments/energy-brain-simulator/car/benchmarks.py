"""Apply the v6 consensus brain to standard small neural-network problems (user, 2026-10-23: "try this algorithm for other problems").

The brain is unchanged (consensus.Brain). It only ever sees: the problem's input bits (continuous inputs are turned into thermometer bits
at training-set quantiles), a slot saying which OUTPUT BIT is being asked, and a reward = fraction of output bits it got right. No labels
are shown, no gradients flow through the problem: reward-only learning. Classes are coded in binary, one poll per output bit.
Every problem is trained until the SMOOTHED held-out accuracy stops setting new bests (same plateau rule as the car), up to a cap.
Baselines on the same data: majority class, logistic regression and a 1-hidden-layer MLP (PyTorch, 1 thread, trained WITH labels and
gradients, so they have an easier job), plus the Bayes ceiling where known. Boolean problems: train on all patterns (sampled), test on all
patterns uniformly (it learns the function). Continuous problems: 2,000 train / 4,000 fresh test points.

python car/benchmarks.py [--seeds 5] [--only moons,parity4]      (1 thread, low priority)
"""
import os, sys, json, time, argparse, itertools
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ['NEURO_NSLOT'] = '24'
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import car.consensus as C

NIN, QBASE, B, CHUNK, SMOOTH, PATIENCE, MAXCHUNKS = 20, 20, 32, 100, 5, 8, 80   # input slots, first question slot, batch, batches per eval, ...
THERM = 8


# ---------------------------------------------------------------- problems: (train X, y, test X, y, n_classes, kind, ceiling) -----
def bool_problem(fn, nbits, ncls=2):
    allx = np.array(list(itertools.product([0, 1], repeat=nbits)))
    ally = np.array([fn(r) for r in allx])
    return lambda rng: (allx, ally, allx, ally, ncls, 'bits', 1.0)


def cont_problem(gen, ncls=2, ceil=None):
    def make(rng):
        X, y = gen(rng, 2000); Xt, yt = gen(rng, 4000)
        return X, y, Xt, yt, ncls, 'cont', ceil
    return make


def g_moons(rng, m):
    y = rng.integers(0, 2, m); t = rng.uniform(0, np.pi, m)
    x = np.where(y == 0, np.cos(t), 1 - np.cos(t)); yy = np.where(y == 0, np.sin(t), 0.5 - np.sin(t))
    return np.stack([x, yy], 1) + rng.normal(0, 0.1, (m, 2)), y
def g_circles(rng, m):
    y = rng.integers(0, 2, m); t = rng.uniform(0, 2 * np.pi, m); r = np.where(y == 0, 1.0, 0.5)
    return np.stack([r * np.cos(t), r * np.sin(t)], 1) + rng.normal(0, 0.05, (m, 2)), y
def g_spirals(rng, m):
    y = rng.integers(0, 2, m); t = rng.uniform(0.3, 3.2, m) * np.pi
    r = t / (3.2 * np.pi); a = t + y * np.pi
    return np.stack([r * np.cos(a), r * np.sin(a)], 1) + rng.normal(0, 0.02, (m, 2)), y
def g_quad(rng, m):
    X = rng.uniform(-1, 1, (m, 2)); return X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
def g_blobs(rng, m):
    c = np.array([[0, 1.0], [-0.9, -0.5], [0.9, -0.5]]); y = rng.integers(0, 3, m)
    return c[y] + rng.normal(0, 0.45, (m, 2)), y


PROBLEMS = {
    'parity4': bool_problem(lambda r: int(r.sum() % 2), 4),
    'parity6': bool_problem(lambda r: int(r.sum() % 2), 6),
    'majority5': bool_problem(lambda r: int(r.sum() >= 3), 5),
    'mux6': bool_problem(lambda r: int(r[2 + 2 * r[0] + r[1]]), 6),
    'adder2': bool_problem(lambda r: int((2 * r[0] + r[1]) + (2 * r[2] + r[3])), 4, ncls=7),
    'quadrants': cont_problem(g_quad, 2, 1.0),
    'moons': cont_problem(g_moons, 2),
    'circles': cont_problem(g_circles, 2),
    'spirals': cont_problem(g_spirals, 2),
    'blobs3': cont_problem(g_blobs, 3),
}


# ---------------------------------------------------------------- HOLD-OUT problems (added 2026-10-24; NOT used for any design decision) ----
def cont_problem_noisy(gen, ncls, flip):
    """train labels have a fraction `flip` replaced by random classes; the test labels are clean"""
    def make(rng):
        X, y = gen(rng, 2000); Xt, yt = gen(rng, 4000)
        bad = rng.random(len(y)) < flip; y = np.where(bad, rng.integers(0, ncls, len(y)), y)
        return X, y, Xt, yt, ncls, 'cont', None
    return make
def g_checker(rng, m):
    X = rng.uniform(0, 1, (m, 2)); return X, ((np.floor(3 * X[:, 0]) + np.floor(3 * X[:, 1])) % 2).astype(int)
def g_sine(rng, m):
    X = rng.uniform(-1, 1, (m, 2)); return X, (X[:, 1] > 0.6 * np.sin(3 * X[:, 0])).astype(int)
def g_blobs4(rng, m):
    c = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float); y = rng.integers(0, 4, m); return c[y] + rng.normal(0, 0.55, (m, 2)), y
def g_rings3(rng, m):
    y = rng.integers(0, 3, m); t = rng.uniform(0, 2 * np.pi, m); r = np.array([0.3, 0.7, 1.1])[y]
    return np.stack([r * np.cos(t), r * np.sin(t)], 1) + rng.normal(0, 0.05, (m, 2)), y
def g_xorgauss(rng, m):
    c = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], float); k = rng.integers(0, 4, m); return c[k] + rng.normal(0, 0.5, (m, 2)), (k < 2).astype(int)

HOLDOUT = {
    'parity3of6': bool_problem(lambda r: int((r[0] + r[2] + r[4]) % 2), 6),
    'cmp3': bool_problem(lambda r: int((4 * r[0] + 2 * r[1] + r[2]) > (4 * r[3] + 2 * r[4] + r[5])), 6),
    'addmod4': bool_problem(lambda r: int(((2 * r[0] + r[1]) + (2 * r[2] + r[3])) % 4), 4, ncls=4),
    'and_or10': bool_problem(lambda r: int(r[3] & (r[7] | r[9])), 10),
    'majority7': bool_problem(lambda r: int(r.sum() >= 4), 7),
    'checker3': cont_problem(g_checker, 2, 1.0),
    'sine': cont_problem(g_sine, 2, 1.0),
    'blobs4': cont_problem(g_blobs4, 4),
    'rings3': cont_problem(g_rings3, 3, 1.0),
    'xorgauss': cont_problem(g_xorgauss, 2),
    'moons_noisy15': cont_problem_noisy(g_moons, 2, 0.15),
}


# ---------------------------------------------------------------- EMERGENCE tasks (added 2026-10-26 for the variety-window hypothesis): each REQUIRES hidden units ---------
EMERGE = {
    'xor2': bool_problem(lambda r: int(r[0] ^ r[1]), 2),
    'parity3': bool_problem(lambda r: int(r.sum() % 2), 3),
    'xor2_d4': bool_problem(lambda r: int(r[0] ^ r[1]), 6),              # xor of the first two bits, four distractor bits
}


# ---------------------------------------------------------------- encoding ----------------------------------------------------------
class Encoder:
    def __init__(s, X, kind):
        s.kind = kind
        if kind == 'cont':
            s.q = np.quantile(X, np.linspace(0, 1, THERM + 2)[1:-1], axis=0)       # (THERM, d) thresholds from the training set
        s.d = X.shape[1] if kind == 'bits' else X.shape[1] * THERM
        assert s.d <= NIN

    def bits(s, X):
        if s.kind == 'bits': return X.astype(float)
        return (X[:, None, :] > s.q[None]).reshape(len(X), -1).astype(float)

    def inputs(s, X, qbit):
        b = s.bits(X); x = np.zeros((len(X), C.D))
        x[:, 0:2 * b.shape[1]:2] = 2 * b - 1
        x[:, 2 * (QBASE + qbit)] = 1.0
        return x


def poll_all(brain, enc, X, k, rng=None):
    outs, recs, used, unres = [], [], [], []
    for q in range(k):
        dec, rec, u, un = brain.settle(enc.inputs(X, q), rng)
        outs.append(dec); recs.append(rec); used.append(u); unres.append(un)
    return np.array(outs), recs, np.mean(used, 0), np.mean(unres, 0)


def to_code(y, k):
    return np.stack([(y >> j) & 1 for j in range(k)])


def from_code(bits):
    return (bits * (1 << np.arange(bits.shape[0]))[:, None]).sum(0)


def run_brain(prob, seed, log=None):
    rng = np.random.default_rng(seed)
    X, y, Xt, yt, ncls, kind, ceil = PROBLEMS[prob](np.random.default_rng(1000 + seed))
    k = max(1, int(np.ceil(np.log2(ncls)))); enc = Encoder(X, kind)
    Y = to_code(y, k)
    brain = C.Brain(rng, window=20, own_cost=True, norm_adv=True)
    erng = np.random.default_rng(555 + seed)
    hist, best, since = [], -1.0, 0
    for ch in range(MAXCHUNKS):
        for _ in range(CHUNK):
            idx = rng.integers(0, len(X), B)
            outs, recs, u, un = poll_all(brain, enc, X[idx], k)
            r = (outs == Y[:, idx]).mean(0)                                   # fraction of output bits right: the ONLY feedback
            brain.apply(recs, r, u, un)
        outs, _, u, _ = poll_all(brain, enc, Xt, k, erng)
        acc = float((from_code(outs) == yt).mean())
        hist.append(acc)
        sm = float(np.mean(hist[-SMOOTH:])) if len(hist) >= SMOOTH else None
        if sm is not None and sm > best: best, since = sm, 0
        else: since += 1
        if since >= PATIENCE: break
    return dict(acc_last=float(np.mean(hist[-SMOOTH:])), acc_best_smoothed=best, chunks=len(hist), neurons=brain.n, sweeps=float(u.mean()),
                curve=hist), (X, y, Xt, yt, ncls)


def baselines(data, seed):
    import torch
    torch.set_num_threads(1); torch.manual_seed(seed)
    X, y, Xt, yt, ncls = data
    mu, sd = X.mean(0), X.std(0) + 1e-6
    f = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32)
    out = dict(majority=float((yt == np.bincount(y).argmax()).mean()))
    for name, net in (('logistic', torch.nn.Linear(X.shape[1], ncls)),
                      ('mlp32', torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, ncls)))):
        opt = torch.optim.Adam(net.parameters(), 0.01); xt_, yt_ = f(X), torch.tensor(y)
        for _ in range(800):
            opt.zero_grad(); torch.nn.functional.cross_entropy(net(xt_), yt_).backward(); opt.step()
        out[name] = float((net(f(Xt)).argmax(1).numpy() == yt).mean())
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--only', default='')
    ap.add_argument('--tag', default='a')
    a = ap.parse_args()
    try:
        import ctypes; ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass
    names = [n for n in PROBLEMS if not a.only or n in a.only.split(',')]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'benchmarks_results_{a.tag}.json')
    res, t0 = {}, time.time()
    print(f'{"problem":10s} {"chance":>6s} {"majority":>8s} {"logistic":>8s} {"mlp32":>7s} | {"brain acc":>10s} {"sd":>5s} {"min":>5s} {"max":>5s} {"neurons":>7s} {"chunks":>6s} {"min":>5s}', flush=True)
    for nm in names:
        rows, bl = [], []
        for sd in range(a.seeds):
            r, data = run_brain(nm, sd); rows.append(r); bl.append(baselines(data, sd))
        acc = np.array([r['acc_last'] for r in rows]); ncls = data[4]
        b = {k: float(np.mean([x[k] for x in bl])) for k in bl[0]}
        res[nm] = dict(brain=rows, baselines=bl)
        print(f'{nm:10s} {1 / ncls:6.2f} {b["majority"]:8.3f} {b["logistic"]:8.3f} {b["mlp32"]:7.3f} | {acc.mean():10.3f} {acc.std(ddof=1):5.3f} {acc.min():5.2f} {acc.max():5.2f} '
              f'{np.mean([r["neurons"] for r in rows]):7.1f} {np.mean([r["chunks"] for r in rows]):6.1f} {(time.time() - t0) / 60:5.1f}', flush=True)
        json.dump(res, open(out, 'w'))
    print(f'done in {(time.time() - t0) / 60:.1f} min')
