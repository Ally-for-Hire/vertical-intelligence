"""NeuroBlock v5 ("cascade of trust"): a random system that learns how much to trust its own randomness.

User idea (2026-10-23, paraphrased): neuron 1 is a random on/off flipper. Neuron 2 also draws a random value and decides how
far to trust it given neuron 1. Neuron 3 sees everything before it and decides how far to trust its own draw. This continues
until the whole brain is processed; then the on-neurons are polled for the decision. The whole thing is scored and must be
objectively better than random. Network grows itself.

Mechanics (all measured, nothing tuned per task):
  neuron i sees  z_i = [inputs (+-1), states of neurons 1..i-1 (+-1), bias]            (cascade: it sees everything before it)
  it has two heads:  opinion  p_i = sigmoid(w_i . z_i)           what it would say if it trusted itself
                     trust    t_i = sigmoid(v_i . z_i)           how far to follow that opinion instead of its random draw
  its state is on with probability  P_i = t_i * p_i + (1 - t_i) * 0.5      (trust 0 = pure coin flip, trust 1 = deterministic)
  decision: mode 'poll' = majority of all neuron states (tie = coin flip); mode 'last' = the last neuron's state.
Learning: REINFORCE (policy gradient) with a running baseline, one global reward (+1 right / 0 wrong); so the randomness is
  managed by the reward: trust rises where the context gives information, and stays near 0 where it does not.
Growth: every window, if reward has not improved by more than 2 standard errors vs the previous window, add a neuron (to MAXN).
Controls: a coin flip, the best constant answer, and the Bayes ceiling (best possible from the given inputs).
Legacy/related: see car/RESEARCH_LOG.md (v1 Brain2+net, v2 neuro.py, v3 neurogrid.py, v4 voter.py).

python car/cascade.py [--seeds 20] [--ticks 20000]    (pure NumPy, 1 thread)
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass

B, LR, WINDOW = 32, 0.2, 20                 # batch (ticks per update), learning rate, batches per growth window
COST = True                                 # exponential neuron cost (user, 2026-10-23); --nocost reproduces the unlimited run b
COST0 = 2 * np.sqrt(0.25 / (WINDOW * B))    # 2 standard errors of a window's accuracy = the smallest gain we can even detect (~0.04)
MAXN = float('inf')                         # neuron cap: unlimited by default (user, 2026-10-23); --maxn 10 reproduces the first run


# ---------------------------------------------------------------- problems: (inputs as 0/1 bits, target bit) -------------------
def p_mirror(rng, m):
    a = rng.integers(0, 2, m); return a[:, None], a
def p_and_noise(rng, m):
    X = rng.integers(0, 2, (m, 5)); return X, X[:, 0] & X[:, 1]
def _num(rng, m):
    n = rng.integers(1, 101, m); return n, np.stack([n % 2 == 0, n % 5 == 0, n % 10 == 0], 1).astype(int)
def p_ends5(rng, m):
    n, X = _num(rng, m); return X, (n % 10 == 5).astype(int)
def p_mult4(rng, m):
    n, X = _num(rng, m); return X, (n % 4 == 0).astype(int)        # impossible from the bits: ceiling 75%
def p_gt50(rng, m):
    n, X = _num(rng, m); return X, (n > 50).astype(int)            # impossible: ceiling 50%
def p_xor(rng, m):
    X = rng.integers(0, 2, (m, 2)); return X, X[:, 0] ^ X[:, 1]
PROBLEMS = dict(mirror=p_mirror, and_noise=p_and_noise, ends5=p_ends5, mult4=p_mult4, gt50=p_gt50, xor=p_xor)


def ceilings(prob, rng, m=200_000):
    """Bayes ceiling = best accuracy any rule can get from the given bits; also best constant answer"""
    X, y = prob(rng, m)
    key = X @ (1 << np.arange(X.shape[1]))
    best = sum(max(np.sum((key == k) & (y == 1)), np.sum((key == k) & (y == 0))) for k in np.unique(key)) / m
    return float(best), float(max(y.mean(), 1 - y.mean()))


# ---------------------------------------------------------------- the network --------------------------------------------------
def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


class Cascade:
    def __init__(s, d, rng, mode='poll'):
        s.d, s.rng, s.mode = d, rng, mode
        s.W = [s._new(0)]; s.V = [s._new(0)]               # opinion and trust weights per neuron
        s.base = 0.5; s.hist = []; s.trial = None; s.retired = 0

    def _new(s, i):
        w = s.rng.normal(0, 0.1, s.d + i + 1)               # inputs + earlier neurons + bias
        return w

    @property
    def n(s):
        return len(s.W)

    def grow(s):
        if s.n < MAXN:
            i = s.n; s.W.append(s._new(i)); s.V.append(s._new(i))
            s.V[-1][-1] = -1.0                              # a newborn starts sceptical of itself (low trust)

    def forward(s, X, rng, frozen=False):
        m, n = len(X), s.n
        x = 2.0 * X - 1
        S = np.zeros((m, n)); Z = []; P = np.zeros((m, n)); T = np.zeros((m, n)); O = np.zeros((m, n))
        for i in range(n):
            z = np.concatenate([x, 2.0 * S[:, :i] - 1, np.ones((m, 1))], 1)
            o = sig(z @ s.W[i]); t = sig(z @ s.V[i]); p = t * o + (1 - t) * 0.5
            S[:, i] = rng.random(m) < p
            Z.append(z); P[:, i], T[:, i], O[:, i] = p, t, o
        if s.mode == 'last':
            dec = S[:, -1].astype(int)
        else:
            tot = S.sum(1); half = n / 2
            dec = np.where(tot > half, 1, np.where(tot < half, 0, rng.integers(0, 2, m)))
        return dec, S, Z, P, T, O

    def update(s, X, y, rng):
        dec, S, Z, P, T, O = s.forward(X, rng)
        r = (dec == y).astype(float)
        adv = r - s.base
        s.base += 0.05 * (r.mean() - s.base)
        for i in range(s.n):
            Pi = np.clip(P[:, i], 1e-4, 1 - 1e-4)
            dlogP = (S[:, i] - Pi) / (Pi * (1 - Pi))                       # d log P(s) / dP
            dPa = T[:, i] * O[:, i] * (1 - O[:, i])                        # dP/d(opinion logit)
            dPg = T[:, i] * (1 - T[:, i]) * (O[:, i] - 0.5)                # dP/d(trust logit)
            s.W[i] += LR * (Z[i] * (adv * dlogP * dPa)[:, None]).mean(0)
            s.V[i] += LR * (Z[i] * (adv * dlogP * dPg)[:, None]).mean(0)
        return r.mean()

    def cost(s, k):
        """accuracy a k-th neuron must add to be worth keeping: doubles with every neuron (neuron 1 is free)"""
        return 0.0 if k <= 1 else COST0 * 2.0 ** (k - 2)

    def total_cost(s):
        return sum(s.cost(k) for k in range(1, s.n + 1))

    def maybe_grow(s):
        h = s.hist
        if COST and s.trial is not None:                      # a newborn is on trial: keep it only if it paid for itself
            if len(h) - s.trial['start'] >= 2 * WINDOW:
                gain = float(np.mean(h[-WINDOW:])) - s.trial['pre']
                if gain <= s.cost(s.n):
                    s.W.pop(); s.V.pop(); s.retired += 1
                s.trial = None
            return
        if len(h) >= 2 * WINDOW and len(h) % WINDOW == 0:
            a, b = np.array(h[-2 * WINDOW:-WINDOW]), np.array(h[-WINDOW:])
            se = np.sqrt(a.var() / WINDOW + b.var() / WINDOW) + 1e-9
            if (b.mean() - a.mean()) < 2 * se and s.n < MAXN:
                s.grow()
                if COST: s.trial = dict(start=len(h), pre=float(b.mean()))


def run(name, mode, seed, ticks, evaln=20000):
    prob = PROBLEMS[name]
    rng = np.random.default_rng(seed)
    d = prob(np.random.default_rng(0), 4)[0].shape[1]
    net = Cascade(d, rng, mode)
    for _ in range(ticks // B):
        X, y = prob(rng, B)
        net.hist.append(net.update(X, y, rng)); net.maybe_grow()
    if net.trial is not None: net.W.pop(); net.V.pop(); net.retired += 1      # unproven newborn at the end is retired
    X, y = prob(np.random.default_rng(10_000 + seed), evaln)
    dec = net.forward(X, rng)[0]
    acc = float((dec == y).mean())
    t = np.mean([np.mean(net.forward(X[:2000], rng)[4], 0) for _ in range(1)], 0)
    return dict(acc=acc, neurons=net.n, retired=net.retired, net=acc - net.total_cost() if COST else acc, trust=[round(float(v), 2) for v in t])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=20); ap.add_argument('--ticks', type=int, default=20000)
    ap.add_argument('--maxn', type=int, default=0, help='0 = unlimited')
    ap.add_argument('--tag', default='c_cost'); ap.add_argument('--nocost', action='store_true')
    a = ap.parse_args()
    if a.maxn: MAXN = a.maxn
    if a.nocost: COST = False
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cascade_results_%s.json' % a.tag)
    t0, res = time.time(), {}
    print(f'{"problem":10s} {"mode":5s} {"coin":>5s} {"const":>6s} {"ceil":>6s} | {"acc mean":>8s} {"sd":>5s} {"min":>5s} {"max":>5s} {"z vs .5":>8s} {"neurons":>8s}  trust(last seed)', flush=True)
    for name in PROBLEMS:
        ceil, const = ceilings(PROBLEMS[name], np.random.default_rng(1))
        for mode in ('poll', 'last'):
            r = [run(name, mode, sd, a.ticks) for sd in range(a.seeds)]
            acc = np.array([x['acc'] for x in r])
            z = (acc.mean() - 0.5) / (acc.std(ddof=1) / np.sqrt(len(acc)) + 1e-12)
            res[f'{name}/{mode}'] = dict(ceiling=ceil, const=const, accs=acc.tolist(), neurons=[x['neurons'] for x in r], retired=[x['retired'] for x in r], trust=r[-1]['trust'])
            print(f'{name:10s} {mode:5s} {0.5:5.2f} {const:6.3f} {ceil:6.3f} | {acc.mean():8.3f} {acc.std(ddof=1):5.3f} {acc.min():5.2f} {acc.max():5.2f} {z:8.1f} {np.mean([x["neurons"] for x in r]):5.1f} ret {np.mean([x["retired"] for x in r]):4.1f} net {np.mean([x["net"] for x in r]):.3f}  {r[-1]["trust"]}', flush=True)
            json.dump(res, open(out, 'w'))
    print(f'done in {time.time() - t0:.0f}s')
