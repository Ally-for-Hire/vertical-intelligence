"""NeuroBlock v6 ("consensus brain"): ONE persistent brain; random neurons that re-read each other until >=70% agree; then a poll.

User spec (2026-10-23):
  * neurons are random at their base; they all cascade out and interpret each other until they agree, or until 70% agree
  * a poll is taken for every problem, and it is the SAME brain every time (no reset, no problem id)
  * the brain works on itself mid-problem, using itself: no outside teacher; growth and retirement are decided by the brain from its
    own statistics (its reward history, its own agreement); the experimenter supplies only the problem stream and the +1/0 score.

Mechanics
  state: every neuron holds a bit, initially a coin flip.
  sweep: neurons are visited in random order; neuron i reads the problem input and the CURRENT bits of all other neurons,
         computes opinion p = sigmoid(w.z) and trust t = sigmoid(v.z), and re-draws its bit with P(on) = t*p + (1-t)*0.5
         (trust 0 = pure randomness, trust 1 = follows its opinion).
  consensus: after each sweep, if >= CONSENSUS of the neurons hold the same bit the brain stops and that bit is the poll result;
         after MAXSWEEPS without consensus the poll is taken anyway (majority, ties random) and counted as UNRESOLVED.
  learning: REINFORCE on the last sweep's draws (the ones that produced the consensus), +1 right / 0 wrong, running baseline.
  self-regulation (all from the brain's own history): when its reward has not improved by 2 standard errors (or it is often
         unresolved) it adds a neuron ON TRIAL; the newborn is kept only if reward rises by more than its cost, which doubles
         with every neuron (the cost scale COST0 is the brain's own 2-SE detection floor). Otherwise it is retired.
Problems share one input vector with disjoint slots (mirror, AND+noise, ends-in-5, XOR); inactive slots are all-zero, so the brain is
never told which problem it is facing. Phases: each problem alone in sequence (tests forgetting), then all mixed.
Legacy/related: cascade.py (v5, per-problem fresh brain), see car/RESEARCH_LOG.md.

python car/consensus.py [--seeds 10] [--phase 12000] [--mixed 24000]    (pure NumPy, 1 thread)
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass

B, LR, WINDOW = 32, 0.2, 20
CONSENSUS, MAXSWEEPS = 0.70, 20
CAP = 64                                                  # array size only; the cost keeps the real size far below it
COST0 = 2 * np.sqrt(0.25 / (WINDOW * B))
NSLOT = 11
ENC = 'pm'          # 'pm': one signed channel per bit (absent=0, no free per-problem bias); 'onehot': round-0 encoding
VOTE = 'abstain'    # 'abstain': trust = probability of VOTING (non-voters are readable observers/hidden units); 'mix': round-0 noise mixing
D = 2 * NSLOT
DIM = D + CAP + 1


# ---------------------------------------------------------------- problems on a shared input vector ----------------------------
def _enc(bits, slots, m):
    x = np.zeros((m, D))
    for k, sl in enumerate(slots):
        if ENC == 'onehot': x[np.arange(m), 2 * sl + bits[:, k]] = 1.0
        else: x[:, 2 * sl] = 2.0 * bits[:, k] - 1
    return x


def t_mirror(rng, m):
    a = rng.integers(0, 2, m); return _enc(a[:, None], [0], m), a
def t_and(rng, m):
    X = rng.integers(0, 2, (m, 5)); return _enc(X, [1, 2, 3, 4, 5], m), X[:, 0] & X[:, 1]
def t_ends5(rng, m):
    n = rng.integers(1, 101, m); X = np.stack([n % 2 == 0, n % 5 == 0, n % 10 == 0], 1).astype(int)
    return _enc(X, [6, 7, 8], m), (n % 10 == 5).astype(int)
def t_xor(rng, m):
    X = rng.integers(0, 2, (m, 2)); return _enc(X, [9, 10], m), X[:, 0] ^ X[:, 1]
TASKS = dict(mirror=t_mirror, and_noise=t_and, ends5=t_ends5, xor=t_xor)
CONST = dict(mirror=0.5, and_noise=0.75, ends5=0.9, xor=0.5)          # best constant answer
CEIL = dict(mirror=1.0, and_noise=1.0, ends5=1.0, xor=1.0)            # Bayes ceiling from the given bits


def sig(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


# ---------------------------------------------------------------- the brain -----------------------------------------------------
class Brain:
    def __init__(s, rng):
        s.rng = rng
        s.W = np.zeros((CAP, DIM)); s.V = np.zeros((CAP, DIM))
        s.n = 0; s.hist = []; s.trial = None; s.retired = 0; s.unres = []; s.sweeps = []
        s._add(first=True)

    def _add(s, first=False):
        i = s.n
        s.W[i, :D] = s.rng.normal(0, 0.1, D); s.W[i, -1] = s.rng.normal(0, 0.1)
        s.V[i, :D] = s.rng.normal(0, 0.1, D); s.V[i, -1] = 0.0 if first else -1.0      # newborn starts sceptical of itself
        s.n += 1

    def _retire_last(s):
        s.n -= 1; s.W[s.n] = 0; s.V[s.n] = 0; s.W[:, D + s.n] = 0; s.V[:, D + s.n] = 0; s.retired += 1   # also forget what others learned about it

    def cost(s, k):
        return 0.0 if k <= 1 else COST0 * 2.0 ** (k - 2)

    def settle(s, Xe, rng=None):
        """cascade until >=CONSENSUS of neurons agree. returns decision, records for learning, sweeps used, unresolved mask"""
        rng = rng or s.rng
        m, n = len(Xe), s.n
        spm = np.zeros((m, CAP)); spm[:, :n] = 2.0 * (rng.random((m, n)) < 0.5) - 1       # random start
        rec = [dict(z=np.zeros((m, DIM)), p=np.full(m, .5), o=np.full(m, .5), t=np.zeros(m), s=np.zeros(m), v=np.ones(m)) for _ in range(n)]
        vote = np.ones((m, n))
        done = np.zeros(m, bool); used = np.full(m, MAXSWEEPS)
        one = np.ones((m, 1))
        for k in range(MAXSWEEPS):
            live = ~done
            for i in rng.permutation(n):
                z = np.concatenate([Xe, spm, one], 1)
                z[:, D + i] = 0.0                                                    # a neuron does not read itself
                o = sig(z @ s.W[i]); t = sig(z @ s.V[i])
                if VOTE == 'abstain': p = o; vt = (rng.random(m) < t).astype(float)       # bit from opinion; trust = chance of voting
                else: p = t * o + (1 - t) * 0.5; vt = np.ones(m)
                draw = (rng.random(m) < p).astype(float)
                r = rec[i]
                for key, val in (('z', z), ('p', p), ('o', o), ('t', t), ('s', draw), ('v', vt)):
                    r[key][live] = val[live]
                spm[live, i] = 2 * draw[live] - 1; vote[live, i] = vt[live]
            nv = vote.sum(1); pos = ((spm[:, :n] > 0) * vote).sum(1)
            agree = (nv > 0) & (np.maximum(pos, nv - pos) >= CONSENSUS * np.maximum(nv, 1))        # >=70% of the VOTERS agree
            newly = live & agree
            used[newly] = k + 1; done |= agree
            if done.all(): break
        nv = vote.sum(1); tot = ((spm[:, :n] > 0) * vote).sum(1)
        dec = np.where(2 * tot > nv, 1, np.where(2 * tot < nv, 0, rng.integers(0, 2, m)))
        return dec, rec, used, ~done

    def learn(s, Xe, y):
        dec, rec, used, unres = s.settle(Xe)
        r = (dec == y).astype(float)
        base = getattr(s, 'base', 0.5)
        adv = r - base; s.base = base + 0.05 * (r.mean() - base)
        for i in range(s.n):
            R = rec[i]; P = np.clip(R['p'], 1e-4, 1 - 1e-4)
            if VOTE == 'abstain':                                                   # both draws are Bernoulli logits: d log P = (draw - prob)
                ga = (R['z'] * (adv * (R['s'] - R['o']))[:, None]).mean(0); gv = (R['z'] * (adv * (R['v'] - R['t']))[:, None]).mean(0)
            else:
                dlogP = (R['s'] - P) / (P * (1 - P))
                dPa = R['t'] * R['o'] * (1 - R['o']); dPg = R['t'] * (1 - R['t']) * (R['o'] - 0.5)
                ga = (R['z'] * (adv * dlogP * dPa)[:, None]).mean(0); gv = (R['z'] * (adv * dlogP * dPg)[:, None]).mean(0)
            ga[D + i] = gv[D + i] = 0
            s.W[i] += LR * ga; s.V[i] += LR * gv
        s.hist.append(float(r.mean())); s.unres.append(float(unres.mean())); s.sweeps.append(float(used.mean()))
        s.self_regulate()

    def self_regulate(s):
        h = s.hist
        if s.trial is not None:                                   # newborn on trial: keep only if it paid for itself
            if len(h) - s.trial['start'] >= 2 * WINDOW:
                if float(np.mean(h[-WINDOW:])) - s.trial['pre'] <= s.cost(s.n): s._retire_last()
                s.trial = None
            return
        if len(h) >= 2 * WINDOW and len(h) % WINDOW == 0 and s.n < CAP:
            a, b = np.array(h[-2 * WINDOW:-WINDOW]), np.array(h[-WINDOW:])
            se = np.sqrt(a.var() / WINDOW + b.var() / WINDOW) + 1e-9
            confused = np.mean(s.unres[-WINDOW:]) > 0.1                # often failing to reach consensus = needs more minds
            if (b.mean() - a.mean()) < 2 * se or confused:
                s._add(); s.trial = dict(start=len(h), pre=float(b.mean()))

    def evaluate(s, task, seed, m=4000):
        rng = np.random.default_rng(777 + seed)
        Xe, y = TASKS[task](np.random.default_rng(10_000 + seed), m)
        dec, _, used, unres = s.settle(Xe, rng)
        return float((dec == y).mean()), float(used.mean()), float(unres.mean())


def lifetime(seed, phase, mixed, log=None):
    rng = np.random.default_rng(seed)
    br = Brain(rng); names = list(TASKS); marks = {}
    def snap(tag):
        marks[tag] = {t: br.evaluate(t, seed) for t in names}; marks[tag]['n'] = br.n
    for nm in names:                                              # each problem alone, in order
        for _ in range(phase // B):
            Xe, y = TASKS[nm](rng, B); br.learn(Xe, y)
        snap(f'after_{nm}')
    for _ in range(mixed // B):                                   # then everything mixed, same brain
        Xe, y = TASKS[names[rng.integers(len(names))]](rng, B); br.learn(Xe, y)
    snap('final')
    marks['retired'] = br.retired
    return marks


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--phase', type=int, default=12000); ap.add_argument('--mixed', type=int, default=24000)
    ap.add_argument('--tag', default='a'); ap.add_argument('--enc', default='pm'); ap.add_argument('--vote', default='abstain')
    a = ap.parse_args(); ENC, VOTE = a.enc, a.vote
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'consensus_results_{a.tag}.json')
    t0 = time.time()
    R = [lifetime(sd, a.phase, a.mixed) for sd in range(a.seeds)]
    json.dump(R, open(out, 'w'))
    names = list(TASKS)
    print(f'{"checkpoint":14s} ' + ' '.join(f'{t:>16s}' for t in names) + '   neurons')
    for tag in [f'after_{t}' for t in names] + ['final']:
        row = [np.mean([r[tag][t][0] for r in R]) for t in names]
        sds = [np.std([r[tag][t][0] for r in R], ddof=1) for t in names]
        print(f'{tag:14s} ' + ' '.join(f'{m:8.3f} +-{s:5.3f}' for m, s in zip(row, sds)) + f'   {np.mean([r[tag]["n"] for r in R]):.1f}')
    print(f'{"best const":14s} ' + ' '.join(f'{CONST[t]:16.3f}' for t in names))
    print(f'{"ceiling":14s} ' + ' '.join(f'{CEIL[t]:16.3f}' for t in names))
    fin = [r['final'] for r in R]
    print('final sweeps used / unresolved:', {t: (round(np.mean([f[t][1] for f in fin]), 2), round(np.mean([f[t][2] for f in fin]), 3)) for t in names},
          'retired/run', np.mean([r['retired'] for r in R]))
    print(f'done in {time.time() - t0:.0f}s')
