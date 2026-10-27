"""NeuroBlock v7 ("the sense of what will work"): predict P(success) for every action, pick the most probable, think harder until sure.

User spec (2026-10-23): instead of a poll, every neuron predicts the probability that each action leads to a good outcome, given what it
knows and what it does not know (its own "shape" = the part of the input it looks at). The agent takes the most probable action; if no
action reaches TAU (70%, later 90%) it improves its odds (consults more senses, spawns new random ones, explores) until one does.
The probabilities should be RIGHT (calibrated); training an AI to calibrate them is allowed. Try several methods.

Problem setting (bandit feedback, reward only): context bits x, actions a in 0..K-1, outcome success in {0,1}. After acting the agent learns
ONLY whether the chosen action succeeded (never the right answer). Success here = "the answer was correct".

Senses (each sees all samples, learns from the chosen action's outcome, and is judged prequentially against the base rate):
  Table  : a random subset of input bits (its shape; the first sense looks at nothing) -> Beta counts of success/failure per (context, action).
  RNet   : a fixed random tanh layer over the inputs and an action code, with a trained logistic output layer (SGD).
Pooling of the senses' logits (best sense first, so thinking can stop early):
  avg    : quality- and evidence-weighted average of logits (no learning)
  stack  : a logistic regression over the senses' logits, trained online on the chosen action's outcome (the "AI that calibrates")
Methods:  A = Table + avg ("assign values to actions"),  B = RNet + avg ("random neural network"),  C = Table + RNet + stack (both, calibrated).
Thinking: consult senses in quality order until max_a P >= TAU; if none gets there, sample an action in proportion to P (explore).
Growth: when many decisions stay below TAU a new random sense is spawned; a sense whose prequential gain over the base rate is <= 0 after
TRIAL updates is retired (the first sense is never retired).
Metrics (frozen agent, fresh test data): accuracy, mean confidence of the chosen action, ECE (calibration error, 10 bins), Brier score,
share of decisions with confidence >= TAU, accuracy within them, senses consulted.

python car/predict.py [--seeds 5] [--tau 0.7] [--methods A,B,C] [--only parity4,moons] [--workers 16]
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import car.benchmarks as BM

B, CHUNK, SMOOTH, PATIENCE, MAXCHUNKS = 32, 100, 5, 8, 80
TRIAL, MAXSENSE, SPAWN_EVERY, UNSURE_SPAWN = 300, 60, 10, 0.10
H_RNET, LR_RNET, LR_STACK = 24, 0.1, 0.05


def sigm(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


class Sense:
    n_updates = 0
    gain_sum = 0.0; gain_sq = 0.0
    def quality(s):
        if s.n_updates < 30: return 0.0
        return max(s.gain_sum / s.n_updates, 0.0)
    def track(s, p_pred, base_p, y):
        """prequential gain over the base rate, as log-loss difference, for the chosen action's outcome"""
        ll = lambda p: -(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1)))
        g = ll(base_p) - ll(p_pred)
        s.gain_sum += float(g.sum()); s.gain_sq += float((g * g).sum()); s.n_updates += len(y)


class Table(Sense):
    def __init__(s, d, K, rng, kmax, first=False):
        k = 0 if first else int(rng.integers(1, kmax + 1))
        s.bits = np.sort(rng.choice(d, k, replace=False)).astype(int); s.w2 = (1 << np.arange(k)).astype(int)
        s.cnt = np.zeros((2 ** k, K, 2)); s.first = first
    def key(s, X):
        return (X[:, s.bits] @ s.w2).astype(int) if len(s.bits) else np.zeros(len(X), int)
    def predict(s, X):                                   # logits (m,K) and evidence counts (m,K)
        c = s.cnt[s.key(X)]; n = c.sum(2)
        return logit((c[..., 1] + 1) / (n + 2)), n
    def update(s, X, a, y):
        np.add.at(s.cnt, (s.key(X), a, y.astype(int)), 1)


class RNet(Sense):
    def __init__(s, d, K, rng):
        s.R = rng.normal(0, 1.0 / np.sqrt(d + 1), (d, H_RNET)); s.b = rng.normal(0, 0.5, H_RNET); s.A = rng.normal(0, 1.0, (K, H_RNET))
        s.u = np.zeros(H_RNET + 1); s.seen = 0; s.first = False
    def feats(s, X):                                     # (m,K,H+1)
        z = np.tanh((2 * X - 1) @ s.R + s.b)[:, None, :] + s.A[None]
        z = np.tanh(z); return np.concatenate([z, np.ones(z.shape[:2] + (1,))], 2)
    def predict(s, X):
        f = s.feats(X); return f @ s.u, np.full(f.shape[:2], float(s.seen))
    def update(s, X, a, y):
        f = s.feats(X)[np.arange(len(X)), a]; p = sigm(f @ s.u)
        s.u += LR_RNET * ((y - p)[:, None] * f).mean(0); s.seen += len(X)


class Agent:
    def __init__(s, d, K, rng, method, tau):
        s.d, s.K, s.rng, s.method, s.tau = d, K, rng, method, tau
        s.kmax = min(d, 6)
        s.senses = [Table(d, K, rng, s.kmax, first=True)]
        s.base = np.full(K, 0.5); s.base_n = np.zeros(K)
        s.stack = method == 'C'; s.w = np.array([1.0]); s.w0 = 0.0
        s.unsure = []; s.steps = 0; s.born = 1; s.retired = 0

    # ---- thinking: pool senses, best first, stop when sure ---------------------------------------------------------------------
    def order(s):
        return np.argsort([-x.quality() for x in s.senses], kind='stable')

    def think(s, X):
        """returns per-sample chosen action, its pooled probability, senses consulted, the full pooled P at the stopping point, per-sense logits"""
        m, S, K = len(X), len(s.senses), s.K
        L = np.zeros((S, m, K)); N = np.zeros((S, m, K))
        for i, x in enumerate(s.senses): L[i], N[i] = x.predict(X)
        o = s.order()
        q = np.array([s.senses[i].quality() for i in o])
        if s.stack:
            cum = np.cumsum(s.w[o][:, None, None] * L[o], 0) + s.w0
        else:
            wt = (N[o] / (N[o] + 4.0)) * np.maximum(q, 0.02)[:, None, None]
            cum = np.cumsum(wt * L[o], 0) / (np.cumsum(wt, 0) + 0.02)
        P = sigm(cum)                                    # (S,m,K): pooled probability after consulting the first j+1 senses
        reach = P.max(2) >= s.tau                        # (S,m)
        first = np.where(reach.any(0), reach.argmax(0), S - 1)
        idx = np.arange(m)
        Pm = P[first, idx]                               # (m,K)
        return Pm, first + 1, reach.any(0), L

    def act(s, X, explore=True):
        Pm, used, sure, L = s.think(X)
        a = Pm.argmax(1)
        if explore:                                      # not sure enough: probability matching instead of the max
            pr = Pm / Pm.sum(1, keepdims=True)
            cum = pr.cumsum(1); u = s.rng.random(len(X))[:, None]
            samp = (u > cum).sum(1).clip(0, s.K - 1)
            a = np.where(sure, a, samp)
        return a, Pm[np.arange(len(X)), a], used, sure, L

    # ---- learning ---------------------------------------------------------------------------------------------------------------
    def learn(s, X, a, y, sure, L):
        idx = np.arange(len(X))
        for k in range(s.K):                                                       # base rate per action
            mk = a == k
            if mk.any(): s.base_n[k] += mk.sum(); s.base[k] += (y[mk].mean() - s.base[k]) * mk.sum() / s.base_n[k]
        for i, x in enumerate(s.senses):
            p = sigm(L[i][idx, a]); x.track(p, s.base[a], y)
        if s.stack:
            o = s.order()
            F = np.stack([L[i][idx, a] for i in range(len(s.senses))], 1)         # (m,S) chosen-action logits
            pr = sigm(F @ s.w + s.w0)
            s.w += LR_STACK * ((y - pr)[:, None] * F).mean(0); s.w0 += LR_STACK * float((y - pr).mean())
        for x in s.senses: x.update(X, a, y)
        s.unsure.append(float((~sure).mean())); s.steps += 1
        s.regulate()

    def regulate(s):
        keep = [i for i, x in enumerate(s.senses) if x.first or x.n_updates < TRIAL or x.gain_sum > 0]
        if len(keep) < len(s.senses):
            s.retired += len(s.senses) - len(keep)
            s.senses = [s.senses[i] for i in keep]
            if s.stack: s.w = s.w[keep]
        if s.steps % SPAWN_EVERY == 0 and len(s.senses) < MAXSENSE and np.mean(s.unsure[-SPAWN_EVERY:]) > UNSURE_SPAWN:
            if s.method == 'A': new = Table(s.d, s.K, s.rng, s.kmax)
            elif s.method == 'B': new = RNet(s.d, s.K, s.rng)
            else: new = Table(s.d, s.K, s.rng, s.kmax) if s.rng.random() < 0.5 else RNet(s.d, s.K, s.rng)
            s.senses.append(new); s.born += 1
            if s.stack: s.w = np.append(s.w, 0.0)


def calibration(conf, correct, bins=10):
    e = 0.0
    for b in range(bins):
        m = (conf >= b / bins) & (conf < (b + 1) / bins + (b == bins - 1))
        if m.any(): e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def evaluate(ag, enc, Xt, yt):
    Xb = enc.bits(Xt)
    a, conf, used, sure, _ = ag.act(Xb, explore=False)
    ok = (a == yt).astype(float)
    return dict(acc=float(ok.mean()), conf=float(conf.mean()), ece=calibration(conf, ok), brier=float(((conf - ok) ** 2).mean()),
                sure=float(sure.mean()), acc_sure=float(ok[sure].mean()) if sure.any() else float('nan'), used=float(used.mean()),
                senses=len(ag.senses))


def run(job):
    prob, method, seed, tau = job
    rng = np.random.default_rng(seed)
    X, y, Xt, yt, ncls, kind, ceil = BM.PROBLEMS[prob](np.random.default_rng(1000 + seed))
    enc = BM.Encoder(X, kind); Xb = enc.bits(X)
    ag = Agent(Xb.shape[1], ncls, rng, method, tau)
    hist, best, since, reach_tau = [], -1.0, 0, None
    for ch in range(MAXCHUNKS):
        for _ in range(CHUNK):
            idx = rng.integers(0, len(Xb), B)
            a, conf, used, sure, L = ag.act(Xb[idx])
            ag.learn(Xb[idx], a, (a == y[idx]).astype(float), sure, L)
        ev = evaluate(ag, enc, Xt, yt); ev['chunk'] = ch; hist.append(ev)
        if reach_tau is None and ev['conf'] >= tau and ev['ece'] <= 0.05: reach_tau = ch
        sm = float(np.mean([h['acc'] for h in hist[-SMOOTH:]])) if len(hist) >= SMOOTH else None
        if sm is not None and sm > best: best, since = sm, 0
        else: since += 1
        if since >= PATIENCE: break
    last = {k: float(np.mean([h[k] for h in hist[-SMOOTH:]])) for k in ('acc', 'conf', 'ece', 'brier', 'sure', 'used', 'senses')}
    return dict(prob=prob, method=method, seed=seed, tau=tau, chunks=len(hist), reach_tau_chunk=reach_tau, ceil=ceil, born=ag.born, retired=ag.retired, **last)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--tau', type=float, default=0.7)
    ap.add_argument('--methods', default='A,B,C'); ap.add_argument('--only', default=''); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--tag', default='a')
    a = ap.parse_args()
    names = [n for n in BM.PROBLEMS if not a.only or n in a.only.split(',')]
    jobs = [(p, m, sd, a.tau) for p in names for m in a.methods.split(',') for sd in range(a.seeds)]
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(a.workers) as pool: res = pool.map(run, jobs, chunksize=1)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'predict_results_{a.tag}.json')
    json.dump(res, open(out, 'w'))
    print(f'tau {a.tau}   (acc = held-out accuracy; conf = mean stated probability of its pick; ece = calibration error, lower is better)')
    print(f'{"problem":10s} {"m":1s} {"acc":>6s} {"conf":>6s} {"ece":>6s} {"brier":>6s} {"sure":>5s} {"senses":>6s} {"used":>5s} {"reach":>6s}')
    for p in names:
        for m in a.methods.split(','):
            r = [x for x in res if x['prob'] == p and x['method'] == m]
            rc = [x['reach_tau_chunk'] for x in r if x['reach_tau_chunk'] is not None]
            print(f'{p:10s} {m:1s} ' + ' '.join(f'{np.mean([x[k] for x in r]):{w}.{pr}f}' for k, w, pr in (('acc', 6, 3), ('conf', 6, 3), ('ece', 6, 3), ('brier', 6, 3), ('sure', 5, 2), ('senses', 6, 1), ('used', 5, 1)))
                  + f' {len(rc)}/{len(r)}@{int(np.mean(rc)) if rc else "-"}')
    print(f'done in {(time.time() - t0) / 60:.1f} min')
