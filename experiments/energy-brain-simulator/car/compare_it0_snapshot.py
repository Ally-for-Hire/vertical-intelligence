"""Is the predictive agent (predict.py) purely better than an equivalent neural network?  (user, 2026-10-24)

Both learners get IDENTICAL conditions: the same problems (benchmarks.PROBLEMS), the same sample stream (batch 32, with replacement), bandit
feedback only (they see whether the action they took succeeded, never the right answer), the same fixed budget, the same held-out test.
  NN   : MLP(in-32-32-K, tanh) policy trained by REINFORCE with a running baseline and an entropy bonus (PyTorch, 1 thread). Its learning rate and
         entropy weight are TUNED PER PROBLEM on separate tuning seeds (strong baseline). Reported parameter count.
  AGENT: predict.Agent (global settings, never tuned per problem).
  REF  : the same MLP trained WITH labels (cross-entropy). Sets the level L = 0.95 x REF accuracy (REF is the "equivalent network" ceiling).
Declared in advance, PURELY BETTER = on every problem: (1) final accuracy >= NN's (mean last-5 evals, difference > -0.01), (2) samples to reach L
  <= NN's (median over seeds; never reaching L counts as the budget), (3) CPU seconds of TRAINING to reach L <= NN's. Statistics: 10 seeds.
python car/compare.py [--seeds 10] [--budget 4000] [--tau 0.9] [--methods C] [--only moons]   (parallel workers)
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import car.benchmarks as BM
import car.predict as P

BATCH, EVAL_BATCHES = 32, 25                        # evaluate every 25 batches (800 samples)
GRID = [(lr, ent) for lr in (0.003, 0.01, 0.03, 0.1) for ent in (0.0, 0.01)]


def problem(prob, seed):
    X, y, Xt, yt, ncls, kind, ceil = BM.PROBLEMS[prob](np.random.default_rng(1000 + seed))
    return X, y, Xt, yt, ncls, kind


def mlp(d, K):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(d, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, K))


def nn_params(d, K):
    return (d + 1) * 32 + 33 * 32 + 33 * K


def run_nn(prob, seed, lr, ent, budget, labelled=False):
    import torch
    torch.set_num_threads(1); torch.manual_seed(seed)
    X, y, Xt, yt, ncls, kind = problem(prob, seed)
    enc = BM.Encoder(X, kind); Xb = enc.bits(X); Xtb = enc.bits(Xt)
    f = lambda A: torch.tensor(A * 2 - 1, dtype=torch.float32)       # bits as +-1, same view of the data as the agent
    net = mlp(Xb.shape[1], ncls); opt = torch.optim.Adam(net.parameters(), lr)
    rng = np.random.default_rng(seed); base, curve, cpu = 0.5, [], 0.0
    xs, ys = f(Xb), torch.tensor(y); xt = f(Xtb)
    for b in range(budget):
        t0 = time.process_time()
        idx = rng.integers(0, len(Xb), BATCH); xb = xs[idx]; logits = net(xb)
        if labelled:
            loss = torch.nn.functional.cross_entropy(logits, ys[idx])
        else:
            lp = torch.log_softmax(logits, 1); pr = lp.exp()
            a = torch.multinomial(pr.detach(), 1).squeeze(1)
            r = (a.numpy() == y[idx]).astype(np.float32); base += 0.05 * (r.mean() - base)
            adv = torch.tensor(r - base)
            ent_term = -(pr * lp).sum(1).mean()
            loss = -(adv * lp[torch.arange(BATCH), a]).mean() - ent * ent_term
        opt.zero_grad(); loss.backward(); opt.step(); cpu += time.process_time() - t0
        if (b + 1) % EVAL_BATCHES == 0:
            with torch.no_grad(): acc = float((net(xt).argmax(1).numpy() == yt).mean())
            curve.append(((b + 1) * BATCH, acc, cpu))
    return curve, nn_params(Xb.shape[1], ncls)


def run_agent(prob, method, seed, tau, budget):
    X, y, Xt, yt, ncls, kind = problem(prob, seed)
    enc = BM.Encoder(X, kind); Xb = enc.bits(X)
    rng = np.random.default_rng(seed); ag = P.Agent(Xb.shape[1], ncls, rng, method, tau)
    curve, cpu = [], 0.0
    for b in range(budget):
        t0 = time.process_time()
        idx = rng.integers(0, len(Xb), BATCH)
        a, conf, used, sure, L = ag.act(Xb[idx]); ag.learn(Xb[idx], a, (a == y[idx]).astype(float), sure, L)
        cpu += time.process_time() - t0
        if (b + 1) % EVAL_BATCHES == 0:
            ev = P.evaluate(ag, enc, Xt, yt); curve.append(((b + 1) * BATCH, ev['acc'], cpu))
    return curve, sum(getattr(x, 'cnt', np.zeros(0)).size + (x.u.size + x.R.size if hasattr(x, 'u') else 0) for x in ag.senses)


def reach(curve, L, budget_samples, smooth=2):
    """first sample count where the mean of the last `smooth` evals >= L (else budget); cpu seconds at that point"""
    for i in range(len(curve)):
        if i + 1 >= smooth and np.mean([c[1] for c in curve[i + 1 - smooth:i + 1]]) >= L: return curve[i][0], curve[i][2]
    return budget_samples, curve[-1][2]


def job(args):
    kind, prob, seed, hp, budget = args
    if kind == 'ref': return args, run_nn(prob, seed, 0.01, 0.0, budget, labelled=True)
    if kind == 'nn': return args, run_nn(prob, seed, hp[0], hp[1], budget)
    return args, run_agent(prob, hp[0], seed, hp[1], budget)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10); ap.add_argument('--budget', type=int, default=4000); ap.add_argument('--tau', type=float, default=0.9)
    ap.add_argument('--methods', default='C'); ap.add_argument('--only', default=''); ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--tag', default='a')
    a = ap.parse_args()
    names = [n for n in BM.PROBLEMS if not a.only or n in a.only.split(',')]
    from multiprocessing import Pool
    t0 = time.time(); pool = Pool(a.workers)
    # 1) tune the NN per problem on tuning seeds 100-102 (area under the learning curve); REF level from labelled MLPs on the evaluation seeds
    tune_jobs = [('nn', p, 100 + s, hp, a.budget) for p in names for hp in GRID for s in range(3)]
    tune = pool.map(job, tune_jobs, chunksize=1)
    best = {}
    for p in names:
        sc = {hp: np.mean([np.mean([c[1] for c in cur]) for (k, pp, s, h, b), (cur, _) in tune if pp == p and h == hp]) for hp in GRID}
        best[p] = max(sc, key=sc.get)
    jobs = [('ref', p, s, None, a.budget) for p in names for s in range(a.seeds)]
    jobs += [('nn', p, s, best[p], a.budget) for p in names for s in range(a.seeds)]
    jobs += [('agent', p, s, (m, a.tau), a.budget) for p in names for m in a.methods.split(',') for s in range(a.seeds)]
    out = pool.map(job, jobs, chunksize=1)
    res = {}
    for (kind, p, s, hp, b), (cur, npar) in out:
        key = (p, kind if kind != 'agent' else 'agent' + hp[0]); res.setdefault(key, []).append((cur, npar))
    samples = a.budget * BATCH
    print(f'budget {samples} samples, tau {a.tau}.  L = 0.95 x labelled-MLP accuracy.  Lower is better for samples/cpu; accuracy higher.')
    print(f'{"problem":10s} {"L":>5s} | {"model":9s} {"final acc":>10s} {"samples->L":>11s} {"reach":>6s} {"cpu s->L":>9s} {"params":>7s} | verdict')
    allwin = True; table = {}
    for p in names:
        L = 0.95 * np.mean([np.mean([c[1] for c in cur[-5:]]) for cur, _ in res[(p, 'ref')]])
        rows = {}
        for key in [k for k in res if k[0] == p and k[1] != 'ref']:
            r = res[key]
            fin = [np.mean([c[1] for c in cur[-5:]]) for cur, _ in r]
            rc = [reach(cur, L, samples) for cur, _ in r]
            rows[key[1]] = dict(final=float(np.mean(fin)), fin_sd=float(np.std(fin, ddof=1)), samples=float(np.median([x[0] for x in rc])), reach=sum(x[0] < samples for x in rc),
                                cpu=float(np.median([x[1] for x in rc])), params=int(np.mean([n for _, n in r])), finals=fin)
        table[p] = dict(L=float(L), rows=rows, nn_hp=best[p])
        nn = rows['nn']
        for mname, v in rows.items():
            verdict = ''
            if mname.startswith('agent'):
                ok1, ok2, ok3 = v['final'] >= nn['final'] - 0.01, v['samples'] <= nn['samples'], v['cpu'] <= nn['cpu']
                verdict = ('acc ' + ('OK' if ok1 else 'LOSS') + ' samples ' + ('OK' if ok2 else 'LOSS') + ' cpu ' + ('OK' if ok3 else 'LOSS'))
                allwin &= ok1 and ok2 and ok3
            print(f'{p:10s} {L:5.2f} | {mname:9s} {v["final"]:6.3f}+-{v["fin_sd"]:.3f} {v["samples"]:11.0f} {v["reach"]:3d}/{a.seeds:<2d} {v["cpu"]:9.2f} {v["params"]:7d} | {verdict}')
    print('PURELY BETTER on every problem:', allwin, f'  ({(time.time() - t0) / 60:.1f} min)')
    json.dump(table, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'compare_results_{a.tag}.json'), 'w'))
