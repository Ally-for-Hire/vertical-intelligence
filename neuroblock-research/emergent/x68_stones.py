"""X6.8 (= EMG-3): do STEPPING STONES reach hard problems sooner? Nature built hard traits through intermediate ones
that were useful (or at least viable) on the way. Targets: spirals and checker3 (the two benchmark problems the plain
optimizer does not reliably solve), same nets, same targets as X6.5/X6.6 (spirals 0.98, checker3 0.96 test accuracy).
Optimizer: the series-3 best (P=40, B=16, cull 90%, init 0.5), task performance only through the external cull.
Arms (10 seeds 600-609 per problem = 20 runs per arm, budget 1e7 INCLUDING every stage):
  a  direct: the full problem only (bit-identical to general.evolve; selftest)
  b  fixed ramp: the problem grows on a fixed schedule: each non-final stage gets STAGE_BUDGET = 1e6 of cost, then the next; the full problem gets the rest
  c  self-paced ramp: the next stage starts when the population's best net reaches the stage target = the same test accuracy
     target as the full problem, measured on the stage's own test set (external, allowed)
  d  mixed: every scoring batch is drawn from all stages at once
Stages: spirals, turn range t_max/pi = 0.5, 1.2, 1.9, 2.6, 3.2 (full) i.e. 0.25 to 1.6 turns; checker3, grid 2x2 then 3x3 (full).
Every stage uses the FULL problem's input normalisation, so the same nets can move between stages; when a stage changes, the life-average
scores are reset (the scoring distribution changed) and the population is kept.
ARBITRARY (open problems): the five spiral stage sizes, STAGE_BUDGET = 1e6 (10% of the run). DERIVED: stage target = the full target.
Pass (registered): (b) or (c) solves MORE of the 20 runs than (a), OR has a geometric mean over the two problems of the median cost
>= 1.5x lower than (a)'s (unsolved runs counted at the budget). Manipulation checks: (1) (a) fails in >= 5 of 20 runs (else there is no
impasse to cross: VOID); (2) (c) actually reaches the full problem in >= 15 of 20 runs (else the stones were not traversed); (3) parity.
Usage: python emergent/x68_stones.py selftest | run [--workers N]      Results: runs/x68_stones.json (resumable)"""
import os, sys, json, math, time, argparse
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
OUTF = os.path.join(HERE, 'runs', 'x68_stones.json')
CFG = dict(P=40, B=16, cull=0.9, init=0.5)
PROBS = ['spirals', 'checker3']; SEEDS = list(range(600, 610)); BUDGET = 1e7; STAGE_BUDGET = 1e6
SPIRAL_TMAX = [0.5, 1.2, 1.9, 2.6, 3.2]; CHECK_K = [2, 3]
ARMS = {'a': 'direct', 'b': 'fixed', 'c': 'self', 'd': 'mixed'}
_TASK = {}


def spiral(rng, m, tmax):
    y = rng.integers(0, 2, m); t = rng.uniform(0.3, tmax, m) * np.pi
    r = t / (3.2 * np.pi); a = t + y * np.pi
    return np.stack([r * np.cos(a), r * np.sin(a)], 1) + rng.normal(0, 0.02, (m, 2)), y


def checker(rng, m, k):
    X = rng.uniform(0, 1, (m, 2)); return X, ((np.floor(k * X[:, 0]) + np.floor(k * X[:, 1])) % 2).astype(int)


def make_task(prob):
    import torch, general as G, car.compare as CP

    class Staged(G.NetTask):
        def __init__(s, prob):
            super().__init__(prob)
            dev = s.Xtr.device; X = np.asarray(CP.problem(prob, 0)[0], float); mu, sd = X.mean(0), X.std(0) + 1e-9
            s.S = len(SPIRAL_TMAX) if prob == 'spirals' else len(CHECK_K); s.data = []
            for k in range(s.S - 1):
                rng = np.random.default_rng(1000 + 100 * (k + 1))
                gen = (lambda m, r=rng, kk=k: spiral(r, m, SPIRAL_TMAX[kk])) if prob == 'spirals' else (lambda m, r=rng, kk=k: checker(r, m, CHECK_K[kk]))
                Xs, ys = gen(2000); Xt, yt = gen(4000)
                f = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32, device=dev)
                s.data.append((f(Xs), torch.tensor(ys, device=dev), f(Xt), torch.tensor(yt, device=dev)))
            s.data.append((s.Xtr, s.ytr, s.Xte, s.yte))
            s.Xmix = torch.cat([d[0] for d in s.data]); s.ymix = torch.cat([d[1] for d in s.data]); s.cur = (s.Xtr, s.ytr)
        def score(s, th, g, B, kind='ll'):
            X_, y_ = s.cur; i = torch.randint(0, len(X_), (B,), device=s.Xtr.device, generator=g); X, y = X_[i], y_[i]
            lp = torch.log_softmax(s.logits(th, X), 2); return lp.gather(2, y[None, :, None].expand(lp.shape[0], -1, 1))[..., 0].mean(1), th.shape[0] * B
        def stage_done(s, th1, k): return float(s.acc(th1, s.data[k][2], s.data[k][3])[0]) >= s.target
    t = Staged(prob); t.target = json.load(open(os.path.join(HERE, 'runs', 'general.json')))['ref'][prob]['target']
    return t


def evolve_staged(task, cfg, seed, budget, mode):
    """the general.evolve main path with stage control; with mode 'direct' it consumes the random stream exactly like general.evolve"""
    import torch
    dev = task.Xtr.device
    with torch.no_grad():
        g = torch.Generator(device=dev).manual_seed(seed); rn = lambda *s: torch.randn(*s, device=dev, generator=g); ru = lambda *s: torch.rand(*s, device=dev, generator=g)
        P, n = cfg['P'], task.n; nk = max(1, int(round(P * (1 - cfg['cull'])))); nc = P - nk; tau = 1.0 / math.sqrt(n)
        th = rn(P, n) * cfg['init']; sig = 10 ** (-2.0 + 2.0 * ru(P))
        ssum = torch.zeros(P, device=dev); scnt = torch.zeros(P, device=dev); cost = 0; B = float(cfg['B']); gen = 0
        last = task.S - 1; stage = 0 if mode in ('fixed', 'self') else last; in_stage = 0; log = []
        while cost < budget:
            task.cur = (task.Xmix, task.ymix) if mode == 'mixed' else (task.data[stage][0], task.data[stage][1])
            sc, c = task.score(th, g, max(1, int(B)), 'll'); cost += c; in_stage += c
            ssum += sc; scnt += 1; sc = ssum / scnt
            order = torch.argsort(sc + 1e-7 * ru(P), descending=True); best = th[order[0]:order[0] + 1]
            if stage == last and task.done(best): return dict(cost=float(cost), solved=True, gen=gen + 1, stage=stage, log=log)
            if stage < last and ((mode == 'fixed' and in_stage >= STAGE_BUDGET) or (mode == 'self' and task.stage_done(best, stage))):
                log.append((stage, float(in_stage))); stage += 1; in_stage = 0; ssum = torch.zeros(P, device=dev); scnt = torch.zeros(P, device=dev)
            keep = order[:nk]
            pa = keep[torch.randint(0, nk, (nc,), device=dev, generator=g)]; cs = sig[pa] * torch.exp(tau * rn(nc)); z = rn(nc, n)
            child = th[pa] + cs[:, None] * z
            th = torch.cat([th[keep], child]); sig = torch.cat([sig[keep], cs])
            z0 = torch.zeros(nc, device=dev); ssum = torch.cat([ssum[keep], z0]); scnt = torch.cat([scnt[keep], z0]); gen += 1
        return dict(cost=float(cost), solved=False, gen=gen, stage=stage, log=log)


def run_one(arm, prob, seed, budget=None):
    if prob not in _TASK: _TASK[prob] = make_task(prob)
    t0 = time.time(); r = evolve_staged(_TASK[prob], CFG, seed, budget or BUDGET, ARMS[arm]); r['secs'] = round(time.time() - t0, 1)
    return arm, prob, seed, r


def selftest():
    import general as G
    ok = True
    print('1. parity: arm a (direct) reproduces general.evolve exactly')
    for prob, seed, bud in (('spirals', 600, 3e6), ('spirals', 601, 3e6), ('checker3', 600, 4e5)):
        t = make_task(prob); a = G.evolve(t, CFG, seed, bud); b = run_one('a', prob, seed, bud)[3]
        same = a[0] == b['cost'] and a[1] == b['solved'] and a[2] == b['gen']; ok &= same
        print(f'   {prob} seed {seed}: general {a}  x68 {(b["cost"], b["solved"], b["gen"])}  identical {same}')
    print('2. stage machinery runs (spirals seed 600, budget 4e6): arms b, c, d')
    for arm in 'bcd':
        r = run_one(arm, 'spirals', 600, 4e6)[3]
        print(f'   arm {arm}: cost {r["cost"]:,.0f} solved {r["solved"]} stage reached {r["stage"]} of 4 stage log {[(s, round(c)) for s, c in r["log"]]}')
    t = make_task('spirals'); print('   stage data shapes', [tuple(d[0].shape) for d in t.data], 'mixed pool', tuple(t.Xmix.shape))
    print('SELFTEST', 'PASSED' if ok else 'FAILED'); return ok


def summarize(res):
    gm = lambda v: float(math.exp(sum(math.log(x) for x in v) / len(v)))
    out = {}
    for arm in ARMS:
        rows = {p: [res[f'{arm}|{p}|{s}'] for s in SEEDS] for p in PROBS}
        per = {p: float(np.median([r['cost'] for r in rows[p]])) for p in PROBS}
        allr = [r for p in PROBS for r in rows[p]]
        out[arm] = dict(solved=int(sum(r['solved'] for r in allr)), total=len(allr), solved_by_problem={p: int(sum(r['solved'] for r in rows[p])) for p in PROBS},
                        median_cost=per, gm_cost=gm(list(per.values())), reached_full=int(sum(r['stage'] == (len(SPIRAL_TMAX) - 1 if r is not None and p == 'spirals' else len(CHECK_K) - 1)
                                                                                           for p in PROBS for r in rows[p])))
    a = out['a']
    for arm, o in out.items():
        o['cheaper_than_a'] = a['gm_cost'] / o['gm_cost']; o['more_solved'] = o['solved'] > a['solved']
        o['pass'] = bool(o['more_solved'] or o['cheaper_than_a'] >= 1.5) if arm in 'bc' else None
    out['checks'] = dict(a_fails=a['total'] - a['solved'], a_impasse=(a['total'] - a['solved']) >= 5, c_reached_full=out['c']['reached_full'], c_traversed=out['c']['reached_full'] >= 15)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--workers', type=int, default=0); a = ap.parse_args()
    if a.mode == 'selftest': sys.exit(0 if selftest() else 1)
    sys.path.insert(0, '~/curious-net/vi'); import budget as BU
    n = a.workers or min(BU.workers(per_worker_gb=1.0), int(os.environ.get('X68_MAX_WORKERS', '6')))
    res = json.load(open(OUTF))['runs'] if os.path.exists(OUTF) else {}
    jobs = [(arm, p, s) for p in PROBS for arm in ARMS for s in SEEDS if f'{arm}|{p}|{s}' not in res]
    print(f'{len(jobs)} runs, {n} workers, budget {BUDGET:g}', flush=True); t0 = time.time()
    with mp.get_context('spawn').Pool(n) as pool:
        pend = [pool.apply_async(run_one, j) for j in jobs]
        for i, f in enumerate(pend):
            arm, prob, seed, r = f.get(); res[f'{arm}|{prob}|{seed}'] = r; json.dump({'budget': BUDGET, 'runs': res}, open(OUTF, 'w'))
            print(f'[{i + 1}/{len(jobs)}] arm {arm} {prob} seed {seed}: cost {r["cost"]:,.0f} solved {r["solved"]} stage {r["stage"]} ({r["secs"]}s, {time.time() - t0:.0f}s elapsed)', flush=True)
    summ = summarize(res); json.dump({'budget': BUDGET, 'runs': res, 'summary': summ}, open(OUTF, 'w')); print(json.dumps(summ, indent=1, default=float))
