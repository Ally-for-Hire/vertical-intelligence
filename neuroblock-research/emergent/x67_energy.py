"""X6.7 (= EMG-2): does selection work with NO GRADER? Early-life selection: whatever keeps being copied while
resources are limited. Problem: xorgauss (noisy XOR, d-16-K tanh nets), target 0.95 test accuracy (best possible 0.955).
Arms (20 seeds, 500-519, budget 3e6 sample-forwards; cost counted as net x point evaluations, as everywhere):
  a  graded cull: series-2 best (P=40, B=16, cull 80%, init 0.5), general.evolve (the grader ranks and culls)
  b  ENERGY WORLD, no ranking and no cull fraction: each step B=16 shared training points are shown to every living net;
     food per point = P_CAP = 40 shared equally among the nets that answer that point correctly (nothing is paid if none does);
     living costs 1 per point per net; a net with energy >= 2 splits into itself + a blind-noise copy (energy halved);
     energy <= 0 dies. Start: 40 random nets, energy 1 (the midpoint of death 0 and split 2).
  c  energy world with UNLIMITED food: a fixed payment per correct answer (PAY = 2), no competition (density-dependence check)
  d  random search: 40 fresh random nets per generation graded on the shared batch; the best is the reported net
  e  SENSITIVITY (registered before running, because the plan does not say whether energy is updated per step or per point):
     the same energy world as (b) but each step's energy change is divided by B (the per-point average), so thresholds 0 and 2 are
     on the scale of one point's flows instead of 16 points'. (b) is the literal reading; (e) is the per-point reading.
Reported net (monitoring only, never used by selection): the richest living net in (b)/(c), the best-graded in (a)/(d).
Pass: (b) reaches 0.95 in >= 18/20 seeds within 3e6. Manipulation checks: Spearman(life accuracy, offspring) > 0 in (b) (median
over seeds); (c) either explodes (population hits N_HARD) in >= 10/20 runs or solves < 18/20; else (b) is VOID.
Units are a choice of units plus ONE parity constant (P_CAP = P = 40), not tuned. ARBITRARY (open problems): PAY = 2, N_HARD = 4000,
MIN_LIFE = 5 steps (a net's accuracy counts in the correlation only after this many steps), initial energy 1 (midpoint).
Gradient references (default Adam lr 1e-3; best of 5 learning rates) are computed on the same seeds and target and reported with no threshold.
Usage: python emergent/x67_energy.py selftest | run [--workers N]      Results: runs/x67_energy.json (resumable)"""
import os, sys, json, math, time, argparse
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
OUTF = os.path.join(HERE, 'runs', 'x67_energy.json')
SEEDS = list(range(500, 520)); BUDGET = 3e6; TARGET = 0.95
P, B, P_CAP, LIVING, SPLIT_AT, E0 = 40, 16, 40.0, 1.0, 2.0, 1.0
PAY, N_HARD, MIN_LIFE = 2.0, 4000, 5
CFG_A = dict(P=40, B=16, cull=0.8, init=0.5)
ARMS = ('a', 'b', 'c', 'd', 'e'); LRS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
_TASK = {}


def task():
    if 'x' not in _TASK:
        import general as G
        t = G.NetTask('xorgauss'); t.target = TARGET; _TASK['x'] = t
    return _TASK['x']


def food_gain(correct, mode):
    """energy gained by each net from one batch. correct: (N, B) bool"""
    import torch
    if mode == 'shared':
        c = correct.sum(0).clamp(min=1).to(torch.float32)
        return (correct.to(torch.float32) * (P_CAP / c)[None]).sum(1)
    return correct.sum(1).to(torch.float32) * PAY


def energy_world(t, seed, budget, mode, norm='step'):
    import torch
    from scipy.stats import spearmanr
    dev = t.Xtr.device; n = t.n; tau = 1.0 / math.sqrt(n)
    with torch.no_grad():
        g = torch.Generator(device=dev).manual_seed(seed); rn = lambda *s: torch.randn(*s, device=dev, generator=g); ru = lambda *s: torch.rand(*s, device=dev, generator=g)
        th = rn(P, n) * 0.5; sig = 10 ** (-2.0 + 2.0 * ru(P)); E = torch.full((P,), E0, device=dev)
        hits = torch.zeros(P, device=dev); seen = torch.zeros(P, device=dev); kids = torch.zeros(P, device=dev)
        done_stats = []; cost = 0; step = 0; peak = P
        def finish(solved, exploded=False, extinct=False):
            rows = done_stats + list(zip((hits / seen.clamp(min=1)).tolist(), kids.tolist(), (seen / B).tolist()))
            rows = [(a, k) for a, k, life in rows if life >= MIN_LIFE]
            rho = float(spearmanr([a for a, _ in rows], [k for _, k in rows])[0]) if len(rows) > 5 else float('nan')
            return dict(cost=float(cost), solved=bool(solved), steps=step, exploded=exploded, extinct=extinct, peak=int(peak), final_n=int(th.shape[0]), rho=rho)
        while cost < budget:
            N = th.shape[0]
            if N == 0: return finish(False, extinct=True)
            i = torch.randint(0, len(t.Xtr), (B,), device=dev, generator=g); Xb, yb = t.Xtr[i], t.ytr[i]
            correct = t.logits(th, Xb).argmax(2) == yb[None]
            dE = food_gain(correct, mode) - B * LIVING
            E = E + (dE / B if norm == 'point' else dE); hits += correct.sum(1); seen += B; cost += N * B
            r = int(E.argmax())
            if t.done(th[r:r + 1]): return finish(True)
            dead = E <= 0
            if bool(dead.any()):
                done_stats += list(zip((hits[dead] / seen[dead]).tolist(), kids[dead].tolist(), (seen[dead] / B).tolist()))
                k = ~dead; th, sig, E, hits, seen, kids = th[k], sig[k], E[k], hits[k], seen[k], kids[k]
            split = E >= SPLIT_AT; ns = int(split.sum())
            if ns:
                idx = split.nonzero()[:, 0]
                cs = sig[idx] * torch.exp(tau * rn(ns)); child = th[idx] + cs[:, None] * rn(ns, n)
                E[idx] = E[idx] / 2; kids[idx] += 1; z = torch.zeros(ns, device=dev)
                th = torch.cat([th, child]); sig = torch.cat([sig, cs]); E = torch.cat([E, E[idx]]); hits = torch.cat([hits, z]); seen = torch.cat([seen, z]); kids = torch.cat([kids, z])
            peak = max(peak, th.shape[0]); step += 1
            if th.shape[0] > N_HARD: return finish(False, exploded=True)
        return finish(False)


def random_search(t, seed, budget):
    import torch
    dev = t.Xtr.device
    with torch.no_grad():
        g = torch.Generator(device=dev).manual_seed(seed); cost = 0
        while cost < budget:
            th = torch.randn(P, t.n, device=dev, generator=g) * 0.5; sc, c = t.score(th, g, B, 'll'); cost += c
            b = int(sc.argmax())
            if t.done(th[b:b + 1]): return dict(cost=float(cost), solved=True)
    return dict(cost=float(cost), solved=False)


def run_one(arm, seed):
    import general as G
    t = task(); t0 = time.time()
    if arm == 'a': c, s, gen = G.evolve(t, CFG_A, seed, BUDGET); r = dict(cost=float(c), solved=bool(s), gen=gen)
    elif arm == 'b': r = energy_world(t, seed, BUDGET, 'shared')
    elif arm == 'c': r = energy_world(t, seed, BUDGET, 'free')
    elif arm == 'e': r = energy_world(t, seed, BUDGET, 'shared', norm='point')
    elif arm == 'd': r = random_search(t, seed, BUDGET)
    elif arm.startswith('adam'):
        lr = float(arm[4:]); c, best = G.gradient(t, seed, lr, target=TARGET); r = dict(cost=float(c) if c is not None else None, solved=c is not None, best=best)
    r['secs'] = round(time.time() - t0, 1); return arm, seed, r


def selftest():
    import torch
    t = task(); dev = t.Xtr.device; ok = True
    print('1. food conservation: when at least one net answers a point, exactly P_CAP is shared out')
    g = torch.Generator(device=dev).manual_seed(3); th = torch.randn(50, t.n, device=dev, generator=g) * 0.5
    correct = t.logits(th, t.Xtr[:16]).argmax(2) == t.ytr[:16][None]
    paid = float(food_gain(correct, 'shared').sum()); expect = P_CAP * int((correct.sum(0) > 0).sum())
    print(f'   paid {paid:.4f}  expected {expect:.4f}'); ok &= abs(paid - expect) < 1e-3
    print('2. energy world runs: arm b for 3e5 on seed 500; arm c explodes or learns; arm d, arm a run')
    for arm in ('a', 'b', 'c', 'd', 'adam0.001'):
        global BUDGET; old = BUDGET; BUDGET = 3e5
        r = run_one(arm, 500)[2]; BUDGET = old
        print('  ', arm, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print('SELFTEST', 'PASSED' if ok else 'FAILED'); return ok


def summarize(res):
    from scipy.stats import binomtest
    med = lambda v: float(np.median(v))
    out = {}
    arms = list(ARMS) + ['adam0.001'] + [f'adam{lr}' for lr in LRS[1:]]
    for arm in arms:
        rows = [res[f'{arm}|{s}'] for s in SEEDS if f'{arm}|{s}' in res]
        if not rows: continue
        costs = [r['cost'] if r['cost'] is not None else 3 * 20000 * 32 for r in rows]
        o = dict(solved=int(sum(r['solved'] for r in rows)), total=len(rows), median_cost=med(costs), median_solved_cost=med([r['cost'] for r in rows if r['solved']]) if any(r['solved'] for r in rows) else None)
        if arm in ('b', 'c', 'e'):
            o['exploded'] = int(sum(r.get('exploded', False) for r in rows)); o['extinct'] = int(sum(r.get('extinct', False) for r in rows))
            rhos = [r['rho'] for r in rows if r['rho'] == r['rho']]; o['rho_median'] = med(rhos) if rhos else None
            o['rho_positive'] = int(sum(x > 0 for x in rhos)); o['rho_n'] = len(rhos); o['peak_median'] = med([r['peak'] for r in rows])
        out[arm] = o
    lrs = [a for a in out if a.startswith('adam')]
    if lrs: out['adam_tuned'] = min((out[a] for a in lrs), key=lambda o: o['median_cost']) | dict(lr=min(lrs, key=lambda a: out[a]['median_cost']))
    for arm in ('b', 'c', 'd', 'e'):
        if arm in out and 'a' in out: out[arm]['cost_vs_a'] = out[arm]['median_cost'] / out['a']['median_cost']
    if 'b' in out and 'adam0.001' in out: out['b']['cost_vs_default_adam'] = out['b']['median_cost'] / out['adam0.001']['median_cost']
    b, c = out.get('b'), out.get('c')
    if b and c:
        out['checks'] = dict(pass_b=b['solved'] >= 18, rho_positive=(b['rho_median'] or 0) > 0, c_explodes_or_fails=(c['exploded'] >= 10 or c['solved'] < 18))
        out['checks']['b_void'] = not (out['checks']['rho_positive'] and out['checks']['c_explodes_or_fails'])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--workers', type=int, default=0); a = ap.parse_args()
    if a.mode == 'selftest': sys.exit(0 if selftest() else 1)
    sys.path.insert(0, '~/curious-net/vi'); import budget as BU
    n = a.workers or min(BU.workers(per_worker_gb=1.0), int(os.environ.get('X67_MAX_WORKERS', '6')))
    res = json.load(open(OUTF))['runs'] if os.path.exists(OUTF) else {}
    jobs = [(arm, s) for arm in list(ARMS) + [f'adam{lr}' for lr in LRS] for s in SEEDS if f'{arm}|{s}' not in res]
    print(f'{len(jobs)} runs, {n} workers, budget {BUDGET:g}', flush=True); t0 = time.time()
    with mp.get_context('spawn').Pool(n) as pool:
        pend = [pool.apply_async(run_one, j) for j in jobs]
        for i, f in enumerate(pend):
            arm, seed, r = f.get(); res[f'{arm}|{seed}'] = r; json.dump({'budget': BUDGET, 'runs': res}, open(OUTF, 'w'))
            print(f'[{i + 1}/{len(jobs)}] {arm} seed {seed}: cost {r["cost"]} solved {r["solved"]} ({r["secs"]}s, {time.time() - t0:.0f}s elapsed)', flush=True)
    summ = summarize(res); json.dump({'budget': BUDGET, 'runs': res, 'summary': summ}, open(OUTF, 'w')); print(json.dumps(summ, indent=1, default=float))
