"""X6.9 (= idea I10): FIXED-RATE unit duplication, with no gene. In X6.6 the duplication arm solved spirals 1.7x faster and grew the
nets to width ~28, although its heritable duplication gene did not evolve (it drifted around 0.5). So the CAPACITY to grow may be what
helped, not the evolved rate. Test that directly: the same function-preserving duplication into spare slots (width 16 up to HMAX = 32), but
the per-child duplication chance u is a CONSTANT (no gene, nothing inherited or mutated).
Arms (all start from the series-3 best: P=40, B=16, cull 90%, init 0.5): a = no duplication; g10, g30, g50, g100 = fixed u = 0.1, 0.3, 0.5, 1.0.
Judge: 7 problems x 10 fresh seeds 700-709 per arm (350 runs), budget 1e7. Confirm: (a) and every arm that passed, 20 fresh seeds 710-729.
Pass (the X6.6 rule): solves >= as many runs as (a) AND its geometric mean over problems of the median cost is >= 1.25x lower than (a)'s
(unsolved counted at the budget). Manipulation check: the nets actually grow (mean final active width >= 17 for every u > 0 arm), else VOID.
ARBITRARY (open problems): the four rates, HMAX = 32. Reuses emergent/x66_machinery.py (parity with general.evolve verified).
Usage: python emergent/x69_growth.py judge | confirm [--workers N]     Results: runs/x69_growth.json (resumable)"""
import os, sys, json, math, time, argparse
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import x66_machinery as M
OUTF = os.path.join(HERE, 'runs', 'x69_growth.json')
ARMS = {'a': None, 'g10': 0.1, 'g30': 0.3, 'g50': 0.5, 'g100': 1.0}
JUDGE_SEEDS, CONFIRM_SEEDS = list(range(700, 710)), list(range(710, 730))
PROBLEMS, ORDER, BUDGET = M.PROBLEMS, M.ORDER, 1e7
_TASKS = {}


def run_one(arm, prob, seed, budget=None):
    u = ARMS[arm]; key = (prob, u is not None)
    if key not in _TASKS: _TASKS[key] = M.make_task(prob, key[1])
    t0 = time.time()
    r = M.evolve(_TASKS[key], M.CFG, seed, budget or BUDGET, ('dup',) if u is not None else (), {'dup': u} if u is not None else None)
    r['secs'] = round(time.time() - t0, 1); return arm, prob, seed, r


def summarize(res, seeds_by_arm):
    gm = lambda v: float(math.exp(sum(math.log(x) for x in v) / len(v)))
    out = {}
    for arm, seeds in seeds_by_arm.items():
        rows = {p: [res[f'{arm}|{p}|{s}'] for s in seeds] for p in PROBLEMS}
        per = {p: float(np.median([r['cost'] for r in rows[p]])) for p in PROBLEMS}; allr = [r for p in PROBLEMS for r in rows[p]]
        out[arm] = dict(solved=int(sum(r['solved'] for r in allr)), total=len(allr), gm_cost=gm(list(per.values())), median_cost=per,
                        solved_by_problem={p: int(sum(r['solved'] for r in rows[p])) for p in PROBLEMS}, width=float(np.mean([r['width'] for r in allr])))
    a = out['a']
    for arm, o in out.items():
        o['cheaper_than_a'] = a['gm_cost'] / o['gm_cost']; o['solved_ge_a'] = o['solved'] >= a['solved']
        o['spirals_vs_a'] = a['median_cost']['spirals'] / o['median_cost']['spirals']
        o['grew'] = o['width'] >= 17.0 if arm != 'a' else None
        o['pass'] = bool(o['grew'] and o['solved_ge_a'] and o['cheaper_than_a'] >= 1.25) if arm != 'a' else None
    return out


def pool_run(jobs, n, res):
    t0 = time.time()
    with mp.get_context('spawn').Pool(n) as pool:
        pend = [pool.apply_async(run_one, j) for j in jobs]
        for i, f in enumerate(pend):
            arm, prob, seed, r = f.get(); res[f'{arm}|{prob}|{seed}'] = r; json.dump({'budget': BUDGET, 'runs': res}, open(OUTF, 'w'))
            print(f'[{i + 1}/{len(jobs)}] arm {arm} {prob} seed {seed}: cost {r["cost"]:,.0f} solved {r["solved"]} width {r["width"]:.1f} ({r["secs"]}s, {time.time() - t0:.0f}s elapsed)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--workers', type=int, default=0); a = ap.parse_args()
    sys.path.insert(0, '~/curious-net/vi'); import budget as BU
    n = a.workers or min(BU.workers(per_worker_gb=1.0), int(os.environ.get('X69_MAX_WORKERS', '6')))
    res = json.load(open(OUTF))['runs'] if os.path.exists(OUTF) else {}
    if a.mode == 'judge': arms, seeds = list(ARMS), JUDGE_SEEDS
    else:
        j = summarize(res, {x: JUDGE_SEEDS for x in ARMS}); arms = ['a'] + [x for x in ARMS if x != 'a' and j[x]['pass']]; seeds = CONFIRM_SEEDS; print('confirming arms', arms, flush=True)
    jobs = [(x, p, s) for p in ORDER for x in arms for s in seeds if f'{x}|{p}|{s}' not in res]
    print(f'{len(jobs)} runs, {n} workers, budget {BUDGET:g}', flush=True); pool_run(jobs, n, res)
    summ = summarize(res, {x: seeds for x in arms}); json.dump({'budget': BUDGET, 'runs': res, 'summary_' + a.mode: summ}, open(OUTF, 'w')); print(json.dumps(summ, indent=1, default=float))
