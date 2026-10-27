"""X6.5 (= EMG-0, infra): repair the emergent-optimizer benchmark. In series 3 most unsolved runs on
spirals and checker3 hit the 3e6 budget and were scored AT the budget, which hides their real cost. Re-run
series-3 round 0 (cull 80%) and its best (cull 90%) at budget 1e7 with the same seeds 0-9 on all 7 problems.
Report solved/70 FIRST, then the geometric mean with unsolved runs flagged. Parity check: every run that
was solved under 3e6 must give a bit-identical (cost, solved, generations); extending the budget changes
nothing else.

Runs in a process pool sized by curious-net/vi/budget.py (15% machine margin, capped by X65_MAX_WORKERS);
each worker is its own CUDA process. Results are written to runs/x65_budget.json as they arrive, so an
interrupted run resumes.
Usage: python emergent/x65_budget.py [--workers N] [--budget 1e7]"""
import os, sys, json, math, time, argparse
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
OUTF = os.path.join(HERE, 'runs', 'x65_budget.json')
REFF = os.path.join(HERE, 'runs', 'general.json')
CFGS = {'round0_cull80': dict(P=40, B=16, cull=0.8, init=0.5),
        'best_cull90': dict(P=40, B=16, cull=0.9, init=0.5)}
PROBLEMS = ['xorgauss', 'moons', 'circles', 'spirals', 'blobs3', 'checker3', 'parity4']
SEEDS = list(range(10))
ORDER = ['spirals', 'checker3', 'parity4', 'moons', 'circles', 'xorgauss', 'blobs3']   # longest first
_TASKS = {}


def run_one(label, prob, seed, budget):
    import general as G
    if prob not in _TASKS:
        t = G.NetTask(prob)
        t.target = json.load(open(REFF))['ref'][prob]['target']
        _TASKS[prob] = t
    t0 = time.time()
    cost, solved, gen = G.evolve(_TASKS[prob], CFGS[label], seed, budget)
    return label, prob, seed, [float(cost), bool(solved), int(gen)], round(time.time() - t0, 1)


def summarize(res):
    ref = json.load(open(REFF))
    old = ref['runs']
    gm = lambda v: float(math.exp(sum(math.log(x) for x in v) / len(v)))
    out = {}
    for label, cfg in CFGS.items():
        k = json.dumps(cfg, sort_keys=True)
        rows = {p: [res[f'{label}|{p}|{s}'] for s in SEEDS] for p in PROBLEMS}
        per = {p: float(np.median([r[0] for r in rows[p]])) for p in PROBLEMS}
        par_ok = par_n = 0
        for p in PROBLEMS:
            for s, r in enumerate(rows[p]):
                o = old[k][p][s]
                if o[1]:                                  # solved under the old 3e6 budget
                    par_n += 1
                    par_ok += int(o[0] == r[0] and r[1] and o[2] == r[2])
        out[label] = {'solved_1e7': int(sum(r[1] for p in PROBLEMS for r in rows[p])),
                      'solved_3e6': int(sum(o[1] for p in PROBLEMS for o in old[k][p])), 'total': 70,
                      'gm_vs_default': gm([per[p] / ref['ref'][p]['default'] for p in PROBLEMS]),
                      'gm_vs_tuned': gm([per[p] / ref['ref'][p]['tuned'] for p in PROBLEMS]),
                      'unsolved_per_problem': {p: int(sum(1 for r in rows[p] if not r[1])) for p in PROBLEMS},
                      'median_cost_per_problem': per, 'parity_identical': f'{par_ok}/{par_n}'}
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--budget', type=float, default=1e7)
    a = ap.parse_args()
    sys.path.insert(0, '~/curious-net/vi')
    import budget as BU
    n = a.workers or min(BU.workers(per_worker_gb=1.0), int(os.environ.get('X65_MAX_WORKERS', '6')))
    res = json.load(open(OUTF))['runs'] if os.path.exists(OUTF) else {}
    jobs = [(l, p, s, a.budget) for p in ORDER for l in CFGS for s in SEEDS if f'{l}|{p}|{s}' not in res]
    print(f'{len(jobs)} runs to do, {n} workers, budget {a.budget:g}', flush=True)
    t0 = time.time()
    with mp.get_context('spawn').Pool(n) as pool:
        pending = [pool.apply_async(run_one, j) for j in jobs]
        for i, f in enumerate(pending):
            l, p, s, r, secs = f.get()
            res[f'{l}|{p}|{s}'] = r
            json.dump({'budget': a.budget, 'runs': res}, open(OUTF, 'w'))
            print(f'[{i + 1}/{len(jobs)}] {l} {p} seed {s}: cost {r[0]:,.0f} solved {r[1]} ({secs}s, {time.time() - t0:.0f}s elapsed)', flush=True)
    summ = summarize(res)
    json.dump({'budget': a.budget, 'runs': res, 'summary': summ}, open(OUTF, 'w'))
    print(json.dumps(summ, indent=1))
