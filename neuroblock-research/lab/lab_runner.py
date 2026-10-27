"""Run one lab job.   python lab_runner.py <job_dir>        (runs it)
                      python lab_runner.py --plan <spec.json>  (validates, prints {"ok":..., "total_runs":..., "error":...})

A job directory holds spec.json (input) and gets progress.json plus result files (results.json | grid.json |
compare.json | brain.json). The spec is data only; see lab_core.py for what it may contain.
"""
import datetime
import json
import math
import os
import sys
import time
import traceback
from multiprocessing import Pool, freeze_support

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, r'~\.claude\tools\live-dashboard')
import lab_core as L
import sweep  # Aggregator, agg_spec, paired_z, clean, write_json (module-level code is import-safe)
import livelog

MAX_RUNS = 300000
MAX_VALUES = 40
now = lambda: datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def canon(b, t, u):
    return json.dumps([b, t, u], sort_keys=True)


def read_values(var):
    p = var.get('param')
    if p not in L.PARAMS:
        raise ValueError(f'Unknown setting "{p}".')
    vals, seen = [], set()
    for v in var.get('values', []):
        c = L.coerce(p, v)
        if c not in seen:
            seen.add(c)
            vals.append(c)
    if len(vals) < 2:
        raise ValueError(f'"{p}" needs at least 2 different values to compare.')
    if len(vals) > MAX_VALUES:
        raise ValueError(f'"{p}" has {len(vals)} values; the limit is {MAX_VALUES}.')
    return p, vals


def lab(v):
    return sweep.lab(v)


class Plan:
    """unique (brain, task, us) configurations -> ids, in submission order"""

    def __init__(self, spec):
        self.spec = spec
        self.bc = spec.get('brain_class', 'Brain2')
        self.env = spec.get('env', 'gaussian')
        if self.bc not in L.BRAINS:
            raise ValueError(f'Unknown brain design "{self.bc}". Available: {list(L.BRAINS)}')
        if self.env not in L.ENVS:
            raise ValueError(f'Unknown task type "{self.env}". Available: {list(L.ENVS)}')
        self.base = L.full_config(spec.get('base'))
        self.ids, self.specs, self.desc = {}, [], {}

    def sid(self, cfg, why, bc=None, env=None):
        bc, env = bc or self.bc, env or self.env
        if bc not in L.BRAINS:
            raise ValueError(f'Unknown brain design "{bc}". Available: {list(L.BRAINS)}')
        b, t, u = L.split_config(cfg, bc, env)
        key = canon(b, t, u) + f'|{bc}|{env}'
        if key not in self.ids:
            self.ids[key] = len(self.specs)
            self.specs.append(dict(id=self.ids[key], b=b, t=t, u=u, cfg=cfg, bc=bc, env=env))
            self.desc[self.ids[key]] = why
        return self.ids[key]


def seeds_of(spec):
    n = int(spec.get('seeds', 24))
    if not 2 <= n <= 2000:
        raise ValueError('Seeds must be between 2 and 2000.')
    return list(range(n))


def prepare(spec):
    """validate and expand a spec -> (plan, kind-specific info, tasks)"""
    typ = spec.get('type')
    seeds = seeds_of(spec)
    P = Plan(spec)
    info = dict(type=typ, seeds=seeds)
    record = None
    if typ == 'sweep':
        vars_, plan_vars = spec.get('variables') or [], []
        if not 1 <= len(vars_) <= 30:
            raise ValueError('A sweep needs 1 to 30 variables.')
        base_id = P.sid(P.base, 'baseline')
        for var in vars_:
            p, vals = read_values(var)
            sids = [P.sid({**P.base, p: v}, f'{p} = {lab(v)}') for v in vals]
            plan_vars.append(dict(name=p, group=L.PARAMS[p]['group'], labels=[lab(v) for v in vals], spec_ids=sids,
                                  ref_id=base_id, base_idx=sids.index(base_id) if base_id in sids else -1))
        info['plan'] = {'lab': dict(label=spec.get('title') or 'My sweep', base_id=base_id, vars=plan_vars)}
    elif typ == 'grid':
        (px, vx), (py, vy) = read_values(spec.get('x') or {}), read_values(spec.get('y') or {})
        if px == py:
            raise ValueError('Pick two different settings for the grid.')
        if len(vx) * len(vy) > 400:
            raise ValueError('A grid is limited to 400 cells.')
        info.update(px=px, py=py, vx=vx, vy=vy, base_id=P.sid(P.base, 'baseline'),
                    cells=[[P.sid({**P.base, px: a, py: b}, f'{px}={lab(a)}, {py}={lab(b)}') for a in vx] for b in vy])
    elif typ == 'compare':
        over = spec.get('b') or {}
        if not over and not (spec.get('b_brain_class') and spec['b_brain_class'] != P.bc):
            raise ValueError('Design B needs at least one setting (or a different brain design) that differs from design A.')
        cfg_b = L.full_config(over, P.base)
        b_bc = spec.get('b_brain_class') or P.bc
        diff = [dict(param=k, a=P.base[k], b=cfg_b[k]) for k in P.base if P.base[k] != cfg_b[k]]
        if b_bc != P.bc:
            diff.insert(0, dict(param='brain design', a=P.bc, b=b_bc))
        if not diff:
            raise ValueError('Design B is identical to design A.')
        record = 'series'
        info.update(a_id=P.sid(P.base, 'A'), b_id=P.sid(cfg_b, 'B', bc=b_bc), diff=diff,
                    labels=spec.get('labels') or dict(a='Design A', b='Design B'))
        if info['a_id'] == info['b_id']:
            raise ValueError('Design B is identical to design A.')
    elif typ == 'search':
        rngs = spec.get('ranges') or []
        if not 1 <= len(rngs) <= 30:
            raise ValueError('A search needs 1 to 30 settings to vary.')
        n = int(spec.get('samples', 200))
        if not 8 <= n <= 20000:
            raise ValueError('Samples must be between 8 and 20,000.')
        gen = np.random.default_rng(int(spec.get('search_seed', 12345)))
        cols, parsed = {}, []
        for r in rngs:
            p = r.get('param')
            if p not in L.PARAMS:
                raise ValueError(f'Unknown setting "{p}".')
            pd = L.PARAMS[p]
            if pd['type'] in ('float', 'int'):
                lo, hi = L.coerce(p, r.get('min')), L.coerce(p, r.get('max'))
                if not lo < hi:
                    raise ValueError(f'"{p}": the minimum must be below the maximum.')
                log = r.get('scale') == 'log'
                if log and lo <= 0:
                    raise ValueError(f'"{p}": a log scale needs a minimum above 0.')
                u = (gen.permutation(n) + gen.random(n)) / n  # Latin hypercube: one sample in each of n equal slices
                v = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if log else lo + u * (hi - lo)
                cols[p] = [int(round(x)) if pd['type'] == 'int' else float(x) for x in v]
                parsed.append(dict(param=p, type=pd['type'], min=lo, max=hi, scale='log' if log else 'linear'))
            else:
                choices = [False, True] if pd['type'] == 'bool' else list(pd['choices'])
                sel = r.get('values') or choices
                sel = [L.coerce(p, x) for x in sel]
                cols[p] = [sel[i] for i in gen.permutation(np.resize(np.arange(len(sel)), n))]
                parsed.append(dict(param=p, type=pd['type'], values=sel))
        samples = []
        for i in range(n):
            over = {p: cols[p][i] for p in cols}
            samples.append(dict(id=i, params=over, sid=P.sid({**P.base, **{k: L.coerce(k, v) for k, v in over.items()}}, f'sample {i}')))
        info.update(samples=samples, ranges=parsed, base_id=P.sid(P.base, 'starting setup'), acc_floor=float(spec.get('acc_floor', 0.9)))
    elif typ == 'watch':
        seeds = [int(spec.get('seed', 0))]
        info.update(seeds=seeds, a_id=P.sid(P.base, 'watch'))
        record = 'snap'
    else:
        raise ValueError(f'Unknown experiment type "{typ}".')
    tasks = [(s['id'], s['b'], s['t'], s['u'], seed, s['bc'], s['env'], record) for s in P.specs for seed in seeds]
    if len(tasks) > MAX_RUNS:
        raise ValueError(f'That is {len(tasks):,} runs; the limit is {MAX_RUNS:,}. Use fewer values or seeds.')
    return P, info, tasks


def clean_json(o):
    return sweep.clean(o)


def ms(a):
    a = np.array(a, float)
    a = a[np.isfinite(a)]
    return (float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))) if len(a) >= 2 else (None, None)


class SearchAgg:
    """collects finished samples of a search as their repeats complete"""

    def __init__(self, info, runs, seeds, spec, P):
        self.info, self.runs, self.seeds, self.spec, self.P = info, runs, seeds, spec, P
        self.cache = {}

    def complete(self, sid):
        return sid in self.runs and len(self.runs[sid]) == len(self.seeds)

    def metrics(self, sid):
        if sid not in self.cache:
            rs = [self.runs[sid][s] for s in self.seeds]
            keys = [k for k in rs[0] if k not in ('series', 'snaps', 'rec_never')]
            self.cache[sid] = {k: ms([r[k] for r in rs]) for k in keys}
            self.cache[sid]['shut_any'] = (float(np.mean([r.get('shutdowns', 0) > 0 for r in rs])), None)
        return {k: list(v) for k, v in self.cache[sid].items()}

    def result(self, title, partial):
        done = [s for s in self.info['samples'] if self.complete(s['sid'])]
        base = self.metrics(self.info['base_id']) if self.complete(self.info['base_id']) else None
        return clean_json(dict(type='search', title=title, partial=partial, seeds=len(self.seeds), acc_floor=self.info['acc_floor'],
                               ranges=self.info['ranges'], brain=self.P.bc, env=self.P.env, base_overrides=self.spec.get('base') or {},
                               total=len(self.info['samples']), base=base,
                               samples=[dict(id=s['id'], params=s['params'], m=self.metrics(s['sid'])) for s in done]))


def series_mean(runs, seeds):
    out = {}
    ser = [runs[s]['series'] for s in seeds if s in runs]
    for k in ('acc', 't', 'n', 'cov', 'spawns', 'mean_edge'):
        a = np.array([x[k] for x in ser], float)
        out[k] = dict(mean=a.mean(0).tolist(), se=(a.std(0, ddof=1) / np.sqrt(len(a))).tolist() if len(a) > 1 else [0] * a.shape[1])
    out['w'] = ser[0]['w']
    out['drift_w'] = ser[0]['drift_w']
    return out


# ---------------------------------------------------------------- execution
def main(job_dir):
    spec = json.load(open(os.path.join(job_dir, 'spec.json'), encoding='utf-8'))
    cpu = spec.get('cpu') or {}
    workers = max(1, min(int(cpu.get('workers', os.cpu_count() or 1)), os.cpu_count() or 1))
    priority = cpu.get('priority', 'normal')
    lim = L.limits()  # the user's limits always win over what a spec asks for
    workers = max(1, min(workers, lim['max_workers']))
    if L.PRIORITY_RANK.get(priority, 1) > L.PRIORITY_RANK[lim['priority_cap']]:
        priority = lim['priority_cap']
    prog_path = os.path.join(job_dir, 'progress.json')
    cancel_path = os.path.join(job_dir, 'cancel.flag')
    started = now()

    def progress(done, total, current, status='running', error=None, extra=None):
        p = dict(status=status, phase={'running': 'Running', 'done': 'Finished', 'cancelled': 'Cancelled', 'error': 'Failed'}[status],
                 done=done, total=total, current=current, started=started, updated=now(), workers=workers, priority=priority)
        if status != 'running':
            p['finished'] = p['updated']
        if error:
            p['error'] = error
        if extra:
            p.update(extra)
        sweep.write_json(prog_path, p)

    try:
        P, info, tasks = prepare(spec)
    except Exception as e:
        progress(0, 0, '', 'error', str(e))
        livelog.log('error', f'Lab job "{spec.get("title", "?")}" could not start: {e}', 'lab')
        return 2
    sweep.SEEDS = info['seeds']
    seeds, total = info['seeds'], len(tasks)
    title = spec.get('title') or spec['type']
    keep_awake = bool(cpu.get('keep_awake', True))
    if os.name == 'nt' and keep_awake:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    L.set_priority(priority)
    livelog.log('step', f'Lab job "{title}" started: {total:,} runs on {workers} CPU threads ({priority} priority)', 'lab')
    runs, done, errors, last = {}, 0, [], 0.0
    plan = info.get('plan')
    agg = sweep.Aggregator(plan) if plan else None
    if agg:
        agg.runs = runs
    sa = SearchAgg(info, runs, seeds, spec, P) if spec['type'] == 'search' else None
    meta = dict(seeds=len(seeds), updated=started, note='95% intervals use 1.96 x standard error; z is paired by seed against the reference.')
    finished = set()
    t0 = time.time()
    status = 'done'
    try:
        with Pool(workers, initializer=L.worker_init, initargs=(priority,)) as pool:
            for key, seed, res, secs, err in pool.imap_unordered(L.worker, tasks, chunksize=1 if total < 400 else 2):
                done += 1
                if err:
                    errors.append(f'{P.desc[key]} seed {seed}: {err.splitlines()[0]}')
                    if len(errors) > 25:
                        raise RuntimeError('Too many failed runs; first error: ' + errors[0])
                else:
                    runs.setdefault(key, {})[seed] = dict(res['metrics'], **({'series': res['series']} if 'series' in res else {}),
                                                        **({'snaps': res['snaps']} if 'snaps' in res else {}))
                if os.path.exists(cancel_path):
                    status = 'cancelled'
                    pool.terminate()
                    break
                if time.time() - last > 1.0:
                    last = time.time()
                    progress(done, total, P.desc[key], extra=dict(rate=round(done / max(time.time() - t0, 1e-6), 2)))
                    if agg:
                        meta['updated'] = now()
                        sweep.write_json(os.path.join(job_dir, 'results.json'), agg.results(finished, meta))
                    if sa:
                        sweep.write_json(os.path.join(job_dir, 'search.json'), sa.result(title, True))
    except Exception as e:
        status = 'error'
        err_text = f'{type(e).__name__}: {e}'
        progress(done, total, '', 'error', err_text)
        livelog.log('error', f'Lab job "{title}" failed: {err_text}', 'lab')
    finally:
        if os.name == 'nt' and keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    if status == 'error':
        return 1
    if status == 'cancelled':
        progress(done, total, '', 'cancelled')
        livelog.log('warn', f'Lab job "{title}" was cancelled after {done:,} of {total:,} runs', 'lab')
        return 0

    # ---- final result files
    typ = spec['type']
    if typ == 'sweep':
        finished.add('lab')
        meta['updated'] = now()
        sweep.write_json(os.path.join(job_dir, 'results.json'), agg.results(finished, meta))
    elif typ == 'grid':
        def cell(sid):
            a = sweep.agg_spec({s: runs[sid][s] for s in seeds if s in runs[sid]}) if sid in runs and len(runs[sid]) == len(seeds) else None
            return {k: dict(mean=a[k][0], se=a[k][1]) for k in sweep.MK} if a else None
        out = dict(type='grid', title=title, seeds=len(seeds),
                   x=dict(param=info['px'], group=L.PARAMS[info['px']]['group'], labels=[lab(v) for v in info['vx']]),
                   y=dict(param=info['py'], group=L.PARAMS[info['py']]['group'], labels=[lab(v) for v in info['vy']]),
                   base=cell(info['base_id']), cells=[[cell(s) for s in row] for row in info['cells']])
        sweep.write_json(os.path.join(job_dir, 'grid.json'), clean_json(out))
    elif typ == 'compare':
        ra, rb = runs[info['a_id']], runs[info['b_id']]
        out = dict(type='compare', title=title, seeds=len(seeds), labels=info['labels'], config_diff=info['diff'], metrics={}, series={})
        for k in sweep.ZK + ['sd']:
            if k == 'sd':
                sa, sb = (float(np.std([ra[s]['acc'] for s in seeds], ddof=1)), float(np.std([rb[s]['acc'] for s in seeds], ddof=1)))
                out['metrics'][k] = dict(a=dict(mean=sa, se=None), b=dict(mean=sb, se=None), diff=None)
                continue
            a = np.array([ra[s][k] for s in seeds], float)
            b = np.array([rb[s][k] for s in seeds], float)
            d = (b - a)
            d = d[np.isfinite(d)]
            z = sweep.paired_z(rb, ra, k)
            dm, dse = (float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))) if len(d) >= 2 else (None, None)
            out['metrics'][k] = dict(a=dict(zip(('mean', 'se'), ms(a))), b=dict(zip(('mean', 'se'), ms(b))),
                                     diff=dict(mean=dm, se=dse, z=z, lo=None if dm is None else dm - 1.96 * dse,
                                               hi=None if dm is None else dm + 1.96 * dse))
        out['series'] = dict(a=series_mean(ra, seeds), b=series_mean(rb, seeds))
        sweep.write_json(os.path.join(job_dir, 'compare.json'), clean_json(out))
    elif typ == 'search':
        sweep.write_json(os.path.join(job_dir, 'search.json'), sa.result(title, False))
    elif typ == 'watch':
        r = runs[info['a_id']][seeds[0]]
        out = dict(type='watch', title=title, seed=seeds[0], config=P.base, metrics={k: v for k, v in r.items() if k not in ('series', 'snaps')},
                   series=r['series'], snaps=r['snaps'])
        sweep.write_json(os.path.join(job_dir, 'brain.json'), clean_json(out))
    el = time.time() - t0
    progress(done, total, 'all runs', 'done', extra=dict(rate=round(done / max(el, 1e-6), 2), failed=len(errors)))
    livelog.log('done', f'Lab job "{title}" finished: {done:,} runs in {el:.0f} s ({len(errors)} failed)', 'lab')
    return 0


if __name__ == '__main__':
    freeze_support()
    if len(sys.argv) >= 3 and sys.argv[1] == '--plan':
        try:
            spec = json.load(open(sys.argv[2], encoding='utf-8'))
            P, info, tasks = prepare(spec)
            print(json.dumps(dict(ok=True, total_runs=len(tasks), configs=len(P.specs), seeds=len(info['seeds']))))
        except Exception as e:
            print(json.dumps(dict(ok=False, error=str(e))))
        sys.exit(0)
    sys.exit(main(sys.argv[1]))
