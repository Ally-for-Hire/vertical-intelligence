"""One-variable-at-a-time sweep of every neuroblock / Brain2 variable, streamed to the live dashboard.

Design
- A baseline configuration per condition; each variable is varied alone while everything else stays at baseline.
- Same seeds for every setting, so each setting is compared with the baseline PAIRED by seed (common random numbers).
- Conditions: medium/hard task x shift/replace drift (all features on), plus two "solo" conditions that add each
  feature alone to a bare brain.
- run_ext() is run() from neuroblock.py with four hard-coded constants exposed (shift size, class skew, number of
  replaced concepts); validate() proves it matches the original bit for bit at the defaults.

Usage: python sweep.py [--quick] [--procs 12]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
import csv
import datetime
import json
import math
import sys
import time
from multiprocessing import Pool, freeze_support

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'~\.claude\tools\live-dashboard')
import neuroblock
from neuroblock import spearman, summarize
from brain2 import Brain2
import livelog

LIVE = r'~\claude-live\projects\neuroblock'
OUT = os.path.join(HERE, 'sweeps')
SEEDS = list(range(24))
PROJECT = 'neuroblock'

BASE_BRAIN = dict(rho=0.5, move=True, merge=True, prune=True, select='fast', lr=0.1, min_sep=0.3, spawn_len=1.0,
                  eta=0.5, spawn_cost=50.0, rigid=False, c0=30.0, a0=5.0, expand=False, support=8, frac=0.5,
                  gain_frac=0.15, buf=400)
BASE_TASK = dict(K=8, D=8, sigma=0.10, T=6000, win=200, drift=True, drift_mode='shift', shift_sd=0.12, zipf=1.0,
                 n_replace=3)
BARE = dict(move=False, merge=False, prune=False)

CONDS = {
    'medium|shift': dict(label='Medium task · shift drift', brain={}, task=dict(), mode='oat'),
    'hard|shift': dict(label='Hard task · shift drift', brain=dict(rho=0.7), task=dict(sigma=0.15, K=10), mode='oat'),
    'medium|replace': dict(label='Medium task · replace drift', brain={}, task=dict(drift_mode='replace'), mode='oat'),
    'hard|replace': dict(label='Hard task · replace drift', brain=dict(rho=0.7),
                         task=dict(sigma=0.15, K=10, drift_mode='replace'), mode='oat'),
    'solo medium|shift': dict(label='Medium · each feature alone on a bare brain', brain=dict(BARE), task=dict(),
                              mode='solo'),
    'solo hard|shift': dict(label='Hard · each feature alone on a bare brain', brain=dict(BARE, rho=0.7),
                            task=dict(sigma=0.15, K=10), mode='solo'),
}


def V(name, group, kind, param, values, **kw):
    return dict(name=name, group=group, kind=kind, param=param, values=values, **kw)


VARS = [
    V('rho', 'Geometry', 'brain', 'rho', [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
    V('min_sep', 'Geometry', 'brain', 'min_sep', [0, 0.1, 0.2, 0.3, 0.5, 0.8]),
    V('spawn_len', 'Geometry', 'brain', 'spawn_len', [0.25, 0.5, 1.0, 2.0, 4.0]),
    V('us_per_unit', 'Geometry', 'us', 'us', [10, 20, 40, 80, 160]),
    V('move', 'Condensing', 'brain', 'move', [False, True]),
    V('merge', 'Condensing', 'brain', 'merge', [False, True]),
    V('prune', 'Condensing', 'brain', 'prune', [False, True]),
    V('eta', 'Condensing', 'brain', 'eta', [0.1, 0.25, 0.5, 0.75, 1.0]),
    V('select', 'Selection', 'brain', 'select', ['fast', 'best']),
    V('lr', 'Hardening', 'brain', 'lr', [0.01, 0.03, 0.1, 0.2, 0.4]),
    V('rigid', 'Hardening', 'brain', 'rigid', [False, True]),
    V('c0 (rigid on)', 'Hardening', 'brain', 'c0', [3, 10, 30, 100, 300, 1e9], extras=dict(rigid=True)),
    V('a0 (rigid on)', 'Hardening', 'brain', 'a0', [1, 2, 5, 10, 20, 50], extras=dict(rigid=True)),
    V('spawn_cost', 'Timing', 'brain', 'spawn_cost', [0, 25, 50, 100, 200]),
    V('win', 'Timing', 'task', 'win', [50, 100, 200, 300, 400]),
    V('T', 'Timing', 'task', 'T', [2400, 4000, 6000, 12000]),
    V('expand', 'Expansion', 'brain', 'expand', [False, True]),
    V('support (expand on)', 'Expansion', 'brain', 'support', [3, 5, 8, 12, 20], extras=dict(expand=True)),
    V('frac (expand on)', 'Expansion', 'brain', 'frac', [0.25, 0.4, 0.5, 0.7, 0.9], extras=dict(expand=True)),
    V('gain_frac (expand on)', 'Expansion', 'brain', 'gain_frac', [0.05, 0.15, 0.3, 0.6, 1.0], extras=dict(expand=True)),
    V('buf (expand on)', 'Expansion', 'brain', 'buf', [100, 200, 400, 800], extras=dict(expand=True)),
    V('sigma', 'Environment', 'task', 'sigma', [0.03, 0.06, 0.10, 0.15, 0.2, 0.3]),
    V('K (classes)', 'Environment', 'task', 'K', [3, 5, 8, 10, 12, 16, 24]),
    V('D (inputs)', 'Environment', 'task', 'D', [2, 4, 8, 16, 32]),
    V('drift', 'Environment', 'drift', 'drift', ['none', 'replace', 'shift']),
    V('zipf (class skew)', 'Environment', 'task', 'zipf', [0, 0.5, 1.0, 1.5, 2.0]),
    V('shift_sd', 'Environment', 'task', 'shift_sd', [0, 0.05, 0.12, 0.2, 0.3], only='shift'),
    V('n_replace', 'Environment', 'task', 'n_replace', [1, 2, 3, 4, 6], only='replace'),
]
SOLO_VARS = [
    V('move', 'Condensing', 'brain', 'move', [False, True]),
    V('merge', 'Condensing', 'brain', 'merge', [False, True]),
    V('prune', 'Condensing', 'brain', 'prune', [False, True]),
    V('rigid', 'Hardening', 'brain', 'rigid', [False, True]),
    V('expand', 'Expansion', 'brain', 'expand', [False, True]),
    V('select', 'Selection', 'brain', 'select', ['fast', 'best']),
]
ENERGY = ['uptime', 'shutdowns', 'E', 'Eleft', 'e_move', 'e_think', 'e_spawn', 'e_upkeep', 'e_other', 'eff', 'mastery', 'sleep_share', 'sleeps', 'tired_sleeps']  # only filled by energy brains
MK = ['acc', 't', 'n', 'F', 'edge', 'rec', 'sd', 'cov', 'spawns', 'exp'] + ENERGY  # metrics sent to the dashboard
ZK = ['acc', 't', 'n', 'F', 'edge', 'rec', 'cov', 'spawns', 'exp'] + ENERGY  # metrics with a paired z-score
RUN_KEYS = ['acc', 't', 'n', 'F', 'edge', 'rec', 'cov', 'spawns', 'exp', 'pre_acc', 'pre_t', 'pre_n', 'rec_never'] + ENERGY


def lab(v):
    if isinstance(v, bool):
        return 'on' if v else 'off'
    if isinstance(v, float) and v >= 1e8:
        return 'never'
    return f'{v:g}' if isinstance(v, (int, float)) else str(v)


# ---------------------------------------------------------------- simulation
def run_ext(cfg, seed, K=6, D=8, sigma=0.06, T=6000, win=200, drift=True, drift_mode='replace',
            shift_sd=0.12, zipf=1.0, n_replace=3):
    """neuroblock.run() with the hard-coded environment constants exposed. Defaults reproduce run() exactly."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 1, (K, D))
    probs = 1 / np.arange(1, K + 1) ** zipf
    probs /= probs.sum()
    b = Brain2(D, K, rng=np.random.default_rng(seed + 1000), **cfg)
    W = T // win
    dw = W // 2
    rows = []

    def sample(n):
        y = rng.choice(K, n, p=probs)
        return centers[y] + rng.normal(0, sigma, (n, D)), y

    for w in range(W):
        if drift and w == dw:
            if drift_mode == 'replace':
                sel = [i for i in range(1, 1 + n_replace) if i < K]
                centers[sel] = rng.uniform(0, 1, (len(sel), D))
            else:
                centers += rng.normal(0, shift_sd, centers.shape)
        X, Y = sample(win)
        ts, sp = [], 0
        for x, y in zip(X, Y):
            t, _, spawned = b.step(x, y)
            ts.append(t)
            sp += spawned
            b.remember(x, y)
        n_before = b.n
        b.condense()
        Xt, Yt = sample(400)
        tq, acc, cov = b.query(Xt, Yt)
        rows.append(dict(w=w, t_train=float(np.mean(ts)), spawns=sp, n_before=n_before, n=b.n, t_query=tq, acc=acc,
                         cov=cov, exp=getattr(b, 'exp_total', 0),
                         depth_corr=spearman(b.cnt[1:], b.d[1:]) if b.n > 2 else np.nan,
                         mean_edge=float(np.mean([np.linalg.norm(b.pos[i] - b.pos[b.par[i]])
                                                  for i in range(1, len(b.pos))])) if b.n else 0.0))
    return rows, dw


def metrics_of(rows, dw, K):
    o = summarize(rows, dw, K=K)
    rec = o['recovery_windows']
    fin = math.isfinite(rec)
    return dict(acc=o['post_acc'], t=o['post_t_query'], n=o['post_n'], F=o['post_F'], edge=o['post_mean_edge'],
                rec=float(rec) if fin else float('nan'), rec_never=0.0 if fin else 1.0, pre_acc=o['pre_acc'],
                pre_t=o['pre_t_query'], pre_n=o['pre_n'], cov=o['post_cov'], spawns=o['post_spawns'],
                exp=float(rows[-1]['exp']))


def worker(task):
    sid, bcfg, tcfg, us, seed = task
    t0 = time.time()
    try:
        neuroblock.US = float(us)
        rows, dw = run_ext(bcfg, seed, **tcfg)
        return sid, seed, metrics_of(rows, dw, tcfg['K']), time.time() - t0, None
    except Exception as e:  # keep the sweep alive; the dashboard shows a gap
        return sid, seed, None, time.time() - t0, f'{type(e).__name__}: {e}'


# ---------------------------------------------------------------- plan
def apply(var, b, t, u, val):
    b, t = dict(b), dict(t)
    b.update(var.get('extras', {}))
    k = var['kind']
    if k == 'brain':
        b[var['param']] = val
    elif k == 'task':
        t[var['param']] = val
    elif k == 'us':
        u = val
    elif k == 'drift':
        if val == 'none':
            t['drift'] = False
        else:
            t['drift'] = True
            t['drift_mode'] = val
    return b, t, u


def make_plan(cond_keys):
    ids, spec_list, desc = {}, [], {}

    def sid_of(b, t, u, why):
        key = json.dumps([b, t, u], sort_keys=True)
        if key not in ids:
            ids[key] = len(spec_list)
            spec_list.append(dict(id=ids[key], b=b, t=t, u=u))
            desc[ids[key]] = why
        return ids[key]

    plan = {}
    for ck in cond_keys:
        c = CONDS[ck]
        base_b, base_t = {**BASE_BRAIN, **c['brain']}, {**BASE_TASK, **c['task']}
        entry = dict(label=c['label'], vars=[], base_id=sid_of(base_b, base_t, 40, f'{ck} · baseline'))
        for var in (VARS if c['mode'] == 'oat' else SOLO_VARS):
            if var.get('only') and var['only'] != c['task'].get('drift_mode', 'shift'):
                continue
            ref = sid_of({**base_b, **var.get('extras', {})}, base_t, 40, f'{ck} · {var["name"]} reference')
            sids = []
            for val in var['values']:
                b, t, u = apply(var, base_b, base_t, 40, val)
                sids.append(sid_of(b, t, u, f'{ck} · {var["name"]} = {lab(val)}'))
            entry['vars'].append(dict(name=var['name'], group=var['group'], labels=[lab(v) for v in var['values']],
                                      spec_ids=sids, ref_id=ref, base_idx=sids.index(ref) if ref in sids else -1))
        plan[ck] = entry
    return plan, spec_list, desc


# ---------------------------------------------------------------- aggregation
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def agg_spec(runs):
    out = {}
    for k in ['acc', 't', 'n', 'F', 'edge', 'rec', 'cov', 'spawns', 'exp', 'pre_acc'] + ENERGY:
        a = np.array([runs[s].get(k, float('nan')) for s in SEEDS], float)
        a = a[np.isfinite(a)]
        out[k] = (float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))) if len(a) >= 2 else (None, None)
    acc = np.array([runs[s]['acc'] for s in SEEDS], float)
    sd = float(np.std(acc, ddof=1))
    out['sd'] = (sd, sd / np.sqrt(2 * (len(SEEDS) - 1)))
    return out


def paired_z(ra, rb, k):
    a = np.array([ra[s].get(k, float('nan')) for s in SEEDS], float)
    b = np.array([rb[s].get(k, float('nan')) for s in SEEDS], float)
    d = (a - b)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return None
    sdv = d.std(ddof=1)
    if sdv == 0:
        return 0.0 if d.mean() == 0 else math.copysign(99.0, d.mean())
    return float(max(-99.0, min(99.0, d.mean() / (sdv / np.sqrt(len(d))))))


class Aggregator:
    def __init__(self, plan):
        self.plan, self.runs, self.agg, self.zc = plan, {}, {}, {}

    def complete(self, sid):
        return sid in self.runs and len(self.runs[sid]) == len(SEEDS)

    def get_agg(self, sid):
        if not self.complete(sid):
            return None
        if sid not in self.agg:
            self.agg[sid] = agg_spec(self.runs[sid])
        return self.agg[sid]

    def get_z(self, sid, ref, k):
        if not (self.complete(sid) and self.complete(ref)):
            return None
        key = (sid, ref, k)
        if key not in self.zc:
            self.zc[key] = 0.0 if sid == ref else paired_z(self.runs[sid], self.runs[ref], k)
        return self.zc[key]

    def var_entry(self, v):
        e = dict(group=v['group'], labels=v['labels'], base_idx=v['base_idx'],
                 m={k: dict(mean=[], se=[]) for k in MK}, z={k: [] for k in ZK}, ref={})
        for sid in v['spec_ids']:
            a = self.get_agg(sid)
            for k in MK:
                e['m'][k]['mean'].append(a[k][0] if a else None)
                e['m'][k]['se'].append(a[k][1] if a else None)
            for k in ZK:
                e['z'][k].append(self.get_z(sid, v['ref_id'], k))
        ra = self.get_agg(v['ref_id'])
        if ra:
            e['ref'] = {k: dict(mean=ra[k][0], se=ra[k][1]) for k in MK}
        return e

    def results(self, finished, meta):
        conds = {}
        for ck, p in self.plan.items():
            ba = self.get_agg(p['base_id'])
            conds[ck] = dict(label=p['label'], done=ck in finished,
                             baseline={k: dict(mean=ba[k][0], se=ba[k][1]) for k in MK} if ba else {},
                             vars={v['name']: self.var_entry(v) for v in p['vars']})
        return clean(dict(meta=meta, conditions=conds))


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, allow_nan=False)
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)


def short_effect(e):
    """one-line range summary for the activity feed"""
    def rng(k, scale=1.0):
        v = [x for x in e['m'][k]['mean'] if x is not None]
        return (min(v) * scale, max(v) * scale) if v else (float('nan'),) * 2
    zmax = max([abs(z) for k in ('acc', 't', 'n', 'F') for z in e['z'][k] if z is not None] or [0])
    a, t, n = rng('acc', 100), rng('t'), rng('n')
    return (f"accuracy {a[0]:.1f}-{a[1]:.1f}%, thought {t[0]:.0f}-{t[1]:.0f} µs, neurons {n[0]:.0f}-{n[1]:.0f} "
            f"→ {'beyond noise' if zmax > 3 else 'within noise'} (max |z| {zmax:.1f})")


# ---------------------------------------------------------------- validation
def validate():
    cfg = dict(rho=0.5, move=True, merge=True, prune=True)
    task = dict(sigma=0.10, K=8)
    ok = True
    for seed in (0, 1):
        r_orig, _, dw0 = neuroblock.run(cfg, seed, cls=Brain2, drift_mode='shift', **task)
        r_new, dw1 = run_ext({**BASE_BRAIN, **cfg}, seed, drift_mode='shift', **task, D=8, T=6000, win=200)
        keys = ['t_query', 'acc', 'n', 'cov', 'spawns', 'mean_edge', 'exp']
        same = dw0 == dw1 and all(a[k] == b[k] for a, b in zip(r_orig, r_new) for k in keys)
        ok = ok and same
    return ok


def main():
    global SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--procs', type=int, default=12)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LIVE, exist_ok=True)
    cond_keys = ['medium|shift', 'solo medium|shift'] if args.quick else list(CONDS)
    if args.quick:
        SEEDS = list(range(3))

    ok = validate()
    livelog.log('result' if ok else 'error',
                'Harness check: extended run() reproduces neuroblock.run() bit-for-bit on seeds 0-1'
                if ok else 'Harness check FAILED: extended run() differs from neuroblock.run()', PROJECT)
    if not ok:
        sys.exit(1)

    plan, specs, desc = make_plan(cond_keys)
    tasks = [(s['id'], s['b'], s['t'], s['u'], seed) for s in specs for seed in SEEDS]
    total = len(tasks)
    agg = Aggregator(plan)
    sid_vars = {}
    for ck, p in plan.items():
        for v in p['vars']:
            for sid in v['spec_ids'] + [v['ref_id']]:
                sid_vars.setdefault(sid, []).append((ck, v))
    var_logged, finished = set(), set()
    cond_of_spec = {}
    for ck, p in plan.items():
        for v in p['vars']:
            for sid in v['spec_ids'] + [v['ref_id']]:
                cond_of_spec.setdefault(sid, ck)
    started = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    meta = dict(seeds=len(SEEDS), updated=started, baseline_brain=BASE_BRAIN, baseline_task=BASE_TASK,
                note='95% intervals use 1.96 x standard error across seeds; z is paired by seed against the reference setting.')
    livelog.log('step', f'Sweep started: {len(specs)} unique configurations x {len(SEEDS)} seeds = {total:,} runs on '
                        f'{args.procs} CPU cores, {len(plan)} conditions', PROJECT)

    def push(done, current, cond, final=False):
        p = dict(phase=('Sweep complete' if final else f'Sweeping · {plan[cond]["label"]}') if cond else 'Starting',
                 done=done, total=total, current=current, cond=cond, started=started,
                 updated=datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
        if final:
            p['finished'] = p['updated']
        write_json(os.path.join(LIVE, 'progress.json'), p)
        meta['updated'] = p['updated']
        write_json(os.path.join(LIVE, 'results.json'), agg.results(finished, meta))

    push(0, 'starting', cond_keys[0])
    done, last_push, errors = 0, time.time(), 0
    cur, curcond = 'starting', cond_keys[0]
    t_start = time.time()
    with Pool(args.procs) as pool:
        for sid, seed, m, secs, err in pool.imap_unordered(worker, tasks, chunksize=2):
            done += 1
            cur, curcond = desc[sid], cond_of_spec.get(sid, curcond)
            if err:
                errors += 1
                livelog.log('warn', f'Run failed ({desc[sid]}, seed {seed}): {err}', PROJECT)
                continue
            agg.runs.setdefault(sid, {})[seed] = m
            if agg.complete(sid):
                for ck, v in sid_vars.get(sid, []):
                    key = (ck, v['name'])
                    if key in var_logged or not all(agg.complete(x) for x in v['spec_ids'] + [v['ref_id']]):
                        continue
                    var_logged.add(key)
                    if ck in ('medium|shift', 'hard|shift') or ck.startswith('solo'):
                        livelog.log('result', f'{ck}: {v["name"]} - {short_effect(agg.var_entry(v))}', PROJECT)
                    if all((c, vv['name']) in var_logged for c, p in plan.items() if c == ck for vv in p['vars']):
                        if ck not in finished:
                            finished.add(ck)
                            livelog.log('done', f'Finished condition "{plan[ck]["label"]}" '
                                                f'({len(finished)}/{len(plan)})', PROJECT)
            if time.time() - last_push > 1.5:
                push(done, cur, curcond)
                last_push = time.time()
    push(done, 'all configurations', cond_keys[-1], final=True)

    # raw per-run table + plan, for documentation and later analysis
    with open(os.path.join(OUT, 'raw_runs.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['spec_id', 'seed'] + RUN_KEYS + ['config'])
        for s in specs:
            for seed in SEEDS:
                r = agg.runs.get(s['id'], {}).get(seed)
                if r:
                    w.writerow([s['id'], seed] + [r[k] for k in RUN_KEYS] + [json.dumps([s['b'], s['t'], s['u']], sort_keys=True)])
    write_json(os.path.join(OUT, 'plan.json'), dict(plan=plan, specs=[dict(id=s['id'], desc=desc[s['id']], b=s['b'], t=s['t'], u=s['u']) for s in specs]))
    write_json(os.path.join(OUT, 'results.json'), agg.results(finished, meta))
    el = time.time() - t_start
    livelog.log('done', f'Sweep complete: {done:,} runs in {el/60:.1f} min ({errors} failed). Raw data saved to '
                        f'neuroblock-research\\sweeps. Analysis next.', PROJECT)


if __name__ == '__main__':
    freeze_support()
    main()
