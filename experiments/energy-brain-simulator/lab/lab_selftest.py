"""Self-test for the lab: python lab_selftest.py   (about a minute, uses a few CPU threads)"""
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PROJ)
import lab_core as L
import sweep

fails = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  {detail}' if detail else ''))
    if not ok:
        fails.append(name)


# 1. the lab engine reproduces the sweep harness (and so the original simulator) exactly
for seed in (0, 1):
    cfg = L.full_config({})
    b, t, u = L.split_config(cfg)
    got = L.simulate(b, t, u, seed)['metrics']
    rows, dw = sweep.run_ext({**sweep.BASE_BRAIN}, seed, **sweep.BASE_TASK)
    want = sweep.metrics_of(rows, dw, sweep.BASE_TASK['K'])
    same = all(got[k] == want[k] or (got[k] != got[k] and want[k] != want[k]) for k in want)
    check(f'engine == sweep harness (seed {seed})', same)

# 1b. the lean brain with both self-adjustments off is exactly the original brain at its default settings
for seed in (0, 1):
    cfgb = L.full_config({})
    b2, t2, u2 = L.split_config(cfgb, 'Brain2')
    bl, tl, ul = L.split_config(L.full_config({'tune': False, 'deliberate': False}), 'LeanBrain')
    check('lean brain keeps no legacy settings', not (set(bl) & L.LEGACY), str(sorted(set(bl))))
    a = L.simulate(b2, t2, u2, seed)['metrics']
    l = L.simulate(bl, tl, ul, seed, 'LeanBrain')['metrics']
    check(f'LeanBrain(off, off) == Brain2 (seed {seed})', all(a[k] == l[k] or (a[k] != a[k] and l[k] != l[k]) for k in a))

# 1c. energy brain: with every price at zero it is exactly the lean brain; the economy then behaves as designed
FREE = dict(e_seed=9000, e_cap=1000, e_birth=0, regen=0, floor=0.3, economize=False)
ZERO = dict(c_move=0, squeeze_k=0, c_think=0, c_spawn=0, c_learn=0, upkeep=0, c_merge=0, c_prune=0, tune_energy=0, deliberate_energy=0,
            c_sense=0, c_basal=0)  # the Lab refuses zero prices (nothing is free); the code itself still allows them for this one check
for name in [k for k in ZERO if k != 'squeeze_k']:  # squeeze_k is a shape exponent, not a price: 0 just means no extra squeeze price
    try:
        L.full_config({name: 0})
        check(f'Lab refuses a free {name}', False)
    except ValueError:
        pass
check('Lab refuses every zero price (nothing is free)', True)
for seed in (0, 1):
    bl, tl, ul = L.split_config(L.full_config({}), 'LeanBrain')
    be, te, ue = L.split_config(L.full_config(FREE), 'EnergyBrain')
    be.update(ZERO)
    a = L.simulate(bl, tl, ul, seed, 'LeanBrain')['metrics']
    e = L.simulate(be, te, ue, seed, 'EnergyBrain')['metrics']
    check(f'EnergyBrain(all prices 0) == LeanBrain (seed {seed})', all(a[k] == e[k] or (a[k] != a[k] and e[k] != e[k]) for k in a if k in e and k not in L.ENERGY_KEYS + ['eff']),
          str({k: (a[k], e[k]) for k in a if k in e and a[k] != e[k] and k not in L.ENERGY_KEYS + ['eff']}))


def energy_run(over, seeds=range(4)):
    out = []
    for s in seeds:
        b, t, u = L.split_config(L.full_config(over), 'EnergyBrain')
        out.append(L.simulate(b, t, u, s, 'EnergyBrain')['metrics'])
    return {k: sum(x[k] for x in out) / len(out) for k in out[0] if out[0][k] == out[0][k]}


rich = dict(e_seed=9000, e_cap=1000, e_birth=0)
on, off = energy_run({**rich, 'economize': True}), energy_run({**rich, 'economize': False})
check('natural economy saves energy (>5x less than cost-blind squeezing)', on['E'] * 5 < off['E'], f'{on["E"]:.0f} vs {off["E"]:.0f} E per 1000')
check('natural economy stays accurate', on['acc'] > 0.95, f'acc {on["acc"]:.3f}')
starve = energy_run(dict(regen=0.0, e_seed=3.0))
check('no income -> forced shutdowns and low uptime', starve['shutdowns'] >= 2 and starve['uptime'] < 0.5, f'{starve["shutdowns"]:.1f} shutdowns, uptime {starve["uptime"]:.2f}')
perm = energy_run(dict(regen=0.0, e_seed=3.0, reboot_after=0))
check('reboot_after=0 -> one permanent shutdown', perm['shutdowns'] == 1, f'{perm["shutdowns"]:.1f}')
healthy = energy_run({})
check('default economy is viable', healthy['uptime'] > 0.9 and healthy['acc'] > 0.95, f'uptime {healthy["uptime"]:.2f}, acc {healthy["acc"]:.3f}')
over_on = energy_run(dict(overload_limit=0.02, fatigue_gain=0.9))
check('overload limit forces shutdowns that the default does not', over_on['shutdowns'] > healthy['shutdowns'], f'{over_on["shutdowns"]:.1f} vs {healthy["shutdowns"]:.1f}')
fat0 = energy_run(dict(overload_limit=0.05, fatigue_gain=0.0))
fat9 = energy_run(dict(overload_limit=0.05, fatigue_gain=0.95, fatigue_windows=10))
check('fatigue lowers the overload threshold (more shutdowns with fatigue)', fat9['shutdowns'] >= fat0['shutdowns'], f'{fat9["shutdowns"]:.1f} vs {fat0["shutdowns"]:.1f}')

# 2. validation rejects bad input with a readable message
for name, fn in [('unknown setting', lambda: L.full_config({'bogus': 1})), ('out of range', lambda: L.full_config({'rho': 99})),
                 ('bad choice', lambda: L.full_config({'select': 'nope'})), ('NaN', lambda: L.full_config({'rho': float('nan')})),
                 ('too few windows', lambda: L.split_config(L.full_config({'T': 400, 'win': 200})))]:
    try:
        fn()
        check(f'rejects {name}', False)
    except ValueError as e:
        check(f'rejects {name}', True, str(e)[:60])

# 3. one job of every type, through the real runner process
tmp = tempfile.mkdtemp(prefix='labtest_')
specs = {
    'sweep': dict(type='sweep', title='t', seeds=6, cpu=dict(workers=4, priority='normal', keep_awake=False),
                  variables=[dict(param='rho', values=[0.4, 0.5, 0.6]), dict(param='select', values=['fast', 'best'])]),
    'grid': dict(type='grid', title='t', seeds=3, cpu=dict(workers=4), x=dict(param='rho', values=[0.4, 0.5]),
                 y=dict(param='sigma', values=[0.1, 0.15])),
    'compare': dict(type='compare', title='t', seeds=6, cpu=dict(workers=4), b=dict(select='best')),
    'watch': dict(type='watch', title='t', seed=1, cpu=dict(workers=2)),
    'plugin-compare': dict(type='compare', title='t', seeds=6, cpu=dict(workers=4), b_brain_class='CautiousBrain', b=dict(spawn_patience=3)),
    'search': dict(type='search', title='t', seeds=3, samples=16, brain_class='EnergyBrain', cpu=dict(workers=4),
                   ranges=[dict(param='regen', min=0.005, max=0.1, scale='log'), dict(param='c_move', min=0.05, max=1.0, scale='log'),
                           dict(param='upkeep_mode'), dict(param='economize')]),
}
outfile = dict(sweep='results.json', grid='grid.json', compare='compare.json', watch='brain.json')
outfile['plugin-compare'] = 'compare.json'
outfile['search'] = 'search.json'
for typ, spec in specs.items():
    d = os.path.join(tmp, typ)
    os.makedirs(d)
    json.dump(spec, open(os.path.join(d, 'spec.json'), 'w'))
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, 'lab_runner.py'), d], capture_output=True, text=True, timeout=600)
    prog = json.load(open(os.path.join(d, 'progress.json')))
    ok = r.returncode == 0 and prog['status'] == 'done' and os.path.exists(os.path.join(d, outfile[typ]))
    check(f'job {typ}', ok, f'{time.time()-t0:.0f}s, {prog["done"]}/{prog["total"]} runs' + ('' if ok else ' ' + (r.stderr or str(prog))[-300:]))
    if ok and typ == 'sweep':
        res = json.load(open(os.path.join(d, 'results.json')))['conditions']['lab']
        check('sweep result has paired z-scores', res['vars']['select']['z']['acc'][1] is not None)
    if ok and typ == 'search':
        sj = json.load(open(os.path.join(d, 'search.json')))
        check('search returns every sample with energy metrics', len(sj['samples']) == 16 and sj['samples'][0]['m']['E'][0] is not None and not sj['partial'])
        regens = [s['params']['regen'] for s in sj['samples']]
        check('search samples stay inside their ranges', all(0.005 <= r <= 0.1 for r in regens) and len(set(round(r, 6) for r in regens)) == 16)
    if ok and typ == 'plugin-compare':
        c = json.load(open(os.path.join(d, 'compare.json')))
        check('plugin compare reports the brain-design difference', any(x['param'] == 'brain design' and x['b'] == 'CautiousBrain' for x in c['config_diff']))
    if ok and typ == 'compare':
        c = json.load(open(os.path.join(d, 'compare.json')))
        check('compare has series + paired diff', len(c['series']['a']['acc']['mean']) >= 8 and c['metrics']['acc']['diff']['z'] is not None)
    if ok and typ == 'watch':
        w = json.load(open(os.path.join(d, 'brain.json')))
        check('watch has snapshots', len(w['snaps']) >= 8 and len(w['snaps'][-1]['pos']) > 0)

# 4. planning and cancel
bad = os.path.join(tmp, 'bad.json')
json.dump(dict(type='sweep', variables=[dict(param='rho', values=[0.5])]), open(bad, 'w'))
r = subprocess.run([sys.executable, os.path.join(HERE, 'lab_runner.py'), '--plan', bad], capture_output=True, text=True)
check('plan rejects 1-value variable', json.loads(r.stdout)['ok'] is False, json.loads(r.stdout).get('error', '')[:50])
d = os.path.join(tmp, 'cancel')
os.makedirs(d)
json.dump(dict(type='sweep', title='c', seeds=60, cpu=dict(workers=2, keep_awake=False), variables=[dict(param='sigma', values=[0.2, 0.25, 0.3])]), open(os.path.join(d, 'spec.json'), 'w'))
open(os.path.join(d, 'cancel.flag'), 'w').close()
t0 = time.time()
subprocess.run([sys.executable, os.path.join(HERE, 'lab_runner.py'), d], capture_output=True, text=True, timeout=300)
prog = json.load(open(os.path.join(d, 'progress.json')))
check('cancel flag stops a job quickly', prog['status'] == 'cancelled' and time.time() - t0 < 60, f'{time.time()-t0:.0f}s')

print('\nALL PASSED' if not fails else f'\nFAILED: {fails}')
sys.exit(1 if fails else 0)
