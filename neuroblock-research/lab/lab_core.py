"""Lab engine: validated experiment specs -> simulation runs.

A job spec is DATA only (numbers, booleans, names). The web UI can never send code. New brain designs or tasks are
added as plugin files in lab/plugins/ (code written by you or Claude, on disk), see plugins/README.md.
"""
import glob
import importlib.util
import json
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
import neuroblock
from neuroblock import summarize
from brain2 import Brain2

# ---------------------------------------------------------------- parameter catalog
PARAMS = {}


def add(name, kind, typ, default, group, owner=None, **kw):
    PARAMS[name] = dict(name=name, kind=kind, type=typ, default=default, group=group, owner=owner, **kw)


add('rho', 'brain', 'float', 0.5, 'Geometry', min=0.02, max=5, step=0.05)
add('min_sep', 'brain', 'float', 0.3, 'Geometry', min=0, max=3, step=0.05)
add('spawn_len', 'brain', 'float', 1.0, 'Geometry', min=0.05, max=10, step=0.05)
add('us_per_unit', 'us', 'float', 40.0, 'Geometry', min=0.1, max=10000, step=1)
add('move', 'brain', 'bool', True, 'Condensing')
add('merge', 'brain', 'bool', True, 'Condensing')
add('prune', 'brain', 'bool', True, 'Condensing')
add('eta', 'brain', 'float', 0.5, 'Condensing', min=0.01, max=1, step=0.05)
add('select', 'brain', 'choice', 'fast', 'Selection', choices=['fast', 'best'])
add('lr', 'brain', 'float', 0.1, 'Hardening', min=0.001, max=1, step=0.01)
add('rigid', 'brain', 'bool', False, 'Hardening')
add('c0', 'brain', 'float', 30.0, 'Hardening', min=0.1, max=1e9, step=1)
add('a0', 'brain', 'float', 5.0, 'Hardening', min=0.1, max=1e6, step=1)
add('spawn_cost', 'brain', 'float', 50.0, 'Timing', min=0, max=1e5, step=5)
add('win', 'task', 'int', 200, 'Timing', min=10, max=2000, step=10)
add('T', 'task', 'int', 6000, 'Timing', min=400, max=200000, step=200)
add('expand', 'brain', 'bool', False, 'Expansion')
add('support', 'brain', 'int', 8, 'Expansion', min=1, max=200, step=1)
add('frac', 'brain', 'float', 0.5, 'Expansion', min=0.01, max=5, step=0.05)
add('gain_frac', 'brain', 'float', 0.15, 'Expansion', min=0, max=10, step=0.05)
add('buf', 'brain', 'int', 400, 'Expansion', min=50, max=5000, step=50)
add('sigma', 'task', 'float', 0.10, 'Environment', min=0, max=2, step=0.01)
add('K', 'task', 'int', 8, 'Environment', min=2, max=64, step=1)
add('D', 'task', 'int', 8, 'Environment', min=1, max=128, step=1)
add('drift', 'task', 'choice', 'shift', 'Environment', choices=['none', 'replace', 'shift'])
add('zipf', 'task', 'float', 1.0, 'Environment', min=0, max=4, step=0.1)
add('shift_sd', 'task', 'float', 0.12, 'Environment', min=0, max=1, step=0.01)
add('n_replace', 'task', 'int', 3, 'Environment', min=1, max=32, step=1)

PRESETS = {
    'Medium task, all features on': {},
    'Hard task, all features on': dict(rho=0.7, sigma=0.15, K=10),
    'Bare brain (no condensing)': dict(_brain='Brain2', move=False, merge=False, prune=False),
    'Medium task, replace drift': dict(drift='replace'),
    'Everything on (hardening + expansion)': dict(_brain='Brain2', rigid=True, expand=True),
    # energy economies (the special key _brain also switches the brain design)
    'Energy: standard economy': dict(_brain='EnergyBrain'),
    'Energy: endowment only, no income (it will starve)': dict(_brain='EnergyBrain', regen=0.0, e_seed=3.0),
    'Energy: growth feeds it, stagnation starves it': dict(_brain='EnergyBrain', regen=0.0, growth_yield=2.0, upkeep=0.0008, upkeep_mode='all'),
    'Energy: everything has a passive cost': dict(_brain='EnergyBrain', regen=0.08, upkeep=0.0005, upkeep_mode='all'),
    'Energy: only active neurons cost': dict(_brain='EnergyBrain', regen=0.045, upkeep=0.004, upkeep_mode='active'),
    'Energy: it eats when it is right': dict(_brain='EnergyBrain', regen=0.0, reward_yield=0.09),
    'Energy: overload and fatigue shutdowns': dict(_brain='EnergyBrain', overload_limit=0.06, fatigue_gain=0.8, fatigue_windows=6),
    'Energy: cost-blind movement (no natural economy)': dict(_brain='EnergyBrain', economize=False),
}
CORE_PARAMS = set(PARAMS)
# Settings the 2026-10-18 sweep showed to be unnecessary or settled (see CORE_SETTINGS.md). A "lean" brain design has them fixed at
# their best values inside the code, so they are not passed to it and the Lab hides them. Brain2 keeps them for reproducibility.
LEGACY = {'min_sep', 'spawn_len', 'us_per_unit', 'move', 'merge', 'prune', 'eta', 'select', 'lr', 'rigid', 'c0', 'a0', 'spawn_cost',
          'expand', 'support', 'frac', 'gain_frac', 'buf'}
LEAN_BRAINS = set()


# ---------------------------------------------------------------- task environments and brain classes
class GaussianEnv:
    """The original task: K Gaussian clusters in D dimensions with Zipf-skewed class frequencies."""

    def __init__(self, rng, K, D, sigma, zipf=1.0, shift_sd=0.12, n_replace=3, **_):
        self.rng, self.K, self.D, self.sigma = rng, K, D, sigma
        self.shift_sd, self.n_replace = shift_sd, n_replace
        self.centers = rng.uniform(0, 1, (K, D))
        p = 1 / np.arange(1, K + 1) ** zipf
        self.probs = p / p.sum()

    def sample(self, n):
        y = self.rng.choice(self.K, n, p=self.probs)
        return self.centers[y] + self.rng.normal(0, self.sigma, (n, self.D)), y

    def drift(self, mode):
        if mode == 'replace':
            sel = [i for i in range(1, 1 + self.n_replace) if i < self.K]
            self.centers[sel] = self.rng.uniform(0, 1, (len(sel), self.D))
        else:
            self.centers += self.rng.normal(0, self.shift_sd, self.centers.shape)


ENERGY_KEYS = ['uptime', 'shutdowns', 'E', 'Eleft', 'e_move', 'e_think', 'e_spawn', 'e_upkeep', 'e_other', 'mastery', 'sleep_share', 'sleeps', 'tired_sleeps']
BRAINS = {'Brain2': Brain2}
ENVS = {'gaussian': GaussianEnv}
PLUGINS, PLUGIN_ERRORS = [], []


def load_plugins():
    """Import lab/plugins/*.py. A plugin defines PLUGIN = dict(name, about, brains={}, envs={}, params=[dict(...)])."""
    pdir = os.path.join(HERE, 'plugins')
    if pdir not in sys.path:
        sys.path.insert(0, pdir)  # plugins may import each other (e.g. `from lean_brain import LeanBrain`)
    for path in sorted(glob.glob(os.path.join(pdir, '*.py'))):
        if os.path.basename(path).startswith('_'):
            continue
        try:
            name = os.path.basename(path)[:-3]
            mod = sys.modules.get(name)
            if mod is None:
                spec = importlib.util.spec_from_file_location(name, path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
            pl = mod.PLUGIN
            for n, cls in pl.get('brains', {}).items():
                BRAINS[n] = cls
            for n, cls in pl.get('envs', {}).items():
                ENVS[n] = cls
            LEAN_BRAINS.update(pl.get('lean_brains', []))
            for p in pl.get('params', []):
                add(p['name'], p.get('kind', 'brain'), p.get('type', 'float'), p['default'], p.get('group', pl['name']),
                    owner=p.get('owner'), **{k: v for k, v in p.items() if k in ('min', 'max', 'step', 'choices', 'about')})
            PLUGINS.append(dict(name=pl['name'], about=pl.get('about', ''), brains=list(pl.get('brains', {})),
                                envs=list(pl.get('envs', {})), file=os.path.basename(path)))
        except Exception as e:  # a broken plugin must never break the lab
            PLUGIN_ERRORS.append(dict(file=os.path.basename(path), error=f'{type(e).__name__}: {e}'))


load_plugins()


PRIORITY_RANK = {'low': 0, 'normal': 1, 'high': 2}


def limits():
    """the user's compute limits (lab/limits.json); the default is the gentlest setting"""
    lim = {'max_workers': 1, 'priority_cap': 'low'}
    try:
        lim.update({k: v for k, v in json.load(open(os.path.join(HERE, 'limits.json'), encoding='utf-8')).items() if k in lim})
    except Exception:
        pass
    lim['max_workers'] = max(1, int(lim['max_workers']))
    return lim


def catalog():
    default = 'EnergyBrain' if 'EnergyBrain' in BRAINS else ('LeanBrain' if 'LeanBrain' in BRAINS else 'Brain2')
    return dict(params=list(PARAMS.values()), presets=PRESETS, brains=list(BRAINS), envs=list(ENVS), plugins=PLUGINS,
                plugin_errors=PLUGIN_ERRORS, cores=os.cpu_count() or 1, lean_brains=sorted(LEAN_BRAINS), legacy=sorted(LEGACY),
                default_brain=default, limits=limits())


# ---------------------------------------------------------------- validation
def coerce(name, v):
    p = PARAMS.get(name)
    if p is None:
        raise ValueError(f'Unknown setting "{name}".')
    t = p['type']
    if t == 'bool':
        if isinstance(v, str):
            v = v.strip().lower() in ('1', 'true', 'on', 'yes')
        return bool(v)
    if t == 'choice':
        if v not in p['choices']:
            raise ValueError(f'"{name}" must be one of {p["choices"]}, not {v!r}.')
        return v
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f'"{name}" needs a number, not {v!r}.')
    if not math.isfinite(x):
        raise ValueError(f'"{name}" must be a finite number.')
    lo, hi = p.get('min'), p.get('max')
    if lo is not None and x < lo or hi is not None and x > hi:
        raise ValueError(f'"{name}" must be between {lo:g} and {hi:g} (got {x:g}).')
    return int(round(x)) if t == 'int' else x


def full_config(overrides, base=None):
    """defaults + base + overrides, every value validated"""
    cfg = {n: p['default'] for n, p in PARAMS.items()}
    for src in (base or {}, overrides or {}):
        for k, v in src.items():
            if k.startswith('_'):
                continue  # preset directives such as _brain are handled by the page
            cfg[k] = coerce(k, v)
    return cfg


def owned(owner, name):
    """a setting's owner is None (everyone), one brain/task name, or a list of names"""
    return owner is None or owner == name or (isinstance(owner, (list, tuple)) and name in owner)


def split_config(cfg, brain_class='Brain2', env='gaussian'):
    """flat validated config -> (brain kwargs, task kwargs, us)"""
    b, t, u = {}, {}, cfg['us_per_unit']
    for n, p in PARAMS.items():
        if p['kind'] == 'brain' and owned(p['owner'], brain_class):
            if brain_class in LEAN_BRAINS and n in LEGACY:
                continue  # fixed inside the lean design
            b[n] = cfg[n]
        elif p['kind'] == 'task' and owned(p['owner'], env):
            t[n] = cfg[n]
    t['drift_mode'] = 'shift' if t['drift'] == 'none' else t['drift']
    t['drift'] = cfg['drift'] != 'none'
    W = t['T'] // t['win']
    if W < 8:
        raise ValueError(f'Run length T ({t["T"]}) divided by window win ({t["win"]}) gives only {W} windows; need at least 8.')
    return b, t, u


# ---------------------------------------------------------------- one simulated lifetime
def simulate(brain_kw, task, us, seed, brain_class='Brain2', env='gaussian', record=None):
    """record: None (metrics only), 'series' (per-window curves) or 'snap' (series + brain snapshots)."""
    neuroblock.US = float(us)
    rng = np.random.default_rng(seed)
    K, D, T, win = task['K'], task['D'], task['T'], task['win']
    env_args = {k: v for k, v in task.items() if k not in ('T', 'win', 'drift', 'drift_mode')}
    world = ENVS[env](rng, **env_args)
    b = BRAINS[brain_class](D, K, rng=np.random.default_rng(seed + 1000), **brain_kw)
    W = T // win
    dw = W // 2
    rows, snaps, diags = [], [], []
    for w in range(W):
        if task['drift'] and w == dw:
            world.drift(task['drift_mode'])
        X, Y = world.sample(win)
        ts, sp = [], 0
        for x, y in zip(X, Y):
            t, _, spawned = b.step(x, y)
            ts.append(t)
            sp += spawned
            if hasattr(b, 'remember'):
                b.remember(x, y)
        n_before = b.n
        b.condense()
        Xt, Yt = world.sample(400)
        tq, acc, cov = b.query(Xt, Yt)
        edge = float(np.mean([np.linalg.norm(b.pos[i] - b.pos[b.par[i]]) for i in range(1, len(b.pos))])) if b.n else 0.0
        rows.append(dict(w=w, t_train=float(np.mean(ts)), spawns=sp, n_before=n_before, n=b.n, t_query=tq, acc=acc, cov=cov,
                         exp=getattr(b, 'exp_total', 0), mean_edge=edge,
                         depth_corr=neuroblock.spearman(b.cnt[1:], b.d[1:]) if b.n > 2 else np.nan))
        if record and hasattr(b, 'diag'):
            diags.append(b.diag())  # the brain's own dials (radius, how often it thinks twice, ...)
        if record == 'snap':
            keep = np.argsort(-b.cnt)[:600] if len(b.cnt) > 600 else np.arange(len(b.cnt))
            snaps.append(dict(w=w, idx=keep.tolist(), pos=np.round(b.pos[keep], 3).tolist(), par=b.par[keep].tolist(),
                              cnt=np.round(b.cnt[keep], 1).tolist(), ema=np.round(b.ema[keep], 2).tolist(),
                              d=np.round(b.d[keep], 1).tolist(), n=b.n))
    o = summarize(rows, dw, K=K)
    rec = o['recovery_windows']
    fin = math.isfinite(rec)
    metrics = dict(acc=o['post_acc'], t=o['post_t_query'], n=o['post_n'], F=o['post_F'], edge=o['post_mean_edge'],
                   rec=float(rec) if fin else float('nan'), rec_never=0.0 if fin else 1.0, pre_acc=o['pre_acc'],
                   pre_t=o['pre_t_query'], pre_n=o['pre_n'], cov=o['post_cov'], spawns=o['post_spawns'],
                   exp=float(rows[-1]['exp']))
    rep = b.energy_report() if hasattr(b, 'energy_report') else {}
    for k in ENERGY_KEYS:
        metrics[k] = float(rep.get(k, float('nan')))
    metrics['eff'] = metrics['acc'] / max(metrics['E'], 1e-3) if math.isfinite(metrics['E']) else float('nan')
    out = dict(metrics=metrics)
    if record:
        out['series'] = dict(w=list(range(W)), drift_w=dw if task['drift'] else None,
                             **{k: [float(r[k]) for r in rows] for k in ('acc', 'cov', 'n', 'spawns', 'exp', 'mean_edge')},
                             t=[float(r['t_query']) for r in rows])
    if diags:
        out['series']['diag'] = {k: [float(d[k]) for d in diags] for k in diags[0]}
    if record == 'snap':
        out['snaps'] = snaps
    return out


def set_priority(level):
    """level: 'low' | 'normal' | 'high' (above normal). Never real-time."""
    if os.name != 'nt':
        return False
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.SetPriorityClass.restype = wintypes.BOOL
    cls = {'low': 0x4000, 'normal': 0x20, 'high': 0x8000}.get(level, 0x20)
    return bool(k.SetPriorityClass(k.GetCurrentProcess(), cls))


def worker_init(priority):
    set_priority(priority)


def worker(task):
    key, bkw, tsk, us, seed, bcls, env, record = task
    t0 = time.time()
    try:
        return key, seed, simulate(bkw, tsk, us, seed, bcls, env, record), time.time() - t0, None
    except Exception as e:
        return key, seed, None, time.time() - t0, f'{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}'
