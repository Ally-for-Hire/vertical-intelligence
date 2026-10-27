"""Summarize an energy search from the Lab.  Usage: python energy_report.py <job_id> [accuracy_floor] [uptime_floor]
Prints the best trade-offs, which settings went with efficiency, and how the income styles compare."""
import json
import math
import sys

import numpy as np

JOBS = r'~\claude-live\lab\jobs'
jid = sys.argv[1]
floor = float(sys.argv[2]) if len(sys.argv) > 2 else None
upf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
d = json.load(open(f'{JOBS}\\{jid}\\search.json', encoding='utf-8'))
floor = floor if floor is not None else d['acc_floor']
S = d['samples']
m = lambda x, k: x['m'][k][0] if x['m'].get(k) and x['m'][k][0] is not None else float('nan')
good = [x for x in S if m(x, 'acc') >= floor and m(x, 'uptime') >= upf]
print(f'{d["title"]}\n{len(S)} of {d["total"]} combinations; accuracy >= {floor:.0%} and uptime >= {upf:.0%}: {len(good)} ({100*len(good)/max(1,len(S)):.0f}%)')
if d.get('base') and d['base'].get('E'):
    print(f'starting setup: {m({"m": d["base"]}, "acc"):.3f} acc, uptime {m({"m": d["base"]}, "uptime"):.2f}, {m({"m": d["base"]}, "E"):.1f} E per 1000')
allE = np.array([m(x, 'E') for x in S])
print(f'energy per 1000 experiences across all combinations: min {np.nanmin(allE):.1f}, median {np.nanmedian(allE):.1f}, max {np.nanmax(allE):.1f}')
print('\nTOP 8 cheapest good combinations:')
for x in sorted(good, key=lambda x: m(x, 'E'))[:8]:
    p = x['params']
    print(f'  E {m(x,"E"):5.1f} | acc {m(x,"acc"):.3f} up {m(x,"uptime"):.2f} t {m(x,"t"):4.0f}us n {m(x,"n"):4.1f} | ' + ', '.join(f'{k} {v:.3g}' if isinstance(v, float) else f'{k} {v}' for k, v in p.items()))

# which settings went with efficiency (rank correlation with accuracy x uptime / energy; categories by group means)
score = np.array([m(x, 'eff') * m(x, 'uptime') for x in S])
ok = np.isfinite(score)
rows = []
for r in d['ranges']:
    vals = [x['params'][r['param']] for x in S]
    if r['type'] in ('float', 'int'):
        a = np.argsort(np.argsort(np.array(vals, float)[ok])); b = np.argsort(np.argsort(score[ok]))
        rho = float(np.corrcoef(a, b)[0, 1])
        rows.append((abs(rho), r['param'], f'{"higher" if rho > 0 else "lower"} is more efficient (rank correlation {rho:+.2f})'))
    else:
        g = {}
        for v, sc, o in zip(vals, score, ok):
            if o: g.setdefault(str(v), []).append(sc)
        means = {k: np.mean(v) for k, v in g.items()}
        spread = (max(means.values()) - min(means.values())) / (np.mean(list(means.values())) or 1)
        rows.append((min(1, spread / 2), r['param'], 'best choice: ' + max(means, key=means.get) + '  (' + ', '.join(f'{k}: {v:.3f}' for k, v in means.items()) + ')'))
print('\nWHICH SETTINGS WENT WITH EFFICIENCY (accuracy x uptime per energy):')
for v, name, txt in sorted(rows, reverse=True):
    print(f'  {v:.2f}  {name:16s} {txt}')

# income style: which source of income did the good brains rely on?
def style(p):
    parts = []
    if p.get('regen', 0) > 0.02: parts.append('regen')
    if p.get('growth_yield', 0) > 1.0: parts.append('growth')
    if p.get('reward_yield', 0) > 0.05: parts.append('reward')
    return '+'.join(parts) or 'little income'
buckets = {}
for x in S:
    buckets.setdefault(style(x['params']), []).append(x)
print('\nINCOME STYLE (regen > 0.02, growth yield > 1, reward yield > 0.05 count as "relied on"):')
for k, v in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
    g = [x for x in v if x in good]
    print(f'  {k:22s} n={len(v):4d}  good {100*len(g)/len(v):3.0f}%  mean uptime {np.nanmean([m(x,"uptime") for x in v]):.2f}  mean energy of the good ones {np.nanmean([m(x,"E") for x in g]) if g else float("nan"):.1f}')
for key in ('economize', 'upkeep_mode'):
    if any(key in x['params'] for x in S):
        print(f'\n{key.upper()}:')
        for val in sorted({str(x['params'][key]) for x in S}):
            v = [x for x in S if str(x['params'][key]) == val]
            g = [x for x in v if x in good]
            print(f'  {val:8s} n={len(v):4d} good {100*len(g)/len(v):3.0f}%  mean uptime {np.nanmean([m(x,"uptime") for x in v]):.2f}  mean energy (all) {np.nanmean([m(x,"E") for x in v]):6.1f}')
