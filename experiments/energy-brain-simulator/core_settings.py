"""Which settings actually affected the outcome?  Reads sweeps/results.json, writes CORE_SETTINGS.md + sweeps/core_settings.json.

A setting "mattered" in a condition when its effect is both real (paired z > 3) AND big enough to care about:
    accuracy moves >= 2 points, OR thought time moves >= 15% of baseline, OR neuron count moves >= 25% of baseline.
A setting that mattered can still be harmful or one-sided: the verdict also says whether any value of it BEAT the starting value.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'sweeps', 'results.json'), encoding='utf-8'))['conditions']
OAT = [k for k in R if not k.startswith('solo')]
Z = 3.0
ACC_PTS, T_FRAC, N_FRAC = 0.02, 0.15, 0.25
WORLD = {'sigma', 'K (classes)', 'D (inputs)', 'zipf (class skew)', 'drift', 'shift_sd', 'n_replace', 'T'}
UNIT = {'us_per_unit'}


def vals(e, k):
    return [x for x in e['m'][k]['mean']]


def judge(name, c):
    e = R[c]['vars'].get(name)
    if not e:
        return None
    b = R[c]['baseline']
    out = dict(cond=c, matters=False, why=[], better=False, best={})
    for k, thr, rel in (('acc', ACC_PTS, False), ('t', T_FRAC, True), ('n', N_FRAC, True)):
        m = vals(e, k)
        ok = [(x, i) for i, x in enumerate(m) if x is not None]
        if len(ok) < 2:
            continue
        lo, hi = min(ok), max(ok)
        rng = hi[0] - lo[0]
        base = e['ref'].get(k, {}).get('mean') or b[k]['mean']
        size = rng / base if rel else rng
        zmax = max([abs(z) for z in e['z'][k] if z is not None] or [0])
        if zmax > Z and size >= thr:
            out['matters'] = True
            out['why'].append(f'{k}: {lo[0]:.3g} to {hi[0]:.3g} (z {zmax:.0f})')
        # does any value beat the reference on accuracy (up) or time/neurons (down) with a real difference?
        for x, i in ok:
            z = e['z'][k][i]
            if z is not None and abs(z) > Z and ((k == 'acc' and x > base and x - base >= thr) or (k != 'acc' and x < base and (base - x) / base >= thr)):
                out['better'] = True
                out['best'][k] = e['labels'][i]
    return out


rows = []
for name in R[OAT[0]]['vars']:
    per = [judge(name, c) for c in OAT]
    per = [p for p in per if p]
    # include the other drift-specific settings that only exist in some conditions
    n_mat = sum(p['matters'] for p in per)
    better = any(p['better'] for p in per)
    why = next((p['why'] for p in per if p['why']), [])
    rows.append(dict(name=name, conds_mattered=n_mat, of=len(per), ever_better=better, why=why, best=[p['best'] for p in per if p['best']]))
for name in ('shift_sd', 'n_replace'):
    if not any(r['name'] == name for r in rows):
        per = [judge(name, c) for c in OAT]
        per = [p for p in per if p]
        rows.append(dict(name=name, conds_mattered=sum(p['matters'] for p in per), of=len(per), ever_better=any(p['better'] for p in per),
                         why=next((p['why'] for p in per if p['why']), []), best=[]))

# decision rules (documented in CORE_SETTINGS.md)
SUB = {'c0 (rigid on)': 'rigid', 'a0 (rigid on)': 'rigid', 'support (expand on)': 'expand', 'frac (expand on)': 'expand',
       'gain_frac (expand on)': 'expand', 'buf (expand on)': 'expand'}
solo_gain = {}
for c in R:
    if c.startswith('solo'):
        for name, e in R[c]['vars'].items():
            solo_gain.setdefault(name, []).append(e)
START_LABEL = {}
for r in rows:
    e = R[OAT[0]]['vars'][r['name']] if r['name'] in R[OAT[0]]['vars'] else None
    START_LABEL[r['name']] = e['labels'][e['base_idx']] if e and e['base_idx'] >= 0 else None
# judgement calls that numbers alone cannot make (reasons are printed in CORE_SETTINGS.md)
JUDGEMENT = {
    'min_sep': ('constant', 'The physical size of a neuron, not a design choice. Its best value is "as small as possible", so a brain allowed to tune it would '
                            'just drive it to zero for free speed. Keep it fixed at 0.3 (this is the speed limit assumption).'),
}
for r in rows:
    n = r['name']
    if n in WORLD:
        r['group'], r['decision'] = 'world', 'Keep as a test-world control (it is the task, not the brain).'
    elif n in UNIT:
        r['group'], r['decision'] = 'constant', 'Fix at 40 (it only converts distance to microseconds; scaling was exactly linear).'
    elif n in JUDGEMENT:
        r['group'], r['decision'] = JUDGEMENT[n]
    elif r['conds_mattered'] == 0:
        r['group'], r['decision'] = 'dropped', 'Drop: no real effect in any task.'
    elif not r['ever_better']:
        if START_LABEL.get(n) == 'on':
            r['group'], r['decision'] = 'builtin', 'Matters, and the start value (on) is already the best in every task: make it a built-in behaviour that is always on.'
        elif START_LABEL.get(n) == 'off':
            r['group'], r['decision'] = 'dropped', 'Drop: switching it on only ever made things worse or no better; keep it permanently off.'
        else:
            r['group'], r['decision'] = 'builtin', f'Matters, but the start value ({START_LABEL.get(n)}) is already the best in every task: fix it as a constant.'
    else:
        r['group'], r['decision'] = 'candidate', 'Matters and some value beat the start: next question is whether the best value DEPENDS ON THE WORLD (if so, the brain should adjust it itself).'
for r in rows:
    if r['name'] in SUB:
        parent = next(x for x in rows if x['name'] == SUB[r['name']])
        if parent['group'] == 'dropped':
            r['group'], r['decision'] = 'dropped', f'Drop with its parent switch "{SUB[r["name"]]}", which never helped.'

order = {'candidate': 0, 'builtin': 1, 'constant': 2, 'world': 3, 'dropped': 4}
rows.sort(key=lambda r: (order[r['group']], -r['conds_mattered'], r['name']))
json.dump(dict(rules=dict(z=Z, acc_points=ACC_PTS, time_fraction=T_FRAC, neuron_fraction=N_FRAC), rows=rows), open(os.path.join(HERE, 'sweeps', 'core_settings.json'), 'w'), indent=1)
lines = ['# Which settings mattered (auto-generated by core_settings.py)', '',
         f'A setting counts as having **mattered** in a task when its effect is real (paired z > {Z:g}) and big enough to care about: accuracy moves '
         f'{ACC_PTS*100:g}+ points, or thought time moves {T_FRAC*100:g}%+ of baseline, or neuron count moves {N_FRAC*100:g}%+ of baseline. '
         'Four task conditions (medium/hard x shift/replace), 24 repeats each.', '',
         '| Setting | Mattered in | What moved | Beat the start value? | Decision |', '|---|---|---|---|---|']
for r in rows:
    lines.append(f'| {r["name"]} | {r["conds_mattered"]}/{r["of"]} tasks | {"; ".join(r["why"][:2]) or "-"} | {"yes" if r["ever_better"] else "no"} | **{r["group"]}**: {r["decision"]} |')
open(os.path.join(HERE, 'CORE_SETTINGS.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
for g in order:
    print(f'{g:9s}:', ', '.join(r['name'] for r in rows if r['group'] == g))
