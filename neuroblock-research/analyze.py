"""Turn sweeps/results.json into graphs (PNG) and markdown tables.   Usage: python analyze.py [results.json]
Graphs go to sweeps/graphs and to the live dashboard folder."""
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'sweeps', 'results.json')
OUTS = [os.path.join(HERE, 'sweeps', 'graphs'), r'~\claude-live\projects\neuroblock\graphs']
R = json.load(open(SRC, encoding='utf-8'))
C = R['conditions']
OAT = [k for k in C if not k.startswith('solo')]
SOLO = [k for k in C if k.startswith('solo')]
SIG = 3.0

BG, PANEL, GRID, TXT, MUT = '#0b0f17', '#121826', '#232c44', '#e6edf7', '#8a97ad'
ACC, VIO, WARN, GOOD, BAD = '#4cc9f0', '#a78bfa', '#ffb454', '#3ddc97', '#ff5d73'
GROUP_COL = {'Geometry': '#4cc9f0', 'Condensing': '#3ddc97', 'Selection': '#fb923c', 'Hardening': '#a78bfa',
             'Timing': '#f472b6', 'Expansion': '#facc15', 'Environment': '#f1f5f9'}
NOISE_COL = '#2b3350'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': PANEL, 'axes.edgecolor': GRID, 'axes.labelcolor': MUT,
                     'text.color': TXT, 'xtick.color': MUT, 'ytick.color': MUT, 'grid.color': GRID,
                     'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold', 'savefig.facecolor': BG})
M = {'acc': ('Accuracy after drift (%)', 100), 't': ('Thought time (µs)', 1), 'n': ('Neuron count', 1),
     'F': ('Fitness (accuracy per effort)', 1), 'sd': ('Seed-to-seed accuracy SD (%)', 100)}
saved = []


def save(fig, name, caption):
    for d in OUTS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name), dpi=125, bbox_inches='tight')
    plt.close(fig)
    saved.append(dict(src=f'projects/neuroblock/graphs/{name}', caption=caption))


def arr(x):
    return np.array([np.nan if v is None else v for v in x], float)


def stats(v, k):
    m = arr(v['m'][k]['mean'])
    ok = np.isfinite(m)
    if ok.sum() < 2:
        return None
    z = [abs(q) for q in v['z'].get(k, []) if q is not None] or [0]
    i0, i1 = np.nanargmin(m), np.nanargmax(m)
    return dict(range=float(m[i1] - m[i0]), lo=float(m[i0]), hi=float(m[i1]), lo_l=v['labels'][i0],
                hi_l=v['labels'][i1], z=float(max(z)))


# ---------------------------------------------------------------- 1. tornado charts, 4 conditions per metric
for k in ('acc', 't', 'n', 'F'):
    fig, axs = plt.subplots(1, len(OAT), figsize=(5.0 * len(OAT), 9.2))
    for ax, ck in zip(np.atleast_1d(axs), OAT):
        rows = [(n, stats(v, k), v['group']) for n, v in C[ck]['vars'].items()]
        rows = [(n, s, g) for n, s, g in rows if s]
        rows.sort(key=lambda r: r[1]['range'])
        y = np.arange(len(rows))
        ax.barh(y, [r[1]['range'] * M[k][1] for r in rows],
                color=[GROUP_COL[r[2]] if r[1]['z'] > SIG else NOISE_COL for r in rows], height=.72)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=8)
        ax.set_title(C[ck]['label'], fontsize=10)
        ax.grid(axis='x', alpha=.5)
        ax.set_xlabel(f'range of mean {M[k][0]}')
    fig.suptitle(f'Which settings move {M[k][0].lower()}?  Bright colour = a real effect; dark bar = no more than random luck',
                 y=1.0, fontsize=12, fontweight='bold')
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=g) for g, c in GROUP_COL.items()] + [Patch(color=NOISE_COL, label='within random luck')],
               loc='lower center', ncol=8, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    save(fig, f'tornado_{k}.png', f'Variable importance for {M[k][0].lower()}, all four task conditions (colour = task group).')

# ---------------------------------------------------------------- 2. small multiples (curves with 95% CI)
def multiples(ck, k, fname, caption):
    items = list(C[ck]['vars'].items())
    ncol = 6
    nrow = math.ceil(len(items) / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.35 * nrow))
    axs = np.atleast_2d(axs)
    for ax in axs.ravel():
        ax.set_visible(False)
    for ax, (name, v) in zip(axs.ravel(), items):
        ax.set_visible(True)
        m, se = arr(v['m'][k]['mean']) * M[k][1], arr(v['m'][k]['se']) * M[k][1]
        x = np.arange(len(m))
        ax.errorbar(x, m, yerr=1.96 * se, color=VIO, ecolor=MUT, capsize=2.5, lw=1.6, marker='none')
        z = v['z'].get(k, [])
        for i in x:
            sig = i < len(z) and z[i] is not None and abs(z[i]) > SIG
            ax.scatter(i, m[i], s=46 if i == v['base_idx'] else 30, facecolor=ACC if sig else PANEL,
                       edgecolor=WARN if i == v['base_idx'] else VIO, linewidth=2 if i == v['base_idx'] else 1.2, zorder=3)
        ref = v.get('ref', {}).get(k, {}).get('mean')
        if ref is not None:
            ax.axhline(ref * M[k][1], color=WARN, ls='--', lw=.9, alpha=.7)
        if k == 'acc':
            ax.set_ylim(top=100.5)
        ax.set_xticks(x)
        ax.set_xticklabels(v['labels'], fontsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.set_title(name, fontsize=9, color=GROUP_COL[v['group']])
        ax.grid(alpha=.4)
    fig.suptitle(f'{C[ck]["label"]}: {M[k][0]} as each variable changes alone. Blue dot = beyond noise, orange ring/dashes = reference',
                 y=1.0, fontsize=11, fontweight='bold')
    fig.tight_layout()
    save(fig, fname, caption)


for ck, tag in (('medium|shift', 'medium'), ('hard|shift', 'hard')):
    if ck in C:
        for k in ('acc', 't', 'n'):
            multiples(ck, k, f'curves_{tag}_{k}.png', f'{C[ck]["label"]}: {M[k][0]} versus every variable, one at a time.')

# ---------------------------------------------------------------- 3. consistency heat map across conditions
names = []
for ck in OAT:
    for n in C[ck]['vars']:
        if n not in names:
            names.append(n)
fig, axs = plt.subplots(1, 3, figsize=(15, 0.36 * len(names) + 2.2), sharey=True)
panels = [('acc', 'Accuracy range (points)', 100, 1), ('t', 'Thought-time range (% of baseline)', None, 1), ('n', 'Neuron-count range (% of baseline)', None, 1)]
for ax, (k, title, sc, _) in zip(axs, panels):
    grid = np.full((len(names), len(OAT)), np.nan)
    sg = np.zeros_like(grid, bool)
    for j, ck in enumerate(OAT):
        base = (C[ck]['baseline'].get(k) or {}).get('mean')
        for i, n in enumerate(names):
            v = C[ck]['vars'].get(n)
            s = stats(v, k) if v else None
            if s:
                grid[i, j] = s['range'] * sc if sc else (100 * s['range'] / base if base else np.nan)
                sg[i, j] = s['z'] > SIG
    im = ax.imshow(grid, aspect='auto', cmap='magma', vmin=0, vmax=np.nanpercentile(grid, 90) if np.isfinite(grid).any() else 1)
    ax.set_xticks(range(len(OAT)))
    ax.set_xticklabels([C[c]['label'].replace(' task · ', '\n').replace(' drift', '') for c in OAT], fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(OAT)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f'{grid[i, j]:.0f}' if grid[i, j] >= 10 else f'{grid[i, j]:.1f}', ha='center', va='center', fontsize=7,
                        color='white' if sg[i, j] else '#7b869c', fontweight='bold' if sg[i, j] else 'normal')
fig.suptitle('Does a variable matter in every condition?  White bold = beyond noise, grey = within noise', fontsize=12, fontweight='bold', y=1.0)
fig.tight_layout()
save(fig, 'consistency_heatmap.png', 'Effect size of every variable in each of the four task conditions (bold white = beyond noise).')

# ---------------------------------------------------------------- 4. speed / accuracy / size trade-off
fig, axs = plt.subplots(1, 2, figsize=(15, 6.2))
for ax, ck in zip(axs, ('medium|shift', 'hard|shift')):
    if ck not in C:
        continue
    pts = []
    for n, v in C[ck]['vars'].items():
        a, t, nn = arr(v['m']['acc']['mean']) * 100, arr(v['m']['t']['mean']), arr(v['m']['n']['mean'])
        for i in range(len(a)):
            if np.isfinite(a[i]) and np.isfinite(t[i]):
                pts.append((t[i], a[i], nn[i], v['group'], n, v['labels'][i]))
    if not pts:
        ax.set_title(C[ck]['label'] + ' (no complete settings yet)')
        continue
    for g, col in GROUP_COL.items():
        sel = [p for p in pts if p[3] == g]
        ax.scatter([p[0] for p in sel], [p[1] for p in sel], s=[max(10, min(p[2], 400) * 1.2) for p in sel], c=col, alpha=.55, label=g, edgecolors='none')
    P = sorted(pts, key=lambda p: p[0])
    front, best = [], -1
    for p in P:
        if p[1] > best:
            front.append(p)
            best = p[1]
    ax.plot([p[0] for p in front], [p[1] for p in front], color=WARN, lw=1.6, label='best accuracy at each speed')
    for p in (front[0], front[-1]):
        ax.annotate(f'{p[4]} = {p[5]}', (p[0], p[1]), fontsize=8, color=TXT, xytext=(6, -12), textcoords='offset points')
    b = C[ck]['baseline']
    if b.get('t') and b.get('acc'):
        ax.scatter([b['t']['mean']], [b['acc']['mean'] * 100], marker='*', s=260, c='white', edgecolors=WARN, linewidths=1.5, label='baseline', zorder=5)
    ax.set_xscale('log')
    ax.set_xlabel('thought time (µs, log scale)  ← faster')
    ax.set_ylabel('accuracy after drift (%)')
    ax.set_title(C[ck]['label'] + '  (bubble size = neuron count)')
    ax.grid(alpha=.4)
axs[0].legend(fontsize=7, ncol=2, loc='lower right')
fig.tight_layout()
save(fig, 'tradeoff_speed_accuracy.png', 'Every tested setting as a point: speed versus accuracy, bubble size = neurons, orange line = best accuracy at each speed.')

# ---------------------------------------------------------------- 5. each feature alone on a bare brain
if SOLO:
    fig, axs = plt.subplots(len(SOLO), 3, figsize=(14, 3.6 * len(SOLO)))
    axs = np.atleast_2d(axs)
    for r, ck in enumerate(SOLO):
        for c, k in enumerate(('acc', 't', 'n')):
            ax = axs[r, c]
            labels, d, e, cols = [], [], [], []
            for n, v in C[ck]['vars'].items():
                m, se = arr(v['m'][k]['mean']), arr(v['m'][k]['se'])
                b = v['base_idx'] if v['base_idx'] >= 0 else 0
                for i in range(len(m)):
                    if i != b and np.isfinite(m[i]) and np.isfinite(m[b]):
                        labels.append(f'{n}: {v["labels"][i]}')
                        d.append((m[i] - m[b]) * M[k][1])
                        e.append(1.96 * math.hypot(se[i], se[b]) * M[k][1])
                        z = v['z'].get(k, [None] * len(m))[i]
                        cols.append((GOOD if (d[-1] > 0) == (k == 'acc') else BAD) if (z is not None and abs(z) > SIG) else '#3a4460')
            y = np.arange(len(labels))
            ax.barh(y, d, xerr=e, color=cols, ecolor=MUT, capsize=2, height=.7)
            ax.axvline(0, color=MUT, lw=.8)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_title(f'{C[ck]["label"]}\nΔ {M[k][0]}', fontsize=9)
            ax.grid(axis='x', alpha=.4)
    fig.suptitle('Each feature added ALONE to a bare brain (green = better, red = worse, grey = within noise; bars = 95% interval)', fontsize=12, fontweight='bold', y=1.0)
    fig.tight_layout()
    save(fig, 'solo_features.png', 'What each feature does when it is the only thing switched on (versus a bare brain).')

# ---------------------------------------------------------------- tables
lines = ['# Sweep tables (auto-generated)', '', f'Source: `{os.path.basename(SRC)}`; seeds: {R["meta"]["seeds"]}; |z| > {SIG:g} counts as beyond noise (paired by seed).', '']
for ck in OAT + SOLO:
    lines += [f'## {C[ck]["label"]} (`{ck}`)', '', '| variable | accuracy % (lo → hi) | thought µs (lo → hi) | neurons (lo → hi) | max abs z (acc / time / neurons / fitness) |', '|---|---|---|---|---|']
    for n, v in C[ck]['vars'].items():
        s = {k: stats(v, k) for k in ('acc', 't', 'n', 'F')}
        if not all(s.values()):
            continue
        f = lambda k, sc=1, d=1: f'{s[k]["lo"]*sc:.{d}f} ({s[k]["lo_l"]}) → {s[k]["hi"]*sc:.{d}f} ({s[k]["hi_l"]})'
        lines.append(f'| {n} | {f("acc", 100)} | {f("t", 1, 0)} | {f("n", 1, 1)} | {s["acc"]["z"]:.1f} / {s["t"]["z"]:.1f} / {s["n"]["z"]:.1f} / {s["F"]["z"]:.1f} |')
    lines.append('')
open(os.path.join(HERE, 'sweeps', 'analysis_tables.md'), 'w', encoding='utf-8').write('\n'.join(lines))
json.dump(saved, open(os.path.join(HERE, 'sweeps', 'graphs_index.json'), 'w'), indent=1)
print(f'{len(saved)} graphs, tables written')
