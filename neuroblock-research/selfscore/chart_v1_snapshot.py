"""Phone-friendly progress chart of the self-scored grid: reads the run logs (running AND finished jobs), averages over seeds, saves a portrait PNG.
python selfscore/chart.py [--stage c1] [--out selfscore/runs/live.png]
"""
import os, re, glob, time, argparse, collections
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, 'runs')
LINE = re.compile(r'g\s*(\d+) test acc mean ([\d.]+) max ([\d.]+).*?\| C ([\d.]+)')
NAME = re.compile(r'(c\d+)_(\w+?)_(none|grow_and_pred|grow|pred|nonreg)_(nocull|ext|lin|step|Q|C)_s(\d)')
COL = {'none': '#8b949e', 'grow': '#2f81f7', 'grow_and_pred': '#f0883e'}
LS = {'nocull': ':', 'ext': '-', 'lin': '--', 'step': '-.', 'Q': (0, (1, 1)), 'C': (0, (3, 1, 1, 1))}
ap = argparse.ArgumentParser(); ap.add_argument('--stage', default='c1'); ap.add_argument('--out', default=os.path.join(RUNS, 'live.png')); a = ap.parse_args()
data = collections.defaultdict(lambda: collections.defaultdict(list)); nfin = nrun = 0
for lg in glob.glob(os.path.join(RUNS, f'{a.stage}_*.log')):
    m = NAME.match(os.path.basename(lg)[:-4])
    if not m: continue
    pts = [(int(x.group(1)), float(x.group(3)), float(x.group(4))) for x in map(LINE.search, open(lg, errors='ignore').read().splitlines()) if x]
    if not pts: continue
    if os.path.exists(lg[:-4] + '.json'): nfin += 1
    else: nrun += 1
    data[(m.group(2), m.group(3), m.group(4))][m.group(5)] = pts
probs = sorted({k[0] for k in data}) or ['parity4']
fig, axes = plt.subplots(len(probs), 2, figsize=(7.2, 3.1 * len(probs) + 1.2), squeeze=False)
fig.patch.set_facecolor('#0d1117')
for r, p in enumerate(probs):
    for c, (ttl, idx, logy) in enumerate((('best accuracy', 1, False), ('mean complexity C', 2, True))):
        ax = axes[r][c]; ax.set_facecolor('#161b22'); ax.tick_params(colors='#8b949e', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#30363d')
        for (pp, inner, cull), seeds in data.items():
            if pp != p: continue
            L = min(len(v) for v in seeds.values()); g = [x[0] for x in list(seeds.values())[0][:L]]
            y = np.mean([[x[idx] for x in v[:L]] for v in seeds.values()], 0)
            ax.plot(g, y, color=COL[inner], ls=LS[cull], lw=1.2, alpha=0.9)
        ax.set_title(f'{p}: {ttl}', color='#e6edf3', fontsize=9)
        if logy: ax.set_yscale('log')
        else: ax.set_ylim(0.45, 1.02)
        ax.grid(color='#21262d', lw=0.5)
from matplotlib.lines import Line2D
h = [Line2D([0], [0], color=COL[k], lw=2, label=f'inner: {k}') for k in COL] + [Line2D([0], [0], color='#c9d1d9', ls=LS[k], lw=1.4, label=f'cull: {k}') for k in LS]
fig.legend(handles=h, loc='lower center', ncol=3, fontsize=7, frameon=False, labelcolor='#c9d1d9')
fig.suptitle(f'self-scored grid {a.stage} · {nfin} done, {nrun} running · {time.strftime("%H:%M")}', color='#e6edf3', fontsize=10)
fig.tight_layout(rect=(0, 0.08, 1, 0.97)); fig.savefig(a.out, dpi=110, facecolor=fig.get_facecolor()); print(a.out, nfin, nrun)
