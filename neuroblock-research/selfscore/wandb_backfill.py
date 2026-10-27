"""Upload already-finished runs (saved JSON histories) to Weights & Biases so they appear next to the live ones. Marks uploaded runs with a .wb file.
python selfscore/wandb_backfill.py [--stage c1]"""
import os, re, json, glob, argparse
import wandb
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, 'runs')
NAME = re.compile(r'(c\d+)_(\w+?)_(none|grow_and_pred|grow|pred|nonreg)_(nocull|ext|lin|step|Q|C)_s(\d)')
ap = argparse.ArgumentParser(); ap.add_argument('--stage', default='c1'); a = ap.parse_args()
for f in sorted(glob.glob(os.path.join(RUNS, f'{a.stage}_*.json'))):
    base = f[:-5]
    if os.path.exists(base + '.wb'): continue
    nm = os.path.basename(base); m = NAME.match(nm)
    if not m: continue
    hist = json.load(open(f))
    r = wandb.init(project='neuroblock-selfscore', name=nm, group=f'{m.group(1)}_{m.group(2)}', tags=[m.group(2), m.group(3), 'backfilled'],
                   config=dict(problem=m.group(2), inner=m.group(3), cull_cfg=m.group(4), seed=int(m.group(5))), reinit=True, settings=wandb.Settings(console='off'))
    for rec in hist: r.log({k: v for k, v in rec.items() if k not in ('g', 'sec')} | {'minutes': rec['sec'] / 60}, step=rec['g'])
    r.summary.update(dict(final_acc_mean=hist[-1]['acc_mean'], final_acc_max=hist[-1]['acc_max'], final_C=hist[-1]['C_mean'])); r.finish()
    open(base + '.wb', 'w').write('uploaded'); print('uploaded', nm)
