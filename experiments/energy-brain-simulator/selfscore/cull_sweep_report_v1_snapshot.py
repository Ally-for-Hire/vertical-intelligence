"""Summarise the cull-fraction sweep (stage c2) from saved run histories. python selfscore/cull_sweep_report.py"""
import json, glob, re, os, numpy as np, collections
N = re.compile(r'c[23]b?_(\w+?)_(none|grow)_(p\d\d|one)_s(\d)')
R = collections.defaultdict(list)
for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'c[23]*.json')):
    m = N.match(os.path.basename(f)[:-5])
    if m: R[(m.group(1), m.group(2), m.group(3))].append(json.load(open(f)))
order = ['p00', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'one']
for prob in sorted({k[0] for k in R}):
    for inner in ('none', 'grow'):
        rows = [(c, R[(prob, inner, c)]) for c in order if R.get((prob, inner, c))]
        if not rows: continue
        print(f'--- {prob}  inner={inner}   (g60 = first generation where population-mean accuracy >= 0.60; sel-diff = survivors minus population accuracy)')
        print('cull   n  acc_mean  acc_max  g60   C     sel_diff')
        for c, runs in rows:
            fin = [r[-1] for r in runs]; g60 = [next((h['g'] for h in r if h['acc_mean'] >= 0.60), None) for r in runs]
            g60s = f'{np.mean([g for g in g60 if g is not None]):5.0f}' if any(g is not None for g in g60) else '  n/a'
            print(f'{c:5s} {len(runs)}  {np.mean([x["acc_mean"] for x in fin]):.3f}     {np.mean([x["acc_max"] for x in fin]):.3f}    {g60s}  {np.mean([x["C_mean"] for x in fin]):5.1f}  {np.mean([r[len(r) // 2]["sel_diff"] for r in runs]):+.3f}')
