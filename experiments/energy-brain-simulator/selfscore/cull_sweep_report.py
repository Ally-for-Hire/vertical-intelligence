"""Summarise the cull-fraction experiments from saved run histories, one table per (stage, problem, inner). python selfscore/cull_sweep_report.py [stage ...]
g60/g95 = first logged generation where population-mean held-out accuracy >= 0.60 / 0.95 (speed to competence); sel-diff = survivors minus population accuracy."""
import json, glob, re, os, sys, numpy as np, collections
N = re.compile(r'(c\d+b?)_(\w+?)_(none|grow)_(p\d\d|one)_s(\d)')
R = collections.defaultdict(list); want = set(sys.argv[1:])
for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'c[234]*.json')):
    m = N.match(os.path.basename(f)[:-5])
    if m and (not want or m.group(1) in want): R[(m.group(1), m.group(2), m.group(3), m.group(4))].append(json.load(open(f)))
order = ['p00', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'one']
def first(r, thr): return next((h['g'] for h in r if h['acc_mean'] >= thr), None)
def mean_or_na(v): v = [x for x in v if x is not None]; return f'{np.mean(v):5.0f}' if v else '  n/a'
for stage, prob, inner in sorted({(k[0], k[1], k[2]) for k in R}):
    rows = [(c, R[(stage, prob, inner, c)]) for c in order if R.get((stage, prob, inner, c))]
    print(f'--- {stage} {prob} inner={inner}')
    print('cull   n  acc_mean  acc_max   g60    g95    C     sel_diff')
    for c, runs in rows:
        fin = [r[-1] for r in runs]
        print(f'{c:5s} {len(runs)}  {np.mean([x["acc_mean"] for x in fin]):.3f}     {np.mean([x["acc_max"] for x in fin]):.3f}   {mean_or_na([first(r, .60) for r in runs])} {mean_or_na([first(r, .95) for r in runs])} {np.mean([x["C_mean"] for x in fin]):5.1f}  {np.mean([r[len(r) // 2]["sel_diff"] for r in runs]):+.3f}')
