"""round 6: candidate settings judged directly on 20 seeds (seeds 200-219, never used before) to avoid picking lucky 5-seed winners; gradient references on the same seeds"""
import sys, os, json, itertools, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emergent.search import evolve, gradient_cost, data
Xtr, ytr, Xte, yte, K = data(); S = range(200, 220); OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'round6.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def med(v): return float(np.median(v))
for lr in (1e-3, 3e-2):
    k = f'gradient lr={lr}'
    if k not in res: res[k] = [gradient_cost(s, Xtr, ytr, Xte, yte, K, H=8, lr=lr)[0] for s in S]; json.dump(res, open(OUT, 'w'))
    print(f'{k}: median {med(res[k]):,.0f}', flush=True)
base = dict(life=True, fit='ll', H=8, recomb='none', noise='gauss')
grid = [dict(base, P=P, B=B, cull=c, init=i) for P, B, c, i in itertools.product((10, 20, 40), (16, 32), (0.8, 0.9), (0.5, 1.0, 2.0))]
for cfg in grid:
    k = json.dumps(cfg, sort_keys=True)
    if k not in res:
        r = [evolve(cfg, s, Xtr, ytr, Xte, yte, K, 1.0e6) for s in S]; res[k] = [(c, o) for c, o, _ in r]; json.dump(res, open(OUT, 'w'))
    c = [x for x, _ in res[k]]; n = sum(o for _, o in res[k])
    print(f'P={cfg["P"]:3d} B={cfg["B"]:3d} cull={cfg["cull"]} init={cfg["init"]}: solved {n}/20 | median {med(c):,.0f} = {med(c) / med(res["gradient lr=0.001"]):.2f}x default Adam, {med(c) / med(res["gradient lr=0.03"]):.1f}x tuned Adam', flush=True)
