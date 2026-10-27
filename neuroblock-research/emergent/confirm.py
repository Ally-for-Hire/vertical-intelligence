"""One-command reproduction of the series-2 result (2026-10-26): blind random variation + external culling matches gradient descent at default settings on xorgauss.
20 fresh seeds (300-319, never used in the search). Prints evolution vs default Adam vs tuned Adam, costs in sample-forward-equivalents to reach 0.95 test accuracy.
python emergent/confirm.py      (about 1-2 minutes on the RTX 5070 laptop GPU)"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emergent.search import evolve, gradient_cost, data

BEST = dict(P=40, B=16, cull=0.8, init=0.5, life=True, fit='ll', H=8, recomb='none', noise='gauss')

if __name__ == '__main__':
    Xtr, ytr, Xte, yte, K = data(); S = range(300, 320)
    g1 = [gradient_cost(s, Xtr, ytr, Xte, yte, K, H=8, lr=1e-3)[0] for s in S]; g2 = [gradient_cost(s, Xtr, ytr, Xte, yte, K, H=8, lr=3e-2)[0] for s in S]
    r = [evolve(BEST, s, Xtr, ytr, Xte, yte, K, 1.0e6) for s in S]; c = [x for x, _, _ in r]
    print(f'gradient descent, default Adam (lr 0.001): median {np.median(g1):,.0f}')
    print(f'gradient descent, tuned Adam (lr 0.03):    median {np.median(g2):,.0f}')
    print(f'evolution (blind noise + external culling): solved {sum(o for _, o, _ in r)}/20, median {np.median(c):,.0f} = {np.median(c) / np.median(g1):.2f}x default Adam, {np.median(c) / np.median(g2):.1f}x tuned Adam')
