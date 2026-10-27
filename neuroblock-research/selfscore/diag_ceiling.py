"""Diagnostics for 'culling seems to have no effect on parity4 accuracy' (user, 2026-10-26)."""
import itertools, numpy as np, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# (1) best accuracy of ANY network without hidden units on parity4 = best linear threshold function of 4 bits
X = np.array(list(itertools.product([0, 1], repeat=4)), float); y = (X.sum(1) % 2).astype(int)
rng = np.random.default_rng(0); best = 0; counts = {}
W = rng.normal(size=(400000, 4)); b = rng.normal(size=(400000, 1)) * 1.5
pred = ((X @ W.T + b.T) > 0).astype(int)                     # (16, n)
acc = (pred == y[:, None]).mean(0); best = acc.max()
print(f'(1) best accuracy of any linear threshold (no hidden units) on parity4: {best:.4f} = {int(round(best * 16))}/16  (400k random classifiers)')
# (2) hand-built parity4 network with hidden units inside the Pop engine: does the forward pass compute it? deterministic and stochastic
from selfscore.pop import Pop, dev
for stoch in (False, True):
    d, K, H = 4, 2, 4
    p = Pop(1, d, K, H, stochastic=stoch, S=16)
    p.W.zero_(); p.M.zero_(); p.b.zero_(); p.alive.zero_(); p.beta.fill_(1.0)
    xs = torch.tensor(X * 2 - 1, dtype=torch.float32, device=dev)          # inputs as +-1 as in the real runs
    # hidden h_k = [sum(x_pm) >= thr_k], thr from the bit-count: count = (sum_pm + 4)/2, thresholds count>=1..4  -> sum_pm >= -2,0,2,4 (minus 0.5)
    thr = [-2.5 + 0.0, -0.5, 1.5, 3.5]
    g = 6.0
    for k in range(4):
        j = d + k; p.alive[0, k] = True
        for i in range(d): p.M[0, i, j] = True; p.W[0, i, j] = g
        p.b[0, j] = -g * thr[k]
    o0 = d + H                                                              # output 0 = "answer 1 (odd)", output 1 = "answer 0 (even)"
    sgn = [1, -1, 1, -1]
    for k in range(4):
        p.M[0, d + k, o0] = True; p.W[0, d + k, o0] = 4.0 * sgn[k]
    p.b[0, o0] = -1.0                                                       # odd-parity detector: h1-h2+h3-h4 > 0.5
    p.M[0, d, o0 + 1] = False
    p.b[0, o0 + 1] = 0.0
    q = p.forward(xs)[0]                                                    # (B,K): P(feedback) for each action; here output 0 means 'predict 1', output 1 constant 0.5
    pred1 = (q[:, 0] > 0.5).long().cpu().numpy()
    print(f'(2) hand-built hidden-unit parity4, stochastic={stoch}: accuracy {(pred1 == y).mean():.4f}  (hidden units work if this is ~1.0)')
