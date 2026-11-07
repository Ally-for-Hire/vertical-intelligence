"""What is the best accuracy the brains' function class could possibly reach on the 24-problem task? (gradient-trained reference models on the SAME inputs, task code included)
 (a) shared linear   : one weight matrix over [16-line bus + 5 task-code lines] -> outputs, shared by all 24 problems. This is exactly what a T=1 brain is (hidden nodes cannot affect the output at T=1).
 (b) per-problem linear: a separate linear classifier for each problem (more than the brain can do at T=1: the task code could only shift a bias).
 (c) shared MLP, 1 hidden layer (what a T>=2 brain could represent in principle).
python selfscore/ceiling.py"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.multi import POOL
from selfscore.multi_big import BusTests
from selfscore.pop2 import dev
torch.manual_seed(0); T = BusTests(POOL); K = T.Kmax
Xtr = {p: T.full_x(p, T.train0[p][0]) for p in POOL}; Ytr = {p: T.train0[p][1] for p in POOL}
Xte = {p: T.full_x(p, T.ref[p][0]) for p in POOL}; Yte = {p: T.ref[p][1] for p in POOL}
Km = {p: (torch.arange(K, device=dev) < T.K[p]) for p in POOL}
def acc(logit_fn):
    out = {}
    for p in POOL:
        lg = logit_fn(p, Xte[p]).masked_fill(~Km[p][None], -1e9); out[p] = float((lg.argmax(1) == Yte[p]).float().mean())
    return out
def fit(params, logit_fn, steps=3000, lr=0.03):
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        loss = 0
        for p in POOL: loss = loss + torch.nn.functional.cross_entropy(logit_fn(p, Xtr[p]).masked_fill(~Km[p][None], -1e9), Ytr[p])
        opt.zero_grad(); loss.backward(); opt.step()
res = {}
W = torch.zeros(T.D, K, device=dev, requires_grad=True); b = torch.zeros(K, device=dev, requires_grad=True)
f = lambda p, x: x @ W + b; fit([W, b], f); res['(a) shared linear + task code (= a T=1 brain)'] = acc(f)
Wp = {p: torch.zeros(T.D, K, device=dev, requires_grad=True) for p in POOL}; bp = {p: torch.zeros(K, device=dev, requires_grad=True) for p in POOL}
g = lambda p, x: x @ Wp[p] + bp[p]; fit(list(Wp.values()) + list(bp.values()), g); res['(b) separate linear classifier per problem'] = acc(g)
for H in (32, 128):
    W1 = (torch.randn(T.D, H, device=dev) * 0.3).requires_grad_(); b1 = torch.zeros(H, device=dev, requires_grad=True); W2 = (torch.randn(H, K, device=dev) * 0.3).requires_grad_(); b2 = torch.zeros(K, device=dev, requires_grad=True)
    h = lambda p, x: torch.tanh(x @ W1 + b1) @ W2 + b2; fit([W1, b1, W2, b2], h, steps=4000); res[f'(c) shared MLP, {H} hidden units, task code'] = acc(h)
print(f'{"model":52s} mean accuracy over 24 problems | problems at 90%+')
for k, v in res.items(): print(f'{k:52s} {sum(v.values()) / len(v):.3f}                          | {sum(x >= 0.9 for x in v.values())}')
print('(evolved brains in the runs: mean ~0.56-0.61, 1 problem at 90%+; random brains: 0.451)')
print('\nper problem (a shared linear | b per-problem linear | c MLP128):')
for p in POOL: print(f'  {p:14s} {res["(a) shared linear + task code (= a T=1 brain)"][p]:.2f} | {res["(b) separate linear classifier per problem"][p]:.2f} | {res["(c) shared MLP, 128 hidden units, task code"][p]:.2f}')
