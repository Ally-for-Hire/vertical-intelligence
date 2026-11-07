"""E1: does the algebra match a direct simulation? (abstract Gaussian trait; no networks involved)"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore import theory as T
from scipy.stats import norm
# sanity of the building blocks against brute force
r = np.random.default_rng(1)
print('selection intensity i: formula (infinite n) vs finite-n exact vs brute-force Monte Carlo')
print('  n    k    p      i_inf   i_finite  MC(400k)')
for n, k in [(256, 128), (256, 64), (256, 26), (256, 3), (256, 1), (64, 6), (1024, 10)]:
    z = np.sort(r.standard_normal((4000, n)), 1)[:, -k:].mean(1).mean()
    print(f' {n:4d} {k:4d} {k / n:6.3f} {T.i_inf(k / n):7.4f} {T.i_finite(n, k):8.4f}  {z:7.4f}')
print()
print('one-generation selection differential S and variance, formula vs simulation (mean over 400 reps), gaussian trait sd 0.1')
print('  n    c     sig_e/sd  S_sim    S_theory   rel.err   var_sim  var_theory')
worst = 0
for n in (64, 256):
    for c in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
        k = max(1, n - int(round(c * n)))
        for se in (0.0, 0.5, 2.0):
            sim = T.simulate(n, k, se * 0.1, 0.0, 0.0, 1, sd0=0.1, reps=800, seed=3)[0]
            m, v, S, rho = T.step(0.0, 0.1 ** 2, n, k, (se * 0.1) ** 2, 0.0, 0.0)
            f = (n - k) / n
            rel = abs(sim[2] - S) / max(abs(S), 1e-9) if S > 1e-6 else 0.0; worst = max(worst, rel)
            if se in (0.0, 2.0) and c in (0.0, 0.5, 0.9, 0.99): print(f' {n:4d} {c:5.2f} {se:6.1f}   {sim[2]:7.5f}  {S:8.5f}   {rel:6.3f}   {sim[1]:7.5f}  {v:7.5f}')
print('worst relative error of the selection differential over the whole grid:', round(worst, 3))
print()
print('multi-generation trajectory (mean accuracy proxy), simulation vs two-moment recursion; mutation mu_m=-0.002 (harmful on average), sig_m=0.02, inner life g~N(-0.001,0.01)')
print('  n    c    sig_e   gens  mean_sim  mean_theory  sd_sim  sd_theory')
for n, c, se in [(256, 0.5, 0.0), (256, 0.5, 0.1), (256, 0.9, 0.1), (256, 0.99, 0.1), (64, 0.9, 0.1)]:
    k = max(1, n - int(round(c * n))); G = 40
    sim = T.simulate(n, k, se, -0.002, 0.02, G, sd0=0.05, mu_g=-0.001, sig_g=0.01, reps=300, seed=5)
    th = T.trajectory(0.0, 0.05 ** 2, G, n, k, se ** 2, -0.002, 0.02 ** 2, -0.001, 0.01 ** 2)
    for g in (4, 19, 39): print(f' {n:4d} {c:5.2f} {se:6.2f} {g + 1:4d}  {sim[g, 0]:8.4f}  {th[g, 0]:10.4f}  {np.sqrt(sim[g, 1]):7.4f}  {np.sqrt(th[g, 1]):7.4f}')
