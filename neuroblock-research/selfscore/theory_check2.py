"""E2: distribution-level theory (density operator) vs direct simulation, multi-generation, including heavy culls and score noise."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore import theory as T, density as D
grid = D.Grid(-0.5, 2.5, 6001)
print('mass check (retention integrates to k/n): ', end='')
f0 = D.from_gaussian(grid, 0.0, 0.05)
for n, k, se in [(256, 128, 0.0), (256, 26, 0.1), (256, 1, 0.1), (64, 6, 0.05)]:
    r = D.retention(grid, f0, n, k, se); print(f'{(f0 * r).sum() * grid.dz * n / k:.4f}', end=' ')
print('\n')
print('trajectory: simulation (300 reps) vs 2-moment recursion vs DENSITY operator.  mu_m=-0.002 sig_m=0.02, inner g~N(-0.001,0.01)')
print('   n    c    sig_e  gen |  mean: sim    moment   density |  sd: sim    moment   density')
for n, c, se in [(256, 0.5, 0.0), (256, 0.5, 0.1), (256, 0.9, 0.1), (256, 0.99, 0.1), (64, 0.9, 0.1), (256, 0.75, 0.02)]:
    k = max(1, n - int(round(c * n))); G = 40
    sim = T.simulate(n, k, se, -0.002, 0.02, G, sd0=0.05, mu_g=-0.001, sig_g=0.01, reps=300, seed=5)
    mom = T.trajectory(0.0, 0.05 ** 2, G, n, k, se ** 2, -0.002, 0.02 ** 2, -0.001, 0.01 ** 2)
    den, _ = D.trajectory(grid, D.from_gaussian(grid, 0.0, 0.05), G, n, k, se, -0.002, 0.02, -0.001, 0.01)
    for g in (4, 19, 39):
        print(f' {n:4d} {c:5.2f} {se:5.2f} {g + 1:4d} |      {sim[g, 0]:7.4f}  {mom[g, 0]:7.4f}  {den[g]["mean"]:7.4f} |      {np.sqrt(sim[g, 1]):7.4f}  {np.sqrt(mom[g, 1]):7.4f}  {np.sqrt(den[g]["var"]):7.4f}')
