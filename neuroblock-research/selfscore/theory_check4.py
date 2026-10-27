"""E4: steady gain per generation vs cull fraction c. Simulation vs two-moment fixed point vs density operator. Where is the optimum cull?"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from selfscore import theory as T, density as D
grid = D.Grid(-0.5, 4.5, 6001)
n, sig_m, mu_m = 256, 0.02, -0.002
cs = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.996]
for se in (0.0, 0.05, 0.1):
    print(f'--- n={n}, score noise sig_e={se}, mutation mu_m={mu_m} sig_m={sig_m}; gain/generation = slope of the mean over generations 20-50')
    print('   c      k   gain_sim   gain_moment  gain_density   sd_sim')
    rows = []
    for c in cs:
        k = max(1, n - int(round(c * n))); G = 50
        sim = T.simulate(n, k, se, mu_m, sig_m, G, sd0=0.03, reps=200, seed=11)
        g_sim = (sim[-1, 0] - sim[19, 0]) / 30
        fp = T.fixed_point(n, k, se ** 2, mu_m, sig_m ** 2)
        den, _ = D.trajectory(grid, D.from_gaussian(grid, 0.0, 0.03), G, n, k, se, mu_m, sig_m)
        g_den = (den[-1]['mean'] - den[19]['mean']) / 30
        rows.append((c, g_sim, fp['gain'], g_den)); print(f' {c:6.3f} {k:4d}   {g_sim:8.5f}   {fp["gain"]:10.5f}   {g_den:10.5f}     {np.sqrt(sim[-1, 1]):.4f}')
    a = np.array(rows)
    print(f'  best cull:  simulation c*={a[a[:, 1].argmax(), 0]:.3f}   moment c*={a[a[:, 2].argmax(), 0]:.3f}   density c*={a[a[:, 3].argmax(), 0]:.3f}\n')
