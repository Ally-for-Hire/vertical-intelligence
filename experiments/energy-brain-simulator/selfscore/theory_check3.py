"""E3: is the density-operator error caused by clonal (non-independent) populations? Vary mutation spread sig_m (more spread = less clonal)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore import theory as T, density as D
grid = D.Grid(-0.5, 3.5, 8001)
print('n=256, c=0.9 (k=26), score noise sig_e=0.1, initial sd 0.05; 20 generations; mean over 300 sim reps')
print(' sig_m   mean_sim  mean_density  err%    sd_sim  sd_density')
for sm in (0.02, 0.04, 0.08, 0.16, 0.32):
    n, k, se, G = 256, 26, 0.1, 20
    sim = T.simulate(n, k, se, 0.0, sm, G, sd0=0.05, reps=300, seed=7)
    den, _ = D.trajectory(grid, D.from_gaussian(grid, 0.0, 0.05), G, n, k, se, 0.0, sm)
    print(f' {sm:5.2f}   {sim[-1, 0]:8.4f}  {den[-1]["mean"]:10.4f}  {100 * (den[-1]["mean"] / sim[-1, 0] - 1):+6.1f}   {np.sqrt(sim[-1, 1]):7.4f}  {np.sqrt(den[-1]["var"]):7.4f}')
print()
print('same, but kids are NOT clones: each kid is a fresh iid draw around a random survivor with a large spread (sanity: independence restores the theory)')
