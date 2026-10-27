"""Why do measured selection differentials exceed the independent-noise prediction? Every individual is scored on the SAME random batch (common random numbers),
so their scoring errors are positively correlated and the ranking is more accurate. Test with the actual correctness matrix I (P x patterns) of real populations:
  E_shared = expectation over the cull's batch draw with ONE shared batch (what the code does)   E_indep = the same but an independent batch per individual (the earlier model)"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.price_run import run
rng = np.random.default_rng(1)
def exp_S(I, z, k, Be, shared, reps=300):
    P, n = I.shape; S = np.empty(reps)
    for r in range(reps):
        if shared: idx = rng.integers(0, n, Be); sc = I[:, idx].mean(1)
        else: sc = np.array([I[i, rng.integers(0, n, Be)].mean() for i in range(P)]) if P <= 64 else rng.binomial(Be, z) / Be
        sc = sc + 1e-9 * rng.random(P); top = np.argpartition(-sc, k - 1)[:k]; S[r] = z[top].mean() - z.mean()
    return S.mean(), S.std()
print('P=256 real populations (and_or10), 25 generations x 2 seeds per cell. standardised residual = (S_measured - E)/sd: should have mean 0 and sd 1 if the model is exact')
print('   Be     c     |  shared batch: mean   sd   |  independent batches: mean   sd  |  mean S_measured  E_shared  E_indep')
for Be in (32, 128, 1024):
    for c in (0.5, 0.9, 0.99):
        P = 256; k = max(1, P - int(round(c * P))); zs_sh, zs_in, Sm, Es, Ei = [], [], [], [], []
        for seed in (1, 2):
            for x in run('and_or10', P, k, Be, 25, 100 * seed + Be, keepI=True):
                if x['sd'] < 1e-4: continue
                I = x['I'].astype(float); z = I.mean(1)
                a, sa = exp_S(I, z, k, Be, True); b, sb = exp_S(I, z, k, Be, False)
                zs_sh.append((x['S'] - a) / max(sa, 1e-6)); zs_in.append((x['S'] - b) / max(sb, 1e-6)); Sm.append(x['S']); Es.append(a); Ei.append(b)
        print(f' {Be:5d} {c:5.2f}   |   {np.mean(zs_sh):+6.2f} {np.std(zs_sh):5.2f}       |   {np.mean(zs_in):+6.2f} {np.std(zs_in):5.2f}          |   {np.mean(Sm):.4f}       {np.mean(Es):.4f}    {np.mean(Ei):.4f}')
