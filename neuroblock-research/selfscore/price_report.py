"""Analyse price_run output: (1) identity residual, (2) exact conditional selection differential vs measured, (3) Gaussian closed form vs exact, (4) accuracy vs cull level.
python selfscore/price_report.py [selfscore/runs/price_and_or10.json]"""
import json, sys, os, numpy as np, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore import theory as T
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'price_and_or10.json')
R = json.load(open(path)); rng = np.random.default_rng(0)

def exact_ES(z, k, Be, reps=300):
    """E[S | z] and sd(S | z): expectation over the cull's own scoring noise (binomial: each of Be draws is correct w.p. z_i), random tie-break; no normality assumed"""
    z = np.asarray(z); P = len(z)
    cnt = rng.binomial(Be, np.clip(z, 0, 1), size=(reps, P)) / Be + 1e-9 * rng.random((reps, P))
    top = np.argpartition(-cnt, k - 1, axis=1)[:, :k]
    S = z[top].mean(1) - z.mean()
    return S.mean(), S.std()

# (1) identity
res = [abs(x['resid']) for r in R for x in r['recs']]
print(f'(1) ACCOUNTING IDENTITY  zbar\' - zbar = S_w + T_kids : {len(res)} generations from {len(R)} runs, max |residual| = {max(res):.2e}, mean = {np.mean(res):.2e}\n')

# (2)+(3)
rows = []
for r in R:
    P, k, Be = r['P'], r['k'], r['Be']
    for x in r['recs']:
        if x['sd'] < 1e-4 or k >= P: continue
        z = np.array(x['z']); e_exact, sd_exact = exact_ES(z, k, Be)
        sig_e2 = float((z * (1 - z)).mean() / Be); sd = x['sd']
        rho_th = sd / np.sqrt(sd ** 2 + sig_e2); S_g = rho_th * T.i_finite(P, k) * sd
        S_ge = (x['rho'] if x['rho'] == x['rho'] else 0.0) * T.i_finite(P, k) * sd
        rows.append(dict(P=P, k=k, Be=Be, c=r['c'], S=x['S'], E=e_exact, sdS=sd_exact, Sg=S_g, Sge=S_ge, zbar=x['zbar'], sd=sd, rho_th=rho_th, rho=x['rho']))
a = {k: np.array([row[k] for row in rows], float) for k in rows[0] if k != 'c'}
big = a['E'] > 0.002                                   # generations where the selection differential is not negligible
zs = (a['S'] - a['E']) / np.maximum(a['sdS'], 1e-6)
print(f'(2) EXACT CONDITIONAL SELECTION DIFFERENTIAL (shape-free, given the realised accuracies and the binomial scoring noise): {len(a["S"])} generations')
print(f'    mean(S_measured - E[S|z]) = {np.mean(a["S"] - a["E"]):+.5f}   (should be 0)   standardised residual mean {np.mean(zs):+.3f}, sd {np.std(zs):.3f} (should be 0 and 1)')
print(f'    corr(S_measured, E[S|z]) = {np.corrcoef(a["S"], a["E"])[0, 1]:.4f}\n')
print(f'(3) GAUSSIAN CLOSED FORM  S = rho i(n,k) sigma  vs the exact conditional expectation (generations with E[S]>0.002: {int(big.sum())})')
for name, key in (('theory rho (from noise model)', 'Sg'), ('measured rho', 'Sge')):
    rel = np.abs(a[key][big] - a['E'][big]) / a['E'][big]
    print(f'    {name:30s}: median |rel err| {np.median(rel):.3f}, 90th percentile {np.percentile(rel, 90):.3f}, corr with exact {np.corrcoef(a[key][big], a["E"][big])[0, 1]:.4f}, slope {np.polyfit(a["E"][big], a[key][big], 1)[0]:.3f}')
print(f'    theory rho vs measured rank-correlation: corr {np.nanmean(np.abs(a["rho_th"] - a["rho"])):.3f} mean abs difference\n')
print('    by score-noise level Be (smaller batch = noisier score); Gaussian-theory median relative error vs exact:')
for Be in sorted(set(a['Be'])):
    m = big & (a['Be'] == Be); rel = np.abs(a['Sg'][m] - a['E'][m]) / a['E'][m]; print(f'      Be={int(Be):5d}: n={int(m.sum()):5d}  median rel err {np.median(rel):.3f}   mean S measured {a["S"][m].mean():.4f}')
print('    by cull level c:')
cs = np.array([str(row['c']) for row in rows])
for c in ['0.25', '0.5', '0.75', '0.9', '0.95', '0.99', 'one']:
    m = big & (cs == c)
    if m.sum(): rel = np.abs(a['Sg'][m] - a['E'][m]) / a['E'][m]; print(f'      c={c:5s}: n={int(m.sum()):5d}  median rel err {np.median(rel):.3f}')

# (4) accuracy vs cull
print('\n(4) MEAN ACCURACY OF THE POPULATION AFTER g GENERATIONS, by cull level and scoring noise (P=256, mean over seeds)')
for Be in (32, 128, 1024):
    print(f'    Be={Be}:  c      g=3     g=6     g=10    g=20    g=40')
    for c in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 'one']:
        rs = [r for r in R if r['P'] == 256 and r['Be'] == Be and r['c'] == c]
        if rs: print(f'          {str(c):5s}  ' + '  '.join(f'{np.mean([r["recs"][g]["zbar"] for r in rs]):.4f}' for g in (3, 6, 10, 20, 40)))
