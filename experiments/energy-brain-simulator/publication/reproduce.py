"""One command re-checks the VERIFIED claims of publication/CLAIMS_LEDGER.md in a few minutes:   python publication/reproduce.py
Each check prints PASS/FAIL with the measured value. Exit code 0 only if everything passes. Uses the GPU if present (CPU works, slower)."""
import os, sys, re, subprocess, time, itertools
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
rows = []
def check(claim, name, fn):
    t = time.time()
    try: ok, detail = fn()
    except Exception as e: ok, detail = False, f'error: {type(e).__name__}: {e}'
    rows.append((claim, name, ok, detail, time.time() - t)); print(f'{"PASS" if ok else "FAIL"}  {claim:3s} {name:62s} {detail}  [{time.time() - t:.0f}s]', flush=True)

def c1():
    from selfscore.price_run import run
    worst = 0.0; n = 0
    for P, k, Be, seed in ((64, 16, 128, 3), (256, 26, 32, 4), (256, 1, 1024, 5)):
        for x in run('and_or10', P, k, Be, 20, seed): worst = max(worst, abs(x['resid'])); n += 1
    return worst < 1e-6, f'max |residual| = {worst:.1e} over {n} real generations (expected 0)'

def c3a():
    from selfscore import theory as T
    r = np.random.default_rng(1); out = []
    for n, k in ((256, 26), (256, 1), (64, 6)):
        mc = np.sort(r.standard_normal((4000, n)), 1)[:, -k:].mean(1).mean(); out.append(abs(T.i_finite(n, k) / mc - 1))
    return max(out) < 0.02, f'exact finite-population intensity vs brute force: worst relative error {max(out):.3%}'

def c3b():
    """closed-form selection differential vs simulation, judged by z-score against the simulation's own Monte Carlo standard error (a bare percentage threshold fails on noisy cases, e.g. a single winner)"""
    from selfscore import theory as T
    worst_z, worst_rel = 0.0, 0.0
    for n in (64, 256):
        for c in (0.5, 0.9, 0.99):
            k = max(1, n - int(round(c * n)))
            for se in (0.0, 2.0):
                S = T.step(0.0, 0.1 ** 2, n, k, (se * 0.1) ** 2, 0.0, 0.0)[2]
                r = np.random.default_rng(11); z = 0.1 * r.standard_normal((6000, n)); sc = z + se * 0.1 * r.standard_normal(z.shape)
                top = np.argsort(-sc, 1)[:, :k]; Sr = np.take_along_axis(z, top, 1).mean(1) - z.mean(1); SE = Sr.std() / np.sqrt(6000)
                worst_z = max(worst_z, abs(Sr.mean() - S) / SE); worst_rel = max(worst_rel, abs(Sr.mean() - S) / S)
    return worst_z < 3.0, f'closed form vs simulation (6000 reps, 12 conditions): worst |z| = {worst_z:.1f} standard errors (pass < 3), worst relative error {worst_rel:.1%}'

def c2():
    from selfscore.price_run import run
    rng = np.random.default_rng(1)
    def exp_S(I, z, k, Be, shared, reps=150):
        P, n = I.shape; S = np.empty(reps)
        for r in range(reps):
            sc = I[:, rng.integers(0, n, Be)].mean(1) if shared else rng.binomial(Be, z) / Be
            sc = sc + 1e-9 * rng.random(P); S[r] = z[np.argpartition(-sc, k - 1)[:k]].mean() - z.mean()
        return S.mean(), S.std()
    zs, zi = [], []; P, k, Be = 256, 26, 128
    for x in run('and_or10', P, k, Be, 15, 128, keepI=True):
        if x['sd'] < 1e-4: continue
        I = x['I'].astype(float); z = I.mean(1); a, sa = exp_S(I, z, k, Be, True); b, sb = exp_S(I, z, k, Be, False)
        zs.append((x['S'] - a) / max(sa, 1e-6)); zi.append((x['S'] - b) / max(sb, 1e-6))
    return abs(np.mean(zs)) < 0.6 and np.mean(zi) > 1.0, f'standardised residual: shared batch {np.mean(zs):+.2f} (expected 0), independent-noise model {np.mean(zi):+.2f} (biased)'

def c_engine():
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'selfscore', 'test_pop2.py')], capture_output=True, text=True, cwd=ROOT).stdout
    acc = re.findall(r'acc ([\d.]+)', out); sp = re.search(r'output on parity4 inputs[^:]*: ([\d.]+)', out); cap = re.search(r'max output change ([\d.]+)', out)
    ok = len(acc) >= 2 and all(float(a) == 1.0 for a in acc[:2]) and sp and float(sp.group(1)) < 1e-4 and cap and float(cap.group(1)) < 1e-6
    return bool(ok), f'hand-built parity-4 brain accuracy {acc[:2]}, split-edge change {sp.group(1) if sp else "?"}, capacity-growth change {cap.group(1) if cap else "?"}'

def files():
    need = ['selfscore/THEORY.md', 'selfscore/VARIETY_HYPOTHESIS.md', 'publication/CLAIMS_LEDGER.md', 'car/compare_run_it2holdout.log', 'car/compare_run_it2dev.log', 'selfscore/runs/price_and_or10.json']
    miss = [f for f in need if not os.path.exists(os.path.join(ROOT, f))]
    return not miss, 'all evidence files present' if not miss else f'missing: {miss}'

print('Reproducing the verified claims (a few minutes)...')
check('C1', 'accounting identity on real culled populations', c1)
check('C3', 'finite-population selection intensity i(n,k)', c3a)
check('C3', 'closed-form selection differential vs simulation', c3b)
check('C2', 'shared-batch scoring vs independent-noise model', c2)
check('eng', 'engine unit tests (hand-built brain, split edge, growth)', c_engine)
check('doc', 'evidence files for the ledger exist', files)
bad = [r for r in rows if not r[2]]
print(f'\n{len(rows) - len(bad)}/{len(rows)} checks passed in {sum(r[4] for r in rows):.0f}s' + ('' if not bad else '  -> FAILED: ' + ', '.join(r[1] for r in bad)))
sys.exit(1 if bad else 0)
