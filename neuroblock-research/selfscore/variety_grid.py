"""Pre-registered variety grid (VARIETY_HYPOTHESIS.md): fixed here, not tuned after seeing results.
python selfscore/variety_grid.py --workers 4"""
import os, subprocess, argparse, itertools, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
P, GENS = 128, 300
PROBLEMS = ['xor2', 'parity3', 'xor2_d4']; MS = [1, 3, 8, 26, 64, 128]; EPS = [0.01, 0.1, 0.5]; SEEDS = [0, 1, 2]
def run(job):
    p, m, e, s = job; out = os.path.join(HERE, 'runs', f'v1_{p}_M{m}_E{e}_s{s}')
    if os.path.exists(out + '.json'): return out, 'skip'
    cmd = (f'python -u {os.path.join(HERE, "cull.py")} --problem {p} --inner grow --rank ext --wean none --stoch --S 4 --model v2 --keep {m} --slight {e} '
           f'--gens {GENS} --P {P} --every 10 --init_edges 0 --track_variety --seed {s} --out {out}.json')
    with open(out + '.log', 'w') as f: r = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    return out, r.returncode
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=4); a = ap.parse_args()
    J = list(itertools.product(PROBLEMS, MS, EPS, SEEDS)); t0 = time.time(); n = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for out, rc in ex.map(run, J):
            n += 1; print(f'[{n}/{len(J)}] {os.path.basename(out)} rc={rc} ({(time.time() - t0) / 60:.1f} min)', flush=True)
