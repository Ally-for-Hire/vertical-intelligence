"""Job queue for the self-scored grid: limited concurrency (a laptop GPU runs out of memory with many processes), skips finished jobs.
python selfscore/grid.py --stage c1 --workers 4
"""
import os, sys, subprocess, argparse, itertools, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CULL = {'nocull': '--cull 0.0 --rank ext --wean none', 'ext': '--rank ext --wean none', 'lin': '--rank Q --wean linear',
        'step': '--rank Q --wean step', 'Q': '--rank Q --wean all', 'C': '--rank C --wean all',
        # cull-fraction sweep (rank by the actual problem; the bottom fraction is pruned; one = only the single best model survives)
        'p00': '--cull 0.0 --rank ext', 'p25': '--cull 0.25 --rank ext', 'p50': '--cull 0.5 --rank ext', 'p75': '--cull 0.75 --rank ext',
        'p90': '--cull 0.9 --rank ext', 'p95': '--cull 0.95 --rank ext', 'p99': '--cull 0.99 --rank ext', 'one': '--keep 1 --rank ext'}
WB = ''          # set by --wandb: ' --wandb neuroblock-selfscore --name {name} --group {group}' (off by default: the user needs a free alternative)
STAGES = {
    'c1': dict(problems=['parity4', 'moons', 'mux6'], inner=['none', 'grow', 'grow_and_pred'], cull=['nocull', 'ext', 'lin', 'step', 'Q', 'C'], seeds=[0, 1], gens=1500, P=256),
    'c3': dict(problems=['majority5', 'and_or10'], inner=['none'], cull=['p00', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'one'], seeds=[0, 1, 2], gens=800, P=256),   # linearly solvable: cull sanity check
    'c2': dict(problems=['parity4', 'moons', 'mux6'], inner=['none', 'grow'], cull=['p00', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'one'], seeds=[0, 1, 2], gens=1500, P=256),
}
def jobs(stage):
    s = STAGES[stage]
    for p, i, c, sd in itertools.product(s['problems'], s['inner'], s['cull'], s['seeds']):
        yield stage, p, i, c, sd, s
def run(job):
    stage, p, i, c, sd, s = job
    out = os.path.join(HERE, 'runs', f'{stage}_{p}_{i}_{c}_s{sd}')
    if os.path.exists(out + '.json'): return out, 'skip'
    cmd = f'python -u {os.path.join(HERE, "cull.py")} --problem {p} --inner {i} {CULL[c]} --stoch --S 4 --seed {sd} --gens {s["gens"]} --P {s["P"]} --every 25 {WB.format(name=stage + "_" + p + "_" + i + "_" + c + "_s" + str(sd), group=stage + "_" + p)} --out {out}.json'
    with open(out + '.log', 'w') as f: r = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    return out, r.returncode
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', default='c1'); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--wandb', action='store_true'); a = ap.parse_args()
    if a.wandb: WB = ' --wandb neuroblock-selfscore --name {name} --group {group}'
    os.makedirs(os.path.join(HERE, 'runs'), exist_ok=True)
    J = list(jobs(a.stage)); t0 = time.time(); done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for out, rc in ex.map(run, J):
            done += 1; print(f'[{done}/{len(J)}] {os.path.basename(out)} rc={rc} ({(time.time() - t0) / 60:.1f} min)', flush=True)
