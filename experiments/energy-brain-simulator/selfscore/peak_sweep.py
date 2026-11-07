"""Peak-of-the-wave test (user, 2026-10-26): "run a test literally comparing performance of the peak of the wave from 0.1 to 0.4".
Every run repeats wave1000_s0 exactly except the wave's PEAK (cull_max): rank by mean score, cosine cull wave from 0.15 up to the peak and back every 1000 generations,
refill = 100% blends, the old hand-set starting brains (wave1000_s0 predates the all-random start), 1000 brains, 3000 generations, no early stop.
Peaks 0.10 .. 0.40 in steps of 0.05, plus 0.50 (= the original wave1000_s0 setting, as the reference); 3 seeds each; one run at a time; finished runs are skipped, so re-running continues.
At peak 0.10 the 'wave' dips from 0.15 down to 0.10 and back; at 0.15 the cull is a constant 15%.
python selfscore/peak_sweep.py           (writes selfscore/runs/peak_sweep_summary.md)"""
import os, sys, json, subprocess, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, 'runs')
PEAKS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]; SEEDS = [0, 1, 2]


def tag(pk, sd): return f'peak{int(round(pk * 100)):02d}_s{sd}'


def run_one(pk, sd):
    t = tag(pk, sd); js = os.path.join(RUNS, f'multi_{t}.json')
    if os.path.exists(js) and len(json.load(open(js))['recs']) >= 3000: return
    cmd = [sys.executable, '-u', os.path.join(HERE, 'multi_big.py'), '--tag', t, '--seed', str(sd), '--P', '1000', '--gens', '3000', '--target', '0',
           '--cull_rule', 'mean', '--mode_p', '0,0,0,1,0,0,0', '--cull_min', '0.15', '--cull_max', str(pk), '--cull_period', '1000', '--cull_shape', 'cos', '--hand_init',
           '--wandb', 'neuroblock-emergence', '--group', 'peak-sweep']
    with open(os.path.join(RUNS, f'multi_{t}.out'), 'a') as o, open(os.path.join(RUNS, f'multi_{t}.err'), 'a') as e:
        subprocess.run(cmd, stdout=o, stderr=e, env=dict(os.environ, PYTHONIOENCODING='utf-8'), cwd=os.path.dirname(HERE))


def metrics(pk, sd):
    r = json.load(open(os.path.join(RUNS, f'multi_{tag(pk, sd)}.json')))['recs']
    z = [x['z2'] for x in r if x.get('z2') is not None]
    last = [x['score_mean'] for x in r[-500:]]
    return dict(final_ref=z[-1], best_ref=max(z), last500=sum(last) / len(last), best_brain_max=max(x['score_max'] for x in r))


def summary():
    rows = ['| wave peak | seeds done | population mean accuracy at the end (reference data) | best population mean during the run | population mean, last 500 generations | best single brain (batch) |', '|---|---|---|---|---|---|']
    for pk in PEAKS:
        ms = [metrics(pk, sd) for sd in SEEDS if os.path.exists(os.path.join(RUNS, f'multi_{tag(pk, sd)}.json')) and len(json.load(open(os.path.join(RUNS, f'multi_{tag(pk, sd)}.json')))['recs']) >= 3000]
        if not ms: rows.append(f'| {pk:.2f} | 0 | | | | |'); continue
        f = lambda k: f'{st.mean(m[k] for m in ms):.3f}' + (f' +/- {st.stdev(m[k] for m in ms):.3f}' if len(ms) > 1 else '')
        rows.append(f'| {pk:.2f}{" (original)" if pk == 0.5 else ""} | {len(ms)} | {f("final_ref")} | {f("best_ref")} | {f("last500")} | {f("best_brain_max")} |')
    txt = '# Peak-of-the-wave test\n\nEach row: wave1000_s0 repeated with a different wave peak (trough 0.15, period 1000, blends, 1000 brains, 3000 generations); mean +/- SD over seeds.\n\n' + '\n'.join(rows) + '\n'
    open(os.path.join(RUNS, 'peak_sweep_summary.md'), 'w', encoding='utf-8').write(txt); print(txt, flush=True)


if __name__ == '__main__':
    for sd in SEEDS:                                  # seed-major order: a complete first pass over all peaks comes early
        for pk in PEAKS: run_one(pk, sd); summary()
