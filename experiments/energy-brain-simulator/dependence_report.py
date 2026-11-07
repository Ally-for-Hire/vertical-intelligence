"""Does the best value of a setting depend on the world?  Reads the four dependence grids from the Lab and prints the best value per row.
Usage: python dependence_report.py <id_a> <id_b> <id_c> <id_d>"""
import json
import sys

JOBS = r'~\claude-live\lab\jobs'
names = ['rho x D', 'rho x sigma', 'win x sigma', 'select x sigma']
for name, jid in zip(names, sys.argv[1:5]):
    g = json.load(open(f'{JOBS}\\{jid}\\grid.json', encoding='utf-8'))
    print(f'\n=== {name}  ({g["seeds"]} repeats)  across: {g["x"]["param"]}  down: {g["y"]["param"]}')
    for j, row in enumerate(g['cells']):
        accs = [c['acc']['mean'] if c else None for c in row]
        ns = [c['n']['mean'] if c else None for c in row]
        ts = [c['t']['mean'] if c else None for c in row]
        Fs = [c['F']['mean'] if c else None for c in row]
        best_acc = max(range(len(row)), key=lambda i: accs[i] if accs[i] is not None else -1)
        best_F = max(range(len(row)), key=lambda i: Fs[i] if Fs[i] is not None else -1)
        top = accs[best_acc]
        # smallest-neuron setting that is within 1 point of the best accuracy (what a sensible brain would choose)
        ok = [i for i in range(len(row)) if accs[i] is not None and accs[i] >= top - 0.01]
        lean = min(ok, key=lambda i: ns[i])
        print(f'  {g["y"]["param"]}={g["y"]["labels"][j]:>5}: best accuracy at {g["x"]["labels"][best_acc]:>5} ({top*100:5.1f}%), '
              f'best fitness at {g["x"]["labels"][best_F]:>5}, smallest brain within 1 pt at {g["x"]["labels"][lean]:>5} ({ns[lean]:.0f} neurons, {ts[lean]:.0f} us)')
        print('     acc by ' + g['x']['param'] + ': ' + ' '.join(f'{a*100:5.1f}' if a is not None else '  -  ' for a in accs))
