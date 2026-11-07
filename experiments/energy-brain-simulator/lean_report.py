"""Print the lean-brain test results from the Lab.  Usage: python lean_report.py <grid_D_id> <grid_sigma_id> <compare_id>"""
import json
import sys

JOBS = r'~\claude-live\lab\jobs'
a, b, c = sys.argv[1:4]
for title, jid in (('MANY INPUTS (D)', a), ('NOISE (sigma)', b)):
    g = json.load(open(f'{JOBS}\\{jid}\\grid.json', encoding='utf-8'))
    print(f'\n=== {title}: accuracy % / neurons / thought us   (rows: self-tuning off, on)')
    print('      ' + ' '.join(f'{g["x"]["param"]}={l:>6s}' for l in g['x']['labels']))
    for j, row in enumerate(g['cells']):
        print(f'{g["y"]["labels"][j]:>5s} acc ' + ' '.join(f'{c_["acc"]["mean"]*100:12.1f}' for c_ in row))
        print(f'      n   ' + ' '.join(f'{c_["n"]["mean"]:12.0f}' for c_ in row))
        print(f'      us  ' + ' '.join(f'{c_["t"]["mean"]:12.0f}' for c_ in row))
d = json.load(open(f'{JOBS}\\{c}\\compare.json', encoding='utf-8'))
print('\n=== OLD brain vs LEAN self-adjusting brain (medium world, %d repeats)' % d['seeds'])
for k in ('acc', 't', 'n', 'F', 'cov'):
    m = d['metrics'][k]
    z = m['diff']['z']
    print(f'  {k:4s} old {m["a"]["mean"]:8.3f}  lean {m["b"]["mean"]:8.3f}  diff {m["diff"]["mean"]:+8.3f}  z {z if z is None else round(z, 1)}')
