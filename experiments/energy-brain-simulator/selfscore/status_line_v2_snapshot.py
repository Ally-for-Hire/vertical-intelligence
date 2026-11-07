"""One-line progress summary with sparklines for phone push notifications (<200 chars).
python selfscore/status_line.py            -> push line
python selfscore/status_line.py --long     -> multi-line table with sparklines for chat replies"""
import os, re, sys, glob, collections
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.monitor import gather
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, 'runs')
LINE = re.compile(r'g\s*(\d+) test acc mean ([\d.]+) max ([\d.]+).*?\| C ([\d.]+)')
NAME = re.compile(r'(c\d+)_(\w+?)_(none|grow_and_pred|grow|pred|nonreg)_(nocull|ext|lin|step|Q|C)_s(\d)')
BARS = '▁▂▃▄▅▆▇█'
def spark(v, lo=0.45, hi=1.0, n=10):
    if not v: return ''
    k = max(1, len(v) // n); v = [sum(v[i:i + k]) / len(v[i:i + k]) for i in range(0, len(v), k)][-n:]
    return ''.join(BARS[min(7, max(0, int((x - lo) / (hi - lo) * 8)))] for x in v)
curves = collections.defaultdict(list)
for lg in glob.glob(os.path.join(RUNS, 'c*_*.log')):
    m = NAME.match(os.path.basename(lg)[:-4])
    if not m: continue
    pts = [(int(x.group(1)), float(x.group(2)), float(x.group(4))) for x in map(LINE.search, open(lg, errors='ignore').read().splitlines()) if x]
    if pts: curves[(m.group(3), m.group(4))].append(pts)
rows = []
for (inner, cull), runs in curves.items():
    L = min(len(r) for r in runs); acc = [sum(r[i][1] for r in runs) / len(runs) for i in range(L)]; C = sum(r[L - 1][2] for r in runs) / len(runs)
    rows.append((acc[-1], inner, cull, acc, C, len(runs)))
rows.sort(reverse=True)
d = gather(); g = (d.get('grids') or [{}])[-1]
head = f"{g.get('name','grid')} {g.get('done',0)}/{g.get('total',0)}" + (f" ~{g['eta']:.0f}m left" if g.get('eta') else '')
if '--long' in sys.argv:
    print(head + f" | GPU {d.get('gpu',{}).get('util','?')}%")
    for a, inner, cull, acc, C, n in rows[:12]: print(f"{inner[:13]:13s} {cull:6s} {spark(acc)} acc {a:.2f} C {C:5.0f} (n={n})")
else:
    s = head + ''.join(f" | {inner[:4]}/{cull} {spark(acc, n=6)} {a:.2f}" for a, inner, cull, acc, C, n in rows[:3]) + f" | GPU {d.get('gpu',{}).get('util','?')}%"
    print(s[:195])
