"""One-line progress summary for phone push notifications (<200 chars). Uses the same gatherer as monitor.py (which is no longer served)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.monitor import gather
d = gather()
g = (d.get('grids') or [{}])[-1]
top = (d.get('board') or [None])[0]
s = f"{g.get('name','grid')}: {g.get('done',0)}/{g.get('total',0)}" + (f", ~{g['eta']:.0f} min left" if g.get('eta') else '')
if top: s += f" | best {top['cfg']} acc {top['acc_mean']:.2f} C {top['C']:.0f}"
if d.get('gpu'): s += f" | GPU {d['gpu']['util']}%"
if d.get('errors'): s += f" | {len(d['errors'])} errors"
print(s[:195])
