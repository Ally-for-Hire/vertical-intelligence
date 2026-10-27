"""Tiny smoke test, single process: fresh learner on a few D1 episodes, then the abs learner with sleep."""
import os, sys, random, time
os.system('')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runes as R
from learner import Learner

try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass
rng = random.Random(0)
t0 = time.time()
for mode in ('fresh', 'lib', 'abs'):
    L = Learner(mode)
    rng = random.Random(0)
    rows = []
    slot = 0
    for dom, n in (('D1', 8), ('D2', 8)):
        for i in range(n):
            ep = R.make_episode(rng, dom)
            if L.asleep:
                L.sleep_step(); rows.append(f'{dom}:zzz'); continue
            res = L.run_episode(ep)
            L.after_episode(res)
            rows.append(f'{dom}:{res["acc"]:.1f}/{res["work"]//1000}k')
    print(f'{mode:6s}', ' '.join(rows), f'| lib {len(L.library)} abs {len(L.abstractions)} sleeps {L.sleeps}', flush=True)
    if L.abstractions:
        for name, a in L.abstractions.items():
            print('   ', name, a['arity'], a['body'])
print('elapsed', round(time.time() - t0, 1), 's')
