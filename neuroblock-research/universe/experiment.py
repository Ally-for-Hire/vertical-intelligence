"""Rune universe experiment (single process, low priority).
Curriculum D1 -> D2 -> D3 -> D4 (D4 shares no blocks: control). Per domain: mean work (search evaluations) per episode and function-solve accuracy.
Modes: fresh (no memory), lib (remembers programs), abs (remembers + chooses its own sleep and names recurring parts)."""
import os, sys, random, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import runes as R
from learner import Learner, C_COMPRESS
try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass
SEEDS, PER = 8, 10
ORDER = ['D1', 'D2', 'D3', 'D4']
out = {}
for mode in ('fresh', 'lib', 'abs'):
    work = {d: [] for d in ORDER}; acc = {d: [] for d in ORDER}; en = {d: [] for d in ORDER}; fcor = {d: [] for d in ORDER}
    sl, ab, lb, missed = [], [], [], []
    for seed in range(SEEDS):
        rng = random.Random(seed)
        L = Learner(mode)
        miss = 0
        for d in ORDER:
            for i in range(PER):
                ep = R.make_episode(rng, d)
                if L.asleep:
                    c = L.sleep_step(); en[d].append(c); miss += 1; continue
                r = L.run_episode(ep); L.after_episode(r)
                work[d].append(r['work']); acc[d].append(r['acc']); en[d].append(r['energy'])
                fcor[d].append(np.mean([f['correct'] for f in r['functions']]))
        sl.append(L.sleeps); ab.append(len(L.abstractions)); lb.append(len(L.library)); missed.append(miss)
    out[mode] = dict(work={d: float(np.mean(work[d])) for d in ORDER}, acc={d: float(np.mean(acc[d])) for d in ORDER},
                     energy={d: float(np.sum(en[d]) / SEEDS) for d in ORDER}, funcs_correct={d: float(np.mean(fcor[d])) for d in ORDER},
                     sleeps=float(np.mean(sl)), abstractions=float(np.mean(ab)), library=float(np.mean(lb)), missed=float(np.mean(missed)))
    o = out[mode]
    print(f'{mode:6s} work/ep ' + ' '.join(f'{d}:{o["work"][d]:8.0f}' for d in ORDER) + ' | acc ' + ' '.join(f'{o["acc"][d]:.2f}' for d in ORDER)
          + f' | E ' + ' '.join(f'{o["energy"][d]:6.2f}' for d in ORDER) + f' | sleeps {o["sleeps"]:.1f} abs {o["abstractions"]:.1f} lib {o["library"]:.1f} missed {o["missed"]:.1f}', flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_results.json'), 'w'), indent=1)
