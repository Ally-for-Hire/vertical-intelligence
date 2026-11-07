import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, time
sys.path.insert(0, ".")
import numpy as np, refiner, wstream
name = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 72
CF = {"none": dict(size="grow", parent_pick="none", flat_off=True), "corr": dict(size="grow", parent_pick="corr", flat_off=True, n_par=1),
      "all": dict(size="grow", parent_pick="all", flat_off=True), "all_pool": dict(size="grow", parent_pick="all", flat_off=True, pool="local")}
W = wstream.Stream(960000, dict(noise=0.0, n_tr=300, lib_size=12, lib="rec", mix_extra=0.0)); d = [W.next() for _ in range(T)]
lv = np.array([x["S"][0] for x in d]); t = time.time(); L = refiner.Learner(960000, CF[name]); r = []
for i, x in enumerate(d):
    r.append(L.step(x)["r2"])
    if (i + 1) % 12 == 0: print(name, i + 1, "units", len(L.units), round(time.time() - t), "s", flush=True)
r = np.array(r); m = lambda lo, hi: round(float(r[(lv >= lo) & (lv <= hi)].mean()), 2) if ((lv >= lo) & (lv <= hi)).any() else None
print(name, "R2 by level 0:", m(0, 0), "1:", m(1, 1), "2-3:", m(2, 3), "4+:", m(4, 99), "units", len(L.units), "weights", L.size(), round(time.time() - t), "s", flush=True)
