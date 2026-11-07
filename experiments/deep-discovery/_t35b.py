import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, time
sys.path.insert(0, ".")
import numpy as np, refiner, wstream, wrec
g = float(sys.argv[2]); wrec.GAIN = g
W = wstream.Stream(960000, dict(noise=0.0, n_tr=300, lib_size=10, lib="rec", mix_extra=0.0)); d = [W.next() for _ in range(40)]
B = dict(size="grow", flat_off=True, n_par=1, hid=6)
CF = {"CORR": dict(B, parent_pick="corr"), "CORR_BAR": dict(B, parent_pick="corr", bar=True), "CLONE_BAR": dict(B, parent_pick="corr", clone=True, bar=True)}
name = sys.argv[1]; t = time.time(); L = refiner.Learner(960000, CF[name]); r = [L.step(x)["r2"] for x in d]
print(name, g, "R2", round(float(np.mean(r)), 3), "last10", round(float(np.mean(r[-10:])), 3), "units", len(L.units), "size", L.size(), round(time.time() - t, 1), "s", flush=True)
