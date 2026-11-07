import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, time
sys.path.insert(0, ".")
import numpy as np, cell, wstream, wrec, x1_35
g = float(sys.argv[1]); refit = int(sys.argv[2]); wrec.GAIN = g
W = wstream.Stream(960000, dict(noise=0.0, n_tr=300, lib_size=5, lib="rec")); data = [wstream.build(W.lib, sp, 960000, W.cfg) for sp in x1_35.room_specs(1, 1)]
L = cell.Learner(960000, dict(refit=refit)); t = time.time(); r = []
for i, d in enumerate(data):
    r.append(L.step(d)["r2"])
    if (i + 1) % 10 == 0: print("problem", i + 1, "R2 last10", round(float(np.mean(r[-10:])), 3), "nodes", len(L.nodes), round(time.time() - t), "s", flush=True)
exam = [wstream.build(W.lib, dict(t=90000 + j, S=[j % 5], a=[1.0], stage=0, has_new=False), 960000, W.cfg) for j in range(20)]
ex = L.exam(exam); print("g", g, "refit", refit, "exam", round(float(np.mean([e[0] for e in ex])), 3), "links", round(float(np.mean([e[1] for e in ex])), 2), "nodes", len(L.nodes), "size", L.size(), round(time.time() - t), "s", flush=True)
rng = np.random.default_rng(1); xp = rng.standard_normal((1000, 8)); V = W.lib.values(xp); rec = L.recovered(xp, V, list(range(5))); print("corr^2 per level", [round(v ** 2, 3) for v in rec.values()])
