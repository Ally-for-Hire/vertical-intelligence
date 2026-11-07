import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, time
sys.path.insert(0, ".")
import constants, x1_35
o = constants.C
x1_35.C = lambda k: 3 if k == "x135_rooms" else (6 if k == "x132_exam" else o(k))
a = sys.argv[1]; g = float(sys.argv[2]); t = time.time(); r = x1_35.cell((g, a, 960000))
for rm in r["rooms"]: print(a, g, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rm.items()}, flush=True)
print(a, g, "total", round(time.time() - t), "s", flush=True)
