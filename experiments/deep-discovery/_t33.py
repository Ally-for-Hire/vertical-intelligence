import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, time
sys.path.insert(0, ".")
import constants, x1_33
o = constants.C
x1_33.C = lambda k: 1 if k == "x133_t_per" else (3 if k == "x132_exam" else o(k))
open("../results/_t33.flag", "w").write("")
for fam, rung in (("CAP_base", 1), ("HID_base", 0), ("CAP_pool", 0)):
    t = time.time(); r = x1_33.cell((12, fam, rung, 960000)); print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, round(time.time() - t))
