"""X1.15 tuning, part 3: unit_lr was kept at the grid edge (0.01); test 0.003 with the same rule and worlds."""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
from constants import C
import budget
import x1_15 as T

if __name__ == "__main__":
    from x0_1 import boot_ci
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    p = os.path.join(T.OUT, "x1_15_tune.json"); tj = json.load(open(p)); best = tj["best"]
    seeds = [C("x115_tune_seed0") + i for i in range(C("x115_tune_worlds"))]
    J = lambda rs: np.array([r["J"] for r in rs])
    with mp.get_context("spawn").Pool(budget.workers()) as pool:
        cur = pool.starmap(T.eval_cfg, [(s, best) for s in seeds])
        cfg = dict(best, unit_lr=0.003); rs = pool.starmap(T.eval_cfg, [(s, cfg) for s in seeds])   # CONST: x115_grid
    ci = boot_ci(J(rs) - J(cur), C("x115_tune_seed0"))
    print(f"unit_lr=0.003: gain {ci[0]:+.4f} [{ci[1]:+.4f}, {ci[2]:+.4f}]; best J {np.median(J(cur)):.4f}; sec/world {np.median([r['sec'] for r in cur]):.2f}")
    tj["log_part3"] = dict(knob="unit_lr", value=0.003, gain=ci)   # CONST: x115_grid
    if ci[0] > 0 and ci[1] > 0: tj["best"] = cfg; print("   -> keep")
    json.dump(tj, open(p, "w"), indent=1, default=float)
