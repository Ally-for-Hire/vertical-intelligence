"""X1.15 tuning, part 2 (the first sweep hit the 10-minute tool limit in round 2; its round-1 keeps are the start point). Extends the knobs whose
kept value sat at the edge of the grid (ridge, ft_patience, ft_lr) and finishes round 2's remaining knobs. Same rule, same tuning worlds."""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiprocessing as mp
from constants import C
import flat, budget
import x1_15 as T

START = dict(flat.DEFAULT, n_props=4, refit_all=True, ridge=1.0, ft_lr=0.03, ft_patience=30)   # CONST: x115_grid
GRID2 = [("ridge", [0.3, 3.0, 10.0]), ("ft_patience", [60, 100]), ("ft_lr", [0.1]), ("stall", [0.0]), ("unit_lr", [0.01, 0.1]),   # CONST: x115_grid
         ("ft_steps", [500]), ("sweep", [False]), ("n_props", [6]), ("new", [0.0])]   # CONST: x115_grid

if __name__ == "__main__":
    from x0_1 import boot_ci
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    seeds = [C("x115_tune_seed0") + i for i in range(C("x115_tune_worlds"))]
    J = lambda rs: np.array([r["J"] for r in rs]); log = []
    with mp.get_context("spawn").Pool(budget.workers()) as pool:
        best = dict(START); cur = pool.starmap(T.eval_cfg, [(s, best) for s in seeds])
        print(f"start: J {np.median(J(cur)):.4f}", flush=True)
        for rnd in range(2):
            changed = False
            for knob, vals in GRID2:
                for v in vals:
                    if v == best[knob]: continue
                    cfg = dict(best, **{knob: v}); rs = pool.starmap(T.eval_cfg, [(s, cfg) for s in seeds])
                    ci = boot_ci(J(rs) - J(cur), C("x115_tune_seed0"))
                    log.append(dict(round=rnd, knob=knob, value=v, gain=ci, J=float(np.median(J(rs)))))
                    print(f"b{rnd} {knob}={v}: gain {ci[0]:+.4f} [{ci[1]:+.4f}, {ci[2]:+.4f}]  J {np.median(J(rs)):.4f}", flush=True)
                    if ci[0] > 0 and ci[1] > 0: best, cur, changed = cfg, rs, True; print(f"   -> keep {knob}={v}", flush=True)
            if not changed: break
    json.dump({"best": best, "start": START, "log_part2": log, "J_best": float(np.median(J(cur))),
               "note": "part 1 (x1_15.py tune) kept n_props=4, refit_all=True, ridge=1.0, ft_lr=0.03, ft_patience=30 in round 0 and nothing in round 1 up to stall=0.003; it was stopped by the tool limit"},
              open(os.path.join(T.OUT, "x1_15_tune.json"), "w"), indent=1, default=float)
    print("BEST", json.dumps(best, default=float), flush=True)
