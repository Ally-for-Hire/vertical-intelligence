"""X1.2: learning progress plus OPTIMISM toward the untried (follows X1.1, where learning progress was 6x slower than random at finding a
newly opened learnable region). Arms: LP (as X1.1), LPO (learning progress with an OPTIMISTIC start: an untried region starts at the largest
reward the drive can pay, 1.0; nothing tuned), NOVO (error-seeking with the same optimistic start), UR, NOV, RANDOM, ORACLE; fresh seeds.
Usage: python x1_2.py          (writes ../results/x1_2_run.json)"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_1 as M
import x0_1 as X

ARMS = ("UR", "NOV", "LP", "RANDOM", "LPO", "NOVO", "ORACLE")


def main():
    per = {a: {} for a in ARMS}
    for i in range(C("x01_worlds")):
        s = C("x12_seed0") + i
        for a in ARMS: per[a][s] = M.run(s, a)
    g = lambda a, k: np.array([per[a][s][k] for s in sorted(per[a])], float)
    med = lambda a, k: float(np.nanmedian(g(a, k)))
    trap_a = lambda a: g(a, "dark_a") + g(a, "noise_a")
    S = {k: {a: med(a, k) for a in ARMS} for k in ("dark_a", "noise_a", "l1_a", "dark_b", "noise_b", "l1_b", "l2_b", "t_l2", "err_noise", "err_dark_late", "acc_l2_end")}
    S["trap_a"] = {a: float(np.median(trap_a(a))) for a in ARMS}
    S["t_l2 LPO-LP"] = X.boot_ci(g("LPO", "t_l2") - g("LP", "t_l2"), C("x12_seed0"))
    S["t_l2 LPO-RANDOM"] = X.boot_ci(g("LPO", "t_l2") - g("RANDOM", "t_l2"), C("x12_seed0"))
    S["trap_a LPO-UR"] = X.boot_ci(trap_a("LPO") - trap_a("UR"), C("x12_seed0"))
    S["trap_a LPO-NOV"] = X.boot_ci(trap_a("LPO") - trap_a("NOV"), C("x12_seed0"))
    os.makedirs(X.OUT, exist_ok=True)
    json.dump({"per_world": per, "summary": S}, open(os.path.join(X.OUT, "x1_2_run.json"), "w"), indent=1, default=float)
    print(json.dumps(S, indent=1, default=float))


if __name__ == "__main__":
    main()
