"""Q17: bitwise parity of the self-learning transformer at zero experience. prior=None-equivalent (an unfitted NNPrior returns p=1 everywhere -> bias = log 1 = 0)
must reproduce the plain transformer's per-tick errors EXACTLY, on 10 worlds."""
import sys
sys.path.insert(0, "../src")
import numpy as np, torch
torch.set_num_threads(1)
from worlds import sample_world, rollout
from tfm import run_scenario_tf
from selfnn import NNPrior

ok = []
for seed in range(7000, 7010):
    w = sample_world(seed)
    X, Y = rollout(w, 600, np.random.default_rng(seed + 7))
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    a = run_scenario_tf(X, Y, cp, w.n_act, 600, 100, seed, "plain", lr=1e-3)["err"]
    b = run_scenario_tf(X, Y, cp, w.n_act, 600, 100, seed, "prior", prior=NNPrior(), lr=1e-3)["err"]
    ok.append(bool(np.array_equal(a, b)))
print("bitwise identical per-tick errors on", sum(ok), "of", len(ok), "worlds")
