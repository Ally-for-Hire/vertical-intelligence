"""A suite of heterogeneous problems with controlled overlap in structure, behind one interface.
Each problem exposes: scalars() (named floats), grid() (optional bit view), act_names (global action-bit names), actions (n_actions x n_bits),
step(a), score. Names are the only link between problems: the same name in two problems is an *offer* of shared meaning that the learner must
verify (trust), not an assumption. env.py (the car) defines its own world constants; the problems below define theirs in this file (WORLD)."""
import numpy as np
from env import CarEnv, ACCEL

DT = 0.1


class CarProblem:
    def __init__(self, seed=0, n_d=8, mirror=False, classes=None, key="car", kind=None, respawn=60, drag=None):
        from env import Track
        track = Track(seed, kind=kind) if kind else None
        self.env = CarEnv(seed=seed, n_distract=n_d, mirror=mirror, classes=classes, track=track, respawn=respawn, drag=drag)
        self.key = key
        self.names = self.env.names
        self.act_names = ["a_L", "a_R", "a_T"]
        self.actions = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1]], np.float32)
        self.has_grid = True

    @property
    def score(self):
        return self.env.score

    def scalars(self):
        return self.env.scalars()

    def grid(self):
        return self.env.view()

    def step(self, a):
        self.env.step(a)

    def truth(self):
        """variable-level parent sets per child (lag 1), only for variables shared by name across problems"""
        t = {}
        for p, c in self.env.truth_edges():
            t.setdefault(c, set()).add(p)
        return t


class Cart1D:
    """x' = x + v dt ; v' = v + (F (a_R - a_L) - drag v) dt ; reflecting walls. Shares the names v, score, a_L, a_R with the car but not its dynamics."""
    def __init__(self, seed=0, key="cart", **_):
        self.rng = np.random.default_rng(seed + 31)
        self.key = key
        self.drag = self.rng.uniform(0.4, 1.0)
        self.F = 3.0
        self.L = 5.0
        self.x, self.v, self.score = 0.0, 0.0, 1.0
        self.names = ["x", "v", "score"]
        self.act_names = ["a_L", "a_R"]
        self.actions = np.array([[1, 0], [0, 0], [0, 1]], np.float32)
        self.has_grid = False

    def scalars(self):
        return np.array([self.x, self.v, self.score])

    def grid(self):
        return None

    def step(self, a):
        u = (-1.0, 0.0, 1.0)[a]
        self.v += (self.F * u - self.drag * self.v) * DT
        self.x += self.v * DT
        if abs(self.x) > self.L:
            self.x = np.sign(self.x) * self.L
            self.v = -0.5 * self.v
        self.score = 1.0 - abs(self.x) / self.L

    def truth(self):
        return {"x": {"x", "v"}, "v": {"v", "a_L", "a_R"}, "score": {"x"}}


class Pendulum:
    """theta from upright; omega' = omega + (g sin(theta) + tau u - damp omega) dt. Shares sin, cos, omega, score, a_L, a_R with the car."""
    def __init__(self, seed=0, key="pend", **_):
        self.rng = np.random.default_rng(seed + 41)
        self.key = key
        self.th, self.om = float(self.rng.uniform(-3, 3)), 0.0
        self.g, self.tau, self.damp = 4.0, 6.0, 0.3
        self.names = ["theta", "omega", "sin", "cos", "score"]
        self.act_names = ["a_L", "a_R"]
        self.actions = np.array([[1, 0], [0, 0], [0, 1]], np.float32)
        self.has_grid = False
        self.score = 0.0

    def scalars(self):
        return np.array([self.th, self.om, np.sin(self.th), np.cos(self.th), self.score])

    def grid(self):
        return None

    def step(self, a):
        u = (-1.0, 0.0, 1.0)[a]
        self.om += (self.g * np.sin(self.th) + self.tau * u - self.damp * self.om) * DT
        self.th = (self.th + self.om * DT + np.pi) % (2 * np.pi) - np.pi
        self.score = 0.5 * (1 + np.cos(self.th))

    def truth(self):
        return {"theta": {"theta", "omega"}, "omega": {"theta", "omega", "sin", "a_L", "a_R"}, "sin": {"sin", "cos", "omega"},
                "cos": {"sin", "cos", "omega"}, "score": {"cos"}}


class RandomBoolNet:
    """12 bits, each next bit a random function of 2 others (+5% flips); 3 action bits force bits 0..2. Unrelated to every car variable
    but reuses the action-bit names a_L, a_R, a_T (adversarial naming)."""
    def __init__(self, seed=0, key="rbn", n=12, **_):
        self.rng = np.random.default_rng(seed + 51)
        self.key = key
        self.n = n
        self.par = np.array([self.rng.choice(n, 2, replace=False) for _ in range(n)])
        self.tab = self.rng.integers(0, 2, (n, 4))
        self.b = self.rng.integers(0, 2, n)
        self.names = [f"b{i}" for i in range(n)] + ["score"]
        self.act_names = ["a_L", "a_R", "a_T"]
        self.actions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        self.has_grid = False
        self.score = float(self.b.mean())

    def scalars(self):
        return np.concatenate([self.b.astype(float), [self.score]])

    def grid(self):
        return None

    def step(self, a):
        idx = self.b[self.par[:, 0]] * 2 + self.b[self.par[:, 1]]
        nb = self.tab[np.arange(self.n), idx]
        flip = self.rng.random(self.n) < 0.05
        nb = np.where(flip, 1 - nb, nb)
        if a > 0:
            nb[a - 1] = 1
        self.b = nb
        self.score = float(self.b.mean())

    def truth(self):
        t = {f"b{i}": {f"b{p}" for p in self.par[i]} for i in range(self.n)}
        t["score"] = {f"b{i}" for i in range(self.n)}
        return t


def warm_actions(prob, rng, n):
    return [int(rng.integers(len(prob.actions))) for _ in range(n)]
