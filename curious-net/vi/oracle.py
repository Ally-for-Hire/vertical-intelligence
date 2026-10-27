"""Exact survival oracle for W0 (orthogonal=True): value iteration on a grid over the two internal
variables. Valid because with orthogonal directions the per-tick effect distribution (energy, integrity)
is the same in every season: a mixture of clipped N(0, effect_scale) pairs (learnable objects) and
N(0, noise_sd) pairs (noise objects). The oracle KNOWS the current object's effects; it maximises the
discounted count of -deaths. It is the CEILING arm and the ground truth for state-aware valence.
Cached to ../results/x0_6_oracle.npz."""
import os
import numpy as np
from constants import C

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "x0_6_oracle.npz")
DRIFT = None


def _drift():
    return np.array([-C("w0_energy_decay"), C("w0_integrity_heal")])


def sample_effects(M, rng, rho=0.0):
    s, c = C("w0_effect_scale"), C("w0_effect_clip")
    z = rng.standard_normal((M, 2))
    z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho * rho) * z[:, 1]
    learn = np.clip(s * z, -c, c)
    noise = rng.normal(0, C("w0_noise_sd"), (M, 2))
    return np.where((rng.random(M) < C("w0_noise_frac"))[:, None], noise, learn)


def interp(V, p):
    G = V.shape[0]
    x, y = p[..., 0] * (G - 1), p[..., 1] * (G - 1)
    i0 = np.clip(np.floor(x).astype(int), 0, G - 2)
    j0 = np.clip(np.floor(y).astype(int), 0, G - 2)
    fx, fy = x - i0, y - j0
    return (V[i0, j0] * (1 - fx) * (1 - fy) + V[i0 + 1, j0] * fx * (1 - fy)
            + V[i0, j0 + 1] * (1 - fx) * fy + V[i0 + 1, j0 + 1] * fx * fy)


def outcome(V, h, take, e):
    """Discounted value of one action from state h with object effect e (broadcastable)."""
    g = C("oracle_gamma")
    raw = h + take * e + _drift()
    died = (raw <= 0).any(-1)
    nxt = interp(V, np.clip(raw, 0, 1))
    resp = interp(V, np.full(2, C("w0_respawn"))[None, :])[0]
    return np.where(died, -1.0 + g * resp, g * nxt)


def solve(rho=0.0):
    G = C("oracle_grid")
    rng = np.random.default_rng(C("oracle_seed"))
    E = sample_effects(C("oracle_samples"), rng, rho)
    ax = np.linspace(0, 1, G)
    H = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1)
    V = np.zeros((G, G))
    for it in range(C("oracle_maxit")):
        Qs = outcome(V, H, 0.0, 0.0)
        Qe = outcome(V, H[:, :, None, :], 1.0, E[None, None, :, :])
        Vn = np.maximum(Qs[..., None], Qe).mean(-1)
        d = float(np.abs(Vn - V).max())
        V = Vn
        if d < C("oracle_tol"):
            break
    return V, it, d


def load(rho=0.0):
    path = CACHE if rho == 0.0 else CACHE.replace("x0_6_oracle", f"x0_7_oracle_rho{rho}")
    if os.path.exists(path):
        return np.load(path)["V"]
    V, it, d = solve(rho)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, V=V, iters=it, delta=d)
    return V


class Oracle:
    def __init__(self, rho=0.0):
        self.V = load(rho)

    def qdiff(self, h, e):
        """Q_eat - Q_skip at state h for an object with effect e."""
        h, e = np.asarray(h, float), np.asarray(e, float)
        return float(outcome(self.V, h, 1.0, e) - outcome(self.V, h, 0.0, 0.0))

    def value(self, h):
        return interp(self.V, np.asarray(h, float)[None, :])[0]


if __name__ == "__main__":
    import time
    t = time.time()
    o = Oracle()
    print("V range", float(o.V.min()), float(o.V.max()), "V(respawn)", o.value([0.5, 0.5]), "secs", round(time.time() - t, 1))
