"""W0: the shared world of the vertical-intelligence experiment plan (Knowledge-Wiki
domains/vertical-intelligence/wiki/experiment-plan.md). An encounter world: every tick one object
appears; the agent eats it or skips it. Eating changes the agent's two INTERNAL variables, energy and
integrity, which must stay above 0 (death -> respawn; learned weights are kept, deaths are counted).

Modules implemented in this version (v0): internal variables with metabolism and repair; objects whose
effects follow hidden directions shared across SEASONS (directions drift at each season swap, keeping
part of the old structure); an unlearnable NOISE class (marked by a visible flag, effects random);
food that is often harmful (energy and integrity directions independent).
Planned, not yet implemented: the dark corner, threats, impasse problems, the oracle teacher,
the real-data module. They will be added as the experiments that need them are registered.

The object stream depends only on the world seed, never on the agent's actions, so every arm of an
experiment sees exactly the same objects (paired comparison)."""
import numpy as np
from constants import C


def _unit(v):
    return v / np.linalg.norm(v)


class W0:
    def __init__(self, seed, orthogonal=False, corr=None):
        """orthogonal=True keeps the energy and integrity directions orthogonal (also across seasons), so the
        two effects are exactly independent and the effect distribution is the same in every season; this
        is what makes the exact survival oracle (oracle.py) valid. Default False reproduces X0.1/X0.1b."""
        self.orthogonal = orthogonal
        self.corr = 0.0 if (orthogonal and corr is None) else corr  # corr: fixed dot product u_i . u_e (None = free)
        self.rng = np.random.default_rng(seed)
        d = C("w0_feat_dim")
        self.u_e = _unit(self.rng.standard_normal(d))
        r = C("w0_tradeoff_corr")
        self.u_i = _unit(r * self.u_e + np.sqrt(1 - r * r) * _unit(self.rng.standard_normal(d)))
        if self.corr is not None:
            self.u_i = self._orth()
        self.t = 0
        self.season = 0

    def obs_dim(self):
        return C("w0_feat_dim") + 1  # features + visible noise flag

    def _orth(self):
        v = _unit(self.u_i - (self.u_i @ self.u_e) * self.u_e)
        return self.corr * self.u_e + np.sqrt(1 - self.corr * self.corr) * v

    def _drift(self, u):
        k = C("w0_season_keep")
        fresh = _unit(self.rng.standard_normal(u.size))
        fresh = _unit(fresh - (fresh @ u) * u)  # orthogonal to u, so corr(new, old) = k exactly
        return _unit(k * u + np.sqrt(1 - k * k) * fresh)

    def effects(self, x, noise, nz=None):
        """(d_energy, d_integrity) of eating an object with features x."""
        if noise:
            return nz
        s, c = C("w0_effect_scale"), C("w0_effect_clip")
        return np.clip(s * np.array([self.u_e @ x, self.u_i @ x]), -c, c)

    def next_object(self):
        """Advance one tick; return (obs, true effects). Season swaps happen here."""
        if self.t > 0 and self.t % C("w0_season_len") == 0:
            self.u_e, self.u_i = self._drift(self.u_e), self._drift(self.u_i)
            if self.corr is not None:
                self.u_i = self._orth()
            self.season += 1
        self.t += 1
        x = self.rng.standard_normal(C("w0_feat_dim"))
        noise = self.rng.random() < C("w0_noise_frac")
        nz = self.rng.normal(0, C("w0_noise_sd"), 2)
        obs = np.append(x, float(noise))
        return obs, self.effects(x, noise, nz)

    @staticmethod
    def apply(h, eat, eff):
        """Internal-variable update. Returns (h_after_clipped, died, h_before_respawn)."""
        h = h + (eff if eat else 0)
        h = h + np.array([-C("w0_energy_decay"), C("w0_integrity_heal")])
        h = np.clip(h, 0, 1)
        died = bool((h <= 0).any())
        pre = h.copy()
        if died:
            h = np.full(2, C("w0_respawn"))
        return h, died, pre


def check(seed=12345, n=20000):
    """Verify each implemented module does what it claims. Returns a dict of measurements."""
    w, w2 = W0(seed), W0(seed)
    X, E, N = [], [], []
    same = True
    for _ in range(n):
        (o, e), (o2, e2) = w.next_object(), w2.next_object()
        same &= np.array_equal(o, o2) and np.array_equal(e, e2)
        X.append(o[:-1]); E.append(e); N.append(o[-1] > 0)
    X, E, N = np.array(X), np.array(E), np.array(N)

    def r2(mask, j):
        A = np.c_[X[mask], np.ones(mask.sum())]
        coef, *_ = np.linalg.lstsq(A, E[mask, j], rcond=None)
        res = E[mask, j] - A @ coef
        return 1 - res.var() / E[mask, j].var()
    first = np.arange(n) < C("w0_season_len")  # one season, so the linear fit is exact apart from clipping
    out = {
        "stream_deterministic": bool(same),
        "noise_share": float(N.mean()),
        "r2_learnable_energy": float(r2(first & ~N, 0)),
        "r2_noise_energy": float(r2(first & N, 0)),
        "harmful_food_share": float(((E[:, 0] > 0) & (E[:, 1] < 0) & ~N).mean() / (~N).mean()),
        "seasons_seen": int(w.season),
    }
    out["pass"] = (out["stream_deterministic"] and out["r2_learnable_energy"] > 0.9
                   and out["r2_noise_energy"] < 0.05 and out["harmful_food_share"] > 0.15)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(check(), indent=1))
