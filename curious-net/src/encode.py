"""Encoders (hard / soft / qubit-style). Each scalar -> n_thresholds level readouts + n_thresholds increment readouts."""
import numpy as np
from constants import C

GRID_BITS = 49          # CONST: grid_bits
ACT_BITS = 3            # WORLD: a_L, a_R, a_T
ACTIONS = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1]], np.float32)   # WORLD: steer L/S/R x throttle
BP = {"hard": 8, "soft": 8, "qubit": 8}   # 2 * n_thresholds


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))      # NUMERICAL clip


class Encoder:
    def __init__(self, names, mode="hard", width=None):
        self.names, self.mode = names, mode
        self.width = C("soft_width") if width is None else width
        self.nt = C("n_thresholds")
        self.bp = 2 * self.nt
        self.prev = None
        self.var_of_bit = ["grid"] * GRID_BITS + [n for n in names for _ in range(self.bp)]
        self.n_state = len(self.var_of_bit)
        self.var_of_in = self.var_of_bit + ["a_L", "a_R", "a_T"]

    def calibrate(self, S):
        S = np.asarray(S)
        pc = C("range_percentile")
        frac = np.arange(1, self.nt + 1) / (self.nt + 1.0)
        lo, hi = np.percentile(S, pc, axis=0), np.percentile(S, 100 - pc, axis=0)
        self.sp = np.maximum(hi - lo, 0.0) / (self.nt + 1.0)
        self.q = lo[:, None] + (hi - lo)[:, None] * frac[None, :]
        d = np.diff(S, axis=0)
        self.dsc = d.std(0) + 1e-9                                       # NUMERICAL guard
        dlo, dhi = np.percentile(d, pc, axis=0), np.percentile(d, 100 - pc, axis=0)
        self.dsp = np.maximum(dhi - dlo, 0.0) / (self.nt + 1.0)
        self.dq = dlo[:, None] + (dhi - dlo)[:, None] * frac[None, :]

    def encode(self, grid, s):
        dv = s - (self.prev if self.prev is not None else s)
        self.prev = s.copy()
        g = grid.astype(np.float32)
        if self.mode == "hard":
            body = np.concatenate([(s[:, None] > self.q), (dv[:, None] > self.dq)], 1).astype(np.float32)
            return np.concatenate([g, body.ravel()])
        w = np.maximum(self.sp * self.width, 1e-12)[:, None]
        zs = np.where(self.sp[:, None] > 1e-9, sig((s[:, None] - self.q) / w), (s[:, None] > self.q).astype(float))
        if self.mode == "soft":
            wd = np.maximum(self.dsp * self.width, 1e-12)[:, None]
            inc = np.where(self.dsp[:, None] > 1e-9, sig((dv[:, None] - self.dq) / wd), (dv[:, None] > self.dq).astype(float))
            return np.concatenate([g, np.concatenate([zs, inc], 1).ravel()]).astype(np.float32)
        th = np.pi * zs
        pz = np.sin(th / 2) ** 2
        px = 0.5 * (1 + np.sin(th) * np.tanh(dv / self.dsc)[:, None])
        return np.concatenate([g, np.concatenate([pz, np.repeat(px[:, :1], 0, 1) if False else px], 1).ravel()]).astype(np.float32)
