"""Per-problem encoder: every scalar -> n_thresholds level bits + n_thresholds increment bits (hard), thresholds spread over the problem's own
seen range, so a rescaled copy of a sensor yields the same bits. Bit names are '<variable>#<role>' (and 'grid#i'): the names are the global keys."""
import numpy as np
from constants import C


class ProblemEncoder:
    def __init__(self, prob):
        self.names = prob.names
        self.nt = C("n_thresholds")
        self.grid_n = 0
        self.prev = None
        self.state_names = []
        self.has_grid = prob.has_grid
        self.prob = prob

    def calibrate(self, S, g0=None):
        S = np.asarray(S)
        pc = C("range_percentile")
        frac = np.arange(1, self.nt + 1) / (self.nt + 1.0)
        lo, hi = np.percentile(S, pc, axis=0), np.percentile(S, 100 - pc, axis=0)
        self.q = lo[:, None] + (hi - lo)[:, None] * frac[None, :]
        d = np.diff(S, axis=0)
        dlo, dhi = np.percentile(d, pc, axis=0), np.percentile(d, 100 - pc, axis=0)
        self.dq = dlo[:, None] + (dhi - dlo)[:, None] * frac[None, :]
        self.grid_n = 0 if g0 is None else len(g0)
        self.state_names = [f"grid#{i}" for i in range(self.grid_n)] + [f"{v}#{r}" for v in self.names for r in range(2 * self.nt)]

    def encode(self, grid, s):
        dv = s - (self.prev if self.prev is not None else s)
        self.prev = s.copy()
        body = np.concatenate([(s[:, None] > self.q), (dv[:, None] > self.dq)], 1).astype(np.float32).ravel()
        if self.grid_n:
            return np.concatenate([grid.astype(np.float32), body])
        return body
