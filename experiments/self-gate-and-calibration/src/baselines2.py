"""Neural baselines for the lifelong setting. Same bit inputs and targets as the network, same shared-name registry (so the MLP gets name-sharing
for free): input = zero-padded global vector, outputs only for the active targets of the current problem. Hyperparameters are baseline-only,
tuned by sweep (see experiments/expJ_curriculum.py)."""
import numpy as np


class GlobalMLP:
    def __init__(self, W, hidden, lr, seed=0, replay=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 1.0 / np.sqrt(64), (W, hidden)).astype(np.float64)
        self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, 1.0 / np.sqrt(hidden), (hidden, W))
        self.b2 = np.zeros(W)
        self.lr, self.replay = lr, replay
        self.buf, self.nseen = [], 0
        self.rng = r

    def _fwd_update(self, xin_idx, x, yout_idx, y, train=True):
        h = np.tanh(x @ self.W1[xin_idx] + self.b1)
        z = h @ self.W2[:, yout_idx] + self.b2[yout_idx]
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        loss = -(y * np.log2(pc) + (1 - y) * np.log2(1 - pc))
        if train:
            dz = p - y
            dh = (dz @ self.W2[:, yout_idx].T) * (1 - h * h)
            self.W2[:, yout_idx] -= self.lr * np.outer(h, dz)
            self.b2[yout_idx] -= self.lr * dz
            self.W1[xin_idx] -= self.lr * np.outer(x, dh)
            self.b1 -= self.lr * dh
        return loss

    def step(self, xin_idx, x, yout_idx, y):
        loss = self._fwd_update(xin_idx, x, yout_idx, y)
        if self.replay:
            item = (xin_idx, x.copy(), yout_idx, y.copy())
            self.nseen += 1
            if len(self.buf) < self.replay:
                self.buf.append(item)
            else:
                j = self.rng.integers(self.nseen)
                if j < self.replay:
                    self.buf[j] = item
            xi, xx, yi, yy = self.buf[self.rng.integers(len(self.buf))]
            self._fwd_update(xi, xx, yi, yy)
        return loss
