"""Online MLP baseline: same inputs (bits + action bits), same targets (next-state bits), prequential cross-entropy in bits, one SGD step per tick.
Hyperparameters (hidden width, learning rate) are NOT part of the agent; they are tuned by sweep in experiments/expI_regime.py and reported there."""
import numpy as np


class OnlineMLP:
    def __init__(self, n_in, n_out, hidden, lr, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 1.0 / np.sqrt(n_in), (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, 1.0 / np.sqrt(hidden), (hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.lr = lr

    def step(self, x, y):
        h = np.tanh(x @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        loss = -(y * np.log2(pc) + (1 - y) * np.log2(1 - pc))
        dz = (p - y)
        dh = (dz @ self.W2.T) * (1 - h * h)
        self.W2 -= self.lr * np.outer(h, dz)
        self.b2 -= self.lr * dz
        self.W1 -= self.lr * np.outer(x, dh)
        self.b1 -= self.lr * dh
        return loss
