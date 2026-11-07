"""Plain online neural networks (NumPy, Adam, batch size 1). These are the base learners of the self-learning-network line and its baselines.
Hyperparameters are baseline settings chosen by sweep on validation worlds (experiments/expK_headroom.py), not agent constants."""
import numpy as np


class AdamMLP:
    def __init__(self, n_in, n_out, hidden=64, lr=1e-3, seed=0, mask=None, group_lasso=0.0, in_dropout=0.0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 1.0 / np.sqrt(max(n_in, 1)), (n_in, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, 1.0 / np.sqrt(hidden), (hidden, n_out)); self.b2 = np.zeros(n_out)
        self.lr = lr
        self.m = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2, self.b2)]
        self.v = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2, self.b2)]
        self.t = 0
        self.mask = mask                      # optional (n_in,) 0/1 input mask (the only difference between the plain net and a gated one)
        self.gl, self.drop = group_lasso, in_dropout     # baselines only: group-lasso on first-layer rows (proximal step), input dropout (training only); 0 = plain net exactly
        self.drng = np.random.default_rng(seed + 991)

    def predict(self, x):
        if self.mask is not None:
            x = x * self.mask
        h = np.tanh(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2, h, x

    def step(self, x, y):
        """prequential: returns the prediction made BEFORE the update, then takes one Adam step on (x,y)"""
        p, h, xm = self.predict(x)
        if self.drop > 0:                        # the prequential prediction p above is made WITHOUT dropout; training uses a dropped copy
            dm = (self.drng.random(len(xm)) >= self.drop) / (1 - self.drop)
            xm = xm * dm
            h = np.tanh(xm @ self.W1 + self.b1)
            e = (h @ self.W2 + self.b2) - y
        else:
            e = p - y
        gW2 = np.outer(h, e); gb2 = e
        dh = (e @ self.W2.T) * (1 - h * h)
        gW1 = np.outer(xm, dh); gb1 = dh
        self.t += 1
        for i, (P, g) in enumerate(zip((self.W1, self.b1, self.W2, self.b2), (gW1, gb1, gW2, gb2))):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t); vh = self.v[i] / (1 - 0.999 ** self.t)
            P -= self.lr * mh / (np.sqrt(vh) + 1e-8)
        if self.gl > 0:                          # proximal group-lasso shrinkage of each input's weight row
            nrm = np.linalg.norm(self.W1, axis=1, keepdims=True) + 1e-12
            self.W1 *= np.maximum(0.0, 1.0 - self.lr * self.gl / nrm)
        return p


class RunStd:
    """online standardiser (mean/std of everything seen so far): same for every arm"""
    def __init__(self, n):
        self.n, self.m, self.s2 = 0, np.zeros(n), np.ones(n)

    def update(self, x):
        self.n += 1
        d = x - self.m
        self.m = self.m + d / self.n
        self.s2 = self.s2 + (d * (x - self.m) - self.s2) / self.n

    def __call__(self, x):
        return (x - self.m) / np.sqrt(np.maximum(self.s2, 1e-6))
