"""The one learner shared by every arm: a small one-step value network trained by gradient descent
(Adam), as U18 asks (gradient descent inside; what differs between arms is what MANAGES it, here the
value signal). Input = object observation + the agent's own internal variables. Output = predicted
value of each action (0 = skip, 1 = eat). Trained on the taken action only (squared error)."""
import numpy as np
from constants import C


class ValueNet:
    def __init__(self, n_in, seed, lr):
        rng = np.random.default_rng(seed)
        H, s = C("hidden"), C("init_scale")
        self.p = {"W1": rng.standard_normal((n_in, H)) * s / np.sqrt(n_in), "b1": np.zeros(H),
                  "W2": rng.standard_normal((H, 2)) * s / np.sqrt(H), "b2": np.zeros(2)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.lr, self.t = lr, 0
        self.rng = np.random.default_rng(seed + 1)

    def q(self, z):
        h = np.tanh(z @ self.p["W1"] + self.p["b1"])
        return h @ self.p["W2"] + self.p["b2"], h

    def act(self, z):
        if self.rng.random() < C("epsilon"):
            return int(self.rng.integers(2))
        return int(np.argmax(self.q(z)[0]))

    def learn(self, z, a, r):
        out, h = self.q(z)
        g_out = np.zeros(2); g_out[a] = out[a] - r
        g = {"W2": np.outer(h, g_out), "b2": g_out}
        g_h = (self.p["W2"] @ g_out) * (1 - h * h)
        g["W1"], g["b1"] = np.outer(z, g_h), g_h
        self.t += 1
        b1, b2, eps = C("adam_b1"), C("adam_b2"), C("adam_eps")
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh, vh = self.m[k] / (1 - b1 ** self.t), self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + eps)
