"""GENERATOR VERSION NOTE (iteration 13): v2 since iteration 3b; the per-variable delayed-effect draw changed the random stream, so seeds built with v1 (exp K, L, L2, M, N1, N2) give different worlds than v2 (later experiments). Compare across experiments only via their own in-run plain arm.
Random scenario generator for the self-learning-network line. A scenario is a small stochastic dynamical system
    s_{t+1} = f(s_t, a_t) + noise
with a sparse random causal graph, a score, and an arbitrary variable ORDER / SCALE / OFFSET (so nothing leaks through position or units).
The ground-truth graph is returned only for evaluation. World-definition numbers live here (WORLD status), not in the learner."""
import numpy as np

KINDS = ("lag", "integrator", "tanh", "product", "copy")      # "xor" (pure interaction) is only used by the pure_interaction family


class World:
    def __init__(self, seed, n_state=6, n_act=2, n_distract=0, noise=0.05, max_parents=2, kinds=None, delay_prob=0.0, d_kinds=None, min_parents=1, fly_like=False):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n_core, self.n_act, self.n_d = n_state, n_act, n_distract
        kinds = tuple(kinds) if kinds else KINDS
        self.delay_prob = delay_prob
        self.n_state = n_state + n_distract
        self.noise = noise
        # core wiring: each core variable i has parents among core states + actions (always includes itself with a lag-type persistence)
        self.parents, self.kind, self.gain, self.coef, self.delayed = [], [], [], [], []
        for i in range(n_state):
            k = int(rng.integers(min_parents, max_parents + 1))
            if kinds == ("xor",):
                k = 2
            if fly_like:                                    # measured fly wiring: median ~9 inputs (>=5-synapse pairs), heavy tail
                k = int(np.clip(round(rng.lognormal(np.log(8.0), 0.5)), 4, 16))
            cand = [("s", j) for j in range(n_state) if j != i] + [("a", a) for a in range(n_act)]
            ch = [cand[q] for q in rng.choice(len(cand), size=min(k, len(cand)), replace=False)]
            self.parents.append(ch)
            self.kind.append(kinds[int(rng.integers(len(kinds)))])
            self.gain.append(float(rng.uniform(0.1, 0.6)))
            if fly_like:                                    # power-law strengths (exponent 0.6): top 3 of ~8 hold ~half the weight; ~47% inhibitory-like signs
                rk = rng.permutation(len(ch)) + 1.0
                cf = (1.0 / rk ** 0.6) / (1.0 / rk ** 0.6).sum() * 1.5
                self.coef.append(np.where(rng.random(len(ch)) < 0.47, -1.0, 1.0) * cf)
            else:
                self.coef.append(rng.choice([-1.0, 1.0], size=len(ch)) * rng.uniform(0.5, 1.5, size=len(ch)))
            self.delayed.append(rng.random(len(ch)) < delay_prob)
        # make sure every action affects at least one core variable (otherwise it is an irrelevant input)
        for a in range(n_act):
            if not any(("a", a) in p for p in self.parents):
                i = int(rng.integers(n_state))
                self.parents[i].append(("a", a)); self.coef[i] = np.append(self.coef[i], rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5)); self.delayed[i] = np.append(self.delayed[i], False)
        # distractors: autonomous AR(1) / sinusoid / white noise, independent of everything
        dk = tuple(d_kinds) if d_kinds else ("ar", "sin", "wn", "copy")
        self.d_kind = [dk[int(rng.integers(len(dk)))] for _ in range(n_distract)]
        self.d_src = rng.integers(0, max(n_state, 1), size=n_distract)
        self.d_par = rng.uniform(0.5, 0.95, size=n_distract)
        # presentation: random permutation, scale and offset of the observed variables
        self.perm = rng.permutation(self.n_state)
        self.scale = np.exp(rng.normal(0, 1.0, self.n_state))
        self.offset = rng.normal(0, 2.0, self.n_state)
        self.aperm = rng.permutation(n_act)
        self.target = int(rng.integers(n_state))
        self.s = rng.normal(0, 1, self.n_state)
        self.s_prev = self.s.copy()
        self.t = 0
        self.action_levels = np.array([-1.0, 0.0, 1.0])

    def _f(self, i, s, a):
        k = self.kind[i]
        sp = self.s_prev
        u = sum(c * ((sp[j] if dl else s[j]) if t == "s" else a[j]) for (t, j), c, dl in zip(self.parents[i], self.coef[i], self.delayed[i]))
        g = self.gain[i]
        if k == "xor":                                  # pure interaction: the change depends on the PRODUCT of two parents, no marginal association with either
            vals = [(sp[j] if dl else s[j]) if t == "s" else a[j] for (t, j), dl in zip(self.parents[i][:2], self.delayed[i][:2])]
            return 0.5 * s[i] + 0.5 * np.tanh(2.0 * self.coef[i][0] * self.coef[i][1] * vals[0] * vals[1])
        if k == "lag":
            return s[i] + g * (-s[i] + np.tanh(u))
        if k == "integrator":
            return 0.95 * s[i] + g * u
        if k == "tanh":
            return np.tanh(u + 0.5 * s[i])
        if k == "product":
            return 0.9 * s[i] + g * (u * np.tanh(s[i]))
        return 0.5 * s[i] + 0.5 * u

    def step(self, a):
        """a: vector of n_act values in {-1,0,1} (in the world's own action order)"""
        s, n = self.s, self.n_core
        new = s.copy()
        for i in range(n):
            new[i] = self._f(i, s, a) + self.noise * self.rng.normal()
        for q in range(self.n_d):
            j = n + q
            kd = self.d_kind[q]
            if kd == "ar":
                new[j] = self.d_par[q] * s[j] + 0.3 * self.rng.normal()
            elif kd == "sin":
                new[j] = np.sin(0.05 * (self.t + 17 * q) * (1 + self.d_par[q]))
            elif kd == "copy":
                new[j] = new[self.d_src[q]] + 0.3 * self.rng.normal()
            else:
                new[j] = self.rng.normal()
        self.s_prev = s.copy()
        self.s = np.clip(new, -6, 6)
        self.t += 1

    def observe(self):
        return (self.s[self.perm] * self.scale + self.offset).astype(np.float64)

    def score(self):
        return -abs(self.s[self.target])

    def truth(self):
        """boolean (n_out x (n_in+n_act)) in OBSERVED order: out i depends on in j (state) or action. Self-dependence included for persistent kinds."""
        n = self.n_state
        T = np.zeros((n, n + self.n_act), bool)
        inv = np.argsort(self.perm)                 # observed index of world variable w is position where perm==w
        pos = {w: int(np.nonzero(self.perm == w)[0][0]) for w in range(n)}
        for i in range(self.n_core):
            T[pos[i], pos[i]] = True
            for (t, j) in self.parents[i]:
                if t == "s":
                    T[pos[i], pos[j]] = True
                else:
                    T[pos[i], n + int(np.nonzero(self.aperm == j)[0][0])] = True
        for q in range(self.n_d):
            if self.d_kind[q] == "ar":
                T[pos[self.n_core + q], pos[self.n_core + q]] = True
        return T

    def truth_top(self, k=3):
        """like truth(), but only the k strongest parents (by |coefficient|) of each core output: what sparse recovery could realistically find"""
        n = self.n_state
        T = np.zeros((n, n + self.n_act), bool)
        pos = {wv: int(np.nonzero(self.perm == wv)[0][0]) for wv in range(n)}
        for i in range(self.n_core):
            T[pos[i], pos[i]] = True
            order = np.argsort(-np.abs(self.coef[i]))[:k]
            for q in order:
                t, j = self.parents[i][q]
                if t == "s":
                    T[pos[i], pos[j]] = True
                else:
                    T[pos[i], n + int(np.nonzero(self.aperm == j)[0][0])] = True
        return T

    def act_to_world(self, a_obs):
        """observed action vector (in observed order) -> world order"""
        w = np.zeros(self.n_act)
        for k in range(self.n_act):
            w[self.aperm[k]] = a_obs[k]
        return w


def rollout(world, T, rng, hold=5):
    """random held actions; returns X (T, n_state+n_act) inputs and Y (T, n_state) next-state targets, observed order, standardised later by the learner"""
    ns, na = world.n_state, world.n_act
    X = np.zeros((T, ns + na)); Y = np.zeros((T, ns))
    a = np.zeros(na)
    obs = world.observe()
    for t in range(T):
        if t % hold == 0:
            a = rng.integers(-1, 2, size=na).astype(float)
        X[t, :ns] = obs; X[t, ns:] = a
        world.step(world.act_to_world(a))
        obs = world.observe()
        Y[t] = obs
    return X, Y


def sample_world(seed):
    """diverse scenario distribution used by the meta-learning experiments (WORLD status)"""
    r = np.random.default_rng(seed + 123456)
    return World(seed, n_state=int(r.integers(4, 11)), n_act=int(r.integers(1, 4)), n_distract=int(r.choice([10, 30, 100, 200])),
                 noise=float(r.choice([0.02, 0.05, 0.1, 0.2])), max_parents=int(r.integers(1, 4)))


FAMILIES = {
    "standard": dict(),
    "kinds_tanh_product": dict(kinds=("tanh", "product")),
    "many_parents_4_6": dict(min_parents=4, max_parents=6),
    "high_noise_0.4": dict(noise=0.4),
    "copy_distractors_only": dict(d_kinds=("copy",)),
    "delayed_effects": dict(delay_prob=0.7),
    "fly_like": dict(fly_like=True),
    "pure_interaction": dict(kinds=("xor",), min_parents=2, max_parents=2),
}


def sample_world_family(seed, family):
    """same hyper-sampling as sample_world, with one family-specific change (out-of-family tests; WORLD status)"""
    r = np.random.default_rng(seed + 123456)
    kw = dict(n_state=int(r.integers(4, 11)), n_act=int(r.integers(1, 4)), n_distract=int(r.choice([10, 30, 100, 200])),
              noise=float(r.choice([0.02, 0.05, 0.1, 0.2])), max_parents=int(r.integers(1, 4)))
    if family == "fly_like":                           # fly-like worlds need enough variables to hold ~8 parents per output
        kw["n_state"] = int(r.integers(10, 19))
        kw["n_distract"] = int(r.choice([30, 100, 200]))
    kw.update(FAMILIES[family])
    return World(seed, **kw)
