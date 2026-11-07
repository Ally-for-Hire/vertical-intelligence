"""wstream.py: an open-ended world that is FINITE at every moment and expands only when asked (author, 2026-11-02: 'loop in a constant process of
training the learner and expanding the complexity. save these worlds for future experimentation').

Stream(seed): starts with winf's grammar and a fixed library of FINITE['lib_size'] concepts. next() draws one problem from the CURRENT library
(popularity-driven mix, continuous weights, fresh points); expand(n) adds n concepts built on the existing ones (nothing else ever adds concepts).
Every problem's points come from their own seed [seed, 5, t], so a saved world replays bit-exactly.
save(path) writes the realized world as JSON (library: every concept's op, parents, parameters, standardisation; problems: concepts, weights,
stage; expansion points). load(path) -> (data, info): the same problems, identical numbers, for any future learner.
Usage: python wstream.py      (self-check: save -> load reproduces every problem exactly)"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import json, math, os
import numpy as np
import winf

POINT_STREAM = 5                                                      # CONST: winf_cfg


class Stream:
    def __init__(self, seed, cfg=None):
        self.cfg = dict(winf.CFG, **{**winf.FINITE, **(cfg or {})}); self.seed = seed
        self.rng = np.random.default_rng([seed, 1]); self.lib = (__import__('wrec').RecLibrary if self.cfg.get('lib') == 'rec' else winf.Library)(self.rng, self.cfg)
        for _ in range(self.cfg["pool0"]): self.lib.new(base=True)
        while len(self.lib.c) < self.cfg["lib_size"]: self.lib.new()
        self.t, self.stage, self.problems, self.events, self.newest = 0, 0, [], [], []

    def expand(self, n):
        self.newest = [self.lib.new() for _ in range(n)]; self.stage += 1
        self.events.append(dict(t=self.t, added=len(self.newest), concepts=len(self.lib.c)))

    def next(self):
        r, lib, cfg = self.rng, self.lib, self.cfg
        k = min(cfg["mix_cap"], 1 + r.poisson(cfg["mix_extra"]))
        lead = []
        if self.newest and r.random() < cfg["p_new_after"]:
            w = np.array([1.0 + lib.uses[i] for i in self.newest]); lead = [int(r.choice(self.newest, p=w / w.sum()))]
        S = lead + lib.pick(max(0, k - len(lead)), exclude=set(lead))
        for s in S: lib.uses[s] += 1
        a = r.choice([-1, 1], len(S)) * r.uniform(cfg["a_lo"], cfg["a_hi"], len(S))
        spec = dict(t=self.t, S=[int(s) for s in S], a=[float(x) for x in a], stage=self.stage, has_new=bool(set(S) & set(self.newest)))
        self.problems.append(spec); self.t += 1
        return build(lib, spec, self.seed, cfg)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conv = lambda v: v.tolist() if isinstance(v, np.ndarray) else v
        lib = [dict(op=c["op"], par=list(c["par"]), mu=c["mu"], sd=c["sd"], depth=c["depth"], p={k: conv(v) for k, v in c["p"].items()}) for c in self.lib.c]
        json.dump(dict(seed=self.seed, cfg=self.cfg, library=lib, problems=self.problems, events=self.events), open(path, "w"),
                  default=lambda o: o.item() if hasattr(o, "item") else o.tolist())


def build(lib, spec, seed, cfg):
    """One problem from its spec: points and noise from the problem's own seed, so it is identical whenever it is rebuilt."""
    g = np.random.default_rng([seed, POINT_STREAM, spec["t"]]); S, a = spec["S"], np.array(spec["a"])
    n = cfg["n_tr"] + cfg["n_va"] + cfg["n_te"] + cfg["ref_n"]
    x = g.standard_normal((n, cfg["dim"])); V = lib.values(x, upto=max(S) + 1)[:, S]
    y = V @ a; y = y + g.standard_normal(n) * cfg["noise"] * y.std()
    mu, sd = y[-cfg["ref_n"]:].mean(), y[-cfg["ref_n"]:].std(); y = (y - mu) / sd
    i1, i2, i3 = cfg["n_tr"], cfg["n_tr"] + cfg["n_va"], cfg["n_tr"] + cfg["n_va"] + cfg["n_te"]
    return dict(tr=(x[:i1], y[:i1]), va=(x[i1:i2], y[i1:i2]), te=(x[i2:i3], y[i2:i3]), S=list(S), Ctrue=(V[:i1], V[i1:i2], V[i2:i3]),
                stage=spec["stage"], has_new=spec["has_new"], t=spec["t"])


class _SavedLib(winf.Library):
    def __init__(self, concepts):
        self.c = [dict(op=c["op"], par=tuple(c["par"]), mu=c["mu"], sd=c["sd"], depth=c["depth"],
                       p={k: (np.array(v) if isinstance(v, list) else v) for k, v in c["p"].items()}) for c in concepts]


def load(path):
    W = json.load(open(path)); lib = _SavedLib(W["library"])
    data = [build(lib, spec, W["seed"], W["cfg"]) for spec in W["problems"]]
    return data, dict(seed=W["seed"], events=W["events"], concepts=len(W["library"]), problems=len(W["problems"]))


if __name__ == "__main__":
    import tempfile
    s = Stream(990001); live = []   # CONST: winf_cfg
    for t in range(50):   # CONST: winf_cfg
        if t in (20, 35): s.expand(12)                                 # CONST: winf_cfg
        live.append(s.next())
    p = os.path.join(tempfile.gettempdir(), "wstream_check.json"); s.save(p)
    back, info = load(p)
    diff = max(float(np.max(np.abs(a[k][j] - b[k][j]))) for a, b in zip(live, back) for k in ("tr", "va", "te") for j in (0, 1))
    print(f"save -> load max difference {diff:.1e}; {info['problems']} problems, {info['concepts']} concepts, events {info['events']}; file {os.path.getsize(p)} bytes")
