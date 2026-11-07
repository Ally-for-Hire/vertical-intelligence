"""Runner for sequences of problems. Streams are generated once per (problem spec, seed) with random held actions, so every arm sees identical data.
Arms: 'life' (one LNet for the whole sequence), 'fresh' (a new LNet per problem: zero transfer), plus NN baselines in baselines2.py."""
import time
import numpy as np
import common  # noqa: F401
from constants import C
from encode2 import ProblemEncoder
from core2 import LNet
import problems as PB


def make_problem(spec, seed):
    kind, kw = spec
    return {"car": PB.CarProblem, "cart": PB.Cart1D, "pend": PB.Pendulum, "rbn": PB.RandomBoolNet}[kind](seed=seed, **kw)


def make_stream(spec, seed, T):
    P = make_problem(spec, seed)
    rng = np.random.default_rng(seed + C("seed_offset_warmup"))
    S = []
    for _ in range(C("warmup_ticks")):
        P.step(int(rng.integers(len(P.actions))))
        S.append(P.scalars())
    enc = ProblemEncoder(P)
    g0 = P.grid()
    enc.calibrate(S, g0)
    enc.encode(g0, P.scalars())
    X = np.zeros((T, len(enc.state_names) + len(P.act_names)), np.float32)
    Y = np.zeros((T, len(enc.state_names)), np.float32)
    x = enc.encode(P.grid(), P.scalars())
    a = 0
    hold = C("hold_ticks")
    sc = np.zeros(T)
    for t in range(T):
        if t % hold == 0:
            a = int(rng.integers(len(P.actions)))
        X[t] = np.concatenate([x, P.actions[a]])
        P.step(a)
        x = enc.encode(P.grid(), P.scalars())
        Y[t] = x
        sc[t] = P.score
    return dict(spec=spec, key=P.key, names=enc.state_names, act_names=P.act_names, X=X, Y=Y, truth=P.truth(), scalars=P.names,
                bits_per=2 * C("n_thresholds"), grid_n=enc.grid_n, score=sc)


def var_of_bit(names):
    return [n.split("#")[0] for n in names]


def run_life(streams, arm="life", seed=0, sleep_mode="self"):
    """streams: list of make_stream outputs. Returns per-problem block losses, trust traces, complexity trace."""
    net = None
    out = []
    for k, st in enumerate(streams):
        if net is None or arm == "fresh":
            net = LNet(seed)
        pid = net.add_problem(st["key"], st["names"], st["act_names"], len(st["X"]))
        net.start_problem(pid)
        sleeps = []
        T = len(st["X"])
        loss_t = np.zeros(T)
        trust_t = np.zeros(T)
        bit_loss = np.zeros(len(st["names"]))
        bit_trust = np.zeros(len(st["names"]))
        n_first = min(C("first_window"), T)
        for t in range(T):
            loss, trust = net.observe(pid, st["X"][t], st["Y"][t])
            loss_t[t] = loss.sum()
            trust_t[t] = trust.mean()
            if sleep_mode != "end" and net.should_sleep():
                sleeps.append((t, net.sleep()))
            if t < n_first:
                bit_loss += loss
                bit_trust += trust
        info = net.sleep()
        nv, nt = net.n_variants()
        out.append(dict(key=st["key"], loss=loss_t, trust=trust_t, bit_loss_first=bit_loss / n_first, bit_trust_first=bit_trust / n_first,
                        info=info, sleeps=len(sleeps), complexity=net.complexity(), variants=nv, targets=nt, edges=net.n_edges() if hasattr(net, "n_edges") else len(net.edges())))
    return out


def run_cont(streams, arm="life", seed=0):
    """continuous learner: same streams, no sleep"""
    from core4 import CNet
    net = None
    out = []
    for k, st in enumerate(streams):
        if net is None or arm == "fresh":
            net = CNet(seed)
        pid = net.add_problem(st["key"], st["names"], st["act_names"])
        T = len(st["X"])
        loss_t = np.zeros(T)
        trust_t = np.zeros(T)
        bit_loss = np.zeros(len(st["names"]))
        bit_trust = np.zeros(len(st["names"]))
        n_first = min(C("first_window"), T)
        for t in range(T):
            loss, trust = net.observe(pid, st["X"][t], st["Y"][t])
            loss_t[t] = loss.sum()
            trust_t[t] = trust.mean()
            if t < n_first:
                bit_loss += loss
                bit_trust += trust
        nv, nt = net.n_variants()
        out.append(dict(key=st["key"], loss=loss_t, trust=trust_t, bit_loss_first=bit_loss / n_first, bit_trust_first=bit_trust / n_first,
                        complexity=net.complexity(), variants=nv, targets=nt, edges=net.n_edges(), forks=net.forks, merges=net.merges, adds=net.adds, prunes=net.prunes))
    return out
