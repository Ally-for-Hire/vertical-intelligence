"""X1.47: the LINEAR base rung of the X1.6 complexity ladder (author, 2026-11-04).

Author: "each output is a linear function, even underlying it should be linear, and just see if the neural network can map the entire thing
before increasing in complexity". X1.6 started its ladder at rung 1 (one bend) and never checked that the learner could fully learn the rung
below it. This test is that rung 0, and a GATE: the ladder should only go up for a learner that passes it.

World: x ~ N(0, I_6). A task has 6 outputs, y = A x, with A = L_d ... L_1 a stack of d random LINEAR layers (d = 1..5): the hidden rule is linear
at every layer, so every output is a linear function of x at every depth. Each output is scaled to variance 1. No noise: the whole rule can be
learned exactly from the 100 training points (least squares does it; that is the fair-test check).
'Map the entire thing' is scored two ways on unseen points: IN RANGE (x drawn like the training points) and OUT OF RANGE (x with sd 3). A learner
that holds the linear rule, not just the cloud of training points, must pass both. A task is MASTERED when every output has R^2 >= x47_gate.
Arms (all use the X1.6 learner: 2 hidden tanh layers of 32 from the one shared starting point, 6 outputs):
  FRESH10   the X1.6 regime with no meta-training: 10 plain gradient steps
  META10    the X1.6 regime: Meta-SGD meta-trained on error after 10 steps (MIX objective) on linear tasks of depth 1-3
  CONV      the same tanh network trained to convergence (full-batch Adam, x47_conv_steps)
  LINNET    the same network with the tanh removed (identity units), trained the same way: separates 'the activation' from 'the optimizer'
  LSQ       least squares (reference ceiling; not a network)
Usage: python x1_47.py selftest | run      (writes ../results/x1_47_<mode>.json)"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C

OUT = os.path.join(str(vipaths.RESULTS))
ARMS = ("FRESH10", "META10", "CONV", "LINNET", "LSQ")
LN2 = math.log(2.0)


# ---------------------------------------------------------------- world -------------------------------------------------------------
def make_task(rng, depth):
    """Returns xs, ys (train), xq, yq (unseen, in range), xo, yo (unseen, out of range). Every output is linear in x."""
    m, k = C("x16_dim"), C("x47_outputs")
    A = np.eye(m)
    for i in range(depth):
        rows = k if i == depth - 1 else m
        A = (rng.standard_normal((rows, m)) / math.sqrt(m)) @ A                       # one more LINEAR layer of the hidden rule
    A = A / np.linalg.norm(A, axis=1, keepdims=True)                                 # each output has variance 1 for x ~ N(0, I)
    xs = rng.standard_normal((C("x16_n_sup"), m))
    xq = rng.standard_normal((C("x16_n_qry"), m))
    xo = C("x47_out_scale") * rng.standard_normal((C("x16_n_qry"), m))
    f = lambda a: a.astype(np.float32)
    return tuple(f(a) for a in (xs, xs @ A.T, xq, xq @ A.T, xo, xo @ A.T))


def batch(rng, depths):
    T = [make_task(rng, d) for d in depths]
    return [np.stack([t[i] for t in T]) for i in range(6)]


# ---------------------------------------------------------------- learner -----------------------------------------------------------
def init_params(torch):
    """The ONE shared starting point (same seed and hidden size as X1.6; the output layer has 6 units instead of 1)."""
    g = torch.Generator().manual_seed(C("x16_init_seed"))
    m, h, k = C("x16_dim"), C("x16_hidden"), C("x47_outputs")
    shapes = [(m, h), (h,), (h, h), (h,), (h, k), (k,)]
    P = []
    for s in shapes:
        w = torch.randn(*s, generator=g) / math.sqrt(s[0]) if len(s) == 2 else torch.zeros(*s)
        P.append(w.requires_grad_())
    L = [torch.full(s, math.log(C("x16_lr0"))).requires_grad_() for s in shapes]
    return P, L


def forward(torch, W, x, act):
    """W: per-task weights [T, ...]; x [T, n, m] -> [T, n, k]."""
    a = act(torch.bmm(x, W[0]) + W[1][:, None, :])
    a = act(torch.bmm(a, W[2]) + W[3][:, None, :])
    return torch.bmm(a, W[4]) + W[5][:, None, :]   # CONST: x16_hidden  (index 4,5 = output layer of the 6-tensor list)


def mse(torch, W, x, y, act):
    return ((forward(torch, W, x, act) - y) ** 2).mean((1, 2))


def adapt(torch, P, L, xs, ys, xq, yq, steps, second_order):
    """Inner loop of the X1.6 learner: full-batch SGD with per-weight step sizes. Returns (final per-task weights, query MSE after the last step)."""
    T = xs.shape[0]
    W = [p.unsqueeze(0).expand(T, *p.shape) for p in P]
    lr = [l.exp().unsqueeze(0) for l in L]
    for _ in range(steps):
        G = torch.autograd.grad(mse(torch, W, xs, ys, torch.tanh).sum(), W, create_graph=second_order)
        W = [w - r * gr for w, r, gr in zip(W, lr, G)]
    return W, mse(torch, W, xq, yq, torch.tanh)


def meta_train(torch, seed):
    """META10: the X1.6 MIX objective (query error after K steps), linear tasks of depth 1-3, same outer budget as X1.6."""
    P, L = init_params(torch)
    opt = torch.optim.Adam(P + L, lr=C("lr_default"))
    rng = np.random.default_rng([seed, 1])
    for _ in range(C("x16_iters")):
        d = list(rng.choice(C("x47_train_depths"), C("x16_meta_batch")))
        xs, ys, xq, yq, _, _ = (torch.from_numpy(a) for a in batch(rng, d))
        _, q = adapt(torch, P, L, xs, ys, xq, yq, C("x16_K"), True)
        loss = q.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for l in L: l.clamp_(math.log(C("x16_lr_min")), math.log(C("x16_lr_max")))
    return P, L


def train_conv(torch, xs, ys, act):
    """Train the shared starting point to convergence on each task separately (Adam is per-element, so batching tasks keeps them independent)."""
    P, _ = init_params(torch)
    T = xs.shape[0]
    W = [p.detach().unsqueeze(0).repeat(T, *([1] * p.dim())).requires_grad_() for p in P]
    opt = torch.optim.Adam(W, lr=C("x47_conv_lr"))
    for _ in range(C("x47_conv_steps")):
        loss = mse(torch, W, xs, ys, act).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return [w.detach() for w in W]


# ---------------------------------------------------------------- scoring -----------------------------------------------------------
def r2(pred, y):
    """[T, n, k] -> [T, k]: R^2 of every output of every task on these points."""
    return 1 - ((pred - y) ** 2).mean(1) / y.var(1)


def score(pin, yq, pout, yo, depths):
    ri, ro = r2(pin, yq), r2(pout, yo)
    gate = C("x47_gate")
    bits = 0.5 * np.log2(1 / np.clip(1 - ri, 1e-6, None))
    res = {"share_in": float(np.mean(ri.min(1) >= gate)), "share_out": float(np.mean(ro.min(1) >= gate)),
           "bits_in": float(bits.mean()), "r2_in_min_med": float(np.median(ri.min(1))), "r2_out_min_med": float(np.median(ro.min(1)))}
    for d in C("x47_depths"):
        sel = np.array(depths) == d
        res[f"share_in_d{d}"] = float(np.mean(ri[sel].min(1) >= gate)); res[f"share_out_d{d}"] = float(np.mean(ro[sel].min(1) >= gate))
    return res


def one(seed):
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    rng = np.random.default_rng([seed, C("x16_test_stream")])
    depths = [d for d in C("x47_depths") for _ in range(C("x47_test_tasks"))]
    xs, ys, xq, yq, xo, yo = batch(rng, depths)
    tt = lambda a: torch.from_numpy(a)
    out = {}
    ident = lambda z: z

    def regime(P, L):
        W, _ = adapt(torch, [p.detach().requires_grad_() for p in P], [l.detach() for l in L], tt(xs), tt(ys), tt(xq), tt(yq), C("x16_K"), False)
        with torch.no_grad():
            return forward(torch, W, tt(xq), torch.tanh).numpy(), forward(torch, W, tt(xo), torch.tanh).numpy()

    out["FRESH10"] = score(*_pair(regime(*init_params(torch)), yq, yo), depths)
    out["META10"] = score(*_pair(regime(*meta_train(torch, seed)), yq, yo), depths)
    for arm, act in (("CONV", torch.tanh), ("LINNET", ident)):
        W = train_conv(torch, tt(xs), tt(ys), act)
        with torch.no_grad():
            out[arm] = score(*_pair((forward(torch, W, tt(xq), act).numpy(), forward(torch, W, tt(xo), act).numpy()), yq, yo), depths)
    A = [np.linalg.lstsq(xs[i], ys[i], rcond=None)[0] for i in range(len(depths))]
    out["LSQ"] = score(*_pair((np.stack([xq[i] @ A[i] for i in range(len(A))]), np.stack([xo[i] @ A[i] for i in range(len(A))])), yq, yo), depths)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _pair(preds, yq, yo):
    return preds[0], yq, preds[1], yo


def summarize(per, seed0):
    from x0_1 import boot_ci
    seeds = sorted(per)
    M = {a: {k: np.array([per[s][a][k] for s in seeds]) for k in per[seeds[0]][a]} for a in ARMS}
    S = {"median": {a: {k: float(np.median(v)) for k, v in M[a].items()} for a in ARMS},
         "min_share": {a: {"in": float(M[a]["share_in"].min()), "out": float(M[a]["share_out"].min())} for a in ARMS},
         "paired": {"bits_in META10-FRESH10": boot_ci(M["META10"]["bits_in"] - M["FRESH10"]["bits_in"], seed0),
                    "share_in d5-d1 CONV": boot_ci(M["CONV"][f"share_in_d{C('x47_depths')[-1]}"] - M["CONV"]["share_in_d1"], seed0)}}
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run."""
    m, pr, hi, lo = S["median"], S["paired"], C("x47_share_hi"), C("x47_share_lo")
    f = lambda k: f"{pr[k][0]:+.3f} [{pr[k][1]:+.3f}, {pr[k][2]:+.3f}]"
    manip = S["min_share"]["LSQ"]["in"] == 1 and S["min_share"]["LSQ"]["out"] == 1 and m["FRESH10"]["bits_in"] > 0
    V = [("M-X1.47", "Fair test: the world is exactly linear and learnable from the training points (least squares masters every task in and out of range in every world) and the untrained X1.6 learner holds some structure after 10 steps",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"LSQ min share in/out {S['min_share']['LSQ']['in']:.2f}/{S['min_share']['LSQ']['out']:.2f}; FRESH10 bits {m['FRESH10']['bits_in']:.3f}")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P563", "The tanh network trained to convergence maps the linear world IN RANGE (median share of tasks mastered >= hi)", m["CONV"]["share_in"] >= hi, f"CONV share in {m['CONV']['share_in']:.2f}")
    v("P564", "It does NOT map it OUT OF RANGE (sd 3 inputs): median share mastered < lo (it fits the cloud, not the line)", m["CONV"]["share_out"] < lo, f"CONV share out {m['CONV']['share_out']:.2f}")
    v("P565", "The same network without tanh maps it in AND out of range (both medians >= hi): the gap is the activation, not the optimizer", m["LINNET"]["share_in"] >= hi and m["LINNET"]["share_out"] >= hi, f"LINNET in {m['LINNET']['share_in']:.2f} out {m['LINNET']['share_out']:.2f}")
    v("P566", "The X1.6 regime (10 steps) does not map even rung 0 (META10 median share in < lo), but meta-training helps (META10 - FRESH10 bits held, paired range above 0)",
      m["META10"]["share_in"] < lo and pr["bits_in META10-FRESH10"][1] > 0, f"META10 share in {m['META10']['share_in']:.2f}; bits {f('bits_in META10-FRESH10')}")
    v("P567", "Depth of the hidden LINEAR rule does not matter to the converged tanh network: share mastered in range at depth 5 minus depth 1, median within tol",
      abs(pr["share_in d5-d1 CONV"][0]) <= C("x47_depth_tol"), f("share_in d5-d1 CONV"))
    return V


def main(mode):
    import budget
    t0 = time.time()
    seed0 = C("x47_selftest_seed") if mode == "selftest" else C("x47_seed0")
    nw = C("x47_selftest_worlds") if mode == "selftest" else C("x47_worlds")
    res = budget.pmap(one, [(seed0 + i,) for i in range(nw)])
    per = {seed0 + i: r for i, r in enumerate(res)}
    S = summarize(per, seed0)
    out = {"per_world": per, "summary": S, "verdicts": judge(S)}
    os.makedirs(OUT, exist_ok=True)
    json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, f"x1_47_{mode}.json"), "w"), indent=1, default=float)
    keys = ("share_in", "share_out", "bits_in", "r2_in_min_med", "r2_out_min_med")
    print("arm      " + "  ".join(f"{k:>14}" for k in keys))
    for a in ARMS:
        print(f"{a:8} " + "  ".join(f"{S['median'][a][k]:14.3f}" for k in keys))
    print("per depth, share mastered in/out (median):")
    for a in ARMS:
        print(f"{a:8} " + "  ".join(f"d{d} {S['median'][a][f'share_in_d{d}']:.2f}/{S['median'][a][f'share_out_d{d}']:.2f}" for d in C("x47_depths")))
    for v in out["verdicts"]: print(v[0], v[2], "|", v[3])
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1])
