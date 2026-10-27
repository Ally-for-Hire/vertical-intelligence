"""X1.1 (= P-draft 9): which drive avoids the dark room and the noise? (the author's theorem U12 and open question Q9)

World D: four regions the agent can sit in each tick. Every visit shows one symbol and the agent first predicts it with its own table
(next symbol given the region's previous symbol), then learns the true one. Prediction error is 1 if wrong, 0 if right.
  DARK    always the same symbol: a perfectly predictable room (zero surprise after one visit), nothing to learn
  NOISE   a uniformly random symbol: unlearnable, high error forever
  LEARN1  a deterministic rule (a random single cycle over the 16 symbols): high error at first, learnable in about 16 visits
  LEARN2  the same kind of region with a different rule, which OPENS at the middle of the run (a second thing to grow into)
Arms differ ONLY in the intrinsic reward the region-choosing bandit receives (the same value estimate and the same epsilon-greedy
rule for all):
  UR      uncertainty reduction: reward = - prediction error (the drive the program began with; 'surprise minimisation')
  NOV     novelty / error-seeking: reward = + prediction error
  LP      learning progress: reward = slow average error - fast average error of the visited region (how fast the error is falling)
  ORACLE  sits in a learnable region until it is learned, then in the dark room (ceiling for learning)
  RANDOM  a uniformly random available region
Metrics: share of ticks in each region (first half / second half); accuracy of the learned tables of LEARN1 (at the middle) and LEARN2
(at the end); ticks after LEARN2 opens until its table is 95% accurate (capped at the half run).
Usage: python x1_1.py check | run          (writes ../results/x1_1_<mode>.json)"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x0_1 as X

ARMS = ("UR", "NOV", "LP", "ORACLE", "RANDOM", "LPO", "NOVO")      # LPO / NOVO = LP / NOV with an OPTIMISTIC start (X1.2); new arms are appended so the old arms keep their random streams
DARK, NOISE, L1, L2 = 0, 1, 2, 3
OUT = X.OUT


def cycle(rng, S):
    p = rng.permutation(S); f = np.zeros(S, int)
    for i in range(S):
        f[p[i]] = p[(i + 1) % S]
    return f


def run(seed, arm):
    S, T = C("d_symbols"), C("d_ticks"); half = int(T * C("d_open"))
    rw = np.random.default_rng(seed); f = {L1: cycle(rw, S), L2: cycle(rw, S)}
    ra = np.random.default_rng(seed + C("x02_stream_stride") * (ARMS.index(arm) + 1))
    table = {r: ra.integers(0, S, S) for r in range(4)}             # each arm's own initial guesses
    prev = {r: 0 for r in range(4)}; v = np.zeros(4); fast = np.full(4, 1 - 1 / S); slow = np.full(4, 1 - 1 / S)
    acc = lambda r: float((table[r] == f[r]).mean())
    opt = arm in ("LPO", "NOVO")
    if opt: v[[DARK, NOISE, L1]] = C("opt_value")                      # untried regions start at the largest reward the drive can pay
    visits = np.zeros((T, 4), bool); errs = np.zeros((T, 4)); acc1_half = 0.0; t_l2 = None; err_hist = {r: [] for r in range(4)}
    for t in range(T):
        if opt and t == half: v[L2] = C("opt_value")
        avail = [DARK, NOISE, L1] + ([L2] if t >= half else [])
        if arm == "RANDOM" or ra.random() < C("d_eps"):
            r = int(ra.choice(avail))
        elif arm == "ORACLE":
            r = L2 if (L2 in avail and acc(L2) < C("d_acc")) else L1 if acc(L1) < C("d_acc") else DARK
        else:
            r = max(avail, key=lambda q: v[q] + 1e-9 * ra.random())
        nxt = 0 if r == DARK else int(ra.integers(0, S)) if r == NOISE else int(f[r][prev[r]])
        err = float(table[r][prev[r]] != nxt); table[r][prev[r]] = nxt; prev[r] = nxt
        fast[r] += C("lp_fast") * (err - fast[r]); slow[r] += C("lp_slow") * (err - slow[r])
        rew = -err if arm == "UR" else err if arm in ("NOV", "NOVO") else (slow[r] - fast[r])
        v[r] += C("d_alpha") * (rew - v[r])
        visits[t, r] = True; err_hist[r].append(err)
        if t == half - 1: acc1_half = acc(L1)
        if t_l2 is None and t >= half and acc(L2) >= C("d_acc"): t_l2 = t - half
    share = lambda sl, r: float(visits[sl, r].mean())
    a, b = slice(0, half), slice(half, T)
    return dict(dark_a=share(a, DARK), noise_a=share(a, NOISE), l1_a=share(a, L1), dark_b=share(b, DARK), noise_b=share(b, NOISE), l1_b=share(b, L1), l2_b=share(b, L2),
                acc_l1_mid=acc1_half, acc_l2_end=acc(L2), t_l2=float(t_l2 if t_l2 is not None else T - half),
                err_noise=float(np.mean(err_hist[NOISE])) if err_hist[NOISE] else float("nan"), err_dark_late=float(np.mean(err_hist[DARK][1:])) if len(err_hist[DARK]) > 1 else float("nan"))


def summarize(per):
    g = lambda a, k: np.array([per[a][s][k] for s in sorted(per[a])], float)
    S = {}
    for k in ("dark_a", "noise_a", "l1_a", "dark_b", "noise_b", "l1_b", "l2_b", "acc_l1_mid", "acc_l2_end", "t_l2", "err_noise", "err_dark_late"):
        S[k] = {a: float(np.nanmedian(g(a, k))) for a in ARMS}
    for b in ("UR", "NOV"):
        S[f"t_l2 LP-{b}"] = X.boot_ci(g("LP", "t_l2") - g(b, "t_l2"), C("x11_seed0"))
    return S


def main(mode):
    if mode == "check":
        out = {a: run(C("x11_check_seed"), a) for a in ARMS}
    else:
        per = {a: {} for a in ARMS}
        for i in range(C("x01_worlds")):
            for a in ARMS: per[a][C("x11_seed0") + i] = run(C("x11_seed0") + i, a)
        out = {"per_world": per, "summary": summarize(per)}
    os.makedirs(OUT, exist_ok=True)
    json.dump({"mode": mode, "result": out}, open(os.path.join(OUT, f"x1_1_{mode}.json"), "w"), indent=1, default=float)
    print(json.dumps(out if mode == "check" else out["summary"], indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
