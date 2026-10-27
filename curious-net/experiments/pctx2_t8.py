"""PIECE-CTX2: theorem T8 as a FRESH prediction, plus 'verify when unsure'.
World: P6's 6 angular regions, but each region's rule = a shared base rule + delta x its own deviation; delta (rule CONTRAST) in {0.25, 0.5, 1.0}. 48 seeds each. Learner: 6 parts + learned router, trained with the
true region on every tick (q = 1, as CTX), so only TEST-time routing is imperfect.
T8 forecast, computed from quantities measured WITHOUT using the misrouted points: a = router accuracy; E_r = error of the true-region part; E_w = mean error of the OTHER parts over all test points
(not just the misrouted ones). Forecast total = a E_r + (1 - a) E_w, compared with the actual total error.
Verify when unsure: on the 10% of test points with the smallest top-two router margin, the learner ASKS (is given the true region).
PREDICTIONS (registered before running): P92 the T8 forecast is within +-30% of the actual total error in each of the 3 contrast levels (median over seeds); P93 the self/oracle error ratio rises with contrast delta;
P94 asking on the 10% least-confident test points cuts test error by >= 50% (at delta = 1)."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pp6_self as P6

NF, RLR, DELTAS = 64, 0.05, (0.25, 0.5, 1.0)


def world(rng, delta):
    a0, b0, c0, d0 = rng.normal(0, 2, 2), rng.normal(0, 0.5), rng.uniform(0.5, 1.5), rng.normal(0, 0.5)
    params = []
    for _ in range(P6.NR):
        params.append((a0 + delta * rng.normal(0, 2, 2), b0 + delta * rng.normal(0, 0.5), c0 + delta * rng.normal(0, 0.5), d0 + delta * rng.normal(0, 0.5)))
    region = lambda x: int(((np.arctan2(x[1], x[0]) + np.pi) / (2 * np.pi) * P6.NR)) % P6.NR
    def f(x):
        a, b, c, d = params[region(x)]
        return float(np.tanh(a @ x + b) * c + d)
    return region, f


def run(args):
    seed, delta = args
    from nn import AdamMLP
    rng = np.random.default_rng(seed); region, f = world(rng, delta)
    xs, ys = [], []
    for t in range(P6.T):
        x = P6.sample_x(rng, min(P6.NR, 1 + t // P6.UNLOCK), region); xs.append(x); ys.append(f(x) + 0.05 * rng.normal())
    test = [P6.sample_x(rng, P6.NR, region) for _ in range(2000)]; ty = np.array([f(x) for x in test]); tr = np.array([region(x) for x in test])
    frng = np.random.default_rng(seed + 4242); Wf = frng.normal(0, 3, (2, NF)); bf = frng.uniform(0, 2 * np.pi, NF)
    phi = lambda x: np.cos(x @ Wf + bf) * np.sqrt(2 / NF)
    parts = [AdamMLP(2, 1, P6.HP, P6.LR, seed * 100 + i) for i in range(P6.NR)]; R = np.zeros((P6.NR, NF))
    for t in range(P6.T):
        x, y = xs[t], ys[t]; ph = phi(x); k = region(x)
        lg = R @ ph; pr_ = np.exp(lg - lg.max()); pr_ /= pr_.sum()
        parts[k].step(x, np.array([y])); tg = np.zeros(P6.NR); tg[k] = 1.0; R += RLR * np.outer(tg - pr_, ph)
    Pall = np.array([[p.predict(x)[0][0] for p in parts] for x in test])          # (n_test, parts)
    E = (Pall - ty[:, None]) ** 2
    L = np.array([R @ phi(x) for x in test]); route = L.argmax(1)
    srt = np.sort(L, 1); margin = srt[:, -1] - srt[:, -2]
    actual = float(E[np.arange(len(test)), route].mean())
    a = float(np.mean(route == tr)); Er = float(E[np.arange(len(test)), tr].mean())
    mask = np.ones_like(E, bool); mask[np.arange(len(test)), tr] = False; Ew = float(E[mask].mean())
    forecast = a * Er + (1 - a) * Ew
    ask = margin <= np.quantile(margin, 0.10); r2 = route.copy(); r2[ask] = tr[ask]
    asked = float(E[np.arange(len(test)), r2].mean())
    return dict(delta=delta, actual=actual, forecast=forecast, a=a, Er=Er, Ew=Ew, oracle=Er, asked=asked)


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, [(s, d) for d in DELTAS for s in range(48)])
    json.dump(res, open("../results/pctx2_t8.json", "w"))
    print("contrast | router acc | E_right | E_wrong (all points) | actual error | T8 forecast | forecast/actual (median) | self/oracle (median) | with asking on 10% least confident: error, ratio to no-ask")
    for d in DELTAS:
        r = [x for x in res if x["delta"] == d]
        print(f"  {d:4.2f}  | {np.median([x['a'] for x in r]):.3f} | {np.median([x['Er'] for x in r]):.4f} | {np.median([x['Ew'] for x in r]):.3f} | {np.median([x['actual'] for x in r]):.4f} | {np.median([x['forecast'] for x in r]):.4f} | "
              f"{np.median([x['forecast'] / x['actual'] for x in r]):.2f} | {np.median([x['actual'] / x['oracle'] for x in r]):.1f} | {np.median([x['asked'] for x in r]):.4f}, {np.median([x['asked'] / x['actual'] for x in r]):.2f}")
    print("done", round(time.time() - t0), "s")
