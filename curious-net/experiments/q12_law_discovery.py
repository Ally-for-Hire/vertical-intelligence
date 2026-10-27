"""Q12: law discovery and claim checking from the agent's own drop experiments (the user's ball-drop example).
World: a ball dropped from height h takes time T(h) to fall. Truth family: quadratic drag, T = (vt/g) arccosh(exp(g h / vt^2)); vt = inf gives free fall T = sqrt(2h/g). g in [1, 25] m/s^2
(other planets), vt in {inf, 40, 15, 6} m/s. Measurement: T * (1 + 0.03 e1) + 0.05 e2 (normal e).
Agent hypotheses: H1 free fall (1 parameter g), H2 constant speed T = h/v (1 parameter v), H3 drag model (g, vt). It runs n experiments choosing the height by a strategy:
random (log-uniform 1-100 m), grid (fixed ladder), curious (the height where the three currently fitted hypotheses disagree most). Model chosen by BIC.
Claim check at the end: planted claims (height, claimed time), half true (within 2%), half false (time x 0.4, 0.6, 1.7 or 2.5); verdict consistent if |claim - prediction| <= 3 sigma (sigma = residual noise scale),
else inconsistent; 'no connection / need more research' if fewer than 3 experiments or the best model's residuals are not noise-like. Metrics: verdict accuracy, prediction error at 15/50/100 m,
whether the user's example claim (15 m in 5 s) is rejected on an Earth-like world."""
import numpy as np
from scipy.optimize import minimize_scalar, least_squares

HEIGHTS = np.logspace(0, 2, 60)
NOISE_REL, NOISE_ABS = 0.03, 0.05


def true_T(h, g, vt):
    h = np.asarray(h, float)
    if not np.isfinite(vt):
        return np.sqrt(2 * h / g)
    return (vt / g) * np.arccosh(np.exp(np.minimum(g * h / vt ** 2, 600.0)))


def measure(h, g, vt, rng):
    T = true_T(h, g, vt)
    return T * (1 + NOISE_REL * rng.normal(size=np.shape(T))) + NOISE_ABS * rng.normal(size=np.shape(T))


def fit_models(h, T):
    out = {}
    r = minimize_scalar(lambda g: np.sum((T - np.sqrt(2 * h / g)) ** 2), bounds=(0.3, 60), method="bounded")
    out["free_fall"] = (("g",), np.array([r.x]), lambda hh, p: np.sqrt(2 * hh / p[0]), np.sum((T - np.sqrt(2 * h / r.x)) ** 2))
    r2 = minimize_scalar(lambda v: np.sum((T - h / v) ** 2), bounds=(0.3, 200), method="bounded")
    out["const_speed"] = (("v",), np.array([r2.x]), lambda hh, p: hh / p[0], np.sum((T - h / r2.x) ** 2))
    best = None
    for g0, v0 in ((9.8, 30.0), (3.0, 10.0), (20.0, 60.0)):
        try:
            f = least_squares(lambda p: T - true_T(h, p[0], p[1]), [g0, v0], bounds=([0.3, 2.0], [60.0, 400.0]))
            if best is None or f.cost < best.cost:
                best = f
        except Exception:
            pass
    if best is not None:
        out["drag"] = (("g", "vt"), best.x, lambda hh, p: true_T(hh, p[0], p[1]), 2 * best.cost)
    return out


def bic(rss, n, k):
    return n * np.log(max(rss, 1e-12) / n) + k * np.log(n)


def choose_height(strategy, hs, Ts, rng, step):
    if strategy == "random":
        return float(10 ** rng.uniform(0, 2))
    if strategy == "grid":
        return float(np.array([5, 10, 20, 40, 80, 2.5, 15, 30, 60, 100, 7, 12, 25, 50, 90, 3, 8, 17, 35, 70, 1.5, 4, 6, 9, 14, 22, 45, 65, 85, 95])[step % 30])
    if len(hs) < 3:
        return float(10 ** rng.uniform(0, 2)) if len(hs) % 2 == 0 else float(10 ** rng.uniform(1, 2))
    fits = fit_models(np.array(hs), np.array(Ts))
    preds = np.array([m[2](HEIGHTS, m[1]) for m in fits.values()])
    sd = preds.std(0) / (np.abs(preds.mean(0)) + 1e-9)          # relative disagreement between hypotheses
    return float(HEIGHTS[int(np.argmax(sd))])


def run_agent(g, vt, strategy, n, rng):
    hs, Ts = [], []
    for step in range(n):
        h = choose_height(strategy, hs, Ts, rng, step)
        hs.append(h)
        Ts.append(float(measure(h, g, vt, rng)))
    hs, Ts = np.array(hs), np.array(Ts)
    fits = fit_models(hs, Ts)
    scores = {k: bic(m[3], n, len(m[0])) for k, m in fits.items()}
    best = min(scores, key=scores.get)
    names, p, f, rss = fits[best]
    sigma = np.sqrt(rss / max(n - len(p), 1))
    return best, p, f, sigma, hs, Ts


def verdict(n, best_f, p, sigma, h, claim):
    if n < 3:
        return "need more research"
    pred = float(best_f(np.array([h]), p)[0])
    tol = 3.0 * np.sqrt(sigma ** 2 + (NOISE_REL * pred) ** 2 + NOISE_ABS ** 2)
    return "consistent" if abs(claim - pred) <= tol else "inconsistent"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    NW = 200
    worlds = [(float(np.exp(rng.uniform(np.log(1.0), np.log(25.0)))), [np.inf, 40.0, 15.0, 6.0][int(rng.integers(4))]) for _ in range(NW)]
    print(f"{NW} random worlds (g from 1 to 25 m/s^2; vt in inf/40/15/6 m/s); per cell: verdict accuracy on planted claims | median relative error of the predicted fall time at 15/50/100 m | right model family chosen (free fall when vt=inf)")
    print("strategy   n   verdict-acc   median-rel-err   free-fall-identified(vt=inf worlds)")
    for strategy in ("random", "grid", "curious"):
        for n in (3, 5, 8, 12, 20):
            acc, errs, ff = [], [], []
            for (g, vt) in worlds:
                best, p, f, sigma, hs, Ts = run_agent(g, vt, strategy, n, rng)
                for _ in range(8):                                        # planted claims
                    hc = float(rng.uniform(5, 60)); Tt = float(true_T(hc, g, vt))
                    if rng.random() < 0.5:
                        claim, truth = Tt * (1 + rng.uniform(-0.02, 0.02)), "consistent"
                    else:
                        claim, truth = Tt * float(rng.choice([0.4, 0.6, 1.7, 2.5])), "inconsistent"
                    v = verdict(n, f, p, sigma, hc, claim)
                    acc.append(v == truth)
                for hq in (15.0, 50.0, 100.0):
                    errs.append(abs(float(f(np.array([hq]), p)[0]) - float(true_T(hq, g, vt))) / float(true_T(hq, g, vt)))
                if not np.isfinite(vt):
                    ff.append(best == "free_fall")
            print(f"{strategy:8s} {n:3d}   {np.mean(acc):.3f}         {np.median(errs):.3f}            {np.mean(ff):.2f}")
    # the user's example on an Earth-like world
    g, vt = 9.8, np.inf
    rej, nres = 0, 0
    for _ in range(300):
        best, p, f, sigma, hs, Ts = run_agent(g, vt, "curious", 8, rng)
        v = verdict(8, f, p, sigma, 15.0, 5.0)
        rej += (v == "inconsistent"); nres += 1
    print(f"\nuser's example (15 m in 5 s) on an Earth-like world (g=9.8, no drag), curious agent with 8 drops: rejected as inconsistent in {rej}/{nres} runs; true fall time {true_T(15.0, 9.8, np.inf):.2f} s")
    # what would make the claim consistent: required g for free fall
    print("free fall with g such that 15 m takes 5 s: g = %.2f m/s^2 (the Moon is about 1.62)" % (2 * 15.0 / 5.0 ** 2))
