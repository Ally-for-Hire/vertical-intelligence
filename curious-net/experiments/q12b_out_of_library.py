"""Q12b: does the agent know when NONE of its hypotheses fits? Library (as in q12): free fall, constant speed, quadratic-drag model. Worlds:
  in-library : true law inside the library (as q12: g 1-25, quadratic drag with vt in inf/40/15/6)
  stokes     : LINEAR drag dv/dt = g - k v (k in 0.15, 0.4, 1.0 per s): not in the library
  varg       : gravity that changes along the fall, g(x) = g0 (R/(R + h - x))^2 with R in {30, 60} m (a small body: stronger pull near the ground): not in the library
The agent runs n_main drops at log-uniform random heights plus 6 replicate drops (3 heights, each dropped twice) from which it measures its pure-error noise level. After BIC model selection it
runs a lack-of-fit F test (relative residuals vs the replicate pure error, alpha = 0.01, an experiment setting). Flagged = 'no known law fits: need more research'. Reported: detection rate in
out-of-library worlds, false-alarm rate in in-library worlds, accuracy of verdicts on planted claims when the agent answers vs when it abstains, and what happens after ONE library expansion
(adding a free power law T = a h^b) for flagged worlds."""
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.integrate import solve_ivp
from scipy.stats import f as fdist
import q12_law_discovery as Q

rng = np.random.default_rng(0)


def T_stokes(h, g, k):
    f = lambda T: (g / k) * T - (g / k ** 2) * (1 - np.exp(-k * T)) - h
    hi = 1.0
    while f(hi) < 0:
        hi *= 2
    return brentq(f, 1e-6, hi)


def T_varg(h, g0, R):
    def rhs(t, y):
        x, v = y
        return [v, g0 * (R / (R + h - x)) ** 2]
    ev = lambda t, y: y[0] - h
    ev.terminal = True
    ev.direction = 1
    s = solve_ivp(rhs, [0, 60], [0.0, 0.0], events=ev, rtol=1e-8, atol=1e-10)
    return float(s.t_events[0][0])


def true_time(world, h):
    kind, a, b = world
    h = np.atleast_1d(h).astype(float)
    if kind == "inlib":
        return Q.true_T(h, a, b)
    if kind == "stokes":
        return np.array([T_stokes(x, a, b) for x in h])
    return np.array([T_varg(x, a, b) for x in h])


def noisy(world, h):
    T = true_time(world, h)
    return T * (1 + Q.NOISE_REL * rng.normal(size=T.shape)) + Q.NOISE_ABS * rng.normal(size=T.shape)


def run(world, n_main):
    hs = 10 ** rng.uniform(0, 2, n_main)
    Ts = noisy(world, hs)
    rep_h = 10 ** rng.uniform(0.3, 1.9, 3)
    t1, t2 = noisy(world, rep_h), noisy(world, rep_h)
    hs_all = np.concatenate([hs, rep_h, rep_h]); Ts_all = np.concatenate([Ts, t1, t2])
    fits = Q.fit_models(hs_all, Ts_all)
    n = len(hs_all)
    scores = {k: Q.bic(m[3], n, len(m[0])) for k, m in fits.items()}
    best = min(scores, key=scores.get)
    names, p, f, rss = fits[best]
    pred = f(hs_all, p)
    rel = (Ts_all - pred) / Ts_all
    sig_pe2 = np.mean((t1 - t2) ** 2 / (2 * ((t1 + t2) / 2) ** 2))
    dfm = max(n - len(p), 1)
    F = (np.sum(rel ** 2) / dfm) / max(sig_pe2, 1e-12)
    pval = 1 - fdist.cdf(F, dfm, 3)
    return best, p, f, hs_all, Ts_all, pval, np.sqrt(np.sum(rel ** 2) / dfm)


def verdict(f, p, sig_rel, h, claim):
    pred = float(f(np.array([h]), p)[0])
    tol = 3.0 * pred * np.sqrt(sig_rel ** 2 + Q.NOISE_REL ** 2) + 3 * Q.NOISE_ABS
    return "consistent" if abs(claim - pred) <= tol else "inconsistent"


def expand_and_check(hs, Ts):
    r = least_squares(lambda q: Ts - q[0] * hs ** q[1], [0.5, 0.5], bounds=([1e-3, 0.05], [50, 3.0]))
    pred = r.x[0] * hs ** r.x[1]
    return r.x, pred


if __name__ == "__main__":
    NW = 150
    inlib = [("inlib", float(np.exp(rng.uniform(0, np.log(25)))), [np.inf, 40.0, 15.0, 6.0][int(rng.integers(4))]) for _ in range(NW)]
    stokes = [("stokes", float(np.exp(rng.uniform(np.log(3), np.log(25)))), float(rng.choice([0.15, 0.4, 1.0]))) for _ in range(NW)]
    varg = [("varg", float(np.exp(rng.uniform(np.log(3), np.log(25)))), float(rng.choice([30.0, 60.0]))) for _ in range(NW)]
    print("n_main | flag rate: in-library (false alarm) / stokes / varg | verdict accuracy on planted claims: unflagged claims (answered) vs flagged claims if the agent had answered anyway")
    for n_main in (6, 10, 16, 24):
        res = {}
        for name, worlds in (("inlib", inlib), ("stokes", stokes), ("varg", varg)):
            flags, acc_unf, acc_fl = [], [], []
            for w in worlds:
                best, p, f, hs, Ts, pval, sig = run(w, n_main)
                flagged = pval < 0.01
                flags.append(flagged)
                for _ in range(6):
                    hc = float(rng.uniform(5, 60)); Tt = float(true_time(w, hc)[0])
                    if rng.random() < 0.5:
                        claim, truth = Tt * (1 + rng.uniform(-0.02, 0.02)), "consistent"
                    else:
                        claim, truth = Tt * float(rng.choice([0.4, 0.6, 1.7, 2.5])), "inconsistent"
                    ok = verdict(f, p, sig, hc, claim) == truth
                    (acc_fl if flagged else acc_unf).append(ok)
            res[name] = (np.mean(flags), np.mean(acc_unf) if acc_unf else float("nan"), np.mean(acc_fl) if acc_fl else float("nan"), len(acc_unf), len(acc_fl))
        print(f"{n_main:4d}   | {res['inlib'][0]:.2f} / {res['stokes'][0]:.2f} / {res['varg'][0]:.2f} | unflagged-acc in-lib {res['inlib'][1]:.3f}  stokes {res['stokes'][1]:.3f}  varg {res['varg'][1]:.3f} | flagged-acc (if answered) in-lib {res['inlib'][2]:.3f} stokes {res['stokes'][2]:.3f} varg {res['varg'][2]:.3f}")
    # one expansion for flagged out-of-library worlds, n_main = 16
    print("\nafter ONE library expansion (free power law T = a h^b) in flagged out-of-library worlds, n_main=16: lack-of-fit passes? median relative prediction error at 15/50/100 m, before vs after")
    for name, worlds in (("stokes", stokes), ("varg", varg)):
        passed, e_before, e_after = [], [], []
        for w in worlds:
            best, p, f, hs, Ts, pval, sig = run(w, 16)
            if pval >= 0.01:
                continue
            x, predn = expand_and_check(hs, Ts)
            rel = (Ts - predn) / Ts
            n = len(hs); sig_pe2 = np.var(rel) * 0 + (Q.NOISE_REL ** 2 + (Q.NOISE_ABS / np.maximum(Ts, 0.3)) ** 2).mean()
            F = (np.sum(rel ** 2) / max(n - 2, 1)) / sig_pe2
            passed.append(1 - fdist.cdf(F, max(n - 2, 1), 3) >= 0.01)
            for hq in (15.0, 50.0, 100.0):
                tq = float(true_time(w, hq)[0])
                e_before.append(abs(float(f(np.array([hq]), p)[0]) - tq) / tq)
                e_after.append(abs(x[0] * hq ** x[1] - tq) / tq)
        print(f"  {name:7s} flagged worlds {len(passed):3d}  passes lack-of-fit after expansion {np.mean(passed) if passed else float('nan'):.2f}   median rel err before {np.median(e_before):.3f} after {np.median(e_after):.3f}")
