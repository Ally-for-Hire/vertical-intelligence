"""R6 + X1: the student runs its OWN experiments (drop heights) in the ball-drop law worlds, with PERFECT measurement by default (user, 2026-10-22) and a noisy-measurement robustness arm.
Fixes from the review: (a) the 'chosen' design starts from the SAME first two random drops as the random design (per-world seeded), so differences come only from the choice rule; (b) the agent never reads the
generator's noise level: it estimates it (noise-free arm: from fit residuals, floored at the numerical tolerance REL_TOL; noisy arm: from replicate drops).
Worlds: in-library (free fall / quadratic drag, g 1-25, vt inf/40/15/6), Stokes linear drag (k 0.15/0.4/1.0), changing gravity (R 30/60 m) [out of library]. Library: free fall, constant speed, quadratic drag.
Arms: random heights, grid, CHOSEN (height where the fitted hypotheses disagree most, relative). Budget n drops. Outputs: right family (in-library), 'no law fits' flag rate (out-of-library) and false flags,
claim verdict accuracy by claim class (exact truth; off by 1%; off by 10%; off by x0.4-2.5), with abstention on flagged worlds counted separately.
REL_TOL = 1e-6 (NUMERICAL: well above the ODE/root-finder precision ~1e-8 used to produce 'exact' times, far below any physically interesting deviation)."""
import os, sys, json, time
sys.path.insert(0, ".")
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize_scalar, least_squares
from scipy.stats import f as fdist
import q12_law_discovery as Q
import q12b_out_of_library as B

REL_TOL = 1e-6
ALPHA = 0.01                      # lack-of-fit test level in the noisy arm (EXPERIMENT setting, as in q12b)
NOISE = 0.03                      # noisy-arm relative measurement noise (robustness arm only; never given to the agent)
NS = (3, 5, 8, 12, 20)
GRID = np.array([5, 10, 20, 40, 80, 2.5, 15, 30, 60, 100, 7, 12, 25, 50, 90, 3, 8, 17, 35, 70])
CLAIMS = (("exact", 1.0), ("off_1pct", None), ("off_10pct", None), ("off_large", None))


def fit_all(h, T):
    out = {}
    r = minimize_scalar(lambda g: np.sum((T / np.sqrt(2 * h / g) - 1) ** 2), bounds=(0.3, 60), method="bounded", options=dict(xatol=1e-12))
    out["free_fall"] = (np.array([r.x]), lambda hh, p: np.sqrt(2 * hh / p[0]))
    r2 = minimize_scalar(lambda v: np.sum((T / (h / v) - 1) ** 2), bounds=(0.3, 200), method="bounded", options=dict(xatol=1e-12))
    out["const_speed"] = (np.array([r2.x]), lambda hh, p: hh / p[0])
    best = None
    for g0, v0 in ((9.8, 30.0), (3.0, 10.0), (20.0, 60.0), (9.8, 300.0)):
        try:
            f = least_squares(lambda p: Q.true_T(h, p[0], p[1]) / T - 1, [g0, v0], bounds=([0.3, 2.0], [60.0, 1e4]), xtol=1e-14, ftol=1e-14, gtol=1e-14)
            if best is None or f.cost < best.cost:
                best = f
        except Exception:
            pass
    out["drag"] = (best.x, lambda hh, p: Q.true_T(hh, p[0], p[1]))
    res = {}
    for k, (p, fn) in out.items():
        rel = fn(h, p) / T - 1
        res[k] = (p, fn, float(np.sum(rel ** 2)), len(p))
    return res


def select(fits, n, floor):
    """BIC on relative residuals; residual sums below the floor count as the floor (residuals below the numerical tolerance are zero)"""
    sc = {k: n * np.log(max(v[2], floor) / n) + v[3] * np.log(n) for k, v in fits.items()}
    return min(sc, key=sc.get)


def choose(strategy, hs, Ts, rng0, step):
    if strategy == "grid":
        return float(GRID[step % len(GRID)])
    if strategy == "random" or len(hs) < 2:
        return float(10 ** rng0.uniform(0, 2))                       # chosen arm shares the same first two random drops
    fits = fit_all(np.array(hs), np.array(Ts))
    P = np.array([v[1](Q.HEIGHTS, v[0]) for v in fits.values()])
    return float(Q.HEIGHTS[int(np.argmax(P.std(0) / P.mean(0)))])


def measure(world, h, rng, noisy):
    T = float(B.true_time(world, h)[0])
    return T * (1 + NOISE * rng.normal()) if noisy else T


def run_world(args):
    wid, world, noisy = args
    out = []
    for strategy in ("random", "grid", "chosen"):
        for n in NS:
            rng0 = np.random.default_rng(wid)                          # identical first random heights across arms
            rngm = np.random.default_rng(10 ** 6 + wid)                # measurement noise stream
            hs, Ts = [], []
            for step in range(n if not noisy else n // 2 + n % 2):
                h = choose(strategy, hs, Ts, rng0, step)
                reps = 2 if noisy else 1                                   # noisy arm: each height dropped twice, budget counts drops
                for _ in range(reps):
                    hs.append(h); Ts.append(measure(world, h, rngm, noisy))
            h, T = np.array(hs[:max(n, 3)]), np.array(Ts[:max(n, 3)])
            nn = len(h)
            fits = fit_all(h, T)
            floor = nn * REL_TOL ** 2
            best = select(fits, nn, floor)
            p, fn, rss, k = fits[best]
            if not noisy:
                sig = max(np.sqrt(rss / max(nn - k, 1)), REL_TOL)
                flag = bool(np.sqrt(rss / nn) > 10 * REL_TOL) if nn > k else False     # exact data: any residual beyond numerical tolerance = the law does not fit
            else:
                uh = np.unique(h)
                pe = sum(np.sum((T[h == x] / T[h == x].mean() - 1) ** 2) for x in uh); dfp = nn - len(uh)
                sig = np.sqrt(pe / max(dfp, 1)) if dfp > 0 else np.sqrt(rss / max(nn - k, 1))
                lof = rss - pe; dfl = len(uh) - k
                flag = bool(dfl > 0 and dfp > 0 and pe > 0 and fdist.sf((lof / dfl) / (pe / dfp), dfl, dfp) < ALPHA)
            crng = np.random.default_rng(5 * 10 ** 6 + wid)
            verd = {c: [] for c, _ in CLAIMS}
            for c, _ in CLAIMS:
                for _ in range(10):
                    hc = float(crng.uniform(5, 60)); Tt = float(B.true_time(world, hc)[0])
                    if c == "exact":
                        claim, truth = Tt, "consistent"
                    elif c == "off_1pct":
                        claim, truth = Tt * (1 + crng.choice([-1, 1]) * 0.01), "inconsistent"
                    elif c == "off_10pct":
                        claim, truth = Tt * (1 + crng.choice([-1, 1]) * 0.10), "inconsistent"
                    else:
                        claim, truth = Tt * float(crng.choice([0.4, 0.6, 1.7, 2.5])), "inconsistent"
                    if flag:
                        verd[c].append("abstain"); continue
                    pred = float(fn(np.array([hc]), p)[0])
                    v = "consistent" if abs(claim / pred - 1) <= 3 * sig else "inconsistent"
                    verd[c].append("right" if v == truth else "wrong")
            out.append(dict(wid=wid, kind=world[0], strategy=strategy, n=n, best=best, flag=flag, sig=float(sig),
                            truth_ff=bool(world[0] == "inlib" and not np.isfinite(world[2])), verd=verd))
    return out


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(1)
    worlds = []
    for i in range(60):
        worlds.append(("inlib", float(np.exp(rng.uniform(0, np.log(25)))), [np.inf, 40.0, 15.0, 6.0][i % 4]))
    for i in range(30):
        worlds.append(("stokes", float(np.exp(rng.uniform(0, np.log(25)))), [0.15, 0.4, 1.0][i % 3]))
    for i in range(30):
        worlds.append(("varg", float(np.exp(rng.uniform(0, np.log(25)))), [30.0, 60.0][i % 2]))
    with Pool(20) as p:
        res = {}
        for noisy in (False, True):
            r = p.map(run_world, [(i, w, noisy) for i, w in enumerate(worlds)])
            res["noisy" if noisy else "exact"] = [x for y in r for x in y]
    json.dump(res, open("../results/r6_x1.json", "w"))
    for arm in ("exact", "noisy"):
        R = res[arm]
        print(f"\n=== measurement: {arm} ({'perfect instruments' if arm == 'exact' else '3% noise, replicate drops, noise estimated by the agent'}) ===")
        print("strategy  n | free fall identified (vt=inf) | drag worlds: drag picked | false flag (in-lib) | flag Stokes | flag varg | claims right: exact / 1% / 10% / large (answered worlds) | abstain (out-of-lib)")
        for s in ("random", "grid", "chosen"):
            for n in NS:
                C = [x for x in R if x["strategy"] == s and x["n"] == n]
                ff = np.mean([x["best"] == "free_fall" for x in C if x["truth_ff"]])
                dr = np.mean([x["best"] == "drag" for x in C if x["kind"] == "inlib" and not x["truth_ff"]])
                fa = np.mean([x["flag"] for x in C if x["kind"] == "inlib"])
                fs = np.mean([x["flag"] for x in C if x["kind"] == "stokes"]); fv = np.mean([x["flag"] for x in C if x["kind"] == "varg"])
                acc = []
                for c, _ in CLAIMS:
                    v = [e for x in C for e in x["verd"][c] if e != "abstain"]
                    acc.append(np.mean([e == "right" for e in v]) if v else float("nan"))
                ab = np.mean([e == "abstain" for x in C if x["kind"] != "inlib" for c, _ in CLAIMS for e in x["verd"][c]])
                print(f"{s:7s} {n:3d} |   {ff:.2f}   |   {dr:.2f}   |   {fa:.2f}   |   {fs:.2f}   |   {fv:.2f}   |   " + " / ".join(f"{a:.2f}" for a in acc) + f"   |   {ab:.2f}")
    print("done", round(time.time() - t0), "s")
