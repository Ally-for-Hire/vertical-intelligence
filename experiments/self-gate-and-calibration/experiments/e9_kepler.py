"""E9: law derivation and hole-poking on REAL data (the user's 'learns from real data, pokes holes in bad theories, derives laws').
Data (copied verbatim, fetched 2026-10-22):
  planets: NASA NSSDCA Planetary Fact Sheet, https://nssdc.gsfc.nasa.gov/planetary/factsheet/ ('Distance from Sun (10^6 km)', 'Orbital Period (days)')
  Galilean moons: JPL Solar System Dynamics, planetary satellite mean elements, https://ssd.jpl.nasa.gov/sats/elem/ (semi-major axis km, period days; MEAN elements, 'not intended for ephemeris computation')
No noise model is given: every hypothesis is fitted in log space with its own residual scale estimated by maximum likelihood; models compared by BIC (n log(RSS/n) + k log n, k counts the noise scale).
PREDICTIONS P194-P197 (registered before running)."""
import json
import numpy as np
from scipy.optimize import least_squares

PLANETS = dict(Mercury=(57.9, 88.0), Venus=(108.2, 224.7), Earth=(149.6, 365.2), Mars=(228.0, 687.0), Jupiter=(778.5, 4331.0), Saturn=(1432.0, 10747.0), Uranus=(2867.0, 30589.0), Neptune=(4515.0, 59800.0))
PLUTO = (5906.4, 90560.0)
import os
if os.environ.get("SIDEREAL") == "1":   # E9b: per-planet NSSDCA fact sheets (semimajor axis 10^6 km, SIDEREAL orbit period days), all fetched 2026-10-22
    PLANETS = dict(Mercury=(57.909, 87.969), Venus=(108.210, 224.701), Earth=(149.598, 365.256), Mars=(227.956, 686.980), Jupiter=(778.479, 4332.589), Saturn=(1432.041, 10755.699), Uranus=(2867.043, 30685.400), Neptune=(4514.953, 60189.018))
    PLUTO = (5869.656, 90560.0)
MOONS = dict(Io=(0.4218, 1.762732), Europa=(0.6711, 3.525463), Ganymede=(1.0704, 7.155588), Callisto=(1.8827, 16.690440))   # a in 10^6 km
if os.environ.get("MOONS") == "NSSDC":   # E9c (audit V4): sidereal periods from the NSSDC Jovian Satellite Fact Sheet, https://nssdc.gsfc.nasa.gov/planetary/factsheet/joviansatfact.html, fetched 2026-10-28 (the JPL mean-elements table lists 1.762732 / 3.525463 for Io / Europa)
    MOONS = dict(Io=(0.4218, 1.769138), Europa=(0.6711, 3.551181), Ganymede=(1.0704, 7.154553), Callisto=(1.8827, 16.689017))


def bic(rss, n, k):
    return n * np.log(rss / n) + k * np.log(n)


def fit_library(a, T):
    la, lT = np.log(a), np.log(T); n = len(a); out = {}
    for p in (1.0, 1.5, 2.0):
        lc = np.mean(lT - p * la); r = lT - (lc + p * la); out[f"T~a^{p:g}"] = dict(rss=float(r @ r), k=2, params=[float(np.exp(lc)), p])
    A = np.c_[np.ones(n), la]; coef, *_ = np.linalg.lstsq(A, lT, rcond=None); r = lT - A @ coef
    out["T=c a^p (p free)"] = dict(rss=float(r @ r), k=3, params=[float(np.exp(coef[0])), float(coef[1])])
    f = least_squares(lambda q: np.log(np.maximum(q[0] * a + q[1], 1e-9)) - lT, [T[-1] / a[-1], 0.0]); r = f.fun
    out["T=c a + d"] = dict(rss=float(r @ r), k=3, params=[float(v) for v in f.x])
    f = least_squares(lambda q: (q[0] + a / q[1]) - lT, [lT[0], a[-1]], bounds=([-50, 1e-3], [50, 1e6])); r = f.fun
    out["T=c exp(a/s)"] = dict(rss=float(r @ r), k=3, params=[float(np.exp(f.x[0])), float(f.x[1])])
    for m in out.values():
        m["bic"] = float(bic(max(m["rss"], 1e-300), n, m["k"]))
    return out


def slope_se(a, T):
    la, lT = np.log(a), np.log(T); A = np.c_[np.ones(len(a)), la]; coef, *_ = np.linalg.lstsq(A, lT, rcond=None)
    r = lT - A @ coef; s2 = r @ r / (len(a) - 2); return float(coef[1]), float(np.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1]))


if __name__ == "__main__":
    a = np.array([v[0] for v in PLANETS.values()]); T = np.array([v[1] for v in PLANETS.values()])
    lib = fit_library(a, T); best = min(lib, key=lambda k: lib[k]["bic"])
    print("8 planets, hypothesis library (BIC, lower is better):")
    for k, m in sorted(lib.items(), key=lambda kv: kv[1]["bic"]):
        print(f"  {k:18s} BIC {m['bic']:8.1f} | dBIC vs a^1.5 {m['bic'] - lib['T~a^1.5']['bic']:7.1f} | params {m['params']}")
    p, se = slope_se(a, T); print(f"free exponent {p:.4f} +- {se:.4f} (1 s.e.); selected: {best}")
    c15 = lib["T~a^1.5"]["params"][0]; pl = c15 * PLUTO[0] ** 1.5
    print(f"Pluto: predicted {pl:.0f} d vs listed {PLUTO[1]:.0f} d -> error {100 * (pl / PLUTO[1] - 1):+.2f}%")
    am = np.array([v[0] for v in MOONS.values()]); Tm = np.array([v[1] for v in MOONS.values()])
    pm, sem = slope_se(am, Tm); print(f"Galilean moons alone: free exponent {pm:.4f} +- {sem:.4f}")
    aa, TT = np.r_[a, am], np.r_[T, Tm]; la, lT = np.log(aa), np.log(TT); n = len(aa); g = np.r_[np.zeros(len(a)), np.ones(len(am))]
    r1 = lT - 1.5 * la; r1 = r1 - r1.mean(); b_one = bic(r1 @ r1, n, 2)
    r2 = lT - 1.5 * la; r2 = r2 - np.where(g == 0, r2[g == 0].mean(), r2[g == 1].mean()); b_two = bic(r2 @ r2, n, 3)
    cp = np.exp(np.mean((lT - 1.5 * la)[g == 0])); cm = np.exp(np.mean((lT - 1.5 * la)[g == 1]))
    print(f"one constant for all orbits: BIC {b_one:.1f} | one constant per central body: BIC {b_two:.1f} | difference {b_one - b_two:.1f}")
    print(f"implied M_sun / M_jupiter = (c_moons / c_planets)^2 = {(cm / cp) ** 2:.1f} (literature ~1047, from memory)")
    for name, (x, y) in MOONS.items():
        print(f"  {name:9s} residual vs per-body law {100 * (cm * x ** 1.5 / y - 1):+.2f}%")
    json.dump(dict(lib=lib, p=p, se=se, pluto_err=float(pl / PLUTO[1] - 1), moons_p=pm, moons_se=sem, dbic_hole=float(b_one - b_two), mass_ratio=float((cm / cp) ** 2), max_abs_moon_resid_pct=float(max(abs(100 * (cm * x ** 1.5 / y - 1)) for x, y in MOONS.values()))), open("../results/e9_kepler%s.json" % (("_sidereal" if os.environ.get("SIDEREAL") == "1" else "") + ("_nssdc" if os.environ.get("MOONS") == "NSSDC" else "")), "w"), indent=1)
