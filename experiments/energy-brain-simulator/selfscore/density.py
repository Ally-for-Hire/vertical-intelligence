"""Distribution-level theory of culling (no normality assumption). See THEORY.md, Layer 3.

State: f(z), the probability density of true accuracy z over the population. One generation = inner life -> noisy-score truncation cull (exact for finite n) -> clone + mutate.

  inner life         f1 = f * N(mu_g, sig_g^2)                                        (convolution; mass beyond the accuracy bounds piles up at them)
  retention          r(z) = P(individual with true accuracy z is among the k survivors)
                          = E_e[ BinCDF(k-1; n-1, Sbar(z+e)) ],   Sbar = survival function of the score s = z + e over the population
                     (exactly: it survives iff at most k-1 of the other n-1 outrank it)
  survivors          g(z) = (n/k) f1(z) r(z)                                          (integrates to 1)
  next generation    f2 = (k/n) g + ((n-k)/n) (g * N(mu_m, sig_m^2))                 (survivors plus mutated copies)

Everything the cull does to accuracy is in r(z): the cull never sees z, only the score, and r(z) is the exact probability that z survives. The selection differential is
S = integral z g - integral z f1. For Gaussian f1 this reduces to the breeder's equation S = rho i(n,k) sigma (selfscore/theory.py).
"""
import numpy as np
from scipy.stats import norm, binom


class Grid:
    def __init__(s, lo=-0.6, hi=1.6, G=4401):
        s.z = np.linspace(lo, hi, G); s.dz = s.z[1] - s.z[0]; s.G = G

    def gauss(s, mu, sig):
        """discrete kernel for convolution on the grid (centred at mu)"""
        if sig < 1e-9:
            k = np.zeros(2 * 1 + 1); sh = int(round(mu / s.dz)); K = np.zeros(2 * abs(sh) + 3); K[abs(sh) + 1 + sh] = 1.0; return K
        L = int(np.ceil((abs(mu) + 6 * sig) / s.dz)); off = np.arange(-L, L + 1) * s.dz
        K = norm.pdf(off, mu, sig) * s.dz; return K / K.sum()

    def conv(s, f, mu, sig):
        K = s.gauss(mu, sig); return np.convolve(f, K, mode='same')

    def clamp(s, f, lo_b, hi_b):
        """accuracy is bounded: mass outside [lo_b, hi_b] is moved to the bound"""
        f = f.copy(); mass = f * s.dz
        below = s.z < lo_b; above = s.z > hi_b
        ib = int(np.argmin(np.abs(s.z - lo_b))); ia = int(np.argmin(np.abs(s.z - hi_b)))
        f[ib] += mass[below].sum() / s.dz; f[ia] += mass[above].sum() / s.dz; f[below] = 0; f[above] = 0
        return f

    def mean(s, f): return float((s.z * f).sum() * s.dz / (f.sum() * s.dz))
    def var(s, f): m = s.mean(f); return float(((s.z - m) ** 2 * f).sum() * s.dz / (f.sum() * s.dz))


def retention(grid, f1, n, k, sig_e):
    """r(z): exact probability that an individual of true accuracy z survives a cull of the top k of n, ranking by a score with Gaussian noise sig_e"""
    dz = grid.dz
    if sig_e < 1e-9: fs = f1
    else: fs = grid.conv(f1, 0.0, sig_e)
    Sbar = 1.0 - np.cumsum(fs) * dz + 0.5 * fs * dz                                  # P(another score is above s), midpoint rule
    Sbar = np.clip(Sbar, 0.0, 1.0)
    h = binom.cdf(k - 1, n - 1, Sbar) if n > 1 else np.ones_like(Sbar)
    return h if sig_e < 1e-9 else grid.conv(h, 0.0, sig_e)                            # average over the score noise


def step(grid, f, n, k, sig_e, mu_m, sig_m, mu_g=0.0, sig_g=0.0, bounds=(-1e9, 1e9)):
    f1 = f if (abs(mu_g) < 1e-12 and sig_g < 1e-12) else grid.conv(f, mu_g, sig_g)
    f1 = grid.clamp(f1, *bounds) if bounds[0] > -1e8 else f1
    f1 = f1 / (f1.sum() * grid.dz)
    r = retention(grid, f1, n, k, sig_e)
    g = (n / k) * f1 * r; mass = g.sum() * grid.dz; g = g / mass                      # mass ~ 1 up to quadrature error
    S = grid.mean(g) - grid.mean(f1)
    kids = grid.conv(g, mu_m, sig_m)
    if bounds[0] > -1e8: kids = grid.clamp(kids, *bounds)
    f2 = (k / n) * g + ((n - k) / n) * kids
    return f2 / (f2.sum() * grid.dz), dict(S=S, mean=grid.mean(f2), var=grid.var(f2), mass=mass, mean_after_inner=grid.mean(f1), var_after_inner=grid.var(f1))


def from_gaussian(grid, mean, sd):
    f = norm.pdf(grid.z, mean, sd); return f / (f.sum() * grid.dz)


def trajectory(grid, f0, gens, n, k, sig_e, mu_m, sig_m, mu_g=0.0, sig_g=0.0, bounds=(-1e9, 1e9)):
    f = f0; out = []
    for _ in range(gens):
        f, d = step(grid, f, n, k, sig_e, mu_m, sig_m, mu_g, sig_g, bounds); out.append(d)
    return out, f
