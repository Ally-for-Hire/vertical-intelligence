"""Theory of how culling changes accuracy (selfscore/THEORY.md). Pure functions + an abstract Monte Carlo that tests the algebra independently of any network.

Setting. A population of n individuals has a true accuracy z_i (the quantity we care about) that the cull never sees directly: it ranks by a noisy score s_i = z_i + e_i
(e.g. accuracy on a random batch). The bottom fraction c is removed (k = n - round(c n) survivors remain), survivors are multiplied back up to n with a small random
change m to each copy, and each individual may also change by its own 'inner life' g before the cull.

LAYER 0  (exact identity, no assumptions):   zbar' - zbar = T_inner + S_w + T_kids          [Price equation specialised to this algorithm]
LAYER 1  (selection differential; needs only that (z, s) are jointly normal):   E[S] = rho * i(n,k) * sigma_z,   rho = sigma_z / sqrt(sigma_z^2 + sigma_e^2)
LAYER 2  (variance recursion; Bulmer effect + mutation):   sigma'^2 = rho^2 sigma_z^2 v(n,k) + (1-rho^2) sigma_z^2 (k-1)/k + f sigma_m^2 + f(1-f) mu_m^2
"""
import numpy as np
from scipy.stats import norm, binom

# ---------------------------------------------------------------- selection intensity ---------------------------------------------------------------
def x_trunc(p):
    """truncation point (in SD units) that retains the top fraction p of a standard normal"""
    return norm.isf(p)


def i_inf(p):
    """selection intensity, infinite population: mean of the retained top fraction p of N(0,1) = phi(x_p)/p"""
    return norm.pdf(norm.isf(p)) / p


def k_inf(p):
    """variance-reduction coefficient: retained variance = 1 - k, with k = i (i - x_p)"""
    i = i_inf(p); return i * (i - norm.isf(p))


_cache = {}
def i_finite(n, k):
    """EXACT expected mean of the top k of n iid N(0,1) values (finite-population selection intensity).
    E[sum of top k] = n * integral x phi(x) P(Bin(n-1, 1-Phi(x)) <= k-1) dx  (a point is in the top k iff at most k-1 of the other n-1 exceed it)."""
    key = ('i', n, k)
    if key not in _cache:
        x = np.linspace(-9, 9, 36001); dx = x[1] - x[0]
        w = norm.pdf(x) * binom.cdf(k - 1, n - 1, norm.sf(x)) if n > 1 else norm.pdf(x)
        _cache[key] = float(n * np.sum(x * w) * dx / k)
    return _cache[key]


def v_finite(n, k, reps=4000, seed=0):
    """expected population variance (1/k sum (z - zbar)^2) of the top k of n iid N(0,1), by Monte Carlo (deterministic seed); -> 1 - k_inf(p) as n grows"""
    key = ('v', n, k)
    if key not in _cache:
        r = np.random.default_rng(seed); z = np.sort(r.standard_normal((reps, n)), 1)[:, -k:]
        _cache[key] = float(z.var(1).mean())
    return _cache[key]


# ---------------------------------------------------------------- moment recursion -------------------------------------------------------------------
def step(mean, var, n, k, sig_e2, mu_m, sig_m2, mu_g=0.0, sig_g2=0.0, finite=True):
    """one generation of the two-moment recursion; returns (new_mean, new_var, S_predicted, rho)"""
    mean = mean + mu_g; var = var + sig_g2                                # inner life
    rho2 = var / (var + sig_e2); rho = np.sqrt(rho2)
    ii = i_finite(n, k) if finite else i_inf(k / n)
    vv = v_finite(n, k) if finite else 1 - k_inf(k / n)
    S = rho * ii * np.sqrt(var)                                            # layer 1
    var_s = rho2 * var * vv + (1 - rho2) * var * (k - 1) / k               # survivors' variance (layer 2)
    f = (n - k) / n                                                        # fraction of the new population that is mutated copies
    new_mean = mean + S + f * mu_m
    new_var = var_s + f * sig_m2 + f * (1 - f) * mu_m ** 2
    return new_mean, new_var, S, rho


def trajectory(mean0, var0, gens, n, k, sig_e2, mu_m, sig_m2, mu_g=0.0, sig_g2=0.0, finite=True):
    m, v = mean0, var0; out = []
    for _ in range(gens):
        m, v, S, rho = step(m, v, n, k, sig_e2, mu_m, sig_m2, mu_g, sig_g2, finite); out.append((m, v, S, rho))
    return np.array(out)


def fixed_point(n, k, sig_e2, mu_m, sig_m2, mu_g=0.0, sig_g2=0.0, finite=True, iters=400):
    """stationary variance and per-generation gain (unbounded trait): iterate the variance recursion to convergence"""
    m, v = 0.0, max(sig_m2, 1e-6)
    for _ in range(iters): m2, v2, S, rho = step(m, v, n, k, sig_e2, mu_m, sig_m2, mu_g, sig_g2, finite); gain = m2 - m; m, v = 0.0, v2
    return dict(var=v, sd=np.sqrt(v), gain=gain, S=S, rho=rho)


# ---------------------------------------------------------------- abstract Monte Carlo (tests the algebra, not any network) -----------------------------------
def simulate(n, k, sig_e, mu_m, sig_m, gens, mean0=0.0, sd0=0.1, mu_g=0.0, sig_g=0.0, reps=200, seed=0):
    """direct simulation of: inner life -> noisy-score truncation cull -> clone survivors with Gaussian mutation. returns per-generation mean/var/S averaged over reps"""
    r = np.random.default_rng(seed); z = mean0 + sd0 * r.standard_normal((reps, n))
    out = np.zeros((gens, 4))
    for t in range(gens):
        z = z + mu_g + sig_g * r.standard_normal(z.shape)
        s = z + sig_e * r.standard_normal(z.shape)
        order = np.argsort(-s, 1); keep = np.take_along_axis(z, order[:, :k], 1)                   # true z of survivors, ranked by the noisy score
        S = keep.mean(1) - z.mean(1)
        kids = keep[:, np.arange(n - k) % k] + mu_m + sig_m * r.standard_normal((reps, n - k))
        z = np.concatenate([keep, kids], 1)
        out[t] = [z.mean(1).mean(), z.var(1).mean(), S.mean(), 0.0]
    return out
