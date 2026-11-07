"""The problem suite: 50+ small input->choice problems that a plain neural network can learn.

Every problem has bit inputs (continuous values are thermometer-coded), K options, and a sampler. The probability net only ever
learns from "did my chosen option succeed?", never from the right answer.
"""
import zlib
import numpy as np

SUITE = []


def problem(name, family, n_in, K, ceiling=1.0):
    """register a sampler gen(rng, n) -> (X uint8 [n, n_in], y int [n]). ceiling = best possible accuracy (label noise)."""
    def deco(gen):
        SUITE.append(dict(name=name, family=family, n_in=n_in, K=K, ceiling=ceiling, gen=gen))
        return gen
    return deco


def bits(rng, n, m):
    return (rng.random((n, m)) < 0.5).astype(np.uint8)


def thermo(x, L=4):
    """x in [0,1] (n, d) -> (n, d*L) bits"""
    th = (np.arange(L) + 1) / (L + 1)
    return (x[:, :, None] > th).reshape(len(x), -1).astype(np.uint8)


def noisy(rng, y, K, p):
    flip = rng.random(len(y)) < p
    return np.where(flip, (y + rng.integers(1, K, len(y))) % K, y)


# ---------------------------------------------------------------- boolean, 2 inputs
def _b2(name, f):
    @problem(name, 'bool2', 2, 2)
    def gen(rng, n):
        X = bits(rng, n, 2)
        return X, f(X[:, 0], X[:, 1]).astype(int)
    return gen


for _n, _f in [('AND', lambda a, b: a & b), ('OR', lambda a, b: a | b), ('XOR', lambda a, b: a ^ b),
               ('NAND', lambda a, b: 1 - (a & b)), ('NOR', lambda a, b: 1 - (a | b)), ('XNOR', lambda a, b: 1 - (a ^ b)),
               ('A AND NOT B', lambda a, b: a & (1 - b)), ('IMPLIES', lambda a, b: (1 - a) | b),
               ('NOT A', lambda a, b: 1 - a)]:
    _b2(_n, _f)


# ---------------------------------------------------------------- boolean, 3 inputs
def _b3(name, f):
    @problem(name, 'bool3', 3, 2)
    def gen(rng, n):
        X = bits(rng, n, 3)
        return X, f(X[:, 0], X[:, 1], X[:, 2]).astype(int)
    return gen


for _n, _f in [('MAJ3', lambda a, b, c: (a + b + c >= 2)), ('PARITY3', lambda a, b, c: a ^ b ^ c),
               ('MUX (a picks b or c)', lambda a, b, c: np.where(a == 1, b, c)),
               ('AND3', lambda a, b, c: a & b & c), ('OR3', lambda a, b, c: a | b | c),
               ('EXACTLY ONE', lambda a, b, c: (a + b + c == 1)), ('A XOR (B AND C)', lambda a, b, c: a ^ (b & c)),
               ('(A AND B) OR C', lambda a, b, c: (a & b) | c)]:
    _b3(_n, _f)


# ---------------------------------------------------------------- boolean, larger
def _bn(name, n_in, f, K=2, ceiling=1.0, noise=0.0):
    @problem(name, 'boolN', n_in, K, ceiling)
    def gen(rng, n):
        X = bits(rng, n, n_in)
        y = f(X).astype(int)
        return X, noisy(rng, y, K, noise) if noise else y
    return gen


_bn('PARITY4', 4, lambda X: X.sum(1) % 2)
_bn('PARITY5', 5, lambda X: X.sum(1) % 2)
_bn('MAJ5', 5, lambda X: X.sum(1) >= 3)
_bn('3 OF 6', 6, lambda X: X.sum(1) >= 3)
_bn('MUX 2-select', 6, lambda X: X[np.arange(len(X)), 2 + (X[:, 0] * 2 + X[:, 1])])
_bn('ALL EQUAL 4', 4, lambda X: (X.min(1) == X.max(1)))
_bn('PALINDROME 4', 4, lambda X: (X[:, 0] == X[:, 3]) & (X[:, 1] == X[:, 2]))
_bn('EVEN ONES of 6', 6, lambda X: (X.sum(1) % 2 == 0))
_bn('XOR + 10% noise', 2, lambda X: X[:, 0] ^ X[:, 1], ceiling=0.90, noise=0.10)
_bn('MAJ5 + 10% noise', 5, lambda X: X.sum(1) >= 3, ceiling=0.90, noise=0.10)


# ---------------------------------------------------------------- distractors: only a few inputs matter
def _distract(name, n_rel, n_junk, f):
    @problem(name, 'distractor', n_rel + n_junk, 2)
    def gen(rng, n):
        X = bits(rng, n, n_rel + n_junk)
        return X, f(X[:, :n_rel]).astype(int)
    return gen


_distract('AND of 2, 6 junk bits', 2, 6, lambda R: R[:, 0] & R[:, 1])
_distract('XOR of 2, 6 junk bits', 2, 6, lambda R: R[:, 0] ^ R[:, 1])
_distract('MAJ3, 5 junk bits', 3, 5, lambda R: R.sum(1) >= 2)
_distract('ONE bit, 10 junk bits', 1, 10, lambda R: R[:, 0])


# ---------------------------------------------------------------- arithmetic on small numbers
def _num(X, a, b):
    return X[:, a] * 2 + X[:, a + 1], X[:, b] * 2 + X[:, b + 1]


@problem('COMPARE 2-bit (<,=,>)', 'arith', 4, 3)
def _(rng, n):
    X = bits(rng, n, 4); a, b = _num(X, 0, 2)
    return X, np.sign(a - b).astype(int) + 1


@problem('MAX of two 2-bit', 'arith', 4, 4)
def _(rng, n):
    X = bits(rng, n, 4); a, b = _num(X, 0, 2)
    return X, np.maximum(a, b).astype(int)


@problem('SUM >= 3 of two 2-bit', 'arith', 4, 2)
def _(rng, n):
    X = bits(rng, n, 4); a, b = _num(X, 0, 2)
    return X, (a + b >= 3).astype(int)


@problem('COUNT ONES of 4', 'arith', 4, 5)
def _(rng, n):
    X = bits(rng, n, 4)
    return X, X.sum(1).astype(int)


@problem('ADD two 2-bit (7 sums)', 'arith', 4, 7)
def _(rng, n):
    X = bits(rng, n, 4); a, b = _num(X, 0, 2)
    return X, (a + b).astype(int)


# ---------------------------------------------------------------- 2-D classification (each coordinate -> 4 thermometer bits)
def _c2(name, K, label, ceiling=1.0, noise=0.0, d=2):
    @problem(name, 'classify', d * 4, K, ceiling)
    def gen(rng, n):
        P = rng.random((n, d))
        y = label(P).astype(int)
        return thermo(P), noisy(rng, y, K, noise) if noise else y
    return gen


_c2('HALF PLANE', 2, lambda P: P[:, 0] > 0.5)
_c2('DIAGONAL', 2, lambda P: P[:, 0] + P[:, 1] > 1)
_c2('XOR QUADRANTS', 2, lambda P: (P[:, 0] > 0.5) ^ (P[:, 1] > 0.5))
_c2('CHECKERBOARD 4x4', 2, lambda P: (np.floor(P[:, 0] * 4) + np.floor(P[:, 1] * 4)) % 2)
_c2('CIRCLE', 2, lambda P: np.hypot(P[:, 0] - .5, P[:, 1] - .5) < 0.32)
_c2('RING of 3 bands', 3, lambda P: np.minimum(2, (np.hypot(P[:, 0] - .5, P[:, 1] - .5) / 0.2).astype(int)))
_c2('STRIPES', 2, lambda P: np.floor(P[:, 0] * 5) % 2)
_c2('DIAGONAL BAND', 2, lambda P: np.abs(P[:, 0] - P[:, 1]) < 0.2)
_c2('FOUR CORNERS', 4, lambda P: (P[:, 0] > .5) * 2 + (P[:, 1] > .5))
_c2('HALF PLANE + 10% noise', 2, lambda P: P[:, 0] > 0.5, ceiling=0.90, noise=0.10)
_c2('3D OCTANT PARITY', 2, lambda P: ((P[:, 0] > .5).astype(int) + (P[:, 1] > .5) + (P[:, 2] > .5)) % 2, d=3)


def _blobs(name, K, d, spread):
    rng0 = np.random.default_rng(zlib.crc32(name.encode()))
    C = rng0.uniform(0.2, 0.8, (K, d))

    @problem(name, 'classify', d * 4, K)
    def gen(rng, n):
        y = rng.integers(0, K, n)
        P = np.clip(C[y] + rng.normal(0, spread, (n, d)), 0, 0.999)
        return thermo(P), y
    return gen


_blobs('BLOBS 2 classes', 2, 2, 0.08)
_blobs('BLOBS 3 classes', 3, 2, 0.07)
_blobs('BLOBS 4 classes in 4D', 4, 4, 0.07)


@problem('TWO MOONS', 'classify', 8, 2)
def _(rng, n):
    y = rng.integers(0, 2, n); t = rng.random(n) * np.pi
    x = np.where(y == 0, np.cos(t), 1 - np.cos(t)); yy = np.where(y == 0, np.sin(t), 0.5 - np.sin(t))
    P = np.stack([(x + 1.2) / 3.4, (yy + 0.9) / 2.2], 1) + rng.normal(0, 0.02, (n, 2))
    return thermo(np.clip(P, 0, 0.999)), y


@problem('SPIRAL arms', 'classify', 8, 2)
def _(rng, n):
    y = rng.integers(0, 2, n); r = rng.random(n) * 0.45 + 0.03
    ang = r * 9 + y * np.pi
    P = np.stack([0.5 + r * np.cos(ang), 0.5 + r * np.sin(ang)], 1)
    return thermo(np.clip(P, 0, 0.999)), y


# ---------------------------------------------------------------- higher-dimensional linear / threshold
def _lin(name, d, active):
    w0 = np.random.default_rng(zlib.crc32(name.encode())).normal(0, 1, d) * (np.arange(d) < active)

    @problem(name, 'linear', d, 2)
    def gen(rng, n):
        X = bits(rng, n, d)
        return X, ((X * 2 - 1) @ w0 > 0).astype(int)
    return gen


_lin('THRESHOLD 6 inputs', 6, 6)
_lin('THRESHOLD 8 inputs', 8, 8)
_lin('THRESHOLD 3 of 8 matter', 8, 3)


# ---------------------------------------------------------------- pictures
@problem('3x3 LINE: row/col/none', 'pattern', 9, 3)
def _(rng, n):
    y = rng.integers(0, 3, n); G = (rng.random((n, 3, 3)) < 0.12).astype(np.uint8)
    for i in range(n):
        k = rng.integers(0, 3)
        if y[i] == 0:
            G[i, k, :] = 1
        elif y[i] == 1:
            G[i, :, k] = 1
        else:
            for r in range(3):
                if G[i, r, :].all():
                    G[i, r, rng.integers(0, 3)] = 0
                if G[i, :, r].all():
                    G[i, rng.integers(0, 3), r] = 0
    return G.reshape(n, 9), y


@problem('5x5 SHAPES plus/box/X (5% noise)', 'pattern', 25, 3, 0.95)
def _(rng, n):
    base = np.zeros((3, 5, 5), np.uint8)
    base[0, 2, :] = 1; base[0, :, 2] = 1
    base[1, 0, :] = 1; base[1, 4, :] = 1; base[1, :, 0] = 1; base[1, :, 4] = 1
    for i in range(5):
        base[2, i, i] = 1; base[2, i, 4 - i] = 1
    y = rng.integers(0, 3, n); G = base[y].reshape(n, 25).copy()
    flip = rng.random((n, 25)) < 0.05
    return G ^ flip.astype(np.uint8), y


@problem('LEFT vs RIGHT heavier (4x4)', 'pattern', 16, 2)
def _(rng, n):
    G = (rng.random((3 * n, 4, 4)) < 0.5).astype(np.uint8)
    L, R = G[:, :, :2].sum((1, 2)), G[:, :, 2:].sum((1, 2))
    keep = np.where(L != R)[0][:n]
    G, L, R = G[keep], L[keep], R[keep]
    return G.reshape(len(G), 16), (R > L).astype(int)


@problem('MIRROR SYMMETRIC 3x3?', 'pattern', 9, 2)
def _(rng, n):
    G = (rng.random((n, 3, 3)) < 0.5).astype(np.uint8)
    sym = rng.random(n) < 0.5
    G[sym, :, 2] = G[sym, :, 0]
    y = (G[:, :, 0] == G[:, :, 2]).all(1).astype(int)
    return G.reshape(n, 9), y


@problem('CORNERS ON count (3 classes)', 'pattern', 9, 3)
def _(rng, n):
    G = bits(rng, n, 9)
    c = G[:, [0, 2, 6, 8]].sum(1)
    return G, np.where(c <= 1, 0, np.where(c == 2, 1, 2))


# ---------------------------------------------------------------- sequences: window of recent bits -> next bit
def _seq(name, m, nxt, pattern=None):
    @problem(name, 'sequence', m, 2)
    def gen(rng, n):
        if pattern is not None:
            P = len(pattern); ph = rng.integers(0, P, n)
            idx = (ph[:, None] + np.arange(m + 1)) % P
            S = np.asarray(pattern)[idx].astype(np.uint8)
        else:
            S = bits(rng, n, m + 1)
        return S[:, :m], nxt(S).astype(int)
    return gen


_seq('NEXT: pattern 110', 3, lambda S: S[:, 3], [1, 1, 0])
_seq('NEXT: pattern 1000', 4, lambda S: S[:, 4], [1, 0, 0, 0])
_seq('NEXT: pattern 10110', 5, lambda S: S[:, 5], [1, 0, 1, 1, 0])
_seq('NEXT: alternate 01', 2, lambda S: S[:, 2], [0, 1])
_seq('NEXT = XOR of last two', 4, lambda S: S[:, 3] ^ S[:, 2])
_seq('NEXT = bit from 3 ago', 4, lambda S: S[:, 1])
_seq('NEXT = majority of last 3', 4, lambda S: S[:, 1:4].sum(1) >= 2)
_seq('NEXT = parity of window of 4', 4, lambda S: S[:, :4].sum(1) % 2)

for i, p in enumerate(SUITE):
    p['id'] = i


def sample(p, rng, n):
    return p['gen'](rng, n)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    fams = {}
    for p in SUITE:
        X, y = p['gen'](rng, 2000)
        assert X.shape[1] == p['n_in'], p['name']
        assert y.min() >= 0 and y.max() < p['K'], p['name']
        fams[p['family']] = fams.get(p['family'], 0) + 1
        print(f"{p['id']:2d} {p['family']:10s} {p['name']:34s} in {p['n_in']:2d} K {p['K']} chance {np.bincount(y).max() / len(y):.2f}")
    print(len(SUITE), 'problems', fams)
