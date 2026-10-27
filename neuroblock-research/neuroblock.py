"""
Neuron-block sandbox.

- Nodes live on a 2D plane. Node 0 is the sensor hub (fixed at origin).
- Every other node is a specialist with a prototype vector over the sensor inputs.
- Edges form a tree; transit delay along an edge = length * US_PER_UNIT microseconds.
- Core instinct = EXPANSION: an input nobody explains spawns a new specialist.
- Thinking = signal floods out from the hub; the first matching specialist to be
  reached fires (select='fast'), or the best-fitting matching one (select='best').
- Condensing (at the end of each window): move nodes along traffic-weighted springs,
  merge near-duplicate specialists, prune unused ones.
"""
import numpy as np

US = 40.0  # microseconds per unit of edge length


class Brain:
    def __init__(s, D, K, rho, rng, lr=0.1, move=False, merge=False, prune=False,
                 min_sep=0.3, select='fast', spawn_len=1.0, eta=0.5, spawn_cost=50.0):
        s.D, s.K, s.rho, s.rng, s.lr = D, K, rho, rng, lr
        s.move, s.merge, s.prune = move, merge, prune
        s.min_sep, s.select, s.spawn_len, s.eta, s.spawn_cost = min_sep, select, spawn_len, eta, spawn_cost
        s.pos = np.zeros((1, 2))
        s.w = np.full((1, D), np.nan)
        s.par = np.array([-1])
        s.d = np.zeros(1)
        s.cnt = np.zeros(1)
        s.ema = np.zeros(1)
        s.age = np.zeros(1, int)
        s.win_use = np.zeros(1)
        s.lab = np.zeros((1, K))

    def lr_of(s, i):
        return s.lr

    def eta_vec(s):
        return np.full(len(s.pos), s.eta)

    @property
    def n(s):
        return len(s.pos) - 1

    # ---------- one experience ----------
    def step(s, x, y):
        dist = None
        if s.n > 0:
            dist = np.linalg.norm(s.w[1:] - x, axis=1)
            m = dist < s.rho
            if m.any():
                idx = np.where(m)[0] + 1
                win = idx[np.argmin(s.d[idx])] if s.select == 'fast' else idx[np.argmin(dist[idx - 1])]
                s.w[win] += s.lr_of(win) * (x - s.w[win])
                s.cnt[win] += 1
                s.win_use[win] += 1
                s.lab[win, y] += 1
                return s.d[win], win, False
        # nobody explains it -> exhaustive flood, then expansion
        t = (s.d.max() if s.n > 0 else 0.0) + s.spawn_cost
        par = 1 + int(np.argmin(dist)) if s.n > 0 else 0
        ang = s.rng.uniform(0, 2 * np.pi)
        p = s.pos[par] + s.spawn_len * np.array([np.cos(ang), np.sin(ang)])
        s.pos = np.vstack([s.pos, p])
        s.w = np.vstack([s.w, x])
        s.par = np.append(s.par, par)
        s.d = np.append(s.d, s.d[par] + s.spawn_len * US)
        s.cnt = np.append(s.cnt, 1.0)
        s.ema = np.append(s.ema, 1.0)
        s.age = np.append(s.age, 0)
        s.win_use = np.append(s.win_use, 1.0)
        l = np.zeros((1, s.K)); l[0, y] = 1
        s.lab = np.vstack([s.lab, l])
        return t, s.n, True

    # ---------- pure query (no learning) ----------
    def query(s, X, Y):
        """returns mean time-to-fire (us), accuracy, coverage for test points"""
        if s.n == 0:
            return s.spawn_cost, 0.0, 0.0
        dist = np.linalg.norm(X[:, None, :] - s.w[None, 1:, :], axis=2)
        m = dist < s.rho
        covered = m.any(1)
        if s.select == 'fast':
            tt = np.where(m, s.d[1:][None, :], np.inf)
        else:
            tt = np.where(m, dist, np.inf)
        win = tt.argmin(1) + 1
        t = np.where(covered, s.d[win], s.d.max() + s.spawn_cost)
        pred = s.lab[win].argmax(1)
        acc = np.mean(covered & (pred == Y))
        return float(t.mean()), float(acc), float(covered.mean())

    # ---------- condensing ----------
    def condense(s):
        s.ema = 0.8 * s.ema + s.win_use
        s.age += 1
        if s.prune and s.n > 0:
            s._prune()
        if s.merge and s.n > 1:
            s._merge()
        if s.move and s.n > 0:
            s._move()
        s._recompute()
        s.win_use[:] = 0

    def _apply(s, redirect, merge_into):
        """redirect: {removed: new_parent_target}; merge_into: {removed: survivor} (for stats)"""
        if not redirect:
            return
        for j, i in merge_into.items():
            tot = s.cnt[i] + s.cnt[j]
            s.w[i] = (s.w[i] * s.cnt[i] + s.w[j] * s.cnt[j]) / max(tot, 1e-9)
            s.cnt[i] = tot
            s.ema[i] += s.ema[j]
            s.win_use[i] += s.win_use[j]
            s.lab[i] += s.lab[j]
        removed = set(redirect)
        newpar = s.par.copy()
        for k in range(1, len(s.par)):
            p = newpar[k]
            while p in removed:
                p = redirect[p]
            newpar[k] = p
        keep = np.array([i not in removed for i in range(len(s.par))])
        remap = np.cumsum(keep) - 1
        s.par = np.array([-1] + [remap[newpar[k]] for k in range(1, len(s.par)) if keep[k]])
        for name in ['pos', 'w', 'cnt', 'ema', 'age', 'win_use', 'lab']:
            setattr(s, name, getattr(s, name)[keep])
        s.d = np.zeros(len(s.pos))

    def _prune(s):
        red = {int(j): int(s.par[j]) for j in range(1, len(s.par)) if s.ema[j] < 0.5 and s.age[j] > 3}
        s._apply(red, {})

    def _merge(s):
        thr = 0.5 * s.rho
        W = s.w[1:]
        D = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
        n = len(W)
        gone, red = set(), {}
        for i in range(n):
            if i in gone:
                continue
            for j in range(i + 1, n):
                if j not in gone and D[i, j] < thr:
                    gone.add(j)
                    red[j + 1] = i + 1
        s._apply(red, dict(red))

    def _move(s):
        n = len(s.pos)
        s.pos = s.pos + s.rng.normal(0, 1e-3, s.pos.shape) * (np.arange(n)[:, None] > 0)
        sub = s.ema.copy(); sub[0] = 0
        for i in range(n - 1, 0, -1):
            sub[s.par[i]] += sub[i]
        sub = sub + 0.2
        num = np.zeros((n, 2)); den = np.zeros(n)
        for i in range(1, n):
            p = s.par[i]; wgt = sub[i]
            num[i] += wgt * s.pos[p]; den[i] += wgt
            if p > 0:
                num[p] += wgt * s.pos[i]; den[p] += wgt
        mask = den > 0; mask[0] = False
        target = num[mask] / den[mask, None]
        s.pos[mask] = s.pos[mask] + s.eta_vec()[mask][:, None] * (target - s.pos[mask])
        for _ in range(8):
            s._separate()

    def _separate(s):
        if s.min_sep <= 0:
            return
        P = s.pos
        diff = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(diff, axis=2) + np.eye(len(P)) * 1e9
        over = np.clip(s.min_sep - dist, 0, None)
        push = ((over / (dist + 1e-9))[..., None] * diff).sum(1) * 0.5
        push[0] = 0
        s.pos = P + push

    def _recompute(s):
        s.d = np.zeros(len(s.pos))
        for i in range(1, len(s.pos)):
            s.d[i] = s.d[s.par[i]] + np.linalg.norm(s.pos[i] - s.pos[s.par[i]]) * US


# ---------------- experiment harness ----------------
def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def run(cfg, seed, K=6, D=8, sigma=0.06, T=6000, win=200, drift=True, keep_snapshots=False,
        cls=None, drift_mode='replace'):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 1, (K, D))
    probs = 1 / np.arange(1, K + 1); probs /= probs.sum()
    b = (cls or Brain)(D, K, rng=np.random.default_rng(seed + 1000), **cfg)
    W = T // win
    dw = W // 2
    rows, snaps = [], {}

    def sample(n):
        y = rng.choice(K, n, p=probs)
        return centers[y] + rng.normal(0, sigma, (n, D)), y

    for w in range(W):
        if drift and w == dw:
            if drift_mode == 'replace':
                sel = [1, 2, 3]
                centers[sel] = rng.uniform(0, 1, (len(sel), D))
            else:  # 'shift': every concept slides a modest distance (needs prototypes to move)
                centers += rng.normal(0, 0.12, centers.shape)
        X, Y = sample(win)
        ts, sp = [], 0
        for x, y in zip(X, Y):
            t, _, spawned = b.step(x, y)
            ts.append(t); sp += spawned
            if hasattr(b, 'remember'):
                b.remember(x, y)
        n_before = b.n
        b.condense()
        Xt, Yt = sample(400)
        tq, acc, cov = b.query(Xt, Yt)
        rows.append(dict(w=w, t_train=float(np.mean(ts)), spawns=sp, n_before=n_before, n=b.n,
                         t_query=tq, acc=acc, cov=cov, exp=getattr(b, 'exp_total', 0),
                         lr_mean=float(np.mean([b.lr_of(i) for i in range(1, len(b.pos))])) if b.n else 0.0,
                         depth_corr=spearman(b.cnt[1:], b.d[1:]) if b.n > 2 else np.nan,
                         mean_edge=float(np.mean([np.linalg.norm(b.pos[i] - b.pos[b.par[i]]) for i in range(1, len(b.pos))])) if b.n else 0.0))
        if keep_snapshots and w in (dw - 1, W - 1):
            snaps[w] = dict(pos=b.pos.tolist(), par=b.par.tolist(), cnt=b.cnt.tolist(), d=b.d.tolist())
    return rows, snaps, dw


def summarize(rows, dw, K=6):
    pre = rows[dw - 4:dw]; post = rows[-4:]
    def avg(rs, k):
        return float(np.nanmean([r[k] for r in rs]))
    out = {}
    for tag, rs in (('pre', pre), ('post', post)):
        for k in ('t_query', 'acc', 'cov', 'n', 'depth_corr', 'mean_edge', 'spawns'):
            out[f'{tag}_{k}'] = avg(rs, k)
        out[f'{tag}_F'] = out[f'{tag}_acc'] / ((out[f'{tag}_t_query'] / 100 + 0.5) * (0.5 + out[f'{tag}_n'] / K))
    target = 0.9 * out['pre_acc']
    rec = next((r['w'] - dw for r in rows[dw:] if r['acc'] >= target), np.inf)
    out['recovery_windows'] = rec
    return out
