"""A general optimizer evolved from randomness (user, 2026-10-26: "a system that best optimizes any other given system evolved from randomness"), and the benchmark that judges it.
RULE (user): task performance is used ONLY by the external cull (who lives, who dies) and to pick what to report; every change inside an individual is blind random variation.

evolve(system, cfg, seed, budget) optimizes ANY system that provides:
    system.n                         number of parameters
    system.score(theta, rng)         EXTERNAL grade of P parameter vectors theta (P, n) -> (scores (P,), cost)   [the only place performance enters]
    system.done(theta_1)             monitoring only: has the reported individual reached the target?
Benchmark: 7 classification problems (different shapes, 2-3 classes, continuous and Boolean inputs), each solved by a d-16-K tanh net. Speed = compute to reach the problem's
target test accuracy, in sample-forward-equivalents (one sample through one net forward = 1; a backward pass = 2 more). References: gradient descent with default Adam and with
the best of 5 learning rates per problem. Suite score = geometric mean over problems of (evolution cost / default-Adam cost); unsolved runs count at the budget.

python emergent/general.py                 # reference numbers (cached) + the configurations in CONFIGS, 10 seeds per problem -> emergent/runs/general.json, emergent/GENERAL.md
"""
import os, sys, json, math, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import car.compare as CP

HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, 'runs'); os.makedirs(RUNS, exist_ok=True); RES = os.path.join(RUNS, 'general.json')
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if dev.type == 'cuda': torch.cuda.set_per_process_memory_fraction(0.3)
PROBLEMS = ['xorgauss', 'moons', 'circles', 'spirals', 'blobs3', 'checker3', 'parity4']
H = 16


class NetTask:
    """a d-H-K tanh net on one problem; parameters are flat vectors"""
    def __init__(s, prob, H=H):
        X, y, Xt, yt, K, kind = CP.problem(prob, 0); X = np.asarray(X, float); Xt = np.asarray(Xt, float)
        if kind == 'bits': X, Xt = 2 * X - 1, 2 * Xt - 1
        else: mu, sd = X.mean(0), X.std(0) + 1e-9; X, Xt = (X - mu) / sd, (Xt - mu) / sd
        f = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        s.Xtr, s.ytr, s.Xte, s.yte = f(X), torch.tensor(y, device=dev), f(Xt), torch.tensor(yt, device=dev)
        s.prob, s.d, s.H, s.K = prob, X.shape[1], H, K; s.n = s.d * H + H + H * K + K; s.target = None
    def split(s, th):
        d, H, K = s.d, s.H, s.K; a = d * H; b = a + H; c = b + H * K
        return th[:, :a].view(-1, d, H), th[:, a:b], th[:, b:c].view(-1, H, K), th[:, c:]
    def logits(s, th, X):
        W1, b1, W2, b2 = s.split(th); return torch.tanh(X @ W1 + b1[:, None]) @ W2 + b2[:, None]
    def acc(s, th, X, y): return (s.logits(th, X).argmax(2) == y[None]).float().mean(1)
    def score(s, th, g, B, kind='ll'):
        i = torch.randint(0, len(s.Xtr), (B,), device=dev, generator=g); X, y = s.Xtr[i], s.ytr[i]
        if kind == 'acc': return s.acc(th, X, y), th.shape[0] * B
        lp = torch.log_softmax(s.logits(th, X), 2); return lp.gather(2, y[None, :, None].expand(lp.shape[0], -1, 1))[..., 0].mean(1), th.shape[0] * B
    def done(s, th1): return float(s.acc(th1, s.Xte, s.yte)[0]) >= s.target


# ---- gradient-descent references ---------------------------------------------------------------------------------------------------------------------------------------
def gradient(task, seed, lr, max_steps=20000, bs=32, target=None):
    """returns (cost to reach target or None, best test accuracy seen)"""
    torch.manual_seed(20_000 + seed); d, Hh, K = task.d, task.H, task.K
    net = torch.nn.Sequential(torch.nn.Linear(d, Hh), torch.nn.Tanh(), torch.nn.Linear(Hh, K)).to(dev); opt = torch.optim.Adam(net.parameters(), lr=lr); best = 0.0
    for step in range(1, max_steps + 1):
        i = torch.randint(0, len(task.Xtr), (bs,), device=dev); l = torch.nn.functional.cross_entropy(net(task.Xtr[i]), task.ytr[i]); opt.zero_grad(); l.backward(); opt.step()
        if step % 10 == 0:
            with torch.no_grad(): a = float((net(task.Xte).argmax(1) == task.yte).float().mean())
            best = max(best, a)
            if target is not None and a >= target: return 3 * step * bs, best
    return None, best


# ---- the optimizer ------------------------------------------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def evolve(task, cfg, seed, budget):
    """blind variation + external culling. Returns (cost, solved, generations). cfg (all optional):
    P population | B scoring samples per generation (shared by all) | B_grow: B multiplies by this every generation (external evaluator spends more as it goes) |
    cull fraction that dies | init scale of the random start | fit 'll' or 'acc' | life: grade = average over the individual's life |
    sig0 range of random starting noise sizes (log10 lo, hi) | tau learning rate of the self-adaptive noise size (default 1/sqrt(n)) |
    mirror: children come in pairs parent+noise / parent-noise (blind antithetic noise) | pm: fraction of parameters a mutation touches; 'adapt' = each individual's own random, inherited gene"""
    g = torch.Generator(device=dev).manual_seed(seed); rn = lambda *s: torch.randn(*s, device=dev, generator=g); ru = lambda *s: torch.rand(*s, device=dev, generator=g)
    P, n = cfg.get('P', 40), task.n; cull = cfg.get('cull', 0.8); nk = max(1, int(round(P * (1 - cull)))); nc = P - nk
    tau = cfg.get('tau', 1.0 / math.sqrt(n)); lo, hi = cfg.get('sig0', (-2.0, 0.0))
    th = rn(P, n) * cfg.get('init', 0.5); sig = 10 ** (lo + (hi - lo) * ru(P)); pmv = 10 ** (-2 + 2 * ru(P)) if cfg.get('pm') == 'adapt' else None
    ssum = torch.zeros(P, device=dev); scnt = torch.zeros(P, device=dev); cost = 0; B = float(cfg.get('B', 16)); gen = 0
    while cost < budget:
        sc, c = task.score(th, g, max(1, int(B)), cfg.get('fit', 'll')); cost += c; B *= cfg.get('B_grow', 1.0)
        if cfg.get('life', True): ssum += sc; scnt += 1; sc = ssum / scnt
        order = torch.argsort(sc + 1e-7 * ru(P), descending=True)
        if task.done(th[order[0]:order[0] + 1]): return cost, True, gen + 1
        keep = order[:nk]
        if cfg.get('mirror', False):
            h = (nc + 1) // 2; pa = keep[torch.randint(0, nk, (h,), device=dev, generator=g)]; cs = sig[pa] * torch.exp(tau * rn(h)); z = rn(h, n)
            child = torch.cat([th[pa] + cs[:, None] * z, th[pa] - cs[:, None] * z])[:nc]; cs = torch.cat([cs, cs])[:nc]; pa = torch.cat([pa, pa])[:nc]
        else:
            pa = keep[torch.randint(0, nk, (nc,), device=dev, generator=g)]; cs = sig[pa] * torch.exp(tau * rn(nc)); z = rn(nc, n)
            if pmv is not None: z = z * (ru(nc, n) < pmv[pa, None])
            elif cfg.get('pm', 1.0) < 1.0: z = z * (ru(nc, n) < cfg['pm'])
            child = th[pa] + cs[:, None] * z
        th = torch.cat([th[keep], child]); sig = torch.cat([sig[keep], cs])
        if pmv is not None: pmv = torch.cat([pmv[keep], (pmv[pa] * torch.exp(tau * rn(nc))).clamp(1.0 / n, 1.0)])
        z0 = torch.zeros(nc, device=dev); ssum = torch.cat([ssum[keep], z0]); scnt = torch.cat([scnt[keep], z0]); gen += 1
    return cost, False, gen


# ---- benchmark ----------------------------------------------------------------------------------------------------------------------------------------------------------
LRS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def references(tasks, store, seeds):
    """per problem: target = (median over 3 seeds of the best test accuracy tuned Adam reaches in 20,000 steps) - 0.02, rounded down to 0.005; then the cost of default Adam and of the
    best learning rate to reach it, over the benchmark seeds"""
    for t in tasks:
        r = store.setdefault('ref', {}).setdefault(t.prob, {})
        if 'target' not in r:
            best = [max(gradient(t, s, lr, max_steps=20000)[1] for lr in (1e-2, 3e-2)) for s in range(3)]
            r['target'] = math.floor((float(np.median(best)) - 0.02) * 200) / 200; json.dump(store, open(RES, 'w'))
        t.target = r['target']
        for lr in LRS:
            k = f'lr{lr}'
            if k not in r: r[k] = [gradient(t, s, lr, target=t.target)[0] for s in seeds]; json.dump(store, open(RES, 'w'))
        med = lambda v: float(np.median([x if x is not None else 3 * 20000 * 32 for x in v]))
        r['default'] = med(r['lr0.001']); r['tuned'] = min(med(r[f'lr{lr}']) for lr in LRS); r['tuned_lr'] = min(LRS, key=lambda lr: med(r[f'lr{lr}']))


def suite(tasks, store, cfg, seeds, budget, label=None):
    k = label or json.dumps(cfg, sort_keys=True); rows = store.setdefault('runs', {}).setdefault(k, {})
    for t in tasks:
        if t.prob not in rows: rows[t.prob] = [list(evolve(t, cfg, s, budget)) for s in seeds]; json.dump(store, open(RES, 'w'))
    rat_d, rat_t, solved = [], [], 0
    for t in tasks:
        cs = [c for c, _, _ in rows[t.prob]]; solved += sum(o for _, o, _ in rows[t.prob]); m = float(np.median(cs)); r = store['ref'][t.prob]
        rat_d.append(m / r['default']); rat_t.append(m / r['tuned'])
    gm = lambda v: float(np.exp(np.mean(np.log(v))))
    return dict(cfg=k, solved=solved, total=len(tasks) * len(seeds), vs_default=gm(rat_d), vs_tuned=gm(rat_t), per=dict(zip([t.prob for t in tasks], [round(x, 2) for x in rat_d])))


BASE = dict(P=40, B=16, cull=0.8, init=0.5)                          # series-2 winner (noisy XOR) as the starting point
CONFIGS = [dict(BASE, cull=0.9, B=8), dict(BASE, cull=0.9, B=8, mirror=True), dict(BASE, cull=0.9, mirror=True), dict(BASE, cull=0.9, P=80, B=8), dict(BASE, cull=0.9, P=80, B=8, mirror=True), dict(BASE, cull=0.95, P=80, B=8),
           dict(BASE, cull=0.95), dict(BASE, cull=0.9, P=160, B=4), dict(BASE, cull=0.95, P=160, B=4), dict(BASE, cull=0.95, P=80, B=8, mirror=True)]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=10); ap.add_argument('--budget', type=float, default=3e6); a = ap.parse_args()
    store = json.load(open(RES)) if os.path.exists(RES) else {}; tasks = [NetTask(p) for p in PROBLEMS]; seeds = list(range(a.seeds))
    references(tasks, store, seeds)
    for t in tasks: r = store['ref'][t.prob]; print(f'{t.prob:9s} n={t.n:4d} target {t.target:.3f} | default Adam {r["default"]:>11,.0f} | tuned Adam (lr {r["tuned_lr"]}) {r["tuned"]:>9,.0f}', flush=True)
    for cfg in CONFIGS:
        t0 = time.time(); s = suite(tasks, store, cfg, seeds, a.budget)
        print(f'{s["cfg"]}: solved {s["solved"]}/{s["total"]} | geometric-mean cost {s["vs_default"]:.2f}x default Adam, {s["vs_tuned"]:.1f}x tuned Adam | per problem vs default {s["per"]} | {time.time() - t0:.0f}s', flush=True)
