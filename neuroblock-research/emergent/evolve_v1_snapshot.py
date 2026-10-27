"""Emergent neural-net behaviour from pure random weights and biases + internal randomness + external culling (see README.md; every attempt is in LOG.md).
One problem: xorgauss (noisy XOR, raw 2-D inputs; linear models are at chance 0.50, best possible 0.955). Success: best brain (picked on TRAINING accuracy) >= 0.90 TEST accuracy.

python emergent/evolve.py                     # the whole comparison: evolution vs random search vs gradient descent, 5 seeds each -> emergent/RESULTS.md
python emergent/evolve.py --only evolve --seeds 0   (one method, one seed)
Methods
  evolve   : P brains with pure random weights/biases N(0,1); each generation every brain is scored on the training set, the worst `cull` fraction die (external culling),
             and each empty slot gets a copy of a uniformly random survivor plus random noise on every weight and bias (internal randomness). Each brain carries its own
             noise size sigma, inherited with a random log-normal change (self-adaptation, learning rate tau = 1/sqrt(number of parameters), the textbook ES value); starting
             sigmas are random (log-uniform 0.01..1). Survivors are not changed.
  random   : the same random brains, no culling: draw P new random brains per round, keep the best ever (pure random search, same evaluation budget).
  gradient : one normal net of the same size trained with Adam (PyTorch defaults), batches of 32; counted in training samples seen.
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import car.compare as CP

HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, 'runs'); os.makedirs(RUNS, exist_ok=True)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if dev.type == 'cuda': torch.cuda.set_per_process_memory_fraction(0.3)
BAYES = 0.955


def data():
    X, y, Xt, yt, K, kind = CP.problem('xorgauss', 0)
    f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
    return f(X), torch.tensor(y, device=dev), f(Xt), torch.tensor(yt, device=dev), K


def nparams(d, H, K): return d * H + H + H * K + K


def forward(W1, b1, W2, b2, X):
    """P networks at once: (P, rows, K) logits"""
    return torch.tanh(X @ W1 + b1[:, None, :]) @ W2 + b2[:, None, :]


def accs(W1, b1, W2, b2, X, y): return (forward(W1, b1, W2, b2, X).argmax(2) == y[None]).float().mean(1)


@torch.no_grad()
def run_evolve(a, seed, Xtr, ytr, Xte, yte, K):
    g = torch.Generator(device=dev).manual_seed(seed); rn = lambda *s: torch.randn(*s, device=dev, generator=g); ru = lambda *s: torch.rand(*s, device=dev, generator=g)
    P, H, d = a.P, a.H, Xtr.shape[1]; n = nparams(d, H, K); tau = 1.0 / math.sqrt(n)
    W1, b1, W2, b2 = rn(P, d, H), rn(P, H), rn(P, H, K), rn(P, K)                      # pure random weights and biases
    sig = 10 ** (-2 + 2 * ru(P))                                                        # random starting noise sizes, log-uniform 0.01..1
    evals, hist, t0 = 0, [], time.time(); nk = max(1, int(round(P * (1 - a.cull))))
    for gen in range(a.max_gens):
        tr = accs(W1, b1, W2, b2, Xtr, ytr); evals += P
        order = torch.argsort(tr + 1e-6 * ru(P), descending=True); best = order[0]
        te = float(accs(W1[best:best + 1], b1[best:best + 1], W2[best:best + 1], b2[best:best + 1], Xte, yte)[0])
        hist.append(dict(gen=gen, evals=evals, train_best=float(tr[best]), train_mean=float(tr.mean()), test_best=te, sigma_median=float(sig.median())))
        if gen % a.every == 0: print(f'  evolve seed {seed} gen {gen:5d} | evals {evals:9d} | train best {float(tr[best]):.3f} mean {float(tr.mean()):.3f} | TEST of best {te:.3f} | sigma median {float(sig.median()):.3f}', flush=True)
        if te >= a.target: return dict(method='evolve', seed=seed, solved=True, evals=evals, gens=gen + 1, test=te, sec=time.time() - t0, hist=hist)
        keep = order[:nk]; par = keep[torch.randint(0, nk, (P - nk,), device=dev, generator=g)]          # external culling; parents uniformly random among survivors
        cs = sig[par] * torch.exp(tau * rn(P - nk))                                                         # inherited, randomly changed noise size
        nw = lambda T: T[par] + cs.view(-1, *([1] * (T.dim() - 1))) * rn(*T[par].shape)                  # child = parent + random noise
        W1, b1, W2, b2 = (torch.cat([T[keep], nw(T)]) for T in (W1, b1, W2, b2)); sig = torch.cat([sig[keep], cs])
    return dict(method='evolve', seed=seed, solved=False, evals=evals, gens=a.max_gens, test=hist[-1]['test_best'], sec=time.time() - t0, hist=hist)


@torch.no_grad()
def run_random(a, seed, Xtr, ytr, Xte, yte, K):
    g = torch.Generator(device=dev).manual_seed(10_000 + seed); rn = lambda *s: torch.randn(*s, device=dev, generator=g)
    P, H, d = a.P, a.H, Xtr.shape[1]; best_tr, best_te, evals, t0 = -1.0, 0.0, 0, time.time(); hist = []
    for rnd in range(a.max_gens):
        W1, b1, W2, b2 = rn(P, d, H), rn(P, H), rn(P, H, K), rn(P, K); tr = accs(W1, b1, W2, b2, Xtr, ytr); evals += P
        i = int(tr.argmax())
        if float(tr[i]) > best_tr: best_tr = float(tr[i]); best_te = float(accs(W1[i:i + 1], b1[i:i + 1], W2[i:i + 1], b2[i:i + 1], Xte, yte)[0])
        hist.append(dict(gen=rnd, evals=evals, train_best=best_tr, test_best=best_te))
        if rnd % a.every == 0: print(f'  random seed {seed} round {rnd:5d} | evals {evals:9d} | best ever: train {best_tr:.3f} TEST {best_te:.3f}', flush=True)
        if best_te >= a.target: return dict(method='random', seed=seed, solved=True, evals=evals, gens=rnd + 1, test=best_te, sec=time.time() - t0, hist=hist)
    return dict(method='random', seed=seed, solved=False, evals=evals, gens=a.max_gens, test=best_te, sec=time.time() - t0, hist=hist)


def run_gradient(a, seed, Xtr, ytr, Xte, yte, K):
    torch.manual_seed(20_000 + seed); d = Xtr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, a.H), torch.nn.Tanh(), torch.nn.Linear(a.H, K)).to(dev); opt = torch.optim.Adam(net.parameters()); t0 = time.time(); hist = []
    for step in range(1, a.max_steps + 1):
        i = torch.randint(0, len(Xtr), (32,), device=dev); loss = torch.nn.functional.cross_entropy(net(Xtr[i]), ytr[i]); opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            with torch.no_grad(): te = float((net(Xte).argmax(1) == yte).float().mean())
            hist.append(dict(step=step, samples=step * 32, test=te))
            if te >= a.target: return dict(method='gradient', seed=seed, solved=True, steps=step, samples=step * 32, test=te, sec=time.time() - t0, hist=hist)
    return dict(method='gradient', seed=seed, solved=False, steps=a.max_steps, samples=a.max_steps * 32, test=hist[-1]['test'], sec=time.time() - t0, hist=hist)


def results_md(a, res):
    L = [f'# Results: {a.tag}', '', f'Problem xorgauss (linear models: 0.50, best possible {BAYES}); success = best brain >= {a.target} TEST accuracy (picked on training accuracy). '
         f'Brains: 2-{a.H}-2 tanh nets ({nparams(2, a.H, 2)} weights+biases). Evolution: P={a.P}, cull {a.cull:.0%} per generation. Budget: {a.max_gens} generations (= {a.max_gens * a.P:,} network evaluations).', '',
         '| method | seeds solved | network evaluations to success (median, range) | training samples processed (median) | test accuracy reached (median) |', '|---|---|---|---|---|']
    for m in ('evolve', 'random', 'gradient'):
        rs = [r for r in res if r['method'] == m]
        if not rs: continue
        ok = [r for r in rs if r['solved']]; med = lambda v: float(np.median(v)) if v else float('nan')
        if m == 'gradient':
            L.append(f'| gradient descent (Adam) | {len(ok)}/{len(rs)} | - (1 network) | {med([r["samples"] for r in ok]):,.0f} samples, forward+backward | {med([r["test"] for r in rs]):.3f} |')
        else:
            ev = [r['evals'] for r in ok]
            L.append(f'| {"evolution (random noise + culling)" if m == "evolve" else "random search (no culling)"} | {len(ok)}/{len(rs)} | {med(ev):,.0f} ({min(ev) if ev else 0:,}-{max(ev) if ev else 0:,}) | '
                     f'{med(ev) * 2000:,.0f} samples, forward only | {med([r["test"] for r in rs]):.3f} |')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='baseline'); ap.add_argument('--only', default='', help='evolve | random | gradient (default: all three)')
    ap.add_argument('--seeds', default='0,1,2,3,4'); ap.add_argument('--P', type=int, default=1000); ap.add_argument('--H', type=int, default=8); ap.add_argument('--cull', type=float, default=0.5)
    ap.add_argument('--target', type=float, default=0.90); ap.add_argument('--max_gens', type=int, default=2000); ap.add_argument('--max_steps', type=int, default=200000); ap.add_argument('--every', type=int, default=50)
    a = ap.parse_args(); Xtr, ytr, Xte, yte, K = data(); res = []
    print(f'xorgauss: {len(Xtr)} train / {len(Xte)} test points, {K} classes; target test accuracy {a.target}', flush=True)
    for m in (['evolve', 'random', 'gradient'] if not a.only else a.only.split(',')):
        for sd in [int(s) for s in a.seeds.split(',')]:
            r = {'evolve': run_evolve, 'random': run_random, 'gradient': run_gradient}[m](a, sd, Xtr, ytr, Xte, yte, K); res.append(r)
            print(f'{m} seed {sd}: {"SOLVED" if r["solved"] else "not solved"} test {r["test"]:.3f} | ' + (f'{r["samples"]:,} samples' if m == 'gradient' else f'{r["evals"]:,} network evaluations, {r["gens"]} generations') + f' | {r["sec"]:.1f}s', flush=True)
    json.dump(dict(args=vars(a), results=res), open(os.path.join(RUNS, f'{a.tag}.json'), 'w'))
    md = results_md(a, res); open(os.path.join(HERE, f'RESULTS_{a.tag}.md'), 'w', encoding='utf-8').write(md); print(md)
