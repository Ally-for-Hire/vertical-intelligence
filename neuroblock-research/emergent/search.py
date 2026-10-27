"""Search for a culling/evolution method that reaches gradient-descent SPEED on xorgauss (user, 2026-10-26).
MAIN RULE (user): "nothing can be directly guided by task performance internally" -> task performance is used ONLY by the external cull (who lives, who dies). Every change inside a
net is blind random variation; no gradients, no keep-or-undo decisions inside a net, no performance-guided mutation. (The 'undo' and 'stack' rules of evolve.py break this rule.)

Speed = compute to reach 0.95 TEST accuracy (best possible 0.955), counted in sample-forward-equivalents: one training sample through one net forward = 1, a backward pass = 2 more.
Gradient descent (Adam, batches of 32, same 2-H-2 net) is the bar, measured here with the same counting. Success for a method = 5/5 seeds reach 0.95 and the median cost <= gradient's.

python emergent/search.py            (runs the configurations in CONFIGS that are not yet in emergent/runs/search_results.json; appends to emergent/SEARCH.md)
"""
import os, sys, json, time, math, argparse, itertools
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emergent.evolve import data, nparams, split, accs, forward, dev


def score(th, d, H, K, X, y, kind):
    """EXTERNAL score used only by the cull: 'acc' = fraction correct; 'll' = mean log-probability of the correct answer (softmax of the outputs): a smooth, less noisy grade of the same task performance"""
    if kind == 'acc': return accs(*split(th, d, H, K), X, y)
    lp = torch.log_softmax(forward(*split(th, d, H, K), X), 2); return lp.gather(2, y[None, :, None].expand(lp.shape[0], -1, 1))[..., 0].mean(1)

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, 'runs', 'search_results.json'); TARGET = 0.95


def gradient_cost(seed, Xtr, ytr, Xte, yte, K, H=8, lr=1e-3, bs=32, max_steps=200000):
    torch.manual_seed(20_000 + seed); net = torch.nn.Sequential(torch.nn.Linear(2, H), torch.nn.Tanh(), torch.nn.Linear(H, K)).to(dev); opt = torch.optim.Adam(net.parameters(), lr=lr)
    for step in range(1, max_steps + 1):
        i = torch.randint(0, len(Xtr), (bs,), device=dev); l = torch.nn.functional.cross_entropy(net(Xtr[i]), ytr[i]); opt.zero_grad(); l.backward(); opt.step()
        if step % 5 == 0:
            with torch.no_grad():
                if float((net(Xte).argmax(1) == yte).float().mean()) >= TARGET: return 3 * step * bs, True
    return 3 * max_steps * bs, False


@torch.no_grad()
def evolve(cfg, seed, Xtr, ytr, Xte, yte, K, budget):
    """blind variation + external culling. cfg keys: P (population), B (training samples each net is scored on per generation; shared batch), cull (fraction that dies),
    sel (trunc | tourn), noise (gauss | cauchy), pm (fraction of parameters a child's mutation touches), recomb (none | avg2), adapt (self-adaptive sigma), H (hidden units), init (scale of random start)"""
    g = torch.Generator(device=dev).manual_seed(seed); rn = lambda *s: torch.randn(*s, device=dev, generator=g); ru = lambda *s: torch.rand(*s, device=dev, generator=g)
    P, B, H = cfg['P'], cfg['B'], cfg.get('H', 8); d = Xtr.shape[1]; n = nparams(d, H, K); tau = 1.0 / math.sqrt(n)
    th = rn(P, n) * cfg.get('init', 1.0); sig = 10 ** (-2 + 2 * ru(P)); cost = 0; ssum = torch.zeros(P, device=dev); scnt = torch.zeros(P, device=dev); nk = max(1, int(round(P * (1 - cfg['cull'])))); gen = 0
    while cost < budget:
        i = torch.randint(0, len(Xtr), (B,), device=dev, generator=g); acc = score(th, d, H, K, Xtr[i], ytr[i], cfg.get('fit', 'acc')); cost += P * B
        if cfg.get('life', False): ssum += acc; scnt += 1; acc = ssum / scnt      # external score = average over every batch the net has been scored on during its life (reuses evaluations)
        order = torch.argsort(acc + 1e-6 * ru(P), descending=True); best = order[0]
        if float(accs(*split(th[best:best + 1], d, H, K), Xte, yte)[0]) >= TARGET: return cost, True, gen + 1      # monitoring only (not counted, not used by evolution)
        if cfg.get('sel', 'trunc') == 'trunc': keep = order[:nk]; pa = keep[torch.randint(0, nk, (P - nk,), device=dev, generator=g)]; pb = keep[torch.randint(0, nk, (P - nk,), device=dev, generator=g)]
        else:                                                                                   # tournament of 2 for each parent; the losers of the cull are the worst nk..P
            keep = order[:nk]; c1 = keep[torch.randint(0, nk, (2, P - nk), device=dev, generator=g)]; pa = torch.where(acc[c1[0]] >= acc[c1[1]], c1[0], c1[1])
            c2 = keep[torch.randint(0, nk, (2, P - nk), device=dev, generator=g)]; pb = torch.where(acc[c2[0]] >= acc[c2[1]], c2[0], c2[1])
        rc = cfg.get('recomb', 'none')
        child = (th[pa] + th[pb]) / 2 if rc == 'avg2' else (th[keep].mean(0, keepdim=True).expand(P - nk, -1).clone() if rc == 'centroid' else th[pa].clone())      # blind recombination: no use of scores
        cs = sig[pa] * torch.exp(tau * rn(P - nk)) if cfg.get('adapt', True) else sig[pa]
        z = rn(P - nk, n) if cfg.get('noise', 'gauss') == 'gauss' else torch.tan(math.pi * (ru(P - nk, n) - 0.5)).clamp(-50, 50)
        if cfg.get('pm', 1.0) < 1.0: z = z * (ru(P - nk, n) < cfg['pm'])
        child += cs[:, None] * z
        th = torch.cat([th[keep], child]); sig = torch.cat([sig[keep], cs]); gen += 1
        z0 = torch.zeros(P - nk, device=dev); ssum = torch.cat([ssum[keep], z0]); scnt = torch.cat([scnt[keep], z0])
    return cost, False, gen


def key(cfg): return json.dumps(cfg, sort_keys=True)


CONFIGS = []
for H, P, B, cull, noise in itertools.product((8, 32), (10, 20), (32, 64), (0.8, 0.9), ('gauss', 'cauchy')): CONFIGS.append(dict(P=P, B=B, cull=cull, life=True, fit='ll', H=H, recomb='none', noise=noise))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--budget', type=float, default=3e6); a = ap.parse_args()
    Xtr, ytr, Xte, yte, K = data(); done = json.load(open(RES)) if os.path.exists(RES) else {}
    if 'gradient' not in done:
        gc = [gradient_cost(s, Xtr, ytr, Xte, yte, K) for s in range(a.seeds)]; done['gradient'] = dict(costs=[c for c, _ in gc], solved=sum(o for _, o in gc)); json.dump(done, open(RES, 'w'))
    for Hg in (16, 32):                                                                       # gradient descent at the other widths, so a wider evolved net is compared with an equally wide trained net
        if f'gradient_H{Hg}' not in done:
            gc = [gradient_cost(s, Xtr, ytr, Xte, yte, K, H=Hg) for s in range(a.seeds)]; done[f'gradient_H{Hg}'] = dict(costs=[c for c, _ in gc], solved=sum(o for _, o in gc)); json.dump(done, open(RES, 'w'))
        print(f'gradient descent, H={Hg}: median cost {float(np.median(done[f"gradient_H{Hg}"]["costs"])):,.0f}', flush=True)
    gmed = float(np.median(done['gradient']['costs'])); print(f'gradient descent: median cost {gmed:,.0f} sample-forward-equivalents ({done["gradient"]["solved"]}/{a.seeds} solved)', flush=True)
    for cfg in CONFIGS:
        k = key(cfg)
        if k in done: continue
        t0 = time.time(); rs = [evolve(cfg, s, Xtr, ytr, Xte, yte, K, a.budget) for s in range(a.seeds)]
        done[k] = dict(cfg=cfg, costs=[c for c, _, _ in rs], solved=sum(o for _, o, _ in rs), gens=[gg for _, _, gg in rs], sec=time.time() - t0); json.dump(done, open(RES, 'w'))
        ok = [c for c, o, _ in rs if o]; med = float(np.median([c for c, _, _ in rs])) if rs else float('nan')
        gw = float(np.median(done['gradient' if cfg.get('H', 8) == 8 else f'gradient_H{cfg["H"]}']['costs'])) if (cfg.get('H', 8) == 8 or f'gradient_H{cfg["H"]}' in done) else gmed
        print(f'{k}: solved {len(ok)}/{a.seeds} | median cost {med:,.0f} ({med / gmed:.2f}x gradient at H=8, {med / gw:.2f}x gradient at the same width) | {time.time() - t0:.1f}s', flush=True)
