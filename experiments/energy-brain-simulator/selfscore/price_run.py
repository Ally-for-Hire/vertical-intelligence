"""Instrumented real-network cull runs that measure every term of the accounting identity and every prediction of the theory (THEORY.md).

True accuracy z_i of an individual is computed EXACTLY (all patterns of a Boolean problem; deterministic networks with no hidden units, so no sampling noise in z).
The cull ranks by a NOISY score: accuracy on a random batch of Be patterns drawn with replacement (so Var(score | z) = z(1-z)/Be exactly).
Each generation records: z-bar, sd(z), the survivors' selection differential S, the weighted differential S_w, the transmission term T_kids, the realised rank correlation rho,
the next generation's z-bar, and the identity residual  (zbar' - zbar) - (S_w + T_kids)  which must be zero.

python selfscore/price_run.py [--problem and_or10] [--gens 60]
"""
import os, sys, json, argparse, itertools, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.pop import Pop, dev
from selfscore.run import load


@torch.no_grad()
def run(problem, P, k, Be, gens, seed, slight=0.1, keepI=False):
    torch.manual_seed(seed)
    X, y, Xt, yt, K = load(problem); d = X.shape[1]
    pop = Pop(P, d, K, 4, init_edges=0, stochastic=False)
    ok_all = lambda p: (p.forward(X).argmax(2) == y[None, :]).float()                    # (P, n_patterns) correctness indicators
    recs = []
    for g in range(gens):
        I = ok_all(pop); z = I.mean(1)                                                    # EXACT true accuracy of each individual
        idx = torch.randint(0, X.shape[0], (Be,), device=dev); s = I[:, idx].mean(1)      # the cull's noisy score
        order = torch.argsort(s + 1e-9 * torch.rand(P, device=dev), descending=True); keep = order[:k]
        zbar, sd = float(z.mean()), float(z.std(unbiased=False))
        S = float(z[keep].mean() - z.mean())
        # breed: survivors + (P-k) slightly randomised copies cycled over the survivors
        nk = P - k; parent_of_kid = keep[torch.arange(nk, device=dev) % k]
        kids = pop.take(parent_of_kid); kids._weight(torch.arange(nk, device=dev), slight)
        zk = ok_all(kids).mean(1); dz = zk - z[parent_of_kid]
        newpop = Pop.cat(pop.take(keep), kids)
        znew = ok_all(newpop).mean(1); zbar_next = float(znew.mean())
        # weighted selection differential: survivor j appears once + (number of kids it parented) times in the new population
        cnt = torch.zeros(P, device=dev); cnt.scatter_add_(0, parent_of_kid, torch.ones(nk, device=dev)); cnt[keep] += 1
        Sw = float((cnt * z).sum() / P - z.mean()); Tk = float(dz.sum() / P)
        rho = float(torch.corrcoef(torch.stack([z, s]))[0, 1]) if sd > 1e-9 and float(s.std()) > 1e-9 else float('nan')
        sig_e2 = float((z * (1 - z)).mean() / Be)
        recs.append(dict(g=g, zbar=zbar, sd=sd, S=S, Sw=Sw, Tk=Tk, mu_m=float(dz.mean()), sd_m=float(dz.std(unbiased=False)), rho=rho, sig_e2=sig_e2,
                         zbar_next=zbar_next, resid=(zbar_next - zbar) - (Sw + Tk), z=[round(v, 5) for v in z.tolist()], **({'I': I.cpu().numpy().astype(bool)} if keepI else {})))
        pop = newpop
    return recs


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--problem', default='and_or10'); ap.add_argument('--gens', type=int, default=60)
    ap.add_argument('--seeds', type=int, default=4); ap.add_argument('--out', default='')
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', f'price_{a.problem}.json')
    res = []
    for P, Be in itertools.product((64, 256), (32, 128, 1024)):
        for c in (0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 'one'):
            k = 1 if c == 'one' else max(1, P - int(round(c * P)))
            for sd in range(a.seeds):
                res.append(dict(P=P, Be=Be, c=c, k=k, seed=sd, recs=run(a.problem, P, k, Be, a.gens, 1000 * sd + P + Be)))
        print(f'done P={P} Be={Be}', flush=True)
    json.dump(res, open(out, 'w')); print('saved', out, len(res), 'runs')
