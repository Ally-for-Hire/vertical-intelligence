"""Self-grown populations with a hard 50% cull at the actual problem (user's design, 2026-10-26).

Everything evolves in parallel and every decision is HARD pass/fail:
  INNER LIFE (per individual, parallel): random structural changes; a change is kept iff it passes the inner rule (primary: the network GREW).
        The cull never trains the individuals; they are trained only on their own growth (optionally gated by not regressing on their own stimulus prediction).
  OUTER CULL (population): every generation all individuals are tested on the actual problem; the bottom CULL fraction (50%) is pruned.
        Survivors are multiplied (each survivor + one slightly randomised copy) and evolution continues from them.
Accuracy is never an objective of the individuals; it can only appear through growth + culling.
Weaning: the cull may rank by internal score instead of external accuracy with probability lam(g) (ext -> internal over the run).

python selfscore/cull.py --problem parity4 --inner grow --rank ext --wean none --seed 0 --gens 2000
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.pop import Pop, dev
from selfscore.run import load

INNER = {  # hard pass/fail of a child against its own parent (growth is the primary criterion)
    'none': None,
    'grow': lambda Qk, Qp, Ck, Cp: Ck > Cp,
    'grow_and_pred': lambda Qk, Qp, Ck, Cp: (Ck > Cp) & (Qk >= Qp),
    'pred': lambda Qk, Qp, Ck, Cp: Qk >= Qp,
    'nonreg': lambda Qk, Qp, Ck, Cp: (Ck >= Cp) & (Qk >= Qp) & ((Ck > Cp) | (Qk > Qp)),
}


class Mem:
    """the network's own memory of past stimuli (x, action taken, feedback received)"""
    def __init__(s, d, size=4096):
        s.x = torch.zeros(size, d, device=dev); s.a = torch.zeros(size, dtype=torch.long, device=dev); s.f = torch.zeros(size, device=dev)
        s.n = 0; s.ptr = 0; s.size = size
    def add(s, x, a, f):
        n = len(x); pos = (torch.arange(n, device=dev) + s.ptr) % s.size
        s.x[pos], s.a[pos], s.f[pos] = x, a, f; s.ptr = (s.ptr + n) % s.size; s.n = min(s.size, s.n + n)
    def sample(s, k):
        i = torch.randint(0, s.n, (k,), device=dev); return s.x[i], s.a[i], s.f[i]


def self_score(allp, xb, yb, mem, pfb, eps, Rs=256):
    """internal quality: log-likelihood of the network's prediction of its OWN feedback stimuli (fresh, revealed w.p. pfb, plus replay of its memory)"""
    B, K = xb.shape[0], allp.K
    q = allp.forward(xb)
    explore = torch.rand(B, device=dev) < eps; ar = torch.randint(0, K, (B,), device=dev)
    ag = q.argmax(2); a = torch.where(explore[None, :], ar[None, :], ag)
    fb = (a == yb[None, :]).float(); pq = q.gather(2, a[..., None])[..., 0].clamp(1e-4, 1 - 1e-4)
    ll = fb * torch.log(pq) + (1 - fb) * torch.log(1 - pq)
    reveal = torch.rand(B, device=dev) < pfb
    num = (ll * reveal[None, :]).sum(1); den = float(reveal.sum())
    if mem.n > 0:
        rx, ra, rf = mem.sample(Rs); qr = allp.forward(rx)
        pr = qr.gather(2, ra[None, :, None].expand(allp.P, -1, 1))[..., 0].clamp(1e-4, 1 - 1e-4)
        num = num + (rf[None] * torch.log(pr) + (1 - rf[None]) * torch.log(1 - pr)).sum(1); den += Rs
    Q = num / max(den, 1.0)
    rv = torch.nonzero(reveal)[:, 0][:64]
    if len(rv): mem.add(xb[rv], a[0, rv], fb[0, rv])
    return Q, (ag == yb[None, :]).float().mean(1)


def lam_sched(wean, g, gens):
    h = gens / 2
    return {'none': 0.0, 'linear': min(1.0, g / h) if g < h else 1.0, 'step': 0.0 if g < h else 1.0, 'all': 1.0}[wean]


@torch.no_grad()
def run(problem, inner, rank, wean, seed, gens=2000, P=384, k_inner=2, cull=0.5, B=256, Be=1024, H=32, eps=0.2, slight=0.1, pfb=1.0, stoch=False, S=4, log=print, every=50):
    torch.manual_seed(seed); np.random.seed(seed)
    X, y, Xt, yt, K = load(problem); d = X.shape[1]
    pop = Pop(P, d, K, H, stochastic=stoch, S=S); mem = Mem(d); rule = INNER[inner]
    hist = []; t0 = time.time(); ipass = 0.0
    for g in range(gens):
        # ---- INNER LIFE: parallel hard pass/fail on the individual's own growth (the cull plays no part here) -------------------------------------
        if rule is not None:
            for _ in range(k_inner):
                kid = pop.take(torch.arange(P, device=dev)); kid.mutate()
                allp = Pop.cat(pop, kid); idx = torch.randint(0, len(X), (B,), device=dev)
                Q, _ = self_score(allp, X[idx], y[idx], mem, pfb, eps)
                C, _, _ = allp.complexity()
                ok = rule(Q[P:], Q[:P], C[P:], C[:P]); ipass = float(ok.float().mean())
                pop = allp.take(torch.where(ok, P + torch.arange(P, device=dev), torch.arange(P, device=dev)))
        # ---- OUTER CULL at the actual problem: bottom `cull` fraction pruned, survivors multiplied and slightly randomised ----------------------------
        ie = torch.randint(0, len(X), (Be,), device=dev)
        ext = (pop.forward(X[ie]).argmax(2) == y[ie][None, :]).float().mean(1)                  # results at the actual problem
        lam = lam_sched(wean, g, gens)
        if rank == 'ext' or torch.rand(1).item() >= lam: score = ext
        elif rank == 'Q': score = self_score(pop, X[ie][:B], y[ie][:B], mem, pfb, eps)[0]
        elif rank == 'C': score = pop.complexity()[0]
        else: score = ext
        nkeep = int(round(P * (1 - cull)))
        order = torch.argsort(score + 1e-6 * torch.rand(P, device=dev), descending=True)
        keep = order[:nkeep]                                                                       # hard: the rest are removed
        surv = pop.take(keep)
        kids = pop.take(keep[torch.arange(P - nkeep, device=dev) % nkeep])
        kids._weight(torch.arange(kids.P, device=dev), slight)                                    # slight randomisation only
        sel_diff = float(ext[keep].mean() - ext.mean())
        pop = Pop.cat(surv, kids)
        if g % every == 0 or g == gens - 1:
            tacc = (pop.forward(Xt).argmax(2) == yt[None, :]).float().mean(1)
            if stoch:                                                                                  # also the raw single-pass behaviour (no averaging)
                S0 = pop.S; pop.S = 1; tacc1 = (pop.forward(Xt).argmax(2) == yt[None, :]).float().mean(1); pop.S = S0
            Cc, Cn, Ce = pop.complexity()
            rec = dict(g=g, lam=lam, acc_mean=float(tacc.mean()), acc_max=float(tacc.max()), ext_mean=float(ext.mean()), sel_diff=sel_diff, C_mean=float(Cc.mean()), C_max=float(Cc.max()),
                       nodes=float(Cn.mean()), edges=float(Ce.mean()), inner_pass=ipass, sec=time.time() - t0)
            if stoch: rec.update(acc1_mean=float(tacc1.mean()), beta_mean=float(pop.beta.mean()))
            hist.append(rec)
            log(f"{problem} inner={inner} rank={rank} wean={wean} s{seed} g{g:5d} test acc mean {rec['acc_mean']:.3f} max {rec['acc_max']:.3f}" + (f" (1-pass {rec['acc1_mean']:.3f}, beta {rec['beta_mean']:.2f})" if stoch else "") + f" | C {rec['C_mean']:.1f} (n {rec['nodes']:.1f} e {rec['edges']:.1f}) | inner pass {ipass:.2f} sel-diff {sel_diff:+.3f} [{rec['sec']:.0f}s]")
    return hist


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', default='parity4'); ap.add_argument('--inner', default='grow'); ap.add_argument('--rank', default='ext'); ap.add_argument('--wean', default='none')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--gens', type=int, default=2000); ap.add_argument('--P', type=int, default=384)
    ap.add_argument('--cull', type=float, default=0.5); ap.add_argument('--out', default=''); ap.add_argument('--stoch', action='store_true'); ap.add_argument('--S', type=int, default=4); ap.add_argument('--H', type=int, default=32)
    a = ap.parse_args()
    h = run(a.problem, a.inner, a.rank, a.wean, a.seed, a.gens, P=a.P, cull=a.cull, stoch=a.stoch, S=a.S, H=a.H)
    if a.out: json.dump(h, open(a.out, 'w'))
