"""Generational self-scored evolution of growing networks (see DESIGN.md).

python selfscore/run.py --problem parity4 --rule grow_and_pred --sched full --seed 0 [--gens 2000]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.pop import Pop, dev
import car.benchmarks as BM
import car.compare as CP

# (pass mask rule, picker) -------------------------------------------------------------------------------------------------------
def _rules(Qk, Qp, Ck, Cp, Ak, Ap):
    return {
        'ext': Ak > Ap,
        'pred': Qk >= Qp,
        'grow': Ck > Cp,
        'grow_and_pred': (Ck > Cp) & (Qk >= Qp),
        'nonreg': (Ck >= Cp) & (Qk >= Qp) & ((Ck > Cp) | (Qk > Qp)),
        'pred_pickC': Qk >= Qp,
        'grow_pickQ': Ck > Cp,
    }
PICK = {'pred_pickC': 'C', 'grow_pickQ': 'Q'}          # default picker is random among passers


def p_feedback(sched, g, gens):
    h = gens / 2
    return {'full': 1.0, 'linear': max(0.0, 1 - g / h), 'step': 1.0 if g < h else 0.0, 'none': 0.0}[sched]


def load(problem):
    X, y, Xt, yt, K, kind = CP.problem(problem, 0)
    enc = BM.Encoder(X, kind)
    f = lambda A: torch.tensor(enc.bits(A) * 2.0 - 1.0, dtype=torch.float32, device=dev)
    return f(X), torch.tensor(y, device=dev), f(Xt), torch.tensor(yt, device=dev), K


@torch.no_grad()
def run(problem, rule, sched, seed, gens=2000, P=256, m=4, B=512, H=32, eps=0.2, Rsz=4096, Rs=256, log=print, every=50):
    torch.manual_seed(seed); np.random.seed(seed)
    X, y, Xt, yt, K = load(problem); d = X.shape[1]
    pop = Pop(P, d, K, H)
    rx = torch.zeros(Rsz, d, device=dev); ra = torch.zeros(Rsz, dtype=torch.long, device=dev); rf = torch.zeros(Rsz, device=dev); rn = 0; rptr = 0
    hist = []; t0 = time.time(); passrate = 0.0
    for g in range(gens):
        pfb = p_feedback(sched, g, gens)
        kids = pop.take(torch.arange(P, device=dev).repeat(m)); kids.mutate()         # kid i*? -> parent (i % P)
        allp = Pop.cat(pop, kids)
        idx = torch.randint(0, len(X), (B,), device=dev); xb, yb = X[idx], y[idx]
        q = allp.forward(xb)                                                           # (Pt,B,K)
        explore = torch.rand(B, device=dev) < eps; ra_rand = torch.randint(0, K, (B,), device=dev)
        a_greedy = q.argmax(2); a = torch.where(explore[None, :], ra_rand[None, :], a_greedy)   # same exploration noise for every genome
        fb = (a == yb[None, :]).float()                                                # feedback stimulus for the action each genome took
        pq = q.gather(2, a[..., None])[..., 0].clamp(1e-4, 1 - 1e-4)
        ll = fb * torch.log(pq) + (1 - fb) * torch.log(1 - pq)
        reveal = torch.rand(B, device=dev) < pfb
        num = (ll * reveal[None, :]).sum(1); den = float(reveal.sum())
        if rn > 0:                                                                     # the network scores itself on its own stored memory too
            ridx = torch.randint(0, rn, (Rs,), device=dev)
            qr = allp.forward(rx[ridx]); pr = qr.gather(2, ra[ridx][None, :, None].expand(allp.P, -1, 1))[..., 0].clamp(1e-4, 1 - 1e-4)
            llr = rf[ridx][None, :] * torch.log(pr) + (1 - rf[ridx][None, :]) * torch.log(1 - pr)
            num = num + llr.sum(1); den += Rs
        Q = num / max(den, 1.0) if den > 0 else torch.zeros(allp.P, device=dev)
        acc = (a_greedy == yb[None, :]).float().mean(1)                                # external accuracy (only `ext` may select on it)
        C, Cn, Ce = allp.complexity()
        Qp, Qk = Q[:P], Q[P:].view(m, P); Cp, Ck = C[:P], C[P:].view(m, P); Ap, Ak = acc[:P], acc[P:].view(m, P)
        passm = _rules(Qk, Qp[None], Ck, Cp[None], Ak, Ap[None])[rule]                 # (m,P) HARD pass/fail child vs its own parent
        pick = PICK.get(rule, 'rand')
        score = torch.rand(m, P, device=dev)
        if pick == 'C': score = Ck + 1e-3 * score
        elif pick == 'Q': score = Qk + 1e-6 * score
        score = torch.where(passm, score + 1e6, torch.full_like(score, -1e9))
        best = score.argmax(0); anyp = passm.any(0)
        sel = torch.where(anyp, P + best * P + torch.arange(P, device=dev), torch.arange(P, device=dev))
        pop = allp.take(sel); passrate = float(passm.float().mean())
        # store fresh revealed stimuli (from the parents' own actions) into the memory
        rv = torch.nonzero(reveal)[:, 0]
        if len(rv):
            take = rv[:64]; n = len(take); pos = (torch.arange(n, device=dev) + rptr) % Rsz
            rx[pos] = xb[take]; ra[pos] = a[0, take]; rf[pos] = fb[0, take]; rptr = (rptr + n) % Rsz; rn = min(Rsz, rn + n)
        if g % every == 0 or g == gens - 1:
            qt = pop.forward(Xt); tacc = (qt.argmax(2) == yt[None, :]).float().mean(1)                  # held-out accuracy of the greedy action (reporting only)
            Cc, Cn2, Ce2 = pop.complexity()
            Qsel = Q[:P] if False else None
            rec = dict(g=g, pfb=pfb, acc_mean=float(tacc.mean()), acc_med=float(tacc.median()), acc_max=float(tacc.max()), C_mean=float(Cc.mean()), C_max=float(Cc.max()),
                       nodes=float(Cn2.mean()), edges=float(Ce2.mean()), passrate=passrate, replay=rn, sec=time.time() - t0)
            if tacc.std() > 0 and Cc.std() > 0: rec['corr_C_acc'] = float(torch.corrcoef(torch.stack([Cc, tacc]))[0, 1])
            hist.append(rec)
            log(f"{problem} {rule} {sched} s{seed} g{g:5d} pfb {pfb:.2f} acc mean {rec['acc_mean']:.3f} max {rec['acc_max']:.3f} C {rec['C_mean']:.1f} (nodes {rec['nodes']:.1f} edges {rec['edges']:.1f}) pass {passrate:.3f} [{rec['sec']:.0f}s]")
    return hist


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', default='parity4'); ap.add_argument('--rule', default='pred'); ap.add_argument('--sched', default='full')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--gens', type=int, default=2000); ap.add_argument('--P', type=int, default=256)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    h = run(a.problem, a.rule, a.sched, a.seed, a.gens, P=a.P)
    if a.out: json.dump(h, open(a.out, 'w'))
