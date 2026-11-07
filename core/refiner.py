"""refiner.py: digger + POOLED REFINEMENT (author 2026-11-03: no arbitrary stages; formulas of ever-increasing accuracy that make things simple).
One continuous rule, no schedule: after a problem links its functions, every function it touched (and the functions those are built from) is re-fitted on ALL
the stored problems that use it, jointly (each problem keeps its own readout, refit exactly every step); the new version replaces the old only if pooled
decision-point error drops, and the pass repeats until nothing improves (convergence, not a fixed count). So every formula only ever gets more accurate, and
what it learns from many problems at once is what none of them shows alone. cfg['pool'] = 'local' (touched functions + ancestors) or 'all' (every function).
Exam: exam(problems) freezes the library and only fits readouts on existing functions: how much of the WHOLE environment does the library explain."""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import numpy as np, torch
import digger
from digger import Cache, all_act, forward, node_input, nparams

torch.set_num_threads(1)


def tfwd(nd, z):
    if nd[0] == "f": return torch.tanh(z @ nd[1][:-1] + nd[1][-1])
    k = z.shape[1]; th = nd[1]; H = (th.shape[0] - 1) // (k + 2)
    W1 = th[:k * H].reshape(k, H); b1 = th[k * H:k * H + H]; w2 = th[k * H + H:k * H + 2 * H]; return torch.tanh(torch.tanh(z @ W1 + b1) @ w2 + th[-1])


class Learner(digger.Learner):
    def __init__(self, seed, cfg=None):
        super().__init__(seed, dict(dict(pool="off"), **(cfg or {})))

    def train(self, E, c):
        super().train(E, c)
        if self.cfg["pool"] != "off": self.pooled(E, c)

    def closure(self, E):
        U, stack = set(), list(E)
        while stack:
            i = stack.pop()
            if i in U: continue
            U.add(i); stack += list(self.units[i][2] if self.units[i][0] == "s" else ())
        return U

    def pooled(self, E, c):
        cfg, units = self.cfg, self.units; rid = cfg["ridge"]
        U = set(range(len(units))) if cfg["pool"] == "all" else self.closure(E)
        P = [(self.caches[p], self.heads[p]) for p in range(len(self.heads)) if (cfg["pool"] == "all" or set(self.heads[p]) & U)] + [(c, list(E))]
        if len(P) < 2 and cfg["pool"] != "all": return
        order = sorted(U); first = order[0]
        par = {i: torch.tensor(digger.pvec(units[i]), dtype=torch.float64, requires_grad=True) for i in order}
        tx = [(torch.tensor(cx.xtr), torch.tensor(cx.ytr), torch.tensor(cx.xva), torch.tensor(cx.yva), torch.tensor(cx.tr), torch.tensor(cx.va)) for cx, _ in P]
        meta = [(units[i][0], units[i][2] if units[i][0] == "s" else ()) for i in range(len(units))]

        def run(prm, train):
            tot, outs = 0.0, []
            for (cx, Ep), (Xt, Yt, Xv, Yv, Ct, Cv) in zip(P, tx):
                X, Y, Cc = (Xt, Yt, Ct) if train else (Xv, Yv, Cv); memo = {}

                def get(i):
                    if i not in prm: return Cc[:, i]
                    if i not in memo:
                        kind, pa = meta[i]; z = X if (kind == "f" or not pa) else torch.cat([X, torch.stack([get(q) for q in pa], 1)], 1)
                        memo[i] = tfwd((kind, prm[i]), z)
                    return memo[i]
                outs.append((Ep, [get(i) for i in Ep], Y))
            return outs

        def beta_fit():
            bs = []
            for Ep, hs, Y in run({i: p.detach() for i, p in par.items()}, True):
                F = torch.cat([torch.ones(len(Y), 1, dtype=torch.float64), torch.stack(hs, 1)], 1) if Ep else torch.ones(len(Y), 1, dtype=torch.float64)
                bs.append(torch.linalg.solve(F.T @ F + rid * torch.eye(F.shape[1], dtype=torch.float64), F.T @ Y))
            return bs

        def pooled_val(prm, bs):
            s = 0.0
            for b, (Ep, hs, Y) in zip(bs, run(prm, False)):
                F = torch.cat([torch.ones(len(Y), 1, dtype=torch.float64), torch.stack(hs, 1)], 1) if Ep else torch.ones(len(Y), 1, dtype=torch.float64)
                s = s + torch.mean((F @ b - Y) ** 2)
            return float(s)

        bs = beta_fit(); best = pooled_val({i: p.detach() for i, p in par.items()}, bs); bestP = {i: p.detach().clone() for i, p in par.items()}; since = 0
        opt = torch.optim.Adam(list(par.values()), lr=cfg["ft_lr"])
        for s in range(cfg["ft_steps"]):
            bs = beta_fit(); loss = 0.0
            for b, (Ep, hs, Y) in zip(bs, run(par, True)):
                F = torch.cat([torch.ones(len(Y), 1, dtype=torch.float64), torch.stack(hs, 1)], 1) if Ep else torch.ones(len(Y), 1, dtype=torch.float64)
                loss = loss + torch.mean((F @ b - Y) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            v = pooled_val({i: p.detach() for i, p in par.items()}, beta_fit())
            if v < best - 1e-12: best, bestP, since = v, {i: p.detach().clone() for i, p in par.items()}, 0
            else:
                since += 1
                if since >= cfg["ft_patience"]: break
        for i in order: units[i] = digger.rebuild(units[i], bestP[i].numpy())
        for cx in self.caches: cx.refresh(units, first)

    def exam(self, problems, n_lab=60):
        """Frozen library: link existing functions only (no new ones, no training); readout on n_lab random labels; returns per-problem (R^2, links)."""
        out = []
        for d in problems:
            self.m = d["tr"][0].shape[1]; idx = self.rng.choice(len(d["tr"][1]), n_lab, replace=False); xtr, ytr = d["tr"][0][idx], d["tr"][1][idx]
            c = Cache(xtr, ytr, d["va"][0], d["va"][1], self.units); E, _, _ = self.link_step(c, [], len(self.heads), False)
            r = self.score(E, d, xtr, ytr); out.append((r["r2"], len(E)))
        return out
