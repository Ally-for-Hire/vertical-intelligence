"""Revision 4 (user, 2026-10-26): random-cull evolution of self-growing brains, scored by average ability across every problem, with total-wipe restarts.

* ONE population of self-growing brains (model v2: any-direction growth, evolvable mechanisms), no problem id: every problem has its own input slots, the others are zero.
* Each generation: the cull fraction c is drawn UNIFORMLY from [0, cull_max] (user: cap at 0.9, so the population is never wiped and there is never a true restart).
  Survivors = round(P*(1-c)); the bottom c is removed purely on performance (average accuracy over the epoch's tests).
* Individuals are trained only on themselves: inner life = their own random self-modification, kept iff the brain grew (hard pass/fail). The cull never trains them.
* Survivors repopulate the empty slots EVENLY; every copy gets a variation whose size is itself random (nudge one number ... re-randomise a whole module).
* OPTIONAL (off at the default cap of 0.9; kept for completeness): with --cull_max 1.0 a draw that leaves no survivor WIPES THE WHOLE POPULATION. The best candidates from just
  before the wipe become the starting population, the rest of the slots are filled with PURE RANDOM brand-new brains, and a NEW SET OF TESTS is drawn (a random subset of the
  24 problems, with fresh data samples). A new epoch begins.
* Everything is logged in plain English (story file), as an at-a-glance summary (markdown) and as JSON for analysis.

python selfscore/multi.py --tag s0 --seed 0 --gens 1500
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.pop2 import Pop2, dev
import car.benchmarks as BM
import car.compare as CP

HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, 'runs')
POOL = list(BM.PROBLEMS) + list(BM.HOLDOUT) + list(BM.EMERGE)                  # every problem in our data sets (24)


class Tests:
    """all problems on one shared input vector (disjoint slots); instances can be re-drawn per epoch; reference test sets stay fixed for reporting"""
    def __init__(s, pool, nref=300):
        s.pool = pool; s.enc, s.K, s.d, s.off, s.kind, s.gen, s.ref, s.train0, s.chance = {}, {}, {}, {}, {}, {}, {}, {}, {}
        o = 0
        for p in pool:
            X, y, Xt, yt, K, kind = CP.problem(p, 0)
            enc = BM.Encoder(X, kind); s.enc[p] = enc; s.K[p] = K; s.d[p] = enc.d; s.off[p] = o; o += enc.d; s.kind[p] = kind; s.chance[p] = 1.0 / K
            s.gen[p] = BM.PROBLEMS.get(p) or BM.HOLDOUT.get(p) or BM.EMERGE[p]
            sel = np.random.default_rng(5).permutation(len(Xt))[:nref]
            s.ref[p] = (s._bits(p, Xt[sel]), torch.tensor(yt[sel], device=dev)); s.train0[p] = (s._bits(p, X), torch.tensor(y, device=dev))
        s.D = o; s.Kmax = max(s.K.values())

    def _bits(s, p, A):
        return torch.tensor(s.enc[p].bits(A) * 2.0 - 1.0, dtype=torch.float32, device=dev)

    def instance(s, epoch):
        """fresh data samples for the epoch (continuous problems are re-generated; Boolean problems are fixed functions, so only the sampling differs)"""
        out = {}
        for i, p in enumerate(s.pool):
            if s.kind[p] == 'bits': out[p] = s.train0[p]
            else:
                X, y = s.gen[p](np.random.default_rng(1000 + epoch * 17 + i))[:2]
                out[p] = (s._bits(p, X), torch.tensor(y, device=dev))
        return out

    def full_x(s, p, bits):
        x = torch.zeros(len(bits), s.D, device=dev); x[:, s.off[p]:s.off[p] + s.d[p]] = bits; return x

    def batch(s, inst, active, per):
        xs, ys = [], []
        for p in active:
            X, y = inst[p]; i = torch.randint(0, len(X), (per,), device=dev); xs.append(s.full_x(p, X[i])); ys.append(y[i])
        return torch.cat(xs), torch.cat(ys)

    def kmask(s, active, per):
        m = torch.zeros(len(active), s.Kmax, dtype=torch.bool, device=dev)
        for j, p in enumerate(active): m[j, :s.K[p]] = True
        return m.repeat_interleave(per, 0)


@torch.no_grad()
def acc_by_problem(pop, x, y, kmask_rows, nact, per):
    q = pop.forward(x).masked_fill(~kmask_rows[None], -1.0)
    return (q.argmax(2) == y[None]).float().view(pop.P, nact, per).mean(2)                 # (P, n_active)


def bar(v, w=10):
    n = int(round(max(0, min(1, v)) * w)); return '#' * n + '.' * (w - n)


class Story:
    def __init__(s, path): s.f = open(path, 'a', encoding='utf-8')
    def __call__(s, m): print(m, flush=True); s.f.write(m + '\n'); s.f.flush()


@torch.no_grad()
def run(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.default_rng(a.seed)
    os.makedirs(RUNS, exist_ok=True)
    story = Story(os.path.join(RUNS, f'multi_{a.tag}.story.txt')); T = Tests(POOL)
    P = a.P; pop = Pop2(P, T.D, T.Kmax, a.H, stochastic=True, S=a.S, init_edges=0)
    epoch = 0; active = list(POOL); inst = T.instance(epoch); t0 = time.time(); recs, epochs = [], []
    ep = dict(epoch=0, start=0, tests=len(active), seeds='none (generation 0: all random)', best0=None)
    wipe_note = 'a draw that leaves no survivor WIPES everything' if a.cull_max >= 1 else f'no wipes possible: at least {int(round(P * (1 - a.cull_max)))} survive every generation'
    story(f'=== RUN {a.tag} | {len(POOL)} problems in the pool | population {P} | cull draw each generation ~ uniform(0, {a.cull_max}); {wipe_note} ===')
    story(f'The tests are ALL {len(active)} problems, every generation. Brain input = {T.D} slots, outputs = {T.Kmax}.')
    # fixed measurement set (all problems) used only to MEASURE progress, never to select
    E = [(p, T.ref[p][0][:a.nE], T.ref[p][1][:a.nE]) for p in POOL]
    xE = torch.cat([T.full_x(p, b) for p, b, _ in E]); yE = torch.cat([y for _, _, y in E])
    kmE = torch.cat([torch.arange(T.Kmax, device=dev)[None].expand(len(b), -1) < T.K[p] for p, b, _ in E])
    sizesE = [len(b) for _, b, _ in E]
    def zmeasure(pp):
        q = pp.forward(xE).masked_fill(~kmE[None], -1.0); c = (q.argmax(2) == yE[None]).float()
        per = torch.stack([blk.mean(1) for blk in torch.split(c, sizesE, dim=1)], 1)            # (P, 24) accuracy per problem on the measurement set
        return per.mean(1), per
    best_ever = -1.0; solved_seen = set()
    for g in range(a.gens):
        c = a.cull_max * float(torch.rand(1))                                                   # the random cull fraction of this generation, uniform on [0, cull_max]
        nkeep = int(round(P * (1 - c)))
        z0, _ = zmeasure(pop); pop.ensure_capacity()
        # ---- INNER LIFE: self-directed growth, hard pass/fail (the cull plays no part) ----------------------------------------------------------
        for _ in range(a.k_inner):
            kid = pop.take(torch.arange(P, device=dev)); kid.mutate()
            allp = Pop2.cat(pop, kid); C = allp.complexity()[0]
            ok = C[P:] > C[:P]; pop = allp.take(torch.where(ok, P + torch.arange(P, device=dev), torch.arange(P, device=dev)))
        z1, _ = zmeasure(pop)
        # ---- SCORE on the epoch's tests (fresh random batch each generation) ------------------------------------------------------------------------
        x, y = T.batch(inst, active, a.Be); km = T.kmask(active, a.Be)
        accp = acc_by_problem(pop, x, y, km, len(active), a.Be); score = accp.mean(1)           # average ability across every test of the epoch
        order = torch.argsort(score + 1e-7 * torch.rand(P, device=dev), descending=True)
        rec = dict(g=g, epoch=epoch, c=c, nkeep=nkeep, score_mean=float(score.mean()), score_max=float(score.max()), z0=float(z0.mean()), z1=float(z1.mean()))
        if nkeep == 0:
            # ---- TOTAL WIPE: best candidates become the starting population; the rest is pure random; a new set of tests is drawn ----------------------------------
            nseed = max(1, int(round(a.seed_frac * P))); seeds = order[:nseed]
            ep.update(end=g, score_end=float(score.mean()), best_end=float(score.max()), wiped_by=c); epochs.append(dict(ep)); sc_seed = score[seeds]
            story(f'** WIPE at generation {g}: cull draw {c:.4f} left 0 survivors. The whole population is erased. ' + f'The {nseed} best candidates (scores {float(sc_seed.min()):.3f} to {float(sc_seed.max()):.3f}) are kept as the starting population; '
                  + f'{P - nseed} slots are filled with PURE RANDOM new brains. **')
            fresh = Pop2(P - nseed, T.D, T.Kmax, pop.H, stochastic=True, S=a.S, init_edges=0)
            pop = Pop2.cat(pop.take(seeds), fresh)
            epoch += 1; k = min(len(POOL), a.subset); active = sorted(rng.choice(POOL, size=k, replace=False).tolist()); inst = T.instance(epoch)
            story(f'EPOCH {epoch} begins at generation {g + 1} with a NEW SET OF TESTS: {k} of {len(POOL)} problems drawn at random, with fresh data: {", ".join(active)}')
            ep = dict(epoch=epoch, start=g + 1, tests=k, seeds=f'{nseed} best + {P - nseed} random')
            rec['wipe'] = True
        else:
            keep = order[:nkeep]; surv = pop.take(keep); nk = P - nkeep
            if nk > 0:
                kids = pop.take(keep[torch.arange(nk, device=dev) % nkeep]); kids.vary(); pop = Pop2.cat(surv, kids)    # even repopulation + random-sized variation
            else: pop = surv
            rec['S_meas'] = float(z1[keep].mean() - z1.mean())
        z2, _ = zmeasure(pop); rec['z2'] = float(z2.mean()); rec['C_mean'] = float(pop.complexity()[0].mean()) if g % a.every == 0 else None
        recs.append(rec)
        # ---- logging (plain English) --------------------------------------------------------------------------------------------------------------------------
        if g % a.every == 0 or g == a.gens - 1:
            bi = int(z2.argmax()); _, per = zmeasure(pop.take(torch.tensor([bi], device=dev)))
            solved = int(((per[0] >= 0.9)).sum()); st = pop.stats()
            story(f'gen {g:5d} | epoch {epoch} ({len(active)} tests) | last cull {c * 100:5.1f}% -> kept {nkeep:3d}/{P} | score avg {rec["score_mean"]:.3f} best {rec["score_max"]:.3f} '
                  f'| best brain solves {solved:2d}/24 problems at 90%+ | brains: {st["hidden"]:.0f} hidden nodes avg, T={st["T"]:.1f}, randomness={st["beta"]:.2f} | {time.time() - t0:.0f}s')
            now = {POOL[i] for i in range(len(POOL)) if float(per[0, i]) >= 0.9}
            for p in sorted(now - solved_seen): story(f'    * NEW: problem "{p}" is now solved at 90%+ by the best brain (first time in this run)')
            solved_seen |= now
            if rec['score_max'] > best_ever + 0.02: best_ever = rec['score_max']; story(f'    * new best score of the run: {best_ever:.3f}')
        if g % a.summary_every == 0 or g == a.gens - 1:
            write_summary(a, T, pop, z2, recs, epochs, ep, epoch, active, g, time.time() - t0)
    json.dump(dict(args=vars(a), recs=recs, epochs=epochs), open(os.path.join(RUNS, f'multi_{a.tag}.json'), 'w'))
    story(f'=== RUN {a.tag} finished: {a.gens} generations, {len(epochs)} wipes, {time.time() - t0:.0f}s ===')


@torch.no_grad()
def write_summary(a, T, pop, z, recs, epochs, ep, epoch, active, g, secs):
    """at-a-glance markdown: headline, epoch table, per-problem bars for the best brain, cull-fraction response"""
    bi = int(z.argmax()); best = pop.take(torch.tensor([bi], device=dev)); accs = []
    for p in POOL:
        b, yy = T.ref[p]; q = best.forward(T.full_x(p, b)).masked_fill(~(torch.arange(T.Kmax, device=dev)[None] < T.K[p]), -1.0)
        accs.append(float((q.argmax(2) == yy[None]).float().mean()))
    chance = [T.chance[p] for p in POOL]; norm = [(x - c) / (1 - c) for x, c in zip(accs, chance)]
    L = [f'# Run {a.tag} at a glance', '', f'**Generation {g} of {a.gens}** | epoch {epoch} | {secs / 60:.1f} min | {len(epochs)} total wipes so far', '',
         f'**Headline:** the best brain averages **{np.mean(accs):.0%}** accuracy over all {len(POOL)} problems ({np.mean(norm):.0%} once chance is subtracted); solves **{sum(x >= 0.9 for x in accs)}** at 90%+.', '',
         '## Epochs (a new epoch starts after every wipe)', '', '| epoch | from gen | to gen | tests | start | score when it ended | ended by |', '|---|---|---|---|---|---|---|']
    for e in epochs + [dict(ep, end='running', score_end=recs[-1]['score_mean'], wiped_by=None)]:
        L.append(f'| {e["epoch"]} | {e["start"]} | {e.get("end")} | {e["tests"]} problems | {e["seeds"]} | {e.get("score_end", float("nan")):.3f} | ' + (f'wipe (draw {e["wiped_by"]:.4f})' if e.get('wiped_by') else 'still running') + ' |')
    L += ['', '## Best brain, every problem (fixed reference test data; each # is 10% accuracy; chance-adjusted = how far above guessing)', '', '| problem | accuracy | | chance-adjusted | in this epoch\'s tests |', '|---|---|---|---|---|']
    for i, p in enumerate(POOL): L.append(f'| {p} | {accs[i]:.0%} | `{bar(accs[i])}` | {norm[i]:+.0%} | {"yes" if p in active else ""} |')
    cs = np.array([r['c'] for r in recs if r.get('S_meas') is not None]); dz = np.array([r['z2'] - r['z0'] for r in recs if r.get('S_meas') is not None])
    if len(cs) > 50:
        L += ['', '## What the random cull did (change in average accuracy, bucketed by that generation\'s cull draw)', '', '| cull draw | generations | avg change in accuracy |', '|---|---|---|']
        for lo in np.arange(0, 1, 0.2):
            m = (cs >= lo) & (cs < lo + 0.2)
            if m.sum(): L.append(f'| {lo:.0%}-{lo + 0.2:.0%} | {int(m.sum())} | {dz[m].mean():+.4f} |')
    open(os.path.join(RUNS, f'multi_{a.tag}_summary.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='s0'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--gens', type=int, default=1500); ap.add_argument('--P', type=int, default=128)
    ap.add_argument('--H', type=int, default=32); ap.add_argument('--S', type=int, default=2); ap.add_argument('--k_inner', type=int, default=2); ap.add_argument('--Be', type=int, default=32)
    ap.add_argument('--nE', type=int, default=16); ap.add_argument('--seed_frac', type=float, default=0.1); ap.add_argument('--subset', type=int, default=12)
    ap.add_argument('--cull_max', type=float, default=0.9); ap.add_argument('--every', type=int, default=10); ap.add_argument('--summary_every', type=int, default=50)
    run(ap.parse_args())
