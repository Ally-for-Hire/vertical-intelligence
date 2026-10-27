"""[Revision 6, user 2026-10-26: REFILL = random collections of the current brains. Each empty slot draws a mode: carbon copy | random-sized variation | crossover of 3 random brains |
blend of 2 | scramble (same structure, random weights) | fresh random brain | exact opposite (output connections negated). Parents are uniformly random, not rank-weighted.
--refill cycle restores the old rank-ordered copying.]
[Revision 5, user 2026-10-26: NO internal gates. Each brain makes random changes to itself every generation and KEEPS them unconditionally (--inner free, the default);
only the performance cull removes brains. The cull fraction is 0.9 * u**cull_power (default 6): small culls are the norm, mass culls are possible but rare.
--inner grow restores the old growth-gated inner life, --cull_power 1 restores the uniform draw.]
Random-cull evolution of self-growing brains at LARGE population (10,000 brains per generation). Same rules as multi.py, engineered to fit one 8 GB GPU.

Rules (user, 2026-10-26): each generation the cull fraction is drawn uniformly from [0, cull_max=0.9] (so the population is never wiped); brains are ranked purely on their average
accuracy across ALL 24 problems; survivors repopulate the empty slots evenly with random-SIZED variations (nudge one number ... re-randomise a module); every brain trains only on itself
(inner life: its own random self-modification, kept iff it grew). Model v2 (any-direction growth, evolvable mechanisms, stochastic nodes).

Engineering for 10,000 brains: the 24 problems share a 16-line input BUS plus a 5-line TASK CODE (the brain is told which problem it is looking at, and may ignore it) instead of 245
mostly-empty per-problem input slots (a 20x smaller weight matrix); evaluation is chunked over brains; copies are one gather; brain size is capped (--hmax).

python selfscore/multi_big.py --tag big10k --P 10000 --gens 3000
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.pop2 import Pop2, dev, spawn_children, MODES, MODE_P
from selfscore.multi import Tests, Story, POOL, RUNS, bar, write_summary, acc_by_problem


class BusTests(Tests):
    """all problems share one input bus; a binary task code (5 lines) says which problem is being asked (set code=False to remove it)"""
    def __init__(s, pool, nref=300, code=True):
        super().__init__(pool, nref)
        s.maxd = max(s.d.values()); s.code = code; s.D = s.maxd + (5 if code else 0)
        for p in pool: s.off[p] = 0
        s.cbits = {p: torch.tensor([1.0 if (i >> b) & 1 else -1.0 for b in range(5)], device=dev) for i, p in enumerate(pool)}

    def full_x(s, p, bits):
        x = torch.zeros(len(bits), s.D, device=dev); x[:, :s.d[p]] = bits
        if s.code: x[:, s.maxd:] = s.cbits[p][None]
        return x


@torch.no_grad()
def acc_chunked(pop, x, y, km, nact, per, chunk):
    return torch.cat([acc_by_problem(pop.view(a, min(a + chunk, pop.P)), x, y, km, nact, per) for a in range(0, pop.P, chunk)])


class PhaseTimer:
    def __init__(s, on): s.on = on; s.t = {}; s.last = None
    def mark(s, name):
        if not s.on: return
        torch.cuda.synchronize(); now = time.time()
        if s.last is not None: s.t[name] = s.t.get(name, 0.0) + now - s.last
        s.last = now


@torch.no_grad()
def run(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    story = Story(os.path.join(RUNS, f'multi_{a.tag}.story.txt')); T = BusTests(POOL, code=not a.no_code); P = a.P
    t0 = time.time(); pop = Pop2(P, T.D, T.Kmax, a.H, stochastic=True, S=a.S, init_edges=0)
    inst = T.instance(0); active = list(POOL); lin = torch.arange(P, device=dev); recs, epochs = [], []
    wipe_note = f'no wipes possible: at least {int(round(P * (1 - a.cull_max)))} brains survive every generation'
    story(f'=== RUN {a.tag} | {len(POOL)} problems | POPULATION {P} brains | cull draw each generation ~ uniform(0, {a.cull_max}); {wipe_note} ===')
    uu = np.random.default_rng(0).random(200000) ** a.cull_power * a.cull_max
    story(f'Self-modification: {"FREE (every random change is kept, no gate)" if a.inner == "free" else "gated by growth"}. Cull-size distribution: median {np.median(uu) * 100:.1f}%, mean {uu.mean() * 100:.1f}%, 90th percentile {np.percentile(uu, 90) * 100:.0f}%, '
          f'mass culls (>50%) in {(uu > 0.5).mean() * 100:.1f}% of generations, (>80%) in {(uu > 0.8).mean() * 100:.1f}%.')
    story(f'Refill of empty slots: {"random collections of the current brains, each slot a random mode: " + ", ".join(f"{m} {float(x) * 100:.0f}%" for m, x in zip(MODES, a.mode_p.split(",")) if float(x) > 0) if a.refill == "modes" else "top-ranked survivors copied with random-sized variation (old rule)"}.')
    story(f'Input bus {T.maxd} lines + {"5 task-code lines" if T.code else "no task code"} = {T.D} inputs, {T.Kmax} outputs. Brain size cap: {a.hmax} hidden nodes. GPU memory free at start: {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB')
    g0 = torch.Generator().manual_seed(7)                                                           # fixed measurement set: exactly nE rows per problem (sampled with replacement)
    E = []
    for p in POOL:
        ii = torch.randint(0, len(T.ref[p][0]), (a.nE,), generator=g0).to(dev); E.append((p, T.ref[p][0][ii], T.ref[p][1][ii]))
    xE = torch.cat([T.full_x(p, b) for p, b, _ in E]); yE = torch.cat([y for _, _, y in E])
    kmE = torch.cat([torch.arange(T.Kmax, device=dev)[None].expand(len(b), -1) < T.K[p] for p, b, _ in E])
    def zmeasure(pp): return acc_chunked(pp, xE, yE, kmE, len(POOL), a.nE, a.chunk)             # (P, 24) on the fixed measurement set (never used to select)
    pmode = torch.full((P,), -1, dtype=torch.long, device=dev); cum_dev = torch.zeros(len(MODES), device=dev); cum_n = torch.zeros(len(MODES), device=dev); next_lin = P
    probs = [float(x) for x in a.mode_p.split(',')]; assert len(probs) == len(MODES)
    z2 = zmeasure(pop).mean(1); C = pop.complexity_chunked(a.cchunk)[0]; best_ever = -1.0; solved_seen = set()
    story(f'generation 0 population ready ({time.time() - t0:.0f}s); average accuracy of the random starting brains: {float(z2.mean()):.3f}')
    PT = PhaseTimer(a.profile)
    for g in range(a.gens):
        tg = time.time(); PT.mark('(other)')
        c = a.cull_max * float(torch.rand(1)) ** a.cull_power; nkeep = max(1, int(round(P * (1 - c)))); z0 = z2
        if pop.H < a.hmax and pop.ensure_capacity(min_free=2, step=16) and a.inner != 'free': C = pop.complexity_chunked(a.cchunk)[0]
        # ---- INNER LIFE: each brain modifies itself with its own machinery; kept only if it grew (the cull plays no part) -------------------------------
        for _ in range(a.k_inner):
            if a.inner == 'free': pop.mutate_chunked(a.mchunk); continue                                   # no gate: every random change is kept
            kid = pop.take(torch.arange(P, device=dev)); kid.mutate_chunked(a.mchunk); Ck = kid.complexity_chunked(a.cchunk)[0]
            ok = Ck > C; pop.assign_where(ok, kid); C = torch.where(ok, Ck, C); del kid
        PT.mark('inner life (random self-modification)')
        zp = zmeasure(pop); z1 = zp.mean(1)
        PT.mark('measure z (diagnostic forward pass)')
        # ---- SCORE: average accuracy over all tests, fresh random batch of every problem this generation -------------------------------------------------
        x, y = T.batch(inst, POOL, a.Be); km = T.kmask(POOL, a.Be)
        score = acc_chunked(pop, x, y, km, len(POOL), a.Be, a.chunk).mean(1)
        PT.mark('score (forward pass on the epoch tests)')
        order = torch.argsort(score + 1e-7 * torch.rand(P, device=dev), descending=True); keep = order[:nkeep]
        S_meas = float(z1[keep].mean() - z1.mean())
        # ---- CULL + EVEN REPOPULATION with random-sized variation ---------------------------------------------------------------------------------------
        nk = P - nkeep; dev_by_mode = score - score.mean()                                         # how much better/worse than average each brain scored this generation
        for m in range(len(MODES)):
            sel_m = pmode == m; cum_n[m] += sel_m.sum(); cum_dev[m] += dev_by_mode[sel_m].sum()
        if a.refill == 'modes':
            surv = pop.take(keep); kids, mode, par = spawn_children(surv, nk, probs) if nk > 0 else (None, None, None)
            newpop = Pop2.cat(surv, kids) if nk > 0 else surv
            lin_k = lin[keep][par].clone() if nk > 0 else lin[:0]
            if nk > 0:
                fr = mode == 5; nf = int(fr.sum()); lin_k[fr] = torch.arange(next_lin, next_lin + nf, device=dev); next_lin += nf          # fresh random brains start new lineages
            lin = torch.cat([lin[keep], lin_k]); pmode = torch.cat([torch.full((nkeep,), -1, dtype=torch.long, device=dev), mode if nk > 0 else pmode[:0]]); del surv, kids, pop
        else:
            src = torch.cat([keep, keep[torch.arange(nk, device=dev) % nkeep]]); newpop = pop.take(src); lin = lin[src]; del pop
            if nk > 0: newpop.view(nkeep, P).vary_chunked(a.mchunk)
            pmode = torch.full((P,), -1, dtype=torch.long, device=dev)
        PT.mark('cull + refill (spawn_children)')
        pop = newpop; z2 = zmeasure(pop).mean(1)
        PT.mark('measure z (diagnostic forward pass)')
        log_now = g % a.every == 0 or g == a.gens - 1
        if a.inner != 'free' or log_now: C = pop.complexity_chunked(a.cchunk)[0]
        rec = dict(g=g, c=c, nkeep=nkeep, score_mean=float(score.mean()), score_max=float(score.max()), z0=float(z0.mean()), z1=float(z1.mean()), z2=float(z2.mean()),
                   S_meas=S_meas, C_mean=(float(C.mean()) if (a.inner != 'free' or log_now) else None), C_max=(float(C.max()) if (a.inner != 'free' or log_now) else None), H=pop.H, sec=time.time() - tg)
        recs.append(rec)
        if g % a.every == 0 or g == a.gens - 1:
            bi = int(z2.argmax()); per = zmeasure(pop.take(torch.tensor([bi], device=dev)))[0]
            solved = int((per >= 0.9).sum()); st = pop.view(0, min(P, 600)).stats(); n_lin = float(1.0 / ((torch.bincount(lin).float() / P) ** 2).sum())
            story(f'gen {g:5d} | last cull {c * 100:5.1f}% -> kept {nkeep:5d}/{P} | score avg {rec["score_mean"]:.3f} best {rec["score_max"]:.3f} | best brain solves {solved:2d}/24 at 90%+ '
                  f'| brains: {st["hidden"]:.0f} hidden nodes avg, T={st["T"]:.1f} | {n_lin:.0f} lineages | {rec["sec"]:.1f}s/gen, {time.time() - t0:.0f}s total, GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak')
            if g % (a.every * 5) == 0 and a.refill == 'modes' and float(cum_n.sum()) > 0:
                tab = ' | '.join(f'{MODES[m]} {float(cum_dev[m] / cum_n[m].clamp(min=1)) * 100:+.2f}' for m in range(len(MODES)) if probs[m] > 0)
                story(f'    refill modes, score vs the generation average (percentage points, all generations so far): {tab}')
            now = {POOL[i] for i in range(len(POOL)) if float(per[i]) >= 0.9}
            for p in sorted(now - solved_seen): story(f'    * NEW: problem "{p}" is now solved at 90%+ by the best brain (first time in this run)')
            solved_seen |= now
            if rec['score_max'] > best_ever + 0.02: best_ever = rec['score_max']; story(f'    * new best score of the run: {best_ever:.3f}')
        if g % a.summary_every == 0 or g == a.gens - 1:
            write_summary(a, T, pop, z2, recs, epochs, dict(epoch=0, start=0, tests=len(POOL), seeds='all random'), 0, POOL, g, time.time() - t0)
        if g % 200 == 0: json.dump(dict(args=vars(a), recs=recs), open(os.path.join(RUNS, f'multi_{a.tag}.json'), 'w'))
    json.dump(dict(args=vars(a), recs=recs), open(os.path.join(RUNS, f'multi_{a.tag}.json'), 'w'))
    if a.profile: story('PROFILE (seconds per generation): ' + ' | '.join(f'{k} {v / a.gens:.2f}' for k, v in sorted(PT.t.items(), key=lambda kv: -kv[1])))
    story(f'=== RUN {a.tag} finished: {a.gens} generations, {time.time() - t0:.0f}s ===')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='big10k'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--gens', type=int, default=3000); ap.add_argument('--P', type=int, default=10000)
    ap.add_argument('--H', type=int, default=32); ap.add_argument('--hmax', type=int, default=96); ap.add_argument('--S', type=int, default=2); ap.add_argument('--k_inner', type=int, default=1)
    ap.add_argument('--Be', type=int, default=32); ap.add_argument('--nE', type=int, default=16); ap.add_argument('--cull_max', type=float, default=0.9)
    ap.add_argument('--chunk', type=int, default=500); ap.add_argument('--cchunk', type=int, default=2000); ap.add_argument('--mchunk', type=int, default=2000)
    ap.add_argument('--refill', default='modes', choices=['modes', 'cycle']); ap.add_argument('--mode_p', default='0.25,0.25,0.20,0.10,0.10,0.10,0.0')   # opposite dropped at the user's request (probability 0; code kept)
    ap.add_argument('--_unused', default=''); ap.add_argument('--inner', default='free', choices=['free', 'grow']); ap.add_argument('--cull_power', type=float, default=6.0); ap.add_argument('--every', type=int, default=10); ap.add_argument('--summary_every', type=int, default=50); ap.add_argument('--no_code', action='store_true'); ap.add_argument('--profile', action='store_true')
    run(ap.parse_args())
