"""Random-cull evolution of self-growing brains at LARGE population, tracked in Weights & Biases. [Revision 7, 2026-10-26; earlier revisions: multi_big_v1..v6_snapshot.py]

Rules (user): REVISION 13 (2026-10-26): cull_rule 'improvement' (default): brains are ranked by LINEAGE IMPROVEMENT = cumulative same-batch gain (score after minus score before the brain's own self-modification, summed since the brain was born), bottom 50% culled every generation, refill = 100% fresh random brains (each starts a new lineage); previous revision: multi_big_v13_snapshot.py. REVISION 12 (2026-10-26): cull = smooth cosine wave 0.15 -> 0.5 -> 0.15 every 50 generations (--cull_period 0 restores the random draw); refill = 100% blends of two random current brains (--mode_p default 0,0,0,1,0,0,0; the earlier mix 0.25,0.25,0.20,0.10,0.10,0.10,0 is multi_big_v12_snapshot.py). Older: each generation the cull fraction is cull_max * u**cull_power. REVISION 9 (2026-10-26): cull = cull_min + (cull_max - cull_min) * u**cull_power, defaults 0.2, 0.6, power 1 = UNIFORM on 0.2..0.6 (revision 8: uniform 0..0.75, multi_big_v8_snapshot.py) (revision 7 used 0.9 * u^6, skewed to small culls; see multi_big_v7_snapshot.py); the population is never wiped;
brains are ranked purely on performance over ALL 24 problems; the empty slots are refilled with random collections of the current brains (carbon copy | random-sized variation |
crossover of 3 | blend of 2 | scramble | fresh random brain); every brain also changes itself randomly each generation and KEEPS it (no gate). Model v2 (any-direction growth,
evolvable mechanisms, stochastic nodes). The 24 problems share a 16-line input BUS plus a 5-line TASK CODE (the brain is told which problem it faces; --no_code removes it).
--cull_rule mean (default, the user's rule: rank by average accuracy) | niche (rank by the best standardised accuracy on ANY single problem: keeps specialists).
Revision 7 adds: performance pass (sorted fast forward; diagnostics every --measure_every generations), W&B tracking (--wandb PROJECT), cross-validated expertise metrics
(POPULATION_DENSITY_HYPOTHESIS.md, amendment 1), the four variety measures, best-brain checkpoints as W&B artifacts, alerts.

python selfscore/multi_big.py --tag N500_s0 --P 500 --gens 1000 --wandb neuroblock-population
"""
import os, sys, json, time, math, argparse, subprocess
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.pop2 import Pop2, dev, spawn_children, MODES, MODE_P, FIELDS
from selfscore.multi import Tests, Story, POOL, RUNS, write_summary, acc_by_problem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

    def batch(s, inst, active, per):
        """a fresh random batch of `per` rows of every active problem, drawn with ONE gather (performance pass 2; same distribution as Tests.batch: uniform rows with replacement).
        The padded data tensor is built once per instance of the problems."""
        key = (id(inst), tuple(active))
        if getattr(s, '_bkey', None) != key:
            nmax = max(len(inst[p][0]) for p in active); s._n = torch.tensor([len(inst[p][0]) for p in active], device=dev).float()
            s._X = torch.zeros(len(active), nmax, s.D, device=dev); s._Y = torch.zeros(len(active), nmax, dtype=torch.long, device=dev)
            for j, p in enumerate(active): X, y = inst[p]; s._X[j, :len(X)] = s.full_x(p, X); s._Y[j, :len(X)] = y
            s._bkey = key; s._ar = torch.arange(len(active), device=dev)[:, None]
        i = (torch.rand(len(active), per, device=dev) * s._n[:, None]).long().clamp(max=(s._n[:, None] - 1).long())
        return s._X[s._ar, i].reshape(-1, s.D), s._Y[s._ar, i].reshape(-1)


@torch.no_grad()
def acc_chunked(pop, x, y, km, nact, per, chunk):
    """per-problem accuracy of every brain; brains with similar T and live size are evaluated together (performance pass)"""
    out = torch.empty(pop.P, nact, device=dev)
    idx = pop.sorted_chunks(pop.P)[0]; out[idx] = acc_by_problem(pop.take(idx), x, y, km, nact, per)      # one sort + one copy; forward() splits itself into memory-bounded blocks of similar brains
    return out


def expertise(zA, zB):
    """cross-validated expertise: pick each problem's best brain and the champion on evaluation A, judge them on the independent evaluation B"""
    ar = torch.arange(zA.shape[1], device=dev); best = zA.argmax(0); champ = int(zA.mean(1).argmax())
    adv = zB[best, ar] - zB[champ]
    return dict(E1=int(best.unique().numel()), E2_cv=float(adv.mean()), E3_cv=int((adv >= 0.05).sum()), champion_acc=float(zB[champ].mean()), best_per_problem_acc=float(zB[best, ar].mean()))


@torch.no_grad()
def variety(pop, lin, zA, xE, nsub=2000):
    """the four variety measures of VARIETY_HYPOTHESIS.md"""
    P = pop.P; f = torch.bincount(lin).float() / P
    sub = pop.take(torch.randperm(P, device=dev)[:min(P, nsub)]); st = sub.stochastic; sub.stochastic = False
    sig = sub.forward(xE).argmax(2); n_func = int(torch.unique(sig, dim=0).shape[0]); sub.stochastic = st
    ids = torch.randperm(P, device=dev)[:48]; V = (pop.W[ids] * pop.M[ids]).flatten(1)
    return dict(n_lin=float(1.0 / (f ** 2).sum()), n_func=n_func / sub.P, d_gen=float(torch.cdist(V, V).sum() / (48 * 47) / V.norm(dim=1).mean().clamp(min=1e-6)), sd_score=float(zA.mean(1).std()))


class PhaseTimer:
    def __init__(s, on): s.on = on; s.t = {}; s.last = None
    def mark(s, name):
        if not s.on: return
        torch.cuda.synchronize(); now = time.time()
        if s.last is not None: s.t[name] = s.t.get(name, 0.0) + now - s.last
        s.last = now


def git_commit():
    try: return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception: return 'unknown'


def cull_at(a, g):
    """cull fraction for generation g. cull_period > 0 (default): a smooth cosine wave that starts at cull_min, rises to cull_max at half a period and comes back, repeating (stateless: depends only on g, so resuming is exact).
    cull_period 0: the older random draw cull_min + (cull_max - cull_min) * u**cull_power."""
    if a.cull_period > 0: return a.cull_min + (a.cull_max - a.cull_min) * (1.0 - math.cos(2.0 * math.pi * g / a.cull_period)) / 2.0
    return a.cull_min + (a.cull_max - a.cull_min) * float(torch.rand(1)) ** a.cull_power


@torch.no_grad()
def run(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    story = Story(os.path.join(RUNS, f'multi_{a.tag}.story.txt')); T = BusTests(POOL, code=not a.no_code); P = a.P
    t0 = time.time(); pop = Pop2(P, T.D, T.Kmax, a.H, stochastic=True, S=a.S, init_edges=0)
    inst = T.instance(0); lin = torch.arange(P, device=dev); recs = []
    # ---- RESUME: if a crash/stop left a state file for this tag, continue from it (re-run the same command; --fresh ignores it) ---------------------------------
    SP = os.path.join(RUNS, f'multi_{a.tag}.state.pt'); ST = None
    if os.path.exists(SP) and not a.fresh:
        try:
            ST = torch.load(SP, map_location='cpu', weights_only=False)
            pop = Pop2(P, T.D, T.Kmax, ST['H'], stochastic=True, S=a.S, tensors={k: v.to(dev) for k, v in ST['tensors'].items()}); lin = ST['lin'].to(dev); recs = ST['recs']; t0 = time.time() - ST['elapsed']
        except Exception as e:
            print(f'state file unreadable ({e}); starting fresh', flush=True); ST = None
    # ---- Weights & Biases ------------------------------------------------------------------------------------------------------------------------------
    wr = None; wandb = None
    if a.wandb:
        try:
            import wandb
            cfg = dict(vars(a)); cfg.update(gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__, commit=git_commit(), n_problems=len(POOL), problems=POOL, inputs=T.D, outputs=T.Kmax)
            wr = wandb.init(project=a.wandb, entity=a.entity or None, name=a.tag, id=(ST['wandb_id'] if ST and ST.get('wandb_id') else None), resume=('allow' if ST else None), group=a.group or f'N{P}-{a.cull_rule}', job_type='population-study', config=cfg, save_code=True, mode=a.wandb_mode,
                            tags=[f'N{P}', a.cull_rule, a.inner, a.refill, 'task-code' if not a.no_code else 'no-task-code'], notes='random-cull self-growing brains on 24 problems (selfscore/multi_big.py)', reinit=True)
            wr.define_metric('generation'); wr.define_metric('*', step_metric='generation')
        except Exception as e:
            print(f'W&B disabled: {e}', flush=True); wr = None
    def wlog(d):
        if wr is not None:
            try: wr.log(d)
            except Exception as e: print(f'W&B log error: {e}', flush=True)
    def walert(title, text):
        if wr is not None:
            try: wr.alert(title=title, text=text)
            except Exception: pass
    # ---- header ----------------------------------------------------------------------------------------------------------------------------------------
    uu = np.array([a.cull_min + (a.cull_max - a.cull_min) * (1.0 - math.cos(2.0 * math.pi * i / a.cull_period)) / 2.0 for i in range(int(a.cull_period))]) if a.cull_period > 0 else a.cull_min + np.random.default_rng(0).random(200000) ** a.cull_power * (a.cull_max - a.cull_min)
    cull_txt = (f'fixed {a.cull_min:.0%}' if a.cull_min == a.cull_max else f'smooth cosine wave {a.cull_min} -> {a.cull_max} -> {a.cull_min} every {a.cull_period:g} generations' if a.cull_period > 0 else f'random draw {a.cull_min} + ({a.cull_max} - {a.cull_min}) * u^{a.cull_power}')
    story(f'=== RUN {a.tag} | {len(POOL)} problems | POPULATION {P} brains | cull rule: {a.cull_rule} | cull = {cull_txt}; at least {int(round(P * (1 - a.cull_max)))} brains survive every generation ===')
    story(f'Self-modification: {"FREE (every random change is kept, no gate)" if a.inner == "free" else "gated by growth"}. Cull-size distribution: median {np.median(uu) * 100:.1f}%, mean {uu.mean() * 100:.1f}%, 90th percentile {np.percentile(uu, 90) * 100:.0f}%, '
          f'mass culls (>50%) in {(uu > 0.5).mean() * 100:.1f}% of generations, (>80%) in {(uu > 0.8).mean() * 100:.1f}%.')
    probs = [float(x) for x in a.mode_p.split(',')]; assert len(probs) == len(MODES)
    story('Refill of empty slots: ' + ('random collections of the current brains, each slot a random mode: ' + ', '.join(f'{m} {p * 100:.0f}%' for m, p in zip(MODES, probs) if p > 0) if a.refill == 'modes' else 'top-ranked survivors copied with random-sized variation (old rule)') + '.')
    story(f'Input bus {T.maxd} lines + {"5 task-code lines" if T.code else "no task code"} = {T.D} inputs, {T.Kmax} outputs. Brain size cap: {a.hmax} hidden nodes. GPU memory free at start: {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB. W&B: {("run " + str(wr.url or wr.id)) if wr is not None else "off"}')
    # measurement sets: A and B are DIFFERENT rows of each problem's reference test set (B is only used to cross-validate expertise); problems with fewer than 2*nE patterns must share patterns
    EA, EB = [], []
    for p in POOL:
        n = len(T.ref[p][0]); perm = torch.randperm(n, generator=torch.Generator().manual_seed(7)).to(dev)
        ia = perm[:a.nE] if n >= a.nE else perm.repeat((a.nE + n - 1) // n)[:a.nE]
        ib = perm[a.nE:2 * a.nE] if n >= 2 * a.nE else ia
        EA.append((p, T.ref[p][0][ia], T.ref[p][1][ia])); EB.append((p, T.ref[p][0][ib], T.ref[p][1][ib]))
    xE = torch.cat([T.full_x(p, b) for p, b, _ in EA]); yE = torch.cat([y for _, _, y in EA]); xB = torch.cat([T.full_x(p, b) for p, b, _ in EB]); yB = torch.cat([y for _, _, y in EB])
    kmE = torch.cat([torch.arange(T.Kmax, device=dev)[None].expand(len(b), -1) < T.K[p] for p, b, _ in EA])
    def zfull(pp):
        with torch.random.fork_rng(devices=[0]): return acc_chunked(pp, xE, yE, kmE, len(POOL), a.nE, a.chunk)      # diagnostics never consume the run's random stream, so a run does not depend on how often it is measured
    def _unused(pp): return acc_chunked(pp, xE, yE, kmE, len(POOL), a.nE, a.chunk)               # (P, 24) on the fixed measurement set (never used to select)
    def zfull_B(pp):
        with torch.random.fork_rng(devices=[0]): return acc_chunked(pp, xB, yB, kmE, len(POOL), a.nE, a.chunk)
    pmode = torch.full((P,), -1, dtype=torch.long, device=dev); cum_dev = torch.zeros(len(MODES), device=dev); cum_n = torch.zeros(len(MODES), device=dev); next_lin = P
    z2 = zfull(pop).mean(1); C = pop.complexity_chunked(a.cchunk)[0]; best_ever = -1.0; solved_seen = set(); best_hist = []; last_stag_alert = -10**9
    start = 0; cumI = torch.zeros(P, device=dev)
    if ST is not None:
        pmode = ST['pmode'].to(dev); cum_dev = ST['cum_dev'].to(dev); cum_n = ST['cum_n'].to(dev); next_lin = ST['next_lin']; best_ever = ST['best_ever']; solved_seen = set(ST['solved_seen'])
        cumI = ST['cumI'].to(dev) if 'cumI' in ST else cumI; best_hist = ST['best_hist']; last_stag_alert = ST['last_stag_alert']; start = ST['g']; epochs = ST['epochs']
        torch.set_rng_state(ST['rng']['cpu']); torch.cuda.set_rng_state(ST['rng']['cuda']); np.random.set_state(ST['rng']['np']); z2 = zfull(pop).mean(1)
        story(f'*** RESUMED from the state file at generation {start} (recorded {ST["saved"]}); {len(recs)} generation records restored. ***'); del ST
    else: story(f'generation 0 population ready ({time.time() - t0:.0f}s); average accuracy of the random starting brains: {float(z2.mean()):.3f}')
    last_save = time.time()
    def save_state(g):
        tmp = SP + '.tmp'
        torch.save(dict(g=g + 1, H=pop.H, tensors={k: getattr(pop, k).cpu() for k in FIELDS}, lin=lin.cpu(), pmode=pmode.cpu(), cum_dev=cum_dev.cpu(), cum_n=cum_n.cpu(), next_lin=next_lin, recs=recs,
                        best_ever=best_ever, cumI=cumI.cpu(), solved_seen=sorted(solved_seen), best_hist=best_hist, last_stag_alert=last_stag_alert, epochs=epochs, elapsed=time.time() - t0, wandb_id=(wr.id if wr is not None else None),
                        saved=time.strftime('%Y-%m-%d %H:%M:%S'), rng=dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(), np=np.random.get_state()), args=vars(a)), tmp)
        os.replace(tmp, SP)                                                                          # atomic: a crash while saving never damages the previous state
    PT = PhaseTimer(a.profile); epochs = []
    for g in range(start, a.gens):
        tg = time.time(); PT.mark('(other)')
        log_now = g % a.every == 0 or g == a.gens - 1; meas = log_now or g % a.measure_every == 0 or g % a.summary_every == 0
        c = cull_at(a, g); nkeep = max(1, int(round(P * (1 - c)))); z0 = z2
        if pop.H < a.hmax and pop.ensure_capacity(min_free=2, step=16) and a.inner != 'free': C = pop.complexity_chunked(a.cchunk)[0]
        if a.cull_rule == 'improvement':                                                              # this generation's batch is drawn FIRST so each brain is scored before and after its own self-modification on the SAME rows
            x, y = T.batch(inst, POOL, a.Be); km = T.kmask(POOL, a.Be); score_before = acc_chunked(pop, x, y, km, len(POOL), a.Be, a.chunk).mean(1)
        # ---- INNER LIFE: each brain modifies itself with its own machinery (free = keep everything; grow = keep only if it grew) ----------------------------------
        for _ in range(a.k_inner):
            if a.inner == 'free': pop.mutate_chunked(a.mchunk); continue
            kid = pop.take(torch.arange(P, device=dev)); kid.mutate_chunked(a.mchunk); Ck = kid.complexity_chunked(a.cchunk)[0]
            ok = Ck > C; pop.assign_where(ok, kid); C = torch.where(ok, Ck, C); del kid
        PT.mark('inner life (random self-modification)')
        z1 = zfull(pop).mean(1) if meas else None
        PT.mark('measure z (diagnostic forward pass)')
        # ---- SCORE: fresh random batch of every problem this generation ------------------------------------------------------------------------------------
        if a.cull_rule != 'improvement': x, y = T.batch(inst, POOL, a.Be); km = T.kmask(POOL, a.Be)
        accp = acc_chunked(pop, x, y, km, len(POOL), a.Be, a.chunk); score = accp.mean(1)
        PT.mark('score (forward pass on the tests)')
        if a.cull_rule == 'niche': rank_score = ((accp - accp.mean(0)) / accp.std(0).clamp(min=1e-6)).amax(1)       # best standardised accuracy on ANY single problem
        elif a.cull_rule == 'improvement':                                                           # lineage improvement = total gain its own self-modifications have produced since the lineage began (same-batch before/after)
            gain = score - score_before; cumI = cumI + gain; rank_score = cumI
        else: rank_score = score
        order = torch.argsort(rank_score + 1e-7 * torch.rand(P, device=dev), descending=True); keep = order[:nkeep]
        S_meas = float(z1[keep].mean() - z1.mean()) if meas else None
        nk = P - nkeep
        isk = pmode >= 0; pm = pmode.clamp(min=0)                                                    # how did each kind of refilled brain score, relative to the population? (no host sync)
        cum_dev.index_add_(0, pm, (score - score.mean()) * isk); cum_n.index_add_(0, pm, isk.float())
        # ---- CULL + REFILL ---------------------------------------------------------------------------------------------------------------------------------
        if a.refill == 'modes':
            surv = pop.take(keep); kids, mode, par = spawn_children(surv, nk, probs) if nk > 0 else (None, None, None)
            newpop = Pop2.cat(surv, kids) if nk > 0 else surv
            lin_k = lin[keep][par].clone() if nk > 0 else lin[:0]
            if nk > 0:
                fr = mode == 5; nf = int(fr.sum()); lin_k[fr] = torch.arange(next_lin, next_lin + nf, device=dev); next_lin += nf
            cumI = torch.cat([cumI[keep], torch.zeros(nk, device=dev)]); lin = torch.cat([lin[keep], lin_k]); pmode = torch.cat([torch.full((nkeep,), -1, dtype=torch.long, device=dev), mode if nk > 0 else pmode[:0]]); del surv, kids, pop
        else:
            src = torch.cat([keep, keep[torch.arange(nk, device=dev) % nkeep]]); newpop = pop.take(src); lin = lin[src]; cumI = torch.cat([cumI[keep], torch.zeros(nk, device=dev)]); del pop
            if nk > 0: newpop.view(nkeep, P).vary_chunked(a.mchunk)
            pmode = torch.full((P,), -1, dtype=torch.long, device=dev)
        PT.mark('cull + refill (spawn_children)')
        pop = newpop
        zfull_now = zfull(pop) if meas else None; z2 = zfull_now.mean(1) if meas else z2
        PT.mark('measure z (diagnostic forward pass)')
        if a.inner != 'free' or log_now: C = pop.complexity_chunked(a.cchunk)[0]
        rec = dict(g=g, c=c, nkeep=nkeep, score_mean=float(score.mean()), score_max=float(score.max()), z0=(float(z0.mean()) if meas else None), z1=(float(z1.mean()) if meas else None),
                   z2=(float(z2.mean()) if meas else None), S_meas=S_meas, C_mean=(float(C.mean()) if (a.inner != 'free' or log_now) else None),
                   C_max=(float(C.max()) if (a.inner != 'free' or log_now) else None), H=pop.H, sec=time.time() - tg)
        recs.append(rec)
        wd = {'improvement/mean_gain_this_gen': float((score - score_before).mean()) if a.cull_rule == 'improvement' else 0.0, 'improvement/lineage_gain_max': float(cumI.max()), 'improvement/lineage_gain_mean': float(cumI.mean()), 'generation': g, 'cull/fraction': c, 'cull/kept': nkeep, 'score/mean': rec['score_mean'], 'score/max': rec['score_max'], 'brain/H_capacity': pop.H, 'time/sec_per_gen': rec['sec']}
        if meas: wd.update({'measure/z_before': rec['z0'], 'measure/z_after_selfmod': rec['z1'], 'measure/z_after_cull': rec['z2'], 'measure/selection_differential': S_meas, 'measure/gain': rec['z2'] - rec['z0']})
        if rec['C_mean'] is not None: wd.update({'brain/complexity_mean': rec['C_mean'], 'brain/complexity_max': rec['C_max']})
        wlog(wd)
        # ---- logging (plain English + W&B tables, histograms, expertise, variety) -----------------------------------------------------------------------------------
        if log_now:
            bi = int(z2.argmax()); zA = zfull_now; zB = zfull_B(pop)      # B: different rows than A, for cross-validated expertise
            per = zA[bi]; solved = int((per >= 0.9).sum()); st = pop.view(0, min(P, 600)).stats()
            with torch.random.fork_rng(devices=[0]): ex = expertise(zA, zB); vr = variety(pop, lin, zA, xE)
            best_hist.append((g, rec['score_max']))
            story(f'gen {g:5d} | last cull {c * 100:5.1f}% -> kept {nkeep:5d}/{P} | score avg {rec["score_mean"]:.3f} best {rec["score_max"]:.3f} | best brain solves {solved:2d}/24 at 90%+ '
                  f'| brains: {st["hidden"]:.0f} hidden nodes avg, T={st["T"]:.1f} | {vr["n_lin"]:.0f} lineages, {vr["n_func"] * 100:.0f}% distinct functions | experts: E1={ex["E1"]} E2cv={ex["E2_cv"] * 100:+.1f}pp E3cv={ex["E3_cv"]} '
                  f'| {rec["sec"]:.1f}s/gen, {time.time() - t0:.0f}s total')
            if g % (a.every * 5) == 0 and a.refill == 'modes' and float(cum_n.sum()) > 0:
                story('    refill modes, score vs the generation average (percentage points, all generations so far): ' + ' | '.join(f'{MODES[m]} {float(cum_dev[m] / cum_n[m].clamp(min=1)) * 100:+.2f}' for m in range(len(MODES)) if probs[m] > 0))
            now = {POOL[i] for i in range(len(POOL)) if float(per[i]) >= 0.9}
            for p in sorted(now - solved_seen):
                story(f'    * NEW: problem "{p}" is now solved at 90%+ by the best brain (first time in this run)'); walert(f'{a.tag}: solved {p}', f'generation {g}: best brain solves {p} at 90%+')
            solved_seen |= now
            if rec['score_max'] > best_ever + 0.02: best_ever = rec['score_max']; story(f'    * new best score of the run: {best_ever:.3f}'); walert(f'{a.tag}: new best {best_ever:.3f}', f'generation {g}')
            if len(best_hist) > 30 and best_hist[-1][1] - best_hist[-31][1] < 0.002 and g - last_stag_alert > 600:
                last_stag_alert = g; walert(f'{a.tag}: stagnating', f'best score gained < 0.002 over the last 300 generations (gen {g}, best {rec["score_max"]:.3f})')
            if wr is not None:
                tab = wandb.Table(columns=['problem', 'accuracy', 'chance', 'chance_adjusted'], data=[[p, float(per[i]), T.chance[p], (float(per[i]) - T.chance[p]) / (1 - T.chance[p])] for i, p in enumerate(POOL)])
                mt = wandb.Table(columns=['mode', 'score_vs_generation_avg_pp', 'brains_seen'], data=[[MODES[m], float(cum_dev[m] / cum_n[m].clamp(min=1)) * 100, int(cum_n[m])] for m in range(len(MODES)) if probs[m] > 0])
                d = {'generation': g, 'best_brain/solved_at_90': solved, 'best_brain/mean_accuracy': float(per.mean()), 'best_brain/per_problem': wandb.plot.bar(tab, 'problem', 'accuracy', title='best brain: accuracy per problem'),
                     'best_brain/table': tab, 'refill/mode_quality': mt, 'hist/score': wandb.Histogram(score.cpu().numpy()), 'hist/cull_so_far': wandb.Histogram(np.array([r['c'] for r in recs])),
                     'brain/hidden_nodes': st['hidden'], 'brain/T_mean': st['T'], 'brain/randomness_beta': st['beta'], 'brain/back_edges': st['back_edges'], 'brain/sigma_mutation': st['sigma'],
                     'variety/lineages': vr['n_lin'], 'variety/distinct_functions_frac': vr['n_func'], 'variety/genotype_distance': vr['d_gen'], 'variety/score_sd': vr['sd_score'],
                     'expertise/E1_distinct_best_brains': ex['E1'], 'expertise/E2_cv_advantage': ex['E2_cv'], 'expertise/E3_cv_problems': ex['E3_cv'], 'expertise/champion_accuracy': ex['champion_acc'], 'expertise/best_per_problem_accuracy': ex['best_per_problem_acc']}
                if rec['C_mean'] is not None: d['hist/complexity'] = wandb.Histogram(C.cpu().numpy())
                for k, v in st['op_probs'].items(): d[f'brain/op_prob/{k}'] = v
                for k, v in enumerate(st['types']): d[f'brain/node_type/{k}'] = v
                wlog(d)
        if a.ckpt_every and (g % a.ckpt_every == 0 or g == a.gens - 1) and g > 0 and wr is not None:      # checkpoint the 8 best brains as a versioned artifact
            try:
                top = pop.take(torch.argsort(z2, descending=True)[:8]); path = os.path.join(RUNS, f'best_brains_{a.tag}.pt')
                torch.save({k: getattr(top, k).cpu() for k in FIELDS} | {'generation': g, 'tests': POOL}, path)
                art = wandb.Artifact(f'best-brains-{a.tag}', type='model', metadata=dict(generation=g, population=P, score=rec['score_max'])); art.add_file(path); wr.log_artifact(art, aliases=['latest', f'gen{g}'])
            except Exception as e: print(f'checkpoint error: {e}', flush=True)
        if g % a.summary_every == 0 or g == a.gens - 1:
            with torch.random.fork_rng(devices=[0]): write_summary(a, T, pop, z2, recs, epochs, dict(epoch=0, start=0, tests=len(POOL), seeds='all random'), 0, POOL, g, time.time() - t0)
        if time.time() - last_save >= a.state_every_sec or g == a.gens - 1: save_state(g); last_save = time.time()
        if g % 200 == 0: json.dump(dict(args=vars(a), recs=recs), open(os.path.join(RUNS, f'multi_{a.tag}.json'), 'w'))
    json.dump(dict(args=vars(a), recs=recs), open(os.path.join(RUNS, f'multi_{a.tag}.json'), 'w'))
    if a.profile: story('PROFILE (seconds per generation): ' + ' | '.join(f'{k} {v / a.gens:.2f}' for k, v in sorted(PT.t.items(), key=lambda kv: -kv[1])))
    story(f'=== RUN {a.tag} finished: {a.gens} generations, {time.time() - t0:.0f}s ===')
    if wr is not None:
        try:
            wr.summary.update(dict(final_score_max=recs[-1]['score_max'], final_score_mean=recs[-1]['score_mean'], solved_problems=len(solved_seen), seconds=time.time() - t0))
            art = wandb.Artifact(f'run-record-{a.tag}', type='run-record'); [art.add_file(os.path.join(RUNS, f)) for f in (f'multi_{a.tag}.story.txt', f'multi_{a.tag}_summary.md', f'multi_{a.tag}.json') if os.path.exists(os.path.join(RUNS, f))]
            wr.log_artifact(art); walert(f'{a.tag}: finished', f'{a.gens} generations, best score {recs[-1]["score_max"]:.3f}, {len(solved_seen)} problems solved at 90%+'); wr.finish()
        except Exception as e: print(f'W&B finish error: {e}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='big10k'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--gens', type=int, default=3000); ap.add_argument('--P', type=int, default=10000)
    ap.add_argument('--H', type=int, default=32); ap.add_argument('--hmax', type=int, default=96); ap.add_argument('--S', type=int, default=2); ap.add_argument('--k_inner', type=int, default=1)
    ap.add_argument('--Be', type=int, default=32); ap.add_argument('--nE', type=int, default=32); ap.add_argument('--cull_max', type=float, default=0.5); ap.add_argument('--cull_min', type=float, default=0.5); ap.add_argument('--cull_period', type=float, default=50.0)
    ap.add_argument('--chunk', type=int, default=500); ap.add_argument('--cchunk', type=int, default=2000); ap.add_argument('--mchunk', type=int, default=2000)
    ap.add_argument('--refill', default='modes', choices=['modes', 'cycle']); ap.add_argument('--mode_p', default='0,0,0,0,0,1,0')
    ap.add_argument('--inner', default='free', choices=['free', 'grow']); ap.add_argument('--cull_power', type=float, default=1.0); ap.add_argument('--cull_rule', default='improvement', choices=['mean', 'niche', 'improvement'])
    ap.add_argument('--every', type=int, default=25); ap.add_argument('--summary_every', type=int, default=50); ap.add_argument('--measure_every', type=int, default=25)
    ap.add_argument('--fresh', action='store_true', help='ignore an existing state file for this tag and start over'); ap.add_argument('--state_every_sec', type=float, default=120.0)
    ap.add_argument('--no_code', action='store_true'); ap.add_argument('--profile', action='store_true'); ap.add_argument('--ckpt_every', type=int, default=0)
    ap.add_argument('--wandb', default='', help='W&B project name (empty = off)'); ap.add_argument('--entity', default=''); ap.add_argument('--group', default='')
    ap.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'])
    run(ap.parse_args())
