"""X6.6 (= EMG-1): can the variation MACHINERY evolve itself? Nature improved its own search method (mutation
rate, sex, gene duplication) with no outer loop. Test three pieces of machinery, each BLIND and HERITABLE:
  dir  an inherited direction gene m in [0,1]: child change = m x (parent's last change) + sigma x fresh noise
  rec  an inherited recombination gene r in [0,1]: with chance r a child is a per-parameter mix of two survivors
  dup  an inherited duplication gene u in [0,1]: with chance u a child copies a random active hidden unit into a
       spare slot with the outgoing weights halved on both (function-preserving), so nets grow (width 16 -> up to 32)
Arms (all start from the series-3 best: P=40, B=16, cull 90%, init 0.5; task performance enters ONLY through the
external cull): a = none (only the noise size heritable), b = dir, c = rec, d = dup, e = dir+rec+dup.
Genes start uniform random in [0,1] and mutate by additive N(0, tau) (tau = 1/sqrt(n), the SAME rule as the noise-size gene), clipped to [0,1].
With every feature off this function consumes the random stream exactly like general.evolve (parity test in selftest).
ARBITRARY constants (open problems): spare capacity HMAX = 32 (twice the base width 16); gene start distribution U[0,1].
DERIVED: gene mutation sd = tau; duplication halving (function-preserving).

Usage: python emergent/x66_machinery.py selftest | judge | confirm [--workers N]
  judge   : 5 arms x 7 problems x seeds 400-409 (350 runs), budget 1e7
  confirm : (a) and every arm that passed the judge, 20 fresh seeds 410-429
Results: runs/x66_machinery.json (resumable). Pool size by curious-net/vi/budget.py (15% machine margin)."""
import os, sys, json, math, time, argparse
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
OUTF = os.path.join(HERE, 'runs', 'x66_machinery.json')
CFG = dict(P=40, B=16, cull=0.9, init=0.5)
ARMS = {'a': (), 'b': ('dir',), 'c': ('rec',), 'd': ('dup',), 'e': ('dir', 'rec', 'dup')}
GENES = {'dir': 'm', 'rec': 'r', 'dup': 'u'}
PROBLEMS = ['xorgauss', 'moons', 'circles', 'spirals', 'blobs3', 'checker3', 'parity4']
ORDER = ['spirals', 'checker3', 'parity4', 'moons', 'circles', 'xorgauss', 'blobs3']      # longest first
JUDGE_SEEDS, CONFIRM_SEEDS = list(range(400, 410)), list(range(410, 430))
HMAX, BUDGET, TRACE_EVERY = 32, 1e7, 25
_TASKS = {}


def make_task(prob, grow):
    import torch, general as G

    class GrowNet(G.NetTask):
        """d-HMAX-K tanh net where only the hidden units flagged in self.act (P, HMAX) take part"""
        def __init__(s, prob):
            super().__init__(prob, H=HMAX); s.act = None
        def logits(s, th, X):
            W1, b1, W2, b2 = s.split(th)
            h = torch.tanh(X @ W1 + b1[:, None])
            if s.act is not None: h = h * s.act[:, None, :].to(h.dtype)
            return h @ W2 + b2[:, None]
    t = GrowNet(prob) if grow else G.NetTask(prob)
    t.target = json.load(open(os.path.join(HERE, 'runs', 'general.json')))['ref'][prob]['target']
    return t


def _dup(task, child, actc, ud, g, dev):
    """in place on the children rows chosen with chance ud: copy a random active unit into a random spare slot, halve its outgoing weights on both"""
    import torch
    do = (torch.rand(child.shape[0], device=dev, generator=g) < ud) & (~actc).any(1)
    if not bool(do.any()): return
    ri = do.nonzero()[:, 0]
    src = (torch.rand(len(ri), actc.shape[1], device=dev, generator=g) * actc[ri]).argmax(1)
    dst = (torch.rand(len(ri), actc.shape[1], device=dev, generator=g) * (~actc[ri])).argmax(1)
    W1, b1, W2, b2 = task.split(child)
    W1[ri, :, dst] = W1[ri, :, src]; b1[ri, dst] = b1[ri, src]
    half = W2[ri, src, :] * 0.5
    W2[ri, dst, :] = half; W2[ri, src, :] = half
    actc[ri, dst] = True


def evolve(task, cfg, seed, budget, feats=(), fixed=None):
    """returns dict(cost, solved, gen, genes=final population means, start=initial means, width=mean active width, trace=[(cost, best score)])"""
    import torch
    dev = task.Xtr.device
    with torch.no_grad():
        g = torch.Generator(device=dev).manual_seed(seed); rn = lambda *s: torch.randn(*s, device=dev, generator=g); ru = lambda *s: torch.rand(*s, device=dev, generator=g)
        P, n = cfg['P'], task.n; nk = max(1, int(round(P * (1 - cfg['cull'])))); nc = P - nk; tau = 1.0 / math.sqrt(n)
        th = rn(P, n) * cfg['init']; sig = 10 ** (-2.0 + 2.0 * ru(P))
        fixed = fixed or {}                                                  # a feature in `fixed` has a CONSTANT value (no gene: not heritable, not mutated, no random draw)
        gene = {f: (torch.full((P,), float(fixed[f]), device=dev) if f in fixed else ru(P)) for f in feats if f in GENES}   # drawn after the baseline draws; none when feats is empty
        start = {GENES[f]: float(gene[f].mean()) for f in gene}
        grow = 'dup' in feats
        act = None
        if grow:
            act = torch.zeros(P, task.H, dtype=torch.bool, device=dev); act[:, :16] = True
        last = torch.zeros(P, n, device=dev) if 'dir' in feats else None
        ssum = torch.zeros(P, device=dev); scnt = torch.zeros(P, device=dev); cost = 0; B = float(cfg['B']); gen = 0; trace = []
        def finish(solved):
            return dict(cost=float(cost), solved=bool(solved), gen=gen + 1, genes={GENES[f]: float(gene[f].mean()) for f in gene}, start=start,
                        width=float(act.sum(1).float().mean()) if grow else 16.0, trace=trace)
        while cost < budget:
            task.act = act
            sc, c = task.score(th, g, max(1, int(B)), 'll'); cost += c
            ssum += sc; scnt += 1; sc = ssum / scnt
            order = torch.argsort(sc + 1e-7 * ru(P), descending=True)
            if gen % TRACE_EVERY == 0: trace.append((float(cost), float(sc[order[0]])))
            if grow: task.act = act[order[0]:order[0] + 1]
            if task.done(th[order[0]:order[0] + 1]): return finish(True)
            keep = order[:nk]
            pa = keep[torch.randint(0, nk, (nc,), device=dev, generator=g)]; cs = sig[pa] * torch.exp(tau * rn(nc)); z = rn(nc, n)
            base = th[pa]; cgene = {f: (torch.full((nc,), float(fixed[f]), device=dev) if f in fixed else (gene[f][pa] + tau * rn(nc)).clamp(0, 1)) for f in gene}
            if 'rec' in feats:
                pb = keep[torch.randint(0, nk, (nc,), device=dev, generator=g)]
                use_b = (ru(nc) < gene['rec'][pa])[:, None] & (ru(nc, n) < 0.5)
                base = torch.where(use_b, th[pb], base)
            actc = act[pa].clone() if grow else None
            if grow: _dup(task, base, actc, gene['dup'][pa], g, dev)
            child = base + cs[:, None] * z
            if 'dir' in feats: child = child + cgene['dir'][:, None] * last[pa]
            th_new = torch.cat([th[keep], child]); sig = torch.cat([sig[keep], cs])
            if 'dir' in feats: last = torch.cat([last[keep], child - th[pa]])
            for f in gene: gene[f] = torch.cat([gene[f][keep], cgene[f]])
            if grow: act = torch.cat([act[keep], actc])
            th = th_new
            z0 = torch.zeros(nc, device=dev); ssum = torch.cat([ssum[keep], z0]); scnt = torch.cat([scnt[keep], z0]); gen += 1
        return finish(False)


def run_one(arm, prob, seed, budget):
    key = (prob, 'dup' in ARMS[arm])
    if key not in _TASKS: _TASKS[key] = make_task(prob, key[1])
    t0 = time.time(); r = evolve(_TASKS[key], CFG, seed, budget, ARMS[arm]); r['secs'] = round(time.time() - t0, 1)
    return arm, prob, seed, r


def selftest():
    import torch, general as G
    print('1. parity: feats=() reproduces general.evolve exactly')
    ok = True
    for prob, seed in (('blobs3', 0), ('blobs3', 1), ('xorgauss', 0), ('parity4', 2)):
        t = make_task(prob, False); a = G.evolve(t, CFG, seed, 3e6); b = evolve(t, CFG, seed, 3e6, ())
        same = (a[0] == b['cost'] and a[1] == b['solved'] and a[2] == b['gen']); ok &= same
        print(f'   {prob} seed {seed}: general {a}  x66 {(b["cost"], b["solved"], b["gen"])}  identical {same}')
    print('2. duplication is function-preserving')
    t = make_task('blobs3', True); dev = t.Xtr.device
    g = torch.Generator(device=dev).manual_seed(1)
    th = torch.randn(6, t.n, device=dev, generator=g) * 0.5; act = torch.zeros(6, HMAX, dtype=torch.bool, device=dev); act[:, :16] = True
    t.act = act; before = t.logits(th, t.Xtr[:50]).clone(); new = th.clone(); act2 = act.clone()
    _dup(t, new, act2, torch.ones(6, device=dev), g, dev); t.act = act2; after = t.logits(new, t.Xtr[:50])
    diff = float((before - after).abs().max()); width = act2.sum(1).tolist()
    print(f'   max |logit change| after duplication = {diff:.2e}; active widths {width}'); ok &= diff < 1e-5 and all(w == 17 for w in width)
    print('3. smoke: every arm runs on blobs3 (budget 3e5)')
    for arm in ARMS:
        r = run_one(arm, 'blobs3', 400, 3e5)[3]
        print(f'   arm {arm}: cost {r["cost"]:,.0f} solved {r["solved"]} genes {({k: round(v, 2) for k, v in r["genes"].items()})} width {r["width"]:.1f}')
    print('SELFTEST', 'PASSED' if ok else 'FAILED'); return ok


def sign_p(deltas):
    from scipy.stats import binomtest
    d = [x for x in deltas if x != 0]; return float(binomtest(sum(x > 0 for x in d), len(d), 0.5).pvalue) if d else 1.0


def summarize(res, seeds_by_arm):
    gm = lambda v: float(math.exp(sum(math.log(x) for x in v) / len(v)))
    out = {}
    for arm, seeds in seeds_by_arm.items():
        rows = {p: [res[f'{arm}|{p}|{s}'] for s in seeds] for p in PROBLEMS}
        per = {p: float(np.median([r['cost'] for r in rows[p]])) for p in PROBLEMS}
        allr = [r for p in PROBLEMS for r in rows[p]]
        genes = {}
        for f in ARMS[arm]:
            k = GENES[f]; d = [r['genes'][k] - r['start'][k] for r in allr]
            genes[k] = dict(median_shift=float(np.median(d)), p=sign_p(d), mean_final=float(np.mean([r['genes'][k] for r in allr])))
        evo = []
        for r in allr:
            tr = r['trace']
            if len(tr) >= 4:
                h = len(tr) // 2; s1 = (tr[h][1] - tr[0][1]) / max(1.0, tr[h][0] - tr[0][0]); s2 = (tr[-1][1] - tr[h][1]) / max(1.0, tr[-1][0] - tr[h][0])
                if s1 > 0: evo.append(s2 / s1)
        out[arm] = dict(solved=int(sum(r['solved'] for r in allr)), total=len(allr), gm_cost=gm(list(per.values())), median_cost=per, genes=genes,
                        gene_valid=all(v['p'] < 0.01 for v in genes.values()), evolvability_median=float(np.median(evo)) if evo else None,
                        width=float(np.mean([r['width'] for r in allr])))
    base = out['a']
    for arm, o in out.items():
        o['cheaper_than_a'] = base['gm_cost'] / o['gm_cost']; o['solved_ge_a'] = o['solved'] >= base['solved']
        o['pass'] = bool(o['gene_valid'] and o['solved_ge_a'] and o['cheaper_than_a'] >= 1.25) if arm != 'a' else None
        o['hard_vs_rest'] = dict(hard=gm([base['median_cost'][p] / o['median_cost'][p] for p in ('spirals', 'checker3')]),
                                 rest=gm([base['median_cost'][p] / o['median_cost'][p] for p in PROBLEMS if p not in ('spirals', 'checker3')]))
    return out


def pool_run(jobs, n, res):
    t0 = time.time()
    with mp.get_context('spawn').Pool(n) as pool:
        pending = [pool.apply_async(run_one, j) for j in jobs]
        for i, f in enumerate(pending):
            arm, prob, seed, r = f.get(); res[f'{arm}|{prob}|{seed}'] = r
            json.dump({'budget': BUDGET, 'runs': res}, open(OUTF, 'w'))
            print(f'[{i + 1}/{len(jobs)}] arm {arm} {prob} seed {seed}: cost {r["cost"]:,.0f} solved {r["solved"]} ({r["secs"]}s, {time.time() - t0:.0f}s elapsed)', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--workers', type=int, default=0); a = ap.parse_args()
    if a.mode == 'selftest': sys.exit(0 if selftest() else 1)
    sys.path.insert(0, '~/curious-net/vi'); import budget as BU
    n = a.workers or min(BU.workers(per_worker_gb=1.0), int(os.environ.get('X66_MAX_WORKERS', '6')))
    res = json.load(open(OUTF))['runs'] if os.path.exists(OUTF) else {}
    if a.mode == 'judge':
        arms, seeds = list(ARMS), JUDGE_SEEDS
    else:
        j = summarize(res, {x: JUDGE_SEEDS for x in ARMS}); arms = ['a'] + [x for x in ARMS if x != 'a' and j[x]['pass']]; seeds = CONFIRM_SEEDS
        print('confirming arms', arms, flush=True)
    jobs = [(x, p, s, BUDGET) for p in ORDER for x in arms for s in seeds if f'{x}|{p}|{s}' not in res]
    print(f'{len(jobs)} runs, {n} workers, budget {BUDGET:g}', flush=True); pool_run(jobs, n, res)
    summ = summarize(res, {x: seeds for x in arms}); json.dump({'budget': BUDGET, 'runs': res, 'summary_' + a.mode: summ}, open(OUTF, 'w')); print(json.dumps(summ, indent=1, default=float))
