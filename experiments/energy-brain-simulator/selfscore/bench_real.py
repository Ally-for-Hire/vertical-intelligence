"""stage timing of the forward pass on a population that has been through real selection (performance pass 2): python selfscore/bench_real.py [--P 1000] [--gens 120]"""
import os, sys, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.pop2 import Pop2, dev, spawn_children, MODE_P
from selfscore.multi import POOL
from selfscore.multi_big import BusTests, acc_chunked
ap = argparse.ArgumentParser(); ap.add_argument('--P', type=int, default=1000); ap.add_argument('--gens', type=int, default=120); a = ap.parse_args()
torch.manual_seed(0); T = BusTests(POOL); P = a.P; pop = Pop2(P, T.D, T.Kmax, 32, True, 4, init_edges=0); inst = T.instance(0)
@torch.no_grad()
def step(pop):
    pop.ensure_capacity(min_free=2, step=16); pop.mutate_chunked(2000)
    x, y = T.batch(inst, POOL, 32); km = T.kmask(POOL, 32); sc = acc_chunked(pop, x, y, km, len(POOL), 32, 500).mean(1)
    nk = int(P * (1 - 0.2 - 0.4 * float(torch.rand(1)))); keep = sc.argsort(descending=True)[:nk]; surv = pop.take(keep); kids, _, _ = spawn_children(surv, P - nk, MODE_P)
    return Pop2.cat(surv, kids), x, y, km
with torch.no_grad():
    for g in range(a.gens): pop, x, y, km = step(pop)
    torch.cuda.synchronize(); print(f'after {a.gens} generations: hidden avg {float(pop.alive.float().sum(1).mean()):.1f}, H={pop.H}, T avg {float(pop.T.float().mean()):.2f}')
    for label, dtype in (('bf16', torch.bfloat16), ('fp32', torch.float32)):
        Pop2.fwd_dtype = dtype; acc_chunked(pop, x, y, km, 24, 32, 500); torch.cuda.synchronize(); Pop2.trace = {}; t = time.time()
        for _ in range(5): acc_chunked(pop, x, y, km, 24, 32, 500)
        torch.cuda.synchronize(); tot = (time.time() - t) / 5; tr = Pop2.trace; Pop2.trace = None
        print(f'{label}: forward over all {P} brains {tot * 1000:.1f} ms | stages (ms): ' + ' | '.join(f'{k} {v / 5 * 1000:.1f}' for k, v in tr.items()))
    Pop2.fwd_dtype = torch.float32
    def tm(name, f, n=10):
        f(); torch.cuda.synchronize(); t = time.time()
        for _ in range(n): r = f()
        torch.cuda.synchronize(); print(f'  {name:34s} {(time.time() - t) / n * 1000:7.2f} ms'); return r
    print('pieces of the score phase:')
    tm('T.batch', lambda: T.batch(inst, POOL, 32)); tm('T.kmask', lambda: T.kmask(POOL, 32))
    tm('sorted_chunks', lambda: pop.sorted_chunks(P)); idx = pop.sorted_chunks(P)[0]; tm('pop.take(idx)', lambda: pop.take(idx))
    sp = pop.take(idx); tm('forward on sorted pop', lambda: sp.forward(x))
    tm('acc_chunked (all)', lambda: acc_chunked(pop, x, y, km, 24, 32, 500))
    tm('mutate_chunked', lambda: pop.mutate_chunked(2000))
    sc = acc_chunked(pop, x, y, km, 24, 32, 500).mean(1)
    def refill():
        keep = sc.argsort(descending=True)[:600]; surv = pop.take(keep); kids, _, _ = spawn_children(surv, P - 600, MODE_P); return Pop2.cat(surv, kids)
    tm('cull + refill (cat)', refill)
