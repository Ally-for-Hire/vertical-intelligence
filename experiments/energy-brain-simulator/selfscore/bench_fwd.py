"""Benchmark + equivalence harness for the forward pass at 1,000 brains (performance pass 2, 2026-10-26).
Builds a population by self-modification only (no selection) so brains have realistic sizes, then times each forward variant on the same brains and checks that outputs agree.
python selfscore/bench_fwd.py [--P 1000] [--gens 60] [--profile]
"""
import os, sys, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.pop2 import Pop2, dev
from selfscore.multi import POOL
from selfscore.multi_big import BusTests, acc_chunked


def timeit(f, n=3):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): r = f()
    torch.cuda.synchronize(); return (time.time() - t) / n, r


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--P', type=int, default=1000); ap.add_argument('--gens', type=int, default=60); ap.add_argument('--chunk', type=int, default=500)
    ap.add_argument('--profile', action='store_true'); a = ap.parse_args()
    torch.manual_seed(0)
    T = BusTests(POOL); pop = Pop2(a.P, T.D, T.Kmax, 32, stochastic=True, S=4, init_edges=0)
    t = time.time()
    for g in range(a.gens):
        if pop.ensure_capacity(min_free=2, step=16): pass
        pop.mutate_chunked(2000)
    torch.cuda.synchronize(); print(f'built population: {a.gens} mutation rounds in {time.time() - t:.1f}s; H={pop.H}; hidden avg {float(pop.alive.float().sum(1).mean()):.1f}, max {int(pop.alive.sum(1).max())}')
    inst = T.instance(0); x, y = T.batch(inst, POOL, 32); km = T.kmask(POOL, 32)
    from selfscore.pop2_v8_snapshot import Pop2 as Old
    old = Old(a.P, pop.d, pop.K, pop.H, True, pop.S, tensors=pop._t()); old.fast = True
    def run_old(): return acc_chunked(old, x, y, km, len(POOL), 32, a.chunk)
    def run_new(): return acc_chunked(pop, x, y, km, len(POOL), 32, 10**9)         # chunking is automatic now
    so, ao = timeit(run_old); sn, an = timeit(run_new)
    print(f'forward over all brains | old {so * 1000:.0f} ms | new {sn * 1000:.0f} ms | speedup {so / sn:.2f}x | mean accuracy old {float(ao.mean()):.4f} new {float(an.mean()):.4f}')
    pop.stochastic = False; old.stochastic = False; ad_o = run_old(); ad_n = run_new(); pop.stochastic = True
    print(f'deterministic mode, old vs new max abs difference in accuracy: {float((ad_o - ad_n).abs().max()):.6f}')
    Pop2.fwd_dtype = torch.bfloat16; sb, ab = timeit(run_new); Pop2.fwd_dtype = torch.float32
    print(f'bf16 variant: {sb * 1000:.0f} ms ({sn / sb:.2f}x vs new fp32), mean accuracy {float(ab.mean()):.4f}')
    torch.backends.cuda.matmul.allow_tf32 = True; st, at = timeit(run_new); torch.backends.cuda.matmul.allow_tf32 = False
    print(f'tf32 matmul variant: {st * 1000:.0f} ms ({sn / st:.2f}x vs new fp32), mean accuracy {float(at.mean()):.4f}')
    print(f'peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB')
