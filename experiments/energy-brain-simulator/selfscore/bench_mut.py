"""profile one self-modification round and one refill at P brains (performance pass 2)"""
import os, sys, time, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.pop2 import Pop2, dev, spawn_children
from selfscore.multi import POOL
from selfscore.multi_big import BusTests
ap = argparse.ArgumentParser(); ap.add_argument('--P', type=int, default=1000); ap.add_argument('--gens', type=int, default=30); a = ap.parse_args()
torch.manual_seed(0); T = BusTests(POOL); pop = Pop2(a.P, T.D, T.Kmax, 32, True, 4, init_edges=0)
for g in range(a.gens): pop.ensure_capacity(min_free=2, step=16); pop.mutate_chunked(2000)
def tm(f, n=5):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000
print(f'H={pop.H} hidden avg {float(pop.alive.float().sum(1).mean()):.1f}')
print(f'mutate_chunked   {tm(lambda: pop.mutate_chunked(2000)):.1f} ms')
print(f'spawn_children   {tm(lambda: spawn_children(pop.take(torch.arange(600, device=dev)), 400)):.1f} ms')
print(f'complexity       {tm(lambda: pop.complexity_chunked(2000)):.1f} ms')
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof: pop.mutate_chunked(2000); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=8, max_name_column_width=40))
print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=8, max_name_column_width=40))
