"""The fast forward pass must compute the SAME function as the reference. Deterministic mode: outputs equal to float precision. Stochastic mode: mean over many passes equal within noise. Then timing."""
import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.pop2 import Pop2, dev
torch.manual_seed(0)
P, D, K = 500, 21, 7
def population(stoch):
    pop = Pop2(P, D, K, 32, stochastic=stoch, S=2, init_edges=0)
    for g in range(25): pop.ensure_capacity(2, 16); pop.mutate()
    return pop
x = torch.randn(256, D, device=dev)
with torch.no_grad():
    pd = population(False)
    ref = pd.forward_ref(x); fast = pd.forward_fast(x); chunk = pd.forward_chunked(x, 100)
    print(f'DETERMINISTIC  max|fast - ref| = {float((fast - ref).abs().max()):.2e}   max|chunked(sorted) - ref| = {float((chunk - ref).abs().max()):.2e}   (hidden alive avg {float(pd.alive.sum(1).float().mean()):.1f}, T max {int(pd.T.max())}, T mean {float(pd.T.float().mean()):.2f})')
    ps = population(True); ps.S = 1
    R = 60; m_ref = sum(ps.forward_ref(x) for _ in range(R)) / R; m_fast = sum(ps.forward_fast(x) for _ in range(R)) / R
    sd = (m_ref - m_fast).abs()
    print(f'STOCHASTIC     mean over {R} passes: mean|diff| = {float(sd.mean()):.4f}, max|diff| = {float(sd.max()):.3f}  (pure sampling noise at {R} passes is about 0.5/sqrt({R}) = {0.5 / R ** .5:.3f} per output)')
    ps.S = 2; xx = torch.randn(768, D, device=dev)
    for name, f in (('reference forward', lambda: ps.forward_ref(xx)), ('fast forward', lambda: ps.forward_fast(xx)), ('fast, sorted chunks of 100', lambda: ps.forward_chunked(xx, 100))):
        f(); torch.cuda.synchronize(); t = time.time()
        for _ in range(3): f()
        torch.cuda.synchronize(); print(f'  {name:30s} {(time.time() - t) / 3 * 1000:7.0f} ms for {P} brains x {768 * 2} rows')
