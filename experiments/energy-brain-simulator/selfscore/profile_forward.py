"""Where does one forward pass spend its time? Manual synchronised timers around each piece of Pop2.forward on a realistic population (the CUPTI profiler is unavailable without admin rights)."""
import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from selfscore.pop2 import Pop2, dev
torch.manual_seed(0)
P, D, K, B = 500, 21, 7, 768
pop = Pop2(P, D, K, 32, stochastic=True, S=2, init_edges=0)
for g in range(25): pop.ensure_capacity(2, 16); pop.mutate()
x = torch.randn(B, D, device=dev)
s = pop; d, N, o0 = s.d, s.N, s.o0; S = 2
print(f'population: P {P}, N {N}, hidden alive avg {float(pop.alive.sum(1).float().mean()):.1f}, rows {B} x S={S} = {B * S}, T mean {float(pop.T.float().mean()):.2f}, T max {int(pop.T.max())}')
T = {}
def tm(name, f, reps=1):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(reps): r = f()
    torch.cuda.synchronize(); T[name] = T.get(name, 0.0) + (time.time() - t) / reps; return r
with torch.no_grad():
    for _ in range(2): pop.forward(x)
    tm('whole forward (reference)', lambda: pop.forward(x), 5); tot = T['whole forward (reference)']
    xr = tm('x.repeat', lambda: x.repeat(S, 1))
    al = torch.cat([torch.ones(P, d, device=dev), s.alive.float(), torch.ones(P, s.K, device=dev)], 1)
    Wm = tm('build Wm = W*M*alive*alive', lambda: s.W * s.M * al[:, :, None] * al[:, None, :], 5)
    beta = s.beta[:, None, None]; a = torch.zeros(P, B * S, N, device=dev); a[:, :, :d] = xr
    z = tm('bmm(a, Wm)  [per step]', lambda: torch.bmm(a, Wm) + s.b[:, None, :], 10)
    zh = z[:, :, d:o0]; ty = s.typ[:, d:o0][:, None, :]
    def acts():
        h = torch.tanh(zh)
        h = torch.where(ty == 0, (torch.rand_like(zh) < torch.sigmoid(beta * zh)).float() * 2 - 1, h)
        h = torch.where(ty == 2, torch.relu(zh).clamp(max=3.0), h); h = torch.where(ty == 3, torch.sin(zh), h)
        h = torch.where(ty == 4, torch.exp(-zh * zh), h); h = torch.where(ty == 5, zh.clamp(-3.0, 3.0), h); return h
    h = tm('hidden activations (tanh+rand+5 where)  [per step]', acts, 10)
    q = tm('output sigmoid  [per step]', lambda: torch.sigmoid(beta * z[:, :, o0:]), 10)
    tm('cat + mask new state  [per step]', lambda: torch.cat([xr[None].expand(P, -1, -1), h, q], 2) * al[:, None, :], 10)
    tm('rand_like only (hidden block)', lambda: torch.rand_like(zh), 10)
steps = float(pop.T.float().mean())
print(f'\nwhole forward: {tot * 1000:.0f} ms for {P} brains ({tot / P * 1000 * 10000 / 1000:.2f} s for 10,000 at this size)')
for k, v in T.items():
    if k.startswith('whole'): continue
    per = v * (steps if 'per step' in k else 1)
    print(f'  {k:55s} {v * 1000:7.2f} ms each' + (f' x {steps:.1f} steps = {per * 1000:6.1f} ms' if 'per step' in k else '') + f'  ({per / tot:5.1%} of forward)')
