"""Run-time cost of a decision: trained agent E vs MLP forward pass (same inputs). 1 thread, perf_counter wall time, best of 5 repeats."""
import os, sys, time, numpy as np
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch; torch.set_num_threads(1)
import car.benchmarks as BM, car.predict as P, car.compare as CP
def best(fn, n=5):
    t = []
    for _ in range(n):
        c = time.perf_counter(); fn(); t.append(time.perf_counter() - c)
    return min(t)
for prob in ('moons', 'parity6', 'spirals', 'mux6'):
    X, y, Xt, yt, K, kind = CP.problem(prob, 0)
    enc = BM.Encoder(X, kind); Xb = enc.bits(X); rng = np.random.default_rng(0); ag = P.Agent(Xb.shape[1], K, rng, 'E', 0.9)
    for _ in range(1500):
        idx = rng.integers(0, len(Xb), 32); a, c, u, sure, L = ag.act(Xb[idx]); ag.learn(Xb[idx], a, (a == y[idx]).astype(float), sure, L)
    Xq = enc.bits(Xt)[:1000]
    ta = best(lambda: ag.act(Xq, explore=False))
    net = CP.mlp(Xb.shape[1], K, 32); xt = torch.tensor(Xq * 2 - 1, dtype=torch.float32)
    tn = best(lambda: net(xt).argmax(1))
    net1 = CP.mlp(Xb.shape[1], K, 1) if False else None
    a1 = best(lambda: [ag.act(Xq[i:i + 1], explore=False) for i in range(100)]); n1 = best(lambda: [net(xt[i:i + 1]).argmax(1) for i in range(100)])
    print(f'{prob:8s} senses {len(ag.senses):2d} | batch of 1000: agent {ta * 1e3:7.2f} ms  MLP32 {tn * 1e3:6.2f} ms | single decision: agent {a1 * 10:6.2f} ms  MLP32 {n1 * 10:6.2f} ms | agent numbers {P.agent_size(ag)[0]}  state {P.agent_size(ag)[1]}  MLP params {CP.nn_params(Xb.shape[1], K, 32)}')
