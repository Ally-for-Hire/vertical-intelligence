"""Diagnose held-out evaluations where the frozen agent scores ~0 (does the car stand still? which action does it pick?)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import car.predict as P, car.predict_drive as PD
from car.env import CarEnv
from car.neuro import NS
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
EP = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rng = np.random.default_rng(seed); ag = P.Agent(NS, 6, rng, 'E', 0.9)
envs = [CarEnv(seed=s) for s in PD.TRAIN_TRACKS[:PD.N_ENV]]; cb = PD.CtxBase()
pend, sc_hist, base, tick = [], [], 0.0, 0
def diag():
    es = [CarEnv(seed=s) for s in PD.TEST_TRACKS]; sc = np.zeros(len(es)); acts = []; v = []
    for _ in range(PD.T_EVAL):
        Xb = PD.bits_of(es); a, c, u, sure, _ = ag.act(Xb, explore=False); acts.append(a)
        for k, e in enumerate(es): e.step(int(a[k]))
        sc += [e.score for e in es]; v.append([e.v for e in es])
    acts = np.array(acts); v = np.array(v)
    thr = (acts >= 3).mean(); stuck = (v[-50:].mean(0) < 0.05).mean()
    return sc.mean() / PD.T_EVAL, thr, stuck, np.bincount(acts.ravel(), minlength=6) / acts.size
for tick in range(1, 60001):
    Xb = PD.bits_of(envs); a, c, used, sure, L = ag.act(Xb)
    for k, e in enumerate(envs): e.step(int(a[k]))
    sc_hist.append(np.array([e.score for e in envs])); pend.append((Xb, a, sure))
    if EP and tick % EP == 0:
        for e in envs: e.reset()
    if len(pend) > PD.H:
        X0, a0, s0 = pend.pop(0); R = np.mean(sc_hist[-PD.H:], 0)
        y = cb.success(X0, R, base); base += PD.BASE_RATE * (float(R.mean()) - base) if tick > 100 else (float(R.mean()) - base) * 0.05
        ag.learn(X0, a0, y, s0, PD.pooled_logits(ag, X0))
    if tick % PD.EVAL_EVERY == 0:
        g, thr, stuck, hist = diag()
        print(f'tick {tick:6d} score {g:.3f} throttle-on {thr:.2f} cars-standing-still-at-end {stuck:.2f} action-mix {np.round(hist, 2)}', flush=True)
        sc_hist = sc_hist[-PD.H:]
