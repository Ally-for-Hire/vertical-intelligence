"""Why does the agent sometimes get WORSE? Train agent E on the car (ctx success, episodic 300) and log, at every held-out evaluation, the
score plus candidate causes: sense births/retirements since the last eval, movement of the calibrator weights, policy churn on a FIXED probe set of
situations (share of probe situations whose greedy action changed), throttle-on rate, cars standing still, stated confidence, exploration rate.
python car/stability_probe.py --seed 0 [--ticks 90000] [--method E]
"""
import os, sys, json, argparse, copy, numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import car.predict as P, car.predict_drive as PD
from car.env import CarEnv
from car.neuro import NS

ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--ticks', type=int, default=90000)
ap.add_argument('--method', default='E'); ap.add_argument('--tag', default='probe')
a_ = ap.parse_args()
rng = np.random.default_rng(a_.seed); ag = P.Agent(NS, 6, rng, a_.method, 0.9)
envs = [CarEnv(seed=s) for s in PD.TRAIN_TRACKS[:PD.N_ENV]]; cb = PD.CtxBase()
pend, sc_hist, base, probe, prev_act, prev_w, prev_born, prev_ret, rows = [], [], 0.0, [], None, None, 1, 0, []
VAL_TRACKS = list(range(200, 216))                       # validation tracks: never trained on, never used for the test score
def val(agent):
    es = [CarEnv(seed=s) for s in VAL_TRACKS]; sc = np.zeros(len(es))
    for _ in range(PD.T_EVAL):
        a = agent.act(PD.bits_of(es), explore=False)[0]
        for k, e in enumerate(es): e.step(int(a[k]))
        sc += [e.score for e in es]
    return float(sc.mean() / PD.T_EVAL)
def test_score(agent):
    es = [CarEnv(seed=s) for s in PD.TEST_TRACKS]; sc = np.zeros(len(es))
    for _ in range(PD.T_EVAL):
        a = agent.act(PD.bits_of(es), explore=False)[0]
        for k, e in enumerate(es): e.step(int(a[k]))
        sc += [e.score for e in es]
    return float(sc.mean() / PD.T_EVAL)
champion, champ_val = None, -1.0
def diag():
    es = [CarEnv(seed=s) for s in PD.TEST_TRACKS]; sc = np.zeros(len(es)); acts = []; v = []; conf = []
    for _ in range(PD.T_EVAL):
        Xb = PD.bits_of(es); a, c, u, sure, _ = ag.act(Xb, explore=False); acts.append(a); conf.append(c)
        for k, e in enumerate(es): e.step(int(a[k]))
        sc += [e.score for e in es]; v.append([e.v for e in es])
    acts = np.array(acts); v = np.array(v)
    return float(sc.mean() / PD.T_EVAL), float((acts >= 3).mean()), float((v[-50:].mean(0) < 0.05).mean()), float(np.mean(conf))
for tick in range(1, a_.ticks + 1):
    Xb = PD.bits_of(envs); a, c, used, sure, L = ag.act(Xb)
    for k, e in enumerate(envs): e.step(int(a[k]))
    sc_hist.append(np.array([e.score for e in envs])); pend.append((Xb, a, sure))
    if tick % 300 == 0:
        for e in envs: e.reset()
    if 3000 <= tick < 6000 and tick % 5 == 0: probe.append(Xb)
    if len(pend) > PD.H:
        X0, a0, s0 = pend.pop(0); R = np.mean(sc_hist[-PD.H:], 0)
        y = cb.success(X0, R, base); base += PD.BASE_RATE * (float(R.mean()) - base) if tick > 100 else (float(R.mean()) - base) * 0.05
        ag.learn(X0, a0, y, s0, PD.pooled_logits(ag, X0))
    if tick % PD.EVAL_EVERY == 0:
        g, thr, still, conf = diag()
        v_now = val(ag)
        if v_now > champ_val: champion, champ_val = copy.deepcopy(ag), v_now           # promote only on a strictly better validation score
        champ_test = test_score(champion)
        row = dict(tick=tick, champion=champ_test, val=v_now, champ_val=champ_val, score=g, throttle=thr, still=still, conf=conf, senses=len(ag.senses), born=ag.born - prev_born, retired=ag.retired - prev_ret,
                   sure_succ=ag.sure_succ, unsure=float(np.mean(ag.unsure[-100:])), wnorm=float(np.linalg.norm(ag.w)))
        prev_born, prev_ret = ag.born, ag.retired
        if tick > 6000 and probe:                                  # probe set is frozen after tick 6000
            Pm = np.concatenate(probe); act = ag.act(Pm, explore=False)[0]
            row['churn'] = float((act != prev_act).mean()) if prev_act is not None else None; prev_act = act
            row['probe_throttle'] = float((act >= 3).mean())
        row['dw'] = float(np.linalg.norm(ag.w - prev_w)) if prev_w is not None and len(prev_w) == len(ag.w) else None; prev_w = ag.w.copy()
        rows.append(row); sc_hist = sc_hist[-PD.H:]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'stability_{a_.tag}_{a_.method}_seed{a_.seed}.json')
json.dump(rows, open(out, 'w'))
