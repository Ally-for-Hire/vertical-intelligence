"""v7 predictive agent (predict.py) attached to the car. Scored only by env.score.

Context: the 12 sensor bits of neuro.sense_bits. Actions: the 6 car actions (steer left/straight/right x throttle off/on) as ONE choice.
Outcome the agent predicts: SUCCESS = the mean score over the next H ticks beats the agent's own long-run average score (EMA, rate 0.002)
  = "this action will raise my score above what I normally get". The stated probability is P(success | context, action).
Same senses/pooling/thinking/growth as predict.py (methods A table, B random nets, C both + learned calibrator). Delayed outcomes: senses may
  change during the H-tick delay, so the pooled logits are recomputed when the outcome arrives.
Stops by the plateau rule (smoothed held-out score fails to set a new best for PATIENCE evaluations) or a time limit.
Grade: held-out tracks, frozen agent, greedy (no exploration). Also reports calibration (ECE) of the stated probabilities on held-out driving.

python car/predict_drive.py --method A --seed 0 [--tau 0.7] [--minutes 30]
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import car.predict as P
from car.env import CarEnv
from car.neuro import sense_bits, NS

H, N_ENV, TRAIN_TRACKS, TEST_TRACKS, T_EVAL, EVAL_EVERY = 5, 8, list(range(8)), list(range(100, 108)), 300, 1500
SMOOTH, PATIENCE, BASE_RATE = 5, 8, 0.002


def bits_of(envs):
    return np.array([sense_bits(e) for e in envs], float)


def pooled_logits(ag, X):
    S, m, K = len(ag.senses), len(X), ag.K
    L = np.zeros((S, m, K))
    for i, x in enumerate(ag.senses): L[i] = x.predict(X)[0]
    return L


def grade(ag, base, seed=0):
    envs = [CarEnv(seed=s) for s in TEST_TRACKS]
    sc, conf, X = [], [], []
    for _ in range(T_EVAL):
        Xb = bits_of(envs); a, c, used, sure, _ = ag.act(Xb, explore=False)
        for k, e in enumerate(envs): e.step(int(a[k]))
        sc.append([e.score for e in envs]); conf.append(c)
    sc, conf = np.array(sc), np.array(conf)                          # (T,m)
    R = np.array([sc[t + 1:t + 1 + H].mean(0) for t in range(T_EVAL - H - 1)])
    ok = (R > base).astype(float); cf = conf[:len(R)]
    return float(sc.mean()), P.calibration(cf.ravel(), ok.ravel()), float(cf.mean()), float(ok.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='A'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--tau', type=float, default=0.7)
    ap.add_argument('--minutes', type=float, default=30)
    a_ = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__)); tag = f'{a_.method}_tau{int(a_.tau * 100)}_seed{a_.seed}'
    logf = open(os.path.join(here, f'predict_drive_{tag}.log'), 'a')
    def log(m):
        print(m, flush=True); logf.write(m + '\n'); logf.flush()
    rng = np.random.default_rng(a_.seed)
    ag = P.Agent(NS, 6, rng, a_.method, a_.tau)
    envs = [CarEnv(seed=s) for s in TRAIN_TRACKS[:N_ENV]]
    pend, sc_hist, hist, best, since, t0, tick, base = [], [], [], -1.0, 0, time.time(), 0, 0.0
    sure_hist = []
    while True:
        Xb = bits_of(envs); a, c, used, sure, L = ag.act(Xb)
        for k, e in enumerate(envs): e.step(int(a[k]))
        sc_hist.append(np.array([e.score for e in envs])); pend.append((Xb, a, sure)); sure_hist.append(float(sure.mean())); tick += 1
        if len(pend) > H:
            X0, a0, s0 = pend.pop(0)
            R = np.mean(sc_hist[-H:], 0)
            y = (R > base).astype(float); base += BASE_RATE * (float(R.mean()) - base) if tick > 100 else (float(R.mean()) - base) * 0.05
            ag.learn(X0, a0, y, s0, pooled_logits(ag, X0))
        if tick % EVAL_EVERY == 0:
            g, ece, mc, pos = grade(ag, base)
            hist.append(dict(tick=tick, test=g, ece=ece, conf=mc, success_rate=pos, senses=len(ag.senses), sure_train=float(np.mean(sure_hist[-EVAL_EVERY:])),
                             minutes=(time.time() - t0) / 60))
            sm = float(np.mean([h['test'] for h in hist[-SMOOTH:]])) if len(hist) >= SMOOTH else None
            if sm is not None and sm > best: best, since = sm, 0
            else: since += 1
            log(f"tick {tick:7d} test {g:.3f} smoothed {sm if sm is None else round(sm, 3)} best {best:.3f} since {since} senses {len(ag.senses)} "
                f"stated-conf {mc:.2f} actual-success {pos:.2f} ECE {ece:.3f} sure(train) {hist[-1]['sure_train']:.2f} [{hist[-1]['minutes']:.1f} min]")
            json.dump(dict(hist=hist), open(os.path.join(here, f'predict_drive_{tag}.json'), 'w'))
            sc_hist = sc_hist[-H:]
            if since >= PATIENCE:
                log(f'PLATEAU best smoothed {best:.3f}; last 5: {[round(h["test"], 3) for h in hist[-5:]]}'); break
            if hist[-1]['minutes'] > a_.minutes:
                log(f'TIME LIMIT best smoothed {best:.3f}'); break


if __name__ == '__main__':
    main()
