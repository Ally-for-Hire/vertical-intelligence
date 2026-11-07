"""v6 consensus brain attached to the car. Runs until it scores consistently on held-out tracks (or a limit is hit).

The SAME brain class as consensus.py (random neurons that re-read each other until >=70% of the voters agree; trust = chance of voting;
growth by trial with exponential cost, cost unit and window measured from its own reward history). Here it answers three yes/no
questions per tick through three polls: forward? right? left?  (right+left or neither = straight). A question slot tells the brain
which question is being polled; neurons are shared by all three.
Sensors: the 12 boolean bits of neuro.sense_bits. Reward: env.score ONLY (on track, centred, moving), averaged over the next H ticks.
Training: N_ENV cars on different training tracks step in parallel (same brain). Grade: held-out tracks, brain frozen but still sampling.
SUCCESS (declared in advance): held-out mean score >= TARGET on SUCCESS_RUN consecutive evaluations.
Controls: coin flip, always-forward-straight, and the hand-written PD controller are logged for reference.

python car/drive_consensus.py [--seed 0] [--minutes 90] [--target 0.5]   (1 thread, low priority)
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ['NEURO_NSLOT'] = '15'
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import car.consensus as C
from car.env import CarEnv
from car.neuro import sense_bits, NS

H, N_ENV, TRAIN_TRACKS, TEST_TRACKS, T_EVAL = 5, 8, list(range(8)), list(range(100, 108)), 300
EVAL_EVERY, SUCCESS_RUN = 1500, 3
NQ = 3                                                     # forward, right, left


def encode(envs, q):
    m = len(envs)
    x = np.zeros((m, C.D))
    bits = np.array([sense_bits(e) for e in envs], float)
    x[:, 0:2 * NS:2] = 2 * bits - 1                        # slot j uses channel 2j
    x[:, 2 * (NS + q)] = 1.0                               # which question is being asked
    return x


def poll(brain, envs, rng=None):
    decs, recs, used, unres = [], [], [], []
    for q in range(NQ):
        dec, rec, u, un = brain.settle(encode(envs, q), rng)
        decs.append(dec); recs.append(rec); used.append(u); unres.append(un)
    return np.array(decs), recs, np.mean(used, 0), np.mean(unres, 0)


def action(dec):
    f, rt, lf = dec
    return (1 if rt == lf else 0 if rt else 2) + 3 * int(f)


def grade(brain, tracks=TEST_TRACKS, T=T_EVAL, seed=0):
    rng = np.random.default_rng(900 + seed)
    envs = [CarEnv(seed=s) for s in tracks]
    score = np.zeros(len(envs)); off = np.zeros(len(envs)); sweeps = []
    for _ in range(T):
        dec, _, u, _ = poll(brain, envs, rng)
        for k, e in enumerate(envs):
            e.step(action(dec[:, k])); score[k] += e.score; off[k] += 1 - e.ontrack
        sweeps.append(u.mean())
    return float(score.mean() / T), float(off.mean() / T), float(np.mean(sweeps))


def reference(tracks=TEST_TRACKS, T=T_EVAL):
    rng = np.random.default_rng(5)
    out = {}
    for name in ('coin', 'forward_straight', 'pd'):
        sc = 0.0
        for s in tracks:
            e = CarEnv(seed=s)
            for _ in range(T):
                if name == 'coin': a = int(rng.integers(0, 6))
                elif name == 'forward_straight': a = 4
                else:
                    u = 2.0 * e.herr + 0.5 * e.lat; a = (0 if u > 0.05 else 2 if u < -0.05 else 1) + 3
                e.step(a); sc += e.score
        out[name] = sc / (T * len(tracks))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--minutes', type=float, default=90)
    ap.add_argument('--target', type=float, default=0.5)
    a = ap.parse_args()
    try:
        import ctypes; ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__)); tag = f'seed{a.seed}'
    logf = open(os.path.join(here, f'drive_consensus_{tag}.log'), 'a')
    def log(m):
        print(m, flush=True); logf.write(m + '\n'); logf.flush()
    ref = reference(); log(f'references (held-out score): {ref}  target {a.target}  success = {SUCCESS_RUN} evals in a row >= target')
    rng = np.random.default_rng(a.seed)
    brain = C.Brain(rng, window=100, own_cost=True, norm_adv=True)
    envs = [CarEnv(seed=s) for s in TRAIN_TRACKS[:N_ENV]]
    pend, sc_hist, hist, streak, t0, tick = [], [], [], 0, time.time(), 0
    while True:
        dec, recs, u, un = poll(brain, envs)
        for k, e in enumerate(envs): e.step(action(dec[:, k]))
        sc_hist.append(np.array([e.score for e in envs])); pend.append((recs, u, un)); tick += 1
        if len(pend) > H:                                       # the poll made H ticks ago has now seen its H future ticks
            recs0, u0, un0 = pend.pop(0)
            R = np.mean(sc_hist[-H:], 0)
            brain.apply(recs0, R, u0, un0)
        if tick % EVAL_EVERY == 0:
            g, off, sw = grade(brain)
            streak = streak + 1 if g >= a.target else 0
            rec = dict(tick=tick, test=g, off=off, neurons=brain.n, retired=brain.retired, sweeps=sw,
                       train=float(np.mean(sc_hist[-EVAL_EVERY:])), minutes=(time.time() - t0) / 60, cost0=brain.cost0)
            hist.append(rec)
            log(f"tick {tick:7d} test {g:.3f} off {off:.2f} train {rec['train']:.3f} neurons {brain.n} retired {brain.retired} sweeps {sw:.2f} cost0 {brain.cost0:.4f} streak {streak} [{rec['minutes']:.1f} min]")
            json.dump(dict(refs=ref, hist=hist, target=a.target), open(os.path.join(here, f'drive_consensus_{tag}.json'), 'w'))
            np.savez(os.path.join(here, f'drive_consensus_{tag}_brain.npz'), W=brain.W, V=brain.V, n=brain.n)
            sc_hist = sc_hist[-H:]
            if streak >= SUCCESS_RUN:
                log(f'SUCCESS at tick {tick}: held-out score >= {a.target} on {SUCCESS_RUN} evaluations in a row'); break
            if rec['minutes'] > a.minutes:
                log(f'STOPPED at the time limit ({a.minutes} min) without success; best test {max(h["test"] for h in hist):.3f}'); break


if __name__ == '__main__':
    main()
