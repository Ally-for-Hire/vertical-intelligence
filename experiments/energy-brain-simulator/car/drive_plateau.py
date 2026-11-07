"""Keep training the car brain for as long as it keeps improving (user, 2026-10-23: "continue as long as there is consistent improvement").

Same brain/driver as drive_consensus.py (imported, unchanged). Differences: no success stop; optional --resume from a saved brain;
stops only when the SMOOTHED held-out score (mean of the last SMOOTH evaluations) has not set a new best for PATIENCE evaluations,
or at the time limit. "Consistent improvement" = the smoothed score keeps setting new bests. Each eval is 8 held-out tracks x 300 ticks.

python car/drive_plateau.py --seed 0 --resume [--minutes 60]
"""
import os, sys, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import car.drive_consensus as D
C = D.C
SMOOTH, PATIENCE = 5, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--minutes', type=float, default=60); ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    try:
        import ctypes; ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__)); tag = f'seed{a.seed}_plateau'
    logf = open(os.path.join(here, f'drive_consensus_{tag}.log'), 'a')
    def log(m):
        print(m, flush=True); logf.write(m + '\n'); logf.flush()
    rng = np.random.default_rng(a.seed + 500)
    brain = C.Brain(rng, window=100, own_cost=True, norm_adv=True)
    if a.resume:
        z = np.load(os.path.join(here, f'drive_consensus_seed{a.seed}_brain.npz'))
        brain.W, brain.V, brain.n = z['W'].copy(), z['V'].copy(), int(z['n'])
        log(f'resumed seed {a.seed} brain with {brain.n} neurons')
    envs = [D.CarEnv(seed=s) for s in D.TRAIN_TRACKS[:D.N_ENV]]
    pend, sc_hist, hist, best, since, t0, tick = [], [], [], -1.0, 0, time.time(), 0
    while True:
        dec, recs, u, un = D.poll(brain, envs)
        for k, e in enumerate(envs): e.step(D.action(dec[:, k]))
        sc_hist.append(np.array([e.score for e in envs])); pend.append((recs, u, un)); tick += 1
        if len(pend) > D.H:
            recs0, u0, un0 = pend.pop(0)
            brain.apply(recs0, np.mean(sc_hist[-D.H:], 0), u0, un0)
        if tick % D.EVAL_EVERY == 0:
            g, off, sw = D.grade(brain)
            hist.append(dict(tick=tick, test=g, off=off, neurons=brain.n, retired=brain.retired, sweeps=sw, minutes=(time.time() - t0) / 60))
            sm = float(np.mean([h['test'] for h in hist[-SMOOTH:]])) if len(hist) >= SMOOTH else None
            if sm is not None and sm > best: best, since = sm, 0
            else: since += 1
            log(f"tick {tick:7d} test {g:.3f} off {off:.2f} smoothed {sm if sm is None else round(sm, 3)} best {best:.3f} since-best {since} neurons {brain.n} sweeps {sw:.2f} [{hist[-1]['minutes']:.1f} min]")
            json.dump(dict(refs=D.reference(), hist=hist), open(os.path.join(here, f'drive_consensus_{tag}.json'), 'w'))
            np.savez(os.path.join(here, f'drive_consensus_{tag}_brain.npz'), W=brain.W, V=brain.V, n=brain.n)
            sc_hist = sc_hist[-D.H:]
            if since >= PATIENCE:
                log(f'PLATEAU: smoothed held-out score {best:.3f} has not improved for {PATIENCE} evaluations (last {SMOOTH}: {[round(h["test"], 3) for h in hist[-SMOOTH:]]})'); break
            if hist[-1]['minutes'] > a.minutes:
                log(f'TIME LIMIT with smoothed best {best:.3f}'); break


if __name__ == '__main__':
    main()
