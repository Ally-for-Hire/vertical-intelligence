"""Equivalent neural network for the car: value-based MLP that predicts P(success) for each of the 6 actions from the 12 sensor bits.

Identical conditions to predict_drive.py: same context bits, same 6 actions, same SUCCESS event (mean score over the next H ticks beats the
agent's own long-run average), same 8 training cars, same plateau stop, same held-out grade, bandit feedback only. The net is trained by binary
cross-entropy on the chosen action's outcome (batch = all cars, delayed H ticks), epsilon-greedy. Hyperparameters (lr, epsilon, hidden size)
are given on the command line and tried over a grid by the launcher (the best config is chosen by its OWN results: generous to the net).

python car/nn_drive.py --lr 0.01 --eps 0.1 --hidden 64 --seed 0 [--minutes 30]
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
torch.set_num_threads(1)
import car.predict_drive as PD
from car.env import CarEnv
from car.neuro import NS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lr', type=float, default=0.01); ap.add_argument('--eps', type=float, default=0.1); ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--minutes', type=float, default=30)
    a_ = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__)); tag = f'nnq_lr{a_.lr}_eps{a_.eps}_h{a_.hidden}_seed{a_.seed}'
    logf = open(os.path.join(here, f'nn_drive_{tag}.log'), 'a')
    def log(m):
        print(m, flush=True); logf.write(m + '\n'); logf.flush()
    torch.manual_seed(a_.seed); rng = np.random.default_rng(a_.seed)
    h = a_.hidden
    net = torch.nn.Sequential(torch.nn.Linear(NS, h), torch.nn.Tanh(), torch.nn.Linear(h, h), torch.nn.Tanh(), torch.nn.Linear(h, 6))
    opt = torch.optim.Adam(net.parameters(), a_.lr)
    f = lambda X: torch.tensor(2 * X - 1, dtype=torch.float32)
    def greedy(X):
        with torch.no_grad(): return net(f(X)).argmax(1).numpy()
    def grade(base):
        envs = [CarEnv(seed=s) for s in PD.TEST_TRACKS]; sc, conf = [], []
        for _ in range(PD.T_EVAL):
            Xb = PD.bits_of(envs)
            with torch.no_grad(): q = torch.sigmoid(net(f(Xb)))
            a = q.argmax(1).numpy(); conf.append(q.max(1).values.numpy())
            for k, e in enumerate(envs): e.step(int(a[k]))
            sc.append([e.score for e in envs])
        sc, conf = np.array(sc), np.array(conf)
        R = np.array([sc[t + 1:t + 1 + PD.H].mean(0) for t in range(PD.T_EVAL - PD.H - 1)])
        ok = (R > base).astype(float); cf = conf[:len(R)]
        return float(sc.mean()), float(cf.mean()), float(ok.mean())
    envs = [CarEnv(seed=s) for s in PD.TRAIN_TRACKS[:PD.N_ENV]]
    pend, sc_hist, hist, best, since, t0, tick, base, cpu = [], [], [], -1.0, 0, time.time(), 0, 0.0, 0.0
    while True:
        Xb = PD.bits_of(envs); a = greedy(Xb)
        explore = rng.random(len(envs)) < a_.eps; a = np.where(explore, rng.integers(0, 6, len(envs)), a)
        for k, e in enumerate(envs): e.step(int(a[k]))
        sc_hist.append(np.array([e.score for e in envs])); pend.append((Xb, a)); tick += 1
        if len(pend) > PD.H:
            c0 = time.process_time()
            X0, a0 = pend.pop(0); R = np.mean(sc_hist[-PD.H:], 0)
            y = (R > base).astype(np.float32); base += PD.BASE_RATE * (float(R.mean()) - base) if tick > 100 else (float(R.mean()) - base) * 0.05
            logit = net(f(X0))[torch.arange(len(a0)), torch.tensor(a0)]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor(y))
            opt.zero_grad(); loss.backward(); opt.step(); cpu += time.process_time() - c0
        if tick % PD.EVAL_EVERY == 0:
            g, mc, pos = grade(base)
            hist.append(dict(tick=tick, test=g, conf=mc, success_rate=pos, cpu=cpu, minutes=(time.time() - t0) / 60))
            sm = float(np.mean([x['test'] for x in hist[-PD.SMOOTH:]])) if len(hist) >= PD.SMOOTH else None
            if sm is not None and sm > best: best, since = sm, 0
            else: since += 1
            log(f"tick {tick:7d} test {g:.3f} smoothed {sm if sm is None else round(sm, 3)} best {best:.3f} since {since} cpu {cpu:.1f}s [{hist[-1]['minutes']:.1f} min]")
            json.dump(dict(hist=hist), open(os.path.join(here, f'nn_drive_{tag}.json'), 'w'))
            if since >= PD.PATIENCE: log(f'PLATEAU best smoothed {best:.3f}'); break
            if hist[-1]['minutes'] > a_.minutes: log(f'TIME LIMIT best smoothed {best:.3f}'); break


if __name__ == '__main__':
    main()
