"""PIECE-B1: fast + slow memory with replay (hippocampus + neocortex; complementary learning systems, McClelland et al. 1995).
World: one causal world (6 core variables, 2 actions, 10 distractors, generator v2) whose dynamics switch between 3 REGIMES (same wiring, different coefficient strengths/signs and gains, like a drag change),
in the order A B C A B C A B C, 600 ticks each (5,400 ticks). The learner never sees which regime is active.
Arms (per-output online MLPs, Adam, batch 1, same settings as exp M):
  slow     : the plain online network
  replay   : slow network + one replayed past sample per tick (reservoir buffer of 2,000, all regimes)
  fast     : episodic store only (k-nearest-neighbour regression over the last 2,000 samples, k = 10)
  cls      : slow network with replay + fast store; prediction = mix weighted by each one's recent squared error (exponential window 50 ticks)
Metrics (normalised squared error of the predicted change): NEW = first 100 ticks after a switch into a regime never seen before (blocks 2, 3); RETURN = first 100 ticks after switching back into a seen regime (blocks 4-9);
TOTAL = all ticks. 48 worlds. Buffer size, k, window and replay ratio are EXPERIMENT settings, not tuned.
PREDICTIONS (PREDICTIONS.md, registered before running): P8 replay cuts RETURN error vs slow by >= 20%; P9 cls cuts NEW error vs slow by >= 15%; P10 replay raises NEW error vs slow by <= 10%;
P11 cls has the lowest TOTAL error of the four arms."""
import os, sys, json, time, copy
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

BLOCK, ORDER = 600, (0, 1, 2, 0, 1, 2, 0, 1, 2)
BUF, KNN, WIN, HID, LR = 2000, 10, 50, 32, 3e-3


def make(seed):
    from worlds import World
    w = World(seed, n_state=6, n_act=2, n_distract=10, noise=0.05, max_parents=2)
    r = np.random.default_rng(seed + 31337)
    regimes = []
    for k in range(3):
        coef = [c * r.uniform(0.3, 1.7, size=len(c)) * np.where(r.random(len(c)) < 0.3, -1.0, 1.0) for c in w.coef]
        gain = [float(r.uniform(0.1, 0.6)) for _ in w.gain]
        regimes.append((coef, gain))
    return w, regimes


def stream(seed):
    w, regimes = make(seed)
    rng = np.random.default_rng(seed + 7)
    ns, na = w.n_state, w.n_act
    T = BLOCK * len(ORDER)
    X = np.zeros((T, ns + na)); Y = np.zeros((T, ns)); a = np.zeros(na)
    obs = w.observe()
    for t in range(T):
        if t % BLOCK == 0:
            w.coef, w.gain = copy.deepcopy(regimes[ORDER[t // BLOCK]])
        if t % 5 == 0:
            a = rng.integers(-1, 2, size=na).astype(float)
        X[t, :ns] = obs; X[t, ns:] = a
        w.step(w.act_to_world(a)); obs = w.observe(); Y[t] = obs
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    return X, Y, cp


def run(seed):
    from nn import AdamMLP, RunStd
    X, Y, cp = stream(seed)
    T, n_in = X.shape; n_out = len(cp)
    D = (Y - X[:, :Y.shape[1]])[:, cp]
    sx, sy = RunStd(n_in), RunStd(n_out)
    rng = np.random.default_rng(seed + 99)
    nets = {arm: [AdamMLP(n_in, 1, HID, LR, seed + j) for j in range(n_out)] for arm in ("slow", "slow2", "replay", "cls")}
    bufX = np.zeros((BUF, n_in)); bufY = np.zeros((BUF, n_out)); nb = 0; seen = 0
    errs = {arm: np.zeros(T) for arm in ("slow", "slow2", "replay", "fast", "cls")}
    e_slow, e_fast = 1.0, 1.0
    for t in range(T):
        sx.update(X[t]); sy.update(D[t])
        xs, ys = sx(X[t]), sy(D[t])
        # fast store prediction (before storing this sample)
        if nb > KNN:
            d2 = ((bufX[:nb] - xs) ** 2).sum(1)
            idx = np.argpartition(d2, KNN)[:KNN]
            wts = 1.0 / (d2[idx] + 1e-3)
            pf = (bufY[idx] * wts[:, None]).sum(0) / wts.sum()
        else:
            pf = np.zeros(n_out)
        out = {}
        for arm in ("slow", "slow2", "replay", "cls"):
            out[arm] = np.array([nets[arm][j].step(xs, ys[j:j + 1])[0] for j in range(n_out)])
        for j in range(n_out):                                              # equal-compute control: a second update on the CURRENT sample
            nets["slow2"][j].step(xs, ys[j:j + 1])
        if nb > 0:                                                          # one replayed past sample per tick (replay and cls arms)
            q = int(rng.integers(nb))
            for arm in ("replay", "cls"):
                for j in range(n_out):
                    nets[arm][j].step(bufX[q], bufY[q, j:j + 1])
        mix_w = (1.0 / e_slow) / (1.0 / e_slow + 1.0 / e_fast)
        pc = mix_w * out["cls"] + (1 - mix_w) * pf
        errs["slow"][t] = np.mean((out["slow"] - ys) ** 2)
        errs["slow2"][t] = np.mean((out["slow2"] - ys) ** 2)
        errs["replay"][t] = np.mean((out["replay"] - ys) ** 2)
        errs["fast"][t] = np.mean((pf - ys) ** 2)
        errs["cls"][t] = np.mean((pc - ys) ** 2)
        e_slow += (np.mean((out["cls"] - ys) ** 2) - e_slow) / WIN
        e_fast += (errs["fast"][t] - e_fast) / WIN
        # reservoir buffer (shared by fast store and replay)
        seen += 1
        if nb < BUF:
            bufX[nb], bufY[nb] = xs, ys; nb += 1
        else:
            k = int(rng.integers(seen))
            if k < BUF:
                bufX[k], bufY[k] = xs, ys
    res = {}
    for arm, e in errs.items():
        new = np.concatenate([e[b * BLOCK:b * BLOCK + 100] for b in (1, 2)])
        ret = np.concatenate([e[b * BLOCK:b * BLOCK + 100] for b in range(3, 9)])
        res[arm] = dict(new=float(new.mean()), ret=float(ret.mean()), total=float(e.mean()), curve=e.reshape(len(ORDER), BLOCK).mean(1).tolist())
    return res


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(9000, 9048))
    json.dump(res, open("../results/pb1b_memory.json" if os.environ.get("PB1B") else "../results/pb1_memory.json", "w"))
    arms = ("slow", "slow2", "replay", "fast", "cls")
    for m in ("new", "ret", "total"):
        base = np.array([r["slow"][m] for r in res])
        line = []
        for a in arms:
            v = np.array([r[a][m] for r in res]); rel = 1 - v / base
            line.append(f"{a} {v.mean():.3f} ({100 * rel.mean():+.1f}% vs slow, SE {100 * rel.std(ddof=1) / np.sqrt(len(rel)):.1f})")
        print(f"{m:5s}: " + " | ".join(line))
    print("per-block mean error (slow / replay / cls):")
    for b in range(len(ORDER)):
        print(f"  block {b} regime {'ABC'[ORDER[b]]}: " + " / ".join(f"{np.mean([r[a]['curve'][b] for r in res]):.3f}" for a in ("slow", "replay", "cls")))
    print("done", round(time.time() - t0), "s")
