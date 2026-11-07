import time
import numpy as np
import common  # noqa: F401  (1 thread, low priority)
from env import CarEnv
from cbn import CBN, Encoder, ACTIONS, GRID_BITS, ACT_BITS

WARM = 500


def make(seed, n_d, T, policy_seed=None, drag=None, track=None, mode="hard", width=0.3, gate=False):
    env = CarEnv(seed=seed, n_distract=n_d, drag=drag, track=track)
    rng = np.random.default_rng(seed + 77)
    S = []
    for _ in range(WARM):
        env.step(int(rng.integers(3)) + 3 * int(rng.random() < 0.8))
        S.append(env.scalars())
    enc = Encoder(env.names, mode, width)
    enc.calibrate(S)
    enc.encode(env.view(), env.scalars())          # prime delta memory
    model = CBN(enc, T + 10, seed=seed, gate=gate)
    return env, enc, model


def score_bit_idx(enc):
    j = enc.names.index("score")
    return np.arange(GRID_BITS + j * enc.bp, GRID_BITS + j * enc.bp + 4)


def struct_metrics(env, E):
    """action-link precision/recall (interventional truth) and count of false links touching variables that are
    independent by construction (noise / clock distractors), self-edges excluded (delta bits make them legitimate)."""
    truth = env.truth_edges()
    indep = {n for n, c in zip(env.names[len(env.CAUSAL):], env.dist_class) if c in ("noise", "clock")}
    act_true = {e for e in truth if e[0].startswith("a_")}
    act_found = {e for e in E if e[0].startswith("a_")}
    tp = len(act_found & act_true)
    prec = tp / max(1, len(act_found))
    rec = tp / max(1, len(act_true))
    fp = sum(1 for p, c in E if p != c and (p in indep or c in indep))
    return prec, rec, fp


def run_learn(seed=0, n_d=0, T=20000, policy="curious", G=500, log_every=500, verbose=False, mode="hard", width=0.3, gate=False):
    env, enc, m = make(seed, n_d, T, mode=mode, width=width, gate=gate)
    rng = np.random.default_rng(seed + 5)
    x = np.zeros(m.Bi + 1, np.float32)
    x[:m.Bs] = enc.encode(env.view(), env.scalars())
    log = dict(t=[], bits=[], bits_noise=[], score=[], n_edges=[], n_neurons=[], f1=[], prec=[], rec=[], junk=[], act=[])
    truth = env.truth_edges()
    acc_loss = acc_score = 0.0
    noise_idx = np.array([i for i, v in enumerate(enc.var_of_bit) if "noise" in v], int)
    acc_noise = 0.0
    t0 = time.time()
    for t in range(T):
        if policy == "random":
            a = int(rng.integers(6))
        else:
            a = m.choose_action(x)
        x[m.Bs:m.Bs + ACT_BITS] = ACTIONS[a]
        env.step(a)
        y = enc.encode(env.view(), env.scalars())
        loss = m.observe(x, y)
        acc_loss += loss.sum()
        acc_noise += loss[noise_idx].sum() if len(noise_idx) else 0.0
        acc_score += env.score
        x[:m.Bs] = y
        if (t + 1) % G == 0:
            m.grow()
        if (t + 1) % log_every == 0:
            E = m.edges()
            prec, rec, junk = struct_metrics(env, E)
            log["t"].append(t + 1)
            log["bits"].append(acc_loss / log_every)
            log["bits_noise"].append(acc_noise / log_every)
            log["score"].append(acc_score / log_every)
            log["n_edges"].append(m.n_edges())
            log["n_neurons"].append(m.n_neurons())
            log["prec"].append(prec)
            log["rec"].append(rec)
            log["junk"].append(junk)
            log["f1"].append(2 * prec * rec / max(1e-9, prec + rec))
            log["act"].append(float(len({p for p, c in E if p.startswith("a_")})))
            acc_loss = acc_noise = acc_score = 0.0
            if verbose:
                print(f"t={t+1} bits/tick={log['bits'][-1]:.1f} edges={m.n_edges()} junk={junk} prec={prec:.2f} score={log['score'][-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return env, enc, m, log


if __name__ == "__main__":
    import sys
    n_d = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    run_learn(0, n_d, T, "curious", verbose=True, log_every=500)
