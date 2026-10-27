"""NeuroBlock v4 ("the poll"): every tick the WHOLE network is polled; each neuron answers yes / no / abstain. Scored only on steering.

Contract (user, 2026-10-23):
  * the net starts as ONE neuron in the centre of the grid and expands itself
  * every tick, for each action question (forward? right? left?) EVERY neuron votes yes, no or abstain
  * majority wins; a tie (including everyone abstaining) is decided by a coin flip
  * each neuron is trained by what it said and what happened afterwards
  * grade = env.score (on track, centred, moving) on held-out tracks; nothing else

A neuron = a situation + a record.
  situation: which of the 12 boolean sensor bits it cares about and the value it wants for each ('care' mask + 'val').
             It only has an opinion when the current sensors match; otherwise it abstains.
  record:    for each action a, the reward that followed when a was ON vs OFF while this neuron's situation held
             (reward = mean env.score over the next H ticks; running sums, so everything is measured, nothing hand-set).
  vote:      yes if 'on' was significantly better than 'off', no if significantly worse, abstain otherwise
             (|z| < Z_VOTE or too little data = abstain). Z_VOTE/Z_SPLIT are plain statistics conventions, not tuned.
Self-expansion: when a neuron's record shows that an action's effect DIFFERS between the two values of some sensor bit
  (interaction z > Z_SPLIT), it spawns two children, one per value of that bit, in free cells next to it. Children inherit
  the matching slice of the parent's record. The parent stays (general rule); children are the specific rules.
Legacy/related: neuro.py (v2 Boolean graph, mutate-and-keep), neurogrid.py (v3 wired grid), agent.py/train.py (v1 Brain2 + net).
See car/RESEARCH_LOG.md.

python car/voter.py [--laps 100] [--seeds 3]   (1 thread, low priority)
"""
import os, sys, json, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from car.env import CarEnv
from car.neuro import SENSORS, NS, sense_bits

try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass

ACTIONS = ('forward', 'right', 'left')       # env action = (0 if right-only, 2 if left-only, else 1 straight) + 3*forward
NA = len(ACTIONS)
G, MAXN = 16, 256
H = 5                                        # reward horizon in ticks
Z_VOTE, Z_SPLIT, NMIN, SPLIT_EVERY, DECAY_EVERY = 2.0, 3.0, 4, 25, 200
TRAIN, TEST, T = [0, 1, 2], list(range(100, 108)), 300


class Poll:
    def __init__(s, rng):
        s.rng = rng
        s.n = 1
        s.pos = [(G // 2, G // 2)]                                   # one neuron, centre of the grid
        s.care = np.zeros((MAXN, NS), bool)
        s.val = np.zeros((MAXN, NS), bool)
        s.S1 = np.zeros((MAXN, NA, 2)); s.S2 = np.zeros((MAXN, NA, 2)); s.N = np.zeros((MAXN, NA, 2))
        s.B1 = np.zeros((MAXN, NA, 2, NS, 2)); s.B2 = np.zeros_like(s.B1); s.BN = np.zeros_like(s.B1)
        s.split_done = np.zeros((MAXN, NS), bool)
        s.birth = [0]                                                # tick of birth, for the log
        s.t = 0
        s.stats = dict(yes=0, no=0, abstain=0, ties=0, polls=0)

    # ---- polling ------------------------------------------------------------------------
    def applicable(s, bits):
        n = s.n
        return np.nonzero((~s.care[:n] | (s.val[:n] == bool_arr(bits))).all(1))[0]

    def votes(s, idx):
        """votes[k, a] in {+1 yes, -1 no, 0 abstain} for each applicable neuron idx[k] and action a"""
        N, S1, S2 = s.N[idx], s.S1[idx], s.S2[idx]
        ok = (N >= NMIN).all(2)
        m = S1 / np.maximum(N, 1)
        var = np.maximum(S2 / np.maximum(N, 1) - m * m, 1e-9)
        se = np.sqrt((var / np.maximum(N, 1)).sum(2) * H)            # x H: rewards over overlapping windows are correlated
        z = (m[..., 1] - m[..., 0]) / se
        v = np.where(ok & (z > Z_VOTE), 1, np.where(ok & (z < -Z_VOTE), -1, 0))
        return v

    def decide(s, bits, learn):
        idx = s.applicable(bits)
        v = s.votes(idx)
        yes, no = (v > 0).sum(0), (v < 0).sum(0)
        on = np.where(yes > no, 1, np.where(no > yes, 0, s.rng.integers(0, 2, NA)))
        s.stats['polls'] += 1
        s.stats['yes'] += int((v > 0).sum()); s.stats['no'] += int((v < 0).sum()); s.stats['abstain'] += int((v == 0).sum())
        s.stats['ties'] += int((yes == no).sum())
        return on.astype(int), idx

    # ---- training: what each neuron said + what happened ------------------------------------
    def train(s, idx, bits, taken, R):
        for a in range(NA):
            c = taken[a]
            s.S1[idx, a, c] += R; s.S2[idx, a, c] += R * R; s.N[idx, a, c] += 1
            bv = bits.astype(int)
            s.B1[idx[:, None], a, c, np.arange(NS)[None, :], bv[None, :]] += R
            s.B2[idx[:, None], a, c, np.arange(NS)[None, :], bv[None, :]] += R * R
            s.BN[idx[:, None], a, c, np.arange(NS)[None, :], bv[None, :]] += 1

    def decay(s):
        for arr in (s.S1, s.S2, s.N, s.B1, s.B2, s.BN): arr[:s.n] *= 0.9

    # ---- self-expansion ------------------------------------------------------------------------
    def maybe_split(s):
        n = s.n
        if n + 2 > MAXN: return
        BN, B1, B2 = s.BN[:n], s.B1[:n], s.B2[:n]
        m = B1 / np.maximum(BN, 1)
        var = np.maximum(B2 / np.maximum(BN, 1) - m * m, 1e-9)
        ok = (BN >= NMIN).all(2)                                      # (n, NA, NS, 2) all of on/off cells have data
        g = m[:, :, 1] - m[:, :, 0]                                   # (n, NA, NS, 2): effect of action per bit value
        v = (var / np.maximum(BN, 1)).sum(2) * H                      # variance of that effect
        z = (g[..., 1] - g[..., 0]) / np.sqrt(v[..., 0] + v[..., 1])  # (n, NA, NS): effect differs between bit=0 and bit=1
        valid = ok.all(3) & ~s.split_done[:n][:, None, :] & ~s.care[:n][:, None, :]
        zz = np.where(valid, np.abs(z), 0)
        i, a, b = np.unravel_index(int(np.argmax(zz)), zz.shape)
        if zz[i, a, b] < Z_SPLIT: return
        s.split_done[i, b] = True
        for bv in (0, 1):
            care, val = s.care[i].copy(), s.val[i].copy()
            care[b], val[b] = True, bool(bv)
            if any((s.care[j] == care).all() and (s.val[j] == val).all() for j in range(s.n)): continue
            k = s.n; s.n += 1
            s.care[k], s.val[k] = care, val
            s.pos.append(free_cell(s.pos, s.pos[i], s.rng)); s.birth.append(s.t)
            s.N[k], s.S1[k], s.S2[k] = s.BN[i, :, :, b, bv], s.B1[i, :, :, b, bv], s.B2[i, :, :, b, bv]   # inherited slice


def bool_arr(bits):
    return np.asarray(bits, bool)


def free_cell(pos, near, rng):
    taken = set(pos)
    for r in range(1, G):
        cand = [(x, y) for x in range(max(0, near[0] - r), min(G, near[0] + r + 1))
                for y in range(max(0, near[1] - r), min(G, near[1] + r + 1)) if (x, y) not in taken]
        if cand: return cand[int(rng.integers(len(cand)))]
    return near


def drive(net, seed, T=T, learn=False, mode='poll'):
    """one lap. mode 'poll' = the network decides; 'random' = coin flip control; 'pd' not used here."""
    env = CarEnv(seed=seed)
    score = off = lat = 0.0
    rec, scores = [], []
    for t in range(T):
        bits = sense_bits(env)
        if mode == 'random':
            on = net.rng.integers(0, 2, NA); idx = None
        else:
            on, idx = net.decide(bits, learn)
        f, rt, lf = on
        env.step((1 if rt == lf else 0 if rt else 2) + 3 * int(f))
        score += env.score; off += 1 - env.ontrack; lat += env.lat_d
        if learn and idx is not None:
            scores.append(env.score); rec.append((idx, bits, on, t))
            if len(rec) > H:
                j = len(rec) - 1 - H                                  # record from H ticks ago has now seen its H future ticks
                idx0, bits0, on0, t0 = rec[j]
                net.train(idx0, bits0, on0, float(np.mean(scores[t0:t0 + H])))
                net.t += 1
                if net.t % SPLIT_EVERY == 0: net.maybe_split()
                if net.t % DECAY_EVERY == 0: net.decay()
    return dict(score=score / T, off=off / T, lat=lat / T)


def grade(net, tracks, mode='poll'):
    r = [drive(net, s, mode=mode) for s in tracks]
    return {k: float(np.mean([x[k] for x in r])) for k in r[0]}


def live(seed, laps, log, eval_every=10):
    net = Poll(np.random.default_rng(seed))
    hist = []
    for lap in range(laps):
        drive(net, TRAIN[lap % len(TRAIN)], learn=True)
        if lap % eval_every == eval_every - 1 or lap == laps - 1:
            st = dict(net.stats); net.stats = dict(yes=0, no=0, abstain=0, ties=0, polls=0)
            gr = grade(net, TEST)
            tot = max(st['yes'] + st['no'] + st['abstain'], 1)
            gr.update(lap=lap, neurons=net.n, yes=st['yes'] / tot, no=st['no'] / tot, abstain=st['abstain'] / tot,
                      tie_rate=st['ties'] / max(st['polls'] * NA, 1))
            hist.append(gr)
            log(f'poll seed{seed} lap{lap:3d} test {gr["score"]:.3f} off {gr["off"]:.2f} lat {gr["lat"]:.2f} neurons {net.n} '
                f'yes/no/abs {gr["yes"]:.2f}/{gr["no"]:.2f}/{gr["abstain"]:.2f} ties {gr["tie_rate"]:.2f}')
            net.stats = dict(yes=0, no=0, abstain=0, ties=0, polls=0)
    return hist, net


def describe(net):
    """human-readable list of the grown neurons: situation, position, birth tick"""
    rows = []
    for i in range(net.n):
        cond = ' & '.join(f'{SENSORS[b]}={int(net.val[i, b])}' for b in range(NS) if net.care[i, b]) or '(anything)'
        rows.append(dict(id=i, at=net.pos[i], born=net.birth[i], situation=cond))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--laps', type=int, default=100); ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'voter_results.json')
    from car.agent import pd_rollout
    t0 = time.time()
    res = dict(ref_pd=float(np.mean([pd_rollout(s, T=T)['score'] for s in TEST])),
               ref_random=grade(Poll(np.random.default_rng(0)), TEST, mode='random')['score'], runs={}, neurons={})
    print('reference: pd', round(res['ref_pd'], 3), 'coin-flip', round(res['ref_random'], 3), flush=True)
    for sd in range(a.seeds):
        h, net = live(sd, a.laps, lambda m: print(m, flush=True))
        res['runs'][f'poll{sd}'] = h; res['neurons'][f'poll{sd}'] = describe(net)
        json.dump(res, open(out, 'w'))
    print(f'done in {time.time() - t0:.0f}s')
