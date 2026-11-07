"""PIECE-P1: innate core knowledge (Spelke & Kinzler 2007 [v]; ARC's priors, Chollet 2019 [v]; counterpoint: the Bitter Lesson, Sutton 2019 [v]), tested in the T3 rune universe (structured: 6 hidden blocks, tasks = 2 blocks).
Library learner (as T3) born with: NONE (6 runes only); RIGHT priors (2 of the 6 hidden blocks already in its vocabulary); WRONG priors (2 rune pairs that are never used by any hidden block).
60 tasks, 100 seeds. Median search cost on the first 10 and the last 20 tasks.
PREDICTIONS (registered before running): P108 right priors cut first-10 cost by >= 40% vs none; P109 by the last 20 tasks the right-prior advantage (none / right) is at most half of its first-10 value in log terms
(learning catches up); P110 wrong priors raise first-10 cost by >= 10% vs none."""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
import pt3_runes as T


def run(seed):
    rng = np.random.default_rng(seed)
    probes = [tuple(int(v) for v in rng.integers(0, T.K, T.L)) for _ in range(40)]
    blocks = []
    while len(blocks) < 6:
        b = tuple(rng.choice(T.NAMES, 2))
        if all(T.signature(b, probes) != T.signature(c, probes) for c in blocks) and T.signature(b, probes) != T.signature((), probes):
            blocks.append(b)
    used_sigs = {T.signature(b, probes) for b in blocks}
    wrong = []
    for a in T.NAMES:
        for b in T.NAMES:
            sg = T.signature((a, b), probes)
            if sg not in used_sigs and sg != T.signature((), probes) and all(sg != T.signature(c, probes) for c in [(r,) for r in T.NAMES]) and len(wrong) < 2:
                wrong.append((a, b))
    trng = np.random.default_rng(seed * 7)
    tasks = []
    for t in range(T.NT):
        i, j = trng.integers(0, 6, 2); tasks.append(T.make_task(trng, blocks[i] + blocks[j]))
    out = {}
    for prior, extra in (("none", []), ("right", blocks[:2]), ("wrong", wrong)):
        vocab = [(r,) for r in T.NAMES] + list(extra); sigs = {T.signature(v, probes) for v in vocab}; used = []; costs = []
        for task in tasks:
            combo, prog, c = T.solve(task, vocab); costs.append(c)
            if combo is not None:
                used.append(combo); pairs = {}
                for cb in used:
                    for a, b in zip(cb, cb[1:]):
                        pairs[(a, b)] = pairs.get((a, b), 0) + 1
                for (a, b), cnt in pairs.items():
                    if cnt >= 2:
                        item = vocab[a] + vocab[b]; sg = T.signature(item, probes)
                        if sg not in sigs and len(item) <= 4:
                            vocab.append(item); sigs.add(sg)
        out[prior] = dict(first10=float(np.median(costs[:10])), last20=float(np.median(costs[-20:])))
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(int(os.environ.get("OFF", "0")), int(os.environ.get("OFF", "0")) + 100))
    json.dump(res, open("../results/pp1_priors%s.json" % os.environ.get("OFF", ""), "w"))
    for pr in ("none", "right", "wrong"):
        print(f"{pr:6s} median cost first 10: {np.median([r[pr]['first10'] for r in res]):.0f} | last 20: {np.median([r[pr]['last20'] for r in res]):.0f}")
    f = np.array([r["none"]["first10"] / r["right"]["first10"] for r in res]); l = np.array([r["none"]["last20"] / r["right"]["last20"] for r in res])
    print("right-prior advantage (none / right): first 10 median %.2f, last 20 median %.2f | first-10 cost cut %.0f%%" % (np.median(f), np.median(l), 100 * (1 - np.median([r['right']['first10'] / r['none']['first10'] for r in res]))))
    print("wrong / none first 10: median %.2f" % np.median([r["wrong"]["first10"] / r["none"]["first10"] for r in res]))
    print("done", round(time.time() - t0), "s")
