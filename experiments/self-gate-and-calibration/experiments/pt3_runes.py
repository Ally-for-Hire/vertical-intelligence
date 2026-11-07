"""PIECE-T3: foundational blocks in a made-up rune universe (the user's belief: the universe is built on foundational blocks and all problems relate; DreamCoder library learning, Ellis et al. 2021 [v-prior]).
Objects: strings of 6 symbols from {0..4}. 6 basic RUNES (atoms): rotate-left, reverse, add-1 (mod 5), copy-first-onto-last, swap-first-two, negate (5 - x mod 5).
STRUCTURED universe: 6 hidden BLOCKS, each a fixed pair of runes; every task = a composition of 2 blocks (4 runes), shown as 5 input -> output examples.
CONTROL universe: every task = 4 random runes (no shared blocks). NEW-DOMAIN transfer: tasks = 3 blocks (6 runes) from the same hidden blocks.
Learners solve tasks by enumerating programs shortest-first over their vocabulary until one fits all 5 examples (cap 60,000 candidates; cost = candidates tried):
  flat    : vocabulary = the 6 runes, forever
  library : starts with the 6 runes; after each solved task, any contiguous pair of vocabulary items used in >= 2 solved programs is added as a new item (a learned block), if it is a new function
Sequence: 60 tasks per universe, 100 seeds. Metrics: median cost on the last 20 tasks; true blocks recovered (library items functionally equal to a hidden block); cost on 20 new-domain tasks after the 60.
PREDICTIONS (registered before running): P71 structured universe: library cost on the last 20 tasks <= 25% of flat; P72 library recovers >= 4 of the 6 hidden blocks (median over seeds);
P73 control universe: library cost >= 80% of flat (no shared blocks, no benefit); P74 new-domain tasks (3 blocks): library cost <= 30% of flat."""
import os, sys, json, time, itertools
import numpy as np
from multiprocessing import Pool

K, L, NEX, CAP, NT = 5, 6, 5, 60000, 60
RUNES = {
    "rot": lambda s: s[1:] + s[:1],
    "rev": lambda s: s[::-1],
    "inc": lambda s: tuple((x + 1) % K for x in s),
    "cpy": lambda s: s[:-1] + s[:1],
    "swp": lambda s: (s[1], s[0]) + s[2:],
    "neg": lambda s: tuple((K - x) % K for x in s),
}
NAMES = list(RUNES)


def apply(prog, s):
    for r in prog:
        s = RUNES[r](s)
    return s


def signature(prog, probes):
    return tuple(apply(prog, p) for p in probes)


def make_task(rng, runes_seq):
    xs = [tuple(int(v) for v in rng.integers(0, K, L)) for _ in range(NEX)]
    return [(x, apply(runes_seq, x)) for x in xs]


def solve(task, vocab):
    """vocab: list of rune tuples; shortest-first enumeration; returns (program as rune tuple or None, cost)"""
    cost = 0
    for n in range(1, 7):
        for combo in itertools.product(range(len(vocab)), repeat=n):
            prog = tuple(r for i in combo for r in vocab[i])
            if len(prog) > 8:
                continue
            cost += 1
            if all(apply(prog, x) == y for x, y in task):
                return combo, prog, cost
            if cost >= CAP:
                return None, None, cost
    return None, None, cost


def run(seed):
    rng = np.random.default_rng(seed)
    probes = [tuple(int(v) for v in rng.integers(0, K, L)) for _ in range(40)]
    blocks = []
    while len(blocks) < 6:                                               # 6 hidden blocks: distinct, non-trivial rune pairs
        b = tuple(rng.choice(NAMES, 2))
        if all(signature(b, probes) != signature(c, probes) for c in blocks) and signature(b, probes) != signature((), probes):
            blocks.append(b)
    out = {}
    for universe in ("structured", "control"):
        trng = np.random.default_rng(seed * 7 + (universe == "control"))
        seqs = []
        for t in range(NT):
            if universe == "structured":
                i, j = trng.integers(0, 6, 2); seqs.append(blocks[i] + blocks[j])
            else:
                seqs.append(tuple(trng.choice(NAMES, 4)))
        tasks = [make_task(trng, s) for s in seqs]
        res = {}
        for learner in ("flat", "library"):
            vocab = [(r,) for r in NAMES]; sigs = {signature(v, probes) for v in vocab}
            used = []; costs = []
            for task in tasks:
                combo, prog, c = solve(task, vocab); costs.append(c)
                if learner == "library" and combo is not None:
                    used.append(combo)
                    pairs = {}
                    for cb in used:
                        for a, b in zip(cb, cb[1:]):
                            pairs[(a, b)] = pairs.get((a, b), 0) + 1
                    for (a, b), cnt in pairs.items():
                        if cnt >= 2:
                            item = vocab[a] + vocab[b]
                            sg = signature(item, probes)
                            if sg not in sigs and len(item) <= 4:
                                vocab.append(item); sigs.add(sg)
            rec = sum(any(signature(v, probes) == signature(b, probes) for v in vocab[6:]) for b in blocks)
            new_costs = []
            if universe == "structured":
                nrng = np.random.default_rng(seed * 13 + 5)
                for t in range(20):
                    i, j, k = nrng.integers(0, 6, 3)
                    new_costs.append(solve(make_task(nrng, blocks[i] + blocks[j] + blocks[k]), vocab)[2])
            res[learner] = dict(last20=float(np.median(costs[-20:])), first10=float(np.median(costs[:10])), recovered=int(rec), vocab=len(vocab), new=float(np.median(new_costs)) if new_costs else None,
                                unsolved=int(sum(c >= CAP for c in costs)))
        out[universe] = res
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(100))
    json.dump(res, open("../results/pt3_runes.json", "w"))
    for u in ("structured", "control"):
        print(f"{u} universe:")
        for lr in ("flat", "library"):
            print(f"   {lr:8s} median cost first 10 tasks {np.median([r[u][lr]['first10'] for r in res]):.0f} | last 20 tasks {np.median([r[u][lr]['last20'] for r in res]):.0f} | unsolved {np.mean([r[u][lr]['unsolved'] for r in res]):.1f}/60 | vocabulary {np.median([r[u][lr]['vocab'] for r in res]):.0f} | hidden blocks recovered {np.median([r[u][lr]['recovered'] for r in res]):.0f}/6")
        rat = np.array([r[u]["library"]["last20"] / max(r[u]["flat"]["last20"], 1) for r in res])
        print(f"   library / flat cost, last 20 tasks: median {np.median(rat):.3f}")
    rn = np.array([r["structured"]["library"]["new"] / max(r["structured"]["flat"]["new"], 1) for r in res])
    print("new-domain tasks (3 blocks): flat median cost %.0f, library %.0f, ratio median %.3f" % (np.median([r["structured"]["flat"]["new"] for r in res]), np.median([r["structured"]["library"]["new"] for r in res]), np.median(rn)))
    print("done", round(time.time() - t0), "s")
