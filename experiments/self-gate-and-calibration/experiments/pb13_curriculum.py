"""PIECE-B13: development / curriculum: does the ORDER of teaching matter for a learner that builds its own blocks? (Bengio et al. 2009 curriculum learning [u]; developmental stages)
Rune universe (T3): 6 hidden blocks; the SAME 60 tasks for every order: 20 of depth 1 (one block), 20 of depth 2, 20 of depth 3. Library learner as T3.
Orders: curriculum (depth 1 -> 2 -> 3), random (shuffled), anti (depth 3 -> 2 -> 1). 100 seeds. Search cap 60,000 candidates (an unsolved task costs the cap).
PREDICTIONS (registered before running): P130 curriculum total search cost <= 0.7 x random order; P131 anti-curriculum total cost >= 1.2 x random; P132 after the first 20 tasks the curriculum learner has recovered >= 2 more hidden blocks than the random-order learner (median)."""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
import pt3_runes as T


def learn(tasks, probes, blocks):
    vocab = [(r,) for r in T.NAMES]; sigs = {T.signature(v, probes) for v in vocab}; used = []; costs = []; rec20 = None
    for n, task in enumerate(tasks):
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
            if len(combo) == 1 and len(prog) == 2:                       # a depth-1 task solved by ONE new item is evidence of a block, but learning happens only through re-use
                pass
        if n == 19:
            rec20 = sum(any(T.signature(v, probes) == T.signature(b, probes) for v in vocab[6:]) for b in blocks)
    return float(np.sum(costs)), int(rec20), int(sum(c >= T.CAP for c in costs))


def run(seed):
    rng = np.random.default_rng(seed)
    probes = [tuple(int(v) for v in rng.integers(0, T.K, T.L)) for _ in range(40)]
    blocks = []
    while len(blocks) < 6:
        b = tuple(rng.choice(T.NAMES, 2))
        if all(T.signature(b, probes) != T.signature(c, probes) for c in blocks) and T.signature(b, probes) != T.signature((), probes):
            blocks.append(b)
    trng = np.random.default_rng(seed * 11)
    tasks = {1: [], 2: [], 3: []}
    for depth in (1, 2, 3):
        for _ in range(20):
            seq = sum((blocks[int(i)] for i in trng.integers(0, 6, depth)), ())
            tasks[depth].append(T.make_task(trng, seq))
    allt = tasks[1] + tasks[2] + tasks[3]
    order_r = list(np.random.default_rng(seed * 13).permutation(60))
    out = {}
    for name, seq in (("curriculum", allt), ("random", [allt[i] for i in order_r]), ("anti", tasks[3] + tasks[2] + tasks[1])):
        out[name] = learn(seq, probes, blocks)
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(int(os.environ.get("OFF", "0")), int(os.environ.get("OFF", "0")) + 100))
    json.dump(res, open("../results/pb13_curriculum%s.json" % os.environ.get("OFF", ""), "w"))
    for o in ("curriculum", "random", "anti"):
        print(f"{o:10s} total cost median {np.median([r[o][0] for r in res]):.0f} | hidden blocks recovered after 20 tasks median {np.median([r[o][1] for r in res]):.0f} | unsolved tasks {np.mean([r[o][2] for r in res]):.1f}")
    print("curriculum / random total: median %.2f | anti / random: median %.2f | blocks-after-20 difference curriculum - random: median %.0f" % (
        np.median([r["curriculum"][0] / r["random"][0] for r in res]), np.median([r["anti"][0] / r["random"][0] for r in res]), np.median([r["curriculum"][1] - r["random"][1] for r in res])))
    print("done", round(time.time() - t0), "s")
