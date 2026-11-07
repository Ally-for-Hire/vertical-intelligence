"""PIECE-B9c: sleep as ABSTRACTION (offline compression of the day's solutions into building blocks; DreamCoder's wake-sleep, Ellis et al. 2021 [v-prior]).
Rune universe as T3 (6 runes, 6 hidden blocks, tasks = 2 blocks), 60 tasks, 100 seeds. Library learners differ only in WHEN and HOW blocks are made:
  awake    : as T3: after every solved task, any pair of vocabulary items used in >= 2 solved programs becomes a block
  sleep    : no block-making while awake; after every 10 tasks ('a day') a sleep makes blocks from all solutions so far (same >= 2 rule)
  refactor : as sleep, and during sleep it RE-SOLVES the day's tasks with the new blocks (shorter programs), recounts pairs on the refactored programs and makes blocks again (second pass); the re-solving search is offline compute, reported separately
Score: total AWAKE search cost over the 60 tasks (median), hidden blocks recovered at the end, offline (sleep) search cost.
PREDICTIONS (registered before running): P141 delaying block-making to sleep costs little: sleep total awake cost <= 1.1 x awake; P142 refactoring in sleep lowers total awake cost to <= 0.85 x awake; P143 refactoring recovers at least as many hidden blocks as awake (median)."""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
import pt3_runes as T

DAY = 10


def make_blocks(vocab, sigs, programs, probes):
    pairs = {}
    for cb in programs:
        for a, b in zip(cb, cb[1:]):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
    for (a, b), cnt in pairs.items():
        if cnt >= 2:
            item = vocab[a] + vocab[b]; sg = T.signature(item, probes)
            if sg not in sigs and len(item) <= 4:
                vocab.append(item); sigs.add(sg)


def run(seed):
    rng = np.random.default_rng(seed)
    probes = [tuple(int(v) for v in rng.integers(0, T.K, T.L)) for _ in range(40)]
    blocks = []
    while len(blocks) < 6:
        b = tuple(rng.choice(T.NAMES, 2))
        if all(T.signature(b, probes) != T.signature(c, probes) for c in blocks) and T.signature(b, probes) != T.signature((), probes):
            blocks.append(b)
    trng = np.random.default_rng(seed * 7)
    tasks = []
    for t in range(T.NT):
        i, j = trng.integers(0, 6, 2); tasks.append(T.make_task(trng, blocks[i] + blocks[j]))
    out = {}
    for mode in ("awake", "sleep", "refactor", "prune"):
        vocab = [(r,) for r in T.NAMES]; sigs = {T.signature(v, probes) for v in vocab}; used = []; cost = 0; offline = 0
        for n, task in enumerate(tasks):
            combo, prog, c = T.solve(task, vocab); cost += c
            if combo is not None:
                used.append(combo)
            if mode == "awake":
                make_blocks(vocab, sigs, used, probes)
            elif (n + 1) % DAY == 0:
                make_blocks(vocab, sigs, used, probes)
                if mode in ("refactor", "prune"):
                    re = []
                    for tk in tasks[:n + 1]:
                        cb, pg, cc = T.solve(tk, vocab); offline += cc
                        if cb is not None:
                            re.append(cb)
                    make_blocks(vocab, sigs, re, probes)
                    used = re
                    if mode == "prune":                                  # keep the base runes + only the learned blocks used by the refactored programs (re-solve once more with the full library first)
                        re2 = []
                        for tk in tasks[:n + 1]:
                            cb, pg, cc = T.solve(tk, vocab); offline += cc
                            if cb is not None:
                                re2.append(cb)
                        keep = set(range(6)) | {i for cb in re2 for i in cb}
                        old = vocab; idx = sorted(keep)
                        remap = {o: k for k, o in enumerate(idx)}
                        vocab = [old[i] for i in idx]; sigs = {T.signature(v, probes) for v in vocab}
                        used = [tuple(remap[i] for i in cb) for cb in re2]
        rec = sum(any(T.signature(v, probes) == T.signature(b, probes) for v in vocab[6:]) for b in blocks)
        out[mode] = dict(cost=float(cost), rec=int(rec), offline=float(offline), vocab=len(vocab))
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(100))
    json.dump(res, open("../results/pb9c_sleep_abstraction.json", "w"))
    for m in ("awake", "sleep", "refactor", "prune"):
        print(f"{m:8s} total awake cost median {np.median([r[m]['cost'] for r in res]):.0f} | hidden blocks recovered median {np.median([r[m]['rec'] for r in res]):.0f} | vocabulary {np.median([r[m]['vocab'] for r in res]):.0f} | offline cost {np.median([r[m]['offline'] for r in res]):.0f}")
    print("sleep / awake cost median %.2f | refactor / awake %.2f | refactor - awake blocks median %.0f" % (np.median([r["sleep"]["cost"] / r["awake"]["cost"] for r in res]), np.median([r["refactor"]["cost"] / r["awake"]["cost"] for r in res]), np.median([r["refactor"]["rec"] - r["awake"]["rec"] for r in res])))
    print("prune / awake cost median %.2f | prune vocabulary - awake vocabulary median %.0f" % (np.median([r["prune"]["cost"] / r["awake"]["cost"] for r in res]), np.median([r["prune"]["vocab"] - r["awake"]["vocab"] for r in res])))
    print("done", round(time.time() - t0), "s")
