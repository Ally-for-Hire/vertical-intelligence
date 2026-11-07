"""PIECE-T3b: a TRUE null and a dose-response for the rune universe (follow-up to T3, whose control shared pieces by chance).
10 runes (T3's 6 + rotate-right, double mod 5, swap-last-two, zero-first). 30 tasks per run (each 4 runes = 3 adjacent pairs). A fraction f of tasks is built from 6 hidden blocks (2 blocks each);
the rest are 'unique' tasks whose 3 adjacent rune pairs never occur in any other task (sampled without reuse from the 100 possible pairs). f in {0, 0.25, 0.5, 0.75, 1}. Solvers as T3 (flat vs library). 100 seeds.
Score: median cost on the last 10 tasks, flat / library ratio ('advantage').
PREDICTIONS (registered before running): P75 at f = 0 (true null) library cost >= 90% of flat; P76 the advantage (flat/library) rises monotonically with f across the 5 levels; P77 at f = 1 library cost <= 35% of flat."""
import os, sys, json, time, itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, ".")
import pt3_runes as T

K = T.K
T.RUNES.update({
    "rotr": lambda s: s[-1:] + s[:-1],
    "dbl": lambda s: tuple((2 * x) % K for x in s),
    "swl": lambda s: s[:-2] + (s[-1], s[-2]),
    "z0": lambda s: (0,) + s[1:],
})
NAMES = list(T.RUNES)
NT, FS = 30, (0.0, 0.25, 0.5, 0.75, 1.0)


def run(seed):
    rng = np.random.default_rng(seed)
    probes = [tuple(int(v) for v in rng.integers(0, K, T.L)) for _ in range(40)]
    out = {}
    for f in FS:
        r2 = np.random.default_rng(seed * 31 + int(100 * f))
        blocks = []
        while len(blocks) < 6:
            b = tuple(r2.choice(NAMES, 2))
            if all(T.signature(b, probes) != T.signature(c, probes) for c in blocks) and T.signature(b, probes) != T.signature((), probes):
                blocks.append(b)
        used_pairs = set(blocks)
        n_struct = int(round(f * NT)); seqs = []
        for t in range(n_struct):
            i, j = r2.integers(0, 6, 2); seqs.append(blocks[i] + blocks[j])
        tries = 0
        while len(seqs) < NT and tries < 100000:                         # unique tasks: none of their 3 adjacent pairs appear anywhere else
            tries += 1
            s = tuple(r2.choice(NAMES, 4)); pr = {(s[0], s[1]), (s[1], s[2]), (s[2], s[3])}
            if len(pr) == 3 and not (pr & used_pairs):
                seqs.append(s); used_pairs |= pr
        order = r2.permutation(len(seqs)); seqs = [seqs[i] for i in order]
        tasks = [T.make_task(r2, s) for s in seqs]
        res = {}
        for learner in ("flat", "library"):
            vocab = [(r,) for r in NAMES]; sigs = {T.signature(v, probes) for v in vocab}; used = []; costs = []
            for task in tasks:
                combo, prog, c = T.solve(task, vocab); costs.append(c)
                if learner == "library" and combo is not None:
                    used.append(combo); pairs = {}
                    for cb in used:
                        for a, b in zip(cb, cb[1:]):
                            pairs[(a, b)] = pairs.get((a, b), 0) + 1
                    for (a, b), cnt in pairs.items():
                        if cnt >= 2:
                            item = vocab[a] + vocab[b]; sg = T.signature(item, probes)
                            if sg not in sigs and len(item) <= 4:
                                vocab.append(item); sigs.add(sg)
            res[learner] = float(np.median(costs[-10:]))
        res["n_tasks"] = len(seqs)
        out[str(f)] = res
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(int(os.environ.get("OFF", "0")), int(os.environ.get("OFF", "0")) + 100))
    json.dump(res, open("../results/pt3b_runes%s.json" % os.environ.get("OFF", ""), "w"))
    print("fraction of tasks built from shared blocks | median cost last 10: flat / library | advantage flat/library (median of per-seed ratios)")
    for f in FS:
        fl = np.median([r[str(f)]["flat"] for r in res]); lb = np.median([r[str(f)]["library"] for r in res])
        adv = np.median([r[str(f)]["flat"] / max(r[str(f)]["library"], 1) for r in res])
        print(f"  f={f:.2f} | {fl:.0f} / {lb:.0f} | {adv:.2f} (tasks generated: {np.mean([r[str(f)]['n_tasks'] for r in res]):.1f})")
    print("done", round(time.time() - t0), "s")
