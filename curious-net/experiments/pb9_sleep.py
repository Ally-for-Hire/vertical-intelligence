"""PIECE-B9: sleep as offline consolidation, tested on the fragmentation problem found in PIECE-P5b.
Same task as P5/P5b: 10 problems x 10 encounters in random order (24 world sets), relative recognition rule (a stored chunk is used only if it beats a fresh network on the first 50 ticks).
Every chunk keeps the last 100 samples it saw (its replay buffer). Every 10 episodes the learner 'sleeps'.
Arms:
  awake   : no sleep (= P5b librel)
  rehearse: sleep = each chunk replays its own buffer once (pure rehearsal: the equal-compute control, no merging)
  merge   : sleep = the same rehearsal, plus MERGING: chunk j is merged into chunk i if i predicts j's replay buffer at least as well as j predicts it itself (no threshold); i is then trained on j's buffer
            (that compute is counted) and j is deleted
  oracle  : identity given (reference)
PREDICTIONS (registered before running): P27 merge ends with <= 20 chunks per world set (P5b: 42); P28 merge re-uses the problem's own chunk in >= 70% of repeat encounters (P5b: 50%);
P29 merge's median error at the 10th encounter <= 0.80 x its 1st (P5b: 0.91) AND its median error over repeat encounters is >= 5% below rehearse's."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pp5_mastery as P

SLEEP_EVERY, KEEP = 10, 100


def score_on(ch, buf, ref):
    return float(np.mean([P.chunk_error_only(ch, x, d, ref)[0] for x, d, ref in buf])) if buf else np.inf


def sleep(store, owner, mode, stats):
    for ch in store:                                                     # rehearsal (both sleeping arms)
        for x, d, ref in ch.buf:
            stats["upd"] += ch.step(x, d, ref)[2]
    if mode != "merge":
        return
    merged = True
    while merged and len(store) > 1:
        merged = False
        own = [score_on(c, c.buf, None) if False else score_on(c, c.buf, None) for c in store]
        for j in range(len(store)):
            for i in range(len(store)):
                if i == j:
                    continue
                if score_on(store[i], store[j].buf, None) <= own[j]:
                    for x, d, ref in store[j].buf:
                        stats["upd"] += store[i].step(x, d, ref)[2]
                    store[i].buf = (store[i].buf + store[j].buf)[-KEEP:]
                    stats["merges"] += 1; stats["merged_same"] += int(owner[i] == owner[j])
                    del store[j]; del owner[j]
                    merged = True
                    break
            if merged:
                break


def run(world_set):
    rng = np.random.default_rng(world_set)
    pseeds = [20000 + 100 * world_set + i for i in range(P.NP)]
    refs = [P.ref_sd(ps) for ps in pseeds]
    order = rng.permutation(np.repeat(np.arange(P.NP), P.REP))
    n_in, n_out = 18, 6
    stores = {a: [] for a in ("awake", "rehearse", "merge")}; owners = {a: [] for a in stores}
    stats = {a: dict(upd=0, merges=0, merged_same=0) for a in stores}
    ora = {}; seen = np.zeros(P.NP, int); rec = []
    for e, pid in enumerate(order):
        k = int(seen[pid]); seen[pid] += 1
        X, D = P.episode_data(pseeds[pid], world_set * 100000 + e); ref = refs[pid]
        row = dict(pid=int(pid), k=k)
        for arm in stores:
            store, owner = stores[arm], owners[arm]
            new = P.Chunk(n_in, n_out, 7 * e + 3); new.buf = []
            score = np.zeros(len(store)); fresh_cum = 0.0; errs = []
            for t in range(P.REC):
                use = int(np.argmin(score)) if len(store) and t > 0 and score.min() < fresh_cum else None
                scored = [P.chunk_error_only(c, X[t], D[t], ref) for c in store]
                for i, sc in enumerate(scored):
                    score[i] += sc[0]
                ag, ev, u = new.step(X[t], D[t], ref); stats[arm]["upd"] += u; fresh_cum += ag
                errs.append(scored[use][1] if use is not None else ev)
            chosen = int(np.argmin(score)) if len(store) and score.min() < fresh_cum else None
            cur = store[chosen] if chosen is not None else new
            for t in range(P.REC, P.EP):
                ag, ev, u = cur.step(X[t], D[t], ref); stats[arm]["upd"] += u
                errs.append(ev)
            cur.buf = (cur.buf + [(X[t], D[t], ref) for t in range(P.EP - KEEP, P.EP)])[-KEEP:]
            if chosen is None:
                store.append(cur); owner.append(int(pid)); outcome = "new"
            else:
                outcome = "own" if owner[chosen] == pid else "other"
            row[arm] = float(np.mean(errs)); row[arm + "_outcome"] = outcome
            if arm != "awake" and (e + 1) % SLEEP_EVERY == 0:
                sleep(store, owner, arm, stats[arm])
            row[arm + "_nchunks"] = len(store)
        if pid not in ora:
            ora[pid] = P.Chunk(n_in, n_out, 7 * e + 5)
        row["oracle"] = float(np.mean([ora[pid].step(X[t], D[t], ref)[1] for t in range(P.EP)]))
        rec.append(row)
    return dict(rows=rec, stats=stats)


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(24))
    json.dump(res, open("../results/pb9_sleep.json", "w"), default=float)
    rows = [r for s in res for r in s["rows"]]
    arms = ("awake", "rehearse", "merge", "oracle")
    print("MEDIAN error by encounter k")
    for k in range(P.REP):
        sel = [r for r in rows if r["k"] == k]
        print(f"{k + 1:2d}  " + "  ".join(f"{a} {np.median([r[a] for r in sel]):.3f}" for a in arms))
    for a in ("awake", "rehearse", "merge"):
        oc = [r[a + "_outcome"] for r in rows if r["k"] > 0]
        e1 = np.median([r[a] for r in rows if r["k"] == 0]); e10 = np.median([r[a] for r in rows if r["k"] == 9])
        st = [s["stats"][a] for s in res]
        print(f"{a:8s}: own {oc.count('own') / len(oc):.3f} other {oc.count('other') / len(oc):.3f} new {oc.count('new') / len(oc):.3f} | final chunks {np.mean([s['rows'][-1][a + '_nchunks'] for s in res]):.1f} | "
              f"10th/1st {e10 / e1:.2f} | repeat median {np.median([r[a] for r in rows if r['k'] > 0]):.3f} | updates {np.mean([x['upd'] for x in st]):.0f} | merges {np.mean([x['merges'] for x in st]):.1f} (same-problem {np.sum([x['merged_same'] for x in st]) / max(np.sum([x['merges'] for x in st]), 1):.2f})")
    print("done", round(time.time() - t0), "s")
