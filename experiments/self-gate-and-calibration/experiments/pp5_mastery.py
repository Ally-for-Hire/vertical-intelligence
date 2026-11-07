"""PIECE-P5: mastery and chunking ('understand a problem so well that doing it again costs nothing'; chess chunking, Chase & Simon; automaticity).
Setting: a rotation of 10 different problems (causal worlds with the same layout: 6 core variables, 2 actions, 10 distractors, generator v2), each met 10 times in random order = 100 episodes of 300 ticks.
The learner is never told which problem it is facing. Each episode restarts the problem from a fresh state.
Arms (per-output online MLPs, Adam, exp M settings):
  fresh   : a new network every episode (no memory at all)
  single  : one network carried across all episodes (memory, but every problem overwrites the others)
  library : a store of 'chunks' (one trained network per recognised problem). For the first 50 ticks of an episode it predicts with whichever stored chunk has the lowest error so far, while a fresh
            network trains alongside; at tick 50 it ACCEPTS the best chunk if its mean error < ACCEPT, else keeps the fresh network as a new chunk. The used chunk is saved back at episode end.
  freeze  : as library, but an accepted chunk is NOT trained further while its running error stays below ACCEPT (learning compute ~0 when the problem is mastered)
  oracle  : library with the true problem identity given (upper bound for chunking)
Metrics by encounter number k = 1..10 of each problem: mean prediction error per episode in FIXED evaluator units (each output's error divided by that problem's reference std, never shown to the learner) and number of weight updates (compute). ACCEPT = 0.5 (normalised error; a fresh net starts near 1): an EXPERIMENT setting, untuned.
PREDICTIONS (registered before running): P19 library error at the 2nd encounter <= 50% of the 1st, and at the 10th <= 35% of the 1st; P20 the library recognises the right problem in >= 90% of
repeat encounters (k >= 2); P21 library beats single on mean error over repeat encounters by >= 20%; P22 at the 10th encounter, freeze keeps >= 90% of library's accuracy (error <= library / 0.9) with <= 20% of its updates."""
import os, sys, json, time, copy
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool

NP, REP, EP, REC, ACCEPT, HID, LR = 10, 10, 300, 50, 0.5, 32, 3e-3


def episode_data(pseed, ep_seed, T=None):
    T = T or EP
    from worlds import World
    w = World(pseed, n_state=6, n_act=2, n_distract=10, noise=0.05, max_parents=2)
    rng = np.random.default_rng(ep_seed)
    w.s = rng.normal(0, 1, w.n_state)
    ns, na = w.n_state, w.n_act
    burn = int(os.environ.get("BURN", "0"))                     # P5f control: let the world settle onto its usual behaviour before recording
    a = np.zeros(na)
    for t in range(burn):
        if t % 5 == 0:
            a = rng.integers(-1, 2, size=na).astype(float)
        w.step(w.act_to_world(a))
    X = np.zeros((T, ns + na)); Y = np.zeros((T, ns)); a = np.zeros(na); obs = w.observe()
    for t in range(T):
        if t % 5 == 0:
            a = rng.integers(-1, 2, size=na).astype(float)
        X[t, :ns] = obs; X[t, ns:] = a
        w.step(w.act_to_world(a)); obs = w.observe(); Y[t] = obs
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    return X, (Y - X[:, :ns])[:, cp]


class Chunk:
    def __init__(self, n_in, n_out, seed):
        from nn import AdamMLP, RunStd
        self.nets = [AdamMLP(n_in, 1, HID, LR, seed + j) for j in range(n_out)]
        self.sx, self.sy = RunStd(n_in), RunStd(n_out)

    def _raw(self, p):
        return p * np.sqrt(np.maximum(self.sy.s2, 1e-6)) + self.sy.m

    def step(self, x, d, ref, learn=True):
        """prequential step. Returns (agent_err in this chunk's own standardised units, eval_err in the evaluator's fixed per-problem units, updates)"""
        self.sx.update(x); self.sy.update(d)
        xs, ys = self.sx(x), self.sy(d)
        if learn:
            p = np.array([n.step(xs, ys[j:j + 1])[0] for j, n in enumerate(self.nets)])
        else:
            p = np.array([n.predict(xs)[0][0] for n in self.nets])
        return float(np.mean((p - ys) ** 2)), float(np.mean(((self._raw(p) - d) / ref) ** 2)), (len(self.nets) if learn else 0)


def chunk_error_only(ch, x, d, ref):
    """score a stored chunk on a new tick without changing it (its standardisers stay frozen): (agent_err, eval_err)"""
    xs = ch.sx(x); ys = ch.sy(d)
    p = np.array([n.predict(xs)[0][0] for n in ch.nets])
    return float(np.mean((p - ys) ** 2)), float(np.mean(((ch._raw(p) - d) / ref) ** 2))


def ref_sd(pseed):
    """evaluator-only scale: std of each output's change over a long reference rollout (never shown to any learner)"""
    X, D = episode_data(pseed, 999999, T=2000)
    return D.std(0) + 1e-9


def run(world_set):
    rng = np.random.default_rng(world_set)
    pseeds = [20000 + 100 * world_set + i for i in range(NP)]
    refs = [ref_sd(ps) for ps in pseeds]
    order = rng.permutation(np.repeat(np.arange(NP), REP))
    n_in, n_out = 18, 6
    single = Chunk(n_in, n_out, world_set)
    from nn import RunStd
    G = RunStd(n_out)                                                    # P5g: one stable yardstick = variance of every target the agent has ever seen
    stores = {"library": [], "librel": [], "freeze": [], "libg": [], "libh": []}; owners = {"library": [], "librel": [], "freeze": [], "libg": [], "libh": []}; ora = {}
    seen = np.zeros(NP, int)
    rec = []
    for e, pid in enumerate(order):
        k = int(seen[pid]); seen[pid] += 1
        X, D = episode_data(pseeds[pid], world_set * 100000 + e); ref = refs[pid]
        row = dict(pid=int(pid), k=k)
        ch = Chunk(n_in, n_out, 7 * e + 1)
        row["fresh"] = float(np.mean([ch.step(X[t], D[t], ref)[1] for t in range(EP)])); row["fresh_u"] = EP * n_out
        out = [single.step(X[t], D[t], ref) for t in range(EP)]
        row["single"] = float(np.mean([o[1] for o in out])); row["single_u"] = sum(o[2] for o in out)
        for arm in ("library", "librel", "freeze", "libg", "libh"):
            rel = arm in ("librel", "freeze", "libg", "libh"); glob = arm in ("libg", "libh"); clear = arm == "libh"
            per = []             # P5b: relative rule, no threshold: a stored chunk must beat a fresh net on the same ticks
            store, owner = stores[arm], owners[arm]
            fresh_cum = 0.0
            new = Chunk(n_in, n_out, 7 * e + 3)
            score = np.zeros(len(store)); errs, upd = [], 0
            for t in range(REC):
                use = None
                if len(store) and t > 0 and (score.min() < fresh_cum if rel else score.min() / t < ACCEPT):
                    use = int(np.argmin(score))                       # predict with the best stored chunk only while it is plausibly the right one
                scored = [chunk_error_only(c, X[t], D[t], ref) for c in store]
                if glob:
                    gv = np.maximum(G.s2, 1e-6)
                    row_e = [float(np.mean((c._raw(np.array([n.predict(c.sx(X[t]))[0][0] for n in c.nets])) - D[t]) ** 2 / gv)) for c in store]
                    fe = float(np.mean((new._raw(np.array([n.predict(new.sx(X[t]))[0][0] for n in new.nets])) - D[t]) ** 2 / gv))
                    for i, v_ in enumerate(row_e):
                        score[i] += v_
                    fresh_cum += fe
                    per.append(row_e + [fe])
                else:
                    for i, sc in enumerate(scored):
                        score[i] += sc[0]
                ag, ev, u = new.step(X[t], D[t], ref); upd += u
                if not glob:
                    fresh_cum += ag
                errs.append(scored[use][1] if use is not None else ev)
            ok = len(store) > 0 and ((score.min() < fresh_cum) if rel else (score.min() / REC < ACCEPT))
            if clear and len(store):                                     # P5h: accept only if the best stored chunk is SIGNIFICANTLY better than every other candidate (other chunks and a fresh net)
                E = np.array(per)                                        # (REC, n_chunks + 1); last column = fresh net
                b = int(np.argmin(E[:, :-1].sum(0)))
                zs = []
                for q in range(E.shape[1]):
                    if q == b:
                        continue
                    dq = E[:, q] - E[:, b]
                    zs.append(dq.mean() / (dq.std(ddof=1) / np.sqrt(len(dq)) + 1e-12))
                ok = min(zs) > 2.326                                     # one-sided alpha 0.01 against the runner-up
                score = E[:, :-1].sum(0)
            chosen = int(np.argmin(score)) if len(store) and ok else None
            cur = store[chosen] if chosen is not None else new
            run_err = ACCEPT * 0.5
            for t in range(REC, EP):
                learn = not (arm == "freeze" and chosen is not None and run_err < ACCEPT)
                ag, ev, u = cur.step(X[t], D[t], ref, learn); upd += u
                run_err += (ag - run_err) / 20
                errs.append(ev)
            if chosen is None:
                store.append(cur); owner.append(int(pid)); outcome = "new"
            else:
                outcome = "own" if owner[chosen] == pid else "other"
            row[arm] = float(np.mean(errs)); row[arm + "_u"] = upd; row[arm + "_outcome"] = outcome; row[arm + "_nchunks"] = len(store)
        for t in range(EP):
            G.update(D[t])
        if pid not in ora:
            ora[pid] = Chunk(n_in, n_out, 7 * e + 5)
        out = [ora[pid].step(X[t], D[t], ref) for t in range(EP)]
        row["oracle"] = float(np.mean([o[1] for o in out])); row["oracle_u"] = sum(o[2] for o in out)
        rec.append(row)
    return rec


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(24))
    json.dump(res, open("../results/pp5h_mastery.json", "w"))
    rows = [r for s in res for r in s]
    arms = ("fresh", "librel", "libg", "libh", "oracle")
    print("mean error by encounter k (1 = first time) | 24 world sets x 10 problems")
    print("MEDIAN error; k   " + "  ".join(f"{a:>8s}" for a in arms) + "   | updates: librel  freeze")
    for k in range(REP):
        sel = [r for r in rows if r["k"] == k]
        print(f"{k + 1:2d}  " + "  ".join(f"{np.median([r[a] for r in sel]):8.3f}" for a in arms) + f"   | {np.mean([r['librel_u'] for r in sel]):7.0f} {np.mean([r['freeze_u'] for r in sel]):7.0f}")
    rep = [r for r in rows if r["k"] > 0]
    for arm in ("librel", "libg", "libh"):
        oc = [r[arm + "_outcome"] for r in rows if r["k"] > 0]
        print(f"{arm} on repeat encounters: reused OWN chunk {oc.count('own') / len(oc):.3f}, reused ANOTHER problem's chunk {oc.count('other') / len(oc):.3f}, made a new chunk {oc.count('new') / len(oc):.3f}; chunks per world set {np.mean([s[-1][arm + '_nchunks'] for s in res]):.1f} (true problems: 10)")
    eh = np.median([r["libh"] for r in rows if r["k"] == 0])
    print("libh: 1st-encounter median %.3f (fresh %.3f); 10th/1st %.2f; repeat median %.3f vs oracle %.3f" % (eh, np.median([r["fresh"] for r in rows if r["k"] == 0]), np.median([r["libh"] for r in rows if r["k"] == 9]) / eh, np.median([r["libh"] for r in rows if r["k"] > 0]), np.median([r["oracle"] for r in rows if r["k"] > 0])))
    eg = np.median([r["libg"] for r in rows if r["k"] == 0])
    print("libg median error ratio vs first encounter: k=2 %.2f, k=10 %.2f; repeat median libg %.3f vs oracle %.3f" % (np.median([r["libg"] for r in rows if r["k"] == 1]) / eg, np.median([r["libg"] for r in rows if r["k"] == 9]) / eg, np.median([r["libg"] for r in rows if r["k"] > 0]), np.median([r["oracle"] for r in rows if r["k"] > 0])))
    e1 = np.median([r["librel"] for r in rows if r["k"] == 0])
    print("librel median error ratio vs first encounter: k=2 %.2f, k=10 %.2f" % (np.median([r["librel"] for r in rows if r["k"] == 1]) / e1, np.median([r["librel"] for r in rows if r["k"] == 9]) / e1))
    s10 = [r for r in rows if r["k"] == 9]
    print("10th encounter: freeze/librel median error %.2f, updates %.2f" % (np.median([r["freeze"] for r in s10]) / np.median([r["librel"] for r in s10]), np.mean([r["freeze_u"] for r in s10]) / np.mean([r["librel_u"] for r in s10])))
    print("done", round(time.time() - t0), "s")
