"""X1.13: reward generalization directly; the LESS DIRECT the connection, the MORE a node is paid for assisting (author, 2026-11-02).

A node is credited for helping problems it was NOT built for. Distance is the learner's OWN view: two problems are 1 apart if they share a node,
2 if linked through another problem, and so on; no path = as far as the number of problems seen. When a node made for problem o helps problem t,
its measured help is multiplied by (1 + d(o, t)) (d = 0 for its own problem and for a brand-new node).
Base learner: X1.9's best flat learner (one layer of nodes, the author's link rule, all linked nodes train), which generalized best in X1.12.
Arms:  FLAT (no distance reward)   FLAT_G (distance reward NORMALISED over the candidates of each step: far beats near, the average bar is unchanged)
       FLAT_GR (distance reward RAW, unlimited: also lowers the bar)   FLAT_GC (FLAT_G + X1.12's confirmation on two halves)   DENSE.
Generalization test as X1.12 (new combinations of known concepts, 20 points, connect only).
Usage: python x1_13.py run | log"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time, math
from collections import deque
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_7 as B
import x1_9 as M9
import x1_10 as X
import x1_12 as G

OUT = B.OUT
ARMS = ("FLAT", "FLAT_G", "FLAT_GR", "FLAT_GC", "DENSE")
CTX = {"mode": "FLAT", "links": {}, "origin": {}, "far": []}


def distances_from(t):
    """Hops from problem t to every other problem in the learner's own graph (problems sharing a node are adjacent); one search per step."""
    links = CTX["links"]; node_users = {}
    for p, E in links.items():
        for e in E: node_users.setdefault(e, set()).add(p)
    far = max(1, len(links)); out = {t: 0}; q = deque([t])
    while q:
        p = q.popleft()
        for e in links.get(p, ()):
            for p2 in node_users.get(e, ()):
                if p2 not in out: out[p2] = out[p] + 1; q.append(p2)
    return out, far


def fit_head(arm, units, deg, caches, t, E, P, rng, allow_new, new_from=0, created=None):
    """x1_7.fit_head with (a) the distance reward on reuse of other problems' nodes and (b) optional confirmation on two halves."""
    mode = CTX["mode"]; c = caches[t]; n = len(c.yva); h = n // 2
    confirm = mode == "FLAT_GC"
    for _ in range(C("x17_max_steps")):
        F, Fv, Gi, beta = B.solve(c, E)
        r = c.ytr - F @ beta; pv = Fv @ beta
        idx = np.array([i for i in range(new_from, len(units)) if i not in E], int)
        Z, Zv = c.tr[:, idx], c.va[:, idx]
        nu = None
        if allow_new:
            nu = B.train_unit(rng, c.xtr, r)
            Z = np.hstack([Z, np.tanh(c.xtr @ nu[0] + nu[1])[:, None]]); Zv = np.hstack([Zv, np.tanh(c.xva @ nu[0] + nu[1])[:, None]])
        K = Z.shape[1]
        if K == 0: break
        cost = math.log2(K + 1)
        Wc = Gi @ (F.T @ Z); sch = (Z ** 2).sum(0) + C("x17_ridge") - (Z * (F @ Wc)).sum(0); cf = (Z.T @ r) / sch
        err = c.yva[:, None] - (pv[:, None] + cf * (Zv - Fv @ Wc))
        if confirm:
            base_a = float(B.vbits(c.yva[:h] - pv[:h], h)); base_b = float(B.vbits(c.yva[h:] - pv[h:], n - h))
            help_ = B.vbits(err[:h], h) - base_a; hb = B.vbits(err[h:], n - h) - base_b
        else:
            help_ = B.vbits(err, n) - float(B.vbits(c.yva - pv, n)); hb = None
        isnew = np.zeros(K, bool); dg = np.zeros(K)
        if nu is not None: isnew[-1] = True
        if len(idx): dg[:len(idx)] = np.asarray(deg, float)[idx]
        v = B.adjusted(arm, help_, isnew, dg, P, cost)
        dmap, far = distances_from(t)                                   # measured in every arm (FLAT only records it)
        dist = np.array([dmap.get(CTX["origin"].get(int(i), t), far) for i in idx] + ([0] if nu is not None else []), float)
        if mode in ("FLAT_G", "FLAT_GC", "FLAT_GR"):
            mult = 1 + dist
            if mode != "FLAT_GR": mult = mult / mult.mean()                 # normalised: far beats near, average bar unchanged
            v = np.where(v > 0, v * mult, v)
        if CTX.get("usage"):                                               # X1.14: consistently used nodes pay less (2^-recent use); new nodes pay full
            ucur = np.array([CTX["use"].get(int(i), 0.0) for i in idx] + ([0.0] if nu is not None else []))
            v = np.where(v > 0, v * 2.0 ** (-ucur), v)
        if hb is not None: v = np.where(hb > 0, v, -np.inf)                 # confirmed connections only
        k = int(np.argmax(v))
        if not np.isfinite(v[k]) or v[k] <= cost: break
        if isnew[k]:
            units.append(nu); deg.append(0); j = len(units) - 1
            CTX["origin"][j] = t
            for cx in caches:
                if cx is not None: cx.add(*nu)
            if created is not None: created.append(j)
        else:
            j = int(idx[k]); CTX["far"].append(dist[k])
        E.append(j); deg[j] += 1
        while len(E) > 1:
            F, Fv, Gi, beta = B.solve(c, E); bf = float(B.vbits(c.yva - Fv @ beta, n))
            jj = np.arange(1, len(E) + 1)
            Bm = beta[:, None] - Gi[:, jj] * (beta[jj] / Gi[jj, jj])
            hl = bf - B.vbits(c.yva[:, None] - Fv @ Bm, n)
            av = B.adjusted(arm, hl, np.zeros(len(E), bool), np.array([deg[e] - 1 for e in E], float), P, cost)
            w = int(np.argmin(av))
            if av[w] > cost: break
            deg[E[w]] -= 1; E.pop(w)
    CTX["links"][t] = set(E)
    if CTX.get("usage") and allow_new:                                    # X1.14: one fading step per problem (main call only, not re-checks)
        for k_ in list(CTX["use"]): CTX["use"][k_] *= C("x114_use_keep")
        for e in E: CTX["use"][e] = CTX["use"].get(e, 0.0) + 1
    return E


B.fit_head = fit_head                                                   # x1_9.run_grow calls B.fit_head


def run_flat(arm, data, gens, seed, mode=None, usage=False):
    CTX.update(mode=mode or arm, links={}, origin={}, far=[], usage=usage, use={})
    holder = {}; orig = B.head_bits

    def capture(units, E, d, split="te"):
        holder["units"] = units; return orig(units, E, d, split)
    B.head_bits = capture
    try: r = M9.run_grow("FULLALL", data, np.zeros((1, C("x17_dim"))), np.zeros(1), np.zeros((C("x17_probe_n"), C("x17_dim"))), seed)
    finally: B.head_bits = orig
    units = holder["units"]; rng = np.random.default_rng([seed, 2 + ARMS.index(arm)]); gb = []
    far = float(np.mean(CTX["far"])) if CTX["far"] else 0.0
    P = len(data)
    for gi, d in enumerate(gens):
        c = B.Cache(d)
        for u in units: c.add(*u)
        cs = [None] * (P + 1); cs[P] = c
        E = fit_head("FULL", list(units), [0] * len(units), cs, P, [], P, rng, False)
        CTX["links"].pop(P, None)
        gb.append(B.head_bits(units, E, d))
    return dict(after=r["after"], final=r["final"], nodes=r["units"], gen=gb, far=far)


def world(seed):
    t0 = time.time()
    data, _ = X.make_world(seed); gens = G.gen_problems(seed, data)
    res = {}
    for a in ARMS:
        if a == "DENSE":
            CTX.update(mode="FLAT", links={}, origin={}, far=[], usage=False, use={})
            r = G.run_dense_gen(data, gens, seed); res[a] = dict(after=r["after"], final=r["final"], nodes=r["nodes"], gen=r["gen"], far=0.0)
        else: res[a] = run_flat(a, data, gens, seed)
    half = len(data) // 2
    out = {"arms": {a: {"late": float(np.mean(r["after"][half:])), "early": float(np.mean(r["after"][:half])), "final_all": float(np.mean(r["final"])),
                        "gen": float(np.mean(r["gen"])), "nodes": r["nodes"], "far": r["far"]} for a, r in res.items()}}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(per, seed0):
    from x0_1 import boot_ci
    seeds = sorted(per)
    g = lambda a, k: np.array([per[s]["arms"][a][k] for s in seeds], float)
    S = {"median": {a: {k: float(np.median(g(a, k))) for k in per[seeds[0]]["arms"][a]} for a in ARMS}, "paired": {}, "worlds": len(seeds)}
    for a, b, k in (("FLAT_G", "FLAT", "far"), ("FLAT_G", "FLAT", "gen"), ("FLAT_G", "FLAT", "late"), ("FLAT_GR", "FLAT", "gen"), ("FLAT_GC", "FLAT_G", "gen"),
                    ("FLAT_G", "DENSE", "gen"), ("FLAT_GR", "FLAT", "nodes"), ("FLAT_G", "FLAT", "final_all")):
        S["paired"][f"{k} {a}-{b}"] = boot_ci(g(a, k) - g(b, k), seed0)
    S["vertical FLAT_G late-early"] = boot_ci(g("FLAT_G", "late") - g("FLAT_G", "early"), seed0)
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run. A win = paired 95% range on the predicted side of 0."""
    m, pr = S["median"], S["paired"]
    f = lambda k: f"{pr[k][0]:+.4f} [{pr[k][1]:+.4f}, {pr[k][2]:+.4f}]"
    up = lambda k: pr[k][1] > 0
    down = lambda k: pr[k][2] < 0
    manip = up("far FLAT_G-FLAT")
    V = [("M-X1.13", "Fair test: the distance reward acts: FLAT_G's reused links come from farther problems than FLAT's (mean distance, range above 0)",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"far FLAT_G-FLAT {f('far FLAT_G-FLAT')}")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P423", "Paying more for less direct assistance improves generalization to new combinations: FLAT_G minus FLAT, above 0", up("gen FLAT_G-FLAT"), f("gen FLAT_G-FLAT"))
    v("P424", "It also helps on later problems in the stream: FLAT_G minus FLAT, above 0", up("late FLAT_G-FLAT"), f("late FLAT_G-FLAT"))
    v("P425", "The raw, unlimited distance reward improves generalization too: FLAT_GR minus FLAT, above 0", up("gen FLAT_GR-FLAT"), f("gen FLAT_GR-FLAT"))
    v("P426", "Adding confirmation on two halves helps generalization further: FLAT_GC minus FLAT_G, above 0", up("gen FLAT_GC-FLAT_G"), f("gen FLAT_GC-FLAT_G"))
    v("P427", "The distance-rewarded learner generalizes better than the plain network: FLAT_G minus DENSE, above 0", up("gen FLAT_G-DENSE"), f("gen FLAT_G-DENSE"))
    v("P428", "Vertical: FLAT_G later minus earlier bits, above 0", S["vertical FLAT_G late-early"][1] > 0,
      f"{S['vertical FLAT_G late-early'][0]:+.4f} [{S['vertical FLAT_G late-early'][1]:+.4f}, {S['vertical FLAT_G late-early'][2]:+.4f}]")
    return V


ARM_TEXT = {"FLAT": ("Flat learner (X1.9)", "One layer of nodes, the author's link rule; no reward for distance."),
            "FLAT_G": ("Pays more for distant help", "A node helping a problem far from the one it was made for earns more (normalised: far beats near)."),
            "FLAT_GR": ("Pays more for distant help, unlimited", "The same reward, raw and unlimited (also accepts smaller gains)."),
            "FLAT_GC": ("Distant help + confirmed links", "The normalised distance reward plus X1.12's two-half confirmation."),
            "DENSE": ("Plain neural network", "32 shared units.")}


def log_wandb(out):
    from track import Tracker
    S = out["summary"]
    tk = Tracker("X1.13", "Rewarding help to distant problems", "If a node is paid more the less directly it is connected to the problem it helps, does the learner generalize better?",
                 X.WORLD_TEXT + " After the stream, 10 new problems mix known concepts in new ways, with only 20 examples and no new learning.",
                 ARM_TEXT, B.READ_TEXT + " 'Generalization' = bits held on those new problems.", dict(worlds=S["worlds"]), tags=["generalization", "distance", "flat"])
    keys = [("late", "Bits per point, later problems"), ("final_all", "Bits per point at the end, all problems"), ("gen", "Generalization (higher is better)"),
            ("far", "Mean distance of reused links"), ("nodes", "Nodes in total")]
    tk.table("Summary by arm (median over worlds)", ["Arm"] + [k[1] for k in keys], [[tk.name(a)] + [round(S["median"][a][k], 4) for k, _ in keys] for a in ARMS])
    tk.table("Paired differences with 95% ranges (same worlds, A minus B)", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    import multiprocessing as mp, budget
    t0 = time.time()
    if mode == "log":
        log_wandb(json.load(open(os.path.join(OUT, "x1_13_run.json")))["result"]); return
    seed0 = C("x113_seed0"); nw = C("x110_max_worlds")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    per, history, bs, V, S = {}, [], C("x110_batch"), None, None
    with mp.get_context("spawn").Pool(budget.workers()) as pool:
        for start in range(0, nw, bs):
            seeds = [seed0 + i for i in range(start, min(nw, start + bs))]
            for s_, r in zip(seeds, pool.map(world, seeds, chunksize=1)): per[s_] = r
            S = summarize(per, seed0); V = judge(S); history.append([v[2] for v in V])
            print(f"{len(per)} worlds, {time.time() - t0:.0f} s:", " ".join(f"{v[0]}={v[2].split()[0]}" for v in V), flush=True)
            if len(history) >= 2 and history[-1] == history[-2]: break
    S["stopped_at_worlds"] = len(per)
    out = {"per_world": per, "summary": S, "verdicts": V, "verdict_history": history}
    os.makedirs(OUT, exist_ok=True)
    json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, f"x1_13_{mode}.json"), "w"), indent=1, default=float)
    if mode == "run": log_wandb(out)
    for v in V: print(v[0], v[2], "|", v[3])


if __name__ == "__main__":
    main(sys.argv[1])
