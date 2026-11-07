"""X1.12: reward NEW FRONTIERS and CORRECT connections, and test GENERALIZATION (author, 2026-11-02: 'reward new frontiers specifically for now.
more correct connections about topics = better. we want to see generalization'; the fresh-evidence check from X1.11 was 'the right direction').

Tree of X1.11 (precision reward 1/MSE, internal self-weighting, enter at the base, only used nodes learn), with:
  CORRECT: each problem's decision points are split in half. A candidate is chosen on the first half; it is added only if it ALSO raises
           precision on the second half (a confirmed, 'correct' connection). Luck on one half rarely repeats on the other.
  FRONTIER: a confirmed NEW node (structure the tree did not have) is paid x (1 + x112_frontier).
GENERALIZATION test after the stream: 10 new problems, each mixing 3 concepts the stream already contained in a combination it never showed, with only
20 train + 20 decision points; the learner may only connect what it already has (no new nodes, no training). DENSE connects its 32 units by the same measured-help rule.
Arms: TREE_CF (correct + frontier)  TREE_C (correct, no frontier bonus)  TREE_P (X1.11, no confirmation)  FLAT (X1.9 best, link re-use only on the
generalization test)  DENSE.
Usage: python x1_12.py run | log"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_7 as B
import x1_10 as X
import x1_11 as P11

OUT = B.OUT
ARMS = ("TREE_CF", "TREE_C", "TREE_P", "FLAT", "DENSE")
_p11_grow = P11.grow_head


def grow_confirmed(arm, tree, c, P, rng, allow_new=True):
    E = [0]; floor = C("x17_mse_floor"); n = len(c.yva); h = n // 2
    A, Bh = slice(0, h), slice(h, n)
    for _ in range(C("x110_max_used")):
        F, Fv, Gi, beta = B.solve(c, E); r = c.ytr - F @ beta; pv = Fv @ beta
        R0a = 1.0 / max(float(np.mean((c.yva[A] - pv[A]) ** 2)), floor); R0b = 1.0 / max(float(np.mean((c.yva[Bh] - pv[Bh]) ** 2)), floor)
        kids = sorted({k for e in E for k in tree.kids[e]} - set(E))
        Z, Zv = c.tr[:, kids], c.va[:, kids]; cand = list(kids); isnew = [False] * len(kids); props = []
        if allow_new:
            best_p = E[int(np.argmax([abs(np.corrcoef(c.tr[:, e], r)[0, 1]) if c.tr[:, e].std() > 0 else 0 for e in E]))]
            for p in sorted({best_p, 0}):
                nd, ztr, zva = X.new_node(rng, tree, c, p, r); props.append(nd)
                Z = np.hstack([Z, ztr[:, None]]); Zv = np.hstack([Zv, zva[:, None]]); cand.append(-len(props)); isnew.append(True)
        K = Z.shape[1]
        if K == 0: break
        cost = math.log2(K + 1)
        Wc = Gi @ (F.T @ Z); sch = (Z ** 2).sum(0) + C("x17_ridge") - (Z * (F @ Wc)).sum(0); cf = (Z.T @ r) / sch
        err = c.yva[:, None] - (pv[:, None] + cf * (Zv - Fv @ Wc))
        ga = h * (1.0 / np.maximum(np.mean(err[A] ** 2, 0), floor) - R0a)
        gb = (n - h) * (1.0 / np.maximum(np.mean(err[Bh] ** 2, 0), floor) - R0b)
        isnew = np.array(isnew, bool)
        mult, _ = X.multipliers("TREE", tree, P, cand, isnew, len(E))
        if arm == "TREE_CF": mult = mult * np.where(isnew, 1 + C("x112_frontier"), 1.0)
        v = np.where((ga > 0) & (gb > 0), ga * mult, -np.inf)          # only CONFIRMED connections count
        k = int(np.argmax(v))
        if not np.isfinite(v[k]) or v[k] <= cost: break
        if isnew[k]:
            u, vv, b, p = props[-cand[k] - 1]; j = tree.add(u, vv, b, p); c.update([j])
        else: j = cand[k]
        E.append(j)
    if allow_new:
        for e in E: tree.deg[e] += 1
    return sorted(E)


def grow_head(arm, tree, c, P, rng):
    if arm in ("TREE_CF", "TREE_C"): return grow_confirmed(arm, tree, c, P, rng)
    return _p11_grow(arm, tree, c, P, rng)


X.grow_head = grow_head
X.ARMS = ARMS


def gen_problems(seed, data):
    """New combinations of concepts already met in the stream; few points. Rebuilds the world's concepts from the same seed."""
    rng = np.random.default_rng([seed, 1])
    m, T, k = C("x17_dim"), C("x17_problems"), C("x17_k")
    W, V, Bb, Par = [], [], [], []

    def newc(root):
        p = -1 if (root or rng.random() >= C("x110_p_child")) else int(rng.integers(len(W)))
        W.append(rng.standard_normal(m) * C("x17_gain") / math.sqrt(m)); Bb.append(rng.standard_normal() * C("x17_bias_sd"))
        V.append(0.0 if p < 0 else float(rng.choice([-1, 1]) * rng.uniform(C("x110_v_lo"), C("x110_v_hi")))); Par.append(p)
    for _ in range(C("x17_pool0")): newc(True)
    for t in range(T):                                                  # replay X.make_world's draws exactly so the concepts are identical
        if rng.random() < C("x17_p_new"):
            newc(False); rng.choice(len(W) - 1, k - 1, replace=False)
        else: rng.choice(len(W), k, replace=False)
        rng.choice([-1, 1], k); rng.uniform(C("x17_a_lo"), C("x17_a_hi"), k)
    seen = sorted({j for d in data for j in d["S"]}); combos = {tuple(sorted(d["S"])) for d in data}
    g = np.random.default_rng([seed, C("x112_gen_stream")]); out = []
    Wm, Vm, Bm = np.array(W), np.array(V), np.array(Bb)

    def concepts(x):
        Cc = np.zeros((len(x), len(W)))
        for j in range(len(W)): Cc[:, j] = np.tanh(x @ Wm[j] + Bm[j] + (Vm[j] * Cc[:, Par[j]] if Par[j] >= 0 else 0))
        return Cc
    while len(out) < C("x112_gen_n"):
        S = sorted(g.choice(seen, k, replace=False).tolist())
        if tuple(S) in combos: continue
        a = g.choice([-1, 1], k) * g.uniform(C("x17_a_lo"), C("x17_a_hi"), k)
        nf = C("x112_gen_tr") * 2 + C("x17_n_te") + C("x17_norm_n")
        x = g.standard_normal((nf, m)); y = concepts(x)[:, S] @ a; y = y + g.standard_normal(nf) * C("x17_noise") * y.std()
        mu, sd = y[-C("x17_norm_n"):].mean(), y[-C("x17_norm_n"):].std(); y = (y - mu) / sd
        i1, i2, i3 = C("x112_gen_tr"), 2 * C("x112_gen_tr"), 2 * C("x112_gen_tr") + C("x17_n_te")
        Cg = concepts(x)[:, S]                                          # true concepts (no random draws): per-problem ceiling
        out.append(dict(tr=(x[:i1], y[:i1]), va=(x[i1:i2], y[i1:i2]), te=(x[i2:i3], y[i2:i3]), S=S, Ctrue=(Cg[:i1], Cg[i1:i2], Cg[i2:i3])))
    return out


def run_tree_gen(arm, data, gens, seed):
    """Run the stream (as X.run_tree), then the generalization problems with the final tree: connect only, no new nodes, no training."""
    rng = np.random.default_rng([seed, 2 + ARMS.index(arm)])
    holder = {}
    orig_bits = X.tree_bits

    def capture(tree, E, d, split="te"):
        holder["tree"] = tree; return orig_bits(tree, E, d, split)
    X.tree_bits = capture
    try: r = X.run_tree(arm, data, seed)
    finally: X.tree_bits = orig_bits
    tree = holder["tree"]; gb = []
    for d in gens:
        c = X.TCache(d, tree)
        E = grow_confirmed(arm, tree, c, len(data), rng, allow_new=False) if arm in ("TREE_CF", "TREE_C") else _gen_unconfirmed(tree, c, len(data))
        gb.append(X.tree_bits(tree, E, d))
    r["gen"] = gb
    return r


def _gen_unconfirmed(tree, c, P):
    """TREE_P on the generalization test: same greedy connect step, plain measured precision, no new nodes."""
    E = [0]; floor = C("x17_mse_floor"); n = len(c.yva)
    for _ in range(C("x110_max_used")):
        F, Fv, Gi, beta = B.solve(c, E); r = c.ytr - F @ beta; pv = Fv @ beta; R0 = 1.0 / max(float(np.mean((c.yva - pv) ** 2)), floor)
        kids = sorted({k for e in E for k in tree.kids[e]} - set(E))
        if not kids: break
        Z, Zv = c.tr[:, kids], c.va[:, kids]; cost = math.log2(len(kids) + 1)
        Wc = Gi @ (F.T @ Z); sch = (Z ** 2).sum(0) + C("x17_ridge") - (Z * (F @ Wc)).sum(0); cf = (Z.T @ r) / sch
        gain = n * (1.0 / np.maximum(np.mean((c.yva[:, None] - (pv[:, None] + cf * (Zv - Fv @ Wc))) ** 2, 0), floor) - R0)
        mult, _ = X.multipliers("TREE", tree, P, kids, np.zeros(len(kids), bool), len(E))
        v = np.where(gain > 0, gain * mult, gain); k = int(np.argmax(v))
        if v[k] <= cost: break
        E.append(kids[k])
    return sorted(E)


def run_flat_gen(data, gens, seed):
    import x1_9 as M9
    holder = {}
    orig = M9.B.head_bits

    def capture(units, E, d, split="te"):
        holder["units"] = units; return orig(units, E, d, split)
    M9.B.head_bits = capture
    try: r = M9.run_grow("FULLALL", data, np.zeros((1, C("x17_dim"))), np.zeros(1), np.zeros((C("x17_probe_n"), C("x17_dim"))), seed)
    finally: M9.B.head_bits = orig
    units = holder["units"]; deg = [0] * len(units); rng = np.random.default_rng([seed, 2 + ARMS.index("FLAT")]); gb = []
    for d in gens:
        c = B.Cache(d)
        for u in units: c.add(*u)
        E = B.fit_head("FULL", list(units), list(deg), [c], 0, [], len(data), rng, False)
        gb.append(B.head_bits(units, E, d))
    return dict(after=r["after"], final=r["final"], nodes=r["units"], used=r["edges"] / len(data), depth=0.0, gen=gb)


def run_dense_gen(data, gens, seed):
    import x1_8 as M8
    r = M8.run_dense(data, np.zeros((1, C("x17_dim"))), np.zeros(1), np.zeros((C("x17_probe_n"), C("x17_dim"))), seed)
    W, b = B._dense_last; units = [(W[:, i], float(b[i])) for i in range(W.shape[1])]; E = list(range(W.shape[1]))
    rng = np.random.default_rng([seed, 2 + ARMS.index("DENSE")]); gb = []
    for d in gens:                                                      # same connect-only rule over its 32 units (a fair readout on 20 points)
        c = B.Cache(d)
        for u in units: c.add(*u)
        Eg = B.fit_head("HELP", list(units), [0] * len(units), [c], 0, [], len(data), rng, False)
        gb.append(B.head_bits(units, Eg, d))
    return dict(after=r["after"], final=r["final"], nodes=C("x17_dense_h"), used=float(C("x17_dense_h")), depth=0.0, gen=gb)


def world(seed):
    t0 = time.time()
    data, cdepth = X.make_world(seed); gens = gen_problems(seed, data)
    res = {}
    for a in ARMS:
        if a == "FLAT": res[a] = run_flat_gen(data, gens, seed)
        elif a == "DENSE": res[a] = run_dense_gen(data, gens, seed)
        else: res[a] = run_tree_gen(a, data, gens, seed)
    half = len(data) // 2
    out = {"arms": {}}
    for a, r in res.items():
        out["arms"][a] = {"late": float(np.mean(r["after"][half:])), "early": float(np.mean(r["after"][:half])), "final_all": float(np.mean(r["final"])),
                          "gen": float(np.mean(r["gen"])), "nodes": r["nodes"], "used": r["used"], "depth": r["depth"]}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(per, seed0):
    from x0_1 import boot_ci
    seeds = sorted(per)
    g = lambda a, k: np.array([per[s]["arms"][a][k] for s in seeds], float)
    S = {"median": {a: {k: float(np.median(g(a, k))) for k in per[seeds[0]]["arms"][a]} for a in ARMS}, "paired": {}, "worlds": len(seeds)}
    for a, b, k in (("TREE_C", "TREE_P", "used"), ("TREE_CF", "TREE_C", "late"), ("TREE_CF", "TREE_C", "gen"), ("TREE_C", "TREE_P", "late"),
                    ("TREE_C", "TREE_P", "gen"), ("TREE_CF", "FLAT", "gen"), ("TREE_CF", "DENSE", "gen"), ("TREE_CF", "FLAT", "late"), ("TREE_CF", "TREE_C", "nodes")):
        S["paired"][f"{k} {a}-{b}"] = boot_ci(g(a, k) - g(b, k), seed0)
    S["vertical TREE_CF late-early"] = boot_ci(g("TREE_CF", "late") - g("TREE_CF", "early"), seed0)
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run. A win = paired 95% range on the predicted side of 0."""
    m, pr = S["median"], S["paired"]
    f = lambda k: f"{pr[k][0]:+.4f} [{pr[k][1]:+.4f}, {pr[k][2]:+.4f}]"
    up = lambda k: pr[k][1] > 0
    down = lambda k: pr[k][2] < 0
    manip = down("used TREE_C-TREE_P") and up("nodes TREE_CF-TREE_C")
    V = [("M-X1.12", "Fair test: confirmation acts (TREE_C uses fewer nodes per problem than TREE_P) and the frontier bonus acts (TREE_CF makes more nodes than TREE_C)",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"used TREE_C-TREE_P {f('used TREE_C-TREE_P')}; nodes TREE_CF-TREE_C {f('nodes TREE_CF-TREE_C')}")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P416", "Confirmed connections beat unconfirmed ones on later problems: TREE_C minus TREE_P, above 0", up("late TREE_C-TREE_P"), f("late TREE_C-TREE_P"))
    v("P417", "Confirmed connections generalize better to new combinations: TREE_C minus TREE_P, generalization bits, above 0", up("gen TREE_C-TREE_P"), f("gen TREE_C-TREE_P"))
    v("P418", "Rewarding new frontiers helps on later problems: TREE_CF minus TREE_C, above 0", up("late TREE_CF-TREE_C"), f("late TREE_CF-TREE_C"))
    v("P419", "Rewarding new frontiers helps generalization: TREE_CF minus TREE_C, generalization bits, above 0", up("gen TREE_CF-TREE_C"), f("gen TREE_CF-TREE_C"))
    v("P420", "The frontier tree generalizes better than X1.9's best flat learner: TREE_CF minus FLAT, generalization bits, above 0", up("gen TREE_CF-FLAT"), f("gen TREE_CF-FLAT"))
    v("P421", "The frontier tree generalizes better than the plain network: TREE_CF minus DENSE, generalization bits, above 0", up("gen TREE_CF-DENSE"), f("gen TREE_CF-DENSE"))
    v("P422", "Vertical: TREE_CF later minus earlier bits, above 0", S["vertical TREE_CF late-early"][1] > 0,
      f"{S['vertical TREE_CF late-early'][0]:+.4f} [{S['vertical TREE_CF late-early'][1]:+.4f}, {S['vertical TREE_CF late-early'][2]:+.4f}]")
    return V


ARM_TEXT = {"TREE_CF": ("Tree, confirmed links + frontier reward", "Keeps a link only if it helps on two separate halves of the held-back examples; new nodes that hold up pay 5x."),
            "TREE_C": ("Tree, confirmed links", "Keeps a link only if it helps on both halves; no frontier bonus."),
            "TREE_P": ("Tree, precision reward (X1.11)", "No confirmation."),
            "FLAT": ("Best earlier learner (X1.9)", "One layer of nodes, the author's link rule."),
            "DENSE": ("Plain neural network", "32 shared units; on new problems it links its units by measured help.")}


def log_wandb(out):
    from track import Tracker
    S = out["summary"]
    tk = Tracker("X1.12", "Frontiers, correct links and generalization", "Does keeping only confirmed connections and rewarding new frontiers make the tree generalize to new combinations of what it knows?",
                 X.WORLD_TEXT + " After the stream, 10 new problems mix known concepts in new ways, with only 20 examples and no new learning.",
                 ARM_TEXT, B.READ_TEXT + " 'Generalization' = bits held on those new problems.", dict(worlds=S["worlds"]), tags=["tree", "frontier", "generalization"])
    keys = [("late", "Bits per point, later problems"), ("final_all", "Bits per point at the end, all problems"), ("gen", "Generalization: bits on new combinations (higher is better)"),
            ("used", "Nodes used per problem"), ("nodes", "Nodes in total")]
    tk.table("Summary by arm (median over worlds)", ["Arm"] + [k[1] for k in keys], [[tk.name(a)] + [round(S["median"][a][k], 4) for k, _ in keys] for a in ARMS])
    tk.table("Paired differences with 95% ranges (same worlds, A minus B)", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    import multiprocessing as mp, budget
    t0 = time.time()
    if mode == "log":
        log_wandb(json.load(open(os.path.join(OUT, "x1_12_run.json")))["result"]); return
    seed0 = C("x112_seed0"); nw = C("x110_max_worlds")
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
    json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, f"x1_12_{mode}.json"), "w"), indent=1, default=float)
    if mode == "run": log_wandb(out)
    for v in V: print(v[0], v[2], "|", v[3])


if __name__ == "__main__":
    main(sys.argv[1])
