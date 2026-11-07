"""X1.11: the author's tree with a PRECISION reward instead of a depth reward (author, 2026-11-02: 'don't reward depth specifically so much as
increasing precision, with exponentially scaling results as it gets closer to 100%. an initial guess gives a low reward, but 99.9% -> 99.95% is a
HUUUUUUGE reward'; 'both deeper thinking AND correct relative weighting').

Precision reward R = 1 / MSE on the decision points (target variance 1, so R = 1 knows nothing; R = 1000 is 99.9% of the variance explained).
R = 2^(2 x bits held): exponential in the bits, so halving the error from 0.1% to 0.05% pays +1000 per point while a first guess from 100% to 80% pays
+0.25. The value of an addition = n_val x (R_after - R_before) x specificity x newcomer x internal application weight, accepted if above the
same selection cost log2(#candidates) as before (no ceiling on the reward). Everything else is X1.10's tree (vi/x1_10.py): enter at the base, grow,
only used nodes learn, internal self-weighting.
Arms: TREE_P (precision reward + self-weighting)  TREE_P_A0 (precision reward, no self-weighting)  TREE_D0 (X1.10's tree with no reward shaping:
plain measured bits)  TREE (X1.10's depth reward 2^(nodes used))  FLAT (X1.9's best one-layer learner)  DENSE (plain network).
Usage: python x1_11.py run | log"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_7 as B
import x1_10 as X

OUT = B.OUT
ARMS = ("TREE_P", "TREE_P_A0", "TREE_D0", "TREE", "FLAT", "DENSE")
_orig_grow = X.grow_head


def grow_head(arm, tree, c, P, rng):
    if arm not in ("TREE_P", "TREE_P_A0"): return _orig_grow(arm, tree, c, P, rng)
    E = [0]; nva = len(c.yva); floor = C("x17_mse_floor")
    for _ in range(C("x110_max_used")):
        F, Fv, Gi, beta = B.solve(c, E); r = c.ytr - F @ beta; pv = Fv @ beta
        R0 = 1.0 / max(float(np.mean((c.yva - pv) ** 2)), floor)
        kids = sorted({k for e in E for k in tree.kids[e]} - set(E))
        Z, Zv = c.tr[:, kids], c.va[:, kids]; cand = list(kids); isnew = [False] * len(kids); props = []
        best_p = E[int(np.argmax([abs(np.corrcoef(c.tr[:, e], r)[0, 1]) if c.tr[:, e].std() > 0 else 0 for e in E]))]
        for p in sorted({best_p, 0}):
            nd, ztr, zva = X.new_node(rng, tree, c, p, r); props.append(nd)
            Z = np.hstack([Z, ztr[:, None]]); Zv = np.hstack([Zv, zva[:, None]]); cand.append(-len(props)); isnew.append(True)
        K = Z.shape[1]; cost = math.log2(K + 1)
        Wc = Gi @ (F.T @ Z); sch = (Z ** 2).sum(0) + C("x17_ridge") - (Z * (F @ Wc)).sum(0); cf = (Z.T @ r) / sch
        mse = np.mean((c.yva[:, None] - (pv[:, None] + cf * (Zv - Fv @ Wc))) ** 2, 0)
        gain = nva * (1.0 / np.maximum(mse, floor) - R0)                         # precision reward: exponential in the bits held
        isnew = np.array(isnew)
        mult, _ = X.multipliers("TREE_A0" if arm == "TREE_P_A0" else "TREE", tree, P, cand, isnew, len(E))
        v = np.where(gain > 0, gain * mult, gain)
        k = int(np.argmax(v))
        if v[k] <= cost: break
        if isnew[k]:
            u, vv, b, p = props[-cand[k] - 1]; j = tree.add(u, vv, b, p); c.update([j])
        else: j = cand[k]
        E.append(j)
    for e in E: tree.deg[e] += 1
    return sorted(E)


X.grow_head = grow_head
X.ARMS = ARMS                                                                    # random-stream ids for the arms of this experiment


def world(seed):
    t0 = time.time()
    data, cdepth = X.make_world(seed)
    z = (np.zeros((1, C("x17_dim"))), np.zeros(1), np.zeros((C("x17_probe_n"), C("x17_dim"))))
    res = {}
    for a in ARMS:
        if a == "FLAT":
            import x1_9 as M9
            r = M9.run_grow("FULLALL", data, *z, seed); r = dict(after=r["after"], final=r["final"], nodes=r["units"], used=r["edges"] / len(data), depth=0.0)
        elif a == "DENSE":
            import x1_8 as M8
            r = M8.run_dense(data, *z, seed); r = dict(after=r["after"], final=r["final"], nodes=C("x17_dense_h"), used=float(C("x17_dense_h")), depth=0.0)
        else: r = X.run_tree(a, data, seed)
        res[a] = r
    half = len(data) // 2
    out = {"arms": {}}
    for a, r in res.items():
        out["arms"][a] = {"late": float(np.mean(r["after"][half:])), "early": float(np.mean(r["after"][:half])), "final_all": float(np.mean(r["final"])),
                          "backward": float(np.mean(np.array(r["final"][:half]) - np.array(r["after"][:half]))), "nodes": r["nodes"], "used": r["used"], "depth": r["depth"],
                          "late_top": float(np.percentile(r["after"][half:], 90))}   # CONST: x111_top_pct
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(per, seed0):
    from x0_1 import boot_ci
    seeds = sorted(per)
    g = lambda a, k: np.array([per[s]["arms"][a][k] for s in seeds], float)
    S = {"median": {a: {k: float(np.median(g(a, k))) for k in per[seeds[0]]["arms"][a]} for a in ARMS}, "paired": {}, "worlds": len(seeds)}
    for a, b, k in (("TREE_P", "TREE_D0", "used"), ("TREE_P", "TREE_D0", "late"), ("TREE_P", "TREE", "late"), ("TREE_P", "TREE_P_A0", "late"),
                    ("TREE_P", "FLAT", "late"), ("TREE_P", "DENSE", "final_all"), ("TREE_P", "TREE_D0", "late_top"), ("TREE_P", "FLAT", "final_all")):
        S["paired"][f"{k} {a}-{b}"] = boot_ci(g(a, k) - g(b, k), seed0)
    S["vertical TREE_P late-early"] = boot_ci(g("TREE_P", "late") - g("TREE_P", "early"), seed0)
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run. A win = paired 95% range on the predicted side of 0."""
    m, pr = S["median"], S["paired"]
    f = lambda k: f"{pr[k][0]:+.4f} [{pr[k][1]:+.4f}, {pr[k][2]:+.4f}]"
    up = lambda k: pr[k][1] > 0
    down = lambda k: pr[k][2] < 0
    manip = up("used TREE_P-TREE_D0") or down("used TREE_P-TREE_D0")
    V = [("M-X1.11", "Fair test: the precision reward changes what the tree does (nodes used per problem differ from the unshaped tree, range excludes 0)",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"used TREE_P-TREE_D0 {f('used TREE_P-TREE_D0')}")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P410", "The precision reward helps: TREE_P minus TREE_D0 later-problem bits, above 0", up("late TREE_P-TREE_D0"), f("late TREE_P-TREE_D0"))
    v("P411", "Rewarding precision beats rewarding depth: TREE_P minus TREE (X1.10's depth reward), later bits, above 0", up("late TREE_P-TREE"), f("late TREE_P-TREE"))
    v("P412", "Correct relative weighting helps under the precision reward: TREE_P minus TREE_P_A0, later bits, above 0", up("late TREE_P-TREE_P_A0"), f("late TREE_P-TREE_P_A0"))
    v("P413", "The precision tree beats X1.9's best flat learner on later problems: TREE_P minus FLAT, above 0", up("late TREE_P-FLAT"), f("late TREE_P-FLAT"))
    v("P414", "Vertical: TREE_P later minus earlier bits, above 0", S["vertical TREE_P late-early"][1] > 0,
      f"{S['vertical TREE_P late-early'][0]:+.4f} [{S['vertical TREE_P late-early'][1]:+.4f}, {S['vertical TREE_P late-early'][2]:+.4f}]")
    v("P415", "The precision tree beats the plain network on all problems at the end: TREE_P minus DENSE, above 0", up("final_all TREE_P-DENSE"), f("final_all TREE_P-DENSE"))
    return V


ARM_TEXT = {"TREE_P": ("Tree, precision reward", "Rewards getting closer to perfect: each halving of the remaining error pays twice as much as the last."),
            "TREE_P_A0": ("Tree, precision reward, no self-weighting", "Same, without the network's own weighting of its nodes."),
            "TREE_D0": ("Tree, no reward shaping", "X1.10's tree paid only by plain measured help."),
            "TREE": ("Tree, depth reward (X1.10)", "Pays 2^(nodes used): rewards using more nodes."),
            "FLAT": ("Best earlier learner (X1.9)", "One layer of nodes, the author's link rule, all linked nodes learn."),
            "DENSE": ("Plain neural network", "32 shared units, gradient descent on everything.")}


def log_wandb(out):
    from track import Tracker
    S = out["summary"]
    tk = Tracker("X1.11", "A tree rewarded for precision", "Does rewarding precision (exponentially more as error approaches zero) instead of depth make the author's tree learn better?",
                 X.WORLD_TEXT, ARM_TEXT, B.READ_TEXT, dict(worlds=S["worlds"]), tags=["tree", "precision", "vertical"])
    keys = [("late", "Bits per point, later problems (higher is better)"), ("early", "Bits per point, earlier problems"), ("final_all", "Bits per point at the end, all problems"),
            ("late_top", "Best 10% of later problems (bits)"), ("used", "Nodes used per problem"), ("depth", "Deepest node used"), ("nodes", "Nodes in total")]
    tk.table("Summary by arm (median over worlds)", ["Arm"] + [k[1] for k in keys], [[tk.name(a)] + [round(S["median"][a][k], 4) for k, _ in keys] for a in ARMS])
    tk.table("Paired differences with 95% ranges (same worlds, A minus B)", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    import multiprocessing as mp, budget
    t0 = time.time()
    if mode == "log":
        log_wandb(json.load(open(os.path.join(OUT, "x1_11_run.json")))["result"]); return
    seed0 = C("x111_seed0"); nw = C("x110_max_worlds")
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
    json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, f"x1_11_{mode}.json"), "w"), indent=1, default=float)
    if mode == "run": log_wandb(out)
    for v in V: print(v[0], v[2], "|", v[3])


if __name__ == "__main__":
    main(sys.argv[1])
