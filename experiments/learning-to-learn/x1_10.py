"""X1.10: a TREE that every problem enters at its base and grows from (author, 2026-11-02).

Author's rules: (1) nodes ONLY update if actively used; (2) every problem starts at the base of the tree and expands from learned behaviour;
(3) deeper growth (more nodes used per problem) earns exponentially higher reward; (4) nodes are weighted by their perceived application to
other nodes, defined entirely by the network itself (internal). Carried over from X1.7-X1.9: every addition is paid by its EXACT measured help
(bits on decision points), with the rare-node (specificity) and new-node bonuses as multipliers.

World: as X1.7, but concepts can be built ON other concepts: a new concept is, with probability x110_p_child, c = tanh(w.x + v*c_parent + b)
(a deeper rung), else a root concept. Problems mix 3 concepts as before.
Learner: nodes h_i = tanh(u_i.x + v_i*h_parent(i) + b_i); node 0 = the base (no parent). A problem's USED set starts as {base} and grows one node
at a time: candidates = existing children of used nodes + a new node trained (on the residual) under the used node whose output best matches the
residual, and under the base. Value = help x specificity x newcomer x DEPTH (2^(nodes already used)) x APPLICATION (1 + log2(1 + a_i/mean a)),
accepted if above log2(#candidates) bits. After the problem, ONLY the used nodes (and the readout) train (Adam, stop when the decision error
stalls). Application a_i = sd of the node's output x (sum of |v| of its children + sum of |readout weight| of the problems using it): how much
the rest of the network draws on it, read off the network's own weights and activity.
Arms: TREE (all rules)  TREE_D0 (no depth reward)  TREE_A0 (no application weighting)  TREE_FZ (nodes never update)  TREE_TRAP (depth reward paid
on existence, whatever the help)  FRESH (a new tree for every problem)  FLAT (X1.9's best: one layer, all linked nodes train)  DENSE (plain network).
Every arm re-fits readouts on stored train points at the end ('final').
Usage: python x1_10.py run | log"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_7 as B
import x1_8 as M8
import x1_9 as M9

OUT = B.OUT
ARMS = ("TREE", "TREE_D0", "TREE_A0", "TREE_FZ", "TREE_TRAP", "FRESH", "FLAT", "DENSE")


# ---------------------------------------------------------------- world: concepts on concepts ------------------------------------------
def make_world(seed):
    rng = np.random.default_rng([seed, 1])
    m, T, k = C("x17_dim"), C("x17_problems"), C("x17_k")
    W, V, Bb, Par = [], [], [], []

    def newc(root):
        p = -1 if (root or rng.random() >= C("x110_p_child")) else int(rng.integers(len(W)))
        W.append(rng.standard_normal(m) * C("x17_gain") / math.sqrt(m)); Bb.append(rng.standard_normal() * C("x17_bias_sd"))
        V.append(0.0 if p < 0 else float(rng.choice([-1, 1]) * rng.uniform(C("x110_v_lo"), C("x110_v_hi")))); Par.append(p)
    for _ in range(C("x17_pool0")): newc(True)
    probs = []
    for t in range(T):
        if rng.random() < C("x17_p_new"):
            newc(False); S = [len(W) - 1] + list(rng.choice(len(W) - 1, k - 1, replace=False))
        else: S = list(rng.choice(len(W), k, replace=False))
        probs.append((S, rng.choice([-1, 1], k) * rng.uniform(C("x17_a_lo"), C("x17_a_hi"), k)))
    Wm, Vm, Bm = np.array(W), np.array(V), np.array(Bb)

    def concepts(x):
        Cc = np.zeros((len(x), len(W)))
        for j in range(len(W)): Cc[:, j] = np.tanh(x @ Wm[j] + Bm[j] + (Vm[j] * Cc[:, Par[j]] if Par[j] >= 0 else 0))
        return Cc
    data = []
    for S, a in probs:
        n = C("x17_n_tr") + C("x17_n_val") + C("x17_n_te") + C("x17_norm_n")
        x = rng.standard_normal((n, m)); Cc = concepts(x)
        y = Cc[:, S] @ a; y = y + rng.standard_normal(n) * C("x17_noise") * y.std()
        mu, sd = y[-C("x17_norm_n"):].mean(), y[-C("x17_norm_n"):].std(); y = (y - mu) / sd
        i1, i2, i3 = C("x17_n_tr"), C("x17_n_tr") + C("x17_n_val"), C("x17_n_tr") + C("x17_n_val") + C("x17_n_te")
        data.append(dict(tr=(x[:i1], y[:i1]), va=(x[i1:i2], y[i1:i2]), te=(x[i2:i3], y[i2:i3]), S=S, Cte=Cc[i2:i3][:, S],
                          Ctrue=(Cc[:i1][:, S], Cc[i1:i2][:, S], Cc[i2:i3][:, S])))      # true concepts (no random draws): per-problem ceiling
    depth = [0] * len(W)
    for j in range(len(W)): depth[j] = 0 if Par[j] < 0 else depth[Par[j]] + 1
    return data, depth


# ---------------------------------------------------------------- the tree ------------------------------------------------------------------
class Tree:
    def __init__(self): self.u, self.v, self.b, self.par, self.kids, self.deg, self.app = [], [], [], [], [], [], []

    def add(self, u, v, b, p):
        self.u.append(u); self.v.append(v); self.b.append(b); self.par.append(p); self.kids.append([]); self.deg.append(0); self.app.append(0.0)
        if p >= 0: self.kids[p].append(len(self.u) - 1)
        return len(self.u) - 1

    def act(self, x, upto=None):
        n = len(self.u) if upto is None else upto
        H = np.zeros((len(x), n))
        for i in range(n): H[:, i] = np.tanh(x @ self.u[i] + self.b[i] + (self.v[i] * H[:, self.par[i]] if self.par[i] >= 0 else 0))
        return H

    def subtree(self, i):
        out, st = [], [i]
        while st:
            j = st.pop(); out.append(j); st += self.kids[j]
        return sorted(out)


class TCache:
    def __init__(self, d, tree):
        self.xtr, self.ytr = d["tr"]; self.xva, self.yva = d["va"]; self.tree = tree; self.recompute()

    def recompute(self): self.tr = self.tree.act(self.xtr); self.va = self.tree.act(self.xva)

    def update(self, idx):
        T = self.tree
        if self.tr.shape[1] < len(T.u):
            self.tr = np.hstack([self.tr, np.zeros((len(self.ytr), len(T.u) - self.tr.shape[1]))]); self.va = np.hstack([self.va, np.zeros((len(self.yva), len(T.u) - self.va.shape[1]))])
        for i in sorted(idx):
            for A, x in ((self.tr, self.xtr), (self.va, self.xva)):
                A[:, i] = np.tanh(x @ T.u[i] + T.b[i] + (T.v[i] * A[:, T.par[i]] if T.par[i] >= 0 else 0))


def new_node(rng, tree, c, p, r):
    """Train a new node under parent p (p = -1 only for the very first base node) on residual r."""
    xa = c.xtr if p < 0 else np.hstack([c.xtr, c.tr[:, [p]]])
    w, b = B.train_unit(rng, xa, r)
    u = w[:c.xtr.shape[1]]; v = 0.0 if p < 0 else float(w[-1])
    ztr = np.tanh(c.xtr @ u + b + (v * c.tr[:, p] if p >= 0 else 0)); zva = np.tanh(c.xva @ u + b + (v * c.va[:, p] if p >= 0 else 0))
    return (u, v, b, p), ztr, zva


def multipliers(arm, tree, P, cand, isnew, n_used):
    ls, ln = B.BON["FULL"]
    dg = np.array([0 if nw else tree.deg[i] for i, nw in zip(cand, isnew)], float)
    spec = np.where(isnew, math.log2(P + 1), np.log2((P + 1) / (dg + 1.0)))
    mult = (1 + ls * spec) * (1 + np.where(isnew, ln, 0))
    if arm != "TREE_A0":
        a = np.array(tree.app, float); ma = a.mean() if len(a) and a.mean() > 0 else 1.0
        ap = np.array([0.0 if nw else tree.app[i] for i, nw in zip(cand, isnew)])
        mult = mult * (1 + np.log2(1 + ap / ma))
    depth = 2.0 ** (n_used - 1) if arm not in ("TREE_D0",) else 1.0
    return mult, depth


def grow_head(arm, tree, c, P, rng):
    """Enter at the base and expand one node at a time by (adjusted) measured help."""
    E = [0]; nva = len(c.yva)
    for _ in range(C("x110_max_used")):                                 # no reward ceiling (author): grows while any addition pays
        F, Fv, Gi, beta = B.solve(c, E); r = c.ytr - F @ beta; pv = Fv @ beta; b0 = float(B.vbits(c.yva - pv, nva))
        kids = sorted({k for e in E for k in tree.kids[e]} - set(E))
        Z, Zv = c.tr[:, kids], c.va[:, kids]; cand = list(kids); isnew = [False] * len(kids); props = []
        best_p = E[int(np.argmax([abs(np.corrcoef(c.tr[:, e], r)[0, 1]) if c.tr[:, e].std() > 0 else 0 for e in E]))]
        for p in sorted({best_p, 0}):
            nd, ztr, zva = new_node(rng, tree, c, p, r); props.append(nd)
            Z = np.hstack([Z, ztr[:, None]]); Zv = np.hstack([Zv, zva[:, None]]); cand.append(-len(props)); isnew.append(True)
        K = Z.shape[1]; cost = math.log2(K + 1)
        Wc = Gi @ (F.T @ Z); sch = (Z ** 2).sum(0) + C("x17_ridge") - (Z * (F @ Wc)).sum(0); cf = (Z.T @ r) / sch
        help_ = B.vbits(c.yva[:, None] - (pv[:, None] + cf * (Zv - Fv @ Wc)), nva) - b0
        isnew = np.array(isnew); mult, depth = multipliers(arm, tree, P, cand, isnew, len(E))
        if arm == "TREE_TRAP": v = np.where(help_ > 0, help_ * mult, help_) + cost * (depth - 1)       # depth reward paid on existence
        else: v = np.where(help_ > 0, help_ * mult * depth, help_)
        k = int(np.argmax(v))
        if v[k] <= cost: break
        if isnew[k]:
            u, vv, b, p = props[-cand[k] - 1]; j = tree.add(u, vv, b, p); c.update([j])
        else: j = cand[k]
        E.append(j)
    for e in E: tree.deg[e] += 1
    return sorted(E)


def train_used(tree, E, d):
    """ONLY the used nodes (and the readout) train: Adam through the used subtree, every other node fixed; stop when the decision error stalls."""
    (xtr, ytr), (xva, yva) = d["tr"], d["va"]; n = len(ytr); E = sorted(E); pos = {e: k for k, e in enumerate(E)}
    P = [np.array([tree.u[e] for e in E]), np.array([tree.v[e] for e in E]), np.array([tree.b[e] for e in E])]
    par = [pos.get(tree.par[e], -1) for e in E]

    def fwd(x, P):
        H = np.zeros((len(x), len(E)))
        for k in range(len(E)): H[:, k] = np.tanh(x @ P[0][k] + P[2][k] + (P[1][k] * H[:, par[k]] if par[k] >= 0 else 0))
        return H
    H = fwd(xtr, P); F = np.hstack([np.ones((n, 1)), H]); beta = np.linalg.solve(F.T @ F + C("x17_ridge") * np.eye(len(E) + 1), F.T @ ytr)
    Q = P + [beta]; mo = [np.zeros_like(q) for q in Q]; ve = [np.zeros_like(q) for q in Q]
    lr, b1, b2, eps = C("lr_default") * C("x17_dense_lr_mult"), C("adam_b1"), C("adam_b2"), C("adam_eps")
    val = lambda Q: float(np.mean((Q[3][0] + fwd(xva, Q) @ Q[3][1:] - yva) ** 2))
    best, bestQ, since = val(Q), [q.copy() for q in Q], 0
    for s in range(1, C("x18_ft_steps") + 1):
        H = fwd(xtr, Q); e = (Q[3][0] + H @ Q[3][1:] - ytr) * (2.0 / n)
        gbeta = np.concatenate([[e.sum()], H.T @ e])
        dH = np.outer(e, Q[3][1:]); gu = np.zeros_like(Q[0]); gv = np.zeros_like(Q[1]); gb = np.zeros_like(Q[2])
        for k in range(len(E) - 1, -1, -1):                              # backprop through the used subtree (children before parents)
            dz = dH[:, k] * (1 - H[:, k] ** 2)
            gu[k] = xtr.T @ dz; gb[k] = dz.sum()
            if par[k] >= 0: gv[k] = dz @ H[:, par[k]]; dH[:, par[k]] += dz * Q[1][k]
        for q, g in enumerate((gu, gv, gb, gbeta)):
            mo[q] = b1 * mo[q] + (1 - b1) * g; ve[q] = b2 * ve[q] + (1 - b2) * g * g
            Q[q] = Q[q] - lr * (mo[q] / (1 - b1 ** s)) / (np.sqrt(ve[q] / (1 - b2 ** s)) + eps)
        v = val(Q)
        if v < best: best, bestQ, since = v, [q.copy() for q in Q], 0
        else:
            since += 1
            if since >= C("x18_ft_patience"): break
    for k, e in enumerate(E): tree.u[e], tree.v[e], tree.b[e] = bestQ[0][k].copy(), float(bestQ[1][k]), float(bestQ[2][k])


def update_application(tree, heads, caches):
    """Internal weighting: how much the rest of the network draws on each node (children's |v| + readouts' |weight|) x its output spread."""
    draw = np.zeros(len(tree.u)); sd = np.zeros(len(tree.u)); cnt = np.zeros(len(tree.u))
    for i in range(len(tree.u)):
        if tree.par[i] >= 0: draw[tree.par[i]] += abs(tree.v[i])
    for E, c in zip(heads, caches):
        F, Fv, Gi, beta = B.solve(c, E)
        for k, e in enumerate(E): draw[e] += abs(beta[k + 1]); sd[e] += c.tr[:, e].std(); cnt[e] += 1
    tree.app = list(draw * np.where(cnt > 0, sd / np.maximum(cnt, 1), 0.0))


def tree_bits(tree, E, d, split="te"):
    xtr, ytr = d["tr"]; x, y = d[split]
    H, Ht = tree.act(xtr)[:, E], tree.act(x)[:, E]
    F = np.hstack([np.ones((len(ytr), 1)), H]); Ft = np.hstack([np.ones((len(y), 1)), Ht])
    beta = np.linalg.solve(F.T @ F + C("x17_ridge") * np.eye(F.shape[1]), F.T @ ytr)
    return B.bits(float(np.mean((y - Ft @ beta) ** 2)), len(y)) / len(y)


def run_tree(arm, data, seed):
    rng = np.random.default_rng([seed, 2 + ARMS.index(arm)])
    tree, heads, caches, after, used, depth_used = Tree(), [], [], [], [], []
    trees = []
    for t, d in enumerate(data):
        if arm == "FRESH" or t == 0:
            if arm == "FRESH": tree = Tree()
            c0 = TCache(d, tree); nd, _, _ = new_node(rng, tree, c0, -1, d["tr"][1]); tree.add(*nd)
        c = TCache(d, tree)
        if arm == "FRESH": caches_t = [c]
        else: caches.append(c)
        E = grow_head(arm, tree, c, t, rng)
        if arm != "TREE_FZ":
            train_used(tree, E, d)
            for cx in (caches if arm != "FRESH" else [c]): cx.update(E + [j for e in E for j in tree.subtree(e) if j not in E])
        heads.append(E); used.append(len(E))
        dep = {0: 0}
        for i in range(1, len(tree.u)): dep[i] = dep[tree.par[i]] + 1
        depth_used.append(max(dep[e] for e in E))
        if arm != "FRESH": update_application(tree, heads, caches)
        trees.append(tree)
        after.append(tree_bits(tree, E, d))
    final = [tree_bits(trees[s] if arm == "FRESH" else tree, E, data[s]) for s, E in enumerate(heads)]
    nodes = sum(len(tr.u) for tr in trees) if arm == "FRESH" else len(tree.u)
    return dict(after=after, final=final, nodes=nodes, used=float(np.mean(used)), depth=float(np.mean(depth_used)))


def world(seed):
    t0 = time.time()
    data, cdepth = make_world(seed)
    res = {}
    for a in ARMS:
        if a == "FLAT":
            r = M9.run_grow("FULLALL", data, np.zeros((1, C("x17_dim"))), np.zeros(1), np.zeros((C("x17_probe_n"), C("x17_dim"))), seed)
            r = dict(after=r["after"], final=r["final"], nodes=r["units"], used=r["edges"] / len(data), depth=0.0)
        elif a == "DENSE":
            r = M8.run_dense(data, np.zeros((1, C("x17_dim"))), np.zeros(1), np.zeros((C("x17_probe_n"), C("x17_dim"))), seed)
            r = dict(after=r["after"], final=r["final"], nodes=C("x17_dense_h"), used=float(C("x17_dense_h")), depth=0.0)
        else: r = run_tree(a, data, seed)
        res[a] = r
    half = len(data) // 2
    out = {"world_concept_depth_max": int(max(cdepth)), "arms": {}}
    for a, r in res.items():
        out["arms"][a] = {"late": float(np.mean(r["after"][half:])), "early": float(np.mean(r["after"][:half])), "final_all": float(np.mean(r["final"])),
                          "backward": float(np.mean(np.array(r["final"][:half]) - np.array(r["after"][:half]))), "nodes": r["nodes"], "used": r["used"], "depth": r["depth"]}
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------- scoring -----------------------------------------------------------
def summarize(per, seed0):
    from x0_1 import boot_ci
    seeds = sorted(per)
    g = lambda a, k: np.array([per[s]["arms"][a][k] for s in seeds], float)
    S = {"median": {a: {k: float(np.median(g(a, k))) for k in per[seeds[0]]["arms"][a]} for a in ARMS}, "paired": {}, "worlds": len(seeds)}
    for a, b, k in (("TREE", "FRESH", "late"), ("TREE_TRAP", "TREE", "used"), ("TREE", "FLAT", "late"), ("TREE", "FLAT", "final_all"),
                    ("TREE", "TREE_D0", "late"), ("TREE", "TREE_A0", "late"), ("TREE", "TREE_FZ", "final_all"), ("TREE_TRAP", "TREE", "late"),
                    ("TREE", "DENSE", "final_all"), ("TREE", "TREE_D0", "used"), ("TREE", "TREE_FZ", "late")):
        S["paired"][f"{k} {a}-{b}"] = boot_ci(g(a, k) - g(b, k), seed0)
    S["vertical TREE late-early"] = boot_ci(g("TREE", "late") - g("TREE", "early"), seed0)
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run. A win = paired 95% range on the predicted side of 0."""
    m, pr = S["median"], S["paired"]
    f = lambda k: f"{pr[k][0]:+.4f} [{pr[k][1]:+.4f}, {pr[k][2]:+.4f}]"
    up = lambda k: pr[k][1] > 0
    down = lambda k: pr[k][2] < 0
    manip = up("late TREE-FRESH") and up("used TREE_TRAP-TREE") and up("used TREE-TREE_D0") and m["TREE"]["depth"] >= 1
    V = [("M-X1.10", "Fair test: reuse pays (TREE beats a new tree per problem), the depth reward acts (TREE uses more nodes per problem than TREE_D0; TRAP more than TREE), and the tree really grows deep (median deepest used node >= 1 below the base)",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"TREE-FRESH {f('late TREE-FRESH')}; used TREE-D0 {f('used TREE-TREE_D0')}; used TRAP-TREE {f('used TREE_TRAP-TREE')}; depth {m['TREE']['depth']:.2f}")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P403", "The tree beats X1.9's best flat learner on later problems: TREE minus FLAT, above 0", up("late TREE-FLAT"), f("late TREE-FLAT"))
    v("P404", "The exponential depth reward helps: TREE minus TREE_D0, later-problem bits, above 0", up("late TREE-TREE_D0"), f("late TREE-TREE_D0"))
    v("P405", "Internal application weighting helps: TREE minus TREE_A0, later-problem bits, above 0", up("late TREE-TREE_A0"), f("late TREE-TREE_A0"))
    v("P406", "Updating only used nodes beats never updating: TREE minus TREE_FZ, all problems at the end, above 0", up("final_all TREE-TREE_FZ"), f("final_all TREE-TREE_FZ"))
    v("P407", "The guard: the depth reward paid on existence holds less: TREE_TRAP minus TREE, later-problem bits, below 0", down("late TREE_TRAP-TREE"), f("late TREE_TRAP-TREE"))
    v("P408", "The tree beats the plain network on all problems at the end (equal re-fits): TREE minus DENSE, above 0", up("final_all TREE-DENSE"), f("final_all TREE-DENSE"))
    v("P409", "Vertical: TREE later minus earlier bits, above 0", S["vertical TREE late-early"][1] > 0,
      f"{S['vertical TREE late-early'][0]:+.4f} [{S['vertical TREE late-early'][1]:+.4f}, {S['vertical TREE late-early'][2]:+.4f}]")
    return V


ARM_TEXT = {"TREE": ("The author's tree", "Every problem enters at the base node and grows outward; only nodes it uses learn; using more nodes pays exponentially more; nodes the network itself relies on get priority."),
            "TREE_D0": ("Tree, no depth reward", "Same, but using more nodes earns no extra reward."),
            "TREE_A0": ("Tree, no self-weighting", "Same, but nodes get no priority from how much the network relies on them."),
            "TREE_FZ": ("Tree, nodes never learn", "Same growth, but nodes never change after they are made."),
            "TREE_TRAP": ("Tree, depth paid regardless (trap)", "The depth reward is paid for using more nodes even when they help nothing."),
            "FRESH": ("New tree every problem", "Nothing is reused between problems."),
            "FLAT": ("Best earlier learner (X1.9)", "One layer of nodes, the author's link rule, all linked nodes learn."),
            "DENSE": ("Plain neural network", "32 shared units, gradient descent on everything.")}
WORLD_TEXT = B.WORLD_TEXT + " Some hidden concepts are built on top of other concepts, so deeper structure exists to be found."


def log_wandb(out):
    from track import Tracker
    S = out["summary"]
    tk = Tracker("X1.10", "A tree every problem grows from", "If every problem enters a tree at its base and grows outward, only used nodes learn, deeper use pays exponentially and nodes are weighted by how much the network itself relies on them, does it learn better?",
                 WORLD_TEXT, ARM_TEXT, B.READ_TEXT, dict(worlds=S["worlds"], problems=C("x17_problems")), tags=["tree", "depth", "use-dependent", "vertical"])
    keys = [("late", "Bits per point, later problems (higher is better)"), ("early", "Bits per point, earlier problems"), ("final_all", "Bits per point at the end, all problems"),
            ("backward", "Old problems changed later (bits)"), ("used", "Nodes used per problem"), ("depth", "Deepest node used (levels below the base)"), ("nodes", "Nodes in total")]
    tk.table("Summary by arm (median over worlds)", ["Arm"] + [k[1] for k in keys], [[tk.name(a)] + [round(S["median"][a][k], 4) for k, _ in keys] for a in ARMS])
    tk.table("Paired differences with 95% ranges (same worlds, A minus B)", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    import multiprocessing as mp, budget
    t0 = time.time()
    if mode == "log":
        log_wandb(json.load(open(os.path.join(OUT, "x1_10_run.json")))["result"]); return
    seed0 = C("x110_seed0"); nw = C("x110_max_worlds")
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
    json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, f"x1_10_{mode}.json"), "w"), indent=1, default=float)
    if mode == "run": log_wandb(out)
    for v in V: print(v[0], v[2], "|", v[3])


if __name__ == "__main__":
    main(sys.argv[1])
