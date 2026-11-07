"""X1.9: train the FEWEST nodes possible; every other node stays exactly as it was (author, 2026-11-02: 'make training affect the minimal amount
of nodes possible, with surrounding nodes effecting no change'). Supersedes X1.8 (registered, withdrawn before any result).

Same world, link rule and arms as X1.7 (vi/x1_7.py). After a problem's links are chosen, nodes are trained ONE AT A TIME:
  1. rank the problem's linked nodes by the size of the train-error gradient on their own weights (readout fixed);
  2. train only the top node (Adam on that node and the readout; every other node fixed), best decision-error step kept;
  3. keep the change only if its NET measured help pays the selection cost log2(k+1) bits: bits gained on this problem's decision points
     MINUS bits lost on every old problem that links to the node (their readouts re-fitted on their stored train points, scored on their
     stored decision points): 'exactly how it helps or hurts';
  4. repeat with the next node while changes keep paying; stop at the first that does not ('stop early if progress stops').
Scoring as X1.8: every arm, DENSE included, re-fits readouts on stored train points at the end ('final').
Arms: FRESH, HELP, SPEC, NEW, FULL, TRAP, ANY (all with minimal training), DENSE, FULLFZ (author's rule, no node ever trains), FULLALL (author's rule,
every linked node trains together: X1.8's FULL), FULLNC (minimal training that ignores harm to old problems).
Usage: python x1_9.py run | log        (writes ../results/x1_9_<mode>.json)"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_7 as B
import x1_8 as M

OUT = B.OUT
ARMS = ("FRESH", "HELP", "SPEC", "NEW", "FULL", "TRAP", "ANY", "DENSE", "FULLFZ", "FULLALL", "FULLNC")
CORE = {"FULLFZ": "FULL", "FULLALL": "FULL", "FULLNC": "FULL"}
B.BON.update({a: B.BON["FULL"] for a in CORE})


def train_one(units, E, i, d):
    """Adam on node i (u, b) and the readout only; every other node fixed. Returns the trained (u, b)."""
    (xtr, ytr), (xva, yva) = d["tr"], d["va"]
    n, m = xtr.shape; j = E.index(i)
    X1, V1 = np.hstack([xtr, np.ones((n, 1))]), np.hstack([xva, np.ones((len(yva), 1))])
    U = np.array([np.concatenate([units[e][0], [units[e][1]]]) for e in E])
    H = np.tanh(X1 @ U.T); Hv = np.tanh(V1 @ U.T)
    F = np.hstack([np.ones((n, 1)), H])
    beta = np.linalg.solve(F.T @ F + C("x17_ridge") * np.eye(len(E) + 1), F.T @ ytr)
    w = U[j].copy(); P = [w, beta]; mo = [np.zeros_like(w), np.zeros_like(beta)]; ve = [np.zeros_like(w), np.zeros_like(beta)]
    lr, b1, b2, eps = C("lr_default") * C("x17_dense_lr_mult"), C("adam_b1"), C("adam_b2"), C("adam_eps")

    def val(w_, b_):
        Hv[:, j] = np.tanh(V1 @ w_); return float(np.mean((b_[0] + Hv @ b_[1:] - yva) ** 2))
    best, bw, since = val(w, beta), w.copy(), 0
    for s in range(1, C("x18_ft_steps") + 1):
        H[:, j] = np.tanh(X1 @ P[0]); e = (P[1][0] + H @ P[1][1:] - ytr) * (2.0 / n)
        gb = np.concatenate([[e.sum()], H.T @ e])
        gw = X1.T @ (e * P[1][1 + j] * (1 - H[:, j] ** 2))
        for q, g in enumerate((gw, gb)):
            mo[q] = b1 * mo[q] + (1 - b1) * g; ve[q] = b2 * ve[q] + (1 - b2) * g * g
            P[q] = P[q] - lr * (mo[q] / (1 - b1 ** s)) / (np.sqrt(ve[q] / (1 - b2 ** s)) + eps)
        v = val(P[0], P[1])
        if v < best: best, bw, since = v, P[0].copy(), 0
        else:
            since += 1
            if since >= C("x18_ft_patience"): break
    return (bw[:m].copy(), float(bw[m]))


def head_val_bits(c, E):
    F, Fv, Gi, beta = B.solve(c, E)
    return float(B.vbits(c.yva - Fv @ beta, len(c.yva)))


def grads(units, E, d):
    """Size of the train-error gradient on each linked node's own weights (readout fixed at its ridge fit)."""
    (xtr, ytr) = d["tr"]; n = len(ytr)
    X1 = np.hstack([xtr, np.ones((n, 1))])
    U = np.array([np.concatenate([units[e][0], [units[e][1]]]) for e in E]); H = np.tanh(X1 @ U.T)
    F = np.hstack([np.ones((n, 1)), H]); beta = np.linalg.solve(F.T @ F + C("x17_ridge") * np.eye(len(E) + 1), F.T @ ytr)
    e = (F @ beta - ytr) * (2.0 / n)
    G = ((e[:, None] * beta[1:][None, :]) * (1 - H * H)).T @ X1
    return np.linalg.norm(G, axis=1)


def minimal_train(units, E, t, heads, caches, collateral):
    """Train the fewest nodes possible, one at a time, each kept only if its NET measured help pays; returns how many nodes changed."""
    changed, tried = 0, set()
    while len(tried) < len(E):
        order = [E[k] for k in np.argsort(-grads(units, E, {"tr": (caches[t].xtr, caches[t].ytr)})) if E[k] not in tried]
        i = order[0]; tried.add(i)
        users = [s for s in range(t) if i in heads[s]] if collateral else []
        before = {s: head_val_bits(caches[s], heads[s]) for s in users}; b0 = head_val_bits(caches[t], E)
        old = units[i]; units[i] = train_one(units, E, i, {"tr": (caches[t].xtr, caches[t].ytr), "va": (caches[t].xva, caches[t].yva)})
        M.refresh([caches[s] for s in users + [t]], units, [i])
        net = head_val_bits(caches[t], E) - b0 - sum(before[s] - head_val_bits(caches[s], heads[s]) for s in users)
        if net <= math.log2(len(E) + 1):                                 # does not pay: undo, and stop (progress stopped)
            units[i] = old; M.refresh([caches[s] for s in users + [t]], units, [i]); break
        M.refresh(caches, units, [i]); changed += 1
    return changed


def run_grow(arm, data, Wm, Bm, probe, seed):
    rng = np.random.default_rng([seed, 2 + ARMS.index(arm)])
    core = CORE.get(arm, arm)
    units, deg, heads, after, made, nchg = [], [], [], [], [], []
    caches = [None] * len(data)
    for t, d in enumerate(data):
        if arm == "FRESH":
            u_t, g_t, cs, created = [], [], [None] * len(data), []
            cs[t] = B.Cache(d)
            E = B.fit_head(core, u_t, g_t, cs, t, [], t, rng, True, created=created)
            if E: nchg.append(minimal_train(u_t, E, t, [[]] * t, cs, False))
            heads.append((u_t, E)); made.append(len(created)); after.append(B.head_bits(u_t, E, d)); continue
        caches[t] = B.Cache(d)
        for u in units: caches[t].add(*u)
        n0 = len(units); created = []
        if arm == "ANY":
            for _ in range(C("x17_any_new")):
                F, Fv, Gi, beta = B.solve(caches[t], list(range(len(units))))
                nu = B.train_unit(rng, caches[t].xtr, caches[t].ytr - F @ beta); units.append(nu); deg.append(0); created.append(len(units) - 1)
                for cx in caches:
                    if cx is not None: cx.add(*nu)
            E = list(range(len(units)))
        else:
            E = B.fit_head(core, units, deg, caches, t, [], t, rng, True, created=created)
        made.append(len(created))
        hs = heads if arm != "ANY" else [list(range(len(units)))] * t
        if E and arm == "FULLALL":
            M.finetune(units, E, d); M.refresh(caches, units, E); nchg.append(len(E))
        elif E and arm != "FULLFZ":
            nchg.append(minimal_train(units, E, t, hs, caches, arm != "FULLNC"))
        else: nchg.append(0)
        if arm != "ANY" and created:
            for s in range(t): heads[s] = B.fit_head(core, units, deg, caches, s, heads[s], t, rng, False, new_from=n0)
        heads.append(E); after.append(B.head_bits(units, E, d))
    if arm == "ANY": heads = [list(range(len(units)))] * len(data)
    if arm == "FRESH":
        final = [B.head_bits(u, E, data[s]) for s, (u, E) in enumerate(heads)]
        allu = [u for (us, _) in heads for u in us]; edges = sum(len(E) for _, E in heads)
    else:
        final = [B.head_bits(units, E, data[s]) for s, E in enumerate(heads)]; allu = units; edges = sum(len(E) for E in heads)
    H = np.tanh(probe @ np.array([u[0] for u in allu]).T + np.array([u[1] for u in allu])) if allu else np.zeros((len(probe), 0))
    cn, sel = B.conceptness(H, Wm, Bm, probe, len(Wm))
    return dict(after=after, final=final, final_head=final, units=len(allu), edges=edges, made=made, concept=cn, select=sel,
                trained=float(np.mean(nchg)) if nchg else 0.0)


def world(seed):
    t0 = time.time()
    data, Wm, Bm, probe = B.make_world(seed)
    res = {}
    for a in ARMS:
        if a == "DENSE": r = M.run_dense(data, Wm, Bm, probe, seed); r["trained"] = float(C("x17_dense_h"))
        else: r = run_grow(a, data, Wm, Bm, probe, seed)
        res[a] = r
    half = len(data) // 2
    out = {"oracle_late": float(np.mean(B.oracle(data, Wm, Bm)[half:])), "arms": {}}
    for a, r in res.items():
        out["arms"][a] = {"late": float(np.mean(r["after"][half:])), "early": float(np.mean(r["after"][:half])), "final_all": float(np.mean(r["final"])),
                          "units": r["units"], "edges": r["edges"], "concept": r["concept"], "select": r["select"], "trained": r["trained"],
                          "backward": float(np.mean(np.array(r["final"][:half]) - np.array(r["after"][:half])))}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(per, seed0):
    from x0_1 import boot_ci
    seeds = sorted(per)
    g = lambda a, k: np.array([per[s]["arms"][a][k] for s in seeds], float)
    S = {"median": {a: {k: float(np.median(g(a, k))) for k in per[seeds[0]]["arms"][a]} for a in ARMS}, "paired": {}, "worlds": len(seeds)}
    S["oracle_late"] = float(np.median([per[s]["oracle_late"] for s in seeds]))
    for a, b, k in (("HELP", "FRESH", "late"), ("TRAP", "HELP", "units"), ("FULLALL", "FULL", "trained"),
                    ("FULL", "FULLFZ", "late"), ("FULL", "FULLFZ", "final_all"), ("FULL", "FULLALL", "backward"), ("FULL", "FULLALL", "final_all"),
                    ("FULL", "FULLNC", "backward"), ("FULL", "DENSE", "final_all"), ("FULL", "DENSE", "late"), ("SPEC", "HELP", "late"),
                    ("TRAP", "FULL", "late"), ("FULL", "HELP", "late"), ("NEW", "HELP", "late"), ("ANY", "HELP", "late"), ("FULLALL", "FULLFZ", "late")):
        S["paired"][f"{k} {a}-{b}"] = boot_ci(g(a, k) - g(b, k), seed0)
    S["vertical FULL late-early"] = boot_ci(g("FULL", "late") - g("FULL", "early"), seed0)
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run. A win = paired 95% range on the predicted side of 0."""
    m, pr = S["median"], S["paired"]
    f = lambda k: f"{pr[k][0]:+.4f} [{pr[k][1]:+.4f}, {pr[k][2]:+.4f}]"
    up = lambda k: pr[k][1] > 0
    down = lambda k: pr[k][2] < 0
    manip = up("late HELP-FRESH") and up("units TRAP-HELP") and up("trained FULLALL-FULL") and m["FULL"]["trained"] > 0
    V = [("M-X1.9", "Fair test: reuse pays (HELP-FRESH), the existence-paid bonus acts (TRAP more units), and minimal training is minimal but not empty (FULL trains fewer nodes per problem than FULLALL, and more than 0)",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"HELP-FRESH {f('late HELP-FRESH')}; nodes trained per problem FULL {m['FULL']['trained']:.2f} vs FULLALL {m['FULLALL']['trained']:.2f}")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P395", "Training the fewest nodes beats training none: FULL minus FULLFZ, later-problem bits, above 0", up("late FULL-FULLFZ"), f("late FULL-FULLFZ"))
    v("P396", "Training the fewest nodes protects old problems better than training every linked node: FULL minus FULLALL, change in old problems' bits after they were learned, above 0", up("backward FULL-FULLALL"), f("backward FULL-FULLALL"))
    v("P397", "Overall, minimal training holds more on ALL problems at the end than training every linked node: FULL minus FULLALL final, above 0", up("final_all FULL-FULLALL"), f("final_all FULL-FULLALL"))
    v("P398", "Counting harm to old problems matters: FULL minus FULLNC, change in old problems' bits, above 0", up("backward FULL-FULLNC"), f("backward FULL-FULLNC"))
    v("P399", "With equal end-of-stream re-fits, the author's learner beats the plain network on all problems: FULL minus DENSE final, above 0", up("final_all FULL-DENSE"), f("final_all FULL-DENSE"))
    v("P400", "The specificity bonus still carries the gain: SPEC minus HELP later bits, above 0", up("late SPEC-HELP"), f("late SPEC-HELP"))
    v("P401", "The guard still holds: TRAP minus FULL later bits, below 0", down("late TRAP-FULL"), f("late TRAP-FULL"))
    v("P402", "Vertical: FULL later minus earlier bits, above 0", S["vertical FULL late-early"][1] > 0,
      f"{S['vertical FULL late-early'][0]:+.4f} [{S['vertical FULL late-early'][1]:+.4f}, {S['vertical FULL late-early'][2]:+.4f}]")
    return V


ARM_TEXT = dict(B.ARM_TEXT)
ARM_TEXT.update({"FULLFZ": ("Author's rule, no node trains", "Nodes never change after they are made (X1.7)."),
                 "FULLALL": ("Author's rule, all linked nodes train", "Every node a problem links to is trained on it (X1.8's design)."),
                 "FULLNC": ("Author's rule, fewest nodes, harm ignored", "Trains the fewest nodes, but does not count harm to old problems.")})
for a in ("FRESH", "HELP", "SPEC", "NEW", "FULL", "TRAP", "ANY"):
    ARM_TEXT[a] = (ARM_TEXT[a][0], ARM_TEXT[a][1] + " Trains the fewest nodes possible, one at a time, each only if it pays net of harm to old problems.")


def log_wandb(out):
    from track import Tracker
    S = out["summary"]
    tk = Tracker("X1.9", "Growing by connecting, training the fewest nodes", "If each problem changes only the fewest nodes it must (everything else untouched), does the author's connection rule keep its gains and protect what it already knows?",
                 B.WORLD_TEXT, ARM_TEXT, B.READ_TEXT + " 'At the end' re-fits every learner's output weights on each problem's stored examples, so all arms are scored the same way.",
                 dict(worlds=S["worlds"], problems=C("x17_problems")), tags=["connection", "growth", "vertical", "minimal-plasticity"])
    keys = [("late", "Bits per point, later problems (higher is better)"), ("early", "Bits per point, earlier problems"), ("final_all", "Bits per point at the end, all problems"),
            ("backward", "Old problems changed later (bits)"), ("trained", "Nodes trained per problem"), ("units", "Nodes"), ("edges", "Links")]
    tk.table("Summary by arm (median over worlds)", ["Arm"] + [k[1] for k in keys], [[tk.name(a)] + [round(S["median"][a][k], 4) for k, _ in keys] for a in ARMS])
    tk.table("Paired differences with 95% ranges (same worlds, A minus B)", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    import multiprocessing as mp, budget
    t0 = time.time()
    if mode == "log":
        log_wandb(json.load(open(os.path.join(OUT, "x1_9_run.json")))["result"]); return
    seed0 = C("x19_seed0"); nw = C("x17_worlds")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    per, history, bs, V, S = {}, [], C("x17_batch"), None, None
    with mp.get_context("spawn").Pool(budget.workers()) as pool:
        for start in range(0, nw, bs):
            seeds = [seed0 + i for i in range(start, min(nw, start + bs))]
            for s_, r in zip(seeds, pool.map(world, seeds, chunksize=1)): per[s_] = r
            S = summarize(per, seed0); V = judge(S); history.append([v[2] for v in V])
            print(f"{len(per)} worlds, {time.time() - t0:.0f} s:", " ".join(f"{v[0]}={v[2].split()[0]}" for v in V), flush=True)
            k = C("x17_stable_batches")
            if len(history) >= k and all(h == history[-1] for h in history[-k:]): break
    S["stopped_at_worlds"] = len(per)
    out = {"per_world": per, "summary": S, "verdicts": V, "verdict_history": history}
    os.makedirs(OUT, exist_ok=True)
    json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, f"x1_9_{mode}.json"), "w"), indent=1, default=float)
    if mode == "run": log_wandb(out)
    for v in V: print(v[0], v[2], "|", v[3])


if __name__ == "__main__":
    main(sys.argv[1])
