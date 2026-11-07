"""X1.8: X1.7 with NO frozen nodes (author, 2026-11-02: 'stop freezing nodes, everything should be equally malleable').

Same world, rule and arms as X1.7 (vi/x1_7.py). New: after a problem's links are chosen, every node linked to that problem AND its readout are
trained further by gradient descent (Adam) on the problem's train points, stopping early when the decision error stops improving. Shared nodes
therefore move, and can help or hurt the old problems that use them, exactly like the plain network's shared units.
Fairness fix (found while designing this test; applies to the X1.7 reading too): X1.7 re-fitted every growing arm's old readouts on that problem's
stored train points at scoring time, but not DENSE's. Here EVERY arm, DENSE included, is scored at the end with readouts re-fitted on stored train
points ('final', a linear probe of what the nodes still carry); 'final_head' keeps DENSE's own last-trained heads for reference.
Arms: X1.7's nine (now malleable) plus FULLFZ = the author's rule with frozen nodes (X1.7's FULL), for the direct comparison.
Usage: python x1_8.py selftest | run | log        (writes ../results/x1_8_<mode>.json)"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_7 as B

OUT = B.OUT
ARMS = ("FRESH", "HELP", "SPEC", "NEW", "FULL", "FULL4", "TRAP", "ANY", "DENSE", "FULLFZ")
BON = dict(B.BON); BON["FULLFZ"] = BON["FULL"]
B.BON.update(BON)


def finetune(units, E, d):
    """Adam on the linked nodes (u, b) and the readout, on the train points; keep the best decision-error step; stop when it stalls."""
    if not E: return
    (xtr, ytr), (xva, yva) = d["tr"], d["va"]
    n, m = xtr.shape; k = len(E)
    X1, V1 = np.hstack([xtr, np.ones((n, 1))]), np.hstack([xva, np.ones((len(yva), 1))])
    U = np.array([np.concatenate([units[i][0], [units[i][1]]]) for i in E])          # [k, m+1]
    H = np.tanh(X1 @ U.T); F = np.hstack([np.ones((n, 1)), H])
    beta = np.linalg.solve(F.T @ F + C("x17_ridge") * np.eye(k + 1), F.T @ ytr)
    P = [U, beta]; mo = [np.zeros_like(U), np.zeros_like(beta)]; ve = [np.zeros_like(U), np.zeros_like(beta)]
    lr, b1, b2, eps = C("lr_default") * C("x17_dense_lr_mult"), C("adam_b1"), C("adam_b2"), C("adam_eps")
    val = lambda U_, b_: float(np.mean((b_[0] + np.tanh(V1 @ U_.T) @ b_[1:] - yva) ** 2))
    best, bestP, since = val(U, beta), (U.copy(), beta.copy()), 0
    for s in range(1, C("x18_ft_steps") + 1):
        H = np.tanh(X1 @ P[0].T); e = (P[1][0] + H @ P[1][1:] - ytr) * (2.0 / n)
        gb = np.concatenate([[e.sum()], H.T @ e])
        gU = ((e[:, None] * P[1][1:][None, :]) * (1 - H * H)).T @ X1
        for i, g in enumerate((gU, gb)):
            mo[i] = b1 * mo[i] + (1 - b1) * g; ve[i] = b2 * ve[i] + (1 - b2) * g * g
            P[i] = P[i] - lr * (mo[i] / (1 - b1 ** s)) / (np.sqrt(ve[i] / (1 - b2 ** s)) + eps)
        v = val(P[0], P[1])
        if v < best: best, bestP, since = v, (P[0].copy(), P[1].copy()), 0
        else:
            since += 1
            if since >= C("x18_ft_patience"): break                     # progress stopped: stop early
    for j, i in enumerate(E): units[i] = (bestP[0][j, :m].copy(), float(bestP[0][j, m]))


def refresh(caches, units, idx):
    for c in caches:
        if c is None: continue
        for i in idx:
            u, b = units[i]; c.tr[:, i] = np.tanh(c.xtr @ u + b); c.va[:, i] = np.tanh(c.xva @ u + b)


def run_grow(arm, data, Wm, Bm, probe, seed):
    rng = np.random.default_rng([seed, 2 + ARMS.index(arm)])
    frozen = arm == "FULLFZ"; core = "FULL" if frozen else arm
    units, deg, heads, after, made = [], [], [], [], []
    caches = [None] * len(data); moved = []
    for t, d in enumerate(data):
        if arm == "FRESH":
            u_t, g_t, cs, created = [], [], [None] * len(data), []
            cs[t] = B.Cache(d)
            E = B.fit_head(core, u_t, g_t, cs, t, [], t, rng, True, created=created)
            finetune(u_t, E, d)
            heads.append((u_t, E)); made.append(len(created)); after.append(B.head_bits(u_t, E, d)); continue
        caches[t] = B.Cache(d)
        for u in units: caches[t].add(*u)
        if arm == "ANY":
            for _ in range(C("x17_any_new")):
                F, Fv, Gi, beta = B.solve(caches[t], list(range(len(units))))
                nu = B.train_unit(rng, caches[t].xtr, caches[t].ytr - F @ beta); units.append(nu); deg.append(0)
                for cx in caches:
                    if cx is not None: cx.add(*nu)
            E = list(range(len(units))); made.append(C("x17_any_new"))
        else:
            n0 = len(units); created = []
            E = B.fit_head(core, units, deg, caches, t, [], t, rng, True, created=created)
            made.append(len(created))
        if not frozen and E:
            before = [np.concatenate([units[i][0], [units[i][1]]]) for i in E]
            finetune(units, E, d); refresh(caches, units, E)
            moved += [float(np.linalg.norm(np.concatenate([units[i][0], [units[i][1]]]) - b0) / (np.linalg.norm(b0) + 1e-9)) for i, b0 in zip(E, before)]
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
                moved=float(np.median(moved)) if moved else 0.0)


def run_dense(data, Wm, Bm, probe, seed):
    """X1.7's plain network, plus the same end-of-stream readout re-fit on stored train points that every growing arm gets."""
    r = B.run_dense(data, Wm, Bm, probe, seed)
    W, b = B._dense_last
    units = [(W[:, i], float(b[i])) for i in range(W.shape[1])]
    E = list(range(W.shape[1]))
    r["final_head"] = r["final"]; r["final"] = [B.head_bits(units, E, d) for d in data]; r["moved"] = 0.0
    return r


def world(seed):
    t0 = time.time()
    data, Wm, Bm, probe = B.make_world(seed)
    res = {a: (run_dense(data, Wm, Bm, probe, seed) if a == "DENSE" else run_grow(a, data, Wm, Bm, probe, seed)) for a in ARMS}
    half = len(data) // 2
    out = {"oracle_late": float(np.mean(B.oracle(data, Wm, Bm)[half:])), "arms": {}}
    for a, r in res.items():
        out["arms"][a] = {"late": float(np.mean(r["after"][half:])), "early": float(np.mean(r["after"][:half])), "final_all": float(np.mean(r["final"])),
                          "final_head": float(np.mean(r["final_head"])), "units": r["units"], "edges": r["edges"],
                          "made_early": int(sum(r["made"][:half])), "made_late": int(sum(r["made"][half:])), "concept": r["concept"], "select": r["select"],
                          "backward": float(np.mean(np.array(r["final"][:half]) - np.array(r["after"][:half]))), "moved": r["moved"]}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarize(per, seed0):
    from x0_1 import boot_ci
    seeds = sorted(per)
    g = lambda a, k: np.array([per[s]["arms"][a][k] for s in seeds], float)
    S = {"median": {a: {k: float(np.median(g(a, k))) for k in per[seeds[0]]["arms"][a]} for a in ARMS}, "paired": {}, "worlds": len(seeds)}
    S["oracle_late"] = float(np.median([per[s]["oracle_late"] for s in seeds]))
    for a, b, k in (("HELP", "FRESH", "late"), ("FULL", "FULLFZ", "late"), ("FULL", "FULLFZ", "final_all"), ("FULL", "FULLFZ", "backward"),
                    ("FULL", "DENSE", "late"), ("FULL", "DENSE", "final_all"), ("SPEC", "HELP", "late"), ("TRAP", "FULL", "late"), ("TRAP", "HELP", "units"),
                    ("FULL", "HELP", "late"), ("NEW", "HELP", "late"), ("ANY", "HELP", "late"), ("FULL4", "FULL", "late")):
        S["paired"][f"{k} {a}-{b}"] = boot_ci(g(a, k) - g(b, k), seed0)
    S["vertical FULL late-early"] = boot_ci(g("FULL", "late") - g("FULL", "early"), seed0)
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run. A win = paired 95% range on the predicted side of 0."""
    m, pr = S["median"], S["paired"]
    f = lambda k: f"{pr[k][0]:+.4f} [{pr[k][1]:+.4f}, {pr[k][2]:+.4f}]"
    up = lambda k: pr[k][1] > 0
    down = lambda k: pr[k][2] < 0
    manip = up("late HELP-FRESH") and up("units TRAP-HELP") and m["FULL"]["moved"] > C("x18_moved_min")
    V = [("M-X1.8", "Fair test: reuse pays (HELP-FRESH later bits above 0), the existence-paid bonus acts (TRAP more units than HELP), and unfreezing acts (median relative change of a linked node's weights per fine-tune above 0.01)",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"HELP-FRESH {f('late HELP-FRESH')}; TRAP-HELP units {f('units TRAP-HELP')}; FULL moved {m['FULL']['moved']:.3f}")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P388", "Unfrozen nodes learn each new problem better: FULL minus FULLFZ, later-problem bits right after learning, above 0", up("late FULL-FULLFZ"), f("late FULL-FULLFZ"))
    v("P389", "Unfreezing costs memory: at the end of the stream FULL holds less on all problems than FULLFZ (final, readouts re-fitted for both), below 0", down("final_all FULL-FULLFZ"), f("final_all FULL-FULLFZ"))
    v("P390", "With equal end-of-stream readout re-fits, the author's rule still beats the plain network on all problems: FULL minus DENSE final, above 0", up("final_all FULL-DENSE"), f("final_all FULL-DENSE"))
    v("P391", "The author's rule beats the plain network on later problems: FULL minus DENSE later bits, above 0", up("late FULL-DENSE"), f("late FULL-DENSE"))
    v("P392", "The specificity bonus still carries the gain with unfrozen nodes: SPEC minus HELP later bits, above 0", up("late SPEC-HELP"), f("late SPEC-HELP"))
    v("P393", "The guard still holds: TRAP minus FULL later bits, below 0", down("late TRAP-FULL"), f("late TRAP-FULL"))
    v("P394", "Vertical with unfrozen nodes: FULL's later-problem bits exceed its earlier-problem bits, above 0", S["vertical FULL late-early"][1] > 0,
      f"{S['vertical FULL late-early'][0]:+.4f} [{S['vertical FULL late-early'][1]:+.4f}, {S['vertical FULL late-early'][2]:+.4f}]")
    return V


ARM_TEXT = dict(B.ARM_TEXT); ARM_TEXT["FULLFZ"] = ("The author's rule, frozen nodes", "X1.7's version: nodes never change after they are made.")
for a in ("FRESH", "HELP", "SPEC", "NEW", "FULL", "FULL4", "TRAP", "ANY"):
    ARM_TEXT[a] = (ARM_TEXT[a][0] + " (unfrozen)", ARM_TEXT[a][1] + " Nodes keep learning: each problem trains the nodes it links to.")


def log_wandb(out):
    from track import Tracker
    S = out["summary"]
    tk = Tracker("X1.8", "Growing by connecting, nothing frozen", "If every node stays as malleable as the plain network's units, does the author's connection rule still win, and what does unfreezing cost or buy?",
                 B.WORLD_TEXT, ARM_TEXT, B.READ_TEXT + " 'At the end' re-fits every learner's output weights on each problem's stored examples, so all arms are scored the same way.",
                 dict(worlds=S["worlds"], problems=C("x17_problems")), tags=["connection", "growth", "vertical", "malleable"])
    keys = [("late", "Bits per point, later problems (higher is better)"), ("early", "Bits per point, earlier problems"), ("final_all", "Bits per point at the end, all problems"),
            ("units", "Nodes"), ("edges", "Links"), ("backward", "Old problems changed later (bits)"), ("moved", "How much a node moves per training (relative)")]
    tk.table("Summary by arm (median over worlds)", ["Arm"] + [k[1] for k in keys], [[tk.name(a)] + [round(S["median"][a][k], 4) for k, _ in keys] for a in ARMS])
    tk.table("Paired differences with 95% ranges (same worlds, A minus B)", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    import multiprocessing as mp, budget
    t0 = time.time()
    if mode == "log":
        log_wandb(json.load(open(os.path.join(OUT, "x1_8_run.json")))["result"]); return
    seed0 = C("x18_selftest_seed") if mode == "selftest" else C("x18_seed0")
    nw = C("x17_selftest_worlds") if mode == "selftest" else C("x17_worlds")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(k, "1")
    per, history, bs, V, S = {}, [], C("x17_batch"), None, None
    with mp.get_context("spawn").Pool(budget.workers()) as pool:          # ONE pool for the whole run (X1.7 restarted it per batch)
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
    json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, f"x1_8_{mode}.json"), "w"), indent=1, default=float)
    if mode == "run": log_wandb(out)
    for v in V: print(v[0], v[2], "|", v[3])


if __name__ == "__main__":
    main(sys.argv[1])
