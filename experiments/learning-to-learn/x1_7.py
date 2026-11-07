"""X1.7: a network that grows by CONNECTING (author's design, 2026-11-02; inbox I21).

The author's rules: every node is valued by how it relates to everything else and EXACTLY how each connection helps or hurts; the structure is
continuously modified; elevated payment for NEW nodes; elevated reward for connecting to more SPECIFIC / less connected nodes. Node = a hidden unit.

World: a stream of T problems. Hidden 'concepts' c_j(x) = tanh(w_j.x + b_j) on x ~ N(0, I_m). Each problem's target is a sparse sum of 3 concepts
plus a little noise; with probability p_new one of the 3 is a brand-new concept, so the pool keeps growing (endless room) while most of each problem
reuses what came before (vertical: a learner that connects to old nodes needs fewer examples). Each problem gives 60 train, 60 decision ('val')
and 500 test points; test points are never used for any decision.
Learner (all growing arms): a pool of units (nodes), each frozen once made; each problem has an output head (ridge readout) with a SET of edges to
units. Per problem, greedy: candidates = every unconnected unit + one freshly trained unit (fit to the current residual on the train split). The
measured help of a candidate = bits it adds on the val split (n_val x 0.5 log2(MSE without / MSE with)); exact, not estimated. Accept the best
candidate if its arm-adjusted value beats the selection cost log2(#candidates) bits; then prune any edge whose adjusted help is below the cost.
After each problem, every old head re-runs the add/prune loop with the units made since (structure keeps changing; backward transfer).
Arms:  FRESH (no reuse: each problem starts with no units)   HELP (measured help only)   SPEC (help x specificity bonus)   NEW (help x newcomer bonus)
       FULL (help x both; the author's rule)   FULL4 (same, bonuses 4x)   TRAP (both bonuses PAID ON EXISTENCE: added in bits whatever the help)
       ANY (connect everything: every unit to every head, a new unit every step)   DENSE (plain network: 32 shared units + heads, gradient descent on all)
Specificity bonus: multiplier 1 + lam*log2((P+1)/(deg+1)), P = problems seen, deg = heads the unit already serves (rarer node pays more).
Newcomer bonus: multiplier 1 + lam on a unit that does not exist yet.
Usage: python x1_7.py selftest | run | log        (writes ../results/x1_7_<mode>.json)"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C

OUT = os.path.join(str(vipaths.RESULTS))
ARMS = ("FRESH", "HELP", "SPEC", "NEW", "FULL", "FULL4", "TRAP", "ANY", "DENSE")
BON = {"HELP": (0, 0), "SPEC": (1, 0), "NEW": (0, 1), "FULL": (1, 1), "FULL4": (C("x17_lam_hi"), C("x17_lam_hi")), "TRAP": (1, 1), "FRESH": (0, 0)}


# ---------------------------------------------------------------- world -------------------------------------------------------------
def make_world(seed):
    rng = np.random.default_rng([seed, 1])
    m, T, k = C("x17_dim"), C("x17_problems"), C("x17_k")
    W, B, probs = [], [], []
    newc = lambda: (W.append(rng.standard_normal(m) * C("x17_gain") / math.sqrt(m)), B.append(rng.standard_normal() * C("x17_bias_sd")))
    for _ in range(C("x17_pool0")): newc()
    for t in range(T):
        if rng.random() < C("x17_p_new"):
            newc(); S = [len(W) - 1] + list(rng.choice(len(W) - 1, k - 1, replace=False))
        else:
            S = list(rng.choice(len(W), k, replace=False))
        a = rng.choice([-1, 1], k) * rng.uniform(C("x17_a_lo"), C("x17_a_hi"), k)
        probs.append((S, a))
    Wm, Bm = np.array(W), np.array(B)
    data = []
    for S, a in probs:
        n = C("x17_n_tr") + C("x17_n_val") + C("x17_n_te") + C("x17_norm_n")
        x = rng.standard_normal((n, m))
        y = np.tanh(x @ Wm[S].T + Bm[S]) @ a
        y = y + rng.standard_normal(n) * C("x17_noise") * y.std()
        mu, sd = y[-C("x17_norm_n"):].mean(), y[-C("x17_norm_n"):].std()
        y = (y - mu) / sd
        i1, i2, i3 = C("x17_n_tr"), C("x17_n_tr") + C("x17_n_val"), C("x17_n_tr") + C("x17_n_val") + C("x17_n_te")
        data.append(dict(tr=(x[:i1], y[:i1]), va=(x[i1:i2], y[i1:i2]), te=(x[i2:i3], y[i2:i3]), S=S))
    probe = rng.standard_normal((C("x17_probe_n"), m))
    return data, Wm, Bm, probe


# ---------------------------------------------------------------- readout and measured help -------------------------------------
def ridge(F, y):
    F1 = np.hstack([np.ones((len(F), 1)), F])
    A = F1.T @ F1 + C("x17_ridge") * np.eye(F1.shape[1])
    return np.linalg.solve(A, F1.T @ y)


def predict(F, beta):
    return beta[0] + F @ beta[1:]


def mse_with(Ftr, ytr, Fva, yva):
    if Ftr.shape[1] == 0: return float(np.mean((yva - ytr.mean()) ** 2))
    return float(np.mean((yva - predict(Fva, ridge(Ftr, ytr))) ** 2))


def bits(mse, n):
    return n * 0.5 * math.log2(1.0 / max(mse, C("x17_mse_floor")))


def train_unit(rng, x, r):
    """Fit a*tanh(u.x+b) to residual r by Adam (the only gradient-descent step in the growing arms); stops early once progress stalls.
    Parameters packed in one vector [u, b, a] so each Adam step is a handful of array operations."""
    n, m = x.shape
    p = np.concatenate([rng.standard_normal(m) / math.sqrt(m), [0.0, C("x17_a0")]])
    X1 = np.hstack([x, np.ones((n, 1))])
    mo = np.zeros(m + 2); ve = np.zeros(m + 2); g = np.zeros(m + 2)
    lr, b1, b2, eps = C("x17_unit_lr"), C("adam_b1"), C("adam_b2"), C("adam_eps"); last = float(np.mean(r ** 2)); ck = C("x17_check_every")
    k = 2.0 / n
    for s in range(1, C("x17_unit_steps") + 1):
        h = np.tanh(X1 @ p[:-1]); e = p[-1] * h - r
        g[:-1] = X1.T @ (e * (p[-1] * k) * (1 - h * h)); g[-1] = k * (e @ h)
        mo = b1 * mo + (1 - b1) * g; ve = b2 * ve + (1 - b2) * g * g
        p = p - lr * (mo / (1 - b1 ** s)) / (np.sqrt(ve / (1 - b2 ** s)) + eps)
        if s % ck == 0:
            cur = float(e @ e) / n
            if cur > last * (1 - C("x17_stall")): break
            last = cur
    return (p[:m], float(p[m]))


def adjusted(arm, help_, newcomer, deg, P, cost):
    """Vectorised over candidates: help_ (bits), newcomer (bool array), deg (heads already served)."""
    ls, ln = BON.get(arm, (0, 0))
    spec = np.where(newcomer, math.log2(P + 1), np.log2((P + 1) / (deg + 1.0)))
    nb = np.where(newcomer, ln, 0)
    if arm == "TRAP": return help_ + cost * (ls * spec + nb)             # bonuses paid on existence: added in bits, whatever the help
    return np.where(help_ > 0, help_ * (1 + ls * spec) * (1 + nb), help_)


class Cache:
    """Activations of every unit on one problem's train / decision points, computed once per unit."""
    def __init__(self, d):
        self.xtr, self.ytr = d["tr"]; self.xva, self.yva = d["va"]
        self.tr = np.zeros((len(self.ytr), 0)); self.va = np.zeros((len(self.yva), 0))

    def add(self, u, b):
        self.tr = np.hstack([self.tr, np.tanh(self.xtr @ u + b)[:, None]]); self.va = np.hstack([self.va, np.tanh(self.xva @ u + b)[:, None]])


def solve(c, E):
    """Exact ridge readout on units E; returns the pieces reused by the vectorised add / remove tests."""
    F = np.hstack([np.ones((len(c.ytr), 1)), c.tr[:, E]]); Fv = np.hstack([np.ones((len(c.yva), 1)), c.va[:, E]])
    Gi = np.linalg.inv(F.T @ F + C("x17_ridge") * np.eye(F.shape[1]))
    return F, Fv, Gi, Gi @ (F.T @ c.ytr)


def vbits(err, n):
    return n * 0.5 * np.log2(1.0 / np.maximum(np.mean(err ** 2, 0), C("x17_mse_floor")))


def fit_head(arm, units, deg, caches, t, E, P, rng, allow_new, new_from=0, created=None):
    """Greedy add / prune on head t with EXACT measured help (rank-one ridge updates give the same numbers as refitting each candidate)."""
    c = caches[t]; nva = len(c.yva)
    for _ in range(C("x17_max_steps")):
        F, Fv, Gi, beta = solve(c, E)
        r = c.ytr - F @ beta; pv = Fv @ beta; b0 = float(vbits(c.yva - pv, nva))
        idx = np.array([i for i in range(new_from, len(units)) if i not in E], int)
        Z, Zv = c.tr[:, idx], c.va[:, idx]
        nu = None
        if allow_new:
            nu = train_unit(rng, c.xtr, r)
            Z = np.hstack([Z, np.tanh(c.xtr @ nu[0] + nu[1])[:, None]]); Zv = np.hstack([Zv, np.tanh(c.xva @ nu[0] + nu[1])[:, None]])
        K = Z.shape[1]
        if K == 0: break
        cost = math.log2(K + 1)
        Wc = Gi @ (F.T @ Z)
        sch = (Z ** 2).sum(0) + C("x17_ridge") - (Z * (F @ Wc)).sum(0)
        cf = (Z.T @ r) / sch
        help_ = vbits(c.yva[:, None] - (pv[:, None] + cf * (Zv - Fv @ Wc)), nva) - b0
        isnew = np.zeros(K, bool); dg = np.zeros(K)
        if nu is not None: isnew[-1] = True
        if len(idx): dg[:len(idx)] = np.asarray(deg, float)[idx]
        v = adjusted(arm, help_, isnew, dg, P, cost)
        k = int(np.argmax(v))
        if v[k] <= cost: break
        if isnew[k]:
            units.append(nu); deg.append(0); j = len(units) - 1
            for cx in caches:
                if cx is not None: cx.add(*nu)
            if created is not None: created.append(j)
        else: j = int(idx[k])
        E.append(j); deg[j] += 1
        while len(E) > 1:                                              # prune: drop the link that pays least, if it no longer pays its cost
            F, Fv, Gi, beta = solve(c, E); bf = float(vbits(c.yva - Fv @ beta, nva))
            jj = np.arange(1, len(E) + 1)
            B = beta[:, None] - Gi[:, jj] * (beta[jj] / Gi[jj, jj])        # exact leave-one-out coefficients
            hl = bf - vbits(c.yva[:, None] - Fv @ B, nva)
            av = adjusted(arm, hl, np.zeros(len(E), bool), np.array([deg[e] - 1 for e in E], float), P, cost)
            w = int(np.argmin(av))
            if av[w] > cost: break
            deg[E[w]] -= 1; E.pop(w)
    return E


def head_bits(units, E, d, split="te"):
    xtr, ytr = d["tr"]; x, y = d[split]
    act = lambda z: np.tanh(z @ np.array([units[i][0] for i in E]).T + np.array([units[i][1] for i in E])) if E else np.zeros((len(z), 0))
    F = np.hstack([np.ones((len(ytr), 1)), act(xtr)]); Ft = np.hstack([np.ones((len(y), 1)), act(x)])
    beta = np.linalg.solve(F.T @ F + C("x17_ridge") * np.eye(F.shape[1]), F.T @ ytr)
    return bits(float(np.mean((y - Ft @ beta) ** 2)), len(y)) / len(y)


def conceptness(H, Wm, Bm, probe, n_concepts):
    """Per unit activation H [n, U]: max |corr| with one true concept and the gap to the second-best (selectivity)."""
    if H.shape[1] == 0: return 0.0, 0.0
    Cc = np.tanh(probe @ Wm[:n_concepts].T + Bm[:n_concepts])
    Hz = (H - H.mean(0)) / (H.std(0) + 1e-9); Cz = (Cc - Cc.mean(0)) / (Cc.std(0) + 1e-9)
    R = np.abs(Hz.T @ Cz) / len(probe)
    s = np.sort(R, 1)
    return float(s[:, -1].mean()), float((s[:, -1] - (s[:, -2] if s.shape[1] > 1 else 0)).mean())


def run_grow(arm, data, Wm, Bm, probe, seed):
    rng = np.random.default_rng([seed, 2 + ARMS.index(arm)])
    units, deg, heads, after, made = [], [], [], [], []
    caches = [None] * len(data)
    for t, d in enumerate(data):
        if arm == "FRESH":                                             # no reuse: a private pool per problem
            u_t, g_t, cs, created = [], [], [None] * len(data), []
            cs[t] = Cache(d)
            E = fit_head(arm, u_t, g_t, cs, t, [], t, rng, True, created=created)
            heads.append((u_t, E)); made.append(len(created)); after.append(head_bits(u_t, E, d)); continue
        caches[t] = Cache(d)
        for u in units: caches[t].add(*u)
        if arm == "ANY":
            for _ in range(C("x17_any_new")):
                F, Fv, Gi, beta = solve(caches[t], list(range(len(units))))
                nu = train_unit(rng, caches[t].xtr, caches[t].ytr - F @ beta); units.append(nu); deg.append(0)
                for cx in caches:
                    if cx is not None: cx.add(*nu)
            E = list(range(len(units))); made.append(C("x17_any_new"))
        else:
            n0 = len(units); created = []
            E = fit_head(arm, units, deg, caches, t, [], t, rng, True, created=created)
            made.append(len(created))
            if created:                                                # the structure keeps changing: old heads may connect to the new units
                for s in range(t): heads[s] = fit_head(arm, units, deg, caches, s, heads[s], t, rng, False, new_from=n0)
        heads.append(E); after.append(head_bits(units, E, d))
    if arm == "ANY": heads = [list(range(len(units)))] * len(data)
    if arm == "FRESH":
        final = [head_bits(u, E, data[s]) for s, (u, E) in enumerate(heads)]
        allu = [u for (us, _) in heads for u in us]; edges = sum(len(E) for _, E in heads)
    else:
        final = [head_bits(units, E, data[s]) for s, E in enumerate(heads)]; allu = units; edges = sum(len(E) for E in heads)
    H = np.tanh(probe @ np.array([u[0] for u in allu]).T + np.array([u[1] for u in allu])) if allu else np.zeros((len(probe), 0))
    cn, sel = conceptness(H, Wm, Bm, probe, len(Wm))
    return dict(after=after, final=final, units=len(allu), edges=edges, made=made, concept=cn, select=sel)


def run_dense(data, Wm, Bm, probe, seed):
    """Plain network: H shared tanh units, one linear head per problem, Adam on everything, problems in sequence, best-val step kept."""
    rng = np.random.default_rng([seed, 2 + ARMS.index("DENSE")])
    m, H, T = C("x17_dim"), C("x17_dense_h"), len(data)
    P = {"W": rng.standard_normal((m, H)) / math.sqrt(m), "b": np.zeros(H), "V": np.zeros((H, T)), "c": np.zeros(T)}
    mo = {k: np.zeros_like(v) for k, v in P.items()}; ve = {k: np.zeros_like(v) for k, v in P.items()}
    lr, b1, b2 = C("lr_default") * C("x17_dense_lr_mult"), C("adam_b1"), C("adam_b2"); step = 0
    pred = lambda x, t: np.tanh(x @ P["W"] + P["b"]) @ P["V"][:, t] + P["c"][t]
    after = []
    for t, d in enumerate(data):
        (xtr, ytr), (xva, yva) = d["tr"], d["va"]
        best = (np.mean((pred(xva, t) - yva) ** 2), {k: v.copy() for k, v in P.items()}); since = 0
        for _ in range(C("x17_dense_steps")):
            step += 1
            h = np.tanh(xtr @ P["W"] + P["b"]); e = (h @ P["V"][:, t] + P["c"][t] - ytr) * 2 / len(ytr)
            gV = np.zeros_like(P["V"]); gV[:, t] = h.T @ e
            gc = np.zeros_like(P["c"]); gc[t] = e.sum()
            gz = np.outer(e, P["V"][:, t]) * (1 - h ** 2)
            G = {"W": xtr.T @ gz, "b": gz.sum(0), "V": gV, "c": gc}
            for k in P:
                mo[k] = b1 * mo[k] + (1 - b1) * G[k]; ve[k] = b2 * ve[k] + (1 - b2) * G[k] ** 2
                P[k] = P[k] - lr * (mo[k] / (1 - b1 ** step)) / (np.sqrt(ve[k] / (1 - b2 ** step)) + C("adam_eps"))
            v = np.mean((pred(xva, t) - yva) ** 2)
            if v < best[0]: best = (v, {k: q.copy() for k, q in P.items()}); since = 0
            else:
                since += 1
                if since >= C("x17_dense_patience"): break                # progress stopped: stop early
        P.update(best[1])
        xte, yte = d["te"]; after.append(bits(np.mean((pred(xte, t) - yte) ** 2), len(yte)) / len(yte))
    final = [bits(np.mean((pred(d["te"][0], s) - d["te"][1]) ** 2), len(d["te"][1])) / len(d["te"][1]) for s, d in enumerate(data)]
    cn, sel = conceptness(np.tanh(probe @ P["W"] + P["b"]), Wm, Bm, probe, len(Wm))
    global _dense_last; _dense_last = (P["W"].copy(), P["b"].copy())       # read by x1_8 (no effect on X1.7's numbers)
    return dict(after=after, final=final, units=H, edges=H * T, made=[0] * T, concept=cn, select=sel)


def oracle(data, Wm, Bm):
    """Reference: the readout given the TRUE concepts of each problem (ceiling for this sample size)."""
    out = []
    for d in data:
        f = lambda x: np.tanh(x @ Wm[d["S"]].T + Bm[d["S"]])
        out.append(bits(mse_with(f(d["tr"][0]), d["tr"][1], f(d["te"][0]), d["te"][1]), len(d["te"][1])) / len(d["te"][1]))
    return out


def world(seed):
    t0 = time.time()
    data, Wm, Bm, probe = make_world(seed)
    res = {a: (run_dense(data, Wm, Bm, probe, seed) if a == "DENSE" else run_grow(a, data, Wm, Bm, probe, seed)) for a in ARMS}
    half = len(data) // 2
    new_flags = [len(set(d["S"]) - set().union(*[set(e["S"]) for e in data[:i]])) > 0 if i else True for i, d in enumerate(data)]
    out = {"oracle_late": float(np.mean(oracle(data, Wm, Bm)[half:])), "n_new_concepts": int(sum(new_flags)), "seconds": 0.0, "arms": {}}
    for a, r in res.items():
        out["arms"][a] = {"late": float(np.mean(r["after"][half:])), "early": float(np.mean(r["after"][:half])), "final_all": float(np.mean(r["final"])),
                          "units": r["units"], "edges": r["edges"], "made_early": int(sum(r["made"][:half])), "made_late": int(sum(r["made"][half:])),
                          "concept": r["concept"], "select": r["select"], "backward": float(np.mean(np.array(r["final"][:half]) - np.array(r["after"][:half])))}
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------- scoring -----------------------------------------------------------
def summarize(per, seed0):
    from x0_1 import boot_ci
    seeds = sorted(per)
    g = lambda a, k: np.array([per[s]["arms"][a][k] for s in seeds], float)
    S = {"median": {a: {k: float(np.median(g(a, k))) for k in per[seeds[0]]["arms"][a]} for a in ARMS}, "paired": {}}
    S["oracle_late"] = float(np.median([per[s]["oracle_late"] for s in seeds]))
    pr = lambda a, b, k: boot_ci(g(a, k) - g(b, k), seed0)
    for a, b, k in (("HELP", "FRESH", "late"), ("FULL", "HELP", "late"), ("SPEC", "HELP", "concept"), ("SPEC", "HELP", "select"), ("NEW", "HELP", "late"),
                    ("TRAP", "FULL", "late"), ("TRAP", "FULL", "units"), ("ANY", "HELP", "late"), ("FULL", "DENSE", "late"), ("FULL", "DENSE", "final_all"),
                    ("SPEC", "HELP", "late"), ("FULL4", "FULL", "late"), ("TRAP", "HELP", "units")):
        S["paired"][f"{k} {a}-{b}"] = pr(a, b, k)
    S["vertical FULL late-early"] = boot_ci(g("FULL", "late") - g("FULL", "early"), seed0)
    S["vertical FRESH late-early"] = boot_ci(g("FRESH", "late") - g("FRESH", "early"), seed0)
    S["worlds"] = len(seeds)
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run. A 'win' = the paired 95% range on the predicted
    side of 0 ('wins are wins', author 2026-11-02: no minimum-effect filter; effect sizes are reported, not used to discard)."""
    m, pr = S["median"], S["paired"]
    f = lambda k: f"{pr[k][0]:+.4f} [{pr[k][1]:+.4f}, {pr[k][2]:+.4f}]"
    up = lambda k: pr[k][1] > 0
    down = lambda k: pr[k][2] < 0
    manip = pr["late HELP-FRESH"][1] > 0 and pr["units TRAP-HELP"][1] > 0 and S["oracle_late"] > m["HELP"]["late"]
    V = [("M-X1.7", "Fair test: connecting to old nodes pays (HELP holds more bits per point than FRESH on the second half of the stream, range above 0); the existence-paid bonus acts (TRAP makes more units than HELP); the true-concept readout is above HELP (room left)",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"HELP-FRESH {f('late HELP-FRESH')}; TRAP-HELP units {f('units TRAP-HELP')}; oracle {S['oracle_late']:.3f} vs HELP {m['HELP']['late']:.3f}")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P381", "The author's rule (help x specificity x newcomer bonuses) holds more structure on later problems than measured help alone: FULL minus HELP, late bits", up("late FULL-HELP"), f("late FULL-HELP"))
    v("P382", "The specificity bonus makes nodes concept-like: SPEC's units match a single true concept more selectively than HELP's (selectivity, range above 0)", pr["select SPEC-HELP"][1] > 0, f("select SPEC-HELP"))
    v("P383", "The newcomer bonus helps: NEW minus HELP, late bits", up("late NEW-HELP"), f("late NEW-HELP"))
    v("P384", "THE GUARD: paying the bonuses on existence instead of on measured help makes more units AND holds less structure: TRAP minus FULL, units above 0 and late bits below 0", pr["units TRAP-FULL"][1] > 0 and down("late TRAP-FULL"), f"units {f('units TRAP-FULL')}; late {f('late TRAP-FULL')}")
    v("P385", "'More connected is better' fails without measured help: connecting everything (ANY) holds less than HELP, late bits", down("late ANY-HELP"), f("late ANY-HELP"))
    v("P386", "Vertical: the author's learner gets better at NEW problems with experience: FULL's second-half bits exceed its first-half bits (range above 0) while the no-reuse learner's do not rise by as much", S["vertical FULL late-early"][1] > 0 and S["vertical FULL late-early"][0] > S["vertical FRESH late-early"][0], f"FULL {S['vertical FULL late-early'][0]:+.4f} [{S['vertical FULL late-early'][1]:+.4f}, {S['vertical FULL late-early'][2]:+.4f}]; FRESH {S['vertical FRESH late-early'][0]:+.4f}")
    v("P387", "The author's learner beats a plain gradient-descent network with shared units on later problems: FULL minus DENSE, late bits", up("late FULL-DENSE"), f("late FULL-DENSE"))
    return V


# ---------------------------------------------------------------- reporting ---------------------------------------------------------
ARM_TEXT = {"FRESH": ("No reuse", "Builds new nodes for every problem; never connects to nodes made for earlier problems."),
            "HELP": ("Measured help only", "Connects to (or builds) whichever node adds the most structure on held-back points; cuts links that stop paying."),
            "SPEC": ("Help + specificity bonus", "As 'measured help', and a link to a rarely used node earns more."),
            "NEW": ("Help + newcomer bonus", "As 'measured help', and a brand-new node earns more."),
            "FULL": ("The author's rule", "Measured help multiplied by both bonuses (rare node, new node)."),
            "FULL4": ("The author's rule, bonuses x4", "Same with four times stronger bonuses."),
            "TRAP": ("Bonuses paid on existence (trap)", "The bonuses are paid for making a node or linking to a rare node even if the link helps nothing."),
            "ANY": ("Connect everything", "Every node links to every problem; new nodes are added every problem; more connected is assumed better."),
            "DENSE": ("Plain neural network", "32 shared units, one output per problem, gradient descent on everything, problems in sequence.")}
WORLD_TEXT = ("A long stream of prediction problems. Behind them are hidden 'concepts'; each problem mixes three of them, and every so often a brand-new "
              "concept appears, so there is always more to learn while most of each new problem reuses old ones. Each problem gives only 60 examples to "
              "learn from and 60 to judge links with; scoring uses 500 more it never sees.")
READ_TEXT = ("'Bits per point' = how much of a problem's real structure the learner holds, measured on unseen points (higher is better). 'Later problems' = the "
             "second half of the stream, where connecting to old nodes should pay. 'Concept match' = how cleanly a node tracks ONE hidden concept (1 = exactly).")


def log_wandb(out):
    from track import Tracker
    S = out["summary"]
    tk = Tracker("X1.7", "A network that grows by connecting", "Does the author's connection rule (measured help, bonuses for new and rarely used nodes) build a learner that gets better at new problems with experience, and do its nodes become concepts?",
                 WORLD_TEXT, ARM_TEXT, READ_TEXT, dict(worlds=S["worlds"], problems=C("x17_problems")), tags=["connection", "growth", "vertical", "concepts"])
    keys = [("late", "Bits per point, later problems (higher is better)"), ("early", "Bits per point, earlier problems"), ("final_all", "Bits per point at the end, all problems"),
            ("units", "Nodes"), ("edges", "Links"), ("concept", "Concept match of nodes (1 = exact)"), ("select", "Selectivity (one concept, not several)"), ("backward", "Old problems improved later (bits)")]
    tk.table("Summary by arm (median over worlds)", ["Arm"] + [k[1] for k in keys], [[tk.name(a)] + [round(S["median"][a][k], 4) for k, _ in keys] for a in ARMS])
    tk.table("Paired differences with 95% ranges (same worlds, A minus B)", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    import budget
    t0 = time.time()
    if mode == "log":
        log_wandb(json.load(open(os.path.join(OUT, "x1_7_run.json")))["result"]); return
    seed0 = C("x17_selftest_seed") if mode == "selftest" else C("x17_seed0")
    nw = C("x17_selftest_worlds") if mode == "selftest" else C("x17_worlds")
    per, history, bs = {}, [], C("x17_batch")
    for start in range(0, nw, bs):                                     # batches; stop once the verdicts have stopped changing
        seeds = [seed0 + i for i in range(start, min(nw, start + bs))]
        for s_, r in zip(seeds, budget.pmap(world, [(s_,) for s_ in seeds])): per[s_] = r
        S = summarize(per, seed0); V = judge(S); history.append([v[2] for v in V])
        print(f"{len(per)} worlds, {time.time() - t0:.0f} s:", " ".join(f"{v[0]}={v[2].split()[0]}" for v in V), flush=True)
        k = C("x17_stable_batches")
        if len(history) >= k and all(h == history[-1] for h in history[-k:]): break
    S["stopped_at_worlds"] = len(per)
    out = {"per_world": per, "summary": S, "verdicts": V, "verdict_history": history}
    os.makedirs(OUT, exist_ok=True)
    json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, f"x1_7_{mode}.json"), "w"), indent=1, default=float)
    print(json.dumps({"median": S["median"], "oracle_late": S["oracle_late"]}, indent=1, default=float))
    if mode == "run": log_wandb(out)
    for v in out["verdicts"]: print(v[0], v[2], "|", v[3])


if __name__ == "__main__":
    main(sys.argv[1])
