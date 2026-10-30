"""X8.1c: X8.1b with ONE change: the novelty test's variance is the case's own sampling variance (block-wise standard error of its first 50 ticks).
Registered in paper/PREDICTIONS.md (P353-P356) before this was run. Usage: python x8_1c.py selftest | run | log   (writes ../results/x8_1c_run.json)"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "experiments"), os.path.join(ROOT, "src"), HERE):
    sys.path.insert(0, p)
os.environ["CN_FULL"] = "1"
import numpy as np
from constants import C
from x8_1 import signature
from x8_1b import summarize, ARMS, ARM_TEXT, DIM

OUT = os.path.join(ROOT, "results")


def case_se2(X, w, gs):
    """squared standard error of each signature feature from the case itself: variance of the block values / number of blocks"""
    nb = C("x81c_blocks"); B = X[:w].reshape(nb, w // nb, -1)
    m = B.mean(1) / gs; s = np.log((B.std(1) + 1e-3) / gs)
    return np.concatenate([m.var(0, ddof=1), s.var(0, ddof=1)]) / nb


class Sig3:
    def __init__(self):
        self.n = []; self.mu = []

    def update(self, z, c):
        while len(self.n) <= c:
            self.n.append(0); self.mu.append(np.zeros(DIM))
        self.n[c] += 1; self.mu[c] = self.mu[c] + (z - self.mu[c]) / self.n[c]

    def d2(self, z, se2, c):
        return float((((z - self.mu[c]) ** 2) / (se2 * (1 + 1 / self.n[c]) + C("x81_var_floor"))).sum())

    def nearest(self, z, se2):
        live = [c for c, n in enumerate(self.n) if n >= 1]
        if not live:
            return None, np.inf
        d = [self.d2(z, se2, c) for c in live]; i = int(np.argmin(d))
        return live[i], d[i]


def one(wset):
    import pp5_mastery as P
    from scipy.stats import chi2
    from nn import RunStd
    NP = P.NP; w = C("x81b_window"); base = C("x81c_seed0") + 100 * wset; thr = float(chi2.ppf(C("x81b_novel_q"), DIM))
    pseeds = [base + i for i in range(NP)]; refs = [P.ref_sd(ps) for ps in pseeds]
    rng = np.random.default_rng(base); order = rng.permutation(np.repeat(np.arange(NP), P.REP)); seen = np.zeros(NP, int)
    libs = {"SIG_CREATE": [], "ERR_CREATE": []}; sig = Sig3(); tru = Sig3(); ora = {}; G = RunStd(18); rows = []
    for e, pid in enumerate(order):
        k = int(seen[pid]); seen[pid] += 1; X, D = P.episode_data(pseeds[pid], base * 1000 + e); ref = refs[pid]
        row = dict(pid=int(pid), k=k); gm = G.m.copy(); gs = np.sqrt(np.maximum(G.s2, 1e-6))
        z = signature(X, w, gm, gs) if G.n > 1 else np.zeros(DIM); se2 = case_se2(X, w, gs) if G.n > 1 else np.ones(DIM)
        row["true_exceed"] = bool(k >= 1 and len(tru.n) > pid and tru.n[pid] >= 1 and tru.d2(z, se2, int(pid)) > thr)
        for arm in ("SIG_CREATE", "ERR_CREATE"):
            lib = libs[arm]
            cs = np.cumsum(np.array([[P.chunk_error_only(c, X[t], D[t], ref)[0] for t in range(P.EP)] for c in lib]), 1) if lib else np.zeros((0, P.EP))
            new = P.Chunk(18, 6, 7 * e + 3); fresh_cum = 0.0; errs = []
            for t in range(w):
                ag, ev, _ = new.step(X[t], D[t], ref); fresh_cum += ag; errs.append(ev)
            if arm == "ERR_CREATE":
                use = int(np.argmin(cs[:, w - 1])) if lib and cs[:, w - 1].min() < fresh_cum else None
            else:
                c_, d2 = sig.nearest(z, se2); use = c_ if (lib and c_ is not None and d2 <= thr) else None
            cur = lib[use] if use is not None else new
            for t in range(w, P.EP):
                errs.append(cur.step(X[t], D[t], ref)[1])
            if use is None:
                lib.append(new); rid = len(lib) - 1
            else:
                rid = use
            if arm == "SIG_CREATE":
                label = rid
                if use is not None and len(cs):
                    v = int(np.argmin(cs[:, -1]))
                    if v != use and cs[v, -1] < cs[use, -1]:
                        label = v
                sig.update(z, label)
            row[arm] = dict(late=float(np.mean(errs[w:])), rid=rid, created=use is None, n=len(lib))
        tru.update(z, int(pid))
        ch = ora.setdefault(int(pid), P.Chunk(18, 6, 7 * e + 5)); oe = [ch.step(X[t], D[t], ref)[1] for t in range(P.EP)]
        row["ORACLE"] = dict(late=float(np.mean(oe[w:]))); rows.append(row)
        for t in range(P.EP):
            G.update(X[t])
    return rows


def judge(S, per):
    m = S["median"]; v = lambda c: "PASSED" if c else "FAILED"
    fa = float(np.median([np.mean([r["true_exceed"] for r in per[s] if r["k"] >= 1]) for s in per])); man = fa <= C("x81c_false_alarm")
    n = m["n_routines"]["SIG_CREATE"]
    return [("P353", "Per-case variance is calibrated (manipulation)", "PASSED" if man else "VOID", f"share of repeat episodes whose own problem (true class mean) exceeds the 0.999 chi-square threshold: {fa:.3f} (bar <= {C('x81c_false_alarm')})"),
            ("P354", "Routine count near 10", v(C("x81c_min_routines") <= n <= C("x81c_max_routines")), f"SIG_CREATE {n:.1f}, ERR_CREATE {m['n_routines']['ERR_CREATE']:.1f} routines"),
            ("P355", "Rarely wrong, mostly right", v(m["mis"]["SIG_CREATE"] <= C("x81b_misreuse") and m["own"]["SIG_CREATE"] >= C("x81c_own")), f"own {m['own']['SIG_CREATE']:.3f} (ERR {m['own']['ERR_CREATE']:.3f}), another's {m['mis']['SIG_CREATE']:.3f} (ERR {m['mis']['ERR_CREATE']:.3f}), new on repeat {m['made']['SIG_CREATE']:.3f} (ERR {m['made']['ERR_CREATE']:.3f})"),
            ("P356", "Error near the oracle and better than the error rule", v(m["err"]["SIG_CREATE"] <= C("x81b_err_ratio") * m["err"]["ORACLE"] and S["sets_better"] >= C("x81_sets_better")), f"ORACLE {m['err']['ORACLE']:.3f}, SIG {m['err']['SIG_CREATE']:.3f} ({m['err']['SIG_CREATE'] / m['err']['ORACLE']:.2f}x), ERR {m['err']['ERR_CREATE']:.3f} ({m['err']['ERR_CREATE'] / m['err']['ORACLE']:.2f}x); SIG better in {S['sets_better']} of 24 sets")]


def log_wandb(out):
    from track import Tracker
    S = out["summary"]; m = S["median"]
    tk = Tracker("X8.1c", "Does measuring familiarity from the case itself stop the library merging situations?", "Same as X8.1b, but the test for whether a situation is unfamiliar uses the case's own measurement noise instead of a spread that includes differences between situations.",
                 "Ten different small causal worlds met ten times each in random order. No routines are given.", ARM_TEXT, "Routines = count at the end (10 would be perfect). Reuse = share of repeat encounters where the right routine was reused. Error: lower is better.", dict(sets=len(out["per_world"])), tags=["recognition", "creation"])
    tk.table("Summary by arm (median over world sets)", ["Arm", "Routines", "Reused right", "Reused wrong", "Made new on repeat", "Repeat error"], [[a, *[round(m[k].get(a, float("nan")), 3) for k in ("n_routines", "own", "mis", "made", "err")]] for a in ARMS])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    t0 = time.time()
    if mode == "selftest":
        r = one(70); S = summarize({0: r}); print(json.dumps(S["median"]), np.mean([x["true_exceed"] for x in r if x["k"] >= 1]), round(time.time() - t0), "s"); print("selftest ok"); return
    if mode == "log":
        log_wandb(json.load(open(os.path.join(OUT, "x8_1c_run.json")))["result"]); return
    from budget import pmap, workers
    n = workers(0.5); print("workers", n, flush=True); sets = list(range(C("x81_sets")))
    res = pmap(one, [(s,) for s in sets], n); per = dict(zip(sets, res)); S = summarize(per); out = {"per_world": per, "summary": S, "verdicts": judge(S, per)}
    json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x8_1c_run.json"), "w"), default=float)
    print(json.dumps(S["median"]))
    for v in out["verdicts"]: print(v[0], v[2], "|", v[3], flush=True)
    log_wandb(out)


if __name__ == "__main__":
    main(sys.argv[1])
