"""X8.1d: routine creation with the within-problem spread learned from audit-confirmed assignments (error rule as bootstrap), plus a FRESH (no memory) control.
Registered in paper/PREDICTIONS.md (P357-P360) before this was run. Usage: python x8_1d.py selftest | run | log   (writes ../results/x8_1d_run.json)"""
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


class Sig4:
    """class means plus ONE pooled within-class scatter, fed only by deviations from a class that already had a member"""
    def __init__(self):
        self.n = []; self.mu = []; self.ss = np.zeros(DIM); self.dof = 0

    def update(self, z, c):
        while len(self.n) <= c:
            self.n.append(0); self.mu.append(np.zeros(DIM))
        if self.n[c] >= 1:
            d = z - self.mu[c]; self.n[c] += 1; self.mu[c] = self.mu[c] + d / self.n[c]; self.ss = self.ss + d * (z - self.mu[c]); self.dof += 1
        else:
            self.n[c] = 1; self.mu[c] = z.copy()

    def ready(self):
        return self.dof >= C("x81d_dof_min")

    def thr(self):
        from scipy.stats import f
        return float(DIM * f.ppf(C("x81b_novel_q"), DIM, self.dof))

    def d2(self, z, c):
        var = np.maximum(self.ss / self.dof, C("x81_var_floor")) * (1 + 1 / self.n[c])
        return float((((z - self.mu[c]) ** 2) / var).sum())

    def nearest(self, z):
        live = [c for c, n in enumerate(self.n) if n >= 1]
        d = [self.d2(z, c) for c in live]; i = int(np.argmin(d))
        return live[i], d[i]


def one(wset):
    import pp5_mastery as P
    from nn import RunStd
    NP = P.NP; w = C("x81b_window"); base = C("x81d_seed0") + 100 * wset
    pseeds = [base + i for i in range(NP)]; refs = [P.ref_sd(ps) for ps in pseeds]
    rng = np.random.default_rng(base); order = rng.permutation(np.repeat(np.arange(NP), P.REP)); seen = np.zeros(NP, int)
    libs = {"SIG_CREATE": [], "ERR_CREATE": []}; sig = Sig4(); tru = Sig4(); ora = {}; G = RunStd(18); rows = []
    for e, pid in enumerate(order):
        k = int(seen[pid]); seen[pid] += 1; X, D = P.episode_data(pseeds[pid], base * 1000 + e); ref = refs[pid]
        row = dict(pid=int(pid), k=k); gm = G.m.copy(); gs = np.sqrt(np.maximum(G.s2, 1e-6))
        z = signature(X, w, gm, gs) if G.n > 1 else np.zeros(DIM)
        row["true_exceed"] = (bool(tru.d2(z, int(pid)) > tru.thr()) if (k >= 1 and tru.ready() and len(tru.n) > pid and tru.n[pid] >= 1) else None)
        for arm in ("SIG_CREATE", "ERR_CREATE"):
            lib = libs[arm]
            cs = np.cumsum(np.array([[P.chunk_error_only(c, X[t], D[t], ref)[0] for t in range(P.EP)] for c in lib]), 1) if lib else np.zeros((0, P.EP))
            new = P.Chunk(18, 6, 7 * e + 3); fresh_cum = 0.0; errs = []
            for t in range(w):
                ag, ev, _ = new.step(X[t], D[t], ref); fresh_cum += ag; errs.append(ev)
            err_use = int(np.argmin(cs[:, w - 1])) if lib and cs[:, w - 1].min() < fresh_cum else None
            if arm == "ERR_CREATE" or not sig.ready():
                use = err_use
            else:
                c_, d2 = sig.nearest(z); use = c_ if d2 <= sig.thr() else None
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
                    label = int(np.argmin(cs[:, -1]))                          # the audit verdict names the class (equals the used routine when it was right)
                sig.update(z, label)
            row[arm] = dict(late=float(np.mean(errs[w:])), rid=rid, created=use is None, n=len(lib))
        tru.update(z, int(pid))
        ch = ora.setdefault(int(pid), P.Chunk(18, 6, 7 * e + 5)); oe = [ch.step(X[t], D[t], ref)[1] for t in range(P.EP)]
        row["ORACLE"] = dict(late=float(np.mean(oe[w:])))
        fr = P.Chunk(18, 6, 7 * e + 9); fe = [fr.step(X[t], D[t], ref)[1] for t in range(P.EP)]
        row["FRESH"] = dict(late=float(np.mean(fe[w:]))); rows.append(row)
        for t in range(P.EP):
            G.update(X[t])
    return rows


def judge(S, per):
    m = S["median"]; v = lambda c: "PASSED" if c else "FAILED"; sets = sorted(per)
    ex = [np.mean([r["true_exceed"] for r in per[s] if r["true_exceed"] is not None]) for s in sets if any(r["true_exceed"] is not None for r in per[s])]
    fa = float(np.median(ex)) if ex else float("nan")
    fre = np.array([np.mean([r["FRESH"]["late"] for r in per[s] if r["k"] >= 1]) for s in sets]); sig = np.array(S["err"]["SIG_CREATE"]); sb_fresh = int((sig < fre).sum()); fr_med = float(np.median(fre))
    man = (fa <= C("x81c_false_alarm")) and m["err"]["ORACLE"] <= C("x81d_oracle_vs_fresh") * fr_med; n = m["n_routines"]["SIG_CREATE"]
    return [("P357", "Calibrated novelty test with true labels AND memory worth having (manipulation)", "PASSED" if man else "VOID", f"(a) true-label exceed share {fa:.3f} (bar <= {C('x81c_false_alarm')}); (b) ORACLE repeat error {m['err']['ORACLE']:.3f} vs FRESH {fr_med:.3f} (ratio {m['err']['ORACLE'] / fr_med:.2f}, bar <= {C('x81d_oracle_vs_fresh')})"),
            ("P358", "Routine count near 10", v(C("x81c_min_routines") <= n <= C("x81c_max_routines")), f"SIG_CREATE {n:.1f}, ERR_CREATE {m['n_routines']['ERR_CREATE']:.1f} routines"),
            ("P359", "Rarely wrong, mostly right", v(m["mis"]["SIG_CREATE"] <= C("x81b_misreuse") and m["own"]["SIG_CREATE"] >= C("x81c_own")), f"own {m['own']['SIG_CREATE']:.3f} (ERR {m['own']['ERR_CREATE']:.3f}), another's {m['mis']['SIG_CREATE']:.3f} (ERR {m['mis']['ERR_CREATE']:.3f}), new on repeat {m['made']['SIG_CREATE']:.3f} (ERR {m['made']['ERR_CREATE']:.3f})"),
            ("P360", "Near the oracle, better than the error rule and than no memory", v(m["err"]["SIG_CREATE"] <= C("x81b_err_ratio") * m["err"]["ORACLE"] and S["sets_better"] >= C("x81_sets_better") and sb_fresh >= C("x81_sets_better")), f"ORACLE {m['err']['ORACLE']:.3f}, SIG {m['err']['SIG_CREATE']:.3f} ({m['err']['SIG_CREATE'] / m['err']['ORACLE']:.2f}x), ERR {m['err']['ERR_CREATE']:.3f}, FRESH {fr_med:.3f}; SIG beats ERR in {S['sets_better']} sets, beats FRESH in {sb_fresh} of 24")]


def log_wandb(out):
    from track import Tracker
    S = out["summary"]; m = S["median"]; arms = dict(ARM_TEXT); arms["FRESH"] = ("No memory", "Starts a new learner every episode; the control for whether memory helps at all.")
    tk = Tracker("X8.1d", "Can the library learn how much one situation varies, from its own audits?", "Learn the normal spread of one situation from audit-confirmed repeats, then use it to decide whether a case is unfamiliar; compare with trying every routine and with no memory at all.",
                 "Ten different small causal worlds met ten times each in random order. No routines are given.", arms, "Routines = count at the end (10 would be perfect). Reuse = share of repeat encounters where the right routine was reused. Error: lower is better.", dict(sets=len(out["per_world"])), tags=["recognition", "creation"])
    tk.table("Summary by arm (median over world sets)", ["Arm", "Routines", "Reused right", "Reused wrong", "Made new on repeat", "Repeat error"], [[a, *[round(m[k].get(a, float("nan")), 3) for k in ("n_routines", "own", "mis", "made", "err")]] for a in ARMS])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    t0 = time.time()
    if mode == "selftest":
        r = one(70); S = summarize({0: r}); print(json.dumps(S["median"]), round(time.time() - t0), "s"); print("selftest ok"); return
    if mode == "log":
        log_wandb(json.load(open(os.path.join(OUT, "x8_1d_run.json")))["result"]); return
    from budget import pmap, workers
    n = workers(0.5); print("workers", n, flush=True); sets = list(range(C("x81_sets")))
    res = pmap(one, [(s,) for s in sets], n); per = dict(zip(sets, res)); S = summarize(per); out = {"per_world": per, "summary": S, "verdicts": judge(S, per)}
    json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x8_1d_run.json"), "w"), default=float)
    print(json.dumps(S["median"]))
    for v in out["verdicts"]: print(v[0], v[2], "|", v[3], flush=True)
    log_wandb(out)


if __name__ == "__main__":
    main(sys.argv[1])
