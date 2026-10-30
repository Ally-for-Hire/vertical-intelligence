"""X8.1b: routines CREATED from an empty library on the P5 stream (X8.1 gave them). SIG_CREATE vs ERR_CREATE vs ORACLE.
Registered in paper/PREDICTIONS.md (P347-P352) before this was run. Usage: python x8_1b.py selftest | run | log   (writes ../results/x8_1b_run.json)"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "experiments"), os.path.join(ROOT, "src"), HERE):
    sys.path.insert(0, p)
os.environ["CN_FULL"] = "1"
import numpy as np
from constants import C
from x8_1 import signature

OUT = os.path.join(ROOT, "results")
ARMS = ("SIG_CREATE", "ERR_CREATE", "ORACLE")
DIM = 36


class Sig2:
    """class means from labelled signatures; nearest class by squared distance on pooled within-class variance (once dof >= DIM) else the variance of all signatures"""
    def __init__(self):
        self.n = []; self.mu = []; self.ss = np.zeros(DIM); self.g = [0, np.zeros(DIM), np.zeros(DIM)]

    def update(self, z, c):
        while len(self.n) <= c:
            self.n.append(0); self.mu.append(np.zeros(DIM))
        self.n[c] += 1; d = z - self.mu[c]; self.mu[c] = self.mu[c] + d / self.n[c]; self.ss = self.ss + d * (z - self.mu[c])
        self.g[0] += 1; dg = z - self.g[1]; self.g[1] = self.g[1] + dg / self.g[0]; self.g[2] = self.g[2] + dg * (z - self.g[1])

    def nearest(self, z):
        live = [c for c, n in enumerate(self.n) if n >= 1]
        if not live or self.g[0] < 2:
            return None, np.inf
        dof = self.g[0] - len(live)
        var = self.ss / dof if dof >= C("x81b_dof_pooled") else self.g[2] / (self.g[0] - 1)
        var = np.maximum(var, C("x81_var_floor"))
        d2 = np.array([(((z - self.mu[c]) ** 2) / var).sum() for c in live]); i = int(np.argmin(d2))
        return live[i], float(d2[i])


def one(wset):
    import pp5_mastery as P
    from scipy.stats import chi2
    from nn import RunStd
    NP = P.NP; w = C("x81b_window"); base = C("x81b_seed0") + 100 * wset; thr = float(chi2.ppf(C("x81b_novel_q"), DIM))
    pseeds = [base + i for i in range(NP)]; refs = [P.ref_sd(ps) for ps in pseeds]
    rng = np.random.default_rng(base); order = rng.permutation(np.repeat(np.arange(NP), P.REP)); seen = np.zeros(NP, int)
    libs = {"SIG_CREATE": [], "ERR_CREATE": []}; sig = Sig2(); ora = {}; G = RunStd(18); rows = []
    for e, pid in enumerate(order):
        k = int(seen[pid]); seen[pid] += 1; X, D = P.episode_data(pseeds[pid], base * 1000 + e); ref = refs[pid]
        row = dict(pid=int(pid), k=k); gm = G.m.copy(); gs = np.sqrt(np.maximum(G.s2, 1e-6)); z = signature(X, w, gm, gs) if G.n > 1 else np.zeros(DIM)
        for arm in ("SIG_CREATE", "ERR_CREATE"):
            lib = libs[arm]
            cs = np.cumsum(np.array([[P.chunk_error_only(c, X[t], D[t], ref)[0] for t in range(P.EP)] for c in lib]), 1) if lib else np.zeros((0, P.EP))
            new = P.Chunk(18, 6, 7 * e + 3); fresh_cum = 0.0; errs = []
            for t in range(w):
                ag, ev, _ = new.step(X[t], D[t], ref); fresh_cum += ag; errs.append(ev)
            if arm == "ERR_CREATE":
                use = int(np.argmin(cs[:, w - 1])) if lib and cs[:, w - 1].min() < fresh_cum else None
            else:
                c_, d2 = sig.nearest(z); use = c_ if (lib and c_ is not None and d2 <= thr) else None
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
                        label = v                                           # audit verdict corrects the label
                sig.update(z, label)
            row[arm] = dict(late=float(np.mean(errs[w:])), rid=rid, created=use is None, n=len(lib))
        ch = ora.setdefault(int(pid), P.Chunk(18, 6, 7 * e + 5)); oe = [ch.step(X[t], D[t], ref)[1] for t in range(P.EP)]
        row["ORACLE"] = dict(late=float(np.mean(oe[w:])))
        rows.append(row)
        for t in range(P.EP):
            G.update(X[t])
    return rows


def summarize(per):
    sets = sorted(per); S = {"n_routines": {a: [] for a in ARMS[:2]}, "own": {a: [] for a in ARMS[:2]}, "mis": {a: [] for a in ARMS[:2]}, "made": {a: [] for a in ARMS[:2]}, "err": {a: [] for a in ARMS}, "mastery": {a: [] for a in ARMS}}
    for s in sets:
        R = per[s]
        for a in ARMS[:2]:
            votes = {}
            for r in R:
                votes.setdefault(r[a]["rid"], []).append(r["pid"])
            maj = {rid: max(set(v), key=v.count) for rid, v in votes.items()}; rep = [r for r in R if r["k"] >= 1]
            S["n_routines"][a].append(R[-1][a]["n"])
            S["made"][a].append(np.mean([r[a]["created"] for r in rep]))
            S["own"][a].append(np.mean([(not r[a]["created"]) and maj[r[a]["rid"]] == r["pid"] for r in rep]))
            S["mis"][a].append(np.mean([(not r[a]["created"]) and maj[r[a]["rid"]] != r["pid"] for r in rep]))
        for a in ARMS:
            S["err"][a].append(float(np.mean([r[a]["late"] for r in R if r["k"] >= 1])))
            S["mastery"][a].append(float(np.mean([r[a]["late"] for r in R if r["k"] == 9]) / np.mean([r[a]["late"] for r in R if r["k"] == 0])))
    med = lambda v: float(np.median(v)); S["median"] = {key: {a: med(v) for a, v in d.items()} for key, d in S.items() if isinstance(d, dict) and key not in ("median",)}
    S["sets_fewer"] = int(sum(np.array(S["n_routines"]["SIG_CREATE"]) < np.array(S["n_routines"]["ERR_CREATE"])))
    S["sets_better"] = int(sum(np.array(S["err"]["SIG_CREATE"]) < np.array(S["err"]["ERR_CREATE"])))
    S["own_gain"] = med(np.array(S["own"]["SIG_CREATE"]) - np.array(S["own"]["ERR_CREATE"]))
    return S


def judge(S):
    m = S["median"]; v = lambda c: "PASSED" if c else "FAILED"
    man = m["mastery"]["ORACLE"] <= C("x81b_oracle_gain")
    return [("P347", "Practice with the right memory helps (manipulation)", "PASSED" if man else "VOID", f"ORACLE error 10th/1st encounter {m['mastery']['ORACLE']:.3f} (bar <= {C('x81b_oracle_gain')})"),
            ("P348", "Few routines, fewer than the error rule", v(m["n_routines"]["SIG_CREATE"] <= C("x81b_max_routines") and S["sets_fewer"] >= C("x81_sets_better")), f"routines for 10 true problems: SIG_CREATE {m['n_routines']['SIG_CREATE']:.1f}, ERR_CREATE {m['n_routines']['ERR_CREATE']:.1f}; SIG fewer in {S['sets_fewer']} of 24 sets"),
            ("P349", "Reuses the right routine", v(m["own"]["SIG_CREATE"] >= C("x81b_reuse") and S["own_gain"] >= C("x81b_reuse_gain")), f"repeat episodes reusing own problem's routine: SIG {m['own']['SIG_CREATE']:.3f}, ERR {m['own']['ERR_CREATE']:.3f}; paired gain {S['own_gain']:.3f}; made a new routine on repeats: SIG {m['made']['SIG_CREATE']:.3f}, ERR {m['made']['ERR_CREATE']:.3f}"),
            ("P350", "Rarely reuses another problem's routine", v(m["mis"]["SIG_CREATE"] <= C("x81b_misreuse")), f"SIG {m['mis']['SIG_CREATE']:.3f}, ERR {m['mis']['ERR_CREATE']:.3f} (bar <= {C('x81b_misreuse')})"),
            ("P351", "Error near the oracle", v(m["err"]["SIG_CREATE"] <= C("x81b_err_ratio") * m["err"]["ORACLE"]), f"repeat error ticks 50-299: ORACLE {m['err']['ORACLE']:.3f}, SIG {m['err']['SIG_CREATE']:.3f} ({m['err']['SIG_CREATE'] / m['err']['ORACLE']:.2f}x), ERR {m['err']['ERR_CREATE']:.3f} ({m['err']['ERR_CREATE'] / m['err']['ORACLE']:.2f}x)"),
            ("P352", "Beats the error rule", v(S["sets_better"] >= C("x81_sets_better")), f"SIG_CREATE lower error in {S['sets_better']} of 24 sets")]


ARM_TEXT = {"SIG_CREATE": ("Recognise by input statistics, create when unfamiliar", "Starts with no routines; reuses the nearest known situation unless it is statistically unfamiliar; learns from audit verdicts."),
            "ERR_CREATE": ("Recognise by trying the routines (old rule)", "Reuses a routine only if it beats a fresh learner on the first 50 steps, else makes a new one."),
            "ORACLE": ("True identity", "Told the problem; ceiling.")}


def log_wandb(out):
    from track import Tracker
    S = out["summary"]; m = S["median"]
    tk = Tracker("X8.1b", "Can a mind build its own library of routines without fragmenting it?", "Starting with no routines at all, can the learner create one routine per recurring situation, reuse the right one, and avoid piling up near-duplicates (the earlier P5 failure)?",
                 "Ten different small causal worlds met ten times each in random order. No routines are given.", ARM_TEXT, "Routines = count at the end (10 would be perfect). Reuse = share of repeat encounters where the right routine was reused. Error: lower is better.", dict(sets=len(out["per_world"])), tags=["recognition", "creation"])
    tk.table("Summary by arm (median over world sets)", ["Arm", "Routines", "Reused right", "Reused wrong", "Made new on repeat", "Repeat error"], [[a, *[round(m[k].get(a, float("nan")), 3) for k in ("n_routines", "own", "mis", "made", "err")]] for a in ARMS])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    t0 = time.time()
    if mode == "selftest":
        r = one(70); S = summarize({0: r}); print(json.dumps(S["median"]), round(time.time() - t0), "s"); print("selftest ok"); return
    if mode == "log":
        log_wandb(json.load(open(os.path.join(OUT, "x8_1b_run.json")))["result"]); return
    from budget import pmap, workers
    n = workers(0.5); print("workers", n, flush=True); sets = list(range(C("x81_sets")))
    res = pmap(one, [(s,) for s in sets], n); per = dict(zip(sets, res)); S = summarize(per); out = {"per_world": per, "summary": S, "verdicts": judge(S)}
    json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x8_1b_run.json"), "w"), default=float)
    print(json.dumps(S["median"]))
    for v in out["verdicts"]: print(v[0], v[2], "|", v[3], flush=True)
    log_wandb(out)


if __name__ == "__main__":
    main(sys.argv[1])
