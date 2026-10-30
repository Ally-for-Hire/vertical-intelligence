"""X8.1: can audit verdicts teach the recogniser? P5 mastery world, 10 stored routines given, recognition from the first w ticks.
Registered in paper/PREDICTIONS.md (P341-P346) before this was run. Usage: python x8_1.py selftest | run | log   (writes ../results/x8_1_run.json)"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "experiments"), os.path.join(ROOT, "src"), HERE):
    sys.path.insert(0, p)
os.environ["CN_FULL"] = "1"
import numpy as np
from constants import C

OUT = os.path.join(ROOT, "results")
ARMS = ("ERR", "SIG_AUD", "SIG_TRUE", "SIG_SHUF", "ORACLE")


class Sig:
    """diagonal-Gaussian nearest class on the case signature; class means and one pooled within-class variance, learned online from labels"""
    def __init__(self, n_cls, dim):
        self.n = np.zeros(n_cls); self.mu = np.zeros((n_cls, dim)); self.ss = np.zeros(dim); self.tot = 0

    def update(self, z, c):
        self.n[c] += 1; d = z - self.mu[c]; self.mu[c] += d / self.n[c]; self.ss += d * (z - self.mu[c]); self.tot += 1

    def predict(self, z):
        ok = self.n >= 1
        if not ok.any():
            return None
        dof = max(self.tot - int(ok.sum()), 1); var = np.maximum(self.ss / dof, C("x81_var_floor")) if self.tot > int(ok.sum()) else np.ones_like(self.ss)
        dist = (((z - self.mu) ** 2) / var).sum(1); dist[~ok] = np.inf
        return int(np.argmin(dist))


def signature(X, w, gm, gs):
    """mean and sd of each input over the first w ticks, on the agent's own running global scale (never targets, never identity)"""
    return np.concatenate([(X[:w].mean(0) - gm) / gs, np.log((X[:w].std(0) + 1e-3) / gs)])


def one(wset):
    import pp5_mastery as P
    from nn import RunStd
    ws = tuple(C("x81_windows")); NP = P.NP; base = C("x81_seed0") + 100 * wset
    pseeds = [base + i for i in range(NP)]; refs = [P.ref_sd(ps) for ps in pseeds]
    experts = []
    for i, ps in enumerate(pseeds):
        ch = P.Chunk(18, 6, 1000 * wset + i)
        for ep in range(C("x81_train_eps")):
            X, D = P.episode_data(ps, 777000 + 100 * wset + 10 * i + ep)
            for t in range(P.EP):
                ch.step(X[t], D[t], refs[i])
        experts.append(ch)
    rng = np.random.default_rng(base); order = rng.permutation(np.repeat(np.arange(NP), P.REP)); seen = np.zeros(NP, int)
    G = RunStd(18); dim = 36
    cls = {(a, w): Sig(NP, dim) for a in ("SIG_AUD", "SIG_TRUE", "SIG_SHUF") for w in ws}; rows = []
    for e, pid in enumerate(order):
        k = int(seen[pid]); seen[pid] += 1
        X, D = P.episode_data(pseeds[pid], base * 1000 + e)
        sc = np.zeros((NP, P.EP)); sce = np.zeros((NP, P.EP))
        for j, c in enumerate(experts):
            for t in range(P.EP):
                sc[j, t], sce[j, t] = P.chunk_error_only(c, X[t], D[t], refs[pid])
        cs = np.cumsum(sc, 1); verdict = int(np.argmin(cs[:, -1]))
        gm = G.m.copy(); gs = np.sqrt(np.maximum(G.s2, 1e-6)); row = dict(pid=int(pid), k=k, verdict=verdict, pred={}, err={})
        for w in ws:
            err_pred = int(np.argmin(cs[:, w - 1])); z = signature(X, w, gm, gs) if G.n > 1 else np.zeros(dim)
            pred = {"ERR": err_pred, "ORACLE": int(pid)}
            for a in ("SIG_AUD", "SIG_TRUE", "SIG_SHUF"):
                r = cls[(a, w)].predict(z); pred[a] = err_pred if r is None else r
            row["pred"][w] = pred; row["err"][w] = {a: float(sce[pred[a], w:].mean()) for a in ARMS}
        row["cost_err_ticks"] = {w: 10 * w for w in ws}                          # forward passes of a stored routine needed by ERR before choosing (SIG needs none)
        rows.append(row)
        for w in ws:                                                              # labels arrive AFTER the episode: audit verdict (outcomes only), true id (ceiling), permuted id (control)
            z = signature(X, w, gm, gs) if G.n > 1 else np.zeros(dim)
            cls[("SIG_AUD", w)].update(z, verdict); cls[("SIG_TRUE", w)].update(z, int(pid)); cls[("SIG_SHUF", w)].update(z, int(rng.integers(NP)))
        for t in range(P.EP):
            G.update(X[t])
    return rows


def summarize(per):
    ws = tuple(C("x81_windows")); kmin = C("x81_min_enc"); S = {"acc": {}, "paired": {}, "sets_better": {}, "err": {}, "verdict": [], "gap": None}
    sets = sorted(per)
    for w in ws:
        acc = {a: [] for a in ARMS}; er = {a: [] for a in ARMS}
        for s in sets:
            R = [r for r in per[s] if r["k"] >= kmin]
            for a in ARMS:
                acc[a].append(np.mean([r["pred"][str(w) if str(w) in r["pred"] else w][a] == r["pid"] for r in R]))
                er[a].append(np.mean([r["err"][str(w) if str(w) in r["err"] else w][a] for r in R]))
        S["acc"][w] = {a: float(np.median(v)) for a, v in acc.items()}; S["err"][w] = {a: float(np.median(v)) for a, v in er.items()}
        d = np.array(acc["SIG_AUD"]) - np.array(acc["ERR"]); S["paired"][w] = float(np.median(d)); S["sets_better"][w] = int((d > 0).sum())
        S.setdefault("per_set_acc", {})[w] = {a: [float(x) for x in v] for a, v in acc.items()}
        S.setdefault("per_set_err", {})[w] = {a: [float(x) for x in v] for a, v in er.items()}
    S["verdict"] = float(np.median([np.mean([r["verdict"] == r["pid"] for r in per[s]]) for s in sets]))
    S["by_k"] = {w: {a: [float(np.mean([r["pred"][w][a] == r["pid"] for s in sets for r in per[s] if r["k"] == k])) for k in range(10)] for a in ("ERR", "SIG_AUD", "SIG_TRUE")} for w in ws}
    return S


def judge(S):
    w = 50; a = S["acc"][w]; e = S["err"][w]; d = S["paired"][w]
    ok = a["SIG_TRUE"] >= C("x81_acc_sig") and a["SIG_SHUF"] <= C("x81_acc_shuf"); v = lambda c: "PASSED" if c else "FAILED"
    gap = a["SIG_TRUE"] - a["SIG_AUD"]
    return [("P341", "Signature carries problem information and is not fitting noise", "PASSED" if ok else "FAILED", f"w=50 accuracy SIG_TRUE {a['SIG_TRUE']:.3f} (bar >= {C('x81_acc_sig')}), SIG_SHUF {a['SIG_SHUF']:.3f} (bar <= {C('x81_acc_shuf')}), chance 0.1"),
            ("P342", "Audit verdicts are good labels", v(S["verdict"] >= C("x81_verdict_bar")), f"verdict equals the true problem in {S['verdict']:.3f} of episodes (median over sets; bar {C('x81_verdict_bar')})"),
            ("P343", "Audit-trained signature beats the error-window rule", v(d >= C("x81_gain") and S["sets_better"][w] >= C("x81_sets_better")), f"w=50 accuracy SIG_AUD {a['SIG_AUD']:.3f} vs ERR {a['ERR']:.3f}; median paired gain {d:.3f}; better in {S['sets_better'][w]} of {C('x81_sets')} sets"),
            ("P344", "Label noise costs little", v(gap <= C("x81_label_tol")), f"SIG_TRUE {a['SIG_TRUE']:.3f} minus SIG_AUD {a['SIG_AUD']:.3f} = {gap:.3f} (bar <= {C('x81_label_tol')})"),
            ("P345", "Above the T13 recognition floor", v(a["SIG_AUD"] >= C("x81_floor")), f"SIG_AUD {a['SIG_AUD']:.3f} vs floor {C('x81_floor')}"),
            ("P346", "Recognition buys near-oracle error", v(e["SIG_AUD"] <= C("x81_err_ratio") * e["ORACLE"] and e["ERR"] > e["SIG_AUD"]), f"error ticks 50-299: ORACLE {e['ORACLE']:.3f}, SIG_AUD {e['SIG_AUD']:.3f} ({e['SIG_AUD'] / e['ORACLE']:.3f}x), ERR {e['ERR']:.3f} ({e['ERR'] / e['ORACLE']:.3f}x)")]


ARM_TEXT = {"ERR": ("Error-window rule", "Run all 10 stored routines on the first ticks; pick the one with the lowest error (the old P5 recogniser)."),
            "SIG_AUD": ("Learned from audit verdicts", "Recognise from the case's input statistics; learns only from which routine turned out best after each finished episode."),
            "SIG_TRUE": ("Learned from true labels", "Same recogniser told the true problem after each episode (ceiling)."),
            "SIG_SHUF": ("Learned from scrambled labels", "Same recogniser given random labels (control: must stay near chance)."),
            "ORACLE": ("True identity", "Told the problem (ceiling for what recognition can buy).")}


def log_wandb(out):
    from track import Tracker
    S = out["summary"]; ws = tuple(C("x81_windows")); g = lambda d, w: d[w] if w in d else d[str(w)]
    tk = Tracker("X8.1", "Can a mind learn to recognise a situation from its own after-the-fact audits?", "Ten practised routines are stored. A case arrives; which routine fits? Does a recogniser that learns only from after-the-fact audit verdicts beat running every routine for a while and picking the best (the old P5 failure)?",
                 "Ten different small causal worlds, each met ten times in random order (300 steps each). Ten stored routines, one per world, are given. Nobody tells the recogniser which world it is in.", ARM_TEXT, "Accuracy = share of 3rd-and-later encounters where the chosen routine is the right one (higher is better). Error = prediction error with the chosen routine (lower is better).", dict(sets=len(out["per_world"])), tags=["recognition", "automaticity"])
    tk.table("Recognition accuracy by window (median over world sets)", ["Window (ticks)"] + list(ARM_TEXT), [[w, *[round(g(S["acc"], w)[a], 3) for a in ARM_TEXT]] for w in ws])
    tk.table("Error after recognition (median over world sets)", ["Window (ticks)"] + list(ARM_TEXT), [[w, *[round(g(S["err"], w)[a], 3) for a in ARM_TEXT]] for w in ws])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    t0 = time.time()
    if mode == "selftest":
        r = one(C("x81_sets") + 5000 // 100)                                 # an unscored world set, never in the scored block
        S = summarize({0: r}); print(json.dumps({"acc50": S["acc"][50], "verdict": S["verdict"]}), round(time.time() - t0), "s"); print("selftest ok"); return
    if mode == "log":
        log_wandb(json.load(open(os.path.join(OUT, "x8_1_run.json")))["result"]); return
    from budget import pmap, workers
    n = workers(0.5); print("workers", n, flush=True); sets = list(range(C("x81_sets")))
    res = pmap(one, [(s,) for s in sets], n); per = dict(zip(sets, res)); S = summarize(per); out = {"per_world": per, "summary": S, "verdicts": judge(S)}
    os.makedirs(OUT, exist_ok=True); json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x8_1_run.json"), "w"), default=float)
    for w in C("x81_windows"): print("w", w, {a: round(S["acc"][w][a], 3) for a in ARMS}, {a: round(S["err"][w][a], 3) for a in ARMS})
    for v in out["verdicts"]: print(v[0], v[2], "|", v[3], flush=True)
    log_wandb(out)


if __name__ == "__main__":
    main(sys.argv[1])
