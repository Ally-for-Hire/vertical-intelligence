"""X7.7: unanswerable situations that LOOK answerable. Look-alike contexts copy an answerable context's geometry plus a shared 'tell'; their label is the true class with probability s, else random.
Registered in paper/PREDICTIONS.md (P364-P369) before this was run. Usage: python x7_7.py selftest | run | log   (writes ../results/x7_7_run.json)"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x0_1 as X
from x7_1 import Mirror
from x7_2 import make_sensor
from x7_3 import FAMS, train_sensor, raw_net, util_loss, realised
from x7_4 import World
from x7_5 import SetSelfR

OUT = X.OUT


class World7(World):
    """x7_4.World plus look-alike sampling: geometry of answerable context `base`, a shared tell offset, label = true class w.p. s else uniform random."""
    def __init__(s, seed):
        super().__init__(seed); u = np.random.default_rng(seed + 777).normal(size=C("x71_in_dim")); s.tell = u / np.linalg.norm(u)

    def sample_la(s, rng, n, bases, level, cue):
        c = rng.choice(bases, n); t = rng.integers(0, C("x71_regimes"), n)
        x = np.einsum("nij,nj->ni", np.stack([s.Q[k] for k in c]), s.protos[t] + C("x72_class_sd") * rng.normal(size=(n, C("x71_in_dim")))) + cue * C("x72_class_sd") * s.tell
        lv = np.broadcast_to(np.asarray(level, float), (n,)); y = np.where(rng.random(n) < lv, t, rng.integers(0, C("x71_regimes"), n))
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y), torch.tensor(t), torch.tensor(lv, dtype=torch.float32)


def fit7(model, fwd, y, rp, seed):
    """x7_6.fit6 generalised: a case's label is redrawn uniformly at random with probability rp[case] every time it is drawn (rp = 0 for plain, 1 for flagged, 1-s for graded)."""
    torch.manual_seed(seed); opt = torch.optim.Adam(model.parameters(), lr=C("x71_lr")); rng = np.random.default_rng(seed)
    for _ in range(C("x73_self_iters")):
        i = torch.tensor(rng.integers(0, len(y), C("x71_host_batch"))); yb = y[i].clone()
        m = torch.tensor(rng.random(len(i))) < rp[i]; yb[m] = torch.tensor(rng.integers(0, C("x71_regimes"), int(m.sum())))
        loss = util_loss(fwd(i), yb); opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def one(seed):
    torch.set_num_threads(1); NR, NJ = C("x73_real"), C("x73_junk"); K = NR + NJ; W = World7(seed); rs = lambda o: np.random.default_rng(seed * 10 + o); LV = C("x77_levels"); TB = list(C("x77_train_bases"))
    sens, mirs, cov = [], [], []
    for j in range(K):
        cj = [j % 4, (j + 1) % 4]; m = train_sensor(make_sensor(FAMS[j % 4], seed + j), W, cj, seed * 100 + j, j >= NR); sens.append(m); mirs.append(Mirror(m, seed + j)); cov.append(cj)

    def read(x):
        with torch.no_grad(): return [mirs[j].read(x) for j in range(K)], torch.stack([F.softmax(sens[j](x), -1) for j in range(K)], 1)
    seen = W.sample(rs(2), C("x73_test_seen"), [0, 1, 2, 3]); TSseen = read(seen[0]); out = {"util": {}, "abstain": {}, "cue_curve": {}, "manip": {}}
    ys = seen[1]; cs = seen[2]; ok = [(TSseen[1][:, j].argmax(1) == ys).numpy() for j in range(K)]
    out["manip"]["sensor_in"] = float(np.mean([ok[j][np.isin(cs, cov[j])].mean() for j in range(NR)])); out["manip"]["sensor_out"] = float(np.mean([ok[j][~np.isin(cs, cov[j])].mean() for j in range(NR)])); out["manip"]["junk_acc"] = float(np.mean([ok[j].mean() for j in range(NR, K)]))
    out["manip"]["conf_seen"] = float(np.mean([TSseen[1][:, j].max(1).values.mean() for j in range(NR)]))
    for cue in C("x77_cues"):
        ck = f"cue{cue:g}"; U, A = out["util"].setdefault(ck, {}), out["abstain"].setdefault(ck, {})
        tests = {f"s{lv:g}": W.sample_la(rs(300 + int(10 * lv)), C("x73_test_other"), [3], lv, cue)[:2] for lv in LV}; TS = {k: read(v[0]) for k, v in tests.items()}; TS["seen"] = TSseen; tests["seen"] = seen[:2]
        if cue == C("x77_cues")[-1]:
            out["manip"]["la_acc"] = float(np.mean([(TS["s0"][1][:, j].argmax(1) == tests["s0"][1]).float().mean() for j in range(NR)])); out["manip"]["conf_la"] = float(np.mean([TS["s0"][1][:, j].max(1).values.mean() for j in range(NR)]))

        def sc(name, k, o):
            u, a = realised(o, tests[k][1] if k != "train" else ytr); U.setdefault(name, {})[k] = u; A.setdefault(name, {})[k] = a

        def build(N, graded):
            nl = N // 2
            if graded:
                lv = np.repeat(np.asarray(C("x77_graded_levels"), float), int(np.ceil(nl / len(C("x77_graded_levels")))))[:nl]
            else: lv = np.zeros(nl)
            xa, ya, _ = W.sample(rs(100 + N + graded), N - nl, [0, 1, 2, 3]); xl, yl, tl, _ = W.sample_la(rs(200 + N + graded), nl, TB, lv, cue)
            x = torch.cat([xa, xl]); y = torch.cat([ya, yl]); t = torch.cat([ya, tl]); level = torch.cat([torch.ones(N - nl), torch.tensor(lv, dtype=torch.float32)]); is_la = torch.cat([torch.zeros(N - nl), torch.ones(nl)]) > 0
            return x, y, t, level, is_la
        # arms: (name, insides?, N, graded, refresh mode)
        arms = [("SET", True, 300, False, "none"), ("ANS", False, 300, False, "none"), ("REFRESH_SET", True, 300, False, "flag"), ("REFRESH_ANS", False, 300, False, "flag"),
                ("REFRESH_GRADED", True, 300, True, "graded"), ("REFRESH_SET_1500", True, C("x77_big_n"), False, "flag")]
        for name, state, N, graded, mode in arms:
            x, y, t, level, is_la = build(N, graded); ytr = y; S, P = read(x); mu = [s_.mean(0) for s_ in S]; sd = [s_.std(0) + 1e-6 for s_ in S]
            feat = lambda S_, P_, j: torch.cat([P_[:, j], (S_[j] - mu[j]) / sd[j]], 1) if state else P_[:, j]
            rp = torch.zeros(N) if mode == "none" else (is_la.float() * 1.0 if mode == "flag" else is_la.float() * (1 - level))
            m = SetSelfR([feat(S, P, j).shape[1] for j in range(K)]); fit7(m, lambda i: m([feat(S, P, j)[i] for j in range(K)], P[i]), (t if mode == "graded" else y), rp, seed)
            with torch.no_grad():
                for k in TS: sc(name, k, m([feat(TS[k][0], TS[k][1], j) for j in range(K)], TS[k][1]))
                if cue == C("x77_cues")[-1] and name == "SET": sc(name, "train", m([feat(S, P, j) for j in range(K)], P))
        x, y, t, level, is_la = build(300, False); ytr = y; rp = is_la.float(); m = raw_net(); fit7(m, lambda i: m(x[i]), y, rp, seed)
        with torch.no_grad():
            for k in tests: sc("RAW_REFRESH", k, m(tests[k][0]))
    return out


def summarize(per):
    seeds = sorted(per); g = lambda f: np.array([f(per[s]) for s in seeds], float); a0 = per[seeds[0]]; ci = lambda d: X.boot_ci(d, C("x77_seed0"))
    S = {"median_util": {}, "median_abstain": {}, "paired": {}, "manip": {}}
    for ck in a0["util"]:
        S["median_util"][ck] = {a: {k: float(np.median(g(lambda r: r["util"][ck][a][k]))) for k in a0["util"][ck][a]} for a in a0["util"][ck]}
        S["median_abstain"][ck] = {a: {k: float(np.median(g(lambda r: r["abstain"][ck][a][k]))) for k in a0["abstain"][ck][a]} for a in a0["abstain"][ck]}
    U = lambda ck, a, k: g(lambda r: r["util"][ck][a][k]); A = lambda ck, a, k: g(lambda r: r["abstain"][ck][a][k]); c3 = f"cue{C('x77_cues')[-1]:g}"
    S["paired"]["gap SET-ANS (REFRESH, cue 3)"] = ci((A(c3, "REFRESH_SET", "s0") - A(c3, "REFRESH_SET", "seen")) - (A(c3, "REFRESH_ANS", "s0") - A(c3, "REFRESH_ANS", "seen")))
    S["paired"]["util s0.75 GRADED-REFRESH_SET (cue 3)"] = ci(U(c3, "REFRESH_GRADED", "s0.75") - U(c3, "REFRESH_SET", "s0.75"))
    S["paired"]["util seen 1500-300 (cue 3)"] = ci(U(c3, "REFRESH_SET_1500", "seen") - U(c3, "REFRESH_SET", "seen"))
    S["manip"] = {k: float(np.median(g(lambda r: r["manip"][k]))) for k in a0["manip"]}; S["manip"]["train_util_SET"] = S["median_util"][c3]["SET"]["train"]
    return S


def judge(S):
    p, M, mu, ab = S["paired"], S["manip"], S["median_util"], S["median_abstain"]; c3, c0 = f"cue{C('x77_cues')[-1]:g}", f"cue{C('x77_cues')[0]:g}"; r = lambda k: [round(x, 3) for x in p[k]]; a3 = ab[c3]; tol = C("x77_ord_tol")
    manip = M["sensor_out"] <= C("x72_unseen_ratio") * M["sensor_in"] and M["junk_acc"] <= C("x73_junk_acc") and M["la_acc"] <= C("x74_chance_acc") and M["conf_la"] >= C("x77_conf_ratio") * M["conf_seen"] and M["train_util_SET"] >= C("x75_mem_bar")
    gap0 = abs(ab[c0]["REFRESH_SET"]["s0"] - ab[c0]["REFRESH_SET"]["seen"]); curve = [a3["REFRESH_GRADED"][f"s{lv:g}"] for lv in C("x77_levels")]
    mono = all(curve[i + 1] <= curve[i] + tol for i in range(len(curve) - 1))
    return [("P364", "Manipulation: sensors as before, look-alike looks answerable, plain SET memorises", "PASSED" if manip else "VOID", f"inside/outside {M['sensor_in']:.2f}/{M['sensor_out']:.2f}; junk {M['junk_acc']:.2f}; sensors on s=0 look-alike accuracy {M['la_acc']:.2f}, confidence {M['conf_la']:.2f} vs {M['conf_seen']:.2f} on answerable; SET train utility {M['train_util_SET']:.3f}"),
            ("P365", "Negative control: at cue 0 nothing separates the cases", "PASSED" if gap0 <= C("x77_null_tol") else "FAILED", f"REFRESH_SET abstain s=0 {ab[c0]['REFRESH_SET']['s0']:.2f} vs answerable {ab[c0]['REFRESH_SET']['seen']:.2f}; gap {gap0:.2f}"),
            ("P366", "A tell in the insides lets the Self doubt look-alikes", "PASSED" if a3["REFRESH_SET"]["s0"] >= C("x75_abstain_bar") and a3["REFRESH_SET"]["seen"] <= C("x76_seen_abstain_max") and p["gap SET-ANS (REFRESH, cue 3)"][1] > 0 else "FAILED", f"REFRESH_SET abstain s=0 {a3['REFRESH_SET']['s0']:.2f}, answerable {a3['REFRESH_SET']['seen']:.2f}; REFRESH_ANS {a3['REFRESH_ANS']['s0']:.2f}/{a3['REFRESH_ANS']['seen']:.2f}; gap difference {r('gap SET-ANS (REFRESH, cue 3)')}; RAW_REFRESH {a3['RAW_REFRESH']['s0']:.2f}/{a3['RAW_REFRESH']['seen']:.2f}"),
            ("P367", "Trained on s=0 only the Self learns 'tell means coin flip', not a grade", "PASSED" if a3["REFRESH_SET"]["s0.75"] >= C("x75_abstain_bar") and p["util s0.75 GRADED-REFRESH_SET (cue 3)"][1] > 0 else "FAILED", f"REFRESH_SET abstain at s=0.75 {a3['REFRESH_SET']['s0.75']:.2f}, utility {mu[c3]['REFRESH_SET']['s0.75']:.3f}; GRADED utility {mu[c3]['REFRESH_GRADED']['s0.75']:.3f}; difference {r('util s0.75 GRADED-REFRESH_SET (cue 3)')}"),
            ("P368", "Graded refresh gives a graded doubt curve", "PASSED" if mono and curve[0] >= C("x75_abstain_bar") and curve[-1] <= C("x76_seen_abstain_max") and a3["REFRESH_GRADED"]["seen"] <= C("x76_seen_abstain_max") else "FAILED", "abstain over s=" + ",".join(f"{lv:g}" for lv in C("x77_levels")) + " = " + ", ".join(f"{c:.2f}" for c in curve) + f"; answerable {a3['REFRESH_GRADED']['seen']:.2f}"),
            ("P369", "Refresh plus more labels recovers competence and keeps doubt", "PASSED" if p["util seen 1500-300 (cue 3)"][1] > 0 and a3["REFRESH_SET_1500"]["s0"] >= C("x75_abstain_bar") else "FAILED", f"seen utility 1500 {mu[c3]['REFRESH_SET_1500']['seen']:.3f} vs 300 {mu[c3]['REFRESH_SET']['seen']:.3f}; difference {r('util seen 1500-300 (cue 3)')}; abstain s=0 at 1500 {a3['REFRESH_SET_1500']['s0']:.2f}")]


ARM_TEXT = {"SET": ("Self: answers + insides, plain", "Practice cases as given, no refresh."), "ANS": ("Self: answers only, plain", ""), "REFRESH_SET": ("Self: answers + insides, fresh coin flips", "Look-alike practice cases get a new random label every time they are seen."),
            "REFRESH_ANS": ("Self: answers only, fresh coin flips", "Same trick, answers only."), "REFRESH_GRADED": ("Self: answers + insides, graded refresh", "Look-alike practice cases at signal 0, 0.5, 1; a case's label is redrawn with probability 1 - signal."),
            "REFRESH_SET_1500": ("Self: answers + insides, fresh coin flips, 1500 cases", "Five times more practice."), "RAW_REFRESH": ("Raw input only, fresh coin flips", "Control that sees the tell directly.")}


def log_wandb(out):
    from track import Tracker
    S, per = out["summary"], out["per_world"]
    tk = Tracker("X7.7", "Can a Self doubt situations that look answerable?", "Some situations look just like ones the advisors know, but the correct answer is partly or entirely luck. Can a Self learn to say 'I don't know' for them, in proportion to how lucky they are, without losing competence?",
                 "Twelve advisor networks (four junk). Look-alike situations carry a small 'tell' in the input; their answer is right with probability s (0 = pure luck, 1 = fully answerable). +1 right, -2 wrong, 0 abstain.", ARM_TEXT, "Utility per case, higher is better; abstain rate is the share of 'I don't know'.", dict(worlds=len(per)), tags=["self", "doubt", "look-alike"])
    for ck in S["median_util"]:
        ks = list(S["median_util"][ck]["SET"]); tk.table(f"Utility and abstain by arm, tell size {ck} (median over worlds)", ["Arm"] + [f"Utility {k}" for k in ks] + [f"Abstain {k}" for k in ks],
                                                         [[a, *[round(S["median_util"][ck][a].get(k, float("nan")), 3) for k in ks], *[round(S["median_abstain"][ck][a].get(k, float("nan")), 3) for k in ks]] for a in S["median_util"][ck]])
    tk.table("Paired differences with 95% ranges", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    t0 = time.time()
    if mode == "selftest":
        r = one(C("x71_seed0")); c3 = f"cue{C('x77_cues')[-1]:g}"
        for ck in r["abstain"]: print(ck, json.dumps({a: {k: round(v, 2) for k, v in r["abstain"][ck][a].items()} for a in r["abstain"][ck]}))
        print(json.dumps({k: round(v, 2) for k, v in r["manip"].items()})); print("selftest ok", round(time.time() - t0)); return
    if mode == "log": log_wandb(json.load(open(os.path.join(OUT, "x7_7_run.json")))["result"]); return
    from budget import pmap, workers
    seeds = [C("x77_seed0") + i for i in range(C("x77_worlds"))]; n = min(workers(1.5), C("x75_workers_cap")); print("workers", n, flush=True); res = pmap(one, [(s,) for s in seeds], n); per = dict(zip(seeds, res))
    S = summarize(per); out = {"per_world": per, "summary": S, "verdicts": judge(S)}
    os.makedirs(OUT, exist_ok=True); json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x7_7_run.json"), "w"), indent=1, default=float)
    for v in out["verdicts"]: print(v[0], v[2], "|", v[3], flush=True)
    log_wandb(out)


if __name__ == "__main__":
    main(sys.argv[1])
