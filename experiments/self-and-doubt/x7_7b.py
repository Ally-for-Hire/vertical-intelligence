"""X7.7b: look-alikes whose signal level is READABLE (tell size = cue x (1-s)) with a matched answerable control (context 3, no tell).
Registered in paper/PREDICTIONS.md (P370-P375) before this was run. Usage: python x7_7b.py selftest | run | log   (writes ../results/x7_7b_run.json)"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x0_1 as X
from x7_1 import Mirror
from x7_2 import make_sensor
from x7_3 import FAMS, train_sensor, raw_net, realised
from x7_5 import SetSelfR
from x7_7 import World7, fit7

OUT = X.OUT


class World7b(World7):
    def sample_lb(s, rng, n, bases, level, cue_max):
        """Look-alike: label = true class w.p. level else random; tell size = cue_max x (1 - level) per case."""
        c = rng.choice(bases, n); t = rng.integers(0, C("x71_regimes"), n); lv = np.broadcast_to(np.asarray(level, float), (n,))
        x = np.einsum("nij,nj->ni", np.stack([s.Q[k] for k in c]), s.protos[t] + C("x72_class_sd") * rng.normal(size=(n, C("x71_in_dim")))) + (cue_max * (1 - lv) * C("x72_class_sd"))[:, None] * s.tell[None]
        y = np.where(rng.random(n) < lv, t, rng.integers(0, C("x71_regimes"), n))
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y), torch.tensor(t), torch.tensor(lv, dtype=torch.float32)


def one(seed):
    torch.set_num_threads(1); NR, NJ = C("x73_real"), C("x73_junk"); K = NR + NJ; W = World7b(seed); rs = lambda o: np.random.default_rng(seed * 10 + o); LV = C("x77_levels"); TB = list(C("x77_train_bases")); GL = np.asarray(C("x77_graded_levels"), float)
    sens, mirs, cov = [], [], []
    for j in range(K):
        cj = [j % 4, (j + 1) % 4]; m = train_sensor(make_sensor(FAMS[j % 4], seed + j), W, cj, seed * 100 + j, j >= NR); sens.append(m); mirs.append(Mirror(m, seed + j)); cov.append(cj)

    def read(x):
        with torch.no_grad(): return [mirs[j].read(x) for j in range(K)], torch.stack([F.softmax(sens[j](x), -1) for j in range(K)], 1)
    out = {"util": {}, "abstain": {}, "manip": {}}; seen = W.sample(rs(2), C("x73_test_seen"), [0, 1, 2, 3]); TSs = read(seen[0]); ys, cs = seen[1], seen[2]
    ok = [(TSs[1][:, j].argmax(1) == ys).numpy() for j in range(K)]
    out["manip"]["sensor_in"] = float(np.mean([ok[j][np.isin(cs, cov[j])].mean() for j in range(NR)])); out["manip"]["sensor_out"] = float(np.mean([ok[j][~np.isin(cs, cov[j])].mean() for j in range(NR)])); out["manip"]["junk_acc"] = float(np.mean([ok[j].mean() for j in range(NR, K)]))
    for cue in C("x77b_cues"):
        ck = f"cue{cue:g}"; U, A = out["util"].setdefault(ck, {}), out["abstain"].setdefault(ck, {})
        tests = {f"s{lv:g}": W.sample_lb(rs(300 + int(10 * lv)), C("x73_test_other"), [3], lv, cue)[:2] for lv in LV}; tests["m3"] = W.sample(rs(9), C("x73_test_other"), [3])[:2]; tests["seen"] = seen[:2]
        TS = {k: read(v[0]) for k, v in tests.items()}
        if cue == C("x77b_cues")[-1]: out["manip"]["la_acc"] = float(np.mean([(TS["s0"][1][:, j].argmax(1) == tests["s0"][1]).float().mean() for j in range(NR)]))

        def sc(name, k, o):
            u, a = realised(o, tests[k][1]); U.setdefault(name, {})[k] = u; A.setdefault(name, {})[k] = a

        def build(graded):
            N = C("x73_self_n"); nl = N // 2; lv = np.repeat(GL, int(np.ceil(nl / len(GL))))[:nl] if graded else np.zeros(nl)
            xa, ya, _ = W.sample(rs(100 + graded), N - nl, [0, 1, 2, 3]); xl, yl, tl, _ = W.sample_lb(rs(200 + graded), nl, TB, lv, cue)
            return torch.cat([xa, xl]), torch.cat([ya, yl]), torch.cat([ya, tl]), torch.cat([torch.ones(N - nl), torch.tensor(lv, dtype=torch.float32)]), torch.cat([torch.zeros(N - nl), torch.ones(nl)]) > 0
        for name, state, graded in (("REFRESH_SET", True, False), ("REFRESH_GRADED", True, True), ("ANS_GRADED", False, True)):
            x, y, t, level, is_la = build(graded); S, P = read(x); mu = [s_.mean(0) for s_ in S]; sd = [s_.std(0) + 1e-6 for s_ in S]
            feat = lambda S_, P_, j: torch.cat([P_[:, j], (S_[j] - mu[j]) / sd[j]], 1) if state else P_[:, j]
            rp = is_la.float() * ((1 - level) if graded else 1.0); base = t if graded else y
            m = SetSelfR([feat(S, P, j).shape[1] for j in range(K)]); fit7(m, lambda i: m([feat(S, P, j)[i] for j in range(K)], P[i]), base, rp, seed)
            with torch.no_grad():
                for k in TS: sc(name, k, m([feat(TS[k][0], TS[k][1], j) for j in range(K)], TS[k][1]))
        x, y, t, level, is_la = build(True); m = raw_net(); fit7(m, lambda i: m(x[i]), t, is_la.float() * (1 - level), seed)
        with torch.no_grad():
            for k in tests: sc("RAW_GRADED", k, m(tests[k][0]))
        for name in U:
            U[name]["mean_all"] = float(np.mean([U[name][f"s{lv:g}"] for lv in LV] + [U[name]["m3"]])); U[name]["mean_mid"] = float(np.mean([U[name][f"s{lv:g}"] for lv in C("x77b_mid_levels")]))
    return out


def summarize(per):
    seeds = sorted(per); g = lambda f: np.array([f(per[s]) for s in seeds], float); a0 = per[seeds[0]]; ci = lambda d: X.boot_ci(d, C("x77b_seed0"))
    S = {"median_util": {}, "median_abstain": {}, "paired": {}, "manip": {}}
    for ck in a0["util"]:
        S["median_util"][ck] = {a: {k: float(np.median(g(lambda r: r["util"][ck][a][k]))) for k in a0["util"][ck][a]} for a in a0["util"][ck]}
        S["median_abstain"][ck] = {a: {k: float(np.median(g(lambda r: r["abstain"][ck][a][k]))) for k in a0["abstain"][ck][a]} for a in a0["abstain"][ck]}
    U = lambda ck, a, k: g(lambda r: r["util"][ck][a][k]); c3 = f"cue{C('x77b_cues')[-1]:g}"
    S["paired"]["mid util GRADED-REFRESH_SET"] = ci(U(c3, "REFRESH_GRADED", "mean_mid") - U(c3, "REFRESH_SET", "mean_mid")); S["paired"]["mean_all GRADED minus 0"] = ci(U(c3, "REFRESH_GRADED", "mean_all"))
    S["paired"]["mean_all GRADED-ANS_GRADED"] = ci(U(c3, "REFRESH_GRADED", "mean_all") - U(c3, "ANS_GRADED", "mean_all"))
    S["manip"] = {k: float(np.median(g(lambda r: r["manip"][k]))) for k in a0["manip"]}
    return S


def judge(S):
    p, M, mu, ab = S["paired"], S["manip"], S["median_util"], S["median_abstain"]; c3, c0 = f"cue{C('x77b_cues')[-1]:g}", f"cue{C('x77b_cues')[0]:g}"; r = lambda k: [round(x, 3) for x in p[k]]; a3 = ab[c3]; tol = C("x77_ord_tol")
    gap0 = abs(ab[c0]["REFRESH_SET"]["s0"] - ab[c0]["REFRESH_SET"]["m3"]); ctl = gap0 <= C("x77_null_tol") and M["sensor_out"] <= C("x72_unseen_ratio") * M["sensor_in"] and M["junk_acc"] <= C("x73_junk_acc") and M["la_acc"] <= C("x74_chance_acc")
    curve = [a3["REFRESH_GRADED"][f"s{lv:g}"] for lv in C("x77_levels")]; mono = all(curve[i + 1] <= curve[i] + tol for i in range(len(curve) - 1)); V = lambda ok: ("PASSED" if ok else "FAILED") if ctl else "VOID"
    return [("P370", "Matched control", "PASSED" if ctl else "FAILED", f"cue 0: REFRESH_SET abstain s=0 {ab[c0]['REFRESH_SET']['s0']:.2f} vs matched answerable {ab[c0]['REFRESH_SET']['m3']:.2f}, gap {gap0:.2f}; sensors inside/outside {M['sensor_in']:.2f}/{M['sensor_out']:.2f}, junk {M['junk_acc']:.2f}, look-alike accuracy {M['la_acc']:.2f}"),
            ("P371", "Tell-trained-at-s0 Self doubts the look-alike", V(a3["REFRESH_SET"]["s0"] >= C("x75_abstain_bar") and a3["REFRESH_SET"]["m3"] <= C("x76_seen_abstain_max")), f"REFRESH_SET abstain s=0 {a3['REFRESH_SET']['s0']:.2f}, matched answerable {a3['REFRESH_SET']['m3']:.2f}"),
            ("P372", "Graded refresh gives a graded doubt curve", V(mono and curve[0] >= C("x75_abstain_bar") and curve[-1] <= C("x76_seen_abstain_max") and a3["REFRESH_GRADED"]["m3"] <= C("x76_seen_abstain_max")), "abstain over s=" + ",".join(f"{lv:g}" for lv in C("x77_levels")) + " = " + ", ".join(f"{c:.2f}" for c in curve) + f"; matched answerable {a3['REFRESH_GRADED']['m3']:.2f}"),
            ("P373", "Graded beats s0-only on held levels", V(p["mid util GRADED-REFRESH_SET"][1] > 0), f"mean utility at s=.25,.5,.75: GRADED {mu[c3]['REFRESH_GRADED']['mean_mid']:.3f} vs REFRESH_SET {mu[c3]['REFRESH_SET']['mean_mid']:.3f}; difference {r('mid util GRADED-REFRESH_SET')}"),
            ("P374", "Skeptic bar: better than always abstaining", V(p["mean_all GRADED minus 0"][1] > 0), f"GRADED mean utility over five levels + matched answerable {mu[c3]['REFRESH_GRADED']['mean_all']:.3f}; range {r('mean_all GRADED minus 0')}"),
            ("P375", "The insides add value", V(p["mean_all GRADED-ANS_GRADED"][1] > 0), f"GRADED {mu[c3]['REFRESH_GRADED']['mean_all']:.3f} vs ANS_GRADED {mu[c3]['ANS_GRADED']['mean_all']:.3f}; RAW_GRADED {mu[c3]['RAW_GRADED']['mean_all']:.3f}; difference {r('mean_all GRADED-ANS_GRADED')}")]


ARM_TEXT = {"REFRESH_SET": ("Self: insides, practice luck cases only", "Look-alike practice cases all pure luck, label redrawn each time."), "REFRESH_GRADED": ("Self: insides, graded practice", "Practice look-alikes at signal 0, 0.5, 1; label redrawn with probability 1 - signal."),
            "ANS_GRADED": ("Self: answers only, graded practice", "Same, without the advisors' insides."), "RAW_GRADED": ("Raw input only, graded practice", "Control that sees the tell directly.")}


def log_wandb(out):
    from track import Tracker
    S, per = out["summary"], out["per_world"]
    tk = Tracker("X7.7b", "Can a Self learn graded doubt when the grade is readable?", "Look-alike situations carry a tell whose size shows how lucky they are. Can a Self say 'I don't know' in proportion, beat always-abstaining, and does a matched control confirm the setup is fair?",
                 "Twelve advisor networks (four junk). +1 right, -2 wrong, 0 abstain; always abstaining scores 0.", ARM_TEXT, "Utility per case, higher is better; abstain rate is the share of 'I don't know'.", dict(worlds=len(per)), tags=["self", "doubt", "graded"])
    for ck in S["median_util"]:
        ks = list(S["median_util"][ck]["REFRESH_SET"]); tk.table(f"Utility and abstain, tell size {ck} (median over worlds)", ["Arm"] + [f"Utility {k}" for k in ks] + [f"Abstain {k}" for k in ks if k not in ("mean_all", "mean_mid")],
                                                                 [[a, *[round(S["median_util"][ck][a].get(k, float("nan")), 3) for k in ks], *[round(S["median_abstain"][ck][a].get(k, float("nan")), 3) for k in ks if k not in ("mean_all", "mean_mid")]] for a in S["median_util"][ck]])
    tk.table("Paired differences with 95% ranges", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    t0 = time.time()
    if mode == "selftest":
        r = one(C("x71_seed0"))
        for ck in r["abstain"]: print(ck, json.dumps({a: {k: round(v, 2) for k, v in r["abstain"][ck][a].items()} for a in r["abstain"][ck]}))
        print(json.dumps({k: round(v, 2) for k, v in r["manip"].items()})); print("selftest ok", round(time.time() - t0)); return
    if mode == "log": log_wandb(json.load(open(os.path.join(OUT, "x7_7b_run.json")))["result"]); return
    from budget import pmap, workers
    seeds = [C("x77b_seed0") + i for i in range(C("x77_worlds"))]; n = min(workers(1.5), C("x75_workers_cap")); print("workers", n, flush=True); res = pmap(one, [(s,) for s in seeds], n); per = dict(zip(seeds, res))
    S = summarize(per); out = {"per_world": per, "summary": S, "verdicts": judge(S)}
    os.makedirs(OUT, exist_ok=True); json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x7_7b_run.json"), "w"), indent=1, default=float)
    for v in out["verdicts"]: print(v[0], v[2], "|", v[3], flush=True)
    log_wandb(out)


if __name__ == "__main__":
    main(sys.argv[1])
