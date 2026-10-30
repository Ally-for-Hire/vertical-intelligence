"""X7.6: doubt that cannot be memorised. Same as X7.5, but every time a practice case KNOWN to be unanswerable is drawn into a training batch its label is redrawn at random.
Registered in paper/PREDICTIONS.md (P326-P331) before this was run. Usage: python x7_6.py selftest | run | log   (writes ../results/x7_6_run.json)"""
import sys, os, json, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x0_1 as X
from x7_1 import Mirror
from x7_2 import make_sensor
from x7_3 import FAMS, train_sensor, raw_net, util_loss, realised
from x7_4 import World, UNANS_TRAIN, UNANS_HELD, NOVEL
from x7_5 import SetSelfR

OUT = X.OUT
SETS = ("seen", "unans", "novel")


def fit6(model, fwd, y, flag, seed, refresh):
    torch.manual_seed(seed); opt = torch.optim.Adam(model.parameters(), lr=C("x71_lr")); rng = np.random.default_rng(seed)
    for _ in range(C("x73_self_iters")):
        i = torch.tensor(rng.integers(0, len(y), C("x71_host_batch"))); yb = y[i].clone()
        if refresh: yb[flag[i]] = torch.tensor(rng.integers(0, C("x71_regimes"), int(flag[i].sum())))              # a flagged case gets a fresh coin flip every time it is seen
        loss = util_loss(fwd(i), yb); opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def one(seed):
    torch.set_num_threads(1); NR, NJ = C("x73_real"), C("x73_junk"); K = NR + NJ; W = World(seed); rs = lambda o: np.random.default_rng(seed * 10 + o)
    sens, mirs, cov = [], [], []
    for j in range(K):
        cj = [j % 4, (j + 1) % 4]; m = train_sensor(make_sensor(FAMS[j % 4], seed + j), W, cj, seed * 100 + j, j >= NR); sens.append(m); mirs.append(Mirror(m, seed + j)); cov.append(cj)

    def read(x):
        with torch.no_grad(): return [mirs[j].read(x) for j in range(K)], torch.stack([F.softmax(sens[j](x), -1) for j in range(K)], 1)
    tests = {"seen": W.sample(rs(2), C("x73_test_seen"), [0, 1, 2, 3]), "unans": W.sample(rs(3), C("x73_test_other"), UNANS_HELD), "novel": W.sample(rs(4), C("x73_test_other"), NOVEL)}
    TS = {k: read(v[0]) for k, v in tests.items()}; out = {"util": {}, "abstain": {}}

    def sc(name, k, o, y):
        u, a = realised(o, y); out["util"].setdefault(name, {})[k] = u; out["abstain"].setdefault(name, {})[k] = a
    for idx, N in enumerate((C("x73_self_n"), C("x76_big_n"))):
        nu = N // 2; xa, ya, _ = W.sample(rs(100 + idx), N - nu, [0, 1, 2, 3]); xu, yu, _ = W.sample(rs(100 + 3 + idx), nu, UNANS_TRAIN); x, y = torch.cat([xa, xu]), torch.cat([ya, yu])
        flag = torch.cat([torch.zeros(N - nu, dtype=torch.bool), torch.ones(nu, dtype=torch.bool)]); S, P = read(x); mu = [s.mean(0) for s in S]; sd = [s.std(0) + 1e-6 for s in S]
        feat = lambda S_, P_, j, state: torch.cat([P_[:, j], (S_[j] - mu[j]) / sd[j]], 1) if state else P_[:, j]
        arms = [("SET_6000", True, False)] if idx else [("SET", True, False), ("ANS", False, False), ("REFRESH_SET", True, True), ("REFRESH_ANS", False, True)]
        for name, state, refresh in arms:
            m = SetSelfR([feat(S, P, j, state).shape[1] for j in range(K)]); fit6(m, lambda i: m([feat(S, P, j, state)[i] for j in range(K)], P[i]), y, flag, seed, refresh)
            with torch.no_grad():
                for k in SETS: sc(name, k, m([feat(TS[k][0], TS[k][1], j, state) for j in range(K)], TS[k][1]), tests[k][1])
                sc(name, "train", m([feat(S, P, j, state) for j in range(K)], P), y)
        if not idx:
            m = raw_net(); torch.manual_seed(seed); fit6(m, lambda i: m(x[i]), y, flag, seed, False)
            with torch.no_grad():
                for k in SETS: sc("RAW", k, m(tests[k][0]), tests[k][1])
    ys = tests["seen"][1]; cs = tests["seen"][2]; ok = [(TS["seen"][1][:, j].argmax(1) == ys).numpy() for j in range(K)]
    out["sensor_in"] = float(np.mean([ok[j][np.isin(cs, cov[j])].mean() for j in range(NR)])); out["sensor_out"] = float(np.mean([ok[j][~np.isin(cs, cov[j])].mean() for j in range(NR)])); out["junk_acc"] = float(np.mean([ok[j].mean() for j in range(NR, K)]))
    out["real_acc_unans"] = float(np.mean([(TS["unans"][1][:, j].argmax(1) == tests["unans"][1]).float().mean() for j in range(NR)]))
    return out


def summarize(per):
    seeds = sorted(per); g = lambda f: np.array([f(per[s]) for s in seeds], float); a0 = per[seeds[0]]; ci = lambda d: X.boot_ci(d, C("x76_seed0"))
    U = lambda a, k: g(lambda r: r["util"][a][k]); A = lambda a, k: g(lambda r: r["abstain"][a][k]); arms = list(a0["util"])
    S = {"median_util": {a: {k: float(np.median(U(a, k))) for k in a0["util"][a]} for a in arms}, "median_abstain": {a: {k: float(np.median(A(a, k))) for k in a0["abstain"][a]} for a in arms}, "paired": {}}; P = S["paired"]
    P["held REFRESH_SET-SET"] = ci(U("REFRESH_SET", "unans") - U("SET", "unans")); P["novel REFRESH_SET-SET"] = ci(U("REFRESH_SET", "novel") - U("SET", "novel")); P["seen REFRESH_SET-REFRESH_ANS"] = ci(U("REFRESH_SET", "seen") - U("REFRESH_ANS", "seen"))
    P["held REFRESH_SET-SET_6000"] = ci(U("REFRESH_SET", "unans") - U("SET_6000", "unans")); P["seen REFRESH_SET-SET_6000"] = ci(U("REFRESH_SET", "seen") - U("SET_6000", "seen")); P["seen REFRESH_SET-ANS"] = ci(U("REFRESH_SET", "seen") - U("ANS", "seen"))
    S["manip"] = {k: float(np.median(g(lambda r: r[k]))) for k in ("sensor_in", "sensor_out", "junk_acc", "real_acc_unans")}
    return S


def judge(S):
    p, M, mu, ab = S["paired"], S["manip"], S["median_util"], S["median_abstain"]; r = lambda k: [round(x, 3) for x in p[k]]; bar = C("x75_abstain_bar")
    ok = M["sensor_out"] <= C("x72_unseen_ratio") * M["sensor_in"] and M["junk_acc"] <= C("x73_junk_acc") and M["real_acc_unans"] <= C("x74_chance_acc") and mu["SET"]["train"] >= C("x75_mem_bar") and ab["SET"]["unans"] <= C("x75_abstain_low")
    return [("P326", "Sensors as before and plain SET memorises", "PASSED" if ok else "VOID", f"inside/outside {M['sensor_in']:.2f}/{M['sensor_out']:.2f}; junk {M['junk_acc']:.2f}; held {M['real_acc_unans']:.2f}; SET train utility {mu['SET']['train']:.3f}, held abstain {ab['SET']['unans']:.2f}"),
            ("P327", "Refresh restores doubt", "PASSED" if ab["REFRESH_SET"]["unans"] >= bar and p["held REFRESH_SET-SET"][1] > 0 else "FAILED", f"REFRESH_SET held abstain {ab['REFRESH_SET']['unans']:.2f}, utility {mu['REFRESH_SET']['unans']:.3f} vs SET {mu['SET']['unans']:.3f}; difference {r('held REFRESH_SET-SET')}; REFRESH_ANS abstain {ab['REFRESH_ANS']['unans']:.2f}"),
            ("P328", "It stays competent", "PASSED" if ab["REFRESH_SET"]["seen"] <= C("x76_seen_abstain_max") and p["seen REFRESH_SET-REFRESH_ANS"][1] > 0 else "FAILED", f"seen abstain {ab['REFRESH_SET']['seen']:.2f}; seen utility {mu['REFRESH_SET']['seen']:.3f} vs REFRESH_ANS {mu['REFRESH_ANS']['seen']:.3f}; difference {r('seen REFRESH_SET-REFRESH_ANS')}"),
            ("P329", "Doubt generalises to a context nobody saw", "PASSED" if ab["REFRESH_SET"]["novel"] >= bar and p["novel REFRESH_SET-SET"][1] > 0 else "FAILED", f"novel abstain {ab['REFRESH_SET']['novel']:.2f}, utility {mu['REFRESH_SET']['novel']:.3f} vs SET {mu['SET']['novel']:.3f}; difference {r('novel REFRESH_SET-SET')}"),
            ("P330", "Refresh is worth more than 20x the data", "PASSED" if p["held REFRESH_SET-SET_6000"][1] >= -C("x75_held_tol") and p["seen REFRESH_SET-SET_6000"][1] >= -C("x76_seen_tol") else "FAILED", f"held {r('held REFRESH_SET-SET_6000')}; seen {r('seen REFRESH_SET-SET_6000')}; SET_6000 held {mu['SET_6000']['unans']:.3f}, seen {mu['SET_6000']['seen']:.3f}"),
            ("P331", "The insides pay in the doubting Self", "PASSED" if p["seen REFRESH_SET-ANS"][1] > 0 else "FAILED", f"REFRESH_SET minus plain ANS on seen {r('seen REFRESH_SET-ANS')}; plain ANS seen {mu['ANS']['seen']:.3f}")]


ARM_TEXT = {"SET": ("Self: answers + insides, plain", "X7.5's 300-case Self."), "ANS": ("Self: answers only, plain", ""), "REFRESH_SET": ("Self: answers + insides, fresh coin flips", "Unanswerable practice cases get a new random label every time they are seen."),
            "REFRESH_ANS": ("Self: answers only, fresh coin flips", "Same trick for the answers-only Self."), "SET_6000": ("Self: answers + insides, 6000 cases", "Plain, 20 times more practice."), "RAW": ("Raw input only", "Control.")}


def log_wandb(out):
    from track import Tracker
    S, per = out["summary"], out["per_world"]
    tk = Tracker("X7.6", "Can a Self learn doubt from cases it cannot memorise?", "If the labels of known-unanswerable practice cases are redrawn at random every time they are seen, does the Self learn to say 'I don't know' in new unanswerable situations and in a never-seen one, keeping its competence?",
                 "Twelve advisor networks (four junk). Half of the practice cases are coin flips. +1 right, -2 wrong, 0 abstain.", ARM_TEXT, "Utility per case, higher is better; abstain rate is the share of 'I don't know'.", dict(worlds=len(per)), tags=["self", "doubt", "refresh"])
    tk.table("Utility and abstain by arm (median over worlds)", ["Arm"] + [f"Utility {k}" for k in S["median_util"]["SET"]] + [f"Abstain {k}" for k in S["median_abstain"]["SET"]], [[a, *[round(S["median_util"][a].get(k, float("nan")), 3) for k in S["median_util"]["SET"]], *[round(S["median_abstain"][a].get(k, float("nan")), 3) for k in S["median_abstain"]["SET"]]] for a in S["median_util"]])
    tk.table("Paired differences with 95% ranges", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    t0 = time.time()
    if mode == "selftest":
        r = one(C("x71_seed0")); print(json.dumps({a: {k: round(v, 2) for k, v in r["util"][a].items()} for a in r["util"]}), json.dumps({a: r["abstain"][a]["unans"] for a in r["abstain"]})); print("selftest ok"); return
    if mode == "log": log_wandb(json.load(open(os.path.join(OUT, "x7_6_run.json")))["result"]); return
    from budget import pmap, workers
    seeds = [C("x76_seed0") + i for i in range(C("x76_worlds"))]; n = min(workers(1.5), C("x75_workers_cap")); print("workers", n, flush=True); res = pmap(one, [(s,) for s in seeds], n); per = dict(zip(seeds, res))
    S = summarize(per); out = {"per_world": per, "summary": S, "verdicts": judge(S)}
    os.makedirs(OUT, exist_ok=True); json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x7_6_run.json"), "w"), indent=1, default=float)
    for v in out["verdicts"]: print(v[0], v[2], "|", v[3], flush=True)
    log_wandb(out)


if __name__ == "__main__":
    main(sys.argv[1])
