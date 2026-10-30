"""X7.5: taming the complete internal state. X7.4's Self memorised its 300 training cases from the sensors' full internal states and never learned doubt;
does more data (N = 300, 1500, 6000) or a single constraint on the state channel (dropout, bottleneck, weight decay) fix it?
Registered in paper/PREDICTIONS.md (P320-P325) before this was run. Usage: python x7_5.py selftest | run | log   (writes ../results/x7_5_run.json)"""
import sys, os, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x0_1 as X
from x7_1 import Mirror
from x7_2 import make_sensor
from x7_3 import FAMS, train_sensor, SetSelf, raw_net, util_loss, realised
from x7_4 import World, UNANS_TRAIN, UNANS_HELD, NOVEL

OUT = X.OUT
SETS = ("seen", "unans", "novel")


class SetSelfR(SetSelf):
    """x7_3.SetSelf with an optional adapter width and dropout on the state columns (training only)."""
    def __init__(s, widths, A=None, drop=0.0):
        nn.Module.__init__(s); R, H, A = C("x71_regimes"), C("x71_hidden"), A or C("x73_attn_w"); s.drop, s.R = drop, R
        s.ad = nn.ModuleList([nn.Linear(w, A) for w in widths]); s.q = nn.Linear(A, 1); s.head = nn.Sequential(nn.Linear(A + R, H), nn.Tanh(), nn.Linear(H, R + 1))

    def forward(s, feats, P):
        if s.drop and s.training: feats = [torch.cat([f[:, :s.R], F.dropout(f[:, s.R:], s.drop, True)], 1) for f in feats]
        return SetSelf.forward(s, feats, P)


def fit5(model, fwd, y, seed, wd=0.0):
    torch.manual_seed(seed); opt = (torch.optim.AdamW(model.parameters(), lr=C("x71_lr"), weight_decay=wd) if wd else torch.optim.Adam(model.parameters(), lr=C("x71_lr"))); rng = np.random.default_rng(seed)
    for _ in range(C("x73_self_iters")):
        i = torch.tensor(rng.integers(0, len(y), C("x71_host_batch"))); loss = util_loss(fwd(i), y[i]); opt.zero_grad(); loss.backward(); opt.step()
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
    sc = lambda name, k, o, y: (out["util"].setdefault(name, {}).__setitem__(k, realised(o, y)[0]), out["abstain"].setdefault(name, {}).__setitem__(k, realised(o, y)[1]))
    for idx, N in enumerate(C("x75_sizes")):
        nu = N // 2; xa, ya, _ = W.sample(rs(100 + idx), N - nu, [0, 1, 2, 3]); xu, yu, _ = W.sample(rs(100 + len(C("x75_sizes")) + idx), nu, UNANS_TRAIN); x, y = torch.cat([xa, xu]), torch.cat([ya, yu])
        S, P = read(x); mu = [s.mean(0) for s in S]; sd = [s.std(0) + 1e-6 for s in S]
        feat = lambda S_, P_, j, state: torch.cat([P_[:, j], (S_[j] - mu[j]) / sd[j]], 1) if state else P_[:, j]
        arms = [(f"SET{N}", True, None, 0.0, 0.0), (f"ANS{N}", False, None, 0.0, 0.0)]
        if N == C("x75_sizes")[0]: arms += [("DROP", True, None, C("x75_drop"), 0.0), ("BOT", True, C("x75_bot"), 0.0, 0.0), ("WD", True, None, 0.0, C("x75_wd"))]
        for name, state, A, drop, wd in arms:
            m = SetSelfR([feat(S, P, j, state).shape[1] for j in range(K)], A, drop); fit5(m, lambda i: m([feat(S, P, j, state)[i] for j in range(K)], P[i]), y, seed, wd)
            with torch.no_grad():
                for k in SETS: sc(name, k, m([feat(TS[k][0], TS[k][1], j, state) for j in range(K)], TS[k][1]), tests[k][1])
                sc(name, "train", m([feat(S, P, j, state) for j in range(K)], P), y)
        if N == C("x75_sizes")[0]:
            m = raw_net(); torch.manual_seed(seed); fit5(m, lambda i: m(x[i]), y, seed)
            with torch.no_grad():
                for k in SETS: sc("RAW", k, m(tests[k][0]), tests[k][1])
    ys = tests["seen"][1]; cs = tests["seen"][2]; ok = [(TS["seen"][1][:, j].argmax(1) == ys).numpy() for j in range(K)]
    out["sensor_in"] = float(np.mean([ok[j][np.isin(cs, cov[j])].mean() for j in range(NR)])); out["sensor_out"] = float(np.mean([ok[j][~np.isin(cs, cov[j])].mean() for j in range(NR)])); out["junk_acc"] = float(np.mean([ok[j].mean() for j in range(NR, K)]))
    out["real_acc_unans"] = float(np.mean([(TS["unans"][1][:, j].argmax(1) == tests["unans"][1]).float().mean() for j in range(NR)]))
    return out


def ci99(d):
    d = np.asarray(d, float); meds = np.median(np.random.default_rng(C("x75_seed0")).choice(d, (C("boot"), d.size)), axis=1)
    return [float(np.median(d)), float(np.percentile(meds, C("x75_ci_lo"))), float(np.percentile(meds, C("x75_ci_hi")))]


def summarize(per):
    seeds = sorted(per); g = lambda f: np.array([f(per[s]) for s in seeds], float); a0 = per[seeds[0]]; Ns = C("x75_sizes"); n0, n1 = Ns[0], Ns[-1]
    U = lambda a, k: g(lambda r: r["util"][a][k]); A = lambda a, k: g(lambda r: r["abstain"][a][k]); arms = list(a0["util"])
    S = {"median_util": {a: {k: float(np.median(U(a, k))) for k in a0["util"][a]} for a in arms}, "median_abstain": {a: {k: float(np.median(A(a, k))) for k in a0["abstain"][a]} for a in arms}, "paired": {}}
    P = S["paired"]; P["held SET6000-SET300"] = ci99(U(f"SET{n1}", "unans") - U(f"SET{n0}", "unans")); P["train SET6000-SET300"] = ci99(U(f"SET{n1}", "train") - U(f"SET{n0}", "train"))
    for r in ("DROP", "BOT", "WD"): P[f"held {r}-SET300"] = ci99(U(r, "unans") - U(f"SET{n0}", "unans")); P[f"seen {r}-ANS300"] = ci99(U(r, "seen") - U(f"ANS{n0}", "seen")); P[f"novel {r}-SET300"] = ci99(U(r, "novel") - U(f"SET{n0}", "novel"))
    P["seen SET6000-ANS6000"] = ci99(U(f"SET{n1}", "seen") - U(f"ANS{n1}", "seen")); P["held SET6000-ANS6000"] = ci99(U(f"SET{n1}", "unans") - U(f"ANS{n1}", "unans"))
    for N in Ns: P[f"seen SET{N}-ANS{N}"] = ci99(U(f"SET{N}", "seen") - U(f"ANS{N}", "seen"))
    S["manip"] = {k: float(np.median(g(lambda r: r[k]))) for k in ("sensor_in", "sensor_out", "junk_acc", "real_acc_unans")}
    return S


def judge(S):
    p, M, mu, ab = S["paired"], S["manip"], S["median_util"], S["median_abstain"]; Ns = C("x75_sizes"); n0, n1 = Ns[0], Ns[-1]; r = lambda k: [round(x, 3) for x in p[k]]
    ok = M["sensor_out"] <= C("x72_unseen_ratio") * M["sensor_in"] and M["junk_acc"] <= C("x73_junk_acc") and M["real_acc_unans"] <= C("x74_chance_acc") and mu[f"SET{n0}"]["train"] >= C("x75_mem_bar") and ab[f"SET{n0}"]["unans"] <= C("x75_abstain_low")
    cand = [x for x in ("DROP", "BOT", "WD") if ab[x]["unans"] >= C("x75_abstain_bar") and p[f"held {x}-SET300"][1] > 0]; best = max(cand, key=lambda x: mu[x]["unans"]) if cand else None
    tbl = {x: f"{x}: abstain {ab[x]['unans']:.2f}, held utility {mu[x]['unans']:.3f}, diff vs SET300 {r(f'held {x}-SET300')}, seen vs ANS300 {r(f'seen {x}-ANS300')}" for x in ("DROP", "BOT", "WD")}
    return [("P320", "Sensors as before and X7.4's memorisation reproduces", "PASSED" if ok else "VOID", f"inside/outside {M['sensor_in']:.2f}/{M['sensor_out']:.2f}; junk {M['junk_acc']:.2f}; real sensors held {M['real_acc_unans']:.2f}; SET300 train utility {mu[f'SET{n0}']['train']:.3f}, held abstain {ab[f'SET{n0}']['unans']:.2f}"),
            ("P321", "More labels restore doubt", "PASSED" if ab[f"SET{n1}"]["unans"] >= C("x75_abstain_bar") and p["held SET6000-SET300"][1] > 0 else "FAILED", f"SET{n1} held abstain {ab[f'SET{n1}']['unans']:.2f}, utility {mu[f'SET{n1}']['unans']:.3f} (N=300 {mu[f'SET{n0}']['unans']:.3f}); difference {r('held SET6000-SET300')}; by N held utility {[round(mu[f'SET{N}']['unans'], 3) for N in Ns]}, abstain {[round(ab[f'SET{N}']['unans'], 2) for N in Ns]}"),
            ("P322", "Memorisation shrinks with data", "PASSED" if p["train SET6000-SET300"][2] < 0 else "FAILED", f"training-case utility by N {[round(mu[f'SET{N}']['train'], 3) for N in Ns]}; difference {r('train SET6000-SET300')}"),
            ("P323", "A constraint on the state channel restores doubt at N=300", "PASSED" if best else "FAILED", "; ".join(tbl.values()) + f"; passing: {cand}"),
            ("P324", "Restored doubt keeps the insides' value", "PASSED" if best and p[f"seen {best}-ANS300"][1] >= 0 else "FAILED", f"best arm {best}; " + (f"seen {best} minus ANS300 {r(f'seen {best}-ANS300')}" if best else "no arm passed P323")),
            ("P325", "With enough data the insides pay again", "PASSED" if p["seen SET6000-ANS6000"][1] > 0 and p["held SET6000-ANS6000"][1] >= -C("x75_held_tol") else "FAILED", f"seen SET minus ANS at N=6000 {r('seen SET6000-ANS6000')}; held {r('held SET6000-ANS6000')}; by N seen {[r(f'seen SET{N}-ANS{N}')[0] for N in Ns]}")]


ARM_TEXT = {"SET300": ("Self: answers + complete insides, 300 cases", "X7.4's Self."), "SET1500": ("Same, 1500 cases", "More practice."), "SET6000": ("Same, 6000 cases", "Much more practice."), "ANS300": ("Self: answers only, 300 cases", "Sees only the sensors' answers."),
            "ANS1500": ("Answers only, 1500", ""), "ANS6000": ("Answers only, 6000", ""), "DROP": ("300 cases, dropout on insides", "Half of the insides' numbers hidden at random during practice."), "BOT": ("300 cases, narrow channel", "Each sensor's insides squeezed to 4 numbers."),
            "WD": ("300 cases, weight decay", "Large weights penalised."), "RAW": ("Raw input only, 300 cases", "Control.")}


def log_wandb(out):
    from track import Tracker
    S, per = out["summary"], out["per_world"]
    tk = Tracker("X7.5", "Can a Self learn to use a network's complete insides without memorising them?", "X7.4's Self learned its practice cases by heart and never learned to say 'I don't know'. Does more practice, or restricting its use of the sensors' insides, fix it while keeping the insides' value?",
                 "Same twelve advisors, half of the practice cases unanswerable coin flips, +1 right, -2 wrong, 0 abstain.", ARM_TEXT, "Utility per case, higher is better; abstain rate is the share of 'I don't know'. Ranges are 99% (3 comparisons).", dict(worlds=len(per)), tags=["self", "memorisation"])
    tk.table("Utility and abstain by arm (median over worlds)", ["Arm"] + [f"Utility {k}" for k in S["median_util"]["SET300"]] + [f"Abstain {k}" for k in S["median_abstain"]["SET300"]], [[a, *[round(S["median_util"][a].get(k, float("nan")), 3) for k in S["median_util"]["SET300"]], *[round(S["median_abstain"][a].get(k, float("nan")), 3) for k in S["median_abstain"]["SET300"]]] for a in S["median_util"]])
    tk.table("Paired differences with 99% ranges", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    t0 = time.time()
    if mode == "selftest": r = one(C("x71_seed0")); print(json.dumps({a: {k: round(v, 2) for k, v in r["util"][a].items()} for a in r["util"]})); print("selftest ok"); return
    if mode == "log": log_wandb(json.load(open(os.path.join(OUT, "x7_5_run.json")))["result"]); return
    from budget import pmap, workers
    seeds = [C("x75_seed0") + i for i in range(C("x75_worlds"))]; n = min(workers(1.5), C("x75_workers_cap")); print("workers", n, flush=True); res = pmap(one, [(s,) for s in seeds], n); per = dict(zip(seeds, res))
    S = summarize(per); out = {"per_world": per, "summary": S, "verdicts": judge(S)}
    os.makedirs(OUT, exist_ok=True); json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x7_5_run.json"), "w"), indent=1, default=float)
    for v in out["verdicts"]: print(v[0], v[2], "|", v[3], flush=True)
    log_wandb(out)


if __name__ == "__main__":
    main(sys.argv[1])
