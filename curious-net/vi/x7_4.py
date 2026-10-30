"""X7.4: can the Self learn doubt? X7.3's pool of 12 sensors (4 junk), same Self and goal, but half of the Self's training cases come from UNANSWERABLE contexts (labels independent of the input).
Registered in paper/PREDICTIONS.md (P314-P319) before this was run. Usage: python x7_4.py selftest | run | log   (writes ../results/x7_4_run.json)"""
import sys, os, json, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x0_1 as X
from x7_1 import Mirror
from x7_2 import make_sensor
from x7_3 import FAMS, train_sensor, SetSelf, raw_net, fit, realised, rule_util

OUT = X.OUT
SETS = ("seen", "unans", "novel")
UNANS_TRAIN, UNANS_HELD, NOVEL = [4, 5, 6], [7], [8]


class World:
    def __init__(s, seed):
        g = np.random.default_rng(seed); I, R = C("x71_in_dim"), C("x71_regimes")
        s.protos = g.normal(size=(R, I)); s.Q = [np.linalg.qr(g.normal(size=(I, I)))[0] for _ in range(C("x74_contexts"))]
        s.perm = [g.permutation(R) for _ in range(C("x74_contexts"))]

    def sample(s, rng, n, ctxs):
        c = rng.choice(ctxs, n); t = rng.integers(0, C("x71_regimes"), n)
        x = np.einsum("nij,nj->ni", np.stack([s.Q[k] for k in c]), s.protos[t] + C("x72_class_sd") * rng.normal(size=(n, C("x71_in_dim"))))
        y = np.where(np.isin(c, UNANS_TRAIN + UNANS_HELD), rng.integers(0, C("x71_regimes"), n), t)               # unanswerable contexts: label independent of the input
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y), c


def one(seed):
    torch.set_num_threads(1); NR, NJ = C("x73_real"), C("x73_junk"); K = NR + NJ; W = World(seed); rs = lambda o: np.random.default_rng(seed * 10 + o)
    sens, mirs, cov = [], [], []
    for j in range(K):
        cj = [j % 4, (j + 1) % 4]; m = train_sensor(make_sensor(FAMS[j % 4], seed + j), W, cj, seed * 100 + j, j >= NR); sens.append(m); mirs.append(Mirror(m, seed + j)); cov.append(cj)
    n = C("x73_self_n"); nu = int(C("x74_unans_share") * n)
    xa, ya, ca = W.sample(rs(1), n - nu, [0, 1, 2, 3]); xu, yu, cu = W.sample(rs(5), nu, UNANS_TRAIN)
    data = {"train": (torch.cat([xa, xu]), torch.cat([ya, yu]), np.concatenate([ca, cu])), "solv": (xa, ya, ca), "seen": W.sample(rs(2), C("x73_test_seen"), [0, 1, 2, 3]),
            "unans": W.sample(rs(3), C("x73_test_other"), UNANS_HELD), "novel": W.sample(rs(4), C("x73_test_other"), NOVEL)}
    P, S = {}, {}
    for k, (x, y, c) in data.items():
        with torch.no_grad(): S[k] = [mirs[j].read(x) for j in range(K)]; P[k] = torch.stack([F.softmax(sens[j](x), -1) for j in range(K)], 1)
    mu = [s.mean(0) for s in S["train"]]; sd = [s.std(0) + 1e-6 for s in S["train"]]
    feat = lambda k, j, state: torch.cat([P[k][:, j], (S[k][j] - mu[j]) / sd[j]], 1) if state else P[k][:, j]
    out = {"util": {}, "abstain": {}}

    def score(name, k, o):
        u, a = realised(o, data[k][1]); out["util"].setdefault(name, {})[k] = u; out["abstain"].setdefault(name, {})[k] = a
    for name, state, tr in (("SET12", True, "train"), ("ANS12", False, "train"), ("SET12_SOLVONLY", True, "solv")):
        y = data[tr][1]; m = SetSelf([feat(tr, j, state).shape[1] for j in range(K)]); fit(m, lambda i: m([feat(tr, j, state)[i] for j in range(K)], P[tr][i]), y, seed)
        with torch.no_grad():
            for k in SETS + ("train",): score(name, k, m([feat(k, j, state) for j in range(K)], P[k]))
    m = raw_net(); fit(m, lambda i: m(data["train"][0][i]), data["train"][1], seed)
    with torch.no_grad():
        for k in SETS: score("RAW", k, m(data[k][0]))
    ytr = data["train"][1]; b = int(np.argmax([(P["train"][:, j].argmax(1) == ytr).float().mean() for j in range(K)]))
    for k in SETS:
        y = data[k][1]; picks = {"BEST": P[k][:, b], "MAJ": P[k].mean(1), "CONF": P[k][torch.arange(len(y)), P[k].max(2).values.argmax(1)]}
        for tag, p in picks.items():
            u, a = rule_util(p, y); out["util"].setdefault(tag, {})[k] = u; out["abstain"].setdefault(tag, {})[k] = a
    ys = data["seen"][1]; cs = data["seen"][2]; ok = [(P["seen"][:, j].argmax(1) == ys).numpy() for j in range(K)]
    out["sensor_in"] = float(np.mean([ok[j][np.isin(cs, cov[j])].mean() for j in range(NR)])); out["sensor_out"] = float(np.mean([ok[j][~np.isin(cs, cov[j])].mean() for j in range(NR)])); out["junk_acc"] = float(np.mean([ok[j].mean() for j in range(NR, K)]))
    out["real_acc_unans"] = float(np.mean([(P["unans"][:, j].argmax(1) == data["unans"][1]).float().mean() for j in range(NR)]))
    return out


def summarize(per):
    seeds = sorted(per); g = lambda f: np.array([f(per[s]) for s in seeds], float); ci = lambda d: X.boot_ci(d, C("x74_seed0"))
    U = lambda a, k: g(lambda r: r["util"][a][k]); A = lambda a, k: g(lambda r: r["abstain"][a][k]); arms = list(per[seeds[0]]["util"])
    S = {"median_util": {a: {k: float(np.median(U(a, k))) for k in SETS} for a in arms}, "median_abstain": {a: {k: float(np.median(A(a, k))) for k in SETS} for a in arms}, "paired": {}}
    for k in SETS:
        rule = np.max([U(t, k) for t in ("BEST", "MAJ", "CONF")], 0); S["paired"][f"{k} SET12-SOLVONLY"] = ci(U("SET12", k) - U("SET12_SOLVONLY", k)); S["paired"][f"{k} SET12-ANS12"] = ci(U("SET12", k) - U("ANS12", k))
        S["paired"][f"{k} SET12-RAW"] = ci(U("SET12", k) - U("RAW", k)); S["paired"][f"{k} SET12-best rule"] = ci(U("SET12", k) - rule)
    S["paired"]["held+novel SET12-ANS12"] = ci((U("SET12", "unans") + U("SET12", "novel")) / 2 - (U("ANS12", "unans") + U("ANS12", "novel")) / 2)
    S["train_abstain"] = {a: float(np.median(A(a, "train"))) for a in ("SET12", "ANS12", "SET12_SOLVONLY")}; S["train_util"] = {a: float(np.median(U(a, "train"))) for a in ("SET12", "ANS12", "SET12_SOLVONLY")}
    S["manip"] = {k: float(np.median(g(lambda r: r[k]))) for k in ("sensor_in", "sensor_out", "junk_acc", "real_acc_unans")}
    return S


def judge(S):
    p, M, mu, ab = S["paired"], S["manip"], S["median_util"], S["median_abstain"]; r = lambda k: [round(x, 3) for x in p[k]]; tol = C("x74_cost_tol")
    ok = M["sensor_out"] <= C("x72_unseen_ratio") * M["sensor_in"] and M["junk_acc"] <= C("x73_junk_acc") and M["real_acc_unans"] <= C("x74_chance_acc")
    return [("P314", "Sensors fallible, junk bad, and nothing can be known in the unanswerable context", "PASSED" if ok else "VOID", f"inside/outside {M['sensor_in']:.2f}/{M['sensor_out']:.2f}; junk {M['junk_acc']:.2f}; real sensors' accuracy in the unanswerable-held context {M['real_acc_unans']:.2f}"),
            ("P315", "Doubt where the truth is unrecoverable", "PASSED" if ab["SET12"]["unans"] >= C("x74_abstain_bar") and p["unans SET12-SOLVONLY"][1] > 0 else "FAILED", f"abstain rate {ab['SET12']['unans']:.2f} [diagnostic, on its own training cases: abstain {S['train_abstain']['SET12']:.2f}, utility {S['train_util']['SET12']:.3f}] (SOLVONLY {ab['SET12_SOLVONLY']['unans']:.2f}); utility {mu['SET12']['unans']:.3f} vs {mu['SET12_SOLVONLY']['unans']:.3f}; difference {r('unans SET12-SOLVONLY')}"),
            ("P316", "Doubt does not wreck competence", "PASSED" if ab["SET12"]["seen"] <= C("x74_seen_abstain_max") and p["seen SET12-SOLVONLY"][1] >= -tol else "FAILED", f"seen abstain rate {ab['SET12']['seen']:.2f}; utility {mu['SET12']['seen']:.3f} vs SOLVONLY {mu['SET12_SOLVONLY']['seen']:.3f}; difference {r('seen SET12-SOLVONLY')}"),
            ("P317", "Doubt generalises to a context nobody saw", "PASSED" if ab["SET12"]["novel"] >= C("x74_abstain_novel_bar") and p["novel SET12-SOLVONLY"][1] > 0 else "FAILED", f"novel abstain rate {ab['SET12']['novel']:.2f} (SOLVONLY {ab['SET12_SOLVONLY']['novel']:.2f}); utility {mu['SET12']['novel']:.3f} vs {mu['SET12_SOLVONLY']['novel']:.3f}; difference {r('novel SET12-SOLVONLY')}; vs best rule {r('novel SET12-best rule')}"),
            ("P318", "The insides help it doubt", "PASSED" if p["held+novel SET12-ANS12"][1] > 0 else "FAILED", f"SET12 minus ANS12 pooled over held and novel {r('held+novel SET12-ANS12')}; RAW comparison held {r('unans SET12-RAW')}, novel {r('novel SET12-RAW')}"),
            ("P319", "Checking still beats re-solving", "PASSED" if p["seen SET12-RAW"][1] > 0 else "FAILED", f"SET12 minus RAW on seen {r('seen SET12-RAW')}; utilities {mu['SET12']['seen']:.3f} vs {mu['RAW']['seen']:.3f}")]


ARM_TEXT = {"SET12": ("Self: answers + complete insides", "The proposal, now also practised on unanswerable situations."), "ANS12": ("Self: answers only", "Same, sees only answers."), "RAW": ("Self: raw input only", "Control: tries to do the job itself."),
            "SET12_SOLVONLY": ("Self trained only on answerable cases", "X7.3-style training: never saw an unanswerable case."), "BEST": ("Rule: best single sensor", "With derived abstain threshold."), "MAJ": ("Rule: average of sensors", "With derived abstain threshold."), "CONF": ("Rule: most confident sensor", "With derived abstain threshold.")}


def log_wandb(out):
    from track import Tracker
    S, per = out["summary"], out["per_world"]
    tk = Tracker("X7.4", "Can a Self with a goal learn when not to answer?", "If part of the Self's practice is on situations where nobody can know the answer, does it learn to say 'I don't know' there, in a new unanswerable situation and in a situation it has never seen, while still answering when it can?",
                 "Twelve advisor networks (four junk). Some situations are coin flips (labels unrelated to the input). +1 right, -2 wrong, 0 abstain.", ARM_TEXT, "Utility per case, higher is better; abstain rate is the share of cases answered with 'I don't know'.", dict(worlds=len(per)), tags=["self", "doubt"])
    tk.table("Utility and abstain rate by arm (median over worlds)", ["Arm"] + [f"Utility {k}" for k in SETS] + [f"Abstain {k}" for k in SETS], [[a, *[round(S["median_util"][a][k], 3) for k in SETS], *[round(S["median_abstain"][a][k], 3) for k in SETS]] for a in S["median_util"]])
    tk.table("Paired differences with 95% ranges", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    t0 = time.time()
    if mode == "selftest": r = one(C("x71_seed0")); print(json.dumps({a: {k: round(v, 2) for k, v in r["util"][a].items()} for a in ("SET12", "SET12_SOLVONLY", "RAW")}), json.dumps(r["abstain"]["SET12"]), r["real_acc_unans"]); print("selftest ok"); return
    if mode == "log": log_wandb(json.load(open(os.path.join(OUT, "x7_4_run.json")))["result"]); return
    from budget import pmap
    seeds = [C("x74_seed0") + i for i in range(C("x74_worlds"))]; res = pmap(one, [(s,) for s in seeds], C("x73_workers")); per = dict(zip(seeds, res))
    S = summarize(per); out = {"per_world": per, "summary": S, "verdicts": judge(S)}
    os.makedirs(OUT, exist_ok=True); json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x7_4_run.json"), "w"), indent=1, default=float)
    for v in out["verdicts"]: print(v[0], v[2], "|", v[3], flush=True)
    log_wandb(out)


if __name__ == "__main__":
    main(sys.argv[1])
