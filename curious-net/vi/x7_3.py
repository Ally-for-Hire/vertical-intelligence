"""X7.3: a Self with a life goal (be correct; say so when unsure) over a pool of up to 12 sensor networks (4 of them junk), no raw input, a small label budget.
Registered in paper/PREDICTIONS.md (P307-P313) before this was run. The state reader is x7_1.Mirror (forward hooks on every leaf module).
Usage: python x7_3.py selftest | run | log        (writes ../results/x7_3_<mode>.json)"""
import sys, os, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x0_1 as X
from x7_1 import Mirror
from x7_2 import make_sensor

OUT = X.OUT
FAMS = ("MLP", "CONV1D", "GRU", "TRANSFORMER")
SETS = ("seen", "blackout", "novel")


class World:
    def __init__(s, seed):
        g = np.random.default_rng(seed); I, R = C("x71_in_dim"), C("x71_regimes")
        s.protos = g.normal(size=(R, I)); s.Q = [np.linalg.qr(g.normal(size=(I, I)))[0] for _ in range(C("x73_contexts"))]
        s.perm = [g.permutation(R) for _ in range(C("x73_contexts"))]                      # the junk sensors' fixed relabelling per context

    def sample(s, rng, n, ctxs):
        c = rng.choice(ctxs, n); y = rng.integers(0, C("x71_regimes"), n)
        x = np.einsum("nij,nj->ni", np.stack([s.Q[k] for k in c]), s.protos[y] + C("x72_class_sd") * rng.normal(size=(n, C("x71_in_dim"))))
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y), c


def train_sensor(m, W, ctxs, seed, junk):
    opt = torch.optim.Adam(m.parameters(), lr=C("x71_lr")); rng = np.random.default_rng(seed)
    for _ in range(C("x73_sensor_steps")):
        x, y, c = W.sample(rng, C("x71_host_batch"), ctxs)
        if junk: y = torch.tensor(np.array([W.perm[k][t] for k, t in zip(c, y.numpy())]))
        loss = F.cross_entropy(m(x), y); opt.zero_grad(); loss.backward(); opt.step()
    return m


class SetSelf(nn.Module):
    """The Self: one adapter per sensor (answer, optionally complete internal state) -> attention pool over the K sensors present -> 4 classes + abstain."""
    def __init__(s, widths):
        super().__init__(); R, A, H = C("x71_regimes"), C("x73_attn_w"), C("x71_hidden")
        s.ad = nn.ModuleList([nn.Linear(w, A) for w in widths]); s.q = nn.Linear(A, 1); s.head = nn.Sequential(nn.Linear(A + R, H), nn.Tanh(), nn.Linear(H, R + 1))

    def forward(s, feats, P):                                                # feats: list of (n, w_j) for the K sensors present; P (n,K,R)
        e = torch.stack([torch.tanh(s.ad[j](f)) for j, f in enumerate(feats)], 1); a = F.softmax(s.q(e).squeeze(-1), 1)
        return s.head(torch.cat([(a[..., None] * e).sum(1), (a[..., None] * P).sum(1)], 1))


def raw_net():
    I, R, H = C("x71_in_dim"), C("x71_regimes"), C("x71_hidden")
    return nn.Sequential(nn.Linear(I, H), nn.Tanh(), nn.Linear(H, H), nn.Tanh(), nn.Linear(H, R + 1))


def util_loss(o, y):
    R = C("x71_regimes"); p = F.softmax(o, -1); pc = p[torch.arange(len(y)), y]; pw = p[:, :R].sum(1) - pc
    ce = F.cross_entropy(o[:, :R], y)                                       # amendment A1 (see PREDICTIONS): learn to read the classes before learning to doubt
    return -(pc - C("x73_wrong_cost") * pw).mean() + C("x73_aux_w") * ce    # minus expected utility (the life goal) plus the reading aid


def fit(model, fwd, y, seed):
    torch.manual_seed(seed); opt = torch.optim.Adam(model.parameters(), lr=C("x71_lr")); rng = np.random.default_rng(seed)
    for _ in range(C("x73_self_iters")):
        i = torch.tensor(rng.integers(0, len(y), C("x71_host_batch"))); loss = util_loss(fwd(i), y[i]); opt.zero_grad(); loss.backward(); opt.step()
    return model


def realised(o, y):
    """Utility of the decision argmax over [classes, abstain]; returns (mean utility, abstain rate)."""
    R, d = C("x71_regimes"), o.argmax(1); ab = d == R
    u = torch.where(ab, torch.zeros(len(y)), torch.where(d == y, torch.ones(len(y)), -C("x73_wrong_cost") * torch.ones(len(y))))
    return float(u.mean()), float(ab.float().mean())


def rule_util(p, y):
    thr = C("x73_wrong_cost") / (1 + C("x73_wrong_cost")); conf, ans = p.max(1); answer = conf >= thr           # DERIVED: answer only when confidence >= cost/(1+cost)
    u = torch.where(answer, torch.where(ans == y, torch.ones(len(y)), -C("x73_wrong_cost") * torch.ones(len(y))), torch.zeros(len(y)))
    return float(u.mean()), float(1 - answer.float().mean())


def one(seed):
    torch.set_num_threads(1); R, NR, NJ = C("x71_regimes"), C("x73_real"), C("x73_junk"); K = NR + NJ; W = World(seed); rs = lambda o: np.random.default_rng(seed * 10 + o)
    seen = list(range(4)); sens, mirs, cov = [], [], []                                                           # four contexts are 'seen'
    for j in range(K):
        junk = j >= NR; cj = [j % 4, (j + 1) % 4]; m = train_sensor(make_sensor(FAMS[j % 4], seed + j), W, cj, seed * 100 + j, junk); sens.append(m); mirs.append(Mirror(m, seed + j)); cov.append(cj)
    tr_ctx = list(range(5)); data = {"train": W.sample(rs(1), C("x73_self_n"), tr_ctx), "seen": W.sample(rs(2), C("x73_test_seen"), seen),
                                     "blackout": W.sample(rs(3), C("x73_test_other"), [4]), "novel": W.sample(rs(4), C("x73_test_other"), [5])}
    P, S = {}, {}
    for k, (x, y, c) in data.items():
        with torch.no_grad(): S[k] = [mirs[j].read(x) for j in range(K)]; P[k] = torch.stack([F.softmax(sens[j](x), -1) for j in range(K)], 1)
    mu = [s.mean(0) for s in S["train"]]; sd = [s.std(0) + 1e-6 for s in S["train"]]; ytr = data["train"][1]
    feat = lambda k, j, state: torch.cat([P[k][:, j], (S[k][j] - mu[j]) / sd[j]], 1) if state else P[k][:, j]
    out = {"util": {}, "abstain": {}}
    sc = lambda name, k, o: out["util"].setdefault(name, {}).__setitem__(k, realised(o, data[k][1])[0]) or out["abstain"].setdefault(name, {}).__setitem__(k, realised(o, data[k][1])[1])
    for Kl in C("x73_levels"):
        for tag, state in (("SET", True), ("ANS", False)):
            m = SetSelf([feat("train", j, state).shape[1] for j in range(Kl)]); fit(m, lambda i: m([feat("train", j, state)[i] for j in range(Kl)], P["train"][i, :Kl]), ytr, seed)
            with torch.no_grad():
                for k in SETS: sc(f"{tag}{Kl}", k, m([feat(k, j, state) for j in range(Kl)], P[k][:, :Kl]))
        acc_tr = [(P["train"][:, j].argmax(1) == ytr).float().mean() for j in range(Kl)]; b = int(np.argmax(acc_tr))
        for k in SETS:
            Pk = P[k][:, :Kl]; y = data[k][1]; picks = {"BEST": Pk[:, b], "MAJ": Pk.mean(1), "CONF": Pk[torch.arange(len(y)), Pk.max(2).values.argmax(1)]}
            for tag, p in picks.items():
                u, a = rule_util(p, y); out["util"].setdefault(f"{tag}{Kl}", {})[k] = u; out["abstain"].setdefault(f"{tag}{Kl}", {})[k] = a
    m = raw_net(); fit(m, lambda i: m(data["train"][0][i]), ytr, seed)
    with torch.no_grad():
        for k in SETS: sc("RAW", k, m(data[k][0]))
    for k in SETS: out["util"].setdefault("ORACLE", {})[k] = float((P[k][:, :NR].argmax(2) == data[k][1][:, None]).any(1).float().mean())
    ys = data["seen"][1]; ok = [(P["seen"][:, j].argmax(1) == ys).numpy() for j in range(K)]; cs = data["seen"][2]
    out["sensor_in"] = float(np.mean([ok[j][np.isin(cs, cov[j])].mean() for j in range(NR)])); out["sensor_out"] = float(np.mean([ok[j][~np.isin(cs, cov[j])].mean() for j in range(NR)]))
    out["junk_acc"] = float(np.mean([ok[j].mean() for j in range(NR, K)])); out["best_single_real"] = float(max(ok[j].mean() for j in range(NR)))
    return out


def summarize(per):
    seeds = sorted(per); g = lambda f: np.array([f(per[s]) for s in seeds], float); ci = lambda d: X.boot_ci(d, C("x73_seed0")); lv = C("x73_levels"); top = lv[-1]
    U = lambda a, k="seen": g(lambda r: r["util"][a][k]); S = {"paired": {}, "median_util": {a: {k: float(np.median(U(a, k))) for k in SETS} for a in per[seeds[0]]["util"] if a != "ORACLE"},
                                                               "median_abstain": {a: {k: float(np.median(g(lambda r: r["abstain"][a][k]))) for k in SETS} for a in per[seeds[0]]["abstain"]}}
    S["curve_seen"] = {f"SET{k}": S["median_util"][f"SET{k}"]["seen"] for k in lv}; S["curve_seen_ans"] = {f"ANS{k}": S["median_util"][f"ANS{k}"]["seen"] for k in lv}
    S["paired"]["seen SET8-SET1"] = ci(U("SET8") - U("SET1")); S["paired"]["seen SET8-SET4"] = ci(U("SET8") - U("SET4")); S["paired"]["seen SET12-SET8"] = ci(U("SET12") - U("SET8"))
    S["paired"]["seen SET12-RAW"] = ci(U("SET12") - U("RAW")); S["paired"]["seen SET12-ANS12"] = ci(U("SET12") - U("ANS12"))
    for k in ("blackout", "novel"):
        rule = np.max([U(f"{t}{top}", k) for t in ("BEST", "MAJ", "CONF")], 0); S["paired"][f"{k} SET12-best rule"] = ci(U("SET12", k) - rule); S["paired"][f"{k} SET12-RAW"] = ci(U("SET12", k) - U("RAW", k))
    rule = np.max([U(f"{t}{top}") for t in ("BEST", "MAJ", "CONF")], 0); S["paired"]["seen SET12-best rule"] = ci(U("SET12") - rule)
    S["manip"] = {k: float(np.median(g(lambda r: r[k]))) for k in ("sensor_in", "sensor_out", "junk_acc", "best_single_real")}; S["manip"]["oracle"] = float(np.median(U("ORACLE")))
    return S


def judge(S):
    p, M = S["paired"], S["manip"]; r = lambda k: [round(x, 4) for x in p[k]]; tol = C("x73_noharm"); lv = [k for k in C("x73_levels") if k <= 8]; cur = S["curve_seen"]
    mono = all(cur[f"SET{b}"] >= cur[f"SET{a}"] - C("x73_mono_tol") for a, b in zip(lv, lv[1:]))
    ok = M["sensor_out"] <= C("x72_unseen_ratio") * M["sensor_in"] and M["junk_acc"] <= C("x73_junk_acc") and M["best_single_real"] <= C("x72_room_bar") and M["oracle"] >= C("x72_oracle_bar")
    return [("P307", "Real sensors are fallible, junk sensors are bad, and there is room", "PASSED" if ok else "VOID", f"accuracy inside/outside contexts {M['sensor_in']:.2f}/{M['sensor_out']:.2f}; junk {M['junk_acc']:.2f}; best single real {M['best_single_real']:.2f}; oracle {M['oracle']:.3f}"),
            ("P308", "More input helps", "PASSED" if p["seen SET8-SET1"][1] > 0 and p["seen SET8-SET4"][1] > 0 and mono else "FAILED", f"K=8 minus K=1 {r('seen SET8-SET1')}; minus K=4 {r('seen SET8-SET4')}; curve {S['curve_seen']}; non-decreasing: {mono}"),
            ("P309", "Junk does not hurt", "PASSED" if p["seen SET12-SET8"][1] >= -tol else "FAILED", f"K=12 minus K=8 {r('seen SET12-SET8')}"),
            ("P310", "The Self checks rather than re-solves", "PASSED" if p["seen SET12-RAW"][1] > 0 else "FAILED", f"SET12 minus RAW {r('seen SET12-RAW')}; median utilities SET12 {S['median_util']['SET12']['seen']:.3f}, RAW {S['median_util']['RAW']['seen']:.3f}"),
            ("P311", "Learned distrust where it trained (blackout)", "PASSED" if p["blackout SET12-best rule"][1] > 0 else "FAILED", f"SET12 minus best rule {r('blackout SET12-best rule')}; SET12 abstains {S['median_abstain']['SET12']['blackout']:.2f}; medians {S['median_util']['SET12']['blackout']:.3f}"),
            ("P312", "Distrust generalises to a context nobody saw (novel)", "PASSED" if p["novel SET12-best rule"][1] >= -tol else "FAILED", f"SET12 minus best rule {r('novel SET12-best rule')}; SET12 abstains {S['median_abstain']['SET12']['novel']:.2f}; median utilities {S['median_util']['SET12']['novel']:.3f}"),
            ("P313", "Insides matter at scale", "PASSED" if p["seen SET12-ANS12"][1] > 0 else "FAILED", f"SET12 minus ANS12 {r('seen SET12-ANS12')}")]


ARM_TEXT = {"SET": ("Self: answers + complete insides", "Reads every sensor's answer and complete internal state; may answer or say 'I don't know'."), "ANS": ("Self: answers only", "Same Self, sees only the sensors' answers."),
            "RAW": ("Self: raw input only", "Control: tries to do the job itself from the same 300 examples."), "BEST": ("Rule: best single sensor", "Trusts the sensor best on the labelled examples; abstains below the derived confidence."),
            "MAJ": ("Rule: average of sensors", "Averages probabilities; abstains below the derived confidence."), "CONF": ("Rule: most confident sensor", "Picks the most confident sensor; abstains below the derived confidence."),
            "ORACLE": ("Ceiling: any real sensor right", "Not achievable; shows the room.")}


def log_wandb(out):
    from track import Tracker
    S, per = out["summary"], out["per_world"]
    tk = Tracker("X7.3", "A Self with a goal over many sensors", "With no raw input and only 300 solved examples, does a Self with the goal 'be correct, say so when unsure' keep improving as sensors are added, ignore junk sensors, beat a Self that does the job itself, and abstain when everything it sees is unreliable?",
                 "Twelve advisor networks (eight real, each reliable in two of four contexts; four junk and confidently wrong). A supervisor sees their answers (and insides) and may answer or abstain. +1 right, -2 wrong, 0 abstain. Context 4 is unknown to all advisors; context 5 to everyone.", ARM_TEXT,
                 "Utility per case: higher is better. Paired differences compare arms on the same worlds with 95% ranges.", dict(worlds=len(per), levels=list(C("x73_levels"))), tags=["self", "goal", "many-sensors"])
    tk.table("Utility by arm and context (median over worlds)", ["Arm", "Seen contexts", "Blackout context", "Novel context", "Abstain rate in novel context"], [[a, *[round(S["median_util"][a][k], 3) for k in SETS], round(S["median_abstain"][a]["novel"], 3)] for a in S["median_util"] if a in S["median_abstain"]])
    tk.table("Paired differences with 95% ranges", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.table("Results per world (Self at 12 sensors, seen contexts)", ["World seed", "Utility SET12", "Utility RAW", "Utility ANS12"], [[int(s), round(r["util"]["SET12"]["seen"], 3), round(r["util"]["RAW"]["seen"], 3), round(r["util"]["ANS12"]["seen"], 3)] for s, r in per.items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    t0 = time.time()
    if mode == "selftest": r = one(C("x71_seed0")); print(json.dumps({a: r["util"][a] for a in ("SET12", "ANS12", "RAW", "CONF12", "SET1")}), r["sensor_in"], r["sensor_out"], r["junk_acc"]); print("selftest ok"); return
    if mode == "log": log_wandb(json.load(open(os.path.join(OUT, "x7_3_run.json")))["result"]); return
    from budget import pmap
    seeds = [C("x73_seed0") + i for i in range(C("x73_worlds"))]; res = pmap(one, [(s,) for s in seeds], C("x73_workers")); per = dict(zip(seeds, res))
    S = summarize(per); out = {"per_world": per, "summary": S, "verdicts": judge(S)}
    os.makedirs(OUT, exist_ok=True); json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x7_3_run.json"), "w"), indent=1, default=float)
    for v in out["verdicts"]: print(v[0], v[2], "|", v[3], flush=True)
    log_wandb(out)


if __name__ == "__main__":
    main(sys.argv[1])
