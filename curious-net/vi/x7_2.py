"""X7.2: a Self that reads sensor networks' COMPLETE internal states (any architecture, via hooks) plus their answers, and decides to go with a sensor, blend them, or override.
Registered in paper/PREDICTIONS.md (P300-P306) before this was run. The state reader is x7_1.Mirror (forward hooks on every leaf module).
Usage: python x7_2.py selftest | run | log        (writes ../results/x7_2_<mode>.json)"""
import sys, os, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x0_1 as X
from x7_1 import Mirror, Conv, GRUHost, TfHost

OUT = X.OUT
SENSORS = ("MLP", "CONV1D", "GRU", "TRANSFORMER", "MLP")
RULES = ("BEST_SINGLE", "MAJORITY", "CONF_PICK", "ORACLE")
SELVES = ("SELF_OUT", "SELF_STATE", "SELF_NOISE", "SELF_X", "SELF_X_OUT")
ARMS = RULES + SELVES


class World:
    def __init__(s, seed):
        g = np.random.default_rng(seed); I, R = C("x71_in_dim"), C("x71_regimes")
        s.protos = g.normal(size=(R, I)); s.Q = [np.linalg.qr(g.normal(size=(I, I)))[0] for _ in range(C("x72_contexts"))]

    def sample(s, rng, n, ctxs):
        c = rng.choice(ctxs, n); y = rng.integers(0, C("x71_regimes"), n)
        x = np.einsum("nij,nj->ni", np.stack([s.Q[k] for k in c]), s.protos[y] + C("x72_class_sd") * rng.normal(size=(n, C("x71_in_dim"))))
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y), c


def make_sensor(fam, seed):
    torch.manual_seed(seed); I, R, H = C("x71_in_dim"), C("x71_regimes"), C("x71_hidden")
    return {"MLP": lambda: nn.Sequential(nn.Linear(I, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, R)), "CONV1D": Conv, "GRU": GRUHost, "TRANSFORMER": TfHost}[fam]()


def train_sensor(m, W, ctxs, seed):
    opt = torch.optim.Adam(m.parameters(), lr=C("x71_lr")); rng = np.random.default_rng(seed)
    for _ in range(C("x72_sensor_steps")):
        x, y, _ = W.sample(rng, C("x71_host_batch"), ctxs); loss = F.cross_entropy(m(x), y); opt.zero_grad(); loss.backward(); opt.step()
    return m


class SelfNet(nn.Module):
    """The Self: identical for every arm. Gate over [sensor 1..K, own]; own class distribution; answer = gated mixture."""
    def __init__(s, d_in):
        super().__init__(); K, R, H = C("x72_sensors"), C("x71_regimes"), C("x71_hidden"); s.K = K
        s.net = nn.Sequential(nn.Linear(d_in, H), nn.Tanh(), nn.Linear(H, H), nn.Tanh(), nn.Linear(H, K + 1 + R))

    def forward(s, f, P):                                                    # f (n,d_in), P (n,K,R) sensors' class probabilities
        o = s.net(f); g = F.softmax(o[:, :s.K + 1], -1); own = F.softmax(o[:, s.K + 1:], -1)
        return (g[:, :s.K, None] * P).sum(1) + g[:, s.K:] * own


def fit_self(f, P, y, seed):
    torch.manual_seed(seed); m = SelfNet(f.shape[1]); opt = torch.optim.Adam(m.parameters(), lr=C("x71_lr")); rng = np.random.default_rng(seed)
    for _ in range(C("x72_self_iters")):
        i = torch.tensor(rng.integers(0, len(f), C("x71_host_batch"))); loss = -torch.log(m(f[i], P[i])[torch.arange(len(i)), y[i]] + 1e-9).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return m


def metrics(p, y):
    return float((p.argmax(1) == y).float().mean()), float(-torch.log(p[torch.arange(len(y)), y] + 1e-9).mean())


def one(seed):
    torch.set_num_threads(1); K, R = C("x72_sensors"), C("x71_regimes"); W = World(seed); rs = lambda o: np.random.default_rng(seed * 10 + o)
    seen, nov = list(range(C("x72_seen_ctx"))), [C("x72_contexts") - 1]
    sens, mirs, cov = [], [], []
    for j, fam in enumerate(SENSORS):
        cj = [j % C("x72_seen_ctx"), (j + 1) % C("x72_seen_ctx")]; m = train_sensor(make_sensor(fam, seed + j), W, cj, seed * 10 + j); sens.append(m); mirs.append(Mirror(m, seed + j)); cov.append(cj)
    sets = {"train": W.sample(rs(1), C("x72_self_n"), seen), "seen": W.sample(rs(2), C("x72_test_n"), seen), "novel": W.sample(rs(3), C("x72_test_n"), nov)}
    feats = {}                                                                # per set: probs (n,K,R), states (n,sum D), x
    for k, (x, y, c) in sets.items():
        with torch.no_grad():
            sg = [mirs[j].read(x) for j in range(K)]; P = torch.stack([F.softmax(sens[j](x), -1) for j in range(K)], 1)
        feats[k] = (P, torch.cat(sg, 1), x)
    mu, sd = feats["train"][1].mean(0), feats["train"][1].std(0) + 1e-6; Wd = feats["train"][1].shape[1]
    inputs = {"SELF_OUT": lambda P, S, x: P.flatten(1), "SELF_STATE": lambda P, S, x: torch.cat([P.flatten(1), (S - mu) / sd], 1),
              "SELF_NOISE": lambda P, S, x: torch.cat([P.flatten(1), C("x72_noise_w") * torch.randn(len(P), Wd)], 1), "SELF_X": lambda P, S, x: x, "SELF_X_OUT": lambda P, S, x: torch.cat([x, P.flatten(1)], 1)}
    Ptr, Str, xtr = feats["train"]; ytr = sets["train"][1]; out = {"acc": {}, "ll": {}}
    acc_tr = [(Ptr[:, j].argmax(1) == ytr).float().mean() for j in range(K)]; best = int(np.argmax(acc_tr))
    selves = {a: fit_self(inputs[a](Ptr, Str, xtr), Ptr, ytr, seed) for a in SELVES}
    pred = {}
    for k in ("seen", "novel"):
        P, S, x = feats[k]; y = sets[k][1]; ps = {}
        ps["BEST_SINGLE"] = P[:, best]; ps["MAJORITY"] = P.mean(1); ps["CONF_PICK"] = P[torch.arange(len(y)), P.max(2).values.argmax(1)]
        with torch.no_grad():
            for a in SELVES: ps[a] = selves[a](inputs[a](P, S, x), P)
        for a, p in ps.items(): out["acc"][f"{a}|{k}"], out["ll"][f"{a}|{k}"] = metrics(p, y)
        out["acc"][f"ORACLE|{k}"] = float(((P.argmax(2) == y[:, None]).any(1)).float().mean()); pred[k] = ps
    ysn = sets["seen"][1]; c_seen = sets["seen"][2]; a_in, a_out = [], []                                  # sensor reliability inside / outside the contexts it was trained on
    for j in range(K):
        ok = (feats["seen"][0][:, j].argmax(1) == ysn).numpy(); inside = np.isin(c_seen, cov[j]); a_in.append(ok[inside].mean()); a_out.append(ok[~inside].mean())
    out["sensor_in"], out["sensor_out"] = float(np.mean(a_in)), float(np.mean(a_out))
    ps = pred["seen"]; ch = ps["SELF_STATE"].argmax(1) != ps["CONF_PICK"].argmax(1); right_s, right_c = ps["SELF_STATE"].argmax(1) == ysn, ps["CONF_PICK"].argmax(1) == ysn
    fixes, breaks = int((ch & right_s & ~right_c).sum()), int((ch & ~right_s & right_c).sum())
    out["overrides"], out["fixes"], out["breaks"] = int(ch.sum()), fixes, breaks
    return out


def summarize(per):
    seeds = sorted(per); g = lambda f: np.array([f(per[s]) for s in seeds], float); ci = lambda d: X.boot_ci(d, C("x72_seed0"))
    A = lambda a, k="seen": g(lambda r: r["acc"][f"{a}|{k}"]); L = lambda a, k="seen": g(lambda r: r["ll"][f"{a}|{k}"]); S = {"paired": {}, "median_acc": {a: float(np.median(A(a))) for a in ARMS}, "median_acc_novel": {a: float(np.median(A(a, "novel"))) for a in ARMS},
                                                                                                           "median_ll": {a: float(np.median(L(a))) for a in ARMS if a != "ORACLE"}, "median_ll_novel": {a: float(np.median(L(a, "novel"))) for a in ARMS if a != "ORACLE"}}
    rule = np.max([A(a) for a in ("BEST_SINGLE", "MAJORITY", "CONF_PICK")], 0); S["paired"]["acc SELF_STATE - best rule"] = ci(A("SELF_STATE") - rule)
    for b in ("SELF_OUT", "SELF_NOISE", "SELF_X", "SELF_X_OUT"): S["paired"][f"acc SELF_STATE - {b}"] = ci(A("SELF_STATE") - A(b))
    for b in ("CONF_PICK", "BEST_SINGLE"): S["paired"][f"novel log loss SELF_STATE - {b}"] = ci(L("SELF_STATE", "novel") - L(b, "novel"))
    S["in_out"] = (float(np.median(g(lambda r: r["sensor_in"]))), float(np.median(g(lambda r: r["sensor_out"])))); S["median_oracle"] = float(np.median(A("ORACLE"))); S["median_best_single"] = float(np.median(A("BEST_SINGLE")))
    fx, bk, ov = g(lambda r: r["fixes"]), g(lambda r: r["breaks"]), g(lambda r: r["overrides"]); S["net_fixes_median"] = float(np.median(fx - bk)); S["precision_median"] = float(np.median(fx / np.maximum(fx + bk, 1))); S["overrides_median"] = float(np.median(ov))
    S["gap_closed"] = float(np.median((A("SELF_STATE") - rule) / np.maximum(A("ORACLE") - rule, 1e-9)))
    return S


def judge(S):
    p = S["paired"]; r = lambda k: [round(x, 4) for x in p[k]]; pos = lambda k: p[k][1] > 0; neg = lambda k: p[k][2] <= 0
    return [("P300", "Sensors are fallible by context and there is room to improve", "PASSED" if S["in_out"][1] <= C("x72_unseen_ratio") * S["in_out"][0] and S["median_oracle"] >= C("x72_oracle_bar") and S["median_best_single"] <= C("x72_room_bar") else "VOID",
             f"sensor accuracy inside/outside its contexts {S['in_out'][0]:.2f}/{S['in_out'][1]:.2f}; oracle {S['median_oracle']:.3f}; best single {S['median_best_single']:.3f}"),
            ("P301", "The Self beats every rule", "PASSED" if pos("acc SELF_STATE - best rule") else "FAILED", f"SELF_STATE minus best rule {r('acc SELF_STATE - best rule')}; median accuracy {S['median_acc']}"),
            ("P302", "The sensors' insides matter beyond their answers", "PASSED" if pos("acc SELF_STATE - SELF_OUT") and pos("acc SELF_STATE - SELF_NOISE") else "FAILED", f"vs answers-only {r('acc SELF_STATE - SELF_OUT')}; vs noise-padded {r('acc SELF_STATE - SELF_NOISE')}"),
            ("P303", "The Self is checking, not re-solving", "PASSED" if pos("acc SELF_STATE - SELF_X") and pos("acc SELF_STATE - SELF_X_OUT") else "FAILED", f"vs raw-input-only {r('acc SELF_STATE - SELF_X')}; vs input+answers {r('acc SELF_STATE - SELF_X_OUT')}"),
            ("P304", "Overrides are net-positive", "PASSED" if S["net_fixes_median"] > 0 and S["precision_median"] > 0.5 else "FAILED", f"median overrides {S['overrides_median']:.0f}, net fixes {S['net_fixes_median']:.0f}, precision {S['precision_median']:.2f}"),
            ("P305", "In the context nobody saw, the Self is no worse calibrated than the rules", "PASSED" if neg("novel log loss SELF_STATE - CONF_PICK") and neg("novel log loss SELF_STATE - BEST_SINGLE") else "FAILED", f"vs CONF_PICK {r('novel log loss SELF_STATE - CONF_PICK')}; vs BEST_SINGLE {r('novel log loss SELF_STATE - BEST_SINGLE')}; median novel log loss {S['median_ll_novel']}"),
            ("P306", "The Self closes at least half the gap between the best rule and the oracle", "PASSED" if S["gap_closed"] >= C("x72_gap_bar") else "FAILED", f"median share of gap closed {S['gap_closed']:.2f}")]


ARM_TEXT = {"BEST_SINGLE": ("Best single sensor", "Always trust the one sensor that did best on the Self's labelled set."), "MAJORITY": ("Average of sensors", "Average the sensors' probabilities."),
            "CONF_PICK": ("Most confident sensor", "Trust whichever sensor is most sure on this case."), "ORACLE": ("Ceiling: any sensor right", "Counts a case as right if any sensor was right. Not achievable; shows the room."),
            "SELF_OUT": ("Self sees answers only", "Supervisor that sees only the sensors' answers."), "SELF_STATE": ("Self sees answers and insides", "The proposal: sees every sensor's complete internal state and answer."),
            "SELF_NOISE": ("Self sees answers plus noise", "Control: same width as the insides, but meaningless."), "SELF_X": ("Self sees the raw input only", "Control: just re-solves the task from the same labelled examples."),
            "SELF_X_OUT": ("Self sees input and answers", "Control: raw input plus the sensors' answers, no insides.")}


def log_wandb(out):
    from track import Tracker
    S, per = out["summary"], out["per_world"]
    tk = Tracker("X7.2", "A Self that double-checks sensor networks from the inside", "If a small Self reads five sensor networks' complete internal states and answers and is trained only on being right, can it learn whom to trust, when to blend and when to override, and do the insides tell it more than the answers?",
                 "Four classes. Five hidden contexts that distort the input. Each of five networks (MLP, conv, GRU, transformer, MLP) was trained on only two of four contexts, so each is confidently wrong elsewhere. A fifth context is unknown to everyone. The Self never sees the raw input or the context in the main arm.", ARM_TEXT,
                 "Accuracy higher is better; log loss lower is better (it punishes confident mistakes). Paired differences compare arms on the same worlds.", dict(worlds=len(per)), tags=["self", "sensors", "any-network"])
    tk.table("Summary (median over worlds)", ["Arm", "Accuracy, seen contexts", "Accuracy, novel context", "Log loss seen (lower is better)", "Log loss novel (lower is better)"], [[tk.name(a), round(S["median_acc"][a], 3), round(S["median_acc_novel"][a], 3), round(S["median_ll"].get(a, float("nan")), 3), round(S["median_ll_novel"].get(a, float("nan")), 3)] for a in ARMS])
    tk.table("Paired differences with 95% ranges", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    tk.table("Results per world", ["World seed"] + [f"{tk.name(a)} accuracy" for a in ARMS], [[int(s)] + [round(r["acc"][f"{a}|seen"], 3) for a in ARMS] for s, r in per.items()])
    tk.verdicts(out["verdicts"]); tk.finish()


def selftest():
    r = one(C("x71_seed0")); print(json.dumps({k: r[k] for k in ("sensor_in", "sensor_out", "overrides", "fixes", "breaks")}), {a: round(r["acc"][f"{a}|seen"], 3) for a in ARMS}); print("selftest ok")


def main(mode):
    t0 = time.time()
    if mode == "selftest": selftest(); return
    if mode == "log": log_wandb(json.load(open(os.path.join(OUT, "x7_2_run.json")))["result"]); return
    from budget import pmap
    seeds = [C("x72_seed0") + i for i in range(C("x72_worlds"))]; res = pmap(one, [(s,) for s in seeds], C("x72_workers")); per = dict(zip(seeds, res))
    S = summarize(per); out = {"per_world": per, "summary": S, "verdicts": judge(S)}
    os.makedirs(OUT, exist_ok=True); json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, "x7_2_run.json"), "w"), indent=1, default=float)
    for v in out["verdicts"]: print(v[0], v[2], "|", v[3])
    log_wandb(out)


if __name__ == "__main__":
    main(sys.argv[1])
