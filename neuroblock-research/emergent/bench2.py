"""bench2: the repaired benchmark (audit 2026-10-28, items V6/V7; curious-net paper/AUDIT_2026-10-28.md).

What was wrong in general.py (series 2/3, X6.5-X6.9) and what this file changes:
  * every run shared ONE dataset and the target came from the same test set          -> train / VALIDATION / TEST are three independent draws (problem seeds 0 / 1 / 2);
    every choice (target, stop, tuning) uses validation only; the test set is read ONCE, at the stop, to report.
  * evolution peeked at TEST accuracy every generation (640 cost) and Adam every 10 steps (960 cost); first passage rewards the arm that peeks more
                                                                                    -> both arms check VALIDATION accuracy on the same cost interval (x2b_interval = 960 cost units)
                                                                                       and must hit the target on TWO consecutive checks ('hold-2').
  * Adam was tuned on learning rate only (4 values, best at the grid edge), evolution over ~190 settings -> both arms get a grid of comparable size, tuned per problem by the SAME
    procedure (successive halving on TUNING seeds): Adam 7 lr x 3 batch sizes x 2 init gains = 42; evolution 3 culls x 3 (P,B) x 3 inits x mirror on/off = 54.
  * the headline was selected and scored on the same seeds 0-9                      -> tuning seeds 100-103 (never scored); scored seeds 800-819 (held out); bootstrap CIs.
  * censored runs were scored at the budget inside a geometric mean              -> per problem: solved fraction first; ratios only where both arms solve >= half the runs.
Cost accounting is unchanged (one sample through one net forward = 1; a backward pass = 2 more): Adam step = 3 x batch; evolution generation = P x B.
Usage: python bench2.py selftest | target | tune [workers] | score [workers] | report | log"""
import os, sys, json, math, time
import numpy as np
import torch
import multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import general as G
import car.compare as CP

RUNS = os.path.join(HERE, "runs"); os.makedirs(RUNS, exist_ok=True)
CONST = dict(interval=960, hold=2, budget_tune1=1_000_000, budget=3_000_000, ref_margin=0.02, tune_seeds=(100, 101, 102, 103), score_seeds=tuple(range(800, 820)), top_k=6, min_solved=0.5)
ADAM_GRID = [dict(lr=lr, bs=bs, gain=gn) for lr in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1) for bs in (16, 32, 128) for gn in (0.5, 1.0)]
EVO_GRID = [dict(P=P, B=B, cull=c, init=i, mirror=m) for c in (0.8, 0.9, 0.95) for (P, B) in ((40, 16), (80, 8), (160, 4)) for i in (0.25, 0.5, 1.0) for m in (False, True)]
ADAM_DEFAULT = dict(lr=1e-3, bs=32, gain=1.0); EVO_DEFAULT = dict(P=40, B=16, cull=0.8, init=0.5, mirror=False)
_TASKS = {}


class TaskV(G.NetTask):
    """NetTask with independent validation and test sets and a stateful, equal-interval, hold-2 stop on VALIDATION accuracy."""
    def __init__(s, prob, H=16):
        super().__init__(prob, H)
        X, y, Xt, yt, K, kind = CP.problem(prob, 0); X = np.asarray(X, float); mu, sd = X.mean(0), X.std(0) + 1e-9
        prep = (lambda A: 2 * np.asarray(A, float) - 1) if kind == "bits" else (lambda A: (np.asarray(A, float) - mu) / sd)
        f = lambda a: torch.tensor(a, dtype=torch.float32, device=G.dev)
        _, _, Xv, yv, _, _ = CP.problem(prob, 1); _, _, Xs, ys, _, _ = CP.problem(prob, 2)
        s.Xv, s.yv, s.Xs, s.ys = f(prep(Xv)), torch.tensor(yv, device=G.dev), f(prep(Xs)), torch.tensor(ys, device=G.dev)
        s.rule = "new"; s.target = None; s.test_at_stop = None

    def begin(s, gen_cost, rule="new"):
        s.gc, s.calls, s.rule, s.hits, s.next_check, s.test_at_stop = gen_cost, 0, rule, 0, CONST["interval"], None

    def done(s, th1):                                          # called by general.evolve once per generation, after the generation's cost is spent
        s.calls += 1; cost = s.calls * s.gc
        if s.rule == "old":                                    # the audited rule: TEST accuracy every generation, first passage
            ok = float(s.acc(th1, s.Xs, s.ys)[0]) >= s.target
        else:
            if cost < s.next_check: return False
            while cost >= s.next_check: s.next_check += CONST["interval"]
            s.hits = s.hits + 1 if float(s.acc(th1, s.Xv, s.yv)[0]) >= s.target else 0; ok = s.hits >= CONST["hold"]
        if ok: s.test_at_stop = float(s.acc(th1, s.Xs, s.ys)[0])
        return ok


def task(prob):
    if prob not in _TASKS: _TASKS[prob] = TaskV(prob)
    return _TASKS[prob]


def adam_run(t, seed, cfg, budget, rule="new", target=None, record_best=False):
    """returns (cost, solved, test accuracy at stop or at the end, best validation accuracy seen)"""
    target = t.target if target is None else target
    torch.manual_seed(20_000 + seed); d, Hh, K = t.d, t.H, t.K
    net = torch.nn.Sequential(torch.nn.Linear(d, Hh), torch.nn.Tanh(), torch.nn.Linear(Hh, K)).to(G.dev)
    if cfg["gain"] != 1.0:
        with torch.no_grad():
            for p in net.parameters(): p.mul_(cfg["gain"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"]); bs = cfg["bs"]; steps = int(budget // (3 * bs))
    every = 10 if rule == "old" else max(1, round(CONST["interval"] / (3 * bs))); hits = 0; best_v = 0.0
    for step in range(1, steps + 1):
        i = torch.randint(0, len(t.Xtr), (bs,), device=G.dev); l = torch.nn.functional.cross_entropy(net(t.Xtr[i]), t.ytr[i]); opt.zero_grad(); l.backward(); opt.step()
        if step % every == 0:
            with torch.no_grad():
                av = float((net(t.Xv).argmax(1) == t.yv).float().mean()); best_v = max(best_v, av)
                if target is None: continue
                ok = float((net(t.Xs).argmax(1) == t.ys).float().mean()) >= target if rule == "old" else None
                if rule != "old": hits = hits + 1 if av >= target else 0; ok = hits >= CONST["hold"]
                if ok: return 3 * step * bs, True, float((net(t.Xs).argmax(1) == t.ys).float().mean()), best_v
    return budget, False, None, best_v


def evo_run(t, seed, cfg, budget, rule="new"):
    t.begin(cfg["P"] * cfg["B"], rule); cost, solved, gen = G.evolve(t, dict(cfg), seed, budget)
    return (cost if solved else budget), bool(solved), t.test_at_stop if solved else None, None


def job(spec):
    arm, prob, seed, cfg, budget, rule, target = spec["arm"], spec["prob"], spec["seed"], spec["cfg"], spec["budget"], spec.get("rule", "new"), spec.get("target")
    t = task(prob); t.target = target
    r = adam_run(t, seed, cfg, budget, rule, target) if arm == "adam" else evo_run(t, seed, cfg, budget, rule)
    return dict(spec=dict(arm=arm, prob=prob, seed=seed, cfg=cfg, budget=budget, rule=rule, stage=spec.get("stage")), cost=r[0], solved=r[1], test=r[2], best_val=r[3])


def pmap(specs, n):
    if n <= 1: return [job(s) for s in specs]
    with mp.get_context("spawn").Pool(n) as p: return p.map(job, specs, chunksize=1)


def load(name): p = os.path.join(RUNS, name); return json.load(open(p)) if os.path.exists(p) else {}
def save(name, o): json.dump(o, open(os.path.join(RUNS, name), "w"), indent=1, default=float)
def key(cfg): return json.dumps(cfg, sort_keys=True)


def make_targets(n):
    """target[prob] = floor(( median over 3 tuning seeds of the best VALIDATION accuracy that Adam (lr 1e-2 / 3e-2, batch 32) reaches in the budget ) - margin, 0.005); validation only"""
    specs = [dict(arm="adam", prob=p, seed=s, cfg=dict(lr=lr, bs=32, gain=1.0), budget=CONST["budget"], target=None) for p in G.PROBLEMS for s in CONST["tune_seeds"][:3] for lr in (1e-2, 3e-2)]
    res = pmap(specs, n); tg = {}
    for p in G.PROBLEMS:
        best = [max(r["best_val"] for r in res if r["spec"]["prob"] == p and r["spec"]["seed"] == s) for s in CONST["tune_seeds"][:3]]
        tg[p] = math.floor((float(np.median(best)) - CONST["ref_margin"]) * 200) / 200
    save("bench2_targets.json", tg); return tg


def tune(n):
    """successive halving per arm and problem on TUNING seeds: stage 1 = every config, seed 100, budget 1e6; stage 2 = top 6, seeds 101-103, budget 3e6; choose by median cost (censored at budget)."""
    tg = load("bench2_targets.json"); out = load("bench2_tuned.json")
    for arm, grid in (("adam", ADAM_GRID), ("evo", EVO_GRID)):
        for p in G.PROBLEMS:
            k = f"{arm}|{p}"
            if k in out: continue
            s1 = pmap([dict(arm=arm, prob=p, seed=CONST["tune_seeds"][0], cfg=c, budget=CONST["budget_tune1"], target=tg[p], stage=1) for c in grid], n)
            rank = sorted(range(len(grid)), key=lambda i: (not s1[i]["solved"], s1[i]["cost"]))[:CONST["top_k"]]
            s2 = pmap([dict(arm=arm, prob=p, seed=s, cfg=grid[i], budget=CONST["budget"], target=tg[p], stage=2) for i in rank for s in CONST["tune_seeds"][1:]], n)
            med = {i: float(np.median([r["cost"] for r in s2 if key(r["spec"]["cfg"]) == key(grid[i])])) for i in rank}
            best = min(rank, key=lambda i: med[i]); out[k] = dict(cfg=grid[best], median_cost_tune=med[best], stage1_solved=int(sum(r["solved"] for r in s1)), n_configs=len(grid))
            save("bench2_tuned.json", out); print(k, out[k], flush=True)
    return out


def score(n):
    tg = load("bench2_targets.json"); tuned = load("bench2_tuned.json"); res = load("bench2_scored.json") or {"runs": []}; done = {json.dumps(r["spec"], sort_keys=True, default=float) for r in res["runs"]}
    specs = []
    for p in G.PROBLEMS:
        for s in CONST["score_seeds"]:
            for arm, dflt in (("adam", ADAM_DEFAULT), ("evo", EVO_DEFAULT)):
                for setting, cfg in (("tuned", tuned[f"{arm}|{p}"]["cfg"]), ("default", dflt)):
                    for rule in ("new",) + (("old",) if setting == "tuned" else ()):
                        sp = dict(arm=arm, prob=p, seed=s, cfg=cfg, budget=CONST["budget"], rule=rule, target=tg[p], stage=setting)
                        if json.dumps(dict(arm=arm, prob=p, seed=s, cfg=cfg, budget=CONST["budget"], rule=rule, stage=setting), sort_keys=True, default=float) not in done: specs.append(sp)
    for i in range(0, len(specs), 140):
        res["runs"] += pmap(specs[i:i + 140], n); save("bench2_scored.json", res); print("scored", len(res["runs"]), flush=True)
    return res


def boot(v, f, B=2000, seed=0):
    rng = np.random.default_rng(seed); v = np.asarray(v, float); return [float(np.percentile([f(rng.choice(v, v.size)) for _ in range(B)], q)) for q in (2.5, 97.5)]


def report():
    res = load("bench2_scored.json")["runs"]; rows = {}
    for r in res:
        sp = r["spec"]; rows.setdefault((sp["prob"], sp["arm"], sp["stage"], sp["rule"]), []).append(r)
    summ = {}
    for (p, arm, st, rule), rs in sorted(rows.items()):
        c = np.array([r["cost"] for r in rs], float); sol = np.array([r["solved"] for r in rs], bool)
        summ[f"{p}|{arm}|{st}|{rule}"] = dict(n=len(rs), solved=int(sol.sum()), median_cost=float(np.median(c)), median_cost_solved=float(np.median(c[sol])) if sol.any() else None,
                                           test_at_stop_median=float(np.median([r["test"] for r in rs if r["test"] is not None])) if sol.any() else None)
    ratios = {}
    for st in ("tuned", "default"):
        for rule in ("new", "old"):
            if rule == "old" and st == "default": continue
            per = {}
            for p in G.PROBLEMS:
                a, e = rows.get((p, "adam", st, rule)), rows.get((p, "evo", st, rule))
                if not a or not e: continue
                ca, ce = np.array([r["cost"] for r in a], float), np.array([r["cost"] for r in e], float)
                ok = (np.mean([r["solved"] for r in a]) >= CONST["min_solved"]) and (np.mean([r["solved"] for r in e]) >= CONST["min_solved"])
                per[p] = dict(ratio=float(np.median(ce) / np.median(ca)), ci=boot(np.arange(len(ca)), lambda ix: np.median(ce[ix.astype(int)]) / np.median(ca[ix.astype(int)])), both_solve_half=bool(ok))
            use = [v["ratio"] for v in per.values() if v["both_solve_half"]]
            ratios[f"{st}|{rule}"] = dict(per=per, geomean=float(np.exp(np.mean(np.log(use)))) if use else None, n_problems=len(use))
    # cross ratio (the audited claim 'evolution is faster than DEFAULT Adam') and the winner's-curse measure
    cross, per = {}, {}
    for p in G.PROBLEMS:
        a, e = rows.get((p, "adam", "default", "new")), rows.get((p, "evo", "tuned", "new"))
        if a and e and np.mean([r["solved"] for r in a]) >= CONST["min_solved"] and np.mean([r["solved"] for r in e]) >= CONST["min_solved"]:
            per[p] = float(np.median([r["cost"] for r in e]) / np.median([r["cost"] for r in a]))
    cross["evo_tuned_over_adam_default"] = dict(per=per, geomean=float(np.exp(np.mean(np.log(list(per.values()))))) if per else None)
    tuned = load("bench2_tuned.json"); curse = {}
    for arm in ("adam", "evo"):
        rs = [float(np.median([r["cost"] for r in rows[(p, arm, "tuned", "new")]])) / tuned[f"{arm}|{p}"]["median_cost_tune"] for p in G.PROBLEMS if (p, arm, "tuned", "new") in rows]
        curse[arm] = float(np.exp(np.mean(np.log(rs)))) if rs else None
    V = {}
    tn, to = ratios.get("tuned|new"), ratios.get("tuned|old")
    if tn and tn["geomean"]:
        used = [d for d in tn["per"].values() if d["both_solve_half"]]
        V["P282"] = (tn["geomean"] > 1.0 and sum(d["ci"][0] > 1.0 for d in used) > len(used) / 2, f"geometric-mean ratio {tn['geomean']:.2f} over {tn['n_problems']} problems; {sum(d['ci'][0] > 1.0 for d in used)} with the 95% range above 1")
    if cross["evo_tuned_over_adam_default"]["geomean"]: V["P283"] = (cross["evo_tuned_over_adam_default"]["geomean"] > 0.9, f"tuned evolution / default Adam = {cross['evo_tuned_over_adam_default']['geomean']:.2f}")
    if curse["evo"]: V["P284"] = (curse["evo"] >= 1.10, f"evolution held-out cost / tuning-seed cost = {curse['evo']:.2f} (Adam {curse['adam']:.2f})")
    if tn and to and tn["geomean"] and to["geomean"]:
        common = [p for p in tn["per"] if p in to["per"] and tn["per"][p]["both_solve_half"] and to["per"][p]["both_solve_half"]]
        gm = lambda r: float(np.exp(np.mean([np.log(r["per"][p]["ratio"]) for p in common]))) if common else None
        if common: V["P286"] = (gm(to) / gm(tn) <= 0.87, f"old rule / new rule ratio of ratios = {gm(to) / gm(tn):.2f} over {len(common)} problems")
    ce = rows.get(("checker3", "evo", "tuned", "new"))
    if ce: V["P287"] = (np.mean([r["solved"] for r in ce]) < 0.5, f"evolution solved {int(sum(r['solved'] for r in ce))}/{len(ce)} on checker3")
    out = dict(summary=summ, ratios=ratios, cross=cross, curse=curse, verdicts={k: dict(passed=bool(v[0]), evidence=v[1]) for k, v in V.items()}); save("bench2_report.json", out); return out


def selftest():
    t = task("moons"); X0 = CP.problem("moons", 0); X1 = CP.problem("moons", 1); X2 = CP.problem("moons", 2)
    ok = dict(val_differs_from_test=not np.array_equal(np.asarray(X1[2]), np.asarray(X2[2])), val_differs_from_old_test=not np.array_equal(np.asarray(X1[2]), np.asarray(X0[2])),
              sizes=(len(t.Xtr), len(t.Xv), len(t.Xs)))
    t.target = 0.80; t.begin(640); stops = [t.done(t.Xtr[:0].new_zeros((1, t.n))) for _ in range(5)]; ok["no_check_before_interval"] = (stops == [False] * 5)       # 5 gens = 3200 cost; check at 960, 1920, 2880: random net should not hit 0.8 twice
    ok["evo_checks_every_interval"] = (t.next_check == 3840)
    r1 = adam_run(t, 0, dict(lr=1e-2, bs=32, gain=1.0), 200_000, "new", 0.5); r2 = adam_run(t, 0, dict(lr=1e-2, bs=32, gain=1.0), 200_000, "new", 0.5)
    ok["adam_deterministic"] = (r1[0] == r2[0] and r1[1] == r2[1]); ok["adam_hold2_cost_multiple_of_interval"] = (r1[0] % (3 * 32 * max(1, round(960 / 96))) == 0)
    e1 = evo_run(t, 0, EVO_DEFAULT, 400_000, "new"); e2 = evo_run(t, 0, EVO_DEFAULT, 400_000, "new"); ok["evo_deterministic"] = (e1[0] == e2[0] and e1[1] == e2[1])
    ok["adam_example"] = r1[:3]; ok["evo_example"] = e1[:3]
    print(json.dumps(ok, indent=1, default=str)); return ok


def log_wandb():
    sys.path.insert(0, r"~/curious-net/vi")
    from track import Tracker
    rep = load("bench2_report.json"); tuned = load("bench2_tuned.json"); tg = load("bench2_targets.json")
    arms = {"adam": ("Gradient descent (Adam)", "The standard way of training the small network, tuned over 42 settings (learning rate, batch size, starting weights) per problem."),
            "evo": ("Evolution by variation and culling", "Many copies of the network are randomly perturbed and the worst are discarded; tuned over 54 settings per problem by the same procedure.")}
    tk = Tracker("BENCH2", "A fair race between gradient descent and evolution", "How much compute does evolution need compared with gradient descent to train the same small network to the same accuracy, when both are tuned equally and judged on data neither used?",
                 "Seven small classification problems. Each method trains the same 16-unit network until its accuracy on a separate VALIDATION set reaches a target on two checks in a row; checks happen at the same cost interval for both. The accuracy reported at the stop is on a third, untouched TEST set. Cost counts every sample passed through the network (a backward pass counts as two more). Settings were tuned on separate tuning seeds; the race is run on 20 held-out seeds.",
                 arms, "Ratio = evolution's median cost divided by Adam's median cost (above 1 means evolution is slower). 'Solved' counts runs that reached the target within 3 million cost units; ratios are only compared on problems where both methods solve at least half the runs.",
                 dict(tuning_seeds=list(CONST["tune_seeds"]), scored_seeds="800-819", budget=CONST["budget"], interval=CONST["interval"]), tags=["benchmark repair", "audit"])
    rows = []
    for k, v in rep["summary"].items():
        p, arm, st, rule = k.split("|"); rows.append([p, tk.name(arm), st, rule, v["solved"], v["n"], round(v["median_cost"]), v["test_at_stop_median"]])
    tk.table("Runs per problem (20 held-out seeds)", ["Problem", "Method", "Settings", "Stop rule (new = fair validation rule, old = audited test-peeking rule)", "Runs solved", "Runs", "Median cost", "Median test accuracy at the stop"], rows)
    rr = []
    for k, v in rep["ratios"].items():
        for p, d in v["per"].items(): rr.append([k, p, round(d["ratio"], 3), round(d["ci"][0], 3), round(d["ci"][1], 3), "yes" if d["both_solve_half"] else "no"])
        rr.append([k, "GEOMETRIC MEAN over problems both methods solve", v["geomean"], None, None, f"{v['n_problems']} problems"])
    tk.table("Evolution cost divided by Adam cost", ["Settings | rule", "Problem", "Ratio (above 1 = evolution slower)", "Lower end of 95% range", "Upper end", "Both methods solve at least half the runs (or number of problems used)"], rr)
    tk.table("Chosen settings", ["Method | problem", "Chosen setting", "Median cost on tuning seeds"], [[k, json.dumps(v["cfg"]), v["median_cost_tune"]] for k, v in tuned.items()])
    tk.table("Accuracy targets (validation accuracy)", ["Problem", "Target"], [[p, v] for p, v in tg.items()])
    tk.finish()


if __name__ == "__main__":
    mode = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    {"selftest": lambda: selftest(), "target": lambda: print(make_targets(n)), "tune": lambda: tune(n), "score": lambda: score(n), "report": lambda: print(json.dumps(report(), indent=1, default=float)[:3000]), "log": log_wandb}[mode]()
