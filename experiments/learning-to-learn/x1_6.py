"""X1.6: train a network on its ABILITY TO INCREASE COMPLEXITY (author's request, 2026-11-01; inbox I20).

What is trained: not the answer to one problem, but the network's LEARNING MACHINERY (Meta-SGD: a starting point theta0 plus one learnable
step size per weight). Every arm has the same inner learner, the same tasks (where the arm allows), the same compute; ONLY the outer signal differs.
"Complexity" is measured as STRUCTURE HELD: bits = 0.5*log2(Var(y)/MSE) on held-out points of a task (Var(y)=1 by construction), i.e. how
much of the task's real structure the network now encodes. Noise tasks hold 0 bits whatever the network does, so held structure cannot be faked.

World (ladder): x ~ N(0, I_m); a depth-d task is y = first coordinate of g_d(...g_1(x)), g_i(z) = phi_i(R_i z), R_i a random rotation, phi_i one of
{sin(bz), |z| - c, tanh(bz)} (b = x16_bend); y is standardised. Depth = complexity rung. Noise tasks: y independent of x. Training rungs 1-3; test rungs 1-5
(4-5 never trained on), all test tasks use FRESH random rotations.
Arms (outer objective; tasks per meta-batch carry a fixed noise share in every arm):
  FRESH    no outer training (the shared starting point; zero-experience parity)
  GOAL     reach a fixed goal: held-out error after K steps, rung-2 tasks only (U3's 'trained on a goal')
  MIX      held-out error after K steps, rungs 1-3 uniform (fairness control: diversity without the complexity objective)
  CLIMB_U  ability to increase complexity: mean held bits over the K steps (how fast structure is gained), same tasks as MIX
  CLIMB    as CLIMB_U, and the network chooses its own rung by learning progress (self-driven data)
  RAWCX    the trap: complexity WITHOUT reference to truth: bits of fit on the points it trained on (rewards memorising noise), same tasks as MIX
Usage: python x1_6.py selftest | run | log      (writes ../results/x1_6_<mode>.json)"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C

OUT = os.path.join(str(vipaths.RESULTS))
ARMS = ("FRESH", "GOAL", "MIX", "CLIMB_U", "CLIMB", "RAWCX")
LN2 = math.log(2.0)


# ---------------------------------------------------------------- world -------------------------------------------------------------
def rotation(rng, m):
    q, r = np.linalg.qr(rng.standard_normal((m, m)))
    return q * np.sign(np.diag(r))


def make_task(rng, depth, n_sup, n_qry):
    """Returns (xs, ys, xq, yq) float32 arrays. depth 0 = noise task."""
    m = C("x16_dim")
    n_norm = C("x16_norm_n")
    x = rng.standard_normal((n_sup + n_qry + n_norm, m))
    if depth == 0:
        y = rng.standard_normal(len(x))
    else:
        z = x
        for _ in range(depth):
            k = rng.integers(3)
            z = z @ rotation(rng, m).T
            z = np.sin(C("x16_bend") * z) if k == 0 else (np.abs(z) - C("x16_abs_c") if k == 1 else np.tanh(C("x16_bend") * z))
        y = z[:, 0]
        y = (y - y[n_sup + n_qry:].mean()) / (y[n_sup + n_qry:].std() + 1e-9)        # standardised on points never shown to the learner
    f = lambda a: a.astype(np.float32)
    return f(x[:n_sup]), f(y[:n_sup]), f(x[n_sup:n_sup + n_qry]), f(y[n_sup:n_sup + n_qry])


def batch(rng, depths):
    T = [make_task(rng, d, C("x16_n_sup"), C("x16_n_qry")) for d in depths]
    return [np.stack([t[i] for t in T]) for i in range(4)]


# ---------------------------------------------------------------- learner -----------------------------------------------------------
def init_params(torch, seed):
    """Shared starting point (identical in every arm and world-seed: zero-experience parity)."""
    g = torch.Generator().manual_seed(seed)
    m, h = C("x16_dim"), C("x16_hidden")
    shapes = [(m, h), (h,), (h, h), (h,), (h, 1), (1,)]
    P = []
    for s in shapes:
        w = torch.randn(*s, generator=g) / math.sqrt(s[0]) if len(s) == 2 else torch.zeros(*s)
        P.append(w.requires_grad_())
    L = [torch.full(s, math.log(C("x16_lr0"))).requires_grad_() for s in shapes]
    return P, L


def forward(torch, W, x):
    """W: list of per-task weights [T, ...]; x [T, n, m] -> [T, n]."""
    a = torch.tanh(torch.bmm(x, W[0]) + W[1][:, None, :])
    a = torch.tanh(torch.bmm(a, W[2]) + W[3][:, None, :])
    return (torch.bmm(a, W[4]) + W[5][:, None, :])[..., 0]   # CONST: x16_hidden  (index 4,5 = output layer of the 6-tensor list)


def adapt(torch, P, L, xs, ys, xq, yq, steps, second_order):
    """Inner loop: full-batch SGD with the learned per-weight step sizes. Returns per-step query MSE [steps+1, T] and support MSE [steps+1, T]."""
    T = xs.shape[0]
    W = [p.unsqueeze(0).expand(T, *p.shape) for p in P]
    lr = [l.exp().unsqueeze(0) for l in L]
    q, s = [], []
    for k in range(steps + 1):
        ls = ((forward(torch, W, xs) - ys) ** 2).mean(1)
        q.append(((forward(torch, W, xq) - yq) ** 2).mean(1)); s.append(ls)
        if k == steps: break
        G = torch.autograd.grad(ls.sum(), W, create_graph=second_order)
        W = [w - r * gr for w, r, gr in zip(W, lr, G)]
    return torch.stack(q), torch.stack(s)


def bits(torch, mse):
    return -0.5 * torch.log(mse + 1e-6) / LN2                                       # Var(y)=1: bits of structure held


def outer_loss(torch, arm, q, s):
    if arm in ("GOAL", "MIX"): return q[-1].mean()
    if arm in ("CLIMB_U", "CLIMB"): return -bits(torch, q[1:]).mean()
    if arm == "RAWCX": return -bits(torch, s[-1]).mean()
    raise ValueError(arm)


def train(seed, arm, iters):
    import torch
    torch.set_num_threads(1)
    P, L = init_params(torch, C("x16_init_seed"))
    if arm == "FRESH": return P, L, {}
    opt = torch.optim.Adam(P + L, lr=C("lr_default"))
    rng = np.random.default_rng([seed, 1 if arm in ("MIX", "CLIMB_U", "RAWCX") else 2 + ARMS.index(arm)])   # MIX, CLIMB_U, RAWCX see identical tasks
    nb, nn = C("x16_meta_batch"), C("x16_noise_per_batch")
    rungs = C("x16_train_rungs")
    fast = np.zeros(len(rungs)); slow = np.zeros(len(rungs)); eps = C("x16_lp_eps")
    choice = np.zeros(len(rungs))
    for it in range(iters):
        if arm == "GOAL": d = [C("x16_goal_rung")] * (nb - nn)
        elif arm == "CLIMB":
            lp = np.abs(fast - slow); p = (1 - eps) * (lp / lp.sum() if lp.sum() > 0 else np.full(len(rungs), 1 / len(rungs))) + eps / len(rungs)
            d = list(rng.choice(rungs, nb - nn, p=p))
        else: d = list(rng.choice(rungs, nb - nn))
        for r in d: choice[rungs.index(r)] += 1
        xs, ys, xq, yq = (torch.from_numpy(a) for a in batch(rng, d + [0] * nn))
        q, s = adapt(torch, P, L, xs, ys, xq, yq, C("x16_K"), True)
        loss = outer_loss(torch, arm, q, s)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for p in L: p.clamp_(math.log(C("x16_lr_min")), math.log(C("x16_lr_max")))
        if arm == "CLIMB":                                                           # learning progress per rung from its own held bits
            b = bits(torch, q[-1]).detach().numpy()
            for j, r in enumerate(rungs):
                hit = [b[i] for i in range(nb - nn) if d[i] == r]
                if hit:
                    v = float(np.mean(hit)); fast[j] += C("lp_fast") * (v - fast[j]); slow[j] += C("lp_slow") * (v - slow[j])
    return P, L, {"rung_share": (choice / choice.sum()).tolist()}


def evaluate(torch, P, L, seed):
    rng = np.random.default_rng([seed, C("x16_test_stream")])
    res = {}
    for d in (0,) + tuple(C("x16_test_rungs")):
        xs, ys, xq, yq = (torch.from_numpy(a) for a in batch(rng, [d] * C("x16_test_tasks")))
        q, s = adapt(torch, [p.detach().requires_grad_() for p in P], [l.detach() for l in L], xs, ys, xq, yq, C("x16_K"), False)
        bq, bs = bits(torch, q).detach().numpy(), bits(torch, s).detach().numpy()
        res[d] = {"held_end": float(bq[-1].mean()), "held_auc": float(bq[1:].mean()), "fit_end": float(bs[-1].mean()), "held_start": float(bq[0].mean())}
    return res


def one(seed, arm, iters):
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    P, L, info = train(seed, arm, iters)
    r = evaluate(torch, P, L, seed)
    lr_med = float(np.median(np.concatenate([l.detach().exp().numpy().ravel() for l in L])))
    return {"test": r, "info": info, "lr_median": lr_med, "seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- scoring -----------------------------------------------------------
def summarize(per, seed0):
    from x0_1 import boot_ci
    seeds = sorted(per["FRESH"])
    deep = C("x16_deep_rungs"); seen = C("x16_train_rungs")
    g = lambda a, rungs, k: np.array([np.mean([per[a][s]["test"][str(d) if str(d) in per[a][s]["test"] else d][k] for d in rungs]) for s in seeds])
    M = {}
    for a in per:
        M[a] = {"deep_held": g(a, deep, "held_end"), "seen_held": g(a, seen, "held_end"), "seen_auc": g(a, seen, "held_auc"),
                "noise_held": g(a, (0,), "held_end"), "noise_fit": g(a, (0,), "fit_end"),
                "off_goal_held": g(a, [d for d in C("x16_test_rungs") if d != C("x16_goal_rung")], "held_end")}
        for d in C("x16_test_rungs"): M[a][f"held_r{d}"] = g(a, (d,), "held_end")
    S = {"median": {a: {k: float(np.median(v)) for k, v in M[a].items()} for a in M}, "paired": {}}
    pr = lambda a, b, k: boot_ci(M[a][k] - M[b][k], seed0)
    for a, b, k in (("CLIMB_U", "MIX", "deep_held"), ("CLIMB_U", "MIX", "seen_auc"), ("CLIMB", "CLIMB_U", "deep_held"), ("CLIMB", "MIX", "deep_held"),
                    ("RAWCX", "MIX", "noise_fit"), ("RAWCX", "MIX", "seen_held"), ("GOAL", "MIX", "off_goal_held"), ("MIX", "FRESH", "seen_held"),
                    ("CLIMB_U", "MIX", "noise_held"), ("CLIMB_U", "FRESH", "deep_held")):
        S["paired"][f"{k} {a}-{b}"] = pr(a, b, k)
    S["climb_rung_share"] = np.median([per["CLIMB"][s]["info"]["rung_share"] for s in seeds], axis=0).tolist()
    S["lr_median"] = {a: float(np.median([per[a][s]["lr_median"] for s in seeds])) for a in per}
    S["ladder_drop"] = float(np.mean(M["FRESH"]["held_r1"] > M["FRESH"]["held_r5"]))
    return S


def judge(S):
    """Verdict rules fixed in code and registered in PREDICTIONS.md BEFORE the scored run."""
    m, pr = S["median"], S["paired"]
    f = lambda k: f"{pr[k][0]:+.3f} [{pr[k][1]:+.3f}, {pr[k][2]:+.3f}]"
    noise_max = max(m[a]["noise_held"] for a in m)
    manip = pr["seen_held MIX-FRESH"][1] > 0 and noise_max <= C("x16_noise_tol") and S["ladder_drop"] >= C("x16_ladder_share")
    V = [("M-X1.6", "Fair test: meta-training does something (MIX holds more structure than FRESH on rungs 1-3, paired range above 0); noise is unlearnable (no arm's median held bits on noise above tol; overfitting can only push it below 0); the ladder is real (FRESH holds less on rung 5 than rung 1 in >= 90% of worlds)",
          "PASSED" if manip else "FAILED (all other verdicts VOID)", f"MIX-FRESH {f('seen_held MIX-FRESH')}; max noise held {noise_max:+.3f}; ladder {S['ladder_drop']:.0%}")]

    def v(pid, stmt, ok, ev): V.append((pid, stmt, ("PASSED" if ok else "FAILED") if manip else "VOID", ev))
    v("P376", "Training on the ability to increase held structure beats training on error, at equal task diversity, on rungs NEVER trained on: CLIMB_U minus MIX in held bits on rungs 4-5, paired range above 0", pr["deep_held CLIMB_U-MIX"][1] > 0, f("deep_held CLIMB_U-MIX"))
    v("P377", "Self-chosen rungs add to it: CLIMB minus CLIMB_U in held bits on rungs 4-5, paired range above 0", pr["deep_held CLIMB-CLIMB_U"][1] > 0, f("deep_held CLIMB-CLIMB_U"))
    v("P378", "THE TRAP: complexity measured without reference to truth memorises noise: RAWCX fits noise tasks' training points better than MIX (range above 0) AND holds less real structure on rungs 1-3 (range below 0)", pr["noise_fit RAWCX-MIX"][1] > 0 and pr["seen_held RAWCX-MIX"][2] < 0, f"noise fit {f('noise_fit RAWCX-MIX')}; seen held {f('seen_held RAWCX-MIX')}")
    v("P379", "U3 reading: training on one fixed goal is single-minded: GOAL holds less structure than MIX on the rungs other than its goal rung (1, 3, 4, 5), range below 0", pr["off_goal_held GOAL-MIX"][2] < 0, f("off_goal_held GOAL-MIX"))
    v("P380", "SKEPTIC BAR: the held-structure objective does not buy depth by absorbing noise: CLIMB_U's held bits on noise tasks are not below MIX's by more than tol (lower end of the range >= -tol)", pr["noise_held CLIMB_U-MIX"][1] >= -C("x16_noise_tol"), f("noise_held CLIMB_U-MIX"))
    return V


# ---------------------------------------------------------------- reporting ---------------------------------------------------------
ARM_TEXT = {"FRESH": ("Untrained learner", "The shared starting point with no outer training. Every other arm starts exactly here."),
            "GOAL": ("Trained on one goal", "Its learning machinery is trained to reach low error on one fixed kind of problem (rung 2)."),
            "MIX": ("Trained on error, all rungs", "Trained to reach low error after its learning steps, on rungs 1-3 mixed (same problems as the next arm)."),
            "CLIMB_U": ("Trained to grow structure", "Trained on how fast it gains real structure (bits held on unseen points, averaged over its learning steps), rungs 1-3 mixed."),
            "CLIMB": ("Grows structure, picks its own rung", "As above, and it chooses which rung to practise by its own learning progress."),
            "RAWCX": ("Trained to grow raw complexity (trap)", "Trained to fit the points it saw as tightly as possible; this rewards memorising noise.")}
WORLD_TEXT = ("Each problem is a hidden rule built by stacking simple twists (rotate, then bend) on top of each other; the number of stacked twists is "
              "the problem's rung on a complexity ladder. Some problems are pure noise with nothing to learn. The learner sees 100 examples, takes 10 "
              "learning steps, and is scored on examples it never saw. Training uses rungs 1-3; the test adds rungs 4-5 and always uses brand-new rules.")
READ_TEXT = ("'Bits held' = how much of a problem's real structure the learner encodes after learning, measured on unseen examples (0 = knows "
             "nothing, higher is better; noise problems can only hold 0). 'Fit on seen points' counts how tightly it hugs the examples it trained on, "
             "which can include memorised noise.")


def log_wandb(out):
    from track import Tracker
    S, per = out["summary"], out["per_world"]
    tk = Tracker("X1.6", "Training a network on its ability to increase complexity",
                 "If a network's learning machinery is trained on how fast it gains real structure, instead of on reaching a goal, does it climb higher on a complexity ladder it was never trained on? And does 'complexity' without reference to truth become a trap?",
                 WORLD_TEXT, ARM_TEXT, READ_TEXT, dict(worlds=C("x16_worlds"), iters=C("x16_iters"), K=C("x16_K")), tags=["complexity", "meta-learning", "drive"])
    keys = [("seen_held", "Bits held, rungs 1-3 (higher is better)"), ("deep_held", "Bits held, rungs 4-5 never trained on (higher is better)"),
            ("seen_auc", "Speed of gaining structure, rungs 1-3 (mean bits over learning steps)"), ("noise_fit", "Fit on seen noise points, bits (memorisation)"),
            ("noise_held", "Bits held on noise (should be 0)")]
    tk.table("Summary by arm (median over worlds)", ["Arm"] + [k[1] for k in keys], [[tk.name(a)] + [round(S["median"][a][k], 4) for k, _ in keys] for a in ARMS])
    tk.table("Bits held per rung (median)", ["Arm"] + [f"Rung {d}" for d in C("x16_test_rungs")], [[tk.name(a)] + [round(S["median"][a][f"held_r{d}"], 4) for d in C("x16_test_rungs")] for a in ARMS])
    tk.table("Paired differences with 95% ranges (same worlds, A minus B)", ["Comparison", "Median difference", "Lower end", "Upper end"], [[k, *[round(x, 4) for x in v]] for k, v in S["paired"].items()])
    seeds = sorted(per["FRESH"])
    tk.table("Results per world", ["World seed", "Arm"] + [f"Bits held, rung {d}" for d in C("x16_test_rungs")] + ["Noise fit"],
             [[s, tk.name(a)] + [round(per[a][s]["test"][str(d)]["held_end"], 4) for d in C("x16_test_rungs")] + [round(per[a][s]["test"]["0"]["fit_end"], 4)] for s in seeds for a in ARMS])
    tk.verdicts(out["verdicts"]); tk.finish()


def main(mode):
    import budget
    t0 = time.time()
    if mode == "log":
        log_wandb(json.load(open(os.path.join(OUT, "x1_6_run.json")))["result"]); return
    seed0 = C("x16_selftest_seed") if mode == "selftest" else C("x16_seed0")
    nw = C("x16_selftest_worlds") if mode == "selftest" else C("x16_worlds")
    jobs = [(seed0 + i, a, C("x16_iters")) for i in range(nw) for a in ARMS]
    res = budget.pmap(one, jobs)
    per = {a: {} for a in ARMS}
    for (s, a, _), r in zip(jobs, res):
        r["test"] = {str(k): v for k, v in r["test"].items()}; per[a][s] = r
    S = summarize(per, seed0); out = {"per_world": per, "summary": S, "verdicts": judge(S)}
    os.makedirs(OUT, exist_ok=True)
    json.dump({"mode": mode, "seconds": round(time.time() - t0, 1), "result": out}, open(os.path.join(OUT, f"x1_6_{mode}.json"), "w"), indent=1, default=float)
    print(json.dumps({"median": S["median"], "climb_rung_share": S["climb_rung_share"], "lr_median": S["lr_median"]}, indent=1, default=float))
    if mode == "run": log_wandb(out)
    for v in out["verdicts"]: print(v[0], v[2], "|", v[3])


if __name__ == "__main__":
    main(sys.argv[1])
