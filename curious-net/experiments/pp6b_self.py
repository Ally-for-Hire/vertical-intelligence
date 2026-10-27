"""PIECE-P6b: Self v0.1 with a LEARNED ROUTER (same world and baselines as P6).
Self v0.1: parts are 2 -> 8 -> 1 nets. After each label, every part's error is computed (responsibility). The sample is EXPLAINED if the best part's error <= SPAWN x the median settled baseline of the parts;
an explained sample trains only the best part (winner-take-all, hard EM); an UNEXPLAINED sample trains the newest part (novelty goes to the newcomer). A new part is spawned when, over the last 100 samples,
>= 30% were unexplained and the newest part has settled (>= 400 samples). Router: softmax over parts on 64 fixed random Fourier features of x, trained online to predict which part had the lowest error;
predictions use the router's choice. SPAWN = 2, 30% and 400 are EXPERIMENT settings.
PREDICTIONS (registered before running): P82 Self v0.1 ends with 6 +- 2 parts (median); P83 final test error <= 0.5 x the wide net; P84 final test error <= 3 x the oracle."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pp6_self as P6

SPAWN, UNEXP, SETTLE, NF, RLR = 2.0, 0.3, 400, 64, 0.05


def run(args):
    from nn import AdamMLP
    seed, SPW, UNX, FIXED = args if isinstance(args, tuple) else (args, SPAWN, UNEXP, 0)
    base = P6.run(seed)
    rng = np.random.default_rng(seed); region, f = P6.world(rng)
    xs, ys = [], []
    for t in range(P6.T):
        x = P6.sample_x(rng, min(P6.NR, 1 + t // P6.UNLOCK), region); xs.append(x); ys.append(f(x) + 0.05 * rng.normal())
    test = [P6.sample_x(rng, P6.NR, region) for _ in range(2000)]; ty = np.array([f(x) for x in test])
    frng = np.random.default_rng(seed + 4242); Wf = frng.normal(0, 3, (2, NF)); bf = frng.uniform(0, 2 * np.pi, NF)
    phi = lambda x: np.cos(x @ Wf + bf) * np.sqrt(2 / NF)
    n0 = FIXED if FIXED else 1
    parts = [AdamMLP(2, 1, P6.HP, P6.LR, seed * 100 + i) for i in range(n0)]; nseen = [0] * n0; errs_hist = [[] for _ in range(n0)]; base_err = [None] * n0
    R = np.zeros((n0, NF)); assign = np.zeros((40, P6.NR)); unexp = []; pe = []; forward = 0
    for t in range(P6.T):
        x, y = xs[t], ys[t]; ph = phi(x)
        logits = R @ ph; pr_ = np.exp(logits - logits.max()); pr_ /= pr_.sum()
        k_route = int(np.argmax(pr_))
        preds = np.array([p.predict(x)[0][0] for p in parts]); forward += len(parts)
        e_all = (preds - y) ** 2
        pe.append(float(e_all[k_route]))
        best = int(np.argmin(e_all))
        settled = [b for b in base_err if b is not None]
        thr = SPW * float(np.median(settled)) if settled else np.inf
        explained = e_all[best] <= thr or FIXED > 0
        k_train = best if explained else len(parts) - 1
        parts[k_train].step(x, np.array([y])); nseen[k_train] += 1; assign[k_train, region(x)] += 1; errs_hist[k_train].append(float(e_all[k_train]))
        if base_err[k_train] is None and nseen[k_train] == SETTLE:
            base_err[k_train] = float(np.mean(errs_hist[k_train][SETTLE // 2:SETTLE]))
        target = np.zeros(len(parts)); target[best] = 1.0
        R += RLR * np.outer(target - pr_, ph)                            # softmax router: online cross-entropy step
        unexp.append(not explained); unexp = unexp[-100:]
        if not FIXED and len(parts) < 40 and len(unexp) == 100 and np.mean(unexp) >= UNX and nseen[-1] >= SETTLE:
            parts.append(AdamMLP(2, 1, P6.HP, P6.LR, seed * 100 + len(parts))); nseen.append(0); errs_hist.append([]); base_err.append(None)
            R = np.vstack([R, np.zeros(NF)]); unexp = []
    pt = []; hit = 0
    owner = assign[:len(parts)].argmax(1)
    for x in test:
        k = int(np.argmax(R @ phi(x))); pt.append(parts[k].predict(x)[0][0]); hit += int(owner[k] == region(x))
    purity = float(assign[:len(parts)].max(1).sum() / max(assign.sum(), 1))
    base["self_v01"] = dict(test=float(np.mean((np.array(pt) - ty) ** 2)), preq=float(np.mean(pe)), parts=len(parts), compute=float(P6.T * P6.HP), forward=forward, purity=purity, route_acc=hit / len(test))
    return base


if __name__ == "__main__" and os.environ.get("P6C"):
    t0 = time.time(); allr = {}
    configs = [("fixed6", SPAWN, UNEXP, 6), ("spawn1.5_u0.3", 1.5, 0.3, 0), ("spawn2_u0.1", 2.0, 0.1, 0), ("spawn3_u0.3", 3.0, 0.3, 0), ("spawn1.5_u0.1", 1.5, 0.1, 0)]
    with Pool(20) as p:
        for name, sp, ux, fx in configs:
            res = p.map(run, [(s, sp, ux, fx) for s in range(48)])
            w = np.array([r["wide"]["test"] for r in res]); o = np.array([r["oracle"]["test"] for r in res]); v = np.array([r["self_v01"]["test"] for r in res])
            allr[name] = [r["self_v01"] for r in res]
            print(f"{name:14s} | test {np.median(v):.4f} | x wide {np.median(v / w):.2f} | x oracle {np.median(v / o):.1f} | parts {np.median([r['self_v01']['parts'] for r in res]):.0f} | training purity {np.median([r['self_v01']['purity'] for r in res]):.2f} | test routing to the right specialist {np.median([r['self_v01']['route_acc'] for r in res]):.2f}")
    json.dump(allr, open("../results/pp6c_self.json", "w"), default=float)
    print("done", round(time.time() - t0), "s"); raise SystemExit
if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(48))
    json.dump(res, open("../results/pp6b_self.json", "w"), default=float)
    w = np.array([r["wide"]["test"] for r in res]); o = np.array([r["oracle"]["test"] for r in res])
    print("arm      | final test error (median) | ratio to wide | ratio to oracle | parts (median, range)")
    for a in ("small", "wide", "oracle", "self", "self_v01"):
        v = np.array([r[a]["test"] for r in res]); pts = [r[a]["parts"] for r in res]
        print(f"{a:8s} | {np.median(v):.4f} | {np.median(v / w):.2f} | {np.median(v / o):.2f} | {np.median(pts):.0f} ({min(pts)}-{max(pts)})")
    print("self_v01 forward passes per tick (responsibility evaluation): %.1f" % (np.mean([r['self_v01']['forward'] for r in res]) / 6000))
    print("done", round(time.time() - t0), "s")
