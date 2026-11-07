"""PIECE-CTX: latent-context inference, the shared bottleneck of P5/P6 (C45). World and baselines as P6 (6 hidden angular regions unlocking over time, 6,000 ticks, 48 seeds). All Self variants have 6 parts (2 -> 8 -> 1)
and a softmax router on 64 random Fourier features of x, trained online to predict the training assignment; test predictions use the router.
(a) CUE DOSE-RESPONSE: on a fraction q of ticks the world shows the true region (a 'context cue'); on those ticks the cue decides the training assignment and the router target; on the rest the
    best-predicting part does (winner-take-all). q in {0, 0.1, 0.25, 0.5, 1}. Measures how much context information is worth.
(b) SOFT, ANNEALED ASSIGNMENT (no cue): every part trains on every sample with weight r_k = softmax(-err_k / tau) (its learning rate scaled by r_k); tau starts at the median error and shrinks x0.999 per tick
    (soft at first, hard later), the standard remedy for winner-take-all lock-in in mixtures (deterministic annealing, Ueda & Nakano 1998 [u]).
PREDICTIONS (registered before running): P88 test error falls monotonically with cue fraction q across the 5 levels; P89 at q = 0.25 test error <= 3 x oracle (a quarter of labelled moments is enough for the router to learn the regions);
P90 soft annealed assignment (no cue) reaches <= 0.7 x the wide net."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import pp6_self as P6

NF, RLR, QS = 64, 0.05, (0.0, 0.1, 0.25, 0.5, 1.0)


def run(seed):
    from nn import AdamMLP
    base = P6.run(seed)
    rng = np.random.default_rng(seed); region, f = P6.world(rng)
    xs, ys = [], []
    for t in range(P6.T):
        x = P6.sample_x(rng, min(P6.NR, 1 + t // P6.UNLOCK), region); xs.append(x); ys.append(f(x) + 0.05 * rng.normal())
    test = [P6.sample_x(rng, P6.NR, region) for _ in range(2000)]; ty = np.array([f(x) for x in test])
    frng = np.random.default_rng(seed + 4242); Wf = frng.normal(0, 3, (2, NF)); bf = frng.uniform(0, 2 * np.pi, NF)
    phi = lambda x: np.cos(x @ Wf + bf) * np.sqrt(2 / NF)
    out = dict(wide=base["wide"]["test"], oracle=base["oracle"]["test"])
    for mode in ([f"cue{q}" for q in (0.0, 0.25)] + ["ask10", "askdis"]) if os.environ.get("ASK") else ([f"cue{q}" for q in QS] + ["soft"]):
        crng = np.random.default_rng(seed + 777)
        parts = [AdamMLP(2, 1, P6.HP, P6.LR, seed * 100 + i) for i in range(P6.NR)]; R = np.zeros((P6.NR, NF))
        tau = None; margins = []; asks = 0
        for t in range(P6.T):
            x, y = xs[t], ys[t]; ph = phi(x)
            lg = R @ ph; pr_ = np.exp(lg - lg.max()); pr_ /= pr_.sum()
            e_all = np.array([(p.predict(x)[0][0] - y) ** 2 for p in parts])
            if mode == "soft":
                if tau is None:
                    tau = float(np.median(e_all)) + 1e-6
                r = np.exp(-(e_all - e_all.min()) / tau); r /= r.sum(); tau *= 0.999
                for k, p in enumerate(parts):
                    W1b, b1b, W2b, b2b = p.W1.copy(), p.b1.copy(), p.W2.copy(), p.b2.copy()
                    p.step(x, np.array([y]))
                    p.W1 = W1b + (p.W1 - W1b) * r[k]; p.b1 = b1b + (p.b1 - b1b) * r[k]; p.W2 = W2b + (p.W2 - W2b) * r[k]; p.b2 = b2b + (p.b2 - b2b) * r[k]
                target = r
            elif mode == "askdis":                                      # CTX4: ask when the router's choice and the best-predicting part DISAGREE, within a 10% question budget
                unsure = int(np.argmax(lg)) != int(np.argmin(e_all)) and asks < 0.10 * (t + 1)
                k = region(x) if unsure else int(np.argmin(e_all)); asks += int(unsure)
                parts[k].step(x, np.array([y])); target = np.zeros(P6.NR); target[k] = 1.0
            elif mode == "ask10":                                       # ask for the true context only when the router is unsure (bottom 10% of recent top-two margins)
                srt = np.sort(lg); mg = srt[-1] - srt[-2]; margins.append(mg); margins[:] = margins[-200:]
                unsure = len(margins) >= 20 and mg <= np.quantile(margins, 0.10)
                k = region(x) if unsure else int(np.argmin(e_all)); asks += int(unsure)
                parts[k].step(x, np.array([y])); target = np.zeros(P6.NR); target[k] = 1.0
            else:
                q = float(mode[3:])
                k = region(x) if crng.random() < q else int(np.argmin(e_all))
                parts[k].step(x, np.array([y])); target = np.zeros(P6.NR); target[k] = 1.0
            R += RLR * np.outer(target - pr_, ph)
        ks = [int(np.argmax(R @ phi(x))) for x in test]
        pt = [parts[k].predict(x)[0][0] for k, x in zip(ks, test)]
        out[mode] = float(np.mean((np.array(pt) - ty) ** 2))
        if mode in ("ask10", "askdis"):
            out["asks" if mode == "ask10" else "asks_dis"] = asks
        if mode == 'cue1.0':
            out['router_acc'] = float(np.mean([k == region(x) for k, x in zip(ks, test)]))
            Pm = []
            for x in test:
                lg = R @ phi(x); w_ = np.exp(lg - lg.max()); w_ /= w_.sum()
                Pm.append(float(sum(w_[k] * parts[k].predict(x)[0][0] for k in range(P6.NR))))
            out['cue1_mix'] = float(np.mean((np.array(Pm) - ty) ** 2))
            out['cue1_true_route'] = float(np.mean((np.array([parts[region(x)].predict(x)[0][0] for x in test]) - ty) ** 2))
    return out


if __name__ == "__main__":
    t0 = time.time()
    with Pool(20) as p:
        res = p.map(run, range(48))
    json.dump(res, open("../results/pctx_ask.json" if os.environ.get("ASK") else "../results/pctx_context.json", "w"))
    w = np.array([r["wide"] for r in res]); o = np.array([r["oracle"] for r in res])
    print("mode     | test error (median) | x wide (median) | x oracle (median)")
    for m in (["wide", "oracle", "cue0.0", "cue0.25", "ask10", "askdis"] if os.environ.get("ASK") else ["wide", "oracle"] + [f"cue{q}" for q in QS] + ["soft"]):
        v = np.array([r[m] for r in res])
        print(f"{m:8s} | {np.median(v):.4f} | {np.median(v / w):.2f} | {np.median(v / o):.1f}")
    if os.environ.get("ASK"):
        print("ask10: asked on %.1f%% of ticks; ask10 / cue0.25 median %.2f" % (100 * np.mean([r["asks"] for r in res]) / 6000, np.median([r["ask10"] / r["cue0.25"] for r in res])))
        print("askdis: asked on %.1f%% of ticks; askdis / ask10 median %.2f; askdis / cue0.25 median %.2f" % (100 * np.mean([r["asks_dis"] for r in res]) / 6000, np.median([r["askdis"] / r["ask10"] for r in res]), np.median([r["askdis"] / r["cue0.25"] for r in res])))
        print("done", round(time.time() - t0), "s"); raise SystemExit
    print("cue1.0 router accuracy (test points sent to their true region part): median %.3f; cue1.0 parts with TRUE routing at test: median x oracle %.1f" % (np.median([r["router_acc"] for r in res]), np.median([r["cue1_true_route"] / r["oracle"] for r in res])))
    mx = np.array([r["cue1_mix"] / r["cue1.0"] for r in res])
    print("cue1.0 hedged MIX prediction: median x oracle %.1f; mix / top-part error ratio median %.2f" % (np.median([r["cue1_mix"] / r["oracle"] for r in res]), np.median(mx)))
    print("done", round(time.time() - t0), "s")
