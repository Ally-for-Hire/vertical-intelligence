"""E1 + E2 (oracle direction): is an ORACLE-TAUGHT student's confidence honest, inside and outside what it was taught, and does it know when it is outside?
Student = relevance prior over the 8 pair descriptors, trained on 100 standard worlds (seeds 5000-5099) with ORACLE labels (ground-truth parent sets); comparison student trained on OWN labels (permutation tests).
Query: P(input i is a true parent of output j), scored against ground truth on NON-SELF pairs (an output's own variable is always a parent and trivially flagged).
Test sets: standard (48 worlds, 9000-9047; first 24 = conformal calibration, last 24 = in-distribution test), 7 shifted families (24 worlds each, seeds 9100-9123), 40 connectome worlds (q19 held-out subgraphs).
Metrics: ECE (10 equal-width bins), Brier, AUROC, mean predicted vs actual positive rate; split-conformal prediction sets (target coverage 0.90) calibrated on standard worlds; E2: world-level novelty scores
(5-seed ensemble disagreement; mean nearest-neighbour distance to training descriptors) and their AUROC for 'standard vs shifted'.
Predictions (QUEUE E1/E2, stated before running): logistic near-calibrated in-distribution (ECE < 0.05), 2x32 net overconfident (0.05-0.10); ECE rises under shift, worst on pure-interaction and connectome; conformal coverage
holds in-distribution and fails under shift; ensemble-disagreement AUROC 0.7-0.8 for pure-interaction/connectome, ~0.55 for mild shifts."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
import expM_curriculum_scaling as M

SHIFTS = ("kinds_tanh_product", "many_parents_4_6", "high_noise_0.4", "copy_distractors_only", "delayed_effects", "fly_like", "pure_interaction")
NB, ALPHA = 10, 0.10


def desc_job(args):
    kind, seed = args
    from selfnn import run_scenario
    if kind == "connectome":
        import q19_connectome_world as C
        X, Y, cp, truth, na = C.build(seed)
    elif kind == "standard":
        X, Y, cp, truth, nd, na = M.build(seed)
    else:
        from worlds import sample_world_family, rollout
        w = sample_world_family(seed, kind)
        X, Y = rollout(w, M.W + 1, np.random.default_rng(seed + 7))
        cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
        truth, na = w.truth()[cp], w.n_act
    r = run_scenario(X[:M.W + 1], Y[:M.W + 1], cp, na, M.W + 1, M.HID, M.LR, M.W, seed, "plain")
    d = r["desc"]
    nonself = np.ones(truth.shape, bool)
    for j, c in enumerate(cp):
        nonself[j, c] = False
    return dict(kind=kind, seed=seed, F=d[nonself], y=truth[nonself].astype(float))


def ece(p, y):
    b = np.minimum((p * NB).astype(int), NB - 1)
    return float(sum(abs(p[b == k].mean() - y[b == k].mean()) * (b == k).mean() for k in range(NB) if (b == k).any()))


def auroc(s, y):
    y = y.astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    r = s.argsort().argsort()
    return float((r[y].mean() - (y.sum() - 1) / 2) / (~y).sum())


def reliability(p, y):
    b = np.minimum((p * NB).astype(int), NB - 1)
    return [[float(p[b == k].mean()), float(y[b == k].mean()), int((b == k).sum())] for k in range(NB) if (b == k).any()]


if __name__ == "__main__":
    from selfnn import PriorModel, NNPrior
    from scipy.spatial import cKDTree
    t0 = time.time()
    jobs = [("standard", s) for s in range(9000, 9048)] + [(f, s) for f in SHIFTS for s in range(9100, 9124)] + [("connectome", g) for g in range(20, 60)]
    with Pool(20) as p:
        tr = p.map(M.train_job, range(5000, 5100))
        te = p.map(desc_job, jobs)
    Ftr, ytr_truth, ytr_own = [], [], []
    for t in tr:
        d, truth, lab = t["desc"], t["truth"], t["lab"]
        ns = d[..., 6] < 0.5                                   # descriptor 6 = 'this input is the output's own variable'
        Ftr.append(d[ns]); ytr_truth.append(truth[ns].astype(float)); ytr_own.append(lab[ns])
    Ftr, ytr_truth, ytr_own = np.concatenate(Ftr), np.concatenate(ytr_truth), np.concatenate(ytr_own)
    print("training pairs %d (non-self); oracle positive rate %.3f; own-label positive rate %.3f" % (len(ytr_truth), ytr_truth.mean(), ytr_own.mean()), flush=True)
    students = {}
    lr_o = PriorModel(M.RIDGE, "soft"); lr_o.fit(Ftr, ytr_truth); students["oracle_logistic"] = lr_o
    ens = []
    for s in range(5):
        m = NNPrior(seed=s); m.fit(Ftr, ytr_truth); ens.append(m)
    students["oracle_net"] = ens[0]
    lr_s = PriorModel(M.RIDGE, "soft"); lr_s.fit(Ftr, ytr_own); students["own_logistic"] = lr_s
    nn_s = NNPrior(seed=0); nn_s.fit(Ftr, ytr_own); students["own_net"] = nn_s
    sets = {"standard_cal": [t for t in te if t["kind"] == "standard" and t["seed"] < 9024], "standard_test": [t for t in te if t["kind"] == "standard" and t["seed"] >= 9024]}
    for f in SHIFTS + ("connectome",):
        sets[f] = [t for t in te if t["kind"] == f]
    out = dict(metrics={}, reliability={}, conformal={}, novelty={})
    for sn, m in students.items():
        Fc = np.concatenate([t["F"] for t in sets["standard_cal"]]); yc = np.concatenate([t["y"] for t in sets["standard_cal"]])
        pc = m.prob(Fc)
        sc = np.where(yc > 0.5, 1 - pc, pc)                    # nonconformity = 1 - p(true label)
        q = float(np.quantile(sc, np.ceil((len(sc) + 1) * (1 - ALPHA)) / len(sc)))
        for setn, S in sets.items():
            F = np.concatenate([t["F"] for t in S]); y = np.concatenate([t["y"] for t in S]); pr = m.prob(F)
            out["metrics"][f"{sn}|{setn}"] = dict(ece=ece(pr, y), brier=float(np.mean((pr - y) ** 2)), auroc=auroc(pr, y), mean_p=float(pr.mean()), pos_rate=float(y.mean()), n=int(len(y)))
            out["reliability"][f"{sn}|{setn}"] = reliability(pr, y)
            in1 = (1 - pr) <= q; in0 = pr <= q
            cov = np.where(y > 0.5, in1, in0)
            out["conformal"][f"{sn}|{setn}"] = dict(coverage=float(cov.mean()), mean_set_size=float((in1.astype(int) + in0).mean()), coverage_positives=float(cov[y > 0.5].mean()) if (y > 0.5).any() else float("nan"))
    tree = cKDTree(((Ftr - Ftr.mean(0)) / (Ftr.std(0) + 1e-9))[np.random.default_rng(0).choice(len(Ftr), min(30000, len(Ftr)), replace=False)])
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9

    def novelty(t):
        P = np.stack([e.prob(t["F"]) for e in ens])
        dist = tree.query((t["F"] - mu) / sd, k=1)[0]
        return float(P.std(0).mean()), float(np.median(dist)), float(np.mean(-(P.mean(0) * np.log(P.mean(0) + 1e-12) + (1 - P.mean(0)) * np.log(1 - P.mean(0) + 1e-12))))
    nov = {setn: [novelty(t) for t in S] for setn, S in sets.items()}
    base = nov["standard_test"]
    for setn in list(SHIFTS) + ["connectome"]:
        res = {}
        for k, nm in enumerate(("ensemble_disagreement", "nn_distance", "entropy")):
            s = np.array([v[k] for v in base] + [v[k] for v in nov[setn]]); lab = np.r_[np.zeros(len(base)), np.ones(len(nov[setn]))]
            res[nm] = auroc(s, lab)
        out["novelty"][setn] = res
    json.dump(out, open("../results/e1_calibration.json", "w"), indent=1)
    order = ["standard_test"] + list(SHIFTS) + ["connectome"]
    for sn in students:
        print(f"\n{sn}: set | ECE | Brier | AUROC | mean predicted p vs actual positive rate | conformal coverage (target 0.90) / mean set size / coverage of true parents")
        for setn in order:
            a = out["metrics"][f"{sn}|{setn}"]; c = out["conformal"][f"{sn}|{setn}"]
            print(f"  {setn:22s} {a['ece']:.3f}  {a['brier']:.3f}  {a['auroc']:.3f}   {a['mean_p']:.3f} vs {a['pos_rate']:.3f}   {c['coverage']:.3f} / {c['mean_set_size']:.2f} / {c['coverage_positives']:.3f}")
    print("\nE2 novelty AUROC (standard_test vs set; 0.5 = cannot tell):  ensemble disagreement | NN distance | entropy")
    for setn in list(SHIFTS) + ["connectome"]:
        r = out["novelty"][setn]
        print(f"  {setn:22s} {r['ensemble_disagreement']:.3f} | {r['nn_distance']:.3f} | {r['entropy']:.3f}")
    print("done", round(time.time() - t0), "s")
