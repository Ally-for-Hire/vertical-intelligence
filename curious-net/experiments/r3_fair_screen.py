"""R3 (review 2.2): a FAIR no-experience baseline for the learned relevance gate. Same worlds, protocol and test seeds as exp M (48 test worlds 9000-9047, MLP per-output framework, gate switched on at W = 100).
Arms: plain; oracle hard mask; hard Bonferroni screen (the old baseline); SOFT p-value screen (p = 1 - two-sided p-value of the input's lag-1 correlation with the output's change, z = |r| sqrt(W):
zero tuned constants); TUNED soft screen p = sigmoid(a (f1 - c)) on the same single descriptor f1 = log1p(z), with (a, c) picked on 24 VALIDATION worlds (seeds 1000-1023) by first-1000-tick error from a
5 x 6 grid (two constants = the minimum 'experience'); the learned priors from exp M's recipe at K = 100 (ridge logistic, 2x32 net, own labels). Gains over plain in ticks 0-1000 (exp M convention) and 100-1000.
Prediction (stated before running): unknown; if the tuned soft screen recovers most of the nn_soft gain (+0.128 in exp M), the headline becomes 'soft gating of a calibrated screen', not 'experience'."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
sys.path.insert(0, ".")
import numpy as np, common
from multiprocessing import Pool
from scipy.stats import norm
import expM_curriculum_scaling as M

SLOPES = (1.0, 2.0, 4.0, 8.0, 16.0)
CENTRES = (0.5, 0.8, 1.0, 1.3, 1.6, 2.0)
VAL = range(1000, 1024)
TEST = range(9000, 9048)


class Screen:
    """no-experience soft gate from the single descriptor f1 = log1p(|r| sqrt(W)); kind 'pvalue' (no constants) or 'sigmoid' (slope a, centre c)"""
    rule = "soft"

    def __init__(self, kind, a=None, c=None):
        self.kind, self.a, self.c = kind, a, c

    def prob(self, F):
        z = np.expm1(F[:, 0])
        if self.kind == "pvalue":
            return 1.0 - 2.0 * norm.sf(z)
        return 1.0 / (1.0 + np.exp(-self.a * (F[:, 0] - self.c)))


def run_arms(args):
    seed, specs = args
    from selfnn import run_scenario
    from worlds import sample_world, rollout
    X, Y, cp, truth, nd, na = M.build(seed)
    out = {}
    for name, mode, prior in specs:
        out[name] = run_scenario(X, Y, cp, na, M.T, M.HID, M.LR, M.W, seed, mode, prior=prior, truth_mask=truth)["err"]
    return out


if __name__ == "__main__":
    from selfnn import PriorModel, NNPrior
    t0 = time.time()
    grid = [(a, c) for a in SLOPES for c in CENTRES]
    with Pool(20) as p:
        val_specs = [("plain", "plain", None)] + [(f"s{a}_{c}", "prior", Screen("sigmoid", a, c)) for a, c in grid]
        val = p.map(run_arms, [(s, val_specs) for s in VAL])
        vplain = np.array([v["plain"][:1000].mean() for v in val])
        vg = {k: float(np.mean(vplain - np.array([v[k][:1000].mean() for v in val]))) for k, _, _ in val_specs[1:]}
        best = max(vg, key=vg.get)
        a_best, c_best = [float(x) for x in best[1:].split("_")]
        print("validation: best sigmoid screen a=%g c=%g, validation gain %+.3f; grid range %+.3f .. %+.3f" % (a_best, c_best, vg[best], min(vg.values()), max(vg.values())), flush=True)
        tr = p.map(M.train_job, range(5000, 5100))
        F = np.concatenate([t["desc"].reshape(-1, t["desc"].shape[-1]) for t in tr])
        y = np.concatenate([np.asarray(t["lab"], float).reshape(-1) for t in tr])
        lr = PriorModel(M.RIDGE, "soft"); lr.fit(F, y)
        nn = NNPrior(seed=100); nn.fit(F, y)
        specs = [("plain", "plain", None), ("oracle", "oracle", None), ("hard_screen", "screen", None), ("pvalue_screen", "prior", Screen("pvalue")),
                 ("tuned_screen", "prior", Screen("sigmoid", a_best, c_best)), ("lr_soft100", "prior", lr), ("nn_soft100", "prior", nn)]
        te = p.map(run_arms, [(s, specs) for s in TEST])
    json.dump(dict(best=[a_best, c_best], val_grid=vg, test=[{k: v.tolist() for k, v in t.items()} for t in te]), open("../results/r3_fair_screen.json", "w"))
    print("48 test worlds (exp M seeds); gain over plain, mean +- SE (paired):")
    for sl, lab in ((slice(0, 1000), "ticks 0-1000"), (slice(100, 1000), "ticks 100-1000")):
        pl = np.array([t["plain"][sl].mean() for t in te])
        print(f" {lab}: plain error {pl.mean():.3f}")
        for k, _, _ in specs[1:]:
            d = pl - np.array([t[k][sl].mean() for t in te])
            print(f"   {k:14s} {d.mean():+.3f} +- {d.std(ddof=1)/np.sqrt(len(d)):.3f}")
    pl = np.array([t["plain"][:1000].mean() for t in te])
    dn = pl - np.array([t["nn_soft100"][:1000].mean() for t in te]); ds = pl - np.array([t["tuned_screen"][:1000].mean() for t in te])
    print(" learned nn prior minus tuned screen (paired): %+.3f +- %.3f" % ((dn - ds).mean(), (dn - ds).std(ddof=1) / np.sqrt(len(dn))))
    print("done", round(time.time() - t0), "s")
