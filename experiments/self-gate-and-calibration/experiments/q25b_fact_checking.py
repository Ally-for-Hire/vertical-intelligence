"""Q25b: fact-checking a systematically wrong teacher in causal worlds (the testable form of the user's skeptical-and-curious student, e.g. the ball-drop claim).
For each world the teacher asserts, for every (action a, core output j) pair, 'a affects j' or 'a does not affect j'. Teacher types: honest; random (each claim wrong with prob 0.1);
SYSTEMATIC (one output j* is claimed to be affected by every action: over-connection; one action a* is claimed to affect nothing: under-connection).
Student evidence: (i) its learned relevance prior p(a->j) from the first 100 random ticks (NNPrior trained on K=100 standard scenarios, own labels); (ii) own intervention experiments:
pulse action a (+1 for 5 ticks, 0 for 5 ticks) for 40 ticks and test whether output j's next-step change differs between phases (two-sample t test, Bonferroni level 1/(number of claims)); cost = 40 ticks.
Students: credulous (believes the teacher), prior-only (p >= 0.5), verify-all (experiment on every claim), targeted (experiment only where the teacher and the prior disagree, otherwise accept the consensus).
Metric: accuracy of the final beliefs over all (a, j) pairs vs ground truth; number of experiments."""
import os, sys, json, time
os.environ["CN_FULL"] = "1"
sys.path.insert(0, "../src")
import numpy as np, common
from multiprocessing import Pool
from scipy.stats import norm

T, W, HID, LR = 1500, 100, 32, 3e-3
EXP_TICKS = 40


def build(seed):
    from worlds import sample_world, rollout
    w = sample_world(seed)
    X, Y = rollout(w, T, np.random.default_rng(seed + 7))
    cp = [int(np.nonzero(w.perm == i)[0][0]) for i in range(w.n_core)]
    return w, X, Y, cp, w.truth()[cp], w.n_act


def train_job(seed):
    from selfnn import run_scenario, perm_labels
    w, X, Y, cp, truth, na = build(seed)
    r = run_scenario(X, Y, cp, na, T, HID, LR, W, seed, "plain")
    lab = perm_labels(r["nets"], r["Xs"], r["Ds"], seed=seed)
    return r["desc"].reshape(-1, r["desc"].shape[-1]), lab.reshape(-1)


def run_experiment(seed, a, j_pos, rng):
    """intervene on action a in a fresh copy of the world (own experiment), return t statistic of the effect on output (observed column j_pos)"""
    from worlds import sample_world
    w = sample_world(seed)
    ns, na = w.n_state, w.n_act
    for _ in range(50):                                     # let the copy settle (random actions), then run the pulse protocol
        w.step(w.act_to_world(rng.integers(-1, 2, size=na).astype(float)))
    prev = w.observe()
    on, off = [], []
    for t in range(EXP_TICKS):
        phase_on = (t // 5) % 2 == 0
        act = np.zeros(na)
        if phase_on:
            act[a] = 1.0
        w.step(w.act_to_world(act))
        cur = w.observe()
        (on if phase_on else off).append(cur[j_pos] - prev[j_pos])
        prev = cur
    on, off = np.array(on), np.array(off)
    se = np.sqrt(on.var(ddof=1) / len(on) + off.var(ddof=1) / len(off)) + 1e-12
    return abs(on.mean() - off.mean()) / se


def world_job(args):
    seed, prior = args
    w, X, Y, cp, truth, na = build(seed)
    n_core = len(cp)
    n_in = X.shape[1]
    act_cols = list(range(n_in - na, n_in))
    T_true = truth[:, act_cols]                                          # (n_core, n_act) bool: does action a affect output j
    from selfnn import run_scenario
    r = run_scenario(X, Y, cp, na, 300, HID, LR, W, seed, "plain")
    desc = r["desc"]
    p = prior.prob(desc.reshape(-1, desc.shape[-1])).reshape(n_core, n_in)[:, act_cols]   # prior belief a -> j
    rng = np.random.default_rng(seed)
    out = {}
    for teacher in ("honest", "random", "systematic"):
        claim = T_true.copy()
        if teacher == "random":
            flip = rng.random(claim.shape) < 0.10
            claim = np.where(flip, ~claim, claim)
        elif teacher == "systematic":
            jstar = int(rng.integers(n_core)); astar = int(rng.integers(na))
            claim[jstar, :] = True                                          # over-connection: everything affects j*
            claim[:, astar] = False                                         # under-connection: a* affects nothing
        n_claims = claim.size
        zc = norm.isf(1.0 / (2 * n_claims))                                 # Bonferroni two-sided critical value for the experiments
        res = {}
        res["credulous"] = (float((claim == T_true).mean()), 0)
        res["prior_only"] = (float(((p >= 0.5) == T_true).mean()), 0)
        belief = np.zeros_like(claim); nexp = 0
        for j in range(n_core):
            for a in range(na):
                z = run_experiment(seed, a, cp[j], rng)
                belief[j, a] = z > zc; nexp += 1
        res["verify_all"] = (float((belief == T_true).mean()), nexp)
        belief = claim.copy(); nexp = 0
        for j in range(n_core):
            for a in range(na):
                if claim[j, a] != (p[j, a] >= 0.5):                         # teacher and prior disagree -> curiosity: run the experiment
                    z = run_experiment(seed, a, cp[j], rng)
                    belief[j, a] = z > zc; nexp += 1
        res["targeted"] = (float((belief == T_true).mean()), nexp)
        out[teacher] = dict(res=res, n_claims=n_claims, n_wrong_claims=int((claim != T_true).sum()))
    return out


if __name__ == "__main__":
    from selfnn import NNPrior
    t0 = time.time()
    with Pool(20) as pool:
        trn = pool.map(train_job, range(5000, 5100))
        prior = NNPrior(seed=100)
        prior.fit(np.concatenate([t[0] for t in trn]), np.concatenate([t[1] for t in trn]))
        print("prior fitted", round(time.time() - t0), "s", flush=True)
        res = pool.map(world_job, [(s, prior) for s in range(9000, 9048)])
    json.dump(res, open("../results/q25b.json", "w"), default=float)
    print("teacher | student | accuracy (mean +- se over 48 worlds) | experiments per world")
    for teacher in ("honest", "random", "systematic"):
        wr = np.mean([r[teacher]["n_wrong_claims"] / r[teacher]["n_claims"] for r in res])
        print(f"-- teacher {teacher} (fraction of its claims that are wrong: {wr:.3f})")
        for st in ("credulous", "prior_only", "verify_all", "targeted"):
            acc = np.array([r[teacher]["res"][st][0] for r in res]); ne = np.array([r[teacher]["res"][st][1] for r in res])
            print(f"   {st:11s} acc {acc.mean():.3f} +- {acc.std(ddof=1)/np.sqrt(len(acc)):.3f}   experiments {ne.mean():5.1f}   (claims per world {np.mean([r[teacher]['n_claims'] for r in res]):.1f})")
