"""X1.39: is the cell FAMILY the limit? (X1.37: more labels did not help; X1.38: more hidden units helped only to 24 at g=1.2, no exact recovery.) The world's rule is a SINE of a sum; the cell's inner activation was tanh.
Arms (every one an arm, same learner and world as X1.38, room 1, N=600, refit 150, 3 worlds): ACT tanh | ACT sin, each at hid 6 and 24, g = 0.8 / 1.2 / 2.0.
Honest framing: sin hands the learner the FAMILY of the answer, so a win is 'the family was the limit', not 'discovered from nothing'. Usage: python x1_39.py run | python x1_39.py run mix [gains] [hids] / run mixl0.03,mixl0.1 1.2 24 (X1.41: L1 on output weights; X1.40: the family-free arm: units mix tanh/sin/relu/identity; same cells cache)"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constants import C
import x1_28, x1_35

OUT = x1_28.OUT; N = int(sys.argv[5]) if len(sys.argv) > 5 else 600; CACHE = os.path.join(OUT, "x1_39_cells" if (N == 600 and "X139_LR" not in os.environ) else f"x1_39_cells_n{N}" + os.environ.get("X139_LR", "")); HIDS = tuple(int(a) for a in sys.argv[4].split(",")) if len(sys.argv) > 4 else (6, 24); ACTS = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("tanh", "sin"); GS = tuple(float(a) for a in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0.8, 1.2, 2.0); SEEDS = 3; REFIT = 150   # CONST: x139_cfg


def cell_run(task):
    g, hid, act, seed = task[:4]; init = task[4] if len(task) > 4 else 0; f = os.path.join(CACHE, f"g{g}_h{hid}_{act}_{seed}" + (f"_i{init}" if init else "") + ".json")
    if os.path.exists(f): return json.load(open(f))
    import cell, wstream, wrec
    wrec.GAIN = g; t0 = time.time(); NE = C("x132_exam"); k = 1; Lv = 5; base, _, l1 = act.partition("l"); L = cell.Learner(seed, dict(refit=int(os.environ.get('X139_REFIT', REFIT)), n_lab=N, hid=hid, act=base, l1=float(l1) if l1 else 0.0, init=init, lr=float(os.environ.get('X139_LR', 0.03)), patience=int(os.environ.get('X139_PAT', 10))))      # act 'mixl0.03' = mix cell with L1 0.03 on the output weights
    W = wstream.Stream(seed, dict(noise=0.0, n_tr=N, n_va=N // 2, lib_size=Lv, lib="rec")); data = [wstream.build(W.lib, sp, seed, W.cfg) for sp in x1_35.room_specs(k, k)]; r2 = []
    for i, d in enumerate(data): r2.append(L.step(d)["r2"]); x1_28.hb(seed, f"g{g}_h{hid}_{act}", i + 1, 50, r2)
    rng = np.random.default_rng([seed, 7, k]); lvls = rng.integers(0, Lv, NE)
    exam = [wstream.build(W.lib, dict(t=90000 + j, S=[int(lv)], a=[1.0], stage=0, has_new=False), seed, W.cfg) for j, lv in enumerate(lvls)]
    e = np.array([x[0] for x in L.exam(exam)]); xp = rng.standard_normal((C("x126_probe"), W.cfg["dim"])); V = W.lib.values(xp); rec = L.recovered(xp, V, list(range(Lv))); c2 = np.array([rec[j] ** 2 for j in range(Lv)])
    out = dict(g=g, hid=hid, act=act, seed=seed, init=init, exam=float(e.mean()), c2=[float(x) for x in c2], deep99=int((c2[1:] >= 0.99).sum()), n99=int((c2 >= 0.99).sum()), n90=int((c2 >= 0.9).sum()), weights=L.size(), seconds=round(time.time() - t0, 1))
    os.makedirs(CACHE, exist_ok=True); json.dump(out, open(f, "w")); return out


def run():
    import multiprocessing as mp, budget
    for kk in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(kk, "1")
    seed0 = C("x128_seed0"); t0 = time.time(); seed0 += int(os.environ.get('X139_SEED_OFFSET', 0)); nseeds = int(os.environ.get('X139_SEEDS', SEEDS)); tasks = [(g, h, a, s) for g in GS for h in HIDS for a in ACTS for s in range(seed0, seed0 + nseeds)]
    if os.environ.get('X139_INITS'): tasks = [(g, h, a, int(s), int(i)) for g in GS for h in HIDS for a in ACTS for s in os.environ['X139_ONLY'].split(',') for i in os.environ['X139_INITS'].split(',')]
    with mp.get_context("spawn").Pool(max(1, min(budget.workers(per_worker_gb=3.0), 3))) as pool: res = pool.map(cell_run, tasks, chunksize=1)
    for g in GS:
        for h in HIDS:
            for a in ACTS:
                x = [q for q in res if q["g"] == g and q["hid"] == h and q["act"] == a]
                print(f"g {g} hid {h} act {a}: exam {np.median([q['exam'] for q in x]):.3f} n90 {np.median([q['n90'] for q in x]):.0f} deep99 worlds {sum(q['deep99'] > 0 for q in x)}/{len(x)} median max corr2 {np.median([max(q['c2'][1:]) for q in x]):.3f} {np.median([q['seconds'] for q in x]):.0f}s", flush=True)
    # result-file name: the settings override (X139_LR) is checked FIRST (X1.44 once overwrote x1_40_run.json)
    json.dump(dict(per_cell=res, seconds=round(time.time() - t0, 1)), open(os.path.join(OUT, (("x1_45_run.json" if "X139_INITS" in os.environ else "x1_44b_run.json" if "X139_SEED_OFFSET" in os.environ else "x1_44_run.json") if "X139_LR" in os.environ else "x1_39_run.json" if len(sys.argv) <= 2 else f"x1_43_n{N}_run.json" if N != 600 else "x1_40_run.json" if "l" not in sys.argv[2] else "x1_41_run.json")), "w"), indent=1, default=float)
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
