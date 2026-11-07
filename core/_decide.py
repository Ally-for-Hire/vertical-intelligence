"""Decision check (author 2026-11-03: runs test theories; stop once they are proven or disproven). Reads finished units (X1.29 worlds, X1.34 cells), computes each registered
prediction's status with the units so far: DECIDED-UP / DECIDED-DOWN when the paired 99% bootstrap range (interim looks; stricter than the 95% of the full run, to pay for looking repeatedly) excludes 0 (needs >= MIN_UNITS = 6), UNDECIDED otherwise.
Usage: python _decide.py x1_29 | x1_34"""
import sys as _s, pathlib as _p; _s.path.insert(0, str(next(q for q in _p.Path(__file__).resolve().parents if (q / 'core' / 'vipaths.py').exists()) / 'core')); import vipaths  # vi-bootstrap
import sys, os, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x0_1 import boot_ci
R = os.path.join(str(vipaths.RESULTS)); MIN_UNITS, LOOK_PCT = 6, (0.5, 99.5)   # CONST: decide_min_units, decide_look_range (ARBITRARY): interim looks use a 99% range, only the full run uses 95%


def verdict(d):
    if len(d) < MIN_UNITS: return f"UNDECIDED (n={len(d)} < {MIN_UNITS})"
    d = np.array(d, float); meds = np.median(np.random.default_rng(1).choice(d, (2000, d.size)), axis=1); m = float(np.median(d)); lo, hi = (float(np.percentile(meds, p)) for p in LOOK_PCT)
    return ("DECIDED-UP" if lo > 0 else "DECIDED-DOWN" if hi < 0 else "UNDECIDED") + f" n={len(d)} mean {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def x129():
    old = json.load(open(os.path.join(R, "x1_28_run.json")))["result"]["per_world"]; W = {}
    for f in glob.glob(os.path.join(R, "x1_29_worlds", "*.json")): W[int(os.path.basename(f)[:-5])] = json.load(open(f))
    print(f"X1.29: {len(W)} of 12 worlds finished")
    for s, nm in (("MDL", "DIG_MDL"), ("GROW", "DIG_GROW"), ("BUDGET", "DIG_BUDGET")):
        for pk in ("rand", "mix", "auto"): print(f"  P499 {s}_{pk} minus default compute:", verdict([W[k][f"{s}_{pk}"]["last"] - old[str(k)][nm]["last"] for k in W]))
    print("  P502 FLAT_BIG minus FLAT:", verdict([W[k]["FLAT_BIG"]["last"] - old[str(k)]["FLAT"]["last"] for k in W]))
    for s in ("MDL", "GROW", "BUDGET"):
        print(f"  P500 {s}: auto minus rand:", verdict([W[k][f"{s}_auto"]["last"] - W[k][f"{s}_rand"]["last"] for k in W]), "| mix minus rand:", verdict([W[k][f"{s}_mix"]["last"] - W[k][f"{s}_rand"]["last"] for k in W]))
    deep = [(k, a, W[k][a]["rec_end_d1"], W[k][a]["rec_end_d2"]) for k in W for a in W[k] if isinstance(W[k][a], dict) and (W[k][a].get("rec_end_d1", 0) or 0) + (W[k][a].get("rec_end_d2", 0) or 0) > 0]
    print(f"  P501 any arm recovers a concept built on another: {'YES (decided: proven) ' + str(deep[:3]) if deep else 'none in ' + str(len(W)) + ' worlds x 10 arms (disproven if this holds at the end; stop rule: 6 worlds empty)'}")


def x134():
    C = [json.load(open(f)) for f in glob.glob(os.path.join(R, "x1_34_cells", "*.json"))]
    print(f"X1.34: {len(C)} of 96 cells finished")
    for g in (0.8, 1.2, 1.6, 2.0):
        by = {a: {c["seed"]: c for c in C if c["g"] == g and c["arm"] == a} for a in ("NONE", "CORR", "ALL", "CORR_POOL")}
        for a in ("CORR", "ALL", "CORR_POOL"):
            ks = sorted(set(by[a]) & set(by["NONE"])); print(f"  P521 g={g} {a} reach minus NONE:", verdict([by[a][k]["reach"][0] - by["NONE"][k]["reach"][0] for k in ks]))
        ks = sorted(by["ALL"]); print(f"  P522 g={g} ALL reach(T) minus reach(T/3):", verdict([by["ALL"][k]["reach"][0] - by["ALL"][k]["reach_third"][0] for k in ks]))


def x135():
    Cc = [json.load(open(f)) for f in glob.glob(os.path.join(R, "x1_35_cells", "*.json"))]
    print(f"X1.35: {len(Cc)} of 108 cells finished")
    for g in (0.8, 1.2, 2.0):
        by = {a: {c["seed"]: c for c in Cc if c["g"] == g and c["arm"] == a} for a in ("NONE", "CORR", "ALL", "CORR_POOL", "CORR_CLONE", "ALL_CLONE")}
        for a in ("CORR", "ALL", "CORR_POOL", "CORR_CLONE", "ALL_CLONE"):
            ks = sorted(set(by[a]) & set(by["NONE"]))
            for r in range(len(by["NONE"][ks[0]]["rooms"])) if ks else []:
                print(f"  P525 g={g} {a} room{r + 1} exam minus NONE:", verdict([by[a][k]["rooms"][r]["exam"] - by["NONE"][k]["rooms"][r]["exam"] for k in ks]), "| links:", verdict([by[a][k]["rooms"][r]["links"] - by["NONE"][k]["rooms"][r]["links"] for k in ks]))


def x136():
    Cc = [json.load(open(f)) for f in glob.glob(os.path.join(R, "x1_36_cells", "*.json"))]
    print(f"X1.36: {len(Cc)} of 72 cells finished")
    for g in (0.8, 1.2, 2.0):
        by = {r: {c["seed"]: c for c in Cc if c["g"] == g and c["refit"] == r} for r in (0, 30, 150, 600)}
        for r in (30, 150, 600):
            ks = sorted(set(by[r]) & set(by[0]))
            for k in range(4) if ks else []:
                print(f"  P529 g={g} refit {r} minus 0, room{k + 1} exam:", verdict([by[r][s]["rooms"][k]["exam"] - by[0][s]["rooms"][k]["exam"] for s in ks]), "| corr2_mean:", verdict([by[r][s]["rooms"][k]["corr2_mean"] - by[0][s]["rooms"][k]["corr2_mean"] for s in ks]))


if __name__ == "__main__":
    import io, contextlib, time
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): {"x1_29": x129, "x1_34": x134, "x1_35": x135, "x1_36": x136}[sys.argv[1]]()
    print(buf.getvalue(), end="")
    head = "=== look " + time.strftime("%Y-%m-%d %H:%M") + " " + sys.argv[1] + chr(10)
    open(os.path.join(R, "decide_log.txt"), "a").write(head + buf.getvalue())     # every look is on record
