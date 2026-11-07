import json, numpy as np
te = json.load(open("../results/expQ.json"))["test"]
fams = ["standard", "kinds_tanh_product", "many_parents_4_6", "high_noise_0.4", "copy_distractors_only", "delayed_effects"]
print("gain over plain per-output MLP, first-1000-tick error (24 worlds per family); share of oracle gain in brackets")
print(f"{'family':24s} {'plain err':>9s} {'oracle':>16s} {'screen':>16s} {'nn_soft':>20s} {'seq_long':>20s}")
rows = {}
for f in fams:
    r = [x for x in te if x["fam"] == f]
    plain = np.array([np.mean(x["arms"]["plain"][:1000]) for x in r])
    cells = []
    og = (plain - np.array([np.mean(x["arms"]["oracle"][:1000]) for x in r])).mean()
    for a in ("oracle", "screen", "nn_soft", "seq_long"):
        d = plain - np.array([np.mean(x["arms"][a][:1000]) for x in r])
        rows[(f, a)] = (d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
        share = "" if a == "oracle" else f" ({100*d.mean()/og:4.0f}%)" if og > 0.01 else " (n/a)"
        cells.append(f"{d.mean():+.3f}±{d.std(ddof=1)/np.sqrt(len(d)):.3f}{share}")
    print(f"{f:24s} {plain.mean():9.3f} " + " ".join(f"{c:>20s}" for c in cells))
print("\nnegative transfer check (prior gain < 0 by more than 2 se):")
for f in fams:
    for a in ("nn_soft", "seq_long"):
        m, s = rows[(f, a)]
        if m < -2 * s:
            print(f"  {f} / {a}: {m:+.3f} +- {s:.3f}  <- prior HURTS")
print("done")
