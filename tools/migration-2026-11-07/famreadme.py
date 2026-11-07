"""Generate experiments/<family>/README.md for the five code families (scripts, result files, what to read)."""
import os, re, glob, sys
NEW = "~/vertical-intelligence"
INFO = {
 "value-and-self": ("Value and the self; drives", "Does value defined through the agent's own needs give need-dependent valence, safety, attention to every need and a hidden need only depth can satisfy? Includes the first curiosity drives (learning progress, optimism, noisy-TV and dark-room controls).", "X0.1 - X0.10, X1.1, X1.2", "findings-map section 0"),
 "boredom-and-prediction": ("Prediction-only life and boredom", "An entity whose whole existence is prediction and internal models; how exact must its boredom be.", "X1.3, X1.4", "findings-map section 0 / drives"),
 "self-and-doubt": ("The Self, sensors and doubt", "A Self that audits networks from the inside, learns when not to answer, creates and reuses routines, plus closed-form checks of the combined model.", "X7.1 - X7.7b, X8.0 - X8.1e", "findings-map Self and doubt sections"),
 "learning-to-learn": ("Learning to learn: connection learners and the 'vertical' test", "Meta-learned climb, connecting nets, trees, the flat connection learner, digger and refiner: does a learner get better at learning as it learns more? The first clean positive is X1.18 (finite world with a no-memory control).", "X1.5 - X1.28b", "feasibility audit block C (the 'vertical' quantity was confounded in X1.10-X1.17)"),
 "deep-discovery": ("Discovery from nothing: the deep-composition world", "The ultimate goal (author, 2026-11-03): infinitely deepening discoveries from nothing via one simple rule. World `wrec.py`, cell `cell.py`; X1.47 is the linear-base rung/gate.", "X1.29 - X1.47", "feasibility audit block D (no valid test of the goal yet; learner was handed the rule's family and structure)"),
}
for fam, (title, blurb, exps, note) in INFO.items():
    d = f"{NEW}/experiments/{fam}"
    scripts = sorted(f for f in os.listdir(d) if f.endswith(".py"))
    ids = sorted({re.match(r"(x\d+_\d+[a-z]?)", f).group(1) for f in scripts if re.match(r"x\d+_\d+", f)}, key=lambda k: [int(t) if t.isdigit() else t for t in re.findall(r"\d+|[a-z]", k)])
    rows = []
    for i in ids:
        res = sorted(os.path.basename(p) for p in glob.glob(f"{NEW}/results/{i}_*") if os.path.isfile(p))
        rows.append(f"| {i.upper().replace('_', '.')} | `{i}.py` | " + (", ".join(f"`{r}`" for r in res[:4]) + (f" (+{len(res)-4} more)" if len(res) > 4 else "") if res else "none yet") + " |")
    other = [s for s in scripts if not re.match(r"x\d+_\d+", s)]
    txt = f"""# {title}

{blurb}

- Experiments: {exps}. Registered predictions: `paper/PREDICTIONS.md` (search the X-id). Narrative: `LOG.md`.
- Concept and verdicts: wiki `experiment-registry.md` (section for this family); {note}.
- Run a script from this folder, e.g. `python {ids[0] if ids else 'x0_1'}.py` (the docstring lists its modes). Shared code is in `../../core/` and is on the import path automatically (`core/vipaths.py`). Results go to the shared `../../results/`.

| Experiment | Script | Result files in `results/` |
|---|---|---|
""" + "\n".join(rows) + ("\n\nOther scripts here: " + ", ".join(f"`{o}`" for o in other) + "\n" if other else "\n")
    open(f"{d}/README.md", "w", encoding="utf-8", newline="\n").write(txt)
    print(fam, len(ids), "experiments")
