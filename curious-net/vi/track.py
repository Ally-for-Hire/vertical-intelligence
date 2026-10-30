"""Weights & Biases view of an experiment, written for researchers who are NOT specialists (author's standing rule, 2026-10-28).

One W&B run per experiment (never one per simulation: the coordinating process logs, the workers do not). The raw JSON in results/ stays the
record of truth; this is the readable view of it. Everything a reader sees has a plain-English name:
  * run name   : "X0.10 | Hidden need that only depth can satisfy | scored run"
  * description: the question, the world in plain words, what each arm is, how to read the numbers (shown on the run's Overview tab)
  * tables     : "Arms explained", "Results per world", "Prediction verdicts", "Summary by arm"
  * charts     : metric names carry the unit and the direction, e.g. "Deaths per 10,000 steps (lower is better)"
The project is created PRIVATE by wandb's default; making it public is a publishing decision for the author. If wandb cannot start (no network,
no login) the run is written offline into results/wandb_local and can be synced later with `wandb sync`; the experiment itself never fails
because of logging."""
import os
import json

PROJECT = "vertical-intelligence"
LOCAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "wandb_local")


class Tracker:
    def __init__(self, exp_id, title, question, world, arms, how_to_read, config=None, kind="scored run", tags=()):
        """arms: {code: (plain name, one-sentence explanation)}"""
        self.ok = False; self.exp_id = exp_id; self.arms = arms
        notes = (f"## Question\n{question}\n\n## The world, in plain words\n{world}\n\n## What each arm is\n" +
                 "\n".join(f"- **{v[0]}** (`{k}`): {v[1]}" for k, v in arms.items()) + f"\n\n## How to read the numbers\n{how_to_read}\n\n"
                 "Every number comes from simulated worlds that start from different random seeds; each world is one independent trial. "
                 "Medians are used because a few unlucky worlds would otherwise distort an average. Intervals are 95% bootstrap ranges "
                 "for the median difference between two arms on the SAME worlds (paired), so they show whether a gap is bigger than luck.")
        try:
            import wandb
            os.makedirs(LOCAL, exist_ok=True)
            mode = os.environ.get("WANDB_MODE") or "online"
            try:
                self.run = wandb.init(project=PROJECT, name=f"{exp_id} | {title} | {kind}", group=exp_id, job_type=kind, notes=notes,
                                      config=config or {}, tags=list(tags) + [exp_id], dir=LOCAL, mode=mode, reinit=True,
                                      settings=wandb.Settings(silent=True))
            except Exception:
                self.run = wandb.init(project=PROJECT, name=f"{exp_id} | {title} | {kind}", group=exp_id, job_type=kind, notes=notes,
                                      config=config or {}, tags=list(tags) + [exp_id], dir=LOCAL, mode="offline", reinit=True,
                                      settings=wandb.Settings(silent=True))
            self.wandb = wandb; self.ok = True
            self.table("Arms explained", ["Arm", "Code name in the data files", "What it is"], [[v[0], k, v[1]] for k, v in arms.items()])
        except Exception as e:
            print("W&B logging unavailable:", repr(e)[:120])

    def name(self, code):
        return self.arms.get(code, (code,))[0]

    def table(self, title, columns, rows):
        if self.ok: self.run.log({title: self.wandb.Table(columns=columns, data=rows)})

    def log(self, metrics, step=None):
        if self.ok: self.run.log(metrics, step=step) if step is not None else self.run.log(metrics)

    def summary(self, **kv):
        if self.ok:
            for k, v in kv.items(): self.run.summary[k] = v

    def verdicts(self, items):
        """items: list of (id, plain statement, 'PASSED'|'FAILED'|'VOID', evidence text)"""
        self.table("Prediction verdicts (written down BEFORE the scored run)", ["Id", "What we predicted", "Verdict", "Evidence"], [list(i) for i in items])
        self.summary(**{f"Verdict {i[0]}": i[2] for i in items})

    def finish(self):
        if self.ok:
            self.run.finish()
