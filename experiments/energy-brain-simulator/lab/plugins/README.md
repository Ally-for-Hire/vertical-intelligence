# Lab plugins: add your own brain designs and tasks

Drop a `.py` file in this folder. The Lab finds it automatically (reload the Lab page; no restart needed). Files starting
with `_` are ignored. A plugin that fails to load is reported in the Lab and never breaks anything else.

Plugins are code on your disk, written by you or by Claude. The Lab web page can never send code; it only sends numbers
and names, so adding a plugin is the only way new behaviour enters the system.

## What a plugin can add

```python
from brain2 import Brain2            # the current brain; subclass it to change one behaviour
import numpy as np

class MyBrain(Brain2):
    def __init__(self, *a, my_knob=1.0, **kw):      # your new setting must be a keyword argument with a default
        super().__init__(*a, **kw)
        self.my_knob = my_knob
    # override any method: step(x, y) handles one experience, query(X, Y) scores a test set,
    # condense() is the end-of-window cleanup, _move/_merge/_prune are its parts.

class MyWorld:                                       # a new task. Same constructor arguments as the default one
    def __init__(self, rng, K, D, sigma, **_):
        ...
    def sample(self, n):                             # return (X, Y): n inputs (n x D) and their category numbers (0..K-1)
        ...
    def drift(self, mode):                           # called once at the halfway point; mode is 'shift' or 'replace'
        ...

PLUGIN = dict(
    name='My ideas',
    about='One line on what this adds.',
    brains={'MyBrain': MyBrain},                     # appears in "brain design"
    envs={'myworld': MyWorld},                       # appears in "task type"
    params=[dict(name='my_knob', kind='brain', type='float', default=1.0, min=0, max=10, step=0.1,
                 owner='MyBrain', group='My ideas',
                 about='Plain-English explanation shown when you hover the setting.')],
)
```

* `kind` is `'brain'` (passed to the brain's constructor) or `'task'` (passed to the world's constructor).
* `owner` limits a setting to one brain or task type; leave it out to pass it to every brain/task.
* `type` is `float`, `int`, `bool` or `choice` (add `choices=[...]`). `min`/`max` are enforced, so bad values are rejected
  with a readable message instead of crashing a run.
* Choose which brain or task a job uses with `brain_class` / `env` in the job spec (the designer will expose this next).

## Rules of thumb

* Keep per-step work light: every experiment runs thousands of lifetimes of thousands of steps each.
* Use `self.rng` for randomness, never the global `random`/`np.random`, so runs stay repeatable by seed.
* After writing a plugin run `python lab_selftest.py` and compare `MyBrain` against `Brain2` with a **Compare** job.

`example_cautious_spawn.py` is a working example.
