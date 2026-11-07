"""Example plugin: a brain that is more careful about growing, your principle 7 ("don't spawn if negative for the whole").

CautiousBrain only creates a neuron for an unrecognised input once `spawn_patience` similar unrecognised inputs have
been seen recently; until then the input is simply "not understood yet". The idea: one-off noise should not cost the
brain a new neuron. spawn_patience = 1 behaves exactly like the normal Brain2.

This is an EXAMPLE of the plugin format, not a claim that it works well. Test it with a Compare job.
"""
import numpy as np

from brain2 import Brain2


class CautiousBrain(Brain2):
    def __init__(self, *a, spawn_patience=1, **kw):
        super().__init__(*a, **kw)
        self.patience = max(1, int(spawn_patience))
        self._pending = []  # recent unrecognised inputs, not yet worth a neuron

    def step(self, x, y):
        if self.patience <= 1 or self.n == 0:
            return super().step(x, y)
        dist = np.linalg.norm(self.w[1:] - x, axis=1)
        if (dist < self.rho).any():
            return super().step(x, y)  # recognised: normal behaviour
        near = sum(1 for p in self._pending if np.linalg.norm(p - x) < self.rho)
        if near + 1 < self.patience:  # not enough evidence yet: remember it, do not grow
            self._pending.append(x)
            self._pending = self._pending[-200:]
            return self.d.max() + self.spawn_cost, 0, False
        self._pending = [p for p in self._pending if np.linalg.norm(p - x) >= self.rho]
        return super().step(x, y)


PLUGIN = dict(
    name='Cautious growth (example)',
    about='A brain that waits for repeated evidence before growing a neuron.',
    brains={'CautiousBrain': CautiousBrain},
    params=[dict(name='spawn_patience', kind='brain', type='int', default=1, min=1, max=50, step=1, owner='CautiousBrain',
                 group='Cautious growth',
                 about='How many similar unrecognised inputs must be seen before a new neuron is grown. 1 = grow immediately (normal). Only used by the CautiousBrain design.')],
)
