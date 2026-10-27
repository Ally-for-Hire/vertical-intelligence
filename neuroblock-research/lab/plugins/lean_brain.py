"""LeanBrain: the brain with only the settings that mattered, and two of them adjusted by the brain itself.

What the 2026-10-18 sweep (CORE_SETTINGS.md) says to keep:
  * built in, always on, not a knob: moving neurons, merging, pruning (they only ever helped); step size 0.5, learning rate 0.1.
  * fixed physical constants: neuron spacing 0.3 (the speed limit), 40 us per unit of wire.
  * dropped entirely: hardening, the expand drive, newborn wire length, the not-recognised penalty.
  * adjusted by the brain itself, each with a price:
      1. the matching radius ("rho")  -> self-tuning, paid for with a small extra delay on every thought
      2. which neuron answers          -> "think twice" only when candidates disagree, paid for per use

SELF-TUNING THE RADIUS (grading itself on predictions made BEFORE learning).
  Whenever a new experience arrives, before learning from it, the brain checks what each of a range of candidate radii (about half to
  two and a half times the current one) WOULD have answered, and whether that answer would have been right (it learns the true label
  right after). This running tally is its own prediction error, the way predictive processing describes it. At each cleanup round it
  moves to a clearly better radius if there is one, preferring the one nearest where it already is. If it stays put for a couple of
  rounds it looks less and less often (every 1, 2, 4, 8 rounds), and goes back to every round as soon as it moves. So it works hard only
  while the uncertainty is dropping, and goes quiet when settled.
  Prices: tune_cost microseconds on every thought (the self-monitor circuit), plus a delay that grows with the brain's size on rounds when
  it actually looks. Without a price the brain would get the benefit for free.

THINKING TWICE. Normally the first matching signal to arrive wins (fast). If the matching neurons disagree about the answer, the brain
  compares them and picks the best-fitting one, paying deliberate_cost microseconds, but only for those conflicts.

LeanBrain(tune=False, deliberate=False) behaves exactly like the original Brain2 with default settings (checked in lab_selftest.py).
"""
import numpy as np

from brain2 import Brain2


class LeanBrain(Brain2):
    STEP = 1.25          # ratio between neighbouring candidate radii
    SPAN = range(-3, 5)  # candidates are the current radius x STEP^k for these k (about 0.5x to 2.4x)
    MARGIN = 0.01        # a change must beat staying put by this much accuracy
    MIN_TALLY = 60       # do not move on fewer than this many graded predictions
    SEARCH_US = 0.02     # delay per neuron per candidate radius, on rounds when the brain looks

    def __init__(self, D, K, rng, rho=0.5, tune=True, deliberate=True, tune_cost=2.0, deliberate_cost=10.0, **_ignored):
        super().__init__(D, K, rho=rho, rng=rng, lr=0.1, move=True, merge=True, prune=True, min_sep=0.3, select='fast',
                         spawn_len=1.0, eta=0.5, spawn_cost=50.0, rigid=False, expand=False)
        self.tune, self.deliberate = bool(tune), bool(deliberate)
        self.tune_cost, self.deliberate_cost = float(tune_cost), float(deliberate_cost)
        self.rho_start = float(rho)
        self.tune_every, self.still, self.round = 1, 0, 0
        self.watching = True                       # is the brain grading candidate radii this round?
        self._hits, self._cnt = np.zeros(len(self.SPAN)), 0.0
        self._set_candidates()
        self.search_delay = 0.0     # extra microseconds on this round's thoughts because the brain just looked
        self.last_conflict = 0.0    # share of the last test inputs that needed thinking twice

    # hooks for subclasses that keep an energy ledger (no-ops here)
    def _on_conflict(self):
        pass

    def _on_search(self, work):
        pass

    def _allow_search(self, work):
        return True

    # ---- answering: first signal wins unless candidates disagree ----
    def _pick(self, m, dist, d):
        """m: bool matches (rows x neurons). Returns (winner column per row, conflict flag per row)."""
        fast = np.where(m, d[None, :], np.inf).argmin(1)
        if not self.deliberate:
            return fast, np.zeros(len(fast), bool)
        lab = self.lab[1:].argmax(1)
        conflict = (m & (lab[None, :] != lab[fast][:, None])).any(1) & m.any(1)
        best = np.where(m, dist, np.inf).argmin(1)
        return np.where(conflict, best, fast), conflict

    def step(self, x, y):
        if self.tune and self.watching and self.n > 0:
            self._grade_candidates(x, y)  # predict with every candidate radius BEFORE learning from this example
        if not self.deliberate or self.n == 0:
            return super().step(x, y)
        dist = np.linalg.norm(self.w[1:] - x, axis=1)
        m = dist < self.rho
        if not m.any():
            return super().step(x, y)  # nothing recognises it: grow, as before
        idx = np.where(m)[0] + 1
        labs = self.lab[idx].argmax(1)
        if (labs == labs[np.argmin(self.d[idx])]).all():
            return super().step(x, y)  # everyone agrees: first signal wins, no extra thinking
        win = idx[np.argmin(dist[idx - 1])]  # conflict: think twice, take the best fit
        self._on_conflict()
        self.w[win] += self.lr_of(win) * (x - self.w[win])
        self.cnt[win] += 1
        self.win_use[win] += 1
        self.lab[win, y] += 1
        return self.d[win] + self.deliberate_cost, win, False

    def query(self, X, Y):
        if self.n == 0:
            return self.spawn_cost, 0.0, 0.0
        dist = np.linalg.norm(X[:, None, :] - self.w[None, 1:, :], axis=2)
        m = dist < self.rho
        covered = m.any(1)
        col, conflict = self._pick(m, dist, self.d[1:])
        win = col + 1
        t = np.where(covered, self.d[win] + np.where(conflict, self.deliberate_cost, 0.0), self.d.max() + self.spawn_cost)
        pred = self.lab[win].argmax(1)
        self.last_conflict = float((conflict & covered).sum() / max(1, covered.sum()))
        overhead = (self.tune_cost + self.search_delay) if self.tune else 0.0
        return float(t.mean()) + overhead, float(np.mean(covered & (pred == Y))), float(covered.mean())

    # ---- self-tuning the radius ----
    def _set_candidates(self):
        self._radii = self.rho * self.STEP ** np.array(list(self.SPAN), float)

    def _grade_candidates(self, x, y):
        """what would each candidate radius have answered for this new experience, and was it right?"""
        dist = np.linalg.norm(self.w[1:] - x, axis=1)
        M = dist[None, :] < self._radii[:, None]              # candidates x neurons
        covered = M.any(1)
        lab = self.lab[1:].argmax(1)
        fast = np.where(M, self.d[1:][None, :], np.inf).argmin(1)
        pred = lab[fast]
        if self.deliberate:
            conflict = (M & (lab[None, :] != pred[:, None])).any(1) & covered
            if conflict.any():
                best = np.where(M, dist[None, :], np.inf).argmin(1)
                pred = np.where(conflict, lab[best], pred)
        self._hits += covered & (pred == y)
        self._cnt += 1.0

    def _tune_radius(self):
        self.search_delay = 0.0
        moved = False
        if self.watching and self._cnt >= self.MIN_TALLY and self.n > 0:
            work = self.n * len(self.SPAN)
            if self._allow_search(work):
                scores = self._hits / self._cnt
                here = list(self.SPAN).index(0)
                near = [i for i, sc in enumerate(scores) if sc >= scores.max() - 0.005]  # equally good radii: keep the one closest to where it is
                i = min(near, key=lambda j: (abs(j - here), j))
                self.search_delay = self.SEARCH_US * work   # looking takes time, more for a bigger brain
                self._on_search(work)
                if i != here and scores[i] - scores[here] > self.MARGIN:
                    self.rho = float(self._radii[i])
                    self.tune_every, self.still, moved = 1, 0, True    # still improving: keep looking every round
                else:
                    self.still += 1
                    if self.still >= 2:
                        self.tune_every = min(8, self.tune_every * 2)  # settled: look less and less often
            self._hits *= 0.5
            self._cnt *= 0.5
        self.round += 1
        self.watching = (self.round % self.tune_every == 0)
        if moved:
            self._set_candidates()
            self._hits[:], self._cnt = 0.0, 0.0   # the old tallies were for different radii

    def condense(self):
        if self.tune:
            self._tune_radius()
        super().condense()

    def diag(self):
        """the brain's own dials, recorded each round for the Lab"""
        return dict(rho=self.rho, think_twice=self.last_conflict, looking_every=float(self.tune_every))


PLUGIN = dict(
    name='Lean brain (self-adjusting)',
    about='Only the settings that mattered; the matching radius and the "think twice" rule adjust themselves, at a price.',
    brains={'LeanBrain': LeanBrain},
    lean_brains=['LeanBrain'],
    params=[
        dict(name='tune', kind='brain', type='bool', default=True, owner=['LeanBrain', 'EnergyBrain'], group='Self-adjustment',
             about='Let the brain adjust its own matching radius. For every new experience it checks, before learning, what a smaller or larger radius '
                   'would have answered, and moves to a clearly better one. It looks less often once settled. The radius setting above then only '
                   'sets where it STARTS. Costs: tune cost on every thought, plus a delay while it looks.'),
        dict(name='deliberate', kind='brain', type='bool', default=True, owner=['LeanBrain', 'EnergyBrain'], group='Self-adjustment',
             about='Think twice when it matters. Normally the first matching neuron answers. If the matching neurons disagree about the '
                   'answer, the brain compares them and picks the best fit, paying the deliberation cost, but only for those conflicts.'),
        dict(name='tune_cost', kind='brain', type='float', default=2.0, min=0.01, max=500, step=0.5, owner=['LeanBrain', 'EnergyBrain'], group='Self-adjustment',
             about='The price of having the self-tuning circuit: microseconds added to EVERY thought while self-adjusting is on. Raise it to '
                   'see whether the benefit still pays for itself. Looking also adds a delay proportional to brain size.'),
        dict(name='deliberate_cost', kind='brain', type='float', default=10.0, min=0.01, max=500, step=1, owner=['LeanBrain', 'EnergyBrain'], group='Self-adjustment',
             about='The price of thinking twice: microseconds added to a thought only when the matching neurons disagree.'),
    ],
)
