"""EnergyBrain: the lean brain with a body. Every action costs energy (E); E must be earned; overload or an empty pool forces a shutdown.

FOUNDATIONAL RULE (the user's): a brain works off energy, and EVERYTHING in life costs some, however small. So no action is free:
every cost setting has a small positive minimum, receiving an input costs energy (c_sense), simply being alive costs energy on every
experience (c_basal), a shut-down brain still pays a tiny resting cost, and upkeep cannot be switched off. Income is the only free thing.

THE ECONOMY (every number is a Lab setting, so the whole economy can be searched)
  Pool     One shared pool of E. It starts at `e_seed`. Each neuron brings `e_birth` E when it is born ("1 E for every connected
           neuron"). The pool holds at most `e_cap` per neuron; income beyond that is wasted.
  Costs    move      c_move per neuron per move, times a SQUEEZE factor that explodes as the neuron ends up close to its neighbours:
                     exp(squeeze_k * (1/length - 1)) for lengths under 1. There is no hard spacing floor (`floor` brings one back).
           think     c_think per microsecond of thought, so slow thoughts burn more energy and short wires pay back.
           spawn     c_spawn to grow a neuron (unaffordable = the new thing is simply not learned).
           learn     c_learn per update of a neuron's pattern.    upkeep: `upkeep` per neuron per experience (all / only the one that fired / none).
           merge, prune, tune (per neuron-candidate searched), deliberate (per use): c_merge, c_prune, tune_energy, deliberate_energy.
  Income   regen (free, per experience) | growth_yield (per neuron grown: complexity feeds it; with upkeep, stagnation starves it) |
           reward_yield (per correct answer while learning: eating when it succeeds).

FORCED SHUTDOWN (not death)
  Two triggers:  (1) the pool runs dry paying a mandatory cost (thinking, learning, growing, upkeep);
                 (2) OVERLOAD: the energy used per experience, over one cleanup round, passes a limit. The limit is `overload_limit` when
                     the brain is fresh, but sustained high use builds FATIGUE that lowers it:
                         limit_now = overload_limit * (1 - fatigue_gain * min(1, fatigue / overload_limit))
                     where fatigue is a running average of recent use (memory: `fatigue_windows` rounds). Rest lets fatigue fade.
  A shutdown lasts `reboot_after` cleanup rounds (0 = permanent). Offline, the brain misses its experiences, scores nothing, and rests
  (fatigue halves each round; regeneration keeps running). Then it reboots with whatever energy it has.
  Optional actions (moves, merges, prunes, searches, deliberation) are simply refused when unaffordable, so only mandatory costs can
  cause a shutdown. starve='brownout' replaces shutdown-on-empty with "pool stays at zero".

SLEEP (chosen by the brain; nothing schedules or forces it)
  can_sleep=True gives the brain a sleep it controls. Awake, it senses and learns, and only a quick radius check runs; the heavy tidying (move, merge, prune) waits for sleep.
  Two internal drives decide when to sleep, whichever is stronger:
    NEED (it 'has to'): work builds SLEEP DEBT (energy spent / need_scale). Ignoring it has consequences: while in debt, thinking and learning cost more (tired_cost),
         and if the debt ever reaches collapse_limit the brain collapses into a forced shutdown (a consequence of ignoring exhaustion, not a schedule; 0 = never).
    WANT (it 'wants to'): growth since the last sleep (and, with automatic neurons, how much of what it uses is not yet automatic) is unconsolidated; the wish to strengthen
         those connections is want_weight x backlog x (1 - novelty), so it does not want to sleep through a stretch where everything is new.
  It sleeps when max(need, want) reaches sleep_threshold. Asleep it senses nothing (those experiences are missed), pays only a resting cost, recovers fast, does the deferred
  tidying, and dreams (replays memories to strengthen neurons that were right). It wakes by itself once rested and consolidated (wake_level), with sleep_max as a runaway guard.

NATURAL ECONOMY (how it minimizes energy use by itself; switch off with economize=False to compare)
  * A move is made only if it is WORTH IT: the energy expected to be saved on future thinking (shorter wires x how much traffic uses
    them x `horizon` rounds ahead x c_think x microseconds-per-unit) must beat the price of the move. Because the price explodes with
    closeness while the saving grows gently, an optimum spacing appears by itself: no floor, no tuning.
  * Optional actions never dip into a reserve of `reserve` x the pool's capacity, which is kept for mandatory costs.
  * It looks for a better radius less and less often once settled, and thinks twice only when matching neurons disagree.

With every price forced to zero in code (the Lab refuses zero prices), floor=0.3, economize=False and a huge e_seed, EnergyBrain behaves exactly like LeanBrain (lab_selftest.py).
"""
import math

import numpy as np

import neuroblock
from brain2 import Brain2
from lean_brain import LeanBrain

ACTIONS = ('move', 'think', 'spawn', 'learn', 'upkeep', 'merge', 'prune', 'tune', 'deliberate', 'sense', 'basal')
REST_FRACTION = 0.1      # a shut-down brain still pays this share of the basal cost
MOVE_EPS = 0.002        # a planned move smaller than this means "already there": no move, no cost
FRESH_WINDOWS = 3       # no overload shutdown during the first rounds (the brain is still being born)
MIN_AWAKE = 2           # a brain never chooses to sleep before it has been awake this many rounds
SLEEP_COOLING = 0.3     # fatigue left after each round of sleep (shutdown rest keeps 0.5)
DREAM_DISCOUNT = 0.25   # replaying a memory costs this share of a real learning update
NOVELTY_SCALE = 10.0    # this many new neurons in one round counts as 'everything is new'
TIRED_CAP = 3.0         # sleep debt beyond this adds no further tiredness cost


class EnergyBrain(LeanBrain):
    def __init__(self, D, K, rng, rho=0.5, tune=True, deliberate=True, tune_cost=2.0, deliberate_cost=10.0,
                 e_seed=5.0, e_birth=1.0, e_cap=4.0, regen=0.05, growth_yield=0.0, reward_yield=0.0,
                 c_move=0.2, squeeze_k=0.3, floor=0.0, c_think=0.0005, c_spawn=0.3, c_learn=0.002,
                 upkeep=0.00005, upkeep_mode='all', c_sense=0.0005, c_basal=0.001, c_merge=0.1, c_prune=0.05, tune_energy=0.002, deliberate_energy=0.02,
                 starve='shutdown', reboot_after=4, overload_limit=0.0, fatigue_gain=0.8, fatigue_windows=6,
                 economize=True, horizon=8, reserve=0.2, automatic_after=0.0, automatic_floor=0.02, can_sleep=False, sleep_threshold=0.8, need_scale=60.0, tired_cost=0.5, want_weight=1.0, collapse_limit=3.0, wake_level=0.2, sleep_max=6, dream_replay=150, **_ignored):
        super().__init__(D, K, rng=rng, rho=rho, tune=tune, deliberate=deliberate, tune_cost=tune_cost, deliberate_cost=deliberate_cost)
        self.min_sep = float(floor)  # 0 = no hard floor; the squeeze price is the only brake
        self.E = float(e_seed)
        self.e_birth, self.e_cap = float(e_birth), float(e_cap)
        self.regen, self.growth_yield, self.reward_yield = float(regen), float(growth_yield), float(reward_yield)
        self.c_move, self.squeeze_k = float(c_move), float(squeeze_k)
        self.c_think, self.c_spawn, self.c_learn = float(c_think), float(c_spawn), float(c_learn)
        self.upkeep, self.upkeep_mode = float(upkeep), upkeep_mode
        self.c_sense, self.c_basal = float(c_sense), float(c_basal)
        self.c_merge, self.c_prune = float(c_merge), float(c_prune)
        self.tune_energy, self.deliberate_energy = float(tune_energy), float(deliberate_energy)
        self.starve, self.reboot_after = starve, int(reboot_after)
        self.overload_limit, self.fatigue_gain, self.fatigue_windows = float(overload_limit), float(fatigue_gain), max(1.0, float(fatigue_windows))
        self.economize, self.horizon, self.reserve = bool(economize), float(horizon), float(reserve)
        self.automatic_after, self.automatic_floor = float(automatic_after), float(automatic_floor)
        self._hist = []  # (experiences lived, total energy spent) after each cleanup round, to measure the learning curve
        self.can_sleep, self.sleep_threshold, self.need_scale = bool(can_sleep), float(sleep_threshold), max(1e-6, float(need_scale))
        self.tired_cost, self.want_weight, self.collapse_limit = float(tired_cost), float(want_weight), float(collapse_limit)
        self.wake_level, self.sleep_max, self.dream_replay = float(wake_level), max(1, int(sleep_max)), int(dream_replay)
        self.asleep, self.awake_windows, self.slept_for = False, 0, 0
        self.debt, self._debt_mark, self.new_since_sleep, self.window_spawns = 0.0, 0.0, 0.0, 0
        self.need_now, self.want_now = 0.0, 0.0
        self.sleeps, self.tired_sleeps, self.sleep_windows = 0, 0, 0
        self.spent = dict.fromkeys(ACTIONS, 0.0)
        self.income = dict(birth=0.0, regen=0.0, growth=0.0, reward=0.0, wasted=0.0)
        self.offline_left, self.shutdowns, self.reason = 0, 0, ''
        self.windows, self.offline_windows, self.steps = 0, 0, 0
        self.fatigue, self.threshold = 0.0, self.overload_limit
        self._spent0, self._steps0 = 0.0, 0

    # ---- the ledger ----
    @property
    def offline(self):
        return self.offline_left > 0

    def _total_spent(self):
        return sum(self.spent.values())

    def _capacity(self):
        return self.e_cap * (self.n + 1)

    def _gain(self, kind, amount):
        if amount > 0:
            got = min(amount, max(0.0, self._capacity() - self.E))
            self.E += got
            self.income[kind] += got
            self.income['wasted'] += amount - got

    def _shutdown(self, reason):
        self.shutdowns += 1
        self.reason = reason
        self.offline_left = self.reboot_after if self.reboot_after > 0 else 10 ** 9

    def _pay(self, kind, amount, mandatory):
        """Mandatory costs are always paid and can force a shutdown. Optional ones are refused if they would dip into the reserve."""
        if amount <= 0:
            return True
        if mandatory:
            self.E -= amount
            self.spent[kind] += amount
            if self.E < 0:
                self.E = 0.0
                if self.starve == 'shutdown' and not self.offline:
                    self._shutdown('empty')
            return True
        if self.E - amount >= self.reserve * self._capacity():
            self.E -= amount
            self.spent[kind] += amount
            return True
        return False

    def _squeeze(self, length):
        """the exponential price of squeezing close together"""
        if length >= 1.0 or self.squeeze_k <= 0:
            return 1.0
        return min(1e9, math.exp(self.squeeze_k * (1.0 / max(length, 1e-3) - 1.0)))

    # ---- hooks from the lean brain ----
    def _on_conflict(self):
        self._pay('deliberate', self.deliberate_energy, True)

    def _allow_search(self, work):
        return self.E - self.tune_energy * work >= self.reserve * self._capacity()

    def _on_search(self, work):
        self._pay('tune', self.tune_energy * work, False)

    # ---- one experience ----
    def step(self, x, y):
        if self.offline or self.asleep:
            self._rest()  # even resting costs a little; resting still recovers. Asleep: no sensing, no learning
            return 0.0, 0, False
        self.steps += 1
        self._pay('sense', self.c_sense, True)   # receiving any input costs energy
        self._pay('basal', self.c_basal, True)   # so does simply being alive
        if self.offline:                         # that may just have emptied the pool
            return 0.0, 0, False
        needs_spawn = True
        if self.n > 0:
            needs_spawn = not (np.linalg.norm(self.w[1:] - x, axis=1) < self.rho).any()
        if needs_spawn and self.E < self.c_spawn:
            t = (self.d.max() if self.n > 0 else 0.0) + self.spawn_cost  # tried to grow but cannot afford it: not learned
            self._pay('think', self.c_think * t, True)
            self._passive(None)
            return t, 0, False
        t, idx, spawned = super().step(x, y)
        f = 1.0 if spawned else self._discount(idx)  # something new costs full price; something well known costs almost nothing
        if self.can_sleep:
            f *= 1.0 + self.tired_cost * min(self.debt, TIRED_CAP)  # ignoring the need to sleep makes everything cost more
        self._pay('think', self.c_think * t * f, True)
        if spawned:
            self._pay('spawn', self.c_spawn, True)
            self.window_spawns += 1
            self.new_since_sleep += 1.0
            self._gain('birth', self.e_birth)
            self._gain('growth', self.growth_yield)
        else:
            self._pay('learn', self.c_learn * f, True)
            if self.reward_yield and int(self.lab[idx].argmax()) == int(y):
                self._gain('reward', self.reward_yield)
        self._passive(None if spawned else idx)
        return t, idx, spawned

    def _passive(self, fired):
        self._pay('upkeep', self.upkeep * (1 if self.upkeep_mode == 'active' and fired is not None else (0 if self.upkeep_mode == 'active' else self.n)), True)
        self._gain('regen', self.regen)

    def _rest(self):
        cost = min(self.E, self.c_basal * REST_FRACTION)
        self.E -= cost
        self.spent['basal'] += cost
        self._gain('regen', self.regen)

    def query(self, X, Y):
        if self.offline:
            return self.spawn_cost, 0.0, 0.0  # offline: answers nothing
        return super().query(X, Y)

    # ---- the cleanup round: prices, fatigue, shutdown ----
    def condense(self):
        self.windows += 1
        if self.offline:
            self.offline_windows += 1
            self.fatigue *= 0.5  # resting cools it down
            self.offline_left -= 1
            self._spent0, self._steps0 = self._total_spent(), self.steps
            return
        if self.can_sleep:
            if self.asleep:
                self._sleep_window()
                return
            self.awake_windows += 1
            if self.tune:
                self._tune_radius()  # the quick check that must run live; the heavy tidying waits for sleep
            self._hist.append((self.steps, self._total_spent()))
            self._check_overload()
            self._decide_sleep()
            return
        super().condense()
        self._hist.append((self.steps, self._total_spent()))
        self._check_overload()

    def _check_overload(self):
        used = self._total_spent() - self._spent0
        lived = max(1, self.steps - self._steps0)
        u = used / lived                                        # energy used per experience in this round
        if self.overload_limit > 0:
            self.threshold = self.overload_limit * max(0.05, 1.0 - self.fatigue_gain * min(1.0, self.fatigue / self.overload_limit))
            if self.windows > FRESH_WINDOWS and u > self.threshold and self.starve == 'shutdown' and not self.offline:
                self._shutdown('overload')
        self.fatigue += (u - self.fatigue) / self.fatigue_windows  # sustained use builds fatigue, quiet lets it fade
        self._spent0, self._steps0 = self._total_spent(), self.steps

    def _apply(self, redirect, merge_into):
        if not redirect:
            return
        unit, kind = (self.c_merge, 'merge') if merge_into else (self.c_prune, 'prune')
        if unit > 0:
            room = max(0.0, self.E - self.reserve * self._capacity())
            k = min(len(redirect), int(room // unit))  # only as many as it can afford
            if k < len(redirect):
                keep = list(redirect)[:k]
                redirect = {j: redirect[j] for j in keep}
                merge_into = {j: merge_into[j] for j in keep} if merge_into else {}
            if not redirect:
                return
            self._pay(kind, unit * len(redirect), False)
        super()._apply(redirect, merge_into)

    def _move(self):
        s, n = self, len(self.pos)
        s.pos = s.pos + s.rng.normal(0, 1e-3, s.pos.shape) * (np.arange(n)[:, None] > 0)
        sub = s.ema.copy()
        sub[0] = 0
        for i in range(n - 1, 0, -1):
            sub[s.par[i]] += sub[i]
        sub = sub + 0.2
        num, den = np.zeros((n, 2)), np.zeros(n)
        for i in range(1, n):
            p, wgt = s.par[i], sub[i]
            num[i] += wgt * s.pos[p]
            den[i] += wgt
            if p > 0:
                num[p] += wgt * s.pos[i]
                den[p] += wgt
        mask = den > 0
        mask[0] = False
        ids = np.where(mask)[0]
        if not len(ids):
            return
        target = num[ids] / den[ids, None]
        new = s.pos[ids] + s.eta_vec()[ids][:, None] * (target - s.pos[ids])
        size = np.linalg.norm(new - s.pos[ids], axis=1)
        kids = {}
        for c in range(1, n):
            kids.setdefault(int(s.par[c]), []).append(c)
        rate = 0.2  # ema is a decayed sum (about 5 x per-round use), so this converts it to uses per round
        for k in np.argsort(-sub[ids]):  # busiest neurons first
            if size[k] <= MOVE_EPS and (self.c_move > 0 or self.economize):
                continue  # already there: no move, no cost
            i = int(ids[k])
            p = int(s.par[i])
            edges = [(p, sub[i])] + [(c, sub[c]) for c in kids.get(i, [])]  # (other end, traffic) of every wire on this neuron
            old_len = [float(np.linalg.norm(s.pos[i] - s.pos[o])) for o, _ in edges]
            new_len = [float(np.linalg.norm(new[k] - s.pos[o])) for o, _ in edges]
            cost = self.c_move * self._squeeze(min(new_len))
            if self.economize:
                shorter = sum(w * (a - b) for (o, w), a, b in zip(edges, old_len, new_len))  # traffic-weighted wire saved
                saving = self.c_think * neuroblock.US * rate * shorter * self.horizon        # energy saved on future thinking
                if saving <= cost:
                    continue  # not worth the price
            if self.c_move > 0 and not self._pay('move', cost, False):
                continue  # cannot afford this one; a cheaper one may still fit
            s.pos[i] = new[k]
        for _ in range(8):
            s._separate()

    # ---- sleep: the brain's own choice, driven by need and want ----
    def _backlog(self):
        """how much recent learning is not yet consolidated: growth since the last sleep, and (with automatic neurons) how much of its knowledge is not yet automatic"""
        growth = min(1.0, self.new_since_sleep / max(self.n, 1))
        if self.automatic_after > 0 and self.n > 0:
            return max(growth, 1.0 - float(np.mean(self._habit(np.arange(1, len(self.cnt))))))
        return growth

    def _decide_sleep(self):
        spent = self._total_spent()
        self.debt += (spent - self._debt_mark) / self.need_scale  # work done builds sleep debt
        self._debt_mark = spent
        novelty = min(1.0, self.window_spawns / NOVELTY_SCALE)
        self.window_spawns = 0
        self.need_now = self.debt
        self.want_now = self.want_weight * self._backlog() * (1.0 - novelty)  # not worth sleeping through a stretch where everything is new
        self.pressure = max(self.need_now, self.want_now)
        if self.collapse_limit > 0 and self.debt >= self.collapse_limit and not self.offline:
            self._shutdown('exhaustion')  # a consequence of ignoring the need, not a schedule
            return
        if self.pressure >= self.sleep_threshold and self.awake_windows >= MIN_AWAKE:
            self.asleep, self.slept_for = True, 0
            self.sleeps += 1
            self.tired_sleeps += int(self.need_now >= self.want_now)

    def _sleep_window(self):
        self.sleep_windows += 1
        self.slept_for += 1
        self.fatigue *= SLEEP_COOLING
        Brain2.condense(self)   # the deferred tidying (move, merge, prune): offline, but still priced
        self._dream()
        self.debt *= 0.4        # rest pays the debt down quickly
        self.new_since_sleep *= 0.5
        if (self.debt < self.wake_level and self._backlog() < self.wake_level) or self.slept_for >= self.sleep_max:
            self.asleep, self.awake_windows, self._debt_mark = False, 0, self._total_spent()  # it wakes itself once rested and consolidated
        self._spent0, self._steps0 = self._total_spent(), self.steps
        self._hist.append((self.steps, self._total_spent()))

    def _dream(self):
        """replay remembered experiences: strengthens neurons that were right (so they become more automatic). Never grows new neurons."""
        if self.dream_replay <= 0 or self.n == 0 or not self.bx:
            return
        pick = self.rng.choice(len(self.bx), size=min(self.dream_replay, len(self.bx)), replace=False)
        for i in pick:
            x, y = self.bx[i], self.by[i]
            dist = np.linalg.norm(self.w[1:] - x, axis=1)
            m = np.where(dist < self.rho)[0]
            if not len(m):
                continue
            win = int(m[np.argmin(dist[m])]) + 1
            if not self._pay('learn', self.c_learn * DREAM_DISCOUNT, False):
                break  # too tired or too poor to keep dreaming
            self.w[win] += self.lr_of(win) * (x - self.w[win])
            self.cnt[win] += 1
            self.lab[win, y] += 1

    # ---- mastery: what it knows well becomes (almost) free ----
    def _habit(self, idx):
        lab = self.lab[idx]
        tot = lab.sum(axis=-1)
        purity = np.where(tot > 0, lab.max(axis=-1) / np.maximum(tot, 1e-9), 0.0)  # how reliably this neuron is right
        return purity * (1.0 - np.exp(-self.cnt[idx] / self.automatic_after))                # ... and how often it has answered

    def _discount(self, idx):
        if self.automatic_after <= 0:
            return 1.0
        return self.automatic_floor + (1.0 - self.automatic_floor) * (1.0 - float(self._habit(idx)))

    # ---- reporting ----
    def diag(self):
        d = super().diag()
        d.update(E=self.E, online=0.0 if self.offline else 1.0, fatigue=self.fatigue, shutdown_limit=self.threshold)
        d.update(asleep=1.0 if self.asleep else 0.0, sleep_need=self.need_now, sleep_want=self.want_now)
        d['automatic'] = float(np.mean(self._habit(np.arange(1, len(self.cnt))))) if self.automatic_after > 0 and self.n > 0 else 0.0
        for k, v in self.spent.items():
            d['spent_' + k] = v
        return d

    def energy_report(self):
        """run metrics: energy per 1000 experiences lived, split by what it was spent on, plus how reliably it stayed online"""
        lived = max(self.steps, 1)
        per = lambda v: 1000.0 * v / lived
        sp = self.spent
        h = self._hist
        mastery = float('nan')
        if len(h) >= 6:  # how much cheaper per experience the last third of life was than the first third
            k = len(h) // 3
            early = h[k - 1][1] / max(h[k - 1][0], 1)
            late = (h[-1][1] - h[-k - 1][1]) / max(h[-1][0] - h[-k - 1][0], 1)
            mastery = 1.0 - late / early if early > 0 else float('nan')
        return dict(sleep_share=self.sleep_windows / max(self.windows, 1), sleeps=float(self.sleeps), tired_sleeps=float(self.tired_sleeps), mastery=mastery, uptime=1.0 - self.offline_windows / max(self.windows, 1), shutdowns=float(self.shutdowns), E=per(sum(sp.values())),
                    Eleft=self.E, e_move=per(sp['move']), e_think=per(sp['think']), e_spawn=per(sp['spawn']), e_upkeep=per(sp['upkeep'] + sp['basal']),
                    e_other=per(sp['learn'] + sp['merge'] + sp['prune'] + sp['tune'] + sp['deliberate'] + sp['sense']))


def _p(name, typ, default, about, group, **kw):
    return dict(name=name, kind='brain', type=typ, default=default, owner='EnergyBrain', group=group, about=about, **kw)


PLUGIN = dict(
    name='Energy brain',
    about='The lean brain with a body: every action costs energy (E), income must be earned, and overload or an empty pool forces a shutdown.',
    brains={'EnergyBrain': EnergyBrain},
    lean_brains=['EnergyBrain'],
    params=[
        _p('e_seed', 'float', 5.0, 'Energy (E) in the pool before any neuron exists. Needs to cover the first few neurons.', 'Energy: supply', min=0, max=1e4, step=0.5),
        _p('e_birth', 'float', 1.0, 'E that each newly grown neuron brings into the shared pool ("1 E for every connected neuron").', 'Energy: supply', min=0, max=100, step=0.1),
        _p('e_cap', 'float', 4.0, 'The most E the pool can hold per neuron. Income beyond this is wasted.', 'Energy: supply', min=0.1, max=1000, step=0.5),
        _p('regen', 'float', 0.05, 'Free energy regained per experience (like a blood supply). 0 = none. A healthy brain uses roughly 0.035 per experience, so 0.05 is viable but not lavish.', 'Energy: supply', min=0, max=100, step=0.005),
        _p('growth_yield', 'float', 0.0, 'E gained for every neuron grown. With upkeep on, a brain that stops growing starves ("stagnation is death").', 'Energy: supply', min=0, max=100, step=0.1),
        _p('reward_yield', 'float', 0.0, 'E gained for every correct answer while learning (eating when it succeeds).', 'Energy: supply', min=0, max=100, step=0.005),
        _p('c_move', 'float', 0.2, 'E to move one neuron one step. The price is multiplied by the squeeze factor when the neuron ends up very close to its neighbours.', 'Energy: costs', min=1e-6, max=100, step=0.05),
        _p('squeeze_k', 'float', 0.3, 'How steeply the price of moving close explodes: price multiplier = exp(k x (1/length - 1)) for lengths under 1. Larger = closeness gets expensive sooner. 0 = no extra price.', 'Energy: costs', min=0, max=20, step=0.05),
        _p('floor', 'float', 0.0, 'A hard minimum spacing between neurons. 0 = none: neurons may get as close as they can afford. 0.3 = the old fixed speed limit.', 'Energy: costs', min=0, max=3, step=0.05),
        _p('c_think', 'float', 0.0005, 'E burned per microsecond of thought. Slow thoughts cost more, so shorter wires save energy.', 'Energy: costs', min=1e-6, max=10, step=0.0001),
        _p('c_spawn', 'float', 0.3, 'E to grow a new neuron. If the brain cannot afford it, the new thing is simply not learned.', 'Energy: costs', min=1e-6, max=100, step=0.05),
        _p('c_learn', 'float', 0.002, "E for each update of a neuron's remembered pattern.", 'Energy: costs', min=1e-6, max=10, step=0.001),
        _p('upkeep', 'float', 0.00005, 'Passive running cost per neuron per experience, always on (nothing in life is free). Who pays is set by "upkeep mode".', 'Energy: costs', min=1e-6, max=10, step=0.00001),
        _p('c_sense', 'float', 0.0005, 'E to receive each input (sensing is not free).', 'Energy: costs', min=1e-6, max=10, step=0.0001),
        _p('c_basal', 'float', 0.001, 'E per experience just to be alive (basal metabolism). A shut-down brain still pays a tenth of it.', 'Energy: costs', min=1e-6, max=10, step=0.0001),
        _p('upkeep_mode', 'choice', 'all', 'Who pays upkeep: every neuron all the time (all), or only the neuron that fired (active). Upkeep cannot be switched off.', 'Energy: costs', choices=['all', 'active']),
        _p('c_merge', 'float', 0.1, 'E to merge two near-duplicate neurons.', 'Energy: costs', min=1e-6, max=100, step=0.05),
        _p('c_prune', 'float', 0.05, 'E to delete an unused neuron.', 'Energy: costs', min=1e-6, max=100, step=0.05),
        _p('tune_energy', 'float', 0.002, 'E per neuron per candidate radius that the self-tuning search examines. Bigger brains pay more to look.', 'Energy: costs', min=1e-6, max=10, step=0.001),
        _p('deliberate_energy', 'float', 0.02, 'E each time the brain thinks twice because matching neurons disagreed.', 'Energy: costs', min=1e-6, max=10, step=0.005),
        _p('starve', 'choice', 'shutdown', 'What happens when mandatory costs drain the pool, or overload hits. shutdown: the brain is forced offline for a while. brownout: it stays on but optional actions stop; overload is ignored.', 'Energy: shutdown', choices=['shutdown', 'brownout']),
        _p('reboot_after', 'int', 4, 'How many cleanup rounds a forced shutdown lasts. 0 = permanent (the brain never comes back). Offline, it misses experiences, scores nothing, and rests.', 'Energy: shutdown', min=0, max=1000, step=1),
        _p('overload_limit', 'float', 0.0, 'Overload shutdown: the most energy the brain may burn per experience (in one cleanup round) when fresh. Sustained high use builds fatigue, which LOWERS this limit for a while. 0 = no overload shutdown.', 'Energy: shutdown', min=0, max=100, step=0.01),
        _p('fatigue_gain', 'float', 0.8, 'How much fatigue lowers the overload limit: limit now = limit x (1 - gain x fatigue/limit). 0 = fatigue has no effect; near 1 = steady high use drives the limit towards zero.', 'Energy: shutdown', min=0, max=1, step=0.05),
        _p('fatigue_windows', 'float', 6.0, 'How long fatigue lingers: the memory, in cleanup rounds, of recent energy use. Longer = slower to build and slower to fade.', 'Energy: shutdown', min=1, max=200, step=1),
        _p('economize', 'bool', True, 'Natural economy: a neuron only moves if the energy it expects to save on future thinking beats the price of the move. Off = it moves whenever it can afford it.', 'Energy: economy'),
        _p('horizon', 'float', 8.0, 'How many cleanup rounds ahead the brain counts the savings of a move. Longer = more willing to pay now for later savings.', 'Energy: economy', min=0, max=500, step=1),
        _p('automatic_after', 'float', 0.0, 'Mastery: how many times a neuron must answer (reliably) before using it is almost free. A neuron that has answered often and is nearly always right becomes automatic, and answering with it costs only the "automatic floor" share of the normal energy. Something new always costs full price. 0 = off. Small = learns habits fast.', 'Energy: economy', min=0, max=1000, step=1),
        _p('automatic_floor', 'float', 0.02, 'The share of the normal energy that a fully automatic answer still costs (nothing is entirely free).', 'Energy: economy', min=0.001, max=1, step=0.01),
        _p('can_sleep', 'bool', False, 'Gives the brain a sleep that IT controls (nothing schedules or forces it). Awake it senses and learns; the heavy tidying waits for sleep. Asleep it senses nothing, rests, tidies and dreams. Off = it tidies a little after every round and never sleeps.', 'Energy: sleep'),
        _p('sleep_threshold', 'float', 0.8, 'How readily it chooses to sleep: it sleeps when its strongest drive (need or want) reaches this level. Lower = sleeps earlier and more often; higher = holds out longer.', 'Energy: sleep', min=0.05, max=5, step=0.05),
        _p('need_scale', 'float', 60.0, 'NEED to sleep: sleep debt grows by (energy spent) / this, so the need reaches 1 after this much work since the last sleep. Smaller = it tires sooner.', 'Energy: sleep', min=1, max=100000, step=5),
        _p('tired_cost', 'float', 0.5, 'The consequence of ignoring the need: while in sleep debt, thinking and learning cost (1 + this x debt) times as much. 0 = being tired costs nothing.', 'Energy: sleep', min=0, max=20, step=0.1),
        _p('want_weight', 'float', 1.0, 'WANT to sleep: how much it wants to strengthen connections after recent growth (and not-yet-automatic knowledge). It does not want to sleep through a stretch where everything is new. 0 = it only ever sleeps from need.', 'Energy: sleep', min=0, max=10, step=0.1),
        _p('collapse_limit', 'float', 3.0, 'If sleep debt ever reaches this, the brain collapses into a forced shutdown. This is the consequence of ignoring exhaustion for too long, not a schedule. 0 = never collapses.', 'Energy: sleep', min=0, max=100, step=0.5),
        _p('wake_level', 'float', 0.2, 'It wakes by itself once both its need and its backlog of unconsolidated learning are below this.', 'Energy: sleep', min=0.01, max=2, step=0.05),
        _p('sleep_max', 'int', 6, 'Safety guard: the most rounds one sleep can last, so a brain that never feels consolidated cannot sleep forever.', 'Energy: sleep', min=1, max=100, step=1),
        _p('dream_replay', 'int', 150, 'How many remembered experiences are replayed per round of sleep. Replaying strengthens neurons that were right (making them more automatic). Each replay costs a quarter of a real learning update. It never grows new neurons.', 'Energy: sleep', min=0, max=5000, step=10),
        _p('reserve', 'float', 0.2, 'A share of the pool that optional actions (moves, merges, searches...) may never touch, kept for mandatory costs so the brain is not shut down by its own tidying.', 'Energy: economy', min=0, max=0.95, step=0.05),
    ],
)
