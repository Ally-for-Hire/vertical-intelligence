# Energy economy: first findings (2026-10-18)

The energy brain (`lab/plugins/energy_brain.py`) pays energy (E) for everything: sensing, being alive, thinking (per microsecond), learning, growing, moving (with an
exponentially rising "squeeze" price as neurons get close), merging, pruning, searching for a better radius, and thinking twice. Income can come from free regeneration, from
growing, and from being right. Overload or an empty pool forces a **shutdown**; sustained high use builds **fatigue** that lowers the overload limit. Foundational rule: no action is free.
Everything below came from three Lab searches (about 30,000 simulated lifetimes, 6 repeats per combination): `A` standard world (2,000 combinations), `B` the same plus overload/fatigue/reboot
(1,500), `C` a harder, noisier world (1,500). "Good" = accuracy at least 95% (85% in C) and online at least 90% of the time.

## What the numbers say

1. **Natural economy works.** When neurons are allowed to squeeze as close as they can afford, thought time drops to 14 µs but moving alone burns about 6,300 E per life (against about 190 E for everything else).
   With the natural economy (move only if the expected energy saving beats the price), the same brain spends about **36 times less** and stays at 100% accuracy. Across the searches, economy-on brains were
   2 to 4 times more efficient (accuracy x uptime per energy: 0.026 vs 0.012 in A, 0.024 vs 0.006 in B).
2. **Diversified income is the safest way to live; growth-only income starves.** Brains relying on regeneration + growth + reward were good 90% of the time in both worlds (A and C). Brains relying on
   growth alone were good only 37% (A) and 26% (C) of the time and were offline about half the time: **once they stop growing the income stops, which is "stagnation is death" in the data.** Almost no income: 28% / 17% good.
3. **Only paying for neurons that fire is slightly better than paying for all of them** (efficiency 0.021 vs 0.017).
4. **Shutdown rules cost viability.** Adding overload and fatigue rules cut the good share from 64% to 34%. A looser overload limit and shorter shutdowns were more efficient. Fatigue (how much it lowers the limit,
   and how long it lingers) had almost no measurable effect at the ranges tried.
5. **Automatic neurons (known things become almost free)** cut the energy bill from about 30 to about 10 E per 1000 experiences at equal accuracy (3-run probe), and raise the "energy saved by experience" measure from about 42% to about 78%.
   More careful numbers come from the strategy search (see below).

## A caution about how to read the searches

The ranking by energy rewards **cheap prices**: the strongest "efficiency" link in A and B is simply the price of thinking (`c_think`), which is the world's rule, not a choice the brain makes. Prices and income
are *physics*; economize, horizon, reserve, automatic neurons, self-tuning and thinking twice are *strategy*. To find the best strategy the prices must be held fixed, which is what the follow-up **strategy search** does.
Also, correlation is a hint, not proof, because settings interact.

## Not yet done

* The strategy search (400 combinations x 4 repeats, 1 thread, low priority) was still running when this was written; its results are in the Lab.
* Everything here is a single problem per brain life. The user's central expectation, that each new problem connects to all previous ones and gets cheaper, needs the lifelong multi-problem run described in `FOUNDATIONS.md`.
* The compute limit (1 thread, low priority, `lab/limits.json`) means large searches now take a long time.
