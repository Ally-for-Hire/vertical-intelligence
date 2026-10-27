# selfscore: random-cull evolution of self-growing brains (one page)

**What it is.** A population of brains (any-direction recurrent networks with stochastic nodes and evolvable mechanisms) is scored on average accuracy over all 24 problems.
Each generation a cull fraction is drawn (small most of the time, rare mass culls, never above 0.9). The worst are removed; the empty slots are refilled with random
collections of the current brains (carbon copies, random variations, crossovers, blends, scrambles, fresh brains). Every brain also changes itself randomly every generation and keeps all
of it. Nothing else filters anything. Full design history: `DESIGN.md`, `RESEARCH_LOG.md`, `THEORY.md` (what the cull does, with exact identity and tests).

**Run it (the only commands you need).**
- Big run in the cloud (10,000 brains, 3,000 generations, one L4 GPU, 12 h cap, about $4-10; needs a payment method on the Modal account):
  `modal run --detach selfscore/modal_run.py`
- Look at it any time: `python selfscore/modal_peek.py` (prints the headline and the last story lines, saves the files to `selfscore/runs/modal/<tag>/`).
- Small local check (1,000 brains, a minute): `python selfscore/multi_big.py --tag check --P 1000 --gens 50`

**Where to look.** `runs/multi_<tag>.story.txt` is the plain-English log (one line per 10 generations, plus an event line when a problem is first solved). `runs/multi_<tag>_summary.md` is the
at-a-glance page: a headline, then every problem with a bar and a chance-adjusted score for the best brain. Cloud runs write the same files into the Modal volume `neuroblock-results`.

**Settings that matter** (defaults are the agreed ones): cull = 0.9 x u^6 (`--cull_power`), refill mix 25% copy / 25% variation / 20% crossover / 10% blend / 10% scramble / 10% fresh
(`--mode_p`), brain size cap 96 hidden nodes (`--hmax`), a 5-line task code on a shared 16-line input bus (`--no_code` removes the code).
