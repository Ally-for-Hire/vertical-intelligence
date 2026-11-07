# Foundations: what the brain is (the user's framing, 2026-10-18)

Status: **the user's statements are fixed givens. Everything marked "proposed" is Claude's attempt to make them testable and has not been agreed.**

## What the user said (given)

1. The brain is a machine that **converts the body's senses into complex, abstract understanding, and lowers uncertainty in its environment.**
   The user's own caveat: this is hard to work from, and it is where the project's blank area is.
2. It is **fundamentally split into parts, not one thing.** Probably at least five, judging by the human brain. Three named so far:
   * **Emotional converter**: converts inputs into emotions.
   * **Intellectual converter**: converts inputs into context graphs that can be followed.
   * **Self**: takes all of this input and decides what to do with it, how to live, how to act.
3. **Building the Self, the bottom-level self, is probably the main focus of the project.** Everything else mainly defines the scope and
   the ideas, and what a brain even is.
4. **Energy is foundational.** A brain works off energy; everything in life requires some, even a small amount. So no action in the simulator is free:
   every cost has a positive minimum, receiving an input costs energy, simply being alive costs energy, even a shut-down brain pays a tiny resting cost,
   and passive upkeep cannot be switched off. Income is the only free thing, and it has to be given or earned.
5. **The end product is probably a few parts** (the user is fuzzy on the details): *a brain*, meaning a physical representation that can grow and that is a
   thinking machine (takes inputs and contexts, produces outputs); plus **a purpose**. Once the brain is defined it must be given a reason to do more:
   a **learning criterion, a de facto meaning of life**, and with it the **stepping stones to consciousness.**
6. **Efficiency is vital.** The user would like a machine with no energy concerns, but any end result must maximize efficiency.
7. **The end result is a general, abstract problem-solver.** It takes a set of inputs (some situation), has *some feeling about that situation*, and then uses its
   understanding and architecture to create a solution to an arbitrary problem. It should apply to **any problem that can be framed**: solve it, **or learn from it and do it
   later, or see something and connect it to what it knows and recreate it.** It must **improve upon itself, ideally infinitely**: an infinitely growing machine that takes
   inputs, **recognizes what it does not understand about them, and pushes at the boundaries with everything else it knows until it finds a solution.**
8. **Infinite growth should come from free solving.** The brain should be able to solve, *instantly and with zero energy*, any problem it already knows how to solve. Once it has a
   solution, using it becomes free, immediately for best efficiency (human brains sometimes take a while to "intellectualize"). So the machine is **infinitely recursive**, and may have
   as much storage as it wants, but **storage is not the point: it should be small.** Store as little as possible long term and pull from it, because reuse is the most energy-efficient
   thing, rather than developing a new solution for every new problem. The user expects a brain that, for **every new problem, builds new patterns and connections with every previous
   problem**, so that each new problem is solved faster and with less energy. It should generate the best solutions from the *minimal context*, which still requires foundational
   understanding. The machine should be **based on the principles of life** ("life is better at making brains than humans are"); human brains are probably not the most efficient thinking
   machines, but they are the only place to start, and the smartest thing we know, so they are worth studying.
9. **It should be given problems outside its scope**, for example predicting patterns in 4-dimensional abstractions (later), and it should **build its own efficient architecture**: self-refining and
   self-learning. It should not only want an answer but **understand every problem so intrinsically that it takes zero energy to do it again.**
10. **All problems intrinsically relate** (the user's belief). Getting really good at one thing (math, a game) should teach the machine *how to think*: how to understand a subject deeply and quickly apply that to any other problem.
    The user's view of genius is **motivation plus processing speed, motivation being the big one**, and the user wants to work from **what works in humans** (including books on maximizing the brain), because humans are the only place to start.
11. **The universe is built on foundational blocks, and everything is connected to them** (the user's belief, which they hold even after the far-transfer evidence; the research says transfer is weak *unless structure is shared*, which is exactly what this belief claims is everywhere).
    The machine should **recognize the underlying connections in everything and self-improve on those concepts.** It should *not* necessarily get better at specific problems; it should get better **at taking information and producing more information.**
12. Earlier principles still stand (see `README_HANDOFF.md`): time in microseconds; neurons on a physical graph that move; condense toward
   a compact shape; hardening; trusting outside signals with decay and treating internal signals as biased data; no spawning that is negative
   for the whole; reward effort only while uncertainty is dropping. And from the energy discussion: every action costs energy (E), a brain starts
   with 1 E per neuron, moving costs 0.2 E, closeness gets exponentially expensive, income can be regeneration / growth / reward, passive costs
   may apply to everything or only to some actions, overload and fatigue force shutdowns, and the brain should **minimize energy use naturally.**

## Where the current simulator sits against that picture (observed)

| Part in the user's picture | What exists today | Honest gap |
|---|---|---|
| Senses | Sensor hub plus noisy examples from a toy world | Only one sense; no body signals except energy |
| Intellectual converter | The neuron map: neurons are concepts, wires are a tree. This is a primitive "context graph" | Wires are a spawn tree, not meaningful relations; nothing can be "followed" yet |
| Emotional converter | **None** | Not built |
| Self | **None as one thing.** Scattered rules already make Self-like decisions: self-tuned radius, think-twice, "is this move worth it", the energy reserve | These are hard-wired rules inside the brain, not a coordinating part |
| Body | The energy economy: pool, costs, income, fatigue, forced shutdown | New (this session) |

So today's machine is roughly **an intellectual converter with a body**. The Self and the emotional converter are the blank areas.

## Proposed: making "lowers uncertainty" measurable (not yet agreed)

* **Uncertainty = surprise:** how often (and how wrongly) the brain's prediction, made *before* it learns from an input, turns out wrong.
  The self-tuning radius already tallies exactly this (prequential prediction error), so there is a number to work from today.
* **"Lowering uncertainty" = surprise falling over time**, and it should fall at the lowest energy cost. This ties the two ideas together:
  the brain's purpose is *less surprise per unit of energy.*
* **Abstract understanding** = a structure that predicts well across situations, not just stored examples. A test: does it keep predicting after
  the world changes (drift), and with fewer neurons?

## Proposed: a minimal, testable version of each part (not yet agreed)

* **Emotional converter v0:** a few signals computed from the input stream and the body, read by the Self. For example:
  *alarm* = surprise rate; *relief/interest* = surprise falling; *strain* = low energy or high fatigue; *curiosity* = surprise the brain can afford to chase.
  The user's wording says it converts *inputs* into emotions, so a v1 should react to properties of the input itself (novelty, similarity to known threats), not only to internal state.
* **Intellectual converter v1:** the neuron map with **meaningful** wires (relations between concepts, general graph with shortcuts), so a path through it can be followed.
* **Self v0:** *one* controller that reads the emotion signals, the energy state and the map, and decides among a small menu: explore (grow), consolidate
  (merge, prune, move), think twice, rest (avoid shutdown), and what to adjust. Today's scattered rules would move into it. **Being a Self has a price**
  (energy and time per decision), so the sweep of its own cost is a real experiment: how much Self is worth paying for?
* **Other parts (the user expects at least five):** candidates to discuss, not decided: memory/consolidation, attention, action/motor, a body regulator
  (interoception: hunger, fatigue). Which does the user have in mind?

## Proposed: "what defines a brain" as a checklist we can test by removing parts

A system counts as brain-like here if it (1) turns senses into a compact predictive representation, (2) lowers its surprise over time, (3) pays for
everything from an energy budget and can be forced offline, (4) has several parts that can disagree and be coordinated, and (5) has a Self that integrates them.
Each can be tested by **ablation**: switch one part off and measure what fails (surprise, energy, uptime, adaptation after change).

## Generality and open-ended growth (the user's target; the tests below are proposed)

What the user describes is a loop: **situation in, feeling about it, recognize what is not understood, push at that boundary using everything already known, find a solution,
keep what was learned, repeat forever.** Mapped to what exists and what does not:

| Step of the loop | In the simulator today | Gap |
|---|---|---|
| Situation in | A noisy example from a toy world | One narrow kind of problem |
| Feeling about it | Nothing (emotional converter not built) | Needs the emotion signals |
| Recognize what is not understood | **Partly:** an unrecognised input creates a neuron; prediction error is measured before learning | Only detects "far from anything known", not *what* is missing |
| Push at the boundary with what it knows | **Not built** | Needs combining known concepts to guess a new one (analogy / composition), then testing the guess |
| Find a solution | A label prediction | No solutions beyond classifying; no outputs that act on the world |
| Keep what was learned, reuse it | Neurons persist and adapt | Nothing tests reuse on a *different* problem |
| Improve itself, forever | Self-tuned radius, condensing | No open-ended growth of ability; growth is capped by cost |

### A tension to decide on

"Infinitely growing" and "energy is foundational" pull against each other: with finite energy, growth has to **pay for itself.** One consistent reading is that
unbounded growth is only possible if *efficiency keeps improving*, for example by compressing what it knows so the same understanding costs less to run, while the
supply of energy stays bounded. That would make **efficiency gains over a lifetime** the measurable form of "improves upon itself." This is a proposal, not a decision.

### Proposed way to test generality (needs a common way to frame problems)

* **A problem interface.** "Any problem that can be framed" needs a standard frame: what the brain observes, what it may do, and what feedback it gets.
  Today's worlds only *give* examples. The next step is interactive problems (observe, act, get feedback), which also gives the brain real outputs.
* **A battery of problem families**, each measured the same way (accuracy, energy, uptime, speed of learning): classification (exists), predicting sequences,
  discovering a hidden rule, tasks where the answer needs combining two things it already knows.
* **Learn from it and do it later:** retention. Teach, then give a long unrelated stretch, then retest.
* **Connect to what it knows and recreate it:** transfer. A new world with the *same hidden structure* as one already learned (e.g. rotated or rescaled): does it learn faster than a fresh brain?
* **Improves upon itself:** across a sequence of worlds, do accuracy and **energy per correct answer** keep improving? Does it ever plateau?
* **Pushes at boundaries:** when it meets something it cannot explain, does it spend energy *selectively* on that frontier (instead of everywhere)?

## Mastery: "understand it so well that doing it again costs nothing" (first version built; the rest is proposed)

**Built, 2026-10-18 (energy brain setting `automatic_after`):** a neuron that has answered many times and is nearly always right becomes *automatic*; answering with it costs only a tiny floor share
(`automatic_floor`, default 2%, because nothing is entirely free) of the normal energy, while anything new still costs full price. The habit is derived from the neuron's own record (how often it answered x
how reliably it was right), so it can also be *lost* if the world changes and the neuron starts being wrong. First measurement (3 runs each, standard world): energy per 1000 experiences fell from about 30
without automaticity to about 10 with it (about 8 when habits form fast), at the same accuracy. A new measure, **energy saved by experience** (how much cheaper the last third of life was than the first), puts the
learning curve in one number; even without automaticity it is about 40%, because early experiences are expensive (new things are flooded and grown).

**What this does NOT yet do, and the user's expectation that it should:**

* **Connections across problems.** The user expects every new problem to build connections with *every* previous problem so it is solved faster and cheaper. Today each simulated life is **one problem**
  (one world); nothing carries over to a second problem. The next architectural step is a **lifelong run**: one brain, a *sequence* of problems, measuring energy and speed per problem and against a fresh brain.
* **Understanding versus memorizing.** "Zero energy to do it again" is easy to get by memorizing exact answers. Understanding means a *rule* compact enough to cover variants. Test: after mastering a problem,
  give a variant (rotated, rescaled, partly hidden) and see whether it is still nearly free, with **fewer** stored neurons than memorizing would need.
* **Small long-term storage.** Storage is allowed to be unlimited but should end up small. Today neurons cost upkeep; a *consolidation* step that replaces many specific neurons with one general one, and frees the
  rest, is the mechanism to build and measure (neurons stored per problem should *fall* as problems accumulate).
* **Building its own architecture.** The brain currently has a fixed design with a few self-tuned dials. Self-built architecture means it decides *what structures to add* (new relations, shortcuts, hierarchies)
  when something is not understood, and pays for them. This is where the Self (the user's main focus) acts.
* **Out-of-scope problems** (the user's 4-dimensional example): needs the general problem frame from the earlier section, so that a problem can be plugged in without rebuilding the brain.

## A first general solver: the algorithm (discussed 2026-10-18; proposed, not yet agreed or built)

The user's framing: give it *any* problem (their example: predict the shape and points of a graph from a function, the kind of thing a neural network is applied to) and have it generate the best
solution it can, **in the quickest time, given its boundaries and tools**, then **store the way it thought of the solution** so it is farther along on every future problem.

**Loop**

1. **Problem frame:** what it can observe, what it may do, the feedback, and a budget (energy, time). Plus its *tools* (a small set of primitives, for example lines, curves, waves, a tiny neural net).
2. **Feel the situation (cheap probes):** smooth or jumpy, repeating, noisy, monotonic. A few numbers. This is a first emotional-converter-like signal and it steers the rest.
3. **Recall:** compare the feelings with past problems; try stored solutions first; verify on a few held-back points. If it fits, done at the cost of verifying (the "free once known" case).
4. **If surprised: push at the boundary:** search by *composing known pieces*, guided by the feelings, within the budget, keeping the best answer so far (anytime: good fast, better with more budget).
5. **Store how it thought:** (a) the solution, for exact repeats; (b) the recipe, for similar problems; (c) the rule from feelings to recipes (its intuition). Merge overlapping recipes, drop redundant ones, keep storage small.
6. **Meter every step in energy.**

**Test:** a sequence of related problems (for example 30 functions from a few families). Energy and time per problem should fall; later problems should be solved from stored recipes; a fresh solver per problem is the baseline.
Also test *understanding vs memorizing* with variants (shifted, scaled, noisier) and watch whether the library stays small.

**Honest limits:** no solver is best at literally every problem (no free lunch), so "best given tools and budget" must be stated; reuse only pays when new problems resemble old ones, and the stored feelings-to-recipes rule is how it should generalize, which is what the test checks;
related research exists (meta-learning, program synthesis, library learning), which is useful for knowing what works.

**Open choices:** which problem family first (1-D function graphs is the cheapest), which primitives count as tools, how energy is charged for search steps, and how this connects to the neuron-map brain
(the stored recipes could *be* the neurons, or the solver could sit beside the map as the Self's toolbox).

## What research says about the user's beliefs (checked 2026-10-18; sources and caveats in the wiki page `what-works-learning-expertise-motivation-2026-10-18`)

**Supports:** expertise is chunked long-term knowledge that makes known things nearly automatic and frees working memory (matches the automatic neurons); "learning to learn" is a real effect (Harlow's learning sets, reproduced by a meta-reinforcement-learning network);
curiosity is best described as **learning progress** (Oudeyer, Schmidhuber), which is the project's "reward only while uncertainty is dropping"; practice testing and spaced practice are the best-supported study techniques.

**Challenges:** "good at X makes you better at everything" (far transfer) is **weakly supported** in careful studies; it appears when the new problem shares structure with the old one, so *relatedness is in the structure, and the learner has to find it*. Deliberate practice explains only part of performance differences
(about 26% in games, 21% music, 18% sports, 4% education, under 1% professions, definitions contested). Growth-mindset interventions show weak effects. "Genius = motivation + processing speed" has **not been checked**.

**What it suggests for the machine (proposed):** test "all problems relate" by building problem families with *controllable shared structure* and measuring transfer against the overlap; make **motivation = learning progress per unit energy** (the Self decides what to work on); store strategies explicitly;
consolidate many neurons into templates and replay stored recipes at spaced intervals.

## Foundational blocks: how to test the belief (proposed 2026-10-18, not yet agreed or built)

**What exists that is relevant (checked via search summaries):** DreamCoder (Ellis et al., 2021) alternates solving problems with a *sleep* phase that refactors the solved programs to find common sub-components and adds them to a growing library, and reports solving more problems, faster
([paper](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf)); I did not confirm from the abstract how libraries carry over between *different* domains. Lin, Tegmark & Rolnick argue deep learning works because the functions we care about have physics-like structure (symmetry, locality, compositionality, hierarchy), so they need far fewer parameters than a generic function would
([paper](https://arxiv.org/abs/1608.08225v4)). Both are consistent with the user's belief that real problems are built from reusable parts; neither proves that *everything* is.

**The loop (proposed):** *wake:* solve the problems in front of it; *sleep:* look across everything solved so far for parts that keep recurring, turn each into a named concept (a block), rewrite old solutions in terms of the blocks, and drop what became redundant; *repeat forever.*
Concepts are first-class: they are what it stores and what it improves.

**What "getting better at producing information" means, measurably:** (1) *learning speed on unseen problems* (energy and examples needed to reach a given accuracy, against a fresh brain); (2) *information produced*: new correct facts it can derive or generate from what it knows, per unit of energy; (3) *library economy*: description length of the concepts plus solutions should shrink as it learns.

**A controlled test, the synthetic universe:** build a small universe with a few hidden foundational blocks (for example shift, scale, repeat, mirror, accumulate, add an offset). Generate many *domains* (families of sequences, shapes or functions), each a composition of two to four blocks plus noise. Some domains share blocks; some share none (the control).
Show the machine domains one after another and let it learn from examples, under an energy budget. Because the true blocks are known, we can score: did its stored concepts recover them; does the cost of a new domain fall with the number of shared blocks; does it stay flat when nothing is shared (the research's far-transfer result); does the library stay small?
**Honest limit:** this tests whether the machine can *find* blocks that exist; we put them there, so it says nothing yet about whether the real world has such blocks.

## Purpose: the learning criterion (the user's next big item; everything below is proposed, nothing is agreed)

The user's idea: defining a brain is step one; **step two is a purpose**, a criterion the brain learns by, which acts as its meaning of life.

Candidates, with where each comes from and how it could be tested in this simulator:

| Candidate criterion | Idea | What it would make the brain do | Test |
|---|---|---|---|
| **Minimize surprise** | Keep predictions right (predictive processing / free-energy idea) | Learn structure; sit in predictable situations | Does surprise fall over time? |
| **Stay alive** | Keep energy up and avoid forced shutdown (homeostasis) | Economize, rest, avoid overload | Uptime, shutdowns |
| **Less surprise per unit of energy** (proposed default) | Combine the two: understand the world *cheaply* | Grow only when it pays, tidy when worth it | Accuracy per energy, uptime |
| **Curiosity / learning progress** | Reward surprise *dropping*, not surprise itself (this is the earlier "reward effort only while uncertainty is dropping" principle) | Seek out things it can learn, ignore noise it cannot | Does it explore, then settle? |
| **Empowerment** | Prefer situations where it has more options and control | Needs real actions on the world | Only after outputs exist |

Two gaps to be honest about before any of these can matter:

* **Outputs.** Today the only "output" is a predicted label, and nothing the brain does changes what it senses. A purpose like "reduce uncertainty in its environment"
  becomes interesting when the brain can **act to change what it senses** (choose what to attend to, ask for a label, move toward or away from something).
* **Who holds the purpose?** The natural holder is the **Self**: it reads the body (energy, fatigue) and the emotional signals and picks actions in service of the criterion.

### Stepping stones towards consciousness (proposed ladder; functional steps only)

Each rung is something we can build and measure. Whether the top of the ladder amounts to consciousness is not something a simulation can settle, and this project should not claim it.

1. Sense and predict, with surprise measured (exists, in part).
2. A body that costs energy and can be forced offline (exists as of this session).
3. Valenced signals about its own state: alarm, relief, strain, curiosity (the emotional converter v0, not built).
4. A Self that makes the choices in one place, and pays for being a Self (not built).
5. A **self-model**: the Self predicts its *own* next state and next error (the earlier principle "predict own next error"), so it can notice when it is the one who is wrong.
6. Attention and report: it can say what it is currently modelling and why.
7. Counterfactuals: it can run "what if" before acting (the radius search is a tiny early version).

## Open questions for the user

1. Which parts besides emotional converter, intellectual converter and Self do you picture?
2. Should emotions be *signals the Self reads* (my proposal), or should they also change how the other parts behave directly (e.g. fear speeds thinking, costs more energy)?
3. Is the Self meant to have its own goals beyond "lower uncertainty cheaply" (the earlier note said a goal bigger than self-preservation)?
4. How much of the human-brain analogy should be copied versus used only as inspiration?
5. Which purpose feels closest to what you mean by a "meaning of life" for the brain: staying alive, understanding the world, understanding it cheaply, curiosity, or something else?
6. Should the brain eventually get real outputs (acts that change what it senses)? Without them the purpose can only be about prediction.
7. Are the "stepping stones" above the right kind of ladder, or is there a rung you would add or remove?
8. Do you want a standard "problem frame" (observe, act, feedback) built next, so that any problem can be plugged into the Lab? It is the prerequisite for testing generality.
9. For "infinitely growing" under finite energy: is the right reading that growth must pay for itself through rising efficiency (compression)? Or do you imagine the energy supply itself growing?
10. For "connections with every previous problem": should the first lifelong test be a *sequence of related worlds* (same hidden structure, different surface), so transfer can be measured cleanly?
11. For "zero energy to do it again": is an almost-free floor (2% here) acceptable, or must a mastered problem truly cost zero (and the floor then lives elsewhere, e.g. in basal metabolism)?
12. Which should come first: the **problem frame plus a sequence of problems with controllable shared structure** (tests "all problems relate" and transfer), or the **motivation controller** (learning progress per energy)? I suggest the first, since the second needs problems to choose between.
13. Should the next build be the **synthetic universe** (known hidden blocks, domains built from them, a wake-sleep concept-library learner, energy-metered)? It tests "everything is connected to foundational blocks" and "learn to learn" together, and it is cheap enough for the 1-thread limit.
